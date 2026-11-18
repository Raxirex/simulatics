"""Velocity-command tracking task for the humanoid: the recipe of Rudin et al. 2021.

"Learning to Walk in Minutes Using Massively Parallel Deep RL" (Rudin, Hoeller, Reist, Hutter; CoRL 2021,
arXiv 2109.11978) defines the task most learned locomotion controllers in the literature are trained on:

* **Command** c = (vx, vy, wz) in the robot's heading frame, resampled every few seconds, with a share of
  zero commands so the policy also learns to stand still (20 % is a common choice).
* **Observation** (proprioceptive only, deployable on hardware): base angular velocity, projected gravity,
  command, joint positions minus the default pose, joint velocities, last action, and the sin/cos of a
  gait clock (common for humanoids). Optionally a history of past frames.
* **Privileged critic observation** (asymmetric actor-critic, Pinto et al. 2018): adds the true base
  linear velocity, base height, foot contacts and the domain-randomisation parameters of the episode.
* **Action**: joint-target offsets, q_target = q_default + action_scale * a, for the leg and abdomen
  actuators. Arms are held at their default pose. MuJoCo position servos are the PD loop.
* **Reward**: exponential tracking kernels exp(-|c - v|^2 / sigma) for linear xy velocity and yaw rate,
  and small penalties (z velocity, roll/pitch rate, orientation, base height, torques, joint
  acceleration, action rate, joint limits, hip roll/yaw), the feet-air-time bonus, a
  contact-phase term, a stand-still term for zero commands and a termination penalty. As in Rudin et al. 2021,
  every weight is multiplied by the control dt, and the sum (without termination) can be clipped at zero
  (`only_positive_rewards`) so an early policy is not taught to fall on purpose.
* **Domain randomisation**: floor/foot friction scale, added torso mass and servo stiffness are drawn
  per model variant (MuJoCo's threaded rollout accepts one model per environment), plus random pushes.

All maths is batched over a leading environment axis so the same code drives the vectorised trainer
(`sim/vec_env.py`) and a single robot in the live session (`RLController` in `train/ppo.py`). Robot
dimensions (stand height, joint ranges, gains, torque limits) are read from the compiled MjModel.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict

import mujoco
import numpy as np

STATE_SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS

# actuators whose joint name contains one of these are arms / hands / head: held at their default target
ARM_KEYWORDS = ("shoulder", "elbow", "wrist", "arm", "hand", "gripper", "finger", "thumb", "palm", "neck", "head")

# Reference weights from Rudin et al. 2021 where they exist, common humanoid values
# for the humanoid-specific terms (contact phase 0.18, alive 0.15, hip_pos), the rest are starting points.
DEFAULT_REWARDS: dict[str, float] = {
    "tracking_lin_vel": 1.0,
    "tracking_ang_vel": 0.5,
    "lin_vel_z": -2.0,
    "ang_vel_xy": -0.05,
    "orientation": -1.0,
    "base_height": -10.0,
    "torques": -1e-5,
    "dof_acc": -2.5e-7,
    "action_rate": -0.01,
    "dof_pos_limits": -5.0,
    "feet_air_time": 1.0,
    "contact_phase": 0.18,
    "hip_pos": -0.5,
    "stand_still": -0.5,
    "alive": 0.15,
    "termination": -100.0,
}


@dataclass
class VelTaskConfig:
    # timing
    decimation: int = 4                 # physics steps per policy step (0.005 s * 4 = 50 Hz policy)
    episode_seconds: float = 20.0
    # action
    action_scale: float = 0.25          # rad per unit action (common reference value: 0.25)
    clip_actions: float = 5.0           # actions are clipped to +-this before scaling
    knee_bend: float = 0.0              # default pose knee flexion (rad); hips/ankles keep the foot flat
    # Forward lean of the default pose (rad, about the ankles). The built-in humanoid's ankle sits at the
    # back of its foot, so standing straight puts the CoM ~4 cm behind the foot centre and small pushes
    # tip it backwards. Measured with 64 envs over 5 s: lean 0 -> 30 % fall with zero action and 94 %
    # under N(0, 0.1) action noise; lean 0.02 -> 0 % and 2 %. (A bent-knee default without a balance
    # controller was worse: it sinks backwards under fixed servo targets.)
    lean: float = 0.025
    # commands (robot heading frame)
    cmd_vx: tuple[float, float] = (-0.4, 0.8)
    cmd_vy: tuple[float, float] = (-0.3, 0.3)
    cmd_wz: tuple[float, float] = (-0.8, 0.8)
    resample_seconds: float = 8.0
    zero_cmd_prob: float = 0.2
    small_cmd: float = 0.1              # |v_xy| below this -> 0 (Rudin et al. 2021)
    # observation
    gait_period: float = 0.8            # s; 0 disables the clock input and the contact-phase reward
    history: int = 0                    # number of past proprioceptive frames appended to the actor obs
    obs_noise: float = 1.0              # multiplier on the reference noise scales (0 = no noise)
    asymmetric_critic: bool = True
    # reward
    rewards: dict = field(default_factory=lambda: dict(DEFAULT_REWARDS))
    tracking_sigma: float = 0.25
    only_positive_rewards: bool = True
    base_height_frac: float = 0.97      # target root height as a fraction of the default standing height
    feet_air_time_target: float = 0.4   # s
    contact_force: float = 1.0          # N on the foot touch sensor counts as contact
    soft_joint_limit: float = 0.9       # fraction of the joint range before the limit penalty starts
    # termination
    min_height_frac: float = 0.6
    max_tilt_deg: float = 60.0
    # initial state
    init_joint_noise: float = 0.05      # rad (uniform)
    init_vel_noise: float = 0.1
    random_yaw: bool = True
    # domain randomisation
    randomize: bool = True
    n_model_variants: int = 8           # variant 0 is always the nominal model
    friction_range: tuple[float, float] = (0.6, 1.25)   # scale on every geom's sliding friction
    added_mass_range: tuple[float, float] = (-2.0, 3.0)  # kg on the root body
    kp_scale_range: tuple[float, float] = (0.9, 1.1)     # scale on servo stiffness
    push_interval_s: float = 8.0        # 0 disables pushes
    push_vel: float = 0.4               # max added base xy velocity (m/s)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VelTaskConfig":
        c = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        for k in ("cmd_vx", "cmd_vy", "cmd_wz", "friction_range", "added_mass_range", "kp_scale_range"):
            setattr(c, k, tuple(float(x) for x in getattr(c, k)))
        c.rewards = {**DEFAULT_REWARDS, **{k: float(v) for k, v in (c.rewards or {}).items()}}
        return c


@dataclass
class VelTaskState:
    """Per-environment buffers that are not part of MuJoCo's state."""
    commands: np.ndarray        # (N, 3) vx, vy, wz
    last_action: np.ndarray     # (N, A)
    last_dqd: np.ndarray        # (N, A) joint velocities at the previous step (for joint acceleration)
    feet_air_time: np.ndarray   # (N, 2)
    last_contact: np.ndarray    # (N, 2) bool
    t: np.ndarray               # (N,) seconds since reset
    cmd_timer: np.ndarray       # (N,) seconds until the next command resample
    push_timer: np.ndarray      # (N,)
    variant: np.ndarray         # (N,) int, index of the model variant (domain randomisation)
    history: np.ndarray         # (N, H, P) past proprioceptive frames (H may be 0)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz quaternions -> (N, 3, 3) rotation matrices (body -> world)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1),
    ], axis=1)


def _quat_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternion batches (N, 4)."""
    a1, b1, c1, d1 = p.T
    a2, b2, c2, d2 = q.T
    return np.stack([a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2, a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                     a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2, a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2], axis=1)


def _quat_yaw_mul(yaw: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate wxyz quaternions q (N, 4) by a world yaw: q_yaw * q."""
    a1, d1 = np.cos(yaw / 2), np.sin(yaw / 2)
    a2, b2, c2, d2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([a1 * a2 - d1 * d2, a1 * b2 - d1 * c2, a1 * c2 + d1 * b2, a1 * d2 + d1 * a2], axis=1)


def _lowest_point(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Lowest z of any geom of the robot (the tree under the free root joint); boxes, capsules and
    spheres exactly, other shapes by their bounding sphere."""
    zmin = np.inf
    root = int(model.jnt_bodyid[0])
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b == 0 or model.body_rootid[b] != model.body_rootid[root]:
            continue
        t, size, p, r = model.geom_type[g], model.geom_size[g], data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3)
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            z = p[2] - np.abs(r[2] * size).sum()
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            z = p[2] - abs(r[2, 2]) * size[1] - size[0]
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            z = p[2] - size[0]
        else:
            z = p[2] - model.geom_rbound[g]
        zmin = min(zmin, float(z))
    return zmin


class VelocityTask:
    """Batched observation / action / reward maths for velocity-command tracking."""

    def __init__(self, model: mujoco.MjModel, cfg: VelTaskConfig | None = None):
        if not (model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError("velocity task needs a model whose first joint is a free joint")
        self.cfg = cfg = cfg or VelTaskConfig()
        self.nq, self.nv, self.nu = model.nq, model.nv, model.nu
        self.nstate = mujoco.mj_stateSize(model, STATE_SPEC)
        self.dt = float(model.opt.timestep) * cfg.decimation
        self.episode_steps = max(1, int(round(cfg.episode_seconds / self.dt)))
        self.weights = {k: float(v) for k, v in {**DEFAULT_REWARDS, **cfg.rewards}.items() if v != 0.0}

        # ---- actuators: controlled (legs + abdomen) vs held (arms) -------------------------------
        names, jids = [], []
        for i in range(model.nu):
            if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError("velocity task supports joint-transmission actuators only")
            jid = int(model.actuator_trnid[i, 0])
            names.append(model.actuator(i).name or model.joint(jid).name)
            jids.append(jid)
        jnames = [model.joint(j).name for j in jids]
        is_arm = np.array([any(k in n.lower() for k in ARM_KEYWORDS) for n in jnames])
        self.actuator_names = names
        self.ctrl_idx = np.flatnonzero(~is_arm)
        self.arm_idx = np.flatnonzero(is_arm)
        self.joint_names = [jnames[i] for i in self.ctrl_idx]
        self.act_dim = len(self.ctrl_idx)
        jids = np.array(jids)
        self.qadr_all = model.jnt_qposadr[jids]
        self.qadr = self.qadr_all[self.ctrl_idx]
        self.dadr = model.jnt_dofadr[jids][self.ctrl_idx]

        # joint ranges (radians in the compiled model; actuator ctrlrange may be in other units)
        lim = model.jnt_limited[jids].astype(bool)
        lo = np.where(lim, model.jnt_range[jids, 0], -np.inf)
        hi = np.where(lim, model.jnt_range[jids, 1], np.inf)
        self.ctrl_lo, self.ctrl_hi = lo, hi
        self.q_lo, self.q_hi = lo[self.ctrl_idx], hi[self.ctrl_idx]
        mid, half = (self.q_lo + self.q_hi) / 2, (self.q_hi - self.q_lo) / 2
        self.soft_lo = np.where(np.isfinite(half), mid - cfg.soft_joint_limit * half, -np.inf)
        self.soft_hi = np.where(np.isfinite(half), mid + cfg.soft_joint_limit * half, np.inf)

        # servo gains (position actuators: gain = kp, bias = [0, -kp, -kv]) and torque limits
        self.kp = model.actuator_gainprm[self.ctrl_idx, 0].copy()
        self.kv = -model.actuator_biasprm[self.ctrl_idx, 2].copy()
        frc = np.where(model.actuator_forcelimited[self.ctrl_idx].astype(bool),
                       model.actuator_forcerange[self.ctrl_idx, 1], np.inf)
        self.tau_max = frc
        self.hip_cols = np.array([k for k, n in enumerate(self.joint_names)
                                  if "hip_x" in n or "hip_z" in n or "hip_roll" in n or "hip_yaw" in n], int)

        # ---- default pose ---------------------------------------------------------------------
        default = np.zeros(model.nu)
        bend = float(cfg.knee_bend)
        for i, n in enumerate(jnames):
            if bend and n.endswith("knee"):
                default[i] = bend
            elif bend and n.endswith("hip_y"):
                default[i] = -bend / 2
            elif bend and n.endswith("ankle_y"):
                default[i] = -bend / 2
        d = mujoco.MjData(model)
        if model.nkey > 0:  # an imported robot's first keyframe defines its standing pose
            mujoco.mj_resetDataKeyframe(model, d, 0)
            if not bend:
                default = d.qpos[self.qadr_all].copy()
        else:
            mujoco.mj_resetData(model, d)
        # forward lean: the whole body pitches forward about the ankles (ankle pitch -lean, root +lean)
        ankle_pitch = np.array([n.endswith("ankle_y") or "ankle_pitch" in n for n in jnames])
        lean = float(cfg.lean) if ankle_pitch.any() else 0.0
        default[ankle_pitch] -= lean
        if lean:
            d.qpos[3:7] = _quat_mul(np.array([[np.cos(lean / 2), 0.0, np.sin(lean / 2), 0.0]]), d.qpos[None, 3:7])[0]
        default = np.clip(default, lo, hi)
        self.ctrl_default = default
        self.q_default = default[self.ctrl_idx]
        # the soft-limit band never penalises the default pose (a straight knee sits on its lower limit)
        self.soft_lo = np.minimum(self.soft_lo, self.q_default)
        self.soft_hi = np.maximum(self.soft_hi, self.q_default)
        d.qpos[self.qadr_all] = self.ctrl_default
        mujoco.mj_forward(model, d)
        d.qpos[2] += 0.002 - _lowest_point(model, d)  # put the feet on the floor
        mujoco.mj_forward(model, d)
        self.z0 = float(d.qpos[2])
        self.init_state = np.zeros(self.nstate)
        mujoco.mj_getState(model, d, self.init_state, STATE_SPEC)
        self.init_qpos = d.qpos.copy()

        # foot touch sensors (left first), used for air time / contact phase / the critic
        touch = [(model.sensor(i).name, int(model.sensor_adr[i])) for i in range(model.nsensor)
                 if model.sensor_type[i] == mujoco.mjtSensor.mjSENS_TOUCH and "foot" in model.sensor(i).name]
        touch.sort(key=lambda t: (0 if "left" in t[0] else 1, t[0]))
        self.touch_adr = np.array([a for _, a in touch[:2]], int)
        self.root_body = int(model.jnt_bodyid[0])

        # ---- observation layout --------------------------------------------------------------
        a = self.act_dim
        self.clock = cfg.gait_period > 0
        self.proprio_dim = 3 + 3 + 3 + 3 * a + (2 if self.clock else 0)
        self.obs_dim = self.proprio_dim * (1 + cfg.history)
        self.critic_obs_dim = (self.proprio_dim + 3 + 1 + 2 + 3) if cfg.asymmetric_critic else self.obs_dim
        # Reference scales: lin_vel 2.0, ang_vel 0.25, dof_pos 1.0, dof_vel 0.05; commands scaled alike
        self.cmd_scale = np.array([2.0, 2.0, 0.25])
        noise = np.concatenate([np.full(3, 0.2 * 0.25), np.full(3, 0.05), np.zeros(3), np.full(a, 0.01),
                                np.full(a, 1.5 * 0.05), np.zeros(a), np.zeros(2 if self.clock else 0)])
        self.noise_scale = noise * cfg.obs_noise

    # ---- buffers ------------------------------------------------------------------------------------
    def make_state(self, n: int) -> VelTaskState:
        a = self.act_dim
        return VelTaskState(
            commands=np.zeros((n, 3)), last_action=np.zeros((n, a)), last_dqd=np.zeros((n, a)),
            feet_air_time=np.zeros((n, 2)), last_contact=np.zeros((n, 2), bool), t=np.zeros(n),
            cmd_timer=np.zeros(n), push_timer=np.zeros(n), variant=np.zeros(n, int),
            history=np.zeros((n, self.cfg.history, self.proprio_dim)))

    def reset_envs(self, states: np.ndarray, ts: VelTaskState, idx: np.ndarray, rng: np.random.Generator,
                   n_variants: int = 1) -> None:
        """Write fresh initial MuJoCo states into states[idx] and clear the task buffers of those envs."""
        idx = np.asarray(idx, int)
        if idx.size == 0:
            return
        cfg, n = self.cfg, idx.size
        s = np.tile(self.init_state, (n, 1))
        qpos = s[:, 1:1 + self.nq]
        qvel = s[:, 1 + self.nq:1 + self.nq + self.nv]
        qpos[:, self.qadr] += rng.uniform(-cfg.init_joint_noise, cfg.init_joint_noise, (n, self.act_dim))
        qpos[:, self.qadr] = np.clip(qpos[:, self.qadr], self.q_lo, self.q_hi)
        if cfg.random_yaw:
            qpos[:, 3:7] = _quat_yaw_mul(rng.uniform(-np.pi, np.pi, n), qpos[:, 3:7])
        qvel[:, 0:2] += rng.uniform(-cfg.init_vel_noise, cfg.init_vel_noise, (n, 2))
        qvel[:, 3:6] += rng.uniform(-cfg.init_vel_noise, cfg.init_vel_noise, (n, 3))
        states[idx] = s
        for buf in (ts.last_action, ts.last_dqd, ts.feet_air_time, ts.t, ts.history):
            buf[idx] = 0
        ts.last_contact[idx] = False
        ts.variant[idx] = rng.integers(0, max(1, n_variants), n)
        ts.push_timer[idx] = rng.uniform(0.5, 1.0, n) * cfg.push_interval_s
        self.sample_commands(ts, idx, rng)

    def sample_commands(self, ts: VelTaskState, idx: np.ndarray, rng: np.random.Generator) -> None:
        cfg, n = self.cfg, len(idx)
        c = np.stack([rng.uniform(*cfg.cmd_vx, n), rng.uniform(*cfg.cmd_vy, n), rng.uniform(*cfg.cmd_wz, n)], axis=1)
        c[np.linalg.norm(c[:, :2], axis=1) < cfg.small_cmd, :2] = 0.0
        c[rng.random(n) < cfg.zero_cmd_prob] = 0.0
        ts.commands[idx] = c
        ts.cmd_timer[idx] = cfg.resample_seconds

    # ---- action ---------------------------------------------------------------------------------------
    def action_to_ctrl(self, actions: np.ndarray) -> np.ndarray:
        """(N, A) policy actions -> (N, nu) servo targets; arms get their default target."""
        a = np.clip(actions, -self.cfg.clip_actions, self.cfg.clip_actions)
        ctrl = np.tile(self.ctrl_default, (len(a), 1))
        ctrl[:, self.ctrl_idx] = self.q_default + self.cfg.action_scale * a
        return np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)

    def ctrl_to_action(self, ctrl: np.ndarray) -> np.ndarray:
        """Inverse of action_to_ctrl for the controlled joints (used to label teacher demonstrations)."""
        a = (np.atleast_2d(ctrl)[:, self.ctrl_idx] - self.q_default) / self.cfg.action_scale
        return np.clip(a, -self.cfg.clip_actions, self.cfg.clip_actions)

    # ---- kinematics helpers -------------------------------------------------------------------------
    def split(self, states: np.ndarray):
        return states[:, 1:1 + self.nq], states[:, 1 + self.nq:1 + self.nq + self.nv]

    @staticmethod
    def base_frame(qpos: np.ndarray, qvel: np.ndarray):
        """-> rotation (N,3,3), body-frame linear velocity, body-frame angular velocity, projected gravity."""
        rot = quat_to_mat(qpos[:, 3:7])
        lin = np.einsum("nji,nj->ni", rot, qvel[:, 0:3])  # R^T v
        ang = qvel[:, 3:6]                                   # free-joint angular velocity is body-frame
        grav = -rot[:, 2, :]                                 # R^T (0, 0, -1)
        return rot, lin, ang, grav

    def phase(self, ts: VelTaskState) -> np.ndarray:
        return (ts.t / self.cfg.gait_period) % 1.0 if self.clock else np.zeros_like(ts.t)

    # ---- observation ----------------------------------------------------------------------------------
    def observe(self, qpos: np.ndarray, qvel: np.ndarray, ts: VelTaskState, contact: np.ndarray | None = None,
                props: np.ndarray | None = None, rng: np.random.Generator | None = None,
                update_history: bool = True):
        """-> (actor obs (N, obs_dim), critic obs (N, critic_obs_dim)).

        Observation noise is applied to the actor observation only, when `rng` is given.
        With history > 0 the actor sees [frame_t, frame_{t-1}, ..., frame_{t-H}] and, when
        update_history is set, frame_t is pushed into the buffer (the only side effect).
        """
        _, lin, ang, grav = self.base_frame(qpos, qvel)
        dq = qpos[:, self.qadr] - self.q_default
        dqd = qvel[:, self.dadr]
        parts = [ang * 0.25, grav, ts.commands * self.cmd_scale, dq, dqd * 0.05, ts.last_action]
        if self.clock:
            ph = 2 * np.pi * self.phase(ts)
            parts.append(np.stack([np.sin(ph), np.cos(ph)], axis=1))
        clean = np.concatenate(parts, axis=1)
        clean = np.where(np.isfinite(clean), clean, 0.0)
        frame = clean
        if rng is not None and self.cfg.obs_noise > 0:
            frame = clean + self.noise_scale * rng.uniform(-1.0, 1.0, clean.shape)
        if self.cfg.history > 0:
            obs = np.concatenate([frame, ts.history.reshape(len(frame), -1)], axis=1)
            if update_history:
                ts.history[:, 1:] = ts.history[:, :-1]
                ts.history[:, 0] = frame
        else:
            obs = frame
        if not self.cfg.asymmetric_critic:
            return obs, obs
        n = len(clean)
        c = np.zeros((n, 2)) if contact is None else contact.astype(np.float64)
        p = np.zeros((n, 3)) if props is None else props
        priv = np.concatenate([clean, lin * 2.0, (qpos[:, 2:3] / self.z0 - 1.0) * 10.0, c, p], axis=1)
        return obs, np.where(np.isfinite(priv), priv, 0.0)

    # ---- reward / termination -------------------------------------------------------------------------
    def contacts_from_sensordata(self, sensordata: np.ndarray) -> np.ndarray:
        """(N, nsub, nsensordata) touch readings over the substeps -> (N, 2) bool foot contact."""
        if self.touch_adr.size < 2:
            return np.zeros((sensordata.shape[0], 2), bool)
        return sensordata[:, :, self.touch_adr].max(axis=1) > self.cfg.contact_force

    def reward(self, qpos: np.ndarray, qvel: np.ndarray, actions: np.ndarray, ctrl: np.ndarray,
               contact: np.ndarray, ts: VelTaskState, kp_scale: np.ndarray | None = None):
        """Reward for the step that ended in (qpos, qvel); updates the per-env buffers.

        Returns (reward (N,), {term: weighted reward (N,)}, terminated (N,) bool).
        """
        cfg, w, dt = self.cfg, self.weights, self.dt
        _, lin, ang, grav = self.base_frame(qpos, qvel)
        q = qpos[:, self.qadr]
        dq = q - self.q_default
        dqd = qvel[:, self.dadr]
        cmd = ts.commands
        moving = np.linalg.norm(cmd[:, :2], axis=1) > cfg.small_cmd
        still = ~moving & (np.abs(cmd[:, 2]) < cfg.small_cmd)
        z = qpos[:, 2]
        terminated = ((z < cfg.min_height_frac * self.z0) | (-grav[:, 2] < np.cos(np.deg2rad(cfg.max_tilt_deg)))
                      | ~np.isfinite(qpos).all(axis=1) | ~np.isfinite(qvel).all(axis=1))

        raw: dict[str, np.ndarray] = {}

        def term(name, fn):
            if name in w:
                raw[name] = fn()

        sig = cfg.tracking_sigma
        term("tracking_lin_vel", lambda: np.exp(-np.sum((cmd[:, :2] - lin[:, :2]) ** 2, axis=1) / sig))
        term("tracking_ang_vel", lambda: np.exp(-(cmd[:, 2] - ang[:, 2]) ** 2 / sig))
        term("lin_vel_z", lambda: lin[:, 2] ** 2)
        term("ang_vel_xy", lambda: np.sum(ang[:, :2] ** 2, axis=1))
        term("orientation", lambda: np.sum(grav[:, :2] ** 2, axis=1))
        term("base_height", lambda: (z - cfg.base_height_frac * self.z0) ** 2)

        def torques():
            kp = self.kp if kp_scale is None else self.kp * kp_scale[:, None]
            tau = kp * (ctrl[:, self.ctrl_idx] - q) - self.kv * dqd
            return np.sum(np.clip(tau, -self.tau_max, self.tau_max) ** 2, axis=1)

        term("torques", torques)
        term("dof_acc", lambda: np.sum(((dqd - ts.last_dqd) / dt) ** 2, axis=1))
        term("action_rate", lambda: np.sum((actions - ts.last_action) ** 2, axis=1))
        term("dof_pos_limits", lambda: np.sum(np.clip(self.soft_lo - q, 0, None) + np.clip(q - self.soft_hi, 0, None),
                                              axis=1))
        # feet air time (Rudin et al. 2021): reward each touchdown by (air time - target), only when commanded to move
        contact_filt = contact | ts.last_contact
        first_contact = (ts.feet_air_time > 0) & contact_filt
        air = ts.feet_air_time + dt
        term("feet_air_time", lambda: np.sum((air - cfg.feet_air_time_target) * first_contact, axis=1) * moving)
        if self.clock:
            ph = self.phase(ts)
            stance = np.stack([ph < 0.55, ((ph + 0.5) % 1.0) < 0.55], axis=1)
            term("contact_phase", lambda: np.where(moving, np.sum(contact == stance, axis=1),
                                                   np.sum(contact, axis=1)).astype(float))
        if self.hip_cols.size:
            term("hip_pos", lambda: np.sum(dq[:, self.hip_cols] ** 2, axis=1))
        term("stand_still", lambda: np.sum(np.abs(dq), axis=1) * still)
        term("alive", lambda: np.ones(len(z)))

        terms = {k: np.where(np.isfinite(v), v, 0.0) * w[k] * dt for k, v in raw.items()}
        total = np.zeros(len(z))
        for v in terms.values():
            total += v
        if cfg.only_positive_rewards:
            total = np.clip(total, 0.0, None)
        if "termination" in w:
            terms["termination"] = terminated.astype(float) * w["termination"] * dt
            total += terms["termination"]

        # advance buffers
        ts.feet_air_time[:] = air * ~contact_filt
        ts.last_contact[:] = contact
        ts.last_action[:] = actions
        ts.last_dqd[:] = np.where(np.isfinite(dqd), dqd, 0.0)
        ts.t += dt
        ts.cmd_timer -= dt
        ts.push_timer -= dt
        return total, terms, terminated


def make_variants(model: mujoco.MjModel, cfg: VelTaskConfig, rng: np.random.Generator,
                  task: VelocityTask | None = None):
    """Domain randomisation by model variants: copies of `model` with scaled friction, extra root mass and
    scaled servo stiffness. Returns (models, props (K, 3) = [friction scale, added kg, kp scale]).
    Variant 0 is the nominal model. MuJoCo's rollout steps each environment with its own variant."""
    k = max(1, int(cfg.n_model_variants)) if cfg.randomize else 1
    props = np.zeros((k, 3))
    props[:, 0], props[:, 2] = 1.0, 1.0
    models = [model]
    root = task.root_body if task is not None else int(model.jnt_bodyid[0])
    for i in range(1, k):
        m = copy.deepcopy(model)
        f = rng.uniform(*cfg.friction_range)
        dm = rng.uniform(*cfg.added_mass_range)
        ks = rng.uniform(*cfg.kp_scale_range)
        m.geom_friction[:, 0] *= f
        base = float(model.body_mass[root])
        new = max(0.2 * base, base + dm)
        m.body_mass[root] = new
        m.body_inertia[root] *= new / base
        servo = m.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_gainprm[servo, 0] *= ks
        m.actuator_biasprm[servo, 1] *= ks
        props[i] = (f, new - base, ks)
        models.append(m)
    return models, props


def normalized_props(props: np.ndarray) -> np.ndarray:
    """DR parameters centred around the nominal model for the critic: [friction-1, kg/5, kp-1]."""
    return np.stack([props[:, 0] - 1.0, props[:, 1] / 5.0, props[:, 2] - 1.0], axis=1)
