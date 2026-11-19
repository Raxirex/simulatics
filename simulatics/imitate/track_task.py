"""DeepMimic-style motion tracking task: batched observation, action, reward and termination maths.

Peng, Abbeel, Levine, van de Panne, "DeepMimic: Example-guided deep reinforcement learning of
physics-based character skills", SIGGRAPH 2018 (arXiv 1804.02717), with the 2023-26 humanoid tracking
practice (ASAP, BeyondMimic) where it helps a CPU budget:

* **Reference** (reference.py): the clip resampled to the policy rate (50 Hz), with model-FK key bodies
  and centre of mass and feed-forward servo targets u_ref (inverse dynamics of the reference).
* **Observation** (actor): proprioception - root height, gravity in the root frame, root linear and
  angular velocity (root frame), the controlled joints' angles and velocities, the previous action -
  plus the phase (sin / cos of 2 pi phi; progress for clips that do not loop), the current tracking
  errors a balance law acts on (root position, root velocity and centre-of-mass errors in the heading
  frame, root orientation error as a rotation vector, the reference's stance feet; cf. BeyondMimic's
  anchor-pose error) and **future reference frames** k + 1, k + 2, k + 5 (configurable): joint-angle
  differences to the reference, the reference root position and orientation (6-D: two columns of the
  rotation) in the robot's heading frame. The critic additionally sees the end-effector errors and the
  time left (asymmetric actor-critic, Pinto et al. 2018).
* **Action**: joint target offsets around the reference, ctrl = u_ref(k + 1) + scale * a, for the
  abdomen, leg and arm (shoulder, elbow) servos; wrists and grippers get their reference targets.
* **Reward** (Peng et al. 2018, Eq. 4-8): r = sum_i w_i exp(-s_i e_i) over
    pose     e = sum_j (q_j - q_ref_j)^2                         (controlled joints)
    vel      e = sum_j (qd_j - qd_ref_j)^2
    end_eff  e = sum_e |p_e - p_ref_e|^2   hands and feet, relative to the root in its heading frame
    com      e = |c - c_ref|^2                                    (world)
    root     e = |p - p_ref|^2 + 0.1 ang(R_ref^T R)^2 + 0.01 |v - v_ref|^2 + 0.001 |w - w_ref|^2
  with the paper's scales (2, 0.1, 40, 10) and 5 for the root term. The paper's weights are 0.65 / 0.1 /
  0.15 / 0.1 without a root term; with it we use 0.5 / 0.05 / 0.15 / 0.1 / 0.2. Weights sum to 1, so
  following the reference exactly gives r = 1 per step.
* **Reference state initialisation** (RSI): episodes start at a random frame, in the reference state
  (plus small joint / velocity noise). **Early termination**: root height off the reference by more than
  `term_height`, root tilt off by more than `term_tilt_deg`, or root position off by more than
  `term_distance` (BeyondMimic / ASAP use the same kind of deviation test). The end of a clip that does
  not loop is a time-out, not a termination.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from .motion import Motion, mat_to_rotvec, quat_to_mat
from .reference import MotionReference, clip_ctrl, servo_params

STATE_SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS
# actuators whose joint name contains one of these are driven by the reference only, not the policy
PASSIVE_KEYWORDS = ("wrist", "gripper", "finger", "thumb", "palm", "hand")
END_EFFECTORS = ("left_hand", "right_hand", "left_foot", "right_foot")


@dataclass
class TrackConfig:
    decimation: int = 4                 # physics steps per policy step (0.005 s x 4 = 50 Hz)
    episode_seconds: float = 10.0
    action_scale: float = 0.2           # rad per unit action, around the feed-forward target
    clip_actions: float = 5.0
    feedforward: str = "dynamic"        # "dynamic" | "static" | "none" (see reference.py)
    future: tuple = (1, 2, 5)           # reference frames ahead in the observation
    weights: dict = field(default_factory=lambda: {"pose": 0.5, "vel": 0.05, "end_eff": 0.15, "com": 0.1,
                                                   "root": 0.2})
    scales: dict = field(default_factory=lambda: {"pose": 2.0, "vel": 0.1, "end_eff": 40.0, "com": 10.0,
                                                  "root": 5.0})
    reward_scale: float = 0.02          # what PPO sees: r * reward_scale (= r * dt: returns ~1 at gamma 0.99)
    term_height: float = 0.25           # m, |root z - reference root z|
    term_tilt_deg: float = 50.0         # angle between the root's and the reference root's up axes
    term_distance: float = 0.5          # m, |root position - reference root position|
    rsi: bool = True
    init_joint_noise: float = 0.02      # rad (uniform), on the controlled joints at reset
    init_vel_noise: float = 0.05        # m/s and rad/s on the root, rad/s on the joints
    asymmetric_critic: bool = True
    push_interval_s: float = 0.0        # random pushes (0 = off)
    push_vel: float = 0.3

    def to_dict(self) -> dict:
        d = asdict(self)
        d["future"] = list(self.future)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrackConfig":
        c = cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})
        c.future = tuple(int(x) for x in c.future)
        c.weights = {**cls().weights, **{k: float(v) for k, v in (c.weights or {}).items()}}
        c.scales = {**cls().scales, **{k: float(v) for k, v in (c.scales or {}).items()}}
        return c


@dataclass
class TrackState:
    """Per-environment task buffers (not part of MuJoCo's state)."""
    frame: np.ndarray        # (N,) reference frame index of the current state
    origin: np.ndarray       # (N, 3) planar anchor of the reference (x, y, yaw); zeros in training
    last_action: np.ndarray  # (N, A)
    t: np.ndarray            # (N,) seconds since reset
    push_timer: np.ndarray   # (N,)


def _heading(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) -> yaw-only rotations (N, 3, 3) of the same heading."""
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    z, o = np.zeros_like(yaw), np.ones_like(yaw)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def _tvec(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """R^T v for batches: R (N, 3, 3), v (N, 3) or (N, K, 3)."""
    if v.ndim == 2:
        return np.einsum("nji,nj->ni", R, v)
    return np.einsum("nji,nkj->nki", R, v)


class TrackTask:
    """Batched tracking maths for one reference motion on one model (see the module docstring)."""

    def __init__(self, model: mujoco.MjModel, motion: Motion, cfg: TrackConfig | None = None):
        self.cfg = cfg = cfg or TrackConfig()
        self.model = model
        self.nq, self.nv, self.nu = model.nq, model.nv, model.nu
        self.nstate = mujoco.mj_stateSize(model, STATE_SPEC)
        self.dt = float(model.opt.timestep) * cfg.decimation
        self.ref = MotionReference(model, motion, self.dt, cfg.feedforward)
        self.motion = self.ref.motion
        self.qa, self.da = self.ref.qa, self.ref.da
        qadr, dadr, kp, kv, servo = servo_params(model)
        names = []
        for a in range(model.nu):
            names.append(model.joint(int(model.actuator_trnid[a, 0])).name
                         if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT else model.actuator(a).name)
        self.actuator_names = names
        ctrl_idx = [a for a in range(model.nu) if servo[a] and qadr[a] >= 0
                    and not any(k in names[a].lower() for k in PASSIVE_KEYWORDS)]
        if not ctrl_idx:
            raise ValueError("no position-servo actuators to control")
        self.ctrl_idx = np.array(ctrl_idx, int)
        self.joint_names = [names[a] for a in ctrl_idx]
        self.act_dim = len(ctrl_idx)
        self.qadr = qadr[self.ctrl_idx]
        self.dadr = dadr[self.ctrl_idx]
        self.q_lo = model.jnt_range[model.actuator_trnid[self.ctrl_idx, 0], 0]
        self.q_hi = model.jnt_range[model.actuator_trnid[self.ctrl_idx, 0], 1]
        self.ee = np.array([self.ref.key_names.index(k) for k in END_EFFECTORS if k in self.ref.key_names], int)
        self.ee_names = [self.ref.key_names[i] for i in self.ee]
        self.fk = self.ref.fk
        # template full-physics state (time 0, no actuator state)
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        self.init_state = np.zeros(self.nstate)
        mujoco.mj_getState(model, d, self.init_state, STATE_SPEC)
        # observation layout
        a = self.act_dim
        self.future = tuple(int(f) for f in cfg.future)
        self.proprio_dim = 1 + 3 + 3 + 3 + 3 * a + 3
        self.error_dim = 3 + 3 + 3 + 3 + 2
        self.obs_dim = self.proprio_dim + self.error_dim + len(self.future) * (a + 9)
        self.critic_obs_dim = self.obs_dim + (3 * len(self.ee) + 1 if cfg.asymmetric_critic else 0)
        self.max_steps = int(round(cfg.episode_seconds / self.dt))
        w = cfg.weights
        tot = sum(w.values())
        self.weights = {k: v / tot for k, v in w.items()} if tot > 0 else dict(w)

    # ---- buffers / state --------------------------------------------------------------------------------
    def make_state(self, n: int) -> TrackState:
        return TrackState(frame=np.zeros(n, np.int64), origin=np.zeros((n, 3)), last_action=np.zeros((n, self.act_dim)),
                          t=np.zeros(n), push_timer=np.full(n, np.inf))

    def split(self, states: np.ndarray):
        return states[:, 1:1 + self.nq], states[:, 1 + self.nq:1 + self.nq + self.nv]

    def reference_states(self, frames: np.ndarray, origin: np.ndarray | None = None) -> np.ndarray:
        """Full-physics state vectors (N, nstate) of the reference at frames (origin = planar anchors)."""
        f = self.ref.frames(frames, origin, fields=("qpos", "qvel"))
        s = np.tile(self.init_state, (len(frames), 1))
        s[:, 0] = 0.0
        s[:, 1:1 + self.nq] = f["qpos"]
        s[:, 1 + self.nq:1 + self.nq + self.nv] = f["qvel"]
        return s

    def reset_envs(self, states: np.ndarray, ts: TrackState, idx, rng: np.random.Generator) -> None:
        """Reference state initialisation for envs idx (random frame, small noise)."""
        idx = np.asarray(idx, int)
        if idx.size == 0:
            return
        cfg, n = self.cfg, idx.size
        last = self.ref.n if self.ref.loop else max(1, self.ref.n - 1)
        k0 = rng.integers(0, last, n) if cfg.rsi else np.zeros(n, np.int64)
        s = self.reference_states(k0, ts.origin[idx])
        qpos = s[:, 1:1 + self.nq]
        qvel = s[:, 1 + self.nq:1 + self.nq + self.nv]
        if cfg.init_joint_noise > 0:
            qpos[:, self.qadr] = np.clip(qpos[:, self.qadr] + rng.uniform(-cfg.init_joint_noise, cfg.init_joint_noise,
                                                                          (n, self.act_dim)), self.q_lo, self.q_hi)
        if cfg.init_vel_noise > 0:
            qvel[:, self.da:self.da + 6] += rng.uniform(-cfg.init_vel_noise, cfg.init_vel_noise, (n, 6))
            qvel[:, self.dadr] += rng.uniform(-cfg.init_vel_noise, cfg.init_vel_noise, (n, self.act_dim))
        states[idx] = s
        ts.frame[idx] = k0
        ts.last_action[idx] = 0.0
        ts.t[idx] = 0.0
        ts.push_timer[idx] = (rng.uniform(0.5, 1.0, n) * cfg.push_interval_s if cfg.push_interval_s > 0
                              else np.inf)

    # ---- action -----------------------------------------------------------------------------------------
    def action_to_ctrl(self, actions: np.ndarray, ts: TrackState) -> np.ndarray:
        """(N, A) actions -> (N, nu) servo targets for the step from frame k to k + 1."""
        a = np.clip(np.nan_to_num(actions), -self.cfg.clip_actions, self.cfg.clip_actions)
        ctrl = self.ref.frames(ts.frame + 1, fields=("ctrl",))["ctrl"].copy()
        ctrl[:, self.ctrl_idx] += self.cfg.action_scale * a
        return clip_ctrl(self.model, ctrl)

    # ---- observation ------------------------------------------------------------------------------------
    def observe(self, qpos: np.ndarray, qvel: np.ndarray, ts: TrackState, fk: tuple | None = None):
        """-> (actor obs (N, obs_dim), critic obs (N, critic_obs_dim)) for the states at reference frame ts.frame.
        fk: precomputed (end-effector positions (N, E, 3), CoM (N, 3)) of these states, else computed."""
        n = len(qpos)
        qa, da = self.qa, self.da
        R = quat_to_mat(qpos[:, qa + 3:qa + 7])
        Rh = _heading(R)
        p = qpos[:, qa:qa + 3]
        lin_b = _tvec(R, qvel[:, da:da + 3])
        ang_b = qvel[:, da + 3:da + 6]
        grav = -R[:, 2, :]
        q = qpos[:, self.qadr]
        qd = qvel[:, self.dadr]
        ph = self.ref.phase(ts.frame)
        prog = ph if not self.ref.loop else np.zeros(n)
        parts = [p[:, 2:3], grav, lin_b * 0.5, ang_b * 0.25, q, qd * 0.1, ts.last_action,
                 np.sin(2 * np.pi * ph)[:, None], np.cos(2 * np.pi * ph)[:, None], prog[:, None]]
        # current-frame tracking errors in the heading frame: what a balance law acts on (the CoM / root
        # drifting from the reference and the root tilting away from it), plus the reference's stance feet
        cur = self.ref.frames(ts.frame, ts.origin, fields=("qpos", "qvel", "keys", "com"))
        keys, com = fk if fk is not None else self.fk(qpos, self.ee_names)
        Rr = quat_to_mat(cur["qpos"][:, qa + 3:qa + 7])
        rot_err = mat_to_rotvec(np.einsum("nji,njk->nik", Rr, R))
        stance = self.ref.stance[np.asarray(ts.frame) % self.ref.n].astype(float)
        if stance.shape[1] != 2:
            stance = np.zeros((n, 2))
        parts += [_tvec(Rh, cur["qpos"][:, qa:qa + 3] - p), _tvec(Rh, cur["qvel"][:, da:da + 3] - qvel[:, da:da + 3]),
                  _tvec(Rh, cur["com"] - com), rot_err, stance]
        for f in self.future:
            r = self.ref.frames(ts.frame + f, ts.origin, fields=("qpos",))["qpos"]
            Rf = quat_to_mat(r[:, qa + 3:qa + 7])
            rel = np.einsum("nji,njk->nik", Rh, Rf)
            parts += [r[:, self.qadr] - q, _tvec(Rh, r[:, qa:qa + 3] - p), rel[:, :, 0], rel[:, :, 1]]
        obs = np.concatenate(parts, axis=1)
        obs = np.where(np.isfinite(obs), obs, 0.0)
        if not self.cfg.asymmetric_critic:
            return obs, obs
        e_cur = _tvec(Rh, keys - p[:, None])
        e_ref = _tvec(_heading(Rr), cur["keys"][:, self.ee] - cur["qpos"][:, None, qa:qa + 3])
        tleft = 1.0 - ts.t / max(self.cfg.episode_seconds, 1e-6)
        cobs = np.concatenate([obs, (e_ref - e_cur).reshape(n, -1), tleft[:, None]], axis=1)
        return obs, np.where(np.isfinite(cobs), cobs, 0.0)

    # ---- reward -----------------------------------------------------------------------------------------
    def errors(self, qpos: np.ndarray, qvel: np.ndarray, ts: TrackState) -> dict:
        """Tracking errors of states (N) against the reference at ts.frame (the e_i of the module docstring),
        plus diagnostics: joint_rmse (rad), ee_err (mean m), root_dist (m), tilt (rad), dz (m)."""
        qa, da = self.qa, self.da
        ref = self.ref.frames(ts.frame, ts.origin)
        rq, rv = ref["qpos"], ref["qvel"]
        dq = qpos[:, self.qadr] - rq[:, self.qadr]
        dqd = qvel[:, self.dadr] - rv[:, self.dadr]
        keys, com = self.fk(qpos, self.ee_names)
        R = quat_to_mat(qpos[:, qa + 3:qa + 7])
        Rr = quat_to_mat(rq[:, qa + 3:qa + 7])
        p, pr = qpos[:, qa:qa + 3], rq[:, qa:qa + 3]
        e_cur = _tvec(_heading(R), keys - p[:, None])
        e_ref = _tvec(_heading(Rr), ref["keys"][:, self.ee] - pr[:, None])
        ee_sq = np.sum((e_cur - e_ref) ** 2, axis=-1)
        Rd = np.einsum("nji,njk->nik", Rr, R)
        ang = np.arccos(np.clip((np.trace(Rd, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0))
        tilt = np.arccos(np.clip(np.sum(R[:, :, 2] * Rr[:, :, 2], axis=1), -1.0, 1.0))
        dp = p - pr
        root = (np.sum(dp ** 2, axis=1) + 0.1 * ang ** 2
                + 0.01 * np.sum((qvel[:, da:da + 3] - rv[:, da:da + 3]) ** 2, axis=1)
                + 0.001 * np.sum((qvel[:, da + 3:da + 6] - rv[:, da + 3:da + 6]) ** 2, axis=1))
        return {"pose": np.sum(dq ** 2, axis=1), "vel": np.sum(dqd ** 2, axis=1), "end_eff": np.sum(ee_sq, axis=1),
                "com": np.sum((com - ref["com"]) ** 2, axis=1), "root": root,
                "joint_rmse": np.sqrt(np.mean(dq ** 2, axis=1)), "ee_err": np.mean(np.sqrt(ee_sq), axis=1),
                "root_dist": np.linalg.norm(dp, axis=1), "tilt": tilt, "dz": np.abs(dp[:, 2]),
                "fk": (keys, com)}

    def reward(self, qpos: np.ndarray, qvel: np.ndarray, ts: TrackState):
        """Reward (unscaled, in [0, 1]) of states that should be at reference frame ts.frame; terms (each
        w_i exp(-s_i e_i)), terminated mask and the error diagnostics."""
        e = self.errors(qpos, qvel, ts)
        terms = {k: self.weights[k] * np.exp(-self.cfg.scales[k] * e[k]) for k in self.weights}
        r = np.zeros(len(qpos))
        for v in terms.values():
            r += v
        cfg = self.cfg
        bad = ~(np.isfinite(qpos).all(axis=1) & np.isfinite(qvel).all(axis=1))
        terminated = (bad | (e["dz"] > cfg.term_height) | (e["tilt"] > np.radians(cfg.term_tilt_deg))
                      | (e["root_dist"] > cfg.term_distance))
        r = np.where(bad, 0.0, r)
        return r, terms, terminated, e
