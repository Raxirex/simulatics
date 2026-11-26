"""Turn a reference motion into a physically simulated one: sampling-based control reconstruction.

A reference clip (motion capture retargeted to the robot) is only a sequence of poses. This module finds motor
commands that make the simulated robot perform it under physics, after Liu et al., "Sampling-based
contact-rich motion control" (SIGGRAPH 2010), known as SAMCON:

    The motors are position servos, so the natural command is the reference joint angles. Played open loop
    they fail, because the reference ignores balance, contact timing and the robot's own mass distribution.
    So the clip is cut into short windows (0.1 s). In each window many candidate corrections (a constant
    offset added to the servo targets) are simulated in parallel from the states kept from the previous
    window; the candidates whose end state is closest to the reference (pose, root orientation, centre of
    mass and its velocity, feet and hands, balance) are kept as starting points for the next window. The best
    chain of windows is the result. Further passes resample around it with a smaller spread.

The result is a `Tape`: the servo target of every motor at every control step, and the state to start from.
Playing a tape is plain physics: only data.ctrl is written (see `TapeController`). Nothing holds the robot
up; the feet push on the floor and gravity does the rest.

A tape is open loop. It reproduces exactly on the model it was made for, started from its start state, with
solver warm-starting off (so the physics is a pure function of state and command). It is not robust to pushes
or to a different robot: `Tape.matches(model)` says whether a model is the one it was made for.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from mujoco import rollout

from .motion import Motion, free_root, quat_slerp

STATE = int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
FEATURE_BODIES = ("left_foot", "right_foot", "left_hand", "right_hand")


def fingerprint(model: mujoco.MjModel) -> str:
    """Hash of everything that decides how the robot moves: masses, geometry, joints, motors, time step."""
    h = hashlib.sha256()
    for a in (model.body_mass, model.body_inertia, model.body_pos, model.geom_size, model.geom_pos,
              model.geom_friction, model.jnt_range, model.dof_damping, model.dof_armature, model.actuator_gainprm,
              model.actuator_biasprm, model.actuator_forcerange, model.actuator_ctrlrange,
              np.array([model.opt.timestep]), model.opt.gravity):
        h.update(np.ascontiguousarray(np.round(np.asarray(a, float), 7)).tobytes())
    return h.hexdigest()[:16]


def prepare_model(model: mujoco.MjModel) -> dict:
    """Put a model in the mode recorded motions use, and return what to restore afterwards.

    Warm-starting off: the physics is then a pure function of state and command, so a recording reproduces.
    Command clamp off: the motors are position servos (torque = kp (target - angle) - kv speed, clipped to the
    motor's torque limit). Clamping the target to the joint range caps the torque a servo can ask for near
    the end of its travel, and the servo's damping eats the rest at speed. Without the clamp the command can
    ask for any torque up to the limit. The torque limits themselves (forcerange) are not changed."""
    saved = {"disableflags": int(model.opt.disableflags), "ctrllimited": model.actuator_ctrllimited.copy()}
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
    model.actuator_ctrllimited[:] = 0
    return saved


def restore_model(model: mujoco.MjModel, saved: dict | None) -> None:
    if saved:
        model.opt.disableflags = saved["disableflags"]
        model.actuator_ctrllimited[:] = saved["ctrllimited"]


@dataclass
class TrackConfig:
    control_steps: int = 4          # physics steps per command (0.02 s at a 5 ms time step)
    window: int = 5                 # commands per window (0.1 s)
    samples: int = 1200             # candidates simulated per window and refinement round
    refine: int = 3                 # rounds per window: resample around the best candidates of the last round
    w_mid: float = 0.5              # weight of the mid-window comparison
    keep: int = 80                  # states carried to the next window
    passes: int = 3                 # independent attempts; stops at the first that ends on target
    good_enough: float = 0.15       # final-window cost that counts as ending on target
    sigma_legs: float = 0.20        # rad, spread of the target offset for hips, knees, ankles, waist
    sigma_arms: float = 0.10
    lead_in: float = 0.5            # s, blend from the standing pose into the clip
    lead_out: float = 0.6           # s, blend from the clip's last pose to standing
    hold: float = 1.2               # s of standing at the end
    w_pose: float = 4.0
    w_root: float = 12.0
    w_com: float = 40.0
    w_comvel: float = 1.5
    w_ee: float = 18.0
    w_balance: float = 25.0
    w_place: float = 0.05           # weight of horizontal position relative to height (within w_com)
    threads: int = 0                # 0 = all cores
    seed: int = 0


@dataclass
class Tape:
    """Servo targets for every control step, and where to start."""
    ctrl: np.ndarray                # (n, nu)
    control_steps: int
    timestep: float
    qpos0: np.ndarray
    qvel0: np.ndarray
    states: np.ndarray              # (n + 1, nq): the simulated poses (what playing the tape produces)
    reference: np.ndarray           # (n + 1, nq): the poses it was asked to follow
    model: str                      # fingerprint of the model it was made for
    name: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.ctrl) * self.control_steps * self.timestep

    def matches(self, model: mujoco.MjModel) -> bool:
        return fingerprint(model) == self.model

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ctrl=self.ctrl, qpos0=self.qpos0, qvel0=self.qvel0,
                            states=self.states.astype(np.float32), reference=self.reference.astype(np.float32),
                            header=json.dumps({"kind": "simulatics-tape", "version": 1, "name": self.name,
                                               "control_steps": self.control_steps, "timestep": self.timestep,
                                               "model": self.model, "meta": self.meta}))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Tape":
        with np.load(path, allow_pickle=False) as z:
            h = json.loads(str(z["header"]))
            if h.get("kind") != "simulatics-tape":
                raise ValueError(f"{path} is not a motion tape")
            return cls(z["ctrl"].astype(float), int(h["control_steps"]), float(h["timestep"]), z["qpos0"].copy(),
                       z["qvel0"].copy(), z["states"].astype(float), z["reference"].astype(float), h["model"],
                       h.get("name", ""), h.get("meta", {}))


# =========================================================================================================
# Reference: standing -> clip -> standing, in the model's qpos layout
# =========================================================================================================
def _lowest_sole(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> float:
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    feet = [g for g in range(model.ngeom) if model.body(int(model.geom_bodyid[g])).name.endswith("_foot")]
    low = []
    for g in feet:
        R = data.geom_xmat[g].reshape(3, 3)
        low.append(float(data.geom_xpos[g][2] - np.abs(R[2] * model.geom_size[g]).sum()))
    return min(low) if low else float(q[2])


def _blend(a: np.ndarray, b: np.ndarray, n: int, qa: int) -> np.ndarray:
    """n poses easing from a to b (exclusive of a, inclusive of b)."""
    w = (np.arange(1, n + 1) / n)
    w = 3 * w ** 2 - 2 * w ** 3
    out = (1 - w[:, None]) * a + w[:, None] * b
    out[:, qa + 3:qa + 7] = np.array([quat_slerp(a[qa + 3:qa + 7], b[qa + 3:qa + 7], float(x)) for x in w])
    return out


def build_reference(model: mujoco.MjModel, motion: Motion, q_stand: np.ndarray, cfg: TrackConfig) -> np.ndarray:
    """(T, nq) at the control rate: ease from standing into the clip, the clip, ease back to standing, hold.
    Frame 0 is the start state: the standing pose, raised onto whatever the clip starts on."""
    from .motion import yaw_of_quat, yaw_quat
    _, qa, _ = free_root(model)
    dt = cfg.control_steps * float(model.opt.timestep)
    m = motion.resample(1.0 / dt) if abs(motion.fps - 1.0 / dt) > 1e-6 else motion
    stand_yaw = float(yaw_of_quat(q_stand[qa + 3:qa + 7]))
    clip = m.aligned(float(q_stand[qa]), float(q_stand[qa + 1]), stand_yaw).qpos(model)
    d = mujoco.MjData(model)
    rise = _lowest_sole(model, d, clip[0]) - _lowest_sole(model, d, q_stand)
    if abs(rise) < 0.04:
        clip[:, qa + 2] -= rise                 # same floor: remove the small mismatch
    else:
        q_stand = q_stand.copy()
        q_stand[qa + 2] += rise                 # the clip starts on something (a box): stand on it
    end = q_stand.copy()
    end[qa:qa + 2] = clip[-1, qa:qa + 2]
    end[qa + 3:qa + 7] = yaw_quat(float(yaw_of_quat(clip[-1, qa + 3:qa + 7])))
    end[qa + 2] += _lowest_sole(model, d, clip[-1]) - _lowest_sole(model, d, end)     # stand where the clip ends
    n_in, n_out, n_hold = (max(1, int(round(t / dt))) for t in (cfg.lead_in, cfg.lead_out, cfg.hold))
    return np.concatenate([q_stand[None], _blend(q_stand, clip[0], n_in, qa), clip[1:], _blend(clip[-1], end, n_out, qa),
                           np.repeat(end[None], n_hold, axis=0)])


def settle_lying(model: mujoco.MjModel, pose: np.ndarray, seconds: float = 1.2) -> np.ndarray:
    """Where the robot comes to rest when it is laid down in `pose` and its servos hold that pose: the pose is
    lifted clear of the floor, dropped, and simulated until still. This is the physical start state of a motion
    that begins on the ground (a retargeted pose itself floats or sinks by a few centimetres)."""
    _, qa, _ = free_root(model)
    d = mujoco.MjData(model)
    q = np.array(pose, float)
    robot = np.flatnonzero(model.body_rootid[model.geom_bodyid] == model.body_rootid[1])
    d.qpos[:] = q
    mujoco.mj_kinematics(model, d)
    low = float(np.min(d.geom_xpos[robot, 2] - model.geom_rbound[robot]))
    q[qa + 2] += max(0.0, 0.01 - low)                        # nothing under the floor at the start
    d.qpos[:] = q
    d.qvel[:] = 0
    act_q = np.array([model.jnt_qposadr[model.actuator_trnid[a, 0]] for a in range(model.nu)])
    d.ctrl[:] = q[act_q]
    mujoco.mj_forward(model, d)
    mujoco.mj_step(model, d, int(round(seconds / model.opt.timestep)))
    return d.qpos.copy()


def build_reference_from_ground(model: mujoco.MjModel, motion: Motion, q_stand: np.ndarray, cfg: TrackConfig) -> np.ndarray:
    """(T, nq) for a motion that starts on the ground: the settled lying pose, a short ease into the clip, the
    clip, an ease to standing, a hold. Frame 0 is the start state (lying, at rest)."""
    from .motion import yaw_of_quat, yaw_quat
    _, qa, _ = free_root(model)
    dt = cfg.control_steps * float(model.opt.timestep)
    m = motion.resample(1.0 / dt) if abs(motion.fps - 1.0 / dt) > 1e-6 else motion
    clip = m.qpos(model)
    clip[:, qa:qa + 2] += q_stand[qa:qa + 2] - clip[0, qa:qa + 2]          # lie down where the robot stands
    for i in range(len(clip)):                                              # non-robot coordinates (objects) as in the scene
        pass
    full = np.tile(q_stand, (len(clip), 1))
    nrobot = len(clip[0])
    full[:, :nrobot] = clip
    clip = full
    start = settle_lying(model, clip[0])
    d = mujoco.MjData(model)
    end = q_stand.copy()
    end[qa:qa + 2] = clip[-1, qa:qa + 2]
    end[qa + 3:qa + 7] = yaw_quat(float(yaw_of_quat(clip[-1, qa + 3:qa + 7])))
    end[qa + 2] += _lowest_sole(model, d, clip[-1]) - _lowest_sole(model, d, end)
    n_in, n_out, n_hold = (max(1, int(round(t / dt))) for t in (0.3, cfg.lead_out, cfg.hold))
    k = min(n_in, len(clip) - 1)
    return np.concatenate([start[None], _blend(start, clip[k], n_in, qa), clip[k + 1:], _blend(clip[-1], end, n_out, qa),
                           np.repeat(end[None], n_hold, axis=0)])


# =========================================================================================================
# Tracker
# =========================================================================================================
class Tracker:
    def __init__(self, xml: str, cfg: TrackConfig | None = None):
        self.cfg = cfg or TrackConfig()
        self.plain = mujoco.MjModel.from_xml_string(xml)          # the model as the session builds it
        spec = mujoco.MjSpec.from_string(xml)
        root = self.plain.body(1).name
        S, O = mujoco.mjtSensor, mujoco.mjtObj
        spec.add_sensor(name="trk_com", type=S.mjSENS_SUBTREECOM, objtype=O.mjOBJ_BODY, objname=root)
        spec.add_sensor(name="trk_comvel", type=S.mjSENS_SUBTREELINVEL, objtype=O.mjOBJ_BODY, objname=root)
        self.bodies = [b for b in FEATURE_BODIES if mujoco.mj_name2id(self.plain, O.mjOBJ_BODY, b) >= 0]
        for b in self.bodies:
            spec.add_sensor(name=f"trk_{b}", type=S.mjSENS_FRAMEPOS, objtype=O.mjOBJ_BODY, objname=b)
        self.model = spec.compile()                               # same dynamics, plus feature sensors
        for m in (self.model, self.plain):
            prepare_model(m)
        m = self.model
        self.nfeat = 6 + 3 * len(self.bodies)
        self.fadr = m.nsensordata - self.nfeat
        _, self.qa, self.da = free_root(m)
        self.nstate = mujoco.mj_stateSize(m, STATE)
        self.act_q = np.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        names = [m.joint(int(m.actuator_trnid[a, 0])).name for a in range(m.nu)]
        self.sigma = np.array([0.0 if "gripper" in n else
                               (self.cfg.sigma_arms if any(k in n for k in ("shoulder", "elbow", "wrist"))
                                else self.cfg.sigma_legs) for n in names])
        hinge = [j for j in range(m.njnt) if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)]
        self.jq = np.array([m.jnt_qposadr[j] for j in hinge])
        self.jw = np.array([0.0 if "gripper" in m.joint(j).name else
                            (0.3 if any(k in m.joint(j).name for k in ("shoulder", "elbow", "wrist")) else 1.0)
                            for j in hinge])
        kp = m.actuator_gainprm[:, 0]
        self.kv_over_kp = np.where(kp > 0, -m.actuator_biasprm[:, 2] / np.maximum(kp, 1e-9), 0.0)
        self.act_v = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        # how far past the joint range a command may go: enough to ask for the full torque limit
        reach = np.where(kp > 0, m.actuator_forcerange[:, 1] / np.maximum(kp, 1e-9), 0.0) * 1.5
        self.lo, self.hi = m.actuator_ctrlrange[:, 0] - reach, m.actuator_ctrlrange[:, 1] + reach
        self.limited = np.ones(m.nu, bool)
        import os
        self.threads = self.cfg.threads or max(1, (os.cpu_count() or 4))
        self.datas = [mujoco.MjData(m) for _ in range(self.threads)]
        self.pool = rollout.Rollout(nthread=self.threads)
        self.scratch = mujoco.MjData(m)

    def close(self) -> None:
        self.pool.close()

    # ---- features -------------------------------------------------------------------------------------
    def ref_features(self, qref: np.ndarray) -> np.ndarray:
        """(T, nfeat) of the reference, with its velocity from finite differences."""
        m, d = self.model, self.scratch
        dt = self.cfg.control_steps * float(m.opt.timestep)
        out = np.zeros((len(qref), self.nfeat))
        for t in range(len(qref)):
            d.qpos[:] = qref[t]
            d.qvel[:] = 0
            a, b = qref[max(t - 1, 0)], qref[min(t + 1, len(qref) - 1)]
            mujoco.mj_differentiatePos(m, d.qvel, dt * (min(t + 1, len(qref) - 1) - max(t - 1, 0)) or dt, a, b)
            mujoco.mj_forward(m, d)
            out[t] = d.sensordata[self.fadr:]
        return out

    def cost(self, state: np.ndarray, feat: np.ndarray, qref: np.ndarray, fref: np.ndarray) -> np.ndarray:
        """Cost of each simulated end state (n, nstate) with features (n, nfeat) against one reference frame."""
        c, qa = self.cfg, self.qa
        q = state[:, 1:1 + self.model.nq]
        pose = (self.jw * (q[:, self.jq] - qref[self.jq]) ** 2).sum(1) / self.jw.sum()
        dot = np.abs(q[:, qa + 3:qa + 7] @ qref[qa + 3:qa + 7])
        root = (2 * np.arccos(np.clip(dot, 0, 1))) ** 2
        com, comv = feat[:, 0:3], feat[:, 3:6]
        # height matters exactly; where on the floor matters little (a jump that lands short must not be dragged
        # to the reference's landing spot: that tips the robot over)
        e_com = ((com - fref[0:3]) ** 2 * np.array([c.w_place, c.w_place, 2.0])).sum(1)
        e_comv = ((comv - fref[3:6]) ** 2).sum(1)
        ee = feat[:, 6:].reshape(len(feat), -1, 3).copy()
        ee_ref = fref[6:].reshape(-1, 3).copy()
        ee[:, :, :2] -= com[:, None, :2]                           # feet and hands relative to the centre of mass
        ee_ref[:, :2] -= fref[0:2]
        w_ee = np.array([1.0 if "foot" in b else 0.3 for b in self.bodies])
        e_ee = (w_ee * ((ee - ee_ref) ** 2).sum(2)).sum(1) / w_ee.sum()
        feet = [i for i, b in enumerate(self.bodies) if "foot" in b]
        if feet:   # balance: where the centre of mass sits over the feet, as in the reference
            e_bal = ((ee[:, feet, :2].mean(1) - ee_ref[feet, :2].mean(0)) ** 2).sum(1)
        else:
            e_bal = 0.0
        total = (c.w_pose * pose + c.w_root * root + c.w_com * e_com + c.w_comvel * e_comv + c.w_ee * e_ee
                 + c.w_balance * e_bal)
        return np.where(np.isfinite(total), total, 1e9)

    # ---- one pass -------------------------------------------------------------------------------------
    def _targets(self, qref: np.ndarray) -> np.ndarray:
        """Servo targets from the reference: the pose one command ahead, plus velocity feed-forward."""
        dt = self.cfg.control_steps * float(self.model.opt.timestep)
        speed = (qref[1:, self.act_q] - qref[:-1, self.act_q]) / dt
        tgt = qref[1:, self.act_q] + self.kv_over_kp * speed     # cancel the servo's damping along the reference
        return np.clip(tgt, self.lo, self.hi)

    def _pass(self, s0: np.ndarray, qref: np.ndarray, fref: np.ndarray, mean: np.ndarray, sigma: np.ndarray,
              rng: np.random.Generator, log=None) -> tuple[np.ndarray, float, list[float]]:
        c, m = self.cfg, self.model
        n_cmd = len(qref) - 1
        n_win = int(np.ceil(n_cmd / c.window))
        targets = self._targets(qref)
        starts = s0[None].copy()
        hist_delta, hist_parent, hist_cost = [], [], []
        for w in range(n_win):
            i0, i1 = w * c.window, min((w + 1) * c.window, n_cmd)
            mu, sd = mean[w].copy(), sigma.copy()
            all_delta, all_parent, all_end, all_cost = [], [], [], []
            for it in range(c.refine):
                parents = np.arange(c.samples) % len(starts)
                delta = mu + sd * rng.standard_normal((c.samples, m.nu))
                delta[:len(starts)] = mu                          # every kept state also tries the mean itself
                tg = targets[i0:i1][None] + delta[:, None, :]
                tg = np.where(self.limited, np.clip(tg, self.lo, self.hi), tg)
                ctrl = np.repeat(tg, c.control_steps, axis=1)
                st, sens = self.pool.rollout(m, self.datas, starts[parents], ctrl)
                end, feat = st[:, -1], sens[:, -1, self.fadr:]
                cost = self.cost(end, feat, qref[i1], fref[i1])
                if c.w_mid > 0 and i1 - i0 > 2:                   # also judge the middle of the window
                    k = (i1 - i0) // 2
                    cost = cost + c.w_mid * self.cost(st[:, k * c.control_steps - 1],
                                                      sens[:, k * c.control_steps - 1, self.fadr:], qref[i0 + k], fref[i0 + k])
                all_delta.append(delta)
                all_parent.append(parents)
                all_end.append(end)
                all_cost.append(cost)
                elite = delta[np.argsort(cost)[:max(8, c.samples // 20)]]
                mu = elite.mean(0)                                # resample around what worked
                sd = np.maximum(elite.std(0), 0.25 * sigma)
            delta, parents, end, cost = (np.concatenate(x) for x in (all_delta, all_parent, all_end, all_cost))
            order = np.argsort(cost)[:c.keep]
            hist_delta.append(delta[order])
            hist_parent.append(parents[order])
            hist_cost.append(cost[order])
            starts = end[order]
            if log and (w % 10 == 0 or w == n_win - 1):
                log(f"    window {w + 1}/{n_win}  best {cost[order[0]]:.4f}  median kept {np.median(cost[order]):.4f}")
        # back-track the best chain
        k = 0
        best = np.zeros((n_win, m.nu))
        costs = []
        for w in range(n_win - 1, -1, -1):
            best[w] = hist_delta[w][k]
            costs.append(float(hist_cost[w][k]))
            k = int(hist_parent[w][k])
        return best, float(np.mean(costs)), costs[::-1]

    def replay(self, s0: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
        """Plain mj_step replay of a tape on the session's own model (no sensors). Returns qpos per command."""
        m = self.plain
        d = mujoco.MjData(m)
        mujoco.mj_setState(m, d, s0, STATE)
        mujoco.mj_forward(m, d)
        out = [d.qpos.copy()]
        for u in ctrl:
            d.ctrl[:] = u
            mujoco.mj_step(m, d, self.cfg.control_steps)
            out.append(d.qpos.copy())
        return np.array(out)

    def settle_to(self, qpos: np.ndarray, qvel: np.ndarray, q_stand: np.ndarray, blend: float = 0.6,
                  hold: float = 1.2, log=None) -> tuple[np.ndarray, np.ndarray, float]:
        """Commands that bring the robot from any moving state back to a standing pose and keep it there, with
        balance (the same windowed search as track()). Returns (ctrl (n, nu), simulated qpos (n + 1, nq), cost).
        Used to end a trained move in the stance the balance controller takes over from."""
        c, m, qa = self.cfg, self.model, self.qa
        dt = c.control_steps * float(m.opt.timestep)
        target = np.array(qpos, float)                      # everything that is not the robot stays as it is
        nrobot = [int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.body_rootid[m.jnt_bodyid[j]] == m.body_rootid[1]
                  and m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
        target[nrobot] = q_stand[nrobot]
        target[qa:qa + 7] = q_stand[qa:qa + 7]
        qref = np.concatenate([np.array(qpos, float)[None], _blend(np.array(qpos, float), target, max(1, int(round(blend / dt))), qa),
                               np.repeat(target[None], max(1, int(round(hold / dt))), axis=0)])
        fref = self.ref_features(qref)
        d = self.scratch
        mujoco.mj_resetData(m, d)
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        mujoco.mj_forward(m, d)
        s0 = np.zeros(self.nstate)
        mujoco.mj_getState(m, d, s0, STATE)
        n_cmd = len(qref) - 1
        n_win = int(np.ceil(n_cmd / c.window))
        best, best_cost = None, np.inf
        for p in range(c.passes):
            delta, cost, per_window = self._pass(s0, qref, fref, np.zeros((n_win, m.nu)), self.sigma.copy(),
                                                 np.random.default_rng(c.seed + 100 + p), log)
            if cost < best_cost:
                best, best_cost = delta, cost
            if per_window[-1] < c.good_enough:
                break
        targets = np.clip(self._targets(qref) + np.repeat(best, c.window, axis=0)[:n_cmd], self.lo, self.hi)
        dp = mujoco.MjData(self.plain)
        dp.qpos[:] = qpos
        dp.qvel[:] = qvel
        mujoco.mj_forward(self.plain, dp)
        states = [dp.qpos.copy()]
        for u in targets:
            dp.ctrl[:] = u
            mujoco.mj_step(self.plain, dp, c.control_steps)
            states.append(dp.qpos.copy())
        return targets, np.array(states), best_cost

    def track(self, motion: Motion, qpos0: np.ndarray, qvel0: np.ndarray | None = None, name: str = "",
              log=None, from_ground: bool = False) -> Tape:
        c, m = self.cfg, self.model
        t_start = time.perf_counter()
        qref = (build_reference_from_ground if from_ground else build_reference)(m, motion, qpos0, c)
        fref = self.ref_features(qref)
        qpos0 = qref[0].copy()
        d = self.scratch
        d.qpos[:] = qpos0
        d.qvel[:] = 0 if qvel0 is None else qvel0
        d.time = 0.0
        mujoco.mj_forward(m, d)
        s0 = np.zeros(self.nstate)
        mujoco.mj_getState(m, d, s0, STATE)
        n_cmd = len(qref) - 1
        n_win = int(np.ceil(n_cmd / c.window))
        mean, sigma = np.zeros((n_win, m.nu)), self.sigma.copy()
        best, best_cost, history, best_windows = None, np.inf, [], []
        for p in range(c.passes):      # independent attempts (different random draws); the best chain wins
            if log:
                log(f"  attempt {p + 1}/{c.passes}")
            delta, cost, per_window = self._pass(s0, qref, fref, mean, sigma, np.random.default_rng(c.seed + p), log)
            history.append(round(cost, 5))
            if cost < best_cost:
                best, best_cost, best_windows = delta, cost, per_window
            if per_window[-1] < c.good_enough:
                break
        targets = self._targets(qref) + np.repeat(best, c.window, axis=0)[:n_cmd]
        targets = np.where(self.limited, np.clip(targets, self.lo, self.hi), targets)
        s0p = np.zeros(mujoco.mj_stateSize(self.plain, STATE))
        dp = mujoco.MjData(self.plain)
        dp.qpos[:] = qpos0
        dp.qvel[:] = 0 if qvel0 is None else qvel0
        mujoco.mj_getState(self.plain, dp, s0p, STATE)
        states = self.replay(s0p, targets)
        meta = {"cost": round(best_cost, 5), "cost_per_pass": history, "worst_window": round(float(max(best_windows)), 4),
                "seconds": round(time.perf_counter() - t_start, 1), "config": asdict(c), **summarise(self.plain, states, qref, c)}
        return Tape(targets, c.control_steps, float(m.opt.timestep), np.array(qpos0, float),
                    np.zeros(m.nv) if qvel0 is None else np.array(qvel0, float), states, qref, fingerprint(self.plain),
                    name, meta)


def summarise(model: mujoco.MjModel, states: np.ndarray, qref: np.ndarray, cfg: TrackConfig) -> dict:
    """What actually happened in the simulated motion, measured from its states."""
    from .motion import quat_to_mat
    _, qa, _ = free_root(model)
    d = mujoco.MjData(model)
    dt = cfg.control_steps * float(model.opt.timestep)
    floor_like = [g for g in range(model.ngeom) if model.geom_bodyid[g] == 0]
    robot = np.flatnonzero(model.body_rootid[model.geom_bodyid] == model.body_rootid[1])
    airborne, com_z = [], []
    for q in states:
        d.qpos[:] = q
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        mujoco.mj_collision(model, d)
        touching = any((d.contact[i].geom1 in floor_like) != (d.contact[i].geom2 in floor_like) for i in range(d.ncon))
        airborne.append(not touching)
        com_z.append(float(d.subtree_com[1][2]))
    airborne = np.array(airborne)
    up = quat_to_mat(states[:, qa + 3:qa + 7])[:, 2, 2]
    # net rotation about horizontal axes: rotation vectors of successive frames summed (a flip is about 2 pi)
    R = quat_to_mat(states[:, qa + 3:qa + 7])
    total = np.zeros(3)
    for a, b in zip(R[:-1], R[1:]):
        dR = b @ a.T
        ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        axis = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
        n = np.linalg.norm(axis)
        if n > 1e-9:
            total += ang * axis / n
    rot = float(np.linalg.norm(total[:2]))
    longest = run = 0
    for a in airborne:
        run = run + 1 if a else 0
        longest = max(longest, run)
    err = np.linalg.norm(states[:, qa:qa + 3] - qref[:, qa:qa + 3], axis=1)
    return {"flight_s": round(longest * dt, 3), "com_rise_m": round(max(com_z) - com_z[0], 3),
            "rotation_rad": round(rot, 2), "min_up": round(float(up.min()), 2), "final_up": round(float(up[-1]), 3),
            "final_height_m": round(float(states[-1, qa + 2]), 3), "reference_final_height_m": round(float(qref[-1, qa + 2]), 3),
            "root_error_mean_m": round(float(err.mean()), 3), "root_error_max_m": round(float(err.max()), 3),
            "final_root_error_m": round(float(err[-1]), 3)}


class TapeController:
    """Plays a Tape in the live simulation: each control step it writes the servo targets and nothing else."""
    name = "motion-physical"

    def __init__(self, model: mujoco.MjModel, tape: Tape):
        if not tape.matches(model):
            raise ValueError("this recorded motion was made for a different robot or world (its parameters, "
                             "accessories or objects changed); it only reproduces on the one it was made for")
        self.model, self.tape = model, tape
        self.control_dt = tape.control_steps * tape.timestep
        self.i = 0
        self.done = False
        self._saved = prepare_model(model)

    def release(self, model) -> None:
        """Give the model back its normal settings (warm-starting, command clamp). Call when the tape is over."""
        restore_model(model, self._saved)
        self._saved = None

    def reset(self, model, data) -> None:
        """Start state of the motion: the standing pose it was made from, at rest."""
        self.i, self.done = 0, False
        data.qpos[:] = self.tape.qpos0
        data.qvel[:] = self.tape.qvel0
        if model.na:
            data.act[:] = 0
        data.ctrl[:] = self.tape.ctrl[0]
        mujoco.mj_forward(model, data)

    def set_path(self, path_xy, final_yaw=None) -> None:
        pass

    def step(self, model, data) -> None:
        if self.i < len(self.tape.ctrl):
            data.ctrl[:] = self.tape.ctrl[self.i]
            self.i += 1
        else:
            self.done = True

    def status(self) -> dict:
        return {"mode": "done" if self.done else "walk", "motion": self.tape.name,
                "message": f"'{self.tape.name}' in physics (recorded motor commands, open loop)",
                "time_s": round(self.i * self.control_dt, 3), "duration_s": round(self.tape.duration, 3)}
