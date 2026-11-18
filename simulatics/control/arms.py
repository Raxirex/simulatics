"""Arm and hand skills: reach, grasp, release, point, wave, raise, home.

An LLM pilot (Claude, looking at camera images) decides *what* to do; this module
does the *how* with classical, model-based control, so every skill is deterministic
and works on any body the parametric generator produces.

Method
------
Inverse kinematics
    Damped least squares (DLS) on the chain shoulder_y, shoulder_x, elbow, wrist_y,
    wrist_x (Wampler 1986; Nakamura & Hanafusa 1986; see Buss 2004, "Introduction to
    inverse kinematics with Jacobian transpose, pseudoinverse and damped least squares
    methods"): dq = J^T (J J^T + lambda^2 I)^-1 e. J comes from ``mujoco.mj_jac`` at
    the hand site. The task is the palm position (3 rows) plus, optionally, the
    direction the fingers point (2 effective rows, the error is the rotation vector
    a x a* projected off the pointing axis). Joints that sit on a limit and are pushed
    further out are removed from the next solve ("clamping loop", Baerlocher & Boulic
    2004), and the result is clamped to the joint ranges. In the redundant case
    (position only) a null-space term keeps the wrist near neutral.

    The solve runs on a scratch ``MjData`` whose root, legs and waist are copied from
    the live state at every control step, so world-frame targets are re-solved against
    the measured torso pose: a reach keeps its target while the body sways or walks.

Trajectories
    Minimum-jerk interpolation (Flash & Hogan 1985), s = 10t^3 - 15t^4 + 6t^5, in
    Cartesian space for reaches (the start point is carried in the torso frame) and in
    joint space for gestures. A reach with an approach direction goes through a
    pre-grasp point 8 cm back along that direction, then moves in at 10 cm/s.
    Before a reach the straight hand path is ray-cast (``mj_ray``) against the world
    (tables, obstacles, props, floor), and the whole arm's sweep while tracking it is
    predicted with a kinematic IK dry run and checked for clearance (``mj_geomDistance``),
    since the tracked hand can sag well below the line. If it is blocked, a hand that hangs beside the
    body first folds into a fixed "tuck" pose (elbow folded, hand up at the chest
    pointing down, behind any table edge) and a hand that is further out rises early
    through a clear via point. Gestures that start with the hand out in front fold
    through the same tuck pose, so the arm drops behind the table edge, not through it.

Servo targets
    The arm motors are position servos (torque = kp (ctrl - q) - kv qdot). To make
    the joint actually reach q_des the controller adds the torque the servo has to
    supply, ctrl = q_des + (qfrc_bias + J^T m_held g + kv qdot_des) / kp: gravity of the
    arm and of a held object (the "PD + gravity compensation" law, Takegaki & Arimoto
    1981) plus velocity feed-forward so the servo damping does not drag the joint behind
    a moving target, and a small bounded joint-space integral term once a motion ends.
    Joint targets are rate-limited (4 rad/s) so a re-solved target can never whip an arm.

Balance
    The arm controller never moves the legs. With the legs simply held at their
    standing targets the built-in humanoid tips over backwards when one straight arm is
    held about 25 deg back (two arms: about 15 deg; measured), so the shoulder pitch is
    limited to 5 deg backwards except in the folded tuck pose (20 deg, arm mass near the
    shoulder). Pointing at something behind the robot fails with "turn towards it first".

Grasping
    "grasp" closes the gripper; a prop is held once both finger pads touch the same
    body (MuJoCo contact list) for 0.15 s. Friction alone holds 4-6 cm objects (pads
    have mu = 1.5 and torsional friction, condim 4, and near-hard contacts, solimp
    0.99/0.999, solref 0.01: with MuJoCo's default soft contacts a 70 g cube crept
    9.5 mm out of the grip in 3 s, with these about 1 mm). An optional grasp assist
    (``grasp_assist=True``) additionally switches on a weld equality between hand and
    prop at that moment and off on release; it is a simulation convenience for
    objects that friction cannot hold, not a physical model.

Usage
-----
    arms = ArmController(model)
    arms.reset(model, data)                    # binds the live MjData
    every control step, AFTER the leg/base controller wrote data.ctrl:
        arms.step(model, data)                 # overwrites only arm/hand actuators of engaged arms
    arms.command("right", "reach", target=[x, y, z], approach=[0, 0, -1])
    arms.command("right", "grasp"); arms.command("right", "release")
    arms.status()                              # see RobotInterface.arm_status()

An arm is "engaged" from its first command until "relax"; only engaged arms write
``data.ctrl``, so a walking controller keeps swinging the arms until they are used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np

SIDES = ("left", "right")
ARM_JOINTS = ("shoulder_y", "shoulder_x", "elbow", "wrist_y", "wrist_x")
ROLL_JOINTS = ("shoulder_x", "wrist_x")      # mirrored between sides (positive = outward on the left)
SKILLS = ("reach", "grasp", "release", "point_at", "wave", "raise", "home", "stop", "relax")
ALIASES = {"close": "grasp", "open": "release", "point": "point_at", "drop": "release", "rest": "home"}


def min_jerk(tau: float) -> float:
    """Minimum-jerk position profile (Flash & Hogan 1985) for tau in [0, 1]."""
    t = min(1.0, max(0.0, float(tau)))
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


def _vec3(v, what: str) -> np.ndarray:
    """A finite 3-vector from user input, else ValueError. A NaN or inf that reached the IK
    would end up in data.ctrl, and MuJoCo then zeroes *every* actuator (legs included)."""
    a = np.asarray(v, float).reshape(3)
    if not np.isfinite(a).all():
        raise ValueError(f"{what} must be three finite numbers")
    return a


def _unit(v) -> np.ndarray | None:
    v = np.asarray(v, float).reshape(3)
    s = float(np.abs(v).max())
    if not np.isfinite(s) or s <= 1e-9:
        return None
    v = v / s                    # pre-scale: a huge component must not overflow the norm to inf
    return v / float(np.linalg.norm(v))


def _nlerp(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    v = (1 - s) * a + s * b
    n = np.linalg.norm(v)
    if n < 1e-6:  # opposite directions: go round through any perpendicular
        perp = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        v = (1 - s) * a + s * b + 0.5 * np.sin(np.pi * s) * perp / np.linalg.norm(perp)
        n = np.linalg.norm(v)
    return v / n


# ---- trajectories -----------------------------------------------------------------
@dataclass
class _CartPath:
    """Piecewise min-jerk path through world waypoints. The start rides with the torso."""
    start_local: np.ndarray            # start point in the torso frame
    points: list[np.ndarray]           # world waypoints after the start
    durations: list[float]
    t0: float
    a0_local: np.ndarray | None = None  # initial pointing direction (torso frame)
    approach: np.ndarray | None = None  # pointing direction to reach during the first segment

    def at(self, t: float, torso_pos: np.ndarray, torso_R: np.ndarray):
        """-> (position, pointing direction or None, finished)."""
        pts = [torso_pos + torso_R @ self.start_local] + self.points
        el = t - self.t0
        a = self.approach
        for k, T in enumerate(self.durations):
            if el < T:
                s = min_jerk(el / T)
                if k == 0 and a is not None and self.a0_local is not None:
                    a = _nlerp(torso_R @ self.a0_local, a, s)
                return pts[k] + (pts[k + 1] - pts[k]) * s, a, False
            el -= T
        return pts[-1], a, True

    @property
    def total(self) -> float:
        return float(sum(self.durations))


@dataclass
class _JointPath:
    """Min-jerk joint-space keyframes; a target may be a callable re-evaluated every step."""
    q0: np.ndarray
    frames: list[tuple[np.ndarray | Callable[[], np.ndarray], float]]
    t0: float

    def at(self, t: float):
        prev, el = self.q0, t - self.t0
        for target, T in self.frames:
            q = target() if callable(target) else target
            if el < T:
                return prev + (q - prev) * min_jerk(el / T), False
            el -= T
            prev = q
        return prev, True


# ---- one arm ------------------------------------------------------------------------
@dataclass
class _Arm:
    side: str
    sign: float
    names: list[str]
    qadr: np.ndarray
    dadr: np.ndarray
    act: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    kp: np.ndarray                     # servo stiffness (0 = not a position servo: no feed-forward)
    body: int                          # effector body (hand, or forearm without hands)
    site: int                          # "<side>_hand" site or -1
    local: np.ndarray                  # effector point in body frame when there is no site
    shoulder_jnt: int
    kv: np.ndarray | None = None       # servo damping (velocity feed-forward kv / kp * qdot_des)
    hi_joint0: float = 0.0             # model's upper limit of shoulder pitch (before balance limits)
    reach_len: float = 0.0
    grip_act: int = -1
    grip_qadr: int = -1
    grip_open: float = 0.0
    pad_gap0: float = 0.0              # half the gap between the finger pads at zero opening
    fingers: tuple[int, int] = (-1, -1)
    welds: dict[int, int] = field(default_factory=dict)   # prop body -> weld equality id
    geoms: np.ndarray = field(default_factory=lambda: np.zeros(0, int))  # every geom from the upper arm down
    # ---- runtime ----
    engaged: bool = False
    skill: str = ""
    state: str = "idle"
    message: str = ""
    q_des: np.ndarray | None = None
    q_int: np.ndarray | None = None
    grip_cmd: float = 0.0
    path: _CartPath | _JointPath | None = None
    queue: list = field(default_factory=list)  # callables that start the next motion phase
    goal: np.ndarray | None = None      # world target the error is measured against
    approach: np.ndarray | None = None  # held pointing direction after the path
    hold_local: tuple | None = None     # (target, approach) in torso frame once the base walked away
    hold_xy: np.ndarray | None = None
    point_target: np.ndarray | None = None
    t_mark: float = 0.0
    t_contact: float | None = None
    t_lost: float | None = None
    holding: int | None = None          # body id
    weld_on: int = -1
    hand_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    error_m: float | None = None


class ArmController:
    """Skill-level control of both arms and hands (see module docstring for the method)."""
    name = "arms"

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData | None = None, grasp_assist: bool = False,
                 speed: float = 0.25, damping: float = 0.02, max_back_deg: float | None = 5.0,
                 base_filter_s: float = 0.0):
        """grasp_assist: weld a pinched prop to the hand (see module docstring).
        speed: average Cartesian reach speed (m/s). damping: DLS lambda (m).
        max_back_deg: never swing an upper arm further back than this (see "Balance" in the
        module docstring). None = no limit.
        base_filter_s: time constant of a low-pass filter on the torso pose that targets are
        resolved against. 0 (default) re-solves against the measured pose every step; a value
        of ~0.5 s makes the arm follow fast body sway rigidly and compensate only slow drift,
        which decouples the arm from a balance controller that moves the torso.
        """
        if not self.supports(model):
            raise ValueError("model has no <side>_shoulder_y / shoulder_x / elbow joints with actuators")
        self.model = model
        self.max_back = None if max_back_deg is None else np.deg2rad(float(max_back_deg))
        self.retract_back = np.deg2rad(20.0)
        self.grasp_assist = bool(grasp_assist)
        self.speed = float(speed)          # m/s average Cartesian speed of a reach
        self.damping = float(damping)      # DLS lambda (m)
        self.control_dt = float(model.opt.timestep) * 4
        self.reach_tol = 0.01              # IK residual above which a target counts as out of reach (m)
        self.approach_tol = np.deg2rad(15)
        self.done_tol = 0.005
        self.fail_tol = 0.03
        self.ki = 4.0                      # joint integral gain (1/s)
        self.max_joint_speed = 4.0         # rad/s
        self.w_rot = 0.15                  # m per rad: weight of the pointing task against position
        self._scratch = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._data: mujoco.MjData | None = None
        self._t_last: float | None = None
        from ..sim.props import prop_names
        self._prop_names = prop_names(model)
        self.arms: dict[str, _Arm] = {}
        for side, sign in (("left", 1.0), ("right", -1.0)):
            arm = self._make_arm(model, side, sign)
            if arm is not None:
                self.arms[side] = arm
        self.root = int(model.body_rootid[model.jnt_bodyid[next(iter(self.arms.values())).shoulder_jnt]])
        jr = int(model.body_jntadr[self.root])
        self._root_qadr = (int(model.jnt_qposadr[jr]) if jr >= 0 and model.jnt_type[jr] == mujoco.mjtJoint.mjJNT_FREE
                           else None)
        # world geometry an arm can hit (tables, obstacles, props, floor): for the sweep check of a reach
        arm_geoms = np.concatenate([a.geoms for a in self.arms.values()])
        arm_type = int(np.bitwise_or.reduce(model.geom_contype[arm_geoms])) if len(arm_geoms) else 0
        arm_aff = int(np.bitwise_or.reduce(model.geom_conaffinity[arm_geoms])) if len(arm_geoms) else 0
        self._world_geoms = np.array([g for g in range(model.ngeom)
                                      if model.body_rootid[model.geom_bodyid[g]] != self.root
                                      and (model.geom_conaffinity[g] & arm_type or model.geom_contype[g] & arm_aff)], int)
        self.sweep_clearance = 0.02        # m: predicted arm-to-world gap below which a straight reach is "blocked"
        self.base_filter_s = float(base_filter_s)
        self._fp: np.ndarray | None = None   # low-pass filtered torso position / orientation
        self._fq: np.ndarray | None = None
        if data is not None:
            self.reset(model, data)

    # ---- model introspection -------------------------------------------------------
    @staticmethod
    def _joint_actuator(model: mujoco.MjModel, jid: int) -> int:
        for i in range(model.nu):
            if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_trnid[i, 0] == jid:
                return i
        return -1

    @classmethod
    def supports(cls, model: mujoco.MjModel) -> bool:
        for side in SIDES:
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{j}") for j in ARM_JOINTS[:3]]
            if all(i >= 0 and cls._joint_actuator(model, i) >= 0 for i in ids):
                return True
        return False

    def _make_arm(self, m: mujoco.MjModel, side: str, sign: float) -> _Arm | None:
        names, jids, acts = [], [], []
        for j in ARM_JOINTS:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{j}")
            a = self._joint_actuator(m, jid) if jid >= 0 else -1
            if a >= 0:
                names.append(j), jids.append(jid), acts.append(a)
        if names[:3] != list(ARM_JOINTS[:3]):
            return None
        jids_a, acts_a = np.array(jids), np.array(acts)
        servo = m.actuator_biastype[acts_a] == mujoco.mjtBias.mjBIAS_AFFINE
        kp = np.where(servo, m.actuator_gainprm[acts_a, 0], 0.0)
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_hand")
        if site >= 0:
            body, local = int(m.site_bodyid[site]), np.zeros(3)
        else:  # no hand: use the far end of the forearm capsule
            body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_forearm")
            g = int(m.body_geomadr[body])
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            axis = R.reshape(3, 3)[:, 2] * m.geom_size[g, 1]
            ends = [m.geom_pos[g] + axis, m.geom_pos[g] - axis]
            local = max(ends, key=np.linalg.norm)
        arm = _Arm(side=side, sign=sign, names=names, qadr=m.jnt_qposadr[jids_a].copy(), dadr=m.jnt_dofadr[jids_a].copy(),
                   act=acts_a, lo=m.jnt_range[jids_a, 0].copy(), hi=m.jnt_range[jids_a, 1].copy(), kp=kp,
                   body=body, site=site, local=np.asarray(local, float), shoulder_jnt=int(jids_a[0]))
        arm.kv = np.where(servo, -m.actuator_biasprm[acts_a, 2], 0.0)
        arm.hi_joint0 = float(arm.hi[0])
        if self.max_back is not None:   # shoulder pitch: positive = arm swings backwards (axis +y, z down)
            arm.hi[0] = max(arm.lo[0], min(arm.hi[0], self.max_back))
        gj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_gripper")
        if gj >= 0 and self._joint_actuator(m, gj) >= 0:
            arm.grip_act, arm.grip_qadr = self._joint_actuator(m, gj), int(m.jnt_qposadr[gj])
            arm.grip_open = float(m.jnt_range[gj, 1])
            arm.fingers = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_finger1"),
                           mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_finger2"))
            if arm.fingers[0] >= 0:
                f = arm.fingers[0]
                arm.pad_gap0 = float(abs(m.geom_pos[f, 1]) - m.geom_size[f, 1])
        for e in range(m.neq):
            n = m.equality(e).name
            if m.eq_type[e] == mujoco.mjtEq.mjEQ_WELD and n.startswith(f"{side}_hold_"):
                b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "prop_" + n.rsplit("_", 1)[1])
                if b >= 0 and int(m.eq_obj1id[e]) == body:
                    arm.welds[b] = e
        upper = int(m.jnt_bodyid[arm.shoulder_jnt])
        in_arm = np.zeros(m.nbody, bool)
        for b in range(m.nbody):     # bodies are numbered parents first: one pass marks the subtree
            in_arm[b] = b == upper or (b > 0 and in_arm[m.body_parentid[b]])
        arm.geoms = np.flatnonzero(in_arm[m.geom_bodyid])
        # full arm length: shoulder joint to effector with the arm straight down (qpos0)
        sd = self._scratch
        mujoco.mj_resetData(m, sd)
        mujoco.mj_kinematics(m, sd)
        arm.reach_len = float(np.linalg.norm(self._eff_pos(arm, sd) - sd.xanchor[arm.shoulder_jnt]))
        return arm

    # ---- kinematics ----------------------------------------------------------------
    def _eff_pos(self, arm: _Arm, d: mujoco.MjData) -> np.ndarray:
        if arm.site >= 0:
            return d.site_xpos[arm.site].copy()
        return d.xpos[arm.body] + d.xmat[arm.body].reshape(3, 3) @ arm.local

    def _eff_axis(self, arm: _Arm, d: mujoco.MjData) -> np.ndarray:
        """Direction the fingers point (hand -z)."""
        R = d.site_xmat[arm.site] if arm.site >= 0 else d.xmat[arm.body]
        return -R.reshape(3, 3)[:, 2].copy()

    def _fk(self, arm: _Arm, q: np.ndarray):
        """Scratch-data forward kinematics with the arm at q (the rest of the robot as last loaded)."""
        m, sd = self.model, self._scratch
        sd.qpos[arm.qadr] = q
        mujoco.mj_kinematics(m, sd)
        mujoco.mj_comPos(m, sd)
        p = self._eff_pos(arm, sd)
        mujoco.mj_jac(m, sd, self._jacp, self._jacr, p, arm.body)
        return p, self._eff_axis(arm, sd), self._jacp[:, arm.dadr], self._jacr[:, arm.dadr]

    def _update_base(self, d: mujoco.MjData, dt: float) -> None:
        """First-order low-pass of the torso pose the IK solves against (base_filter_s = 0: raw)."""
        p, q = d.xpos[self.root], d.xquat[self.root]
        if self._fp is None or self.base_filter_s <= 0:
            self._fp, self._fq = p.copy(), q.copy()
            return
        a = min(1.0, dt / self.base_filter_s)
        self._fp += a * (p - self._fp)
        qq = q if float(q @ self._fq) >= 0 else -q
        self._fq = self._fq + a * (qq - self._fq)
        self._fq /= np.linalg.norm(self._fq)

    def _base(self, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """(position, rotation matrix) of the torso frame targets are resolved in."""
        if self._fp is None:
            self._update_base(d, 0.0)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self._fq)
        return self._fp, R.reshape(3, 3)

    def _load_base(self, d: mujoco.MjData) -> None:
        self._scratch.qpos[:] = d.qpos
        if self._root_qadr is not None and self._fp is not None:
            a = self._root_qadr
            self._scratch.qpos[a:a + 3] = self._fp
            self._scratch.qpos[a + 3:a + 7] = self._fq

    def _ik(self, arm: _Arm, q: np.ndarray, p_goal: np.ndarray, a_goal: np.ndarray | None,
            iters: int, rest_gain: float = 0.05):
        """DLS iterations from q. Returns (q, position error m, pointing error rad) of the returned q."""
        n = len(q)
        q = np.clip(q, arm.lo, arm.hi)
        lam2 = self.damping ** 2
        wrist = np.array([nm.startswith("wrist") for nm in arm.names], float)
        best, stall = np.inf, 0
        for it in range(iters + 1):
            p, a, Jp, Jr = self._fk(arm, q)
            e = p_goal - p
            ang = 0.0
            J, err = Jp, e
            if a_goal is not None:
                c = np.cross(a, a_goal)
                s, cs = float(np.linalg.norm(c)), float(a @ a_goal)
                ang = float(np.arctan2(s, cs))
                if s > 1e-9:
                    eo = c / s * ang
                else:
                    eo = np.zeros(3) if cs > 0 else np.cross(a, [1.0, 0, 0]) * np.pi
                P = np.eye(3) - np.outer(a, a)
                J = np.vstack([Jp, self.w_rot * (P @ Jr)])
                err = np.concatenate([e, self.w_rot * eo])
            score = float(np.linalg.norm(e)) + self.w_rot * ang
            best, stall = (score, 0) if score < best - 1e-6 else (best, stall + 1)
            # converged, out of iterations, or stuck (an unreachable target: no progress in 12 steps)
            if it == iters or (np.linalg.norm(e) < 1e-4 and ang < 1e-3) or stall >= 12:
                return q, float(np.linalg.norm(e)), ang
            # clamping loop: drop joints that sit on a limit and would be pushed beyond it, and
            # re-solve until no free joint is pushed out (each pass frees fewer joints, so it ends
            # within n passes). Jinv and Jf always belong to the final free set, which the
            # null-space projector below needs (a fixed pass count left them one set behind).
            free = np.ones(n, bool)
            at_lo, at_hi = q <= arm.lo + 1e-6, q >= arm.hi - 1e-6
            while True:
                Jf = J[:, free]
                A = Jf @ Jf.T + lam2 * np.eye(J.shape[0])
                Jinv = Jf.T @ np.linalg.solve(A, np.eye(J.shape[0]))
                dq = np.zeros(n)
                dq[free] = Jinv @ err
                push = free & ((at_lo & (dq < 0)) | (at_hi & (dq > 0)))
                if not push.any():
                    break
                free &= ~push
            if a_goal is None and rest_gain > 0:   # redundant: keep the wrist straight in the null space
                N = np.eye(int(free.sum())) - Jinv @ Jf
                dq[free] += N @ (rest_gain * wrist[free] * -q[free])
            step = np.abs(dq).max()
            if step > 0.3:
                dq *= 0.3 / step
            q = np.clip(q + dq, arm.lo, arm.hi)
        return q, float(np.linalg.norm(e)), ang

    def _solve_static(self, arm: _Arm, target: np.ndarray, a_goal: np.ndarray | None,
                      seeds: list[np.ndarray] | None = None, iters: int = 120):
        """Full IK solve from several seeds against the current base pose (reachability check)."""
        d = self._data
        self._load_base(d)
        if seeds is None:
            fwd = {"shoulder_y": -0.9, "shoulder_x": 0.1 * arm.sign, "elbow": -1.0, "wrist_y": 1.0}
            seeds = [arm.q_des if arm.q_des is not None else d.qpos[arm.qadr].copy(),
                     np.zeros(len(arm.names)), self._pose(arm, fwd)]
        best = None
        for s in seeds:
            q, e, ang = self._ik(arm, np.array(s, float), target, a_goal, iters)
            score = e + self.w_rot * ang
            if best is None or score < best[3]:
                best = (q, e, ang, score)
            if e < 1e-3 and ang < 0.01:
                break
        self._load_base(d)
        return best[0], best[1], best[2]

    def _pose(self, arm: _Arm, values: dict[str, float]) -> np.ndarray:
        """Named joint values (left-side convention; roll joints mirrored on the right) -> clipped q."""
        q = np.array([values.get(nm, 0.0) * (arm.sign if nm in ROLL_JOINTS else 1.0) for nm in arm.names])
        return np.clip(q, arm.lo, arm.hi)

    # ---- controller protocol ---------------------------------------------------------
    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._data = data
        self._fp = self._fq = None
        self._update_base(data, 0.0)
        self._t_last = None
        for arm in self.arms.values():
            if arm.weld_on >= 0:
                data.eq_active[arm.weld_on] = 0
            arm.engaged, arm.skill, arm.state, arm.message = False, "", "idle", ""
            arm.path = arm.goal = arm.approach = arm.hold_local = arm.hold_xy = arm.point_target = None
            arm.queue.clear()
            arm.q_des = arm.q_int = None
            arm.holding, arm.weld_on, arm.t_contact, arm.t_lost = None, -1, None, None
            arm.hand_xyz, arm.error_m = self._eff_pos(arm, data), None

    def set_path(self, path_xy, final_yaw=None) -> None:  # arms do not walk
        pass

    @property
    def busy(self) -> bool:
        """True while any arm is executing a motion or closing/opening its gripper."""
        return any(a.state == "moving" for a in self.arms.values())

    def step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Write the engaged arms' actuator targets. Call after the base controller's step()."""
        t = float(data.time)
        if self._data is not data or (self._t_last is not None and t < self._t_last - 1e-9):
            self.reset(model, data)        # new data or the simulation was reset under us
        dt = self.control_dt if self._t_last is None else min(0.1, max(0.0, t - self._t_last))
        self._t_last = t
        self._update_base(data, dt)
        contacts = None
        for arm in self.arms.values():
            arm.hand_xyz = self._eff_pos(arm, data)
            if not arm.engaged:
                continue
            if arm.grip_act >= 0:
                if contacts is None:
                    contacts = self._finger_contacts(data)
                self._update_grip(arm, data, t, contacts)
            self._track(arm, data, t, dt)

    def status(self) -> dict:
        """Per-arm skill state (see RobotInterface.arm_status). Each field is read once, so a
        caller that polls without the sim lock (robot_adapter.wait does) sees a stale but
        consistent value instead of hitting a field that step() cleared in between."""
        out = {}
        data = self._data
        for side in SIDES:
            arm = self.arms.get(side)
            if arm is None:
                out[side] = {"skill": "", "state": "failed", "hand_xyz": None, "error_m": None, "holding": None,
                             "message": "this robot has no such arm"}
                continue
            held, err, hand = arm.holding, arm.error_m, arm.hand_xyz
            out[side] = {
                "skill": arm.skill, "state": arm.state, "hand_xyz": [round(float(v), 4) for v in hand],
                "error_m": None if err is None else round(float(err), 4),
                "holding": None if held is None else self._prop_names.get(held, self.model.body(held).name),
                "gripper_opening_m": round(float(2 * (data.qpos[arm.grip_qadr] + arm.pad_gap0)), 4)
                if (arm.grip_qadr >= 0 and data is not None) else None,
                "message": arm.message,
            }
        return out

    # ---- commands ----------------------------------------------------------------------
    def command(self, side: str, skill: str, **kw) -> str:
        """Start a skill on "left", "right" (or "both" for anything but reach). Returns a short message.

        reach     target=[x, y, z] (world, palm/grasp point), optional approach=[dx, dy, dz]
                  (direction the fingers move in, e.g. [0, 0, -1] = from above; opens the gripper
                  unless holding), duration=s, open_gripper=bool
        grasp     close the gripper; "holding" once both fingers touch the same object
        release   open the gripper
        point_at  target=[x, y, z]
        wave | raise | home    gestures;  stop: freeze where it is;  relax: hand the arm back
        """
        skill = ALIASES.get(str(skill).lower().strip(), str(skill).lower().strip())
        if skill not in SKILLS:
            return f"unknown arm skill '{skill}' (known: {', '.join(SKILLS)})"
        side = str(side).lower().strip()
        sides = list(self.arms) if side == "both" else [side]
        if side == "both" and skill == "reach":
            return "reach needs one hand: 'left' or 'right'"
        if any(s not in self.arms for s in sides):
            return f"no arm called '{side}' (this robot has: {', '.join(self.arms)})"
        if self._data is None:
            return "arm controller is not attached to a simulation yet (call reset(model, data))"
        target = kw.get("target")
        if target is None and all(k in kw for k in ("x", "y", "z")):
            target = [kw["x"], kw["y"], kw["z"]]
        msgs = []
        for s in sides:
            arm = self.arms[s]
            self._engage(arm)
            if skill not in ("grasp", "release"):   # a new arm motion replaces any queued phases
                arm.queue.clear()
            try:
                if skill == "reach":
                    msg = self._cmd_reach(arm, target, kw.get("approach"), kw.get("duration"), kw.get("open_gripper"))
                elif skill == "point_at":
                    msg = self._cmd_point(arm, target)
                elif skill == "grasp":
                    msg = self._cmd_grasp(arm)
                elif skill == "release":
                    msg = self._cmd_release(arm)
                elif skill in ("wave", "raise", "home"):
                    msg = self._cmd_gesture(arm, skill)
                elif skill == "stop":
                    arm.path, arm.goal, arm.skill, arm.state = None, None, "stop", "holding" if arm.holding else "done"
                    arm.q_des = self._data.qpos[arm.qadr].copy()
                    msg = f"{s} arm stopped"
                else:  # relax
                    if arm.holding is not None:
                        msg = f"{s} arm is holding {self._held_name(arm)}: release it before relaxing"
                    else:
                        arm.engaged, arm.path, arm.skill, arm.state = False, None, "relax", "idle"
                        msg = f"{s} arm handed back to the body controller"
            except (TypeError, ValueError) as e:
                arm.state, msg = "failed", f"{s} {skill}: bad arguments ({e})"
            arm.message = msg
            msgs.append(msg)
        return "; ".join(msgs)

    def _engage(self, arm: _Arm) -> None:
        if arm.engaged:
            return
        d = self._data
        arm.engaged = True
        arm.q_des = np.clip(d.qpos[arm.qadr].copy(), arm.lo, arm.hi)
        arm.q_int = np.zeros(len(arm.names))
        arm.grip_cmd = float(d.qpos[arm.grip_qadr]) if arm.grip_qadr >= 0 else 0.0

    def _start(self, arm: _Arm, skill: str, path) -> None:
        arm.skill, arm.state, arm.path = skill, "moving", path
        arm.hold_local = arm.hold_xy = None
        arm.t_mark = float(self._data.time)
        arm.q_int = np.zeros(len(arm.names)) if arm.q_int is None else arm.q_int

    def _cmd_reach(self, arm: _Arm, target, approach, duration, open_gripper) -> str:
        d, s = self._data, arm.side
        if target is None:
            raise ValueError("reach needs target=[x, y, z]")
        goal = _vec3(target, "target")
        if duration is not None:
            duration = float(duration)
            if not (np.isfinite(duration) and duration > 0):
                raise ValueError("duration must be a positive number of seconds")
        shoulder = d.xanchor[arm.shoulder_jnt].copy()
        dist_sh = float(np.linalg.norm(goal - shoulder))
        a = None
        if approach is not None:
            a = _unit(_vec3(approach, "approach"))
            if a is None:
                raise ValueError("approach direction must be non-zero")
        p_now = self._eff_pos(arm, d)
        # 1) reachability: full IK solve against the current torso pose
        if a is None:   # keep the hand's current pointing direction if the arm can, else leave it free
            # (position has priority: keeping the orientation must not cost more than 2 mm)
            keep = self._eff_axis(arm, d)
            q_goal, err, ang = self._solve_static(arm, goal, keep)
            if err < 0.002 and ang < self.approach_tol:
                a_use = keep
            else:
                q_goal, err, ang = self._solve_static(arm, goal, None)
                a_use = None
        else:
            q_goal, err, ang = self._solve_static(arm, goal, a)
            a_use = a
        if err > self.reach_tol:
            arm.skill, arm.state, arm.path, arm.goal, arm.point_target = "reach", "failed", None, None, None
            arm.error_m = float(np.linalg.norm(goal - p_now))
            where = f"({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f})"
            if dist_sh > arm.reach_len:
                why = (f"it is {dist_sh:.2f} m from the {s} shoulder and the arm reaches {arm.reach_len:.2f} m "
                       f"({dist_sh - arm.reach_len:.2f} m short); walk closer")
            else:
                why = (f"joint limits keep the hand {err:.2f} m away (no shoulder yaw: each hand works on its own "
                       f"side and in front); try the other hand or move the body")
            return f"{s} hand cannot reach {where}: out of reach, {why}"
        if a is not None and ang > self.approach_tol:
            arm.skill, arm.state, arm.path, arm.goal, arm.point_target = "reach", "failed", None, None, None
            return (f"{s} hand can reach ({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}) but not with the fingers "
                    f"pointing along {np.round(a, 2).tolist()} (best is {np.degrees(ang):.0f} deg off); "
                    f"try another approach direction or none")
        # 2) waypoints: lift first if starting low, pre-grasp point back along the approach, then the goal
        pts: list[np.ndarray] = []
        if a is not None:
            for back in (0.08, 0.05, 0.03):
                pre = goal - a * back
                _, e_pre, ang_pre = self._solve_static(arm, pre, a, seeds=[q_goal], iters=60)
                if e_pre < self.reach_tol and ang_pre < self.approach_tol:
                    pts.append(pre)
                    break
        has_pre = bool(pts)
        first = pts[0] if pts else goal
        tp, tR = self._base(d)
        tuck_first = False
        if self._blocked(arm, p_now, first) or self._sweep_hits(arm, p_now, first, a_use):
            # Something (a table edge, say) is in the way of the straight move. From beside the
            # body, fold into the tuck pose (hand up at the chest, behind the edge) and approach
            # from there; from further out, rise early via the nearest reachable clear point at
            # the first waypoint's height.
            if (tR.T @ (p_now - shoulder))[0] < 0.15:
                tuck_first = True
            else:
                for frac in (0.3, 0.45, 0.6):
                    via = p_now + frac * (first - p_now)
                    via[2] = max(first[2], p_now[2]) + 0.02
                    _, e_via, _ = self._solve_static(arm, via, None, iters=80)
                    if e_via < self.reach_tol and not self._blocked(arm, via, first):
                        pts.insert(0, via)
                        break
                else:
                    tuck_first = True
        pts.append(goal)

        def cart_phase() -> float:
            dd = self._data
            p0 = self._eff_pos(arm, dd)
            durs = []
            for k, (a0, a1) in enumerate(zip([p0] + pts[:-1], pts)):
                L = float(np.linalg.norm(a1 - a0))
                final_approach = has_pre and k == len(pts) - 1
                durs.append(max(0.8, L / 0.1) if final_approach else float(np.clip(L / self.speed, 0.5, 3.0)))
            if duration is not None and not tuck_first:
                durs = list(np.array(durs) * max(0.2, float(duration)) / sum(durs))
            bp, bR = self._base(dd)
            path = _CartPath(start_local=bR.T @ (p0 - bp), points=pts, durations=durs, t0=float(dd.time),
                             a0_local=bR.T @ self._eff_axis(arm, dd) if a_use is not None else None, approach=a_use)
            self._start(arm, "reach", path)
            arm.goal, arm.approach, arm.point_target = goal, a_use, None
            return path.total

        if tuck_first:
            q0, tuck = arm.q_des.copy(), self._tuck_pose(arm)
            T = float(np.clip(np.abs(tuck - q0).max() / 1.2, 0.8, 2.5))
            self._start(arm, "reach", _JointPath(q0, [(tuck, T)], float(d.time)))
            arm.goal, arm.approach, arm.point_target = goal, None, None
            arm.queue.append(cart_phase)
            total = T + float(np.linalg.norm(goal - p_now)) / self.speed
        else:
            total = cart_phase()
        if open_gripper is None:
            open_gripper = a is not None and arm.holding is None
        if open_gripper and arm.grip_act >= 0 and arm.holding is None:
            arm.grip_cmd = arm.grip_open
        via = " via the tuck pose" if tuck_first else (f" via {len(pts) - 1} waypoint(s)" if len(pts) > 1 else "")
        return (f"{s} hand reaching ({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f}){via}, "
                f"{float(np.linalg.norm(goal - p_now)):.2f} m away, ~{total:.1f} s")

    def _blocked(self, arm: _Arm, a: np.ndarray, b: np.ndarray) -> bool:
        """True if the straight move a -> b of the hand would hit world geometry (obstacles,
        tables, props, floor: geom groups 1-2), tested with rays from the grasp point and from
        5 cm below it (the fingertips and anything held)."""
        d, v = self._data, np.asarray(b, float) - np.asarray(a, float)
        if np.linalg.norm(v) < 1e-6:
            return False
        groups = np.array([0, 1, 1, 0, 0, 0], np.uint8)
        gid = np.zeros(1, np.int32)
        skip = arm.holding if arm.holding is not None else -1
        # moving (mostly) upwards the fingertips trail the hand; their ray would start inside
        # the surface the hand is leaving
        for dz in ((0.0,) if v[2] > 0.7 * np.linalg.norm(v) else (0.0, -0.05)):
            x = mujoco.mj_ray(self.model, d, np.asarray(a, float) + [0.0, 0.0, dz], v, groups, 1, skip, gid)
            if 0.0 <= x <= 1.0:
                return True
        return False

    def _sweep_hits(self, arm: _Arm, a: np.ndarray, b: np.ndarray, a_goal: np.ndarray | None) -> bool:
        """True if the arm itself (upper arm, forearm, hand, fingers) would pass within
        ``sweep_clearance`` of world geometry while the hand tracks the straight line a -> b.

        The rays of ``_blocked`` assume the hand follows that line, but the arm cannot always:
        from the hanging pose a point low in front needs the shoulder further back than the
        balance limit allows (about 17 deg for the default body, limit 5 deg), so the tracked
        hand sags up to ~7 cm below the line and the forearm can clip a table edge and knock
        the cube off its spot (measured before this check: 2 of 72 legs-locked picks, 2 of 34
        with the walker). This approximates the tracking with the same warm-started DLS as
        ``_track`` (5 iterations per 1.5 cm of line, on the scratch model) and measures signed
        distances with ``mj_geomDistance``. From the straight, limit-pinned hanging arm the
        dry run sags further than the live arm does, so it errs on the safe side (the reach
        then folds through the tuck pose). Pairs that start close (a hand leaving a surface
        it rests on) only count if they get closer still.
        """
        m, d, sd = self.model, self._data, self._scratch
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        if L < 1e-3 or not len(self._world_geoms) or not len(arm.geoms):
            return False
        sh = d.xanchor[arm.shoulder_jnt]
        reach = arm.reach_len + 0.1
        near = [int(w) for w in self._world_geoms
                if (arm.holding is None or m.geom_bodyid[w] != arm.holding)
                and (m.geom_rbound[w] <= 0 or np.linalg.norm(d.geom_xpos[w] - sh) - m.geom_rbound[w] < reach)]
        if not near:
            return False
        tol, dmax = self.sweep_clearance, self.sweep_clearance + 0.05
        a0 = self._eff_axis(arm, d)
        q = arm.q_des.copy() if arm.q_des is not None else d.qpos[arm.qadr].copy()
        n = max(2, int(np.ceil(L / 0.015)))
        self._load_base(d)
        start, hit = None, False
        for k in range(n + 1):
            s = k / n
            if k == 0:
                self._fk(arm, np.clip(q, arm.lo, arm.hi))
            else:
                aim = None if a_goal is None else _nlerp(a0, a_goal, s)
                q = self._ik(arm, q, a + (b - a) * s, aim, iters=5)[0]   # leaves sd at the returned q
            dist = np.array([[mujoco.mj_geomDistance(m, sd, int(g), w, dmax, None) for w in near] for g in arm.geoms])
            if start is None:
                start = dist
            elif ((dist < tol) & (dist < start - 0.005)).any():
                hit = True
                break
        self._load_base(d)
        return hit

    def _point_goal(self, arm: _Arm, target: np.ndarray):
        """Hand position on the shoulder->target ray (95 % of arm length) with the fingers along it."""
        sh = self._data.xanchor[arm.shoulder_jnt]
        u = _unit(target - sh)
        if u is None:
            u = np.array([1.0, 0, 0])
        return sh + 0.95 * arm.reach_len * u, u

    def _cmd_point(self, arm: _Arm, target) -> str:
        if target is None:
            raise ValueError("point_at needs target=[x, y, z]")
        tgt = _vec3(target, "target")
        p, u = self._point_goal(arm, tgt)
        q_goal, err, ang = self._solve_static(arm, p, u)
        if ang > np.deg2rad(30) or err > 0.15:
            d = self._data
            tR = d.xmat[self.root].reshape(3, 3)
            local = tR.T @ u
            bearing = float(np.degrees(np.arctan2(local[1], local[0])))
            arm.skill, arm.state, arm.path, arm.error_m = "point_at", "failed", None, None
            arm.goal = arm.point_target = None
            return (f"the {arm.side} arm cannot point at ({tgt[0]:.2f}, {tgt[1]:.2f}, {tgt[2]:.2f}): it is at bearing "
                    f"{bearing:.0f} deg (0 = straight ahead, + = left) and the best the arm can do is "
                    f"{np.degrees(ang):.0f} deg off; turn towards it first"
                    + ("" if (bearing > 0) == (arm.sign > 0) else " or use the other arm"))

        def goal_q() -> np.ndarray:  # re-solved every step: follows the body and the shoulder
            self._load_base(self._data)
            pp, uu = self._point_goal(arm, tgt)
            arm._q_point = self._ik(arm, getattr(arm, "_q_point", q_goal), pp, uu, iters=4)[0]
            return arm._q_point

        arm._q_point = q_goal
        T = float(np.clip(np.abs(q_goal - arm.q_des).max() / 1.2, 0.8, 2.5))
        self._start(arm, "point_at", _JointPath(arm.q_des.copy(), [(goal_q, T)], float(self._data.time)))
        arm.goal, arm.approach, arm.point_target = None, None, tgt
        note = "" if ang < np.deg2rad(10) and err < 0.05 else f" (the arm can only point within {np.degrees(ang):.0f} deg of it)"
        return f"{arm.side} arm pointing at ({tgt[0]:.2f}, {tgt[1]:.2f}, {tgt[2]:.2f}){note}"

    def _gesture_frames(self, arm: _Arm, skill: str, q0: np.ndarray) -> list:
        home = self._pose(arm, {})
        if skill == "home":
            return [(home, float(np.clip(np.abs(home - q0).max() / 1.2, 0.8, 2.5)))]
        if skill == "raise":
            up = self._pose(arm, {"shoulder_y": -2.05, "shoulder_x": 0.2, "elbow": -0.1})
            return [(up, float(np.clip(np.abs(up - q0).max() / 1.5, 0.8, 2.5)))]
        # wave: hand up in front of the shoulder, swing side to side three times, back home
        up = {"shoulder_y": -1.9, "shoulder_x": 0.35, "elbow": -0.9}
        a = self._pose(arm, {**up, "shoulder_x": 0.6, "wrist_x": 0.45})
        b = self._pose(arm, {**up, "shoulder_x": 0.15, "wrist_x": -0.45})
        upq = self._pose(arm, up)
        frames = [(upq, float(np.clip(np.abs(upq - q0).max() / 1.5, 0.8, 2.0)))]
        return frames + [(a, 0.35), (b, 0.35)] * 3 + [(upq, 0.3), (home, 1.2)]

    def _tuck_pose(self, arm: _Arm) -> np.ndarray:
        """Elbow fully folded, shoulder slightly back, wrist bent so the fingers point down.

        Interpolating joint-wise from this pose to the hanging pose keeps the hand pointing
        down (wrist = -(shoulder + elbow) all the way) and close to the chest, so the arm
        drops behind a table edge instead of sweeping through it. Folded, the arm's mass
        stays near the shoulder, so the extra backward shoulder angle (retract_back) does
        not tip the robot the way a straight arm held back does.
        """
        back = self.retract_back if self.max_back is None else max(self.max_back, self.retract_back)
        values = {"shoulder_y": min(back, arm.hi_joint0)}
        if "elbow" in arm.names:
            values["elbow"] = float(arm.lo[arm.names.index("elbow")])
        if "wrist_y" in arm.names:
            values["wrist_y"] = -(values["shoulder_y"] + values.get("elbow", 0.0))
        q = np.array([values.get(nm, 0.0) for nm in arm.names])
        hi = arm.hi.copy()
        hi[0] = max(hi[0], values["shoulder_y"])
        return np.clip(q, arm.lo, hi)

    def _cmd_gesture(self, arm: _Arm, skill: str) -> str:
        """Joint-space gesture. A hand that is out in front (over a table, say) first lifts
        6 cm if it is low, then folds into the tuck pose, and only then moves to the gesture,
        so the swing down passes behind a table edge instead of through it."""
        d = self._data
        tp, tR = self._base(d)
        p = self._eff_pos(arm, d)
        local = tR.T @ (p - d.xanchor[arm.shoulder_jnt])
        arm.goal = arm.approach = arm.point_target = arm.error_m = None

        def joint_phase() -> None:
            q0 = arm.q_des.copy()
            self._start(arm, skill, _JointPath(q0, self._gesture_frames(arm, skill, q0), float(self._data.time)))
            arm.goal = None

        def tuck_phase() -> None:
            q0, tuck = arm.q_des.copy(), self._tuck_pose(arm)
            T = float(np.clip(np.abs(tuck - q0).max() / 1.2, 0.8, 2.5))
            self._start(arm, skill, _JointPath(q0, [(tuck, T)], float(self._data.time)))
            arm.goal = None
            arm.queue.append(joint_phase)

        if local[0] > 0.15:
            if local[2] < -0.2:   # low and in front: probably at a table, lift straight up first
                self._start(arm, skill, _CartPath(tR.T @ (p - tp), [p + np.array([0.0, 0.0, 0.06])], [0.6],
                                                  float(d.time)))
                arm.queue.append(tuck_phase)
            else:
                tuck_phase()
            return f"{arm.side} arm: {skill} (folding in first to clear what is in front)"
        joint_phase()
        return f"{arm.side} arm: {skill} (~{sum(T for _, T in arm.path.frames):.1f} s)"

    def _cmd_grasp(self, arm: _Arm) -> str:
        if arm.grip_act < 0:
            arm.state = "failed"
            return f"the {arm.side} arm has no gripper"
        if arm.holding is not None:
            return f"{arm.side} hand is already holding {self._held_name(arm)}"
        arm.skill, arm.state, arm.t_mark, arm.t_contact = "grasp", "moving", float(self._data.time), None
        arm.grip_cmd = 0.0
        return f"{arm.side} hand closing"

    def _cmd_release(self, arm: _Arm) -> str:
        if arm.grip_act < 0:
            arm.state = "failed"
            return f"the {arm.side} arm has no gripper"
        held = self._held_name(arm) if arm.holding is not None else None
        self._drop(arm)
        arm.skill, arm.state, arm.t_mark = "release", "moving", float(self._data.time)
        arm.grip_cmd = arm.grip_open
        return f"{arm.side} hand opening" + (f", releasing {held}" if held else "")

    # ---- per-step tracking -------------------------------------------------------------
    def _track(self, arm: _Arm, d: mujoco.MjData, t: float, dt: float) -> None:
        tp, tR = self._base(d)
        q = arm.q_des
        finished = False
        if isinstance(arm.path, _CartPath):
            p_ref, a_ref, finished = arm.path.at(t, tp, tR)
            self._load_base(d)
            q = self._ik(arm, q, p_ref, a_ref, iters=3)[0]
            if finished:
                arm.hold_xy, arm.hold_local = tp[:2].copy(), None
        elif isinstance(arm.path, _JointPath):
            q, finished = arm.path.at(t)
        elif arm.goal is not None:      # holding a Cartesian target after a reach
            # World-fixed while the robot stands; once the base has walked 10 cm away the
            # target (and pointing direction) is carried along in the torso frame instead.
            if arm.hold_local is None and arm.hold_xy is not None and np.linalg.norm(tp[:2] - arm.hold_xy) > 0.1:
                arm.hold_local = (tR.T @ (arm.goal - tp),
                                  None if arm.approach is None else tR.T @ arm.approach)
            if arm.hold_local is not None:
                arm.goal = tp + tR @ arm.hold_local[0]
                arm.approach = None if arm.hold_local[1] is None else tR @ arm.hold_local[1]
            self._load_base(d)
            q = self._ik(arm, q, arm.goal, arm.approach, iters=2)[0]
        elif arm.point_target is not None and arm.skill == "point_at":
            self._load_base(d)
            pp, uu = self._point_goal(arm, arm.point_target)
            q = self._ik(arm, q, pp, uu, iters=2)[0]
        if not np.isfinite(q).all():    # a failed solve must never reach the motors: hold still
            q = arm.q_des
        # rate limit: a re-solved target may jump (e.g. an unreachable pointing direction);
        # never ask a joint to move faster than max_joint_speed. dt = 0 (stepped twice at the
        # same sim time) moves nothing and feeds forward no velocity.
        step = self.max_joint_speed * dt
        q = arm.q_des + np.clip(q - arm.q_des, -step, step)
        qd_des = (q - arm.q_des) / dt if dt > 0 else np.zeros(len(q))
        arm.q_des = q
        if finished:
            arm.path, arm.t_mark = None, t
            if arm.queue:
                arm.queue.pop(0)()
        # servo targets: q_des + gravity feed-forward + bounded integral (only once the motion is over)
        q_meas = d.qpos[arm.qadr]
        if arm.path is None:
            arm.q_int = np.clip(arm.q_int + self.ki * dt * (q - q_meas), -0.2, 0.2)
        tau = d.qfrc_bias[arm.dadr].copy()
        if arm.holding is not None:
            # the held object's weight, felt at the hand: tau = J^T (m g up) at the object's position
            mujoco.mj_jac(self.model, d, self._jacp, None, d.xpos[arm.holding], arm.body)
            weight = float(self.model.body_subtreemass[arm.holding]) * -float(self.model.opt.gravity[2])
            tau += self._jacp[2, arm.dadr] * weight
        # torque feed-forward tau / kp, and kv / kp * qdot_des so the servo's damping does not
        # drag the joint behind a moving target
        ff = np.divide(tau + arm.kv * qd_des, arm.kp, out=np.zeros(len(q)), where=arm.kp > 0)
        u = np.clip(q + ff + arm.q_int, arm.lo - 0.4, arm.hi + 0.4)
        if np.isfinite(u).all():        # MuJoCo zeroes ALL controls (legs too) on one bad value
            d.ctrl[arm.act] = u
        if arm.grip_act >= 0:
            d.ctrl[arm.grip_act] = arm.grip_cmd
        self._update_state(arm, d, t)

    def _update_state(self, arm: _Arm, d: mujoco.MjData, t: float) -> None:
        if arm.skill == "reach" and arm.goal is not None:
            arm.error_m = float(np.linalg.norm(arm.goal - arm.hand_xyz))
        elif arm.skill == "point_at" and arm.point_target is not None:
            v = arm.point_target - arm.hand_xyz
            a = self._eff_axis(arm, d)
            arm.error_m = float(np.linalg.norm(v - (v @ a) * a)) if v @ a > 0 else float(np.linalg.norm(v))
        if arm.state != "moving" or arm.path is not None or arm.skill in ("grasp", "release"):
            return
        settled = t - arm.t_mark
        rest = "holding" if arm.holding is not None else "done"
        if arm.skill == "reach":
            # done once within 5 mm; after 1.5 s settle accept anything within 3 cm; give a
            # slow IK convergence up to 3 s before calling the hand blocked
            if arm.error_m < self.done_tol or (settled > 1.5 and arm.error_m < self.fail_tol) or settled > 3.0:
                ok = arm.error_m < self.fail_tol
                arm.state = rest if ok else "failed"
                g = arm.goal
                arm.message = (f"{arm.side} hand at ({g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f}), error {arm.error_m * 100:.1f} cm"
                               if ok else f"{arm.side} hand blocked {arm.error_m:.2f} m short of the target (collision?)")
        elif settled > 0.4:
            arm.state = rest
            if arm.skill == "point_at":
                arm.message = f"{arm.side} arm pointing, the line passes {arm.error_m:.2f} m from the target"
            else:
                arm.message = f"{arm.side} arm: {arm.skill} done"

    # ---- gripper and held objects ---------------------------------------------------
    def _finger_contacts(self, d: mujoco.MjData) -> dict[int, set[int]]:
        """finger geom -> set of non-robot bodies it touches."""
        out: dict[int, set[int]] = {}
        if d.ncon == 0:
            return out
        g = d.contact.geom[:d.ncon]
        fingers = {f for arm in self.arms.values() for f in arm.fingers if f >= 0}
        m = self.model
        for g1, g2 in g:
            for f, o in ((g1, g2), (g2, g1)):
                if f in fingers:
                    b = int(m.geom_bodyid[o])
                    if b != 0 and int(m.body_rootid[b]) != self.root:
                        out.setdefault(int(f), set()).add(b)
        return out

    def _pinched(self, arm: _Arm, contacts: dict[int, set[int]]) -> set[int]:
        return contacts.get(arm.fingers[0], set()) & contacts.get(arm.fingers[1], set())

    def _held_name(self, arm: _Arm) -> str:
        return self._prop_names.get(arm.holding, self.model.body(arm.holding).name)

    def _update_grip(self, arm: _Arm, d: mujoco.MjData, t: float, contacts) -> None:
        opening = float(d.qpos[arm.grip_qadr])
        both = self._pinched(arm, contacts)
        if arm.skill == "grasp" and arm.state == "moving":
            if both:
                arm.t_contact = t if arm.t_contact is None else arm.t_contact
                if t - arm.t_contact >= 0.15:
                    arm.holding = min(both, key=lambda b: np.linalg.norm(d.xpos[b] - arm.hand_xyz))
                    arm.t_lost = None
                    assist = self.grasp_assist and self._weld(arm, d, arm.holding)
                    arm.state = "holding"
                    arm.message = f"{arm.side} hand holding {self._held_name(arm)}" + (" (grasp assist)" if assist else "")
            else:
                arm.t_contact = None
                if opening < 0.003 and t - arm.t_mark > 0.3:
                    arm.state, arm.message = "failed", f"{arm.side} hand closed on nothing"
                elif t - arm.t_mark > 2.0:
                    arm.state, arm.message = "failed", f"{arm.side} hand could not get both fingers on an object"
        elif arm.skill == "release" and arm.state == "moving":
            if opening > 0.85 * arm.grip_open or t - arm.t_mark > 1.5:
                arm.state, arm.message = "done", f"{arm.side} hand open"
        if arm.holding is not None and not (arm.skill == "grasp" and arm.state == "moving"):
            if arm.weld_on >= 0:
                lost = np.linalg.norm(d.xpos[arm.holding] - arm.hand_xyz) > 0.15
            else:
                lost = arm.holding not in both
            if not lost:
                arm.t_lost = None
            elif arm.t_lost is None:
                arm.t_lost = t
            elif t - arm.t_lost > 0.3:
                name = self._held_name(arm)
                self._drop(arm)
                arm.state, arm.message = "failed", f"{arm.side} hand dropped {name}"

    def _weld(self, arm: _Arm, d: mujoco.MjData, body: int) -> bool:
        """Grasp assist: weld the prop to the hand at their current relative pose."""
        eq = arm.welds.get(body)
        if eq is None:
            return False
        m, h = self.model, arm.body
        Rh = d.xmat[h].reshape(3, 3)
        rel = np.zeros(4)
        neg = np.zeros(4)
        mujoco.mju_negQuat(neg, d.xquat[h])
        mujoco.mju_mulQuat(rel, neg, d.xquat[body])
        m.eq_data[eq, 0:3] = 0.0                       # anchor: origin of the prop
        m.eq_data[eq, 3:6] = Rh.T @ (d.xpos[body] - d.xpos[h])
        m.eq_data[eq, 6:10] = rel
        d.eq_active[eq] = 1
        arm.weld_on = eq
        return True

    def _drop(self, arm: _Arm) -> None:
        if arm.weld_on >= 0 and self._data is not None:
            self._data.eq_active[arm.weld_on] = 0
        arm.weld_on, arm.holding, arm.t_lost = -1, None, None
