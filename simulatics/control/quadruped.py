"""Trot walking controller for quadrupeds with FL/FR/RL/RR x hip/thigh/calf legs (Unitree Go2 and alike).

    path (world waypoints) -> velocity command (vx, vy, yaw rate) in the body's heading frame
    trot gait: diagonal pairs (FL+RR, FR+RL) alternate between stance and swing
        stance foot: moves opposite to the commanded body velocity; its height carries the roll / pitch /
                     height feedback (the leg on the low side pushes)
        swing foot : smooth lifted arc to a Raibert foothold
                     nominal + v * T_stance / 2 + k_v * (v - v_cmd)  (+ the same for the yaw rate)
    analytic leg inverse kinematics (abduction + two-link sagittal chain) -> joint targets
    joint PD (gains from pd_hold.py, capped by joint inertia) + weight feed-forward J^T f on the stance legs

All foot positions are kept in the "level" frame: origin at the base, x along the heading, z up (world
vertical). They are rotated into the tilted body frame before the IK, so a swing foot lands on the ground
plane even when the trunk is tilted.

Standing still is a plain PD hold of the home pose with the weight fed forward, so it does not drift.
The floor is assumed flat (height feedback uses the base height above the floor it was reset on).
"""
from __future__ import annotations

import math
import re

import mujoco
import numpy as np

from .pd_hold import PDHoldController

LEGS = ("FL", "FR", "RL", "RR")
PARTS = ("hip", "thigh", "calf")
G = 9.81
_PI = math.pi


def _wrap(a: float) -> float:
    return (a + _PI) % (2 * _PI) - _PI


def _sat(v: float, lim: float) -> float:
    return lim if v > lim else (-lim if v < -lim else v)


def _leg_joints(model: mujoco.MjModel) -> np.ndarray | None:
    """Joint ids (4, 3) of FL/FR/RL/RR x hip/thigh/calf, or None when the model has no such legs."""
    found: dict[tuple[str, str], int] = {}
    pat = re.compile(r"^(FL|FR|RL|RR)_(hip|thigh|calf)(_joint)?$")
    for j in range(model.njnt):
        mt = pat.match(model.joint(j).name)
        if mt and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            found.setdefault((mt.group(1), mt.group(2)), j)
    if len(found) != 12:
        return None
    return np.array([[found[(leg, part)] for part in PARTS] for leg in LEGS], int)


class QuadWalkController:
    """Trot to a clicked point and stand. See the module text for the method.

    API (the Controller protocol of control/base.py):
        reset(model, data)            home stance on the floor at the current x, y and heading
        set_path(path_xy, final_yaw)  walk along world waypoints, stop on the last one, then turn to final_yaw
        set_velocity(vx, vy, wz)      continuous trot (m/s, m/s, rad/s in the heading frame); zeros = stop
        step(model, data)             one control step: writes data.ctrl of the twelve leg motors
        status()                      {"mode": "stand"|"walk"|"done"|"fallen", "message", "goal_error_m", ...}
    """
    name = "quad-walker"

    # gait
    period = 0.40            # s, one full trot cycle (each pair: half stance, half swing)
    duty = 0.5               # stance fraction of the cycle
    step_height = 0.07       # m, swing apex
    kv_foothold = 0.12       # s, Raibert gain on the velocity error
    max_reach = 0.16         # m, a foothold stays this close to its nominal spot
    # feedback on the stance legs
    k_att = 0.8              # foot height per (lever arm * tilt): 1 = legs hold the trunk like a rigid table
    k_att_rate = 0.03        # s, the same on the roll / pitch rate
    k_height = 0.6           # leg extension per metre of height error
    # speeds
    v_max = 0.55             # m/s forward along a path
    v_back = 0.3             # m/s backward / final approach
    v_side = 0.25            # m/s sideways
    w_max = 1.5              # rad/s
    accel = 1.5              # m/s^2 on the command
    w_accel = 6.0            # rad/s^2 on the command
    # path following
    pass_radius = 0.08       # m, an intermediate waypoint counts as passed inside this
    final_radius = 0.35      # m, inside this the last waypoint is approached in any direction (no turning first)
    turn_first = 0.6         # rad, heading error above which the dog turns on the spot before walking
    goal_tolerance = 0.025   # m, stop trotting inside this
    yaw_tolerance = math.radians(2.5)
    accept_goal = 0.06       # m, after standing: a larger error triggers a correction walk
    accept_yaw = math.radians(6.0)
    max_corrections = 3
    settle_time = 0.3        # s standing still before "done"

    @staticmethod
    def supports(model: mujoco.MjModel) -> bool:
        """A free-floating robot with FL/FR/RL/RR legs of hip, thigh and calf hinge joints, each with a motor."""
        if model.njnt < 13 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            return False
        jid = _leg_joints(model)
        if jid is None:
            return False
        driven = {int(model.actuator_trnid[i, 0]) for i in range(model.nu)
                  if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
                  and model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_NONE}
        return all(int(j) in driven for j in jid.ravel())

    def __init__(self, model: mujoco.MjModel):
        if not self.supports(model):
            raise ValueError("not a quadruped with FL/FR/RL/RR hip/thigh/calf legs driven by torque motors")
        self.control_dt = float(model.opt.timestep)      # torque PD needs every physics step
        dt = self.control_dt
        jid = _leg_joints(model)
        self.jid = jid
        self.qadr = model.jnt_qposadr[jid]               # (4, 3)
        self.dadr = model.jnt_dofadr[jid]
        act_of = {int(model.actuator_trnid[i, 0]): i for i in range(model.nu)
                  if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT}
        self.act = np.array([[act_of[int(j)] for j in row] for row in jid], int)
        self.root = int(model.jnt_bodyid[0])

        # gains and limits: the stand hold's logic (torque limit, capped by joint inertia for a stable explicit PD)
        hold = PDHoldController(model)
        order = {int(a): k for k, a in enumerate(hold.act)}
        pick = np.array([[order[int(a)] for a in row] for row in self.act], int)
        self.kp, self.kd, self.gear = hold.kp[pick], hold.kd[pick], hold.gear[pick]
        self.q_home = hold.base_target[pick].copy()
        self.q_lo, self.q_hi = hold.target_lo[pick] + 0.02, hold.target_hi[pick] - 0.02
        lim = np.where(hold.ctrl_limited[pick], hold.hi[pick] * self.gear, np.inf)
        frc = model.actuator_forcerange[self.act, 1]
        self.tau_max = np.where(model.actuator_forcelimited[self.act].astype(bool) & (frc > 0),
                                np.minimum(lim, frc), lim)

        # leg geometry from the body tree: hip offset, abduction link, thigh and shank lengths
        hip_b = model.jnt_bodyid[jid[:, 0]]
        thigh_b = model.jnt_bodyid[jid[:, 1]]
        calf_b = model.jnt_bodyid[jid[:, 2]]
        self.hip_off = model.body_pos[hip_b].copy()                      # (4, 3) in the base frame
        self.side_d = model.body_pos[thigh_b][:, 1].copy()               # signed sideways offset of the thigh
        self.l1 = float(np.mean(np.abs(model.body_pos[calf_b][:, 2])))
        l2, radius = [], []
        for b in calf_b:                                                 # the foot: the lowest sphere on the calf
            gs = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b
                  and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[g]]
            if gs:
                g = min(gs, key=lambda k: model.geom_pos[k][2])
                l2.append(abs(float(model.geom_pos[g][2])))
                radius.append(float(model.geom_size[g][0]))
        self.l2 = float(np.mean(l2)) if l2 else self.l1
        self.foot_radius = float(np.mean(radius)) if radius else 0.02
        c3_lo, c3_hi = math.cos(float(self.q_lo[0, 2])), math.cos(float(self.q_hi[0, 2]))
        self._r2_min = self.l1 ** 2 + self.l2 ** 2 + 2 * self.l1 * self.l2 * min(c3_lo, c3_hi)
        self._r2_max = self.l1 ** 2 + self.l2 ** 2 + 2 * self.l1 * self.l2 * max(c3_lo, c3_hi)

        self.weight = float(model.body_subtreemass[self.root]) * G
        self.nom = self.hip_off + self._fk(self.q_home)                  # home feet in the base frame
        self.height = float(-np.mean(self.nom[:, 2]))                    # base above the foot centres
        self.stand_z = self.height + self.foot_radius                    # base above the floor
        d0 = mujoco.MjData(model)
        d0.qpos[:] = model.qpos0
        d0.qpos[self.qadr] = self.q_home
        mujoco.mj_forward(model, d0)
        com = d0.xmat[self.root].reshape(3, 3).T @ (d0.subtree_com[self.root] - d0.xpos[self.root])
        self.com_off = com[:2].copy()                                    # CoM ahead of the base origin
        self.walk_nom = self.nom.copy()
        self.walk_nom[:, :2] += self.com_off - self.nom[:, :2].mean(axis=0)   # feet centred under the CoM

        self._off = np.array([0.0, 0.5, 0.5, 0.0])                       # trot: FL+RR, then FR+RL
        self._dt = dt
        self.floor_z = 0.0
        self._reset_state()
        # The feet are soft (they sink some millimetres under the weight): let the home stance settle once on
        # a scratch copy, so that reset() starts at the true standing height and the height feedback aims at it.
        d0.qpos[:] = model.qpos0
        d0.qvel[:] = 0.0
        self.reset(model, d0)
        n = max(1, int(round(self.control_dt / model.opt.timestep)))
        for _ in range(int(0.6 / self.control_dt)):
            self.step(model, d0)
            mujoco.mj_step(model, d0, n)
        if np.isfinite(d0.qpos).all() and 0.5 * self.stand_z < d0.qpos[2] < 1.1 * self.stand_z:
            self.stand_z = float(d0.qpos[2])
        self._reset_state()

    # ------------------------------------------------------------------ kinematics
    def _fk(self, q: np.ndarray) -> np.ndarray:
        """Foot centres (4, 3) relative to each hip joint, in the base frame."""
        q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2]
        x = -self.l1 * np.sin(q2) - self.l2 * np.sin(q2 + q3)
        zz = -self.l1 * np.cos(q2) - self.l2 * np.cos(q2 + q3)
        c1, s1 = np.cos(q1), np.sin(q1)
        return np.stack([x, self.side_d * c1 - zz * s1, self.side_d * s1 + zz * c1], axis=1)

    def _ik(self, p: np.ndarray) -> np.ndarray:
        """Joint angles (4, 3) for foot centres p (4, 3) relative to each hip joint, in the base frame
        (knees bent backward; the reach is clamped to the knee range)."""
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        d = self.side_d
        zz2 = np.maximum(y * y + z * z - d * d, 1e-6)
        zz = -np.sqrt(zz2)
        q1 = np.arctan2(z, y) - np.arctan2(zz, d)
        r2_raw = x * x + zz2
        r2 = np.minimum(np.maximum(r2_raw, self._r2_min), self._r2_max)
        k = np.sqrt(r2 / r2_raw)
        xc, zz = x * k, zz * k
        q1 = (q1 + _PI) % (2 * _PI) - _PI
        c3 = (r2 - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2)
        q3 = -np.arccos(np.minimum(np.maximum(c3, -1.0), 1.0))
        q2 = np.arctan2(-xc, -zz) - np.arctan2(self.l2 * np.sin(q3), self.l1 + self.l2 * c3)
        q = np.stack([q1, q2, q3], axis=1)
        return np.minimum(np.maximum(q, self.q_lo), self.q_hi)

    def _weight_torque(self, q: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Joint torques (4, 3) with which each leg pushes on the ground with force f (4, 3, base frame): J^T f."""
        q1, q2, q23 = q[:, 0], q[:, 1], q[:, 1] + q[:, 2]
        s2, c2, s23, c23 = np.sin(q2), np.cos(q2), np.sin(q23), np.cos(q23)
        c1, s1 = np.cos(q1), np.sin(q1)
        x = -self.l1 * s2 - self.l2 * s23
        zz = -self.l1 * c2 - self.l2 * c23
        y = self.side_d * c1 - zz * s1
        z = self.side_d * s1 + zz * c1
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        lat = -s1 * fy + c1 * fz                       # force along the leg plane's "down" axis
        t1 = -z * fy + y * fz
        t2 = zz * fx - x * lat
        t3 = -self.l2 * c23 * fx + self.l2 * s23 * lat
        return np.stack([t1, t2, t3], axis=1)

    # ------------------------------------------------------------------ protocol
    def _reset_state(self) -> None:
        self.mode, self.message = "stand", "standing"
        self._wps: np.ndarray | None = None
        self._wp_i = 0
        self._final_yaw: float | None = None
        self._vel: tuple[float, float, float] | None = None
        self._stage = "idle"             # idle | path | stopping | settle
        self._turning = False
        self.goal_xy: np.ndarray | None = None
        self.goal_err, self.yaw_err = 0.0, 0.0
        self.corrections = 0
        self._settle = 0.0
        self.cmd = np.zeros(3)           # ramped command (vx, vy, wz) in the heading frame
        self._gait = False
        self.phase = 0.0
        self._still = 0.0                # s the command has been zero while trotting
        self.p = self.nom.copy()         # commanded feet, level frame, relative to the base
        self.p0 = self.nom.copy()        # where each swing started
        self._swing = np.zeros(4, bool)
        self.q_des = self.q_home.copy()
        self._q_prev = self.q_home.copy()
        self._blend = 1.0                # 0..1 from the last trot pose to the home pose
        self._q_from = self.q_home.copy()
        self._vf = np.zeros(2)           # filtered base velocity, heading frame
        self.speed = 0.0
        self.up_z = 1.0

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Home stance with the feet on the floor at the current x, y and heading; zero velocity.
        Writes data.qpos / qvel / ctrl of the root and the legs and calls mj_forward."""
        self._reset_state()
        w, x, y, z = (float(v) for v in data.qpos[3:7])
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        if not math.isfinite(yaw):
            yaw = 0.0
        if not np.isfinite(data.qpos[:2]).all():
            data.qpos[:2] = 0.0
        data.qpos[2] = self.floor_z + self.stand_z
        data.qpos[3:7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        data.qpos[self.qadr] = self.q_home
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        f = np.zeros((4, 3))
        f[:, 2] = -self.weight / 4
        data.ctrl[self.act] = self._clip(self._weight_torque(self.q_home, f)) / self.gear

    def set_path(self, path_xy, final_yaw: float | None = None) -> None:
        """Walk along world-frame waypoints (n, 2), stop on the last one and turn to final_yaw (rad; None = keep
        the arrival heading). None or an empty path = stop and stand. Raises ValueError for non-finite input."""
        pts = None if path_xy is None else np.asarray(path_xy, float).reshape(-1, 2)
        if (pts is not None and not np.isfinite(pts).all()) or (final_yaw is not None
                                                                 and not math.isfinite(final_yaw)):
            raise ValueError("set_path needs finite waypoints and final yaw")
        self._vel = None
        if pts is None or len(pts) == 0:
            self._wps, self.goal_xy, self._final_yaw = None, None, None
            if self.mode != "fallen":
                self.mode = "stand"
                self.message = "stopping" if self._gait else "standing"
            self._stage = "stopping" if self._gait else "idle"
            return
        self._wps, self._wp_i = pts.copy(), 0
        self._final_yaw = None if final_yaw is None else float(final_yaw)
        self.goal_xy = pts[-1].copy()
        self.corrections, self._turning, self._settle = 0, False, 0.0
        self._stage = "path"
        if self.mode != "fallen":
            self.mode, self.message = "walk", "walking"

    def set_velocity(self, vx: float, vy: float = 0.0, wz: float = 0.0) -> None:
        """Continuous trot: forward / left speed (m/s) and yaw rate (rad/s) in the heading frame, clipped to the
        limits. (0, 0, 0) = stop and stand."""
        if not np.isfinite([vx, vy, wz]).all():
            raise ValueError("set_velocity needs finite vx, vy, wz")
        if abs(vx) + abs(vy) + abs(wz) < 1e-6:
            self.set_path(None)
            return
        self._wps, self.goal_xy, self._final_yaw = None, None, None
        self._vel = (min(max(float(vx), -self.v_back), self.v_max), _sat(float(vy), self.v_side),
                     _sat(float(wz), self.w_max))
        self._stage = "idle"
        if self.mode != "fallen":
            self.mode, self.message = "walk", "velocity command"

    def status(self) -> dict:
        """{"mode": "stand"|"walk"|"done"|"fallen", "message", "goal_error_m", "heading_error_deg" (None without
        a goal / final yaw), "footsteps": [] (a trot has no footstep plan), "support": "all"|"FL+RR"|"FR+RL",
        "speed": m/s, "gait_phase": 0..1 (None while standing), "command": [vx, vy, wz]}"""
        has_goal = self.goal_xy is not None
        if not self._gait:
            support = "all"
        else:
            support = "FR+RL" if self._swing[0] else "FL+RR"
        return {"mode": self.mode, "message": self.message, "footsteps": [], "support": support,
                "goal_error_m": self.goal_err if has_goal else None,
                "heading_error_deg": (math.degrees(self.yaw_err)
                                      if has_goal and self._final_yaw is not None else None),
                "speed": self.speed, "gait_phase": self.phase if self._gait else None,
                "command": [float(self.cmd[0]), float(self.cmd[1]), float(self.cmd[2])]}

    # ------------------------------------------------------------------ control
    def _clip(self, tau: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(tau, -self.tau_max), self.tau_max)

    def step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        dt = self._dt
        qpos, qvel = data.qpos, data.qvel
        px, py, pz, w, x, y, z = (float(v) for v in qpos[:7])
        r00, r10 = 1 - 2 * (y * y + z * z), 2 * (x * y + w * z)
        r20, r21, r22 = 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)
        yaw = math.atan2(r10, r00)
        cy, sy = math.cos(yaw), math.sin(yaw)
        self.up_z = r22
        vxw, vyw = float(qvel[0]), float(qvel[1])
        vx, vy = cy * vxw + sy * vyw, -sy * vxw + cy * vyw
        self.speed = math.hypot(vxw, vyw)
        self._vf[0] += 0.1 * (vx - self._vf[0])
        self._vf[1] += 0.1 * (vy - self._vf[1])

        if self.mode == "fallen" or r22 < 0.3 or not math.isfinite(pz):
            if self.mode != "fallen":
                self.mode, self.message = "fallen", "robot fell"
                self._gait, self._stage, self._wps, self._vel = False, "idle", None, None
            data.ctrl[self.act] = self._clip(-self.kd * qvel[self.dadr]) / self.gear     # limp, damped
            return

        want = self._command(px, py, yaw)                 # None = stand, else (vx, vy, wz)
        if want is None:
            tx = ty = tw = 0.0
        else:
            tx, ty, tw = want
        c = self.cmd
        c[0] += _sat(tx - c[0], self.accel * dt)
        c[1] += _sat(ty - c[1], self.accel * dt)
        c[2] += _sat(tw - c[2], self.w_accel * dt)

        if want is not None and not self._gait:           # start trotting from the stand
            self._gait, self.phase, self._still = True, 0.0, 0.0
            self.p[:] = self.nom
            self.p0[:] = self.nom
            self._swing[:] = False
        if self._gait:
            # rotation that takes level-frame vectors into the (tilted) base frame: R^T Rz(yaw)
            rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                            [r10, 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                            [r20, r21, r22]])
            m_lb = rot.T @ np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            roll, pitch = math.atan2(r21, r22), -math.asin(max(-1.0, min(1.0, r20)))
            ended = self._trot(dt, want is None, roll, pitch, float(qvel[3]), float(qvel[4]),
                               pz - self.floor_z)
            q_des = self._ik(self.p @ m_lb.T - self.hip_off)
            stance = ~self._swing
            n_st = max(1, int(stance.sum()))
            f = np.outer(stance * (-self.weight / n_st), m_lb[:, 2])
            if ended:
                self._gait, self._blend = False, 0.0
                self._q_from[:] = q_des
                self._swing[:] = False
        else:
            if self._blend < 1.0:
                self._blend = min(1.0, self._blend + dt / 0.25)
                s = self._blend * self._blend * (3 - 2 * self._blend)
                q_des = self._q_from + (self.q_home - self._q_from) * s
            else:
                q_des = self.q_home
            f = np.zeros((4, 3))
            f[:, 2] = -self.weight / 4

        qd_des = (q_des - self._q_prev) * (1.0 / dt)
        np.minimum(np.maximum(qd_des, -15.0, out=qd_des), 15.0, out=qd_des)
        self._q_prev = q_des
        self.q_des = q_des
        tau = (self.kp * (q_des - qpos[self.qadr]) + self.kd * (qd_des - qvel[self.dadr])
               + self._weight_torque(q_des, f))
        data.ctrl[self.act] = self._clip(tau) / self.gear

        if not self._gait and self._blend >= 1.0:
            self._standing(px, py, yaw, dt)

    def _trot(self, dt: float, stopping: bool, roll: float, pitch: float, wx: float, wy: float,
              z_base: float) -> bool:
        """Advance the gait by dt and update the commanded feet self.p (level frame). Returns True when a
        requested stop is complete (all feet down near their home spots)."""
        vx, vy, wz = float(self.cmd[0]), float(self.cmd[1]), float(self.cmd[2])
        moving = abs(vx) + abs(vy) + 0.2 * abs(wz) > 0.02
        self._still = 0.0 if (moving or not stopping) else self._still + dt
        prev = self.phase
        self.phase = (self.phase + dt / self.period) % 1.0
        if stopping and self._still > self.period and (self.phase < prev or (prev < 0.5 <= self.phase)):
            return True                                   # a pair has just landed: all four feet are down
        psi = (self.phase + self._off) % 1.0
        swing = psi >= self.duty
        lift = swing & ~self._swing
        if lift.any():
            self.p0[lift] = self.p[lift]
        self._swing = swing
        p = self.p
        nom = self.walk_nom
        # stance: the ground moves backward under the body
        sx = p[:, 0] - (vx - wz * p[:, 1]) * dt
        sy = p[:, 1] - (vy + wz * p[:, 0]) * dt
        h = self.height + min(max(self.k_height * (self.stand_z - z_base), -0.03), 0.03)
        sz_t = -h + (self.k_att * roll + self.k_att_rate * wx) * nom[:, 1] \
                  - (self.k_att * pitch + self.k_att_rate * wy) * nom[:, 0]
        sz = p[:, 2] + 0.25 * (sz_t - p[:, 2])
        # swing: arc to the Raibert foothold
        u = np.minimum(np.maximum((psi - self.duty) / (1.0 - self.duty), 0.0), 1.0)
        s = u * u * (3.0 - 2.0 * u)
        half = 0.5 * self.duty * self.period
        fvx, fvy = float(self._vf[0]), float(self._vf[1])
        ox = _sat(half * fvx + self.kv_foothold * (fvx - vx), self.max_reach)
        oy = _sat(half * fvy + self.kv_foothold * (fvy - vy), self.max_reach)
        tx = nom[:, 0] + ox - half * wz * nom[:, 1]
        ty = nom[:, 1] + oy + half * wz * nom[:, 0]
        b = u * (1.0 - u)
        wxp = self.p0[:, 0] + (tx - self.p0[:, 0]) * s
        wyp = self.p0[:, 1] + (ty - self.p0[:, 1]) * s
        wzp = self.p0[:, 2] + (-self.height - self.p0[:, 2]) * s + self.step_height * 16.0 * b * b
        p[:, 0] = np.where(swing, wxp, sx)
        p[:, 1] = np.where(swing, wyp, sy)
        p[:, 2] = np.where(swing, wzp, sz)
        return False

    # ------------------------------------------------------------------ path following
    def _command(self, px: float, py: float, yaw: float):
        """Velocity command (vx, vy, wz) in the heading frame, or None to stand."""
        if self._vel is not None:
            return self._vel
        if self.goal_xy is not None:
            self.goal_err = math.hypot(self.goal_xy[0] - px, self.goal_xy[1] - py)
            self.yaw_err = _wrap(yaw - self._final_yaw) if self._final_yaw is not None else 0.0
        if self._stage != "path" or self._wps is None:
            return None
        wps, n = self._wps, len(self._wps)
        i = self._wp_i
        while i < n - 1 and math.hypot(wps[i, 0] - px, wps[i, 1] - py) < self.pass_radius:
            i += 1
        self._wp_i = i
        dx, dy = float(wps[i, 0]) - px, float(wps[i, 1]) - py
        dist = math.hypot(dx, dy)
        cy, sy = math.cos(yaw), math.sin(yaw)
        last = i == n - 1
        if last and dist < self.final_radius:
            # final approach: straight to the point in any direction, turning to the final heading on the way
            yaw_e = -self.yaw_err
            if dist < self.goal_tolerance and abs(yaw_e) < self.yaw_tolerance:
                self._stage = "stopping"
                return None
            bx, by = cy * dx + sy * dy, -sy * dx + cy * dy
            return (_sat(2.5 * bx, self.v_back), _sat(2.5 * by, self.v_side), _sat(2.5 * yaw_e, self.w_max))
        e = _wrap(math.atan2(dy, dx) - yaw)
        if abs(e) > (0.2 if self._turning else self.turn_first):
            self._turning = True
            return (0.0, 0.0, _sat(3.0 * e, self.w_max))
        self._turning = False
        v = min(self.v_max, 0.15 + 1.5 * dist) * max(0.3, math.cos(e))
        return (v, 0.0, _sat(3.0 * e, self.w_max))

    def _standing(self, px: float, py: float, yaw: float, dt: float) -> None:
        """Bookkeeping while all four feet hold the home stance."""
        if self.mode == "stand" and self.message == "stopping":
            self.message = "standing"
        if self.mode != "walk":
            return
        if self._stage != "stopping":
            if self._vel is None and self._stage == "idle":
                self.mode, self.message = "stand", "standing"
            return
        self._settle = self._settle + dt if self.speed < 0.03 else 0.0
        if self._settle < self.settle_time:
            return
        self._settle = 0.0
        if self.goal_xy is None:
            self.mode, self.message, self._stage = "stand", "standing", "idle"
            return
        if self.corrections < self.max_corrections and (self.goal_err > self.accept_goal
                                                        or abs(self.yaw_err) > self.accept_yaw):
            self.corrections += 1
            self._stage = "path"
            self.message = f"correction {self.corrections}"
            return
        self.mode, self._stage = "done", "idle"
        self.message = f"goal reached ({self.goal_err:.3f} m, {math.degrees(self.yaw_err):.1f} deg)"
