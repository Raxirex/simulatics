"""Whole-body layer of the MPC walker: CoM placement + leg IK, swing leg, hip strategy, servo feed-forward.

The MPC (control/mpc_lipm.py) decides where the centre of mass goes and where
the feet land; this module turns that into joint targets for the position
servos of the built-in humanoid. It is an adapted copy of the corresponding
parts of control/walker.py (WalkController._whole_body, _swing_leg_world,
_torso_feedback, _feedforward, _motor_load; same project and licence), copied
rather than imported because walker.py is being extended in parallel and its
private methods may change. Differences from the walker:

* the contact wrench of the feed-forward is built around the MPC's ZMP: the
  reference CoM acceleration fed to the floating-base inverse dynamics is the
  plan's (so the centre of pressure of the feed-forward is the planned ZMP),
  and the DCM feedback moves it from there;
* the torso / joint accelerations are given explicitly instead of being
  differentiated from a reference that the MPC re-anchors on the measured
  state every update.

Pipeline per control tick (see control/walker.py for the derivations):
  1. torso position whose whole-body (arms included) forward-kinematics CoM is the reference CoM,
     legs from the closed-form 6-DOF IK of control/walk_ik.py;
  2. swing leg solved against the low-passed measured pelvis pose (lands where planned in the world);
  3. torso pitch/roll error -> stance hip offsets (hip strategy);
  4. ctrl = q + (kv/kp) qd + tau/kp, tau from M qdd + h = S^T tau + J_c^T f with the contact wrench f split
     over the stance feet and the ankle torques limited to what the sole can carry.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .walk_footsteps import LEFT, RIGHT
from .walk_ik import ABDOMEN, LEG_JOINTS, SIDES, RobotGeometry, leg_ik, rot_z, yaw_of

G = 9.81
SIDE_NAME = {LEFT: "left", RIGHT: "right"}


@dataclass
class Meas:
    com: np.ndarray
    com_vel: np.ndarray               # low-passed
    R_torso: np.ndarray
    p_pelvis: np.ndarray
    R_pelvis: np.ndarray
    feet: dict = field(default_factory=dict)     # side -> (x, y, yaw) of the sole centre
    fallen: bool = False


class LegBody:
    """Model-derived kinematics and dynamics helpers for the leg/abdomen actuators (14 of them)."""

    def __init__(self, model: mujoco.MjModel, hip_height: float, vel_filter_tau: float, swing_world_tau: float,
                 control_dt: float):
        self.model = model
        self.geo = RobotGeometry.from_model(model)
        self.mass = self.geo.mass
        self.torso_id = model.body("torso").id
        self.pelvis_id = model.body("pelvis").id
        self.foot_body = {s: model.body(f"{SIDE_NAME[s]}_foot").id for s in (LEFT, RIGHT)}
        names = list(ABDOMEN) + [f"{s}_{j}" for s in SIDES for j in LEG_JOINTS]
        jid = [model.joint(n).id for n in names]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jid])
        self.act = np.array([model.actuator(n).id for n in names])
        self.scratch = mujoco.MjData(model)
        leg = self.geo.leg_length
        self.width = self.geo.hip_width
        self.z_torso = (self.geo.ankle_height + hip_height * leg - self.geo.legs["left"].hip[2]
                        - self.geo.pelvis[2])
        self._com_off: dict[str, np.ndarray] = {}
        self.feet0 = {LEFT: np.array([0.0, self.width / 2, 0.0]), RIGHT: np.array([0.0, -self.width / 2, 0.0])}
        self.scratch.qpos[:] = model.qpos0
        _, p_t, com = self.whole_body(np.zeros(2), 0.0, self.targets(self.feet0))
        self.z_c = float(com[2])
        self.omega = float(np.sqrt(G / self.z_c))
        self.stand_offset = p_t[:2].copy()           # torso relative to the sole midpoint (heading frame)
        alpha = lambda tau: float(1.0 - np.exp(-control_dt / max(tau, 1e-6)))   # noqa: E731
        self._vf_alpha, self._pel_alpha = alpha(vel_filter_tau), alpha(swing_world_tau)
        self.reset_filters()

    def reset_filters(self) -> None:
        self._vfilt = None
        self._pel_err = np.zeros(6)

    # ------------------------------------------------------------ measuring
    def foot_pose(self, data, side) -> tuple[float, float, float]:
        b = self.foot_body[side]
        R = data.xmat[b].reshape(3, 3)
        c = data.xpos[b] + R @ self.geo.legs[SIDE_NAME[side]].sole
        return float(c[0]), float(c[1]), yaw_of(R)

    def foot_height(self, data, side) -> float:
        b = self.foot_body[side]
        R = data.xmat[b].reshape(3, 3)
        return float((data.xpos[b] + R @ self.geo.legs[SIDE_NAME[side]].sole)[2])

    def measure(self, model, data) -> Meas:
        mujoco.mj_subtreeVel(model, data)
        R_t = data.xmat[self.torso_id].reshape(3, 3).copy()
        vel = data.subtree_linvel[self.torso_id].copy()
        if self._vfilt is None:
            self._vfilt = vel
        self._vfilt = self._vfilt + self._vf_alpha * (vel - self._vfilt)
        return Meas(com=data.subtree_com[self.torso_id].copy(), com_vel=self._vfilt.copy(), R_torso=R_t,
                    p_pelvis=data.xpos[self.pelvis_id].copy(),
                    R_pelvis=data.xmat[self.pelvis_id].reshape(3, 3).copy(),
                    feet={s: np.array(self.foot_pose(data, s)) for s in (LEFT, RIGHT)},
                    fallen=bool(R_t[2, 2] < 0.6 or data.qpos[2] < 0.6 * self.z_torso
                                or not np.isfinite(data.qpos).all()))

    # ----------------------------------------------------------- kinematics
    @staticmethod
    def targets(feet: dict) -> dict:
        return {s: (np.array([feet[s][0], feet[s][1], 0.0]), rot_z(float(feet[s][2]))) for s in (LEFT, RIGHT)}

    def legs(self, p_t, R_t, targets) -> np.ndarray:
        p_pel = p_t + R_t @ self.geo.pelvis
        q = np.zeros(14)
        for s in (LEFT, RIGHT):
            pos, R = targets[s]
            q[2 + 6 * s:8 + 6 * s], _ = leg_ik(self.geo.legs[SIDE_NAME[s]], p_pel, R_t, pos, R)
        return q

    def fk_com(self, p_t, R_t, q) -> np.ndarray:
        d = self.scratch
        d.qpos[0:3] = p_t
        mujoco.mju_mat2Quat(d.qpos[3:7], R_t.ravel())
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        return d.subtree_com[self.torso_id].copy()

    def whole_body(self, com_xy, yaw, targets, key: str = "ref"):
        """Torso position + joint angles whose whole-body CoM (x, y) is com_xy at the nominal torso height
        (fixed-point iteration on forward kinematics; scratch data holds the current arm angles)."""
        R_t = rot_z(yaw)
        off = self._com_off.get(key, self._com_off.get("ref", np.zeros(2)))
        p_t = np.array([com_xy[0] + off[0], com_xy[1] + off[1], self.z_torso])
        com = np.zeros(3)
        for _ in range(4):
            com = self.fk_com(p_t, R_t, self.legs(p_t, R_t, targets))
            step = com_xy - com[:2]
            p_t[:2] += step
            if abs(step[0]) + abs(step[1]) < 1e-5:
                break
        self._com_off[key] = p_t[:2] - com_xy
        return self.legs(p_t, R_t, targets), p_t, com

    def swing_leg_world(self, q, side, target, p_t, yaw_t, meas: Meas, gain: float = 1.0) -> None:
        """Solve the swing leg against the reference pelvis corrected by the low-passed measured pelvis error
        (side None: only update the filter)."""
        R_t = rot_z(yaw_t)
        p_pel = p_t + R_t @ self.geo.pelvis
        dR = R_t.T @ meas.R_pelvis
        err = np.concatenate([meas.p_pelvis - p_pel,
                              0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])])
        self._pel_err += self._pel_alpha * (err - self._pel_err)
        if side is None or gain <= 0:
            return
        e = gain * self._pel_err
        th = float(np.linalg.norm(e[3:]))
        Rc = np.eye(3)
        if th > 1e-9:
            k = e[3:] / th
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            Rc = Rc + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        i0 = 2 + 6 * side
        q[i0:i0 + 6], _ = leg_ik(self.geo.legs[SIDE_NAME[side]], p_pel + e[:3], R_t @ Rc, *target)

    def torso_feedback(self, q, meas: Meas, yaw_t: float, stance: list, k_torso: float) -> np.ndarray:
        """Hip strategy: stance hip offsets proportional to the torso pitch / roll error."""
        if k_torso <= 0:
            return q
        R = rot_z(-yaw_t) @ meas.R_torso
        pitch = float(np.arctan2(-R[2, 0], R[2, 2]))
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        for s in stance:
            q[2 + 6 * s + 2] += k_torso * pitch
            q[2 + 6 * s + 0] += k_torso * roll
        return q

    # -------------------------------------------------------------- dynamics
    def feedforward(self, model, data, meas: Meas, stance: list, acc_torso: np.ndarray, qdd: np.ndarray,
                    zmp_ref: np.ndarray, dp: np.ndarray) -> np.ndarray:
        """Joint torques of the floating-base inverse dynamics for the reference accelerations, with the
        contact wrench split over the stance feet (weights from zmp_ref between the feet in double support)
        and its centre of pressure moved by dp (DCM feedback, with the LIPM force change -m omega^2 dp). The
        ankle torques are those of the wrench with its CoP clipped to the sole."""
        nv = model.nv
        vec = np.zeros(nv)
        vec[0:3] = acc_torso
        vec[self.dadr] = qdd
        h = np.zeros(nv)
        mujoco.mj_mulM(model, data, h, vec)
        h += data.qfrc_bias
        J, o = {}, {}
        for s in stance:
            o[s] = np.array([meas.feet[s][0], meas.feet[s][1], 0.0])
            jp, jr = np.zeros((3, nv)), np.zeros((3, nv))
            mujoco.mj_jac(model, data, jp, jr, o[s], self.foot_body[s])
            J[s] = np.vstack([jp, jr])
        if len(stance) == 2:
            a, b = o[LEFT][:2], o[RIGHT][:2]
            d = a - b
            alpha = float(np.clip(np.dot(zmp_ref - b, d) / max(np.dot(d, d), 1e-9), 0.0, 1.0))
            share = {LEFT: max(alpha, 0.02), RIGHT: max(1.0 - alpha, 0.02)}
        else:
            share = {stance[0]: 1.0}
        A = np.hstack([J[s][:, :6].T for s in stance])
        lw = self.geo.legs["left"].half_width ** 2
        winv = np.concatenate([share[s] * np.array([1, 1, 1, lw, lw, lw]) for s in stance])
        try:
            f_all = winv * (A.T @ np.linalg.solve((A * winv) @ A.T, h[:6]))
        except np.linalg.LinAlgError:
            f_all = np.zeros(6 * len(stance))
        tau, tau_ankle = h[self.dadr].copy(), h[self.dadr].copy()
        for i, s in enumerate(stance):
            f = f_all[6 * i:6 * i + 6].copy()
            fz = f[2]
            f[0:2] -= self.mass * self.omega ** 2 * dp * share[s] / sum(share.values())
            fc = f.copy()
            if fz > 1.0:
                f[3] += dp[1] * fz
                f[4] -= dp[0] * fz
                lg = self.geo.legs[SIDE_NAME[s]]
                c, si = np.cos(meas.feet[s][2]), np.sin(meas.feet[s][2])
                rel = np.array([-f[4], f[3]]) / fz
                lx = np.clip(c * rel[0] + si * rel[1], -0.8 * lg.half_length, 0.8 * lg.half_length)
                ly = np.clip(-si * rel[0] + c * rel[1], -0.8 * lg.half_width, 0.8 * lg.half_width)
                fc[3] = (si * lx + c * ly) * fz
                fc[4] = -(c * lx - si * ly) * fz
            tau -= J[s][:, self.dadr].T @ f
            tau_ankle -= J[s][:, self.dadr].T @ fc
        for s in stance:
            tau[2 + 6 * s + 4:2 + 6 * s + 6] = tau_ankle[2 + 6 * s + 4:2 + 6 * s + 6]
        return np.clip(tau, -self.torque_limits(model), self.torque_limits(model))

    def torque_limits(self, model) -> np.ndarray:
        lim = np.abs(model.actuator_forcerange[self.act]).max(axis=1)
        return np.where(model.actuator_forcelimited[self.act].astype(bool) & (lim > 0), lim, np.inf)

    def clip_ctrl(self, model, u: np.ndarray) -> np.ndarray:
        lo, hi = model.actuator_ctrlrange[self.act].T
        limited = model.actuator_ctrllimited[self.act].astype(bool)
        return np.where(limited, np.clip(u, lo, hi), u)

    def motor_load(self, t_single: float, step: float) -> tuple[float, float]:
        """Predicted motor use of the nominal gait as fractions of the torque limits: (static single support,
        swing hip pitch). Same estimate as WalkController._motor_load."""
        m, d = self.model, self.scratch
        feet0 = self.feet0
        d.qpos[:] = m.qpos0
        q, p_t, _ = self.whole_body(feet0[LEFT][:2], 0.0, self.targets(feet0), key="load")
        d.qpos[0:3] = p_t
        d.qpos[3:7] = (1, 0, 0, 0)
        d.qpos[self.qadr] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, jr, feet0[LEFT], self.foot_body[LEFT])
        tau = d.qfrc_bias[self.dadr] - jp[:, self.dadr].T @ np.array([0.0, 0.0, self.mass * G])
        lim = self.torque_limits(m)
        static = float(np.max(np.abs(tau) / lim))
        hy = 2 + 6 * RIGHT + 2
        e = np.zeros(m.nv)
        e[self.dadr[hy]] = 1.0
        Me = np.zeros(m.nv)
        mujoco.mj_mulM(m, d, Me, e)
        acc = 5.77 * (2.0 * step / self.geo.leg_length) / (0.85 * t_single) ** 2
        swing = float(Me[self.dadr[hy]] * acc / lim[hy])
        self._com_off.pop("load", None)
        return static, swing
