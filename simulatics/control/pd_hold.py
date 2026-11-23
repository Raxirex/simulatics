"""Joint-space PD "hold pose" controller for robots driven by torque motors (imported Menagerie robots).

    tau = kp * (q_target - q) - kd * qd        (clipped to each motor's torque limit)

The target is the model's first keyframe (e.g. the Go2's "home" stance) or its default pose. Gains are
derived from each motor's torque limit, so that the full torque is reached at `full_torque_error` rad of
error. This is how such robots are usually held standing in simulation before a learned or model-based
gait takes over. Position-servo actuators are given the target angle directly.
"""
from __future__ import annotations

import mujoco
import numpy as np


class PDHoldController:
    name = "pd-hold"

    def __init__(self, model: mujoco.MjModel, full_torque_error: float = 0.08, damping_ratio: float = 0.04):
        self.control_dt = float(model.opt.timestep)  # every physics step: torque PD needs the full rate
        self.act, self.qadr, self.dadr = [], [], []
        for i in range(model.nu):
            if model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            j = int(model.actuator_trnid[i, 0])
            if model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            self.act.append(i)
            self.qadr.append(int(model.jnt_qposadr[j]))
            self.dadr.append(int(model.jnt_dofadr[j]))
        if not self.act:
            raise ValueError("no joint actuators to hold")
        self.act, self.qadr, self.dadr = (np.array(v, int) for v in (self.act, self.qadr, self.dadr))
        self.servo = model.actuator_biastype[self.act] == mujoco.mjtBias.mjBIAS_AFFINE
        gear = np.abs(model.actuator_gear[self.act, 0])
        gear[gear < 1e-9] = 1.0
        limited = model.actuator_forcelimited[self.act].astype(bool) | model.actuator_ctrllimited[self.act].astype(bool)
        tmax = np.where(model.actuator_forcelimited[self.act].astype(bool), model.actuator_forcerange[self.act, 1],
                        model.actuator_ctrlrange[self.act, 1] * gear)
        tmax = np.where(limited & (tmax > 0), tmax, 50.0)
        # Gains from the torque limit, capped by each joint's effective inertia: an explicit PD on a light
        # link (a wrist) is numerically unstable once kd * dt / I exceeds ~1, and then it spins and throws
        # the whole robot over. I comes from the joint-space mass matrix at the hold pose (+ armature).
        d0 = mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, d0, 0)
        mujoco.mj_forward(model, d0)
        M = np.zeros((model.nv, model.nv))
        try:
            mujoco.mj_fullM(model, d0, M)       # MuJoCo 4: (model, data, dst)
        except TypeError:
            mujoco.mj_fullM(model, M, d0.qM)    # MuJoCo 3: (model, dst, qM)
        inertia = np.maximum(np.diag(M)[self.dadr], 1e-6)
        dt = float(model.opt.timestep) * max(1, int(round(self.control_dt / model.opt.timestep)))
        kp = np.minimum(tmax / full_torque_error, 0.2 * inertia / dt ** 2)
        kd = np.minimum(np.maximum(2 * 0.7 * np.sqrt(kp * inertia), kp * damping_ratio), 0.5 * inertia / dt)
        self.kp, self.kd = kp, kd
        self.gear = gear
        self.lo = model.actuator_ctrlrange[self.act, 0]
        self.hi = model.actuator_ctrlrange[self.act, 1]
        self.ctrl_limited = model.actuator_ctrllimited[self.act].astype(bool)
        d = mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, d, 0)
        self.target = d.qpos[self.qadr].copy()
        jr = model.jnt_range[model.actuator_trnid[self.act, 0]]
        jl = model.jnt_limited[model.actuator_trnid[self.act, 0]].astype(bool)
        self.target_lo = np.where(jl, jr[:, 0], -np.pi)
        self.target_hi = np.where(jl, jr[:, 1], np.pi)
        self._mode = "stand"
        self.base_target = self.target.copy()

    def reset(self, model, data) -> None:
        self._mode = "stand"
        self.target[:] = self.base_target

    def set_path(self, path_xy, final_yaw=None) -> None:
        if path_xy is not None and len(path_xy):
            self._mode = "stand"  # holds a pose; it cannot walk

    def set_target(self, actuator: int, angle: float) -> None:
        k = np.flatnonzero(self.act == int(actuator))
        if k.size:
            self.target[k[0]] = float(np.clip(angle, self.target_lo[k[0]], self.target_hi[k[0]]))

    def step(self, model, data) -> None:
        q, qd = data.qpos[self.qadr], data.qvel[self.dadr]
        tau = self.kp * (self.target - q) - self.kd * qd
        u = np.where(self.servo, self.target, tau / self.gear)
        u = np.where(self.ctrl_limited, np.clip(u, self.lo, self.hi), u)
        data.ctrl[self.act] = u

    def status(self) -> dict:
        return {"mode": self._mode, "message": "holding the stand pose (PD); this robot has no walking controller"}
