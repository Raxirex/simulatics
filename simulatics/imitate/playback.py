"""Play reference motions on the robot: kinematically (preview) or through the servos with balance feedback.

Kinematic playback writes the reference qpos / qvel straight into MjData (no physics), for previewing a
clip in the GUI. `MotionPreviewController` wraps it in the `Controller` protocol so the live session can
show it like any other controller (it also sends the reference servo targets, so the physics step the
session runs between ticks barely moves the robot).

PD playback is the physical check of a clip: the reference becomes servo targets (with the inverse-
dynamics feed-forward of reference.py) while a balance controller keeps the floating base up:

* mode "upper": WalkController holds the legs and abdomen in its balanced stand (DCM / capture-point
  feedback, model-based), the arms follow the reference. For gestures (wave, reach).
* mode "full": every joint follows the reference, plus two classical balance terms computed against the
  reference's own centre of mass and torso attitude: an ankle strategy (the divergent-component-of-motion
  error xi = c + c'/omega moves the stance ankles' pitch / roll targets, i.e. the centre of pressure) and a
  hip strategy (the torso pitch / roll error moves the stance hips), as in WalkController.
* mode "auto": "upper" when the clip's legs barely move (< 0.1 rad from its first frame), else "full".

`pd_playback` runs this headless and reports whether, and when, the robot falls.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .motion import Motion, heading_pose, planar_between, quat_slerp, rot_z
from .reference import MotionReference, clip_ctrl, servo_params

G = 9.81


def _anchor_for(motion_or_ref, data: mujoco.MjData, qa: int) -> np.ndarray:
    """Planar transform that puts the clip's first heading frame on the robot's current one."""
    ref = motion_or_ref
    q0 = ref.qpos[0]
    h0 = heading_pose(q0[qa:qa + 3], q0[qa + 3:qa + 7])
    hr = heading_pose(data.qpos[qa:qa + 3], data.qpos[qa + 3:qa + 7])
    return planar_between(h0, hr)


class KinematicPlayer:
    """Reference state at any time t, optionally anchored at a planar pose (x, y, yaw)."""

    def __init__(self, model: mujoco.MjModel, motion: Motion, anchor: np.ndarray | None = None,
                 speed: float = 1.0):
        self.model = model
        self.ref = MotionReference(model, motion, 1.0 / motion.fps, feedforward="none")
        self.anchor = None if anchor is None else np.asarray(anchor, float).reshape(3)
        self.speed = float(speed)
        self.qa, self.da = self.ref.qa, self.ref.da

    @property
    def duration(self) -> float:
        return self.ref.duration

    def anchor_at(self, data: mujoco.MjData) -> None:
        """Start the clip where the robot in `data` stands (its xy and yaw)."""
        self.anchor = _anchor_for(self.ref, data, self.qa)

    def state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(qpos, qvel, servo targets) at time t (looping clips repeat, others hold their last frame)."""
        f = float(t) * self.speed / self.ref.dt
        if not self.ref.loop:
            f = min(max(f, 0.0), self.ref.n - 1)
        k = int(np.floor(f))
        a = f - k
        o = None if self.anchor is None else self.anchor[None]
        A = self.ref.frames(np.array([k]), o)
        B = self.ref.frames(np.array([k + 1]), o)
        q = A["qpos"][0] + a * (B["qpos"][0] - A["qpos"][0])
        qa = self.qa
        q[qa + 3:qa + 7] = quat_slerp(A["qpos"][0, qa + 3:qa + 7], B["qpos"][0, qa + 3:qa + 7], a)
        v = A["qvel"][0] + a * (B["qvel"][0] - A["qvel"][0])
        u = A["ctrl"][0] + a * (B["ctrl"][0] - A["ctrl"][0])
        return q, v * self.speed, u

    def apply(self, data: mujoco.MjData, t: float, forward: bool = True) -> None:
        q, v, u = self.state(t)
        data.qpos[:] = q
        data.qvel[:] = v
        data.ctrl[:] = clip_ctrl(self.model, u)
        if forward:
            mujoco.mj_forward(self.model, data)


class MotionPreviewController:
    """Kinematic preview as a `Controller` (name "motion-preview"): every tick writes the reference state."""
    name = "motion-preview"

    def __init__(self, model: mujoco.MjModel, motion: Motion, speed: float = 1.0):
        self.player = KinematicPlayer(model, motion, speed=speed)
        self.motion = motion
        self.control_dt = float(model.opt.timestep) * 4
        self.t = 0.0
        self.pause = 1.0                  # s the last pose is held before a one-shot clip replays
        self.replays = 0

    def reset(self, model, data) -> None:
        self.t = 0.0
        self.player.anchor_at(data)
        self.player.apply(data, 0.0)

    def set_path(self, path_xy, final_yaw=None) -> None:
        pass

    def step(self, model, data) -> None:
        self.t += self.control_dt
        if not self.player.ref.loop and self.t > self.player.duration + self.pause:
            self.t = 0.0                  # a one-shot clip plays again after a short pause (never a frozen pose)
            self.replays += 1
        self.player.apply(data, self.t)

    def status(self) -> dict:
        done = False
        return {"mode": "done" if done else "walk", "message": f"previewing '{self.motion.name}' (kinematic)",
                "motion": self.motion.name, "time_s": round(self.t, 3),
                "phase": float(self.player.ref.phase(int(self.t / self.player.ref.dt)))}


def legs_move(motion: Motion, threshold: float = 0.1) -> bool:
    """True when any leg / abdomen joint moves more than `threshold` rad from the clip's first frame, or the
    root travels more than 5 cm."""
    cols = [i for i, n in enumerate(motion.joint_names)
            if any(k in n for k in ("hip", "knee", "ankle", "abdomen"))]
    moved = np.abs(motion.joint_pos[:, cols] - motion.joint_pos[0, cols]).max() if cols else 0.0
    travel = np.linalg.norm(motion.root_pos[:, :2] - motion.root_pos[0, :2], axis=1).max()
    return bool(moved > threshold or travel > 0.05 or np.linalg.norm(motion.cycle[:2]) > 0.01)


class PDPlaybackController:
    """Reference servo targets + balance feedback (see the module docstring). Implements `Controller`."""
    name = "motion-pd"

    def __init__(self, model: mujoco.MjModel, motion: Motion, mode: str = "auto", k_ankle: float = 0.8,
                 k_torso: float = 1.0):
        self.model, self.motion = model, motion
        self.mode = ("full" if legs_move(motion) else "upper") if mode == "auto" else mode
        if self.mode not in ("upper", "full"):
            raise ValueError("mode must be 'auto', 'upper' or 'full'")
        self.ref = MotionReference(model, motion, 1.0 / motion.fps, feedforward="dynamic")
        dt = float(model.opt.timestep)
        self.control_dt = dt
        self.k_ankle, self.k_torso = float(k_ankle), float(k_torso)
        self.qa, self.da = self.ref.qa, self.ref.da
        self.root = int(model.jnt_bodyid[0])
        qadr, dadr, kp, kv, servo = servo_params(model)
        name = lambda a: model.joint(int(model.actuator_trnid[a, 0])).name  # noqa: E731
        self.act_names = [name(a) if qadr[a] >= 0 else "" for a in range(model.nu)]
        self.arm_act = np.array([a for a, n in enumerate(self.act_names)
                                 if any(k in n for k in ("shoulder", "elbow", "wrist", "gripper"))], int)
        self.idx = {n: a for a, n in enumerate(self.act_names) if n}
        self.walker = None
        if self.mode == "upper":
            from ..control.walker import WalkController
            self.walker = WalkController(model)
            self.control_dt = self.walker.control_dt
        self.t = 0.0
        self.anchor = np.zeros(3)
        self._mode, self._msg = "walk", ""
        self.z_c = float(np.mean(self.ref.com[:, 2]))
        self.omega = float(np.sqrt(G / max(self.z_c, 0.1)))
        self._vf = None

    def reset(self, model, data) -> None:
        self.t = 0.0
        self.anchor = _anchor_for(self.ref, data, self.qa)
        f = self.ref.frames(np.array([0]), self.anchor[None])
        data.qpos[:] = f["qpos"][0]
        data.qvel[:] = f["qvel"][0]
        mujoco.mj_forward(model, data)
        if self.walker is not None:
            self.walker.reset(model, data)        # bent-knee stand over the current feet; arms keep the clip's
        data.ctrl[:] = clip_ctrl(model, f["ctrl"][0])
        self._mode, self._msg, self._vf = "walk", f"PD playback of '{self.motion.name}' ({self.mode})", None

    def set_path(self, path_xy, final_yaw=None) -> None:
        pass

    def _ref_at(self, t: float) -> dict:
        f = t / self.ref.dt
        if not self.ref.loop:
            f = min(f, self.ref.n - 1)
        k = int(np.floor(f))
        a = f - k
        A = self.ref.frames(np.array([k]), self.anchor[None])
        B = self.ref.frames(np.array([k + 1]), self.anchor[None])
        out = {key: A[key][0] + a * (B[key][0] - A[key][0]) for key in A}
        qa = self.qa
        out["qpos"][qa + 3:qa + 7] = quat_slerp(A["qpos"][0, qa + 3:qa + 7], B["qpos"][0, qa + 3:qa + 7], a)
        out["stance"] = self.ref.stance[int(round(f)) % self.ref.n]
        return out

    def step(self, model, data) -> None:
        self.t += self.control_dt
        r = self._ref_at(self.t)
        u = r["ctrl"].copy()
        if self.walker is not None:
            self.walker.step(model, data)            # writes abdomen + leg actuators
            data.ctrl[self.arm_act] = clip_ctrl(model, u)[self.arm_act]
            if self.walker.mode == "fallen":
                self._mode, self._msg = "fallen", "robot fell"
            return
        mujoco.mj_subtreeVel(model, data)
        com = data.subtree_com[self.root]
        vel = data.subtree_linvel[self.root]
        self._vf = vel.copy() if self._vf is None else self._vf + 0.2 * (vel - self._vf)
        xi = (com[:2] + self._vf[:2] / self.omega) - (r["com"][:2] + r["com_vel"][:2] / self.omega)
        R = data.xmat[self.root].reshape(3, 3)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        c, s = np.cos(yaw), np.sin(yaw)
        e_f, e_l = c * xi[0] + s * xi[1], -s * xi[0] + c * xi[1]
        Rm = rot_z(-yaw) @ R
        Rr = rot_z(-yaw) @ _quat_mat(r["qpos"][self.qa + 3:self.qa + 7])
        pitch = float(np.arctan2(-Rm[2, 0], Rm[2, 2]) - np.arctan2(-Rr[2, 0], Rr[2, 2]))
        roll = float(np.arctan2(Rm[2, 1], Rm[2, 2]) - np.arctan2(Rr[2, 1], Rr[2, 2]))
        for i, side in enumerate(("left", "right")):
            if not r["stance"][i]:
                continue
            for jn, val in ((f"{side}_ankle_y", self.k_ankle * e_f), (f"{side}_ankle_x", -self.k_ankle * e_l),
                            (f"{side}_hip_y", self.k_torso * pitch), (f"{side}_hip_x", self.k_torso * roll)):
                if jn in self.idx:
                    u[self.idx[jn]] += val
        data.ctrl[:] = clip_ctrl(model, u)
        if R[2, 2] < 0.5 or data.qpos[self.qa + 2] < 0.55 * r["qpos"][self.qa + 2]:
            self._mode, self._msg = "fallen", "robot fell"

    def status(self) -> dict:
        done = (not self.ref.loop) and self.t >= self.ref.duration and self._mode != "fallen"
        return {"mode": "done" if done else self._mode, "message": self._msg, "motion": self.motion.name,
                "playback": self.mode, "time_s": round(self.t, 3)}


def _quat_mat(q):
    from .motion import quat_to_mat
    return quat_to_mat(q)


def pd_playback(model: mujoco.MjModel, motion: Motion, seconds: float | None = None, mode: str = "auto",
                **kw) -> dict:
    """Run PD playback headless on a fresh MjData. seconds defaults to two periods of a loop (at least 4 s)
    or the clip's duration. Returns {"mode", "fell", "fall_time_s", "seconds", "joint_rmse_deg" (tracked
    1-DOF joints, before any fall), "root_err_m" (mean root position error), "key_err_m" (mean hand / foot
    error relative to the root)}."""
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    ctl = PDPlaybackController(model, motion, mode, **kw)
    ctl.reset(model, d)
    if seconds is None:
        seconds = max(4.0, 2 * motion.duration) if motion.loop else motion.duration
    n_sub = max(1, int(round(ctl.control_dt / model.opt.timestep)))
    qadr, _, _, _, servo = servo_params(model)
    track = np.array(sorted({int(qadr[a]) for a in range(model.nu) if servo[a] and qadr[a] >= 0
                             and "gripper" not in ctl.act_names[a]}), int)
    ee = [ctl.ref.key_names.index(k) for k in ("left_hand", "right_hand", "left_foot", "right_foot")
          if k in ctl.ref.key_names]
    fk = ctl.ref.fk
    jerr, rerr, kerr = [], [], []
    t, fall_t = 0.0, None
    while t < seconds - 1e-9:
        ctl.step(model, d)
        mujoco.mj_step(model, d, n_sub)
        t += ctl.control_dt
        if ctl.status()["mode"] == "fallen" or not np.isfinite(d.qpos).all():
            fall_t = t
            break
        r = ctl._ref_at(ctl.t)
        jerr.append(np.mean((d.qpos[track] - r["qpos"][track]) ** 2))
        rerr.append(float(np.linalg.norm(d.qpos[ctl.qa:ctl.qa + 3] - r["qpos"][ctl.qa:ctl.qa + 3])))
        if ee:
            keys, _ = fk(d.qpos[None].copy())
            kr = r["keys"][ee] - r["qpos"][ctl.qa:ctl.qa + 3]
            km = keys[0, ee] - d.qpos[ctl.qa:ctl.qa + 3]
            kerr.append(float(np.mean(np.linalg.norm(km - kr, axis=1))))
    return {"motion": motion.name, "mode": ctl.mode, "fell": fall_t is not None, "fall_time_s": fall_t,
            "seconds": t, "joint_rmse_deg": float(np.degrees(np.sqrt(np.mean(jerr)))) if jerr else None,
            "root_err_m": float(np.mean(rerr)) if rerr else None, "key_err_m": float(np.mean(kerr)) if kerr else None}
