"""Locomotion task: batched observation / action / reward maths.

Everything here works on arrays with a leading batch dimension so the same
code serves the live GUI simulation (batch of 1) and the parallel trainer
(batch of hundreds). It only needs a model whose root is a free joint
(qpos[0:7], qvel[0:6]), so it works for the built-in parametric humanoid and
for imported robots (e.g. Unitree G1/H1 from MuJoCo Menagerie).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import mujoco
import numpy as np


@dataclass
class TaskConfig:
    frame_skip: int = 4            # physics steps per control step
    action_scale: float = 0.6      # rad around the nominal pose (position servos)
    w_vel: float = 2.0             # reward per m/s towards the target heading
    w_alive: float = 1.0           # reward per second for staying up
    w_ctrl: float = 0.05           # penalty on squared action
    w_lateral: float = 0.5         # penalty per m/s of sideways drift
    v_max: float = 2.5             # velocity reward saturates here
    min_height_frac: float = 0.6   # fall = root below this fraction of standing height
    min_up: float = 0.35           # fall = torso tilted beyond this (cos of tilt)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TaskConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


STATE_SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS


def has_free_root(model: mujoco.MjModel) -> bool:
    return model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE


class LocoTask:
    def __init__(self, model: mujoco.MjModel, cfg: TaskConfig | None = None):
        if not has_free_root(model):
            raise ValueError("locomotion task needs a model whose first joint is a free joint")
        self.cfg = cfg or TaskConfig()
        self.nq, self.nv, self.nu, self.na = model.nq, model.nv, model.nu, model.na
        self.nstate = mujoco.mj_stateSize(model, STATE_SPEC)
        self.dt = float(model.opt.timestep) * self.cfg.frame_skip
        # Only the robot's own joints are observed: free bodies in the world (graspable props)
        # also add qpos/qvel entries, and must not change the policy's input.
        root = model.body_rootid[model.jnt_bodyid[0]]
        q_idx, v_idx = [], []
        for j in range(1, model.njnt):
            if model.body_rootid[model.jnt_bodyid[j]] != root:
                continue
            nq_j = {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4}.get(int(model.jnt_type[j]), 1)
            nv_j = {int(mujoco.mjtJoint.mjJNT_FREE): 6, int(mujoco.mjtJoint.mjJNT_BALL): 3}.get(int(model.jnt_type[j]), 1)
            q_idx += range(model.jnt_qposadr[j], model.jnt_qposadr[j] + nq_j)
            v_idx += range(model.jnt_dofadr[j], model.jnt_dofadr[j] + nv_j)
        self.q_idx, self.v_idx = np.array(q_idx, int), np.array(v_idx, int)
        self.obs_dim = 1 + 3 + 3 + 3 + len(self.q_idx) + len(self.v_idx) + 2
        self.act_dim = self.nu

        # nominal pose: first keyframe if the model has one, else defaults
        d = mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, d, 0)
        else:
            mujoco.mj_resetData(model, d)
        self.ctrl0 = d.ctrl.copy()
        self.z0 = float(d.qpos[2])

        lo, hi = model.actuator_ctrlrange[:, 0].copy(), model.actuator_ctrlrange[:, 1].copy()
        limited = model.actuator_ctrllimited.astype(bool)
        lo[~limited], hi[~limited] = -1.0, 1.0
        self.ctrl_lo, self.ctrl_hi = lo, hi
        is_servo = model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE
        half = (hi - lo) / 2
        self.ctrl_scale = np.where(is_servo, np.minimum(self.cfg.action_scale, half * 2), half)

    # ---- state helpers ----------------------------------------------------
    @staticmethod
    def initial_state(model: mujoco.MjModel, data: mujoco.MjData | None = None) -> np.ndarray:
        d = data or mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, d, 0)
        else:
            mujoco.mj_resetData(model, d)
        s = np.zeros(mujoco.mj_stateSize(model, STATE_SPEC))
        mujoco.mj_getState(model, d, s, STATE_SPEC)
        return s

    def split(self, state: np.ndarray):
        """state (B, nstate) -> qpos (B, nq), qvel (B, nv) views."""
        return state[:, 1:1 + self.nq], state[:, 1 + self.nq:1 + self.nq + self.nv]

    # ---- observation ------------------------------------------------------
    @staticmethod
    def yaw_up(qpos: np.ndarray):
        w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        # world up expressed in the body frame = third row of the rotation matrix
        up = np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1)
        return yaw, up

    def obs(self, qpos: np.ndarray, qvel: np.ndarray, target_yaw: np.ndarray) -> np.ndarray:
        yaw, up = self.yaw_up(qpos)
        c, s = np.cos(yaw), np.sin(yaw)
        vx, vy = qvel[:, 0], qvel[:, 1]
        v_fwd, v_lat = c * vx + s * vy, -s * vx + c * vy
        rel = target_yaw - yaw
        return np.concatenate([
            qpos[:, 2:3] / self.z0,
            up,
            np.stack([v_fwd, v_lat, qvel[:, 2]], axis=1),
            qvel[:, 3:6],
            qpos[:, self.q_idx],
            qvel[:, self.v_idx] * 0.1,
            np.stack([np.cos(rel), np.sin(rel)], axis=1),
        ], axis=1)

    # ---- action -----------------------------------------------------------
    def ctrl(self, action: np.ndarray) -> np.ndarray:
        """Policy output in [-1, 1] -> actuator controls."""
        return np.clip(self.ctrl0 + action * self.ctrl_scale, self.ctrl_lo, self.ctrl_hi)

    # ---- reward -----------------------------------------------------------
    def reward(self, qpos0: np.ndarray, qpos1: np.ndarray, action: np.ndarray, target_yaw: np.ndarray):
        """Reward and termination for one control step qpos0 -> qpos1."""
        cfg = self.cfg
        dxy = (qpos1[:, 0:2] - qpos0[:, 0:2]) / self.dt
        ct, st = np.cos(target_yaw), np.sin(target_yaw)
        v_t = dxy[:, 0] * ct + dxy[:, 1] * st
        v_l = -dxy[:, 0] * st + dxy[:, 1] * ct
        _, up = self.yaw_up(qpos1)
        r = self.dt * (cfg.w_vel * np.minimum(v_t, cfg.v_max) + cfg.w_alive
                       - cfg.w_lateral * np.abs(v_l)
                       - cfg.w_ctrl * np.mean(action * action, axis=1))
        done = ((qpos1[:, 2] < cfg.min_height_frac * self.z0) | (up[:, 2] < cfg.min_up)
                | ~np.isfinite(qpos1).all(axis=1))
        return np.where(np.isfinite(r), r, 0.0), done
