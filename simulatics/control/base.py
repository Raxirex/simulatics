"""Controller contract: anything that can make the robot stand and follow a planned path.

The live session ticks one controller at a time:

    ctrl.reset(model, data)                    after (re)building or resetting the sim
    ctrl.set_path(path_xy, final_yaw)          from the planner; None clears it (stand still)
    every ctrl.control_dt seconds:
        ctrl.step(model, data)                 writes data.ctrl for the next physics steps
    ctrl.status()                              for the GUI / LLM

Implementations: ManualController (here), PolicyController (ES/GA heading policies,
here), WalkController (model-based, control/walker.py), RLController (PPO
velocity-tracking policy, train/ppo.py).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import mujoco
import numpy as np

from .follower import PathFollower


@runtime_checkable
class Controller(Protocol):
    name: str
    control_dt: float  # seconds between step() calls (a multiple of model.opt.timestep)

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None: ...

    def set_path(self, path_xy: np.ndarray | None, final_yaw: float | None = None) -> None:
        """World-frame waypoints (n, 2) ending at the goal; final_yaw (rad) is the heading to end with."""

    def step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None: ...

    def status(self) -> dict:
        """At least {"mode": "stand"|"walk"|"done"|"fallen", "message": str}.
        Optional: "goal_error_m", "heading_error_deg", "footsteps": [[x, y, yaw, side], ...]."""


def base_pose(data: mujoco.MjData) -> tuple[float, float, float]:
    """(x, y, yaw) of the free-floating root body."""
    w, x, y, z = data.qpos[3:7]
    return float(data.qpos[0]), float(data.qpos[1]), float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class ManualController:
    """Holds the joint targets set from the Motors tab."""
    name = "manual"

    def __init__(self, model: mujoco.MjModel, ctrl0: np.ndarray | None = None):
        self.control_dt = float(model.opt.timestep) * 4
        # shares the caller's array, so edits from the Motors tab apply immediately
        self.target = ctrl0 if isinstance(ctrl0, np.ndarray) else np.zeros(model.nu)

    def reset(self, model, data) -> None:
        pass

    def set_path(self, path_xy, final_yaw=None) -> None:
        pass

    def step(self, model, data) -> None:
        data.ctrl[:] = self.target

    def status(self) -> dict:
        return {"mode": "stand", "message": "manual joint targets (no walking)"}


class PolicyController:
    """ES/GA policies from train/evolve.py: they only take a target heading, so a path follower steers them."""
    name = "es-policy"

    def __init__(self, model: mujoco.MjModel, policy, task):
        self.policy, self.task = policy, task
        self.control_dt = task.dt
        self.follower = PathFollower(final_tolerance=0.35)
        self.heading = 0.0  # used when there is no path
        self.limit = max(0.3, float(policy.meta.get("heading_range", 0.5)))
        self._mode, self._msg = "walk", ""

    def reset(self, model, data) -> None:
        self._mode, self._msg = "walk", ""

    def set_path(self, path_xy, final_yaw=None) -> None:
        self.follower.set_path(path_xy, final_yaw)
        self._mode = "walk"

    def step(self, model, data) -> None:
        x, y, yaw = base_pose(data)
        target = self.heading
        if self.follower.active:
            cmd = self.follower.command(x, y, yaw)
            if cmd.done:
                data.ctrl[:] = self.task.ctrl0  # arrived: try to stand
                self._mode, self._msg = "done", f"goal reached ({cmd.position_error:.2f} m)"
                return
            target = cmd.aim_yaw
        # never ask for a sharper turn than the policy saw in training
        rel = float(np.clip(wrap(target - yaw), -self.limit, self.limit))
        qpos, qvel = data.qpos[None].copy(), data.qvel[None].copy()
        obs = self.task.obs(qpos, qvel, np.array([yaw + rel]))
        data.ctrl[:] = self.task.ctrl(self.policy.act(obs))[0]

    def status(self) -> dict:
        return {"mode": self._mode, "message": self._msg}
