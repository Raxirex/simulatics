"""Path follower for velocity-commanded walkers (RL policies, ES heading policies).

Far from the goal it does pure pursuit (turn towards a look-ahead point, walk
forward). Inside `approach_radius` it switches to a holonomic final approach:
the position error is resolved in the robot frame into forward and sideways
velocity, and the heading is servoed to the requested final yaw. That final
stage is what makes stopping on the exact spot possible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Command:
    vx: float           # forward m/s (robot frame)
    vy: float           # left m/s (robot frame)
    wz: float           # yaw rate rad/s
    aim_yaw: float      # world heading the follower is steering towards
    done: bool
    position_error: float
    heading_error: float  # rad


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class PathFollower:
    def __init__(self, v_max: float = 0.8, vy_max: float = 0.3, wz_max: float = 1.0, lookahead: float = 0.6,
                 approach_radius: float = 0.6, final_tolerance: float = 0.05, yaw_tolerance_deg: float = 5.0,
                 k_pos: float = 1.5, k_yaw: float = 2.0):
        self.v_max, self.vy_max, self.wz_max = v_max, vy_max, wz_max
        self.lookahead, self.approach_radius = lookahead, approach_radius
        self.final_tolerance, self.yaw_tolerance = final_tolerance, np.deg2rad(yaw_tolerance_deg)
        self.k_pos, self.k_yaw = k_pos, k_yaw
        self.path = np.zeros((0, 2))
        self.final_yaw: float | None = None
        self._i = 0

    @property
    def active(self) -> bool:
        return len(self.path) > 0

    def set_path(self, path_xy, final_yaw: float | None = None) -> None:
        self.path = np.zeros((0, 2)) if path_xy is None else np.asarray(path_xy, float).reshape(-1, 2)
        self.final_yaw = final_yaw
        self._i = 0

    def command(self, x: float, y: float, yaw: float) -> Command:
        if not self.active:
            return Command(0.0, 0.0, 0.0, yaw, True, 0.0, 0.0)
        pos = np.array([x, y])
        goal = self.path[-1]
        err = goal - pos
        dist = float(np.linalg.norm(err))
        want_yaw = self.final_yaw
        if dist < self.approach_radius:
            # final approach: holonomic, robot frame
            c, s = np.cos(yaw), np.sin(yaw)
            ex, ey = c * err[0] + s * err[1], -s * err[0] + c * err[1]
            yaw_target = want_yaw if want_yaw is not None else (yaw if dist < 0.15 else float(np.arctan2(err[1], err[0])))
            eyaw = _wrap(yaw_target - yaw)
            done = dist < self.final_tolerance and (want_yaw is None or abs(eyaw) < self.yaw_tolerance)
            if done:
                return Command(0.0, 0.0, 0.0, yaw_target, True, dist, eyaw)
            scale = min(1.0, dist / self.approach_radius) * 0.6 + 0.4
            vx = float(np.clip(self.k_pos * ex, -0.3, self.v_max * scale))
            vy = float(np.clip(self.k_pos * ey, -self.vy_max, self.vy_max))
            wz = float(np.clip(self.k_yaw * eyaw, -self.wz_max, self.wz_max))
            return Command(vx, vy, wz, yaw_target, False, dist, eyaw)
        # pure pursuit along the path
        while self._i < len(self.path) - 1 and np.linalg.norm(self.path[self._i] - pos) < self.lookahead:
            self._i += 1
        aim = self._lookahead_point(pos)
        aim_yaw = float(np.arctan2(aim[1] - y, aim[0] - x))
        eyaw = _wrap(aim_yaw - yaw)
        vx = self.v_max * max(0.0, float(np.cos(eyaw))) ** 2 * min(1.0, dist / 1.5 + 0.3)
        wz = float(np.clip(self.k_yaw * eyaw, -self.wz_max, self.wz_max))
        return Command(float(vx), 0.0, wz, aim_yaw, False, dist, _wrap((want_yaw or aim_yaw) - yaw))

    def _lookahead_point(self, pos: np.ndarray) -> np.ndarray:
        for j in range(self._i, len(self.path)):
            if np.linalg.norm(self.path[j] - pos) >= self.lookahead:
                return self.path[j]
        return self.path[-1]
