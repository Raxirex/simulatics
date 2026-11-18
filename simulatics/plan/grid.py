"""2D occupancy grids for navigation.

Two maps of the same area:
  truth   rasterised directly from the static geoms in the MuJoCo model
  lidar   built online from lidar scans (log-odds), i.e. what the robot knows

Planning = inflate obstacles by the robot radius, A*, line-of-sight smoothing
(all three run in the C++ simcore library when it is built).
"""
from __future__ import annotations

import mujoco
import numpy as np

from .. import native


class GridMap:
    def __init__(self, size: float = 24.0, res: float = 0.1):
        self.size, self.res = size, res
        self.n = int(round(size / res))
        self.origin = np.array([-size / 2, -size / 2])
        self.truth = np.zeros((self.n, self.n), np.uint8)
        self.logodds = np.zeros((self.n, self.n), np.float32)

    # ---- coordinates -------------------------------------------------------
    def to_cell(self, xy) -> np.ndarray:
        c = np.floor((np.asarray(xy, float) - self.origin) / self.res).astype(np.int32)
        return np.clip(c, 0, self.n - 1)

    def to_world(self, cells) -> np.ndarray:
        return (np.asarray(cells, float) + 0.5) * self.res + self.origin

    # ---- ground truth ------------------------------------------------------
    def rasterize_truth(self, model: mujoco.MjModel, data: mujoco.MjData, min_height: float = 0.1) -> None:
        """Mark cells covered by static (world-body) geoms taller than min_height."""
        self.truth[:] = 0
        centers = self.to_world(np.stack(np.meshgrid(np.arange(self.n), np.arange(self.n)), axis=-1))  # (h, w, 2)
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != 0 or model.geom_type[g] in (mujoco.mjtGeom.mjGEOM_PLANE,
                                                                   mujoco.mjtGeom.mjGEOM_HFIELD):
                continue
            pos, r = data.geom_xpos[g], float(model.geom_rbound[g])
            if pos[2] + r < min_height:
                continue
            rel = centers - pos[:2]
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                rot = data.geom_xmat[g].reshape(3, 3)
                lx = rel @ rot[:2, 0] / max(np.linalg.norm(rot[:2, 0]), 1e-9)
                ly = rel @ rot[:2, 1] / max(np.linalg.norm(rot[:2, 1]), 1e-9)
                hx, hy = model.geom_size[g, 0], model.geom_size[g, 1]
                mask = (np.abs(lx) <= hx + self.res / 2) & (np.abs(ly) <= hy + self.res / 2)
            else:
                # upright cylinders: footprint is their radius; other shapes: bounding sphere
                foot = model.geom_size[g, 0] if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER else r
                mask = np.hypot(rel[..., 0], rel[..., 1]) <= foot + self.res / 2
            self.truth[mask] = 1

    # ---- lidar mapping -----------------------------------------------------
    def reset_lidar(self) -> None:
        self.logodds[:] = 0

    def integrate_lidar(self, origin: np.ndarray, points: np.ndarray, hit: np.ndarray, h_samples: int,
                        z_min: float = 0.15, z_max: float = 1.8) -> None:
        """Fuse one 3D scan (channels x h_samples rays, row-major) into the 2D map.

        The scan is first collapsed to a planar scan, one range per azimuth: the
        nearest obstacle return in that column, or the farthest free return if
        no channel saw an obstacle. Otherwise rays that pass *over* a low wall
        would wrongly mark its cells as free.
        """
        obstacle = (hit & (points[:, 2] > z_min) & (points[:, 2] < z_max)).reshape(-1, h_samples)
        xy = points[:, :2].reshape(-1, h_samples, 2)
        r = np.linalg.norm(xy - origin[:2], axis=2)
        has = obstacle.any(axis=0)
        row = np.where(has, np.where(obstacle, r, np.inf).argmin(axis=0), np.where(obstacle, -1.0, r).argmax(axis=0))
        ends = xy[row, np.arange(h_samples)]
        raw = np.floor((ends - self.origin) / self.res).astype(np.int32)
        native.integrate_scan(self.logodds, tuple(self.to_cell(origin[:2])), raw, has)

    @property
    def lidar_occ(self) -> np.ndarray:
        return (self.logodds > 0.5).astype(np.uint8)

    def lidar_image(self) -> np.ndarray:
        """uint8 (h, w): 0 unknown, 1 free, 2 occupied."""
        img = np.zeros((self.n, self.n), np.uint8)
        img[self.logodds < -0.5] = 1
        img[self.logodds > 0.5] = 2
        return img

    # ---- planning ----------------------------------------------------------
    def plan(self, start_xy, goal_xy, robot_radius: float = 0.35, source: str = "truth") -> np.ndarray:
        """Returns a world-frame path (n, 2), empty when the goal is unreachable."""
        occ = self.truth if source == "truth" else self.lidar_occ
        cost = native.inflate(occ, robot_radius / self.res)
        s, g = self.to_cell(start_xy), self.to_cell(goal_xy)
        # the robot may already be inside an inflated cell: free its own footprint
        r = 3
        cost[max(0, s[1] - r):s[1] + r + 1, max(0, s[0] - r):s[0] + r + 1] = 0
        cells = native.astar(cost, (int(s[0]), int(s[1])), (int(g[0]), int(g[1])))
        if len(cells) == 0:
            return np.zeros((0, 2))
        cells = native.smooth_path(cost, cells, limit=60)
        path = self.to_world(cells)
        path[0], path[-1] = np.asarray(start_xy, float), np.asarray(goal_xy, float)
        return path

    def blocked(self, path: np.ndarray, robot_radius: float = 0.35, source: str = "truth") -> bool:
        if len(path) < 2:
            return False
        occ = self.truth if source == "truth" else self.lidar_occ
        cost = native.inflate(occ, robot_radius * 0.7 / self.res)
        for a, b in zip(path[:-1], path[1:]):
            n = max(2, int(np.linalg.norm(b - a) / self.res))
            c = self.to_cell(a + (b - a) * np.linspace(0, 1, n)[:, None])
            if (cost[c[1:, 1], c[1:, 0]] == 255).any():
                return True
        return False
