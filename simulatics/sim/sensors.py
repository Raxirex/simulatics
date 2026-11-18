"""Simulated sensors: spinning lidar (ray casting) and cameras (RGB / depth)."""
from __future__ import annotations

import io
from dataclasses import dataclass, asdict

import mujoco
import numpy as np


@dataclass
class LidarConfig:
    enabled: bool = True
    h_samples: int = 180      # rays per revolution
    channels: int = 8         # vertical layers
    v_min: float = -25.0      # deg
    v_max: float = 5.0        # deg
    max_range: float = 12.0   # m
    rate_hz: float = 10.0

    def to_dict(self) -> dict:
        return asdict(self)


class Lidar:
    """Casts h_samples x channels rays from a site with MuJoCo's mj_multiRay (C, BVH-accelerated)."""

    def __init__(self, model: mujoco.MjModel, cfg: LidarConfig, robot_root_body: int = 1):
        self.cfg = cfg
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
        self.root = robot_root_body
        az = np.linspace(0, 2 * np.pi, cfg.h_samples, endpoint=False)
        el = np.deg2rad(np.linspace(cfg.v_min, cfg.v_max, max(1, cfg.channels)))
        az, el = np.meshgrid(az, el)
        self.local = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1).reshape(-1, 3)
        self.n = len(self.local)
        self._geomid = np.zeros(self.n, np.int32)
        self._dist = np.zeros(self.n)
        # Our humanoid keeps robot geoms in group 0, obstacles in 1, floor in 2,
        # so the lidar can ignore the robot's own body exactly. Imported models
        # fall back to excluding the root body only.
        own = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0 and model.geom_group[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")] == 2
        self.groups = np.array([0, 1, 1, 1, 1, 1], np.uint8) if own else None

    def scan(self, model: mujoco.MjModel, data: mujoco.MjData):
        """Returns (origin (3,), points (n, 3) world, dist (n,), hit (n,) bool)."""
        if self.site >= 0:
            origin = data.site_xpos[self.site].copy()
            rot = data.site_xmat[self.site].reshape(3, 3)
        else:
            origin = data.xpos[self.root] + np.array([0, 0, 0.5])
            rot = data.xmat[self.root].reshape(3, 3)
        dirs = self.local @ rot.T
        mujoco.mj_multiRay(model, data, origin, dirs.reshape(-1), self.groups, True,
                           -1 if self.groups is not None else self.root,
                           self._geomid, self._dist, None, self.n, self.cfg.max_range)
        hit = self._dist >= 0
        dist = np.where(hit, self._dist, self.cfg.max_range)
        return origin, origin + dirs * dist[:, None], dist, hit


class CameraRig:
    """Offscreen renderer. Must be created, used and closed on one thread (it owns a GL context)."""

    def __init__(self, model: mujoco.MjModel, width: int = 320, height: int = 240):
        self.renderer = mujoco.Renderer(model, height, width)
        self.names = [model.camera(i).name for i in range(model.ncam)]

    def rgb(self, data: mujoco.MjData, camera: str | int = -1) -> np.ndarray:
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera=camera)
        return self.renderer.render()

    def depth(self, data: mujoco.MjData, camera: str | int = -1) -> np.ndarray:
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=camera)
        d = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return d

    def jpeg(self, data: mujoco.MjData, camera: str | int = -1, depth: bool = False, max_depth: float = 8.0) -> bytes:
        from PIL import Image
        if depth:
            d = np.clip(self.depth(data, camera) / max_depth, 0, 1)
            img = Image.fromarray((255 * (1 - d)).astype(np.uint8))
        else:
            img = Image.fromarray(self.rgb(data, camera))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    def close(self) -> None:
        self.renderer.close()
