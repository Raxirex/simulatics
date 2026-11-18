"""The live simulation exposed through `RobotInterface`, for LLM pilots.

Pilots run on their own thread. Rendering needs the OpenGL context owned by the
simulation thread, so camera / segmentation requests are queued here and
served by `SimSession._sensors` (see `serve_render_requests`).
"""
from __future__ import annotations

import io
import queue
import threading
import time

import mujoco
import numpy as np

from ..control import base_pose
from ..llm.robot import RobotInterface


class _RenderRequest:
    def __init__(self, camera: str, width: int, height: int, segmentation: bool):
        self.camera, self.width, self.height, self.segmentation = camera, width, height, segmentation
        self.result = None
        self.error: str | None = None
        self.done = threading.Event()


class SessionRobot(RobotInterface):
    def __init__(self, session):
        self.s = session
        self.requests: "queue.Queue[_RenderRequest]" = queue.Queue()
        self.cancelled = threading.Event()
        self._renderers: dict[tuple[int, int, int], mujoco.Renderer] = {}
        self._renderer_version = -1

    # ------------------------------------------------------------- render plumbing (sim thread side)
    def serve_render_requests(self) -> None:
        """Called by the simulation thread: render everything that pilots asked for."""
        while True:
            try:
                req = self.requests.get_nowait()
            except queue.Empty:
                return
            try:
                req.result = self._render(req)
            except Exception as e:  # noqa: BLE001 - reported to the pilot as text
                req.error = f"{type(e).__name__}: {e}"
            req.done.set()

    def close(self) -> None:
        """Free the GL renderers. Must run on the simulation thread (it owns their GL contexts)."""
        for r in self._renderers.values():
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass
        self._renderers = {}
        while True:  # unblock anyone still waiting for a frame
            try:
                req = self.requests.get_nowait()
            except queue.Empty:
                break
            req.error = "simulation stopped"
            req.done.set()

    def _renderer(self, w: int, h: int) -> mujoco.Renderer:
        if self._renderer_version != self.s.scene_version:
            for r in self._renderers.values():
                r.close()
            self._renderers, self._renderer_version = {}, self.s.scene_version
        key = (w, h, self.s.scene_version)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.s.model, h, w)
        return self._renderers[key]

    def _render(self, req: _RenderRequest):
        m, d = self.s.model, self.s.data
        r = self._renderer(req.width, req.height)
        cam = req.camera if req.camera in [m.camera(i).name for i in range(m.ncam)] else -1
        if req.segmentation:
            r.enable_segmentation_rendering()
            r.update_scene(d, camera=cam)
            seg = r.render().copy()
            r.disable_segmentation_rendering()
            return seg
        r.update_scene(d, camera=cam)
        return r.render().copy()

    def _request(self, camera: str, width: int, height: int, segmentation: bool, timeout: float = 5.0):
        req = _RenderRequest(camera, width, height, segmentation)
        self.requests.put(req)
        if not req.done.wait(timeout):
            raise TimeoutError("render request timed out (is the simulation thread alive?)")
        if req.error:
            raise RuntimeError(req.error)
        return req.result

    # ------------------------------------------------------------- state
    def pose(self) -> dict:
        with self.s.lock:
            d, task = self.s.data, self.s.task
            x, y, yaw = base_pose(d)
            up = True
            if task is not None:
                _, u = task.yaw_up(d.qpos[None])
                up = bool(u[0, 2] > 0.7 and d.qpos[2] > 0.75 * task.z0)
            return {"x": round(x, 3), "y": round(y, 3), "yaw_deg": round(float(np.rad2deg(yaw)), 1),
                    "height": round(float(d.qpos[2]), 3), "speed": round(float(np.hypot(d.qvel[0], d.qvel[1])), 2),
                    "upright": up, "time": round(float(d.time), 2)}

    def nav_status(self) -> dict:
        with self.s.lock:
            nav, c = self.s.nav, self.s.controller
            st = c.status() if c is not None else {}
            goal = nav.goals[0] if nav.goals else nav.last_goal
            return {"state": nav.state, "goal": None if goal is None else [round(float(v), 3) for v in goal[0]],
                    "goal_yaw_deg": None if goal is None or goal[1] is None else round(float(np.rad2deg(goal[1])), 1),
                    "error_m": st.get("goal_error_m"), "message": nav.message or st.get("message", ""),
                    "controller": getattr(c, "name", "none")}

    def sim_time(self) -> float:
        return float(self.s.data.time)

    # ------------------------------------------------------------- senses
    def cameras(self) -> list[str]:
        m = self.s.model
        return [m.camera(i).name for i in range(m.ncam)]

    def camera_jpeg(self, camera: str = "head_cam", width: int = 480, height: int = 360) -> bytes:
        from PIL import Image
        img = self._request(camera, int(width), int(height), segmentation=False)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def lidar_summary(self, sectors: int = 12) -> list[dict]:
        with self.s.lock:
            m, d = self.s.model, self.s.data
            origin, pts, dist, hit = self.s.lidar.scan(m, d)
            x, y, yaw = base_pose(d)
        ok = hit & (pts[:, 2] > 0.1)
        rel = pts[ok, :2] - origin[:2]
        bearing = (np.arctan2(rel[:, 1], rel[:, 0]) - yaw + np.pi) % (2 * np.pi) - np.pi
        rng = np.hypot(rel[:, 0], rel[:, 1])
        width = 2 * np.pi / sectors
        out = []
        for k in range(sectors):
            centre = -np.pi + (k + 0.5) * width
            sel = np.abs((bearing - centre + np.pi) % (2 * np.pi) - np.pi) <= width / 2
            out.append({"bearing_deg": round(float(np.rad2deg(centre)), 0),
                        "nearest_m": round(float(rng[sel].min()), 2) if sel.any() else None})
        return out

    def objects(self) -> list[dict]:
        """World objects and props with names; `visible` from a head-camera segmentation render."""
        visible_geoms: set[int] = set()
        for cam in ("head_cam", "work_cam"):
            if cam not in self.cameras():
                continue
            try:
                seg = self._request(cam, 160, 120, segmentation=True)
            except Exception:  # noqa: BLE001 - visibility is best-effort
                continue
            ids, types = seg[..., 0], seg[..., 1]
            geom_ids, counts = np.unique(ids[types == int(mujoco.mjtObj.mjOBJ_GEOM)], return_counts=True)
            visible_geoms |= {int(g) for g, n in zip(geom_ids, counts) if g >= 0 and n >= 4}
        with self.s.lock:
            m, d = self.s.model, self.s.data
            x, y, yaw = base_pose(d)
            out = []
            for i, o in enumerate(self.s.obstacles):
                gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"obj_{i}")
                if gid < 0:
                    continue
                out.append(self._describe(o.get("name", f"object {i}"), o.get("colour", ""), o.get("shape", "box"),
                                          d.geom_xpos[gid], 2 * max(o["size"][:2]), gid in visible_geoms, x, y, yaw))
            for i, p in enumerate(getattr(self.s, "props", []) or []):
                gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"prop_{i}")
                if gid < 0:
                    continue
                o = self._describe(p.get("name", f"prop {i}"), p.get("colour", ""), p.get("shape", "box"),
                                   d.geom_xpos[gid], 2 * float(np.max(p.get("size", [0.03]))), gid in visible_geoms,
                                   x, y, yaw)
                o["z"] = round(float(d.geom_xpos[gid][2]), 3)
                o["graspable"] = True
                out.append(o)
            return out

    @staticmethod
    def _describe(name, colour, shape, pos, size, visible, x, y, yaw) -> dict:
        dx, dy = float(pos[0]) - x, float(pos[1]) - y
        bearing = (np.arctan2(dy, dx) - yaw + np.pi) % (2 * np.pi) - np.pi
        return {"name": name, "colour": colour, "shape": shape, "x": round(float(pos[0]), 3),
                "y": round(float(pos[1]), 3), "size_m": round(float(size), 2),
                "distance_m": round(float(np.hypot(dx, dy)), 2),
                "bearing_deg": round(float(np.rad2deg(bearing)), 1), "visible": bool(visible)}

    # ------------------------------------------------------------- walking
    def walk_to(self, x: float, y: float, yaw_deg: float | None = None) -> str:
        with self.s.lock:
            if not self.s.running:
                self.s.running = True
            self.s.nav.set_goal([float(x), float(y)], None if yaw_deg is None else np.deg2rad(float(yaw_deg)))
        return f"walking to ({x:.2f}, {y:.2f})" + ("" if yaw_deg is None else f" facing {yaw_deg:.0f} deg")

    def walk_relative(self, forward_m: float, left_m: float = 0.0, turn_deg: float = 0.0) -> str:
        p = self.pose()
        yaw = np.deg2rad(p["yaw_deg"])
        gx = p["x"] + np.cos(yaw) * forward_m - np.sin(yaw) * left_m
        gy = p["y"] + np.sin(yaw) * forward_m + np.cos(yaw) * left_m
        return self.walk_to(float(gx), float(gy), p["yaw_deg"] + float(turn_deg))

    def stop(self) -> str:
        with self.s.lock:
            self.s.nav.clear()
        return "stopping"

    def wait(self, seconds: float) -> None:
        t_end = self.sim_time() + max(0.0, float(seconds))
        wall_end = time.monotonic() + max(2.0, float(seconds) * 4 + 5)  # never hang if the sim is slow
        time.sleep(0.05)
        while not self.cancelled.is_set() and time.monotonic() < wall_end:
            if not self.s.running or self.sim_time() >= t_end:
                return
            with self.s.lock:
                walking = self.s.nav.state == "walking"
                arms = self.s.arms
                moving = bool(arms is not None and arms.busy)
            if not walking and not moving:
                return
            time.sleep(0.05)

    # ------------------------------------------------------------- arms
    def _arms(self):
        arms = getattr(self.s, "arms", None)
        if arms is None:
            raise RuntimeError("this robot has no controllable arms/hands (enable Arms and Hands in the Model tab)")
        if not self.s.running:
            self.s.running = True
        return arms

    def reach(self, side: str, x: float, y: float, z: float, approach: list[float] | None = None) -> str:
        with self.s.lock:
            kw = {"approach": [float(v) for v in approach]} if approach else {}
            return self._arms().command(side, "reach", target=[float(x), float(y), float(z)], **kw)

    def grasp(self, side: str) -> str:
        with self.s.lock:
            return self._arms().command(side, "grasp")

    def release(self, side: str) -> str:
        with self.s.lock:
            return self._arms().command(side, "release")

    def point_at(self, side: str, x: float, y: float, z: float) -> str:
        with self.s.lock:
            return self._arms().command(side, "point_at", target=[float(x), float(y), float(z)])

    def arm_gesture(self, side: str, gesture: str) -> str:
        with self.s.lock:
            return self._arms().command(side, gesture)

    def arm_status(self) -> dict:
        arms = getattr(self.s, "arms", None)
        if arms is None:
            return {}
        with self.s.lock:
            return arms.status()
