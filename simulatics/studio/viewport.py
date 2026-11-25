"""The 3D viewport: MuJoCo's own renderer inside a Qt GL widget, with camera navigation and scene tools.

MuJoCo renders into its offscreen framebuffer, which is then blitted into the widget (a QOpenGLWidget draws
into a framebuffer of its own, not the window's). The session lock is held only while the scene is copied
out of the simulation, never while drawing.
"""
from __future__ import annotations

import time

import mujoco
import numpy as np
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget

GL_READ_FRAMEBUFFER, GL_DRAW_FRAMEBUFFER, GL_FRAMEBUFFER = 0x8CA8, 0x8CA9, 0x8D40
GL_COLOR_BUFFER_BIT, GL_NEAREST = 0x4000, 0x2600

TOOLS = {
    "select": ("Select", "Q", "Click the floor: the robot walks there. Click a part or an object to select it; drag an object to move it."),
    "orbit": ("Orbit", "W", "Drag to orbit. Right-drag orbits and middle-drag pans with any tool; wheel zooms."),
    "goal": ("Walk to", "G", "Click the floor: the robot plans a path and walks there. Shift-click adds a waypoint."),
    "push": ("Push", "P", "Click a part to shove it away from the camera."),
    "grab": ("Grab", "B", "Drag a part to pull it with a spring. Shift-drag moves it horizontally."),
    "place": ("Place accessory", "A", "Click the robot to mount the chosen accessory there."),
    "measure": ("Measure", "M", "Click two points to measure the distance between them."),
    "build": ("Add object", "O", "Click the floor to drop the chosen object there. It can then be selected, dragged, resized or deleted."),
}
VIEWS = {"Perspective": (135, -20), "Front": (180, 0), "Back": (0, 0), "Left": (-90, 0), "Right": (90, 0),
         "Top": (180, -89.9)}
FLAGS = {
    "Contact points": mujoco.mjtVisFlag.mjVIS_CONTACTPOINT, "Contact forces": mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,
    "Joints": mujoco.mjtVisFlag.mjVIS_JOINT, "Actuators": mujoco.mjtVisFlag.mjVIS_ACTUATOR,
    "Centre of mass": mujoco.mjtVisFlag.mjVIS_COM, "Inertia boxes": mujoco.mjtVisFlag.mjVIS_INERTIA,
    "Transparent": mujoco.mjtVisFlag.mjVIS_TRANSPARENT, "Cameras": mujoco.mjtVisFlag.mjVIS_CAMERA,
}


def configure_gl() -> None:
    """MuJoCo draws with legacy OpenGL: ask for a compatibility context. Call before QApplication exists."""
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)


class Viewport(QOpenGLWidget):
    bodyPicked = Signal(int)
    message = Signal(str)
    sceneChanged = Signal()       # the model was rebuilt (after placing an accessory, for example)
    toolChanged = Signal(str)
    objectPicked = Signal(str)    # group id of a scene object ("" = none)
    _move_takes_scene: bool | None = None

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.sess = session
        self.setMinimumSize(240, 160)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.cam.azimuth, self.cam.elevation, self.cam.distance = 135, -20, 4.5
        self.cam.lookat[:] = (0, 0, 0.8)
        self.scn = None
        self.con = None
        self._model = None            # the model the scene / context were made for
        self._version = -1
        self._off = (0, 0)
        self.tool = "select"
        self.place_type = "lidar"
        self.place_snap = 0.01
        self.push_force = 150.0
        self.selected = -1
        self.show = {"Lidar points": True, "Path and goals": True, "Overlay text": True, "Collision shapes": False,
                     "Shadows": True, "Reflections": True}
        self.auto_quality = True      # drop reflections, then shadows, when the frame rate is too low
        self._slow = 0
        self.follow = False
        self.camera_name = "Free"
        self.measure_pts: list[np.ndarray] = []
        self.touch_walk = True        # Select tool: a click on the floor is a walking goal
        self.build_kind, self.build_scale, self.build_colour, self.build_yaw = "box", 1.0, "blue", 0.0
        self.build_snap = 0.05
        self.selected_object = ""
        self._drag_object = ""
        self._mark = None
        self._pushes: list[tuple[int, np.ndarray, float]] = []   # (body, force, end sim time)
        self._grab_body = -1
        self._last = None
        self._press = None
        self._frames, self._fps, self._fps_t = 0, 0.0, time.perf_counter()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

    # ---- GL lifetime ------------------------------------------------------
    def initializeGL(self) -> None:
        self.con = self.scn = None    # a new GL context (the widget moved to another window): old resources are gone
        self.context().aboutToBeDestroyed.connect(self._free)

    def _free(self) -> None:
        if self.con is not None:
            self.makeCurrent()
            self.con.free()
            self.doneCurrent()
        self.con = self.scn = None

    def _ensure(self, w: int, h: int) -> None:
        s = self.sess
        if (self.con is not None and self._version == s.scene_version and self._model is s.model
                and self._off[0] >= w and self._off[1] >= h):
            return
        m = s.model
        # the offscreen buffer must cover the widget; never shrink it (the session's camera renderers share the model)
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), w)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), h)
        m.vis.rgba.fog[:] = (0.09, 0.10, 0.12, 1.0)      # background where there is no skybox
        if self.con is not None:
            self.con.free()
        self.scn = mujoco.MjvScene(m, maxgeom=20000)
        self.con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_100)
        self._off = (int(m.vis.global_.offwidth), int(m.vis.global_.offheight))
        changed = self._model is not None and self._model is not m
        self._model, self._version = m, s.scene_version
        if self.selected >= m.nbody:
            self.selected = -1
        self._grab_body, self._pushes = -1, []
        self.pert.active = 0
        if self.auto_quality and changed:
            self.show["Shadows"] = self.show["Reflections"] = True   # a new model gets a fresh chance
            self._slow = 0
        if self.camera_name not in ("Free", "Follow") and self.camera_name not in self.camera_names():
            self.set_camera("Free")
        if changed:
            self.sceneChanged.emit()

    # ---- drawing ----------------------------------------------------------
    def _tick(self) -> None:
        if self.isVisible():
            self.update()

    def paintGL(self) -> None:
        r = self.devicePixelRatioF()
        w, h = max(1, int(self.width() * r)), max(1, int(self.height() * r))
        s = self.sess
        with s.lock:
            self._ensure(w, h)
            m, d = s.model, s.data
            self._apply_forces(m, d)
            self.pert.select = self.selected if self.selected > 0 else 0
            self.opt.geomgroup[3] = 1 if (self.show["Collision shapes"] and s.custom_path) else 0
            mujoco.mjv_updateScene(m, d, self.opt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
            lidar = s.lidar_points if self.show["Lidar points"] and s.lidar_cfg.enabled else None
            path = s.nav.path.copy() if self.show["Path and goals"] else None
            goals = [g.copy() for g, _ in s.nav.goals] if self.show["Path and goals"] else []
            hud = (f"t {d.time:7.2f} s   {self._fps:4.0f} fps", f"{getattr(s.controller, 'name', '')}")
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1 if self.show["Shadows"] else 0
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1 if self.show["Reflections"] else 0
        self._overlays(path, goals)
        vp = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
        mujoco.mjr_render(vp, self.scn, self.con)
        if self.show["Overlay text"]:
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, vp, hud[0], hud[1],
                               self.con)
            if self.tool == "measure" and len(self.measure_pts) == 2:
                dist = float(np.linalg.norm(self.measure_pts[1] - self.measure_pts[0]))
                dz = float(self.measure_pts[1][2] - self.measure_pts[0][2])
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPRIGHT, vp,
                                   f"{dist:.3f} m", f"height {dz:+.3f} m", self.con)
        f = self.context().extraFunctions()
        f.glBindFramebuffer(GL_READ_FRAMEBUFFER, self.con.offFBO)
        f.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.defaultFramebufferObject())
        f.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST)
        f.glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())
        if lidar is not None and len(lidar):
            self._draw_points(lidar)
        self._frames += 1
        now = time.perf_counter()
        if now - self._fps_t >= 0.5:
            self._fps, self._frames, self._fps_t = self._frames / (now - self._fps_t), 0, now
            self._slow = self._slow + 1 if self._fps < 15 else 0
            if self.auto_quality and self._slow >= 3 and (self.show["Reflections"] or self.show["Shadows"]):
                which = "Reflections" if self.show["Reflections"] else "Shadows"
                self.show[which], self._slow = False, 0
                self.message.emit(f"{which} switched off to keep the viewport responsive "
                                  f"(View > Show in Viewport turns them back on).")

    def project(self, pts: np.ndarray):
        """World points (n, 3) -> widget pixel coordinates (n, 2) and a mask of those in front of the camera."""
        c0, c1 = self.scn.camera[0], self.scn.camera[1]
        pos = (np.array(c0.pos) + np.array(c1.pos)) / 2
        fwd, up = np.array(c0.forward, float), np.array(c0.up, float)
        right = np.cross(fwd, up)
        v = np.asarray(pts, float) - pos
        z = v @ fwd
        ok = z > max(float(c0.frustum_near), 1e-6)
        zs = np.where(ok, z, 1.0)
        near = float(c0.frustum_near)
        half_h = (float(c0.frustum_top) - float(c0.frustum_bottom)) / 2
        half_w = float(c0.frustum_width) if float(c0.frustum_width) > 0 else half_h * self.width() / max(1, self.height())
        cy = (float(c0.frustum_top) + float(c0.frustum_bottom)) / 2
        cx = (float(c0.frustum_center) + float(c1.frustum_center)) / 2
        nx = ((v @ right) * near / zs - cx) / half_w
        ny = ((v @ up) * near / zs - cy) / half_h
        px = np.stack([(nx + 1) / 2 * self.width(), (1 - ny) / 2 * self.height()], axis=1)
        return px, ok & (np.abs(nx) <= 1) & (np.abs(ny) <= 1)

    def _draw_points(self, pts: np.ndarray) -> None:
        """Lidar returns as 2D points over the 3D image: one draw call instead of a 3D shape per point."""
        px, ok = self.project(pts)
        px = px[ok]
        if not len(px):
            return
        p = QPainter(self)
        pen = QPen(QColor(70, 240, 150, 230), 3.0)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawPoints(QPolygonF([QPointF(float(x), float(y)) for x, y in px]))
        p.end()

    def _geom(self, kind, size, pos, rgba, mat=None):
        scn = self.scn
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, kind, np.asarray(size, float), np.asarray(pos, float),
                            np.eye(3).reshape(-1) if mat is None else mat, np.asarray(rgba, np.float32))
        scn.ngeom += 1
        return g

    def _line(self, a, b, width, rgba):
        g = self._geom(mujoco.mjtGeom.mjGEOM_CAPSULE, (0.01, 0.01, 0.01), (0, 0, 0), rgba)
        if g is not None:
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float))

    def _overlays(self, path, goals) -> None:
        if self.selected_object and self._mark is not None:        # a ring under the selected scene object
            x, y, r = self._mark
            self._geom(mujoco.mjtGeom.mjGEOM_CYLINDER, (r, 0.004, 0), (x, y, 0.006), (0.3, 0.6, 1.0, 0.55))
        if path is not None and len(path) > 1:
            for a, b in zip(path[:-1], path[1:]):
                self._line((*a, 0.03), (*b, 0.03), 0.012, (0.3, 0.6, 1.0, 0.9))
        for i, g in enumerate(goals):
            self._geom(mujoco.mjtGeom.mjGEOM_CYLINDER, (0.12, 0.006, 0), (*g, 0.008), (1.0, 0.75, 0.2, 0.9))
            self._line((*g, 0.0), (*g, 0.5), 0.008, (1.0, 0.75, 0.2, 0.9))
        for p in self.measure_pts:
            self._geom(mujoco.mjtGeom.mjGEOM_SPHERE, (0.02, 0, 0), p, (1.0, 0.3, 0.5, 1.0))
        if len(self.measure_pts) == 2:
            self._line(self.measure_pts[0], self.measure_pts[1], 0.006, (1.0, 0.3, 0.5, 1.0))

    def _apply_forces(self, m, d) -> None:
        """Push impulses and the grab spring. Called with the session lock held."""
        if self._pushes:
            keep = []
            for body, force, end in self._pushes:
                if body < m.nbody and d.time < end:
                    d.xfrc_applied[body, :3] = force
                    keep.append((body, force, end))
                elif body < m.nbody:
                    d.xfrc_applied[body, :3] = 0.0
            self._pushes = keep
        if self._grab_body > 0 and self.pert.active:
            d.xfrc_applied[self._grab_body] = 0.0
            mujoco.mjv_applyPerturbForce(m, d, self.pert)

    # ---- camera -----------------------------------------------------------
    def camera_names(self) -> list[str]:
        m = self.sess.model
        return [m.camera(i).name for i in range(m.ncam)]

    def set_camera(self, name: str) -> None:
        self.camera_name = name
        if name == "Follow":
            self.cam.type, self.cam.trackbodyid, self.cam.fixedcamid = mujoco.mjtCamera.mjCAMERA_TRACKING, 1, -1
        elif name in self.camera_names():
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = mujoco.mj_name2id(self.sess.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        else:
            self.camera_name = "Free"
            self.cam.type, self.cam.fixedcamid, self.cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_FREE, -1, -1

    def set_view(self, name: str) -> None:
        if self.camera_name not in ("Free", "Follow"):
            self.set_camera("Free")
        self.cam.azimuth, self.cam.elevation = VIEWS[name]

    def frame_robot(self) -> None:
        s = self.sess
        with s.lock:
            if s.model.nbody < 2:
                return
            body = self.selected if self.selected > 0 else 1
            com = s.data.subtree_com[body].copy()
            height = float(s.data.subtree_com[1][2]) if s.model.nbody > 1 else 1.0
        if self.camera_name not in ("Free", "Follow"):
            self.set_camera("Free")
        self.cam.lookat[:] = com
        self.cam.distance = float(np.clip(4.5 * max(height, 0.2), 1.0, 12.0)) if body == 1 else 1.2

    def set_flag(self, name: str, on: bool) -> None:
        if name in FLAGS:
            self.opt.flags[FLAGS[name]] = 1 if on else 0
        else:
            self.show[name] = on
            if name in ("Shadows", "Reflections") and on:
                self.auto_quality = False     # the user asked for it: keep it

    def flag(self, name: str) -> bool:
        return bool(self.opt.flags[FLAGS[name]]) if name in FLAGS else self.show[name]

    # ---- tools ------------------------------------------------------------
    def set_tool(self, name: str) -> None:
        if name not in TOOLS:
            return
        self.tool = name
        self.measure_pts = []
        self.setCursor({"orbit": Qt.OpenHandCursor, "grab": Qt.PointingHandCursor}.get(name, Qt.CrossCursor)
                       if name != "select" else Qt.ArrowCursor)
        self.message.emit(f"{TOOLS[name][0]}: {TOOLS[name][2]}")
        self.toolChanged.emit(name)

    def select_body(self, body: int) -> None:
        self.selected = int(body)

    def pick(self, x: float, y: float):
        """(body id, world point, geom id) under widget position (x, y); body is -1 for empty space."""
        if self.scn is None:
            return -1, np.zeros(3), -1
        pnt, geom, flex, skin = np.zeros(3), np.zeros(1, np.int32), np.zeros(1, np.int32), np.zeros(1, np.int32)
        s = self.sess
        with s.lock:
            if self._model is not s.model:
                return -1, pnt, -1
            body = mujoco.mjv_select(s.model, s.data, self.opt, self.width() / max(1, self.height()),
                                     x / max(1, self.width()), 1 - y / max(1, self.height()), self.scn,
                                     pnt, geom, flex, skin)
        return int(body), pnt, int(geom[0])

    def floor_point(self, x: float, y: float) -> np.ndarray | None:
        """Where the view ray through widget position (x, y) meets the floor (z = 0); None if it does not."""
        if self.scn is None:
            return None
        c0, c1 = self.scn.camera[0], self.scn.camera[1]
        pos = (np.array(c0.pos) + np.array(c1.pos)) / 2
        fwd, up = np.array(c0.forward, float), np.array(c0.up, float)
        right = np.cross(fwd, up)
        near = float(c0.frustum_near)
        half_h = (float(c0.frustum_top) - float(c0.frustum_bottom)) / 2
        half_w = float(c0.frustum_width) if float(c0.frustum_width) > 0 else half_h * self.width() / max(1, self.height())
        cy = (float(c0.frustum_top) + float(c0.frustum_bottom)) / 2
        cx = (float(c0.frustum_center) + float(c1.frustum_center)) / 2
        nx, ny = 2 * x / max(1, self.width()) - 1, 1 - 2 * y / max(1, self.height())
        d = fwd * near + right * (nx * half_w + cx) + up * (ny * half_h + cy)
        if d[2] > -1e-9:
            return None
        return pos + d * (-pos[2] / d[2])

    def _walk_to(self, point, append: bool) -> None:
        s = self.sess
        with s.lock:
            if s.task is None:
                self.message.emit("This model cannot walk (no free-floating body).")
                return
            s.nav.set_goal(point[:2], append=append)
            s.running = True
        self.message.emit(f"Walking to ({point[0]:.2f}, {point[1]:.2f}) m")

    def _camera_pos_forward(self):
        c0, c1 = self.scn.camera[0], self.scn.camera[1]
        return (np.array(c0.pos) + np.array(c1.pos)) / 2, np.array(c0.forward, float)

    def _normal_at(self, x, y, point, geom):
        """Surface normal at a picked point, from two neighbouring picks (falls back to facing the camera)."""
        cam_pos, fwd = self._camera_pos_forward()
        n = -fwd
        _, p1, g1 = self.pick(x + 3, y)
        _, p2, g2 = self.pick(x, y + 3)
        if g1 == geom and g2 == geom:
            c = np.cross(p1 - point, p2 - point)
            if np.linalg.norm(c) > 1e-12:
                n = c / np.linalg.norm(c)
                if np.dot(n, cam_pos - point) < 0:
                    n = -n
        return n

    def _click(self, x: float, y: float, mods) -> None:
        body, point, geom = self.pick(x, y)
        s, tool = self.sess, self.tool
        if tool in ("select", "orbit"):
            group = s.object_at(geom) if geom >= 0 and not s.custom_path else None
            if group:                                   # a scene object (static or loose)
                self.selected = body if body > 0 else -1
                self.selected_object = group
                self.objectPicked.emit(group)
            elif body > 0:                              # a part of the robot
                self.selected, self.selected_object = body, ""
                self.bodyPicked.emit(body)
            elif body == 0 and tool == "select" and self.touch_walk:
                self._walk_to(point, bool(mods & Qt.ShiftModifier))      # the floor (or ground to walk on)
            else:
                self.selected, self.selected_object = -1, ""
                self.bodyPicked.emit(-1)
        elif tool == "build":
            p = point if body == 0 else self.floor_point(x, y)
            if p is None:
                self.message.emit("Click the floor to place it.")
                return
            k = self.build_snap
            px, py = (round(p[0] / k) * k, round(p[1] / k) * k) if k > 0 else (p[0], p[1])
            try:
                group = s.add_object(self.build_kind, px, py, np.deg2rad(self.build_yaw), self.build_scale, self.build_colour)
                self.selected_object = group
                self.objectPicked.emit(group)
                self.message.emit(f"Added a {self.build_kind.replace('_', ' ')} at ({px:.2f}, {py:.2f}) m")
            except Exception as e:  # noqa: BLE001 - shown to the user
                self.message.emit(f"Could not place it: {e}")
        elif tool == "goal":
            if body != 0:
                self.message.emit("Click the floor to set a walking goal." if body < 0 else
                                  "That is the robot, not the floor.")
                return
            with s.lock:
                if s.task is None:
                    self.message.emit("This model cannot walk (no free-floating body).")
                    return
                s.nav.set_goal(point[:2], append=bool(mods & Qt.ShiftModifier))
                s.running = True
            self.message.emit(f"Goal set at ({point[0]:.2f}, {point[1]:.2f}) m")
        elif tool == "push":
            if body <= 0:
                return
            _, fwd = self._camera_pos_forward()
            fwd[2] = 0.0
            if np.linalg.norm(fwd) < 1e-6:
                return
            with s.lock:
                self._pushes.append((body, fwd / np.linalg.norm(fwd) * self.push_force, float(s.data.time) + 0.15))
                s.running = True
                name = s.model.body(body).name
            self.message.emit(f"Pushed {name} with {self.push_force:.0f} N for 0.15 s")
        elif tool == "place":
            if body <= 0 or geom < 0:
                self.message.emit("Click a part of the robot to mount the accessory.")
                return
            normal = self._normal_at(x, y, point, geom)
            try:
                acc = s.place_accessory(self.place_type, geom, point.tolist(), normal.tolist(), self.place_snap)
                self.message.emit(f"Mounted {acc['type']} on {acc['body']}")
            except Exception as e:  # noqa: BLE001 - shown to the user
                self.message.emit(f"Could not place it: {e}")
        elif tool == "measure":
            if body < 0:
                return
            if len(self.measure_pts) >= 2:
                self.measure_pts = []
            self.measure_pts.append(point.copy())
            if len(self.measure_pts) == 2:
                self.message.emit(f"Distance {np.linalg.norm(self.measure_pts[1] - self.measure_pts[0]):.3f} m")

    # ---- mouse ------------------------------------------------------------
    def _cam_args(self) -> tuple:
        """Trailing arguments of mjv_moveCamera: MuJoCo 4 takes only the camera, MuJoCo 3 the scene too."""
        if Viewport._move_takes_scene is None:
            Viewport._move_takes_scene = "scn" in (mujoco.mjv_moveCamera.__doc__ or "").split(")")[0]
        return (self.scn, self.cam) if Viewport._move_takes_scene else (self.cam,)

    def mousePressEvent(self, e) -> None:
        self._last = e.position()
        self._press = (e.position(), e.button())
        self._drag_object = ""
        if e.button() == Qt.LeftButton and self.tool == "select" and not (e.modifiers() & Qt.AltModifier) \
                and not self.sess.custom_path:
            _b, _p, g = self.pick(e.position().x(), e.position().y())
            self._drag_object = (self.sess.object_at(g) or "") if g >= 0 else ""
        if e.button() == Qt.LeftButton and self.tool == "grab" and not (e.modifiers() & Qt.AltModifier):
            body, point, _ = self.pick(e.position().x(), e.position().y())
            if body > 0:
                s = self.sess
                with s.lock:
                    self.selected = body
                    self.pert.select = body
                    self.pert.skinselect = -1
                    rel = point - s.data.xpos[body]
                    self.pert.localpos[:] = s.data.xmat[body].reshape(3, 3).T @ rel
                    self.pert.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
                    mujoco.mjv_initPerturb(s.model, s.data, self.scn, self.pert)
                    self._grab_body = body
                    s.running = True
                self.bodyPicked.emit(body)

    def mouseMoveEvent(self, e) -> None:
        if self._last is None or not e.buttons():
            return
        dx, dy = e.position().x() - self._last.x(), e.position().y() - self._last.y()
        self._last = e.position()
        h = max(1, self.height())
        left, alt = bool(e.buttons() & Qt.LeftButton), bool(e.modifiers() & Qt.AltModifier)
        if left and self._grab_body > 0 and self.pert.active:
            act = mujoco.mjtMouse.mjMOUSE_MOVE_H if e.modifiers() & Qt.ShiftModifier else mujoco.mjtMouse.mjMOUSE_MOVE_V
            with self.sess.lock:
                if self._model is self.sess.model:
                    mujoco.mjv_movePerturb(self.sess.model, self.sess.data, act, dx / h, dy / h, self.scn, self.pert)
            return
        if self.scn is None or self.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED:
            return
        if (e.buttons() & Qt.RightButton and not e.modifiers() & Qt.ShiftModifier) or (left and (alt or self.tool == "orbit")):
            act = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif e.buttons() & Qt.MiddleButton or e.buttons() & Qt.RightButton:
            act = mujoco.mjtMouse.mjMOUSE_MOVE_H if e.modifiers() & Qt.ShiftModifier else mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            return
        mujoco.mjv_moveCamera(self.sess.model, act, dx / h, dy / h, *self._cam_args())

    def mouseReleaseEvent(self, e) -> None:
        if self._grab_body > 0:
            with self.sess.lock:
                if self._grab_body < self.sess.model.nbody:
                    self.sess.data.xfrc_applied[self._grab_body] = 0.0
            self._grab_body, self.pert.active = -1, 0
        elif self._press is not None and e.button() == Qt.LeftButton and self._press[1] == Qt.LeftButton:
            moved = (e.position() - self._press[0]).manhattanLength()
            if moved < 4 and not e.modifiers() & Qt.AltModifier:
                self._click(e.position().x(), e.position().y(), e.modifiers())
            elif moved >= 6 and self._drag_object:              # an object was dragged: put it down there
                p = self.floor_point(e.position().x(), e.position().y())
                if p is not None:
                    k = self.build_snap
                    px, py = (round(p[0] / k) * k, round(p[1] / k) * k) if k > 0 else (p[0], p[1])
                    try:
                        self.sess.update_object(self._drag_object, x=px, y=py)
                        self.selected_object = self._drag_object
                        self.objectPicked.emit(self._drag_object)
                        self.message.emit(f"Moved to ({px:.2f}, {py:.2f}) m")
                    except Exception as err:  # noqa: BLE001
                        self.message.emit(f"Could not move it: {err}")
        self._drag_object = ""
        self._press = self._last = None

    def mouseDoubleClickEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            body, point, _ = self.pick(e.position().x(), e.position().y())
            if body >= 0 and self.cam.type != mujoco.mjtCamera.mjCAMERA_FIXED:
                self.cam.lookat[:] = point

    def wheelEvent(self, e) -> None:
        if self.scn is None or self.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED:
            return
        mujoco.mjv_moveCamera(self.sess.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, 0.0006 * e.angleDelta().y(),
                              *self._cam_args())
