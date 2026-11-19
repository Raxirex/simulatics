"""RosBridge: publish the simulated robot to ROS 2 and take commands back.

    from simulatics.bridge import RosBridge, RosbridgeClient
    bridge = RosBridge(session, RosbridgeClient("ws://localhost:9090")).start()
    ...
    bridge.status()   # connection state, per-topic counts and rates, last events
    bridge.stop()     # joins its threads and stops the transport

The bridge runs on its own thread (plus one for camera frames, which wait on the simulation thread's
renderer) and publishes at configurable wall-clock rates, stamped with simulation time:

    /clock               rosgraph_msgs/Clock          sim time (run ROS nodes with use_sim_time:=true)
    /joint_states        sensor_msgs/JointState       every hinge / slide joint of the robot
    /tf                  tf2_msgs/TFMessage           map->odom (identity or SLAM), odom->base_link (robot pose),
                                                      URDF links, moving sensors
    /tf_static           tf2_msgs/TFMessage (latched) base_link->sensors, base_link->URDF root, fixed joints
    /odom                nav_msgs/Odometry            pose in odom, twist in base_link
    /imu                 sensor_msgs/Imu              the model's gyro / accelerometer / framequat sensors
    /scan                sensor_msgs/LaserScan        the 3D lidar collapsed to a planar scan
    /head_cam/image/compressed, /head_cam/camera_info  JPEG + pinhole intrinsics (throttled)
    /map                 nav_msgs/OccupancyGrid (latched, throttled)  the lidar-built (or ground truth) grid
    /plan                nav_msgs/Path                the navigator's current path
    /robot_description   std_msgs/String (latched)    URDF from design/urdf.export_urdf
    /simulatics/events   std_msgs/String              JSON results of the commands below

and subscribes to

    /cmd_vel                 geometry_msgs/Twist        -> controller.set_velocity (or a short relative goal)
    /goal_pose               geometry_msgs/PoseStamped  -> navigation goal with heading (RViz "2D Goal Pose")
    /simulatics/arm_command  std_msgs/String (JSON)     -> an arm skill, e.g. {"skill": "wave", "side": "right"}

The session is only touched through the small `SessionLike` protocol below; every read of MuJoCo
state happens under ``session.lock``.
"""
from __future__ import annotations

import collections
import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import mujoco
import numpy as np

from . import msgs
from .transport import Transport
from .urdf_tf import UrdfTree

log = logging.getLogger("simulatics.bridge")

DEFAULT_TOPICS = {
    "clock": "/clock", "joint_states": "/joint_states", "tf": "/tf", "tf_static": "/tf_static",
    "odom": "/odom", "imu": "/imu", "scan": "/scan", "image": "/{camera}/image/compressed",
    "camera_info": "/{camera}/camera_info", "map": "/map", "plan": "/plan",
    "robot_description": "/robot_description", "events": "/simulatics/events",
    "cmd_vel": "/cmd_vel", "goal_pose": "/goal_pose", "arm_command": "/simulatics/arm_command",
}

# key -> (ROS type, direction, latched)
TOPIC_TYPES = {
    "clock": (msgs.CLOCK, "out", False), "joint_states": (msgs.JOINT_STATE, "out", False),
    "tf": (msgs.TF_MESSAGE, "out", False), "tf_static": (msgs.TF_MESSAGE, "out", True),
    "odom": (msgs.ODOMETRY, "out", False), "imu": (msgs.IMU, "out", False), "scan": (msgs.LASER_SCAN, "out", False),
    "image": (msgs.COMPRESSED_IMAGE, "out", False), "camera_info": (msgs.CAMERA_INFO, "out", False),
    "map": (msgs.OCCUPANCY_GRID, "out", True), "plan": (msgs.PATH, "out", False),
    "robot_description": (msgs.STRING, "out", True), "events": (msgs.STRING, "out", False),
    "cmd_vel": (msgs.TWIST, "in", False), "goal_pose": (msgs.POSE_STAMPED, "in", False),
    "arm_command": (msgs.STRING, "in", False),
}

# ROS camera body frame (x forward, y left, z up) expressed in MuJoCo's camera frame (x right, y up, looks along -z)
_CAM_BODY_IN_MJ = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])


@dataclass
class BridgeConfig:
    """Rates are wall-clock Hz (0 disables a topic); stamps are simulation time."""
    clock_hz: float = 50.0
    joint_states_hz: float = 30.0
    tf_hz: float = 30.0
    odom_hz: float = 30.0
    imu_hz: float = 50.0
    scan_hz: float | None = None          # None: the lidar's own rate (LidarConfig.rate_hz)
    camera_hz: float = 2.0
    map_hz: float = 0.5                   # at most; only sent when the grid changed
    plan_hz: float = 2.0                  # at most; sent when the path changed, and every 2 s while walking
    camera: str = "head_cam"
    camera_width: int = 320
    camera_height: int = 240
    image_format: str = "rgb8; jpeg compressed bgr8"   # what compressed_image_transport writes for JPEG
    map_source: str = "lidar"             # "lidar" (built from scans) | "truth" (static geoms of the model)
    link_tf: bool = True                  # TF of every URDF link (turn off when running robot_state_publisher)
    map_to_odom: bool = True              # publish map->odom (identity, or from slam_pose); off when a SLAM node does
    scan_z_min: float = 0.15              # world height band of returns that count as obstacles in /scan
    scan_z_max: float = 1.8
    scan_range_min: float = 0.05
    scan_no_return: float | None = None   # None: +inf if the transport can carry it (REP-117), else range_max + 1
    cmd_vel_timeout: float = 0.5          # s without /cmd_vel -> stop
    cmd_vel_horizon: float = 1.0          # s of motion per relative goal (controllers without set_velocity)
    auto_run: bool = True                 # commands un-pause the simulation, like the GUI's click-to-walk
    topics: dict = field(default_factory=dict)   # overrides of DEFAULT_TOPICS
    frames: dict = field(default_factory=dict)   # overrides: map, odom, base_link, lidar, imu, camera

    def resolved_topics(self) -> dict[str, str]:
        t = {**DEFAULT_TOPICS, **self.topics}
        return {k: v.format(camera=self.camera) for k, v in t.items()}

    def resolved_frames(self) -> dict[str, str]:
        f = {"map": msgs.MAP, "odom": msgs.ODOM, "base_link": msgs.BASE_LINK, "lidar": msgs.LIDAR_LINK,
             "imu": msgs.IMU_LINK, "camera": self.camera, **self.frames}
        f["optical"] = f["camera"] + msgs.OPTICAL_SUFFIX
        return f


# --------------------------------------------------------------------------- the session contract
@runtime_checkable
class SessionLike(Protocol):
    """What the bridge needs from a simulation (simulatics.server.session.SimSession via SessionAdapter).

    Required:
      lock        a threading.(R)Lock held by whoever steps `data`; the bridge holds it while reading
      model, data MuJoCo MjModel / MjData (may be replaced: the bridge notices a new model object)
      controller  the active controller (control/base.py); optional set_velocity(vx, vy, wz)
      nav         navigator: set_goal(xy, yaw=None), clear(), path ((n, 2) array), goals (list), state (str);
                  optional take_new_path() (to discard the stop request clear() leaves for the controller)
      lidar       sim/sensors.Lidar or None: scan(model, data) -> (origin, points, dist, hit); cfg.enabled,
                  cfg.h_samples, cfg.max_range, cfg.rate_hz
      grid        plan/grid.GridMap: lidar_image() (0 unknown, 1 free, 2 occupied), truth, res, origin
      robot       llm/robot.RobotInterface (camera_jpeg, cameras, arm skills, pose)
    Optional:
      running     bool attribute; set True when a command arrives (BridgeConfig.auto_run)
      urdf()      -> URDF string for /robot_description and link TF
    """
    lock: object
    model: mujoco.MjModel
    data: mujoco.MjData
    controller: object
    nav: object
    lidar: object
    grid: object
    robot: object


class SessionAdapter:
    """`SessionLike` view of a live `SimSession`. Attributes are looked up on every access, because
    SimSession.build() replaces model, data, controller and lidar."""

    def __init__(self, session):
        self.s = session

    lock = property(lambda self: self.s.lock)
    model = property(lambda self: self.s.model)
    data = property(lambda self: self.s.data)
    controller = property(lambda self: self.s.controller)
    nav = property(lambda self: self.s.nav)
    lidar = property(lambda self: self.s.lidar)
    grid = property(lambda self: self.s.grid)
    robot = property(lambda self: self.s.robot)

    @property
    def running(self) -> bool:
        return bool(self.s.running)

    @running.setter
    def running(self, v: bool) -> None:
        self.s.running = bool(v)

    def urdf(self) -> str:
        from ..design.urdf import export_urdf
        return export_urdf(xml=self.s.export_xml())


def as_session_like(session):
    """Wrap a SimSession (anything with export_xml and lidar_cfg) in a SessionAdapter; pass others through."""
    if isinstance(session, SessionAdapter) or hasattr(session, "urdf"):
        return session
    if hasattr(session, "export_xml") and hasattr(session, "lidar_cfg"):
        return SessionAdapter(session)
    return session


# --------------------------------------------------------------------------- per-model cache
def _rel_pose(p_root, R_root, p, R) -> tuple[np.ndarray, tuple]:
    """Pose of a world frame (p, R) in the root frame: translation and xyzw quaternion."""
    R_root = np.asarray(R_root).reshape(3, 3)
    return R_root.T @ (np.asarray(p) - np.asarray(p_root)), msgs.matrix_to_quat(R_root.T @ np.asarray(R).reshape(3, 3))


class _ModelInfo:
    """Indices into one compiled model (rebuilt whenever the session's model object changes)."""

    def __init__(self, m: mujoco.MjModel, camera: str):
        self.model = m
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.root = int(m.jnt_bodyid[free[0]]) if free else (1 if m.nbody > 1 else 0)
        self.free_qadr = int(m.jnt_qposadr[free[0]]) if free else None
        self.free_dadr = int(m.jnt_dofadr[free[0]]) if free else None
        rootid = m.body_rootid[self.root]
        moving = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        joints = [j for j in range(m.njnt) if m.jnt_type[j] in moving and m.body_rootid[m.jnt_bodyid[j]] == rootid]
        self.joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}" for j in joints]
        self.joint_qadr = np.array([m.jnt_qposadr[j] for j in joints], int)
        self.joint_dadr = np.array([m.jnt_dofadr[j] for j in joints], int)
        self.lidar_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "lidar")
        self.imu_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.camera = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        self.camera_fovy = float(m.cam_fovy[self.camera]) if self.camera >= 0 else 60.0
        self.imu_adr: dict[str, int] = {}
        kinds = {mujoco.mjtSensor.mjSENS_GYRO: "gyro", mujoco.mjtSensor.mjSENS_ACCELEROMETER: "accel",
                 mujoco.mjtSensor.mjSENS_FRAMEQUAT: "quat"}
        for i in range(m.nsensor):
            k = kinds.get(m.sensor_type[i])
            if k and k not in self.imu_adr and self.imu_site >= 0 and m.sensor_objtype[i] == mujoco.mjtObj.mjOBJ_SITE \
                    and m.sensor_objid[i] == self.imu_site:
                self.imu_adr[k] = int(m.sensor_adr[i])
        on_root = lambda body: int(body) == self.root   # noqa: E731
        self.lidar_static = self.lidar_site >= 0 and on_root(m.site_bodyid[self.lidar_site])
        self.imu_static = self.imu_site >= 0 and on_root(m.site_bodyid[self.imu_site])
        self.camera_static = (self.camera >= 0 and on_root(m.cam_bodyid[self.camera])
                              and m.cam_mode[self.camera] == mujoco.mjtCamLight.mjCAMLIGHT_FIXED)

    def base_pose(self, d: mujoco.MjData) -> tuple[np.ndarray, tuple]:
        """Root body position and xyzw orientation in the world (= odom) frame."""
        if self.free_qadr is not None:
            q = d.qpos[self.free_qadr:self.free_qadr + 7]
            return q[:3].copy(), msgs.quat_wxyz_to_xyzw(q[3:7])
        return d.xpos[self.root].copy(), msgs.quat_wxyz_to_xyzw(d.xquat[self.root])

    def sensor_frames(self, d: mujoco.MjData, frames: dict, which: str) -> list[tuple]:
        """[(parent, child, translation, quat)] of the sensor frames, `which` = "static" | "dynamic"."""
        static = which == "static"
        p_r, R_r = d.xpos[self.root], d.xmat[self.root]
        out = []
        if self.lidar_site >= 0:
            if self.lidar_static == static:
                out.append((frames["lidar"], *_rel_pose(p_r, R_r, d.site_xpos[self.lidar_site],
                                                        d.site_xmat[self.lidar_site])))
        elif not static:  # sim/sensors.Lidar falls back to 0.5 m above the root (world z)
            out.append((frames["lidar"], *_rel_pose(p_r, R_r, p_r + [0, 0, 0.5], R_r)))
        if self.imu_site >= 0 and self.imu_static == static:
            out.append((frames["imu"], *_rel_pose(p_r, R_r, d.site_xpos[self.imu_site], d.site_xmat[self.imu_site])))
        if self.camera >= 0 and self.camera_static == static:
            R_cam = d.cam_xmat[self.camera].reshape(3, 3) @ _CAM_BODY_IN_MJ
            out.append((frames["camera"], *_rel_pose(p_r, R_r, d.cam_xpos[self.camera], R_cam)))
        return [(frames["base_link"], child, p, q) for child, p, q in out]


def _tf(stamp, parent, child, p, q) -> dict:
    return msgs.transform_stamped(stamp, parent, child, p, q)


def _se2_compose(a, b):
    """a * b for (x, y, yaw) poses."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2]


def _se2_inv(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return -(c * a[0] + s * a[1]), -(-s * a[0] + c * a[1]), -a[2]


# --------------------------------------------------------------------------- the bridge
class RosBridge:
    def __init__(self, session, transport: Transport, config: BridgeConfig | None = None,
                 slam_pose: Callable[[], tuple[float, float, float] | None] | None = None,
                 own_transport: bool = True):
        """session: SessionLike (a SimSession is wrapped automatically). transport: RosbridgeClient or
        RclpyTransport. slam_pose: optional callable returning the robot's (x, y, yaw) in the map frame
        from a localisation estimate; map->odom then follows it instead of being the identity.
        own_transport: stop() also stops the transport."""
        self.s = as_session_like(session)
        self.t = transport
        self.cfg = config or BridgeConfig()
        self.slam_pose = slam_pose
        self.own_transport = own_transport
        self.topics = self.cfg.resolved_topics()
        self.frames = self.cfg.resolved_frames()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cam_thread: threading.Thread | None = None
        self._info: _ModelInfo | None = None
        self._static_frames: list = []
        self._urdf: UrdfTree | None = None
        self._urdf_xml: str | None = None
        self._next: dict[str, float] = {}
        self._last_stamp: dict[str, float] = {}
        self._plan_sig = None
        self._plan_sent = 0.0
        self._map_sig = None
        self._map_odom = (0.0, 0.0, 0.0)     # current map->odom (x, y, yaw)
        self._cmd_lock = threading.Lock()
        self._vel_active = False
        self._last_cmd = 0.0
        self._last_rel_goal = 0.0
        self._sim_time = 0.0
        self.events: collections.deque = collections.deque(maxlen=50)
        self.received: collections.Counter = collections.Counter()
        self.error = ""
        self.camera_error = ""
        self.description_error = ""
        self._started = 0.0

    # ------------------------------------------------------------------ lifecycle
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> "RosBridge":
        if self.running:
            return self
        self._stop.clear()
        self.t.start()
        for key, (typ, direction, latched) in TOPIC_TYPES.items():
            topic = self.topics[key]
            if direction == "out":
                self.t.advertise(topic, typ, latch=latched, queue_size=1 if latched else 10)
        self.t.subscribe(self.topics["cmd_vel"], msgs.TWIST, self._on_cmd_vel)
        self.t.subscribe(self.topics["goal_pose"], msgs.POSE_STAMPED, self._on_goal_pose)
        self.t.subscribe(self.topics["arm_command"], msgs.STRING, self._on_arm_command)
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._loop, name="ros-bridge", daemon=True)
        self._thread.start()
        if self.cfg.camera_hz > 0:
            self._cam_thread = threading.Thread(target=self._camera_loop, name="ros-bridge-camera", daemon=True)
            self._cam_thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """Stop publishing, stop the robot if /cmd_vel was driving it, join threads, stop the transport."""
        self._stop.set()
        for th in (self._thread, self._cam_thread):
            if th is not None and th is not threading.current_thread():
                th.join(timeout)
        self._thread = self._cam_thread = None
        with self._cmd_lock:
            if self._vel_active:
                self._apply_velocity(0.0, 0.0, 0.0)
        if self.own_transport:
            self.t.stop()

    def status(self) -> dict:
        """For the GUI: connection, per-topic statistics, recent command events."""
        ts = self.t.status()
        return {"running": self.running, "connected": self.t.connected, "transport": ts.get("transport"),
                "url": ts.get("url"), "state": ts.get("state"), "error": self.error or ts.get("error", ""),
                "sim_time": round(self._sim_time, 3), "topics": ts.get("topics", {}),
                "received": dict(self.received), "events": list(self.events)[-10:],
                "camera_error": self.camera_error, "robot_description": self._urdf_xml is not None,
                "description_error": self.description_error, "velocity_mode": self._vel_active,
                "uptime_s": round(time.monotonic() - self._started, 1) if self._started else 0.0}

    # ------------------------------------------------------------------ main loop
    def _rates(self) -> dict[str, float]:
        c = self.cfg
        lidar = getattr(self.s, "lidar", None)
        scan = c.scan_hz if c.scan_hz is not None else float(getattr(getattr(lidar, "cfg", None), "rate_hz", 10.0))
        return {"clock": c.clock_hz, "joint_states": c.joint_states_hz, "tf": c.tf_hz, "odom": c.odom_hz,
                "imu": c.imu_hz, "scan": scan, "map": c.map_hz, "plan": c.plan_hz}

    def _loop(self) -> None:
        while not self._stop.is_set():
            rates = {k: r for k, r in self._rates().items() if r and r > 0}
            period = min([1.0 / r for r in rates.values()] + [0.1])
            try:
                self._tick(time.monotonic(), rates)
                self.error = ""
            except Exception as e:  # noqa: BLE001 - keep bridging; report in status()
                self.error = f"{type(e).__name__}: {e}"
                log.exception("bridge tick failed")
            self._stop.wait(max(0.002, period / 2))

    def _due(self, now: float, rates: dict[str, float]) -> set[str]:
        due = set()
        for k, r in rates.items():
            nxt = self._next.get(k, 0.0)
            if now >= nxt:
                due.add(k)
                p = 1.0 / r
                self._next[k] = nxt + p if now - nxt < p else now + p  # no catch-up bursts after a stall
        return due

    _TIMED = frozenset({"clock", "joint_states", "tf", "odom", "imu", "scan"})   # one message per sim instant

    def _tick(self, now: float, rates: dict[str, float]) -> None:
        self._watchdog(now)
        due = self._due(now, rates)
        if not self.t.connected:
            due -= self._TIMED | {"plan"}  # nothing to send them to (latched ones are kept for the reconnect)
        if not due and self._info is not None and self.s.model is self._info.model:
            return
        lidar = getattr(self.s, "lidar", None)
        lidar_on = lidar is not None and bool(getattr(getattr(lidar, "cfg", None), "enabled", True))
        snap: dict = {}
        with self.s.lock:
            m, d = self.s.model, self.s.data
            changed = self._info is None or self._info.model is not m
            if changed:
                self._info = _ModelInfo(m, self.cfg.camera)
                snap["static_frames"] = self._info.sensor_frames(d, self.frames, "static")
            info = self._info
            t = float(d.time)
            snap["t"] = t
            due -= {k for k in self._TIMED if self._last_stamp.get(k) == t}  # paused: nothing new
            if due & {"joint_states", "tf", "odom", "imu"}:
                snap["qpos"], snap["qvel"] = d.qpos.copy(), d.qvel.copy()
                snap["base"] = info.base_pose(d)
                snap["R_root"] = d.xmat[info.root].reshape(3, 3).copy()
            if "joint_states" in due:
                snap["effort"] = d.qfrc_actuator[info.joint_dadr].copy()
            if "imu" in due:
                snap["sensordata"] = d.sensordata.copy()
            if "tf" in due:
                snap["dyn_frames"] = info.sensor_frames(d, self.frames, "dynamic")
            if "scan" in due and lidar_on:
                origin, pts, _, hit = lidar.scan(m, d)
                if info.lidar_site >= 0:
                    R_l = d.site_xmat[info.lidar_site].reshape(3, 3).copy()
                else:
                    R_l = d.xmat[info.root].reshape(3, 3).copy()
                snap["scan"] = (origin, R_l, pts, hit, int(lidar.cfg.h_samples), float(lidar.cfg.max_range),
                                float(getattr(lidar.cfg, "rate_hz", 10.0)))
            if "plan" in due:
                nav = self.s.nav
                snap["plan"] = (np.array(getattr(nav, "path", np.zeros((0, 2))), float).reshape(-1, 2),
                                getattr(nav, "state", ""))
            if "map" in due:
                grid = self.s.grid
                img = (np.where(grid.truth > 0, 2, 1).astype(np.uint8) if self.cfg.map_source == "truth"
                       else grid.lidar_image())
                snap["map"] = (img, float(grid.res), np.asarray(grid.origin, float).copy())
        self._sim_time = t
        if changed:
            self._on_model_changed(t, snap["static_frames"])
        self._update_map_odom(snap)
        self._publish_periodic(due, snap)

    # ------------------------------------------------------------------ model / static data
    def _on_model_changed(self, t: float, static_frames: list) -> None:
        self._urdf, self._urdf_xml, self.description_error = None, None, ""
        self._plan_sig = self._map_sig = None
        urdf_fn = getattr(self.s, "urdf", None)
        if urdf_fn is not None:
            try:
                self._urdf_xml = urdf_fn()
                self._urdf = UrdfTree(self._urdf_xml)
            except Exception as e:  # noqa: BLE001 - e.g. a custom MJCF with mesh files: no description
                self.description_error = f"{type(e).__name__}: {e}"
                self._urdf, self._urdf_xml = None, None
        else:
            self.description_error = "session has no urdf()"
        if self._urdf_xml is not None:
            self.t.publish(self.topics["robot_description"], msgs.string(self._urdf_xml))
        self._static_frames = static_frames
        self._publish_static(t)

    def _publish_static(self, t: float) -> None:
        f = self.frames
        tfs = [_tf(t, parent, child, p, q) for parent, child, p, q in self._static_frames]
        if self._info is not None and self._info.camera >= 0:
            tfs.append(_tf(t, f["camera"], f["optical"], (0, 0, 0), msgs.OPTICAL_IN_BODY_XYZW))
        if self.cfg.link_tf and self._urdf is not None:
            if self._urdf.root and self._urdf.root != f["base_link"]:
                tfs.append(_tf(t, f["base_link"], self._urdf.root, (0, 0, 0), (0, 0, 0, 1)))
            tfs += [_tf(t, parent, child, p, q) for parent, child, p, q in self._urdf.fixed_transforms()]
        self.t.publish(self.topics["tf_static"], msgs.tf_message(tfs))

    def _update_map_odom(self, snap: dict) -> None:
        if self.slam_pose is None or "base" not in snap:
            return
        try:
            est = self.slam_pose()
        except Exception as e:  # noqa: BLE001
            self.error = f"slam_pose: {type(e).__name__}: {e}"
            return
        if est is None:
            return
        p, q = snap["base"]
        odom_base = (float(p[0]), float(p[1]), msgs.quat_to_yaw(q))
        self._map_odom = _se2_compose(tuple(float(v) for v in est), _se2_inv(odom_base))

    @property
    def world_frame(self) -> str:
        """Frame of the simulator's world coordinates (/map, /plan): map when map->odom is the identity."""
        return self.frames["map"] if (self.cfg.map_to_odom and self.slam_pose is None) else self.frames["odom"]

    # ------------------------------------------------------------------ periodic topics
    def _fresh(self, key: str, t: float) -> bool:
        """Only one message per simulation instant (a paused sim must not repeat stamps: TF_REPEATED_DATA)."""
        if self._last_stamp.get(key) == t:
            return False
        self._last_stamp[key] = t
        return True

    def _publish_periodic(self, due: set[str], snap: dict) -> None:
        t, info, f, T = snap["t"], self._info, self.frames, self.topics
        if "clock" in due and self._fresh("clock", t):
            self.t.publish(T["clock"], msgs.clock(t))
        if "joint_states" in due and self._fresh("joint_states", t):
            q, v = snap["qpos"], snap["qvel"]
            self.t.publish(T["joint_states"], msgs.joint_state(
                t, info.joint_names, q[info.joint_qadr], v[info.joint_dadr], snap["effort"]))
        if "tf" in due and self._fresh("tf", t):
            self.t.publish(T["tf"], msgs.tf_message(self._dynamic_tfs(t, snap)))
        if "odom" in due and self._fresh("odom", t):
            self.t.publish(T["odom"], self._odom_msg(t, snap))
        if "imu" in due and self._fresh("imu", t):
            self.t.publish(T["imu"], self._imu_msg(t, snap))
        if "scan" in snap and self._fresh("scan", t):
            self.t.publish(T["scan"], self._scan_msg(t, *snap["scan"]))
        if "plan" in snap:
            pts, state = snap["plan"]
            sig = (pts.round(4).tobytes(), self.world_frame)
            now = time.monotonic()
            if sig != self._plan_sig or (len(pts) and now - self._plan_sent > 2.0):
                self._plan_sig, self._plan_sent = sig, now
                self.t.publish(T["plan"], msgs.path(t, self.world_frame, pts))
        if "map" in snap:
            img, res, origin = snap["map"]
            occ = np.where(img == 2, 100, np.where(img == 1, 0, -1)).astype(np.int8)
            sig = (hash(occ.tobytes()), self.world_frame)
            if sig != self._map_sig:
                self._map_sig = sig
                self.t.publish(T["map"], msgs.occupancy_grid(t, self.world_frame, occ, res, origin))

    def _dynamic_tfs(self, t: float, snap: dict) -> list[dict]:
        f, info = self.frames, self._info
        p, q = snap["base"]
        tfs = [_tf(t, f["odom"], f["base_link"], p, q)]
        if self.cfg.map_to_odom:   # identity until a slam_pose estimate moves it
            x, y, yaw = self._map_odom
            tfs.append(_tf(t, f["map"], f["odom"], (x, y, 0.0), msgs.yaw_to_quat(yaw)))
        tfs += [_tf(t, parent, child, pp, qq) for parent, child, pp, qq in snap["dyn_frames"]]
        if self.cfg.link_tf and self._urdf is not None:
            qpos = snap["qpos"]
            positions = {n: float(qpos[a]) for n, a in zip(info.joint_names, info.joint_qadr)}
            tfs += [_tf(t, parent, child, pp, qq) for parent, child, pp, qq in self._urdf.moving_transforms(positions)]
        return tfs

    def _odom_msg(self, t: float, snap: dict) -> dict:
        info = self._info
        p, q = snap["base"]
        lin = ang = np.zeros(3)
        if info.free_dadr is not None:
            v = snap["qvel"][info.free_dadr:info.free_dadr + 6]
            lin = snap["R_root"].T @ v[:3]   # free joint: world-frame linear velocity -> base_link
            ang = v[3:6]                     # free joint: angular velocity is already in the body frame
        return msgs.odometry(t, self.frames["odom"], self.frames["base_link"], p, q, lin, ang)

    def _imu_msg(self, t: float, snap: dict) -> dict:
        info, sd = self._info, snap["sensordata"]
        a = info.imu_adr
        frame = self.frames["imu"] if info.imu_site >= 0 else self.frames["base_link"]
        quat = msgs.quat_wxyz_to_xyzw(sd[a["quat"]:a["quat"] + 4]) if "quat" in a else snap["base"][1]
        if "gyro" in a:
            gyro = sd[a["gyro"]:a["gyro"] + 3]
        elif info.free_dadr is not None:
            gyro = snap["qvel"][info.free_dadr + 3:info.free_dadr + 6]
        else:
            gyro = None
        acc = sd[a["accel"]:a["accel"] + 3] if "accel" in a else None
        return msgs.imu(t, frame, quat, gyro, acc)

    def _scan_msg(self, t, origin, R_l, pts, hit, h, max_range, rate) -> dict:
        """Collapse the 3D scan to one range per azimuth (like pointcloud_to_laserscan): the nearest return
        between scan_z_min and scan_z_max (world height), measured in the lidar's plane."""
        c = self.cfg
        rel = (pts - origin) @ R_l                       # world -> lidar frame
        rng = np.hypot(rel[:, 0], rel[:, 1]).reshape(-1, h)
        z = pts[:, 2].reshape(-1, h)
        ok = hit.reshape(-1, h) & (z > c.scan_z_min) & (z < c.scan_z_max)
        r = np.where(ok, rng, np.inf).min(axis=0)
        no_return = c.scan_no_return
        if no_return is None:
            no_return = math.inf if getattr(self.t, "supports_nonfinite", False) else max_range + 1.0
        r = np.where(np.isfinite(r), r, no_return)
        k0 = (h + 1) // 2                                # start at -pi: column k has azimuth 2 pi k / h
        inc = 2 * math.pi / h
        return msgs.laser_scan(t, self.frames["lidar"], np.roll(r, -k0), angle_min=k0 * inc - 2 * math.pi,
                               angle_increment=inc, range_min=c.scan_range_min, range_max=max_range,
                               scan_time=1.0 / max(rate, 1e-3))

    # ------------------------------------------------------------------ camera thread
    def _camera_loop(self) -> None:
        c = self.cfg
        period = 1.0 / c.camera_hz
        failures, last_t = 0, None
        next_t = time.monotonic()
        while not self._stop.wait(max(0.0, next_t - time.monotonic())):
            next_t = max(next_t + period, time.monotonic())
            info, robot = self._info, getattr(self.s, "robot", None)
            if info is None or robot is None or not self.t.connected:
                continue
            if info.camera < 0:
                self.camera_error = f"no camera named {c.camera!r}"
                continue
            t = self._sim_time
            if t == last_t:
                continue  # paused: the picture would be the same
            try:
                jpeg = robot.camera_jpeg(c.camera, c.camera_width, c.camera_height)
            except Exception as e:  # noqa: BLE001 - no OpenGL, sim stopped, ...
                failures += 1
                self.camera_error = f"{type(e).__name__}: {e}"
                if failures >= 3:
                    next_t = time.monotonic() + 10.0  # back off; keep trying slowly
                continue
            failures, last_t, self.camera_error = 0, t, ""
            self.t.publish(self.topics["image"], msgs.compressed_image(t, self.frames["optical"], jpeg,
                                                                       c.image_format))
            self.t.publish(self.topics["camera_info"], msgs.camera_info(t, self.frames["optical"], c.camera_width,
                                                                        c.camera_height, info.camera_fovy))

    # ------------------------------------------------------------------ commands from ROS
    def _event(self, source: str, ok: bool, message: str, **extra) -> None:
        ev = {"t": round(self._sim_time, 3), "source": source, "ok": bool(ok), "message": message, **extra}
        self.events.append(ev)
        try:
            self.t.publish(self.topics["events"], msgs.string(json.dumps(ev)))
        except Exception:  # noqa: BLE001
            pass

    def _run_sim(self) -> None:
        if self.cfg.auto_run and hasattr(self.s, "running"):
            self.s.running = True

    def _robot_pose(self) -> tuple[float, float, float]:
        with self.s.lock:
            info = self._info or _ModelInfo(self.s.model, self.cfg.camera)
            p, q = info.base_pose(self.s.data)
        return float(p[0]), float(p[1]), msgs.quat_to_yaw(q)

    def _apply_velocity(self, vx: float, vy: float, wz: float) -> str:
        """Velocity command into the controller (called with _cmd_lock held)."""
        moving = abs(vx) + abs(vy) + abs(wz) > 1e-6
        with self.s.lock:
            c, nav = self.s.controller, self.s.nav
            if hasattr(c, "set_velocity"):
                if getattr(nav, "goals", None) or getattr(nav, "state", "") == "walking":
                    nav.clear()           # a goal in progress would re-send its path over the velocity command
                    take = getattr(nav, "take_new_path", None)
                    if take is not None:
                        take()            # ... and so would the stop request clear() leaves for the controller
                c.set_velocity(vx, vy, wz)
                if moving:
                    self._run_sim()
                self._vel_active = moving
                return "velocity"
        # controllers without set_velocity: a short goal ahead, refreshed at most twice a second
        now = time.monotonic()
        if not moving:
            with self.s.lock:
                self.s.nav.clear()
            self._vel_active = False
            self._last_rel_goal = 0.0   # the next command after a stop must not wait for the refresh period
            return "stop"
        if now - self._last_rel_goal < 0.5:
            return "goal (unchanged)"
        self._last_rel_goal = now
        x, y, yaw = self._robot_pose()
        h = self.cfg.cmd_vel_horizon
        dx, dy = vx * h, vy * h
        gx, gy = x + math.cos(yaw) * dx - math.sin(yaw) * dy, y + math.sin(yaw) * dx + math.cos(yaw) * dy
        with self.s.lock:
            self.s.nav.set_goal([gx, gy], yaw + wz * h)
            self._run_sim()
        self._vel_active = True
        return "relative goal"

    def _watchdog(self, now: float) -> None:
        with self._cmd_lock:
            if self._vel_active and now - self._last_cmd > self.cfg.cmd_vel_timeout:
                try:
                    self._apply_velocity(0.0, 0.0, 0.0)
                    self._event("cmd_vel", True, f"no /cmd_vel for {self.cfg.cmd_vel_timeout} s: stopping")
                except Exception as e:  # noqa: BLE001
                    self.error = f"cmd_vel stop: {type(e).__name__}: {e}"
                self._vel_active = False

    def _on_cmd_vel(self, msg: dict) -> None:
        self.received["cmd_vel"] += 1
        try:
            vx, vy, _, _, _, wz = msgs.twist_to_tuple(msg)
        except Exception as e:  # noqa: BLE001
            self._event("cmd_vel", False, f"bad Twist: {e}")
            return
        with self._cmd_lock:
            self._last_cmd = time.monotonic()
            was = self._vel_active
            try:
                mode = self._apply_velocity(vx, vy, wz)
            except Exception as e:  # noqa: BLE001
                self._event("cmd_vel", False, f"{type(e).__name__}: {e}")
                return
        if was != self._vel_active:
            self._event("cmd_vel", True, f"{mode}: vx={vx:.2f} vy={vy:.2f} wz={wz:.2f}")

    def to_world(self, frame: str, x: float, y: float, yaw: float) -> tuple[float, float, float] | None:
        """A planar pose in `frame` (map, odom or base_link) -> simulator world (odom) coordinates."""
        f = self.frames
        frame = frame.lstrip("/")
        if frame in ("", f["map"]):
            if not (self.cfg.map_to_odom and self.slam_pose is not None):
                return x, y, yaw   # map == odom (identity, or an external SLAM that started where the sim did)
            return _se2_compose(_se2_inv(self._map_odom), (x, y, yaw))
        if frame in (f["odom"], "world"):
            return x, y, yaw
        if frame == f["base_link"]:
            return _se2_compose(self._robot_pose(), (x, y, yaw))
        return None

    def _on_goal_pose(self, msg: dict) -> None:
        self.received["goal_pose"] += 1
        try:
            frame, x, y, _, yaw = msgs.pose_stamped_to_xyyaw(msg)
        except Exception as e:  # noqa: BLE001
            self._event("goal_pose", False, f"bad PoseStamped: {e}")
            return
        goal = self.to_world(frame, x, y, yaw)
        if goal is None:
            self._event("goal_pose", False, f"unsupported frame {frame!r} (use map, odom or base_link)")
            return
        gx, gy, gyaw = goal
        with self._cmd_lock:
            with self.s.lock:
                self.s.nav.set_goal([gx, gy], float(math.atan2(math.sin(gyaw), math.cos(gyaw))))
                self._run_sim()
            self._vel_active = False
        self._event("goal_pose", True, f"walking to ({gx:.2f}, {gy:.2f}) facing {math.degrees(gyaw):.0f} deg",
                    goal=[round(gx, 3), round(gy, 3), round(gyaw, 4)])

    def _on_arm_command(self, msg: dict) -> None:
        """std_msgs/String with JSON {"skill": reach|grasp|release|point_at|home|wave|raise, "side": left|right,
        "target": [x, y, z], "approach": [dx, dy, dz], "frame": odom|map|base_link}; plain "wave right" works too."""
        self.received["arm_command"] += 1
        text = str(msg.get("data", "")).strip()
        try:
            cmd = json.loads(text) if text.startswith("{") else dict(zip(("skill", "side"), text.split()))
            skill = str(cmd.get("skill", "")).lower()
            side = str(cmd.get("side", "right")).lower()
            if side not in ("left", "right"):
                raise ValueError("side must be left or right")
            robot = self.s.robot
            if skill in ("reach", "point_at"):
                target = self._arm_target(cmd)
                if skill == "reach":
                    res = robot.reach(side, *target, approach=cmd.get("approach"))
                else:
                    res = robot.point_at(side, *target)
            elif skill == "grasp":
                res = robot.grasp(side)
            elif skill == "release":
                res = robot.release(side)
            elif skill in ("home", "wave", "raise", "gesture"):
                res = robot.arm_gesture(side, cmd.get("gesture", skill) if skill == "gesture" else skill)
            else:
                raise ValueError(f"unknown skill {skill!r} (reach, grasp, release, point_at, home, wave, raise)")
            self._run_sim()
            self._event("arm_command", True, str(res), skill=skill, side=side)
        except Exception as e:  # noqa: BLE001 - reported on /simulatics/events
            self._event("arm_command", False, f"{type(e).__name__}: {e}")

    def _arm_target(self, cmd: dict) -> tuple[float, float, float]:
        x, y, z = (float(v) for v in cmd["target"])
        frame = str(cmd.get("frame", "odom")).lstrip("/")
        if frame == self.frames["base_link"]:
            with self.s.lock:
                info = self._info or _ModelInfo(self.s.model, self.cfg.camera)
                p, q = info.base_pose(self.s.data)
            w = np.asarray(p) + msgs.quat_rotate(q, (x, y, z))
            return float(w[0]), float(w[1]), float(w[2])
        goal = self.to_world(frame, x, y, 0.0)
        if goal is None:
            raise ValueError(f"unsupported frame {frame!r}")
        return goal[0], goal[1], z
