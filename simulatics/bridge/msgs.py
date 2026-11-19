"""ROS 2 messages as plain dicts, in the JSON shape rosbridge v2 and roslibjs use.

Every builder returns a dict whose keys are exactly the fields of the ROS 2 message (ROS 2 names:
``builtin_interfaces/Time`` is ``{"sec", "nanosec"}``, ``std_msgs/Header`` has no ``seq``,
``sensor_msgs/CameraInfo`` uses lower-case ``d, k, r, p``). ``uint8[]`` fields (CompressedImage.data)
are base64 strings, which is how rosbridge carries binary data in JSON; it also accepts lists of ints.

`MSG_DEFS` holds the ``.msg`` definitions the builders follow (ROS 2 common_interfaces /
rcl_interfaces, as in Humble and Jazzy) and `validate` checks a dict against them; the mock
rosbridge server uses it to reject malformed messages the way a real server would.

Frame conventions (REP-103, REP-105)
------------------------------------
* Axes are right-handed. Body frames: x forward, y left, z up. World frames (map, odom): x east /
  forward at spawn, y north / left, z up (ENU). Units: metres, radians, seconds.
* Camera optical frames (suffix ``_optical``): z forward, x right, y down. Images and CameraInfo are
  stamped in the optical frame.
* Quaternions are ``{x, y, z, w}`` (ROS order). MuJoCo stores ``w, x, y, z``: use `quat_wxyz_to_xyzw`.

Frames published by the bridge::

    map --(identity, or a SLAM estimate)--> odom --(robot pose)--> base_link
    base_link --> <URDF root link> (identity) --> ... every URDF link
    base_link --> lidar_link, imu_link, head_cam --> head_cam_optical
"""
from __future__ import annotations

import base64
import math
import re
from typing import Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------- frames
MAP = "map"
ODOM = "odom"
BASE_LINK = "base_link"
LIDAR_LINK = "lidar_link"
IMU_LINK = "imu_link"
HEAD_CAM = "head_cam"
OPTICAL_SUFFIX = "_optical"
HEAD_CAM_OPTICAL = HEAD_CAM + OPTICAL_SUFFIX

# Rotation of a camera optical frame (z fwd, x right, y down) relative to the camera's body frame
# (x fwd, y left, z up): roll -90 deg then yaw -90 deg, quaternion (x, y, z, w).
OPTICAL_IN_BODY_XYZW = (-0.5, 0.5, -0.5, 0.5)

# --------------------------------------------------------------------------- type names
STRING = "std_msgs/msg/String"
HEADER = "std_msgs/msg/Header"
TIME = "builtin_interfaces/msg/Time"
CLOCK = "rosgraph_msgs/msg/Clock"
JOINT_STATE = "sensor_msgs/msg/JointState"
LASER_SCAN = "sensor_msgs/msg/LaserScan"
IMU = "sensor_msgs/msg/Imu"
COMPRESSED_IMAGE = "sensor_msgs/msg/CompressedImage"
CAMERA_INFO = "sensor_msgs/msg/CameraInfo"
ODOMETRY = "nav_msgs/msg/Odometry"
OCCUPANCY_GRID = "nav_msgs/msg/OccupancyGrid"
PATH = "nav_msgs/msg/Path"
TWIST = "geometry_msgs/msg/Twist"
POSE_STAMPED = "geometry_msgs/msg/PoseStamped"
TRANSFORM_STAMPED = "geometry_msgs/msg/TransformStamped"
TF_MESSAGE = "tf2_msgs/msg/TFMessage"

# --------------------------------------------------------------------------- .msg definitions
# Transcribed from the ROS 2 interface definitions (common_interfaces, rcl_interfaces, geometry2),
# comments and constants dropped. Unqualified nested types live in the same package.
MSG_DEFS: dict[str, str] = {
    "builtin_interfaces/msg/Time": "int32 sec\nuint32 nanosec",
    "std_msgs/msg/Header": "builtin_interfaces/Time stamp\nstring frame_id",
    "std_msgs/msg/String": "string data",
    "rosgraph_msgs/msg/Clock": "builtin_interfaces/Time clock",
    "geometry_msgs/msg/Vector3": "float64 x\nfloat64 y\nfloat64 z",
    "geometry_msgs/msg/Point": "float64 x\nfloat64 y\nfloat64 z",
    "geometry_msgs/msg/Quaternion": "float64 x 0\nfloat64 y 0\nfloat64 z 0\nfloat64 w 1",
    "geometry_msgs/msg/Pose": "Point position\nQuaternion orientation",
    "geometry_msgs/msg/PoseStamped": "std_msgs/Header header\nPose pose",
    "geometry_msgs/msg/PoseWithCovariance": "Pose pose\nfloat64[36] covariance",
    "geometry_msgs/msg/Twist": "Vector3 linear\nVector3 angular",
    "geometry_msgs/msg/TwistWithCovariance": "Twist twist\nfloat64[36] covariance",
    "geometry_msgs/msg/Transform": "Vector3 translation\nQuaternion rotation",
    "geometry_msgs/msg/TransformStamped": "std_msgs/Header header\nstring child_frame_id\nTransform transform",
    "tf2_msgs/msg/TFMessage": "geometry_msgs/TransformStamped[] transforms",
    "sensor_msgs/msg/JointState": ("std_msgs/Header header\nstring[] name\nfloat64[] position\n"
                                   "float64[] velocity\nfloat64[] effort"),
    "sensor_msgs/msg/LaserScan": ("std_msgs/Header header\nfloat32 angle_min\nfloat32 angle_max\n"
                                  "float32 angle_increment\nfloat32 time_increment\nfloat32 scan_time\n"
                                  "float32 range_min\nfloat32 range_max\nfloat32[] ranges\nfloat32[] intensities"),
    "sensor_msgs/msg/Imu": ("std_msgs/Header header\ngeometry_msgs/Quaternion orientation\n"
                            "float64[9] orientation_covariance\ngeometry_msgs/Vector3 angular_velocity\n"
                            "float64[9] angular_velocity_covariance\ngeometry_msgs/Vector3 linear_acceleration\n"
                            "float64[9] linear_acceleration_covariance"),
    "sensor_msgs/msg/CompressedImage": "std_msgs/Header header\nstring format\nuint8[] data",
    "sensor_msgs/msg/RegionOfInterest": ("uint32 x_offset\nuint32 y_offset\nuint32 height\nuint32 width\n"
                                         "bool do_rectify"),
    "sensor_msgs/msg/CameraInfo": ("std_msgs/Header header\nuint32 height\nuint32 width\nstring distortion_model\n"
                                   "float64[] d\nfloat64[9] k\nfloat64[9] r\nfloat64[12] p\nuint32 binning_x\n"
                                   "uint32 binning_y\nRegionOfInterest roi"),
    "nav_msgs/msg/MapMetaData": ("builtin_interfaces/Time map_load_time\nfloat32 resolution\nuint32 width\n"
                                 "uint32 height\ngeometry_msgs/Pose origin"),
    "nav_msgs/msg/OccupancyGrid": "std_msgs/Header header\nMapMetaData info\nint8[] data",
    "nav_msgs/msg/Odometry": ("std_msgs/Header header\nstring child_frame_id\n"
                              "geometry_msgs/PoseWithCovariance pose\ngeometry_msgs/TwistWithCovariance twist"),
    "nav_msgs/msg/Path": "std_msgs/Header header\ngeometry_msgs/PoseStamped[] poses",
}

_INT_RANGES = {"int8": (-2 ** 7, 2 ** 7 - 1), "uint8": (0, 2 ** 8 - 1), "byte": (0, 255), "char": (0, 255),
               "int16": (-2 ** 15, 2 ** 15 - 1), "uint16": (0, 2 ** 16 - 1),
               "int32": (-2 ** 31, 2 ** 31 - 1), "uint32": (0, 2 ** 32 - 1),
               "int64": (-2 ** 63, 2 ** 63 - 1), "uint64": (0, 2 ** 64 - 1)}
_FLOATS = {"float32", "float64"}
PRIMITIVES = set(_INT_RANGES) | _FLOATS | {"bool", "string", "wstring"}
_BINARY = {"uint8", "byte", "char"}   # arrays of these travel as base64 strings
_FIELD_RE = re.compile(r"^(?P<type>[\w/]+)(?P<bound><=\d+)?(?P<array>\[(?:<=)?\d*\])?\s+(?P<name>\w+)(?:\s+.*)?$")


def normalize_type(name: str, package: str = "") -> str:
    """'geometry_msgs/Point', 'geometry_msgs/msg/Point' or 'Point' (inside `package`) -> 'pkg/msg/Name'."""
    parts = name.split("/")
    if len(parts) == 3:
        return name
    if len(parts) == 2:
        return f"{parts[0]}/msg/{parts[1]}"
    return f"{package}/msg/{name}" if package else name


def parse_fields(msg_type: str) -> list[tuple[str, str, int | None]]:
    """[(field name, element type ('float64' or 'pkg/msg/Name'), array length: None scalar, -1 unbounded, n fixed)]"""
    msg_type = normalize_type(msg_type)
    if msg_type not in MSG_DEFS:
        raise KeyError(f"no definition for {msg_type}")
    pkg = msg_type.split("/")[0]
    out = []
    for line in MSG_DEFS[msg_type].splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = _FIELD_RE.match(line)
        if m is None:
            raise ValueError(f"cannot parse field line {line!r} of {msg_type}")
        t = m["type"]
        t = t if t in PRIMITIVES else normalize_type(t, pkg)
        arr = m["array"]
        n = None if arr is None else (-1 if arr in ("[]",) or arr.startswith("[<=") else int(arr[1:-1]))
        out.append((m["name"], t, n))
    return out


def validate(msg_type: str, msg, strict: bool = True, allow_nonfinite: bool = False, _path: str = "") -> list[str]:
    """Check a message dict against its ROS 2 definition. Returns a list of problems (empty = valid).

    strict: every field must be present (rosbridge itself fills missing fields with defaults, so the
    mock server validates with strict=False). Unknown fields are always an error, as in rosbridge.
    """
    path = _path or normalize_type(msg_type)
    if not isinstance(msg, dict):
        return [f"{path}: expected an object, got {type(msg).__name__}"]
    try:
        fields = parse_fields(msg_type)
    except KeyError as e:
        return [f"{path}: {e}"]
    errs = []
    names = {f[0] for f in fields}
    for k in msg:
        if k not in names:
            errs.append(f"{path}.{k}: no such field in {normalize_type(msg_type)}")
    for name, t, n in fields:
        p = f"{path}.{name}"
        if name not in msg:
            if strict:
                errs.append(f"{p}: missing")
            continue
        v = msg[name]
        if n is None:
            errs += _check_value(t, v, p, strict, allow_nonfinite)
            continue
        if t in _BINARY and isinstance(v, str):
            try:
                raw = base64.b64decode(v, validate=True)
            except Exception:  # noqa: BLE001
                errs.append(f"{p}: not valid base64")
                continue
            if n >= 0 and len(raw) != n:
                errs.append(f"{p}: expected {n} bytes, got {len(raw)}")
            continue
        if not isinstance(v, (list, tuple)):
            errs.append(f"{p}: expected an array, got {type(v).__name__}")
            continue
        if n >= 0 and len(v) != n:
            errs.append(f"{p}: expected {n} elements, got {len(v)}")
        for i, x in enumerate(v):
            e = _check_value(t, x, f"{p}[{i}]", strict, allow_nonfinite)
            if e:
                errs += e
                break  # one bad element is enough to report
    return errs


def _check_value(t: str, v, p: str, strict: bool, allow_nonfinite: bool) -> list[str]:
    if t in _FLOATS:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return [f"{p}: expected {t}, got {type(v).__name__}"]
        if not allow_nonfinite and not math.isfinite(v):
            return [f"{p}: non-finite {t} cannot be sent as JSON"]
        return []
    if t in _INT_RANGES:
        if isinstance(v, bool) or not isinstance(v, int):
            return [f"{p}: expected {t}, got {type(v).__name__}"]
        lo, hi = _INT_RANGES[t]
        return [] if lo <= v <= hi else [f"{p}: {v} out of range for {t}"]
    if t == "bool":
        return [] if isinstance(v, bool) else [f"{p}: expected bool, got {type(v).__name__}"]
    if t in ("string", "wstring"):
        return [] if isinstance(v, str) else [f"{p}: expected string, got {type(v).__name__}"]
    return validate(t, v, strict, allow_nonfinite, p)


# --------------------------------------------------------------------------- small math (ROS xyzw quaternions)
def quat_wxyz_to_xyzw(q) -> tuple[float, float, float, float]:
    w, x, y, z = (float(v) for v in q)
    return x, y, z, w


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


def quat_to_yaw(q) -> float:
    """Yaw (rad, about +z) of an xyzw quaternion (dict or sequence)."""
    x, y, z, w = _xyzw(q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_mul(a, b) -> tuple[float, float, float, float]:
    ax, ay, az, aw = _xyzw(a)
    bx, by, bz, bw = _xyzw(b)
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def quat_conj(q) -> tuple[float, float, float, float]:
    x, y, z, w = _xyzw(q)
    return -x, -y, -z, w


def quat_rotate(q, v) -> np.ndarray:
    """Rotate vector v by the xyzw quaternion q."""
    return quat_to_matrix(q) @ np.asarray(v, float)


def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = _xyzw(q)
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n > 0 else 0.0
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                     [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def matrix_to_quat(R) -> tuple[float, float, float, float]:
    """3x3 rotation matrix -> xyzw quaternion with w >= 0."""
    R = np.asarray(R, float).reshape(3, 3)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return tuple(float(v) for v in q)


def _xyzw(q) -> tuple[float, float, float, float]:
    if isinstance(q, dict):
        return float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"])
    x, y, z, w = q
    return float(x), float(y), float(z), float(w)


# --------------------------------------------------------------------------- builders: primitives
def time_msg(t: float) -> dict:
    """builtin_interfaces/Time from seconds (negative / non-finite -> 0)."""
    t = float(t)
    if not math.isfinite(t) or t < 0:
        t = 0.0
    sec = int(math.floor(t))
    nanosec = int(round((t - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec, nanosec = sec + 1, nanosec - 1_000_000_000
    return {"sec": sec, "nanosec": nanosec}


def time_to_sec(m: dict) -> float:
    return float(m.get("sec", 0)) + float(m.get("nanosec", 0)) * 1e-9


def _stamp(stamp) -> dict:
    return dict(stamp) if isinstance(stamp, dict) else time_msg(stamp)


def header(stamp, frame_id: str) -> dict:
    """std_msgs/Header; stamp is seconds or a Time dict."""
    return {"stamp": _stamp(stamp), "frame_id": str(frame_id)}


def string(data: str) -> dict:
    return {"data": str(data)}


def clock(t: float) -> dict:
    return {"clock": time_msg(t)}


def vector3(v=(0.0, 0.0, 0.0)) -> dict:
    x, y, z = (float(a) for a in v)
    return {"x": x, "y": y, "z": z}


def point(p=(0.0, 0.0, 0.0)) -> dict:
    return vector3(p)


def quaternion(q=(0.0, 0.0, 0.0, 1.0)) -> dict:
    x, y, z, w = _xyzw(q)
    return {"x": x, "y": y, "z": z, "w": w}


def pose(position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0)) -> dict:
    return {"position": point(position), "orientation": quaternion(orientation)}


def twist(linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0)) -> dict:
    return {"linear": vector3(linear), "angular": vector3(angular)}


def _floats(a, decimals: int | None = None) -> list[float]:
    arr = np.asarray(a, float).ravel()
    if decimals is not None:
        arr = np.round(arr, decimals)
    return [float(v) for v in arr]


def _cov(c, n: int) -> list[float]:
    if c is None:
        return [0.0] * n
    if np.isscalar(c):  # one variance for every axis -> diagonal
        k = int(round(math.sqrt(n)))
        return _floats(np.eye(k) * float(c))
    out = _floats(c)
    if len(out) == int(round(math.sqrt(n))):  # a diagonal
        out = _floats(np.diag(out))
    if len(out) != n:
        raise ValueError(f"covariance needs {n} values, got {len(out)}")
    return out


# --------------------------------------------------------------------------- builders: messages
def pose_stamped(stamp, frame_id: str, position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0)) -> dict:
    return {"header": header(stamp, frame_id), "pose": pose(position, orientation)}


def transform_stamped(stamp, parent: str, child: str, translation=(0.0, 0.0, 0.0),
                      rotation=(0.0, 0.0, 0.0, 1.0)) -> dict:
    return {"header": header(stamp, parent), "child_frame_id": str(child),
            "transform": {"translation": vector3(translation), "rotation": quaternion(rotation)}}


def tf_message(transforms: Iterable[dict]) -> dict:
    return {"transforms": list(transforms)}


def joint_state(stamp, names: Sequence[str], position=(), velocity=(), effort=(), frame_id: str = "",
                decimals: int | None = 6) -> dict:
    return {"header": header(stamp, frame_id), "name": [str(n) for n in names],
            "position": _floats(position, decimals), "velocity": _floats(velocity, decimals),
            "effort": _floats(effort, decimals)}


def laser_scan(stamp, frame_id: str, ranges, angle_min: float, angle_increment: float, range_min: float,
               range_max: float, scan_time: float = 0.0, time_increment: float = 0.0, intensities=(),
               decimals: int | None = 4) -> dict:
    r = _floats(ranges, decimals)
    n = len(r)
    return {"header": header(stamp, frame_id), "angle_min": float(angle_min),
            "angle_max": float(angle_min + angle_increment * max(n - 1, 0)),
            "angle_increment": float(angle_increment), "time_increment": float(time_increment),
            "scan_time": float(scan_time), "range_min": float(range_min), "range_max": float(range_max),
            "ranges": r, "intensities": _floats(intensities, decimals)}


def imu(stamp, frame_id: str, orientation=None, angular_velocity=None, linear_acceleration=None,
        orientation_cov=1e-4, angular_velocity_cov=1e-4, linear_acceleration_cov=1e-2) -> dict:
    """sensor_msgs/Imu. A quantity given as None is marked unknown (covariance[0] = -1, REP-145)."""
    def part(v, cov, zero):
        if v is None:
            return zero, [-1.0] + [0.0] * 8
        return v, _cov(cov, 9)
    q, qc = part(orientation, orientation_cov, (0.0, 0.0, 0.0, 1.0))
    w, wc = part(angular_velocity, angular_velocity_cov, (0.0, 0.0, 0.0))
    a, ac = part(linear_acceleration, linear_acceleration_cov, (0.0, 0.0, 0.0))
    return {"header": header(stamp, frame_id), "orientation": quaternion(q), "orientation_covariance": qc,
            "angular_velocity": vector3(w), "angular_velocity_covariance": wc,
            "linear_acceleration": vector3(a), "linear_acceleration_covariance": ac}


def compressed_image(stamp, frame_id: str, data: bytes, fmt: str = "jpeg") -> dict:
    """sensor_msgs/CompressedImage; data is the encoded image (sent as base64)."""
    return {"header": header(stamp, frame_id), "format": str(fmt), "data": base64.b64encode(bytes(data)).decode()}


def camera_info(stamp, frame_id: str, width: int, height: int, fovy_deg: float | None = None,
                fx: float | None = None, fy: float | None = None, cx: float | None = None,
                cy: float | None = None) -> dict:
    """sensor_msgs/CameraInfo of an ideal pinhole camera (no distortion, square pixels by default).

    Give the vertical field of view (MuJoCo's ``fovy``) or fx/fy directly. The principal point
    defaults to the image centre, ((w - 1) / 2, (h - 1) / 2) in ROS pixel-centre coordinates."""
    w, h = int(width), int(height)
    if fy is None:
        if fovy_deg is None:
            raise ValueError("camera_info needs fovy_deg or fy")
        fy = (h / 2.0) / math.tan(math.radians(float(fovy_deg)) / 2.0)
    fx = float(fy if fx is None else fx)
    fy = float(fy)
    cx = (w - 1) / 2.0 if cx is None else float(cx)
    cy = (h - 1) / 2.0 if cy is None else float(cy)
    return {"header": header(stamp, frame_id), "height": h, "width": w, "distortion_model": "plumb_bob",
            "d": [0.0] * 5, "k": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
            "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "p": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
            "binning_x": 0, "binning_y": 0,
            "roi": {"x_offset": 0, "y_offset": 0, "height": 0, "width": 0, "do_rectify": False}}


def odometry(stamp, frame_id: str, child_frame_id: str, position, orientation, linear=(0.0, 0.0, 0.0),
             angular=(0.0, 0.0, 0.0), pose_cov=1e-4, twist_cov=1e-3) -> dict:
    """nav_msgs/Odometry. The pose is in frame_id, the twist in child_frame_id (as in the message docs)."""
    return {"header": header(stamp, frame_id), "child_frame_id": str(child_frame_id),
            "pose": {"pose": pose(position, orientation), "covariance": _cov(pose_cov, 36)},
            "twist": {"twist": twist(linear, angular), "covariance": _cov(twist_cov, 36)}}


def occupancy_grid(stamp, frame_id: str, cells, resolution: float, origin_xy=(0.0, 0.0), origin_yaw: float = 0.0,
                   map_load_time=None) -> dict:
    """nav_msgs/OccupancyGrid from an int array (height, width) indexed [row = y, col = x]:
    -1 unknown, 0..100 occupancy probability in percent. origin_xy is the corner of cell (0, 0)."""
    c = np.asarray(cells)
    if c.ndim != 2:
        raise ValueError("occupancy_grid needs a 2D array")
    h, w = c.shape
    data = np.clip(c, -1, 100).astype(np.int8).ravel()  # row-major, x fastest, starting at the origin cell
    return {"header": header(stamp, frame_id),
            "info": {"map_load_time": _stamp(stamp if map_load_time is None else map_load_time),
                     "resolution": float(resolution), "width": int(w), "height": int(h),
                     "origin": pose((float(origin_xy[0]), float(origin_xy[1]), 0.0), yaw_to_quat(origin_yaw))},
            "data": data.tolist()}


def path(stamp, frame_id: str, points_xy, yaws=None, z: float = 0.0) -> dict:
    """nav_msgs/Path through world points (n, 2). Without yaws each pose faces the next point."""
    pts = np.asarray(points_xy, float).reshape(-1, 2)
    n = len(pts)
    if yaws is None:
        d = np.diff(pts, axis=0)
        yaws = np.arctan2(d[:, 1], d[:, 0]) if n > 1 else np.zeros(n)
        yaws = np.append(yaws, yaws[-1]) if n > 1 else np.zeros(n)
    poses = [pose_stamped(stamp, frame_id, (p[0], p[1], z), yaw_to_quat(float(a))) for p, a in zip(pts, yaws)]
    return {"header": header(stamp, frame_id), "poses": poses}


# --------------------------------------------------------------------------- readers (incoming messages)
def twist_to_tuple(msg: dict) -> tuple[float, float, float, float, float, float]:
    """geometry_msgs/Twist -> (vx, vy, vz, wx, wy, wz); missing fields read as 0."""
    lin, ang = msg.get("linear") or {}, msg.get("angular") or {}
    vals = [lin.get("x", 0.0), lin.get("y", 0.0), lin.get("z", 0.0), ang.get("x", 0.0), ang.get("y", 0.0),
            ang.get("z", 0.0)]
    out = tuple(float(v) for v in vals)
    if not all(math.isfinite(v) for v in out):
        raise ValueError("twist has non-finite values")
    return out  # type: ignore[return-value]


def pose_stamped_to_xyyaw(msg: dict) -> tuple[str, float, float, float, float]:
    """geometry_msgs/PoseStamped -> (frame_id, x, y, z, yaw rad)."""
    hdr = msg.get("header") or {}
    p = (msg.get("pose") or {}).get("position") or {}
    q = (msg.get("pose") or {}).get("orientation") or {"x": 0, "y": 0, "z": 0, "w": 1}
    x, y, z = float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0))
    q = {k: float(q.get(k, 1.0 if k == "w" else 0.0)) for k in "xyzw"}
    if not all(math.isfinite(v) for v in (x, y, z, *q.values())):
        raise ValueError("pose has non-finite values")
    if q["x"] == q["y"] == q["z"] == q["w"] == 0.0:  # an all-zero quaternion: treat as "no heading"
        q["w"] = 1.0
    return str(hdr.get("frame_id", "")), x, y, z, quat_to_yaw(q)
