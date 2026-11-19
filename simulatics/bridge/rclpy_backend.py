"""Native ROS 2 transport through rclpy, for machines where a ROS 2 install is sourced.

Same interface as RosbridgeClient (transport.Transport), but publishers and subscriptions are real
ROS 2 ones on a node of this process, with real message types; no rosbridge_server needed.

rclpy is imported lazily: importing this module never fails, `available()` says whether rclpy can be
imported, and constructing `RclpyTransport` without it raises a RuntimeError that says what to do.

Message dicts (msgs.py JSON form) are converted to and from the generated Python classes through the
rosidl introspection API every generated message has (``get_fields_and_field_types``). uint8[]
fields accept base64 strings, as rosbridge does.
"""
from __future__ import annotations

import array
import base64
import importlib
import logging
import re
import threading

from .transport import Callback, RateLimiter, TopicStats

log = logging.getLogger("simulatics.bridge.rclpy")


def available() -> bool:
    """True when `import rclpy` works (a ROS 2 environment is sourced for this Python)."""
    try:
        importlib.import_module("rclpy")
    except Exception:  # noqa: BLE001 - ImportError, or a broken install raising something else
        return False
    return True


# --------------------------------------------------------------------------- dict <-> message conversion
_SEQ = re.compile(r"^sequence<([^,>]+)(?:,\s*\d+)?>$")
_FIXED = re.compile(r"^(.+)\[(\d+)\]$")
_FLOAT = {"float": "f", "double": "d", "float32": "f", "float64": "d"}
_INT = {"int8": "b", "uint8": "B", "int16": "h", "uint16": "H", "int32": "i", "uint32": "I",
        "int64": "q", "uint64": "Q", "octet": "B", "byte": "B", "char": "B"}


def import_msg_class(type_name: str):
    """'geometry_msgs/msg/Twist' or 'geometry_msgs/Twist' -> the generated class."""
    parts = type_name.split("/")
    pkg, name = parts[0], parts[-1]
    return getattr(importlib.import_module(f"{pkg}.msg"), name)


def _base(t: str) -> str:
    return t.split("<=")[0] if t.startswith(("string<=", "wstring<=")) else t


def dict_to_msg(cls, d: dict, resolve=import_msg_class):
    """Build an instance of the generated class `cls` from a msgs.py-style dict (missing fields keep defaults)."""
    msg = cls()
    for name, typ in cls.get_fields_and_field_types().items():
        if name not in d:
            continue
        setattr(msg, name, _to_field(typ, d[name], resolve))
    return msg


def _to_field(typ: str, v, resolve):
    m = _SEQ.match(typ) or _FIXED.match(typ)
    if m:
        elem = _base(m.group(1))
        if elem in ("uint8", "octet", "byte", "char") and isinstance(v, str):
            v = base64.b64decode(v)
        if elem in _FLOAT:
            if _FIXED.match(typ):
                import numpy as np
                return np.asarray([float(x) for x in v], dtype=np.float32 if _FLOAT[elem] == "f" else np.float64)
            return array.array(_FLOAT[elem], (float(x) for x in v))
        if elem in _INT:
            if _FIXED.match(typ):
                import numpy as np
                return np.asarray([int(x) for x in v], dtype=np.dtype(_INT[elem]))
            return array.array(_INT[elem], (int(x) for x in v))
        if elem in ("string", "wstring"):
            return [str(x) for x in v]
        if elem == "boolean":
            return [bool(x) for x in v]
        sub = resolve(elem)
        return [dict_to_msg(sub, x, resolve) for x in v]
    base = _base(typ)
    if base in _FLOAT:
        return float(v)
    if base in _INT:
        return int(v)
    if base == "boolean":
        return bool(v)
    if base in ("string", "wstring"):
        return str(v)
    return dict_to_msg(resolve(base), v, resolve)


def msg_to_dict(msg) -> dict:
    """Generated message instance -> msgs.py-style dict (uint8[] as base64, like rosbridge)."""
    out = {}
    for name, typ in msg.get_fields_and_field_types().items():
        v = getattr(msg, name)
        m = _SEQ.match(typ) or _FIXED.match(typ)
        if m:
            elem = _base(m.group(1))
            if elem in ("uint8", "octet", "byte", "char"):
                out[name] = base64.b64encode(bytes(bytearray(v))).decode()
            elif elem in _FLOAT:
                out[name] = [float(x) for x in v]
            elif elem in _INT:
                out[name] = [int(x) for x in v]
            elif elem in ("string", "wstring", "boolean"):
                out[name] = list(v)
            else:
                out[name] = [msg_to_dict(x) for x in v]
        elif hasattr(v, "get_fields_and_field_types"):
            out[name] = msg_to_dict(v)
        elif isinstance(v, (bytes, bytearray)):
            out[name] = int(v[0]) if len(v) == 1 else base64.b64encode(bytes(v)).decode()
        else:
            out[name] = v
    return out


# --------------------------------------------------------------------------- transport
class RclpyTransport:
    """transport.Transport on a native rclpy node (spun on its own thread)."""
    name = "rclpy"
    supports_nonfinite = True

    def __init__(self, node_name: str = "simulatics_bridge", namespace: str = "", domain_id: int | None = None):
        if not available():
            raise RuntimeError("rclpy is not importable in this Python: source a ROS 2 install first "
                               "(Linux/WSL2: `source /opt/ros/jazzy/setup.bash`; Windows: run the ROS 2 "
                               "`local_setup.bat` / pixi shell), with a matching Python version, or use the "
                               "rosbridge transport instead")
        self.node_name, self.namespace, self.domain_id = node_name, namespace, domain_id
        self._lock = threading.Lock()
        self._pubs: dict[str, tuple] = {}            # topic -> (publisher, class)
        self._subs: dict[str, object] = {}
        self._limits = RateLimiter()
        self.stats = TopicStats()
        self._ctx = self._node = self._executor = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.state, self.error = "idle", ""

    @property
    def url(self) -> str:
        return f"rclpy://{self.namespace or '/'}{self.node_name}"

    @property
    def connected(self) -> bool:
        return self._node is not None and self.state == "connected"

    def start(self) -> None:
        if self._node is not None:
            return
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        self._ctx = rclpy.Context()
        kw = {"domain_id": self.domain_id} if self.domain_id is not None else {}
        rclpy.init(context=self._ctx, **kw)
        self._node = rclpy.create_node(self.node_name, namespace=self.namespace or None, context=self._ctx)
        self._executor = SingleThreadedExecutor(context=self._ctx)
        self._executor.add_node(self._node)
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, name="rclpy-spin", daemon=True)
        self._thread.start()
        self.state = "connected"

    def _spin(self) -> None:
        while not self._stop.is_set() and self._ctx.ok():
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except Exception as e:  # noqa: BLE001
                self.error = f"{type(e).__name__}: {e}"

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self._thread = None
        try:
            if self._executor is not None:
                self._executor.shutdown()
            if self._node is not None:
                self._node.destroy_node()
            if self._ctx is not None and self._ctx.ok():
                import rclpy
                rclpy.shutdown(context=self._ctx)
        finally:
            self._node = self._executor = self._ctx = None
            self.state = "stopped"

    def _qos(self, latch: bool, depth: int):
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1 if latch else max(1, int(depth)),
                          reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL if latch else DurabilityPolicy.VOLATILE)

    def advertise(self, topic: str, msg_type: str, latch: bool = False, queue_size: int = 10,
                  max_rate_hz: float | None = None) -> None:
        cls = import_msg_class(msg_type)
        with self._lock:
            if topic not in self._pubs:
                self._pubs[topic] = (self._node.create_publisher(cls, topic, self._qos(latch, queue_size)), cls)
        self._limits.set_rate(topic, max_rate_hz)

    def unadvertise(self, topic: str) -> None:
        with self._lock:
            pub = self._pubs.pop(topic, None)
        if pub is not None:
            self._node.destroy_publisher(pub[0])

    def publish(self, topic: str, msg: dict) -> bool:
        with self._lock:
            entry = self._pubs.get(topic)
        if entry is None:
            raise KeyError(f"advertise {topic} before publishing to it")
        if not self._limits.allow(topic):
            self.stats.on_dropped(topic)
            return False
        pub, cls = entry
        pub.publish(dict_to_msg(cls, msg))
        self.stats.on_sent(topic)
        return True

    def subscribe(self, topic: str, msg_type: str, callback: Callback, throttle_ms: int = 0,
                  queue_length: int = 1) -> None:
        cls = import_msg_class(msg_type)
        limiter = RateLimiter()
        if throttle_ms > 0:
            limiter.set_rate(topic, 1000.0 / throttle_ms)

        def cb(m, topic=topic):
            if not limiter.allow(topic):
                return
            self.stats.on_received(topic)
            try:
                callback(msg_to_dict(m))
            except Exception:  # noqa: BLE001
                log.exception("callback for %s failed", topic)
        with self._lock:
            self._subs[topic] = self._node.create_subscription(cls, topic, cb, max(10, int(queue_length)))

    def unsubscribe(self, topic: str) -> None:
        with self._lock:
            sub = self._subs.pop(topic, None)
        if sub is not None:
            self._node.destroy_subscription(sub)

    def call_service(self, service: str, args: dict | None = None, timeout: float = 5.0,
                     srv_type: str | None = None) -> dict:
        """Needs srv_type ('std_srvs/srv/Trigger'): rclpy, unlike rosbridge, cannot look the type up."""
        if srv_type is None:
            raise NotImplementedError("the rclpy transport needs srv_type for call_service")
        pkg, _, name = srv_type.split("/")
        cls = getattr(importlib.import_module(f"{pkg}.srv"), name)
        client = self._node.create_client(cls, service)
        try:
            if not client.wait_for_service(timeout_sec=timeout):
                raise TimeoutError(f"service {service} not available")
            fut = client.call_async(dict_to_msg(cls.Request, args or {}))
            done = threading.Event()
            fut.add_done_callback(lambda _f: done.set())
            if not done.wait(timeout):
                raise TimeoutError(f"no response from {service} within {timeout} s")
            return msg_to_dict(fut.result())
        finally:
            self._node.destroy_client(client)

    def status(self) -> dict:
        with self._lock:
            adverts, subs = sorted(self._pubs), sorted(self._subs)
        return {"transport": self.name, "url": self.url, "state": self.state, "connected": self.connected,
                "error": self.error, "connects": 1 if self._node is not None else 0, "advertised": adverts,
                "subscribed": subs, "topics": self.stats.snapshot()}
