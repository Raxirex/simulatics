"""rosbridge v2 protocol client: talk to a running rosbridge_server over a websocket.

    client = RosbridgeClient("ws://localhost:9090")
    client.start()                                       # connects in the background, reconnects on loss
    client.advertise("/chatter", "std_msgs/msg/String")
    client.publish("/chatter", {"data": "hi"})
    client.subscribe("/cmd_vel", "geometry_msgs/msg/Twist", on_twist)
    client.stop()

Operations sent: advertise (with ``latch`` and the newer ``qos`` object, so both older and 2.1+
servers make latched topics transient-local), unadvertise, publish, subscribe (``throttle_rate``,
``queue_length``), unsubscribe, call_service. Operations handled: publish (to subscription
callbacks), service_response, status (logged), fragment (reassembled).

Threads: one I/O thread that connects, sends and reconnects with exponential backoff, plus one
reader thread per live connection (callbacks run there). `publish` never blocks on the network:
each topic has a one-message mailbox, so a slow link drops stale messages instead of queueing them.
After a reconnect every advertisement and subscription is re-sent, followed by the last message of
each latched topic.

Written from the public protocol description (ROSBRIDGE_PROTOCOL.md of rosbridge_suite); no code
from rosbridge_suite or roslibjs is used.
"""
from __future__ import annotations

import collections
import itertools
import json
import logging
import random
import threading
import time

import numpy as np

from .transport import Callback, RateLimiter, TopicStats

log = logging.getLogger("simulatics.bridge.rosbridge")

LATCHED_QOS = {"history": "keep_last", "depth": 1, "reliability": "reliable", "durability": "transient_local"}


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (bytes, bytearray)):
        import base64
        return base64.b64encode(bytes(o)).decode()
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")


def encode(op: dict) -> str:
    """Compact JSON; NaN / inf are refused (they are not JSON and rosbridge's parsers differ on them)."""
    return json.dumps(op, separators=(",", ":"), allow_nan=False, default=_json_default)


class RosbridgeClient:
    """Transport over the rosbridge v2 JSON protocol (see module docstring)."""
    name = "rosbridge"
    supports_nonfinite = False

    def __init__(self, url: str = "ws://localhost:9090", *, reconnect: bool = True,
                 backoff_initial: float = 0.5, backoff_max: float = 10.0, open_timeout: float = 5.0,
                 ping_interval: float | None = 20.0, compression: str | None = "deflate"):
        self.url = url
        self.reconnect = reconnect
        self.backoff_initial, self.backoff_max = float(backoff_initial), float(backoff_max)
        self.open_timeout = float(open_timeout)
        self.ping_interval = ping_interval
        self.compression = compression
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._ids = itertools.count(1)
        self._client_tag = f"simulatics_{random.randrange(16 ** 6):06x}"
        self._adverts: dict[str, dict] = {}            # topic -> advertise op
        self._subs: dict[str, dict] = {}               # topic -> {"op": subscribe op, "callbacks": [...]}
        self._latched: dict[str, str] = {}             # topic -> last encoded publish op
        self._ctrl: collections.deque[str] = collections.deque()               # control ops, in order
        self._data: "collections.OrderedDict[str, str]" = collections.OrderedDict()  # topic -> newest publish
        self._calls: dict[str, dict] = {}              # call_service id -> {"event", "response"}
        self._fragments: dict[str, dict[int, str]] = {}
        self._limits = RateLimiter()
        self.stats = TopicStats()
        self._ws = None
        self._live = False                             # a connection is up and the registry was replayed
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._reader: threading.Thread | None = None
        self.state = "idle"                            # idle | connecting | connected | disconnected | stopped
        self.error = ""
        self.connects = 0                              # successful connections so far
        self.last_status: dict | None = None           # last "status" op from the server

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="rosbridge-io", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Unadvertise / unsubscribe (best effort), close the socket and join the threads."""
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout)
        self._thread = None
        with self._lock:
            for c in self._calls.values():
                c["event"].set()
        self.state = "stopped"

    @property
    def connected(self) -> bool:
        return self._live

    def wait_connected(self, timeout: float = 5.0) -> bool:
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            if self._live:
                return True
            time.sleep(0.02)
        return self._live

    # ------------------------------------------------------------------ API
    def advertise(self, topic: str, msg_type: str, latch: bool = False, queue_size: int = 10,
                  max_rate_hz: float | None = None) -> None:
        op = {"op": "advertise", "id": f"advertise:{topic}:{self._client_tag}", "topic": topic, "type": msg_type,
              "latch": bool(latch), "queue_size": int(queue_size)}
        if latch:
            op["qos"] = dict(LATCHED_QOS)
        self._limits.set_rate(topic, max_rate_hz)
        with self._cond:
            self._adverts[topic] = op
            if self._live:
                self._ctrl.append(encode(op))
                self._cond.notify_all()

    def unadvertise(self, topic: str) -> None:
        with self._cond:
            op = self._adverts.pop(topic, None)
            self._latched.pop(topic, None)
            self._data.pop(topic, None)
            if op is not None and self._live:
                self._ctrl.append(encode({"op": "unadvertise", "id": op["id"], "topic": topic}))
                self._cond.notify_all()

    def publish(self, topic: str, msg: dict) -> bool:
        with self._lock:
            adv = self._adverts.get(topic)
        if adv is None:
            raise KeyError(f"advertise {topic} before publishing to it")
        if not self._limits.allow(topic):
            self.stats.on_dropped(topic)
            return False
        payload = encode({"op": "publish", "id": f"publish:{topic}:{next(self._ids)}", "topic": topic, "msg": msg})
        with self._cond:
            if adv.get("latch"):
                self._latched[topic] = payload
            if not self._live:
                self.stats.on_dropped(topic)
                return False
            if topic in self._data:  # the previous one never left: the link is slower than the producer
                self.stats.on_dropped(topic)
                del self._data[topic]
            self._data[topic] = payload
            self._cond.notify_all()
        return True

    def subscribe(self, topic: str, msg_type: str, callback: Callback, throttle_ms: int = 0,
                  queue_length: int = 1) -> None:
        with self._cond:
            sub = self._subs.get(topic)
            if sub is None:
                op = {"op": "subscribe", "id": f"subscribe:{topic}:{self._client_tag}", "topic": topic,
                      "type": msg_type, "throttle_rate": int(throttle_ms), "queue_length": int(queue_length)}
                sub = self._subs[topic] = {"op": op, "callbacks": []}
                if self._live:
                    self._ctrl.append(encode(op))
                    self._cond.notify_all()
            sub["callbacks"].append(callback)

    def unsubscribe(self, topic: str) -> None:
        with self._cond:
            sub = self._subs.pop(topic, None)
            if sub is not None and self._live:
                self._ctrl.append(encode({"op": "unsubscribe", "id": sub["op"]["id"], "topic": topic}))
                self._cond.notify_all()

    def call_service(self, service: str, args: dict | list | None = None, timeout: float = 5.0) -> dict:
        """Blocking service call through rosbridge. Raises TimeoutError, ConnectionError or RuntimeError
        (the service reported failure)."""
        if not self._live:
            raise ConnectionError(f"not connected to {self.url}")
        cid = f"call_service:{service}:{next(self._ids)}"
        entry = {"event": threading.Event(), "response": None}
        op = {"op": "call_service", "id": cid, "service": service, "args": args if args is not None else {},
              "timeout": float(timeout)}
        with self._cond:
            self._calls[cid] = entry
            self._ctrl.append(encode(op))
            self._cond.notify_all()
        try:
            if not entry["event"].wait(timeout) or entry["response"] is None:
                raise TimeoutError(f"no response from {service} within {timeout} s")
        finally:
            with self._lock:
                self._calls.pop(cid, None)
        resp = entry["response"]
        if not resp.get("result", False):
            raise RuntimeError(f"{service} failed: {resp.get('values')}")
        values = resp.get("values")
        return values if isinstance(values, dict) else {"values": values}

    def status(self) -> dict:
        with self._lock:
            adverts, subs = sorted(self._adverts), sorted(self._subs)
        return {"transport": self.name, "url": self.url, "state": self.state, "connected": self._live,
                "error": self.error, "connects": self.connects, "advertised": adverts, "subscribed": subs,
                "topics": self.stats.snapshot(), "last_status": self.last_status}

    # ------------------------------------------------------------------ I/O thread
    def _run(self) -> None:
        from websockets.sync.client import connect
        delay = self.backoff_initial
        while not self._stop.is_set():
            self.state = "connecting"
            opened = False
            try:
                # context-manager form: what websockets >= 16 expects (and older versions support)
                with connect(self.url, open_timeout=self.open_timeout, close_timeout=1.0, max_size=None,
                             ping_interval=self.ping_interval, ping_timeout=self.ping_interval,
                             compression=self.compression, proxy=None) as ws:
                    opened = True
                    delay = self.backoff_initial
                    self._serve(ws)
            except Exception as e:  # noqa: BLE001 - refused, DNS, handshake, timeout: all mean "retry later"
                if not opened:
                    self.error = f"connect {self.url}: {type(e).__name__}: {e}"
                    self.state = "disconnected"
                    if not self.reconnect:
                        break
                    self._stop.wait(delay * random.uniform(0.85, 1.15))
                    delay = min(delay * 2.0, self.backoff_max)
                    continue
            if not self._stop.is_set():
                self.state = "disconnected"
                if not self.reconnect:
                    break
                self._stop.wait(delay)
        self.state = "stopped" if self._stop.is_set() else self.state

    def _serve(self, ws) -> None:
        """Run one connection: replay the registry, then send until the link drops or stop() is called."""
        lost = threading.Event()
        with self._cond:
            self._ws = ws
            self._ctrl.clear()
            self._data.clear()
            replay = [encode(op) for op in self._adverts.values()]
            replay += [encode(s["op"]) for s in self._subs.values()]
            replay += [p for t, p in self._latched.items() if t in self._adverts]
            self._ctrl.extend(replay)
            self._live = True
        self.connects += 1
        self.state, self.error = "connected", ""
        log.info("connected to %s", self.url)
        self._reader = threading.Thread(target=self._read, args=(ws, lost), name="rosbridge-reader", daemon=True)
        self._reader.start()
        try:
            while True:
                with self._cond:
                    while not (self._ctrl or self._data or lost.is_set() or self._stop.is_set()):
                        self._cond.wait(0.5)
                    if lost.is_set() or self._stop.is_set():
                        break
                    ctrl = list(self._ctrl)
                    self._ctrl.clear()
                    data = list(self._data.items())
                    self._data.clear()
                for payload in ctrl:
                    ws.send(payload)
                for topic, payload in data:
                    ws.send(payload)
                    self.stats.on_sent(topic)
        except Exception as e:  # noqa: BLE001 - ConnectionClosed or a socket error: reconnect
            if not self._stop.is_set():
                self.error = f"connection lost: {type(e).__name__}: {e}"
        finally:
            with self._cond:
                self._live = False
                self._ws = None
            if self._stop.is_set() and not lost.is_set():
                self._goodbye(ws)
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass
            if self._reader is not None:
                self._reader.join(3.0)
                self._reader = None
            if not self._stop.is_set():
                log.warning("disconnected from %s (%s)", self.url, self.error or "closed")

    def _goodbye(self, ws) -> None:
        """Best effort: tell the server we leave, so it drops our publishers / subscriptions at once."""
        with self._lock:
            ops = [{"op": "unadvertise", "id": op["id"], "topic": t} for t, op in self._adverts.items()]
            ops += [{"op": "unsubscribe", "id": s["op"]["id"], "topic": t} for t, s in self._subs.items()]
        try:
            for op in ops:
                ws.send(encode(op))
        except Exception:  # noqa: BLE001
            pass

    def _read(self, ws, lost: threading.Event) -> None:
        try:
            for raw in ws:
                if isinstance(raw, (bytes, bytearray)):
                    continue  # CBOR / BSON are only sent when asked for; we never ask
                try:
                    self._handle(json.loads(raw))
                except Exception as e:  # noqa: BLE001 - one bad message must not kill the link
                    log.warning("bad message from rosbridge: %s: %s", type(e).__name__, e)
        except Exception as e:  # noqa: BLE001
            if not self._stop.is_set():
                self.error = f"connection lost: {type(e).__name__}: {e}"
        finally:
            lost.set()
            with self._cond:
                self._cond.notify_all()

    def _handle(self, msg: dict) -> None:
        op = msg.get("op")
        if op == "publish":
            topic = msg.get("topic", "")
            with self._lock:
                sub = self._subs.get(topic)
                callbacks = list(sub["callbacks"]) if sub else []
            if callbacks:
                self.stats.on_received(topic)
            for cb in callbacks:
                try:
                    cb(msg.get("msg") or {})
                except Exception:  # noqa: BLE001
                    log.exception("callback for %s failed", topic)
        elif op == "service_response":
            with self._lock:
                entry = self._calls.get(msg.get("id", ""))
            if entry is not None:
                entry["response"] = msg
                entry["event"].set()
        elif op == "status":
            self.last_status = {k: msg.get(k) for k in ("level", "msg", "id")}
            if msg.get("level") in ("error", "warning"):
                log.warning("rosbridge %s: %s", msg.get("level"), msg.get("msg"))
        elif op == "fragment":
            fid, num, total = str(msg.get("id")), int(msg.get("num", 0)), int(msg.get("total", 1))
            parts = self._fragments.setdefault(fid, {})
            parts[num] = msg.get("data", "")
            if len(parts) == total:
                del self._fragments[fid]
                self._handle(json.loads("".join(parts[i] for i in range(total))))
