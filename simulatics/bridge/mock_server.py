"""A small stand-in for rosbridge_server, for tests and offline demos (no ROS needed).

It speaks the rosbridge v2 JSON protocol over websockets and behaves like a rosbridge with a ROS
graph behind it, as far as a client can tell:

* advertise / unadvertise / publish: messages are recorded per topic (with the wall time they
  arrived), checked against the ROS 2 definitions in msgs.MSG_DEFS (a wrong field earns a
  ``status`` error, as rosbridge reports a failed conversion), and forwarded to every client
  subscribed to the topic. The last message of a latched topic is kept for late subscribers.
* subscribe / unsubscribe: `inject` publishes into a topic "from ROS", like a ROS node would.
* call_service: answered by handlers registered with `add_service`.

    srv = MockRosbridgeServer(port=0).start()      # port 0: pick a free port
    ...  RosbridgeClient(srv.url) ...
    srv.wait_for("/tf", n=10)
    srv.inject("/cmd_vel", {"linear": {"x": 0.3, "y": 0, "z": 0}, "angular": {"x": 0, "y": 0, "z": 0}})
    srv.stop()
"""
from __future__ import annotations

import collections
import json
import logging
import threading
import time
from typing import Callable

from . import msgs

log = logging.getLogger("simulatics.bridge.mock")


class _Client:
    def __init__(self, conn, cid: int):
        self.conn, self.id = conn, cid
        self.adverts: dict[str, dict] = {}      # topic -> advertise op
        self.subs: dict[str, dict] = {}         # topic -> subscribe op
        self.send_lock = threading.Lock()

    def send(self, op: dict) -> None:
        with self.send_lock:
            self.conn.send(json.dumps(op, separators=(",", ":")))


class MockRosbridgeServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0, validate: bool = True, keep: int = 2000):
        self.host, self.port = host, int(port)
        self.validate = validate
        self.keep = keep
        self._lock = threading.RLock()
        self._server = None
        self._thread: threading.Thread | None = None
        self._clients: dict[int, _Client] = {}
        self._next_client = 1
        self._services: dict[str, Callable[[dict], dict]] = {}
        self._types: dict[str, str] = {}                                   # topic -> type (advertised or subscribed)
        self._latched: dict[str, dict] = {}                                # topic -> last msg of a latched topic
        self.published: dict[str, collections.deque] = {}                  # topic -> deque[(wall time, msg)]
        self.counts: collections.Counter = collections.Counter()           # topic -> messages received
        self.ops: collections.Counter = collections.Counter()              # op name -> count
        self.adverts_seen: collections.Counter = collections.Counter()     # topic -> advertise ops received
        self.errors: list[str] = []
        self.connections = 0                                               # clients that ever connected

    # ------------------------------------------------------------------ lifecycle
    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> "MockRosbridgeServer":
        from websockets.sync.server import serve
        if self.running:
            return self
        self._server = serve(self._handler, self.host, self.port, max_size=None, compression=None)
        self.port = int(self._server.socket.getsockname()[1])
        self._thread = threading.Thread(target=self._server.serve_forever, name="mock-rosbridge", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 3.0) -> None:
        """Close every connection (code 1001, like a server going down) and the listening socket."""
        if self._server is not None:
            self._server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout)
        self._server, self._thread = None, None
        with self._lock:
            self._clients.clear()

    def restart(self, down_for: float = 0.0) -> None:
        """Stop, optionally stay down for a while, and listen again on the same port."""
        self.stop()
        if down_for > 0:
            time.sleep(down_for)
        self.start()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ------------------------------------------------------------------ inspection
    @property
    def clients(self) -> int:
        with self._lock:
            return len(self._clients)

    def advertised(self) -> dict[str, dict]:
        """topic -> {"type", "latch", "qos"} over all connected clients."""
        with self._lock:
            out = {}
            for c in self._clients.values():
                for t, op in c.adverts.items():
                    out[t] = {"type": op.get("type"), "latch": bool(op.get("latch")), "qos": op.get("qos")}
            return out

    def subscribed(self) -> dict[str, dict]:
        with self._lock:
            out = {}
            for c in self._clients.values():
                for t, op in c.subs.items():
                    out[t] = {"type": op.get("type"), "throttle_rate": op.get("throttle_rate", 0)}
            return out

    def messages(self, topic: str, since: float | None = None) -> list[dict]:
        with self._lock:
            items = list(self.published.get(topic, ()))
        return [m for t, m in items if since is None or t >= since]

    def timed(self, topic: str) -> list[tuple[float, dict]]:
        with self._lock:
            return list(self.published.get(topic, ()))

    def last(self, topic: str) -> dict | None:
        with self._lock:
            d = self.published.get(topic)
            return d[-1][1] if d else None

    def rate(self, topic: str, window: float = 2.0) -> float:
        """Messages per second that arrived on `topic` during the last `window` seconds."""
        now = time.monotonic()
        with self._lock:
            n = sum(1 for t, _ in self.published.get(topic, ()) if now - t <= window)
        return n / window

    def wait_for(self, topic: str, n: int = 1, timeout: float = 5.0, since: float | None = None) -> bool:
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            if len(self.messages(topic, since)) >= n:
                return True
            time.sleep(0.02)
        return False

    def wait_clients(self, n: int = 1, timeout: float = 5.0) -> bool:
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            if self.clients >= n:
                return True
            time.sleep(0.02)
        return False

    def clear(self) -> None:
        with self._lock:
            self.published.clear()
            self.counts.clear()
            self.errors.clear()

    def summary(self, window: float = 2.0) -> list[dict]:
        """[{"topic", "type", "count", "rate_hz"}] for every topic seen."""
        adv = self.advertised()
        with self._lock:
            topics = sorted(set(self.published) | set(adv))
            counts = dict(self.counts)
        return [{"topic": t, "type": adv.get(t, {}).get("type") or self._types.get(t, "?"),
                 "count": counts.get(t, 0), "rate_hz": round(self.rate(t, window), 1)} for t in topics]

    # ------------------------------------------------------------------ acting like ROS
    def inject(self, topic: str, msg: dict, msg_type: str | None = None) -> int:
        """Publish `msg` on `topic` as a ROS node would. Returns how many subscriptions received it."""
        with self._lock:
            if msg_type:
                self._types.setdefault(topic, msg_type)
            targets = [c for c in self._clients.values() if topic in c.subs]
        n = 0
        for c in targets:
            try:
                c.send({"op": "publish", "topic": topic, "msg": msg})
                n += 1
            except Exception:  # noqa: BLE001 - client went away
                pass
        return n

    def add_service(self, name: str, handler: Callable[[dict], dict]) -> None:
        """handler(args) -> response values; raising makes the call fail (result: false)."""
        with self._lock:
            self._services[name] = handler

    # ------------------------------------------------------------------ protocol
    def _handler(self, conn) -> None:
        with self._lock:
            client = _Client(conn, self._next_client)
            self._next_client += 1
            self._clients[client.id] = client
            self.connections += 1
        try:
            for raw in conn:
                if isinstance(raw, (bytes, bytearray)):
                    continue
                try:
                    op = json.loads(raw)
                except ValueError as e:
                    self._error(client, f"invalid JSON: {e}")
                    continue
                try:
                    self._dispatch(client, op)
                except Exception as e:  # noqa: BLE001
                    self._error(client, f"{op.get('op')}: {type(e).__name__}: {e}", op.get("id"))
        except Exception:  # noqa: BLE001 - connection dropped
            pass
        finally:
            with self._lock:
                self._clients.pop(client.id, None)
                # a publisher that leaves takes its latched message with it (transient-local semantics)
                for t, op in client.adverts.items():
                    if op.get("latch") and not any(t in c.adverts for c in self._clients.values()):
                        self._latched.pop(t, None)

    def _error(self, client: _Client, text: str, op_id=None) -> None:
        with self._lock:
            self.errors.append(text)
        try:
            msg = {"op": "status", "level": "error", "msg": text}
            if op_id is not None:
                msg["id"] = op_id
            client.send(msg)
        except Exception:  # noqa: BLE001
            pass

    def _dispatch(self, client: _Client, op: dict) -> None:
        name = op.get("op")
        with self._lock:
            self.ops[name] += 1
        if name == "advertise":
            topic, typ = op["topic"], op["type"]
            with self._lock:
                known = self._types.get(topic)
                if known and msgs.normalize_type(known) != msgs.normalize_type(typ):
                    raise ValueError(f"{topic} already has type {known}, cannot advertise as {typ}")
                self._types[topic] = typ
                client.adverts[topic] = op
                self.adverts_seen[topic] += 1
        elif name == "unadvertise":
            with self._lock:
                client.adverts.pop(op["topic"], None)
        elif name == "publish":
            self._on_publish(client, op)
        elif name == "subscribe":
            topic = op["topic"]
            with self._lock:
                client.subs[topic] = op
                if op.get("type"):
                    self._types.setdefault(topic, op["type"])
                latched = self._latched.get(topic)
            if latched is not None:
                client.send({"op": "publish", "topic": topic, "msg": latched})
        elif name == "unsubscribe":
            with self._lock:
                client.subs.pop(op["topic"], None)
        elif name == "call_service":
            self._on_call(client, op)
        elif name in ("set_level", "auth"):
            pass
        else:
            raise ValueError(f"unknown op {name!r}")

    def _on_publish(self, client: _Client, op: dict) -> None:
        topic, msg = op["topic"], op.get("msg")
        with self._lock:
            adv = client.adverts.get(topic)
            if adv is None and op.get("type"):  # implicit advertise, as rosbridge allows
                adv = client.adverts[topic] = {"op": "advertise", "topic": topic, "type": op["type"],
                                               "latch": bool(op.get("latch"))}
                self._types[topic] = op["type"]
            typ = (adv or {}).get("type") or self._types.get(topic)
        if typ is None:
            raise ValueError(f"cannot publish to {topic}: topic has no type (advertise it first)")
        if self.validate and msgs.normalize_type(typ) in msgs.MSG_DEFS:
            errs = msgs.validate(typ, msg, strict=False)
            if errs:
                raise ValueError(f"{topic} ({typ}): " + "; ".join(errs[:3]))
        with self._lock:
            self.published.setdefault(topic, collections.deque(maxlen=self.keep)).append((time.monotonic(), msg))
            self.counts[topic] += 1
            if adv is not None and adv.get("latch"):
                self._latched[topic] = msg
            targets = [c for c in self._clients.values() if topic in c.subs and c is not client]
        for c in targets:
            try:
                c.send({"op": "publish", "topic": topic, "msg": msg})
            except Exception:  # noqa: BLE001
                pass

    def _on_call(self, client: _Client, op: dict) -> None:
        service = op.get("service", "")
        with self._lock:
            handler = self._services.get(service)
        resp = {"op": "service_response", "service": service}
        if op.get("id") is not None:
            resp["id"] = op["id"]
        if handler is None:
            resp.update(result=False, values=f"service {service} does not exist")
        else:
            try:
                resp.update(result=True, values=handler(op.get("args") or {}))
            except Exception as e:  # noqa: BLE001
                resp.update(result=False, values=f"{type(e).__name__}: {e}")
        client.send(resp)
