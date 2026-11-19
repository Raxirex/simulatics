"""The publish/subscribe contract both transports implement (rosbridge websocket client, native rclpy).

Messages are plain dicts in ROS 2 JSON form (see msgs.py); types are 'pkg/msg/Name' strings.
Callbacks of subscriptions run on the transport's own thread and receive the message dict.
"""
from __future__ import annotations

import collections
import threading
import time
from typing import Callable, Protocol, runtime_checkable

Callback = Callable[[dict], None]


@runtime_checkable
class Transport(Protocol):
    name: str                   # "rosbridge" | "rclpy" | ...
    supports_nonfinite: bool    # can float fields carry inf / nan (JSON cannot)

    def start(self) -> None:
        """Connect (in the background; returns immediately)."""

    def stop(self, timeout: float = 3.0) -> None:
        """Disconnect and join every thread the transport started."""

    @property
    def connected(self) -> bool: ...

    def advertise(self, topic: str, msg_type: str, latch: bool = False, queue_size: int = 10,
                  max_rate_hz: float | None = None) -> None:
        """Declare a publisher. latch = transient-local durability (late joiners get the last message).
        max_rate_hz throttles `publish` for this topic (extra messages are dropped)."""

    def unadvertise(self, topic: str) -> None: ...

    def publish(self, topic: str, msg: dict) -> bool:
        """Queue one message; False when it was dropped (throttled or not connected)."""

    def subscribe(self, topic: str, msg_type: str, callback: Callback, throttle_ms: int = 0,
                  queue_length: int = 1) -> None: ...

    def unsubscribe(self, topic: str) -> None: ...

    def call_service(self, service: str, args: dict | None = None, timeout: float = 5.0) -> dict:
        """Call a ROS service and return its response values (optional: may raise NotImplementedError)."""

    def status(self) -> dict: ...


class TopicStats:
    """Thread-safe per-topic counters and a sliding-window rate (wall clock)."""

    def __init__(self, window: int = 50):
        self._lock = threading.Lock()
        self._window = window
        self.sent: collections.Counter = collections.Counter()
        self.dropped: collections.Counter = collections.Counter()
        self.received: collections.Counter = collections.Counter()
        self._times: dict[str, collections.deque] = {}

    def on_sent(self, topic: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            self.sent[topic] += 1
            self._times.setdefault(topic, collections.deque(maxlen=self._window)).append(now)

    def on_dropped(self, topic: str) -> None:
        with self._lock:
            self.dropped[topic] += 1

    def on_received(self, topic: str) -> None:
        with self._lock:
            self.received[topic] += 1

    def rate(self, topic: str, now: float | None = None, horizon: float = 5.0) -> float:
        """Messages per second over the recent window (0 when nothing was sent for `horizon` s)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            ts = self._times.get(topic)
            if not ts or now - ts[-1] > horizon or len(ts) < 2:
                return 0.0
            span = ts[-1] - ts[0]
            return (len(ts) - 1) / span if span > 0 else 0.0

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self._lock:
            topics = set(self.sent) | set(self.dropped) | set(self.received)
        return {t: {"sent": self.sent[t], "dropped": self.dropped[t], "received": self.received[t],
                    "rate_hz": round(self.rate(t, now), 2)} for t in sorted(topics)}


class RateLimiter:
    """Per-topic minimum interval between messages (wall clock)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._period: dict[str, float] = {}
        self._last: dict[str, float] = {}

    def set_rate(self, topic: str, max_rate_hz: float | None) -> None:
        with self._lock:
            if max_rate_hz is None or max_rate_hz <= 0:
                self._period.pop(topic, None)
            else:
                self._period[topic] = 1.0 / float(max_rate_hz)

    def allow(self, topic: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            period = self._period.get(topic)
            if period is None:
                return True
            last = self._last.get(topic)
            # 5 % slack, so a producer ticking at exactly the limit is not dropped by timer jitter
            if last is not None and now - last < 0.95 * period:
                return False
            self._last[topic] = now
            return True
