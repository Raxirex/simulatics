"""Mission plumbing shared by every pilot: cancellation, budgets, usage and the event transcript.

A *pilot* (Claude, scripted, ...) decides what to do; the `MissionContext` is what it acts
through. Keeping this in one place means every pilot produces the same transcript format
for the GUI and obeys the same stop / budget rules:

    ctx.check_limits()            raises LimitReached before an expensive step
    ctx.call_tool(name, args)     runs a Toolbox tool, emitting tool_call / tool_result / image events
    ctx.emit(kind, text=...)      appends a transcript event
    ctx.add_usage(...)            accumulates tokens and estimated cost

Cancellation is cooperative: `stop()` sets an event, and every blocking step (model request,
robot wait) polls `ctx.cancelled()` at least every ~0.25 s and raises `Cancelled`.
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

EVENT_KINDS = ("user", "text", "thinking", "tool_call", "tool_result", "image", "error", "done")
_MAX_EVENT_TEXT = 4000


class Cancelled(Exception):
    """The mission was stopped (by the user or because the wall-clock budget ran out)."""

    def __init__(self, reason: str = "stopped by user"):
        super().__init__(reason)
        self.reason = reason


class LimitReached(Exception):
    """A mission budget (turns, cost, wall time) is used up."""


@dataclass
class Outcome:
    """What a pilot's `run()` returns. `state` is the runner's terminal state."""
    state: str            # "done" | "error" | "stopped"
    summary: str = ""
    error: str | None = None
    success: bool | None = None


@dataclass
class Limits:
    max_turns: int = 30
    max_cost_usd: float = 2.0
    max_wall_seconds: float = 600.0


def jsonable(v: Any) -> Any:
    """Plain-JSON copy of tool arguments / results (numpy scalars, tuples, NaN -> None)."""
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, int):
        return int(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f if math.isfinite(f) else None


class MissionContext:
    """Everything a pilot needs while a mission runs. Thread-safe for one writer (the pilot
    thread) and many readers (`snapshot()` from the GUI thread)."""

    def __init__(self, robot, toolbox, limits: Limits | None = None, max_events: int = 2000):
        self.robot = robot
        self.toolbox = toolbox
        self.limits = limits or Limits()
        self.max_events = max_events
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self.cancel_reason = "stopped by user"
        self.t0 = time.monotonic()
        self.events: list[dict] = []
        self._seq = 0
        self.turns = 0
        self.cost_usd = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0

    # ---- cancellation ----------------------------------------------------
    def cancel(self, reason: str = "stopped by user") -> None:
        self.cancel_reason = reason
        self._cancel.set()

    def cancelled(self) -> bool:
        """True once stop() was called or the wall-clock budget is spent (polled by waits)."""
        if not self._cancel.is_set() and self.elapsed() > self.limits.max_wall_seconds:
            self.cancel(f"wall-clock limit of {self.limits.max_wall_seconds:.0f} s reached")
        return self._cancel.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled():
            raise Cancelled(self.cancel_reason)

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    # ---- budgets ---------------------------------------------------------
    def check_limits(self) -> None:
        """Call before each model request / scripted step."""
        self.raise_if_cancelled()
        if self.turns >= self.limits.max_turns:
            raise LimitReached(f"turn limit reached ({self.limits.max_turns} model calls)")
        if self.cost_usd >= self.limits.max_cost_usd:
            raise LimitReached(f"cost limit reached (${self.cost_usd:.4f} >= ${self.limits.max_cost_usd:.2f})")

    def add_usage(self, *, input_tokens: int = 0, output_tokens: int = 0, cache_read: int = 0,
                  cache_write: int = 0, cost_usd: float = 0.0) -> None:
        with self._lock:
            self.input_tokens += int(input_tokens)
            self.output_tokens += int(output_tokens)
            self.cache_read_tokens += int(cache_read)
            self.cache_write_tokens += int(cache_write)
            self.cost_usd += float(cost_usd)

    # ---- transcript ------------------------------------------------------
    def emit(self, kind: str, text: str = "", *, name: str | None = None, args: dict | None = None,
             image_b64: str | None = None, **extra) -> dict:
        if kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        if len(text) > _MAX_EVENT_TEXT:
            text = text[:_MAX_EVENT_TEXT] + " ..."
        ev = {"seq": 0, "t": round(self.elapsed(), 2), "kind": kind, "text": text, "name": name,
              "args": jsonable(args) if args is not None else None, "image_b64": image_b64}
        ev.update(jsonable(extra))
        with self._lock:
            ev["seq"] = self._seq
            self._seq += 1
            self.events.append(ev)
            if len(self.events) > self.max_events:   # keep the start (mission) and the recent tail
                del self.events[1:len(self.events) - self.max_events + 1]
        return ev

    def events_since(self, seq: int = 0) -> list[dict]:
        with self._lock:
            return [dict(e) for e in self.events if e["seq"] >= seq]

    def event_count(self) -> int:
        return self._seq

    # ---- tools -----------------------------------------------------------
    def call_tool(self, name: str, args: dict, tool_use_id: str | None = None):
        """Run one Toolbox tool with transcript events. Raises Cancelled; never other errors."""
        self.raise_if_cancelled()
        self.emit("tool_call", _brief_call(name, args), name=name, args=args, tool_use_id=tool_use_id)
        result = self.toolbox.execute(name, args)  # raises Cancelled, catches everything else
        thumb = None
        if result.images:
            from .senses import thumbnail_b64
            try:
                thumb = thumbnail_b64(result.images[0])
            except Exception:  # noqa: BLE001 - a thumbnail is cosmetic
                thumb = None
        self.emit("tool_result", result.text, name=name, is_error=result.is_error, tool_use_id=tool_use_id)
        if thumb:
            self.emit("image", f"{name}: camera view", name=name, image_b64=thumb)
        return result


def _brief_call(name: str, args: dict) -> str:
    try:
        inner = ", ".join(f"{k}={json.dumps(v)}" for k, v in (args or {}).items())
    except (TypeError, ValueError):
        inner = str(args)
    return f"{name}({inner})"
