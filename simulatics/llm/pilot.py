"""PilotRunner: the mission lifecycle the GUI talks to.

    runner = PilotRunner(robot)                       # robot implements RobotInterface
    runner.start("walk to the red box", {"pilot": "claude", "model": "claude-opus-5-5", "effort": "medium"})
    runner.brief()          -> {"run", "state", "turns", "cost_usd", "input_tokens", "output_tokens", "error",
                                "n_events", ...}            (cheap: put it in every GUI frame)
    runner.events(since)    -> {"events": [...], "next": n}  (new transcript entries only)
    runner.status(since=0)  -> brief() + {"events": [...], "event_count"}
    runner.stop()                                     # cancels within ~0.25 s (model call or robot wait)

Settings accept the GUI form's names too: "provider" for "pilot" and "images" for "send_images".

One mission runs at a time on a daemon thread. `state` is "idle" | "running" | "done" | "error" |
"stopped". Events are transcript entries {"seq", "t", "kind", "text", "name", "args", "image_b64", ...}
with kind in user | text | thinking | tool_call | tool_result | image | error | done; poll with
`status(since=last_seq + 1)` to get only new ones. The last event of a mission is always `done`
(with "state" set to the terminal state).

Stopping: `stop()` sets the mission's cancel flag; the model request is abandoned (its HTTP stream is
closed) and robot waits are sliced so they notice within one slice. If the robot exposes a
`cancelled` threading.Event (SessionRobot and FakeRobot do), it is set too so `robot.wait` returns at
once; it is cleared when the mission thread ends. The robot is told to `stop()` walking afterwards.

API keys: a key passed in `settings["api_key"]` lives only in memory (in the client object); it is
never included in `status()`, events, logs or files.
"""
from __future__ import annotations

import math
import os
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field, fields

from .claude import DEFAULT_MODEL, EFFORTS, MODELS, THINKING_DISPLAYS, ClaudeConfig, ClaudePilot
from .mission import Cancelled, Limits, LimitReached, MissionContext, Outcome
from .tools import ToolConfig, Toolbox

PILOTS = ("claude", "scripted")
_TRUE = frozenset({"true", "1", "yes", "on"})
_FALSE = frozenset({"false", "0", "no", "off", ""})


def _clamp(v, lo: float, hi: float, key: str) -> float:
    """float(v) clamped to [lo, hi]; NaN and non-numbers raise ValueError (shown to the GUI as a 400)."""
    if isinstance(v, bool):
        raise ValueError(f"{key} must be a number, got {v!r}")
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number, got {v!r}") from None
    if math.isnan(f):
        raise ValueError(f"{key} must be a finite number, got NaN")
    return min(max(f, float(lo)), float(hi))


def _flag(v, key: str) -> bool:
    """Strict boolean: `bool("false")` is True, so strings from form posts are parsed explicitly."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in _TRUE | _FALSE:
        return v.strip().lower() in _TRUE
    raise ValueError(f"{key} must be true or false, got {v!r}")


@dataclass
class PilotSettings:
    pilot: str = "claude"                  # "claude" | "scripted"
    model: str = DEFAULT_MODEL
    effort: str = "medium"
    thinking_display: str = "summarized"
    fallbacks: bool = True                 # server-side refusal fallback ("default" routing)
    stream: bool = True
    max_tokens: int = 16000
    max_turns: int = 30
    max_cost_usd: float = 2.0
    max_wall_seconds: float = 600.0
    camera: str = "head_cam"
    image_width: int = 480
    image_height: int = 360
    send_images: bool = True               # camera images to the model (initial view and `look`)
    initial_image: bool = True
    stop_robot_on_cancel: bool = True
    api_key: str | None = field(default=None, repr=False)

    ALIASES = {"provider": "pilot", "images": "send_images"}   # names used by the GUI form

    @classmethod
    def from_dict(cls, d: dict | None) -> "PilotSettings":
        """Validate GUI settings; unknown keys are ignored, out-of-range numbers clamped, bad enums rejected."""
        d = {cls.ALIASES.get(k, k): v for k, v in dict(d or {}).items()}
        known = {f.name for f in fields(cls)}
        s = cls(**{k: v for k, v in d.items() if k in known and v is not None})
        if s.pilot not in PILOTS:
            raise ValueError(f"pilot must be one of {list(PILOTS)}")
        if s.model not in MODELS:
            raise ValueError(f"model must be one of {list(MODELS)}")
        if s.effort not in EFFORTS:
            raise ValueError(f"effort must be one of {list(EFFORTS)}")
        if s.thinking_display not in THINKING_DISPLAYS:
            raise ValueError(f"thinking_display must be one of {list(THINKING_DISPLAYS)}")
        # NaN would silently disable a budget (every `cost >= NaN` comparison is False), so it is
        # rejected; +-inf is clamped like any other out-of-range value.
        s.max_tokens = int(_clamp(s.max_tokens, 1024, 128000, "max_tokens"))
        s.max_turns = int(_clamp(s.max_turns, 1, 500, "max_turns"))
        s.max_cost_usd = _clamp(s.max_cost_usd, 0.0, 1000.0, "max_cost_usd")
        s.max_wall_seconds = _clamp(s.max_wall_seconds, 1.0, 24 * 3600.0, "max_wall_seconds")
        s.image_width = int(_clamp(s.image_width, 64, 1568, "image_width"))
        s.image_height = int(_clamp(s.image_height, 64, 1568, "image_height"))
        for b in ("fallbacks", "stream", "send_images", "initial_image", "stop_robot_on_cancel"):
            setattr(s, b, _flag(getattr(s, b), b))
        if not s.send_images:
            s.initial_image = False
        if s.api_key is not None:
            s.api_key = str(s.api_key).strip() or None
        return s

    def public_dict(self) -> dict:
        """Settings safe to show/store: everything except the key itself."""
        d = asdict(self)
        d.pop("api_key", None)
        d["api_key_set"] = bool(self.api_key)
        return d

    def claude_config(self) -> ClaudeConfig:
        return ClaudeConfig(model=self.model, effort=self.effort, thinking_display=self.thinking_display,
                            fallbacks=self.fallbacks, stream=self.stream, max_tokens=self.max_tokens,
                            initial_image=self.initial_image, camera=self.camera, image_width=self.image_width,
                            image_height=self.image_height, api_key=self.api_key)


def options() -> dict:
    """Choices for GUI dropdowns."""
    return {"pilots": list(PILOTS), "models": list(MODELS), "default_model": DEFAULT_MODEL,
            "efforts": list(EFFORTS), "default_effort": "medium", "thinking_displays": list(THINKING_DISPLAYS),
            "env_credentials": bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
            "defaults": PilotSettings().public_dict()}


class PilotRunner:
    """Runs one mission at a time in a background thread (see module docstring).

    `client` / `client_factory(ClaudeConfig)` override how the Anthropic client is made (tests pass a fake);
    `pilot_factory(settings)` overrides the pilot entirely."""

    def __init__(self, robot, *, client=None, client_factory=None, pilot_factory=None,
                 tool_config: ToolConfig | None = None, max_events: int = 2000):
        self.robot = robot
        self._client = client
        self._client_factory = client_factory
        self._pilot_factory = pilot_factory
        self.tool_config = tool_config
        self.max_events = max_events
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._ctx: MissionContext | None = None
        self.state = "idle"
        self.mission = ""
        self.summary = ""
        self.error: str | None = None
        self.success: bool | None = None
        self.settings = PilotSettings()
        self._public_settings = self.settings.public_dict()
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.run_id = 0

    # ---- lifecycle -----------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, mission: str, settings: dict | PilotSettings | None = None) -> None:
        mission = str(mission or "").strip()
        if not mission:
            raise ValueError("mission text is empty")
        # A PilotSettings object is copied (and validated like a dict) so the caller's object, including
        # its api_key, is never modified by the runner.
        s = PilotSettings.from_dict(asdict(settings) if isinstance(settings, PilotSettings) else settings)
        with self._lock:
            if self.running:
                raise RuntimeError("a mission is already running; stop it first")
            cfg = self.tool_config or ToolConfig(default_camera=s.camera, image_width=s.image_width,
                                                 image_height=s.image_height, send_images=s.send_images)
            ctx = MissionContext(self.robot, None, Limits(s.max_turns, s.max_cost_usd, s.max_wall_seconds),
                                 max_events=self.max_events)
            ctx.toolbox = Toolbox(self.robot, cfg, cancelled=ctx.cancelled)
            pilot = self._make_pilot(s)
            self._public_settings = s.public_dict()
            s.api_key = None            # the client/pilot holds it now; do not keep a second copy
            self.settings, self._ctx, self.mission = s, ctx, mission
            self.state, self.summary, self.error, self.success = "running", "", None, None
            self.started_at, self.ended_at = time.time(), None
            self.run_id += 1
            ev = getattr(self.robot, "cancelled", None)
            if isinstance(ev, threading.Event):
                ev.clear()
            self._thread = threading.Thread(target=self._run, args=(pilot, mission, ctx), name="pilot", daemon=True)
            self._thread.start()

    def _make_pilot(self, s: PilotSettings):
        if self._pilot_factory is not None:
            return self._pilot_factory(s)
        if s.pilot == "scripted":
            from .scripted import ScriptedPilot
            return ScriptedPilot()
        factory = self._client_factory
        return ClaudePilot(s.claude_config(), client=self._client, client_factory=factory)

    def stop(self, timeout: float = 2.0) -> bool:
        """Cancel the running mission; returns True once its thread has ended (within `timeout`)."""
        ctx, th = self._ctx, self._thread
        if ctx is None or th is None or not th.is_alive():
            return True
        ctx.cancel("stopped by user")
        ev = getattr(self.robot, "cancelled", None)
        if isinstance(ev, threading.Event):
            ev.set()
        th.join(timeout)
        if not th.is_alive() and isinstance(ev, threading.Event):
            ev.clear()          # the thread may have ended before we set it; never leave it set
        return not th.is_alive()

    def join(self, timeout: float | None = None) -> bool:
        th = self._thread
        if th is not None:
            th.join(timeout)
        return not self.running

    def brief(self) -> dict:
        """Status without events: cheap enough to put in every GUI frame. `run` changes per mission,
        `n_events` tells the GUI whether to fetch `events(since)`."""
        ctx = self._ctx
        st = {"run": self.run_id, "state": self.state, "pilot": self.settings.pilot, "model": self.settings.model,
              "mission": self.mission, "summary": self.summary, "error": self.error, "success": self.success,
              "turns": 0, "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
              "cache_write_tokens": 0, "elapsed_s": 0.0, "n_events": 0, "settings": dict(self._public_settings)}
        if ctx is not None:
            end = self.ended_at or time.time()
            st.update(turns=ctx.turns, cost_usd=round(ctx.cost_usd, 6), input_tokens=ctx.input_tokens,
                      output_tokens=ctx.output_tokens, cache_read_tokens=ctx.cache_read_tokens,
                      cache_write_tokens=ctx.cache_write_tokens,
                      elapsed_s=round(end - (self.started_at or end), 2), n_events=ctx.event_count())
        return st

    def events(self, since: int = 0) -> dict:
        """{"events": [events with seq >= since], "next": seq to ask for next time}."""
        ctx = self._ctx
        if ctx is None:
            return {"events": [], "next": 0}
        return {"events": ctx.events_since(int(since)), "next": ctx.event_count()}

    def status(self, since: int = 0) -> dict:
        """brief() plus the transcript events with seq >= since (and `event_count`)."""
        st = self.brief()
        st["events"] = self.events(since)["events"]
        st["event_count"] = st["n_events"]
        return st

    # ---- worker thread ---------------------------------------------------------
    def _run(self, pilot, mission: str, ctx: MissionContext) -> None:
        outcome: Outcome
        detail = ""
        try:
            outcome = pilot.run(mission, ctx)
        except Cancelled as c:
            outcome = Outcome("stopped", summary=ctx.cancel_reason if ctx.cancelled() else c.reason)
        except LimitReached as e:
            outcome = Outcome("stopped", summary=str(e), error=str(e))
        except Exception as e:  # noqa: BLE001 - shown in the GUI instead of killing the thread silently
            outcome = Outcome("error", error=f"{type(e).__name__}: {e}")
            detail = "\n" + traceback.format_exc(limit=6)
        if outcome.state == "stopped" and self.settings.stop_robot_on_cancel:
            try:
                self.robot.stop()
            except Exception:  # noqa: BLE001
                pass
        if outcome.error and outcome.state == "error":
            ctx.emit("error", outcome.error + detail)
        done_text = {"done": outcome.summary or "mission finished",
                     "stopped": f"mission stopped: {outcome.summary}",
                     "error": f"mission failed: {outcome.error}"}[outcome.state]
        ctx.emit("done", done_text, state=outcome.state, success=outcome.success)
        ev = getattr(self.robot, "cancelled", None)
        if isinstance(ev, threading.Event):
            ev.clear()
        self.summary = outcome.summary if outcome.state != "error" else (outcome.error or "")
        self.error, self.success = outcome.error, outcome.success
        self.ended_at = time.time()
        self.state = outcome.state
