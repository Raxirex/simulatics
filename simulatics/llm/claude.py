"""ClaudePilot: Claude drives the robot through the Anthropic Messages API (official `anthropic` SDK).

Method: a manual tool-use agent loop (Anthropic tool use; Inner Monologue-style text feedback,
Huang et al. 2022; skill-API piloting as in
Vemprala et al. 2023). The manual loop (instead of the SDK's beta tool runner) is used because the GUI
needs prompt cancellation during a model request, per-turn budget checks, refusal handling and a
transcript event per block.

Request shape (rules from the Claude API skill docs, verified against anthropic 2.10.0):

- `client.beta.messages.stream(...)`: the beta namespace carries the server-side refusal fallback
  (`betas=["server-side-fallback-2026-07-01"]`, `fallbacks="default"`), on by default and switchable.
  Claude Haiku 5.5 has no server-side fallback, so it is never sent for that model.
- Adaptive thinking only (`thinking={"type": "adaptive", "display": ...}`): `budget_tokens` and
  `{"type": "disabled"}` are 400s on these models. Depth is `output_config={"effort": ...}` (default
  "medium"). `display="summarized"` makes the reasoning readable in the GUI transcript.
- `tool_choice={"type": "auto"}`: forced `any`/`tool` returns 400 on Opus 5.5 / Sonnet 5.5 / Fable 5.2,
  so tools are `strict: true` and the system prompt asks for `finish`. If Claude ends a turn without a
  tool call it is nudged once, then its last text is taken as the summary.
- Prompt caching: an explicit breakpoint on the frozen system prompt (tools render before it, so tools +
  system are cached together) plus top-level automatic caching for the growing history.
- History is append-only: the full `response.content` (thinking blocks unchanged, including signatures)
  is appended every turn; all tool_result blocks of a turn go back in ONE user message, failed tools
  with `is_error: true`, `look` results as image blocks. After a server-side fallback boundary, blocks
  that must not be echoed (thinking / tool_use before the last `fallback` block) are dropped as the
  docs require, and only tool calls after the boundary are executed.
- `stop_reason` is checked before content is used: `refusal` stops the mission with `stop_details`;
  `max_tokens` is retried once with a doubled limit; `pause_turn` continues.
- Typed SDK exceptions are mapped to readable errors. Usage is accumulated per request, using
  `usage.iterations` (per-attempt, per-model) when present so fallback attempts are priced correctly.

`eager_input_streaming` is deliberately left off: the tool inputs are a few scalars (nothing to gain),
and buffered streaming keeps the server-side strict-schema validation of tool inputs.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from .mission import Cancelled, MissionContext, Outcome
from .senses import image_block, observe, thumbnail_b64

DEFAULT_MODEL = "claude-opus-5-5"
MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-2")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
THINKING_DISPLAYS = ("summarized", "omitted", "updates")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
THINKING_UPDATES_BETA = "thinking-display-updates-2026-08-18"
# Models with server-side refusal fallback. Claude Haiku 5.5 has none (a fallback list is a 400 there).
SERVER_FALLBACK_MODELS = frozenset({"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-2"})
UPDATES_DISPLAY_MODELS = frozenset({"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-2"})


@dataclass(frozen=True)
class Price:
    """USD per million tokens: input, output, cache read, 5-minute cache write (1.25x input)."""
    input: float
    output: float
    cache_read: float
    cache_write: float


# From the Claude API skill's model table (cached 2026-11-03). Haiku 5.5's cache-read price is not
# listed there and is assumed to be 0.1x input; Haiku 5.5 prompts over 100K tokens cost more ($0.50/$2.50)
# and that tier is not modelled. Opus 5 / Opus 4.8 appear only as server-side fallback targets.
PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": Price(2.00, 10.00, 0.10, 2.50),
    "claude-haiku-5-5": Price(0.10, 0.50, 0.01, 0.125),
    "claude-fable-5-2": Price(8.00, 40.00, 0.20, 10.00),
    "claude-opus-5": Price(5.00, 25.00, 0.50, 6.25),
    "claude-opus-4-8": Price(5.00, 25.00, 0.50, 6.25),
}

SYSTEM_PROMPT = """\
You are the pilot of a simulated humanoid robot in Simulatics, a MuJoCo robot simulation workbench. You \
perceive the world only through the observations and tools below, and you act only through the tools. A \
human operator gives you a mission and watches your progress in a transcript.

World and units
- Metres, degrees and simulation seconds. The world frame has z up; at spawn the robot faces +x and +y is \
to its left. World yaw is measured counter-clockwise from +x.
- Bearings in observations are relative to the robot: 0 = straight ahead, positive = to the left, negative \
= to the right.

How the robot works
- Walking is handled by a path planner and a walking controller; you choose goals (walk_to, walk_to_object, \
walk_relative). The robot walks at about 0.25-0.3 m/s, turns on the spot when needed, and stops within a \
few centimetres of the goal.
- Motion tools block until the motion ends, fails or times out, and then report the outcome with the fresh \
pose, navigation and arm status. Read the outcome before deciding the next step; an outcome marked as an \
error means the goal was not achieved.
- Each hand reaches about 0.6 m from its shoulder, on its own side of the body. To pick something up: \
walk_to_object (it stops close enough for the hands, or does nothing if already there), look with work_cam \
(it shows the hands and the table), reach_object with the hand on the object's side (left if its bearing \
is positive, otherwise right), grasp, and check that the result says the object is held. To \
lift it, reach to a point about 0.2 m higher. release opens the hand.
- Arm gestures (wave, raise, home) are for communicating with people.

Perception
- The object list comes from ground-truth segmentation: names, positions and the visible flag are \
reliable. Use the exact names it gives.
- look returns a camera image. Use it to check what things look like and to verify results. If an \
object is not in view, turn with walk_relative(0, 0, 60) or (0, 0, -60) and look again.
- scan_surroundings gives the nearest lidar return per direction; check it when the way ahead is unclear.

Working rules
- Issue motion tools one at a time, after sensing. Several sensing tools may be called together.
- If a tool returns an error, read it and change the approach; do not repeat the same failing call more \
than twice.
- Keep text short: at most one sentence before a tool call, so the operator can follow along.
- When the mission is complete, or cannot be completed, call finish with success true or false and a \
one-to-three sentence summary of what was done and the final state (where the robot is, what it holds).
- If a mission asks for something this robot cannot do safely or at all, explain that in finish instead \
of attempting it.
"""

NUDGE = ("You ended your turn without calling a tool. If the mission is complete or cannot be completed, "
         "call finish with a summary; otherwise continue with the tools.")


@dataclass
class ClaudeConfig:
    model: str = DEFAULT_MODEL
    effort: str = "medium"
    thinking_display: str = "summarized"
    fallbacks: bool = True
    stream: bool = True
    max_tokens: int = 16000
    max_tokens_ceiling: int = 64000
    cache: bool = True
    initial_image: bool = True
    camera: str = "head_cam"
    image_width: int = 480
    image_height: int = 360
    request_timeout_s: float = 300.0
    max_retries: int = 2
    api_key: str | None = field(default=None, repr=False)   # memory only; never logged or saved

    def validate(self) -> "ClaudeConfig":
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {list(MODELS)}")
        if self.effort not in EFFORTS:
            raise ValueError(f"effort must be one of {list(EFFORTS)}")
        if self.thinking_display not in THINKING_DISPLAYS:
            raise ValueError(f"thinking_display must be one of {list(THINKING_DISPLAYS)}")
        if not 1024 <= int(self.max_tokens) <= 128000:
            raise ValueError("max_tokens must be in [1024, 128000]")
        return self


# ---------------------------------------------------------------------- request construction
def build_request(cfg: ClaudeConfig, tools: list[dict], messages: list, max_tokens: int | None = None) -> dict:
    """Keyword arguments for `client.beta.messages.stream/create` (pure function, tested directly)."""
    display = cfg.thinking_display
    if display == "updates" and cfg.model not in UPDATES_DISPLAY_MODELS:
        display = "summarized"
    system: list[dict] = [{"type": "text", "text": SYSTEM_PROMPT}]
    params: dict[str, Any] = {
        "model": cfg.model,
        "max_tokens": int(max_tokens or cfg.max_tokens),
        "system": system,
        "tools": tools,
        "tool_choice": {"type": "auto"},
        "messages": list(messages),
        "thinking": {"type": "adaptive", "display": display},
        "output_config": {"effort": cfg.effort},
    }
    if cfg.cache:
        system[-1]["cache_control"] = {"type": "ephemeral"}       # tools + system prefix
        params["cache_control"] = {"type": "ephemeral"}           # growing conversation tail
    betas: list[str] = []
    if cfg.fallbacks and cfg.model in SERVER_FALLBACK_MODELS:
        betas.append(FALLBACK_BETA)
        params["fallbacks"] = "default"
    if display == "updates":
        betas.append(THINKING_UPDATES_BETA)
    if betas:
        params["betas"] = betas
    return params


def _get(block, key, default=None):
    return block.get(key, default) if isinstance(block, dict) else getattr(block, key, default)


def _fallback_boundary(content) -> int:
    idx = -1
    for i, b in enumerate(content):
        if _get(b, "type") == "fallback":
            idx = i
    return idx


def echo_content(content) -> list:
    """Assistant content to append to the history.

    Unchanged unless a server-side fallback happened mid-output: then, before the LAST `fallback` block,
    only text blocks are echoed (thinking / redacted_thinking / tool_use / unpaired server-tool blocks of
    the declined model are omitted) and the fallback block stays exactly where it was."""
    idx = _fallback_boundary(content)
    if idx < 0:
        return list(content)
    return [b for i, b in enumerate(content) if i >= idx or _get(b, "type") == "text"]


def executable_tool_uses(content) -> list:
    """tool_use blocks the client must run: those after the last fallback boundary."""
    idx = _fallback_boundary(content)
    return [b for i, b in enumerate(content) if i > idx and _get(b, "type") == "tool_use"]


# ---------------------------------------------------------------------- usage and cost
def price_for(model: str | None) -> Price | None:
    if not model:
        return None
    for key in sorted(PRICES, key=len, reverse=True):
        if model == key or model.startswith(key):
            return PRICES[key]
    return None


def usage_cost(usage, requested_model: str) -> dict:
    """Tokens and estimated USD for one response. Uses `usage.iterations` (one entry per sampling attempt,
    each with its own model) when present - the per-attempt source of truth for fallback billing."""
    entries = _get(usage, "iterations") or None
    rows = []
    if entries:
        for it in entries:
            rows.append((_get(it, "model") or requested_model, it))
    else:
        rows.append((requested_model, usage))
    tot = {"input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_write": 0, "cost_usd": 0.0,
           "unpriced_models": []}
    for model, u in rows:
        i = int(_get(u, "input_tokens") or 0)
        o = int(_get(u, "output_tokens") or 0)
        cr = int(_get(u, "cache_read_input_tokens") or 0)
        cw = int(_get(u, "cache_creation_input_tokens") or 0)
        p = price_for(str(model)) or price_for(requested_model)
        if price_for(str(model)) is None:
            tot["unpriced_models"].append(str(model))
        tot["input_tokens"] += i
        tot["output_tokens"] += o
        tot["cache_read"] += cr
        tot["cache_write"] += cw
        if p is not None:
            tot["cost_usd"] += (i * p.input + o * p.output + cr * p.cache_read + cw * p.cache_write) / 1e6
    return tot


# ---------------------------------------------------------------------- the pilot
def default_client_factory(cfg: ClaudeConfig):
    """`anthropic.Anthropic()` resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login`
    profiles; a key typed into the GUI is passed explicitly and only kept in memory."""
    import anthropic
    kwargs: dict[str, Any] = {"max_retries": int(cfg.max_retries), "timeout": float(cfg.request_timeout_s)}
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    return anthropic.Anthropic(**kwargs)


class ClaudePilot:
    """Runs one mission with Claude. `client` (or `client_factory(cfg)`) is injectable for tests."""

    name = "claude"

    def __init__(self, config: ClaudeConfig | None = None, client=None,
                 client_factory: Callable[[ClaudeConfig], Any] | None = None):
        self.config = (config or ClaudeConfig()).validate()
        self._client = client
        self._factory = client_factory or default_client_factory
        self.history: list[dict] = []
        self.requests = 0

    # ---- model call with prompt cancellation -------------------------------
    def _call(self, client, params: dict, ctx: MissionContext):
        """One Messages request on a worker thread; the pilot thread polls for stop() every 50 ms and
        closes the HTTP stream if cancelled, so a stop never waits for a slow generation."""
        box: dict[str, Any] = {}
        done = threading.Event()
        holder: dict[str, Any] = {}

        def work():
            try:
                if self.config.stream:
                    with client.beta.messages.stream(**params) as stream:
                        holder["stream"] = stream
                        for _event in stream:
                            if ctx.cancelled():
                                box["cancelled"] = True
                                return
                        box["message"] = stream.get_final_message()
                else:
                    box["message"] = client.beta.messages.create(**params)
            except BaseException as e:  # noqa: BLE001 - re-raised on the pilot thread
                box["error"] = e
            finally:
                done.set()

        threading.Thread(target=work, name="claude-request", daemon=True).start()
        while not done.wait(0.05):
            if ctx.cancelled():
                s = holder.get("stream")
                if s is not None:
                    try:
                        s.close()
                    except Exception:  # noqa: BLE001 - best effort, the thread is abandoned anyway
                        pass
                raise Cancelled(ctx.cancel_reason)
        if ctx.cancelled() or box.get("cancelled"):
            raise Cancelled(ctx.cancel_reason)
        if "error" in box:
            raise box["error"]
        return box["message"]

    # ---- helpers -------------------------------------------------------------
    def _first_user_content(self, mission: str, ctx: MissionContext) -> list[dict]:
        cfg = self.config
        obs = observe(ctx.robot, camera=cfg.camera if cfg.initial_image else None,
                      width=cfg.image_width, height=cfg.image_height)
        content: list[dict] = []
        if obs.image is not None:
            content.append(image_block(obs.image))
        content.append({"type": "text", "text": "Observation at the start of the mission:\n" + obs.text()})
        content.append({"type": "text", "text": f"Mission: {mission}"})
        ctx.emit("user", mission, has_image=obs.image is not None)
        if obs.image is not None:
            try:
                ctx.emit("image", f"{obs.camera}: initial view", name=obs.camera, image_b64=thumbnail_b64(obs.image))
            except Exception:  # noqa: BLE001 - cosmetic
                pass
        return content

    def _emit_blocks(self, content, ctx: MissionContext) -> None:
        for b in content:
            t = _get(b, "type")
            if t == "thinking" and (_get(b, "thinking") or "").strip():
                ctx.emit("thinking", _get(b, "thinking"))
            elif t == "text" and (_get(b, "text") or "").strip():
                ctx.emit("text", _get(b, "text"))
            elif t == "fallback":
                frm, to = _get(_get(b, "from_") or _get(b, "from"), "model"), _get(_get(b, "to"), "model")
                cat = _get(_get(b, "trigger"), "category")
                ctx.emit("text", f"[refusal fallback] {frm} declined (category {cat or 'none'}); {to} continued",
                         name="fallback")

    def _account(self, msg, ctx: MissionContext) -> None:
        u = _get(msg, "usage")
        if u is None:
            return
        c = usage_cost(u, self.config.model)
        ctx.add_usage(input_tokens=c["input_tokens"], output_tokens=c["output_tokens"], cache_read=c["cache_read"],
                      cache_write=c["cache_write"], cost_usd=c["cost_usd"])

    # ---- main loop -------------------------------------------------------------
    def run(self, mission: str, ctx: MissionContext) -> Outcome:
        import anthropic

        cfg = self.config
        try:
            client = self._client if self._client is not None else self._factory(cfg)
        except Exception as e:  # noqa: BLE001
            return Outcome("error", error=f"could not create the Anthropic client: {type(e).__name__}: {e}")
        tools = ctx.toolbox.definitions()
        self.history = [{"role": "user", "content": self._first_user_content(mission, ctx)}]
        nudged = False
        max_tokens = cfg.max_tokens
        while True:
            ctx.check_limits()
            params = build_request(cfg, tools, self.history, max_tokens)
            try:
                msg = self._call(client, params, ctx)
            except Cancelled:
                raise
            except anthropic.AuthenticationError:
                return Outcome("error", error="authentication failed (401): the API key is missing or invalid")
            except anthropic.PermissionDeniedError as e:
                return Outcome("error", error=f"permission denied (403): {e.message}")
            except anthropic.NotFoundError:
                return Outcome("error", error=f"model '{cfg.model}' was not found or is not available to this "
                                              "account (404)")
            except anthropic.RateLimitError as e:
                ra = e.response.headers.get("retry-after") if e.response is not None else None
                return Outcome("error", error="rate limited (429) after the SDK's retries"
                                              + (f"; retry after {ra} s" if ra else ""))
            except anthropic.BadRequestError as e:
                return Outcome("error", error=f"bad request (400): {e.message}")
            except anthropic.APIStatusError as e:
                return Outcome("error", error=f"API error {e.status_code}: {e.message}")
            except anthropic.APITimeoutError:
                return Outcome("error", error=f"request timed out after {cfg.request_timeout_s:.0f} s")
            except anthropic.APIConnectionError:
                return Outcome("error", error="network error: could not reach the Anthropic API")
            except TypeError as e:
                if "authentication" in str(e).lower():   # the SDK raises TypeError when no credential resolves
                    return Outcome("error", error="no Anthropic credentials: set ANTHROPIC_API_KEY, run "
                                                  "`ant auth login`, or enter an API key in the GUI")
                raise
            self.requests += 1
            ctx.turns += 1
            self._account(msg, ctx)
            stop = _get(msg, "stop_reason")
            content = list(_get(msg, "content") or [])

            if stop == "refusal":   # check before using content: it may be empty or a partial
                det = _get(msg, "stop_details")
                cat = _get(det, "category") if det is not None else None
                expl = _get(det, "explanation") if det is not None else None
                text = f"Claude declined this request (stop_reason=refusal, category={cat or 'none'})" + \
                       (f": {expl}" if expl else "")
                if not cfg.fallbacks and cfg.model in SERVER_FALLBACK_MODELS:
                    text += ". Server-side fallbacks are off; enabling them may let another model continue."
                return Outcome("error", error=text)

            if stop == "max_tokens":   # truncated turn: discard it (a cut-off tool_use must not run)
                # without streaming, large max_tokens risk HTTP timeouts, so only the streaming path grows it
                ceiling = cfg.max_tokens_ceiling if cfg.stream else cfg.max_tokens
                if max_tokens < ceiling:
                    max_tokens = min(2 * max_tokens, ceiling)
                    ctx.emit("error", f"response hit max_tokens; retrying the turn with max_tokens={max_tokens}")
                    continue
                return Outcome("error", error=f"response truncated at max_tokens={max_tokens}")

            self._emit_blocks(content, ctx)
            echoed = echo_content(content)
            if echoed:   # an empty reply (possible right after tool results) must not be echoed: the API
                # rejects an empty non-final assistant message; the nudge below then follows the results
                self.history.append({"role": "assistant", "content": echoed})
            if stop == "pause_turn":
                continue

            uses = executable_tool_uses(content)
            if uses:
                results: list[dict] = []
                for tu in uses:
                    res = ctx.call_tool(_get(tu, "name"), dict(_get(tu, "input") or {}), tool_use_id=_get(tu, "id"))
                    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": _get(tu, "id"),
                                             "content": res.content_blocks()}
                    if res.is_error:
                        block["is_error"] = True
                    results.append(block)
                    if res.finished:
                        return Outcome("done", summary=res.summary or "", success=res.success)
                self.history.append({"role": "user", "content": results})   # all results, one message
                continue

            final_text = "\n".join(_get(b, "text") for b in content if _get(b, "type") == "text").strip()
            if stop == "end_turn" and not nudged:
                nudged = True
                self.history.append({"role": "user", "content": [{"type": "text", "text": NUDGE}]})
                ctx.emit("user", NUDGE, auto=True)
                continue
            if stop not in ("end_turn", "stop_sequence"):
                return Outcome("error", error=f"unexpected stop_reason {stop!r}")
            return Outcome("done", summary=final_text or "Claude ended the mission without a summary.")
