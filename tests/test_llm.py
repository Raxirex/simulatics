"""LLM pilot tests. No network: Claude is replaced by a scripted fake client (and, in one test, the real
SDK talking to an in-process mock HTTP transport). The live test runs only with SIMULATICS_LIVE_CLAUDE=1."""
import base64
import itertools
import json
import math
import os
import time
from types import SimpleNamespace

import anthropic
import httpx2
import pytest
from anthropic.types.beta import BetaMessage

from simulatics.llm import (FakeObject, FakeRobot, PilotRunner, PilotSettings, RobotInterface, ScriptedPilot,
                            ToolConfig, Toolbox, find_object, observe, tool_definitions)
from simulatics.llm.claude import (FALLBACK_BETA, NUDGE, PRICES, THINKING_UPDATES_BETA, ClaudeConfig,
                                   build_request, usage_cost)
from simulatics.llm.tools import TOOL_NAMES

EXPECTED_TOOLS = {"look", "scan_surroundings", "list_objects", "walk_to", "walk_relative", "walk_to_object", "stop",
                  "wait", "reach", "reach_object", "grasp", "release", "point_at", "gesture", "get_status", "finish"}

# ---------------------------------------------------------------------------- fake Anthropic client
_ids = itertools.count()


def thinking(text, sig):
    return {"type": "thinking", "thinking": text, "signature": sig}


def text(t):
    return {"type": "text", "text": t}


def tool_use(tid, name, args):
    return {"type": "tool_use", "id": tid, "name": name, "input": args}


def reply(*blocks, stop="tool_use", usage=None, model="claude-opus-5-5", stop_details=None):
    d = {"id": f"msg_{next(_ids)}", "type": "message", "role": "assistant", "model": model, "content": list(blocks),
         "stop_reason": stop, "stop_sequence": None,
         "usage": usage or {"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 3000,
                            "cache_creation_input_tokens": 500}}
    if stop_details is not None:
        d["stop_details"] = stop_details
    return BetaMessage.model_validate(d)


class Slow:
    """Script item: a response whose stream trickles `n` events, `delay` seconds apart."""

    def __init__(self, message, n=200, delay=0.05):
        self.message, self.n, self.delay = message, n, delay


class FakeStream:
    def __init__(self, message, n=3, delay=0.0):
        self.message, self.n, self.delay, self.closed = message, n, delay, False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __iter__(self):
        for _ in range(self.n):
            if self.closed:
                raise RuntimeError("stream closed")
            if self.delay:
                time.sleep(self.delay)
            yield {"type": "ping"}

    def get_final_message(self):
        return self.message

    def close(self):
        self.closed = True


class FakeMessages:
    def __init__(self, script):
        self.script, self.requests, self.streams = list(script), [], []

    def _next(self, kw):
        self.requests.append({**kw, "messages": list(kw["messages"])})
        assert self.script, "fake client script exhausted"
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def stream(self, **kw):
        item = self._next(kw)
        s = FakeStream(item.message, item.n, item.delay) if isinstance(item, Slow) else FakeStream(item)
        self.streams.append(s)
        return s

    def create(self, **kw):
        item = self._next(kw)
        return item.message if isinstance(item, Slow) else item


class FakeClient:
    def __init__(self, script):
        self.messages = FakeMessages(script)
        self.beta = SimpleNamespace(messages=self.messages)


def run_claude(robot, script, mission="pick up the blue cube", timeout=20.0, **settings):
    client = FakeClient(script)
    runner = PilotRunner(robot, client=client)
    runner.start(mission, {"pilot": "claude", **settings})
    assert runner.join(timeout), "mission did not finish"
    return runner, client.messages, runner.status()


def _http_error(cls, status, msg="boom", headers=None):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls(msg, response=httpx2.Response(status, request=req, headers=headers or {}), body=None)


def _jpeg(b64):
    return base64.b64decode(b64)[:2] == b"\xff\xd8"


# ---------------------------------------------------------------------------- tools and senses
def test_fake_robot_implements_interface():
    assert isinstance(FakeRobot(), RobotInterface)


def test_tool_definitions_are_strict_and_deterministic():
    defs = tool_definitions()
    assert {t["name"] for t in defs} == EXPECTED_TOOLS == set(TOOL_NAMES)
    assert json.dumps(defs) == json.dumps(tool_definitions())          # stable bytes -> cacheable prefix

    def walk(schema):
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        assert set(schema["required"]) <= set(schema["properties"])
        for p in schema["properties"].values():
            assert not {"minimum", "maximum", "minLength", "maxLength"} & set(p)   # unsupported in strict mode

    for t in defs:
        assert t["strict"] is True and t["description"] and set(t) == {"name", "description", "input_schema", "strict"}
        walk(t["input_schema"])


def test_toolbox_validates_clamps_and_explains():
    tb = Toolbox(FakeRobot())
    r = tb.execute("walk_to", {"x": float("nan"), "y": 0})
    assert r.is_error and "finite" in r.text
    r = tb.execute("walk_to", {"x": 500, "y": 0})
    assert r.is_error and "out of range" in r.text
    r = tb.execute("walk_relative", {"forward_m": 12, "left_m": 0, "turn_deg": 0})
    assert not r.is_error and "clamped from 12 to 5" in r.text
    r = tb.execute("reach", {"side": "middle", "x": 0, "y": 0, "z": 1})
    assert r.is_error and "'left' or 'right'" in r.text
    r = tb.execute("reach_object", {"side": "right", "name": "red box"})
    assert r.is_error and "out of arm's reach" in r.text and "walk closer" in r.text.lower()
    r = tb.execute("walk_to_object", {"name": "purple unicorn"})
    assert r.is_error and "Known objects" in r.text and "red box" in r.text
    r = tb.execute("get_status", {"verbose": True})
    assert r.is_error and "unexpected argument" in r.text
    r = tb.execute("teleport", {})
    assert r.is_error and "unknown tool" in r.text
    r = tb.execute("grasp", {"side": "left"})
    assert r.is_error and "Nothing is held" in r.text
    r = tb.execute("wait", {"seconds": 5})
    assert not r.is_error and "nothing was moving" in r.text


def test_walk_tools_block_until_arrival_and_report_status():
    robot = FakeRobot()
    tb = Toolbox(robot)
    r = tb.execute("walk_to", {"x": 2.0, "y": 3.0, "yaw_deg": 90})
    assert not r.is_error and "reached" in r.text and "pose: x=2.0" in r.text and "navigation: reached" in r.text
    p = robot.pose()
    assert math.hypot(p["x"] - 2, p["y"] - 3) < 0.05 and abs(p["yaw_deg"] - 90) < 2
    r = tb.execute("walk_to", {"x": 4.0, "y": 1.0})            # inside the red box
    assert r.is_error and "FAILED" in r.text and "no path" in r.text


def test_arm_command_that_never_starts_is_an_error():
    class Detached(FakeRobot):          # the live sim answers like this before the arm controller attaches
        def arm_gesture(self, side, gesture):
            return "arm controller is not attached to a simulation yet"

    r = Toolbox(Detached()).execute("gesture", {"side": "left", "gesture": "wave"})
    assert r.is_error and "did not start moving" in r.text and "not attached" in r.text


def test_paused_simulation_is_reported_instead_of_hanging():
    robot = FakeRobot()
    robot.paused = True
    tb = Toolbox(robot, ToolConfig(stall_wall_s=0.3))
    t = time.perf_counter()
    r = tb.execute("walk_to", {"x": 2.0, "y": 0.0})
    assert r.is_error and "paused" in r.text and time.perf_counter() - t < 1.5


def test_observation_bundle():
    robot = FakeRobot()
    obs = observe(robot, camera="head_cam", width=320, height=240)
    assert obs.image[:2] == b"\xff\xd8" and not obs.errors
    t = obs.text()
    for s in ("pose: x=0.00", "navigation: idle", "arms: left", "lidar nearest m", "red box", "VISIBLE"):
        assert s in t
    ahead = [s for s in obs.lidar if s["bearing_deg"] == 15.0][0]        # the red box sits at +14 deg, 4.1 m
    assert ahead["nearest_m"] is not None and 3.4 < ahead["nearest_m"] < 4.0
    obj, msg = find_object(robot.objects(), "the Red Box")
    assert obj["name"] == "red box"
    obj, msg = find_object(robot.objects(), "box")
    assert obj is None and "red box" in msg and "yellow box" in msg
    obj, _ = find_object(robot.objects(), "box", prefer_nearest=True)
    assert obj["name"] in ("red box", "yellow box")


# ---------------------------------------------------------------------------- Claude pilot (fake client)
def test_claude_pilot_look_walk_reach_grasp_finish_request_shapes():
    robot = FakeRobot()
    r1 = reply(thinking("Look first.", "sig-1"), text("Let me look around."),
               tool_use("tu_1", "look", {"camera": "head_cam"}), tool_use("tu_2", "list_objects", {"visible_only": False}))
    r2 = reply(thinking("Walk to the cube.", "sig-2"), tool_use("tu_3", "walk_to_object", {"name": "blue cube", "standoff_m": 0.5}))
    r3 = reply(tool_use("tu_4", "reach_object", {"side": "right", "name": "blue cube"}))
    r4 = reply(tool_use("tu_5", "grasp", {"side": "right"}))
    r5 = reply(text("Got it."), tool_use("tu_6", "finish", {"summary": "Picked up the blue cube.", "success": True}))
    runner, fake, st = run_claude(robot, [r1, r2, r3, r4, r5])

    assert st["state"] == "done" and st["success"] is True and st["summary"] == "Picked up the blue cube."
    assert robot.arm_status()["right"]["holding"] == "blue cube"
    reqs = fake.requests
    assert len(reqs) == 5 and st["turns"] == 5

    first = reqs[0]
    assert first["model"] == "claude-opus-5-5" and first["max_tokens"] == 16000
    assert first["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert first["output_config"] == {"effort": "medium"}
    assert first["betas"] == [FALLBACK_BETA] and first["fallbacks"] == "default"
    assert first["tool_choice"] == {"type": "auto"}
    assert first["tools"] == tool_definitions() and all(t["strict"] for t in first["tools"])
    assert first["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert first["cache_control"] == {"type": "ephemeral"}
    assert "budget_tokens" not in json.dumps(first["thinking"]) and "temperature" not in first
    m0 = first["messages"]
    assert len(m0) == 1 and m0[0]["role"] == "user"
    img = [b for b in m0[0]["content"] if b["type"] == "image"]
    assert len(img) == 1 and img[0]["source"]["media_type"] == "image/jpeg" and _jpeg(img[0]["source"]["data"])
    assert any("Mission: pick up the blue cube" in b.get("text", "") for b in m0[0]["content"])

    # turn 2: full assistant content replayed unchanged (thinking block + signature), results batched
    m1 = reqs[1]["messages"]
    assert len(m1) == 3 and m1[1]["role"] == "assistant"
    assert m1[1]["content"] == r1.content and m1[1]["content"][0] is r1.content[0]
    assert m1[1]["content"][0].signature == "sig-1"
    assert m1[2]["role"] == "user" and [b["tool_use_id"] for b in m1[2]["content"]] == ["tu_1", "tu_2"]
    assert all(b["type"] == "tool_result" and not b.get("is_error") for b in m1[2]["content"])
    look_blocks = m1[2]["content"][0]["content"]
    assert look_blocks[0]["type"] == "image" and _jpeg(look_blocks[0]["source"]["data"])
    assert look_blocks[-1]["type"] == "text" and "head_cam image" in look_blocks[-1]["text"]
    # append-only history: every request extends the previous one
    for a, b in zip(reqs, reqs[1:]):
        assert b["messages"][:len(a["messages"])] == a["messages"]
        assert b["tools"] == a["tools"] and b["system"] == a["system"]
    assert "holding blue cube" in reqs[4]["messages"][-1]["content"][0]["content"][-1]["text"]

    ev = st["events"]
    kinds = [e["kind"] for e in ev]
    assert kinds[0] == "user" and kinds[-1] == "done" and ev[-1]["state"] == "done"
    assert {"image", "thinking", "text", "tool_call", "tool_result"} <= set(kinds)
    assert [e["name"] for e in ev if e["kind"] == "tool_call"] == \
        ["look", "list_objects", "walk_to_object", "reach_object", "grasp", "finish"]
    assert any(e["kind"] == "thinking" and e["text"] == "Look first." for e in ev)
    assert all(_jpeg(e["image_b64"]) for e in ev if e["kind"] == "image")
    assert [e["seq"] for e in ev] == list(range(len(ev)))
    assert runner.status(since=ev[-2]["seq"])["events"] == ev[-2:]

    p = PRICES["claude-opus-5-5"]
    per_turn = (1000 * p.input + 200 * p.output + 3000 * p.cache_read + 500 * p.cache_write) / 1e6
    assert st["cost_usd"] == pytest.approx(5 * per_turn, rel=1e-6)
    assert (st["input_tokens"], st["output_tokens"], st["cache_read_tokens"]) == (5000, 1000, 15000)


def test_failed_tools_return_is_error_in_one_message():
    robot = FakeRobot()
    r1 = reply(tool_use("a", "walk_to_object", {"name": "purple unicorn"}), tool_use("b", "walk_to", {"x": 1000, "y": 0}),
               tool_use("c", "get_status", {}))
    r2 = reply(tool_use("d", "finish", {"summary": "No unicorn here.", "success": False}))
    _, fake, st = run_claude(robot, [r1, r2], mission="find the purple unicorn")
    results = fake.requests[1]["messages"][-1]["content"]
    assert [b["tool_use_id"] for b in results] == ["a", "b", "c"]
    assert results[0]["is_error"] is True and "Known objects" in results[0]["content"][-1]["text"]
    assert results[1]["is_error"] is True and "out of range" in results[1]["content"][-1]["text"]
    assert "is_error" not in results[2]
    assert st["state"] == "done" and st["success"] is False
    assert [e["is_error"] for e in st["events"] if e["kind"] == "tool_result"] == [True, True, False, False]


def test_refusal_stops_before_running_tools():
    robot = FakeRobot()
    r = reply(tool_use("x", "walk_to", {"x": 1, "y": 1}), stop="refusal",
              stop_details={"type": "refusal", "category": "cyber", "explanation": "declined"})
    _, fake, st = run_claude(robot, [r])
    assert st["state"] == "error" and "refusal" in st["error"] and "cyber" in st["error"]
    assert robot.log == [] and not [e for e in st["events"] if e["kind"] == "tool_call"]
    assert st["events"][-2]["kind"] == "error" and st["events"][-1]["state"] == "error"


@pytest.mark.parametrize("exc,expect", [
    (lambda: _http_error(anthropic.RateLimitError, 429, headers={"retry-after": "7"}), "rate limited (429)"),
    (lambda: _http_error(anthropic.AuthenticationError, 401), "authentication failed"),
    (lambda: _http_error(anthropic.NotFoundError, 404), "not found"),
    (lambda: _http_error(anthropic.InternalServerError, 529, "overloaded"), "API error 529"),
    (lambda: anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com")), "network"),
])
def test_api_errors_end_the_mission_with_a_readable_error(exc, expect):
    _, _, st = run_claude(FakeRobot(), [exc()])
    assert st["state"] == "error" and expect in st["error"]
    if "429" in expect:
        assert "retry after 7 s" in st["error"]


def test_end_turn_without_tool_is_nudged_once_then_accepted():
    r1 = reply(text("I think I am done."), stop="end_turn")
    r2 = reply(text("The robot is already next to the red box."), stop="end_turn")
    _, fake, st = run_claude(FakeRobot(), [r1, r2])
    last = fake.requests[1]["messages"]
    assert last[-1] == {"role": "user", "content": [{"type": "text", "text": NUDGE}]}
    assert st["state"] == "done" and st["summary"] == "The robot is already next to the red box."


def test_max_tokens_is_retried_once_with_a_larger_limit():
    r1 = reply(tool_use("t", "get_status", {}), stop="max_tokens")
    r2 = reply(tool_use("f", "finish", {"summary": "ok", "success": True}))
    _, fake, st = run_claude(FakeRobot(), [r1, r2])
    assert [r["max_tokens"] for r in fake.requests] == [16000, 32000]
    assert fake.requests[1]["messages"] == fake.requests[0]["messages"]   # truncated turn not appended
    assert st["state"] == "done"


def test_fallback_boundary_echo_execution_and_iteration_cost():
    robot = FakeRobot()
    fb = {"type": "fallback", "from": {"model": "claude-opus-5-5"}, "to": {"model": "claude-opus-4-8"},
          "trigger": {"type": "refusal", "category": "cyber"}}
    usage = {"input_tokens": 120, "output_tokens": 80, "iterations": [
        {"type": "message", "model": "claude-opus-5-5", "input_tokens": 100, "output_tokens": 50,
         "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 120, "output_tokens": 80,
         "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}]}
    r1 = reply(thinking("declined model", "sig-x"), tool_use("old", "walk_to", {"x": 1, "y": 1}), fb,
               text("Continuing."), tool_use("new", "get_status", {}), usage=usage, model="claude-opus-4-8")
    r2 = reply(tool_use("f", "finish", {"summary": "ok", "success": True}), usage={"input_tokens": 0, "output_tokens": 0})
    _, fake, st = run_claude(robot, [r1, r2])
    assert robot.log == []                                       # the declined model's walk_to never ran
    echoed = fake.requests[1]["messages"][1]["content"]
    assert [b.type for b in echoed] == ["fallback", "text", "tool_use"]
    assert [b["tool_use_id"] for b in fake.requests[1]["messages"][2]["content"]] == ["new"]
    a, b = PRICES["claude-opus-5-5"], PRICES["claude-opus-4-8"]
    expect = (100 * a.input + 50 * a.output + 120 * b.input + 80 * b.output) / 1e6
    assert st["cost_usd"] == pytest.approx(expect, rel=1e-9)
    assert any(e["name"] == "fallback" and "claude-opus-4-8" in e["text"] for e in st["events"])


def test_budget_limits_stop_the_mission():
    loop = [reply(tool_use(f"s{i}", "get_status", {})) for i in range(5)]
    _, fake, st = run_claude(FakeRobot(), loop, max_turns=2)
    assert st["state"] == "stopped" and "turn limit" in st["error"] and len(fake.requests) == 2
    pricey = [reply(tool_use("s", "get_status", {}), usage={"input_tokens": 1_000_000, "output_tokens": 0})] * 3
    _, fake, st = run_claude(FakeRobot(), pricey, max_cost_usd=1.0)
    assert st["state"] == "stopped" and "cost limit" in st["error"] and len(fake.requests) == 1


def test_build_request_variants():
    tools = tool_definitions()
    hk = build_request(ClaudeConfig(model="claude-haiku-5-5"), tools, [])
    assert "fallbacks" not in hk and "betas" not in hk                  # Haiku 5.5 has no server-side fallback
    off = build_request(ClaudeConfig(fallbacks=False), tools, [])
    assert "fallbacks" not in off and "betas" not in off
    upd = build_request(ClaudeConfig(model="claude-fable-5-2", thinking_display="updates", effort="high"), tools, [])
    assert upd["betas"] == [FALLBACK_BETA, THINKING_UPDATES_BETA] and upd["thinking"]["display"] == "updates"
    assert upd["output_config"] == {"effort": "high"}
    nc = build_request(ClaudeConfig(cache=False), tools, [])
    assert "cache_control" not in nc and "cache_control" not in nc["system"][0]
    with pytest.raises(ValueError):
        ClaudeConfig(model="gpt-4").validate()


def test_usage_cost_without_iterations():
    c = usage_cost({"input_tokens": 2_000_000, "output_tokens": 100_000, "cache_read_input_tokens": 1_000_000,
                    "cache_creation_input_tokens": 0}, "claude-sonnet-5-5")
    assert c["cost_usd"] == pytest.approx(2 * 2.0 + 0.1 * 10.0 + 1 * 0.10)


# ---------------------------------------------------------------------------- cancellation
def test_stop_cancels_during_robot_wait_within_1s():
    robot = FakeRobot(realtime=1.0)
    r0 = reply(tool_use("w0", "walk_relative", {"forward_m": 5, "left_m": 0, "turn_deg": 0}))
    client = FakeClient([r0, reply(tool_use("f", "finish", {"summary": "x", "success": True}))])
    runner = PilotRunner(robot, client=client)
    runner.start("walk forward slowly", {"initial_image": False})
    t_end = time.monotonic() + 5
    while not any(e["kind"] == "tool_call" for e in runner.status()["events"]) and time.monotonic() < t_end:
        time.sleep(0.02)
    time.sleep(0.3)
    t = time.perf_counter()
    assert runner.stop(timeout=2.0)
    dt = time.perf_counter() - t
    st = runner.status()
    assert dt < 1.0 and st["state"] == "stopped" and st["events"][-1]["state"] == "stopped"
    assert ("stop",) in robot.log and not robot.cancelled.is_set()
    assert robot.pose()["x"] < 2.0                                        # it really was interrupted mid-walk


def test_stop_cancels_during_model_request_within_1s():
    client = FakeClient([Slow(reply(tool_use("f", "finish", {"summary": "x", "success": True})), n=400, delay=0.05)])
    runner = PilotRunner(FakeRobot(), client=client)
    runner.start("anything", {"initial_image": False})
    time.sleep(0.3)
    t = time.perf_counter()
    assert runner.stop(timeout=2.0)
    assert time.perf_counter() - t < 1.0
    assert runner.status()["state"] == "stopped" and client.messages.streams[0].closed


def test_wall_clock_budget_cancels_a_long_motion():
    robot = FakeRobot(realtime=1.0)
    client = FakeClient([reply(tool_use("w", "walk_relative", {"forward_m": 5, "left_m": 0, "turn_deg": 0}))])
    runner = PilotRunner(robot, client=client)
    t = time.perf_counter()
    runner.start("walk", {"initial_image": False, "max_wall_seconds": 1.0})
    assert runner.join(5)
    st = runner.status()
    assert st["state"] == "stopped" and "wall-clock limit" in st["summary"] and time.perf_counter() - t < 2.0


def test_runner_rejects_concurrent_missions_and_bad_settings():
    client = FakeClient([Slow(reply(tool_use("f", "finish", {"summary": "x", "success": True})), n=100, delay=0.05)])
    runner = PilotRunner(FakeRobot(), client=client)
    runner.start("first", {"initial_image": False})
    with pytest.raises(RuntimeError):
        runner.start("second")
    runner.stop()
    with pytest.raises(ValueError):
        runner.start("x", {"model": "claude-3-opus"})
    with pytest.raises(ValueError):
        runner.start("   ")


# ---------------------------------------------------------------------------- real SDK over a mock transport
def _sse(message: dict) -> bytes:
    start = {**message, "content": [], "stop_reason": None}
    events = [{"type": "message_start", "message": start}]
    for i, b in enumerate(message["content"]):
        if b["type"] == "thinking":
            events += [{"type": "content_block_start", "index": i, "content_block": {**b, "thinking": "", "signature": ""}},
                       {"type": "content_block_delta", "index": i, "delta": {"type": "thinking_delta", "thinking": b["thinking"]}},
                       {"type": "content_block_delta", "index": i, "delta": {"type": "signature_delta", "signature": b["signature"]}}]
        elif b["type"] == "text":
            events += [{"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}},
                       {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": b["text"]}}]
        else:
            events += [{"type": "content_block_start", "index": i, "content_block": {**b, "input": {}}},
                       {"type": "content_block_delta", "index": i,
                        "delta": {"type": "input_json_delta", "partial_json": json.dumps(b["input"])}}]
        events.append({"type": "content_block_stop", "index": i})
    events += [{"type": "message_delta", "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                "usage": {"output_tokens": message["usage"]["output_tokens"]}}, {"type": "message_stop"}]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def test_real_sdk_request_bodies_over_mock_transport():
    key = "sk-ant-test-0123456789"
    turns = [
        {"content": [thinking("Check the camera.", "sig-real-1"), tool_use("toolu_1", "look", {"camera": "head_cam"})],
         "stop_reason": "tool_use"},
        {"content": [tool_use("toolu_2", "finish", {"summary": "Saw the red box.", "success": True})],
         "stop_reason": "tool_use"},
    ]
    seen = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append((request.headers, json.loads(request.content)))
        t = turns[len(seen) - 1]
        msg = {"id": f"msg_real_{len(seen)}", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
               "stop_sequence": None, "usage": {"input_tokens": 900, "output_tokens": 60}, **t}
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(msg))

    def factory(cfg):
        return anthropic.Anthropic(api_key=cfg.api_key, max_retries=0,
                                   http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))

    runner = PilotRunner(FakeRobot(), client_factory=factory)
    runner.start("what do you see?", {"api_key": key})
    assert runner.join(20)
    st = runner.status()
    assert st["state"] == "done" and st["summary"] == "Saw the red box." and len(seen) == 2
    h0, b0 = seen[0]
    # anthropic 2.10 sends an explicit api_key as "Authorization: Bearer"; older releases used x-api-key
    assert key in (h0.get("x-api-key"), (h0.get("authorization") or "").removeprefix("Bearer "))
    assert FALLBACK_BETA in h0.get("anthropic-beta", "")
    assert b0["stream"] is True and b0["fallbacks"] == "default" and b0["model"] == "claude-opus-5-5"
    assert b0["thinking"] == {"type": "adaptive", "display": "summarized"} and b0["output_config"] == {"effort": "medium"}
    assert b0["tool_choice"] == {"type": "auto"} and all(t["strict"] is True for t in b0["tools"])
    assert "betas" not in b0                                     # betas travel as a header, not in the body
    _, b1 = seen[1]
    assert b1["messages"][1] == {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "Check the camera.", "signature": "sig-real-1"},
        {"type": "tool_use", "id": "toolu_1", "name": "look", "input": {"camera": "head_cam"}}]}
    res = b1["messages"][2]["content"]
    assert len(res) == 1 and res[0]["tool_use_id"] == "toolu_1" and res[0]["content"][0]["type"] == "image"
    assert b1["messages"][:1] == b0["messages"]
    dumped = json.dumps(st) + repr(runner.settings) + repr(runner.__dict__)
    assert key not in dumped and st["settings"]["api_key_set"] is True and runner.settings.api_key is None


def test_missing_credentials_are_reported(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(k, raising=False)

    def handler(request):
        return httpx2.Response(401, json={"type": "error", "error": {"type": "authentication_error", "message": "x"}})

    def factory(cfg):
        return anthropic.Anthropic(max_retries=0,
                                   http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))

    runner = PilotRunner(FakeRobot(), client_factory=factory)
    runner.start("hello", {"initial_image": False})
    assert runner.join(10)
    st = runner.status()
    assert st["state"] == "error"
    assert "credentials" in st["error"] or "authentication failed" in st["error"]


# ---------------------------------------------------------------------------- scripted pilot
def _scripted(mission, robot=None):
    robot = robot or FakeRobot()
    runner = PilotRunner(robot)
    runner.start(mission, {"pilot": "scripted"})
    assert runner.join(10)
    return robot, runner.status()


def test_scripted_walk_to_the_red_box():
    robot, st = _scripted("walk to the red box")
    assert st["state"] == "done" and st["success"] is True
    p, box = robot.pose(), robot.obj("red box")
    d = math.hypot(p["x"] - box.x, p["y"] - box.y)
    assert abs(d - (box.size_m / 2 + 0.5)) < 0.06                       # stopped at the standoff
    bearing = [o for o in robot.objects() if o["name"] == "red box"][0]["bearing_deg"]
    assert abs(bearing) < 3 and [e["name"] for e in st["events"] if e["kind"] == "tool_call"] == ["walk_to_object"]


def test_scripted_other_missions():
    robot, st = _scripted("go to 2 3")
    assert st["state"] == "done" and math.hypot(robot.x - 2, robot.y - 3) < 0.05
    robot, st = _scripted("pick up the blue cube")
    assert st["state"] == "done" and any(a["holding"] == "blue cube" for a in robot.arm_status().values())
    assert robot.obj("blue cube").z > 0.9                                # lifted off the table
    robot, st = _scripted("wave")
    assert st["state"] == "done" and ("gesture", "right", "wave") in robot.log
    robot, st = _scripted("turn left 90 degrees then walk forward 1 m")
    assert st["state"] == "done" and abs(robot.x) < 0.05 and abs(robot.y - 1) < 0.05
    robot, st = _scripted("walk to the red box and wave")
    assert st["state"] == "done" and ("gesture", "right", "wave") in robot.log
    robot, st = _scripted("do a backflip")
    assert st["state"] == "error" and "understands" in st["error"]
    robot, st = _scripted("walk to the purple unicorn")
    assert st["state"] == "error" and "Known objects" in st["error"]


def test_gui_form_settings_brief_and_events_api():
    r1 = reply(tool_use("l", "look", {}))
    r2 = reply(tool_use("f", "finish", {"summary": "ok", "success": True}))
    client = FakeClient([r1, r2])
    runner = PilotRunner(FakeRobot(), client=client)
    assert runner.brief()["state"] == "idle" and runner.events() == {"events": [], "next": 0}
    runner.start("look", {"provider": "claude", "images": False, "max_turns": 40, "max_cost_usd": 2,
                          "fallbacks": False, "mission": "ignored here"})
    assert runner.join(10)
    b = runner.brief()
    assert b["run"] == 1 and b["state"] == "done" and b["n_events"] > 0 and "events" not in b
    first = client.messages.requests[0]
    assert not [x for x in first["messages"][0]["content"] if x["type"] == "image"] and "fallbacks" not in first
    look = client.messages.requests[1]["messages"][-1]["content"][0]["content"]
    assert [x["type"] for x in look] == ["text"] and "disabled by the operator" in look[0]["text"]
    ev = runner.events(since=2)
    assert ev["next"] == b["n_events"] and ev["events"][0]["seq"] == 2


def test_settings_are_validated_and_key_is_private():
    s = PilotSettings.from_dict({"model": "claude-sonnet-5-5", "max_turns": 10_000, "api_key": " sk-x ", "bogus": 1})
    assert s.max_turns == 500 and s.api_key == "sk-x" and "sk-x" not in repr(s)
    assert "api_key" not in s.public_dict() and s.public_dict()["api_key_set"] is True
    with pytest.raises(ValueError):
        PilotSettings.from_dict({"effort": "extreme"})
    assert ScriptedPilot().plan("go to (1.5, -2) facing 90", FakeRobot())[0].calls[0] == \
        ("walk_to", {"x": 1.5, "y": -2.0, "yaw_deg": 90.0})


# ---------------------------------------------------------------------------- review regressions
def test_settings_reject_nan_budgets_and_parse_bool_strings():
    # NaN compared with anything is False, so a NaN budget used to disable the cost / wall-clock limit
    for key in ("max_cost_usd", "max_wall_seconds", "max_turns", "max_tokens"):
        with pytest.raises(ValueError):
            PilotSettings.from_dict({key: float("nan")})
    with pytest.raises(ValueError):
        PilotSettings.from_dict({"max_cost_usd": "lots"})
    s = PilotSettings.from_dict({"max_cost_usd": float("inf"), "max_turns": "7"})
    assert s.max_cost_usd == 1000.0 and s.max_turns == 7
    # bool("false") is True: a form post with "false" used to turn images / fallbacks ON
    s = PilotSettings.from_dict({"images": "false", "fallbacks": "False", "stream": 0, "initial_image": "yes"})
    assert (s.send_images, s.fallbacks, s.stream, s.initial_image) == (False, False, False, False)
    assert PilotSettings.from_dict({"fallbacks": "true"}).fallbacks is True
    with pytest.raises(ValueError):
        PilotSettings.from_dict({"images": "maybe"})


def test_start_with_settings_object_does_not_mutate_it():
    mine = PilotSettings(pilot="scripted", api_key="sk-keep-me")
    runner = PilotRunner(FakeRobot())
    runner.start("wave", mine)
    assert runner.join(10) and runner.status()["state"] == "done"
    assert mine.api_key == "sk-keep-me" and runner.settings.api_key is None    # caller's copy untouched
    with pytest.raises(ValueError):
        runner.start("wave", PilotSettings(pilot="autopilot"))                 # objects are validated too


def test_arm_reset_mid_motion_is_reported_as_an_error():
    class ResetMidReach(FakeRobot):
        """Like the live session's auto-reset after a fall: the arm comes back fresh (idle) mid-motion."""

        def wait(self, seconds):
            super().wait(seconds)
            if self.t > 0.3:
                for a in self.arms.values():
                    a.state, a.target, a.skill = "idle", None, "home"

    r = Toolbox(ResetMidReach(objects=[])).execute("reach", {"side": "right", "x": 0.4, "y": -0.2, "z": 1.0})
    assert r.is_error and "INTERRUPTED" in r.text
    ok = Toolbox(FakeRobot(objects=[])).execute("reach", {"side": "right", "x": 0.4, "y": -0.2, "z": 1.0})
    assert not ok.is_error and "done after" in ok.text


def test_empty_reply_is_not_echoed_as_an_empty_assistant_message():
    # an end_turn with no content (seen right after tool results) must not become {"role": "assistant",
    # "content": []} in the history: the API rejects empty non-final messages, killing the mission with a 400
    r1 = reply(tool_use("s", "get_status", {}))
    r2 = reply(stop="end_turn")
    r3 = reply(tool_use("f", "finish", {"summary": "ok", "success": True}))
    _, fake, st = run_claude(FakeRobot(), [r1, r2, r3], initial_image=False)
    msgs = fake.requests[2]["messages"]
    assert all(m["content"] for m in msgs)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "user"] and msgs[-1]["content"][0]["text"] == NUDGE
    assert st["state"] == "done"


def test_scripted_and_clauses_do_not_drop_actions():
    p, robot = ScriptedPilot(), FakeRobot()
    assert [s.label for s in p.plan("wave and walk to the red box", robot)] == ["wave", "walk to the red box"]
    assert [s.label for s in p.plan("look and wave", robot)] == ["look", "wave"]
    assert [s.label for s in p.plan("raise your left hand and walk forward 1 m", robot)] == ["raise hand", "walk +1 m"]
    assert [s.label for s in p.plan("walk to the red box and wave", robot)] == ["walk to the red box", "wave"]
    named = FakeRobot(objects=[FakeObject("black and white box", "white", "box", 3.0, 0.0, 0.3, 0.6)])
    assert [s.label for s in p.plan("walk to the black and white box", named)] == ["walk to the black and white box"]
    robot, st = _scripted("wave and walk to the red box")
    assert st["state"] == "done" and [e["name"] for e in st["events"] if e["kind"] == "tool_call"] == \
        ["gesture", "gesture", "walk_to_object"]


# ---------------------------------------------------------------------------- optional live test
@pytest.mark.skipif(os.environ.get("SIMULATICS_LIVE_CLAUDE") != "1", reason="set SIMULATICS_LIVE_CLAUDE=1 (costs money)")
def test_live_claude_walks_to_the_red_box():
    robot = FakeRobot()
    runner = PilotRunner(robot)
    runner.start("Walk to the red box, then finish.", {
        "model": os.environ.get("SIMULATICS_LIVE_MODEL", "claude-opus-5-5"), "effort": "low",
        "max_turns": 12, "max_cost_usd": 0.5, "max_wall_seconds": 240})
    assert runner.join(300)
    st = runner.status()
    print(json.dumps({k: st[k] for k in ("state", "turns", "cost_usd", "input_tokens", "output_tokens", "summary")}))
    assert st["state"] == "done", st["error"]
    box = robot.obj("red box")
    assert math.hypot(robot.x - box.x, robot.y - box.y) < 1.5
