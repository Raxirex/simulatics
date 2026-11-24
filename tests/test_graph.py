"""Node graph engine: graph editing, execution, caching, cancellation, and the built-in nodes on a real session."""
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from simulatics.graph import (EXAMPLES, REGISTRY, Executor, Graph, GraphError, Port, RobotSpec, coerce, compatible,
                              node, summarize)

CALLS: dict[str, int] = {}


@pytest.fixture(scope="module", autouse=True)
def instrumented_nodes():
    """A few instrumented node types, removed again so they never show up in the editor."""
    @node("_test.add", "Add one", "Script", inputs=[Port("x", "number")], outputs=[Port("y", "number")],
          params=[Port("k", "number", 1.0)], doc="Test node: y = x + k, counting its calls.")
    def add(ctx, x, k):
        CALLS[ctx.node_id] = CALLS.get(ctx.node_id, 0) + 1
        return {"y": x + k}

    @node("_test.live", "Live", "Script", inputs=[Port("x", "number")], outputs=[Port("y", "number")], volatile=True,
          doc="Test node: volatile passthrough, counting its calls.")
    def live(ctx, x):
        CALLS[ctx.node_id] = CALLS.get(ctx.node_id, 0) + 1
        return {"y": x}

    @node("_test.sleep", "Sleep", "Script", inputs=[Port("x", "number")], outputs=[Port("y", "number")],
          params=[Port("seconds", "number", 5.0)], doc="Test node: waits, reporting progress.")
    def sleep(ctx, x, seconds):
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            ctx.progress((time.monotonic() - t0) / seconds, "waiting")
            time.sleep(0.01)
        return {"y": x}

    @node("_test.fail", "Fail", "Script", inputs=[Port("x", "number")], outputs=[Port("y", "number")],
          doc="Test node: always raises.")
    def fail(ctx, x):
        raise ValueError("boom")

    yield
    for k in [k for k in REGISTRY if k.startswith("_test.")]:
        del REGISTRY[k]


def chain(*types, params=None):
    """value.number -> the given single-input test nodes, wired in a row."""
    g = Graph()
    src = g.add_node("value.number", params={"value": 1.0}, id="src")
    prev, port = src, "value"
    for i, t in enumerate(types):
        n = g.add_node(t, pos=(260 * (i + 1), 0), id=f"n{i}", params=(params or {}).get(i))
        g.connect(prev.id, port, n.id, "x")
        prev, port = n, "y"
    return g


def record(events):
    return lambda kind, nid, payload: events.append((kind, nid, payload))


def kinds(events, kind):
    return [nid for k, nid, _ in events if k == kind]


# ---- graph editing ---------------------------------------------------------------------------------------
def test_connect_checks_types_ports_and_loops():
    g = Graph()
    r = g.add_node("robot.builtin")
    w = g.add_node("world.obstacles")
    a = g.add_node("sim.apply")
    num = g.add_node("value.number")
    assert g.connect(r.id, "robot", w.id, "robot").dst == w.id
    g.connect(w.id, "robot", a.id, "robot")
    with pytest.raises(GraphError, match="number"):
        g.connect(num.id, "value", a.id, "robot")                 # a number is not a robot
    with pytest.raises(GraphError, match="no output"):
        g.connect(r.id, "nope", a.id, "robot")
    with pytest.raises(GraphError, match="no input"):
        g.connect(r.id, "robot", a.id, "nope")
    with pytest.raises(GraphError, match="loop"):
        g.connect(a.id, "robot", w.id, "robot")                   # a -> w -> a
    with pytest.raises(GraphError, match="loop"):
        g.connect(w.id, "robot", w.id, "robot")
    assert g.can_connect(num.id, "value", a.id, "robot")[0] is False
    assert g.can_connect(r.id, "robot", a.id, "robot") == (True, "")
    g.connect(r.id, "robot", a.id, "robot")                       # replaces the link into a.robot
    assert [(l.src, l.dst) for l in g.links if l.dst == a.id] == [(r.id, a.id)]
    g.disconnect(a.id, "robot")
    assert not g.inputs_of(a.id)
    g.remove_node(r.id)
    assert r.id not in g.nodes and not g.links
    with pytest.raises(GraphError):
        g.add_node("no.such.type")
    with pytest.raises(GraphError):
        g.add_node("value.number", params={"nope": 1})


def test_compatible_and_coerce():
    assert compatible("int", "number") and not compatible("number", "int")
    assert compatible("robot", "any") and compatible("any", "series") and not compatible("text", "number")
    assert coerce(Port("n", "int", 0, min=1, max=10), "42.4") == 10
    assert coerce(Port("b", "bool"), "false") is False and coerce(Port("x", "number"), 3) == 3.0
    with pytest.raises(GraphError):
        coerce(Port("c", "choice", "a", choices=["a", "b"]), "z")
    with pytest.raises(GraphError):
        coerce(Port("x", "number"), "abc")


def test_topo_order_and_targets():
    g = chain("_test.add", "_test.add")
    side = g.add_node("value.text", id="side")
    order = g.topo_order()
    assert order.index("src") < order.index("n0") < order.index("n1") and "side" in order
    assert g.topo_order(["n0"]) == ["src", "n0"]
    assert g.topo_order("side") == ["side"]
    with pytest.raises(GraphError):
        g.topo_order(["missing"])


def test_json_round_trip(tmp_path):
    g = EXAMPLES["Walk to a goal"]()
    g.nodes[next(iter(g.nodes))].title = "My robot"
    g.save(tmp_path / "g.json")
    h = Graph.load(tmp_path / "g.json")
    assert h.to_dict() == g.to_dict() and h.to_dict()["version"] == 1
    assert list(h.nodes) == list(g.nodes) and h.links == g.links
    n = next(iter(h.nodes.values()))
    assert isinstance(n.pos, tuple) and n.title == "My robot"
    assert h.validate() == []


def test_validate_reports_missing_inputs_and_bad_params():
    g = Graph()
    a = g.add_node("sim.apply")
    s = g.add_node("script.python")          # a, b, c are optional
    m = g.add_node("math.expr")              # a, b have defaults
    lib = g.add_node("robot.library", params={"name": "r2d2"})
    problems = g.validate()
    assert len(problems) == 2
    assert any("robot" in p and a.id in p for p in problems) and any(lib.id in p for p in problems)
    assert not any(s.id in p or m.id in p for p in problems)


def test_every_node_type_is_documented_and_constructs():
    assert len(REGISTRY) >= 30
    g = Graph()
    for tid, nt in REGISTRY.items():
        assert nt.doc.strip() and "\n" not in nt.doc, tid
        assert nt.color.startswith("#") and nt.category and nt.title, tid
        inst = g.add_node(tid)
        assert set(inst.params) == {p.name for p in nt.params}, tid
        for p in nt.params:
            assert coerce(p, inst.params[p.name]) == inst.params[p.name], (tid, p.name)
    cats = {nt.category: nt.color for nt in REGISTRY.values() if not nt.id.startswith("_test.")}
    assert len(set(cats.values())) == len(cats)        # one header colour per category
    for name, make in EXAMPLES.items():
        assert make().validate() == [], name


def test_summarize():
    assert summarize(np.zeros((5, 3))) == "5 points"
    assert summarize(np.zeros((240, 320, 3), np.uint8)) == "image 320x240"
    assert "2 rows" in summarize([{"a": 1}, {"a": 2}])
    assert summarize(RobotSpec()) == "built-in humanoid"
    assert "1 line" in summarize({"title": "t", "x": [1, 2], "y": {"a": [1, 2]}})
    assert summarize(1.23456789) == "1.235" and summarize(None) == "none"


# ---- execution -------------------------------------------------------------------------------------------
def test_cache_reuses_results_and_reruns_only_downstream_of_a_change():
    CALLS.clear()
    g = chain("_test.add", "_test.add")
    ex, ev = Executor(), []
    out = ex.run(g, on_event=record(ev))
    assert out["n1"]["y"] == 3.0
    assert kinds(ev, "done") == ["src", "n0", "n1"] and not kinds(ev, "cached")
    assert ev[-1][0] == "finished" and ev[-1][2]["ok"] and not ev[-1][2]["cancelled"]
    done = next(p for k, n, p in ev if k == "done" and n == "n1")
    assert done["outputs"] == {"y": "3"} and done["seconds"] >= 0

    ev.clear()
    assert ex.run(g, on_event=record(ev))["n1"]["y"] == 3.0
    assert kinds(ev, "cached") == ["src", "n0", "n1"] and not kinds(ev, "start")
    assert CALLS == {"n0": 1, "n1": 1}

    ev.clear()
    g.nodes["n1"].params["k"] = 5.0
    g.nodes["n1"].pos = (999, 9)                       # moving a node is not a change
    assert ex.run(g, on_event=record(ev))["n1"]["y"] == 7.0
    assert kinds(ev, "cached") == ["src", "n0"] and kinds(ev, "done") == ["n1"]

    ev.clear()
    g.nodes["src"].params["value"] = 10.0              # upstream change: everything below re-runs
    assert ex.run(g, on_event=record(ev))["n1"]["y"] == 16.0
    assert kinds(ev, "done") == ["src", "n0", "n1"]
    ex.clear_cache()
    assert not ex.cache


def test_volatile_nodes_and_their_downstream_always_rerun():
    CALLS.clear()
    g = chain("_test.add", "_test.live", "_test.add")
    ex, ev = Executor(), []
    ex.run(g)
    ex.run(g, on_event=record(ev))
    assert kinds(ev, "cached") == ["src", "n0"] and kinds(ev, "done") == ["n1", "n2"]
    assert CALLS == {"n0": 1, "n1": 2, "n2": 2}
    assert "n1" not in ex.cache and "n2" not in ex.cache


def test_cancel_mid_run_stops_the_node_and_skips_the_rest():
    CALLS.clear()
    g = chain("_test.sleep", "_test.add")
    ex, ev, cancel = Executor(), [], threading.Event()
    threading.Timer(0.2, cancel.set).start()
    t0 = time.monotonic()
    out = ex.run(g, on_event=record(ev), cancel=cancel)
    assert time.monotonic() - t0 < 3.0
    assert list(out) == ["src"] and "n1" not in CALLS
    assert kinds(ev, "progress") and set(kinds(ev, "progress")) == {"n0"}
    assert not kinds(ev, "error") and "n0" not in kinds(ev, "done")
    assert ev[-1] == ("finished", None, ev[-1][2]) and ev[-1][2]["cancelled"] and not ev[-1][2]["ok"]
    # the same executor runs again afterwards
    g.nodes["n0"].params["seconds"] = 0.0
    assert ex.run(g)["n1"]["y"] == 2.0


def test_failing_node_reports_error_and_stops_the_run():
    CALLS.clear()
    g = chain("_test.add", "_test.fail", "_test.add")
    ex, ev = Executor(), []
    out = ex.run(g, on_event=record(ev))               # does not raise
    assert list(out) == ["src", "n0"] and "n2" not in CALLS
    (nid, payload), = [(n, p) for k, n, p in ev if k == "error"]
    assert nid == "n1" and payload["message"] == "ValueError: boom" and "boom" in payload["traceback"]
    assert "n2" not in kinds(ev, "start")
    assert ev[-1][0] == "finished" and ev[-1][2] == {"ok": False, "cancelled": False, "seconds": ev[-1][2]["seconds"]}


def test_unconnected_required_input_is_an_error_event():
    g = Graph()
    g.add_node("out.print", id="p")
    ev = []
    assert Executor().run(g, on_event=record(ev)) == {}
    assert "not connected" in next(p["message"] for k, n, p in ev if k == "error")


def test_script_node_and_values():
    g = EXAMPLES["Script"]()
    ev = []
    out = Executor().run(g, on_event=record(ev))
    script = next(n for n in g.nodes.values() if n.type == "script.python")
    printer = next(n for n in g.nodes.values() if n.type == "out.print")
    assert out[script.id]["out"] == 10.0 and "a squared plus one is 10.0" in out[script.id]["stdout"]
    assert out[printer.id]["text"] == "10"
    assert ev[-1][2]["ok"]

    g = Graph()
    a = g.add_node("value.number", params={"value": 3})
    b = g.add_node("value.int", params={"value": 4})
    e = g.add_node("math.expr", params={"expr": "sqrt(a**2 + b**2)"})
    s = g.add_node("script.python", params={
        "code": "rows = [{'i': i, 'sq': i * i} for i in range(int(a))]\nout = rows\nprint(len(rows), b, c, session)"})
    t = g.add_node("series.from_table", params={"x": "i", "y": "sq"})
    p = g.add_node("out.plot")
    g.connect(a.id, "value", e.id, "a")
    g.connect(b.id, "value", e.id, "b")                # int feeds number
    g.connect(e.id, "out", s.id, "a")
    g.connect(s.id, "out", t.id, "table")              # any feeds table
    g.connect(t.id, "series", p.id, "series")
    out = Executor().run(g)
    assert out[e.id]["out"] == 5.0
    assert out[s.id]["stdout"].strip() == "5 None None None"
    assert out[p.id]["series"]["x"] == [0, 1, 2, 3, 4] and out[p.id]["series"]["y"] == {"sq": [0, 1, 4, 9, 16]}

    bad = Graph()
    bad.add_node("script.python", params={"code": "out = 1 / 0"}, id="s")
    ev = []
    Executor().run(bad, on_event=record(ev))
    assert "ZeroDivisionError" in next(p["message"] for k, n, p in ev if k == "error")


def test_robot_nodes_build_specs_without_touching_shared_values(tmp_path):
    g = Graph()
    r = g.add_node("robot.builtin", params={"thigh_length": 0.45})
    p = g.add_node("robot.set_param", params={"name": "knee_kp", "value": 99999})
    cam = g.add_node("robot.accessory", params={"type": "camera", "body": "torso", "x": 0.12})
    box = g.add_node("robot.accessory", params={"type": "payload", "body": "pelvis", "mass": 3.0})
    w = g.add_node("world.terrain", params={"kind": "stairs"})
    m0 = g.add_node("design.mass_report")
    m1 = g.add_node("design.mass_report")
    save = g.add_node("out.save_json", params={"path": str(tmp_path / "robot.json")})
    g.connect(r.id, "robot", p.id, "robot")
    g.connect(p.id, "robot", cam.id, "robot")
    g.connect(p.id, "robot", box.id, "robot")           # two branches off one spec
    g.connect(box.id, "robot", w.id, "robot")
    g.connect(p.id, "robot", m0.id, "robot")
    g.connect(box.id, "robot", m1.id, "robot")
    g.connect(w.id, "robot", save.id, "value")
    ev = []
    out = Executor().run(g, on_event=record(ev))
    assert ev[-1][2]["ok"], [p for k, n, p in ev if k == "error"]
    base = out[p.id]["robot"]
    assert base.params == {"thigh_length": 0.45, "knee_kp": 800.0} and base.accessories is None   # clipped, unshared
    assert [a["type"] for a in out[cam.id]["robot"].accessories] == ["camera"]
    assert [a["type"] for a in out[box.id]["robot"].accessories] == ["payload"]
    assert out[w.id]["robot"].world == {"terrain": "stairs", "seed": 0} and out[w.id]["goal_x"] > 1.0
    assert out[box.id]["robot"].world is None
    assert out[m1.id]["total"] == pytest.approx(out[m0.id]["total"] + 3.0, abs=1e-3)
    assert any(row["body"] == "torso" for row in out[m0.id]["table"])
    assert '"terrain": "stairs"' in Path(out[save.id]["path"]).read_text()
    assert RobotSpec.from_dict(out[w.id]["robot"].to_dict()).key() == out[w.id]["robot"].key()


def test_library_robot_that_is_not_downloaded_is_a_clear_error(monkeypatch):
    from simulatics import robots
    monkeypatch.setattr(robots, "available", lambda name: False)
    monkeypatch.setattr(robots, "fetch", lambda *a, **k: pytest.fail("must not download"))
    g = Graph()
    g.add_node("robot.library", params={"name": "go2"})
    ev = []
    Executor().run(g, on_event=record(ev))
    assert "not downloaded" in next(p["message"] for k, n, p in ev if k == "error")


# ---- nodes on a live session -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def live():
    ex = Executor()                 # creates its own SimSession on first use
    yield ex
    ex.close()
    assert ex.session is None


def test_builtin_robot_runs_in_a_real_session(live):
    g = Graph()
    r = g.add_node("robot.builtin", params={"default_lidar": True})
    a = g.add_node("sim.apply")
    s = g.add_node("sim.run", params={"seconds": 1.0, "speed": 8.0})
    l = g.add_node("sensor.lidar")
    g.connect(r.id, "robot", a.id, "robot")
    g.connect(a.id, "robot", s.id, "robot")
    g.connect(s.id, "robot", l.id, "robot")
    ev = []
    out = live.run(g, on_event=record(ev))
    assert ev[-1][2]["ok"], [p for k, n, p in ev if k == "error"]
    sess = live.session
    assert sess is not None and sess.robot_name == "builtin" and sess.controller.name == "walker"
    series = out[s.id]["series"]
    assert set(series["y"]) == {"root height (m)", "speed (m/s)"}
    assert len(series["x"]) >= 5 and len(series["x"]) == len(series["y"]["root height (m)"])
    assert 0.95 <= series["x"][-1] < 2.0 and series["x"] == sorted(series["x"])
    assert min(series["y"]["root height (m)"]) > 0.5 and out[s.id]["falls"] == 0 and out[s.id]["distance"] < 0.5
    assert sess.running is False                                   # left paused, as sim.apply left it
    assert out[l.id]["points"].shape[1] == 3 and out[l.id]["hits"] == len(out[l.id]["points"]) > 0
    # sim.run and sensor.lidar reused what sim.apply built (one rebuild only)
    version = sess.scene_version
    ev.clear()
    live.run(g, targets=[r.id])
    assert sess.scene_version == version
    # volatile nodes are never cached; the pure robot node is
    live.run(g, on_event=record(ev))
    assert kinds(ev, "cached") == [r.id] and kinds(ev, "done") == [a.id, s.id, l.id]


def test_walk_to_goal_with_obstacles_and_a_mounted_camera(live):
    g = Graph()
    r = g.add_node("robot.builtin")
    c = g.add_node("robot.accessory", params={"type": "camera", "body": "torso", "x": 0.12, "z": 0.1})
    w = g.add_node("world.obstacles", params={"n": 6, "seed": 1})
    k = g.add_node("sim.walk_to", params={"x": 0.8, "y": 0.3, "timeout": 30.0, "speed": 8.0})
    p = g.add_node("sim.push_test", params={"force": 40.0, "watch": 1.0, "speed": 8.0})
    g.connect(r.id, "robot", c.id, "robot")
    g.connect(c.id, "robot", w.id, "robot")
    g.connect(w.id, "robot", k.id, "robot")
    g.connect(k.id, "robot", p.id, "robot")
    ev = []
    out = live.run(g, on_event=record(ev))
    assert ev[-1][2]["ok"], [p for k_, n, p in ev if k_ == "error"]
    sess = live.session
    assert len(sess.obstacles) == 6 and "cam_g1" in sess.robot.cameras()
    assert [a["id"] for a in sess.accessories] == ["lidar", "g1"]
    assert out[k.id]["reached"] is True and out[k.id]["final_error"] < 0.15 and 0 < out[k.id]["seconds"] < 30
    path = out[k.id]["path"]
    assert len(path["x"]) > 3 and abs(path["x"][-1] - 0.8) < 0.15
    assert out[p.id]["survived"] is True
    assert sess.speed == 1.0 and sess.running is False             # speed and pause state are restored
    # a camera picture through the simulation thread's renderer
    g2 = Graph()
    cam = g2.add_node("sensor.camera", params={"camera": "cam_g1", "width": 160, "height": 120})
    g2.nodes[w.id] = g.nodes[w.id]
    for nid in (r.id, c.id):
        g2.nodes[nid] = g.nodes[nid]
    g2.links = [l for l in g.links if l.dst in (c.id, w.id)]
    g2.connect(w.id, "robot", cam.id, "robot")
    ev.clear()
    version = sess.scene_version
    out = live.run(g2, on_event=record(ev))
    errors = [p["message"] for k_, n, p in ev if k_ == "error"]
    if errors and any(w in errors[0].lower() for w in ("opengl", "gl context", "glfw", "egl", "render")):
        pytest.skip(f"offscreen rendering not available: {errors[0]}")
    assert not errors, errors
    assert out[cam.id]["image"].shape == (120, 160, 3) and out[cam.id]["image"].dtype == np.uint8
    assert sess.scene_version == version                           # same spec: nothing was rebuilt


def test_tiny_es_training_gives_a_policy_file_and_a_curve(live):
    name = f"_test_graph_es_{int(time.time() * 1000) % 100000000}"
    run_dir = Path("runs") / name
    try:
        g = Graph()
        r = g.add_node("robot.builtin")
        t = g.add_node("train.run", params={"algo": "es", "name": name, "population": 8, "generations": 2,
                                            "episode_seconds": 1.0})
        u = g.add_node("policy.use")
        s = g.add_node("sim.run", params={"seconds": 0.5, "speed": 8.0})
        g.connect(r.id, "robot", t.id, "robot")
        g.connect(r.id, "robot", u.id, "robot")
        g.connect(t.id, "policy", u.id, "policy")
        g.connect(u.id, "robot", s.id, "robot")
        ev = []
        out = live.run(g, on_event=record(ev))
        assert ev[-1][2]["ok"], [p for k, n, p in ev if k == "error"]
        policy = out[t.id]["policy"]
        assert policy["name"] == name and policy["kind"] == "es"
        assert Path(policy["path"]).is_file() and Path(policy["path"]).parent == run_dir
        curve = out[t.id]["series"]
        assert curve["x"] == [1, 2] and set(curve["y"]) == {"best", "mean"} and len(curve["y"]["best"]) == 2
        assert np.isfinite(out[t.id]["best"]) and out[t.id]["best"] == pytest.approx(max(curve["y"]["best"]), abs=10)
        assert live.session.controller.name == "es-policy" and live.session.policy_name == name
        assert len(out[s.id]["series"]["x"]) > 2
        # training is cached by its settings; the simulation nodes run again
        ev.clear()
        live.run(g, on_event=record(ev))
        assert t.id in kinds(ev, "cached") and t.id not in kinds(ev, "start") and s.id in kinds(ev, "done")
        # the next graph without a policy gets the model-based walker back
        g2 = Graph()
        r2 = g2.add_node("robot.builtin")
        a2 = g2.add_node("sim.apply")
        g2.connect(r2.id, "robot", a2.id, "robot")
        live.run(g2)
        assert live.session.controller.name == "walker" and live.session.policy_name == ""
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_training_can_be_cancelled(live):
    name = f"_test_graph_cancel_{int(time.time() * 1000) % 100000000}"
    try:
        g = Graph()
        r = g.add_node("robot.builtin")
        t = g.add_node("train.run", params={"algo": "es", "name": name, "population": 16, "generations": 100000,
                                            "episode_seconds": 2.0})
        g.connect(r.id, "robot", t.id, "robot")
        ev, cancel = [], threading.Event()
        threading.Timer(1.5, cancel.set).start()
        t0 = time.monotonic()
        out = live.run(g, on_event=record(ev), cancel=cancel)
        assert time.monotonic() - t0 < 20 and t.id not in out and t.id not in live.cache
        assert ev[-1][2]["cancelled"] and not kinds(ev, "error")
        progress = [p for k, n, p in ev if k == "progress" and n == t.id]
        assert progress and {"frac", "message", "history"} <= set(progress[-1])
        assert not any(th.name == "trainer" and th.is_alive() for th in threading.enumerate())
    finally:
        shutil.rmtree(Path("runs") / name, ignore_errors=True)


def test_lidar_on_a_library_dog(live):
    from simulatics import robots
    if not robots.available("go2"):
        pytest.skip("go2 not downloaded (python -m simulatics.robots fetch go2)")
    g = EXAMPLES["Lidar on a dog"]()
    ev = []
    out = live.run(g, on_event=record(ev))
    assert ev[-1][2]["ok"], [p for k, n, p in ev if k == "error"]
    sess = live.session
    assert sess.robot_name == "go2" and sess.lidar.mounted and sess.accessories[0]["body"] == "base"
    lidar = next(n for n in g.nodes.values() if n.type == "sensor.lidar")
    assert out[lidar.id]["hits"] > 100
    printer = next(n for n in g.nodes.values() if n.type == "out.print")
    assert float(out[printer.id]["text"]) == out[lidar.id]["hits"]


def test_remaining_value_output_and_policy_nodes(live, tmp_path):
    g = Graph()
    txt = g.add_node("value.text", params={"value": "hello"})
    flag = g.add_node("value.bool", params={"value": True})
    pr = g.add_node("out.print")
    pols = g.add_node("policy.list")
    tab = g.add_node("out.table")
    s1 = g.add_node("script.python", params={
        "code": "out = {'title': 'one', 'x': [0, 1, 2], 'y': {'v': [0, 1, 2]}, 'xlabel': 't', 'ylabel': ''}\nprint(a, b)"})
    s2 = g.add_node("script.python", params={"code": "out = {'title': 'two', 'x': [0, 2], 'y': {'v': [10, 30]}}"})
    mer = g.add_node("series.merge")
    img = g.add_node("script.python", params={"code": "out = np.zeros((4, 6, 3), np.uint8)"})
    oi = g.add_node("out.image")
    r = g.add_node("robot.builtin")
    wo = g.add_node("world.obstacles")
    wc = g.add_node("world.clear")
    g.connect(txt.id, "value", pr.id, "value")
    g.connect(txt.id, "value", s1.id, "a")
    g.connect(flag.id, "value", s1.id, "b")
    g.connect(pols.id, "table", tab.id, "table")
    g.connect(s1.id, "out", mer.id, "a")
    g.connect(s2.id, "out", mer.id, "b")
    g.connect(img.id, "out", oi.id, "image")
    g.connect(r.id, "robot", wo.id, "robot")
    g.connect(wo.id, "robot", wc.id, "robot")
    ev = []
    out = Executor().run(g, on_event=record(ev))
    assert ev[-1][2]["ok"], [p for k, n, p in ev if k == "error"]
    assert out[pr.id]["text"] == "hello" and out[s1.id]["stdout"] == "hello True\n"
    assert any(row["name"] == "pretrained_walker" for row in out[tab.id]["table"])
    assert out[mer.id]["series"]["x"] == [0, 1, 2]
    assert out[mer.id]["series"]["y"] == {"v": [0, 1, 2], "v (b)": [10.0, 20.0, 30.0]}   # b resampled onto a's x
    assert out[oi.id]["image"].shape == (4, 6, 3)
    assert out[wo.id]["robot"].world == {"obstacles": 12, "seed": 0} and out[wc.id]["robot"].world is None
    assert hash(out[wc.id]["robot"]) == hash(out[r.id]["robot"]) and out[wc.id]["robot"] == out[r.id]["robot"]


def test_robot_from_file_is_weighed_headless():
    from simulatics import robots
    if not robots.available("go2"):
        pytest.skip("go2 not downloaded (python -m simulatics.robots fetch go2)")
    g = Graph()
    f = g.add_node("robot.file", params={"path": str(robots.path("go2"))})
    acc = g.add_node("robot.accessory", params={"type": "payload", "mass": 2.0})
    m0, m1 = g.add_node("design.mass_report"), g.add_node("design.mass_report")
    g.connect(f.id, "robot", acc.id, "robot")
    g.connect(f.id, "robot", m0.id, "robot")
    g.connect(acc.id, "robot", m1.id, "robot")
    ex, ev = Executor(), []
    out = ex.run(g, on_event=record(ev))
    assert ev[-1][2]["ok"], [p for k, n, p in ev if k == "error"]
    assert out[f.id]["robot"].kind == "file" and ex.session is None          # no simulation was started
    assert 5.0 < out[m0.id]["total"] < 30.0 and len(out[m0.id]["table"]) == 13
    # the payload compiles onto the root body; go2's bodies carry explicit inertials, so MuJoCo does not add
    # the accessory geom's mass there (same in the live session) - only check that it mounts cleanly
    assert out[acc.id]["robot"].accessories[0]["body"] == "" and out[m1.id]["total"] >= out[m0.id]["total"]
    missing = Graph()
    missing.add_node("robot.file", params={"path": "no/such/robot.xml"})
    ev.clear()
    ex.run(missing, on_event=record(ev))
    assert "no such robot file" in next(p["message"] for k, n, p in ev if k == "error")
