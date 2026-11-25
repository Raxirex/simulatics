"""Simulatics Studio (the Qt desktop app): the window, viewport tools, panels and the node-graph editor.

Needs PySide6 and a real display with OpenGL; skipped otherwise.
"""
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def spin(app, cond=lambda: False, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return cond()


@pytest.fixture(scope="module")
def app():
    from simulatics.studio import theme
    from simulatics.studio.viewport import configure_gl
    existing = QApplication.instance()
    if existing is None:
        configure_gl()
    a = existing or QApplication(["test"])
    theme.apply(a)
    return a


@pytest.fixture(scope="module")
def win(app):
    from simulatics.studio.main_window import StudioWindow
    w = StudioWindow(restore=False)
    w.resize(1400, 900)
    w.show()
    if not spin(app, lambda: w.viewport.scn is not None and w.viewport._frames + w.viewport._fps > 0, 15):
        w.close()
        pytest.skip("no OpenGL context for the viewport on this machine")
    w.viewport.set_camera("Free")
    yield w
    w._settings = lambda: __import__("PySide6.QtCore", fromlist=["QSettings"]).QSettings("Simulatics", "StudioTests")
    w.close()
    spin(app, timeout=0.2)


def settle(app, win):
    win.sess.running = False
    win.reset_sim()
    win.viewport.set_camera("Free")
    win.viewport.set_view("Perspective")
    win.viewport.frame_robot()
    spin(app, timeout=0.25)       # a couple of frames, so picking uses the current camera


def pixel_of(win, point):
    px, ok = win.viewport.project(np.array([point], float))
    assert ok[0], "point is outside the view"
    return float(px[0, 0]), float(px[0, 1])


def test_viewport_draws_the_robot(app, win):
    settle(app, win)
    img = win.viewport.grabFramebuffer()
    assert img.width() > 200 and img.height() > 100
    colours = {img.pixel(x, y) for x in range(0, img.width(), 37) for y in range(0, img.height(), 29)}
    assert len(colours) > 20                      # a rendered scene, not a blank widget
    assert win.viewport._frames + win.viewport._fps > 0     # frames are being drawn


def test_click_selects_a_body_and_fills_the_inspector(app, win):
    settle(app, win)
    with win.sess.lock:
        torso = win.sess.data.xpos[1].copy()
    x, y = pixel_of(win, torso)
    body, point, geom = win.viewport.pick(x, y)
    assert body > 0 and geom >= 0 and np.linalg.norm(point - torso) < 0.4
    win.viewport.set_tool("select")
    win.viewport._click(x, y, Qt.NoModifier)
    spin(app, timeout=0.05)
    assert win.viewport.selected == body and win.inspector.kind == "body" and win.inspector.ident == body
    assert win.outliner.tree.selectedItems()[0].text(0) == win.sess.model.body(body).name
    win.viewport._click(5, 5, Qt.NoModifier)       # empty space clears the selection
    assert win.viewport.selected == -1 and win.inspector.kind == "scene"


def test_goal_tool_sets_a_walking_goal_on_the_floor(app, win):
    settle(app, win)
    win.viewport.set_tool("goal")
    x, y = pixel_of(win, [1.5, 0.6, 0.0])
    win.viewport._click(x, y, Qt.NoModifier)
    goals = [g for g, _ in win.sess.nav.goals]
    assert len(goals) == 1 and np.linalg.norm(goals[0] - [1.5, 0.6]) < 0.08 and win.sess.running
    x2, y2 = pixel_of(win, [0.4, 1.4, 0.0])
    win.viewport._click(x2, y2, Qt.ShiftModifier)  # shift adds a waypoint
    assert len(win.sess.nav.goals) == 2
    win.sess.running = False
    win.sess.nav.clear()


def test_place_tool_mounts_an_accessory_where_clicked(app, win):
    settle(app, win)
    before = len(win.sess.accessories or [])
    win.arm_place("camera")
    assert win.viewport.tool == "place" and not win.sess.running
    with win.sess.lock:
        torso = win.sess.data.xpos[1].copy()
    x, y = pixel_of(win, torso)
    win.viewport._click(x, y, Qt.NoModifier)
    spin(app, lambda: win._version == win.sess.scene_version, 3)
    acc = win.sess.accessories
    assert len(acc) == before + 1 and acc[-1]["type"] == "camera"
    assert f"cam_{acc[-1]['id']}" in win.viewport.camera_names()
    assert win.assets.acc.count() == len(acc)       # the Assets list followed the rebuild
    win.sess.delete_accessory(acc[-1]["id"])
    spin(app, timeout=0.2)
    assert len(win.sess.accessories) == before


def test_measure_and_push_tools(app, win):
    settle(app, win)
    win.viewport.set_tool("measure")
    for pt in ([1.0, 0.0, 0.0], [1.0, 1.0, 0.0]):
        win.viewport._click(*pixel_of(win, pt), Qt.NoModifier)
    assert len(win.viewport.measure_pts) == 2
    assert abs(np.linalg.norm(win.viewport.measure_pts[1] - win.viewport.measure_pts[0]) - 1.0) < 0.05
    win.viewport.set_tool("push")
    with win.sess.lock:
        torso = win.sess.data.xpos[1].copy()
    win.viewport._click(*pixel_of(win, torso), Qt.NoModifier)
    assert len(win.viewport._pushes) == 1 and win.sess.running
    assert spin(app, lambda: not win.viewport._pushes, 5)           # the impulse ends by itself
    with win.sess.lock:
        assert not win.sess.data.xfrc_applied.any()
    win.sess.running = False


def test_workspaces_rearrange_docks_and_restore(app, win):
    from simulatics.studio.main_window import WORKSPACES
    for name in WORKSPACES:
        win.apply_workspace(name)
        spin(app, timeout=0.05)
        assert win.docks["viewport"].isVisible() and win.workspace == name
    win.apply_workspace("Graph")
    assert win.docks["graph"].isVisible() and not win.docks["outliner"].isVisible()
    state = win.saveState()
    win.apply_workspace("Layout")
    assert win.docks["outliner"].isVisible()
    assert win.restoreState(state)
    spin(app, timeout=0.05)
    assert win.docks["graph"].isVisible() and not win.docks["outliner"].isVisible()
    win.apply_workspace("Layout")


def test_timeline_records_and_scrubs_back(app, win):
    settle(app, win)
    win.sess.speed = 4.0
    win.toggle_play()
    assert spin(app, lambda: win.sess.data.time > 1.0, 20)
    win.toggle_play()
    spin(app, timeout=0.1)
    rec = win.timeline.rec
    assert len(rec.frames) > 5
    t_end = float(win.sess.data.time)
    rec.restore(len(rec.frames) // 2)
    assert not win.sess.running and 0.0 < float(win.sess.data.time) < t_end
    assert abs(float(win.sess.data.time) - rec.frames[len(rec.frames) // 2][0]) < 1e-9
    win.sess.speed = 1.0


def test_inspector_edits_reach_the_live_model(app, win):
    settle(app, win)
    win.select("actuator", 0)
    from PySide6.QtWidgets import QDoubleSpinBox
    spins = win.inspector.holder.findChildren(QDoubleSpinBox)
    assert spins
    kp0 = float(win.sess.model.actuator_gainprm[0, 0])
    kp_spin = next(s for s in spins if abs(s.value() - kp0) < 1e-6)
    kp_spin.setValue(kp0 + 11.0)
    assert abs(float(win.sess.model.actuator_gainprm[0, 0]) - (kp0 + 11.0)) < 1e-6
    kp_spin.setValue(kp0)
    win.motors.rebuild()
    assert win.motors.table.rowCount() == win.sess.model.nu


def test_robot_parameter_rebuilds_and_panels_follow(app, win):
    settle(app, win)
    v0 = win.sess.scene_version
    old = float(win.sess.params.thigh_length)
    win.robot_panel._set_param("thigh_length", old + 0.03)
    spin(app, timeout=0.1)
    assert win.sess.scene_version > v0 and abs(win.sess.params.thigh_length - (old + 0.03)) < 1e-9
    assert win._version == win.sess.scene_version
    assert abs(win.robot_panel.sliders["thigh_length"].value() - (old + 0.03)) < 1e-6
    win.robot_panel._set_param("thigh_length", old)
    spin(app, timeout=0.1)
    assert win.viewport.grabFramebuffer().width() > 0       # the viewport survived the model swap


def test_console_and_script_share_the_session(app, win):
    c = win.console
    c.line.setText("falls_seen = session.falls + 40")
    c._enter()
    assert c.ns["falls_seen"] == win.sess.falls + 40
    c.line.setText("print('hello', 6 * 7)")
    c._enter()
    assert "hello 42" in c.out.toPlainText()
    c.run_script("made_by_script = int(session.model.nu)\nprint('script ran')")
    assert spin(app, lambda: "made_by_script" in c.ns and c.run_btn.isEnabled(), 10)
    assert c.ns["made_by_script"] == win.sess.model.nu and "script ran" in c.out.toPlainText()
    c.run_script("raise ValueError('boom')")
    assert spin(app, lambda: "ValueError: boom" in c.out.toPlainText(), 10)


def test_graph_editor_builds_wires_and_runs(app, win):
    from simulatics import graph as G
    from simulatics.studio.graph_editor import EdgeItem, NodeItem
    gp = win.graph_panel
    gp.new()
    gp.add_node("value.number", (0, 0))
    gp.add_node("script.python", (300, 0))
    gp.add_node("out.print", (600, 0))
    a, b, c = list(gp.graph.nodes.values())
    a.params["value"] = 5.0
    b.params["code"] = "out = a * 3"
    items = gp.scene.node_items
    # wire by dragging between sockets, as a user does
    src = items[a.id].outputs[next(iter(items[a.id].outputs))]
    dst = items[b.id].inputs["a"]
    gp.scene.begin_wire(src)
    gp.scene.move_wire(dst.scenePos())
    gp.scene.end_wire(dst.scenePos())
    out_b = items[b.id].outputs["out"]
    dst_c = items[c.id].inputs[next(iter(items[c.id].inputs))]
    gp.scene.begin_wire(out_b)
    gp.scene.end_wire(dst_c.scenePos())
    assert len(gp.graph.links) == 2 and sum(isinstance(i, EdgeItem) for i in gp.scene.items()) == 2
    gp.run()
    assert spin(app, lambda: not gp.running() and gp.run_btn.isEnabled(), 30)
    spin(app, timeout=0.1)
    assert all(items[n].state in ("done", "cached") for n in (a.id, b.id, c.id)), [items[n].message for n in items]
    assert win.results.tabs.count() >= 1
    text = win.results.tabs.currentWidget().toPlainText()
    assert "15" in text
    gp.run()                                        # nothing changed: everything comes from the cache
    assert spin(app, lambda: not gp.running() and gp.run_btn.isEnabled(), 30)
    spin(app, timeout=0.1)
    assert items[a.id].state == "cached"            # (script nodes always re-run: they may touch the session)
    # an error is shown on the node and the run stops
    b.params["code"] = "out = undefined_name"
    gp.run()
    assert spin(app, lambda: not gp.running() and gp.run_btn.isEnabled(), 30)
    spin(app, timeout=0.1)
    assert gp.scene.node_items[b.id].state == "error"
    # deleting a node removes its wires
    gp.scene.node_items[b.id].setSelected(True)
    gp.scene.delete_selected()
    assert b.id not in gp.graph.nodes and not gp.graph.links
    assert sum(isinstance(i, NodeItem) for i in gp.scene.items()) == 2
    # every example loads into the editor
    for name in G.EXAMPLES:
        gp.load_example(name)
        assert len(gp.scene.node_items) == len(gp.graph.nodes) > 1
    gp.new()


def test_graph_drives_the_live_simulation(app, win):
    from simulatics import graph as G
    settle(app, win)
    gp = win.graph_panel
    gp.new()
    g = G.Graph()
    r = g.add_node("robot.builtin", (0, 0))
    ap = g.add_node("sim.apply", (300, 0))
    run = g.add_node("sim.run", (600, 0), params={"seconds": 1.0, "speed": 8.0})
    plot = g.add_node("out.plot", (900, 0))
    g.connect(r.id, "robot", ap.id, "robot")
    g.connect(ap.id, "robot", run.id, "robot")
    series = next(p.name for p in G.REGISTRY["sim.run"].outputs if p.type == "series")
    g.connect(run.id, series, plot.id, next(p.name for p in G.REGISTRY["out.plot"].inputs))
    gp.set_graph(g)
    t0 = float(win.sess.data.time)
    gp.run()
    assert spin(app, lambda: not gp.running() and gp.run_btn.isEnabled(), 60)
    spin(app, timeout=0.1)
    assert gp.scene.node_items[run.id].state == "done", gp.scene.node_items[run.id].message
    assert float(win.sess.data.time) != t0
    from simulatics.studio.widgets import PlotWidget
    assert any(isinstance(win.results.tabs.widget(i), PlotWidget) for i in range(win.results.tabs.count()))
    gp.new()
    win.sess.running = False


def test_typing_in_text_fields_does_not_trigger_shortcuts(app, win):
    """Space, R, F, G and . are window shortcuts; typed into a text field they must be text."""
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QPlainTextEdit
    settle(app, win)
    win.activateWindow()
    win.viewport.set_tool("select")
    win.show_dock("console")
    typed = "r f g . x"
    for w in (win.console.line,):
        win.console.tabs.setCurrentIndex(0)
        w.setFocus()
        spin(app, timeout=0.05)
        w.clear()
        QTest.keyClicks(w, typed)
        assert w.text() == typed
    win.console.tabs.setCurrentIndex(1)
    ed = win.console.editor
    ed.setFocus()
    spin(app, timeout=0.05)
    ed.clear()
    QTest.keyClicks(ed, typed)
    assert ed.toPlainText() == typed
    # a code box on a graph node (a widget embedded in the canvas)
    win.apply_workspace("Graph")
    gp = win.graph_panel
    gp.new()
    gp.add_node("script.python", (0, 0))
    spin(app, timeout=0.2)
    item = next(iter(gp.scene.node_items.values()))
    box = item.proxy.widget().findChild(QPlainTextEdit)
    gp.view.setFocus()
    gp.scene.setFocusItem(item.proxy)
    box.setFocus()
    spin(app, timeout=0.05)
    box.clear()
    QTest.keyClicks(box, typed)
    assert box.toPlainText() == typed and item.inst.params["code"] == typed
    assert not win.sess.running and win.viewport.tool == "select" and float(win.sess.data.time) == 0.0
    gp.new()
    win.apply_workspace("Layout")


def test_keys_on_the_viewport_and_graph_canvas(app, win):
    from PySide6.QtTest import QTest
    settle(app, win)
    win.activateWindow()
    win.viewport.setFocus()
    spin(app, timeout=0.05)
    QTest.keyClick(win.viewport, Qt.Key_G)
    assert win.viewport.tool == "goal"
    QTest.keyClick(win.viewport, Qt.Key_Q)
    assert win.viewport.tool == "select"
    QTest.keyClick(win.viewport, Qt.Key_Space)
    assert win.sess.running
    QTest.keyClick(win.viewport, Qt.Key_Space)
    assert not win.sess.running
    win.apply_workspace("Graph")
    gp = win.graph_panel
    gp.load_example("Script")
    spin(app, timeout=0.3)
    gp.view.setFocus()
    spin(app, timeout=0.05)
    win.viewport.cam.distance = 7.7
    QTest.keyClick(gp.view, Qt.Key_F)                 # fits the graph, does not re-frame the robot
    assert abs(win.viewport.cam.distance - 7.7) < 1e-9
    gp.view.setFocus()
    spin(app, timeout=0.05)
    QTest.keyClick(gp.view, Qt.Key_Tab)               # opens the node search
    assert spin(app, lambda: gp.view.search.isVisible(), 2)
    QTest.keyClicks(gp.view.search.edit, "plot")
    QTest.keyClick(gp.view.search.edit, Qt.Key_Return)
    spin(app, timeout=0.1)
    assert any(n.type == "out.plot" for n in gp.graph.nodes.values()) and not gp.view.search.isVisible()
    n = len(gp.graph.nodes)
    for it in gp.scene.node_items.values():
        it.setSelected(it.inst.type == "out.plot")
    gp.view.setFocus()
    QTest.keyClick(gp.view, Qt.Key_Delete)
    assert len(gp.graph.nodes) == n - 1
    gp.new()
    win.apply_workspace("Layout")
    win.viewport.frame_robot()


def test_tool_options_bar_follows_the_tool(app, win):
    win.viewport.set_tool("push")
    assert win.opt_stack.currentIndex() == 1
    win.push_force.setValue(240)
    assert win.viewport.push_force == 240
    win.arm_place("payload")
    assert win.opt_stack.currentIndex() == 2 and win.place_type.currentData() == "payload"
    win.viewport.set_tool("select")
    assert win.opt_stack.currentIndex() == 0


def test_mouse_orbit_pan_and_zoom_move_the_camera(app, win):
    """Real mouse and wheel events on the viewport (these call MuJoCo's camera functions)."""
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QMouseEvent, QWheelEvent
    from PySide6.QtCore import QEvent
    settle(app, win)
    v = win.viewport
    c = QPointF(v.width() / 2, v.height() / 2)

    def mouse(kind, pos, button, buttons, mods=Qt.NoModifier):
        return QMouseEvent(kind, pos, v.mapToGlobal(pos), button, buttons, mods)

    d0 = v.cam.distance
    v.wheelEvent(QWheelEvent(c, v.mapToGlobal(c), QPoint(0, 0), QPoint(0, 120), Qt.NoButton, Qt.NoModifier,
                             Qt.NoScrollPhase, False))
    assert v.cam.distance < d0                               # wheel up zooms in
    az, el = v.cam.azimuth, v.cam.elevation
    v.mousePressEvent(mouse(QEvent.MouseButtonPress, c, Qt.RightButton, Qt.RightButton))
    v.mouseMoveEvent(mouse(QEvent.MouseMove, c + QPointF(60, 25), Qt.NoButton, Qt.RightButton))
    v.mouseReleaseEvent(mouse(QEvent.MouseButtonRelease, c + QPointF(60, 25), Qt.RightButton, Qt.NoButton))
    assert abs(v.cam.azimuth - az) > 1 and abs(v.cam.elevation - el) > 0.5      # right-drag orbits
    look = v.cam.lookat.copy()
    v.mousePressEvent(mouse(QEvent.MouseButtonPress, c, Qt.MiddleButton, Qt.MiddleButton))
    v.mouseMoveEvent(mouse(QEvent.MouseMove, c + QPointF(40, 30), Qt.NoButton, Qt.MiddleButton))
    v.mouseReleaseEvent(mouse(QEvent.MouseButtonRelease, c + QPointF(40, 30), Qt.MiddleButton, Qt.NoButton))
    assert np.linalg.norm(v.cam.lookat - look) > 0.01        # middle-drag pans
    # grab tool: dragging a body pulls it with a spring force
    v.set_tool("grab")
    with win.sess.lock:
        torso = win.sess.data.xpos[1].copy()
    v.frame_robot()
    spin(app, timeout=0.25)
    x, y = pixel_of(win, torso)
    p = QPointF(x, y)
    v.mousePressEvent(mouse(QEvent.MouseButtonPress, p, Qt.LeftButton, Qt.LeftButton))
    assert v._grab_body > 0
    v.mouseMoveEvent(mouse(QEvent.MouseMove, p + QPointF(30, 0), Qt.NoButton, Qt.LeftButton))
    spin(app, timeout=0.2)
    with win.sess.lock:
        assert np.abs(win.sess.data.xfrc_applied).max() > 0
    v.mouseReleaseEvent(mouse(QEvent.MouseButtonRelease, p + QPointF(30, 0), Qt.LeftButton, Qt.NoButton))
    assert v._grab_body == -1
    with win.sess.lock:
        assert not win.sess.data.xfrc_applied.any()
    win.sess.running = False
    v.set_tool("select")


def test_motor_control_panel_drives_one_motor_on_the_stand(app, win):
    settle(app, win)
    win.apply_workspace("Motors")
    spin(app, timeout=0.2)
    mp = win.motor
    assert mp.isVisible() and mp.motor.count() == win.sess.model.nu
    mp.stand.setChecked(True)
    spin(app, lambda: win.sess.support_stand and win.sess.controller.name == "manual", 5)
    assert win.sess.controller.name == "manual"
    mp.motor.setCurrentText("left_knee")
    mp._select()
    mp.target.spin.setValue(35.0)                       # degrees, typed exactly
    i = mp._i()
    assert abs(win.sess.manual_ctrl[i] - np.deg2rad(35.0)) < 1e-6
    mp.kind.setCurrentIndex(mp.kind.findData("step"))
    mp.amp.setValue(20.0)
    win.sess.speed = 8.0
    mp.run.click()
    t0 = float(win.sess.data.time)
    assert spin(app, lambda: win.sess.data.time - t0 > 1.5 and len(win.sess.motor_trace) > 40, 20)
    mp.refresh()
    assert "rise" in mp.metrics.text() and "peak torque" in mp.metrics.text()
    assert mp.scope_angle.series is not None and len(mp.scope_angle.series["y"]["angle"]) > 40
    kp0 = mp.gains["kp"].value()
    mp.gains["kp"].setValue(kp0 + 50)
    assert abs(win.sess.actuators()[i]["kp"] - (kp0 + 50)) < 1e-6
    mp.gains["kp"].setValue(kp0)
    mp.stop.click()
    mp.stand.setChecked(False)
    spin(app, lambda: win.sess.controller.name == "walker", 5)
    assert win.sess.controller.name == "walker" and not win.sess.support_stand
    win.sess.speed = 1.0
    win.apply_workspace("Layout")


def test_world_model_panel_runs_the_stand_in_on_an_instruction(app, win):
    settle(app, win)
    wm = win.world_model
    win.show_dock("worldmodel")
    assert "stand-in" in wm.model_name.text() and "Placeholder" in wm.findChildren(type(wm.state))[0].text()
    wm.text.setText("walk to 1 0.5")
    wm.send.click()
    assert spin(app, lambda: len(win.sess.nav.goals) == 1, 5)
    assert np.allclose(win.sess.nav.goals[0][0], [1.0, 0.5])
    wm.refresh()
    assert "walking to (1, 0.5)" in wm.log.toPlainText()
    wm.instruct("stop")
    assert spin(app, lambda: not win.sess.nav.goals, 5)
    wm.stop.click()
    assert not wm.runner.running
    win.sess.running = False


def test_training_panel_offers_skill_training(app, win):
    tp = win.train
    tp.algo.setCurrentIndex(tp.algo.findData("skill"))
    tp._preset()
    assert tp.motion.isEnabled() and tp.motion.text() == "uppercut" and tp.motion_label.text() == "Skill"
    tp.algo.setCurrentIndex(tp.algo.findData("ppo"))
    tp._preset()
    assert tp.motion.text() == "walk" and not tp.motion.isEnabled()


def test_samples_menu_runs_a_world_and_a_physical_motion(app, win):
    from simulatics import samples
    win.run_sample("kitchen")
    spin(app, timeout=0.3)
    assert len(win.sess.props) >= 5 and win.assets.acc.count() >= 1
    win.sess.running = False
    if "vertical_jump" in samples.physical():
        win.run_sample("vertical_jump")
        assert win.sess.controller.name == "motion-physical"
        win.sess.speed = 8.0
        assert spin(app, lambda: win.sess.controller.name == "walker", 60)
        win.sess.speed = 1.0
    win.sess.running = False
    win.run_sample("stand")
    win.sess.running = False
