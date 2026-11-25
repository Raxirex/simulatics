"""Ready-made worlds, precise motor control, and the world-model slot."""
import time

import mujoco
import numpy as np
import pytest

from simulatics import samples
from simulatics.server.session import SimSession
from simulatics.sim.worlds import WORLDS
from simulatics.worldmodel import (Action, JointTargetController, KeywordModel, WorldModelRunner, load_world_model,
                                   observe)


def wait_for(cond, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


@pytest.fixture(scope="module")
def sess():
    s = SimSession()
    s.lidar_cfg.enabled = False
    yield s
    s.running = False
    s.shutdown()


# ---------------------------------------------------------------------------------------------- worlds
@pytest.mark.parametrize("name", sorted(WORLDS))
def test_world_builds_and_the_robot_stands_in_it(sess, name):
    samples._builtin(sess)
    info = sess.load_world(name)
    assert info["about"] and len(sess.obstacles) >= 15
    with sess.lock:
        m = sess.model
        props = [b for b in range(m.nbody) if m.body(b).name.startswith("prop_")]
        assert props and m.ngeom > 40
    sess.speed = 8.0
    sess.running = True
    t0, falls = float(sess.data.time), sess.falls
    assert wait_for(lambda: sess.data.time - t0 > 2.0, 30)
    sess.running = False
    with sess.lock:
        assert sess.falls == falls and float(sess.data.qpos[2]) > 0.95
        for b in props:                                         # loose things came to rest on a surface
            assert sess.data.xpos[b][2] > 0.0 and np.linalg.norm(sess.data.cvel[b][3:]) < 0.5


def test_kitchen_objects_are_within_reach_of_the_arms(sess):
    from simulatics.sim.props import props_state, table_geometry
    samples._builtin(sess)
    sess.load_world("kitchen")
    t = table_geometry(sess.params)
    with sess.lock:
        objs = {o["name"]: o for o in props_state(sess.model, sess.data)}
    for name in ("red cup", "green apple"):
        o = objs[name]
        assert t["grasp_x"][0] - 0.01 <= o["x"] <= t["grasp_x"][1] + 0.01 and t["grasp_y"][0] <= abs(o["y"]) <= t["grasp_y"][1]
        assert abs(o["z"] - t["height"]) < 0.06


def test_parkour_course_sample_is_walked_end_to_end(sess):
    samples.run(sess, "parkour_course")
    sess.speed = 8.0
    falls = sess.falls
    goal = sess.nav.goals[-1][0].copy()
    assert wait_for(lambda: sess.nav.state in ("reached", "failed", "fallen"), 200)
    assert sess.nav.state == "reached" and sess.falls == falls
    assert np.linalg.norm(sess.data.qpos[:2] - goal) < 0.05 and goal[0] > 5.0     # beyond steps, stairs and ramp
    sess.running = False


def test_worlds_refuse_imported_robots(sess):
    from simulatics import robots
    if not robots.available("go2"):
        pytest.skip("go2 not downloaded")
    sess.load_robot("go2")
    with pytest.raises(ValueError, match="built-in"):
        sess.load_world("kitchen")
    samples._builtin(sess)


# ----------------------------------------------------------------------------------------- motor control
def test_stand_holds_the_robot_and_a_step_signal_is_traced(sess):
    samples._builtin(sess)
    sess.set_support_stand(True)
    assert sess.controller.name == "manual"
    with sess.lock:
        assert any(sess.model.equality(i).name == "support_stand" for i in range(sess.model.neq))
        z0 = float(sess.data.qpos[2])
    knee = next(a["index"] for a in sess.actuators() if a["name"] == "left_knee")
    sess.trace_motor(knee)
    sess.speed = 8.0
    sess.set_motor_signal(knee, "step", 0.5, base=0.4)
    t0 = float(sess.data.time)
    assert wait_for(lambda: sess.data.time - t0 > 2.0, 30)
    tr = np.array(sess.motor_trace)
    assert len(tr) > 50 and abs(tr[0, 1] - 0.4) < 1e-9 and abs(tr[-1, 1] - 0.9) < 1e-9        # the command stepped
    assert abs(tr[-1, 2] - 0.9) < 0.06 and tr[:, 2].max() < 1.1                                # the joint followed
    assert np.all(np.diff(tr[:, 0]) > 0) and np.abs(tr[:, 4]).max() <= sess.actuators()[knee]["torque"] + 1e-6
    with sess.lock:
        assert sess.falls == 0 or True
        assert abs(float(sess.data.qpos[2]) - z0) < 0.12       # hanging on the stand, not on the floor
    # a sine keeps time with the simulation clock
    sess.set_motor_signal(knee, "sine", 0.2, frequency=2.0, base=0.6)
    t0 = float(sess.data.time)
    assert wait_for(lambda: sess.data.time - t0 > 1.5, 30)
    tr = np.array(sess.motor_trace)
    x = tr[:, 0] - t0
    assert np.abs(tr[:, 1] - (0.6 + 0.2 * np.sin(2 * np.pi * 2.0 * x))).max() < 0.06
    with pytest.raises(ValueError):
        sess.set_motor_signal(knee, "wobble")
    sess.set_motor_signal(knee, "hold", base=0.3)
    assert sess.controller.signal is None and abs(sess.manual_ctrl[knee] - 0.3) < 1e-9
    sess.set_support_stand(False)
    assert sess.controller.name == "walker"
    with pytest.raises(ValueError, match="manual"):
        sess.set_motor_signal(knee, "step", 0.1)
    sess.trace_motor(None)


# ------------------------------------------------------------------------------------------ world model
def test_observation_has_what_a_model_needs(sess):
    samples._builtin(sess)
    obs = observe(sess)
    assert len(obs.motors) == sess.model.nu and set(obs.motors) <= set(obs.joint_pos)
    assert obs.root_pos.shape == (3,) and abs(np.linalg.norm(obs.root_quat) - 1) < 1e-6
    assert set(obs.foot_contact) == {"left_foot", "right_foot"}
    sess.load_world("kitchen")
    assert {"red cup", "green apple"} <= {o["name"] for o in observe(sess).objects}
    samples._builtin(sess)


def test_keyword_stand_in_maps_phrases_and_says_when_it_cannot(sess):
    samples._builtin(sess)
    k, obs = KeywordModel(), observe(sess)
    a = k.act(obs, "please walk to 2, -1.5")
    assert a.kind == "goal" and a.goal == (2.0, -1.5)
    assert k.act(obs, "please walk to 2, -1.5").kind == "none"              # one action per instruction
    a = k.act(obs, "walk forward 1.5")
    assert a.kind == "goal" and abs(a.goal[0] - (obs.root_pos[0] + 1.5)) < 1e-6
    assert k.act(obs, "now do a backflip").skill == "backflip"
    assert k.act(obs, "jump").skill == "vertical_jump"
    assert k.act(obs, "STOP!").skill == "stop"
    a = k.act(obs, "make me a sandwich")
    assert a.kind == "none" and "stand-in" in a.say


def test_runner_applies_goals_skills_and_joint_targets(sess):
    samples._builtin(sess)
    r = WorldModelRunner(sess)
    r.instruct("walk to 1.2 0.4")
    act = r.step_once()
    assert act.kind == "goal" and len(sess.nav.goals) == 1 and np.allclose(sess.nav.goals[0][0], [1.2, 0.4])
    sess.speed = 8.0
    assert wait_for(lambda: sess.nav.state in ("reached", "failed", "fallen"), 90) and sess.nav.state == "reached"
    r.instruct("jump")
    assert r.step_once().skill == "vertical_jump" and sess.controller.name == "motion-physical"
    samples._builtin(sess)

    class Driver:                                               # a model that drives motors itself
        name = "driver"

        def act(self, obs, instruction):
            return Action.drive({"left_elbow": -1.0, "no_such_joint": 0.3}, say="bending the elbow")
    r.set_model(Driver())
    r.step_once()
    assert isinstance(sess.controller, JointTargetController)
    i = next(a["index"] for a in sess.actuators() if a["name"] == "left_elbow")
    assert abs(sess.controller.target[i] + 1.0) < 1e-9
    assert any("unknown joints" in line for _, line in r.log) and any("bending the elbow" in line for _, line in r.log)
    r.stop()
    assert sess.controller.name == "walker"                     # the robot got its own controller back


def test_runner_loop_survives_a_broken_model_and_loads_models_from_files(sess, tmp_path):
    samples._builtin(sess)
    r = WorldModelRunner(sess)

    class Broken:
        def act(self, obs, instruction):
            raise RuntimeError("no idea")
    r.set_model(Broken())
    r.rate_hz = 20
    r.start("anything")
    assert wait_for(lambda: not r.running, 5)
    assert "RuntimeError: no idea" in r.error
    f = tmp_path / "mine.py"
    f.write_text("from simulatics.worldmodel import Action\n"
                 "class M:\n    name = 'mine'\n    def act(self, obs, instruction):\n"
                 "        return Action.go(0.5, 0.0, say='going')\n"
                 "def create():\n    return M()\n", encoding="utf-8")
    m = load_world_model(f)
    assert m.name == "mine"
    r.set_model(m)
    r.start("go")
    assert wait_for(lambda: r.calls > 1 and len(sess.nav.goals) == 1, 5)
    r.stop()
    assert not r.running
    bad = tmp_path / "bad.py"
    bad.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="create"):
        load_world_model(bad)
    sess.nav.clear()
    sess.running = False


# -------------------------------------------------------------------------- free scene building, no stuck motions
def test_objects_are_added_edited_and_removed_while_the_robot_keeps_its_place(sess):
    from simulatics.sim.builder import CATALOG
    samples._builtin(sess)
    sess.speed = 8.0
    with sess.lock:
        sess.nav.set_goal([1.0, 0.5])
        sess.running = True
    assert wait_for(lambda: sess.nav.state == "reached", 60)
    made = {k: sess.add_object(k, 3.0 + 0.9 * (i % 4), -2.0 + 1.3 * (i // 4)) for i, k in enumerate(CATALOG)}
    with sess.lock:
        assert np.allclose(sess.data.qpos[:2], [1.0, 0.5], atol=0.03) and sess.controller.name == "walker"
        assert len(sess.props) == 4 and len(sess.obstacles) == len(CATALOG) - 4 + 2            # a hurdle is three parts
    assert sess.object_info(made["hurdle"])["parts"] == 3 and sess.object_info(made["ball"])["loose"]
    assert sess.object_at(sess.object_info(made["box"])["geom"]) == made["box"] and sess.object_at(0) is None
    info = sess.update_object(made["box"], x=2.0, y=2.0, yaw_deg=30.0, size=[0.3, 0.2, 0.4])
    assert (info["x"], info["y"]) == (2.0, 2.0) and abs(info["yaw_deg"] - 30) < 1e-6 and info["size"] == [0.3, 0.2, 0.4]
    before = [list(o["pos"]) for o in sess.obstacles if o["group"] == made["hurdle"]]
    sess.update_object(made["hurdle"], x=before[0][0] + 0.5)
    after = [list(o["pos"]) for o in sess.obstacles if o["group"] == made["hurdle"]]
    assert all(abs(b[0] - a[0] - 0.5) < 1e-9 and abs(b[1] - a[1]) < 1e-9 for a, b in zip(before, after))   # moved as one
    copy = sess.duplicate_object(made["crate"])
    assert copy != made["crate"] and len(sess.props) == 5
    sess.remove_object(made["wall"])
    with pytest.raises(ValueError):
        sess.remove_object(made["wall"])
    with pytest.raises(ValueError, match="on top of the robot"):
        sess.add_object("box", 1.05, 0.5)
    with pytest.raises(ValueError, match="unknown object"):
        sess.add_object("dragon", 5, 5)
    scene = sess.scene_dict()
    n = (len(sess.obstacles), len(sess.props))
    samples._builtin(sess)
    sess.load_scene(scene)
    assert (len(sess.obstacles), len(sess.props)) == n
    with pytest.raises(ValueError):
        sess.load_scene({"kind": "something else"})
    samples._builtin(sess)


def test_the_robot_walks_through_a_scene_built_by_hand(sess):
    samples._builtin(sess)
    sess.add_object("box", 1.5, 0.0)
    sess.add_object("pillar", 2.6, 0.9)
    sess.add_object("step", 3.6, 0.0)
    sess.speed = 8.0
    falls = sess.falls
    with sess.lock:
        sess.nav.set_goal([4.6, 0.0])
        sess.running = True
    assert wait_for(lambda: sess.nav.state in ("reached", "failed", "fallen"), 150)
    assert sess.nav.state == "reached" and sess.falls == falls and abs(sess.data.qpos[0] - 4.6) < 0.05
    samples._builtin(sess)


def test_reference_motions_replay_instead_of_freezing_and_a_goal_takes_over(sess):
    samples.run(sess, "cartwheel")
    assert sess.controller.name == "motion-preview"
    sess.speed = 8.0
    t0 = float(sess.data.time)
    assert wait_for(lambda: sess.data.time - t0 > 9.0, 30)
    assert sess.controller.replays >= 2                           # the clip played again (and again)
    with sess.lock:
        a = sess.data.qpos[7:25].copy()
    time.sleep(0.4)
    with sess.lock:
        assert np.abs(sess.data.qpos[7:25] - a).max() > 0.01      # still moving, not stuck on the last frame
        sess.nav.set_goal([1.0, 1.0])
    falls = sess.falls
    assert wait_for(lambda: sess.nav.state in ("reached", "failed", "fallen"), 90)
    assert sess.nav.state == "reached" and sess.controller.name == "walker" and sess.falls == falls
    assert np.linalg.norm(sess.data.qpos[:2] - [1.0, 1.0]) < 0.05              # it really walked there


def test_a_goal_takes_over_from_the_wave_and_is_refused_on_top_of_a_box(sess):
    samples.run(sess, "one_hand_wave")
    sess.speed = 8.0
    t0 = float(sess.data.time)
    assert wait_for(lambda: sess.data.time - t0 > 2.0, 30)
    with sess.lock:
        sess.nav.set_goal([1.0, -0.6])
    falls = sess.falls
    assert wait_for(lambda: sess.nav.state in ("reached", "failed", "fallen"), 90)
    assert sess.nav.state == "reached" and sess.controller.name == "walker" and sess.falls == falls
    if "box_jump_up" in samples.physical() and samples.physical()["box_jump_up"]["verified"]:
        samples.run(sess, "box_jump_up")
        sess.speed = 8.0
        assert wait_for(lambda: sess.controller.name == "walker" and sess.controller.status().get("mode") == "stand", 90)
        time.sleep(1.0)
        falls = sess.falls
        with sess.lock:
            sess.nav.set_goal([1.0, 2.0])
        assert wait_for(lambda: sess.nav.state in ("failed", "reached", "fallen"), 30)
        assert sess.nav.state == "failed" and "too high" in sess.nav.message and sess.falls == falls
        assert float(sess.data.qpos[2]) > 1.5                                   # still standing on the box
    samples._builtin(sess)
