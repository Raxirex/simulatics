"""End-to-end tests of the live session: planner + walker + arms + pilot through the same paths the GUI uses."""
import time

import numpy as np
import pytest

from simulatics.control import base_pose
from simulatics.server.session import SimSession


def wait_for(cond, timeout: float) -> bool:
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if cond():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def sess():
    s = SimSession()
    s.lidar_cfg.enabled = False
    s.speed = 8.0  # as fast as the CPU allows
    yield s
    s.pilot.stop(timeout=1.0)
    s.running = False
    s.shutdown()


def test_default_controller_is_the_walker_and_arms_are_attached(sess):
    assert sess.controller.name == "walker"
    assert sess.arms is not None
    # an arm command right after start-up must work (the controller is bound to the live data)
    msg = sess.robot.arm_gesture("right", "raise")
    assert "not attached" not in msg


def test_push_moves_the_standing_robot_without_a_fall(sess):
    sess.running = True
    time.sleep(0.3)
    with sess.lock:
        y0 = float(sess.data.qvel[1])
    msg = sess.push("left", force=60.0, duration=0.1)
    assert "pushing left" in msg
    assert wait_for(lambda: sess._push is None, timeout=10)
    with sess.lock:
        assert np.allclose(sess.data.xfrc_applied, 0.0)  # the force is removed after the push
    time.sleep(1.0)
    assert sess.falls == 0
    with pytest.raises(ValueError):
        sess.push("up")


def test_terrain_course_is_walked_across(sess):
    gx, gy = sess.terrain_course("stairs", seed=0)
    assert any(o.get("kind") == "terrain" for o in sess.obstacles)
    assert sess.grid.truth.sum() == 0          # stairs are walkable, not walls, for the planner
    with sess.lock:
        sess.nav.set_goal([gx, gy], yaw=0.0)
        sess.running = True
    assert wait_for(lambda: sess.nav.state in ("reached", "failed", "fallen"), timeout=120), sess.nav.to_dict()
    assert sess.nav.state == "reached", sess.nav.to_dict()
    assert sess.falls == 0


def test_motion_preview_and_trained_imitation_policy_play(sess):
    import pathlib
    msg = sess.play_motion("wave", "preview")
    assert "wave" in msg and sess.controller.name == "motion-preview"
    time.sleep(1.0)
    policy = pathlib.Path("runs/imitate_walk/imitation_policy.npz")
    if not policy.exists():
        pytest.skip("trained imitation policy not present")
    sess.play_motion(None, "policy", str(policy.resolve()))
    assert sess.controller.name == "imitation"
    with sess.lock:
        x0 = float(sess.data.qpos[0])
        t0 = float(sess.data.time)
    assert wait_for(lambda: sess.data.time > t0 + 6.0, timeout=60)
    with sess.lock:
        moved = abs(float(sess.data.qpos[0]) - x0) + abs(float(sess.data.qpos[1]))
    assert sess.falls == 0 and moved > 0.5   # it walks the reference clip without falling


def test_mpc_controller_choice_walks_to_a_goal(sess):
    sess.set_controller("mpc")
    assert sess.controller.name == "mpc"
    with sess.lock:
        sess.nav.set_goal([1.5, 0.5], yaw=0.5)
        sess.running = True
    assert wait_for(lambda: sess.nav.state in ("reached", "failed", "fallen"), timeout=60), sess.nav.to_dict()
    assert sess.nav.state == "reached"
    with sess.lock:
        x, y, _ = base_pose(sess.data)
    assert np.hypot(x - 1.5, y - 0.5) < 0.05
    assert sess.falls == 0


def test_navigation_on_the_slam_map_and_pose(sess):
    sess.lidar_cfg.enabled = True
    sess.random_obstacles(10, seed=7)
    with sess.lock:
        sess.nav_on_slam = True
        sess.nav.grid, sess.nav.source = sess.slam_grid, "lidar"
        sess.running = True
    time.sleep(1.0)  # a few scans to start the map
    with sess.lock:
        sess.nav.set_goal([2.5, 1.0])
    assert wait_for(lambda: sess.nav.state in ("reached", "failed", "fallen"), timeout=90), sess.nav.to_dict()
    assert sess.nav.state == "reached", sess.nav.to_dict()
    with sess.lock:
        x, y, _ = base_pose(sess.data)
        ex, ey, _ = sess.slam.pose
        assert sess.slam.scans > 20
    assert np.hypot(ex - x, ey - y) < 0.15, ((x, y), (ex, ey))  # SLAM agrees with the truth
    assert np.hypot(x - 2.5, y - 1.0) < 0.2                     # and the robot got there
    assert sess.falls == 0


def test_click_to_walk_reaches_queued_goals_precisely(sess):
    with sess.lock:
        sess.nav.set_goal([1.2, 0.0])
        sess.nav.set_goal([1.2, 1.0], yaw=np.pi / 2, append=True)
        sess.running = True
    assert wait_for(lambda: sess.nav.state == "reached" and not sess.nav.goals, timeout=60), sess.nav.to_dict()
    with sess.lock:
        x, y, yaw = base_pose(sess.data)
    assert np.hypot(x - 1.2, y - 1.0) < 0.05, (x, y)
    assert abs((yaw - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi) < np.deg2rad(5)
    assert "goal reached" in sess.nav.message
    assert sess.falls == 0


def test_manipulation_scene_objects_and_props(sess):
    sess.manipulation_scene(seed=1)
    objs = sess.robot.objects()
    props = [o for o in objs if o.get("graspable")]
    assert props and all("z" in o for o in props)
    assert any(o["visible"] for o in props), objs  # the head camera looks at the table
    jpeg = sess.robot.camera_jpeg("head_cam", 160, 120)
    assert jpeg[:2] == b"\xff\xd8"


def test_scripted_pilot_picks_up_a_prop(sess):
    sess.manipulation_scene(seed=3)
    target = sess.props[0]["name"]
    z0 = next(o for o in sess.robot.objects() if o["name"] == target)["z"]
    sess.running = True
    sess.pilot.start(f"pick up the {target}", {"pilot": "scripted", "max_wall_seconds": 120})
    assert wait_for(lambda: not sess.pilot.running, timeout=120)
    st = sess.pilot.brief()
    assert st["state"] == "done" and st["success"] is not False, sess.pilot.events(0)["events"][-4:]
    held = {a.get("holding") for a in sess.arms.status().values()}
    assert target in held
    z1 = next(o for o in sess.robot.objects() if o["name"] == target)["z"]
    assert z1 > z0 + 0.08, (z0, z1)
    assert sess.falls == 0


def test_scripted_pilot_walks_to_the_table_then_picks(sess):
    sess.manipulation_scene(seed=5)
    with sess.lock:  # start 1.2 m back from the table, so the robot must walk up to it first
        sess.nav.set_goal([-1.2, 0.3])
        sess.running = True
    assert wait_for(lambda: sess.nav.state == "reached", timeout=60)
    target = sess.props[0]["name"]
    sess.pilot.start(f"pick up the {target}", {"pilot": "scripted", "max_wall_seconds": 150})
    assert wait_for(lambda: not sess.pilot.running, timeout=150)
    st = sess.pilot.brief()
    assert st["state"] == "done", sess.pilot.events(0)["events"][-4:]
    assert target in {a.get("holding") for a in sess.arms.status().values()}
    assert sess.falls == 0


def test_scripted_pilot_walks_to_a_named_object(sess):
    sess.random_obstacles(6, seed=4)
    target = min(sess.obstacles, key=lambda o: np.hypot(*o["pos"]))
    sess.running = True
    sess.pilot.start(f"walk to the {target['name']}", {"pilot": "scripted", "max_wall_seconds": 120})
    assert wait_for(lambda: not sess.pilot.running, timeout=120)
    st = sess.pilot.brief()
    assert st["state"] == "done", st
    with sess.lock:
        x, y, _ = base_pose(sess.data)
    gap = np.hypot(x - target["pos"][0], y - target["pos"][1]) - max(target["size"][:2])
    assert gap < 1.2, (x, y, target)
