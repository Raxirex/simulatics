"""Model-based walker (control/walker.py): unit tests of its parts and the click-to-walk acceptance runs.

Each walking test is one headless episode of a few simulated seconds (1-5 s wall time, single thread).
"""
import re

import mujoco
import numpy as np
import pytest

from simulatics.control.walk_footsteps import LEFT, RIGHT, StepLimits, plan_footsteps, square_up, wrap
from simulatics.control.walk_ik import SIDES, RobotGeometry, leg_ik, model_supported, rot_x, rot_y, rot_z
from simulatics.control.walk_preview import ZmpPreview
from simulatics.control.walker import WalkController, simulate_walk
from simulatics.models import HumanoidParams, build_mjcf
from simulatics.plan.grid import GridMap

BODIES = {
    "default": {},
    "short legs": {"thigh_length": 0.32, "shin_length": 0.32},
    "long legs, wide hips": {"thigh_length": 0.5, "shin_length": 0.5, "pelvis_width": 0.26},
}


def make(p: dict | None = None, obstacles=None):
    m = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams(**(p or {})), obstacles))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def run(c, m, d, seconds, stop_when_done=True):
    end = d.time + seconds
    while d.time < end:
        c.step(m, d)
        mujoco.mj_step(m, d, c.substeps)
        if c.mode == "fallen" or (stop_when_done and c.mode == "done"):
            break
    return c.status()


# ---------------------------------------------------------------- components
@pytest.mark.parametrize("body", list(BODIES))
def test_leg_ik_matches_mujoco_forward_kinematics(body):
    m, d = make(BODIES[body])
    assert model_supported(m) and WalkController.supports(m)
    geo = RobotGeometry.from_model(m)
    rng = np.random.default_rng(0)
    for _ in range(50):
        yaw = rng.uniform(-3, 3)
        R_t = rot_z(yaw) @ rot_y(rng.uniform(-0.15, 0.15)) @ rot_x(rng.uniform(-0.15, 0.15))
        p_t = np.array([*rng.uniform(-1, 1, 2), 0.93 * geo.leg_length + geo.ankle_height + 0.32])
        d.qpos[:] = 0
        d.qpos[0:3] = p_t
        mujoco.mju_mat2Quat(d.qpos[3:7], R_t.ravel())
        p_pel = p_t + R_t @ geo.pelvis
        want = {}
        for side in SIDES:
            R_f = rot_z(yaw + rng.uniform(-0.3, 0.3)) @ rot_y(rng.uniform(-0.2, 0.2))
            sole = p_pel + R_t @ geo.legs[side].hip + np.array([*rng.uniform(-0.15, 0.15, 2), 0.0])
            sole[2] = rng.uniform(0.0, 0.08)
            q, reach = leg_ik(geo.legs[side], p_pel, R_t, sole, R_f)
            assert reach < 1.0
            d.qpos[[m.joint(f"{side}_{j}").qposadr[0] for j in
                    ("hip_x", "hip_z", "hip_y", "knee", "ankle_y", "ankle_x")]] = q
            want[side] = (sole, R_f)
        mujoco.mj_kinematics(m, d)
        for side in SIDES:
            b = m.body(f"{side}_foot").id
            R = d.xmat[b].reshape(3, 3)
            assert np.allclose(d.xpos[b] + R @ geo.legs[side].sole, want[side][0], atol=1e-9)
            assert np.allclose(R, want[side][1], atol=1e-9)


def test_supports_only_the_builtin_leg_structure():
    blob = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint/><geom size='0.1'/></body></worldbody></mujoco>")
    assert not WalkController.supports(blob)
    with pytest.raises(ValueError):
        WalkController(blob)


def test_zmp_preview_tracks_footstep_pattern():
    pc = ZmpPreview(0.005, 0.9)
    n = int(6.0 / 0.005)
    t = np.arange(n + pc.N) * 0.005
    ref = np.zeros((n + pc.N, 2))
    for i, t0 in enumerate(np.arange(1.0, 3.4, 0.6)):
        ref[t >= t0] = [0.2 * (i + 1), 0.1 if i % 2 == 0 else -0.1]
    ref[t >= 3.4] = [1.0, 0.0]
    x = np.zeros((3, 2))
    for k in range(n):
        x = pc.step(x, ref[k + 1:k + 1 + pc.N])
    assert np.allclose(x[0], [1.0, 0.0], atol=1e-4) and np.abs(x[1:]).max() < 1e-3   # no steady-state offset


def test_footstep_plan_respects_limits_and_ends_on_goal():
    lim = StepLimits()
    left, right = np.array([0.0, 0.1, 0.0]), np.array([0.0, -0.1, 0.0])
    for path, yaw in (([[0, 0], [3, 0]], None), ([[0, 0], [2, 2]], np.pi / 2), ([[0, 0], [2, 0], [2, 2]], None),
                      ([[0.0, 0.0]], np.pi), ([[0, 0], [-0.3, 0.1]], None)):
        steps, gy = plan_footsteps(left, right, np.array(path, float), yaw, lim)
        feet = {LEFT: left, RIGHT: right}
        for x, y, a, side in steps:
            stance = feet[1 - side]
            assert abs(wrap(a - stance[2])) <= lim.turn + 1e-6
            c, s = np.cos(stance[2]), np.sin(stance[2])
            lat = -s * (x - stance[0]) + c * (y - stance[1])
            assert (lat if side == LEFT else -lat) >= lim.min_width - 1e-6     # feet never cross
            assert np.hypot(x - feet[side][0], y - feet[side][1]) <= 2 * lim.forward + 0.05
            feet[side] = np.array([x, y, a])
        goal = np.array(path[-1], float)
        mid = (feet[LEFT][:2] + feet[RIGHT][:2]) / 2
        assert np.linalg.norm(mid - goal) < 1e-6 and abs(wrap(feet[LEFT][2] - gy)) < 1e-6
        if yaw is not None:
            assert abs(wrap(gy - yaw)) < 1e-9
    assert square_up(left, right, LEFT, lim) == []          # already side by side


# ---------------------------------------------------------------- behaviour
def test_stands_ten_seconds_and_leaves_the_arms_alone():
    m, d = make()
    c = WalkController(m)
    c.reset(m, d)
    arms = [i for i in range(m.nu) if i not in set(c.act.tolist())]
    d.ctrl[arms] = 0.123
    z0 = d.qpos[2]
    s = run(c, m, d, 10.0)
    assert s["mode"] == "stand" and s["support"] == "double"
    assert abs(d.qpos[2] - z0) < 0.02 and np.linalg.norm(d.qvel[:3]) < 0.05
    assert np.allclose(d.ctrl[arms], 0.123)                  # arm / hand actuators belong to someone else
    assert set(s) >= {"mode", "message", "goal_error_m", "heading_error_deg", "footsteps", "support"}


def test_walks_to_3_0_and_stops_on_it():
    m, _ = make()
    r = simulate_walk(m, (3.0, 0.0), timeout=40)
    assert r["reached"] and not r["fell"] and r["error_m"] <= 0.08
    assert r["peak_torque_frac"] <= 1.0 and 0 < r["cost_of_transport"] < 5


def test_walks_to_2_2_ending_at_90_deg():
    m, _ = make()
    r = simulate_walk(m, (2.0, 2.0), np.pi / 2, timeout=40)
    assert r["reached"] and r["error_m"] <= 0.10 and abs(r["heading_error_deg"]) <= 10


def test_turns_180_deg_in_place():
    m, d = make()
    c = WalkController(m)
    c.reset(m, d)
    run(c, m, d, 0.3, stop_when_done=False)
    x0, y0 = d.qpos[0], d.qpos[1]
    c.set_path(np.array([[x0, y0]]), np.pi)
    s = run(c, m, d, 25.0)
    assert s["mode"] == "done" and s["goal_error_m"] < 0.05 and abs(s["heading_error_deg"]) < 5


def test_follows_astar_path_around_an_obstacle():
    obst = [{"pos": [2.0, 0.0], "size": [0.3, 1.0, 0.5], "yaw": 0.0}]
    m, d = make(obstacles=obst)
    grid = GridMap()
    grid.rasterize_truth(m, d)
    path = grid.plan([0.0, 0.0], [4.0, 0.0], robot_radius=0.35)
    assert len(path) >= 3 and np.abs(path[:, 1]).max() > 1.0        # the plan goes around the wall
    r = simulate_walk(m, (4.0, 0.0), timeout=60, path=path)
    assert r["reached"] and not r["fell"] and r["error_m"] < 0.05


@pytest.mark.parametrize("body", list(BODIES))
def test_different_bodies_walk_without_retuning(body):
    m, _ = make(BODIES[body])
    r = simulate_walk(m, (1.5, -1.0), np.deg2rad(-45), timeout=40)
    assert r["reached"] and r["error_m"] < 0.05 and abs(r["heading_error_deg"]) < 5


def test_replan_stop_and_velocity_mode():
    m, d = make()
    c = WalkController(m)
    c.reset(m, d)
    c.set_path(np.array([[0.0, 0.0], [3.0, 0.0]]), None)
    run(c, m, d, 3.0, stop_when_done=False)
    assert c.mode == "walk" and len(c.status()["footsteps"]) > 0
    x, y = d.qpos[0], d.qpos[1]
    c.set_path(np.array([[x, y], [x, 1.0]]), 0.0)          # new goal while a foot is in the air
    s = run(c, m, d, 30.0)
    assert s["mode"] == "done" and s["goal_error_m"] < 0.05
    c.set_velocity(0.2, 0.0, 0.0)
    x0 = d.qpos[0]
    run(c, m, d, 4.0, stop_when_done=False)
    assert c.mode == "walk" and d.qpos[0] - x0 > 0.3
    c.set_path(None)                                        # stop: square the feet and stand
    s = run(c, m, d, 3.0, stop_when_done=False)
    assert s["mode"] == "stand" and s["support"] == "double" and not s["footsteps"]
    assert np.linalg.norm(d.qvel[:2]) < 0.05


# ------------------------------------------------- regressions (independent review)
def test_long_path_is_walked_step_by_step_not_leapt():
    """The planner used to stop at a fixed 400 footsteps and then put the closing pair on the goal, i.e. one
    footstep of tens of metres for long paths (a small body crossing the 24 m map). The budget now follows
    the path length."""
    lim = StepLimits(forward=0.1, backward=0.06, lateral=0.05)
    left, right = np.array([0.0, 0.1, 0.0]), np.array([0.0, -0.1, 0.0])
    steps, _ = plan_footsteps(left, right, np.array([[0.0, 0.0], [60.0, 0.0]]), None, lim)
    feet = {LEFT: left, RIGHT: right}
    for x, y, a, side in steps:
        assert np.hypot(x - feet[side][0], y - feet[side][1]) <= 2 * lim.forward + 1e-6
        feet[side] = np.array([x, y, a])
    assert len(steps) > 400 and np.linalg.norm((feet[LEFT][:2] + feet[RIGHT][:2]) / 2 - [60.0, 0.0]) < 1e-6


def test_non_finite_requests_are_rejected():
    """NaN / inf used to hang the planner (inf path length) or end in "done" with a NaN error."""
    lim = StepLimits()
    left, right = np.array([0.0, 0.1, 0.0]), np.array([0.0, -0.1, 0.0])
    for path, yaw in (([[np.inf, 0.0]], None), ([[np.nan, 0.0]], None), ([[1.0, 0.0]], np.nan)):
        with pytest.raises(ValueError):
            plan_footsteps(left, right, np.array(path), yaw, lim)
    m, d = make()
    c = WalkController(m)
    c.reset(m, d)
    with pytest.raises(ValueError):
        c.set_path(np.array([[0.0, 0.0], [np.inf, 1.0]]))
    with pytest.raises(ValueError):
        c.set_path(np.array([[0.0, 0.0], [1.0, 1.0]]), float("nan"))
    with pytest.raises(ValueError):
        c.set_velocity(float("nan"), 0.0, 0.0)
    assert c.status()["mode"] == "stand" and c.status()["footsteps"] == []


def test_goal_the_robot_already_stands_on_needs_no_steps():
    """Re-sending the goal the robot stands on (a re-click, walk_relative(0, 0), a navigator re-plan) used to
    plan two in-place steps (~2.8 s) every time, so frequent re-sends never reached "done"."""
    m, d = make()
    c = WalkController(m)
    c.reset(m, d)
    run(c, m, d, 0.3, stop_when_done=False)
    here = d.qpos[:2].copy()
    for _ in range(2):
        c.set_path(np.array([here, here]), None)
        assert c.status()["footsteps"] == []
        s = run(c, m, d, 1.0)
        assert s["mode"] == "done" and s["goal_error_m"] < 0.01


def test_motors_without_torque_or_ctrl_limits():
    """No forcerange on the legs used to divide by zero in the motor-load estimate; no ctrlrange (ctrlrange
    0 0, ctrllimited off) used to clip every leg target to 0."""
    m, _ = make()
    m.actuator_forcelimited[:] = 0
    assert WalkController(m).gait_scale == 1.0
    xml = re.sub(r'ctrlrange="[^"]*" ', "", build_mjcf(HumanoidParams()))
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    c = WalkController(m)
    assert not m.actuator_ctrllimited[c.act].any()
    c.reset(m, d)
    z0 = d.qpos[2]
    s = run(c, m, d, 1.0)
    knee = m.actuator("left_knee").id
    assert s["mode"] == "stand" and d.ctrl[knee] > 0.2 and abs(d.qpos[2] - z0) < 0.02


def test_simulate_walk_ends_on_goal_even_if_the_path_does_not():
    m, _ = make()
    r = simulate_walk(m, (0.6, 0.0), path=np.array([[0.0, 0.0], [0.3, 0.0]]), timeout=20)
    assert r["reached"] and r["error_m"] < 0.03
