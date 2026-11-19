"""Rough terrain: scene generators (sim/terrain.py), ground queries and footstep repair (control/walk_terrain.py)
and the model-based walker on steps, stairs and a ramp. Each walking test is one headless episode (5-12 s wall
time, single thread)."""
import mujoco
import numpy as np
import pytest

from simulatics.control.walk_footsteps import LEFT, StepLimits, plan_footsteps
from simulatics.control.walk_terrain import TerrainMap, TerrainReach, repair_footsteps
from simulatics.control.walker import WalkController, WalkParams, simulate_walk
from simulatics.models import HumanoidParams, build_mjcf
from simulatics.sim import terrain as T


def model(obstacles):
    m = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams(), obstacles))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


# ------------------------------------------------------------------ generators
def test_generators_are_tagged_walkable_and_compile():
    for kind in ("random_steps", "stairs", "slope", "mixed"):
        obs, goal = T.terrain_course(kind, seed=3)
        assert obs and all(o["kind"] == "terrain" for o in obs)
        assert goal[0] > T.course_length(obs) and goal[1] == 0.0
        m, _ = model(obs)
        assert m.ngeom >= len(obs)
    with pytest.raises(ValueError):
        T.terrain_course("lava")


def test_random_steps_are_seeded_flat_topped_and_low():
    a, b = T.random_steps(seed=4), T.random_steps(seed=4)
    assert a == b and a != T.random_steps(seed=5)
    for o in a:
        assert 0.01 - 1e-9 <= 2 * o["size"][2] <= 0.05 + 1e-9          # 1-5 cm by default
        assert 2 * o["size"][0] >= 0.3                                  # long enough to stand on
        assert "pitch" not in o


def test_stairs_and_ramp_profiles():
    m, d = model(T.stairs(n_steps=4, rise=0.06, tread=0.35, x0=0.8))
    tm = TerrainMap(m, 0.11, 0.045)
    assert tm.has_terrain
    for i in range(4):                                                  # middle of each tread going up
        assert abs(tm.height(0.8 + 0.35 * i + 0.175, 0.0)[0] - 0.06 * (i + 1)) < 1e-6
    m, d = model(T.slope(angle_deg=8.0, length=1.0, x0=0.8))
    tm = TerrainMap(m, 0.11, 0.045)
    z, gid, n = tm.height(1.3, 0.0)
    assert abs(z - 0.5 * np.tan(np.deg2rad(8))) < 1e-4
    assert abs(np.degrees(np.arccos(n[2])) - 8.0) < 0.05


def test_build_mjcf_ramp_keys_leave_old_scenes_unchanged():
    obs = [{"pos": [2.0, 0.0], "size": [0.3, 1.0, 0.5], "yaw": 0.3}]
    xml = build_mjcf(HumanoidParams(), obs)
    assert 'pos="2 0 0.5" euler="0 0 17.189"' in xml                    # resting on the floor, yaw only
    tilted = build_mjcf(HumanoidParams(), [{**obs[0], "pitch": 0.1, "z": 0.2}])
    assert 'pos="2 0 0.2" quat=' in tilted


# ------------------------------------------------------------- ground queries
def test_footholds_reject_edges_and_tilt_onto_ramps():
    m, _ = model(T.stairs(x0=0.8))
    tm = TerrainMap(m, 0.11, 0.045)
    assert tm.foothold(0.4, 0.1, 0.0).ok and tm.foothold(0.4, 0.1, 0.0).flat
    edge = tm.foothold(0.8, 0.1, 0.0)                                   # sole across the first riser
    assert not edge.ok and "edge" in edge.why
    on = tm.foothold(0.8 + 0.175, 0.1, 0.0, z_ref=0.0)
    assert on.ok and abs(on.z - 0.06) < 1e-6
    m, _ = model(T.slope(angle_deg=8.0, x0=0.8))
    tm = TerrainMap(m, 0.11, 0.045)
    fh = tm.foothold(1.3, 0.1, 0.0, z_ref=0.06)
    assert fh.ok and abs(np.degrees(np.arccos(fh.R[2, 2])) - 8.0) < 0.2   # sole pitched onto the ramp
    assert fh.R[0, 2] < 0                                               # toe up when walking uphill (+x)


def test_tall_obstacles_are_not_terrain():
    m, _ = model([{"pos": [2.0, 0.0], "size": [0.3, 1.0, 0.5]}])        # a 1 m wall
    assert not TerrainMap(m, 0.11, 0.045).has_terrain
    assert WalkController(m).terrain is None                             # flat-ground walker, unchanged


def test_repair_puts_every_sole_fully_on_one_surface():
    lim = StepLimits(forward=0.176, backward=0.096, lateral=0.072, width=0.2, min_width=0.13, lookahead=0.36,
                     blend=0.6)
    reach = TerrainReach(forward=0.288, backward=0.096, min_width=0.13, max_width=0.272)
    for obs in (T.stairs(), T.random_steps(seed=2)):
        m, _ = model(obs)
        tm = TerrainMap(m, 0.11, 0.045)
        left, right = np.array([0.044, 0.1, 0.0]), np.array([0.044, -0.1, 0.0])
        goal = (T.course_length(obs) + 0.5, 0.0)
        steps, _ = plan_footsteps(left, right, np.array([[0.0, 0.0], goal]), None, lim)
        rep = repair_footsteps(steps, left, right, tm, reach, lim.forward)
        assert len(rep) == len(steps)
        feet = {LEFT: left, 1 - LEFT: right}
        for x, y, yaw, side, z, R in rep:
            assert tm.foothold(x, y, yaw, z).ok
            stance = feet[1 - side]
            assert x - stance[0] <= reach.forward + 1e-9                 # within the (terrain) reach
            lat = (y - stance[1]) * (1 if side == LEFT else -1)
            assert lat >= reach.min_width - 1e-9                         # never crossing the legs
            feet[side] = np.array([x, y, yaw])


# --------------------------------------------------------------------- walking
@pytest.mark.parametrize("course", ["stairs", "slope", "random_steps"])
def test_walks_across_terrain_and_stops_on_the_goal(course):
    """Up and down a 4-step stair (6 cm risers), up and down an 8 deg ramp, across 1-5 cm random blocks."""
    obs, goal = T.terrain_course(course, seed=0)
    m, _ = model(obs)
    r = simulate_walk(m, goal, timeout=60)
    assert r["reached"] and not r["fell"] and r["error_m"] <= 0.03
    assert r["status"]["terrain"] is True


def test_terrain_status_reports_footstep_heights():
    obs, goal = T.terrain_course("stairs")
    m, d = model(obs)
    c = WalkController(m)
    c.reset(m, d)
    c.set_path(np.array([d.qpos[:2], goal]), None)
    st = c.status()
    assert len(st["footsteps_z"]) == len(st["footsteps"]) and max(st["footsteps_z"]) == pytest.approx(0.24)


def test_terrain_can_be_switched_off():
    m, _ = model(T.stairs())
    assert WalkController(m, WalkParams(terrain=False)).terrain is None
    assert WalkController(m).terrain is not None
