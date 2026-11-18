import mujoco
import numpy as np
import pytest

from simulatics import native
from simulatics.models import HumanoidParams, PARAM_SPEC, build_mjcf
from simulatics.plan import GridMap, Navigator
from simulatics.sim.parallel import ParallelEvaluator
from simulatics.sim.sensors import Lidar, LidarConfig
from simulatics.sim.task import LocoTask
from simulatics.train.evolve import Trainer, TrainConfig
from simulatics.train.policy import MLPSpec, Policy, RunningNorm

OBST = [{"pos": [3.0, 0.0], "size": [0.3, 2.0, 0.6], "yaw": 0.0}]


def model(p=None, obstacles=None):
    m = mujoco.MjModel.from_xml_string(build_mjcf(p or HumanoidParams(), obstacles))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_humanoid_compiles_at_parameter_extremes():
    for pick in (0, 1):
        p = HumanoidParams.from_dict({k: v[pick] for k, v in PARAM_SPEC.items()})
        m, d = model(p)
        assert m.nu == 20 and m.nq == 27
        feet = [d.geom(f"{s}_foot").xpos[2] for s in ("left", "right")]
        assert all(0.02 < z < 0.04 for z in feet)  # feet rest on the floor, not through it


def test_same_size_signature_with_and_without_morph_change():
    a, _ = model()
    b, _ = model(HumanoidParams(thigh_length=0.55, hip_kp=100))
    assert (a.nq, a.nv, a.nu, a.ngeom) == (b.nq, b.nv, b.nu, b.ngeom)


def test_stands_still_with_zero_action():
    m, d = model()
    mujoco.mj_step(m, d, 1000)  # 5 s
    assert d.qpos[2] > 0.95 * HumanoidParams().stand_height


def test_no_arms_variant():
    m, _ = model(HumanoidParams(arms=False))
    assert m.nu == 14


def test_task_obs_and_reward_shapes():
    m, d = model()
    task = LocoTask(m)
    s = np.tile(LocoTask.initial_state(m), (5, 1))
    qpos, qvel = task.split(s)
    obs = task.obs(qpos, qvel, np.zeros(5))
    assert obs.shape == (5, task.obs_dim) and np.isfinite(obs).all()
    moved = qpos.copy()
    moved[:, 0] += 0.02
    r, done = task.reward(qpos, moved, np.zeros((5, task.act_dim)), np.zeros(5))
    assert (r > 0).all() and not done.any()
    r_back, _ = task.reward(moved, qpos, np.zeros((5, task.act_dim)), np.zeros(5))
    assert (r_back < r).all()


def test_native_mlp_matches_numpy():
    spec = MLPSpec([12, 16, 16, 4])
    rng = np.random.default_rng(0)
    params = np.stack([spec.init(rng, 1.0) for _ in range(33)])
    obs = rng.standard_normal((33, 12)).astype(np.float32)
    ref = spec.forward_batch(params, obs, use_native=False)
    assert np.allclose(ref[0], spec.forward_batch(params[0], obs[:1])[0], atol=1e-5)
    if native.available():
        assert np.allclose(ref, spec.forward_batch(params, obs, use_native=True), atol=1e-5)


@pytest.mark.skipif(not native.available(), reason="simcore not built")
def test_native_planner_matches_python():
    rng = np.random.default_rng(1)
    occ = (rng.random((60, 60)) < 0.03).astype(np.uint8)
    occ[:3, :3] = 0
    occ[-3:, -3:] = 0
    cost_n, cost_p = native.inflate(occ, 1.0), native.inflate(occ, 1.0, force_python=True)
    assert ((cost_n == 255) == (cost_p == 255)).mean() > 0.97  # chamfer vs exact distance at the rim
    a = native.astar(cost_n, (0, 0), (59, 59))
    b = native.astar(cost_n, (0, 0), (59, 59), force_python=True)
    assert len(a) > 0 and len(b) > 0

    def length(path):
        steps = np.diff(path, axis=0)
        w = 1 + cost_n[path[1:, 1], path[1:, 0]] / 50
        return float((np.hypot(steps[:, 0], steps[:, 1]) * w).sum())

    assert abs(length(a) - length(b)) < 1e-3  # both optimal
    sa, sb = native.smooth_path(cost_n, a, 60), native.smooth_path(cost_n, a, 60, force_python=True)
    assert np.array_equal(sa, sb)
    la, lb = np.zeros((40, 40), np.float32), np.zeros((40, 40), np.float32)
    ends = rng.integers(-5, 45, (50, 2)).astype(np.int32)
    hit = (rng.random(50) < 0.5).astype(np.uint8)
    native.integrate_scan(la, (20, 20), ends, hit)
    native.integrate_scan(lb, (20, 20), ends, hit, force_python=True)
    assert np.allclose(la, lb)


def test_planner_goes_around_wall_and_reports_unreachable():
    m, d = model(obstacles=OBST)
    grid = GridMap()
    grid.rasterize_truth(m, d)
    assert grid.truth.sum() > 0
    path = grid.plan([0, 0], [6, 0])
    assert len(path) >= 3 and np.abs(path[:, 1]).max() > 2.0  # detours around the 4 m wide wall
    assert not grid.blocked(path)
    assert grid.blocked(np.array([[0.0, 0.0], [6.0, 0.0]]))
    assert len(grid.plan([0, 0], [3.0, 0.0])) == 0  # goal inside the obstacle


def test_lidar_sees_obstacle_and_builds_map():
    m, d = model(obstacles=OBST)
    lidar = Lidar(m, LidarConfig(h_samples=360, channels=4, v_min=-10, v_max=0))
    origin, pts, dist, hit = lidar.scan(m, d)
    wall = hit & (pts[:, 2] > 0.2)
    assert wall.any() and abs(pts[wall, 0].min() - 2.7) < 0.05  # wall face at x = 3.0 - 0.3
    assert not (hit & (dist < 0.3)).any()  # never hits the robot itself
    grid = GridMap()
    grid.integrate_lidar(origin, pts, hit, 360)
    c = grid.to_cell([2.72, 0.0])
    assert grid.lidar_occ[c[1], c[0] - 1:c[0] + 2].any()
    free = grid.to_cell([1.5, 0.0])
    assert grid.logodds[free[1], free[0]] < 0


def test_navigator_behaviour_tree():
    m, d = model(obstacles=OBST)
    grid = GridMap()
    grid.rasterize_truth(m, d)
    nav = Navigator(grid)
    assert nav.tick(np.zeros(2), 0.0) is None
    nav.set_goal([6, 0])
    yaw = nav.tick(np.zeros(2), 0.0)
    assert yaw is not None and nav.active == "FollowPath" and abs(yaw) > 0.3  # steers around the wall
    assert nav.tick(np.array([5.9, 0.0]), 0.0) is None and not nav.goals
    nav.set_goal([3.0, 0.0])
    assert nav.tick(np.zeros(2), 0.0) is None and "no path" in nav.message


def test_parallel_evaluator_matches_single_thread():
    m, _ = model()
    task = LocoTask(m)
    spec = MLPSpec([task.obs_dim, 8, task.act_dim])
    params = np.stack([spec.init(np.random.default_rng(i), 0.5) for i in range(6)])
    norm = RunningNorm(task.obs_dim)
    out = []
    for nt in (1, 4):
        with ParallelEvaluator(nt) as ev:
            fit, steps, obs = ev.evaluate(m, task, spec, params, 40, norm, np.random.default_rng(0))
            out.append(fit)
            assert steps > 0 and obs.shape[1] == task.obs_dim
    assert np.allclose(out[0], out[1])


@pytest.mark.parametrize("algo,body", [("es", False), ("ga", True)])
def test_trainer_runs_and_saves_loadable_policy(tmp_path, algo, body):
    cfg = TrainConfig(algo=algo, population=16, generations=2, episode_seconds=1.0, evolve_body=body,
                      threads=2, name="t")
    tr = Trainer(cfg, HumanoidParams(), runs_dir=tmp_path)
    tr.run()
    assert len(tr.history) == 2 and tr.policy_path.exists()
    pol = Policy.load(tr.policy_path)
    m, _ = model(HumanoidParams.from_dict(pol.meta["humanoid"]))
    task = LocoTask(m)
    s = LocoTask.initial_state(m)[None]
    act = pol.act(task.obs(*task.split(s), np.zeros(1)))
    assert act.shape == (1, 20) and np.abs(act).max() <= 1
