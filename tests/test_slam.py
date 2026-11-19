"""2D lidar SLAM (plan/slam.py, native/src/slam.cpp): kernels, odometry model, scan matching, and the
walking acceptance runs (SLAM vs odometry-only trajectory error, map overlap, timing, feature-poor scene).

The two walking episodes take ~25 s and ~5 s wall time (single thread for physics, <= 2 OpenMP threads).
Run with -s to see the measured numbers.
"""
import time

import mujoco
import numpy as np
import pytest

from simulatics.models import HumanoidParams, build_mjcf
from simulatics.plan import slam as S
from simulatics.plan.grid import GridMap
from simulatics.sim.scene import random_objects
from simulatics.sim.sensors import Lidar, LidarConfig

native = pytest.mark.skipif(not S.native_available(), reason="native slam library not built")


# ------------------------------------------------------------------------------------------- helpers
def _random_field(n=200, seed=1):
    rng = np.random.default_rng(seed)
    lo = np.zeros((n, n), np.float32)
    for _ in range(25):
        x0, y0 = rng.integers(0, n, 2)
        L = int(rng.integers(5, 60))
        if rng.random() < 0.5:
            lo[y0, x0:x0 + L] = 2.0
        else:
            lo[y0:y0 + L, x0] = 2.0
    lo[rng.random((n, n)) < 0.002] = 1.0
    return lo


def _room(res=0.01):
    """8 x 6 m room with two boxes and a pillar, as a fine occupancy grid centred on the origin."""
    n = int(12 / res)
    occ = np.zeros((n, n), bool)
    o = -6.0

    def box(x0, y0, x1, y1):
        occ[int((y0 - o) / res):int((y1 - o) / res), int((x0 - o) / res):int((x1 - o) / res)] = True

    box(-4, -3, 4, -2.9), box(-4, 2.9, 4, 3), box(-4, -3, -3.9, 3), box(3.9, -3, 4, 3)
    box(1.0, 0.8, 1.8, 1.4), box(-2.5, -1.8, -1.9, -0.9), box(0.4, -1.6, 0.6, -1.4)
    return occ, res, o


def _cast(occ, res, o, pose, n_rays=180, max_r=8.0):
    """Synthetic planar scan from `pose` (robot-frame points, hit mask)."""
    rel = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    ang = pose[2] + rel
    r = np.arange(0.05, max_r, res / 2)
    xs = pose[0] + np.cos(ang)[:, None] * r
    ys = pose[1] + np.sin(ang)[:, None] * r
    cx = np.clip(((xs - o) / res).astype(int), 0, occ.shape[1] - 1)
    cy = np.clip(((ys - o) / res).astype(int), 0, occ.shape[0] - 1)
    blocked = occ[cy, cx]
    hit = blocked.any(axis=1)
    rr = np.where(hit, r[blocked.argmax(axis=1)], max_r)
    return np.stack([rr * np.cos(rel), rr * np.sin(rel)], axis=1), hit


def _search_inputs(n=200, K=21, npts=120, seed=2):
    rng = np.random.default_rng(seed)
    cells = np.stack([rng.integers(-8, n + 8, (K, npts)), rng.integers(-8, n + 8, (K, npts))], -1).astype(np.int32)
    return cells, rng.integers(0, 2000, K).astype(np.int64), 25


# ------------------------------------------------------------------------------------------- algebra / odometry
def test_se2_compose_inverse_between():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b = rng.uniform(-3, 3, 3), rng.uniform(-3, 3, 3)
        assert np.allclose(S.compose(a, S.between(a, b))[:2], b[:2])
        assert abs(S.wrap(S.compose(a, S.between(a, b))[2] - b[2])) < 1e-12
        assert np.allclose(S.compose(a, S.inverse(a)), 0, atol=1e-12)


def test_odometry_noise_model_is_deterministic_and_has_the_configured_bias():
    true = [np.array([0.05 * k, 0.0, 0.0]) for k in range(201)]           # 10 m straight
    runs = []
    for _ in range(2):
        odo = S.OdometrySim(S.OdomNoise(scale=0.03, yaw_drift_deg_per_m=2.0, trans_walk=0.0, yaw_walk_deg=0.0), seed=4)
        runs.append(np.array([odo.update(p) for p in true]))
    assert np.array_equal(runs[0], runs[1])
    final = runs[0][-1]
    assert np.isclose(np.rad2deg(final[2]), 20.0, atol=1e-6)               # 2 deg/m over 10 m
    seg = np.hypot(*np.diff(runs[0][:, :2], axis=0).T).sum()
    assert np.isclose(seg, 10.3, atol=1e-6)                                 # 3 % long
    noisy = S.OdometrySim(S.OdomNoise(), seed=4)
    a = np.array([noisy.update(p) for p in true])
    noisy.reset()
    b = np.array([noisy.update(p) for p in true])
    assert np.array_equal(a, b)
    clean = S.OdometrySim(S.OdomNoise.none(), seed=1)
    c = np.array([clean.update(p) for p in true])
    assert np.allclose(c, np.array(true), atol=1e-12)


# ------------------------------------------------------------------------------------------- kernels
def test_window_max_semantics():
    F = S.likelihood_field(_random_field(), 0.0, S.likelihood_lut(2.0, 6.0), use_native=False)
    rng = np.random.default_rng(3)
    for B in (1, 3, 4):
        M = S.window_max(F, B, use_native=False)
        assert M.shape == (F.shape[0] + B - 1, F.shape[1] + B - 1)
        for _ in range(300):
            x, y = rng.integers(-(B - 1), F.shape[0], 2)
            sub = F[max(y, 0):max(y + B, 0), max(x, 0):max(x + B, 0)]
            assert M[y + B - 1, x + B - 1] == (sub.max() if sub.size else 0)


def test_likelihood_field_is_exact_distance_based():
    lo = np.zeros((40, 40), np.float32)
    lo[20, 20] = 1.0
    lut = S.likelihood_lut(2.0, 6.0)
    F = S.likelihood_field(lo, 0.0, lut, use_native=False)
    yy, xx = np.mgrid[0:40, 0:40]
    d2 = (yy - 20) ** 2 + (xx - 20) ** 2
    assert np.array_equal(F, np.where(d2 < len(lut), lut[np.minimum(d2, len(lut) - 1)], 0))


@pytest.mark.parametrize("use_native", [False, pytest.param(True, marks=native)])
@pytest.mark.parametrize("B", [1, 3, 4, 5])
def test_branch_and_bound_equals_brute_force(use_native, B):
    F = S.likelihood_field(_random_field(), 0.0, S.likelihood_lut(2.0, 6.0), use_native)
    fmax = S.window_max(F, B, use_native)
    cells, rot_pen, lam = _search_inputs()
    win = 6
    G = S.score_grid(F, cells, win, rot_pen, lam, use_native)
    k, dx, dy, s, expanded = S.search(F, fmax, cells, win, B, rot_pen, lam, use_native)
    assert s == G.max() == G[k, dy + win, dx + win]
    if B > 1:
        assert expanded < G.shape[0] * ((2 * win) // B + 1) ** 2   # the bound pruned something


@native
def test_native_kernels_match_numpy_bit_for_bit():
    lo = _random_field(300, seed=5)
    lut = S.likelihood_lut(2.0, 6.0)
    Fn, Fp = S.likelihood_field(lo, 0.0, lut, True), S.likelihood_field(lo, 0.0, lut, False)
    assert np.array_equal(Fn, Fp)
    for B in (2, 4, 8):
        assert np.array_equal(S.window_max(Fn, B, True), S.window_max(Fn, B, False))
    cells, rot_pen, lam = _search_inputs(300)
    assert np.array_equal(S.score_grid(Fn, cells, 5, rot_pen, lam, True), S.score_grid(Fn, cells, 5, rot_pen, lam, False))
    fm = S.window_max(Fn, 4, True)
    assert S.search(Fn, fm, cells, 7, 4, rot_pen, lam, True) == S.search(Fn, fm, cells, 7, 4, rot_pen, lam, False)


# ------------------------------------------------------------------------------------------- scans
def _scene_model(obstacles, robot_xy=(0.0, 0.0), robot_yaw=0.0):
    m = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams(), obstacles))
    d = mujoco.MjData(m)
    d.qpos[0:2] = robot_xy
    mujoco.mju_mat2Quat(d.qpos[3:7], np.array([[np.cos(robot_yaw), -np.sin(robot_yaw), 0],
                                               [np.sin(robot_yaw), np.cos(robot_yaw), 0], [0, 0, 1.0]]).ravel())
    mujoco.mj_forward(m, d)
    return m, d


def test_planar_scan_does_not_depend_on_the_true_pose():
    """Move the robot AND the world by the same rigid transform: the robot-frame scan must not change."""
    obs = random_objects(10, seed=2)
    m, d = _scene_model(obs)
    T = np.array([2.3, -1.7, 0.9])
    moved = []
    for o in obs:
        p = S.transform_points(T, np.array([o["pos"]]))[0]
        moved.append({**o, "pos": [float(p[0]), float(p[1])], "yaw": o["yaw"] + T[2]})
    m2, d2 = _scene_model(moved, T[:2], T[2])
    lid, lid2 = Lidar(m, LidarConfig()), Lidar(m2, LidarConfig())
    a, b = S.scan_robot_frame(m, d, lid), S.scan_robot_frame(m2, d2, lid2)
    assert a.hit.sum() > 30
    assert np.array_equal(a.hit, b.hit)
    assert np.allclose(a.points[a.hit], b.points[b.hit], atol=2e-3)   # MJCF rounds positions to ~1e-5 m, yaw to ~1e-6 rad
    assert np.allclose(a.origin, b.origin, atol=1e-9)


def test_planar_scan_ignores_tops_of_low_boxes_for_matching():
    low = {"name": "low box", "shape": "box", "pos": [1.5, 0.0], "size": [1.0, 1.0, 0.5], "yaw": 0.0}
    tall = {"name": "tall box", "shape": "box", "pos": [-2.5, 0.0], "size": [0.25, 1.0, 1.2], "yaw": 0.0}
    m, d = _scene_model([low, tall])
    sc = S.scan_robot_frame(m, d, Lidar(m, LidarConfig()))
    ang = np.arctan2(sc.points[:, 1], sc.points[:, 0])
    front = sc.hit & (np.abs(ang) < 0.3)
    back = sc.hit & (np.abs(S.wrap(ang - np.pi)) < 0.3)
    assert front.sum() > 5 and back.sum() > 5
    assert not sc.match[front].any()        # beams clear the near face and land on the top: not for matching
    assert sc.match[back].all()             # a vertical face is


# ------------------------------------------------------------------------------------------- matching
@pytest.mark.parametrize("use_native", [False, pytest.param(True, marks=native)])
@pytest.mark.parametrize("odom_step, tol_m", [([0.32, 0.12, 2.0], 0.02),      # 4 cm, 2 deg off
                                              ([0.20, 0.25, 1.0], 0.05)])     # 18 cm, 3 deg off
def test_scan_matching_recovers_the_pose_from_wrong_odometry(use_native, odom_step, tol_m):
    """True motion 0.36 m / 4 deg; a single-scan map at 0.05 m. The second case is far beyond a real
    per-scan odometry error (mm at 10 Hz): the search still finds the pose, the motion prior pulls
    the estimate a few cm towards the odometry."""
    occ, res, o = _room()
    p0 = np.array([-1.0, 0.3, 0.2])
    p1 = S.compose(p0, [0.35, 0.1, np.deg2rad(4.0)])
    odom = S.compose(p0, [odom_step[0], odom_step[1], np.deg2rad(odom_step[2])])
    slam = S.Slam2D(size=12.0, res=0.05, use_native=use_native)
    slam.reset(p0)
    pts, hit = _cast(occ, res, o, p0)
    slam.update(p0, pts, hit)
    pts, hit = _cast(occ, res, o, p1)
    est = np.array(slam.update(odom, pts, hit))
    assert slam.state == "tracking", slam.stats()
    assert np.hypot(*(est[:2] - p1[:2])) < tol_m
    assert abs(np.rad2deg(S.wrap(est[2] - p1[2]))) < 0.6
    assert np.hypot(*(odom[:2] - p1[:2])) > tol_m      # it did correct the odometry
    assert slam.score > 0.6


def test_no_obstacles_means_pure_dead_reckoning():
    slam = S.Slam2D(size=12.0, res=0.1)
    slam.reset((0.5, 0.0, 0.1))
    ring = np.stack([np.cos(np.linspace(0, 6, 90)), np.sin(np.linspace(0, 6, 90))], 1) * 12.0
    odom = np.zeros(3)
    want = np.array([0.5, 0.0, 0.1])
    slam.update(odom, ring, np.zeros(90, bool))                     # first reading: no motion yet
    for k in range(20):
        step = np.array([0.05, 0.01, 0.02])
        odom = S.compose(odom, step)
        want = S.compose(want, step)
        est = slam.update(odom, ring, np.zeros(90, bool))
    assert slam.state == "odometry" and slam.score == 0.0
    assert np.allclose(est, want, atol=1e-12)
    assert slam.stats()["sigma_xy"][1] > 0.01                       # uncertainty grows with distance


def test_public_api_and_map_layout():
    slam = S.Slam2D(size=24.0, res=0.1, odom_noise=S.OdomNoise(), seed=0)
    g = GridMap(24.0, 0.1)
    assert slam.map_logodds.shape == g.logodds.shape and slam.map_logodds.dtype == g.logodds.dtype
    occ, res, o = _room()
    pts, hit = _cast(occ, res, o, (0, 0, 0))
    pose = slam.update((0, 0, 0), pts, hit)
    assert isinstance(pose, tuple) and len(pose) == 3 and pose == slam.pose
    assert slam.occupancy().dtype == np.uint8 and slam.occupancy().sum() > 50
    assert set(slam.stats()) >= {"scans", "match_ms", "score", "loop_closures"}
    d = slam.to_dict()
    assert set(d) >= {"pose", "state", "ellipse", "traj", "odom_traj"}
    gm = slam.gridmap()
    assert gm.logodds is slam.map_logodds and gm.lidar_occ.sum() == slam.occupancy().sum()
    fine = S.Slam2D(size=24.0, res=0.05)
    fine.update((0, 0, 0), pts, hit)
    coarse = fine.resampled(0.1)
    assert coarse.shape == g.logodds.shape and coarse.dtype == np.float32
    assert ((coarse > 0.5).sum() > 0.4 * slam.occupancy().sum())
    path = np.array([[0.0, 0.0], [1.0, 2.0]])
    assert np.allclose(slam.to_controller_frame(path, slam.pose), path)
    shifted = slam.to_controller_frame(path, (1.0, 0.0, 0.0))
    assert np.allclose(shifted, path + [1.0, 0.0])
    slam.reset()
    assert slam.scans == 0 and not slam.map_logodds.any() and slam.pose == (0.0, 0.0, 0.0)


# ------------------------------------------------------------------------------------------- walking acceptance
WORLD_SEED = 7
WAYPOINTS = [(0, 0), (3, 0), (3, 3), (-2.5, 3), (-2.5, -1.5), (0.5, -1.5)]


def _route(model, data):
    g = GridMap()
    g.rasterize_truth(model, data)
    route = [np.array(WAYPOINTS[0], float)]
    for b in WAYPOINTS[1:]:
        p = g.plan(route[-1], b, robot_radius=0.6)
        if len(p) >= 2:
            route.extend(p[1:])
    return np.array(route)


@pytest.fixture(scope="module")
def episode():
    xml = build_mjcf(HumanoidParams(), random_objects(12, WORLD_SEED))
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    route = _route(m, d)
    t = time.perf_counter()
    r = S.run_slam_episode(xml, route, seconds=140.0, seed=0, keep_log=True)
    r["wall_s"] = time.perf_counter() - t
    r["route_m"] = float(np.hypot(*np.diff(route, axis=0).T).sum())
    return r


def test_slam_beats_odometry_on_a_walked_route(episode):
    r = episode
    print(f"\nroute {r['route_m']:.1f} m planned, {r['distance_m']:.1f} m walked by the torso, {r['scans']} scans, "
          f"{r['wall_s']:.1f} s wall")
    print(f"ATE  SLAM {r['ate_slam']:.3f} m   odometry {r['ate_odom']:.3f} m   (rigidly aligned: "
          f"{r['ate_slam_aligned']:.3f} / {r['ate_odom_aligned']:.3f} m)")
    print(f"yaw RMSE SLAM {r['yaw_rmse_slam_deg']:.2f} deg   odometry {r['yaw_rmse_odom_deg']:.2f} deg; "
          f"final error {r['final_err_slam']:.3f} / {r['final_err_odom']:.3f} m; max {r['max_err_slam']:.3f} m")
    print(f"match time {r['match_ms_mean']:.2f} ms mean, {r['match_ms_p95']:.2f} ms p95; states {r['states']}")
    assert r["reached"] and not r["fell"]
    assert 15.0 <= r["route_m"] <= 25.0
    assert r["ate_odom"] >= 0.5
    assert r["ate_slam"] <= 0.10
    assert r["final_err_slam"] < 0.15
    assert r["yaw_rmse_slam_deg"] < 1.0


def test_slam_map_overlaps_ground_truth(episode):
    ms, mo, mr = episode["map_slam"], episode["map_odom"], episode["map_ref"]
    print(f"\nIoU cell-for-cell SLAM {ms['iou']:.3f} / odometry {mo['iou']:.3f} / true pose {mr['iou']:.3f}; "
          f"on the 0.1 m grid {ms['iou_coarse']:.3f} / {mo['iou_coarse']:.3f} / {mr['iou_coarse']:.3f}; "
          f"precision {ms['precision']:.3f} / {mo['precision']:.3f}")
    assert ms["precision"] > 0.9                    # occupied cells sit on real obstacles
    assert ms["iou_coarse"] > 0.6 * mr["iou_coarse"]  # close to what perfect localisation gives
    assert ms["iou"] > mo["iou"] + 0.05             # the odometry map is smeared away


def test_native_and_numpy_agree_on_a_recorded_walk(episode):
    log = dict(episode["log"])
    log["scans"] = log["scans"][:300]
    out = {}
    for use_native in ([True, False] if S.native_available() else [False]):
        slam = S.Slam2D(log["size"], 0.05, use_native=use_native)
        t = time.perf_counter()
        r = S.replay_slam(log, slam)
        out[use_native] = (np.array(slam.traj_est), r["match_ms_mean"], time.perf_counter() - t)
    for k, (_, ms, wall) in out.items():
        print(f"\n{'native' if k else 'numpy '}: {ms:.2f} ms per scan match, {wall:.1f} s for {len(log['scans'])} scans")
    if len(out) == 2:
        assert np.array_equal(out[True][0], out[False][0])
        assert out[True][1] < out[False][1]


def _corridor_xml():
    walls = [{"name": f"wall {i}", "shape": "box", "pos": [0.0, y], "size": [20.0, 0.1, 0.75], "yaw": 0.0}
             for i, y in enumerate((-1.6, 1.6))]
    return build_mjcf(HumanoidParams(), walls)


def test_feature_poor_corridor_degrades_gracefully():
    """Two long parallel walls: across-corridor position and heading are observable, the along-corridor
    position is not. SLAM must fix the former, follow odometry on the latter (no jumps) and say so."""
    r = S.run_slam_episode(_corridor_xml(), np.array([[0.0, 0.0], [5.0, 0.0]]), seconds=40.0, seed=0)
    TR, OD, ES = r["true"], r["odom"], r["slam"]
    err_s, err_o = ES[-1] - TR[-1], OD[-1] - TR[-1]
    print(f"\ncorridor: final error SLAM x {err_s[0]:+.3f} y {err_s[1]:+.3f} yaw {np.rad2deg(err_s[2]):+.2f} deg; "
          f"odometry x {err_o[0]:+.3f} y {err_o[1]:+.3f} yaw {np.rad2deg(S.wrap(err_o[2])):+.2f} deg; "
          f"states {r['states']}; max per-scan correction {r['max_correction']:.3f} m")
    assert r["reached"] and not r["fell"]
    assert abs(err_s[1]) < 0.05 and abs(err_o[1]) > 2 * abs(err_s[1])
    assert abs(np.rad2deg(S.wrap(err_s[2]))) < 1.0 < abs(np.rad2deg(S.wrap(err_o[2])))
    assert abs(err_s[0] - err_o[0]) < 0.08            # along the corridor it keeps to the odometry
    assert r["max_correction"] < 0.03                 # no jumps
    assert r["states"].get("degenerate", 0) > 0.8 * r["scans"]
    stats = r["slam_obj"].stats()
    ev, vec = np.linalg.eigh(np.array(stats["cov"])[:2, :2])
    assert np.sqrt(ev[1] / ev[0]) > 3.0                # elongated uncertainty ...
    assert abs(vec[0, 1]) > np.cos(np.deg2rad(20))     # ... along the corridor (x)
