"""MPC walker (control/mpc.py, mpc_lipm.py, mpc_qp.py, mpc_bench.py): solver, formulation and acceptance runs.

Walking tests are single-threaded headless episodes of a few simulated seconds (2-6 s wall each). The push
benchmark at the end runs 2 x 24 episodes (about 3 minutes); set SIMULATICS_SKIP_PUSH_BENCH=1 to skip it.
"""
import os

import mujoco
import numpy as np
import pytest

from simulatics.control.mpc import MPCParams, MPCWalkController, simulate_mpc_walk
from simulatics.control.mpc_bench import CONDITIONS, push_schedule, push_table
from simulatics.control.mpc_body import Meas
from simulatics.control.mpc_lipm import Foot, LipmMPC, Problem, StepBox, Support
from simulatics.control.mpc_qp import INFEASIBLE, OK, native_available, solve_qp
from simulatics.control.walk_footsteps import LEFT, RIGHT
from simulatics.models import HumanoidParams, build_mjcf
from simulatics.plan.grid import GridMap

BODIES = {
    "default": {},
    "short legs": {"thigh_length": 0.32, "shin_length": 0.32},
    "long legs, wide hips": {"thigh_length": 0.5, "shin_length": 0.5, "pelvis_width": 0.26},
}
ACCURACY_M, ACCURACY_DEG = 0.03, 5.0


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


def accurate(r) -> bool:
    return (r["reached"] and not r["fell"] and r["error_m"] <= ACCURACY_M
            and abs(r["heading_error_deg"]) <= ACCURACY_DEG)


# ------------------------------------------------------------------ QP solver
def _random_qp(rng, n, m):
    M = rng.normal(size=(n, n))
    H = M @ M.T + 1e-2 * np.eye(n)
    g = rng.normal(size=n) * 3
    C = rng.normal(size=(m, n))
    b = C @ (rng.normal(size=n) * 0.1) + rng.uniform(0, 1, m)          # feasible by construction
    return H, g, C, b


@pytest.mark.parametrize("force_python", [True, False])
def test_qp_solution_satisfies_kkt(force_python):
    if not force_python and not native_available():
        pytest.skip("native mpc kernel not built (native.build(target='mpc'))")
    rng = np.random.default_rng(1)
    for _ in range(100):
        H, g, C, b = _random_qp(rng, int(rng.integers(2, 40)), int(rng.integers(0, 120)))
        r = solve_qp(H, g, C, b, force_python=force_python)
        assert r.status == OK and r.native == (not force_python)
        assert (C @ r.x - b).max(initial=-1.0) < 1e-7                              # primal feasible
        assert (r.mult >= -1e-12).all()                                            # dual feasible
        grad = H @ r.x + g + (C[r.active].T @ r.mult if len(r.active) else 0.0)  # stationary
        assert np.abs(grad).max() < 1e-6 * max(1.0, np.abs(g).max())
        assert np.abs(C[r.active] @ r.x - b[r.active]).max(initial=0.0) < 1e-7    # active = tight
        warm = solve_qp(H, g, C, b, warm=r.active[::-1], force_python=force_python)
        assert np.abs(warm.x - r.x).max() < 1e-7                                  # warm start: same optimum


def test_qp_numpy_and_native_agree():
    if not native_available():
        pytest.skip("native mpc kernel not built")
    rng = np.random.default_rng(2)
    for _ in range(100):
        H, g, C, b = _random_qp(rng, int(rng.integers(2, 40)), int(rng.integers(1, 120)))
        a, n = solve_qp(H, g, C, b, force_python=True), solve_qp(H, g, C, b)
        assert np.abs(a.x - n.x).max() < 1e-9 and a.iterations == n.iterations
        assert sorted(a.active.tolist()) == sorted(n.active.tolist())


def test_qp_reports_infeasibility():
    H, g = np.eye(2), np.zeros(2)
    C, b = np.array([[1.0, 0.0], [-1.0, 0.0]]), np.array([-1.0, -1.0])          # x <= -1 and x >= 1
    for fp in (True, False):
        if not fp and not native_available():
            continue
        assert solve_qp(H, g, C, b, force_python=fp).status == INFEASIBLE


# ------------------------------------------------------------ MPC formulation
def test_lipm_mpc_holds_the_zmp_on_the_sole_and_moves_the_footstep_after_a_push():
    """Single support on the right sole, the left landing point free: an outward CoM velocity must move the
    footstep outwards and keep the ZMP inside the rotated sole at every sample."""
    mpc = LipmMPC(0.9, half_length=0.08, half_width=0.03)
    yaw = 0.3
    right = Foot(RIGHT, np.array([0.0, -0.1]), yaw)
    left = Foot(LEFT, np.array([-0.06, 0.09]), yaw, var=0, key=7)              # the planned step
    feet = [right, left]
    sup = [Support("single", (0,), ((0, 1.0),)) if k < 5 else Support("single", (1,), ((1, 1.0),))
           for k in range(16)]
    plans = {}
    for vy in (0.0, 0.3):
        s0 = np.array([[0.0, -0.08], [0.0, vy], [0.0, 0.0]])
        plans[vy] = p = mpc.solve(Problem(s0, feet, sup, [StepBox(0, 0, LEFT)]))
        assert p.status == OK
        c, s = np.cos(yaw), np.sin(yaw)
        for k in range(16):
            f = right.xy if k < 5 else p.steps[0]
            rel = p.zmp[k] - f
            assert abs(c * rel[0] + s * rel[1]) <= 0.08 + 1e-6 and abs(-s * rel[0] + c * rel[1]) <= 0.03 + 1e-6
    assert plans[0.3].steps[0][1] - plans[0.0].steps[0][1] > 0.05                  # stepped outwards (left)


def test_mpc_on_an_ideal_pendulum_keeps_to_the_footstep_plan():
    """Closed loop on a perfect LIPM (the CoM follows the plan, feet land on target): the formulation is
    consistent with the planner's steps (small adaptation, every QP optimal) and stops on the goal stance."""
    m, d = make()
    for goal, yaw in (((3.0, 0.0), None), ((1.0, 0.0), np.pi / 2), ((-0.5, 0.0), 0.0)):
        c = MPCWalkController(m)
        c.reset(m, d)
        com, vel = np.array([0.044, 0.0]), np.zeros(2)
        c.set_path(np.array([com, goal]), yaw)
        worst, bad = 0.0, 0
        for i in range(int(25 / c.control_dt)):
            feet = {s: c.feet[s].copy() for s in (LEFT, RIGHT)}
            if c.swing is not None and c.swing.left <= 1:
                feet[c.swing.side] = c.swing.target.copy()
            meas = Meas(com=np.array([*com, c.z_c]), com_vel=np.array([*vel, 0.0]), R_torso=np.eye(3),
                        p_pelvis=np.zeros(3), R_pelvis=np.eye(3), feet=feet)
            c.ticks += 1
            c._advance_phase(meas)
            if c._need_solve or c.plan is None or (c.ticks - c.plan_tick) >= c.n_mpc:
                c._solve(meas, c.ticks * c.control_dt)
                bad += c.plan.status != OK
            st = c.plan.state((c.ticks - c.plan_tick + 1) * c.control_dt)
            com, vel = st[0].copy(), st[1].copy()
            if c.swing is not None:
                c._swing_pose(c.swing)
                worst = max(worst, float(np.linalg.norm(c.swing.target[:2] - c.swing.plan[:2])))
            if c.swing is None and not c.queue and c.phase_left <= 0 and i * c.control_dt > 3:
                break
        mid = (c.feet[LEFT][:2] + c.feet[RIGHT][:2]) / 2
        assert bad == 0 and worst < 0.03 and c.retimed == 0
        assert np.linalg.norm(mid - np.asarray(goal)) < 0.05      # (plus the torso offset, ~2 cm)


# ---------------------------------------------------------------- behaviour
def test_supports_only_the_builtin_leg_structure_and_api():
    blob = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint/><geom size='0.1'/></body></worldbody></mujoco>")
    assert not MPCWalkController.supports(blob)
    with pytest.raises(ValueError):
        MPCWalkController(blob)
    m, d = make()
    c = MPCWalkController(m)
    assert c.name == "mpc" and abs(c.control_dt - 0.005) < 1e-12 and MPCWalkController.supports(m)
    c.reset(m, d)
    with pytest.raises(ValueError):
        c.set_path(np.array([[0.0, 0.0], [np.inf, 1.0]]))
    with pytest.raises(ValueError):
        c.set_velocity(float("nan"), 0.0, 0.0)


def test_stands_ten_seconds_and_leaves_the_arms_alone():
    m, d = make()
    c = MPCWalkController(m)
    c.reset(m, d)
    arms = [i for i in range(m.nu) if i not in set(c.act.tolist())]
    d.ctrl[arms] = 0.123
    z0 = d.qpos[2]
    s = run(c, m, d, 10.0)
    assert s["mode"] == "stand" and s["support"] == "double"
    assert abs(d.qpos[2] - z0) < 0.02 and np.linalg.norm(d.qvel[:3]) < 0.05
    assert np.allclose(d.ctrl[arms], 0.123)                  # arm / hand actuators belong to someone else
    assert set(s) >= {"mode", "message", "goal_error_m", "heading_error_deg", "footsteps", "support", "mpc"}
    assert set(s["mpc"]) >= {"solve_ms", "iterations", "horizon_s"} and abs(s["mpc"]["horizon_s"] - 1.6) < 1e-9


def test_walks_to_3_0_and_stops_on_it():
    m, _ = make()
    r = simulate_mpc_walk(m, (3.0, 0.0), timeout=40)
    assert accurate(r), r
    assert r["peak_torque_frac"] <= 1.0 and 0 < r["cost_of_transport"] < 5
    assert r["mpc"]["qp_failures"] == 0 and r["mpc"]["solves"] > 50


def test_walks_to_2_2_ending_at_90_deg():
    m, _ = make()
    r = simulate_mpc_walk(m, (2.0, 2.0), np.pi / 2, timeout=40)
    assert accurate(r), r


def test_follows_astar_path_around_an_obstacle():
    obst = [{"pos": [2.0, 0.0], "size": [0.3, 1.0, 0.5], "yaw": 0.0}]
    m, d = make(obstacles=obst)
    grid = GridMap()
    grid.rasterize_truth(m, d)
    path = grid.plan([0.0, 0.0], [4.0, 0.0], robot_radius=0.35)
    assert len(path) >= 3 and np.abs(path[:, 1]).max() > 1.0
    r = simulate_mpc_walk(m, (4.0, 0.0), timeout=60, path=path)
    assert accurate(r), r


@pytest.mark.parametrize("body", list(BODIES))
def test_different_bodies_walk_without_retuning(body):
    m, _ = make(BODIES[body])
    r = simulate_mpc_walk(m, (1.5, -1.0), np.deg2rad(-45), timeout=40)
    assert accurate(r), r


def test_turn_in_place_sidestep_and_backwards():
    m, _ = make()
    for goal, yaw in (((0.04, 0.0), np.pi), ((0.04, 0.4), 0.0), ((-0.5, 0.0), 0.0)):
        r = simulate_mpc_walk(m, goal, yaw, timeout=40)
        assert accurate(r), (goal, r)


def test_replan_mid_swing_stop_and_velocity_mode():
    m, d = make()
    c = MPCWalkController(m)
    c.reset(m, d)
    c.set_path(np.array([[0.0, 0.0], [3.0, 0.0]]), None)
    run(c, m, d, 3.0, stop_when_done=False)
    while c.swing is None:                                   # re-plan while a foot is in the air
        run(c, m, d, c.control_dt, stop_when_done=False)
    assert c.mode == "walk" and len(c.status()["footsteps"]) > 0
    x, y = d.qpos[0], d.qpos[1]
    c.set_path(np.array([[x, y], [x, 1.0]]), 0.0)
    s = run(c, m, d, 30.0)
    assert s["mode"] == "done" and s["goal_error_m"] < ACCURACY_M
    c.set_velocity(0.2, 0.0, 0.0)
    x0 = d.qpos[0]
    run(c, m, d, 4.0, stop_when_done=False)
    assert c.mode == "walk" and d.qpos[0] - x0 > 0.3
    c.set_path(None)
    s = run(c, m, d, 3.0, stop_when_done=False)
    assert s["mode"] == "stand" and s["support"] == "double" and not s["footsteps"]
    assert np.linalg.norm(d.qvel[:2]) < 0.05


def test_numpy_fallback_walks_the_same_and_solve_times():
    """The numpy QP gives the same walk as the C++ kernel; per-update solve times (problem build + QP)."""
    m, _ = make()
    r_np = simulate_mpc_walk(m, (1.0, 0.0), timeout=30, params=MPCParams(qp_backend="numpy"))
    assert accurate(r_np) and not r_np["mpc"]["native"]
    print(f"\nnumpy QP: solve {r_np['mpc']['solve_ms_mean']:.2f} ms mean / {r_np['mpc']['solve_ms_p95']:.2f} p95 / "
          f"{r_np['mpc']['solve_ms_max']:.2f} max, {r_np['mpc']['iterations_mean']:.1f} iterations mean")
    assert r_np["mpc"]["solve_ms_p95"] < 200.0          # wall clock on a shared CPU: only a sanity bound
    if native_available():
        r = simulate_mpc_walk(m, (1.0, 0.0), timeout=30)
        assert r["mpc"]["native"] and abs(r["error_m"] - r_np["error_m"]) < 0.005
        print(f"C++ QP:   solve {r['mpc']['solve_ms_mean']:.2f} ms mean / {r['mpc']['solve_ms_p95']:.2f} p95 / "
              f"{r['mpc']['solve_ms_max']:.2f} max")
        assert r["mpc"]["solve_ms_p95"] < 200.0


# ------------------------------------------------------------------ push recovery
def test_push_formats():
    from simulatics.control.mpc import _push_list
    a = _push_list({"t": 1.0, "force": (0, 60, 0)})
    b = _push_list((1.0, (0.0, 60.0, 0.0), 0.1))
    c = _push_list([(1.0, (0.0, 60.0), 0.1)])
    assert a[0]["t"] == b[0]["t"] == c[0]["t"] == 1.0 and np.allclose(a[0]["force"], c[0]["force"])
    with pytest.raises(ValueError):
        _push_list((np.nan, (0, 1, 0), 0.1))


def test_push_schedule_is_deterministic():
    a = push_schedule(1, 8, (0.0, 60.0, 0.0), True)
    assert a == push_schedule(1, 8, (0.0, 60.0, 0.0), True) and len(a) == 8
    assert all(1.5 <= p["t"] <= 6.0 and p["duration"] == 0.1 and abs(abs(p["force"][1]) - 60) < 1e-9 for p in a)


@pytest.mark.skipif(os.environ.get("SIMULATICS_SKIP_PUSH_BENCH") == "1", reason="SIMULATICS_SKIP_PUSH_BENCH=1")
def test_push_recovery_against_the_walker():
    """8 seeded 0.1 s torso pushes per condition while walking 2.5 m, identical for both controllers.
    Floors are for the MPC only (measured: see the report); the walker's numbers are printed, not asserted,
    because control/walker.py is under parallel development."""
    m, _ = make()
    ctrls = {"mpc": lambda mm: MPCWalkController(mm)}
    try:
        from simulatics.control.walker import WalkController
        WalkController(m)
        ctrls["walker"] = lambda mm: WalkController(mm)
    except Exception as e:  # noqa: BLE001 - a half-edited walker must not fail the MPC's tests
        print(f"\nWalkController unavailable: {type(e).__name__}: {e}")
    table = push_table(m, ctrls, n=8)
    print()
    for cond, res in table.items():
        print(f"{cond:14s} " + "  ".join(f"{name} {sum(r['ok'] for r in rows)}/8 (falls {sum(r['fell'] for r in rows)})"
                                          for name, rows in res.items()))
    ok = {cond: sum(r["ok"] for r in res["mpc"]) for cond, res in table.items()}
    assert set(ok) == set(CONDITIONS)
    assert ok["forward 100 N"] >= 7 and ok["side 60 N"] >= 4         # measured 8/8 and 5/8
