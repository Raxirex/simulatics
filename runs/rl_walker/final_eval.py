"""Held-out evaluation of runs/rl_walker/rl_policy.npz (seeds not used for choosing the snapshot).

    .venv\\Scripts\\python.exe runs\\rl_walker\\final_eval.py [policy.npz] [out.json]

1. 40 goal trials 2-4 m away in any direction from a random start heading, through RLController + its
   PathFollower on the nominal robot (the GUI's click-to-walk path), 2 s held after "done".
2. 42 goal trials on 7 randomised models (friction / mass / servo stiffness from the training ranges) with
   a random 0.1 m/s push every 4 s.
3. Velocity tracking and turning in place through RLController.set_velocity (10 s each, measured after 3 s).
4. Long idle: 6 goal trials holding 30 s after arrival (stepping in place + position hold).
5. Controller cost per step (observation + MLP + servo targets), single thread.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root
import mujoco  # noqa: E402
import numpy as np
from threadpoolctl import threadpool_limits

threadpool_limits(1)
from simulatics.models import HumanoidParams, build_mjcf  # noqa: E402
from simulatics.sim.vel_task import VelocityTask, make_variants  # noqa: E402
from simulatics.train.ppo import RLController, RLPolicy, evaluate_commands, evaluate_goals  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else r"runs\rl_walker\rl_policy.npz"
out_path = sys.argv[2] if len(sys.argv) > 2 else r"runs\rl_walker\eval.json"
pol = RLPolicy.load(path)
m = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams()))
res = {"policy": path}

nom = evaluate_goals(pol, m, trials=40, seed=2024)
res["goals_nominal"] = nom
cfg = pol.task_config()
variants, props = make_variants(m, cfg, np.random.default_rng(31337), VelocityTask(m, cfg))
rows = []
for k in range(1, len(variants)):
    rows += evaluate_goals(pol, variants[k], trials=6, seed=500 + k, push_vel=0.1, push_interval=4.0)["trials"]
errs = np.array([r["final_error_m"] for r in rows if not r["fell"]])
res["goals_dr_push"] = {"trials": rows, "n": len(rows), "success_rate": float(np.mean([r["success"] for r in rows])),
                        "falls": int(sum(r["fell"] for r in rows)), "error_median_m": float(np.median(errs)),
                        "error_p90_m": float(np.percentile(errs, 90)), "error_max_m": float(errs.max()),
                        "variants": props.tolist()}
res["commands"] = evaluate_commands(pol, m)
res["long_idle"] = evaluate_goals(pol, m, trials=6, seed=99, hold=30.0)

ctrl = RLController(m, pol)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
ctrl.reset(m, d)
ctrl.set_velocity(0.3, 0.0, 0.0)
ts = []
for _ in range(1000):
    t0 = time.perf_counter()
    ctrl.step(m, d)
    ts.append(time.perf_counter() - t0)
    mujoco.mj_step(m, d, 2)
res["controller_ms_per_step"] = {"mean": 1e3 * float(np.mean(ts)), "p99": 1e3 * float(np.percentile(ts, 99)),
                                 "control_period_ms": 1e3 * ctrl.control_dt}

with open(out_path, "w") as f:
    json.dump(res, f, indent=1)

g, r = res["goals_nominal"], res["goals_dr_push"]
print(f"goals nominal : success {g['success_rate']:.0%} ({sum(t['success'] for t in g['trials'])}/40), falls {g['falls']}, "
      f"error median {g['error_median_m']:.3f} p90 {g['error_p90_m']:.3f} max {g['error_max_m']:.3f} m, "
      f"mean time {g['time_mean_s']:.1f} s, realtime x{g['realtime_factor']:.1f}")
print(f"goals DR+push : success {r['success_rate']:.0%} ({sum(t['success'] for t in r['trials'])}/{r['n']}), falls {r['falls']}, "
      f"error median {r['error_median_m']:.3f} p90 {r['error_p90_m']:.3f} max {r['error_max_m']:.3f} m")
li = res["long_idle"]
print(f"long idle 30 s: falls {li['falls']}/6, error median {li['error_median_m']:.3f} max {li['error_max_m']:.3f} m")
for c in res["commands"]:
    print(f"cmd {c['clamped']}: vx {c['vx']:+.3f} vy {c['vy']:+.3f} wz {c['wz']:+.3f} | err ({c['vx_err']:+.3f}, "
          f"{c['vy_err']:+.3f}, {c['wz_err']:+.3f}) drift {c['drift_m'] if c['drift_m'] is None else round(c['drift_m'], 3)} "
          f"fell {c['fell']}")
cm = res["controller_ms_per_step"]
print(f"controller: {cm['mean']:.3f} ms/step mean, {cm['p99']:.3f} ms p99, period {cm['control_period_ms']:.0f} ms")
