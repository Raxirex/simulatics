"""Compare candidate policies: nominal goal trials (30, seed 1) and randomised physics + pushes (21 trials)."""
import sys, json, time
import numpy as np
import mujoco
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))  # repository root
from threadpoolctl import threadpool_limits
threadpool_limits(1)
from simulatics.models import HumanoidParams, build_mjcf
from simulatics.sim.vel_task import make_variants, VelocityTask
from simulatics.train.ppo import RLPolicy, evaluate_goals

m = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams()))
out = {}
for path in sys.argv[1:]:
    t0 = time.perf_counter()
    pol = RLPolicy.load(path)
    nom = evaluate_goals(pol, m, trials=30, seed=1)
    cfg = pol.task_config()
    variants, props = make_variants(m, cfg, np.random.default_rng(123), VelocityTask(m, cfg))
    rows = []
    for k in range(1, len(variants)):
        r = evaluate_goals(pol, variants[k], trials=3, seed=100 + k, push_vel=0.1, push_interval=4.0)
        rows += r["trials"]
    errs = np.array([r["final_error_m"] for r in rows if not r["fell"]])
    dr = {"success_rate": float(np.mean([r["success"] for r in rows])), "falls": int(sum(r["fell"] for r in rows)),
          "n": len(rows), "error_median_m": float(np.median(errs)) if len(errs) else None,
          "error_p90_m": float(np.percentile(errs, 90)) if len(errs) else None}
    out[path] = {"nominal": {k: v for k, v in nom.items() if k != "trials"}, "nominal_trials": nom["trials"],
                 "dr_push": dr, "dr_props": props.tolist()}
    n = nom
    f3 = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731  (None when every trial fell)
    print(f"{path}: nominal success {n['success_rate']:.0%} falls {n['falls']}/30 err med {f3(n['error_median_m'])} "
          f"p90 {f3(n['error_p90_m'])} max {f3(n['error_max_m'])} | DR+push success {dr['success_rate']:.0%} "
          f"falls {dr['falls']}/{dr['n']} err med {f3(dr['error_median_m'])} | {time.perf_counter() - t0:.0f} s",
          flush=True)
with open(sys.argv[1] + ".compare.json", "w") as f:
    json.dump(out, f, indent=1)

