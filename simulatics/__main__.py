"""Command line: `python -m simulatics [gui|train|bench|build-native]`."""
from __future__ import annotations

import argparse
import time


def cmd_gui(a) -> None:
    import uvicorn
    from .server.app import create_app
    url = f"http://{a.host}:{a.port}"
    print(f"Simulatics GUI: {url}")
    if not a.no_browser:
        import threading
        import webbrowser
        threading.Timer(1.0, webbrowser.open, [url]).start()
    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="warning")


def cmd_train(a) -> None:
    from .models import HumanoidParams
    from .train.evolve import Trainer, TrainConfig
    cfg = TrainConfig(algo=a.algo, population=a.population, generations=a.generations, evolve_body=a.evolve_body,
                      heading_range=a.heading_range, threads=a.threads, name=a.name, seed=a.seed,
                      episode_seconds=a.episode_seconds)
    tr = Trainer(cfg, HumanoidParams())
    tr.start()
    seen = 0
    try:
        while tr.state in ("idle", "running"):
            time.sleep(0.5)
            hist = tr.status()["history"]
            for h in hist[seen - len(hist):] if len(hist) > seen else []:
                print(f"gen {h['gen']:5d}  best {h['best']:8.2f}  mean {h['mean']:8.2f}  center {h['center']:8.2f}  "
                      f"{h['sps'] / 1000:6.0f}k steps/s")
            seen = max(seen, len(tr.history))
    except KeyboardInterrupt:
        tr.stop()
    tr.join()
    print(f"{tr.state}: {tr.error or tr.policy_path}")


def cmd_bench(a) -> None:
    import mujoco
    import numpy as np
    from mujoco import rollout
    from . import native
    from .models import HumanoidParams, build_mjcf
    from .sim.task import LocoTask
    print(native.status())
    m = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams()))
    s0 = LocoTask.initial_state(m)
    for nt in (1, 2, 4, 8, 16):
        with rollout.Rollout(nthread=nt) as r:
            datas = [mujoco.MjData(m) for _ in range(nt)]
            ctrl = np.zeros((a.batch, 1, m.nu))
            t = time.perf_counter()
            r.rollout(m, datas, np.tile(s0, (a.batch, 1)), ctrl, nstep=200)
            print(f"{nt:2d} threads: {a.batch * 200 / (time.perf_counter() - t) / 1000:7.0f}k physics steps/s")


def cmd_build(a) -> None:
    from . import native
    ok = native.build()
    print(native.status())
    raise SystemExit(0 if ok else 1)


def main() -> None:
    p = argparse.ArgumentParser(prog="simulatics")
    sub = p.add_subparsers(dest="cmd")
    g = sub.add_parser("gui", help="start the GUI (default)")
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--port", type=int, default=8765)
    g.add_argument("--no-browser", action="store_true")
    g.set_defaults(fn=cmd_gui)
    t = sub.add_parser("train", help="headless evolutionary training")
    t.add_argument("--algo", default="es", choices=["es", "ga"])
    t.add_argument("--population", type=int, default=256)
    t.add_argument("--generations", type=int, default=200)
    t.add_argument("--episode-seconds", type=float, default=8.0)
    t.add_argument("--evolve-body", action="store_true")
    t.add_argument("--heading-range", type=float, default=0.0)
    t.add_argument("--threads", type=int, default=0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--name", default="run")
    t.set_defaults(fn=cmd_train)
    b = sub.add_parser("bench", help="physics throughput vs thread count")
    b.add_argument("--batch", type=int, default=512)
    b.set_defaults(fn=cmd_bench)
    n = sub.add_parser("build-native", help="compile the C++ simcore library")
    n.set_defaults(fn=cmd_build)
    a = p.parse_args()
    if a.cmd is None:
        a = p.parse_args(["gui"])
    a.fn(a)


if __name__ == "__main__":
    main()
