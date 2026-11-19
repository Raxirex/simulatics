"""Push-recovery benchmark: the same seeded pushes for any walking controller.

Protocol (used by tests/test_mpc.py and the MPC report):
* a fresh robot stands 0.5 s, then walks straight to `goal` (default 2.5 m ahead);
* one 0.1 s force on the torso centre of mass (MuJoCo xfrc_applied) at a random time in [t_min, t_max] s
  after the walk command, while walking; sideways pushes get a random sign;
* success = the controller ends "done" (stopped) within `tolerance` of the goal without falling.

`push_schedule(seed, n, force)` draws the (time, force) list once, so every controller sees exactly the same
pushes. Conditions of the report: 60 N and 100 N sideways, 100 N forward, 8 pushes each, seeds 1, 2, 3.
"""
from __future__ import annotations

import time
from typing import Callable

import mujoco
import numpy as np

from .mpc import run_episode

CONDITIONS = {"side 60 N": ((0.0, 60.0, 0.0), True, 1),
              "side 100 N": ((0.0, 100.0, 0.0), True, 2),
              "forward 100 N": ((100.0, 0.0, 0.0), False, 3)}


def push_schedule(seed: int, n: int, force, random_sign: bool, t_min: float = 1.5, t_max: float = 6.0,
                  duration: float = 0.1) -> list[dict]:
    """n pushes {"t", "force", "duration"} with onset times uniform in [t_min, t_max] (s after the walk command)
    and, if random_sign, a random sign of the force."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        t = float(rng.uniform(t_min, t_max))
        sign = float(rng.choice([-1.0, 1.0])) if random_sign else 1.0
        out.append({"t": t, "force": (sign * np.asarray(force, float)).tolist(), "duration": duration})
    return out


def push_trials(model: mujoco.MjModel, make_controller: Callable, pushes: list[dict], goal=(2.5, 0.0),
                timeout: float = 25.0, tolerance: float = 0.1) -> list[dict]:
    """Run one episode per push; returns [{"push", "ok", "fell", "error_m", "reached", "wall_s"}]."""
    rows = []
    for p in pushes:
        t0 = time.perf_counter()
        r = run_episode(model, make_controller(model), goal, None, timeout, None, p, tolerance)
        rows.append({"push": p, "ok": bool(r["reached"] and not r["fell"]), "fell": bool(r["fell"]),
                     "error_m": float(r["error_m"]), "reached": bool(r["reached"]),
                     "wall_s": time.perf_counter() - t0})
    return rows


def push_table(model: mujoco.MjModel, controllers: dict, n: int = 8, conditions: dict | None = None,
               seed_offset: int = 0, **kw) -> dict:
    """{condition: {controller name: rows}} for controllers = {name: make_controller(model)}."""
    out = {}
    for cond, (force, signed, seed) in (conditions or CONDITIONS).items():
        pushes = push_schedule(seed + seed_offset, n, force, signed)
        out[cond] = {name: push_trials(model, mk, pushes, **kw) for name, mk in controllers.items()}
    return out
