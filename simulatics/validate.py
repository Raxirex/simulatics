"""Robustness sweep: does a behaviour survive when the physics is not exactly what it was tuned on?

Two families of variation, as is standard practice (domain randomisation;
checking a policy under a second engine or other solver settings before trusting it):

    random     the world and the robot are perturbed at random within stated ranges: floor friction, link
               masses, motor strength, servo stiffness, joint damping, a payload on the torso, and a shove
    settings   the same world under other numerical settings of the physics: half the time step, another
               integrator, another friction cone, another solver, more solver iterations, softer contacts

Each variant runs the task headless on its own copy of the model, with a fresh controller, and reports
whether it succeeded and how far it ended from where it should be. A behaviour that only works on the
nominal model is exploiting the simulator.

    from simulatics import validate
    report = validate.run("walk", suite="both", n=12, seed=0)
    print(validate.summary(report))

Tasks: "stand" (stay up for 6 s, with a shove in the random suite), "walk" (to a point 3 m ahead), "square"
(four waypoints), and "recording:<sample id>" (a recorded motion or trained skill: open loop, so expected to
be fragile; the sweep shows how fragile).
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

RANGES = {"friction": (0.5, 1.4), "mass": (0.85, 1.15), "motor strength": (0.8, 1.2), "servo stiffness": (0.8, 1.25),
          "joint damping": (0.6, 1.6), "payload kg": (0.0, 4.0), "shove N": (0.0, 60.0)}


@dataclass
class Variant:
    label: str
    changes: dict = field(default_factory=dict)       # what was changed, for the report
    apply: object = None                              # fn(model) -> None
    shove: tuple | None = None                        # (time s, force (3,), duration s)


def _scale_mass(m: mujoco.MjModel, k: float) -> None:
    robot = np.flatnonzero(m.body_rootid == m.body_rootid[1]) if m.nbody > 1 else np.zeros(0, int)
    m.body_mass[robot] *= k
    m.body_inertia[robot] *= k
    mujoco.mj_setConst(m, mujoco.MjData(m))


def random_variants(n: int, seed: int = 0) -> list[Variant]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        v = {k: float(rng.uniform(*r)) for k, r in RANGES.items()}
        ang = float(rng.uniform(0, 2 * np.pi))
        force = np.array([np.cos(ang), np.sin(ang), 0.0]) * v["shove N"]

        def apply(m, v=v):
            floor = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0]
            m.geom_friction[floor, 0] *= v["friction"]
            _scale_mass(m, v["mass"])
            m.body_mass[1] += v["payload kg"]
            m.actuator_forcerange[:] *= v["motor strength"]
            m.actuator_gainprm[:, 0] *= v["servo stiffness"]
            m.actuator_biasprm[:, 1] *= v["servo stiffness"]
            m.dof_damping[6:] *= v["joint damping"]
            mujoco.mj_setConst(m, mujoco.MjData(m))
        out.append(Variant(f"random {i + 1}", {k: round(x, 2) for k, x in v.items()}, apply, (1.5, force, 0.1)))
    return out


def setting_variants() -> list[Variant]:
    I, C, S = mujoco.mjtIntegrator, mujoco.mjtCone, mujoco.mjtSolver

    def opt(**kw):
        def apply(m):
            for k, x in kw.items():
                setattr(m.opt, k, x)
        return apply

    def soft(m):
        m.geom_solref[:, 0] = np.maximum(m.geom_solref[:, 0], 0.02) * 2.0     # softer contacts
    return [
        Variant("half time step", {"timestep": "x0.5"}, lambda m: setattr(m.opt, "timestep", m.opt.timestep * 0.5)),
        Variant("Euler integrator", {"integrator": "Euler"}, opt(integrator=int(I.mjINT_EULER))),
        Variant("RK4 integrator", {"integrator": "RK4"}, opt(integrator=int(I.mjINT_RK4))),
        Variant("elliptic friction cone", {"cone": "elliptic"}, opt(cone=int(C.mjCONE_ELLIPTIC))),
        Variant("CG solver", {"solver": "CG"}, opt(solver=int(S.mjSOL_CG))),
        Variant("more solver iterations", {"iterations": 200, "tolerance": 1e-10}, opt(iterations=200, tolerance=1e-10)),
        Variant("softer contacts", {"contact time constant": "x2"}, soft),
        Variant("impratio 5", {"impratio": 5}, opt(impratio=5.0)),
    ]


# ------------------------------------------------------------------------------------------------ episodes
def _scene(task: str) -> tuple[str, np.ndarray, dict]:
    """(xml, start qpos, info) for a task, built the way the live session builds it."""
    from . import samples
    from .server.session import SimSession
    s = SimSession()
    try:
        s.lidar_cfg.enabled = False
        info: dict = {}
        if task.startswith("recording:"):
            sid = task.split(":", 1)[1]
            from .imitate.physical import Tape
            phys, shipped = samples.physical(), {}
            try:
                from . import skills
                shipped = skills.shipped()
            except Exception:  # noqa: BLE001
                pass
            if sid in shipped:
                tape = Tape.load(skills.DATA / shipped[sid]["file"])
                samples._builtin(s, tape.meta.get("obstacles") or [], tape.meta.get("params") or None, tape.meta.get("props") or [])
            elif sid in phys and phys[sid].get("verified"):
                e = phys[sid]
                tape = Tape.load(samples.DATA / e["file"])
                samples._builtin(s, e.get("props") or [], e.get("params") or None)
            else:
                raise ValueError(f"no recorded motion called {sid!r}")
            info["tape"] = tape
        else:
            samples._builtin(s)
        with s.lock:
            return s.export_xml(), s.data.qpos.copy(), info
    finally:
        s.shutdown()


GOALS = {"walk": [(3.0, 0.0)], "square": [(1.5, 0.0), (1.5, 1.5), (0.0, 1.5), (0.0, 0.0)]}


def episode(xml: str, q0: np.ndarray, task: str, variant: Variant | None, info: dict, limit_s: float = 60.0) -> dict:
    """One headless run. Returns {"ok", "seconds", "error_m", "note"}."""
    from .control.walker import WalkController
    m = mujoco.MjModel.from_xml_string(xml)
    if variant is not None and variant.apply is not None:
        variant.apply(m)
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)
    z0 = float(d.qpos[2])

    def fallen() -> bool:
        return (not np.isfinite(d.qpos).all()) or d.qpos[2] < 0.6 * z0 or d.xmat[1].reshape(3, 3)[2, 2] < 0.35

    def shove() -> None:
        if variant is not None and variant.shove is not None:
            t, f, dur = variant.shove
            d.xfrc_applied[1, :3] = f if t <= d.time < t + dur else 0.0

    tape = info.get("tape")
    if tape is not None:                                    # open-loop recording, then balance takes over
        from .imitate.physical import prepare_model, restore_model
        saved = prepare_model(m)
        d.qpos[:] = tape.qpos0
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        z_start = float(d.qpos[2])
        for u in tape.ctrl:
            d.ctrl[:] = u
            mujoco.mj_step(m, d, int(round(tape.control_steps * tape.timestep / m.opt.timestep)))
            if not np.isfinite(d.qpos).all():
                return {"ok": False, "seconds": float(d.time), "error_m": float("nan"), "note": "the simulation blew up"}
        restore_model(m, saved)
        up = float(d.xmat[1].reshape(3, 3)[2, 2])
        err = float(np.linalg.norm(d.qpos[:3] - tape.states[-1][:3]))
        ok = up > 0.9 and abs(d.qpos[2] - tape.states[-1][2]) < 0.2
        return {"ok": bool(ok), "seconds": float(d.time), "error_m": err,
                "note": "ended standing where the recording ends" if ok else f"ended {err:.2f} m from the recording's end, upright {up:.2f}"}
    c = WalkController(m)
    c.reset(m, d)
    n = max(1, int(round(c.control_dt / m.opt.timestep)))
    goals = list(GOALS.get(task, []))
    target = np.array(goals[-1]) if goals else d.qpos[:2].copy()
    if goals:
        c.set_path(np.array([d.qpos[:2].copy(), *goals]))
    done_at = None
    while d.time < limit_s:
        shove()
        c.step(m, d)
        mujoco.mj_step(m, d, n)
        if fallen():
            return {"ok": False, "seconds": float(d.time), "error_m": float(np.linalg.norm(d.qpos[:2] - target)), "note": "fell"}
        mode = c.status().get("mode")
        if task == "stand":
            if d.time >= 6.0:
                break
        elif mode in ("done", "stand") and d.time > 1.0 and np.linalg.norm(d.qpos[:2] - target) < 0.3:
            done_at = done_at or d.time
            if d.time - done_at > 1.0:
                break
    err = float(np.linalg.norm(d.qpos[:2] - target))
    ok = err < (0.25 if task == "stand" else 0.1) and d.time < limit_s
    return {"ok": bool(ok), "seconds": float(d.time), "error_m": err,
            "note": "ok" if ok else ("ran out of time" if d.time >= limit_s else f"ended {err:.2f} m off")}


def run(task: str = "walk", suite: str = "both", n: int = 8, seed: int = 0, progress=None, cancel=None) -> dict:
    """Run a task on the nominal model and on every variant of the suite ("random", "settings" or "both")."""
    if not (task in ("stand", "walk", "square") or task.startswith("recording:")):
        raise ValueError("task must be stand, walk, square or recording:<sample id>")
    xml, q0, info = _scene(task)
    variants: list[Variant | None] = [None]
    if suite in ("random", "both"):
        variants += random_variants(n, seed)
    if suite in ("settings", "both"):
        variants += setting_variants()
    rows, t0 = [], time.perf_counter()
    for i, v in enumerate(variants):
        if cancel is not None and cancel.is_set():
            break
        try:
            r = episode(xml, q0, task, v, info)
        except Exception as e:  # noqa: BLE001 - a setting the model cannot run under is a result too
            r = {"ok": False, "seconds": 0.0, "error_m": float("nan"), "note": f"{type(e).__name__}: {e}"}
        rows.append({"variant": v.label if v else "nominal", "result": "pass" if r["ok"] else "FAIL",
                     "error_m": round(r["error_m"], 3) if np.isfinite(r["error_m"]) else None, "sim_seconds": round(r["seconds"], 1),
                     "note": r["note"], "changes": ", ".join(f"{k} {x}" for k, x in (v.changes.items() if v else []))})
        if progress:
            progress((i + 1) / len(variants), rows[-1])
    passed = sum(r["result"] == "pass" for r in rows)
    return {"task": task, "suite": suite, "rows": rows, "passed": passed, "total": len(rows),
            "nominal_ok": bool(rows and rows[0]["result"] == "pass"), "seconds": round(time.perf_counter() - t0, 1),
            "ranges": dict(RANGES) if suite in ("random", "both") else {}}


def summary(report: dict) -> str:
    worst = [r for r in report["rows"] if r["result"] != "pass"]
    s = f"{report['task']}: {report['passed']} of {report['total']} variants passed"
    if not report["nominal_ok"]:
        s += " (it fails on the unchanged model too)"
    elif worst:
        s += "; failed under: " + ", ".join(r["variant"] for r in worst[:6]) + (" ..." if len(worst) > 6 else "")
    return s
