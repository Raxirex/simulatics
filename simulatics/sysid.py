"""System identification: make a simulated motor behave like a measured one.

A motor on the test bench (robot on the stand, Motor Control panel) is driven with a known command and its
angle is recorded. Given such a recording, from the simulation or from a real robot, `fit_motor` adjusts the
joint's physical parameters until the simulation reproduces the recording:

    damping        viscous friction of the joint (Nm s/rad)
    armature       reflected rotor inertia (kg m^2)
    frictionloss   dry (Coulomb) friction (Nm)
    kp, kv         the servo's stiffness and damping, when asked for

It is a bounded nonlinear least-squares fit (scipy) in which every evaluation replays the recorded command in
the physics and compares angles. This is the step that most narrows the gap between simulation and hardware.

    rec = sysid.record(session, actuator)                 # what the bench just measured
    result = sysid.fit_motor(session, rec)                # -> fitted values, error before / after
    sysid.apply(session, actuator, result)

A recording is plain data: {"time": [...], "target": [...], "angle": [...]} (seconds, rad, rad) plus, for
bench recordings, the full start state. `load_csv` reads one from a file with those three columns, so a log
from a real motor can be fitted the same way.

`self_test` is the honest check that the method works: it changes the joint's true parameters, records a
response, forgets the changes, and fits them back.
"""
from __future__ import annotations

import copy
import csv
from pathlib import Path

import mujoco
import numpy as np

FIELDS = ("damping", "armature", "frictionloss", "kp", "kv")
BOUNDS = {"damping": (1e-3, 200.0), "armature": (1e-5, 5.0), "frictionloss": (0.0, 60.0), "kp": (1.0, 5e4), "kv": (0.0, 5e3)}
DEFAULT_FIT = ("damping", "armature", "frictionloss")


def _dof(model: mujoco.MjModel, actuator: int) -> tuple[int, int]:
    j = int(model.actuator_trnid[actuator, 0])
    return int(model.jnt_dofadr[j]), int(model.jnt_qposadr[j])


def get_params(model: mujoco.MjModel, actuator: int) -> dict:
    dof, _ = _dof(model, actuator)
    return {"damping": float(model.dof_damping[dof]), "armature": float(model.dof_armature[dof]),
            "frictionloss": float(model.dof_frictionloss[dof]), "kp": float(model.actuator_gainprm[actuator, 0]),
            "kv": float(-model.actuator_biasprm[actuator, 2])}


def set_params(model: mujoco.MjModel, actuator: int, values: dict) -> None:
    dof, _ = _dof(model, actuator)
    if "damping" in values:
        model.dof_damping[dof] = max(0.0, float(values["damping"]))
    if "armature" in values:
        model.dof_armature[dof] = max(0.0, float(values["armature"]))
    if "frictionloss" in values:
        model.dof_frictionloss[dof] = max(0.0, float(values["frictionloss"]))
    if "kp" in values:
        model.actuator_gainprm[actuator, 0] = float(values["kp"])
        model.actuator_biasprm[actuator, 1] = -float(values["kp"])
    if "kv" in values:
        model.actuator_biasprm[actuator, 2] = -float(values["kv"])


def record(session, actuator: int | None = None) -> dict:
    """The bench recording the session holds for the traced motor (Motor Control: run a signal first)."""
    with session.lock:
        start = getattr(session, "motor_trace_start", None)
        tr = np.array(session.motor_trace)
        if start is None or len(tr) < 10:
            raise ValueError("no recording yet: put the robot on the stand and run a test signal on a motor first")
        if actuator is not None and int(actuator) != int(start["actuator"]):
            raise ValueError("the recording is for a different motor: run a signal on this one first")
        return {"actuator": int(start["actuator"]), "name": session.model.actuator(int(start["actuator"])).name,
                "time": tr[:, 0] - tr[0, 0], "target": tr[:, 1].copy(), "angle": tr[:, 2].copy(),
                "qpos0": start["qpos"].copy(), "qvel0": start["qvel"].copy(), "ctrl0": start["ctrl"].copy(),
                "dt": float(np.median(np.diff(tr[:, 0])))}


def load_csv(path: str | Path) -> dict:
    """A recording from a file with columns time, target, angle (seconds, rad, rad; `angle_deg` / `target_deg`
    columns are accepted too)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("the file has no rows")
    cols = {k.strip().lower(): k for k in rows[0]}

    def col(name: str) -> np.ndarray:
        if name in cols:
            return np.array([float(r[cols[name]]) for r in rows])
        if f"{name}_deg" in cols:
            return np.deg2rad([float(r[cols[f"{name}_deg"]]) for r in rows])
        raise ValueError(f"the file needs the columns time, target, angle (missing: {name})")
    t = col("time")
    return {"time": t - t[0], "target": col("target"), "angle": col("angle"), "dt": float(np.median(np.diff(t)))}


def save_csv(rec: dict, path: str | Path) -> Path:
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "target", "angle"])
        for row in zip(rec["time"], rec["target"], rec["angle"]):
            w.writerow([f"{v:.6f}" for v in row])
    return path


def simulate(model: mujoco.MjModel, actuator: int, rec: dict) -> np.ndarray:
    """The joint angle the model produces for the recorded command, sampled like the recording."""
    d = mujoco.MjData(model)
    _, qadr = _dof(model, actuator)
    if rec.get("qpos0") is not None and len(rec["qpos0"]) == model.nq:
        d.qpos[:] = rec["qpos0"]
        d.qvel[:] = rec["qvel0"]
        base = np.array(rec["ctrl0"], float)
    else:                                   # a recording from elsewhere: start at rest at its first angle
        mujoco.mj_resetData(model, d)
        d.qpos[qadr] = rec["angle"][0]
        base = d.qpos[[model.jnt_qposadr[model.actuator_trnid[a, 0]] for a in range(model.nu)]].copy()
    mujoco.mj_forward(model, d)
    sub = max(1, int(round(rec["dt"] / model.opt.timestep)))
    out = np.zeros(len(rec["target"]))
    for i, u in enumerate(rec["target"]):
        d.ctrl[:] = base
        d.ctrl[actuator] = u
        mujoco.mj_step(model, d, sub)
        out[i] = d.qpos[qadr]
    return out


def fit_motor(session_or_model, rec: dict, fields: tuple[str, ...] = DEFAULT_FIT, actuator: int | None = None) -> dict:
    """Fit the chosen parameters of one motor so the simulation reproduces `rec`.

    Returns {"fitted": {field: value}, "before": {field: value}, "rms_before", "rms_after" (rad),
    "improved": bool, "simulated": angle array after the fit, "evaluations": int}."""
    from scipy.optimize import least_squares
    model = session_or_model if isinstance(session_or_model, mujoco.MjModel) else None
    if model is None:
        with session_or_model.lock:
            model = copy.copy(session_or_model.model)
    else:
        model = copy.copy(model)
    a = int(rec["actuator"] if actuator is None else actuator)
    fields = tuple(f for f in fields if f in FIELDS)
    if not fields:
        raise ValueError(f"choose at least one of {FIELDS} to fit")
    before = get_params(model, a)
    target = np.asarray(rec["angle"], float)
    lo = np.array([BOUNDS[f][0] for f in fields])
    hi = np.array([BOUNDS[f][1] for f in fields])
    # fit in a squashed space so parameters of very different size are treated alike
    scale = np.array([max(before[f], 0.05 * (BOUNDS[f][1] ** 0.5), 1e-3) for f in fields])
    x0 = np.clip(np.array([before[f] for f in fields]), lo + 1e-9, hi) / scale
    n = [0]

    def residual(x):
        n[0] += 1
        set_params(model, a, dict(zip(fields, np.clip(x * scale, lo, hi))))
        return simulate(model, a, rec) - target

    r0 = residual(x0)
    sol = least_squares(residual, x0, bounds=(lo / scale, hi / scale), diff_step=0.05, xtol=1e-6, ftol=1e-8, max_nfev=120)
    fitted = dict(zip(fields, (float(v) for v in np.clip(sol.x * scale, lo, hi))))
    set_params(model, a, fitted)
    sim = simulate(model, a, rec)
    rms0, rms1 = float(np.sqrt(np.mean(r0 ** 2))), float(np.sqrt(np.mean((sim - target) ** 2)))
    return {"actuator": a, "fields": list(fields), "fitted": fitted, "before": {f: before[f] for f in fields},
            "rms_before": rms0, "rms_after": rms1, "improved": rms1 < rms0, "simulated": sim, "evaluations": n[0]}


def apply(session, actuator: int, result: dict) -> None:
    """Put fitted values into the live robot (they are kept across rebuilds, like any motor edit)."""
    with session.lock:
        m = session.model
        dof, _ = _dof(m, actuator)
        for f, v in result["fitted"].items():
            if f in ("damping", "armature", "kp", "kv"):
                session.set_actuator(actuator, f, v)
            elif f == "frictionloss":
                m.dof_frictionloss[dof] = v
                session.motor_overrides.setdefault(m.actuator(actuator).name, {})["frictionloss"] = float(v)


def self_test(session, actuator: int, hidden: dict | None = None, signal: str = "chirp", amplitude: float = 0.35,
              seconds: float = 4.0, noise: float = 0.0, seed: int = 0) -> dict:
    """Hide a change, measure, fit it back. Needs the robot on the stand (manual control).

    The joint's true parameters are multiplied by `hidden` (default: damping x3, armature x4, dry friction +2 Nm),
    a test signal is recorded with those, the model is put back as it was, and the fit must find them.
    Returns the fit result plus "truth" and "relative_error" per field."""
    import time
    hidden = hidden or {"damping": 3.0, "armature": 4.0, "frictionloss": 2.0}
    with session.lock:
        if getattr(session.controller, "name", "") != "manual":
            raise ValueError("put the robot on the stand first (Motor Control: hold the robot on a stand)")
        m = session.model
        original = get_params(m, actuator)
        truth = dict(original)
        for f, k in hidden.items():
            truth[f] = original[f] + k if f == "frictionloss" else original[f] * k
        set_params(m, actuator, truth)
        speed, session.speed = session.speed, 8.0
    try:
        session.trace_motor(actuator)
        session.set_motor_signal(actuator, signal, amplitude, 3.0)
        t0 = float(session.data.time)
        deadline = time.monotonic() + 60
        while float(session.data.time) - t0 < seconds and time.monotonic() < deadline:
            time.sleep(0.02)
        rec = record(session, actuator)
        session.set_motor_signal(actuator, "hold")
    finally:
        with session.lock:
            set_params(session.model, actuator, original)        # forget the hidden change
            session.speed = speed
    if noise > 0:
        rec["angle"] = rec["angle"] + noise * np.random.default_rng(seed).standard_normal(len(rec["angle"]))
    out = fit_motor(session, rec, tuple(hidden))
    out["truth"] = {f: truth[f] for f in hidden}
    out["relative_error"] = {f: abs(out["fitted"][f] - truth[f]) / max(abs(truth[f]), 1e-9) for f in hidden}
    out["recording"] = rec
    return out
