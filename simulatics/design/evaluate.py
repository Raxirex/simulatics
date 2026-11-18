"""Score a body design (HumanoidParams) by simulating it. Higher scores are better.

Walk mode (default when simulatics.control.walker imports and supports the model): the model-based
WalkController walks the robot headless to each goal with `simulate_walk`, which reports whether the
goal was reached, the final position error, a fall flag, the cost of transport
CoT = sum |tau qdot| dt / (m g distance) and the peak actuator torque as a fraction of its limit.
Per goal:

    s = w_progress * clip(1 - error / goal_distance, -1, 1) + w_reach * reached - w_fall * fell
        - w_cot * min(CoT, 5) - w_torque * peak_torque_frac

Stand mode (documented fallback when no walking controller is available, or mode="stand"): the robot
stands on its position servos at the nominal pose and gets horizontal pushes on the torso (fixed seed,
so every design meets the same pushes: common random numbers). With T_up the time it stayed up out of
T, tilt the mean of (1 - torso up-axis z):

    s = w_progress * T_up / T - w_fall * fell - 2 * tilt - w_cot * min(P / (m g), 5) - w_torque * peak

where P is the mean mechanical power sum |tau qdot|. Both modes subtract w_mass * mass / mass_ref,
mass_ref being the mass of the default robot, so lighter bodies win ties.

The robot's mass is the compiled model's (primitive geoms with HumanoidParams.density), or, with
EvalConfig.cad set, the CAD mass properties of design.cad (slower: the parts are rebuilt per design).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import mujoco
import numpy as np

from ..models import HumanoidParams, build_mjcf


@dataclass
class EvalConfig:
    mode: str = "auto"                 # "auto" | "walk" | "stand"
    goals: list = field(default_factory=lambda: [[1.5, 0.0], [1.0, 0.8]])  # m, relative to the start
    timeout: float = 12.0              # s per walk
    tolerance: float = 0.15            # m, "reached"
    stand_seconds: float = 4.0
    push_force: float = 40.0           # N
    push_interval: float = 1.0         # s
    seed: int = 0
    w_progress: float = 1.0
    w_reach: float = 1.0
    w_fall: float = 1.0
    w_cot: float = 0.2
    w_torque: float = 0.5
    w_mass: float = 0.5
    mass_ref: float | None = None      # kg; default: mass of HumanoidParams()
    cad: dict | None = None            # CadOptions dict -> evaluate with CAD masses and inertias

    def __post_init__(self):
        # an unknown mode used to fall through to "stand" silently, scoring a different objective
        if self.mode not in ("auto", "walk", "stand"):
            raise ValueError(f"EvalConfig.mode must be 'auto', 'walk' or 'stand', got {self.mode!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "EvalConfig":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


_MASS_REF: dict[str, float] = {}


def _default_mass() -> float:
    if "m" not in _MASS_REF:
        m = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams()))
        _MASS_REF["m"] = float(m.body_subtreemass[1])
    return _MASS_REF["m"]


def _robot_mass(model: mujoco.MjModel) -> float:
    return float(model.body_subtreemass[1])


def walker_available(model: mujoco.MjModel | None = None) -> bool:
    try:
        from ..control.walker import WalkController, simulate_walk  # noqa: F401
    except Exception:  # noqa: BLE001 - another module's import error must not break design search
        return False
    return model is None or not hasattr(WalkController, "supports") or WalkController.supports(model)


def _walk(model: mujoco.MjModel, cfg: EvalConfig) -> tuple[float, dict, int]:
    from ..control.walker import simulate_walk
    scores, runs, steps = [], [], 0
    for goal in cfg.goals:
        g = np.asarray(goal, float)
        r = simulate_walk(model, g, timeout=cfg.timeout, tolerance=cfg.tolerance)
        if not all(np.isfinite(float(r[k])) for k in ("error_m", "energy_J", "peak_torque_frac") if k in r):
            r = {**r, "reached": False, "fell": True, "error_m": float(np.linalg.norm(g)) * 2,
                 "cost_of_transport": 5.0, "peak_torque_frac": 1.0}  # unstable simulation = failure
        dist = float(np.linalg.norm(g))
        progress = float(np.clip(1.0 - r["error_m"] / max(dist, 1e-6), -1.0, 1.0))
        cot = float(r.get("cost_of_transport", 0.0))
        cot = cot if np.isfinite(cot) else 5.0
        s = (cfg.w_progress * progress + cfg.w_reach * float(r["reached"]) - cfg.w_fall * float(r["fell"])
             - cfg.w_cot * min(cot, 5.0) - cfg.w_torque * float(r.get("peak_torque_frac", 0.0)))
        scores.append(s)
        steps += int(round((r.get("time_s", 0.0) + 0.5) / model.opt.timestep))
        runs.append({k: (float(v) if isinstance(v, (int, float, np.floating, bool)) else v)
                     for k, v in r.items() if k != "status"})
    metrics = {
        "reached": float(np.mean([r["reached"] for r in runs])),
        "fell": float(np.mean([r["fell"] for r in runs])),
        "error_m": float(np.mean([r["error_m"] for r in runs])),
        "cost_of_transport": float(np.mean([r.get("cost_of_transport", np.nan) for r in runs])),
        "peak_torque_frac": float(np.max([r.get("peak_torque_frac", 0.0) for r in runs])),
        "runs": runs,
    }
    return float(np.mean(scores)), metrics, steps


def _stand(model: mujoco.MjModel, cfg: EvalConfig) -> tuple[float, dict, int]:
    rng = np.random.default_rng(cfg.seed)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    data.ctrl[:] = 0.0
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    torso = torso if torso >= 0 else 1
    dt = model.opt.timestep
    n = int(cfg.stand_seconds / dt)
    z0 = float(data.qpos[2])
    lim = np.abs(model.actuator_forcerange).max(axis=1)
    lim = np.where(model.actuator_forcelimited.astype(bool) & (lim > 0), lim, np.inf)
    dofs = np.array([model.jnt_dofadr[model.actuator_trnid[a, 0]] for a in range(model.nu)])
    mass = _robot_mass(model)
    energy, peak, tilt, fell, k = 0.0, 0.0, 0.0, False, 0
    next_push, push_end, push = 0.5, -1.0, np.zeros(3)
    for k in range(1, n + 1):
        t = k * dt
        if t >= next_push:
            ang = rng.uniform(0, 2 * np.pi)
            push = cfg.push_force * rng.uniform(0.5, 1.0) * np.array([np.cos(ang), np.sin(ang), 0.0])
            push_end, next_push = t + 0.12, t + cfg.push_interval
        data.xfrc_applied[torso, :3] = push if t < push_end else 0.0
        mujoco.mj_step(model, data)
        frc = data.actuator_force
        energy += float(np.abs(frc * data.qvel[dofs]).sum()) * dt
        peak = max(peak, float(np.max(np.abs(frc) / lim)))
        tilt += 1.0 - float(data.xmat[torso, 8])
        if data.qpos[2] < 0.6 * z0 or not np.isfinite(data.qpos).all():
            fell = True
            break
    T_up = k * dt
    power = energy / max(T_up, dt)
    tilt /= max(k, 1)
    s = (cfg.w_progress * T_up / cfg.stand_seconds - cfg.w_fall * float(fell) - 2.0 * tilt
         - cfg.w_cot * min(power / (mass * 9.81), 5.0) - cfg.w_torque * peak)
    return float(s), {"time_up_s": T_up, "fell": float(fell), "tilt": tilt, "power_W": power,
                      "peak_torque_frac": peak}, k


def evaluate_design(params: HumanoidParams | dict, cfg: EvalConfig | dict | None = None) -> dict:
    """Simulate one design. Returns {"score", "mass", "mode", "metrics", "sim_steps", "seconds"}."""
    t0 = time.perf_counter()
    cfg = cfg if isinstance(cfg, EvalConfig) else EvalConfig.from_dict(cfg)
    p = params if isinstance(params, HumanoidParams) else HumanoidParams.from_dict(params)
    xml = build_mjcf(p)
    if cfg.cad is not None:
        from .cad import CadOptions, cad_model_xml
        xml, _ = cad_model_xml(p, CadOptions.from_dict(cfg.cad), xml=xml)
    model = mujoco.MjModel.from_xml_string(xml)
    mode = cfg.mode
    if mode == "auto":
        mode = "walk" if walker_available(model) else "stand"
    if mode == "walk":
        score, metrics, steps = _walk(model, cfg)
    else:
        score, metrics, steps = _stand(model, cfg)
    mass = _robot_mass(model)
    mass_ref = cfg.mass_ref or _default_mass()
    score -= cfg.w_mass * mass / mass_ref
    return {"score": float(score) if np.isfinite(score) else -10.0, "mass": mass, "mode": mode,
            "metrics": metrics, "sim_steps": int(steps), "seconds": time.perf_counter() - t0}


def evaluate_worker(job: tuple[dict, dict]) -> dict:
    """ProcessPoolExecutor entry point: (params dict, EvalConfig dict) -> evaluate_design result.

    Never raises: a crashing design scores -10 with the error message, so one bad body cannot stop a run.
    """
    params, cfg = job
    try:
        return evaluate_design(params, cfg)
    except Exception as e:  # noqa: BLE001
        return {"score": -10.0, "mass": float("nan"), "mode": "error", "metrics": {"error": f"{type(e).__name__}: {e}"},
                "sim_steps": 0, "seconds": 0.0}
