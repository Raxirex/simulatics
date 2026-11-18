"""Link load cases from simulation: joint reaction wrenches recorded while the robot walks or is pushed.

Method: MuJoCo's `mj_rnePostConstraint` (recursive Newton-Euler after the constraint solve) gives
`cfrc_int[b]`, the 6D wrench [torque; force] that the parent exerts on body b's subtree, expressed
at the subtree centre of mass c of the kinematic tree with world axes. It contains everything that
crosses the joint: actuator torque, constraint and contact reactions carried up the chain, gravity
and inertia of the subtree. We move it to the proximal joint point p (M_p = M_c - (p - c) x F) and
rotate it into the link frame.

The time series of every link (left and right pooled, the right side mirrored into the left link's
frame) is reduced to a handful of extreme static load cases: the instants of maximum resultant force,
bending moment, torsion and axial force. This is the equivalent-static-loads idea (Park and Kang
2003; Jang, Lee and Park 2012) used to drive topology optimisation from dynamic simulation; note
that Stolpe et al. (2018) show it is a heuristic, so check the optimised links in simulation again.

Scenarios: walking with the model-based WalkController (simulatics.control.walker) when it is
importable and supports the model; otherwise (or with scenario="push") the robot stands on its servos
and is shoved by horizontal pushes on the torso. Samples after a fall (torso below 60 % of its
initial height) are discarded, so impacts of falling never become design loads.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import mujoco
import numpy as np

from ..models import HumanoidParams, build_mjcf

LINKS = ("thigh", "shin", "upper_arm", "forearm")
_MIRROR_F = np.array([1.0, -1.0, 1.0])     # reflection y -> -y applied to a force (polar vector)
_MIRROR_M = np.array([-1.0, 1.0, -1.0])    # ... and to a moment (axial vector): det(S) S M


@dataclass
class LoadConfig:
    scenario: str = "auto"          # "auto" (walk if a walker is available, else push) | "walk" | "push"
    goal: tuple = (2.0, 0.0)        # walking goal (m, world) for the walk scenario
    walk_seconds: float = 8.0
    push_seconds: float = 6.0
    push_force: float = 40.0        # N, peak horizontal push on the torso (the unbalanced stand tips over
                                    # at ~40-60 N; samples after a fall are dropped anyway)
    push_duration: float = 0.12     # s
    push_interval: float = 1.0      # s between pushes
    settle: float = 0.3             # s at the start that are not recorded (touch-down transient)
    dynamic_factor: float = 1.5     # multiplies the extracted loads (safety / dynamic amplification)
    max_cases: int = 4
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LoadCase:
    """A static wrench on a link: what its parent exerts on it at the proximal joint, in the link frame."""
    link: str
    force: np.ndarray        # (3,) N
    moment: np.ndarray       # (3,) N m about the proximal joint
    time: float              # s into the rollout
    side: str                # body the sample came from ("left" / "right")
    reason: str              # which extreme picked it
    source: str              # "walk" | "push"

    def to_dict(self) -> dict:
        return {"link": self.link, "force": self.force.tolist(), "moment": self.moment.tolist(),
                "time": self.time, "side": self.side, "reason": self.reason, "source": self.source}


@dataclass
class LoadReport:
    cases: dict[str, list[LoadCase]]
    source: str
    seconds: float                                  # simulated time that was kept
    peak: dict[str, dict] = field(default_factory=dict)
    series: dict[str, np.ndarray] = field(default_factory=dict)   # link -> (T*2, 6) [F, M], pooled

    def summary(self) -> dict:
        return {"source": self.source, "seconds": self.seconds, "peak": self.peak,
                "cases": {k: [c.to_dict() for c in v] for k, v in self.cases.items()}}


# ---------------------------------------------------------------------------------------------
def link_wrenches(model: mujoco.MjModel, data: mujoco.MjData, bodies: np.ndarray) -> np.ndarray:
    """(len(bodies), 6) [force, moment] the parent exerts on each body, at its proximal joint, body frame.

    Needs data.cfrc_int to be current (call mj_rnePostConstraint after mj_forward).
    """
    out = np.empty((len(bodies), 6))
    for n, b in enumerate(bodies):
        c = data.subtree_com[model.body_rootid[b]]
        torque, force = data.cfrc_int[b, :3], data.cfrc_int[b, 3:]
        j = model.body_jntadr[b]
        p = data.xanchor[j] if j >= 0 else data.xpos[b]
        M = torque - np.cross(p - c, force)
        R = data.xmat[b].reshape(3, 3)
        out[n, :3] = R.T @ force
        out[n, 3:] = R.T @ M
    return out


def _bodies(model: mujoco.MjModel) -> tuple[list[str], np.ndarray]:
    names, ids = [], []
    for link in LINKS:
        for side in ("left", "right"):
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_{link}")
            if b >= 0:
                names.append(f"{side}_{link}")
                ids.append(b)
    return names, np.array(ids, int)


def _record(model, data, ids, rows, times, t) -> None:
    mujoco.mj_forward(model, data)
    mujoco.mj_rnePostConstraint(model, data)
    rows.append(link_wrenches(model, data, ids))
    times.append(t)


def _walk_rollout(model, cfg: LoadConfig, ids) -> tuple[list, list, str] | None:
    try:
        from ..control.walker import WalkController
    except ImportError:
        return None
    if hasattr(WalkController, "supports") and not WalkController.supports(model):
        return None
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ctrl = WalkController(model)
    ctrl.reset(model, data)
    n_sub = max(1, int(round(ctrl.control_dt / model.opt.timestep)))
    z0 = float(data.qpos[2])
    rows, times = [], []
    t = 0.0
    goal_set = False
    while t < cfg.walk_seconds:
        if not goal_set and t >= 0.5:  # settle, then walk
            start = data.qpos[:2].copy()
            ctrl.set_path(np.array([start, np.asarray(cfg.goal, float)]), None)
            goal_set = True
        ctrl.step(model, data)
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
            t += model.opt.timestep
            if data.qpos[2] < 0.6 * z0 or not np.isfinite(data.qpos).all():
                return rows, times, "walk"
            if t >= cfg.settle:
                _record(model, data, ids, rows, times, t)
        if goal_set and ctrl.status().get("mode") in ("done", "fallen"):
            break
    return rows, times, "walk"


def _push_rollout(model, cfg: LoadConfig, ids) -> tuple[list, list, str]:
    """Stand on the servos at the nominal pose and get shoved by random horizontal pushes."""
    rng = np.random.default_rng(cfg.seed)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    data.ctrl[:] = 0.0
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    torso = torso if torso >= 0 else 1
    z0 = float(data.qpos[2])
    dt = model.opt.timestep
    rows, times = [], []
    n = int(cfg.push_seconds / dt)
    next_push, push_end, push = 0.5, -1.0, np.zeros(3)
    for k in range(n):
        t = k * dt
        if t >= next_push:
            ang = rng.uniform(0, 2 * np.pi)
            push = cfg.push_force * rng.uniform(0.5, 1.0) * np.array([np.cos(ang), np.sin(ang), 0.0])
            push_end, next_push = t + cfg.push_duration, t + cfg.push_interval
        data.xfrc_applied[torso, :3] = push if t < push_end else 0.0
        mujoco.mj_step(model, data)
        if data.qpos[2] < 0.6 * z0 or not np.isfinite(data.qpos).all():
            break
        if t + dt >= cfg.settle:
            _record(model, data, ids, rows, times, t + dt)
    return rows, times, "push"


def _pick(link: str, series: np.ndarray, sides: np.ndarray, times: np.ndarray, source: str,
          cfg: LoadConfig) -> list[LoadCase]:
    F, M = series[:, :3], series[:, 3:]
    criteria = [
        ("max |F|", np.linalg.norm(F, axis=1)),
        ("max bending", np.hypot(M[:, 0], M[:, 1])),
        ("max torsion", np.abs(M[:, 2])),
        ("max axial", np.abs(F[:, 2])),
        ("max shear", np.hypot(F[:, 0], F[:, 1])),
    ]
    picked, out = set(), []
    for reason, val in criteria:
        i = int(np.argmax(val))
        if i in picked:
            continue
        picked.add(i)
        out.append(LoadCase(link, cfg.dynamic_factor * F[i].copy(), cfg.dynamic_factor * M[i].copy(),
                            float(times[i]), str(sides[i]), reason, source))
        if len(out) >= cfg.max_cases:
            break
    return out


def extract_link_loads(params: HumanoidParams | None = None, cfg: LoadConfig | None = None,
                       xml: str | None = None) -> LoadReport:
    """Simulate the robot and return the extreme load cases of thigh, shin, upper_arm and forearm.

    Forces/moments are in the LEFT link's body frame (right-side samples mirrored), already multiplied
    by cfg.dynamic_factor. The proximal joint is the body origin; the link extends along its geoms.
    """
    cfg = cfg or LoadConfig()
    xml = xml or build_mjcf(params or HumanoidParams())
    model = mujoco.MjModel.from_xml_string(xml)
    names, ids = _bodies(model)
    if not len(ids):
        raise ValueError("model has none of the bodies " + ", ".join(f"left_{n}" for n in LINKS))
    got = None
    if cfg.scenario in ("auto", "walk"):
        got = _walk_rollout(model, cfg, ids)
        if got is not None and len(got[0]) < int(1.0 / model.opt.timestep) and cfg.scenario == "auto":
            got = None  # fell within a second: not representative, use pushes
    if got is None:
        if cfg.scenario == "walk":
            raise RuntimeError("walk scenario requested but no walking controller supports this model")
        got = _push_rollout(model, cfg, ids)
    rows, times, source = got
    if not rows:
        raise RuntimeError("the robot fell before any load sample was recorded")
    W = np.stack(rows)               # (T, nb, 6)
    times = np.asarray(times)
    cases, peak, series_out = {}, {}, {}
    for link in LINKS:
        cols = [k for k, n in enumerate(names) if n.endswith(link) and n.split("_", 1)[1] == link]
        if not cols:
            continue
        parts, sides, ts = [], [], []
        for k in cols:
            w = W[:, k].copy()
            side = names[k].split("_", 1)[0]
            if side == "right":
                w[:, :3] *= _MIRROR_F
                w[:, 3:] *= _MIRROR_M
            parts.append(w)
            sides += [side] * len(w)
            ts.append(times)
        series = np.concatenate(parts)
        sides_a, ts_a = np.array(sides), np.concatenate(ts)
        cases[link] = _pick(link, series, sides_a, ts_a, source, cfg)
        series_out[link] = series
        peak[link] = {"force_N": float(np.linalg.norm(series[:, :3], axis=1).max()),
                      "bending_Nm": float(np.hypot(series[:, 3], series[:, 4]).max()),
                      "torsion_Nm": float(np.abs(series[:, 5]).max())}
    return LoadReport(cases, source, float(times[-1]), peak, series_out)
