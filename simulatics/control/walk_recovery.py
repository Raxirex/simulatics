"""Push recovery: where and when to step, from the divergent component of motion (DCM).

Under the linear inverted pendulum (LIPM) the DCM xi = c + c'/omega moves away from the centre of pressure p
as xi' = omega (xi - p), so with a constant p it is xi(t) = p + (xi0 - p) exp(omega t). A robot is
*captured* (it can come to rest without another step) when its DCM is inside the support polygon, because
the centre of pressure can then be put on the DCM (Pratt, Carff, Drakunov, Goswami, "Capture point: a step
toward humanoid push recovery", Humanoids 2006; Koolen, de Boer, Rebula, Goswami, Pratt,
"Capturability-based analysis and control of legged locomotion, part 1", IJRR 2012).

After a push the walker calls this module every other control tick (about 1.5-3 ms in NumPy); it searches a
small set of step timings and landing points
(a discretised version of the step location + timing adaptation of Khadiv, Herzog, Moosavian, Righetti,
"Walking control based on step timing adaptation", IEEE T-RO 2020, and the timing / location rules of
Griffin, Wiedebach, Bertrand, Leonessa, Pratt, "Walking stabilization using step timing and location
adjustment on the humanoid robot, Atlas", IROS 2017):

* in single support: when the swing foot should touch down (early touchdown shortens the time the DCM
  diverges) and where (the landing point moves with the predicted DCM);
* in double support / standing: whether the DCM can be held inside the support polygon (no step), or which
  foot should take an emergency step, when and where;
* if one step cannot capture the DCM - typically a sideways push towards the stance foot, which the swing
  foot can only answer by crossing the legs - a two-step plan: a quick short step that keeps a minimum
  width, then a recovery step of the other foot (N-step capturability, Koolen et al. 2012).

The centre of pressure is assumed to stay on the point of the current support closest to the DCM (the ankle
strategy at its limit), which is what the walker's recovery controller commands. All functions are pure
NumPy and work on (x, y, yaw) sole poses; nothing here touches MuJoCo.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LEFT, RIGHT = 0, 1


@dataclass
class CaptureLimits:
    """Reach and timing of recovery steps (all lengths in m, times in s).

    forward / backward: sole advance of a recovery step in the stance sole frame; min_width / max_width:
    lateral sole distance (the swing foot never comes closer than min_width: no crossed legs); hl / hw: half
    length / width of the sole region the centre of pressure may use; t_min: shortest swing; t_max: longest
    swing considered for a step started from double support; t_ds: double support between recovery steps;
    foot_speed: mean horizontal swing-foot speed limit; margin: the DCM must end inside this fraction of the
    support region to count as captured."""
    forward: float = 0.32
    backward: float = 0.22
    min_width: float = 0.13
    max_width: float = 0.50
    hl: float = 0.088
    hw: float = 0.036
    t_min: float = 0.24
    t_max: float = 0.45
    t_ds: float = 0.06
    foot_speed: float = 1.6
    margin: float = 0.85
    width: float = 0.2
    depth: float = 0.0            # cost per gauge^2 of the final DCM (> 0 prefers capturing it deeper inside)


@dataclass
class StepChoice:
    side: int
    target: np.ndarray            # (x, y, yaw) landing sole pose
    duration: float               # s: until touchdown (current swing) or swing duration (new step)


@dataclass
class RecoveryPlan:
    steps: list[StepChoice] = field(default_factory=list)   # empty = hold (no step needed)
    lift_delay: float = 0.0       # s of double support before the first new step lifts off
    captured: bool = True         # predicted DCM inside the final support
    capture_norm: float = 0.0     # support-region norm of the final DCM (<= 1 inside)
    cost: float = 0.0


def effective_cop(xi, p_model, omega: float, dcm_rate=None):
    """Centre of pressure to predict with: the model's p_model, or - when the measured DCM rate shows a
    faster divergence (a push still acting, or a centre of pressure the feet do not fully realise) - the
    effective one xi - xi' / omega that reproduces the measured rate."""
    xi, p_model = np.asarray(xi, float), np.asarray(p_model, float)
    if dcm_rate is None:
        return p_model
    d_meas = np.asarray(dcm_rate, float) / omega
    d_model = xi - p_model
    return xi - d_meas if float(np.linalg.norm(d_meas)) > float(np.linalg.norm(d_model)) else p_model


def dcm_after(xi, p, omega: float, t):
    """LIPM DCM after time t with a constant centre of pressure p (broadcasts over leading axes)."""
    return p + (xi - p) * np.exp(omega * np.asarray(t, float))[..., None]


def _local(v, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return v[..., 0] * c + v[..., 1] * s, -v[..., 0] * s + v[..., 1] * c


def _world(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([x * c - y * s, x * s + y * c], axis=-1)


def hull_norm(q, c0, c1, yaw, hl: float, hw: float):
    """Gauge of point(s) q w.r.t. the convex hull of two equal rectangles (half sizes hl, hw, heading yaw)
    centred at c0 and c1: <= 1 inside, 1 on the boundary, about distance / half-size outside.

    The hull of two translated copies of one rectangle is the segment c0-c1 swept by the rectangle, so the
    gauge is min over lambda in [0, 1] of the rectangle norm of q - c0 - lambda (c1 - c0); that function is
    convex and piecewise linear in lambda, so its minimum is at one of six breakpoints. Broadcasts."""
    ax, ay = _local(np.asarray(q, float) - c0, yaw)
    dx, dy = _local(np.asarray(c1, float) - c0, yaw)
    ax, ay, dx, dy = np.broadcast_arrays(ax, ay, dx, dy)
    num = np.stack([ax, ay, ax / hl - ay / hw, ax / hl + ay / hw])
    den = np.stack([dx, dy, dx / hl - dy / hw, dx / hl + dy / hw])
    ok = np.abs(den) > 1e-12
    lam = np.where(ok, num / np.where(ok, den, 1.0), 0.0)
    lam = np.clip(np.concatenate([lam, np.ones((1,) + ax.shape)]), 0.0, 1.0)   # breakpoints, 0 (via ~ok), 1
    lam = np.concatenate([lam, np.zeros((1,) + ax.shape)])
    return np.maximum(np.abs(ax - lam * dx) / hl, np.abs(ay - lam * dy) / hw).min(axis=0)


def clamp_to_sole(p, c, yaw, hl: float, hw: float):
    """Closest point to p inside the rectangle (half sizes hl, hw) centred at c with heading yaw."""
    x, y = _local(np.asarray(p, float) - c, yaw)
    return c + _world(np.clip(x, -hl, hl), np.clip(y, -hw, hw), yaw)


def _corners(c, yaw, hl, hw):
    loc = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
    return c + _world(loc[:, 0], loc[:, 1], yaw)


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull (Andrew's monotone chain)."""
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def clamp_to_support(p, soles: list, hl: float, hw: float) -> np.ndarray:
    """Closest point to p inside the convex hull of the given soles [(x, y, yaw), ...] shrunk to (hl, hw)."""
    p = np.asarray(p, float)
    if len(soles) == 1:
        s = soles[0]
        return clamp_to_sole(p, np.asarray(s[:2], float), float(s[2]), hl, hw)
    poly = _convex_hull(np.vstack([_corners(np.asarray(s[:2], float), float(s[2]), hl, hw) for s in soles]))
    n = len(poly)
    inside = True
    best, best_d = p, np.inf
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        if e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0]) < 0:
            inside = False
        t = float(np.clip(np.dot(p - a, e) / max(np.dot(e, e), 1e-12), 0.0, 1.0))
        q = a + t * e
        d = float(np.dot(p - q, p - q))
        if d < best_d:
            best, best_d = q, d
    return p.copy() if inside else best


def support_norm(xi, soles: list, hl: float, hw: float) -> float:
    """hull_norm of xi for one or two soles (heading of the first sole)."""
    c0 = np.asarray(soles[0][:2], float)
    c1 = np.asarray(soles[-1][:2], float)
    return float(hull_norm(np.asarray(xi, float), c0, c1, float(soles[0][2]), hl, hw))


def _box_grid(c0, yaw, side: int, lim: CaptureLimits, nx: int = 7, ny: int = 5) -> np.ndarray:
    """Landing points (nx * ny, 2) on a grid over the reach box of a step of `side` from stance centre c0."""
    sign = 1.0 if side == LEFT else -1.0
    gx, gy = np.meshgrid(np.linspace(-lim.backward, lim.forward, nx), np.linspace(lim.min_width, lim.max_width, ny),
                         indexing="ij")
    return c0 + _world(gx.ravel(), sign * gy.ravel(), yaw)


def _clip_to_box(u, c0, yaw, side: int, lim: CaptureLimits):
    """Clip landing point(s) u into the reach box of `side` from stance sole centre(s) c0 (broadcasts)."""
    sign = 1.0 if side == LEFT else -1.0
    x, y = _local(np.asarray(u, float) - c0, yaw)
    return c0 + _world(np.clip(x, -lim.backward, lim.forward),
                       sign * np.clip(sign * y, lim.min_width, lim.max_width), yaw)


@dataclass
class _Weights:
    capture: float = 400.0        # per (norm excess)^2
    shift: float = 1.0            # per m^2 of landing point moved from the plan
    timing: float = 0.5           # per s^2 of touchdown time changed
    second_step: float = 0.02     # flat cost of needing a second step
    shift2: float = 0.3


W = _Weights()


def _second_step(xi_td, u, yaw, side2: int, foot2_xy, u2_plan, omega, lim: CaptureLimits, t2_choices):
    """Best second step (by the other foot) after landing at u (n, 2) with DCM xi_td (n, 2).
    Returns (cost (n,), u2 (n, 2), t2 (n,), norm (n,))."""
    n = len(u)
    # double support: weight moves onto u; the CoP sits on u's sole as close to the DCM as it can
    p = clamp_to_sole(xi_td, u, yaw, lim.hl, lim.hw)
    xi_ds = dcm_after(xi_td, p, omega, np.full(n, lim.t_ds))
    p2 = clamp_to_sole(xi_ds, u, yaw, lim.hl, lim.hw)
    best = (np.full(n, np.inf), np.zeros((n, 2)), np.zeros(n), np.full(n, np.inf))
    sign = 1.0 if side2 == LEFT else -1.0
    out0 = _world(np.zeros(1), np.array([sign]), yaw)
    for t2 in t2_choices:
        xi2 = dcm_after(xi_ds, p2, omega, np.full(n, t2))
        d = xi2 - u
        nd = np.linalg.norm(d, axis=1, keepdims=True)
        d = np.where(nd > 1e-9, d / np.maximum(nd, 1e-9), out0)
        for k in (0.03,):
            u2 = _clip_to_box(xi2 + k * d, u, yaw, side2, lim)
            reach_ok = np.linalg.norm(u2 - foot2_xy, axis=1) <= lim.foot_speed * t2
            nrm = hull_norm(xi2, u, u2, yaw, lim.hl, lim.hw)
            cost = (W.capture * np.maximum(nrm - lim.margin, 0.0) ** 2 + lim.depth * nrm ** 2 + W.second_step
                    + W.shift2 * np.sum((u2 - u2_plan) ** 2, axis=1) + 2.0 * W.timing * (t2 - lim.t_max) ** 2
                    + np.where(reach_ok, 0.0, 1e3))
            better = cost < best[0]
            best = (np.where(better, cost, best[0]), np.where(better[:, None], u2, best[1]),
                    np.where(better, t2, best[2]), np.where(better, nrm, best[3]))
    return best


def _best_step(xi_td, t1, c0, yaw, side: int, foot_xy, plan_xy, t_nom: float, omega: float, lim: CaptureLimits,
               u2_plan, two_step: bool):
    """Evaluate one step of `side` from stance centre c0 for several touchdown times at once.

    xi_td (T, 2): DCM at touchdown for each time t1 (T,). Candidates per time: a grid over the reach box,
    the planned landing point and two points on / beyond the DCM (clipped into the box). Returns
    (cost, t1, landing (2,), (u2, t2) or None, final support gauge) of the cheapest plan."""
    grid = _box_grid(c0, yaw, side, lim)
    T = len(t1)
    out = xi_td - c0
    out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    extra = np.stack([np.broadcast_to(plan_xy, xi_td.shape), xi_td, xi_td + 0.04 * out], axis=1)   # (T, 3, 2)
    extra = _clip_to_box(extra, c0, yaw, side, lim)
    cand = np.concatenate([np.broadcast_to(grid, (T,) + grid.shape), extra], axis=1)                # (T, K, 2)
    K = cand.shape[1]
    cand = cand.reshape(-1, 2)
    xi_r = np.repeat(xi_td, K, axis=0)
    t_r = np.repeat(np.asarray(t1, float), K)
    reach_ok = np.linalg.norm(cand - foot_xy, axis=1) <= lim.foot_speed * np.maximum(t_r, 1e-3) + 0.02
    nrm = hull_norm(xi_r, c0, cand, yaw, lim.hl, lim.hw)
    base = (W.shift * np.sum((cand - plan_xy) ** 2, axis=1) + W.timing * (t_r - t_nom) ** 2
            + np.where(reach_ok, 0.0, 1e3))
    cost = base + W.capture * np.maximum(nrm - lim.margin, 0.0) ** 2 + lim.depth * nrm ** 2
    use2 = np.zeros(len(cand), bool)
    if two_step:
        c2, u2, t2, n2 = _second_step(xi_r, cand, yaw, 1 - side, c0, u2_plan, omega, lim,
                                      np.linspace(lim.t_min, lim.t_max, 3))
        use2 = base + c2 < cost
        cost = np.where(use2, base + c2, cost)
    i = int(np.argmin(cost))
    if use2[i]:
        return float(cost[i]), float(t_r[i]), cand[i], (u2[i], float(t2[i])), float(n2[i])
    return float(cost[i]), float(t_r[i]), cand[i], None, float(nrm[i])


def plan_swing(xi, omega: float, stance, swing_side: int, swing_xy, t_rem: float, plan_target,
               lim: CaptureLimits, t_land_min: float = 0.08, next_plan=None, two_step: bool = True,
               dcm_rate=None) -> RecoveryPlan:
    """Recovery plan while a foot is in the air.

    xi: measured DCM (2,); stance: (x, y, yaw) stance sole; swing_xy: current swing sole position;
    t_rem: nominal time to touchdown; plan_target: (x, y, yaw) planned landing pose; t_land_min: shortest
    time in which the swing foot can still be brought down; next_plan: planned pose of the next footstep
    (other foot), or None; dcm_rate: measured DCM velocity (see effective_cop). Returns the best plan:
    steps[0] = this swing (landing pose, time to touchdown), optionally steps[1] = the following step of the
    other foot (two_step=True)."""
    xi = np.asarray(xi, float)
    c0, yaw = np.asarray(stance[:2], float), float(stance[2])
    plan_xy = np.asarray(plan_target[:2], float)
    t_lo = max(float(t_land_min), 0.03)
    t_hi = max(t_lo, float(t_rem))
    t_choices = np.linspace(t_lo, t_hi, 6)
    p0 = effective_cop(xi, clamp_to_sole(xi, c0, yaw, lim.hl, lim.hw), omega, dcm_rate)
    u2_plan = np.asarray(next_plan[:2], float) if next_plan is not None else c0
    t2_choices = np.linspace(lim.t_min, lim.t_max, 3)
    xi_td = dcm_after(np.broadcast_to(xi, (len(t_choices), 2)), p0, omega, t_choices)
    cost, t1, u, second, final = _best_step(xi_td, t_choices, c0, yaw, swing_side, np.asarray(swing_xy, float),
                                            plan_xy, t_hi, omega, lim, u2_plan, two_step)
    steps = [StepChoice(swing_side, np.array([u[0], u[1], float(plan_target[2])]), t1)]
    if second is not None:
        yaw2 = float(next_plan[2]) if next_plan is not None else yaw
        steps.append(StepChoice(1 - swing_side, np.array([second[0][0], second[0][1], yaw2]), second[1]))
    return RecoveryPlan(steps=steps, captured=final <= lim.margin, capture_norm=final, cost=cost)


def plan_double_support(xi, omega: float, feet: dict, lim: CaptureLimits, next_side: int | None = None,
                        next_plan=None, hold_margin: float | None = None, side: int | None = None,
                        lift_delay: float | None = None, two_step: bool = True, dcm_rate=None) -> RecoveryPlan:
    """Recovery plan with both feet on the ground (standing, or between steps).

    feet: {LEFT: (x, y, yaw), RIGHT: ...}. Returns an empty plan (hold: no step) when the DCM is inside the
    support polygon (gauge <= hold_margin, default lim.margin); otherwise the best one- or two-step plan.
    side restricts the first stepping foot (None = try both); lift_delay (default lim.t_ds) is the double
    support left before it lifts off. next_side / next_plan: the planned next footstep, which a recovery step
    of that foot stays close to (otherwise a foot steps from where it is); dcm_rate: measured DCM velocity
    (see effective_cop)."""
    xi = np.asarray(xi, float)
    hm = lim.margin if hold_margin is None else hold_margin
    nrm_now = support_norm(xi, [feet[LEFT], feet[RIGHT]], lim.hl, lim.hw)
    if side is None and nrm_now <= hm:
        return RecoveryPlan(captured=True, capture_norm=nrm_now)
    delay = lim.t_ds if lift_delay is None else max(0.0, float(lift_delay))
    best: RecoveryPlan | None = None
    for sd in ((LEFT, RIGHT) if side is None else (side,)):
        stance = feet[1 - sd]
        c0, yaw = np.asarray(stance[:2], float), float(stance[2])
        foot_xy = np.asarray(feet[sd][:2], float)
        plan_xy = np.asarray(next_plan[:2], float) if (next_plan is not None and next_side == sd) else foot_xy
        p = effective_cop(xi, clamp_to_sole(xi, c0, yaw, lim.hl, lim.hw), omega, dcm_rate)
        xi0 = dcm_after(xi, p, omega, delay)
        p1 = effective_cop(xi0, clamp_to_sole(xi0, c0, yaw, lim.hl, lim.hw), omega,
                           None if dcm_rate is None else np.asarray(dcm_rate, float) * np.exp(omega * delay))
        t_choices = np.linspace(lim.t_min, lim.t_max, 4)
        xi_td = dcm_after(np.broadcast_to(xi0, (len(t_choices), 2)), p1, omega, t_choices)
        cost, t1, u, second, final = _best_step(xi_td, t_choices, c0, yaw, sd, foot_xy, plan_xy, lim.t_max, omega,
                                                lim, c0, two_step)
        if best is None or cost < best.cost:
            steps = [StepChoice(sd, np.array([u[0], u[1], yaw]), t1)]
            if second is not None:
                steps.append(StepChoice(1 - sd, np.array([second[0][0], second[0][1], yaw]), second[1]))
            best = RecoveryPlan(steps=steps, lift_delay=delay, captured=final <= lim.margin,
                                capture_norm=final, cost=cost)
    return best
