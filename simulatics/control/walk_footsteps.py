"""Footstep planning along a 2D path, ending with both feet side by side on the goal.

The planner moves a *walking frame* W = (x, y, yaw) - the point midway between
the feet - along the path one footstep at a time, and puts the stepping foot
half a stance width to its side of W. This is the classic step-sequence
construction used by sampling footstep planners (e.g. Chestnutt et al.,
"Footstep planning for the Honda ASIMO humanoid", ICRA 2005; IHMC's
`ihmc-footstep-planning`), reduced to a greedy search along a given path:

* each step advances W as far along the path as a kinematic step limit allows:
  an ellipse in the previous frame (forward, backward and sideways reach) that
  shrinks while the frame also turns, with at most `turn` radians per step;
* the heading looks a short distance ahead on the path (so corners are cut
  smoothly and sharp corners become turns on the spot) and blends into the
  requested final yaw over the last part of the path;
* short moves (less than about two steps) keep the current heading and shuffle
  sideways/backwards instead of turning towards a goal that is right next to
  the robot;
* the last two footsteps put the left and right foot exactly at the goal
  stance, so the arrival error is the landing error of those two steps and
  does not accumulate along the path.

Also here: `square_up` (stop in mid-walk: bring the trailing foot next to the
leading one) and `velocity_footsteps` (a short rolling plan for velocity
commands). Footsteps are (x, y, yaw, side) with (x, y) the sole centre and
side 0 = left, 1 = right.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEFT, RIGHT = 0, 1


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def mean_yaw(a: float, b: float) -> float:
    return float(a + wrap(b - a) / 2)


@dataclass
class StepLimits:
    forward: float = 0.22     # m, sole-to-sole advance along the walking direction per footstep
    backward: float = 0.10
    lateral: float = 0.07
    turn: float = 0.35        # rad per footstep
    width: float = 0.20       # nominal lateral distance between the two sole centres
    min_width: float = 0.13   # never place the feet closer than this (sideways, in the stance foot frame)
    lookahead: float = 0.35   # m, heading looks this far ahead along the path
    blend: float = 0.6        # m, distance over which the heading turns into the final yaw


def _offset(pose, side: int, half: float) -> np.ndarray:
    x, y, yaw = pose
    s = half if side == LEFT else -half
    return np.array([x - np.sin(yaw) * s, y + np.cos(yaw) * s, yaw])


class _Path:
    def __init__(self, pts: np.ndarray):
        seg = np.diff(pts, axis=0)
        keep = np.concatenate([[True], np.linalg.norm(seg, axis=1) > 1e-6])
        self.p = pts[keep]
        seg = np.diff(self.p, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.s[-1])

    def at(self, s: float) -> np.ndarray:
        if len(self.p) == 1:
            return self.p[0].copy()
        s = float(np.clip(s, 0.0, self.length))
        i = int(np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.seg_len) - 1))
        t = (s - self.s[i]) / max(self.seg_len[i], 1e-9)
        return self.p[i] + (self.p[i + 1] - self.p[i]) * t

    def heading(self, s: float, lookahead: float) -> float:
        if len(self.p) == 1:
            return 0.0
        a = self.at(s)
        b = self.at(s + lookahead)
        if np.linalg.norm(b - a) < 1e-3 * lookahead:      # at the end: last segment direction
            b, a = self.p[-1], self.p[-2]
        return float(np.arctan2(b[1] - a[1], b[0] - a[0]))


def goal_stance(goal_xy, goal_yaw: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Left and right sole poses (x, y, yaw) of the final stance on the goal."""
    g = (float(goal_xy[0]), float(goal_xy[1]), float(goal_yaw))
    return _offset(g, LEFT, width / 2), _offset(g, RIGHT, width / 2)


def _setup(left, right, path_xy, final_yaw: float | None, lim: StepLimits):
    """Shared start of planning: the walking frame between the feet, the path from it, whether the move is
    short (under two steps: shuffle instead of turning towards the goal) and the final heading."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    pts = np.asarray(path_xy, float).reshape(-1, 2)
    if not (np.isfinite(left).all() and np.isfinite(right).all() and np.isfinite(pts).all()
            and (final_yaw is None or np.isfinite(final_yaw))):
        raise ValueError("footstep planning needs finite feet, path and final yaw")
    w0 = np.array([(left[0] + right[0]) / 2, (left[1] + right[1]) / 2, mean_yaw(left[2], right[2])])
    if len(pts) == 0:
        pts = w0[None, :2].copy()
    if len(pts) > 1 and np.linalg.norm(pts[0] - w0[:2]) < 0.3:
        pts = pts.copy()                  # the path starts at the robot: start it exactly between the feet
        pts[0] = w0[:2]
    else:
        pts = np.vstack([w0[:2], pts])
    path = _Path(pts)
    short = path.length < 2.0 * lim.forward
    if final_yaw is not None:
        goal_yaw = float(final_yaw)
    elif short:
        goal_yaw = float(w0[2])
    else:
        goal_yaw = path.heading(path.length, lim.lookahead)
    return left, right, w0, pts, path, short, goal_yaw


def goal_heading(left, right, path_xy, final_yaw: float | None, lim: StepLimits) -> float:
    """The final yaw plan_footsteps chooses for this request: final_yaw if given, else the current heading for
    a short move, else the arrival direction of the path."""
    return _setup(left, right, path_xy, final_yaw, lim)[-1]


def _at_stance(foot, want, tol_m: float = 0.01, tol_rad: float = np.deg2rad(2)) -> bool:
    e = np.asarray(foot, float) - np.asarray(want, float)
    return bool(np.hypot(e[0], e[1]) <= tol_m and abs(wrap(e[2])) <= tol_rad)


def plan_footsteps(left, right, path_xy, final_yaw: float | None, lim: StepLimits,
                   first_side: int | None = None, max_steps: int | None = None) -> tuple[list[list[float]], float]:
    """Footsteps from the current feet (sole poses (x, y, yaw)) along path_xy to a side-by-side stance at
    path_xy[-1]. first_side forces the leading foot (used when re-planning in mid-walk, so the feet keep
    alternating); None picks the foot on the side the robot first moves or turns to. A short request whose goal
    stance the feet already hold (1 cm / 2 deg) plans no steps. max_steps (None = from the path length) only
    guards against a runaway loop; running out raises RuntimeError instead of leaping to the goal.
    Returns (footsteps [[x, y, yaw, side], ...], goal_yaw). Raises ValueError for non-finite input."""
    left, right, w0, pts, path, short, goal_yaw = _setup(left, right, path_xy, final_yaw, lim)
    L = path.length
    goal_l, goal_r = goal_stance(pts[-1], goal_yaw, lim.width)
    if short and _at_stance(left, goal_l) and _at_stance(right, goal_r):
        return [], goal_yaw
    if max_steps is None:
        # each footstep either advances along the path by at least ~0.4 x the smallest reach (the step ellipse
        # shrinks to 0.4 while turning at the limit) or turns on the spot; budget both generously
        reach = max(0.4 * min(lim.forward, lim.backward, lim.lateral), 1e-3)
        turns = int(np.ceil(np.pi / max(lim.turn, 1e-3)))
        max_steps = 50 + int(np.ceil(L / reach)) + 2 * (len(path.p) + 2) * turns

    def heading_at(s: float) -> float:
        if short:
            return float(w0[2] + wrap(goal_yaw - w0[2]) * (s / L if L > 1e-9 else 1.0))
        h = path.heading(s, lim.lookahead)
        blend = min(lim.blend, L / 2)
        k = float(np.clip((s - (L - blend)) / max(blend, 1e-9), 0.0, 1.0))
        return float(h + wrap(goal_yaw - h) * k)

    steps: list[list[float]] = []
    W, s = w0.copy(), 0.0
    side = first_side
    ds = 0.01
    for _ in range(max_steps + 1):
        if s >= L - 1e-9 and abs(wrap(W[2] - goal_yaw)) < 1e-3:
            break
        if len(steps) >= max_steps:
            # the closing pair below would otherwise jump from here straight onto the goal
            raise RuntimeError(f"footstep plan did not reach the goal in {max_steps} steps "
                               f"({s:.2f} of {L:.2f} m)")
        c, si = np.cos(W[2]), np.sin(W[2])
        best = None
        s_try = s
        while True:
            s_next = min(L, s_try + ds) if s_try < L else L
            target = heading_at(s_next)
            dth = float(np.clip(wrap(target - W[2]), -lim.turn, lim.turn))
            p = path.at(s_next)
            dx = c * (p[0] - W[0]) + si * (p[1] - W[1])
            dy = -si * (p[0] - W[0]) + c * (p[1] - W[1])
            ax = lim.forward if dx >= 0 else lim.backward
            scale = 1.0 - 0.6 * abs(dth) / lim.turn
            if (dx / ax) ** 2 + (dy / lim.lateral) ** 2 > scale ** 2:
                break
            best = (s_next, p, dth)
            if s_next >= L:
                break
            s_try = s_next
        if best is None:                         # cannot even hold position while turning: turn in place
            best = (s, path.at(s), float(np.clip(wrap(heading_at(s) - W[2]), -lim.turn, lim.turn)))
        s, p, dth = best
        newW = np.array([p[0], p[1], W[2] + dth])
        if side is None:                         # lead with the foot on the side we are moving/turning to
            dy = -si * (p[0] - W[0]) + c * (p[1] - W[1])
            side = LEFT if (dth > 1e-6 or (abs(dth) <= 1e-6 and dy >= 0)) else RIGHT
        W = newW
        steps.append([*_offset(W, side, lim.width / 2), side])
        side = 1 - side
    # final stance: both feet on the goal
    want = {LEFT: goal_l, RIGHT: goal_r}
    if steps:
        steps[-1][:3] = want[steps[-1][3]].tolist()
        steps.append([*want[1 - steps[-1][3]], 1 - steps[-1][3]])
    else:
        cur = {LEFT: left, RIGHT: right}
        order = (LEFT, RIGHT) if first_side in (None, LEFT) else (RIGHT, LEFT)
        for sd in order:
            if not _at_stance(cur[sd], want[sd]):
                steps.append([*want[sd], sd])
    _keep_feet_apart(steps, left, right, lim)
    return [[float(v) for v in st[:3]] + [int(st[3])] for st in steps], goal_yaw


def square_up(left, right, last_side: int, lim: StepLimits) -> list[list[float]]:
    """Footsteps that bring the feet side by side, next to the foot that landed last (a stop in mid-walk)."""
    feet = {LEFT: np.asarray(left, float), RIGHT: np.asarray(right, float)}
    anchor = feet[last_side]
    other = 1 - last_side
    W = _offset(anchor, other, lim.width / 2)
    W[2] = anchor[2]
    want = _offset(W, other, lim.width / 2)
    e = feet[other] - want
    if np.hypot(e[0], e[1]) <= 0.01 and abs(wrap(e[2])) <= np.deg2rad(2):
        return []
    return [[float(want[0]), float(want[1]), float(want[2]), int(other)]]


def _keep_feet_apart(steps: list[list[float]], left, right, lim: StepLimits) -> None:
    """Push a footstep outwards when it would land closer than min_width to the stance foot."""
    feet = {LEFT: np.asarray(left, float), RIGHT: np.asarray(right, float)}
    for st in steps:
        side = int(st[3])
        stance = feet[1 - side]
        c, s = np.cos(stance[2]), np.sin(stance[2])
        rel = np.array([st[0] - stance[0], st[1] - stance[1]])
        lat = -s * rel[0] + c * rel[1]
        sign = 1.0 if side == LEFT else -1.0
        if sign * lat < lim.min_width:
            push = lim.min_width - sign * lat
            st[0] += -s * sign * push
            st[1] += c * sign * push
        feet[side] = np.array(st[:3], float)


def velocity_footsteps(left, right, next_side: int, vx: float, vy: float, wz: float, step_period: float,
                       lim: StepLimits, n: int = 4) -> list[list[float]]:
    """A short rolling plan for continuous velocity commands (robot frame m/s, rad/s).

    The walking frame continues from the foot that landed last (the one that is *not* next_side), so
    re-planning at every touchdown keeps the full step length instead of restarting from the feet midpoint.
    """
    last = np.asarray(left if next_side == RIGHT else right, float)
    W = _offset(last, next_side, lim.width / 2)
    W[2] = last[2]
    dx, dy, dth = vx * step_period, vy * step_period, wz * step_period
    dth = float(np.clip(dth, -lim.turn, lim.turn))
    ax = lim.forward if dx >= 0 else lim.backward
    scale = 1.0 - 0.6 * abs(dth) / lim.turn
    norm = np.hypot(dx / ax, dy / lim.lateral)
    if norm > scale:
        dx, dy = dx * scale / norm, dy * scale / norm
    steps, side = [], next_side
    for _ in range(n):
        c, s = np.cos(W[2]), np.sin(W[2])
        W = np.array([W[0] + c * dx - s * dy, W[1] + s * dx + c * dy, W[2] + dth])
        steps.append([*_offset(W, side, lim.width / 2), side])
        side = 1 - side
    _keep_feet_apart(steps, left, right, lim)
    return [[float(v) for v in st[:3]] + [int(st[3])] for st in steps]
