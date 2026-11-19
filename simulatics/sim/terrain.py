"""Rough-terrain scenes for the model-based walker: random steps, stairs, ramps and a mixed course.

Every generator returns a list of obstacle dicts that models.humanoid.build_mjcf accepts (flat-topped boxes,
ramps as boxes tilted with "pitch"), each tagged "kind": "terrain" so planners treat it as walkable ground
rather than an obstacle to walk around (the session passes these geoms to GridMap.rasterize_truth as
walkable). Courses run along +x from the robot's start at the origin, centred on y = 0, and leave the start
and the far end on the floor so walks can start and stop on flat ground.

Sizes follow the walker's needs: every flat top is at least 0.3 m long (the default sole is 0.22 m), height
changes are at most a few centimetres per step for random terrain, and stair risers / ramp angles are
parameters (the walker's footstep planner puts each sole fully on one surface; see control/walk_terrain.py).
"""
from __future__ import annotations

import math

import numpy as np

TERRAIN_RGBA = {
    "steps": (0.55, 0.60, 0.66, 1.0),
    "stairs": (0.62, 0.58, 0.50, 1.0),
    "ramp": (0.50, 0.62, 0.55, 1.0),
}


OVERLAP = 0.002   # m: adjacent blocks overlap a little, so no ray or foot falls through a shared face


def _box(name: str, x0: float, x1: float, y0: float, y1: float, top: float, rgba, **extra) -> dict:
    """Axis-aligned box resting on the floor from x0..x1, y0..y1 (each widened by OVERLAP) up to `top`."""
    x0, x1, y0, y1 = x0 - OVERLAP, x1 + OVERLAP, y0 - OVERLAP, y1 + OVERLAP
    return {"name": name, "kind": "terrain", "shape": "box",
            "pos": [float((x0 + x1) / 2), float((y0 + y1) / 2)],
            "size": [float((x1 - x0) / 2), float((y1 - y0) / 2), float(top / 2)], "yaw": 0.0,
            "rgba": [float(v) for v in rgba], **extra}


def random_steps(seed: int | None = None, length: float = 2.6, x0: float = 0.4, width: float = 1.2,
                 h_min: float = 0.01, h_max: float = 0.05, cell: tuple[float, float] = (0.45, 0.7),
                 floor_prob: float = 0.2, lanes: int = 2) -> list[dict]:
    """Flat-topped blocks of random height, laid out in `lanes` lanes along x from x0 to x0 + length.

    Each lane is cut into cells of random length in `cell` (m); a cell is either floor (probability
    floor_prob, no box) or a box h_min..h_max high (default 1-5 cm). With two lanes (y < 0 and y > 0) the
    left and right feet usually stand on different heights. Returns obstacle dicts with "kind": "terrain"."""
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    edges = np.linspace(-width / 2, width / 2, lanes + 1)
    for lane in range(lanes):
        x = x0
        while x < x0 + length - 1e-6:
            seg = float(min(rng.uniform(*cell), x0 + length - x))
            if x0 + length - (x + seg) < cell[0] * 0.6:       # no sliver at the end
                seg = x0 + length - x
            if rng.random() >= floor_prob:
                h = float(np.round(rng.uniform(h_min, h_max), 3))
                out.append(_box(f"step block {len(out) + 1}", x, x + seg, float(edges[lane]), float(edges[lane + 1]),
                                h, TERRAIN_RGBA["steps"]))
            x += seg
    return out


def stairs(n_steps: int = 4, rise: float = 0.06, tread: float = 0.35, x0: float = 0.8, width: float = 1.2,
           landing: float = 0.8, down: bool = True) -> list[dict]:
    """A flight of n_steps steps up (risers `rise` m, treads `tread` m) starting at x = x0, a top landing
    `landing` m long, and (down=True) the mirrored flight back down to the floor.

    Defaults: 4 steps of 6 cm on 0.35 m treads. Each step is one box from the floor to its tread height, so
    the faces are vertical risers. Returns obstacle dicts with "kind": "terrain"."""
    if n_steps < 1 or rise <= 0 or tread <= 0:
        raise ValueError("stairs need n_steps >= 1, rise > 0 and tread > 0")
    out: list[dict] = []
    rgba = TERRAIN_RGBA["stairs"]
    x = x0
    for i in range(1, n_steps + 1):
        out.append(_box(f"stair {i}", x, x + tread, -width / 2, width / 2, i * rise, rgba))
        x += tread
    out.append(_box("stair landing", x, x + landing, -width / 2, width / 2, n_steps * rise, rgba))
    x += landing
    if down:
        for i in range(n_steps - 1, 0, -1):
            out.append(_box(f"stair {2 * n_steps - i}", x, x + tread, -width / 2, width / 2, i * rise, rgba))
            x += tread
    return out


def _ramp(name: str, x_start: float, length: float, angle: float, z_start: float, width: float,
          thickness: float = 0.04) -> dict:
    """A plank tilted about y whose top surface rises by length * tan(angle) from z_start at x_start
    (angle < 0: descends). The plank's lower part may sink into the floor (world geoms do not collide)."""
    run = float(length)
    slope_len = run / math.cos(angle)
    pitch = -angle                                           # MuJoCo: +pitch tips the +x end down
    t = thickness / 2
    # top-surface centre, then the box centre is t below it along the plank normal
    xc_top = x_start + run / 2
    zc_top = z_start + run / 2 * math.tan(angle)
    nx, nz = -math.sin(angle), math.cos(angle)
    return {"name": name, "kind": "terrain", "shape": "box",
            "pos": [float(xc_top - t * nx), 0.0], "z": float(zc_top - t * nz),
            "size": [float(slope_len / 2), float(width / 2), float(t)], "yaw": 0.0, "pitch": float(pitch),
            "rgba": [float(v) for v in TERRAIN_RGBA["ramp"]]}


def slope(angle_deg: float = 8.0, length: float = 1.0, x0: float = 0.8, width: float = 1.2, top: float = 0.8,
          down: bool = True) -> list[dict]:
    """A ramp up at angle_deg (<= ~12 deg is sensible) over `length` m of run from x = x0, a flat top
    `top` m long, and (down=True) a ramp back down. Returns obstacle dicts with "kind": "terrain"."""
    a = math.radians(angle_deg)
    if not 0 < a < math.radians(30):
        raise ValueError("slope angle must be between 0 and 30 deg")
    h = length * math.tan(a)
    out = [_ramp("ramp up", x0, length, a, 0.0, width)]
    out.append(_box("ramp top", x0 + length, x0 + length + top, -width / 2, width / 2, h, TERRAIN_RGBA["ramp"]))
    if down:
        out.append(_ramp("ramp down", x0 + length + top, length, -a, h, width))
    return out


def course_length(obstacles: list[dict]) -> float:
    """x of the far end of a course (the walk can stop on the floor beyond it)."""
    ends = []
    for o in obstacles:
        hx = o["size"][0] * (math.cos(o.get("pitch", 0.0)) if "pitch" in o else 1.0)
        ends.append(o["pos"][0] + hx)
    return float(max(ends)) if ends else 0.0


def mixed_course(seed: int | None = None, x0: float = 0.6) -> list[dict]:
    """Random steps (up to 4 cm), then a 6 deg ramp up and down, then a 3-step stair (5 cm risers) up and down,
    with flat floor between the sections. Returns obstacle dicts with "kind": "terrain"."""
    rng = np.random.default_rng(seed)
    out = random_steps(seed=int(rng.integers(1 << 30)), length=1.4, x0=x0, h_max=0.04)
    x = course_length(out) + 0.6
    out += slope(angle_deg=6.0, length=0.8, x0=x, top=0.6)
    x = course_length(out) + 0.6
    out += stairs(n_steps=3, rise=0.05, tread=0.35, x0=x, landing=0.7)
    return out


GENERATORS = {"random_steps": random_steps, "stairs": stairs, "slope": slope, "mixed": mixed_course}


def terrain_course(kind: str = "mixed", seed: int | None = None, **kw) -> tuple[list[dict], tuple[float, float]]:
    """(obstacles, goal_xy) for a named course ("random_steps", "stairs", "slope", "mixed"): the goal is on
    the floor 0.5 m beyond the end of the course, on the course's centre line (y = 0)."""
    if kind not in GENERATORS:
        raise ValueError(f"unknown terrain course {kind!r} (choose from {sorted(GENERATORS)})")
    fn = GENERATORS[kind]
    obs = fn(seed=seed, **kw) if kind in ("random_steps", "mixed") else fn(**kw)
    return obs, (course_length(obs) + 0.5, 0.0)
