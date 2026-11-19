"""Terrain for the model-based walker: ground height under the feet, footholds, swing clearance.

The ground is queried with mujoco.mj_ray, cast straight down against the world geoms of groups 1 and 2
(obstacles / terrain and the floor) - never the robot, whose geoms are group 0. A *foothold* is checked by
casting a 3 x 3 grid of rays over the sole rectangle and fitting a plane z = a + b x + c y to the hits:

* the plane gives the sole height and its tilt (ramps): the foot is pitched / rolled onto the surface;
* points far off the plane mean the sole straddles an edge (a step, a block), so that foothold is rejected
  and the footstep is moved fully onto or fully off the surface (the same rule as footstep planners that
  place each foot on one planar region; e.g. Deits & Tedrake, "Footstep planning on uneven terrain with
  mixed-integer convex optimization", Humanoids 2014; Griffin et al., "Footstep planning for autonomous
  walking over rough terrain", Humanoids 2019);
* a surface much higher or lower than the stance foot, too steep, or a free body (a prop) is rejected.

`repair_footsteps` walks through a planned footstep sequence and replaces every bad foothold by the nearest
good one along the step direction (or slightly outwards), within the step reach measured from the actual
stance foot - longer than on flat ground for steps that change height (stairs need the sole to reach the
next tread); a shorter step (up to 0.3 m back) is the fallback when nothing nearer fits.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .walk_footsteps import LEFT, wrap
from .walk_ik import rot_z

GROUPS = np.array([0, 1, 1, 0, 0, 0], np.uint8)     # mj_ray geom groups: obstacles / terrain (1), floor (2)


@dataclass
class TerrainLimits:
    max_rise: float = 0.10          # m, highest step up / down between consecutive footholds
    max_slope_deg: float = 15.0
    edge_tol: float = 0.003         # m, largest distance of a sole sample from the fitted plane (a kink too)
    margin: float = 0.025           # m, the sole is checked this much larger than it is (landing errors)
    flat_tol: float = 0.003         # m, below this the ground counts as the flat floor (z = 0, no tilt)


@dataclass
class Foothold:
    ok: bool
    z: float                        # sole centre height
    R: np.ndarray                   # sole orientation (heading yaw, tilted onto the surface)
    normal: np.ndarray
    why: str = ""

    @property
    def flat(self) -> bool:
        return abs(self.z) < 1e-9 and self.normal[2] > 1 - 1e-12


def tilted(yaw: float, normal) -> np.ndarray:
    """Rotation with heading `yaw` whose z axis is the surface normal (x stays in the vertical heading plane)."""
    n = np.asarray(normal, float)
    n = n / max(np.linalg.norm(n), 1e-12)
    if n[2] > 1 - 1e-12:
        return rot_z(yaw)
    h = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    x = h - np.dot(h, n) * n
    x /= max(np.linalg.norm(x), 1e-12)
    y = np.cross(n, x)
    return np.column_stack([x, y, n])


class TerrainMap:
    """Ground queries against the static world of one model (see the module docstring)."""

    def __init__(self, model: mujoco.MjModel, half_length: float, half_width: float,
                 limits: TerrainLimits | None = None, max_top: float = 0.5):
        self.model = model
        self.data = mujoco.MjData(model)               # the static world does not move: one forward is enough
        mujoco.mj_forward(model, self.data)
        self.hl, self.hw = float(half_length), float(half_width)
        self.lim = limits or TerrainLimits()
        self._gid = np.zeros(1, np.int32)
        self._n = np.zeros(3)
        self._down = np.array([0.0, 0.0, -1.0])
        tops = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != 0 or model.geom_type[g] in (mujoco.mjtGeom.mjGEOM_PLANE,
                                                                   mujoco.mjtGeom.mjGEOM_HFIELD):
                continue
            if model.geom_group[g] not in (1, 2):
                continue
            R = self.data.geom_xmat[g].reshape(3, 3)
            size = model.geom_size[g]
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                ext = float(np.abs(R[2]) @ size)
            else:
                ext = float(model.geom_rbound[g])
            tops.append(float(self.data.geom_xpos[g][2] + ext))
        # terrain = static geoms low enough to be walked on; tall ones (walls, tables) are only obstacles
        self.has_terrain = any(t <= max_top for t in tops)
        self.max_top = max_top

    # ------------------------------------------------------------ raw queries
    def height(self, x: float, y: float, z_from: float = 1.0) -> tuple[float, int, np.ndarray]:
        """(height, geom id, surface normal) under (x, y), casting from z_from down. No hit = (-inf, -1, up)."""
        d = mujoco.mj_ray(self.model, self.data, np.array([x, y, z_from]), self._down, GROUPS, 1, -1, self._gid,
                          self._n)
        if d < 0:
            return -np.inf, -1, np.array([0.0, 0.0, 1.0])
        return z_from - d, int(self._gid[0]), self._n.copy()

    def _samples(self, x, y, yaw, scale_l=1.0, scale_w=1.0) -> np.ndarray:
        hl, hw = self.hl * scale_l + self.lim.margin, self.hw * scale_w + self.lim.margin
        loc = np.array([[a, b] for a in (-hl, 0.0, hl) for b in (-hw, 0.0, hw)])
        c, s = np.cos(yaw), np.sin(yaw)
        return np.column_stack([x + c * loc[:, 0] - s * loc[:, 1], y + s * loc[:, 0] + c * loc[:, 1], loc])

    # ------------------------------------------------------------ footholds
    def foothold(self, x: float, y: float, yaw: float, z_ref: float = 0.0) -> Foothold:
        """Check a sole placed at (x, y) with heading yaw; z_ref = the height of the foot it steps from."""
        lim = self.lim
        pts = self._samples(x, y, yaw)
        z_from = z_ref + lim.max_rise + 0.25
        zs = np.empty(len(pts))
        for i, p in enumerate(pts):
            z, gid, _ = self.height(p[0], p[1], z_from)
            if gid >= 0 and self.model.geom_bodyid[gid] != 0:
                return Foothold(False, 0.0, rot_z(yaw), np.array([0, 0, 1.0]), "a free object is in the way")
            zs[i] = z
        if not np.isfinite(zs).all():
            return Foothold(False, 0.0, rot_z(yaw), np.array([0, 0, 1.0]), "no ground")
        if np.abs(zs).max() < lim.flat_tol:
            return Foothold(True, 0.0, rot_z(yaw), np.array([0.0, 0.0, 1.0]))
        A = np.column_stack([np.ones(len(pts)), pts[:, 2], pts[:, 3]])        # plane in the sole frame
        coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
        res = zs - A @ coef
        a, b, c = coef
        nl = np.array([-b, -c, 1.0])
        nl /= np.linalg.norm(nl)
        cy, sy = np.cos(yaw), np.sin(yaw)
        n = np.array([cy * nl[0] - sy * nl[1], sy * nl[0] + cy * nl[1], nl[2]])
        z = float(a + max(0.0, res.max()))                                    # rest on the highest points
        R = tilted(yaw, n)
        if res.max() - res.min() > lim.edge_tol:
            return Foothold(False, z, R, n, "edge under the sole")
        if np.degrees(np.arccos(min(1.0, n[2]))) > lim.max_slope_deg:
            return Foothold(False, z, R, n, "too steep")
        if abs(z - z_ref) > lim.max_rise:
            return Foothold(False, z, R, n, "step too high")
        return Foothold(True, z, R, n)

    def ground(self, x: float, y: float, yaw: float, z_ref: float = 0.0) -> Foothold:
        """Like foothold, but always returns the fitted pose (ok tells whether it is a good one)."""
        return self.foothold(x, y, yaw, z_ref)

    def clearance(self, start, end, z_ref: float = 0.0, n: int = 12) -> float:
        """Highest ground under the sole sweeping from start to end ((x, y, yaw) each), sampled along the
        line and across the sole's width and length."""
        z_from = z_ref + self.lim.max_rise + 0.3
        top = -np.inf
        yaw = float(start[2]) + float(wrap(end[2] - start[2])) / 2
        for t in np.linspace(0.0, 1.0, n):
            x = start[0] + (end[0] - start[0]) * t
            y = start[1] + (end[1] - start[1]) * t
            for p in self._samples(x, y, yaw):
                top = max(top, self.height(p[0], p[1], z_from)[0])
        return float(top)


# ------------------------------------------------------------------------------- footstep repair
@dataclass
class TerrainReach:
    forward: float                  # m, sole advance (stance frame) for a step that changes height
    backward: float
    min_width: float
    max_width: float


def _rel(stance, x, y):
    c, s = np.cos(stance[2]), np.sin(stance[2])
    dx, dy = x - stance[0], y - stance[1]
    return c * dx + s * dy, -s * dx + c * dy


def repair_footsteps(steps: list, left, right, terrain: TerrainMap, reach: TerrainReach, flat_forward: float,
                     z_feet: dict | None = None, shift: float = 0.16, ds: float = 0.01,
                     shift_back: float = 0.30) -> list:
    """Move footsteps off edges. steps: [[x, y, yaw, side], ...] as planned on flat ground; left / right: the
    current sole poses. Returns [[x, y, yaw, side, z, R (3x3)], ...] where every foothold is good when one
    could be found: each step is shifted along its own heading by up to `shift` m forward or `shift_back` m
    back (a shorter step is always within reach; then 3 / 6 cm outwards)
    to the nearest good foothold that the stance foot can reach (flat_forward ahead for level steps,
    reach.forward for steps that change height). The final two footsteps (the closing pair on the goal) are
    kept where they are. A step with no good foothold in reach keeps its planned place (its fitted ground
    height and tilt are still used)."""
    feet = {LEFT: np.asarray(left, float)[:3].copy(), 1 - LEFT: np.asarray(right, float)[:3].copy()}
    zf = dict(z_feet or {LEFT: 0.0, 1 - LEFT: 0.0})
    out: list = []
    n = len(steps)
    for i in range(n):
        x, y, yaw, side = steps[i][:4]
        side = int(side)
        stance = feet[1 - side]
        z_ref = zf[1 - side]
        closing = i >= n - 2
        best = None
        if closing:
            fh = terrain.foothold(x, y, yaw, z_ref)
            best = (x, y, fh)
        else:
            c, s = np.cos(yaw), np.sin(yaw)
            sign = 1.0 if side == LEFT else -1.0
            cands = []
            for lat in (0.0, 0.03, 0.06):
                for d in np.arange(-shift_back, shift + 1e-9, ds):
                    # first tier: within +-shift; only then the longer shifts back (shorter steps)
                    cands.append((abs(d) + 2.0 * lat + (10.0 if d < -shift - 1e-9 else 0.0), d, lat))
            cands.sort(key=lambda t: t[0])
            for _, d, lat in cands:
                cx = x + c * d - s * sign * lat
                cy = y + s * d + c * sign * lat
                fx, fy = _rel(stance, cx, cy)
                if sign * fy < reach.min_width - 1e-9 or sign * fy > reach.max_width:
                    continue
                if fx < -reach.backward:
                    continue
                if fx > reach.forward:
                    continue
                fh = terrain.foothold(cx, cy, yaw, z_ref)
                if not fh.ok:
                    continue
                if fx > flat_forward and abs(fh.z - z_ref) < terrain.lim.flat_tol:
                    continue                       # long steps only where the height changes
                best = (cx, cy, fh)
                break
            if best is None:
                fh = terrain.foothold(x, y, yaw, z_ref)
                best = (x, y, fh)
        cx, cy, fh = best
        out.append([float(cx), float(cy), float(yaw), side, float(fh.z), fh.R])
        feet[side] = np.array([cx, cy, yaw])
        zf[side] = float(fh.z)
    return out
