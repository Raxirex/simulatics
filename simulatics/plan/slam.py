"""2D lidar SLAM: the robot's pose and an occupancy map from leg odometry + lidar, without the true pose.

    odometry increment ──► prediction ──► correlative search ──► Gauss-Newton ──► pose ──► map update
       (noisy legs/IMU)    x_prev (+) u    (multi-resolution,     (smoothed          (log-odds,
                                            around the            likelihood          native ray
                                            prediction)           field + prior)      tracing)
    (experimental, off by default)  keyframes ──► loop closure (scan-to-submap search) ──► pose graph (LM)

Front end
---------
* **Odometry** (OdometrySim, OdomNoise): what a walking robot gets from leg kinematics + IMU. The true
  base motion between two readings is corrupted with a translation scale error, a heading drift per
  metre walked, a rotation scale error and random walks, deterministically from a seed. The estimator
  only ever sees these increments.
* **Scan** (planar_scan): the multi-channel lidar scan is levelled with the torso roll/pitch an IMU
  measures and collapsed to one range per azimuth (nearest obstacle return between z_min and z_max,
  like GridMap.integrate_lidar), in the robot's gravity-aligned heading frame. Returns from the top of
  low obstacles (horizontal surfaces whose hit point moves with the robot) are mapped but not matched.
* **Surfaces, not samples**: bearing-adjacent returns closer than seg_max_gap are joined into surface
  segments. The matcher uses only returns on such segments (plus points along them), and the map gets
  the gaps filled. A 180-ray lidar samples far or grazing walls sparsely and the sample pattern moves
  with the robot; matching isolated samples against the dots an earlier scan left pins the pose to
  where that scan was taken (it froze the along-corridor estimate in the corridor test).
* **Correlative scan matching** against the current map (E. Olson, "Real-time correlative scan
  matching", ICRA 2009): the map is turned into a likelihood field (each cell scores exp(-d^2 / 2 s^2),
  d = distance to the nearest occupied cell; Thrun, Burgard & Fox, Probabilistic Robotics 2005, 6.4),
  and every candidate pose in a window around the odometry prediction is scored by summing the field
  under the scan's end points, minus a quadratic motion prior (Olson's p(x | u)). The search is
  multi-resolution and exact: a coarse grid whose cells hold the max of the fine field over a B x B
  window bounds the score of B x B translations at once, and fine candidates are only scored inside
  coarse cells whose bound beats the best score so far (branch and bound as in Hess, Kohler, Rapp &
  Andor, "Real-time loop closure in 2D LIDAR SLAM", ICRA 2016). Field values are integers so the C++
  kernel (native/src/slam.cpp, OpenMP) and the numpy fallback make identical decisions.
* **Local refinement**: Gauss-Newton on the bilinearly interpolated likelihood field, minimising
  sum (1 - M(S_i(x)))^2 as in Hector SLAM (S. Kohlbrecher, O. von Stryk, J. Meyer, U. Klingauf, "A
  flexible and scalable SLAM system with full 3D motion estimation", SSRR 2011), plus the same
  motion prior. Its Hessian gives the match covariance; a near-singular translational block flags a
  degenerate (corridor-like) scene, where the prior keeps the estimate on the odometry along the
  unobservable direction instead of letting it jump.
* **Map**: log-odds occupancy built from the estimated poses with native.integrate_scan (same layout
  as GridMap.logodds, so the GUI and the planner can use it unchanged).

Back end (experimental, off by default: SlamParams(loop_closure=True))
---------------------------------------------------------------------
Keyframes every ~0.5 m / 20 deg form a pose graph. Consecutive keyframes are linked by the front end's
relative motion; when the robot comes back near an old keyframe after walking far enough, the current
scan is matched against a submap rendered from the old keyframes' scans with a wide correlative search,
verified with Gauss-Newton, and added as a loop-closure constraint. The graph (2D poses, relative-pose
constraints; Lu & Milios, "Globally consistent range scan alignment for environment mapping", Autonomous
Robots 1997; Grisetti, Kuemmerle, Stachniss & Burgard, "A tutorial on graph-based SLAM", IEEE ITS
Magazine 2010) is optimised with Levenberg-Marquardt, then every stored scan is re-anchored to its
corrected keyframe and the map is rebuilt.

Everything here is original code written for Simulatics (MIT). The papers are cited for the methods;
no code was taken from GMapping, Hector SLAM, Karto, Cartographer or slam_toolbox.
"""
from __future__ import annotations

import ctypes
import time
from dataclasses import asdict, dataclass

import numpy as np

from .. import native

Q = 10000  # likelihood field quantisation: field values are integers 0..Q (exact sums, see slam.cpp)


# =====================================================================================================
# SE(2)
# =====================================================================================================
def wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


def compose(a, b) -> np.ndarray:
    """a (+) b: pose b, given in the frame of pose a, expressed in a's parent frame."""
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], float(wrap(a[2] + b[2]))])


def inverse(a) -> np.ndarray:
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2]])


def between(a, b) -> np.ndarray:
    """a^-1 (+) b: pose b in the frame of pose a."""
    return compose(inverse(a), b)


def transform_points(pose, pts: np.ndarray) -> np.ndarray:
    c, s = np.cos(pose[2]), np.sin(pose[2])
    pts = np.asarray(pts, float)
    return np.stack([c * pts[:, 0] - s * pts[:, 1] + pose[0], s * pts[:, 0] + c * pts[:, 1] + pose[1]], axis=1)


def root_pose(data, body: int = 1) -> np.ndarray:
    """(x, y, yaw) of a body from MuJoCo's kinematics (xpos/xmat), i.e. at the same instant the lidar's
    site_xpos refers to. Used for ground truth and to feed the odometry simulator, never by Slam2D."""
    R = data.xmat[body].reshape(3, 3)
    return np.array([data.xpos[body][0], data.xpos[body][1], np.arctan2(R[1, 0], R[0, 0])])


# =====================================================================================================
# odometry
# =====================================================================================================
@dataclass
class OdomNoise:
    """Leg-odometry error model, applied to each true base increment (distance d, turn dyaw):

        measured translation = (1 + scale) * true + trans_walk * sqrt(d) * N(0, I)  + lateral * d (sideways)
        measured turn        = (1 + yaw_scale) * dyaw + yaw_drift * d + yaw_walk * sqrt(d) * N(0, 1)

    scale 0.03 means the robot believes it walked 3 % further than it did; yaw_drift_deg_per_m is a
    heading bias per metre walked (gyro bias + foot slip); the random walks are per sqrt(metre)."""
    scale: float = 0.03
    yaw_drift_deg_per_m: float = 2.0
    yaw_scale: float = 0.0
    trans_walk: float = 0.005
    yaw_walk_deg: float = 0.2
    lateral: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def none(cls) -> "OdomNoise":
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class OdometrySim:
    """Noisy odometry from the simulator's true base poses, deterministic given the seed.

    update(true_pose) -> odometry pose (x, y, yaw). The odometry frame starts at the first true pose
    (or at `odom_pose` passed to reset), so its trajectory is directly comparable to the truth.
    """

    def __init__(self, noise: OdomNoise | None = None, seed: int = 0):
        self.noise = noise or OdomNoise()
        self.seed = seed
        self.reset()

    def reset(self, true_pose=None, odom_pose=None) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._prev = None if true_pose is None else np.asarray(true_pose, float).copy()
        start = odom_pose if odom_pose is not None else true_pose
        self._started = start is not None
        self.pose = np.zeros(3) if start is None else np.asarray(start, float).copy()
        self.distance = 0.0

    def increment(self, true_delta) -> np.ndarray:
        """Corrupt one true increment (dx, dy, dyaw in the previous base frame)."""
        n = self.noise
        dx, dy, dth = (float(v) for v in true_delta)
        d = float(np.hypot(dx, dy))
        e = self.rng.standard_normal(3)  # always 3 draws: the sequence depends only on the call count
        sd = np.sqrt(d)
        mx = (1 + n.scale) * dx + n.trans_walk * sd * e[0]
        my = (1 + n.scale) * dy + n.trans_walk * sd * e[1] + n.lateral * d
        mth = (1 + n.yaw_scale) * dth + np.deg2rad(n.yaw_drift_deg_per_m) * d + np.deg2rad(n.yaw_walk_deg) * sd * e[2]
        self.distance += d
        return np.array([mx, my, mth])

    def update(self, true_pose) -> np.ndarray:
        true_pose = np.asarray(true_pose, float)
        if self._prev is None:
            self._prev = true_pose.copy()
            if not self._started:
                self.pose, self._started = true_pose.copy(), True
            return self.pose.copy()
        delta = between(self._prev, true_pose)
        self._prev = true_pose.copy()
        self.pose = compose(self.pose, self.increment(delta))
        return self.pose.copy()


# =====================================================================================================
# scans
# =====================================================================================================
@dataclass
class PlanarScan:
    points: np.ndarray   # (m, 2) end points in the robot frame (gravity-aligned, x forward, at the base)
    hit: np.ndarray      # (m,) bool: True = obstacle return, False = end of a free ray (no obstacle)
    origin: np.ndarray   # (2,) sensor position in the robot frame (rays start here)
    match: np.ndarray | None = None   # (m,) bool: obstacle returns usable for scan matching (default: hit)


def planar_scan(data, origin, points, hit, h_samples: int, root_body: int = 1,
                z_min: float = 0.15, z_max: float = 1.8, flat_dz: float = 0.03, face_dr: float = 0.08) -> PlanarScan:
    """One 3D lidar scan (Lidar.scan output, world frame, channels x h_samples rays row-major) ->
    planar scan in the robot frame, one point per azimuth.

    Only body-relative quantities survive: v_robot = Rz(yaw)^T (p_world - p_base). Since p_world =
    p_base + R_base (sensor offset + sensor-frame ray), this equals R_tilt (offset + ray) with
    R_tilt = Rz(yaw)^T R_base = Ry(pitch) Rx(roll), i.e. the sensor-frame scan levelled with the IMU's
    roll and pitch. The true base position and yaw cancel exactly; nothing of them reaches SLAM.
    Heights use the base height (leg kinematics on flat ground) + the levelled ray.

    `match` excludes returns from the TOP of low obstacles: close to a box lower than the sensor every
    beam clears its near face and lands on the top, at a range that moves with the robot (a horizontal
    surface biases matching). A column's nearest return counts as such when another channel hits the
    same height (+- flat_dz) farther out and no channel hits near its range (vertical face). These
    points are still mapped as obstacles; only the matcher ignores them.
    """
    pb = data.xpos[root_body]
    R = data.xmat[root_body].reshape(3, 3)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    rel = (np.asarray(points, float) - pb) @ Rz          # rows: Rz^T v
    o = (np.asarray(origin, float) - pb) @ Rz
    z = rel[:, 2] + pb[2]
    obstacle = (np.asarray(hit, bool) & (z > z_min) & (z < z_max)).reshape(-1, h_samples)
    xy = rel[:, :2].reshape(-1, h_samples, 2)
    r = np.linalg.norm(xy - o[:2], axis=2)
    has = obstacle.any(axis=0)
    row = np.where(has, np.where(obstacle, r, np.inf).argmin(axis=0), np.where(obstacle, -1.0, r).argmax(axis=0))
    cols = np.arange(h_samples)
    ends = xy[row, cols]
    r0, z0 = r[row, cols], z.reshape(-1, h_samples)[row, cols]
    zz = z.reshape(-1, h_samples)
    other = obstacle & (np.arange(obstacle.shape[0])[:, None] != row[None, :])
    flat = (other & (np.abs(zz - z0) < flat_dz) & (r - r0 > face_dr)).any(axis=0)
    face = (other & (np.abs(r - r0) <= face_dr)).any(axis=0)
    return PlanarScan(ends, has, o[:2].copy(), has & (face | ~flat))


def scan_robot_frame(model, data, lidar, z_min: float = 0.15, z_max: float = 1.8) -> PlanarScan:
    """Cast the lidar (sim/sensors.Lidar) and return the planar robot-frame scan."""
    origin, pts, _, hit = lidar.scan(model, data)
    return planar_scan(data, origin, pts, hit, lidar.cfg.h_samples, lidar.root, z_min, z_max)


# =====================================================================================================
# kernels: native (native/src/slam.cpp) with numpy fallbacks that make identical decisions
# =====================================================================================================
_i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
_i64p = np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS")
_f32p = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
_lib = None
_lib_tried = False


def _load():
    global _lib, _lib_tried
    if _lib_tried:
        return _lib
    _lib_tried = True
    lib = native.load_library("slam")
    if lib is None:
        return None
    try:
        ci, cf, cll = ctypes.c_int, ctypes.c_float, ctypes.c_longlong
        lib.sl_version.restype = ci
        lib.sl_likelihood_field.argtypes = [_f32p, ci, ci, cf, _i32p, ci, _i32p, ci]
        lib.sl_likelihood_field.restype = None
        lib.sl_window_max.argtypes = [_i32p, ci, ci, ci, _i32p, ci]
        lib.sl_window_max.restype = None
        lib.sl_score_grid.argtypes = [_i32p, ci, ci, _i32p, ci, ci, ci, _i64p, cll, _i64p, ci]
        lib.sl_score_grid.restype = None
        lib.sl_search.argtypes = [_i32p, _i32p, ci, ci, _i32p, ci, ci, ci, ci, _i64p, cll, _i32p, _i64p, ci]
        lib.sl_search.restype = ci
        _lib = lib
    except AttributeError:   # an older build without these symbols
        _lib = None
    return _lib


def native_available() -> bool:
    return _load() is not None


def likelihood_lut(sigma_cells: float, cutoff_cells: float) -> np.ndarray:
    """int32 table: field value for squared distance d2 (cells), 0 at and beyond the cutoff."""
    n = int(np.floor(cutoff_cells ** 2)) + 1
    d2 = np.arange(n, dtype=float)
    return np.rint(Q * np.exp(-0.5 * d2 / sigma_cells ** 2)).astype(np.int32)


def likelihood_field(logodds: np.ndarray, thresh: float, lut: np.ndarray, use_native: bool = True,
                     threads: int = 2) -> np.ndarray:
    """int32 field (h, w): lut[squared distance to the nearest cell with logodds > thresh], 0 beyond the
    table. Exact Euclidean distances in both implementations."""
    logodds = np.ascontiguousarray(logodds, np.float32)
    lut = np.ascontiguousarray(lut, np.int32)
    h, w = logodds.shape
    lib = _load() if use_native else None
    if lib is not None:
        out = np.empty((h, w), np.int32)
        lib.sl_likelihood_field(logodds, w, h, float(np.float32(thresh)), lut, len(lut), out, int(threads))
        return out
    from scipy.ndimage import distance_transform_edt
    occ = logodds > np.float32(thresh)
    if not occ.any():
        return np.zeros((h, w), np.int32)
    d = distance_transform_edt(~occ)
    d2 = np.rint(d * d).astype(np.int64)
    return np.where(d2 < len(lut), lut[np.minimum(d2, len(lut) - 1)], 0).astype(np.int32)


def window_max(field: np.ndarray, B: int, use_native: bool = True, threads: int = 2) -> np.ndarray:
    """(h + B - 1, w + B - 1): out[y + B - 1, x + B - 1] = max field[y:y+B, x:x+B] (outside = 0),
    for x, y from -(B - 1)."""
    field = np.ascontiguousarray(field, np.int32)
    h, w = field.shape
    lib = _load() if use_native else None
    if lib is not None:
        out = np.empty((h + B - 1, w + B - 1), np.int32)
        lib.sl_window_max(field, w, h, int(B), out, int(threads))
        return out
    if B == 1:
        return field.copy()
    p = np.pad(field, B - 1)
    from numpy.lib.stride_tricks import sliding_window_view
    a = sliding_window_view(p, B, axis=1).max(axis=-1)         # (h + 2B - 2, w + B - 1)
    return np.ascontiguousarray(sliding_window_view(a, B, axis=0).max(axis=-1))


def _gather_sum(F: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """sum over the last axis of F[y, x] (0 outside the grid), int64."""
    h, w = F.shape
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    v = F[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]
    return np.where(inside, v, 0).sum(axis=-1, dtype=np.int64)


def score_grid(field: np.ndarray, cells: np.ndarray, win: int, rot_pen: np.ndarray, lam: int,
               use_native: bool = True, threads: int = 2) -> np.ndarray:
    """Brute force: (K, 2 win + 1, 2 win + 1) int64 scores S(k, dy, dx) (see slam.cpp)."""
    field = np.ascontiguousarray(field, np.int32)
    cells = np.ascontiguousarray(cells, np.int32)
    rot_pen = np.ascontiguousarray(rot_pen, np.int64)
    K, n = cells.shape[:2]
    S = 2 * win + 1
    lib = _load() if use_native else None
    if lib is not None:
        out = np.empty((K, S, S), np.int64)
        lib.sl_score_grid(field, field.shape[1], field.shape[0], cells, K, n, int(win), rot_pen, int(lam), out,
                          int(threads))
        return out
    t = np.arange(-win, win + 1)
    out = np.empty((K, S, S), np.int64)
    for k in range(K):
        x = cells[k, :, 0][None, None, :] + t[None, :, None]
        y = cells[k, :, 1][None, None, :] + t[:, None, None]
        out[k] = _gather_sum(field, x, y) - n * (rot_pen[k] + lam * (t[:, None] ** 2 + t[None, :] ** 2))
    return out


def search(field: np.ndarray, fmax: np.ndarray, cells: np.ndarray, win: int, B: int, rot_pen: np.ndarray,
           lam: int, use_native: bool = True, threads: int = 2) -> tuple[int, int, int, int, int]:
    """Exact branch-and-bound maximiser of S(k, dx, dy) -> (k, dx, dy, score, expanded coarse cells)."""
    field = np.ascontiguousarray(field, np.int32)
    fmax = np.ascontiguousarray(fmax, np.int32)
    cells = np.ascontiguousarray(cells, np.int32)
    rot_pen = np.ascontiguousarray(rot_pen, np.int64)
    K, n = cells.shape[:2]
    h, w = field.shape
    lib = _load() if use_native else None
    if lib is not None:
        best = np.zeros(3, np.int32)
        sc = np.zeros(1, np.int64)
        ex = lib.sl_search(field, fmax, w, h, cells, K, n, int(win), int(B), rot_pen, int(lam), best, sc,
                           int(threads))
        return int(best[0]), int(best[1]), int(best[2]), int(sc[0]), int(ex)
    nv = (2 * win) // B + 1
    st = -win + B * np.arange(nv)
    en = np.minimum(st + B - 1, win)
    ms = np.where((st <= 0) & (en >= 0), 0, np.minimum(np.abs(st), np.abs(en)) ** 2).astype(np.int64)
    cx, cy = cells[..., 0].astype(np.int64), cells[..., 1].astype(np.int64)
    X = cx[:, None, None, :] + (st + B - 1)[None, None, :, None]
    Y = cy[:, None, None, :] + (st + B - 1)[None, :, None, None]
    ub = _gather_sum(fmax, X, Y) - n * (rot_pen[:, None, None] + lam * (ms[None, :, None] + ms[None, None, :]))
    flat = ub.ravel()
    order = np.argsort(-flat, kind="stable")
    top, bk, bx, by, expanded = None, -1, 0, 0, 0
    for v in order:
        if top is not None and flat[v] <= top:
            break
        expanded += 1
        k, iy, ix = np.unravel_index(int(v), ub.shape)
        dys = np.arange(st[iy], en[iy] + 1)
        dxs = np.arange(st[ix], en[ix] + 1)
        s = (_gather_sum(field, cx[k][None, None, :] + dxs[None, :, None], cy[k][None, None, :] + dys[:, None, None])
             - n * (rot_pen[k] + lam * (dys[:, None] ** 2 + dxs[None, :] ** 2)))
        j = int(np.argmax(s))
        if top is None or s.flat[j] > top:
            top = int(s.flat[j])
            bk, by, bx = int(k), int(dys[j // len(dxs)]), int(dxs[j % len(dxs)])
    return bk, bx, by, int(top), expanded


# =====================================================================================================
# parameters
# =====================================================================================================
@dataclass
class SlamParams:
    # likelihood field
    sigma: float = 0.0              # m, field std; 0 = auto (max(0.075, 1.5 res))
    cutoff: float = 3.0             # field is 0 beyond cutoff * sigma
    occ_thresh: float = 0.0         # log-odds above which a cell is an obstacle for matching
    # correlative search (front end)
    win_xy: float = 0.3             # m, +- window around the prediction
    win_yaw_deg: float = 6.0        # deg
    coarse: int = 4                 # coarse cell = coarse x coarse fine cells
    prior_xy: float = 1.0           # motion prior, mean-likelihood units per m^2
    prior_yaw: float = 10.0         # ... per rad^2
    # Gauss-Newton refinement
    gn_iters: int = 12
    # scan handling
    min_range: float = 0.3
    match_max_range: float = 0.0    # m, farthest return used for matching; 0 = auto: sensor range - margin
    match_range_margin: float = 2.0  # (sensor range = longest free ray seen). Returns near the range limit
                                    # would make the edge of the mapped area look like the end of a wall
    seg_max_gap: float = 0.3        # m: neighbouring returns closer than this are one surface (see update)
    match_step: float = 0.1         # m: spacing of the surface samples the matcher uses
    min_points: int = 12            # fewer obstacle returns: dead reckoning
    min_score: float = 0.1          # mean likelihood below which the match is rejected (dead reckoning)
    degenerate_ratio: float = 0.05  # min / max eigenvalue of the translational information
    # map
    l_free: float = -0.4
    l_occ: float = 0.9
    l_clamp: float = 4.0
    map_update_dist: float = 0.05   # m moved since the last insertion
    map_update_deg: float = 2.0
    map_update_every: int = 10      # scans: insert at least this often
    # back end
    loop_closure: bool = False      # experimental back end (see module docstring), off by default
    keyframe_dist: float = 0.5
    keyframe_deg: float = 20.0
    lc_radius: float = 2.5          # m, candidate keyframes near the current pose
    lc_min_travel: float = 5.0      # m walked since the candidate keyframe
    lc_min_age: int = 10            # keyframes between candidate and current
    lc_submap: int = 4              # keyframes on each side of the candidate rendered into its submap
    lc_win_xy: float = 1.0
    lc_win_yaw_deg: float = 20.0
    lc_min_score: float = 0.55      # mean likelihood a loop closure must reach
    lc_every: int = 1               # try loop closures at every n-th keyframe
    # misc
    threads: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


# =====================================================================================================
# SLAM
# =====================================================================================================
class Slam2D:
    """2D lidar SLAM (see module docstring).

        slam = Slam2D(size=24.0, res=0.05, odom_noise=OdomNoise(), seed=0)
        slam.reset(initial_pose)                         # map frame = where the robot starts
        odom = slam.odometry.update(true_base_pose)      # simulator only: noisy leg odometry
        x, y, yaw = slam.update(odom, scan.points, scan.hit, origin=scan.origin, match=scan.match)

    `odom_noise` / `seed` configure `slam.odometry` (an OdometrySim for simulator users); the estimator
    itself only sees what update() receives. map_logodds has GridMap.logodds's layout ((n, n) float32,
    row y, column x, origin at -size/2).
    """

    def __init__(self, size: float = 24.0, res: float = 0.05, odom_noise: OdomNoise | None = None, seed: int = 0,
                 params: SlamParams | None = None, initial_pose=(0.0, 0.0, 0.0), use_native: bool = True):
        self.size, self.res = float(size), float(res)
        self.n = int(round(size / res))
        self.origin = np.array([-size / 2, -size / 2])
        self.p = params or SlamParams()
        self.odom_noise = odom_noise or OdomNoise()
        self.odometry = OdometrySim(self.odom_noise, seed)
        self.use_native = bool(use_native) and native_available()
        self.logodds = np.zeros((self.n, self.n), np.float32)
        sig = self.p.sigma if self.p.sigma > 0 else max(0.075, 1.5 * self.res)
        self.sigma = sig
        self._lut = likelihood_lut(sig / self.res, self.p.cutoff * sig / self.res)
        self.max_traj = 36000                     # trajectory samples kept (1 h at 10 Hz)
        self.reset(initial_pose)

    # ------------------------------------------------------------------ public API
    def reset(self, initial_pose=(0.0, 0.0, 0.0)) -> None:
        """Clear the map and the trajectory; the map frame is anchored at initial_pose."""
        self.logodds[:] = 0
        self._pose = np.asarray(initial_pose, float).reshape(3).copy()
        self._pose[2] = float(wrap(self._pose[2]))
        self.initial_pose = self._pose.copy()
        self._odom_prev = None
        self._odom_map = self._pose.copy()        # dead-reckoning-only pose in the map frame
        self._map_version = 0
        self._field_version = -1
        self._field = self._fieldf = self._fmax = None
        self._last_insert = None
        self._since_insert = 0
        self.scans = 0
        self.inserted = 0
        self.state = "init"
        self.score = 0.0
        self.cov = np.diag([1e-6, 1e-6, 1e-8])
        self.match_ms = 0.0
        self.update_ms = 0.0
        self._ms_sum = 0.0
        self._matched = 0
        self.max_correction = 0.0
        self.loop_closures = 0
        self.traj_est: list[np.ndarray] = []      # estimates, re-anchored when the pose graph changes
        self.traj_online: list[np.ndarray] = []   # estimates as they were at the time (causal)
        self.traj_odom: list[np.ndarray] = []     # dead reckoning only, in the map frame
        self._traj_kf: list[int] = []             # keyframe each estimate is anchored to (-1: none)
        # back end
        self._scans: list[tuple] = []             # inserted scans: (kf index, pose rel. to kf, pts, hit, origin)
        self.kf_pose: list[np.ndarray] = []       # keyframe poses (optimised)
        self.kf_scan: list[tuple] = []            # keyframe scans: (pts (m, 2) obstacle returns, origin)
        self.kf_dist: list[float] = []            # odometry distance walked at each keyframe
        self.edges: list[tuple] = []              # (i, j, z_ij, info 3x3, kind)
        self._walked = 0.0
        self._sensor_range = np.inf if self.p.match_max_range > 0 else 0.0
        self._dict_cache = None
        self.odometry.reset()

    @property
    def pose(self) -> tuple[float, float, float]:
        return float(self._pose[0]), float(self._pose[1]), float(self._pose[2])

    @property
    def map_logodds(self) -> np.ndarray:
        return self.logodds

    def occupancy(self) -> np.ndarray:
        """uint8 (n, n), 1 = occupied: same rule as GridMap.lidar_occ (log-odds > 0.5)."""
        return (self.logodds > 0.5).astype(np.uint8)

    def image(self) -> np.ndarray:
        """uint8 (n, n): 0 unknown, 1 free, 2 occupied (same as GridMap.lidar_image)."""
        img = np.zeros((self.n, self.n), np.uint8)
        img[self.logodds < -0.5] = 1
        img[self.logodds > 0.5] = 2
        return img

    def resampled(self, res: float) -> np.ndarray:
        """The log-odds map on a coarser grid (res an integer multiple of self.res), e.g. 0.1 m to drop
        into a GridMap(24, 0.1): a block is occupied evidence if any cell is (max), otherwise free (min)."""
        f = int(round(res / self.res))
        if f <= 1:
            return self.logodds.copy()
        m = self.n // f
        b = self.logodds[:m * f, :m * f].reshape(m, f, m, f)
        hi, lo = b.max(axis=(1, 3)), b.min(axis=(1, 3))
        return np.where(hi > 0.5, hi, lo).astype(np.float32)

    def gridmap(self):
        """A GridMap whose lidar map IS this SLAM map (shared array), for GridMap.plan / Navigator with
        source="lidar". Its truth layer is empty (rasterise it yourself if needed)."""
        from .grid import GridMap
        g = GridMap(self.size, self.res)
        g.logodds = self.logodds
        return g

    def stats(self) -> dict:
        ev = np.linalg.eigvalsh(self.cov[:2, :2])
        return {"scans": self.scans, "match_ms": round(self.match_ms, 3),
                "match_ms_mean": round(self._ms_sum / max(self._matched, 1), 3),
                "update_ms": round(self.update_ms, 3), "score": round(self.score, 4),
                "loop_closures": self.loop_closures, "state": self.state, "keyframes": len(self.kf_pose),
                "inserted": self.inserted, "native": self.use_native,
                "sigma_xy": [round(float(np.sqrt(max(e, 0))), 4) for e in ev],
                "sigma_yaw_deg": round(float(np.rad2deg(np.sqrt(max(self.cov[2, 2], 0)))), 4),
                "cov": np.round(self.cov, 8).tolist(), "max_correction": round(self.max_correction, 4)}

    def to_dict(self, max_points: int = 400) -> dict:
        """Everything the GUI needs: pose for a ghost marker, uncertainty ellipse, both trajectories
        (thinned to <= max_points + 1 each; cached until the next update, so 30 Hz frames are cheap)."""
        if self._dict_cache is not None and self._dict_cache[0] == max_points:
            return self._dict_cache[1]
        ev, vec = np.linalg.eigh(self.cov[:2, :2])
        ang = float(np.arctan2(vec[1, 1], vec[0, 1]))

        def thin(tr):
            if not tr:
                return []
            a = np.asarray(tr)[:, :2]
            step = max(1, int(np.ceil(len(a) / max_points)))
            a = np.vstack([a[::step], a[-1:]])
            return np.round(a, 3).tolist()

        out = {"pose": [round(v, 4) for v in self.pose], "state": self.state, "score": round(self.score, 3),
               "ellipse": [round(float(np.sqrt(max(ev[1], 0))), 4), round(float(np.sqrt(max(ev[0], 0))), 4),
                           round(ang, 4)],
               "sigma_yaw_deg": round(float(np.rad2deg(np.sqrt(max(self.cov[2, 2], 0)))), 3),
               "loop_closures": self.loop_closures, "scans": self.scans, "match_ms": round(self.match_ms, 2),
               "traj": thin(self.traj_est), "odom_traj": thin(self.traj_odom)}
        self._dict_cache = (max_points, out)
        return out

    def to_controller_frame(self, path_xy, ctrl_pose) -> np.ndarray:
        """Map-frame points -> the frame of a controller that believes the robot is at ctrl_pose
        (e.g. the simulator's true pose). The robot then walks, relative to its body, exactly the path
        planned on the SLAM map from the estimated pose."""
        T = compose(np.asarray(ctrl_pose, float), inverse(self._pose))
        pts = np.asarray(path_xy, float).reshape(-1, 2)
        return transform_points(T, pts) if len(pts) else pts.copy()

    def update(self, odom, points, hit=None, origin=(0.0, 0.0), delta: bool = False,
               match=None) -> tuple[float, float, float]:
        """One lidar scan.

        odom: odometry pose (x, y, yaw) in the odometry frame, or with delta=True the increment since the
        previous update (in the previous robot frame). points: (m, 2|3) scan end points in the robot
        frame; hit: (m,) bool, False for free-space rays without an obstacle return (default all True);
        origin: sensor position in the robot frame; match: (m,) bool, obstacle returns the matcher may use
        (PlanarScan.match; default = hit). Returns the estimated (x, y, yaw) in the map frame.
        """
        t0 = time.perf_counter()
        odom = np.asarray(odom, float).reshape(3)
        if delta:
            d = odom.copy()
        else:
            d = np.zeros(3) if self._odom_prev is None else between(self._odom_prev, odom)
            self._odom_prev = odom.copy()
        pred = compose(self._pose, d)
        self._odom_map = compose(self._odom_map, d)
        self._walked += float(np.hypot(d[0], d[1]))

        origin = np.asarray(origin, float).reshape(2)
        pts, hit, use, rng = self._prepare(points, hit, origin, match)
        if (~hit).any():
            self._sensor_range = max(self._sensor_range, float(rng[~hit].max()))
        if self.p.match_max_range > 0:
            lim = self.p.match_max_range
        else:   # no free ray seen yet (enclosed space): the range limit has not been reached either
            lim = self._sensor_range - self.p.match_range_margin if self._sensor_range > 0 else np.inf
        # Match surfaces, not samples: a sparse lidar samples a wall at spacings that grow with range and
        # incidence, and those samples move with the robot. Matching isolated samples against the dots
        # an earlier scan left in the map pins the pose to where that scan was taken (aliasing; it froze
        # the along-corridor estimate). So the matcher uses returns that have an angular neighbour on the
        # same surface (gap <= seg_max_gap) plus points along those gaps, and the map gets the gaps filled.
        cand = use & (rng <= lim)
        virt, member = self.surface_points(pts, cand, max(self.p.match_step, self.res))
        obs = np.vstack([pts[cand & member], virt])

        if self.inserted == 0:
            pose, self.state, self.score = pred, "init", 0.0
        elif len(obs) < self.p.min_points:
            pose, self.state, self.score = pred, "odometry", 0.0
            self.cov = self.cov + self._odom_cov(d)
        else:
            pose = self._match(obs, pred)
            self.match_ms = (time.perf_counter() - t0) * 1e3
            self._ms_sum += self.match_ms
            self._matched += 1
            self.max_correction = max(self.max_correction, float(np.hypot(*(pose[:2] - pred[:2]))))
        self._pose = pose
        self.scans += 1
        self._maybe_insert(pts, hit, origin, use)  # may run the back end and move self._pose
        self.traj_est.append(self._pose.copy())
        self.traj_online.append(self._pose.copy())
        self._traj_kf.append(len(self.kf_pose) - 1)
        self.traj_odom.append(self._odom_map.copy())
        if len(self.traj_est) > self.max_traj:    # bounded memory on long sessions (oldest dropped)
            k = len(self.traj_est) - self.max_traj
            for tr in (self.traj_est, self.traj_online, self._traj_kf, self.traj_odom):
                del tr[:k]
        self._dict_cache = None
        self.update_ms = (time.perf_counter() - t0) * 1e3
        return self.pose

    # ------------------------------------------------------------------ scans
    def _prepare(self, points, hit, origin, match=None):
        """Finite returns beyond min_range, sorted by bearing -> (pts, hit, use, range)."""
        pts = np.asarray(points, float).reshape(len(points), -1)[:, :2]
        hit = np.ones(len(pts), bool) if hit is None else np.asarray(hit, bool).reshape(-1)
        use = hit if match is None else (np.asarray(match, bool).reshape(-1) & hit)
        rel = pts - origin
        rng = np.hypot(rel[:, 0], rel[:, 1])
        ok = np.isfinite(pts).all(axis=1) & (rng >= self.p.min_range)
        order = np.argsort(np.arctan2(rel[ok, 1], rel[ok, 0]), kind="stable")
        return pts[ok][order], hit[ok][order], use[ok][order], rng[ok][order]

    def surface_points(self, pts: np.ndarray, mask: np.ndarray, step: float):
        """Points every `step` m along the gaps between bearing-adjacent returns (pts sorted by bearing,
        wrapping around) that are both in `mask` and at most seg_max_gap apart -> (virtual points (k, 2),
        mask of the returns that belong to such a surface segment)."""
        m = len(pts)
        if m < 2:
            return np.zeros((0, 2)), np.zeros(m, bool)
        nxt = (np.arange(m) + 1) % m
        d = pts[nxt] - pts
        gap = np.hypot(d[:, 0], d[:, 1])
        seg = mask & mask[nxt] & (gap <= self.p.seg_max_gap) & (gap > 1e-9)
        member = seg | np.roll(seg, 1)
        i = np.nonzero(seg)[0]
        k = np.maximum(np.ceil(gap[i] / step).astype(int) - 1, 0)    # interior points per gap
        if k.sum() == 0:
            return np.zeros((0, 2)), member
        rep = np.repeat(i, k)
        j = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k) + 1
        t = j / np.repeat(k + 1, k)
        return pts[rep] + t[:, None] * d[rep], member

    def insertion_scan(self, points, hit=None, origin=(0.0, 0.0)):
        """The (points, hit) a scan contributes to the map: the returns plus the filled surface gaps."""
        origin = np.asarray(origin, float).reshape(2)
        pts, hit, _, _ = self._prepare(points, hit, origin)
        virt, _ = self.surface_points(pts, hit, self.res)
        return np.vstack([pts, virt]), np.concatenate([hit, np.ones(len(virt), bool)])

    # ------------------------------------------------------------------ front end
    def _odom_cov(self, d) -> np.ndarray:
        n = self.odom_noise
        dist = float(np.hypot(d[0], d[1]))
        st = max(n.scale, 0.01) * dist + max(n.trans_walk, 1e-3) * np.sqrt(dist)
        sy = np.deg2rad(max(n.yaw_drift_deg_per_m, 0.5)) * dist + np.deg2rad(max(n.yaw_walk_deg, 0.05)) * np.sqrt(dist)
        return np.diag([st ** 2, st ** 2, sy ** 2])

    def _ensure_field(self) -> None:
        if self._field_version == self._map_version:
            return
        thr = self.p.threads
        self._field = likelihood_field(self.logodds, self.p.occ_thresh, self._lut, self.use_native, thr)
        self._fieldf = self._field.astype(np.float32) * np.float32(1.0 / Q)
        self._fmax = window_max(self._field, self.p.coarse, self.use_native, thr)
        self._field_version = self._map_version

    def _candidates(self, obs: np.ndarray, pred: np.ndarray, win_xy: float, win_yaw: float,
                    prior_xy: float, prior_yaw: float):
        """Rotations around the prediction and the scan's cells under each of them."""
        r = np.hypot(obs[:, 0], obs[:, 1])
        dth = self.res / max(float(r.max()), 2.0)          # farthest point moves about one cell per step
        K0 = int(np.ceil(win_yaw / dth))
        steps = np.arange(-K0, K0 + 1)
        th = pred[2] + dth * steps
        c, s = np.cos(th)[:, None], np.sin(th)[:, None]
        wx = c * obs[:, 0] - s * obs[:, 1] + pred[0]
        wy = s * obs[:, 0] + c * obs[:, 1] + pred[1]
        cells = np.stack([np.floor((wx - self.origin[0]) / self.res), np.floor((wy - self.origin[1]) / self.res)],
                         axis=-1).astype(np.int32)
        rot_pen = np.rint(Q * prior_yaw * (dth * steps) ** 2).astype(np.int64)
        win = max(1, int(np.ceil(win_xy / self.res)))
        lam = int(np.rint(Q * prior_xy * self.res ** 2))
        return th, np.ascontiguousarray(cells), rot_pen, win, lam

    def correlative(self, obs: np.ndarray, pred: np.ndarray, win_xy: float | None = None,
                    win_yaw: float | None = None, field=None, fmax=None, prior_scale: float = 1.0):
        """Best pose of the scan `obs` (robot frame) within the window around `pred` on the current
        likelihood field -> (pose, mean likelihood without prior, expanded coarse cells)."""
        win_xy = self.p.win_xy if win_xy is None else win_xy
        win_yaw = np.deg2rad(self.p.win_yaw_deg) if win_yaw is None else win_yaw
        if field is None:
            self._ensure_field()
            field, fmax = self._field, self._fmax
        th, cells, rot_pen, win, lam = self._candidates(obs, pred, win_xy, win_yaw, prior_scale * self.p.prior_xy,
                                                        prior_scale * self.p.prior_yaw)
        k, dx, dy, s, ex = search(field, fmax, cells, win, self.p.coarse, rot_pen, lam, self.use_native,
                                  self.p.threads)
        n = len(obs)
        raw = (s + n * (int(rot_pen[k]) + lam * (dx * dx + dy * dy))) / (Q * n)
        pose = np.array([pred[0] + dx * self.res, pred[1] + dy * self.res, float(wrap(th[k]))])
        return pose, float(raw), ex

    def _residuals(self, F: np.ndarray, obs: np.ndarray, x: np.ndarray, jac: bool = True):
        """Bilinear likelihood M at the transformed points, r = 1 - M, and dM/dx (n, 3)."""
        c, s = np.cos(x[2]), np.sin(x[2])
        px, py = obs[:, 0], obs[:, 1]
        wx = c * px - s * py + x[0]
        wy = s * px + c * py + x[1]
        u = (wx - self.origin[0]) / self.res - 0.5      # cell centres at integer coordinates
        v = (wy - self.origin[1]) / self.res - 0.5
        x0, y0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
        a, b = u - x0, v - y0
        h, w = F.shape
        inside = (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1)
        xc, yc = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
        f00 = F[yc, xc].astype(float)
        f10 = F[yc, xc + 1].astype(float)
        f01 = F[yc + 1, xc].astype(float)
        f11 = F[yc + 1, xc + 1].astype(float)
        M = np.where(inside, (1 - b) * ((1 - a) * f00 + a * f10) + b * ((1 - a) * f01 + a * f11), 0.0)
        r = 1.0 - M
        if not jac:
            return r, None
        gu = np.where(inside, (1 - b) * (f10 - f00) + b * (f11 - f01), 0.0) / self.res
        gv = np.where(inside, (1 - a) * (f01 - f00) + a * (f11 - f10), 0.0) / self.res
        J = np.stack([gu, gv, gu * (-s * px - c * py) + gv * (c * px - s * py)], axis=1)
        return r, J

    def refine(self, obs: np.ndarray, x0: np.ndarray, pred: np.ndarray, F: np.ndarray | None = None,
               prior: bool = True):
        """Gauss-Newton on sum (1 - M)^2 + n * prior (Hector-style, with the motion prior) ->
        (pose, mean likelihood, Hessian incl. prior, scan-only J^T J, sum of squared residuals)."""
        F = self._fieldf if F is None else F
        n = len(obs)
        lam = n * np.array([self.p.prior_xy, self.p.prior_xy, self.p.prior_yaw]) if prior else np.zeros(3)

        def cost(x, r):
            e = x - pred
            e[2] = wrap(e[2])
            return float(r @ r + (lam * e * e).sum())

        x = np.asarray(x0, float).copy()
        r, J = self._residuals(F, obs, x)
        c0 = cost(x, r)
        for _ in range(self.p.gn_iters):
            A = J.T @ J
            e = pred - x
            e[2] = wrap(e[2])
            H = A + np.diag(lam) + 1e-9 * np.eye(3)
            g = J.T @ r + lam * e
            try:
                dlt = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            m = max(1.0, np.hypot(dlt[0], dlt[1]) / (2 * self.res), abs(dlt[2]) / 0.05)
            dlt /= m
            improved = False
            for _ in range(6):
                xn = x + dlt
                xn[2] = wrap(xn[2])
                rn, Jn = self._residuals(F, obs, xn)
                cn = cost(xn, rn)
                if cn <= c0:
                    x, r, J, c0, improved = xn, rn, Jn, cn, True
                    break
                dlt *= 0.5
            if not improved or (np.hypot(dlt[0], dlt[1]) < 1e-4 and abs(dlt[2]) < 2e-5):   # 0.1 mm, 0.001 deg
                break
        A = J.T @ J
        H = A + np.diag(lam) + 1e-9 * np.eye(3)
        return x, float(1.0 - r.mean()), H, A, float(r @ r)

    def _match(self, obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
        xc, _, _ = self.correlative(obs, pred)
        x, score, H, A, ssr = self.refine(obs, xc, pred)
        n = len(obs)
        sig2 = max(ssr / max(n - 3, 1), 1e-3)
        self.cov = sig2 * np.linalg.inv(H)
        ev = np.linalg.eigvalsh(A[:2, :2])
        self.score = score
        if score < self.p.min_score:
            self.state = "weak"
            self.cov = self.cov + self._odom_cov(between(self._pose, pred))
            return pred
        self.state = "degenerate" if ev[0] < self.p.degenerate_ratio * max(ev[1], 1e-12) else "tracking"
        return x

    # ------------------------------------------------------------------ map
    def _integrate(self, logodds: np.ndarray, pose, pts, hit, origin) -> None:
        wp = transform_points(pose, pts)
        ends = np.floor((wp - self.origin) / self.res).astype(np.int32)
        o = transform_points(pose, origin[None])[0]
        oc = np.clip(np.floor((o - self.origin) / self.res).astype(np.int32), 0, self.n - 1)
        native.integrate_scan(logodds, (int(oc[0]), int(oc[1])), ends, hit.astype(np.uint8),
                              self.p.l_free, self.p.l_occ, self.p.l_clamp)

    def _maybe_insert(self, pts, hit, origin, use) -> None:
        self._since_insert += 1
        if self._last_insert is not None:
            d = between(self._last_insert, self._pose)
            if (np.hypot(d[0], d[1]) < self.p.map_update_dist and abs(d[2]) < np.deg2rad(self.p.map_update_deg)
                    and self._since_insert < self.p.map_update_every):
                return
        if len(pts) == 0:
            return
        virt, _ = self.surface_points(pts, hit, self.res)
        self._integrate(self.logodds, self._pose, np.vstack([pts, virt]),
                        np.concatenate([hit, np.ones(len(virt), bool)]), origin)
        self._last_insert = self._pose.copy()
        self._since_insert = 0
        self.inserted += 1
        self._map_version += 1
        if self.p.loop_closure:
            self._backend_insert(pts, hit, origin, use)

    # ------------------------------------------------------------------ back end
    def _backend_insert(self, pts, hit, origin, use) -> None:
        """Store the inserted scan relative to its keyframe; make keyframes; try loop closures."""
        new_kf = not self.kf_pose
        if not new_kf:
            d = between(self.kf_pose[-1], self._pose)
            new_kf = np.hypot(d[0], d[1]) >= self.p.keyframe_dist or abs(d[2]) >= np.deg2rad(self.p.keyframe_deg)
        if new_kf:
            i = len(self.kf_pose)
            if i > 0:
                z = between(self.kf_pose[-1], self._pose)
                self.edges.append((i - 1, i, z, self._odom_edge_info(z), "odom"))
            self.kf_pose.append(self._pose.copy())
            self.kf_scan.append((pts[use].astype(np.float32), origin.copy()))
            self.kf_dist.append(self._walked)
        kf = len(self.kf_pose) - 1
        self._scans.append((kf, between(self.kf_pose[kf], self._pose), pts.astype(np.float32), hit.copy(),
                            origin.copy()))
        if new_kf and self.p.loop_closure and kf > 0 and kf % max(self.p.lc_every, 1) == 0:
            self._try_loop_closure(kf)

    def _odom_edge_info(self, z) -> np.ndarray:
        """Information of a sequential constraint (front-end relative motion): a few cm / tenths of a
        degree per keyframe step, growing with the distance."""
        dist = float(np.hypot(z[0], z[1]))
        st = 0.02 + 0.02 * dist
        sy = np.deg2rad(0.5) + np.deg2rad(1.0) * dist
        return np.diag([1 / st ** 2, 1 / st ** 2, 1 / sy ** 2])

    def _submap(self, centre: int):
        """Likelihood field of the keyframes around `centre` rendered at their current graph poses
        (window of the global grid's geometry, so cells line up with the map frame)."""
        lo, hi = max(0, centre - self.p.lc_submap), min(len(self.kf_pose) - 1, centre + self.p.lc_submap)
        lo_ = np.full((self.n, self.n), 0.0, np.float32)
        for j in range(lo, hi + 1):
            pts, origin = self.kf_scan[j]
            if len(pts):
                self._integrate(lo_, self.kf_pose[j], pts.astype(float), np.ones(len(pts), bool), origin)
        F = likelihood_field(lo_, self.p.occ_thresh, self._lut, self.use_native, self.p.threads)
        return F, window_max(F, self.p.coarse, self.use_native, self.p.threads), F.astype(np.float32) / Q

    def _try_loop_closure(self, cur: int) -> None:
        here = self.kf_pose[cur]
        cands = [j for j in range(cur - self.p.lc_min_age + 1)
                 if np.hypot(*(self.kf_pose[j][:2] - here[:2])) < self.p.lc_radius
                 and self.kf_dist[cur] - self.kf_dist[j] >= self.p.lc_min_travel]
        if not cands:
            return
        j = min(cands, key=lambda c: np.hypot(*(self.kf_pose[c][:2] - here[:2])))
        obs = self.kf_scan[cur][0].astype(float)
        if len(obs) < self.p.min_points * 2:
            return
        F, Fm, Ff = self._submap(j)
        xc, raw, _ = self.correlative(obs, here, self.p.lc_win_xy, np.deg2rad(self.p.lc_win_yaw_deg), F, Fm,
                                      prior_scale=0.1)
        x, score, H, A, ssr = self.refine(obs, xc, here, Ff, prior=False)
        ev = np.linalg.eigvalsh(A[:2, :2])
        if score < self.p.lc_min_score or ev[0] < 0.2 * max(ev[1], 1e-12):
            return
        # the submap is drawn at the graph poses, so x is the current keyframe's pose in the graph frame
        # as seen from keyframe j's neighbourhood: constraint j -> cur
        z = between(self.kf_pose[j], x)
        info = np.diag([1 / 0.03 ** 2, 1 / 0.03 ** 2, 1 / np.deg2rad(1.0) ** 2])
        self.edges.append((j, cur, z, info, "loop"))
        self.loop_closures += 1
        self._optimise()

    def _optimise(self, iters: int = 20) -> None:
        """Levenberg-Marquardt over the keyframe poses (keyframe 0 fixed), then re-anchor everything."""
        X = np.array(self.kf_pose)
        old = X.copy()
        m = len(X)
        lam = 1e-4

        def total(X):
            c = 0.0
            for i, j, z, info, _ in self.edges:
                e = _edge_error_jacobians(X[i], X[j], z)[0]
                c += float(e @ info @ e)
            return c

        cost = total(X)
        for _ in range(iters):
            H = np.zeros((3 * m, 3 * m))
            g = np.zeros(3 * m)
            for i, j, z, info, _ in self.edges:
                e, Ji, Jj = _edge_error_jacobians(X[i], X[j], z)
                si, sj = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
                H[si, si] += Ji.T @ info @ Ji
                H[si, sj] += Ji.T @ info @ Jj
                H[sj, si] += Jj.T @ info @ Ji
                H[sj, sj] += Jj.T @ info @ Jj
                g[si] += Ji.T @ info @ e
                g[sj] += Jj.T @ info @ e
            H[:3, :3] += 1e12 * np.eye(3)   # gauge: first keyframe fixed
            improved = False
            for _ in range(8):
                D = H + lam * np.diag(np.diag(H) + 1e-9)
                try:
                    dx = -np.linalg.solve(D, g)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                Xn = X + dx.reshape(m, 3)
                Xn[:, 2] = wrap(Xn[:, 2])
                cn = total(Xn)
                if cn < cost:
                    X, cost, lam, improved = Xn, cn, max(lam / 10, 1e-9), True
                    break
                lam *= 10
            if not improved or np.abs(dx).max() < 1e-7:
                break
        self.kf_pose = [x.copy() for x in X]
        # the current pose follows its keyframe; the map is rebuilt from every stored scan
        last = m - 1
        self._pose = compose(X[last], between(old[last], self._pose))
        if self._last_insert is not None:
            self._last_insert = compose(X[last], between(old[last], self._last_insert))
        self.logodds[:] = 0
        for kf, rel, pts, hit, origin in self._scans:
            self._integrate(self.logodds, compose(X[kf], rel), pts.astype(float), hit, origin)
        # re-anchor the stored trajectory (for display / evaluation) to its keyframes
        self._reanchor_traj(old, X)
        self._map_version += 1

    def _reanchor_traj(self, old: np.ndarray, new: np.ndarray) -> None:
        """Move past trajectory samples with the keyframe that was current when they were estimated."""
        for t, kf in enumerate(self._traj_kf):
            if 0 <= kf < len(new):
                self.traj_est[t] = compose(new[kf], between(old[kf], self.traj_est[t]))


def _edge_error_jacobians(xi, xj, z):
    """e = z^-1 (+) (xi^-1 (+) xj) and its Jacobians w.r.t. xi and xj (analytic)."""
    ci, si = np.cos(xi[2]), np.sin(xi[2])
    cz, sz = np.cos(z[2]), np.sin(z[2])
    RiT = np.array([[ci, si], [-si, ci]])
    dRiT = np.array([[-si, ci], [-ci, -si]])
    RzT = np.array([[cz, sz], [-sz, cz]])
    dt = xj[:2] - xi[:2]
    e = np.empty(3)
    e[:2] = RzT @ (RiT @ dt - z[:2])
    e[2] = float(wrap(xj[2] - xi[2] - z[2]))
    Ji = np.zeros((3, 3))
    Jj = np.zeros((3, 3))
    Ji[:2, :2] = -RzT @ RiT
    Ji[:2, 2] = RzT @ (dRiT @ dt)
    Ji[2, 2] = -1.0
    Jj[:2, :2] = RzT @ RiT
    Jj[2, 2] = 1.0
    return e, Ji, Jj


# =====================================================================================================
# evaluation and a headless walking episode
# =====================================================================================================
def align_rigid(est_xy: np.ndarray, true_xy: np.ndarray) -> np.ndarray:
    """est points moved by the least-squares rotation + translation onto true (Umeyama 1991, no scale)."""
    a, b = np.asarray(est_xy, float), np.asarray(true_xy, float)
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    U, _, Vt = np.linalg.svd((b - mb).T @ (a - ma))
    R = U @ np.diag([1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return (a - ma) @ R.T + mb


def ate(est, true, align: bool = False) -> float:
    """Absolute trajectory error: RMSE of the xy positions (Sturm et al., IROS 2012). The map frame is
    anchored at the known start pose, so by default no alignment is applied (stricter)."""
    a, b = np.asarray(est, float)[:, :2], np.asarray(true, float)[:, :2]
    if align:
        a = align_rigid(a, b)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def map_metrics(logodds: np.ndarray, truth: np.ndarray, observed: np.ndarray, pool: int = 2) -> dict:
    """Occupied cells (log-odds > 0.5) vs ground truth.

    iou: IoU with the ground-truth occupied cells the lidar could see (truth & observed, where observed =
    cells the true-pose map touched; obstacle interiors are never seen), cell for cell.
    iou_coarse: the same on a grid `pool` times coarser (any occupied / seen sub-cell counts), i.e. on
    the 0.1 m planning grid for a 0.05 m map; a one-cell wall offset no longer counts as a miss.
    precision: fraction of occupied map cells within one cell of a ground-truth obstacle (ghost walls
    lower it)."""
    from scipy.ndimage import binary_dilation
    A = logodds > 0.5
    T = truth.astype(bool) & observed

    def iou(a, t):
        union = int((a | t).sum())
        return float((a & t).sum() / union) if union else 0.0

    def coarse(x):
        n = (x.shape[0] // pool) * pool
        return x[:n, :n].reshape(n // pool, pool, n // pool, pool).any(axis=(1, 3))

    near = binary_dilation(truth.astype(bool), np.ones((3, 3), bool))
    Tc = coarse(truth.astype(bool)) & coarse(observed)
    return {"iou": iou(A, T), "iou_coarse": iou(coarse(A), Tc),
            "precision": float((A & near).sum() / max(int(A.sum()), 1)), "occupied": int(A.sum())}


def run_slam_episode(model_xml: str, route, seconds: float = 120.0, seed: int = 0, *, res: float = 0.05,
                     size: float = 24.0, odom_noise: OdomNoise | None = None, params: SlamParams | None = None,
                     lidar_cfg=None, use_native: bool = True, rate_hz: float = 10.0, settle_s: float = 0.5,
                     stop_when_done: bool = True, keep_log: bool = False) -> dict:
    """Walk the robot headless along `route` ((n, 2) world waypoints, e.g. a GridMap.plan path) with the
    model-based WalkController, scanning at rate_hz; run SLAM online from noisy odometry + lidar.

    Returns {"t", "true", "odom", "slam", "slam_online" ((k, 3) poses at each scan), "ate_slam",
    "ate_slam_online", "ate_odom", "ate_slam_aligned", "ate_odom_aligned", "yaw_rmse_slam_deg",
    "yaw_rmse_odom_deg", "final_err_slam", "final_err_odom", "max_err_slam", "max_err_odom", "distance_m",
    "scans", "match_ms_mean", "match_ms_p95", "map_slam" / "map_odom" / "map_ref" (map_metrics of the SLAM
    map, the odometry-only map and the true-pose map), "loop_closures", "max_correction" (largest per-scan
    shift away from the odometry prediction), "states" (scan count per Slam2D.state), "fell", "reached",
    "slam_obj" (the Slam2D, for its map), "log" (with keep_log: everything replay_slam needs)}.
    The controller walks on the simulator's state (as in the GUI); SLAM only sees odometry and scans.
    """
    import mujoco

    from ..control.walker import WalkController
    from ..sim.sensors import Lidar, LidarConfig
    from .grid import GridMap

    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ctrl = WalkController(model)
    ctrl.reset(model, data)
    lidar = Lidar(model, lidar_cfg or LidarConfig())
    for _ in range(int(round(settle_s / ctrl.control_dt))):
        ctrl.step(model, data)
        mujoco.mj_step(model, data, ctrl.substeps)
    start = root_pose(data, lidar.root)
    slam = Slam2D(size, res, odom_noise, seed, params, initial_pose=start, use_native=use_native)
    slam.odometry.reset(start)
    truth = GridMap(size, res)
    truth.rasterize_truth(model, data)
    ref = np.zeros_like(slam.logodds)
    odo_map = np.zeros_like(slam.logodds)
    route = np.asarray(route, float).reshape(-1, 2)
    ctrl.set_path(route)
    every = max(1, int(round(1.0 / (rate_hz * ctrl.control_dt))))
    T, TR, OD, MS, scans_log, states = [], [], [], [], [], {}
    tick = 0
    t_end = data.time + seconds
    while data.time < t_end:
        if tick % every == 0:
            sc = scan_robot_frame(model, data, lidar)
            tp = root_pose(data, lidar.root)
            op = slam.odometry.update(tp)
            before = slam.inserted
            slam.update(op, sc.points, sc.hit, sc.origin, match=sc.match)
            if slam.state in ("tracking", "degenerate", "weak"):
                MS.append(slam.match_ms)
            states[slam.state] = states.get(slam.state, 0) + 1
            if slam.inserted > before:
                ip, ih = slam.insertion_scan(sc.points, sc.hit, sc.origin)
                slam._integrate(ref, tp, ip, ih, sc.origin)
                slam._integrate(odo_map, slam.traj_odom[-1], ip, ih, sc.origin)
            T.append(data.time)
            TR.append(tp)
            OD.append(slam.traj_odom[-1].copy())
            if keep_log:
                scans_log.append((tp, op, sc))
        if ctrl.mode == "fallen" or (stop_when_done and ctrl.mode == "done" and tick % every == every - 1):
            break
        ctrl.step(model, data)
        mujoco.mj_step(model, data, ctrl.substeps)
        tick += 1
    out = _episode_result(slam, np.array(TR), np.array(OD), truth.truth, ref, odo_map, MS)
    out.update({"t": np.array(T), "states": states, "fell": ctrl.mode == "fallen",
                "reached": ctrl.mode == "done", "walk_status": ctrl.status()["message"]})
    if keep_log:
        out["log"] = {"model_xml": model_xml, "size": size, "start": start, "scans": scans_log}
    return out


def _episode_result(slam: Slam2D, TR: np.ndarray, OD: np.ndarray, truth: np.ndarray, ref: np.ndarray,
                    odo_map: np.ndarray, MS: list) -> dict:
    est, onl = np.array(slam.traj_est), np.array(slam.traj_online)
    e_s = np.hypot(*(est[:, :2] - TR[:, :2]).T)
    e_o = np.hypot(*(OD[:, :2] - TR[:, :2]).T)
    observed = ref != 0
    ms = np.array(MS) if MS else np.zeros(1)
    return {"true": TR, "odom": OD, "slam": est, "slam_online": onl,
            "ate_slam": ate(est, TR), "ate_slam_online": ate(onl, TR), "ate_odom": ate(OD, TR),
            "ate_slam_aligned": ate(est, TR, align=True), "ate_odom_aligned": ate(OD, TR, align=True),
            "yaw_rmse_slam_deg": float(np.rad2deg(np.sqrt(np.mean(wrap(est[:, 2] - TR[:, 2]) ** 2)))),
            "yaw_rmse_odom_deg": float(np.rad2deg(np.sqrt(np.mean(wrap(OD[:, 2] - TR[:, 2]) ** 2)))),
            "final_err_slam": float(e_s[-1]), "final_err_odom": float(e_o[-1]),
            "max_err_slam": float(e_s.max()), "max_err_odom": float(e_o.max()),
            "distance_m": float(np.hypot(*np.diff(TR[:, :2], axis=0).T).sum()) if len(TR) > 1 else 0.0,
            "scans": slam.scans, "match_ms_mean": float(ms.mean()), "match_ms_p95": float(np.percentile(ms, 95)),
            "map_slam": map_metrics(slam.logodds, truth, observed), "map_odom": map_metrics(odo_map, truth, observed),
            "map_ref": map_metrics(ref, truth, observed), "loop_closures": slam.loop_closures,
            "max_correction": slam.max_correction, "slam_obj": slam}


def replay_slam(log: dict, slam: Slam2D) -> dict:
    """Run a Slam2D (any resolution / backend / parameters) over the scans recorded by
    run_slam_episode(keep_log=True); returns the same metrics (no walking)."""
    import mujoco

    from .grid import GridMap
    model = mujoco.MjModel.from_xml_string(log["model_xml"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    truth = GridMap(slam.size, slam.res)
    truth.rasterize_truth(model, data)
    slam.reset(log["start"])
    ref = np.zeros_like(slam.logodds)
    odo_map = np.zeros_like(slam.logodds)
    TR, OD, MS = [], [], []
    for tp, op, sc in log["scans"]:
        before = slam.inserted
        slam.update(op, sc.points, sc.hit, sc.origin, match=sc.match)
        if slam.state in ("tracking", "degenerate", "weak"):
            MS.append(slam.match_ms)
        if slam.inserted > before:
            ip, ih = slam.insertion_scan(sc.points, sc.hit, sc.origin)
            slam._integrate(ref, tp, ip, ih, sc.origin)
            slam._integrate(odo_map, slam.traj_odom[-1], ip, ih, sc.origin)
        TR.append(tp)
        OD.append(slam.traj_odom[-1].copy())
    return _episode_result(slam, np.array(TR), np.array(OD), truth.truth, ref, odo_map, MS)
