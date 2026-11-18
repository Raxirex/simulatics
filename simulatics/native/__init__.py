"""ctypes bindings for the C++ kernels, with pure-Python fallbacks.

Every `src/<name>.cpp` compiles to its own library `bin/<name>.dll`. This
module binds `simcore` (planning, mapping, batched MLP); other modules load
their own library with `load_library("<name>")`.

Build with `python -m simulatics build-native` (needs CMake + a C++
compiler). Everything works without the libraries, only slower.
"""
from __future__ import annotations

import ctypes
import heapq
import subprocess
import sys
from pathlib import Path

import numpy as np

_DIR = Path(__file__).parent
_BIN = _DIR / "bin"
_lib = None
_load_error = ""

_u8p = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
_i32p = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
_f32p = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
_c_int, _c_float = ctypes.c_int, ctypes.c_float


def lib_path(name: str) -> Path | None:
    """Path of the compiled library `name` (e.g. "simcore", "topopt"), or None if not built."""
    for ext in (".dll", ".so", ".dylib"):
        if (_BIN / f"{name}{ext}").exists():
            return _BIN / f"{name}{ext}"
    return None


def load_library(name: str) -> ctypes.CDLL | None:
    """Load bin/<name>.dll; None when it is not built or cannot load. Callers set argtypes themselves."""
    path = lib_path(name)
    if path is None:
        return None
    try:
        return ctypes.CDLL(str(path))
    except OSError:
        return None


def _lib_path() -> Path | None:
    return lib_path("simcore")


def _load() -> None:
    global _lib, _load_error
    path = _lib_path()
    if path is None:
        _lib, _load_error = None, "not built"
        return
    try:
        lib = ctypes.CDLL(str(path))
        lib.sc_version.restype = _c_int
        lib.sc_mlp_forward_batch.argtypes = [_f32p, _c_int, _c_int, _i32p, _c_int, _f32p, _f32p]
        lib.sc_mlp_forward_batch.restype = None
        lib.sc_inflate.argtypes = [_u8p, _c_int, _c_int, _c_float, _u8p]
        lib.sc_inflate.restype = None
        lib.sc_astar.argtypes = [_u8p] + [_c_int] * 6 + [_i32p, _c_int]
        lib.sc_astar.restype = _c_int
        lib.sc_smooth_path.argtypes = [_u8p, _c_int, _c_int, _i32p, _c_int, _c_int]
        lib.sc_smooth_path.restype = _c_int
        lib.sc_integrate_scan.argtypes = [_f32p, _c_int, _c_int, _c_int, _c_int, _i32p, _i32p, _u8p, _c_int,
                                          _c_float, _c_float, _c_float]
        lib.sc_integrate_scan.restype = None
        _lib, _load_error = lib, ""
    except OSError as e:  # wrong architecture, missing runtime...
        _lib, _load_error = None, str(e)


_load()


def available() -> bool:
    return _lib is not None


def status() -> str:
    return f"native simcore v{_lib.sc_version()}" if _lib else f"python fallback ({_load_error})"


def build(verbose: bool = True, target: str | None = None) -> bool:
    """Configure + compile the native kernels with CMake. Returns True on success.

    target: build only bin/<target>.dll (e.g. "topopt"). Use it while another
    process holds a different kernel's DLL loaded; Windows cannot overwrite it.
    """
    global _lib
    # This process loaded simcore at import; Windows refuses to overwrite a loaded DLL.
    if _lib is not None and target in (None, "simcore") and sys.platform == "win32":
        import _ctypes
        _ctypes.FreeLibrary(_lib._handle)
        _lib = None
    build_dir = _DIR / "build"
    cmds = [
        ["cmake", "-S", str(_DIR), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"],
        ["cmake", "--build", str(build_dir), "--config", "Release"] + (["--target", target] if target else []),
    ]
    for cmd in cmds:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if verbose:
            print(r.stdout[-3000:])
        if r.returncode != 0:
            print((r.stdout + r.stderr)[-3000:], file=sys.stderr)
            _load()
            return False
    _load()
    return available()


# --------------------------------------------------------------------------
# kernels (native when available, otherwise numpy / pure Python)
# --------------------------------------------------------------------------
def mlp_forward_batch(params: np.ndarray, sizes: list[int], obs: np.ndarray) -> np.ndarray:
    out = np.empty((obs.shape[0], sizes[-1]), np.float32)
    _lib.sc_mlp_forward_batch(params, obs.shape[0], params.shape[1], np.asarray(sizes, np.int32), len(sizes),
                              obs, out)
    return out


def inflate(grid: np.ndarray, radius_cells: float, force_python: bool = False) -> np.ndarray:
    """Occupancy (h, w) uint8 -> cost grid: 255 blocked, 0..200 soft cost near obstacles."""
    grid = np.ascontiguousarray(grid, np.uint8)
    h, w = grid.shape
    if _lib and not force_python:
        out = np.empty_like(grid)
        _lib.sc_inflate(grid, w, h, float(radius_cells), out)
        return out
    # brute-force distance to the nearest occupied cell (fine for small grids)
    ys, xs = np.nonzero(grid)
    if len(xs) == 0:
        return np.zeros_like(grid)
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.full((h, w), 1e9)
    for i in range(0, len(xs), 256):
        d = np.hypot(xx[..., None] - xs[i:i + 256], yy[..., None] - ys[i:i + 256]).min(axis=-1)
        dist = np.minimum(dist, d)
    out = np.zeros((h, w), np.uint8)
    soft = (dist > radius_cells) & (dist < 2 * radius_cells)
    out[soft] = (200.0 * (2 * radius_cells - dist[soft]) / radius_cells).astype(np.uint8)
    out[dist <= radius_cells] = 255
    return out


def astar(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int], force_python: bool = False) -> np.ndarray:
    """A* on a cost grid (h, w). start/goal are (x, y) cells. Returns (n, 2) int cells, empty if no path."""
    cost = np.ascontiguousarray(cost, np.uint8)
    h, w = cost.shape
    if _lib and not force_python:
        buf = np.empty(2 * w * h, np.int32)
        n = _lib.sc_astar(cost, w, h, start[0], start[1], goal[0], goal[1], buf, w * h)
        return buf[:2 * max(n, 0)].reshape(-1, 2).copy()
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h) or cost[gy, gx] == 255:
        return np.zeros((0, 2), np.int32)
    r2 = 2 ** 0.5

    def heur(x, y):
        dx, dy = abs(x - gx), abs(y - gy)
        return dx + dy + (r2 - 2) * min(dx, dy)

    g = {(sx, sy): 0.0}
    parent: dict = {}
    closed = set()
    heap = [(heur(sx, sy), sx, sy)]
    nbrs = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, r2), (1, -1, r2), (-1, 1, r2), (-1, -1, r2)]
    while heap:
        _, cx, cy = heapq.heappop(heap)
        if (cx, cy) in closed:
            continue
        closed.add((cx, cy))
        if (cx, cy) == (gx, gy):
            path = [(gx, gy)]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            return np.array(path[::-1], np.int32)
        for dx, dy, base in nbrs:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h) or (nx, ny) in closed or cost[ny, nx] == 255:
                continue
            if dx and dy and (cost[cy, nx] == 255 or cost[ny, cx] == 255):
                continue
            ng = g[(cx, cy)] + base * (1.0 + cost[ny, nx] / 50.0)
            if ng < g.get((nx, ny), 1e30):
                g[(nx, ny)] = ng
                parent[(nx, ny)] = (cx, cy)
                heapq.heappush(heap, (ng + heur(nx, ny), nx, ny))
    return np.zeros((0, 2), np.int32)


def _bresenham(x0, y0, x1, y1):
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            return
        e2 = 2 * err
        if e2 >= dy:
            err += dy; x0 += sx
        if e2 <= dx:
            err += dx; y0 += sy


def smooth_path(cost: np.ndarray, path: np.ndarray, limit: int = 0, force_python: bool = False) -> np.ndarray:
    """Shortcut a cell path wherever the straight line stays on cells with cost <= limit."""
    if len(path) <= 2:
        return path
    cost = np.ascontiguousarray(cost, np.uint8)
    h, w = cost.shape
    if _lib and not force_python:
        buf = np.ascontiguousarray(path, np.int32).reshape(-1).copy()
        n = _lib.sc_smooth_path(cost, w, h, buf, len(path), int(limit))
        return buf[:2 * n].reshape(-1, 2).copy()
    keep, anchor, n = [0], 0, len(path)
    while anchor < n - 1:
        best = anchor + 1
        for j in range(n - 1, anchor + 1, -1):
            if all(cost[y, x] <= limit for x, y in _bresenham(*path[anchor], *path[j])):
                best = j
                break
        keep.append(best)
        anchor = best
    return path[keep]


def integrate_scan(logodds: np.ndarray, origin: tuple[int, int], ends: np.ndarray, hit: np.ndarray,
                   l_free: float = -0.4, l_occ: float = 0.9, l_clamp: float = 4.0,
                   force_python: bool = False) -> None:
    """Update a log-odds occupancy grid (h, w) float32 in place from one lidar scan."""
    h, w = logodds.shape
    ends = np.ascontiguousarray(ends, np.int32)
    hit = np.ascontiguousarray(hit, np.uint8)
    if _lib and not force_python:
        ex, ey = np.ascontiguousarray(ends[:, 0]), np.ascontiguousarray(ends[:, 1])
        _lib.sc_integrate_scan(logodds, w, h, int(origin[0]), int(origin[1]), ex, ey, hit, len(hit),
                               l_free, l_occ, l_clamp)
        return
    for (ex, ey), hi in zip(ends, hit):
        for x, y in _bresenham(int(origin[0]), int(origin[1]), int(ex), int(ey)):
            if not (0 <= x < w and 0 <= y < h):
                break
            if x == ex and y == ey:
                if hi:
                    logodds[y, x] = min(l_clamp, max(-l_clamp, logodds[y, x] + l_occ))
                break
            logodds[y, x] = min(l_clamp, max(-l_clamp, logodds[y, x] + l_free))
