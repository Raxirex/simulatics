"""The lidar's memory: every point it has hit so far, kept as a voxel-downsampled 3D point cloud.

No Qt and no MuJoCo here, only numpy. The simulation thread calls :meth:`PointCloudMap.add` with each scan, the
interface thread reads :attr:`PointCloudMap.points` (or :meth:`PointCloudMap.snapshot`) to draw it.
"""
from __future__ import annotations

import os
import threading

import numpy as np

_BITS = 21                       # bits per axis in the packed voxel key
_OFF = 1 << (_BITS - 1)          # voxel indices are shifted by this to become non-negative
_CHUNK = 16384                   # smallest buffer the cloud starts with; it then doubles
_RECENT = 32768                  # new voxel keys wait in a small sorted array and are merged in this many at a time


def _empty3() -> np.ndarray:
    return np.zeros((0, 3), np.float32)


class PointCloudMap:
    """Everything the lidar has hit so far, as a voxel-downsampled 3D point cloud.

    Space is cut into cubes of ``voxel`` metres and each cube keeps at most one point: the FIRST hit that fell
    into it (the real hit position, not the cube centre, and not a running mean). First-seen makes the stored
    arrays append-only, so the arrays handed out by :attr:`points` / :attr:`stamps` / :attr:`trajectory` are
    read-only snapshots that are never modified afterwards, and no copy is needed to take one.

    Safe to use from two threads: every mutation happens under one lock, and readers only ever get snapshots.

    ``seq`` increases whenever anything a view would draw changes: new voxels, a new trajectory point, a
    non-empty :meth:`clear`, or :meth:`assign`. Adding a scan that brings nothing new leaves it untouched.
    """

    traj_step = 0.05             # sensor positions closer than this to the last kept one are skipped (metres)

    def __init__(self, voxel: float = 0.05, max_points: int = 400_000):
        if not voxel > 0:
            raise ValueError("voxel must be positive")
        self.voxel = float(voxel)
        self.max_points = int(max_points)
        self.seq = 0
        self._lock = threading.RLock()
        self._reset()

    def _reset(self) -> None:
        """Fresh, empty buffers (never reuse the old ones: snapshots handed out earlier still point at them)."""
        cap = max(1, min(_CHUNK, self.max_points))
        self._pts = np.zeros((cap, 3), np.float32)
        self._stamps = np.zeros(cap, np.float32)
        self._n = 0
        # packed keys of the occupied voxels, in two sorted arrays: inserting into the big one costs a pass
        # over all of it, so new keys go into the small one, which is folded into the big one now and then
        self._keys = np.zeros(0, np.int64)
        self._recent = np.zeros(0, np.int64)
        self._traj = np.zeros((256, 3), np.float32)
        self._m = 0
        self._lo = None
        self._hi = None

    # ------------------------------------------------------------------ voxel keys
    def _pack(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Packed int64 voxel key of each point, and which points have one (finite and within range)."""
        with np.errstate(invalid="ignore", over="ignore"):
            q = np.floor(pts.astype(np.float64) / self.voxel)
            ok = np.isfinite(q).all(axis=1) & (np.abs(q) < _OFF - 1).all(axis=1)
        qi = q[ok].astype(np.int64) + _OFF
        return (qi[:, 0] << (2 * _BITS)) | (qi[:, 1] << _BITS) | qi[:, 2], ok

    # ------------------------------------------------------------------ filling
    def add(self, points: np.ndarray, origin: np.ndarray | None = None, time: float = 0.0) -> int:
        """Add one scan (``points`` (n, 3), world frame). Returns how many NEW voxels it filled.

        ``origin`` (the sensor position), when given, is appended to the trajectory unless it is within
        ``traj_step`` of the last one. Non-finite points and points further than about 50 km (at 5 cm voxels)
        from the world origin are ignored. Once ``max_points`` is reached no more voxels are added.
        """
        pts = np.asarray(points, np.float32).reshape(-1, 3) if points is not None else _empty3()
        added = 0
        with self._lock:
            changed = self._add_origin(origin)
            room = self.max_points - self._n
            if len(pts) and room > 0:
                keys, ok = self._pack(pts)
                if not ok.all():
                    pts = pts[ok]
                ukeys, first = np.unique(keys, return_index=True)         # one candidate per voxel: its first hit
                fresh = ~(self._has(self._keys, ukeys) | self._has(self._recent, ukeys))
                added = int(np.count_nonzero(fresh))
                if added:
                    ukeys, first = ukeys[fresh], first[fresh]
                    if added > room:                                        # keep the earliest hits of the scan
                        keep = np.sort(np.argpartition(first, room - 1)[:room])
                        ukeys, first = ukeys[keep], first[keep]
                        added = room
                    new = pts[np.sort(first)]                               # stored in scan order
                    self._grow(self._n + added)
                    self._pts[self._n:self._n + added] = new
                    self._stamps[self._n:self._n + added] = time
                    self._n += added
                    self._recent = np.insert(self._recent, np.searchsorted(self._recent, ukeys), ukeys)
                    if len(self._recent) >= _RECENT:                        # two sorted runs: a cheap stable sort
                        self._keys = np.sort(np.concatenate([self._keys, self._recent]), kind="stable")
                        self._recent = np.zeros(0, np.int64)
                    lo, hi = new.min(axis=0), new.max(axis=0)
                    self._lo = lo if self._lo is None else np.minimum(self._lo, lo)
                    self._hi = hi if self._hi is None else np.maximum(self._hi, hi)
                    changed = True
            if changed:
                self.seq += 1
        return added

    @staticmethod
    def _has(sorted_keys: np.ndarray, keys: np.ndarray) -> np.ndarray:
        if not len(sorted_keys):
            return np.zeros(len(keys), bool)
        pos = np.searchsorted(sorted_keys, keys)
        return sorted_keys[np.minimum(pos, len(sorted_keys) - 1)] == keys

    def _add_origin(self, origin) -> bool:
        if origin is None:
            return False
        o = np.asarray(origin, np.float32).reshape(-1)[:3]
        if o.shape != (3,) or not np.isfinite(o).all():
            return False
        if self._m and float(np.linalg.norm(o - self._traj[self._m - 1])) < self.traj_step:
            return False
        if self._m == len(self._traj):
            bigger = np.zeros((2 * len(self._traj), 3), np.float32)
            bigger[:self._m] = self._traj[:self._m]
            self._traj = bigger
        self._traj[self._m] = o
        self._m += 1
        return True

    def _grow(self, need: int) -> None:
        cap = len(self._pts)
        if need <= cap:
            return
        cap = min(max(2 * cap, need), max(self.max_points, need))
        pts, stamps = np.zeros((cap, 3), np.float32), np.zeros(cap, np.float32)
        pts[:self._n] = self._pts[:self._n]
        stamps[:self._n] = self._stamps[:self._n]
        self._pts, self._stamps = pts, stamps        # the old buffers live on untouched under any snapshot

    def clear(self) -> None:
        """Forget everything (points and trajectory)."""
        with self._lock:
            had = self._n > 0 or self._m > 0
            self._reset()
            if had:
                self.seq += 1

    def assign(self, other: "PointCloudMap") -> None:
        """Become a copy of ``other`` in place (points, stamps, trajectory, voxel, max_points).

        For loading a file into a map that another thread keeps adding to: the object stays the same.
        """
        d = other.to_dict()
        with self._lock:
            self._set(d["points"], d["stamps"], d["trajectory"], d["voxel"], d["max_points"])
            self.seq += 1

    def _set(self, points, stamps, trajectory, voxel, max_points) -> None:
        pts = np.array(points, np.float32).reshape(-1, 3)                  # private copies
        n = len(pts)
        st = np.zeros(n, np.float32) if stamps is None else np.array(stamps, np.float32).reshape(-1)
        if len(st) != n:
            raise ValueError("stamps and points differ in length")
        tr = _empty3() if trajectory is None else np.array(trajectory, np.float32).reshape(-1, 3)
        self.voxel = float(voxel)
        self.max_points = max(int(max_points), n)
        self._pts, self._stamps, self._n = pts if n else np.zeros((1, 3), np.float32), st if n else np.zeros(1, np.float32), n
        self._traj, self._m = (tr if len(tr) else np.zeros((256, 3), np.float32)), len(tr)
        keys, ok = self._pack(pts)
        self._keys = np.unique(keys)
        self._recent = np.zeros(0, np.int64)
        good = pts[np.isfinite(pts).all(axis=1)]
        self._lo = good.min(axis=0) if len(good) else None
        self._hi = good.max(axis=0) if len(good) else None

    # ------------------------------------------------------------------ reading
    @staticmethod
    def _frozen(a: np.ndarray) -> np.ndarray:
        a.flags.writeable = False
        return a

    @property
    def points(self) -> np.ndarray:
        """(n, 3) float32 snapshot of the cloud: read-only, free to take, never modified afterwards."""
        with self._lock:
            return self._frozen(self._pts[:self._n])

    @property
    def stamps(self) -> np.ndarray:
        """(n,) float32: the simulation time each voxel was first seen."""
        with self._lock:
            return self._frozen(self._stamps[:self._n])

    @property
    def trajectory(self) -> np.ndarray:
        """(m, 3) float32 sensor positions over time (thinned to one every ``traj_step``)."""
        with self._lock:
            return self._frozen(self._traj[:self._m])

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """(points, stamps, trajectory, seq) taken at one instant, so their lengths agree."""
        with self._lock:
            return (self._frozen(self._pts[:self._n]), self._frozen(self._stamps[:self._n]),
                    self._frozen(self._traj[:self._m]), self.seq)

    @property
    def full(self) -> bool:
        """True once ``max_points`` voxels are stored; further scans add nothing."""
        return self._n >= self.max_points

    def __len__(self) -> int:
        return self._n

    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(min, max) corners of the cloud, or None when it is empty."""
        with self._lock:
            if self._n == 0 or self._lo is None:
                return None
            return self._lo.copy(), self._hi.copy()

    # ------------------------------------------------------------------ files
    def to_dict(self) -> dict:
        """The whole state as plain arrays and numbers (what the .npz file holds)."""
        with self._lock:
            return {"points": self._pts[:self._n].copy(), "stamps": self._stamps[:self._n].copy(),
                    "trajectory": self._traj[:self._m].copy(), "voxel": self.voxel, "max_points": self.max_points}

    @classmethod
    def from_dict(cls, d: dict) -> "PointCloudMap":
        return cls.from_arrays(d["points"], d.get("stamps"), d.get("trajectory"),
                               voxel=float(d.get("voxel", 0.05)), max_points=int(d.get("max_points", 400_000)))

    @classmethod
    def from_arrays(cls, points, stamps=None, trajectory=None, voxel: float = 0.05,
                    max_points: int = 400_000) -> "PointCloudMap":
        """A map holding exactly these arrays (no re-voxelising; ``max_points`` is raised to fit them)."""
        m = cls(voxel, max_points)
        m._set(points, stamps, trajectory, voxel, max_points)
        m.seq = 1 if len(m) or m._m else 0
        return m

    def save(self, path) -> None:
        """Write the cloud to ``path``: ".npz" (points, stamps, trajectory, voxel, max_points; exact) or ".ply"
        (ASCII PLY with x y z, for common point-cloud viewers; points only)."""
        path = os.fspath(path)
        ext = os.path.splitext(path)[1].lower()
        d = self.to_dict()
        if ext == ".npz":
            with open(path, "wb") as f:
                np.savez_compressed(f, points=d["points"], stamps=d["stamps"], trajectory=d["trajectory"],
                                    voxel=np.float64(d["voxel"]), max_points=np.int64(d["max_points"]))
        elif ext == ".ply":
            pts = d["points"]
            with open(path, "w", encoding="ascii", newline="\n") as f:
                f.write("ply\nformat ascii 1.0\ncomment Simulatics lidar point cloud\n"
                        f"comment voxel {d['voxel']!r}\n"
                        f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n"
                        "end_header\n")
                if len(pts):
                    np.savetxt(f, pts, fmt="%.6f")
        else:
            raise ValueError(f"cannot save a point cloud as '{ext or path}': use .ply or .npz")

    @classmethod
    def load(cls, path) -> "PointCloudMap":
        """Read a cloud written by :meth:`save` (.npz, exact) or any ASCII .ply with x y z vertices (points only)."""
        path = os.fspath(path)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npz":
            with np.load(path) as z:
                return cls.from_arrays(z["points"], z["stamps"] if "stamps" in z else None,
                                       z["trajectory"] if "trajectory" in z else None,
                                       voxel=float(z["voxel"]) if "voxel" in z else 0.05,
                                       max_points=int(z["max_points"]) if "max_points" in z else 400_000)
        if ext == ".ply":
            pts, voxel = _read_ply(path)
            return cls.from_arrays(pts, voxel=voxel)
        raise ValueError(f"cannot load a point cloud from '{ext or path}': use .ply or .npz")


def _read_ply(path: str) -> tuple[np.ndarray, float]:
    """Vertices (n, 3) and the voxel size noted in the header (0.05 when absent) of an ASCII PLY file."""
    voxel, count, props, element, skip = 0.05, None, [], None, 0
    with open(path, "r", encoding="ascii", errors="replace") as f:
        if f.readline().strip() != "ply":
            raise ValueError(f"{path} is not a PLY file")
        for line in f:
            w = line.split()
            if not w:
                continue
            if w[0] == "format" and w[1] != "ascii":
                raise ValueError(f"{path} is a binary PLY ({w[1]}); only ASCII PLY can be loaded")
            if w[0] == "comment" and len(w) == 3 and w[1] == "voxel":
                try:
                    voxel = float(w[2])
                except ValueError:
                    pass
            elif w[0] == "element":
                element = w[1]
                if element == "vertex":
                    count = int(w[2])
                elif count is None:
                    skip += int(w[2])                 # rows of elements listed before the vertices
            elif w[0] == "property" and element == "vertex":
                props.append(w[-1])
            elif w[0] == "end_header":
                break
        else:
            raise ValueError(f"{path}: PLY header has no end_header")
        if count is None or not all(a in props for a in "xyz"):
            raise ValueError(f"{path}: PLY file has no x y z vertices")
        if count == 0:
            return _empty3(), voxel
        cols = [props.index(a) for a in "xyz"]
        pts = np.loadtxt(f, dtype=np.float64, usecols=cols, skiprows=skip, max_rows=count, ndmin=2)
    if len(pts) != count:
        raise ValueError(f"{path}: expected {count} vertices, found {len(pts)}")
    return pts.astype(np.float32), (voxel if voxel > 0 else 0.05)
