"""3D SIMP topology optimisation on a voxel grid (minimum compliance, volume constraint).

Method (density-based topology optimisation, "SIMP"):

* Bendsoe (1989) material-distribution formulation; Sigmund's 99-line code (SMO 21, 2001); the
  88-line code of Andreassen, Clausen, Schevenels, Lazarov and Sigmund (DTU, SMO 43, 2011) for the
  density filter and the optimality-criteria (OC) update; the 3D codes top3d (Liu and Tovar, SMO 50,
  2014) and top3D125 (Ferrari and Sigmund, DTU, SMO 62, 2020) for the hex8 discretisation.
* Design variable x_e in [0, 1] per voxel. Density filter (Bruns and Tortorelli 2001, Bourdin 2001):
  xPhys = (H x) / (H 1),  H_ei = max(0, rmin - |c_e - c_i|).
* Modified SIMP stiffness: E_e = Emin + xPhys_e^p (E0 - Emin), p = 3, Emin = 1e-9 E0.
* Linear elasticity K(xPhys) U_k = F_k for every load case k; objective c = sum_k w_k F_k^T U_k
  (sum of compliances), sensitivity dc/dxPhys_e = -sum_k w_k p xPhys_e^(p-1) (E0 - Emin) u_ek^T k0 u_ek,
  chained back through the filter.
* OC update with move limit m: x_new = clamp(x sqrt(-dc / (lambda dv)), x - m, x + m), with the
  Lagrange multiplier found by bisection so that mean(xPhys(x_new)) = volfrac.

The state equation is solved matrix-free in C++ (`native/src/topopt.cpp`, OpenMP, 8-colour element
parallelism), warm-started from the previous design iteration, with conjugate gradients preconditioned
either by a geometric multigrid V-cycle (MGCG of Amir, Aage and Lazarov, SMO 49, 2014: Galerkin
coarse operators, damped-Jacobi smoothing, banded Cholesky on the coarsest grid; used when the grid
dimensions are even) or by Jacobi. Without the compiled library the Jacobi-PCG runs on an assembled
scipy.sparse matrix (`backend="numpy"`); native and numpy give the same iterates up to round-off
(tested on small grids).

Units are SI: voxel side in m, E0 in Pa, forces in N, displacements in m, compliance in J (N m).
Grid convention: element (i, j, k) with i along x; arrays have shape (nelx, nely, nelz), C order;
node (i, j, k) has id (i*(nely+1) + j)*(nelz+1) + k and DOFs 3*id + (0, 1, 2).
"""
from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np

from .. import native

# ---------------------------------------------------------------------------------------------
# element stiffness
# ---------------------------------------------------------------------------------------------
_CORNERS = np.array([[(a >> 2) & 1, (a >> 1) & 1, a & 1] for a in range(8)], dtype=float)


def elasticity_matrix(E: float, nu: float) -> np.ndarray:
    """6x6 isotropic constitutive matrix (Voigt order xx, yy, zz, xy, yz, zx; engineering shear strains)."""
    c = E / ((1 + nu) * (1 - 2 * nu))
    D = np.zeros((6, 6))
    D[:3, :3] = c * nu
    np.fill_diagonal(D[:3, :3], c * (1 - nu))
    D[3:, 3:] = np.eye(3) * c * (1 - 2 * nu) / 2
    return D


def hex8_stiffness(nu: float, h: float = 1.0, E: float = 1.0) -> np.ndarray:
    """24x24 stiffness of a cubic trilinear brick of side h (2x2x2 Gauss quadrature).

    Local node a sits at corner offset ((a>>2)&1, (a>>1)&1, a&1); DOF 3a + c. k0(h) = h k0(1).
    """
    D = elasticity_matrix(E, nu)
    s = _CORNERS * 2 - 1  # natural coordinates of the nodes, in {-1, 1}
    g = 1 / np.sqrt(3)
    KE = np.zeros((24, 24))
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                p = np.array([xi, eta, zeta])
                one = 1 + s * p  # (8, 3)
                dN = np.stack([s[:, 0] * one[:, 1] * one[:, 2],
                               s[:, 1] * one[:, 0] * one[:, 2],
                               s[:, 2] * one[:, 0] * one[:, 1]], axis=1) / 8 * (2 / h)
                B = np.zeros((6, 24))
                for a in range(8):
                    nx_, ny_, nz_ = dN[a]
                    B[0, 3 * a], B[1, 3 * a + 1], B[2, 3 * a + 2] = nx_, ny_, nz_
                    B[3, 3 * a], B[3, 3 * a + 1] = ny_, nx_
                    B[4, 3 * a + 1], B[4, 3 * a + 2] = nz_, ny_
                    B[5, 3 * a], B[5, 3 * a + 2] = nz_, nx_
                KE += B.T @ D @ B * (h ** 3 / 8)
    return (KE + KE.T) / 2


# ---------------------------------------------------------------------------------------------
# problem / config / result
# ---------------------------------------------------------------------------------------------
@dataclass
class TopOptProblem:
    """A voxel design domain with supports and load cases.

    fixed: bool (ndof,) - DOFs held at zero displacement.
    loads: (n_cases, ndof) nodal forces in N, one row per load case.
    weights: (n_cases,) weights of the compliances in the objective (default 1).
    passive_solid / passive_void: bool (nelx, nely, nelz) - elements kept at density 1 / 0.
    """
    nelx: int
    nely: int
    nelz: int
    voxel: float = 1.0
    E0: float = 1.0
    nu: float = 0.3
    fixed: np.ndarray | None = None
    loads: np.ndarray | None = None
    weights: np.ndarray | None = None
    passive_solid: np.ndarray | None = None
    passive_void: np.ndarray | None = None

    def __post_init__(self):
        n = self.ndof
        self.fixed = np.zeros(n, bool) if self.fixed is None else np.asarray(self.fixed, bool).reshape(n)
        self.loads = np.zeros((1, n)) if self.loads is None else np.atleast_2d(np.asarray(self.loads, float))
        if self.loads.shape[1] != n:
            raise ValueError(f"loads must have {n} columns (3 per node), got {self.loads.shape}")
        self.weights = (np.ones(len(self.loads)) if self.weights is None
                        else np.asarray(self.weights, float).reshape(len(self.loads)))
        shape = self.shape
        self.passive_solid = (np.zeros(shape, bool) if self.passive_solid is None
                              else np.asarray(self.passive_solid, bool).reshape(shape))
        self.passive_void = (np.zeros(shape, bool) if self.passive_void is None
                             else np.asarray(self.passive_void, bool).reshape(shape))
        if (self.passive_solid & self.passive_void).any():
            raise ValueError("an element cannot be both passive solid and passive void")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nelx, self.nely, self.nelz)

    @property
    def nel(self) -> int:
        return self.nelx * self.nely * self.nelz

    @property
    def node_shape(self) -> tuple[int, int, int]:
        return (self.nelx + 1, self.nely + 1, self.nelz + 1)

    @property
    def ndof(self) -> int:
        return 3 * int(np.prod(self.node_shape))

    def node_coords(self) -> np.ndarray:
        """(nnode, 3) node positions in metres (grid origin at the corner of element (0, 0, 0))."""
        i, j, k = np.meshgrid(*(np.arange(n) for n in self.node_shape), indexing="ij")
        return np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1) * self.voxel

    def nodes_where(self, axis: int, index: int) -> np.ndarray:
        """Node ids on the grid plane where the node index along `axis` equals `index` (-1 = last)."""
        ids = np.arange(int(np.prod(self.node_shape))).reshape(self.node_shape)
        return np.take(ids, index, axis=axis).ravel()

    @staticmethod
    def dofs(nodes: np.ndarray, components=(0, 1, 2)) -> np.ndarray:
        nodes = np.asarray(nodes)
        return (3 * nodes[:, None] + np.asarray(components)[None]).ravel()

    def element_centres(self) -> np.ndarray:
        """(nelx, nely, nelz, 3) element centre coordinates in metres."""
        g = np.meshgrid(*(np.arange(n) + 0.5 for n in self.shape), indexing="ij")
        return np.stack(g, axis=-1) * self.voxel


@dataclass
class TopOptConfig:
    volfrac: float = 0.3        # target fraction of the (non-void) design domain
    penal: float = 3.0          # SIMP exponent p
    rmin: float = 1.5           # density filter radius, in elements
    move: float = 0.2           # OC move limit
    max_iter: int = 100
    tol_change: float = 0.01    # stop when max |x_new - x| falls below this
    emin_ratio: float = 1e-9    # Emin / E0
    cg_tol: float = 1e-6        # relative residual of the PCG solve
    cg_maxit: int = 5000
    threads: int = 4            # OpenMP threads of the native kernels
    backend: str = "auto"       # "auto" | "native" | "numpy"
    solver: str = "auto"        # native: "auto" (multigrid when the grid coarsens) | "mg" | "jacobi"
    mg_levels: int = 4
    mg_sweeps: int = 2          # damped-Jacobi sweeps before and after each coarse correction
    mg_omega: float = 0.6

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TopOptResult:
    x: np.ndarray                     # physical (filtered) densities, (nelx, nely, nelz)
    x_design: np.ndarray              # design variables
    compliance: float                 # weighted sum of compliances of the returned design (J)
    compliance_cases: list[float]
    volume: float                     # fraction of the non-void domain
    iterations: int
    converged: bool
    stopped: bool
    backend: str
    seconds: float
    history: list[dict] = field(default_factory=list)
    displacements: np.ndarray | None = None  # (n_cases, ndof) for the returned design

    def summary(self) -> dict:
        return {"compliance": self.compliance, "compliance_cases": self.compliance_cases, "volume": self.volume,
                "iterations": self.iterations, "converged": self.converged, "stopped": self.stopped,
                "backend": self.backend, "seconds": self.seconds}


# ---------------------------------------------------------------------------------------------
# backends: filter, linear solve, element energies
# ---------------------------------------------------------------------------------------------
def _filter_kernel(rmin: float) -> np.ndarray:
    R = max(0, int(np.ceil(rmin)) - 1)
    d = np.arange(-R, R + 1)
    di, dj, dk = np.meshgrid(d, d, d, indexing="ij")
    return np.maximum(0.0, rmin - np.sqrt(di ** 2 + dj ** 2 + dk ** 2))


class _NumpyBackend:
    """Assembled scipy.sparse stiffness + scipy CG with the same Jacobi preconditioner."""
    name = "numpy"

    def __init__(self, prob: TopOptProblem, KE: np.ndarray, cfg: TopOptConfig):
        from scipy import ndimage, sparse
        self._ndimage, self._sparse = ndimage, sparse
        self.prob, self.KE, self.cfg = prob, KE, cfg
        self.kernel = _filter_kernel(cfg.rmin)
        nx, ny, nz = prob.shape
        ns = prob.node_shape
        i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        base = ((i * ns[1] + j) * ns[2] + k).ravel()
        off = (_CORNERS[:, 0] * ns[1] * ns[2] + _CORNERS[:, 1] * ns[2] + _CORNERS[:, 2]).astype(np.int64)
        nodes = base[:, None] + off[None]                       # (nel, 8)
        self.edof = (3 * nodes[:, :, None] + np.arange(3)).reshape(-1, 24)  # (nel, 24)
        n = prob.ndof
        iK = np.repeat(self.edof, 24, axis=1).ravel()
        jK = np.tile(self.edof, (1, 24)).ravel()
        keys = iK.astype(np.int64) * n + jK
        uniq, self.inv = np.unique(keys, return_inverse=True)
        rows, cols = uniq // n, uniq % n
        self.indptr = np.searchsorted(rows, np.arange(n + 1)).astype(np.int64)
        self.indices = cols.astype(np.int64)
        fixed = prob.fixed
        self.zero_mask = fixed[rows] | fixed[cols]
        self.fixed_diag = np.flatnonzero(fixed[rows] & (rows == cols))
        self.ke_flat = KE.ravel()
        self.nnz = len(uniq)

    def filter(self, a: np.ndarray) -> np.ndarray:
        return self._ndimage.correlate(a.reshape(self.prob.shape), self.kernel, mode="constant", cval=0.0).ravel()

    def matrix(self, E: np.ndarray):
        sK = (E[:, None] * self.ke_flat[None]).ravel()
        data = np.bincount(self.inv, weights=sK, minlength=self.nnz)
        data[self.zero_mask] = 0.0
        data[self.fixed_diag] = 1.0
        n = self.prob.ndof
        return self._sparse.csr_matrix((data, self.indices, self.indptr), shape=(n, n))

    def solve(self, E: np.ndarray, f: np.ndarray, u0: np.ndarray) -> tuple[np.ndarray, int]:
        from scipy.sparse.linalg import cg
        K = self.matrix(E)
        d = K.diagonal().copy()
        d[d <= 0] = 1.0
        M = self._sparse.diags(1.0 / d)
        f = np.where(self.prob.fixed, 0.0, f)
        x0 = np.where(self.prob.fixed, 0.0, u0)
        if not f.any():
            return np.zeros_like(f), 0
        count = [0]

        def cb(_):
            count[0] += 1
        u, info = cg(K, f, x0=x0, rtol=self.cfg.cg_tol, atol=0.0, maxiter=self.cfg.cg_maxit, M=M, callback=cb)
        return u, count[0] if info == 0 else -count[0]

    def solve_all(self, E: np.ndarray, F: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, list[int]]:
        out, its = np.empty_like(U), []
        for k in range(len(F)):
            out[k], i = self.solve(E, F[k], U[k])
            its.append(i)
        return out, its

    def energy(self, u: np.ndarray) -> np.ndarray:
        ue = u[self.edof]
        return np.einsum("ij,jk,ik->i", ue, self.KE, ue)


_lib = None


def _native_lib():
    """Load bin/topopt.dll once and declare its ctypes signatures; None when it is not built."""
    global _lib
    if _lib is None:
        lib = native.load_library("topopt")
        if lib is None:
            _lib = False
            return None
        f64 = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
        u8 = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
        i, d = ctypes.c_int, ctypes.c_double
        lib.tp_version.restype = i
        lib.tp_matvec.argtypes = [i, i, i, f64, f64, u8, f64, f64, i]
        lib.tp_matvec.restype = None
        lib.tp_diag.argtypes = [i, i, i, f64, f64, u8, f64, i]
        lib.tp_diag.restype = None
        lib.tp_pcg.argtypes = [i, i, i, f64, f64, u8, f64, f64, d, i, i, ctypes.POINTER(d)]
        lib.tp_pcg.restype = i
        lib.tp_element_energy.argtypes = [i, i, i, f64, f64, f64, i]
        lib.tp_element_energy.restype = None
        lib.tp_filter.argtypes = [i, i, i, d, f64, f64, i]
        lib.tp_filter.restype = None
        i32 = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
        lib.tp_mgpcg.argtypes = [i, i, i, f64, f64, u8, i, f64, f64, d, i, i, i, i, d, i32, f64]
        lib.tp_mgpcg.restype = i
        _lib = lib
    return _lib or None


def native_available() -> bool:
    return _native_lib() is not None


class _NativeBackend:
    name = "native"

    def __init__(self, prob: TopOptProblem, KE: np.ndarray, cfg: TopOptConfig):
        self.lib = _native_lib()
        if self.lib is None:
            raise RuntimeError("native topopt library is not built (python -m simulatics build-native)")
        self.prob, self.cfg = prob, cfg
        self.KE = np.ascontiguousarray(KE, np.float64)
        self.fixed = np.ascontiguousarray(prob.fixed, np.uint8)
        self.dims = prob.shape

    def filter(self, a: np.ndarray) -> np.ndarray:
        a = np.ascontiguousarray(a, np.float64).ravel()
        out = np.empty_like(a)
        self.lib.tp_filter(*self.dims, float(self.cfg.rmin), a, out, int(self.cfg.threads))
        return out

    def solve(self, E: np.ndarray, f: np.ndarray, u0: np.ndarray) -> tuple[np.ndarray, int]:
        u = np.array(u0, np.float64, copy=True, order="C")
        rel = ctypes.c_double(0.0)
        it = self.lib.tp_pcg(*self.dims, self.KE, np.ascontiguousarray(E, np.float64), self.fixed,
                             np.ascontiguousarray(f, np.float64), u, float(self.cfg.cg_tol), int(self.cfg.cg_maxit),
                             int(self.cfg.threads), ctypes.byref(rel))
        return u, int(it)

    def solve_all(self, E: np.ndarray, F: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, list[int]]:
        """All load cases at once; multigrid-preconditioned when the grid can be coarsened."""
        cfg = self.cfg
        U = np.array(U, np.float64, copy=True, order="C")
        if cfg.solver in ("auto", "mg"):
            its = np.zeros(len(F), np.int32)
            rel = np.zeros(len(F))
            levels = self.lib.tp_mgpcg(*self.dims, self.KE, np.ascontiguousarray(E, np.float64), self.fixed, len(F),
                                       np.ascontiguousarray(F, np.float64), U, float(cfg.cg_tol), int(cfg.cg_maxit),
                                       int(cfg.threads), int(cfg.mg_levels), int(cfg.mg_sweeps), float(cfg.mg_omega),
                                       its, rel)
            self.levels = levels
            if levels > 0:
                return U, [int(i) for i in its]
            if cfg.solver == "mg":
                raise ValueError(f"multigrid needs even grid dimensions, got {self.dims}")
        self.levels = 1
        its = []
        for k in range(len(F)):
            U[k], i = self.solve(E, F[k], U[k])
            its.append(i)
        return U, its

    def energy(self, u: np.ndarray) -> np.ndarray:
        out = np.empty(self.prob.nel)
        self.lib.tp_element_energy(*self.dims, self.KE, np.ascontiguousarray(u, np.float64), out,
                                   int(self.cfg.threads))
        return out

    def matvec(self, E: np.ndarray, u: np.ndarray) -> np.ndarray:
        y = np.empty(self.prob.ndof)
        self.lib.tp_matvec(*self.dims, self.KE, np.ascontiguousarray(E, np.float64), self.fixed,
                           np.ascontiguousarray(u, np.float64), y, int(self.cfg.threads))
        return y


def make_backend(prob: TopOptProblem, cfg: TopOptConfig):
    KE = hex8_stiffness(prob.nu, prob.voxel)
    if cfg.backend == "numpy" or (cfg.backend == "auto" and not native_available()):
        return _NumpyBackend(prob, KE, cfg)
    return _NativeBackend(prob, KE, cfg)


# ---------------------------------------------------------------------------------------------
# optimisation loop
# ---------------------------------------------------------------------------------------------
def _stiffness(xp: np.ndarray, prob: TopOptProblem, cfg: TopOptConfig) -> np.ndarray:
    emin = cfg.emin_ratio * prob.E0
    return emin + xp ** cfg.penal * (prob.E0 - emin)


def evaluate_compliance(prob: TopOptProblem, density: np.ndarray, cfg: TopOptConfig | None = None,
                        penal: float | None = None) -> tuple[float, list[float], np.ndarray]:
    """Compliance of a given (physical) density field: (weighted total, per case, displacements)."""
    cfg = cfg or TopOptConfig()
    if penal is not None:
        cfg = TopOptConfig(**{**cfg.to_dict(), "penal": penal})
    be = make_backend(prob, cfg)
    E = _stiffness(np.asarray(density, float).ravel(), prob, cfg)
    U, _ = be.solve_all(E, prob.loads, np.zeros((len(prob.loads), prob.ndof)))
    cases = [float(f @ u) for f, u in zip(prob.loads, U)]
    return float(np.dot(prob.weights, cases)), cases, U


def _check_inputs(prob: TopOptProblem, cfg: TopOptConfig, x0: np.ndarray | None) -> None:
    """Reject problems the OC loop cannot solve, with a message instead of NaNs or a division by zero."""
    if not 0.0 < cfg.volfrac <= 1.0:
        raise ValueError(f"volfrac must be in (0, 1], got {cfg.volfrac}")
    if not cfg.rmin > 0.0:
        raise ValueError(f"rmin must be > 0 elements (rmin <= 1 means no filtering), got {cfg.rmin}")
    if not np.isfinite(prob.loads).all():
        raise ValueError("loads contain NaN or inf")
    if not prob.fixed.any():
        raise ValueError("no fixed DOFs: the structure is unsupported (rigid-body modes)")
    if not np.abs(prob.loads[:, ~prob.fixed]).any():
        raise ValueError("no load acts on a free DOF: every compliance is zero")
    if x0 is not None and np.size(x0) != prob.nel:
        raise ValueError(f"x0 must have {prob.nel} entries {prob.shape}, got {np.shape(x0)}")


def run_topopt(prob: TopOptProblem, cfg: TopOptConfig | None = None,
               callback: Callable[[dict], None] | None = None,
               should_stop: Callable[[], bool] | None = None,
               x0: np.ndarray | None = None) -> TopOptResult:
    """Minimise the weighted sum of compliances subject to mean(xPhys) = volfrac (OC, density filter).

    callback(info) is called after every iteration with keys iteration, compliance, volume, change,
    cg_iterations, seconds and x (the physical densities the iteration evaluated, not a copy). should_stop() is polled between iterations; returning True ends the run
    early (the result then has stopped=True).
    """
    cfg = cfg or TopOptConfig()
    _check_inputs(prob, cfg, x0)
    t_start = time.perf_counter()
    be = make_backend(prob, cfg)
    solid, void = prob.passive_solid.ravel(), prob.passive_void.ravel()
    active = ~(solid | void)
    n_dom = max(1, int((~void).sum()))
    if solid.sum() / n_dom > cfg.volfrac:
        raise ValueError(f"passive solid elements alone exceed the volume fraction {cfg.volfrac}")
    Hs = be.filter(np.ones(prob.nel))

    def physical(x):
        xp = be.filter(x) / Hs
        xp[solid], xp[void] = 1.0, 0.0
        return xp

    if x0 is not None:
        x = np.clip(np.asarray(x0, float).ravel().copy(), 0, 1)
    else:
        # uniform start whose physical volume equals volfrac
        x = np.full(prob.nel, 0.0)
        n_act = max(1, int(active.sum()))
        x[active] = (cfg.volfrac * n_dom - solid.sum()) / n_act
    x[solid], x[void] = 1.0, 0.0
    L = len(prob.loads)
    U = np.zeros((L, prob.ndof))
    dE = (1 - cfg.emin_ratio) * prob.E0
    history: list[dict] = []
    converged = stopped = False
    it = 0
    for it in range(1, cfg.max_iter + 1):
        if should_stop is not None and should_stop():
            stopped = True
            it -= 1
            break
        t0 = time.perf_counter()
        xp = physical(x)
        E = _stiffness(xp, prob, cfg)
        c, cases = 0.0, []
        dc = np.zeros(prob.nel)
        U, its = be.solve_all(E, prob.loads, U)
        cg_its = int(sum(abs(i) for i in its))
        for k in range(L):
            ck = float(prob.loads[k] @ U[k])
            cases.append(ck)
            c += prob.weights[k] * ck
            dc -= prob.weights[k] * cfg.penal * xp ** (cfg.penal - 1) * dE * be.energy(U[k])
        if not np.isfinite(c):
            raise RuntimeError(f"linear solve failed at iteration {it} (compliance {c}); check supports and loads")
        vol = float(xp[~void].sum() / n_dom)
        # chain rule through the density filter (H is symmetric)
        dc = be.filter(dc / Hs)
        dv = be.filter(np.ones(prob.nel) / Hs)
        g = np.maximum(-dc, 0.0) / np.maximum(dv, 1e-30)
        g[~active] = 0.0
        gmax = g.max()
        if gmax > 0:
            g /= gmax
        lo_b, hi_b = np.maximum(0.0, x - cfg.move), np.minimum(1.0, x + cfg.move)
        # bisection on the multiplier; a positive lower end keeps it finite when even the largest step
        # cannot reach volfrac (move limit, volfrac = 1): it then ends at l2 ~ l1 with x at its upper bounds
        l1, l2 = 1e-30, 1e9
        xnew = x
        while (l2 - l1) / (l1 + l2) > 1e-4:
            lm = 0.5 * (l1 + l2)
            xnew = np.clip(x * np.sqrt(g / lm), lo_b, hi_b)
            xnew[solid], xnew[void] = 1.0, 0.0
            if physical(xnew)[~void].sum() / n_dom > cfg.volfrac:
                l1 = lm
            else:
                l2 = lm
        change = float(np.abs(xnew - x)[active].max()) if active.any() else 0.0
        x = xnew
        info = {"iteration": it, "compliance": c, "volume": vol, "change": change,
                "cg_iterations": cg_its, "seconds": time.perf_counter() - t0}
        history.append(info)
        if callback is not None:
            callback({**info, "x": xp.reshape(prob.shape)})
        if change < cfg.tol_change:
            converged = True
            break
    # evaluate the returned design so compliance and densities belong together
    xp = physical(x)
    E = _stiffness(xp, prob, cfg)
    U, _ = be.solve_all(E, prob.loads, U)
    cases = [float(prob.loads[k] @ U[k]) for k in range(L)]
    return TopOptResult(
        x=xp.reshape(prob.shape), x_design=x.reshape(prob.shape), compliance=float(np.dot(prob.weights, cases)),
        compliance_cases=cases, volume=float(xp[~void].sum() / n_dom), iterations=it, converged=converged,
        stopped=stopped, backend=be.name, seconds=time.perf_counter() - t_start, history=history,
        displacements=U)


# ---------------------------------------------------------------------------------------------
# problem builders
# ---------------------------------------------------------------------------------------------
def cantilever(nelx: int = 48, nely: int = 12, nelz: int = 12, voxel: float = 1.0, E0: float = 1.0,
               nu: float = 0.3, force: float = 1.0) -> TopOptProblem:
    """Classic 3D cantilever benchmark (as in top3d): face x = 0 clamped, a downward line load of total
    `force` along the bottom edge (z = 0) of the free end x = L."""
    prob = TopOptProblem(nelx, nely, nelz, voxel=voxel, E0=E0, nu=nu)
    prob.fixed[TopOptProblem.dofs(prob.nodes_where(0, 0))] = True
    tip = np.intersect1d(prob.nodes_where(0, -1), prob.nodes_where(2, 0))
    prob.loads[0, 3 * tip + 2] = -force / len(tip)
    return prob


def wrench_to_nodal_forces(points: np.ndarray, centre: np.ndarray, force: np.ndarray,
                           moment: np.ndarray) -> np.ndarray:
    """Distribute a wrench (force, moment about `centre`) over interface nodes, RBE3-like.

    f_i = F/n + w x r_i with r_i = p_i - centroid and w = I_r^-1 M_o, where
    I_r = sum_i (|r_i|^2 Id - r_i r_i^T) and M_o is the moment about the node centroid. The nodal forces
    reproduce the resultant force and moment exactly.
    """
    points = np.asarray(points, float)
    o = points.mean(axis=0)
    r = points - o
    force, moment = np.asarray(force, float), np.asarray(moment, float)
    M_o = moment + np.cross(np.asarray(centre, float) - o, force)
    I_r = (r * r).sum() * np.eye(3) - r.T @ r
    w = np.linalg.lstsq(I_r, M_o, rcond=None)[0]
    return force / len(points) + np.cross(w, r)


# ---------------------------------------------------------------------------------------------
# density field -> surface mesh
# ---------------------------------------------------------------------------------------------
def density_to_mesh(x: np.ndarray, voxel: float, iso: float = 0.5, smooth_iterations: int = 6,
                    origin=(0.0, 0.0, 0.0), min_component_fraction: float = 0.02):
    """Iso-surface of a density field as a closed trimesh.Trimesh (marching cubes + Taubin smoothing).

    Marching cubes: Lorensen and Cline (1987), scikit-image's Lewiner variant; the field is padded with
    zeros so the surface is closed. Taubin (1995) lambda/mu smoothing removes voxel steps without the
    shrinkage of plain Laplacian smoothing. Components smaller than `min_component_fraction` of the
    total volume (floating islands) are dropped. Vertex coordinates are in metres, element (0,0,0)
    occupying [0, voxel]^3 shifted by `origin`.
    """
    import warnings

    import trimesh
    from skimage.measure import marching_cubes

    x = np.asarray(x, float)
    if x.max() < iso:
        raise ValueError("density field has no material above the iso level")
    padded = np.pad(x, 1, mode="constant", constant_values=0.0)
    with warnings.catch_warnings():  # scikit-image 0.26 trips a NumPy 2.5 deprecation internally
        warnings.simplefilter("ignore", DeprecationWarning)
        verts, faces, _normals, _vals = marching_cubes(padded, level=iso, spacing=(voxel, voxel, voxel))
    verts = verts - 0.5 * voxel + np.asarray(origin, float)  # padded index p -> element centre (p - 0.5) h
    mesh = trimesh.Trimesh(verts, faces, process=True)
    if mesh.volume < 0:
        mesh.invert()
    if smooth_iterations > 0:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=0.53, iterations=smooth_iterations)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        vols = np.array([abs(p.volume) for p in parts])
        keep = [p for p, v in zip(parts, vols) if v >= min_component_fraction * vols.sum()]
        mesh = trimesh.util.concatenate(keep or [parts[int(np.argmax(vols))]])  # never drop everything
    if mesh.volume < 0:
        mesh.invert()
    return mesh
