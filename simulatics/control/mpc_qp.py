"""Small dense convex QP solver for the walking MPC: a dual active-set method.

    minimise  1/2 x^T H x + g^T x     subject to  C x <= b          (H symmetric positive definite)

Algorithm: the dual active-set method of Goldfarb and Idnani ("A numerically
stable dual method for solving strictly convex quadratic programs",
Mathematical Programming 27, 1983), written here from the paper's description
in the dual-space form below (no code was taken from any QP library):

* Start at the unconstrained minimum x0 = -H^-1 g (always dual feasible) with an
  empty active set A.
* Every primal point is x = x0 - V_A u_A, with V = H^-1 C^T and the
  multipliers u_A >= 0 of the active constraints, so the whole method works on
  the m x m matrix W = C H^-1 C^T (the dual Hessian).
* Pick a violated constraint p and raise its multiplier t while the active
  constraints stay tight: the active multipliers move by -t r with
  r = W_AA^-1 W_Ap, and the slack of p grows at the rate
  zn = W_pp - W_pA r (a Schur complement, > 0 unless p depends on A).
  - full step t2 = -slack_p / zn: p becomes active;
  - partial step t1 = min over r_j > 0 of u_j / r_j: an active multiplier
    reaches zero and that constraint leaves A, then try p again.
* Stop when no constraint is violated (the KKT conditions hold), or report
  infeasibility when neither step is bounded.

Each iteration solves one |A| x |A| system (a Cholesky factorisation of W_AA;
|A| stays below ~40 for the MPC), so no factor updates are needed and the
numpy and C++ (native/src/mpc.cpp) versions follow exactly the same steps with
the same tie-breaking. Warm start: `warm` lists constraints to try first (the
previous solve's active set, mapped onto this problem), which removes most of
the add/drop churn of a cold start; the result does not depend on it.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

OK, MAX_ITER, INFEASIBLE, NOT_CONVEX = 0, 1, 2, 3
STATUS = {OK: "ok", MAX_ITER: "max iterations", INFEASIBLE: "infeasible", NOT_CONVEX: "H not positive definite"}

_TOL = 1e-9          # constraint violation tolerance (units of b)
_EPS_R = 1e-12       # r_j > _EPS_R counts as a blocking active constraint
_EPS_Z = 1e-11       # zn <= _EPS_Z * W_pp: p is linearly dependent on the active set


@dataclass
class QPResult:
    x: np.ndarray
    active: np.ndarray       # indices of the active constraints (order of activation)
    mult: np.ndarray         # their multipliers (>= 0)
    iterations: int
    status: int              # OK / MAX_ITER / INFEASIBLE / NOT_CONVEX
    native: bool = False

    @property
    def ok(self) -> bool:
        return self.status == OK


# ------------------------------------------------------------------ native binding
_native = None
_native_tried = False


def _load_native():
    global _native, _native_tried
    if _native_tried:
        return _native
    _native_tried = True
    try:
        from simulatics import native
        lib = native.load_library("mpc")
        if lib is None:
            return None
        d = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
        i = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
        lib.mpc_qp_solve.argtypes = [ctypes.c_int, ctypes.c_int, d, d, d, d, i, ctypes.c_int, ctypes.c_int,
                                     d, i, d, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        lib.mpc_qp_solve.restype = ctypes.c_int
        lib.mpc_version.restype = ctypes.c_int
        _native = lib
    except (OSError, AttributeError, ImportError):
        _native = None
    return _native


def native_available() -> bool:
    return _load_native() is not None


def solve_qp(H, g, C, b, warm=None, max_iter: int = 300, force_python: bool = False) -> QPResult:
    """Solve min 1/2 x'Hx + g'x s.t. Cx <= b. warm: constraint indices to try first (e.g. the last active set).

    Uses the C++ kernel (native/src/mpc.cpp, built with native.build(target="mpc")) when it is built, else the
    numpy implementation of the same algorithm. On MAX_ITER / INFEASIBLE the returned x is the last iterate:
    optimal for the constraints activated so far (a sensible fallback inside a controller)."""
    H = np.ascontiguousarray(H, np.float64)
    g = np.ascontiguousarray(g, np.float64).ravel()
    n = g.size
    C = np.ascontiguousarray(np.asarray(C, np.float64).reshape(-1, n))
    b = np.ascontiguousarray(b, np.float64).ravel()
    m = b.size
    w = np.ascontiguousarray([] if warm is None else warm, np.int32).ravel()
    lib = None if force_python else _load_native()
    if lib is not None:
        x = np.zeros(n)
        act = np.zeros(max(m, 1), np.int32)
        mult = np.zeros(max(m, 1))
        n_act, iters = ctypes.c_int(0), ctypes.c_int(0)
        st = lib.mpc_qp_solve(n, m, H, g, C if m else np.zeros((1, n)), b if m else np.zeros(1), w, w.size,
                              int(max_iter), x, act, mult, ctypes.byref(n_act), ctypes.byref(iters))
        k = n_act.value
        return QPResult(x, act[:k].copy(), mult[:k].copy(), iters.value, int(st), True)
    return _solve_numpy(H, g, C, b, w, max_iter)


def _solve_numpy(H, g, C, b, warm, max_iter) -> QPResult:
    n, m = g.size, b.size
    try:
        L = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        return QPResult(np.zeros(n), np.zeros(0, int), np.zeros(0), 0, NOT_CONVEX)
    x0 = -cho_solve((L, True), g)
    if m == 0:
        return QPResult(x0, np.zeros(0, int), np.zeros(0), 0, OK)
    Y = solve_triangular(L, C.T, lower=True)          # L Y = C^T
    V = solve_triangular(L.T, Y, lower=False)         # V = H^-1 C^T
    W = Y.T @ Y                                       # C H^-1 C^T
    x = x0.copy()
    A: list[int] = []
    u = np.zeros(0)
    queue = [int(i) for i in warm if 0 <= int(i) < m]
    qpos = 0
    it = 0
    is_active = np.zeros(m, bool)
    while True:
        s = b - C @ x
        p = -1
        while qpos < len(queue):
            c = queue[qpos]
            qpos += 1
            if not is_active[c] and s[c] < -_TOL:
                p = c
                break
        if p < 0:
            s_free = np.where(is_active, np.inf, s)
            p = int(np.argmin(s_free))
            if s_free[p] >= -_TOL:
                return QPResult(x, np.array(A, int), u.copy(), it, OK)
        up = 0.0
        while True:
            it += 1
            if it > max_iter:
                return QPResult(x, np.array(A, int), u.copy(), it - 1, MAX_ITER)
            if A:
                try:
                    r = cho_solve(cho_factor(W[np.ix_(A, A)], lower=True), W[A, p])
                except np.linalg.LinAlgError:
                    return QPResult(x, np.array(A, int), u.copy(), it, NOT_CONVEX)
                zn = W[p, p] - W[p, A] @ r
            else:
                r = np.zeros(0)
                zn = W[p, p]
            t1, k = np.inf, -1
            for j in range(len(A)):
                if r[j] > _EPS_R:
                    ratio = u[j] / r[j]
                    if ratio < t1:
                        t1, k = ratio, j
            sp = b[p] - C[p] @ x
            t2 = -sp / zn if zn > _EPS_Z * max(W[p, p], 1e-300) else np.inf
            if not np.isfinite(t1) and not np.isfinite(t2):
                return QPResult(x, np.array(A, int), u.copy(), it, INFEASIBLE)
            if t2 <= t1:                                 # full step: p joins the active set
                u = np.append(u - t2 * r, up + t2)
                A.append(p)
                is_active[p] = True
                x = x0 - V[:, A] @ u
                break
            # partial step: active constraint k drops out, then try p again
            u = np.maximum(u - t1 * r, 0.0)
            up += t1
            is_active[A[k]] = False
            del A[k]
            u = np.delete(u, k)
            x = x0 - (V[:, A] @ u if A else 0.0) - V[:, p] * up
