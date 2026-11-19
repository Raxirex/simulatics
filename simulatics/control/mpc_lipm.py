"""Linear MPC of the linear inverted pendulum with automatic footstep placement.

The pattern generator of the MPC walker (control/mpc.py), after

* P.-B. Wieber, "Trajectory free linear model predictive control for stable
  walking in the presence of strong perturbations", Humanoids 2006: the CoM
  jerk over a receding horizon is the decision variable, the ZMP must stay in
  the support polygon at every sample, and the QP is re-solved from the
  *measured* CoM state, so perturbations are rejected without a stored
  reference trajectory;
* A. Herdt, H. Diedam, P.-B. Wieber, D. Dimitrov, K. Mombaur, M. Diehl,
  "Online walking motion generation with automatic footstep placement",
  Advanced Robotics 24(5-6), 2010: the positions of the next footsteps are
  decision variables too, the support polygon of each sample is the
  (yaw-rotated) sole of the foot that will be on the ground then, and the
  footsteps are kept in a kinematically reachable box relative to the previous
  foot;
* the terminal capture-point (DCM) term follows the stability condition of
  Scianca, De Simone, Lanari, Oriolo, "MPC for humanoid gait generation:
  stability and feasibility", IEEE T-RO 36(4), 2020 (here as a cost, not a
  constraint): the DCM at the end of the horizon should sit on the ZMP
  reference, so the predicted motion does not diverge after the horizon.

Model, per horizontal axis, sampling period T (cart-table / LIPM with constant
CoM height z_c, Kajita et al. ICRA 2003):

    s = (c, c', c''),  s[k+1] = A s[k] + B jerk[k],  zmp = c - (z_c / g) c''

Stacking N samples: C = Pps s0 + Ppu J,  V = Pvs s0 + Pvu J,  Z = Pzs s0 + Pzu J.

Decision vector  [jerk_x (N), foot_x (m), jerk_y (N), foot_y (m)]  (m <= 3 footsteps).

Cost (one factor 1/2 dropped):
    zmp  * sum_k |Z_k - zref_k(feet)|^2       ZMP centred on the support foot (zref is linear in the footsteps)
  + vel  * sum_k |V_k - vref_k|^2             velocity / reference-path tracking
  + jerk * sum_k |J_k|^2
  + step * sum_j |foot_j - planned_j|^2      stay close to the footstep planner's steps
  + dcm  * |C_N + V_N / omega - zref_N|^2    terminal capture point

Constraints (all linear because footstep yaws come from the plan):
    ZMP of every sample inside its support polygon, shrunk by a margin:
      single support -> the rotated sole rectangle of that (fixed or variable) foot;
      both feet landed -> the exact convex hull of the two soles;
      final side-by-side stance with a variable foot -> inside both soles' fore-aft strips and between them;
    footstep j inside a box in the frame of the previous (opposite) foot: forward/backward reach and a lateral
    band that never lets the legs cross;
    the swing foot's landing point within reach of where the foot is now (speed x remaining swing time).

Sampling: Problem.dts sets the jerk intervals (default N x T from now). The controller can also lay the
interval boundaries on the contact switches and add ZMP checks between samples (Problem.points: constraint
rows only, no extra variables). Note that with T = 0.1 s the ZMP is almost linear in time inside one
interval, so an interval that straddles a lift-off cannot both hold the ZMP on one sole and then carry it to
the other; checks inside such an interval make the QP infeasible (found with an LP feasibility check).

The full objective of a solution (with its constant terms) is returned as Plan.cost, so plans with different
contact timings can be compared (used for step timing adaptation in control/mpc.py).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .mpc_qp import QPResult, solve_qp

G = 9.81


@dataclass
class MPCWeights:
    zmp: float = 1.0          # ZMP centring
    vel: float = 0.02         # CoM velocity tracking
    jerk: float = 1e-5        # CoM jerk
    step: float = 3.0         # footstep closeness to the plan
    dcm: float = 5.0          # terminal capture point


@dataclass
class Foot:
    """A sole on the ground at some point of the horizon: fixed (a landed foot, or a planned step beyond the
    decision variables) or decision variable `var` (then xy is its planned / nominal position)."""
    side: int
    xy: np.ndarray
    yaw: float
    var: int = -1
    key: int = 0              # identity of the footstep (for warm starts)
    weight: float = 1.0       # multiplies MPCWeights.step for this footstep


@dataclass
class Support:
    """Support of one sample. kind: "single" (feet[0]), "hull" (two fixed feet), "strip" (final stance:
    feet = (left, right), at least one variable). ref: ((foot index, weight), ...); the ZMP reference is
    sum(weight * foot position)."""
    kind: str
    feet: tuple
    ref: tuple
    vref: np.ndarray = field(default_factory=lambda: np.zeros(2))


@dataclass
class StepBox:
    var: int                  # constrained footstep variable
    prev: int                 # index (into Problem.feet) of the previous, opposite foot
    side: int                 # side of the constrained footstep (0 left, 1 right)


@dataclass
class Problem:
    s0: np.ndarray                    # (3, 2): CoM position, velocity, acceleration for x and y
    feet: list                        # [Foot]
    supports: list                    # [Support] * N
    boxes: list                       # [StepBox]
    reach: tuple | None = None        # (var, (x, y) now, radius): landing point reachable by the swing foot
    t_now: float = 0.0                # absolute time (s), only used to match warm-start constraints
    dts: np.ndarray | None = None     # lengths of the N jerk intervals (None = N x T); samples at their ends
    points: list = field(default_factory=list)   # [(tau, Support)]: extra ZMP constraint instants (no cost)


@dataclass
class Plan:
    s0: np.ndarray                    # (3, 2) initial state the plan starts from
    jerk: np.ndarray                  # (N, 2)
    steps: np.ndarray                 # (m, 2) optimised footstep positions
    zmp: np.ndarray                   # (N, 2) predicted ZMP at the samples
    com: np.ndarray                   # (N, 2) predicted CoM at the samples
    dts: np.ndarray                   # (N,) sampling intervals (the first one may be shorter / longer than T)
    iterations: int
    status: int
    solve_ms: float
    native: bool
    cost: float = 0.0

    def state(self, tau: float) -> np.ndarray:
        """(3, 2) CoM position / velocity / acceleration tau seconds after the start of the plan."""
        s = self.s0.copy()
        t = max(0.0, float(tau))
        k = 0
        while t > 1e-12 and k < len(self.jerk):
            h = min(t, float(self.dts[k]))
            j = self.jerk[k]
            s = np.array([s[0] + s[1] * h + s[2] * h * h / 2 + j * h ** 3 / 6,
                          s[1] + s[2] * h + j * h * h / 2,
                          s[2] + j * h])
            t -= h
            k += 1
        if t > 1e-12:                                # past the horizon: constant acceleration (rare)
            s = np.array([s[0] + s[1] * t + s[2] * t * t / 2, s[1] + s[2] * t, s[2]])
        return s


class LipmMPC:
    """Condensed LIPM MPC; build once per body (z_c) and call solve() every update."""

    def __init__(self, z_c: float, T: float = 0.1, N: int = 16, weights: MPCWeights | None = None,
                 half_length: float = 0.08, half_width: float = 0.03, reach_fwd: float = 0.25,
                 reach_back: float = 0.15, width_min: float = 0.13, width_max: float = 0.36):
        self.z_c, self.T, self.N = float(z_c), float(T), int(N)
        self.N_nominal = self.N                       # samples of a uniform grid; Problem.dts may differ
        self.omega = float(np.sqrt(G / self.z_c))
        self.w = weights or MPCWeights()
        self.hl, self.hw = float(half_length), float(half_width)        # usable sole half sizes (margins in)
        self.reach_fwd, self.reach_back = float(reach_fwd), float(reach_back)
        self.width_min, self.width_max = float(width_min), float(width_max)
        self._grids: dict = {}
        self._hulls: dict = {}
        self._use_grid(np.full(self.N, self.T))
        self._warm_keys: list = []

    def _use_grid(self, dts) -> np.ndarray:
        """Prediction matrices for jerk intervals of lengths dts (any number, any lengths; small cache)."""
        dts = np.asarray(dts, float)
        key = tuple(np.round(dts, 6))
        g = self._grids.get(key)
        if g is None:
            N = len(dts)
            Ps = np.zeros((N, 3, 3))
            Pu = np.zeros((N, 3, N))
            S, U = np.eye(3), np.zeros((3, N))
            for k in range(N):
                h = dts[k]
                A = np.array([[1.0, h, h * h / 2], [0.0, 1.0, h], [0.0, 0.0, 1.0]])
                S, U = A @ S, A @ U
                U[:, k] += (h ** 3 / 6, h * h / 2, h)
                Ps[k], Pu[k] = S, U
            hz = self.z_c / G
            g = {"dts": dts.copy(), "Pps": Ps[:, 0, :], "Pvs": Ps[:, 1, :], "Pas": Ps[:, 2, :],
                 "Ppu": Pu[:, 0, :], "Pvu": Pu[:, 1, :], "Pau": Pu[:, 2, :]}
            g["Pzs"], g["Pzu"] = g["Pps"] - hz * g["Pas"], g["Ppu"] - hz * g["Pau"]
            g["Pxs"] = g["Pps"][-1] + g["Pvs"][-1] / self.omega             # terminal DCM row
            g["Pxu"] = g["Ppu"][-1] + g["Pvu"][-1] / self.omega
            if len(self._grids) > 256:
                self._grids.clear()
            self._grids[key] = g
        for name, val in g.items():
            setattr(self, name, val)
        self.N = len(g["dts"])
        return g["dts"]

    def reset_warm_start(self) -> None:
        self._warm_keys = []

    # -------------------------------------------------------------- building
    def _zref(self, prob: Problem, m: int):
        """Per sample: ZMP reference = const (N, 2) + Sv (N, m) @ footstep variables."""
        N = self.N
        const, Sv = np.zeros((N, 2)), np.zeros((N, m))
        for k, sup in enumerate(prob.supports):
            for fi, wgt in sup.ref:
                f = prob.feet[fi]
                if f.var >= 0:
                    Sv[k, f.var] += wgt
                else:
                    const[k] += wgt * f.xy
        return const, Sv

    def _constraints(self, prob: Problem, m: int):
        N, nb = self.N, self.N + m
        n = 2 * nb
        rows, rhs, keys = [], [], []
        # constraint instants: the N samples plus the extra points (contact switches, mid-samples)
        taus = np.concatenate([np.cumsum(self.dts), [t for t, _ in prob.points]])
        sups = list(prob.supports) + [sp for _, sp in prob.points]
        if prob.points:
            Pzs, Pzu = self._zmp_rows(taus)
        else:                                       # samples only: the condensed matrices themselves
            Pzs, Pzu = self.Pzs, self.Pzu
        zs = np.stack([Pzs @ prob.s0[:, 0], Pzs @ prob.s0[:, 1]], axis=1)              # free ZMP
        t_ids = np.rint((prob.t_now + taus) / 0.005).astype(int)                        # absolute ids (5 ms)
        zk, znrm, zvar, zd = [], [], [], []

        def zmp_row(k, nrm, foot: Foot | None, d, key):
            # nrm . (z(tau_k) - foot) <= d     (collected; built below in one vectorised pass)
            nrm = np.asarray(nrm, float)
            if foot is not None and foot.var < 0:
                d = d + float(nrm @ foot.xy)
            zk.append(k)
            znrm.append(nrm)
            zvar.append(foot.var if foot is not None else -1)
            zd.append(d)
            keys.append(key)

        for k, sup in enumerate(sups):
            kid = int(t_ids[k])
            if sup.kind == "single":
                f = prob.feet[sup.feet[0]]
                c, s = np.cos(f.yaw), np.sin(f.yaw)
                for e, (nrm, d) in enumerate((((c, s), self.hl), ((-c, -s), self.hl), ((-s, c), self.hw),
                                               ((s, -c), self.hw))):
                    zmp_row(k, np.array(nrm), f, d, ("z", kid, f.key, e))
            elif sup.kind == "hull":
                fa, fb = prob.feet[sup.feet[0]], prob.feet[sup.feet[1]]
                hk = (fa.key, fb.key, *np.round(fa.xy, 9), round(fa.yaw, 9), *np.round(fb.xy, 9), round(fb.yaw, 9))
                edges = self._hulls.get(hk)
                if edges is None:
                    if len(self._hulls) > 64:
                        self._hulls.clear()
                    edges = self._hulls[hk] = _hull_edges([fa, fb], self.hl, self.hw)
                for e, (nrm, d) in enumerate(edges):
                    zmp_row(k, nrm, None, d, ("h", kid, fa.key, fb.key, e))
            elif sup.kind == "strip":                   # final side-by-side stance
                fl, fr = prob.feet[sup.feet[0]], prob.feet[sup.feet[1]]
                yaw = fl.yaw + _wrap(fr.yaw - fl.yaw) / 2
                c, s = np.cos(yaw), np.sin(yaw)
                e1, e2 = np.array([c, s]), np.array([-s, c])
                for e, (nrm, f, d) in enumerate(((e1, fl, self.hl), (-e1, fl, self.hl), (e1, fr, self.hl),
                                                 (-e1, fr, self.hl), (e2, fl, self.hw), (-e2, fr, self.hw))):
                    zmp_row(k, nrm, f, d, ("s", kid, fl.key, fr.key, e))
        if zk:
            K, NR, V = np.array(zk), np.array(znrm), np.array(zvar)
            Z = np.zeros((len(K), n))
            Z[:, 0:N] = NR[:, :1] * Pzu[K]
            Z[:, nb:nb + N] = NR[:, 1:] * Pzu[K]
            iv = np.nonzero(V >= 0)[0]
            Z[iv, N + V[iv]] -= NR[iv, 0]
            Z[iv, nb + N + V[iv]] -= NR[iv, 1]
            rows += list(Z)
            rhs += list(np.array(zd) - (NR * zs[K]).sum(1))
        for bx in prob.boxes:
            P, F = prob.feet[bx.prev], None
            for f in prob.feet:
                if f.var == bx.var:
                    F = f
                    break
            c, s = np.cos(P.yaw), np.sin(P.yaw)
            e1, e2 = np.array([c, s]), np.array([-s, c])
            lat = ((e2, self.width_max), (-e2, -self.width_min)) if bx.side == 0 else \
                ((e2, -self.width_min), (-e2, self.width_max))
            for e, (nrm, d) in enumerate(((e1, self.reach_fwd), (-e1, self.reach_back)) + lat):
                r = np.zeros(n)                         # nrm . (foot - prev) <= d
                r[N + bx.var] += nrm[0]
                r[nb + N + bx.var] += nrm[1]
                bb = d
                if P.var >= 0:
                    r[N + P.var] -= nrm[0]
                    r[nb + N + P.var] -= nrm[1]
                else:
                    bb += nrm @ P.xy
                rows.append(r)
                rhs.append(bb)
                keys.append(("b", F.key if F is not None else bx.var, e))
        if prob.reach is not None:
            j, ctr, rad = prob.reach
            for e, (ax, sg) in enumerate(((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0))):
                r = np.zeros(n)
                r[ax * nb + N + j] = sg
                rows.append(r)
                rhs.append(rad + sg * ctr[ax])
                keys.append(("r", e))
        C = np.array(rows) if rows else np.zeros((0, n))
        return C, np.array(rhs), keys

    def _zmp_rows(self, taus) -> tuple[np.ndarray, np.ndarray]:
        """ZMP at arbitrary times tau (s from now) = Pzs(tau) s0 + Pzu(tau) jerk (rows of (len, 3), (len, N)):
        the state at the start of the interval containing tau, propagated by the jerk of that interval."""
        N, hz = self.N, self.z_c / G
        bounds = np.concatenate([[0.0], np.cumsum(self.dts)])
        Ps = np.zeros((len(taus), 3))
        Pu = np.zeros((len(taus), N))
        for i, tau in enumerate(taus):
            k = int(np.clip(np.searchsorted(bounds, tau - 1e-12) - 1, 0, N - 1))
            h = float(tau - bounds[k])
            if k == 0:
                S, U = np.eye(3), np.zeros((3, N))
            else:
                S = np.stack([self.Pps[k - 1], self.Pvs[k - 1], self.Pas[k - 1]])
                U = np.stack([self.Ppu[k - 1], self.Pvu[k - 1], self.Pau[k - 1]])
            # position and acceleration rows after h seconds with jerk k
            pos_s = S[0] + h * S[1] + h * h / 2 * S[2]
            acc_s = S[2]
            pos_u = U[0] + h * U[1] + h * h / 2 * U[2]
            acc_u = U[2].copy()
            pos_u[k] += h ** 3 / 6
            acc_u[k] += h
            Ps[i] = pos_s - hz * acc_s
            Pu[i] = pos_u - hz * acc_u
        return Ps, Pu

    def _cost(self, prob: Problem, m: int):
        N, nb, w = self.N, self.N + m, self.w
        const, Sv = self._zref(prob, m)
        ws = self.dts / self.T                                             # per-sample weights (uneven grid)
        D = np.hstack([self.Pzu, -Sv])                                    # ZMP centring residual rows
        Dv = np.hstack([self.Pvu, np.zeros((N, m))])
        Dx = np.concatenate([self.Pxu, -Sv[-1]])
        Hb = w.zmp * (D.T * ws) @ D + w.vel * (Dv.T * ws) @ Dv + w.dcm * np.outer(Dx, Dx)
        Hb[:N, :N] += w.jerk * np.diag(ws)
        sw = np.ones(m)
        for f in prob.feet:
            if f.var >= 0:
                sw[f.var] = f.weight
        Hb[N:, N:] += w.step * np.diag(sw)
        H = np.zeros((2 * nb, 2 * nb))
        H[:nb, :nb] = Hb
        H[nb:, nb:] = Hb
        g = np.zeros(2 * nb)
        vref = np.array([s.vref for s in prob.supports])
        nom = np.zeros((m, 2))
        for f in prob.feet:
            if f.var >= 0:
                nom[f.var] = f.xy
        for a in range(2):
            s0 = prob.s0[:, a]
            e = self.Pzs @ s0 - const[:, a]
            ev = self.Pvs @ s0 - vref[:, a]
            ex = self.Pxs @ s0 - const[-1, a]
            ga = w.zmp * D.T @ (ws * e) + w.vel * Dv.T @ (ws * ev) + w.dcm * Dx * ex
            ga[N:] -= w.step * sw * nom[:, a]
            g[a * nb:(a + 1) * nb] = ga
        return H, g

    # ----------------------------------------------------------------- solve
    def solve(self, prob: Problem, force_python: bool = False) -> Plan:
        t0 = time.perf_counter()
        self._use_grid(np.full(self.N_nominal, self.T) if prob.dts is None else prob.dts)
        m = 1 + max([f.var for f in prob.feet] + [-1])
        N, nb = self.N, self.N + m
        H, g = self._cost(prob, m)
        C, b, keys = self._constraints(prob, m)
        index = {k: i for i, k in enumerate(keys)}
        warm = [index[k] for k in self._warm_keys if k in index]
        res: QPResult = solve_qp(H, g, C, b, warm=warm, force_python=force_python)
        self._warm_keys = [keys[i] for i in res.active]
        x = res.x
        jerk = np.stack([x[:N], x[nb:nb + N]], axis=1)
        steps = np.stack([x[N:nb], x[nb + N:]], axis=1) if m else np.zeros((0, 2))
        zmp = np.stack([self.Pzs @ prob.s0[:, a] + self.Pzu @ jerk[:, a] for a in range(2)], axis=1)
        com = np.stack([self.Pps @ prob.s0[:, a] + self.Ppu @ jerk[:, a] for a in range(2)], axis=1)
        cost = self._objective(prob, jerk, steps, zmp, m)
        return Plan(prob.s0.copy(), jerk, steps, zmp, com, self.dts.copy(), res.iterations, res.status,
                    (time.perf_counter() - t0) * 1e3, res.native, cost)

    def _objective(self, prob: Problem, jerk, steps, zmp, m: int) -> float:
        """The full cost of a solution (with the constant terms), so plans of different contact timings
        can be compared."""
        w = self.w
        const, Sv = self._zref(prob, m)
        zr = const + Sv @ steps if m else const
        ws = self.dts / self.T
        vel = np.stack([self.Pvs @ prob.s0[:, a] + self.Pvu @ jerk[:, a] for a in range(2)], axis=1)
        vref = np.array([sp.vref for sp in prob.supports])
        xi = np.array([self.Pxs @ prob.s0[:, a] + self.Pxu @ jerk[:, a] for a in range(2)])
        c = w.zmp * float(ws @ ((zmp - zr) ** 2).sum(1)) + w.vel * float(ws @ ((vel - vref) ** 2).sum(1))
        c += w.jerk * float(ws @ (jerk ** 2).sum(1)) + w.dcm * float(((xi - zr[-1]) ** 2).sum())
        for f in prob.feet:
            if f.var >= 0:
                c += w.step * f.weight * float(((steps[f.var] - f.xy) ** 2).sum())
        return c


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def sole_corners(f: Foot, hl: float, hw: float) -> np.ndarray:
    c, s = np.cos(f.yaw), np.sin(f.yaw)
    R = np.array([[c, -s], [s, c]])
    return f.xy + np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]]) @ R.T


def _hull_edges(feet, hl: float, hw: float) -> list:
    """Convex hull of fixed soles as half-planes [(normal, d)] with normal . z <= d (Andrew's monotone chain)."""
    pts = np.vstack([sole_corners(f, hl, hw) for f in feet])
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
    hull = lower[:-1] + upper[:-1]                    # counter-clockwise
    edges = []
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        t = b - a
        ln = float(np.hypot(*t))
        if ln < 1e-9:
            continue
        nrm = np.array([t[1], -t[0]]) / ln             # outward for a CCW polygon
        edges.append((nrm, float(nrm @ a)))
    return edges
