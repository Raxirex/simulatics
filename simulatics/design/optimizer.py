"""AI design optimisation: search body proportions and motor gains by simulating every candidate.

Two black-box optimisers over a chosen subset of HumanoidParams ("genes"), each normalised to [0, 1]
over its PARAM_SPEC range:

* "cmaes": CMA-ES (Hansen and Ostermeier 2001; the update equations follow Hansen's tutorial,
  arXiv:1604.00772): rank-mu and rank-one covariance updates, cumulative step-size adaptation,
  default population lambda = 4 + floor(3 ln n). Bounds by repair: the clipped point is simulated
  and the squared distance to it is added as a penalty. As in simulatics.train.evolve.Trainer, the
  distribution mean is evaluated every generation as the "center".
* "bo": Bayesian optimisation with a Gaussian process (Matern-5/2 ARD kernel, hyperparameters by
  maximum marginal likelihood with L-BFGS-B restarts; Rasmussen and Williams 2006) and the expected
  improvement acquisition (Mockus 1978; Jones, Schonlau and Welch 1998, "EGO"). A generation proposes
  `population` points at once with the kriging-believer heuristic (Ginsbourger, Le Riche and
  Carraro 2010) so they can be simulated in parallel. Generation 1 is a scrambled Sobol design that
  includes the starting body.

Each candidate is scored by simulatics.design.evaluate.evaluate_design (walking to goals with the
model-based walker, or the documented standing fallback) in a ProcessPoolExecutor
(max_workers = DesignConfig.workers). The lifecycle mirrors train/evolve.Trainer: start() / stop() /
join() / status(), and the best design so far is written to runs/<name>/design.json.
"""
from __future__ import annotations

import json
import math
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc

from ..models import HumanoidParams, PARAM_SPEC
from .evaluate import EvalConfig, evaluate_worker

# genes that make sense to search (geometry + motors); physics settings are excluded
GENE_NAMES = [k for k, v in PARAM_SPEC.items() if v[4] in ("Geometry", "Motors")]
DEFAULT_GENES = ["thigh_length", "shin_length", "foot_length", "foot_width", "leg_radius", "hip_kp", "knee_kp",
                 "ankle_kp"]


def design_genes() -> list[dict]:
    """[{name, label, default_on, min, max}] for the GUI gene picker."""
    return [{"name": k, "label": PARAM_SPEC[k][3], "default_on": k in DEFAULT_GENES,
             "min": PARAM_SPEC[k][0], "max": PARAM_SPEC[k][1]} for k in GENE_NAMES]


@dataclass
class DesignConfig:
    algo: str = "cmaes"                 # "cmaes" | "bo"
    genes: list[str] = field(default_factory=lambda: list(DEFAULT_GENES))
    generations: int = 10
    population: int = 8                 # candidates per generation (CMA-ES lambda, BO batch size)
    workers: int = 4                    # parallel processes (1 = evaluate in this process)
    sigma0: float = 0.15                # CMA-ES initial step, fraction of each gene's range
    seed: int = 0
    name: str = "design"
    eval: dict = field(default_factory=dict)   # EvalConfig fields

    @classmethod
    def from_dict(cls, d: dict) -> "DesignConfig":
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        cfg.genes = [g for g in cfg.genes if g in PARAM_SPEC] or list(DEFAULT_GENES)
        cfg.population = max(2, int(cfg.population))
        cfg.workers = max(1, int(cfg.workers))
        return cfg


# ---------------------------------------------------------------------------------------------
# CMA-ES
# ---------------------------------------------------------------------------------------------
class CMAES:
    """(mu/mu_w, lambda)-CMA-ES minimising a cost (Hansen's tutorial, arXiv:1604.00772, Algorithm table)."""

    def __init__(self, x0: np.ndarray, sigma0: float, popsize: int | None = None, seed: int = 0):
        n = len(x0)
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.lam = int(popsize or 4 + int(3 * math.log(n)))
        self.mu = self.lam // 2
        w = math.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / float((self.w ** 2).sum())
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.mean = np.asarray(x0, float).copy()
        self.sigma = float(sigma0)
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.B, self.D, self.C = np.eye(n), np.ones(n), np.eye(n)
        self.invsqrtC = np.eye(n)
        self.gen = 0

    def ask(self) -> np.ndarray:
        z = self.rng.standard_normal((self.lam, self.n))
        return self.mean + self.sigma * (z * self.D) @ self.B.T

    def tell(self, X: np.ndarray, cost: np.ndarray) -> None:
        n = self.n
        order = np.argsort(cost)
        old = self.mean
        y = (X[order[:self.mu]] - old) / self.sigma
        yw = self.w @ y
        self.mean = old + self.sigma * yw
        self.gen += 1
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (self.invsqrtC @ yw)
        hsig = (np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chiN
                < 1.4 + 2 / (n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        rank_mu = (y.T * self.w) @ y
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.C = (self.C + self.C.T) / 2
        evals, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(evals, 1e-20))
        self.invsqrtC = self.B @ np.diag(1 / self.D) @ self.B.T


# ---------------------------------------------------------------------------------------------
# Gaussian-process Bayesian optimisation
# ---------------------------------------------------------------------------------------------
def _matern52(A: np.ndarray, B: np.ndarray, ls: np.ndarray, s2: float) -> np.ndarray:
    d = np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) / ls) ** 2, 0).sum(-1))
    r = math.sqrt(5) * d
    return s2 * (1 + r + r * r / 3) * np.exp(-r)


class GaussianProcess:
    """GP regression with a Matern-5/2 ARD kernel on standardised targets (Rasmussen and Williams 2006)."""

    def __init__(self, dim: int, seed: int = 0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.theta = np.concatenate([np.log(np.full(dim, 0.3)), [0.0, math.log(1e-2)]])

    def _unpack(self, th):
        return np.exp(th[:self.dim]), math.exp(th[self.dim]), math.exp(th[self.dim + 1]) + 1e-6

    def _nll(self, th: np.ndarray) -> float:
        ls, s2, sn2 = self._unpack(th)
        K = _matern52(self.X, self.X, ls, s2) + sn2 * np.eye(len(self.X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        a = np.linalg.solve(L.T, np.linalg.solve(L, self.ys))
        # weak log-normal prior on length scales keeps the fit sane with few points
        prior = 0.5 * float(((th[:self.dim] - math.log(0.3)) / 1.5) ** 2 @ np.ones(self.dim))
        return float(0.5 * self.ys @ a + np.log(np.diag(L)).sum()) + prior

    def fit(self, X: np.ndarray, y: np.ndarray, restarts: int = 3) -> None:
        self.X = np.asarray(X, float)
        self.ymu, self.ysd = float(np.mean(y)), float(np.std(y)) or 1.0
        self.ys = (np.asarray(y, float) - self.ymu) / self.ysd
        bounds = [(math.log(0.02), math.log(5.0))] * self.dim + [(math.log(0.05), math.log(20.0)),
                                                                  (math.log(1e-6), math.log(1.0))]
        best = (self._nll(self.theta), self.theta)
        starts = [self.theta] + [np.array([self.rng.uniform(lo, hi) for lo, hi in bounds]) for _ in range(restarts)]
        for th0 in starts:
            r = minimize(self._nll, th0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
            if r.fun < best[0]:
                best = (float(r.fun), r.x)
        self.theta = best[1]
        self._factor()

    def _factor(self) -> None:
        ls, s2, sn2 = self._unpack(self.theta)
        K = _matern52(self.X, self.X, ls, s2) + sn2 * np.eye(len(self.X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.ys))

    def add_fantasy(self, x: np.ndarray, y: float) -> None:
        """Kriging believer: add (x, y) without refitting the hyperparameters."""
        self.X = np.vstack([self.X, x[None]])
        self.ys = np.append(self.ys, (y - self.ymu) / self.ysd)
        self._factor()

    def predict(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ls, s2, _ = self._unpack(self.theta)
        Ks = _matern52(np.atleast_2d(Xs), self.X, ls, s2)
        mu = Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        var = np.maximum(s2 - (v * v).sum(0), 1e-12)
        return mu * self.ysd + self.ymu, np.sqrt(var) * self.ysd


def expected_improvement(mu: np.ndarray, sd: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    """EI for maximisation."""
    imp = mu - best - xi
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z)


class BayesOpt:
    """Batch BO on [0, 1]^n: Sobol initial design, then GP + EI with kriging-believer batches."""

    def __init__(self, dim: int, x0: np.ndarray | None = None, seed: int = 0):
        self.dim, self.seed = dim, seed
        self.rng = np.random.default_rng(seed)
        self.x0 = None if x0 is None else np.clip(np.asarray(x0, float), 0, 1)
        self.X = np.zeros((0, dim))
        self.y = np.zeros(0)
        self.gp = GaussianProcess(dim, seed)

    def ask(self, q: int) -> np.ndarray:
        if len(self.y) < 2:
            m = max(1, math.ceil(math.log2(max(q, 2))))  # Sobol balance needs a power of 2
            pts = qmc.Sobol(self.dim, scramble=True, seed=self.seed).random_base2(m)[:q]
            if self.x0 is not None:
                pts[0] = self.x0
            return pts
        self.gp.fit(self.X, self.y)
        out = []
        for _ in range(q):
            best = float(self.gp.ys.max() * self.gp.ysd + self.gp.ymu)
            top = self.X[np.argsort(-self.y)[:5]]  # local candidates around the best points so far
            cand = np.vstack([self.rng.random((2048, self.dim)),
                              np.clip(top + 0.05 * self.rng.standard_normal(top.shape), 0, 1)])
            mu, sd = self.gp.predict(cand)
            ei = expected_improvement(mu, sd, best)
            x_best, ei_best = cand[int(np.argmax(ei))], float(ei.max())
            for s in cand[np.argsort(-ei)[:3]]:
                r = minimize(lambda x: -float(expected_improvement(*self.gp.predict(x[None]), best)[0]), s,
                             method="L-BFGS-B", bounds=[(0, 1)] * self.dim, options={"maxiter": 50})
                if -r.fun > ei_best:
                    x_best, ei_best = np.clip(r.x, 0, 1), -float(r.fun)
            out.append(x_best)
            self.gp.add_fantasy(x_best, float(self.gp.predict(x_best[None])[0][0]))
        return np.array(out)

    def tell(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = np.vstack([self.X, X])
        self.y = np.concatenate([self.y, y])


# ---------------------------------------------------------------------------------------------
# the optimiser (background thread, Trainer-like lifecycle)
# ---------------------------------------------------------------------------------------------
class DesignOptimizer:
    """Runs in a background thread; status() is safe to call from anywhere."""

    def __init__(self, cfg: DesignConfig, base: HumanoidParams | None = None, runs_dir: str | Path = "runs"):
        self.cfg, self.base = cfg, base or HumanoidParams()
        self.genes = list(cfg.genes)
        self.lo = np.array([PARAM_SPEC[g][0] for g in self.genes])
        self.hi = np.array([PARAM_SPEC[g][1] for g in self.genes])
        self.run_dir = Path(runs_dir) / cfg.name
        self.design_path = self.run_dir / "design.json"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.history: list[dict] = []
        self.state, self.error = "idle", ""
        self.best_score = -np.inf
        self.best_params: HumanoidParams | None = None
        self.best_result: dict | None = None
        self.evaluations = 0
        self.total_steps = 0

    # ---- genes <-> params --------------------------------------------------------------------
    def params_from_genes(self, g: np.ndarray) -> HumanoidParams:
        d = self.base.to_dict()
        for name, v in zip(self.genes, np.clip(g, 0, 1)):
            lo, hi = PARAM_SPEC[name][:2]
            d[name] = float(lo + v * (hi - lo))
        return HumanoidParams.from_dict(d)

    def genes_from_params(self, p: HumanoidParams) -> np.ndarray:
        return np.clip((np.array([getattr(p, g) for g in self.genes]) - self.lo) / (self.hi - self.lo), 0, 1)

    # ---- lifecycle -----------------------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_safe, name="design-optimizer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def status(self) -> dict:
        with self._lock:
            return {
                "state": self.state, "error": self.error, "name": self.cfg.name, "algo": self.cfg.algo,
                "generation": len(self.history), "generations": self.cfg.generations,
                "best": None if not np.isfinite(self.best_score) else float(self.best_score),
                "best_params": ({g: getattr(self.best_params, g) for g in self.genes}
                                if self.best_params is not None else None),
                "best_metrics": (self.best_result or {}).get("metrics"),
                "evaluations": self.evaluations, "total_steps": self.total_steps,
                "history": self.history[-400:],
                "design": str(self.design_path) if self.design_path.exists() else None,
            }

    def _run_safe(self) -> None:
        try:
            self.state = "running"
            self.run()
            self.state = "stopped" if self._stop.is_set() else "finished"
        except Exception as e:  # noqa: BLE001 - surface in the GUI
            import traceback
            traceback.print_exc()
            self.error, self.state = f"{type(e).__name__}: {e}", "error"

    # ---- evaluation --------------------------------------------------------------------------------
    def _evaluate(self, pool, genomes: np.ndarray) -> list[dict] | None:
        eval_cfg = EvalConfig.from_dict(self.cfg.eval).to_dict()
        jobs = [(self.params_from_genes(g).to_dict(), eval_cfg) for g in genomes]
        if pool is None:
            out = []
            for j in jobs:
                if self._stop.is_set():
                    return None
                out.append(evaluate_worker(j))
            return out
        futures: list[Future] = [pool.submit(evaluate_worker, j) for j in jobs]
        pending = set(futures)
        while pending:
            if self._stop.is_set():
                for f in pending:
                    f.cancel()
                return None
            _, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
        return [f.result() for f in futures]

    def _record(self, gen: int, genomes: np.ndarray, results: list[dict], center: float, t0: float) -> None:
        scores = np.array([r["score"] for r in results])
        steps = int(sum(r["sim_steps"] for r in results))
        dt = time.perf_counter() - t0
        i = int(np.argmax(scores))
        improved = False
        with self._lock:
            self.evaluations += len(results)
            self.total_steps += steps
            self.history.append({"gen": gen, "best": float(scores.max()), "mean": float(scores.mean()),
                                 "center": float(center), "sps": steps / max(dt, 1e-9), "seconds": dt})
            if scores[i] > self.best_score:
                self.best_score = float(scores[i])
                self.best_params = self.params_from_genes(genomes[i])
                self.best_result = results[i]
                improved = True
        if improved:
            self._save()
        (self.run_dir / "history.json").write_text(json.dumps(self.history))

    def _save(self) -> None:
        r = dict(self.best_result or {})
        doc = {"name": self.cfg.name, "algo": self.cfg.algo, "score": self.best_score,
               "generation": len(self.history), "genes": self.genes,
               "gene_values": {g: getattr(self.best_params, g) for g in self.genes},
               "params": self.best_params.to_dict(), "mass": r.get("mass"), "mode": r.get("mode"),
               "metrics": r.get("metrics"), "config": asdict(self.cfg)}
        self.design_path.write_text(json.dumps(doc, indent=2, default=float))

    # ---- main loop ---------------------------------------------------------------------------------
    def run(self) -> None:
        cfg = self.cfg
        self.run_dir.mkdir(parents=True, exist_ok=True)
        x0 = self.genes_from_params(self.base)
        pool = ProcessPoolExecutor(max_workers=cfg.workers) if cfg.workers > 1 else None
        try:
            if cfg.algo == "bo":
                self._run_bo(pool, x0)
            elif cfg.algo == "cmaes":
                self._run_cmaes(pool, x0)
            else:
                raise ValueError(f"unknown algo {cfg.algo!r} (use 'cmaes' or 'bo')")
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)

    def _run_cmaes(self, pool, x0: np.ndarray) -> None:
        es = CMAES(x0, self.cfg.sigma0, popsize=self.cfg.population, seed=self.cfg.seed)
        for gen in range(1, self.cfg.generations + 1):
            if self._stop.is_set():
                return
            t0 = time.perf_counter()
            X = es.ask()
            Xc = np.clip(X, 0, 1)
            genomes = np.vstack([Xc, np.clip(es.mean, 0, 1)[None]])  # last row: the distribution mean
            results = self._evaluate(pool, genomes)
            if results is None:
                return
            scores = np.array([r["score"] for r in results])
            penalty = 10.0 * ((X - Xc) ** 2).sum(axis=1)
            es.tell(X, -scores[:-1] + penalty)
            self._record(gen, genomes, results, float(scores[-1]), t0)

    def _run_bo(self, pool, x0: np.ndarray) -> None:
        bo = BayesOpt(len(self.genes), x0, seed=self.cfg.seed)
        for gen in range(1, self.cfg.generations + 1):
            if self._stop.is_set():
                return
            t0 = time.perf_counter()
            X = bo.ask(self.cfg.population)
            results = self._evaluate(pool, X)
            if results is None:
                return
            scores = np.array([r["score"] for r in results])
            bo.tell(X, scores)
            self._record(gen, X, results, float(bo.y.max()), t0)
