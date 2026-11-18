"""Evolutionary training of walking controllers (and optionally the body).

Two "generation" algorithms over one flat genome = [policy weights | body genes]:

  es  evolution strategy (Salimans et al. 2017): antithetic Gaussian perturbations of a
      mean genome, centered-rank fitness shaping, Adam update.
  ga  Genetic algorithm: truncation selection, Gaussian mutation, elitism.

A whole generation is evaluated at once by `ParallelEvaluator` (one episode
per candidate, all stepping in parallel threads). With `evolve_body` on, every
candidate also carries body genes, gets its own compiled MuJoCo model, and
morphology and controller are optimised together.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import mujoco
import numpy as np

from ..models import HumanoidParams, PARAM_SPEC, MORPH_GENES, build_mjcf
from ..sim.parallel import ParallelEvaluator
from ..sim.task import LocoTask, TaskConfig
from .policy import MLPSpec, Policy, RunningNorm


@dataclass
class TrainConfig:
    algo: str = "es"               # "es" | "ga"
    population: int = 256
    generations: int = 200
    episode_seconds: float = 8.0
    hidden: list[int] = field(default_factory=lambda: [64, 64])
    sigma: float = 0.01            # policy-weight perturbation std
    lr: float = 0.005              # ES step size (Adam)
    weight_decay: float = 0.002
    elite_frac: float = 0.15       # GA: fraction kept as parents
    evolve_body: bool = False
    body_sigma: float = 0.03       # body-gene perturbation std (fraction of the allowed range)
    heading_range: float = 0.0     # rad; >0 trains steering towards random headings
    threads: int = 0               # 0 = all logical cores
    seed: int = 0
    name: str = "run"

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _centered_ranks(x: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(x))
    ranks[np.argsort(x)] = np.arange(len(x))
    return ranks / (len(x) - 1) - 0.5


class Trainer:
    """Runs in a background thread; `status()` is safe to call from anywhere."""

    def __init__(self, cfg: TrainConfig, base: HumanoidParams, task_cfg: TaskConfig | None = None,
                 model_xml: str | None = None, runs_dir: str | Path = "runs", init_policy: Policy | None = None):
        self.cfg, self.base = cfg, base
        self.task_cfg = task_cfg or TaskConfig()
        # a custom (imported) model cannot be re-generated, so its body is fixed
        self.model_xml = model_xml
        if model_xml is not None:
            cfg.evolve_body = False
        self.run_dir = Path(runs_dir) / cfg.name
        self.init_policy = init_policy
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.history: list[dict] = []
        self.state = "idle"
        self.error = ""
        self.best_fitness = -np.inf
        self.total_steps = 0
        self.policy_path = self.run_dir / "policy.npz"

    # ---- body genes <-> HumanoidParams -------------------------------------
    def _body_from_genes(self, genes: np.ndarray) -> HumanoidParams:
        d = self.base.to_dict()
        for name, g in zip(MORPH_GENES, np.clip(genes, 0, 1)):
            lo, hi = PARAM_SPEC[name][:2]
            d[name] = lo + float(g) * (hi - lo)
        return HumanoidParams.from_dict(d)

    def _genes_from_body(self, p: HumanoidParams) -> np.ndarray:
        return np.array([(getattr(p, n) - PARAM_SPEC[n][0]) / (PARAM_SPEC[n][1] - PARAM_SPEC[n][0])
                         for n in MORPH_GENES])

    # ---- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_safe, name="trainer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self) -> None:
        if self._thread:
            self._thread.join()

    def status(self) -> dict:
        with self._lock:
            return {
                "state": self.state, "error": self.error, "name": self.cfg.name,
                "generation": len(self.history), "generations": self.cfg.generations,
                "best": None if not np.isfinite(self.best_fitness) else float(self.best_fitness),
                "total_steps": int(self.total_steps),
                "history": self.history[-400:],
                "policy": str(self.policy_path) if self.policy_path.exists() else None,
            }

    def _run_safe(self) -> None:
        try:
            self.state = "running"
            self.run()
            self.state = "stopped" if self._stop.is_set() else "finished"
        except Exception as e:  # surface in the GUI instead of dying silently
            import traceback
            traceback.print_exc()
            self.error, self.state = f"{type(e).__name__}: {e}", "error"

    # ---- main loop ---------------------------------------------------------
    def run(self) -> None:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        # model_xml is the path of an imported MJCF/URDF file (compiled from disk so its assets resolve)
        base_model = (mujoco.MjModel.from_xml_path(self.model_xml) if self.model_xml
                      else mujoco.MjModel.from_xml_string(build_mjcf(self.base)))
        task = LocoTask(base_model, self.task_cfg)
        spec = MLPSpec([task.obs_dim, *cfg.hidden, task.act_dim])
        norm = RunningNorm(task.obs_dim)
        n_pol = spec.n_params
        n_body = len(MORPH_GENES) if cfg.evolve_body else 0
        # body genes live in "sigma units" so one isotropic perturbation serves both parts
        body_scale = cfg.body_sigma / cfg.sigma

        if self.init_policy is not None and self.init_policy.spec.sizes == spec.sizes:
            theta_pol = self.init_policy.params.astype(np.float64)
            norm.mean, norm.var = self.init_policy.obs_mean.copy(), (self.init_policy.obs_std - 1e-3) ** 2
            norm.n = 1e6
        else:
            theta_pol = spec.init(rng).astype(np.float64)
        theta = np.concatenate([theta_pol, self._genes_from_body(self.base) / body_scale]) if n_body else theta_pol
        n = len(theta)
        pop = max(4, cfg.population // 2 * 2)
        steps = max(1, int(round(cfg.episode_seconds / task.dt)))
        m_adam, v_adam = np.zeros(n), np.zeros(n)
        ga_pop = theta + cfg.sigma * rng.standard_normal((pop, n))
        ga_pop[0] = theta

        def models_for(genomes: np.ndarray):
            if not n_body:
                return base_model
            return [mujoco.MjModel.from_xml_string(build_mjcf(self._body_from_genes(g[n_pol:] * body_scale)))
                    for g in genomes]

        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(
            {"train": asdict(cfg), "task": self.task_cfg.to_dict(), "humanoid": self.base.to_dict()}, indent=2))

        with ParallelEvaluator(cfg.threads or None) as ev:
            if norm.n < 1:  # warm up the observation normaliser with the initial policy
                _, _, obs0 = ev.evaluate(base_model, task, spec, np.tile(theta[:n_pol].astype(np.float32), (64, 1)),
                                         steps, norm, rng, heading_range=cfg.heading_range, init_noise=0.1)
                norm.update(obs0)
            for gen in range(cfg.generations):
                if self._stop.is_set():
                    break
                t0 = time.perf_counter()
                if cfg.algo == "ga":
                    genomes = ga_pop
                else:
                    eps = rng.standard_normal((pop // 2, n))
                    # last row is the unperturbed mean, to track real progress
                    genomes = np.concatenate([theta + cfg.sigma * eps, theta - cfg.sigma * eps, theta[None]])
                fit, sim_steps, obs = ev.evaluate(
                    models_for(genomes), task, spec, genomes[:, :n_pol].astype(np.float32), steps, norm, rng,
                    heading_range=cfg.heading_range, should_stop=self._stop.is_set)
                if self._stop.is_set():
                    break
                norm_used = (norm.mean.copy(), norm.std.copy())
                norm.update(obs)

                if cfg.algo == "ga":
                    order = np.argsort(-fit)
                    champion, champ_fit = genomes[order[0]].copy(), float(fit[order[0]])
                    parents = genomes[order[:max(2, int(pop * cfg.elite_frac))]]
                    children = parents[rng.integers(0, len(parents), pop)] + cfg.sigma * rng.standard_normal((pop, n))
                    children[0] = champion  # elitism
                    ga_pop = children
                    center_fit = champ_fit
                else:
                    center_fit = float(fit[-1])
                    ranks = _centered_ranks(fit[:pop])
                    grad = (ranks[:pop // 2] - ranks[pop // 2:]) @ eps / (pop * cfg.sigma)
                    grad -= cfg.weight_decay * theta
                    m_adam = 0.9 * m_adam + 0.1 * grad
                    v_adam = 0.999 * v_adam + 0.001 * grad * grad
                    theta = theta + cfg.lr * (m_adam / (1 - 0.9 ** (gen + 1))) / (
                        np.sqrt(v_adam / (1 - 0.999 ** (gen + 1))) + 1e-8)
                    if n_body:
                        theta[n_pol:] = np.clip(theta[n_pol:], 0, 1 / body_scale)
                    best_i = int(np.argmax(fit))
                    champion, champ_fit = genomes[best_i].copy(), float(fit[best_i])
                    if center_fit >= champ_fit * 0.9:  # prefer the (more robust) mean when it is close
                        champion, champ_fit = genomes[-1].copy(), center_fit

                dt = time.perf_counter() - t0
                with self._lock:
                    self.total_steps += sim_steps
                    self.history.append({
                        "gen": gen + 1, "best": float(fit.max()), "mean": float(fit.mean()),
                        "center": center_fit, "sps": sim_steps / dt, "seconds": dt,
                    })
                    improved = champ_fit > self.best_fitness
                    if improved:
                        self.best_fitness = champ_fit
                if improved or gen % 10 == 9:
                    body = self._body_from_genes(champion[n_pol:] * body_scale) if n_body else self.base
                    Policy(spec, champion[:n_pol].astype(np.float32), norm_used[0], norm_used[1], {
                        "humanoid": body.to_dict(), "task": self.task_cfg.to_dict(), "fitness": champ_fit,
                        "generation": gen + 1, "algo": cfg.algo, "heading_range": cfg.heading_range, "custom_model": self.model_xml is not None,
                    }).save(self.policy_path if improved else self.run_dir / "latest.npz")
                    (self.run_dir / "history.json").write_text(json.dumps(self.history))
