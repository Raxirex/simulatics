"""PPO velocity-command tracking and teacher-student distillation on CPU, without PyTorch.

Methods and where they come from
--------------------------------
* **PPO** (Schulman et al. 2017, arXiv 1707.06347) with the settings common in the
  legged-locomotion literature (Rudin et al. 2021):
  - generalised advantage estimation (GAE, Schulman et al. 2016), gamma 0.99, lambda 0.95;
  - clipped surrogate objective (clip 0.2), clipped value loss, entropy bonus 0.01;
  - 5 learning epochs x 4 minibatches per iteration of 24 steps per environment;
  - adaptive learning rate: lr / 1.5 when KL(old || new) > 2 * 0.01, lr * 1.5 when KL < 0.01 / 2;
  - global gradient-norm clipping at 1.0, state-independent Gaussian log-std, empirical observation
    normalisation, time-out bootstrapping r += gamma * V(s) on truncated episodes (the usual approximation);
  - asymmetric actor-critic (Pinto et al. 2018): the critic sees privileged simulator state.
* **Fine-tuning a distilled student with PPO**: the actor starts from the student (architecture and
  observation layout are taken from the checkpoint), the critic is warmed up with the actor frozen, and a
  behaviour-cloning term towards the frozen student, annealed over training, keeps the gait while PPO
  improves it (the "PPO + lambda * KL(student || teacher)" schedule of Radosavovic et al., Science
  Robotics 2024). An optional mirror-symmetry loss |mu(s) - M_a(mu(M_s(s)))|^2 (Yu, Turk & Liu, SIGGRAPH
  2018) keeps left and right steps alike.
* **Teacher-student distillation**: behaviour cloning followed by **DAgger** (Ross, Gordon, Bagnell 2011):
  roll out a mixture of teacher and student (P(teacher) = beta, decaying per round), label every visited
  state with the teacher's action, aggregate and refit. This is how Lee et al. 2020, Kumar et al. 2021
  and many later humanoid papers obtain deployable students; here the teacher can be any `Controller` (e.g. the model-based WalkController) and the student
  is the same MLP policy PPO trains, so it can be fine-tuned with PPO afterwards. For a footstep walker the
  student observes the walker's step schedule as a gait clock (VelTaskConfig.clock = "steps"), so the
  walker's hidden timing is an input rather than something to guess, and runs at 100 Hz.

Public pieces
-------------
PPOConfig, PPO (the algorithm), run_ppo (generic loop over any vectorised env), PPOTrainer (background
thread with the same status shape as train/evolve.py's Trainer, for the GUI), RLPolicy (the deployable
artifact, runs/<name>/rl_policy.npz), evaluate_policy, evaluate_goals, evaluate_commands, StandingTeacher,
WalkerTeacherFactory, walker_teacher_factory, walker_task_config, DistillConfig, distill, and RLController
(the live-session Controller, name "rl").
"""
from __future__ import annotations

import contextlib
import copy
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from ..control.base import base_pose
from ..control.follower import PathFollower
from ..sim.vec_env import VecEnv
from ..sim.vel_task import VelocityTask, VelTaskConfig, STATE_SPEC, make_variants
from .nn import Adam, GaussianPolicy, RunningMeanStd, ValueNet, clip_grad_norm, load_npz, save_npz

POLICY_FORMAT = "simulatics-rl-policy"


# The BLAS thread count is process-wide, but several users can overlap in the GUI server (a PPOTrainer
# thread plus an evaluate_policy / bench / distill / second trainer). threadpool_limits contexts from
# different threads exit out of order and each restores what it saw on entry, which left the process at
# 1-4 BLAS threads after concurrent runs (measured: 3 PPO threads -> 3 or 4 instead of 16). And the
# Windows OpenBLAS build ignores or races set_num_threads calls from threads other than the one that set
# it first, so a cross-thread "restore" cannot be relied on either. Hence one thread at a time owns the
# setting (nesting inside that thread is fine, `with` keeps it LIFO); other threads' requests are no-ops
# and run with whatever is set.
_BLAS_LOCK = threading.Lock()
_BLAS_OWNER: list = [None, 0]  # [thread ident that holds the limit, nesting depth]


@contextlib.contextmanager
def blas_threads(n: int | None):
    """Limit numpy's BLAS thread pool (threadpoolctl, when installed) for the calling thread's block; a no-op
    when threadpoolctl is missing or another thread currently holds a limit."""
    if not n:
        yield
        return
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    me = threading.get_ident()
    with _BLAS_LOCK:
        mine = _BLAS_OWNER[0] in (None, me)
        if mine:
            _BLAS_OWNER[0] = me
            _BLAS_OWNER[1] += 1
    if not mine:
        yield
        return
    try:
        with threadpool_limits(limits=int(n), user_api="blas"):
            yield
    finally:
        with _BLAS_LOCK:
            _BLAS_OWNER[1] -= 1
            if _BLAS_OWNER[1] == 0:
                _BLAS_OWNER[0] = None


def _finite(x, default=0.0):
    """JSON-safe number: NaN / inf (e.g. statistics of an iteration without data) become `default`."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    return v if np.isfinite(v) else default


def _clean_row(row: dict) -> dict:
    return {k: (_finite(v) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}


# =========================================================================================================
# PPO
# =========================================================================================================
@dataclass
class PPOConfig:
    num_envs: int = 256
    steps_per_env: int = 24          # rollout length per iteration (Rudin et al. 2021: 24)
    iterations: int = 1500
    learning_epochs: int = 5
    num_minibatches: int = 4
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    entropy_coef: float = 0.01
    learning_rate: float = 1e-3
    schedule: str = "adaptive"       # "adaptive" (KL-based) | "fixed"
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    init_noise_std: float = 0.5      # 0.8 is common for humanoids; this lighter-footed robot falls in ~1.4 s
                                     # under 0.8 * 0.25 rad noise vs ~1.8 s at 0.5 (measured, 64 envs)
    actor_hidden: list[int] = field(default_factory=lambda: [256, 128, 64])
    critic_hidden: list[int] = field(default_factory=lambda: [256, 128, 64])
    activation: str = "elu"
    normalize_obs: bool = True
    threads: int = 8                 # MuJoCo rollout threads
    blas_threads: int = 4            # numpy BLAS threads during the update
    collect_blas_threads: int = 1    # during rollouts: small matmuls; more BLAS threads fight MuJoCo's pool
                                     # (measured 7.6k vs 26.9k env-steps/s, 8 rollout threads, 256 envs)
    seed: int = 0
    name: str = "ppo"
    max_seconds: float = 0.0         # wall-clock budget (0 = no limit)
    save_interval: int = 10
    # ---- fine-tuning from a checkpoint (a distilled student or an earlier PPO run) --------------------
    critic_warmup_iters: int = 0     # iterations that train only the critic (the actor is frozen)
    bc_coef: float = 0.0             # weight of ||mu(o) - mu_ref(o)||^2 towards the initial actor ...
    bc_coef_final: float = 0.0       # ... annealed linearly to this ...
    bc_anneal_iters: int = 0         # ... over these iterations (after the warm-up)
    sym_coef: float = 0.0            # mirror-symmetry loss weight (needs the env's task mirror maps)
    reset_std: float = 0.0           # > 0: exploration std after loading a checkpoint
    freeze_normalizer: bool = False  # keep the loaded observation normaliser fixed (its statistics define the
                                     # student's input scaling; drifting them changes what the student sees)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PPOConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RolloutStorage:
    """(T, N, ...) buffers for one PPO iteration, plus GAE."""

    def __init__(self, t: int, n: int, obs_dim: int, cobs_dim: int, act_dim: int):
        self.obs = np.zeros((t, n, obs_dim))
        self.cobs = np.zeros((t, n, cobs_dim))
        self.actions = np.zeros((t, n, act_dim))
        self.logp = np.zeros((t, n))
        self.mu = np.zeros((t, n, act_dim))
        self.values = np.zeros((t, n))
        self.rewards = np.zeros((t, n))
        self.dones = np.zeros((t, n))
        self.returns = np.zeros((t, n))
        self.advantages = np.zeros((t, n))
        self.step = 0
        self.t, self.n = t, n

    def compute_returns(self, last_values: np.ndarray, gamma: float, lam: float) -> None:
        adv = np.zeros(self.n)
        for k in reversed(range(self.t)):
            next_v = last_values if k == self.t - 1 else self.values[k + 1]
            not_done = 1.0 - self.dones[k]
            delta = self.rewards[k] + not_done * gamma * next_v - self.values[k]
            adv = delta + not_done * gamma * lam * adv
            self.returns[k] = adv + self.values[k]
        a = self.returns - self.values
        self.advantages = (a - a.mean()) / (a.std() + 1e-8)

    def minibatches(self, num: int, rng: np.random.Generator):
        size = self.t * self.n
        flat = lambda x: x.reshape(size, *x.shape[2:])  # noqa: E731
        data = [flat(x) for x in (self.obs, self.cobs, self.actions, self.logp, self.mu, self.values,
                                  self.returns, self.advantages)]
        perm = rng.permutation(size)
        num = max(1, min(int(num), size))  # never an empty minibatch (tiny configs: means of nothing are NaN)
        mb = size // num
        for i in range(num):
            idx = perm[i * mb:(i + 1) * mb]
            yield [x[idx] for x in data]


class PPO:
    """The PPO learner (actor, critic, normalisers, optimiser); environment-agnostic."""

    def __init__(self, obs_dim: int, critic_obs_dim: int, act_dim: int, cfg: PPOConfig,
                 rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.seed)
        self.actor = GaussianPolicy(obs_dim, act_dim, cfg.actor_hidden, cfg.activation, cfg.init_noise_std, self.rng)
        self.critic = ValueNet(critic_obs_dim, cfg.critic_hidden, cfg.activation, self.rng)
        self.obs_norm = RunningMeanStd(obs_dim)
        self.cobs_norm = RunningMeanStd(critic_obs_dim)
        self.opt = Adam(self.actor.params + self.critic.params, cfg.learning_rate)
        self.lr = cfg.learning_rate
        self.storage: RolloutStorage | None = None
        self.iteration = 0                      # number of update() calls so far
        self.ref_actor: GaussianPolicy | None = None   # frozen copy for the behaviour-cloning regulariser
        self.mirror: tuple[Callable, Callable] | None = None  # (raw obs -> mirrored, actions -> mirrored)
        self.freeze_obs_norm = bool(cfg.freeze_normalizer)

    # ---- data collection -----------------------------------------------------------------------------
    def init_storage(self, n: int) -> None:
        self.storage = RolloutStorage(self.cfg.steps_per_env, n, self.actor.obs_dim, self.cobs_norm.mean.size,
                                      self.actor.act_dim)

    def _norm(self, obs, cobs, update: bool):
        if not self.cfg.normalize_obs:
            return obs, cobs
        if update:
            if not self.freeze_obs_norm:
                self.obs_norm.update(obs)
            self.cobs_norm.update(cobs)
        return self.obs_norm.normalize(obs), self.cobs_norm.normalize(cobs)

    def act(self, obs: np.ndarray, cobs: np.ndarray) -> np.ndarray:
        s = self.storage
        o, c = self._norm(obs, cobs, update=True)
        a, logp, mu = self.actor.sample(o, self.rng)
        k = s.step
        s.obs[k], s.cobs[k], s.actions[k], s.logp[k], s.mu[k] = o, c, a, logp, mu
        s.values[k] = self.critic(c)
        return a

    def process_env_step(self, rewards: np.ndarray, dones: np.ndarray, time_outs: np.ndarray | None) -> None:
        s = self.storage
        r = rewards.astype(np.float64).copy()
        if time_outs is not None:  # bootstrap truncated episodes with V(s_t)
            r += self.cfg.gamma * s.values[s.step] * time_outs
        s.rewards[s.step], s.dones[s.step] = r, dones
        s.step += 1

    def compute_returns(self, last_cobs: np.ndarray) -> None:
        c = self.cobs_norm.normalize(last_cobs) if self.cfg.normalize_obs else last_cobs
        self.storage.compute_returns(self.critic(c), self.cfg.gamma, self.cfg.lam)

    # ---- regularisers --------------------------------------------------------------------------------
    def set_reference(self, actor: GaussianPolicy | None = None) -> None:
        """Freeze a copy of `actor` (default: the current actor) as the behaviour-cloning reference."""
        src = actor or self.actor
        self.ref_actor = GaussianPolicy.from_arrays(copy.deepcopy(src.to_arrays("a_")), self.cfg.activation, "a_")

    def bc_weight(self) -> float:
        cfg = self.cfg
        if cfg.bc_coef <= 0 or self.ref_actor is None:
            return 0.0
        k = max(0, self.iteration - cfg.critic_warmup_iters)
        if cfg.bc_anneal_iters <= 0:
            return float(cfg.bc_coef)
        f = min(1.0, k / cfg.bc_anneal_iters)
        return float(cfg.bc_coef + (cfg.bc_coef_final - cfg.bc_coef) * f)

    def _mirror_targets(self, obs_n: np.ndarray) -> np.ndarray:
        """M_a(mu(M_s(s))) for normalised observations (de-normalise, mirror, re-normalise)."""
        fo, fa = self.mirror
        if self.cfg.normalize_obs:
            raw = obs_n * self.obs_norm.std + self.obs_norm.mean
            m = self.obs_norm.normalize(fo(raw))
        else:
            m = fo(obs_n)
        return fa(self.actor.mean(m))

    # ---- learning ------------------------------------------------------------------------------------
    def update(self) -> dict:
        cfg, s = self.cfg, self.storage
        stats = {"surrogate": 0.0, "value_loss": 0.0, "kl": 0.0, "clip_frac": 0.0, "grad_norm": 0.0,
                 "bc_loss": 0.0, "sym_loss": 0.0}
        n_updates = 0
        actor_on = self.iteration >= cfg.critic_warmup_iters
        bc = self.bc_weight() if actor_on else 0.0
        sym = cfg.sym_coef if (actor_on and self.mirror is not None) else 0.0
        std_old_all = np.exp(self.actor.log_std).copy()  # log_std is state-independent: same for every sample
        for _ in range(cfg.learning_epochs):
            for obs, cobs, act, logp_old, mu_old, v_old, ret, adv in s.minibatches(cfg.num_minibatches, self.rng):
                surr, g_actor, info = self.actor.ppo_loss(obs, act, logp_old, adv, cfg.clip_param, cfg.entropy_coef)
                if bc > 0:
                    l_bc, g_bc = self.actor.bc_loss(obs, self.ref_actor.mean(obs))
                    g_actor = [g + bc * h for g, h in zip(g_actor, g_bc)]
                    stats["bc_loss"] += l_bc
                if sym > 0:
                    l_sym, g_sym = self.actor.bc_loss(obs, self._mirror_targets(obs))
                    g_actor = [g + sym * h for g, h in zip(g_actor, g_sym)]
                    stats["sym_loss"] += l_sym
                if not actor_on:
                    g_actor = [np.zeros_like(g) for g in g_actor]
                kl = float(np.mean(GaussianPolicy.kl(mu_old, std_old_all, info["mu"], self.actor.std)))
                if actor_on and cfg.schedule == "adaptive" and cfg.desired_kl > 0 and np.isfinite(kl):
                    if kl > 2.0 * cfg.desired_kl:
                        self.lr = max(1e-5, self.lr / 1.5)
                    elif 0.0 < kl < cfg.desired_kl / 2.0:
                        self.lr = min(1e-2, self.lr * 1.5)
                    self.opt.lr = self.lr
                vloss, g_critic = self.critic.loss(cobs, ret, v_old,
                                                   cfg.clip_param if cfg.use_clipped_value_loss else None)
                grads = g_actor + [g * cfg.value_loss_coef for g in g_critic]
                grads, gnorm = clip_grad_norm(grads, cfg.max_grad_norm)
                if not all(np.isfinite(g).all() for g in grads):
                    continue  # never let one bad minibatch write NaN into the networks
                self.opt.step(grads)
                np.clip(self.actor.log_std, np.log(0.01), np.log(5.0), out=self.actor.log_std)
                stats["surrogate"] += surr
                stats["value_loss"] += vloss
                stats["kl"] += kl
                stats["clip_frac"] += info["clip_frac"]
                stats["grad_norm"] += gnorm
                n_updates += 1
        s.step = 0
        self.iteration += 1
        out = {k: _finite(v / max(1, n_updates)) for k, v in stats.items()}
        out.update(lr=self.lr, std=float(self.actor.std.mean()), bc_coef=bc, actor_frozen=not actor_on)
        return out

    # ---- export --------------------------------------------------------------------------------------
    def policy(self, meta: dict) -> "RLPolicy":
        norm = self.obs_norm if self.cfg.normalize_obs else None
        return RLPolicy(self.actor, norm, {**meta, "activation": self.cfg.activation})

    def checkpoint_arrays(self) -> dict[str, np.ndarray]:
        return {**self.actor.to_arrays("actor_"), **self.critic.net.to_arrays("critic_"),
                **self.obs_norm.to_arrays("obs_"), **self.cobs_norm.to_arrays("cobs_")}

    def load_checkpoint(self, arrays) -> None:
        """Load an rl_checkpoint.npz or an rl_policy.npz (actor + observation normaliser; e.g. a distilled
        student). The actor architecture is taken from the file (cfg.actor_hidden is updated to match), so a
        student distilled with different hidden sizes can be fine-tuned; the observation and action sizes
        must match the environment. The critic is loaded when its shape matches, else it stays fresh."""
        act = self.cfg.activation
        actor = GaussianPolicy.from_arrays(arrays, act)
        if actor.obs_dim != self.actor.obs_dim or actor.act_dim != self.actor.act_dim:
            raise ValueError(f"checkpoint actor maps {actor.obs_dim} -> {actor.act_dim}, the environment needs "
                             f"{self.actor.obs_dim} -> {self.actor.act_dim}")
        if actor.mu.sizes != self.actor.mu.sizes:
            self.cfg.actor_hidden = list(actor.mu.sizes[1:-1])
        actor.mu.params = [np.ascontiguousarray(p, np.float32) for p in actor.mu.params]
        self.actor = actor
        if "critic_sizes" in arrays:
            from .nn import MLP
            critic = MLP.from_arrays(arrays, "critic_", act)
            if critic.sizes == self.critic.net.sizes:
                for p, q in zip(self.critic.params, critic.params):
                    p[...] = q
        if "obs_mean" in arrays:
            self.obs_norm = RunningMeanStd.from_arrays(arrays, "obs_")
        if "cobs_mean" in arrays and len(arrays["cobs_mean"]) == self.cobs_norm.mean.size:
            self.cobs_norm = RunningMeanStd.from_arrays(arrays, "cobs_")
        self.opt = Adam(self.actor.params + self.critic.params, self.lr)


def _merge_episode_stats(events: list[dict]) -> dict:
    """Count-weighted mean of the per-step `info["episode"]` dicts of one iteration."""
    if not events:
        return {}
    total = sum(e.get("count", 1) for e in events)
    keys = {k for e in events for k in e if k != "count"}
    return {k: sum(e.get(k, 0.0) * e.get("count", 1) for e in events) / total for k in keys} | {"count": total}


def run_ppo(env, cfg: PPOConfig, iterations: int | None = None, algo: PPO | None = None,
            callback: Callable[[int, dict, PPO], None] | None = None,
            should_stop: Callable[[], bool] = lambda: False) -> tuple[PPO, list[dict]]:
    """Generic PPO loop over a vectorised env (reset() -> (obs, cobs); step(a) -> (obs, cobs, r, done, info)).

    Returns the learner and a per-iteration log with episode statistics, losses and throughput.
    """
    algo = algo or PPO(env.obs_dim, env.critic_obs_dim, env.act_dim, cfg)
    algo.init_storage(env.num_envs)
    obs, cobs = env.reset()
    log: list[dict] = []
    recent: deque[float] = deque(maxlen=100)
    n_iter = cfg.iterations if iterations is None else iterations
    t_start = time.perf_counter()
    substeps = getattr(getattr(env, "cfg", None), "decimation", 1)
    for it in range(n_iter):
        if should_stop() or (cfg.max_seconds and time.perf_counter() - t_start > cfg.max_seconds):
            break
        t0 = time.perf_counter()
        events, step_rew = [], 0.0
        with blas_threads(cfg.collect_blas_threads):
            for _ in range(cfg.steps_per_env):
                a = algo.act(obs, cobs)
                obs, cobs, r, done, info = env.step(a)
                algo.process_env_step(r, done.astype(float), info.get("time_outs"))
                step_rew += float(np.mean(r))
                if "episode" in info:
                    events.append(info["episode"])
                    recent.extend([info["episode"]["return"]] * info["episode"].get("count", 1))
            algo.compute_returns(cobs)
        t1 = time.perf_counter()
        with blas_threads(cfg.blas_threads):
            stats = algo.update()
        t2 = time.perf_counter()
        ep = _merge_episode_stats(events)
        steps = cfg.steps_per_env * env.num_envs
        entry = {"iter": it + 1, "step_reward": step_rew / cfg.steps_per_env,
                 "mean_return": float(np.mean(recent)) if recent else None,
                 "env_sps": steps / (t2 - t0), "sps": steps * substeps / (t2 - t0),
                 "collect_s": t1 - t0, "learn_s": t2 - t1, "episodes": ep, **stats}
        log.append(entry)
        if callback is not None:
            callback(it, entry, algo)
    return algo, log


# =========================================================================================================
# Deployable policy
# =========================================================================================================
class RLPolicy:
    """A trained actor + its observation normaliser + metadata (task config, robot, joint names).

    `act(obs)` returns the deterministic (mean) action, which is what runs on the robot.
    """

    def __init__(self, actor: GaussianPolicy, obs_norm: RunningMeanStd | None, meta: dict):
        self.actor, self.obs_norm, self.meta = actor, obs_norm, dict(meta)

    @property
    def obs_dim(self) -> int:
        return self.actor.obs_dim

    @property
    def act_dim(self) -> int:
        return self.actor.act_dim

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(obs)
        if self.obs_norm is not None:
            x = self.obs_norm.normalize(x)
        return self.actor.mean(x)

    def save(self, path: str | Path) -> Path:
        arrays = self.actor.to_arrays("actor_")
        if self.obs_norm is not None:
            arrays.update(self.obs_norm.to_arrays("obs_"))
        meta = {"format": POLICY_FORMAT, "version": 1, **self.meta}
        return save_npz(path, arrays, meta)

    @classmethod
    def load(cls, path: str | Path) -> "RLPolicy":
        arrays, meta = load_npz(path)
        if meta.get("format") != POLICY_FORMAT:
            raise ValueError(f"{path} is not a Simulatics RL policy (format {meta.get('format')!r})")
        actor = GaussianPolicy.from_arrays(arrays, meta.get("activation", "elu"))
        norm = RunningMeanStd.from_arrays(arrays, "obs_") if "obs_mean" in arrays else None
        return cls(actor, norm, meta)

    def task_config(self) -> VelTaskConfig:
        return VelTaskConfig.from_dict(self.meta.get("task", {}))


def is_rl_policy(path: str | Path) -> bool:
    """True when `path` is an rl_policy.npz written by this module (cheap check for the GUI's policy list)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return "meta" in z.files and json.loads(str(z["meta"])).get("format") == POLICY_FORMAT
    except Exception:
        return False


def _policy_meta(task: VelocityTask, model_meta: dict, algo: str, **extra) -> dict:
    return {"algo": algo, "task": task.cfg.to_dict(), "joint_names": task.joint_names,
            "actuator_names": task.actuator_names, "obs_dim": task.obs_dim, "act_dim": task.act_dim,
            "dt": task.dt, "z0": task.z0, **model_meta, **extra}


def evaluate_policy(policy: RLPolicy, model: mujoco.MjModel, command=(0.0, 0.0, 0.0), seconds: float = 5.0,
                    num_envs: int = 8, nthread: int = 4, seed: int = 0) -> dict:
    """Deterministic rollouts of the mean action under a fixed command on the nominal model.

    Returns {"fall_rate", "lin_vel_error" (m/s RMS, body frame xy), "yaw_rate_error", "mean_vx",
    "height_end"}: the numbers to look at when deciding whether a policy walks.
    """
    cfg = policy.task_config()
    cfg.randomize, cfg.obs_noise, cfg.zero_cmd_prob = False, 0.0, 0.0
    cfg.resample_seconds = cfg.episode_seconds = 1e9
    with VecEnv(model, cfg, num_envs, nthread, seed) as env, blas_threads(1):
        env.reset()
        env.ts.commands[:] = np.asarray(command, float)
        env.ts.history[:] = 0.0  # reset() already pushed a frame with its random command into the history
        env.ts.walk_t[:] = -1.0  # ... and may have started the step clock with it
        obs, _ = env._observe()
        steps = max(1, int(round(seconds / env.dt)))
        alive = np.ones(num_envs, bool)
        err_v, err_w, vx = [], [], []
        for _ in range(steps):
            obs, _, _, done, info = env.step(policy.act(obs))
            alive &= ~info["terminated"]
            qpos, qvel = env.task.split(env.state)
            _, lin, ang, _ = VelocityTask.base_frame(qpos, qvel)
            m = alive
            if m.any():
                err_v.append(np.sum((lin[m, :2] - np.asarray(command[:2])) ** 2, axis=1))
                err_w.append((ang[m, 2] - command[2]) ** 2)
                vx.append(lin[m, 0])
        qpos, _ = env.task.split(env.state)
        cat = lambda xs: np.concatenate(xs) if xs else np.array([np.nan])  # noqa: E731
        return {"fall_rate": float(1.0 - alive.mean()), "lin_vel_error": float(np.sqrt(np.mean(cat(err_v)))),
                "yaw_rate_error": float(np.sqrt(np.mean(cat(err_w)))), "mean_vx": float(np.mean(cat(vx))),
                "height_end": float(qpos[alive, 2].mean()) if alive.any() else 0.0}


# =========================================================================================================
# Trainer (background thread, GUI-compatible status)
# =========================================================================================================
def _checkpoint_task(path: str | Path | None) -> VelTaskConfig | None:
    try:
        _, meta = load_npz(path)
    except Exception:
        return None
    return VelTaskConfig.from_dict(meta["task"]) if isinstance(meta.get("task"), dict) else None


class PPOTrainer:
    """PPO velocity-tracking training in a background thread.

    Lifecycle and `status()` match train/evolve.py's Trainer so the GUI chart works unchanged. One PPO
    iteration is reported as one "generation"; history entries carry
        gen    iteration number
        mean   mean return of the episodes that ended in this iteration (rolling mean if none ended)
        best   best return among them
        center rolling mean return over the last 100 episodes (the usual "mean reward")
        sps    physics steps per second (env steps x decimation), like the evolution trainer
    plus PPO diagnostics: track_lin / track_ang (per-second episode sums of the tracking
    terms; 1.0 and 0.5 are perfect), ep_len_s, fell, kl, lr, std, value_loss, env_sps.
    Output: runs/<name>/rl_policy.npz (latest policy), rl_checkpoint.npz (critic + normalisers too, for
    resuming), config.json, history.json.

    init_checkpoint may be an rl_checkpoint.npz or an rl_policy.npz (e.g. a distilled student). Its actor
    architecture is used, and when its observation layout differs from `task_cfg` (a student distilled from
    the footstep walker observes its step clock and a frame history) the checkpoint's task configuration
    replaces `task_cfg`, so the GUI's default VelTaskConfig() can continue any loaded policy.
    """

    def __init__(self, cfg: PPOConfig, base=None, task_cfg: VelTaskConfig | None = None,
                 model_xml: str | None = None, runs_dir: str | Path = "runs", init_checkpoint: str | Path | None = None):
        from ..models import HumanoidParams
        self.cfg = cfg
        self.base = base or HumanoidParams()
        self.task_cfg = task_cfg or VelTaskConfig()
        self.model_xml = model_xml
        self.init_checkpoint = init_checkpoint
        self.run_dir = Path(runs_dir) / cfg.name
        self.policy_path = self.run_dir / "rl_policy.npz"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.history: list[dict] = []
        self.state = "idle"
        self.error = ""
        self.best_fitness = -np.inf
        self.total_steps = 0

    # ---- lifecycle -----------------------------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_safe, name="ppo-trainer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self) -> None:
        if self._thread:
            self._thread.join()

    def status(self) -> dict:
        with self._lock:
            return {
                "state": self.state, "error": self.error, "name": self.cfg.name, "algo": "ppo",
                "generation": len(self.history), "generations": self.cfg.iterations,
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

    # ---- main loop -----------------------------------------------------------------------------------
    def build_model(self) -> mujoco.MjModel:
        from ..models import build_mjcf
        if self.model_xml:
            return mujoco.MjModel.from_xml_path(self.model_xml)
        return mujoco.MjModel.from_xml_string(build_mjcf(self.base))

    def _resolve_task(self, model: mujoco.MjModel) -> None:
        """Use the checkpoint's task configuration when the given one has a different observation layout."""
        if not self.init_checkpoint:
            return
        ck = _checkpoint_task(self.init_checkpoint)
        if ck is None:
            return
        if VelocityTask(model, ck).obs_dim != VelocityTask(model, self.task_cfg).obs_dim:
            self.task_cfg = ck

    def run(self) -> None:
        cfg = self.cfg
        model = self.build_model()
        self._resolve_task(model)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        model_meta = {"humanoid": None if self.model_xml else self.base.to_dict(),
                      "custom_model": self.model_xml is not None}
        rng = np.random.default_rng(cfg.seed)
        with VecEnv(model, self.task_cfg, cfg.num_envs, cfg.threads, cfg.seed) as env:
            algo = PPO(env.obs_dim, env.critic_obs_dim, env.act_dim, cfg, rng)
            init_meta: dict = {}
            if self.init_checkpoint:
                arrays, init_meta = load_npz(self.init_checkpoint)
                algo.load_checkpoint(arrays)
                if cfg.reset_std > 0:
                    algo.actor.log_std[:] = np.log(cfg.reset_std)
                if cfg.bc_coef > 0:
                    algo.set_reference()
            if cfg.sym_coef > 0:
                try:
                    env.task.mirror_maps()
                    algo.mirror = (env.task.mirror_obs, env.task.mirror_actions)
                except ValueError:
                    algo.mirror = None
            (self.run_dir / "config.json").write_text(json.dumps(
                {"ppo": cfg.to_dict(), "task": self.task_cfg.to_dict(),
                 "humanoid": None if self.model_xml else self.base.to_dict(), "custom_model": self.model_xml,
                 "init_checkpoint": str(self.init_checkpoint) if self.init_checkpoint else None}, indent=2))
            recent_best: deque[float] = deque(maxlen=100)
            extra = {k: init_meta[k] for k in ("teacher", "teacher_params") if k in init_meta}

            def save(it: int) -> None:
                meta = _policy_meta(env.task, model_meta, "ppo", iteration=it,
                                    mean_return=self.history[-1]["center"] if self.history else None,
                                    init_checkpoint=str(self.init_checkpoint) if self.init_checkpoint else None,
                                    **extra)
                algo.policy(meta).save(self.policy_path)
                save_npz(self.run_dir / "rl_checkpoint.npz", algo.checkpoint_arrays(), {**meta, "ppo": cfg.to_dict()})
                (self.run_dir / "history.json").write_text(json.dumps(self.history))

            def on_iter(it: int, e: dict, _algo: PPO) -> None:
                ep = e["episodes"]
                recent = list(env.completed)
                if ep:
                    recent_best.append(ep["return"])
                rolling = float(np.mean([c["return"] for c in recent])) if recent else 0.0
                terms = lambda k: float(np.mean([c["terms"].get(k, 0.0) for c in recent])) if recent else 0.0  # noqa: E731
                mean = ep.get("return", rolling)
                best = max((c["return"] for c in recent[-int(ep.get("count", 0)):]), default=mean) if ep else mean
                row = {"gen": it + 1, "best": float(best), "mean": float(mean), "center": rolling,
                       "sps": e["sps"], "env_sps": e["env_sps"], "seconds": e["collect_s"] + e["learn_s"],
                       "collect_s": e["collect_s"], "learn_s": e["learn_s"],
                       "track_lin": terms("tracking_lin_vel"), "track_ang": terms("tracking_ang_vel"),
                       "ep_len_s": float(np.mean([c["length_s"] for c in recent])) if recent else 0.0,
                       "fell": float(np.mean([c["fell"] for c in recent])) if recent else 0.0,
                       "step_reward": e["step_reward"], "kl": e["kl"], "lr": e["lr"], "std": e["std"],
                       "value_loss": e["value_loss"], "surrogate": e["surrogate"],
                       "bc_coef": e.get("bc_coef", 0.0), "bc_loss": e.get("bc_loss", 0.0),
                       "sym_loss": e.get("sym_loss", 0.0)}
                row = _clean_row(row)   # the GUI parses this as JSON, where NaN is not allowed
                with self._lock:
                    self.total_steps += cfg.steps_per_env * cfg.num_envs * self.task_cfg.decimation
                    self.history.append(row)
                    if recent and np.isfinite(rolling) and rolling > self.best_fitness:
                        self.best_fitness = rolling
                if (it + 1) % max(1, cfg.save_interval) == 0:
                    save(it + 1)

            run_ppo(env, cfg, algo=algo, callback=on_iter, should_stop=self._stop.is_set)
            save(len(self.history))


# =========================================================================================================
# Teachers
# =========================================================================================================
class StandingTeacher:
    """Hold the task's default pose, optionally with ankle-strategy balance feedback. Implements `Controller`.

    With zero gains the servo targets are constant. With k_tilt / k_rate > 0 the ankle pitch and roll targets
    add  k_tilt * tilt_error + k_rate * tilt_rate  (torso pitch/roll from the IMU's projected gravity), the
    textbook ankle strategy for an inverted pendulum; that makes the demonstrations state-dependent, which
    is what DAgger needs to have something to correct.
    """
    name = "stand"

    def __init__(self, model: mujoco.MjModel, task_cfg: VelTaskConfig | None = None, k_tilt: float = 0.0,
                 k_rate: float = 0.0):
        cfg = task_cfg or VelTaskConfig()
        task = VelocityTask(model, cfg)
        self.ctrl = task.ctrl_default.copy()
        names = task.actuator_names
        self.pitch_idx = np.array([i for i, n in enumerate(names) if n.endswith("ankle_y") or "ankle_pitch" in n], int)
        self.roll_idx = np.array([i for i, n in enumerate(names) if n.endswith("ankle_x") or "ankle_roll" in n], int)
        self.lean = float(cfg.lean)
        self.k_tilt, self.k_rate = float(k_tilt), float(k_rate)
        self.lo, self.hi = task.ctrl_lo, task.ctrl_hi
        self.control_dt = float(model.opt.timestep) * cfg.decimation

    def reset(self, model, data) -> None:
        pass

    def set_path(self, path_xy, final_yaw=None) -> None:
        pass

    def step(self, model, data) -> None:
        ctrl = self.ctrl.copy()
        if self.k_tilt or self.k_rate:
            _, _, ang, grav = VelocityTask.base_frame(data.qpos[None], data.qvel[None])
            pitch = float(np.arcsin(np.clip(grav[0, 0], -1, 1))) - self.lean  # forward lean > 0
            roll = -float(np.arcsin(np.clip(grav[0, 1], -1, 1)))
            ctrl[self.pitch_idx] += self.k_tilt * pitch + self.k_rate * ang[0, 1]
            ctrl[self.roll_idx] += self.k_tilt * roll + self.k_rate * ang[0, 0]
        data.ctrl[:] = np.clip(ctrl, self.lo, self.hi)

    def status(self) -> dict:
        return {"mode": "stand", "message": "holding the default pose"}


class WalkerTeacherFactory:
    """Picklable factory for the model-based WalkController (control/walker.py) as a distillation teacher.

    `params` are WalkParams fields (e.g. step_forward=0.32 for a 0.4 m/s top speed). Picklable so the
    distillation worker processes can build their own teachers. `source` optionally names a copy of
    walker.py (e.g. a committed version, `git show <rev>:simulatics/control/walker.py`) to load instead of
    the live module, so a long run is pinned to one teacher while control/walker.py is being worked on; it is
    imported as a private module of simulatics.control (its relative imports resolve there)."""

    def __init__(self, source: str | None = None, **params):
        self.source = str(source) if source else None
        self.params = dict(params)

    def module(self):
        if self.source is None:
            from ..control import walker
            return walker
        import hashlib
        import importlib.util
        import sys
        key = "simulatics.control._walker_pinned_" + hashlib.sha1(Path(self.source).read_bytes()).hexdigest()[:12]
        if key not in sys.modules:
            spec = importlib.util.spec_from_file_location(key, self.source)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[key] = mod
            spec.loader.exec_module(mod)
        return sys.modules[key]

    def __call__(self, model: mujoco.MjModel):
        mod = self.module()
        return mod.WalkController(model, mod.WalkParams(**self.params) if self.params else None)

    def __repr__(self) -> str:
        return f"WalkerTeacherFactory(source={self.source!r}, {self.params})"


def walker_teacher_factory(source: str | None = None, **params) -> Callable[[mujoco.MjModel], object] | None:
    """Factory for the model-based WalkController (control/walker.py, or the copy at `source`) as a teacher,
    or None if unavailable. Keyword arguments are WalkParams fields.

    Push recovery (WalkParams.recovery, when the walker has it) is off unless asked for: recovery re-times and
    inserts steps from the measured state, so the step schedule stops being a function of time and the
    student's "steps" clock could no longer follow it (measured: 24 % of ticks out of phase)."""
    factory = WalkerTeacherFactory(source, **params)
    try:
        WalkParams = factory.module().WalkParams
    except (ImportError, AttributeError, OSError):
        return None
    if "recovery" in getattr(WalkParams, "__dataclass_fields__", {}) and "recovery" not in params:
        factory.params["recovery"] = False
    return factory


def _is_footstep_walker(teacher) -> bool:
    if not hasattr(teacher, "set_velocity"):
        return False
    try:
        return "support" in teacher.status()
    except Exception:
        return False


def measure_step_timing(model: mujoco.MjModel, teacher_factory, command=(0.2, 0.0, 0.0),
                        seconds: float = 6.0) -> dict | None:
    """The step schedule of a footstep walker in velocity mode, measured from its public status()["support"]
    ("double" / "left" / "right") one physics tick at a time, so the student's "steps" clock follows the
    teacher even if its timing constants change. Returns {"clock_start_s", "clock_ss_s", "clock_ds_s",
    "first_swing"} in the VelTaskState.walk_t convention (walk_t = 0 at the observation where the command
    arrives, teacher call c = walk_t / timestep + 1), or None when the teacher reports no support phases or
    does not run at the physics rate."""
    t = teacher_factory(model)
    dt = float(model.opt.timestep)
    if abs(float(getattr(t, "control_dt", dt)) - dt) > 1e-9 or not _is_footstep_walker(t):
        return None
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    t.reset(model, d)
    t.set_velocity(*command)
    sup = []
    for _ in range(int(round(seconds / dt))):
        t.step(model, d)
        sup.append(t.status().get("support"))
        mujoco.mj_step(model, d)
    swing = np.array([s in ("left", "right") for s in sup])
    if not swing.any():
        return None
    first = int(np.argmax(swing))                         # index of the first swing call (call number first + 1)
    runs, cur, n = [], swing[first], 0
    for s in swing[first:]:
        if s == cur:
            n += 1
        else:
            runs.append((bool(cur), n))
            cur, n = s, 1
    if len(runs) < 3 or not (runs[0][0] and not runs[1][0] and runs[2][0]) or runs[0][1] != runs[2][1]:
        return None
    side = "left" if sup[first] == "right" else "right"   # support "right" = the left foot swings
    return {"clock_start_s": first * dt, "clock_ss_s": runs[0][1] * dt, "clock_ds_s": runs[1][1] * dt,
            "first_swing": side}


def walker_task_config(model: mujoco.MjModel, teacher_factory=None, base: VelTaskConfig | None = None,
                       **overrides) -> VelTaskConfig:
    """Task configuration for distilling a footstep walker (control/walker.py) into a policy.

    * 100 Hz policy (decimation 2 at the 5 ms physics step; the walker itself runs at 200 Hz, and a 50 Hz
      zero-order hold of its targets already walks, so 100 Hz represents it with margin);
    * the walker's knees-bent stance as the default pose and start pose (init_qpos);
    * the "steps" gait clock with the walker's step timing (measured, see measure_step_timing) plus 3 past
      frames (30 ms) of history, enough to estimate velocities from the proprioception;
    * commands inside what the walker can do: vx -0.2..0.4 m/s, vy +-0.15 m/s, yaw rate +-0.5 rad/s, with
      goal walking (RLController's PathFollower) limited to 0.3 m/s and 0.45 rad/s;
    * domain randomisation (friction, mass, servo stiffness, pushes) the walker survives; the base linear
      velocity as an input and no observation noise (the walker's balance feedback needs the velocity).
    """
    factory = teacher_factory or walker_teacher_factory()
    if factory is None:
        raise RuntimeError("control/walker.py WalkController is not available")
    timing = measure_step_timing(model, factory)
    if timing is None:
        raise ValueError("the teacher does not report a step schedule (status()['support'])")
    if timing["first_swing"] != "left":
        raise ValueError("the 'steps' clock assumes the walker starts with the left foot")
    t = factory(model)
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    t.reset(model, d)
    q0 = d.qpos.copy()
    q0[0:2] = 0.0
    cfg = VelTaskConfig.from_dict((base or VelTaskConfig()).to_dict())
    dt = float(model.opt.timestep)
    cfg.decimation = max(1, int(round(0.01 / dt)))
    cfg.init_qpos = q0.tolist()
    cfg.clock = "steps"
    cfg.clock_start_s, cfg.clock_ss_s, cfg.clock_ds_s = (timing["clock_start_s"], timing["clock_ss_s"],
                                                         timing["clock_ds_s"])
    cfg.history = 3
    cfg.cmd_vx, cfg.cmd_vy, cfg.cmd_wz = (-0.2, 0.4), (-0.15, 0.15), (-0.5, 0.5)
    # goal walking (PathFollower) slower and with gentler turns than the command range: measured teacher falls
    # came from pure pursuit's fast-and-turning corner (vx ~0.35 with wz 0.6) on low friction
    cfg.follow_speed, cfg.follow_turn = 0.3, 0.45
    cfg.small_cmd = 0.03
    cfg.zero_cmd_prob = 0.1
    cfg.turn_prob = 0.2
    cfg.resample_seconds = 5.0
    # No observation noise, and the base linear velocity as an input: the walker's balance feedback runs on the
    # centre-of-mass velocity; with reference-level joint-velocity noise (+-0.75 rad/s at half scale) the
    # student learned to ignore velocity and drifted over within ~1.5 s when standing on its own (measured).
    cfg.obs_noise = 0.0
    cfg.obs_lin_vel = True
    cfg.init_joint_noise = 0.02
    cfg.init_vel_noise = 0.05
    cfg.friction_range = (0.85, 1.2)      # at 0.74 the teacher itself slips and falls turning at 0.6 rad/s
    cfg.added_mass_range = (-1.5, 2.0)
    cfg.kp_scale_range = (0.9, 1.1)
    cfg.push_interval_s = 6.0
    cfg.push_vel = 0.1
    cfg.tracking_sigma = 0.05
    cfg.max_tilt_deg = 50.0
    for k, v in overrides.items():
        if k not in VelTaskConfig.__dataclass_fields__:
            raise TypeError(f"unknown VelTaskConfig field {k!r}")
        setattr(cfg, k, v)
    return VelTaskConfig.from_dict(cfg.to_dict())


# =========================================================================================================
# Teacher-student distillation (behaviour cloning + DAgger)
# =========================================================================================================
@dataclass
class DistillConfig:
    num_envs: int = 8                # environments per worker (or in total when workers = 0)
    rounds: int = 4                  # round 0 = behaviour cloning on teacher rollouts, then DAgger
    episode_seconds: float = 5.0     # simulated seconds per environment per round (episodes continue
                                     # across rounds and end on a fall or after VelTaskConfig.episode_seconds)
    beta_decay: float = 0.3          # P(execute teacher action) = beta_decay ** round (round 0: 1)
    epochs: int = 30                 # passes over the aggregated dataset per round (when grad_steps = 0)
    batch_size: int = 256
    lr: float = 1e-3
    hidden: list[int] = field(default_factory=lambda: [256, 128, 64])   # = PPOConfig.actor_hidden
    activation: str = "elu"
    init_std: float = 0.3            # exploration std stored with the student (for later PPO fine-tuning)
    commands: str = "auto"           # "auto" ("goals" if the teacher has set_velocity, else "zero") | "zero" |
                                     # "sample" (task command sampling) | "goals" (PathFollower to random goals
                                     # and random command schedules: forward, turn in place, sideways, ...)
    seed: int = 0
    name: str = "distill"
    # ---- scaling up (all optional; the defaults keep the small in-process behaviour) ------------------
    workers: int = 0                 # > 0: collect in this many processes (num_envs each), 1 thread each
    grad_steps: int = 0              # > 0: minibatch steps per round instead of `epochs` passes
    lr_final: float = 0.0            # > 0: learning rate decays geometrically to this by the last round
    buffer_size: int = 1_000_000     # aggregated samples kept (random replacement beyond)
    keep_every: int = 1              # store every k-th student step (consecutive steps are near duplicates)
    goal_prob: float = 0.6           # "goals": share of episodes driven by a PathFollower
    stand_prob: float = 0.0          # share of episodes that stand 2-10 s before their first command
    min_beta: float = 0.0            # floor of the teacher-execution probability after round 0
    resume: str | None = None        # rl_policy.npz of an earlier distill run to continue from
    save_every: int = 1              # write the student every this many rounds
    keep_rounds: bool = False        # also keep every round's student as rl_policy_rNN.npz (to pick the best)


def _touch_contacts(task: VelocityTask, datas: list[mujoco.MjData]) -> np.ndarray:
    if task.touch_adr.size < 2:
        return np.zeros((len(datas), 2), bool)
    return np.stack([d.sensordata[task.touch_adr] > task.cfg.contact_force for d in datas])


def _default_follower(cfg: VelTaskConfig) -> PathFollower:
    """The path follower RLController uses for a policy trained on `cfg` (commands within its ranges)."""
    lim = lambda r: max(abs(r[0]), abs(r[1]))  # noqa: E731
    v_max = min(0.8, max(0.05, cfg.cmd_vx[1]))
    wz_max = min(1.0, lim(cfg.cmd_wz))
    if cfg.follow_speed > 0:
        v_max = min(v_max, cfg.follow_speed)
    if cfg.follow_turn > 0:
        wz_max = min(wz_max, cfg.follow_turn)
    return PathFollower(v_max=v_max, vy_max=min(0.3, lim(cfg.cmd_vy)), wz_max=wz_max, final_tolerance=0.1)


def _clamp_command(cfg: VelTaskConfig, cmd) -> np.ndarray:
    """Clip a command to the training ranges and zero a translation below small_cmd (as RLController does)."""
    c = np.asarray(cmd, float).copy()
    c[0] = np.clip(c[0], *cfg.cmd_vx)
    c[1] = np.clip(c[1], *cfg.cmd_vy)
    c[2] = np.clip(c[2], *cfg.cmd_wz)
    if np.linalg.norm(c[:2]) < cfg.small_cmd:
        c[:2] = 0.0
    return c


class _CommandSource:
    """Per-environment command generator for distillation."""

    def __init__(self, mode: str, cfg: VelTaskConfig, rng: np.random.Generator, goal_prob: float,
                 stand_prob: float = 0.0):
        self.mode, self.cfg, self.rng, self.goal_prob = mode, cfg, rng, goal_prob
        self.stand_prob = float(stand_prob)
        self.follower = _default_follower(cfg)
        self.kind = "zero"
        self.timer = 0.0
        self.cmd = np.zeros(3)
        self.hold = 0.0

    def reset(self) -> None:
        r = self.rng
        self.cmd = np.zeros(3)
        # stand before the first command: usually a moment, sometimes long (an idle robot after a reset must
        # stand indefinitely, and with the "steps" clock standing only ever happens before the first command)
        self.timer = float(r.uniform(2.0, 10.0) if r.random() < self.stand_prob else r.uniform(0.0, 1.0))
        if self.mode == "goals":
            self.kind = "goal" if r.random() < self.goal_prob else "schedule"
        else:
            self.kind = self.mode
        self.follower.set_path(None)
        self.hold = 0.0

    def _random_command(self) -> np.ndarray:
        cfg, r = self.cfg, self.rng
        vx_lo, vx_hi = cfg.cmd_vx
        vy = max(abs(cfg.cmd_vy[0]), abs(cfg.cmd_vy[1]))
        wz = max(abs(cfg.cmd_wz[0]), abs(cfg.cmd_wz[1]))
        u = r.random()
        if u < 0.12:
            c = (0.0, 0.0, 0.0)                                          # step in place
        elif u < 0.40:
            c = (r.uniform(0.1, vx_hi), 0.0, r.uniform(-0.15, 0.15) * wz)  # forward
        elif u < 0.58:
            c = (0.0, 0.0, r.choice([-1, 1]) * r.uniform(0.15, 1.0) * wz)  # turn in place
        elif u < 0.68:
            c = (0.0, r.choice([-1, 1]) * r.uniform(0.3, 1.0) * vy, 0.0)   # sideways
        elif u < 0.76:
            c = (r.uniform(vx_lo, min(-0.05, vx_lo / 3)), 0.0, 0.0)      # backwards
        else:
            c = (r.uniform(vx_lo, vx_hi), r.uniform(-vy, vy), r.uniform(-wz, wz))
        return np.array(c, float)

    def command(self, x: float, y: float, yaw: float, dt: float) -> np.ndarray:
        """Advance by dt and return the (clamped) command for this step."""
        cfg, r = self.cfg, self.rng
        self.timer -= dt
        if self.kind == "zero":
            return np.zeros(3)
        if self.timer > 0 and self.kind != "goal":
            return self.cmd
        if self.kind == "sample":
            v = np.stack([r.uniform(*cfg.cmd_vx), r.uniform(*cfg.cmd_vy), r.uniform(*cfg.cmd_wz)])
            if np.linalg.norm(v[:2]) < cfg.small_cmd:
                v[:2] = 0.0
            if r.random() < cfg.zero_cmd_prob:
                v[:] = 0.0
            self.cmd, self.timer = v, cfg.resample_seconds
            return self.cmd
        if self.kind == "schedule":
            self.cmd = _clamp_command(cfg, self._random_command())
            self.timer = float(r.uniform(1.5, 4.0))
            return self.cmd
        # "goal": walk to random goals with the deployment path follower, wait a little at each
        if self.timer > 0:
            return np.zeros(3)
        if not self.follower.active:
            dist = r.uniform(0.3, 4.5)
            ang = r.uniform(-np.pi, np.pi)
            goal = np.array([x + dist * np.cos(ang), y + dist * np.sin(ang)])
            fy = float(r.uniform(-np.pi, np.pi)) if r.random() < 0.4 else None
            self.follower.set_path(np.array([[x, y], goal]), fy)
            self.hold = float(r.uniform(15.0, 30.0))      # give up on a goal after this long
        c = self.follower.command(x, y, yaw)
        self.hold -= dt
        if c.done or self.hold <= 0:
            self.follower.set_path(None)
            self.timer = float(r.uniform(0.5, 2.5))       # stand (step in place) at the goal
            return np.zeros(3)
        return _clamp_command(cfg, (c.vx, c.vy, c.wz))


class _DistillEnvs:
    """A slice of the DAgger environments: MuJoCo data, a teacher and a command source per environment.

    Runs in a worker process (or in the caller's process when DistillConfig.workers = 0). Environments
    persist across rounds; each round executes the teacher with probability beta per step, otherwise the
    student, and labels every visited state with the teacher's action (DAgger)."""

    def __init__(self, model: mujoco.MjModel, teacher_factory, task_cfg: dict, dcfg: dict, n_envs: int,
                 seed: int, variant_seed: int):
        self.model = model
        self.tcfg = VelTaskConfig.from_dict(task_cfg)
        self.dcfg = DistillConfig(**dcfg)
        self.task = task = VelocityTask(model, self.tcfg)
        self.rng = np.random.default_rng(seed)
        self.n = n = int(n_envs)
        variants, _ = make_variants(model, self.tcfg, np.random.default_rng(variant_seed), task)
        self.models = [variants[i % len(variants)] for i in range(n)]
        self.teachers = [teacher_factory(model) for _ in range(n)]   # nominal model: same step timing for all
        t0 = self.teachers[0]
        self.dt_phys = float(model.opt.timestep)
        self.tick = max(1, int(round(float(getattr(t0, "control_dt", task.dt)) / self.dt_phys)))
        self.has_vel = all(hasattr(t, "set_velocity") for t in self.teachers)
        mode = self.dcfg.commands
        if mode == "auto":
            mode = "goals" if self.has_vel else "zero"
        if not self.has_vel:
            mode = "zero"
        self.cmd_mode = mode
        self.steps_clock = task.step_clock_on and self.has_vel
        self.walker = self.steps_clock and _is_footstep_walker(t0)
        self.sources = [_CommandSource(mode, self.tcfg, self.rng, self.dcfg.goal_prob, self.dcfg.stand_prob)
                        for _ in range(n)]
        self.datas = [mujoco.MjData(m) for m in self.models]
        self.ts = task.make_state(n)
        self.counter = np.zeros(n, int)
        self.latest = np.zeros((n, model.nu))
        self.sent = [None] * n            # command last sent to each teacher
        self.ep_t = np.zeros(n)
        self.push_timer = np.zeros(n)
        self.pushes = self.tcfg.randomize and self.tcfg.push_interval_s > 0 and self.tcfg.push_vel > 0
        self.student: GaussianPolicy | None = None
        self.norm: RunningMeanStd | None = None
        self._states = np.zeros((n, task.nstate))
        self._reset(np.arange(n))

    # ---- environment management ------------------------------------------------------------------------
    def _reset(self, idx) -> None:
        task, cfg, r = self.task, self.tcfg, self.rng
        idx = np.asarray(idx, int)
        task.reset_envs(self._states, self.ts, idx, r)
        for i in idx:
            d, m = self.datas[i], self.models[i]
            mujoco.mj_resetData(m, d)
            mujoco.mj_setState(m, d, self._states[i], STATE_SPEC)
            d.ctrl[:] = task.ctrl_default
            mujoco.mj_forward(m, d)
            # The teacher's own start pose and internal state. (No set_path(None) after it: for the footstep
            # walker that plans square-up steps whenever the feet are not exactly in its nominal stance, so it
            # would start stepping on its own, out of step with the student's clock.)
            self.teachers[i].reset(m, d)
            # small perturbations after the teacher placed the robot
            d.qpos[task.qadr] += r.uniform(-cfg.init_joint_noise, cfg.init_joint_noise, task.act_dim)
            d.qpos[task.qadr] = np.clip(d.qpos[task.qadr], task.q_lo, task.q_hi)
            d.qvel[0:2] += r.uniform(-cfg.init_vel_noise, cfg.init_vel_noise, 2)
            mujoco.mj_forward(m, d)
            self.sources[i].reset()
            self.sent[i] = None
            self.counter[i] = 0
            self.ep_t[i] = 0.0
            self.push_timer[i] = r.uniform(0.3, 1.0) * max(cfg.push_interval_s, 1e-6)
        self.ts.commands[idx] = 0.0
        self.ts.cmd_timer[idx] = 1e9

    def set_student(self, arrays: dict | None) -> None:
        if arrays is None:
            self.student = self.norm = None
            return
        self.student = GaussianPolicy.from_arrays(arrays, self.dcfg.activation)
        self.norm = RunningMeanStd.from_arrays(arrays, "obs_")

    def _send_command(self, i: int, cmd: np.ndarray) -> None:
        """Pass this step's command to teacher i. With the "steps" clock a footstep walker never stops once it
        walks: a zero command becomes stepping in place (the clock keeps running, as at deployment)."""
        t = self.teachers[i]
        if not self.has_vel:
            return
        if self.steps_clock:
            walking = self.ts.walk_t[i] >= 0
            if not walking:
                return
            send = tuple(map(float, cmd)) if np.any(np.abs(cmd) > 1e-6) else (0.0, 0.0, 1e-5)
        else:
            send = tuple(map(float, cmd))
        if send != self.sent[i]:
            t.set_velocity(*send)
            self.sent[i] = send

    # ---- one round ---------------------------------------------------------------------------------------
    def run(self, student: dict | None, beta: float, seconds: float, keep_every: int = 1,
            reset: bool = False) -> dict:
        task, cfg, r, n = self.task, self.tcfg, self.rng, self.n
        self.set_student(student)
        if reset:
            self._reset(np.arange(n))
        steps = max(1, int(round(seconds / task.dt)))
        keep_every = max(1, int(keep_every))
        obs_buf, act_buf = [], []
        falls = timeouts = 0
        dis_sum, dis_n, stud_steps, stud_falls = 0.0, 0, 0, 0
        chk = mis = 0
        wall0 = time.perf_counter()
        for s in range(steps):
            # commands
            for i, d in enumerate(self.datas):
                x, y, yaw = base_pose(d)
                self.ts.commands[i] = self.sources[i].command(x, y, yaw, task.dt)
            task.update_clock(self.ts)
            for i in range(n):
                self._send_command(i, self.ts.commands[i])
            qpos = np.stack([d.qpos for d in self.datas])
            qvel = np.stack([d.qvel for d in self.datas])
            obs, _ = task.observe(qpos, qvel, self.ts, rng=r)
            obs = np.where(np.isfinite(obs), obs, 0.0)
            use_teacher = (r.random(n) < beta) if self.student is not None else np.ones(n, bool)
            a_student = None
            if self.student is not None and not use_teacher.all():
                a_student = self.student.mean(self.norm.normalize(obs))
                a_student = np.clip(np.where(np.isfinite(a_student), a_student, 0.0), -cfg.clip_actions,
                                    cfg.clip_actions)
                ctrl_student = task.action_to_ctrl(a_student)
            labels = np.zeros((n, task.act_dim))
            check = self.walker and (s % 5 == 0)
            expect = task.step_clock(self.ts.walk_t)["in_swing"] if check else None
            for i, d in enumerate(self.datas):
                m, t = self.models[i], self.teachers[i]
                exec_ctrl = None
                for j in range(cfg.decimation):
                    if self.counter[i] % self.tick == 0:
                        t.step(m, d)
                        self.latest[i] = d.ctrl
                    if j == 0:
                        labels[i] = task.ctrl_to_action(self.latest[i])[0]
                        if check:
                            sup = t.status().get("support")
                            want = "double" if not expect[i].any() else ("right" if expect[i, 0] else "left")
                            chk += 1
                            mis += int(sup != want)
                        exec_ctrl = (task.action_to_ctrl(labels[i][None])[0] if use_teacher[i]
                                     else ctrl_student[i])
                    d.ctrl[:] = exec_ctrl
                    mujoco.mj_step(m, d)
                    self.counter[i] += 1
            executed = labels if a_student is None else np.where(use_teacher[:, None], labels, a_student)
            if a_student is not None:
                st = ~use_teacher
                dis_sum += float(np.sum((a_student[st] - labels[st]) ** 2))
                dis_n += int(st.sum()) * task.act_dim
                stud_steps += int(st.sum())
            if s % keep_every == 0:
                obs_buf.append(obs.astype(np.float16))
                act_buf.append(labels.astype(np.float32))
            # bookkeeping (what VelocityTask.reward does for the buffers the observation uses)
            self.ts.last_action[:] = executed
            task.advance_time(self.ts)
            self.ep_t += task.dt
            qpos = np.stack([d.qpos for d in self.datas])
            qvel = np.stack([d.qvel for d in self.datas])
            _, _, _, grav = VelocityTask.base_frame(qpos, qvel)
            fell = ((qpos[:, 2] < cfg.min_height_frac * task.z0) | (-grav[:, 2] < np.cos(np.deg2rad(cfg.max_tilt_deg)))
                    | ~np.isfinite(qpos).all(axis=1))
            if self.pushes:
                self.push_timer -= task.dt
                for i in np.flatnonzero(self.push_timer <= 0):
                    self.datas[i].qvel[0:2] += r.uniform(-cfg.push_vel, cfg.push_vel, 2)
                    self.push_timer[i] = cfg.push_interval_s * r.uniform(0.7, 1.3)
            done = fell | (self.ep_t >= cfg.episode_seconds)
            falls += int(fell.sum())
            if a_student is not None:
                stud_falls += int((fell & ~use_teacher).sum())
            timeouts += int((done & ~fell).sum())
            if done.any():
                self._reset(np.flatnonzero(done))
        return {"obs": np.concatenate(obs_buf) if obs_buf else np.zeros((0, task.obs_dim), np.float16),
                "act": np.concatenate(act_buf) if act_buf else np.zeros((0, task.act_dim), np.float32),
                "falls": falls, "timeouts": timeouts, "env_seconds": steps * task.dt * n,
                "dis_sum": dis_sum, "dis_n": dis_n, "student_steps": stud_steps, "student_falls": stud_falls,
                "clock_checks": chk, "clock_mismatch": mis, "wall": time.perf_counter() - wall0}


def _distill_worker_main(conn, spec: dict) -> None:
    """Worker process: build the environments once, then serve "run" requests until "close"."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass
    envs = _DistillEnvs(**spec)
    while True:
        msg = conn.recv()
        if msg[0] == "close":
            break
        try:
            conn.send(("ok", envs.run(**msg[1])))
        except Exception as e:  # report instead of hanging the parent
            import traceback
            conn.send(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
    conn.close()


class _WorkerPool:
    """The distillation workers: `workers` processes (one thread each) or the caller's process."""

    def __init__(self, model, teacher_factory, tcfg: VelTaskConfig, dcfg: DistillConfig):
        n_workers = max(0, int(dcfg.workers))
        self.procs, self.conns, self.local = [], [], None
        base = {"model": model, "teacher_factory": teacher_factory, "task_cfg": tcfg.to_dict(),
                "dcfg": asdict(dcfg), "n_envs": dcfg.num_envs}
        if n_workers == 0:
            self.local = _DistillEnvs(**base, seed=dcfg.seed, variant_seed=dcfg.seed + 7919)
            return
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
        try:
            for k in saved:
                os.environ[k] = "1"   # one BLAS thread per worker process (the CPU is shared)
            for w in range(n_workers):
                a, b = ctx.Pipe()
                p = ctx.Process(target=_distill_worker_main, daemon=True, name=f"distill-{w}",
                                args=(b, {**base, "seed": dcfg.seed * 1000 + w + 1, "variant_seed": dcfg.seed + 7919}))
                p.start()
                self.procs.append(p)
                self.conns.append(a)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def run(self, **kw) -> list[dict]:
        if self.local is not None:
            return [self.local.run(**kw)]
        for c in self.conns:
            c.send(("run", kw))
        out = []
        for p, c in zip(self.procs, self.conns):
            while not c.poll(5.0):   # a worker killed by the OS (e.g. out of memory) must not hang the run
                if not p.is_alive():
                    raise RuntimeError(f"distillation worker {p.name} died (exit code {p.exitcode})")
            status, res = c.recv()
            if status != "ok":
                raise RuntimeError(f"distillation worker failed: {res}")
            out.append(res)
        return out

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(("close",))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self.procs, self.conns = [], []


def _obs_std_floor(task: VelocityTask) -> np.ndarray:
    """Per-feature lower bound on the normaliser's std (fix for students that never saw a feature vary):
    commands get the spread of the command distribution they may be given, everything else 0.05."""
    cfg, a = task.cfg, task.act_dim
    rng_std = lambda r: (r[1] - r[0]) / np.sqrt(12.0)  # noqa: E731
    cmd = np.array([rng_std(cfg.cmd_vx), rng_std(cfg.cmd_vy), rng_std(cfg.cmd_wz)]) * task.cmd_scale
    cmd = np.maximum(cmd, 0.05)
    frame = np.full(task.proprio_dim, 0.05)
    frame[6:9] = cmd
    c0 = 9 + 3 * a
    if task.step_clock_on:
        frame[c0:c0 + 6] = 0.2
        frame[c0 + 6:c0 + 9] = cmd
    elif task.clock:
        frame[c0:c0 + 2] = 0.2
    return np.tile(frame, 1 + cfg.history)


def distill(model: mujoco.MjModel, teacher_factory: Callable[[mujoco.MjModel], object],
            task_cfg: VelTaskConfig | None = None, cfg: DistillConfig | None = None,
            run_dir: str | Path | None = None, model_meta: dict | None = None,
            callback: Callable[[dict], None] | None = None) -> tuple[RLPolicy, list[dict]]:
    """Distil any teacher controller into an MLP student with behaviour cloning + DAgger.

    teacher_factory(model) -> object with reset(model, data) and step(model, data) writing data.ctrl
    (the `Controller` protocol); optional set_velocity(vx, vy, wz) makes the student command-conditioned.
    Each environment gets its own teacher instance (teachers keep state, e.g. a gait phase) and the
    teacher is ticked at its own `control_dt` (e.g. 200 Hz for WalkController) on whatever state the
    *executed* policy produced, while the student acts at the task rate. The label of a student step is the
    teacher's output at the state the student observed; a teacher-executed step applies that same output for
    the whole step (zero-order hold, i.e. what the student can represent). Executing a mixture (teacher with
    probability beta, decaying per round) and labelling every visited state is DAgger (Ross et al. 2011):
    the student learns what to do in the states its own mistakes lead to. Domain randomisation (model
    variants: friction, mass, servo stiffness; each environment keeps one) and pushes from the task config
    perturb the robots so recovery behaviour is demonstrated too; the teacher is built on the nominal model,
    so it is privileged only in seeing the true state. Arms are held at their default targets.

    With task_cfg None and a footstep-walker teacher (set_velocity + status()["support"]) the task comes from
    walker_task_config: 100 Hz, the walker's stance, and the walker's step schedule as a gait clock in the
    observation ("steps" clock), so the walker's hidden timing is an input. Commands then come from a
    PathFollower walking to random goals (the deployment path) mixed with random command schedules.

    Commands with commands="zero" (or a teacher without set_velocity) are always zero (meta "command_mode").
    The observation normaliser has a per-feature std floor (commands: the spread of the task's command
    ranges), so a feature the student never saw vary, such as a command, cannot blow its input up.

    Returns (student policy, per-round report). With run_dir set, writes run_dir/rl_policy.npz (and an
    rl_checkpoint.npz that PPOTrainer / the GUI's "continue" can fine-tune) every `save_every` rounds.
    """
    cfg = cfg or DistillConfig()
    if task_cfg is None:
        probe = teacher_factory(model)
        if _is_footstep_walker(probe) and float(getattr(probe, "control_dt", 0.0)) <= float(model.opt.timestep) + 1e-9:
            tcfg = walker_task_config(model, teacher_factory)
        else:
            tcfg = VelTaskConfig()
    else:
        tcfg = VelTaskConfig.from_dict(task_cfg.to_dict())
    task = VelocityTask(model, tcfg)
    rng = np.random.default_rng(cfg.seed)
    student = GaussianPolicy(task.obs_dim, task.act_dim, cfg.hidden, cfg.activation, cfg.init_std, rng)
    norm = RunningMeanStd(task.obs_dim, std_floor=_obs_std_floor(task))
    start_round = 0
    lr0 = float(cfg.lr)
    if cfg.resume:
        arrays, rmeta = load_npz(cfg.resume)
        student = GaussianPolicy.from_arrays(arrays, cfg.activation)
        student.mu.params = [np.ascontiguousarray(p, np.float32) for p in student.mu.params]
        norm = RunningMeanStd.from_arrays(arrays, "obs_")
        if student.obs_dim != task.obs_dim:
            raise ValueError("resume: the student's observation size does not match this task")
        start_round = int(rmeta.get("distill_round", -1)) + 1
        # The aggregated dataset is not saved (hundreds of MB), so a resumed run starts with only fresh data:
        # continue at the learning rate the earlier run had reached instead of restarting at cfg.lr, which
        # would refit a small dataset hard and could undo the student.
        prev = [r.get("lr") for r in rmeta.get("report", []) if r.get("lr")]
        if prev:
            lr0 = float(prev[-1])
    opt = Adam(student.mu.params, cfg.lr)
    pool = _WorkerPool(model, teacher_factory, tcfg, cfg)
    cap = max(1, int(cfg.buffer_size))
    buf_obs = np.zeros((0, task.obs_dim), np.float16)
    buf_act = np.zeros((0, task.act_dim), np.float32)
    filled = 0
    report: list[dict] = []
    has_vel = pool.local.has_vel if pool.local is not None else hasattr(teacher_factory(model), "set_velocity")
    mode = cfg.commands if cfg.commands != "auto" else ("goals" if has_vel else "zero")
    if not has_vel:
        mode = "zero"
    teacher_name = type(teacher_factory(model)).__name__
    teacher_params = getattr(teacher_factory, "params", None)
    if getattr(teacher_factory, "source", None):
        teacher_params = {**(teacher_params or {}), "source": Path(teacher_factory.source).name}
    run_dir = Path(run_dir) if run_dir is not None else None

    def save(rnd: int) -> RLPolicy:
        meta = _policy_meta(task, model_meta or {}, "distill", teacher=teacher_name, teacher_params=teacher_params,
                            distill={k: v for k, v in asdict(cfg).items() if k != "resume"}, report=report,
                            distill_round=rnd, command_mode=mode)
        policy = RLPolicy(student, norm, {**meta, "activation": cfg.activation})
        if run_dir is not None:
            policy.save(run_dir / "rl_policy.npz")
            if cfg.keep_rounds:
                policy.save(run_dir / f"rl_policy_r{rnd:02d}.npz")
            save_npz(run_dir / "rl_checkpoint.npz", {**student.to_arrays("actor_"), **norm.to_arrays("obs_")},
                     {**meta, "format": "simulatics-rl-checkpoint"})
            (run_dir / "distill_report.json").write_text(json.dumps(report, indent=1))
        return policy

    policy = None
    try:
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.json").write_text(json.dumps({"distill": asdict(cfg), "task": tcfg.to_dict(),
                                                             "teacher": teacher_name, "teacher_params": teacher_params},
                                                            indent=2))
        for rnd in range(start_round, start_round + cfg.rounds):
            beta = 1.0 if rnd == 0 else max(cfg.min_beta, cfg.beta_decay ** rnd)
            t0 = time.perf_counter()
            arrays = None if rnd == 0 else {**student.to_arrays("actor_"), **norm.to_arrays("obs_")}
            results = pool.run(student=arrays, beta=beta, seconds=cfg.episode_seconds, keep_every=cfg.keep_every,
                               reset=(rnd == start_round))
            t_collect = time.perf_counter() - t0
            new_obs = np.concatenate([res["obs"] for res in results])
            new_act = np.concatenate([res["act"] for res in results])
            # DAgger loss on the newly visited states before refitting (generalisation to the student's states)
            pre_mse = None
            if rnd > 0 and len(new_obs):
                k = rng.choice(len(new_obs), min(len(new_obs), 20000), replace=False)
                with blas_threads(4):
                    pre_mse = float(np.mean((student.mean(norm.normalize(new_obs[k].astype(np.float32))) - new_act[k]) ** 2))
            # aggregate (random replacement once full)
            m = len(new_obs)
            if filled + m <= cap:
                buf_obs = np.concatenate([buf_obs[:filled], new_obs])
                buf_act = np.concatenate([buf_act[:filled], new_act])
                filled += m
            else:
                if filled < cap:
                    take = cap - filled
                    buf_obs = np.concatenate([buf_obs[:filled], new_obs[:take]])
                    buf_act = np.concatenate([buf_act[:filled], new_act[:take]])
                    filled = cap
                    new_obs, new_act = new_obs[take:], new_act[take:]
                if len(new_obs):
                    slots = rng.choice(cap, len(new_obs), replace=False)
                    buf_obs[slots], buf_act[slots] = new_obs, new_act
            del new_obs, new_act
            # refit the normaliser (std floor keeps unseen features bounded) and the student
            t1 = time.perf_counter()
            fit = rng.choice(filled, min(filled, 200000), replace=False)
            fresh = RunningMeanStd(task.obs_dim, std_floor=_obs_std_floor(task))
            fresh.update(buf_obs[fit].astype(np.float64))
            norm = fresh
            lr = lr0
            if cfg.lr_final > 0 and cfg.rounds > 1:
                lr = lr0 * (min(cfg.lr_final, lr0) / lr0) ** ((rnd - start_round) / (cfg.rounds - 1))
            opt.lr = lr
            bs = max(1, int(cfg.batch_size))
            n_steps = int(cfg.grad_steps) if cfg.grad_steps > 0 else int(cfg.epochs * int(np.ceil(filled / bs)))
            losses = []
            with blas_threads(4):
                if cfg.grad_steps > 0:
                    for _ in range(n_steps):
                        idx = rng.integers(0, filled, bs)
                        loss, grads = student.bc_loss(norm.normalize(buf_obs[idx].astype(np.float32)), buf_act[idx])
                        grads, _ = clip_grad_norm(grads[:-1], 1.0)
                        opt.step(grads)
                        losses.append(loss)
                else:
                    xn = norm.normalize(buf_obs[:filled].astype(np.float32)).astype(np.float32)
                    for _ in range(cfg.epochs):
                        perm = rng.permutation(filled)
                        for j in range(0, filled, bs):
                            idx = perm[j:j + bs]
                            loss, grads = student.bc_loss(xn[idx], buf_act[idx])
                            grads, _ = clip_grad_norm(grads[:-1], 1.0)
                            opt.step(grads)
                            losses.append(loss)
                    del xn
                k = rng.choice(filled, min(filled, 50000), replace=False)
                mse = float(np.mean((student.mean(norm.normalize(buf_obs[k].astype(np.float32))) - buf_act[k]) ** 2))
            t_train = time.perf_counter() - t1
            agg = lambda key: sum(res[key] for res in results)  # noqa: E731
            env_s = agg("env_seconds")
            row = {"round": rnd, "beta": beta, "samples": int(filled), "falls": int(agg("falls")),
                   "train_mse": mse,
                   "rollout_disagreement": (agg("dis_sum") / agg("dis_n")) if agg("dis_n") else None,
                   "pre_fit_mse": pre_mse, "falls_per_min": 60.0 * agg("falls") / max(env_s, 1e-9),
                   "student_falls_per_min": (60.0 * agg("student_falls") / max(agg("student_steps") * task.dt, 1e-9)
                                             if agg("student_steps") else None),
                   "clock_mismatch": (agg("clock_mismatch") / agg("clock_checks")) if agg("clock_checks") else None,
                   "env_seconds": env_s, "collect_s": t_collect, "train_s": t_train, "lr": lr, "grad_steps": n_steps,
                   "last_loss": float(np.mean(losses[-50:])) if losses else None}
            report.append(_clean_row(row))
            if callback is not None:
                callback(report[-1])
            if run_dir is not None and ((rnd - start_round + 1) % max(1, cfg.save_every) == 0
                                        or rnd == start_round + cfg.rounds - 1):
                policy = save(rnd)
    finally:
        pool.close()
    if policy is None:
        policy = save(start_round + cfg.rounds - 1)
    return policy, report


# =========================================================================================================
# Controller for the live session
# =========================================================================================================
class RLController:
    """Runs an rl_policy.npz (PPO or distilled) on the robot; implements `control.base.Controller`.

    A PathFollower turns the planner's path into (vx, vy, wz) commands, clamped to the ranges the policy
    was trained on; `set_velocity` gives direct joystick-style commands. The policy writes the leg and
    abdomen servo targets; arms (and hands) are held at their default targets.

    Policies trained with a start pose (VelTaskConfig.init_qpos, e.g. a student of the footstep walker) are
    put into that stance by reset(), as WalkController.reset does. With the "steps" gait clock the robot
    stands until its first command and then keeps stepping (in place at a zero command), as it was trained.
    Non-finite observations are zeroed and a non-finite action holds the last good one (status reports it).
    """
    name = "rl"

    def __init__(self, model: mujoco.MjModel, policy: "RLPolicy | str | Path", follower: PathFollower | None = None):
        self.policy = policy if isinstance(policy, RLPolicy) else RLPolicy.load(policy)
        cfg = self.policy.task_config()
        cfg.obs_noise = 0.0
        self.task = VelocityTask(model, cfg)
        names = self.policy.meta.get("joint_names")
        if names is not None and list(names) != self.task.joint_names:
            raise ValueError(f"policy was trained for joints {names}, this robot has {self.task.joint_names}")
        if self.policy.obs_dim != self.task.obs_dim or self.policy.act_dim != self.task.act_dim:
            raise ValueError("policy observation/action size does not match this robot")
        self.control_dt = self.task.dt
        self.follower = follower or _default_follower(cfg)
        self.ts = self.task.make_state(1)
        self.velocity: tuple[float, float, float] | None = None
        self._mode, self._msg = "stand", ""
        self._goal_err, self._yaw_err = None, None
        self.nan_steps = 0
        # Position hold: a "steps"-clock policy keeps stepping at a zero command and slowly wanders (measured
        # 5-18 cm in 30 s). After a path ends or is cleared it holds the goal (or where it was) with small
        # corrective commands, inside what it was trained on: start above hold_start m, stop below hold_stop m.
        self.hold_start, self.hold_stop, self.hold_speed = 0.08, 0.03, 0.15
        self._hold: np.ndarray | None = None
        self._hold_pending = False
        self._holding = False

    def reset(self, model, data) -> None:
        self.ts = self.task.make_state(1)
        self._mode, self._msg = "stand", ""
        self.nan_steps = 0
        self._hold, self._hold_pending, self._holding = None, False, False
        if self.task.cfg.init_qpos is not None:
            self._place_in_stance(model, data)

    def _place_in_stance(self, model, data) -> None:
        """Put the robot into the trained start stance where it stands (x, y and heading kept): root height and
        tilt, abdomen and legs from init_qpos, velocities zero, leg/abdomen servo targets on the stance. Arms and
        hands keep their angles and targets."""
        task = self.task
        q0 = np.asarray(task.cfg.init_qpos, float)
        if not np.isfinite(data.qpos).all():
            mujoco.mj_resetData(model, data)
        x, y, yaw = base_pose(data)
        c, s = np.cos(yaw / 2), np.sin(yaw / 2)
        qy = np.array([c, 0.0, 0.0, s])
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, qy, q0[3:7] / np.linalg.norm(q0[3:7]))
        data.qpos[0:3] = (x, y, task.init_qpos[2])
        data.qpos[3:7] = quat
        data.qpos[task.qadr] = task.q_default
        data.qvel[:] = 0.0
        data.ctrl[task.ctrl_idx] = task.ctrl_default[task.ctrl_idx]
        mujoco.mj_forward(model, data)

    def set_path(self, path_xy, final_yaw=None) -> None:
        """A non-empty path replaces any velocity command; None / an empty path means stop and stand
        (the Controller contract, as in WalkController). The session's reset and the navigator's
        stop/clear arrive here as an empty path, so they also cancel a `set_velocity` command."""
        self.follower.set_path(path_xy, final_yaw)
        self.velocity = None
        self._holding = False
        if path_xy is not None and len(np.atleast_2d(path_xy)):
            self._mode, self._msg = "walk", ""
            self._hold, self._hold_pending = None, False
        else:
            self._goal_err = self._yaw_err = None
            self._hold_pending = self._hold is None   # keep holding a reached goal, else hold where it stands
            if self._mode != "fallen":
                self._mode, self._msg = "stand", ""

    def set_velocity(self, vx: float, vy: float = 0.0, wz: float = 0.0) -> None:
        """Direct velocity command (robot frame); clears any path (and the position hold)."""
        self.follower.set_path(None)
        self.velocity = (float(vx), float(vy), float(wz))
        self._hold, self._hold_pending, self._holding = None, False, False
        self._mode = "walk" if any(abs(v) > 1e-3 for v in self.velocity) else "stand"

    def _hold_command(self, x: float, y: float, yaw: float) -> np.ndarray:
        """Small corrective velocity back to the hold point (only while the gait is stepping)."""
        if self._hold_pending:
            self._hold, self._hold_pending = np.array([x, y]), False
        if self._hold is None or not (self.task.step_clock_on and self.ts.walk_t[0] >= 0):
            return np.zeros(3)
        e = self._hold - np.array([x, y])
        d = float(np.hypot(*e))
        self._holding = d > self.hold_start or (self._holding and d > self.hold_stop)
        if not self._holding:
            return np.zeros(3)
        c, s = np.cos(yaw), np.sin(yaw)
        v = np.array([c * e[0] + s * e[1], -s * e[0] + c * e[1]]) * 1.2
        n = float(np.hypot(*v))
        if n > self.hold_speed:
            v *= self.hold_speed / n
        return np.array([v[0], v[1], 0.0])

    def _command(self, data) -> np.ndarray:
        x, y, yaw = base_pose(data)
        cmd = np.zeros(3)
        if self.follower.active:
            goal = self.follower.path[-1].copy()
            c = self.follower.command(x, y, yaw)
            self._goal_err, self._yaw_err = c.position_error, c.heading_error
            if c.done:
                self.follower.set_path(None)
                self._mode, self._msg = "done", f"goal reached ({c.position_error:.2f} m)"
                self._hold, self._hold_pending, self._holding = goal, False, False
            else:
                cmd[:] = (c.vx, c.vy, c.wz)
                self._mode = "walk"
        elif self.velocity is not None:
            cmd[:] = self.velocity
        else:
            cmd[:] = self._hold_command(x, y, yaw)
        cmd = np.where(np.isfinite(cmd), cmd, 0.0)
        return _clamp_command(self.task.cfg, cmd)

    def step(self, model, data) -> None:
        task, ts = self.task, self.ts
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            self._mode, self._msg = "fallen", "simulation state is not finite"
            return
        ts.commands[0] = self._command(data)
        task.update_clock(ts)
        qpos, qvel = data.qpos[None].copy(), data.qvel[None].copy()
        obs, _ = task.observe(qpos, qvel, ts)
        obs = np.where(np.isfinite(obs), obs, 0.0)
        action = self.policy.act(obs)
        if not np.isfinite(action).all():   # NaN guard: never send NaN to the servos; hold the last good action
            self.nan_steps += 1
            action = ts.last_action.copy()
            self._msg = f"policy output was not finite ({self.nan_steps} steps); holding the last action"
        action = np.clip(action, -task.cfg.clip_actions, task.cfg.clip_actions)
        data.ctrl[:] = task.action_to_ctrl(action)[0]
        ts.last_action[:] = action
        ts.last_dqd[:] = qvel[:, task.dadr]
        task.advance_time(ts)
        up = -VelocityTask.base_frame(qpos, qvel)[3][0, 2]  # cos(tilt) = -(projected gravity z)
        if qpos[0, 2] < task.cfg.min_height_frac * task.z0 or up < np.cos(np.deg2rad(task.cfg.max_tilt_deg)):
            self._mode, self._msg = "fallen", "robot fell"

    def status(self) -> dict:
        out = {"mode": self._mode, "message": self._msg, "command": self.ts.commands[0].round(3).tolist()}
        if self._goal_err is not None:
            out["goal_error_m"] = float(self._goal_err)
        if self._yaw_err is not None:
            out["heading_error_deg"] = float(np.rad2deg(self._yaw_err))
        if self.task.step_clock_on:
            out["gait"] = "stepping" if self.ts.walk_t[0] >= 0 else "standing"
            out["holding_position"] = bool(self._holding)
        return out


# =========================================================================================================
# Evaluation through the deployment path (RLController + PathFollower on one MjData, like the session)
# =========================================================================================================
def _controller_episode(model, ctrl, data=None):
    data = data or mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    ctrl.reset(model, data)
    ctrl.set_path(None)
    return data


def evaluate_goals(policy: "RLPolicy | str | Path", model: mujoco.MjModel, trials: int = 10,
                   dist: tuple[float, float] = (2.0, 4.0), seed: int = 0, timeout: float = 45.0,
                   settle: float = 0.5, hold: float = 2.0, tolerance: float = 0.25, start_yaw: bool = True,
                   push_vel: float = 0.0, push_interval: float = 4.0) -> dict:
    """Goal-reaching trials through RLController with its PathFollower (the GUI's click-to-walk path).

    Each trial: fresh robot in its start pose (random heading when start_yaw), `settle` s standing, then a goal
    `dist` metres away in a uniformly random direction relative to the robot; runs until the follower reports
    done (or a fall / timeout), then `hold` s more (stepping in place) and measures the final torso (x, y)
    distance to the goal. success = no fall and final error <= tolerance. push_vel > 0 adds a random base
    velocity kick of up to that many m/s (per axis) every push_interval s while walking (robustness check)."""
    pol = policy if isinstance(policy, RLPolicy) else RLPolicy.load(policy)
    rng = np.random.default_rng(seed)
    push_rng = np.random.default_rng(seed + 10_007)   # separate stream: the goals do not depend on push_vel
    rows = []
    for k in range(trials):
        ctrl = RLController(model, pol)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        if start_yaw:
            a0 = rng.uniform(-np.pi, np.pi)
            data.qpos[3:7] = (np.cos(a0 / 2), 0.0, 0.0, np.sin(a0 / 2))
        mujoco.mj_forward(model, data)
        ctrl.reset(model, data)
        ctrl.set_path(None)
        sub = max(1, int(round(ctrl.control_dt / model.opt.timestep)))
        for _ in range(int(round(settle / ctrl.control_dt))):
            ctrl.step(model, data)
            mujoco.mj_step(model, data, sub)
        x0, y0, yaw0 = base_pose(data)
        d = rng.uniform(*dist)
        rel = rng.uniform(-np.pi, np.pi)
        goal = np.array([x0 + d * np.cos(yaw0 + rel), y0 + d * np.sin(yaw0 + rel)])
        ctrl.set_path(np.array([[x0, y0], goal]), None)
        t, t_done, fell = 0.0, None, False
        wall = 0.0
        next_push = push_interval
        while t < timeout + hold:
            if push_vel > 0 and t >= next_push:
                data.qvel[0:2] += push_rng.uniform(-push_vel, push_vel, 2)
                next_push += push_interval
            w0 = time.perf_counter()
            ctrl.step(model, data)
            mujoco.mj_step(model, data, sub)
            wall += time.perf_counter() - w0
            t += ctrl.control_dt
            mode = ctrl.status()["mode"]
            if mode == "fallen":
                fell = True
                break
            if t_done is None and mode == "done":
                t_done = t
            if t_done is not None and t >= t_done + hold:
                break
            if t_done is None and t >= timeout:
                break
        err = float(np.hypot(*(data.qpos[:2] - goal)))
        rows.append({"trial": k, "distance_m": float(d), "direction_deg": float(np.rad2deg(rel)),
                     "fell": fell, "reached": t_done is not None, "time_s": t_done, "final_error_m": err,
                     "success": (not fell) and err <= tolerance, "realtime_factor": t / max(wall, 1e-9)})
    errs = np.array([r["final_error_m"] for r in rows if not r["fell"]])
    q = (lambda p: float(np.percentile(errs, p))) if len(errs) else (lambda p: None)  # noqa: E731
    return {"trials": rows, "success_rate": float(np.mean([r["success"] for r in rows])),
            "falls": int(sum(r["fell"] for r in rows)), "error_median_m": q(50), "error_p90_m": q(90),
            "error_max_m": float(errs.max()) if len(errs) else None, "error_mean_m": float(errs.mean()) if len(errs) else None,
            "time_mean_s": float(np.mean([r["time_s"] for r in rows if r["time_s"] is not None]))
            if any(r["time_s"] is not None for r in rows) else None,
            "realtime_factor": float(np.median([r["realtime_factor"] for r in rows]))}


def evaluate_commands(policy: "RLPolicy | str | Path", model: mujoco.MjModel, commands=None,
                      seconds: float = 10.0, settle: float = 3.0) -> list[dict]:
    """Velocity tracking through RLController.set_velocity: per command, the mean body-frame velocity and yaw
    rate measured after `settle` s (robot frame), RMS errors, net drift (m, for turning in place) and falls."""
    pol = policy if isinstance(policy, RLPolicy) else RLPolicy.load(policy)
    commands = commands or [(0.2, 0, 0), (0.3, 0, 0), (0.4, 0, 0), (0.0, 0.0, 0.5), (0.0, 0.0, -0.5),
                            (0.0, 0.1, 0.0), (0.0, -0.1, 0.0), (-0.15, 0, 0), (0.3, 0.0, 0.3), (0, 0, 0)]
    out = []
    for cmd in commands:
        ctrl = RLController(model, pol)
        data = _controller_episode(model, ctrl)
        sub = max(1, int(round(ctrl.control_dt / model.opt.timestep)))
        ctrl.set_velocity(*cmd)
        want = _clamp_command(ctrl.task.cfg, cmd)
        v, w, fell = [], [], False
        p_start = None
        yaw_prev, yaw_acc = None, 0.0
        for k in range(int(round(seconds / ctrl.control_dt))):
            ctrl.step(model, data)
            mujoco.mj_step(model, data, sub)
            if ctrl.status()["mode"] == "fallen":
                fell = True
                break
            x, y, yaw = base_pose(data)
            if k * ctrl.control_dt >= settle:
                if p_start is None:
                    p_start, yaw_prev = np.array([x, y]), yaw
                c, s = np.cos(yaw), np.sin(yaw)
                v.append([c * data.qvel[0] + s * data.qvel[1], -s * data.qvel[0] + c * data.qvel[1]])
                yaw_acc += float((yaw - yaw_prev + np.pi) % (2 * np.pi) - np.pi)
                yaw_prev = yaw
        span = max(len(v) * ctrl.control_dt, 1e-9)
        vm = np.mean(v, axis=0) if v else np.array([np.nan, np.nan])
        out.append({"command": [float(c) for c in cmd], "clamped": want.round(3).tolist(), "fell": fell,
                    "vx": float(vm[0]), "vy": float(vm[1]), "wz": yaw_acc / span,
                    "vx_err": float(vm[0] - want[0]), "vy_err": float(vm[1] - want[1]),
                    "wz_err": yaw_acc / span - float(want[2]),
                    "drift_m": float(np.hypot(*(data.qpos[:2] - p_start))) if p_start is not None else None,
                    "seconds": span})
    return out


# =========================================================================================================
# Command line:  python -m simulatics.train.ppo {train,distill,finetune,eval,bench} ...
# =========================================================================================================
def bench_env(threads: int = 8, num_envs: int = 256, steps: int = 100, seed: int = 0) -> dict:
    """Env-steps per second of the vectorised humanoid task with a [256,128,64] policy in the loop."""
    from ..models import HumanoidParams, build_mjcf
    model = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams()))
    rng = np.random.default_rng(seed)
    with VecEnv(model, VelTaskConfig(), num_envs, threads, seed) as env, blas_threads(1):
        pol = GaussianPolicy(env.obs_dim, env.act_dim, [256, 128, 64], "elu", 0.5, rng)
        obs, _ = env.reset()
        for _ in range(5):
            obs, *_ = env.step(pol.sample(obs, rng)[0])
        t_env = 0.0
        t0 = time.perf_counter()
        for _ in range(steps):
            a = pol.sample(obs, rng)[0]
            t1 = time.perf_counter()
            obs, *_ = env.step(a)
            t_env += time.perf_counter() - t1
        wall = time.perf_counter() - t0
    n = num_envs * steps
    return {"threads": threads, "num_envs": num_envs, "env_steps_per_s": n / wall, "env_only_steps_per_s": n / t_env,
            "physics_steps_per_s": n * env.cfg.decimation / t_env}


def _print_eval(policy_path: str | Path, model, trials: int, seed: int) -> dict:
    goals = evaluate_goals(policy_path, model, trials=trials, seed=seed)
    for r in goals["trials"]:
        print(f"  goal {r['trial']:2d}: {r['distance_m']:.2f} m at {r['direction_deg']:+5.0f} deg -> "
              f"{'FELL' if r['fell'] else ('ok  ' if r['success'] else 'miss')} error {r['final_error_m']:.3f} m "
              f"time {r['time_s'] if r['time_s'] is None else round(r['time_s'], 1)} s", flush=True)
    print(f"goals: success {goals['success_rate']:.0%}, falls {goals['falls']}/{trials}, error median "
          f"{goals['error_median_m']} p90 {goals['error_p90_m']} max {goals['error_max_m']} m, "
          f"realtime x{goals['realtime_factor']:.1f}", flush=True)
    cmds = evaluate_commands(policy_path, model)
    for r in cmds:
        print(f"  cmd {r['clamped']}: vx {r['vx']:+.3f} vy {r['vy']:+.3f} wz {r['wz']:+.3f}  "
              f"err ({r['vx_err']:+.3f}, {r['vy_err']:+.3f}, {r['wz_err']:+.3f}) drift {r['drift_m']} fell {r['fell']}",
              flush=True)
    return {"goals": goals, "commands": cmds}


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m simulatics.train.ppo", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train", help="PPO velocity tracking on the built-in humanoid")
    tr.add_argument("--name", default="rl_walker")
    tr.add_argument("--iterations", type=int, default=1500)
    tr.add_argument("--seconds", type=float, default=0.0, help="wall-clock budget (0 = none)")
    tr.add_argument("--envs", type=int, default=256)
    tr.add_argument("--threads", type=int, default=8)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--resume", default=None, help="rl_checkpoint.npz to continue from")
    tr.add_argument("--runs", default="runs")
    di = sub.add_parser("distill", help="behaviour cloning + DAgger from a teacher controller")
    di.add_argument("--teacher", choices=["stand", "balance", "walker"], default="balance")
    di.add_argument("--name", default="distilled")
    di.add_argument("--rounds", type=int, default=4)
    di.add_argument("--envs", type=int, default=8, help="environments (per worker with --workers)")
    di.add_argument("--seconds", type=float, default=5.0, help="simulated seconds per environment per round")
    di.add_argument("--workers", type=int, default=0, help="collection processes (1 thread each)")
    di.add_argument("--grad-steps", type=int, default=0, help="minibatch steps per round (0 = epochs)")
    di.add_argument("--batch", type=int, default=256)
    di.add_argument("--hidden", type=int, nargs="+", default=None)
    di.add_argument("--lr", type=float, default=1e-3)
    di.add_argument("--lr-final", type=float, default=0.0)
    di.add_argument("--keep-every", type=int, default=1)
    di.add_argument("--buffer", type=int, default=1_000_000)
    di.add_argument("--beta-decay", type=float, default=0.3)
    di.add_argument("--stand-prob", type=float, default=0.0, help="share of episodes standing 2-10 s first")
    di.add_argument("--keep-rounds", action="store_true", help="keep every round's student (rl_policy_rNN.npz)")
    di.add_argument("--goal-prob", type=float, default=0.6, help="share of episodes walking to random goals")
    di.add_argument("--step-forward", type=float, default=None, help="walker WalkParams.step_forward (x leg)")
    di.add_argument("--walker-source", default=None,
                    help="a copy of control/walker.py to use as the teacher (pins a long run to one version)")
    di.add_argument("--walker-params", default=None,
                    help='JSON dict of further WalkParams fields, e.g. \'{"recovery": false}\' (fixed step timing)')
    di.add_argument("--resume", default=None, help="rl_policy.npz of an earlier distill run")
    di.add_argument("--seed", type=int, default=0)
    di.add_argument("--runs", default="runs")
    ft = sub.add_parser("finetune", help="PPO fine-tuning of a distilled student (its task, DR, BC regulariser)")
    ft.add_argument("--init", required=True, help="rl_policy.npz / rl_checkpoint.npz of the student")
    ft.add_argument("--name", default="rl_walker_ft")
    ft.add_argument("--seconds", type=float, default=1800.0)
    ft.add_argument("--iterations", type=int, default=100000)
    ft.add_argument("--envs", type=int, default=192)
    ft.add_argument("--threads", type=int, default=6)
    ft.add_argument("--lr", type=float, default=2e-4)
    ft.add_argument("--std", type=float, default=0.15)
    ft.add_argument("--bc", type=float, default=1.0)
    ft.add_argument("--bc-final", type=float, default=0.05)
    ft.add_argument("--bc-iters", type=int, default=1500)
    ft.add_argument("--sym", type=float, default=0.1)
    ft.add_argument("--warmup", type=int, default=50)
    ft.add_argument("--kl", type=float, default=0.02, help="KL target of the adaptive learning rate (a small "
                    "exploration std makes the KL of a given mean change large, so 0.01 would stall the lr)")
    ft.add_argument("--seed", type=int, default=0)
    ft.add_argument("--runs", default="runs")
    ev = sub.add_parser("eval", help="goal reaching + velocity tracking through RLController")
    ev.add_argument("--policy", required=True)
    ev.add_argument("--trials", type=int, default=10)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--out", default=None, help="write the results as JSON here")
    be = sub.add_parser("bench", help="vectorised env throughput")
    be.add_argument("--threads", type=int, nargs="+", default=[4, 8])
    be.add_argument("--envs", type=int, default=256)
    a = ap.parse_args(argv)

    from ..models import HumanoidParams, build_mjcf
    if a.cmd == "bench":
        for t in a.threads:
            print(json.dumps({k: (round(v) if isinstance(v, float) else v) for k, v in bench_env(t, a.envs).items()}))
        return
    if a.cmd == "eval":
        pol = RLPolicy.load(a.policy)
        hp = HumanoidParams.from_dict(pol.meta["humanoid"]) if pol.meta.get("humanoid") else HumanoidParams()
        model = mujoco.MjModel.from_xml_string(build_mjcf(hp))
        with blas_threads(1):
            res = _print_eval(a.policy, model, a.trials, a.seed)
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1))
        return
    if a.cmd in ("train", "finetune"):
        if a.cmd == "train":
            cfg = PPOConfig(name=a.name, iterations=a.iterations, max_seconds=a.seconds, num_envs=a.envs,
                            threads=a.threads, seed=a.seed)
            trainer = PPOTrainer(cfg, HumanoidParams(), VelTaskConfig(), runs_dir=a.runs, init_checkpoint=a.resume)
        else:
            tcfg = _checkpoint_task(a.init)
            if tcfg is None:
                raise SystemExit(f"{a.init} has no task configuration")
            if tcfg.clock == "steps":
                # a footstep-walker student: its stance height is the target, and it steps sideways / turns
                # with hip roll / yaw, which a hip-position penalty would fight
                tcfg.base_height_frac = 1.0
                tcfg.rewards = {**tcfg.rewards, "hip_pos": 0.0}
            cfg = PPOConfig(name=a.name, iterations=a.iterations, max_seconds=a.seconds, num_envs=a.envs,
                            threads=a.threads, seed=a.seed, learning_rate=a.lr, gamma=0.995, lam=0.95,
                            entropy_coef=0.0, desired_kl=a.kl, reset_std=a.std, critic_warmup_iters=a.warmup,
                            bc_coef=a.bc, bc_coef_final=a.bc_final, bc_anneal_iters=a.bc_iters, sym_coef=a.sym,
                            blas_threads=4, freeze_normalizer=True, save_interval=25)
            trainer = PPOTrainer(cfg, HumanoidParams(), tcfg, runs_dir=a.runs, init_checkpoint=a.init)
        trainer.start()
        try:
            shown = 0
            while trainer._thread.is_alive():
                time.sleep(2.0)
                with trainer._lock:
                    rows = trainer.history[shown:]
                for r in rows:
                    print(f"it {r['gen']:5d}  track_lin {r['track_lin']:.3f}  track_ang {r['track_ang']:.3f}  "
                          f"ep_len {r['ep_len_s']:5.2f}s  fell {r['fell']:.2f}  return {r['center']:7.2f}  "
                          f"std {r['std']:.3f}  kl {r['kl']:.4f}  lr {r['lr']:.1e}  bc {r.get('bc_loss', 0):.4f}  "
                          f"{r['env_sps']:6.0f} env-steps/s", flush=True)
                shown += len(rows)
        except KeyboardInterrupt:
            trainer.stop()
        trainer.join()
        st = trainer.status()
        print(f"{st['state']} {st['error']}  policy: {st['policy']}")
        return
    model = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams()))
    tcfg = None
    if a.teacher == "walker":
        params = {} if a.step_forward is None else {"step_forward": a.step_forward}
        if a.walker_params:
            params.update(json.loads(a.walker_params))
        factory = walker_teacher_factory(a.walker_source, **params)
        if factory is None:
            raise SystemExit("control/walker.py WalkController is not available")
    else:
        tcfg = VelTaskConfig()
        gains = (0.5, 0.05) if a.teacher == "balance" else (0.0, 0.0)
        factory = lambda m: StandingTeacher(m, tcfg, *gains)  # noqa: E731
    run_dir = Path(a.runs) / a.name
    dcfg = DistillConfig(num_envs=a.envs, rounds=a.rounds, episode_seconds=a.seconds, workers=a.workers,
                         grad_steps=a.grad_steps, batch_size=a.batch, lr=a.lr, lr_final=a.lr_final,
                         keep_every=a.keep_every, buffer_size=a.buffer, beta_decay=a.beta_decay, resume=a.resume,
                         stand_prob=a.stand_prob, goal_prob=a.goal_prob, keep_rounds=a.keep_rounds,
                         seed=a.seed, name=a.name, **({"hidden": a.hidden} if a.hidden else {}))
    _, report = distill(model, factory, tcfg, dcfg, run_dir=run_dir,
                        model_meta={"humanoid": HumanoidParams().to_dict(), "custom_model": False},
                        callback=lambda r: print(json.dumps(r), flush=True))
    print(f"student: {run_dir / 'rl_policy.npz'}")


if __name__ == "__main__":
    main()
