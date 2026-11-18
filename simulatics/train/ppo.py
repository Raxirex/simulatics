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
* **Teacher-student distillation**: behaviour cloning followed by **DAgger** (Ross, Gordon, Bagnell 2011):
  roll out a mixture of teacher and student (P(teacher) = beta, decaying per round), label every visited
  state with the teacher's action, aggregate and refit. This is how Lee et al. 2020, Kumar et al. 2021
  and many later humanoid papers obtain deployable students; here the teacher can be any `Controller` (e.g. the model-based WalkController) and the student
  is the same MLP policy PPO trains, so it can be fine-tuned with PPO afterwards.

Public pieces
-------------
PPOConfig, PPO (the algorithm), run_ppo (generic loop over any vectorised env), PPOTrainer (background
thread with the same status shape as train/evolve.py's Trainer, for the GUI), RLPolicy (the deployable
artifact, runs/<name>/rl_policy.npz), evaluate_policy, StandingTeacher, DistillConfig, distill, and
RLController (the live-session Controller, name "rl").
"""
from __future__ import annotations

import contextlib
import json
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
from ..sim.vel_task import VelocityTask, VelTaskConfig, STATE_SPEC
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
        flat = lambda x: x.reshape(size, *x.shape[2:])
        data = [flat(x) for x in (self.obs, self.cobs, self.actions, self.logp, self.mu, self.values,
                                  self.returns, self.advantages)]
        perm = rng.permutation(size)
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

    # ---- data collection -----------------------------------------------------------------------------
    def init_storage(self, n: int) -> None:
        self.storage = RolloutStorage(self.cfg.steps_per_env, n, self.actor.obs_dim, self.cobs_norm.mean.size,
                                      self.actor.act_dim)

    def _norm(self, obs, cobs, update: bool):
        if not self.cfg.normalize_obs:
            return obs, cobs
        if update:
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

    # ---- learning ------------------------------------------------------------------------------------
    def update(self) -> dict:
        cfg, s = self.cfg, self.storage
        stats = {"surrogate": 0.0, "value_loss": 0.0, "kl": 0.0, "clip_frac": 0.0, "grad_norm": 0.0}
        n_updates = 0
        std_old_all = np.exp(self.actor.log_std).copy()  # log_std is state-independent: same for every sample
        for _ in range(cfg.learning_epochs):
            for obs, cobs, act, logp_old, mu_old, v_old, ret, adv in s.minibatches(cfg.num_minibatches, self.rng):
                surr, g_actor, info = self.actor.ppo_loss(obs, act, logp_old, adv, cfg.clip_param, cfg.entropy_coef)
                kl = float(np.mean(GaussianPolicy.kl(mu_old, std_old_all, info["mu"], self.actor.std)))
                if cfg.schedule == "adaptive" and cfg.desired_kl > 0:
                    if kl > 2.0 * cfg.desired_kl:
                        self.lr = max(1e-5, self.lr / 1.5)
                    elif 0.0 < kl < cfg.desired_kl / 2.0:
                        self.lr = min(1e-2, self.lr * 1.5)
                    self.opt.lr = self.lr
                vloss, g_critic = self.critic.loss(cobs, ret, v_old,
                                                   cfg.clip_param if cfg.use_clipped_value_loss else None)
                grads = g_actor + [g * cfg.value_loss_coef for g in g_critic]
                grads, gnorm = clip_grad_norm(grads, cfg.max_grad_norm)
                self.opt.step(grads)
                np.clip(self.actor.log_std, np.log(0.01), np.log(5.0), out=self.actor.log_std)
                stats["surrogate"] += surr
                stats["value_loss"] += vloss
                stats["kl"] += kl
                stats["clip_frac"] += info["clip_frac"]
                stats["grad_norm"] += gnorm
                n_updates += 1
        s.step = 0
        out = {k: v / max(1, n_updates) for k, v in stats.items()}
        out.update(lr=self.lr, std=float(self.actor.std.mean()))
        return out

    # ---- export --------------------------------------------------------------------------------------
    def policy(self, meta: dict) -> "RLPolicy":
        norm = self.obs_norm if self.cfg.normalize_obs else None
        return RLPolicy(self.actor, norm, {**meta, "activation": self.cfg.activation})

    def checkpoint_arrays(self) -> dict[str, np.ndarray]:
        return {**self.actor.to_arrays("actor_"), **self.critic.net.to_arrays("critic_"),
                **self.obs_norm.to_arrays("obs_"), **self.cobs_norm.to_arrays("cobs_")}

    def load_checkpoint(self, arrays) -> None:
        act = self.cfg.activation
        actor = GaussianPolicy.from_arrays(arrays, act)
        if actor.mu.sizes != self.actor.mu.sizes:
            raise ValueError(f"checkpoint actor {actor.mu.sizes} does not match {self.actor.mu.sizes}")
        for p, q in zip(self.actor.params, actor.params):
            p[...] = q
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
        cat = lambda xs: np.concatenate(xs) if xs else np.array([np.nan])
        return {"fall_rate": float(1.0 - alive.mean()), "lin_vel_error": float(np.sqrt(np.mean(cat(err_v)))),
                "yaw_rate_error": float(np.sqrt(np.mean(cat(err_w)))), "mean_vx": float(np.mean(cat(vx))),
                "height_end": float(qpos[alive, 2].mean()) if alive.any() else 0.0}


# =========================================================================================================
# Trainer (background thread, GUI-compatible status)
# =========================================================================================================
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

    def run(self) -> None:
        cfg = self.cfg
        model = self.build_model()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(
            {"ppo": cfg.to_dict(), "task": self.task_cfg.to_dict(),
             "humanoid": None if self.model_xml else self.base.to_dict(), "custom_model": self.model_xml}, indent=2))
        model_meta = {"humanoid": None if self.model_xml else self.base.to_dict(),
                      "custom_model": self.model_xml is not None}
        rng = np.random.default_rng(cfg.seed)
        with VecEnv(model, self.task_cfg, cfg.num_envs, cfg.threads, cfg.seed) as env:
            algo = PPO(env.obs_dim, env.critic_obs_dim, env.act_dim, cfg, rng)
            if self.init_checkpoint:
                arrays, _ = load_npz(self.init_checkpoint)
                algo.load_checkpoint(arrays)
            recent_best: deque[float] = deque(maxlen=100)

            def save(it: int) -> None:
                meta = _policy_meta(env.task, model_meta, "ppo", iteration=it,
                                    mean_return=self.history[-1]["center"] if self.history else None)
                algo.policy(meta).save(self.policy_path)
                save_npz(self.run_dir / "rl_checkpoint.npz", algo.checkpoint_arrays(), {**meta, "ppo": cfg.to_dict()})
                (self.run_dir / "history.json").write_text(json.dumps(self.history))

            def on_iter(it: int, e: dict, _algo: PPO) -> None:
                ep = e["episodes"]
                recent = list(env.completed)
                if ep:
                    recent_best.append(ep["return"])
                rolling = float(np.mean([c["return"] for c in recent])) if recent else 0.0
                terms = lambda k: float(np.mean([c["terms"].get(k, 0.0) for c in recent])) if recent else 0.0
                mean = ep.get("return", rolling)
                best = max((c["return"] for c in recent[-int(ep.get("count", 0)):]), default=mean) if ep else mean
                row = {"gen": it + 1, "best": float(best), "mean": float(mean), "center": rolling,
                       "sps": e["sps"], "env_sps": e["env_sps"], "seconds": e["collect_s"] + e["learn_s"],
                       "collect_s": e["collect_s"], "learn_s": e["learn_s"],
                       "track_lin": terms("tracking_lin_vel"), "track_ang": terms("tracking_ang_vel"),
                       "ep_len_s": float(np.mean([c["length_s"] for c in recent])) if recent else 0.0,
                       "fell": float(np.mean([c["fell"] for c in recent])) if recent else 0.0,
                       "step_reward": e["step_reward"], "kl": e["kl"], "lr": e["lr"], "std": e["std"],
                       "value_loss": e["value_loss"], "surrogate": e["surrogate"]}
                with self._lock:
                    self.total_steps += cfg.steps_per_env * cfg.num_envs * self.task_cfg.decimation
                    self.history.append(row)
                    if recent and rolling > self.best_fitness:
                        self.best_fitness = rolling
                if (it + 1) % max(1, cfg.save_interval) == 0:
                    save(it + 1)

            run_ppo(env, cfg, algo=algo, callback=on_iter, should_stop=self._stop.is_set)
            save(len(self.history))


# =========================================================================================================
# Teacher-student distillation (behaviour cloning + DAgger)
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


def walker_teacher_factory() -> Callable[[mujoco.MjModel], object] | None:
    """Factory for the model-based WalkController (control/walker.py) as a teacher, or None if unavailable."""
    try:
        from ..control.walker import WalkController
    except ImportError:
        return None
    return lambda model: WalkController(model)


@dataclass
class DistillConfig:
    num_envs: int = 8
    rounds: int = 4                  # round 0 = behaviour cloning on teacher rollouts, then DAgger
    episode_seconds: float = 5.0
    beta_decay: float = 0.3          # P(execute teacher action) = beta_decay ** round
    epochs: int = 30                 # passes over the aggregated dataset per round
    batch_size: int = 256
    lr: float = 1e-3
    hidden: list[int] = field(default_factory=lambda: [128, 128])
    activation: str = "elu"
    init_std: float = 0.3            # exploration std stored with the student (for later PPO fine-tuning)
    commands: str = "auto"           # "auto" (sample if the teacher has set_velocity) | "zero" | "sample"
    seed: int = 0
    name: str = "distill"


def _touch_contacts(task: VelocityTask, datas: list[mujoco.MjData]) -> np.ndarray:
    if task.touch_adr.size < 2:
        return np.zeros((len(datas), 2), bool)
    return np.stack([d.sensordata[task.touch_adr] > task.cfg.contact_force for d in datas])


def distill(model: mujoco.MjModel, teacher_factory: Callable[[mujoco.MjModel], object],
            task_cfg: VelTaskConfig | None = None, cfg: DistillConfig | None = None,
            run_dir: str | Path | None = None, model_meta: dict | None = None,
            callback: Callable[[dict], None] | None = None) -> tuple[RLPolicy, list[dict]]:
    """Distil any teacher controller into an MLP student with behaviour cloning + DAgger.

    teacher_factory(model) -> object with reset(model, data) and step(model, data) writing data.ctrl
    (the `Controller` protocol); optional set_velocity(vx, vy, wz) makes the student command-conditioned.
    Each environment gets its own teacher instance (teachers keep state, e.g. a gait phase) and the
    teacher is ticked at its own `control_dt` (e.g. 200 Hz for WalkController) on whatever state the
    *executed* policy produced, while the student acts at the task rate (50 Hz). The label of a student
    step is the teacher's output at the state the student observed. Executing a mixture (teacher with
    probability beta, decaying per round) and labelling every visited state is DAgger (Ross et al. 2011):
    the student learns what to do in the states its own mistakes lead to. Pushes from the task config
    (when `randomize` is on) perturb the robots so recovery behaviour is demonstrated too. Arms are held
    at their default targets during execution, as in deployment.

    Limitation: the student only sees proprioception, the command and its own gait clock; a teacher with
    hidden internal state (a footstep plan) can only be imitated as far as that state is predictable from
    those inputs (`VelTaskConfig.history` helps).

    Returns (student policy, per-round report). With run_dir set, writes run_dir/rl_policy.npz.
    """
    cfg = cfg or DistillConfig()
    tcfg = VelTaskConfig.from_dict((task_cfg or VelTaskConfig()).to_dict())
    task = VelocityTask(model, tcfg)
    rng = np.random.default_rng(cfg.seed)
    n, dt_phys = cfg.num_envs, float(model.opt.timestep)
    teachers = [teacher_factory(model) for _ in range(n)]
    tick = max(1, int(round(float(getattr(teachers[0], "control_dt", task.dt)) / dt_phys)))
    has_vel = all(hasattr(t, "set_velocity") for t in teachers)
    sample_cmd = cfg.commands == "sample" or (cfg.commands == "auto" and has_vel)
    pushes = tcfg.randomize and tcfg.push_interval_s > 0 and tcfg.push_vel > 0
    datas = [mujoco.MjData(model) for _ in range(n)]
    steps = max(1, int(round(cfg.episode_seconds / task.dt)))
    student = GaussianPolicy(task.obs_dim, task.act_dim, cfg.hidden, cfg.activation, cfg.init_std, rng)
    norm = RunningMeanStd(task.obs_dim)
    data_obs: list[np.ndarray] = []
    data_act: list[np.ndarray] = []
    report: list[dict] = []
    states = np.zeros((n, task.nstate))
    ts = task.make_state(n)
    counter = np.zeros(n, int)          # physics steps since reset (teacher tick schedule)
    latest = np.zeros((n, model.nu))    # the teacher's most recent output

    def tick_teacher(i: int) -> None:
        teachers[i].step(model, datas[i])
        latest[i] = datas[i].ctrl

    def reset(idx):
        task.reset_envs(states, ts, idx, rng)
        if not sample_cmd:
            ts.commands[idx] = 0.0
            ts.cmd_timer[idx] = 1e9
        for i in idx:
            d = datas[i]
            mujoco.mj_setState(model, d, states[i], STATE_SPEC)
            d.ctrl[:] = task.ctrl_default
            mujoco.mj_forward(model, d)
            teachers[i].reset(model, d)  # may move the robot into the teacher's own start pose
            if hasattr(teachers[i], "set_path"):
                teachers[i].set_path(None)
            if has_vel:
                teachers[i].set_velocity(*map(float, ts.commands[i]))
            counter[i] = 0

    for rnd in range(cfg.rounds):
        beta = 1.0 if rnd == 0 else cfg.beta_decay ** rnd
        reset(np.arange(n))
        falls, disagreement = 0, []
        for _ in range(steps):
            qpos = np.stack([d.qpos for d in datas])
            qvel = np.stack([d.qvel for d in datas])
            obs, _ = task.observe(qpos, qvel, ts, rng=rng)
            a_student = student.mean(norm.normalize(obs)) if rnd > 0 else None
            use_teacher = rng.random(n) < beta if rnd > 0 else np.ones(n, bool)
            ctrl_student = task.action_to_ctrl(a_student) if a_student is not None else None
            labels = np.zeros((n, model.nu))
            for i, d in enumerate(datas):
                for j in range(tcfg.decimation):
                    if counter[i] % tick == 0:
                        tick_teacher(i)
                    if j == 0:
                        labels[i] = latest[i]
                    if use_teacher[i]:  # teacher's own (higher-rate) targets, arms held at default
                        d.ctrl[:] = task.action_to_ctrl(task.ctrl_to_action(latest[i]))[0]
                    else:
                        d.ctrl[:] = ctrl_student[i]
                    mujoco.mj_step(model, d)
                    counter[i] += 1
            a_teacher = task.ctrl_to_action(labels)
            a_exec = a_teacher if a_student is None else np.where(use_teacher[:, None], a_teacher, a_student)
            if a_student is not None:
                disagreement.append(float(np.mean((a_student - a_teacher) ** 2)))
            data_obs.append(obs)
            data_act.append(a_teacher)
            qpos = np.stack([d.qpos for d in datas])
            qvel = np.stack([d.qvel for d in datas])
            _, _, terminated = task.reward(qpos, qvel, a_exec, task.action_to_ctrl(a_exec),
                                           _touch_contacts(task, datas), ts)
            if sample_cmd:
                due = np.flatnonzero(ts.cmd_timer <= 0)
                if due.size:
                    task.sample_commands(ts, due, rng)
                    for i in due:
                        teachers[i].set_velocity(*map(float, ts.commands[i]))
            if pushes:
                for i in np.flatnonzero(ts.push_timer <= 0):
                    datas[i].qvel[0:2] += rng.uniform(-tcfg.push_vel, tcfg.push_vel, 2)
                    ts.push_timer[i] = tcfg.push_interval_s
            done = np.flatnonzero(terminated | (ts.t >= tcfg.episode_seconds))
            falls += int(terminated.sum())
            if done.size:
                reset(done)

        # refit the student on the aggregated dataset (supervised regression of the teacher's action)
        x = np.concatenate(data_obs)
        y = np.concatenate(data_act)
        norm = RunningMeanStd(task.obs_dim)
        norm.update(x)
        xn = norm.normalize(x)
        opt = Adam(student.mu.params, cfg.lr)
        for _ in range(cfg.epochs):
            perm = rng.permutation(len(xn))
            for j in range(0, len(xn), cfg.batch_size):
                idx = perm[j:j + cfg.batch_size]
                _, grads = student.bc_loss(xn[idx], y[idx])
                grads, _ = clip_grad_norm(grads[:-1], 1.0)
                opt.step(grads)
        mse = float(np.mean((student.mean(xn) - y) ** 2))
        row = {"round": rnd, "beta": beta, "samples": int(len(x)), "falls": falls, "train_mse": mse,
               "rollout_disagreement": float(np.mean(disagreement)) if disagreement else None}
        report.append(row)
        if callback is not None:
            callback(row)

    meta = _policy_meta(task, model_meta or {}, "distill", teacher=type(teachers[0]).__name__,
                        distill=asdict(cfg), report=report)
    policy = RLPolicy(student, norm, {**meta, "activation": cfg.activation})
    if run_dir is not None:
        policy.save(Path(run_dir) / "rl_policy.npz")
    return policy, report


# =========================================================================================================
# Controller for the live session
# =========================================================================================================
class RLController:
    """Runs an rl_policy.npz (PPO or distilled) on the robot; implements `control.base.Controller`.

    A PathFollower turns the planner's path into (vx, vy, wz) commands, clamped to the ranges the policy
    was trained on; `set_velocity` gives direct joystick-style commands. The policy writes the leg and
    abdomen servo targets; arms (and hands) are held at their default targets.
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
        lim = lambda r: max(abs(r[0]), abs(r[1]))
        self.follower = follower or PathFollower(v_max=min(0.8, max(0.05, cfg.cmd_vx[1])),
                                                 vy_max=min(0.3, lim(cfg.cmd_vy)), wz_max=min(1.0, lim(cfg.cmd_wz)),
                                                 final_tolerance=0.1)
        self.ts = self.task.make_state(1)
        self.velocity: tuple[float, float, float] | None = None
        self._mode, self._msg = "stand", ""
        self._goal_err, self._yaw_err = None, None

    def reset(self, model, data) -> None:
        self.ts = self.task.make_state(1)
        self._mode, self._msg = "stand", ""

    def set_path(self, path_xy, final_yaw=None) -> None:
        """A non-empty path replaces any velocity command; None / an empty path means stop and stand
        (the Controller contract, as in WalkController). The session's reset and the navigator's
        stop/clear arrive here as an empty path, so they also cancel a `set_velocity` command."""
        self.follower.set_path(path_xy, final_yaw)
        self.velocity = None
        if path_xy is not None and len(np.atleast_2d(path_xy)):
            self._mode, self._msg = "walk", ""
        else:
            self._goal_err = self._yaw_err = None
            if self._mode != "fallen":
                self._mode, self._msg = "stand", ""

    def set_velocity(self, vx: float, vy: float = 0.0, wz: float = 0.0) -> None:
        """Direct velocity command (robot frame); clears any path."""
        self.follower.set_path(None)
        self.velocity = (float(vx), float(vy), float(wz))
        self._mode = "walk" if any(abs(v) > 1e-3 for v in self.velocity) else "stand"

    def _command(self, data) -> np.ndarray:
        x, y, yaw = base_pose(data)
        cmd = np.zeros(3)
        if self.follower.active:
            c = self.follower.command(x, y, yaw)
            self._goal_err, self._yaw_err = c.position_error, c.heading_error
            if c.done:
                self.follower.set_path(None)
                self._mode, self._msg = "done", f"goal reached ({c.position_error:.2f} m)"
            else:
                cmd[:] = (c.vx, c.vy, c.wz)
                self._mode = "walk"
        elif self.velocity is not None:
            cmd[:] = self.velocity
        cfg = self.task.cfg
        cmd[0] = np.clip(cmd[0], *cfg.cmd_vx)
        cmd[1] = np.clip(cmd[1], *cfg.cmd_vy)
        cmd[2] = np.clip(cmd[2], *cfg.cmd_wz)
        if np.linalg.norm(cmd[:2]) < cfg.small_cmd:
            cmd[:2] = 0.0
        return cmd

    def step(self, model, data) -> None:
        task, ts = self.task, self.ts
        ts.commands[0] = self._command(data)
        qpos, qvel = data.qpos[None].copy(), data.qvel[None].copy()
        obs, _ = task.observe(qpos, qvel, ts)
        action = self.policy.act(obs)
        action = np.clip(action, -task.cfg.clip_actions, task.cfg.clip_actions)
        data.ctrl[:] = task.action_to_ctrl(action)[0]
        ts.last_action[:] = action
        ts.last_dqd[:] = qvel[:, task.dadr]
        ts.t += task.dt
        up = -VelocityTask.base_frame(qpos, qvel)[3][0, 2]  # cos(tilt) = -(projected gravity z)
        if qpos[0, 2] < task.cfg.min_height_frac * task.z0 or up < np.cos(np.deg2rad(task.cfg.max_tilt_deg)):
            self._mode, self._msg = "fallen", "robot fell"

    def status(self) -> dict:
        out = {"mode": self._mode, "message": self._msg, "command": self.ts.commands[0].round(3).tolist()}
        if self._goal_err is not None:
            out["goal_error_m"] = float(self._goal_err)
        if self._yaw_err is not None:
            out["heading_error_deg"] = float(np.rad2deg(self._yaw_err))
        return out


# =========================================================================================================
# Command line:  python -m simulatics.train.ppo {train,distill,bench} ...
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
    di.add_argument("--envs", type=int, default=8)
    di.add_argument("--seconds", type=float, default=5.0, help="episode length per round")
    di.add_argument("--runs", default="runs")
    be = sub.add_parser("bench", help="vectorised env throughput")
    be.add_argument("--threads", type=int, nargs="+", default=[4, 8])
    be.add_argument("--envs", type=int, default=256)
    a = ap.parse_args(argv)

    from ..models import HumanoidParams, build_mjcf
    if a.cmd == "bench":
        for t in a.threads:
            print(json.dumps({k: (round(v) if isinstance(v, float) else v) for k, v in bench_env(t, a.envs).items()}))
        return
    if a.cmd == "train":
        cfg = PPOConfig(name=a.name, iterations=a.iterations, max_seconds=a.seconds, num_envs=a.envs,
                        threads=a.threads, seed=a.seed)
        trainer = PPOTrainer(cfg, HumanoidParams(), VelTaskConfig(), runs_dir=a.runs, init_checkpoint=a.resume)
        trainer.start()
        try:
            shown = 0
            while trainer._thread.is_alive():
                time.sleep(2.0)
                with trainer._lock:
                    rows = trainer.history[shown:]
                for r in rows:
                    print(f"it {r['gen']:5d}  track_lin {r['track_lin']:.3f}  track_ang {r['track_ang']:.3f}  "
                          f"ep_len {r['ep_len_s']:5.2f}s  return {r['center']:7.2f}  std {r['std']:.2f}  "
                          f"{r['env_sps']:6.0f} env-steps/s", flush=True)
                shown += len(rows)
        except KeyboardInterrupt:
            trainer.stop()
        trainer.join()
        st = trainer.status()
        print(f"{st['state']} {st['error']}  policy: {st['policy']}")
        return
    model = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams()))
    tcfg = VelTaskConfig()
    if a.teacher == "walker":
        factory = walker_teacher_factory()
        if factory is None:
            raise SystemExit("control/walker.py WalkController is not available")
    else:
        gains = (0.5, 0.05) if a.teacher == "balance" else (0.0, 0.0)
        factory = lambda m: StandingTeacher(m, tcfg, *gains)
    run_dir = Path(a.runs) / a.name
    _, report = distill(model, factory, tcfg, DistillConfig(num_envs=a.envs, rounds=a.rounds, episode_seconds=a.seconds),
                        run_dir=run_dir, model_meta={"humanoid": HumanoidParams().to_dict(), "custom_model": False},
                        callback=lambda r: print(json.dumps(r), flush=True))
    print(f"student: {run_dir / 'rl_policy.npz'}")


if __name__ == "__main__":
    main()
