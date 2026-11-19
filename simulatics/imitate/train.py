"""Train, save, evaluate and run motion-tracking policies (DeepMimic-style, PPO on CPU).

* `ImitationTrainer`: PPO (train/ppo.py, unchanged) on `ImitationEnv` in a background thread, with the same
  lifecycle and `status()` shape as `PPOTrainer`, so the GUI's training chart works unchanged:
      start(), stop(), join(), status() -> {"state", "error", "name", "algo": "imitation", "generation",
      "generations", "best", "total_steps", "history": [{gen, best, mean, center, sps, ...}], "policy"}
  History rows add tracking diagnostics: reward_per_step (1 = perfect), r_pose / r_vel / r_end_eff /
  r_com / r_root (weighted DeepMimic terms per step), joint_err_deg, ee_err_m, root_err_m, ep_len_s, fell.
  Output: runs/<name>/imitation_policy.npz (actor + observation normaliser + the reference motion +
  task config), imitation_checkpoint.npz (critic too, for resuming), config.json, history.json.
* `ImitationPolicy`: the deployable file (format "simulatics-imitation-policy", deliberately not the
  velocity-policy format, so the GUI's RL policy list does not pick it up as a walker).
* `ImitationController`: plays a trained policy in the live session (`Controller` protocol, name
  "imitation"): reset() puts the robot in the clip's first frame where it stands, step() writes the
  abdomen, leg and arm servo targets (wrists / grippers get the reference targets).
* `evaluate_imitation`: deterministic rollouts -> time before falling, joint / end-effector / root error;
  `zero_action=True` gives the feed-forward-only baseline the policy has to beat.

Hyper-parameters (`imitation_ppo_config`): PPO with the common legged-locomotion settings (adaptive KL learning rate, clip 0.2, GAE
lambda 0.95) with discount 0.99 (DeepMimic used 0.95 at 30 Hz; at 50 Hz that weights a fall 1 s ahead by
only 0.08, while 0.99 gives 0.6 - the usual choice of current trackers), entropy 0.005, exploration std
0.3 x action scale 0.2 rad (larger noise topples the walk within ~1.5 s). The reward PPO sees is scaled by
dt (TrackConfig.reward_scale) so the clipped value loss can follow the returns.

    python -m simulatics.imitate.train train --motion walk --name imitate_walk --minutes 35 --threads 4
    python -m simulatics.imitate.train eval runs/imitate_walk/imitation_policy.npz
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

from ..train.nn import GaussianPolicy, RunningMeanStd, load_npz, save_npz
from ..train.ppo import PPO, PPOConfig, blas_threads, run_ppo
from .library import load_motion
from .motion import Motion, heading_pose, planar_between
from .track_env import TERMS, ImitationEnv
from .track_task import TrackConfig, TrackTask

POLICY_FORMAT = "simulatics-imitation-policy"


def _finite(v, default=0.0):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return f if np.isfinite(f) else default


def _clean(row: dict) -> dict:
    return {k: (_finite(v) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}


def imitation_ppo_config(**kw) -> PPOConfig:
    """PPOConfig with the tracking defaults (see the module docstring); keyword arguments override."""
    base = dict(num_envs=256, steps_per_env=24, iterations=1500, gamma=0.99, lam=0.95, entropy_coef=0.005,
                init_noise_std=0.3, learning_rate=1e-3, actor_hidden=[256, 128, 64],
                critic_hidden=[256, 128, 64], threads=4, blas_threads=4, collect_blas_threads=1,
                name="imitate", save_interval=10)
    base.update(kw)
    return PPOConfig(**base)


def build_model(base=None, model_xml: str | None = None) -> mujoco.MjModel:
    from ..models import HumanoidParams, build_mjcf
    if model_xml:
        return mujoco.MjModel.from_xml_path(model_xml)
    return mujoco.MjModel.from_xml_string(build_mjcf(base or HumanoidParams()))


# =========================================================================================================
# Policy file
# =========================================================================================================
class ImitationPolicy:
    """Actor + observation normaliser + reference motion + metadata. `act` returns the mean action."""

    def __init__(self, actor: GaussianPolicy, obs_norm: RunningMeanStd | None, motion: Motion, meta: dict):
        self.actor, self.obs_norm, self.motion, self.meta = actor, obs_norm, motion, dict(meta)

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

    def task_config(self) -> TrackConfig:
        return TrackConfig.from_dict(self.meta.get("task", {}))

    def save(self, path: str | Path) -> Path:
        arrays = {**self.actor.to_arrays("actor_"), **self.motion.to_arrays("motion_")}
        if self.obs_norm is not None:
            arrays.update(self.obs_norm.to_arrays("obs_"))
        meta = {**self.meta, "format": POLICY_FORMAT, "version": 1, "motion_header": self.motion.header()}
        return save_npz(path, arrays, meta)

    @classmethod
    def load(cls, path: str | Path) -> "ImitationPolicy":
        arrays, meta = load_npz(path)
        if meta.get("format") != POLICY_FORMAT:
            raise ValueError(f"{path} is not a Simulatics imitation policy (format {meta.get('format')!r})")
        actor = GaussianPolicy.from_arrays(arrays, meta.get("activation", "elu"))
        norm = RunningMeanStd.from_arrays(arrays, "obs_") if "obs_mean" in arrays else None
        motion = Motion.from_arrays(arrays, meta["motion_header"], prefix="motion_")
        return cls(actor, norm, motion, meta)


def is_imitation_policy(path: str | Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as z:
            return "meta" in z.files and json.loads(str(z["meta"])).get("format") == POLICY_FORMAT
    except Exception:
        return False


def _policy_meta(task: TrackTask, model_meta: dict, **extra) -> dict:
    return {"algo": "ppo-imitation", "task": task.cfg.to_dict(), "joint_names": task.joint_names,
            "actuator_names": task.actuator_names, "obs_dim": task.obs_dim, "act_dim": task.act_dim,
            "dt": task.dt, "motion": task.motion.name, **model_meta, **extra}


# =========================================================================================================
# Trainer
# =========================================================================================================
class ImitationTrainer:
    """PPO motion tracking in a background thread (status shape of train.ppo.PPOTrainer)."""

    def __init__(self, cfg: PPOConfig | None = None, motion: str | Path | Motion = "walk",
                 task_cfg: TrackConfig | None = None, base=None, model_xml: str | None = None,
                 runs_dir: str | Path = "runs", init_checkpoint: str | Path | None = None):
        from ..models import HumanoidParams
        self.cfg = cfg or imitation_ppo_config()
        self.motion = motion if isinstance(motion, Motion) else load_motion(motion)
        self.task_cfg = task_cfg or TrackConfig()
        self.base = base or HumanoidParams()
        self.model_xml = model_xml
        self.init_checkpoint = init_checkpoint
        self.run_dir = Path(runs_dir) / self.cfg.name
        self.policy_path = self.run_dir / "imitation_policy.npz"
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
        self._thread = threading.Thread(target=self._run_safe, name="imitation-trainer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def status(self) -> dict:
        with self._lock:
            return {"state": self.state, "error": self.error, "name": self.cfg.name, "algo": "imitation",
                    "motion": self.motion.name,
                    "generation": len(self.history), "generations": self.cfg.iterations,
                    "best": None if not np.isfinite(self.best_fitness) else float(self.best_fitness),
                    "total_steps": int(self.total_steps), "history": self.history[-400:],
                    "policy": str(self.policy_path) if self.policy_path.exists() else None}

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
    def run(self) -> None:
        cfg = self.cfg
        model = build_model(self.base, self.model_xml)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        model_meta = {"humanoid": None if self.model_xml else self.base.to_dict(),
                      "custom_model": self.model_xml is not None}
        rng = np.random.default_rng(cfg.seed)
        with ImitationEnv(model, self.motion, self.task_cfg, cfg.num_envs, cfg.threads, cfg.seed) as env:
            algo = PPO(env.obs_dim, env.critic_obs_dim, env.act_dim, cfg, rng)
            if self.init_checkpoint:
                arrays, _ = load_npz(self.init_checkpoint)
                algo.load_checkpoint(arrays)
            (self.run_dir / "config.json").write_text(json.dumps(
                {"ppo": cfg.to_dict(), "task": self.task_cfg.to_dict(), "motion": self.motion.summary(),
                 "humanoid": model_meta["humanoid"], "custom_model": self.model_xml,
                 "init_checkpoint": str(self.init_checkpoint) if self.init_checkpoint else None}, indent=2))
            t_start = time.perf_counter()

            def save(it: int) -> None:
                meta = _policy_meta(env.task, model_meta, iteration=it, activation=cfg.activation,
                                    reward_per_step=self.history[-1]["reward_per_step"] if self.history else None)
                ImitationPolicy(algo.actor, algo.obs_norm if cfg.normalize_obs else None, env.task.motion,
                                meta).save(self.policy_path)
                save_npz(self.run_dir / "imitation_checkpoint.npz",
                         {**algo.checkpoint_arrays(), **env.task.motion.to_arrays("motion_")},
                         {**meta, "ppo": cfg.to_dict(), "motion_header": env.task.motion.header()})
                (self.run_dir / "history.json").write_text(json.dumps(self.history))

            def on_iter(it: int, e: dict, _algo: PPO) -> None:
                ep = e["episodes"]
                recent = list(env.completed)
                mean_of = lambda f: float(np.mean([f(c) for c in recent])) if recent else 0.0  # noqa: E731
                rolling = mean_of(lambda c: c["return"])
                mean = ep.get("return", rolling) if ep else rolling
                best = (max((c["return"] for c in recent[-int(ep.get("count", 0)):]), default=mean)
                        if ep else mean)
                row = {"gen": it + 1, "best": float(best), "mean": float(mean), "center": rolling,
                       "sps": e["sps"], "env_sps": e["env_sps"], "seconds": e["collect_s"] + e["learn_s"],
                       "wall_s": time.perf_counter() - t_start,
                       "collect_s": e["collect_s"], "learn_s": e["learn_s"],
                       "reward_per_step": mean_of(lambda c: c["reward_per_step"]),
                       "ep_len_s": mean_of(lambda c: c["length_s"]), "fell": mean_of(lambda c: float(c["fell"])),
                       "joint_err_deg": float(np.degrees(mean_of(lambda c: c["terms"]["joint_rmse"]))),
                       "ee_err_m": mean_of(lambda c: c["terms"]["ee_err"]),
                       "root_err_m": mean_of(lambda c: c["terms"]["root_dist"]),
                       **{f"r_{k}": mean_of(lambda c, k=k: c["terms"][k]) for k in TERMS},
                       "step_reward": e["step_reward"] / max(self.task_cfg.reward_scale, 1e-12),
                       "kl": e["kl"], "lr": e["lr"], "std": e["std"], "value_loss": e["value_loss"],
                       "surrogate": e["surrogate"]}
                row = _clean(row)
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
# Evaluation
# =========================================================================================================
def evaluate_imitation(policy: "ImitationPolicy | str | Path | None", model: mujoco.MjModel | None = None,
                       seconds: float = 10.0, num_envs: int = 16, nthread: int = 3, rsi: bool = True, seed: int = 0,
                       zero_action: bool = False, motion: Motion | None = None,
                       task_cfg: TrackConfig | None = None) -> dict:
    """Deterministic rollouts (mean action, no reset noise) of one episode per env.

    rsi=True starts the envs at evenly spread frames of the clip, else all at frame 0. zero_action=True
    (or policy None) plays the feed-forward targets alone. Returns {"fell_frac", "survival_s" (mean),
    "survival_min_s", "joint_err_deg" (RMS over controlled joints), "ee_err_m" (hands + feet relative to
    the root), "root_err_m", "reward_per_step", "seconds", "envs"}; errors are averaged over the steps
    before a fall."""
    if isinstance(policy, (str, Path)):
        policy = ImitationPolicy.load(policy)
    if isinstance(policy, ImitationPolicy):
        motion = motion or policy.motion
        task_cfg = task_cfg or policy.task_config()
    pol = None if (policy is None or zero_action) else policy
    if motion is None:
        raise ValueError("evaluate_imitation needs a policy or a motion")
    model = model or build_model()
    cfg = TrackConfig.from_dict((task_cfg or TrackConfig()).to_dict())
    cfg.init_joint_noise = cfg.init_vel_noise = 0.0
    cfg.episode_seconds = float(seconds)
    cfg.push_interval_s = 0.0
    with ImitationEnv(model, motion, cfg, num_envs, nthread, seed) as env, blas_threads(1):
        obs, _ = env.reset()
        if rsi:
            last = env.task.ref.n if env.task.ref.loop else max(1, env.task.ref.n - 1)
            env.ts.frame[:] = np.linspace(0, last, num_envs, endpoint=False).astype(np.int64)
        else:
            env.ts.frame[:] = 0
        env.set_reference_state()
        obs, _ = env.observe()
        alive = np.ones(num_envs, bool)
        surv = np.full(num_envs, float(seconds))
        fell = np.zeros(num_envs, bool)
        sums = {k: np.zeros(num_envs) for k in ("joint_rmse", "ee_err", "root_dist", "reward")}
        count = np.zeros(num_envs)
        steps = int(round(seconds / env.dt))
        for k in range(steps):
            a = np.zeros((num_envs, env.act_dim)) if pol is None else pol.act(obs)
            obs, _, _, done, info = env.step(a)
            err = info["errors"]
            m = alive & ~info["terminated"]
            for key in ("joint_rmse", "ee_err", "root_dist"):
                sums[key][m] += err[key][m]
            sums["reward"][m] += info["reward_raw"][m]
            count[m] += 1
            newly = alive & info["terminated"]
            surv[newly] = (k + 1) * env.dt
            fell |= newly
            alive &= ~done
            if not alive.any():
                break
        c = np.maximum(count, 1)
        return {"envs": num_envs, "seconds": float(seconds), "fell_frac": float(fell.mean()),
                "survival_s": float(surv.mean()), "survival_min_s": float(surv.min()),
                "joint_err_deg": float(np.degrees(np.mean(sums["joint_rmse"] / c))),
                "ee_err_m": float(np.mean(sums["ee_err"] / c)), "root_err_m": float(np.mean(sums["root_dist"] / c)),
                "reward_per_step": float(np.mean(sums["reward"] / c)), "zero_action": pol is None,
                "motion": motion.name}


# =========================================================================================================
# Controller for the live session
# =========================================================================================================
class ImitationController:
    """Plays a trained tracking policy (name "imitation"); implements `control.base.Controller`.

    reset() writes the clip's first frame (anchored at the robot's current x, y, yaw) into data.qpos /
    qvel, like reference-state initialisation in training, and restarts the clip. step() writes every
    actuator: the policy's targets for the abdomen, legs, shoulders and elbows, the reference targets for
    wrists and grippers. Paths from the planner are ignored (the reference motion decides where it goes).
    A clip that does not loop holds its last frame and reports "done"."""
    name = "imitation"

    def __init__(self, model: mujoco.MjModel, policy: "ImitationPolicy | str | Path", motion: Motion | None = None):
        self.policy = policy if isinstance(policy, ImitationPolicy) else ImitationPolicy.load(policy)
        cfg = self.policy.task_config()
        self.motion = motion or self.policy.motion
        self.task = TrackTask(model, self.motion, cfg)
        names = self.policy.meta.get("joint_names")
        if names is not None and list(names) != self.task.joint_names:
            raise ValueError(f"policy was trained for joints {names}, this robot has {self.task.joint_names}")
        if self.policy.obs_dim != self.task.obs_dim or self.policy.act_dim != self.task.act_dim:
            raise ValueError("policy observation/action size does not match this robot and motion")
        self.control_dt = self.task.dt
        self.ts = self.task.make_state(1)
        self._mode, self._msg = "stand", "not started"
        self._err: dict = {}

    def reset(self, model, data) -> None:
        task, qa = self.task, self.task.qa
        q0 = task.ref.qpos[0]
        h0 = heading_pose(q0[qa:qa + 3], q0[qa + 3:qa + 7])
        hr = heading_pose(data.qpos[qa:qa + 3], data.qpos[qa + 3:qa + 7])
        self.ts = task.make_state(1)
        self.ts.origin[0] = planar_between(h0, hr)
        f = task.ref.frames(np.array([0]), self.ts.origin, fields=("qpos", "qvel", "ctrl"))
        data.qpos[:] = f["qpos"][0]
        data.qvel[:] = f["qvel"][0]
        data.ctrl[:] = f["ctrl"][0]
        mujoco.mj_forward(model, data)
        self._mode, self._msg, self._err = "walk", f"tracking '{self.motion.name}'", {}

    def set_path(self, path_xy, final_yaw=None) -> None:
        pass   # the reference motion decides where the robot goes

    def step(self, model, data) -> None:
        task, ts = self.task, self.ts
        qpos, qvel = data.qpos[None].copy(), data.qvel[None].copy()
        if not (np.isfinite(qpos).all() and np.isfinite(qvel).all()):
            self._mode, self._msg = "fallen", "simulation state is not finite"
            return
        err = task.errors(qpos, qvel, ts)
        fk = err.pop("fk")
        self._err = {k: float(v[0]) for k, v in err.items()}
        cfg = task.cfg
        if self._mode != "fallen" and (err["dz"][0] > cfg.term_height or err["tilt"][0] > np.radians(cfg.term_tilt_deg)):
            self._mode, self._msg = "fallen", "robot fell (left the reference)"
        obs, _ = task.observe(qpos, qvel, ts, fk=fk)
        a = np.clip(self.policy.act(obs), -cfg.clip_actions, cfg.clip_actions)
        data.ctrl[:] = task.action_to_ctrl(a, ts)[0]
        ts.last_action[:] = a
        if not task.ref.finished(ts.frame + 1)[0] or task.ref.loop:
            ts.frame += 1
        elif self._mode != "fallen":
            self._mode, self._msg = "done", f"'{self.motion.name}' finished"
        ts.t += task.dt

    def status(self) -> dict:
        e = self._err
        return {"mode": self._mode, "message": self._msg, "motion": self.motion.name,
                "phase": float(self.task.ref.phase(self.ts.frame)[0]), "time_s": round(float(self.ts.t[0]), 3),
                "joint_err_deg": float(np.degrees(e["joint_rmse"])) if e else None,
                "ee_err_m": e.get("ee_err") if e else None, "root_err_m": e.get("root_dist") if e else None}


# =========================================================================================================
# Command line
# =========================================================================================================
def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m simulatics.imitate.train", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train", help="PPO motion tracking")
    tr.add_argument("--motion", default="walk", help="library clip name or motion .npz")
    tr.add_argument("--name", default="imitate_walk")
    tr.add_argument("--minutes", type=float, default=30.0, help="wall-clock budget")
    tr.add_argument("--iterations", type=int, default=100000)
    tr.add_argument("--envs", type=int, default=256)
    tr.add_argument("--threads", type=int, default=4)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--resume", default=None, help="imitation_checkpoint.npz to continue from")
    tr.add_argument("--runs", default="runs")
    ev = sub.add_parser("eval", help="deterministic evaluation of a policy against the feed-forward baseline")
    ev.add_argument("policy")
    ev.add_argument("--seconds", type=float, default=10.0)
    ev.add_argument("--envs", type=int, default=16)
    ev.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(argv)
    if a.cmd == "eval":
        pol = ImitationPolicy.load(a.policy)
        for zero in (True, False):
            for rsi in (False, True):
                r = evaluate_imitation(pol, seconds=a.seconds, num_envs=a.envs, nthread=a.threads, rsi=rsi,
                                       zero_action=zero)
                print(json.dumps({"rsi": rsi, **{k: (round(v, 4) if isinstance(v, float) else v)
                                                 for k, v in r.items()}}), flush=True)
        return
    cfg = imitation_ppo_config(name=a.name, iterations=a.iterations, max_seconds=a.minutes * 60.0,
                               num_envs=a.envs, threads=a.threads, blas_threads=min(4, a.threads), seed=a.seed)
    trainer = ImitationTrainer(cfg, a.motion, TrackConfig(), runs_dir=a.runs, init_checkpoint=a.resume)
    trainer.start()
    shown = 0
    try:
        while trainer._thread.is_alive():
            trainer.join(5.0)
            with trainer._lock:
                rows = trainer.history[shown:]
            for r in rows:
                print(f"it {r['gen']:5d}  {r['wall_s']:7.0f}s  r/step {r['reward_per_step']:.3f}  "
                      f"ep_len {r['ep_len_s']:5.2f}s  fell {r['fell']:.2f}  jerr {r['joint_err_deg']:5.2f}deg  "
                      f"ee {r['ee_err_m']:.3f}m  root {r['root_err_m']:.3f}m  std {r['std']:.2f}  "
                      f"{r['env_sps']:6.0f} env-steps/s", flush=True)
            shown += len(rows)
    except KeyboardInterrupt:
        trainer.stop()
        trainer.join()
    st = trainer.status()
    print(f"{st['state']} {st['error']}  policy: {st['policy']}", flush=True)


if __name__ == "__main__":
    main()
