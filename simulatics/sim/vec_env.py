"""Batched environments for on-policy RL, stepped in parallel by MuJoCo's C++ rollout thread pool.

The usual batched-environment pattern (Rudin et al. 2021), on the CPU: all N environments
advance in lockstep, one policy step at a time. Per step:

1. the policy's actions (N, A) become servo targets (N, nu) (`VelocityTask.action_to_ctrl`);
2. `mujoco.rollout` advances all N MuJoCo states `decimation` physics steps inside its thread pool, without
   the Python GIL. Every environment may use its own MjModel, which is how friction / mass / gain domain
   randomisation is applied (`vel_task.make_variants`);
3. rewards, terminations and time-outs are computed in numpy for the whole batch;
4. terminated or timed-out environments are reset in place (auto-reset), commands are resampled on their
   timers and random pushes are applied by editing the base velocity in the saved state;
5. actor and privileged critic observations are returned for the post-reset states.

Episode bookkeeping (return, length, per-term reward sums) is kept per environment and reported
when episodes end.
"""
from __future__ import annotations

from collections import deque

import mujoco
import numpy as np
from mujoco import rollout

from .vel_task import VelocityTask, VelTaskConfig, make_variants, normalized_props


class VecEnv:
    """N copies of the velocity-tracking task. Interface used by `train.ppo`:

        obs, critic_obs = env.reset()
        obs, critic_obs, reward, done, info = env.step(actions)   # info["time_outs"], info["episode"]
    """

    def __init__(self, model: mujoco.MjModel, cfg: VelTaskConfig | None = None, num_envs: int = 64,
                 nthread: int = 4, seed: int = 0):
        self.model = model
        self.task = VelocityTask(model, cfg)
        self.cfg = self.task.cfg
        self.num_envs = int(num_envs)
        self.nthread = max(1, int(nthread))
        self.rng = np.random.default_rng(seed)
        self.obs_dim, self.critic_obs_dim, self.act_dim = self.task.obs_dim, self.task.critic_obs_dim, self.task.act_dim
        self.dt = self.task.dt
        self.variants, props = make_variants(model, self.cfg, self.rng, self.task)
        self.props = props
        self.props_norm = normalized_props(props)
        self._nominal = [model] * self.num_envs  # rollout with skip_checks wants one model per env
        self._pool = rollout.Rollout(nthread=self.nthread)
        self._datas = [mujoco.MjData(model) for _ in range(self.nthread)]
        n, sub = self.num_envs, self.cfg.decimation
        self.state = np.zeros((n, self.task.nstate))
        self._out_state = np.zeros((n, sub, self.task.nstate))
        self._out_sens = np.zeros((n, sub, max(1, model.nsensordata)))
        self.ts = self.task.make_state(n)
        self.contact = np.zeros((n, 2), bool)
        self.ep_return = np.zeros(n)
        self.ep_len = np.zeros(n, int)
        self.ep_terms: dict[str, np.ndarray] = {k: np.zeros(n) for k in self.task.weights}
        self.completed: deque[dict] = deque(maxlen=200)
        self.total_steps = 0
        self._closed = False

    # ---- lifecycle -----------------------------------------------------------------------------------
    def close(self) -> None:
        if not self._closed:
            self._pool.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- api -------------------------------------------------------------------------------------------
    def reset(self):
        idx = np.arange(self.num_envs)
        self.task.reset_envs(self.state, self.ts, idx, self.rng, len(self.variants))
        self.ep_return[:] = 0
        self.ep_len[:] = 0
        for v in self.ep_terms.values():
            v[:] = 0
        self.contact[:] = False
        return self._observe()

    def _observe(self):
        qpos, qvel = self.task.split(self.state)
        self.task.update_clock(self.ts)  # "steps" gait clock: starts at the first non-zero command (idempotent)
        return self.task.observe(qpos, qvel, self.ts, self.contact, self.props_norm[self.ts.variant], self.rng)

    def _models(self) -> list[mujoco.MjModel]:
        if len(self.variants) == 1:
            return self._nominal
        return [self.variants[v] for v in self.ts.variant]

    def step(self, actions: np.ndarray):
        task, cfg, n = self.task, self.cfg, self.num_envs
        actions = np.clip(np.asarray(actions, np.float64), -cfg.clip_actions, cfg.clip_actions)
        ctrl = task.action_to_ctrl(actions)
        control = np.ascontiguousarray(np.broadcast_to(ctrl[:, None, :], (n, cfg.decimation, self.model.nu)))
        self._pool.rollout(self._models(), self._datas, self.state, control, nstep=cfg.decimation,
                           skip_checks=True, state=self._out_state, sensordata=self._out_sens)
        self.state[:] = self._out_state[:, -1]
        self.contact = task.contacts_from_sensordata(self._out_sens)
        qpos, qvel = task.split(self.state)
        reward, terms, terminated = task.reward(qpos, qvel, actions, ctrl, self.contact, self.ts,
                                                self.props[self.ts.variant, 2])
        time_out = (self.ts.t >= cfg.episode_seconds - 1e-9) & ~terminated
        done = terminated | time_out
        self.total_steps += n

        # bookkeeping
        self.ep_return += reward
        self.ep_len += 1
        for k, v in terms.items():
            self.ep_terms[k] += v
        info: dict = {"time_outs": time_out, "terminated": terminated}
        idx = np.flatnonzero(done)
        if idx.size:
            info["episode"] = self._finish(idx, terminated)
            task.reset_envs(self.state, self.ts, idx, self.rng, len(self.variants))
            self.contact[idx] = False

        # command resampling and pushes on per-env timers
        resample = np.flatnonzero(self.ts.cmd_timer <= 0)
        if resample.size:
            task.sample_commands(self.ts, resample, self.rng)
        if cfg.randomize and cfg.push_interval_s > 0 and cfg.push_vel > 0:
            push = np.flatnonzero(self.ts.push_timer <= 0)
            if push.size:
                vel = self.state[:, 1 + task.nq:1 + task.nq + 2]
                vel[push] += self.rng.uniform(-cfg.push_vel, cfg.push_vel, (push.size, 2))
                self.ts.push_timer[push] = cfg.push_interval_s
        obs, critic_obs = self._observe()
        return obs, critic_obs, reward, done, info

    def _finish(self, idx: np.ndarray, terminated: np.ndarray) -> dict:
        """Record finished episodes; return the per-term means (term sums per max episode second)."""
        max_s = self.cfg.episode_seconds
        out = {"return": float(self.ep_return[idx].mean()), "length_s": float(self.ep_len[idx].mean() * self.dt),
               "fell": float(terminated[idx].mean()), "count": int(idx.size)}
        for k, v in self.ep_terms.items():
            out[f"rew_{k}"] = float(v[idx].mean() / max_s)
        for i in idx:
            self.completed.append({"return": float(self.ep_return[i]), "length_s": float(self.ep_len[i] * self.dt),
                                   "fell": bool(terminated[i]),
                                   "terms": {k: float(v[i] / max_s) for k, v in self.ep_terms.items()}})
        self.ep_return[idx] = 0
        self.ep_len[idx] = 0
        for v in self.ep_terms.values():
            v[idx] = 0
        return out
