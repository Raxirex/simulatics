"""Batched motion-tracking environments stepped by MuJoCo's rollout thread pool.

Same pattern as sim/vec_env.py (batched environments on the CPU): N copies advance in lockstep; per policy
step the actions become servo targets around the reference (TrackTask.action_to_ctrl), `mujoco.rollout`
advances all N states `decimation` physics steps in its C++ thread pool without the GIL, then rewards,
terminations and time-outs are computed in numpy and finished environments are reset in place with
reference state initialisation. The interface is the one `train.ppo.run_ppo` expects:

    obs, critic_obs = env.reset()
    obs, critic_obs, reward, done, info = env.step(actions)   # info["time_outs"], info["episode"]

The reward handed to PPO is the DeepMimic reward times `cfg.reward_scale`; episode statistics report the
unscaled reward (1 per step = perfect tracking).
"""
from __future__ import annotations

from collections import deque

import mujoco
import numpy as np
from mujoco import rollout

from .motion import Motion
from .track_task import TrackConfig, TrackTask

TERMS = ("pose", "vel", "end_eff", "com", "root")


class ImitationEnv:
    def __init__(self, model: mujoco.MjModel, motion: Motion, cfg: TrackConfig | None = None, num_envs: int = 64,
                 nthread: int = 4, seed: int = 0):
        self.model = model
        self.task = TrackTask(model, motion, cfg)
        self.cfg = self.task.cfg
        self.num_envs = int(num_envs)
        self.nthread = max(1, int(nthread))
        self.rng = np.random.default_rng(seed)
        self.obs_dim, self.critic_obs_dim, self.act_dim = (self.task.obs_dim, self.task.critic_obs_dim,
                                                           self.task.act_dim)
        self.dt = self.task.dt
        n, sub = self.num_envs, self.cfg.decimation
        self._models = [model] * n
        self._pool = rollout.Rollout(nthread=self.nthread)
        self._datas = [mujoco.MjData(model) for _ in range(self.nthread)]
        self.state = np.zeros((n, self.task.nstate))
        self._out_state = np.zeros((n, sub, self.task.nstate))
        self._out_sens = np.zeros((n, sub, max(1, model.nsensordata)))
        self.ts = self.task.make_state(n)
        self.ep_return = np.zeros(n)
        self.ep_len = np.zeros(n, int)
        self.ep_sums = {k: np.zeros(n) for k in TERMS + ("joint_rmse", "ee_err", "root_dist")}
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
        self.task.reset_envs(self.state, self.ts, idx, self.rng)
        self.ep_return[:] = 0
        self.ep_len[:] = 0
        for v in self.ep_sums.values():
            v[:] = 0
        return self.observe()

    def observe(self):
        qpos, qvel = self.task.split(self.state)
        return self.task.observe(qpos, qvel, self.ts)

    def set_reference_state(self, idx=None) -> None:
        """Write the reference state of each env's current frame into the simulation state (for tests)."""
        idx = np.arange(self.num_envs) if idx is None else np.asarray(idx, int)
        self.state[idx] = self.task.reference_states(self.ts.frame[idx], self.ts.origin[idx])

    def step(self, actions: np.ndarray):
        task, cfg, n = self.task, self.cfg, self.num_envs
        actions = np.clip(np.nan_to_num(np.asarray(actions, np.float64)), -cfg.clip_actions, cfg.clip_actions)
        ctrl = task.action_to_ctrl(actions, self.ts)
        control = np.ascontiguousarray(np.broadcast_to(ctrl[:, None, :], (n, cfg.decimation, self.model.nu)))
        self._pool.rollout(self._models, self._datas, self.state, control, nstep=cfg.decimation,
                           skip_checks=True, state=self._out_state, sensordata=self._out_sens)
        self.state[:] = self._out_state[:, -1]
        self.ts.frame += 1
        self.ts.t += self.dt
        self.ts.last_action[:] = actions
        return self._after_step(actions)

    def _after_step(self, actions: np.ndarray):
        task, cfg = self.task, self.cfg
        qpos, qvel = task.split(self.state)
        r, terms, terminated, err = task.reward(qpos, qvel, self.ts)
        time_out = ((self.ts.t >= cfg.episode_seconds - 1e-9) | task.ref.finished(self.ts.frame)) & ~terminated
        done = terminated | time_out
        self.total_steps += self.num_envs
        self.ep_return += r
        self.ep_len += 1
        for k in TERMS:
            self.ep_sums[k] += terms[k]
        for k in ("joint_rmse", "ee_err", "root_dist"):
            self.ep_sums[k] += err[k]
        info: dict = {"time_outs": time_out, "terminated": terminated, "reward_raw": r, "errors": err}
        idx = np.flatnonzero(done)
        keys, com = err.pop("fk")
        if idx.size:
            info["episode"] = self._finish(idx, terminated)
            task.reset_envs(self.state, self.ts, idx, self.rng)
            qp, _ = task.split(self.state[idx])
            keys[idx], com[idx] = task.fk(qp, task.ee_names)
        if cfg.push_interval_s > 0 and cfg.push_vel > 0:
            self.ts.push_timer -= self.dt
            push = np.flatnonzero(self.ts.push_timer <= 0)
            if push.size:
                v = self.state[:, 1 + task.nq + task.da:1 + task.nq + task.da + 2]
                v[push] += self.rng.uniform(-cfg.push_vel, cfg.push_vel, (push.size, 2))
                self.ts.push_timer[push] = cfg.push_interval_s
        qpos, qvel = task.split(self.state)
        obs, cobs = task.observe(qpos, qvel, self.ts, fk=(keys, com))
        return obs, cobs, r * cfg.reward_scale, done, info

    def _finish(self, idx: np.ndarray, terminated: np.ndarray) -> dict:
        """Record finished episodes; returns flat count-weighted means for run_ppo's log."""
        steps = np.maximum(self.ep_len[idx], 1)
        out = {"return": float(self.ep_return[idx].mean()), "length_s": float(self.ep_len[idx].mean() * self.dt),
               "fell": float(terminated[idx].mean()), "reward_per_step": float((self.ep_return[idx] / steps).mean()),
               "count": int(idx.size)}
        for k, v in self.ep_sums.items():
            out[k] = float((v[idx] / steps).mean())
        for i in idx:
            s = max(1, int(self.ep_len[i]))
            self.completed.append({"return": float(self.ep_return[i]), "length_s": float(self.ep_len[i] * self.dt),
                                   "fell": bool(terminated[i]), "reward_per_step": float(self.ep_return[i] / s),
                                   "terms": {k: float(v[i] / s) for k, v in self.ep_sums.items()}})
        self.ep_return[idx] = 0
        self.ep_len[idx] = 0
        for v in self.ep_sums.values():
            v[idx] = 0
        return out
