"""Parallel closed-loop episodes on CPU threads.

MuJoCo's `rollout` module steps a whole batch of states inside a C++ thread
pool (no Python GIL). We call it once per control step: batched observations
and policy inference happen in numpy / our native library, then all
environments advance `frame_skip` physics steps in parallel. Each batch entry
may use its own MjModel, so every candidate can have a different body.
"""
from __future__ import annotations

import os
from typing import Sequence

import mujoco
import numpy as np
from mujoco import rollout

from .task import LocoTask
from ..train.policy import MLPSpec, RunningNorm


class ParallelEvaluator:
    def __init__(self, nthread: int | None = None):
        self.nthread = int(nthread or os.cpu_count() or 1)
        self._pool = rollout.Rollout(nthread=self.nthread)
        self._datas: list[mujoco.MjData] = []
        self._sig = None

    def close(self) -> None:
        self._pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _ensure_datas(self, model: mujoco.MjModel) -> None:
        sig = (model.nq, model.nv, model.nu, model.nbody, model.ngeom, model.nsensordata)
        if sig != self._sig:
            self._datas = [mujoco.MjData(model) for _ in range(self.nthread)]
            self._sig = sig

    def evaluate(
        self,
        models: mujoco.MjModel | Sequence[mujoco.MjModel],
        task: LocoTask,
        spec: MLPSpec,
        params: np.ndarray,              # (B, n_params): one policy per episode
        episode_steps: int,
        norm: RunningNorm,
        rng: np.random.Generator,
        heading_range: float = 0.0,      # target heading sampled in +-range (rad)
        init_noise: float = 0.02,
        collect_obs: bool = True,
        should_stop=lambda: False,
    ):
        """Run B episodes in parallel. Returns (fitness, env_steps, obs_samples)."""
        b = params.shape[0]
        single = isinstance(models, mujoco.MjModel)
        first = models if single else models[0]
        self._ensure_datas(first)

        if single:
            state = np.tile(LocoTask.initial_state(first, self._datas[0]), (b, 1))
        else:
            state = np.stack([LocoTask.initial_state(m, self._datas[0]) for m in models])
        nq, nv = task.nq, task.nv
        state[:, 1 + 7:1 + nq] += rng.normal(0, init_noise, (b, nq - 7))
        state[:, 1 + nq:1 + nq + nv] += rng.normal(0, init_noise * 2, (b, nv))
        target_yaw = rng.uniform(-heading_range, heading_range, b) if heading_range > 0 else np.zeros(b)

        mean, std = norm.mean, norm.std
        fitness = np.zeros(b)
        alive = np.ones(b, bool)
        samples, steps = [], 0
        for t in range(episode_steps):
            idx = np.flatnonzero(alive)
            if idx.size == 0 or should_stop():
                break
            s = state[idx]
            qpos, qvel = task.split(s)
            obs = task.obs(qpos, qvel, target_yaw[idx])
            if collect_obs and t % 4 == 0:
                samples.append(obs[:: max(1, idx.size // 64)])
            action = spec.forward_batch(params[idx], np.clip((obs - mean) / std, -5, 5))
            ctrl = task.ctrl(action)
            mods = first if single else [models[i] for i in idx]
            out, _ = self._pool.rollout(mods, self._datas, s, ctrl[:, None, :], nstep=task.cfg.frame_skip)
            new = out[:, -1]
            r, done = task.reward(qpos, new[:, 1:1 + nq], action, target_yaw[idx])
            fitness[idx] += r
            state[idx] = new
            alive[idx[done]] = False
            steps += idx.size
        obs_samples = np.concatenate(samples) if samples else np.zeros((0, task.obs_dim))
        return fitness, steps * task.cfg.frame_skip, obs_samples
