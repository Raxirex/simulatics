"""Tiny MLP policy stored as one flat parameter vector (the evolution genome).

`forward_batch` evaluates a *different* network for every row, which is what
an evolutionary population needs: row i of `params` controls environment i.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import native


@dataclass
class MLPSpec:
    sizes: list[int]  # [obs_dim, hidden..., act_dim]

    @property
    def n_params(self) -> int:
        return sum(i * o + o for i, o in zip(self.sizes[:-1], self.sizes[1:]))

    def init(self, rng: np.random.Generator, out_scale: float = 0.01) -> np.ndarray:
        parts = []
        n_layers = len(self.sizes) - 1
        for k, (i, o) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
            scale = (out_scale if k == n_layers - 1 else 1.0) / np.sqrt(i)
            parts.append((rng.standard_normal(o * i) * scale).astype(np.float32))
            parts.append(np.zeros(o, np.float32))
        return np.concatenate(parts)

    def forward_batch(self, params: np.ndarray, obs: np.ndarray, use_native: bool = True) -> np.ndarray:
        """params (B, n) or (n,), obs (B, obs_dim) -> actions (B, act_dim) in [-1, 1]."""
        obs = np.ascontiguousarray(obs, np.float32)
        if params.ndim == 1:
            return self._forward_shared(params, obs)
        params = np.ascontiguousarray(params, np.float32)
        if use_native and native.available():
            return native.mlp_forward_batch(params, self.sizes, obs)
        x, off = obs, 0
        b = obs.shape[0]
        for i, o in zip(self.sizes[:-1], self.sizes[1:]):
            w = params[:, off:off + o * i].reshape(b, o, i); off += o * i
            bias = params[:, off:off + o]; off += o
            x = np.tanh(np.matmul(w, x[:, :, None])[:, :, 0] + bias)
        return x

    def _forward_shared(self, params: np.ndarray, obs: np.ndarray) -> np.ndarray:
        x, off = obs, 0
        for i, o in zip(self.sizes[:-1], self.sizes[1:]):
            w = params[off:off + o * i].reshape(o, i); off += o * i
            bias = params[off:off + o]; off += o
            x = np.tanh(x @ w.T + bias)
        return x


@dataclass
class Policy:
    """A trained controller: network + observation normaliser + what it was trained on."""
    spec: MLPSpec
    params: np.ndarray
    obs_mean: np.ndarray
    obs_std: np.ndarray
    meta: dict = field(default_factory=dict)  # humanoid params, task config, fitness...

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = np.clip((obs - self.obs_mean) / self.obs_std, -5, 5)
        return self.spec.forward_batch(self.params, x)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, params=self.params, sizes=np.array(self.spec.sizes),
                 obs_mean=self.obs_mean, obs_std=self.obs_std, meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        z = np.load(path, allow_pickle=False)
        return cls(MLPSpec([int(s) for s in z["sizes"]]), z["params"], z["obs_mean"], z["obs_std"],
                   json.loads(str(z["meta"])))


class RunningNorm:
    """Running mean / std of observations (shared by the whole population)."""

    def __init__(self, dim: int):
        self.n = 1e-4
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        bn, bm, bv = x.shape[0], x.mean(axis=0), x.var(axis=0)
        tot = self.n + bn
        delta = bm - self.mean
        self.var = (self.var * self.n + bv * bn + delta * delta * self.n * bn / tot) / tot
        self.mean = self.mean + delta * bn / tot
        self.n = tot

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var) + 1e-3
