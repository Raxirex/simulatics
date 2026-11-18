"""Small neural networks in numpy with hand-written backpropagation, for PPO and behaviour cloning on CPU.

There is no PyTorch in this project, so everything a policy-gradient learner needs is written out here:

* `MLP`: fully connected network (ELU / tanh / ReLU hidden layers, linear output) with an explicit
  forward pass that returns a cache and a backward pass that turns d(loss)/d(output) into parameter
  gradients. Weights use orthogonal initialisation (Saxe et al. 2014) with gain sqrt(2) on hidden layers
  and a small gain on the policy output layer, the choice recommended by Andrychowicz et al.,
  "What Matters in On-Policy Reinforcement Learning?" (2021).
* `GaussianPolicy`: diagonal Gaussian whose mean is an MLP and whose log standard deviation is a
  state-independent parameter vector, the actor commonly used for
  learned legged locomotion (Rudin et al. 2021). It provides the PPO clipped-surrogate loss (Schulman et al. 2017,
  arXiv 1707.06347) and a behaviour-cloning loss, both with analytic gradients.
* `ValueNet`: critic MLP with the (optionally clipped) value loss used by common PPO implementations.
* `Adam` (Kingma & Ba 2015), global-norm gradient clipping, and `RunningMeanStd` (the observation
  usual empirical observation normaliser).

Networks default to float32: numpy's float32 exp/tanh are SIMD-vectorised and about 4-6x faster than
float64 here, and the hidden-layer activations, not the matrix products, dominate the cost on this CPU.
Pass dtype=np.float64 for the finite-difference gradient checks in tests/test_ppo.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ACTIVATIONS = ("elu", "tanh", "relu")
LOG_2PI = float(np.log(2 * np.pi))


# ---- activations ------------------------------------------------------------------------------------
def _act(name: str, z: np.ndarray) -> np.ndarray:
    """Activation, computed in place in z's buffer where possible."""
    if name == "elu":
        # elu(z) = max(z, exp(min(z, 0)) - 1): one exp, no masks (3x faster than where + expm1)
        neg = np.minimum(z, 0.0)
        np.exp(neg, out=neg)
        neg -= 1.0
        return np.maximum(z, neg, out=neg)
    if name == "tanh":
        return np.tanh(z, out=z)
    if name == "relu":
        return np.maximum(z, 0.0, out=z)
    raise ValueError(f"unknown activation {name!r} (use one of {ACTIVATIONS})")


def _act_grad(name: str, h: np.ndarray) -> np.ndarray:
    """d act(z) / dz expressed through the output h = act(z) (so the pre-activation need not be kept)."""
    if name == "elu":
        return np.minimum(h, 0.0) + 1.0   # 1 for z > 0, exp(z) = h + 1 otherwise
    if name == "tanh":
        return 1.0 - h * h
    return (h > 0).astype(h.dtype)


def orthogonal(rng: np.random.Generator, n_in: int, n_out: int, gain: float) -> np.ndarray:
    """Orthogonal (n_in, n_out) matrix scaled by `gain` (Saxe et al. 2014; the usual choice)."""
    a = rng.standard_normal((max(n_in, n_out), min(n_in, n_out)))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))
    w = q if n_in >= n_out else q.T
    return gain * w[:n_in, :n_out]


# ---- MLP ----------------------------------------------------------------------------------------------
class MLP:
    """y = W_L(act(... act(W_1 x + b_1) ...)) + b_L.

    `params` is a flat list [W1, b1, W2, b2, ...] with W of shape (fan_in, fan_out). Optimisers update
    these arrays in place, so references held elsewhere stay valid.
    """

    def __init__(self, sizes: list[int], activation: str = "elu", out_gain: float = 1.0,
                 rng: np.random.Generator | None = None, dtype=np.float32):
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}")
        self.sizes = [int(s) for s in sizes]
        self.activation = activation
        rng = rng or np.random.default_rng(0)
        self.params: list[np.ndarray] = []
        n_layers = len(self.sizes) - 1
        for k, (i, o) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
            gain = out_gain if k == n_layers - 1 else np.sqrt(2.0)
            self.params += [np.ascontiguousarray(orthogonal(rng, i, o, gain), dtype), np.zeros(o, dtype)]

    @property
    def dtype(self):
        return self.params[0].dtype

    @property
    def n_params(self) -> int:
        return int(sum(p.size for p in self.params))

    def forward(self, x: np.ndarray, keep: bool = False):
        """x (B, in) -> y (B, out). With keep=True also returns the cache [x, h1, h2, ...] for `backward`."""
        h = np.asarray(x, self.dtype)
        cache = [h]
        n_layers = len(self.params) // 2
        for k in range(n_layers):
            w, b = self.params[2 * k], self.params[2 * k + 1]
            z = h @ w
            z += b
            h = _act(self.activation, z) if k < n_layers - 1 else z
            if keep and k < n_layers - 1:
                cache.append(h)
        return (h, cache) if keep else h

    def backward(self, cache: list[np.ndarray], grad_out: np.ndarray, need_input_grad: bool = False):
        """Gradients of a scalar loss w.r.t. params, given d loss / d y (B, out). Returns (grads, d loss / d x)."""
        n_layers = len(self.params) // 2
        grads: list[np.ndarray] = [None] * len(self.params)  # type: ignore[list-item]
        g = np.asarray(grad_out, self.dtype)
        for k in reversed(range(n_layers)):
            h_in = cache[k]  # input of layer k
            grads[2 * k] = h_in.T @ g
            grads[2 * k + 1] = g.sum(axis=0)
            if k > 0 or need_input_grad:
                g = g @ self.params[2 * k].T
            if k > 0:
                g *= _act_grad(self.activation, h_in)
        return grads, (g if need_input_grad else None)

    # ---- (de)serialisation ------------------------------------------------------------------------
    def to_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        out = {f"{prefix}sizes": np.array(self.sizes, np.int64)}
        for j, p in enumerate(self.params):
            out[f"{prefix}p{j}"] = p
        return out

    @classmethod
    def from_arrays(cls, arrays, prefix: str, activation: str) -> "MLP":
        net = cls([int(s) for s in arrays[f"{prefix}sizes"]], activation)
        net.params = [np.array(arrays[f"{prefix}p{j}"]) for j in range(len(net.params))]  # keeps the saved dtype
        return net


# ---- Gaussian policy ---------------------------------------------------------------------------------
class GaussianPolicy:
    """pi(a | o) = N(mu_theta(o), diag(exp(log_std))^2) with a state-independent log_std."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int], activation: str = "elu",
                 init_std: float = 1.0, rng: np.random.Generator | None = None, dtype=np.float32):
        self.mu = MLP([obs_dim, *hidden, act_dim], activation, out_gain=0.01, rng=rng, dtype=dtype)
        self.log_std = np.full(act_dim, np.log(init_std), np.float64)
        self.obs_dim, self.act_dim = int(obs_dim), int(act_dim)

    @property
    def params(self) -> list[np.ndarray]:
        return self.mu.params + [self.log_std]

    @property
    def std(self) -> np.ndarray:
        return np.exp(self.log_std)

    def mean(self, obs: np.ndarray) -> np.ndarray:
        return self.mu.forward(obs).astype(np.float64)

    def sample(self, obs: np.ndarray, rng: np.random.Generator):
        """-> actions, log-probabilities, means (float64)."""
        mu = self.mean(obs)
        a = mu + self.std * rng.standard_normal(mu.shape)
        return a, self.log_prob(mu, self.log_std, a), mu

    @staticmethod
    def log_prob(mu: np.ndarray, log_std: np.ndarray, a: np.ndarray) -> np.ndarray:
        z = (a - mu) * np.exp(-log_std)
        return -0.5 * np.sum(z * z, axis=-1) - np.sum(log_std) - 0.5 * mu.shape[-1] * LOG_2PI

    def entropy(self) -> float:
        return float(np.sum(self.log_std) + 0.5 * self.act_dim * (1.0 + LOG_2PI))

    @staticmethod
    def kl(mu_old, std_old, mu_new, std_new) -> np.ndarray:
        """KL(old || new) per sample, the quantity an adaptive learning-rate schedule tracks."""
        return np.sum(np.log(std_new / std_old) + (std_old ** 2 + (mu_old - mu_new) ** 2) / (2.0 * std_new ** 2)
                      - 0.5, axis=-1)

    def ppo_loss(self, obs, actions, old_logp, adv, clip: float, entropy_coef: float):
        """Clipped surrogate  L = -E[min(r A, clip(r, 1-e, 1+e) A)] - c_H H  and its gradients.

        r = exp(logp - logp_old). Gradients flow only through the unclipped branch when it is the minimum
        (where clipping is active the gradient is zero), which is the subgradient PyTorch autograd gives.
        Returns (loss, grads aligned with `params`, info dict with the new mean for the KL check).
        """
        mu, cache = self.mu.forward(obs, keep=True)
        mu = mu.astype(np.float64)
        log_std = self.log_std
        inv_std = np.exp(-log_std)
        diff = actions - mu
        z = diff * inv_std
        logp = -0.5 * np.sum(z * z, axis=1) - np.sum(log_std) - 0.5 * self.act_dim * LOG_2PI
        ratio = np.exp(np.clip(logp - old_logp, -20.0, 20.0))
        s1 = ratio * adv
        s2 = np.clip(ratio, 1.0 - clip, 1.0 + clip) * adv
        n = len(adv)
        ent = self.entropy()
        loss = -np.mean(np.minimum(s1, s2)) - entropy_coef * ent
        # d loss / d logp_i = -(A_i r_i / n) where the unclipped branch is active
        active = s1 <= s2
        g_logp = np.where(active, -adv * ratio, 0.0) / n
        # d logp / d mu = (a - mu) / std^2 ; d logp / d log_std = z^2 - 1
        g_mu = g_logp[:, None] * diff * inv_std * inv_std
        g_log_std = (g_logp[:, None] * (z * z - 1.0)).sum(axis=0) - entropy_coef
        grads, _ = self.mu.backward(cache, g_mu)
        clip_frac = float(np.mean(np.abs(ratio - 1.0) > clip))
        return float(loss), grads + [g_log_std], {"mu": mu, "clip_frac": clip_frac, "entropy": ent}

    def bc_loss(self, obs, target_actions, weights: np.ndarray | None = None):
        """Behaviour cloning: mean squared error between the policy mean and the teacher action."""
        mu, cache = self.mu.forward(obs, keep=True)
        diff = mu.astype(np.float64) - target_actions
        w = np.ones(len(obs)) if weights is None else weights
        denom = w.sum() * self.act_dim
        loss = float(np.sum(w[:, None] * diff * diff) / denom)
        grads, _ = self.mu.backward(cache, 2.0 * w[:, None] * diff / denom)
        return loss, grads + [np.zeros_like(self.log_std)]

    def to_arrays(self, prefix: str = "actor_") -> dict[str, np.ndarray]:
        return {**self.mu.to_arrays(prefix), f"{prefix}log_std": self.log_std}

    @classmethod
    def from_arrays(cls, arrays, activation: str, prefix: str = "actor_") -> "GaussianPolicy":
        mu = MLP.from_arrays(arrays, prefix, activation)
        pol = cls(mu.sizes[0], mu.sizes[-1], [], activation)
        pol.mu = mu
        pol.log_std = np.array(arrays[f"{prefix}log_std"], np.float64)
        return pol


class ValueNet:
    """State-value critic V(o) with PPO's clipped value loss (as in common PPO implementations)."""

    def __init__(self, obs_dim: int, hidden: list[int], activation: str = "elu",
                 rng: np.random.Generator | None = None, dtype=np.float32):
        self.net = MLP([obs_dim, *hidden, 1], activation, out_gain=1.0, rng=rng, dtype=dtype)

    @property
    def params(self) -> list[np.ndarray]:
        return self.net.params

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.net.forward(obs)[:, 0].astype(np.float64)

    def loss(self, obs, returns, old_values, clip: float | None):
        """mean(max((V - R)^2, (V_clip - R)^2)) with V_clip = V_old + clip(V - V_old, -c, c); plain MSE if clip is None."""
        v, cache = self.net.forward(obs, keep=True)
        v = v[:, 0].astype(np.float64)
        n = len(v)
        e1 = v - returns
        if clip is None:
            loss = float(np.mean(e1 * e1))
            g = 2.0 * e1 / n
        else:
            dv = v - old_values
            v_clip = old_values + np.clip(dv, -clip, clip)
            e2 = v_clip - returns
            use1 = e1 * e1 >= e2 * e2
            loss = float(np.mean(np.where(use1, e1 * e1, e2 * e2)))
            inside = np.abs(dv) < clip
            g = np.where(use1, 2.0 * e1, np.where(inside, 2.0 * e2, 0.0)) / n
        grads, _ = self.net.backward(cache, g[:, None])
        return loss, grads


# ---- optimiser ----------------------------------------------------------------------------------------
def global_norm(grads: list[np.ndarray]) -> float:
    return float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))


def clip_grad_norm(grads: list[np.ndarray], max_norm: float) -> tuple[list[np.ndarray], float]:
    """Scale all gradients together so their global L2 norm is at most max_norm (torch clip_grad_norm_)."""
    norm = global_norm(grads)
    if max_norm > 0 and norm > max_norm:
        s = max_norm / (norm + 1e-6)
        grads = [g * s for g in grads]
    return grads, norm


class Adam:
    """Adam (Kingma & Ba 2015) updating a list of numpy arrays in place. `lr` may be changed between steps."""

    def __init__(self, params: list[np.ndarray], lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params, self.lr, self.betas, self.eps = params, float(lr), betas, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        b1, b2 = self.betas
        c1, c2 = 1.0 - b1 ** self.t, 1.0 - b2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


# ---- observation normalisation -----------------------------------------------------------------------
class RunningMeanStd:
    """Running mean / variance over batches (parallel-variance formula of Chan et al.), as in common PPO
    implementations. `normalize` clips to +-clip standard deviations."""

    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-2):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = 0.0
        self.clip, self.eps = float(clip), float(eps)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, np.float64).reshape(-1, self.mean.size)
        if x.shape[0] == 0:
            return
        n, m, v = x.shape[0], x.mean(axis=0), x.var(axis=0)
        tot = self.count + n
        delta = m - self.mean
        self.mean = self.mean + delta * n / tot
        self.var = (self.var * self.count + v * n + delta * delta * self.count * n / tot) / tot
        self.count = tot

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var) + self.eps

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / self.std, -self.clip, self.clip)

    def to_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        return {f"{prefix}mean": self.mean, f"{prefix}var": self.var, f"{prefix}count": np.array(self.count),
                f"{prefix}clip": np.array(self.clip), f"{prefix}eps": np.array(self.eps)}

    @classmethod
    def from_arrays(cls, arrays, prefix: str) -> "RunningMeanStd":
        r = cls(len(arrays[f"{prefix}mean"]), float(arrays[f"{prefix}clip"]), float(arrays[f"{prefix}eps"]))
        r.mean = np.array(arrays[f"{prefix}mean"], np.float64)
        r.var = np.array(arrays[f"{prefix}var"], np.float64)
        r.count = float(arrays[f"{prefix}count"])
        return r


# ---- files ---------------------------------------------------------------------------------------------
def save_npz(path: str | Path, arrays: dict[str, np.ndarray], meta: dict, retry_seconds: float = 3.0) -> Path:
    """Write arrays plus a JSON metadata string to an .npz file (loadable without pickle).

    The file is written next to the target and renamed over it, which is atomic on the same volume, so a
    reader never sees half a file. On Windows the rename fails with PermissionError while any reader
    (the GUI listing policies, a controller loading one) has the target open, so it is retried for up to
    `retry_seconds` instead of killing a long training run over a millisecond-long read.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
    deadline = time.monotonic() + max(0.0, retry_seconds)
    while True:
        try:
            tmp.replace(path)
            return path
        except PermissionError:
            if time.monotonic() >= deadline:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.02)


def load_npz(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return arrays, meta
