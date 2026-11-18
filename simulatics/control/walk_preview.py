"""ZMP preview control of the cart-table (linear inverted pendulum) model.

Kajita, Kanehiro, Kaneko, Fujiwara, Harada, Yokoi, Hirukawa, "Biped walking
pattern generation by using preview control of zero-moment point", ICRA 2003.

The centre of mass (CoM) moves at constant height z_c. Per horizontal axis the
state is x = (c, c', c'') and the input is the CoM jerk u, so with period dt

    x[k+1] = A x[k] + B u[k],    p[k] = C x[k] = c - (z_c / g) c''

where p is the zero-moment point (ZMP). Minimising sum Qe (p - p_ref)^2 + R u^2
over an infinite horizon with the reference known N steps ahead gives

    u[k] = -K x[k] + sum_{j=1..N} f_j p_ref[k + j]

with K and f_j from the discrete algebraic Riccati equation (scipy). Because
the controller looks ahead, the CoM starts to move before the ZMP reference
jumps to the next foot, which is what makes the motion dynamically feasible.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are

G = 9.81


class ZmpPreview:
    def __init__(self, dt: float, z_c: float, preview_s: float = 1.6, q_e: float = 1.0, r: float = 1e-6):
        self.dt, self.z_c = float(dt), float(z_c)
        self.N = max(1, int(round(preview_s / dt)))
        A = np.array([[1.0, dt, dt * dt / 2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        B = np.array([[dt ** 3 / 6], [dt * dt / 2], [dt]])
        C = np.array([[1.0, 0.0, -z_c / G]])
        P = solve_discrete_are(A, B, C.T @ C * q_e, np.array([[r]]))
        S = float(r + (B.T @ P @ B)[0, 0])
        self.K = ((B.T @ P @ A) / S).ravel()                          # (3,)
        Ac = A - B @ self.K[None, :]
        f = np.empty(self.N)
        v = C.T * q_e                                                 # (A - BK)^T^(j-1) C^T Qe
        for j in range(self.N):
            f[j] = float((B.T @ v)[0, 0]) / S
            v = Ac.T @ v
        # The gains of the infinite preview sum to K[0] (a constant reference is an equilibrium). Truncating at
        # N drops a small tail (~1% for 1.6 s), which would leave a steady-state offset; assigning the tail to
        # the last sample is the same as assuming the reference stays constant beyond the horizon.
        f[-1] += self.K[0] - f.sum()
        self.f = f
        self.A, self.B, self.C = A, B, C.ravel()

    def step(self, x: np.ndarray, p_ref_future: np.ndarray) -> np.ndarray:
        """Advance the state x (3, d) one period. p_ref_future (N, d) holds p_ref[k+1 .. k+N]."""
        u = -self.K @ x + self.f @ p_ref_future                         # (d,)
        return self.A @ x + self.B @ u[None, :]

    def zmp(self, x: np.ndarray) -> np.ndarray:
        return self.C @ x
