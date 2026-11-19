"""Per-frame reference tables for tracking and playback, and a batched forward-kinematics helper.

`MotionReference` resamples a `Motion` to the controller rate and precomputes, for every frame of one
cycle, everything a tracker compares against: model qpos / qvel, key-body world positions and the
whole-body centre of mass (forward kinematics of *this* model, not the stored key_pos), and the servo
targets that reproduce the motion ("feed-forward ctrl"). Lookups take integer frame indices past the end
(loops repeat with the cycle transform) plus an optional planar anchor that moves the whole motion
(x, y, yaw) - e.g. to start it where the robot stands.

Feed-forward servo targets
--------------------------
Every actuator here is a position servo, force = kp (ctrl - q) - kv qdot. Sending ctrl = q_ref makes a
soft servo sag under gravity (a 300 Nm/rad hip holding ~60 Nm sits 0.2 rad low). So, as WalkController
and ArmController do, the target includes the force the joint needs:

    ctrl = q_ref + (tau + kv qdot_ref) / kp

with tau from floating-base inverse dynamics of the reference, M qdd + h - passive = S^T tau + J_c^T f:
the six unactuated root rows fix the total contact wrench f, which is split between the feet that are on
the ground in the reference (sole within 2 cm of the floor) by a weighted minimum-norm solve (weights
follow the centre of mass between the feet; moments are expensive, so load is shared through vertical
forces). qdd comes from finite differences of the reference velocities, lightly smoothed.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .motion import (Motion, free_root, planar_apply_points, planar_apply_quats, planar_apply_vectors,
                     planar_compose, quat_to_mat, robot_keypoints)


# =========================================================================================================
# Batched forward kinematics (numpy)
# =========================================================================================================
def _axis_rot(axis: np.ndarray, q: np.ndarray) -> np.ndarray:
    """(N,) angles about a fixed unit axis -> (N, 3, 3) (Rodrigues)."""
    k = np.asarray(axis, float)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    s, c = np.sin(q)[:, None, None], np.cos(q)[:, None, None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


class BatchFK:
    """Key-point positions and whole-body centre of mass for a batch of qpos, without MjData.

    Handles the robot's tree under its free root with hinge and slide joints (others raise). Matches
    mj_kinematics + mj_comPos (tests/test_imitate.py checks it)."""

    def __init__(self, model: mujoco.MjModel, keypoints: dict | None = None):
        root, qa, _ = free_root(model)
        self.root, self.qa = root, qa
        self.bodies = [b for b in range(model.nbody) if b == root or (b > 0 and model.body_rootid[b] == root
                                                                         and b != root and self._under(model, b, root))]
        self.parent = {b: int(model.body_parentid[b]) for b in self.bodies}
        self.pos = {b: model.body_pos[b].copy() for b in self.bodies}
        self.R0 = {b: quat_to_mat(model.body_quat[b]) for b in self.bodies}
        self.joints = {}
        for b in self.bodies:
            js = []
            for j in range(int(model.body_jntadr[b]), int(model.body_jntadr[b]) + int(model.body_jntnum[b])):
                if j < 0:
                    continue
                t = model.jnt_type[j]
                if b == root and t == mujoco.mjtJoint.mjJNT_FREE:
                    continue
                if t not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                    raise ValueError("BatchFK supports hinge and slide joints below the free root")
                js.append((int(t), int(model.jnt_qposadr[j]), model.jnt_axis[j].copy(), model.jnt_pos[j].copy()))
            self.joints[b] = js
        self.mass = np.array([model.body_mass[b] for b in self.bodies])
        self.ipos = {b: model.body_ipos[b].copy() for b in self.bodies}
        self.total = float(self.mass.sum())
        self.keypoints = robot_keypoints(model) if keypoints is None else dict(keypoints)
        self.key_names = list(self.keypoints)

    @staticmethod
    def _under(model, b, root) -> bool:
        while b > 0:
            if b == root:
                return True
            b = int(model.body_parentid[b])
        return False

    def __call__(self, qpos: np.ndarray, keys: list[str] | None = None):
        """qpos (N, nq) -> (key positions (N, K, 3) for `keys` (default all), CoM (N, 3))."""
        qpos = np.atleast_2d(qpos)
        n = len(qpos)
        xp, xm = {}, {}
        com = np.zeros((n, 3))
        for b in self.bodies:
            if b == self.root:
                p = qpos[:, self.qa:self.qa + 3].copy()
                R = quat_to_mat(qpos[:, self.qa + 3:self.qa + 7])
            else:
                pp, Rp = xp[self.parent[b]], xm[self.parent[b]]
                p = pp + Rp @ self.pos[b]
                R = Rp @ self.R0[b]
                for t, adr, axis, jpos in self.joints[b]:
                    if t == mujoco.mjtJoint.mjJNT_HINGE:
                        anchor = p + R @ jpos
                        R = R @ _axis_rot(axis, qpos[:, adr])
                        p = anchor - R @ jpos
                    else:
                        p = p + (R @ axis) * qpos[:, adr:adr + 1]
            xp[b], xm[b] = p, R
        for i, b in enumerate(self.bodies):
            com += self.mass[i] * (xp[b] + xm[b] @ self.ipos[b])
        com /= self.total
        names = self.key_names if keys is None else keys
        out = np.zeros((n, len(names), 3))
        for k, name in enumerate(names):
            b, off = self.keypoints[name]
            out[:, k] = xp[b] + xm[b] @ off
        return out, com


# =========================================================================================================
# Inverse-dynamics feed-forward
# =========================================================================================================
def _sole_contacts(model: mujoco.MjModel):
    """[(foot body, sole centre in the body frame, box corners in the body frame)] for <side>_foot geoms."""
    out = []
    for s in ("left", "right"):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_foot")
        if g < 0:
            continue
        R = quat_to_mat(model.geom_quat[g])
        size = model.geom_size[g]
        corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) * size
        out.append((int(model.geom_bodyid[g]), model.geom_pos[g] + R @ np.array([0.0, 0.0, -size[2]]),
                    model.geom_pos[g] + corners @ R.T, float(size[1])))
    return out


def servo_params(model: mujoco.MjModel):
    """Per actuator: (joint qpos adr, dof adr, kp, kv, is position servo) for joint-transmission actuators."""
    nu = model.nu
    qadr = np.full(nu, -1)
    dadr = np.full(nu, -1)
    kp = np.zeros(nu)
    kv = np.zeros(nu)
    servo = np.zeros(nu, bool)
    for a in range(nu):
        if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        j = int(model.actuator_trnid[a, 0])
        qadr[a], dadr[a] = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        g = float(model.actuator_gainprm[a, 0])
        if (model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE and g > 0
                and abs(model.actuator_biasprm[a, 1] + g) <= 1e-6 * max(1.0, g)):
            servo[a] = True
            kp[a] = g
            kv[a] = -float(model.actuator_biasprm[a, 2])
    return qadr, dadr, kp, kv, servo


def clip_ctrl(model: mujoco.MjModel, ctrl: np.ndarray) -> np.ndarray:
    lim = model.actuator_ctrllimited.astype(bool)
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    out = np.where(lim, np.clip(ctrl, lo, hi), ctrl)
    return np.where(np.isfinite(out), out, 0.0)


def feedforward_ctrl(model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray, qacc: np.ndarray,
                     contact_height: float = 0.02, gravity_only: bool = False):
    """Servo targets (F, nu) reproducing a reference trajectory, and the stance mask (F, n_feet).

    See the module docstring. gravity_only=True drops the inertial term (qacc = 0): a static feed-forward."""
    d = mujoco.MjData(model)
    _, _, da = free_root(model)
    feet = _sole_contacts(model)
    qadr, dadr, kp, kv, servo = servo_params(model)
    flim = model.actuator_forcelimited.astype(bool)
    frange = model.actuator_forcerange
    F = len(qpos)
    ctrl = np.zeros((F, model.nu))
    stance = np.zeros((F, len(feet)), bool)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    root = slice(da, da + 6)
    for t in range(F):
        d.qpos[:] = qpos[t]
        d.qvel[:] = qvel[t]
        mujoco.mj_forward(model, d)
        h = np.zeros(model.nv)
        if not gravity_only:
            mujoco.mj_mulM(model, d, h, qacc[t])
        h += d.qfrc_bias - d.qfrc_passive
        Js, w, pts = [], [], []
        for i, (b, sole, corners, half_w) in enumerate(feet):
            R = d.xmat[b].reshape(3, 3)
            low = float(np.min(d.xpos[b][2] + corners @ R[2]))
            if low < contact_height:
                stance[t, i] = True
                p = d.xpos[b] + R @ sole
                mujoco.mj_jac(model, d, jp, jr, p, b)
                Js.append(np.vstack([jp, jr]))
                pts.append(p)
                w.append(half_w ** 2)
        tau = h.copy()
        if Js:
            com = d.subtree_com[int(model.jnt_bodyid[0])]
            if len(pts) == 2:   # share: 1 at a foot, 0 at the other, linear in the CoM between them
                a, b = pts[0][:2], pts[1][:2]
                seg = a - b
                alpha = float(np.clip(np.dot(com[:2] - b, seg) / max(np.dot(seg, seg), 1e-9), 0.0, 1.0))
                share = [max(alpha, 0.05), max(1 - alpha, 0.05)]
            else:
                share = [1.0]
            A = np.hstack([J[:, root].T for J in Js])
            winv = np.concatenate([s * np.array([1, 1, 1, m2, m2, m2]) for s, m2 in zip(share, w)])
            try:
                f = winv * (A.T @ np.linalg.solve((A * winv) @ A.T, h[root]))
            except np.linalg.LinAlgError:
                f = np.zeros(6 * len(Js))
            for i, J in enumerate(Js):
                tau -= J.T @ f[6 * i:6 * i + 6]
        for a in range(model.nu):
            if qadr[a] < 0:
                continue
            force = tau[dadr[a]] / max(abs(float(model.actuator_gear[a, 0])), 1e-9)
            if flim[a]:
                force = float(np.clip(force, frange[a, 0], frange[a, 1]))
            if servo[a]:
                ctrl[t, a] = qpos[t, qadr[a]] + (force + kv[a] * qvel[t, dadr[a]]) / kp[a]
            else:
                ctrl[t, a] = force / max(float(model.actuator_gainprm[a, 0]), 1e-9)
    return clip_ctrl(model, ctrl), stance


# =========================================================================================================
# Reference tables
# =========================================================================================================
class MotionReference:
    """A motion resampled to `dt` with per-frame tables (see the module docstring)."""

    def __init__(self, model: mujoco.MjModel, motion: Motion, dt: float, feedforward: str = "dynamic"):
        """feedforward: "dynamic" (inverse dynamics), "static" (gravity + contact only), or "none"
        (ctrl = q_ref + kv/kp qdot_ref)."""
        self.model = model
        m = motion if abs(motion.fps * dt - 1.0) < 1e-6 else motion.resample(1.0 / dt)
        self.motion, self.dt = m, float(dt)
        self.loop = m.loop
        self.n = m.n_frames
        _, self.qa, self.da = free_root(model)
        self.qpos = m.qpos(model)
        self.qvel = m.qvel(model)
        # accelerations: central differences of qvel (looping or clamped), smoothed [1 2 1] / 4
        k = np.arange(self.n)
        if self.loop:
            lo, hi = (k - 1) % self.n, (k + 1) % self.n
            span = np.full(self.n, 2 * dt)
        else:
            lo, hi = np.maximum(k - 1, 0), np.minimum(k + 1, self.n - 1)
            span = np.maximum(hi - lo, 1) * dt
        qacc = (self.qvel[hi] - self.qvel[lo]) / span[:, None]
        if self.loop:
            # the root's linear velocity rotates with the cycle: express both ends in the same frame
            v_hi = self.qvel[hi, self.da:self.da + 3].copy()
            wrap = hi < k
            if wrap.any() and abs(m.cycle[2]) > 0:
                c, s = np.cos(m.cycle[2]), np.sin(m.cycle[2])
                v = v_hi[wrap]
                v_hi[wrap] = np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], 1)
            v_lo = self.qvel[lo, self.da:self.da + 3].copy()
            wrap = lo > k
            if wrap.any() and abs(m.cycle[2]) > 0:
                c, s = np.cos(-m.cycle[2]), np.sin(-m.cycle[2])
                v = v_lo[wrap]
                v_lo[wrap] = np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]], 1)
            qacc[:, self.da:self.da + 3] = (v_hi - v_lo) / span[:, None]
            qacc = 0.25 * (np.roll(qacc, 1, 0) + 2 * qacc + np.roll(qacc, -1, 0))
        elif self.n > 2:
            qacc[1:-1] = 0.25 * (qacc[:-2] + 2 * qacc[1:-1] + qacc[2:])
        self.qacc = np.clip(qacc, -200.0, 200.0)
        # key points and CoM by FK of this model
        self.fk = BatchFK(model)
        self.key_names = self.fk.key_names
        self.keys, self.com = self.fk(self.qpos)
        if self.loop:
            self.com_vel = (self.frames(k + 1, fields=("com",))["com"]
                            - self.frames(k - 1, fields=("com",))["com"]) / (2 * dt)
        else:
            self.com_vel = (self.com[hi] - self.com[lo]) / span[:, None]
        # feed-forward servo targets
        qadr, dadr, kp, kv, servo = servo_params(model)
        if feedforward == "none":
            ctrl = np.zeros((self.n, model.nu))
            for a in range(model.nu):
                if qadr[a] >= 0 and servo[a]:
                    ctrl[:, a] = self.qpos[:, qadr[a]] + kv[a] / kp[a] * self.qvel[:, dadr[a]]
            self.ctrl, self.stance = clip_ctrl(model, ctrl), np.zeros((self.n, 2), bool)
        else:
            self.ctrl, self.stance = feedforward_ctrl(model, self.qpos, self.qvel, self.qacc,
                                                      gravity_only=feedforward == "static")
        self.feedforward = feedforward

    # ---- lookups ----------------------------------------------------------------------------------------
    @property
    def duration(self) -> float:
        return self.motion.duration

    def phase(self, k) -> np.ndarray:
        k = np.asarray(k)
        return (k % self.n) / self.n if self.loop else np.clip(k / max(1, self.n - 1), 0.0, 1.0)

    def finished(self, k) -> np.ndarray:
        """True where a non-looping motion has reached its last frame."""
        k = np.asarray(k)
        return np.zeros(k.shape, bool) if self.loop else k >= self.n - 1

    def _transform(self, k, origin=None):
        i, c = self.motion._index(k)
        T = self.motion.cycle_transform(c)
        if origin is not None:
            T = planar_compose(np.broadcast_to(origin, T.shape), T)
        return i, T

    def frames(self, k, origin: np.ndarray | None = None, fields=("qpos", "qvel", "keys", "com", "com_vel",
                                                                   "ctrl")) -> dict:
        """Reference at integer frames k (N,) moved by planar anchors origin (N, 3) (x, y, yaw) or None."""
        i, T = self._transform(np.asarray(k), origin)
        out = {}
        qa, da = self.qa, self.da
        if "qpos" in fields:
            q = self.qpos[i].copy()
            q[:, qa:qa + 3] = planar_apply_points(T, q[:, qa:qa + 3])
            q[:, qa + 3:qa + 7] = planar_apply_quats(T, q[:, qa + 3:qa + 7])
            out["qpos"] = q
        if "qvel" in fields:
            v = self.qvel[i].copy()
            v[:, da:da + 3] = planar_apply_vectors(T, v[:, da:da + 3])
            out["qvel"] = v
        if "keys" in fields:
            out["keys"] = planar_apply_points(T, self.keys[i])
        if "com" in fields:
            out["com"] = planar_apply_points(T, self.com[i])
        if "com_vel" in fields:
            out["com_vel"] = planar_apply_vectors(T, self.com_vel[i])
        if "ctrl" in fields:
            out["ctrl"] = self.ctrl[i]
        return out
