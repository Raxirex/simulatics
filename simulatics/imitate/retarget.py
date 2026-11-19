"""Retarget human motion capture (BVH) to the robot, with a key-body recipe.

Kinematic retargeting in recent humanoid tracking work follows a short recipe: match human and
robot *key bodies*, scale the human locally to the robot's size, solve inverse kinematics per frame on
those key bodies (warm-started from the previous frame, within joint limits) and post-process the result.
This module implements that recipe from the description, with MuJoCo's own Jacobians:

1. **Key-body matching.** Human joints are mapped to robot key points (simulatics.imitate.motion.
   robot_keypoints) by name: pelvis (hip-joint midpoint), torso (shoulder midpoint), head, hips, knees,
   ankles, shoulders, elbows and wrists. Common naming schemes (common BVH and
   "LeftUpLeg", "l_knee", "Hip_L", ASF-style "lfemur") are recognised; an explicit `joint_map` overrides.
2. **Scaling by leg length.** Trunk and leg targets are the human positions times s = robot leg length /
   human leg length (hip-knee-ankle), so step length and pelvis height scale with the legs. Arms are
   scaled separately (s_arm = robot / human shoulder-elbow-wrist) and expressed relative to the
   shoulder, so a human with longer or shorter arms still reaches the right place relative to the robot's
   own shoulder ("non-uniform local scaling").
3. **Orientation targets.** The pelvis and feet take the rotation of the corresponding human joint
   relative to its rest (T-) pose; the torso frame comes from the shoulder line and the spine direction.
4. **Per-frame IK.** Damped least squares (Wampler 1986; Buss 2004) on all key-body position, relative
   position and orientation errors at once: dq = (J^T W J + lambda^2 I)^-1 J^T W e with J from mj_jac /
   mj_jacBody, the free root included. Joints at a limit that the step would push further out are
   frozen and the step re-solved (the clamping loop of Baerlocher & Boulic 2004); the result is clipped
   to the joint ranges. Each frame starts from the previous solution.
5. **Ground alignment.** A global height shift puts the feet on the floor (5th percentile of the lowest
   sole point over the clip at z = 0, robust to a few bad frames).
6. **Foot-skate cleanup.** A foot is in stance when its sole is within `stance_height` of the floor and
   moves slower than `stance_speed` leg lengths per second. Each stance segment is locked to one flat pose
   (median position, mean yaw) and those frames are re-solved with the foot pinned, blending into and out
   of the lock over a few frames - the standard mocap clean-up for sliding feet.

The output is a `Motion` (root + joint angles + key-body positions) at the requested frame rate, plus a
report with the per-key retargeting error.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .bvh import Bvh, load_bvh, parse_bvh, up_conversion
from .motion import (Motion, mat_to_quat, mat_to_rotvec, quat_slerp, quat_to_mat, robot_keypoints, rot_z,
                     yaw_of_mat)

SIDES = ("left", "right")
LIMB_PARTS = ("hip", "knee", "foot", "toe", "shoulder", "elbow", "hand", "clavicle", "finger")
# (part, keywords in order of preference). Checked in this order, so "forearm" is an elbow before it can
# look like an "arm" and "upleg" is a hip before it can look like a "leg".
_PART_RULES = [
    ("skip", ("end", "nub", "site", "twist", "roll", "dummy", "top", "effector")),
    ("finger", ("thumb", "index", "middle", "ring", "pinky", "finger", "little", "digit", "carpal")),
    ("toe", ("toe", "ball")),
    ("foot", ("ankle", "foot")),
    ("hip", ("upleg", "upperleg", "thigh", "femur", "hip")),
    ("knee", ("knee", "lowerleg", "shin", "calf", "tibia", "crus", "leg")),
    ("hand", ("wrist", "hand", "palm")),
    ("elbow", ("forearm", "lowerarm", "elbow", "radius", "ulna")),
    ("shoulder", ("upperarm", "humerus", "arm", "shoulder")),
    ("clavicle", ("clavicle", "collar", "scapula")),
    ("head", ("head", "skull")),
    ("neck", ("neck",)),
    ("spine", ("spine", "chest", "torso", "back", "abdomen", "thorax", "waist", "belly")),
    ("pelvis", ("hips", "pelvis", "root")),
]
_SIDE_WORDS = {"left": "left", "l": "left", "lft": "left", "right": "right", "r": "right", "rgt": "right"}
ROLES = ("pelvis", "head", "neck") + tuple(f"{s}_{p}" for s in SIDES
                                           for p in ("hip", "knee", "foot", "toe", "shoulder", "elbow", "hand"))


# =========================================================================================================
# Joint name matching
# =========================================================================================================
def _tokens(name: str) -> list[str]:
    base = re.split(r"[:|]", name)[-1]
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", base)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


def _part(joined: str) -> tuple[str | None, int]:
    for part, kws in _PART_RULES:
        for rank, kw in enumerate(kws):
            if kw in joined:
                return part, rank
    return None, 0


def classify_joint(name: str) -> tuple[str | None, str | None, int]:
    """-> (side "left"/"right"/None, part, rank) of a mocap joint name; part None if unrecognised.
    rank orders synonyms (0 = preferred), e.g. "LeftArm" (rank 2) beats "LeftShoulder" (3) as the
    shoulder joint, "LeftUpLeg" beats "LHipJoint"."""
    toks = _tokens(name)
    side, rest = None, []
    for t in toks:
        if side is None and t in _SIDE_WORDS and len(toks) > 1:
            side = _SIDE_WORDS[t]
            continue
        rest.append(t)
    joined = "".join(rest)
    part, rank = _part(joined)
    if part == "skip":
        return side, None, 0
    if part in LIMB_PARTS and side is None:
        if len(joined) > 1 and joined[0] in "lr" and _part(joined[1:])[0] == part:
            side = "left" if joined[0] == "l" else "right"
        elif part == "hip":
            part, rank = "pelvis", 3
        else:
            part = None
    if part not in LIMB_PARTS:
        side = None
    return side, part, rank


def guess_joint_map(bvh: Bvh) -> dict[str, int]:
    """Role -> BVH joint index for every role whose joint could be identified by name (see ROLES)."""
    depth = []
    for j in bvh.joints:
        depth.append(0 if j.parent < 0 else depth[j.parent] + 1)
    best: dict[str, tuple[int, int, int]] = {}
    for ji, j in enumerate(bvh.joints):
        side, part, rank = classify_joint(j.name)
        if part is None:
            continue
        role = f"{side}_{part}" if side else part
        if role not in ROLES:
            continue
        # among synonyms prefer the better keyword; for ties the joint nearer the root (a hand before its
        # finger roots, the first spine joint before the next)
        key = (rank, depth[ji], ji)
        if role not in best or key < best[role]:
            best[role] = key
    out = {r: v[2] for r, v in best.items()}
    if "pelvis" not in out:
        out["pelvis"] = next(i for i, j in enumerate(bvh.joints) if j.parent < 0)
    return out


REQUIRED_ROLES = ("left_hip", "right_hip", "left_knee", "right_knee", "left_foot", "right_foot")


# =========================================================================================================
# Whole-body IK
# =========================================================================================================
@dataclass
class IKTask:
    """One IK objective. kind "pos": world point of (body, offset) -> target (3,); "rel": that point minus
    the point (base_body, base_offset) -> target (3,); "ori": body orientation -> target (3, 3)."""
    kind: str
    body: int
    target: np.ndarray
    weight: float = 1.0
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_body: int = -1
    base_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    name: str = ""


def ik_dof_mask(model: mujoco.MjModel, exclude=("gripper", "finger", "wrist"), include_root: bool = True,
                only: tuple[str, ...] | None = None) -> np.ndarray:
    """Boolean mask over qvel of the DOFs the IK may move. `only` = substrings of the joint names to keep."""
    mask = np.zeros(model.nv, bool)
    for j in range(model.njnt):
        name = model.joint(j).name.lower()
        da = int(model.jnt_dofadr[j])
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            mask[da:da + 6] = include_root and only is None
            continue
        nd = 3 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1
        ok = (any(k in name for k in only) if only is not None else not any(k in name for k in exclude))
        mask[da:da + nd] = ok
    return mask


class WholeBodyIK:
    """Damped-least-squares IK over key-body tasks with joint limits (see the module docstring)."""

    def __init__(self, model: mujoco.MjModel, dofs: np.ndarray | None = None, damping: float = 0.03,
                 max_step: float = 0.25):
        self.model = model
        self.data = mujoco.MjData(model)
        self.dofs = ik_dof_mask(model) if dofs is None else np.asarray(dofs, bool)
        self.damping, self.max_step = float(damping), float(max_step)
        nv = model.nv
        self.lim_dof = np.zeros(nv, bool)
        self.lo = np.full(nv, -np.inf)
        self.hi = np.full(nv, np.inf)
        self.dof_qadr = np.full(nv, -1)
        for j in range(model.njnt):
            if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                d = int(model.jnt_dofadr[j])
                self.dof_qadr[d] = int(model.jnt_qposadr[j])
                if model.jnt_limited[j]:
                    self.lim_dof[d] = True
                    self.lo[d], self.hi[d] = model.jnt_range[j]
        self._jp = np.zeros((3, nv))
        self._jr = np.zeros((3, nv))
        self._jp2 = np.zeros((3, nv))

    def _fk(self, q: np.ndarray) -> None:
        d = self.data
        d.qpos[:] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)

    def _point(self, body: int, off: np.ndarray) -> np.ndarray:
        return self.data.xpos[body] + self.data.xmat[body].reshape(3, 3) @ off

    def residuals(self, tasks: list[IKTask], with_jac: bool = True):
        """(weighted Jacobian rows, weighted errors, unweighted error norm per task) at the current FK."""
        m, d = self.model, self.data
        rows, errs, norms = [], [], []
        for t in tasks:
            if t.kind == "ori":
                R = d.xmat[t.body].reshape(3, 3)
                e = mat_to_rotvec(t.target @ R.T)
                if with_jac:
                    mujoco.mj_jacBody(m, d, None, self._jr, t.body)
                    J = self._jr.copy()
            else:
                p = self._point(t.body, t.offset)
                if t.kind == "rel":
                    pb = self._point(t.base_body, t.base_offset)
                    e = t.target - (p - pb)
                    if with_jac:
                        mujoco.mj_jac(m, d, self._jp, None, p, t.body)
                        mujoco.mj_jac(m, d, self._jp2, None, pb, t.base_body)
                        J = self._jp - self._jp2
                else:
                    e = t.target - p
                    if with_jac:
                        mujoco.mj_jac(m, d, self._jp, None, p, t.body)
                        J = self._jp.copy()
            norms.append(float(np.linalg.norm(e)))
            errs.append(t.weight * e)
            if with_jac:
                rows.append(t.weight * J)
        if not with_jac:
            return None, np.concatenate(errs) if errs else np.zeros(0), np.array(norms)
        return np.vstack(rows), np.concatenate(errs), np.array(norms)

    def solve(self, q0: np.ndarray, tasks: list[IKTask], iters: int = 20, tol: float = 1e-6,
              posture: np.ndarray | None = None, posture_weight: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Iterate DLS steps from q0. Returns (qpos, per-task error norms at the returned qpos)."""
        m = self.model
        q = np.array(q0, float)
        lam2 = self.damping ** 2
        hinge = self.dof_qadr >= 0
        for _ in range(max(0, iters)):
            self._fk(q)
            J, e, _ = self.residuals(tasks)
            active = self.dofs.copy()
            at_lo = self.lim_dof & (q[np.maximum(self.dof_qadr, 0)] <= self.lo + 1e-7)
            at_hi = self.lim_dof & (q[np.maximum(self.dof_qadr, 0)] >= self.hi - 1e-7)
            dq = np.zeros(m.nv)
            for _pass in range(4):
                idx = np.flatnonzero(active)
                Ja = J[:, idx]
                A = Ja.T @ Ja
                A[np.diag_indices_from(A)] += lam2
                b = Ja.T @ e
                if posture is not None and posture_weight > 0:
                    h = hinge[idx]
                    A[np.diag_indices_from(A)] += posture_weight * h
                    b += posture_weight * h * (posture[np.maximum(self.dof_qadr[idx], 0)]
                                               - q[np.maximum(self.dof_qadr[idx], 0)])
                dq[:] = 0.0
                dq[idx] = np.linalg.solve(A, b)
                push = active & ((at_lo & (dq < 0)) | (at_hi & (dq > 0)))
                if not push.any():
                    break
                active &= ~push
            step = float(np.abs(dq).max()) if dq.size else 0.0
            if step > self.max_step:
                dq *= self.max_step / step
            mujoco.mj_integratePos(m, q, dq, 1.0)
            lim_q = self.dof_qadr[self.lim_dof]
            q[lim_q] = np.clip(q[lim_q], self.lo[self.lim_dof], self.hi[self.lim_dof])
            if step < tol:
                break
        self._fk(q)
        _, _, norms = self.residuals(tasks, with_jac=False)
        return q, norms


# =========================================================================================================
# Human data in robot coordinates
# =========================================================================================================
@dataclass
class HumanMotion:
    """Human key-joint trajectories in the robot's world convention (Z up, facing +X at frame 0).

    pos: role -> (T, 3) joint positions (human units: only ratios matter); rot: "pelvis" / "left_foot" /
    "right_foot" -> (T, 3, 3) rotations relative to the rest pose (a flat foot / upright pelvis facing +X is
    the identity); leg_length / arm_length in the same units as pos."""
    fps: float
    pos: dict[str, np.ndarray]
    rot: dict[str, np.ndarray]
    leg_length: float
    arm_length: float | None
    floor: float = 0.0
    info: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(next(iter(self.pos.values())))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _common_ancestor(bvh: Bvh, a: int, b: int) -> int:
    """Lowest common ancestor joint of joints a and b (the chest for the two shoulders), -1 if none."""
    chain = set()
    while a >= 0:
        a = bvh.joints[a].parent
        if a >= 0:
            chain.add(a)
    while b >= 0:
        b = bvh.joints[b].parent
        if b in chain:
            return b
    return -1


def detect_up_axis(bvh: Bvh, jmap: dict[str, int]) -> str:
    """Up axis of a BVH from its rest pose: the dominant axis of (shoulders or head) - feet."""
    rest, _, _ = bvh.forward_kinematics(rest=True)
    rest = rest[0]
    top = [jmap[r] for r in ("head", "left_shoulder", "right_shoulder", "neck") if r in jmap]
    if not top:
        top = [jmap["pelvis"]]
    feet = [jmap[r] for r in ("left_foot", "right_foot") if r in jmap]
    v = rest[top].mean(axis=0) - rest[feet].mean(axis=0)
    ax = int(np.argmax(np.abs(v)))
    return ("-" if v[ax] < 0 else "") + "xyz"[ax]


def bvh_to_human(bvh: Bvh, joint_map: dict[str, int | str] | None = None, up_axis: str = "auto",
                 align_heading: bool = True, fps: float | None = None) -> HumanMotion:
    """Forward kinematics of a BVH, converted to Z-up / facing +X and resampled to `fps` (default: the file's)."""
    jmap = guess_joint_map(bvh)
    for role, j in (joint_map or {}).items():
        jmap[role] = bvh.index(j) if isinstance(j, str) else int(j)
    missing = [r for r in REQUIRED_ROLES if r not in jmap]
    if missing:
        raise ValueError(f"cannot find BVH joints for {missing}; joints are {bvh.names}. Pass joint_map="
                         "{role: joint name} (roles: " + ", ".join(ROLES) + ")")
    if bvh.n_frames < 1:
        raise ValueError("the BVH file has no frames")
    up = detect_up_axis(bvh, jmap) if up_axis == "auto" else up_axis
    P = up_conversion(up)          # bvh = P robot  ->  robot = P^T bvh
    pos, rot, end = bvh.forward_kinematics()
    rpos, rrot, rend = bvh.forward_kinematics(rest=True)
    pos = pos @ P                  # row vectors: (P^T x)^T = x^T P
    rpos = rpos[0] @ P
    # rest heading (the direction the rest pose faces, from its hip line) and frame-0 heading
    def heading(lh, rh):
        left = lh - rh
        fwd = np.cross(left, np.array([0.0, 0.0, 1.0]))
        return float(np.arctan2(fwd[1], fwd[0]))
    psi_rest = heading(rpos[jmap["left_hip"]], rpos[jmap["right_hip"]])
    psi0 = heading(pos[0, jmap["left_hip"]], pos[0, jmap["right_hip"]]) if align_heading else 0.0
    A = rot_z(-psi0)
    origin = np.zeros(3)
    if align_heading:
        origin[:2] = (pos[0, jmap["left_hip"], :2] + pos[0, jmap["right_hip"], :2]) / 2
    pos = (pos - origin) @ A.T
    out_pos = {r: pos[:, j].copy() for r, j in jmap.items()}
    # toes: use a toe joint if present, else a foot End Site
    for s in SIDES:
        if f"{s}_toe" not in out_pos and f"{s}_foot" in jmap and np.isfinite(end[0, jmap[f"{s}_foot"]]).all():
            out_pos[f"{s}_toe"] = (((end[:, jmap[f"{s}_foot"]] @ P) - origin) @ A.T)
    # floor: lowest foot / toe point over the clip
    low = [out_pos[k][:, 2] for k in out_pos if k.endswith("_foot") or k.endswith("_toe")]
    floor = float(np.min(np.concatenate(low)))
    # rotations relative to the rest pose: R = A P^T G(t) G_rest^T P Rz(psi_rest)
    Rrest = rot_z(psi_rest)
    out_rot = {}
    rot_src = {r: jmap[r] for r in ("pelvis", "left_foot", "right_foot") if r in jmap}
    if "left_shoulder" in jmap and "right_shoulder" in jmap:
        chest = _common_ancestor(bvh, jmap["left_shoulder"], jmap["right_shoulder"])
        if chest >= 0:
            rot_src["torso"] = chest
    for role, j in rot_src.items():
        G = rot[:, j] @ rrot[0, j].T
        out_rot[role] = A @ P.T @ G @ P @ Rrest
    # lengths from the rest pose
    def chain(a, b, c):
        return float(np.linalg.norm(rpos[jmap[b]] - rpos[jmap[a]]) + np.linalg.norm(rpos[jmap[c]] - rpos[jmap[b]]))
    leg = np.mean([chain(f"{s}_hip", f"{s}_knee", f"{s}_foot") for s in SIDES])
    arm = None
    if all(f"{s}_{p}" in jmap for s in SIDES for p in ("shoulder", "elbow", "hand")):
        arm = float(np.mean([chain(f"{s}_shoulder", f"{s}_elbow", f"{s}_hand") for s in SIDES]))
    hm = HumanMotion(bvh.fps, out_pos, out_rot, float(leg), arm, floor,
                     {"up_axis": up, "joint_map": {r: bvh.joints[j].name for r, j in jmap.items()},
                      "frames_in": bvh.n_frames, "fps_in": bvh.fps})
    return resample_human(hm, fps) if fps else hm


def resample_human(h: HumanMotion, fps: float) -> HumanMotion:
    n = h.n_frames
    if n < 2 or abs(fps - h.fps) < 1e-9:
        return h
    dur = (n - 1) / h.fps
    m = int(np.floor(dur * fps + 1e-9)) + 1
    f = np.minimum(np.arange(m) / fps * h.fps, n - 1)
    i0 = np.minimum(np.floor(f).astype(int), n - 2)
    a = f - i0
    pos = {k: v[i0] + (v[i0 + 1] - v[i0]) * a[:, None] for k, v in h.pos.items()}
    rot = {}
    for k, v in h.rot.items():
        q = mat_to_quat(v)
        rot[k] = quat_to_mat(quat_slerp(q[i0], q[i0 + 1], a))
    return HumanMotion(fps, pos, rot, h.leg_length, h.arm_length, h.floor, dict(h.info))


# =========================================================================================================
# Retargeting
# =========================================================================================================
DEFAULT_WEIGHTS = {"pelvis": 3.0, "pelvis_ori": 1.0, "torso": 1.0, "torso_ori": 0.6, "head": 0.3, "hip": 0.5,
                   "knee": 1.0, "foot": 3.0, "foot_ori": 0.8, "elbow": 1.0, "hand": 2.0,
                   "foot_lock": 8.0, "foot_lock_ori": 3.0}


@dataclass
class RetargetConfig:
    fps: float = 50.0                 # output frame rate
    up_axis: str = "auto"             # "auto" | "y" | "z" | "x" (optionally "-y" ...)
    align_heading: bool = True        # start at the origin facing +X
    leg_scale: float | None = None    # None: robot leg length / human leg length
    arm_scale: float | None = None    # None: robot arm length / human arm length
    iters: int = 25                   # IK iterations per frame (warm-started)
    first_iters: int = 200
    damping: float = 0.03
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    ground: bool = True
    foot_cleanup: bool = True
    stance_speed: float = 0.25        # leg lengths per second
    stance_height: float = 0.03       # m above the floor (lowest sole point)
    blend_frames: int = 4
    min_stance_frames: int = 3
    loop: bool = False


@dataclass
class RetargetResult:
    motion: Motion
    errors: dict                      # key -> {"mean", "max"} metres (rotations: radians), + "mean"/"max"
    errors_before_cleanup: dict
    scale: dict
    stance: np.ndarray                # (T, 2) bool, left / right
    report: dict


def _foot_geom_corners(model: mujoco.MjModel) -> dict[str, tuple[int, np.ndarray, np.ndarray]]:
    """side -> (foot body, sole centre in the body frame, 8 box corners in the body frame)."""
    out = {}
    for s in SIDES:
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_foot")
        if g < 0:
            continue
        b = int(model.geom_bodyid[g])
        R = quat_to_mat(model.geom_quat[g])
        size = model.geom_size[g]
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * size
        else:
            corners = np.array([[0.0, 0.0, -float(model.geom_rbound[g])]])
        corners = model.geom_pos[g] + corners @ R.T
        sole = model.geom_pos[g] + R @ np.array([0.0, 0.0, -size[2]])
        out[s] = (b, sole, corners)
    return out


def _feet_state(model, data, feet) -> dict:
    """side -> (sole centre world, foot yaw, lowest corner z) after mj_kinematics."""
    out = {}
    for s, (b, sole, corners) in feet.items():
        R = data.xmat[b].reshape(3, 3)
        p = data.xpos[b]
        out[s] = (p + R @ sole, float(yaw_of_mat(R)), float(np.min(p[2] + corners @ R[2])))
    return out


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(mask) - 1))
    return segs


def retarget_human(human: HumanMotion, model: mujoco.MjModel, cfg: RetargetConfig | None = None,
                   name: str = "retargeted", meta: dict | None = None) -> RetargetResult:
    """Solve the robot motion that matches `human` (see the module docstring)."""
    cfg = cfg or RetargetConfig()
    w = {**DEFAULT_WEIGHTS, **(cfg.weights or {})}
    kp = robot_keypoints(model)
    for k in ("pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_foot", "right_foot"):
        if k not in kp:
            raise ValueError(f"robot has no key point {k!r}: retargeting needs the humanoid's legs")
    feet = _foot_geom_corners(model)
    ik = WholeBodyIK(model, damping=cfg.damping)
    d0 = mujoco.MjData(model)
    mujoco.mj_kinematics(model, d0)
    P0 = {k: d0.xpos[b] + d0.xmat[b].reshape(3, 3) @ o for k, (b, o) in kp.items()}
    robot_leg = float(np.mean([np.linalg.norm(P0[f"{s}_knee"] - P0[f"{s}_hip"])
                               + np.linalg.norm(P0[f"{s}_foot"] - P0[f"{s}_knee"]) for s in SIDES]))
    has_arms = all(f"{s}_{p}" in kp for s in SIDES for p in ("shoulder", "elbow", "hand"))
    robot_arm = (float(np.mean([np.linalg.norm(P0[f"{s}_elbow"] - P0[f"{s}_shoulder"])
                                + np.linalg.norm(P0[f"{s}_hand"] - P0[f"{s}_elbow"]) for s in SIDES]))
                 if has_arms else None)
    s_leg = cfg.leg_scale or robot_leg / human.leg_length
    s_arm = cfg.arm_scale or ((robot_arm / human.arm_length) if (robot_arm and human.arm_length) else s_leg)
    H = {k: v.copy() for k, v in human.pos.items()}
    for v in H.values():
        v[:, 2] -= human.floor
    T = human.n_frames

    # ---- targets
    tgt: dict[str, np.ndarray] = {}
    tgt["pelvis"] = s_leg * (H["left_hip"] + H["right_hip"]) / 2
    for s in SIDES:
        for p in ("hip", "knee", "foot"):
            tgt[f"{s}_{p}"] = s_leg * H[f"{s}_{p}"]
    arms_ok = has_arms and all(f"{s}_{p}" in H for s in SIDES for p in ("shoulder", "elbow", "hand"))
    if all(f"{s}_shoulder" in H for s in SIDES) and "torso" in kp:
        tgt["torso"] = s_leg * (H["left_shoulder"] + H["right_shoulder"]) / 2
        if "torso" in human.rot:      # the chest joint's rotation from its rest pose
            tgt["torso_ori"] = human.rot["torso"]
        else:                         # shoulder line + spine direction
            y = _unit(H["left_shoulder"] - H["right_shoulder"])
            up = _unit((H["left_shoulder"] + H["right_shoulder"]) / 2 - (H["left_hip"] + H["right_hip"]) / 2)
            x = _unit(np.cross(y, up))
            z = np.cross(x, y)
            tgt["torso_ori"] = np.stack([x, y, z], axis=-1)
    if "head" in H and "head" in kp:
        tgt["head"] = s_leg * H["head"]
    if arms_ok:
        for s in SIDES:
            tgt[f"{s}_elbow_rel"] = s_arm * (H[f"{s}_elbow"] - H[f"{s}_shoulder"])
            tgt[f"{s}_hand_rel"] = s_arm * (H[f"{s}_hand"] - H[f"{s}_shoulder"])
    if "pelvis" in human.rot:
        tgt["pelvis_ori"] = human.rot["pelvis"]
    else:   # hip line + vertical
        y = _unit(H["left_hip"] - H["right_hip"])
        y[:, 2] = 0
        y = _unit(y)
        z = np.tile([0.0, 0.0, 1.0], (T, 1))
        tgt["pelvis_ori"] = np.stack([np.cross(y, z), y, z], axis=-1)
    for s in SIDES:
        if f"{s}_foot" in human.rot:
            tgt[f"{s}_foot_ori"] = human.rot[f"{s}_foot"]

    def tasks_at(t: int, lock: dict | None = None) -> list[IKTask]:
        ts = [IKTask("pos", kp["pelvis"][0], tgt["pelvis"][t], w["pelvis"], kp["pelvis"][1], name="pelvis"),
              IKTask("ori", kp["pelvis"][0], tgt["pelvis_ori"][t], w["pelvis_ori"], name="pelvis_ori")]
        for s in SIDES:
            ts.append(IKTask("pos", kp[f"{s}_hip"][0], tgt[f"{s}_hip"][t], w["hip"], kp[f"{s}_hip"][1],
                             name=f"{s}_hip"))
            ts.append(IKTask("pos", kp[f"{s}_knee"][0], tgt[f"{s}_knee"][t], w["knee"], kp[f"{s}_knee"][1],
                             name=f"{s}_knee"))
            if lock is not None and s in lock:
                p, R, a = lock[s]
                ts.append(IKTask("pos", kp[f"{s}_foot"][0], p, w["foot"] + a * (w["foot_lock"] - w["foot"]),
                                 kp[f"{s}_foot"][1], name=f"{s}_foot"))
                ts.append(IKTask("ori", kp[f"{s}_foot"][0], R, w["foot_ori"] + a * (w["foot_lock_ori"] - w["foot_ori"]),
                                 name=f"{s}_foot_ori"))
            else:
                ts.append(IKTask("pos", kp[f"{s}_foot"][0], tgt[f"{s}_foot"][t], w["foot"], kp[f"{s}_foot"][1],
                                 name=f"{s}_foot"))
                if f"{s}_foot_ori" in tgt:
                    ts.append(IKTask("ori", kp[f"{s}_foot"][0], tgt[f"{s}_foot_ori"][t], w["foot_ori"],
                                     name=f"{s}_foot_ori"))
        if "torso" in tgt:
            ts.append(IKTask("pos", kp["torso"][0], tgt["torso"][t], w["torso"], kp["torso"][1], name="torso"))
            ts.append(IKTask("ori", kp["torso"][0], tgt["torso_ori"][t], w["torso_ori"], name="torso_ori"))
        if "head" in tgt:
            ts.append(IKTask("pos", kp["head"][0], tgt["head"][t], w["head"], kp["head"][1], name="head"))
        if arms_ok:
            for s in SIDES:
                sb, so = kp[f"{s}_shoulder"]
                for p in ("elbow", "hand"):
                    ts.append(IKTask("rel", kp[f"{s}_{p}"][0], tgt[f"{s}_{p}_rel"][t], w[p], kp[f"{s}_{p}"][1],
                                     sb, so, name=f"{s}_{p}"))
        return ts

    # ---- initial pose: the model's default posture under the first pelvis target
    q = model.qpos0.copy()
    from .motion import free_root
    _, qa, _ = free_root(model)
    q[qa:qa + 3] += tgt["pelvis"][0] - P0["pelvis"]
    q[qa + 3:qa + 7] = mat_to_quat(rot_z(yaw_of_mat(tgt["pelvis_ori"][0])))
    qs = np.zeros((T, model.nq))
    names0: list[str] = []
    norms = []
    for t in range(T):
        q, nrm = ik.solve(q, tasks_at(t), cfg.first_iters if t == 0 else cfg.iters)
        qs[t] = q
        norms.append(nrm)
        if t == 0:
            names0 = [tk.name for tk in tasks_at(0)]
    errors_before = _summarise(names0, np.array(norms))

    # ---- ground alignment: 5th percentile of the lowest sole point at z = 0
    d = ik.data
    def lowest(qs_):
        out = np.zeros(len(qs_))
        for t, qq in enumerate(qs_):
            d.qpos[:] = qq
            mujoco.mj_kinematics(model, d)
            out[t] = min(v[2] for v in _feet_state(model, d, feet).values()) if feet else qq[qa + 2]
        return out
    shift = 0.0
    if cfg.ground and feet:
        shift = -float(np.percentile(lowest(qs), 5))
        qs[:, qa + 2] += shift
        for k, v in tgt.items():
            if not (k.endswith("_ori") or k.endswith("_rel")):
                v[:, 2] += shift

    # ---- foot-skate cleanup
    stance = np.zeros((T, 2), bool)
    skate_before = skate_after = 0.0
    if cfg.foot_cleanup and feet and T >= 3:
        sole = np.zeros((T, 2, 3))
        yaw = np.zeros((T, 2))
        low = np.zeros((T, 2))
        for t in range(T):
            d.qpos[:] = qs[t]
            mujoco.mj_kinematics(model, d)
            st = _feet_state(model, d, feet)
            for i, s in enumerate(SIDES):
                sole[t, i], yaw[t, i], low[t, i] = st[s]
        fps = human.fps
        vel = np.zeros((T, 2))
        vel[1:-1] = np.linalg.norm(sole[2:, :, :2] - sole[:-2, :, :2], axis=-1) * fps / 2
        vel[0], vel[-1] = vel[1], vel[-2]
        stance = (low < cfg.stance_height) & (vel < cfg.stance_speed * robot_leg)
        locks: list[dict] = [dict() for _ in range(T)]
        for i, s in enumerate(SIDES):
            # close one-frame gaps, then drop segments that are too short
            m = stance[:, i].copy()
            for t in range(1, T - 1):
                if not m[t] and m[t - 1] and m[t + 1]:
                    m[t] = True
            segs = [(a, b) for a, b in _segments(m) if b - a + 1 >= cfg.min_stance_frames]
            m[:] = False
            for a, b in segs:
                m[a:b + 1] = True
            stance[:, i] = m
            b_id, sole_local, _ = feet[s]
            k_off = kp[f"{s}_foot"][1]
            for a, b in segs:
                xy = np.median(sole[a:b + 1, i, :2], axis=0)
                yw = float(np.arctan2(np.mean(np.sin(yaw[a:b + 1, i])), np.mean(np.cos(yaw[a:b + 1, i]))))
                R = rot_z(yw)
                p_body = np.array([xy[0], xy[1], 0.0]) - R @ sole_local
                p_key = p_body + R @ k_off
                for t in range(max(0, a - cfg.blend_frames), min(T, b + 1 + cfg.blend_frames)):
                    alpha = 1.0 if a <= t <= b else 1.0 - (min(abs(t - a), abs(t - b)) / (cfg.blend_frames + 1))
                    if s in locks[t] and locks[t][s][2] >= alpha:
                        continue
                    if alpha >= 1.0:
                        locks[t][s] = (p_key, R, 1.0)
                    else:  # blend from the free target into the lock
                        p0 = tgt[f"{s}_foot"][t]
                        R0 = tgt.get(f"{s}_foot_ori", np.broadcast_to(R, (T, 3, 3)))[t]
                        q0, q1 = mat_to_quat(R0), mat_to_quat(R)
                        locks[t][s] = (p0 + alpha * (p_key - p0), quat_to_mat(quat_slerp(q0, q1, alpha)), alpha)
        skate_before = float(np.mean(vel[stance])) if stance.any() else 0.0
        touched = [t for t in range(T) if locks[t]]
        touched_set = set(touched)
        for t in touched:
            q0 = qs[t - 1] if (t - 1) in touched_set else qs[t]
            qs[t], norms[t] = ik.solve(q0, tasks_at(t, locks[t]), cfg.iters + 10)
        # re-measure skate
        for t in range(T):
            d.qpos[:] = qs[t]
            mujoco.mj_kinematics(model, d)
            st = _feet_state(model, d, feet)
            for i, s in enumerate(SIDES):
                sole[t, i] = st[s][0]
        vel2 = np.zeros((T, 2))
        vel2[1:-1] = np.linalg.norm(sole[2:, :, :2] - sole[:-2, :, :2], axis=-1) * fps / 2
        vel2[0], vel2[-1] = vel2[1], vel2[-2]
        skate_after = float(np.mean(vel2[stance])) if stance.any() else 0.0

    # ---- errors against the (ground-shifted) free targets
    final = []
    for t in range(T):
        ik._fk(qs[t])
        _, _, nrm = ik.residuals(tasks_at(t), with_jac=False)
        final.append(nrm)
    errors = _summarise(names0, np.array(final))
    lows = lowest(qs) if feet else np.zeros(T)
    mmeta = {"source": "retarget", **(meta or {})}
    if cfg.loop and T > 3:   # the clip's last frame is one period after its first
        motion = Motion.from_cycle(model, qs, human.fps, name=name, meta=mmeta)
    else:
        motion = Motion.from_qpos(model, qs, human.fps, name=name, loop=False, meta=mmeta)
    report = {"frames": T, "fps": human.fps, "leg_scale": s_leg, "arm_scale": s_arm, "ground_shift_m": shift,
              "stance_fraction": [float(stance[:, 0].mean()), float(stance[:, 1].mean())],
              "skate_speed_before_mps": skate_before, "skate_speed_after_mps": skate_after,
              "min_sole_height_m": float(lows.min()) if T else 0.0, "key_error_mean_m": errors["mean"],
              "key_error_max_m": errors["max"], **human.info}
    return RetargetResult(motion, errors, errors_before, {"leg": s_leg, "arm": s_arm}, stance, report)


def _summarise(names: list[str], norms: np.ndarray) -> dict:
    """Per-task mean/max error; overall mean/max over the position tasks (metres)."""
    out: dict = {}
    pos_cols = []
    for i, n in enumerate(names):
        col = norms[:, i]
        out[n] = {"mean": float(col.mean()), "max": float(col.max())}
        if not n.endswith("_ori"):
            pos_cols.append(i)
    p = norms[:, pos_cols] if pos_cols else np.zeros((1, 1))
    out["mean"] = float(p.mean())
    out["max"] = float(p.max())
    return out


def retarget_bvh(source: str | Path | Bvh, model: mujoco.MjModel, cfg: RetargetConfig | None = None,
                 joint_map: dict | None = None, name: str | None = None) -> RetargetResult:
    """BVH file / text / parsed Bvh -> robot Motion + report. See the module docstring for the method."""
    cfg = cfg or RetargetConfig()
    src_name = "bvh"
    if isinstance(source, Bvh):
        bvh = source
    else:
        s = str(source)
        if "\n" not in s and Path(s).exists():
            bvh = load_bvh(s)
            src_name = Path(s).stem
        else:
            bvh = parse_bvh(s)
    human = bvh_to_human(bvh, joint_map, cfg.up_axis, cfg.align_heading, cfg.fps)
    return retarget_human(human, model, cfg, name=name or src_name,
                          meta={"bvh": src_name, "bvh_joint_map": human.info.get("joint_map", {})})
