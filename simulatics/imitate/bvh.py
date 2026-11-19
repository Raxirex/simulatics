"""Biovision Hierarchy (BVH) motion files: parser, writer, forward kinematics and an exporter for robot motions.

Written from the format description (Biovision's BVH, as documented by Meredith & Maddock, "Motion capture
file formats explained", University of Sheffield, 2001), not from any existing parser:

    HIERARCHY
    ROOT Hips
    {
        OFFSET 0 0 0
        CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
        JOINT LeftUpLeg
        {
            OFFSET 9 0 0
            CHANNELS 3 Zrotation Xrotation Yrotation
            ...
            End Site
            {
                OFFSET 0 -5 10
            }
        }
    }
    MOTION
    Frames: 2
    Frame Time: 0.0333333
    <one line of channel values per frame, in hierarchy order>

Conventions implemented here:

* OFFSET is the joint's position in its parent's frame (any length unit; retargeting rescales).
* Rotation channels are Euler angles in degrees, applied in the order listed: "Zrotation Xrotation
  Yrotation" means R_local = Rz(a) Rx(b) Ry(c), and a point in the joint frame maps to the parent frame
  as p_parent = offset + R_local p (intrinsic rotations about the moving axes, the usual BVH reading).
* Position channels give the joint's translation in its parent frame and replace the matching OFFSET
  component (the root's channels are its absolute position; files that also put position channels on
  inner joints store the bone offset there).
* Any subset and order of the six channels per joint is accepted; "End Site" blocks are leaf points.

The exporter writes a robot `Motion` as BVH (Y-up or Z-up, any rotation order), so robot motions can be
viewed in common 3D tools and so tests can build synthetic BVH files with known answers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

AXES = {"X": 0, "Y": 1, "Z": 2}
ROT_CHANNELS = ("Xrotation", "Yrotation", "Zrotation")
POS_CHANNELS = ("Xposition", "Yposition", "Zposition")
ORDERS = ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX")


# =========================================================================================================
# Euler angles
# =========================================================================================================
def axis_rotation(axis: int, angle: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation about coordinate axis 0/1/2 by angle (radians)."""
    a = np.asarray(angle, float)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    if axis == 0:
        rows = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == 1:
        rows = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        rows = [[c, -s, z], [s, c, z], [z, z, o]]
    return np.stack([np.stack(r, axis=-1) for r in rows], axis=-2)


def euler_to_matrix(angles_deg: np.ndarray, order: str) -> np.ndarray:
    """Euler angles (..., n) in degrees for the axes in `order` (e.g. "ZXY") -> R = R_o0 R_o1 R_o2 ..."""
    ang = np.radians(np.asarray(angles_deg, float))
    R = np.broadcast_to(np.eye(3), ang.shape[:-1] + (3, 3)).copy()
    for k, ax in enumerate(order.upper()):
        R = R @ axis_rotation(AXES[ax], ang[..., k])
    return R


def matrix_to_euler(R: np.ndarray, order: str) -> np.ndarray:
    """Inverse of euler_to_matrix for the six Tait-Bryan orders: (..., 3, 3) -> (..., 3) degrees.

    For R = R_i(a) R_j(b) R_k(c) with distinct axes i, j, k and s = +1 for a cyclic (even) order, -1
    otherwise: R[i, k] = s sin b, a = atan2(-s R[j, k], R[k, k]), c = atan2(-s R[i, j], R[i, i]).
    At gimbal lock (|b| = 90 deg) c is set to 0 and a absorbs the remaining rotation.
    """
    order = order.upper()
    if len(order) != 3 or len(set(order)) != 3 or set(order) - set("XYZ"):
        raise ValueError(f"unsupported Euler order {order!r} (use one of {ORDERS})")
    i, j, k = (AXES[c] for c in order)
    s = 1.0 if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1.0
    R = np.asarray(R, float)
    sb = np.clip(s * R[..., i, k], -1.0, 1.0)
    b = np.arcsin(sb)
    a = np.arctan2(-s * R[..., j, k], R[..., k, k])
    c = np.arctan2(-s * R[..., i, j], R[..., i, i])
    lock = np.abs(sb) > 1.0 - 1e-9
    if np.any(lock):
        # R_i(a) R_j(+-90) R_k(0): solve a from the rotation that remains in the j-k plane
        a_lock = np.arctan2(s * R[..., k, j], R[..., j, j])
        a = np.where(lock, a_lock, a)
        c = np.where(lock, 0.0, c)
    return np.degrees(np.stack([a, b, c], axis=-1))


# =========================================================================================================
# Data model
# =========================================================================================================
@dataclass
class BvhJoint:
    name: str
    parent: int                          # index of the parent joint, -1 for the root
    offset: np.ndarray                   # (3,) in the parent frame
    channels: list[str]
    end_site: np.ndarray | None = None   # offset of an End Site child, if any
    children: list[int] = field(default_factory=list)

    @property
    def rot_order(self) -> str:
        return "".join(c[0].upper() for c in self.channels if c in ROT_CHANNELS)


@dataclass
class Bvh:
    joints: list[BvhJoint]
    frame_time: float
    frames: np.ndarray                   # (F, total channels)

    def __post_init__(self):
        self.frames = np.atleast_2d(np.asarray(self.frames, float))
        if self.frames.size == 0:
            self.frames = np.zeros((0, self.n_channels))
        self._index_channels()

    def _index_channels(self) -> None:
        self.channel_start = []
        k = 0
        for j in self.joints:
            self.channel_start.append(k)
            k += len(j.channels)
        if self.frames.shape[1] != k:
            raise ValueError(f"BVH frames have {self.frames.shape[1]} values per frame, the hierarchy {k}")

    @property
    def n_channels(self) -> int:
        return sum(len(j.channels) for j in self.joints)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time if self.frame_time > 0 else 0.0

    @property
    def names(self) -> list[str]:
        return [j.name for j in self.joints]

    def index(self, name: str) -> int:
        return self.names.index(name)

    def channel_values(self, joint: int, channel: str) -> np.ndarray | None:
        j = self.joints[joint]
        if channel not in j.channels:
            return None
        return self.frames[:, self.channel_start[joint] + j.channels.index(channel)]

    # ---- kinematics -----------------------------------------------------------------------------------
    def local_transforms(self, frames: np.ndarray | None = None):
        """-> translations (F, J, 3) and rotations (F, J, 3, 3) of every joint in its parent frame."""
        fr = self.frames if frames is None else np.atleast_2d(np.asarray(frames, float))
        n = len(fr)
        trans = np.zeros((n, len(self.joints), 3))
        rots = np.zeros((n, len(self.joints), 3, 3))
        for ji, j in enumerate(self.joints):
            k0 = self.channel_start[ji]
            t = np.tile(j.offset, (n, 1))
            R = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
            for ci, ch in enumerate(j.channels):
                v = fr[:, k0 + ci]
                if ch in POS_CHANNELS:
                    t[:, POS_CHANNELS.index(ch)] = v
                elif ch in ROT_CHANNELS:
                    R = R @ axis_rotation(ROT_CHANNELS.index(ch), np.radians(v))
            trans[:, ji], rots[:, ji] = t, R
        return trans, rots

    def forward_kinematics(self, frames: np.ndarray | None = None, rest: bool = False):
        """World positions (F, J, 3), rotations (F, J, 3, 3) and End Site positions (F, J, 3; NaN where a
        joint has none). rest=True evaluates the rest pose (offsets only, no channels) as a single frame."""
        if rest:
            n = 1
            trans = np.array([[j.offset for j in self.joints]], float)
            rots = np.broadcast_to(np.eye(3), (1, len(self.joints), 3, 3)).copy()
        else:
            trans, rots = self.local_transforms(frames)
            n = len(trans)
        pos = np.zeros((n, len(self.joints), 3))
        rot = np.zeros((n, len(self.joints), 3, 3))
        end = np.full((n, len(self.joints), 3), np.nan)
        for ji, j in enumerate(self.joints):
            if j.parent < 0:
                pos[:, ji], rot[:, ji] = trans[:, ji], rots[:, ji]
            else:
                Rp = rot[:, j.parent]
                pos[:, ji] = pos[:, j.parent] + np.einsum("nij,nj->ni", Rp, trans[:, ji])
                rot[:, ji] = Rp @ rots[:, ji]
            if j.end_site is not None:
                end[:, ji] = pos[:, ji] + np.einsum("nij,j->ni", rot[:, ji], j.end_site)
        return pos, rot, end

    # ---- writing ----------------------------------------------------------------------------------------
    def to_text(self, precision: int = 6) -> str:
        lines = ["HIERARCHY"]
        fmt = f"{{:.{precision}f}}"

        def num(v: float) -> str:
            s = fmt.format(float(v))
            if s.strip("-0.") == "":
                return "0"
            return s.rstrip("0").rstrip(".") if "." in s else s

        def write(ji: int, depth: int) -> None:
            j = self.joints[ji]
            ind = "\t" * depth
            lines.append(f"{ind}{'ROOT' if j.parent < 0 else 'JOINT'} {j.name}")
            lines.append(f"{ind}{{")
            lines.append(f"{ind}\tOFFSET {' '.join(num(v) for v in j.offset)}")
            lines.append(f"{ind}\tCHANNELS {len(j.channels)}" + "".join(f" {c}" for c in j.channels))
            for c in j.children:
                write(c, depth + 1)
            if j.end_site is not None:
                lines.append(f"{ind}\tEnd Site")
                lines.append(f"{ind}\t{{")
                lines.append(f"{ind}\t\tOFFSET {' '.join(num(v) for v in j.end_site)}")
                lines.append(f"{ind}\t}}")
            lines.append(f"{ind}}}")

        for ji, j in enumerate(self.joints):
            if j.parent < 0:
                write(ji, 0)
        lines += ["MOTION", f"Frames: {self.n_frames}", f"Frame Time: {self.frame_time:.8g}"]
        lines += [" ".join(num(v) for v in row) for row in self.frames]
        return "\n".join(lines) + "\n"

    def save(self, path: str | Path, precision: int = 6) -> Path:
        path = Path(path)
        path.write_text(self.to_text(precision), encoding="utf-8")
        return path


# =========================================================================================================
# Parsing
# =========================================================================================================
class BvhError(ValueError):
    pass


def parse_bvh(text: str) -> Bvh:
    """Parse BVH text. Raises BvhError with the offending line for malformed files."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i, n = 0, len(lines)

    def next_line() -> tuple[int, list[str]]:
        nonlocal i
        while i < n:
            toks = lines[i].split()
            i += 1
            if toks:
                return i, toks
        raise BvhError("unexpected end of file in HIERARCHY")

    ln, toks = next_line()
    if toks[0].upper() != "HIERARCHY":
        raise BvhError(f"line {ln}: expected HIERARCHY, got {' '.join(toks)!r}")
    joints: list[BvhJoint] = []
    stack: list[int] = []          # joint indices whose blocks are open (-2 marks an End Site block)
    pending: tuple[str, int] | None = None
    while True:
        ln, toks = next_line()
        key = toks[0].upper()
        if key in ("ROOT", "JOINT"):
            if len(toks) < 2:
                raise BvhError(f"line {ln}: {toks[0]} without a name")
            name = " ".join(toks[1:])
            parent = next((s for s in reversed(stack) if s >= 0), -1)
            if key == "ROOT" and stack:
                raise BvhError(f"line {ln}: ROOT inside another joint")
            if key == "JOINT" and parent < 0:
                raise BvhError(f"line {ln}: JOINT outside ROOT")
            pending = ("joint", len(joints))
            joints.append(BvhJoint(name, parent, np.zeros(3), []))
            if parent >= 0:
                joints[parent].children.append(len(joints) - 1)
        elif key == "END":
            pending = ("end", next((s for s in reversed(stack) if s >= 0), -1))
            if pending[1] < 0:
                raise BvhError(f"line {ln}: End Site outside a joint")
        elif key == "{":
            if pending is None:
                raise BvhError(f"line {ln}: '{{' without ROOT/JOINT/End Site")
            stack.append(pending[1] if pending[0] == "joint" else -2 - pending[1])
            pending = None
        elif key == "}":
            if not stack:
                raise BvhError(f"line {ln}: unmatched '}}'")
            stack.pop()
            if not stack:
                # the root block closed: either another ROOT follows or MOTION
                save = i
                ln2, t2 = next_line()
                if t2[0].upper() == "MOTION":
                    break
                i = save
        elif key == "OFFSET":
            if len(toks) < 4:
                raise BvhError(f"line {ln}: OFFSET needs three numbers")
            try:
                off = np.array([float(v) for v in toks[1:4]])
            except ValueError as e:
                raise BvhError(f"line {ln}: bad OFFSET {toks[1:4]}") from e
            if not stack:
                raise BvhError(f"line {ln}: OFFSET outside a block")
            top = stack[-1]
            if top >= 0:
                joints[top].offset = off
            else:
                joints[-2 - top].end_site = off
        elif key == "CHANNELS":
            if not stack or stack[-1] < 0:
                raise BvhError(f"line {ln}: CHANNELS outside a joint block")
            try:
                cnt = int(toks[1])
            except (IndexError, ValueError) as e:
                raise BvhError(f"line {ln}: bad CHANNELS line") from e
            chans = toks[2:2 + cnt]
            if len(chans) != cnt:
                raise BvhError(f"line {ln}: CHANNELS {cnt} lists {len(chans)} names")
            norm = []
            for c in chans:
                cc = c[0].upper() + c[1:].lower()
                if cc not in ROT_CHANNELS + POS_CHANNELS:
                    raise BvhError(f"line {ln}: unknown channel {c!r}")
                norm.append(cc)
            joints[stack[-1]].channels = norm
        elif key == "MOTION":
            break
        else:
            raise BvhError(f"line {ln}: unexpected {' '.join(toks)!r}")
    if not joints:
        raise BvhError("no joints in HIERARCHY")
    # ---- MOTION section
    ln, toks = next_line()
    if toks[0].lower().rstrip(":") != "frames":
        raise BvhError(f"line {ln}: expected 'Frames:'")
    try:
        nf = int(toks[-1])
    except ValueError as e:
        raise BvhError(f"line {ln}: bad frame count") from e
    ln, toks = next_line()
    if " ".join(toks[:2]).lower().rstrip(":") != "frame time":
        raise BvhError(f"line {ln}: expected 'Frame Time:'")
    frame_time = float(toks[-1])
    vals = " ".join(lines[i:]).split()
    nch = sum(len(j.channels) for j in joints)
    if nch == 0:
        raise BvhError("the hierarchy has no channels")
    try:
        data = np.array([float(v) for v in vals[:nf * nch]], float)
    except ValueError as e:
        raise BvhError("non-numeric value in the MOTION section") from e
    if data.size < nf * nch:
        raise BvhError(f"MOTION has {data.size // nch} complete frames, header says {nf}")
    return Bvh(joints, frame_time, data.reshape(nf, nch))


def load_bvh(path: str | Path) -> Bvh:
    return parse_bvh(Path(path).read_text(encoding="utf-8", errors="replace"))


# =========================================================================================================
# Coordinate frames
# =========================================================================================================
# Robot world: Z up, X forward, Y left. Y-up BVH (most mocap): Y up, Z forward, X left (= up x forward).
# bvh = P_YUP @ robot, i.e. bvh_x = robot_y, bvh_y = robot_z, bvh_z = robot_x (a cyclic permutation).
P_YUP = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])


def up_conversion(up: str) -> np.ndarray:
    """Rotation P with bvh_vector = P @ robot_vector for an up axis "y", "z", "x" or "-y", "-z", "-x"."""
    u = up.lower().strip()
    sign = -1.0 if u.startswith("-") else 1.0
    ax = u.lstrip("+-")
    if ax == "z":
        P = np.eye(3)
    elif ax == "y":
        P = P_YUP.copy()
    elif ax == "x":
        P = P_YUP.T.copy()  # robot z -> bvh x: bvh = (robot_z, robot_x, robot_y)
    else:
        raise ValueError(f"up axis must be x, y or z (optionally signed), got {up!r}")
    if sign < 0:   # flip up: rotate 180 deg about the robot's forward axis first
        P = P @ np.diag([1.0, -1.0, -1.0])
    return P


# =========================================================================================================
# Export robot motions
# =========================================================================================================
def motion_to_bvh(motion, model: mujoco.MjModel, scale: float = 100.0, up: str = "y", order: str = "ZXY",
                  head: bool = True, frame_time: float | None = None) -> Bvh:
    """Write a robot `Motion` (simulatics.imitate.motion) as a BVH skeleton of the robot's bodies.

    Every body of the robot's tree becomes a joint named after it whose frame is the body frame (offsets =
    MuJoCo body positions x `scale`, e.g. 100 for centimetres), with three rotation channels in `order`
    holding the body's rotation relative to its parent; the root also has position channels. With
    head=True a "head" joint is added at the head geom. Leaves get an End Site at their first geom.
    up="y" writes the usual Y-up / Z-forward convention, up="z" the robot's own axes.
    """
    from .motion import free_root
    P = up_conversion(up)
    root, _, _ = free_root(model)
    bodies = [b for b in range(model.nbody) if b == root or (model.body_rootid[b] == root and b != 0)]
    bodies = [b for b in bodies if b == root or _has_ancestor(model, b, root)]
    index = {b: k for k, b in enumerate(bodies)}
    rot = [f"{a}rotation" for a in order.upper()]
    joints: list[BvhJoint] = []
    for b in bodies:
        parent = -1 if b == root else index[int(model.body_parentid[b])]
        off = np.zeros(3) if b == root else P @ model.body_pos[b] * scale
        ch = (list(POS_CHANNELS) + rot) if b == root else list(rot)
        joints.append(BvhJoint(model.body(b).name, parent, off, ch))
        if parent >= 0:
            joints[parent].children.append(len(joints) - 1)
    head_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "head")
    head_parent = index.get(int(model.geom_bodyid[head_geom]), -1) if (head and head_geom >= 0) else -1
    if head_parent >= 0:
        r = float(model.geom_size[head_geom, 0])
        joints.append(BvhJoint("head", head_parent, P @ model.geom_pos[head_geom] * scale, list(rot),
                               end_site=P @ np.array([0.0, 0.0, r]) * scale))
        joints[head_parent].children.append(len(joints) - 1)
    for k, j in enumerate(joints):
        if not j.children and j.end_site is None and k < len(bodies):
            b = bodies[k]
            g = int(model.body_geomadr[b])
            tip = model.geom_pos[g] if g >= 0 else np.array([0.0, 0.0, -0.05])
            j.end_site = P @ tip * scale
    # frames
    qs = motion.qpos(model)
    d = mujoco.MjData(model)
    nch = sum(len(j.channels) for j in joints)
    frames = np.zeros((len(qs), nch))
    for t, q in enumerate(qs):
        d.qpos[:] = q
        mujoco.mj_kinematics(model, d)
        k = 0
        for ji, b in enumerate(bodies):
            R = d.xmat[b].reshape(3, 3)
            if b == root:
                frames[t, 0:3] = P @ d.xpos[b] * scale
                local = R
                k = 3
            else:
                local = d.xmat[int(model.body_parentid[b])].reshape(3, 3).T @ R
            frames[t, k:k + 3] = matrix_to_euler(P @ local @ P.T, order)
            k += 3
        # the head joint (if any) keeps zero rotation
    ft = frame_time if frame_time is not None else 1.0 / motion.fps
    return Bvh(joints, ft, frames)


def _has_ancestor(model: mujoco.MjModel, b: int, anc: int) -> bool:
    while b > 0:
        if b == anc:
            return True
        b = int(model.body_parentid[b])
    return False
