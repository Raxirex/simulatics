"""Leg geometry read from a compiled MuJoCo model, and closed-form 6-DOF leg inverse kinematics.

The built-in humanoid (models/humanoid.py) has, per leg,

    pelvis --hip_x(x) -> hip_z(z) -> hip_y(y)--> thigh --knee(y)--> shin --ankle_y(y) -> ankle_x(x)--> foot

with the three hip axes meeting in one point and the two ankle axes meeting in
another, so the leg is a "spherical hip + knee + universal ankle" chain. That
structure has a closed-form inverse (Kajita, Hirukawa, Harada, Yokoi,
*Introduction to Humanoid Robotics*, Springer 2014, section 2.5): the knee
follows from the hip-ankle distance (law of cosines), the ankle angles from the
hip position seen from the foot frame, and the hip angles from the remaining
rotation. Only the last step depends on the hip axis order; here it is the
Euler decomposition R = Rx(a) Rz(b) Ry(c).

Every length (pelvis offsets, thigh, shin, ankle height, sole size) comes from
the compiled model, so the same code serves every body the parametric
generator can make.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

SIDES = ("left", "right")
LEG_JOINTS = ("hip_x", "hip_z", "hip_y", "knee", "ankle_y", "ankle_x")
ABDOMEN = ("abdomen_y", "abdomen_x")
# expected joint axes (in the parent body frame) for the closed-form IK to be valid
_AXES = {"hip_x": (1, 0, 0), "hip_z": (0, 0, 1), "hip_y": (0, 1, 0), "knee": (0, 1, 0),
         "ankle_y": (0, 1, 0), "ankle_x": (1, 0, 0), "abdomen_y": (0, 1, 0), "abdomen_x": (1, 0, 0)}


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def yaw_of(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _missing(model: mujoco.MjModel) -> list[str]:
    """Names the walker needs that the model does not have (empty list = supported)."""
    need = []
    for kind, names in (
        (mujoco.mjtObj.mjOBJ_BODY,
         ["torso", "pelvis"] + [f"{s}_{b}" for s in SIDES for b in ("thigh", "shin", "foot")]),
        (mujoco.mjtObj.mjOBJ_JOINT, list(ABDOMEN) + [f"{s}_{j}" for s in SIDES for j in LEG_JOINTS]),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, list(ABDOMEN) + [f"{s}_{j}" for s in SIDES for j in LEG_JOINTS]),
        (mujoco.mjtObj.mjOBJ_GEOM, [f"{s}_foot" for s in SIDES]),
    ):
        need += [n for n in names if mujoco.mj_name2id(model, kind, n) < 0]
    return need


def model_supported(model: mujoco.MjModel) -> bool:
    """True when the model has the built-in humanoid's leg structure (names, axes, free root, box feet)."""
    if model.njnt == 0 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or _missing(model):
        return False
    for name, axis in _AXES.items():
        for full in ([name] if name.startswith("abdomen") else [f"{s}_{name}" for s in SIDES]):
            j = model.joint(full).id
            if (model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or np.linalg.norm(model.jnt_pos[j]) > 1e-6
                    or not np.allclose(model.jnt_axis[j], axis, atol=1e-6)):
                return False
            # the actuator of the same name must be a position servo on that joint (ctrl = target angle)
            a = model.actuator(full).id
            kp = model.actuator_gainprm[a, 0]
            if (model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT or model.actuator_trnid[a, 0] != j
                    or model.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_AFFINE or kp <= 0
                    or abs(model.actuator_biasprm[a, 1] + kp) > 1e-6 * max(1.0, kp)
                    or abs(model.actuator_gear[a, 0] - 1.0) > 1e-9):
                return False
    for s in SIDES:
        if model.geom_type[model.geom(f"{s}_foot").id] != mujoco.mjtGeom.mjGEOM_BOX:
            return False
        for b in ("thigh", "shin", "foot"):
            if not np.allclose(model.body_quat[model.body(f"{s}_{b}").id], (1, 0, 0, 0), atol=1e-6):
                return False
    if model.body_parentid[model.body("pelvis").id] != model.body("torso").id:
        return False
    return True


@dataclass
class LegGeometry:
    """Lengths of one leg, all from the compiled model (pelvis frame / foot frame)."""
    hip: np.ndarray          # hip joint centre in the pelvis frame
    thigh: float             # hip -> knee
    shin: float              # knee -> ankle
    sole: np.ndarray         # sole centre (on the ground plane) in the foot frame
    half_length: float       # sole half length (x) and half width (y)
    half_width: float


@dataclass
class RobotGeometry:
    pelvis: np.ndarray       # pelvis origin in the torso (root) frame
    legs: dict[str, LegGeometry]
    mass: float

    @property
    def leg_length(self) -> float:
        g = self.legs["left"]
        return g.thigh + g.shin

    @property
    def ankle_height(self) -> float:
        return float(-self.legs["left"].sole[2])

    @property
    def hip_width(self) -> float:
        return float(self.legs["left"].hip[1] - self.legs["right"].hip[1])

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "RobotGeometry":
        legs = {}
        for s in SIDES:
            thigh = model.body(f"{s}_thigh")
            shin = model.body(f"{s}_shin")
            foot = model.body(f"{s}_foot")
            g = model.geom(f"{s}_foot").id
            gpos, gsize = model.geom_pos[g], model.geom_size[g]
            legs[s] = LegGeometry(
                hip=model.body_pos[thigh.id].copy(),
                thigh=float(np.linalg.norm(model.body_pos[shin.id])),
                shin=float(np.linalg.norm(model.body_pos[foot.id])),
                sole=np.array([gpos[0], gpos[1], gpos[2] - gsize[2]]),
                half_length=float(gsize[0]), half_width=float(gsize[1]))
        return cls(pelvis=model.body_pos[model.body("pelvis").id].copy(), legs=legs,
                   mass=float(model.body_subtreemass[model.body("torso").id]))


def leg_ik(geo: LegGeometry, p_pelvis: np.ndarray, R_pelvis: np.ndarray, sole_pos: np.ndarray,
           R_foot: np.ndarray) -> tuple[np.ndarray, float]:
    """Joint angles [hip_x, hip_z, hip_y, knee, ankle_y, ankle_x] that put the sole centre at sole_pos with
    orientation R_foot, for a pelvis at (p_pelvis, R_pelvis). All poses in the world frame.

    Returns (q, reach) where reach = hip-ankle distance / leg length; reach > 1 means the target was out of
    range and the leg is fully stretched towards it.

    (Scalar maths on purpose: it runs ~10 times per control tick and numpy's per-call overhead on 3-vectors
    would dominate.)
    """
    A, B = geo.thigh, geo.shin
    ankle = sole_pos - R_foot @ geo.sole
    hip = p_pelvis + R_pelvis @ geo.hip
    r = R_foot.T @ (hip - ankle)          # hip seen from the foot frame
    rx, ry, rz = float(r[0]), float(r[1]), float(r[2])
    C = math.sqrt(rx * rx + ry * ry + rz * rz)
    reach = C / (A + B)
    C = min(max(C, abs(A - B) + 1e-6), (A + B) * 0.9999)
    cos_k = (C * C - A * A - B * B) / (2 * A * B)
    knee = math.acos(min(1.0, max(-1.0, cos_k)))
    ankle_x = math.atan2(ry, rz)
    ryz = math.hypot(ry, rz)
    vx, vz = -A * math.sin(knee), A * math.cos(knee) + B   # hip seen from the ankle in the shin frame
    ankle_y = math.atan2(vx, vz) - math.atan2(rx, ryz)
    # M = R_pelvis^T R_foot Rx(-ankle_x) Ry(-(ankle_y + knee)) = Rx(hip_x) Rz(hip_z) Ry(hip_y)
    N = R_pelvis.T @ R_foot
    ca, sa = math.cos(ankle_x), math.sin(ankle_x)
    b = ankle_y + knee
    cb, sb = math.cos(b), math.sin(b)
    n00, n01, n02 = float(N[0, 0]), float(N[0, 1]), float(N[0, 2])
    m00 = n00 * cb + n01 * sa * sb + n02 * ca * sb
    m01 = n01 * ca - n02 * sa
    m02 = -n00 * sb + n01 * sa * cb + n02 * ca * cb
    m11 = float(N[1, 1]) * ca - float(N[1, 2]) * sa
    m21 = float(N[2, 1]) * ca - float(N[2, 2]) * sa
    hip_z = -math.asin(min(1.0, max(-1.0, m01)))
    hip_x = math.atan2(m21, m11)
    hip_y = math.atan2(m02, m00)
    return np.array([hip_x, hip_z, hip_y, knee, ankle_y, ankle_x]), reach
