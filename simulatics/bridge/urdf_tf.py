"""TF for every URDF link from joint positions: what robot_state_publisher does, without ROS.

The bridge exports the robot as URDF (design/urdf.py, joint names = MuJoCo joint names) and uses
`UrdfTree` to publish parent -> child transforms for every URDF joint, so RViz's RobotModel display
works with nothing but the bridge running. Fixed joints go to /tf_static, moving ones to /tf.
URDF semantics: child frame = parent frame * origin(xyz, rpy) * motion(axis, q), where rpy is a
fixed-axis X-Y-Z rotation, revolute / continuous joints rotate about `axis` and prismatic ones
translate along it.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .msgs import quat_mul, quat_rotate


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Fixed-axis roll (x), pitch (y), yaw (z), i.e. R = Rz(yaw) Ry(pitch) Rx(roll); xyzw quaternion."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def axis_angle_quat(axis, angle: float) -> tuple[float, float, float, float]:
    a = np.asarray(axis, float)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    a = a / n
    s = math.sin(angle / 2)
    return float(a[0] * s), float(a[1] * s), float(a[2] * s), math.cos(angle / 2)


def _vec(text: str | None, n: int = 3) -> np.ndarray:
    if not text:
        return np.zeros(n)
    v = np.array([float(x) for x in text.split()], float)
    return v if len(v) == n else np.zeros(n)


@dataclass
class UrdfJoint:
    name: str
    type: str                       # revolute | continuous | prismatic | fixed | floating | planar
    parent: str
    child: str
    xyz: np.ndarray
    quat: tuple[float, float, float, float]   # origin rotation, xyzw
    axis: np.ndarray

    @property
    def moving(self) -> bool:
        return self.type in ("revolute", "continuous", "prismatic")

    def transform(self, q: float = 0.0) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """(translation, rotation xyzw) of the child link in the parent link frame at joint position q."""
        if self.type in ("revolute", "continuous"):
            return self.xyz.copy(), quat_mul(self.quat, axis_angle_quat(self.axis, q))
        if self.type == "prismatic":
            return self.xyz + quat_rotate(self.quat, self.axis * q), self.quat
        return self.xyz.copy(), self.quat


class UrdfTree:
    def __init__(self, urdf_xml: str):
        root = ET.fromstring(urdf_xml)
        self.name = root.get("name", "")
        self.links = [el.get("name") for el in root.findall("link")]
        self.joints: list[UrdfJoint] = []
        for el in root.findall("joint"):
            o = el.find("origin")
            rpy = _vec(o.get("rpy") if o is not None else None)
            ax = el.find("axis")
            self.joints.append(UrdfJoint(
                name=el.get("name"), type=el.get("type", "fixed"),
                parent=el.find("parent").get("link"), child=el.find("child").get("link"),
                xyz=_vec(o.get("xyz") if o is not None else None), quat=rpy_to_quat(*rpy),
                axis=_vec(ax.get("xyz")) if ax is not None else np.array([1.0, 0.0, 0.0])))
        children = {j.child for j in self.joints}
        roots = [ln for ln in self.links if ln not in children]
        self.root = roots[0] if roots else (self.links[0] if self.links else "")
        self.moving = [j for j in self.joints if j.moving]
        self.fixed = [j for j in self.joints if not j.moving]

    def moving_transforms(self, positions: dict[str, float]):
        """[(parent, child, translation, rotation xyzw)] for the moving joints; missing positions read 0."""
        return [(j.parent, j.child, *j.transform(float(positions.get(j.name, 0.0)))) for j in self.moving]

    def fixed_transforms(self):
        return [(j.parent, j.child, *j.transform()) for j in self.fixed]
