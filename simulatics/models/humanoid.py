"""Parametric humanoid: geometry + motor parameters -> MJCF (MuJoCo XML).

Every number that defines the robot lives in `HumanoidParams`, so the GUI can
edit it with sliders and the evolutionary trainer can mutate it as a genome.
The kinematic tree keeps the same joints/actuators for every parameter value,
which lets MuJoCo's threaded rollout run a different body in every thread.

Conventions: Z up, robot faces +X, angles in degrees, SI units.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, fields

# name -> (min, max, default, label, group)
PARAM_SPEC: dict[str, tuple[float, float, float, str, str]] = {
    # geometry
    "torso_length": (0.20, 0.60, 0.36, "Torso length (m)", "Geometry"),
    "torso_radius": (0.06, 0.18, 0.10, "Torso radius (m)", "Geometry"),
    "shoulder_width": (0.20, 0.60, 0.36, "Shoulder width (m)", "Geometry"),
    "pelvis_width": (0.10, 0.40, 0.20, "Hip width (m)", "Geometry"),
    "thigh_length": (0.20, 0.60, 0.40, "Thigh length (m)", "Geometry"),
    "shin_length": (0.20, 0.60, 0.40, "Shin length (m)", "Geometry"),
    "leg_radius": (0.025, 0.09, 0.05, "Leg radius (m)", "Geometry"),
    "foot_length": (0.10, 0.36, 0.22, "Foot length (m)", "Geometry"),
    "foot_width": (0.04, 0.16, 0.09, "Foot width (m)", "Geometry"),
    "upper_arm_length": (0.12, 0.45, 0.28, "Upper arm length (m)", "Geometry"),
    "forearm_length": (0.12, 0.45, 0.26, "Forearm length (m)", "Geometry"),
    "arm_radius": (0.02, 0.07, 0.035, "Arm radius (m)", "Geometry"),
    "head_radius": (0.05, 0.14, 0.09, "Head radius (m)", "Geometry"),
    "density": (200.0, 2000.0, 900.0, "Density (kg/m^3)", "Geometry"),
    # motors: position servos (PD) with a torque limit
    "hip_kp": (20.0, 800.0, 300.0, "Hip stiffness kp (Nm/rad)", "Motors"),
    "hip_torque": (20.0, 400.0, 150.0, "Hip max torque (Nm)", "Motors"),
    "knee_kp": (20.0, 800.0, 300.0, "Knee stiffness kp (Nm/rad)", "Motors"),
    "knee_torque": (20.0, 400.0, 180.0, "Knee max torque (Nm)", "Motors"),
    "ankle_kp": (10.0, 1200.0, 500.0, "Ankle stiffness kp (Nm/rad)", "Motors"),
    "ankle_torque": (5.0, 300.0, 120.0, "Ankle max torque (Nm)", "Motors"),
    "abdomen_kp": (20.0, 800.0, 300.0, "Waist stiffness kp (Nm/rad)", "Motors"),
    "abdomen_torque": (10.0, 300.0, 100.0, "Waist max torque (Nm)", "Motors"),
    "arm_kp": (5.0, 300.0, 60.0, "Arm stiffness kp (Nm/rad)", "Motors"),
    "arm_torque": (5.0, 150.0, 40.0, "Arm max torque (Nm)", "Motors"),
    "kv_ratio": (0.005, 0.2, 0.06, "Damping ratio kv/kp (s)", "Motors"),
    "joint_damping": (0.0, 10.0, 1.0, "Passive joint damping", "Motors"),
    "armature": (0.0, 0.2, 0.02, "Rotor inertia (armature)", "Motors"),
    # physics
    "timestep": (0.001, 0.01, 0.005, "Physics timestep (s)", "Physics"),
    "friction": (0.1, 2.0, 1.0, "Foot/floor friction", "Physics"),
}

# Parameters the evolutionary trainer may mutate when "evolve body" is on.
MORPH_GENES = [
    "torso_length", "pelvis_width", "thigh_length", "shin_length", "foot_length",
    "foot_width", "leg_radius", "hip_kp", "knee_kp", "ankle_kp", "kv_ratio",
]


@dataclass
class HumanoidParams:
    torso_length: float = 0.36
    torso_radius: float = 0.10
    shoulder_width: float = 0.36
    pelvis_width: float = 0.20
    thigh_length: float = 0.40
    shin_length: float = 0.40
    leg_radius: float = 0.05
    foot_length: float = 0.22
    foot_width: float = 0.09
    upper_arm_length: float = 0.28
    forearm_length: float = 0.26
    arm_radius: float = 0.035
    head_radius: float = 0.09
    density: float = 900.0
    hip_kp: float = 300.0
    hip_torque: float = 150.0
    knee_kp: float = 300.0
    knee_torque: float = 180.0
    ankle_kp: float = 500.0
    ankle_torque: float = 120.0
    abdomen_kp: float = 300.0
    abdomen_torque: float = 100.0
    arm_kp: float = 60.0
    arm_torque: float = 40.0
    kv_ratio: float = 0.06
    joint_damping: float = 1.0
    armature: float = 0.02
    timestep: float = 0.005
    friction: float = 1.0
    arms: bool = True
    # wrist (pitch + roll) and a two-finger parallel gripper on each forearm; needs arms
    hands: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HumanoidParams":
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        if "arms" in kw and "hands" not in kw:
            # a full parameter dict saved before hands existed (e.g. a trained policy's
            # meta) describes a hand-less robot: keep its actuator count unchanged
            kw["hands"] = False
        return cls(**kw).clipped()

    def clipped(self) -> "HumanoidParams":
        for k, (lo, hi, *_rest) in PARAM_SPEC.items():
            setattr(self, k, float(min(hi, max(lo, float(getattr(self, k))))))
        self.arms = bool(self.arms)
        self.hands = bool(self.hands)
        return self

    # ---- derived geometry -------------------------------------------------
    @property
    def ankle_height(self) -> float:
        return 0.06

    @property
    def pelvis_drop(self) -> float:
        return 0.09

    @property
    def waist_offset(self) -> float:
        return self.torso_length / 2 + 0.05

    @property
    def stand_height(self) -> float:
        """Height of the torso (root body) origin when standing straight."""
        return (self.ankle_height + self.shin_length + self.thigh_length
                + self.pelvis_drop + self.waist_offset + 0.002)


def _f(*v: float) -> str:
    return " ".join(f"{x:.5g}" for x in v)


def _leg(p: HumanoidParams, side: str, sign: float) -> str:
    """sign = +1 for left (+Y), -1 for right."""
    r = p.leg_radius
    # roll/yaw ranges are mirrored so "positive = outward" on both sides
    roll = (-15, 35) if sign > 0 else (-35, 15)
    yaw = (-30, 30)
    fl, fw = p.foot_length, p.foot_width
    return f"""
        <body name="{side}_thigh" pos="{_f(0, sign * p.pelvis_width / 2, -p.pelvis_drop)}">
          <joint name="{side}_hip_x" axis="1 0 0" range="{_f(*roll)}"/>
          <joint name="{side}_hip_z" axis="0 0 1" range="{_f(*yaw)}"/>
          <joint name="{side}_hip_y" axis="0 1 0" range="-110 30"/>
          <geom name="{side}_thigh" type="capsule" fromto="0 0 0 0 0 {_f(-p.thigh_length)}" size="{_f(r)}"/>
          <body name="{side}_shin" pos="0 0 {_f(-p.thigh_length)}">
            <joint name="{side}_knee" axis="0 1 0" range="0 140"/>
            <geom name="{side}_shin" type="capsule" fromto="0 0 0 0 0 {_f(-p.shin_length)}" size="{_f(r * 0.85)}"/>
            <body name="{side}_foot" pos="0 0 {_f(-p.shin_length)}">
              <joint name="{side}_ankle_y" axis="0 1 0" range="-45 45"/>
              <joint name="{side}_ankle_x" axis="1 0 0" range="-20 20"/>
              <geom name="{side}_foot" type="box" pos="{_f(fl * 0.2, 0, -0.035)}" size="{_f(fl / 2, fw / 2, 0.025)}"
                    friction="{_f(p.friction, 0.02, 0.001)}" rgba="0.2 0.22 0.26 1"/>
              <site name="{side}_foot_site" pos="{_f(fl * 0.2, 0, -0.035)}" size="{_f(fl / 2, fw / 2, 0.03)}" type="box" rgba="0 0 0 0"/>
            </body>
          </body>
        </body>"""


# ---- hands ------------------------------------------------------------------
# A wrist (pitch "wrist_y", then roll "wrist_x") and a parallel two-finger gripper.
# Sizes are fixed (sized for 4-6 cm objects) and the hand is light (~0.2 kg) so it
# does not change how the robot balances. In the hanging pose the fingers point
# down (hand -z) and close along the hand's y axis.
HAND_PALM = 0.05          # wrist axis -> finger base (m)
HAND_FINGER = 0.05        # finger length (m)
HAND_GRASP = 0.035        # finger base -> grasp point (site "<side>_hand"), 1.5 cm above the tips
GRIPPER_OPEN = 0.045      # finger travel (m); the pads' inner faces are 2 * (0.002 + q) apart
WRIST_Y_RANGE = (-120, 120)
WRIST_X_RANGE = (-60, 60)
# Grasp contacts (finger pads and props): near-hard impedance and the stiffest time
# constant MuJoCo allows at dt = 5 ms. MuJoCo's soft contacts let a pinched object
# creep under load at a rate ~ (1 - d) / d / b; with the default solimp/solref a 70 g
# cube slid 9.5 mm in 3 s, with these 0.9 mm (measured, tests/test_arms.py).
GRASP_CONTACT = 'solimp="0.99 0.999 0.001" solref="0.01 1"'


def _hand(p: HumanoidParams, side: str) -> str:
    """Wrist + gripper, attached at the end of the forearm.

    One actuator drives the finger "<side>_gripper" (q = opening of one finger, 0 =
    closed); the mirror finger is slaved to it with a joint equality, as in the
    Franka Panda hand of MuJoCo Menagerie. Finger pads use condim 4 (sliding +
    torsional friction) so a pinched object does not spin out of the grip.
    """
    pad = (f'type="box" size="0.012 0.005 {_f(HAND_FINGER / 2)}" mass="0.025" condim="4" '
           f'friction="1.5 0.02 0.0001" {GRASP_CONTACT} rgba="0.15 0.16 0.18 1"')
    return f"""
            <body name="{side}_hand" pos="0 0 {_f(-p.forearm_length)}">
              <joint name="{side}_wrist_y" axis="0 1 0" range="{_f(*WRIST_Y_RANGE)}"/>
              <joint name="{side}_wrist_x" axis="1 0 0" range="{_f(*WRIST_X_RANGE)}"/>
              <geom name="{side}_palm" type="box" pos="0 0 {_f(-HAND_PALM / 2)}" size="0.018 0.045 {_f(HAND_PALM / 2)}"
                    mass="0.15" rgba="0.35 0.37 0.42 1"/>
              <site name="{side}_hand" pos="0 0 {_f(-HAND_PALM - HAND_GRASP)}" size="0.01" rgba="1 0.3 0.2 0"/>
              <body name="{side}_finger1" pos="0 0 {_f(-HAND_PALM)}">
                <joint name="{side}_gripper" type="slide" axis="0 1 0" range="0 {_f(GRIPPER_OPEN)}"/>
                <geom name="{side}_finger1" pos="0 0.007 {_f(-HAND_FINGER / 2)}" {pad}/>
              </body>
              <body name="{side}_finger2" pos="0 0 {_f(-HAND_PALM)}">
                <joint name="{side}_gripper_mirror" type="slide" axis="0 -1 0" range="0 {_f(GRIPPER_OPEN)}"/>
                <geom name="{side}_finger2" pos="0 -0.007 {_f(-HAND_FINGER / 2)}" {pad}/>
              </body>
            </body>"""


def _arm(p: HumanoidParams, side: str, sign: float) -> str:
    r = p.arm_radius
    roll = (-10, 100) if sign > 0 else (-100, 10)
    hand = _hand(p, side) if p.hands else ""
    return f"""
        <body name="{side}_upper_arm" pos="{_f(0, sign * p.shoulder_width / 2, p.torso_length / 2 - 0.02)}">
          <joint name="{side}_shoulder_y" axis="0 1 0" range="-120 60"/>
          <joint name="{side}_shoulder_x" axis="1 0 0" range="{_f(*roll)}"/>
          <geom name="{side}_upper_arm" type="capsule" fromto="0 0 0 0 0 {_f(-p.upper_arm_length)}" size="{_f(r)}"/>
          <body name="{side}_forearm" pos="0 0 {_f(-p.upper_arm_length)}">
            <joint name="{side}_elbow" axis="0 1 0" range="-130 0"/>
            <geom name="{side}_forearm" type="capsule" fromto="0 0 0 0 0 {_f(-p.forearm_length)}" size="{_f(r * 0.85)}"/>{hand}
          </body>
        </body>"""


def _props_xml(props: list[dict]) -> tuple[str, str]:
    """Free-floating objects the hands can pick up -> (bodies XML, <custom> name table).

    Bodies and geoms are named prop_<i>; the human-readable name ("red cube") is
    stored as custom text "prop_<i>" so controllers can report it from the model alone.
    """
    bodies, names = "", ""
    for i, o in enumerate(props):
        shape = o.get("shape", "box")
        size = o.get("size", 0.025)
        size = [float(size)] if isinstance(size, (int, float)) else [float(s) for s in size]
        if shape == "sphere":
            gsize = _f(size[0])
        elif shape == "cylinder":
            gsize = _f(size[0], size[1] if len(size) > 1 else size[0])
        else:
            shape = "box"
            gsize = _f(*(size * 3 if len(size) == 1 else size[:3]))
        x, y, z = (float(v) for v in o["pos"])
        rgba = _f(*o.get("rgba", (0.86, 0.22, 0.20, 1.0)))
        yaw = float(o.get("yaw", 0.0)) * 57.29578
        bodies += (f'    <body name="prop_{i}" pos="{_f(x, y, z)}" euler="0 0 {_f(yaw)}">\n'
                   f'      <freejoint name="prop_{i}"/>\n'
                   f'      <geom name="prop_{i}" type="{shape}" size="{gsize}" mass="{_f(float(o.get("mass", 0.08)))}" '
                   f'rgba="{rgba}" group="1" contype="1" conaffinity="1" condim="4" friction="1 0.02 0.0001" '
                   f'{GRASP_CONTACT}/>\n'
                   f'    </body>\n')
        label = str(o.get("name", f"prop {i}")).replace("&", "and").replace('"', "'").replace("<", "").replace(">", "")
        names += f'    <text name="prop_{i}" data="{label}"/>\n'
    return bodies, (f"  <custom>\n{names}  </custom>\n" if names else "")


def _servo(joint: str, kp: float, torque: float, kv_ratio: float, lo: float, hi: float) -> str:
    # ctrl is the target joint angle in radians. lo/hi are given in degrees like the joint
    # ranges, but MuJoCo does NOT convert actuator ctrlrange under compiler angle="degree",
    # so convert here (otherwise the range is e.g. 0..140 rad and never clamps anything).
    return (f'    <position name="{joint}" joint="{joint}" kp="{_f(kp)}" kv="{_f(kp * kv_ratio)}" '
            f'ctrlrange="{_f(math.radians(lo), math.radians(hi))}" forcerange="{_f(-torque, torque)}"/>\n')


def build_mjcf(p: HumanoidParams, obstacles: list[dict] | None = None, props: list[dict] | None = None) -> str:
    """Return a complete MJCF document for the humanoid standing at the origin.

    obstacles: list of {"pos": [x, y], "size": [hx, hy, hz], "yaw": rad} objects
    standing on the floor (used for lidar / planning / LLM scenes). Optional keys:
    "shape" ("box" default, or "cylinder": radius hx, half-height hz),
    "rgba" [r, g, b, a] and "name" (e.g. "red box"; the geom is named obj_<i>).
    A box obstacle about 0.9 m tall works as a table (see sim/props.py).
    Terrain (sim/terrain.py) may also give "z" (centre height, default hz: resting on the floor) and
    "pitch" / "roll" (rad; ramps: orientation Rz(yaw) Ry(pitch) Rx(roll)), and "kind": "terrain".

    props: list of free bodies the hands can pick up: {"name": "red cube",
    "shape": "box"|"sphere"|"cylinder", "size": MuJoCo half-sizes (box [hx, hy, hz],
    sphere [r], cylinder [r, half-height]; a number means a cube / ball),
    "pos": [x, y, z] centre, "rgba", "mass" kg, optional "yaw" rad}. Body and geom
    are named prop_<i> (group 1, colliding with hands, tables, floor and each other).
    They come after the robot in the tree, so the robot's qpos/qvel indices do not
    change, and each hand gets an inactive weld per prop ("<side>_hold_<i>") that
    ArmController can switch on as an optional grasp assist.
    """
    p = HumanoidParams.from_dict(p.to_dict())  # clipped copy
    tl, tr = p.torso_length, p.torso_radius
    sw = p.shoulder_width
    hands = p.arms and p.hands

    legs = _leg(p, "left", 1.0) + _leg(p, "right", -1.0)
    arms = (_arm(p, "left", 1.0) + _arm(p, "right", -1.0)) if p.arms else ""

    act = ""
    act += _servo("abdomen_y", p.abdomen_kp, p.abdomen_torque, p.kv_ratio, -30, 45)
    act += _servo("abdomen_x", p.abdomen_kp, p.abdomen_torque, p.kv_ratio, -25, 25)
    for side, sign in (("left", 1), ("right", -1)):
        roll = (-15, 35) if sign > 0 else (-35, 15)
        act += _servo(f"{side}_hip_x", p.hip_kp, p.hip_torque, p.kv_ratio, *roll)
        act += _servo(f"{side}_hip_z", p.hip_kp, p.hip_torque, p.kv_ratio, -30, 30)
        act += _servo(f"{side}_hip_y", p.hip_kp, p.hip_torque, p.kv_ratio, -110, 30)
        act += _servo(f"{side}_knee", p.knee_kp, p.knee_torque, p.kv_ratio, 0, 140)
        act += _servo(f"{side}_ankle_y", p.ankle_kp, p.ankle_torque, p.kv_ratio, -45, 45)
        act += _servo(f"{side}_ankle_x", p.ankle_kp, p.ankle_torque, p.kv_ratio, -20, 20)
    if p.arms:
        for side, sign in (("left", 1), ("right", -1)):
            roll = (-10, 100) if sign > 0 else (-100, 10)
            act += _servo(f"{side}_shoulder_y", p.arm_kp, p.arm_torque, p.kv_ratio, -120, 60)
            act += _servo(f"{side}_shoulder_x", p.arm_kp, p.arm_torque, p.kv_ratio, *roll)
            act += _servo(f"{side}_elbow", p.arm_kp, p.arm_torque, p.kv_ratio, -130, 0)
    # hand actuators come after the 20 body/arm ones so existing actuator indices stay valid
    equality = ""
    if hands:
        for side in ("left", "right"):
            act += _servo(f"{side}_wrist_y", p.arm_kp * 0.5, p.arm_torque * 0.5, p.kv_ratio, *WRIST_Y_RANGE)
            act += _servo(f"{side}_wrist_x", p.arm_kp * 0.5, p.arm_torque * 0.5, p.kv_ratio, *WRIST_X_RANGE)
            # ctrl = finger opening in metres (0 closed); 400 N/m saturating at 10 N squeezes a
            # 4-6 cm object with 7-10 N
            act += (f'    <position name="{side}_gripper" joint="{side}_gripper" kp="400" kv="{_f(400 * p.kv_ratio)}" '
                    f'ctrlrange="0 {_f(GRIPPER_OPEN)}" forcerange="-10 10"/>\n')
            equality += (f'    <joint name="{side}_gripper_mirror" joint1="{side}_gripper_mirror" joint2="{side}_gripper" '
                         f'polycoef="0 1 0 0 0" solref="0.01 1"/>\n')
        for i in range(len(props or [])):
            for side in ("left", "right"):
                equality += f'    <weld name="{side}_hold_{i}" body1="{side}_hand" body2="prop_{i}" active="false"/>\n'
    equality = f"  <equality>\n{equality}  </equality>\n" if equality else ""
    props_xml, custom = _props_xml(props or [])

    obs_xml = ""
    for i, o in enumerate(obstacles or []):
        hx, hy, hz = o["size"]
        x, y = o["pos"]
        yaw = float(o.get("yaw", 0.0)) * 57.29578
        rgba = _f(*o.get("rgba", (0.80, 0.45, 0.25, 1.0)))
        if o.get("shape") == "cylinder":
            shape = f'type="cylinder" size="{_f(hx, hz)}"'
        else:
            shape = f'type="box" size="{_f(hx, hy, hz)}"'
        zc = float(o.get("z", hz))
        if "pitch" in o or "roll" in o:
            # tilted terrain (ramps): R = Rz(yaw) Ry(pitch) Rx(roll), written as a quaternion
            cy, sy = math.cos(float(o.get("yaw", 0.0)) / 2), math.sin(float(o.get("yaw", 0.0)) / 2)
            cp, sp = math.cos(float(o.get("pitch", 0.0)) / 2), math.sin(float(o.get("pitch", 0.0)) / 2)
            cr, sr = math.cos(float(o.get("roll", 0.0)) / 2), math.sin(float(o.get("roll", 0.0)) / 2)
            quat = (cy * cp * cr + sy * sp * sr, cy * cp * sr - sy * sp * cr, cy * sp * cr + sy * cp * sr,
                    sy * cp * cr - cy * sp * sr)
            orient = f'quat="{_f(*quat)}"'
        else:
            orient = f'euler="0 0 {_f(yaw)}"'
        obs_xml += (f'    <geom name="obj_{i}" {shape} pos="{_f(x, y, zc)}" {orient} '
                    f'rgba="{rgba}" group="1" contype="1" conaffinity="1"/>\n')

    head_z = tl / 2 + p.head_radius + 0.05
    return f"""<mujoco model="simulatics_humanoid">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="{_f(p.timestep)}" integrator="implicitfast"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
  </visual>
  <default>
    <joint type="hinge" damping="{_f(p.joint_damping)}" armature="{_f(p.armature)}" limited="true"/>
    <!-- robot geoms collide with the world (floor, obstacles) but not each other -->
    <geom density="{_f(p.density)}" contype="1" conaffinity="0" condim="3" friction="{_f(p.friction, 0.02, 0.001)}"
          rgba="0.75 0.78 0.82 1"/>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.22 0.24 0.27" rgb2="0.27 0.29 0.33" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="40 40" reflectance="0.05"/>
  </asset>
{custom}  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="40 40 0.1" material="grid" group="2" contype="1" conaffinity="1"
          friction="{_f(p.friction, 0.02, 0.001)}" density="0"/>
{obs_xml}    <body name="torso" pos="0 0 {_f(p.stand_height)}">
      <freejoint name="root"/>
      <geom name="torso" type="capsule" fromto="0 0 {_f(-tl / 2)} 0 0 {_f(tl / 2)}" size="{_f(tr)}"/>
      <geom name="shoulders" type="capsule" fromto="0 {_f(-sw / 2)} {_f(tl / 2 - 0.02)} 0 {_f(sw / 2)} {_f(tl / 2 - 0.02)}" size="{_f(tr * 0.6)}"/>
      <geom name="head" type="sphere" pos="0 0 {_f(head_z)}" size="{_f(p.head_radius)}" rgba="0.92 0.80 0.55 1"/>
      <site name="imu" pos="0 0 0" size="0.02" rgba="1 0 0 0"/>
      <site name="lidar" pos="0 0 {_f(head_z + p.head_radius + 0.03)}" size="0.03" rgba="0.2 0.9 0.4 1"/>
      <camera name="head_cam" pos="{_f(p.head_radius + 0.01, 0, head_z)}" xyaxes="0 -1 0 0.25 0 1" fovy="75"/>
      <!-- looks 55 deg down at the hands' workspace (tables, objects in reach) -->
      <camera name="work_cam" pos="{_f(p.head_radius + 0.01, 0, head_z - 0.02)}" xyaxes="0 -1 0 0.819 0 0.574" fovy="80"/>
      <camera name="chase" pos="-2.6 0 0.9" xyaxes="0 -1 0 0.3 0 1" mode="trackcom" fovy="55"/>
      <body name="pelvis" pos="0 0 {_f(-p.waist_offset)}">
        <joint name="abdomen_y" axis="0 1 0" range="-30 45"/>
        <joint name="abdomen_x" axis="1 0 0" range="-25 25"/>
        <geom name="pelvis" type="capsule" fromto="0 {_f(-p.pelvis_width / 2)} {_f(-p.pelvis_drop)} 0 {_f(p.pelvis_width / 2)} {_f(-p.pelvis_drop)}" size="{_f(max(p.leg_radius * 1.3, 0.05))}"/>{legs}
      </body>{arms}
    </body>
{props_xml}  </worldbody>
{equality}  <actuator>
{act}  </actuator>
  <sensor>
    <gyro name="gyro" site="imu"/>
    <accelerometer name="accel" site="imu"/>
    <framequat name="orientation" objtype="site" objname="imu"/>
    <touch name="left_foot_touch" site="left_foot_site"/>
    <touch name="right_foot_touch" site="right_foot_site"/>
  </sensor>
</mujoco>
"""
