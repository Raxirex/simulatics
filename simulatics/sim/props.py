"""Graspable props and table-top manipulation scenes.

Props are free bodies (``prop_<i>``) built by ``models.humanoid.build_mjcf(...,
props=...)``. Tables are ordinary static box obstacles, so the planner and the
lidar treat them like any other obstacle.

The table height is not a fixed 0.75 m: the built-in humanoid has no hip/waist
bending under arm control, so the table is sized from the robot's own geometry
(shoulder height, arm length, toe position) such that objects on it can be
reached from above. For the default body that gives a 0.92 m counter whose near
edge is 0.25 m in front of the robot's root (centre ~0.5 m ahead).
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..models.humanoid import HAND_GRASP, HAND_PALM, HumanoidParams
from .scene import PALETTE

TABLE_RGBA = [0.55, 0.40, 0.26, 1.0]


def table_geometry(params: HumanoidParams | None = None) -> dict:
    """Where a table must stand so the hands can grasp objects on it from above.

    Returns {"near_x", "depth", "width", "height", "grasp_x": (lo, hi), "grasp_y": (lo, hi)}
    in metres (grasp_y is |y|, one band per arm). The far grasp line is placed where
    the wrist is at 88 % of the arm's length from the shoulder with the fingers
    pointing straight down (a comfortable, non-singular elbow angle for damped least
    squares IK). The arms have no shoulder yaw and only 10 deg of inward roll, so the
    band starts 3 cm inside the shoulder: closer to the midline a top-down grasp is out
    of reach (mapped with ArmController's IK for the default body).
    """
    p = HumanoidParams.from_dict((params or HumanoidParams()).to_dict())
    shoulder_z = p.stand_height + p.torso_length / 2 - 0.02
    toe_x = p.foot_length * 0.7                      # foot box centre 0.2 * fl ahead of the ankle, half-length 0.5 * fl
    near_x = max(0.25, toe_x + 0.09)                 # keep the shins clear of the table edge
    grasp_lo, grasp_hi = near_x + 0.04, near_x + 0.12
    arm = p.upper_arm_length + p.forearm_length
    dz = np.sqrt(max((0.88 * arm) ** 2 - grasp_hi ** 2, 0.01))
    hand = HAND_PALM + HAND_GRASP
    height = float(np.clip(shoulder_z - dz - hand - 0.025, 0.4, 1.3))
    sy = p.shoulder_width / 2
    return {"near_x": near_x, "depth": 0.5, "width": 1.0, "height": height, "grasp_x": (grasp_lo, grasp_hi),
            "grasp_y": (sy - 0.03, sy + 0.14)}


def manipulation_scene(seed: int | None = None, params: HumanoidParams | None = None,
                       n_props: int | None = None) -> tuple[list[dict], list[dict]]:
    """A table in front of the robot with 2-4 small coloured cubes and balls on it.

    Returns (obstacles, props) ready for ``build_mjcf(params, obstacles, props)``.
    Objects are 4-6 cm (graspable by the two-finger hands), uniquely named like
    "red cube" / "blue ball", alternate between the left and right arm's grasp band
    (see table_geometry) and are at least 11 cm apart, so an open hand coming down on
    one does not hit a neighbour.
    """
    rng = np.random.default_rng(seed)
    t = table_geometry(params)
    hz = t["height"] / 2
    table = {"name": "table", "shape": "box", "colour": "brown", "rgba": TABLE_RGBA,
             "pos": [t["near_x"] + t["depth"] / 2, 0.0], "size": [t["depth"] / 2, t["width"] / 2, hz], "yaw": 0.0}
    n = int(np.clip(n_props if n_props is not None else rng.integers(2, 5), 1, 4))
    colours = list(rng.permutation(list(PALETTE)))
    first = rng.choice([-1.0, 1.0])
    props: list[dict] = []
    tries = 0
    while len(props) < n and tries < 2000:
        tries += 1
        x = rng.uniform(*t["grasp_x"])
        y = first * (-1) ** len(props) * rng.uniform(*t["grasp_y"])
        if any(np.hypot(x - q["pos"][0], y - q["pos"][1]) < 0.11 for q in props):
            continue
        colour = str(colours[len(props) % len(colours)])
        if rng.random() < 0.65:
            h = float(rng.uniform(0.02, 0.03))               # 4-6 cm cube
            shape, kind, size, vol = "box", "cube", [h, h, h], (2 * h) ** 3
        else:
            h = float(rng.uniform(0.02, 0.028))              # 4-5.6 cm ball
            shape, kind, size, vol = "sphere", "ball", [h], 4 / 3 * np.pi * h ** 3
        props.append({"name": f"{colour} {kind}", "shape": shape, "colour": colour, "rgba": [*PALETTE[colour], 1.0],
                      "size": size, "pos": [float(x), float(y), t["height"] + h + 0.002],
                      "mass": float(np.clip(vol * 500, 0.03, 0.15)), "yaw": 0.0})
    return [table], props


# ---- reading props back from a compiled model -------------------------------------
def prop_names(model: mujoco.MjModel) -> dict[int, str]:
    """body id -> human-readable name for every prop_<i> body (from the model's custom text)."""
    out: dict[int, str] = {}
    for b in range(model.nbody):
        name = model.body(b).name
        if not name.startswith("prop_"):
            continue
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, name)
        if tid >= 0:
            adr, size = int(model.text_adr[tid]), int(model.text_size[tid])
            out[b] = bytes(model.text_data[adr:adr + size]).split(b"\0")[0].decode("utf-8", "replace")
        else:
            out[b] = name
    return out


def props_state(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict]:
    """[{"name", "body": "prop_<i>", "shape", "x", "y", "z", "size_m"}] with live positions."""
    shapes = {int(mujoco.mjtGeom.mjGEOM_BOX): "cube", int(mujoco.mjtGeom.mjGEOM_SPHERE): "ball",
              int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder"}
    out = []
    for b, name in prop_names(model).items():
        g = int(model.body_geomadr[b])
        x, y, z = (float(v) for v in data.xpos[b])
        out.append({"name": name, "body": model.body(b).name, "shape": shapes.get(int(model.geom_type[g]), "object"),
                    "x": round(x, 4), "y": round(y, 4), "z": round(z, 4),
                    "size_m": round(2 * float(model.geom_size[g, 0]), 4)})
    return out
