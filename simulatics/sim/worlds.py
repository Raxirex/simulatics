"""Ready-made worlds for the built-in humanoid: a playground, a parkour course and a kitchen.

Each builder returns ``(obstacles, props, info)``:
    obstacles   static shapes for models.humanoid.build_mjcf (boxes, cylinders, ramps; "kind": "terrain" marks
                ground the robot walks on rather than around)
    props       free bodies that can be pushed, kicked and (when small) picked up
    info        {"goal": (x, y) or None, "waypoints": [...], "about": one line}

The robot starts at the origin facing +x; every world keeps that spot clear.
"""
from __future__ import annotations

import numpy as np

from .scene import PALETTE
from .terrain import TERRAIN_RGBA, _box, _ramp, course_length, mixed_course

WOOD = [0.72, 0.55, 0.36, 1.0]
STEEL = [0.62, 0.65, 0.70, 1.0]
WHITE = [0.90, 0.90, 0.88, 1.0]
DARK = [0.16, 0.17, 0.20, 1.0]
ORANGE = [0.95, 0.55, 0.15, 1.0]


def _solid(name: str, x: float, y: float, hx: float, hy: float, height: float, rgba, yaw: float = 0.0,
           shape: str = "box", bottom: float = 0.0, **extra) -> dict:
    """A static shape standing on the floor (or floating with its underside at `bottom`)."""
    return {"name": name, "shape": shape, "pos": [float(x), float(y)], "size": [float(hx), float(hy), float(height / 2)],
            "z": float(bottom + height / 2), "yaw": float(yaw), "rgba": [float(v) for v in rgba], **extra}


def _prop(name: str, shape: str, x: float, y: float, z: float, size, mass: float, rgba) -> dict:
    return {"name": name, "shape": shape, "pos": [float(x), float(y), float(z)],
            "size": [float(s) for s in (size if isinstance(size, (list, tuple)) else [size])], "mass": float(mass),
            "rgba": [float(v) for v in rgba], "yaw": 0.0}


def _hurdle(name: str, x: float, y: float, height: float, width: float = 1.0, yaw: float = 0.0) -> list[dict]:
    """A bar on two posts."""
    c, s = np.cos(yaw), np.sin(yaw)
    out = [_solid(name, x, y, 0.015, width / 2, 0.03, ORANGE, yaw, bottom=height - 0.03)]
    for k, side in enumerate((-1, 1)):
        px, py = x - s * side * width / 2, y + c * side * width / 2
        out.append(_solid(f"{name} post {k + 1}", px, py, 0.02, 0.02, height, STEEL, yaw))
    return out


# ------------------------------------------------------------------------------------------- playground
def playground(seed: int | None = 0) -> tuple[list[dict], list[dict], dict]:
    """Things to walk around, climb, push and kick: blocks, a ramp with a platform, low steps, a slalom of
    posts, a tunnel frame, balls and light crates."""
    rng = np.random.default_rng(seed)
    obs: list[dict] = []
    props: list[dict] = []
    col = lambda name: [*PALETTE[name], 1.0]   # noqa: E731
    # ramp up to a platform and steps down the other side (walkable)
    ramp_len, top = 1.6, 0.24
    ramp = _ramp("ramp", 1.4, ramp_len, float(np.arctan2(top, ramp_len)), 0.0, 1.2)
    ramp["pos"][1] = 2.5
    obs.append(ramp)
    obs.append(_box("platform", 3.0, 4.2, 1.9, 3.1, top, TERRAIN_RGBA["ramp"]))
    for i in range(4):
        obs.append(_box(f"step {i + 1}", 4.2 + 0.35 * i, 4.55 + 0.35 * i, 1.9, 3.1, top - 0.05 * (i + 1) + 0.01,
                        TERRAIN_RGBA["stairs"]))
    # stepping blocks
    for i in range(5):
        obs.append(_box(f"stepping block {i + 1}", 1.2 + 0.55 * i, 1.65 + 0.55 * i, -2.9, -2.1, 0.03 + 0.02 * (i % 3),
                        TERRAIN_RGBA["steps"]))
    # slalom posts
    for i in range(5):
        obs.append(_solid(f"post {i + 1}", 1.5 + 1.1 * i, 0.55 * (-1) ** i, 0.07, 0.07, 1.0, col("yellow"), shape="cylinder"))
    # a tunnel frame to walk through (two pillars and a beam overhead)
    obs.append(_solid("tunnel pillar left", -2.6, 0.9, 0.1, 0.1, 2.0, col("blue")))
    obs.append(_solid("tunnel pillar right", -2.6, -0.9, 0.1, 0.1, 2.0, col("blue")))
    obs.append(_solid("tunnel beam", -2.6, 0.0, 0.1, 1.0, 0.16, col("blue"), bottom=2.0))
    # big blocks and cylinders around the edge
    names = list(PALETTE)
    for i in range(10):
        ang = 2 * np.pi * i / 10 + rng.uniform(-0.15, 0.15)
        r = rng.uniform(6.2, 8.0)
        colour = names[i % len(names)]
        if i % 3 == 0:
            obs.append(_solid(f"{colour} tower", r * np.cos(ang), r * np.sin(ang), 0.3, 0.3, rng.uniform(0.8, 1.6),
                              col(colour), shape="cylinder"))
        else:
            obs.append(_solid(f"{colour} block", r * np.cos(ang), r * np.sin(ang), rng.uniform(0.3, 0.7),
                              rng.uniform(0.3, 0.7), rng.uniform(0.4, 1.2), col(colour), yaw=rng.uniform(0, np.pi)))
    # things that move: balls to kick, crates to push, a few small graspable pieces on a low table
    for i, (x, y, r, colour) in enumerate(((1.3, -0.9, 0.11, "red"), (2.4, -1.2, 0.09, "green"), (0.9, -1.9, 0.13, "blue"),
                                           (-1.2, -1.4, 0.10, "yellow"))):
        props.append(_prop(f"{colour} ball", "sphere", x, y, r + 0.01, [r], 0.35, col(colour)))
    for i, (x, y) in enumerate(((-1.5, 1.2), (-1.5, 1.6), (-1.1, 1.4))):
        props.append(_prop(f"crate {i + 1}", "box", x, y, 0.151, [0.15, 0.15, 0.15], 1.2, WOOD))
    props.append(_prop("stacked crate", "box", -1.5, 1.4, 0.452, [0.15, 0.15, 0.15], 1.2, WOOD))
    obs.append(_solid("low table", 0.0, -2.4, 0.35, 0.25, 0.5, WOOD))
    props.append(_prop("red cube", "box", -0.1, -2.4, 0.526, [0.025, 0.025, 0.025], 0.06, col("red")))
    props.append(_prop("green ball", "sphere", 0.12, -2.35, 0.526, [0.025], 0.05, col("green")))
    way = [(6.6, 0.0), (6.6, 1.25), (0.6, 1.25), (0.0, 0.0)]      # out through the posts, back beside the ramp
    return obs, props, {"goal": way[-1], "waypoints": way,
                        "about": "Playground: ramp and platform, stepping blocks, slalom posts, a tunnel, balls and crates"}


# ---------------------------------------------------------------------------------------------- parkour
def parkour(seed: int | None = 0) -> tuple[list[dict], list[dict], dict]:
    """A course along +x. The centre lane is ground the walking controller can cross (low random steps, stairs
    up and down, a ramp); beside it stand the things the jumping samples use (a jump box, hurdles, a high
    platform), and a balance beam."""
    obs = mixed_course(seed=seed)                    # walkable: steps, stairs, ramp (kind = terrain)
    end = course_length(obs)
    # left of the lane: hurdles of rising height
    for i, h in enumerate((0.15, 0.25, 0.35)):
        obs += _hurdle(f"hurdle {i + 1}", 1.4 + 1.3 * i, 1.9, h)
    # right of the lane: jump boxes of rising height and a high platform with a drop
    for i, (h, colour) in enumerate(((0.3, "green"), (0.5, "yellow"), (0.7, "red"))):
        obs.append(_solid(f"{colour} jump box", 1.4 + 1.4 * i, -2.0, 0.35, 0.35, h, [*PALETTE[colour], 1.0]))
    obs.append(_solid("drop platform", end - 0.6, -2.1, 0.6, 0.5, 0.8, WOOD))
    # a balance beam and a wall to finish
    obs.append(_box("balance beam", end + 0.9, end + 3.3, 1.75, 2.05, 0.12, TERRAIN_RGBA["stairs"]))
    obs.append(_solid("finish wall", end + 2.2, 0.0, 0.08, 1.2, 1.1, [*PALETTE["blue"], 1.0], bottom=0.0, yaw=0.0))
    props = [_prop("red ball", "sphere", 0.6, 1.2, 0.11, [0.1], 0.3, [*PALETTE["red"], 1.0])]
    goal = (end + 0.5, 0.0)
    return obs, props, {"goal": goal, "waypoints": [goal],
                        "about": "Parkour: steps, stairs and a ramp down the middle; hurdles, jump boxes, a drop "
                                 "platform and a balance beam beside it"}


# ---------------------------------------------------------------------------------------------- kitchen
def kitchen(seed: int | None = 0, params=None) -> tuple[list[dict], list[dict], dict]:
    """A kitchen: a worktop in front of the robot with things to pick up, an L-shaped counter with a hob, a
    sink and wall cupboards, a fridge, an island, and a dining table with stools."""
    from .props import table_geometry
    t = table_geometry(params)
    h = t["height"]                                  # worktop height the robot's arms can work at
    obs: list[dict] = []
    props: list[dict] = []
    near = t["near_x"]
    # worktop in front of the robot (objects on it are within reach from the start position)
    obs.append(_solid("worktop", near + t["depth"] / 2, 0.0, t["depth"] / 2, 0.9, h, WHITE))
    # back counter along +x behind the worktop, with hob and sink; cupboards above
    back = near + t["depth"] + 0.9
    obs.append(_solid("back counter", back + 0.3, 0.0, 0.3, 2.2, h, WHITE))
    obs.append(_solid("hob", back + 0.3, -0.9, 0.26, 0.3, 0.012, DARK, bottom=h))
    for i, (dx, dy) in enumerate(((-0.12, -0.14), (0.12, -0.14), (-0.12, 0.14), (0.12, 0.14))):
        obs.append(_solid(f"burner {i + 1}", back + 0.3 + dx, -0.9 + dy, 0.07, 0.07, 0.008, [0.45, 0.1, 0.08, 1.0],
                          shape="cylinder", bottom=h + 0.012))
    obs.append(_solid("sink", back + 0.3, 0.8, 0.22, 0.3, 0.02, STEEL, bottom=h))
    obs.append(_solid("tap", back + 0.5, 0.8, 0.015, 0.015, 0.25, STEEL, shape="cylinder", bottom=h))
    obs.append(_solid("wall cupboards", back + 0.42, 0.0, 0.18, 2.2, 0.6, WOOD, bottom=h + 0.55))
    obs.append(_solid("back wall", back + 0.66, 0.0, 0.05, 2.6, 2.3, [0.82, 0.84, 0.80, 1.0]))
    # side counter (the L) and fridge
    obs.append(_solid("side counter", back - 0.9, 2.2 + 0.3, 0.9, 0.3, h, WHITE))
    obs.append(_solid("fridge", back - 0.1, -2.2 - 0.4, 0.38, 0.38, 1.85, STEEL))
    obs.append(_solid("fridge handle", back - 0.5, -2.45, 0.015, 0.015, 0.6, DARK, bottom=0.9))
    obs.append(_solid("oven", back - 1.4, 2.2 + 0.3, 0.3, 0.31, 0.6, DARK, bottom=0.12))
    # island and dining table with stools, behind and beside the start position
    obs.append(_solid("island", -1.9, 1.4, 0.45, 0.8, h, WOOD))
    obs.append(_solid("dining table top", -2.2, -1.6, 0.6, 0.45, 0.04, WOOD, bottom=0.72))
    for i, (dx, dy) in enumerate(((-0.52, -0.37), (0.52, -0.37), (-0.52, 0.37), (0.52, 0.37))):
        obs.append(_solid(f"table leg {i + 1}", -2.2 + dx, -1.6 + dy, 0.03, 0.03, 0.72, WOOD))
    for i, (dx, dy) in enumerate(((0.0, -0.8), (0.0, 0.8), (-0.95, 0.0))):
        obs.append(_solid(f"stool {i + 1}", -2.2 + dx, -1.6 + dy, 0.17, 0.17, 0.45, DARK, shape="cylinder"))
    # things on the worktop, inside the arms' grasp bands (see props.table_geometry)
    gx = float(np.mean(t["grasp_x"]))
    gy = float(np.mean(t["grasp_y"]))
    props.append(_prop("red cup", "cylinder", gx, gy, h + 0.032, [0.025, 0.03], 0.08, [*PALETTE["red"], 1.0]))
    props.append(_prop("green apple", "sphere", gx, -gy, h + 0.03, [0.028], 0.07, [*PALETTE["green"], 1.0]))
    props.append(_prop("butter box", "box", gx + 0.02, gy + 0.16, h + 0.022, [0.03, 0.02, 0.02], 0.06, [*PALETTE["yellow"], 1.0]))
    # further along the worktop and on the other surfaces (to walk to)
    props.append(_prop("blue bowl", "cylinder", near + 0.25, -0.7, h + 0.03, [0.06, 0.028], 0.15, [*PALETTE["blue"], 1.0]))
    props.append(_prop("pan", "cylinder", back + 0.18, -1.04, h + 0.045, [0.09, 0.02], 0.4, DARK))
    props.append(_prop("orange", "sphere", -1.9, 1.3, h + 0.035, [0.033], 0.1, ORANGE))
    props.append(_prop("plate", "cylinder", -2.2, -1.6, 0.775, [0.1, 0.008], 0.2, WHITE))
    way = [(-0.9, 0.0), (-1.0, -0.6), (-1.2, 0.4), (0.0, 0.0)]
    return obs, props, {"goal": None, "waypoints": way,
                        "about": "Kitchen: worktop with a cup, an apple and a butter box in reach; hob, sink, "
                                 "cupboards, fridge, island and a dining table"}


WORLDS = {"playground": playground, "parkour": parkour, "kitchen": kitchen}
LABELS = {"playground": "Playground", "parkour": "Parkour course", "kitchen": "Kitchen"}
