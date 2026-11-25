"""Free scene building: a catalogue of things to drop into the world, and edits on what is there.

Everything in a scene is one of two lists the model is built from (models.humanoid.build_mjcf):
    obstacles   static shapes (boxes, cylinders, ramps); "kind": "terrain" marks low ground to walk on
    props       loose bodies that can be pushed, kicked or picked up
Each entry gets an "oid" so it can be found again after the lists change; things made of several shapes
(a hurdle: a bar and two posts) share a "group".
"""
from __future__ import annotations

import itertools
import math

from .scene import PALETTE
from .terrain import TERRAIN_RGBA, _ramp

_ids = itertools.count(1)
WOOD = [0.72, 0.55, 0.36, 1.0]
STEEL = [0.62, 0.65, 0.70, 1.0]


def new_oid() -> str:
    return f"o{next(_ids)}"


def _rgba(colour: str) -> list[float]:
    return [*PALETTE.get(colour, PALETTE["blue"]), 1.0]


def _static(name, x, y, hx, hy, height, rgba, yaw=0.0, shape="box", bottom=0.0, **extra) -> dict:
    return {"name": name, "shape": shape, "pos": [float(x), float(y)], "size": [float(hx), float(hy), float(height / 2)],
            "z": float(bottom + height / 2), "yaw": float(yaw), "rgba": [float(v) for v in rgba], **extra}


def _loose(name, shape, x, y, z, size, mass, rgba) -> dict:
    return {"name": name, "shape": shape, "pos": [float(x), float(y), float(z)], "size": [float(s) for s in size],
            "mass": float(mass), "rgba": [float(v) for v in rgba], "yaw": 0.0}


def _box(x, y, yaw, k, colour):
    return [_static(f"{colour} box", x, y, 0.25 * k, 0.25 * k, 0.5 * k, _rgba(colour), yaw)], []


def _wall(x, y, yaw, k, colour):
    return [_static("wall", x, y, 0.08, 1.0 * k, 1.2, [0.80, 0.82, 0.78, 1.0], yaw)], []


def _pillar(x, y, yaw, k, colour):
    return [_static(f"{colour} pillar", x, y, 0.15 * k, 0.15 * k, 1.4, _rgba(colour), yaw, shape="cylinder")], []


def _platform(x, y, yaw, k, colour):
    return [_static("platform", x, y, 0.6 * k, 0.6 * k, 0.12, TERRAIN_RGBA["ramp"], yaw, kind="terrain")], []


def _step(x, y, yaw, k, colour):
    return [_static("step", x, y, 0.3, 0.6 * k, 0.05, TERRAIN_RGBA["steps"], yaw, kind="terrain")], []


def _ramp_obj(x, y, yaw, k, colour):
    r = _ramp("ramp", 0.0, 1.4 * k, math.radians(8.0), 0.0, 1.2)
    dx = r["pos"][0]                               # the ramp's centre is half its run ahead of its start
    r["pos"] = [float(x + dx * math.cos(yaw) - 0.0), float(y + dx * math.sin(yaw))]
    r["yaw"] = float(yaw)
    return [r], []


def _table(x, y, yaw, k, colour):
    from .props import table_geometry
    h = table_geometry()["height"]
    return [_static("table", x, y, 0.3 * k, 0.5 * k, h, WOOD, yaw)], []


def _jump_box(x, y, yaw, k, colour):
    return [_static("jump box", x, y, 0.35, 0.35, 0.5 * k, WOOD, yaw)], []


def _hurdle(x, y, yaw, k, colour):
    h, w = 0.3 * k, 1.0
    c, s = math.cos(yaw), math.sin(yaw)
    out = [_static("hurdle", x, y, 0.015, w / 2, 0.03, [0.95, 0.55, 0.15, 1.0], yaw, bottom=h - 0.03)]
    for i, side in enumerate((-1, 1)):
        out.append(_static(f"hurdle post {i + 1}", x - s * side * w / 2, y + c * side * w / 2, 0.02, 0.02, h, STEEL, yaw))
    return out, []


def _ball(x, y, yaw, k, colour):
    r = 0.1 * k
    return [], [_loose(f"{colour} ball", "sphere", x, y, r + 0.005, [r], 0.3 * k ** 3, _rgba(colour))]


def _crate(x, y, yaw, k, colour):
    h = 0.15 * k
    return [], [_loose("crate", "box", x, y, h + 0.005, [h, h, h], 1.2 * k ** 3, WOOD)]


def _cube(x, y, yaw, k, colour):
    return [], [_loose(f"{colour} cube", "box", x, y, 0.03, [0.025, 0.025, 0.025], 0.06, _rgba(colour))]


def _cup(x, y, yaw, k, colour):
    return [], [_loose(f"{colour} cup", "cylinder", x, y, 0.035, [0.025, 0.03], 0.08, _rgba(colour))]


# key -> (label, hint, maker)
CATALOG: dict[str, tuple] = {
    "box": ("Box", "A solid block to walk around", _box),
    "wall": ("Wall", "A 2 m wall, 1.2 m high", _wall),
    "pillar": ("Pillar", "A round column", _pillar),
    "platform": ("Platform", "A low platform the robot can step onto (12 cm)", _platform),
    "step": ("Step", "A 5 cm step to walk over", _step),
    "ramp": ("Ramp", "An 8 degree ramp the robot can walk up", _ramp_obj),
    "table": ("Table", "A worktop at the height the robot's arms work at", _table),
    "jump_box": ("Jump box", "A box to jump onto (too high to step on)", _jump_box),
    "hurdle": ("Hurdle", "A bar on two posts", _hurdle),
    "ball": ("Ball", "A loose ball to kick or push", _ball),
    "crate": ("Crate", "A loose crate to push", _crate),
    "cube": ("Small cube", "A 5 cm cube the hands can pick up", _cube),
    "cup": ("Cup", "A cup the hands can pick up", _cup),
}
COLOURS = list(PALETTE)


def make(kind: str, x: float, y: float, yaw: float = 0.0, scale: float = 1.0, colour: str = "blue") -> tuple[list[dict], list[dict]]:
    """(obstacles, props) for one catalogue item at (x, y); every part carries the same new "group"."""
    if kind not in CATALOG:
        raise ValueError(f"unknown object {kind!r} (choose from {', '.join(CATALOG)})")
    scale = float(min(max(scale, 0.25), 4.0))
    obstacles, props = CATALOG[kind][2](float(x), float(y), float(yaw), scale, colour)
    group = new_oid()
    for o in obstacles + props:
        o["oid"], o["group"], o["made"] = new_oid(), group, kind
    return obstacles, props


def ensure_ids(obstacles: list[dict], props: list[dict]) -> None:
    """Give every entry an oid (worlds built elsewhere have none) and its own group."""
    for o in obstacles + props:
        if "oid" not in o:
            o["oid"] = new_oid()
        if "group" not in o:
            o["group"] = o["oid"]


def members(obstacles: list[dict], props: list[dict], group: str) -> list[dict]:
    return [o for o in obstacles + props if o.get("group") == group]


def move(parts: list[dict], x: float, y: float) -> None:
    """Move a group so that its first part is at (x, y); the others keep their offsets."""
    dx, dy = float(x) - parts[0]["pos"][0], float(y) - parts[0]["pos"][1]
    for o in parts:
        o["pos"] = [o["pos"][0] + dx, o["pos"][1] + dy, *o["pos"][2:]]


def rotate(parts: list[dict], yaw: float) -> None:
    """Turn a group about its first part to face `yaw` (rad)."""
    cx, cy = parts[0]["pos"][:2]
    d = float(yaw) - float(parts[0].get("yaw", 0.0))
    c, s = math.cos(d), math.sin(d)
    for o in parts:
        rx, ry = o["pos"][0] - cx, o["pos"][1] - cy
        o["pos"] = [cx + c * rx - s * ry, cy + s * rx + c * ry, *o["pos"][2:]]
        o["yaw"] = float(o.get("yaw", 0.0)) + d
