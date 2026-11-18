"""World objects with names and colours, so planners and LLMs can refer to them ("the red box")."""
from __future__ import annotations

import numpy as np

PALETTE: dict[str, tuple[float, float, float]] = {
    "red": (0.86, 0.22, 0.20), "blue": (0.22, 0.42, 0.92), "green": (0.22, 0.70, 0.32),
    "yellow": (0.95, 0.80, 0.18), "purple": (0.60, 0.32, 0.82), "orange": (0.96, 0.52, 0.16),
    "white": (0.92, 0.92, 0.92), "cyan": (0.20, 0.80, 0.86),
}


def random_objects(n: int = 12, seed: int | None = None, extent: float = 9.0, spawn_clear: float = 1.2) -> list[dict]:
    """n boxes and cylinders on the floor, each with a unique human-readable name like "blue cylinder 2"."""
    rng = np.random.default_rng(seed)
    objs: list[dict] = []
    counts: dict[str, int] = {}
    colours = list(PALETTE)
    tries = 0
    while len(objs) < n and tries < 20 * n:
        tries += 1
        x, y = rng.uniform(-extent, extent, 2)
        shape = "cylinder" if rng.random() < 0.35 else "box"
        hx, hy = rng.uniform(0.2, 0.9, 2)
        if shape == "cylinder":
            hx = hy = float(rng.uniform(0.15, 0.5))
        if np.hypot(x, y) < spawn_clear + np.hypot(hx, hy):  # keep the spawn area clear
            continue
        colour = colours[int(rng.integers(len(colours)))]
        base = f"{colour} {shape}"
        counts[base] = counts.get(base, 0) + 1
        objs.append({
            "name": base if counts[base] == 1 else f"{base} {counts[base]}",
            "shape": shape, "colour": colour, "rgba": [*PALETTE[colour], 1.0],
            "pos": [float(x), float(y)], "yaw": float(rng.uniform(0, np.pi)),
            "size": [float(hx), float(hy), float(rng.uniform(0.3, 1.0))],
        })
    # names that ended up unique keep no suffix; the first of a duplicated name gets " 1"
    for o in objs:
        base = f"{o['colour']} {o['shape']}"
        if counts[base] > 1 and o["name"] == base:
            o["name"] = f"{base} 1"
    return objs


def manipulation_scene(seed: int | None = None, params=None, n_props: int | None = None) -> tuple[list[dict], list[dict]]:
    """A table in front of the robot with 2-4 graspable cubes / balls -> (obstacles, props).

    See sim/props.py (imported lazily: props.py uses this module's PALETTE).
    """
    from .props import manipulation_scene as _scene
    return _scene(seed, params, n_props)
