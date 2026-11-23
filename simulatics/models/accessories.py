"""Accessories mounted on the robot: lidar, cameras, payloads.

An accessory is a plain dict, stored by the session and applied to the generated MJCF with MjSpec:

    {"id": "a3", "type": "lidar" | "camera" | "payload", "body": "torso",
     "pos": [x, y, z], "quat": [w, x, y, z],     # mount frame in the body's frame
     "fovy": 70, "mass": 0.25, ...}              # type parameters

Each accessory becomes a visual/inertial geom named ``acc_<id>`` on its body (no self-collision; a payload
also collides with the world). The lidar adds the site ``lidar`` that sim/sensors.py casts from (one lidar
per robot); a camera adds a camera ``cam_<id>`` looking along the mount frame's -z axis.

`place()` turns a point and surface normal picked on the robot (at the current pose) into a mount pose on the
touched body, snapped to a grid in the body frame and sitting on the surface.
"""
from __future__ import annotations

import itertools

import mujoco
import numpy as np

TYPES: dict[str, dict] = {
    "lidar": {"label": "Lidar", "shape": "cylinder", "size": [0.045, 0.025], "mass": 0.1,
              "rgba": [0.15, 0.75, 0.45, 1.0], "half_depth": 0.025,
              "hint": "Spinning range sensor. Stays level wherever it is mounted. One per robot."},
    "camera": {"label": "Camera", "shape": "box", "size": [0.025, 0.02, 0.015], "mass": 0.08, "fovy": 70.0,
               "rgba": [0.18, 0.2, 0.26, 1.0], "half_depth": 0.015,
               "hint": "Looks straight out of the surface it is mounted on."},
    "payload": {"label": "Payload box", "shape": "box", "size": [0.08, 0.06, 0.05], "mass": 2.0,
                "rgba": [0.82, 0.6, 0.3, 1.0], "half_depth": 0.05,
                "hint": "A mass (battery, backpack, tool case). Changes balance and inertia."},
}

_ids = itertools.count(1)


def new_id() -> str:
    return f"a{next(_ids)}"


def _quat_from_matrix(R: np.ndarray) -> list[float]:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, float).reshape(-1))
    return [float(v) for v in q]


def default_accessories(xml: str) -> list[dict]:
    """The built-in robot's own lidar (on top of the head), as an editable accessory."""
    spec = mujoco.MjSpec.from_string(xml)
    try:
        site = spec.site("lidar")
    except Exception:  # noqa: BLE001 - model without a lidar site
        return []
    if site is None:
        return []
    pos = np.array(site.pos, float)
    pos[2] -= 0.01  # rest the puck on the head instead of floating above it
    return [{"id": "lidar", "type": "lidar", "body": site.parent.name, "pos": pos.round(4).tolist(),
             "quat": [1.0, 0.0, 0.0, 0.0], "mass": TYPES["lidar"]["mass"]}]


def apply_accessories(xml: str, accessories: list[dict]) -> tuple[str, list[str]]:
    """MJCF with the accessories mounted. Returns (xml, warnings) - unknown bodies are skipped, not fatal."""
    spec = mujoco.MjSpec.from_string(xml)
    warnings = apply_to_spec(spec, accessories)
    return spec.to_xml(), warnings


def apply_to_spec(spec: "mujoco.MjSpec", accessories: list[dict]) -> list[str]:
    """Mount accessories on an MjSpec in place (works for imported robots loaded from file with assets).

    The robot then has a lidar only if one is among the accessories: a built-in "lidar" site is removed
    otherwise (that is how the default head lidar is switched off)."""
    warnings: list[str] = []
    if not any(a.get("type") == "lidar" for a in accessories):
        try:
            old = spec.site("lidar")
            if old is not None:
                spec.delete(old)
        except Exception:  # noqa: BLE001 - no such site
            pass
    lidar_done = False
    for a in accessories:
        t = TYPES.get(a.get("type"))
        if t is None:
            warnings.append(f"unknown accessory type {a.get('type')!r}")
            continue
        try:
            body = spec.body(str(a["body"]))
        except Exception:  # noqa: BLE001
            body = None
        if body is None:
            warnings.append(f"{t['label']} {a['id']}: this robot has no body {a['body']!r} (skipped)")
            continue
        pos, quat = [float(v) for v in a["pos"]], [float(v) for v in a.get("quat", [1, 0, 0, 0])]
        size = list(a.get("size") or t["size"])
        gtype = mujoco.mjtGeom.mjGEOM_CYLINDER if t["shape"] == "cylinder" else mujoco.mjtGeom.mjGEOM_BOX
        size = (size + [0, 0, 0])[:3]
        payload = a["type"] == "payload"
        body.add_geom(name=f"acc_{a['id']}", type=gtype, size=size, pos=pos, quat=quat,
                      mass=float(a.get("mass", t["mass"])), rgba=list(a.get("rgba") or t["rgba"]),
                      contype=1 if payload else 0, conaffinity=0, group=0)
        if a["type"] == "lidar":
            if lidar_done:
                warnings.append("only one lidar is used; extra lidars are mounted as dummies")
                continue
            try:
                old = spec.site("lidar")
                if old is not None:
                    spec.delete(old)
            except Exception:  # noqa: BLE001
                pass
            body.add_site(name="lidar", pos=pos, quat=quat, size=[0.02, 0, 0], rgba=[0, 0, 0, 0])
            lidar_done = True
        elif a["type"] == "camera":
            body.add_camera(name=f"cam_{a['id']}", pos=pos, quat=quat, fovy=float(a.get("fovy", t["fovy"])))
    return warnings


def place(model: mujoco.MjModel, data: mujoco.MjData, acc_type: str, geom: int, point, normal,
          snap: float = 0.01) -> dict:
    """Mount pose for an accessory dropped at world `point` with surface `normal` on robot geom `geom`."""
    t = TYPES.get(acc_type)
    if t is None:
        raise ValueError(f"accessory type must be one of {list(TYPES)}")
    if not 0 <= int(geom) < model.ngeom:
        raise ValueError("drop it on the robot")
    b = int(model.geom_bodyid[int(geom)])
    if b == 0 or model.body_rootid[b] != model.body_rootid[1]:
        raise ValueError("accessories mount on the robot, not on the world")
    p = np.asarray(point, float)
    n = np.asarray(normal, float)
    if p.shape != (3,) or n.shape != (3,) or not np.isfinite(p).all() or not np.isfinite(n).all():
        raise ValueError("point and normal must be three finite numbers each")
    n = n / max(np.linalg.norm(n), 1e-9)
    R_b = data.xmat[b].reshape(3, 3)
    local = R_b.T @ (p - data.xpos[b])
    if snap and snap > 0:
        local = np.round(local / snap) * snap
    n_local = R_b.T @ n
    local = local + n_local * t["half_depth"]  # sit on the surface, not inside it
    if acc_type == "lidar":
        # level in the world at this pose, wherever it is mounted
        R_world = np.eye(3)
    elif acc_type == "camera":
        z = -n  # the camera looks along its -z: straight out of the surface
        up = np.array([0.0, 0.0, 1.0])
        y = up - z * float(up @ z)
        if np.linalg.norm(y) < 1e-6:  # looking straight up/down: pick the robot's forward as "up" in the image
            y = R_b[:, 0] - z * float(R_b[:, 0] @ z)
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        R_world = np.stack([x, y, z], axis=1)
    else:
        # box face against the surface: local z along the outward normal
        z = n
        ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else R_b[:, 0]
        x = np.cross(ref, z)
        x /= max(np.linalg.norm(x), 1e-9)
        y = np.cross(z, x)
        R_world = np.stack([x, y, z], axis=1)
    quat = _quat_from_matrix(R_b.T @ R_world)
    out = {"type": acc_type, "body": model.body(b).name, "pos": [round(float(v), 4) for v in local],
           "quat": [round(v, 6) for v in quat], "mass": t["mass"]}
    if acc_type == "camera":
        out["fovy"] = t["fovy"]
    return out


def describe(model: mujoco.MjModel, data: mujoco.MjData, accessories: list[dict]) -> list[dict]:
    """Accessories with their current world position (for the GUI list)."""
    out = []
    for a in accessories:
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"acc_{a['id']}")
        world = data.geom_xpos[g].round(4).tolist() if g >= 0 else None
        out.append({**a, "label": TYPES.get(a["type"], {}).get("label", a["type"]), "geom": int(g), "world": world,
                    "mounted": g >= 0})
    return out
