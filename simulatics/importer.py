"""Import anything: give it a file (or a project folder) and it works out what it is and brings it in.

    from simulatics import importer
    importer.detect("motors.csv")                 -> "motors"
    importer.import_any(session, "motors.csv")    -> a one-line report of what was done

Kinds: project (folder or project.json), scene (.json), graph (.json), motors (.csv or .json motor values),
robot (.xml / .mjcf / .urdf), motion (.bvh motion capture, or a motion .npz), recording (.tape.npz: a recorded
physical motion or trained skill), policy (.npz), cloud (.ply / cloud .npz lidar map).

Motor values are a table with a `name` column and any of: kp, kv, torque, gear, damping, armature. A name
may be a motor name (`left_knee`), a pattern (`*_knee`, `left_*`) or `*` for all. CSV example:

    name,kp,kv,torque
    *_knee,350,20,220
    left_hip_y,300,18,180
"""
from __future__ import annotations

import csv
import fnmatch
import json
from pathlib import Path

import numpy as np

MOTOR_FIELDS = ("kp", "kv", "torque", "gear", "damping", "armature")
KINDS = {"project": "Project", "scene": "Scene", "graph": "Node graph", "motors": "Motor values", "robot": "Robot model",
         "motion": "Motion", "recording": "Recorded motion or trained skill", "policy": "Trained policy",
         "cloud": "Lidar point cloud"}


# ------------------------------------------------------------------------------------------- motor values
def motor_table(session) -> list[dict]:
    """Current motor values, one row per motor."""
    with session.lock:
        return [{"name": a["name"], **{f: float(a[f]) for f in MOTOR_FIELDS}} for a in session.actuators()]


def export_motors(session, path: str | Path) -> Path:
    """Write the motor values as .csv (or .json when the name ends in .json)."""
    path = Path(path)
    rows = motor_table(session)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps({"kind": "simulatics-motors", "version": 1, "motors": rows}, indent=1), encoding="utf-8")
    else:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["name", *MOTOR_FIELDS])
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    return path


def read_motor_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("motors", data) if isinstance(data, dict) else data
        if isinstance(rows, dict):                       # {"left_knee": {"kp": 300}, ...}
            rows = [{"name": k, **v} for k, v in rows.items()]
    else:
        text = path.read_text(encoding="utf-8-sig")
        dialect = csv.Sniffer().sniff(text.splitlines()[0] if text.strip() else "a,b", delimiters=",;\t")
        rows = list(csv.DictReader(text.splitlines(), dialect=dialect))
    out = []
    for r in rows:
        r = {str(k).strip().lower(): v for k, v in r.items() if k is not None}
        name = str(r.get("name") or r.get("motor") or r.get("joint") or r.get("actuator") or "").strip()
        if not name:
            continue
        vals = {}
        for f in MOTOR_FIELDS:
            v = r.get(f, r.get({"torque": "max_torque", "kv": "kd"}.get(f, f)))
            if v is None or str(v).strip() == "":
                continue
            try:
                vals[f] = float(v)
            except ValueError:
                raise ValueError(f"motor values: '{v}' in column {f} for {name} is not a number") from None
        if vals:
            out.append({"name": name, **vals})
    if not out:
        raise ValueError("no motor values found: the table needs a 'name' column and at least one of " + ", ".join(MOTOR_FIELDS))
    return out


def import_motors(session, path: str | Path) -> dict:
    """Apply motor values from a file. Returns {"applied": n values, "motors": n motors, "unknown": [names]}."""
    rows = read_motor_rows(path)
    with session.lock:
        names = [a["name"] for a in session.actuators()]
        applied, touched, unknown = 0, set(), []
        for r in rows:
            hits = [i for i, n in enumerate(names) if fnmatch.fnmatchcase(n, r["name"])]
            if not hits:
                unknown.append(r["name"])
                continue
            for i in hits:
                for f in MOTOR_FIELDS:
                    if f in r:
                        session.set_actuator(i, f, r[f])
                        applied += 1
                touched.add(i)
    return {"applied": applied, "motors": len(touched), "unknown": unknown}


# ---------------------------------------------------------------------------------------------- detection
def detect(path: str | Path) -> str:
    """What kind of thing a path is (a key of KINDS). Raises ValueError when it is nothing we can import."""
    from . import project
    p = Path(path).expanduser()
    if p.is_dir():
        if project.is_project(p):
            return "project"
        raise ValueError(f"{p.name} is a folder without a project.json in it")
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {p}")
    ext = p.suffix.lower()
    if ext == ".json":
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"{p.name} is not valid JSON: {e}") from None
        kind = data.get("kind") if isinstance(data, dict) else None
        if kind == "simulatics-project":
            return "project"
        if kind == "simulatics-scene":
            return "scene"
        if kind == "simulatics-motors" or (isinstance(data, dict) and "motors" in data):
            return "motors"
        if isinstance(data, dict) and "nodes" in data and "links" in data:
            return "graph"
        if isinstance(data, (list, dict)):
            try:
                read_motor_rows(p)
                return "motors"
            except Exception:  # noqa: BLE001
                pass
        raise ValueError(f"{p.name}: JSON, but not a project, scene, graph or motor table")
    if ext in (".csv", ".tsv"):
        read_motor_rows(p)
        return "motors"
    if ext in (".xml", ".mjcf", ".urdf"):
        return "robot"
    if ext == ".bvh":
        return "motion"
    if ext == ".ply":
        return "cloud"
    if ext == ".npz":
        with np.load(p, allow_pickle=False) as z:
            keys = set(z.files)
            header = str(z["header"]) if "header" in keys else ""
        if "simulatics-tape" in header:
            return "recording"
        if {"points", "voxel"} <= keys:
            return "cloud"
        from .imitate.motion import is_motion_file
        if is_motion_file(p):
            return "motion"
        return "policy"
    raise ValueError(f"do not know how to import {p.name} (supported: project folders, .json, .csv, .xml, .urdf, .bvh, .npz, .ply)")


def import_any(session, path: str | Path, kind: str | None = None) -> dict:
    """Bring a file in. Returns {"kind", "message", ...}. The Studio handles "graph" and extras of "project"
    itself (see studio/main_window.py import_path); everything else is done here."""
    from . import project
    p = Path(path).expanduser()
    kind = kind or detect(p)
    if kind == "project":
        data = project.load(session, p)
        return {"kind": kind, "message": f"Opened project '{data.get('name', p.name)}'", "project": data}
    if kind == "scene":
        scene = json.loads(p.read_text(encoding="utf-8"))
        session.load_scene(scene)
        return {"kind": kind, "message": f"Scene loaded: {len(scene.get('obstacles', []))} static and {len(scene.get('props', []))} loose objects"}
    if kind == "graph":
        return {"kind": kind, "message": f"Node graph {p.name}", "path": str(p)}
    if kind == "motors":
        rep = import_motors(session, p)
        msg = f"Motor values: {rep['applied']} values applied to {rep['motors']} motors"
        if rep["unknown"]:
            msg += f"; no motor matches {', '.join(rep['unknown'][:5])}" + (" ..." if len(rep["unknown"]) > 5 else "")
        return {"kind": kind, "message": msg, **rep}
    if kind == "robot":
        session.load_file(str(p))
        return {"kind": kind, "message": f"Robot model {p.name} loaded"}
    if kind == "motion":
        if p.suffix.lower() == ".bvh":
            from .imitate.retarget import retarget_bvh
            from .jobs import RUNS
            if session.custom_path:
                raise ValueError("motion capture is retargeted to the built-in humanoid; load it first")
            with session.lock:
                model = session.model
            res = retarget_bvh(str(p), model)
            out = RUNS / "motions" / f"{p.stem}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            res.motion.save(out)
            msg = session.play_motion(str(out), "preview")
            return {"kind": kind, "message": f"Motion capture {p.name} retargeted (mean error {res.errors.get('mean', 0) * 100:.1f} cm); {msg}",
                    "path": str(out)}
        return {"kind": kind, "message": session.play_motion(str(p), "preview")}
    if kind == "recording":
        from .imitate.physical import Tape
        tape = Tape.load(p)
        if tape.meta.get("skill"):
            from . import skills
            return {"kind": kind, "message": skills.play(session, tape)}
        return {"kind": kind, "message": session.play_tape(tape)}
    if kind == "policy":
        if p.name == "imitation_policy.npz":
            return {"kind": kind, "message": session.play_motion(None, "policy", str(p))}
        session.load_policy(str(p))
        return {"kind": kind, "message": f"Policy {p.parent.name} loaded"}
    if kind == "cloud":
        from .sim.cloud import PointCloudMap
        session.cloud = PointCloudMap.load(p)
        return {"kind": kind, "message": f"Lidar point cloud loaded: {len(session.cloud):,} points"}
    raise ValueError(f"unknown import kind {kind!r}")
