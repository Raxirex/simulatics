"""Projects: everything about a piece of work in one folder, saved and restored as a whole.

A project is a folder with `project.json` in it (plus, next to it, whatever the app adds: the node graph, the
lidar point cloud). The file is plain JSON and holds: which robot and its parameters, mounted accessories, the
scene (every object), motor edits, the controller, sensor settings, simulation speed, and the exact
simulation state (time, joint positions and velocities) at the moment of saving, so opening a project
continues where it was left.

    from simulatics import project
    project.save(session, "my_robot_lab", name="My robot lab")
    info = project.load(session, "my_robot_lab")

Paths to robot model files inside the project folder are stored relative to it, so a project folder can be
moved or shared. Nothing about the computer or the user is written into a project.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np

FILE = "project.json"
KIND = "simulatics-project"
VERSION = 1


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items() if not str(k).startswith("_")}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def folder_of(path: str | Path) -> Path:
    """The project folder for a path that is either the folder or its project.json."""
    p = Path(path).expanduser()
    return p.parent if p.name == FILE or p.suffix == ".json" else p


def is_project(path: str | Path) -> bool:
    p = Path(path)
    f = p / FILE if p.is_dir() else p
    if not f.is_file() or f.suffix != ".json":
        return False
    try:
        return json.loads(f.read_text(encoding="utf-8")).get("kind") == KIND
    except Exception:  # noqa: BLE001
        return False


def snapshot(session, name: str = "", folder: Path | None = None, notes: str = "") -> dict:
    """The session as plain data."""
    from . import __version__
    with session.lock:
        m, d = session.model, session.data
        if session.custom_path:
            path = Path(session.custom_path)
            library = session.robot_name if _is_library(session.robot_name) else None
            if library:
                robot = {"kind": "library", "name": library}
            else:
                rel = None
                if folder is not None:
                    try:
                        rel = path.resolve().relative_to(Path(folder).resolve()).as_posix()
                    except ValueError:
                        rel = None
                robot = {"kind": "file", "path": rel or str(path), "relative": rel is not None}
        else:
            robot = {"kind": "builtin"}
        robot.update(params=session.params.to_dict(), support_stand=bool(session.support_stand),
                     default_lidar=bool(session.default_lidar), accessories=session.accessories)
        return _jsonable({
            "kind": KIND, "version": VERSION, "app_version": __version__, "name": name or "Untitled project",
            "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "notes": notes,
            "robot": robot,
            "scene": session.scene_dict() if not session.custom_path else {"kind": "simulatics-scene", "version": 1, "obstacles": [], "props": []},
            "motors": session.motor_overrides,
            "controller": session.controller_kind if session.controller_kind not in ("motion-physical", "world-model") else "auto",
            "task": session.task_cfg.to_dict(), "lidar": session.lidar_cfg.to_dict(), "camera": dict(session.camera_cfg),
            "simulation": {"speed": float(session.speed), "auto_reset": bool(session.auto_reset)},
            "goals": [[float(g[0]), float(g[1])] for g, _ in session.nav.goals],
            "state": {"time": float(d.time), "qpos": d.qpos, "qvel": d.qvel, "ctrl": d.ctrl, "nq": int(m.nq), "nv": int(m.nv)},
        })


def _is_library(name: str | None) -> bool:
    from . import robots
    return bool(name) and name in robots.LIBRARY


def save(session, folder: str | Path, name: str | None = None, notes: str | None = None, extra: dict | None = None) -> Path:
    """Write the project to `folder` (created if needed). `extra` is merged in under "app" (the Studio stores its
    layout, camera and graph file name there). Returns the path of project.json."""
    folder = folder_of(folder)
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / FILE
    old = {}
    if f.exists():
        try:
            old = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            old = {}
    data = snapshot(session, name if name is not None else old.get("name", folder.name), folder,
                    notes if notes is not None else old.get("notes", ""))
    data["created"] = old.get("created") or data["saved"]
    data["app"] = {**old.get("app", {}), **(extra or {})}
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(f)
    return f


def read(path: str | Path) -> dict:
    p = Path(path).expanduser()
    f = p / FILE if p.is_dir() else p
    if not f.is_file():
        raise FileNotFoundError(f"no project at {p} (a project is a folder with {FILE} in it)")
    data = json.loads(f.read_text(encoding="utf-8"))
    if data.get("kind") != KIND:
        raise ValueError(f"{f.name} is not a Simulatics project")
    if int(data.get("version", 0)) > VERSION:
        raise ValueError("this project was saved by a newer version of Simulatics")
    return data


def load(session, path: str | Path, restore_state: bool = True) -> dict:
    """Make the session match a saved project. Returns the project data (with "folder" added)."""
    from .models import HumanoidParams
    from .sim.task import TaskConfig
    data = read(path)
    folder = folder_of(Path(path) if Path(path).is_dir() else Path(path))
    if Path(path).is_dir():
        folder = Path(path)
    robot = data.get("robot", {})
    with session.lock:
        session.running = False
        session.unload_policy()
        session.motor_overrides = {}
        session.task_cfg = TaskConfig.from_dict(data["task"]) if data.get("task") else session.task_cfg
        session.default_lidar = bool(robot.get("default_lidar", True))
        session.obstacles, session.props, session.cad_xml = [], [], None
        kind = robot.get("kind", "builtin")
        if kind == "builtin":
            session.custom_path, session.support_stand, session.robot_name = None, bool(robot.get("support_stand", False)), "builtin"
            session.accessories = robot.get("accessories")
            session.params = HumanoidParams.from_dict({**HumanoidParams().to_dict(), **robot.get("params", {})})
            session.build()
        elif kind == "library":
            session.load_robot(robot["name"])
            if bool(robot.get("support_stand", session.support_stand)) != session.support_stand:
                session.set_support_stand(bool(robot["support_stand"]))
        else:
            p = Path(robot["path"])
            p = (folder / p) if robot.get("relative") or not p.is_absolute() else p
            session.load_file(str(p), support_stand=bool(robot.get("support_stand", False)))
        if kind != "builtin" and robot.get("accessories"):
            session.accessories = robot["accessories"]
            session.build()
        scene = data.get("scene") or {}
        if kind == "builtin" and (scene.get("obstacles") or scene.get("props")):
            session.load_scene(scene)
            session.running = False
        session.motor_overrides = {str(k): {str(f): float(v) for f, v in fields.items()} for k, fields in (data.get("motors") or {}).items()}
        session._apply_motor_overrides()
        if data.get("lidar"):
            session.set_lidar(data["lidar"])
        cam = data.get("camera") or {}
        session.camera_cfg.update({k: cam[k] for k in ("name", "depth") if k in cam})
        sim = data.get("simulation") or {}
        session.speed = float(np.clip(sim.get("speed", 1.0), 0.05, 8.0))
        session.auto_reset = bool(sim.get("auto_reset", True))
        wanted = data.get("controller", "auto")
        if wanted != session.controller_kind:
            try:
                session.set_controller(wanted)
            except Exception:  # noqa: BLE001 - a controller this build does not have
                session.set_controller("auto")
        st = data.get("state") or {}
        m, d = session.model, session.data
        restored = False
        if restore_state and st and len(st.get("qpos", [])) == m.nq and len(st.get("qvel", [])) == m.nv:
            d.qpos[:] = st["qpos"]
            d.qvel[:] = st["qvel"]
            d.time = float(st.get("time", 0.0))
            if len(st.get("ctrl", [])) == m.nu:
                d.ctrl[:] = st["ctrl"]
            mujoco.mj_forward(m, d)
            c = session.controller
            if hasattr(c, "adopt"):
                c.adopt(m, d)                       # the walker takes the robot over as it was saved
                c.set_path(None)
            session.reset_slam()
            restored = True
        session.nav.clear()
        session.nav.take_new_path()
        session.scene_version += 1
    data["folder"] = str(folder)
    data["state_restored"] = restored
    return data


def create(folder: str | Path, name: str, session) -> Path:
    """Start a new project in `folder` from the session as it is now."""
    folder = Path(folder).expanduser()
    if (folder / FILE).exists():
        raise FileExistsError(f"{folder} already holds a project")
    return save(session, folder, name=name, notes="")
