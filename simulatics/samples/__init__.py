"""Ready-made samples: one call sets up the robot, the world and what the robot does.

Two kinds, and each sample says which it is:

    physics     the robot is driven by its controllers in the physics simulation (standing, walking,
                recovering from pushes, stairs). What you see is what the motors and contacts produce.
    reference   a human motion-capture clip, retargeted to the robot and played back kinematically (the pose
                is set every tick; physics does not hold it up). These are the acrobatic and parkour clips.
                They are reference motions to look at and to train a tracking policy on, not something the
                robot's controllers can do yet.

    from simulatics import samples
    samples.run(session, "backflip")

Reference clips come from the CMU Graphics Lab Motion Capture Database (see data/SOURCES.md and
THIRD_PARTY_NOTICES.md); `python -m simulatics.samples.build` rebuilds them from the original files.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DATA = Path(__file__).resolve().parent / "data"


@dataclass
class Sample:
    id: str
    title: str
    group: str                 # "Standing and walking" | "Worlds" | "Acrobatics" | "Parkour" | "Reference takes"
    description: str
    kind: str                  # "physics" | "reference"
    setup: Callable            # setup(session) -> message
    source: str = ""           # where a reference clip comes from


SAMPLES: dict[str, Sample] = {}


def _sample(id: str, title: str, group: str, description: str, kind: str = "physics", source: str = ""):
    def deco(fn):
        SAMPLES[id] = Sample(id, title, group, description, kind, fn, source)
        return fn
    return deco


def run(session, sample_id: str) -> str:
    """Set the live session up for a sample and start it. Returns a one-line description of what runs."""
    s = SAMPLES.get(sample_id)
    if s is None:
        raise ValueError(f"unknown sample {sample_id!r} (have: {', '.join(SAMPLES)})")
    _cancel_script(session)
    msg = s.setup(session)
    return f"{s.title} ({s.kind}): {msg}"


def listing() -> list[dict]:
    return [{"id": s.id, "title": s.title, "group": s.group, "description": s.description, "kind": s.kind,
             "source": s.source} for s in SAMPLES.values()]


# ------------------------------------------------------------------------------------------------ helpers
# Stronger motors for the acrobatic and parkour samples: landing from a metre or taking off for a flip needs more
# torque than the default robot has (about 245 Nm per knee to absorb a 1 m drop at 42 kg; the default is 180).
# These values are in the range of current full-size humanoids of similar mass. Applied visibly: the Robot
# panel shows them.
ATHLETIC = {"hip_torque": 300.0, "knee_torque": 360.0, "ankle_torque": 180.0, "abdomen_torque": 200.0,
            "arm_torque": 80.0}


def _builtin(session, obstacles: list[dict] | None = None, params: dict | None = None,
             props: list[dict] | None = None) -> None:
    """Built-in humanoid (default parameters plus `params`), standing at the origin in the given world, walker
    in charge."""
    from ..models import HumanoidParams
    with session.lock:
        session.running = False
        session.unload_policy()
        session.controller_kind = "auto"
        session.support_stand = False
        session.obstacles, session.props, session.cad_xml = list(obstacles or []), list(props or []), None
        session.set_params({**HumanoidParams().to_dict(), **(params or {})})      # rebuilds
        session.speed = 1.0


def _cancel_script(session) -> None:
    ev = getattr(session, "_sample_cancel", None)
    if ev is not None:
        ev.set()
    session._sample_cancel = threading.Event()


def _script(session, steps: list[tuple[float, Callable]]) -> None:
    """Run callables at given simulation times (seconds after now) on a helper thread; a new sample cancels it."""
    cancel, version = session._sample_cancel, session.scene_version
    t0 = float(session.data.time)

    def work():
        for at, fn in steps:
            while float(session.data.time) - t0 < at:
                if cancel.is_set() or session.scene_version != version or not session._alive:
                    return
                time.sleep(0.02)
            try:
                fn()
            except Exception:  # noqa: BLE001 - a sample script must never take the app down
                return
    threading.Thread(target=work, name="sample", daemon=True).start()


def _go(session) -> None:
    with session.lock:
        session.running = True


# --------------------------------------------------------------------------------- standing and walking
@_sample("stand", "Stand and balance", "Standing and walking",
         "The robot stands on its own, balancing with its ankles and hips.")
def stand(session) -> str:
    _builtin(session)
    _go(session)
    return "standing with the model-based balance controller"


@_sample("stand_pushes", "Stand through pushes", "Standing and walking",
         "The standing robot is shoved from the left, the front and the back (60 N each) and recovers.")
def stand_pushes(session) -> str:
    _builtin(session)
    _go(session)
    _script(session, [(1.5, lambda: session.push("left", 60, 0.1)), (4.5, lambda: session.push("forward", 60, 0.1)),
                      (7.5, lambda: session.push("back", 60, 0.1))])
    return "three 60 N pushes at 1.5 s, 4.5 s and 7.5 s"


@_sample("one_hand_wave", "Wave with one hand", "Standing and walking",
         "The robot keeps its balance while one arm waves (the arm follows a recorded clip through its motors).")
def one_hand_wave(session) -> str:
    _builtin(session)
    return session.play_motion("wave", "pd")


@_sample("walk_forward", "Walk forward", "Standing and walking",
         "Walks four metres straight ahead and stops.")
def walk_forward(session) -> str:
    _builtin(session)
    with session.lock:
        session.nav.set_goal([4.0, 0.0])
    _go(session)
    return "walking to (4, 0)"


WAYPOINTS = [(2.5, 0.0), (2.5, 2.5), (0.0, 2.5), (0.0, 0.0)]


@_sample("walk_points", "Walk from point to point", "Standing and walking",
         "Walks a square of four waypoints, stopping exactly on the last one.")
def walk_points(session) -> str:
    _builtin(session)
    with session.lock:
        for i, p in enumerate(WAYPOINTS):
            session.nav.set_goal(list(p), append=i > 0)
    _go(session)
    return "four waypoints: " + " -> ".join(f"({x:g}, {y:g})" for x, y in WAYPOINTS)


OBSTACLE_COURSE = [
    {"name": "red box", "shape": "box", "colour": "red", "rgba": [0.85, 0.25, 0.22, 1], "pos": [1.6, 0.3], "yaw": 0.3, "size": [0.35, 0.5, 0.5]},
    {"name": "blue box", "shape": "box", "colour": "blue", "rgba": [0.25, 0.45, 0.85, 1], "pos": [3.4, -0.9], "yaw": -0.2, "size": [0.5, 0.4, 0.6]},
    {"name": "green cylinder", "shape": "cylinder", "colour": "green", "rgba": [0.3, 0.7, 0.4, 1], "pos": [3.0, 1.5], "yaw": 0.0, "size": [0.35, 0.35, 0.7]},
]


@_sample("walk_around", "Walk around obstacles", "Standing and walking",
         "Plans a path around three obstacles to a goal five metres away and walks it.")
def walk_around(session) -> str:
    _builtin(session, OBSTACLE_COURSE)
    with session.lock:
        session.nav.set_goal([5.0, 0.0])
    _go(session)
    return "planning around three obstacles to (5, 0)"


@_sample("stairs", "Climb stairs", "Standing and walking",
         "Walks up and down a flight of stairs, placing each foot fully on a step.")
def stairs(session) -> str:
    _builtin(session)
    gx, gy = session.terrain_course("stairs", 0)
    with session.lock:
        session.nav.set_goal([gx, gy])
    _go(session)
    return f"stairs, goal ({gx:.1f}, {gy:.1f})"


# ------------------------------------------------------------------------------------------------- worlds
def _world(session, name: str, waypoints: bool = True) -> dict:
    from ..models import HumanoidParams
    with session.lock:
        session.running = False
        session.unload_policy()
        session.controller_kind = "auto"
        session.custom_path, session.cad_xml = None, None
        session.support_stand = False
        session.params = HumanoidParams()
        session.speed = 1.0
    info = session.load_world(name)
    if waypoints and info.get("waypoints"):
        with session.lock:
            for i, p in enumerate(info["waypoints"]):
                session.nav.set_goal(list(p), append=i > 0)
    return info


@_sample("playground", "Playground", "Worlds",
         "A yard of blocks, a ramp and platform, stepping blocks, slalom posts, a tunnel, balls to kick and crates "
         "to push. The robot walks a loop through it.")
def playground(session) -> str:
    info = _world(session, "playground")
    _go(session)
    return info["about"]


@_sample("parkour_course", "Parkour course", "Worlds",
         "Walks the centre lane of the course: random low steps, stairs up and down, and a ramp. Hurdles, jump "
         "boxes, a drop platform and a balance beam stand beside it.")
def parkour_course(session) -> str:
    info = _world(session, "parkour")
    _go(session)
    return info["about"]


@_sample("kitchen", "Kitchen", "Worlds",
         "A kitchen with a worktop (cup, apple and butter box within reach), hob, sink, cupboards, fridge, island "
         "and dining table. The robot walks a short tour and comes back to the worktop.")
def kitchen(session) -> str:
    info = _world(session, "kitchen")
    _go(session)
    return info["about"]


# ------------------------------------------------- motion capture: physical where verified, reference otherwise
def manifest() -> dict:
    p = DATA / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def physical() -> dict:
    """Clips that have been turned into physically simulated motions (samples/build_physical.py)."""
    p = DATA / "physical.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _reference(session, clip: dict) -> str:
    path = DATA / clip["file"]
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing: run  python -m simulatics.samples.build")
    _builtin(session, clip.get("props") or [])
    msg = session.play_motion(str(path), "preview")
    with session.lock:
        session.speed = float(clip.get("speed", 1.0))
    return msg


def _physical(session, entry: dict) -> str:
    path = DATA / entry["file"]
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing: run  python -m simulatics.samples.build_physical")
    _builtin(session, entry.get("props") or [], entry.get("params") or None)
    msg = session.play_tape(str(path))
    if entry.get("params"):
        msg += " with the athletic motor preset (" + ", ".join(f"{k.replace('_', ' ')} {v:g} Nm" for k, v in entry["params"].items()) + ")"
    return msg


def _register_clips() -> None:
    man, phys = manifest(), physical()
    done = {k: v for k, v in phys.items() if v.get("verified") and (DATA / v["file"]).exists()}
    for cid, e in done.items():
        clip = man.get(e.get("clip", cid), {})
        note = " Physically simulated: only motor commands are applied, found by sampling-based tracking of a "                "motion-capture take; open loop, so it reproduces only on this robot from its standing start."
        if e.get("params"):
            note += " Uses the athletic motor preset (stronger hips, knees, ankles and waist)."
        SAMPLES[cid] = Sample(cid, e["title"], clip.get("group", "Parkour"), e["description"] + note, "physics",
                              lambda session, e=e: _physical(session, e), e.get("source", ""))
    for cid, clip in man.items():
        rid = cid if cid not in done else f"{cid}_reference"
        title = clip["title"] if cid not in done else f"{clip['title']} (the reference take)"
        SAMPLES[rid] = Sample(rid, title, clip["group"] if cid not in done else "Reference takes", clip["description"],
                              "reference", lambda session, clip=clip: _reference(session, clip), clip.get("source", ""))


_register_clips()
