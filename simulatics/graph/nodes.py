"""Built-in node types for the node graph, the `RobotSpec` value that flows on "robot" wires, and EXAMPLES.

Robot / World nodes only edit a `RobotSpec` (pure data, cheap, cached). Nothing touches the live simulation
until a Simulate / Sensors / Policy node runs: those make the session match the spec they receive (only when
it differs from what the graph applied last) and then act on it. `sim.apply` always applies, which resets
the simulation.
"""
from __future__ import annotations

import copy
import io
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .core import Graph, GraphError, Port, jsonable, node, summarize
from ..models.humanoid import PARAM_SPEC, HumanoidParams

ACCESSORY_TYPES = ["lidar", "camera", "payload"]
FACINGS = {"forward": (1, 0, 0), "back": (-1, 0, 0), "left": (0, 1, 0), "right": (0, -1, 0), "up": (0, 0, 1),
           "down": (0, 0, -1)}
TERRAIN_KINDS = ["mixed", "random_steps", "stairs", "slope"]
LIBRARY_NAMES = ["adam_lite", "go2"]
_POLICY_FILES = ("policy.npz", "rl_policy.npz", "imitation_policy.npz")


# =========================================================================================================
# The value on "robot" wires
# =========================================================================================================
@dataclass
class RobotSpec:
    """Which robot, with what mounted on it, in which world, driven by which policy. Pure data."""
    kind: str = "builtin"                       # "builtin" | "library" | "file"
    params: dict = field(default_factory=dict)  # HumanoidParams overrides (built-in robot)
    library: str | None = None                  # library robot name (kind "library")
    path: str | None = None                     # MJCF / URDF file (kind "file")
    support_stand: bool = False                 # imported robots: hold the base on a stand
    default_lidar: bool = True                  # built-in robot: keep the head lidar
    accessories: list | None = None             # [{"type", "body", "pos", "quat", "mass"?, "fovy"?}] or None
    world: dict | None = None                   # {"obstacles": n, "seed"} | {"terrain": kind, "seed"} | {"manipulation": True, "seed"}
    policy: dict | None = None                  # {"name", "path", "kind"} loaded when the spec is applied

    # ---- data ------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"kind": self.kind, "params": dict(self.params), "library": self.library, "path": self.path,
                "support_stand": bool(self.support_stand), "default_lidar": bool(self.default_lidar),
                "accessories": copy.deepcopy(self.accessories), "world": copy.deepcopy(self.world),
                "policy": copy.deepcopy(self.policy)}

    @classmethod
    def from_dict(cls, d: dict) -> "RobotSpec":
        return cls(**{k: copy.deepcopy(v) for k, v in d.items() if k in cls.__dataclass_fields__})

    def copy(self, **changes) -> "RobotSpec":
        """A deep copy with some fields replaced (wire values are shared and cached: never edit one in place)."""
        return RobotSpec.from_dict({**self.to_dict(), **changes})

    def key(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=repr)

    def __hash__(self) -> int:
        return hash(self.key())

    def summary(self) -> str:
        name = {"builtin": "built-in humanoid", "library": self.library or "library robot",
                "file": Path(self.path or "?").name}[self.kind]
        bits = [name]
        if self.kind == "builtin" and self.params:
            bits.append(f"{len(self.params)} param(s) changed")
        if self.accessories:
            bits.append("+ " + ", ".join(a["type"] for a in self.accessories))
        if self.world:
            bits.append("world: " + next(k for k in self.world if k != "seed"))
        if self.policy:
            bits.append(f"policy {self.policy.get('name')}")
        return ", ".join(bits)

    # ---- headless model --------------------------------------------------
    def humanoid_params(self) -> HumanoidParams:
        d = {**HumanoidParams().to_dict(), **self.params}
        if self.world and self.world.get("manipulation"):
            d["arms"] = d["hands"] = True   # the table scene needs hands (as the GUI does)
        return HumanoidParams.from_dict(d)

    def file_path(self) -> str | None:
        """The model file of an imported robot (None for the built-in one). Never downloads."""
        if self.kind == "builtin":
            return None
        if self.kind == "library":
            from .. import robots
            if self.library not in robots.LIBRARY:
                raise GraphError(f"unknown library robot {self.library!r} (choose from {list(robots.LIBRARY)})")
            if not robots.available(self.library):
                raise GraphError(f"robot '{self.library}' is not downloaded: run "
                                 f"`python -m simulatics.robots fetch {self.library}` (or load it once in the GUI)")
            return str(robots.path(self.library).resolve())
        p = Path(str(self.path or "")).expanduser()
        if not p.is_file():
            raise GraphError(f"no such robot file: {p}")
        return str(p)

    def _accessory_dicts(self, root_body: str, defaults: list[dict]) -> list[dict]:
        """Session-style accessory dicts (ids g1, g2, ...). A lidar in the spec replaces any other lidar."""
        from ..models.accessories import TYPES
        mine = []
        for i, a in enumerate(self.accessories or []):
            t = TYPES.get(a.get("type"))
            if t is None:
                raise GraphError(f"accessory type must be one of {list(TYPES)}, not {a.get('type')!r}")
            d = {"id": f"g{i + 1}", "type": a["type"], "body": a.get("body") or root_body,
                 "pos": [float(v) for v in a.get("pos", (0, 0, 0))],
                 "quat": [float(v) for v in a.get("quat", (1, 0, 0, 0))],
                 "mass": float(a.get("mass") or t["mass"])}
            if a["type"] == "camera":
                d["fovy"] = float(a.get("fovy") or t["fovy"])
            mine.append(d)
        lidars = [a for a in mine if a["type"] == "lidar"]
        if lidars:
            mine = [a for a in mine if a["type"] != "lidar" or a is lidars[-1]]
            defaults = [a for a in defaults if a["type"] != "lidar"]
        return list(defaults) + mine

    def compile(self):
        """A compiled MuJoCo model of the robot alone (with its accessories, without the world)."""
        import mujoco
        from ..models.accessories import apply_accessories, apply_to_spec, default_accessories
        if self.kind == "builtin":
            from ..models import build_mjcf
            xml = build_mjcf(self.humanoid_params())
            acc = self._accessory_dicts("torso", default_accessories(xml) if self.default_lidar else [])
            xml, warnings = apply_accessories(xml, acc)
            model = mujoco.MjModel.from_xml_string(xml)
        else:
            path = self.file_path()
            model = mujoco.MjSpec.from_file(path).compile()
            warnings = []
            if self.accessories:
                spec = mujoco.MjSpec.from_file(path)
                warnings = apply_to_spec(spec, self._accessory_dicts(model.body(1).name, []))
                model = spec.compile()
        if warnings:
            raise GraphError("; ".join(warnings))
        return model

    # ---- live session ----------------------------------------------------
    def apply(self, session, log=None) -> None:
        """Make the live SimSession match this spec (rebuilds the model, resets and pauses the simulation)."""
        from ..models import build_mjcf
        from ..models.accessories import default_accessories
        s, notes = session, []
        stop_pilot(s)
        with s.lock:
            s.running = False
            if self.kind == "builtin":
                p = self.humanoid_params()
                acc = self._accessory_dicts("torso", default_accessories(build_mjcf(p)) if self.default_lidar else [])
                if s.custom_path:
                    s.custom_path, s.support_stand, s.robot_name = None, False, "builtin"
                s.default_lidar, s.accessories = bool(self.default_lidar), acc
                s.set_params(p.to_dict())
                w = self.world or {}
                if "obstacles" in w:
                    s.random_obstacles(int(w["obstacles"]), w.get("seed"))
                elif "terrain" in w:
                    s.terrain_course(str(w["terrain"]), w.get("seed"))
                elif w.get("manipulation"):
                    s.manipulation_scene(w.get("seed"))
                elif s.obstacles or s.props:
                    s.clear_obstacles()
            elif self.kind in ("library", "file"):
                s.load_file(self.file_path(), support_stand=bool(self.support_stand),
                            robot_name=self.library if self.kind == "library" else None)
                if self.accessories:
                    s.accessories = self._accessory_dicts(s.model.body(1).name, [])
                    s.build()
                if self.world:
                    notes.append("the world (obstacles / terrain / table) only exists around the built-in robot: ignored")
            else:
                raise GraphError(f"unknown robot kind {self.kind!r}")
            if s.accessory_warnings:
                raise GraphError("; ".join(s.accessory_warnings))
            pol = self.policy
            if pol:
                path = str(Path(pol["path"]).resolve())
                if not Path(path).is_file():
                    raise GraphError(f"policy file not found: {path}")
                if Path(path).name == "imitation_policy.npz":
                    s.play_motion(None, "policy", path)
                    s.policy_name = pol.get("name") or Path(path).parent.name
                else:
                    s.load_policy(path)
                s.running = False
            elif (s.policy is not None or s.rl_policy is not None
                  or s.controller_kind in ("es-policy", "rl", "imitation", "motion-preview", "motion-pd")):
                s.unload_policy()
            s.falls = 0
            s.grid.reset_lidar()
            s._graph_applied = (self.key(), s.scene_version)
            notes.append(f"{s.robot_name}: {s.model.nbody - 1} bodies, {s.model.nu} actuators, controller "
                         f"{getattr(s.controller, 'name', 'none')}" + (f" ({s.policy_name})" if s.policy_name else ""))
        for m in notes:   # listeners run outside the lock
            if log is not None:
                log(m)


def stop_pilot(session) -> None:
    """Stop a running AI pilot. Call without the session lock: the pilot thread needs it to wind down."""
    pilot = getattr(session, "pilot", None)
    if pilot is not None and pilot.running:
        pilot.stop(timeout=1.0)


def _robot(robot) -> RobotSpec:
    if not isinstance(robot, RobotSpec):
        raise GraphError(f"expected a robot, got {summarize(robot)}")
    return robot


def ensure_applied(ctx, robot, force: bool = False):
    """The session, showing `robot`: applies the spec unless the graph applied this same spec last and the
    scene has not been rebuilt since (by Studio or another node)."""
    robot = _robot(robot)
    s = ctx.need_session()
    with s.lock:
        stale = force or getattr(s, "_graph_applied", None) != (robot.key(), s.scene_version)
    if stale:
        robot.apply(s, ctx.log)
    return s


# =========================================================================================================
# Simulation helpers
# =========================================================================================================
def _root_state(s):
    """(x, y, z, horizontal speed) of the robot's root. Call with the session lock held."""
    d = s.data
    if s.task is not None:
        return float(d.qpos[0]), float(d.qpos[1]), float(d.qpos[2]), float(np.hypot(d.qvel[0], d.qvel[1]))
    if s.model.nbody > 1:
        p = d.xpos[1]
        return float(p[0]), float(p[1]), float(p[2]), 0.0
    return 0.0, 0.0, 0.0, 0.0


def run_sim(ctx, s, seconds: float, speed: float, sample=None, message: str = "simulating") -> float:
    """Run the session for `seconds` of simulation time at `speed` x real time; returns the simulated time.

    `sample(elapsed, jumped)` is called with the session lock held about every 0.05 s of simulation time;
    returning True ends the run early. Simulation time restarts from zero whenever the robot is reset (a
    fall), so elapsed time is accumulated from positive steps only. Never hangs: gives up after a wall-clock
    limit, when the simulation thread reports an error, or when something else pauses the simulation.
    The session's speed and running state are restored afterwards.
    """
    seconds, speed = max(0.0, float(seconds)), float(np.clip(speed, 0.05, 64.0))
    wall_end = time.monotonic() + seconds * 10.0 / min(speed, 1.0) + 30.0
    nap = float(np.clip(0.05 / speed, 0.002, 0.02))
    with s.lock:
        prev_speed, prev_running = s.speed, s.running
        last_t = float(s.data.time)
        s.speed, s.running = speed, True
    elapsed, t_report = 0.0, 0.0
    try:
        while elapsed < seconds:
            ctx.check_cancel()
            time.sleep(nap)
            with s.lock:
                t = float(s.data.time)
                dt, last_t = t - last_t, t
                if dt > 0:
                    elapsed += dt
                stop = bool(sample(elapsed, dt < 0)) if sample is not None else False
                running, err = s.running, s.error
            if stop:
                break
            if not running:
                if err:
                    raise RuntimeError(f"the simulation stopped: {err}")
                ctx.log("the simulation was paused from outside: stopping early")
                break
            now = time.monotonic()
            if now > wall_end:
                raise TimeoutError(f"the simulation only advanced {elapsed:.1f} of {seconds:.1f} s in time")
            if now - t_report > 0.25:
                t_report = now
                ctx.progress(elapsed / seconds if seconds else 1.0, f"{message} {elapsed:.1f} / {seconds:.1f} s")
    finally:
        with s.lock:
            s.speed, s.running = prev_speed, prev_running
    return elapsed


def _series(title, x, y: dict, xlabel="", ylabel="") -> dict:
    return {"title": title, "x": [float(v) for v in x], "y": {k: [None if v is None else float(v) for v in vs]
                                                               for k, vs in y.items()},
            "xlabel": xlabel, "ylabel": ylabel}


# =========================================================================================================
# Robot
# =========================================================================================================
_BUILTIN_FIELDS = ["thigh_length", "shin_length", "torso_length", "pelvis_width", "foot_length", "density",
                   "hip_kp", "knee_kp"]


def _spec_port(name: str) -> Port:
    lo, hi, dv, label, _ = PARAM_SPEC[name]
    return Port(name, "number", dv, min=lo, max=hi, doc=label)


@node("robot.builtin", "Built-in humanoid", "Robot", outputs=[Port("robot", "robot")],
      params=[*[_spec_port(n) for n in _BUILTIN_FIELDS],
              Port("arms", "bool", True, doc="Arms"), Port("hands", "bool", True, doc="Wrists and grippers (needs arms)"),
              Port("default_lidar", "bool", True, doc="Keep the lidar on top of the head")],
      doc="The parametric humanoid with the given body and motor parameters.")
def robot_builtin(ctx, default_lidar, **params):
    base = HumanoidParams().to_dict()
    changed = {k: v for k, v in params.items() if v != base[k]}
    HumanoidParams.from_dict({**base, **changed})
    return {"robot": RobotSpec(kind="builtin", params=changed, default_lidar=bool(default_lidar))}


@node("robot.library", "Library robot", "Robot", outputs=[Port("robot", "robot")],
      params=[Port("name", "choice", "go2", choices=LIBRARY_NAMES, doc="Robot model"),
              Port("support_stand", "choice", "auto", choices=["auto", "on", "off"],
                   doc="Hold the base on a stand (auto: humanoids yes, quadrupeds no)")],
      doc="A downloaded third-party robot (Adam Lite humanoid, Unitree Go2 dog).")
def robot_library(ctx, name, support_stand):
    from .. import robots
    info = robots.LIBRARY.get(name)
    if info is None:
        raise GraphError(f"unknown library robot {name!r} (choose from {list(robots.LIBRARY)})")
    stand = info["kind"] == "humanoid" if support_stand == "auto" else support_stand == "on"
    spec = RobotSpec(kind="library", library=name, support_stand=stand, default_lidar=False)
    spec.file_path()  # a clear error now if it is not downloaded
    return {"robot": spec}


@node("robot.file", "Robot from file", "Robot", outputs=[Port("robot", "robot")],
      params=[Port("path", "text", "", doc="MJCF (.xml) or URDF file"),
              Port("support_stand", "bool", False, doc="Hold the base on a stand")],
      doc="A robot imported from an MJCF or URDF file.")
def robot_file(ctx, path, support_stand):
    spec = RobotSpec(kind="file", path=str(Path(path.strip().strip('"')).expanduser()) if path.strip() else "",
                     support_stand=bool(support_stand), default_lidar=False)
    if not path.strip():
        raise GraphError("give the path of an MJCF or URDF file")
    spec.file_path()
    return {"robot": spec}


@node("robot.set_param", "Set parameter", "Robot", inputs=[Port("robot", "robot")], outputs=[Port("robot", "robot")],
      params=[Port("name", "text", "thigh_length", doc="A HumanoidParams field, e.g. shin_length, knee_kp, friction"),
              Port("value", "number", 0.4)],
      doc="Change one body / motor / physics parameter of the built-in humanoid.")
def robot_set_param(ctx, robot, name, value):
    robot, name = _robot(robot), name.strip()
    if robot.kind != "builtin":
        raise GraphError("parameters can only be changed on the built-in humanoid")
    base = HumanoidParams().to_dict()
    if name not in base:
        raise GraphError(f"no parameter {name!r}; choose from: {', '.join(base)}")
    if name in PARAM_SPEC:
        lo, hi = PARAM_SPEC[name][:2]
        v = float(min(hi, max(lo, value)))
        if v != value:
            ctx.log(f"{name} limited to {v:g} (allowed {lo:g} .. {hi:g})")
    else:
        v = bool(value)
    return {"robot": robot.copy(params={**robot.params, name: v})}


@node("robot.accessory", "Mount accessory", "Robot", inputs=[Port("robot", "robot")], outputs=[Port("robot", "robot")],
      params=[Port("type", "choice", "lidar", choices=ACCESSORY_TYPES),
              Port("body", "text", "", doc="Body to mount on (empty: the robot's root body)"),
              Port("x", "number", 0.0, doc="Position in the body frame (m)"), Port("y", "number", 0.0),
              Port("z", "number", 0.1),
              Port("facing", "choice", "forward", choices=list(FACINGS), doc="Camera: viewing direction in the body frame"),
              Port("mass", "number", 0.0, min=0.0, max=30.0, doc="kg (0: the type's default)")],
      doc="Mount a lidar, camera or payload on a body of the robot (one lidar per robot: a new one replaces it).")
def robot_accessory(ctx, robot, type, body, x, y, z, facing, mass):
    import mujoco
    robot = _robot(robot)
    pos = [float(x), float(y), float(z)]
    if not all(math.isfinite(v) for v in pos) or max(abs(v) for v in pos) > 2.0:
        raise GraphError("the position must be within 2 m of the body")
    acc = {"type": type, "body": body.strip(), "pos": pos, "quat": [1.0, 0.0, 0.0, 0.0]}
    if type == "camera":
        zc = -np.array(FACINGS[facing], float)       # a camera looks along its -z
        up = np.array([0.0, 0.0, 1.0]) if abs(zc[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        yc = up - zc * float(up @ zc)
        yc /= np.linalg.norm(yc)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.ascontiguousarray(np.stack([np.cross(yc, zc), yc, zc], axis=1)).reshape(-1))
        acc["quat"] = [round(float(v), 6) for v in q]
    if mass > 0:
        acc["mass"] = float(mass)
    return {"robot": robot.copy(accessories=[*(robot.accessories or []), acc])}


# =========================================================================================================
# World
# =========================================================================================================
def _builtin_world(robot, world: dict) -> RobotSpec:
    robot = _robot(robot)
    if robot.kind != "builtin":
        raise GraphError("obstacles, terrain and the table scene only exist around the built-in humanoid")
    return robot.copy(world=world)


@node("world.obstacles", "Random obstacles", "World", inputs=[Port("robot", "robot")], outputs=[Port("robot", "robot")],
      params=[Port("n", "int", 12, min=1, max=60, doc="Number of boxes and cylinders"), Port("seed", "int", 0, min=0)],
      doc="Scatter random boxes and cylinders on the floor around the robot.")
def world_obstacles(ctx, robot, n, seed):
    return {"robot": _builtin_world(robot, {"obstacles": int(n), "seed": int(seed)})}


@node("world.terrain", "Terrain course", "World", inputs=[Port("robot", "robot")],
      outputs=[Port("robot", "robot"), Port("goal_x", "number"), Port("goal_y", "number")],
      params=[Port("kind", "choice", "mixed", choices=TERRAIN_KINDS), Port("seed", "int", 0, min=0)],
      doc="Put a terrain course (steps, stairs, slope) in front of the robot; also gives the goal just beyond it.")
def world_terrain(ctx, robot, kind, seed):
    from ..sim.terrain import terrain_course
    _, goal = terrain_course(kind, int(seed))
    return {"robot": _builtin_world(robot, {"terrain": kind, "seed": int(seed)}),
            "goal_x": float(goal[0]), "goal_y": float(goal[1])}


@node("world.manipulation", "Table with objects", "World", inputs=[Port("robot", "robot")],
      outputs=[Port("robot", "robot")], params=[Port("seed", "int", 0, min=0)],
      doc="A table in front of the robot with small graspable objects (switches arms and hands on).")
def world_manipulation(ctx, robot, seed):
    return {"robot": _builtin_world(robot, {"manipulation": True, "seed": int(seed)})}


@node("world.clear", "Empty world", "World", inputs=[Port("robot", "robot")], outputs=[Port("robot", "robot")],
      doc="Remove obstacles, terrain and props: just the robot on a flat floor.")
def world_clear(ctx, robot):
    return {"robot": _robot(robot).copy(world=None)}


# =========================================================================================================
# Simulate
# =========================================================================================================
@node("sim.apply", "Load into simulation", "Simulate", inputs=[Port("robot", "robot")], outputs=[Port("robot", "robot")],
      volatile=True, doc="Build this robot and world in the live simulation (resets and pauses it).")
def sim_apply(ctx, robot):
    ensure_applied(ctx, robot, force=True)
    return {"robot": robot}


@node("sim.run", "Run simulation", "Simulate", inputs=[Port("robot", "robot")],
      outputs=[Port("series", "series"), Port("falls", "number"), Port("distance", "number"), Port("robot", "robot")],
      params=[Port("seconds", "number", 5.0, min=0.0, max=3600.0, doc="Simulation time to run"),
              Port("speed", "number", 4.0, min=0.05, max=64.0, doc="Times real time (as fast as the CPU allows)")],
      volatile=True, doc="Run the simulation for some seconds; reports root height and speed, falls and distance travelled.")
def sim_run(ctx, robot, seconds, speed):
    s = ensure_applied(ctx, robot)
    ts, hs, vs = [], [], []
    state = {"dist": 0.0, "last": None, "next": 0.0}
    with s.lock:
        falls0 = s.falls

    def sample(elapsed, jumped):
        x, y, z, v = _root_state(s)
        if state["last"] is not None and not jumped:
            state["dist"] += math.hypot(x - state["last"][0], y - state["last"][1])
        state["last"] = (x, y)
        if elapsed >= state["next"] or jumped:
            state["next"] = elapsed + 0.02
            ts.append(elapsed), hs.append(z), vs.append(v)

    with s.lock:
        sample(0.0, False)
    run_sim(ctx, s, seconds, speed, sample)
    with s.lock:
        falls = s.falls - falls0
    return {"series": _series("Simulation", ts, {"root height (m)": hs, "speed (m/s)": vs}, "time (s)"),
            "falls": float(falls), "distance": float(state["dist"]), "robot": robot}


@node("sim.walk_to", "Walk to goal", "Simulate",
      inputs=[Port("robot", "robot"), Port("goal_x", "number", optional=True, doc="Overrides x when connected"),
              Port("goal_y", "number", optional=True, doc="Overrides y when connected")],
      outputs=[Port("reached", "bool"), Port("seconds", "number"), Port("final_error", "number"),
               Port("path", "series"), Port("robot", "robot")],
      params=[Port("x", "number", 2.0, doc="Goal (m)"), Port("y", "number", 0.0),
              Port("timeout", "number", 60.0, min=1.0, max=3600.0, doc="Simulation seconds to allow"),
              Port("speed", "number", 4.0, min=0.05, max=64.0, doc="Times real time")],
      volatile=True, doc="Plan a path to a goal and walk there; reports whether it arrived, the time and the path walked.")
def sim_walk_to(ctx, robot, goal_x, goal_y, x, y, timeout, speed):
    s = ensure_applied(ctx, robot)
    gx, gy = float(x if goal_x is None else goal_x), float(y if goal_y is None else goal_y)
    xs, ys = [], []
    with s.lock:
        if s.task is None:
            raise GraphError("this robot has no free-floating base: it cannot walk")
        s.nav.set_goal([gx, gy], None)
        x0, y0, _, _ = _root_state(s)
        xs.append(x0), ys.append(y0)
    state = {"nav": "walking"}

    def sample(elapsed, jumped):
        px, py, _, _ = _root_state(s)
        if math.hypot(px - xs[-1], py - ys[-1]) > 0.02:
            xs.append(px), ys.append(py)
        state["nav"] = s.nav.state
        return state["nav"] != "walking"

    try:
        elapsed = run_sim(ctx, s, timeout, speed, sample, message=f"walking to ({gx:.2f}, {gy:.2f})")
    except BaseException:
        with s.lock:
            s.nav.clear()
        raise
    with s.lock:
        px, py, _, _ = _root_state(s)
        msg = s.nav.message
        if s.nav.state == "walking":   # timed out: stop walking
            s.nav.clear()
    reached = state["nav"] == "reached"
    ctx.log(f"{state['nav']}{': ' + msg if msg else ''}")
    return {"reached": reached, "seconds": float(elapsed), "final_error": float(math.hypot(px - gx, py - gy)),
            "path": _series("Path walked", xs, {"path": ys}, "x (m)", "y (m)"), "robot": robot}


@node("sim.push_test", "Push test", "Simulate", inputs=[Port("robot", "robot")],
      outputs=[Port("survived", "bool"), Port("displacement", "number"), Port("robot", "robot")],
      params=[Port("direction", "choice", "left", choices=["forward", "back", "left", "right"]),
              Port("force", "number", 60.0, min=0.0, max=1000.0, doc="N"),
              Port("duration", "number", 0.1, min=0.01, max=1.0, doc="s"),
              Port("watch", "number", 3.0, min=0.2, max=60.0, doc="Simulation seconds to watch after the push"),
              Port("speed", "number", 4.0, min=0.05, max=64.0, doc="Times real time")],
      volatile=True, doc="Shove the standing robot's torso and see whether it stays on its feet.")
def sim_push_test(ctx, robot, direction, force, duration, watch, speed):
    s = ensure_applied(ctx, robot)
    with s.lock:
        if s.task is None:
            raise GraphError("this robot has no free-floating base to push")
    run_sim(ctx, s, 0.5, speed, message="settling")
    with s.lock:
        falls0 = s.falls
        x0, y0, _, _ = _root_state(s)
        note = s.push(direction, force, duration)
        s.running = False
    ctx.log(note)
    run_sim(ctx, s, duration + watch, speed, message="watching")
    with s.lock:
        fell = s.falls > falls0
        x1, y1, _, _ = _root_state(s)
    upright = bool(s.robot.pose()["upright"])
    return {"survived": bool(not fell and upright), "displacement": float(math.hypot(x1 - x0, y1 - y0)),
            "robot": robot}


# =========================================================================================================
# Sensors
# =========================================================================================================
@node("sensor.lidar", "Lidar scan", "Sensors", inputs=[Port("robot", "robot")],
      outputs=[Port("points", "points"), Port("hits", "number"), Port("robot", "robot")], volatile=True,
      doc="One scan of the robot's lidar: the points it hit, in world coordinates.")
def sensor_lidar(ctx, robot):
    import mujoco
    s = ensure_applied(ctx, robot)
    with s.lock:
        if not s.lidar.mounted:
            raise GraphError("this robot has no lidar: add a 'Mount accessory' node with type lidar")
        mujoco.mj_forward(s.model, s.data)
        _, pts, _, hit = s.lidar.scan(s.model, s.data)
        pts = np.asarray(pts[hit], np.float32).copy()
    return {"points": pts, "hits": float(len(pts)), "robot": robot}


@node("sensor.camera", "Camera image", "Sensors", inputs=[Port("robot", "robot")],
      outputs=[Port("image", "image"), Port("robot", "robot")],
      params=[Port("camera", "text", "head_cam", doc="Camera name (empty: the free overview camera)"),
              Port("depth", "bool", False, doc="Depth image (near = bright, 8 m range, always 320x240)"),
              Port("width", "int", 480, min=32, max=1920), Port("height", "int", 360, min=32, max=1080)],
      volatile=True, doc="A picture from one of the robot's cameras (colour or depth).")
def sensor_camera(ctx, robot, camera, depth, width, height):
    from PIL import Image
    s = ensure_applied(ctx, robot)
    camera = camera.strip()
    names = s.robot.cameras()
    if camera and camera not in names:
        raise GraphError(f"no camera {camera!r} on this robot; it has: {', '.join(names) or 'none (leave the name empty)'}")
    # GL contexts belong to the simulation thread: both routes below ask it to render (never hold the lock here)
    if not depth:
        jpeg = s.robot.camera_jpeg(camera or "__free__", int(width), int(height))
    else:
        with s.lock:
            old, seq = dict(s.camera_cfg), s.camera_seq
            s.camera_error = ""
            s.camera_cfg.update(enabled=True, name=camera or "__free__", depth=True)
        try:
            end = time.monotonic() + 8.0
            while True:
                ctx.check_cancel()
                time.sleep(0.02)
                with s.lock:
                    if s.camera_error:
                        raise RuntimeError(f"camera: {s.camera_error}")
                    if s.camera_seq != seq and s.camera_jpeg:
                        jpeg = s.camera_jpeg
                        break
                if time.monotonic() > end:
                    raise TimeoutError("no camera frame arrived (is the simulation thread alive?)")
        finally:
            with s.lock:
                s.camera_cfg.update(old)
    img = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"), np.uint8).copy()
    return {"image": img, "robot": robot}


# =========================================================================================================
# Train
# =========================================================================================================
def _wait_job(ctx, job, poll: float = 0.5, label: str = "generation"):
    """Poll a started background job (start / stop / status) until it ends; stops it when the run is cancelled."""
    try:
        while True:
            st = job.status()
            state = st.get("state")
            if state in ("finished", "stopped", "error"):
                return st
            g, n = int(st.get("generation") or 0), max(1, int(st.get("generations") or 1))
            best = st.get("best")
            ctx.progress(g / n, f"{label} {g} / {n}" + (f", best {best:.3f}" if best is not None else ""),
                         history=st.get("history") or [], generation=g, generations=n, best=best)
            ctx.sleep(poll)
    except BaseException:
        job.stop()
        end = time.monotonic() + 300.0   # distillation only stops between rounds
        while job.state == "running" and time.monotonic() < end:
            time.sleep(0.1)
        raise


def _curve(title, history, ylabel="score") -> dict:
    hist = [h for h in history if h.get("best") is not None]
    return _series(title, [h.get("gen", i + 1) for i, h in enumerate(hist)],
                   {"best": [h["best"] for h in hist], "mean": [h.get("mean") for h in hist]}, "generation", ylabel)


@node("train.run", "Train policy", "Train",
      inputs=[Port("robot", "robot"), Port("init", "policy", optional=True, doc="Policy to continue from (optional)")],
      outputs=[Port("policy", "policy"), Port("series", "series"), Port("best", "number")],
      params=[Port("algo", "choice", "es", choices=["ppo", "distill", "imitation", "es", "ga"], doc="Training method"),
              Port("name", "text", "graph_run", doc="Run name: results go to runs/<name>"),
              Port("population", "int", 16, min=1, max=4096, doc="Candidates (es / ga) or parallel robots (ppo)"),
              Port("generations", "int", 3, min=1, max=1000000, doc="Generations / iterations / rounds"),
              Port("lr", "number", 0.005, min=1e-6, max=0.1, doc="Learning rate"),
              Port("threads", "int", 0, min=0, max=256, doc="CPU threads (0: the method's default)"),
              Port("episode_seconds", "number", 2.0, min=0.5, max=120.0, doc="Length of each trial"),
              Port("motion", "text", "walk", doc="imitation only: the motion clip to track")],
      doc="Train a control policy for the robot (evolution, PPO, distillation or motion imitation); cached by its settings.")
def train_run(ctx, robot, init, algo, name, population, generations, lr, threads, episode_seconds, motion):
    from ..jobs import make_trainer
    from ..sim.task import TaskConfig
    robot = _robot(robot)
    d = {"algo": algo, "name": name, "population": population, "generations": generations, "lr": lr,
         "threads": threads, "episode_seconds": episode_seconds, "motion": motion}
    custom = robot.file_path()
    params = robot.humanoid_params()
    if custom is not None:
        from ..sim.task import has_free_root
        m = robot.compile()
        if not has_free_root(m) or m.nu == 0:
            raise GraphError("this model has no free-floating root or no actuators, cannot train locomotion")
    if robot.accessories or robot.world:
        ctx.log("training uses the bare robot on flat ground (accessories and world objects are left out)")
    init_policy, continue_run, export_xml = None, None, None
    if init:
        fname = Path(init["path"]).name
        if algo in ("es", "ga") and fname == "policy.npz":
            from ..train.policy import Policy
            init_policy = Policy.load(init["path"])
        elif algo == "ppo" and fname == "rl_policy.npz":
            continue_run = Path(init["path"]).parent.name
        else:
            ctx.log(f"policy '{init.get('name')}' ({init.get('kind')}) cannot warm-start {algo}: starting fresh")
    if algo == "distill":
        if custom is not None:
            raise GraphError("distillation needs the built-in humanoid (its model-based walker is the teacher)")
        from ..models import build_mjcf
        export_xml = build_mjcf(params)
    trainer = make_trainer(d, params, TaskConfig(), custom_path=custom, export_xml=export_xml,
                           init_policy=init_policy, continue_run=continue_run)
    ctx.log(f"training '{algo}' into runs/{Path(trainer.policy_path).parent.name}")
    trainer.start()
    st = _wait_job(ctx, trainer)
    if st["state"] == "error":
        raise RuntimeError(f"training failed: {st.get('error')}")
    path = Path(trainer.policy_path)
    if not path.exists():
        raise RuntimeError(f"training ended ({st['state']}) without saving a policy at {path}")
    history = list(trainer.history)
    bests = [h["best"] for h in history if h.get("best") is not None]
    best = st.get("best")
    if best is None:
        best = max(bests) if bests else float("nan")
    ctx.progress(1.0, f"{st['state']} after {len(history)} generation(s)", history=history[-400:])
    return {"policy": {"name": path.parent.name, "path": str(path), "kind": algo, "fitness": float(best)},
            "series": _curve(f"Training: {path.parent.name}", history), "best": float(best)}


# =========================================================================================================
# Policy
# =========================================================================================================
def find_policy(name: str) -> dict:
    """{"name", "path", "kind", ...} for a run name under runs/ or the path of a policy .npz file."""
    from ..jobs import list_policies
    name = str(name).strip().strip('"')
    if not name:
        raise GraphError("give a run name or the path of a policy file")
    found = list_policies()
    p = Path(name).expanduser()
    if p.suffix == ".npz":
        if not p.is_file():
            raise GraphError(f"policy file not found: {p}")
        hit = next((f for f in found if Path(f["path"]).resolve() == p.resolve()), None)
        if hit is None:
            if p.name not in _POLICY_FILES:
                raise GraphError(f"a policy file is named one of {', '.join(_POLICY_FILES)}")
            hit = {"name": p.parent.name, "path": str(p),
                   "kind": {"policy.npz": "es", "rl_policy.npz": "ppo", "imitation_policy.npz": "imitation"}[p.name]}
        return dict(hit)
    hit = next((f for f in found if f["name"] == name), None)
    if hit is None:
        names = sorted({f["name"] for f in found})
        raise GraphError(f"no trained policy named {name!r}; available: {', '.join(names) or 'none'}")
    return dict(hit)


@node("policy.load", "Load policy", "Policy", outputs=[Port("policy", "policy")],
      params=[Port("name", "text", "pretrained_walker", doc="Run name under runs/ (or the path of a policy .npz)")],
      doc="A trained policy from the runs folder, by run name.")
def policy_load(ctx, name):
    return {"policy": jsonable(find_policy(name))}


@node("policy.use", "Use policy", "Policy", inputs=[Port("robot", "robot"), Port("policy", "policy")],
      outputs=[Port("robot", "robot")], volatile=True,
      doc="Drive the robot with a trained policy: loads both into the live simulation.")
def policy_use(ctx, robot, policy):
    if not isinstance(policy, dict) or "path" not in policy:
        raise GraphError(f"expected a policy, got {summarize(policy)}")
    out = _robot(robot).copy(policy={k: policy.get(k) for k in ("name", "path", "kind")})
    ensure_applied(ctx, out, force=True)
    return {"robot": out}


@node("policy.list", "List policies", "Policy", outputs=[Port("table", "table")], volatile=True,
      doc="All trained policies found under runs/, newest first.")
def policy_list(ctx):
    from ..jobs import list_policies
    return {"table": jsonable(list_policies())}


# =========================================================================================================
# Design
# =========================================================================================================
@node("design.optimize", "Optimise design", "Design", inputs=[Port("robot", "robot")],
      outputs=[Port("robot", "robot"), Port("series", "series"), Port("table", "table"), Port("best", "number")],
      params=[Port("algo", "choice", "cmaes", choices=["cmaes", "bo"], doc="CMA-ES or Bayesian optimisation"),
              Port("generations", "int", 3, min=1, max=1000),
              Port("population", "int", 4, min=2, max=256, doc="Designs tried per generation"),
              Port("workers", "int", 2, min=1, max=64, doc="Parallel processes"),
              Port("genes", "text", "", doc="Parameters to tune, comma separated (empty: the default leg set)"),
              Port("mode", "choice", "auto", choices=["auto", "walk", "stand"], doc="What each design is scored on"),
              Port("seed", "int", 0, min=0),
              Port("name", "text", "graph_design", doc="Results go to runs/<name>")],
      doc="Search body and motor parameters that walk / stand best (trials = generations x population); cached by its settings.")
def design_optimize(ctx, robot, algo, generations, population, workers, genes, mode, seed, name):
    from ..design.optimizer import DesignConfig, DesignOptimizer
    from ..jobs import RUNS
    robot = _robot(robot)
    if robot.kind != "builtin":
        raise GraphError("design optimisation works on the built-in parametric humanoid")
    gene_list = [g.strip() for g in genes.split(",") if g.strip()]
    bad = [g for g in gene_list if g not in PARAM_SPEC]
    if bad:
        raise GraphError(f"unknown parameter(s): {', '.join(bad)}")
    body = {"algo": algo, "generations": generations, "population": population, "workers": workers, "seed": seed,
            "name": "".join(c for c in name if c.isalnum() or c in "-_") or "graph_design", "eval": {"mode": mode}}
    if gene_list:
        body["genes"] = gene_list
    base = robot.humanoid_params()
    opt = DesignOptimizer(DesignConfig.from_dict(body), base, runs_dir=RUNS)
    opt.start()
    st = _wait_job(ctx, opt)
    if st["state"] == "error":
        raise RuntimeError(f"design optimisation failed: {st.get('error')}")
    bp = opt.best_params
    if bp is None:
        raise RuntimeError("the optimisation ended before any design was evaluated")
    best_genes = {g: float(getattr(bp, g)) for g in opt.genes}
    table = [{"parameter": g, "label": PARAM_SPEC[g][3], "before": float(getattr(base, g)), "after": v,
              "change_%": round(100.0 * (v - float(getattr(base, g))) / (abs(float(getattr(base, g))) or 1.0), 1)}
             for g, v in best_genes.items()]
    history = list(opt.history)
    best = st.get("best")
    return {"robot": robot.copy(params={**robot.params, **best_genes}),
            "series": _curve(f"Design: {opt.cfg.name}", history), "table": table,
            "best": float(best) if best is not None else float("nan")}


@node("design.mass_report", "Mass report", "Design", inputs=[Port("robot", "robot")],
      outputs=[Port("table", "table"), Port("total", "number")],
      doc="Mass of every body of the robot (with its accessories) and the total.")
def design_mass_report(ctx, robot):
    m = _robot(robot).compile()
    if m.nbody < 2:
        return {"table": [], "total": 0.0}
    root = m.body_rootid[1]
    rows = [{"body": m.body(b).name or f"body {b}", "mass_kg": round(float(m.body_mass[b]), 4)}
            for b in range(1, m.nbody) if m.body_rootid[b] == root]
    return {"table": rows, "total": float(sum(float(m.body_mass[b]) for b in range(1, m.nbody)
                                              if m.body_rootid[b] == root))}


# =========================================================================================================
# Script, values, maths
# =========================================================================================================
@node("script.python", "Python script", "Script",
      inputs=[Port("a", "any", optional=True), Port("b", "any", optional=True), Port("c", "any", optional=True)],
      outputs=[Port("out", "any"), Port("stdout", "text")],
      params=[Port("code", "code", "out = a", doc="Python; a, b, c, np, mujoco, ctx and session are defined; set out")],
      volatile=True, doc="Run Python on the inputs a, b, c; whatever it assigns to `out` is the output.")
def script_python(ctx, a, b, c, code):
    import mujoco
    buf = io.StringIO()

    def _print(*args, sep=" ", end="\n", file=None, flush=False):
        if file is None:
            buf.write(sep.join(str(x) for x in args) + end)
        else:
            print(*args, sep=sep, end=end, file=file, flush=flush)

    ns = {"a": a, "b": b, "c": c, "np": np, "mujoco": mujoco, "ctx": ctx, "session": ctx.session, "out": None,
          "print": _print, "__name__": "__graph_script__"}
    exec(compile(code, f"<script {ctx.node_id}>", "exec"), ns)  # noqa: S102 - the user's own script
    return {"out": ns.get("out"), "stdout": buf.getvalue()}


@node("value.number", "Number", "Value", outputs=[Port("value", "number")], params=[Port("value", "number", 0.0)],
      doc="A constant number.")
def value_number(ctx, value):
    return {"value": float(value)}


@node("value.int", "Integer", "Value", outputs=[Port("value", "int")], params=[Port("value", "int", 0)],
      doc="A constant whole number.")
def value_int(ctx, value):
    return {"value": int(value)}


@node("value.text", "Text", "Value", outputs=[Port("value", "text")], params=[Port("value", "text", "")],
      doc="A constant piece of text.")
def value_text(ctx, value):
    return {"value": str(value)}


@node("value.bool", "Boolean", "Value", outputs=[Port("value", "bool")], params=[Port("value", "bool", False)],
      doc="A constant true / false.")
def value_bool(ctx, value):
    return {"value": bool(value)}


_MATH = {k: getattr(np, k) for k in ("sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2", "sqrt", "exp",
                                     "log", "log10", "floor", "ceil", "hypot", "deg2rad", "rad2deg", "clip", "sign",
                                     "pi", "e")}


@node("math.expr", "Expression", "Value", inputs=[Port("a", "number", 0.0), Port("b", "number", 0.0)],
      outputs=[Port("out", "number")],
      params=[Port("expr", "text", "a + b", doc="Uses a, b, np and functions like sin, sqrt, abs, min, max")],
      doc="Evaluate a formula of the numbers a and b.")
def math_expr(ctx, a, b, expr):
    try:
        v = eval(expr, {"np": np, **_MATH, "abs": abs, "min": min, "max": max, "round": round}, {"a": a, "b": b})  # noqa: S307
        return {"out": float(v)}
    except Exception as e:  # noqa: BLE001
        raise GraphError(f"cannot evaluate {expr!r}: {type(e).__name__}: {e}") from None


def _is_series(v) -> bool:
    return isinstance(v, dict) and "x" in v and isinstance(v.get("y"), dict)


@node("series.from_table", "Table to series", "Value", inputs=[Port("table", "table")], outputs=[Port("series", "series")],
      params=[Port("x", "text", "", doc="Column for the x axis (empty: the row number)"),
              Port("y", "text", "", doc="Columns to plot, comma separated (empty: every numeric column)"),
              Port("title", "text", "")],
      doc="Turn columns of a table into a plottable series.")
def series_from_table(ctx, table, x, y, title):
    rows = [r for r in (table or []) if isinstance(r, dict)]
    num = lambda v: isinstance(v, (int, float, np.number)) and not isinstance(v, bool)  # noqa: E731
    x = x.strip()
    cols = [c.strip() for c in y.split(",") if c.strip()]
    if not cols:
        cols = [k for k in (rows[0] if rows else {}) if k != x and all(num(r.get(k)) or r.get(k) is None for r in rows)]
    for c in ([x] if x else []) + cols:
        if rows and not any(c in r for r in rows):
            raise GraphError(f"the table has no column {c!r}; it has: {', '.join(rows[0])}")
    xs = [r.get(x) if num(r.get(x)) else i for i, r in enumerate(rows)] if x else list(range(len(rows)))
    return {"series": _series(title, xs, {c: [r.get(c) if num(r.get(c)) else None for r in rows] for c in cols},
                              x or "row")}


@node("series.merge", "Merge series", "Value", inputs=[Port("a", "series"), Port("b", "series")],
      outputs=[Port("series", "series")], params=[Port("title", "text", "")],
      doc="Put the lines of two series on one plot (b is resampled onto a's x axis when they differ).")
def series_merge(ctx, a, b, title):
    if not _is_series(a) or not _is_series(b):
        raise GraphError("both inputs must be series")
    xa, xb = list(a["x"]), list(b["x"])
    y = {k: list(v) for k, v in a["y"].items()}
    for k, v in b["y"].items():
        if xa != xb:
            ok = [(px, pv) for px, pv in zip(xb, v) if pv is not None]
            v = np.interp(xa, [p[0] for p in ok], [p[1] for p in ok]).tolist() if ok and xa else [None] * len(xa)
        label = k
        while label in y:
            label = f"{label} (b)"
        y[label] = list(v)
    return {"series": {"title": title or a.get("title") or b.get("title") or "", "x": xa, "y": y,
                       "xlabel": a.get("xlabel", ""), "ylabel": a.get("ylabel", "")}}


# =========================================================================================================
# Output
# =========================================================================================================
def describe(value, limit: int = 20000) -> str:
    """Readable full text of a wire value (longer than summarize)."""
    if isinstance(value, str):
        text = value
    elif isinstance(value, np.ndarray):
        text = f"{summarize(value)}\n{np.array2string(value, precision=4, threshold=200)}"
    elif isinstance(value, (dict, list, tuple, RobotSpec)):
        text = json.dumps(jsonable(value), indent=1)
    elif isinstance(value, float):
        text = f"{value:.6g}"
    else:
        text = str(value)
    return text if len(text) <= limit else text[:limit] + f"\n... ({len(text) - limit} more characters)"


@node("out.plot", "Plot", "Output", inputs=[Port("series", "series")], outputs=[Port("series", "series")],
      doc="Show a series as a line plot.")
def out_plot(ctx, series):
    if not _is_series(series):
        raise GraphError(f"expected a series, got {summarize(series)}")
    return {"series": series}


@node("out.print", "Print", "Output", inputs=[Port("value", "any")], outputs=[Port("text", "text")],
      doc="Show any value as text.")
def out_print(ctx, value):
    return {"text": describe(value)}


@node("out.table", "Table", "Output", inputs=[Port("table", "table")], outputs=[Port("table", "table")],
      doc="Show a table.")
def out_table(ctx, table):
    if not isinstance(table, (list, tuple)) or not all(isinstance(r, dict) for r in table):
        raise GraphError(f"expected a table (a list of rows), got {summarize(table)}")
    return {"table": list(table)}


@node("out.image", "Image", "Output", inputs=[Port("image", "image")], outputs=[Port("image", "image")],
      doc="Show an image.")
def out_image(ctx, image):
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise GraphError(f"expected an image, got {summarize(image)}")
    return {"image": image}


@node("out.save_json", "Save JSON", "Output", inputs=[Port("value", "any")], outputs=[Port("path", "text")],
      params=[Port("path", "text", "runs/graph_output.json", doc="File to write")], volatile=True,
      doc="Write any value to a JSON file.")
def out_save_json(ctx, value, path):
    if not path.strip():
        raise GraphError("give a file path")
    p = Path(path.strip().strip('"')).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(jsonable(value), indent=1), encoding="utf-8")
    ctx.log(f"wrote {p}")
    return {"path": str(p)}


# =========================================================================================================
# Ready-made graphs
# =========================================================================================================
_DX = 260


def example_walk_to_goal() -> Graph:
    g = Graph()
    r = g.add_node("robot.builtin", (0, 0))
    w = g.add_node("world.obstacles", (_DX, 0), {"n": 10, "seed": 3})
    a = g.add_node("sim.apply", (2 * _DX, 0))
    k = g.add_node("sim.walk_to", (3 * _DX, 0), {"x": 3.0, "y": 1.0, "timeout": 60.0})
    p = g.add_node("out.plot", (4 * _DX, 0))
    g.connect(r.id, "robot", w.id, "robot")
    g.connect(w.id, "robot", a.id, "robot")
    g.connect(a.id, "robot", k.id, "robot")
    g.connect(k.id, "path", p.id, "series")
    return g


def example_train_and_evaluate() -> Graph:
    g = Graph()
    r = g.add_node("robot.builtin", (0, 0))
    t = g.add_node("train.run", (_DX, 0), {"algo": "es", "name": "graph_example", "population": 16,
                                           "generations": 3, "episode_seconds": 2.0})
    u = g.add_node("policy.use", (2 * _DX, 0))
    s = g.add_node("sim.run", (3 * _DX, 0), {"seconds": 5.0, "speed": 4.0})
    p = g.add_node("out.plot", (4 * _DX, 0), title="Evaluation")
    c = g.add_node("out.plot", (2 * _DX, 260), title="Training curve")
    g.connect(r.id, "robot", t.id, "robot")
    g.connect(r.id, "robot", u.id, "robot")
    g.connect(t.id, "policy", u.id, "policy")
    g.connect(u.id, "robot", s.id, "robot")
    g.connect(s.id, "series", p.id, "series")
    g.connect(t.id, "series", c.id, "series")
    return g


def example_lidar_on_a_dog() -> Graph:
    g = Graph()
    r = g.add_node("robot.library", (0, 0), {"name": "go2"})
    a = g.add_node("robot.accessory", (_DX, 0), {"type": "lidar", "x": 0.0, "y": 0.0, "z": 0.1})
    s = g.add_node("sim.apply", (2 * _DX, 0))
    l = g.add_node("sensor.lidar", (3 * _DX, 0))
    p = g.add_node("out.print", (4 * _DX, 0))
    g.connect(r.id, "robot", a.id, "robot")
    g.connect(a.id, "robot", s.id, "robot")
    g.connect(s.id, "robot", l.id, "robot")
    g.connect(l.id, "hits", p.id, "value")
    return g


def example_script() -> Graph:
    g = Graph()
    n = g.add_node("value.number", (0, 0), {"value": 3.0})
    s = g.add_node("script.python", (_DX, 0), {"code": "out = a ** 2 + 1\nprint('a squared plus one is', out)"})
    p = g.add_node("out.print", (2 * _DX, 0))
    g.connect(n.id, "value", s.id, "a")
    g.connect(s.id, "out", p.id, "value")
    return g


EXAMPLES = {"Walk to a goal": example_walk_to_goal, "Train and evaluate": example_train_and_evaluate,
            "Lidar on a dog": example_lidar_on_a_dog, "Script": example_script}
