"""World model slot: plug in a model that turns an instruction in words plus what the robot senses into action.

This is a placeholder for the real thing. What exists is the socket, not the model:

    interface   `WorldModel.act(observation, instruction) -> Action`, called at a fixed rate
    observation joint angles and speeds, root pose and velocity, foot contact, named objects, and optionally a
                camera image and the lidar points
    action      one of: joint targets (the model drives the motors itself), a walking goal, a named skill (a
                sample such as "backflip", or "stop"), or nothing
    runner      `WorldModelRunner` calls the model on its own thread and applies its actions to the session
    stand-in    `KeywordModel`, which only pattern-matches the instruction ("walk to 2 1", "backflip", "jump",
                "stop"). It understands nothing; it is there so the loop can be seen working end to end.

To plug in your own, write a Python file with a `create()` function returning an object with `act` (and
optionally `reset` and `name`), and load it with `load_world_model(path)` or from the World Model panel.
Loading a file runs its code with full access to this machine: only load files you trust.

    # my_model.py
    from simulatics.worldmodel import Action
    class Mine:
        name = "mine"
        def reset(self, info): ...
        def act(self, obs, instruction):
            return Action.go(1.0, 0.0)              # or Action.drive({joint: angle}), Action.do("backflip")
    def create():
        return Mine()

The instruction is plain text, so it can come from a person or from a language model's output.
"""
from __future__ import annotations

import importlib.util
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np


@dataclass
class Observation:
    time: float
    joint_pos: dict[str, float]            # rad (m for sliding joints), by joint name
    joint_vel: dict[str, float]
    root_pos: np.ndarray                   # (3,) world
    root_quat: np.ndarray                  # (4,) w x y z
    root_lin_vel: np.ndarray               # (3,) world
    root_ang_vel: np.ndarray               # (3,) body frame
    foot_contact: dict[str, bool]
    objects: list[dict]                    # [{"name", "x", "y", "z"}] loose objects in the world
    motors: list[str]                      # names of the joints that have a motor, in motor order
    image: np.ndarray | None = None        # (H, W, 3) uint8 from `camera`, when asked for
    lidar: np.ndarray | None = None        # (n, 3) world points of the last scan, when a lidar is mounted
    info: dict = field(default_factory=dict)


@dataclass
class Action:
    kind: str = "none"                     # "none" | "joints" | "goal" | "skill"
    joints: dict[str, float] | None = None
    goal: tuple[float, float] | None = None
    skill: str | None = None
    say: str = ""                          # shown in the log (what the model thinks it is doing)

    @classmethod
    def none(cls, say: str = "") -> "Action":
        return cls("none", say=say)

    @classmethod
    def joints_(cls, targets: dict[str, float], say: str = "") -> "Action":
        return cls("joints", joints=dict(targets), say=say)

    @classmethod
    def goal_(cls, x: float, y: float, say: str = "") -> "Action":
        return cls("goal", goal=(float(x), float(y)), say=say)

    @classmethod
    def skill_(cls, name: str, say: str = "") -> "Action":
        return cls("skill", skill=str(name), say=say)


# the constructors models are meant to use: Action.go(x, y), Action.do(skill), Action.drive({joint: angle})
Action.go = Action.goal_
Action.do = Action.skill_
Action.drive = Action.joints_


def observe(session, camera: str | None = None, lidar: bool = False) -> Observation:
    """What the robot senses right now, read from a live session."""
    from ..sim.props import props_state
    with session.lock:
        m, d = session.model, session.data
        pos, vel, motors = {}, {}, []
        for j in range(m.njnt):
            if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                name = m.joint(j).name
                pos[name] = float(d.qpos[m.jnt_qposadr[j]])
                vel[name] = float(d.qvel[m.jnt_dofadr[j]])
        for a in range(m.nu):
            if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                motors.append(m.joint(int(m.actuator_trnid[a, 0])).name)
        free = session.task is not None
        contact = {}
        for b in range(m.nbody):
            if m.body(b).name.endswith("_foot"):
                contact[m.body(b).name] = any(m.geom_bodyid[d.contact[i].geom1] == b or m.geom_bodyid[d.contact[i].geom2] == b
                                             for i in range(d.ncon))
        obs = Observation(float(d.time), pos, vel,
                          d.qpos[0:3].copy() if free else np.zeros(3), d.qpos[3:7].copy() if free else np.array([1.0, 0, 0, 0]),
                          d.qvel[0:3].copy() if free else np.zeros(3), d.qvel[3:6].copy() if free else np.zeros(3),
                          contact, [{k: o[k] for k in ("name", "x", "y", "z")} for o in props_state(m, d)], motors,
                          info={"controller": getattr(session.controller, "name", ""), "falls": session.falls,
                                "nav": session.nav.state})
        if lidar and session.lidar_points is not None:
            obs.lidar = session.lidar_points.copy()
    if camera:
        try:
            obs.image = session.robot.camera_image(camera) if hasattr(session.robot, "camera_image") else None
        except Exception:  # noqa: BLE001 - a missing camera must not stop the loop
            obs.image = None
    return obs


class JointTargetController:
    """The controller in charge while a world model drives the motors: writes its latest joint targets."""
    name = "world-model"

    def __init__(self, model: mujoco.MjModel):
        self.control_dt = float(model.opt.timestep) * 4
        self.act = {model.joint(int(model.actuator_trnid[a, 0])).name: a for a in range(model.nu)
                    if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}
        self.target: np.ndarray | None = None
        self.lo, self.hi = model.actuator_ctrlrange[:, 0].copy(), model.actuator_ctrlrange[:, 1].copy()
        self.limited = model.actuator_ctrllimited.astype(bool)

    def reset(self, model, data) -> None:
        self.target = data.ctrl.copy()

    def set_path(self, path_xy, final_yaw=None) -> None:
        pass

    def set_targets(self, data, targets: dict[str, float]) -> list[str]:
        if self.target is None:
            self.target = data.ctrl.copy()
        unknown = []
        for name, v in targets.items():
            a = self.act.get(name)
            if a is None or not np.isfinite(v):
                unknown.append(name)
                continue
            self.target[a] = np.clip(v, self.lo[a], self.hi[a]) if self.limited[a] else v
        return unknown

    def step(self, model, data) -> None:
        if self.target is not None:
            data.ctrl[:] = self.target

    def status(self) -> dict:
        return {"mode": "walk", "message": "a world model is setting the joint targets"}


class KeywordModel:
    """Stand-in, not a world model: matches a few phrases in the instruction and maps them to skills."""
    name = "keyword stand-in (placeholder)"

    def __init__(self):
        self._last = None

    def reset(self, info: dict) -> None:
        self._last = None

    def act(self, obs: Observation, instruction: str) -> Action:
        text = instruction.lower().strip()
        if text == self._last:                    # one action per instruction; then keep quiet
            return Action.none()
        self._last = text
        if not text:
            return Action.none()
        from .. import samples
        if re.search(r"\b(stop|halt|freeze|stand still)\b", text):
            return Action.do("stop", "stopping")
        m = re.search(r"(?:walk|go|move|run)\s+(?:to\s+)?\(?\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)", text)
        if m:
            return Action.go(float(m.group(1)), float(m.group(2)), f"walking to ({m.group(1)}, {m.group(2)})")
        m = re.search(r"(?:walk|go|move)\s+(forward|ahead|back|backward|left|right)(?:\s+(\d+(?:\.\d+)?))?", text)
        if m:
            dist = float(m.group(2) or 1.0)
            yaw = float(np.arctan2(2 * (obs.root_quat[0] * obs.root_quat[3] + obs.root_quat[1] * obs.root_quat[2]),
                                   1 - 2 * (obs.root_quat[2] ** 2 + obs.root_quat[3] ** 2)))
            off = {"forward": 0.0, "ahead": 0.0, "back": np.pi, "backward": np.pi, "left": np.pi / 2, "right": -np.pi / 2}[m.group(1)]
            x, y = obs.root_pos[0] + dist * np.cos(yaw + off), obs.root_pos[1] + dist * np.sin(yaw + off)
            return Action.go(x, y, f"walking {dist:g} m {m.group(1)}")
        words = {"backflip": "backflip", "back flip": "backflip", "front flip": "front_flip", "jump onto": "box_jump_up",
                 "box": "box_jump_up", "hurdle": "hurdle", "jump forward": "broad_jump", "broad jump": "broad_jump",
                 "long jump": "broad_jump", "jump": "vertical_jump", "wave": "one_hand_wave", "stairs": "stairs",
                 "kitchen": "kitchen", "playground": "playground", "parkour": "parkour_course", "stand": "stand"}
        for phrase, sid in words.items():
            if phrase in text and sid in samples.SAMPLES:
                s = samples.SAMPLES[sid]
                return Action.do(sid, f"running the '{s.title}' sample ({s.kind})")
        return Action.none(f"I am only a keyword stand-in and found nothing I know in: {instruction!r}")


def load_world_model(path: str | Path):
    """Import a Python file and return `create()` from it. This runs the file's code: only load files you trust."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {p}")
    spec = importlib.util.spec_from_file_location(f"simulatics_user_worldmodel_{p.stem}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "create"):
        raise ValueError(f"{p.name} must define create() returning an object with act(observation, instruction)")
    model = mod.create()
    if not callable(getattr(model, "act", None)):
        raise ValueError("create() must return an object with an act(observation, instruction) method")
    return model


class WorldModelRunner:
    """Calls a world model at a fixed rate and applies what it returns to the session."""

    def __init__(self, session):
        self.session = session
        self.model = KeywordModel()
        self.instruction = ""
        self.rate_hz = 5.0
        self.camera: str | None = None
        self.lidar = False
        self.log: list[tuple[float, str]] = []
        self.calls = 0
        self.error = ""
        self.last_ms = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _say(self, text: str) -> None:
        if text and (not self.log or self.log[-1][1] != text):
            self.log.append((time.time(), text))
            del self.log[:-500]

    def set_model(self, model) -> None:
        if self.running:
            raise RuntimeError("stop the running model first")
        self.model = model
        self._say(f"model: {getattr(model, 'name', type(model).__name__)}")

    def instruct(self, text: str) -> None:
        """Give the model a new instruction (it sees it on its next call)."""
        self.instruction = str(text)
        self._say(f"instruction: {self.instruction}")

    def start(self, instruction: str | None = None) -> None:
        if self.running:
            if instruction is not None:
                self.instruct(instruction)
            return
        if instruction is not None:
            self.instruct(instruction)
        self.error = ""
        self._stop.clear()
        try:
            if callable(getattr(self.model, "reset", None)):
                self.model.reset({"motors": observe(self.session).motors})
        except Exception as e:  # noqa: BLE001
            self.error = f"reset failed: {type(e).__name__}: {e}"
            self._say(self.error)
            return
        self._thread = threading.Thread(target=self._loop, name="world-model", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and threading.current_thread() is not self._thread:
            self._thread.join(2.0)
        s = self.session
        with s.lock:
            if getattr(s.controller, "name", "") == "world-model":   # give the robot back to its own controller
                s.controller_kind = "auto"
                s.controller = s._make_controller()
                s.controller.reset(s.model, s.data)

    def step_once(self) -> Action:
        """One observe-act-apply cycle (the loop calls this; tests can call it directly)."""
        t = time.perf_counter()
        obs = observe(self.session, self.camera, self.lidar)
        out = self.model.act(obs, self.instruction)
        self.last_ms = 1000 * (time.perf_counter() - t)
        self.calls += 1
        act = out if isinstance(out, Action) else Action(**out) if isinstance(out, dict) else Action.none()
        self.apply(act)
        return act

    def apply(self, act: Action) -> None:
        s = self.session
        self._say(act.say)
        if act.kind == "goal" and act.goal is not None:
            with s.lock:
                if s.task is None:
                    raise ValueError("this robot cannot walk")
                if getattr(s.controller, "name", "") == "world-model":
                    s.controller_kind = "auto"
                    s.controller = s._make_controller()
                    s.controller.reset(s.model, s.data)
                s.nav.set_goal(list(act.goal))
                s.running = True
        elif act.kind == "skill" and act.skill:
            if act.skill == "stop":
                with s.lock:
                    s.nav.clear()
            else:
                from .. import samples
                samples.run(s, act.skill)
        elif act.kind == "joints" and act.joints:
            with s.lock:
                c = s.controller
                if getattr(c, "name", "") != "world-model":
                    s._release_tape()
                    c = JointTargetController(s.model)
                    c.reset(s.model, s.data)
                    s.controller_kind, s.controller = "world-model", c
                unknown = c.set_targets(s.data, act.joints)
                s.running = True
            if unknown:
                self._say(f"ignored unknown joints: {', '.join(unknown[:6])}")

    def _loop(self) -> None:
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                self.step_once()
            except Exception as e:  # noqa: BLE001 - a model's error ends the run and is shown, never crashes the app
                self.error = f"{type(e).__name__}: {e}"
                self._say(f"stopped on an error: {self.error}")
                return
            self._stop.wait(max(0.0, 1.0 / max(self.rate_hz, 0.1) - (time.perf_counter() - t)))
