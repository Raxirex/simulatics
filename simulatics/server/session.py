"""The live simulation behind the GUI: one model, one thread stepping it in real time."""
from __future__ import annotations

import base64
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

from ..control import ManualController, PolicyController, base_pose
from ..models import HumanoidParams, PARAM_SPEC, build_mjcf
from ..plan import GridMap, Navigator
from ..sim.sensors import CameraRig, Lidar, LidarConfig
from ..sim.task import LocoTask, TaskConfig, has_free_root
from ..train.policy import Policy

_GEOM_NAMES = {int(getattr(mujoco.mjtGeom, n)): n[7:].lower() for n in dir(mujoco.mjtGeom) if n.startswith("mjGEOM_")}


def _jsonable(v):
    """numpy scalars/arrays inside controller status dicts -> plain JSON types."""
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return np.round(v, 4).tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _mat2quat(mats: np.ndarray) -> np.ndarray:
    out = np.zeros((len(mats), 4))
    for i, m in enumerate(mats):
        mujoco.mju_mat2Quat(out[i], m)
    return out


class SimSession:
    def __init__(self):
        self.lock = threading.RLock()
        self.params = HumanoidParams()
        self.obstacles: list[dict] = []
        self.props: list[dict] = []      # graspable free bodies (cubes, balls)
        self.arms = None                 # ArmController when the robot has hands
        self.custom_path: str | None = None
        self.cad_xml: str | None = None  # MJCF with CAD meshes / masses (Design tab), replaces the generated model
        self.accessories: list[dict] | None = None  # lidar / cameras / payloads mounted in the Workshop
        self.accessory_warnings: list[str] = []
        self.default_lidar = True        # built-in robot starts with the head lidar
        self.support_stand = False       # imported robots: hold the base on a stand (weld)
        self.robot_name = "builtin"
        self.task_cfg = TaskConfig()
        self.lidar_cfg = LidarConfig()
        self.camera_cfg = {"enabled": False, "name": "head_cam", "depth": False}
        self.policy: Policy | None = None
        self.policy_name = ""
        self.controller_kind = "auto"    # auto | walker | rl | es-policy | manual
        self.controller = None
        self.running = False
        self.speed = 1.0
        self.auto_reset = True
        self.scene_version = 0
        self.grid = GridMap()
        self.nav = Navigator(self.grid)
        self.heading = 0.0               # manual heading command (rad) when no goal is set
        self.error = ""
        # outputs produced by the sim thread, consumed by the websocket
        self.lidar_points: np.ndarray | None = None
        self.lidar_seq = 0
        self.camera_jpeg: bytes | None = None
        self.camera_seq = 0
        self.camera_error = ""
        self.map_seq = 0
        self._rig: CameraRig | None = None
        self._rig_version = -1
        self._alive = True
        self._step_once = False
        from .robot_adapter import SessionRobot
        from ..llm import PilotRunner
        self.robot = SessionRobot(self)  # RobotInterface for AI pilots
        self.pilot = PilotRunner(self.robot)
        self.rl_policy = None            # PPO / distilled policy (train/ppo.py RLPolicy)
        self.imitation_policy: str | None = None   # path of a trained motion-tracking policy
        self.motion_name: str | None = None        # clip name or .npz path for preview / PD playback
        self._arm_tick = 0
        self._push = None                # (end sim time, world force) while a push is applied
        # SLAM: localises from noisy leg odometry + lidar only, never from the true pose
        from ..plan.slam import OdomNoise, Slam2D
        self.slam = Slam2D(size=self.grid.size, res=0.05, odom_noise=OdomNoise(), seed=0)
        self.slam_grid = self.slam.gridmap()
        self.slam_true: list = []        # true positions, to draw against the estimate
        self._slam_t = -1.0
        self.nav_on_slam = False         # plan with the SLAM map and the SLAM pose
        from collections import deque
        self.motor_trace: deque = deque(maxlen=6000)   # (time, target, angle, speed, torque) of the traced motor
        self._trace_act: int | None = None
        self.build()
        self._thread = threading.Thread(target=self._loop, name="sim", daemon=True)
        self._thread.start()

    # ---- model construction ---------------------------------------------
    def build(self) -> None:
        """(Re)compile the model from the current parameters / file and reset the simulation."""
        if getattr(self, "pilot", None) is not None and self.pilot.running:
            self.pilot.stop(timeout=0.5)  # its goals and object names refer to the old world
        with self.lock:
            if self.custom_path:
                model = self._compile_imported()
            elif self.cad_xml:
                model = mujoco.MjModel.from_xml_string(self.cad_xml)
            else:
                xml = self.generated_xml()
                model = self._on_stand(xml) if self.support_stand else mujoco.MjModel.from_xml_string(xml)
            self.model, self.data = model, mujoco.MjData(model)
            self.motor_trace.clear()
            self.task = LocoTask(model, self.task_cfg) if has_free_root(model) and model.nu > 0 else None
            self.manual_ctrl = self.task.ctrl0.copy() if self.task else np.zeros(model.nu)
            self.lidar = Lidar(model, self.lidar_cfg)
            self.arms = self._make_arms()
            self.controller = self._make_controller()
            self.reset()
            # terrain pieces (stairs, ramps, low blocks) are walked on, not walked around
            walkable = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"obj_{i}")
                        for i, o in enumerate(self.obstacles) if o.get("kind") == "terrain"}
            self.grid.rasterize_truth(model, self.data, walkable=walkable - {-1})
            self.grid.reset_lidar()
            self.nav.clear()
            self.scene_version += 1
            self.map_seq += 1

    def generated_xml(self) -> str:
        """MJCF of the built-in robot with the current world objects, props and mounted accessories."""
        from ..models.accessories import apply_accessories, default_accessories
        if self.props:
            xml = build_mjcf(self.params, self.obstacles, props=self.props)
        else:
            xml = build_mjcf(self.params, self.obstacles)
        if self.accessories is None:
            self.accessories = default_accessories(xml) if self.default_lidar else []
        xml, self.accessory_warnings = apply_accessories(xml, self.accessories)
        return xml

    # ---- workshop: accessories -------------------------------------------
    def workshop(self, enter: bool) -> None:
        """Workshop mode: the robot stands still in its reset pose while accessories are edited."""
        with self.lock:
            self.running = False
            if enter:
                self.reset()

    def accessories_info(self) -> dict:
        from ..models.accessories import TYPES, describe
        with self.lock:
            items = describe(self.model, self.data, self.accessories or [])
        return {"types": {k: {"label": v["label"], "hint": v["hint"]} for k, v in TYPES.items()},
                "items": items, "warnings": list(self.accessory_warnings),
                "editable": True, "default_lidar": self.default_lidar, "support_stand": self.support_stand,
                "robot": self.robot_name, "lidar_mounted": self.lidar.mounted}

    def place_accessory(self, acc_type: str, geom: int, point, normal, snap: float = 0.01,
                        acc_id: str | None = None) -> dict:
        """Mount a new accessory (or move `acc_id`) where it was dropped on the robot, then rebuild."""
        from ..models.accessories import new_id, place
        with self.lock:
            items = list(self.accessories or [])
            old = next((a for a in items if a["id"] == acc_id), None) if acc_id else None
            if acc_id and old is None:
                raise ValueError(f"no accessory {acc_id!r}")
            kind = old["type"] if old else acc_type
            mount = place(self.model, self.data, kind, int(geom), point, normal, float(snap))
            if old:
                old.update({k: mount[k] for k in ("body", "pos", "quat")})
                acc = old
            else:
                if kind == "lidar":  # one lidar per robot: placing a new one moves it
                    items = [a for a in items if a["type"] != "lidar"]
                acc = {"id": new_id(), **mount}
                items.append(acc)
            self.accessories = items
            self.cad_xml = None
            self.build()
            self.running = False
        return acc

    def update_accessory(self, acc_id: str, fields: dict) -> dict:
        """Edit a mounted accessory's position (body frame, metres), orientation or parameters."""
        with self.lock:
            acc = next((a for a in (self.accessories or []) if a["id"] == acc_id), None)
            if acc is None:
                raise ValueError(f"no accessory {acc_id!r}")
            if "pos" in fields:
                pos = [float(v) for v in fields["pos"]]
                if len(pos) != 3 or not all(np.isfinite(pos)) or max(abs(v) for v in pos) > 2.0:
                    raise ValueError("position must be three numbers within 2 m of the body")
                acc["pos"] = pos
            if "fovy" in fields and acc["type"] == "camera":
                acc["fovy"] = float(np.clip(float(fields["fovy"]), 10.0, 160.0))
            if "mass" in fields:
                acc["mass"] = float(np.clip(float(fields["mass"]), 0.001, 30.0))
            self.cad_xml = None
            self.build()
            self.running = False
            return acc

    def delete_accessory(self, acc_id: str) -> None:
        with self.lock:
            before = len(self.accessories or [])
            self.accessories = [a for a in (self.accessories or []) if a["id"] != acc_id]
            if len(self.accessories) == before:
                raise ValueError(f"no accessory {acc_id!r}")
            self.cad_xml = None
            self.build()
            self.running = False

    def _make_arms(self):
        try:
            from ..control.arms import ArmController
        except ImportError:
            return None
        try:
            if not ArmController.supports(self.model):
                return None
            return ArmController(self.model, self.data)
        except Exception:  # noqa: BLE001 - model without hands / arms: no arm skills
            return None

    def _make_controller(self):
        """Pick the controller for the current model: an explicit choice if it is possible, else the best available."""
        m, kind = self.model, self.controller_kind
        if self.task is None or (self.support_stand and not self.custom_path):
            return ManualController(m, self.manual_ctrl)     # on the stand: joints are driven directly
        if self.policy is not None and kind in ("auto", "es-policy"):
            return PolicyController(m, self.policy, self.task)
        if getattr(self, "rl_policy", None) is not None and kind in ("auto", "rl"):
            from ..train.ppo import RLController
            return RLController(m, self.rl_policy)
        if kind in ("imitation", "motion-preview", "motion-pd"):
            from ..imitate import ImitationController, MotionPreviewController, PDPlaybackController, load_motion
            if kind == "imitation" and self.imitation_policy:
                return ImitationController(m, self.imitation_policy)
            if kind in ("motion-preview", "motion-pd") and self.motion_name:
                motion = load_motion(self.motion_name)
                return (MotionPreviewController(m, motion) if kind == "motion-preview"
                        else PDPlaybackController(m, motion))
        if kind == "mpc":
            try:
                from ..control.mpc import MPCWalkController
                if MPCWalkController.supports(m):
                    return MPCWalkController(m)
            except ImportError:
                pass
        if kind in ("auto", "walker", "mpc"):
            try:
                from ..control.walker import WalkController
                if WalkController.supports(m):
                    return WalkController(m)
            except ImportError:
                pass
        if kind != "manual":
            try:
                from ..control.pd_hold import PDHoldController
                return PDHoldController(m)  # imported robots: hold the stand pose
            except ValueError:
                pass
        return ManualController(m, self.manual_ctrl)

    def play_motion(self, motion: str | None, mode: str, policy: str | None = None) -> str:
        """Play a reference motion: "preview" (kinematic), "pd" (servos + balance) or "policy" (learned tracker)."""
        kinds = {"preview": "motion-preview", "pd": "motion-pd", "policy": "imitation"}
        if mode not in kinds:
            raise ValueError("mode must be preview, pd or policy")
        with self.lock:
            if mode == "policy":
                if not policy:
                    raise ValueError("choose a trained imitation policy")
                self.imitation_policy = policy
            else:
                self.motion_name = motion
            self.controller_kind = kinds[mode]
            self.controller = self._make_controller()
            if self.controller.name != kinds[mode]:
                raise ValueError("could not start that motion on this robot")
            self.reset()
            self.running = True
        return f"playing {motion or Path(policy).parent.name} ({mode})"

    def play_tape(self, tape) -> str:
        """Play a physically simulated motion (imitate/physical.py Tape, or a path to one): from its standing
        start state the controller writes only the recorded motor commands; physics does the rest."""
        from ..imitate.physical import Tape, TapeController
        if not hasattr(tape, "ctrl"):
            tape = Tape.load(tape)
        with self.lock:
            self._release_tape()
            ctrl = TapeController(self.model, tape)             # raises when the robot or world differs
            self.controller_kind, self.controller = "motion-physical", ctrl
            self.nav.clear()
            self.nav.take_new_path()
            self._push = None
            self.data.xfrc_applied[:] = 0
            ctrl.reset(self.model, self.data)
            self.reset_slam()
            self.running = True
        return f"playing '{tape.name}' in physics ({tape.duration:.1f} s of recorded motor commands)"

    def _release_tape(self) -> None:
        """Undo the model settings a recorded motion plays under (see imitate/physical.py prepare_model)."""
        c = self.controller
        if getattr(c, "name", "") == "motion-physical" and getattr(c, "_saved", None) is not None:
            c.release(self.model)

    def _end_tape(self) -> None:
        """Hand the robot, as it stands after a recorded motion, to the walking controller."""
        self._release_tape()
        self.controller_kind = "auto"
        new = self._make_controller()
        feet = [b for b in range(self.model.nbody) if self.model.body(b).name.endswith("_foot")]
        if getattr(new, "name", "") == "walker" and feet and min(float(self.data.xpos[b][2]) for b in feet) > 0.25:
            # it ended standing on something (a box): the walker must treat that as ground, whatever its height
            from ..control.walker import WalkController, WalkParams
            from ..control.walk_terrain import TerrainLimits, TerrainMap
            new = WalkController(self.model, WalkParams(terrain=True))
            lg = new.geo.legs["left"]
            new.terrain = TerrainMap(self.model, lg.half_length, lg.half_width,
                                     TerrainLimits(margin=new.p.terrain_margin), max_top=3.0)
        if hasattr(new, "adopt"):
            new.adopt(self.model, self.data)
        else:
            new.reset(self.model, self.data)
        new.set_path(None)
        self.controller = new
        goals = list(self.nav.goals)            # plan again for the new controller
        self.nav.clear()
        self.nav.take_new_path()
        for i, (g, yaw) in enumerate(goals):
            self.nav.set_goal(g, yaw, append=i > 0)

    def set_controller(self, kind: str) -> None:
        with self.lock:
            self._release_tape()
            self.controller_kind = kind
            self.controller = self._make_controller()
            self.reset()

    def reset(self) -> None:
        with self.lock:
            if self.model.nkey > 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            else:
                mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.falls = getattr(self, "falls", 0)
            self._push = None  # mj_resetData already cleared xfrc_applied
            if self.controller is not None:
                self.controller.reset(self.model, self.data)
                self.controller.set_path(None)
            if self.arms is not None:
                self.arms = self._make_arms()  # fresh arm state (nothing held, arms home)
            if hasattr(self, "nav"):
                self.nav.clear()
                self.nav.take_new_path()
            self.reset_slam()

    def reset_slam(self) -> None:
        """Restart SLAM at the robot's current pose: the only time it is given the truth, so the map frame
        lines up with the world in the 3D view. Every reset, rebuild and fall teleports the robot."""
        if getattr(self, "slam", None) is None or self.task is None:
            return
        from ..plan.slam import root_pose
        with self.lock:
            p = root_pose(self.data, 1)
            self.slam.reset(p)
            self.slam.odometry.reset(p)
            self.slam_true.clear()
            self._slam_t = -1.0
            self.map_seq += 1

    def set_params(self, d: dict) -> None:
        with self.lock:
            self.params = HumanoidParams.from_dict({**self.params.to_dict(), **d})
            if self.custom_path:
                self.accessories, self.support_stand, self.robot_name = None, False, "builtin"
            self.custom_path = None
            self.cad_xml = None
            self.build()

    def load_file(self, path: str, support_stand: bool = False, robot_name: str | None = None) -> None:
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"no such file: {p}")
        with self.lock:
            old = (self.custom_path, self.accessories, self.support_stand, self.robot_name)
            self.custom_path, self.accessories = str(p), []   # imported robots start bare: place a lidar
            self.support_stand, self.robot_name = bool(support_stand), robot_name or p.stem
            self.policy, self.policy_name, self.rl_policy = None, "", None
            if self.controller_kind in ("es-policy", "rl", "imitation", "motion-preview", "motion-pd", "walker", "mpc"):
                self.controller_kind = "auto"
            try:
                self.build()
            except Exception:
                self.custom_path, self.accessories, self.support_stand, self.robot_name = old
                self.build()
                raise

    def load_robot(self, name: str) -> str:
        """Load a library robot (downloading it first if needed); returns a status message."""
        from .. import robots
        if name == "builtin":
            self.set_params({})
            return "built-in parametric humanoid"
        info = robots.LIBRARY.get(name)
        if info is None:
            raise ValueError(f"unknown robot {name!r}")
        path = robots.fetch(name)
        # humanoids from the library have no balance controller here: start them on a support stand
        self.load_file(str(path), support_stand=info["kind"] == "humanoid", robot_name=name)
        return f"{info['label']} ({info['license']})"

    def set_support_stand(self, on: bool) -> None:
        with self.lock:
            self.support_stand = bool(on)
            self.build()

    def set_default_lidar(self, on: bool) -> None:
        """Built-in robot: start with the head lidar (on) or with no lidar at all (off)."""
        with self.lock:
            self.default_lidar = bool(on)
            items = [a for a in (self.accessories or []) if a["id"] != "lidar"]
            if on and not self.custom_path and not any(a["type"] == "lidar" for a in items):
                from ..models.accessories import default_accessories
                items = default_accessories(build_mjcf(self.params)) + items
            self.accessories = items
            self.cad_xml = None
            self.build()
            self.running = False

    def _on_stand(self, xml: str, lift: float = 0.12) -> mujoco.MjModel:
        """Built-in robot hung on a support stand: the torso is welded in place a little above standing height,
        so the feet are off the floor and single joints can be driven without the robot falling over."""
        spec = mujoco.MjSpec.from_string(xml)
        root = mujoco.MjModel.from_xml_string(xml).body(1).name
        try:
            spec.body(root).pos[2] += lift
        except Exception:  # noqa: BLE001 - older MuJoCo: stand at standing height instead
            pass
        spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY, name1=root, name="support_stand")
        return spec.compile()

    # ---- precise motor control ---------------------------------------------------------------------------
    def trace_motor(self, index: int | None) -> None:
        """Record target, angle, speed and torque of one motor at every control step (None = stop)."""
        with self.lock:
            self._trace_act = None if index is None or not 0 <= int(index) < self.model.nu else int(index)
            self.motor_trace.clear()

    def set_motor_signal(self, index: int, kind: str = "hold", amplitude: float = 0.0, frequency: float = 1.0,
                         base: float | None = None) -> str:
        """Drive one motor with a test signal around `base` (rad): "hold", "step", "sine", "square" or "chirp"
        (0.2 Hz to `frequency` over 10 s). Needs manual control (use the support stand so the robot stays up)."""
        kinds = ("hold", "step", "sine", "square", "chirp")
        if kind not in kinds:
            raise ValueError(f"signal must be one of {kinds}")
        with self.lock:
            c, i = self.controller, int(index)
            if getattr(c, "name", "") != "manual":
                raise ValueError("switch to manual control first (Motor Control: hold the robot on the stand)")
            if not 0 <= i < self.model.nu:
                raise IndexError("motor index out of range")
            b = float(self.manual_ctrl[i] if base is None else base)
            t0, a, f = float(self.data.time), float(amplitude), max(float(frequency), 0.01)
            lo, hi = self.model.actuator_ctrlrange[i]
            lim = bool(self.model.actuator_ctrllimited[i])

            def sig(t, kind=kind):
                x = t - t0
                if kind == "hold":
                    v = b
                elif kind == "step":
                    v = b + (a if x >= 0.2 else 0.0)
                elif kind == "sine":
                    v = b + a * np.sin(2 * np.pi * f * x)
                elif kind == "square":
                    v = b + a * (1.0 if np.sin(2 * np.pi * f * x) >= 0 else -1.0)
                else:
                    k = (f - 0.2) / 10.0
                    v = b + a * np.sin(2 * np.pi * (0.2 * x + 0.5 * k * min(x, 10.0) ** 2 + max(x - 10.0, 0) * (f - 0.2)))
                return i, float(np.clip(v, lo, hi) if lim else v)
            c.signal = None if kind == "hold" else sig
            if kind == "hold":
                self.manual_ctrl[i] = np.clip(b, lo, hi) if lim else b
            self.motor_trace.clear()
            self.running = True
        return f"{kind} on {self.model.actuator(i).name}"

    def _compile_imported(self) -> mujoco.MjModel:
        """Imported MJCF/URDF compiled through MjSpec (assets resolve next to the file), with the mounted
        accessories and, optionally, a support stand: a weld that holds the base where it starts."""
        from ..models.accessories import apply_to_spec
        spec = mujoco.MjSpec.from_file(self.custom_path)
        self.accessory_warnings = apply_to_spec(spec, self.accessories or [])
        if not self.support_stand:
            return spec.compile()
        base = spec.compile()
        if base.njnt == 0 or base.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            return base
        spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                          name1=base.body(int(base.jnt_bodyid[0])).name, name="support_stand")
        return spec.compile()

    def export_xml(self) -> str:
        if self.custom_path:
            return Path(self.custom_path).read_text()
        return self.cad_xml or self.generated_xml()

    # ---- obstacles -------------------------------------------------------
    def random_obstacles(self, n: int = 12, seed: int | None = None) -> None:
        from ..sim.scene import random_objects
        obs = random_objects(n, seed)
        with self.lock:
            self.obstacles, self.props, self.cad_xml = obs, [], None
            if not self.custom_path:
                self.build()

    def terrain_course(self, kind: str = "mixed", seed: int | None = None) -> tuple[float, float]:
        """Build a terrain course (steps, stairs, ramp, mixed) and return the goal just beyond it."""
        from ..sim.terrain import terrain_course
        obstacles, goal = terrain_course(kind, seed)
        with self.lock:
            self.obstacles, self.props, self.cad_xml = obstacles, [], None
            if not self.custom_path:
                self.build()
        return float(goal[0]), float(goal[1])

    def load_world(self, name: str, seed: int | None = 0) -> dict:
        """Build a ready-made world (sim/worlds.py: playground, parkour, kitchen) around the built-in robot.
        Returns its info: {"goal", "waypoints", "about"}."""
        from ..sim.worlds import WORLDS
        if name not in WORLDS:
            raise ValueError(f"unknown world {name!r} (choose from {sorted(WORLDS)})")
        if self.custom_path:
            raise ValueError("ready-made worlds are built around the built-in humanoid; load it first")
        obstacles, props, info = (WORLDS[name](seed, self.params) if name == "kitchen" else WORLDS[name](seed))
        with self.lock:
            self.obstacles, self.props, self.cad_xml = obstacles, props, None
            self.build()
        return info

    def manipulation_scene(self, seed: int | None = None) -> None:
        """A table in front of the robot with small graspable objects on it."""
        from ..sim.scene import manipulation_scene
        obstacles, props = manipulation_scene(seed)
        with self.lock:
            self.obstacles, self.props, self.cad_xml = obstacles, props, None
            if not self.custom_path:
                self.build()

    def clear_obstacles(self) -> None:
        with self.lock:
            self.obstacles, self.props, self.cad_xml = [], [], None
            if not self.custom_path:
                self.build()

    # ---- policy ----------------------------------------------------------
    def load_policy(self, path: str) -> None:
        from ..train.ppo import RLPolicy, is_rl_policy
        if is_rl_policy(path):
            pol = RLPolicy.load(path)
            with self.lock:
                meta = pol.meta or {}
                if not meta.get("custom_model") and meta.get("humanoid"):
                    self.params = HumanoidParams.from_dict(meta["humanoid"])
                    self.custom_path = None
                    self.cad_xml = None
                self.policy, self.rl_policy = None, pol
                self.policy_name = Path(path).parent.name
                self.controller_kind = "rl"
                self.build()  # _make_controller picks RLController (raises if joints differ)
                if getattr(self.controller, "name", "") != "rl":
                    self.rl_policy, self.policy_name, self.controller_kind = None, "", "auto"
                    self.controller = self._make_controller()
                    raise ValueError("learned policy does not match this robot (different leg joints)")
            return
        pol = Policy.load(path)
        with self.lock:
            self.rl_policy = None
            meta = pol.meta
            if not meta.get("custom_model") and "humanoid" in meta:
                # the policy was trained on (possibly evolved) body parameters: adopt them
                self.params = HumanoidParams.from_dict(meta["humanoid"])
                self.custom_path = None
            if "task" in meta:
                self.task_cfg = TaskConfig.from_dict(meta["task"])
            self.build()
            if self.task is None or pol.spec.sizes[0] != self.task.obs_dim or pol.spec.sizes[-1] != self.task.act_dim:
                raise ValueError("policy does not match this robot (different joints/actuators)")
            self.policy, self.policy_name = pol, Path(path).parent.name
            self.controller_kind = "es-policy"
            self.controller = self._make_controller()
            self.reset()

    def unload_policy(self) -> None:
        with self.lock:
            self.policy, self.policy_name = None, ""
            self.rl_policy = None
            self.imitation_policy = None
            if self.controller_kind in ("es-policy", "rl", "imitation", "motion-preview", "motion-pd"):
                self.controller_kind = "auto"
            self.controller = self._make_controller()
            self.reset()

    # ---- motors ----------------------------------------------------------
    def actuators(self) -> list[dict]:
        m, out = self.model, []
        for i in range(m.nu):
            servo = m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE
            jid = int(m.actuator_trnid[i, 0])
            dof = int(m.jnt_dofadr[jid]) if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT else -1
            out.append({
                "index": i, "name": m.actuator(i).name or f"actuator_{i}", "servo": bool(servo),
                "kp": float(m.actuator_gainprm[i, 0]), "kv": float(-m.actuator_biasprm[i, 2]),
                "torque": float(m.actuator_forcerange[i, 1]), "gear": float(m.actuator_gear[i, 0]),
                "lo": float(m.actuator_ctrlrange[i, 0]), "hi": float(m.actuator_ctrlrange[i, 1]),
                "damping": float(m.dof_damping[dof]) if dof >= 0 else 0.0,
                "armature": float(m.dof_armature[dof]) if dof >= 0 else 0.0,
                "ctrl": float(self.manual_ctrl[i]),
            })
        c = self.controller
        if getattr(c, "name", "") == "pd-hold":  # targets are joint angles held by the PD controller
            for k, i in enumerate(c.act):
                out[int(i)].update(ctrl=float(c.target[k]), lo=float(c.target_lo[k]), hi=float(c.target_hi[k]),
                                   servo=True)
        return out

    def set_actuator(self, index: int, field: str, value: float) -> None:
        """Edit a motor in the compiled model in place; takes effect on the next physics step."""
        with self.lock:
            m, i, v = self.model, int(index), float(value)
            if not 0 <= i < m.nu:
                raise IndexError("actuator index out of range")
            jid = int(m.actuator_trnid[i, 0])
            dof = int(m.jnt_dofadr[jid]) if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT else -1
            if field == "kp":
                m.actuator_gainprm[i, 0] = v
                if m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE:
                    m.actuator_biasprm[i, 1] = -v
            elif field == "kv":
                m.actuator_biasprm[i, 2] = -v
            elif field == "torque":
                m.actuator_forcerange[i] = (-abs(v), abs(v))
                m.actuator_forcelimited[i] = 1
            elif field == "gear":
                m.actuator_gear[i, 0] = v
            elif field == "damping" and dof >= 0:
                m.dof_damping[dof] = max(0.0, v)
            elif field == "armature" and dof >= 0:
                m.dof_armature[dof] = max(0.0, v)
            elif field == "ctrl":
                lo, hi = m.actuator_ctrlrange[i]
                self.manual_ctrl[i] = np.clip(v, lo, hi) if m.actuator_ctrllimited[i] else v
                if getattr(self.controller, "name", "") == "pd-hold":
                    self.controller.set_target(i, v)
            else:
                raise ValueError(f"unknown field {field}")

    # ---- sensors config --------------------------------------------------
    def set_lidar(self, d: dict) -> None:
        with self.lock:
            c = self.lidar_cfg
            c.enabled = bool(d.get("enabled", c.enabled))
            c.h_samples = int(np.clip(d.get("h_samples", c.h_samples), 8, 720))
            c.channels = int(np.clip(d.get("channels", c.channels), 1, 64))
            c.v_min = float(np.clip(d.get("v_min", c.v_min), -89, 89))
            c.v_max = float(np.clip(d.get("v_max", c.v_max), c.v_min, 89))
            c.max_range = float(np.clip(d.get("max_range", c.max_range), 1, 40))
            self.lidar = Lidar(self.model, c)
            if not c.enabled:
                self.lidar_points = np.zeros((0, 3), np.float32)
                self.lidar_seq += 1

    # ---- scene description for the 3D view -------------------------------
    def scene(self) -> dict:
        with self.lock:
            m = self.model
            geoms, meshes = [], {}
            for g in range(m.ngeom):
                t = int(m.geom_type[g])
                rgba = m.geom_rgba[g].tolist()
                if m.geom_matid[g] >= 0 and not self.custom_path is None:
                    rgba = m.mat_rgba[m.geom_matid[g]].tolist()
                robot = m.nbody > 1 and m.body_rootid[m.geom_bodyid[g]] == m.body_rootid[1] and m.geom_bodyid[g] != 0
                item = {"type": _GEOM_NAMES.get(t, "none"), "size": m.geom_size[g].tolist(), "rgba": rgba,
                        "group": int(m.geom_group[g]), "name": m.geom(g).name, "robot": bool(robot)}
                if t == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = int(m.geom_dataid[g])
                    item["mesh"] = mid
                    if mid not in meshes:
                        va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                        fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
                        meshes[mid] = {"vert": _b64(m.mesh_vert[va:va + vn].astype(np.float32)),
                                       "face": _b64(m.mesh_face[fa:fa + fn].astype(np.uint32))}
                geoms.append(item)
            return {"version": self.scene_version, "geoms": geoms, "meshes": meshes,
                    "cameras": [m.camera(i).name for i in range(m.ncam)],
                    "info": {"nq": m.nq, "nv": m.nv, "nu": m.nu, "nbody": m.nbody, "ngeom": m.ngeom,
                             "mass": float(m.body_subtreemass[1]) if m.nbody > 1 else 0.0,
                             "timestep": float(m.opt.timestep), "custom": self.custom_path,
                             "trainable": self.task is not None},
                    "grid": {"size": self.grid.size, "n": self.grid.n}}

    def frame(self) -> dict:
        with self.lock:
            d = self.data
            root = d.qpos[:7] if self.task else np.zeros(7)
            speed = float(np.hypot(d.qvel[0], d.qvel[1])) if self.task else 0.0
            touch = {}
            for i in range(self.model.nsensor):
                if self.model.sensor_type[i] == mujoco.mjtSensor.mjSENS_TOUCH:
                    a = self.model.sensor_adr[i]
                    touch[self.model.sensor(i).name] = round(float(d.sensordata[a]), 1)
            return {
                "v": self.scene_version, "t": round(float(d.time), 3),
                "pos": np.round(d.geom_xpos, 4).reshape(-1).tolist(),
                "quat": np.round(_mat2quat(d.geom_xmat), 4).reshape(-1).tolist(),
                "root": np.round(root, 3).tolist(), "speed": round(speed, 2),
                "running": self.running, "sim_speed": self.speed, "falls": self.falls,
                "policy": self.policy_name, "heading": round(self.heading, 3),
                "touch": touch, "nav": self.nav.to_dict(), "error": self.error,
                "camera_error": self.camera_error,
                "controller": {"name": getattr(self.controller, "name", "none"), "kind": self.controller_kind,
                               **_jsonable(self.controller.status() if self.controller else {})},
                "arms": _jsonable(self.arms.status()) if self.arms is not None else None,
                "pilot": self.pilot.brief(),
                "slam": self.slam.to_dict() if self.task is not None and self.slam.scans else None,
                "slam_true": np.round(np.asarray(self.slam_true[::max(1, len(self.slam_true) // 400)]), 3).tolist(),
                "nav_on_slam": self.nav_on_slam,
            }

    # ---- simulation thread -----------------------------------------------
    def shutdown(self, timeout: float = 3.0) -> None:
        """Stop the simulation thread; it frees its own GL renderers before exiting."""
        self._alive = False
        if threading.current_thread() is not self._thread and self._thread.is_alive():
            self._thread.join(timeout)

    def push(self, direction: str = "left", force: float = 60.0, duration: float = 0.1) -> str:
        """Shove the torso: a horizontal force in the robot's frame for `duration` seconds of sim time."""
        dirs = {"forward": (1, 0), "back": (-1, 0), "left": (0, 1), "right": (0, -1)}
        if direction not in dirs:
            raise ValueError(f"direction must be one of {list(dirs)}")
        force = float(np.clip(force, 0.0, 1000.0))
        duration = float(np.clip(duration, 0.01, 1.0))
        with self.lock:
            if self.task is None:
                raise ValueError("this model has no free-floating body to push")
            _, _, yaw = base_pose(self.data)
            fx, fy = dirs[direction]
            c, s = np.cos(yaw), np.sin(yaw)
            self._push = (float(self.data.time) + duration, np.array([c * fx - s * fy, s * fx + c * fy, 0.0]) * force)
            self.running = True
        return f"pushing {direction} with {force:.0f} N for {duration:.2f} s"

    def step_once(self) -> None:
        self._step_once = True

    def _control_step(self) -> None:
        m, d, task, c = self.model, self.data, self.task, self.controller
        if task is not None:
            x, y, yaw = base_pose(d)
            if self.nav_on_slam:
                # plan where SLAM thinks the robot is; convert paths into the controller's (true) frame
                ex, ey, eyaw = self.slam.pose
                self.nav.tick((ex, ey), eyaw, c.status())
                new = self.nav.take_new_path()
                if new is not None:
                    path, gyaw = new
                    tp = (x, y, yaw)
                    c.set_path(self.slam.to_controller_frame(path, tp) if len(path) else path,
                               None if gyaw is None else gyaw + yaw - eyaw)
            else:
                self.nav.tick((x, y), yaw, c.status())
                new = self.nav.take_new_path()
                if new is not None:
                    c.set_path(*new)
            if isinstance(c, PolicyController):
                c.heading = self.heading
        c.step(m, d)
        if self.arms is not None and getattr(c, "name", "") not in ("imitation", "motion-preview", "motion-pd",
                                                                    "motion-physical"):
            # Only arms that received a command write ctrl, so idle arms keep the Motors-tab targets.
            # The arm IK runs at its own (lower) rate; it times itself from data.time.
            self._arm_tick += 1
            every = max(1, int(round(self.arms.control_dt / c.control_dt)))
            if self._arm_tick % every == 0:
                self.arms.step(m, d)
        root = 1 if m.nbody > 1 else 0
        if self._push is not None:
            end, f = self._push
            if d.time < end:
                d.xfrc_applied[root, :3] = f
            else:
                d.xfrc_applied[root, :3] = 0.0
                self._push = None
        mujoco.mj_step(m, d, max(1, int(round(c.control_dt / m.opt.timestep))))
        i = self._trace_act
        if i is not None and i < m.nu:
            j = int(m.actuator_trnid[i, 0])
            self.motor_trace.append((float(d.time), float(d.ctrl[i]), float(d.qpos[m.jnt_qposadr[j]]),
                                     float(d.qvel[m.jnt_dofadr[j]]), float(d.actuator_force[i])))
        if getattr(c, "name", "") == "motion-physical":
            # A recorded physical motion may be upside down on purpose (a flip): no fall check while it plays.
            # When it ends the balance controller takes over the robot as it stands (no reset: it keeps its
            # pose and speed), and falls count again. Holding the last command instead does not work: without
            # balance feedback the robot topples within a second.
            if c.done:
                self._end_tape()
            return
        if task is None or isinstance(c, ManualController) or getattr(c, "name", "") == "motion-preview":
            return      # a kinematic preview cannot fall (flips and handstands are upside down on purpose)
        _, up = LocoTask.yaw_up(d.qpos[None])
        fallen = (d.qpos[2] < task.cfg.min_height_frac * task.z0 or up[0, 2] < task.cfg.min_up
                  or not np.isfinite(d.qpos).all())
        if fallen:
            self.falls += 1
            had_goal = bool(self.nav.goals)
            if self.auto_reset:
                self.reset()
            if had_goal:
                self.nav.state, self.nav.message = "fallen", "robot fell"

    def _sensors(self, now: float) -> None:
        self.robot.serve_render_requests()  # camera / segmentation frames requested by AI pilots
        if not self.lidar.mounted and self.lidar_points is not None and len(self.lidar_points):
            self.lidar_points = np.zeros((0, 3), np.float32)
            self.lidar_seq += 1
        if self.lidar.mounted and self.lidar_cfg.enabled and now - self._t_lidar >= 1.0 / self.lidar_cfg.rate_hz:
            self._t_lidar = now
            origin, pts, _, hit = self.lidar.scan(self.model, self.data)
            self.lidar_points = pts[hit].astype(np.float32)
            self.lidar_seq += 1
            self.grid.integrate_lidar(origin, pts, hit, self.lidar_cfg.h_samples)
            if self.lidar_seq % 5 == 0:
                self.map_seq += 1
        # SLAM runs on simulation time (10 Hz), so fast-forwarding does not stretch the motion between scans
        if (self.task is not None and self.lidar.mounted and self.lidar_cfg.enabled
                and self.data.time - self._slam_t >= 0.1 - 1e-9):
            from ..plan.slam import planar_scan, root_pose
            self._slam_t = float(self.data.time)
            origin, pts, _, hit = self.lidar.scan(self.model, self.data)
            sc = planar_scan(self.data, origin, pts, hit, self.lidar_cfg.h_samples, self.lidar.root)
            tp = root_pose(self.data, self.lidar.root)
            odom = self.slam.odometry.update(tp)          # simulated leg odometry (with drift)
            self.slam.update(odom, sc.points, sc.hit, sc.origin, match=sc.match)
            self.slam_true.append(tp[:2].copy())
            del self.slam_true[:-4000]
            if self.slam.scans % 5 == 0 and self.nav_on_slam:
                self.map_seq += 1
        # at most 10 Hz, and never more than a quarter of the loop's time (heavy meshes render slowly, and
        # rendering holds the simulation)
        if self.camera_cfg["enabled"] and now - self._t_cam >= max(0.1, 4.0 * getattr(self, "_cam_cost", 0.0)):
            self._t_cam = now
            t_render = time.perf_counter()
            try:
                if self._rig is None or self._rig_version != self.scene_version:
                    if self._rig:
                        self._rig.close()
                    self._rig, self._rig_version = CameraRig(self.model), self.scene_version
                name = self.camera_cfg["name"]
                cam = name if name in self._rig.names else -1
                self.camera_jpeg = self._rig.jpeg(self.data, cam, self.camera_cfg["depth"])
                self.camera_seq += 1
                self.camera_error = ""
                self._cam_cost = time.perf_counter() - t_render
            except Exception as e:
                self.camera_error = f"{type(e).__name__}: {e}"
                self.camera_cfg["enabled"] = False

    def _loop(self) -> None:
        self._t_lidar = self._t_cam = 0.0
        next_t = time.perf_counter()
        while self._alive:
            c, m = self.controller, self.model
            dt = max(1, int(round(c.control_dt / m.opt.timestep))) * float(m.opt.timestep)
            try:
                with self.lock:
                    if self.running or self._step_once:
                        self._step_once = False
                        self._control_step()
                    self._sensors(time.perf_counter())
                self.error = ""
            except Exception as e:
                self.error, self.running = f"{type(e).__name__}: {e}", False
            next_t += dt / max(self.speed, 0.05) if self.running else 1 / 30
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()  # running behind real time: don't accumulate debt
        # GL contexts belong to this thread: free them here, not from the garbage collector elsewhere
        self.robot.close()
        if self._rig is not None:
            try:
                self._rig.close()
            except Exception:  # noqa: BLE001
                pass
            self._rig = None


def param_spec() -> list[dict]:
    return [{"name": k, "min": lo, "max": hi, "default": dv, "label": label, "group": group}
            for k, (lo, hi, dv, label, group) in PARAM_SPEC.items()]
