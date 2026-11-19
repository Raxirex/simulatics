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
        self._arm_tick = 0
        self._push = None                # (end sim time, world force) while a push is applied
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
                model = mujoco.MjModel.from_xml_path(self.custom_path)
            elif self.cad_xml:
                model = mujoco.MjModel.from_xml_string(self.cad_xml)
            else:
                model = mujoco.MjModel.from_xml_string(self.generated_xml())
            self.model, self.data = model, mujoco.MjData(model)
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
        """MJCF of the built-in robot with the current world objects and props."""
        if self.props:
            return build_mjcf(self.params, self.obstacles, props=self.props)
        return build_mjcf(self.params, self.obstacles)

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
        if self.task is None:
            return ManualController(m, self.manual_ctrl)
        if self.policy is not None and kind in ("auto", "es-policy"):
            return PolicyController(m, self.policy, self.task)
        if getattr(self, "rl_policy", None) is not None and kind in ("auto", "rl"):
            from ..train.ppo import RLController
            return RLController(m, self.rl_policy)
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
        return ManualController(m, self.manual_ctrl)

    def set_controller(self, kind: str) -> None:
        with self.lock:
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

    def set_params(self, d: dict) -> None:
        with self.lock:
            self.params = HumanoidParams.from_dict({**self.params.to_dict(), **d})
            self.custom_path = None
            self.cad_xml = None
            self.build()

    def load_file(self, path: str) -> None:
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"no such file: {p}")
        with self.lock:
            old = self.custom_path
            self.custom_path = str(p)
            try:
                self.build()
            except Exception:
                self.custom_path = old
                self.build()
                raise
            self.policy, self.policy_name = None, ""

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
            if self.controller_kind in ("es-policy", "rl"):
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
            self.nav.tick((x, y), yaw, c.status())
            new = self.nav.take_new_path()
            if new is not None:
                c.set_path(*new)
            if isinstance(c, PolicyController):
                c.heading = self.heading
        c.step(m, d)
        if self.arms is not None:
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
        if task is None or isinstance(c, ManualController):
            return
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
        if self.lidar_cfg.enabled and now - self._t_lidar >= 1.0 / self.lidar_cfg.rate_hz:
            self._t_lidar = now
            origin, pts, _, hit = self.lidar.scan(self.model, self.data)
            self.lidar_points = pts[hit].astype(np.float32)
            self.lidar_seq += 1
            self.grid.integrate_lidar(origin, pts, hit, self.lidar_cfg.h_samples)
            if self.lidar_seq % 5 == 0:
                self.map_seq += 1
        if self.camera_cfg["enabled"] and now - self._t_cam >= 0.1:
            self._t_cam = now
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
