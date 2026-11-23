"""HTTP + WebSocket API for the GUI (FastAPI). Serves the static front end too."""
from __future__ import annotations

import asyncio
import io
import math
import struct
import tempfile
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import native, __version__
from ..train.evolve import Trainer, TrainConfig
from ..train.policy import Policy
from .session import SimSession, param_spec

STATIC = Path(__file__).parent / "static"
RUNS = Path("runs")

# GUI field defaults per training method (the Train tab swaps them in when the method changes)
TRAIN_PRESETS = {
    "ppo": {"population": 256, "generations": 1500, "lr": 0.001, "threads": 8, "episode_seconds": 20},
    "distill": {"population": 16, "generations": 14, "lr": 0.001, "threads": 5, "episode_seconds": 25},
    "imitation": {"population": 64, "generations": 1500, "lr": 0.001, "threads": 4, "episode_seconds": 10},
    "es": {"population": 256, "generations": 200, "lr": 0.005, "sigma": 0.01, "threads": 0, "episode_seconds": 8},
    "ga": {"population": 256, "generations": 200, "lr": 0.005, "sigma": 0.01, "threads": 0, "episode_seconds": 8},
}


def _finite(v):
    """JSON cannot carry NaN/inf: turn them into None, recursively."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite(x) for x in v]
    return v


class _StopDistill(Exception):
    pass


class _DistillJob:
    """Background distillation (model-based walker -> neural policy) with the Trainer status shape."""

    def __init__(self, model_xml: str, params, name: str, rounds: int, envs: int, workers: int = 5,
                 seconds: float = 25.0):
        import threading
        self.name, self.rounds, self.envs, self.workers, self.seconds = name, rounds, envs, workers, seconds
        self.model_xml, self.params = model_xml, params
        self.state, self.error, self.history = "idle", "", []
        self.policy_path = RUNS / name / "rl_policy.npz"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.state = "running"
        self._thread.start()

    def stop(self):
        # distill() has no stop hook: the per-round callback raises, so it stops after the current round
        self._stop.set()

    def _run(self):
        try:
            import mujoco
            from ..train.ppo import DistillConfig, distill, walker_teacher_factory
            # the recipe that produced runs/rl_walker (40/40 goals): fixed-timing walker able to reach 0.4 m/s
            factory = walker_teacher_factory(step_forward=0.32)
            if factory is None:
                raise RuntimeError("the model-based walker is not available for this robot")
            model = mujoco.MjModel.from_xml_string(self.model_xml)
            cfg = DistillConfig(rounds=self.rounds, num_envs=self.envs, workers=self.workers,
                                episode_seconds=self.seconds, grad_steps=3000, batch_size=1024,
                                hidden=[512, 256, 128], lr=1e-3, lr_final=1e-4, keep_every=3,
                                buffer_size=450_000, beta_decay=0.5, name=self.name)

            def cb(row):
                # lower is better for imitation error, so chart its negative as the "score"
                self.history.append({"gen": len(self.history) + 1, "best": -row["train_mse"],
                                     "mean": -(row.get("rollout_disagreement") or row["train_mse"]),
                                     "center": -row["train_mse"], "sps": None, "falls": row.get("falls"),
                                     "samples": row.get("samples")})
                if self._stop.is_set():
                    raise _StopDistill()

            distill(model, factory, cfg=cfg, run_dir=RUNS / self.name,
                    model_meta={"humanoid": self.params.to_dict(), "custom_model": False}, callback=cb)
            self.state = "finished"
        except _StopDistill:
            self.state = "stopped"
        except Exception as e:  # noqa: BLE001
            self.state, self.error = "error", f"{type(e).__name__}: {e}"

    def status(self):
        state = "stopping after this round" if self._stop.is_set() and self.state == "running" else self.state
        return {"state": state, "error": self.error, "name": self.name, "algo": "distill", "kind": "distill",
                "generation": len(self.history), "generations": self.rounds, "best": None, "total_steps": 0,
                "history": self.history, "policy": str(self.policy_path) if self.policy_path.exists() else None}


def create_app() -> FastAPI:
    app = FastAPI(title="Simulatics")
    sess = SimSession()
    state: dict = {"trainer": None, "cad": None, "cad_opts": None, "topo": None, "opt": None, "ros": None}

    def guard(fn, *a, **k):
        try:
            return fn(*a, **k)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"{type(e).__name__}: {e}")

    def policies() -> list[dict]:
        from ..train.ppo import RLPolicy, is_rl_policy
        out = []
        for p in sorted(list(RUNS.glob("*/policy.npz")) + list(RUNS.glob("*/rl_policy.npz"))
                        + list(RUNS.glob("*/imitation_policy.npz")), key=lambda p: -p.stat().st_mtime):
            try:
                if p.name == "imitation_policy.npz":
                    from ..imitate import is_imitation_policy
                    if not is_imitation_policy(p):
                        continue
                    ev = RUNS / p.parent.name / "evaluation.json"
                    meta, kind = {"algo": "imitation"}, "imitation"
                    if ev.exists():
                        import json
                        e = json.loads(ev.read_text())
                        meta["fitness"] = e.get("reward_per_step", e.get("mean_reward"))
                elif p.name == "rl_policy.npz":
                    if not is_rl_policy(p):
                        continue
                    meta = RLPolicy.load(p).meta or {}
                    kind = meta.get("algo") or ("distilled" if meta.get("distill") else "ppo")
                else:
                    meta, kind = Policy.load(p).meta, None
            except Exception:
                continue
            out.append({"name": p.parent.name, "path": str(p), "fitness": meta.get("fitness", meta.get("best")),
                        "generation": meta.get("generation", meta.get("iteration")), "algo": meta.get("algo"),
                        "kind": kind or meta.get("algo", "es"), "custom_model": meta.get("custom_model", False)})
        return _finite(out)

    def train_status() -> dict:
        tr = state["trainer"]
        if not tr:
            return {"state": "idle", "history": [], "generation": 0, "generations": 0}
        st = tr.status()
        st.setdefault("kind", st.get("algo", "es"))
        return _finite(st)

    def cad_robot():
        """CAD model of the current robot (built on demand with the last options)."""
        from ..design.cad import CadOptions, build_robot_cad
        if state["cad"] is None:
            state["cad_opts"] = state["cad_opts"] or CadOptions()
            state["cad"] = build_robot_cad(sess.params, state["cad_opts"], xml=sess.generated_xml())
        return state["cad"]

    def full_state():
        return {"params": sess.params.to_dict(), "scene": sess.scene(), "actuators": sess.actuators()}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    def get_state():
        from ..design.optimizer import design_genes
        from ..llm import options as pilot_options
        return {"version": __version__, "native": native.status(), "param_spec": param_spec(),
                "params": sess.params.to_dict(), "scene": sess.scene(), "actuators": sess.actuators(),
                "lidar": sess.lidar_cfg.to_dict(), "camera": sess.camera_cfg, "task": sess.task_cfg.to_dict(),
                "train_defaults": asdict(TrainConfig()), "train_presets": TRAIN_PRESETS, "train": train_status(),
                "policies": policies(), "obstacles": len(sess.obstacles), "design_genes": design_genes(),
                "pilot_options": pilot_options()}

    @app.post("/api/params")
    def set_params(d: dict = Body(...)):
        guard(sess.set_params, d)
        state["cad"] = None
        return full_state()

    @app.post("/api/actuator")
    def set_actuator(d: dict = Body(...)):
        guard(sess.set_actuator, d["index"], d["field"], d["value"])
        return {"ok": True}

    @app.get("/api/actuators")
    def get_actuators():
        return sess.actuators()

    @app.post("/api/sim")
    def sim(d: dict = Body(...)):
        a = d.get("action")
        if a == "play":
            sess.running = True
        elif a == "pause":
            sess.running = False
        elif a == "reset":
            sess.reset()
            sess.grid.reset_lidar()
            sess.falls = 0
        elif a == "step":
            sess.step_once()
        if "speed" in d:
            sess.speed = float(np.clip(d["speed"], 0.05, 8))
        if "heading" in d:
            sess.heading = float(d["heading"])
        if "auto_reset" in d:
            sess.auto_reset = bool(d["auto_reset"])
        return {"running": sess.running}

    @app.post("/api/obstacles")
    def obstacles(d: dict = Body(...)):
        if d.get("action") == "random":
            guard(sess.random_obstacles, int(np.clip(d.get("n", 12), 1, 60)), d.get("seed"))
        elif d.get("action") == "terrain":
            kind = str(d.get("kind", "mixed"))
            gx, gy = guard(sess.terrain_course, kind, d.get("seed"))
            state["cad"] = None
            return {"scene": sess.scene(), "obstacles": len(sess.obstacles), "props": 0, "goal": [gx, gy]}
        elif d.get("action") == "manipulation":
            if not sess.params.hands or not sess.params.arms:
                guard(sess.set_params, {"arms": True, "hands": True})
            guard(sess.manipulation_scene, d.get("seed"))
        else:
            guard(sess.clear_obstacles)
        state["cad"] = None
        return {"scene": sess.scene(), "obstacles": len(sess.obstacles), "props": len(sess.props)}

    @app.post("/api/push")
    def push(d: dict = Body(...)):
        return {"message": guard(sess.push, str(d.get("direction", "left")), float(d.get("force", 60)),
                                 float(d.get("duration", 0.1)))}

    @app.get("/api/objects")
    def objects():
        return guard(sess.robot.objects)

    @app.post("/api/arms")
    def arms(d: dict = Body(...)):
        if sess.arms is None:
            raise HTTPException(400, "this robot has no hands: enable Arms and Hands in the Model tab")
        side, skill = str(d.get("side", "auto")), str(d.get("skill", ""))
        kw = {}

        def side_for(x: float, y: float) -> str:
            """The hand on the target's side of the body (each arm works on its own side)."""
            from ..control import base_pose
            with sess.lock:
                bx, by, yaw = base_pose(sess.data)
            left = -math.sin(yaw) * (x - bx) + math.cos(yaw) * (y - by)
            return "left" if left > 0 else "right"

        if skill == "reach_object":
            obj = next((o for o in sess.robot.objects() if o["name"] == d.get("object")), None)
            if obj is None:
                raise HTTPException(400, f"no object named {d.get('object')!r}")
            skill = "reach"
            kw["target"] = [obj["x"], obj["y"], obj.get("z", 0.95)]
            if obj.get("graspable"):
                kw["approach"] = [0.0, 0.0, -1.0]
        elif "target" in d:
            t = [float(v) for v in d["target"]]
            if len(t) != 3 or not all(math.isfinite(v) for v in t):
                raise HTTPException(400, "target must be three finite numbers")
            kw["target"] = t
        if side == "auto":
            if "target" in kw:
                side = side_for(kw["target"][0], kw["target"][1])
            elif skill in ("grasp", "release"):
                side = state.get("arm_side", "right")  # the hand that reached last
            else:
                side = "both" if skill == "home" else state.get("arm_side", "right")
        if skill in ("reach", "point_at"):
            state["arm_side"] = side
        with sess.lock:
            sess.running = True
            msg = guard(sess.arms.command, side, skill, **kw)
        return {"message": msg}

    @app.post("/api/lidar")
    def lidar(d: dict = Body(...)):
        guard(sess.set_lidar, d)
        return sess.lidar_cfg.to_dict()

    @app.post("/api/camera")
    def camera(d: dict = Body(...)):
        for k in ("enabled", "name", "depth"):
            if k in d:
                sess.camera_cfg[k] = d[k]
        return sess.camera_cfg

    @app.post("/api/goal")
    def goal(d: dict = Body(...)):
        yaw = d.get("yaw_deg")
        with sess.lock:
            sess.nav.set_goal([float(d["x"]), float(d["y"])], None if yaw is None else np.deg2rad(float(yaw)),
                              append=bool(d.get("append")))
        return sess.nav.to_dict()

    @app.post("/api/controller")
    def controller(d: dict = Body(...)):
        kind = d.get("kind", "auto")
        if kind not in ("auto", "walker", "mpc", "rl", "es-policy", "manual"):
            raise HTTPException(400, "unknown controller")
        guard(sess.set_controller, kind)
        return {"name": sess.controller.name, "kind": sess.controller_kind}

    @app.post("/api/nav")
    def nav(d: dict = Body(...)):
        with sess.lock:
            if d.get("action") == "clear":
                sess.nav.clear()
            if d.get("source") in ("truth", "lidar", "slam"):
                sess.nav_on_slam = d["source"] == "slam"
                # SLAM: plan on the map built from the estimated pose (its log-odds layer acts as "lidar")
                sess.nav.grid = sess.slam_grid if sess.nav_on_slam else sess.grid
                sess.nav.source = "lidar" if sess.nav_on_slam else d["source"]
                sess.nav.path = np.zeros((0, 2))
                sess.map_seq += 1
            if d.get("action") == "reset_map":
                sess.grid.reset_lidar()
                sess.reset_slam()
        return sess.nav.to_dict()

    @app.post("/api/model/load")
    def model_load(d: dict = Body(...)):
        guard(sess.load_file, str(d["path"]).strip().strip('"'))
        return {"scene": sess.scene(), "actuators": sess.actuators()}

    @app.post("/api/model/builtin")
    def model_builtin():
        guard(sess.set_params, {})
        return {"params": sess.params.to_dict(), "scene": sess.scene(), "actuators": sess.actuators()}

    @app.get("/api/model/export", response_class=PlainTextResponse)
    def model_export():
        return PlainTextResponse(sess.export_xml(), headers={
            "Content-Disposition": 'attachment; filename="humanoid.xml"'})

    @app.post("/api/task")
    def set_task(d: dict = Body(...)):
        with sess.lock:
            for k, v in d.items():
                if hasattr(sess.task_cfg, k):
                    setattr(sess.task_cfg, k, type(getattr(sess.task_cfg, k))(v))
            guard(sess.build)
        return sess.task_cfg.to_dict()

    # ---- training --------------------------------------------------------
    @app.post("/api/train/start")
    def train_start(d: dict = Body(...)):
        tr = state["trainer"]
        if tr and tr.state == "running":
            raise HTTPException(400, "training already running")
        if sess.task is None:
            raise HTTPException(400, "this model has no free-floating root or no actuators, cannot train locomotion")
        algo = d.get("algo", "es")
        name = "".join(c for c in str(d.get("name") or "run") if c.isalnum() or c in "-_") or "run"
        if algo == "ppo":
            from ..train.ppo import PPOConfig, PPOTrainer
            from ..sim.vel_task import VelTaskConfig
            cfg = PPOConfig(num_envs=int(np.clip(d.get("population", 256), 4, 4096)),
                            iterations=int(np.clip(d.get("generations", 1500), 1, 1000000)),
                            learning_rate=float(np.clip(d.get("lr", 1e-3), 1e-6, 0.1)),
                            threads=int(d.get("threads") or 8), name=name)
            init = None
            if d.get("continue_from_loaded") and sess.rl_policy is not None and sess.policy_name:
                ck = RUNS / sess.policy_name / "rl_checkpoint.npz"
                init = ck if ck.exists() else None
            tr = guard(PPOTrainer, cfg, sess.params, VelTaskConfig(), model_xml=sess.custom_path, runs_dir=RUNS,
                       init_checkpoint=init)
        elif algo == "imitation":
            from ..imitate import ImitationTrainer, imitation_ppo_config
            cfg = imitation_ppo_config(name=name, num_envs=int(np.clip(d.get("population", 64), 4, 1024)),
                                       iterations=int(np.clip(d.get("generations", 1500), 1, 1000000)),
                                       threads=int(d.get("threads") or 4))
            motion = str(d.get("motion") or "walk")
            mpath = RUNS / "motions" / f"{motion}.npz"
            tr = guard(ImitationTrainer, cfg, str(mpath) if mpath.exists() else motion, base=sess.params,
                       runs_dir=RUNS)
        elif algo == "distill":
            tr = _DistillJob(sess.export_xml(), sess.params, name, int(np.clip(d.get("generations", 14), 1, 50)),
                             int(np.clip(d.get("population", 16), 1, 64)), int(np.clip(d.get("threads") or 5, 1, 16)),
                             float(np.clip(d.get("episode_seconds", 25), 2, 120)))
        elif algo in ("es", "ga"):
            cfg = TrainConfig.from_dict(d)
            cfg.name = name
            cfg.population = int(np.clip(cfg.population, 8, 4096))
            cfg.generations = int(np.clip(cfg.generations, 1, 100000))
            init = sess.policy if d.get("continue_from_loaded") else None
            tr = Trainer(cfg, sess.params, sess.task_cfg, model_xml=sess.custom_path, runs_dir=RUNS, init_policy=init)
        else:
            raise HTTPException(400, "algo must be ppo, distill, imitation, es or ga")
        state["trainer"] = tr
        tr.start()
        return train_status()

    @app.post("/api/train/stop")
    def train_stop():
        if state["trainer"]:
            state["trainer"].stop()
        return train_status()

    @app.get("/api/train/status")
    def get_train_status():
        return train_status()

    @app.get("/api/policies")
    def get_policies():
        return policies()

    @app.post("/api/policy/load")
    def policy_load(d: dict = Body(...)):
        p = Path(d["path"]).resolve()
        if RUNS.resolve() not in p.parents or p.suffix != ".npz":
            raise HTTPException(400, "policy must be a .npz inside the runs folder")
        if p.name == "imitation_policy.npz":
            guard(sess.play_motion, None, "policy", str(p))
            sess.policy_name = p.parent.name
            return {**full_state(), "policy": p.parent.name}
        guard(sess.load_policy, str(p))
        state["cad"] = None
        return {**full_state(), "policy": sess.policy_name}

    @app.post("/api/policy/unload")
    def policy_unload():
        sess.unload_policy()
        return {"ok": True}

    # ---- workshop: accessories ---------------------------------------------
    @app.post("/api/workshop")
    def workshop(d: dict = Body(...)):
        guard(sess.workshop, bool(d.get("enter", True)))
        return sess.accessories_info()

    @app.get("/api/accessories")
    def accessories():
        return _finite(sess.accessories_info())

    @app.post("/api/accessories/place")
    def accessories_place(d: dict = Body(...)):
        acc = guard(sess.place_accessory, str(d.get("type", "")), int(d.get("geom", -1)), d.get("point"),
                    d.get("normal"), float(d.get("snap", 0.01)), d.get("id"))
        state["cad"] = None
        return _finite({"placed": acc, **sess.accessories_info(), **full_state()})

    @app.post("/api/accessories/update")
    def accessories_update(d: dict = Body(...)):
        guard(sess.update_accessory, str(d.get("id", "")), d)
        state["cad"] = None
        return _finite({**sess.accessories_info(), **full_state()})

    @app.post("/api/accessories/delete")
    def accessories_delete(d: dict = Body(...)):
        guard(sess.delete_accessory, str(d.get("id", "")))
        state["cad"] = None
        return _finite({**sess.accessories_info(), **full_state()})

    # ---- motion imitation ----------------------------------------------------
    @app.get("/api/motions")
    def motions():
        from ..imitate import list_motions
        return _finite(list_motions(extra_dirs=[RUNS / "motions"]))

    @app.post("/api/motions/play")
    def motions_play(d: dict = Body(...)):
        mode = str(d.get("mode", "preview"))
        policy = d.get("policy")
        if policy:
            p = Path(policy).resolve()
            if RUNS.resolve() not in p.parents or p.suffix != ".npz":
                raise HTTPException(400, "policy must be a .npz inside the runs folder")
            policy = str(p)
        motion = d.get("motion")
        if motion and (RUNS / "motions" / f"{motion}.npz").exists():
            motion = str(RUNS / "motions" / f"{motion}.npz")  # an imported clip (library clips resolve by name)
        msg = guard(sess.play_motion, motion, mode, policy)
        return {"message": msg, "controller": sess.controller.name}

    @app.post("/api/motions/import")
    def motions_import(d: dict = Body(...)):
        from ..imitate import retarget_bvh
        src = Path(str(d.get("path", "")).strip().strip('"')).expanduser()
        if not src.is_file() or src.suffix.lower() != ".bvh":
            raise HTTPException(400, f"not a .bvh file: {src}")
        name = "".join(c for c in str(d.get("name") or src.stem) if c.isalnum() or c in "-_") or "imported"
        with sess.lock:
            model = sess.model
        res = guard(retarget_bvh, src, model)
        out = RUNS / "motions" / f"{name}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        res.motion.save(out)
        return _finite({"name": name, "path": str(out), "report": res.report})

    # ---- ROS 2 bridge --------------------------------------------------------
    def ros_status() -> dict:
        from ..bridge import rclpy_available
        b = state.get("ros")
        st = b.status() if b is not None else {"state": "idle", "connected": False, "topics": {}}
        st["rclpy_available"] = rclpy_available()
        return _finite(st)

    @app.post("/api/ros/connect")
    def ros_connect(d: dict = Body(...)):
        from ..bridge import start_bridge
        old = state.get("ros")
        if old is not None:
            old.stop()  # always a fresh bridge per connect (a stopped one is not restarted)
        url = str(d.get("url") or "ws://localhost:9090").strip()
        if not url.startswith(("ws://", "wss://")):
            raise HTTPException(400, "the rosbridge URL must start with ws:// or wss://")
        state["ros"] = guard(start_bridge, sess, url, mode=str(d.get("mode", "rosbridge")))
        return ros_status()

    @app.post("/api/ros/disconnect")
    def ros_disconnect():
        b = state.get("ros")
        if b is not None:
            b.stop()  # sync endpoint: runs in the threadpool, not on the event loop
            state["ros"] = None
        return ros_status()

    @app.get("/api/ros/status")
    def ros_status_ep():
        return ros_status()

    # ---- AI pilot ----------------------------------------------------------
    @app.post("/api/pilot/start")
    def pilot_start(d: dict = Body(...)):
        body = dict(d)  # never log this: it may hold an API key
        mission = body.pop("mission", "")
        try:
            sess.pilot.start(mission, body)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        with sess.lock:
            sess.running = True
        return sess.pilot.brief()

    @app.post("/api/pilot/stop")
    def pilot_stop():
        sess.pilot.stop()
        return sess.pilot.brief()

    @app.get("/api/pilot/events")
    def pilot_events(since: int = 0):
        return sess.pilot.events(int(since))

    @app.get("/api/pilot/options")
    def pilot_options_ep():
        from ..llm import options
        return options()

    # ---- design: CAD ---------------------------------------------------------
    @app.post("/api/design/cad")
    def design_cad(d: dict = Body(...)):
        from ..design.cad import CadOptions, build_robot_cad
        opts = guard(CadOptions.from_dict, d)
        state["cad_opts"] = opts
        state["cad"] = guard(build_robot_cad, sess.params, opts, xml=sess.generated_xml())
        return _finite(state["cad"].summary())

    @app.post("/api/design/cad/use")
    def design_cad_use(d: dict = Body(...)):
        from ..design.cad import CadOptions, build_robot_cad
        if d:
            state["cad_opts"] = guard(CadOptions.from_dict, d)
            state["cad"] = guard(build_robot_cad, sess.params, state["cad_opts"], xml=sess.generated_xml())
        robot = guard(cad_robot)
        with sess.lock:
            sess.cad_xml = guard(robot.mjcf, sess.generated_xml())
            guard(sess.build)
        return full_state()

    @app.get("/api/design/export")
    def design_export(fmt: str = "stl"):
        from ..design.cad import export_urdf
        robot = guard(cad_robot)
        if fmt == "urdf":
            urdf = guard(export_urdf, sess.params, state["cad_opts"], xml=sess.generated_xml(), cad=robot,
                         mesh_prefix="meshes/")
            buf = io.BytesIO()
            with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("humanoid.urdf", urdf)
                for f in guard(robot.export, tmp, ("stl",)):
                    z.write(f, f"meshes/{Path(f).name}")
            return Response(buf.getvalue(), media_type="application/zip",
                            headers={"Content-Disposition": 'attachment; filename="humanoid_urdf.zip"'})
        if fmt not in ("stl", "obj", "glb", "step"):
            raise HTTPException(400, "fmt must be stl, obj, glb, step or urdf")
        buf = io.BytesIO()
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            files = guard(robot.export, tmp, (fmt,))
            if not files:
                raise HTTPException(400, f"{fmt.upper()} export is not available on this machine")
            for f in files:
                z.write(f, Path(f).name)
        return Response(buf.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="humanoid_{fmt}.zip"'})

    # ---- design: topology optimisation ---------------------------------------
    @app.post("/api/design/topopt/start")
    def topopt_start(d: dict = Body(...)):
        from ..design.linkopt import LinkOptConfig, TopOptJob
        job = state["topo"]
        if job is not None and job.status().get("state") == "running":
            raise HTTPException(409, "a topology optimisation is already running")
        link = str(d.get("link", "thigh"))
        nel = int(d.get("nel", 40))
        nel += nel % 2  # the multigrid solver needs even grid sizes
        cfg_d = {**d, "nel": nel}
        if state["cad_opts"] is not None:
            cfg_d.update(wall_mm=state["cad_opts"].wall * 1000, density=state["cad_opts"].mat.density)
        cfg = guard(LinkOptConfig.from_dict, cfg_d)
        # loads come from walking on open ground: obstacle impacts are not design loads
        from ..models import build_mjcf
        job = guard(TopOptJob, sess.params, link, cfg, xml=build_mjcf(sess.params))
        state["topo"] = job
        job.start()
        return _finite(job.status())

    @app.get("/api/design/topopt/status")
    def topopt_status():
        job = state["topo"]
        return _finite(job.status()) if job is not None else {"state": "idle", "history": []}

    @app.post("/api/design/topopt/stop")
    def topopt_stop():
        if state["topo"] is not None:
            state["topo"].stop()
        return topopt_status()

    @app.post("/api/design/topopt/apply")
    def topopt_apply():
        from ..design.cad import build_robot_cad
        job = state["topo"]
        if job is None or job.result is None:
            raise HTTPException(400, "run a topology optimisation first")
        robot = state["cad"] or guard(build_robot_cad, sess.params, job.cfg.cad, xml=sess.generated_xml())
        guard(job.result.apply, robot)
        state["cad"] = robot
        with sess.lock:
            sess.cad_xml = guard(robot.mjcf, sess.generated_xml())
            guard(sess.build)
        return full_state()

    # ---- design: AI optimisation -----------------------------------------------
    @app.post("/api/design/optimize/start")
    def optimize_start(d: dict = Body(...)):
        from ..design.optimizer import DesignConfig, DesignOptimizer
        opt = state["opt"]
        if opt is not None and opt.status().get("state") == "running":
            raise HTTPException(409, "a design optimisation is already running")
        body = {k: v for k, v in d.items() if v not in (None, [], "")}
        cfg = guard(DesignConfig.from_dict, body)
        opt = guard(DesignOptimizer, cfg, sess.params, runs_dir=RUNS)
        state["opt"] = opt
        opt.start()
        return _finite(opt.status())

    @app.get("/api/design/optimize/status")
    def optimize_status():
        opt = state["opt"]
        if opt is None:
            return {"state": "idle", "history": []}
        st = opt.status()
        bp = st.get("best_params")
        if bp is not None and not isinstance(bp, dict):
            bp = bp.to_dict()
        if isinstance(bp, dict):
            genes = set(getattr(opt.cfg, "genes", []) or [])
            st["best_params"] = {k: v for k, v in bp.items() if not genes or k in genes}
        return _finite(st)

    @app.post("/api/design/optimize/stop")
    def optimize_stop():
        if state["opt"] is not None:
            state["opt"].stop()
        return optimize_status()

    @app.post("/api/design/optimize/apply")
    def optimize_apply():
        opt = state["opt"]
        bp = getattr(opt, "best_params", None) if opt is not None else None
        if bp is None:
            raise HTTPException(400, "run a design optimisation first")
        guard(sess.set_params, bp.to_dict() if hasattr(bp, "to_dict") else dict(bp))
        state["cad"] = None
        return full_state()

    # ---- live stream -----------------------------------------------------
    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        lidar_seq = cam_seq = map_seq = -1
        last_train = 0.0
        try:
            while True:
                frame = sess.frame()
                now = time.perf_counter()
                if now - last_train > 0.5:
                    last_train = now
                    st = train_status()
                    frame["train"] = {k: st.get(k) for k in ("state", "generation", "generations", "best",
                                                             "total_steps", "error", "name", "kind")}
                    if st["history"]:
                        frame["train"]["last"] = st["history"][-1]
                await sock.send_json(_finite(frame))
                if sess.lidar_seq != lidar_seq and sess.lidar_points is not None:
                    lidar_seq = sess.lidar_seq
                    await sock.send_bytes(b"\x01" + sess.lidar_points.tobytes())
                if sess.camera_seq != cam_seq and sess.camera_jpeg:
                    cam_seq = sess.camera_seq
                    await sock.send_bytes(b"\x02" + sess.camera_jpeg)
                if sess.map_seq != map_seq:
                    map_seq = sess.map_seq
                    with sess.lock:
                        if sess.nav_on_slam:
                            lo = sess.slam.resampled(sess.grid.res)  # SLAM map, shown on the same 0.1 m grid
                            img = np.zeros(lo.shape, np.uint8)
                            img[lo < -0.5] = 1
                            img[lo > 0.5] = 2
                        else:
                            img = sess.grid.lidar_image()
                        img = img + 3 * sess.grid.truth  # 0..2 map, +3 where truth occupied
                    await sock.send_bytes(b"\x03" + struct.pack("<I", sess.grid.n) + img.tobytes())
                await asyncio.sleep(1 / 30)
        except (WebSocketDisconnect, RuntimeError):
            pass

    @app.on_event("shutdown")
    def _shutdown():
        sess.pilot.stop(timeout=1.0)
        if state.get("ros") is not None:
            state["ros"].stop()
        sess.shutdown()
        for k in ("trainer", "topo", "opt"):
            if state[k] is not None:
                state[k].stop()

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
