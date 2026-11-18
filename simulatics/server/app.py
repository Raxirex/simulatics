"""HTTP + WebSocket API for the GUI (FastAPI). Serves the static front end too."""
from __future__ import annotations

import asyncio
import struct
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .. import native, __version__
from ..train.evolve import Trainer, TrainConfig
from ..train.policy import Policy
from .session import SimSession, param_spec

STATIC = Path(__file__).parent / "static"
RUNS = Path("runs")


def create_app() -> FastAPI:
    app = FastAPI(title="Simulatics")
    sess = SimSession()
    state: dict = {"trainer": None}

    def guard(fn, *a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:
            raise HTTPException(400, f"{type(e).__name__}: {e}")

    def policies() -> list[dict]:
        out = []
        for p in sorted(RUNS.glob("*/policy.npz"), key=lambda p: -p.stat().st_mtime):
            try:
                meta = Policy.load(p).meta
            except Exception:
                continue
            out.append({"name": p.parent.name, "path": str(p), "fitness": meta.get("fitness"),
                        "generation": meta.get("generation"), "algo": meta.get("algo"),
                        "custom_model": meta.get("custom_model", False)})
        return out

    def train_status() -> dict:
        tr: Trainer | None = state["trainer"]
        return tr.status() if tr else {"state": "idle", "history": [], "generation": 0, "generations": 0}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    def get_state():
        return {"version": __version__, "native": native.status(), "param_spec": param_spec(),
                "params": sess.params.to_dict(), "scene": sess.scene(), "actuators": sess.actuators(),
                "lidar": sess.lidar_cfg.to_dict(), "camera": sess.camera_cfg, "task": sess.task_cfg.to_dict(),
                "train_defaults": asdict(TrainConfig()), "train": train_status(), "policies": policies(),
                "obstacles": len(sess.obstacles)}

    @app.post("/api/params")
    def set_params(d: dict = Body(...)):
        guard(sess.set_params, d)
        return {"params": sess.params.to_dict(), "scene": sess.scene(), "actuators": sess.actuators()}

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
        else:
            guard(sess.clear_obstacles)
        return {"scene": sess.scene(), "obstacles": len(sess.obstacles)}

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
        with sess.lock:
            sess.nav.set_goal([float(d["x"]), float(d["y"])], append=bool(d.get("append")))
        return sess.nav.to_dict()

    @app.post("/api/nav")
    def nav(d: dict = Body(...)):
        with sess.lock:
            if d.get("action") == "clear":
                sess.nav.clear()
            if d.get("source") in ("truth", "lidar"):
                sess.nav.source = d["source"]
                sess.nav.path = np.zeros((0, 2))
            if d.get("action") == "reset_map":
                sess.grid.reset_lidar()
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
        tr: Trainer | None = state["trainer"]
        if tr and tr.state == "running":
            raise HTTPException(400, "training already running")
        if sess.task is None:
            raise HTTPException(400, "this model has no free-floating root or no actuators, cannot train locomotion")
        cfg = TrainConfig.from_dict(d)
        cfg.name = "".join(c for c in (cfg.name or "run") if c.isalnum() or c in "-_") or "run"
        cfg.population = int(np.clip(cfg.population, 8, 4096))
        cfg.generations = int(np.clip(cfg.generations, 1, 100000))
        if cfg.algo not in ("es", "ga"):
            raise HTTPException(400, "algo must be es or ga")
        init = sess.policy if d.get("continue_from_loaded") else None
        xml = sess.custom_path
        tr = Trainer(cfg, sess.params, sess.task_cfg, model_xml=xml, runs_dir=RUNS, init_policy=init)
        state["trainer"] = tr
        tr.start()
        return tr.status()

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
        guard(sess.load_policy, str(p))
        return {"params": sess.params.to_dict(), "scene": sess.scene(), "actuators": sess.actuators(),
                "policy": sess.policy_name}

    @app.post("/api/policy/unload")
    def policy_unload():
        sess.unload_policy()
        return {"ok": True}

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
                                                             "total_steps", "error", "name")}
                    if st["history"]:
                        frame["train"]["last"] = st["history"][-1]
                await sock.send_json(frame)
                if sess.lidar_seq != lidar_seq and sess.lidar_points is not None:
                    lidar_seq = sess.lidar_seq
                    await sock.send_bytes(b"\x01" + sess.lidar_points.tobytes())
                if sess.camera_seq != cam_seq and sess.camera_jpeg:
                    cam_seq = sess.camera_seq
                    await sock.send_bytes(b"\x02" + sess.camera_jpeg)
                if sess.map_seq != map_seq:
                    map_seq = sess.map_seq
                    with sess.lock:
                        img = sess.grid.lidar_image() + 3 * sess.grid.truth  # 0..2 lidar, +3 where truth occupied
                    await sock.send_bytes(b"\x03" + struct.pack("<I", sess.grid.n) + img.tobytes())
                await asyncio.sleep(1 / 30)
        except (WebSocketDisconnect, RuntimeError):
            pass

    @app.on_event("shutdown")
    def _shutdown():
        sess.shutdown()
        if state["trainer"]:
            state["trainer"].stop()

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
