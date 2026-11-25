"""Long-running jobs shared by every front end (web GUI, Studio, node graph): training and policy listing."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .train.evolve import Trainer, TrainConfig
from .train.policy import Policy

RUNS = Path("runs")

# GUI field defaults per training method (the Train tab swaps them in when the method changes)
TRAIN_PRESETS = {
    "ppo": {"population": 256, "generations": 1500, "lr": 0.001, "threads": 8, "episode_seconds": 20},
    "distill": {"population": 16, "generations": 14, "lr": 0.001, "threads": 5, "episode_seconds": 25},
    "imitation": {"population": 64, "generations": 1500, "lr": 0.001, "threads": 4, "episode_seconds": 10},
    "skill": {"population": 256, "generations": 120, "lr": 0.001, "threads": 0, "episode_seconds": 2},
    "es": {"population": 256, "generations": 200, "lr": 0.005, "sigma": 0.01, "threads": 0, "episode_seconds": 8},
    "ga": {"population": 256, "generations": 200, "lr": 0.005, "sigma": 0.01, "threads": 0, "episode_seconds": 8},
}


class StopDistill(Exception):
    pass


class DistillJob:
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
            from .train.ppo import DistillConfig, distill, walker_teacher_factory
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
                    raise StopDistill()

            distill(model, factory, cfg=cfg, run_dir=RUNS / self.name,
                    model_meta={"humanoid": self.params.to_dict(), "custom_model": False}, callback=cb)
            self.state = "finished"
        except StopDistill:
            self.state = "stopped"
        except Exception as e:  # noqa: BLE001
            self.state, self.error = "error", f"{type(e).__name__}: {e}"

    def status(self):
        state = "stopping after this round" if self._stop.is_set() and self.state == "running" else self.state
        return {"state": state, "error": self.error, "name": self.name, "algo": "distill", "kind": "distill",
                "generation": len(self.history), "generations": self.rounds, "best": None, "total_steps": 0,
                "history": self.history, "policy": str(self.policy_path) if self.policy_path.exists() else None}


def list_policies() -> list[dict]:
    """Trained policies found under runs/, newest first."""
    from .train.ppo import RLPolicy, is_rl_policy
    out = []
    for p in sorted(list(RUNS.glob("*/policy.npz")) + list(RUNS.glob("*/rl_policy.npz"))
                    + list(RUNS.glob("*/imitation_policy.npz")) + list(RUNS.glob("*/skill.tape.npz")),
                    key=lambda p: -p.stat().st_mtime):
        try:
            if p.name == "skill.tape.npz":                      # a trained skill (skills/): a motor-command recording
                import json
                info = p.parent / "skill.json"
                meta = json.loads(info.read_text(encoding="utf-8")) if info.exists() else {}
                out.append({"name": p.parent.name, "path": str(p), "fitness": meta.get("best"),
                            "generation": len(meta.get("history", [])), "algo": "skill", "kind": "skill",
                            "custom_model": False})
                continue
            if p.name == "imitation_policy.npz":
                from .imitate import is_imitation_policy
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
    return out


def make_trainer(d: dict, params, task_cfg, custom_path: str | None = None, export_xml: str | None = None,
                 init_policy=None, continue_run: str | None = None):
    """Build (not start) the trainer for d["algo"]: ppo, distill, imitation, es or ga.

    Every trainer has start(), stop() and status() -> {state, generation, generations, history, ...}.
    `init_policy` warm-starts es/ga; `continue_run` names a run whose PPO checkpoint to continue.
    """
    algo = d.get("algo", "es")
    name = "".join(c for c in str(d.get("name") or "run") if c.isalnum() or c in "-_") or "run"
    if algo == "ppo":
        from .train.ppo import PPOConfig, PPOTrainer
        from .sim.vel_task import VelTaskConfig
        cfg = PPOConfig(num_envs=int(np.clip(d.get("population", 256), 4, 4096)),
                        iterations=int(np.clip(d.get("generations", 1500), 1, 1000000)),
                        learning_rate=float(np.clip(d.get("lr", 1e-3), 1e-6, 0.1)),
                        threads=int(d.get("threads") or 8), name=name)
        init = None
        if continue_run:
            ck = RUNS / continue_run / "rl_checkpoint.npz"
            init = ck if ck.exists() else None
        tr = PPOTrainer(cfg, params, VelTaskConfig(), model_xml=custom_path, runs_dir=RUNS,
                   init_checkpoint=init)
    elif algo == "imitation":
        from .imitate import ImitationTrainer, imitation_ppo_config
        cfg = imitation_ppo_config(name=name, num_envs=int(np.clip(d.get("population", 64), 4, 1024)),
                                   iterations=int(np.clip(d.get("generations", 1500), 1, 1000000)),
                                   threads=int(d.get("threads") or 4))
        motion = str(d.get("motion") or "walk")
        mpath = RUNS / "motions" / f"{motion}.npz"
        tr = ImitationTrainer(cfg, str(mpath) if mpath.exists() else motion, base=params,
                   runs_dir=RUNS)
    elif algo == "distill":
        tr = DistillJob(export_xml, params, name, int(np.clip(d.get("generations", 14), 1, 50)),
                         int(np.clip(d.get("population", 16), 1, 64)), int(np.clip(d.get("threads") or 5, 1, 16)),
                         float(np.clip(d.get("episode_seconds", 25), 2, 120)))
    elif algo == "skill":
        from .skills import SKILLS, SkillTrainer
        from .skills import TrainConfig as SkillConfig
        sid = str(d.get("skill") or d.get("motion") or "uppercut")
        if sid not in SKILLS:
            raise ValueError(f"unknown skill {sid!r} (have: {', '.join(SKILLS)})")
        tr = SkillTrainer(SKILLS[sid], SkillConfig(population=int(np.clip(d.get("population", 256), 16, 4096)),
                                                   generations=int(np.clip(d.get("generations", 120), 1, 100000)),
                                                   threads=int(d.get("threads") or 0), name=name), runs_dir=RUNS)
    elif algo in ("es", "ga"):
        cfg = TrainConfig.from_dict(d)
        cfg.name = name
        cfg.population = int(np.clip(cfg.population, 8, 4096))
        cfg.generations = int(np.clip(cfg.generations, 1, 100000))
        init = init_policy
        tr = Trainer(cfg, params, task_cfg, model_xml=custom_path, runs_dir=RUNS, init_policy=init)
    else:
        raise ValueError("algo must be ppo, distill, imitation, skill, es or ga")
    return tr


def make_trainer_for_session(sess, d: dict):
    """make_trainer() for the robot loaded in a SimSession."""
    if sess.task is None:
        raise ValueError("this model has no free-floating root or no actuators, cannot train locomotion")
    cont = bool(d.get("continue_from_loaded"))
    return make_trainer(d, sess.params, sess.task_cfg, custom_path=sess.custom_path,
                        export_xml=sess.export_xml() if d.get("algo") == "distill" else None,
                        init_policy=sess.policy if cont else None,
                        continue_run=sess.policy_name if cont and sess.rl_policy is not None and sess.policy_name else None)
