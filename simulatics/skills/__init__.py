"""Skill training: teach the robot a new move by trying thousands of variations in parallel.

A skill is described by a `SkillSpec`: the scene (objects), which joints may move, how long it lasts, and a
reward computed from what happened in the simulation. Training searches for a *motor program*: a smooth
time course of servo-target offsets for the chosen joints (a few knots, interpolated), added to the robot's
standing pose. Every generation, a population of candidate programs is simulated in parallel on all CPU cores
(MuJoCo's C rollout, each candidate on several slightly different starts so the result does not hinge on one
exact situation), scored by the reward, and the search distribution moves toward the best ones (the
cross-entropy method). The result is a `Tape` of motor commands, played in the live session by the same
machinery as the recorded jumps: only motor commands are written, and the balance controller takes over when
the move ends.

    from simulatics import skills
    trainer = skills.SkillTrainer(skills.SKILLS["uppercut"], skills.TrainConfig(generations=120))
    trainer.start(); ...; trainer.status()          # same shape as the other trainers (history of best / mean)
    python -m simulatics.skills uppercut            # command line

What this is and is not:
- It is real training in physics, in parallel, and it is how the shipped uppercut was made.
- It produces an open-loop program for one situation (with a little variation), not a policy that reacts to
  what it sees. Closed-loop policies are what train/ppo.py and imitate/train.py are for.
- The backend is CPU threads (`CpuBackend`). `Backend` is the seam where a GPU simulator (MuJoCo MJX or
  MuJoCo Warp) would plug in; `GpuBackend` is a stub that says what to install, because neither is here.

Write your own skill: a `SkillSpec` with a `reward(r: Rollout) -> (n,) array` and add it to `SKILLS`.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np
from mujoco import rollout as mj_rollout

from ..imitate.physical import STATE, Tape, fingerprint, prepare_model
from ..jobs import RUNS

HAND_BODIES = ("left_hand", "right_hand")
FOOT_BODIES = ("left_foot", "right_foot")


# =========================================================================================================
# What a reward function sees
# =========================================================================================================
class Rollout:
    """A batch of simulated attempts. Arrays are (n, T, ...) over control steps; n = candidates x variations."""

    def __init__(self, model: mujoco.MjModel, state: np.ndarray, sens: np.ndarray, layout: dict, dt: float,
                 program: np.ndarray):
        self.model, self.dt, self.program = model, dt, program
        self.n, self.T = state.shape[:2]
        nq = model.nq
        self.time = state[:, :, 0]
        self.qpos = state[:, :, 1:1 + nq]
        self.qvel = state[:, :, 1 + nq:1 + nq + model.nv]
        self._sens, self._layout = sens, layout

    def _s(self, key: str) -> np.ndarray:
        a, n = self._layout[key]
        return self._sens[:, :, a:a + n]

    @property
    def com(self) -> np.ndarray:
        return self._s("com")

    @property
    def com_vel(self) -> np.ndarray:
        return self._s("comvel")

    @property
    def up(self) -> np.ndarray:
        """(n, T) how upright the torso is: 1 = vertical, 0 = horizontal, -1 = upside down."""
        return self._s("up")[:, :, 2]

    def body_pos(self, name: str) -> np.ndarray:
        return self._s(f"pos:{name}")

    def body_vel(self, name: str) -> np.ndarray:
        return self._s(f"vel:{name}")

    def prop_pos(self, i: int = 0) -> np.ndarray:
        return self._s(f"pos:prop_{i}")

    def prop_vel(self, i: int = 0) -> np.ndarray:
        return self._s(f"vel:prop_{i}")

    def joint(self, name: str) -> np.ndarray:
        return self.qpos[:, :, self.model.jnt_qposadr[self.model.joint(name).id]]

    # ---- pieces most rewards need ---------------------------------------------------------------------
    def stays_standing(self, min_up: float = 0.85, max_drop: float = 0.15) -> np.ndarray:
        """(n,) 1 where the robot never tipped or sank, else 0."""
        ok = (self.up.min(axis=1) > min_up) & ((self.com[:, 0, 2] - self.com[:, :, 2].min(axis=1)) < max_drop)
        return ok.astype(float)

    def feet_moved(self) -> np.ndarray:
        """(n,) how far the feet ended from where they started (m, summed over both)."""
        return sum(np.linalg.norm(self.body_pos(f)[:, -1] - self.body_pos(f)[:, 0], axis=1) for f in FOOT_BODIES)

    def ends_at_rest(self) -> np.ndarray:
        """(n,) speed of the centre of mass over the last 0.2 s (m/s): small = it came back to standing."""
        k = max(1, int(round(0.2 / self.dt)))
        return np.linalg.norm(self.com_vel[:, -k:], axis=2).mean(axis=1)

    def returns_to_start(self) -> tuple[np.ndarray, np.ndarray]:
        """(n,), (n,): how far the centre of mass (m) and the joints (rad, mean) end from where they started.
        Small values mean the move ends in the standing pose it began in, ready for the balance controller."""
        com = np.linalg.norm(self.com[:, -1] - self.com[:, 0], axis=1)
        hinge = [self.model.jnt_qposadr[j] for j in range(self.model.njnt)
                 if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and "gripper" not in self.model.joint(j).name]
        pose = np.abs(self.qpos[:, -1][:, hinge] - self.qpos[:, 0][:, hinge]).mean(axis=1)
        return com, pose

    def effort(self) -> np.ndarray:
        """(n,) mean squared program offset (rad^2): how much the motors were asked to deviate."""
        return (self.program ** 2).mean(axis=(1, 2))


# =========================================================================================================
# Skill definition
# =========================================================================================================
@dataclass
class SkillSpec:
    id: str
    title: str
    description: str
    joints: list[str]                      # joints the program may move; all others hold the standing pose
    reward: Callable[[Rollout], np.ndarray]
    duration: float = 1.2                  # s of free movement
    settle: float = 0.8                    # s at the end in which the program eases back to the standing pose
    knots: int = 8                         # program resolution over `duration`
    sigma: float = 0.35                    # rad, initial spread of the search
    scene: Callable[[], tuple[list[dict], list[dict]]] | None = None     # -> (obstacles, props)
    params: dict | None = None             # robot parameter overrides
    jitter_pose: float = 0.01              # rad, start-pose variation between the attempts of one candidate
    jitter_props: float = 0.01             # m, object position variation
    report: Callable[[Rollout], dict] | None = None     # numbers worth showing about the best attempt
    trained: bool = True                   # False: defined, but no shipped result (experimental)


@dataclass
class TrainConfig:
    population: int = 256                  # candidate programs per generation
    variations: int = 3                    # differently perturbed starts per candidate
    generations: int = 120
    elite: float = 0.12                    # fraction kept to move the search
    smoothing: float = 0.3                 # how much of the old mean / spread is kept each generation
    min_sigma: float = 0.02
    threads: int = 0                       # 0 = all cores
    control_steps: int = 4                 # physics steps per command
    seed: int = 0
    name: str = "skill"


# =========================================================================================================
# Simulation backends
# =========================================================================================================
class Backend:
    """Simulates many attempts at once. One method: states and sensor readings for a batch of command sequences."""
    name = "abstract"

    def rollout(self, start: np.ndarray, ctrl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """start (n, nstate), ctrl (n, steps, nu) -> state (n, steps, nstate), sensors (n, steps, nsensordata)."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class CpuBackend(Backend):
    """MuJoCo's multi-threaded C rollout: one physics instance per CPU thread."""
    name = "cpu"

    def __init__(self, model: mujoco.MjModel, threads: int = 0):
        self.model = model
        self.threads = threads or max(1, os.cpu_count() or 4)
        self.datas = [mujoco.MjData(model) for _ in range(self.threads)]
        self.pool = mj_rollout.Rollout(nthread=self.threads)
        self.steps_done = 0

    def rollout(self, start, ctrl):
        self.steps_done += ctrl.shape[0] * ctrl.shape[1]
        return self.pool.rollout(self.model, self.datas, start, ctrl)

    def close(self) -> None:
        self.pool.close()


class GpuBackend(Backend):
    """Where a GPU simulator plugs in (thousands of robots per step). Not available in this installation."""
    name = "gpu"

    def __init__(self, *_a, **_k):
        raise RuntimeError("no GPU simulator is installed. The Backend interface is ready for one: MuJoCo MJX "
                           "(pip install mujoco-mjx jax[cuda]) or MuJoCo Warp (pip install mujoco-warp), with a "
                           "supported GPU. Until then training runs on CPU threads (CpuBackend).")


# =========================================================================================================
# Environment: the session's own model for this skill, plus the sensors rewards read
# =========================================================================================================
def scene_for(spec: SkillSpec) -> tuple[str, np.ndarray, np.ndarray, list[dict], list[dict]]:
    """(xml, start qpos, start ctrl, obstacles, props): exactly what the live session builds for the skill."""
    from .. import samples
    from ..server.session import SimSession
    obstacles, props = spec.scene() if spec.scene else ([], [])
    s = SimSession()
    try:
        s.lidar_cfg.enabled = False
        samples._builtin(s, obstacles, spec.params, props)
        with s.lock:
            return s.export_xml(), s.data.qpos.copy(), s.data.ctrl.copy(), obstacles, props
    finally:
        s.shutdown()


class SkillEnv:
    def __init__(self, spec: SkillSpec, cfg: TrainConfig, xml: str, qpos0: np.ndarray, ctrl0: np.ndarray,
                 backend: str = "cpu"):
        self.spec, self.cfg = spec, cfg
        self.xml = xml
        self.plain = mujoco.MjModel.from_xml_string(xml)
        sp = mujoco.MjSpec.from_string(xml)
        S, O = mujoco.mjtSensor, mujoco.mjtObj
        root = self.plain.body(1).name
        layout_names = [("com", S.mjSENS_SUBTREECOM, root, 3), ("comvel", S.mjSENS_SUBTREELINVEL, root, 3),
                        ("up", S.mjSENS_FRAMEZAXIS, root, 3)]
        bodies = [b for b in HAND_BODIES + FOOT_BODIES if mujoco.mj_name2id(self.plain, O.mjOBJ_BODY, b) >= 0]
        bodies += [self.plain.body(b).name for b in range(self.plain.nbody) if self.plain.body(b).name.startswith("prop_")]
        for b in bodies:
            layout_names += [(f"pos:{b}", S.mjSENS_FRAMEPOS, b, 3), (f"vel:{b}", S.mjSENS_FRAMELINVEL, b, 3)]
        for key, typ, obj, _n in layout_names:
            sp.add_sensor(name=f"skill_{key.replace(':', '_')}", type=typ, objtype=O.mjOBJ_BODY, objname=obj)
        self.model = sp.compile()
        for m in (self.model, self.plain):
            prepare_model(m)
        adr = self.model.nsensordata - sum(n for *_x, n in layout_names)
        self.layout = {}
        for key, _t, _o, n in layout_names:
            self.layout[key] = (adr, n)
            adr += n
        m = self.model
        self.dt = cfg.control_steps * float(m.opt.timestep)
        self.n_move = int(round(spec.duration / self.dt))
        self.n_settle = int(round(spec.settle / self.dt))
        self.T = self.n_move + self.n_settle
        act_of = {m.joint(int(m.actuator_trnid[a, 0])).name: a for a in range(m.nu)}
        missing = [j for j in spec.joints if j not in act_of]
        if missing:
            raise ValueError(f"skill '{spec.id}': no motor on joints {missing}")
        self.act = np.array([act_of[j] for j in spec.joints])
        self.qpos0, self.ctrl0 = np.array(qpos0, float), np.array(ctrl0, float)
        kp = m.actuator_gainprm[:, 0]
        reach = np.where(kp > 0, m.actuator_forcerange[:, 1] / np.maximum(kp, 1e-9), 0.0) * 1.5
        self.lo, self.hi = m.actuator_ctrlrange[:, 0] - reach, m.actuator_ctrlrange[:, 1] + reach
        self.nstate = mujoco.mj_stateSize(m, STATE)
        # interpolation from knots to control steps: piecewise-linear hat functions over the movement, zero at
        # both ends (the move starts from and returns to standing), then an eased return during the settle
        K = spec.knots
        t = np.arange(self.n_move) / max(self.n_move - 1, 1) * (K + 1)
        self.basis = np.maximum(0.0, 1.0 - np.abs(t[:, None] - np.arange(1, K + 1)[None, :]))     # (n_move, K)
        self.backend: Backend = CpuBackend(m, cfg.threads) if backend == "cpu" else GpuBackend()
        self.prop_q = [(int(m.jnt_qposadr[m.body_jntadr[b]])) for b in range(m.nbody) if m.body(b).name.startswith("prop_")]
        hinge = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        self.hinge_q = np.array([m.jnt_qposadr[j] for j in hinge if "gripper" not in m.joint(j).name])

    def close(self) -> None:
        self.backend.close()

    def offsets(self, theta: np.ndarray) -> np.ndarray:
        """(n, K, J) knot values -> (n, T, J) servo-target offsets per control step."""
        move = np.einsum("tk,nkj->ntj", self.basis, theta)
        return np.concatenate([move, np.zeros((len(theta), self.n_settle, theta.shape[2]))], axis=1)

    def commands(self, theta: np.ndarray) -> np.ndarray:
        """(n, T, nu) servo targets."""
        u = np.tile(self.ctrl0, (len(theta), self.T, 1))
        u[:, :, self.act] += self.offsets(theta)
        return np.clip(u, self.lo, self.hi)

    def starts(self, n: int, rng: np.random.Generator | None) -> np.ndarray:
        """(n, nstate): the standing start, each with a small random change (none when rng is None)."""
        m, d = self.model, mujoco.MjData(self.model)
        out = np.zeros((n, self.nstate))
        for i in range(n):
            q = self.qpos0.copy()
            if rng is not None and i > 0:
                q[self.hinge_q] += self.spec.jitter_pose * rng.standard_normal(len(self.hinge_q))
                for a in self.prop_q:
                    q[a:a + 2] += self.spec.jitter_props * rng.standard_normal(2)
            mujoco.mj_resetData(m, d)
            d.qpos[:] = q
            mujoco.mj_getState(m, d, out[i], STATE)
        return out

    def run(self, theta: np.ndarray, starts: np.ndarray) -> Rollout:
        """Simulate every candidate from every start: (len(theta) * len(starts)) attempts, candidate-major."""
        n, v = len(theta), len(starts)
        u = np.repeat(self.commands(theta), v, axis=0)
        ctrl = np.repeat(u, self.cfg.control_steps, axis=1)
        st, sens = self.backend.rollout(np.tile(starts, (n, 1)), ctrl)
        k = self.cfg.control_steps
        return Rollout(self.model, st[:, k - 1::k], sens[:, k - 1::k], self.layout, self.dt,
                       np.repeat(self.offsets(theta), v, axis=0))


# =========================================================================================================
# Trainer
# =========================================================================================================
class SkillTrainer:
    """Cross-entropy search over motor programs, simulated in parallel. start() / stop() / status()."""

    def __init__(self, spec: SkillSpec, cfg: TrainConfig | None = None, runs_dir: Path | str = RUNS, backend: str = "cpu"):
        self.spec, self.cfg = spec, cfg or TrainConfig()
        self.run_dir = Path(runs_dir) / self.cfg.name
        self.tape_path = self.run_dir / "skill.tape.npz"
        self.backend = backend
        self.state, self.error = "idle", ""
        self.history: list[dict] = []
        self.best = -np.inf
        self.best_theta: np.ndarray | None = None
        self.tape: Tape | None = None
        self.report: dict = {}
        self.total_steps = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ---- life cycle (same shape as train/evolve.py Trainer) --------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.state = "running"
        self._stop.clear()
        self._thread = threading.Thread(target=self._safe, name="skill-trainer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def status(self) -> dict:
        with self._lock:
            return {"state": self.state, "error": self.error, "name": self.cfg.name, "algo": "skill", "kind": "skill",
                    "skill": self.spec.id, "generation": len(self.history), "generations": self.cfg.generations,
                    "best": None if not np.isfinite(self.best) else float(self.best), "total_steps": int(self.total_steps),
                    "history": self.history[-400:], "report": dict(self.report),
                    "policy": str(self.tape_path) if self.tape_path.exists() else None}

    def _safe(self) -> None:
        try:
            self.run()
            self.state = "stopped" if self._stop.is_set() else "finished"
        except Exception as e:  # noqa: BLE001 - shown in the interface
            self.state, self.error = "error", f"{type(e).__name__}: {e}"

    # ---- the search -----------------------------------------------------------------------------------
    def run(self, log: Callable[[str], None] | None = None) -> Tape:
        spec, cfg = self.spec, self.cfg
        xml, q0, c0, obstacles, props = scene_for(spec)
        env = SkillEnv(spec, cfg, xml, q0, c0, self.backend)
        try:
            rng = np.random.default_rng(cfg.seed)
            K, J = spec.knots, len(spec.joints)
            mean, sigma = np.zeros((K, J)), np.full((K, J), spec.sigma)
            n_elite = max(4, int(round(cfg.elite * cfg.population)))
            t0 = time.perf_counter()
            for g in range(cfg.generations):
                if self._stop.is_set():
                    break
                theta = mean + sigma * rng.standard_normal((cfg.population, K, J))
                theta[0] = mean                                           # the current mean is always tried
                if self.best_theta is not None:
                    theta[1] = self.best_theta                            # and so is the best so far
                r = env.run(theta, env.starts(cfg.variations, rng))
                # a candidate is as good as its worst attempt: that is what makes the result dependable
                score = np.asarray(spec.reward(r), float).reshape(cfg.population, cfg.variations).min(axis=1)
                score = np.where(np.isfinite(score), score, -1e9)
                order = np.argsort(-score)
                elite = theta[order[:n_elite]]
                mean = cfg.smoothing * mean + (1 - cfg.smoothing) * elite.mean(axis=0)
                sigma = np.maximum(cfg.smoothing * sigma + (1 - cfg.smoothing) * elite.std(axis=0), cfg.min_sigma)
                with self._lock:
                    if score[order[0]] > self.best:
                        self.best, self.best_theta = float(score[order[0]]), theta[order[0]].copy()
                    self.total_steps = getattr(env.backend, "steps_done", 0)
                    sps = self.total_steps / max(time.perf_counter() - t0, 1e-9)
                    self.history.append({"gen": g + 1, "best": float(self.best), "mean": float(np.mean(score[order[:n_elite]])),
                                         "center": float(score[0]), "sps": float(sps)})
                if log and (g % 10 == 0 or g == cfg.generations - 1):
                    log(f"gen {g + 1:4d}/{cfg.generations}  best {self.best:9.4f}  elite mean {np.mean(score[order[:n_elite]]):9.4f}  "
                        f"spread {sigma.mean():.3f} rad  {sps / 1000:5.0f}k steps/s")
            if self.best_theta is None:
                raise RuntimeError("stopped before the first generation finished")
            self.tape = self._tape(env, obstacles, props)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.tape.save(self.tape_path)
            (self.run_dir / "skill.json").write_text(json.dumps(
                {"skill": spec.id, "title": spec.title, "config": asdict(cfg), "best": self.best, "report": self.report,
                 "history": self.history, "obstacles": obstacles, "props": props, "params": spec.params or {}},
                indent=1), encoding="utf-8")
            return self.tape
        finally:
            env.close()

    def _tape(self, env: SkillEnv, obstacles: list[dict], props: list[dict]) -> Tape:
        """The best program as a recording the live session can play, with what it achieved measured."""
        theta = self.best_theta[None]
        r = env.run(theta, env.starts(1, None))                           # the unperturbed start
        many = env.run(theta, env.starts(24, np.random.default_rng(12345)))
        m = env.plain
        d = mujoco.MjData(m)
        d.qpos[:] = env.qpos0
        mujoco.mj_forward(m, d)
        ctrl = env.commands(theta)[0]
        states = [d.qpos.copy()]
        for u in ctrl:
            d.ctrl[:] = u
            mujoco.mj_step(m, d, self.cfg.control_steps)
            states.append(d.qpos.copy())
        # End in the stance the balance controller takes over from: a balanced return to the starting pose,
        # found by the motion tracker (the program's own "back to zero" leaves the robot sagging on its servos).
        from ..imitate.physical import TrackConfig, Tracker
        tr = Tracker(env.xml, TrackConfig(control_steps=self.cfg.control_steps, passes=3, threads=self.cfg.threads))
        try:
            back, back_states, back_cost = tr.settle_to(d.qpos.copy(), d.qvel.copy(), env.qpos0)
        finally:
            tr.close()
        ctrl = np.concatenate([ctrl, back])
        states = np.concatenate([np.array(states), back_states[1:]])
        rep = {"reward": float(np.asarray(self.spec.reward(r))[0]),
               "reward_over_24_varied_starts": {"min": float(np.min(self.spec.reward(many))), "mean": float(np.mean(self.spec.reward(many)))},
               "stayed_standing_of_24": int(many.stays_standing().sum()), "final_up": float(r.up[0, -1]),
               "feet_moved_m": float(r.feet_moved()[0]), "end_speed": float(r.ends_at_rest()[0]),
               "return_to_stance_cost": float(back_cost),
               "final_height_error_m": float(abs(states[-1][2] - env.qpos0[2]))}
        if self.spec.report:
            rep.update({k: (float(v) if np.isscalar(v) or getattr(v, "ndim", 1) == 0 else v) for k, v in self.spec.report(r).items()})
        self.report = rep
        meta = {"skill": self.spec.id, "title": self.spec.title, "joints": list(self.spec.joints), "report": rep,
                "generations": len(self.history), "population": self.cfg.population, "variations": self.cfg.variations,
                "physics_steps": int(getattr(env.backend, "steps_done", 0)), "obstacles": obstacles, "props": props,
                "params": self.spec.params or {}}
        return Tape(ctrl, self.cfg.control_steps, float(m.opt.timestep), env.qpos0.copy(), np.zeros(m.nv), states, states.copy(),
                    fingerprint(m), self.spec.id, meta)


def play(session, tape: Tape | str | Path) -> str:
    """Set the live session up for a trained skill (its robot, scene and objects) and play it."""
    from .. import samples
    if not hasattr(tape, "ctrl"):
        tape = Tape.load(tape)
    meta = tape.meta
    samples._builtin(session, meta.get("obstacles") or [], meta.get("params") or None, meta.get("props") or [])
    return session.play_tape(tape)


from .tasks import SKILLS  # noqa: E402,F401
