"""Tests for the CPU learning stack: numpy networks, PPO, the velocity task, vectorised envs, distillation
and the RLController. Every test uses at most 4 threads and runs in seconds, except the humanoid smoke
run, which is opt-in:  set SIMULATICS_SLOW=1  (it trains for ~90 s on 8 threads and prints the curve).
"""
import json
import os
import time

import mujoco
import numpy as np
import pytest

from simulatics.control.base import Controller
from simulatics.models import HumanoidParams, build_mjcf
from simulatics.sim.vec_env import VecEnv
from simulatics.sim.vel_task import VelocityTask, VelTaskConfig, make_variants
from simulatics.train.nn import MLP, Adam, GaussianPolicy, RunningMeanStd, ValueNet, clip_grad_norm
from simulatics.train.ppo import (DistillConfig, PPOConfig, PPOTrainer, RLController, RLPolicy, StandingTeacher,
                                  distill, evaluate_policy, is_rl_policy, run_ppo)


@pytest.fixture(scope="module")
def humanoid():
    return mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams()))


# ---- networks -------------------------------------------------------------------------------------------
def _fd_check(params, loss_fn, grads, rng, n=25, eps=1e-6):
    """Worst relative error between analytic and central finite-difference gradients on random entries."""
    worst = 0.0
    for p, g in zip(params, grads):
        for _ in range(min(n, p.size)):
            i = tuple(int(rng.integers(0, s)) for s in p.shape)
            old = p[i]
            p[i] = old + eps
            fp = loss_fn()
            p[i] = old - eps
            fm = loss_fn()
            p[i] = old
            num = (fp - fm) / (2 * eps)
            worst = max(worst, abs(num - g[i]) / max(1e-7, abs(num) + abs(g[i])))
    return worst


@pytest.mark.parametrize("act", ["elu", "tanh", "relu"])
def test_gradients_match_finite_differences(act):
    rng = np.random.default_rng(0)
    pol = GaussianPolicy(7, 3, [16, 8], act, 0.7, rng, dtype=np.float64)
    for p in pol.mu.params[-2:]:
        p += rng.normal(0, 0.3, p.shape)  # move the output layer off its tiny init so ratios vary
    obs, a, adv = rng.normal(size=(24, 7)), rng.normal(size=(24, 3)), rng.normal(size=24)
    old_logp = GaussianPolicy.log_prob(pol.mean(obs), pol.log_std, a) + rng.normal(0, 0.15, 24)
    _, g, info = pol.ppo_loss(obs, a, old_logp, adv, 0.2, 0.01)
    assert 0 < info["clip_frac"] < 1  # both the clipped and the unclipped branch are exercised
    assert _fd_check(pol.params, lambda: pol.ppo_loss(obs, a, old_logp, adv, 0.2, 0.01)[0], g, rng) < 1e-5
    assert _fd_check(pol.params, lambda: pol.bc_loss(obs, a)[0], pol.bc_loss(obs, a)[1], rng) < 1e-5

    v = ValueNet(7, [16, 8], act, rng, dtype=np.float64)
    ret, v_old = rng.normal(size=24), v(obs) + rng.normal(0, 0.3, 24)
    assert _fd_check(v.params, lambda: v.loss(obs, ret, v_old, 0.2)[0], v.loss(obs, ret, v_old, 0.2)[1], rng) < 1e-5

    net = MLP([5, 6, 2], act, rng=rng, dtype=np.float64)
    x = rng.normal(size=(4, 5))
    y, cache = net.forward(x, keep=True)
    _, gx = net.backward(cache, np.ones_like(y), need_input_grad=True)
    num = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        xp, xm = x.copy(), x.copy()
        xp[i] += 1e-6
        xm[i] -= 1e-6
        num[i] = (net.forward(xp).sum() - net.forward(xm).sum()) / 2e-6
    assert np.allclose(gx, num, atol=1e-6)


def test_float32_network_matches_float64():
    rng = np.random.default_rng(1)
    a = MLP([10, 32, 32, 4], "elu", rng=np.random.default_rng(2), dtype=np.float64)
    b = MLP([10, 32, 32, 4], "elu", rng=np.random.default_rng(2), dtype=np.float32)
    x = rng.normal(size=(64, 10))
    assert np.allclose(a.forward(x), b.forward(x), atol=1e-4)


def test_adam_clip_and_normaliser():
    rng = np.random.default_rng(0)
    w = rng.normal(size=5)
    opt = Adam([w], lr=0.05)
    for _ in range(400):
        opt.step([2 * (w - 3.0)])
    assert np.allclose(w, 3.0, atol=1e-2)
    g, norm = clip_grad_norm([np.full(4, 3.0), np.full(3, 4.0)], 1.0)
    assert norm == pytest.approx(np.sqrt(36 + 48))
    assert np.sqrt(sum((x * x).sum() for x in g)) == pytest.approx(1.0, rel=1e-5)
    data = rng.normal(2.0, 3.0, size=(1000, 3))
    rms = RunningMeanStd(3)
    for chunk in np.array_split(data, 7):
        rms.update(chunk)
    assert np.allclose(rms.mean, data.mean(0)) and np.allclose(rms.var, data.var(0))


# ---- PPO on a toy problem ---------------------------------------------------------------------------------
class PointMass:
    """2-D point mass driven towards a random target; reward -distance per step; 40-step episodes."""

    def __init__(self, n=64, horizon=40, seed=0):
        self.num_envs, self.obs_dim, self.critic_obs_dim, self.act_dim = n, 4, 4, 2
        self.h, self.rng = horizon, np.random.default_rng(seed)
        self.pos, self.vel, self.tgt = np.zeros((n, 2)), np.zeros((n, 2)), np.zeros((n, 2))
        self.t, self.ret = np.zeros(n, int), np.zeros(n)

    def _reset(self, idx):
        self.pos[idx] = self.rng.uniform(-1, 1, (len(idx), 2))
        self.tgt[idx] = self.rng.uniform(-1, 1, (len(idx), 2))
        self.vel[idx], self.t[idx], self.ret[idx] = 0, 0, 0

    def _obs(self):
        o = np.concatenate([self.tgt - self.pos, self.vel], 1)
        return o, o

    def reset(self):
        self._reset(np.arange(self.num_envs))
        return self._obs()

    def step(self, a):
        self.vel = 0.8 * self.vel + 0.5 * np.clip(a, -1, 1)
        self.pos += 0.1 * self.vel
        self.t += 1
        r = -0.1 * np.linalg.norm(self.pos - self.tgt, axis=1)
        self.ret += r
        done = self.t >= self.h
        info = {"time_outs": done.astype(float)}
        if done.any():
            idx = np.flatnonzero(done)
            info["episode"] = {"return": float(self.ret[idx].mean()), "count": len(idx)}
            self._reset(idx)
        return (*self._obs(), r, done, info)


def test_ppo_improves_on_toy_problem():
    cfg = PPOConfig(num_envs=64, steps_per_env=40, actor_hidden=[32, 32], critic_hidden=[32, 32],
                    init_noise_std=1.0, entropy_coef=0.0, seed=0, blas_threads=2)
    t0 = time.perf_counter()
    _, log = run_ppo(PointMass(64), cfg, iterations=30)
    first, last = log[0]["mean_return"], np.mean([e["mean_return"] for e in log[-5:]])
    print(f"toy PPO: return {first:.2f} -> {last:.2f} in {time.perf_counter() - t0:.1f} s")
    assert last > first + 2.0 and last > -1.5  # random policy ~ -5, a good controller ~ -0.5


# ---- velocity task + vectorised env -----------------------------------------------------------------------
def test_vel_task_layout_actions_and_rewards(humanoid):
    task = VelocityTask(humanoid, VelTaskConfig(obs_noise=0.0))
    assert task.act_dim == 14 and all(not any(k in n for k in ("shoulder", "elbow", "wrist", "gripper"))
                                      for n in task.joint_names)
    n = 6
    states = np.zeros((n, task.nstate))
    ts = task.make_state(n)
    task.reset_envs(states, ts, np.arange(n), np.random.default_rng(0))
    qpos, qvel = task.split(states)
    obs, cobs = task.observe(qpos, qvel, ts)
    assert obs.shape == (n, task.obs_dim) and cobs.shape == (n, task.critic_obs_dim)
    assert np.isfinite(obs).all() and np.isfinite(cobs).all()
    a = np.random.default_rng(1).normal(size=(n, task.act_dim))
    ctrl = task.action_to_ctrl(a)
    assert np.allclose(ctrl[:, task.arm_idx], task.ctrl_default[task.arm_idx])  # arms held
    assert (ctrl >= task.ctrl_lo - 1e-12).all() and (ctrl <= task.ctrl_hi + 1e-12).all()
    inside = (ctrl[:, task.ctrl_idx] > task.q_lo) & (ctrl[:, task.ctrl_idx] < task.q_hi)
    assert np.allclose(task.ctrl_to_action(ctrl)[inside], a[inside])
    # tracking reward is higher when the base moves at the commanded velocity
    ts.commands[:] = [0.5, 0.0, 0.0]
    still = qvel.copy()
    moving = qvel.copy()
    w, x, y, z = qpos[:, 3:7].T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    moving[:, 0], moving[:, 1] = 0.5 * np.cos(yaw), 0.5 * np.sin(yaw)
    zero = np.zeros((n, task.act_dim))
    contact = np.ones((n, 2), bool)
    _, t_still, done = task.reward(qpos, still, zero, task.action_to_ctrl(zero), contact, task.make_state(n))
    ts2 = task.make_state(n)
    ts2.commands[:] = [0.5, 0.0, 0.0]
    _, t_move, _ = task.reward(qpos, moving, zero, task.action_to_ctrl(zero), contact, ts2)
    assert not done.any()
    assert (t_move["tracking_lin_vel"] > 0.95 * task.weights["tracking_lin_vel"] * task.dt).all()
    fallen = qpos.copy()
    fallen[:, 2] = 0.3
    assert task.reward(fallen, still, zero, task.action_to_ctrl(zero), contact, task.make_state(n))[2].all()


def test_vec_env_steps_resets_and_is_thread_count_independent(humanoid):
    cfg = VelTaskConfig(n_model_variants=3)
    finals = []
    for nthread in (1, 4):
        with VecEnv(humanoid, cfg, num_envs=8, nthread=nthread, seed=7) as env:
            obs, cobs = env.reset()
            assert len(env.variants) == 3
            rng = np.random.default_rng(0)
            for _ in range(20):
                obs, cobs, r, done, info = env.step(rng.normal(0, 0.3, (8, env.act_dim)))
                assert np.isfinite(obs).all() and np.isfinite(r).all()
            finals.append(env.state.copy())
            # force env 0 to fall: it must be terminated, reported and reset in place
            env.state[0, 3] = 0.2  # qpos[2] = root height
            _, _, _, done, info = env.step(np.zeros((8, env.act_dim)))
            assert done[0] and info["terminated"][0] and info["episode"]["fell"] > 0
            assert env.state[0, 3] > 0.9 * env.task.z0 and env.ts.t[0] == 0 and env.completed
    assert np.allclose(finals[0], finals[1])  # same seed, different thread counts -> same physics


def test_domain_randomisation_variants(humanoid):
    cfg = VelTaskConfig(n_model_variants=4)
    models, props = make_variants(humanoid, cfg, np.random.default_rng(0))
    assert len(models) == 4 and np.allclose(props[0], [1, 0, 1]) and models[0] is humanoid
    root = int(humanoid.jnt_bodyid[0])
    for m, (f, dm, ks) in zip(models[1:], props[1:]):
        assert cfg.friction_range[0] <= f <= cfg.friction_range[1]
        assert np.isclose(m.geom_friction[0, 0], humanoid.geom_friction[0, 0] * f)
        assert np.isclose(m.body_mass[root], humanoid.body_mass[root] + dm)
        assert np.isclose(m.actuator_gainprm[0, 0], humanoid.actuator_gainprm[0, 0] * ks)
    assert humanoid.geom_friction[0, 0] == 1.0  # the nominal model is never modified


def test_vec_env_throughput(humanoid):
    """Measured, not asserted tightly (the CPU is shared): 4 threads, 128 envs, random actions."""
    with VecEnv(humanoid, VelTaskConfig(), num_envs=128, nthread=4, seed=0) as env:
        env.reset()
        rng = np.random.default_rng(0)
        a = rng.normal(0, 0.3, (128, env.act_dim))
        env.step(a)
        t0 = time.perf_counter()
        for _ in range(30):
            env.step(a)
        sps = 128 * 30 / (time.perf_counter() - t0)
    print(f"vec env: {sps:.0f} env-steps/s ({sps * 4:.0f} physics steps/s) with 4 threads, 128 envs")
    assert sps > 2000


# ---- trainer, policy file, controller ---------------------------------------------------------------------
def test_ppo_trainer_runs_and_saves_loadable_policy(tmp_path, humanoid):
    cfg = PPOConfig(num_envs=16, steps_per_env=8, iterations=2, threads=2, blas_threads=2, name="t",
                    actor_hidden=[32, 32], critic_hidden=[32, 32], save_interval=1)
    tr = PPOTrainer(cfg, HumanoidParams(), VelTaskConfig(), runs_dir=tmp_path)
    tr.run()
    st = tr.status()
    assert st["generation"] == 2 and st["generations"] == 2 and st["policy"]
    for key in ("gen", "best", "mean", "center", "sps", "track_lin", "ep_len_s", "kl", "lr"):
        assert key in st["history"][-1]
    json.dumps(st)  # the GUI serialises it
    path = tmp_path / "t" / "rl_policy.npz"
    assert is_rl_policy(path) and (tmp_path / "t" / "rl_checkpoint.npz").exists()
    pol = RLPolicy.load(path)
    assert pol.meta["algo"] == "ppo" and pol.meta["joint_names"] == VelocityTask(humanoid).joint_names
    obs = np.zeros((3, pol.obs_dim))
    assert pol.act(obs).shape == (3, 14)
    # resume from the checkpoint into a new trainer
    tr2 = PPOTrainer(PPOConfig(**{**cfg.to_dict(), "iterations": 1, "name": "t2"}), HumanoidParams(),
                     VelTaskConfig(), runs_dir=tmp_path, init_checkpoint=tmp_path / "t" / "rl_checkpoint.npz")
    tr2.run()
    assert RLPolicy.load(tmp_path / "t2" / "rl_policy.npz").obs_norm.count > pol.obs_norm.count  # continued


def test_trainer_thread_lifecycle(tmp_path):
    cfg = PPOConfig(num_envs=8, steps_per_env=4, iterations=1000, threads=2, blas_threads=1, name="bg",
                    actor_hidden=[16], critic_hidden=[16])
    tr = PPOTrainer(cfg, HumanoidParams(), VelTaskConfig(), runs_dir=tmp_path)
    tr.start()
    t0 = time.perf_counter()
    while tr.status()["generation"] < 2 and time.perf_counter() - t0 < 30:
        time.sleep(0.05)
    tr.stop()
    tr.join()
    assert tr.status()["state"] == "stopped" and tr.status()["policy"]


def _run_controller(model, ctrl, seconds, path=None, velocity=None):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    ctrl.reset(model, data)
    ctrl.set_path(path)
    if velocity is not None:
        ctrl.set_velocity(*velocity)
    sub = max(1, int(round(ctrl.control_dt / model.opt.timestep)))
    for _ in range(int(round(seconds / ctrl.control_dt))):
        ctrl.step(model, data)
        mujoco.mj_step(model, data, sub)
    return data


def test_distilled_student_stands_and_drives_rl_controller(tmp_path, humanoid):
    tcfg = VelTaskConfig(push_vel=0.25, push_interval_s=2.0)
    teacher = lambda m: StandingTeacher(m, tcfg, k_tilt=0.5, k_rate=0.05)  # state-dependent labels
    pol, report = distill(humanoid, teacher, tcfg,
                          DistillConfig(num_envs=6, rounds=3, episode_seconds=3.0, epochs=15, hidden=[64, 64]),
                          run_dir=tmp_path / "student")
    assert [r["round"] for r in report] == [0, 1, 2] and report[-1]["beta"] < 0.1
    assert report[-1]["train_mse"] < 1e-2 and report[-1]["rollout_disagreement"] < 1e-2
    assert is_rl_policy(tmp_path / "student" / "rl_policy.npz")

    ctrl = RLController(humanoid, tmp_path / "student" / "rl_policy.npz")
    assert isinstance(ctrl, Controller) and ctrl.name == "rl"
    assert ctrl.control_dt == pytest.approx(4 * humanoid.opt.timestep)
    data = _run_controller(humanoid, ctrl, 5.0)
    z0 = VelocityTask(humanoid).z0
    assert data.qpos[2] > 0.95 * z0 and ctrl.status()["mode"] == "stand"  # stood for 5 s
    assert np.hypot(*data.qpos[:2]) < 0.3
    ev = evaluate_policy(pol, humanoid, (0.0, 0.0, 0.0), seconds=5.0, num_envs=4, nthread=2)
    assert ev["fall_rate"] == 0.0


def test_rl_controller_commands_and_robot_check(tmp_path, humanoid):
    pol, _ = distill(humanoid, lambda m: StandingTeacher(m), VelTaskConfig(),
                     DistillConfig(num_envs=2, rounds=1, episode_seconds=0.5, epochs=2, hidden=[16]))
    ctrl = RLController(humanoid, pol)
    data = _run_controller(humanoid, ctrl, 0.2, path=np.array([[1.0, 0.0], [3.0, 0.0]]), velocity=None)
    st = ctrl.status()
    assert st["mode"] == "walk" and st["command"][0] > 0.1 and "goal_error_m" in st
    lo, hi = ctrl.task.cfg.cmd_vx
    ctrl.set_velocity(5.0, 0.0, 0.0)  # clamped to the training range
    ctrl.step(humanoid, data)
    assert ctrl.status()["command"][0] == pytest.approx(hi)
    arms = ctrl.task.arm_idx
    assert np.allclose(data.ctrl[arms], ctrl.task.ctrl_default[arms])
    # a robot with different leg joints is rejected
    other = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><geom type="plane" size="5 5 .1"/>
      <body name="b" pos="0 0 .5"><freejoint/><geom type="box" size=".1 .1 .1"/>
        <body pos="0 0 -.1"><joint name="leg_a" axis="0 1 0" range="-1 1"/><geom type="capsule" fromto="0 0 0 0 0 -.3" size=".03"/></body>
      </body></worldbody>
      <actuator><position joint="leg_a" kp="50"/></actuator></mujoco>""")
    with pytest.raises(ValueError):
        RLController(other, pol)


# ---- regressions found in review --------------------------------------------------------------------------
def _tiny_rl_policy(model, cfg=None):
    cfg = cfg or VelTaskConfig()
    task = VelocityTask(model, cfg)
    actor = GaussianPolicy(task.obs_dim, task.act_dim, [8], "elu", 0.3, np.random.default_rng(0))
    return RLPolicy(actor, None, {"task": cfg.to_dict(), "joint_names": task.joint_names, "activation": "elu"})


def test_rl_controller_stop_cancels_velocity_command(humanoid):
    """set_path(None) / an empty path (session reset, navigator stop) must stop a set_velocity command, as the
    Controller contract and WalkController do; it used to keep walking while reporting "stand"."""
    ctrl = RLController(humanoid, _tiny_rl_policy(humanoid))
    data = mujoco.MjData(humanoid)
    mujoco.mj_forward(humanoid, data)
    ctrl.reset(humanoid, data)
    for stop in (None, np.zeros((0, 2))):
        ctrl.set_velocity(0.5, 0.0, 0.3)
        ctrl.step(humanoid, data)
        assert ctrl.status()["command"] == pytest.approx([0.5, 0.0, 0.3]) and ctrl.status()["mode"] == "walk"
        ctrl.set_path(stop)
        ctrl.step(humanoid, data)
        assert ctrl.status()["command"] == [0.0, 0.0, 0.0] and ctrl.status()["mode"] == "stand"


def test_blas_thread_limit_survives_overlapping_threads(monkeypatch):
    """Two threads' blas_threads() blocks ending out of order used to leave the whole process at the other
    thread's limit (each threadpool_limits restores what it saw on entry). A fake threadpoolctl with a
    process-global setting makes the interleaving deterministic."""
    import threading
    tp = pytest.importorskip("threadpoolctl")
    from simulatics.train.ppo import blas_threads
    setting = {"n": 16}

    class FakeLimits:
        def __init__(self, limits, user_api=None):
            self.saved, setting["n"] = setting["n"], limits

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            setting["n"] = self.saved

    monkeypatch.setattr(tp, "threadpool_limits", FakeLimits)
    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()
    seen = {}

    def a():
        with blas_threads(1):
            a_in.set()
            b_in.wait(5)
        a_out.set()

    def b():
        a_in.wait(5)
        with blas_threads(3):
            seen["b"] = setting["n"]  # another thread holds the limit: B runs with it
            b_in.set()
            a_out.wait(5)  # A's block ends first

    ts = [threading.Thread(target=f) for f in (a, b)]
    [t.start() for t in ts]
    [t.join(10) for t in ts]
    assert seen["b"] == 1 and setting["n"] == 16  # the process gets its own setting back
    with blas_threads(4):  # and a later user can still set it
        assert setting["n"] == 4
    assert setting["n"] == 16


def test_save_npz_retries_while_a_reader_holds_the_file(tmp_path):
    """On Windows renaming over an open file raises PermissionError; a GUI read during a save used to kill
    the training run."""
    import threading
    from simulatics.train.nn import load_npz, save_npz
    path = tmp_path / "rl_policy.npz"
    save_npz(path, {"a": np.zeros(3)}, {"v": 1})
    reader = open(path, "rb")
    threading.Timer(0.2, reader.close).start()
    save_npz(path, {"a": np.ones(3)}, {"v": 2})
    arrays, meta = load_npz(path)
    assert meta["v"] == 2 and np.all(arrays["a"] == 1) and not (tmp_path / "rl_policy.tmp.npz").exists()


def test_evaluate_policy_starts_with_an_empty_history(humanoid):
    """With frame history, evaluation used to start with a stale frame (the reset's random command) in it."""
    cfg = VelTaskConfig(history=2)
    pol = _tiny_rl_policy(humanoid, cfg)
    seen = []
    act = pol.act
    pol.act = lambda obs: (seen.append(obs.copy()), act(obs))[1]
    evaluate_policy(pol, humanoid, (0.3, 0.0, 0.0), seconds=0.04, num_envs=2, nthread=1)
    p = VelocityTask(humanoid, cfg).proprio_dim
    assert np.all(seen[0][:, p:] == 0.0)  # no past frames yet
    assert np.allclose(seen[1][:, p:2 * p], seen[0][:, :p])  # then the previous frame, with the real command


# ---- opt-in: real PPO smoke run on the humanoid -----------------------------------------------------------
@pytest.mark.skipif(os.environ.get("SIMULATICS_SLOW") != "1", reason="90 s, 8 threads: set SIMULATICS_SLOW=1")
def test_ppo_humanoid_smoke_run_tracking_rises(tmp_path):
    seconds = float(os.environ.get("SIMULATICS_SMOKE_SECONDS", "90"))
    cfg = PPOConfig(num_envs=256, iterations=100000, threads=8, blas_threads=4, max_seconds=seconds, name="smoke", seed=1)
    tr = PPOTrainer(cfg, HumanoidParams(), VelTaskConfig(), runs_dir=tmp_path)
    tr.run()
    h = tr.history
    k = max(3, len(h) // 10)
    early = np.mean([r["track_lin"] for r in h[k:2 * k]])  # skip the first iterations: no episode has ended yet
    late = np.mean([r["track_lin"] for r in h[-k:]])
    for r in h[:: max(1, len(h) // 15)] + [h[-1]]:
        print(f"it {r['gen']:4d} track_lin {r['track_lin']:.3f} track_ang {r['track_ang']:.3f} "
              f"ep_len {r['ep_len_s']:.2f}s return {r['center']:.2f} env_sps {r['env_sps']:.0f}")
    assert late > early and h[-1]["ep_len_s"] > h[k]["ep_len_s"]
