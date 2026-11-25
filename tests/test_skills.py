"""Skill training: parallel search for a motor program, and the trained uppercut that ships."""
import shutil
import time

import numpy as np
import pytest

from simulatics import samples, skills
from simulatics.imitate.physical import Tape
from simulatics.jobs import RUNS, list_policies, make_trainer
from simulatics.models import HumanoidParams
from simulatics.server.session import SimSession
from simulatics.sim.task import TaskConfig


@pytest.fixture(scope="module")
def sess():
    s = SimSession()
    s.lidar_cfg.enabled = False
    yield s
    s.running = False
    s.shutdown()


@pytest.fixture
def run_name():
    name = f"_test_skill_{int(time.time() * 1000) % 100000}"
    yield name
    shutil.rmtree(RUNS / name, ignore_errors=True)


def test_parallel_rollouts_give_rewards_for_every_candidate_and_variation():
    spec = skills.SKILLS["uppercut"]
    cfg = skills.TrainConfig(population=6, variations=3)
    xml, q0, c0, obstacles, props = skills.scene_for(spec)
    assert len(props) == 1 and props[0]["name"] == "ball"
    env = skills.SkillEnv(spec, cfg, xml, q0, c0)
    try:
        theta = 0.2 * np.random.default_rng(0).standard_normal((6, spec.knots, len(spec.joints)))
        theta[0] = 0.0
        starts = env.starts(3, np.random.default_rng(1))
        assert np.allclose(starts[0, 1:1 + len(q0)], q0) and not np.allclose(starts[1], starts[0])   # varied starts
        r = env.run(theta, starts)
        assert r.n == 18 and r.T == env.T and r.qpos.shape == (18, env.T, len(q0))
        reward = spec.reward(r)
        assert reward.shape == (18,) and np.isfinite(reward).all()
        assert r.prop_pos(0).shape == (18, env.T, 3) and r.body_pos("right_hand").shape == (18, env.T, 3)
        assert np.allclose(env.commands(theta)[0, 0], np.clip(c0, env.lo, env.hi))     # a zero program holds the stance
        assert not np.allclose(env.commands(theta)[1], env.commands(theta)[0])
        assert np.allclose(env.offsets(theta)[:, -env.n_settle:], 0.0)                 # and eases back to it
        assert env.backend.steps_done == 18 * env.T * cfg.control_steps
        with pytest.raises(RuntimeError, match="GPU"):
            skills.GpuBackend()
    finally:
        env.close()


def test_training_improves_the_reward_and_saves_a_playable_recording(run_name):
    spec = skills.SKILLS["uppercut"]
    tr = skills.SkillTrainer(spec, skills.TrainConfig(population=48, variations=2, generations=8, name=run_name))
    tr.start()
    assert tr.status()["state"] == "running"
    end = time.monotonic() + 240
    while time.monotonic() < end and tr.status()["state"] == "running":
        time.sleep(0.2)
    st = tr.status()
    assert st["state"] == "finished", st["error"]
    hist = st["history"]
    assert len(hist) == 8 and st["generation"] == 8 and st["kind"] == "skill"
    best = [h["best"] for h in hist]
    assert all(b >= a - 1e-9 for a, b in zip(best, best[1:])) and best[-1] > best[0]          # it learned something
    assert hist[-1]["sps"] > 1000 and st["total_steps"] > 100000
    tape = Tape.load(st["policy"])
    assert tape.meta["skill"] == "uppercut" and tape.ctrl.shape[1] == 26 and len(tape.states) == len(tape.ctrl) + 1
    assert {"ball_rise_m", "fist_speed_at_contact_mps", "stayed_standing_of_24", "return_to_stance_cost"} <= set(tape.meta["report"])
    assert any(p["name"] == run_name and p["kind"] == "skill" for p in list_policies())


def test_trainer_can_be_stopped_and_jobs_build_it(run_name):
    tr = make_trainer({"algo": "skill", "skill": "uppercut", "name": run_name, "population": 32, "generations": 100000},
                      HumanoidParams(), TaskConfig())
    assert isinstance(tr, skills.SkillTrainer) and tr.cfg.generations == 100000
    tr.start()
    time.sleep(3.0)
    tr.stop()
    tr.join(180)
    st = tr.status()
    assert st["state"] == "stopped" and 0 < st["generation"] < 100000 and st["policy"]       # what it had is kept
    with pytest.raises(ValueError, match="unknown skill"):
        make_trainer({"algo": "skill", "skill": "moonwalk"}, HumanoidParams(), TaskConfig())


def test_a_new_skill_is_just_a_scene_and_a_reward(run_name):
    def raise_hand(r: skills.Rollout) -> np.ndarray:
        return r.body_pos("left_hand")[:, :, 2].max(axis=1) + r.stays_standing()
    spec = skills.SkillSpec("raise_hand", "Raise the left hand", "test", joints=["left_shoulder_y", "left_elbow"],
                            reward=raise_hand, duration=0.6, settle=0.4, knots=4)
    tr = skills.SkillTrainer(spec, skills.TrainConfig(population=32, variations=1, generations=6, name=run_name))
    tape = tr.run()
    r0, r1 = tr.history[0]["center"], tr.history[-1]["best"]
    assert r1 > r0 + 0.15                                    # the hand ends up clearly higher than doing nothing
    assert tape.meta["report"]["stayed_standing_of_24"] >= 20


SHIPPED = skills.shipped()


@pytest.mark.skipif("uppercut" not in SHIPPED, reason="no shipped uppercut (python -m simulatics.skills uppercut --ship)")
def test_the_shipped_uppercut_really_punches_and_the_robot_stays_up(sess):
    e = SHIPPED["uppercut"]
    s = samples.SAMPLES["uppercut"]
    assert s.kind == "physics" and s.group == "Trained skills"
    tape = Tape.load(skills.DATA / e["file"])
    assert not tape.qvel0.any()
    live = skills.verify_live(sess, tape)
    assert live["ok"], live
    assert live["object_rise_m"] > 0.5 and live["falls_after"] == 0          # the ball went up, the robot did not go down
    assert e["report"]["fist_upward_speed_at_contact_mps"] > 2.0 and e["report"]["stayed_standing_of_24"] >= 20
    assert sess.controller.name == "walker"
    msg = samples.run(sess, "uppercut")
    assert "physics" in msg and sess.controller.name == "motion-physical"
    sess.running = False
    samples._builtin(sess)
