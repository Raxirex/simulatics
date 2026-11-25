"""Ready-made samples: every one starts, the physics ones do what they say, the reference clips are sane."""
import time

import mujoco
import numpy as np
import pytest

from simulatics import samples
from simulatics.server.session import SimSession


def wait_for(cond, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


@pytest.fixture(scope="module")
def sess():
    s = SimSession()
    s.lidar_cfg.enabled = False
    yield s
    s.running = False
    s.shutdown()


def test_listing_names_the_kinds_the_user_asked_for():
    ids = set(samples.SAMPLES)
    assert {"stand", "stand_pushes", "one_hand_wave", "walk_forward", "walk_points", "walk_around", "stairs",
            "playground", "parkour_course", "kitchen"} <= ids
    for s in samples.SAMPLES.values():
        assert s.kind in ("physics", "reference") and s.title and s.description and s.group
        assert s.kind == "physics" or s.source, f"{s.id}: a reference clip must say where it comes from"
    with pytest.raises(ValueError):
        samples.run(None, "no such sample")


@pytest.mark.parametrize("sid,seconds", [("stand", 3.0), ("stand_pushes", 9.5), ("one_hand_wave", 4.0)])
def test_standing_samples_stay_up(sess, sid, seconds):
    msg = samples.run(sess, sid)
    assert "physics" in msg
    sess.speed = 8.0
    falls, t0 = sess.falls, float(sess.data.time)
    assert wait_for(lambda: sess.data.time - t0 > seconds, 60)
    assert sess.falls == falls and float(sess.data.qpos[2]) > 1.0


@pytest.mark.parametrize("sid,goal", [("walk_forward", (4.0, 0.0)), ("walk_points", (0.0, 0.0)),
                                      ("walk_around", (5.0, 0.0)), ("stairs", None)])
def test_walking_samples_reach_their_goal(sess, sid, goal):
    samples.run(sess, sid)
    sess.speed = 8.0
    falls = sess.falls
    assert wait_for(lambda: sess.nav.state in ("reached", "failed", "fallen"), 150)
    assert sess.nav.state == "reached" and sess.falls == falls
    if goal is not None:
        assert np.linalg.norm(sess.data.qpos[:2] - goal) < 0.05
    else:
        assert float(sess.data.qpos[0]) > 3.0            # past the stairs


def test_walk_points_visits_every_waypoint(sess):
    samples.run(sess, "walk_points")
    sess.speed = 8.0
    seen = set()

    def step():
        with sess.lock:
            p = sess.data.qpos[:2].copy()
        for i, w in enumerate(samples.WAYPOINTS):
            if np.linalg.norm(p - w) < 0.25:
                seen.add(i)
        return sess.nav.state in ("reached", "failed", "fallen")
    assert wait_for(step, 150)
    assert seen == {0, 1, 2, 3}


REFERENCE = [s.id for s in samples.SAMPLES.values() if s.kind == "reference"]


@pytest.mark.skipif(not REFERENCE, reason="reference clips not built (python -m simulatics.samples.build)")
@pytest.mark.parametrize("sid", REFERENCE)
def test_reference_clip_plays_without_falling_or_sinking(sess, sid):
    from simulatics.imitate import load_motion
    clip = samples.manifest()[sid.removesuffix("_reference")]
    motion = load_motion(str(samples.DATA / clip["file"]))
    assert 0.5 < motion.duration < 20 and np.isfinite(motion.root_pos).all() and np.isfinite(motion.joint_pos).all()
    msg = samples.run(sess, sid)
    assert "reference" in msg and sess.controller.name == "motion-preview"
    sess.speed = 4.0
    falls, t0 = sess.falls, float(sess.data.time)
    low = []

    def step():
        with sess.lock:
            m, d = sess.model, sess.data
            robot = np.flatnonzero(m.body_rootid[m.geom_bodyid] == m.body_rootid[1])
            low.append(float(np.min(d.geom_xpos[robot, 2] - m.geom_rbound[robot])))
        return sess.data.time - t0 > min(motion.duration, 6.0)
    assert wait_for(step, 60)
    assert sess.falls == falls                           # a preview is never counted as a fall
    assert min(low) > -0.12, "the robot goes through the floor"
    assert len(sess.obstacles) == len(clip.get("props") or [])


def test_flips_really_rotate_and_jumps_really_leave_the_ground():
    from simulatics.imitate import load_motion
    from simulatics.imitate.motion import quat_to_mat
    man = samples.manifest()
    if not man:
        pytest.skip("reference clips not built")
    for sid in ("backflip", "front_flip"):
        if sid in man:
            m = load_motion(str(samples.DATA / man[sid]["file"]))
            up = quat_to_mat(m.root_quat)[:, 2, 2]       # world z of the torso's up axis
            assert up.min() < -0.5, f"{sid}: the torso never turns upside down"
            assert up[0] > 0.7 and up[-1] > 0.7, f"{sid}: does not start and end upright"
    for sid, c in man.items():
        if c["group"] == "Parkour":
            m = load_motion(str(samples.DATA / c["file"]))
            assert m.root_pos[:, 2].max() - m.root_pos[0, 2] > 0.15, f"{sid}: the pelvis never rises"
