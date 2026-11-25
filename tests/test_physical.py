"""Physically simulated motions: what "physical" means here, checked in the live session.

For every verified recorded motion: it starts from a standing robot at rest; while it plays the only thing
written is the motor command (an independent mj_step replay of the same commands gives the same motion); the
robot really leaves the ground / really rotates; and afterwards the balance controller takes over without a
reset and the robot is upright and still, with no fall counted.
"""
import time

import mujoco
import numpy as np
import pytest

from simulatics import samples
from simulatics.imitate.physical import STATE, Tape, TapeController, fingerprint, prepare_model
from simulatics.server.session import SimSession

PHYS = {k: v for k, v in samples.physical().items() if v.get("verified") and (samples.DATA / v["file"]).exists()}


@pytest.fixture(scope="module")
def sess():
    s = SimSession()
    s.lidar_cfg.enabled = False
    yield s
    s.running = False
    s.shutdown()


def test_the_motions_the_release_promises_are_physical():
    assert {"vertical_jump", "broad_jump", "hurdle", "box_jump_down", "backflip"} <= set(PHYS)
    for sid in PHYS:
        assert samples.SAMPLES[sid].kind == "physics"


@pytest.mark.parametrize("sid", sorted(PHYS))
def test_recorded_motion_is_pure_physics_and_ends_standing(sess, sid):
    entry = PHYS[sid]
    tape = Tape.load(samples.DATA / entry["file"])
    # starts as a standing robot at rest
    assert not tape.qvel0.any()
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, tape.qpos0[3:7])
    assert R[8] > 0.99
    msg = samples.run(sess, sid)
    assert "physics" in msg and isinstance(sess.controller, TapeController)
    with sess.lock:
        m = sess.model
        assert tape.matches(m)
        assert not any(m.equality(i).name == "support_stand" for i in range(m.neq))
        xml = sess.export_xml()
        assert np.allclose(sess.data.qpos, tape.qpos0) and not sess.data.qvel.any()
    sess.speed = 8.0
    falls = sess.falls
    floor_like = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
    seen, air, up, z, forced, still = {}, 0, [], [], False, 0
    end_t = None
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        with sess.lock:
            c, d = sess.controller, sess.data
            forced = forced or bool(d.xfrc_applied.any())
            up.append(float(d.xmat[1].reshape(3, 3)[2, 2]))
            z.append(float(d.qpos[2]))
            if isinstance(c, TapeController):
                seen[c.i] = d.qpos.copy()
                touching = any((d.contact[i].geom1 in floor_like) != (d.contact[i].geom2 in floor_like) for i in range(d.ncon))
                air += 0 if touching else 1
            elif end_t is None:
                end_t = float(d.time)
            now, speed = float(d.time), float(np.linalg.norm(d.qvel[:3]))
        if end_t is not None:                                   # the walker may take a step or two to square up
            still = still + 1 if speed < 0.05 else 0
            if now - end_t > 12.0 or (still > 300 and now - end_t > 2.0):
                break
        time.sleep(0.001)
    assert end_t is not None, "the motion never finished"
    assert not forced                                           # no external force was ever applied
    # independent replay: the same commands through plain mj_step, nothing else
    ref = mujoco.MjModel.from_xml_string(xml)
    prepare_model(ref)
    assert fingerprint(ref) == tape.model
    d2 = mujoco.MjData(ref)
    d2.qpos[:] = tape.qpos0
    mujoco.mj_forward(ref, d2)
    replay = [d2.qpos.copy()]
    for u in tape.ctrl:
        d2.ctrl[:] = u
        mujoco.mj_step(ref, d2, tape.control_steps)
        replay.append(d2.qpos.copy())
    replay = np.array(replay)
    assert np.abs(replay - tape.states).max() < 1e-3            # the stored motion is what the commands produce
    early = [i for i in seen if 0 < i <= len(tape.ctrl) // 3]
    assert early and max(np.abs(seen[i] - replay[i]).max() for i in early) < 1e-3   # and what the session did
    live = entry["live"]
    if "jump" in sid or sid in ("hurdle", "backflip", "front_flip"):
        assert air > 3 and live["flight_s"] >= 0.2              # really off the ground
    if sid in ("backflip", "front_flip"):
        assert min(up) < -0.8 and live["rotation_rad"] > 5.5    # really went over
    if sid == "box_jump_up":
        assert z[-1] - z[0] > 0.4                               # ended up on the box
    if sid == "box_jump_down":
        assert z[0] - z[-1] > 0.4
    with sess.lock:
        assert sess.controller.name == "walker"                 # balance took over, with no reset in between
        assert sess.falls == falls and up[-1] > 0.95 and speed < 0.1
        assert not (sess.model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_WARMSTART))   # settings restored
        assert sess.model.actuator_ctrllimited.all()
    sess.running = False


def test_a_tape_refuses_a_different_robot(sess):
    sid = "vertical_jump"
    tape = Tape.load(samples.DATA / PHYS[sid]["file"])
    samples._builtin(sess)
    sess.set_params({"thigh_length": 0.43})                     # a longer thigh: not the robot it was recorded on
    with pytest.raises(ValueError, match="different robot"):
        sess.play_tape(tape)
    assert sess.controller.name == "walker"
    samples._builtin(sess)


def test_walking_after_a_recorded_motion(sess):
    samples.run(sess, "vertical_jump")
    sess.speed = 8.0
    end = time.monotonic() + 60
    while time.monotonic() < end and sess.controller.name != "walker":
        time.sleep(0.02)
    assert sess.controller.name == "walker"
    time.sleep(0.5)
    falls = sess.falls
    with sess.lock:
        sess.nav.set_goal([1.5, 0.0])
    end = time.monotonic() + 90
    while time.monotonic() < end and sess.nav.state not in ("reached", "failed", "fallen"):
        time.sleep(0.05)
    assert sess.nav.state == "reached" and sess.falls == falls
    sess.running = False


def test_tracker_makes_a_small_hop_physical():
    """The tracker itself, on a tiny problem: a short crouch-and-rise must come out as commands that reproduce."""
    from simulatics.imitate import load_motion
    from simulatics.imitate.physical import TrackConfig, Tracker
    s = SimSession()
    try:
        s.lidar_cfg.enabled = False
        samples._builtin(s)
        xml, q0 = s.export_xml(), s.data.qpos.copy()
    finally:
        s.shutdown()
    tr = Tracker(xml, TrackConfig(samples=300, keep=30, refine=2, passes=1, hold=0.3))
    try:
        tape = tr.track(load_motion("squat"), q0, name="squat")
        assert tape.ctrl.shape[1] == tr.plain.nu and len(tape.states) == len(tape.ctrl) + 1
        assert tape.meta["final_up"] > 0.95 and abs(tape.meta["final_height_m"] - tape.meta["reference_final_height_m"]) < 0.1
        s0 = np.zeros(mujoco.mj_stateSize(tr.plain, STATE))
        d = mujoco.MjData(tr.plain)
        d.qpos[:] = tape.qpos0
        mujoco.mj_getState(tr.plain, d, s0, STATE)
        assert np.allclose(tr.replay(s0, tape.ctrl), tape.states)          # deterministic
    finally:
        tr.close()
