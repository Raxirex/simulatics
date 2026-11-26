"""The Go2 trot controller, driven through the live session and the unchanged Navigator."""
import math
import time

import mujoco
import numpy as np
import pytest

from simulatics import robots
from simulatics.control.quadruped import QuadWalkController

pytestmark = pytest.mark.skipif(not robots.available("go2"), reason="the Go2 model is not downloaded")


def _pose(sess):
    """(xy, yaw, up_z, height, sim time) under the session lock."""
    with sess.lock:
        q = sess.data.qpos[:7].copy()
        t = float(sess.data.time)
    w, x, y, z = q[3:7]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return q[:2], yaw, 1 - 2 * (x * x + y * y), float(q[2]), t


class Track:
    """Samples the robot while a condition is awaited: lowest up-vector z, closest pass to given points."""

    def __init__(self, sess, points=()):
        self.sess, self.points = sess, np.asarray(points, float).reshape(-1, 2)
        self.min_up, self.closest, self.top_speed = 1.0, np.full(len(self.points), np.inf), 0.0

    def wait(self, cond, timeout: float) -> bool:
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            xy, _, up, _, _ = _pose(self.sess)
            self.min_up = min(self.min_up, up)
            self.top_speed = max(self.top_speed, float(self.sess.controller.status().get("speed") or 0.0))
            if len(self.points):
                self.closest = np.minimum(self.closest, np.linalg.norm(self.points - xy, axis=1))
            if cond():
                return True
            time.sleep(0.004)
        return False


@pytest.fixture(scope="module")
def sess():
    from simulatics.server.session import SimSession
    s = SimSession()
    try:
        s.lidar_cfg.enabled = False
        s.auto_reset = False            # a fall must fail the test, not be repaired
        s.load_robot("go2")
        with s.lock:
            c = QuadWalkController(s.model)
            c.reset(s.model, s.data)
            s.controller = c
        s.speed = 8.0
        yield s
    finally:
        s.running = False
        s.shutdown()


@pytest.fixture
def dog(sess):
    """The session with the dog standing at the origin, heading +x, and no goal."""
    with sess.lock:
        sess.running = False
        sess.nav.clear()
        sess.nav.take_new_path()
        c = sess.controller
        assert isinstance(c, QuadWalkController)
        sess.data.qpos[:2] = 0.0
        sess.data.qpos[3:7] = (1, 0, 0, 0)
        c.reset(sess.model, sess.data)
        sess.falls = 0
    yield sess
    with sess.lock:
        sess.running = False
    assert sess.controller is c, "the session replaced the controller (a rebuild or a fall reset)"


def _go(sess, goal, yaw=None, timeout=90.0, points=()):
    with sess.lock:
        sess.nav.set_goal(goal, yaw)
        sess.running = True
    tr = Track(sess, points)
    assert tr.wait(lambda: sess.nav.state in ("reached", "failed", "fallen"), timeout), \
        f"no result: nav {sess.nav.state}, controller {sess.controller.status()}"
    return tr


def test_supports_the_go2_but_not_the_humanoids():
    from simulatics.models import HumanoidParams, build_mjcf
    assert not QuadWalkController.supports(mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams())))
    assert QuadWalkController.supports(mujoco.MjModel.from_xml_path(str(robots.path("go2"))))
    if robots.available("adam_lite"):
        assert not QuadWalkController.supports(mujoco.MjModel.from_xml_path(str(robots.path("adam_lite"))))


def test_leg_kinematics_match_the_model():
    m = mujoco.MjModel.from_xml_path(str(robots.path("go2")))
    d, c = mujoco.MjData(m), QuadWalkController(m)
    assert abs(c.control_dt / m.opt.timestep - round(c.control_dt / m.opt.timestep)) < 1e-9
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = np.stack([rng.uniform(-0.6, 0.6, 4), rng.uniform(0.0, 1.8, 4), rng.uniform(-2.5, -1.0, 4)], axis=1)
        d.qpos[:] = m.qpos0
        d.qpos[c.qadr] = q
        mujoco.mj_forward(m, d)
        rot = d.xmat[1].reshape(3, 3)
        feet = np.array([rot.T @ (d.geom_xpos[m.geom(n).id] - d.xpos[1]) for n in ("FL", "FR", "RL", "RR")])
        fk = c._fk(q)
        assert np.abs(feet - (c.hip_off + fk)).max() < 0.003       # the foot sphere sits 2 mm off the shank axis
        assert np.abs(c._ik(fk) - q).max() < 1e-6


def test_stands_still(dog):
    sess = dog
    xy0, yaw0, _, z0, t0 = _pose(sess)
    with sess.lock:
        sess.running = True
    tr = Track(sess)
    zs = []

    def five_seconds():
        zs.append(_pose(sess)[3])
        return _pose(sess)[4] - t0 >= 5.0
    assert tr.wait(five_seconds, 60.0)
    xy, yaw, up, z, _ = _pose(sess)
    assert max(abs(v - z0) for v in zs) < 0.02
    assert math.degrees(math.acos(min(1.0, tr.min_up))) < 3.0
    assert np.linalg.norm(xy - xy0) < 0.03 and abs(yaw - yaw0) < math.radians(3)
    assert sess.falls == 0
    assert sess.controller.status()["mode"] == "stand"


def test_walks_to_clicked_points(dog):
    sess = dog
    for goal in ([2.0, 1.0], [-1.0, -1.5]):
        tr = _go(sess, goal)
        assert sess.nav.state == "reached", sess.nav.message
        assert tr.wait(lambda: sess.controller.status()["mode"] == "stand", 10.0)   # the navigator cleared the path
        xy = _pose(sess)[0]
        assert np.linalg.norm(xy - goal) < 0.08
        assert tr.min_up > 0.9
        assert tr.top_speed > 0.3
        assert sess.falls == 0
    # it stays where it stopped
    t0 = _pose(sess)[4]
    assert Track(sess).wait(lambda: _pose(sess)[4] - t0 > 1.0, 20.0)
    assert np.linalg.norm(_pose(sess)[0] - [-1.0, -1.5]) < 0.08


def test_follows_a_path_with_corners(dog):
    sess = dog
    pts = np.array([[1, 0], [1, 1.5], [2.5, 1.5]], float)
    with sess.lock:
        sess.controller.set_path(pts)
        sess.running = True
    assert sess.controller.status()["mode"] == "walk"
    tr = Track(sess, pts)
    assert tr.wait(lambda: sess.controller.status()["mode"] in ("done", "fallen"), 90.0)
    st = sess.controller.status()
    assert st["mode"] == "done", st
    assert (tr.closest < 0.15).all(), tr.closest
    assert np.linalg.norm(_pose(sess)[0] - pts[-1]) < 0.08
    assert st["goal_error_m"] < 0.08
    assert tr.min_up > 0.9 and sess.falls == 0
    with sess.lock:
        sess.controller.set_path(None)
    assert sess.controller.status()["mode"] == "stand"


def test_final_yaw_is_honoured(dog):
    sess = dog
    for goal, yaw in (([1.2, -0.6], math.radians(135)), ([1.2, -0.6], math.radians(-60))):   # second: turn on the spot
        tr = _go(sess, goal, yaw)
        assert sess.nav.state == "reached", sess.nav.message
        xy, got, _, _, _ = _pose(sess)
        assert abs((got - yaw + math.pi) % (2 * math.pi) - math.pi) < math.radians(10)
        assert np.linalg.norm(xy - goal) < 0.08
        assert tr.min_up > 0.9 and sess.falls == 0


def test_a_step_is_fast():
    m = mujoco.MjModel.from_xml_path(str(robots.path("go2")))
    d, c = mujoco.MjData(m), QuadWalkController(m)
    c.reset(m, d)
    c.set_velocity(0.4, 0.0, 0.3)
    n = max(1, int(round(c.control_dt / m.opt.timestep)))
    for _ in range(300):
        c.step(m, d)
        mujoco.mj_step(m, d, n)
    spent = 0.0
    for _ in range(2000):
        t = time.perf_counter()
        c.step(m, d)
        spent += time.perf_counter() - t
        mujoco.mj_step(m, d, n)
    assert c.status()["mode"] == "walk" and c.up_z > 0.9
    assert spent / 2000 < 0.5e-3, f"{spent / 2000 * 1e3:.3f} ms per step"
