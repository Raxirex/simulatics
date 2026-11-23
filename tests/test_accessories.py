import time

import mujoco
import numpy as np
import pytest

from simulatics.models import HumanoidParams, build_mjcf
from simulatics.models.accessories import apply_accessories, default_accessories, place
from simulatics.server.session import SimSession


def wait_for(cond, timeout):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_default_lidar_is_an_accessory_and_model_is_unchanged_otherwise():
    xml = build_mjcf(HumanoidParams())
    acc = default_accessories(xml)
    assert len(acc) == 1 and acc[0]["type"] == "lidar" and acc[0]["body"] == "torso"
    m0 = mujoco.MjModel.from_xml_string(xml)
    out, warn = apply_accessories(xml, acc)
    m1 = mujoco.MjModel.from_xml_string(out)
    assert not warn
    assert m1.ngeom == m0.ngeom + 1 and m1.nu == m0.nu  # one visual puck, same actuators
    s = mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_SITE, "lidar")
    assert s >= 0 and np.allclose(m1.site_pos[s], acc[0]["pos"])


def test_place_snaps_to_grid_and_sits_on_the_surface():
    m = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams()))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "torso")
    b = m.geom_bodyid[g]
    point = d.xpos[b] + np.array([0.1003, 0.012, 0.051])  # on the chest front
    lid = place(m, d, "lidar", g, point, [1, 0, 0], snap=0.01)
    assert lid["body"] == "torso"
    assert np.allclose(lid["pos"], [0.1 + 0.025, 0.01, 0.05], atol=1e-6)  # snapped, then out along the normal
    cam = place(m, d, "camera", g, point, [1, 0, 0], snap=0.0)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.array(cam["quat"]))
    assert np.allclose(R.reshape(3, 3)[:, 2], [-1, 0, 0], atol=1e-6)  # looks out of the chest (-z forward)
    with pytest.raises(ValueError):
        place(m, d, "lidar", 0, point, [0, 0, 1])  # the floor is not the robot


@pytest.fixture
def sess():
    s = SimSession()
    s.lidar_cfg.enabled = True
    s.speed = 8.0
    yield s
    s.running = False
    s.shutdown()


def test_session_moves_the_lidar_and_robot_still_walks(sess):
    sess.workshop(True)
    with sess.lock:
        g = mujoco.mj_name2id(sess.model, mujoco.mjtObj.mjOBJ_GEOM, "torso")
        b = sess.model.geom_bodyid[g]
        point = (sess.data.xpos[b] + np.array([0.1, 0.0, 0.05])).tolist()
    acc = sess.place_accessory("lidar", g, point, [1, 0, 0], 0.01)
    assert [a["type"] for a in sess.accessories].count("lidar") == 1  # moved, not duplicated
    with sess.lock:
        site = mujoco.mj_name2id(sess.model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
        origin, pts, dist, hit = sess.lidar.scan(sess.model, sess.data)
        assert np.allclose(sess.model.site_pos[site], acc["pos"])
        assert origin[0] > sess.data.xpos[b][0] + 0.05  # scanning from the chest now
        assert not (hit & (dist < 0.1)).any()           # never sees its own body
    cam = sess.place_accessory("camera", mujoco.mj_name2id(sess.model, mujoco.mjtObj.mjOBJ_GEOM, "left_thigh"),
                               point, [1, 0, 0], 0.0)
    assert f"cam_{cam['id']}" in [sess.model.camera(i).name for i in range(sess.model.ncam)]
    sess.update_accessory(cam["id"], {"fovy": 100})
    assert sess.model.cam_fovy[mujoco.mj_name2id(sess.model, mujoco.mjtObj.mjOBJ_CAMERA, f"cam_{cam['id']}")] == 100
    with sess.lock:
        sess.nav.set_goal([1.5, 0.5])
        sess.running = True
    assert wait_for(lambda: sess.nav.state in ("reached", "failed", "fallen"), 60)
    assert sess.nav.state == "reached" and sess.falls == 0
    sess.delete_accessory(cam["id"])
    assert all(a["id"] != cam["id"] for a in sess.accessories)
