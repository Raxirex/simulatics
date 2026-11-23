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


def test_default_lidar_setting_removes_and_restores_the_head_lidar(sess):
    sess.set_default_lidar(False)
    assert not sess.lidar.mounted and not any(a["type"] == "lidar" for a in sess.accessories)
    sess.set_params({"thigh_length": 0.42})            # survives rebuilds
    assert not sess.lidar.mounted
    with sess.lock:
        o, pts, dist, hit = sess.lidar.scan(sess.model, sess.data)
    assert not hit.any()                                  # no lidar, no returns
    sess.set_default_lidar(True)
    assert sess.lidar.mounted and [a["type"] for a in sess.accessories].count("lidar") == 1


@pytest.mark.parametrize("name,stand", [("go2", False), ("adam_lite", True)])
def test_library_robot_holds_its_pose_and_takes_a_lidar(sess, name, stand):
    from simulatics import robots
    if not robots.available(name):
        pytest.skip(f"{name} not downloaded (python -m simulatics.robots fetch {name})")
    sess.load_robot(name)
    assert sess.controller.name == "pd-hold" and sess.support_stand == stand and not sess.lidar.mounted
    with sess.lock:
        z0 = float(sess.data.qpos[2]); sess.running = True; t0 = sess.data.time
    assert wait_for(lambda: sess.data.time > t0 + 3.0, 60)
    with sess.lock:
        R = sess.data.xmat[1].reshape(3, 3)
        assert abs(float(sess.data.qpos[2]) - z0) < 0.05 and R[2, 2] > 0.98 and sess.falls == 0   # still standing
    sess.workshop(True)
    with sess.lock:
        g = next(g for g in range(sess.model.ngeom) if sess.model.geom_bodyid[g] == 1)
        top = (sess.data.xpos[1] + np.array([0, 0, 0.15])).tolist()
    sess.place_accessory("lidar", g, top, [0, 0, 1], 0.01)
    with sess.lock:
        o, pts, dist, hit = sess.lidar.scan(sess.model, sess.data)
        gid = sess.lidar._geomid
        assert sess.lidar.mounted and hit.sum() > 100
        assert not (hit & (gid >= 0) & sess.lidar.robot_geom[np.maximum(gid, 0)]).any()   # never sees itself
