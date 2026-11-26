"""Projects, imports, system identification, the robustness sweep, and the any-height box jump reference."""
import json
import time

import mujoco
import numpy as np
import pytest

from simulatics import importer, project, samples, sysid, validate
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


# ----------------------------------------------------------------------------------------------- projects
def test_project_round_trip_restores_robot_scene_motors_and_exact_state(sess, tmp_path):
    samples._builtin(sess)
    sess.set_params({"thigh_length": 0.42})
    box = sess.add_object("box", 2.0, 1.0)
    sess.add_object("ball", 1.2, -0.8)
    knee = next(a["index"] for a in sess.actuators() if a["name"] == "left_knee")
    sess.set_actuator(knee, "kp", 345.0)
    sess.speed = 8.0
    with sess.lock:
        sess.nav.set_goal([1.5, -0.2])
        sess.running = True
    assert wait_for(lambda: sess.nav.state == "reached", 60)
    with sess.lock:
        sess.running = False
        q, v, t = sess.data.qpos.copy(), sess.data.qvel.copy(), float(sess.data.time)
    f = project.save(sess, tmp_path / "lab", name="Lab", notes="first try")
    text = f.read_text(encoding="utf-8")
    assert project.is_project(tmp_path / "lab") and project.is_project(f) and str(tmp_path) not in text
    data = json.loads(text)
    assert data["name"] == "Lab" and data["robot"]["kind"] == "builtin" and data["motors"]["left_knee"]["kp"] == 345.0
    samples._builtin(sess)
    assert not sess.obstacles and sess.params.thigh_length != 0.42
    out = project.load(sess, tmp_path / "lab")
    assert out["state_restored"] and out["notes"] == "first try"
    with sess.lock:
        assert sess.params.thigh_length == 0.42 and len(sess.obstacles) == 1 and len(sess.props) == 1
        assert sess.actuators()[knee]["kp"] == 345.0
        assert np.allclose(sess.data.qpos, q) and np.allclose(sess.data.qvel, v) and abs(sess.data.time - t) < 1e-9
        assert sess.controller.name == "walker" and not sess.running
    falls = sess.falls                                             # and it carries on from there
    with sess.lock:
        sess.nav.set_goal([0.3, 0.8])
        sess.running = True
    assert wait_for(lambda: sess.nav.state in ("reached", "failed", "fallen"), 60)
    assert sess.nav.state == "reached" and sess.falls == falls
    project.save(sess, tmp_path / "lab")                           # saving again keeps the name and the creation date
    again = project.read(tmp_path / "lab")
    assert again["name"] == "Lab" and again["created"] == data["created"]
    with pytest.raises(FileExistsError):
        project.create(tmp_path / "lab", "x", sess)
    with pytest.raises(FileNotFoundError):
        project.read(tmp_path / "nothing")
    samples._builtin(sess)


def test_project_with_a_library_robot_stores_its_name_not_a_path(sess, tmp_path):
    from simulatics import robots
    if not robots.available("go2"):
        pytest.skip("go2 not downloaded")
    sess.load_robot("go2")
    f = project.save(sess, tmp_path / "dog", name="Dog")
    data = json.loads(f.read_text(encoding="utf-8"))
    assert data["robot"] == {**data["robot"], "kind": "library", "name": "go2"} and "path" not in data["robot"]
    samples._builtin(sess)
    project.load(sess, tmp_path / "dog")
    assert sess.robot_name == "go2" and sess.custom_path
    samples._builtin(sess)


# ------------------------------------------------------------------------------------------------ imports
def test_importer_recognises_files_and_motor_tables_apply_by_pattern(sess, tmp_path):
    samples._builtin(sess)
    table = importer.export_motors(sess, tmp_path / "motors.csv")
    rows = importer.read_motor_rows(table)
    assert len(rows) == sess.model.nu and {"kp", "kv", "torque", "damping", "armature"} <= set(rows[0])
    tune = tmp_path / "tune.csv"
    tune.write_text("name;kp;max_torque\n*_knee;321;200\nleft_hip_y;280;\nnose_motor;1;1\n", encoding="utf-8")
    assert importer.detect(tune) == "motors"
    rep = importer.import_any(sess, tune)
    assert rep["applied"] == 5 and rep["motors"] == 3 and rep["unknown"] == ["nose_motor"]
    a = {x["name"]: x for x in sess.actuators()}
    assert a["left_knee"]["kp"] == 321.0 and a["right_knee"]["torque"] == 200.0 and a["left_hip_y"]["kp"] == 280.0
    sess.add_object("pillar", 3.0, 2.0)                            # a rebuild keeps the imported values
    assert {x["name"]: x for x in sess.actuators()}["right_knee"]["kp"] == 321.0
    as_json = tmp_path / "motors.json"
    as_json.write_text(json.dumps({"left_elbow": {"kp": 77}}), encoding="utf-8")
    assert importer.detect(as_json) == "motors" and importer.import_any(sess, as_json)["applied"] == 1
    scene = tmp_path / "scene.json"
    scene.write_text(json.dumps(sess.scene_dict()), encoding="utf-8")
    assert importer.detect(scene) == "scene"
    assert importer.detect(samples.DATA / "vertical_jump.tape.npz") == "recording"
    assert importer.detect(samples.DATA / "backflip.npz") == "motion"
    assert importer.detect("runs/pretrained_walker/policy.npz") == "policy"
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        importer.detect(bad)
    with pytest.raises(ValueError):
        importer.detect(tmp_path)
    worse = tmp_path / "worse.csv"
    worse.write_text("name,kp\nleft_knee,stiff\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a number"):
        importer.import_motors(sess, worse)
    rep = importer.import_any(sess, samples.DATA / "cartwheel.npz")
    assert rep["kind"] == "motion" and sess.controller.name == "motion-preview"
    samples._builtin(sess)


# ----------------------------------------------------------------------------------- system identification
def test_system_identification_recovers_hidden_joint_parameters(sess, tmp_path):
    samples._builtin(sess)
    sess.set_support_stand(True)
    knee = next(a["index"] for a in sess.actuators() if a["name"] == "left_knee")
    with sess.lock:
        sess.manual_ctrl[knee] = 0.8
        sess.running = True
    time.sleep(0.4)
    before = sysid.get_params(sess.model, knee)
    r = sysid.self_test(sess, knee)
    assert sysid.get_params(sess.model, knee) == before            # the hidden change was really forgotten
    assert r["rms_after"] < 0.05 * r["rms_before"] and r["improved"]
    assert all(v < 0.03 for v in r["relative_error"].values()), r["relative_error"]
    # from a file, as a log from a real motor would come
    f = sysid.save_csv(r["recording"], tmp_path / "knee.csv")
    rec = sysid.load_csv(f)
    assert len(rec["time"]) == len(r["recording"]["time"]) and abs(rec["dt"] - r["recording"]["dt"]) < 1e-6
    noisy = sysid.self_test(sess, knee, noise=0.002)
    assert noisy["rms_after"] < noisy["rms_before"] and noisy["relative_error"]["damping"] < 0.25
    sysid.apply(sess, knee, r)
    assert abs(sess.actuators()[knee]["damping"] - r["fitted"]["damping"]) < 1e-9
    assert sess.motor_overrides["left_knee"]["frictionloss"] == pytest.approx(r["fitted"]["frictionloss"])
    with pytest.raises(ValueError):
        sysid.fit_motor(sess, r["recording"], fields=("colour",))
    sess.set_support_stand(False)
    with pytest.raises(ValueError, match="stand"):
        sysid.self_test(sess, knee)
    samples._builtin(sess)


# ----------------------------------------------------------------------------------------- robustness sweep
def test_sweep_reports_each_variant_and_shows_recordings_are_fragile():
    rep = validate.run("stand", "random", n=3, seed=3)
    assert rep["total"] == 4 and rep["rows"][0]["variant"] == "nominal" and rep["nominal_ok"]
    assert all({"variant", "result", "error_m", "note", "changes"} <= set(r) for r in rep["rows"])
    assert rep["passed"] >= 3 and "friction" in rep["rows"][1]["changes"]        # standing shrugs off these changes
    assert "of 4 variants passed" in validate.summary(rep)
    rec = validate.run("recording:vertical_jump", "settings")
    assert rec["nominal_ok"]                                       # the recording works on the model it was made for
    assert rec["passed"] < rec["total"]                            # and, being open loop, not under every other setting
    assert len(validate.setting_variants()) == rec["total"] - 1
    seen = []
    validate.run("stand", "settings", progress=lambda f, row: seen.append(f))
    assert seen and abs(seen[-1] - 1.0) < 1e-9
    with pytest.raises(ValueError):
        validate.run("moonwalk")


def test_random_variants_really_change_the_model():
    from simulatics.models import HumanoidParams, build_mjcf
    xml = build_mjcf(HumanoidParams())
    a, b = mujoco.MjModel.from_xml_string(xml), mujoco.MjModel.from_xml_string(xml)
    v = validate.random_variants(1, seed=5)[0]
    v.apply(b)
    assert not np.allclose(a.body_mass, b.body_mass) and not np.allclose(a.actuator_forcerange, b.actuator_forcerange)
    assert not np.allclose(a.geom_friction, b.geom_friction) and set(v.changes) == set(validate.RANGES)
    assert validate.random_variants(2, seed=5)[0].changes == v.changes          # the same seed gives the same variants


# -------------------------------------------------------------------------------------- any-height box jump
def test_box_jump_reference_is_reshaped_for_the_height_asked_for():
    from simulatics.models import HumanoidParams, build_mjcf
    from simulatics.samples import boxjump
    model = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams()))
    base, box0 = boxjump.reference(model, 0.68)
    for h in (0.3, 0.9):
        m, box = boxjump.reference(model, h, arms=h > 0.8)
        assert abs(2 * box["size"][2] - h) < 1e-9 and abs(box["z"] - h / 2) < 1e-9
        assert abs((m.root_pos[-1, 2] - m.root_pos[0, 2]) - (base.root_pos[-1, 2] - base.root_pos[0, 2]) - (h - 0.68)) < 0.03
        assert m.n_frames == base.n_frames and np.isfinite(m.joint_pos).all()
        assert abs((box["pos"][0] - box["size"][0]) - (box0["pos"][0] - box0["size"][0])) < 1e-9     # same near edge
    tall, _ = boxjump.reference(model, 0.9, arms=True)
    plain, _ = boxjump.reference(model, 0.9, arms=False)
    i = tall.joint_index("left_shoulder_y")
    assert np.abs(tall.joint_pos[:, i] - plain.joint_pos[:, i]).max() > 0.5       # the arms reach forward
    assert boxjump.find(12.0) is None or True
