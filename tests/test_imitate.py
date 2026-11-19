"""Tests for motion imitation (simulatics/imitate): motions, BVH, retargeting, the clip library, playback,
the tracking task / env, the ImitationTrainer and the ImitationController.

Every BVH file used here is written by the tests themselves (no third-party motion data). All tests use
at most 3 threads and run in seconds.
"""
import json

import mujoco
import numpy as np
import pytest

from simulatics.control.base import Controller
from simulatics.imitate import (CLIPS, LIBRARY_DIR, BatchFK, ImitationController, ImitationEnv, ImitationPolicy,
                                ImitationTrainer, KinematicPlayer, Motion, MotionPreviewController, RetargetConfig,
                                TrackConfig, evaluate_imitation, guess_joint_map, imitation_ppo_config,
                                is_imitation_policy, list_motions, load_motion, motion_to_bvh, parse_bvh,
                                pd_playback, retarget_bvh, robot_keypoints)
from simulatics.imitate.bvh import BvhError, ORDERS, euler_to_matrix, matrix_to_euler
from simulatics.imitate.motion import keypoint_positions, quat_to_mat, yaw_quat
from simulatics.imitate.retarget import classify_joint
from simulatics.models import HumanoidParams, build_mjcf
from simulatics.train.ppo import is_rl_policy


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams()))


# ---- BVH ----------------------------------------------------------------------------------------------
TINY_BVH = """HIERARCHY
ROOT Hips
{
\tOFFSET 0.0 0.0 0.0
\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
\tJOINT Chest
\t{
\t\tOFFSET 0 10 0
\t\tCHANNELS 3 Xrotation Yrotation Zrotation
\t\tEnd Site
\t\t{
\t\t\tOFFSET 0 5 0
\t\t}
\t}
\tJOINT LeftUpLeg
\t{
\t\tOFFSET 5 -2 0
\t\tCHANNELS 3 Yrotation Zrotation Xrotation
\t\tJOINT LeftLeg
\t\t{
\t\t\tOFFSET 0 -20 0
\t\t\tCHANNELS 1 Xrotation
\t\t\tEnd Site
\t\t\t{
\t\t\t\tOFFSET 0 -20 0
\t\t\t}
\t\t}
\t}
}
MOTION
Frames: 3
Frame Time: 0.04
1 2 3 90 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 30 90
0.5 1.5 2.5 10 20 30 40 50 60 70 80 90 -45
"""


def test_bvh_parse_round_trip():
    b = parse_bvh(TINY_BVH)
    assert b.names == ["Hips", "Chest", "LeftUpLeg", "LeftLeg"]
    assert [j.parent for j in b.joints] == [-1, 0, 0, 2]
    assert b.joints[0].channels[:3] == ["Xposition", "Yposition", "Zposition"]
    assert b.joints[2].rot_order == "YZX" and b.joints[3].channels == ["Xrotation"]
    assert b.n_frames == 3 and abs(b.fps - 25.0) < 1e-9
    np.testing.assert_allclose(b.joints[3].end_site, [0, -20, 0])
    pos, rot, end = b.forward_kinematics()
    # frame 0: root at (1, 2, 3) turned 90 deg about Z: the chest offset (0, 10, 0) points along -X
    np.testing.assert_allclose(pos[0, 1], [1 - 10, 2, 3], atol=1e-9)
    np.testing.assert_allclose(end[0, 1], [1 - 15, 2, 3], atol=1e-9)
    # frame 1: hip rotated 30 deg about X, knee 90 deg about X: knee at hip + Rx(30)(0,-20,0)
    hip = np.array([5.0, -2.0, 0.0])
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    knee = hip + np.array([0.0, -20 * c, -20 * s])
    np.testing.assert_allclose(pos[1, 3], knee, atol=1e-9)
    shin = np.array([0.0, -20 * np.cos(np.radians(120)), -20 * np.sin(np.radians(120))])
    np.testing.assert_allclose(end[1, 3], knee + shin, atol=1e-9)
    # text round trip: identical hierarchy, frames and kinematics
    b2 = parse_bvh(b.to_text())
    assert b2.names == b.names and [j.channels for j in b2.joints] == [j.channels for j in b.joints]
    np.testing.assert_allclose(b2.frames, b.frames)
    np.testing.assert_allclose([j.offset for j in b2.joints], [j.offset for j in b.joints])
    np.testing.assert_allclose(b2.forward_kinematics()[0], pos, atol=1e-9)
    # CRLF line ends and tabs parse the same
    b3 = parse_bvh(TINY_BVH.replace("\n", "\r\n"))
    np.testing.assert_allclose(b3.frames, b.frames)


def test_bvh_errors():
    with pytest.raises(BvhError):
        parse_bvh("HIERARCHY\nROOT A\n{\nOFFSET 0 0\n}\nMOTION\nFrames: 0\nFrame Time: 0.1\n")
    with pytest.raises(BvhError):
        parse_bvh(TINY_BVH.replace("CHANNELS 1 Xrotation", "CHANNELS 1 Wrotation"))
    with pytest.raises(BvhError):   # fewer values than the header promises
        parse_bvh(TINY_BVH.replace("Frames: 3", "Frames: 4"))


@pytest.mark.parametrize("order", ORDERS)
def test_euler_round_trip(order):
    rng = np.random.default_rng(1)
    ang = rng.uniform(-85, 85, (200, 3))
    R = euler_to_matrix(ang, order)
    np.testing.assert_allclose(matrix_to_euler(R, order), ang, atol=1e-8)
    gimbal = euler_to_matrix(np.array([[30.0, 90.0, 0.0], [10.0, -90.0, 0.0]]), order)
    np.testing.assert_allclose(euler_to_matrix(matrix_to_euler(gimbal, order), order), gimbal, atol=1e-8)


def test_joint_name_matching():
    cases = {"LeftUpLeg": ("left", "hip"), "mixamorig:RightForeArm": ("right", "elbow"), "l_knee": ("left", "knee"),
             "Hip_L": ("left", "hip"), "lfemur": ("left", "hip"), "rtibia": ("right", "knee"),
             "LeftArm": ("left", "shoulder"), "RightHandIndex1": ("right", "finger"), "LeftToeBase": ("left", "toe"),
             "Hips": (None, "pelvis"), "Spine1": (None, "spine"), "lowerback": (None, "spine"),
             "HeadTop_End": (None, None), "LeftForeArmRoll": (None, None)}
    for name, (side, part) in cases.items():
        s, p, _ = classify_joint(name)
        assert p == part and (part is None or s == side), (name, s, p)
    # "LeftArm" (glenohumeral) beats "LeftShoulder" (clavicle) as the shoulder role
    b = parse_bvh(human_bvh_text(4))
    jm = {r: b.joints[j].name for r, j in guess_joint_map(b).items()}
    assert jm["left_shoulder"] == "LeftArm" and jm["right_hand"] == "RightHand" and jm["left_knee"] == "LeftLeg"
    assert jm["pelvis"] == "Hips" and jm["head"] == "Head"


# ---- Motion -------------------------------------------------------------------------------------------
def _toy_motion(loop: bool) -> Motion:
    fps, n = 50.0, 40
    t = np.arange(n) / fps
    root = np.stack([0.1 * t, np.zeros(n), np.full(n, 1.0)], 1)
    quat = yaw_quat(0.2 * t)
    joints = np.stack([np.sin(2 * np.pi * t / 0.8), 0.3 * t], 1)
    keys = np.stack([root + [0.0, 0.1, 0.0], root - [0.0, 0.1, 0.0]], 1)
    return Motion(fps, root, quat, ["a", "b"], joints, ["k1", "k2"], keys, loop=loop,
                  cycle=[0.08, 0.0, 0.16] if loop else [0, 0, 0], name="toy", meta={"source": "test"})


def test_motion_save_load_interpolation_loop(tmp_path):
    m = _toy_motion(loop=False)
    p = m.save(tmp_path / "toy.npz")
    m2 = Motion.load(p)
    assert m2.name == "toy" and m2.joint_names == ["a", "b"] and m2.key_names == ["k1", "k2"] and not m2.loop
    np.testing.assert_allclose(m2.joint_pos, m.joint_pos, atol=1e-5)   # stored as float32
    np.testing.assert_allclose(m2.root_pos, m.root_pos, atol=1e-5)
    np.testing.assert_allclose(m2.key_pos, m.key_pos, atol=1e-5)
    assert m2.meta["source"] == "test" and abs(m2.duration - 39 / 50) < 1e-12
    # linear interpolation half way between frames; clamping past the end
    s = m.sample(np.array([0.01, 10.0]))
    np.testing.assert_allclose(s["joint_pos"][0], (m.joint_pos[0] + m.joint_pos[1]) / 2)
    np.testing.assert_allclose(s["joint_pos"][1], m.joint_pos[-1])
    np.testing.assert_allclose(m.joints["b"], m.joint_pos[:, 1])
    # finite-difference velocities: joint b = 0.3 t, root yaw rate 0.2, root speed 0.1 along the heading
    v = m.velocities()
    np.testing.assert_allclose(v["joint"][:, 1], 0.3, atol=1e-9)
    np.testing.assert_allclose(v["root_ang"][:, 2], 0.2, atol=1e-6)
    np.testing.assert_allclose(v["root_lin"][:, 0], 0.1, atol=1e-9)
    # looping: frame N is frame 0 moved by the cycle transform
    L = _toy_motion(loop=True)
    f = L.frames(np.array([0, L.n_frames, 2 * L.n_frames]))
    np.testing.assert_allclose(f["root_pos"][1], L.root_pos[0] + [0.08, 0.0, 0.0], atol=1e-12)
    yaw = np.arctan2(quat_to_mat(f["root_quat"])[:, 1, 0], quat_to_mat(f["root_quat"])[:, 0, 0])
    np.testing.assert_allclose(yaw, [0.0, 0.16, 0.32], atol=1e-12)
    c, s_ = np.cos(0.16), np.sin(0.16)
    np.testing.assert_allclose(f["root_pos"][2, :2], [0.08 + 0.08 * c, 0.08 * s_], atol=1e-12)
    assert abs(L.duration - 40 / 50) < 1e-12
    r = L.resample(25.0)
    assert r.n_frames == 20 and abs(r.duration - L.duration) < 1e-12
    a = L.aligned(1.0, 2.0, np.pi / 2)
    np.testing.assert_allclose(a.root_pos[0, :2], [1.0, 2.0], atol=1e-12)


def test_library_clips(model):
    found = {m["name"]: m for m in list_motions()}
    assert set(CLIPS) <= set(found)
    total = sum((LIBRARY_DIR / f"{n}.npz").stat().st_size for n in CLIPS)
    assert total < 500 * 1024, total
    walk = load_motion("walk")
    assert walk.loop and walk.cycle[0] > 0.1 and walk.key_pos is not None    # it walks forward
    for name in CLIPS:
        m = load_motion(name)
        assert m.n_frames >= 2 and abs(m.fps - 50.0) < 0.5
        cols, adr = m.joint_qpos_map(model)
        assert len(cols) == len(m.joint_names)
        # the loop closes: frame N (next cycle) is close to the last stored frame
        f = m.frames(np.array([m.n_frames - 1, m.n_frames]))
        assert np.abs(f["joint_pos"][1] - f["joint_pos"][0]).max() < 0.15, name
        assert np.linalg.norm(f["root_pos"][1] - f["root_pos"][0]) < 0.05, name


def test_batch_fk_matches_mujoco(model):
    fk = BatchFK(model)
    walk = load_motion("walk")
    qs = walk.qpos(model)[::9]
    rng = np.random.default_rng(0)
    qs[:, 7:] += rng.uniform(-0.3, 0.3, qs[:, 7:].shape)
    keys, com = fk(qs)
    d = mujoco.MjData(model)
    kp = robot_keypoints(model)
    for i, q in enumerate(qs):
        d.qpos[:] = q
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        np.testing.assert_allclose(keys[i], keypoint_positions(model, d, kp), atol=1e-10)
        np.testing.assert_allclose(com[i], d.subtree_com[model.body("torso").id], atol=1e-10)


# ---- retargeting ---------------------------------------------------------------------------------------
def _procedural_robot_motion(model, n=60, fps=50.0) -> Motion:
    """A walking-like robot motion: legs and arms swing, the root moves forward and turns, feet slide."""
    t = np.arange(n) / fps
    q = np.tile(model.qpos0, (n, 1))
    s = np.sin(2 * np.pi * t / 1.2)

    def setj(name, v):
        q[:, model.jnt_qposadr[model.joint(name).id]] = v
    for side, sg in (("left", 1), ("right", -1)):
        setj(f"{side}_hip_y", -0.3 - 0.25 * sg * s)
        setj(f"{side}_knee", 0.5 + 0.2 * sg * s)
        setj(f"{side}_ankle_y", -0.2)
        setj(f"{side}_shoulder_y", -0.4 + 0.3 * sg * s)
        setj(f"{side}_shoulder_x", sg * 0.3)
        setj(f"{side}_elbow", -0.6 - 0.2 * s)
    setj("abdomen_y", 0.1 * s)
    q[:, 0], q[:, 2] = 0.3 * t, 1.1
    q[:, 3:7] = yaw_quat(0.2 * t)
    return Motion.from_qpos(model, q, fps, name="procedural")


def test_retarget_walk_bvh_full_pipeline(model, tmp_path):
    """Acceptance: the shipped walk (recorded from WalkController) written as a BVH of the robot's skeleton
    (proportions = the robot's, scale 93, Y-up, ZYX channels) goes through the whole pipeline - IK, ground
    alignment and foot-skate clean-up - and every key body ends within 3 cm of its target in every frame."""
    src = load_motion("walk")
    path = tmp_path / "walk.bvh"
    motion_to_bvh(src, model, scale=93.0, up="y", order="ZYX").save(path)
    res = retarget_bvh(path, model)
    assert res.errors["max"] < 0.03 and res.errors["mean"] < 0.005
    assert abs(res.report["min_sole_height_m"]) < 0.005
    assert res.stance.mean() > 0.5                    # the stance feet were found and locked


def test_retarget_synthetic_bvh(model, tmp_path):
    """A BVH of the robot's own skeleton, written in centimetres x 0.87 (scale 87), Y-up, YXZ channel order:
    retargeting must undo the unit, axis and Euler conventions and reach the key bodies within 3 cm. The
    procedural motion slides its feet on purpose, so the clean-up moves them (mean error stays < 3 cm)."""
    src = _procedural_robot_motion(model)
    path = tmp_path / "synthetic.bvh"
    motion_to_bvh(src, model, scale=87.0, up="y", order="YXZ").save(path)
    res = retarget_bvh(path, model, RetargetConfig(fps=50.0))
    assert res.report["up_axis"] == "y"
    assert abs(res.scale["leg"] - 1 / 87.0) < 1e-6
    # the IK alone (before foot clean-up) reproduces the key bodies; clean-up then pins sliding stance feet
    assert res.errors_before_cleanup["max"] < 0.03
    assert res.errors["mean"] < 0.03
    assert res.report["skate_speed_after_mps"] < 0.5 * res.report["skate_speed_before_mps"]
    assert res.report["min_sole_height_m"] > -0.01
    m = res.motion
    assert m.n_frames == src.n_frames and np.isfinite(m.joint_pos).all()
    # joint angles come back (knee and elbow: the free root makes the rest unique up to the foot lock)
    i, j = m.joint_index("right_elbow"), src.joint_index("right_elbow")
    np.testing.assert_allclose(m.joint_pos[:, i], src.joint_pos[:, j], atol=0.02)


def human_bvh_text(frames: int = 30) -> str:
    """A small human-proportioned skeleton (CMU-style names, Y-up, centimetres, T-pose rest) doing a
    knee-bending step in place with the arms lowered, written here from scratch."""
    sk = [("Hips", None, (0, 0, 0)), ("LeftUpLeg", "Hips", (9, -6, 0)), ("LeftLeg", "LeftUpLeg", (0, -42, 0)),
          ("LeftFoot", "LeftLeg", (0, -41, 0)), ("LeftToeBase", "LeftFoot", (0, -6, 13)),
          ("RightUpLeg", "Hips", (-9, -6, 0)), ("RightLeg", "RightUpLeg", (0, -42, 0)),
          ("RightFoot", "RightLeg", (0, -41, 0)), ("RightToeBase", "RightFoot", (0, -6, 13)),
          ("Spine", "Hips", (0, 10, 0)), ("Spine1", "Spine", (0, 18, 0)), ("Neck", "Spine1", (0, 22, 0)),
          ("Head", "Neck", (0, 10, 2)), ("LeftShoulder", "Spine1", (3, 19, 0)), ("LeftArm", "LeftShoulder", (15, 0, 0)),
          ("LeftForeArm", "LeftArm", (28, 0, 0)), ("LeftHand", "LeftForeArm", (25, 0, 0)),
          ("RightShoulder", "Spine1", (-3, 19, 0)), ("RightArm", "RightShoulder", (-15, 0, 0)),
          ("RightForeArm", "RightArm", (-28, 0, 0)), ("RightHand", "RightForeArm", (-25, 0, 0))]
    kids = {n: [c for c, p, _ in sk if p == n] for n, _, _ in sk}
    off = {n: o for n, _, o in sk}
    lines = ["HIERARCHY"]

    def write(n, depth):
        ind = "  " * depth
        lines.append(f"{ind}{'ROOT' if depth == 0 else 'JOINT'} {n}")
        lines.append(f"{ind}{{")
        lines.append(f"{ind}  OFFSET {' '.join(str(v) for v in off[n])}")
        ch = "6 Xposition Yposition Zposition" if depth == 0 else "3"
        lines.append(f"{ind}  CHANNELS {ch} Zrotation Xrotation Yrotation")
        for c in kids[n]:
            write(c, depth + 1)
        if not kids[n]:
            lines.extend([f"{ind}  End Site", f"{ind}  {{", f"{ind}    OFFSET 0 5 0", f"{ind}  }}"])
        lines.append(f"{ind}}}")
    write("Hips", 0)
    order = []

    def walk(n):
        order.append(n)
        for c in kids[n]:
            walk(c)
    walk("Hips")
    lines += ["MOTION", f"Frames: {frames}", "Frame Time: 0.0333333"]
    for k in range(frames):
        t = k / 30.0
        s = np.sin(2 * np.pi * t / 1.6)
        v = {n: [0.0, 0.0, 0.0] for n in order}    # (Z, X, Y) degrees
        bend = 20 + 15 * s
        v["LeftUpLeg"][1], v["RightUpLeg"][1] = -bend / 2 - 5 * s, -bend / 2 + 5 * s
        v["LeftLeg"][1] = v["RightLeg"][1] = bend
        v["LeftFoot"][1] = v["RightFoot"][1] = -bend / 2
        v["LeftArm"][0], v["RightArm"][0] = -75.0, 75.0
        v["Spine1"][1] = 5 * s
        row = [0.0, 92 - 8 * (1 - np.cos(np.radians(bend / 2))), 10 * t]
        for n in order:
            row += v[n]
        lines.append(" ".join(f"{x:.4f}" for x in row))
    return "\n".join(lines) + "\n"


def test_retarget_human_proportions(model, tmp_path):
    """Different proportions than the robot (legs 83 cm, arms 53 cm, narrower hips): the scaled targets are
    not all reachable, but the solve stays close, keeps the feet on the floor and lowers the arms."""
    path = tmp_path / "human.bvh"
    path.write_text(human_bvh_text(45))
    res = retarget_bvh(path, model)
    assert res.report["up_axis"] == "y"
    assert res.errors["mean"] < 0.03 and res.errors["max"] < 0.08
    assert res.report["min_sole_height_m"] > -0.01
    m = res.motion
    abd = m.joint_pos[:, m.joint_index("left_shoulder_x")]
    np.testing.assert_allclose(abd, np.radians(15), atol=0.05)        # 75 deg down from the T-pose
    knee = m.joint_pos[:, m.joint_index("left_knee")]
    assert knee.max() > 0.4 and knee.min() >= 0.0


# ---- playback ------------------------------------------------------------------------------------------
def test_kinematic_playback(model):
    squat = load_motion("squat")
    player = KinematicPlayer(model, squat)
    d = mujoco.MjData(model)
    kp = robot_keypoints(model)
    names = squat.key_names
    for k in (0, 17, 80):
        player.apply(d, k / squat.fps)
        np.testing.assert_allclose(d.qpos, squat.qpos(model, np.array([k]))[0], atol=1e-9)
        pos = keypoint_positions(model, d, {n: kp[n] for n in names})
        np.testing.assert_allclose(pos, squat.key_pos[k], atol=1e-4)
    # anchored where the robot stands: the clip starts at that heading frame
    d.qpos[:] = model.qpos0
    d.qpos[0:2] = (2.0, -1.0)
    d.qpos[3:7] = yaw_quat(np.pi / 2)
    ctl = MotionPreviewController(model, squat)
    assert isinstance(ctl, Controller)
    ctl.reset(model, d)
    np.testing.assert_allclose(d.qpos[0:2], [2.0, -1.0], atol=0.05)
    R = d.xmat[model.body("torso").id].reshape(3, 3)
    assert abs(np.arctan2(R[1, 0], R[0, 0]) - np.pi / 2) < 0.02
    for _ in range(10):
        ctl.step(model, d)
    assert ctl.status()["motion"] == "squat"


def test_pd_playback_squat_stays_up(model):
    r = pd_playback(model, load_motion("squat"), seconds=2.0)
    assert r["mode"] == "full" and not r["fell"]
    assert r["joint_rmse_deg"] < 3.0 and r["root_err_m"] < 0.03


# ---- tracking task and env ------------------------------------------------------------------------------
def test_tracking_reward_is_one_on_reference(model):
    env = ImitationEnv(model, load_motion("walk"), TrackConfig(), num_envs=6, nthread=2, seed=0)
    with env:
        obs, cobs = env.reset()
        assert obs.shape == (6, env.obs_dim) and cobs.shape == (6, env.critic_obs_dim)
        assert np.isfinite(obs).all() and np.isfinite(cobs).all()
        for _ in range(5):
            # follow the reference exactly: step, then put every env on the reference of its new frame
            env.step(np.zeros((6, env.act_dim)))
            env.set_reference_state()
            qpos, qvel = env.task.split(env.state)
            r, terms, terminated, err = env.task.reward(qpos, qvel, env.ts)
            np.testing.assert_allclose(r, 1.0, atol=1e-9)
            assert not terminated.any() and err["joint_rmse"].max() < 1e-9
        assert abs(sum(env.task.weights.values()) - 1.0) < 1e-12
        # off the reference the reward drops and large deviations terminate
        qpos2 = qpos.copy()
        qpos2[:, env.task.qadr] += 0.2
        r2, *_ = env.task.reward(qpos2, qvel, env.ts)
        assert (r2 < 0.9).all()
        qpos3 = qpos.copy()
        qpos3[:, 2] -= 0.5
        _, _, term3, _ = env.task.reward(qpos3, qvel, env.ts)
        assert term3.all()


def test_env_runs_and_feedforward_baseline(model):
    """The feed-forward targets alone hold the quasi-static squat for 2 s (deterministic baseline)."""
    r = evaluate_imitation(None, model, seconds=2.0, num_envs=4, nthread=2, motion=load_motion("squat"))
    assert r["zero_action"] and r["fell_frac"] == 0.0 and r["reward_per_step"] > 0.95


# ---- trainer and controller ------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    cfg = imitation_ppo_config(name="smoke", num_envs=8, steps_per_env=16, iterations=3, threads=2, blas_threads=2,
                               actor_hidden=[32, 32], critic_hidden=[32, 32], save_interval=1)
    tr = ImitationTrainer(cfg, "squat", TrackConfig(episode_seconds=0.2), runs_dir=runs)
    assert tr.status()["state"] == "idle"
    tr.start()
    tr.join()
    return tr


def test_imitation_trainer_smoke(trained):
    st = trained.status()
    assert st["state"] == "finished", st["error"]
    assert st["generation"] == 3 and st["generations"] == 3 and st["algo"] == "imitation"
    for key in ("gen", "best", "mean", "center", "sps", "reward_per_step", "joint_err_deg", "ep_len_s"):
        assert key in st["history"][-1]
    json.dumps(st)                                      # the GUI sends this as JSON
    assert st["history"][-1]["ep_len_s"] > 0            # episodes finished and were counted
    path = st["policy"]
    assert path and path.endswith("imitation_policy.npz")
    assert is_imitation_policy(path) and not is_rl_policy(path)
    pol = ImitationPolicy.load(path)
    assert pol.motion.name == "squat" and pol.act_dim == 20
    assert (trained.run_dir / "imitation_checkpoint.npz").exists() and (trained.run_dir / "history.json").exists()
    # stop() before start finishes quickly as "stopped"
    tr = ImitationTrainer(imitation_ppo_config(name="stop", num_envs=4, steps_per_env=4, iterations=1000, threads=1,
                                               blas_threads=1, actor_hidden=[8], critic_hidden=[8]),
                          "stand", runs_dir=trained.run_dir.parent)
    tr.stop()
    tr.start()
    tr.join()
    assert tr.status()["state"] == "stopped"


def test_imitation_controller_protocol(model, trained):
    ctl = ImitationController(model, trained.status()["policy"])
    assert isinstance(ctl, Controller) and ctl.name == "imitation"
    assert abs(ctl.control_dt - 4 * model.opt.timestep) < 1e-12
    d = mujoco.MjData(model)
    d.qpos[0:2] = (1.0, 0.5)
    mujoco.mj_forward(model, d)
    ctl.reset(model, d)
    np.testing.assert_allclose(d.qpos[0:2], [1.0, 0.5], atol=0.05)   # the clip starts where the robot stands
    ctl.set_path(np.array([[3.0, 0.0]]), 0.0)                          # ignored, must not raise
    for _ in range(25):
        ctl.step(model, d)
        assert np.isfinite(d.ctrl).all()
        mujoco.mj_step(model, d, 4)
    st = ctl.status()
    assert st["mode"] in ("walk", "stand", "done", "fallen") and isinstance(st["message"], str)
    assert st["motion"] == "squat" and st["joint_err_deg"] is not None
    ev = evaluate_imitation(trained.status()["policy"], model, seconds=0.5, num_envs=2, nthread=2)
    assert 0.0 <= ev["fell_frac"] <= 1.0 and np.isfinite(ev["reward_per_step"])
