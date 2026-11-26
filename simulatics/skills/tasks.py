"""The skills that come with Simulatics. Each is a scene plus a reward; training does the rest."""
from __future__ import annotations

import numpy as np

from . import Rollout, SkillSpec

RIGHT_ARM = ["right_shoulder_y", "right_shoulder_x", "right_elbow", "right_wrist_y"]
WAIST = ["abdomen_y", "abdomen_x"]
LEGS_SAGITTAL = ["left_hip_y", "left_knee", "left_ankle_y", "right_hip_y", "right_knee", "right_ankle_y"]

BALL_R = 0.07
TEE_TOP = 1.128
BALL_XY = (0.30, -0.18)                      # in front of the right shoulder


def _standing_terms(r: Rollout, min_up: float = 0.85) -> np.ndarray:
    """Shared by every standing skill: stay up, keep the feet planted, end at rest in the pose you started in
    (so the balance controller can take over), and do not thrash."""
    ok = r.stays_standing(min_up=min_up)
    com, pose = r.returns_to_start()
    return (2.0 * ok - 4.0 * (1.0 - ok) - 3.0 * np.minimum(r.feet_moved(), 1.0) - 3.0 * np.minimum(r.ends_at_rest(), 2.0)
            - 10.0 * np.minimum(com, 0.5) - 6.0 * np.minimum(pose, 0.5) - 6.0 * np.maximum(0.0, 0.995 - r.up[:, -1])
            - 0.15 * r.effort())


# ------------------------------------------------------------------------------------------------ uppercut
def uppercut_scene() -> tuple[list[dict], list[dict]]:
    """A ball on a tee at chest height in front of the right shoulder."""
    tee = {"name": "tee", "shape": "cylinder", "pos": [BALL_XY[0], BALL_XY[1]], "size": [0.03, 0.03, TEE_TOP / 2],
           "z": TEE_TOP / 2, "yaw": 0.0, "rgba": [0.62, 0.65, 0.70, 1.0]}
    ball = {"name": "ball", "shape": "sphere", "pos": [BALL_XY[0], BALL_XY[1], TEE_TOP + BALL_R + 0.001], "size": [BALL_R],
            "mass": 0.12, "rgba": [0.95, 0.35, 0.15, 1.0], "yaw": 0.0}
    # no base plate: anything low near the toes is read by the balance controller as a step to stand on
    return [tee], [ball]


def _uppercut_measure(r: Rollout) -> dict:
    ball, hand, hv = r.prop_pos(0), r.body_pos("right_hand"), r.body_vel("right_hand")
    dist = np.linalg.norm(hand - ball, axis=2)                    # (n, T)
    k = dist.argmin(axis=1)
    rows = np.arange(r.n)
    rise = ball[:, :, 2].max(axis=1) - ball[:, 0, 2]
    launch = r.prop_vel(0)[:, :, 2].max(axis=1)
    return {"closest": dist[rows, k], "k": k, "rise": rise, "launch": launch, "fist_up": hv[rows, k, 2],
            "fist_speed": np.linalg.norm(hv[rows, k], axis=1),
            "sideways": np.abs(ball[:, -1, 1] - ball[:, 0, 1])}


def uppercut_reward(r: Rollout) -> np.ndarray:
    """Launch the ball upward with a rising right fist, and still be standing where you started."""
    m = _uppercut_measure(r)
    reach = -3.0 * np.maximum(m["closest"] - 0.09, 0.0)           # until the fist finds the ball
    strike = 5.0 * np.clip(m["rise"], 0.0, 2.5) + 0.6 * np.clip(m["launch"], 0.0, 8.0) + 0.4 * np.clip(m["fist_up"], 0.0, 6.0)
    return reach + strike + _standing_terms(r)


def uppercut_report(r: Rollout) -> dict:
    m = _uppercut_measure(r)
    return {"ball_rise_m": m["rise"][0], "ball_launch_speed_mps": m["launch"][0], "fist_speed_at_contact_mps": m["fist_speed"][0],
            "fist_upward_speed_at_contact_mps": m["fist_up"][0], "contact_time_s": float(m["k"][0] * r.dt),
            "closest_fist_to_ball_m": m["closest"][0]}


# ---------------------------------------------------------------------------------------------------- kick
def kick_scene() -> tuple[list[dict], list[dict]]:
    ball = {"name": "football", "shape": "sphere", "pos": [0.27, -0.10, 0.111], "size": [0.11], "mass": 0.4,
            "rgba": [0.95, 0.95, 0.95, 1.0], "yaw": 0.0}
    return [], [ball]


def kick_reward(r: Rollout) -> np.ndarray:
    """Send the ball forward with the right foot and stay standing."""
    ball, foot = r.prop_pos(0), r.body_pos("right_foot")
    closest = np.linalg.norm(foot - ball, axis=2).min(axis=1)
    forward = ball[:, -1, 0] - ball[:, 0, 0]
    speed = r.prop_vel(0)[:, :, 0].max(axis=1)
    stand = _standing_terms(r, min_up=0.8)
    return -3.0 * np.maximum(closest - 0.14, 0.0) + 3.0 * np.clip(forward, -0.5, 4.0) + 0.5 * np.clip(speed, 0.0, 8.0) + stand


def kick_report(r: Rollout) -> dict:
    ball = r.prop_pos(0)
    return {"ball_travel_m": ball[0, -1, 0] - ball[0, 0, 0], "ball_top_speed_mps": r.prop_vel(0)[0, :, 0].max()}


# -------------------------------------------------------------------------------------------------- get up
GETUP_PARAMS = {"hip_torque": 300.0, "knee_torque": 360.0, "ankle_torque": 180.0, "abdomen_torque": 200.0, "arm_torque": 150.0}
GETUP_JOINTS = ["shoulder_y", "shoulder_x", "elbow", "hip_y", "knee", "ankle_y", "abdomen_y"]


def _lying(face_down: bool):
    def start(model, q_stand):
        from ..imitate.physical import settle_lying
        q = np.array(q_stand, float)
        hinge = [model.jnt_qposadr[j] for j in range(model.njnt) if model.jnt_type[j] == 3]
        q[hinge] = 0.0                                             # straight legs, arms along the body
        a = np.pi / 4 if face_down else -np.pi / 4                # half of +-90 degrees about the sideways axis
        q[3:7] = (np.cos(a), 0.0, np.sin(a), 0.0)
        q[2] = 0.3
        return settle_lying(model, q, seconds=1.5)
    return start


def _plan(env, frames: list[tuple[float, dict]]) -> np.ndarray:
    """(T, nu) servo targets through key poses: [(seconds, {joint without side: angle})], left and right alike.
    Before the first pose the robot holds its start pose; after the last one it holds the standing command."""
    m = env.model
    act = {m.joint(int(m.actuator_trnid[a, 0])).name: a for a in range(m.nu)}
    times, rows = [0.0], [env.ctrl0.copy()]
    for t, pose in frames:
        c = np.zeros(m.nu)
        for j, v in pose.items():
            for name, sign in ((j, 1.0), (f"left_{j}", 1.0), (f"right_{j}", -1.0 if j.endswith(("_x", "_z")) else 1.0)):
                if name in act:
                    c[act[name]] = sign * v
        times.append(float(t))
        rows.append(c)
    times.append(times[-1] + 1.2)
    rows.append(env.ctrl_stand.copy())
    t = np.arange(env.T) * env.dt
    rows = np.array(rows)
    return np.stack([np.interp(t, times, rows[:, a]) for a in range(m.nu)], axis=1)


SQUAT = {"hip_y": -1.92, "knee": 2.44, "ankle_y": -0.79, "abdomen_y": 0.0, "shoulder_y": -1.5}


def getup_back_plan(env) -> np.ndarray:
    """Rock up: legs over the head, throw them down, tuck the heels to the seat, fold over the feet into a deep
    squat (which balances with the arms forward), then stand."""
    return _plan(env, [
        (0.50, {"hip_y": -1.92, "knee": 0.3, "abdomen_y": 0.79, "shoulder_y": -2.0}),
        (0.85, {"hip_y": -0.6, "knee": 2.44, "ankle_y": -0.79, "abdomen_y": 0.0, "shoulder_y": -1.2}),
        (1.15, {"hip_y": -1.92, "knee": 2.44, "ankle_y": -0.79, "abdomen_y": 0.5, "shoulder_y": -1.5}),
        (2.00, SQUAT), (2.40, SQUAT)])


def getup_front_plan(env) -> np.ndarray:
    """Forward roll: this robot's hands only reach the floor beside its hips, so pressing them down lifts the
    hips and tips it over its head. Tuck late in the roll, land on the feet in a deep squat, stand."""
    tuck = {"hip_y": -1.92, "knee": 2.44, "ankle_y": -0.79, "abdomen_y": 0.79, "shoulder_y": -1.5, "elbow": 0.0}
    return _plan(env, [
        (0.50, {"shoulder_y": 0.2, "elbow": -2.27, "ankle_y": -0.79}),
        (1.20, {"shoulder_y": -3.0, "elbow": 1.5, "ankle_y": -0.79}),
        (2.30, {"shoulder_y": -3.9, "elbow": 1.0, "ankle_y": -0.79, "hip_y": -2.2, "abdomen_y": 0.72}),
        (2.60, tuck), (3.10, SQUAT), (3.50, SQUAT)])


def _getup_score(r: Rollout) -> np.ndarray:
    """(n, T) how far along a get-up each moment is. The parts add up, so there is something to climb from
    lying flat: torso upright, feet on the floor, feet under the centre of mass, then height."""
    z = np.clip(r.qpos[:, :, 2] / r.q_stand[2], 0.0, 1.05)
    up = r.up
    feet = 0.5 * (r.body_pos("left_foot") + r.body_pos("right_foot"))
    feet_low = np.clip(1.0 - (feet[:, :, 2] - 0.07) / 0.3, 0.0, 1.0)
    under = np.clip(1.0 - np.linalg.norm(r.com[:, :, :2] - feet[:, :, :2], axis=2) / 0.5, 0.0, 1.0) * feet_low
    return 0.5 * (up + 1.0) + 0.5 * feet_low + 1.5 * under + 3.0 * z * np.maximum(up, 0.0) * under


def getup_reward(r: Rollout) -> np.ndarray:
    """End up standing: as high and upright as the standing pose, feet under the body, at rest."""
    s = _getup_score(r)
    k = int(0.4 * r.T)
    z = r.qpos[:, :, 2] / r.q_stand[2]
    hinge = [r.model.jnt_qposadr[j] for j in range(r.model.njnt) if r.model.jnt_type[j] == 3 and "gripper" not in r.model.joint(j).name]
    pose = np.abs(r.qpos[:, -1][:, hinge] - r.q_stand[hinge]).mean(axis=1)
    last = slice(r.T - max(int(0.5 / r.dt), 1), r.T)                # up for the whole last half second, not in passing
    stood = ((z[:, last].min(axis=1) > 0.9) & (r.up[:, last].min(axis=1) > 0.93)).astype(float)
    return (1.0 * s.max(axis=1) + 1.0 * s[:, k:].mean(axis=1) + 1.5 * s[:, last].min(axis=1) + 4.0 * stood - 1.0 * np.minimum(r.ends_at_rest(), 2.0)
            - 1.0 * np.minimum(pose, 1.0) - 0.02 * r.effort())


def getup_report(r: Rollout) -> dict:
    return {"final_height_fraction": r.qpos[0, -1, 2] / r.q_stand[2], "final_upright": r.up[0, -1],
            "start_height_m": r.qpos[0, 0, 2], "time_to_stand_s": float(np.argmax(r.qpos[0, :, 2] > 0.9 * r.q_stand[2]) * r.dt)}


SKILLS: dict[str, SkillSpec] = {
    "getup_front": SkillSpec(
        "getup_front", "Get up (from lying face down)",
        "From lying face down on the floor, at rest, to standing, by a forward roll onto the feet. Left and right move "
        "alike. Uses the athletic motor preset with stronger arms.",
        joints=GETUP_JOINTS, reward=getup_reward, duration=4.4, settle=1.0, knots=18, sigma=0.3, params=GETUP_PARAMS,
        start=_lying(True), mirror=True, to_stand=True, baseline=getup_front_plan, report=getup_report, jitter_pose=0.02),
    "getup_back": SkillSpec(
        "getup_back", "Get up (from lying on the back)",
        "From lying on the back, at rest, to standing. Experimental: no working result yet (rocking up and propping "
        "up on the arms both failed).",
        joints=GETUP_JOINTS, reward=getup_reward, duration=4.0, settle=1.0, knots=16, sigma=0.35, params=GETUP_PARAMS,
        start=_lying(False), mirror=True, to_stand=True, baseline=getup_back_plan, report=getup_report, jitter_pose=0.02,
        trained=False),
    "uppercut": SkillSpec(
        "uppercut", "Uppercut",
        "A rising punch with the right fist that knocks a ball off a tee at chest height, keeping both feet planted "
        "and ending back in the standing pose.",
        joints=RIGHT_ARM + WAIST + LEGS_SAGITTAL, reward=uppercut_reward, duration=1.0, settle=0.6, knots=8,
        scene=uppercut_scene, report=uppercut_report),
    "kick": SkillSpec(
        "kick", "Kick a ball",
        "Kicks a football forward with the right foot from standing and puts the foot back down. Experimental: "
        "standing on one leg without balance feedback is hard for an open-loop program.",
        joints=["right_hip_y", "right_knee", "right_ankle_y", "left_hip_x", "left_ankle_x", "left_hip_y", "left_ankle_y",
                "right_hip_x"] + WAIST,
        reward=kick_reward, duration=1.2, settle=1.0, knots=8, scene=kick_scene, report=kick_report, trained=False),
}
