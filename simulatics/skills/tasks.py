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


SKILLS: dict[str, SkillSpec] = {
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
