"""Motion imitation for the Simulatics humanoid.

Reference motions (`Motion`, motion.py) come from the project's own controllers and kinematics
(library.py) or from user BVH files retargeted GMR-style (bvh.py, retarget.py). They can be previewed
kinematically or played through the servos with balance feedback (playback.py), and tracked by a
DeepMimic-style PPO policy (track_task.py, track_env.py, train.py) that `ImitationController` runs in
the live session.

Public API (lazy imports keep `import simulatics.imitate` cheap):

    Motion, load_motion, list_motions, build_library, LIBRARY_DIR
    Bvh, parse_bvh, load_bvh, motion_to_bvh
    retarget_bvh, RetargetConfig, RetargetResult, guess_joint_map, WholeBodyIK
    KinematicPlayer, MotionPreviewController, PDPlaybackController, pd_playback
    MotionReference, TrackConfig, TrackTask, ImitationEnv
    ImitationTrainer, imitation_ppo_config, ImitationPolicy, is_imitation_policy,
    ImitationController, evaluate_imitation
"""
from __future__ import annotations

import importlib

_EXPORTS = {
    "Motion": "motion", "is_motion_file": "motion", "robot_keypoints": "motion",
    "load_motion": "library", "list_motions": "library", "build_library": "library", "LIBRARY_DIR": "library",
    "library_path": "library", "CLIPS": "library",
    "Bvh": "bvh", "BvhJoint": "bvh", "parse_bvh": "bvh", "load_bvh": "bvh", "motion_to_bvh": "bvh",
    "retarget_bvh": "retarget", "retarget_human": "retarget", "bvh_to_human": "retarget",
    "RetargetConfig": "retarget", "RetargetResult": "retarget", "guess_joint_map": "retarget",
    "classify_joint": "retarget", "WholeBodyIK": "retarget", "IKTask": "retarget",
    "KinematicPlayer": "playback", "MotionPreviewController": "playback", "PDPlaybackController": "playback",
    "pd_playback": "playback",
    "MotionReference": "reference", "BatchFK": "reference",
    "TrackConfig": "track_task", "TrackTask": "track_task", "ImitationEnv": "track_env",
    "ImitationTrainer": "train", "imitation_ppo_config": "train", "ImitationPolicy": "train",
    "is_imitation_policy": "train", "ImitationController": "train", "evaluate_imitation": "train",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module 'simulatics.imitate' has no attribute {name!r}")
    return getattr(importlib.import_module(f".{mod}", __name__), name)
