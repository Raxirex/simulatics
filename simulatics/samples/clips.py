"""Which motion-capture takes become which reference samples.

All from the CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu), in the BVH conversion by Bruce Hahne
(cgspeed.com, 2010) as mirrored at github.com/una-dinosauria/cmu-mocap. Frame numbers are BVH frame indices
(frame 0 of every file is a T-pose added by the converter and is never used). Subjects 87-89 were captured at
60 fps although the files say 120: `fps` corrects that.
"""
from __future__ import annotations

URL = "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/{s:03d}/{s:02d}_{t:02d}.bvh"

# id -> spec (see build.py)
CLIPS: dict[str, dict] = {}


def clip(id: str, title: str, group: str, description: str, subject: int, trial: int,
         frames: tuple[int, int] | None = None, fps: float = 120.0, props: str | None = None, speed: float = 1.0,
         reverse: bool = False, ground_all: bool = False, **retarget) -> None:
    CLIPS[id] = {"title": title, "group": group, "description": description, "subject": subject, "trial": trial,
                 "url": URL.format(s=subject, t=trial), "frames": frames, "fps": fps, "props": props, "speed": speed,
                 "reverse": reverse, "ground_all": ground_all, "retarget": retarget}


A, P = "Acrobatics", "Parkour"
clip("backflip", "Backflip", A, "A standing back somersault: jump, tuck, a full turn backwards, land on both feet.",
     88, 1, fps=60)
clip("front_flip", "Front flip (reversed backflip)", A,
     "A forward somersault. No clean front flip exists in the source database, so this is the backflip take played "
     "backwards: time-reversed flight is still a valid ballistic flight.", 88, 1, fps=60, reverse=True)
clip("front_handspring", "Front handspring", A,
     "A running forward flip over the hands: the hands touch down briefly, then the feet land.", 90, 11, ground_all=True)
clip("one_hand_stand", "Stand on one hand", A,
     "Kicks up and holds the whole body on one planted hand for about a second and a half, then comes back to "
     "standing.", 89, 3, fps=60, ground_all=True, weights={"hand": 8.0, "pelvis": 1.0})
clip("handstand", "Handstand", A,
     "A handstand with leg kicks. The robot has shorter arms than the performer, so its head comes down to the floor.", 85, 5, frames=(400, 900), ground_all=True, weights={"hand": 8.0, "pelvis": 1.0})
clip("cartwheel", "Cartwheel", A, "One cartwheel.", 88, 7, fps=60, ground_all=True)
clip("box_jump_up", "Jump onto a box", P, "A two-foot standing jump onto a platform about knee to hip high.",
     88, 2, frames=(96, 200), fps=60, props="boxes")
clip("box_jump_down", "Jump down from a box", P, "Steps off a ledge and lands in a deep squat.",
     82, 4, frames=(150, 420), props="boxes", ground_min=True)
clip("hurdle", "Jump a hurdle", P, "A running jump clearing an obstacle.", 127, 25, props="hurdle")
clip("vertical_jump", "Vertical jump", P, "A high jump on the spot.", 16, 3)
clip("broad_jump", "Broad jump", P, "A standing jump forwards.", 13, 11)
