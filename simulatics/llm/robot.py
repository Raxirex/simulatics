"""The contract between an LLM pilot and the simulated robot.

The live GUI session implements `RobotInterface` (server/session.py); tests use
a fake. Everything an LLM may sense or do goes through this interface, so the
same pilot code can later drive a different simulator or a real robot.

Units: metres, degrees, simulation seconds. World frame: z up, x forward at spawn.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RobotInterface(Protocol):
    # ---- proprioception / state ----------------------------------------
    def pose(self) -> dict:
        """{"x", "y", "yaw_deg", "height", "speed", "upright": bool, "time"} of the robot base."""

    def nav_status(self) -> dict:
        """{"state": "idle"|"walking"|"reached"|"failed"|"fallen", "goal": [x, y] | None,
        "goal_yaw_deg": float | None, "error_m": float | None, "message": str}"""

    def sim_time(self) -> float:
        """Current simulation time in seconds."""

    # ---- exteroception ---------------------------------------------------
    def camera_jpeg(self, camera: str = "head_cam", width: int = 480, height: int = 360) -> bytes:
        """A JPEG frame rendered from a robot camera."""

    def cameras(self) -> list[str]:
        """Names of the cameras the robot has."""

    def lidar_summary(self, sectors: int = 12) -> list[dict]:
        """Nearest lidar return per bearing sector, robot-relative:
        [{"bearing_deg": centre (0 = straight ahead, +left), "nearest_m": float | None}]"""

    def objects(self) -> list[dict]:
        """Named world objects the robot knows about:
        [{"name", "colour", "shape", "x", "y", "size_m", "distance_m", "bearing_deg", "visible": bool}]
        `visible` is True when the object is in the head camera's view (segmentation render)."""

    # ---- actions ---------------------------------------------------------
    def walk_to(self, x: float, y: float, yaw_deg: float | None = None) -> str:
        """Start walking to a world point (path planned around obstacles). Returns a short message."""

    def walk_relative(self, forward_m: float, left_m: float = 0.0, turn_deg: float = 0.0) -> str:
        """Start walking to a point given in the robot frame, optionally ending turned by turn_deg."""

    def stop(self) -> str:
        """Stop walking and stand still."""

    def wait(self, seconds: float) -> None:
        """Block until `seconds` of simulation time pass, or the current walk / arm motion ends
        (whichever is first). Must return promptly if the simulation is paused or the pilot is cancelled."""

    # ---- arms and hands ------------------------------------------------
    def reach(self, side: str, x: float, y: float, z: float, approach: list[float] | None = None) -> str:
        """Move the "left" or "right" hand's grasp point (between the finger pads) to a world point.
        approach: optional unit direction of the final motion, e.g. [0, 0, -1] = come down from above
        (opens the gripper first; use it to pick objects off a table). Returns a short message."""

    def grasp(self, side: str) -> str:
        """Close that hand's gripper on whatever is between the fingers."""

    def release(self, side: str) -> str:
        """Open that hand's gripper."""

    def point_at(self, side: str, x: float, y: float, z: float) -> str:
        """Point that arm at a world point."""

    def arm_gesture(self, side: str, gesture: str) -> str:
        """Named motion: "home" (arms down), "wave", "raise"."""

    def arm_status(self) -> dict:
        """{"left": {"skill", "state": "idle"|"moving"|"holding"|"done"|"failed",
                     "hand_xyz": [x, y, z], "error_m": float | None, "holding": object name | None},
            "right": {...}}"""
