"""FakeRobot: a kinematic stand-in for the simulated humanoid, for tests and offline demos.

It implements `simulatics.llm.robot.RobotInterface` without MuJoCo:

- **Base**: unicycle kinematics, turn-then-walk (turn toward the goal, walk when the heading
  error is below 20 deg, finally turn to the requested yaw). Goals inside an object's footprint
  or outside the map fail like the real planner ("no path"); there is no path planning.
- **Arms**: point-mass hands moving at a fixed speed in the robot frame, with a spherical reach
  limit around each shoulder. `grasp` picks up the nearest graspable object within 0.10 m of the
  palm; a held object follows the hand. Gestures run for a fixed time.
- **Senses**: lidar is a ray/circle intersection against the object footprints; the camera is a
  pinhole projection of visible objects drawn with PIL (good enough for Claude to see colours and
  rough positions); `visible` means inside the 60 deg head-camera field of view.
- **Time** is simulated: `wait()` advances it in `dt` steps. With `realtime > 0` every step also
  sleeps `dt / realtime` wall seconds, so cancellation and pausing can be tested. `wait()` returns
  early when `cancelled` (a `threading.Event`) is set or `paused` is True, as the contract requires.
"""
from __future__ import annotations

import io
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ..sim.scene import PALETTE

ARM_SIDES = ("left", "right")


@dataclass
class FakeObject:
    name: str
    colour: str
    shape: str
    x: float
    y: float
    z: float            # centre height (m)
    size_m: float       # footprint diameter / side (m)
    graspable: bool = False


def default_objects() -> list[FakeObject]:
    """A small fixed scene: a red box ahead-left, a table with a blue cube, two more landmarks."""
    return [
        FakeObject("red box", "red", "box", 4.0, 1.0, 0.4, 0.8),
        FakeObject("white table", "white", "box", 2.5, -1.5, 0.375, 0.5),
        FakeObject("blue cube", "blue", "box", 2.5, -1.5, 0.78, 0.06, graspable=True),
        FakeObject("green cylinder", "green", "cylinder", -3.0, 2.0, 0.5, 0.6),
        FakeObject("yellow box", "yellow", "box", 0.5, -4.0, 0.3, 0.6),
    ]


@dataclass
class _Arm:
    side: str
    rest: np.ndarray                        # robot-frame rest position of the palm
    hand: np.ndarray = field(default=None)  # robot-frame palm position
    target: np.ndarray | None = None
    skill: str = "home"
    state: str = "idle"
    error_m: float | None = None
    holding: str | None = None
    timer: float = 0.0
    clipped: float = 0.0                    # how far the requested target was beyond reach

    def __post_init__(self):
        if self.hand is None:
            self.hand = self.rest.copy()


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class FakeRobot:
    """Deterministic, MuJoCo-free robot implementing RobotInterface (see module docstring)."""

    def __init__(self, objects: list[FakeObject] | None = None, *, x: float = 0.0, y: float = 0.0,
                 yaw_deg: float = 0.0, realtime: float = 0.0, dt: float = 0.05, walk_speed: float = 0.5,
                 turn_rate_deg: float = 120.0, world_half_size: float = 10.0):
        self.objects_ = list(default_objects() if objects is None else objects)
        self.x, self.y, self.yaw = float(x), float(y), float(yaw_deg)
        self.t = 0.0
        self.realtime = float(realtime)
        self.dt = float(dt)
        self.walk_speed = float(walk_speed)
        self.turn_rate = float(turn_rate_deg)
        self.world_half_size = float(world_half_size)
        self.body_radius = 0.25
        self.shoulder_height = 1.20
        self.shoulder_offset = 0.20
        self.arm_reach = 0.75
        self.arm_speed = 0.6
        self.hfov_deg = 60.0
        self.cancelled = threading.Event()
        self.paused = False
        self.log: list[tuple] = []            # every action call, for tests
        self._lock = threading.RLock()
        self._speed = 0.0
        self._goal: np.ndarray | None = None
        self._goal_yaw: float | None = None
        self._nav_state = "idle"
        self._nav_msg = ""
        self._nav_err: float | None = None
        self._last_goal: tuple | None = None
        self.arms = {s: _Arm(s, np.array([0.05, (1 if s == "left" else -1) * 0.22, 0.75])) for s in ARM_SIDES}

    # ------------------------------------------------------------------ frames
    def _to_world(self, p: np.ndarray) -> np.ndarray:
        c, s = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        return np.array([self.x + c * p[0] - s * p[1], self.y + s * p[0] + c * p[1], p[2]])

    def _to_local(self, p) -> np.ndarray:
        dx, dy = float(p[0]) - self.x, float(p[1]) - self.y
        c, s = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        return np.array([c * dx + s * dy, -s * dx + c * dy, float(p[2])])

    def _shoulder(self, side: str) -> np.ndarray:
        return np.array([0.0, (1 if side == "left" else -1) * self.shoulder_offset, self.shoulder_height])

    def obj(self, name: str) -> FakeObject:
        return next(o for o in self.objects_ if o.name == name)

    # ------------------------------------------------------------------ state
    def pose(self) -> dict:
        with self._lock:
            return {"x": round(self.x, 3), "y": round(self.y, 3), "yaw_deg": round(_wrap_deg(self.yaw), 1),
                    "height": 1.0, "speed": round(self._speed, 2), "upright": True, "time": round(self.t, 2)}

    def nav_status(self) -> dict:
        with self._lock:
            g = self._goal if self._goal is not None else (self._last_goal[0] if self._last_goal else None)
            gy = self._goal_yaw if self._goal is not None else (self._last_goal[1] if self._last_goal else None)
            return {"state": self._nav_state, "goal": None if g is None else [round(float(v), 3) for v in g],
                    "goal_yaw_deg": None if gy is None else round(gy, 1), "error_m": self._nav_err,
                    "message": self._nav_msg}

    def sim_time(self) -> float:
        return self.t

    # ------------------------------------------------------------------ senses
    def cameras(self) -> list[str]:
        return ["head_cam", "chase"]

    def _bearing(self, x: float, y: float) -> tuple[float, float]:
        dx, dy = x - self.x, y - self.y
        return math.hypot(dx, dy), _wrap_deg(math.degrees(math.atan2(dy, dx)) - self.yaw)

    def objects(self) -> list[dict]:
        with self._lock:
            out = []
            for o in self.objects_:
                dist, bearing = self._bearing(o.x, o.y)
                out.append({"name": o.name, "colour": o.colour, "shape": o.shape, "x": round(o.x, 3),
                            "y": round(o.y, 3), "z": round(o.z, 3), "size_m": round(o.size_m, 2),
                            "distance_m": round(dist, 2), "bearing_deg": round(bearing, 1),
                            "visible": bool(abs(bearing) <= self.hfov_deg / 2 and dist < 12.0),
                            "graspable": o.graspable})
            return out

    def lidar_summary(self, sectors: int = 12) -> list[dict]:
        with self._lock:
            width = 360.0 / sectors
            blockers = [o for o in self.objects_ if not o.graspable and self._held_by(o.name) is None]
            out = []
            for k in range(sectors):
                centre = -180.0 + (k + 0.5) * width
                best = None
                for j in range(5):  # 5 rays per sector
                    ang = math.radians(self.yaw + centre - width / 2 + (j + 0.5) * width / 5)
                    d = np.array([math.cos(ang), math.sin(ang)])
                    for o in blockers:
                        r = self._ray_circle(np.array([self.x, self.y]), d, np.array([o.x, o.y]), o.size_m / 2)
                        if r is not None and r <= 10.0 and (best is None or r < best):
                            best = r
                out.append({"bearing_deg": round(centre, 0), "nearest_m": None if best is None else round(best, 2)})
            return out

    @staticmethod
    def _ray_circle(p, d, c, r) -> float | None:
        f = p - c
        b = float(f @ d)
        disc = b * b - (float(f @ f) - r * r)
        if disc < 0:
            return None
        t = -b - math.sqrt(disc)
        return t if t >= 0 else None

    def camera_jpeg(self, camera: str = "head_cam", width: int = 480, height: int = 360) -> bytes:
        """Pinhole sketch of the visible objects (colour blocks on a floor/sky background)."""
        from PIL import Image, ImageDraw
        if camera not in self.cameras():
            raise ValueError(f"no camera named {camera!r}")
        w, h = int(width), int(height)
        img = Image.new("RGB", (w, h), (176, 200, 228))
        draw = ImageDraw.Draw(img)
        horizon = int(h * 0.45)
        draw.rectangle([0, horizon, w, h], fill=(122, 118, 110))
        f = (w / 2) / math.tan(math.radians(self.hfov_deg / 2))
        cam_h = 1.5 if camera == "head_cam" else 2.2
        with self._lock:
            items = sorted(self.objects(), key=lambda o: -o["distance_m"])
            for o in items:
                if not o["visible"] or o["distance_m"] < 0.2:
                    continue
                dist, b = o["distance_m"], math.radians(o["bearing_deg"])
                fwd = dist * math.cos(b)
                if fwd < 0.2:
                    continue
                u = w / 2 - f * math.tan(b)
                half_w = f * (o["size_m"] / 2) / fwd
                top_z = o["z"] + (o["z"] if not o.get("graspable") else o["size_m"] / 2)
                bot_z = max(0.0, o["z"] - (o["z"] if not o.get("graspable") else o["size_m"] / 2))
                v_top = horizon + f * (cam_h - top_z) / fwd
                v_bot = horizon + f * (cam_h - bot_z) / fwd
                rgb = tuple(int(255 * c) for c in PALETTE.get(o["colour"], (0.5, 0.5, 0.5)))
                if o["shape"] == "cylinder":
                    draw.ellipse([u - half_w, v_top - half_w * 0.3, u + half_w, v_bot], fill=rgb)
                else:
                    draw.rectangle([u - half_w, v_top, u + half_w, v_bot], fill=rgb)
        draw.text((6, 6), f"{camera} t={self.t:.1f}s", fill=(20, 20, 20))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    # ------------------------------------------------------------------ walking
    def walk_to(self, x: float, y: float, yaw_deg: float | None = None) -> str:
        with self._lock:
            self.log.append(("walk_to", float(x), float(y), yaw_deg))
            goal = np.array([float(x), float(y)])
            self._goal_yaw = None if yaw_deg is None else float(yaw_deg)
            if max(abs(goal[0]), abs(goal[1])) > self.world_half_size:
                return self._fail_goal(goal, f"no path to goal - ({x:.2f}, {y:.2f}) is outside the map")
            for o in self.objects_:
                if o.graspable or self._held_by(o.name):
                    continue
                if math.hypot(goal[0] - o.x, goal[1] - o.y) < o.size_m / 2 + self.body_radius:
                    return self._fail_goal(goal, f"no path to goal - it is inside or touching the {o.name}")
            self._goal, self._nav_state, self._nav_msg, self._nav_err = goal, "walking", "", None
            return f"walking to ({x:.2f}, {y:.2f})" + ("" if yaw_deg is None else f" facing {yaw_deg:.0f} deg")

    def _fail_goal(self, goal, msg: str) -> str:
        self._goal, self._last_goal = None, (goal, self._goal_yaw)
        self._nav_state, self._nav_msg, self._nav_err = "failed", msg, None
        return msg

    def walk_relative(self, forward_m: float, left_m: float = 0.0, turn_deg: float = 0.0) -> str:
        with self._lock:
            c, s = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
            gx = self.x + c * forward_m - s * left_m
            gy = self.y + s * forward_m + c * left_m
            return self.walk_to(gx, gy, _wrap_deg(self.yaw + turn_deg))

    def stop(self) -> str:
        with self._lock:
            self.log.append(("stop",))
            if self._goal is not None:
                self._last_goal = (self._goal, self._goal_yaw)
            self._goal, self._nav_state, self._nav_msg, self._speed = None, "idle", "stopped", 0.0
            return "stopping"

    def wait(self, seconds: float) -> None:
        # Like the live sim, time keeps flowing while idle: an idle wait advances one step, then returns.
        end = self.t + max(0.0, float(seconds))
        while self.t < end - 1e-9:
            if self.cancelled.is_set() or self.paused:
                return
            with self._lock:
                moving = self._moving()
                self._step(min(self.dt, end - self.t))
            if self.realtime > 0:
                time.sleep(self.dt / self.realtime)
            if not moving:
                return

    def _moving(self) -> bool:
        return self._nav_state == "walking" or any(a.state == "moving" for a in self.arms.values())

    # ------------------------------------------------------------------ physics-free stepping
    def _step(self, dt: float) -> None:
        self.t += dt
        self._step_base(dt)
        for arm in self.arms.values():
            self._step_arm(arm, dt)
            if arm.holding:
                o = self.obj(arm.holding)
                o.x, o.y, o.z = (float(v) for v in self._to_world(arm.hand))

    def _step_base(self, dt: float) -> None:
        self._speed = 0.0
        if self._nav_state != "walking" or self._goal is None:
            return
        dx, dy = self._goal[0] - self.x, self._goal[1] - self.y
        dist = math.hypot(dx, dy)
        max_turn = self.turn_rate * dt
        if dist > 0.03:
            err = _wrap_deg(math.degrees(math.atan2(dy, dx)) - self.yaw)
            self.yaw += float(np.clip(err, -max_turn, max_turn))
            if abs(err) < 20.0:
                step = min(self.walk_speed * dt, dist)
                self.x += step * math.cos(math.radians(self.yaw))
                self.y += step * math.sin(math.radians(self.yaw))
                self._speed = step / dt
            return
        if self._goal_yaw is not None:
            err = _wrap_deg(self._goal_yaw - self.yaw)
            if abs(err) > 1.0:
                self.yaw += float(np.clip(err, -max_turn, max_turn))
                return
        self._nav_err = round(dist, 3)
        self._last_goal = (self._goal, self._goal_yaw)
        self._goal, self._nav_state, self._nav_msg = None, "reached", f"goal reached ({dist:.3f} m)"

    def _step_arm(self, arm: _Arm, dt: float) -> None:
        if arm.state != "moving":
            return
        arm.timer -= dt
        if arm.target is not None:
            delta = arm.target - arm.hand
            n = float(np.linalg.norm(delta))
            step = self.arm_speed * dt
            arm.hand = arm.target.copy() if n <= step else arm.hand + delta * (step / n)
            if n <= step and arm.timer <= 0:
                arm.state = "failed" if arm.clipped > 0.03 else "done"
                arm.error_m = round(arm.clipped, 3) if arm.clipped > 0.03 else 0.005
        elif arm.timer <= 0:
            arm.state = "done"

    # ------------------------------------------------------------------ arms
    def _held_by(self, name: str) -> str | None:
        return next((s for s, a in self.arms.items() if a.holding == name), None)

    def _arm(self, side: str) -> _Arm:
        if side not in self.arms:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        return self.arms[side]

    def _move_hand(self, arm: _Arm, local_target: np.ndarray, skill: str, min_time: float = 0.0) -> float:
        sh = self._shoulder(arm.side)
        rel = local_target - sh
        n = float(np.linalg.norm(rel))
        clipped = sh + rel * (self.arm_reach / n) if n > self.arm_reach else local_target.copy()
        clipped[2] = max(clipped[2], 0.05)
        arm.clipped = float(np.linalg.norm(clipped - local_target))
        arm.target, arm.skill, arm.state, arm.error_m, arm.timer = clipped, skill, "moving", None, min_time
        return n

    def reach(self, side: str, x: float, y: float, z: float, approach=None) -> str:
        with self._lock:
            self.log.append(("reach", side, float(x), float(y), float(z)))
            arm = self._arm(side)
            n = self._move_hand(arm, self._to_local([x, y, z]), "reach")
            if n > self.arm_reach:
                return (f"target is {n:.2f} m from the {side} shoulder, beyond the {self.arm_reach:.2f} m reach; "
                        "moving as close as possible")
            return f"moving {side} hand to ({x:.2f}, {y:.2f}, {z:.2f})"

    def grasp(self, side: str) -> str:
        with self._lock:
            self.log.append(("grasp", side))
            arm = self._arm(side)
            palm = self._to_world(arm.hand)
            cands = [(float(np.linalg.norm(np.array([o.x, o.y, o.z]) - palm)), o) for o in self.objects_
                     if o.graspable and self._held_by(o.name) is None]
            cands.sort(key=lambda c: c[0])
            arm.skill = "grasp"
            if cands and cands[0][0] <= 0.10:
                arm.holding, arm.state, arm.target = cands[0][1].name, "holding", None
                return f"{side} hand closed on the {cands[0][1].name}"
            arm.state, arm.target = "failed", None
            near = f" (nearest graspable: {cands[0][1].name}, {cands[0][0]:.2f} m from the palm)" if cands else ""
            return f"nothing between the {side} fingers{near}"

    def release(self, side: str) -> str:
        with self._lock:
            self.log.append(("release", side))
            arm = self._arm(side)
            name, arm.holding = arm.holding, None
            arm.skill, arm.state, arm.target = "release", "done", None
            return f"{side} hand opened" + (f", dropped the {name}" if name else "")

    def point_at(self, side: str, x: float, y: float, z: float) -> str:
        with self._lock:
            self.log.append(("point_at", side, float(x), float(y), float(z)))
            arm = self._arm(side)
            sh = self._shoulder(side)
            rel = self._to_local([x, y, z]) - sh
            n = float(np.linalg.norm(rel)) or 1.0
            self._move_hand(arm, sh + rel / n * self.arm_reach * 0.98, "point_at", min_time=0.5)
            return f"pointing {side} arm at ({x:.2f}, {y:.2f}, {z:.2f})"

    def arm_gesture(self, side: str, gesture: str) -> str:
        with self._lock:
            self.log.append(("gesture", side, gesture))
            arm = self._arm(side)
            if gesture == "home":
                self._move_hand(arm, arm.rest.copy(), "home")
            elif gesture == "raise":
                self._move_hand(arm, self._shoulder(side) + np.array([0.1, 0.0, 0.6]), "raise")
            elif gesture == "wave":
                self._move_hand(arm, self._shoulder(side) + np.array([0.15, 0.0, 0.55]), "wave", min_time=2.0)
            else:
                raise ValueError(f"unknown gesture {gesture!r}; use home, wave or raise")
            return f"{side} arm: {gesture}"

    def arm_status(self) -> dict:
        with self._lock:
            return {s: {"skill": a.skill, "state": a.state,
                        "hand_xyz": [round(float(v), 3) for v in self._to_world(a.hand)],
                        "error_m": a.error_m, "holding": a.holding} for s, a in self.arms.items()}
