"""The pilot's tool surface: strict JSON-schema tools mapped onto `RobotInterface`.

Design:

- **Small skill API, not joint control.** As in Vemprala et al. 2023
  and later skill-orchestration work, the model chooses goals and skills; the
  planner, path follower and arm controller in the simulator do the closed-loop work.
- **Strict schemas + client-side validation.** Every tool has `strict: true` and
  `additionalProperties: false`, so arguments always match the schema. Strict mode does not support
  numeric `minimum`/`maximum`, so ranges are stated in the descriptions and enforced here: NaN/inf
  and unknown names are errors, durations/relative moves are clamped (the result says so), absolute
  goals outside the world are errors.
- **Blocking skills with text feedback (Inner Monologue, Huang et al. 2022).** Tools that move
  the robot wait (via `robot.wait`, in short slices so a stop is honoured within ~0.25 s) until the
  motion ends or times out, then return one outcome sentence plus fresh pose / navigation / arm status.
  A motion that does not reach its goal returns `is_error` with the reason, so the model can replan.
- **`look` returns pixels.** The camera JPEG goes back inside the tool_result as an image block.

`Toolbox.execute()` never raises except `Cancelled`; every other failure becomes an error result.
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field
from typing import Callable

from .mission import Cancelled
from .senses import (describe_arms, describe_lidar, describe_nav, describe_object, describe_objects,
                     describe_pose, find_object, image_block, observe)

SIDES = ("left", "right")
GESTURES = ("wave", "raise", "home")


class ToolError(Exception):
    """Invalid arguments or an impossible request; the message is shown to the model."""


@dataclass
class ToolResult:
    text: str
    images: list[bytes] = field(default_factory=list)   # JPEG frames
    is_error: bool = False
    finished: bool = False
    summary: str | None = None
    success: bool | None = None

    def content_blocks(self) -> list[dict]:
        """tool_result content: image blocks first, then the text."""
        return [image_block(j) for j in self.images] + [{"type": "text", "text": self.text or "(no output)"}]


@dataclass
class ToolConfig:
    default_camera: str = "head_cam"
    image_width: int = 480
    image_height: int = 360
    send_images: bool = True             # False: `look` describes the view in text only
    world_limit_m: float = 50.0          # |x|, |y| of absolute goals
    max_relative_m: float = 5.0          # walk_relative per-axis clamp
    max_wait_s: float = 60.0
    min_walk_timeout_s: float = 20.0
    walk_speed_hint_mps: float = 0.25    # conservative speed for walk timeouts
    max_walk_timeout_s: float = 180.0
    arm_timeout_s: float = 15.0
    arm_reach_hint_m: float = 1.0        # horizontal base-to-object distance beyond which reaching is refused
    default_object_z_m: float = 0.5      # used when an object has no height information
    standoff_default_m: float = 0.5
    pick_standoff_m: float = 0.3     # graspable props: close enough for the hands (they reach ~0.45 m forward)
    hand_offset_m: float = 0.2       # stand with a graspable object this far to one hand's side
    reach_adjust_step_m: float = 0.08  # reach_object: step this much closer once if just out of reach (0 = off)
    wait_slice_s: float = 0.25           # sim seconds per robot.wait call (cancellation latency)
    stall_wall_s: float = 5.0            # sim clock not advancing this long -> "simulation paused"
    lidar_sectors: int = 12


def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


_NUM = {"type": "number"}
_SIDE = {"type": "string", "enum": list(SIDES), "description": "Which arm/hand: left or right."}

# Stable, deterministic tool definitions (rendered first in every request, so they are cached).
_TOOL_SPECS: list[dict] = [
    {"name": "look",
     "description": "Take a picture with one of the robot's cameras and return the image. Call this whenever "
                    "the task depends on what things look like, before acting on a visual claim, and after a "
                    "motion to check the result. head_cam looks where the robot faces (75 deg field of view, "
                    "slightly down); work_cam looks 55 deg down at the hands and the table in front (use it "
                    "before and after reaching or grasping); chase is a third-person view from behind.",
     "input_schema": _obj({"camera": {"type": "string", "description": "Camera name, default head_cam."}}, [])},
    {"name": "scan_surroundings",
     "description": "Lidar summary (nearest obstacle per bearing sector, most open direction) plus the object "
                    "list. Call this to check for obstacles before walking somewhere new.",
     "input_schema": _obj({}, [])},
    {"name": "list_objects",
     "description": "List the named world objects with world position, distance, bearing (0 = straight ahead, "
                    "positive = to the left), size and whether each is visible in the head camera. Use the exact "
                    "names it returns with walk_to_object, reach_object and point_at.",
     "input_schema": _obj({"visible_only": {"type": "boolean", "description": "Only objects in the head camera view."}},
                          [])},
    {"name": "walk_to",
     "description": "Walk to a world point (metres) along a planned path around obstacles, optionally ending "
                    "facing yaw_deg (world frame, 0 = +x, counter-clockwise positive). Blocks until the robot "
                    "arrives, fails or times out, then returns the outcome and status. |x|, |y| <= 50.",
     "input_schema": _obj({"x": _NUM, "y": _NUM,
                           "yaw_deg": {"type": "number", "description": "Final heading in degrees (optional)."}},
                          ["x", "y"])},
    {"name": "walk_relative",
     "description": "Walk in the robot's own frame: forward_m ahead (negative = back), left_m to the left "
                    "(negative = right), and end turned by turn_deg (positive = counter-clockwise/left). Each "
                    "distance is clamped to +-5 m and the turn to +-180 deg. Use turn-only moves "
                    "(0, 0, turn_deg) to look around. Blocks until done.",
     "input_schema": _obj({"forward_m": _NUM, "left_m": _NUM, "turn_deg": _NUM},
                          ["forward_m", "left_m", "turn_deg"])},
    {"name": "walk_to_object",
     "description": "Walk up to a named object and end facing it, stopping standoff_m (0-3 m) between the "
                    "robot's centre and the object's near edge. Default 0.3 for graspable objects (close enough "
                    "for the hands) and 0.5 otherwise; does nothing if the robot is already that close and facing "
                    "it. Blocks until done.",
     "input_schema": _obj({"name": {"type": "string", "description": "Exact object name from list_objects."},
                           "standoff_m": {"type": "number", "description": "Gap to the object's edge, metres."}},
                          ["name"])},
    {"name": "stop",
     "description": "Stop walking immediately and stand still.",
     "input_schema": _obj({}, [])},
    {"name": "wait",
     "description": "Let simulation time pass (0.1-60 s) or until the current walk/arm motion ends, whichever is "
                    "first. Returns the status afterwards.",
     "input_schema": _obj({"seconds": _NUM}, ["seconds"])},
    {"name": "reach",
     "description": "Move one hand's grasp point (between the finger pads) to a world point x, y, z in metres "
                    "(z up, 0 = floor, 0-2.5). The point must be within arm's reach (each hand reaches about 0.6 m "
                    "from its shoulder, on its own side of the body). Blocks until the arm stops; reports the "
                    "remaining error.",
     "input_schema": _obj({"side": _SIDE, "x": _NUM, "y": _NUM, "z": _NUM}, ["side", "x", "y", "z"])},
    {"name": "reach_object",
     "description": "Move one hand to a named object's centre so it can be grasped; graspable objects are "
                    "approached from above. Walk close first (walk_to_object). Blocks until the arm stops.",
     "input_schema": _obj({"side": _SIDE, "name": {"type": "string", "description": "Exact object name."}},
                          ["side", "name"])},
    {"name": "grasp",
     "description": "Close that hand's gripper on whatever is between the fingers. Reach the object first; the "
                    "result says what (if anything) is now held.",
     "input_schema": _obj({"side": _SIDE}, ["side"])},
    {"name": "release",
     "description": "Open that hand's gripper, dropping anything it holds.",
     "input_schema": _obj({"side": _SIDE}, ["side"])},
    {"name": "point_at",
     "description": "Point one arm at a named object (give `object`) or at a world point (give x and y, "
                    "optional z, default 1.0 m).",
     "input_schema": _obj({"side": _SIDE, "object": {"type": "string", "description": "Exact object name."},
                           "x": _NUM, "y": _NUM, "z": _NUM}, ["side"])},
    {"name": "gesture",
     "description": "Named arm motion: wave (greeting), raise (hand up), home (arm down by the side).",
     "input_schema": _obj({"side": _SIDE, "gesture": {"type": "string", "enum": list(GESTURES)}},
                          ["side", "gesture"])},
    {"name": "get_status",
     "description": "Current pose, navigation state, arm states, lidar summary and nearby objects, as text.",
     "input_schema": _obj({}, [])},
    {"name": "finish",
     "description": "End the mission. Call this exactly once, when the mission is complete or cannot be "
                    "completed, with a short summary of what was done and observed.",
     "input_schema": _obj({"summary": {"type": "string"},
                           "success": {"type": "boolean", "description": "True if the mission was achieved."}},
                          ["summary", "success"])},
]

TOOL_NAMES = tuple(t["name"] for t in _TOOL_SPECS)


def tool_definitions() -> list[dict]:
    """Strict tool definitions for the Messages API (fresh deep copy, deterministic order)."""
    return [{**copy.deepcopy(t), "strict": True} for t in _TOOL_SPECS]


# ---------------------------------------------------------------------- argument validation
def _number(args: dict, key: str, *, lo: float | None = None, hi: float | None = None, default=None,
            clamp: bool = True, notes: list | None = None, required: bool = True) -> float | None:
    v = args.get(key, default)
    if v is None:
        if required and default is None:
            raise ToolError(f"missing required number '{key}'")
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ToolError(f"'{key}' must be a number, got {json.dumps(v)}")
    v = float(v)
    if not math.isfinite(v):
        raise ToolError(f"'{key}' must be a finite number, got {v}")
    if lo is not None and hi is not None and not lo <= v <= hi:
        if not clamp:
            raise ToolError(f"'{key}'={v:g} is out of range [{lo:g}, {hi:g}]")
        c = min(max(v, lo), hi)
        if notes is not None:
            notes.append(f"{key} clamped from {v:g} to {c:g}")
        v = c
    return v


def _string(args: dict, key: str, required: bool = True) -> str | None:
    v = args.get(key)
    if v is None and not required:
        return None
    if not isinstance(v, str) or not v.strip():
        raise ToolError(f"'{key}' must be a non-empty string")
    return v.strip()


def _side(args: dict) -> str:
    s = args.get("side")
    if s not in SIDES:
        raise ToolError(f"'side' must be 'left' or 'right', got {json.dumps(s)}")
    return s


class Toolbox:
    """Executes tool calls against a RobotInterface. `cancelled()` is polled during every wait."""

    def __init__(self, robot, config: ToolConfig | None = None, cancelled: Callable[[], bool] | None = None):
        self.robot = robot
        self.config = config or ToolConfig()
        self._cancelled = cancelled or (lambda: False)
        self.calls = 0

    # ---- public ----------------------------------------------------------
    def definitions(self) -> list[dict]:
        return tool_definitions()

    def execute(self, name: str, args: dict | None) -> ToolResult:
        self.calls += 1
        args = dict(args or {})
        fn = getattr(self, f"_t_{name}", None) if name in TOOL_NAMES else None
        if fn is None:
            return ToolResult(f"unknown tool '{name}'. Available: {', '.join(TOOL_NAMES)}", is_error=True)
        allowed = set(next(t for t in _TOOL_SPECS if t["name"] == name)["input_schema"]["properties"])
        extra = sorted(set(args) - allowed)
        try:
            if extra:
                raise ToolError(f"unexpected argument(s) {extra}; {name} takes {sorted(allowed) or 'no arguments'}")
            return fn(args)
        except Cancelled:
            raise
        except ToolError as e:
            return ToolResult(f"{name} error: {e}", is_error=True)
        except Exception as e:  # noqa: BLE001 - robot/sim failures go back to the model as text
            return ToolResult(f"{name} failed: {type(e).__name__}: {e}", is_error=True)

    # ---- waiting ---------------------------------------------------------
    def _check(self) -> None:
        if self._cancelled():
            raise Cancelled("stopped")

    def _await(self, phase: Callable[[], str | None], timeout_s: float) -> tuple[str, float]:
        """Wait (in short robot.wait slices) until `phase()` returns a terminal word.

        Returns (outcome, sim seconds waited); outcome is phase's word, "timeout" or "paused". The
        duration is clamped at 0: the live session resets the clock to 0 when it auto-resets a fall."""
        r, cfg = self.robot, self.config
        t0 = last_t = float(r.sim_time())
        last_change = time.monotonic()
        while True:
            self._check()
            word = phase()
            if word is not None:
                return word, max(0.0, float(r.sim_time()) - t0)
            elapsed = float(r.sim_time()) - t0
            if elapsed >= timeout_s:
                return "timeout", elapsed
            r.wait(min(cfg.wait_slice_s, max(0.01, timeout_s - elapsed)))
            now = float(r.sim_time())
            if now > last_t + 1e-9:
                last_t, last_change = now, time.monotonic()
            elif time.monotonic() - last_change > cfg.stall_wall_s:
                return "paused", max(0.0, now - t0)
            else:
                time.sleep(0.02)   # sim not advancing (paused / between ticks): don't spin

    def _await_walk(self, timeout_s: float) -> tuple[str, float, dict]:
        seen = {"walking": False}
        t0 = float(self.robot.sim_time())

        def phase():
            st = self.robot.nav_status().get("state")
            if st == "walking":
                seen["walking"] = True
                return None
            if st in ("reached", "failed", "fallen"):
                return st
            if st == "idle" and (seen["walking"] or float(self.robot.sim_time()) - t0 > 1.0):
                return "stopped"
            return None

        word, dt = self._await(phase, timeout_s)
        return word, dt, self.robot.nav_status()

    def _await_arm(self, side: str) -> tuple[str, float, dict, bool]:
        seen = {"moving": False}
        t0 = float(self.robot.sim_time())

        def phase():
            st = (self.robot.arm_status().get(side) or {}).get("state")
            if st == "moving":
                seen["moving"] = True
                return None
            if st in ("done", "failed", "holding") or (st == "idle" and (seen["moving"] or
                                                                         float(self.robot.sim_time()) - t0 > 0.5)):
                return st
            return None

        word, dt = self._await(phase, self.config.arm_timeout_s)
        return word, dt, (self.robot.arm_status().get(side) or {}), seen["moving"]

    def _walk_timeout(self, dist: float, turn_deg: float = 180.0) -> float:
        c = self.config
        t = c.min_walk_timeout_s + dist / c.walk_speed_hint_mps + abs(turn_deg) / 45.0
        return min(t, c.max_walk_timeout_s)

    def _status_line(self) -> str:
        """Fresh pose / navigation / arm status as three short text lines."""
        obs = observe(self.robot, include_lidar=False, include_objects=False)
        lines = [describe_pose(obs.pose) if obs.pose else "pose: unavailable",
                 describe_nav(obs.nav) if obs.nav else "navigation: unavailable", describe_arms(obs.arms)]
        return "\n".join(lines + [f"sensor error: {e}" for e in obs.errors])

    def _walk_outcome(self, what: str, goal_xy, timeout_s: float, start_msg: str, notes: list) -> ToolResult:
        word, dt, nav = self._await_walk(timeout_s)
        p = self.robot.pose()
        dist = math.hypot(p["x"] - goal_xy[0], p["y"] - goal_xy[1])
        where = f"now at ({p['x']:.2f}, {p['y']:.2f}) yaw {p['yaw_deg']:.0f}deg, {dist:.2f} m from the goal"
        msg = nav.get("message") or ""
        if word == "reached":
            text, err = f"{what}: reached after {dt:.1f} s; {where}.", False
            if dist > 0.5:
                text += " (navigation reported the goal reached but the robot is still far from it)"
                err = True
        elif word == "failed":
            text, err = f"{what}: FAILED ({msg or 'no reason given'}); {where}.", True
        elif word == "fallen":
            text, err = f"{what}: the robot FELL ({msg}); {where}. It may have been reset.", True
        elif word == "stopped":
            text, err = f"{what}: walking stopped before arrival ({msg or 'stopped'}); {where}.", True
        elif word == "paused":
            text, err = f"{what}: the simulation is paused, no progress for {self.config.stall_wall_s:.0f} s " \
                        f"of wall time; {where}. Ask the user to resume it.", True
        else:
            text, err = f"{what}: still walking after the {timeout_s:.0f} s timeout; {where}.", True
        if start_msg:
            text = f"[{start_msg}] " + text
        if notes:
            text += " Note: " + "; ".join(notes) + "."
        return ToolResult(text + "\n" + self._status_line(), is_error=err)

    def _arm_outcome(self, side: str, what: str, start_msg: str, notes: list | None = None) -> ToolResult:
        word, dt, st, moved = self._await_arm(side)
        hand = st.get("hand_xyz")
        hand_s = "(" + ", ".join(f"{v:.2f}" for v in hand) + ")" if hand else "?"
        err_m = st.get("error_m")
        holding = st.get("holding")
        detail = f"palm at {hand_s}" + (f", error {err_m:.3f} m" if err_m is not None else "") + \
                 f", holding {holding or 'nothing'}" + (f" ({st['message']})" if st.get("message") else "")
        if word == "idle" and not moved:   # the command was not accepted (see the robot's message)
            text, err = f"{what}: the arm did not start moving; {detail}.", True
        elif word == "idle":
            # Finished skills end in done / holding / failed; an arm that drops from moving to idle was
            # reset or relaxed mid-motion (e.g. the live session auto-resets after a fall), so the goal
            # was not achieved.
            text, err = (f"{what}: INTERRUPTED - the arm went idle mid-motion (the simulation or arm controller "
                         f"was reset?); {detail}."), True
        elif word in ("done", "holding"):
            text, err = f"{what}: {word} after {dt:.1f} s; {detail}.", False
        elif word == "failed":
            text, err = f"{what}: FAILED; {detail}.", True
        elif word == "paused":
            text, err = f"{what}: the simulation is paused; {detail}.", True
        else:
            text, err = f"{what}: arm still moving after {self.config.arm_timeout_s:.0f} s; {detail}.", True
        text = f"[{start_msg}] {text}" if start_msg else text
        if notes:
            text += " Note: " + "; ".join(notes) + "."
        return ToolResult(text + "\n" + self._status_line(), is_error=err)

    def _object(self, name: str) -> dict:
        obj, msg = find_object(self.robot.objects(), name)
        if obj is None:
            raise ToolError(msg)
        return obj

    # ---- senses ------------------------------------------------------------
    def _t_look(self, a: dict) -> ToolResult:
        cfg = self.config
        cam = _string(a, "camera", required=False) or cfg.default_camera
        cams = list(self.robot.cameras() or [])
        if cams and cam not in cams:
            raise ToolError(f"no camera '{cam}'. Cameras: {', '.join(cams)}")
        jpeg = self.robot.camera_jpeg(cam, cfg.image_width, cfg.image_height) if cfg.send_images else None
        if cfg.send_images and not jpeg:
            raise ToolError(f"camera '{cam}' returned no image")
        p = self.robot.pose()
        vis = [o for o in self.robot.objects() if o.get("visible")]
        vis_s = "; ".join(f"{o['name']} ({o['distance_m']:.1f} m, {o['bearing_deg']:+.0f}deg)" for o in
                          sorted(vis, key=lambda o: o["distance_m"])) or "none"
        what = f"{cam} image {cfg.image_width}x{cfg.image_height}" if jpeg else \
            f"{cam} (camera images are disabled by the operator; text only)"
        text = (f"{what} at t={p.get('time', 0):.1f}s from ({p['x']:.2f}, {p['y']:.2f}) facing "
                f"{p['yaw_deg']:.0f}deg. Segmentation says visible: {vis_s}.")
        return ToolResult(text, images=[jpeg] if jpeg else [])

    def _t_scan_surroundings(self, a: dict) -> ToolResult:
        obs = observe(self.robot, lidar_sectors=self.config.lidar_sectors)
        return ToolResult(obs.text())

    def _t_list_objects(self, a: dict) -> ToolResult:
        vo = a.get("visible_only", False)
        if not isinstance(vo, bool):
            raise ToolError("'visible_only' must be true or false")
        objs = self.robot.objects()
        if vo:
            objs = [o for o in objs if o.get("visible")]
            if not objs:
                return ToolResult("no objects are visible in the head camera; turn (walk_relative 0, 0, 60) or "
                                  "list all objects")
        return ToolResult(describe_objects(objs, limit=100))

    def _t_get_status(self, a: dict) -> ToolResult:
        return ToolResult(observe(self.robot, lidar_sectors=self.config.lidar_sectors).text())

    # ---- walking -------------------------------------------------------------
    def _t_walk_to(self, a: dict) -> ToolResult:
        lim = self.config.world_limit_m
        x = _number(a, "x", lo=-lim, hi=lim, clamp=False)
        y = _number(a, "y", lo=-lim, hi=lim, clamp=False)
        yaw = _number(a, "yaw_deg", required=False)
        if yaw is not None:
            yaw = (yaw + 180.0) % 360.0 - 180.0
        p = self.robot.pose()
        msg = self.robot.walk_to(x, y, yaw)
        return self._walk_outcome(f"walk_to ({x:.2f}, {y:.2f})", (x, y),
                                  self._walk_timeout(math.hypot(x - p["x"], y - p["y"])), msg, [])

    def _t_walk_relative(self, a: dict) -> ToolResult:
        notes: list[str] = []
        m = self.config.max_relative_m
        fwd = _number(a, "forward_m", lo=-m, hi=m, notes=notes)
        left = _number(a, "left_m", lo=-m, hi=m, notes=notes)
        turn = _number(a, "turn_deg", lo=-180.0, hi=180.0, notes=notes)
        p = self.robot.pose()
        yaw = math.radians(p["yaw_deg"])
        gx = p["x"] + math.cos(yaw) * fwd - math.sin(yaw) * left
        gy = p["y"] + math.sin(yaw) * fwd + math.cos(yaw) * left
        msg = self.robot.walk_relative(fwd, left, turn)
        return self._walk_outcome(f"walk_relative (forward {fwd:g} m, left {left:g} m, turn {turn:g} deg)",
                                  (gx, gy), self._walk_timeout(math.hypot(fwd, left), turn), msg, notes)

    def _t_walk_to_object(self, a: dict) -> ToolResult:
        notes: list[str] = []
        name = _string(a, "name")
        obj = self._object(name)
        graspable = bool(obj.get("graspable"))
        default = self.config.pick_standoff_m if graspable else self.config.standoff_default_m
        standoff = _number(a, "standoff_m", lo=0.0, hi=3.0, default=default, notes=notes)
        p = self.robot.pose()
        ox, oy = float(obj["x"]), float(obj["y"])
        dx, dy = p["x"] - ox, p["y"] - oy
        d = math.hypot(dx, dy)
        if d < 1e-6:
            yaw = math.radians(p["yaw_deg"])
            dx, dy, d = -math.cos(yaw), -math.sin(yaw), 1.0
        r = float(obj.get("size_m") or 0.0) / 2.0 + standoff
        bearing = math.degrees(math.atan2(oy - p["y"], ox - p["x"])) - p["yaw_deg"]
        bearing = (bearing + 180.0) % 360.0 - 180.0
        if d <= r + 0.12 and abs(bearing) <= 45.0:
            # already there: walking "to" the standoff point would step backwards, away from the object
            return ToolResult(f"walk_to_object '{obj['name']}': already {d:.2f} m away and facing it "
                              f"(bearing {bearing:.0f} deg), no walking needed" +
                              (" - within arm's reach" if graspable else "") + f"\n{self._status_line()}")
        gx, gy = ox + dx / d * r, oy + dy / d * r
        face = math.degrees(math.atan2(oy - gy, ox - gx))
        hand_note = ""
        if graspable:
            # Each hand works on its own side of the body (no shoulder yaw), so stand with the object in
            # front of one hand rather than dead centre: offset sideways by the shoulder half-width.
            fx, fy = -dx / d, -dy / d              # facing direction (robot -> object)
            lx, ly = -fy, fx                       # robot's left when facing the object
            side = "left" if bearing > 0 else "right"
            off = self.config.hand_offset_m * (1.0 if side == "left" else -1.0)
            gx, gy = gx - lx * off, gy - ly * off
            face = math.degrees(math.atan2(fy, fx))
            hand_note = f"; lined up with the {side} hand"
        msg = self.robot.walk_to(gx, gy, face)
        res = self._walk_outcome(f"walk_to_object '{obj['name']}' (goal ({gx:.2f}, {gy:.2f}), standoff "
                                 f"{standoff:.2f} m{hand_note})",
                                 (gx, gy), self._walk_timeout(math.hypot(gx - p["x"], gy - p["y"])), msg, notes)
        if res.is_error and "no path" in res.text.lower() and "standoff_m" not in a and standoff < 0.5:
            # the close-in point is blocked (e.g. by a big table): retry from the ordinary distance
            retry = self._t_walk_to_object({**a, "standoff_m": 0.5})
            retry.text = (f"(the {standoff:.2f} m standoff point was blocked; walked to a 0.50 m standoff "
                          f"instead)\n" + retry.text)
            return retry
        if res.is_error and "no path" in res.text.lower():
            res.text += "\nHint: the goal point may be inside another object (e.g. a table); try a larger standoff_m."
        try:
            now, _ = find_object(self.robot.objects(), obj["name"])
            if now is not None:
                res.text += f"\nobject now: {describe_object(now)}"
        except Exception:  # noqa: BLE001
            pass
        return res

    def _t_stop(self, a: dict) -> ToolResult:
        msg = self.robot.stop()
        return ToolResult(f"stop: {msg}\n{self._status_line()}")

    def _t_wait(self, a: dict) -> ToolResult:
        notes: list[str] = []
        secs = _number(a, "seconds", lo=0.0, hi=self.config.max_wait_s, notes=notes)
        r = self.robot
        t0 = float(r.sim_time())

        def phase():
            if float(r.sim_time()) - t0 >= secs - 1e-6:
                return "done"
            return None if self._anything_moving() else "motion ended"

        word, dt = self._await(phase, secs + 1e-3) if self._anything_moving() else ("idle", 0.0)
        text = f"waited {dt:.1f} s of simulation time" + {"idle": " (nothing was moving)",
                                                          "motion ended": " (the motion ended)"}.get(word, "")
        if word == "paused":
            text += " - the simulation appears paused"
        if notes:
            text += ". Note: " + "; ".join(notes)
        return ToolResult(text + "\n" + self._status_line())

    def _anything_moving(self) -> bool:
        if self.robot.nav_status().get("state") == "walking":
            return True
        return any((a or {}).get("state") == "moving" for a in (self.robot.arm_status() or {}).values())

    # ---- arms ----------------------------------------------------------------
    def _reach_point(self, side: str, x: float, y: float, z: float, label: str, notes: list,
                     approach: list[float] | None = None) -> ToolResult:
        p = self.robot.pose()
        horiz = math.hypot(x - p["x"], y - p["y"])
        if horiz > self.config.arm_reach_hint_m:
            raise ToolError(f"{label} is {horiz:.2f} m from the robot (horizontal) - out of arm's reach. Walk "
                            f"closer first (walk_to_object, or walk_to).")
        msg = self.robot.reach(side, x, y, z, approach=approach) if approach else self.robot.reach(side, x, y, z)
        return self._arm_outcome(side, f"reach {side} hand to ({x:.2f}, {y:.2f}, {z:.2f})", msg, notes)

    def _t_reach(self, a: dict) -> ToolResult:
        side = _side(a)
        lim = self.config.world_limit_m
        x = _number(a, "x", lo=-lim, hi=lim, clamp=False)
        y = _number(a, "y", lo=-lim, hi=lim, clamp=False)
        notes: list[str] = []
        z = _number(a, "z", lo=0.0, hi=2.5, notes=notes)
        return self._reach_point(side, x, y, z, f"point ({x:.2f}, {y:.2f})", notes)

    def _t_reach_object(self, a: dict) -> ToolResult:
        side = _side(a)
        obj = self._object(_string(a, "name"))
        notes: list[str] = []
        z = obj.get("z")
        if z is None:
            z = self.config.default_object_z_m
            notes.append(f"the object's height is unknown, used z={z:.2f} m")
        # graspable props are picked from above, so the forearm does not sweep them off the table
        approach = [0.0, 0.0, -1.0] if obj.get("graspable") else None
        ox, oy = float(obj["x"]), float(obj["y"])
        res = self._reach_point(side, ox, oy, float(z), f"the {obj['name']}", notes, approach=approach)
        if res.is_error and "out of reach" in res.text.lower() and self.config.reach_adjust_step_m > 0:
            # Just out of reach: one short corrective step towards the object, then try once more.
            p = self.robot.pose()
            dx, dy = ox - p["x"], oy - p["y"]
            dist = math.hypot(dx, dy)
            if 0.2 < dist < self.config.arm_reach_hint_m:
                k = self.config.reach_adjust_step_m / dist
                gx, gy = p["x"] + dx * k, p["y"] + dy * k
                msg = self.robot.walk_to(gx, gy, p["yaw_deg"])
                step = self._walk_outcome("corrective step", (gx, gy), self._walk_timeout(0.3), msg, notes)
                if not step.is_error:
                    retry = self._reach_point(side, ox, oy, float(z), f"the {obj['name']}", notes, approach=approach)
                    retry.text = (f"(first reach was just out of range; stepped {self.config.reach_adjust_step_m:.2f} "
                                  f"m closer and retried)\n" + retry.text)
                    return retry
        return res

    def _t_grasp(self, a: dict) -> ToolResult:
        side = _side(a)
        msg = self.robot.grasp(side)
        res = self._arm_outcome(side, f"grasp {side}", msg)
        held = (self.robot.arm_status().get(side) or {}).get("holding")
        if not held:
            res.is_error = True
            res.text = "Nothing is held. " + res.text
        return res

    def _t_release(self, a: dict) -> ToolResult:
        side = _side(a)
        msg = self.robot.release(side)
        return self._arm_outcome(side, f"release {side}", msg)

    def _t_point_at(self, a: dict) -> ToolResult:
        side = _side(a)
        target = _string(a, "object", required=False)
        notes: list[str] = []
        if target:
            if any(a.get(k) is not None for k in ("x", "y")):
                notes.append("both an object and x/y were given; pointing at the object")
            obj = self._object(target)
            x, y = float(obj["x"]), float(obj["y"])
            z = float(obj.get("z") if obj.get("z") is not None else self.config.default_object_z_m)
            label = f"point {side} arm at the {obj['name']}"
        else:
            if a.get("x") is None or a.get("y") is None:
                raise ToolError("give either 'object' or both 'x' and 'y'")
            lim = self.config.world_limit_m
            x = _number(a, "x", lo=-lim, hi=lim, clamp=False)
            y = _number(a, "y", lo=-lim, hi=lim, clamp=False)
            z = _number(a, "z", lo=0.0, hi=3.0, default=1.0, notes=notes)
            label = f"point {side} arm at ({x:.2f}, {y:.2f}, {z:.2f})"
        msg = self.robot.point_at(side, x, y, z)
        return self._arm_outcome(side, label, msg, notes)

    def _t_gesture(self, a: dict) -> ToolResult:
        side = _side(a)
        g = a.get("gesture")
        if g not in GESTURES:
            raise ToolError(f"'gesture' must be one of {list(GESTURES)}, got {json.dumps(g)}")
        msg = self.robot.arm_gesture(side, g)
        return self._arm_outcome(side, f"gesture {g} ({side})", msg)

    # ---- control ---------------------------------------------------------------
    def _t_finish(self, a: dict) -> ToolResult:
        summary = _string(a, "summary")
        ok = a.get("success")
        if not isinstance(ok, bool):
            raise ToolError("'success' must be true or false")
        return ToolResult(f"mission finished ({'success' if ok else 'not achieved'})", finished=True,
                          summary=summary, success=ok)


def lidar_text(robot, sectors: int = 12) -> str:
    """Convenience for GUIs/tests: the lidar line the model sees."""
    return describe_lidar(robot.lidar_summary(sectors))
