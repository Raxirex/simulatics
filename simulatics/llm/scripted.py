"""ScriptedPilot: an offline, rule-based pilot with the same interface as ClaudePilot.

It parses simple imperative missions with regular expressions and runs a fixed tool sequence for
each clause, through the same Toolbox and transcript events as the LLM pilot. It exists so the GUI
demo and the tests work without an API key, and as the "oracle with the true goal" baseline
(an LLM pilot should never do worse than this).

Understood clauses (joined with "then", "and then", ";" or "and"):
    walk/go/move to (x, y) [facing D]          -> walk_to
    walk/go to [the] <object>                  -> walk_to_object
    walk/go forward|back N [m]                 -> walk_relative
    turn left|right [N degrees] / turn around  -> walk_relative(0, 0, +-N)
    pick up / grab / take [the] <object> [with the left|right hand]
                                               -> walk_to_object, reach_object, grasp, lift 0.2 m
    drop / release / put down [it]             -> release (the hand that holds something)
    point at [the] <object>                    -> point_at
    wave / raise your [left|right] hand / arms down -> gesture
    look / what do you see / scan              -> look + list_objects
    stop                                       -> stop
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Union

from .mission import MissionContext, Outcome
from .senses import find_object

Args = Union[dict, Callable[[], dict]]

_NUM = r"(-?\d+(?:\.\d+)?)"
_SPLIT = re.compile(r"\s*(?:,?\s*\band then\b|,?\s*\bthen\b|;|\.\s+)\s*")
_SIDE = re.compile(r"\b(?:with\s+)?(?:your\s+|the\s+|its\s+|my\s+)?(left|right)\s+(?:hand|arm)\b")
_FILLER = re.compile(r"\b(please|now|slowly|quickly|carefully)\b")


@dataclass
class Step:
    label: str
    calls: list[tuple[str, Args]] = field(default_factory=list)


class ScriptedPilot:
    name = "scripted"
    HELP = ("The offline pilot understands: 'go to 2 3', 'walk to the red box', 'walk forward 2 m', "
            "'turn left 90 degrees', 'pick up the blue cube', 'drop it', 'point at the red box', 'wave', "
            "'raise your left hand', 'look', 'stop' - joined with 'then'. Use the Claude pilot for anything else.")

    def __init__(self, lift_m: float = 0.2, standoff_m: float = 0.5, pick_standoff_m: float = 0.3):
        self.lift_m = lift_m
        self.standoff_m = standoff_m
        self.pick_standoff_m = pick_standoff_m
        self._robot = None

    # ---- parsing -----------------------------------------------------------------
    def plan(self, mission: str, robot) -> list[Step]:
        """Clauses -> Steps. Raises ValueError (with HELP) for anything not understood."""
        self._robot = robot
        text = mission.strip().lower().rstrip(".!? ")
        steps: list[Step] = []
        for clause in [c for c in _SPLIT.split(text) if c.strip()]:
            # "X and Y" is tried as two actions FIRST: the gesture rules match anywhere in a clause, so
            # parsing "wave and walk to the red box" whole would silently drop the walk. If any part does
            # not parse (e.g. an object named "black and white box"), the whole clause is parsed instead.
            if " and " in clause:
                try:
                    parts = [self._clause(p) for p in clause.split(" and ")]
                except ValueError:
                    parts = None
                if parts and all(parts):
                    for p in parts:
                        steps.extend(p)
                    continue
            err: ValueError | None = None
            try:
                parsed = self._clause(clause)
            except ValueError as e:      # e.g. an object name that does not exist
                parsed, err = None, e
            if parsed is None:
                raise err or ValueError(f"could not understand '{clause}'. {self.HELP}")
            steps.extend(parsed)
        if not steps:
            raise ValueError(f"empty mission. {self.HELP}")
        return steps

    def _resolve(self, phrase: str) -> str:
        phrase = _SIDE.sub("", phrase).strip(" ,")
        obj, msg = find_object(self._robot.objects(), phrase, prefer_nearest=True)
        if obj is None:
            raise ValueError(msg)
        return obj["name"]

    def _clause(self, c: str) -> list[Step] | None:
        c = re.sub(r"\s+", " ", _FILLER.sub("", c)).strip(" ,")
        side_m = _SIDE.search(c)
        side = side_m.group(1) if side_m else None
        if re.fullmatch(r"(stop|halt|freeze)( walking| moving)?", c):
            return [Step("stop", [("stop", {})])]
        m = re.fullmatch(r"(?:walk|go|move|navigate|head|come)\s+(?:back\s+)?(?:to|towards?)\s+\(?\s*" + _NUM +
                         r"\s*[, ]\s*" + _NUM + r"\s*\)?(?:\s*(?:facing|yaw|heading)\s*" + _NUM + r"\s*(?:deg\w*)?)?",
                         c)
        if m:
            args = {"x": float(m.group(1)), "y": float(m.group(2))}
            if m.group(3) is not None:
                args["yaw_deg"] = float(m.group(3))
            return [Step(f"walk to ({args['x']:g}, {args['y']:g})", [("walk_to", args)])]
        m = re.fullmatch(r"(?:walk|go|move|step)\s+(forwards?|ahead|backwards?|back)\s+" + _NUM +
                         r"\s*(?:m|meters?|metres?)?", c)
        if m:
            d = float(m.group(2)) * (-1 if m.group(1).startswith("back") else 1)
            return [Step(f"walk {d:+g} m", [("walk_relative", {"forward_m": d, "left_m": 0.0, "turn_deg": 0.0})])]
        m = re.fullmatch(r"turn\s+(left|right|around)(?:\s+(?:by\s+)?" + _NUM + r"\s*(?:deg\w*)?)?", c)
        if m:
            deg = 180.0 if m.group(1) == "around" else float(m.group(2) or 90.0)
            deg = -deg if m.group(1) == "right" else deg
            return [Step(f"turn {deg:+g} deg", [("walk_relative", {"forward_m": 0.0, "left_m": 0.0, "turn_deg": deg})])]
        m = re.fullmatch(r"(?:pick\s+up|grab|grasp|take|fetch)\s+(.+?)(?:\s+up)?", c)
        if m:
            return [self._pick(self._resolve(m.group(1)), side)]
        if re.fullmatch(r"(?:drop|release|let go of|put down|put)(?:\s+(?:it|that|the .+|.+ down|down))?", c):
            return [Step("release", [("release", lambda: {"side": side or self._holding_side() or "right"})])]
        m = re.fullmatch(r"(?:walk|go|move|navigate|head|come)\s+(?:over\s+)?(?:to|towards?|up to|near)\s+(.+)", c)
        if m:
            name = self._resolve(m.group(1))
            return [Step(f"walk to the {name}", [("walk_to_object", {"name": name, "standoff_m": self.standoff_m})])]
        m = re.fullmatch(r"point\s+(?:at|to(?:wards?)?)\s+(.+)", c)
        if m:
            name = self._resolve(m.group(1))
            return [Step(f"point at the {name}", [("point_at", {"side": side or "right", "object": name})])]
        if re.search(r"\bwave\b", c):
            s = side or "right"
            return [Step("wave", [("gesture", {"side": s, "gesture": "wave"}), ("gesture", {"side": s, "gesture": "home"})])]
        if re.search(r"\braise\b|\bhands? up\b|\barms? up\b", c):
            return [Step("raise hand", [("gesture", {"side": side or "right", "gesture": "raise"})])]
        if re.search(r"\b(arms? down|lower (your )?(arms?|hands?)|home)\b", c):
            sides = [side] if side else ["left", "right"]
            return [Step("arms down", [("gesture", {"side": s, "gesture": "home"}) for s in sides])]
        if re.fullmatch(r"(look( around)?|what do you see|describe( the scene| what you see)?|scan( the area)?)", c):
            return [Step("look", [("look", {}), ("list_objects", {"visible_only": False})])]
        return None

    def _pick(self, name: str, side: str | None) -> Step:
        def hand() -> str:
            if side:
                return side
            obj, _ = find_object(self._robot.objects(), name)
            return "left" if obj is not None and float(obj.get("bearing_deg") or 0.0) > 5.0 else "right"

        chosen: dict[str, str] = {}

        def reach_args() -> dict:
            chosen["side"] = hand()
            return {"side": chosen["side"], "name": name}

        def lift_args() -> dict:
            s = chosen.get("side", "right")
            x, y, z = (self._robot.arm_status().get(s) or {}).get("hand_xyz") or [0.0, 0.0, 1.0]
            return {"side": s, "x": float(x), "y": float(y), "z": float(z) + self.lift_m}

        return Step(f"pick up the {name}", [
            ("walk_to_object", {"name": name}),  # tool default: close enough for the hands
            ("reach_object", reach_args),
            ("grasp", lambda: {"side": chosen.get("side", "right")}),
            ("reach", lift_args),
        ])

    def _holding_side(self) -> str | None:
        for s, a in (self._robot.arm_status() or {}).items():
            if (a or {}).get("holding"):
                return s
        return None

    # ---- running -------------------------------------------------------------------
    def run(self, mission: str, ctx: MissionContext) -> Outcome:
        ctx.emit("user", mission)
        try:
            steps = self.plan(mission, ctx.robot)
        except ValueError as e:
            return Outcome("error", error=str(e))
        ctx.emit("text", "Plan: " + "; ".join(s.label for s in steps) + ".", name="scripted")
        done: list[str] = []
        for step in steps:
            for tool, args in step.calls:
                ctx.raise_if_cancelled()
                a = args() if callable(args) else dict(args)
                res = ctx.call_tool(tool, a)
                if res.is_error:
                    first = res.text.splitlines()[0] if res.text else "error"
                    return Outcome("error", error=f"step '{step.label}' failed: {first}",
                                   summary="; ".join(done), success=False)
            done.append(step.label)
        p = ctx.robot.pose()
        held = [f"{s} hand holds the {a['holding']}" for s, a in (ctx.robot.arm_status() or {}).items()
                if (a or {}).get("holding")]
        summary = (f"Done: {'; '.join(done)}. Robot at ({p['x']:.2f}, {p['y']:.2f}) facing {p['yaw_deg']:.0f} deg"
                   + (", " + ", ".join(held) if held else "") + ".")
        return Outcome("done", summary=summary, success=True)
