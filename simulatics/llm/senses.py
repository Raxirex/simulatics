"""Virtual sensory input for an LLM pilot: one compact observation bundle per decision.

Method: instead of raw sensor streams, the
model gets a few hundred tokens of structured text plus at most one small camera image:

- pose / proprioception: x, y, yaw, speed, upright flag, simulation time;
- skill status: navigation state and goal error, each arm's skill, state, palm position and held object;
- lidar: the nearest return per bearing sector, counter-clockwise from straight ahead, with `!` under
  0.6 m and the widest free heading (the sector summary used by NaVILA-style mid-level navigation,
  Cheng et al. 2024, and the Inner Monologue text-feedback recipe, Huang et al. 2022);
- semantic objects: names, distance, bearing and `visible` from segmentation (ground truth in sim;
  a Set-of-Mark style object list, Yang et al. 2023);
- optionally a head-camera JPEG for the vision model.

Every sensor read is best-effort: a failing sensor is reported as text and the rest still works.
"""
from __future__ import annotations

import base64
import io
import json
import math
import re
from dataclasses import dataclass, field

CLOSE_M = 0.6          # lidar ranges below this get a "!" marker


# ---------------------------------------------------------------------- formatting helpers
def _f(v, nd: int = 2, unit: str = "") -> str:
    if v is None:
        return "?"
    try:
        return f"{float(v):.{nd}f}{unit}"
    except (TypeError, ValueError):
        return str(v)


def describe_pose(p: dict) -> str:
    up = "upright" if p.get("upright", True) else "NOT UPRIGHT (fallen?)"
    return (f"pose: x={_f(p.get('x'))} y={_f(p.get('y'))} yaw={_f(p.get('yaw_deg'), 0)}deg "
            f"speed={_f(p.get('speed'), 2)}m/s height={_f(p.get('height'), 2)}m {up} t={_f(p.get('time'), 1)}s")


def describe_nav(n: dict) -> str:
    s = f"navigation: {n.get('state', '?')}"
    if n.get("goal") is not None:
        g = n["goal"]
        s += f", goal ({_f(g[0])}, {_f(g[1])})"
        if n.get("goal_yaw_deg") is not None:
            s += f" facing {_f(n['goal_yaw_deg'], 0)}deg"
    if n.get("error_m") is not None:
        s += f", error {_f(n['error_m'], 3)} m"
    if n.get("message"):
        s += f" ({n['message']})"
    return s


def describe_arms(arms: dict) -> str:
    if not arms:
        return "arms: not available on this robot"
    parts = []
    for side in sorted(arms):
        a = arms[side] or {}
        hand = a.get("hand_xyz")
        hand_s = "(" + ", ".join(_f(v) for v in hand) + ")" if hand else "?"
        skill = a.get("skill") or ""
        s = f"{side} {skill + '/' if skill else ''}{a.get('state', '?')} palm={hand_s}"
        if a.get("error_m") is not None:
            s += f" err={_f(a['error_m'], 3)}m"
        s += f" holding={a.get('holding') or 'nothing'}"
        parts.append(s)
    return "arms: " + "; ".join(parts)


def widest_free(sectors: list[dict]) -> tuple[float, float] | None:
    """(bearing_deg, nearest_m) of the most open sector; a sector with no return counts as infinitely open."""
    best = None
    for s in sectors:
        r = s.get("nearest_m")
        rr = math.inf if r is None else float(r)
        if best is None or rr > best[1] or (rr == best[1] and abs(s["bearing_deg"]) < abs(best[0])):
            best = (float(s["bearing_deg"]), rr)
    return best


def describe_lidar(sectors: list[dict]) -> str:
    """`lidar nearest m by bearing (0=ahead, +left): +0:2.40 | +30:0.55! | ...; most open: +60deg`."""
    if not sectors:
        return "lidar: no data"
    order = sorted(sectors, key=lambda s: float(s["bearing_deg"]) % 360.0)
    cells = []
    for s in order:
        r = s.get("nearest_m")
        b = float(s["bearing_deg"])
        b = b - 360.0 if b > 180.0 else b
        cells.append(f"{b:+.0f}:" + ("clear" if r is None else f"{float(r):.2f}" + ("!" if r < CLOSE_M else "")))
    w = widest_free(sectors)
    tail = "" if w is None else f"; most open: {w[0]:+.0f}deg" + ("" if math.isinf(w[1]) else f" ({w[1]:.1f} m)")
    return "lidar nearest m by bearing (0=ahead, +left, ! = under 0.6 m): " + " | ".join(cells) + tail


def describe_object(o: dict) -> str:
    s = (f"{o.get('name')}: {_f(o.get('distance_m'), 1)} m at bearing {_f(o.get('bearing_deg'), 0)}deg, "
         f"world ({_f(o.get('x'))}, {_f(o.get('y'))}")
    if o.get("z") is not None:
        s += f", z {_f(o['z'])}"
    s += f"), size {_f(o.get('size_m'), 2)} m"
    if o.get("graspable"):
        s += ", graspable"
    s += ", VISIBLE" if o.get("visible") else ", not in view"
    return s


def describe_objects(objects: list[dict], limit: int = 20) -> str:
    if not objects:
        return "objects: none known"
    objs = sorted(objects, key=lambda o: float(o.get("distance_m") or 0.0))
    lines = [f"objects ({len(objs)} known, nearest first):"]
    lines += ["  - " + describe_object(o) for o in objs[:limit]]
    if len(objs) > limit:
        lines.append(f"  ... and {len(objs) - limit} more (use list_objects)")
    return "\n".join(lines)


def thumbnail_b64(jpeg: bytes, max_side: int = 200, quality: int = 70) -> str:
    """Small base64 JPEG for the GUI transcript (never sent to the model)."""
    from PIL import Image
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def image_block(jpeg: bytes) -> dict:
    """A Messages-API image content block (base64 JPEG)."""
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(jpeg).decode("ascii")}}


# ---------------------------------------------------------------------- object lookup
_ARTICLES = re.compile(r"^(the|a|an|that|this)\s+")


def normalise_name(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())
    s = re.sub(r"\s+", " ", s).strip()
    return _ARTICLES.sub("", s)


def find_object(objects: list[dict], query: str, prefer_nearest: bool = False) -> tuple[dict | None, str]:
    """Resolve a (possibly loose) object name: exact -> prefix -> all words contained.

    Returns (object, "") or (None, helpful message). With several matches the nearest is used
    when `prefer_nearest`, otherwise the caller gets the candidate list (no guessing)."""
    q = normalise_name(query)
    if not q:
        return None, "empty object name"
    names = {id(o): normalise_name(o.get("name", "")) for o in objects}
    tiers = [
        [o for o in objects if names[id(o)] == q],
        [o for o in objects if names[id(o)].startswith(q)],
        [o for o in objects if set(q.split()) <= set(names[id(o)].split())],
        [o for o in objects if q in names[id(o)]],
    ]
    for matches in tiers:
        if len(matches) == 1:
            return matches[0], ""
        if len(matches) > 1:
            if prefer_nearest:
                return min(matches, key=lambda o: float(o.get("distance_m") or 0.0)), ""
            cands = ", ".join(sorted(o["name"] for o in matches))
            return None, f"'{query}' matches several objects: {cands}. Use the exact name."
    known = ", ".join(sorted(o.get("name", "?") for o in objects)) or "none"
    return None, f"no object named '{query}'. Known objects: {known}."


# ---------------------------------------------------------------------- the bundle
@dataclass
class Observation:
    pose: dict = field(default_factory=dict)
    nav: dict = field(default_factory=dict)
    arms: dict = field(default_factory=dict)
    lidar: list = field(default_factory=list)
    objects: list = field(default_factory=list)
    image: bytes | None = None
    camera: str | None = None
    errors: list = field(default_factory=list)

    def text(self, objects_limit: int = 20) -> str:
        lines = [describe_pose(self.pose) if self.pose else "pose: unavailable",
                 describe_nav(self.nav) if self.nav else "navigation: unavailable",
                 describe_arms(self.arms), describe_lidar(self.lidar),
                 describe_objects(self.objects, objects_limit)]
        if self.image is not None:
            lines.append(f"camera: {self.camera} image attached")
        lines += [f"sensor error: {e}" for e in self.errors]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"pose": self.pose, "nav": self.nav, "arms": self.arms, "lidar": self.lidar,
                "objects": self.objects, "camera": self.camera, "has_image": self.image is not None,
                "errors": list(self.errors)}

    def status_json(self) -> str:
        """Compact, key-sorted JSON of pose / navigation / arms (appended to motion tool results)."""
        return json.dumps({"pose": self.pose, "nav": self.nav, "arms": self.arms}, sort_keys=True,
                          separators=(",", ":"), default=str)


def _try(errors: list, label: str, fn, default):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - one broken sensor must not end the mission
        errors.append(f"{label}: {type(e).__name__}: {e}")
        return default


def observe(robot, *, lidar_sectors: int = 12, camera: str | None = None, width: int = 480, height: int = 360,
            include_lidar: bool = True, include_objects: bool = True) -> Observation:
    """Read every sense of a RobotInterface into one Observation (camera only if `camera` is given)."""
    errs: list[str] = []
    obs = Observation(errors=errs)
    obs.pose = _try(errs, "pose", robot.pose, {})
    obs.nav = _try(errs, "navigation", robot.nav_status, {})
    obs.arms = _try(errs, "arms", robot.arm_status, {})
    if include_lidar:
        obs.lidar = _try(errs, "lidar", lambda: robot.lidar_summary(lidar_sectors), [])
    if include_objects:
        obs.objects = _try(errs, "objects", robot.objects, [])
    if camera:
        obs.image = _try(errs, f"camera {camera}", lambda: robot.camera_jpeg(camera, width, height), None)
        obs.camera = camera if obs.image is not None else None
    return obs
