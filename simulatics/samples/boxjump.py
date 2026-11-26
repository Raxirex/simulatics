"""Jump onto a box of any height: the robot tries, in physics, and the result says whether it made it.

The one motion-capture take of a jump onto a platform (about 0.68 m at the robot's scale) is reshaped for the
height asked for: the part of the root's rise that comes from the platform is scaled to the new height, with
extra clearance at the top of the flight for taller boxes. For tall boxes the arms are brought forward and down
so the hands can land on the box and help ("arms"). The motion tracker (imitate/physical.py) then searches
for motor commands that perform it, and the result is played in the live session and checked: did the robot
end standing on the box?

    from simulatics.samples import boxjump
    result = boxjump.attempt(session, 0.5)          # tries, plays it, returns what happened
    result["ok"], result["message"]

A try takes one to three minutes of search. Attempts that worked are kept (runs/box_jumps) and replayed at
once the next time the same height is asked for. Heights that are beyond the robot are reported as failed,
with how far it got; nothing is faked.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from . import ATHLETIC, DATA, manifest

MIN_H, MAX_H = 0.10, 1.40
ARMS_FROM = 0.80                       # from this height the arms are used
STRONG_ARMS = {**ATHLETIC, "arm_torque": 150.0}


def store() -> Path:
    from ..jobs import RUNS
    return RUNS / "box_jumps"


def shipped() -> dict:
    """Heights that come with Simulatics as verified recordings: {"0.45": {"file", "arms", ...}}."""
    p = DATA / "box_jumps.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _key(height: float) -> str:
    return f"{float(height):.2f}"


def _sole_low(model: mujoco.MjModel, q: np.ndarray) -> np.ndarray:
    d = mujoco.MjData(model)
    feet = [g for g in range(model.ngeom) if model.body(int(model.geom_bodyid[g])).name.endswith("_foot")]
    out = np.zeros(len(q))
    for t in range(len(q)):
        d.qpos[:] = q[t]
        mujoco.mj_kinematics(model, d)
        out[t] = min(float(d.geom_xpos[g][2] - np.abs(d.geom_xmat[g].reshape(3, 3)[2] * model.geom_size[g]).sum()) for g in feet)
    return out


def reference(model: mujoco.MjModel, height: float, arms: bool = False):
    """(motion, box prop) for a jump onto a box `height` m high, made from the mocap take."""
    from ..imitate import load_motion
    from ..imitate.motion import Motion
    clip = manifest()["box_jump_up"]
    base = load_motion(str(DATA / clip["file"]))
    box = dict(clip["props"][0])
    h0 = 2.0 * float(box["size"][2])
    q = base.qpos(model)
    low = _sole_low(model, q)
    air = low > 0.04
    a = int(np.argmax(air))                                   # take-off
    on = np.flatnonzero((low > 0.6 * h0) & (np.abs(np.gradient(low)) * base.fps < 0.6))
    b = int(on[on > a][0]) if len(on[on > a]) else len(q) - 1  # feet on the box
    w = np.clip((np.arange(len(q)) - a) / max(b - a, 1), 0.0, 1.0)
    w = 3 * w ** 2 - 2 * w ** 3
    extra = max(0.0, height - h0)
    q = q.copy()
    q[:, 2] += (height - h0) * w + 0.35 * extra * 4 * w * (1 - w)        # platform part rescaled, more clearance if taller
    if arms:
        names = {model.joint(j).name: int(model.jnt_qposadr[j]) for j in range(model.njnt)}
        reach = np.clip((np.arange(len(q)) - a) / max(b - a, 1), 0.0, 1.0)
        reach = np.where(np.arange(len(q)) > b + int(0.4 * base.fps), np.clip(1.0 - (np.arange(len(q)) - b - 0.4 * base.fps) / (0.4 * base.fps), 0, 1), reach)
        for side in ("left", "right"):
            if f"{side}_shoulder_y" in names:
                q[:, names[f"{side}_shoulder_y"]] += reach * (-1.5 - q[:, names[f"{side}_shoulder_y"]])   # arms forward
                q[:, names[f"{side}_elbow"]] += reach * (-0.5 - q[:, names[f"{side}_elbow"]])
    motion = Motion.from_qpos(model, q, base.fps, name=f"box_jump_{_key(height)}", keypoints=False)
    depth = max(float(box["size"][0]) + 0.25, 0.55)
    front = float(box["pos"][0]) - float(box["size"][0])                # keep the near edge where the take had it
    prop = {**box, "name": f"box {height:.2f} m", "size": [depth, float(box["size"][1]) + 0.3, height / 2],
            "pos": [front + depth, float(box["pos"][1])], "z": height / 2}
    return motion, prop


def find(height: float) -> tuple[Path, dict] | None:
    """A recording that already works for this height (shipped, or kept from an earlier attempt)."""
    k = _key(height)
    e = shipped().get(k)
    if e and (DATA / e["file"]).exists():
        return DATA / e["file"], e
    idx = store() / "index.json"
    if idx.exists():
        e = json.loads(idx.read_text(encoding="utf-8")).get(k)
        if e and e.get("ok") and (store() / e["file"]).exists():
            return store() / e["file"], e
    return None


def play(session, path: Path, entry: dict) -> str:
    from . import _builtin
    _builtin(session, entry["props"], entry["params"])
    return session.play_tape(str(path))


def search(session, height: float, arms: bool | None = None, budget: dict | None = None, log=None) -> dict:
    """Search for motor commands that make the jump, verify them live, and keep them if they work."""
    from ..imitate.physical import TrackConfig, Tracker
    from .build_physical import _setup, verify
    height = float(np.clip(height, MIN_H, MAX_H))
    arms = height >= ARMS_FROM if arms is None else bool(arms)
    params = STRONG_ARMS if arms else dict(ATHLETIC)
    xml0, _ = _setup(session, [], params)
    motion, box = reference(mujoco.MjModel.from_xml_string(xml0), height, arms)
    xml, q0 = _setup(session, [box], params)
    # the search is a gamble on landing balanced: a second, differently seeded try rescues many heights
    tries = [{"samples": 1800, "keep": 120, "passes": 5}, {"samples": 1800, "keep": 120, "passes": 5, "seed": 7, "sigma_legs": 0.25}]
    for extra in tries[:1] if budget and budget.get("single") else tries:
        cfg = TrackConfig(**{**extra, **{k: v for k, v in (budget or {}).items() if k != "single"}})
        tr = Tracker(xml, cfg)
        try:
            tape = tr.track(motion, q0, name=f"box jump {height:.2f} m", log=log)
        finally:
            tr.close()
        live = verify(session, tape, [box], params, {"on_box": True})
        if live["ok"]:
            break
    if not live["ok"]:                 # leave a usable scene: the robot standing in front of the box, at normal speed
        from . import _builtin
        _builtin(session, [box], params)
        session.running = True
    session.speed = 1.0
    feet_high = round(float(tape.meta.get("com_rise_m", 0.0)), 2)
    entry = {"height_m": height, "arms": arms, "ok": bool(live["ok"]), "props": [box], "params": params,
             "file": f"box_jump_{_key(height)}.tape.npz", "com_rise_m": feet_high, "flight_s": tape.meta.get("flight_s"),
             "seconds_to_find": tape.meta.get("seconds"), "live": live}
    if live["ok"]:
        store().mkdir(parents=True, exist_ok=True)
        tape.meta.update(props=[box], params=params)
        tape.save(store() / entry["file"])
        idx = store() / "index.json"
        data = json.loads(idx.read_text(encoding="utf-8")) if idx.exists() else {}
        data[_key(height)] = entry
        idx.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return entry


def attempt(session, height: float, arms: bool | None = None, budget: dict | None = None, log=None) -> dict:
    """Jump onto a box `height` m high: replay a recording that works, or search for one (minutes) and play it.
    Returns {"ok", "message", "height_m", "arms", "searched"}."""
    height = float(np.clip(height, MIN_H, MAX_H))
    found = find(height)
    if found is not None and (arms is None or bool(found[1].get("arms")) == bool(arms)):
        msg = play(session, *found)
        return {"ok": True, "searched": False, "height_m": height, "arms": bool(found[1].get("arms")),
                "message": f"Jumping onto a {height:.2f} m box" + (" with the arms" if found[1].get("arms") else "") + f": {msg}"}
    e = search(session, height, arms, budget, log)
    if e["ok"]:
        play(session, store() / e["file"], e)
        return {"ok": True, "searched": True, "height_m": height, "arms": e["arms"],
                "message": f"Made it onto the {height:.2f} m box" + (" using the arms" if e["arms"] else "")
                           + f" (found in {e['seconds_to_find']:.0f} s); playing it"}
    why = "; ".join(e["live"].get("why") or []) or "it did not end standing on the box"
    return {"ok": False, "searched": True, "height_m": height, "arms": e["arms"],
            "message": f"Could not get onto a {height:.2f} m box" + (" even with the arms" if e["arms"] else "")
                       + f": centre of mass rose {e['com_rise_m']:.2f} m; {why}"}


def main(argv: list[str] | None = None) -> None:
    """python -m simulatics.samples.boxjump 0.3 0.5 0.9 [--ship]: try heights; --ship adds the ones that work
    to the recordings that come with Simulatics."""
    import shutil
    import sys
    from ..server.session import SimSession
    args = list(argv if argv is not None else sys.argv[1:])
    ship = "--ship" in args
    heights = [float(a) for a in args if not a.startswith("--")]
    s = SimSession()
    s.lidar_cfg.enabled = False
    try:
        for h in heights:
            e = search(s, h)
            print(f"{h:.2f} m  {'arms' if e['arms'] else 'legs'}  {'MADE IT' if e['ok'] else 'failed '}  centre of mass rose {e['com_rise_m']:.2f} m  "
                  f"flight {e['flight_s']} s  {e['seconds_to_find']:.0f} s  {'; '.join(e['live'].get('why') or [])}", flush=True)
            if e["ok"] and ship:
                shutil.copy2(store() / e["file"], DATA / e["file"])
                data = shipped()
                data[_key(h)] = {k: e[k] for k in ("height_m", "arms", "props", "params", "file", "com_rise_m", "flight_s")}
                (DATA / "box_jumps.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    finally:
        s.shutdown()


if __name__ == "__main__":
    main()
