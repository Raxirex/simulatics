"""Make the reference clips physical: find motor commands that perform them in the physics simulation.

    python -m simulatics.samples.build_physical                all
    python -m simulatics.samples.build_physical backflip       some

For each entry of PHYSICAL: set the robot and world up exactly as the sample does, run the tracker
(imitate/physical.py), then play the result in a live SimSession and check what actually happened there: the
robot ended upright and standing still, it never counted as fallen after the balance controller took over,
and (per entry) it really left the ground, really rotated, really ended on the box. Only results that pass
are marked verified; the sample then runs in physics. The rest stay reference motions.
"""
from __future__ import annotations

import json
import sys
import time

import mujoco
import numpy as np

from . import ATHLETIC, DATA, _builtin, manifest

# id -> how to make it physical
#   clip      reference clip to follow (default: the id)
#   params    robot parameter overrides (None = the default robot)
#   end       use the clip only up to this many seconds (then ease to standing)
#   cfg       TrackConfig overrides
#   need      what must have happened: flight_s, com_rise_m, rotation_rad, min_up (at most), on_box (bool)
#   platform  (extra depth behind, extra width each side) in metres added to the box the clip lands on
#   hurdle    place a hurdle under the flight, as high as the feet clear (minus a margin)
PHYSICAL: dict[str, dict] = {
    "vertical_jump": {"need": {"flight_s": 0.3, "com_rise_m": 0.25}},
    "broad_jump": {"need": {"flight_s": 0.2, "travel_m": 0.6}},
    "hurdle": {"clip": "broad_jump", "hurdle": True, "need": {"flight_s": 0.2, "travel_m": 0.6},
               "title": "Jump a hurdle", "description": "A standing jump forwards over a hurdle. The hurdle is as "
               "high as the simulated robot's feet clear (the running take in the reference set starts mid-run, "
               "which a robot starting at rest cannot follow)."},
    "box_jump_up": {"params": ATHLETIC, "platform": (0.5, 0.3), "cfg": {"samples": 1800, "keep": 120, "passes": 5}, "need": {"flight_s": 0.2, "com_rise_m": 0.5, "on_box": True}},
    "box_jump_down": {"params": ATHLETIC, "end": 1.45, "cfg": {"lead_out": 0.9, "samples": 1800, "keep": 120, "passes": 5}, "need": {"flight_s": 0.3, "drop_m": 0.5}},
    "backflip": {"params": ATHLETIC, "need": {"flight_s": 0.4, "rotation_rad": 5.5, "min_up": -0.8}},
    "front_flip": {"params": ATHLETIC, "cfg": {"samples": 2000, "keep": 120, "passes": 4, "hold": 2.5, "lead_out": 0.9},
                   "need": {"flight_s": 0.3, "rotation_rad": 5.5, "min_up": -0.8}},
    "getup_front": {"params": {**ATHLETIC, "arm_torque": 150.0}, "from_ground": True,
                    "cfg": {"samples": 2000, "keep": 120, "passes": 4, "lead_out": 0.9, "hold": 2.0},
                    "need": {"rise_m": 0.7}},
    "getup_back": {"params": {**ATHLETIC, "arm_torque": 150.0}, "from_ground": True,
                   "cfg": {"samples": 2000, "keep": 120, "passes": 4, "lead_out": 0.9, "hold": 2.0},
                   "need": {"rise_m": 0.7}},
    "cartwheel": {"params": {**ATHLETIC, "arm_torque": 150.0}, "cfg": {"samples": 2000, "keep": 120, "passes": 3, "hold": 2.5, "lead_out": 0.9},
                  "need": {"rotation_rad": 4.8, "min_up": -0.7}},
}


def _setup(session, props: list[dict], params: dict | None):
    _builtin(session, props, params)
    with session.lock:
        return session.export_xml(), session.data.qpos.copy()


def verify(session, tape, props: list[dict], params: dict | None, need: dict, settle: float = 6.0) -> dict:
    """Play a tape in the live session and measure what happened. Returns a report with "ok"."""
    from ..imitate.motion import quat_to_mat
    _builtin(session, props, params)
    session.play_tape(tape)
    session.speed = 8.0
    falls0 = session.falls
    names, z, up, xy, air, t_end, still = set(), [], [], [], [], None, 0
    floor_like = {g for g in range(session.model.ngeom) if session.model.geom_bodyid[g] == 0}
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        with session.lock:
            c, d, m = session.controller, session.data, session.model
            names.add(c.name)
            z.append(float(d.qpos[2]))
            xy.append(d.qpos[:2].copy())
            up.append(float(d.xmat[1].reshape(3, 3)[2, 2]))
            touching = any((d.contact[i].geom1 in floor_like) != (d.contact[i].geom2 in floor_like) for i in range(d.ncon))
            if c.name == "motion-physical":
                air.append(not touching)
            elif t_end is None:
                t_end = float(d.time)
            now, speed = float(d.time), float(np.linalg.norm(d.qvel[:3]))
            falls = session.falls - falls0
        if t_end is not None:                      # the balance controller may take a step or two to square up
            still = still + 1 if speed < 0.05 else 0
            if now - t_end >= settle or (still > 300 and now - t_end > 1.5) or falls:
                break
        time.sleep(0.001)
    with session.lock:
        session.running = False
    m = tape.meta
    travel = float(np.linalg.norm(xy[-1] - xy[0]))
    box_top = max((2 * p["size"][2] for p in props if p.get("kind") == "terrain"), default=0.0)
    rep = {"controllers": sorted(names), "falls_after": falls, "final_up": round(up[-1], 3), "final_speed": round(speed, 3),
           "final_height_m": round(z[-1], 3), "travel_m": round(travel, 2), "min_up": round(min(up), 2),
           "flight_s": m.get("flight_s"), "com_rise_m": m.get("com_rise_m"), "rotation_rad": m.get("rotation_rad"),
           "drop_m": round(z[0] - z[-1], 2), "rise_m": round(z[-1] - z[0], 2), "on_box": bool(box_top > 0.2 and z[-1] - z[0] > 0.6 * box_top)}
    ok = (t_end is not None and falls == 0 and up[-1] > 0.95 and speed < 0.1 and "walker" in names)
    why = [] if ok else ["did not end standing still and upright"]
    for k, v in need.items():
        good = (rep[k] <= v) if k == "min_up" else (rep[k] is True if isinstance(v, bool) else rep[k] is not None and rep[k] >= v)
        if not good:
            ok = False
            why.append(f"{k} is {rep[k]}, needed {'at most' if k == 'min_up' else 'at least'} {v}")
    rep["ok"], rep["why"] = bool(ok), why
    return rep


def _hurdle_for(tape, model: mujoco.MjModel) -> dict | None:
    """A hurdle under the middle of the simulated flight, as high as both feet clear there, minus a margin."""
    from .build import find_hurdles
    d = mujoco.MjData(model)
    feet = [b for b in range(model.nbody) if model.body(b).name.endswith("_foot")]
    geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in feet]
    low, mid = [], []
    for q in tape.states:
        d.qpos[:] = q
        mujoco.mj_kinematics(model, d)
        z = []
        for g in geoms:
            R = d.geom_xmat[g].reshape(3, 3)
            z.append(float(d.geom_xpos[g][2] - np.abs(R[2] * model.geom_size[g]).sum()))
        low.append(min(z))
        mid.append(np.mean([d.xpos[b][:2] for b in feet], axis=0))
    low, mid = np.array(low), np.array(mid)
    air = low > 0.06
    if not air.any():
        return None
    k = int(np.argmax(low))
    a = k
    while a > 0 and air[a - 1]:
        a -= 1
    b = k
    while b < len(air) - 1 and air[b + 1]:
        b += 1
    x = float((mid[a][0] + mid[b][0]) / 2)                     # half way between take-off and landing
    near = np.abs(mid[:, 0] - x) < 0.12                         # frames in which the feet are over it
    height = float(max(0.08, low[near & air].min() - 0.05))
    head = mid[b] - mid[a]
    return {"name": "hurdle", "shape": "box", "colour": "orange", "rgba": [0.95, 0.55, 0.15, 1.0],
            "pos": [x, float((mid[a][1] + mid[b][1]) / 2)], "yaw": float(np.arctan2(head[1], head[0])),
            "size": [0.015, 0.6, height / 2]}


def build(cid: str, session, log=print) -> dict:
    from ..imitate import load_motion
    from ..imitate.physical import STATE, Tape, TrackConfig, Tracker, fingerprint
    spec = PHYSICAL[cid]
    man = manifest()
    base = man[spec.get("clip", cid)]
    props = [dict(p) for p in (base.get("props") or [])]
    for p in props:                                           # a deeper, wider platform to land and balance on
        if spec.get("platform") and p.get("kind") == "terrain":
            back, side = spec["platform"]
            p["size"] = [p["size"][0] + back / 2, p["size"][1] + side, p["size"][2]]
            p["pos"] = [p["pos"][0] + back / 2, p["pos"][1]]
    params = spec.get("params")
    motion = load_motion(str(DATA / base["file"]))
    if spec.get("end"):
        from ..imitate.motion import Motion
        n = int(round(spec["end"] * motion.fps)) + 1
        motion = Motion(motion.fps, motion.root_pos[:n], motion.root_quat[:n], list(motion.joint_names),
                        motion.joint_pos[:n], name=motion.name)
    xml, q0 = _setup(session, props, params)
    tr = Tracker(xml, TrackConfig(**spec.get("cfg", {})))
    try:
        tape = tr.track(motion, q0, name=cid, log=None, from_ground=bool(spec.get("from_ground")))
    finally:
        tr.close()
    if spec.get("hurdle"):
        h = _hurdle_for(tape, mujoco.MjModel.from_xml_string(xml))
        if h is not None:                                     # same commands in the world that has the hurdle
            props = props + [h]
            xml2, _ = _setup(session, props, params)
            tr2 = Tracker(xml2, TrackConfig(**spec.get("cfg", {})))
            try:
                s0 = np.zeros(mujoco.mj_stateSize(tr2.plain, STATE))
                d = mujoco.MjData(tr2.plain)
                d.qpos[:] = tape.qpos0
                mujoco.mj_getState(tr2.plain, d, s0, STATE)
                states = tr2.replay(s0, tape.ctrl)
                tape = Tape(tape.ctrl, tape.control_steps, tape.timestep, tape.qpos0, tape.qvel0, states, tape.reference,
                            fingerprint(tr2.plain), cid, {**tape.meta, "hurdle_height_m": round(2 * h["size"][2], 3)})
            finally:
                tr2.close()
    rep = verify(session, tape, props, params, spec.get("need", {}))
    # how fragile is it: the same commands from a start state nudged by a micrometre
    nudged = Tape(tape.ctrl, tape.control_steps, tape.timestep, tape.qpos0 + 1e-6 * np.sign(tape.qpos0 + 0.5), tape.qvel0,
                  tape.states, tape.reference, tape.model, cid, dict(tape.meta))
    rep2 = verify(session, nudged, props, params, spec.get("need", {}))
    tape.meta.update({"verified": rep["ok"], "survives_nudge": rep2["ok"], "live": rep})
    entry = {"file": f"{cid}.tape.npz", "clip": spec.get("clip", cid), "params": params or {}, "props": props,
             "verified": rep["ok"], "survives_nudge": rep2["ok"], "duration_s": round(tape.duration, 2),
             "seconds_to_find": tape.meta["seconds"], "live": rep,
             "title": spec.get("title", base["title"].replace(" (reversed backflip)", "")),
             "description": spec.get("description", base["description"]), "source": base["source"]}
    if rep["ok"]:
        tape.save(DATA / entry["file"])
    log(f"{cid:15s} {'VERIFIED' if rep['ok'] else 'failed  '} nudge {'ok' if rep2['ok'] else 'no'}  flight {rep['flight_s']} s  "
        f"rise {rep['com_rise_m']} m  rotation {rep['rotation_rad']} rad  up {rep['final_up']}  falls {rep['falls_after']}  "
        f"{tape.meta['seconds']} s  {'; '.join(rep['why'])}")
    return entry


def main(argv: list[str] | None = None) -> None:
    from ..server.session import SimSession
    args = list(argv if argv is not None else sys.argv[1:])
    seed = 0
    if "--seed" in args:                                       # another random draw for the ones that failed
        i = args.index("--seed")
        seed = int(args[i + 1])
        del args[i:i + 2]
    ids = args or list(PHYSICAL)
    if seed:
        for cid in ids:
            PHYSICAL[cid] = {**PHYSICAL[cid], "cfg": {**PHYSICAL[cid].get("cfg", {}), "seed": 1000 * seed}}
    path = DATA / "physical.json"
    out = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    session = SimSession()
    session.lidar_cfg.enabled = False
    try:
        for cid in ids:
            if seed and out.get(cid, {}).get("verified"):
                continue                                       # already works
            entry = build(cid, session)
            if entry["verified"] or cid not in out or not out[cid].get("verified"):
                out[cid] = entry                               # never replace a working result with a failed one
            path.write_text(json.dumps({k: out[k] for k in PHYSICAL if k in out}, indent=1), encoding="utf-8")
    finally:
        session.shutdown()


if __name__ == "__main__":
    main()
