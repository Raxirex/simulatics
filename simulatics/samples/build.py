"""Rebuild the reference clips in samples/data from their motion-capture sources.

    python -m simulatics.samples.build              all clips
    python -m simulatics.samples.build backflip     some

Each clip: download the BVH (cached in runs/mocap_cache), cut the wanted frames, retarget to the built-in
humanoid (imitate/retarget.py), derive the props the performer used (the box they jumped onto, the obstacle
they cleared) from the retargeted motion itself, and save data/<id>.npz. The manifest lists every clip with
its source and licence. Only the retargeted joint trajectories are stored, not the original files.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import mujoco
import numpy as np

from . import DATA
from .clips import CLIPS

CACHE = Path("runs") / "mocap_cache"
CMU = "CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu), subject {s} trial {t}"

def fetch(url: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / url.rsplit("/", 1)[-1]
    if not out.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "simulatics-samples"})
        with urllib.request.urlopen(req, timeout=60) as r:   # noqa: S310 - fixed https URLs from CLIPS
            data = r.read()
        if not data.lstrip().startswith(b"HIERARCHY"):
            raise ValueError(f"{url} is not a BVH file")
        out.write_bytes(data)
    return out


def default_model() -> mujoco.MjModel:
    from ..models import HumanoidParams, build_mjcf
    return mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams()))


def sole_tracks(model: mujoco.MjModel, motion) -> np.ndarray:
    """(T, 2, 3): lowest point of each sole (left, right) in every frame."""
    from ..imitate.retarget import _feet_state, _foot_geom_corners
    feet = _foot_geom_corners(model)
    d = mujoco.MjData(model)
    out = np.zeros((motion.n_frames, 2, 3))
    q = motion.qpos(model)
    for t in range(motion.n_frames):
        d.qpos[:] = q[t]
        mujoco.mj_kinematics(model, d)
        st = _feet_state(model, d, feet)
        for i, side in enumerate(sorted(st)):
            centre, _yaw, low = st[side]
            out[t, i] = (centre[0], centre[1], low)
    return out


def lowest_body_height(model: mujoco.MjModel, motion) -> np.ndarray:
    """(T,) height of the lowest point of any robot shape (bounding spheres), for clips where hands touch down."""
    d = mujoco.MjData(model)
    q = motion.qpos(model)
    robot = np.flatnonzero(model.geom_bodyid > 0)
    out = np.zeros(motion.n_frames)
    for t in range(motion.n_frames):
        d.qpos[:] = q[t]
        mujoco.mj_kinematics(model, d)
        out[t] = float(np.min(d.geom_xpos[robot, 2] - model.geom_rbound[robot]))
    return out


def lowest_exact(model: mujoco.MjModel, motion, arms: bool = False) -> np.ndarray:
    """(T,) exact signed distance from the floor to the nearest robot shape (arms excluded unless asked: they
    are handled by clear_floor)."""
    d = mujoco.MjData(model)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    skip = ("arm", "hand", "finger")
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] > 0
             and (arms or not any(k in model.body(int(model.geom_bodyid[g])).name for k in skip))]
    q = motion.qpos(model)
    out = np.zeros(len(q))
    for t in range(len(q)):
        d.qpos[:] = q[t]
        mujoco.mj_kinematics(model, d)
        out[t] = min(mujoco.mj_geomDistance(model, d, g, floor, 1.0, None) for g in geoms)
    return out


def find_boxes(soles: np.ndarray, fps: float, min_height: float = 0.3, still: float = 0.35) -> list[dict]:
    """Platforms the feet rest on: runs of frames where a sole is nearly still well above the floor."""
    T = len(soles)
    vel = np.zeros((T, 2))
    vel[1:] = np.linalg.norm(np.diff(soles, axis=0), axis=2) * fps
    on = (vel < still) & (soles[:, :, 2] > min_height)
    pts = soles[on]
    if len(pts) < 4:
        return []
    boxes: list[dict] = []
    # one platform per height band (clips here use a single box; two bands would give two boxes)
    order = np.argsort(pts[:, 2])
    pts = pts[order]
    groups, start = [], 0
    for i in range(1, len(pts) + 1):
        if i == len(pts) or pts[i, 2] - pts[i - 1, 2] > 0.12:
            groups.append(pts[start:i])
            start = i
    for k, g in enumerate(groups):
        if len(g) < 4:
            continue
        top = float(np.median(g[:, 2])) - 0.005
        lo, hi = g[:, :2].min(0) - 0.22, g[:, :2].max(0) + 0.22
        boxes.append({"name": f"box {k + 1}", "kind": "terrain", "shape": "box", "colour": "wood",
                      "rgba": [0.72, 0.55, 0.36, 1.0], "pos": [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)],
                      "yaw": 0.0, "size": [float((hi[0] - lo[0]) / 2), float((hi[1] - lo[1]) / 2), top / 2]})
    return boxes


def find_hurdles(soles: np.ndarray, root_xy: np.ndarray, min_clear: float = 0.25, width: float = 1.2) -> list[dict]:
    """A hurdle under each flight phase: where both feet are high, as tall as they clear minus a margin."""
    low = soles[:, :, 2].min(axis=1)
    air = low > min_clear
    out: list[dict] = []
    t = 0
    while t < len(air):
        if not air[t]:
            t += 1
            continue
        e = t
        while e + 1 < len(air) and air[e + 1]:
            e += 1
        if e - t >= 3:
            k = t + int(np.argmax(low[t:e + 1]))
            mid = soles[k].mean(axis=0)
            heading = root_xy[min(e + 1, len(root_xy) - 1)] - root_xy[max(t - 1, 0)]
            yaw = float(np.arctan2(heading[1], heading[0]))
            height = float(max(0.15, low[t:e + 1].max() - 0.12))
            out.append({"name": f"hurdle {len(out) + 1}", "shape": "box", "colour": "orange",
                        "rgba": [0.95, 0.55, 0.15, 1.0], "pos": [float(mid[0]), float(mid[1])], "yaw": yaw,
                        "size": [0.02, width / 2, height / 2]})
        t = e + 1
    return out


def smooth_snaps(model: mujoco.MjModel, q: np.ndarray, joint_step: float = 0.6, root_step: float = 0.12,
                 half: int = 5) -> tuple[np.ndarray, int]:
    """Blend across one-frame snaps. Where the robot cannot reach the performer's pose (joint limits in a deep
    crouch, for example) the solver flips between two compromise poses from one frame to the next. Each such
    frame gets a window of `half` frames either side replaced by an eased blend of the poses at its ends."""
    from ..imitate.motion import free_root, quat_slerp
    _, qa, _ = free_root(model)
    q = q.copy()
    rest = np.ones(q.shape[1], bool)
    rest[qa:qa + 7] = False
    count = 0
    for _ in range(6):
        dj = np.abs(np.diff(q[:, rest], axis=0)).max(axis=1)
        dr = np.linalg.norm(np.diff(q[:, qa:qa + 3], axis=0), axis=1)
        score = np.maximum(dj / joint_step, dr / root_step)
        k = int(score.argmax())
        if score[k] <= 1.0:
            break
        lo, hi = max(k - half + 1, 0), min(k + half, len(q) - 1)
        w = (np.arange(lo, hi + 1) - lo) / max(hi - lo, 1)
        w = 3 * w ** 2 - 2 * w ** 3
        a, b = q[lo].copy(), q[hi].copy()
        q[lo:hi + 1] = (1 - w[:, None]) * a + w[:, None] * b
        q[lo:hi + 1, qa + 3:qa + 7] = np.array([quat_slerp(a[qa + 3:qa + 7], b[qa + 3:qa + 7], float(x)) for x in w])
        count += 1
    return q, count


def clear_floor(model: mujoco.MjModel, q: np.ndarray, margin: float = 0.004) -> tuple[np.ndarray, int]:
    """Keep the arms out of the floor: where a hand or forearm dips below it (the robot's arms are not the
    performer's), move that arm's joints the least amount that lifts it clear, eased over neighbouring
    frames. Returns the corrected qpos trajectory and how many frames were changed."""
    from scipy.ndimage import gaussian_filter1d
    d = mujoco.MjData(model)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    arms = []
    for side in ("left", "right"):
        joints = np.array([j for j in range(model.njnt) if model.joint(j).name.startswith(side + "_")
                           and any(k in model.joint(j).name for k in ("shoulder", "elbow", "wrist"))])
        geoms = [g for g in range(model.ngeom)
                 if model.body(int(model.geom_bodyid[g])).name.startswith(side + "_")
                 and any(k in model.body(int(model.geom_bodyid[g])).name for k in ("arm", "hand", "finger"))]
        arms.append((geoms, model.jnt_dofadr[joints], model.jnt_qposadr[joints], joints))
    jac, fromto = np.zeros((3, model.nv)), np.zeros(6)

    def solve(qq: np.ndarray) -> np.ndarray:
        qq = qq.copy()
        for geoms, dofs, adr, joints in arms:
            lim = model.jnt_limited[joints].astype(bool)
            for _ in range(60):
                d.qpos[:] = qq
                mujoco.mj_kinematics(model, d)
                mujoco.mj_comPos(model, d)
                best, point, body = margin, None, -1
                for g in geoms:
                    dist = mujoco.mj_geomDistance(model, d, g, floor, 1.0, fromto)
                    if dist < best:
                        best, point, body = dist, fromto[:3].copy(), int(model.geom_bodyid[g])
                if point is None:
                    break
                mujoco.mj_jac(model, d, jac, None, point, body)
                row = jac[2, dofs]
                qq[adr] += row * (margin - best + 0.003) / (row @ row + 1e-3)
                qq[adr[lim]] = np.clip(qq[adr[lim]], model.jnt_range[joints[lim], 0], model.jnt_range[joints[lim], 1])
        return qq

    delta = np.array([solve(qq) for qq in q]) - q
    changed = int((np.abs(delta).max(axis=1) > 1e-6).sum())
    if not changed:
        return q.copy(), 0
    delta = gaussian_filter1d(delta, 2.0, axis=0, mode="nearest")        # ease in and out
    out = np.array([solve(qq) for qq in q + delta])                      # then make every frame exact
    return out, changed


def build_clip(cid: str, model: mujoco.MjModel | None = None, verbose: bool = True) -> dict:
    from ..imitate.bvh import Bvh, load_bvh
    from ..imitate.retarget import RetargetConfig, retarget_bvh
    spec = CLIPS[cid]
    model = model or default_model()
    bvh = load_bvh(fetch(spec["url"]))
    a, b = spec["frames"] or (1, bvh.n_frames - 1)       # frame 0 is the converter's T-pose
    bvh = Bvh(bvh.joints, 1.0 / spec["fps"], bvh.frames[max(a, 1):b + 1])
    opts = dict(spec["retarget"])
    ground_min = opts.pop("ground_min", False)
    if "weights" in opts:
        from ..imitate.retarget import DEFAULT_WEIGHTS
        opts["weights"] = {**DEFAULT_WEIGHTS, **opts["weights"]}
    # Solve at 100 Hz with extra iterations (fast acrobatics move far between frames, and a warm-started solver
    # that falls behind jumps between poses), then remove what single-frame jumps remain and halve to 50 Hz.
    cfg = RetargetConfig(**{"fps": 100.0, "iters": 60, **opts})
    res = retarget_bvh(bvh, model, cfg, name=cid)
    motion = res.motion
    from scipy.ndimage import gaussian_filter1d, median_filter
    jp = median_filter(motion.joint_pos, size=(7, 1), mode="nearest")
    motion.joint_pos[:] = gaussian_filter1d(jp, 1.5, axis=0, mode="nearest")
    motion = motion.resample(50.0)
    if spec["ground_all"]:   # hands touch the floor: put the lowest point of the whole body on it
        motion.root_pos[:, 2] -= float(np.percentile(lowest_body_height(model, motion), 2))
    if ground_min:           # mostly airborne or on a ledge: the lowest the soles ever get is the floor
        motion.root_pos[:, 2] -= float(sole_tracks(model, motion)[:, :, 2].min())
    from ..imitate.motion import Motion
    qs, snaps = smooth_snaps(model, motion.qpos(model))
    motion = Motion.from_qpos(model, qs, motion.fps, name=cid, meta=dict(motion.meta), keypoints=False)
    lift = np.maximum(0.0, -lowest_exact(model, motion))          # anything else under the floor: raise the body
    if lift.max() > 0:
        from scipy.ndimage import maximum_filter1d
        motion.root_pos[:, 2] += 1.08 * gaussian_filter1d(maximum_filter1d(lift, 13, mode="nearest"), 2.0, mode="nearest")
    qfix, fixed = clear_floor(model, motion.qpos(model))
    m2 = Motion.from_qpos(model, qfix, motion.fps, name=cid, keypoints=False)
    rest = np.maximum(0.0, -lowest_exact(model, m2, arms=True))   # an arm that cannot be bridged clear: lift
    if rest.max() > 0:
        from scipy.ndimage import maximum_filter1d
        free = model.jnt_qposadr[0]
        qfix[:, free + 2] += 1.08 * gaussian_filter1d(maximum_filter1d(rest, 13, mode="nearest"), 2.0, mode="nearest")
    motion = Motion.from_qpos(model, qfix, motion.fps, name=cid, meta=dict(motion.meta))   # also refreshes key points
    if spec["reverse"]:
        motion = Motion.from_qpos(model, motion.qpos(model)[::-1].copy(), motion.fps, name=cid).aligned()
    props: list[dict] = []
    soles = sole_tracks(model, motion)
    if spec["props"] == "boxes":
        props = find_boxes(soles, motion.fps)
    elif spec["props"] == "hurdle":
        props = find_hurdles(soles, motion.root_pos[:, :2])
    DATA.mkdir(parents=True, exist_ok=True)
    motion.save(DATA / f"{cid}.npz")
    entry = {"file": f"{cid}.npz", "title": spec["title"], "group": spec["group"], "description": spec["description"],
             "source": CMU.format(s=spec["subject"], t=spec["trial"]), "url": spec["url"], "frames": spec["frames"],
             "duration_s": round(float(motion.duration), 2), "speed": spec["speed"], "props": props,
             "retarget_error_m": round(float(res.errors.get("mean", 0.0)), 4), "arm_frames_adjusted": fixed, "snaps_blended": snaps}
    if verbose:
        print(f"{cid:16s} {motion.n_frames:4d} frames  {motion.duration:5.2f} s  mean error "
              f"{entry['retarget_error_m'] * 100:.1f} cm  props {len(props)}", flush=True)
    return entry


def main(argv: list[str] | None = None) -> None:
    ids = list(argv if argv is not None else sys.argv[1:]) or list(CLIPS)
    path = DATA / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    model = default_model()
    for cid in ids:
        manifest[cid] = build_clip(cid, model)
    manifest = {k: manifest[k] for k in CLIPS if k in manifest}
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
