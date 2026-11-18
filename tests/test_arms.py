"""Hands, props and arm skills (control/arms.py, sim/props.py, the hand part of models/humanoid.py).

Acceptance: the robot stands with its legs held by ManualController at ctrl0 (and, in one
test, by WalkController when it is available); reach accuracy <= 2 cm for 5 targets per arm;
an unreachable target fails cleanly; a cube is picked off the table, lifted 10 cm, held 3 s
and released by friction alone (grasp_assist=False), and once more with the grasp assist;
the robot never falls. Run with -s to see the measured numbers.
"""
import mujoco
import numpy as np
import pytest

from simulatics.control import ManualController
from simulatics.control.arms import ArmController, min_jerk
from simulatics.models import HumanoidParams, build_mjcf
from simulatics.sim.props import manipulation_scene, prop_names, props_state, table_geometry

ARM_ACTS = {f"{s}_{j}" for s in ("left", "right")
            for j in ("shoulder_y", "shoulder_x", "elbow", "wrist_y", "wrist_x", "gripper")}


class Sim:
    """Legs held by a base controller, arms by ArmController; tracks whether the robot fell."""

    def __init__(self, obstacles=None, props=None, legs="manual", **arm_kw):
        self.m = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams(), obstacles, props))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        if legs == "walker":
            from simulatics.control.walker import WalkController
            self.legs = WalkController(self.m)
            self.legs.reset(self.m, self.d)
        else:
            self.legs = ManualController(self.m, np.zeros(self.m.nu))
        self.arms = ArmController(self.m, **arm_kw)
        self.arms.reset(self.m, self.d)
        self.n = max(1, int(round(self.legs.control_dt / self.m.opt.timestep)))
        self.arm_every = max(1, int(round(self.arms.control_dt / self.legs.control_dt)))
        self.z0 = float(self.d.qpos[2])
        self.min_z, self.min_up = self.z0, 1.0
        self.k = 0

    def run(self, seconds: float, until_idle: str | None = None) -> None:
        """Advance; with until_idle="left"/"right" stop early once that arm is no longer moving."""
        for _ in range(int(round(seconds / self.legs.control_dt))):
            self.legs.step(self.m, self.d)
            if self.k % self.arm_every == 0:
                self.arms.step(self.m, self.d)
            mujoco.mj_step(self.m, self.d, self.n)
            self.k += 1
            q = self.d.qpos
            self.min_z = min(self.min_z, float(q[2]))
            self.min_up = min(self.min_up, float(1 - 2 * (q[4] ** 2 + q[5] ** 2)))
            if until_idle and self.k % self.arm_every == 0 and self.arms.status()[until_idle]["state"] != "moving":
                return

    def upright(self) -> bool:
        return self.min_z > 0.95 * self.z0 and self.min_up > 0.95


# ---- model -------------------------------------------------------------------------------
def test_hands_off_is_the_old_robot_and_legacy_params_load_without_hands():
    old = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams(hands=False)))
    assert (old.nu, old.nq, old.neq) == (20, 27, 0)
    xml = build_mjcf(HumanoidParams(hands=False))
    assert "wrist" not in xml and "gripper" not in xml and "<equality>" not in xml and "<custom>" not in xml
    # a parameter dict saved before hands existed (e.g. runs/pretrained_walker) keeps 20 actuators
    legacy = {k: v for k, v in HumanoidParams().to_dict().items() if k != "hands"}
    assert HumanoidParams.from_dict(legacy).hands is False
    assert HumanoidParams.from_dict({"thigh_length": 0.5}).hands is True
    new = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams()))
    assert new.nu == 26 and new.neq == 2
    # existing actuator indices are unchanged; the hand actuators are appended
    assert [new.actuator(i).name for i in range(20)] == [old.actuator(i).name for i in range(20)]
    assert {new.actuator(i).name for i in range(20, 26)} <= ARM_ACTS
    assert new.body_subtreemass[new.body("left_hand").id] < 0.25          # light hands
    for s in ("left", "right"):
        assert new.site(f"{s}_hand").id >= 0 and new.geom(f"{s}_finger1").condim == 4


def test_props_scene_builds_after_the_robot_and_collides_with_fingers():
    obstacles, props = manipulation_scene(seed=3)
    assert 2 <= len(props) <= 4 and obstacles[0]["name"] == "table"
    t = table_geometry()
    assert 0.8 < t["height"] < 1.1 and obstacles[0]["pos"][0] - obstacles[0]["size"][0] >= 0.25
    names = [p["name"] for p in props]
    assert len(set(names)) == len(names) and all(n.split()[-1] in ("cube", "ball") for n in names)
    for p in props:
        size = 2 * p["size"][0]
        assert 0.04 <= size <= 0.06 and abs(p["pos"][2] - t["height"] - p["size"][0]) < 0.01
    plain = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams(), obstacles))
    m = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams(), obstacles, props))
    assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and m.joint(0).name == "root"
    # the robot's qpos layout is unchanged; props append 7 qpos each
    assert m.nq == plain.nq + 7 * len(props)
    assert all(m.joint(i).name == plain.joint(i).name for i in range(plain.njnt))
    assert sorted(prop_names(m).values()) == sorted(names)
    for i in range(len(props)):
        g = m.geom(f"prop_{i}")
        finger = m.geom("left_finger1")
        assert g.group == 1 and (finger.contype & g.conaffinity)
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, 400)  # 2 s: props settle on the table, the robot keeps standing
    for s in props_state(m, d):
        assert abs(s["z"] - t["height"] - s["size_m"] / 2) < 0.01
    assert d.qpos[2] > 0.95 * HumanoidParams().stand_height


def test_min_jerk_profile():
    assert min_jerk(0) == 0 and min_jerk(1) == 1 and abs(min_jerk(0.5) - 0.5) < 1e-12
    assert min_jerk(-1) == 0 and min_jerk(2) == 1


# ---- reaching --------------------------------------------------------------------------
REACH = [[0.30, 0.25, 1.05], [0.35, 0.15, 1.20], [0.25, 0.35, 0.95], [0.40, 0.20, 1.35], [0.45, 0.30, 1.15]]


def test_reach_accuracy_five_targets_per_arm():
    sim = Sim()
    sim.run(0.3)
    errors = {}
    for side, sign in (("left", 1), ("right", -1)):
        for x, y, z in REACH:
            target = [x, sign * y, z]
            msg = sim.arms.command(side, "reach", target=target)
            assert "reaching" in msg, msg
            sim.run(8.0, until_idle=side)
            sim.run(0.5)  # settled error, not the first sample under the 5 mm "done" threshold
            st = sim.arms.status()[side]
            err = float(np.linalg.norm(np.array(st["hand_xyz"]) - target))
            errors[(side, tuple(target))] = err
            assert st["state"] == "done", st
            assert err <= 0.02, (side, target, err)
    print(f"\nreach error over {len(errors)} targets: mean {np.mean(list(errors.values())) * 1000:.1f} mm, "
          f"max {max(errors.values()) * 1000:.1f} mm; sim {sim.d.time:.1f} s")
    assert sim.upright()


def test_unreachable_targets_fail_cleanly_and_the_arm_stays_put():
    sim = Sim()
    sim.run(0.3)
    before = np.array(sim.arms.status()["left"]["hand_xyz"])
    msg = sim.arms.command("left", "reach", target=[1.5, 0.2, 1.0])
    st = sim.arms.status()["left"]
    assert st["state"] == "failed" and "out of reach" in msg and "walk closer" in msg
    msg = sim.arms.command("left", "reach", target=[0.30, -0.45, 1.0])   # far across the body: joint limits
    assert sim.arms.status()["left"]["state"] == "failed" and "out of reach" in msg
    msg = sim.arms.command("left", "point_at", target=[-2.0, 0.5, 1.0])   # behind the robot
    assert sim.arms.status()["left"]["state"] == "failed" and "turn towards it" in msg
    assert "unknown arm skill" in sim.arms.command("left", "juggle")
    assert "no arm called" in sim.arms.command("middle", "home")
    assert "bad arguments" in sim.arms.command("left", "reach")
    sim.run(1.0)
    assert np.linalg.norm(np.array(sim.arms.status()["left"]["hand_xyz"]) - before) < 0.01
    assert sim.upright()


def test_arm_controller_writes_only_arm_and_hand_actuators():
    sim = Sim()
    sim.arms.command("right", "reach", target=[0.3, -0.25, 1.05])
    sim.d.ctrl[:] = 0.123
    sim.arms.step(sim.m, sim.d)
    right_arm = {a for a in ARM_ACTS if a.startswith("right_")}
    names = [sim.m.actuator(i).name for i in range(sim.m.nu)]
    # legs, waist and the idle (never commanded) left arm keep whatever the base controller wrote
    assert all(sim.d.ctrl[i] == 0.123 for i, n in enumerate(names) if n not in right_arm)
    assert all(sim.d.ctrl[i] != 0.123 for i, n in enumerate(names) if n in right_arm)


# ---- grasping ----------------------------------------------------------------------------
def _pick_cube(sim: Sim, side: str | None = None):
    """Reach a cube from above, grasp, lift 10 cm, hold 3 s, release. Returns measurements."""
    cubes = [p for p in props_state(sim.m, sim.d) if p["shape"] == "cube"]
    cube = cubes[0] if side is None else next(p for p in cubes if (p["y"] > 0) == (side == "left"))
    side = side or ("left" if cube["y"] > 0 else "right")
    b = sim.m.body(cube["body"]).id
    start = sim.d.xpos[b].copy()
    out = {"name": cube["name"], "side": side}
    msg = sim.arms.command(side, "reach", target=start, approach=[0, 0, -1])
    assert "reaching" in msg, msg
    sim.run(10.0, until_idle=side)
    st = sim.arms.status()[side]
    assert st["state"] == "done", st
    out["reach_error_m"] = st["error_m"]
    assert np.linalg.norm(sim.d.xpos[b] - start) < 0.01          # came down around it without pushing it
    sim.arms.command(side, "grasp")
    sim.run(3.0, until_idle=side)
    st = sim.arms.status()[side]
    assert st["state"] == "holding" and st["holding"] == cube["name"], st
    z_grasp = float(sim.d.xpos[b][2])
    sim.arms.command(side, "reach", target=np.array(st["hand_xyz"]) + [0, 0, 0.10])
    sim.run(6.0, until_idle=side)
    assert sim.arms.status()[side]["state"] == "holding"
    rel0 = sim.d.xpos[b] - sim.arms.arms[side].hand_xyz
    sim.run(3.0)
    st = sim.arms.status()[side]
    assert st["state"] == "holding" and st["holding"] == cube["name"], st
    out["lift_m"] = float(sim.d.xpos[b][2] - start[2])
    out["lift_since_grasp_m"] = float(sim.d.xpos[b][2] - z_grasp)
    out["slip_3s_m"] = float(np.linalg.norm(sim.d.xpos[b] - sim.arms.arms[side].hand_xyz - rel0))
    out["weld_during_hold"] = sim.arms.arms[side].weld_on >= 0
    msg = sim.arms.command(side, "release")
    assert "releasing " + cube["name"] in msg
    sim.run(2.0)
    st = sim.arms.status()[side]
    assert st["state"] == "done" and st["holding"] is None
    out["dropped_to_z"] = float(sim.d.xpos[b][2])
    sim.arms.command(side, "home")
    sim.run(8.0, until_idle=side)
    assert sim.arms.status()[side]["state"] == "done"
    return out


@pytest.mark.parametrize("assist", [False, True], ids=["friction", "grasp-assist"])
def test_pick_lift_hold_release(assist):
    obstacles, props = manipulation_scene(seed=0)
    sim = Sim(obstacles, props, grasp_assist=assist)
    sim.run(0.5)
    r = _pick_cube(sim)
    top = table_geometry()["height"]
    print(f"\n[{'assist' if assist else 'friction'}] {r['name']} ({r['side']}): reach err "
          f"{r['reach_error_m'] * 1000:.1f} mm, lift {r['lift_m'] * 100:.1f} cm, slip in 3 s hold "
          f"{r['slip_3s_m'] * 1000:.1f} mm, released onto z {r['dropped_to_z']:.3f} (table {top:.3f}); "
          f"torso min z {sim.min_z:.3f} / {sim.z0:.3f}")
    assert r["lift_m"] >= 0.09 and r["lift_since_grasp_m"] >= 0.09
    assert r["slip_3s_m"] < 0.005
    assert abs(r["dropped_to_z"] - top) < 0.05                   # back on the table
    assert r["weld_during_hold"] == assist
    assert sim.d.eq_active[2:].sum() == 0                        # no weld left on after release
    assert sim.upright()


def test_grasp_on_nothing_and_gestures():
    sim = Sim()
    sim.run(0.3)
    sim.arms.command("left", "grasp")
    sim.run(3.0, until_idle="left")
    st = sim.arms.status()["left"]
    assert st["state"] == "failed" and "nothing" in st["message"]
    for side, skill, kw in [("left", "wave", {}), ("right", "raise", {}), ("both", "home", {}),
                            ("right", "point_at", {"target": [2.0, -1.0, 1.0]})]:
        assert "unknown" not in sim.arms.command(side, skill, **kw)
        sim.run(8.0, until_idle="left" if side == "left" else "right")
        sim.run(0.2)
        for s in (("left", "right") if side == "both" else (side,)):
            assert sim.arms.status()[s]["state"] == "done", (skill, sim.arms.status()[s])
    pointing = sim.arms.status()["right"]
    assert pointing["error_m"] < 0.05                             # pointing line passes within 5 cm
    sim.arms.command("right", "relax")
    assert sim.arms.status()["right"]["state"] == "idle"
    assert sim.upright()


def test_pick_with_the_walking_controller_balancing():
    try:
        from simulatics.control.walker import WalkController
    except ImportError:
        pytest.skip("control/walker.py not available")
    obstacles, props = manipulation_scene(seed=2)
    probe = mujoco.MjModel.from_xml_string(build_mjcf(HumanoidParams(), obstacles, props))
    if not WalkController.supports(probe):
        pytest.skip("WalkController does not support this model")
    sim = Sim(obstacles, props, legs="walker")
    sim.run(1.0)
    r = _pick_cube(sim)
    print(f"\n[walker] {r['name']} ({r['side']}): reach err {r['reach_error_m'] * 1000:.1f} mm, "
          f"lift {r['lift_m'] * 100:.1f} cm, slip {r['slip_3s_m'] * 1000:.1f} mm; torso min z {sim.min_z:.3f}")
    assert r["lift_m"] >= 0.09 and r["slip_3s_m"] < 0.005
    assert sim.upright()


# ---- regressions found in review ---------------------------------------------------------
def test_non_finite_numbers_are_rejected_and_never_reach_the_motors():
    """A NaN/inf target used to be accepted by point_at: the NaN went through the IK into
    data.ctrl, MuJoCo then zeroes EVERY actuator (legs too) on each step, and the arm stayed
    poisoned for all later commands."""
    sim = Sim()
    sim.run(0.3)
    bad = [("point_at", {"target": [np.inf, 0, 1]}), ("point_at", {"target": [np.nan, 0, 1]}),
           ("reach", {"target": [0.3, 0.25, 1.05], "approach": [np.inf, 0, -1]}),
           ("reach", {"target": [0.3, 0.25, 1.05], "duration": float("inf")}),
           ("reach", {"target": [0.3, 0.25, 1.05], "duration": float("nan")}),
           ("reach", {"target": [0.3, 0.25, 1.05], "duration": 0})]
    for skill, kw in bad:
        msg = sim.arms.command("left", skill, **kw)
        assert "bad arguments" in msg, (skill, kw, msg)
        sim.run(0.2)
        assert np.isfinite(sim.d.ctrl).all(), (skill, kw)
    # huge but finite numbers are fine (the direction is pre-scaled before normalising)
    assert "pointing at" in sim.arms.command("left", "point_at", target=[1e9, 0, 1])
    assert "reaching" in sim.arms.command("left", "reach", target=[0.3, 0.25, 1.05], approach=[1e300, 0, -1e300])
    sim.run(0.5)
    assert np.isfinite(sim.d.ctrl).all() and np.isfinite(sim.arms.arms["left"].q_des).all()
    # and a solver that returns garbage can never write it to the motors
    arm = sim.arms.arms["left"]
    sim.arms._ik = lambda *a, **k: (np.full(len(arm.names), np.nan), 0.0, 0.0)
    q_before = arm.q_des.copy()
    sim.run(0.2)
    assert np.isfinite(sim.d.ctrl).all() and np.allclose(arm.q_des, q_before)
    assert sim.upright()


def test_ik_clamping_loop_with_several_joints_on_their_limits():
    """With three joints on limits the old fixed 3-pass clamping loop left Jinv/Jf one free-set
    behind, and the null-space projector crashed step() with a shape error (it stopped the sim)."""
    sim = Sim()
    sim.run(0.2)
    A = sim.arms
    rng = np.random.default_rng(0)
    for side in ("left", "right"):
        arm = A.arms[side]
        A._load_base(sim.d)
        sh = sim.d.xanchor[arm.shoulder_jnt].copy()
        for _ in range(400):
            q0 = np.where(rng.random(len(arm.names)) < 0.5, arm.lo, arm.hi)
            a = rng.normal(size=3) if rng.random() < 0.5 else None
            q, e, ang = A._ik(arm, q0, sh + 0.5 * rng.normal(size=3), None if a is None else a / np.linalg.norm(a), 3)
            assert np.isfinite(q).all() and np.isfinite(e) and np.all(q >= arm.lo - 1e-9) and np.all(q <= arm.hi + 1e-9)


def test_stepping_twice_at_the_same_time_moves_nothing():
    sim = Sim()
    sim.run(0.3)
    sim.arms.command("left", "reach", target=[0.30, 0.25, 1.05])
    sim.run(0.4)
    arm = sim.arms.arms["left"]
    sim.arms.step(sim.m, sim.d)
    q = arm.q_des.copy()
    sim.arms.step(sim.m, sim.d)       # same data.time: dt = 0 (used to move 4 mrad and feed forward 4 rad/s)
    u = sim.d.ctrl[arm.act].copy()
    for _ in range(5):
        sim.arms.step(sim.m, sim.d)
    assert np.array_equal(arm.q_des, q) and np.array_equal(sim.d.ctrl[arm.act], u)


def test_reach_from_the_hanging_arm_does_not_sweep_the_forearm_through_the_table_edge():
    """From the hanging pose the hand cannot follow a straight line up and forward (the shoulder
    would have to swing ~17 deg back, the balance limit is 5): it sags below the line, so the
    old ray test of the straight hand path said 'clear' while the forearm clipped the table edge
    and shoved the cube ~6 cm (seed 26, the orange cube after picking the purple one)."""
    obstacles, props = manipulation_scene(seed=26)
    sim = Sim(obstacles, props)
    sim.run(0.5)
    cube = {p["name"]: p for p in props_state(sim.m, sim.d)}["orange cube"]
    b = sim.m.body(cube["body"]).id
    start = sim.d.xpos[b].copy()
    arm = sim.arms.arms["right"]
    pre = start + [0, 0, 0.08]
    assert not sim.arms._blocked(arm, arm.hand_xyz, pre)              # the hand line itself is clear...
    assert sim.arms._sweep_hits(arm, arm.hand_xyz, pre, np.array([0, 0, -1.0]))   # ...the arm is not
    first = next(p for p in props_state(sim.m, sim.d) if p["name"] == "purple cube")
    assert _pick_lite(sim, first["body"], "left")
    start = sim.d.xpos[b].copy()
    msg = sim.arms.command("right", "reach", target=start, approach=[0, 0, -1])
    assert "tuck" in msg, msg
    sim.run(10.0, until_idle="right")
    assert sim.arms.status()["right"]["state"] == "done"
    assert np.linalg.norm(sim.d.xpos[b] - start) < 0.005, sim.d.xpos[b] - start
    assert sim.upright()


def _pick_lite(sim: Sim, body: str, side: str) -> bool:
    """Reach-grasp-lift-release-home without the measurements of _pick_cube."""
    b = sim.m.body(body).id
    sim.arms.command(side, "reach", target=sim.d.xpos[b].copy(), approach=[0, 0, -1])
    sim.run(10.0, until_idle=side)
    sim.arms.command(side, "grasp")
    sim.run(3.0, until_idle=side)
    ok = sim.arms.status()[side]["state"] == "holding"
    sim.arms.command(side, "reach", target=np.array(sim.arms.status()[side]["hand_xyz"]) + [0, 0, 0.10])
    sim.run(6.0, until_idle=side)
    sim.arms.command(side, "release")
    sim.run(1.5)
    sim.arms.command(side, "home")
    sim.run(8.0, until_idle=side)
    return ok and sim.arms.status()[side]["state"] == "done"
