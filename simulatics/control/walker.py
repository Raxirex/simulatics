"""Model-based walking controller for accurate click-to-walk.

    path -> footsteps -> ZMP preview control (CoM) -> leg IK -> position servos + inverse-dynamics feed-forward

Click-to-walk has to stop *on* the clicked point, so this controller plans
explicit footsteps and walks them with a classic model-based pipeline instead
of a learned velocity policy. Every number it needs (link lengths, mass, CoM
height, servo gains, torque limits) is read from the compiled MuJoCo model, so
any body the parametric generator makes walks without hand tuning.

1. **Footsteps** (control/walk_footsteps.py): along the planner's path with
   step-length / width / turn limits, turning on the spot at sharp corners,
   ending with both feet side by side on the goal at the requested yaw.
2. **Centre of mass** (control/walk_preview.py): the ZMP reference stays on
   the stance sole and ramps between soles in double support; the CoM
   trajectory comes from ZMP preview control of the cart-table model
   (Kajita et al., "Biped walking pattern generation by using preview control
   of zero-moment point", ICRA 2003).
3. **Whole-body placement + leg IK** (control/walk_ik.py): the torso is placed
   so that the forward-kinematics CoM of the full model (arms included,
   wherever another controller holds them) is on the reference CoM; the legs
   come from closed-form 6-DOF IK. The swing leg is solved against the
   low-passed *measured* pelvis pose, so it lands where it was planned in the
   world even when the body lags or leans.
4. **Servo feed-forward = floating-base inverse dynamics.** Every actuator is a
   position servo, torque = kp (ctrl - q) - kv q'. The controller sends
   ctrl = q_des + (kv/kp) q'_des + tau/kp with tau from
   M qdd + h = S^T tau + J_c^T f: the six unactuated rows give the contact
   wrench f for the reference motion (split between the feet in double
   support), the actuated rows give tau. Soft servos then track like
   torque-controlled joints. Only the ankle torques are limited to what the
   sole can carry (centre of pressure inside the foot).
5. **Balance feedback.** The divergent component of motion (DCM / capture
   point, xi = c + c'/omega; Englsberger, Ott, Albu-Schaeffer, IEEE T-RO 2015)
   is measured from MuJoCo's whole-body CoM. Its error moves the centre of
   pressure (ankle strategy, with the matching LIPM force change) and, when it
   is large, the landing point of the swing foot (capture-point stepping; Pratt
   et al., Humanoids 2006). Torso pitch/roll errors are fed back through the
   stance hip(s) (the hip strategy of position-controlled humanoids).
6. **Closed-loop arrival.** Stance feet are tracked where they really are and
   the ZMP reference is rebuilt from where each foot landed (the remaining
   footsteps keep their world positions, so the closing pair still lands on the
   goal); after the final stance the controller measures the torso error and
   re-plans correction steps from the landed feet if it is above tolerance
   (2 cm / 2 deg by default).
7. **Gait sized to the motors.** At construction the controller predicts the
   torque use of its nominal gait from the model (static single-support
   inverse dynamics + the torque to swing the leg's inertia through a stride).
   Heavy bodies whose motors would run above ~55 % get proportionally shorter
   and slower steps instead of saturating (gait_scale, static_load,
   swing_load).
8. **Push recovery** (control/walk_recovery.py; WalkParams.recovery, on by
   default). When the DCM leaves the walking pattern by more than
   recovery_trigger, the planned footsteps are dropped and the controller
   balances from the measured state: every other tick it searches touchdown
   times and landing points (step timing + location adaptation from the DCM,
   after Khadiv et al., T-RO 2020 and Griffin et al., IROS 2017), takes an
   emergency step from double support when the DCM leaves the support
   polygon, and uses a quick crossover-free two-step answer to sideways
   pushes towards the stance foot. The CoM reference then follows the
   pendulum with the centre of pressure the feet can really apply (on the
   support polygon, as close to the DCM as useful), so the servos stop
   pulling the body back onto the old pattern; the DCM used for the decisions
   includes the centroidal angular momentum the joints will absorb. Once the
   robot is quiet in double support it re-plans the walk from where its feet
   are (Urata et al., IROS 2012). recovery=False keeps the fixed-timing walker
   exactly (bit for bit).
9. **Terrain** (control/walk_terrain.py; WalkParams.terrain="auto": on when the
   model has static geoms lower than 0.5 m). Footholds are checked with
   downward mj_ray casts against the world (groups 1 and 2, never the robot):
   a plane fit over the sole gives the height and tilt (ramps), edges are
   rejected and planned footsteps are moved fully onto or off a surface; the
   swing lifts above everything along its line before it moves; the torso
   height follows a smoothed support height under the ZMP reference (the
   pendulum stays planar per step); the knees bend more and single support is
   slower. On models without terrain none of this runs.

Arm and hand actuators are never written: another controller owns them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .walk_footsteps import (LEFT, RIGHT, StepLimits, goal_heading, mean_yaw, plan_footsteps, square_up,
                             velocity_footsteps, wrap)
from .walk_ik import ABDOMEN, LEG_JOINTS, SIDES, RobotGeometry, leg_ik, model_supported, rot_z, yaw_of
from .walk_preview import G, ZmpPreview
from .walk_recovery import CaptureLimits, clamp_to_sole, clamp_to_support, plan_double_support, plan_swing
from .walk_terrain import TerrainLimits, TerrainMap, TerrainReach, repair_footsteps, tilted

SIDE_NAME = {LEFT: "left", RIGHT: "right"}


@dataclass
class WalkParams:
    """Tunables. "x leg" = multiplied by the model's thigh + shin length; "x tc" = multiplied by the pendulum
    time constant sqrt(z_c / g) of the model. The defaults work for every body tested (see tests)."""
    hip_height: float = 0.93          # nominal hip-ankle distance, fraction of the leg length (knees bent)
    step_forward: float = 0.22        # x leg, max sole-to-sole advance per footstep
    step_backward: float = 0.12       # x leg
    step_lateral: float = 0.09        # x leg
    turn_deg: float = 20.0            # max yaw change per footstep
    min_width_frac: float = 0.65      # closest lateral foot spacing, fraction of the hip width
    swing_height: float = 0.075       # x leg
    t_single: float = 1.6             # x tc, single support
    t_double: float = 0.45            # x tc, double support between steps
    t_start: float = 2.0              # x tc, first weight shift from standing
    t_final: float = 1.5              # x tc, final weight shift back onto both feet
    preview_s: float = 1.6            # s, ZMP preview horizon
    k_dcm: float = 1.5                # DCM error -> centre-of-pressure shift
    k_torso: float = 1.0              # torso pitch/roll error -> stance hip offset (rad / rad)
    k_step: float = 0.5               # predicted DCM error at touchdown -> landing point shift
    step_predict: bool = True         # predict the DCM error at touchdown (x exp(omega t_remaining), <= 3)
    step_deadband: float = 0.02       # m, nominal steps stay exactly on the plan below this
    step_adjust: float = 0.15         # x leg, max landing point shift
    swing_world: float = 1.0          # 0..1, how much of the measured pelvis error the swing leg corrects
    swing_world_tau: float = 0.08     # s, low-pass on that pelvis error
    vel_filter_tau: float = 0.02      # s, low-pass on the measured CoM velocity
    track_stance: bool = True         # position targets use the measured stance-foot poses
    load_target: float = 0.55         # predicted static + swing torque fraction above which steps get shorter
                                      # and slower (scale = load_target / load, >= 0.6); 0 = never adapt
    goal_tolerance: float = 0.02      # m, take correction steps when the settled torso error is larger
    yaw_tolerance_deg: float = 2.0
    max_corrections: int = 2
    control_hz: float = 200.0
    # push recovery (control/walk_recovery.py); recovery=False keeps the fixed-timing walker of before
    recovery: bool = True             # step timing / location adaptation and emergency steps after a push
    recovery_trigger: float = 0.0375  # x leg (3 cm on the default body), DCM error vs. the walking pattern
                                      # that starts a recovery (nominal walking stays below ~2 % of the leg)
    recovery_trigger_forward: float = 0.0375  # x leg, the same along the heading (the sole is longer than wide)
    recovery_gain: float = 4.0        # DCM -> centre of pressure gain while balancing on both feet
    recovery_track: float = 0.02      # 0..1 per tick: how fast the recovery CoM reference follows the body
    recovery_t_min: float = 0.8       # x tc, shortest recovery swing
    recovery_reach: float = 0.40      # x leg, forward reach of a recovery step (backward: 0.7 of it)
    recovery_width: float = 0.36      # x leg, how much wider than the nominal stance a recovery step may land
    recovery_cop: float = 0.85        # fraction of the sole the commanded centre of pressure may use
    recovery_foot_speed: float = 2.2  # x leg per s, mean horizontal swing-foot speed of a recovery step
    recovery_settle_s: float = 0.3    # s of quiet double support before walking on
    recovery_timeout_s: float = 8.0
    recovery_hold: float = 0.85       # double support holds (no step) while the DCM gauge of the support is below
    # experimental (tried, not better on the push benchmark; kept off):
    recovery_rate: bool = False       # predict with the measured DCM rate when it diverges faster than the model
    recovery_depth: float = 0.0       # planner cost per gauge^2 of the captured DCM (deeper capture)
    recovery_min_swing: float = 0.0   # fraction of the shortest swing a recovery step keeps from lift-off
    recovery_angmom: float = 1.0      # 0..1: include the centroidal angular momentum in the recovery DCM
    # rough terrain (control/walk_terrain.py); "auto" = on when the model has low static geoms (<= 0.5 m)
    terrain: object = "auto"          # "auto" | True | False
    terrain_hip_height: float = 0.88  # like hip_height, on terrain (more knee bend: reach for step-ups)
    terrain_t_single: float = 1.9     # x tc, single support on terrain (the swing lifts first, then moves)
    terrain_step_forward: float = 0.36  # x leg, sole advance allowed for a step that changes height
    terrain_clearance: float = 0.04   # m, swing sole above the highest ground along the swing
    terrain_smooth_s: float = 0.25    # s, half window of the torso height filter over the support height
    terrain_recovery_trigger: float = 0.1125  # x leg, recovery_trigger on terrain (9 cm on the default body: the
                                              # planar pendulum is a rougher model with height changes)
    terrain_margin: float = 0.025     # m, footholds are checked this much larger than the sole (landing errors)


@dataclass
class _Swing:
    side: int
    start: np.ndarray                 # (x, y, yaw) sole pose at lift-off
    target: np.ndarray                # (x, y, yaw) landing pose (plan + balance adjustment)
    n: int                            # duration in ticks
    k: int = 0
    plan: np.ndarray | None = None    # (x, y, yaw) landing pose from the footstep plan
    # recovery ("receding") mode: the sole reference is a quintic from its current state to the target that is
    # re-solved every tick, so the landing point and the touchdown time can change without a jump
    rec: bool = False
    s: np.ndarray | None = None       # (3, 4): rows position / velocity / acceleration of (x, y, z, yaw)
    zprof: tuple | None = None        # z profile (k0, z0, v0, a0, z_end, ticks, bump) from tick k0
    # terrain: ground (height, surface normal) under the sole at lift-off / landing, apex height of the swing
    ground0: tuple | None = None
    ground1: tuple | None = None
    apex: float | None = None

    def __post_init__(self):
        if self.plan is None:
            self.plan = self.target.copy()


@dataclass
class _Recovery:
    c: np.ndarray                     # CoM reference (x, y) and its velocity
    v: np.ndarray
    p: np.ndarray                     # commanded centre of pressure
    ticks: int = 0
    settle: int = 0
    next: object = None               # StepChoice of the other foot to take after the current swing lands
    sched: object = None              # StepChoice scheduled from double support
    lift_left: int = 0                # ticks of double support before `sched` lifts off
    lift_total: int = 1
    steps: int = 0                    # footsteps this recovery re-timed, moved or inserted
    counted: bool = False             # the current swing is already counted
    xi_prev: np.ndarray | None = None
    xi_rate: np.ndarray | None = None  # low-passed measured DCM velocity


@dataclass
class _Meas:
    com: np.ndarray
    com_vel: np.ndarray               # low-passed
    R_torso: np.ndarray
    p_pelvis: np.ndarray
    R_pelvis: np.ndarray
    feet: dict = field(default_factory=dict)     # side -> (x, y, yaw) of the sole centre
    fallen: bool = False
    com_vel_raw: np.ndarray | None = None
    angmom: np.ndarray | None = None  # whole-body angular momentum about the CoM (world frame)


def _quintic_next(s: np.ndarray, target: np.ndarray, T: float, dt: float) -> np.ndarray:
    """State (3, d) [pos; vel; acc] one step dt along the quintic from s to (target, 0, 0) in time T."""
    if T <= dt + 1e-9:
        return np.vstack([target, np.zeros_like(target), np.zeros_like(target)])
    p0, v0, a0 = s
    h = target - p0
    c3 = (20 * h - 12 * v0 * T - 3 * a0 * T * T) / (2 * T ** 3)
    c4 = (-30 * h + 16 * v0 * T + 3 * a0 * T * T) / (2 * T ** 4)
    c5 = (12 * h - 6 * v0 * T - a0 * T * T) / (2 * T ** 5)
    t = dt
    p = p0 + v0 * t + a0 / 2 * t * t + c3 * t ** 3 + c4 * t ** 4 + c5 * t ** 5
    v = v0 + a0 * t + 3 * c3 * t * t + 4 * c4 * t ** 3 + 5 * c5 * t ** 4
    a = a0 + 6 * c3 * t + 12 * c4 * t * t + 20 * c5 * t ** 3
    return np.vstack([p, v, a])


def _quintic_eval(p0, v0, a0, p1, T: float, t: float) -> tuple[float, float, float]:
    if T <= 1e-9:
        return p1, 0.0, 0.0
    h = p1 - p0
    c3 = (20 * h - 12 * v0 * T - 3 * a0 * T * T) / (2 * T ** 3)
    c4 = (-30 * h + 16 * v0 * T + 3 * a0 * T * T) / (2 * T ** 4)
    c5 = (12 * h - 6 * v0 * T - a0 * T * T) / (2 * T ** 5)
    p = p0 + v0 * t + a0 / 2 * t * t + c3 * t ** 3 + c4 * t ** 4 + c5 * t ** 5
    v = v0 + a0 * t + 3 * c3 * t * t + 4 * c4 * t ** 3 + 5 * c5 * t ** 4
    a = a0 + 6 * c3 * t + 12 * c4 * t * t + 20 * c5 * t ** 3
    return p, v, a


class WalkController:
    """Footstep / ZMP-preview / inverse-dynamics walker for the built-in humanoid (see module docstring)."""
    name = "walker"

    def __init__(self, model: mujoco.MjModel, params: WalkParams | None = None):
        if not model_supported(model):
            raise ValueError("WalkController needs the built-in humanoid's legs (see walk_ik.model_supported)")
        self.p = params or WalkParams()
        self.model = model
        dt = float(model.opt.timestep)
        self.substeps = max(1, int(round(1.0 / (self.p.control_hz * dt))))
        self.control_dt = self.substeps * dt
        self.geo = RobotGeometry.from_model(model)
        self.mass = self.geo.mass
        self.torso_id = model.body("torso").id
        self.pelvis_id = model.body("pelvis").id
        self.foot_body = {s: model.body(f"{SIDE_NAME[s]}_foot").id for s in (LEFT, RIGHT)}
        # driven joints / actuators: abdomen (2) + left leg (6) + right leg (6), in that order
        names = list(ABDOMEN) + [f"{s}_{j}" for s in SIDES for j in LEG_JOINTS]
        jid = [model.joint(n).id for n in names]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jid])
        self.act = np.array([model.actuator(n).id for n in names])
        self.scratch = mujoco.MjData(model)

        leg = self.geo.leg_length
        self.width = self.geo.hip_width
        self.terrain: TerrainMap | None = None
        self._dz = 0.0
        if self.p.terrain is not False:
            lg0 = self.geo.legs["left"]
            tm = TerrainMap(model, lg0.half_length, lg0.half_width, TerrainLimits(margin=self.p.terrain_margin))
            if tm.has_terrain or self.p.terrain is True:
                self.terrain = tm
        hip_h = self.p.terrain_hip_height if self.terrain is not None else self.p.hip_height
        t_single = self.p.terrain_t_single if self.terrain is not None else self.p.t_single
        self.z_torso = (self.geo.ankle_height + hip_h * leg - self.geo.legs["left"].hip[2]
                        - self.geo.pelvis[2])
        # nominal standing posture -> pendulum height and where the torso sits relative to the feet
        self._com_off: dict[str, np.ndarray] = {}
        feet0 = {LEFT: np.array([0.0, self.width / 2, 0.0]), RIGHT: np.array([0.0, -self.width / 2, 0.0])}
        self.scratch.qpos[:] = model.qpos0
        _, p_t, com = self._whole_body(np.zeros(2), 0.0, self._targets(feet0))
        self.z_c = float(com[2])
        self.omega = float(np.sqrt(G / self.z_c))
        self.stand_offset = p_t[:2].copy()          # torso relative to the sole midpoint (heading frame)
        tc = 1.0 / self.omega
        # how hard this body is on its motors at the nominal gait -> shorter, slower steps for heavy bodies
        self.static_load, self.swing_load = self._motor_load(feet0, t_single * tc, self.p.step_forward * leg)
        load = self.static_load + self.swing_load          # 0 when no leg motor is torque-limited
        self.gait_scale = (1.0 if self.p.load_target <= 0 or load <= 0
                           else float(np.clip(self.p.load_target / load, 0.6, 1.0)))
        g = self.gait_scale
        ticks = lambda t: int(round(t * tc / np.sqrt(g) / self.control_dt))   # noqa: E731
        self.n_ss, self.n_ds = max(4, ticks(t_single)), max(2, ticks(self.p.t_double))
        self.n_start, self.n_final = max(2, ticks(self.p.t_start)), max(2, ticks(self.p.t_final))
        self.swing_h = self.p.swing_height * leg
        self.lim = StepLimits(forward=self.p.step_forward * leg * g, backward=self.p.step_backward * leg * g,
                              lateral=self.p.step_lateral * leg * g, turn=np.deg2rad(self.p.turn_deg) * g,
                              width=self.width, min_width=self.p.min_width_frac * self.width,
                              lookahead=0.45 * leg, blend=0.75 * leg)
        self.pc = ZmpPreview(self.control_dt, self.z_c, self.p.preview_s)
        alpha = lambda tau: float(1.0 - np.exp(-self.control_dt / max(tau, 1e-6)))   # noqa: E731
        self._vf_alpha, self._pel_alpha = alpha(self.p.vel_filter_tau), alpha(self.p.swing_world_tau)
        self._tgt_alpha = alpha(0.03)
        lg = self.geo.legs["left"]
        p = self.p
        self.cap = CaptureLimits(
            forward=p.recovery_reach * leg, backward=0.7 * p.recovery_reach * leg, min_width=self.lim.min_width,
            max_width=self.width + p.recovery_width * leg, hl=p.recovery_cop * lg.half_length,
            hw=p.recovery_cop * lg.half_width, t_min=p.recovery_t_min * tc, t_max=max(p.recovery_t_min, 1.4) * tc,
            t_ds=max(2, round(0.2 * tc / self.control_dt)) * self.control_dt,
            foot_speed=p.recovery_foot_speed * leg, margin=0.85, width=self.width, depth=p.recovery_depth)
        self.reach = TerrainReach(forward=p.terrain_step_forward * leg * g, backward=self.lim.backward,
                                  min_width=self.lim.min_width, max_width=self.width + self.lim.lateral)
        self._smooth_w = max(1, int(round(p.terrain_smooth_s / self.control_dt)))
        self._reset_state()

    # ------------------------------------------------------------------ API
    @staticmethod
    def supports(model: mujoco.MjModel) -> bool:
        return model_supported(model)

    @property
    def walk_speed(self) -> float:
        """Nominal straight-line speed (m/s) of this body with these limits."""
        return self.lim.forward / ((self.n_ss + self.n_ds) * self.control_dt)

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Put the robot in the knees-bent standing pose over its current footprints and start balancing.

        Writes data.qpos of the root, abdomen and legs (arms/hands keep their angles), zeroes data.qvel, writes
        data.ctrl of the leg/abdomen actuators and calls mj_forward."""
        self._reset_state()
        mujoco.mj_forward(model, data)
        feet = {s: np.array(self._foot_pose(data, s)) for s in (LEFT, RIGHT)}
        mid = (feet[LEFT][:2] + feet[RIGHT][:2]) / 2
        yaw = mean_yaw(feet[LEFT][2], feet[RIGHT][2])
        self.scratch.qpos[:] = data.qpos
        if self.terrain is not None:
            for s_ in (LEFT, RIGHT):
                fh = self.terrain.ground(*feet[s_], z_ref=float(data.xpos[self.foot_body[s_]][2]) - 0.06)
                self.ground[s_] = (fh.z, fh.normal)
            self._dz = (self.ground[LEFT][0] + self.ground[RIGHT][0]) / 2
            self._zh_hist = [self._dz]
        q, p_t, _ = self._whole_body(mid, yaw, self._ttargets(feet))
        data.qpos[0:3] = p_t
        mujoco.mju_mat2Quat(data.qpos[3:7], rot_z(yaw).ravel())
        data.qpos[self.qadr] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        self.feet = feet
        self.x[0] = data.subtree_com[self.torso_id][:2]
        self.zref = mid[None].copy()
        data.ctrl[self.act] = self._clip_ctrl(model, q)

    def _clip_ctrl(self, model: mujoco.MjModel, u: np.ndarray) -> np.ndarray:
        """Clip to ctrlrange only where the actuator is ctrl-limited (an unlimited one has ctrlrange 0 0)."""
        lo, hi = model.actuator_ctrlrange[self.act].T
        limited = model.actuator_ctrllimited[self.act].astype(bool)
        return np.where(limited, np.clip(u, lo, hi), u)

    def set_path(self, path_xy, final_yaw: float | None = None) -> None:
        """Walk along world-frame waypoints (n, 2) and stop with the torso on the last one, facing final_yaw
        (rad; None = arrival direction). None or an empty path = stop (feet side by side) and stand.
        Raises ValueError for a non-finite path or yaw (nothing changes then)."""
        pts = None if path_xy is None else np.asarray(path_xy, float).reshape(-1, 2)
        if (pts is not None and not np.isfinite(pts).all()) or (final_yaw is not None
                                                                 and not np.isfinite(final_yaw)):
            raise ValueError("set_path needs finite waypoints and final yaw")
        self._vel = None
        if pts is None or len(pts) == 0:
            self._pending = ("stop", None, None, False)
            self.goal_xy = None
            if self.mode != "fallen":
                self.mode, self.message = "stand", "stopping"
        else:
            self._pending = ("path", pts, final_yaw, True)
            self._path = (pts.copy(), final_yaw)
            self.corrections, self.arrivals = 0, []
            self.mode, self.message = "walk", "walking"
        if pts is None or len(pts) == 0:
            self._path = None
        if self.swing is None and self.rec is None:
            self._apply_pending()

    def set_velocity(self, vx: float, vy: float = 0.0, wz: float = 0.0) -> None:
        """Continuous walking: forward / left speed (m/s) and yaw rate (rad/s) in the robot frame, clipped to the
        step limits. (0, 0, 0) = stop. Raises ValueError for non-finite values."""
        if not np.isfinite([vx, vy, wz]).all():
            raise ValueError("set_velocity needs finite vx, vy, wz")
        if abs(vx) + abs(vy) + abs(wz) < 1e-6:
            self.set_path(None)
            return
        self.goal_xy = None
        self._vel = (float(vx), float(vy), float(wz))
        self.mode, self.message = "walk", "velocity command"
        self._path = None
        if self.swing is None:
            self._pending = ("velocity", None, None, False)
            if self.rec is None:
                self._apply_pending()

    def status(self) -> dict:
        """{"mode": "stand"|"walk"|"done"|"fallen", "message", "goal_error_m", "heading_error_deg",
        "footsteps": [[x, y, yaw, side (0 left, 1 right)], ...] still to be taken (the one in the air first),
        "support": "left"|"right"|"double", "recovering": bool (a push recovery is in progress),
        "recovery_steps": footsteps re-timed, moved or inserted by push recoveries so far,
        "terrain": bool (terrain mode), "footsteps_z": ground height of each entry of "footsteps"}"""
        steps = ([[*self.swing.target.round(4).tolist(), self.swing.side]] if self.swing is not None else [])
        steps += [[round(a, 4) for a in st[:3]] + [int(st[3])] for st in self.queue]
        zs = ([round(float(self.swing.ground1[0]), 4) if self.swing.ground1 is not None else 0.0]
              if self.swing is not None else [])
        zs += [round(float(st[4]), 4) if len(st) > 4 else 0.0 for st in self.queue]
        has_goal = self.goal_xy is not None
        return {"mode": self.mode, "message": self.message, "footsteps": steps, "support": self.support,
                "goal_error_m": float(self.goal_err) if has_goal else None,
                "heading_error_deg": float(np.rad2deg(self.yaw_err)) if has_goal else None,
                "recovering": self.rec is not None, "recovery_steps": int(self.recovery_steps),
                "terrain": self.terrain is not None, "footsteps_z": zs}

    def _motor_load(self, feet0: dict, t_single: float, step: float) -> tuple[float, float]:
        """Predicted motor use of the nominal gait, as fractions of the torque limits.

        static: max |tau| / limit standing on one foot with the CoM over it (gravity inverse dynamics with the
        whole weight on that sole) - usually the stance hip_x holding the upper body sideways.
        swing: hip-pitch torque to swing the leg (its joint-space inertia times the peak minimum-jerk angular
        acceleration of a full stride) over the hip_y limit.
        """
        m, d = self.model, self.scratch
        d.qpos[:] = m.qpos0
        q, p_t, _ = self._whole_body(feet0[LEFT][:2], 0.0, self._targets(feet0), key="load")
        d.qpos[0:3] = p_t
        d.qpos[3:7] = (1, 0, 0, 0)
        d.qpos[self.qadr] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jp, jr, feet0[LEFT], self.foot_body[LEFT])
        tau = d.qfrc_bias[self.dadr] - jp[:, self.dadr].T @ np.array([0.0, 0.0, self.mass * G])
        lim = np.abs(m.actuator_forcerange[self.act]).max(axis=1)
        lim = np.where(m.actuator_forcelimited[self.act].astype(bool) & (lim > 0), lim, np.inf)
        static = float(np.max(np.abs(tau) / lim))
        hy = 2 + 6 * RIGHT + 2                            # swing leg = right, hip pitch
        e = np.zeros(m.nv)
        e[self.dadr[hy]] = 1.0
        Me = np.zeros(m.nv)
        mujoco.mj_mulM(m, d, Me, e)
        acc = 5.77 * (2.0 * step / self.geo.leg_length) / (0.85 * t_single) ** 2
        swing = float(Me[self.dadr[hy]] * acc / lim[hy])
        self._com_off.pop("load", None)
        return static, swing

    # --------------------------------------------------------------- state
    def _reset_state(self) -> None:
        self.mode, self.message = "stand", "standing"
        self.queue: list[list[float]] = []
        self.swing: _Swing | None = None
        self.phase, self.phase_left = "ds", 0
        self.support = "double"
        self.feet = {LEFT: np.zeros(3), RIGHT: np.zeros(3)}
        self.last_side = RIGHT
        self.goal_xy, self.goal_yaw = None, 0.0
        self.goal_err, self.yaw_err = 0.0, 0.0
        self.corrections = 0
        self.arrivals: list[tuple[float, float]] = []   # (m, deg) torso error at every stop on the way
        self.settle = 0
        self._pending = None
        self._vel = None
        self.x = np.zeros((3, 2))                        # preview state: CoM position, velocity, acceleration
        self.zref = np.zeros((1, 2))
        self.zk = 0
        self._event = False                              # touchdown this tick: references jump
        self.q_prev = self.qd_prev = None
        self.pt_prev = self.vt_prev = None
        self._acc_t = np.zeros(3)
        self._vfilt = None
        self._pel_err = np.zeros(6)
        self.ticks = 0
        self.rec: _Recovery | None = None               # push recovery in progress
        self.recovery_steps = 0                          # footsteps re-timed / moved / inserted by recoveries
        self.recoveries = 0                              # recoveries started
        self.dcm_error = 0.0
        self._path = None                                # (waypoints, final yaw) of the last path request
        self.ground = {LEFT: (0.0, np.array([0.0, 0.0, 1.0])), RIGHT: (0.0, np.array([0.0, 0.0, 1.0]))}
        self._dz = 0.0                                   # torso height offset over the support (terrain)
        self.zh = None                                   # support height under the ZMP reference (terrain)
        self._zh_hist: list[float] = []

    # --------------------------------------------------------- measurement
    def _foot_pose(self, data, side) -> tuple[float, float, float]:
        b = self.foot_body[side]
        R = data.xmat[b].reshape(3, 3)
        c = data.xpos[b] + R @ self.geo.legs[SIDE_NAME[side]].sole
        return float(c[0]), float(c[1]), yaw_of(R)

    def _measure(self, model, data) -> _Meas:
        mujoco.mj_subtreeVel(model, data)
        R_t = data.xmat[self.torso_id].reshape(3, 3).copy()
        vel = data.subtree_linvel[self.torso_id].copy()
        # low-pass the CoM velocity: contact impacts and servo ringing would otherwise go straight into the
        # ankle torques through the DCM feedback
        if self._vfilt is None:
            self._vfilt = vel
        self._vfilt = self._vfilt + self._vf_alpha * (vel - self._vfilt)
        return _Meas(com=data.subtree_com[self.torso_id].copy(), com_vel=self._vfilt.copy(), R_torso=R_t,
                     p_pelvis=data.xpos[self.pelvis_id].copy(),
                     R_pelvis=data.xmat[self.pelvis_id].reshape(3, 3).copy(),
                     feet={s: np.array(self._foot_pose(data, s)) for s in (LEFT, RIGHT)}, com_vel_raw=vel,
                     angmom=data.subtree_angmom[self.torso_id].copy(),
                     fallen=bool(R_t[2, 2] < 0.6 or data.qpos[2] - self._dz < 0.6 * self.z_torso
                                 or not np.isfinite(data.qpos).all()))

    # ---------------------------------------------------------- kinematics
    @staticmethod
    def _targets(feet: dict) -> dict:
        return {s: (np.array([feet[s][0], feet[s][1], 0.0]), rot_z(float(feet[s][2]))) for s in (LEFT, RIGHT)}

    def _ttargets(self, feet: dict) -> dict:
        """Sole targets of the stance feet: on terrain at the ground height and tilted onto the surface."""
        if self.terrain is None:
            return self._targets(feet)
        out = {}
        for s in (LEFT, RIGHT):
            z, n = self.ground[s]
            out[s] = (np.array([feet[s][0], feet[s][1], z]), tilted(float(feet[s][2]), n))
        return out

    def _legs(self, p_t, R_t, targets) -> np.ndarray:
        """Abdomen (0, 0) + both legs for a torso pose (pelvis follows the torso)."""
        p_pel = p_t + R_t @ self.geo.pelvis
        q = np.zeros(14)
        for s in (LEFT, RIGHT):
            pos, R = targets[s]
            q[2 + 6 * s:8 + 6 * s], _ = leg_ik(self.geo.legs[SIDE_NAME[s]], p_pel, R_t, pos, R)
        return q

    def _fk_com(self, p_t, R_t, q) -> np.ndarray:
        d = self.scratch
        d.qpos[0:3] = p_t
        mujoco.mju_mat2Quat(d.qpos[3:7], R_t.ravel())
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        return d.subtree_com[self.torso_id].copy()

    def _whole_body(self, com_xy, yaw, targets, key: str = "ref"):
        """Torso position + joint angles whose whole-body CoM (x, y) is com_xy, at the nominal torso height.

        Fixed-point iteration on the torso position with forward kinematics of the full model (the scratch
        data holds the current arm/hand angles), warm-started per `key` so the reference and the measured
        solutions stay independent and smooth. Returns (q, torso position, CoM).
        """
        R_t = rot_z(yaw)
        off = self._com_off.get(key, self._com_off.get("ref", np.zeros(2)))
        p_t = np.array([com_xy[0] + off[0], com_xy[1] + off[1], self.z_torso + self._dz])
        com = np.zeros(3)
        for _ in range(4):
            com = self._fk_com(p_t, R_t, self._legs(p_t, R_t, targets))
            step = com_xy - com[:2]
            p_t[:2] += step
            if abs(step[0]) + abs(step[1]) < 1e-5:
                break
        self._com_off[key] = p_t[:2] - com_xy
        return self._legs(p_t, R_t, targets), p_t, com

    # ---------------------------------------------------- footstep schedule
    def _apply_pending(self) -> None:
        """Turn a pending path / stop / velocity request into a footstep queue (called between steps)."""
        if self._pending is None:
            return
        kind, pts, fy, shift = self._pending
        self._pending = None
        walking = self.ticks > 0 and self.phase == "ds" and self.phase_left > 0
        first = (1 - self.last_side) if walking else None
        if kind == "path":
            goal = pts[-1].copy()
            if shift:
                # the torso, not the sole midpoint, should end on the goal: plan the feet to the goal minus the
                # torso offset of the standing posture. (Shifting before planning, not the planned steps after,
                # lets the planner see that a robot already standing on the goal needs no steps.)
                fy = goal_heading(self.feet[LEFT], self.feet[RIGHT], pts, fy, self.lim)
                pts = pts.copy()
                pts[-1] -= rot_z(fy)[:2, :2] @ self.stand_offset
            steps, gy = plan_footsteps(self.feet[LEFT], self.feet[RIGHT], pts, fy, self.lim, first_side=first)
            self.queue = steps
            self.goal_xy, self.goal_yaw = goal, gy
        elif kind == "stop":
            self.queue = square_up(self.feet[LEFT], self.feet[RIGHT], self.last_side, self.lim) if walking else (
                plan_footsteps(self.feet[LEFT], self.feet[RIGHT],
                               ((self.feet[LEFT][:2] + self.feet[RIGHT][:2]) / 2)[None], None, self.lim)[0])
        elif kind == "velocity" and self._vel is not None:
            self.queue = velocity_footsteps(self.feet[LEFT], self.feet[RIGHT], 1 - self.last_side, *self._vel,
                                            (self.n_ss + self.n_ds) * self.control_dt, self.lim)
        if self.terrain is not None and self.queue:
            self.queue = self._terrain_queue(self.queue)
        if self.queue and self.phase == "ds" and self.phase_left <= 0:
            self.phase_left = self.n_start               # starting from standing: slow first weight shift
        self._build_zref()

    def _terrain_queue(self, steps: list) -> list:
        """Footsteps moved off edges (walk_terrain.repair_footsteps), each [x, y, yaw, side, z, nx, ny, nz]."""
        feet = {s: self.feet[s].copy() for s in (LEFT, RIGHT)}
        zf = {s: self.ground[s][0] for s in (LEFT, RIGHT)}
        if self.swing is not None:                       # the queue starts after the foot in the air lands
            feet[self.swing.side] = self.swing.target.copy()
            zf[self.swing.side] = self.swing.ground1[0] if self.swing.ground1 is not None else 0.0
        rep = repair_footsteps([st[:4] for st in steps], feet[LEFT], feet[RIGHT], self.terrain, self.reach,
                               self.lim.forward, z_feet=zf)
        return [[*st[:4], st[4], *st[5][:, 2]] for st in rep]

    def _build_zref(self) -> None:
        """ZMP reference from this tick to the end of the plan plus the preview horizon: on the stance sole in
        single support, a linear ramp between soles in double support, the feet midpoint at the end."""
        p_now = self.zref[min(self.zk, len(self.zref) - 1)].copy()
        feet = {s: self.feet[s].copy() for s in (LEFT, RIGHT)}
        segs = [p_now[None]]
        n_ds = max(1, self.phase_left)
        # support height under the ZMP reference (terrain): same segments, heights of the stance soles
        fz = {s: self.ground[s][0] for s in (LEFT, RIGHT)}
        h_now = self._zh_hist[-1] if self._zh_hist else (fz[LEFT] + fz[RIGHT]) / 2
        hsegs = [np.array([h_now])]
        if self.swing is not None:
            stance = feet[1 - self.swing.side][:2].copy()
            n_sw = max(0, self.swing.n - self.swing.k)
            segs.append(np.tile(stance, (n_sw, 1)))
            hsegs.append(np.full(n_sw, fz[1 - self.swing.side]))
            h_now = fz[1 - self.swing.side]
            feet[self.swing.side] = self.swing.target.copy()
            fz[self.swing.side] = self.swing.ground1[0] if self.swing.ground1 is not None else 0.0
            p_now, n_ds = stance, self.n_ds
        for st in self.queue:
            sd = int(st[3])
            stance = feet[1 - sd][:2].copy()
            ramp = np.arange(1, n_ds + 1) / n_ds
            segs.append(p_now + (stance - p_now) * ramp[:, None])
            segs.append(np.tile(stance, (self.n_ss, 1)))
            hsegs.append(h_now + (fz[1 - sd] - h_now) * ramp)
            hsegs.append(np.full(self.n_ss, fz[1 - sd]))
            h_now = fz[1 - sd]
            feet[sd] = np.array(st[:3])
            fz[sd] = float(st[4]) if len(st) > 4 else 0.0
            p_now, n_ds = stance, self.n_ds
        mid = (feet[LEFT][:2] + feet[RIGHT][:2]) / 2
        n_fin = self.n_final if (self.queue or self.swing is not None) else n_ds
        ramp = np.arange(1, n_fin + 1) / n_fin
        segs.append(p_now + (mid - p_now) * ramp[:, None])
        segs.append(np.tile(mid, (self.pc.N + 1, 1)))
        h_mid = (fz[LEFT] + fz[RIGHT]) / 2
        hsegs.append(h_now + (h_mid - h_now) * ramp)
        hsegs.append(np.full(self.pc.N + 1, h_mid))
        self.zref = np.vstack(segs)
        if self.terrain is not None:
            self.zh = np.concatenate(hsegs)
        self.zk = 0

    def _zref_window(self) -> np.ndarray:
        w = self.zref[self.zk + 1:self.zk + 1 + self.pc.N]
        if len(w) < self.pc.N:
            w = np.vstack([w, np.tile(self.zref[-1], (self.pc.N - len(w), 1))])
        return w

    def _advance_phase(self, meas: _Meas) -> None:
        if self.phase == "ss":
            sw = self.swing
            sw.k += 1
            if sw.k < sw.n:
                return
            # touchdown: from now on the foot is where it actually landed
            self.feet[sw.side] = meas.feet[sw.side].copy()
            if self.terrain is not None:
                if sw.ground1 is not None and not sw.rec:
                    self.ground[sw.side] = sw.ground1
                else:
                    fh = self.terrain.ground(*self.feet[sw.side], z_ref=self.ground[1 - sw.side][0])
                    self.ground[sw.side] = (fh.z, fh.normal)
            self.last_side = sw.side
            self._event = True
            self.swing = None
            self.phase = "ds"
            if self.rec is not None:                 # recovery: the recovery tick decides what comes next
                self.phase_left = 0
                self.rec.counted = False
                if self.rec.next is not None:
                    self._schedule(self.rec.next, self.cap.t_ds)
                    self.rec.next = None
                return
            self.phase_left = self.n_ds if self.queue else self.n_final
            if self._vel is not None and self._pending is None:
                self._pending = ("velocity", None, None, False)
            if self._pending is not None:
                self._apply_pending()
            else:
                self._build_zref()
            return
        if self.rec is not None:
            return
        self.phase_left -= 1
        if self.phase_left > 0 or not self.queue:
            return
        st = self.queue.pop(0)
        if self.terrain is not None:
            self._lift_terrain(st)
            self.phase = "ss"
            self._build_zref()
            return
        self.swing = _Swing(side=int(st[3]), start=self.feet[int(st[3])].copy(), target=np.array(st[:3]),
                            n=self.n_ss)
        self.phase = "ss"
        self._build_zref()

    def _lift_terrain(self, st) -> None:
        """Start a swing on terrain: ground at both ends and the apex over everything along the swing line."""
        side = int(st[3])
        sw = _Swing(side=side, start=self.feet[side].copy(), target=np.array(st[:3], float), n=self.n_ss)
        sw.ground0 = self.ground[side]
        sw.ground1 = ((float(st[4]), np.asarray(st[5:8], float)) if len(st) > 7
                      else (0.0, np.array([0.0, 0.0, 1.0])))
        self._swing_apex(sw)
        self.swing = sw

    def _swing_apex(self, sw: _Swing) -> None:
        tm = self.terrain
        z0, z1 = sw.ground0[0], sw.ground1[0]
        top = tm.clearance(sw.start, sw.target, z_ref=max(z0, z1))
        flat = (max(abs(z0), abs(z1), top) < tm.lim.flat_tol and sw.ground0[1][2] > 1 - 1e-9
                and sw.ground1[1][2] > 1 - 1e-9)
        sw.apex = None if flat else max(z0, z1, top) + self.p.terrain_clearance

    def _swing_pose_terrain(self, sw: _Swing) -> tuple[np.ndarray, np.ndarray]:
        """Swing over terrain: lift to the apex in the first 30 % (the sole barely moves sideways yet), move
        across between 20 % and 85 %, come down onto the landing surface in the last 30 % (minimum-jerk
        pieces, so position, velocity and acceleration are continuous); the sole tilts from the lift-off
        surface to the landing surface with the horizontal progress."""
        phi = (sw.k + 1) / sw.n
        mj = lambda u: u * u * u * (10 - 15 * u + 6 * u * u)      # noqa: E731
        s = mj(min(1.0, max(0.0, (phi - 0.2) / 0.65)))
        xy = sw.start[:2] + (sw.target[:2] - sw.start[:2]) * s
        yaw = sw.start[2] + wrap(sw.target[2] - sw.start[2]) * s
        (z0, n0), (z1, n1), za = sw.ground0, sw.ground1, sw.apex
        if phi <= 0.3:
            z = z0 + (za - z0) * mj(phi / 0.3)
        elif phi < 0.7:
            z = za
        else:
            z = za + (z1 - za) * mj(min(1.0, (phi - 0.7) / 0.3))
        n = n0 + (n1 - n0) * s
        return np.array([xy[0], xy[1], z]), tilted(float(yaw), n)

    def _swing_pose(self, sw: _Swing) -> tuple[np.ndarray, np.ndarray]:
        """Swing sole pose: minimum-jerk in the plane (between 5% and 90% of the swing), a quartic bump of
        height swing_h in z with zero vertical speed at lift-off and touchdown. Recovery swings follow a
        quintic re-solved every tick from the current reference state (see _swing_receding)."""
        if sw.rec:
            return self._swing_receding(sw)
        if sw.apex is not None:
            return self._swing_pose_terrain(sw)
        phi = (sw.k + 1) / sw.n
        u = min(1.0, max(0.0, (phi - 0.05) / 0.85))
        s = u * u * u * (10 - 15 * u + 6 * u * u)
        xy = sw.start[:2] + (sw.target[:2] - sw.start[:2]) * s
        yaw = sw.start[2] + wrap(sw.target[2] - sw.start[2]) * s
        z = self.swing_h * 16 * phi * phi * (1 - phi) ** 2
        return np.array([xy[0], xy[1], z]), rot_z(float(yaw))

    def _nominal_state(self, sw: _Swing) -> np.ndarray:
        """(3, 4) position / velocity / acceleration of (x, y, z, yaw) of the nominal swing profile now."""
        dt = self.control_dt
        rows = []
        for dk in (-1, 0, 1):
            k = sw.k
            sw.k = min(max(k + dk, -1), sw.n - 1)
            pos, R = self._swing_pose(sw)
            sw.k = k
            rows.append(np.array([pos[0], pos[1], pos[2], sw.start[2] + wrap(yaw_of(R) - sw.start[2])]))
        a, b, c = rows
        return np.vstack([b, (c - a) / (2 * dt), (c - 2 * b + a) / (dt * dt)])

    def _start_receding(self, sw: _Swing, state: np.ndarray | None = None) -> None:
        """Switch a swing to recovery mode, continuing from its current reference state."""
        st = self._nominal_state(sw) if state is None else state
        if not sw.rec:
            sw.s = st
            sw.rec = True
        self._z_profile(sw)

    def _z_profile(self, sw: _Swing) -> None:
        """Height profile from the current z state to the ground at touchdown: a quintic plus a C2 bump that
        keeps the sole at least 0.6 swing heights above the ground if there is time to lift it."""
        dt = self.control_dt
        ticks = max(1, sw.n - 1 - sw.k)
        z0, v0, a0 = (float(v) for v in sw.s[:, 2])
        T = ticks * dt
        z_end = 0.0
        if self.terrain is not None:
            fh = self.terrain.ground(float(sw.target[0]), float(sw.target[1]), float(sw.target[2]),
                                     z_ref=self.ground[1 - sw.side][0])
            z_end = fh.z
        mid = _quintic_eval(z0, v0, a0, z_end, T, T / 2)[0]
        bump = max(0.0, max(z0, z_end) + 0.6 * self.swing_h - mid) if T > 0.12 else 0.0
        sw.zprof = (sw.k, z0, v0, a0, z_end, ticks, bump)

    def _swing_receding(self, sw: _Swing) -> tuple[np.ndarray, np.ndarray]:
        dt = self.control_dt
        T = max(0, sw.n - 1 - sw.k) * dt
        tgt = np.array([sw.target[0], sw.target[1], 0.0, sw.s[0, 3] + wrap(sw.target[2] - sw.s[0, 3])])
        nxt = _quintic_next(sw.s, tgt, T, dt)
        k0, z0, v0, a0, z1, ticks, bump = sw.zprof
        t = (sw.k - k0) * dt
        Tz = ticks * dt
        z, vz, az = _quintic_eval(z0, v0, a0, z1, Tz, min(t, Tz))
        u = min(1.0, max(0.0, t / Tz)) if Tz > 0 else 1.0
        z += bump * 64 * u ** 3 * (1 - u) ** 3
        vz += bump * 64 * (3 * u * u * (1 - u) ** 3 - 3 * u ** 3 * (1 - u) ** 2) / max(Tz, 1e-9)
        nxt[:, 2] = (z, vz, az)
        sw.s = nxt
        zmin = 0.0 if self.terrain is None else -np.inf
        return np.array([nxt[0, 0], nxt[0, 1], max(zmin, nxt[0, 2])]), rot_z(float(nxt[0, 3]))

    def _adjust_step(self, sw: _Swing, xi_err: np.ndarray) -> None:
        """Capture-point stepping: move the landing point by the DCM error expected at touchdown.

        Under the LIPM a DCM error grows as exp(omega t) while the stance foot is fixed; landing that much
        further keeps the robot on its nominal DCM pattern relative to its feet (Pratt et al., "Capture
        point", Humanoids 2006; Englsberger et al. 2015). Errors below a dead band leave the plan exact, and
        the following footsteps return to the plan, so a push is absorbed over a few steps.
        """
        phi = (sw.k + 1) / sw.n
        if self.p.k_step <= 0 or phi > 0.8:
            return
        gain = self.p.k_step
        if self.p.step_predict:
            gain *= min(float(np.exp(self.omega * (sw.n - sw.k - 1) * self.control_dt)), 3.0)
        adj = gain * xi_err
        n = float(np.linalg.norm(adj))
        if n <= self.p.step_deadband:
            tgt = sw.plan[:2].copy()
        else:
            adj *= min(n - self.p.step_deadband, self.p.step_adjust * self.geo.leg_length) / n
            tgt = sw.plan[:2] + adj
            stance = self.feet[1 - sw.side]
            c, s = np.cos(stance[2]), np.sin(stance[2])
            lat = -s * (tgt[0] - stance[0]) + c * (tgt[1] - stance[1])
            sign = 1.0 if sw.side == LEFT else -1.0
            if sign * lat < self.lim.min_width:          # never step onto / across the stance foot
                tgt = tgt + sign * (self.lim.min_width - sign * lat) * np.array([-s, c])
        # the landing point moves smoothly (low-pass + speed limit): the swing reference is differentiated
        # twice for the feed-forward, so a jumping target would kick the leg
        d = self._tgt_alpha * (tgt - sw.target[:2])
        vmax = 2.0 * self.lim.forward / (sw.n * self.control_dt) * self.control_dt
        dn = float(np.linalg.norm(d))
        sw.target[:2] += d * (min(dn, vmax) / dn) if dn > 1e-12 else 0.0

    # ----------------------------------------------------------- main loop
    def step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """One control tick: writes data.ctrl for the abdomen and leg actuators only."""
        mujoco.mj_forward(model, data)
        meas = self._measure(model, data)
        self.ticks += 1
        if meas.fallen:
            if self.mode != "fallen":
                self.mode, self.message = "fallen", "robot fell"
                self.queue, self.swing, self._pending = [], None, None
            return
        self._advance_phase(meas)
        self.zk += 1
        # --- reference CoM: ZMP preview control on the footstep plan
        self.x = self.pc.step(self.x, self._zref_window())
        p_ref = self.zref[min(self.zk, len(self.zref) - 1)]
        c_ref, v_ref = self.x[0], self.x[1]
        xi_err = (meas.com[:2] + meas.com_vel[:2] / self.omega) - (c_ref + v_ref / self.omega)
        self.dcm_error = float(np.hypot(xi_err[0], xi_err[1]))
        if self.p.recovery:
            if self.rec is None and self._push_detected(meas, xi_err):
                self._enter_recovery(meas)
            if self.rec is not None:
                c_ref, v_ref, p_ref, xi_err = self._recovery_tick(meas)
        # --- foot targets. The reference solution (differentiated for the velocity / acceleration
        #     feed-forward) keeps the stance feet as they landed; the position targets use where they are now
        #     (a foot can creep or twist a little), so a slipping foot does not drag the torso along.
        if self.terrain is not None:
            self._update_height()
        targets = self._ttargets(self.feet)
        sw = self.swing
        if sw is not None:
            if not sw.rec and sw.apex is None:
                self._adjust_step(sw, xi_err)
            targets[sw.side] = self._swing_pose(sw)
            self.support = "right" if sw.side == LEFT else "left"
        else:
            self.support = "double"
        self.scratch.qpos[:] = data.qpos                  # arms / hands as they are
        yaw_ref = mean_yaw(yaw_of(targets[LEFT][1]), yaw_of(targets[RIGHT][1]))
        q_ref, p_ref_t, _ = self._whole_body(c_ref, yaw_ref, targets, key="ref")
        q, p_t, yaw_t = q_ref.copy(), p_ref_t, yaw_ref
        if self.p.track_stance:
            for s in (LEFT, RIGHT):
                if sw is None or sw.side != s:
                    if self.terrain is None:
                        targets[s] = (np.array([meas.feet[s][0], meas.feet[s][1], 0.0]), rot_z(float(meas.feet[s][2])))
                    else:
                        z, n = self.ground[s]
                        targets[s] = (np.array([meas.feet[s][0], meas.feet[s][1], z]),
                                      tilted(float(meas.feet[s][2]), n))
            yaw_t = mean_yaw(yaw_of(targets[LEFT][1]), yaw_of(targets[RIGHT][1]))
            q, p_t, _ = self._whole_body(c_ref, yaw_t, targets, key="meas")
        if sw is not None:
            self._swing_leg_world(q, sw, targets[sw.side], p_t, yaw_t, meas)
        else:
            self._swing_leg_world(None, None, None, p_t, yaw_t, meas)
        qd, qdd = self._derivatives(q_ref, p_ref_t)
        q = self._torso_feedback(q, meas, yaw_t)
        tau = self._feedforward(model, data, meas, xi_err, qdd, p_ref)
        kp = np.maximum(model.actuator_gainprm[self.act, 0], 1e-6)   # read live: the GUI may edit motors
        kv = -model.actuator_biasprm[self.act, 2]
        data.ctrl[self.act] = self._clip_ctrl(model, q + (kv / kp) * qd + tau / kp)
        self._update_status(data, meas)

    # ------------------------------------------------------------ push recovery
    def _enter_recovery(self, meas: _Meas) -> None:
        """The DCM left the walking pattern by more than recovery_trigger (a push): drop the planned footsteps
        and balance / step from the measured state until the robot is quiet again, then re-plan the walk from
        where the feet are (re-planning from the measured state as in Urata, Nishiwaki, Nakanishi, Okada,
        Kagami, Inaba, "Online walking pattern generation for push recovery and minimum delay to commanded
        change of direction and speed", IROS 2012)."""
        p0 = self.zref[min(self.zk, len(self.zref) - 1)].copy()
        self.rec = _Recovery(c=meas.com[:2].copy(), v=meas.com_vel[:2].copy(), p=p0)
        self.recoveries += 1
        self.queue = []
        self._event = True                               # the reference jumps to the measured CoM
        self.message = "recovering from a push"
        if self.swing is not None and not self.swing.rec:
            self._start_receding(self.swing)
        if self.swing is None:
            self.phase, self.phase_left = "ds", 0

    def _push_detected(self, meas: _Meas, xi_err: np.ndarray) -> bool:
        """DCM error outside the trigger ellipse: recovery_trigger sideways, recovery_trigger_forward along the
        torso heading (both scaled up on terrain, where the planar pendulum is a rougher model)."""
        leg = self.geo.leg_length
        side, fwd = self.p.recovery_trigger * leg, self.p.recovery_trigger_forward * leg
        if self.terrain is not None:
            k = self.p.terrain_recovery_trigger / max(self.p.recovery_trigger, 1e-9)
            side, fwd = side * k, fwd * k
        yaw = float(np.arctan2(meas.R_torso[1, 0], meas.R_torso[0, 0]))
        c, s_ = np.cos(yaw), np.sin(yaw)
        ef, es = c * xi_err[0] + s_ * xi_err[1], -s_ * xi_err[0] + c * xi_err[1]
        return (ef / max(fwd, 1e-9)) ** 2 + (es / max(side, 1e-9)) ** 2 > 1.0

    def _recovery_dcm(self, meas: _Meas) -> np.ndarray:
        """DCM for recovery decisions, with the centroidal angular momentum L that the joints will absorb:
        about the CoM, m c'' = f and L' = (p - c) x f give c'' = omega^2 (c - p) + (-L'_y, L'_x) / (m z), so
        once L is brought back to zero the CoM has gained (L_y, -L_x) / (m z) of velocity (a swinging leg or
        a torso pitching back after a push). Uses the unfiltered CoM velocity (the decisions are urgent)."""
        om = self.omega
        xi = meas.com[:2] + meas.com_vel_raw[:2] / om
        if self.p.recovery_angmom > 0 and meas.angmom is not None:
            L = meas.angmom
            xi = xi + self.p.recovery_angmom * np.array([L[1], -L[0]]) / (self.mass * self.z_c * om)
        return xi

    def _schedule(self, st, delay: float) -> None:
        r = self.rec
        r.sched = st
        r.lift_left = r.lift_total = max(1, int(round(delay / self.control_dt)))

    def _snap_foothold(self, target, side: int) -> np.ndarray:
        """Nearest good foothold to a recovery landing point (along the sole's heading, then sideways)."""
        x, y, yaw = (float(v) for v in target[:3])
        z_ref = self.ground[1 - side][0]
        c, s_ = np.cos(yaw), np.sin(yaw)
        for d in (0.0, 0.02, -0.02, 0.04, -0.04, 0.06, -0.06, 0.09, -0.09, 0.12, -0.12):
            for lat in (0.0, 0.03, -0.03):
                cx, cy = x + c * d - s_ * lat, y + s_ * d + c * lat
                if self.terrain.foothold(cx, cy, yaw, z_ref).ok:
                    return np.array([cx, cy])
        return np.array([x, y])

    def _lift(self, st) -> None:
        """Lift a foot for a recovery step (StepChoice) planned in double support."""
        side = st.side
        if self.terrain is not None:
            st.target[:2] = self._snap_foothold(st.target, side)
        n = max(4, int(round(st.duration / self.control_dt)))
        f = self.feet[side]
        sw = _Swing(side=side, start=f.copy(), target=np.array(st.target, float), n=n)
        sw.s = np.vstack([[f[0], f[1], 0.0, f[2]], np.zeros(4), np.zeros(4)])
        sw.rec = True
        self._z_profile(sw)
        self.swing, self.phase = sw, "ss"
        self.recovery_steps += 1
        self.rec.steps += 1
        self.rec.counted = True

    def _recovery_tick(self, meas: _Meas):
        """One tick of push recovery. Decides the steps (walk_recovery.plan_swing / plan_double_support) and
        produces a CoM reference that is consistent with a centre of pressure the feet can actually apply:
        c'' = omega^2 (c - p) with p on the support polygon as close as useful to the DCM, the reference drifting
        slowly towards the measured CoM (recovery_track) instead of pulling the body back to the old pattern.
        Returns (c_ref, v_ref, centre of pressure, DCM error w.r.t. this reference)."""
        r, dt, om, cap = self.rec, self.control_dt, self.omega, self.cap
        r.ticks += 1
        xi = self._recovery_dcm(meas)
        if r.xi_prev is not None:
            rate = (xi - r.xi_prev) / dt
            r.xi_rate = rate if r.xi_rate is None else r.xi_rate + self._vf_alpha * (rate - r.xi_rate)
        r.xi_prev = xi.copy()
        rate = r.xi_rate if self.p.recovery_rate else None
        sw = self.swing
        if sw is not None:
            stance = self.feet[1 - sw.side]
            rem = (sw.n - 1 - sw.k) * dt
            if rem > 0.03 and (r.ticks % 2 == 1 or not r.counted):
                z = float(sw.s[0, 2])
                # land no sooner than the foot can come down (and, with recovery_min_swing > 0, keep that
                # fraction of the shortest swing from lift-off)
                t_land = max(0.05 + z / 0.8, self.p.recovery_min_swing * cap.t_min - sw.k * dt)
                plan = plan_swing(xi, om, stance, sw.side, sw.s[0, :2], rem, sw.plan, cap,
                                  t_land_min=min(rem, t_land), dcm_rate=rate)
                st = plan.steps[0]
                if self.terrain is not None:
                    st.target[:2] = self._snap_foothold(st.target, sw.side)
                moved = float(np.hypot(*(st.target[:2] - sw.plan[:2]))) > 0.01
                sw.target[:2] = st.target[:2]
                new_n = sw.k + 1 + max(1, int(round(st.duration / dt)))
                retimed = abs(new_n - sw.n) >= 2
                if retimed:
                    sw.n = new_n
                    self._z_profile(sw)
                r.next = plan.steps[1] if len(plan.steps) > 1 else None
                if not r.counted and (moved or retimed or r.next is not None):
                    r.counted = True
                    self.recovery_steps += 1
                    r.steps += 1
            p = clamp_to_sole(xi, stance[:2], float(stance[2]), cap.hl, cap.hw)
        else:
            both = [self.feet[LEFT], self.feet[RIGHT]]
            if r.sched is not None:
                r.lift_left -= 1
                if r.lift_left > 0 and r.ticks % 2 == 1:
                    plan = plan_double_support(xi, om, self.feet, cap, side=r.sched.side,
                                               lift_delay=r.lift_left * dt, dcm_rate=rate)
                    r.sched = plan.steps[0]
                    r.next = plan.steps[1] if len(plan.steps) > 1 else None
                stance = self.feet[1 - r.sched.side]
                lam = 1.0 - r.lift_left / max(r.lift_total, 1)
                p = ((1 - lam) * clamp_to_support(xi, both, cap.hl, cap.hw)
                     + lam * clamp_to_sole(xi, stance[:2], float(stance[2]), cap.hl, cap.hw))
                if r.lift_left <= 0:
                    st, r.sched = r.sched, None
                    self._lift(st)
            else:
                plan = (plan_double_support(xi, om, self.feet, cap, hold_margin=self.p.recovery_hold, dcm_rate=rate)
                        if r.ticks % 2 == 1 else None)
                if plan is not None and plan.steps:
                    self._schedule(plan.steps[0], plan.lift_delay)
                    r.next = plan.steps[1] if len(plan.steps) > 1 else None
                    p = clamp_to_support(xi, both, cap.hl, cap.hw)
                else:
                    goal = (both[0][:2] + both[1][:2]) / 2
                    p = clamp_to_support(goal + self.p.recovery_gain * (xi - goal), both, cap.hl, cap.hw)
                    quiet = (np.linalg.norm(xi - goal) < 0.03 and np.linalg.norm(meas.com_vel[:2]) < 0.05)
                    r.settle = r.settle + 1 if quiet else 0
        # CoM reference consistent with that centre of pressure, kept near the measured state
        b = self.p.recovery_track
        r.c = r.c + b * (meas.com[:2] - r.c)
        r.v = r.v + b * (meas.com_vel[:2] - r.v)
        a = om * om * (r.c - p)
        r.v = r.v + a * dt
        r.c = r.c + r.v * dt
        r.p = np.asarray(p, float)
        self.x = np.array([r.c, r.v, a])
        xi_err = (meas.com[:2] + meas.com_vel[:2] / om) - (r.c + r.v / om)
        c_ref, v_ref = r.c.copy(), r.v.copy()
        if (self.swing is None and r.sched is None and r.settle * dt >= self.p.recovery_settle_s) or (
                r.ticks * dt > self.p.recovery_timeout_s and self.swing is None):
            self._exit_recovery()
        return c_ref, v_ref, r.p.copy(), xi_err

    def _remaining_path(self, xy: np.ndarray) -> np.ndarray:
        pts = self._path[0]
        if len(pts) < 2:
            return pts.copy()
        best, best_d = (0, pts[0]), np.inf
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            e = b - a
            t = float(np.clip(np.dot(xy - a, e) / max(np.dot(e, e), 1e-12), 0.0, 1.0))
            q = a + t * e
            d = float(np.hypot(*(xy - q)))
            if d < best_d:
                best, best_d = (i, q), d
        i, q = best
        return np.vstack([q, pts[i + 1:]])

    def _exit_recovery(self) -> None:
        """Quiet again: hand the reference back to the preview controller and re-plan from the feet."""
        r = self.rec
        self.x = np.array([r.c, r.v, self.omega ** 2 * (r.c - r.p)])
        self.zref = r.p[None].copy()
        self.zk = 0
        self.rec = None
        self.phase, self.phase_left = "ds", 0
        self.queue = []
        if self._pending is None:
            mid = (self.feet[LEFT][:2] + self.feet[RIGHT][:2]) / 2
            if self._vel is not None:
                self._pending = ("velocity", None, None, False)
                self.message = "velocity command"
            elif self.goal_xy is not None and self._path is not None:
                self._pending = ("path", self._remaining_path(mid), self._path[1], True)
                self.mode, self.message = "walk", "walking (after a push)"
            else:
                self._pending = ("stop", None, None, False)
                self.message = "standing" if self.mode == "stand" else self.message
        self._apply_pending()

    def _update_height(self) -> None:
        """Torso height over the terrain: a triangular moving average (+-terrain_smooth_s) of the support
        height under the ZMP reference (past values remembered, future ones from the plan), so the CoM rises
        and sinks smoothly and early for each step up or down while the pendulum stays planar per step."""
        if self.zh is None or len(self.zh) == 0:
            return
        W = self._smooth_w
        k = min(self.zk, len(self.zh) - 1)
        if self.rec is not None:
            # recovering: the plan's future no longer applies; hold the height of the feet on the ground
            sup = [1 - self.swing.side] if self.swing is not None else [LEFT, RIGHT]
            now = float(np.mean([self.ground[s_][0] for s_ in sup]))
            fut = np.full(W, now)
        else:
            now = float(self.zh[k])
            fut = self.zh[k + 1:k + 1 + W]
            if len(fut) < W:
                fut = np.concatenate([fut, np.full(W - len(fut), self.zh[-1])])
        self._zh_hist.append(now)
        if len(self._zh_hist) > W + 1:
            del self._zh_hist[:len(self._zh_hist) - (W + 1)]
        past = np.asarray(self._zh_hist[:-1][::-1])        # j = -1, -2, ...
        w = W + 1 - np.arange(1, W + 1)
        num = (W + 1) * now + float(w[:len(past)] @ past) + float(w @ fut)
        den = (W + 1) + float(w[:len(past)].sum()) + float(w.sum())
        self._dz = num / den

    def _derivatives(self, q_ref: np.ndarray, p_ref_t: np.ndarray):
        """Reference joint velocity / acceleration and torso acceleration by finite differences of the smooth
        reference solution, skipped on a touchdown tick (where the reference jumps to the landed foot)."""
        dt = self.control_dt
        if self.q_prev is None or self._event:
            qd = self.qd_prev if self.qd_prev is not None else np.zeros(14)
            qdd = np.zeros(14)
            vt = self.vt_prev if self.vt_prev is not None else np.zeros(3)
            at = np.zeros(3)
        else:
            qd = (q_ref - self.q_prev) / dt
            qdd = (qd - self.qd_prev) / dt if self.qd_prev is not None else np.zeros(14)
            vt = (p_ref_t - self.pt_prev) / dt
            at = (vt - self.vt_prev) / dt if self.vt_prev is not None else np.zeros(3)
        self._event = False
        self.q_prev, self.qd_prev = q_ref.copy(), qd.copy()
        self.pt_prev, self.vt_prev = p_ref_t.copy(), vt.copy()
        self._acc_t = np.clip(at, -20.0, 20.0)
        return qd, np.clip(qdd, -100.0, 100.0)

    def _swing_leg_world(self, q, sw, target, p_t, yaw_t, meas: _Meas) -> None:
        """Solve the swing leg against the reference pelvis corrected by the low-passed measured pelvis error, so
        the foot lands where it was planned even when the body lags or leans. (The filter matters: reacting to
        fast body oscillations with the heavy swing leg would amplify them.)"""
        R_t = rot_z(yaw_t)
        p_pel = p_t + R_t @ self.geo.pelvis
        dR = R_t.T @ meas.R_pelvis
        err = np.concatenate([meas.p_pelvis - p_pel,
                              0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])])
        self._pel_err += self._pel_alpha * (err - self._pel_err)
        if sw is None or self.p.swing_world <= 0:
            return
        e = self.p.swing_world * self._pel_err
        th = float(np.linalg.norm(e[3:]))
        Rc = np.eye(3)
        if th > 1e-9:                                    # Rodrigues
            k = e[3:] / th
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            Rc = Rc + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        i0 = 2 + 6 * sw.side
        q[i0:i0 + 6], _ = leg_ik(self.geo.legs[SIDE_NAME[sw.side]], p_pel + e[:3], R_t @ Rc, *target)

    def _torso_feedback(self, q, meas: _Meas, yaw_t: float) -> np.ndarray:
        """Hip strategy: the torso hangs on the stance leg(s), torso pitch = foot pitch - (abdomen + hip + knee +
        ankle), so a stance hip offset of +k * pitch turns the torso back upright (same for roll / hip_x)."""
        if self.p.k_torso <= 0:
            return q
        R = rot_z(-yaw_t) @ meas.R_torso                 # torso attitude in the heading frame
        pitch = float(np.arctan2(-R[2, 0], R[2, 2]))     # + = leaning forward
        roll = float(np.arctan2(R[2, 1], R[2, 2]))       # + = leaning right
        for s in ([LEFT, RIGHT] if self.swing is None else [1 - self.swing.side]):
            q[2 + 6 * s + 2] += self.p.k_torso * pitch
            q[2 + 6 * s + 0] += self.p.k_torso * roll
        return q

    def _feedforward(self, model, data, meas: _Meas, xi_err: np.ndarray, qdd: np.ndarray,
                     zmp_ref: np.ndarray) -> np.ndarray:
        """Floating-base inverse dynamics with a contact wrench consistent with the reference motion.

        M qdd + h = S^T tau + sum_feet J_c^T f. The six unactuated (torso) rows fix the total contact wrench
        for the reference accelerations (torso + all driven joints). In double support it is split by a
        weighted minimum norm: weights follow the reference ZMP between the feet, and sole moments are
        expensive so the load is shared through vertical forces, as a real robot does. Balance feedback then
        moves every stance sole's centre of pressure by k_dcm * DCM error (with the LIPM force change
        dF = -m omega^2 dp). Joints above the ankle get the torques for that wrench; the ankle torques are
        those of the wrench with its CoP clipped to the sole: if the CoP saturates the whole body tips about
        the sole edge, but the torso must not be the part that gives way.
        """
        nv = model.nv
        vec = np.zeros(nv)
        vec[0:3] = self._acc_t
        vec[self.dadr] = qdd
        h = np.zeros(nv)
        mujoco.mj_mulM(model, data, h, vec)
        h += data.qfrc_bias
        stance = [LEFT, RIGHT] if self.swing is None else [1 - self.swing.side]
        J, o = {}, {}
        for s in stance:
            o[s] = np.array([meas.feet[s][0], meas.feet[s][1], self.ground[s][0]])
            jp, jr = np.zeros((3, nv)), np.zeros((3, nv))
            mujoco.mj_jac(model, data, jp, jr, o[s], self.foot_body[s])
            J[s] = np.vstack([jp, jr])
        if len(stance) == 2:
            a, b = o[LEFT][:2], o[RIGHT][:2]
            d = a - b
            alpha = float(np.clip(np.dot(zmp_ref - b, d) / max(np.dot(d, d), 1e-9), 0.0, 1.0))
            share = {LEFT: max(alpha, 0.02), RIGHT: max(1.0 - alpha, 0.02)}
        else:
            share = {stance[0]: 1.0}
        A = np.hstack([J[s][:, :6].T for s in stance])                     # 6 x 6k
        lw = self.geo.legs["left"].half_width ** 2
        winv = np.concatenate([share[s] * np.array([1, 1, 1, lw, lw, lw]) for s in stance])
        try:
            f_all = winv * (A.T @ np.linalg.solve((A * winv) @ A.T, h[:6]))
        except np.linalg.LinAlgError:
            f_all = np.zeros(6 * len(stance))
        dp = self.p.k_dcm * xi_err
        tau, tau_ankle = h[self.dadr].copy(), h[self.dadr].copy()
        for i, s in enumerate(stance):
            f = f_all[6 * i:6 * i + 6].copy()
            fz = f[2]
            f[0:2] -= self.mass * self.omega ** 2 * dp * share[s] / sum(share.values())
            fc = f.copy()
            if fz > 1.0:
                f[3] += dp[1] * fz
                f[4] -= dp[0] * fz
                lg = self.geo.legs[SIDE_NAME[s]]
                c, si = np.cos(meas.feet[s][2]), np.sin(meas.feet[s][2])
                rel = np.array([-f[4], f[3]]) / fz                         # CoP relative to the sole centre
                lx = np.clip(c * rel[0] + si * rel[1], -0.8 * lg.half_length, 0.8 * lg.half_length)
                ly = np.clip(-si * rel[0] + c * rel[1], -0.8 * lg.half_width, 0.8 * lg.half_width)
                fc[3] = (si * lx + c * ly) * fz
                fc[4] = -(c * lx - si * ly) * fz
            tau -= J[s][:, self.dadr].T @ f
            tau_ankle -= J[s][:, self.dadr].T @ fc
        for s in stance:
            tau[2 + 6 * s + 4:2 + 6 * s + 6] = tau_ankle[2 + 6 * s + 4:2 + 6 * s + 6]
        lim = np.abs(model.actuator_forcerange[self.act]).max(axis=1)
        lim[~model.actuator_forcelimited[self.act].astype(bool)] = np.inf
        return np.clip(tau, -lim, lim)

    def _update_status(self, data, meas: _Meas) -> None:
        x, y = float(data.qpos[0]), float(data.qpos[1])
        yaw = float(np.arctan2(meas.R_torso[1, 0], meas.R_torso[0, 0]))
        if self.goal_xy is not None:
            self.goal_err = float(np.hypot(x - self.goal_xy[0], y - self.goal_xy[1]))
            self.yaw_err = float(wrap(yaw - self.goal_yaw))
        idle = self.swing is None and not self.queue and self.phase_left <= 0 and self._pending is None
        if not idle or self._vel is not None or self.rec is not None:
            self.settle = 0
            return
        if self.mode == "stand" and self.message == "stopping":
            self.message = "standing"
        if self.mode != "walk":
            return
        self.settle = self.settle + 1 if np.linalg.norm(meas.com_vel[:2]) < 0.03 else 0
        if self.settle * self.control_dt < 0.3:
            return
        self.settle = 0
        if self.goal_xy is None:
            self.mode, self.message = "stand", "standing"
            return
        self.arrivals.append((self.goal_err, float(np.rad2deg(self.yaw_err))))
        if self.corrections < self.p.max_corrections and (
                self.goal_err > self.p.goal_tolerance or abs(self.yaw_err) > np.deg2rad(self.p.yaw_tolerance_deg)):
            # move the stance by the measured torso error (it already contains the torso offset)
            self.corrections += 1
            goal = self.goal_xy.copy()
            tgt = (self.feet[LEFT][:2] + self.feet[RIGHT][:2]) / 2 + (goal - np.array([x, y]))
            self._pending = ("path", tgt[None], self.goal_yaw, False)
            self._apply_pending()
            self.goal_xy = goal
            self.message = f"correction step {self.corrections}"
            return
        self.mode = "done"
        self.message = f"goal reached ({self.goal_err:.3f} m, {np.rad2deg(self.yaw_err):.1f} deg)"


class TimedPush:
    """A push on the robot for the GUI / scripts, independent of the active controller.

        push = TimedPush((0.0, 100.0, 0.0), duration=0.1, frame="robot")   # 100 N to the robot's left
        ... every physics step (or control tick):  push.update(model, data)

    The force (N) acts at the body's centre of mass through data.xfrc_applied from the first update until
    `duration` seconds of simulated time (data.time) later, then it is cleared and update() returns False.
    frame="robot" turns the force by the torso heading at the first update (x forward, y left)."""

    def __init__(self, force_xyz, duration: float = 0.1, body: str = "torso", frame: str = "world"):
        f = np.asarray(force_xyz, float).ravel()
        self.force = np.concatenate([f, np.zeros(3 - len(f))])[:3] if len(f) < 3 else f[:3].copy()
        if not np.isfinite(self.force).all() or not np.isfinite(duration) or duration < 0:
            raise ValueError("a push needs a finite force and a non-negative duration")
        if frame not in ("world", "robot"):
            raise ValueError('frame must be "world" or "robot"')
        self.duration, self.body, self.frame = float(duration), body, frame
        self.t_end: float | None = None
        self.done = False

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        b = model.body(self.body).id
        if self.done:
            return False
        if self.t_end is None:
            self.t_end = float(data.time) + self.duration
            if self.frame == "robot":
                R = data.xmat[b].reshape(3, 3)
                self.force = rot_z(float(np.arctan2(R[1, 0], R[0, 0]))) @ self.force
        if data.time >= self.t_end - 1e-9:
            data.xfrc_applied[b, :3] = 0.0
            self.done = True
            return False
        data.xfrc_applied[b, :3] = self.force
        return True


def _push_list(push) -> list[tuple[float, np.ndarray, float]]:
    """None | (time_s, force_xyz, duration_s) | list of those -> [(t, force (3,), duration), ...] sorted by time."""
    if push is None:
        return []
    if len(push) == 3 and np.ndim(push[0]) == 0 and np.size(push[1]) in (2, 3):
        push = [push]
    out = []
    for t0, f, dur in push:
        f = np.asarray(f, float).ravel()
        f = np.concatenate([f, np.zeros(3 - len(f))]) if len(f) < 3 else f[:3]
        if not (np.isfinite(f).all() and np.isfinite([t0, dur]).all()) or dur < 0:
            raise ValueError("a push needs a finite time, force and non-negative duration")
        out.append((float(t0), f, float(dur)))
    return sorted(out, key=lambda p: p[0])


def simulate_walk(model: mujoco.MjModel, goal_xy, goal_yaw: float | None = None, timeout: float = 30.0,
                  path=None, tolerance: float = 0.1, params: WalkParams | None = None,
                  settle_s: float = 0.5, push=None, push_frame: str = "world",
                  stop_after_push_s: float | None = None) -> dict:
    """Walk a fresh copy of the robot headless from its standing pose to goal_xy (straight, or along `path`;
    goal_xy is appended when the path ends elsewhere). goal_xy=None = stand where it is for `timeout` s (push
    tests).

    push: None, (time_s, force_xyz, duration_s) or a list of them - a force (N) on the torso centre of mass,
    time_s counted from the walk command; force_xyz is in the world frame, or with push_frame="robot" in the
    torso heading frame at the push onset (x forward, y left). stop_after_push_s ends the run that long after
    the last push (survival tests do not need to finish the walk).

    Returns {"reached", "error_m", "heading_error_deg", "time_s", "fell", "energy_J", "cost_of_transport",
    "peak_torque_frac", "distance_m", "speed_mps", "corrections", "arrivals", "status", "pushes",
    "recovery_steps", "recoveries", "max_dcm_error_m"}:
    error_m is the torso (x, y) distance to the goal once stopped; distance_m is the path length of the point
    between the feet (one sample per footstep, so the body's side-to-side sway is not counted); energy_J sums
    |torque * joint speed| * dt over the leg and abdomen actuators; cost_of_transport = energy / (m g distance);
    peak_torque_frac = max |actuator force| / torque limit over them; reached = stopped ("done") within
    `tolerance` without falling (standing: did not fall); arrivals = torso errors (m, deg) at every stop, before
    each correction step; pushes = the world-frame pushes applied [(t, [fx, fy, fz], duration)];
    recovery_steps = footsteps the balance controller re-timed, moved or inserted; recoveries = how often it
    switched to push recovery; max_dcm_error_m = largest DCM error vs. the walking pattern. time_s counts from the walk
    command (after settle_s of standing). Raises ValueError for unsupported models (check
    WalkController.supports first) or a malformed push.
    """
    pushes = _push_list(push)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ctrl = WalkController(model, params)
    ctrl.reset(model, data)
    n_sub, dt = ctrl.substeps, ctrl.control_dt
    for _ in range(int(round(settle_s / dt))):
        ctrl.step(model, data)
        mujoco.mj_step(model, data, n_sub)
    start = data.qpos[:2].copy()
    standing = goal_xy is None
    goal = start.copy() if standing else np.asarray(goal_xy, float).reshape(2)
    if not standing:
        pts = np.asarray(path, float).reshape(-1, 2) if path is not None else np.array([start, goal])
        if len(pts) == 0 or np.linalg.norm(pts[-1] - goal) > 1e-3:
            pts = np.vstack([pts, goal])      # the result is measured against goal_xy, so the walk must end there
        ctrl.set_path(pts, goal_yaw)
    act, dof = ctrl.act, ctrl.dadr
    lim = np.abs(model.actuator_forcerange[act]).max(axis=1)
    lim = np.where(model.actuator_forcelimited[act].astype(bool) & (lim > 0), lim, np.inf)
    energy = peak = t = dist = 0.0
    feet_mid = lambda: (ctrl.feet[0][:2] + ctrl.feet[1][:2]) / 2   # noqa: E731  (moves once per step, no sway)
    prev = feet_mid()
    applied, active = [], None            # pushes as applied (world frame); (end time, force) of the current one
    queue = list(pushes)
    end_t = timeout
    if pushes and stop_after_push_s is not None:
        end_t = min(timeout, max(p[0] + p[2] for p in pushes) + stop_after_push_s)
    max_dcm = 0.0
    torso = ctrl.torso_id
    while t < end_t and (standing or ctrl.mode not in ("done", "fallen")) and ctrl.mode != "fallen":
        while queue and t >= queue[0][0] - 1e-9:
            t0, f, dur = queue.pop(0)
            if push_frame == "robot":
                R = data.xmat[torso].reshape(3, 3)
                f = rot_z(float(np.arctan2(R[1, 0], R[0, 0]))) @ f
            active = (t + dur, f)
            applied.append((round(t, 4), [float(v) for v in f], dur))
        if active is not None and t >= active[0] - 1e-9:
            active = None
        data.xfrc_applied[torso, :3] = active[1] if active is not None else 0.0
        ctrl.step(model, data)
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
            frc = data.actuator_force[act]
            energy += float(np.abs(frc * data.qvel[dof]).sum()) * model.opt.timestep
            peak = max(peak, float((np.abs(frc) / lim).max()))
        t += dt
        dist += float(np.linalg.norm(feet_mid() - prev))
        prev = feet_mid()
        max_dcm = max(max_dcm, float(getattr(ctrl, "dcm_error", 0.0)))
    data.xfrc_applied[torso, :3] = 0.0
    err = float(np.hypot(*(data.qpos[:2] - goal)))
    R = data.xmat[ctrl.torso_id].reshape(3, 3)
    want = ctrl.goal_yaw if goal_yaw is None else goal_yaw
    herr = float(np.rad2deg(wrap(np.arctan2(R[1, 0], R[0, 0]) - want)))
    fell = ctrl.mode == "fallen"
    reached = (not fell) if standing else bool(ctrl.mode == "done" and err <= tolerance)
    return {"reached": reached, "error_m": err, "heading_error_deg": herr,
            "time_s": t, "fell": fell, "energy_J": energy,
            "cost_of_transport": energy / (ctrl.mass * G * max(dist, 1e-6)), "peak_torque_frac": peak,
            "distance_m": dist, "speed_mps": dist / max(t, 1e-6), "corrections": ctrl.corrections,
            "arrivals": list(ctrl.arrivals), "status": ctrl.status(), "pushes": applied,
            "recovery_steps": int(getattr(ctrl, "recovery_steps", 0)), "recoveries": int(getattr(ctrl, "recoveries", 0)),
            "max_dcm_error_m": max_dcm}


def _trial(args):
    src, params, kind, force_xyz, t_push, duration, goal, timeout, after = args
    model = mujoco.MjModel.from_xml_string(src) if isinstance(src, str) else src
    r = simulate_walk(model, None if kind == "stand" else goal, timeout=timeout, params=params,
                      push=(t_push, force_xyz, duration), push_frame="robot",
                      stop_after_push_s=after)
    return {"t_push": t_push, "force_robot_frame": [float(v) for v in force_xyz], "fell": r["fell"],
            "reached": r["reached"], "error_m": r["error_m"], "time_s": r["time_s"],
            "recovery_steps": r["recovery_steps"], "max_dcm_error_m": r["max_dcm_error_m"]}


def push_trials(model_or_xml, kind: str = "walk", force: float = 100.0, direction="lateral", n: int = 8,
                seed: int = 0, duration: float = 0.1, goal_distance: float = 3.0, params: WalkParams | None = None,
                timeout: float = 40.0, stop_after_push_s: float | None = None, workers: int = 1) -> dict:
    """Push-recovery benchmark: n headless runs, each with one push of `force` N for `duration` s on the torso.

    kind="walk": the robot walks goal_distance m straight ahead and is pushed at a random time in one full
    gait cycle (two footsteps) after 2 s of walking, so the push lands at a random gait phase; survival = it
    did not fall (reached also requires it to stop on the goal within 10 cm). kind="stand": the robot stands
    and is pushed at a random time between 0.5 and 1.5 s; it may step.
    direction (robot frame): "lateral" (alternating left / right), "sagittal" (alternating forward /
    backward), "random" (uniform angle) or an angle in degrees (0 = forward, 90 = left).
    model_or_xml: a compiled MjModel (not modified; runs use their own MjData) or an MJCF string.
    workers > 1 runs trials in that many processes (each single-threaded).

    Returns {"survived", "reached", "n", "rate", "trials": [{"t_push", "force_robot_frame", "fell",
    "reached", "error_m", "time_s", "recovery_steps", "max_dcm_error_m"}, ...]}.
    """
    if kind not in ("walk", "stand"):
        raise ValueError('kind must be "walk" or "stand"')
    src = model_or_xml
    probe = mujoco.MjModel.from_xml_string(src) if isinstance(src, str) else src
    rng = np.random.default_rng(seed)
    ctrl = WalkController(probe, params)
    cycle = 2 * (ctrl.n_ss + ctrl.n_ds) * ctrl.control_dt
    jobs = []
    for i in range(n):
        if direction == "lateral":
            ang = 90.0 if i % 2 == 0 else -90.0
        elif direction == "sagittal":
            ang = 0.0 if i % 2 == 0 else 180.0
        elif direction == "random":
            ang = float(rng.uniform(-180.0, 180.0))
        else:
            ang = float(direction)
        f = force * np.array([np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang)), 0.0])
        t_push = float(2.0 + rng.uniform(0.0, cycle)) if kind == "walk" else float(rng.uniform(0.5, 1.5))
        after = stop_after_push_s if stop_after_push_s is not None else (None if kind == "walk" else 4.0)
        jobs.append((src, params, kind, f, t_push, duration, (goal_distance, 0.0),
                     timeout if kind == "walk" else t_push + duration + 4.0, after))
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            trials = list(ex.map(_trial, jobs))
    else:
        trials = [_trial(j) for j in jobs]
    survived = sum(not tr["fell"] for tr in trials)
    return {"survived": survived, "reached": sum(tr["reached"] for tr in trials), "n": n,
            "rate": survived / max(n, 1), "trials": trials}
