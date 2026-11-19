"""MPC walking controller: linear MPC of the inverted pendulum with automatic footstep placement.

    path -> planned footsteps -> LIPM MPC (CoM jerk + next footsteps, re-solved from the measured CoM)
         -> swing / stance foot targets -> whole-body CoM placement + leg IK -> servo targets + feed-forward

This is the MPC alternative to the ZMP-preview walker (control/walker.py). The walker generates
its CoM pattern once per plan change (Kajita preview control) and corrects pushes with hand-made feedback laws
(DCM -> centre of pressure, capture-point step shifts). Here a receding-horizon optimisation does both:

* **Pattern generation by MPC** (control/mpc_lipm.py; Wieber, Humanoids 2006; Herdt, Diedam, Wieber,
  Dimitrov, Mombaur, Diehl, Advanced Robotics 2010). Every `mpc_period` (0.1 s, and at every touchdown and
  lift-off) a QP over the CoM jerk on a 1.6 s horizon (16 x 0.1 s) and the positions of the next <= 3
  footsteps is solved from the **measured** CoM state, blended with the running plan by `anchor` (0.5: a
  first-order observer; 1.0 = purely measured, which walked worse on the full MuJoCo model, see MPCParams). The ZMP
  must stay inside the support polygon at every sample (the yaw-rotated sole of whichever foot will be on the
  ground, the hull of both soles in double support), footsteps stay in a kinematic box relative to the
  previous foot without crossing the legs, and the cost trades ZMP centring, velocity tracking, jerk,
  closeness to the planner's footsteps and a terminal capture-point term. A push shows up as a different
  initial state: the optimum moves the ZMP to the edge of the sole and the next footsteps outwards.
* **Step timing adaptation** (in the spirit of Khadiv, Herzog, Moosavian, Righetti, IEEE T-RO 2020): when the
  measured DCM leaves the plan by more than `timing_trigger`, each update also solves the MPC for a few
  contact timings (stay longer in double support, or land the swing foot 0.1 / 0.2 s earlier) and keeps the
  cheapest plan. Foot placement alone cannot catch a push towards the stance leg with fixed timing.
* **Footstep plan and goal accuracy** (control/walk_footsteps.py, shared with the walker): the planner's
  steps (path following, turns, the closing pair on the goal) are the nominal footsteps the MPC is pulled
  towards; the closing pair carries a higher weight so arrival stays precise, and correction steps are
  re-planned from the landed feet when the settled torso error is above tolerance (as in the walker).
* **Execution on position servos** (control/mpc_body.py): the CoM reference is the MPC plan evaluated at the
  current time (re-anchored at every solve, see above), placed by whole-body forward kinematics and
  closed-form leg IK; the swing foot follows a quintic re-planned every tick towards the MPC's landing point;
  the feed-forward torque comes from floating-base inverse dynamics whose contact wrench has its centre of
  pressure at the planned ZMP (plus a small DCM feedback between solves), so the soft position servos act like
  torque-controlled joints, as in force-based MPC (Di Carlo et al., IROS 2018; Kim et al., WBIC
  2019) but with the LIPM as the prediction model.

QP solver: control/mpc_qp.py, a Goldfarb-Idnani dual active-set method (C++ in native/src/mpc.cpp, numpy
fallback), warm-started from the previous active set.

Only leg and abdomen actuators are written; arms and hands belong to other controllers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .mpc_body import G, LegBody, Meas
from .mpc_lipm import Foot, LipmMPC, MPCWeights, Plan, Problem, StepBox, Support
from .walk_footsteps import (LEFT, RIGHT, StepLimits, goal_heading, mean_yaw, plan_footsteps, square_up,
                             velocity_footsteps, wrap)
from .walk_ik import model_supported, rot_z, yaw_of

SIDE_NAME = {LEFT: "left", RIGHT: "right"}


@dataclass
class MPCParams:
    """Tunables. "x leg" = times thigh + shin; "x tc" = times the pendulum time constant sqrt(z_c / g)."""
    hip_height: float = 0.93          # nominal hip-ankle distance, fraction of the leg length
    # nominal footstep plan (same meaning as WalkParams)
    step_forward: float = 0.22        # x leg
    step_backward: float = 0.12       # x leg
    step_lateral: float = 0.09        # x leg
    turn_deg: float = 20.0
    min_width_frac: float = 0.65      # closest lateral foot spacing, fraction of the hip width
    swing_height: float = 0.075       # x leg
    t_single: float = 1.6             # x tc
    t_double: float = 0.45            # x tc
    t_start: float = 2.0              # x tc, first weight shift from standing
    t_final: float = 1.5              # x tc, final weight shift onto both feet
    # MPC
    mpc_T: float = 0.1                # s, sampling period of the prediction
    mpc_N: int = 16                   # samples (horizon = N T)
    mpc_period: float = 0.1           # s between re-solves (also re-solved at every touchdown / lift-off);
                                      # = mpc_T, so the uniform grid moves by whole samples between solves
    point_min_s: float = 0.03         # s, no extra ZMP constraint instants closer to now than this
    grid: str = "uniform"             # "uniform": N x mpc_T from now (Herdt 2010); "contact": interval
                                      # boundaries on touchdowns / lift-offs + mid-interval ZMP checks
    max_footsteps: int = 3            # footstep decision variables
    qp_backend: str = "auto"          # "auto": C++ kernel (native/src/mpc.cpp) when built, else numpy; "numpy"
    weights: MPCWeights = field(default_factory=MPCWeights)
    final_step_weight: float = 10.0   # closing pair: x weights.step (precise arrival)
    sole_length_frac: float = 0.8     # usable fraction of the sole half length for the ZMP (= the CoP clip of
    sole_width_frac: float = 0.8      # ... and of the half width                            the feed-forward)
    reach_forward: float = 0.32       # x leg, footstep box relative to the previous foot
    reach_backward: float = 0.2       # x leg
    reach_lateral: float = 0.22       # x leg, outwards beyond the nominal width
    swing_speed: float = 1.5          # m/s, how far the landing point may move per second of swing left
    lock_time: float = 0.05           # s of horizontal swing left below which the landing point is frozen
    step_deadband: float = 0.01       # m, landing points closer than this to the plan land exactly on it
    anchor: float = 0.5               # 0..1: each solve starts from plan + anchor * (measured - plan) CoM
                                      # position / velocity (a first-order observer; 1 = purely measured)
    anchor_vel: float | None = None   # separate blend for the velocity (None = anchor)
    # execution
    k_dcm: float = 1.0                # DCM error vs the current plan -> centre of pressure shift (between solves)
    k_torso: float = 1.0              # hip strategy gain
    swing_world: float = 1.0
    swing_world_tau: float = 0.08
    vel_filter_tau: float = 0.02
    track_stance: bool = True
    load_target: float = 0.55
    goal_tolerance: float = 0.02
    yaw_tolerance_deg: float = 2.0
    max_corrections: int = 2
    control_hz: float = 200.0
    # step timing adaptation after a disturbance
    timing_adapt: bool = True
    timing_trigger: float = 0.03      # m, measured DCM vs plan error that opens a recovery window
    timing_window: float = 1.0        # s, timing options are evaluated for this long after the trigger
    timing_weight: float = 0.5        # cost per s^2 of timing change (MPC cost units)
    ds_extensions: tuple = (0.1, 0.2, 0.3)       # s added to the current double support
    max_ds_extension: float = 0.6     # s, total per double support
    swing_cuts: tuple = (0.1, 0.2)    # s removed from the current swing
    min_swing_left: float = 0.12      # s, a swing is never shortened below this
    min_swing_progress: float = 0.15  # the swing foot must be this far into its swing to land early


@dataclass
class _Swing:
    side: int
    start: np.ndarray                 # (x, y, yaw) sole pose at lift-off
    target: np.ndarray                # (x, y, yaw) landing pose (MPC)
    plan: np.ndarray                  # (x, y, yaw) landing pose from the footstep plan
    n: int                            # nominal duration (ticks)
    key: int
    k: int = 0                        # ticks since lift-off
    weight: float = 1.0
    xy: np.ndarray = field(default_factory=lambda: np.zeros((3, 2)))   # horizontal reference p, v, a
    phi: float = 0.0                  # swing progress 0..1 (advances by (1 - phi) / left per tick)
    left: int = 0                     # ticks to touchdown (re-timed by the MPC's timing adaptation)

    def __post_init__(self):
        if self.left <= 0:
            self.left = self.n

    def phi_next(self) -> float:
        return self.phi + (1.0 - self.phi) / max(self.left, 1)

    def time_left(self, dt: float, until: float = 1.0) -> float:
        """Seconds until the progress reaches `until` (phi moves linearly in time to 1 at touchdown)."""
        return max(0.0, (until - self.phi) / max(1.0 - self.phi, 1e-9)) * self.left * dt


class MPCWalkController:
    """LIPM model-predictive walking controller with footstep adaptation (see module docstring)."""
    name = "mpc"

    def __init__(self, model: mujoco.MjModel, params: MPCParams | None = None):
        if not model_supported(model):
            raise ValueError("MPCWalkController needs the built-in humanoid's legs (see walk_ik.model_supported)")
        self.p = params or MPCParams()
        self.model = model
        dt = float(model.opt.timestep)
        self.substeps = max(1, int(round(1.0 / (self.p.control_hz * dt))))
        self.control_dt = self.substeps * dt
        self.body = LegBody(model, self.p.hip_height, self.p.vel_filter_tau, self.p.swing_world_tau,
                            self.control_dt)
        b = self.body
        self.geo, self.mass, self.act, self.dadr = b.geo, b.mass, b.act, b.dadr
        self.torso_id = b.torso_id
        self.z_c, self.omega, self.stand_offset = b.z_c, b.omega, b.stand_offset
        leg = self.geo.leg_length
        self.width = b.width
        tc = 1.0 / self.omega
        self.static_load, self.swing_load = b.motor_load(self.p.t_single * tc, self.p.step_forward * leg)
        load = self.static_load + self.swing_load
        self.gait_scale = (1.0 if self.p.load_target <= 0 or load <= 0
                           else float(np.clip(self.p.load_target / load, 0.6, 1.0)))
        g = self.gait_scale
        ticks = lambda t: int(round(t * tc / np.sqrt(g) / self.control_dt))   # noqa: E731
        self.n_ss, self.n_ds = max(4, ticks(self.p.t_single)), max(2, ticks(self.p.t_double))
        self.n_start, self.n_final = max(2, ticks(self.p.t_start)), max(2, ticks(self.p.t_final))
        self.swing_h = self.p.swing_height * leg
        self.lim = StepLimits(forward=self.p.step_forward * leg * g, backward=self.p.step_backward * leg * g,
                              lateral=self.p.step_lateral * leg * g, turn=np.deg2rad(self.p.turn_deg) * g,
                              width=self.width, min_width=self.p.min_width_frac * self.width,
                              lookahead=0.45 * leg, blend=0.75 * leg)
        lg = self.geo.legs["left"]
        self.mpc = LipmMPC(self.z_c, self.p.mpc_T, self.p.mpc_N, self.p.weights,
                           half_length=self.p.sole_length_frac * lg.half_length,
                           half_width=self.p.sole_width_frac * lg.half_width,
                           reach_fwd=self.p.reach_forward * leg, reach_back=self.p.reach_backward * leg,
                           width_min=self.lim.min_width, width_max=self.width + self.p.reach_lateral * leg)
        self.n_mpc = max(1, int(round(self.p.mpc_period / self.control_dt)))
        self._reset_state()

    # ------------------------------------------------------------------ API
    @staticmethod
    def supports(model: mujoco.MjModel) -> bool:
        return model_supported(model)

    @property
    def walk_speed(self) -> float:
        """Nominal straight-line speed (m/s)."""
        return self.lim.forward / ((self.n_ss + self.n_ds) * self.control_dt)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def feet(self) -> dict:
        return self._feet

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Knees-bent standing pose over the current footprints (writes the root, abdomen and leg qpos, zeroes
        qvel, writes the leg/abdomen ctrl and calls mj_forward); arms and hands keep their angles."""
        self._reset_state()
        b = self.body
        mujoco.mj_forward(model, data)
        feet = {s: np.array(b.foot_pose(data, s)) for s in (LEFT, RIGHT)}
        mid = (feet[LEFT][:2] + feet[RIGHT][:2]) / 2
        yaw = mean_yaw(feet[LEFT][2], feet[RIGHT][2])
        b.scratch.qpos[:] = data.qpos
        q, p_t, _ = b.whole_body(mid, yaw, b.targets(feet))
        data.qpos[0:3] = p_t
        mujoco.mju_mat2Quat(data.qpos[3:7], rot_z(yaw).ravel())
        data.qpos[b.qadr] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        self._feet = feet
        data.ctrl[self.act] = b.clip_ctrl(model, q)

    def set_path(self, path_xy, final_yaw: float | None = None) -> None:
        """Walk along world waypoints (n, 2) and stop with the torso on the last one facing final_yaw (rad;
        None = arrival direction). None / empty = stop and stand. ValueError for non-finite input."""
        pts = None if path_xy is None else np.asarray(path_xy, float).reshape(-1, 2)
        if (pts is not None and not np.isfinite(pts).all()) or (final_yaw is not None
                                                                 and not np.isfinite(final_yaw)):
            raise ValueError("set_path needs finite waypoints and final yaw")
        self._vel = None
        if pts is None or len(pts) == 0:
            self._pending = ("stop", None, None, False)
            self.goal_xy = None
            if self._mode != "fallen":
                self._mode, self.message = "stand", "stopping"
        else:
            self._pending = ("path", pts, final_yaw, True)
            self.corrections, self.arrivals = 0, []
            self._mode, self.message = "walk", "walking"
        if self.swing is None:
            self._apply_pending()

    def set_velocity(self, vx: float, vy: float = 0.0, wz: float = 0.0) -> None:
        """Continuous walking: forward / left speed (m/s) and yaw rate (rad/s) in the robot frame, clipped to the
        step limits; (0, 0, 0) = stop. ValueError for non-finite values."""
        if not np.isfinite([vx, vy, wz]).all():
            raise ValueError("set_velocity needs finite vx, vy, wz")
        if abs(vx) + abs(vy) + abs(wz) < 1e-6:
            self.set_path(None)
            return
        self.goal_xy = None
        self._vel = (float(vx), float(vy), float(wz))
        self._mode, self.message = "walk", "velocity command"
        if self.swing is None:
            self._pending = ("velocity", None, None, False)
            self._apply_pending()

    def status(self) -> dict:
        """{"mode", "message", "goal_error_m", "heading_error_deg", "footsteps": [[x, y, yaw, side], ...]
        still to be taken (the one in the air first; MPC-adapted positions where the MPC has moved them),
        "support": "left"|"right"|"double", "mpc": {"solve_ms", "iterations", "horizon_s", "status", "native",
        "zmp": [x, y], "dcm": [x, y], "steps_adapted_m"}}"""
        steps = ([[*self.swing.target.round(4).tolist(), self.swing.side]] if self.swing is not None else [])
        adapted = self._adapted
        for i, st in enumerate(self.queue):
            xy = adapted.get(self.qkeys[i], st[:2]) if i < len(self.qkeys) else st[:2]
            steps.append([round(float(xy[0]), 4), round(float(xy[1]), 4), round(float(st[2]), 4), int(st[3])])
        has_goal = self.goal_xy is not None
        pl = self.plan
        return {"mode": self._mode, "message": self.message, "footsteps": steps, "support": self.support,
                "goal_error_m": float(self.goal_err) if has_goal else None,
                "heading_error_deg": float(np.rad2deg(self.yaw_err)) if has_goal else None,
                "mpc": {"solve_ms": round(self.solve_ms, 3), "iterations": int(pl.iterations) if pl else 0,
                        "horizon_s": self.p.mpc_N * self.p.mpc_T, "status": int(pl.status) if pl else 0,
                        "native": bool(pl.native) if pl else False,
                        "zmp": [round(float(v), 4) for v in self.zmp_ref],
                        "dcm": [round(float(v), 4) for v in self.dcm_meas],
                        "steps_adapted_m": round(float(self.max_adapt), 4)}}

    # --------------------------------------------------------------- state
    def _reset_state(self) -> None:
        self._mode, self.message = "stand", "standing"
        self.queue: list[list[float]] = []
        self.qkeys: list[int] = []
        self.qweights: list[float] = []
        self.swing: _Swing | None = None
        self.phase, self.phase_left = "ds", 0
        self.starting = False
        self.support = "double"
        self._feet = {LEFT: np.zeros(3), RIGHT: np.zeros(3)}
        self.fkey = {LEFT: 0, RIGHT: 1}
        self._next_key = 2
        self.last_side = RIGHT
        self.goal_xy, self.goal_yaw = None, 0.0
        self.goal_err, self.yaw_err = 0.0, 0.0
        self.corrections = 0
        self.arrivals: list[tuple[float, float]] = []
        self.settle = 0
        self._pending = None
        self._vel = None
        self.plan: Plan | None = None
        self.plan_tick = 0
        self.solve_ms = 0.0
        self.solve_log: list[tuple[float, int]] = []    # (ms, iterations) of every solve
        self._adapted: dict[int, np.ndarray] = {}       # footstep key -> MPC position
        self._need_solve = True
        self.zmp_ref = np.zeros(2)
        self.dcm_meas = np.zeros(2)
        self.max_adapt = 0.0
        self._recover_until = 0
        self._ds_ext = 0
        self.retimed = 0
        self.qp_failures = 0
        self.q_prev = self.qd_prev = None
        self._jump = True
        self.ticks = 0
        self.body.reset_filters()
        self.mpc.reset_warm_start()

    def _new_key(self) -> int:
        self._next_key += 1
        return self._next_key

    # ---------------------------------------------------- footstep schedule
    def _set_queue(self, steps: list, closing: bool) -> None:
        self.queue = [list(st) for st in steps]
        self.qkeys = [self._new_key() for _ in steps]
        n = len(steps)
        self.qweights = [self.p.final_step_weight if (closing and i >= n - 2) else 1.0 for i in range(n)]

    def _apply_pending(self) -> None:
        if self._pending is None:
            return
        kind, pts, fy, shift = self._pending
        self._pending = None
        walking = self.ticks > 0 and self.phase == "ds" and self.phase_left > 0 and not self.starting
        first = (1 - self.last_side) if walking else None
        F = self._feet
        if kind == "path":
            goal = pts[-1].copy()
            if shift:
                fy = goal_heading(F[LEFT], F[RIGHT], pts, fy, self.lim)
                pts = pts.copy()
                pts[-1] -= rot_z(fy)[:2, :2] @ self.stand_offset
            steps, gy = plan_footsteps(F[LEFT], F[RIGHT], pts, fy, self.lim, first_side=first)
            self._set_queue(steps, closing=True)
            self.goal_xy, self.goal_yaw = goal, gy
        elif kind == "stop":
            steps = square_up(F[LEFT], F[RIGHT], self.last_side, self.lim) if walking else (
                plan_footsteps(F[LEFT], F[RIGHT], ((F[LEFT][:2] + F[RIGHT][:2]) / 2)[None], None, self.lim)[0])
            self._set_queue(steps, closing=False)
        elif kind == "velocity" and self._vel is not None:
            steps = velocity_footsteps(F[LEFT], F[RIGHT], 1 - self.last_side, *self._vel,
                                       (self.n_ss + self.n_ds) * self.control_dt, self.lim)
            self._set_queue(steps, closing=False)
        if self.queue and self.phase == "ds" and self.phase_left <= 0:
            self.phase_left = self.n_start               # from standing: slow first weight shift
            self.starting = True
        elif not self.queue and self.starting:           # cancelled before the first lift-off: just stand
            self.starting, self.phase_left = False, 0
        self._need_solve = True

    def _advance_phase(self, meas: Meas) -> None:
        if self.phase == "ss":
            sw = self.swing
            sw.k += 1
            sw.phi = sw.phi_next()
            sw.left -= 1
            if sw.left > 0:
                return
            # touchdown: the foot is where it really landed
            self._feet[sw.side] = meas.feet[sw.side].copy()
            self.fkey[sw.side] = sw.key
            self._adapted.pop(sw.key, None)
            self.last_side = sw.side
            self.swing = None
            self.phase = "ds"
            self.phase_left = self.n_ds if self.queue else self.n_final
            self._ds_ext = 0
            self._jump = True
            self._need_solve = True
            if self._vel is not None and self._pending is None:
                self._pending = ("velocity", None, None, False)
            if self._pending is not None:
                self._apply_pending()
            return
        self.phase_left -= 1
        if self.phase_left > 0 or not self.queue:
            return
        st = self.queue.pop(0)
        key = self.qkeys.pop(0)
        wgt = self.qweights.pop(0)
        side = int(st[3])
        plan = np.array(st[:3])
        target = plan.copy()
        if key in self._adapted:
            target[:2] = self._adapted.pop(key)
        start = self._feet[side].copy()
        self.swing = _Swing(side=side, start=start, target=target, plan=plan, n=self.n_ss, key=key, weight=wgt)
        self.swing.xy[0] = start[:2]
        self.phase = "ss"
        self.starting = False
        self._jump = True
        self._need_solve = True

    # ------------------------------------------------------------ MPC problem
    def _build_problem(self, s0: np.ndarray, t_now: float) -> tuple[Problem, list]:
        """Contact schedule over the horizon -> MPC problem. Returns (problem, [(var, kind, ref)]) where kind is
        "swing" or "queue" and ref the queue key of the footstep each decision variable stands for."""
        dt = self.control_dt
        Hz = self.p.mpc_N * self.p.mpc_T
        F = self._feet
        feet = [Foot(LEFT, F[LEFT][:2].copy(), float(F[LEFT][2]), -1, self.fkey[LEFT]),
                Foot(RIGHT, F[RIGHT][:2].copy(), float(F[RIGHT][2]), -1, self.fkey[RIGHT])]
        cur = {LEFT: 0, RIGHT: 1}
        segs: list = []           # [t0, t1, kind, args..., vref]
        boxes: list = []
        varmap: list = []
        reach = None
        t = 0.0
        last = self.last_side

        def land(side, xy, yaw, key, t_land, weight, fixed=False):
            nv = len(varmap)
            if not fixed and t_land <= Hz + 1e-9 and nv < self.p.max_footsteps:
                f = Foot(side, np.asarray(xy, float).copy(), float(yaw), nv, key, weight)
                boxes.append(StepBox(nv, cur[1 - side], side))
                varmap.append((nv, key))
            else:
                f = Foot(side, np.asarray(xy, float).copy(), float(yaw), -1, key, weight)
            feet.append(f)
            cur[side] = len(feet) - 1
            return f

        def mid():
            return (feet[cur[LEFT]].xy + feet[cur[RIGHT]].xy) / 2

        T_step = (self.n_ds + self.n_ss) * dt
        if self.phase == "ss":
            sw = self.swing
            tr = sw.left * dt
            m0 = mid()
            tr_h = sw.time_left(dt, 0.9)                 # horizontal swing time left
            locked = tr_h < self.p.lock_time
            f = land(sw.side, sw.target[:2] if locked else sw.plan[:2], sw.target[2], sw.key, tr, sw.weight,
                     fixed=locked)
            vref = (mid() - m0) / T_step
            segs.append([0.0, tr, "ss", cur[1 - sw.side], vref])
            if f.var >= 0:
                reach = (f.var, sw.xy[0].copy(), self.p.swing_speed * max(tr_h - self.p.lock_time, 0.0) + 1e-3)
            t, last = tr, sw.side
            first_ds = self.n_ds * dt if self.queue else None
        else:
            first_ds = self.phase_left * dt
        for i, st in enumerate(self.queue):
            if t > Hz:
                break
            side = int(st[3])
            ds = first_ds if i == 0 else self.n_ds * dt
            m0 = mid()
            frm, to = cur[side], cur[1 - side]
            seg_ds = [t, t + ds, "start" if (i == 0 and self.phase == "ds" and self.starting) else "ds",
                      frm, to, cur[LEFT], cur[RIGHT]]
            t += ds
            seg_ss = [t, t + self.n_ss * dt, "ss", to]
            t += self.n_ss * dt
            key = self.qkeys[i]
            land(side, st[:2], st[2], key, t, self.qweights[i])
            vref = (mid() - m0) / T_step
            seg_ds.append(np.zeros(2) if seg_ds[2] == "start" else vref)
            seg_ss.append(vref)
            segs += [seg_ds, seg_ss]
            last = side
        if t <= Hz:
            if self.phase == "ds" and not self.queue:
                tf = self.phase_left * dt                # final ramp in progress (0 = standing)
            else:
                tf = self.n_final * dt
            if tf > 0:
                segs.append([t, t + tf, "final", cur[1 - last], cur[LEFT], cur[RIGHT], np.zeros(2)])
                t += tf
            segs.append([t, np.inf, "stand", cur[LEFT], cur[RIGHT], np.zeros(2)])

        def support_at(tau: float) -> Support:
            sg = segs[-1]
            for cand in segs:
                if tau < cand[1]:
                    sg = cand
                    break
            kind, vref = sg[2], sg[-1]
            w = 0.0 if not np.isfinite(sg[1]) else float(np.clip((tau - sg[0]) / max(sg[1] - sg[0], 1e-9), 0, 1))
            if kind == "ss":
                return Support("single", (sg[3],), ((sg[3], 1.0),), vref)
            if kind in ("ds", "start"):
                frm, to, il, ir = sg[3], sg[4], sg[5], sg[6]
                if abs(tau - sg[1]) < 1e-6:              # the lift-off instant: on the new stance sole
                    return Support("single", (to,), ((to, 1.0),), vref)
                ref = (((il, 0.5 * (1 - w)), (ir, 0.5 * (1 - w)), (to, w)) if kind == "start"
                       else ((frm, 1 - w), (to, w)))
                if feet[frm].var < 0 and feet[to].var < 0:
                    return Support("hull", (frm, to), ref, vref)
                if self.p.grid == "contact":             # bounded at both ends of the transit (samples
                    return Support("free", (), ref, vref)    # sit on the touchdown and the lift-off)
                return Support("single", (frm,) if w < 0.5 else (to,), ref, vref)   # uniform grid: half-way rule
            if kind == "final":
                frm, il, ir = sg[3], sg[4], sg[5]
                ref = ((frm, 1 - w), (il, 0.5 * w), (ir, 0.5 * w))
            else:
                il, ir = sg[3], sg[4]
                frm = None
                ref = ((il, 0.5), (ir, 0.5))
            if feet[il].var < 0 and feet[ir].var < 0:
                return Support("hull", (il, ir), ref, vref)
            if kind == "final" and w < 0.5:
                return Support("single", (frm,), ref, vref)
            return Support("strip", (il, ir), ref, vref)

        bounds = self._grid(segs, Hz) if self.p.grid == "contact" else self.p.mpc_T * np.arange(1, self.p.mpc_N + 1)
        # samples at the interval ends: an interval never straddles a touchdown or lift-off, so the piecewise
        # constant jerk can hold the ZMP on one sole and then carry it across in double support
        supports = [support_at(float(b) - 1e-9) for b in bounds]
        # extra constraint instants (no extra variables): the middle of every interval
        mids = (np.concatenate([[0.0], bounds[:-1]]) + bounds) / 2
        points = [(float(t), support_at(float(t))) for t in mids if t >= self.p.point_min_s]
        points = [(t, sp) for t, sp in points if sp.kind != "free"] if self.p.grid == "contact" else []
        return Problem(s0, feet, supports, boxes, reach, self.ticks * dt, np.diff(np.concatenate([[0.0], bounds])),
                       points), varmap

    def _grid(self, segs: list, Hz: float) -> np.ndarray:
        """Ends of the jerk intervals (s from now): every contact switch in the horizon is a boundary, the
        phases in between are split into intervals of about mpc_T. Before the first switch the intervals are
        laid back from it (or, when standing, on multiples of mpc_T of the controller clock), so re-solving
        every mpc_period keeps the same constraint instants and only the first interval shrinks (T/2..3T/2)."""
        T, dt = self.p.mpc_T, self.control_dt
        t_min = self.p.point_min_s
        events = sorted({float(sg[0]) for sg in segs[1:] if t_min <= sg[0] < Hz - t_min})
        if events:
            first = events[0]
            out = [first - k * T for k in range(int(np.floor((first - T / 2) / T)), 0, -1)]
        else:
            n_T = max(1, int(round(T / dt)))
            k1 = n_T - (self.ticks % n_T)
            if k1 < n_T / 2:
                k1 += n_T
            first = k1 * dt
            out = []
        out = [b for b in out if b >= t_min]
        out.append(first)
        for a, b in zip(events, events[1:] + [Hz]):
            n = max(1, int(round((b - a) / T)))
            out += list(a + (b - a) * np.arange(1, n + 1) / n)
        if not events:
            while out[-1] < Hz - T / 2:
                out.append(out[-1] + T)
        return np.array(out)

    def _solve(self, meas: Meas, t_now: float) -> None:
        """Re-solve the MPC from the measured CoM state (blended with the current plan by `anchor`)."""
        c, v = meas.com[:2], meas.com_vel[:2]
        if self.plan is not None:
            ref = self.plan.state((self.ticks - self.plan_tick) * self.control_dt)
            a = self.p.anchor
            av = a if self.p.anchor_vel is None else self.p.anchor_vel
            c = ref[0] + a * (c - ref[0])
            v = ref[1] + av * (v - ref[1])
            z_now = ref[0] - ref[2] / self.omega ** 2
        else:
            z_now = c.copy()
        s0 = np.array([c, v, self.omega ** 2 * (c - z_now)])
        if self.plan is not None and self.p.timing_adapt:
            # measured DCM vs the plan: a push (or a slip) starts a short recovery window with step timing
            ref = self.plan.state((self.ticks - self.plan_tick) * self.control_dt)
            err = (meas.com[:2] + meas.com_vel[:2] / self.omega) - (ref[0] + ref[1] / self.omega)
            if float(np.linalg.norm(err)) > self.p.timing_trigger:
                self._recover_until = self.ticks + int(round(self.p.timing_window / self.control_dt))
        t0 = time.perf_counter()
        best = None
        options = self._timing_options() if self.ticks < self._recover_until else [("none", 0)]
        for kind, val in options:
            undo = self._apply_timing(kind, val)
            prob, varmap = self._build_problem(s0, t_now)
            plan = self.mpc.solve(prob, force_python=self.p.qp_backend == "numpy")
            shift = (undo or 0) * self.control_dt              # seconds added (ds) or removed (ss)
            cost = plan.cost + self.p.timing_weight * shift ** 2
            if plan.status != 0:
                cost += 1e6
            if best is None or cost < best[0]:
                best = (cost, kind, val, prob, varmap, plan, list(self.mpc._warm_keys))
            self._undo_timing(kind, undo)
        _, kind, val, prob, varmap, plan, warm = best
        self._apply_timing(kind, val)
        if kind != "none":
            self.retimed += 1
        self.mpc._warm_keys = warm
        if len(options) > 1:
            plan.solve_ms = (time.perf_counter() - t0) * 1e3
        self._last_prob = prob
        if plan.status != 0 and self.plan is not None and self.swing is not None:
            # no usable optimum: keep the previous landing point rather than a half-solved one
            varmap = [(vv, key) for vv, key in varmap if key != self.swing.key]
        self.plan, self.plan_tick = plan, self.ticks
        self.solve_ms = plan.solve_ms
        self.solve_log.append((plan.solve_ms, plan.iterations))
        if plan.status != 0:
            self.qp_failures += 1
        self._adapted = {}
        for var, key in varmap:
            xy = plan.steps[var]
            if self.swing is not None and key == self.swing.key:
                dev = xy - self.swing.plan[:2]
                if np.linalg.norm(dev) < self.p.step_deadband:
                    xy = self.swing.plan[:2]
                self.swing.target[:2] = xy
                self.max_adapt = max(self.max_adapt, float(np.linalg.norm(xy - self.swing.plan[:2])))
            else:
                self._adapted[key] = xy.copy()
        self._need_solve = False
        self._jump = True

    def _timing_options(self) -> list:
        """Contact-timing candidates evaluated during a recovery (Khadiv et al., "Walking control based on step
        timing adaptation", IEEE T-RO 2020, does this inside one QP; here a few discrete options, one MPC solve
        each): stay longer in double support (the DCM is still over the foot about to lift), or put the swing
        foot down earlier."""
        dt = self.control_dt
        opts = [("none", 0)]
        if self.phase == "ds" and self.queue and not self.starting and self.phase_left > 0:
            room = int(round(self.p.max_ds_extension / dt)) - self._ds_ext
            for ext in self.p.ds_extensions:
                k = int(round(ext / dt))
                if 0 < k <= room:
                    opts.append(("ds", k))
        elif self.phase == "ss" and self.swing is not None:
            sw = self.swing
            if sw.phi >= self.p.min_swing_progress:
                for cut in self.p.swing_cuts:
                    left = sw.left - int(round(cut / dt))
                    if left * dt >= self.p.min_swing_left:
                        opts.append(("ss", left))
        return opts

    def _apply_timing(self, kind: str, val: int):
        """Apply a timing option; returns what _undo_timing needs (for "ss": the ticks removed)."""
        if kind == "ds":
            self.phase_left += val
            self._ds_ext += val
            return val
        if kind == "ss":
            old = self.swing.left
            self.swing.left = val
            return old - val
        return None

    def _undo_timing(self, kind: str, undo) -> None:
        if kind == "ds":
            self.phase_left -= undo
            self._ds_ext -= undo
        elif kind == "ss":
            self.swing.left += undo

    # ------------------------------------------------------------ swing foot
    def _swing_pose(self, sw: _Swing) -> tuple[np.ndarray, np.ndarray]:
        """Horizontal: a quintic re-planned every tick from the current reference (position, velocity,
        acceleration) to the MPC landing point with zero end velocity / acceleration, between 5 % and 90 % of
        the swing. Vertical: quartic bump of height swing_h. Yaw: minimum-jerk from lift-off to the plan."""
        dt = self.control_dt
        phi = sw.phi_next()
        p = sw.xy
        if phi > 0.05:
            Tr = sw.time_left(dt, 0.9)
            if Tr > dt:
                p0, v0, a0 = p
                h = sw.target[:2] - p0
                T2, T3 = Tr * Tr, Tr ** 3
                c3 = (20 * h - 12 * v0 * Tr - 3 * a0 * T2) / (2 * T3)
                c4 = (-30 * h + 16 * v0 * Tr + 3 * a0 * T2) / (2 * T3 * Tr)
                c5 = (12 * h - 6 * v0 * Tr - a0 * T2) / (2 * T3 * T2)
                t = dt
                p = np.array([p0 + v0 * t + a0 * t * t / 2 + c3 * t ** 3 + c4 * t ** 4 + c5 * t ** 5,
                              v0 + a0 * t + 3 * c3 * t * t + 4 * c4 * t ** 3 + 5 * c5 * t ** 4,
                              a0 + 6 * c3 * t + 12 * c4 * t * t + 20 * c5 * t ** 3])
            else:
                p = np.array([sw.target[:2], np.zeros(2), np.zeros(2)])
            sw.xy = p
        u = min(1.0, max(0.0, (phi - 0.05) / 0.85))
        s = u * u * u * (10 - 15 * u + 6 * u * u)
        yaw = sw.start[2] + wrap(sw.target[2] - sw.start[2]) * s
        z = self.swing_h * 16 * phi * phi * (1 - phi) ** 2
        return np.array([p[0][0], p[0][1], z]), rot_z(float(yaw))

    # ----------------------------------------------------------- main loop
    def step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """One control tick: writes data.ctrl of the abdomen and leg actuators only."""
        b = self.body
        mujoco.mj_forward(model, data)
        meas = b.measure(model, data)
        self.ticks += 1
        if meas.fallen:
            if self._mode != "fallen":
                self._mode, self.message = "fallen", "robot fell"
                self.queue, self.qkeys, self.qweights, self.swing, self._pending = [], [], [], None, None
            return
        self._advance_phase(meas)
        if self._need_solve or self.plan is None or (self.ticks - self.plan_tick) >= self.n_mpc:
            self._solve(meas, float(data.time))
        ref = self.plan.state((self.ticks - self.plan_tick) * self.control_dt)
        c_ref, v_ref, a_ref = ref
        self.zmp_ref = c_ref - a_ref / self.omega ** 2
        xi_meas = meas.com[:2] + meas.com_vel[:2] / self.omega
        self.dcm_meas = xi_meas
        xi_err = xi_meas - (c_ref + v_ref / self.omega)

        targets = b.targets(self._feet)
        sw = self.swing
        if sw is not None:
            targets[sw.side] = self._swing_pose(sw)
            self.support = "right" if sw.side == LEFT else "left"
            stance = [1 - sw.side]
        else:
            self.support = "double"
            stance = [LEFT, RIGHT]
        b.scratch.qpos[:] = data.qpos
        yaw_ref = mean_yaw(yaw_of(targets[LEFT][1]), yaw_of(targets[RIGHT][1]))
        q_ref, _, _ = b.whole_body(c_ref, yaw_ref, targets, key="ref")
        q, p_t, yaw_t = q_ref.copy(), None, yaw_ref
        if self.p.track_stance:
            for s in stance:
                targets[s] = (np.array([meas.feet[s][0], meas.feet[s][1], 0.0]), rot_z(float(meas.feet[s][2])))
            yaw_t = mean_yaw(yaw_of(targets[LEFT][1]), yaw_of(targets[RIGHT][1]))
        q, p_t, _ = b.whole_body(c_ref, yaw_t, targets, key="meas")
        b.swing_leg_world(q, sw.side if sw is not None else None, targets[sw.side] if sw is not None else None,
                          p_t, yaw_t, meas, self.p.swing_world)
        qd, qdd = self._derivatives(q_ref)
        q = b.torso_feedback(q, meas, yaw_t, stance, self.p.k_torso)
        acc_t = np.array([a_ref[0], a_ref[1], 0.0])
        tau = b.feedforward(model, data, meas, stance, acc_t, qdd, self.zmp_ref, self.p.k_dcm * xi_err)
        kp = np.maximum(model.actuator_gainprm[self.act, 0], 1e-6)
        kv = -model.actuator_biasprm[self.act, 2]
        data.ctrl[self.act] = b.clip_ctrl(model, q + (kv / kp) * qd + tau / kp)
        self._update_status(data, meas)

    def _derivatives(self, q_ref: np.ndarray):
        """Joint velocity / acceleration of the reference by finite differences, held over the ticks where the
        reference jumps (a re-solve re-anchors it on the measured state; a touchdown moves a foot)."""
        dt = self.control_dt
        if self.q_prev is None or self._jump:
            qd = self.qd_prev if self.qd_prev is not None else np.zeros(14)
            qdd = np.zeros(14)
        else:
            qd = (q_ref - self.q_prev) / dt
            qdd = (qd - self.qd_prev) / dt if self.qd_prev is not None else np.zeros(14)
        self._jump = False
        self.q_prev, self.qd_prev = q_ref.copy(), qd.copy()
        return qd, np.clip(qdd, -100.0, 100.0)

    def _update_status(self, data, meas: Meas) -> None:
        x, y = float(data.qpos[0]), float(data.qpos[1])
        yaw = float(np.arctan2(meas.R_torso[1, 0], meas.R_torso[0, 0]))
        if self.goal_xy is not None:
            self.goal_err = float(np.hypot(x - self.goal_xy[0], y - self.goal_xy[1]))
            self.yaw_err = float(wrap(yaw - self.goal_yaw))
        idle = self.swing is None and not self.queue and self.phase_left <= 0 and self._pending is None
        if not idle or self._vel is not None:
            self.settle = 0
            return
        if self._mode == "stand" and self.message == "stopping":
            self.message = "standing"
        if self._mode != "walk":
            return
        self.settle = self.settle + 1 if np.linalg.norm(meas.com_vel[:2]) < 0.03 else 0
        if self.settle * self.control_dt < 0.3:
            return
        self.settle = 0
        if self.goal_xy is None:
            self._mode, self.message = "stand", "standing"
            return
        self.arrivals.append((self.goal_err, float(np.rad2deg(self.yaw_err))))
        if self.corrections < self.p.max_corrections and (
                self.goal_err > self.p.goal_tolerance or abs(self.yaw_err) > np.deg2rad(self.p.yaw_tolerance_deg)):
            self.corrections += 1
            goal = self.goal_xy.copy()
            tgt = (self._feet[LEFT][:2] + self._feet[RIGHT][:2]) / 2 + (goal - np.array([x, y]))
            self._pending = ("path", tgt[None], self.goal_yaw, False)
            self._apply_pending()
            self.goal_xy = goal
            self.message = f"correction step {self.corrections}"
            return
        self._mode = "done"
        self.message = f"goal reached ({self.goal_err:.3f} m, {np.rad2deg(self.yaw_err):.1f} deg)"


# --------------------------------------------------------------------------- headless runs
def _push_list(push) -> list[dict]:
    """Pushes as dicts; accepts {"t", "force", "duration", "body"}, the walker's (t, force, duration) tuple, or a
    list of either."""
    if push is None:
        return []
    if isinstance(push, dict) or (isinstance(push, (tuple, list)) and len(push) == 3 and np.ndim(push[0]) == 0):
        push = [push]
    out = []
    for p in push:
        if not isinstance(p, dict):
            p = {"t": p[0], "force": p[1], "duration": p[2]}
        f = np.zeros(3)
        fv = np.asarray(p.get("force", (0.0, 0.0, 0.0)), float).ravel()
        f[:fv.size] = fv[:3]
        q = {"t": float(p["t"]), "force": f, "duration": float(p.get("duration", 0.1)),
             "body": str(p.get("body", "torso"))}
        if not (np.isfinite(q["t"]) and np.isfinite(f).all() and q["duration"] >= 0):
            raise ValueError("a push needs a finite time, force and non-negative duration")
        out.append(q)
    return out


def run_episode(model: mujoco.MjModel, controller, goal_xy, goal_yaw: float | None = None, timeout: float = 30.0,
                path=None, push=None, tolerance: float = 0.1, settle_s: float = 0.5) -> dict:
    """Walk any Controller (MPCWalkController, WalkController, ...) headless from its standing pose to goal_xy,
    optionally pushing a body: push = {"t": s after the walk command, "force": (fx, fy, fz) N world frame,
    "duration": s (0.1), "body": "torso"}, or walker-style (t, force, duration), or a list of them (forces via
    data.xfrc_applied, i.e. at the body's centre of mass).

    Returns {"reached", "error_m", "heading_error_deg", "time_s", "fell", "energy_J", "cost_of_transport",
    "peak_torque_frac", "distance_m", "speed_mps", "corrections", "arrivals", "status"} like
    walker.simulate_walk, plus "pushes" (as applied)."""
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ctrl = controller
    ctrl.reset(model, data)
    n_sub = max(1, int(round(ctrl.control_dt / model.opt.timestep)))
    dt = n_sub * float(model.opt.timestep)
    for _ in range(int(round(settle_s / dt))):
        ctrl.step(model, data)
        mujoco.mj_step(model, data, n_sub)
    start = data.qpos[:2].copy()
    goal = np.asarray(goal_xy, float).reshape(2)
    pts = np.asarray(path, float).reshape(-1, 2) if path is not None else np.array([start, goal])
    if len(pts) == 0 or np.linalg.norm(pts[-1] - goal) > 1e-3:
        pts = np.vstack([pts, goal])
    ctrl.set_path(pts, goal_yaw)
    pushes = _push_list(push)
    pbody = {p["body"]: model.body(p["body"]).id for p in pushes}
    act = np.asarray(ctrl.act)
    dof = np.asarray(ctrl.dadr)
    lim = np.abs(model.actuator_forcerange[act]).max(axis=1)
    lim = np.where(model.actuator_forcelimited[act].astype(bool) & (lim > 0), lim, np.inf)
    energy = peak = t = dist = 0.0
    t0 = data.time
    feet_mid = lambda: (ctrl.feet[0][:2] + ctrl.feet[1][:2]) / 2   # noqa: E731
    prev = feet_mid()
    mode = lambda: ctrl.status()["mode"]   # noqa: E731
    while t < timeout and mode() not in ("done", "fallen"):
        ctrl.step(model, data)
        for _ in range(n_sub):
            if pushes:
                data.xfrc_applied[list(pbody.values()), :] = 0.0
                tt = data.time - t0
                for p in pushes:
                    if p["t"] <= tt < p["t"] + p["duration"]:
                        data.xfrc_applied[pbody[p["body"]], :3] += p["force"]
            mujoco.mj_step(model, data)
            frc = data.actuator_force[act]
            energy += float(np.abs(frc * data.qvel[dof]).sum()) * model.opt.timestep
            peak = max(peak, float((np.abs(frc) / lim).max()))
        t += dt
        dist += float(np.linalg.norm(feet_mid() - prev))
        prev = feet_mid()
    if pushes:
        data.xfrc_applied[list(pbody.values()), :] = 0.0
    err = float(np.hypot(*(data.qpos[:2] - goal)))
    R = data.xmat[ctrl.torso_id].reshape(3, 3)
    want = ctrl.goal_yaw if goal_yaw is None else goal_yaw
    herr = float(np.rad2deg(wrap(np.arctan2(R[1, 0], R[0, 0]) - want)))
    st = ctrl.status()
    return {"reached": bool(st["mode"] == "done" and err <= tolerance), "error_m": err,
            "heading_error_deg": herr, "time_s": t, "fell": st["mode"] == "fallen", "energy_J": energy,
            "cost_of_transport": energy / (ctrl.mass * G * max(dist, 1e-6)), "peak_torque_frac": peak,
            "distance_m": dist, "speed_mps": dist / max(t, 1e-6), "corrections": ctrl.corrections,
            "arrivals": list(ctrl.arrivals), "status": st,
            "pushes": [{**p, "force": p["force"].tolist()} for p in pushes]}


def simulate_mpc_walk(model: mujoco.MjModel, goal_xy, goal_yaw: float | None = None, timeout: float = 30.0,
                      path=None, push=None, tolerance: float = 0.1, params: MPCParams | None = None,
                      settle_s: float = 0.5) -> dict:
    """Walk a fresh copy of the robot headless with the MPC controller (see run_episode for push and the
    result keys; mirrors walker.simulate_walk). Adds "mpc": {"solves", "solve_ms_mean", "solve_ms_p95",
    "solve_ms_max", "iterations_mean", "iterations_max", "native", "max_step_adaptation_m", "retimed" (timing
    changes chosen), "qp_failures" (solves without an optimal QP status)}.
    Raises ValueError for unsupported models (check MPCWalkController.supports first)."""
    ctrl = MPCWalkController(model, params)
    r = run_episode(model, ctrl, goal_xy, goal_yaw, timeout, path, push, tolerance, settle_s)
    log = np.array(ctrl.solve_log) if ctrl.solve_log else np.zeros((1, 2))
    r["mpc"] = {"solves": len(ctrl.solve_log), "solve_ms_mean": float(log[:, 0].mean()),
                "solve_ms_p95": float(np.percentile(log[:, 0], 95)), "solve_ms_max": float(log[:, 0].max()),
                "iterations_mean": float(log[:, 1].mean()), "iterations_max": int(log[:, 1].max()),
                "native": bool(ctrl.plan.native) if ctrl.plan is not None else False,
                "max_step_adaptation_m": float(ctrl.max_adapt), "retimed": int(ctrl.retimed),
                "qp_failures": int(ctrl.qp_failures)}
    return r
