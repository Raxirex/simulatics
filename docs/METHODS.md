# Methods, and where they are in Simulatics

Each method below comes from the published literature; the reference column cites the paper (authors and
year). Simulatics is an independent implementation: no code is taken from the cited works, and citing a
paper does not mean its authors or their institutions are involved with or endorse this project.

The stack is layered, so each method plugs into one place:

```
  AI layer          LLM / VLM pilot: vision, task planning, tool calls               0.1-2 Hz
  planning          A* path on an occupancy map, behaviour tree, footstep planner    1-10 Hz
  locomotion        model-based walker (ZMP/DCM)  |  learned policy (PPO / distilled)  50-200 Hz
  manipulation      arm IK skills: reach, grasp, point, gestures                     50 Hz
  joints            position servos (PD) with torque limits, in the physics engine   200 Hz
```

Status: **built** = in the app and tested; **partial** = works with limits listed; **not built** = and why.

## Locomotion

| Method | Reference | In Simulatics | Status |
|---|---|---|---|
| ZMP preview control on the linear inverted pendulum | Kajita et al. 2003 | `control/walk_preview.py`, Riccati gains via scipy | built |
| Capture point / DCM balance feedback, CoP clamp | Pratt et al. 2006; Englsberger et al. 2015 | `control/walker.py` stabiliser | built |
| Footstep planning along a path, closing step, turn in place | Chestnutt et al. 2005 | `control/walk_footsteps.py` | built |
| Analytic 6-DOF leg IK from the compiled model | Kajita et al., *Introduction to Humanoid Robotics* | `control/walk_ik.py` (FK round-trip error ~1e-15) | built |
| Torso hip strategy, floating-base inverse-dynamics feed-forward | standard practice | `control/walker.py` | built |
| Closed-loop arrival with correction steps | standard replanning | walker re-plans from landed feet; 5-6 mm mean error over 320 random bodies and goals | built |
| Linear MPC with footstep and step-timing adaptation | Wieber 2006; Herdt et al. 2010; Khadiv et al. 2020 | `control/mpc*.py` with its own dual active-set QP solver (numpy + C++) | built: 0.2-0.5 cm at the goal; push recovery equal to the walker |
| Convex single-rigid-body force MPC + whole-body impulse control | Di Carlo et al. 2018; Kim et al. 2019 | not built: needs torque-controlled joints and a whole-body QP; the position-servo robot uses inverse-dynamics feed-forward instead | not built |
| Push recovery: DCM-based step location and timing adaptation | Khadiv et al. 2020; Griffin et al. 2017 | `control/walk_recovery.py` | built: sideways 60 N 15/16, standing 140 N 8/8; 100 N sideways 8/16 |
| Rough-terrain footstep placement (foothold ray casts, edge avoidance, swing clearance) | standard practice | `control/walk_terrain.py`, `sim/terrain.py` | built: steps, stairs (6-8 cm), 8 deg ramps |
| Mixed-integer footstep planning over convex regions, graphs of convex sets | Deits and Tedrake 2014; Marcucci et al. 2023 | not built: needs a MIQP solver | not built |
| Massively parallel PPO velocity tracking | Schulman et al. 2017; Rudin et al. 2021 | `train/ppo.py`, `sim/vel_task.py`, `sim/vec_env.py`: numpy PPO + GAE + KL-adaptive LR, velocity-tracking rewards, domain randomisation, threaded physics | partial: ~24k physics steps/s on CPU; from scratch it needs GPU-scale training, so the shipped walker comes from distillation; BC-regularised PPO fine-tuning is implemented but did not help on CPU |
| Teacher-student distillation (BC + DAgger) | Ross et al. 2011; Lee et al. 2020; Kumar et al. 2021 | `train/ppo.py distill`: any controller, including the model-based walker, as teacher; gait-clock and history inputs | built: `runs/rl_walker` reaches 40/40 held-out goals, 5 cm median error |
| Asymmetric critic, observation history | Pinto et al. 2018 | `VelTaskConfig.asymmetric_critic`, `history` | built |
| Evolution strategies / GA, body + controller co-evolution | Salimans et al. 2017; Gupta et al. 2021 | `train/evolve.py` | built |
| Motion imitation / tracking with kinematic retargeting | Peng et al. 2018 | `imitate/`: BVH import and retargeting, reference clips, tracking task, PPO trainer, controller | built: walk tracker trained on CPU, 25 s without falls; one policy per clip; AMP/adversarial priors not built |
| Vision-language-action policies, world models | (survey literature) | not built: cluster-scale training; the LLM pilot covers the language-and-vision layer instead | not built |

## Perception, planning and the AI layer

| Method | Reference | In Simulatics | Status |
|---|---|---|---|
| Layered "slow VLM planner, fast controller" | (survey literature) | the LLM pilot calls skills; walker and arm IK run underneath | built |
| LLM tool calling over a skill library | Ahn et al. 2022; Liang et al. 2023 | `llm/tools.py`: look, scan, list_objects, walk_to(_object), walk_relative, reach(_object), grasp, release, point_at, gesture, wait, finish | built |
| Closed-loop text + image feedback after every action | Huang et al. 2022 | every motion tool returns the outcome plus fresh pose, navigation and arm status; `look` returns a JPEG | built |
| Semantic object list from segmentation | Huang et al. 2023; Yang et al. 2023 | `server/robot_adapter.py`: named, coloured objects; visibility from head and work-camera segmentation renders | built |
| Safety layer: argument validation, turn / cost / time budgets, stop | standard practice | `llm/tools.py`, `llm/mission.py` | built |
| Ray-cast lidar, depth and RGB cameras | standard practice | `sim/sensors.py` (`mj_multiRay`), offscreen renderer | built |
| Occupancy mapping from lidar (log-odds) | Moravec and Elfes 1985 | `plan/grid.py` + C++ `simcore` | built |
| A* with inflation, close-in final approach | Hart et al. 1968 | `plan/grid.py`, C++ `simcore` | built |
| Behaviour tree action planner | Colledanchise and Ögren 2018 | `plan/navigator.py` | built |
| 2D lidar SLAM (correlative scan matching, likelihood-field refinement, pose graph) | Olson 2009; Hess et al. 2016; Kohlbrecher et al. 2011 | `plan/slam.py` + C++ `native/src/slam.cpp`; navigation can use the SLAM map and pose | built: 2-6 cm error over 20-26 m vs 1.4-1.7 m odometry; loop closure experimental |
| Elevation mapping, 3D SLAM | Fankhauser et al. 2018 | not built | not built |
| ROS 2 interoperability | rosbridge protocol v2 | `bridge/`: rosbridge v2 client, rclpy backend, standard topics and TF | built; verified against a mock server only |

## Design: CAD, topology and AI optimisation

| Method | Reference | In Simulatics | Status |
|---|---|---|---|
| Parametric CAD parts, mass properties from solids, STL/OBJ/GLB/STEP export | open-source CAD kernels (optional dependencies) | `design/cad.py`; the sim can run on CAD masses and meshes | built (STEP needs build123d) |
| URDF export | URDF format | `design/urdf.py` | built |
| 3D SIMP topology optimisation (OC update, density filter, multigrid-PCG) | Sigmund 2001; Andreassen et al. 2011; Amir et al. 2014 | `design/topopt.py` + C++ `native/src/topopt.cpp` (OpenMP) | built: thigh 29% lighter in ~11 s |
| Loads from simulated gait (joint reaction wrenches) | standard practice | `design/loads.py` (`cfrc_int` during walking) | built |
| CMA-ES and Bayesian optimisation of body and motor genes with simulation in the loop | Hansen 2016; Jones et al. 1998 | `design/optimizer.py`, `design/evaluate.py`: candidates walk to goals in parallel processes | built |
| Actuator / gear-ratio modelling (armature = N^2 I_rotor, torque-speed limits) | Wensing et al. 2017 | armature and torque limits are editable genes; no torque-speed curve yet | partial |
| Differentiable-simulation co-design, RL topology-changing co-design | (survey literature) | not built: research-grade, GPU | not built |

## What to upgrade next, in order

1. **GPU simulation for PPO and imitation** when the disk budget allows PyTorch. The tasks, rewards and
   vectorised-environment interfaces are in place.
2. **Force MPC + whole-body QP** on torque-controlled joints, for harder pushes and dynamic motions.
3. **Adversarial motion priors (AMP)** and multi-clip tracking, so one policy covers walking, turning and gestures.
4. **3D perception**: elevation mapping for terrain the walker has not been told about, and loop-closing SLAM
   switched on by default once it is fast enough.
5. **Real ROS 2 validation**: run the bridge against a real rosbridge server and the standard ROS 2 tools.
