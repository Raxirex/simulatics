# Simulatics

A humanoid robot workbench for Windows: design a robot (sliders, CAD parts, topology optimisation, AI design
search), simulate it in MuJoCo, click anywhere and watch it walk there to within millimetres (over steps,
stairs and ramps, recovering from pushes), use its arms and hands, give it a lidar and cameras, localise it
with SLAM, connect it to ROS 2, train walking and motion-imitation policies with methods from the published
literature, and let a language model pilot it from camera images.

MIT-licensed; see `LICENSE` and `THIRD_PARTY_NOTICES.md`.

Python for the application, MuJoCo's C engine for physics, C++ kernels for planning, mapping, batched
inference, SLAM scan matching, MPC and topology optimisation, and a browser GUI served from the local machine.

`docs/METHODS.md` maps each method from the literature, with its paper, to the module that implements it.

## Run it

```bash
.venv\Scripts\python.exe -m simulatics
```

This opens the GUI at http://127.0.0.1:8765. The 3D view loads three.js from a CDN, so the first start needs
an internet connection.

Fresh checkout:

```bash
python -m venv .venv
```
```bash
.venv\Scripts\python.exe -m pip install -e .[dev]
```
```bash
.venv\Scripts\python.exe -m simulatics build-native
```

`build-native` needs CMake and a C++ compiler (Visual Studio Build Tools on Windows). Without it everything
still works through Python fallbacks, only slower. STEP export needs the optional `[step]` extra
(build123d, about 300 MB). No PyTorch or GPU is required.

### AI pilot

The AI Pilot tab uses the Anthropic API. Either set `ANTHROPIC_API_KEY`, run `ant auth login`, or paste a key
into the tab (it stays in the server's memory and is sent only to Anthropic). Missions have turn, cost and
time limits, and a refusal fallback to another model is on by default (switchable). Without credentials, the
offline scripted pilot runs simple missions ("walk to the red box", "pick up the blue cube", "wave").

## What the GUI does

| Tab | Features |
|---|---|
| **Model** | Sliders for limb lengths, radii, density, motor stiffness and torque limits, timestep, friction; arms and hands toggles. Each change regenerates and recompiles the MJCF. Export MJCF, or import any MJCF/URDF file. |
| **Motors** | Per-motor table edited live in the compiled model: kp, kv, torque limit, joint damping, rotor inertia, manual targets. |
| **Sensors** | Spinning lidar drawn as a point cloud; RGB or depth cameras (`head_cam`, `work_cam` looking at the hands, `chase`); foot contact forces. ROS 2 bridge: connect to a rosbridge server (or native rclpy) and publish /tf, /odom, /scan, /imu, /joint_states, camera, /map, /plan, /robot_description; take /cmd_vel, /goal_pose (RViz "2D Goal Pose") and arm commands. |
| **Walk** | Double-click the floor to send the robot there; Ctrl + drag to also set the direction to face; Shift to queue goals. Choose the controller (model-based walker with push recovery, MPC, learned policy, evolved policy, manual). Planned path and footsteps are drawn (at terrain height); the stopping error is shown. Push the robot from any side. Terrain courses (random steps, stairs, ramps). Plan on the ground-truth map, the lidar map, or the SLAM map with the SLAM pose, with the estimate, its uncertainty and the SLAM / odometry / true trails drawn. |
| **Arms** | Table with graspable objects; reach a point or an object, grasp, release, point, wave, raise, home. "Auto" picks the hand on the object's side. |
| **AI Pilot** | Claude (Opus 5.5 by default, or Sonnet 5.5, Haiku 5.5, Fable 5.2) or the offline pilot runs a mission in plain language. The transcript shows the model's text, tool calls, results and the camera images it saw, with token and cost counters. |
| **Design** | CAD parts with real masses; simulate with CAD geometry; download STL/OBJ/GLB/STEP or URDF. Topology-optimise a link under loads taken from simulated walking, view the result, and put it on the robot. CMA-ES or Bayesian optimisation of body proportions and motor gains, each candidate scored by simulated walking. |
| **Train** | PPO velocity tracking (Rudin et al. 2021 recipe), distil the model-based walker into a network, motion imitation (DeepMimic-style tracking), evolution strategy, genetic algorithm. Live chart, trained policies to load and use. Motion clips: preview, play with balance control, import your own BVH (retargeted to the robot). |

## Measured on this laptop (i7-11800H, no GPU used)

| What | Result |
|---|---|
| Click-to-walk, default robot, 3 m | stops 4 mm from the goal, 13 s |
| Walk to (2, 2) ending at 90 deg | 8 mm, 0.2 deg |
| Random bodies and goals (thigh/shin 0.28-0.55 m, pelvis 0.14-0.30 m, feet, density, torso) | 320/320 reached, 0 falls, mean error 5-6 mm, max 1.6 cm |
| Path around obstacles in the GUI | 7 mm at (3.5, 2.0) facing 90 deg |
| Walk up to a table and pick an object (scripted pilot) | 8/8 random scenes and starts |
| Arm reach accuracy | about 1 mm |
| Topology optimisation of the thigh (40x10x10 voxels, 40 iterations, C++) | about 11 s, 29% lighter at 94% of the tube's stiffness |
| PPO environment throughput (64 envs, 8 threads) | about 24k physics steps/s |
| Walker compute | about 0.6 ms per 5 ms control tick (8x real time) |
| MPC walker (LIPM MPC with footstep adaptation, own QP solver) | 0.2-0.5 cm at the goal on three body shapes; 2-4 ms per solve |
| Push recovery while walking, 0.1 s torso push (n=16, unseen seed) | sideways 60 N 15/16 (was 9/16), 100 N 8/16 (was 0/16); standing 140 N 8/8 (was 0/8) |
| Terrain | random 1-5 cm steps 9/10 seeds; 6-8 cm stairs and 8 deg ramps pass; 3-9 mm at the goal |
| SLAM over 20-26 m walks, three worlds | trajectory error 2-6 cm vs 1.4-1.7 m for odometry alone; 1-6 ms per scan |
| Learned walking policy (distilled from the walker with DAgger, 45 min CPU) | 40/40 held-out 2-4 m goals, 0 falls; error median 5 cm, max 7.5 cm; tracks 0.2-0.4 m/s and turns in place |
| Motion imitation (walk clip, 35 min PPO on 4 threads) | 25 s without falls; 0.2 deg joint, 2.7 mm hand/foot, 6.6 mm root error |
| ROS 2 bridge (against a mock rosbridge server) | all topics at target rates; a /goal_pose walked the robot to within 7 mm |

## Command line

| Command | What it does |
|---|---|
| `python -m simulatics` | GUI |
| `python -m simulatics train --generations 500 --name walker` | Headless evolution strategy (`--algo ga`, `--evolve-body`) |
| `python -m simulatics.train.ppo train --seconds 3600 --name rl_walker` | Headless PPO (resume with `--resume`) |
| `python -m simulatics.train.ppo distill --teacher walker` | Distil the model-based walker into a network |
| `python -m simulatics.train.ppo bench --threads 4 8 --envs 256` | Environment throughput |
| `python -m simulatics.imitate.train train --motion walk --name X --minutes 30` | Train a motion-imitation tracker (`eval <policy>` to evaluate) |
| `python -m simulatics.imitate.library` | Rebuild the reference motion clips |
| `python -m simulatics.bridge --url ws://localhost:9090` | Headless sim + ROS 2 bridge (`--mock` runs a local stand-in server); see `docs/ROS2.md` |
| `python -m simulatics bench` | Physics throughput against thread count |
| `python -m pytest` | Tests (set `SIMULATICS_LIVE_CLAUDE=1` to include one live Claude mission, `SIMULATICS_SLOW=1` for the long PPO test) |

## Layout

```
simulatics/
  models/humanoid.py       parametric humanoid -> MJCF (geometry, motors, arms, hands, props)
  control/                 controller protocol, path follower,
    walker.py, walk_*.py   model-based walking: footsteps, ZMP preview, DCM balance, leg IK,
                           push recovery, terrain
    mpc*.py                linear MPC walking with footstep and timing adaptation
    arms.py                arm IK skills and grasping
  plan/                    occupancy grids, A*, behaviour-tree navigator, 2D lidar SLAM (slam.py)
  imitate/                 BVH import, retargeting, reference clips, motion tracking (DeepMimic-style)
  bridge/                  ROS 2 bridge (rosbridge v2 client, rclpy backend, mock server)
  sim/                     locomotion tasks, vectorised environments, sensors, scenes and props
  train/                   evolution (ES/GA), PPO and distillation (numpy), policies
  design/                  CAD, URDF, loads, topology optimisation, design optimisation
  llm/                     Claude and scripted pilots, tools, senses, mission budgets
  native/src/*.cpp         C++ kernels: simcore (A*, mapping, MLP batch), topopt (FEA solver),
                           slam (scan matching), mpc (QP solver)
  server/                  FastAPI app, live session, robot adapter for pilots, front end
docs/METHODS.md            method -> paper -> module map
docs/ROS2.md               using the bridge with ROS 2
```

## Known limits

- **Learned walking** (`runs/rl_walker`, distilled from the walker) stops about 5 cm from goals, not
  millimetres; it never returns to a full stand once it has walked (it steps in place), and it was trained for
  flat ground without sensor noise. PPO fine-tuning on CPU did not improve it. The model-based walker and the
  MPC are the accurate controllers.
- **Pushes.** A 100 N sideways push while walking is survived about half the time; forward/back 100 N
  recovery is no better than without the recovery planner. Push and terrain tests used the default body.
- **Terrain** is rigid and static; the walker snapshots it when it is built. Stairs need treads of about 0.3 m.
- **Body range.** Accurate walking was verified for thigh/shin 0.28-0.55 m, pelvis 0.14-0.30 m, foot
  0.16-0.30 x 0.07-0.13 m, density 600-1300 kg/m3. The sliders allow more; very heavy or short-footed bodies
  can fall.
- **Arms** have no shoulder yaw, so each hand works on its own side and in front.
- **Claude** has not been tested live (no API key on the development machine); request shapes are verified
  against a fake client.
- **SLAM** is 2D, assumes a static world, and its loop closure is experimental (off by default).
- **ROS 2** was verified only against the bundled mock rosbridge server, not a real ROS 2 install.
- **Motion imitation**: one policy per clip; the tracker's observations use simulator state (sim only).

## License

MIT (`LICENSE`). Dependencies are installed separately under their own permissive licences; see
`THIRD_PARTY_NOTICES.md`.
