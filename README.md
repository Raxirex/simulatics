# Simulatics

A humanoid robot workbench for Windows: design a robot (sliders, CAD parts, topology optimisation, AI design
search), simulate it in MuJoCo, click anywhere and watch it walk there to within millimetres, use its arms and
hands, give it a lidar and cameras, train walking policies with methods from the published literature, and let a language model pilot it
from camera images.

Python for the application, MuJoCo's C engine for physics, C++ kernels for planning, mapping, batched
inference and topology optimisation, and a browser GUI served from the local machine.

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
| **Sensors** | Spinning lidar drawn as a point cloud; RGB or depth cameras (`head_cam`, `work_cam` looking at the hands, `chase`); foot contact forces. |
| **Walk** | Double-click the floor to send the robot there; Ctrl + drag to also set the direction to face; Shift to queue goals. Choose the controller (model-based walker, learned policy, evolved policy, manual). Planned path and footsteps are drawn; the stopping error is shown. Random obstacle worlds, ground-truth or lidar-built maps. |
| **Arms** | Table with graspable objects; reach a point or an object, grasp, release, point, wave, raise, home. "Auto" picks the hand on the object's side. |
| **AI Pilot** | Claude (Opus 5.5 by default, or Sonnet 5.5, Haiku 5.5, Fable 5.2) or the offline pilot runs a mission in plain language. The transcript shows the model's text, tool calls, results and the camera images it saw, with token and cost counters. |
| **Design** | CAD parts with real masses; simulate with CAD geometry; download STL/OBJ/GLB/STEP or URDF. Topology-optimise a link under loads taken from simulated walking, view the result, and put it on the robot. CMA-ES or Bayesian optimisation of body proportions and motor gains, each candidate scored by simulated walking. |
| **Train** | PPO velocity tracking (Rudin et al. 2021 recipe), distil the model-based walker into a network, evolution strategy, genetic algorithm. Live chart, trained policies to load and use. |

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

## Command line

| Command | What it does |
|---|---|
| `python -m simulatics` | GUI |
| `python -m simulatics train --generations 500 --name walker` | Headless evolution strategy (`--algo ga`, `--evolve-body`) |
| `python -m simulatics.train.ppo train --seconds 3600 --name rl_walker` | Headless PPO (resume with `--resume`) |
| `python -m simulatics.train.ppo distill --teacher walker` | Distil the model-based walker into a network |
| `python -m simulatics.train.ppo bench --threads 4 8 --envs 256` | Environment throughput |
| `python -m simulatics bench` | Physics throughput against thread count |
| `python -m pytest` | Tests (set `SIMULATICS_LIVE_CLAUDE=1` to include one live Claude mission, `SIMULATICS_SLOW=1` for the long PPO test) |

## Layout

```
simulatics/
  models/humanoid.py       parametric humanoid -> MJCF (geometry, motors, arms, hands, props)
  control/                 controller protocol, path follower,
    walker.py, walk_*.py   model-based walking: footsteps, ZMP preview, DCM balance, leg IK
    arms.py                arm IK skills and grasping
  plan/                    occupancy grids, A*, behaviour-tree navigator
  sim/                     locomotion tasks, vectorised environments, sensors, scenes and props
  train/                   evolution (ES/GA), PPO and distillation (numpy), policies
  design/                  CAD, URDF, loads, topology optimisation, design optimisation
  llm/                     Claude and scripted pilots, tools, senses, mission budgets
  native/src/*.cpp         C++ kernels: simcore (A*, mapping, MLP batch), topopt (FEA solver)
  server/                  FastAPI app, live session, robot adapter for pilots, front end
docs/FRONTIER.md           lab -> method -> module map
docs/RESEARCH.md           tool survey, learning roadmap, research/01-10 detail files
```

## Known limits

- **Learned walking is not trained yet.** PPO runs on CPU; a usable policy needs hours, and the bundled 10-minute run only shuffles forward. The model-based walker is the accurate controller today.
- **Pushes and terrain.** The walker assumes flat ground, uses fixed step timing, and survives sideways pushes of 60 N for 0.1 s only about half the time.
- **Body range.** Accurate walking was verified for thigh/shin 0.28-0.55 m, pelvis 0.14-0.30 m, foot 0.16-0.30 x 0.07-0.13 m, density 600-1300 kg/m3. The sliders allow more; very heavy or short-footed bodies can fall.
- **Arms** have no shoulder yaw, so each hand works on its own side and in front; long or heavily damped arms can unbalance the robot.
- **Claude** has not been tested live on this machine (there is no API key here). The request shapes are verified against a fake client.
- **Pose** comes from the simulator: there is no SLAM. The lidar map is 2D.
- **No ROS 2 bridge** yet.
