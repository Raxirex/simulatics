# Simulatics

A humanoid robot simulation workbench: edit a robot's body and motors, simulate it, give it a lidar and
cameras, plan paths, and evolve walking controllers (and bodies) in parallel.

Python for the application, MuJoCo's C engine for physics, a small C++ library for the remaining hot loops,
and a browser GUI served from the local machine.

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
still works through Python fallbacks, only slower.

Other commands:

| Command | What it does |
|---|---|
| `python -m simulatics train --generations 500 --name walker` | Headless training, prints progress per generation |
| `python -m simulatics train --algo ga --evolve-body` | Genetic algorithm that also evolves limb lengths and motor gains |
| `python -m simulatics bench` | Physics throughput against thread count |
| `python -m pytest` | Tests |

## What the GUI does

| Tab | Features |
|---|---|
| **Model** | Sliders for limb lengths, radii, density, motor stiffness and torque limits, timestep, friction. Each change regenerates and recompiles the MJCF. Export the MJCF, or import any MJCF/URDF file (for example a Unitree G1 from MuJoCo Menagerie). |
| **Motors** | Per-motor table edited live in the compiled model: kp, kv, torque limit, joint damping, rotor inertia, and a manual target angle. |
| **Sensors** | Spinning lidar (rays per turn, channels, vertical field of view, range) drawn as a point cloud; RGB or depth camera rendered by MuJoCo; foot contact forces. |
| **Plan** | Random obstacle fields, double-click the floor to set goals, A* path on either the ground-truth map or the map the robot built from its lidar, and a behaviour tree that turns goals into heading commands. |
| **Train** | Evolution strategy or genetic algorithm over the whole population in parallel, live fitness chart, optional body evolution, list of trained policies to load and play. |

`runs/pretrained_walker` contains a policy trained with `--heading-range 0.5` (so it learns to steer) so path following can be tried
immediately: Train tab, "Load and play", then double-click the floor.

## Layout

```
simulatics/
  models/humanoid.py     parametric humanoid -> MJCF (geometry + motors as one dataclass)
  sim/task.py            batched observation, action mapping and reward for locomotion
  sim/parallel.py        parallel closed-loop episodes on MuJoCo's C++ thread pool
  sim/sensors.py         lidar (mj_multiRay) and camera (offscreen renderer)
  train/policy.py        MLP policy as a flat genome, observation normaliser
  train/evolve.py        evolution strategy and a GA, optional body co-evolution
  plan/grid.py           occupancy grids (ground truth + lidar log-odds), planning
  plan/navigator.py      behaviour tree: goals -> path -> heading command
  native/src/simcore.cpp C++ kernels: batched MLP, inflation, A*, smoothing, scan integration
  native/__init__.py     ctypes bindings with Python fallbacks
  server/                FastAPI app, live simulation session, static front end
docs/RESEARCH.md         tool survey and learning roadmap
```

## How the pieces fit

- **One genome, many bodies.** A candidate is `[policy weights | body genes]`. The trainer compiles a MuJoCo
  model per candidate when body evolution is on, and MuJoCo's `rollout` steps all of them in a thread pool.
- **Planner and controller are decoupled.** The policy only sees a target heading in its observation. The
  behaviour tree produces that heading from the path, so planners can change without retraining.
- **Position servos, not raw torques.** The policy outputs offsets around the standing pose; PD servos turn them
  into torques. This is why evolution finds walking in minutes instead of hours.

## Known limits

- Training is CPU-only (roughly 100-150k physics steps per second on an 8-core laptop). GPU-batched simulation
  (MuJoCo Warp) is the next step for larger populations.
- Evolved gaits are functional, not natural, and steering quality depends on the heading range used in training.
- Robot parts do not collide with each other in the built-in humanoid (they do collide with the world).
- The lidar map is 2D; there is no SLAM, the robot's pose comes from the simulator.
- No ROS 2 bridge yet. See `docs/RESEARCH.md` for when and how to add one.
