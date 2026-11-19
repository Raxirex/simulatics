# ROS 2 bridge

`simulatics.bridge` connects a running Simulatics simulation to a ROS 2 system. It publishes the
robot's clock, joints, TF, odometry, IMU, lidar scan, camera, map, plan and URDF, and it takes
`/cmd_vel`, `/goal_pose` (RViz's "2D Goal Pose") and arm commands back. You can then use RViz,
Nav2, other rosbridge clients or your own nodes with the simulated humanoid.

> **Test status:** the bridge has been tested only against the bundled mock rosbridge server
> (`simulatics/bridge/mock_server.py`) and the rosbridge v2 protocol document. It has **not** been
> run against a real ROS 2 install, a real `rosbridge_server`, RViz or Nav2. The rclpy
> transport is untested because ROS 2 is not installed on the development machine; its tests skip
> when `rclpy` is missing. Treat the ROS-side instructions below as a starting point, and please
> report anything that does not work.

## Two ways to connect

| Transport | When to use it | What it needs on the Simulatics side |
|---|---|---|
| **rosbridge** (default) | ROS 2 runs somewhere else: WSL2, another PC, a container, or Windows in a different Python | only the `websockets` package (already a dependency through uvicorn) |
| **rclpy** | ROS 2 is sourced in the same shell, and its Python is the one that runs Simulatics | a ROS 2 install whose `rclpy` matches this Python version |

The rosbridge transport is a websocket client that speaks the
[rosbridge v2 protocol](https://github.com/RobotWebTools/rosbridge_suite/blob/ros2/ROSBRIDGE_PROTOCOL.md)
to a `rosbridge_server` running on the ROS side. It reconnects with exponential backoff, re-sends
its advertisements and subscriptions after a reconnect, and replays the last message of latched
topics. Each topic keeps only its newest message, so a slow link drops stale data instead of
building up lag.

## Quick start

```bash
# 1. ROS side (Linux / WSL2, ROS 2 Jazzy shown; Lyrical works the same way)
sudo apt install ros-jazzy-rosbridge-suite
source /opt/ros/jazzy/setup.bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml      # listens on ws://0.0.0.0:9090

# 2. Simulatics side (Windows PowerShell, from the repository root)
.venv\Scripts\python.exe -m simulatics.bridge --url ws://localhost:9090
```

The command runs a headless simulation (the same `SimSession` the GUI uses) plus the bridge, and
prints connection state and per-topic rates every 3 s. Useful options:

| Option | Effect |
|---|---|
| `--url ws://HOST:9090` | the rosbridge server to connect to |
| `--mock` | start a local mock rosbridge on that port and print the traffic it receives (no ROS needed) |
| `--goal X Y YAW_DEG` | with `--mock`: after 2 s, inject a `/goal_pose` in the map frame, as RViz would |
| `--rclpy` | publish with rclpy instead of rosbridge |
| `--speed S` | simulation speed (1 = real time) |
| `--obstacles N --seed K` | random obstacles in the world |
| `--no-lidar`, `--no-camera` | turn off `/scan` and the lidar map, or the camera topics |
| `--map-source truth` | publish the ground-truth grid as `/map` instead of the lidar-built one |
| `--no-link-tf` | do not publish URDF link TF; use this when you run `robot_state_publisher` |
| `--no-map-odom` | do not publish `map->odom`; use this when a SLAM node publishes it |
| `--goal-topic T` | listen for navigation goals on T instead of `/goal_pose`; use this with Nav2 |
| `--duration SEC`, `--paused`, `-v` | stop after SEC seconds; start paused; verbose logging |

To try the bridge offline, with no ROS anywhere:

```powershell
.venv\Scripts\python.exe -m simulatics.bridge --mock --goal 1.0 0.5 90
```

### Where ROS 2 can run

* **WSL2 (recommended on Windows).** Install Ubuntu 24.04 with ROS 2 Jazzy, or Ubuntu 26.04 with
  Lyrical, in WSL2, and start `rosbridge_server` there. With WSL's default NAT networking,
  `localhost:9090` on Windows is forwarded to a port that a WSL process listens on, so
  `--url ws://localhost:9090` should work. If it does not, use WSL's mirrored networking mode, or
  pass the WSL VM's IP address (`hostname -I` inside WSL). Nav2 and SLAM are Linux-first, so WSL2 is
  the easiest place for them.
* **Another PC or a robot computer.** Run `rosbridge_server` there and pass
  `--url ws://<its-ip>:9090`. Port 9090 must be reachable through the firewall.
* **Windows native.** ROS 2 Lyrical lists Windows 11 as Tier 1, and RoboStack ships ROS as conda
  packages. Whether `rosbridge_suite` is packaged for Windows in either one is **unverified**. If it
  is not, run rosbridge in WSL2 and keep Simulatics on Windows.

## Topics

Stamps are simulation time. Rates are wall-clock maximums, configurable in `BridgeConfig`. Topics
that depend on time are sent once per simulation instant. While the simulation is paused they are
not repeated, which avoids `TF_REPEATED_DATA` warnings.

| Topic | Type | Dir | Frame | Default rate | Notes |
|---|---|---|---|---|---|
| `/clock` | `rosgraph_msgs/msg/Clock` | out | - | 50 Hz | run ROS nodes with `use_sim_time:=true` |
| `/joint_states` | `sensor_msgs/msg/JointState` | out | - | 30 Hz | every hinge/slide joint of the robot; effort = actuator generalized force |
| `/tf` | `tf2_msgs/msg/TFMessage` | out | - | 30 Hz | `map->odom` (identity, or from `slam_pose`), `odom->base_link`, moving URDF joints, moving sensors |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | out, latched | - | on model change | `base_link->sensors`, `base_link->torso`, fixed joints |
| `/odom` | `nav_msgs/msg/Odometry` | out | `odom` -> `base_link` | 30 Hz | pose in `odom`, twist in `base_link` (ground truth) |
| `/imu` | `sensor_msgs/msg/Imu` | out | `imu_link` | 50 Hz | the model's framequat, gyro and accelerometer at the `imu` site (gravity included, +z at rest) |
| `/scan` | `sensor_msgs/msg/LaserScan` | out | `lidar_link` | lidar rate (10 Hz) | the 3D lidar collapsed to one range per azimuth; see below |
| `/head_cam/image/compressed` | `sensor_msgs/msg/CompressedImage` | out | `head_cam_optical` | 2 Hz | JPEG 320x240; format `rgb8; jpeg compressed bgr8` |
| `/head_cam/camera_info` | `sensor_msgs/msg/CameraInfo` | out | `head_cam_optical` | with each image | ideal pinhole from MuJoCo's `fovy`, `plumb_bob` with zero distortion |
| `/map` | `nav_msgs/msg/OccupancyGrid` | out, latched | `map` | at most 0.5 Hz, only when the grid changed | 24 x 24 m, 0.1 m cells; -1 unknown, 0 free, 100 occupied |
| `/plan` | `nav_msgs/msg/Path` | out | `map` | at most 2 Hz, on change | the built-in navigator's path |
| `/robot_description` | `std_msgs/msg/String` | out, latched | - | on model change | URDF from `design/urdf.export_urdf` |
| `/simulatics/events` | `std_msgs/msg/String` | out | - | per command | JSON `{"t", "source", "ok", "message", ...}` results of the commands below |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | in | `base_link` | - | `linear.x/y` (m/s) and `angular.z` (rad/s); stop after 0.5 s without messages |
| `/goal_pose` | `geometry_msgs/msg/PoseStamped` | in | `map`, `odom` or `base_link` | - | navigate to the pose; the end heading comes from the orientation |
| `/simulatics/arm_command` | `std_msgs/msg/String` | in | - | - | JSON arm skill, see below |

"Latched" means transient-local durability. The bridge sends both the older `latch: true` flag and
the `qos` object that rosbridge 2.1 added. A subscriber has to ask for **Transient Local**
durability to receive these messages; in RViz, set *Durability Policy* on the Map and RobotModel
displays.

**`/scan` details.** The simulated lidar is 3D (8 channels from -25 to +5 degrees by default). For
each azimuth the bridge keeps the nearest return whose world height is between 0.15 and 1.8 m.
That excludes the floor and includes low obstacles, as `pointcloud_to_laserscan` would. The range
is measured in the lidar's own plane. Angles run from -pi to pi over 360 degrees, starting behind
the robot. JSON cannot carry `inf`, so over rosbridge a beam with no return reports
`range_max + 1`; consumers treat values above `range_max` as invalid, so such beams clear nothing.
The rclpy transport sends `+inf` instead, as REP-117 recommends. `BridgeConfig.scan_no_return`
overrides both.

**`/cmd_vel` details.** With the default walking controller (and the PPO controller), commands go
to `controller.set_velocity(vx, vy, wz)`. The walker clips them to its step limits, and it only
changes speed at footstep boundaries, so expect about one step of latency. A command cancels any
goal in progress. If `/cmd_vel` stops for `cmd_vel_timeout` (0.5 s), the robot stops. Controllers
without `set_velocity`, such as the ES/GA heading policies, receive a goal 1 s ahead instead,
refreshed at most twice a second.

**`/goal_pose` details.** A goal in `map` is used as is while `map->odom` is the identity. With a
`slam_pose` estimate supplied, the goal is first moved into the simulator's world frame. Goals in
`odom` are used directly, and goals in `base_link` are relative to the robot. The built-in
navigator plans around obstacles, and the robot ends on the goal facing its heading (typically
within about 5 cm). Commands un-pause the simulation, as clicking in the GUI does
(`BridgeConfig.auto_run`).

**`/simulatics/arm_command`** takes a JSON string, for example:

```json
{"skill": "reach", "side": "right", "target": [0.45, -0.2, 0.9], "approach": [0, 0, -1], "frame": "odom"}
{"skill": "grasp", "side": "right"}
{"skill": "wave", "side": "left"}
```

The skills are `reach`, `point_at` (both need `target`; `frame` is `odom`, `map` or `base_link`),
`grasp`, `release`, `home`, `wave` and `raise`. Plain text such as `wave right` also works. The
result of each command appears on `/simulatics/events`.

## Frames

Frames follow REP-103: right-handed, with body frames x forward, y left, z up. Camera optical
frames use z forward, x right, y down. Frame names follow REP-105. The simulator's world frame is
published as `odom`, and the robot's pose is the exact simulated pose, so odometry has no drift.

```
map --(identity, or from slam_pose)--> odom --(robot pose)--> base_link
base_link --(identity)--> torso --(joints)--> pelvis, thighs, ... every URDF link
base_link --> lidar_link            (the "lidar" site)
base_link --> imu_link              (the "imu" site)
base_link --> head_cam --> head_cam_optical    (rpy -90, 0, -90 deg)
```

| Frame | Parent | Source | Published on |
|---|---|---|---|
| `map` | - | global frame | - |
| `odom` | `map` | identity, or `slam_pose()` | `/tf` (off with `map_to_odom=False`) |
| `base_link` | `odom` | free-joint pose of the robot's root body | `/tf` |
| `torso` (URDF root) | `base_link` | identity | `/tf_static` |
| other URDF links | their parent link | joint positions, with URDF semantics | `/tf` (moving joints), `/tf_static` (fixed joints) |
| `lidar_link` | `base_link` | MuJoCo site `lidar` | `/tf_static` (`/tf` if the site is not on the root body) |
| `imu_link` | `base_link` | MuJoCo site `imu` | `/tf_static` |
| `head_cam` | `base_link` | MuJoCo camera `head_cam`, with axes turned to REP-103 | `/tf_static` |
| `head_cam_optical` | `head_cam` | fixed optical rotation | `/tf_static` |

The bridge computes link TF from the URDF itself, which is what `robot_state_publisher` does, so
RViz's RobotModel works with only the bridge running. If you would rather run
`robot_state_publisher` with the URDF from `/robot_description`, start the bridge with
`--no-link-tf` (`BridgeConfig(link_tf=False)`); otherwise both would publish the same transforms.

## RViz 2

```bash
rviz2 --ros-args -p use_sim_time:=true
```

* **Fixed Frame:** `odom`, or `map`. They coincide unless you supply a SLAM pose.
* **RobotModel:** Description Source = Topic, Description Topic = `/robot_description`,
  Durability Policy = Transient Local.
* **TF:** shows the frames above.
* **LaserScan:** topic `/scan`. Reliable or Best Effort both work against the bridge's reliable publisher.
* **Map:** topic `/map`, Durability Policy = Transient Local.
* **Path:** `/plan`. **Odometry:** `/odom`.
* **Image:** topic `/head_cam/image` with transport hint `compressed`; this needs the
  `compressed_image_transport` plugin (`ros-jazzy-image-transport-plugins`).
* **2D Goal Pose** tool: click and drag on the floor; the robot walks there. Do not also run Nav2
  when you use it this way; see below.

## Nav2

The bridge provides what Nav2 expects from a robot: `/cmd_vel`, `/scan`, `/odom` and
`odom->base_link` TF, plus a `map->odom` TF and a `/map`. Things to set (not tested):

* `robot_base_frame: base_link`, `odom_topic: /odom`, `global_frame: map` (global costmap) and
  `odom` (local costmap), `use_sim_time: true` everywhere.
* **Two planners would race for `/goal_pose`.** Nav2's RViz goal tool and `bt_navigator` use
  `/goal_pose` too, and the bridge treats it as a goal for the built-in navigator. When Nav2 does
  the planning, move the bridge's goal topic out of the way: `--goal-topic /simulatics/goal_pose`,
  or `BridgeConfig(topics={"goal_pose": "/simulatics/goal_pose"})`.
* **`/cmd_vel` type.** The bridge subscribes to `geometry_msgs/msg/Twist`. Newer Nav2 releases can
  publish `TwistStamped` (parameter `enable_stamped_cmd_vel`); set it to `false`, or relay the
  topic. Which default your Nav2 version uses is **unverified**.
* **Speeds.** The robot is a biped. Keep `max_vel_x` around 0.2 to 0.3 m/s, `max_vel_theta`
  around 0.5 rad/s and accelerations low, and expect about one step of latency.
* **Map.** Either use the bridge's `/map` with the static layer (`map_subscribe_transient_local:
  true`; `--map-source truth` gives the exact obstacle map), or run a SLAM node such as
  slam_toolbox on `/scan`. In that case the SLAM node publishes `map->odom`, so turn the bridge's
  off with `--no-map-odom` (`BridgeConfig(map_to_odom=False)`). `/map` and `/plan` are then stamped
  in `odom`.
* **`/scan` values above `range_max`** (no return; see above) mark nothing and clear nothing in
  the costmap.

## Other rosbridge clients

Any client that speaks the rosbridge v2 protocol can connect to the same `rosbridge_server` at
`ws://localhost:9090`; the bridge's topics then appear like any other ROS topics. Clients that list topics
do so through rosbridge's `rosapi` services, which `rosbridge_websocket_launch.xml` starts. The **mock
server does not implement rosapi**, so such clients cannot browse topics on `--mock`; use a real rosbridge.

## rclpy mode

If a ROS 2 environment is sourced in the shell **and** its `rclpy` is importable from the Python
that runs Simulatics, the bridge can publish natively, with no rosbridge involved:

```bash
source /opt/ros/jazzy/setup.bash            # Linux / WSL2 (Jazzy uses Python 3.12)
python -m simulatics.bridge --rclpy
```

`rclpy` is only importable from the Python version ROS was built for. The project's Windows venv
uses Python 3.13, so rclpy mode normally means installing Simulatics into ROS's Python environment
(WSL2, or a RoboStack/pixi environment on Windows). The backend creates one node,
`simulatics_bridge`, with reliable QoS; latched topics are transient-local with depth 1. It uses
the real message classes (`sensor_msgs.msg.LaserScan`, ...). `import simulatics.bridge` never needs
rclpy, and `RclpyTransport()` raises a clear `RuntimeError` when rclpy is missing. **Untested:**
there is no ROS 2 on the development machine.

## Python API

```python
from simulatics.bridge import start_bridge, BridgeConfig, RosBridge, RosbridgeClient, MockRosbridgeServer

bridge = start_bridge(session, "ws://localhost:9090")          # session: a SimSession or any SessionLike
bridge = start_bridge(session, mode="rclpy")                   # native ROS 2
bridge = RosBridge(session, RosbridgeClient(url), BridgeConfig(camera_hz=5, link_tf=False),
                   slam_pose=lambda: (x, y, yaw)).start()      # map->odom from your estimate
bridge.status()   # {"connected", "state", "url", "error", "sim_time", "topics": {topic: {sent, dropped, rate_hz}}, "events", ...}
bridge.stop()     # joins the bridge threads and stops the transport
```

The bridge reads the simulation through a small protocol, `SessionLike`: `lock`, `model`, `data`,
`controller`, `nav`, `lidar`, `grid` and `robot` (a `RobotInterface`), plus optional `running` and
`urdf()`. `SessionAdapter` provides it for `simulatics.server.session.SimSession`, and
`RosBridge` wraps a SimSession automatically. MuJoCo state is only read while holding
`session.lock`.

`MockRosbridgeServer` is a rosbridge stand-in for tests and demos. It records every published
message (`messages(topic)`, `rate(topic)`, `wait_for(topic, n)`), checks each message against the
ROS 2 message definitions, and publishes into topics like a ROS node would (`inject(topic, msg)`).
It also answers `call_service` through handlers you add with `add_service`.

## What was verified, and how

* **Message schemas.** Every builder's output is checked field by field (names, primitive types,
  integer ranges, fixed array lengths, no extra or missing fields) against the ROS 2 `.msg`
  definitions, which are written out twice: once in `msgs.MSG_DEFS` and once, independently, in
  `tests/test_bridge.py`. On 2026-11-19 all 25 definitions (the 15 published or subscribed types
  and the types nested in them) were diffed against the upstream `.msg` files of
  `ros2/common_interfaces`, `ros2/rcl_interfaces` and `ros2/geometry2`, on both the `jazzy` and
  `lyrical` branches. They are identical.
* **Live session over the mock server.** With a headless `SimSession` (lidar on, speed 8) bridged
  to the mock rosbridge, the tests check:
  * every topic passes the schema check and arrives at its configured rate: within about 5 % on an
    idle machine. The test allows -70 % / +20 % for the periodic topics, because about 45 % was
    measured on a loaded test run (see the known limits). For the 2 Hz camera topics it checks
    frame counts instead;
  * the TF tree matches MuJoCo: `base_link` equals the robot pose, every URDF link matches its
    body to 0.1 mm, and the lidar, IMU, camera and optical frames match their sites and camera;
  * `/odom`, `/imu` and `/joint_states` agree with the simulation at the same stamp;
  * `/scan` returns lie on a test box's outline; `/map` marks the box as occupied;
  * a `/goal_pose` makes the robot walk there, ending within 10 cm and 10 degrees;
  * `/cmd_vel` makes it walk, and the watchdog stops it when the commands stop;
  * controllers without `set_velocity` get the relative-goal fallback (tested with a stub session);
  * with a `slam_pose` estimate, `map->odom` follows it, `/map` is stamped in `odom`, and map-frame
    goals are moved into the simulator's world;
  * an arm command runs;
  * after a server restart the client reconnects, re-advertises, re-subscribes and replays latched
    data;
  * `stop()` leaves no threads behind.
* **The rclpy backend's dict-to-message conversion** is tested with classes that mimic the rosidl
  Python API. The live rclpy test skips without ROS.

## Known limits

* Not tested against real ROS 2, rosbridge_server, RViz or Nav2 (see the top of this page).
* `/cmd_vel` accepts `Twist` only, not `TwistStamped`.
* There are no `/clock`-driven lockstep semantics: the simulation runs on its own clock and the
  bridge samples it. Rates are wall-clock, so at `--speed 4` a 30 Hz topic covers about 7.5 Hz of
  simulation time.
* `/scan` uses `range_max + 1` for "no return" over rosbridge, because JSON cannot carry `inf`.
* Publish rates sag when the simulation runs much faster than real time on a busy CPU. With the
  sim at speed 8 during a loaded full test run, the measured rates were about 45 % of target
  (clock 22 of 50 Hz, joint_states 15 of 30 Hz). The cause has not been isolated. The likely one is
  the bridge waiting for `session.lock`, which it must hold to read MuJoCo state; the simulation
  thread re-takes it almost continuously when it runs behind. With the sim at speed 1 to 4 on an
  idle machine, the rates hold to within about 5 %.
* No point cloud (`sensor_msgs/PointCloud2`), depth image, raw image or foot contact topics yet. No
  ROS services or actions (for example `NavigateToPose`), and no `ros2_control` interface.
* The camera is rendered on the simulation thread on request, so a 2 Hz JPEG costs it one render
  per frame. Images need an OpenGL context, as the GUI's camera does.
* `/robot_description` needs `design/urdf.export_urdf` to handle the model. Custom MJCF files with
  external meshes may fail; the bridge then reports `description_error` and skips link TF.
* The mock server implements the operations the bridge uses (advertise, publish, subscribe,
  call_service), not rosapi, fragmentation on send, CBOR/PNG compression or authentication.
