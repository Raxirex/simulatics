"""`python -m simulatics.bridge`: a headless simulation bridged to ROS 2.

    python -m simulatics.bridge --url ws://localhost:9090      # to a running rosbridge_server
    python -m simulatics.bridge --mock                         # local mock rosbridge, prints the traffic
    python -m simulatics.bridge --mock --goal 1.0 0.5 90       # ... and send a /goal_pose as RViz would
    python -m simulatics.bridge --rclpy                        # native ROS 2 (ROS sourced in this shell)
"""
from __future__ import annotations

import argparse
import logging
import math
import time
from urllib.parse import urlparse

from . import DEFAULT_URL, BridgeConfig, MockRosbridgeServer, RclpyTransport, RosBridge, RosbridgeClient, msgs


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m simulatics.bridge", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=DEFAULT_URL, help=f"rosbridge websocket URL (default {DEFAULT_URL})")
    p.add_argument("--mock", action="store_true", help="run a mock rosbridge server on --url's port and show traffic")
    p.add_argument("--rclpy", action="store_true", help="publish with rclpy instead of rosbridge")
    p.add_argument("--speed", type=float, default=1.0, help="simulation speed (1 = real time)")
    p.add_argument("--paused", action="store_true", help="do not start the simulation (commands still start it)")
    p.add_argument("--no-lidar", action="store_true", help="disable the lidar (/scan and the lidar map)")
    p.add_argument("--no-camera", action="store_true", help="do not publish camera images")
    p.add_argument("--obstacles", type=int, default=0, help="random obstacles in the world")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--map-source", choices=["lidar", "truth"], default="lidar")
    p.add_argument("--no-link-tf", action="store_true", help="do not publish URDF link TF (robot_state_publisher does)")
    p.add_argument("--no-map-odom", action="store_true", help="do not publish map->odom (a SLAM node does)")
    p.add_argument("--goal-topic", default="/goal_pose",
                   help="topic for navigation goals (move it away from /goal_pose when Nav2 plans)")
    p.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"),
                   help="with --mock: inject a /goal_pose (map frame) after 2 s")
    p.add_argument("--duration", type=float, default=0.0, help="stop after this many seconds (0 = until Ctrl+C)")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    from ..server.session import SimSession
    mock = None
    if a.mock:
        u = urlparse(a.url)
        mock = MockRosbridgeServer(host=u.hostname or "127.0.0.1", port=9090 if u.port is None else u.port).start()
        a.url = mock.url
        print(f"mock rosbridge listening on {mock.url}")
    session = SimSession()
    if a.obstacles:
        session.random_obstacles(a.obstacles, seed=a.seed)
    session.set_lidar({"enabled": not a.no_lidar})
    session.speed = a.speed
    session.running = not a.paused
    cfg = BridgeConfig(camera_hz=0.0 if a.no_camera else 2.0, map_source=a.map_source, link_tf=not a.no_link_tf,
                       map_to_odom=not a.no_map_odom, topics={"goal_pose": a.goal_topic})
    transport = RclpyTransport() if a.rclpy else RosbridgeClient(a.url)
    bridge = RosBridge(session, transport, cfg).start()
    print(f"bridging a headless simulation to {transport.url}  (Ctrl+C to stop)")
    t0, goal_sent, last_print = time.monotonic(), False, 0.0
    try:
        while a.duration <= 0 or time.monotonic() - t0 < a.duration:
            time.sleep(0.2)
            now = time.monotonic()
            if mock is not None and a.goal and not goal_sent and now - t0 > 2.0:
                x, y, yaw = a.goal
                n = mock.inject(a.goal_topic, msgs.pose_stamped(0.0, "map", (x, y, 0.0),
                                                                msgs.yaw_to_quat(math.radians(yaw))))
                goal_sent = n > 0
                print(f"injected {a.goal_topic} ({x}, {y}, {yaw} deg) -> {n} subscriber(s)")
            if now - last_print >= 3.0:
                last_print = now
                _report(bridge, mock, session)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()
        session.running = False
        session.shutdown()
        if mock is not None:
            mock.stop()
        print("stopped")


def _report(bridge: RosBridge, mock, session) -> None:
    st = bridge.status()
    pose = session.robot.pose()
    print(f"\n[{st['state']}] sim t={st['sim_time']:.1f}s  robot ({pose['x']:.2f}, {pose['y']:.2f}, "
          f"{pose['yaw_deg']:.0f} deg)  nav={session.nav.state}" + (f"  error: {st['error']}" if st["error"] else ""))
    if mock is not None:
        for row in mock.summary():
            print(f"  {row['topic']:32s} {row['type']:32s} {row['count']:6d} msgs {row['rate_hz']:6.1f} Hz")
        if mock.errors:
            print(f"  server errors: {mock.errors[-3:]}")
    else:
        for topic, s in st["topics"].items():
            print(f"  {topic:32s} sent {s['sent']:6d}  dropped {s['dropped']:5d}  {s['rate_hz']:6.1f} Hz")
    for ev in st["events"][-3:]:
        print(f"  event: {ev}")


if __name__ == "__main__":
    main()
