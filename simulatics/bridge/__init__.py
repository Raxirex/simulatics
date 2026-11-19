"""ROS 2 bridge: publish the simulated robot to ROS 2 and take commands back (see docs/ROS2.md).

Two transports with the same interface (transport.Transport):

* RosbridgeClient  rosbridge v2 JSON over a websocket to a running rosbridge_server (ROS in WSL2,
                   another PC, a container). Needs only the `websockets` package here.
* RclpyTransport   native rclpy node, when a ROS 2 install is sourced for this Python.

    from simulatics.bridge import start_bridge
    bridge = start_bridge(sim_session, "ws://localhost:9090")   # or mode="rclpy"
    bridge.status(); bridge.stop()

MockRosbridgeServer is a stand-in rosbridge for tests and offline demos.
"""
from __future__ import annotations

from . import msgs
from .bridge import (DEFAULT_TOPICS, TOPIC_TYPES, BridgeConfig, RosBridge, SessionAdapter, SessionLike,
                     as_session_like)
from .mock_server import MockRosbridgeServer
from .rclpy_backend import RclpyTransport, available as rclpy_available
from .rosbridge import RosbridgeClient
from .transport import Transport

DEFAULT_URL = "ws://localhost:9090"

__all__ = ["msgs", "RosBridge", "BridgeConfig", "SessionLike", "SessionAdapter", "as_session_like",
           "RosbridgeClient", "RclpyTransport", "rclpy_available", "MockRosbridgeServer", "Transport",
           "DEFAULT_TOPICS", "TOPIC_TYPES", "DEFAULT_URL", "start_bridge"]


def start_bridge(session, url: str = DEFAULT_URL, mode: str = "rosbridge", config: BridgeConfig | None = None,
                 slam_pose=None, **transport_kw) -> RosBridge:
    """Create a transport, start a RosBridge on `session` (a SimSession or any SessionLike) and return it.

    mode: "rosbridge" (connects to `url`, reconnecting in the background) or "rclpy" (native node;
    raises RuntimeError when rclpy is missing). Returns at once; watch bridge.status()["connected"]."""
    if mode == "rosbridge":
        transport = RosbridgeClient(url, **transport_kw)
    elif mode == "rclpy":
        transport = RclpyTransport(**transport_kw)
    else:
        raise ValueError(f"mode must be 'rosbridge' or 'rclpy', not {mode!r}")
    return RosBridge(session, transport, config, slam_pose=slam_pose).start()
