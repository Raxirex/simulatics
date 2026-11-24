"""Node graph: a headless engine that wires robots, worlds, simulation, sensors, training and scripts together.

    from simulatics.graph import EXAMPLES, Executor
    ex = Executor()
    out = ex.run(EXAMPLES["Script"](), on_event=print)
    ex.close()
"""
from .core import (CATEGORY_COLORS, PORT_TYPES, REGISTRY, Cancelled, Context, Executor, Graph, GraphError, Link,
                   NodeInstance, NodeType, Port, coerce, compatible, jsonable, node, register, summarize)
from .nodes import EXAMPLES, RobotSpec, describe, ensure_applied, find_policy, run_sim

__all__ = ["CATEGORY_COLORS", "PORT_TYPES", "REGISTRY", "Cancelled", "Context", "Executor", "Graph", "GraphError",
           "Link", "NodeInstance", "NodeType", "Port", "coerce", "compatible", "jsonable", "node", "register",
           "summarize", "EXAMPLES", "RobotSpec", "describe", "ensure_applied", "find_policy", "run_sim"]
