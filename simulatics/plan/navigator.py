"""Action planning: a small behaviour tree that turns a queue of goals into a heading command.

    Sequence
      |- HaveGoal?              any goal left in the queue
      |- Fallback
      |    |- PathValid?        a path exists and is not blocked on the current map
      |    `- PlanPath          A* on the chosen map (ground truth or lidar-built)
      `- FollowPath             pure-pursuit a lookahead point; pops the goal on arrival

The tree is ticked every control step. Its only output is `target_yaw`, the
heading the walking policy is asked to move along, so high-level planning and
the learned low-level controller stay decoupled.
"""
from __future__ import annotations

import numpy as np

from .grid import GridMap

SUCCESS, FAILURE, RUNNING = "success", "failure", "running"


class Node:
    def __init__(self, name: str, fn=None, children=()):
        self.name, self.fn, self.children = name, fn, list(children)

    def tick(self, bb) -> str:
        bb.active = self.name
        return self.fn(bb)


class Sequence(Node):
    def tick(self, bb) -> str:
        for c in self.children:
            r = c.tick(bb)
            if r != SUCCESS:
                return r
        return SUCCESS


class Fallback(Node):
    def tick(self, bb) -> str:
        for c in self.children:
            r = c.tick(bb)
            if r != FAILURE:
                return r
        return FAILURE


class Navigator:
    def __init__(self, grid: GridMap, robot_radius: float = 0.35):
        self.grid = grid
        self.robot_radius = robot_radius
        self.source = "truth"           # "truth" | "lidar"
        self.goals: list[np.ndarray] = []
        self.path = np.zeros((0, 2))
        self.pos = np.zeros(2)
        self.target_yaw = 0.0
        self.active = "idle"
        self.message = ""
        self.lookahead = 0.8
        self.tolerance = 0.45
        self._ticks = 0
        self.tree = Sequence("root", children=[
            Node("HaveGoal", self._have_goal),
            Fallback("EnsurePath", children=[Node("PathValid", self._path_valid), Node("PlanPath", self._plan)]),
            Node("FollowPath", self._follow),
        ])

    # ---- API -----------------------------------------------------------
    def set_goal(self, xy, append: bool = False) -> None:
        if not append:
            self.goals.clear()
        self.goals.append(np.asarray(xy, float))
        self.path = np.zeros((0, 2))
        self.message = ""

    def clear(self) -> None:
        self.goals.clear()
        self.path = np.zeros((0, 2))
        self.active, self.message = "idle", ""

    def tick(self, pos_xy: np.ndarray, yaw: float) -> float | None:
        """Returns the commanded heading, or None when there is nothing to do."""
        self.pos = np.asarray(pos_xy, float)
        self._ticks += 1
        result = self.tree.tick(self)
        if result == RUNNING:
            return self.target_yaw
        if result == FAILURE and not self.goals:
            self.active = "idle"
        return None

    def to_dict(self) -> dict:
        return {"goals": [g.tolist() for g in self.goals], "path": self.path.round(3).tolist(),
                "node": self.active, "message": self.message, "source": self.source}

    # ---- leaves --------------------------------------------------------
    def _have_goal(self, bb) -> str:
        return SUCCESS if self.goals else FAILURE

    def _path_valid(self, bb) -> str:
        if len(self.path) < 2:
            return FAILURE
        # re-check against the map a few times per second (it changes when mapping with lidar)
        if self._ticks % 25 == 0 and self.grid.blocked(self.path, self.robot_radius, self.source):
            self.path = np.zeros((0, 2))
            return FAILURE
        return SUCCESS

    def _plan(self, bb) -> str:
        path = self.grid.plan(self.pos, self.goals[0], self.robot_radius, self.source)
        if len(path) < 2:
            self.message = "no path to goal - skipped"
            self.goals.pop(0)
            return FAILURE
        self.path, self.message = path, ""
        return SUCCESS

    def _follow(self, bb) -> str:
        if np.linalg.norm(self.goals[0] - self.pos) < self.tolerance:
            self.goals.pop(0)
            self.path = np.zeros((0, 2))
            self.message = "goal reached"
            return SUCCESS
        # drop waypoints already passed, then aim at a point `lookahead` metres along the path
        while len(self.path) > 2 and np.linalg.norm(self.path[1] - self.pos) < self.lookahead:
            self.path = self.path[1:]
        self.path[0] = self.pos
        seg = self.path[1] - self.pos
        d = np.linalg.norm(seg)
        aim = self.path[1] if d <= self.lookahead or len(self.path) == 2 else self.pos + seg / d * self.lookahead
        self.target_yaw = float(np.arctan2(aim[1] - self.pos[1], aim[0] - self.pos[0]))
        return RUNNING
