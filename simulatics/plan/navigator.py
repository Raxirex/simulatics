"""Action planning: a behaviour tree that turns a queue of goals into paths for the active controller.

    Sequence
      |- HaveGoal?              any goal left in the queue
      |- Fallback
      |    |- PathValid?        a path exists and is not blocked on the current map
      |    `- PlanPath          A* on the chosen map (ground truth or lidar-built) -> controller.set_path
      `- Follow                 watch the controller: done -> pop the goal; fallen -> fail

The tree is ticked every control step. Planning (where to go) and control (how
to move the legs) stay decoupled: any controller that implements set_path /
status (control/base.py) can be driven by it.
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
        self.goals: list[tuple[np.ndarray, float | None]] = []   # (xy, final yaw rad or None)
        self.path = np.zeros((0, 2))
        self.pos = np.zeros(2)
        self.yaw = 0.0
        self.active = "idle"
        self.message = ""
        self.state = "idle"             # idle | walking | reached | failed | fallen
        self.last_goal: tuple[np.ndarray, float | None] | None = None
        self.new_path: tuple[np.ndarray, float | None] | None = None  # set when the controller must get a new path
        self.ctrl_status: dict = {}
        self._ticks = 0
        self._since_path = 0
        self.tree = Sequence("root", children=[
            Node("HaveGoal", self._have_goal),
            Fallback("EnsurePath", children=[Node("PathValid", self._path_valid), Node("PlanPath", self._plan)]),
            Node("Follow", self._follow),
        ])

    # ---- API -----------------------------------------------------------
    def set_goal(self, xy, yaw: float | None = None, append: bool = False) -> None:
        if not append:
            self.goals.clear()
            self.path = np.zeros((0, 2))
        self.goals.append((np.asarray(xy, float), yaw))
        self.message, self.state = "", "walking"

    def clear(self) -> None:
        self.goals.clear()
        self.path = np.zeros((0, 2))
        self.new_path = (np.zeros((0, 2)), None)
        self.active, self.message, self.state = "idle", "", "idle"

    def tick(self, pos_xy: np.ndarray, yaw: float, ctrl_status: dict) -> None:
        """Advance the tree. Afterwards, `new_path` (if not None) must be handed to the controller."""
        self.pos, self.yaw = np.asarray(pos_xy, float), float(yaw)
        self.ctrl_status = ctrl_status or {}
        self._ticks += 1
        if self.tree.tick(self) == FAILURE and not self.goals and self.state == "walking":
            self.state = "idle"

    def take_new_path(self):
        p, self.new_path = self.new_path, None
        return p

    def to_dict(self) -> dict:
        return {"goals": [[*g.tolist(), None if y is None else float(np.rad2deg(y))] for g, y in self.goals],
                "path": self.path.round(3).tolist(), "node": self.active, "message": self.message,
                "source": self.source, "state": self.state}

    # ---- leaves --------------------------------------------------------
    def _have_goal(self, bb) -> str:
        return SUCCESS if self.goals else FAILURE

    def _path_valid(self, bb) -> str:
        if len(self.path) < 2:
            return FAILURE
        # re-check against the map a few times per second (it changes when mapping with lidar)
        if self._ticks % 25 == 0 and self.grid.blocked(self._remaining(), self.robot_radius, self.source):
            self.path = np.zeros((0, 2))
            return FAILURE
        return SUCCESS

    def _remaining(self) -> np.ndarray:
        if len(self.path) < 2:
            return self.path
        d = np.linalg.norm(self.path - self.pos, axis=1)
        i = int(np.argmin(d))
        return np.vstack([self.pos, self.path[min(i + 1, len(self.path) - 1):]])

    def _plan(self, bb) -> str:
        goal, yaw = self.goals[0]
        path = self.grid.plan(self.pos, goal, self.robot_radius, self.source)
        if len(path) < 2:
            if np.linalg.norm(goal - self.pos) < 0.3:   # already there: a turn on the spot is still a valid "path"
                path = np.array([self.pos, goal])
            else:
                self.message, self.state = "no path to goal - skipped", "failed"
                self.last_goal = self.goals.pop(0)
                return FAILURE
        self.path, self.message = path, ""
        self.new_path = (path, yaw)
        self._since_path = 0
        self.state = "walking"
        return SUCCESS

    def _follow(self, bb) -> str:
        mode = self.ctrl_status.get("mode")
        if self.new_path is not None:
            return RUNNING  # path planned this tick: the controller has not seen it yet, its status is stale
        self._since_path += 1
        if mode == "done" and self.goals and self._since_path < 400:
            # A "done" left over from the previous goal must not complete this one: only accept it
            # once the robot is actually near the current goal (or after a while, as a safety net).
            if np.linalg.norm(self.goals[0][0] - self.pos) > 0.5:
                return RUNNING
        if mode == "fallen":
            self.message, self.state = "robot fell", "fallen"
            self.last_goal = self.goals.pop(0)
            self.path = np.zeros((0, 2))
            return FAILURE
        if mode == "done":
            err = self.ctrl_status.get("goal_error_m")
            self.message = "goal reached" + (f" ({err:.3f} m)" if err is not None else "")
            self.last_goal = self.goals.pop(0)
            self.path = np.zeros((0, 2))
            self.state = "reached" if not self.goals else "walking"
            if not self.goals:
                self.new_path = (np.zeros((0, 2)), None)
            return SUCCESS
        return RUNNING
