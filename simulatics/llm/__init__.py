"""LLM pilots for the simulated robot.

    robot.py     RobotInterface - the fixed contract between pilots and a robot (sim or fake)
    senses.py    observation bundle: pose, navigation, arms, lidar sectors, objects, camera JPEG
    tools.py     strict JSON-schema tools mapped onto RobotInterface (Toolbox)
    claude.py    ClaudePilot - Claude through the Anthropic Messages API
    scripted.py  ScriptedPilot - offline regex pilot with the same interface
    pilot.py     PilotRunner - background mission lifecycle for the GUI (start / stop / status)
    fake.py      FakeRobot - kinematic RobotInterface for tests and offline demos
    mission.py   cancellation, budgets and transcript events shared by the pilots

Importing this package does not import the `anthropic` SDK; it is loaded when a Claude mission runs.
"""
from .claude import DEFAULT_MODEL, EFFORTS, MODELS, ClaudeConfig, ClaudePilot
from .fake import FakeObject, FakeRobot
from .mission import Cancelled, MissionContext, Outcome
from .pilot import PilotRunner, PilotSettings, options
from .robot import RobotInterface
from .scripted import ScriptedPilot
from .senses import Observation, find_object, observe
from .tools import TOOL_NAMES, ToolConfig, Toolbox, ToolResult, tool_definitions

__all__ = [
    "DEFAULT_MODEL", "EFFORTS", "MODELS", "ClaudeConfig", "ClaudePilot", "FakeObject", "FakeRobot", "Cancelled",
    "MissionContext", "Outcome", "PilotRunner", "PilotSettings", "options", "RobotInterface", "ScriptedPilot",
    "Observation", "find_object", "observe", "TOOL_NAMES", "ToolConfig", "Toolbox", "ToolResult",
    "tool_definitions",
]
