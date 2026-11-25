"""World Model panel: plug a model in, give it an instruction in words, watch what it does. A placeholder slot."""
from __future__ import annotations

import time

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

from . import theme
from .widgets import dim, heading

ABOUT = ("Placeholder. This is the socket for a world model: something that takes what the robot senses plus an "
         "instruction in words (typed here, or produced by a language model) and returns what to do: joint targets, a "
         "walking goal, or a named skill. No trained world model ships with Simulatics. The built-in stand-in only "
         "matches phrases such as \"walk to 2 1\", \"walk forward 1.5\", \"backflip\", \"jump\", \"jump onto the box\", "
         "\"kitchen\", \"stop\".")
INTERFACE = '''# my_model.py: load it with "Load model file..."
from simulatics.worldmodel import Action

class Mine:
    name = "my model"
    def reset(self, info):            # info["motors"]: joint names that have a motor
        pass
    def act(self, obs, instruction):  # called several times a second
        # obs.joint_pos / joint_vel (dict by joint name), obs.root_pos, obs.root_quat,
        # obs.root_lin_vel, obs.root_ang_vel, obs.foot_contact, obs.objects,
        # obs.image (if a camera is chosen), obs.lidar (if ticked)
        return Action.go(1.0, 0.0, say="walking to (1, 0)")
        # or Action.drive({"left_knee": 0.6, ...})   drive the motors yourself
        # or Action.do("backflip")                   run a named skill (any sample id, or "stop")
        # or Action.none()

def create():
    return Mine()
'''


class WorldModelPanel(QWidget):
    def __init__(self, studio):
        super().__init__()
        from ..worldmodel import WorldModelRunner
        self.studio, self.sess = studio, studio.sess
        self.runner = WorldModelRunner(self.sess)
        self._shown = 0
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 8)
        note = dim(ABOUT)
        lay.addWidget(note)
        lay.addWidget(heading("Model"))
        row = QHBoxLayout()
        self.model_name = QLabel(self.runner.model.name)
        self.model_name.setStyleSheet("font-weight: 600;")
        load = QPushButton("Load model file...")
        builtin = QPushButton("Use the stand-in")
        iface = QPushButton("Show the interface")
        row.addWidget(self.model_name, 1)
        row.addWidget(load)
        row.addWidget(builtin)
        row.addWidget(iface)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Calls per second"))
        self.rate = QDoubleSpinBox()
        self.rate.setRange(0.2, 50)
        self.rate.setValue(self.runner.rate_hz)
        row.addWidget(self.rate)
        row.addWidget(QLabel("  Camera image"))
        self.camera = QComboBox()
        row.addWidget(self.camera)
        self.lidar = QCheckBox("Lidar points")
        row.addWidget(self.lidar)
        row.addStretch(1)
        lay.addLayout(row)

        lay.addWidget(heading("Instruction (text from you or from a language model)"))
        row = QHBoxLayout()
        self.text = QLineEdit()
        self.text.setPlaceholderText("for example: walk to 2 1     or     do a backflip")
        self.send = QPushButton("Send")
        self.send.setProperty("primary", True)
        self.stop = QPushButton("Stop")
        row.addWidget(self.text, 1)
        row.addWidget(self.send)
        row.addWidget(self.stop)
        lay.addLayout(row)
        self.state = dim("idle")
        lay.addWidget(self.state)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(theme.mono())
        self.log.setMaximumBlockCount(1000)
        lay.addWidget(self.log, 1)

        load.clicked.connect(self._load)
        builtin.clicked.connect(self._builtin)
        iface.clicked.connect(self._interface)
        self.send.clicked.connect(self._send)
        self.text.returnPressed.connect(self._send)
        self.stop.clicked.connect(self._stop)
        self.rate.valueChanged.connect(lambda v: setattr(self.runner, "rate_hz", float(v)))
        self.camera.activated.connect(lambda _i: setattr(self.runner, "camera", self.camera.currentData()))
        self.lidar.toggled.connect(lambda on: setattr(self.runner, "lidar", bool(on)))
        self.sync()

    def _load(self) -> None:
        from ..worldmodel import load_world_model
        path, _ = QFileDialog.getOpenFileName(self, "Load a world model", "", "Python (*.py)")
        if not path:
            return
        ok = QMessageBox.question(self, "Run this file?",
                                  f"Loading a model runs the code in\n{path}\nwith full access to this computer. "
                                  "Only load files you trust.\n\nLoad it?")
        if ok != QMessageBox.Yes:
            return
        try:
            self.runner.stop()
            self.runner.set_model(load_world_model(path))
            self.model_name.setText(getattr(self.runner.model, "name", type(self.runner.model).__name__))
        except Exception as e:  # noqa: BLE001
            self.studio.log(f"Could not load that model: {type(e).__name__}: {e}")

    def _builtin(self) -> None:
        from ..worldmodel import KeywordModel
        self.runner.stop()
        self.runner.set_model(KeywordModel())
        self.model_name.setText(self.runner.model.name)

    def _interface(self) -> None:
        self.studio.console.editor.setPlainText(INTERFACE)
        self.studio.show_dock("console")
        self.studio.console.tabs.setCurrentIndex(1)
        self.studio.say("The interface is in the Python panel's Script tab: save it as a file, then load it here.")

    def _send(self) -> None:
        self.runner.start(self.text.text())
        self.text.selectAll()

    def _stop(self) -> None:
        self.runner.stop()
        self.studio.scene_changed(light=True)

    def instruct(self, text: str) -> None:
        """For scripts and other panels: studio.world_model.instruct("walk to 2 1")."""
        self.text.setText(text)
        self._send()

    def sync(self) -> None:
        with self.sess.lock:
            names = [self.sess.model.camera(i).name for i in range(self.sess.model.ncam)]
        cur = self.camera.currentData()
        self.camera.blockSignals(True)
        self.camera.clear()
        self.camera.addItem("none", None)
        for n in names:
            self.camera.addItem(n, n)
        i = self.camera.findData(cur)
        self.camera.setCurrentIndex(max(i, 0))
        self.camera.blockSignals(False)
        self.runner.camera = self.camera.currentData()

    def refresh(self) -> None:
        r = self.runner
        if len(r.log) < self._shown:                 # the log was trimmed
            self._shown = 0
        if len(r.log) != self._shown:
            for t, line in r.log[self._shown:]:
                self.log.appendPlainText(f"{time.strftime('%H:%M:%S', time.localtime(t))}  {line}")
            self._shown = len(r.log)
            self.studio.scene_changed(light=True)
        if self.isVisible():
            self.state.setText(("running" if r.running else "idle") + f"   {r.calls} calls   last call {r.last_ms:.1f} ms"
                               + (f"   error: {r.error}" if r.error else ""))
