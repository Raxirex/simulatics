"""Validation panel: run a behaviour under randomised physics and under other solver settings."""
from __future__ import annotations

import threading

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSpinBox, QTableWidget,
                               QVBoxLayout, QWidget)

from .widgets import dim, fill_table


class ValidationPanel(QWidget):
    """Does it still work when the physics is a little different? One row per variant, pass or fail."""
    row = Signal(float, object)
    done = Signal(object, object)

    def __init__(self, studio):
        super().__init__()
        from .. import samples
        self.studio = studio
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._rows: list[dict] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 8)
        lay.addWidget(dim("Runs a task on copies of the robot whose physics differ: random friction, masses, motor "
                          "strength, stiffness, damping, a payload and a shove; and other time steps, integrators, "
                          "friction cones and solvers. What only works on the unchanged model is exploiting the simulator."))
        row = QHBoxLayout()
        self.task = QComboBox()
        for label, key in (("Stand (6 s)", "stand"), ("Walk 3 m", "walk"), ("Walk a square", "square")):
            self.task.addItem(label, key)
        phys = [k for k, v in samples.physical().items() if v.get("verified")]
        try:
            from .. import skills
            phys += list(skills.shipped())
        except Exception:  # noqa: BLE001
            pass
        for sid in phys:
            self.task.addItem(f"Recording: {sid.replace('_', ' ')}", f"recording:{sid}")
        self.suite = QComboBox()
        for label, key in (("Random physics and solver settings", "both"), ("Random physics", "random"), ("Solver settings", "settings")):
            self.suite.addItem(label, key)
        self.n = QSpinBox()
        self.n.setRange(1, 200)
        self.n.setValue(8)
        self.n.setPrefix("random variants ")
        self.run_btn = QPushButton("Run sweep")
        self.run_btn.setProperty("primary", True)
        self.stop_btn = QPushButton("Stop")
        for w in (self.task, self.suite, self.n, self.run_btn, self.stop_btn):
            row.addWidget(w)
        row.addStretch(1)
        lay.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        lay.addWidget(self.bar)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("font-weight: 600;")
        lay.addWidget(self.summary)
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        lay.addWidget(self.table, 1)
        self.run_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self._cancel.set)
        self.row.connect(self._on_row)
        self.done.connect(self._on_done)

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        from .. import validate
        if self.running():
            return
        self._rows = []
        self._cancel.clear()
        self.summary.setText("Running...")
        self.run_btn.setEnabled(False)
        task, suite, n = self.task.currentData(), self.suite.currentData(), self.n.value()

        def work():
            try:
                rep = validate.run(task, suite, n=n, progress=lambda f, r: self.row.emit(f, r), cancel=self._cancel)
                self.done.emit(rep, None)
            except Exception as e:  # noqa: BLE001
                self.done.emit(None, f"{type(e).__name__}: {e}")
        self._thread = threading.Thread(target=work, name="validation", daemon=True)
        self._thread.start()

    def _on_row(self, frac: float, r: dict) -> None:
        self._rows.append(r)
        self.bar.setValue(int(1000 * frac))
        fill_table(self.table, self._rows, ["variant", "result", "error_m", "sim_seconds", "note", "changes"])
        for i, rr in enumerate(self._rows):
            it = self.table.item(i, 1)
            if it is not None:
                it.setForeground(Qt.red if rr["result"] != "pass" else Qt.green)

    def _on_done(self, rep, error) -> None:
        from .. import validate
        self.run_btn.setEnabled(True)
        if error:
            self.summary.setText(f"Could not run: {error}")
            self.studio.log(f"Validation failed to run: {error}")
            return
        self.report = rep
        self.summary.setText(validate.summary(rep) + f"   ({rep['seconds']:.0f} s)")
        self.studio.log("Validation: " + validate.summary(rep))
