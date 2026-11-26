"""Motor Control: drive one motor precisely, watch its response, tune its gains."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QPushButton, QSplitter, QVBoxLayout, QWidget)

from . import theme
from .widgets import ParamSlider, PlotWidget, dim, heading

SIGNALS = [("hold", "Hold a target"), ("step", "Step"), ("sine", "Sine wave"), ("square", "Square wave"),
           ("chirp", "Chirp (frequency sweep)")]


class MotorPanel(QWidget):
    """One motor at a time: exact target entry (0.01 degree), test signals timed by the simulation clock, a scope
    of target, angle and torque recorded at every control step, measured response figures, and the gains."""

    def __init__(self, studio):
        super().__init__()
        self.studio, self.sess = studio, studio.sess
        self._quiet = False
        self._n = -1
        split = QSplitter(Qt.Horizontal)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(split)

        left = QWidget()
        lay = QVBoxLayout(left)
        lay.setContentsMargins(8, 6, 8, 8)
        self.stand = QCheckBox("Hold the robot on a stand (feet off the floor)")
        self.stand.setToolTip("Welds the torso in place so single joints can be driven without the robot falling. "
                              "Motors are then under manual control.")
        lay.addWidget(self.stand)
        self.note = dim("")
        lay.addWidget(self.note)
        lay.addWidget(heading("Motor"))
        self.motor = QComboBox()
        lay.addWidget(self.motor)
        self.target = ParamSlider("Target (°)", -180, 180, 0.0, decimals=2)
        self.target.spin.setSingleStep(0.5)
        lay.addWidget(self.target)
        row = QHBoxLayout()
        for label, fn in (("Zero", lambda: self._set_target(0.0)), ("Middle of range", self._middle),
                          ("All motors to start pose", self._all_home)):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, fn=fn: fn())
            row.addWidget(b)
        lay.addLayout(row)

        gb = QGroupBox("Test signal")
        f = QFormLayout(gb)
        self.kind = QComboBox()
        for k, label in SIGNALS:
            self.kind.addItem(label, k)
        self.amp = QDoubleSpinBox()
        self.amp.setRange(-180, 180)
        self.amp.setDecimals(2)
        self.amp.setValue(15.0)
        self.amp.setSuffix(" °")
        self.freq = QDoubleSpinBox()
        self.freq.setRange(0.05, 25)
        self.freq.setDecimals(2)
        self.freq.setValue(1.0)
        self.freq.setSuffix(" Hz")
        f.addRow("Signal", self.kind)
        f.addRow("Amplitude", self.amp)
        f.addRow("Frequency", self.freq)
        row = QHBoxLayout()
        self.run = QPushButton("Run signal")
        self.run.setProperty("primary", True)
        self.stop = QPushButton("Stop")
        row.addWidget(self.run)
        row.addWidget(self.stop)
        f.addRow(row)
        lay.addWidget(gb)

        gb = QGroupBox("Gains and limit")
        f = QFormLayout(gb)
        self.gains: dict[str, QDoubleSpinBox] = {}
        for field, label, hi, suffix in (("kp", "Stiffness kp", 1e5, " Nm/rad"), ("kv", "Damping kv", 1e4, " Nm s/rad"),
                                         ("torque", "Torque limit", 1e4, " Nm")):
            sp = QDoubleSpinBox()
            sp.setRange(0, hi)
            sp.setDecimals(2)
            sp.setSuffix(suffix)
            sp.setKeyboardTracking(False)
            sp.valueChanged.connect(lambda v, field=field: self._gain(field, v))
            f.addRow(label, sp)
            self.gains[field] = sp
        lay.addWidget(gb)

        gb = QGroupBox("Fit the model to a recording (system identification)")
        v = QVBoxLayout(gb)
        v.addWidget(dim("Adjusts this joint's damping, rotor inertia and dry friction until the simulation reproduces "
                        "a recorded response: the one on the scope, or a log from a real motor (CSV with time, "
                        "target, angle)."))
        self.fit_gains = QCheckBox("Also fit the servo gains kp and kv")
        v.addWidget(self.fit_gains)
        row = QHBoxLayout()
        for label, fn in (("Fit to the scope", self._fit_scope), ("Fit to a file...", self._fit_file),
                          ("Save recording...", self._save_rec), ("Self-test", self._self_test)):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, fn=fn: fn())
            row.addWidget(b)
        v.addLayout(row)
        self.fit_text = QLabel("")
        self.fit_text.setFont(theme.mono())
        self.fit_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.fit_text)
        lay.addWidget(gb)

        gb = QGroupBox("Measured response")
        v = QVBoxLayout(gb)
        self.metrics = QLabel("")
        self.metrics.setFont(theme.mono())
        self.metrics.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.metrics)
        lay.addWidget(gb)
        lay.addStretch(1)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.scope_angle = PlotWidget()
        self.scope_torque = PlotWidget()
        rl.addWidget(self.scope_angle, 3)
        rl.addWidget(self.scope_torque, 2)
        split.addWidget(right)
        split.setSizes([360, 700])

        self.stand.toggled.connect(self._stand)
        self.motor.activated.connect(lambda _i: self._select())
        self.target.changed.connect(lambda v: self._set_target(v, from_slider=True))
        self.run.clicked.connect(self._run)
        self.stop.clicked.connect(self._stop)
        self.sync()

    # -- helpers
    def _i(self) -> int:
        return max(0, self.motor.currentIndex())

    def _unit(self) -> float:
        """Degrees per model unit for the selected motor's joint (1 for sliding joints)."""
        m = self.sess.model
        j = int(m.actuator_trnid[self._i(), 0])
        return 180 / np.pi if m.jnt_type[j] == 3 else 1.0

    def _guard(self, fn, *a, **k):
        try:
            out = fn(*a, **k)
            if isinstance(out, str):
                self.studio.say(out)
            return True
        except Exception as e:  # noqa: BLE001
            self.studio.log(f"{e}")
            return False

    def _stand(self, on: bool) -> None:
        if self._quiet:
            return
        self._guard(self.sess.set_support_stand, on)
        self.studio.scene_changed()
        if on:
            with self.sess.lock:
                self.sess.running = True
        self.studio.viewport.frame_robot()

    def _manual(self) -> bool:
        return getattr(self.sess.controller, "name", "") in ("manual", "pd-hold")

    def _set_target(self, deg: float, from_slider: bool = False) -> None:
        if not from_slider:
            self.target.set_value(deg)
        if not self._manual():
            self.note.setText("Tick the stand first: while the robot balances itself, its controller sets the targets.")
            return
        c = self.sess.controller
        if hasattr(c, "signal"):
            c.signal = None
        self._guard(self.sess.set_actuator, self._i(), "ctrl", float(deg) / self._unit())
        with self.sess.lock:
            self.sess.running = True

    def _middle(self) -> None:
        a = self.sess.actuators()[self._i()]
        self._set_target((a["lo"] + a["hi"]) / 2 * self._unit())

    def _all_home(self) -> None:
        if not self._manual():
            return
        with self.sess.lock:
            s = self.sess
            if getattr(s.controller, "signal", None) is not None:
                s.controller.signal = None
            if s.task is not None:
                s.manual_ctrl[:] = s.task.ctrl0
        self._select()

    def _run(self) -> None:
        k = self._unit()
        if self._guard(self.sess.set_motor_signal, self._i(), self.kind.currentData(), self.amp.value() / k,
                       self.freq.value(), self.target.value() / k):
            self.sess.trace_motor(self._i())

    def _stop(self) -> None:
        c = self.sess.controller
        if getattr(c, "signal", None) is not None:
            c.signal = None

    # -- system identification
    def _fields(self) -> tuple:
        from .. import sysid
        return sysid.DEFAULT_FIT + (("kp", "kv") if self.fit_gains.isChecked() else ())

    def _show_fit(self, r: dict, title: str) -> None:
        lines = [title, f"{'':14s}{'before':>10s}{'fitted':>10s}" + (f"{'truth':>10s}" if "truth" in r else "")]
        for f in r["fields"]:
            lines.append(f"{f:14s}{r['before'][f]:10.4g}{r['fitted'][f]:10.4g}" + (f"{r['truth'][f]:10.4g}" if "truth" in r and f in r["truth"] else ""))
        k = self._unit()
        lines.append(f"error (rms)   {r['rms_before'] * k:10.4f}{r['rms_after'] * k:10.4f}  {'deg' if k != 1 else 'm'}")
        self.fit_text.setText("\n".join(lines))
        self.fit_result = r

    def _fit(self, rec: dict, title: str, apply: bool = True) -> None:
        from .. import sysid
        a = self._i()
        fields = self._fields()
        self.fit_text.setText("fitting...")

        def done(result, error):
            if error:
                self.fit_text.setText(error)
                return
            self._show_fit(result, title)
            if apply and result["improved"]:
                sysid.apply(self.sess, a, result)
                self._select()
                self.studio.log(f"Fitted {self.motor.currentText()}: error {result['rms_before']:.4f} -> {result['rms_after']:.4f} rad; values applied")
        self.studio.run_async(lambda: sysid.fit_motor(self.sess, {**rec, "actuator": a}, fields), done, "sysid")

    def _fit_scope(self) -> None:
        from .. import sysid
        try:
            self._fit(sysid.record(self.sess, self._i()), "Fitted to the scope recording (values applied)")
        except Exception as e:  # noqa: BLE001
            self.fit_text.setText(str(e))

    def _fit_file(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        from .. import sysid
        path, _ = QFileDialog.getOpenFileName(self, "A recorded response: time, target, angle", "", "Table (*.csv)")
        if path:
            try:
                self._fit(sysid.load_csv(path), "Fitted to the file (values applied)")
            except Exception as e:  # noqa: BLE001
                self.fit_text.setText(str(e))

    def _save_rec(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        from .. import sysid
        try:
            rec = sysid.record(self.sess, self._i())
        except Exception as e:  # noqa: BLE001
            self.fit_text.setText(str(e))
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save the recording", f"{self.motor.currentText()}.csv", "Table (*.csv)")
        if path:
            sysid.save_csv(rec, path)
            self.studio.log(f"Recording saved to {path}")

    def _self_test(self) -> None:
        """Hide a change in this joint, record, and fit it back: shows that the fit recovers what it should."""
        from .. import sysid
        a = self._i()
        self.fit_text.setText("self-test: hiding a change, recording 4 s, fitting...")

        def done(result, error):
            if error:
                self.fit_text.setText(error)
            else:
                self._show_fit(result, "Self-test: a hidden change, found again from the recording")
        self.studio.run_async(lambda: sysid.self_test(self.sess, a), done, "sysid self-test")

    def _gain(self, field: str, value: float) -> None:
        if not self._quiet:
            self._guard(self.sess.set_actuator, self._i(), field, value)

    def _select(self) -> None:
        acts = self.sess.actuators()
        if not acts:
            return
        a = acts[min(self._i(), len(acts) - 1)]
        k = self._unit()
        self._quiet = True
        lo, hi = (a["lo"] * k, a["hi"] * k) if a["hi"] > a["lo"] else (-180.0, 180.0)
        self.target.lo, self.target.hi = lo, hi
        self.target.spin.setRange(lo, hi)
        self.target.label.setText("Target (°)" if k != 1.0 else "Target (m)")
        self.target.set_value(float(np.clip(a["ctrl"] * k, lo, hi)))
        for field, sp in self.gains.items():
            sp.setValue(a[field])
        self._quiet = False
        self.sess.trace_motor(self._i())

    def sync(self) -> None:
        """After the model changed."""
        acts = self.sess.actuators()
        cur = self.motor.currentText()
        self.motor.blockSignals(True)
        self.motor.clear()
        self.motor.addItems([a["name"] for a in acts])
        if cur and self.motor.findText(cur) >= 0:
            self.motor.setCurrentText(cur)
        self.motor.blockSignals(False)
        self._quiet = True
        self.stand.setChecked(bool(self.sess.support_stand))
        self._quiet = False
        self._select()

    def refresh(self) -> None:
        if not self.isVisible():
            return
        s = self.sess
        self.note.setText("" if self._manual() else
                          f"The '{getattr(s.controller, 'name', '')}' controller is setting the targets. "
                          "The scope shows what it commands; tick the stand to drive motors yourself.")
        with s.lock:
            tr = np.array(s.motor_trace) if len(s.motor_trace) else np.zeros((0, 5))
        if len(tr) == self._n or len(tr) < 2:
            return
        self._n = len(tr)
        k = self._unit()
        unit = "°" if k != 1.0 else "m"
        tr = tr[-1500:]
        t = tr[:, 0] - tr[0, 0]
        self.scope_angle.set_series({"title": f"{self.motor.currentText()}: target and angle ({unit})", "x": t.tolist(),
                                     "y": {"target": (tr[:, 1] * k).tolist(), "angle": (tr[:, 2] * k).tolist()},
                                     "xlabel": "seconds"})
        self.scope_torque.set_series({"title": "motor torque (Nm)", "x": t.tolist(), "y": {"torque": tr[:, 4].tolist()},
                                      "xlabel": "seconds"})
        err = (tr[:, 1] - tr[:, 2]) * k
        limit = self.gains["torque"].value() or 1.0
        lines = [f"error now      {err[-1]:+8.3f} {unit}", f"error rms      {np.sqrt(np.mean(err ** 2)):8.3f} {unit}",
                 f"error max      {np.abs(err).max():8.3f} {unit}", f"peak torque    {np.abs(tr[:, 4]).max():8.2f} Nm",
                 f"at the limit   {100 * np.mean(np.abs(tr[:, 4]) >= 0.98 * limit):8.1f} % of the time",
                 f"peak speed     {np.abs(tr[:, 3]).max() * k:8.1f} {unit}/s"]
        step = np.flatnonzero(np.abs(np.diff(tr[:, 1])) > 1e-6)
        if len(step) == 1:                               # a single step: rise time and overshoot
            i0 = step[0] + 1
            a, b = tr[i0 - 1, 1], tr[i0, 1]
            y = (tr[i0:, 2] - a) / (b - a) if abs(b - a) > 1e-9 else None
            if y is not None and len(y) > 3:
                r10 = np.flatnonzero(y >= 0.1)
                r90 = np.flatnonzero(y >= 0.9)
                if len(r10) and len(r90):
                    lines.append(f"rise 10-90 %   {1000 * (tr[i0 + r90[0], 0] - tr[i0 + r10[0], 0]):8.0f} ms")
                lines.append(f"overshoot      {100 * max(0.0, float(y.max()) - 1.0):8.1f} %")
                lines.append(f"settles at     {100 * float(y[-1]):8.1f} % of the step")
        self.metrics.setText("\n".join(lines))
