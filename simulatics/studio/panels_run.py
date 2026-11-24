"""Run-time panels: timeline, Python console, training, sensors, results, log, AI pilot."""
from __future__ import annotations

import code
import contextlib
import io
import time
import traceback
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF, QSyntaxHighlighter, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit, QProgressBar, QPushButton,
                               QSpinBox, QSplitter, QTableWidget, QTabWidget, QToolButton, QVBoxLayout, QWidget)

from . import theme
from .icons import icon
from .widgets import ImageView, MapView, PlotWidget, dim, fill_table, heading

STATE_SPEC = int(mujoco.mjtState.mjSTATE_FULLPHYSICS)


# ---------------------------------------------------------------------------------------------- timeline
class Recorder:
    """Rolling history of simulation states, so the timeline can scrub back through what just happened."""

    def __init__(self, session, interval: float = 0.05, seconds: float = 180.0):
        self.sess, self.interval = session, interval
        self.frames: deque = deque(maxlen=int(seconds / interval))   # (time, state, height, speed)
        self.falls: list[float] = []
        self._version, self._falls = -1, 0

    def clear(self) -> None:
        self.frames.clear()
        self.falls = []

    def sample(self) -> None:
        s = self.sess
        with s.lock:
            m, d = s.model, s.data
            if s.scene_version != self._version:
                self.clear()
                self._version, self._falls = s.scene_version, s.falls
            t = float(d.time)
            if self.frames and t < self.frames[-1][0] - 1e-9:
                if s.falls > self._falls:           # a fall auto-resets the robot: mark it, keep the history
                    self.falls.append(self.frames[-1][0])
                # time went backwards (reset or scrub): drop the future
                while self.frames and self.frames[-1][0] > t:
                    self.frames.pop()
            self._falls = s.falls
            if self.frames and t - self.frames[-1][0] < self.interval:
                return
            state = np.empty(mujoco.mj_stateSize(m, STATE_SPEC))
            mujoco.mj_getState(m, d, state, STATE_SPEC)
            free = s.task is not None
            self.frames.append((t, state, float(d.qpos[2]) if free else 0.0,
                                float(np.hypot(d.qvel[0], d.qvel[1])) if free else 0.0))

    def restore(self, index: int) -> None:
        s = self.sess
        with s.lock:
            if not self.frames or s.scene_version != self._version:
                return
            t, state, _, _ = self.frames[int(np.clip(index, 0, len(self.frames) - 1))]
            if len(state) != mujoco.mj_stateSize(s.model, STATE_SPEC):
                return
            s.running = False
            mujoco.mj_setState(s.model, s.data, state, STATE_SPEC)
            mujoco.mj_forward(s.model, s.data)


class TrackWidget(QWidget):
    """The timeline strip: recorded root height and speed against time, with a playhead you can drag."""
    scrubbed = Signal(int)

    def __init__(self, recorder: Recorder):
        super().__init__()
        self.rec = recorder
        self.setMinimumHeight(74)
        self.setCursor(Qt.SizeHorCursor)

    def _index_at(self, x: float) -> int:
        n = len(self.rec.frames)
        return int(np.clip(round((x - 48) / max(1.0, self.width() - 56) * (n - 1)), 0, max(0, n - 1)))

    def mousePressEvent(self, e) -> None:
        if self.rec.frames:
            self.scrubbed.emit(self._index_at(e.position().x()))

    mouseMoveEvent = mousePressEvent

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.BG))
        fr = list(self.rec.frames)
        area = QRectF(48, 16, self.width() - 56, self.height() - 22)
        p.setPen(QColor(theme.DIM))
        if len(fr) < 2:
            p.drawText(self.rect(), Qt.AlignCenter, "Press play: the last few minutes are recorded here and can be scrubbed.")
            return
        t = np.array([f[0] for f in fr])
        t0, t1 = float(t[0]), float(max(t[-1], t[0] + 1e-6))
        # time ruler
        step = next(s for s in (0.5, 1, 2, 5, 10, 30, 60, 120) if (t1 - t0) / s <= 12)
        k = np.ceil(t0 / step) * step
        while k <= t1:
            x = area.left() + (k - t0) / (t1 - t0) * area.width()
            p.setPen(QPen(QColor(theme.LINE), 1))
            p.drawLine(QPointF(x, 14), QPointF(x, self.height()))
            p.setPen(QColor(theme.DIM))
            p.drawText(QRectF(x + 3, 0, 60, 14), Qt.AlignLeft | Qt.AlignVCenter, f"{k:g}s")
            k += step
        half = area.height() / 2
        for row, (idx, label, col) in enumerate(((2, "height", "#4c9aff"), (3, "speed", "#3ecf8e"))):
            v = np.array([f[idx] for f in fr])
            lo, hi = float(v.min()), float(v.max())
            if hi - lo < 1e-6:
                lo, hi = lo - 0.5, hi + 0.5
            top = area.top() + row * half
            xs = area.left() + (t - t0) / (t1 - t0) * area.width()
            ys = top + half - 3 - (v - lo) / (hi - lo) * (half - 6)
            p.setPen(QPen(QColor(col), 1.4))
            p.drawPolyline(QPolygonF([QPointF(float(a), float(b)) for a, b in zip(xs, ys)]))
            p.setPen(QColor(col))
            p.drawText(QRectF(2, top, 44, half), Qt.AlignLeft | Qt.AlignVCenter, label)
        p.setPen(QPen(QColor(theme.BAD), 1.5, Qt.DashLine))
        for tf in self.rec.falls:
            if t0 <= tf <= t1:
                x = area.left() + (tf - t0) / (t1 - t0) * area.width()
                p.drawLine(QPointF(x, 14), QPointF(x, self.height()))
        now = float(self.rec.sess.data.time)
        x = area.left() + np.clip((now - t0) / (t1 - t0), 0, 1) * area.width()
        p.setPen(QPen(QColor("white"), 1.5))
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        p.setBrush(QColor("white"))
        p.drawPolygon(QPolygonF([QPointF(x - 5, 0), QPointF(x + 5, 0), QPointF(x, 8)]))


class TimelinePanel(QWidget):
    def __init__(self, studio):
        super().__init__()
        self.studio, self.sess = studio, studio.sess
        self.rec = Recorder(self.sess)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(4)
        bar = QHBoxLayout()
        self.play = self._btn("play", "Play / pause (Space)", studio.toggle_play)
        self.step = self._btn("step", "Advance one control step (.)", self.sess.step_once)
        self.reset = self._btn("reset", "Reset the robot to its start pose (R)", studio.reset_sim)
        for b in (self.play, self.step, self.reset):
            bar.addWidget(b)
        self.speed = QComboBox()
        for v in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            self.speed.addItem(f"{v:g}x", v)
        self.speed.setCurrentIndex(3)
        self.speed.setToolTip("Simulation speed relative to real time")
        self.speed.activated.connect(lambda _i: setattr(self.sess, "speed", float(self.speed.currentData())))
        bar.addWidget(self.speed)
        self.auto_reset = QCheckBox("Stand back up after a fall")
        self.auto_reset.setChecked(self.sess.auto_reset)
        self.auto_reset.toggled.connect(lambda on: setattr(self.sess, "auto_reset", bool(on)))
        bar.addWidget(self.auto_reset)
        bar.addStretch(1)
        self.readout = QLabel("")
        self.readout.setFont(theme.mono())
        bar.addWidget(self.readout)
        lay.addLayout(bar)
        self.track = TrackWidget(self.rec)
        self.track.scrubbed.connect(self.rec.restore)
        lay.addWidget(self.track, 1)

    def _btn(self, name: str, tip: str, fn) -> QToolButton:
        b = QToolButton()
        b.setIcon(icon(name))
        b.setToolTip(tip)
        b.setAutoRaise(True)
        b.clicked.connect(lambda: fn())
        return b

    def refresh(self) -> None:
        self.rec.sample()
        s = self.sess
        self.play.setIcon(icon("pause" if s.running else "play"))
        n = s.nav.state
        self.readout.setText(f"t {float(s.data.time):8.2f} s   falls {s.falls}   nav {n}")
        if self.isVisible():
            self.track.update()


# ----------------------------------------------------------------------------------------------- console
class _PyHighlighter(QSyntaxHighlighter):
    KEYWORDS = ("def class return if elif else for while in not and or import from as with try except finally "
                "raise lambda yield pass break continue None True False is assert global nonlocal").split()

    def __init__(self, doc):
        super().__init__(doc)
        import re
        self.rules = []
        for pattern, colour, bold in ((r"\b(" + "|".join(self.KEYWORDS) + r")\b", "#c792ea", True),
                                      (r"\b\d+(\.\d+)?([eE][-+]?\d+)?\b", "#f78c6c", False),
                                      (r"(\"[^\"\n]*\"|'[^'\n]*')", "#c3e88d", False), (r"#[^\n]*", "#676e95", False)):
            f = QTextCharFormat()
            f.setForeground(QColor(colour))
            if bold:
                f.setFontWeight(QFont.DemiBold)
            self.rules.append((re.compile(pattern), f))

    def highlightBlock(self, text: str) -> None:
        for rx, fmt in self.rules:
            for mt in rx.finditer(text):
                self.setFormat(mt.start(), mt.end() - mt.start(), fmt)


class _HistoryLine(QLineEdit):
    def __init__(self):
        super().__init__()
        self.history: list[str] = []
        self._pos = 0

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_Up and self.history:
            self._pos = max(0, self._pos - 1)
            self.setText(self.history[self._pos])
        elif e.key() == Qt.Key_Down and self.history:
            self._pos = min(len(self.history), self._pos + 1)
            self.setText(self.history[self._pos] if self._pos < len(self.history) else "")
        else:
            super().keyPressEvent(e)

    def remember(self, text: str) -> None:
        if text.strip():
            self.history.append(text)
        self._pos = len(self.history)


SCRIPT_TEMPLATE = '''# Runs inside Studio. In scope: session, viewport, graph, studio, np, mujoco.
# Take session.lock when you touch session.model / session.data directly.
import numpy as np

with session.lock:
    m, d = session.model, session.data
    print(f"{m.nbody - 1} bodies, {m.nu} motors, t = {d.time:.2f} s")
    print("root position:", np.round(d.qpos[:3], 3))

session.nav.set_goal([2.0, 1.0])   # walk to x = 2, y = 1
session.running = True
'''


class ConsolePanel(QWidget):
    """A Python prompt and a script editor with the live session in scope."""

    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self.ns = {"session": studio.sess, "sess": studio.sess, "studio": studio, "viewport": studio.viewport,
                   "np": np, "mujoco": mujoco, "__name__": "__console__"}
        self.interp = code.InteractiveConsole(self.ns)
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(tabs)

        con = QWidget()
        cl = QVBoxLayout(con)
        cl.setContentsMargins(4, 4, 4, 4)
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setFont(theme.mono())
        self.out.setMaximumBlockCount(4000)
        row = QHBoxLayout()
        self.prompt = QLabel(">>>")
        self.prompt.setFont(theme.mono())
        self.line = _HistoryLine()
        self.line.setFont(theme.mono())
        self.line.setPlaceholderText("Python, for example: session.nav.set_goal([2, 0])")
        row.addWidget(self.prompt)
        row.addWidget(self.line, 1)
        cl.addWidget(self.out, 1)
        cl.addLayout(row)
        tabs.addTab(con, "Console")

        scr = QWidget()
        sl = QVBoxLayout(scr)
        sl.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        self.run_btn = QPushButton("Run script")
        self.run_btn.setProperty("primary", True)
        open_btn, save_btn = QPushButton("Open..."), QPushButton("Save as...")
        bar.addWidget(self.run_btn)
        bar.addWidget(open_btn)
        bar.addWidget(save_btn)
        bar.addStretch(1)
        bar.addWidget(dim("Output goes to the Console tab. Long scripts run in the background."))
        self.editor = QPlainTextEdit(SCRIPT_TEMPLATE)
        self.editor.setFont(theme.mono())
        self.editor.setTabStopDistance(4 * self.editor.fontMetrics().horizontalAdvance(" "))
        self._hl = _PyHighlighter(self.editor.document())
        sl.addLayout(bar)
        sl.addWidget(self.editor, 1)
        tabs.addTab(scr, "Script")
        self.tabs = tabs

        self.line.returnPressed.connect(self._enter)
        self.run_btn.clicked.connect(self.run_script)
        open_btn.clicked.connect(self._open)
        save_btn.clicked.connect(self._save)
        self.write("Simulatics console. In scope: session, viewport, graph, studio, np, mujoco.\n")

    def write(self, text: str) -> None:
        if not text:
            return
        self.out.moveCursor(QTextCursor.End)
        self.out.insertPlainText(text)
        self.out.moveCursor(QTextCursor.End)

    def _enter(self) -> None:
        text = self.line.text()
        self.line.clear()
        self.line.remember(text)
        self.write(f"{self.prompt.text()} {text}\n")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                more = self.interp.push(text)
            except SystemExit:
                more = False
                print("exit is disabled here: close the window instead")
        self.write(buf.getvalue())
        self.prompt.setText("..." if more else ">>>")
        self.studio.scene_changed(light=True)

    def run_script(self, source: str | None = None) -> None:
        src = source if isinstance(source, str) else self.editor.toPlainText()
        self.run_btn.setEnabled(False)
        self.tabs.setCurrentIndex(0)
        self.write("--- running script ---\n")

        def work():
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                try:
                    exec(compile(src, "<script>", "exec"), self.ns)   # noqa: S102 - the user's own script
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
            return buf.getvalue()

        def done(result, error):
            self.run_btn.setEnabled(True)
            self.write((result or "") + (f"{error}\n" if error else "") + "--- done ---\n")
            self.studio.scene_changed(light=True)
        self.studio.run_async(work, done, "script")

    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open script", "", "Python (*.py)")
        if path:
            self.editor.setPlainText(Path(path).read_text(encoding="utf-8"))

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save script", "script.py", "Python (*.py)")
        if path:
            Path(path).write_text(self.editor.toPlainText(), encoding="utf-8")


# ---------------------------------------------------------------------------------------------- training
ALGOS = [("ppo", "PPO velocity tracking (reinforcement learning)"),
         ("distill", "Distil the model-based walker into a network"),
         ("imitation", "Motion imitation (track a reference clip)"),
         ("es", "Evolution strategy"), ("ga", "Genetic algorithm")]


class TrainPanel(QWidget):
    """Start a training job, watch its curve, load the result."""

    def __init__(self, studio):
        super().__init__()
        from ..jobs import TRAIN_PRESETS
        self.studio, self.sess, self.presets = studio, studio.sess, TRAIN_PRESETS
        self.trainer = None
        self._t = 0.0
        split = QSplitter(Qt.Horizontal)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)

        left = QWidget()
        f = QFormLayout(left)
        f.setContentsMargins(8, 8, 8, 8)
        self.algo = QComboBox()
        for k, label in ALGOS:
            self.algo.addItem(label, k)
        self.name = QLineEdit("run")
        self.population, self.generations, self.threads = QSpinBox(), QSpinBox(), QSpinBox()
        self.population.setRange(1, 4096)
        self.generations.setRange(1, 1000000)
        self.threads.setRange(0, 64)
        self.threads.setSpecialValueText("auto")
        self.lr, self.seconds = QDoubleSpinBox(), QDoubleSpinBox()
        self.lr.setDecimals(5)
        self.lr.setRange(0.000001, 0.1)
        self.lr.setSingleStep(0.0005)
        self.seconds.setRange(1, 120)
        self.motion = QLineEdit("walk")
        self.cont = QCheckBox("Continue from the loaded policy")
        f.addRow("Method", self.algo)
        f.addRow("Run name", self.name)
        self.pop_label, self.gen_label = QLabel("Parallel robots"), QLabel("Iterations")
        f.addRow(self.pop_label, self.population)
        f.addRow(self.gen_label, self.generations)
        f.addRow("Learning rate", self.lr)
        f.addRow("Episode length (s)", self.seconds)
        f.addRow("CPU threads", self.threads)
        f.addRow("Motion clip", self.motion)
        f.addRow(self.cont)
        row = QHBoxLayout()
        self.start = QPushButton("Start training")
        self.start.setProperty("primary", True)
        self.stop = QPushButton("Stop")
        row.addWidget(self.start)
        row.addWidget(self.stop)
        f.addRow(row)
        self.status = dim("idle")
        self.bar = QProgressBar()
        f.addRow(self.bar)
        f.addRow(self.status)
        split.addWidget(left)

        self.plot = PlotWidget()
        split.addWidget(self.plot)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 8, 8, 8)
        rl.addWidget(heading("Trained policies"))
        self.policies = QListWidget()
        rl.addWidget(self.policies, 1)
        row = QHBoxLayout()
        load, unload, refresh = QPushButton("Load into robot"), QPushButton("Unload"), QPushButton("Refresh")
        for b in (load, unload, refresh):
            row.addWidget(b)
        rl.addLayout(row)
        split.addWidget(right)
        split.setSizes([300, 520, 300])

        self.algo.activated.connect(lambda _i: self._preset())
        self.start.clicked.connect(self._start)
        self.stop.clicked.connect(lambda: self.trainer and self.trainer.stop())
        load.clicked.connect(self._load_policy)
        unload.clicked.connect(lambda: (self.sess.unload_policy(), self.studio.scene_changed()))
        refresh.clicked.connect(self.list_policies)
        self.policies.itemDoubleClicked.connect(lambda _it: self._load_policy())
        self._preset()
        self.list_policies()

    def _preset(self) -> None:
        a = self.algo.currentData()
        p = self.presets[a]
        self.population.setValue(int(p["population"]))
        self.generations.setValue(int(p["generations"]))
        self.lr.setValue(float(p["lr"]))
        self.threads.setValue(int(p["threads"]))
        self.seconds.setValue(float(p["episode_seconds"]))
        self.pop_label.setText("Population" if a in ("es", "ga") else "Parallel robots")
        self.gen_label.setText({"es": "Generations", "ga": "Generations", "distill": "Rounds"}.get(a, "Iterations"))
        self.motion.setEnabled(a == "imitation")

    def _start(self) -> None:
        from ..jobs import make_trainer_for_session
        if self.trainer is not None and self.trainer.status().get("state") == "running":
            self.studio.log("Training is already running.")
            return
        d = {"algo": self.algo.currentData(), "name": self.name.text(), "population": self.population.value(),
             "generations": self.generations.value(), "lr": self.lr.value(), "threads": self.threads.value(),
             "episode_seconds": self.seconds.value(), "motion": self.motion.text(),
             "continue_from_loaded": self.cont.isChecked()}
        try:
            self.trainer = make_trainer_for_session(self.sess, d)
            self.trainer.start()
            self.studio.log(f"Training started: {d['algo']} '{d['name']}'")
        except Exception as e:  # noqa: BLE001
            self.studio.log(f"Could not start training: {e}")

    def list_policies(self) -> None:
        from ..jobs import list_policies
        self.policies.clear()
        for p in list_policies():
            fit = p.get("fitness")
            fit = "" if fit is None or not np.isfinite(fit) else f"   score {fit:.2f}"
            it = QListWidgetItem(f"{p['name']}   [{p['kind']}]{fit}")
            it.setData(Qt.UserRole, p["path"])
            self.policies.addItem(it)

    def _load_policy(self) -> None:
        it = self.policies.currentItem()
        if it is None:
            return
        path = Path(it.data(Qt.UserRole))
        try:
            if path.name == "imitation_policy.npz":
                self.sess.play_motion(None, "policy", str(path))
                self.sess.policy_name = path.parent.name
            else:
                self.sess.load_policy(str(path))
            self.studio.log(f"Loaded policy {path.parent.name}")
        except Exception as e:  # noqa: BLE001
            self.studio.log(f"Could not load that policy: {e}")
        self.studio.scene_changed()

    def job_state(self) -> str:
        return self.trainer.status().get("state", "idle") if self.trainer is not None else "idle"

    def refresh(self) -> None:
        now = time.monotonic()
        if self.trainer is None or now - self._t < 0.5:
            return
        self._t = now
        st = self.trainer.status()
        gen, gens = int(st.get("generation") or 0), int(st.get("generations") or 1)
        self.bar.setRange(0, max(1, gens))
        self.bar.setValue(min(gen, gens))
        best = st.get("best")
        self.status.setText(f"{st.get('state')}   {gen} / {gens}" + (f"   best {best:.3f}" if isinstance(best, float) else "")
                            + (f"\n{st['error']}" if st.get("error") else ""))
        hist = st.get("history") or []
        if hist and self.isVisible():
            xs = [h.get("gen", i + 1) for i, h in enumerate(hist)]
            ys = {k: [h.get(k) for h in hist] for k in ("best", "mean") if any(h.get(k) is not None for h in hist)}
            self.plot.set_series({"title": f"Training: {st.get('name', '')}", "x": xs, "y": ys, "xlabel": "iteration"})
        if st.get("state") in ("finished", "stopped", "error") and getattr(self, "_last_state", "") == "running":
            self.list_policies()
            self.studio.log(f"Training {st.get('state')}: {st.get('name', '')}")
        self._last_state = st.get("state")


# ----------------------------------------------------------------------------------------------- sensors
class SensorsPanel(QWidget):
    """What the robot perceives: a camera, the lidar, and the map built from it."""

    def __init__(self, studio):
        super().__init__()
        self.studio, self.sess = studio, studio.sess
        self._cam_seq = -1
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(tabs)

        mapw = QWidget()
        ml = QVBoxLayout(mapw)
        ml.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        self.source = QComboBox()
        for label, k in (("True map", "truth"), ("Lidar map", "lidar"), ("SLAM map", "slam")):
            self.source.addItem(label, k)
        self.follow = QCheckBox("Follow robot")
        self.follow.setChecked(True)
        self.plan_on = QComboBox()
        self.plan_on.addItem("Plan on the true map", "truth")
        self.plan_on.addItem("Plan on the lidar map", "lidar")
        row.addWidget(self.source)
        row.addWidget(self.plan_on)
        row.addWidget(self.follow)
        row.addStretch(1)
        self.map = MapView(self.sess)
        ml.addLayout(row)
        ml.addWidget(self.map, 1)
        tabs.addTab(mapw, "Map")

        camw = QWidget()
        cl = QVBoxLayout(camw)
        cl.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        self.cam_on = QCheckBox("Live")
        self.cam_name = QComboBox()
        self.depth = QCheckBox("Depth")
        row.addWidget(self.cam_on)
        row.addWidget(self.cam_name, 1)
        row.addWidget(self.depth)
        self.image = ImageView()
        self.image.text = "Tick Live to see through the robot's camera"
        cl.addLayout(row)
        cl.addWidget(self.image, 1)
        tabs.addTab(camw, "Camera")

        lid = QWidget()
        f = QFormLayout(lid)
        f.setContentsMargins(8, 8, 8, 8)
        c = self.sess.lidar_cfg
        self.l_on = QCheckBox("Lidar scanning")
        self.l_on.setChecked(c.enabled)
        self.l_h, self.l_c = QSpinBox(), QSpinBox()
        self.l_h.setRange(8, 720)
        self.l_c.setRange(1, 64)
        self.l_h.setValue(c.h_samples)
        self.l_c.setValue(c.channels)
        self.l_vmin, self.l_vmax, self.l_range = QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()
        for sp, lo, hi, v in ((self.l_vmin, -89, 89, c.v_min), (self.l_vmax, -89, 89, c.v_max), (self.l_range, 1, 40, c.max_range)):
            sp.setRange(lo, hi)
            sp.setValue(v)
        f.addRow(self.l_on)
        f.addRow("Rays per turn", self.l_h)
        f.addRow("Vertical layers", self.l_c)
        f.addRow("Lowest ray (°)", self.l_vmin)
        f.addRow("Highest ray (°)", self.l_vmax)
        f.addRow("Range (m)", self.l_range)
        self.l_note = dim("")
        f.addRow(self.l_note)
        tabs.addTab(lid, "Lidar")

        self.source.activated.connect(lambda _i: (setattr(self.map, "source", self.source.currentData()), self.map.update()))
        self.plan_on.activated.connect(lambda _i: setattr(self.sess.nav, "source", self.plan_on.currentData()))
        self.follow.toggled.connect(lambda on: setattr(self.map, "follow", bool(on)))
        self.map.goalClicked.connect(self._goal)
        self.cam_on.toggled.connect(self._camera)
        self.cam_name.activated.connect(lambda _i: self._camera())
        self.depth.toggled.connect(lambda _on: self._camera())
        self.l_on.toggled.connect(lambda _on: self._lidar())
        for sp in (self.l_h, self.l_c, self.l_vmin, self.l_vmax, self.l_range):
            sp.editingFinished.connect(self._lidar)
        self.sync()

    def _goal(self, x: float, y: float, append: bool) -> None:
        with self.sess.lock:
            if self.sess.task is None:
                self.studio.log("This model cannot walk.")
                return
            self.sess.nav.set_goal([x, y], append=append)
            self.sess.running = True
        self.studio.log(f"Goal set at ({x:.2f}, {y:.2f}) m")

    def _camera(self, *_a) -> None:
        with self.sess.lock:
            self.sess.camera_cfg.update(enabled=self.cam_on.isChecked(), depth=self.depth.isChecked(),
                                        name=self.cam_name.currentText() or "head_cam")
        if not self.cam_on.isChecked():
            self.image.set_jpeg(None)

    def _lidar(self) -> None:
        self.sess.set_lidar({"enabled": self.l_on.isChecked(), "h_samples": self.l_h.value(),
                             "channels": self.l_c.value(), "v_min": self.l_vmin.value(), "v_max": self.l_vmax.value(),
                             "max_range": self.l_range.value()})

    def sync(self) -> None:
        with self.sess.lock:
            names = [self.sess.model.camera(i).name for i in range(self.sess.model.ncam)]
            mounted = self.sess.lidar.mounted
        cur = self.cam_name.currentText() or self.sess.camera_cfg["name"]
        self.cam_name.blockSignals(True)
        self.cam_name.clear()
        self.cam_name.addItems(names or ["(free camera)"])
        if cur in names:
            self.cam_name.setCurrentText(cur)
        self.cam_name.blockSignals(False)
        self.l_note.setText("" if mounted else "No lidar is mounted on this robot. Place one from the Assets panel.")

    def refresh(self) -> None:
        if not self.isVisible():
            return
        s = self.sess
        self.map.update()
        if s.camera_cfg["enabled"] and s.camera_seq != self._cam_seq:
            self._cam_seq = s.camera_seq
            self.image.set_jpeg(s.camera_jpeg)
        if s.camera_error and self.cam_on.isChecked():
            self.cam_on.setChecked(False)
            self.image.text = f"Camera failed: {s.camera_error}"
            self.image.set_jpeg(None)


# ----------------------------------------------------------------------------------------------- results
class ResultsPanel(QWidget):
    """Outputs of graph runs: one tab per output node (charts, tables, images, text)."""

    def __init__(self, studio):
        super().__init__()
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.empty = dim("Results from graph runs appear here: connect something to a Plot, Table, Image or Print node.")
        self.empty.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.empty)
        lay.addWidget(self.tabs, 1)
        self.tabs.hide()
        self._keys: dict[str, QWidget] = {}

    def _close(self, i: int) -> None:
        w = self.tabs.widget(i)
        self._keys = {k: v for k, v in self._keys.items() if v is not w}
        self.tabs.removeTab(i)
        if self.tabs.count() == 0:
            self.tabs.hide()
            self.empty.show()

    def show_value(self, key: str, title: str, value) -> None:
        if isinstance(value, dict) and "y" in value:
            kind = PlotWidget
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            kind = QTableWidget
        elif isinstance(value, np.ndarray) and value.ndim in (2, 3) and value.dtype == np.uint8:
            kind = ImageView
        else:
            kind = QPlainTextEdit
        w = self._keys.get(key)
        if w is None or type(w) is not kind:
            if w is not None:
                self.tabs.removeTab(self.tabs.indexOf(w))
            w = kind()
            if kind is QPlainTextEdit:
                w.setReadOnly(True)
                w.setFont(theme.mono())
            self._keys[key] = w
            self.tabs.addTab(w, title)
        i = self.tabs.indexOf(w)
        self.tabs.setTabText(i, title)
        if kind is PlotWidget:
            w.set_series(value)
        elif kind is QTableWidget:
            fill_table(w, value)
        elif kind is ImageView:
            w.set_array(value)
        else:
            if isinstance(value, np.ndarray):
                value = f"array {value.shape} {value.dtype}\n{np.array2string(value, precision=3, threshold=200)}"
            w.setPlainText(str(value))
        self.tabs.setCurrentIndex(i)
        self.empty.hide()
        self.tabs.show()


# --------------------------------------------------------------------------------------------------- log
class LogPanel(QPlainTextEdit):
    def __init__(self, _studio=None):
        super().__init__()
        self.setReadOnly(True)
        self.setFont(theme.mono())
        self.setMaximumBlockCount(3000)

    def add(self, text: str) -> None:
        self.appendPlainText(f"{time.strftime('%H:%M:%S')}  {text}")


# ------------------------------------------------------------------------------------------------- pilot
class PilotPanel(QWidget):
    """Give the robot a mission in words; an AI pilot (Claude, or a scripted stand-in) carries it out."""

    def __init__(self, studio):
        super().__init__()
        from ..llm import options
        self.studio, self.sess = studio, studio.sess
        self._next, self._run = 0, -1
        opts = options()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 8)
        self.mission = QPlainTextEdit()
        self.mission.setPlaceholderText("Mission, for example: walk to the red cube, pick it up and bring it to the table")
        self.mission.setMaximumHeight(64)
        lay.addWidget(self.mission)
        row = QHBoxLayout()
        self.pilot = QComboBox()
        self.pilot.addItem("Claude", "claude")
        self.pilot.addItem("Scripted (no API key)", "scripted")
        self.model = QComboBox()
        self.model.addItems(opts["models"])
        self.model.setCurrentText(opts["default_model"])
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("API key (empty: use ANTHROPIC_API_KEY)" if not opts["env_credentials"]
                                    else "API key found in the environment")
        row.addWidget(self.pilot)
        row.addWidget(self.model)
        row.addWidget(self.key, 1)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.start = QPushButton("Start mission")
        self.start.setProperty("primary", True)
        self.stop = QPushButton("Stop")
        self.state = dim("idle")
        row.addWidget(self.start)
        row.addWidget(self.stop)
        row.addWidget(self.state, 1)
        lay.addLayout(row)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        lay.addWidget(self.log, 1)
        self.start.clicked.connect(self._start)
        self.stop.clicked.connect(lambda: self.sess.pilot.stop())

    def _start(self) -> None:
        settings = {"pilot": self.pilot.currentData(), "model": self.model.currentText()}
        if self.key.text().strip():
            settings["api_key"] = self.key.text().strip()
        try:
            self.sess.pilot.start(self.mission.toPlainText(), settings)
            with self.sess.lock:
                self.sess.running = True
            self.log.clear()
            self._next = 0
        except Exception as e:  # noqa: BLE001
            self.studio.log(f"Could not start the mission: {e}")
        self.key.clear()

    def refresh(self) -> None:
        if not self.isVisible():
            return
        b = self.sess.pilot.brief()
        if b["run"] != self._run:
            self._run, self._next = b["run"], 0
        self.state.setText(f"{b['state']}   turn {b['turns']}   ${b['cost_usd']:.3f}   {b['elapsed_s']:.0f} s"
                           + (f"   {b['error']}" if b.get("error") else ""))
        if b["n_events"] > self._next:
            ev = self.sess.pilot.events(self._next)
            self._next = ev["next"]
            for e in ev["events"]:
                text = (e.get("text") or "").strip()
                if text or e.get("name"):
                    self.log.appendPlainText(f"[{e['t']:6.1f}s] {e['kind']}: {text or e.get('name')}")
