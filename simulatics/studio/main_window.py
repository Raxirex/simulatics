"""The Studio window: dockable panels around a 3D viewport, a tool bar, menus and saved workspaces."""
from __future__ import annotations

import threading

from PySide6.QtCore import QByteArray, QObject, QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (QApplication, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog, QHBoxLayout,
                               QInputDialog, QLabel, QMainWindow, QMessageBox, QPushButton, QStackedWidget, QToolBar,
                               QWidget)

from .. import __version__
from ..server.session import SimSession
from . import theme
from .graph_editor import GraphPanel
from .icons import icon
from .panels_run import ConsolePanel, LogPanel, PilotPanel, ResultsPanel, TimelinePanel, TrainPanel
from .panels_scene import AssetsPanel, InspectorPanel, MotorsPanel, OutlinerPanel, RobotPanel
from .viewport import FLAGS, TOOLS, VIEWS, Viewport
from .panels_run import SensorsPanel

WORKSPACES = ["Layout", "Graph", "Train", "Sensors", "Animate"]
L, R, T, B = Qt.LeftDockWidgetArea, Qt.RightDockWidgetArea, Qt.TopDockWidgetArea, Qt.BottomDockWidgetArea


class _Async(QObject):
    done = Signal(object, object, object)    # callback, result, error text


class StudioWindow(QMainWindow):
    def __init__(self, session: SimSession | None = None, restore: bool = True):
        super().__init__()
        self.sess = session or SimSession()
        self._owns_session = session is None
        self._async = _Async()
        self._async.done.connect(lambda cb, result, error: cb(result, error) if cb else None)
        self._version = self.sess.scene_version
        self._tick_n = 0
        self._last_error = ""
        self.workspace = "Layout"
        self.setWindowTitle(f"Simulatics Studio {__version__}")
        self.resize(1680, 980)
        self.setDockNestingEnabled(True)
        self.setDockOptions(QMainWindow.AnimatedDocks | QMainWindow.AllowNestedDocks | QMainWindow.AllowTabbedDocks
                            | QMainWindow.GroupedDragging)
        self.setTabPosition(Qt.AllDockWidgetAreas, self.tabPosition(L).North)
        for corner, area in ((Qt.BottomLeftCorner, L), (Qt.BottomRightCorner, R)):
            self.setCorner(corner, area)

        # panels (the log first: everything else may write to it while being built)
        self.log_panel = LogPanel()
        self.viewport = Viewport(self.sess)
        self.results = ResultsPanel(self)
        self.console = ConsolePanel(self)
        self.outliner = OutlinerPanel(self)
        self.inspector = InspectorPanel(self)
        self.robot_panel = RobotPanel(self)
        self.motors = MotorsPanel(self)
        self.assets = AssetsPanel(self)
        self.timeline = TimelinePanel(self)
        self.train = TrainPanel(self)
        self.sensors = SensorsPanel(self)
        self.pilot = PilotPanel(self)
        self.graph_panel = GraphPanel(self)
        self.console.ns["graph"] = self.graph_panel.graph

        self.docks: dict[str, QDockWidget] = {}
        for key, title, widget in (
                ("viewport", "Viewport", self.viewport), ("outliner", "Outliner", self.outliner),
                ("inspector", "Inspector", self.inspector), ("robot", "Robot", self.robot_panel),
                ("motors", "Motors", self.motors), ("assets", "Assets", self.assets),
                ("timeline", "Timeline", self.timeline), ("graph", "Node Graph", self.graph_panel),
                ("results", "Results", self.results), ("train", "Training", self.train),
                ("sensors", "Sensors", self.sensors), ("pilot", "AI Pilot", self.pilot),
                ("console", "Python", self.console), ("log", "Log", self.log_panel)):
            d = QDockWidget(title, self)
            d.setObjectName(f"dock_{key}")
            d.setWidget(widget)
            d.setAllowedAreas(Qt.AllDockWidgetAreas)
            self.docks[key] = d

        self._build_actions()
        self._build_menus()
        self._build_toolbars()
        self._build_status()

        self.viewport.bodyPicked.connect(lambda b: self.select("body", b) if b > 0 else self.select("scene", 0))
        self.viewport.message.connect(self.say)
        self.viewport.sceneChanged.connect(self.scene_changed)
        self.viewport.toolChanged.connect(self._tool_changed)
        self.outliner.selected.connect(lambda kind, ident: self.select(kind, ident, from_outliner=True))

        self.apply_workspace("Layout")
        if restore:
            self._restore()
        self.viewport.frame_robot()
        self.viewport.set_camera("Follow")     # keep the robot in view while it walks; Free unlocks panning
        self._fill_cameras()
        self.viewport.set_tool("select")
        self.log(f"Simulatics Studio {__version__} ready. Press Space to start the simulation.")
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    # ---- actions, menus, tool bars ------------------------------------------------------------------
    def _act(self, text: str, fn, shortcut: str | None = None, ic: str | None = None, checkable=False, tip="") -> QAction:
        a = QAction(icon(ic) if ic else self.windowIcon(), text, self) if ic else QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
            a.setShortcutContext(Qt.WindowShortcut)
        a.setCheckable(checkable)
        if tip or shortcut:
            a.setToolTip(f"{tip or text}" + (f"  ({shortcut})" if shortcut else ""))
        a.triggered.connect(lambda checked=False: fn(checked) if checkable else fn())
        self.addAction(a)
        return a

    def _build_actions(self) -> None:
        self.a_play = self._act("Play / Pause", self.toggle_play, "Space", "play", tip="Start or pause the simulation")
        self.a_step = self._act("Step", self.sess.step_once, ".", "step", tip="Advance one control step")
        self.a_reset = self._act("Reset", self.reset_sim, "R", "reset", tip="Put the robot back in its start pose")
        self.a_frame = self._act("Frame", self._frame, "F", "frame",
                                 tip="Centre the view on the robot or the selection (fits the graph when the graph has focus)")
        self.a_run = self._act("Run Graph", lambda: (self.show_dock("graph"), self.graph_panel.run()), "Ctrl+Return",
                               "run", tip="Execute the node graph")
        self.tool_group = QActionGroup(self)
        self.tool_actions: dict[str, QAction] = {}
        for key, (label, shortcut, tip) in TOOLS.items():
            a = self._act(label, lambda _c=False, key=key: self.viewport.set_tool(key), shortcut, key, checkable=True, tip=f"{label}: {tip}")
            self.tool_group.addAction(a)
            self.tool_actions[key] = a

    def _build_menus(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&File")
        m.addAction(self._act("New Graph", self.graph_panel.new, "Ctrl+N"))
        m.addAction(self._act("Open Graph...", self.graph_panel.open, "Ctrl+O"))
        m.addAction(self._act("Save Graph...", self.graph_panel.save, "Ctrl+S"))
        m.addSeparator()
        m.addAction(self._act("Open Robot Model...", self.robot_panel._open_file))
        m.addAction(self._act("Export Robot as MJCF...", self.robot_panel._export))
        m.addAction(self._act("Save Viewport Image...", self._save_image))
        m.addSeparator()
        m.addAction(self._act("Quit", self.close, "Ctrl+Q"))

        m = mb.addMenu("&Simulation")
        for a in (self.a_play, self.a_step, self.a_reset):
            m.addAction(a)
        m.addSeparator()
        sub = m.addMenu("Speed")
        for v in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            sub.addAction(self._act(f"{v:g}x", lambda v=v: self._set_speed(v)))
        m.addSeparator()
        m.addAction(self._act("Clear Goals", lambda: self.sess.nav.clear()))
        m.addAction(self._act("Push Robot Left", lambda: self._push("left")))
        m.addAction(self._act("Push Robot Forward", lambda: self._push("forward")))

        m = mb.addMenu("&Tools")
        for a in self.tool_actions.values():
            m.addAction(a)

        m = mb.addMenu("&View")
        m.addAction(self.a_frame)
        sub = m.addMenu("Look From")
        for i, name in enumerate(VIEWS):
            sub.addAction(self._act(name, lambda name=name: self.viewport.set_view(name), f"Ctrl+{i + 1}"))
        sub = m.addMenu("Show in Viewport")
        self.show_actions: dict[str, QAction] = {}
        for name in list(self.viewport.show) + list(FLAGS):
            a = self._act(name, lambda on, name=name: self.viewport.set_flag(name, on), checkable=True)
            a.setChecked(self.viewport.flag(name))
            sub.addAction(a)
            self.show_actions[name] = a
        sub.aboutToShow.connect(lambda: [a.setChecked(self.viewport.flag(n)) for n, a in self.show_actions.items()])
        m.addSeparator()
        m.addAction(self._act("Full Screen", lambda: self.showNormal() if self.isFullScreen() else self.showFullScreen(), "F11"))

        m = mb.addMenu("&Window")
        sub = m.addMenu("Workspace")
        for i, name in enumerate(WORKSPACES):
            sub.addAction(self._act(name, lambda name=name: self.apply_workspace(name), f"Alt+{i + 1}"))
        sub.addSeparator()
        sub.addAction(self._act("Save Current Layout As...", self._save_workspace))
        self.user_ws_menu = sub.addMenu("My Layouts")
        self._fill_user_workspaces()
        m.addSeparator()
        for key, d in self.docks.items():
            a = d.toggleViewAction()
            a.setText(d.windowTitle())
            m.addAction(a)

        m = mb.addMenu("&Graph")
        m.addAction(self.a_run)
        m.addAction(self._act("Stop Graph", self.graph_panel.stop))
        m.addAction(self._act("Clear Cached Results", self.graph_panel.clear_cache))
        sub = m.addMenu("Examples")
        from .. import graph as G
        for name in G.EXAMPLES:
            sub.addAction(self._act(name, lambda name=name: (self.apply_workspace("Graph"), self.graph_panel.load_example(name))))

        m = mb.addMenu("&Help")
        m.addAction(self._act("Shortcuts", self._shortcuts))
        m.addAction(self._act("About", self._about))

    def _build_toolbars(self) -> None:
        tools = QToolBar("Tools", self)
        tools.setObjectName("tb_tools")
        tools.setOrientation(Qt.Vertical)
        tools.setIconSize(QSize(22, 22))
        tools.setMovable(True)
        for a in self.tool_actions.values():
            tools.addAction(a)
        tools.addSeparator()
        tools.addAction(self.a_frame)
        self.addToolBar(Qt.LeftToolBarArea, tools)

        top = QToolBar("Main", self)
        top.setObjectName("tb_main")
        top.setIconSize(QSize(20, 20))
        for a in (self.a_play, self.a_step, self.a_reset):
            top.addAction(a)
        top.addSeparator()
        top.addWidget(QLabel(" Camera "))
        self.camera_box = QComboBox()
        self.camera_box.setMinimumWidth(120)
        self.camera_box.activated.connect(lambda _i: self.viewport.set_camera(self.camera_box.currentText()))
        top.addWidget(self.camera_box)
        top.addWidget(QLabel("  View "))
        self.view_box = QComboBox()
        self.view_box.addItems(list(VIEWS))
        self.view_box.activated.connect(lambda _i: self.viewport.set_view(self.view_box.currentText()))
        top.addWidget(self.view_box)
        top.addSeparator()
        top.addAction(self.a_run)
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().horizontalPolicy().Expanding, spacer.sizePolicy().verticalPolicy())
        top.addWidget(spacer)
        top.addWidget(QLabel("Workspace "))
        self.ws_box = QComboBox()
        self.ws_box.addItems(WORKSPACES)
        self.ws_box.activated.connect(lambda _i: self.apply_workspace(self.ws_box.currentText()))
        top.addWidget(self.ws_box)
        self.addToolBar(Qt.TopToolBarArea, top)
        self._fill_cameras()

        # options of the active tool (an options bar)
        from ..models.accessories import TYPES
        opts = QToolBar("Tool Options", self)
        opts.setObjectName("tb_options")
        self.opt_label = QLabel("")
        self.opt_label.setStyleSheet("font-weight: 600;")
        opts.addWidget(self.opt_label)
        self.opt_stack = QStackedWidget()
        self.opt_hint = QLabel("")
        self.opt_hint.setProperty("dim", True)
        self.opt_stack.addWidget(self.opt_hint)

        def row(*widgets):
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            for x in widgets:
                h.addWidget(x)
            h.addStretch(1)
            self.opt_stack.addWidget(w)

        self.push_force = QDoubleSpinBox()
        self.push_force.setRange(5, 1000)
        self.push_force.setSingleStep(25)
        self.push_force.setSuffix(" N")
        self.push_force.setValue(self.viewport.push_force)
        self.push_force.valueChanged.connect(lambda v: setattr(self.viewport, "push_force", float(v)))
        row(QLabel("Force"), self.push_force, QLabel("  applied for 0.15 s, away from the camera"))
        self.place_type = QComboBox()
        for key, t in TYPES.items():
            self.place_type.addItem(t["label"], key)
        self.place_type.activated.connect(lambda _i: setattr(self.viewport, "place_type", self.place_type.currentData()))
        self.place_snap = QComboBox()
        for label, v in (("Snap 1 cm", 0.01), ("Snap 5 mm", 0.005), ("Snap 5 cm", 0.05), ("No snap", 0.0)):
            self.place_snap.addItem(label, v)
        self.place_snap.activated.connect(lambda _i: setattr(self.viewport, "place_snap", self.place_snap.currentData()))
        row(QLabel("Accessory"), self.place_type, self.place_snap, QLabel("  click the robot to mount it"))
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda: setattr(self.viewport, "measure_pts", []))
        row(QLabel("Click two points."), clear)
        opts.addWidget(self.opt_stack)
        self.addToolBarBreak(Qt.TopToolBarArea)
        self.addToolBar(Qt.TopToolBarArea, opts)

    def _build_status(self) -> None:
        sb = self.statusBar()
        self.st_tool, self.st_sim, self.st_ctrl, self.st_job = QLabel(""), QLabel(""), QLabel(""), QLabel("")
        for w in (self.st_job, self.st_ctrl, self.st_sim, self.st_tool):
            w.setContentsMargins(8, 0, 8, 0)
            sb.addPermanentWidget(w)

    # ---- services the panels use --------------------------------------------------------------------
    def log(self, text: str) -> None:
        self.log_panel.add(text)
        self.say(text)

    def say(self, text: str) -> None:
        self.statusBar().showMessage(text.split("\n")[0], 8000)

    def run_async(self, fn, callback=None, label: str = "") -> None:
        """Run fn() on a worker thread; callback(result, error_text) is called on the interface thread."""
        def work():
            try:
                result, error = fn(), None
            except Exception as e:  # noqa: BLE001
                result, error = None, f"{type(e).__name__}: {e}"
            self._async.done.emit(callback, result, error)
        threading.Thread(target=work, name=label or "studio-job", daemon=True).start()

    def show_dock(self, key: str) -> None:
        d = self.docks[key]
        d.show()
        d.raise_()

    def select(self, kind: str, ident: int, from_outliner: bool = False) -> None:
        body = -1
        with self.sess.lock:
            m = self.sess.model
            if kind == "body":
                body = ident
            elif kind == "joint" and ident < m.njnt:
                body = int(m.jnt_bodyid[ident])
            elif kind == "actuator" and ident < m.nu and m.actuator_trntype[ident] == 0:
                body = int(m.jnt_bodyid[int(m.actuator_trnid[ident, 0])])
            elif kind == "geom" and ident < m.ngeom:
                body = int(m.geom_bodyid[ident])
        self.viewport.select_body(body)
        if not from_outliner:
            self.outliner.show_body(body)
        self.inspector.show(kind, ident)

    def arm_place(self, acc_type: str) -> None:
        self.viewport.place_type = acc_type
        self.place_type.setCurrentIndex(max(0, self.place_type.findData(acc_type)))
        self.sess.workshop(True)
        self.viewport.set_tool("place")
        self.say(f"Click the robot to mount the {acc_type}. The simulation is paused while you place things.")

    def scene_changed(self, light: bool = False) -> None:
        """Something rebuilt or reconfigured the model: bring every panel back in line."""
        self.robot_panel.sync()
        if light and self._version == self.sess.scene_version:
            return
        self._version = self.sess.scene_version
        robot = self.sess.custom_path or "builtin"
        if robot != getattr(self, "_robot_shown", "builtin"):
            self.viewport.frame_robot()         # a different robot: bring it into view at its own scale
        self._robot_shown = robot
        self.outliner.rebuild()
        self.motors.rebuild()
        self.assets.sync()
        self.sensors.sync()
        self.inspector.show("scene", 0)
        self.viewport.select_body(-1)
        self._fill_cameras()

    # ---- simulation ---------------------------------------------------------------------------------
    def toggle_play(self) -> None:
        with self.sess.lock:
            self.sess.running = not self.sess.running

    def reset_sim(self) -> None:
        self.sess.reset()
        self.timeline.rec.clear()

    def _set_speed(self, v: float) -> None:
        self.sess.speed = float(v)
        self.timeline.speed.setCurrentIndex(max(0, self.timeline.speed.findData(float(v))))

    def _push(self, direction: str) -> None:
        try:
            self.log(self.sess.push(direction, 80.0, 0.15))
        except Exception as e:  # noqa: BLE001
            self.log(str(e))

    def _frame(self) -> None:
        w = QApplication.focusWidget()
        if w is not None and (w is self.graph_panel or self.graph_panel.isAncestorOf(w)):
            self.graph_panel.view.fit()
        else:
            self.viewport.frame_robot()

    def _tool_changed(self, name: str) -> None:
        self.tool_actions[name].setChecked(True)
        self.st_tool.setText(f"Tool: {TOOLS[name][0]}")
        self.opt_label.setText(f" {TOOLS[name][0]} ")
        self.opt_stack.setCurrentIndex({"push": 1, "place": 2, "measure": 3}.get(name, 0))
        self.opt_hint.setText(TOOLS[name][2] if name not in ("push", "place", "measure") else "")

    def _fill_cameras(self) -> None:
        cur = self.viewport.camera_name
        self.camera_box.blockSignals(True)
        self.camera_box.clear()
        self.camera_box.addItems(["Free", "Follow"] + self.viewport.camera_names())
        self.camera_box.setCurrentText(cur if self.camera_box.findText(cur) >= 0 else "Free")
        self.camera_box.blockSignals(False)

    def _tick(self) -> None:
        self._tick_n += 1
        s = self.sess
        if s.scene_version != self._version:
            self.scene_changed()
        self.timeline.refresh()
        self.a_play.setIcon(icon("pause" if s.running else "play"))
        if self._tick_n % 3 == 0:
            self.inspector.refresh()
            self.sensors.refresh()
            self.train.refresh()
            self.pilot.refresh()
            self.st_sim.setText(f"{'running' if s.running else 'paused'}  {s.speed:g}x   t {float(s.data.time):.2f} s   falls {s.falls}")
            self.st_ctrl.setText(f"{s.robot_name if s.custom_path else 'built-in humanoid'}   controller: {getattr(s.controller, 'name', 'none')}")
            jobs = []
            if self.train.job_state() == "running":
                jobs.append("training")
            if self.graph_panel.running():
                jobs.append("graph running")
            self.st_job.setText("   ".join(jobs))
            self.st_job.setStyleSheet(f"color: {theme.WARN};")
            if s.error and s.error != self._last_error:
                self.log(f"Simulation stopped: {s.error}")
            self._last_error = s.error
        if self._tick_n % 30 == 0:
            self.robot_panel.ctrl_info.setText(f"Running now: {getattr(s.controller, 'name', 'none')}")

    # ---- workspaces ---------------------------------------------------------------------------------
    def apply_workspace(self, name: str) -> None:
        """Arrange the docks for a kind of work. Anything can then be dragged, tabbed, floated or closed."""
        d = self.docks
        for dock in d.values():
            dock.setFloating(False)
            self.removeDockWidget(dock)
        show: list[str]
        H, V = Qt.Horizontal, Qt.Vertical

        def first(key):
            self.addDockWidget(T, d[key])

        def split(a, b, orient):
            self.splitDockWidget(d[a], d[b], orient)

        def tab(first_key, *others):
            for o in others:
                self.tabifyDockWidget(d[first_key], d[o])

        if name == "Graph":
            first("viewport")
            split("viewport", "results", H)
            tab("results", "inspector", "sensors")
            self.addDockWidget(B, d["graph"])           # full width under both
            tab("graph", "console", "log")
            show = ["viewport", "results", "inspector", "sensors", "graph", "console", "log"]
            sizes = ([d["viewport"], d["results"]], [900, 700], H), ([d["viewport"], d["graph"]], [380, 520], V)
            front = ["results", "graph"]
        elif name == "Train":
            first("viewport")
            split("viewport", "train", H)
            split("train", "results", V)
            tab("results", "console", "log")
            split("viewport", "timeline", V)
            show = ["viewport", "train", "results", "console", "log", "timeline"]
            sizes = ([d["viewport"], d["train"]], [760, 900], H), ([d["train"], d["results"]], [520, 300], V),                     ([d["viewport"], d["timeline"]], [640, 150], V)
            front = ["results", "train"]
        elif name == "Sensors":
            first("viewport")
            split("viewport", "sensors", H)
            split("sensors", "pilot", V)
            tab("pilot", "assets", "log")
            split("viewport", "timeline", V)
            show = ["viewport", "sensors", "pilot", "assets", "log", "timeline"]
            sizes = ([d["viewport"], d["sensors"]], [900, 700], H), ([d["sensors"], d["pilot"]], [520, 320], V),                     ([d["viewport"], d["timeline"]], [640, 150], V)
            front = ["pilot"]
        elif name == "Animate":
            first("assets")
            split("assets", "viewport", H)
            split("viewport", "inspector", H)
            tab("inspector", "motors")
            split("viewport", "timeline", V)
            show = ["assets", "viewport", "inspector", "motors", "timeline"]
            sizes = ([d["assets"], d["viewport"], d["inspector"]], [300, 1000, 340], H), ([d["viewport"], d["timeline"]], [600, 260], V)
            front = ["inspector"]
        else:
            name = "Layout"
            first("outliner")
            split("outliner", "viewport", H)
            split("viewport", "inspector", H)
            split("viewport", "timeline", V)
            split("outliner", "assets", V)
            split("inspector", "sensors", V)
            tab("inspector", "robot", "motors")
            tab("timeline", "graph", "train", "results", "pilot", "console", "log")
            show = list(d)
            sizes = ([d["outliner"], d["viewport"], d["inspector"]], [270, 1000, 380], H),                     ([d["viewport"], d["timeline"]], [640, 250], V), ([d["inspector"], d["sensors"]], [470, 330], V),                     ([d["outliner"], d["assets"]], [430, 370], V)
            front = ["inspector", "timeline"]
        for key, dock in d.items():
            if key not in show:
                self.addDockWidget(B, dock)     # in the layout but hidden, so saved layouts remember it is closed
            dock.setVisible(key in show)
        for docks, px, orient in sizes:
            self.resizeDocks(docks, px, orient)
        for key in front:
            d[key].raise_()
        self.workspace = name
        if hasattr(self, "ws_box"):
            self.ws_box.setCurrentText(name)

    def _settings(self) -> QSettings:
        return QSettings("Simulatics", "Studio")

    def _save_workspace(self) -> None:
        name, ok = QInputDialog.getText(self, "Save layout", "Name for this layout:")
        if ok and name.strip():
            self._settings().setValue(f"layouts/{name.strip()}", self.saveState())
            self._fill_user_workspaces()
            self.log(f"Layout saved as '{name.strip()}'")

    def _fill_user_workspaces(self) -> None:
        self.user_ws_menu.clear()
        st = self._settings()
        st.beginGroup("layouts")
        names = st.childKeys()
        st.endGroup()
        for n in names:
            self.user_ws_menu.addAction(n, lambda n=n: self.restoreState(self._settings().value(f"layouts/{n}")))
        self.user_ws_menu.setEnabled(bool(names))

    def _restore(self) -> None:
        st = self._settings()
        geo, state = st.value("window/geometry"), st.value("window/state")
        if isinstance(geo, QByteArray):
            self.restoreGeometry(geo)
        if isinstance(state, QByteArray) and st.value("window/version") == __version__:
            self.restoreState(state)

    # ---- misc ---------------------------------------------------------------------------------------
    def _save_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save viewport image", "viewport.png", "PNG (*.png)")
        if path:
            self.viewport.grabFramebuffer().save(path)
            self.log(f"Saved {path}")

    def _shortcuts(self) -> None:
        rows = [("Space", "play / pause"), (".", "single step"), ("R", "reset the robot"), ("F", "frame the robot or selection"),
                *[(k, f"{label} tool") for label, k, _ in TOOLS.values()],
                ("Right-drag", "orbit the view"), ("Middle-drag", "pan the view"), ("Wheel", "zoom"),
                ("Alt+1 ... Alt+5", "switch workspace"), ("Ctrl+1 ... Ctrl+6", "look from a fixed direction"),
                ("Ctrl+Enter", "run the node graph"), ("Tab (on the graph)", "add a node"),
                ("Delete (on the graph)", "remove selected nodes or wires")]
        QMessageBox.information(self, "Shortcuts", "\n".join(f"{k:<22}{v}" for k, v in rows))

    def _about(self) -> None:
        QMessageBox.about(self, "About Simulatics Studio",
                          f"<b>Simulatics Studio {__version__}</b><br>Humanoid robot workbench.<br><br>"
                          "Physics and rendering: MuJoCo (Apache-2.0)<br>Interface: Qt for Python / PySide6 (LGPL-3)<br>"
                          "Simulatics itself: MIT licence<br><br>"
                          "An independent project, not affiliated with or endorsed by any university, company or "
                          "other software project. Third-party names belong to their owners.")

    def closeEvent(self, e) -> None:
        st = self._settings()
        st.setValue("window/geometry", self.saveGeometry())
        st.setValue("window/state", self.saveState())
        st.setValue("window/version", __version__)
        self._timer.stop()
        self.viewport._timer.stop()
        self.graph_panel.close()
        if self.train.trainer is not None and self.train.job_state() == "running":
            self.train.trainer.stop()
        if self._owns_session:
            self.sess.shutdown()
        super().closeEvent(e)
