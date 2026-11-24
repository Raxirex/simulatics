"""Scene panels: outliner (what is in the scene), inspector (the selection), robot, motors, assets."""
from __future__ import annotations

import mujoco
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QPushButton, QScrollArea, QTableWidget, QTableWidgetItem, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from . import theme
from .widgets import ParamSlider, dim, heading

CONTROLLERS = [("auto", "Automatic (best available)"), ("walker", "Model-based walker"),
               ("mpc", "Model-predictive control"), ("rl", "Learned policy (PPO / distilled)"),
               ("es-policy", "Evolved policy"), ("manual", "Manual joint targets")]


def _scroll(inner: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QScrollArea.NoFrame)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sa.setWidget(inner)
    return sa


def _spin(value: float, lo: float, hi: float, decimals: int = 4, step: float | None = None) -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setDecimals(decimals)
    sp.setRange(lo, hi)
    sp.setValue(float(value))
    sp.setKeyboardTracking(False)
    if step:
        sp.setSingleStep(step)
    return sp


# ---------------------------------------------------------------------------------------------- outliner
class OutlinerPanel(QWidget):
    """Tree of everything in the model: bodies with their joints, motors, shapes, cameras and sites."""
    selected = Signal(str, int)     # kind, id

    ROLE = Qt.UserRole + 1

    def __init__(self, studio):
        super().__init__()
        self.studio, self.sess = studio, studio.sess
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter by name")
        self.filter.setClearButtonEnabled(True)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Type"])
        self.tree.setAlternatingRowColors(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        lay.addWidget(self.filter)
        lay.addWidget(self.tree, 1)
        self._body_items: dict[int, QTreeWidgetItem] = {}
        self._quiet = False
        self.filter.textChanged.connect(self._apply_filter)
        self.tree.itemSelectionChanged.connect(self._on_select)
        self.rebuild()

    def _item(self, parent, name: str, kind: str, ident: int) -> QTreeWidgetItem:
        it = QTreeWidgetItem(parent, [name or f"({kind} {ident})", kind])
        it.setData(0, self.ROLE, (kind, ident))
        it.setForeground(1, self.palette().placeholderText())
        return it

    def rebuild(self) -> None:
        self._quiet = True
        self.tree.clear()
        self._body_items = {}
        with self.sess.lock:
            m = self.sess.model
            names = [m.body(b).name for b in range(m.nbody)]
            parents = m.body_parentid.copy()
            joints = [(j, m.joint(j).name, int(m.jnt_bodyid[j]), int(m.jnt_type[j])) for j in range(m.njnt)]
            acts = []
            for a in range(m.nu):
                body = -1
                if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                    body = int(m.jnt_bodyid[int(m.actuator_trnid[a, 0])])
                acts.append((a, m.actuator(a).name, body))
            geoms = [(g, m.geom(g).name, int(m.geom_bodyid[g]), int(m.geom_type[g])) for g in range(m.ngeom)]
            cams = [(c, m.camera(c).name, int(m.cam_bodyid[c])) for c in range(m.ncam)]
            sites = [(s, m.site(s).name, int(m.site_bodyid[s])) for s in range(m.nsite)]
        world = self._item(self.tree, "World", "world", 0)
        self._body_items[0] = world
        for b in range(1, len(names)):
            self._body_items[b] = self._item(self._body_items.get(int(parents[b]), world), names[b], "body", b)
        jt = {0: "free joint", 1: "ball joint", 2: "slide joint", 3: "hinge"}
        gt = {int(getattr(mujoco.mjtGeom, n)): n[7:].lower() for n in dir(mujoco.mjtGeom) if n.startswith("mjGEOM_")}
        for j, name, body, t in joints:
            self._item(self._body_items[body], name, jt.get(t, "joint"), j).setData(0, self.ROLE, ("joint", j))
        for a, name, body in acts:
            self._item(self._body_items.get(body, world), name, "motor", a).setData(0, self.ROLE, ("actuator", a))
        for c, name, body in cams:
            self._item(self._body_items[body], name, "camera", c)
        for s, name, body in sites:
            self._item(self._body_items[body], name, "site", s)
        for g, name, body, t in geoms:
            if name:  # unnamed shapes are noise in large imported models
                self._item(self._body_items[body], name, gt.get(t, "shape"), g).setData(0, self.ROLE, ("geom", g))
        self.tree.expandToDepth(1)
        self._apply_filter(self.filter.text())
        self._quiet = False

    def _apply_filter(self, text: str) -> None:
        text = text.strip().lower()

        def visit(it: QTreeWidgetItem) -> bool:
            keep = not text or text in it.text(0).lower() or text in it.text(1).lower()
            for i in range(it.childCount()):
                keep = visit(it.child(i)) or keep
            it.setHidden(not keep)
            if text and keep:
                it.setExpanded(True)
            return keep
        for i in range(self.tree.topLevelItemCount()):
            visit(self.tree.topLevelItem(i))

    def _on_select(self) -> None:
        if self._quiet:
            return
        items = self.tree.selectedItems()
        if items:
            kind, ident = items[0].data(0, self.ROLE)
            self.selected.emit(kind, int(ident))

    def show_body(self, body: int) -> None:
        """Highlight a body picked in the viewport (without echoing the selection back)."""
        self._quiet = True
        self.tree.clearSelection()
        it = self._body_items.get(body)
        if it is not None and body > 0:
            it.setSelected(True)
            self.tree.scrollToItem(it)
        self._quiet = False


# --------------------------------------------------------------------------------------------- inspector
class InspectorPanel(QWidget):
    """Properties of the selected part. Edits are applied to the live model in place."""

    def __init__(self, studio):
        super().__init__()
        self.studio, self.sess = studio, studio.sess
        self.kind, self.ident = "scene", 0
        self._live: list = []           # (QLabel, fn(model, data) -> str)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.holder = QWidget()
        self.form = QVBoxLayout(self.holder)
        self.form.setContentsMargins(8, 6, 8, 8)
        outer.addWidget(_scroll(self.holder))
        self.show("scene", 0)

    # -- helpers
    def _clear(self) -> None:
        self._live = []
        while self.form.count():
            w = self.form.takeAt(0).widget()
            if w is not None:
                w.deleteLater()

    def _title(self, text: str, sub: str) -> None:
        t = QLabel(text)
        t.setStyleSheet("font-size: 14px; font-weight: 600;")
        self.form.addWidget(t)
        self.form.addWidget(dim(sub))

    def _group(self, title: str) -> QFormLayout:
        gb = QGroupBox(title)
        f = QFormLayout(gb)
        f.setLabelAlignment(Qt.AlignLeft)
        f.setContentsMargins(8, 8, 8, 8)
        self.form.addWidget(gb)
        return f

    def _live_row(self, form: QFormLayout, label: str, fn) -> None:
        lb = QLabel("")
        lb.setFont(theme.mono())
        lb.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow(label, lb)
        self._live.append((lb, fn))

    def _edit(self, form: QFormLayout, label: str, value: float, lo: float, hi: float, apply, decimals=4) -> None:
        sp = _spin(value, lo, hi, decimals)

        def changed(v, apply=apply):
            with self.sess.lock:
                try:
                    apply(self.sess.model, float(v))
                except Exception as e:  # noqa: BLE001
                    self.studio.log(f"Edit failed: {e}")
        sp.valueChanged.connect(changed)
        form.addRow(label, sp)

    # -- content
    def show(self, kind: str, ident: int) -> None:
        self._clear()
        self.kind, self.ident = kind, int(ident)
        s = self.sess
        with s.lock:
            m = s.model
            try:
                if kind == "body" and 0 < ident < m.nbody:
                    self._body(m, ident)
                elif kind == "joint" and ident < m.njnt:
                    self._body(m, int(m.jnt_bodyid[ident]))
                elif kind == "actuator" and ident < m.nu:
                    self._actuator_only(m, ident)
                elif kind == "geom" and ident < m.ngeom:
                    self._geom(m, ident)
                else:
                    self.kind = "scene"
                    self._scene(m)
            except Exception as e:  # noqa: BLE001
                self.form.addWidget(dim(f"Could not inspect this: {e}"))
        self.form.addStretch(1)
        self.refresh()

    def _scene(self, m) -> None:
        s = self.sess
        self._title(s.robot_name if s.custom_path else "Built-in humanoid", "Nothing selected: scene and physics")
        f = self._group("Model")
        mass = float(m.body_subtreemass[1]) if m.nbody > 1 else 0.0
        for k, v in (("Bodies", m.nbody - 1), ("Joints", m.njnt), ("Degrees of freedom", m.nv), ("Motors", m.nu),
                     ("Shapes", m.ngeom), ("Robot mass", f"{mass:.2f} kg")):
            f.addRow(k, QLabel(str(v)))
        f = self._group("Physics")
        self._edit(f, "Time step (s)", m.opt.timestep, 0.0002, 0.02, lambda mm, v: setattr(mm.opt, "timestep", v), 5)
        self._edit(f, "Gravity z (m/s²)", m.opt.gravity[2], -30, 30, lambda mm, v: mm.opt.gravity.__setitem__(2, v), 3)
        self._edit(f, "Wind x (m/s)", m.opt.wind[0], -40, 40, lambda mm, v: mm.opt.wind.__setitem__(0, v), 2)
        self._edit(f, "Air density", m.opt.density, 0, 2000, lambda mm, v: setattr(mm.opt, "density", v), 3)
        f = self._group("State")
        self._live_row(f, "Time", lambda mm, d: f"{d.time:.3f} s")
        self._live_row(f, "Joint speed (rms)", lambda mm, d: f"{float(np.sqrt(np.mean(d.qvel ** 2))):.3f}"
                       if mm.nv else "0")
        self._live_row(f, "Contacts", lambda mm, d: str(d.ncon))
        self.form.addWidget(dim("Select a part in the viewport or the outliner to edit its joints and motors."))

    def _body(self, m, b: int) -> None:
        self._title(m.body(b).name or f"body {b}", f"Body {b}   parent: {m.body(int(m.body_parentid[b])).name or 'world'}")
        f = self._group("Body")
        self._edit(f, "Mass (kg)", m.body_mass[b], 0.0001, 500, lambda mm, v, b=b: mm.body_mass.__setitem__(b, v))
        f.addRow("Mass with children", QLabel(f"{float(m.body_subtreemass[b]):.3f} kg"))
        self._live_row(f, "Position (m)", lambda mm, d, b=b: "  ".join(f"{v:+.3f}" for v in d.xpos[b]))
        self._live_row(f, "Speed (m/s)", lambda mm, d, b=b: f"{float(np.linalg.norm(d.cvel[b, 3:])):.3f}")
        for j in range(m.njnt):
            if int(m.jnt_bodyid[j]) != b or m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            dof, qa = int(m.jnt_dofadr[j]), int(m.jnt_qposadr[j])
            f = self._group(f"Joint: {m.joint(j).name}")
            if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                deg = m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
                unit, k = ("°", 180 / np.pi) if deg else (" m", 1.0)
                self._live_row(f, "Angle" if deg else "Position",
                               lambda mm, d, qa=qa, k=k, unit=unit: f"{float(d.qpos[qa]) * k:+.2f}{unit}")
                self._live_row(f, "Torque" if deg else "Force",
                               lambda mm, d, dof=dof: f"{float(d.qfrc_actuator[dof]):+.2f}")
                self._edit(f, f"Lower limit ({unit.strip()})", m.jnt_range[j, 0] * k, -1e4, 1e4,
                           lambda mm, v, j=j, k=k: mm.jnt_range.__setitem__((j, 0), v / k), 2)
                self._edit(f, f"Upper limit ({unit.strip()})", m.jnt_range[j, 1] * k, -1e4, 1e4,
                           lambda mm, v, j=j, k=k: mm.jnt_range.__setitem__((j, 1), v / k), 2)
            self._edit(f, "Damping", m.dof_damping[dof], 0, 1000, lambda mm, v, dof=dof: mm.dof_damping.__setitem__(dof, v))
            self._edit(f, "Rotor inertia", m.dof_armature[dof], 0, 10, lambda mm, v, dof=dof: mm.dof_armature.__setitem__(dof, v), 5)
            self._edit(f, "Friction loss", m.dof_frictionloss[dof], 0, 100,
                       lambda mm, v, dof=dof: mm.dof_frictionloss.__setitem__(dof, v))
            for a in range(m.nu):
                if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and int(m.actuator_trnid[a, 0]) == j:
                    self._actuator(m, a)

    def _actuator(self, m, a: int) -> None:
        info = self.sess.actuators()[a]
        f = self._group(f"Motor: {info['name']}")
        sess = self.sess
        if info["servo"]:
            lo, hi = (info["lo"], info["hi"]) if info["hi"] > info["lo"] else (-3.14, 3.14)
            sl = ParamSlider("Target", lo, hi, float(np.clip(info["ctrl"], lo, hi)))
            sl.changed.connect(lambda v, a=a: sess.set_actuator(a, "ctrl", v))
            f.addRow(sl)
        for field, label, hi in (("kp", "Stiffness kp", 1e5), ("kv", "Damping kv", 1e4), ("torque", "Max torque", 1e4),
                                 ("gear", "Gear", 1e4)):
            sp = _spin(info[field], 0, hi, 3)
            sp.valueChanged.connect(lambda v, a=a, field=field: sess.set_actuator(a, field, v))
            f.addRow(label, sp)
        self._live_row(f, "Output now", lambda mm, d, a=a: f"{float(d.actuator_force[a]):+.2f}" if a < mm.nu else "")

    def _actuator_only(self, m, a: int) -> None:
        self._title(m.actuator(a).name or f"motor {a}", f"Motor {a}")
        self._actuator(m, a)

    def _geom(self, m, g: int) -> None:
        self._title(m.geom(g).name or f"shape {g}", f"Shape {g} on {m.body(int(m.geom_bodyid[g])).name or 'world'}")
        f = self._group("Shape")
        f.addRow("Size", QLabel("  ".join(f"{v:.3f}" for v in m.geom_size[g])))
        self._edit(f, "Friction", m.geom_friction[g, 0], 0, 5, lambda mm, v, g=g: mm.geom_friction.__setitem__((g, 0), v), 3)
        self._live_row(f, "Position (m)", lambda mm, d, g=g: "  ".join(f"{v:+.3f}" for v in d.geom_xpos[g]))

    def refresh(self) -> None:
        if not self._live or not self.isVisible():
            return
        with self.sess.lock:
            m, d = self.sess.model, self.sess.data
            for lb, fn in self._live:
                try:
                    lb.setText(fn(m, d))
                except Exception:  # noqa: BLE001 - the model changed under a stale row
                    lb.setText("")


# ------------------------------------------------------------------------------------------------- robot
class RobotPanel(QWidget):
    """Which robot is loaded, its design parameters, and which controller drives it."""

    def __init__(self, studio):
        super().__init__()
        from ..server.session import param_spec
        self.studio, self.sess = studio, studio.sess
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(8, 6, 8, 8)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scroll(inner))

        lay.addWidget(heading("Robot"))
        row = QHBoxLayout()
        self.robot = QComboBox()
        self.load_btn = QPushButton("Load")
        row.addWidget(self.robot, 1)
        row.addWidget(self.load_btn)
        lay.addLayout(row)
        self.info = dim("")
        lay.addWidget(self.info)
        row = QHBoxLayout()
        self.file_btn = QPushButton("Open model file...")
        self.export_btn = QPushButton("Export MJCF...")
        row.addWidget(self.file_btn)
        row.addWidget(self.export_btn)
        lay.addLayout(row)
        self.stand = QCheckBox("Support stand (hold the base in place)")
        self.deflidar = QCheckBox("Default head lidar")
        lay.addWidget(self.stand)
        lay.addWidget(self.deflidar)

        lay.addWidget(heading("Controller"))
        self.controller = QComboBox()
        for k, label in CONTROLLERS:
            self.controller.addItem(label, k)
        lay.addWidget(self.controller)
        self.ctrl_info = dim("")
        lay.addWidget(self.ctrl_info)

        self.sliders: dict[str, ParamSlider] = {}
        self.param_boxes: list[QGroupBox] = []
        groups: dict[str, QVBoxLayout] = {}
        for spec in param_spec():
            if spec["group"] not in groups:
                gb = QGroupBox(f"{spec['group']} (built-in humanoid)")
                gb.setCheckable(True)
                gb.setChecked(spec["group"] == "Geometry")
                v = QVBoxLayout(gb)
                v.setSpacing(3)
                holder = QWidget()
                hv = QVBoxLayout(holder)
                hv.setContentsMargins(0, 0, 0, 0)
                hv.setSpacing(3)
                v.addWidget(holder)
                holder.setVisible(gb.isChecked())
                gb.toggled.connect(holder.setVisible)
                groups[spec["group"]] = hv
                self.param_boxes.append(gb)
                lay.addWidget(gb)
            sl = ParamSlider(spec["label"], spec["min"], spec["max"], spec["default"])
            sl.label.setMinimumWidth(140)
            sl.committed.connect(lambda v, n=spec["name"]: self._set_param(n, v))
            groups[spec["group"]].addWidget(sl)
            self.sliders[spec["name"]] = sl
        self.defaults_btn = QPushButton("Reset parameters to defaults")
        lay.addWidget(self.defaults_btn)
        lay.addStretch(1)

        self.load_btn.clicked.connect(self._load)
        self.file_btn.clicked.connect(self._open_file)
        self.export_btn.clicked.connect(self._export)
        self.stand.toggled.connect(lambda on: self._guard(self.sess.set_support_stand, on))
        self.deflidar.toggled.connect(lambda on: self._guard(self.sess.set_default_lidar, on))
        self.controller.activated.connect(lambda _i: self._guard(self.sess.set_controller, self.controller.currentData()))
        self.defaults_btn.clicked.connect(self._defaults)
        self.sync()

    def _guard(self, fn, *a) -> None:
        try:
            fn(*a)
        except Exception as e:  # noqa: BLE001
            self.studio.log(f"{type(e).__name__}: {e}")
        self.studio.scene_changed()

    def _set_param(self, name: str, value: float) -> None:
        if abs(float(getattr(self.sess.params, name)) - value) < 1e-9 and not self.sess.custom_path:
            return
        self._guard(self.sess.set_params, {name: value})

    def _defaults(self) -> None:
        from ..models import HumanoidParams
        self._guard(self.sess.set_params, HumanoidParams().to_dict())

    def _load(self) -> None:
        name = self.robot.currentData()
        self.load_btn.setEnabled(False)
        self.studio.log(f"Loading {self.robot.currentText()} ...")
        self.studio.run_async(lambda: self.sess.load_robot(name), self._loaded, "load robot")

    def _loaded(self, result, error) -> None:
        self.load_btn.setEnabled(True)
        self.studio.log(f"Could not load the robot: {error}" if error else f"Loaded {result}")
        self.studio.scene_changed()
        self.studio.viewport.frame_robot()

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open robot model", "", "MuJoCo / URDF (*.xml *.urdf *.mjcf)")
        if path:
            self._guard(self.sess.load_file, path)
            self.studio.viewport.frame_robot()

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export MJCF", "robot.xml", "MJCF (*.xml)")
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.sess.export_xml())
                self.studio.log(f"Exported {path}")
            except Exception as e:  # noqa: BLE001
                self.studio.log(f"Export failed: {e}")

    def sync(self) -> None:
        """Bring the widgets in line with the session (after anything rebuilt the model)."""
        from .. import robots
        s = self.sess
        self.robot.blockSignals(True)
        self.robot.clear()
        self.robot.addItem("Built-in parametric humanoid (MIT)", "builtin")
        for r in robots.status():
            note = "" if r["downloaded"] else f", downloads {r['size_mb']} MB"
            self.robot.addItem(f"{r['label']} ({r['license']}{note})", r["name"])
        i = self.robot.findData(s.robot_name if s.custom_path else "builtin")
        self.robot.setCurrentIndex(max(i, 0))
        self.robot.blockSignals(False)
        imported = bool(s.custom_path)
        for w, val, enabled in ((self.stand, s.support_stand, imported), (self.deflidar, s.default_lidar, not imported)):
            w.blockSignals(True)
            w.setChecked(bool(val))
            w.setEnabled(enabled)
            w.blockSignals(False)
        for name, sl in self.sliders.items():
            sl.set_value(float(getattr(s.params, name)))
        for gb in self.param_boxes:
            gb.setEnabled(not imported)
        self.controller.blockSignals(True)
        self.controller.setCurrentIndex(max(self.controller.findData(s.controller_kind), 0))
        self.controller.blockSignals(False)
        self.info.setText(f"Loaded from {s.custom_path}" if imported else
                          "Parametric humanoid: every dimension, mass and motor below is editable.")
        self.ctrl_info.setText(f"Running now: {getattr(s.controller, 'name', 'none')}")


# ------------------------------------------------------------------------------------------------ motors
class MotorsPanel(QWidget):
    """Every motor in one table. Cells are editable; changes apply to the running simulation."""
    COLS = [("name", "Motor"), ("ctrl", "Target"), ("kp", "kp"), ("kv", "kv"), ("torque", "Max torque"),
            ("damping", "Joint damping"), ("armature", "Rotor inertia")]

    def __init__(self, studio):
        super().__init__()
        self.studio, self.sess = studio, studio.sess
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        lay.addWidget(self.table)
        self._quiet = False
        self.table.itemChanged.connect(self._changed)
        self.table.itemSelectionChanged.connect(self._selected)
        self.rebuild()

    def rebuild(self) -> None:
        self._quiet = True
        with self.sess.lock:
            acts = self.sess.actuators()
        t = self.table
        t.setColumnCount(len(self.COLS))
        t.setHorizontalHeaderLabels([c[1] for c in self.COLS])
        t.setRowCount(len(acts))
        for r, a in enumerate(acts):
            for c, (key, _) in enumerate(self.COLS):
                it = QTableWidgetItem(a["name"] if key == "name" else f"{a[key]:.4g}")
                if key == "name":
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                else:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(r, c, it)
        t.resizeColumnsToContents()
        t.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._quiet = False

    def _changed(self, it: QTableWidgetItem) -> None:
        if self._quiet or it.column() == 0:
            return
        try:
            self.sess.set_actuator(it.row(), self.COLS[it.column()][0], float(it.text()))
        except Exception as e:  # noqa: BLE001
            self.studio.log(f"Motor edit failed: {e}")
            self.rebuild()

    def _selected(self) -> None:
        rows = {i.row() for i in self.table.selectedItems()}
        if len(rows) == 1 and not self._quiet:
            self.studio.select("actuator", rows.pop(), from_outliner=True)


# ------------------------------------------------------------------------------------------------ assets
class AssetsPanel(QWidget):
    """Things to put into the scene: world presets, accessories to mount, motions to play."""

    def __init__(self, studio):
        super().__init__()
        from ..models.accessories import TYPES
        self.studio, self.sess = studio, studio.sess
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(8, 6, 8, 8)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scroll(inner))

        lay.addWidget(heading("World"))
        from PySide6.QtWidgets import QGridLayout
        grid = QGridLayout()
        for i, (label, fn) in enumerate((("Obstacles", lambda: self.sess.random_obstacles(12)),
                                         ("Terrain course", lambda: self.sess.terrain_course("mixed")),
                                         ("Table and objects", lambda: self.sess.manipulation_scene()),
                                         ("Empty world", lambda: self.sess.clear_obstacles()))):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, fn=fn: self._world(fn))
            grid.addWidget(b, i // 2, i % 2)
        lay.addLayout(grid)

        lay.addWidget(heading("Accessories"))
        lay.addWidget(dim("Pick one, then click the robot in the viewport to mount it there."))
        row = QHBoxLayout()
        for key, t in TYPES.items():
            b = QPushButton(t["label"])
            b.setToolTip(t["hint"])
            b.clicked.connect(lambda _=False, key=key: self.studio.arm_place(key))
            row.addWidget(b)
        lay.addLayout(row)
        self.acc = QListWidget()
        self.acc.setMaximumHeight(110)
        lay.addWidget(self.acc)
        row = QHBoxLayout()
        self.acc_del = QPushButton("Remove selected")
        self.snap = QComboBox()
        for label, v in (("Snap 1 cm", 0.01), ("Snap 5 mm", 0.005), ("Snap 5 cm", 0.05), ("No snap", 0.0)):
            self.snap.addItem(label, v)
        row.addWidget(self.acc_del)
        row.addWidget(self.snap)
        lay.addLayout(row)
        self.acc_note = dim("")
        lay.addWidget(self.acc_note)

        lay.addWidget(heading("Motions"))
        self.motions = QListWidget()
        self.motions.setMaximumHeight(130)
        lay.addWidget(self.motions)
        row = QHBoxLayout()
        for label, mode in (("Preview", "preview"), ("Play with motors", "pd")):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, mode=mode: self._play(mode))
            row.addWidget(b)
        stop = QPushButton("Back to walking")
        stop.clicked.connect(lambda: self._guard(self.sess.set_controller, "auto"))
        row.addWidget(stop)
        lay.addLayout(row)
        lay.addStretch(1)

        self.acc_del.clicked.connect(self._delete_acc)
        self.snap.activated.connect(lambda _i: setattr(self.studio.viewport, "place_snap", self.snap.currentData()))
        self.sync()

    def _guard(self, fn, *a):
        try:
            out = fn(*a)
            if isinstance(out, str):
                self.studio.log(out)
        except Exception as e:  # noqa: BLE001
            self.studio.log(f"{type(e).__name__}: {e}")
        self.studio.scene_changed()

    def _world(self, fn) -> None:
        if self.sess.custom_path:
            self.studio.log("World presets are built around the built-in humanoid; load it first.")
            return
        self._guard(fn)

    def _play(self, mode: str) -> None:
        it = self.motions.currentItem()
        if it is not None:
            self._guard(self.sess.play_motion, it.data(Qt.UserRole), mode)

    def _delete_acc(self) -> None:
        it = self.acc.currentItem()
        if it is not None:
            self._guard(self.sess.delete_accessory, it.data(Qt.UserRole))

    def sync(self) -> None:
        from ..imitate.library import list_motions
        from ..jobs import RUNS
        info = self.sess.accessories_info()
        self.acc.clear()
        for a in info["items"]:
            it = QListWidgetItem(f"{a.get('label', a['type'])} on {a['body']}   ({a.get('mass', 0):.2f} kg)")
            it.setData(Qt.UserRole, a["id"])
            self.acc.addItem(it)
        note = "" if info["lidar_mounted"] else "No lidar mounted: no scans and no mapping until you place one. "
        self.acc_note.setText(note + " ".join(info["warnings"]))
        if self.motions.count() == 0:
            try:
                for m in list_motions(extra_dirs=[RUNS / "motions"]):
                    it = QListWidgetItem(f"{m['name']}   {m['duration_s']:.1f} s")
                    it.setData(Qt.UserRole, m["name"] if m.get("library") else m["path"])
                    self.motions.addItem(it)
            except Exception as e:  # noqa: BLE001
                self.studio.log(f"Could not list motions: {e}")
