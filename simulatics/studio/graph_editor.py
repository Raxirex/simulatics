"""The node-graph editor: a canvas view of simulatics.graph (which does the actual work, without Qt).

Drag from a socket to another socket to wire nodes. Right-click or press Tab on the canvas to add a node.
Run executes the graph in a background thread; node state, progress and results come back by signal.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QCursor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
                               QGraphicsItem, QGraphicsPathItem, QGraphicsProxyWidget, QGraphicsScene, QGraphicsView,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QPlainTextEdit,
                               QPushButton, QSpinBox, QSplitter, QToolButton, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from .. import graph as G
from . import theme
from .icons import icon

NODE_W, HEADER_H, ROW_H, PORT_R = 230, 26, 22, 5.5
STATE_COLORS = {"idle": theme.LINE, "queued": theme.DIM, "running": theme.WARN, "done": theme.GOOD,
                "cached": "#2f9e77", "error": theme.BAD}


def port_color(type_name: str) -> QColor:
    return QColor(theme.PORT_COLORS.get(type_name, theme.PORT_COLORS["any"]))


class PortItem(QGraphicsItem):
    def __init__(self, node_item: "NodeItem", port: G.Port, is_output: bool):
        super().__init__(node_item)
        self.node_item, self.port, self.is_output = node_item, port, is_output
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CrossCursor)
        self.hot = False
        self.setToolTip(f"{port.name}: {port.type}" + (f"\n{port.doc}" if port.doc else ""))
        self.setZValue(3)

    def boundingRect(self) -> QRectF:
        return QRectF(-PORT_R - 4, -PORT_R - 4, 2 * PORT_R + 8, 2 * PORT_R + 8)

    def paint(self, p: QPainter, _o, _w) -> None:
        p.setRenderHint(QPainter.Antialiasing)
        c = port_color(self.port.type)
        p.setPen(QPen(QColor("white") if self.hot else QColor(theme.BORDER), 1.5))
        p.setBrush(c)
        r = PORT_R + (1.5 if self.hot else 0)
        p.drawEllipse(QPointF(0, 0), r, r)

    def hoverEnterEvent(self, _e) -> None:
        self.hot = True
        self.update()

    def hoverLeaveEvent(self, _e) -> None:
        self.hot = False
        self.update()

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self.scene().begin_wire(self)
            e.accept()

    def mouseMoveEvent(self, e) -> None:
        self.scene().move_wire(e.scenePos())

    def mouseReleaseEvent(self, e) -> None:
        self.scene().end_wire(e.scenePos())


class EdgeItem(QGraphicsPathItem):
    def __init__(self, src: PortItem | None, dst: PortItem | None, type_name: str = "any"):
        super().__init__()
        self.src, self.dst, self.loose = src, dst, None
        self.color = port_color(type_name)
        self.setZValue(-1)
        self.setFlag(QGraphicsItem.ItemIsSelectable, src is not None and dst is not None)
        self.refresh()

    def refresh(self) -> None:
        a = self.src.scenePos() if self.src else self.loose
        b = self.dst.scenePos() if self.dst else self.loose
        if a is None or b is None:
            return
        dx = max(40.0, abs(b.x() - a.x()) * 0.5)
        path = QPainterPath(a)
        path.cubicTo(a.x() + dx, a.y(), b.x() - dx, b.y(), b.x(), b.y())
        self.setPath(path)

    def paint(self, p: QPainter, _o, _w) -> None:
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("white") if self.isSelected() else self.color, 3.0 if self.isSelected() else 2.2)
        if self.src is None or self.dst is None:
            pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.drawPath(self.path())

    def shape(self) -> QPainterPath:
        from PySide6.QtGui import QPainterPathStroker
        st = QPainterPathStroker()
        st.setWidth(10)
        return st.createStroke(self.path())


class NodeItem(QGraphicsItem):
    def __init__(self, inst: G.NodeInstance, panel: "GraphPanel"):
        super().__init__()
        self.inst, self.panel = inst, panel
        self.nt: G.NodeType = G.REGISTRY[inst.type]
        self.state, self.progress, self.message, self.result = "idle", 0.0, "", ""
        self.inputs: dict[str, PortItem] = {}
        self.outputs: dict[str, PortItem] = {}
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setPos(*inst.pos)
        self.setToolTip(self.nt.doc)
        rows = max(len(self.nt.inputs), len(self.nt.outputs))
        y = HEADER_H + ROW_H / 2 + 4
        for i, port in enumerate(self.nt.inputs):
            it = PortItem(self, port, False)
            it.setPos(0, y + i * ROW_H)
            self.inputs[port.name] = it
        for i, port in enumerate(self.nt.outputs):
            it = PortItem(self, port, True)
            it.setPos(NODE_W, y + i * ROW_H)
            self.outputs[port.name] = it
        self.body_top = HEADER_H + rows * ROW_H + 8
        self.proxy = None
        self.params_h = 0.0
        if self.nt.params:
            w = self._param_widget()
            self.proxy = QGraphicsProxyWidget(self)
            self.proxy.setWidget(w)
            self.proxy.setPos(8, self.body_top)
            self.params_h = w.sizeHint().height() + 6
        self.height = self.body_top + self.params_h + 40

    # -- parameters on the node body
    def _param_widget(self) -> QWidget:
        w = QWidget()
        w.setFixedWidth(NODE_W - 16)
        w.setObjectName("nodeParams")
        w.setAttribute(Qt.WA_TranslucentBackground)
        w.setStyleSheet(f"#nodeParams {{ background: transparent; }} "
                        f"QLabel {{ color: {theme.DIM}; background: transparent; font-size: 11px; }}")
        f = QFormLayout(w)
        f.setContentsMargins(0, 0, 0, 0)
        f.setSpacing(3)
        f.setLabelAlignment(Qt.AlignLeft)
        for port in self.nt.params:
            value = self.inst.params.get(port.name, port.default)
            ed = self._editor(port, value)
            ed.setToolTip(port.doc or port.name)
            label = port.name.replace("_", " ")
            if port.type == "code":
                f.addRow(ed)
            else:
                f.addRow(label, ed)
        return w

    def _set(self, name: str, value) -> None:
        self.inst.params[name] = value
        self.panel.mark_dirty()

    def _editor(self, port: G.Port, value):
        t, name = port.type, port.name
        if t == "bool":
            ed = QCheckBox()
            ed.setChecked(bool(value))
            ed.toggled.connect(lambda v: self._set(name, bool(v)))
        elif t == "int":
            ed = QSpinBox()
            ed.setRange(int(port.min if port.min is not None else -10 ** 9), int(port.max if port.max is not None else 10 ** 9))
            ed.setValue(int(value or 0))
            ed.valueChanged.connect(lambda v: self._set(name, int(v)))
        elif t == "number":
            ed = QDoubleSpinBox()
            ed.setDecimals(4)
            ed.setRange(float(port.min if port.min is not None else -1e9), float(port.max if port.max is not None else 1e9))
            ed.setSingleStep(0.1)
            ed.setValue(float(value or 0.0))
            ed.valueChanged.connect(lambda v: self._set(name, float(v)))
        elif t == "choice":
            ed = QComboBox()
            ed.addItems([str(c) for c in (port.choices or [])])
            if value is not None:
                ed.setCurrentText(str(value))
            choices = list(port.choices or [])
            ed.currentIndexChanged.connect(lambda i: self._set(name, choices[i] if 0 <= i < len(choices) else None))
        elif t == "code":
            ed = QPlainTextEdit(str(value or ""))
            ed.setFont(theme.mono())
            ed.setFixedHeight(96)
            ed.setLineWrapMode(QPlainTextEdit.NoWrap)
            ed.textChanged.connect(lambda: self._set(name, ed.toPlainText()))
        else:
            ed = QLineEdit("" if value is None else str(value))
            ed.textChanged.connect(lambda v: self._set(name, v))
        return ed

    # -- drawing
    def boundingRect(self) -> QRectF:
        return QRectF(-2, -2, NODE_W + 4, self.height + 4)

    def paint(self, p: QPainter, _o, _w) -> None:
        p.setRenderHint(QPainter.Antialiasing)
        body = QRectF(0, 0, NODE_W, self.height)
        path = QPainterPath()
        path.addRoundedRect(body, 7, 7)
        p.fillPath(path, QColor("#2a2e37"))
        head = QPainterPath()
        head.addRoundedRect(QRectF(0, 0, NODE_W, HEADER_H + 6), 7, 7)
        p.save()
        p.setClipRect(QRectF(0, 0, NODE_W, HEADER_H))
        p.fillPath(head, QColor(self.nt.color))
        p.restore()
        p.setPen(QColor("white"))
        f = QFont(p.font())
        f.setBold(True)
        p.setFont(f)
        p.drawText(QRectF(10, 0, NODE_W - 36, HEADER_H), Qt.AlignVCenter | Qt.AlignLeft, self.inst.title or self.nt.title)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(STATE_COLORS.get(self.state, theme.LINE)))
        p.drawEllipse(QPointF(NODE_W - 14, HEADER_H / 2), 5, 5)
        f.setBold(False)
        f.setPointSizeF(max(7.0, f.pointSizeF() - 0.5))
        p.setFont(f)
        y = HEADER_H + 4
        for i, port in enumerate(self.nt.inputs):
            p.setPen(QColor(theme.TEXT))
            p.drawText(QRectF(12, y + i * ROW_H, NODE_W / 2 - 14, ROW_H), Qt.AlignVCenter | Qt.AlignLeft, port.name)
        for i, port in enumerate(self.nt.outputs):
            p.setPen(QColor(theme.TEXT))
            p.drawText(QRectF(NODE_W / 2, y + i * ROW_H, NODE_W / 2 - 12, ROW_H), Qt.AlignVCenter | Qt.AlignRight, port.name)
        foot = QRectF(8, self.height - 36, NODE_W - 16, 32)
        if self.state == "running":
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(theme.BG))
            p.drawRoundedRect(QRectF(8, self.height - 12, NODE_W - 16, 5), 2, 2)
            p.setBrush(QColor(theme.WARN))
            p.drawRoundedRect(QRectF(8, self.height - 12, (NODE_W - 16) * float(np.clip(self.progress, 0.02, 1)), 5), 2, 2)
            foot = QRectF(8, self.height - 38, NODE_W - 16, 24)
        text = self.message if self.state in ("running", "error") else self.result
        p.setPen(QColor(theme.BAD if self.state == "error" else theme.DIM))
        p.drawText(foot, Qt.AlignLeft | Qt.AlignVCenter | Qt.TextWordWrap,
                   p.fontMetrics().elidedText(text.replace("\n", "  "), Qt.ElideRight, int(foot.width() * 2 - 20)))
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(theme.ACCENT) if self.isSelected() else QColor(STATE_COLORS.get(self.state, theme.LINE))
                      if self.state in ("running", "error") else QColor(theme.BORDER), 2 if self.isSelected() else 1.2))
        p.drawPath(path)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged and self.scene() is not None:
            self.inst.pos = (float(self.pos().x()), float(self.pos().y()))
            self.scene().refresh_edges(self)
        return super().itemChange(change, value)

    def set_state(self, state: str, message: str = "", progress: float | None = None, result: str | None = None) -> None:
        self.state, self.message = state, message
        if progress is not None:
            self.progress = progress
        if result is not None:
            self.result = result
        self.update()


class GraphScene(QGraphicsScene):
    def __init__(self, panel: "GraphPanel"):
        super().__init__()
        self.panel = panel
        self.setSceneRect(-20000, -20000, 40000, 40000)
        self.node_items: dict[str, NodeItem] = {}
        self.edges: list[EdgeItem] = []
        self._wire: EdgeItem | None = None
        self._wire_from: PortItem | None = None

    def rebuild(self) -> None:
        self.clear()
        self.node_items, self.edges, self._wire = {}, [], None
        g = self.panel.graph
        for inst in g.nodes.values():
            if inst.type not in G.REGISTRY:
                continue
            it = NodeItem(inst, self.panel)
            self.addItem(it)
            self.node_items[inst.id] = it
        self.rebuild_edges()

    def rebuild_edges(self) -> None:
        for e in self.edges:
            self.removeItem(e)
        self.edges = []
        for ln in self.panel.graph.links:
            a, b = self.node_items.get(ln.src), self.node_items.get(ln.dst)
            if a is None or b is None or ln.src_port not in a.outputs or ln.dst_port not in b.inputs:
                continue
            e = EdgeItem(a.outputs[ln.src_port], b.inputs[ln.dst_port], a.outputs[ln.src_port].port.type)
            e.link = ln
            self.addItem(e)
            self.edges.append(e)

    def refresh_edges(self, node_item: NodeItem) -> None:
        for e in self.edges:
            if (e.src and e.src.node_item is node_item) or (e.dst and e.dst.node_item is node_item):
                e.refresh()

    # -- wiring by drag
    def begin_wire(self, port: PortItem) -> None:
        g = self.panel.graph
        if not port.is_output:
            # dragging from a connected input picks the wire up from its source
            ln = next((l for l in g.links if l.dst == port.node_item.inst.id and l.dst_port == port.port.name), None)
            if ln is not None:
                src = self.node_items[ln.src].outputs[ln.src_port]
                g.disconnect(ln.dst, ln.dst_port)
                self.rebuild_edges()
                self.panel.mark_dirty()
                port = src
        self._wire_from = port
        self._wire = EdgeItem(port if port.is_output else None, None if port.is_output else port, port.port.type)
        self._wire.loose = port.scenePos()
        self._wire.refresh()
        self.addItem(self._wire)

    def move_wire(self, pos: QPointF) -> None:
        if self._wire is not None:
            self._wire.loose = pos
            self._wire.refresh()

    def end_wire(self, pos: QPointF) -> None:
        if self._wire is None:
            return
        self.removeItem(self._wire)
        self._wire = None
        a = self._wire_from
        b = next((it for it in self.items(QRectF(pos.x() - 9, pos.y() - 9, 18, 18)) if isinstance(it, PortItem)), None)
        if b is None:
            # dropped on a node body: use its first compatible socket
            node = next((it for it in self.items(pos) if isinstance(it, NodeItem)), None)
            if node is not None and node is not a.node_item:
                pool = node.inputs if a.is_output else node.outputs
                b = next((p for p in pool.values()
                          if G.compatible(*( (a.port.type, p.port.type) if a.is_output else (p.port.type, a.port.type)))), None)
        if b is None or b.is_output == a.is_output:
            return
        src, dst = (a, b) if a.is_output else (b, a)
        try:
            self.panel.graph.connect(src.node_item.inst.id, src.port.name, dst.node_item.inst.id, dst.port.name)
            self.panel.mark_dirty()
        except G.GraphError as e:
            self.panel.say(str(e))
        self.rebuild_edges()

    def delete_selected(self) -> None:
        g = self.panel.graph
        for it in list(self.selectedItems()):
            if isinstance(it, EdgeItem) and getattr(it, "link", None) is not None:
                g.disconnect(it.link.dst, it.link.dst_port)
        for it in list(self.selectedItems()):
            if isinstance(it, NodeItem):
                g.remove_node(it.inst.id)
        self.rebuild()
        self.panel.mark_dirty()


class SearchPopup(QFrame):
    """Type-to-find node picker (Tab or double-click on the canvas)."""
    chosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Popup)
        self.setStyleSheet(f"QFrame {{ background: {theme.RAISED}; border: 1px solid {theme.ACCENT}; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Search nodes")
        self.list = QListWidget()
        lay.addWidget(self.edit)
        lay.addWidget(self.list)
        self.resize(300, 340)
        self.edit.textChanged.connect(self._fill)
        self.edit.returnPressed.connect(self._accept)
        self.list.itemClicked.connect(lambda _it: self._accept())
        self._fill("")

    def _fill(self, text: str) -> None:
        self.list.clear()
        words = text.lower().split()
        for nt in sorted(G.REGISTRY.values(), key=lambda n: (n.category, n.title)):
            hay = f"{nt.category} {nt.title} {nt.id} {nt.doc}".lower()
            if all(w in hay for w in words):
                it = QListWidgetItem(f"{nt.category}  ›  {nt.title}")
                it.setData(Qt.UserRole, nt.id)
                it.setToolTip(nt.doc)
                self.list.addItem(it)
        self.list.setCurrentRow(0)

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key_Down, Qt.Key_Up):
            step = 1 if e.key() == Qt.Key_Down else -1
            self.list.setCurrentRow(int(np.clip(self.list.currentRow() + step, 0, self.list.count() - 1)))
        else:
            super().keyPressEvent(e)

    def _accept(self) -> None:
        it = self.list.currentItem()
        if it is not None:
            self.chosen.emit(it.data(Qt.UserRole))
        self.close()

    def popup(self, global_pos) -> None:
        self.edit.clear()
        self.move(global_pos)
        self.show()
        self.edit.setFocus()


class GraphView(QGraphicsView):
    def __init__(self, scene: GraphScene, panel: "GraphPanel"):
        super().__init__(scene)
        self.panel = panel
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)
        self.setBackgroundBrush(QBrush(QColor("#17191d")))
        self.setAcceptDrops(True)
        self._pan = None
        self.search = SearchPopup(self)
        self.search.chosen.connect(self._add_from_search)
        self._add_at = QPointF(0, 0)

    def drawBackground(self, p: QPainter, rect: QRectF) -> None:
        super().drawBackground(p, rect)
        scale = self.transform().m11()
        for step, col in ((25, "#1d2026"), (125, "#252931")):
            if step * scale < 6:
                continue
            p.setPen(QPen(QColor(col), 0))
            x = np.floor(rect.left() / step) * step
            while x < rect.right():
                p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
                x += step
            y = np.floor(rect.top() / step) * step
            while y < rect.bottom():
                p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
                y += step

    def wheelEvent(self, e) -> None:
        item = self.itemAt(e.position().toPoint())
        if isinstance(item, QGraphicsProxyWidget) or (item is not None and isinstance(item.parentItem(), QGraphicsProxyWidget)):
            if self.scene().focusItem() is not None:      # scrolling inside a focused editor on a node
                return super().wheelEvent(e)
        f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        s = self.transform().m11() * f
        if 0.15 <= s <= 2.5:
            self.scale(f, f)

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and e.modifiers() & Qt.AltModifier):
            self._pan = e.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e) -> None:
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.setTransformationAnchor(QGraphicsView.NoAnchor)
            self.translate(d.x() / self.transform().m11(), d.y() / self.transform().m11())
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        if self._pan is not None:
            self._pan = None
            self.unsetCursor()
            return
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e) -> None:
        if self.itemAt(e.position().toPoint()) is None:
            self._open_search(e.position().toPoint())
        else:
            super().mouseDoubleClickEvent(e)

    def _open_search(self, view_pos) -> None:
        self._add_at = self.mapToScene(view_pos)
        self.search.popup(self.mapToGlobal(view_pos))

    def _add_from_search(self, type_id: str) -> None:
        self.panel.add_node(type_id, (self._add_at.x(), self._add_at.y()))

    def focusInEvent(self, e) -> None:
        super().focusInEvent(e)
        if e.reason() != Qt.MouseFocusReason:
            # Qt hands keyboard focus to the first editor on a node; keys should go to the canvas until the
            # user clicks into an editor
            self.scene().setFocusItem(None)

    def event(self, e) -> bool:
        # Tab opens the node search; left alone, Qt would move keyboard focus into the first editor on a node
        if e.type() == QEvent.KeyPress and e.key() == Qt.Key_Tab and not isinstance(self.scene().focusItem(), QGraphicsProxyWidget):
            self._open_search(self.mapFromGlobal(QCursor.pos()))
            return True
        return super().event(e)

    def keyPressEvent(self, e) -> None:
        focus = self.scene().focusItem()
        typing = isinstance(focus, QGraphicsProxyWidget)
        if not typing and e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.scene().delete_selected()
        else:
            super().keyPressEvent(e)

    def contextMenuEvent(self, e) -> None:
        item = self.itemAt(e.pos())
        while item is not None and not isinstance(item, (NodeItem, EdgeItem)):
            item = item.parentItem()
        menu = QMenu(self)
        if isinstance(item, NodeItem):
            menu.addAction("Run up to this node", lambda: self.panel.run([item.inst.id]))
            menu.addAction("Duplicate", lambda: self.panel.duplicate(item.inst.id))
            menu.addAction("Delete", lambda: (item.setSelected(True), self.scene().delete_selected()))
        elif isinstance(item, EdgeItem):
            menu.addAction("Delete wire", lambda: (item.setSelected(True), self.scene().delete_selected()))
        else:
            pos = self.mapToScene(e.pos())
            cats: dict[str, QMenu] = {}
            for nt in sorted(G.REGISTRY.values(), key=lambda n: (n.category, n.title)):
                sub = cats.get(nt.category) or cats.setdefault(nt.category, menu.addMenu(nt.category))
                act = sub.addAction(nt.title, lambda nt=nt: self.panel.add_node(nt.id, (pos.x(), pos.y())))
                act.setToolTip(nt.doc)
            menu.addSeparator()
            menu.addAction("Search...   Tab", lambda: self._open_search(e.pos()))
        menu.exec(e.globalPos())

    def fit(self) -> None:
        items = [it for it in self.scene().items() if isinstance(it, NodeItem)]
        if not items:
            self.resetTransform()
            self.centerOn(0, 0)
            return
        r = items[0].sceneBoundingRect()
        for it in items[1:]:
            r = r.united(it.sceneBoundingRect())
        self.fitInView(r.adjusted(-60, -60, 60, 60), Qt.KeepAspectRatio)
        if self.transform().m11() > 1.0:
            self.resetTransform()
            self.centerOn(r.center())


class GraphPanel(QWidget):
    """Tool strip + node library + canvas."""
    event = Signal(str, object, object)      # from the worker thread
    finished = Signal(object)

    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self.graph = G.Graph()
        self.executor = G.Executor(session=studio.sess)
        self.path: Path | None = None
        self.dirty = False
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._plot_t = 0.0

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 4)
        self.run_btn = QPushButton(icon("run", "#ffffff"), " Run graph")
        self.run_btn.setProperty("primary", True)
        self.run_btn.setToolTip("Execute the graph (Ctrl+Enter). Unchanged nodes reuse their cached result.")
        self.stop_btn = QPushButton(icon("stop"), " Stop")
        self.stop_btn.setEnabled(False)
        bar.addWidget(self.run_btn)
        bar.addWidget(self.stop_btn)
        for label, tip, fn in (("Add node", "Add a node (Tab or right-click on the canvas)", self._add_menu),
                               ("Examples", "Load a ready-made graph", self._examples_menu),
                               ("New", "Start an empty graph", self.new),
                               ("Open...", "Open a graph file", self.open),
                               ("Save...", "Save this graph as JSON", self.save),
                               ("Clear cache", "Forget cached results so every node runs again", self.clear_cache),
                               ("Fit", "Fit the whole graph in view (F)", lambda: self.view.fit())):
            b = QToolButton()
            b.setText(label)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, fn=fn, b=b: fn(b) if fn in (self._add_menu, self._examples_menu) else fn())
            bar.addWidget(b)
        bar.addStretch(1)
        self.status = QLabel("")
        self.status.setProperty("dim", True)
        bar.addWidget(self.status)
        lay.addLayout(bar)

        split = QSplitter(Qt.Horizontal)
        self.library = QTreeWidget()
        self.library.setHeaderHidden(True)
        self.library.setToolTip("Double-click a node to add it to the canvas")
        for nt in sorted(G.REGISTRY.values(), key=lambda n: (n.category, n.title)):
            parents = self.library.findItems(nt.category, Qt.MatchExactly)
            parent = parents[0] if parents else QTreeWidgetItem(self.library, [nt.category])
            it = QTreeWidgetItem(parent, [nt.title])
            it.setData(0, Qt.UserRole, nt.id)
            it.setToolTip(0, nt.doc)
            it.setForeground(0, QColor(theme.TEXT))
            parent.setForeground(0, QColor(nt.color).lighter(160))
        self.library.expandAll()
        self.library.itemDoubleClicked.connect(self._library_add)
        self.scene = GraphScene(self)
        self.view = GraphView(self.scene, self)
        split.addWidget(self.library)
        split.addWidget(self.view)
        split.setSizes([170, 900])
        split.setStretchFactor(1, 1)
        lay.addWidget(split, 1)

        self.run_btn.clicked.connect(lambda: self.run())
        self.stop_btn.clicked.connect(self.stop)
        self.event.connect(self._on_event)
        self.finished.connect(self._on_finished)
        self.say("Empty graph. Load an example, or press Tab on the canvas to add a node.")

    # -- editing
    def say(self, text: str) -> None:
        self.status.setText(text)

    def mark_dirty(self) -> None:
        self.dirty = True

    def add_node(self, type_id: str, pos=None) -> None:
        if pos is None:
            c = self.view.mapToScene(self.view.viewport().rect().center())
            pos = (c.x() - NODE_W / 2 + 30 * (len(self.graph.nodes) % 5), c.y() - 60 + 30 * (len(self.graph.nodes) % 5))
        inst = self.graph.add_node(type_id, pos=(float(pos[0]), float(pos[1])))
        it = NodeItem(inst, self)
        self.scene.addItem(it)
        self.scene.node_items[inst.id] = it
        self.mark_dirty()

    def duplicate(self, node_id: str) -> None:
        src = self.graph.nodes[node_id]
        self.add_node(src.type, (src.pos[0] + 40, src.pos[1] + 40))
        new = list(self.graph.nodes.values())[-1]
        new.params.update(src.params)
        self.scene.rebuild()

    def _library_add(self, item: QTreeWidgetItem, _col: int) -> None:
        tid = item.data(0, Qt.UserRole)
        if tid:
            self.add_node(tid)

    def _add_menu(self, button) -> None:
        self.view._open_search(self.view.viewport().rect().center())

    def _examples_menu(self, button) -> None:
        menu = QMenu(self)
        for name in G.EXAMPLES:
            menu.addAction(name, lambda name=name: self.load_example(name))
        menu.exec(button.mapToGlobal(button.rect().bottomLeft()))

    def set_graph(self, graph: "G.Graph", note: str = "") -> None:
        if self.running():
            self.say("Stop the running graph first.")
            return
        self.graph = graph
        self.executor.clear_cache()
        self.scene.rebuild()
        QTimer.singleShot(60, self.view.fit)     # after the dock has its final size
        self.dirty = False
        self.studio.console.ns["graph"] = graph
        self.say(note)

    def load_example(self, name: str) -> None:
        self.path = None
        self.set_graph(G.EXAMPLES[name](), f"Example: {name}")

    def new(self) -> None:
        self.path = None
        self.set_graph(G.Graph(), "Empty graph")

    def open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open graph", "", "Simulatics graph (*.json)")
        if path:
            try:
                self.path = Path(path)
                self.set_graph(G.Graph.load(path), f"Opened {self.path.name}")
            except Exception as e:  # noqa: BLE001
                self.say(f"Could not open that file: {e}")

    def save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save graph", str(self.path or "graph.json"), "Simulatics graph (*.json)")
        if path:
            self.graph.save(path)
            self.path, self.dirty = Path(path), False
            self.say(f"Saved {self.path.name}")

    def clear_cache(self) -> None:
        self.executor.clear_cache()
        for it in self.scene.node_items.values():
            it.set_state("idle", result="")
        self.say("Cache cleared")

    # -- running
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run(self, targets: list[str] | None = None) -> None:
        if self.running():
            return
        if not self.graph.nodes:
            self.say("Nothing to run: the graph is empty.")
            return
        problems = self.graph.validate()
        order = self.graph.topo_order(targets)
        if targets is None and problems:
            self.say(problems[0] + (f"  (+{len(problems) - 1} more)" if len(problems) > 1 else ""))
            self.studio.log("Graph problems: " + "; ".join(problems))
            return
        for nid, it in self.scene.node_items.items():
            it.set_state("queued" if nid in order else "idle", progress=0.0)
        self._cancel.clear()
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.say("Running...")
        graph, ex = self.graph, self.executor

        def work():
            try:
                results = ex.run(graph, targets=targets, on_event=lambda k, n, p: self.event.emit(k, n, p), cancel=self._cancel)
            except Exception as e:  # noqa: BLE001 - the executor should not raise; report it if it does
                import traceback
                self.event.emit("error", None, {"message": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
                results = {}
                self.event.emit("finished", None, {"ok": False, "cancelled": False, "seconds": 0.0})
            self.finished.emit(results)
        self._thread = threading.Thread(target=work, name="graph", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._cancel.set()
        self.say("Stopping...")

    def _on_event(self, kind: str, node_id, payload) -> None:
        payload = payload or {}
        it = self.scene.node_items.get(node_id) if node_id else None
        if kind == "start" and it:
            it.set_state("running", "starting", 0.0)
        elif kind == "progress" and it:
            it.set_state("running", str(payload.get("message", "")), float(payload.get("frac") or 0.0))
            hist = payload.get("history")
            if hist and time.monotonic() - self._plot_t > 0.5:
                self._plot_t = time.monotonic()
                ys = {k: [h.get(k) for h in hist] for k in ("best", "mean") if any(h.get(k) is not None for h in hist)}
                self.studio.results.show_value(f"{node_id}:live", f"{it.nt.title} (live)",
                                               {"title": it.nt.title, "x": [h.get("gen", i + 1) for i, h in enumerate(hist)],
                                                "y": ys, "xlabel": "iteration"})
        elif kind == "log":
            self.studio.log(f"[{it.nt.title if it else 'graph'}] {payload.get('message', '')}")
        elif kind in ("done", "cached") and it:
            outs = payload.get("outputs") or {}
            text = "   ".join(f"{k}: {v}" for k, v in outs.items()) if isinstance(outs, dict) else str(outs)
            secs = payload.get("seconds")
            it.set_state(kind, result=text + (f"   ({secs:.1f} s)" if kind == "done" and isinstance(secs, float) and secs >= 0.1 else ""))
        elif kind == "error":
            msg = str(payload.get("message", "error"))
            if it:
                it.set_state("error", msg)
            self.studio.log(f"Graph error{' in ' + it.nt.title if it else ''}: {msg}")
            if payload.get("traceback"):
                self.studio.console.write(payload["traceback"] + "\n")
        elif kind == "finished":
            self.say("Stopped" if payload.get("cancelled") else
                     (f"Finished in {payload.get('seconds', 0):.1f} s" if payload.get("ok") else "Stopped at an error"))

    def _on_finished(self, results) -> None:
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        for nid, it in self.scene.node_items.items():
            if it.state == "queued":
                it.set_state("idle")
            outs = (results or {}).get(nid)
            if outs and it.nt.category == "Output":
                value = next(iter(outs.values()))
                self.studio.results.show_value(nid, it.inst.title or it.nt.title, value)
        self.studio.scene_changed()

    def close(self) -> None:
        self._cancel.set()
        if self._thread is not None:
            self._thread.join(3.0)
        try:
            self.executor.close()
        except Exception:  # noqa: BLE001
            pass
