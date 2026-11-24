"""Small reusable widgets: line chart, labelled slider, top-down map, image view, table helpers."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QDoubleSpinBox, QHBoxLayout, QLabel, QSizePolicy, QSlider, QTableWidget,
                               QTableWidgetItem, QWidget)

from . import theme

SERIES_COLORS = ["#4c9aff", "#3ecf8e", "#ff5c8a", "#f0b429", "#b388ff", "#26c6da", "#ff8a65"]


def heading(text: str) -> QLabel:
    lb = QLabel(text)
    lb.setProperty("heading", True)
    return lb


def dim(text: str = "") -> QLabel:
    lb = QLabel(text)
    lb.setProperty("dim", True)
    lb.setWordWrap(True)
    return lb


def _nice(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1000 or a < 0.01:
        return f"{v:.2g}"
    return f"{v:.2f}".rstrip("0").rstrip(".")


class PlotWidget(QWidget):
    """Line chart of {"title", "x": [...], "y": {label: [...]}, "xlabel", "ylabel"}."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.series: dict | None = None
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_series(self, series: dict | None) -> None:
        self.series = series
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.BG))
        s = self.series
        ys = {k: np.asarray([np.nan if v is None else v for v in vals], float)
              for k, vals in ((s or {}).get("y") or {}).items()}
        ys = {k: v for k, v in ys.items() if len(v)}
        m = QRectF(46, 22, self.width() - 56, self.height() - 44)
        p.setPen(QColor(theme.DIM))
        if s and s.get("title"):
            p.drawText(QRectF(8, 2, self.width() - 16, 18), Qt.AlignLeft | Qt.AlignVCenter, str(s["title"]))
        if not ys or m.width() < 20 or m.height() < 20:
            p.drawText(self.rect(), Qt.AlignCenter, "no data yet")
            return
        n = max(len(v) for v in ys.values())
        x = np.asarray(s.get("x") if s.get("x") is not None and len(s.get("x")) == n else np.arange(n), float)
        allv = np.concatenate([v[np.isfinite(v)] for v in ys.values()]) if ys else np.zeros(0)
        if not len(allv):
            p.drawText(self.rect(), Qt.AlignCenter, "no data yet")
            return
        lo, hi = float(allv.min()), float(allv.max())
        if hi - lo < 1e-12:
            lo, hi = lo - 0.5, hi + 0.5
        pad = 0.06 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        x0, x1 = float(x.min()), float(x.max())
        if x1 - x0 < 1e-12:
            x1 = x0 + 1
        p.setPen(QPen(QColor(theme.LINE), 1))
        for i in range(5):
            yy = m.top() + m.height() * i / 4
            p.drawLine(QPointF(m.left(), yy), QPointF(m.right(), yy))
            p.setPen(QColor(theme.DIM))
            p.drawText(QRectF(0, yy - 8, m.left() - 5, 16), Qt.AlignRight | Qt.AlignVCenter, _nice(hi - (hi - lo) * i / 4))
            p.setPen(QPen(QColor(theme.LINE), 1))
        p.setPen(QColor(theme.DIM))
        p.drawText(QRectF(m.left(), m.bottom() + 2, 80, 16), Qt.AlignLeft, _nice(x0))
        p.drawText(QRectF(m.right() - 80, m.bottom() + 2, 80, 16), Qt.AlignRight, _nice(x1))
        if s.get("xlabel"):
            p.drawText(QRectF(m.left(), m.bottom() + 2, m.width(), 16), Qt.AlignHCenter, str(s["xlabel"]))
        lx = m.right()
        for i, (label, v) in enumerate(ys.items()):
            col = QColor(SERIES_COLORS[i % len(SERIES_COLORS)])
            xs = x[:len(v)]
            ok = np.isfinite(v)
            px = m.left() + (xs - x0) / (x1 - x0) * m.width()
            py = m.bottom() - (v - lo) / (hi - lo) * m.height()
            p.setPen(QPen(col, 1.6))
            seg: list[QPointF] = []
            for a, b, good in zip(px, py, ok):
                if good:
                    seg.append(QPointF(float(a), float(b)))
                elif seg:
                    p.drawPolyline(QPolygonF(seg))
                    seg = []
            if len(seg) > 1:
                p.drawPolyline(QPolygonF(seg))
            elif len(seg) == 1:
                p.drawEllipse(seg[0], 2, 2)
            tw = p.fontMetrics().horizontalAdvance(label) + 18
            lx -= tw
            p.fillRect(QRectF(lx, 8, 8, 8), col)
            p.setPen(QColor(theme.TEXT))
            p.drawText(QRectF(lx + 12, 2, tw, 18), Qt.AlignLeft | Qt.AlignVCenter, label)


class ParamSlider(QWidget):
    """label | slider | spin box, for one float."""
    changed = Signal(float)      # while dragging
    committed = Signal(float)    # on release / edit finished

    def __init__(self, label: str, lo: float, hi: float, value: float, decimals: int = 3, parent=None):
        super().__init__(parent)
        self.lo, self.hi = float(lo), float(hi)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.label = QLabel(label)
        self.label.setMinimumWidth(96)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(self.lo, self.hi)
        self.spin.setSingleStep((self.hi - self.lo) / 100 if self.hi > self.lo else 0.01)
        self.spin.setFixedWidth(76)
        self.spin.setKeyboardTracking(False)
        lay.addWidget(self.label)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self.set_value(value)
        self.slider.valueChanged.connect(self._from_slider)
        self.slider.sliderReleased.connect(lambda: self.committed.emit(self.value()))
        self.spin.valueChanged.connect(self._from_spin)

    def value(self) -> float:
        return float(self.spin.value())

    def set_value(self, v: float) -> None:
        self.spin.blockSignals(True)
        self.slider.blockSignals(True)
        self.spin.setValue(float(v))
        span = self.hi - self.lo
        self.slider.setValue(int(round(1000 * (float(v) - self.lo) / span)) if span > 0 else 0)
        self.spin.blockSignals(False)
        self.slider.blockSignals(False)

    def _from_slider(self, i: int) -> None:
        v = self.lo + (self.hi - self.lo) * i / 1000
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self.changed.emit(v)
        if not self.slider.isSliderDown():
            self.committed.emit(v)

    def _from_spin(self, v: float) -> None:
        self.set_value(v)
        self.changed.emit(v)
        self.committed.emit(v)


class ImageView(QWidget):
    """Shows a QImage / numpy RGB image scaled to fit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pix: QPixmap | None = None
        self.text = "no image"
        self.setMinimumSize(120, 90)

    def set_jpeg(self, data: bytes | None) -> None:
        if data:
            img = QImage.fromData(data, "JPG")
            self.pix = QPixmap.fromImage(img) if not img.isNull() else None
        else:
            self.pix = None
        self.update()

    def set_array(self, a: np.ndarray | None) -> None:
        if a is None:
            self.pix = None
        else:
            a = np.ascontiguousarray(a if a.ndim == 3 else np.repeat(a[..., None], 3, axis=2)).astype(np.uint8)
            h, w = a.shape[:2]
            self.pix = QPixmap.fromImage(QImage(a.data, w, h, 3 * w, QImage.Format_RGB888).copy())
        self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.BG))
        if self.pix is None:
            p.setPen(QColor(theme.DIM))
            p.drawText(self.rect(), Qt.AlignCenter, self.text)
            return
        pm = self.pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        p.drawPixmap((self.width() - pm.width()) // 2, (self.height() - pm.height()) // 2, pm)


class MapView(QWidget):
    """Top-down map: occupancy grid, lidar returns, planned path, goals and the robot. Click to set a goal."""
    goalClicked = Signal(float, float, bool)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.sess = session
        self.source = "truth"       # truth | lidar | slam
        self.span = 12.0            # metres shown across the smaller side
        self.follow = True
        self._img: QImage | None = None
        self._img_key = None
        self.setMinimumSize(160, 160)

    def wheelEvent(self, e) -> None:
        self.span = float(np.clip(self.span * (0.88 if e.angleDelta().y() > 0 else 1.14), 2.0, 30.0))
        self.update()

    def _centre(self):
        s = self.sess
        if self.follow and s.task is not None:
            return float(s.data.qpos[0]), float(s.data.qpos[1])
        return 0.0, 0.0

    def _to_px(self, x, y, c, scale):
        # world +x is up on the map, +y is left (robot's forward is up when it faces +x)
        return self.width() / 2 - (y - c[1]) * scale, self.height() / 2 - (x - c[0]) * scale

    def mousePressEvent(self, e) -> None:
        if e.button() != Qt.LeftButton:
            return
        with self.sess.lock:
            c = self._centre()
        scale = min(self.width(), self.height()) / self.span
        x = c[0] + (self.height() / 2 - e.position().y()) / scale
        y = c[1] + (self.width() / 2 - e.position().x()) / scale
        self.goalClicked.emit(x, y, bool(e.modifiers() & Qt.ShiftModifier))

    def paintEvent(self, _e) -> None:
        s = self.sess
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#14161a"))
        with s.lock:
            c = self._centre()
            if self.source == "slam":
                g = s.slam_grid
                occ = (g.lidar_occ() > 0.5) if hasattr(g, "lidar_occ") else None
            else:
                g = s.grid
                occ = g.truth > 0 if self.source == "truth" else g.lidar_occ() > 0.5
            key = (self.source, s.map_seq, s.scene_version)
            if occ is not None and key != self._img_key:
                # grid[row = y, col = x]; image must have x up and y left -> transpose and flip both axes
                a = np.where(occ, 150, 34).astype(np.uint8).T[::-1, ::-1]
                a = np.ascontiguousarray(a)
                self._img = QImage(a.data, a.shape[1], a.shape[0], a.shape[1], QImage.Format_Grayscale8).copy()
                self._img_key = key
            size, origin = g.size, g.origin.copy()
            pts = None if s.lidar_points is None or not s.lidar_cfg.enabled else s.lidar_points[::2, :2].copy()
            path = s.nav.path.copy()
            goals = [gg.copy() for gg, _ in s.nav.goals]
            pose = (float(s.data.qpos[0]), float(s.data.qpos[1])) if s.task is not None else None
            yaw = 0.0
            if s.task is not None:
                q = s.data.qpos[3:7]
                yaw = float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
        scale = min(self.width(), self.height()) / self.span
        if self._img is not None:
            # image covers x in [ox, ox+size] (top = ox+size), y in [oy, oy+size] (left = oy+size)
            left, top = self._to_px(origin[0] + size, origin[1] + size, c, scale)
            p.setRenderHint(QPainter.SmoothPixmapTransform, False)
            p.drawImage(QRectF(left, top, size * scale, size * scale), self._img)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(255, 255, 255, 22), 1))
        step = 1.0
        for k in range(-int(self.span), int(self.span) + 1):
            gx = round(c[0]) + k * step
            gy = round(c[1]) + k * step
            _, yy = self._to_px(gx, 0, c, scale)
            xx, _ = self._to_px(0, gy, c, scale)
            p.drawLine(QPointF(0, yy), QPointF(self.width(), yy))
            p.drawLine(QPointF(xx, 0), QPointF(xx, self.height()))
        if pts is not None and len(pts):
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(62, 207, 142, 200))
            for x, y in pts[:1500]:
                px, py = self._to_px(float(x), float(y), c, scale)
                p.drawRect(QRectF(px - 1, py - 1, 2, 2))
        if len(path) > 1:
            p.setPen(QPen(QColor(theme.ACCENT), 2))
            p.setBrush(Qt.NoBrush)
            p.drawPolyline(QPolygonF([QPointF(*self._to_px(float(x), float(y), c, scale)) for x, y in path]))
        p.setPen(QPen(QColor(theme.WARN), 2))
        p.setBrush(Qt.NoBrush)
        for g2 in goals:
            p.drawEllipse(QPointF(*self._to_px(float(g2[0]), float(g2[1]), c, scale)), 6, 6)
        if pose is not None:
            px, py = self._to_px(pose[0], pose[1], c, scale)
            p.setPen(QPen(QColor("white"), 1.5))
            p.setBrush(QColor(theme.ACCENT))
            r = max(5.0, 0.2 * scale)
            p.drawEllipse(QPointF(px, py), r, r)
            hx, hy = self._to_px(pose[0] + 0.45 * np.cos(yaw), pose[1] + 0.45 * np.sin(yaw), c, scale)
            p.drawLine(QPointF(px, py), QPointF(hx, hy))
        p.setPen(QColor(theme.DIM))
        p.drawText(QRectF(6, self.height() - 20, self.width() - 12, 16), Qt.AlignLeft,
                   f"{self.source} map   {self.span:.0f} m across   click to walk there")


def fill_table(table: QTableWidget, rows: list[dict], columns: list[str] | None = None) -> None:
    cols = columns or (list(rows[0].keys()) if rows else [])
    table.setColumnCount(len(cols))
    table.setHorizontalHeaderLabels([str(c) for c in cols])
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, k in enumerate(cols):
            v = row.get(k, "")
            if isinstance(v, float):
                v = _nice(v) if abs(v) < 1e6 else f"{v:.3g}"
            it = QTableWidgetItem("" if v is None else str(v))
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            table.setItem(r, c, it)
    table.resizeColumnsToContents()
