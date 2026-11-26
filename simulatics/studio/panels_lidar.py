"""Lidar-only 3D view: what the lidar has seen and remembered so far, as a point cloud you can orbit.

Drawn with QPainter (no OpenGL), so it works everywhere. ``CloudView`` is the 3D widget, ``LidarCloudPanel`` the
dock content around it (record / clear / save / load / colours).
"""
from __future__ import annotations

import contextlib
import math
import os
import time

import numpy as np
import shiboken6
from PySide6.QtCore import QLineF, QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QSpinBox, QVBoxLayout, QWidget)

from ..sim.cloud import PointCloudMap
from .theme import ACCENT, BG, DIM, LINE, PANEL, TEXT
from .widgets import dim

COLOR_MODES = ("height", "distance", "age", "plain")
N_BINS = 12
SCAN_COLOR = "#ff4fd8"          # the most recent scan: magenta, absent from every ramp below
TRAJ_COLOR = "#ffffff"
FOV_DEG = 50.0                  # vertical field of view
DRAG_POINTS = 60_000            # while the mouse is dragging, draw about this many points at most

# height / distance: a perceptual blue -> cyan -> green -> yellow -> red ramp that stays bright on the dark background
_RAMP = np.array([(62, 90, 255), (40, 160, 255), (30, 215, 230), (40, 235, 160), (120, 240, 90), (200, 235, 60),
                  (250, 205, 50), (255, 150, 50), (255, 95, 60)], float)


def _ramp(n: int) -> list[QColor]:
    t = np.linspace(0, len(_RAMP) - 1, n)
    i = np.minimum(t.astype(int), len(_RAMP) - 2)
    c = _RAMP[i] + (_RAMP[i + 1] - _RAMP[i]) * (t - i)[:, None]
    return [QColor(int(r), int(g), int(b)) for r, g, b in c]


def _fade(n: int) -> list[QColor]:
    """Age: the oldest points dim, the newest bright."""
    base = QColor(ACCENT)
    out = []
    for k in range(n):
        f = 0.22 + 0.78 * k / max(n - 1, 1)
        out.append(QColor(int(base.red() * f + 40 * (1 - f)), int(base.green() * f + 44 * (1 - f)),
                          int(base.blue() * f + 52 * (1 - f))))
    out[-1] = QColor("#e8f2ff")
    return out


def _polygon(xy: np.ndarray) -> QPolygonF:
    """A QPolygonF filled straight from an (n, 2) array (no Python loop over the points)."""
    n = len(xy)
    poly = QPolygonF()
    if n:
        poly.resize(n)
        np.frombuffer(shiboken6.VoidPtr(poly.data(), n * 16, True), dtype=np.float64)[:] = \
            np.asarray(xy, np.float64).reshape(-1)
    return poly


class CloudView(QWidget):
    """Orbit view of a point cloud: left-drag orbits, right- or middle-drag pans, the wheel zooms, a double
    click recentres on the cloud. World frame is z up; a 1 m grid lies on z = 0."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.ClickFocus)
        self.color_mode = "height"
        self.point_size = 2
        # camera: looks at `target` from `dist` metres away, `az` around z (0 = from +x), `el` above the ground
        self.target = np.array([0.0, 0.0, 0.5])
        self.az = math.radians(-125.0)
        self.el = math.radians(32.0)
        self.dist = 12.0
        # data
        self._pts = np.zeros((0, 3), np.float32)
        self._stamps = None
        self._traj = None
        self._scan = None
        self._robot = None
        self._seq = None
        self._lo = self._hi = None
        self._dirty = True                    # colour bins need recomputing
        self._sorted = self._pts              # the points reordered so each colour bin is one contiguous slice
        self._starts = np.zeros(1, np.int64)  # bin k is _sorted[_starts[k]:_starts[k + 1]]
        self._colors: list[QColor] = []
        self._legend = ""
        self._version = 0                     # bumped whenever the binned points change
        self._cloud_key = None                # what the cached cloud layer shows
        self._cloud_n = 0
        self.last_bin_ms = 0.0                # how long the last colour binning took
        # interaction
        self._press = None
        self._dragging = False
        self._layers: dict = {}
        self.last_draw_ms = 0.0               # how long the last redraw took
        self.last_drawn = 0                   # how many cloud points it drew
        self._idle = QTimer(self)
        self._idle.setSingleShot(True)
        self._idle.setInterval(140)
        self._idle.timeout.connect(self._settle)

    # ------------------------------------------------------------------ data
    def set_cloud(self, points, stamps=None, trajectory=None, scan=None, robot_xy_yaw=None, seq=None) -> None:
        """Show ``points`` (n, 3). ``stamps`` (n,) colours by age, ``trajectory`` (m, 3) is the sensor path,
        ``scan`` (k, 3) the latest scan (highlighted), ``robot_xy_yaw`` (x, y, yaw) the robot arrow.

        ``seq`` is optional: pass the cloud's ``seq`` and the colouring work is skipped when it has not changed.
        The arrays are not copied and must not be modified afterwards.
        """
        pts = np.zeros((0, 3), np.float32) if points is None else np.asarray(points, np.float32).reshape(-1, 3)
        same = seq is not None and seq == self._seq and len(pts) == len(self._pts)
        self._seq = seq
        if not same:
            self._pts = pts
            st = None if stamps is None else np.asarray(stamps, np.float32).reshape(-1)
            self._stamps = st if st is not None and len(st) == len(pts) else None
            tr = None if trajectory is None else np.asarray(trajectory, np.float32).reshape(-1, 3)
            self._traj = tr if tr is not None and len(tr) else None
            if len(pts):
                self._lo, self._hi = pts.min(axis=0).astype(float), pts.max(axis=0).astype(float)
            else:
                self._lo = self._hi = None
            self._dirty = True
        sc = None if scan is None else np.asarray(scan, np.float32).reshape(-1, 3)
        self._scan = sc if sc is not None and len(sc) else None
        robot = None if robot_xy_yaw is None else tuple(float(v) for v in robot_xy_yaw)
        if self.color_mode == "distance" and self._traj is None and robot != self._robot:
            self._dirty = True
        self._robot = robot
        self.update()

    def set_color_mode(self, mode: str) -> None:
        if mode not in COLOR_MODES:
            raise ValueError(f"colour mode must be one of {COLOR_MODES}")
        if mode != self.color_mode:
            self.color_mode = mode
            self._dirty = True
            self.update()

    def set_point_size(self, px: int) -> None:
        px = int(min(max(px, 1), 4))
        if px != self.point_size:
            self.point_size = px
            self.update()

    def set_look_at(self, xyz) -> None:
        """Move the point the camera orbits (2 or 3 numbers; with 2 the height is kept)."""
        v = np.asarray(xyz, float).reshape(-1)
        if not np.isfinite(v).all():
            return
        new = self.target.copy()
        new[:len(v[:3])] = v[:3]
        if not np.allclose(new, self.target, atol=1e-4):
            self.target = new
            self.update()

    def bounds(self):
        """(min, max) of the shown cloud or None."""
        return None if self._lo is None else (self._lo.copy(), self._hi.copy())

    def fit(self) -> None:
        """Frame the whole cloud (keeps the viewing direction)."""
        self._fit(self.width(), self.height())
        self.update()

    def _fit(self, w: int, h: int) -> None:
        if self._lo is None:
            self.target, self.dist = np.array([0.0, 0.0, 0.5]), 12.0
            return
        centre = 0.5 * (self._lo + self._hi)
        radius = max(0.5 * float(np.linalg.norm(self._hi - self._lo)), 0.5)
        half_v = math.radians(FOV_DEG) / 2
        half_h = math.atan(math.tan(half_v) * max(w, 1) / max(h, 1))
        self.target = centre
        self.dist = float(min(max(1.08 * radius / math.sin(min(half_v, half_h)), 0.3), 5000.0))

    # ------------------------------------------------------------------ camera
    def _camera(self, w: int, h: int):
        """(eye, R, f, near): R's rows are the camera's right, up and forward axes; f is the focal length in px."""
        ce, se, ca, sa = math.cos(self.el), math.sin(self.el), math.cos(self.az), math.sin(self.az)
        back = np.array([ce * ca, ce * sa, se])               # from the target towards the eye
        fwd = -back
        right = np.array([-sa, ca, 0.0])                      # forward x world-up
        up = np.cross(right, fwd)
        eye = self.target + self.dist * back
        f = 0.5 * h / math.tan(math.radians(FOV_DEG) / 2)
        return eye, np.stack([right, up, fwd]), f, max(0.02, 0.002 * self.dist)

    def project(self, points, size=None):
        """World points (n, 3) -> (pixels (n, 2) float, visible (n,) bool).

        ``visible`` means in front of the camera (beyond the near plane); the pixels of points behind it are
        meaningless. Pixels are in widget coordinates (or in a ``size`` = (w, h) image), and may lie outside it.
        """
        w, h = size if size is not None else (self.width(), self.height())
        eye, R, f, near = self._camera(w, h)
        cam = (np.asarray(points, float).reshape(-1, 3) - eye) @ R.T
        z = cam[:, 2]
        zs = np.where(z > near, z, 1.0)
        px = np.stack([0.5 * w + f * cam[:, 0] / zs, 0.5 * h - f * cam[:, 1] / zs], axis=1)
        return px, z > near

    def _segments(self, a: np.ndarray, b: np.ndarray, w: int, h: int) -> np.ndarray:
        """Screen segments (k, 4) of the world segments a[i] -> b[i], clipped against the near plane."""
        eye, R, f, near = self._camera(w, h)
        ca, cb = (np.asarray(a, float) - eye) @ R.T, (np.asarray(b, float) - eye) @ R.T
        za, zb = ca[:, 2], cb[:, 2]
        keep = (za > near) | (zb > near)
        ca, cb, za, zb = ca[keep], cb[keep], za[keep], zb[keep]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(za != zb, (near - za) / (zb - za), 0.0)[:, None]
        cut = ca + (cb - ca) * t
        ca = np.where((za <= near)[:, None], cut, ca)
        cb = np.where((zb <= near)[:, None], cut, cb)
        za, zb = np.maximum(ca[:, 2], near), np.maximum(cb[:, 2], near)
        return np.stack([0.5 * w + f * ca[:, 0] / za, 0.5 * h - f * ca[:, 1] / za,
                         0.5 * w + f * cb[:, 0] / zb, 0.5 * h - f * cb[:, 1] / zb], axis=1)

    # ------------------------------------------------------------------ colours
    def _sensor(self):
        if self._traj is not None:
            return self._traj[-1].astype(float)
        if self._robot is not None:
            return np.array([self._robot[0], self._robot[1], 0.0])
        return None

    def _rebin(self) -> None:
        """Sort the points into colour bins once, so a redraw is a handful of drawPoints calls."""
        t0 = time.perf_counter()
        self._rebin_now()
        self._version += 1
        self.last_bin_ms = (time.perf_counter() - t0) * 1000.0

    def _rebin_now(self) -> None:
        self._dirty = False
        pts, n = self._pts, len(self._pts)
        mode = self.color_mode
        value = None
        self._legend = ""
        if n and mode == "height":
            value, unit = pts[:, 2], "height"
        elif n and mode == "distance":
            ref = self._sensor()
            if ref is None:
                ref = 0.5 * (self._lo + self._hi)
            d = pts - ref.astype(np.float32)
            value, unit = np.sqrt(np.einsum("ij,ij->i", d, d)), "distance"
        elif n and mode == "age" and self._stamps is not None:
            value, unit = self._stamps, "age"
        if value is None:                                                  # plain, empty, or nothing to colour by
            self._sorted, self._starts, self._colors = pts, np.array([0, n]), [QColor("#c3ccd9")]
            return
        sample = value[::max(1, n // 20000)]
        if mode == "age":
            lo, hi = float(sample.min()), float(value.max())
        else:
            lo, hi = (float(v) for v in np.percentile(sample, [1.0, 99.0]))
        if not hi - lo > 1e-6:
            hi = lo + 1e-6
        bins = np.clip((value - lo) * (N_BINS / (hi - lo)), 0, N_BINS - 1).astype(np.uint8)
        order = np.argsort(bins, kind="stable")                            # radix sort on bytes: fast
        self._sorted = pts[order]
        self._starts = np.searchsorted(bins[order], np.arange(N_BINS + 1))
        self._colors = _fade(N_BINS) if mode == "age" else _ramp(N_BINS)
        if mode == "age":
            self._legend = f"age  {hi - lo:.0f} s ago (dim) to now (bright)"
        else:
            self._legend = f"{unit}  {lo:.1f} m (blue) to {hi:.1f} m (red)"

    # ------------------------------------------------------------------ drawing
    def _layer(self, name: str, w: int, h: int) -> QImage:
        img = self._layers.get(name)
        if img is None or img.width() != w or img.height() != h:
            img = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
            self._layers[name] = img
        img.fill(Qt.transparent)
        return img

    def _draw_points(self, p: QPainter, w: int, h: int, stride: int) -> int:
        """The cloud, one drawPoints call per colour bin. Points are drawn one pixel wide into a transparent
        layer, which is then stamped size x size times: far cheaper than a wide pen for hundreds of thousands
        of points, and the cost hardly depends on the point size. The finished layer is kept, so a redraw
        that only moves the robot or replaces the latest scan does not touch the cloud again."""
        s = self.point_size
        key = (self._version, self.az, self.el, self.dist, tuple(self.target), w, h, s, stride)
        if key != self._cloud_key or "cloud" not in self._layers:
            self._cloud_n = self._render_cloud(w, h, stride)
            self._cloud_key = key
        if self._cloud_n:
            p.drawImage(0, 0, self._layers["cloud"])
        return self._cloud_n

    def _render_cloud(self, w: int, h: int, stride: int) -> int:
        s = self.point_size
        out = self._layer("cloud", w, h)
        pts = self._sorted[::stride] if stride > 1 else self._sorted
        if not len(pts):
            return 0
        eye, R, f, near = self._camera(w, h)
        cam = (pts - eye.astype(np.float32)) @ R.T.astype(np.float32)
        z = cam[:, 2]
        inv = np.float32(f) / np.maximum(z, np.float32(near))
        x = cam[:, 0] * inv
        x += np.float32(0.5 * w)
        y = cam[:, 1] * (-inv)
        y += np.float32(0.5 * h)
        ok = (z > near) & (x >= 0) & (x < w - (s - 1)) & (y >= 0) & (y < h - (s - 1))
        sel = np.flatnonzero(ok)
        if not len(sel):
            return 0
        xy = np.empty((len(sel), 2), np.float64)
        xy[:, 0] = x[sel]
        xy[:, 1] = y[sel]
        np.floor(xy, out=xy)
        xy += 0.5                                                          # pixel centres: crisp single pixels
        cuts = np.searchsorted(sel, (self._starts + stride - 1) // stride)
        layer = out if s == 1 else self._layer("points", w, h)
        lp = QPainter(layer)
        for k, color in enumerate(self._colors):
            a, b = int(cuts[k]), int(cuts[k + 1])
            if b > a:
                lp.setPen(QPen(color, 0))
                lp.drawPoints(_polygon(xy[a:b]))
        lp.end()
        if s > 1:                                                          # widen: s stamps across, then s down
            wide = self._layer("wide", w, h)
            wp = QPainter(wide)
            for i in range(s):
                wp.drawImage(i, 0, layer)
            wp.end()
            op = QPainter(out)
            for j in range(s):
                op.drawImage(0, j, wide)
            op.end()
        return len(sel)

    def _draw_grid(self, p: QPainter, w: int, h: int) -> None:
        """1 m lines on the ground (z = 0) under the cloud; every 5th one stronger. Coarser on very large maps."""
        if self._lo is not None:
            lo, hi = np.floor(self._lo[:2]) - 1, np.ceil(self._hi[:2]) + 1
        else:
            c = np.round(self.target[:2])
            lo, hi = c - 8, c + 8
        step = next((s for s in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0)
                     if max(hi - lo) / s <= 120), 1000.0)
        lo, hi = np.floor(lo / step) * step, np.ceil(hi / step) * step
        xs, ys = np.arange(lo[0], hi[0] + step / 2, step), np.arange(lo[1], hi[1] + step / 2, step)
        a = np.concatenate([np.stack([xs, np.full_like(xs, lo[1]), np.zeros_like(xs)], 1),
                            np.stack([np.full_like(ys, lo[0]), ys, np.zeros_like(ys)], 1)])
        b = np.concatenate([np.stack([xs, np.full_like(xs, hi[1]), np.zeros_like(xs)], 1),
                            np.stack([np.full_like(ys, hi[0]), ys, np.zeros_like(ys)], 1)])
        major = np.isclose(np.concatenate([xs, ys]) / (5 * step) % 1, 0, atol=1e-6) | \
            np.isclose(np.concatenate([xs, ys]) / (5 * step) % 1, 1, atol=1e-6)
        eye, R, f, near = self._camera(w, h)
        front = (((a - eye) @ R.T)[:, 2] > near) | (((b - eye) @ R.T)[:, 2] > near)
        seg = np.clip(self._segments(a, b, w, h), -1e6, 1e6)
        major = major[front]
        for strong, color in ((False, QColor(255, 255, 255, 16)), (True, QColor(255, 255, 255, 40))):
            lines = [QLineF(x0, y0, x1, y1) for x0, y0, x1, y1 in seg[major == strong].tolist()]
            if lines:
                p.setPen(QPen(color, 1))
                p.drawLines(lines)

    def _draw_path(self, p: QPainter, w: int, h: int) -> None:
        tr = self._traj
        if tr is None or len(tr) < 2:
            return
        if len(tr) > 4000:                                                 # plenty for a polyline on screen
            tr = np.concatenate([tr[:-1:len(tr) // 2000], tr[-1:]])
        px, vis = self.project(tr, (w, h))
        px = np.clip(px, -1e5, 1e5)
        p.setPen(QPen(QColor(TRAJ_COLOR), 1))                               # hairline: wide antialiased strokes are slow
        edges = np.flatnonzero(np.diff(np.concatenate([[False], vis, [False]]).astype(np.int8)))
        for a, b in zip(edges[::2].tolist(), edges[1::2].tolist()):        # runs of points in front of the camera
            if b - a >= 2:
                p.drawPolyline(_polygon(px[a:b]))

    def _draw_scan(self, p: QPainter, w: int, h: int) -> None:
        if self._scan is None:
            return
        px, vis = self.project(self._scan, (w, h))
        ok = vis & (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        if ok.any():
            pen = QPen(QColor(SCAN_COLOR), self.point_size + 1)
            pen.setCapStyle(Qt.SquareCap)
            p.setPen(pen)
            p.drawPoints(_polygon(px[ok]))

    def _draw_robot(self, p: QPainter, w: int, h: int) -> None:
        if self._robot is None:
            return
        x, y, yaw = (list(self._robot) + [0.0])[:3]
        z = 0.0
        base = np.array([x, y, z])
        tip = base + 0.45 * np.array([math.cos(yaw), math.sin(yaw), 0.0])
        px, vis = self.project(np.stack([base, tip]), (w, h))
        if not vis.all() or not np.all(np.abs(px) < 1e5):
            return
        d = px[1] - px[0]
        n = float(np.linalg.norm(d))
        d = d / n if n > 1e-6 else np.array([0.0, -1.0])
        length = max(n, 28.0)                                              # never smaller than a readable arrow
        side = np.array([-d[1], d[0]])
        a, t = px[0], px[0] + d * length
        head = [t, t - d * 9 + side * 6, t - d * 9 - side * 6]
        p.setPen(QPen(QColor(ACCENT), 2.4))
        p.drawLine(QPointF(*a), QPointF(*(t - d * 8)))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(ACCENT))
        p.drawPolygon(QPolygonF([QPointF(*q) for q in head]))
        p.setBrush(QColor(BG))
        p.setPen(QPen(QColor(ACCENT), 2))
        p.drawEllipse(QPointF(*a), 4.5, 4.5)
        p.setBrush(Qt.NoBrush)

    def _draw_triad(self, p: QPainter, w: int, h: int) -> None:
        _, R, _, _ = self._camera(w, h)
        o = np.array([38.0, h - 38.0])
        axes = [(R[:, 0], "#ef5350", "x"), (R[:, 1], "#3ecf8e", "y"), (R[:, 2], "#4c9aff", "z")]
        for v, color, name in sorted(axes, key=lambda a: -a[0][2]):        # the axis pointing away first
            e = o + 26 * np.array([v[0], -v[1]])
            p.setPen(QPen(QColor(color), 2))
            p.drawLine(QPointF(*o), QPointF(*e))
            p.drawText(QPointF(*(e + 5 * np.array([v[0], -v[1]]) + np.array([-3.0, 4.0]))), name)

    def _paint(self, p: QPainter, w: int, h: int, stride: int = 1) -> None:
        t0 = time.perf_counter()
        if self._dirty:
            self._rebin()
        p.fillRect(0, 0, w, h, QColor(BG))
        p.setRenderHint(QPainter.Antialiasing, True)
        self._draw_grid(p, w, h)
        p.setRenderHint(QPainter.Antialiasing, False)
        self.last_drawn = self._draw_points(p, w, h, stride)
        self._draw_scan(p, w, h)
        p.setRenderHint(QPainter.Antialiasing, True)
        self._draw_path(p, w, h)
        self._draw_robot(p, w, h)
        self._draw_triad(p, w, h)
        p.setPen(QColor(DIM))
        if not len(self._pts):
            p.drawText(0, 0, w, h, Qt.AlignCenter, "No lidar points yet.\nTurn the lidar on and press Record.")
        elif self._legend:
            p.drawText(8, 4, w - 16, 18, Qt.AlignLeft | Qt.AlignVCenter, self._legend)
        if stride > 1:
            p.drawText(8, 4, w - 16, 18, Qt.AlignRight | Qt.AlignVCenter, f"moving: 1 point in {stride}")
        p.setPen(QPen(QColor(LINE), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(0, 0, w - 1, h - 1)
        self.last_draw_ms = (time.perf_counter() - t0) * 1000.0

    def render_image(self, w: int, h: int) -> QImage:
        """The view drawn (in full, never subsampled) into a w x h image."""
        img = QImage(int(w), int(h), QImage.Format_ARGB32_Premultiplied)
        p = QPainter(img)
        try:
            self._paint(p, int(w), int(h), 1)
        finally:
            p.end()
        return img

    def paintEvent(self, _event) -> None:
        stride = 1
        if self._dragging and len(self._pts) > DRAG_POINTS:
            stride = int(math.ceil(len(self._pts) / DRAG_POINTS))
        w, h = self.width(), self.height()
        frame = self._layers.get("frame")         # composed off screen, then one blit (also on scaled displays)
        if frame is None or frame.width() != w or frame.height() != h:
            frame = self._layers["frame"] = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        fp = QPainter(frame)
        try:
            self._paint(fp, w, h, stride)
        finally:
            fp.end()
        p = QPainter(self)
        p.drawImage(0, 0, frame)
        p.end()

    # ------------------------------------------------------------------ mouse
    def _moving(self) -> None:
        self._dragging = True
        self._idle.start()
        self.update()

    def _settle(self) -> None:
        if self._press is not None:              # a button is still down: stay light, check again later
            self._idle.start()
            return
        self._dragging = False
        self.update()

    def mousePressEvent(self, e) -> None:
        self._press = (e.position(), e.button())

    def mouseMoveEvent(self, e) -> None:
        if self._press is None:
            return
        last, button = self._press
        d = e.position() - last
        self._press = (e.position(), button)
        if button == Qt.LeftButton:
            self.az -= 0.008 * d.x()
            self.el = min(max(self.el + 0.008 * d.y(), math.radians(-89.0)), math.radians(89.0))
        else:                                    # right / middle: slide the look-at point in the screen plane
            _, R, f, _ = self._camera(self.width(), self.height())
            self.target = self.target + (-d.x() * R[0] + d.y() * R[1]) * (self.dist / f)
        self._moving()

    def mouseReleaseEvent(self, _e) -> None:
        self._press = None
        if self._dragging:
            self._idle.stop()
            self._settle()

    def wheelEvent(self, e) -> None:
        self.dist = float(min(max(self.dist * 0.88 ** (e.angleDelta().y() / 120.0), 0.2), 5000.0))
        self._moving()

    def mouseDoubleClickEvent(self, _e) -> None:
        if self._lo is not None:
            self.target = 0.5 * (self._lo + self._hi)
            self.update()

    def contextMenuEvent(self, e) -> None:       # the right button pans; no menu
        e.accept()


def _yaw(q) -> float:
    return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


class LidarCloudPanel(QWidget):
    """Dock content: the 3D lidar view with record / clear / save / load and display options."""

    def __init__(self, studio):
        super().__init__()
        self.studio = studio
        self.view = CloudView(self)
        self._shown = None                       # what the view currently shows: (cloud id, seq, scan id, pose)
        self._not_before = 0.0                   # no new cloud is pushed before this time (keeps the interface free)

        self.record = QCheckBox("Record")
        self.record.setToolTip("Add every lidar scan to the cloud")
        self.record.setChecked(bool(getattr(self._sess(), "cloud_on", False)))
        self.record.toggled.connect(self._set_record)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setToolTip("Forget everything the lidar has seen")
        self.clear_btn.clicked.connect(self.clear)
        self.save_btn = QPushButton("Save...")
        self.save_btn.setToolTip("Save the cloud as .ply (point-cloud viewers) or .npz (exact, with times and path)")
        self.save_btn.clicked.connect(self.save)
        self.load_btn = QPushButton("Load...")
        self.load_btn.clicked.connect(self.load)
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setToolTip("Frame the whole cloud")
        self.fit_btn.clicked.connect(self.view.fit)
        self.color = QComboBox()
        for mode in COLOR_MODES:
            self.color.addItem(mode.capitalize(), mode)
        self.color.setToolTip("Height above the ground, distance from the sensor, age (older = dimmer) or one colour")
        self.color.currentIndexChanged.connect(lambda _i: self.view.set_color_mode(self.color.currentData()))
        self.size = QSpinBox()
        self.size.setRange(1, 4)
        self.size.setValue(self.view.point_size)
        self.size.setSuffix(" px")
        self.size.valueChanged.connect(self.view.set_point_size)
        self.follow = QCheckBox("Follow robot")
        self.follow.setToolTip("Keep the view centred on the robot")
        self.follow.toggled.connect(lambda _on: self.refresh(force=True))
        self.info = dim("no points")
        self.info.setWordWrap(False)
        self.info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        # two short rows rather than one long one, so the dock can be narrow
        top = QHBoxLayout()
        top.setSpacing(6)
        for wdg in (self.record, self.clear_btn, self.save_btn, self.load_btn, self.fit_btn):
            top.addWidget(wdg)
        top.addStretch(1)
        opts = QHBoxLayout()
        opts.setSpacing(6)
        for wdg in (QLabel("Colour"), self.color, QLabel("Point size"), self.size, self.follow):
            opts.addWidget(wdg)
        opts.addStretch(1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        lay.addLayout(top)
        lay.addLayout(opts)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.info)

    # ------------------------------------------------------------------ session access
    def _sess(self):
        return getattr(self.studio, "sess", None)

    def _cloud(self):
        return getattr(self._sess(), "cloud", None)

    def _log(self, text: str) -> None:
        log = getattr(self.studio, "log", None)
        if callable(log):
            log(text)

    def _lock(self):
        lock = getattr(self._sess(), "lock", None)
        return lock if lock is not None else contextlib.nullcontext()

    # ------------------------------------------------------------------ actions
    def _set_record(self, on: bool) -> None:
        sess = self._sess()
        if sess is not None:
            try:
                sess.cloud_on = bool(on)
            except AttributeError:
                pass

    def clear(self) -> None:
        cloud = self._cloud()
        if cloud is not None:
            cloud.clear()
        self.refresh(force=True)

    def save(self) -> None:
        cloud = self._cloud()
        if cloud is None or not len(cloud):
            self._log("Lidar cloud: nothing to save yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save lidar point cloud", "lidar_cloud.ply",
                                              "Point cloud (*.ply *.npz);;PLY, for point-cloud viewers (*.ply);;"
                                              "NumPy archive, exact (*.npz)")
        if not path:
            return
        if os.path.splitext(path)[1].lower() not in (".ply", ".npz"):
            path += ".ply"
        try:
            cloud.save(path)
        except Exception as e:                   # a full disk or a read-only folder must not take the app down
            self._log(f"Lidar cloud: could not save {path}: {e}")
            return
        self._log(f"Lidar cloud: saved {len(cloud):,} points to {path}")

    def load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load lidar point cloud", "", "Point cloud (*.ply *.npz)")
        if not path:
            return
        try:
            loaded = PointCloudMap.load(path)
        except Exception as e:
            self._log(f"Lidar cloud: could not load {path}: {e}")
            return
        cloud, sess = self._cloud(), self._sess()
        if cloud is not None and hasattr(cloud, "assign"):
            cloud.assign(loaded)                 # in place: the simulation thread keeps its reference
        elif sess is not None:
            sess.cloud = loaded
        self._log(f"Lidar cloud: loaded {len(loaded):,} points from {path}")
        self.refresh(force=True)
        self.view.fit()

    # ------------------------------------------------------------------ refresh
    def _pose(self, sess):
        """(x, y, yaw) of a free-floating robot, else None. Call with the session lock held."""
        if sess is None or getattr(sess, "task", None) is None:
            return None
        try:
            q = np.array(sess.data.qpos[:7], float)
            return float(q[0]), float(q[1]), _yaw(q[3:7])
        except Exception:
            return None

    def refresh(self, force: bool = False) -> None:
        """Called about ten times a second by the main window. Cheap unless something changed: pushes new
        data into the view when the cloud's ``seq`` moved, the latest scan was replaced or the robot moved.
        Does nothing while the panel is hidden, unless ``force``."""
        sess = self._sess()
        on = bool(getattr(sess, "cloud_on", False))
        if self.record.isChecked() != on:
            self.record.blockSignals(True)
            self.record.setChecked(on)
            self.record.blockSignals(False)
        if not (force or self.isVisible()):
            return
        cloud = self._cloud()
        with self._lock():
            pose = self._pose(sess)
            scan = getattr(sess, "lidar_points", None)
            scan_id = getattr(sess, "lidar_seq", None)
            if scan_id is None:
                scan_id = id(scan)               # the session replaces the array with every scan
            if not getattr(getattr(sess, "lidar_cfg", None), "enabled", True):
                scan = None                      # the lidar is off: its last scan is stale
            if scan is not None:
                scan = np.array(scan, np.float32).reshape(-1, 3)        # a copy, taken under the lock
        if cloud is None:
            pts = stamps = traj = None
            seq = -1
        elif hasattr(cloud, "snapshot"):
            pts, stamps, traj, seq = cloud.snapshot()
        else:
            pts, stamps, traj, seq = cloud.points, getattr(cloud, "stamps", None), getattr(cloud, "trajectory", None), \
                getattr(cloud, "seq", None)
            if stamps is not None and len(stamps) != len(pts):
                n = min(len(stamps), len(pts))
                pts, stamps = pts[:n], stamps[:n]
        key = (id(cloud), seq, scan_id if scan is not None else None,
               None if pose is None else tuple(round(v, 2) for v in pose))
        now = time.monotonic()
        if force or ((key != self._shown or seq is None) and now >= self._not_before):
            self._shown = key
            self.view.set_cloud(pts, stamps, traj, scan, pose, seq=seq)
            self._label(cloud)
            # a big cloud takes a while to colour and draw: leave the interface twice that long to itself
            cost = self.view.last_draw_ms + self.view.last_bin_ms
            self._not_before = now + (3e-3 * cost if cost > 20.0 else 0.0)
        if self.follow.isChecked():
            if pose is not None:
                self.view.set_look_at(pose[:2])
            elif traj is not None and len(traj):
                self.view.set_look_at(traj[-1])

    def _label(self, cloud) -> None:
        if cloud is None:
            self.info.setText("no lidar cloud in this session")
            return
        n = len(cloud)
        text = f"{n:,} points"
        b = cloud.bounds() if n else None
        if b is not None:
            ext = np.asarray(b[1], float) - np.asarray(b[0], float)
            text += f"   {ext[0]:.1f} m x {ext[1]:.1f} m x {ext[2]:.1f} m"
        voxel = float(getattr(cloud, "voxel", 0.0))
        if voxel > 0:
            text += f"   voxel {voxel * 100:g} cm"
        if getattr(cloud, "full", False):
            text += "   (full: not recording new points)"
        self.info.setText(text)
