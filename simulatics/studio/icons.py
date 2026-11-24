"""Icons drawn with QPainter (no image files to ship)."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

from . import theme

_cache: dict[tuple[str, str, int], QIcon] = {}


def _poly(*pts):
    return QPolygonF([QPointF(x, y) for x, y in pts])


def _draw(name: str, p: QPainter, c: QColor) -> None:
    """Draw in a 24 x 24 box."""
    pen = QPen(c, 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    if name == "select":                       # arrow cursor
        p.setBrush(c)
        p.drawPolygon(_poly((6, 3), (6, 19), (10.5, 15), (13.5, 21), (16, 20), (13, 14), (19, 14)))
    elif name == "orbit":
        p.drawEllipse(QRectF(4, 8, 16, 8))
        p.drawEllipse(QRectF(8, 4, 8, 16))
    elif name == "goal":                       # flag
        p.drawLine(QPointF(7, 3), QPointF(7, 21))
        p.setBrush(c)
        p.drawPolygon(_poly((7, 4), (19, 7.5), (7, 11)))
    elif name == "push":                       # arrow hitting a block
        p.drawLine(QPointF(3, 12), QPointF(13, 12))
        p.drawPolyline(_poly((9, 8), (13, 12), (9, 16)))
        p.drawRect(QRectF(15, 6, 5, 12))
    elif name == "grab":                       # hand / spring
        path = QPainterPath(QPointF(4, 18))
        for i in range(5):
            path.lineTo(6 + i * 3, 14 if i % 2 == 0 else 18)
        p.drawPath(path)
        p.setBrush(c)
        p.drawEllipse(QPointF(18, 9), 3.2, 3.2)
        p.drawLine(QPointF(16, 13), QPointF(17.5, 11))
    elif name == "place":                      # box with plus
        p.drawRect(QRectF(4, 9, 11, 11))
        p.drawLine(QPointF(18, 3), QPointF(18, 9))
        p.drawLine(QPointF(15, 6), QPointF(21, 6))
    elif name == "measure":                    # ruler
        p.drawPolygon(_poly((3, 17), (17, 3), (21, 7), (7, 21)))
        for i in range(4):
            x = 7 + i * 3.4
            p.drawLine(QPointF(x, 20 - i * 3.4 - 3), QPointF(x + 1.8, 20 - i * 3.4 - 1.2))
    elif name == "play":
        p.setBrush(c)
        p.drawPolygon(_poly((7, 4), (20, 12), (7, 20)))
    elif name == "pause":
        p.setBrush(c)
        p.drawRect(QRectF(6, 5, 4, 14))
        p.drawRect(QRectF(14, 5, 4, 14))
    elif name == "step":
        p.setBrush(c)
        p.drawPolygon(_poly((5, 5), (15, 12), (5, 19)))
        p.drawRect(QRectF(17, 5, 2.5, 14))
    elif name == "reset":
        p.drawArc(QRectF(5, 5, 14, 14), 60 * 16, 290 * 16)
        p.setBrush(c)
        p.drawPolygon(_poly((12, 2), (17, 5.5), (12, 9)))
    elif name == "stop":
        p.setBrush(c)
        p.drawRect(QRectF(6, 6, 12, 12))
    elif name == "run":                        # lightning: run graph
        p.setBrush(c)
        p.drawPolygon(_poly((13, 2), (5, 13), (11, 13), (9, 22), (19, 10), (13, 10)))
    elif name == "camera":
        p.drawRoundedRect(QRectF(3, 7, 13, 10), 2, 2)
        p.drawPolygon(_poly((16, 10.5), (21, 7.5), (21, 16.5), (16, 13.5)))
    elif name == "frame":                      # frame selection
        for (x, y, dx, dy) in ((4, 4, 1, 1), (20, 4, -1, 1), (4, 20, 1, -1), (20, 20, -1, -1)):
            p.drawPolyline(_poly((x + 5 * dx, y), (x, y), (x, y + 5 * dy)))
        p.setBrush(c)
        p.drawEllipse(QPointF(12, 12), 2.2, 2.2)
    elif name == "graph":
        p.drawRoundedRect(QRectF(3, 4, 7, 6), 1.5, 1.5)
        p.drawRoundedRect(QRectF(14, 14, 7, 6), 1.5, 1.5)
        path = QPainterPath(QPointF(10, 7))
        path.cubicTo(15, 7, 9, 17, 14, 17)
        p.drawPath(path)
    else:                                      # fallback dot
        p.setBrush(c)
        p.drawEllipse(QPointF(12, 12), 4, 4)


def icon(name: str, color: str = theme.TEXT, size: int = 24) -> QIcon:
    key = (name, color, size)
    if key not in _cache:
        ic = QIcon()
        for mode, col in ((QIcon.Normal, color), (QIcon.Disabled, theme.LINE)):
            pm = QPixmap(size * 2, size * 2)
            pm.fill(Qt.transparent)
            p = QPainter(pm)
            p.setRenderHint(QPainter.Antialiasing)
            p.scale(size * 2 / 24, size * 2 / 24)
            _draw(name, p, QColor(col))
            p.end()
            pm.setDevicePixelRatio(2)
            ic.addPixmap(pm, mode)
        _cache[key] = ic
    return _cache[key]
