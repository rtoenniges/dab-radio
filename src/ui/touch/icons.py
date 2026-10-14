"""The symbols of the touch surface, drawn rather than taken from a font: the fonts on
the Raspberry Pi need not have ⏮ or ⏻, and drawn symbols take the colour of the scheme.

Every symbol is laid out on a grid of 24 x 24 units and scaled to the size asked for.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)

GRID = 24.0
_LINE = 2.0  # width of outlined strokes, in grid units
# A badge (`badged_pixmap`), in grid units: in the lower right corner the Wi-Fi leaves
# free, clear of the ends of its arcs and of the dot, standing beside the dot.
_BADGE_LEFT = 20.0
_BADGE_BASELINE = 23.0
_BADGE_HEIGHT = 13.5  # the size of its font


def _polygon(*points: tuple[float, float]) -> QPolygonF:
    return QPolygonF([QPointF(x, y) for x, y in points])


def _play(p: QPainter) -> None:
    p.drawPolygon(_polygon((7, 4.5), (19.5, 12), (7, 19.5)))


def _stop(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(5.5, 5.5, 13, 13), 2, 2)


def _pause(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(5.5, 4.5, 4.5, 15), 1.5, 1.5)
    p.drawRoundedRect(QRectF(14, 4.5, 4.5, 15), 1.5, 1.5)


def _previous(p: QPainter) -> None:
    p.drawRect(QRectF(4.5, 5, 2.5, 14))
    p.drawPolygon(_polygon((19.5, 5), (8.5, 12), (19.5, 19)))


def _next(p: QPainter) -> None:
    p.drawRect(QRectF(17, 5, 2.5, 14))
    p.drawPolygon(_polygon((4.5, 5), (15.5, 12), (4.5, 19)))


def _speaker(p: QPainter) -> None:
    p.drawPolygon(_polygon((3, 9), (7, 9), (12, 4.5), (12, 19.5), (7, 15), (3, 15)))


def _volume(p: QPainter) -> None:
    _speaker(p)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawArc(QRectF(9, 8, 8, 8), -60 * 16, 120 * 16)
    p.drawArc(QRectF(8, 4.5, 14, 15), -60 * 16, 120 * 16)


def _muted(p: QPainter) -> None:
    _speaker(p)
    p.drawLine(QPointF(15, 9), QPointF(21, 15))
    p.drawLine(QPointF(21, 9), QPointF(15, 15))


def _radio(p: QPainter) -> None:
    """A mast sending: a dot with waves to both sides."""
    p.drawEllipse(QPointF(12, 12), 2.2, 2.2)
    p.setBrush(Qt.BrushStyle.NoBrush)
    for radius in (5.5, 9.5):
        box = QRectF(12 - radius, 12 - radius, 2 * radius, 2 * radius)
        p.drawArc(box, -45 * 16, 90 * 16)
        p.drawArc(box, 135 * 16, 90 * 16)


def _stations(p: QPainter) -> None:
    for x in (4, 13):
        for y in (4, 13):
            p.drawRoundedRect(QRectF(x, y, 7, 7), 1.5, 1.5)


def _services(p: QPainter) -> None:
    """A list: three bullets with their lines."""
    for y in (6, 12, 18):
        p.drawEllipse(QPointF(5, y), 1.8, 1.8)
        p.drawLine(QPointF(10, y), QPointF(21, y))


def _calendar(p: QPainter) -> None:
    """A sheet of a calendar: frame, top bar, two rings and a grid of days."""
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(3.5, 5, 17, 15.5), 2.5, 2.5)
    p.drawLine(QPointF(3.5, 9.5), QPointF(20.5, 9.5))
    p.drawLine(QPointF(8, 3), QPointF(8, 6.5))
    p.drawLine(QPointF(16, 3), QPointF(16, 6.5))
    color = p.pen().color()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    for x in (8, 12, 16):
        for y in (13.5, 17):
            p.drawEllipse(QPointF(x, y), 1.1, 1.1)


def _chevron(p: QPainter) -> None:
    """Points to the right: a tap leads on to another page."""
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolyline(_polygon((9, 5), (16, 12), (9, 19)))


def _expand(p: QPainter) -> None:
    """Points down: a tap opens more beneath."""
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolyline(_polygon((5, 8.5), (12, 15.5), (19, 8.5)))


def _collapse(p: QPainter) -> None:
    """Points up: a tap closes what is open beneath."""
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolyline(_polygon((5, 15.5), (12, 8.5), (19, 15.5)))


def _signal(p: QPainter) -> None:
    path = QPainterPath(QPointF(2, 12))
    path.cubicTo(QPointF(5, 2), QPointF(8, 2), QPointF(10, 12))
    path.cubicTo(QPointF(12, 22), QPointF(15, 22), QPointF(17, 12))
    path.cubicTo(QPointF(18.5, 5), QPointF(20.5, 5), QPointF(22, 12))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(path)


def _settings(p: QPainter) -> None:
    """Three sliders."""
    for y, knob in ((6, 15), (12, 8), (18, 13)):
        p.drawLine(QPointF(3, y), QPointF(21, y))
        p.drawEllipse(QPointF(knob, y), 2.6, 2.6)


def _equalizer(p: QPainter) -> None:
    """Four upright sliders, set to different heights: the sound (unlike the settings'
    three lying ones)."""
    for x, knob in ((3.5, 14), (9.17, 8), (14.83, 11), (20.5, 16)):
        p.drawLine(QPointF(x, 3), QPointF(x, 21))
        p.drawEllipse(QPointF(x, knob), 2.2, 2.2)


def _lock(p: QPainter) -> None:
    """A padlock: a network that needs a password."""
    p.drawRoundedRect(QRectF(5, 11, 14, 10), 2, 2)
    p.setBrush(Qt.BrushStyle.NoBrush)
    shackle = QPainterPath(QPointF(8, 11))
    shackle.lineTo(QPointF(8, 8))
    shackle.arcTo(QRectF(8, 3.5, 8, 9), 180, -180)
    shackle.lineTo(QPointF(16, 11))
    p.drawPath(shackle)


def _refresh(p: QPainter) -> None:
    """An arrow round a circle, clockwise: look again."""
    p.save()
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawArc(QRectF(4.5, 4.5, 15, 15), 45 * 16, 270 * 16)
    p.restore()
    p.drawPolygon(_polygon((14.6, 9.4), (20.0, 4.0), (20.6, 10.0)))


def _eye(p: QPainter) -> None:
    """An eye: shows what is typed in place of dots."""
    color = p.pen().color()
    p.setBrush(Qt.BrushStyle.NoBrush)
    outline = QPainterPath(QPointF(2, 12))
    outline.quadTo(QPointF(12, 1), QPointF(22, 12))
    outline.quadTo(QPointF(12, 23), QPointF(2, 12))
    p.drawPath(outline)
    p.setBrush(color)
    p.drawEllipse(QPointF(12, 12), 3.2, 3.2)


def _star_points() -> QPolygonF:
    """Five points, the top one upright; the shape stands in the middle of the grid,
    as high as it is wide."""
    outer = 9.5
    inner = outer * 0.382  # the inner corners on the lines between the outer ones
    # From the top point down to the two lower ones: outer (1 + cos 36°).
    centre_y = GRID / 2 + outer * (1 - math.cos(math.radians(36))) / 2
    points = []
    for corner in range(10):
        angle = math.radians(-90 + 36 * corner)
        radius = outer if corner % 2 == 0 else inner
        points.append((GRID / 2 + radius * math.cos(angle), centre_y + radius * math.sin(angle)))
    return _polygon(*points)


_STAR = _star_points()


def _star(p: QPainter) -> None:
    """A favourite."""
    p.drawPolygon(_STAR)


def _star_outline(p: QPainter) -> None:
    """Not a favourite (yet)."""
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolygon(_STAR)


def _headphones(p: QPainter) -> None:
    """Headphones: the band over the head and two cups. The loudspeaker button of the
    radio page shows them while the audio goes to a Bluetooth device."""
    color = p.pen().color()
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawArc(QRectF(4, 5, 16, 16), 0, 180 * 16)
    p.setBrush(color)
    p.drawRoundedRect(QRectF(3.5, 13, 4, 7), 1.5, 1.5)
    p.drawRoundedRect(QRectF(16.5, 13, 4, 7), 1.5, 1.5)


def _headphones_muted(p: QPainter) -> None:
    """Headphones struck through."""
    _headphones(p)
    p.drawLine(QPointF(5, 20), QPointF(20, 4))


def _bluetooth(p: QPainter) -> None:
    """The Bluetooth rune."""
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolyline(_polygon((6.5, 7), (17.5, 17), (12, 22), (12, 2), (17.5, 7), (6.5, 17)))


def _wifi(p: QPainter) -> None:
    """The Wi-Fi: a dot with three arcs over it."""
    p.drawEllipse(QPointF(12, 20), 1.8, 1.8)
    p.setBrush(Qt.BrushStyle.NoBrush)
    for radius in (5.5, 9.5, 13.5):
        box = QRectF(12 - radius, 20 - radius, 2 * radius, 2 * radius)
        p.drawArc(box, 45 * 16, 90 * 16)


def _loudspeaker(p: QPainter) -> None:
    """A loudspeaker box: a device, unlike `_volume`, which is the sound."""
    color = p.pen().color()
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(6, 2.5, 12, 19), 2, 2)
    p.drawEllipse(QPointF(12, 15), 3.2, 3.2)
    p.setBrush(color)
    p.drawEllipse(QPointF(12, 7.5), 1.3, 1.3)


def _phone(p: QPainter) -> None:
    """A smartphone."""
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(7, 2.5, 10, 19), 2, 2)
    p.drawLine(QPointF(10.5, 18.5), QPointF(13.5, 18.5))


def _folder(p: QPainter) -> None:
    """A folder: its lid with a tab on the left."""
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolygon(_polygon((3, 19), (3, 5.5), (9.5, 5.5), (11.5, 8), (21, 8), (21, 19)))


def _usb(p: QPainter) -> None:
    """A USB stick lying, its plug to the right."""
    color = p.pen().color()
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(2.5, 7.5, 13, 9), 2, 2)
    p.drawRect(QRectF(15.5, 9.5, 6, 5))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    p.drawRect(QRectF(17.5, 10.8, 1.4, 1.2))
    p.drawRect(QRectF(17.5, 12.6, 1.4, 1.2))


def _eject(p: QPainter) -> None:
    """Eject: a triangle over a bar."""
    p.drawPolygon(_polygon((12, 4), (20, 13.5), (4, 13.5)))
    p.drawRoundedRect(QRectF(4, 16.5, 16, 3), 1, 1)


def _arrow_up(p: QPainter) -> None:
    """Up: out of a folder into the one above."""
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawLine(QPointF(12, 20), QPointF(12, 5))
    p.drawPolyline(_polygon((5.5, 11.5), (12, 5), (18.5, 11.5)))


def _journaline(p: QPainter) -> None:
    """A page of a newspaper: a picture at the top left beside two lines, a line
    beneath."""
    color = p.pen().color()
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(3, 4, 18, 16), 2.5, 2.5)
    p.drawLine(QPointF(14, 8.5), QPointF(17.5, 8.5))
    p.drawLine(QPointF(14, 11.5), QPointF(17.5, 11.5))
    p.drawLine(QPointF(6.5, 15.5), QPointF(17.5, 15.5))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    p.drawRoundedRect(QRectF(6, 7, 5.5, 5.5), 1, 1)


def _web(p: QPainter) -> None:
    """A globe: its outline, a meridian and the equator."""
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(12, 12), 8.5, 8.5)
    p.drawEllipse(QPointF(12, 12), 3.8, 8.5)
    p.drawLine(QPointF(3.5, 12), QPointF(20.5, 12))


def _qr(p: QPainter) -> None:
    """A QR code: its three finder patterns and a few modules in the fourth corner."""
    color = p.pen().color()
    p.setBrush(Qt.BrushStyle.NoBrush)
    corners = ((3, 3), (14, 3), (3, 14))
    for x, y in corners:
        p.drawRect(QRectF(x + 1, y + 1, 5, 5))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    for x, y in corners:
        p.drawRect(QRectF(x + 2.7, y + 2.7, 1.6, 1.6))
    for x, y in ((14, 14), (18.6, 14), (16.3, 16.3), (14, 18.6), (18.6, 18.6)):
        p.drawRect(QRectF(x, y, 2.4, 2.4))


_SYMBOLS: dict[str, Callable[[QPainter], None]] = {
    "play": _play,
    "pause": _pause,
    "stop": _stop,
    "previous": _previous,
    "next": _next,
    "volume": _volume,
    "muted": _muted,
    "radio": _radio,
    "stations": _stations,
    "services": _services,
    "signal": _signal,
    "settings": _settings,
    "equalizer": _equalizer,
    "calendar": _calendar,
    "chevron": _chevron,
    "expand": _expand,
    "collapse": _collapse,
    "lock": _lock,
    "refresh": _refresh,
    "eye": _eye,
    "star": _star,
    "star_outline": _star_outline,
    "headphones": _headphones,
    "headphones_muted": _headphones_muted,
    "bluetooth": _bluetooth,
    "wifi": _wifi,
    "loudspeaker": _loudspeaker,
    "phone": _phone,
    "folder": _folder,
    "usb": _usb,
    "eject": _eject,
    "arrow_up": _arrow_up,
    "journaline": _journaline,
    "web": _web,
    "qr": _qr,
}


def symbol_pixmap(
    name: str, color: QColor, size: int, pixel_ratio: float = 1.0, line: float = _LINE
) -> QPixmap:
    """The symbol `name` in `color`, `size` pixels square; its strokes `line` grid units
    wide (wider for a glow around them, `ui.touch.glow`)."""
    pixmap = QPixmap(round(size * pixel_ratio), round(size * pixel_ratio))
    pixmap.setDevicePixelRatio(pixel_ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(size / GRID, size / GRID)
    pen = QPen(color, line)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(color)
    _SYMBOLS[name](painter)
    painter.end()
    return pixmap


def badged_pixmap(
    name: str, color: QColor, size: int, badge: str, font: QFont, pixel_ratio: float = 1.0
) -> QPixmap:
    """The symbol `name` with `badge`, a short text such as "5", bold in its lower right
    corner and on beyond it as far as the text needs: as high as the symbol, at least as
    wide. Meant for a symbol that leaves the corner free, such as the Wi-Fi."""
    if not badge:
        return symbol_pixmap(name, color, size, pixel_ratio)
    scale = size / GRID
    font = QFont(font)
    font.setBold(True)
    font.setPixelSize(round(_BADGE_HEIGHT * scale))
    left = _BADGE_LEFT * scale
    width = max(size, math.ceil(left + QFontMetricsF(font).horizontalAdvance(badge)))
    pixmap = QPixmap(round(width * pixel_ratio), round(size * pixel_ratio))
    pixmap.setDevicePixelRatio(pixel_ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    painter.drawPixmap(0, 0, symbol_pixmap(name, color, size, pixel_ratio))
    painter.setFont(font)
    painter.setPen(color)
    painter.drawText(QPointF(left, _BADGE_BASELINE * scale), badge)
    painter.end()
    return pixmap


def symbol_icon(name: str, color: QColor, size: QSize, active_color: QColor | None = None) -> QIcon:
    """An icon of the symbol; with `active_color` drawn in that colour while the button
    is checked (e.g. the page shown in the navigation)."""
    icon = QIcon(symbol_pixmap(name, color, size.width()))
    if active_color is not None:
        icon.addPixmap(
            symbol_pixmap(name, active_color, size.width()), QIcon.Mode.Normal, QIcon.State.On
        )
    return icon


def symbol_names() -> list[str]:
    return list(_SYMBOLS)
