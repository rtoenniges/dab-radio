"""Marks on the impulse response: the transmitter behind each path (`dsp.tii.TiiPath`).

Over a path that carries the TII of a transmitter stands a filled triangle and the
code on a label, in a colour of its own that the list of transmitters shows as well;
over a later path with the same code (a reflection of it) an outlined triangle in that
colour; over a path without a code an outlined grey triangle and "?" (a reflection, or
a transmitter that sends no TII or too weak a one). Shared by the desktop and the
touch surface, each with colours and font of its own.
"""

from __future__ import annotations

from collections.abc import Sequence

import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

from dsp.tii import TiiPath, TiiTransmitter

# The triangle over a path, in pixels: width, height, distance of its tip from the peak.
TRIANGLE_WIDTH = 12.0
TRIANGLE_HEIGHT = 9.0
TRIANGLE_GAP = 3.0
# Between triangle and label, and around the text of a label.
LABEL_GAP = 3.0
LABEL_PADDING_X = 5.0
LABEL_PADDING_Y = 1.0
# Labels that would overlap move up by a label's height, at most this often.
MAX_TIERS = 3
# Above the curve and the shaded guard interval.
MARK_Z = 20


def code_text(code: tuple[int, int]) -> str:
    """A transmitter as the lists show it: main and sub identifier."""
    return f"{code[0]:02d}-{code[1]:02d}"


def delay_text(delay_s: float | None) -> str:
    """The delay of a transmitter as the lists show it, "-" while unknown."""
    if delay_s is None:
        return "-"
    # Rounded first: a path a hair before the synchronisation would read "-0.0".
    microseconds = round(delay_s * 1e6, 1) + 0.0
    return f"{microseconds:.1f} µs"


class TransmitterColours:
    """A colour per transmitter, kept as long as the channel stays: the first
    transmitter seen (the strongest) takes the first colour of the palette."""

    def __init__(self, palette: Sequence[QColor]) -> None:
        self._palette = list(palette)
        self._indices: dict[tuple[int, int], int] = {}

    def set_palette(self, palette: Sequence[QColor]) -> None:
        self._palette = list(palette)

    def note(self, transmitters: Sequence[TiiTransmitter]) -> None:
        """Gives the transmitters not seen so far their colours, strongest first."""
        for transmitter in transmitters:
            code = (transmitter.main_id, transmitter.sub_id)
            if code not in self._indices:
                self._indices[code] = len(self._indices)

    def colour(self, code: tuple[int, int]) -> QColor:
        index = self._indices.setdefault(code, len(self._indices))
        return QColor(self._palette[index % len(self._palette)])

    def reset(self) -> None:
        self._indices.clear()


class _Mark(QGraphicsItem):
    """A triangle over a path and, for a transmitter or a path without a code, a label
    above it; drawn in pixels at the path's position, whatever the zoom. An `optional`
    label is left out rather than moved up (`tier` None)."""

    def __init__(
        self,
        colour: QColor,
        filled: bool,
        label: str | None,
        chip: QColor | None,
        font: QFont,
        optional: bool = False,
    ) -> None:
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.colour = colour
        self.filled = filled
        self.label = label
        self.optional = optional
        self._chip = chip
        self._font = font
        self.tier: int | None = 0
        if label is None:
            self._label_size = QRectF()
        else:
            metrics = QFontMetricsF(font)
            self._label_size = QRectF(
                0,
                0,
                metrics.horizontalAdvance(label) + 2 * LABEL_PADDING_X,
                metrics.height() + 2 * LABEL_PADDING_Y,
            )

    @property
    def has_label(self) -> bool:
        """Whether a label is drawn."""
        return self.label is not None and self.tier is not None

    def triangle_rect(self) -> QRectF:
        return QRectF(
            -TRIANGLE_WIDTH / 2,
            -TRIANGLE_GAP - TRIANGLE_HEIGHT,
            TRIANGLE_WIDTH,
            TRIANGLE_HEIGHT,
        )

    def label_rect(self, tier: int | None = None) -> QRectF:
        tier = (self.tier or 0) if tier is None else tier
        height = self._label_size.height()
        bottom = -TRIANGLE_GAP - TRIANGLE_HEIGHT - LABEL_GAP - tier * (height + LABEL_GAP)
        width = self._label_size.width()
        return QRectF(-width / 2, bottom - height, width, height)

    def boundingRect(self) -> QRectF:
        rect = self.triangle_rect()
        if self.has_label:
            rect = rect.united(self.label_rect())
        return rect.adjusted(-1, -1, 1, 1)

    def paint(
        self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        triangle = self.triangle_rect()
        path = QPainterPath()
        path.moveTo(triangle.left(), triangle.top())
        path.lineTo(triangle.right(), triangle.top())
        path.lineTo(triangle.center().x(), triangle.bottom())
        path.closeSubpath()
        painter.setPen(QPen(self.colour, 1.5))
        painter.setBrush(self.colour if self.filled else Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        if not self.has_label:
            return
        rect = self.label_rect()
        if self.tier:
            # A label moved up keeps a thin line to its triangle.
            painter.setPen(QPen(self.colour, 1.0))
            painter.drawLine(QPointF(0.0, rect.bottom()), QPointF(0.0, triangle.top()))
        if self._chip is not None:
            painter.setPen(QPen(self.colour, 1.0))
            painter.setBrush(self._chip)
            painter.drawRoundedRect(rect, 4, 4)
        painter.setFont(self._font)
        painter.setPen(self.colour)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.label)


class PathMarks:
    """The marks of the paths on the plot of an impulse response."""

    def __init__(self, plot: pg.PlotWidget, colours: TransmitterColours) -> None:
        self._plot = plot
        self._colours = colours
        self._marks: list[_Mark] = []
        self._paths: list[TiiPath] = []
        self._font = QFont()
        self._unknown = QColor("grey")
        self._chip = QColor("black")
        plot.getViewBox().sigTransformChanged.connect(self._layout)

    def set_style(self, font: QFont, unknown: QColor, chip: QColor) -> None:
        """Font of the labels, colour of a path without a code, fill of a label."""
        self._font = QFont(font)
        self._font.setBold(True)
        self._unknown = QColor(unknown)
        self._chip = QColor(chip)
        self._build(self._paths)

    def show(self, paths: list[TiiPath]) -> None:
        if paths != self._paths:
            self._build(paths)

    def clear(self) -> None:
        self._build([])

    def shown(self) -> list[tuple[str | None, bool, QColor, int | None]]:
        """Per mark, by delay: its label, whether filled, its colour, its tier (None:
        label left out)."""
        return [(mark.label, mark.filled, mark.colour, mark.tier) for mark in self._marks]

    def headroom_db(self, span_db: float, padding: float = 0.1) -> float:
        """How far above the strongest path the axis must reach for its label, with a
        range of `span_db` that `SmoothedRange` pads by `padding` on either side."""
        if not self._marks:
            return 0.0
        height = self._plot.getViewBox().height()
        if height <= 0:
            return 0.0
        label = max(
            (mark.label_rect(0).height() for mark in self._marks if mark.label is not None),
            default=0.0,
        )
        needed = TRIANGLE_GAP + TRIANGLE_HEIGHT + LABEL_GAP + label + 2
        # Above the strongest path lie the headroom h and the padding of the range,
        # padding * (span + h), together `needed / height` of the range
        # (1 + 2 padding) * (span + h); so h = (span + h) * excess.
        excess = min(needed / height * (1 + 2 * padding) - padding, 0.9)
        if excess <= 0:
            return 0.0
        return span_db * excess / (1 - excess)

    def _build(self, paths: list[TiiPath]) -> None:
        self._paths = list(paths)
        for mark in self._marks:
            self._plot.removeItem(mark)
        self._marks = []
        for path in paths:
            if path.transmitter is None:
                mark = _Mark(self._unknown, False, "?", None, self._font, optional=True)
            else:
                colour = self._colours.colour(path.transmitter)
                label = None if path.echo else code_text(path.transmitter)
                mark = _Mark(colour, not path.echo, label, self._chip, self._font)
            mark.setPos(path.delay_s, path.level_db)
            mark.setZValue(MARK_Z)
            self._plot.addItem(mark, ignoreBounds=True)
            self._marks.append(mark)
        self._layout()

    def _layout(self) -> None:
        """Moves labels that would cover another label or triangle up by a tier, those
        of the transmitters first, the stronger paths first; a "?" that would cover
        something is left out instead, its outlined grey triangle tells enough."""
        placed: list[QRectF] = []
        for mark in self._marks:
            placed.append(self._in_scene(mark, mark.triangle_rect()))
        order = sorted(
            (index for index, mark in enumerate(self._marks) if mark.label is not None),
            key=lambda index: (self._marks[index].optional, -self._paths[index].level_db),
        )
        for index in order:
            mark = self._marks[index]
            tiers = 1 if mark.optional else MAX_TIERS + 1
            tier: int | None = next(
                (
                    tier
                    for tier in range(tiers)
                    if not any(
                        self._in_scene(mark, mark.label_rect(tier)).intersects(other)
                        for other in placed
                    )
                ),
                None if mark.optional else MAX_TIERS,
            )
            if mark.tier != tier:
                mark.prepareGeometryChange()
                mark.tier = tier
                # The line of a label moved up passes behind the labels below it.
                mark.setZValue(MARK_Z - (tier or 0))
            if tier is not None:
                placed.append(self._in_scene(mark, mark.label_rect(tier)))

    @staticmethod
    def _in_scene(mark: _Mark, rect: QRectF) -> QRectF:
        """A rectangle in the mark's pixels, at its place in the scene."""
        origin: QPointF = mark.scenePos()
        return rect.translated(origin)
