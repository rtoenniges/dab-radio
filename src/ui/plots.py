"""Plot helpers shared by the tabs of the main window and the touch surface."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPen, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsTextItem

# Propagation speed of radio waves in m/s: 1 µs of delay is 300 m.
SPEED_OF_LIGHT = 299_792_458.0
# Names on a map lie above all that is drawn on it (the highest beneath them, the
# listener's cross, at 20).
MAP_LABEL_Z = 30
# Width of the outline that makes the halo of a name, in pixels; half of it lies
# outside the letters.
HALO_WIDTH = 4.0


class SmoothedRange:
    """Smooths auto-range limits exponentially so that the axes do not rescale up and
    down with every new data block; only slow, real level changes such as a gain or
    frequency change lead to visible rescaling.

    `update()` additionally returns None when the range to display has only shifted
    insignificantly since the last change actually applied: a `setRange()` call with
    practically unchanged limits still triggers a complete recomputation of the view
    in pyqtgraph (another big item of the GUI load according to profiling) and can be
    avoided that way. `apply()` bundles this with the `setRange` call for the common
    case.
    """

    def __init__(
        self, alpha: float = 0.15, padding: float = 0.1, change_threshold: float = 0.02
    ) -> None:
        self._alpha = alpha
        self._padding = padding
        self._change_threshold = change_threshold
        self._low: float | None = None
        self._high: float | None = None
        self._applied_low: float | None = None
        self._applied_high: float | None = None

    def update(self, data_low: float, data_high: float) -> tuple[float, float] | None:
        """Feeds in the limits of a new data block.

        Returns:
            The smoothed, padded range to apply, or None if it has not changed
            noticeably since the last range returned.
        """
        if self._low is None:
            self._low, self._high = data_low, data_high
        else:
            self._low += self._alpha * (data_low - self._low)
            self._high += self._alpha * (data_high - self._high)

        span = self._high - self._low
        pad = span * self._padding if span > 0 else 1.0
        low, high = self._low - pad, self._high + pad

        if self._applied_low is not None:
            applied_span = self._applied_high - self._applied_low
            if (
                applied_span > 0
                and abs(low - self._applied_low) < applied_span * self._change_threshold
                and abs(high - self._applied_high) < applied_span * self._change_threshold
            ):
                return None

        self._applied_low, self._applied_high = low, high
        return low, high

    def apply(
        self, plot: pg.PlotWidget, data_low: float, data_high: float, *, axes: str = "y"
    ) -> None:
        """Feeds in new data limits and applies the smoothed range to the given axes of
        `plot` ("x", "y" or "xy") if it changed noticeably."""
        new_range = self.update(data_low, data_high)
        if new_range is None:
            return
        if "x" in axes:
            plot.setXRange(*new_range, padding=0)
        if "y" in axes:
            plot.setYRange(*new_range, padding=0)


def colour_legend(
    legend: pg.LegendItem,
    text: QColor | str,
    background: QColor | str | None = None,
) -> None:
    """Colours the names of a legend and, if given, its background."""
    if background is not None:
        legend.setBrush(pg.mkBrush(background))
    legend.setLabelTextColor(text)
    # A name takes the colour only when its text is set anew.
    for _sample, label in legend.items:
        label.setText(label.text)


class RollingBitmapPlot:
    """Shows a rolling bitmap (columns = bit position, rows = time) of a bit sequence
    whose row width may change on a service/subchannel change; the history then starts
    afresh instead of mixing rows of different lengths."""

    def __init__(self, title: str, x_label: str, y_label: str, max_rows: int) -> None:
        self.plot = pg.PlotWidget(title=title)
        self.plot.setLabel("bottom", x_label)
        self.plot.setLabel("left", y_label)
        self._image_item = pg.ImageItem()
        self.plot.addItem(self._image_item)
        self._history: deque[np.ndarray] = deque(maxlen=max_rows)
        self._row_width: int | None = None
        self._dirty = False

    def clear(self) -> None:
        self._history.clear()
        self._row_width = None
        self._dirty = False
        self._image_item.clear()

    def update(self, rows: np.ndarray) -> None:
        """Takes new rows but does not draw yet.

        Drawing happens in `render()`, driven by the shared refresh timer (see
        `PLOT_REFRESH_INTERVAL_MS` in the main window). Every row is recorded; only the
        redrawing is batched.
        """
        if self._row_width is not None and rows.shape[1] != self._row_width:
            self._history.clear()
        self._row_width = rows.shape[1]
        self._history.extend(rows)
        self._dirty = True

    def render(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        image = np.array(self._history, dtype=np.uint8)
        self._image_item.setImage(image.T, autoLevels=False, levels=(0, 1))


class MapLabel(pg.TextItem):
    """A name on a map (plain text), above the lines and points drawn on it. A halo in
    the colour of the ground around its letters keeps them legible where a line or a
    point passes beneath and, unlike a filled box, leaves what lies between the letters
    in sight. The default halo suits the black ground of the desktop's plots.

    The halo is a copy of the text beneath it, its letters outlined wide. Painting the
    outlines anew in every frame costs nearly 1 ms per name (with 40 names 40 ms a frame
    instead of 6.5 ms without halo); Qt keeps the halo as a picture instead
    (`DeviceCoordinateCache`, hardly more than without halo), which suits an item that
    moves with the map but keeps its size. After zooming the picture may lie up to a
    pixel off the letters drawn anew."""

    def __init__(
        self,
        text: str = "",
        color: tuple[int, int, int] | QColor | str = (200, 200, 200),
        halo: tuple[int, int, int] | QColor | str = (0, 0, 0),
        font: QFont | None = None,
        anchor: tuple[float, float] = (0, 1),
    ) -> None:
        # TextItem sets the text in its constructor, which the halo follows.
        self._halo = QGraphicsTextItem()
        self._halo_colour = pg.mkColor(halo)
        super().__init__(text, color=color, anchor=anchor)
        self._halo.setParentItem(self)
        self._halo.setZValue(-1)
        self._halo.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.setZValue(MAP_LABEL_Z)
        if font is not None:
            self.setFont(font)

    @property
    def halo(self) -> QColor:
        return QColor(self._halo_colour)

    def set_halo(self, colour: tuple[int, int, int] | QColor | str) -> None:
        colour = pg.mkColor(colour)
        if colour != self._halo_colour:
            self._halo_colour = colour
            self._outline_halo()

    def setPlainText(self, text: str) -> None:
        if text != self._halo.toPlainText():
            self._halo.setPlainText(text)
            self._outline_halo()
        super().setPlainText(text)

    def setFont(self, font: QFont) -> None:
        self._halo.setFont(font)
        super().setFont(font)

    def updateTextPos(self) -> None:
        super().updateTextPos()
        self._halo.setPos(self.textItem.pos())

    def _outline_halo(self) -> None:
        pen = QPen(self._halo_colour, HALO_WIDTH)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        halo = QTextCharFormat()
        halo.setForeground(self._halo_colour)
        halo.setTextOutline(pen)
        cursor = QTextCursor(self._halo.document())
        cursor.select(QTextCursor.SelectionType.Document)
        cursor.mergeCharFormat(halo)


class PathDifferenceAxis(pg.AxisItem):
    """Labels a delay in seconds as a path difference in km."""

    def __init__(self, orientation: str) -> None:
        super().__init__(orientation)
        # Otherwise pyqtgraph puts a factor ("x1e-06") after the label because of the
        # small values in seconds, which does not apply to km.
        self.enableAutoSIPrefix(False)

    def tickStrings(self, values: list[float], scale: float, spacing: float) -> list[str]:
        spacing_km = spacing * SPEED_OF_LIGHT / 1000
        decimals = max(0, -math.floor(math.log10(spacing_km))) if spacing_km > 0 else 0
        return [f"{value * SPEED_OF_LIGHT / 1000:.{decimals}f}" for value in values]
