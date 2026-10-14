"""The traffic flow on a pyqtgraph map: the disturbed stretches of the TFP messages
(`ui.traffic_model.FlowEntry`, from "heavy" on), coloured by their level of service.

Both directions of a road share the points of the TMC location table; each is drawn a
few pixels to the right of its direction of travel, as traffic drives in Germany, so
that one sees which side is slow. The offset is in pixels and so is worked out anew
when the view's scale changes. Levels can be left out (the legend of the touch map):
their stretches are then neither moved nor drawn.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, QTimer

from ui.road_map import pixel_size
from ui.traffic_model import DISTURBED, FLOW_LEVELS, FlowEntry, Key

# Drawing waits this long after the view moved (a pinch draws once at its end).
REDRAW_DELAY_MS = 60


class FlowLayer(QObject):
    """One line item per level; the stretch selected drawn wider beneath them."""

    def __init__(
        self,
        plot: pg.PlotWidget,
        width: float = 4.0,
        offset: float = 4.0,
        z: float = 3.0,
    ) -> None:
        super().__init__(plot)
        self._plot = plot
        self._offset = offset
        # Per level: the stretches as lines of (longitude, latitude), in driving direction.
        self._lines: dict[int, list[np.ndarray]] = {}
        self._selected: list[np.ndarray] = []
        self._hidden: set[int] = set()
        self._scale: tuple[float, float] | None = None
        self._items: dict[int, pg.PlotDataItem] = {}
        for level, (_name, colour) in FLOW_LEVELS.items():
            if level < DISTURBED:
                continue
            item = pg.PlotDataItem(pen=pg.mkPen(*colour, width=width), connect="finite")
            item.setZValue(z + level * 0.01)
            plot.addItem(item, ignoreBounds=True)
            self._items[level] = item
        self._highlight = pg.PlotDataItem(connect="finite")
        self._highlight.setZValue(z - 0.5)
        plot.addItem(self._highlight, ignoreBounds=True)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(REDRAW_DELAY_MS)
        self._timer.timeout.connect(self.redraw)
        plot.getViewBox().sigRangeChanged.connect(self._on_range_changed)

    def set_highlight_pen(self, pen) -> None:
        self._highlight.setPen(pen)

    def set_hidden_levels(self, levels: Iterable[int]) -> None:
        """Leaves the stretches of `levels` out; the stretch selected still stands out."""
        self._hidden = set(levels)
        self.redraw()

    def set_flows(self, flows: list[FlowEntry], selected: Key | None = None) -> None:
        """Shows the disturbed stretches of `flows`; `selected` stands out whole."""
        self._lines = {level: [] for level in self._items}
        self._selected = []
        for flow in flows:
            points = np.array([(p.longitude, p.latitude) for p in flow.points])
            for stretch in flow.stretches:
                if stretch.level in self._lines:
                    # From the far end of the stretch towards the primary point.
                    self._lines[stretch.level].append(points[stretch.start : stretch.end + 1][::-1])
            if flow.key == selected and len(points) > 1:
                self._selected.append(points[::-1])
        self.redraw()

    def count(self) -> int:
        """Disturbed stretches drawn."""
        return sum(len(lines) for level, lines in self._lines.items() if level not in self._hidden)

    def _on_range_changed(self, *_args: object) -> None:
        if self._scale != self._pixel_scale():
            self._timer.start()

    def _pixel_scale(self) -> tuple[float, float]:
        width, height = pixel_size(self._plot.getViewBox())
        return round(width, 9), round(height, 9)

    def redraw(self) -> None:
        self._timer.stop()
        self._scale = self._pixel_scale()
        for level, item in self._items.items():
            x, y = self._joined([] if level in self._hidden else self._lines.get(level, []))
            item.setData(x, y)
        x, y = self._joined(self._selected)
        self._highlight.setData(x, y)

    def _joined(self, lines: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """The lines moved `offset` pixels to the right of their direction, with NaN
        between them: each point along the mean of the normals of the segments it joins.
        All lines at once; one by one, 500 stretches took 18 ms on the PC, held the GIL
        all the while, and came again with every new frame of TPEG."""
        if not lines:
            return np.empty(0), np.empty(0)
        gap = np.full((1, 2), np.nan)
        joined = np.concatenate([part for line in lines for part in (line, gap)])
        width, height = self._scale or (1.0, 1.0)
        if not width or not height:
            return joined[:, 0], joined[:, 1]
        scale = np.array([width, height])
        pixels = joined / scale
        steps = np.diff(pixels, axis=0)
        lengths = np.hypot(steps[:, 0], steps[:, 1])
        lengths[lengths == 0] = 1.0
        # Right of a direction (dx, dy) with y pointing up is (dy, -dx); across a gap
        # between two lines there is none.
        normals = np.nan_to_num(np.column_stack([steps[:, 1], -steps[:, 0]]) / lengths[:, None])
        at_points = np.zeros_like(pixels)
        at_points[1:] += normals
        at_points[:-1] += normals
        norms = np.hypot(at_points[:, 0], at_points[:, 1])
        norms[norms == 0] = 1.0
        at_points /= norms[:, None]
        shifted = (pixels + at_points * self._offset) * scale
        return shifted[:, 0], shifted[:, 1]
