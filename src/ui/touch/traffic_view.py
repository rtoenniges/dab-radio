"""The "Traffic" menu of the services page: what ARD TPEG broadcasts, on four pages to
swipe through (`ui.touch.pager`): the map with the messages and the disturbed traffic
flow on the road network, the messages by distance, the traffic flow by stretch, and
the TPEG stream. Everything comes from `ui.traffic_model.TrafficModel`, which also feeds
the desktop's "Traffic" tab.

On the map one finger turns the page like everywhere; two fingers zoom and move it
(`ui.touch.gestures`), a double tap zooms in, a tap on a symbol shows its message
beneath the map. A tap on a line of the legend hides its layer or shows it again; a
hidden layer is not drawn at all, which keeps a crowded map quick, and the settings
remember it. Names of roads and junctions, and the traffic flow at all, need the
TMC location table (`tools/import_tmc_locations.py`); without it the messages show what
DAB brings and the flow page says how to import it.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF, QSettings, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPixmap
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from dab.road_network import RoadNetwork
from dab.tpeg import AID_SNI, AID_TEC, AID_TFP
from dab.tpeg_tables import application_name
from i18n import tr, tr_noop, tr_plural
from ui.atmosphere_model import CITIES, MAP_EAST, MAP_NORTH, MAP_SOUTH, MAP_WEST
from ui.flow_map import FlowLayer
from ui.plots import MapLabel
from ui.receiver_controller import ReceiverController
from ui.road_map import RoadColours, RoadMap
from ui.tables import fill_table
from ui.touch.gestures import PlotGestures
from ui.touch.pager import Pager
from ui.touch.pages import Page, SwitchedOffNote, pixel_font, rgb, touch_table
from ui.touch.theme import DARK, SMALL_FONT_PX, THEMES, TOUCH_TARGET, Theme, style_plot
from ui.touch.widgets import TapWidget, repolish, set_role
from ui.traffic_model import (
    DISTURBED,
    FLOW_LEVELS,
    SEVERITIES,
    SEVERITY_MARKS,
    FlowEntry,
    Key,
    TrafficEntry,
    TrafficModel,
    describe_tec,
    local,
    short_period,
)

# The pages in the order of swiping.
MAP = "map"
MESSAGES = "messages"
FLOW = "flow"
STREAM = "stream"
PAGES = (MAP, MESSAGES, FLOW, STREAM)

# After new messages the map and the lists are redrawn at most this often (as the
# desktop's tab: a frame brings some 30 messages every two seconds).
REDRAW_SECONDS = 2.0
# The lists hold the nearest (worst) this many; a row costs a few widgets, and farther
# messages are rarely of interest at a radio.
MAX_ROWS = 200
MAX_FLOW_ROWS = 100
# New rows of a list are made for this long at a time (seconds), at least one, then the
# GUI goes on for `BUILD_PAUSE_MS`. Qt polishes the widgets of a row with the style sheet
# as it shows them, 2.5 ms a row at the PC (2026-10-16): the 200 messages made at once
# stopped the swipe that had just turned to their list for half a second, on the Pi for
# longer.
BUILD_SLICE_SECONDS = 0.008
BUILD_PAUSE_MS = 16
MARKER_SIZE = 14
# A tap selects the nearest symbol within this distance (pixels).
TAP_REACH = 32
# A message shown on the map from the list comes into a view at most this wide (degrees).
SHOW_WIDTH = 0.4
# The legend's names are 226 px wide: the German "stockend, zähfließend" (English
# "Queuing, slow") takes 219 px in DejaVu Sans, the widest font likely on the Pi, and so
# keeps to one line.
LEGEND_WIDTH = 260
# A line of the legend; all of it answers a tap. Lower than `TOUCH_TARGET`, so that
# eight of them, their headings and "Reset view" fit beside the map.
LEGEND_ROW_HEIGHT = 50
PROFILE_HEIGHT = 96
# Where the settings keep the layers hidden in the legend (their names, separated by
# spaces).
SETTINGS_HIDDEN = "touch/traffic_hidden"
# The map keeps to Germany with a margin and zooms in to a few hundred metres. Across,
# the wide map shows far more than Germany when it shows all of it from north to south.
_LIMITS = {
    "xMin": MAP_WEST - 12,
    "xMax": MAP_EAST + 12,
    "yMin": MAP_SOUTH - 2,
    "yMax": MAP_NORTH + 2,
    "minYRange": 0.012,
    "maxYRange": MAP_NORTH - MAP_SOUTH + 4,
}
# The flow in the legend: levels drawn alike share a line, and a layer. The names are
# translated where they are shown.
_FLOW_LEGEND = (
    (tr_noop("Heavy traffic"), (2,)),
    (tr_noop("Queuing, slow"), (3, 4)),
    (tr_noop("Stationary traffic"), (5,)),
    (tr_noop("Road closed"), (6,)),
)
_FLOW_LAYERS = {level: f"flow{levels[0]}" for _name, levels in _FLOW_LEGEND for level in levels}
# The symbols of the messages while their layer is hidden.
_HOLLOW_MARKS = ("□", "▽", "◇", "○")
_NO_TPEG = tr_noop("No TPEG in the ensemble.")
_NO_LOCATIONS = tr_noop(
    "The traffic flow names its stretches only as TMC locations. To show it, import the "
    "TMC location table (LCL) once, as the responsible authority of the country "
    "publishes it (in Germany the BASt):\n"
    "uv run python tools/import_tmc_locations.py <LCL download>"
)


def _mark(severity: int) -> str:
    """The symbol of a severity in its colour (rich text)."""
    red, green, blue = SEVERITIES[severity][1]
    return f'<span style="color: rgb({red},{green},{blue})">{SEVERITY_MARKS[severity]}</span>'


def _flow_mark(level: int) -> str:
    """A bar in the colour of a level of service (rich text)."""
    red, green, blue = FLOW_LEVELS.get(level, FLOW_LEVELS[0])[1]
    return f'<span style="color: rgb({red},{green},{blue})">▬</span>'


def road_colours(theme: Theme) -> RoadColours:
    """The road network in the colours of the scheme, behind the coloured messages."""
    surface = pg.mkColor(theme.surface)
    muted = pg.mkColor(theme.muted)

    def mix(share: float) -> tuple[int, int, int]:
        return tuple(
            round(a * share + b * (1 - share))
            for a, b in zip(muted.getRgb()[:3], surface.getRgb()[:3])
        )

    coast = (64, 120, 160) if theme.key == DARK else (110, 165, 205)
    return RoadColours(
        motorway=mix(0.85),
        major=mix(0.6),
        minor=mix(0.35),
        border=mix(0.7),
        coast=coast,
        text=rgb(theme.muted),
        ground=rgb(theme.surface),
    )


def _severity_layer(severity: int) -> str:
    return f"severity{severity}"


def _heading(text: str) -> QLabel:
    label = set_role(QLabel(text), "heading")
    label.setContentsMargins(4, 6, 0, 0)
    return label


class _LegendEntry(TapWidget):
    """A line of the legend that hides its layer of the map with a tap, and shows it
    with the next: shown, the symbol in its colour; hidden, the symbol hollow and the
    line grey. It lights up while pressed."""

    toggled = pyqtSignal(bool)  # shown

    def __init__(self, name: str, mark: str, hollow: str, colour: tuple[int, int, int]) -> None:
        super().__init__()
        set_role(self, "legend")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setMinimumHeight(LEGEND_ROW_HEIGHT)
        self.shown = True
        self._marks = (mark, hollow)
        self._colour = colour
        self._theme = THEMES[DARK]
        self.mark = QLabel()
        self.mark.setFixedWidth(22)
        self.mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name = QLabel(name)
        self.name.setWordWrap(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 0, 2)
        layout.setSpacing(8)
        layout.addWidget(self.mark)
        layout.addWidget(self.name, stretch=1)
        self.tapped.connect(lambda: self.set_shown(not self.shown))
        self._show_state()

    def set_shown(self, shown: bool) -> None:
        if shown == self.shown:
            return
        self.shown = shown
        self._show_state()
        self.toggled.emit(shown)

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        self._show_state()

    def _show_state(self) -> None:
        colour = "rgb({},{},{})".format(*self._colour) if self.shown else self._theme.muted
        mark = self._marks[0 if self.shown else 1]
        self.mark.setText(f'<span style="color: {colour}">{mark}</span>')
        set_role(self.name, "" if self.shown else "muted")
        repolish(self.name)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._set_pressed(True)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._set_pressed(False)
        super().mouseReleaseEvent(event)

    def _set_pressed(self, pressed: bool) -> None:
        self.setProperty("pressed", pressed)
        repolish(self)


class _MapPage(Page):
    """The map with the messages, the disturbed flow and the network; beside it the
    legend, whose lines hide and show their layers, and "Reset view"; the message
    tapped on beneath it."""

    more_requested = pyqtSignal(object)  # Key: show it in the list

    def __init__(self, model: TrafficModel, settings: QSettings | None = None) -> None:
        super().__init__(tr("Map"))
        self._model = model
        self._settings = settings
        self._dirty = True
        self._drawn_at: float | None = None
        self.selected: Key | None = None
        self.selected_flow: Key | None = None
        self._positions = np.zeros((0, 2))  # of the symbols drawn, with their keys
        self._keys: list[Key] = []
        self._placed = 0  # messages with a place, also those hidden

        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True, ratio=float(np.cos(np.radians(51.2))))
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        view_box = self.plot.getViewBox()
        view_box.setLimits(**_LIMITS)
        self.roads = RoadMap(self.plot, font=pixel_font(SMALL_FONT_PX), width_scale=1.4)
        self.flow = FlowLayer(self.plot, width=5.0, offset=5.0)
        # Without the road network a few cities show where the messages are, as on the
        # desktop's map.
        self._cities = pg.ScatterPlotItem(
            [lon for _name, _lat, lon in CITIES], [lat for _name, lat, _lon in CITIES], size=5
        )
        self.plot.addItem(self._cities)
        self._city_labels = []
        for name, latitude, longitude in CITIES:
            label = MapLabel(name, font=pixel_font(SMALL_FONT_PX))
            label.setPos(longitude, latitude)
            self.plot.addItem(label)
            self._city_labels.append(label)
        self._lines = []
        self._markers = []
        for _name, colour, symbol in SEVERITIES:
            line = pg.PlotDataItem(pen=pg.mkPen(*colour, width=4), connect="finite")
            line.setZValue(5)
            self.plot.addItem(line)
            self._lines.append(line)
            markers = pg.ScatterPlotItem(
                pxMode=True, symbol=symbol, size=MARKER_SIZE, brush=pg.mkBrush(*colour)
            )
            markers.setZValue(10)
            self.plot.addItem(markers)
            self._markers.append(markers)
        self._selected_line = pg.PlotDataItem(connect="finite")
        self._selected_line.setZValue(4)
        self.plot.addItem(self._selected_line)
        self._selected_ring = pg.ScatterPlotItem(pxMode=True, symbol="o", size=MARKER_SIZE + 14)
        self._selected_ring.setZValue(9)
        self.plot.addItem(self._selected_ring)
        self._observer = pg.ScatterPlotItem(pxMode=True, symbol="+", size=26)
        self._observer.setZValue(20)
        self.plot.addItem(self._observer)
        self.gestures = PlotGestures(self.plot)
        self.gestures.tapped.connect(self._on_tapped)
        self.gestures.double_tapped.connect(lambda point: self.gestures.zoom(0.5, point))
        # While two fingers move the picture of the map, the names of the towns stay
        # crisp and at their size on it.
        self.gestures.hidden_in_snapshot = [self.roads.labels]
        self.gestures.overlay = self.roads.labels.paint_at

        self.card = set_role(QWidget(), "panel")
        self.card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.card_text = QLabel()
        self.card_text.setTextFormat(Qt.TextFormat.RichText)
        self.more = QPushButton(tr("More ›"))
        self.more.clicked.connect(lambda: self.more_requested.emit(self.selected))
        # As high with the button as without: the map above must not jump.
        self.card.setFixedHeight(TOUCH_TARGET + 12)
        card_layout = QHBoxLayout(self.card)
        card_layout.setContentsMargins(16, 6, 8, 6)
        card_layout.addWidget(self.card_text, stretch=1)
        card_layout.addWidget(self.more)

        # The lines of the legend by the names of their layers.
        self.legend: dict[str, _LegendEntry] = {}
        legend = QVBoxLayout()
        legend.setContentsMargins(0, 0, 0, 0)
        legend.setSpacing(0)
        legend.addWidget(_heading(tr("Messages")))
        for severity, (name, colour, _symbol) in enumerate(SEVERITIES):
            marks = SEVERITY_MARKS[severity], _HOLLOW_MARKS[severity]
            legend.addWidget(self._add_entry(_severity_layer(severity), tr(name), *marks, colour))
        # The flow only where the location table places it.
        self._flow_legend = set_role(QWidget(), "transparent")
        flow_legend = QVBoxLayout(self._flow_legend)
        flow_legend.setContentsMargins(0, 10, 0, 0)
        flow_legend.setSpacing(0)
        flow_legend.addWidget(_heading(tr("Traffic flow")))
        for name, levels in _FLOW_LEGEND:
            colour = FLOW_LEVELS[levels[0]][1]
            entry = self._add_entry(_FLOW_LAYERS[levels[0]], tr(name), "▬", "▭", colour)
            flow_legend.addWidget(entry)
        self._flow_legend.hide()
        legend.addWidget(self._flow_legend)
        legend.addStretch()
        self.reset_button = QPushButton(tr("Reset view"))
        self.reset_button.clicked.connect(self.reset_view)
        legend.addWidget(self.reset_button)
        side = set_role(QWidget(), "transparent")
        side.setFixedWidth(LEGEND_WIDTH)
        side.setLayout(legend)

        left = QVBoxLayout()
        left.setSpacing(8)
        left.addWidget(self.plot, stretch=1)
        left.addWidget(self.card)
        content = QHBoxLayout()
        content.setSpacing(16)
        content.addLayout(left, stretch=1)
        content.addWidget(side)
        self.body.addLayout(content, stretch=1)

        model.messages_changed.connect(self._on_changed)
        model.roads_changed.connect(self._on_roads_changed)
        model.locations_changed.connect(self._on_changed)
        hidden = (
            set() if settings is None else set(str(settings.value(SETTINGS_HIDDEN, "")).split())
        )
        for layer, entry in self.legend.items():
            entry.set_shown(layer not in hidden)
            entry.toggled.connect(self._store_hidden)
        for severity in range(len(SEVERITIES)):
            self.legend[_severity_layer(severity)].toggled.connect(self._on_messages_toggled)
        for layer in set(_FLOW_LAYERS.values()):
            self.legend[layer].toggled.connect(self._on_flow_toggled)
        self.flow.set_hidden_levels(self._hidden_levels())
        self.reset_view()
        self._show_card()

    def _add_entry(
        self, layer: str, name: str, mark: str, hollow: str, colour: tuple[int, int, int]
    ) -> _LegendEntry:
        entry = _LegendEntry(name, mark, hollow, colour)
        self.legend[layer] = entry
        return entry

    def _hidden_levels(self) -> set[int]:
        return {level for level, layer in _FLOW_LAYERS.items() if not self.legend[layer].shown}

    def _store_hidden(self) -> None:
        if self._settings is not None:
            hidden = sorted(layer for layer, entry in self.legend.items() if not entry.shown)
            self._settings.setValue(SETTINGS_HIDDEN, " ".join(hidden))

    def _on_messages_toggled(self) -> None:
        self._draw_messages()
        self._draw_selection()
        self._show_card()

    def _on_flow_toggled(self) -> None:
        self.flow.set_hidden_levels(self._hidden_levels())

    def reset_view(self) -> None:
        """All of Germany."""
        # In one call: with the aspect locked, the view widens to show both ranges.
        self.plot.getViewBox().setRange(
            xRange=(MAP_WEST, MAP_EAST), yRange=(MAP_SOUTH, MAP_NORTH), padding=0
        )

    def apply_theme(self, theme: Theme) -> None:
        style_plot(self.plot, theme)
        self.roads.set_colours(road_colours(theme))
        outline = pg.mkPen(theme.surface, width=2)
        for markers in self._markers:
            markers.setPen(outline)
        self._selected_line.setPen(pg.mkPen(theme.accent, width=10))
        self.flow.set_highlight_pen(pg.mkPen(theme.accent, width=12))
        self._selected_ring.setPen(pg.mkPen(theme.accent, width=3))
        self._selected_ring.setBrush(pg.mkBrush(None))
        self._observer.setPen(pg.mkPen(theme.text, width=3))
        self._cities.setBrush(pg.mkBrush(theme.muted))
        self._cities.setPen(pg.mkPen(None))
        for label in self._city_labels:
            label.setColor(theme.muted)
            label.set_halo(theme.surface)
        for entry in self.legend.values():
            entry.set_theme(theme)
        self._dirty = True

    def _on_changed(self) -> None:
        self._dirty = True

    def _on_roads_changed(self) -> None:
        self._show_network(self._model.roads)

    def _show_network(self, network: RoadNetwork | None) -> None:
        self.roads.set_network(network)
        for item in (self._cities, *self._city_labels):
            item.setVisible(network is None)

    def select(self, key: Key | None) -> None:
        self.selected = key
        self._draw_selection()
        self._show_card()

    def show_entry(self, key: Key) -> None:
        """Selects the message and brings it into view, close enough to see its road;
        its layer is shown if it was hidden."""
        entry = self._model.entry(key)
        if entry is not None:
            self.legend[_severity_layer(entry.severity)].set_shown(True)
        self.select(key)
        if entry is None or not entry.path:
            return
        lon, lat = np.mean(np.array(entry.path), axis=0)
        (x0, x1), (y0, y1) = self.plot.getViewBox().viewRange()
        width = min(x1 - x0, SHOW_WIDTH)
        height = width * (y1 - y0) / (x1 - x0)
        self.plot.setXRange(lon - width / 2, lon + width / 2, padding=0)
        self.plot.setYRange(lat - height / 2, lat + height / 2, padding=0)

    def show_flow(self, key: Key) -> None:
        """Marks a stretch of the traffic flow and brings all of it into view, with the
        layers of its levels shown."""
        self.selected_flow = key
        flow = self._model.flow(key)
        if flow is not None:
            for stretch in flow.stretches:
                if stretch.level in _FLOW_LAYERS:
                    self.legend[_FLOW_LAYERS[stretch.level]].set_shown(True)
        self.flow.set_flows(self._model.flows(), key)
        if flow is None:
            return
        points = np.array(flow.path())
        west, south = points.min(axis=0)
        east, north = points.max(axis=0)
        margin = max(east - west, north - south, 0.05) * 0.15
        self.plot.getViewBox().setRange(
            xRange=(west - margin, east + margin), yRange=(south - margin, north + margin)
        )

    def render(self) -> None:
        model = self._model
        if model.roads is not None and self.roads.network is None:
            self._show_network(model.roads)
        have, _announced = model.counts(AID_TEC)
        if have:
            text = tr_plural(
                "{count} message, {placed} with a location",
                "{count} messages, {placed} with a location",
                have,
            ).format(count=have, placed=self._placed)
        else:
            text = _status(model)
        if model.locations is not None and have:
            disturbed = sum(1 for f in model.flows() if f.worst >= DISTURBED)
            text += " · " + tr_plural(
                "{count} stretch disturbed", "{count} stretches disturbed", disturbed
            ).format(count=disturbed)
        if model.roads is None:
            text += " · " + tr("road network missing")
        self.status.setText(text + _held_since(model))
        self._flow_legend.setVisible(model.locations is not None)
        now = time.monotonic()
        due = self._dirty and (self._drawn_at is None or now - self._drawn_at >= REDRAW_SECONDS)
        if not due:
            return
        self._drawn_at = now
        self._dirty = False
        self._draw()

    def _draw(self) -> None:
        self._draw_messages()
        observer = self._model.observer
        self._observer.setData([observer.longitude], [observer.latitude])
        self.flow.set_flows(self._model.flows(), self.selected_flow)
        self._draw_selection()
        self._show_card()

    def _draw_messages(self) -> None:
        """The messages with a place, those of hidden layers left out (and unselected)."""
        located = [e for e in self._model.entries() if e.path]
        self._placed = len(located)
        shown = [self.legend[_severity_layer(s)].shown for s in range(len(SEVERITIES))]
        entries = [e for e in located if shown[e.severity]]
        lines = [([], []) for _ in SEVERITIES]
        markers = [([], []) for _ in SEVERITIES]
        for entry in entries:
            xs, ys = lines[entry.severity]
            xs.extend([p[0] for p in entry.path] + [np.nan])
            ys.extend([p[1] for p in entry.path] + [np.nan])
            mx, my = markers[entry.severity]
            # The symbol at the start of the line, as on the desktop's map.
            mx.append(entry.path[0][0])
            my.append(entry.path[0][1])
        for item, (xs, ys) in zip(self._lines, lines):
            item.setData(xs, ys)
        for item, (mx, my) in zip(self._markers, markers):
            item.setData(x=mx, y=my)
        self._keys = [e.key for e in entries]
        self._positions = np.array([e.path[0] for e in entries]).reshape(-1, 2)
        if self.selected is not None:
            selected = self._model.entry(self.selected)
            if selected is None or not shown[selected.severity]:
                self.selected = None

    def _draw_selection(self) -> None:
        entry = None if self.selected is None else self._model.entry(self.selected)
        path = () if entry is None else entry.path
        self._selected_line.setData([p[0] for p in path], [p[1] for p in path])
        if path:
            self._selected_ring.setData([path[0][0]], [path[0][1]])
        else:
            self._selected_ring.setData([], [])

    def _show_card(self) -> None:
        entry = None if self.selected is None else self._model.entry(self.selected)
        self.more.setVisible(entry is not None)
        if entry is None:
            self.card_text.setText(tr("Tap a symbol to see its message."))
            return
        parts = [f"{_mark(entry.severity)} {entry.headline()}"]
        where = entry.short_where() or entry.road_kind()
        if where:
            parts.append(where)
        parts.append(entry.distance_text())
        self.card_text.setText(" · ".join(parts))

    def _on_tapped(self, point: QPointF) -> None:
        """Selects the nearest symbol within `TAP_REACH`, or none."""
        if not len(self._positions):
            self.select(None)
            return
        width, height = self.plot.getViewBox().viewPixelSize()
        dx = (self._positions[:, 0] - point.x()) / width
        dy = (self._positions[:, 1] - point.y()) / height
        distances = np.hypot(dx, dy)
        index = int(np.argmin(distances))
        self.select(self._keys[index] if distances[index] <= TAP_REACH else None)


class _Row(TapWidget):
    """A row of a list that opens with a tap: a symbol, a title and a figure on the
    right in its first line, a line beneath, and what opens."""

    opened = pyqtSignal(object)  # Key
    map_requested = pyqtSignal(object)  # Key

    def __init__(self, key: Key, opening: QWidget) -> None:
        super().__init__()
        set_role(self, "programme")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.key = key
        self.mark = set_role(QLabel(), "large")
        self.mark.setFixedWidth(32)
        self.title = set_role(QLabel(), "large")
        self.title.setWordWrap(True)
        self.title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.figure = set_role(QLabel(), "muted")
        self.figure.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        self.below = set_role(QLabel(), "muted")
        self.map_button = QPushButton(tr("Show on the map"))
        self.map_button.clicked.connect(lambda: self.map_requested.emit(self.key))
        self._open_part = set_role(QWidget(), "transparent")
        open_layout = QVBoxLayout(self._open_part)
        open_layout.setContentsMargins(0, 6, 0, 0)
        open_layout.addWidget(opening)
        open_layout.addWidget(self.map_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self._open_part.hide()
        layout = QGridLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(2)
        layout.addWidget(self.mark, 0, 0, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.title, 0, 1)
        layout.addWidget(self.figure, 0, 2)
        layout.addWidget(self.below, 1, 1, 1, 2)
        layout.addWidget(self._open_part, 2, 1, 1, 2)
        layout.setColumnStretch(1, 1)
        self._shown: tuple | None = None
        self.tapped.connect(self._toggle)

    @property
    def is_open(self) -> bool:
        return self._open_part.isVisibleTo(self)

    def set_open(self, open_: bool) -> None:
        self._open_part.setVisible(open_)

    def _toggle(self) -> None:
        opening = not self.is_open
        self.set_open(opening)
        if opening:
            self.opened.emit(self.key)


class _MessageRow(_Row):
    """Symbol, effect and cause, distance and period; the place beneath; the message in
    words opens with a tap."""

    def __init__(self, key: Key) -> None:
        self._details = QLabel()
        self._details.setTextFormat(Qt.TextFormat.RichText)
        self._details.setWordWrap(True)
        self._details.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        super().__init__(key, self._details)

    def show_entry(self, entry: TrafficEntry, now: float) -> None:
        period = short_period(entry.message, now)
        distance = entry.distance_text() + (f" · {period}" if period else "")
        place = entry.where() or entry.road_kind() or ("" if entry.path else tr("no location"))
        shown = (entry.stored.received, entry.severity, place, distance)
        if shown == self._shown:
            return
        self._shown = shown
        self.mark.setText(_mark(entry.severity))
        self.title.setText(entry.headline())
        self.figure.setText(distance)
        self.below.setText(place)
        self.below.setVisible(bool(place))
        self._details.setText("<br>".join(describe_tec(entry.message)[1:]))
        self.map_button.setVisible(bool(entry.path))


class _Profile(QWidget):
    """The speed of a stretch section by section, in the direction of travel from left
    to right, coloured by the level of service; the first and the last point beneath."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(PROFILE_HEIGHT)
        self._flow: FlowEntry | None = None
        self._text = QColor(Qt.GlobalColor.gray)
        self._line = QColor(Qt.GlobalColor.darkGray)

    def set_theme(self, theme: Theme) -> None:
        self._text = theme.color("muted")
        self._line = theme.color("outline")
        self.update()

    def show_flow(self, flow: FlowEntry) -> None:
        self._flow = flow
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        flow = self._flow
        if flow is None or len(flow.points) < 2:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = self.width()
        text_height = SMALL_FONT_PX + 6
        base = self.height() - text_height
        last = len(flow.points) - 1
        top = max([130] + [s.speed for s in flow.stretches if s.speed])
        for stretch in flow.stretches:
            # The primary point (0) lies at the head, on the right.
            left = (last - stretch.end) / last * width
            right = (last - stretch.start) / last * width
            height = (stretch.speed or 0) / top * (base - 4) if stretch.speed else 4
            painter.fillRect(
                QRectF(left + 1, base - height, max(right - left - 2, 1), height),
                QColor(*FLOW_LEVELS.get(stretch.level, FLOW_LEVELS[0])[1]),
            )
        painter.setPen(self._line)
        painter.drawLine(0, round(base), width, round(base))
        painter.setPen(self._text)
        painter.setFont(pixel_font(SMALL_FONT_PX))
        below = QRectF(0, base + 2, width, text_height)
        painter.drawText(below, Qt.AlignmentFlag.AlignLeft, flow.points[-1].label())
        painter.drawText(below, Qt.AlignmentFlag.AlignRight, flow.points[0].label())
        painter.end()


class _FlowRow(_Row):
    """The worst level as a bar, the stretch, level and speeds; the distance beneath;
    the speed profile opens with a tap."""

    def __init__(self, key: Key) -> None:
        self.profile = _Profile()
        super().__init__(key, self.profile)

    def show_flow(self, flow: FlowEntry) -> None:
        figure = tr(FLOW_LEVELS.get(flow.worst, FLOW_LEVELS[0])[0])
        if flow.speeds():
            figure += f" · {flow.speeds()}"
        shown = (flow.stored.received, flow.worst, figure, flow.distance_text())
        if shown == self._shown:
            return
        self._shown = shown
        self.mark.setText(_flow_mark(flow.worst))
        self.title.setText(flow.where())
        self.figure.setText(figure)
        self.below.setText(flow.distance_text())
        self.profile.show_flow(flow)


class _RowList(QScrollArea):
    """Rows by key in a given order, reused from one filling to the next; one open at a
    time. The pager scrolls it.

    A filling takes slices of `BUILD_SLICE_SECONDS`, the first rows first, so that new
    rows come in a few at a time while the GUI goes on (a swipe, the map); `complete`
    tells when they are all there.

    While a row is open the pages leave the list alone (`holding`), and the filling
    waits (`paused`): rows coming in above it, or moving with a new order, would push it
    away while it is read."""

    def __init__(self, make_row: Callable[[Key], _Row]) -> None:
        super().__init__()
        self._make_row = make_row
        self.rows: dict[Key, _Row] = {}
        self.order: list[Key] = []  # the rows in the layout, in their order
        self._opened: Key | None = None  # the row `open` scrolls to
        # The filling: the keys, how to show a row, the text beneath, the next index.
        self._keys: list[Key] = []
        self._show: Callable[[_Row, int], None] = lambda row, index: None
        self._rest_text = ""
        self._next = 0
        self._building = QTimer(self)
        self._building.setSingleShot(True)
        self._building.setInterval(BUILD_PAUSE_MS)
        self._building.timeout.connect(self._go_on)
        self._list = set_role(QWidget(), "transparent")
        self._layout = QVBoxLayout(self._list)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self.rest = set_role(QLabel(), "muted")
        self.rest.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rest.hide()
        # The rows go before these two.
        self._layout.addWidget(self.rest)
        self._layout.addStretch()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWidget(self._list)

    def fill(self, keys: list[Key], show: Callable[[_Row, int], None], rest: str) -> None:
        """Rows for `keys` in their order; `show(row, index)` fills each; `rest` beneath
        once all are there. The rows of other keys go at once, the first slice of the
        filling is done on return."""
        for key in set(self.rows) - set(keys):
            row = self.rows.pop(key)
            self._layout.removeWidget(row)
            row.deleteLater()
        self._keys = list(keys)
        self._show = show
        self._rest_text = rest
        self._next = 0
        self._arrange()
        self._build()

    @property
    def complete(self) -> bool:
        """The last filling is done."""
        return self._next >= len(self._keys)

    @property
    def paused(self) -> bool:
        """The last filling waits for the open row to close; filling anew goes on."""
        return not self.complete and not self._building.isActive()

    def _arrange(self) -> None:
        """The rows there in the order of the keys, moving only those out of place."""
        order = [key for key in self._keys if key in self.rows]
        placed = [key for key in self.order if key in self.rows]
        for index, key in enumerate(order):
            if placed[index] == key:
                continue
            placed.remove(key)
            placed.insert(index, key)
            row = self.rows[key]
            self._layout.removeWidget(row)
            self._layout.insertWidget(index, row)
        self.order = order

    def _go_on(self) -> None:
        if not self.holding:
            self._build()

    def _build(self, through: int = -1) -> None:
        """Shows rows from `_next` on, making those missing, for a slice of time and at
        least through index `through`; goes on after a pause while some are left. Qt
        polishes a row as it is shown, which is most of its cost, so that is done in
        the slice too.

        Showing a widget lays out its visible parents at once, here the whole list for
        each row (1.4 ms a row at the start, 4.3 ms at 200 rows, PC): while the slice
        runs the layout of the list is off, and it is laid out once at its end, so that
        the scroll area knows its height as it did before.

        The slices go on while the list is hidden (another page of the window or menu):
        then its layout is left to Qt, which lays it out as it is shown. Laid out
        hidden, the layout counts as done, and the scroll area, which hears of a new
        height only while it is visible, kept the old one: the rows squeezed into it
        until one was opened."""
        deadline = time.perf_counter() + BUILD_SLICE_SECONDS
        self._layout.setEnabled(False)
        try:
            while self._next < len(self._keys):
                index = self._next
                key = self._keys[index]
                row = self.rows.get(key)
                new = row is None
                if new:
                    row = self._make_row(key)
                    row.opened.connect(self.close_others)
                    self.rows[key] = row
                self._show(row, index)
                if new:
                    # The keys before it all have their rows: it goes at its index.
                    self.order.insert(index, key)
                    self._layout.insertWidget(index, row)
                    row.show()
                self._next += 1
                if index >= through and time.perf_counter() >= deadline:
                    break
        finally:
            self._layout.setEnabled(True)
        if self._list.isVisible():
            self._layout.activate()
        complete = self.complete
        self.rest.setText(self._rest_text if complete else "")
        self.rest.setVisible(complete and bool(self._rest_text))
        if complete:
            self._building.stop()
        else:
            self._building.start()

    @property
    def holding(self) -> bool:
        """A row is open: the list is not to be filled anew until it is closed."""
        return any(row.is_open for row in self.rows.values())

    def close_others(self, key: Key) -> None:
        for other, row in self.rows.items():
            if other != key and row.is_open:
                row.set_open(False)

    def open(self, key: Key) -> None:
        """Opens a row (closing the others) and scrolls it into view; one the filling
        has not reached yet is made first, with those before it."""
        if key not in self.rows and key in self._keys:
            self._build(through=self._keys.index(key))
        row = self.rows.get(key)
        if row is None:
            return
        row.set_open(True)
        self.close_others(key)
        self._opened = key
        # A bound method: PyQt drops it if the list is gone by then.
        QTimer.singleShot(0, self._scroll_to_opened)

    def _scroll_to_opened(self) -> None:
        row = self.rows.get(self._opened)
        if row is not None:
            self.ensureWidgetVisible(row, 0, 40)


class _MessagesPage(Page):
    """The messages by distance from the listener, one open at a time."""

    map_requested = pyqtSignal(object)  # Key

    def __init__(self, model: TrafficModel) -> None:
        super().__init__(tr("Messages"))
        self._model = model
        self._dirty = True
        self._drawn_at: float | None = None
        self.list = _RowList(self._make_row)
        self.body.addWidget(self.list, stretch=1)
        model.messages_changed.connect(self._on_changed)
        model.locations_changed.connect(self._on_changed)

    @property
    def rows(self) -> dict[Key, _Row]:
        return self.list.rows

    @property
    def order(self) -> list[Key]:
        return self.list.order

    def _make_row(self, key: Key) -> _Row:
        row = _MessageRow(key)
        row.map_requested.connect(self.map_requested)
        return row

    def _on_changed(self) -> None:
        self._dirty = True

    def render(self) -> None:
        model = self._model
        entries = model.entries()
        holding = self.list.holding
        news = self._dirty or self.list.paused
        if not entries:
            self.status.setText(_status(model))
        elif holding and news:
            # Translators: the list of messages waits with news while one is read.
            self.status.setText(tr("List paused while a message is open"))
        else:
            self.status.setText(tr("by distance from the location") + _held_since(model))
        now = time.monotonic()
        due = news and (self._drawn_at is None or now - self._drawn_at >= REDRAW_SECONDS)
        if not due or holding:
            return
        self._drawn_at = now
        self._dirty = False
        shown = entries[:MAX_ROWS]
        rest = len(entries) - len(shown)
        now = model.now()
        more = tr_plural(
            "… and {count} more, farther away", "… and {count} more, farther away", rest
        )
        self.list.fill(
            [e.key for e in shown],
            lambda row, index: row.show_entry(shown[index], now),
            more.format(count=rest) if rest else "",
        )

    def open_entry(self, key: Key) -> None:
        """Opens the message (closing the others) and scrolls it into view."""
        self.list.close_others(key)
        self._dirty = True
        self._drawn_at = None
        self.render()
        self.list.open(key)


class _FlowPage(Page):
    """The disturbed stretches of the traffic flow, the worst first, one open at a time;
    without the TMC location table a note how to import it."""

    map_requested = pyqtSignal(object)  # Key

    def __init__(self, model: TrafficModel) -> None:
        super().__init__(tr("Traffic flow"))
        self._model = model
        self._dirty = True
        self._drawn_at: float | None = None
        self._theme = THEMES[DARK]
        self.note = set_role(QLabel(tr(_NO_LOCATIONS)), "muted")
        self.note.setWordWrap(True)
        self.note.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.list = _RowList(self._make_row)
        self.body.addWidget(self.note)
        self.body.addWidget(self.list, stretch=1)
        model.messages_changed.connect(self._on_changed)
        model.locations_changed.connect(self._on_changed)

    @property
    def rows(self) -> dict[Key, _Row]:
        return self.list.rows

    def _make_row(self, key: Key) -> _Row:
        row = _FlowRow(key)
        row.profile.set_theme(self._theme)
        row.map_requested.connect(self.map_requested)
        return row

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        for row in self.list.rows.values():
            row.profile.set_theme(theme)

    def _on_changed(self) -> None:
        self._dirty = True

    def render(self) -> None:
        model = self._model
        located = model.locations is not None
        self.note.setVisible(not located)
        self.list.setVisible(located)
        flows = model.flows()
        disturbed = [f for f in flows if f.worst >= DISTURBED]
        holding = self.list.holding
        news = self._dirty or self.list.paused
        if not located:
            self.status.setText(tr("TMC location table missing"))
        elif holding and news:
            self.status.setText(tr("List paused while a stretch is open"))
        elif flows:
            free = len(flows) - len(disturbed)
            text = tr_plural(
                "{count} stretch disturbed, {free} free",
                "{count} stretches disturbed, {free} free",
                len(disturbed),
            ).format(count=len(disturbed), free=free)
            self.status.setText(text + _held_since(model))
        else:
            self.status.setText(_status(model))
        now = time.monotonic()
        due = news and (self._drawn_at is None or now - self._drawn_at >= REDRAW_SECONDS)
        if not due or not located or holding:
            return
        self._drawn_at = now
        self._dirty = False
        shown = disturbed[:MAX_FLOW_ROWS]
        rest = len(disturbed) - len(shown)
        more = tr_plural("… and {count} more", "… and {count} more", rest)
        self.list.fill(
            [f.key for f in shown],
            lambda row, index: row.show_flow(shown[index]),
            more.format(count=rest) if rest else "",
        )


class _StreamPage(Page):
    """The service with its logo, its applications, and how the stream was checked."""

    def __init__(self, model: TrafficModel) -> None:
        super().__init__(tr("TPEG stream"))
        self._model = model
        self._dirty = True
        self._logo_data: bytes | None = None
        # Logos are made for a light ground (ARD's is dark blue).
        self.logo = set_role(QLabel(), "logo")
        self.logo.hide()
        self.name = QLabel()
        self.name.setWordWrap(True)
        self.name.setTextFormat(Qt.TextFormat.PlainText)
        header = QHBoxLayout()
        header.setSpacing(16)
        header.addWidget(self.logo)
        header.addWidget(self.name, stretch=1)
        self.table = touch_table(
            [
                tr("Application"),
                tr("Version"),
                tr("Messages"),
                tr("Frames"),
                # Translators: the amount of data a component of the TPEG stream brought.
                tr("Data"),
                tr("Share"),
                tr("CRC errors"),
            ],
            stretch=0,
        )
        self.check = set_role(QLabel(), "muted")
        self.check.setWordWrap(True)
        self.body.addLayout(header)
        self.body.addWidget(self.table, stretch=1)
        self.body.addWidget(self.check)
        model.service_changed.connect(self._on_changed)
        model.decoding_changed.connect(self._on_changed)

    def _on_changed(self) -> None:
        self._dirty = True

    def render(self) -> None:
        model = self._model
        self.status.setText(_status(model))
        if not self._dirty:
            return
        self._dirty = False
        service = model.service
        statistics = model.statistics
        name = ""
        if service is not None and service.name:
            name = service.name + (f" – {service.description}" if service.description else "")
        self.name.setText(name)
        self._show_logo(None if service is None else service.logo)
        rows = []
        if service is not None:
            received = {} if statistics is None else statistics.components
            # The share is of all component frames, so the SNI (logo included) has a row
            # too, although the fast tuning table does not list it.
            total = sum(c.bytes for c in received.values())
            applications = [(0, AID_SNI)] + [(e.scid, e.application) for e in service.components]
            for scid, application in applications:
                version = service.versions.get(scid)
                if application in (AID_TEC, AID_TFP):
                    have, announced = model.counts(application)
                    messages = (
                        tr("{have} of {announced}").format(have=have, announced=announced)
                        if announced
                        else str(have)
                    )
                else:
                    announced = service.message_counts.get(scid)
                    messages = "-" if announced is None else str(announced)
                component = received.get(scid)
                share = None if component is None or not total else component.bytes / total
                rows.append(
                    (
                        application_name(application),
                        "-" if version is None else f"{version[0]}.{version[1]}",
                        messages,
                        "-" if component is None else str(component.frames),
                        "-" if component is None else f"{component.bytes / 1000:.0f} kB",
                        "-" if share is None else f"{share * 100:.0f} %",
                        "-" if component is None else str(component.crc_errors),
                    )
                )
        fill_table(self.table, rows, right_aligned=range(1, 7))
        self.check.setText(self._describe_check())

    def _describe_check(self) -> str:
        statistics = self._model.statistics
        if statistics is None:
            return ""
        frames = statistics.stream_directories + statistics.service_frames
        errors = sum(c.crc_errors for c in statistics.components.values())
        if statistics.lost_data_groups:
            groups = tr("Data groups {groups}, {lost} lost").format(
                groups=statistics.data_groups, lost=statistics.lost_data_groups
            )
        else:
            groups = tr("Data groups {groups}").format(groups=statistics.data_groups)
        parts = [
            groups,
            tr(
                "Transport frames {frames} ({directories} directories, {services} service frames)"
            ).format(
                frames=frames,
                directories=statistics.stream_directories,
                services=statistics.service_frames,
            ),
            tr_plural("{count} CRC error", "{count} CRC errors", errors).format(count=errors),
        ]
        if self._model.silent:
            parts.append(tr("no frame for over 30 s"))
        return " · ".join(parts)

    def _show_logo(self, logo: bytes | None) -> None:
        if logo == self._logo_data:
            return
        self._logo_data = logo
        pixmap = QPixmap()
        if logo and pixmap.loadFromData(logo):
            self.logo.setPixmap(
                pixmap.scaledToHeight(48, Qt.TransformationMode.SmoothTransformation)
            )
            self.logo.show()
        else:
            self.logo.clear()
            self.logo.hide()


def _held_since(model: TrafficModel) -> str:
    """The suffix " · As of 14:05" while the messages shown are held from an earlier
    reception; empty while they are live."""
    held = model.held_since()
    return "" if held is None else " · " + tr("As of {time}").format(time=local(held))


def _status(model: TrafficModel) -> str:
    decoding = model.decoding
    if decoding is not None:
        return tr("Decoding “{service}”").format(service=decoding.label)
    if model.tpeg is not None:
        return tr("TPEG is not being decoded.")
    return tr(_NO_TPEG)


class TrafficView(QWidget):
    """The four pages in a pager, and above them a note with a switch while TPEG is
    switched off on the "Data services" menu."""

    def __init__(
        self,
        controller: ReceiverController,
        model: TrafficModel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._model = model
        self._note = SwitchedOffNote(tr("TPEG is switched off in the data services."))
        self._switch_on = self._note.button
        self._switch_on.clicked.connect(self._on_switch_on)

        self._map = _MapPage(model, controller.settings)
        self._messages = _MessagesPage(model)
        self._flow = _FlowPage(model)
        self._pages = {
            MAP: self._map,
            MESSAGES: self._messages,
            FLOW: self._flow,
            STREAM: _StreamPage(model),
        }
        self._map.more_requested.connect(self.show_message)
        self._messages.map_requested.connect(self.show_on_map)
        self._flow.map_requested.connect(self.show_flow_on_map)
        self._pager = Pager()
        for key in PAGES:
            self._pager.add_page(self._pages[key])
        self._pager.current_changed.connect(lambda _index: self.render())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self._note)
        layout.addWidget(self._pager, stretch=1)

        controller.data_components_changed.connect(self._update_note)
        model.decoding_changed.connect(self._update_note)
        self._update_note()
        self.apply_theme(THEMES[DARK])

    @property
    def pager(self) -> Pager:
        return self._pager

    def page(self, key: str) -> Page:
        return self._pages[key]

    def current_page(self) -> str:
        return PAGES[self._pager.current_index()]

    def show_page(self, key: str) -> None:
        self._pager.set_current_index(PAGES.index(key))

    def show_message(self, key: Key | None) -> None:
        """The message in the list, opened."""
        if key is None:
            return
        self.show_page(MESSAGES)
        self._messages.open_entry(key)

    def show_on_map(self, key: Key) -> None:
        self.show_page(MAP)
        self._map.show_entry(key)

    def show_flow_on_map(self, key: Key) -> None:
        self.show_page(MAP)
        self._map.show_flow(key)

    def render(self) -> None:
        """Draws the page shown, and fills the two lists also while they are not, so
        that their rows are there before a swipe brings them; called by the window at
        its plot rate."""
        self._model.expire()
        current = self.current_page()
        self._pages[current].render()
        for key in (MESSAGES, FLOW):
            if key != current:
                self._pages[key].render()

    def apply_theme(self, theme: Theme) -> None:
        self._pager.set_theme(theme)
        for page in self._pages.values():
            page.apply_theme(theme)

    def _update_note(self) -> None:
        tpeg = self._model.tpeg
        off = tpeg is not None and self._controller.is_data_component_switched_off(tpeg)
        self._note.setVisible(off)

    def _on_switch_on(self) -> None:
        tpeg = self._model.tpeg
        if tpeg is not None:
            self._controller.set_data_component_enabled(tpeg.scid, True)
