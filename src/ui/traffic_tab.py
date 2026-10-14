"""The "Traffic" tab: what ARD TPEG broadcasts. Traffic events (TEC) in a list and on a
map, the traffic flow (TFP) along TMC locations as a speed profile, and for the selected
message what it says and how it sits in the TPEG stream, from the transport frame down
to the attribute bytes of each component. The messages are collected by
`ui.traffic_model.TrafficModel`."""

from __future__ import annotations

import math
import time

import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dab.tpeg import (
    AID_TEC,
    AID_TFP,
    ComponentFrame,
    StoredMessage,
    TransportFrame,
    service_id_text,
)
from dab.tpeg_binary import Component
from dab.tpeg_messages import Location, TecMessage, TfpMessage
from dab.tpeg_tables import (
    application_name,
    application_short_name,
    component_name,
    effect_name,
    level_of_service_name,
    sni_table_name,
)
from i18n import tr, tr_noop, tr_plural
from ui.atmosphere_model import CITIES, MAP_EAST, MAP_NORTH, MAP_SOUTH, MAP_WEST
from ui.atmosphere_tab import CITY_TEXT
from ui.flow_map import FlowLayer
from ui.plots import MapLabel
from ui.road_map import RoadMap
from ui.tables import fill_table, make_table
from ui.traffic_model import (
    FLOW_LEVELS,
    SEVERITIES,
    Key,
    TrafficModel,
    cause_text,
    describe_location,
    describe_management,
    describe_tec,
    flow_stretches,
    local,
    local_date,
    message_key,
    period,
    place_text,
)

# After new messages the list and the map are redrawn at most this often: a frame
# brings some 30 messages every two seconds, and a drawing of 650 costs some 10 ms.
REDRAW_SECONDS = 2.0
_ATTRIBUTE_BYTES_SHOWN = 24
_SELECTED = (255, 255, 255)

_TEC_HEADERS = [
    tr_noop("Effect"),
    tr_noop("Cause"),
    tr_noop("Road"),
    tr_noop("Distance"),
    tr_noop("Period"),
    tr_noop("Place"),
]
_TFP_HEADERS = [
    tr_noop("Traffic situation"),
    tr_noop("Speed"),
    tr_noop("Stretch"),
    tr_noop("Direction"),
    tr_noop("Sections"),
    # Translators: the column with the time a traffic flow message is valid from.
    tr_noop("As of"),
]


def _direction(location: Location | None) -> str:
    if location is None or location.tmc is None:
        return "-"
    # Translators: the direction of a TMC location (both, positive or negative).
    if location.tmc.both_directions:
        return tr("both")
    return tr("positive") if location.tmc.positive_direction else tr("negative")


def _worst_level(message: TfpMessage) -> int | None:
    """The worst level of service along the location (tendency left out)."""
    levels = [
        s.level_of_service & 0x07
        for v in message.vectors
        for s in v.sections
        if s.level_of_service is not None
    ]
    return max((level for level in levels if 0 < level < 7), default=None)


def _speeds(message: TfpMessage) -> str:
    speeds = [s.average_speed for v in message.vectors for s in v.sections if s.average_speed]
    if not speeds:
        return "-"
    low, high = min(speeds), max(speeds)
    return f"{low} km/h" if low == high else f"{low}–{high} km/h"


class TrafficTab(QWidget):
    """Top: TEC messages in a table and on a map, or the TFP messages with the speed
    profile of the selected one. Below: the selected message in words, and its structure
    in the TPEG stream.

    Shows what the `TrafficModel` collects while TPEG is decoded (button at the top, as
    on the other tabs of data services); the messages stay until they expire or are
    cancelled. Draws in `render()`, which the main window calls while this tab is
    visible.
    """

    component_requested = pyqtSignal(object)  # SCId of the TPEG component to decode

    def __init__(self, model: TrafficModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._logo: bytes | None = None
        self._tec_keys: list[Key] = []
        self._tfp_keys: list[Key] = []
        self._selected: dict[int, Key | None] = {AID_TEC: None, AID_TFP: None}
        # Drawing: at once after a change of selection, after new data at most every
        # `REDRAW_SECONDS` (time.monotonic of the last drawing).
        self._data_changed = True
        self._selection_changed = True
        self._drawn_at: float | None = None

        self._receive_button = QPushButton(tr("Receive TPEG"))
        self._receive_button.setToolTip(
            tr(
                "Decodes the ensemble's TPEG service in addition to the audio: traffic "
                "messages (TEC) and traffic flow (TFP) with their locations as TMC codes and "
                "OpenLR coordinates."
            )
        )
        self._receive_button.clicked.connect(self._on_receive_clicked)
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        # Name and description come from the broadcast: never read as markup.
        self._status_label.setTextFormat(Qt.TextFormat.PlainText)
        self._logo_label = QLabel()
        self._logo_label.setToolTip(tr("The logo of the service from the SNI (table 7)."))

        self._tec_table = self._message_table([tr(header) for header in _TEC_HEADERS])
        self._tec_table.setToolTip(
            tr(
                "The traffic messages (TPEG TEC) valid now, the nearest first. Distance and "
                "direction count from the location set in the “GNSS › Satellites” tab to the "
                "nearest point of the message. Messages that carry only a TMC code cannot be "
                "placed without the TMC location table and come last."
            )
        )
        self._tec_table.cellClicked.connect(lambda row, _c: self._select_row(AID_TEC, row))
        self._tfp_table = self._message_table([tr(header) for header in _TFP_HEADERS])
        self._tfp_table.setToolTip(
            tr(
                "The traffic flow (TPEG TFP) per stretch, the worst traffic situation first. "
                "The stretches are given as a TMC location with an extent; where they lie is "
                "only in the location table, so there is no map for them."
            )
        )
        self._tfp_table.cellClicked.connect(lambda row, _c: self._select_row(AID_TFP, row))

        self._map = self._build_map()
        self._profile = pg.PlotWidget(title=tr("Traffic flow along the stretch"))
        self._profile.setLabel("left", tr("Speed (km/h)"))
        self._profile.setLabel("bottom", tr("TMC locations from the first location"))
        self._profile.showGrid(x=False, y=True, alpha=0.15)
        self._profile.setToolTip(
            tr(
                "The sections of the selected TFP message: bars up to the average speed, "
                "coloured by the traffic situation (green free flow, yellow heavy or slow, "
                "orange queuing or stationary, red closed, grey unknown). Without a speed "
                "there is only a point on the baseline."
            )
        )
        # A black outline keeps neighbouring bars of the same colour apart.
        self._profile_bars = pg.BarGraphItem(
            x0=[], x1=[], height=[], brushes=[], pen=pg.mkPen(0, 0, 0, width=2)
        )
        self._profile.addItem(self._profile_bars)
        self._profile_points = pg.ScatterPlotItem(
            pxMode=True, hoverable=True, tip=self._tip, hoverSize=12
        )
        self._profile.addItem(self._profile_points)

        tec_page = QSplitter(Qt.Orientation.Horizontal)
        tec_page.addWidget(self._tec_table)
        tec_page.addWidget(self._map)
        tec_page.setSizes([600, 500])
        tfp_page = QSplitter(Qt.Orientation.Horizontal)
        tfp_page.addWidget(self._tfp_table)
        tfp_page.addWidget(self._profile)
        tfp_page.setSizes([600, 500])
        self._views = QTabWidget()
        self._views.addTab(tec_page, tr("Messages (TEC)"))
        self._views.addTab(tfp_page, tr("Traffic flow (TFP)"))
        self._views.currentChanged.connect(self._on_view_changed)

        self._details = QLabel()
        self._details.setWordWrap(True)
        self._details.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._details.setTextFormat(Qt.TextFormat.RichText)
        self._details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        details_scroll = QScrollArea()
        details_scroll.setWidgetResizable(True)
        details_scroll.setWidget(self._details)

        self._structure = QTreeWidget()
        # Translators: the columns of the structure of a TPEG message: its elements,
        # their size in bytes, and their content.
        self._structure.setHeaderLabels([tr("Element"), tr("Bytes"), tr("Content")])
        self._structure.setColumnWidth(0, 300)
        self._structure.setColumnWidth(1, 60)
        self._structure.setToolTip(
            tr(
                "How the selected message was transmitted: the transport frame (sync word "
                "FF0F, length, header CRC), in it the service frame with the service ID and "
                "per application a component frame with a header and data CRC of its own. "
                "The message itself is a tree of TPEG2 components: identifier, length, "
                "attribute block, child components. “Content” shows the attribute bytes; all "
                "of them in the tooltip of the row."
            )
        )

        controls = QHBoxLayout()
        controls.addWidget(self._receive_button)
        controls.addWidget(self._status_label, stretch=1)
        controls.addWidget(self._logo_label)

        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.addWidget(details_scroll)
        bottom.addWidget(self._structure)
        bottom.setSizes([500, 600])
        main = QSplitter(Qt.Orientation.Vertical)
        main.addWidget(self._views)
        main.addWidget(bottom)
        main.setSizes([600, 300])

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(main, stretch=1)
        self.setLayout(layout)

        model.messages_changed.connect(self._on_data_changed)
        model.decoding_changed.connect(self._update_status)
        model.roads_changed.connect(self._on_roads_changed)
        model.locations_changed.connect(self._on_locations_changed)
        self._update_status()

    @staticmethod
    def _message_table(headers: list[str]) -> QTableWidget:
        table = make_table(headers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        return table

    def _build_map(self) -> pg.PlotWidget:
        plot = pg.PlotWidget(title=tr("Traffic messages with an OpenLR location"))
        # A degree of longitude is cos(latitude) times as long as one of latitude.
        plot.setAspectLocked(True, ratio=math.cos(math.radians(51.2)))
        plot.setLabel("bottom", tr("Longitude (° E)"))
        plot.setLabel("left", tr("Latitude (° N)"))
        plot.setXRange(MAP_WEST, MAP_EAST, padding=0.02)
        plot.setYRange(MAP_SOUTH, MAP_NORTH, padding=0.02)
        plot.showGrid(x=True, y=True, alpha=0.15)
        plot.setToolTip(
            tr(
                "The messages with an OpenLR location: the course of the road the message "
                "sends along, and a symbol at its start. A click on a symbol selects the "
                "message. Mouse wheel: zoom, drag: move. The cross is the location.\n\n"
                "Beneath, the road network from OpenStreetMap, if it was built "
                "(tools/build_road_network.py): zoomed in close, down to the district roads."
            )
        )
        # The road network beneath, when it was built; else a few cities for orientation.
        self._road_map = RoadMap(plot)
        # The disturbed traffic flow, when the TMC location table places it.
        self._flow_map = FlowLayer(plot, width=3.0, offset=3.0)
        self._cities = []
        for name, latitude, longitude in CITIES:
            dot = pg.ScatterPlotItem(
                [longitude], [latitude], size=4, pen=None, brush=pg.mkBrush(*CITY_TEXT)
            )
            plot.addItem(dot)
            label = MapLabel(name, color=CITY_TEXT)
            label.setPos(longitude, latitude)
            plot.addItem(label)
            self._cities += [dot, label]
        legend = plot.addLegend(offset=(10, 10))
        self._lines = []
        self._markers = []
        for name, colour, symbol in SEVERITIES:
            line = pg.PlotDataItem(pen=pg.mkPen(*colour, width=2), connect="finite")
            line.setZValue(5)
            plot.addItem(line)
            self._lines.append(line)
            markers = pg.ScatterPlotItem(
                pxMode=True,
                hoverable=True,
                tip=self._tip,
                hoverSize=13,
                symbol=symbol,
                size=9,
                pen=pg.mkPen(0, 0, 0, width=1),
                brush=pg.mkBrush(*colour),
            )
            markers.setZValue(10)
            markers.sigClicked.connect(self._on_marker_clicked)
            plot.addItem(markers)
            self._markers.append(markers)
            legend.addItem(markers, tr(name))
        self._selected_line = pg.PlotDataItem(pen=pg.mkPen(*_SELECTED, width=4))
        self._selected_line.setZValue(15)
        plot.addItem(self._selected_line)
        self._observer_marker = pg.ScatterPlotItem(
            size=14, symbol="+", pen=pg.mkPen(*_SELECTED, width=2), brush=None
        )
        self._observer_marker.setZValue(20)
        plot.addItem(self._observer_marker)
        return plot

    @staticmethod
    def _tip(x: float, y: float, data: tuple[Key, str]) -> str:
        return data[1]

    def render(self) -> None:
        self._update_status()
        self._model.expire()
        now = time.monotonic()
        due = self._data_changed and (
            self._drawn_at is None or now - self._drawn_at >= REDRAW_SECONDS
        )
        if due:
            self._drawn_at = now
            self._data_changed = False
            self._selection_changed = True
            self._fill_tec()
            self._fill_tfp()
            self._draw_map()
        if self._selection_changed:
            self._selection_changed = False
            self._show_selection()

    def _on_data_changed(self) -> None:
        self._data_changed = True

    def _on_locations_changed(self) -> None:
        self._data_changed = True

    def _on_roads_changed(self) -> None:
        roads = self._model.roads
        self._road_map.set_network(roads)
        for item in self._cities:
            item.setVisible(roads is None)
        self._data_changed = True

    # --- Lists ---------------------------------------------------------------------

    def _fill_tec(self) -> None:
        rows = []
        self._tec_keys = []
        for entry in self._model.entries():
            message = entry.message
            # The road from the TMC location table, else what OpenLR says of it.
            road = entry.tmc[0].road if entry.tmc and entry.tmc[0].road else entry.road_kind()
            cells = (
                effect_name(message.effect),
                cause_text(message),
                road or "-",
                "-" if entry.distance is None else entry.distance_text(),
                period(message),
                entry.where() or place_text(message.location),
            )
            self._tec_keys.append(entry.key)
            rows.append(cells)
        fill_table(self._tec_table, rows, right_aligned=range(3, 4))
        self._restore_selection(AID_TEC)

    def _fill_tfp(self) -> None:
        rows = []
        flows = {flow.key: flow for flow in self._model.flows()}
        for stored in self._model.stored(TfpMessage):
            message: TfpMessage = stored.message
            worst = _worst_level(message)
            tmc = None if message.location is None else message.location.tmc
            sections = sum(len(v.sections) for v in message.vectors)
            flow = flows.get(message_key(message))
            if flow is not None:
                place = flow.where()
            else:
                place = "-" if tmc is None else f"TMC {tmc.location} (+{tmc.extent})"
            cells = (
                level_of_service_name(worst),
                _speeds(message),
                place,
                _direction(message.location),
                str(sections),
                local(message.start, "%H:%M"),
            )
            location = -1 if tmc is None else tmc.location
            key = message_key(message)
            rows.append(((-(worst or 0), location, key), key, cells))
        rows.sort(key=lambda row: row[0])
        self._tfp_keys = [key for _, key, _ in rows]
        fill_table(self._tfp_table, [cells for _, _, cells in rows], right_aligned=range(4, 5))
        self._restore_selection(AID_TFP)

    def _keys(self, aid: int) -> list[Key]:
        return self._tec_keys if aid == AID_TEC else self._tfp_keys

    def _table(self, aid: int):
        return self._tec_table if aid == AID_TEC else self._tfp_table

    def _restore_selection(self, aid: int) -> None:
        """Rows move with every refill; the selection follows its message."""
        key = self._selected[aid]
        table = self._table(aid)
        keys = self._keys(aid)
        if key in keys:
            table.selectRow(keys.index(key))
        else:
            table.clearSelection()

    def _select_row(self, aid: int, row: int) -> None:
        keys = self._keys(aid)
        if 0 <= row < len(keys):
            self._selected[aid] = keys[row]
            self._selection_changed = True
            self.render()

    def _on_marker_clicked(self, _item, points, _event=None) -> None:
        if len(points):
            key = points[0].data()[0]
            self._selected[AID_TEC] = key
            self._views.setCurrentIndex(0)
            self._restore_selection(AID_TEC)
            if key in self._tec_keys:
                self._tec_table.scrollToItem(self._tec_table.item(self._tec_keys.index(key), 0))
            self._selection_changed = True
            self.render()

    def _on_view_changed(self) -> None:
        self._selection_changed = True
        self.render()

    # --- Map and profile ---------------------------------------------------------------

    def _draw_map(self) -> None:
        lines: list[tuple[list[float], list[float]]] = [([], []) for _ in SEVERITIES]
        markers: list[tuple[list[float], list[float], list]] = [([], [], []) for _ in SEVERITIES]
        for entry in self._model.entries():
            points = entry.path
            if not points:
                continue
            xs, ys = lines[entry.severity]
            xs.extend([p[0] for p in points] + [math.nan])
            ys.extend([p[1] for p in points] + [math.nan])
            mx, my, data = markers[entry.severity]
            mx.append(points[0][0])
            my.append(points[0][1])
            message = entry.message
            tip = f"{effect_name(message.effect)}: {cause_text(message)}\n{period(message)}"
            if entry.where():
                tip += f"\n{entry.where()}"
            data.append((entry.key, tip))
        for line, (xs, ys) in zip(self._lines, lines):
            line.setData(xs, ys)
        for item, (mx, my, data) in zip(self._markers, markers):
            item.setData(x=mx, y=my, data=data)
        observer = self._model.observer
        self._observer_marker.setData([observer.longitude], [observer.latitude])
        self._flow_map.set_flows(self._model.flows())

    def _draw_selected_path(self, key: Key | None) -> None:
        entry = None if key is None else self._model.entry(key)
        path = () if entry is None else entry.path
        self._selected_line.setData([p[0] for p in path], [p[1] for p in path])
        if not path:
            return
        # Bring the message into view if it is not, keeping the zoom.
        (x0, x1), (y0, y1) = self._map.getViewBox().viewRange()
        lon, lat = path[0]
        if not (x0 <= lon <= x1 and y0 <= lat <= y1):
            half_x, half_y = (x1 - x0) / 2, (y1 - y0) / 2
            self._map.setXRange(lon - half_x, lon + half_x, padding=0)
            self._map.setYRange(lat - half_y, lat + half_y, padding=0)

    def _draw_profile(self, message: TfpMessage | None) -> None:
        """Bars of the sections' speed, from the primary point (0) on; with the names of
        the points where the TMC location table knows them."""
        flow = None if message is None else self._model.flow(message_key(message))
        if message is None:
            stretches = ()
        elif flow is not None:
            stretches = flow.stretches
        else:
            ends = [s.offset for v in message.vectors[:1] for s in v.sections]
            stretches = flow_stretches(message, max(ends, default=0) + 1)
        x0, x1, heights, brushes, points = [], [], [], [], []
        for stretch in stretches:
            colour = FLOW_LEVELS.get(stretch.level, FLOW_LEVELS[0])[1]
            speed = stretch.speed or 0
            x0.append(stretch.start)
            x1.append(stretch.end)
            heights.append(speed)
            brushes.append(pg.mkBrush(*colour))
            tip = tr("Locations +{start} to +{end}: {level}").format(
                start=stretch.start, end=stretch.end, level=tr(FLOW_LEVELS[stretch.level][0])
            )
            if flow is not None:
                tip += (
                    f"\n{flow.points[stretch.end].label()} → {flow.points[stretch.start].label()}"
                )
            if stretch.speed:
                tip += f", {stretch.speed} km/h"
            points.append(
                {
                    "pos": ((stretch.start + stretch.end) / 2, speed),
                    "data": (None, tip),
                    "brush": pg.mkBrush(*colour),
                    "size": 8,
                }
            )
        # In the direction of travel from left to right, as on the touch surface: the
        # head of the stretch (offset 0) on the right.
        self._profile.getViewBox().invertX(True)
        axis = self._profile.getAxis("bottom")
        if flow is not None and len(flow.points) > 1:
            last = len(flow.points) - 1
            ticks = [(i, flow.points[i].label()) for i in sorted({last, last // 2, 0})]
            axis.setTicks([ticks])
            self._profile.setLabel("bottom", tr("Locations in the direction of travel →"))
        else:
            axis.setTicks(None)
            self._profile.setLabel(
                "bottom",
                tr("TMC locations before the head of the stretch, in the direction of travel →"),
            )
        self._profile_bars.setOpts(x0=x0, x1=x1, height=heights, brushes=brushes)
        self._profile_points.setData(points)
        if x0:
            self._profile.setXRange(min(x0), max(x1), padding=0.02)
            self._profile.setYRange(0, max(max(heights), 60) * 1.1, padding=0)

    # --- Selected message ------------------------------------------------------------

    def _show_selection(self) -> None:
        aid = AID_TEC if self._views.currentIndex() == 0 else AID_TFP
        key = self._selected[aid]
        stored = None if key is None else self._model.get(key)
        if aid == AID_TEC:
            self._draw_selected_path(None if stored is None else key)
        else:
            self._draw_profile(None if stored is None else stored.message)
        if stored is None:
            self._details.setText(
                tr(
                    "Click a message in the list or on the map to see it here in words and "
                    "bottom right how it is built."
                )
            )
            self._structure.clear()
            return
        message = stored.message
        if isinstance(message, TecMessage):
            lines = describe_tec(message)
            entry = self._model.entry(key)
            if entry is not None and entry.where():
                lines.insert(
                    1,
                    tr("Location: {location} (TMC location table)").format(location=entry.where()),
                )
        else:
            lines = self._describe_tfp(message)
        lines.append(f"<span style='color: gray'>{describe_management(stored)}</span>")
        self._details.setText("<br>".join(lines))
        self._fill_structure(stored)

    def _describe_tfp(self, message: TfpMessage) -> list[str]:
        worst = _worst_level(message)
        heading = tr("Traffic flow: {level}").format(level=level_of_service_name(worst))
        lines = [f"<b>{heading}</b> ({_speeds(message)})"]
        lines.append(tr("As of {time}").format(time=local_date(message.start)))
        for vector in message.vectors:
            if vector.time_offset == 0:
                # Translators: the traffic flow now, as opposed to a prediction.
                when = tr("now")
            else:
                when = tr("in {minutes} min").format(minutes=vector.time_offset)
            parts = []
            for section in sorted(vector.sections, key=lambda s: s.offset):
                text = f"+{section.offset}: {level_of_service_name(section.level_of_service)}"
                if section.average_speed:
                    text += f" {section.average_speed} km/h"
                parts.append(text)
            if message.spatial_resolution == 0:
                sections = tr("Sections ({when}, positions in TMC locations): {sections}")
            else:
                sections = tr("Sections ({when}, positions in metres): {sections}")
            lines.append(sections.format(when=when, sections="; ".join(parts)))
        lines += describe_location(message.location)
        return lines

    # --- Structure -----------------------------------------------------------------

    def _fill_structure(self, stored: StoredMessage) -> None:
        self._structure.clear()
        frame: TransportFrame = stored.frame
        message = stored.message
        top = QTreeWidgetItem(
            [
                tr("Transport frame"),
                str(frame.size + 7),
                tr("Sync word FF0F, header CRC correct, frame type {type}").format(
                    type=frame.frame_type
                ),
            ]
        )
        self._structure.addTopLevelItem(top)
        if frame.encryption == 0:
            encryption = tr("Encryption {encryption} (none)")
        else:
            encryption = tr("Encryption {encryption} (yes)")
        service = QTreeWidgetItem(
            [
                tr("Service frame {service}").format(service=service_id_text(frame.service_id)),
                str(frame.size),
                encryption.format(encryption=frame.encryption),
            ]
        )
        top.addChild(service)
        for component_frame in frame.components:
            item = self._component_frame_item(component_frame)
            service.addChild(item)
            if component_frame.scid == message.scid:
                context = "tec" if isinstance(message, TecMessage) else "tfp"
                message_item = self._component_item(message.component, context)
                item.addChild(message_item)
                item.setExpanded(True)
                self._expand(message_item)
        top.setExpanded(True)
        service.setExpanded(True)

    @staticmethod
    def _component_frame_item(frame: ComponentFrame) -> QTreeWidgetItem:
        parts = [tr("Header CRC correct") if frame.header_crc_ok else tr("Header CRC wrong")]
        if frame.data_crc_ok is not None:
            parts.append(tr("data CRC correct") if frame.data_crc_ok else tr("data CRC wrong"))
        if frame.group_priority is not None:
            parts.append(tr("priority {priority}").format(priority=frame.group_priority))
        count = frame.message_count
        if count is not None:
            if frame.scid == 0:
                unit = tr_plural("{count} table", "{count} tables", count)
            else:
                unit = tr_plural("{count} message", "{count} messages", count)
            parts.append(unit.format(count=count))
        return QTreeWidgetItem(
            [
                tr("Component frame SCID {scid}: {application}").format(
                    scid=frame.scid, application=application_name(frame.application)
                ),
                str(frame.size + 5),
                ", ".join(parts),
            ]
        )

    def _component_item(self, component: Component, context: str) -> QTreeWidgetItem:
        name, child_context = component_name(context, component.cid)
        attributes = component.attributes
        shown = attributes[:_ATTRIBUTE_BYTES_SHOWN].hex(" ")
        if len(attributes) > _ATTRIBUTE_BYTES_SHOWN:
            shown += " …"
        item = QTreeWidgetItem([f"{name} (ID {component.cid})", str(component.size), shown])
        if attributes:
            count = len(attributes)
            tip = tr_plural(
                "{count} byte of attributes:\n{bytes}",
                "{count} bytes of attributes:\n{bytes}",
                count,
            )
            item.setToolTip(2, tip.format(count=count, bytes=attributes.hex(" ")))
        for child in component.children:
            item.addChild(self._component_item(child, child_context or ""))
        return item

    @staticmethod
    def _expand(item: QTreeWidgetItem) -> None:
        item.setExpanded(True)
        for index in range(item.childCount()):
            TrafficTab._expand(item.child(index))

    # --- Status --------------------------------------------------------------------

    def _on_receive_clicked(self) -> None:
        tpeg = self._model.tpeg
        if tpeg is not None:
            self.component_requested.emit(tpeg.scid)

    def _update_status(self) -> None:
        model = self._model
        tpeg, decoding = model.tpeg, model.decoding
        self._receive_button.setVisible(
            tpeg is not None and (decoding is None or decoding.scid != tpeg.scid)
        )
        if tpeg is not None:
            self._receive_button.setText(tr("Receive “{service}”").format(service=tpeg.label))
        parts = []
        if decoding is not None:
            parts.append(tr("Decoding “{service}”.").format(service=decoding.label))
        elif tpeg is not None:
            parts.append(tr("TPEG is not being decoded."))
        else:
            parts.append(
                tr(
                    "No TPEG in the ensemble; ARD TPEG runs in the ensembles of the regional "
                    "public broadcasters, such as on 7A (NDR, Lower Saxony)."
                )
            )
        service = model.service
        if service is not None and service.name:
            text = service.name
            if service.description:
                text += f" – {service.description}"
            parts.append(text + ".")
        counts = []
        for aid in (AID_TEC, AID_TFP):
            have, announced = model.counts(aid)
            if announced:
                if aid == AID_TEC:
                    text = tr_plural(
                        "{have} of {announced} traffic message",
                        "{have} of {announced} traffic messages",
                        announced,
                    )
                else:
                    text = tr_plural(
                        "{have} of {announced} flow message",
                        "{have} of {announced} flow messages",
                        announced,
                    )
                counts.append(text.format(have=have, announced=announced))
            elif have:
                if aid == AID_TEC:
                    text = tr_plural("{count} traffic message", "{count} traffic messages", have)
                else:
                    text = tr_plural("{count} flow message", "{count} flow messages", have)
                counts.append(text.format(count=have))
        if counts:
            parts.append(", ".join(counts) + ".")
        held = model.held_since()
        if held is not None:
            parts.append(tr("As of {time}").format(time=local(held)) + ".")
        statistics = model.statistics
        if statistics is not None:
            errors = sum(c.crc_errors for c in statistics.components.values())
            frames = statistics.stream_directories + statistics.service_frames
            frames_text = tr_plural("{count} frame", "{count} frames", frames)
            errors_text = tr_plural("{count} CRC error", "{count} CRC errors", errors)
            parts.append(f"{frames_text.format(count=frames)}, {errors_text.format(count=errors)}.")
            if model.silent:
                parts.append(tr("No frame for over 30 s."))
        if service is not None:
            versions = [
                f"{application_short_name(e.application)} {major}.{minor} (SCID {e.scid})"
                for e in service.components
                if e.scid in service.versions
                for major, minor in (service.versions[e.scid],)
            ]
            tables = ", ".join(sni_table_name(t) for t in service.tables)
            self._status_label.setToolTip(
                tr(
                    "Service and network information (SNI) of the TPEG service.\n"
                    "Applications and versions: {versions}\n"
                    "SNI tables received: {tables}"
                ).format(versions=", ".join(versions) or "-", tables=tables or "-")
            )
        self._status_label.setText(" ".join(parts))
        self._show_logo(None if service is None else service.logo)

    def _show_logo(self, logo: bytes | None) -> None:
        if logo == self._logo:
            return
        self._logo = logo
        pixmap = QPixmap()
        if logo and pixmap.loadFromData(logo):
            self._logo_label.setPixmap(
                pixmap.scaledToHeight(40, Qt.TransformationMode.SmoothTransformation)
            )
        else:
            self._logo_label.clear()
