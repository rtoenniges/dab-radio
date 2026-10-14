"""The "Atmosphere" tab under "GNSS": what the SSRZ corrections of GEPOS say about the ionosphere and
the troposphere over Germany, on maps with the reference stations of its grids (collected by
`ui.atmosphere_model.AtmosphereModel`)."""

from __future__ import annotations

import math
import time

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from i18n import tr, tr_noop
from ui.atmosphere_model import (
    CITIES,
    IONOSPHERE_GLOBAL,
    MAP_EAST,
    MAP_NORTH,
    MAP_SOUTH,
    MAP_WEST,
    RASTER_STEP,
    REDRAW_SECONDS,
    TROPOSPHERE_VIEW_FACTOR,
    TROPOSPHERE_VIEW_TOTAL,
    TROPOSPHERE_VIEW_WET,
    Field,
    Station,
    decimal_point,
)
from ui.gnss_model import GnssModel
from ui.plots import MapLabel
from ui.tables import fill_table, make_table

Colour = tuple[int, int, int]

# Colour steps of the station points.
_COLOUR_STEPS = 256
CITY_TEXT: Colour = (170, 170, 180)
_NO_VALUE = QColor(120, 120, 120)

_SATELLITE_HEADERS = [
    tr_noop("Satellite"),
    tr_noop("Regional a00"),
    tr_noop("Gradient north"),
    tr_noop("Gradient east"),
    tr_noop("Grid: mean"),
    tr_noop("Grid: range"),
    tr_noop("Points"),
]

_RASTER_RECT = QRectF(
    MAP_WEST - RASTER_STEP / 2,
    MAP_SOUTH - RASTER_STEP / 2,
    MAP_EAST - MAP_WEST + RASTER_STEP,
    MAP_NORTH - MAP_SOUTH + RASTER_STEP,
)


class MapPanel:
    """A map of Germany: a raster, the stations coloured by their value, a few cities
    with their names above the stations (`MapLabel`, its halo in the colour of `ground`).
    With `hover_tips` the mouse over a station shows its tip and value; a finger has
    nothing to hover with."""

    def __init__(
        self,
        title: str | None,
        city_text: Colour = CITY_TEXT,
        ground: Colour = (0, 0, 0),
        font: QFont | None = None,
        point_size: int = 9,
        hover_tips: bool = True,
    ) -> None:
        self._point_size = point_size
        self._label_style: dict[str, str] = {}
        self.plot = pg.PlotWidget(title=title)
        # A degree of longitude is cos(latitude) times as long as one of latitude.
        self.plot.setAspectLocked(True, ratio=math.cos(math.radians(51.2)))
        self.plot.setLabel("bottom", tr("Longitude (° E)"))
        self.plot.setLabel("left", tr("Latitude (° N)"))
        self.plot.setXRange(MAP_WEST, MAP_EAST, padding=0.02)
        self.plot.setYRange(MAP_SOUTH, MAP_NORTH, padding=0.02)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self._image = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self._image)
        self._colour_map_name = "viridis"
        self._bar = pg.ColorBarItem(
            colorMap=pg.colormap.get(self._colour_map_name), label="", interactive=False
        )
        self._bar.setImageItem(self._image, insert_in=self.plot.getPlotItem())
        self._city_dots: list[pg.ScatterPlotItem] = []
        self._city_labels: list[MapLabel] = []
        for name, latitude, longitude in CITIES:
            dot = pg.ScatterPlotItem(
                [longitude], [latitude], size=4, pen=None, brush=pg.mkBrush(*city_text)
            )
            dot.setZValue(5)
            self.plot.addItem(dot)
            self._city_dots.append(dot)
            label = MapLabel(name, color=city_text, halo=ground, font=font)
            label.setPos(longitude, latitude)
            self.plot.addItem(label)
            self._city_labels.append(label)
        self._points = (
            pg.ScatterPlotItem(pxMode=True, hoverable=True, tip=self._tip, hoverSize=12)
            if hover_tips
            else pg.ScatterPlotItem(pxMode=True)
        )
        self._points.setZValue(10)
        self.plot.addItem(self._points)
        # Pens and brushes are made once: new ones for every point and drawing cost more
        # than the rest of it. The colours of the points come in 256 steps.
        self._outline = pg.mkPen(0, 0, 0, width=1)
        self._no_value_pen = pg.mkPen(_NO_VALUE)
        self._no_value_brush = pg.mkBrush(0, 0, 0)
        self._brushes: dict[str, list[QBrush]] = {}

    @staticmethod
    def _tip(x: float, y: float, data: str) -> str:
        return data

    def set_scale_style(self, font: QFont, width: int, colour: QColor) -> None:
        """Figures and label of the colour scale in this font and colour (the touch
        surface); `width` makes room for the figures."""
        for name in ("left", "right"):
            axis = self._bar.getAxis(name)
            axis.setStyle(tickFont=font)
            axis.setTextPen(pg.mkPen(colour))
            axis.setPen(pg.mkPen(colour))
        self._bar.axis.setWidth(width)
        self._label_style = {"color": colour.name(), "font-size": f"{font.pixelSize()}px"}

    def set_city_colour(self, colour: Colour | QColor, ground: Colour | QColor) -> None:
        """Colours the cities anew, and the halo around their names as the ground (the
        touch surface has a dark and a light scheme)."""
        for dot in self._city_dots:
            dot.setBrush(pg.mkBrush(colour))
        for label in self._city_labels:
            label.setColor(colour)
            label.set_halo(ground)

    def _brush_steps(self, name: str) -> list[QBrush]:
        brushes = self._brushes.get(name)
        if brushes is None:
            table = pg.colormap.get(name).getLookupTable(0.0, 1.0, _COLOUR_STEPS, alpha=True)
            brushes = [pg.mkBrush(*(int(c) for c in rgba)) for rgba in table]
            self._brushes[name] = brushes
        return brushes

    def show(
        self, field: Field | None, stations: list[Station], tips: list[str] | None = None
    ) -> None:
        """`tips`, one per station, only for a panel with `hover_tips`."""
        x = [s.longitude for s in stations]
        y = [s.latitude for s in stations]
        if field is None:
            self._image.clear()
            self._points.setData(
                x=x,
                y=y,
                size=self._point_size - 2,
                pen=self._no_value_pen,
                brush=self._no_value_brush,
                data=tips,
            )
            return
        if field.colour_map != self._colour_map_name:
            self._colour_map_name = field.colour_map
            self._bar.setColorMap(pg.colormap.get(field.colour_map))
        self._bar.getAxis("left").setLabel(field.label, **self._label_style)
        self._bar.setLevels(field.levels)
        if field.raster is None:
            self._image.clear()
        else:
            self._image.setImage(field.raster, levels=field.levels, autoLevels=False)
            # Pixel centres on the raster points; set after the image, whose size it
            # needs.
            self._image.setRect(_RASTER_RECT)
        low, high = field.levels
        span = high - low if high > low else 1.0
        steps = self._brush_steps(field.colour_map)
        finite = np.isfinite(field.values)
        index = np.clip(
            np.nan_to_num((field.values - low) / span * (_COLOUR_STEPS - 1)), 0, _COLOUR_STEPS - 1
        ).astype(int)
        brushes, pens = [], []
        for ok, step in zip(finite, index):
            brushes.append(steps[step] if ok else self._no_value_brush)
            pens.append(self._outline if ok else self._no_value_pen)
        texts = None
        if tips is not None:
            # The unit texts hold one number and no full stop: all points are decimal.
            point = decimal_point()
            texts = [
                f"{tip}\n{field.unit_format.format(value=value).replace('.', point)}" if ok else tip
                for value, tip, ok in zip(field.values, tips, finite)
            ]
        self._points.setData(x=x, y=y, size=self._point_size, pen=pens, brush=brushes, data=texts)


class AtmosphereTab(QWidget):
    """Left the ionosphere, right the troposphere, each on a map of the service area with
    the reference stations of the SSRZ grids; below the corrections per satellite and
    what the metadata say.

    Shows what the `GnssModel` collects while GEPOS is decoded (button at the top, as in
    the "Satellites" tab). The corrections stay after the decoding stops and are then marked
    as out of date. Draws in `render()`, which the main window calls while this tab is
    visible.
    """

    component_requested = pyqtSignal(object)  # SCId of the GEPOS component to decode

    def __init__(self, model: GnssModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        # Drawing: at once after a change of view, after new data at most every
        # `REDRAW_SECONDS` (time.monotonic of the last drawing).
        self._view_changed = True
        self._data_changed = False
        self._drawn_at: float | None = None

        self._receive_button = QPushButton(tr("Receive GEPOS"))
        self._receive_button.setToolTip(
            tr(
                "Decodes the data service GEPOS of the BKG in addition to the audio. Its "
                "SSRZ messages describe, among other things, the ionosphere and the "
                "troposphere."
            )
        )
        self._receive_button.clicked.connect(self._on_receive_clicked)
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)

        self._ionosphere_view = QComboBox()
        self._ionosphere_view.addItem(tr("VTEC, global model (ZM006)"), IONOSPHERE_GLOBAL)
        self._ionosphere_view.setToolTip(
            tr(
                "What the map of the ionosphere shows: the electron content over Germany "
                "according to the global model, or for one satellite the correction at "
                "the stations that is left after the models."
            )
        )
        self._ionosphere_view.currentIndexChanged.connect(self._on_view_changed)
        self._troposphere_view = QComboBox()
        self._troposphere_view.addItem(tr("Zenith wet delay (ZWD)"), TROPOSPHERE_VIEW_WET)
        self._troposphere_view.addItem(tr("Zenith total delay (ZTD)"), TROPOSPHERE_VIEW_TOTAL)
        self._troposphere_view.addItem(tr("Wet scale factor sent"), TROPOSPHERE_VIEW_FACTOR)
        self._troposphere_view.setToolTip(
            tr(
                "What the map of the troposphere shows: the delay straight up (zenith), "
                "computed from the model of the troposphere and the scale factors sent, "
                "or the scale factors themselves."
            )
        )
        self._troposphere_view.currentIndexChanged.connect(self._on_view_changed)

        self._ionosphere_map = MapPanel(tr("Ionosphere"))
        self._ionosphere_map.plot.setToolTip(
            tr(
                "The ionosphere delays the signals of the satellites according to the "
                "number of free electrons on their way (TEC, in TECU = 10¹⁶ electrons/m²; "
                "1 TECU is 16 cm of path on the GPS frequency L1). SSRZ describes it in "
                "stages that add up:\n\n"
                "• Global model (ZM006): vertical TEC as spherical harmonics, in a layer "
                "at 400 km height, turned along with the sun (largest around 14:00 local "
                "time). The map shows it at the pierce points above each place.\n"
                "• Regional per satellite (ZM005): a plane over Germany, values in the "
                "table below.\n"
                "• Grid per satellite (ZM003): what is still missing at each station after "
                "that, converted to the vertical. This part takes most of the bits of the "
                "data stream.\n\n"
                "Points: the reference stations of the grids (ZM013); grey without a value."
            )
        )
        self._troposphere_map = MapPanel(tr("Troposphere"))
        self._troposphere_map.plot.setToolTip(
            tr(
                "The troposphere delays the signals through air pressure (dry part, about "
                "2.3 m at the zenith at sea level) and water vapour (wet part, 5 to 30 cm, "
                "changing with the weather). SSRZ sends no delays but scale factors to a "
                "model that the service and the receiver compute alike: average weather by "
                "latitude and season (RTCA MOPS), brought to the height of the station, "
                "delay after Saastamoinen.\n\n"
                "GEPOS sends the dry part as one polynomial for all of Germany (ZM007) and "
                "the wet part per station in the grid (ZM004). A scale factor of +30 % "
                "means: 30 % more water vapour than in the model.\n\n"
                "The total delay follows above all the height of the station: stations "
                "high up have less air above them."
            )
        )

        self._satellite_table = make_table([tr(header) for header in _SATELLITE_HEADERS])
        self._satellite_table.setToolTip(
            tr(
                "The ionosphere per satellite. Regional (ZM005): the value a00 in the "
                "centre of Germany and the gradients to the north and east, in TECU per "
                "1000 km, each for the pierce points of this satellite's signals. Grid "
                "(ZM003): mean and range of the residual correction over the stations "
                "with a value, in TECU. A satellite without grid values stands too low or "
                "is seen by too few stations."
            )
        )
        self._satellite_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._satellite_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._satellite_table.cellClicked.connect(self._on_satellite_clicked)
        # Short numbers only: all columns share the width.
        header = self._satellite_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._info_label = QLabel()
        self._info_label.setWordWrap(True)
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        controls = QHBoxLayout()
        controls.addWidget(self._receive_button)
        controls.addWidget(self._status_label, stretch=1)

        left = QVBoxLayout()
        left.addWidget(self._ionosphere_view)
        left.addWidget(self._ionosphere_map.plot, stretch=1)
        right = QVBoxLayout()
        right.addWidget(self._troposphere_view)
        right.addWidget(self._troposphere_map.plot, stretch=1)
        maps = QHBoxLayout()
        maps.addLayout(left, stretch=1)
        maps.addLayout(right, stretch=1)

        bottom = QHBoxLayout()
        bottom.addWidget(self._satellite_table, stretch=3)
        bottom.addWidget(self._info_label, stretch=2)

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addLayout(maps, stretch=3)
        layout.addLayout(bottom, stretch=1)
        self.setLayout(layout)

        model.atmosphere_changed.connect(self._on_data_changed)
        model.decoding_changed.connect(self._update_status)
        self._update_status()

    def render(self) -> None:
        self._update_status()
        now = time.monotonic()
        due = self._data_changed and (
            self._drawn_at is None or now - self._drawn_at >= REDRAW_SECONDS
        )
        if not (self._view_changed or due):
            return
        self._drawn_at = now
        self._data_changed = self._view_changed = False
        atmosphere = self._model.atmosphere
        self._update_satellite_choice()
        tips = [atmosphere.station_tip(s) for s in atmosphere.stations]
        self._ionosphere_map.show(self._ionosphere_field(), atmosphere.stations, tips)
        self._troposphere_map.show(self._troposphere_field(), atmosphere.stations, tips)
        self._fill_satellite_table()
        self._info_label.setText(atmosphere.describe())

    # --- Fields --------------------------------------------------------------------

    def _ionosphere_field(self) -> Field | None:
        satellite = self._ionosphere_view.currentData()
        return self._model.atmosphere.ionosphere_field(satellite, self._model.now())

    def _troposphere_field(self) -> Field | None:
        view = self._troposphere_view.currentData()
        return self._model.atmosphere.troposphere_field(view, self._model.now())

    # --- Satellites ------------------------------------------------------------------

    def _update_satellite_choice(self) -> None:
        combo = self._ionosphere_view
        wanted = [None, *self._model.atmosphere.ionosphere_satellites()]
        present = [combo.itemData(i) for i in range(combo.count())]
        if wanted == present:
            return
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(tr("VTEC, global model (ZM006)"), IONOSPHERE_GLOBAL)
        for name in wanted[1:]:
            combo.addItem(
                tr("Residual correction in the grid: {satellite} (ZM003)").format(satellite=name),
                name,
            )
        index = combo.findData(current)
        combo.setCurrentIndex(max(index, 0))
        combo.blockSignals(False)

    def _fill_satellite_table(self) -> None:
        atmosphere = self._model.atmosphere
        regional = atmosphere.regional_ionosphere
        rows = []
        for name in atmosphere.ionosphere_satellites():
            coefficients = {}
            if regional is not None and name in regional.satellites:
                row = regional.coefficients[regional.satellites.index(name)]
                coefficients = dict(zip(regional.orders, row))
            values = np.concatenate(
                [
                    item.values[item.satellites.index(name)]
                    for item in atmosphere.grid_ionosphere.values()
                    if name in item.satellites
                ]
                or [np.array([])]
            )
            finite = values[np.isfinite(values)]

            def cell(value: float | None, spec: str = "+.2f") -> str:
                if value is None or not np.isfinite(value):
                    return "–"
                text = format(value, spec)
                # What rounds to zero has no sign.
                if text in ("-0.00", "+0.00"):
                    text = text[1:]
                return text.replace(".", decimal_point())

            rows.append(
                (
                    name,
                    cell(coefficients.get((0, 0))),
                    cell(coefficients.get((1, 0))),
                    cell(coefficients.get((0, 1))),
                    cell(float(np.mean(finite)) if finite.size else None),
                    cell(float(np.ptp(finite)) if finite.size else None, ".2f"),
                    f"{finite.size}/{values.size}" if values.size else "–",
                )
            )
        fill_table(self._satellite_table, rows, right_aligned=range(1, 7))

    def _on_satellite_clicked(self, row: int, _column: int) -> None:
        item = self._satellite_table.item(row, 0)
        if item is None:
            return
        index = self._ionosphere_view.findData(item.text())
        if index >= 0:
            self._ionosphere_view.setCurrentIndex(index)

    # --- Status --------------------------------------------------------------------

    def _on_data_changed(self) -> None:
        self._data_changed = True

    def _on_view_changed(self) -> None:
        self._view_changed = True

    def _on_receive_clicked(self) -> None:
        gepos = self._model.gepos
        if gepos is not None:
            self.component_requested.emit(gepos.scid)

    def _update_status(self) -> None:
        model = self._model
        gepos, decoding = model.gepos, model.decoding
        self._receive_button.setVisible(gepos is not None and not model.gepos_decoded)
        parts = []
        if decoding is not None:
            parts.append(tr("Decoding “{service}”.").format(service=decoding.label))
        elif gepos is not None:
            parts.append(tr("GEPOS is not being decoded."))
        else:
            parts.append(
                tr(
                    "No GEPOS in the ensemble; it is broadcast nationwide on channel 5C "
                    "(DR Deutschland)."
                )
            )
        received = model.atmosphere.received
        if received is None:
            if decoding is not None:
                parts.append(tr("Waiting for metadata and grids (every 30 s)."))
        else:
            age = model.now() - received
            if model.atmosphere.is_stale(model.now()):
                parts.append(
                    tr("Corrections out of date (received {minutes:.0f} min ago).").format(
                        minutes=age / 60
                    )
                )
            else:
                parts.append(tr("Corrections up to date (new every 30 s)."))
        self._status_label.setText(" ".join(parts))
