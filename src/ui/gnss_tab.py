"""The "Satellites" tab under "GNSS": what the RTCM stream of GEPOS reveals, above all
where the satellites stand in the sky, computed from their broadcast ephemerides alone
(collected by `ui.gnss_model.GnssModel`)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QSettings, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from dab.gnss import Observer
from dab.rtcm import (
    GNSS_BEIDOU,
    GNSS_GALILEO,
    GNSS_GLONASS,
    GNSS_GPS,
    GlonassEphemeris,
    RtcmStatistics,
    message_label,
    message_name,
)
from i18n import tr, tr_context, tr_noop, tr_plural
from net.ntrip import (
    DEFAULT_PORT,
    CasterStatus,
    GgaPosition,
    NtripCaster,
    NtripCasterError,
    mountpoint_for,
)
from ui.atmosphere_model import decimal, grouped
from ui.gnss_model import (
    POSITION_INTERVAL_SECONDS,
    TRACK_INTERVAL_SECONDS,
    GnssModel,
    SatelliteView,
    Track,
    describe_age,
    sky_xy,
)
from ui.receiver_controller import HEIGHT_RANGE_M, LATITUDE_RANGE, LONGITUDE_RANGE
from ui.tables import fill_table, make_table

_SETTINGS_NTRIP_ENABLED = "ntrip/enabled"
_SETTINGS_NTRIP_PORT = "ntrip/port"
_SETTINGS_NTRIP_GGA = "ntrip/position_from_gga"
# How often the state of the caster (clients, positions) is looked at.
CASTER_STATUS_INTERVAL_MS = 1000
# The fix quality of an NMEA GGA sentence, as a receiver reports it.
_GGA_QUALITY = {
    1: "GNSS",
    2: "DGNSS",
    3: "PPS",
    4: tr_noop("RTK fixed"),
    5: "RTK float",
    # Translators: the fix quality of a GNSS receiver (dead reckoning).
    6: tr_noop("estimated"),
    # Translators: the fix quality of a GNSS receiver (position entered by hand).
    7: tr_noop("manual"),
    8: tr_noop("simulation"),
    9: "SBAS",
}

Colour = tuple[int, int, int]


@dataclass(frozen=True)
class SystemStyle:
    name: str
    colour: Colour
    symbol: str


# Colours: the first three slots of the categorical palette in its dark steps (blue,
# orange, aqua), which stay apart in a scatter plot on the black background for every
# pair, also for colour blind readers. A fourth hue cannot do that next to them, so
# BeiDou gets a neutral grey. The system is never told by colour alone: each has a
# symbol of its own, and every satellite is labelled with its name (G12, R11, ...).
_SYSTEMS = {
    GNSS_GPS: SystemStyle("GPS", (0x39, 0x87, 0xE5), "o"),
    GNSS_GLONASS: SystemStyle("GLONASS", (0xD9, 0x59, 0x26), "s"),
    GNSS_GALILEO: SystemStyle("Galileo", (0x19, 0x9E, 0x70), "t1"),
    GNSS_BEIDOU: SystemStyle("BeiDou", (0xC3, 0xC2, 0xB7), "d"),
}
_LEGEND_SYMBOLS = {"o": "●", "s": "■", "t1": "▲", "d": "◆"}


@dataclass(frozen=True)
class SkyColours:
    """The colours of the sky plot besides those of the systems; the default suits the
    black background of the desktop's plots."""

    grid: Colour = (90, 90, 100)
    grid_text: Colour = (150, 150, 160)
    label_text: Colour = (220, 220, 220)
    hollow: Colour = (0, 0, 0)  # inside the symbol of a satellite out of order


_SATELLITE_HEADERS = [
    tr_noop("Satellite"),
    tr_noop("System"),
    tr_noop("Elevation"),
    tr_noop("Azimuth"),
    tr_noop("Distance"),
    # Translators: the column with the reference time of a satellite's ephemeris.
    tr_noop("Ephemeris"),
]
MESSAGE_HEADERS = [
    tr_noop("Message"),
    tr_noop("Content"),
    tr_noop("Count"),
    tr_noop("Data rate"),
    # Translators: the share of a message type in the data rate of the stream.
    tr_noop("Share"),
]
_SKY_TITLE = tr_noop("Sky above the location")


class SkyPlot:
    """The sky above the listener: every satellite with a valid ephemeris, and the path
    each one takes during the next hour. With `hover_tips` the mouse over a satellite
    shows what is known of it; a finger has nothing to hover with. `title` is a text
    marked with `tr_noop`, shown translated."""

    def __init__(
        self,
        title: str | None = _SKY_TITLE,
        colours: SkyColours | None = None,
        font: QFont | None = None,
        point_size: int = 11,
        hover_tips: bool = True,
    ) -> None:
        self._colours = colours or SkyColours()
        self._font = font
        self._point_size = point_size
        self._hover_tips = hover_tips
        self.plot = pg.PlotWidget(title=None if title is None else tr(title))
        self.plot.setAspectLocked(True)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.plot.hideAxis("bottom")
        self.plot.hideAxis("left")
        self.plot.setRange(xRange=(-100, 100), yRange=(-100, 100), padding=0)
        self._grid_lines: list[pg.PlotDataItem] = []
        self._grid_texts: list[pg.TextItem] = []
        self._compass: list[pg.TextItem] = []
        self._draw_grid()
        self._tracks: dict[str, pg.PlotDataItem] = {}
        self._labels: dict[str, pg.TextItem] = {}
        self._points = (
            pg.ScatterPlotItem(pxMode=True, hoverable=True, tip=self._tip, hoverSize=14)
            if hover_tips
            else pg.ScatterPlotItem(pxMode=True)
        )
        self._points.setZValue(10)
        self.plot.addItem(self._points)
        self.legend = QLabel(
            legend_text() + "  " + tr("○&nbsp;flagged unhealthy  —&nbsp;path over the next hour")
        )
        self.legend.setWordWrap(True)

    def _draw_grid(self) -> None:
        pen = pg.mkPen(self._colours.grid, width=1)
        angle = np.linspace(0, 2 * np.pi, 181)
        for elevation in (0, 30, 60):
            radius = 90 - elevation
            circle = pg.PlotDataItem(radius * np.sin(angle), radius * np.cos(angle), pen=pen)
            self.plot.addItem(circle)
            self._grid_lines.append(circle)
            if elevation:
                text = self._text(f"{elevation}°", self._colours.grid_text, (0, 1))
                text.setPos(radius * math.sin(math.radians(20)), radius * math.cos(0.35))
                self.plot.addItem(text)
                self._grid_texts.append(text)
        for azimuth in range(0, 180, 30):
            a = math.radians(azimuth)
            line = pg.PlotDataItem(
                [-90 * math.sin(a), 90 * math.sin(a)],
                [-90 * math.cos(a), 90 * math.cos(a)],
                pen=pen,
            )
            self.plot.addItem(line)
            self._grid_lines.append(line)
        for label, x, y, anchor in (
            # Translators: the points of the compass around the sky plot.
            (tr_context("compass", "N"), 0, 92, (0.5, 1)),
            (tr_context("compass", "E"), 92, 0, (0, 0.5)),
            (tr_context("compass", "S"), 0, -92, (0.5, 0)),
            (tr_context("compass", "W"), -92, 0, (1, 0.5)),
        ):
            text = self._text(label, self._colours.label_text, anchor)
            text.setPos(x, y)
            self.plot.addItem(text)
            self._compass.append(text)

    def _text(self, text: str, colour: Colour, anchor: tuple[float, float]) -> pg.TextItem:
        item = pg.TextItem(text, color=colour, anchor=anchor)
        if self._font is not None:
            item.setFont(self._font)
        return item

    def set_colours(self, colours: SkyColours) -> None:
        """Colours the plot anew (the touch surface has a dark and a light scheme); the
        points follow with their next update."""
        self._colours = colours
        pen = pg.mkPen(colours.grid, width=1)
        for line in self._grid_lines:
            line.setPen(pen)
        for text in self._grid_texts:
            text.setColor(colours.grid_text)
        for text in (*self._compass, *self._labels.values()):
            text.setColor(colours.label_text)

    @staticmethod
    def _tip(x: float, y: float, data: str) -> str:
        return data

    def update_points(self, satellites: list[SatelliteView]) -> None:
        spots = []
        shown = set()
        for satellite in satellites:
            if not satellite.elevation >= 0:
                continue
            ephemeris = satellite.ephemeris
            style = _SYSTEMS[ephemeris.system]
            x, y = sky_xy(np.array(satellite.azimuth), np.array(satellite.elevation))
            colour = QColor(*style.colour)
            spots.append(
                {
                    "pos": (float(x), float(y)),
                    "size": self._point_size,
                    "symbol": style.symbol,
                    "pen": pg.mkPen(colour, width=2),
                    # Unhealthy satellites hollow: the ring still carries the system.
                    "brush": pg.mkBrush(colour if ephemeris.healthy else self._colours.hollow),
                    "data": _describe(satellite) if self._hover_tips else None,
                }
            )
            label = self._labels.get(ephemeris.name)
            if label is None:
                label = self._text(ephemeris.name, self._colours.label_text, (-0.15, 0.5))
                label.setZValue(11)
                self.plot.addItem(label)
                self._labels[ephemeris.name] = label
            label.setPos(float(x), float(y))
            label.setVisible(True)
            shown.add(ephemeris.name)
        for name, label in self._labels.items():
            if name not in shown:
                label.setVisible(False)
        self._points.setData(spots)

    def update_tracks(self, tracks: dict[str, Track]) -> None:
        """Paths per satellite: (system, azimuth, elevation) over the next hour."""
        for name, (system, azimuth, elevation) in tracks.items():
            x, y = sky_xy(azimuth, np.where(elevation >= 0, elevation, np.nan))
            item = self._tracks.get(name)
            if item is None:
                colour = QColor(*_SYSTEMS[system].colour)
                colour.setAlpha(150)
                item = pg.PlotDataItem(pen=pg.mkPen(colour, width=1.5), connect="finite")
                self.plot.addItem(item)
                self._tracks[name] = item
            item.setData(x, y)
            item.setVisible(True)
        for name, item in self._tracks.items():
            if name not in tracks:
                item.setVisible(False)

    def clear(self) -> None:
        self._points.clear()
        for item in (*self._labels.values(), *self._tracks.values()):
            item.setVisible(False)


def _describe(satellite: SatelliteView) -> str:
    """Tooltip of a satellite in the sky plot."""
    ephemeris = satellite.ephemeris
    lines = [
        f"{ephemeris.name} ({_SYSTEMS[ephemeris.system].name})",
        tr("Elevation {elevation:.1f}°, azimuth {azimuth:.1f}°").format(
            elevation=satellite.elevation, azimuth=satellite.azimuth
        ),
        tr("Distance {distance} km").format(distance=grouped(satellite.distance / 1000)),
        tr("Ephemeris reference time: {age}").format(age=describe_age(satellite.age)),
    ]
    if isinstance(ephemeris, GlonassEphemeris):
        lines.append(
            tr("Frequency channel {channel:+d}").format(channel=ephemeris.frequency_channel)
        )
    if not ephemeris.healthy:
        lines.append(tr("flagged unhealthy in its ephemeris"))
    return "\n".join(lines)


def legend_text() -> str:
    """The systems with their symbols, for a label under the sky plot (rich text)."""
    return "  ".join(
        f'<span style="color: rgb{style.colour}">{_LEGEND_SYMBOLS[style.symbol]}</span>'
        f"&nbsp;{style.name}"
        for style in _SYSTEMS.values()
    )


def describe_ephemeris(satellite: SatelliteView) -> str:
    """When the ephemeris of a satellite refers to, and whether it is flagged unhealthy,
    for a table."""
    age = describe_age(satellite.age)
    if satellite.ephemeris.healthy:
        return age
    # Translators: the reference time of an ephemeris ("3 min ago") that flags its
    # satellite unhealthy.
    return tr("{age}, unhealthy").format(age=age)


def system_name(system: str) -> str:
    return _SYSTEMS[system].name


def system_style(system: str) -> SystemStyle:
    """Name, colour and symbol (as pyqtgraph names it) of a system."""
    return _SYSTEMS[system]


def _kilobytes(count: int) -> str:
    return f"{decimal(count / 1000, '.1f')} kB"


def describe_stream(statistics: RtcmStatistics) -> str:
    """How the stream was checked: data groups, frames and their CRC, ephemerides."""
    if statistics.lost_data_groups:
        groups = tr("RTCM stream: {groups} data groups, {lost} of them lost, {size:.1f} kB.")
    else:
        groups = tr("RTCM stream: {groups} data groups, {size:.1f} kB.")
    count = statistics.ephemerides
    lines = [
        groups.format(
            groups=statistics.data_groups,
            lost=statistics.lost_data_groups,
            size=statistics.stream_bytes / 1000,
        ),
        tr(
            "{frames} frames with a valid CRC-24Q, {errors} with a CRC error, "
            "{discarded} bytes outside frames."
        ).format(
            frames=statistics.frames,
            errors=statistics.crc_errors,
            discarded=statistics.discarded_bytes,
        ),
        tr_plural(
            "Ephemeris of {count} satellite decoded.",
            "Ephemerides of {count} satellites decoded.",
            count,
        ).format(count=count),
    ]
    return "\n".join(lines)


def message_rows(statistics: RtcmStatistics) -> list[tuple[str, str, str, str, str]]:
    """Per message type: number, content, count, data rate and share of the stream."""
    seconds = statistics.seconds
    total = sum(m.frame_bytes for m in statistics.messages.values())
    rows = []
    for kind, messages in statistics.messages.items():
        rate = messages.frame_bytes * 8 / seconds if seconds else 0.0
        rows.append(
            (
                message_label(kind),
                message_name(kind),
                str(messages.count),
                f"{rate:.0f} bit/s",
                f"{messages.frame_bytes / total * 100:.0f} %" if total else "-",
            )
        )
    return rows


class GnssTab(QWidget):
    """Top: the sky plot and the satellites with their direction; below: the messages of
    the RTCM stream and how it was checked.

    Shows what the `GnssModel` collects while a TDC component is decoded (see the
    "Data services" tab); the button at the top starts decoding GEPOS. The ephemerides
    stay after the decoding stops and are used as long as they are valid.

    Plots are drawn in `render()`, which the main window calls at the shared refresh rate
    while this tab is visible.

    The tab also controls the NTRIP caster (`net.ntrip`) that passes the RTCM stream on
    to GNSS receivers, and edits the listener's position (`observer_changed`, which the
    main window hands to the controller); a receiver connected to the caster can set it
    by GGA.
    """

    component_requested = pyqtSignal(object)  # SCId of the GEPOS component to decode
    observer_changed = pyqtSignal(object)  # dab.gnss.Observer: the position entered here

    def __init__(
        self,
        model: GnssModel,
        settings: QSettings | None = None,
        caster: NtripCaster | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._settings = settings
        self._caster = caster
        # SCId of the GEPOS component last seen in the ensemble, for the caster.
        self._seen_gepos: int | None = None
        self._positions_at: float | None = None
        self._tracks_at: float | None = None
        self._dirty = True

        self._receive_button = QPushButton(tr("Receive GEPOS"))
        self._receive_button.setToolTip(
            tr(
                "Decodes the data service GEPOS in addition to the audio. GEPOS is the "
                "satellite positioning service of the Federal Agency for Cartography and "
                "Geodesy (BKG): correction data for positions accurate to the centimetre "
                "(PPP-RTK) and the ephemerides of the satellites, sent as RTCM 3 messages "
                "in a transparent data channel (TDC) in packet mode."
            )
        )
        self._receive_button.clicked.connect(self._on_receive_clicked)
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)

        self._latitude = self._coordinate_spin(LATITUDE_RANGE, tr("° N"))
        # Translators: degrees of longitude east.
        self._longitude = self._coordinate_spin(LONGITUDE_RANGE, tr("° E"))
        self._height = QDoubleSpinBox()
        self._height.setRange(*HEIGHT_RANGE_M)
        self._height.setDecimals(0)
        self._height.setSingleStep(10.0)
        self._height.setSuffix(" m")
        self.set_observer(model.observer)
        self._latitude.valueChanged.connect(self._on_position_changed)
        self._longitude.valueChanged.connect(self._on_position_changed)
        self._height.valueChanged.connect(self._on_position_changed)
        position_tooltip = tr(
            "Your own location (latitude and longitude, WGS 84), from which the sky is "
            "shown. The default is the centre of Germany; a degree off moves the "
            "satellites in the sky by about a degree."
        )
        self._latitude.setToolTip(position_tooltip)
        self._longitude.setToolTip(position_tooltip)
        self._height.setToolTip(
            tr(
                "Height of the location above the ellipsoid (WGS 84), that is about the "
                "height above sea level plus 45 to 50 m in Germany. Of no account for the "
                "sky, important for the troposphere of the corrections: the higher, the "
                "less air lies above the receiver."
            )
        )

        self._sky = SkyPlot()
        self._sky.plot.setToolTip(
            tr(
                "Where the satellites stand right now, computed from nothing but their "
                "ephemerides, which GEPOS sends along. In the centre the zenith, on the "
                "outer circle the horizon, in between 30° and 60° elevation; north at the "
                "top, east on the right. No GNSS receiver is needed for this: the DAB+ "
                "signal alone tells which satellites are above the horizon.\n\nGPS, "
                "Galileo and BeiDou describe their orbit by Keplerian elements, GLONASS "
                "by position, velocity and acceleration at one instant; the orbit around "
                "it follows from integrating the equations of motion. The lines show the "
                "path over the next hour, as far as it lies above the horizon."
            )
        )

        self._satellite_table = make_table([tr(header) for header in _SATELLITE_HEADERS])
        self._satellite_table.setToolTip(
            tr(
                "All satellites with a valid ephemeris, sorted by elevation; grey below "
                "the horizon. Ephemeris: reference time of the orbit sent (toe, for "
                "GLONASS tb) relative to now. An orbit holds for some hours around this "
                "time; GEPOS sends the current one for each satellite every few seconds."
            )
        )
        self._message_table = make_table([tr(header) for header in MESSAGE_HEADERS])
        # The names are long, the numbers short: the name column takes the room.
        message_header = self._message_table.horizontalHeader()
        message_header.setStretchLastSection(False)
        message_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._message_table.setToolTip(
            tr(
                "The messages in the RTCM 3 stream (RTCM 10403.3) by type. 1019, 1020, "
                "1042 and 1046 are the ephemerides of GPS, GLONASS, BeiDou and Galileo. "
                "4090 is the proprietary message of Geo++; its subtype 7 carries SSRZ, the "
                "actual corrections: clock and orbit of the satellites, signal biases, "
                "ionosphere and troposphere (Geo++ SSRZ, version 1.1.3). What they say "
                "about the ionosphere and the troposphere is shown in the “Atmosphere” "
                "tab, what it all means for a receiver at the location in the "
                "“Corrections” tab. Data rate including frame header and CRC."
            )
        )
        self._stream_label = QLabel()
        self._stream_label.setWordWrap(True)
        self._stream_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._stream_label.setToolTip(
            tr(
                "The payloads of the data groups, joined in the order of their continuity "
                "index, make up a byte stream. In it every RTCM frame begins with 0xD3 and "
                "its length and ends with a CRC-24Q over the whole frame, the last check of "
                "the chain. Frames run across the boundaries of the data groups; if a data "
                "group is missing, the frame under way is lost."
            )
        )

        output = self._build_output()

        controls = QHBoxLayout()
        controls.addWidget(self._receive_button)
        controls.addWidget(self._status_label, stretch=1)
        controls.addWidget(QLabel(tr("Location:")))
        controls.addWidget(self._latitude)
        controls.addWidget(self._longitude)
        controls.addWidget(self._height)

        sky_side = QVBoxLayout()
        sky_side.addWidget(self._sky.plot, stretch=1)
        sky_side.addWidget(self._sky.legend)
        middle = QHBoxLayout()
        middle.addLayout(sky_side, stretch=3)
        middle.addWidget(self._satellite_table, stretch=2)

        stream_side = QVBoxLayout()
        stream_side.addWidget(self._stream_label, stretch=1)
        stream_side.addWidget(output)
        bottom = QHBoxLayout()
        bottom.addWidget(self._message_table, stretch=3)
        bottom.addLayout(stream_side, stretch=2)

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addLayout(middle, stretch=3)
        layout.addLayout(bottom, stretch=1)
        self.setLayout(layout)

        model.decoding_changed.connect(self._on_decoding_changed)
        model.sky_changed.connect(self._on_sky_changed)
        model.stream_changed.connect(self._show_statistics)
        self._on_decoding_changed()
        self._show_statistics()
        if self._caster is not None and self._stored_bool(_SETTINGS_NTRIP_ENABLED, False):
            self._caster_check.setChecked(True)
            self._apply_caster(True)

    def _build_output(self) -> QGroupBox:
        """The NTRIP caster: on and off, its port, the position from GGA, and who is
        connected."""
        box = QGroupBox(tr("Correction output (NTRIP)"))
        box.setToolTip(
            tr(
                "Passes the RTCM 3 stream of GEPOS on unchanged over NTRIP, frame by frame "
                "as it comes out of the data groups: to RTKLIB, an RTK receiver or an app "
                "in the same network. The mountpoint is the name of the service (GEPOS). "
                "NTRIP 1.0 and 2.0 are understood; a client that sends no request gets "
                "the stream after a second without a header (plain TCP).\n\nNo login: "
                "meant for your own network only. Windows asks to allow it through the "
                "firewall at the first start.\n\nHardly any receiver understands the "
                "corrections themselves (SSRZ) yet, but every one understands the "
                "ephemerides in the stream."
            )
        )
        self._caster_check = QCheckBox(tr("Caster on"))
        self._caster_check.clicked.connect(self._on_caster_clicked)
        self._port_spin = QSpinBox()
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(int(self._stored(_SETTINGS_NTRIP_PORT, DEFAULT_PORT)))
        self._port_spin.setToolTip(tr("TCP port of the caster; 2101 is the usual one for NTRIP."))
        self._port_spin.editingFinished.connect(self._on_port_changed)
        self._gga_check = QCheckBox(tr("Location from the receiver"))
        self._gga_check.setChecked(self._stored_bool(_SETTINGS_NTRIP_GGA, False))
        self._gga_check.setToolTip(
            tr(
                "Takes the position a connected receiver reports as NMEA GGA (as rovers "
                "do for virtual reference stations) as your own location."
            )
        )
        self._gga_check.toggled.connect(self._on_gga_toggled)
        # Translators: the NTRIP caster is off.
        self._caster_label = QLabel(tr("Off."))
        self._caster_label.setWordWrap(True)
        self._caster_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._caster_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._caster_error: str | None = None
        self._adopted_position_at: float | None = None
        self._auto_requested = False

        row = QHBoxLayout()
        row.addWidget(self._caster_check)
        row.addWidget(QLabel(tr("Port:")))
        row.addWidget(self._port_spin)
        row.addWidget(self._gga_check)
        row.addStretch(1)
        layout = QVBoxLayout(box)
        layout.addLayout(row)
        layout.addWidget(self._caster_label)

        self._caster_timer = QTimer(self)
        self._caster_timer.setInterval(CASTER_STATUS_INTERVAL_MS)
        self._caster_timer.timeout.connect(self._poll_caster)
        box.setVisible(self._caster is not None)
        return box

    # --- Input --------------------------------------------------------------------

    def set_observer(self, observer: Observer) -> None:
        """Shows the listener's position without reporting it back."""
        for spin, value in (
            (self._latitude, observer.latitude),
            (self._longitude, observer.longitude),
            (self._height, observer.height),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def render(self) -> None:
        model = self._model
        now = model.now()
        model.drop_stale(now)
        if (
            not self._dirty
            and self._positions_at is not None
            and abs(now - self._positions_at) < POSITION_INTERVAL_SECONDS
        ):
            return
        if (
            self._dirty
            or self._tracks_at is None
            or abs(now - self._tracks_at) >= TRACK_INTERVAL_SECONDS
        ):
            self._sky.update_tracks(model.tracks(now))
            self._tracks_at = now
        satellites = model.satellites(now)
        self._sky.update_points(satellites)
        self._fill_satellite_table(satellites)
        self._positions_at = now
        self._dirty = False
        self._update_status()

    # --- Internals ----------------------------------------------------------------

    def observer(self) -> Observer:
        """The listener's position as entered here."""
        return Observer(self._latitude.value(), self._longitude.value(), self._height.value())

    def _on_sky_changed(self) -> None:
        self._dirty = True

    def _on_decoding_changed(self) -> None:
        gepos = self._model.gepos
        scid = None if gepos is None else gepos.scid
        if gepos is not None and scid != self._seen_gepos:
            # New in the ensemble: with the caster on, receivers are waiting for it.
            self._auto_requested = False
            if self._caster is not None:
                self._caster.add_mountpoint(mountpoint_for(gepos.scid, gepos.label))
            self._request_for_caster()
        self._seen_gepos = scid
        self._update_status()

    def _show_statistics(self) -> None:
        statistics = self._model.statistics
        if statistics is None:
            self._message_table.setRowCount(0)
            self._stream_label.setText("")
        else:
            fill_table(self._message_table, message_rows(statistics), right_aligned=range(2, 5))
            self._stream_label.setText(describe_stream(statistics))
        self._update_status()

    def _fill_satellite_table(self, satellites: list[SatelliteView]) -> None:
        rows = []
        for satellite in satellites:
            ephemeris = satellite.ephemeris
            rows.append(
                (
                    ephemeris.name,
                    _SYSTEMS[ephemeris.system].name,
                    f"{satellite.elevation:.1f}°",
                    f"{satellite.azimuth:.0f}°",
                    f"{grouped(satellite.distance / 1000)} km",
                    describe_ephemeris(satellite),
                )
            )
        below = {row for row, satellite in enumerate(satellites) if satellite.elevation < 0}
        fill_table(self._satellite_table, rows, right_aligned=range(2, 5), greyed=below)

    def _on_receive_clicked(self) -> None:
        gepos = self._model.gepos
        if gepos is not None:
            self.component_requested.emit(gepos.scid)

    def _on_position_changed(self) -> None:
        self.observer_changed.emit(self.observer())

    def _stored(self, key: str, default: float) -> float:
        if self._settings is None:
            return default
        try:
            return float(self._settings.value(key, default))
        except TypeError, ValueError:
            return default

    def _stored_bool(self, key: str, default: bool) -> bool:
        if self._settings is None:
            return default
        return str(self._settings.value(key, default)).lower() in ("true", "1")

    def _store(self, key: str, value: object) -> None:
        if self._settings is not None:
            self._settings.setValue(key, value)

    # --- NTRIP caster --------------------------------------------------------------

    def _on_caster_clicked(self, checked: bool) -> None:
        self._store(_SETTINGS_NTRIP_ENABLED, checked)
        self._apply_caster(checked)

    def _apply_caster(self, on: bool) -> None:
        caster = self._caster
        if caster is None:
            return
        self._caster_error = None
        if on:
            try:
                caster.start(self._port_spin.value())
            except NtripCasterError as error:
                self._caster_error = str(error)
                self._caster_check.setChecked(False)
            else:
                gepos = self._model.gepos
                if gepos is not None:
                    caster.add_mountpoint(mountpoint_for(gepos.scid, gepos.label))
                self._auto_requested = False
                self._request_for_caster()
                self._caster_timer.start()
        else:
            caster.stop()
            self._caster_timer.stop()
        self._poll_caster()

    def _on_port_changed(self) -> None:
        port = self._port_spin.value()
        self._store(_SETTINGS_NTRIP_PORT, port)
        caster = self._caster
        if caster is not None and caster.running and caster.port != port:
            self._apply_caster(True)

    def _on_gga_toggled(self, checked: bool) -> None:
        self._store(_SETTINGS_NTRIP_GGA, checked)
        self._adopted_position_at = None
        self._poll_caster()

    def _request_for_caster(self) -> None:
        """With the caster on, GEPOS is decoded as soon as the ensemble has it, once:
        stopped by hand, it stays stopped until it appears anew."""
        caster = self._caster
        gepos = self._model.gepos
        if caster is None or not caster.running or gepos is None or self._auto_requested:
            return
        self._auto_requested = True
        decoding = self._model.decoding
        if decoding is None or decoding.scid != gepos.scid:
            self.component_requested.emit(gepos.scid)

    def _poll_caster(self) -> None:
        caster = self._caster
        if caster is None:
            return
        status = caster.status()
        if self._gga_check.isChecked():
            self._adopt_position(status.latest_position)
        self._caster_label.setText(self._describe_caster(status))

    def _adopt_position(self, position: GgaPosition | None) -> None:
        if position is None or position.received_at == self._adopted_position_at:
            return
        self._adopted_position_at = position.received_at
        current = self.observer()
        self.set_observer(
            Observer(
                current.latitude if position.latitude is None else position.latitude,
                current.longitude if position.longitude is None else position.longitude,
                current.height if position.height is None else position.height,
            )
        )
        self._on_position_changed()

    def _describe_caster(self, status: CasterStatus) -> str:
        if not status.running:
            return self._caster_error or tr("Off.")
        lines = [tr("Running on port {port}.").format(port=status.port)]
        decoded = self._model.decoding is not None
        for mountpoint in status.mountpoints:
            if mountpoint.bytes_published:
                types = ", ".join(str(t) for t in mountpoint.message_types)
                lines.append(
                    tr("Mountpoint {name}: {size} passed on, messages {types}.").format(
                        name=mountpoint.name,
                        size=_kilobytes(mountpoint.bytes_published),
                        types=types,
                    )
                )
            else:
                lines.append(tr("Mountpoint {name}: no frames yet.").format(name=mountpoint.name))
        if not status.mountpoints:
            lines.append(tr("No mountpoint yet: the ensemble has no GEPOS."))
        elif not decoded:
            lines.append(tr("GEPOS is not being decoded; connected receivers get nothing."))
        if not status.clients:
            lines.append(tr("No receiver connected."))
        for client in status.clients:
            position = client.position
            if position is None:
                text = tr("• {address}, {protocol} on {mountpoint}, since {since}, {size}")
                quality = height = latitude = longitude = ""
            else:
                text = tr(
                    "• {address}, {protocol} on {mountpoint}, since {since}, {size}, "
                    "reports {latitude}° N {longitude}° E{height} ({quality})"
                )
                quality = _GGA_QUALITY.get(position.quality)
                quality = (
                    tr("quality {number}").format(number=position.quality)
                    if quality is None
                    else tr(quality)
                )
                height = "" if position.height is None else f", {position.height:.0f} m"
                latitude = decimal(position.latitude, ".5f")
                longitude = decimal(position.longitude, ".5f")
            lines.append(
                text.format(
                    address=client.address,
                    protocol=client.protocol,
                    mountpoint=client.mountpoint,
                    since=time.strftime("%H:%M:%S", time.localtime(client.connected_at)),
                    size=_kilobytes(client.bytes_sent),
                    latitude=latitude,
                    longitude=longitude,
                    height=height,
                    quality=quality,
                )
            )
        return "\n".join(lines)

    def _update_status(self) -> None:
        model = self._model
        gepos, decoding = model.gepos, model.decoding
        self._receive_button.setVisible(gepos is not None and not model.gepos_decoded)
        parts = []
        if decoding is not None:
            parts.append(tr("Decoding “{service}”.").format(service=decoding.label))
            statistics = model.statistics
            if statistics is not None and statistics.data_groups and not statistics.frames:
                parts.append(tr("No RTCM messages in the data stream."))
        elif gepos is not None:
            parts.append(tr("GEPOS is not being decoded."))
        else:
            parts.append(
                tr(
                    "No GEPOS in the ensemble; it is broadcast nationwide on channel 5C "
                    "(DR Deutschland)."
                )
            )
        count = model.ephemeris_count
        if count:
            parts.append(
                tr_plural(
                    "Ephemeris of {count} satellite.", "Ephemerides of {count} satellites.", count
                ).format(count=count)
            )
        now = time.strftime("%H:%M:%S", time.gmtime(model.now()))
        if model.clock_synchronised:
            parts.append(tr("Time: {time} UTC, clock of the ensemble (FIG 0/10).").format(time=now))
        else:
            parts.append(tr("Time: {time} UTC, computer clock.").format(time=now))
        self._status_label.setText(" ".join(parts))

    @staticmethod
    def _coordinate_spin(limits: tuple[float, float], suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(*limits)
        spin.setDecimals(4)
        spin.setSingleStep(0.1)
        spin.setSuffix(suffix)
        return spin
