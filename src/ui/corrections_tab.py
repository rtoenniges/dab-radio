"""The "Corrections" tab under "GNSS": what the SSRZ corrections of GEPOS mean for a receiver at the
listener's position, per satellite and error source, and how they develop (evaluated by
`ui.gnss_model.GnssModel`)."""

from __future__ import annotations

import math
import time

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dab.gnss import GPS_EPOCH_POSIX, GPS_UTC_LEAP_SECONDS
from i18n import tr, tr_noop, tr_plural
from ui.atmosphere_model import decimal
from ui.gnss_model import CorrectionHistory, GnssModel
from ui.tables import fill_table, make_table

Colour = tuple[int, int, int]
# The first slots of the categorical palette in their dark steps, as in the "Satellites" tab.
_BLUE: Colour = (0x39, 0x87, 0xE5)
_ORANGE: Colour = (0xD9, 0x59, 0x26)
_AQUA: Colour = (0x19, 0x9E, 0x70)
_GREY: Colour = (0xC3, 0xC2, 0xB7)
_TEXT: Colour = (220, 220, 220)

_HEADERS = [
    tr_noop("Satellite"),
    tr_noop("Elevation"),
    "IOD",
    tr_noop("Clock δC"),
    # Translators: the orbit correction radial, along-track and cross-track.
    tr_noop("Orbit R / A / C"),
    tr_noop("Line of sight"),
    tr_noop("Ionosphere"),
    tr_noop("Troposphere"),
    tr_noop("Code biases"),
    tr_noop("Phase biases"),
]

# The lines of the plots of a satellite's course, by the names in their legends.
_CLOCK = tr_noop("δC with fine correction")
_CLOCK_LOW_RATE = tr_noop("ZM002 only (every 30 s)")
_RADIAL = tr_noop("radial")
# Translators: the orbit correction along the orbit (along-track).
_ALONG = tr_noop("along")
# Translators: the orbit correction across the orbit (cross-track).
_CROSS = tr_noop("cross")
_LINE_OF_SIGHT = tr_noop("line of sight")
_IONOSPHERE = tr_noop("Ionosphere")
_TROPOSPHERE = tr_noop("Troposphere")


def decimal_text(value: float, spec: str) -> str:
    """A number with the decimal separator of the language and a true minus; NaN as a
    dash."""
    if value is None or not math.isfinite(value):
        return "–"
    return decimal(value, spec).replace("-", "−")


def _signals(values: dict[str, float], spec: str, skip_zero: bool = False) -> str:
    parts = [
        f"{signal} {decimal_text(value, spec)}"
        for signal, value in values.items()
        if not (skip_zero and value == 0.0)
    ]
    return " · ".join(parts) if parts else "–"


class HistoryPlot:
    """A small time plot of one satellite with a line per quantity; `lines` are named
    by texts marked with `tr_noop`, their legend shows them translated."""

    def __init__(
        self,
        title: str,
        unit: str,
        lines: list[tuple[str, Colour, bool]],
        text: Colour = _TEXT,
        legend_background: tuple[int, int, int, int] = (0, 0, 0, 170),
        legend_columns: int = 1,
        legend_text_size: str = "9pt",
    ) -> None:
        self.widget = pg.PlotWidget(title=title)
        self.widget.showGrid(x=True, y=True, alpha=0.25)
        self.widget.setLabel("left", unit)
        self.widget.setLabel("bottom", tr("Minutes until now"))
        self.widget.setMouseEnabled(x=False, y=False)
        self.widget.hideButtons()
        self.legend = self.widget.addLegend(
            offset=(5, 5),
            labelTextColor=text,
            labelTextSize=legend_text_size,
            brush=pg.mkBrush(*legend_background),
            colCount=legend_columns,
        )
        self.items = {}
        for name, colour, dashed in lines:
            pen = pg.mkPen(
                colour, width=2, style=Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
            )
            self.items[name] = self.widget.plot([], [], pen=pen, name=tr(name), stepMode=None)
        self.legend.setVisible(True)

    def show(self, minutes: np.ndarray, values: dict[str, np.ndarray]) -> None:
        for name, item in self.items.items():
            item.setData(minutes, values[name], connect="finite")

    def clear(self) -> None:
        empty = np.array([])
        self.show(empty, {name: empty for name in self.items})


def clock_plot(**style: object) -> HistoryPlot:
    """Clock correction with and without the fine correction every 5 s."""
    return HistoryPlot(
        tr("Clock correction"),
        "cm",
        [(_CLOCK, _BLUE, False), (_CLOCK_LOW_RATE, _GREY, True)],
        **style,
    )


def orbit_plot(**style: object) -> HistoryPlot:
    """Orbit correction in its three directions and towards the listener."""
    return HistoryPlot(
        tr("Orbit correction"),
        "cm",
        [
            (_RADIAL, _BLUE, False),
            (_ALONG, _ORANGE, False),
            (_CROSS, _AQUA, False),
            (_LINE_OF_SIGHT, _GREY, True),
        ],
        **style,
    )


def atmosphere_plot(**style: object) -> HistoryPlot:
    """Delay by ionosphere and troposphere on the slant path."""
    return HistoryPlot(
        tr("Atmosphere on the slant path"),
        "m",
        [(_IONOSPHERE, _BLUE, False), (_TROPOSPHERE, _ORANGE, False)],
        **style,
    )


def show_history(
    history: CorrectionHistory | None,
    latest_time: float | None,
    clock: HistoryPlot,
    orbit: HistoryPlot,
    atmosphere: HistoryPlot,
) -> None:
    """Draws the course of one satellite into the three plots, in minutes until the
    latest epoch."""
    if history is None or not history.samples or latest_time is None:
        for plot in (clock, orbit, atmosphere):
            plot.clear()
        return
    minutes = (history.column("time") - latest_time) / 60.0
    positions = np.array([s.orbit for s in history.samples], dtype=float) * 100.0
    clock.show(
        minutes,
        {
            _CLOCK: history.column("clock") * 100.0,
            _CLOCK_LOW_RATE: history.column("clock_low_rate") * 100.0,
        },
    )
    orbit.show(
        minutes,
        {
            _RADIAL: positions[:, 0],
            _ALONG: positions[:, 1],
            _CROSS: positions[:, 2],
            _LINE_OF_SIGHT: history.column("orbit_range") * 100.0,
        },
    )
    atmosphere.show(
        minutes,
        {
            _IONOSPHERE: history.column("ionosphere"),
            _TROPOSPHERE: history.column("troposphere"),
        },
    )


class CorrectionsTab(QWidget):
    """Top: a table with every satellite of the corrections and what each error source
    amounts to at the listener's position (set in the "Satellites" tab); below: the course of
    clock, orbit and atmosphere of the satellite chosen in the table.

    Shows what the `GnssModel` evaluates while GEPOS is decoded: the corrections with
    every high rate message, kept for half an hour. `render()` draws, called by the main
    window while this tab is visible.
    """

    component_requested = pyqtSignal(object)  # SCId of the GEPOS component to decode

    def __init__(self, model: GnssModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._selected: str | None = None
        self._dirty = True

        self._receive_button = QPushButton(tr("Receive GEPOS"))
        self._receive_button.setToolTip(
            tr(
                "Decodes the data service GEPOS of the BKG in addition to the audio. Its "
                "SSRZ messages carry the corrections, the ephemerides in the same stream "
                "the orbits they refer to."
            )
        )
        self._receive_button.clicked.connect(self._on_receive_clicked)
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)

        self._table = make_table([tr(header) for header in _HEADERS])
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.cellClicked.connect(self._on_row_clicked)
        self._table.setToolTip(
            tr(
                "What the corrections of GEPOS mean for a receiver at the location set, "
                "per satellite (by elevation, grey below the horizon):\n\n"
                "• IOD: the ephemeris that clock and orbit refer to. Without this "
                "ephemeris the satellite is missing.\n"
                "• Clock δC: correction of the satellite clock from the ephemeris, in "
                "metres (light path). Every 30 s a value to 1 cm (ZM002), every 5 s a fine "
                "correction to 0.1 mm (ZM001).\n"
                "• Orbit R / A / C: correction of the satellite position radially, along "
                "the orbit and across it. Line of sight: how much of it changes the "
                "distance to the location (positive: the satellite is farther away).\n"
                "• Ionosphere: electrons on the slant path (TECU, the sum of the global, "
                "regional and grid stages) and the delay they cause on the reference "
                "signal (L1, G1, E1 or B1).\n"
                "• Troposphere: delay on the slant path from the model, in brackets what "
                "the scale factors sent add to it.\n"
                "• Code biases in metres per signal (the reference signal has 0), phase "
                "biases in cycles, sent modulo 10 cycles.\n\n"
                "Click a row to show its course below."
            )
        )

        self._clock_plot = clock_plot()
        self._clock_plot.widget.setToolTip(
            tr(
                "The correction of the satellite clock over the last half hour. Dashed "
                "grey the value that comes every 30 s (ZM002, to 1 cm), blue with the fine "
                "correction every 5 s (ZM001), which follows the clock between the 30 s "
                "values."
            )
        )
        self._orbit_plot = orbit_plot()
        self._orbit_plot.widget.setToolTip(
            tr(
                "The correction of the satellite position against the ephemeris, split "
                "up along the orbit of the satellite, and dashed its part in the direction "
                "of the location. A new ephemeris (another IOD) starts the correction anew."
            )
        )
        self._atmosphere_plot = atmosphere_plot()
        self._atmosphere_plot.widget.setToolTip(
            tr(
                "Delay by the ionosphere (on the reference signal) and the troposphere on "
                "the way from the satellite to the location. Both grow the lower the "
                "satellite stands."
            )
        )
        self._orbit_plot.widget.setXLink(self._clock_plot.widget)
        self._atmosphere_plot.widget.setXLink(self._clock_plot.widget)

        controls = QHBoxLayout()
        controls.addWidget(self._receive_button)
        controls.addWidget(self._status_label, stretch=1)
        plots = QHBoxLayout()
        plots.addWidget(self._clock_plot.widget)
        plots.addWidget(self._orbit_plot.widget)
        plots.addWidget(self._atmosphere_plot.widget)
        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self._table, stretch=3)
        layout.addLayout(plots, stretch=2)
        self.setLayout(layout)

        model.corrections_changed.connect(self._on_corrections_changed)
        model.decoding_changed.connect(self._update_status)
        self._on_corrections_changed()

    def render(self) -> None:
        self._update_status()
        if not self._dirty:
            return
        self._dirty = False
        self._fill_table()
        self._draw_history()

    def _on_corrections_changed(self) -> None:
        corrections = self._model.corrections
        if self._selected is None and corrections:
            self._selected = corrections[0].name
        self._dirty = True
        self._update_status()

    # --- Drawing -------------------------------------------------------------------

    def _fill_table(self) -> None:
        rows = []
        below = set()
        corrections = self._model.corrections
        for index, c in enumerate(corrections):
            if c.elevation < 0:
                below.add(index)
            radial, along, cross = (decimal_text(100 * v, "+.1f") for v in c.orbit)
            parts = c.troposphere_correction
            rows.append(
                (
                    c.name,
                    f"{decimal_text(c.elevation, '.1f')}°",
                    str(c.ephemeris_iod),
                    f"{decimal_text(c.clock, '+.3f')} m",
                    f"{radial} / {along} / {cross} cm",
                    f"{decimal_text(100 * c.orbit_range, '+.1f')} cm",
                    f"{decimal_text(c.total_stec, '.1f')} TECU · {decimal_text(c.ionosphere, '.2f')} m",
                    f"{decimal_text(c.troposphere, '.2f')} m ({decimal_text(parts, '+.2f')})",
                    _signals(c.code_biases, "+.2f", skip_zero=True),
                    _signals(c.phase_biases, ".2f"),
                )
            )
        fill_table(self._table, rows, right_aligned=range(1, 8), greyed=below)
        names = [c.name for c in corrections]
        if self._selected in names:
            self._table.selectRow(names.index(self._selected))

    def _draw_history(self) -> None:
        if self._selected is not None:
            selected = self._selected
            self._clock_plot.widget.setTitle(
                tr("Clock correction {satellite}").format(satellite=selected)
            )
            self._orbit_plot.widget.setTitle(
                tr("Orbit correction {satellite}").format(satellite=selected)
            )
            self._atmosphere_plot.widget.setTitle(
                tr("Atmosphere {satellite}").format(satellite=selected)
            )
        show_history(
            None if self._selected is None else self._model.history(self._selected),
            self._model.corrections_time,
            self._clock_plot,
            self._orbit_plot,
            self._atmosphere_plot,
        )

    # --- Status --------------------------------------------------------------------

    def _on_row_clicked(self, row: int, _column: int) -> None:
        item = self._table.item(row, 0)
        if item is not None and item.text() != self._selected:
            self._selected = item.text()
            self._draw_history()

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
        o = model.observer
        parts.append(
            tr("Location {latitude}° N {longitude}° E, {height} m (“Satellites” tab).").format(
                latitude=decimal_text(o.latitude, ".4f"),
                longitude=decimal_text(o.longitude, ".4f"),
                height=decimal_text(o.height, ".0f"),
            )
        )
        latest_time = model.corrections_time
        if latest_time is not None:
            posix = latest_time - GPS_UTC_LEAP_SECONDS + GPS_EPOCH_POSIX
            count = len(model.corrections)
            parts.append(
                tr_plural(
                    "Epoch {time} UTC: {count} satellite.",
                    "Epoch {time} UTC: {count} satellites.",
                    count,
                ).format(time=time.strftime("%H:%M:%S", time.gmtime(posix)), count=count)
            )
            missing = model.missing_ephemerides()
            if missing:
                parts.append(
                    tr("Still without a matching ephemeris: {satellites}.").format(
                        satellites=", ".join(missing)
                    )
                )
        elif decoding is not None:
            parts.append(tr("Waiting for metadata, corrections and ephemerides (up to 30 s)."))
        self._status_label.setText(" ".join(parts))
