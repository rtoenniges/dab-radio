"""The "GNSS" menu of the services page: what GEPOS carries, on five pages to swipe
through (`ui.touch.pager`): the sky, the satellites in space (`ui.touch.globe_view`), the
atmosphere, the corrections at the listener's position and the RTCM stream itself.
Everything comes from `ui.gnss_model.GnssModel`, which also feeds the desktop's tabs
under "GNSS"; the NTRIP caster is left to the desktop.

The satellites in space can also be shown on the radio page in the place of the
slideshow (`GlobePanel`, chosen in `ui.touch.display_choice`)."""

from __future__ import annotations

import math
import time

from PyQt6.QtCore import QItemSelection, QItemSelectionModel, QSettings
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from dab.gnss import GPS_EPOCH_POSIX, GPS_UTC_LEAP_SECONDS, Observer
from i18n import tr, tr_noop, tr_plural
from ui.atmosphere_model import (
    IONOSPHERE_GLOBAL,
    REDRAW_SECONDS,
    TROPOSPHERE_VIEW_FACTOR,
    TROPOSPHERE_VIEW_TOTAL,
    TROPOSPHERE_VIEW_WET,
)
from ui.atmosphere_tab import MapPanel
from ui.corrections_tab import (
    HistoryPlot,
    atmosphere_plot,
    clock_plot,
    decimal_text,
    orbit_plot,
    show_history,
)
from ui.globe_gl import CAMERA_DISTANCE_KM, GlobeRenderThread
from ui.gnss_model import (
    HISTORY_MINUTES,
    POSITION_INTERVAL_SECONDS,
    TRACK_INTERVAL_SECONDS,
    GnssModel,
    SatelliteOrbit,
    SatelliteView,
)
from ui.gnss_tab import (
    MESSAGE_HEADERS,
    SkyColours,
    SkyPlot,
    describe_ephemeris,
    describe_stream,
    legend_text,
    message_rows,
)
from ui.receiver_controller import ReceiverController
from ui.tables import fill_table
from ui.touch.globe_view import GlobeView
from ui.touch.pager import Pager
from ui.touch.pages import (
    Page,
    SwitchedOffNote,
    pixel_font,
    rgb,
    style_axes,
    touch_table,
)
from ui.touch.signal_panels import SignalPanel
from ui.touch.theme import DARK, PANEL_RADIUS, SMALL_FONT_PX, THEMES, Theme, style_plot
from ui.touch.widgets import ChoiceButtons, Stepper, set_role

# The pages in the order of swiping.
SKY = "sky"
GLOBE = "globe"
ATMOSPHERE = "atmosphere"
CORRECTIONS = "corrections"
STREAM = "stream"
PAGES = (SKY, GLOBE, ATMOSPHERE, CORRECTIONS, STREAM)

SKY_POINT_SIZE = 16
# The satellites in space are placed anew this often: in that time one moves by 20 km,
# half a pixel in the first view and two at the closest. Their orbits with them cost
# 22 ms for 69 satellites on the PC (measured 07.10.2026), the sky's positions 9 ms.
GLOBE_INTERVAL_SECONDS = 5.0
# On the radio page the view is as high as on the page "3D view" (480 px against
# 468) but only 4:3 instead of 2.4:1: the space beside the Earth is cut off. It begins
# at the distance the user brought it to (2026-10-13), farther than on that page; until
# then an eighth nearer than the page, 56 000 km. Brought nearer or farther, the view
# keeps its distance under this key of the settings, also for the next start.
RADIO_GLOBE_DISTANCE_KM = 97_750.0
SETTINGS_RADIO_GLOBE_DISTANCE = "touch/radio_globe_distance_km"
MAP_POINT_SIZE = 12
# Room for the figures of a map's colour scale, and the size of the legend of a course.
SCALE_WIDTH = 64
LEGEND_PX = 15
# The ionosphere map: the global model, or the gridded residual of the satellite chosen.
_GRID = "grid"
_NO_GEPOS = tr_noop("No GEPOS in the ensemble.")


class _SkyPage(Page):
    """The sky plot on the left, the satellites by elevation on the right."""

    def __init__(self, model: GnssModel) -> None:
        super().__init__(tr("Sky"))
        self._model = model
        self._dirty = True
        self._positions_at: float | None = None
        self._tracks_at: float | None = None
        self.sky = SkyPlot(
            title=None,
            font=pixel_font(SMALL_FONT_PX),
            point_size=SKY_POINT_SIZE,
            hover_tips=False,
        )
        self.table = touch_table([tr("Satellite"), tr("Elevation"), tr("Azimuth"), tr("Ephemeris")])
        self.legend = set_role(QLabel(), "small")
        self.legend.setWordWrap(True)
        self.legend.setText(
            legend_text() + "<br>" + tr("○&nbsp;flagged unhealthy  —&nbsp;next hour")
        )
        side = QVBoxLayout()
        side.setSpacing(8)
        side.addWidget(self.table, stretch=1)
        side.addWidget(self.legend)
        content = QHBoxLayout()
        content.setSpacing(24)
        # The plot keeps the sky round: as wide as it is high, at most half the page.
        content.addWidget(self.sky.plot, stretch=1)
        content.addLayout(side, stretch=1)
        self.body.addLayout(content, stretch=1)
        model.sky_changed.connect(self._on_changed)

    def _on_changed(self) -> None:
        self._dirty = True

    def apply_theme(self, theme: Theme) -> None:
        style_plot(self.sky.plot, theme)
        self.sky.set_colours(
            SkyColours(
                grid=rgb(theme.outline),
                grid_text=rgb(theme.muted),
                label_text=rgb(theme.text),
                hollow=rgb(theme.surface),
            )
        )
        self._dirty = True

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
            self.sky.update_tracks(model.tracks(now))
            self._tracks_at = now
        satellites = model.satellites(now)
        self.sky.update_points(satellites)
        self._fill_table(satellites)
        self._positions_at = now
        self._dirty = False
        above = sum(s.elevation >= 0 for s in satellites)
        if satellites:
            count = len(satellites)
            self.status.setText(
                tr_plural(
                    "{above} of {count} satellite above the horizon",
                    "{above} of {count} satellites above the horizon",
                    count,
                ).format(above=above, count=count)
            )
        elif model.gepos is None:
            self.status.setText(tr(_NO_GEPOS))
        else:
            self.status.setText(tr("Waiting for ephemerides …"))

    def _fill_table(self, satellites: list[SatelliteView]) -> None:
        rows = []
        for satellite in satellites:
            rows.append(
                (
                    satellite.ephemeris.name,
                    f"{satellite.elevation:.0f}°",
                    f"{satellite.azimuth:.0f}°",
                    describe_ephemeris(satellite),
                )
            )
        below = {row for row, satellite in enumerate(satellites) if satellite.elevation < 0}
        fill_table(self.table, rows, right_aligned=range(1, 3), greyed=below)


class GlobeFeed:
    """Keeps a globe view supplied from the model: turns it to the listener's position
    when that changes, and places the satellites anew every `GLOBE_INTERVAL_SECONDS`, at
    once when an ephemeris came or went."""

    def __init__(self, model: GnssModel, view: GlobeView) -> None:
        self._model = model
        self._view = view
        self._dirty = True
        self._drawn_at: float | None = None
        self._observer: Observer | None = None
        model.sky_changed.connect(self.invalidate)
        view.prepared.connect(self.invalidate)

    def invalidate(self) -> None:
        """The satellites are placed anew at the next `render`."""
        self._dirty = True

    def render(self) -> list[SatelliteOrbit] | None:
        """The satellites now placed in the view; None if nothing was due."""
        model = self._model
        now = model.now()
        model.drop_stale(now)
        observer = model.observer
        if observer != self._observer:
            # A new position is entered on the settings page: the view turns to it.
            self._observer = observer
            self._view.look_at(observer.latitude, observer.longitude)
            self._dirty = True
        if (
            not self._dirty
            and self._drawn_at is not None
            and abs(now - self._drawn_at) < GLOBE_INTERVAL_SECONDS
        ):
            return None
        orbits = model.orbits(now)
        self._view.set_scene(orbits, observer.ecef() / 1000)
        self._drawn_at = now
        self._dirty = False
        return orbits


def globe_status(model: GnssModel, view: GlobeView, orbits: list[SatelliteOrbit]) -> str:
    """What the view shows, or why it shows nothing."""
    if not view.available:
        return tr("No 3D view without OpenGL")
    if orbits:
        above = sum(s.elevation >= 0 for s in orbits)
        return tr_plural(
            "{count} satellite, {above} above the horizon",
            "{count} satellites, {above} above the horizon",
            len(orbits),
        ).format(count=len(orbits), above=above)
    if model.gepos is None:
        return tr(_NO_GEPOS)
    return tr("Waiting for ephemerides …")


class _GlobePage(Page):
    """The Earth with the listener's position, the satellites in space and their orbits,
    to be turned, zoomed and shifted with fingers."""

    def __init__(self, model: GnssModel, renderer: GlobeRenderThread | None = None) -> None:
        super().__init__(tr("3D view"))
        self._model = model
        self.view = GlobeView(renderer)
        self.feed = GlobeFeed(model, self.view)
        self.legend = set_role(QLabel(), "small")
        self.legend.setWordWrap(True)
        self.legend.setText(
            legend_text() + " · " + tr("○&nbsp;flagged unhealthy · pale:&nbsp;below the horizon")
        )
        # Beside the legend, as on the traffic map: the view keeps its height.
        self.reset_button = QPushButton(tr("Reset view"))
        self.reset_button.clicked.connect(self.view.reset_view)
        bottom = QHBoxLayout()
        bottom.setSpacing(16)
        bottom.addWidget(self.legend, stretch=1)
        bottom.addWidget(self.reset_button)
        self.body.addWidget(self.view, stretch=1)
        self.body.addLayout(bottom)

    def render(self) -> None:
        orbits = self.feed.render()
        if orbits is not None:
            self.status.setText(globe_status(self._model, self.view, orbits))


class GlobePanel(SignalPanel):
    """The satellites in space on the radio page, in the place of the slideshow: the
    first view of the page "3D view", cut to the shape of the slide and from
    `RADIO_GLOBE_DISTANCE_KM`, with the corners of a panel. What that page says in its
    header stands in the lower left corner. Two fingers bring it nearer or farther
    (`zoom`, from the slide area of `ui.touch.radio_page`, which keeps one finger for
    its long press); it is not turned or shifted, neither by fingers nor by the mouse
    (wish of the user). A new listener's position turns it there; the distance stays,
    with `settings` also for the next start (`SETTINGS_RADIO_GLOBE_DISTANCE`)."""

    def __init__(
        self,
        model: GnssModel,
        renderer: GlobeRenderThread | None = None,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__(tr("3D view"))
        self.heading.hide()  # the view fills the panel
        self.body.setContentsMargins(0, 0, 0, 0)
        self._model = model
        self._settings = settings
        self.view = GlobeView(renderer, distance_km=self._stored_distance())
        self.feed = GlobeFeed(model, self.view)
        self.body.addWidget(self.view, stretch=1)
        self.apply_theme(THEMES[DARK])

    def apply_theme(self, theme: Theme) -> None:
        # Space stays dark in both schemes; only the page around it changes.
        self.view.set_corners(PANEL_RADIUS, theme.color("background"))

    def _stored_distance(self) -> float:
        """The distance the view was last brought to, within that of the camera."""
        if self._settings is None:
            return RADIO_GLOBE_DISTANCE_KM
        try:
            distance = self._settings.value(
                SETTINGS_RADIO_GLOBE_DISTANCE, RADIO_GLOBE_DISTANCE_KM, type=float
            )
        except TypeError, ValueError:  # edited by hand
            return RADIO_GLOBE_DISTANCE_KM
        if not math.isfinite(distance):
            return RADIO_GLOBE_DISTANCE_KM
        low, high = CAMERA_DISTANCE_KM
        return min(max(distance, low), high)

    def zoom(self, factor: float) -> None:
        self.view.zoom(factor)
        self.view.keep_distance()
        if self._settings is not None:
            self._settings.setValue(SETTINGS_RADIO_GLOBE_DISTANCE, self.view.camera.distance_km)

    def render(self) -> None:
        orbits = self.feed.render()
        if orbits is not None:
            self.view.set_caption(globe_status(self._model, self.view, orbits))


class _AtmospherePage(Page):
    """Two maps side by side, each with chips for what it shows."""

    def __init__(self, model: GnssModel) -> None:
        super().__init__(tr("Atmosphere"))
        self._model = model
        self._view_changed = True
        self._data_changed = False
        self._drawn_at: float | None = None

        self.ionosphere_view = ChoiceButtons(
            [(IONOSPHERE_GLOBAL, tr("Global VTEC")), (_GRID, tr("Grid"))], role="chip"
        )
        self.ionosphere_view.set_current(IONOSPHERE_GLOBAL)
        self.ionosphere_view.chosen.connect(self._on_view_changed)
        # The satellite whose gridded residual the map shows.
        self.satellite = Stepper([], value_width=64, symbols=("‹", "›"))
        self.satellite.value_changed.connect(self._on_view_changed)
        self.troposphere_view = ChoiceButtons(
            [
                # Translators: what the map of the troposphere shows, the wet or the
                # total delay, or the wet scale factor.
                (TROPOSPHERE_VIEW_WET, tr("wet")),
                (TROPOSPHERE_VIEW_TOTAL, tr("total")),
                (TROPOSPHERE_VIEW_FACTOR, tr("factor")),
            ],
            role="chip",
        )
        self.troposphere_view.set_current(TROPOSPHERE_VIEW_WET)
        self.troposphere_view.chosen.connect(self._on_view_changed)

        self.ionosphere_map = MapPanel(
            None, font=pixel_font(SMALL_FONT_PX), point_size=MAP_POINT_SIZE, hover_tips=False
        )
        self.troposphere_map = MapPanel(
            None, font=pixel_font(SMALL_FONT_PX), point_size=MAP_POINT_SIZE, hover_tips=False
        )
        columns = QHBoxLayout()
        columns.setSpacing(24)
        for chips, panel, extra in (
            (self.ionosphere_view, self.ionosphere_map, self.satellite),
            (self.troposphere_view, self.troposphere_map, None),
        ):
            panel.plot.hideAxis("left")
            panel.plot.hideAxis("bottom")
            row = QHBoxLayout()
            row.setSpacing(12)
            row.addWidget(chips)
            if extra is not None:
                row.addWidget(extra)
            row.addStretch()
            column = QVBoxLayout()
            column.setSpacing(8)
            column.addLayout(row)
            column.addWidget(panel.plot, stretch=1)
            columns.addLayout(column, stretch=1)
        self.body.addLayout(columns, stretch=1)
        self._show_satellite_choice()
        model.atmosphere_changed.connect(self._on_data_changed)

    def _on_data_changed(self) -> None:
        self._data_changed = True

    def _on_view_changed(self, *_args: object) -> None:
        self._show_satellite_choice()
        self._view_changed = True

    def _show_satellite_choice(self) -> None:
        self.satellite.setVisible(self.ionosphere_view.current() == _GRID)

    def apply_theme(self, theme: Theme) -> None:
        for panel in (self.ionosphere_map, self.troposphere_map):
            style_plot(panel.plot, theme)
            panel.set_city_colour(theme.color("muted"), theme.color("surface"))
            panel.set_scale_style(pixel_font(SMALL_FONT_PX), SCALE_WIDTH, theme.color("muted"))
        self._view_changed = True

    def render(self) -> None:
        model = self._model
        atmosphere = model.atmosphere
        now = model.now()
        self._show_status(now)
        monotonic = time.monotonic()
        due = self._data_changed and (
            self._drawn_at is None or monotonic - self._drawn_at >= REDRAW_SECONDS
        )
        if not (self._view_changed or due):
            return
        self._drawn_at = monotonic
        self._data_changed = self._view_changed = False
        self.satellite.set_values(atmosphere.ionosphere_satellites())
        if self.ionosphere_view.current() == _GRID:
            satellite = self.satellite.value()
            ionosphere = None if satellite is None else atmosphere.ionosphere_field(satellite, now)
        else:
            ionosphere = atmosphere.ionosphere_field(IONOSPHERE_GLOBAL, now)
        troposphere = atmosphere.troposphere_field(self.troposphere_view.current(), now)
        self.ionosphere_map.show(ionosphere, atmosphere.stations)
        self.troposphere_map.show(troposphere, atmosphere.stations)

    def _show_status(self, now: float) -> None:
        model = self._model
        received = model.atmosphere.received
        if received is not None:
            if model.atmosphere.is_stale(now):
                text = tr("Corrections out of date (received {minutes:.0f} min ago)").format(
                    minutes=(now - received) / 60
                )
            else:
                text = tr("Corrections up to date (new every 30 s)")
        elif model.gepos is None:
            text = tr(_NO_GEPOS)
        else:
            text = tr("Waiting for metadata and grids (every 30 s) …")
        self.status.setText(text)


class _CorrectionsPage(Page):
    """The satellites with what each error source amounts to at the listener's position
    on the left; the course of the one tapped on the right."""

    def __init__(self, model: GnssModel) -> None:
        super().__init__(tr("Corrections at the location"))
        self._model = model
        self._dirty = True
        self.selected: str | None = None
        self.table = touch_table(
            [
                tr("Satellite"),
                # Translators: short for elevation, a narrow column.
                tr("Elev."),
                tr("Clock (m)"),
                tr("Orbit (cm)"),
                # Translators: short for the ionosphere, a narrow column.
                tr("Iono (m)"),
                # Translators: short for the troposphere, a narrow column.
                tr("Tropo (m)"),
            ]
        )
        self.table.cellClicked.connect(self._on_row_clicked)
        self.course = set_role(QLabel(), "muted")
        style = {"legend_background": (0, 0, 0, 0), "legend_text_size": f"{LEGEND_PX}px"}
        self.plots: tuple[HistoryPlot, HistoryPlot, HistoryPlot] = (
            clock_plot(legend_columns=2, **style),
            orbit_plot(legend_columns=4, **style),
            atmosphere_plot(legend_columns=2, **style),
        )
        side = QVBoxLayout()
        side.setSpacing(8)
        side.addWidget(self.course)
        for plot, title in zip(self.plots, (tr("Clock"), tr("Orbit"), tr("Atmosphere"))):
            plot.widget.setTitle(None)
            plot.widget.getAxis("bottom").showLabel(False)
            # Always the whole half hour, also while the course is short.
            plot.widget.setXRange(-HISTORY_MINUTES, 0, padding=0.02)
            plot.widget.enableAutoRange(x=False)
            plot.widget.setLabel("left", f"{title} ({plot.widget.getAxis('left').labelText})")
            side.addWidget(plot.widget, stretch=1)
        content = QHBoxLayout()
        content.setSpacing(24)
        content.addWidget(self.table, stretch=11)
        content.addLayout(side, stretch=9)
        self.body.addLayout(content, stretch=1)
        model.corrections_changed.connect(self._on_changed)

    def _on_changed(self) -> None:
        corrections = self._model.corrections
        if self.selected is None and corrections:
            self.selected = corrections[0].name
        self._dirty = True

    def apply_theme(self, theme: Theme) -> None:
        for plot in self.plots:
            style_plot(plot.widget, theme)
            style_axes(plot.widget)
            plot.legend.setLabelTextColor(theme.text)
            # The labels take the colour only when set anew.
            for _sample, label in plot.legend.items:
                label.setText(label.text, color=theme.text, size=f"{LEGEND_PX}px")
        self._dirty = True

    def render(self) -> None:
        model = self._model
        latest_time = model.corrections_time
        if latest_time is not None:
            posix = latest_time - GPS_UTC_LEAP_SECONDS + GPS_EPOCH_POSIX
            count = len(model.corrections)
            self.status.setText(
                tr_plural(
                    "Epoch {time} UTC · {count} satellite",
                    "Epoch {time} UTC · {count} satellites",
                    count,
                ).format(time=time.strftime("%H:%M:%S", time.gmtime(posix)), count=count)
            )
        elif model.gepos is None:
            self.status.setText(tr(_NO_GEPOS))
        else:
            self.status.setText(tr("Waiting for corrections and ephemerides (up to 30 s) …"))
        if not self._dirty:
            return
        self._dirty = False
        self._fill_table()
        self.course.setText(
            tr("Tap a satellite.")
            if self.selected is None
            else tr("Course of {satellite}, last 30 minutes").format(satellite=self.selected)
        )
        show_history(
            None if self.selected is None else model.history(self.selected),
            latest_time,
            *self.plots,
        )

    def _fill_table(self) -> None:
        corrections = self._model.corrections
        rows = [
            (
                c.name,
                f"{decimal_text(c.elevation, '.0f')}°",
                decimal_text(c.clock, "+.3f"),
                decimal_text(100 * c.orbit_range, "+.1f"),
                decimal_text(c.ionosphere, ".2f"),
                decimal_text(c.troposphere, ".2f"),
            )
            for c in corrections
        ]
        below = {index for index, c in enumerate(corrections) if c.elevation < 0}
        fill_table(self.table, rows, right_aligned=range(1, 6), greyed=below)
        # The satellite whose course is shown is selected. The table selects nothing
        # itself (a finger scrolling it comes down on some row), only from here.
        selection = self.table.selectionModel()
        names = [c.name for c in corrections]
        if self.selected in names:
            row = names.index(self.selected)
            model = self.table.model()
            selection.select(
                QItemSelection(model.index(row, 0), model.index(row, model.columnCount() - 1)),
                QItemSelectionModel.SelectionFlag.ClearAndSelect,
            )
        else:
            selection.clearSelection()

    def _on_row_clicked(self, row: int, _column: int) -> None:
        item = self.table.item(row, 0)
        if item is not None and item.text() != self.selected:
            self.selected = item.text()
            self._dirty = True
            self.render()


class _StreamPage(Page):
    """The messages of the RTCM stream by type, and how the stream was checked."""

    def __init__(self, model: GnssModel) -> None:
        super().__init__(tr("RTCM stream"))
        self._model = model
        self._dirty = True
        self.table = touch_table([tr(header) for header in MESSAGE_HEADERS], stretch=1)
        self.check = set_role(QLabel(), "muted")
        self.check.setWordWrap(True)
        self.body.addWidget(self.table, stretch=1)
        self.body.addWidget(self.check)
        model.stream_changed.connect(self._on_changed)
        model.decoding_changed.connect(self._on_changed)

    def _on_changed(self) -> None:
        self._dirty = True

    def render(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        model = self._model
        statistics = model.statistics
        if statistics is None:
            self.table.setRowCount(0)
            self.check.setText("")
        else:
            fill_table(self.table, message_rows(statistics), right_aligned=range(2, 5))
            self.check.setText(describe_stream(statistics))
        decoding = model.decoding
        if decoding is not None:
            if statistics is not None and statistics.data_groups and not statistics.frames:
                text = tr("Decoding “{service}”, no RTCM messages")
            else:
                text = tr("Decoding “{service}”")
            text = text.format(service=decoding.label)
        elif model.gepos is not None:
            text = tr("GEPOS is not being decoded.")
        else:
            text = tr(_NO_GEPOS)
        self.status.setText(text)


class GnssView(QWidget):
    """The five pages in a pager, and above them a note with a switch while GEPOS is
    switched off on the "Data services" menu."""

    def __init__(
        self,
        controller: ReceiverController,
        model: GnssModel,
        parent: QWidget | None = None,
        globe_renderer: GlobeRenderThread | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._model = model

        self._note = SwitchedOffNote(tr("GEPOS is switched off under Data services."))
        self._switch_on = self._note.button
        self._switch_on.clicked.connect(self._on_switch_on)

        self._pages = {
            SKY: _SkyPage(model),
            GLOBE: _GlobePage(model, globe_renderer),
            ATMOSPHERE: _AtmospherePage(model),
            CORRECTIONS: _CorrectionsPage(model),
            STREAM: _StreamPage(model),
        }
        self._pager = Pager()
        for key in PAGES:
            self._pager.add_page(self._pages[key])
        # A page turned to is drawn at once, not only at the next tick of the window.
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

    def render(self) -> None:
        """Draws the page shown; called by the window at its plot rate."""
        self._pages[self.current_page()].render()

    def apply_theme(self, theme: Theme) -> None:
        self._pager.set_theme(theme)
        for page in self._pages.values():
            page.apply_theme(theme)

    def _gepos_switched_off(self) -> bool:
        gepos = self._model.gepos
        return gepos is not None and self._controller.is_data_component_switched_off(gepos)

    def _update_note(self) -> None:
        self._note.setVisible(self._gepos_switched_off())

    def _on_switch_on(self) -> None:
        gepos = self._model.gepos
        if gepos is not None:
            self._controller.set_data_component_enabled(gepos.scid, True)
