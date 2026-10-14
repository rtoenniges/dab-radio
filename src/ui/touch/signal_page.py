"""The signal page: how the received ensemble is decoded, on six pages to swipe through
(`ui.touch.pager`) as in the GNSS menu. First an overview (spectrum, constellation,
impulse response and the figures), then the steps of the signal chain one by one: the
synchronisation, the FFT per symbol, a second of the signal in detail
(`ui.touch.detail_view`), the single frequency network and the error correction of
the station playing. The pages draw from the same signals as the desktop's
tabs, without their tooltips: a finger has nothing to hover with. The panels the radio
page can show as well come from `ui.touch.signal_panels`."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from dab.constants import MODE_I, SAMPLE_RATE
from dsp.iq_correction import IqCorrectionReport
from dsp.mp2_frame import Mp2FrameResult
from dsp.pipeline import SYMBOL_WINDOW_EXCERPT_SYMBOLS, SubchannelStatus, SymbolWindows
from dsp.superframe import SuperframeResult
from dsp.tii import TiiResult
from i18n import tr, tr_noop, tr_plural
from ui.ofdm_tab import OfdmColours, OfdmPlots
from ui.offscreen_gl import RenderThread
from ui.plots import PathDifferenceAxis, SmoothedRange
from ui.receiver_controller import ReceiverController
from ui.service_tab import FrameLayoutPlot, Mp2FrameLayoutPlot, SuperframeLayoutPlot
from ui.signal_tab import MAX_FRAME_MARKERS
from ui.tables import fill_table
from ui.touch.detail_view import DetailPage
from ui.touch.pager import Pager
from ui.touch.pages import Page, pixel_font, touch_table
from ui.touch.signal_panels import (
    ConstellationPanel,
    Figures,
    FrameTally,
    ImpulsePanel,
    LevelPanel,
    OffsetPanel,
    Panel,
    ReceptionFiguresPanel,
    SoundFiguresPanel,
    SpectrumPanel,
    style,
    translucent,
)
from ui.touch.theme import DARK, SMALL_FONT_PX, THEMES, Theme
from ui.touch.widgets import set_role

# The pages in the order of swiping.
OVERVIEW = "overview"
SYNC = "sync"
OFDM = "ofdm"
DETAIL = "detail"
SFN = "sfn"
CORRECTION = "correction"
PAGES = (OVERVIEW, SYNC, OFDM, DETAIL, SFN, CORRECTION)

# Width of the figures beside a plot, and of the transmitters beside the impulse
# response: just room for an identifier ("19-04"), a level ("-18.6 dB") and a share
# ("100.0 %"), so that the impulse response gets the rest.
FIGURES_WIDTH = 460
TII_WIDTH = 310
# The legend of a frame layout, opaque enough to read over the bands beneath it.
LEGEND_ALPHA = 220


def _ofdm_colours(theme: Theme) -> OfdmColours:
    return OfdmColours(
        guard=translucent(theme, "muted", 40),
        window=translucent(theme, "accent", 50),
        null_window=(190, 110, 200, 60),
        magnitude=theme.text,
        guard_copy=theme.notice,
        boundary=theme.danger,
        minor_boundary=translucent(theme, "danger", 170),
        carrier_edge=translucent(theme, "text", 140),
        label_fill=translucent(theme, "surface", 200),
    )


# --- Overview ---------------------------------------------------------------------------


class _OverviewPage(Page):
    """Spectrum, constellation and impulse response of the received ensemble, and the
    figures that tell how well it is decoded; the channel in the status line."""

    def __init__(self, controller: ReceiverController, tally: FrameTally) -> None:
        super().__init__(tr("Overview"))
        figures = ReceptionFiguresPanel(controller, tally)
        self.figures = figures.figures
        self.constellation = ConstellationPanel(controller)
        self._panels = (
            SpectrumPanel(controller),
            self.constellation,
            ImpulsePanel(controller),
            figures,
        )
        grid = QGridLayout()
        grid.setSpacing(16)
        # Half the width each, whatever the figures say: without a stretch, the columns
        # share by the width the panels ask for, and the tiles moved with every value.
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        for index, panel in enumerate(self._panels):
            grid.addWidget(panel, index // 2, index % 2)
        self.body.addLayout(grid, stretch=1)
        controller.tuning_changed.connect(self._show_tuning)
        self._show_tuning(controller.channel, controller.frequency_mhz)

    def apply_theme(self, theme: Theme) -> None:
        for panel in self._panels:
            panel.apply_theme(theme)

    def render(self) -> None:
        for panel in self._panels:
            panel.render()

    def _show_tuning(self, channel: str | None, frequency_mhz: float) -> None:
        frequency = f"{frequency_mhz:.3f} MHz"
        if channel:
            text = tr("Channel {channel} · {frequency}").format(
                channel=channel, frequency=frequency
            )
        else:
            text = frequency
        self.status.setText(text)


# --- Synchronisation --------------------------------------------------------------------

LOCKED = tr_noop("Frame timing locked")
SEARCHING = tr_noop("searching for the null symbol")


class _SyncPage(Page):
    """Where the frames begin and how far off the carriers are. On top the power of the
    samples being decoded, two to three frames of it, with a marker on every null
    symbol the frame synchronisation locked onto: two markers one frame apart enclose
    the frame decoded in this pass (see `ui.signal_tab.SignalTab`). Beneath, the
    frequency offset the receiver corrects, over the last two minutes, and the errors
    of the receiver that the DC and IQ corrections remove."""

    def __init__(self, controller: ReceiverController) -> None:
        super().__init__(tr("Synchronisation"))
        self._controller = controller
        self._pending_envelope: tuple[np.ndarray, np.ndarray, list[float]] | None = None

        self._envelope_plot = pg.PlotWidget()
        self._envelope_plot.setLabel("bottom", tr("Time"), units="s")
        self._envelope_plot.setLabel("left", tr("Power"), units="dB")
        self._envelope_plot.enableAutoRange(x=True, y=False)
        self._envelope_curve = self._envelope_plot.plot()
        self._envelope_range = SmoothedRange()
        self._null_markers = [
            pg.InfiniteLine(angle=90, movable=False) for _ in range(MAX_FRAME_MARKERS)
        ]
        for marker in self._null_markers:
            marker.setVisible(False)
            self._envelope_plot.addItem(marker)

        self.offset = OffsetPanel(controller)
        self.figures = Figures(
            (
                ("offset", tr("Offset")),
                # Translators: the frequency offset relative to the frequency, in ppm.
                ("ppm", tr("relative")),
                ("clipping", tr("Overload")),
                ("dc", tr("DC offset")),
                ("phase", tr("IQ phase")),
                ("gain", tr("IQ gain")),
            )
        )
        figures_panel = Panel(tr("Figures"), self.figures)
        figures_panel.setFixedWidth(FIGURES_WIDTH)
        row = QHBoxLayout()
        row.setSpacing(16)
        row.addWidget(self.offset, stretch=1)
        row.addWidget(figures_panel)
        envelope = Panel(tr("Power over time with the null symbols of the frame synchronisation"))
        envelope.body.addWidget(self._envelope_plot, stretch=1)
        self.body.addWidget(envelope, stretch=1)
        self.body.addLayout(row, stretch=1)

        dsp = controller.dsp_worker
        dsp.power_envelope_ready.connect(self._on_envelope)
        dsp.frequency_offset_ready.connect(self._on_frequency_offset)
        dsp.clipping_ready.connect(self._on_clipping)
        controller.overloaded_changed.connect(self._on_overloaded)
        self._on_overloaded(controller.overloaded)
        dsp.iq_correction_ready.connect(self._on_iq_correction)
        controller.live_reset.connect(self._reset)
        controller.reception_stopped.connect(self._on_reception_stopped)

    def apply_theme(self, theme: Theme) -> None:
        style(self._envelope_plot, theme)
        self.offset.apply_theme(theme)
        self._envelope_curve.setPen(pg.mkPen(theme.accent, width=1))
        for marker in self._null_markers:
            marker.setPen(pg.mkPen(theme.danger, width=2))

    def _reset(self) -> None:
        self._pending_envelope = None
        for marker in self._null_markers:
            marker.setVisible(False)
        for key in ("offset", "ppm", "clipping"):
            self.figures.set(key, "-")
        self.status.setText("")

    def _on_reception_stopped(self, _list_kept: bool) -> None:
        self.status.setText("")

    def _on_envelope(
        self, time_axis: np.ndarray, power_db: np.ndarray, frame_times: list[float]
    ) -> None:
        self._pending_envelope = (time_axis, power_db, frame_times)
        status = tr(LOCKED) if frame_times else tr(SEARCHING)
        if self.status.text() != status:
            self.status.setText(status)

    def _on_frequency_offset(self, offset_hz: float) -> None:
        center_freq_hz = self._controller.center_frequency_hz
        self.figures.set("offset", f"{offset_hz:+.0f} Hz")
        if center_freq_hz:
            self.figures.set("ppm", f"{offset_hz / center_freq_hz * 1e6:+.2f} ppm")

    def _on_clipping(self, fraction: float) -> None:
        text = f"{fraction * 100:.2f} %"
        if self._controller.adc_overload:
            text = tr("{clipping}, ADC overloaded").format(clipping=text)
        self.figures.set("clipping", text)

    def _on_overloaded(self, overloaded: bool) -> None:
        self.figures.set_alert("clipping", overloaded)

    def _on_iq_correction(self, report: IqCorrectionReport | None) -> None:
        off = tr("Correction off")
        dc = None if report is None else report.dc_offset
        phase = None if report is None else report.quadrature_error_deg
        gain = None if report is None else report.gain_imbalance_db
        self.figures.set("dc", off if dc is None else f"{dc:.4f}")
        self.figures.set("phase", off if phase is None else f"{phase:+.2f}°")
        self.figures.set("gain", off if gain is None else f"{gain:+.2f} dB")

    def render(self) -> None:
        if self._pending_envelope is not None:
            time_axis, power_db, frame_times = self._pending_envelope
            self._pending_envelope = None
            self._envelope_curve.setData(time_axis, power_db)
            self._envelope_range.apply(
                self._envelope_plot, float(power_db.min()), float(power_db.max())
            )
            for marker, position in zip(self._null_markers, frame_times):
                marker.setPos(position)
                marker.setVisible(True)
            for marker in self._null_markers[len(frame_times) :]:
                marker.setVisible(False)
        self.offset.render()


# --- OFDM -------------------------------------------------------------------------------


class _OfdmPage(Page):
    """What the FFT per symbol receives and delivers (see `ui.ofdm_tab.OfdmTab`): the
    time signal at the start of the frame cut into guard intervals and FFT windows, and
    beneath it the FFT output, one row per symbol."""

    def __init__(self, controller: ReceiverController) -> None:
        super().__init__("OFDM")
        self.plots = OfdmPlots(MODE_I, SAMPLE_RATE, SYMBOL_WINDOW_EXCERPT_SYMBOLS)
        self._symbols: tuple[int, int] | None = None  # first and last symbol l received
        self.body.addWidget(
            Panel(
                tr("Time signal at the start of the frame: guard intervals and FFT windows"),
                self.plots.time_plot,
            ),
            stretch=2,
        )
        self.body.addWidget(
            Panel(
                tr("FFT output: one row per symbol, one column per bin"),
                self.plots.spectrum_plot,
            ),
            stretch=3,
        )
        controller.dsp_worker.symbol_windows_ready.connect(self._on_windows)
        controller.live_reset.connect(self._reset)

    def apply_theme(self, theme: Theme) -> None:
        self.plots.set_colours(_ofdm_colours(theme))
        for plot in (self.plots.time_plot, self.plots.spectrum_plot):
            style(plot, theme)
        for name in ("left", "right", "top", "bottom"):
            axis = self.plots.colour_bar.getAxis(name)
            axis.setPen(pg.mkPen(theme.outline))
            axis.setTextPen(pg.mkPen(theme.muted))
            axis.setStyle(tickFont=pixel_font(SMALL_FONT_PX))

    def _reset(self) -> None:
        self.plots.clear()
        self._symbols = None
        self.status.setText("")

    def _on_windows(self, windows: SymbolWindows) -> None:
        self.plots.update(windows)
        first = 0 if windows.null_spectrum is not None else 1
        self._symbols = (first, len(windows.spectra))

    def render(self) -> None:
        self.plots.render()
        if self._symbols is not None:
            first, last = self._symbols
            status = tr("{carriers} carriers · symbols {first} to {last}").format(
                carriers=MODE_I.num_carriers, first=first, last=last
            )
            if self.status.text() != status:
                self.status.setText(status)


# --- Single frequency network -----------------------------------------------------------


class _SfnPage(Page):
    """The single frequency network: the impulse response, large, with the path
    difference in km above it, and the transmitters the identification in the null
    symbols names (TII). The status line tells how much of the power comes over other
    paths than the strongest."""

    def __init__(self, controller: ReceiverController) -> None:
        super().__init__(tr("Single frequency network"))
        self._pending_impulse: tuple[np.ndarray, np.ndarray] | None = None
        self._pending_tii: TiiResult | None = None

        guard_seconds = MODE_I.guard_length / SAMPLE_RATE
        self._impulse_plot = pg.PlotWidget(axisItems={"top": PathDifferenceAxis("top")})
        self._impulse_plot.setLabel("bottom", tr("Delay (0 = synchronisation)"), units="s")
        self._impulse_plot.setLabel("top", tr("Path difference (km)"))
        self._impulse_plot.setLabel("left", tr("Power rel. to the strongest path"), units="dB")
        self._impulse_plot.enableAutoRange(x=True, y=False)
        self._guard_region = pg.LinearRegionItem((0.0, guard_seconds), movable=False)
        self._guard_region.setZValue(-10)
        self._impulse_plot.addItem(self._guard_region)
        self._impulse_curve = self._impulse_plot.plot()
        self._impulse_range = SmoothedRange()

        # "TII" over the identifiers: a word ("Identifier") would make the column half
        # as wide again as "19-04" needs.
        self.table = touch_table(["TII", tr("Level"), tr("Share")])
        self.tii_note = set_role(QLabel("-"), "small")
        self.tii_note.setWordWrap(True)
        transmitters = Panel(tr("Transmitters (TII)"))
        transmitters.body.addWidget(self.table, stretch=1)
        transmitters.body.addWidget(self.tii_note)
        transmitters.setFixedWidth(TII_WIDTH)

        row = QHBoxLayout()
        row.setSpacing(16)
        row.addWidget(
            Panel(tr("Impulse response, the guard interval shaded"), self._impulse_plot),
            stretch=1,
        )
        row.addWidget(transmitters)
        self.body.addLayout(row, stretch=1)

        dsp = controller.dsp_worker
        dsp.impulse_response_ready.connect(self._note_impulse)
        dsp.sfn_share_ready.connect(self._on_sfn_share)
        dsp.tii_ready.connect(self._note_tii)
        controller.live_reset.connect(self._reset)

    def apply_theme(self, theme: Theme) -> None:
        style(self._impulse_plot, theme)
        self._impulse_curve.setPen(pg.mkPen(theme.accent, width=2))
        self._guard_region.setBrush(translucent(theme, "accent", 40))
        for line in self._guard_region.lines:
            line.setPen(pg.mkPen(theme.outline))

    def _reset(self) -> None:
        self._pending_impulse = None
        self._pending_tii = None
        fill_table(self.table, [])
        self.tii_note.setText("-")
        self.status.setText("")

    def _note_impulse(self, delays: np.ndarray, power_db: np.ndarray) -> None:
        self._pending_impulse = (delays, power_db)

    def _note_tii(self, result: TiiResult) -> None:
        self._pending_tii = result

    def _on_sfn_share(self, others: float, outside_guard: float) -> None:
        self.status.setText(
            tr("SFN share {others:.1f} %, outside the guard {outside_guard:.1f} %").format(
                others=others * 100, outside_guard=outside_guard * 100
            )
        )

    def render(self) -> None:
        if self._pending_impulse is not None:
            delays, power_db = self._pending_impulse
            self._pending_impulse = None
            self._impulse_curve.setData(delays, power_db)
            self._impulse_range.apply(
                self._impulse_plot, float(power_db.min()), float(power_db.max())
            )
        if self._pending_tii is not None:
            self._show_tii(self._pending_tii)
            self._pending_tii = None

    def _show_tii(self, result: TiiResult) -> None:
        rows = [
            (
                f"{transmitter.main_id:02d}-{transmitter.sub_id:02d}",
                f"{transmitter.level_db:.1f} dB",
                f"{transmitter.share * 100:.1f} %",
            )
            for transmitter in result.transmitters
        ]
        fill_table(self.table, rows, right_aligned=range(1, 3))
        if result.detection_limit_db is None:
            note = tr_plural(
                "None detected, {count} null symbol averaged.",
                "None detected, {count} null symbols averaged.",
                result.frames,
            ).format(count=result.frames)
        else:
            note = tr_plural(
                "{count} null symbol averaged, detection limit {limit:.0f} dB.",
                "{count} null symbols averaged, detection limit {limit:.0f} dB.",
                result.frames,
            ).format(count=result.frames, limit=result.detection_limit_db)
        self.tii_note.setText(note)


# --- Error correction -------------------------------------------------------------------

_SUPERFRAME_HEADING = tr_noop("Layout of the superframes, 120 ms of audio per superframe")


def describe_subchannel(status: SubchannelStatus | None) -> str:
    """The subchannel of the station playing in one line, empty until the FIC told."""
    if status is None or status.component is None or status.subchannel is None:
        return ""
    subchannel = status.subchannel
    kind = "DAB+" if status.component.is_dab_plus else "DAB"
    text = tr("{kind} · Subchannel {number} · {bitrate} kbit/s · {protection}").format(
        kind=kind,
        number=subchannel.subchannel_id,
        bitrate=subchannel.bitrate,
        protection=subchannel.description(),
    )
    if status.announcement is None:
        return text
    announcement = tr("Announcement")
    return f"{text} · {announcement}"


class _CorrectionPage(Page):
    """The station playing from its subchannel to the sound: how its audio frames split
    into bytes and which parts failed their check, what the Reed-Solomon code repaired
    (DAB+ only), and the level of the decoded sound."""

    def __init__(self, controller: ReceiverController, tally: FrameTally) -> None:
        super().__init__(tr("Error correction"))
        self.superframe_plot = SuperframeLayoutPlot(labelled=False)
        self.mp2_plot = Mp2FrameLayoutPlot(labelled=False)
        self.mp2_plot.plot.hide()
        self.layout_panel = Panel(tr(_SUPERFRAME_HEADING))
        self.layout_panel.body.addWidget(self.superframe_plot.plot, stretch=1)
        self.layout_panel.body.addWidget(self.mp2_plot.plot, stretch=1)

        # Over the same time as the frame layout above it.
        self.level = LevelPanel(controller)
        self._sound = SoundFiguresPanel(controller, tally)
        self._sound.setFixedWidth(FIGURES_WIDTH)
        self.figures = self._sound.figures
        row = QHBoxLayout()
        row.setSpacing(16)
        row.addWidget(self.level, stretch=1)
        row.addWidget(self._sound)
        self.body.addWidget(self.layout_panel, stretch=3)
        self.body.addLayout(row, stretch=2)

        dsp = controller.dsp_worker
        dsp.superframe_ready.connect(self._on_superframe)
        dsp.mp2_frame_ready.connect(self._on_mp2_frame)
        dsp.subchannel_info_ready.connect(self._on_subchannel_info)
        controller.service_reset.connect(self._reset)

    def apply_theme(self, theme: Theme) -> None:
        legend = translucent(theme, "surface", LEGEND_ALPHA)
        for layout_plot in (self.superframe_plot, self.mp2_plot):
            style(layout_plot.plot, theme)
            layout_plot.set_legend_colours(theme.text, legend)
        self.level.apply_theme(theme)
        self._sound.apply_theme(theme)

    def _reset(self) -> None:
        for layout_plot in (self.superframe_plot, self.mp2_plot):
            layout_plot.clear()
        self.status.setText("")

    def _on_subchannel_info(self, status: SubchannelStatus | None) -> None:
        self.status.setText(describe_subchannel(status))

    def _show_layout(self, shown: FrameLayoutPlot, heading: str) -> None:
        """Shows the layout diagram matching the audio coding of the service."""
        for candidate in (self.superframe_plot, self.mp2_plot):
            candidate.plot.setVisible(candidate is shown)
        if self.layout_panel.heading.text() != heading:
            self.layout_panel.heading.setText(heading)

    def _on_superframe(self, result: SuperframeResult) -> None:
        self._show_layout(self.superframe_plot, tr(_SUPERFRAME_HEADING))
        self.superframe_plot.update(result)

    def _on_mp2_frame(self, result: Mp2FrameResult) -> None:
        if result.sync_ok:
            duration_ms = result.audio_format.duration_seconds * 1000
            heading = tr("Layout of the MP2 frames, {duration:.0f} ms of audio per frame").format(
                duration=duration_ms
            )
            self._show_layout(self.mp2_plot, heading)
        self.mp2_plot.update(result)

    def render(self) -> None:
        self.superframe_plot.render()
        self.mp2_plot.render()
        self.level.render()


# --- The page ---------------------------------------------------------------------------


class SignalPage(QWidget):
    """The six pages in a pager.

    The plots are drawn by `render`, which the window calls at its plot rate and only
    while this page is shown, and then only those of the page turned to: drawing
    hidden plots would take the compute time the signal processing needs for the audio
    (see `ui.main_window.PLOT_REFRESH_INTERVAL_MS`). What arrives meanwhile is only
    noted, the newest of it.
    """

    def __init__(
        self,
        controller: ReceiverController,
        tally: FrameTally,
        waterfall_renderer: RenderThread | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._overview = _OverviewPage(controller, tally)
        self._pages: dict[str, Page] = {
            OVERVIEW: self._overview,
            SYNC: _SyncPage(controller),
            OFDM: _OfdmPage(controller),
            DETAIL: DetailPage(controller, waterfall_renderer),
            SFN: _SfnPage(controller),
            CORRECTION: _CorrectionPage(controller, tally),
        }
        self._pager = Pager()
        for key in PAGES:
            self._pager.add_page(self._pages[key])
        # A page turned to is drawn at once, not only at the next tick of the window.
        self._pager.current_changed.connect(lambda _index: self.render())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.addWidget(self._pager)
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

    def show_first_page(self) -> None:
        """Back to the overview, the pages sliding by (a second tap on "Signal")."""
        self._pager.set_current_index(0, animated=True)

    def render(self) -> None:
        self._pages[self.current_page()].render()

    def set_constellation_density(self, density: bool) -> None:
        """Draws the constellation of the overview as a density or as points."""
        self._overview.constellation.set_density(density)

    def apply_theme(self, theme: Theme) -> None:
        self._pager.set_theme(theme)
        for page in self._pages.values():
            page.apply_theme(theme)
