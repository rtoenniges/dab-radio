"""The panels of the signal page that also fit the radio page in place of the slideshow
(`ui.touch.display_choice`): spectrum, constellation, impulse response, frequency offset,
audio level and two sets of figures. Each draws from the signals of the signal
processing on its own, so that both pages can have one of the same kind.

What arrives is only noted, the newest of it; `render` draws it. The page holding a
panel calls `render` at the plot rate of the window and only while the panel is shown.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from audio.player import LEVEL_INTERVAL_SECONDS
from dab.constants import MODE_I, SAMPLE_RATE
from dsp.mp2_frame import Mp2Format, Mp2FrameResult
from dsp.superframe import AudioFormat, SuperframeResult
from i18n import tr, tr_noop, tr_plural
from ui.plots import SmoothedRange, colour_legend
from ui.receiver_controller import ReceiverController
from ui.service_tab import AUDIO_LEVEL_HISTORY, HISTORY_SECONDS
from ui.signal_tab import (
    DENSITY_BINS,
    DENSITY_EXTENT,
    MAX_CONSTELLATION_POINTS,
    constellation_density,
    constellation_limit,
)
from ui.touch.pages import style_axes
from ui.touch.theme import Theme, style_plot
from ui.touch.widgets import ElidedLabel, repolish, set_role

# How far back the frequency offset is drawn, and the smallest span of its axis: the
# offset jitters by a few hertz from frame to frame, which a narrower axis would blow
# up into a drift.
OFFSET_HISTORY_SECONDS = 120.0
OFFSET_MIN_SPAN_HZ = 100.0


def translucent(theme: Theme, name: str, alpha: int) -> QColor:
    colour = theme.color(name)
    colour.setAlpha(alpha)
    return colour


def style(plot: pg.PlotWidget, theme: Theme) -> None:
    style_plot(plot, theme)
    style_axes(plot)


def density_colour_map(theme: Theme) -> pg.ColorMap:
    """The colours of the density of the constellation: the surface of the panel
    where no point falls (the image covers the whole plot, so its zero must be the
    background), the accent colour where some do, the colour of the text where they
    are densest. The desktop's inferno starts at black, a blot on the light scheme.
    The accent sits above the middle: the square root of the counts (see
    `ConstellationPanel`) puts the tails of the clouds in the lower half."""
    return pg.ColorMap(
        [0.0, 0.55, 1.0],
        [theme.color("surface"), theme.color("accent"), theme.color("text")],
    )


class Panel(QWidget):
    """A plot or a list of figures on a surface of its own, its name above it."""

    def __init__(self, title: str, content: QWidget | None = None) -> None:
        super().__init__()
        set_role(self, "panel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.heading = set_role(QLabel(title), "heading")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(16, 12, 16, 12)
        self.body.setSpacing(8)
        self.body.addWidget(self.heading)
        if content is not None:
            self.body.addWidget(content, stretch=1)


class SignalPanel(Panel):
    """A panel drawing from the signals of the signal processing."""

    # Whether two fingers moving together on the panel in the place of the slideshow
    # turn it (`turn`); the middle button of a PC turns it wherever `turn` does
    # something (not the globe: it is not turned there at all).
    turned_by_fingers = False

    def render(self) -> None:
        pass

    def apply_theme(self, theme: Theme) -> None:
        pass

    def zoom(self, factor: float) -> None:
        """Two fingers spread (`factor` below 1) or closed on the panel in the place of
        the slideshow; only some panels zoom."""

    def turn(self, dx: float, dy: float) -> None:
        """Two fingers (`turned_by_fingers`) or the middle button of a PC moved by
        (dx, dy) pixels on the panel in the place of the slideshow; only some panels
        turn."""


class Figures(QWidget):
    """One figure per row: its name on the left (translated already), the value on the
    right.

    A value stays on one line and ends in "…" where it does not fit. Wrapped, it would
    make the panels above and beside it share the height by the width of its lines, and
    squeezed a plot to a strip; at its full width, it would widen its panel and narrow
    those beside it whenever it grew (the overview of the signal page, in German, while
    the RSPduo reported overload)."""

    def __init__(self, rows: tuple[tuple[str, str], ...]) -> None:
        super().__init__()
        set_role(self, "transparent")
        self._values: dict[str, QLabel] = {}
        self._grid = grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(12)
        for row, (key, name) in enumerate(rows):
            grid.addWidget(set_role(QLabel(name), "muted"), row, 0)
            value = ElidedLabel()
            value.setText("-")
            grid.addWidget(value, row, 1)
            self._values[key] = value
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(len(rows), 1)

    def spread(self) -> None:
        """Spreads the rows over the height, for a panel much higher than they need
        (on the radio page, in the place of the slideshow)."""
        for row in range(len(self._values)):
            self._grid.setRowStretch(row, 1)
        self._grid.setRowStretch(len(self._values), 0)

    def set(self, key: str, text: str) -> None:
        self._values[key].setText(text)

    def set_alert(self, key: str, alert: bool) -> None:
        """Shows the value in the danger colour and bold, or as before."""
        value = self._values[key]
        if bool(value.property("alert")) != alert:
            value.setProperty("alert", alert)
            repolish(value)

    def text(self, key: str) -> str:
        return self._values[key].text()


class FrameTally(QObject):
    """The audio frames of the station playing since it was chosen: how many were in
    sync, how many bytes the Reed-Solomon code corrected and how many frames were lost
    (DAB+: an uncorrectable block, DAB: a wrong header CRC). Both sets of figures show
    them, on the signal page and on the radio page."""

    changed = pyqtSignal()

    def __init__(self, controller: ReceiverController, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.dab_plus: bool | None = None  # None before the first frame
        self.synced = 0
        self.total = 0
        self.rs_corrections = 0
        self.lost = 0
        dsp = controller.dsp_worker
        dsp.superframe_ready.connect(self._on_superframe)
        dsp.mp2_frame_ready.connect(self._on_mp2_frame)
        controller.service_reset.connect(self.reset)

    def reset(self) -> None:
        self.dab_plus = None
        self.synced = self.total = self.rs_corrections = self.lost = 0
        self.changed.emit()

    def _on_superframe(self, result: SuperframeResult) -> None:
        self.dab_plus = True
        self.total += 1
        self.synced += result.sync_ok
        self.rs_corrections += result.rs_corrections
        self.lost += bool(result.rs_uncorrectable_blocks)
        self.changed.emit()

    def _on_mp2_frame(self, result: Mp2FrameResult) -> None:
        self.dab_plus = False
        self.total += 1
        self.synced += result.sync_ok
        self.lost += result.sync_ok and not result.crc_ok
        self.changed.emit()

    @property
    def unit(self) -> str:
        return tr("Superframes") if self.dab_plus else tr("MP2 frames")


# --- Plots ------------------------------------------------------------------------------


class SpectrumPanel(SignalPanel):
    """The spectrum of the samples received, also of the channels the scan passes."""

    def __init__(self, controller: ReceiverController, title: str | None = None) -> None:
        plot = pg.PlotWidget()
        super().__init__(tr("Spectrum") if title is None else title, plot)
        self.plot = plot
        self._pending: tuple[np.ndarray, np.ndarray] | None = None
        self.plot.setLabel("bottom", tr("Frequency"), units="Hz")
        self.plot.enableAutoRange(x=True, y=False)
        self.curve = self.plot.plot()
        self._range = SmoothedRange()
        controller.dsp_worker.spectrum_ready.connect(self._note)
        controller.scan_spectrum_ready.connect(self._note)

    def apply_theme(self, theme: Theme) -> None:
        style(self.plot, theme)
        self.curve.setPen(pg.mkPen(theme.accent, width=2))

    def _note(self, freqs: np.ndarray, power_db: np.ndarray) -> None:
        self._pending = (freqs, power_db)

    def render(self) -> None:
        if self._pending is None:
            return
        freqs, power_db = self._pending
        self._pending = None
        self.curve.setData(freqs, power_db)
        self._range.apply(self.plot, float(power_db.min()), float(power_db.max()))


class ConstellationPanel(SignalPanel):
    """The differentially demodulated carriers: four clouds where the reception is good.

    Drawn as a sample of the points, or as the density of all of them (chosen on the
    settings page, "Display", for this panel on the signal page and the one on the
    radio page alike): at weak reception the four clouds overlap so much that the points
    look like one blot, while the density still shows the four phases. The histogram of
    a whole frame costs a few milliseconds, and drawing one image is cheaper than
    drawing the sample as points (see `ui.signal_tab.SignalTab`)."""

    def __init__(self, controller: ReceiverController, title: str | None = None) -> None:
        plot = pg.PlotWidget()
        super().__init__(tr("Constellation (D-QPSK)") if title is None else title, plot)
        self.plot = plot
        self._pending: np.ndarray | None = None
        # The constellation last drawn, to redraw it at once when the view is switched.
        self._shown: np.ndarray | None = None
        self._density = False
        self.plot.setAspectLocked(True)
        self.plot.enableAutoRange(x=False, y=False)
        self.curve = self.plot.plot(pen=None, symbol="o", symbolSize=3, symbolPen=None)
        # The density, indexed [I bin, Q bin] as the item expects by default.
        self.image = pg.ImageItem()
        self.image.setVisible(False)
        self.plot.addItem(self.image)
        self._range = SmoothedRange()
        controller.dsp_worker.constellation_ready.connect(self._note)
        controller.live_reset.connect(self._reset)

    def apply_theme(self, theme: Theme) -> None:
        style(self.plot, theme)
        self.curve.setSymbolBrush(theme.accent)
        self.image.setColorMap(density_colour_map(theme))

    @property
    def density(self) -> bool:
        """Whether the density is drawn rather than the points."""
        return self._density

    def set_density(self, density: bool) -> None:
        """Draws the density of all points from now on, or a sample of them as points;
        what is on show is redrawn at once."""
        if density == self._density:
            return
        self._density = density
        self.curve.setVisible(not density)
        self.image.setVisible(density)
        if self._shown is not None:
            self._draw(self._shown)

    def _note(self, constellation: np.ndarray) -> None:
        self._pending = constellation

    def _reset(self) -> None:
        self._pending = None
        self._shown = None

    def render(self) -> None:
        if self._pending is None:
            return
        constellation = self._pending
        self._pending = None
        self._shown = constellation
        self._draw(constellation)

    def _draw(self, constellation: np.ndarray) -> None:
        sample = constellation
        if len(sample) > MAX_CONSTELLATION_POINTS:
            sample = sample[:: len(sample) // MAX_CONSTELLATION_POINTS]
        # The sample suffices for a percentile and costs a fraction of the full frame.
        limit = constellation_limit(sample)
        if self._density:
            extent = DENSITY_EXTENT * limit
            # The square root keeps the sparse tails visible next to the dense clusters.
            density = np.sqrt(constellation_density(constellation, extent, DENSITY_BINS))
            self.image.setImage(
                density, autoLevels=False, levels=(0.0, max(float(density.max()), 1.0))
            )
            self.image.setRect(-extent, -extent, 2 * extent, 2 * extent)
        else:
            self.curve.setData(sample.real, sample.imag)
        self._range.apply(self.plot, -limit, limit, axes="xy")


class ImpulsePanel(SignalPanel):
    """The impulse response of the channel with the guard interval shaded: paths inside
    it do no harm. The signal page's single frequency network draws it larger, with the
    path difference in km (`ui.touch.signal_page`)."""

    def __init__(self, controller: ReceiverController, title: str | None = None) -> None:
        plot = pg.PlotWidget()
        super().__init__(tr("Impulse response") if title is None else title, plot)
        self.plot = plot
        self._pending: tuple[np.ndarray, np.ndarray] | None = None
        self.plot.setLabel("bottom", tr("Delay"), units="s")
        self.plot.enableAutoRange(x=True, y=False)
        guard_seconds = MODE_I.guard_length / SAMPLE_RATE
        self._guard_region = pg.LinearRegionItem((0.0, guard_seconds), movable=False)
        self._guard_region.setZValue(-10)
        self.plot.addItem(self._guard_region)
        self.curve = self.plot.plot()
        self._range = SmoothedRange()
        controller.dsp_worker.impulse_response_ready.connect(self._note)
        controller.live_reset.connect(self._reset)

    def apply_theme(self, theme: Theme) -> None:
        style(self.plot, theme)
        self.curve.setPen(pg.mkPen(theme.accent, width=2))
        self._guard_region.setBrush(translucent(theme, "accent", 40))
        for line in self._guard_region.lines:
            line.setPen(pg.mkPen(theme.outline))

    def _note(self, delays: np.ndarray, power_db: np.ndarray) -> None:
        self._pending = (delays, power_db)

    def _reset(self) -> None:
        self._pending = None

    def render(self) -> None:
        if self._pending is None:
            return
        delays, power_db = self._pending
        self._pending = None
        self.curve.setData(delays, power_db)
        self._range.apply(self.plot, float(power_db.min()), float(power_db.max()))


class OffsetPanel(SignalPanel):
    """The frequency offset the receiver corrects, over the last two minutes."""

    def __init__(self, controller: ReceiverController, title: str | None = None) -> None:
        plot = pg.PlotWidget()
        super().__init__(tr("Frequency offset, last 2 min") if title is None else title, plot)
        self.plot = plot
        self._offsets: deque[tuple[float, float]] = deque()  # (monotonic seconds, Hz)
        self._dirty = False
        self.plot.setLabel("bottom", tr("Time (0 = now)"), units="s")
        self.plot.setLabel("left", tr("Offset"), units="Hz")
        self.plot.enableAutoRange(x=False, y=False)
        self.plot.setXRange(-OFFSET_HISTORY_SECONDS, 0, padding=0.02)
        self.curve = self.plot.plot()
        self._range = SmoothedRange()
        controller.dsp_worker.frequency_offset_ready.connect(self._on_frequency_offset)
        controller.live_reset.connect(self._reset)

    def apply_theme(self, theme: Theme) -> None:
        style(self.plot, theme)
        self.curve.setPen(pg.mkPen(theme.accent, width=2))

    def _reset(self) -> None:
        self._offsets.clear()
        self._dirty = False
        self.curve.clear()

    def _on_frequency_offset(self, offset_hz: float) -> None:
        now = time.monotonic()
        self._offsets.append((now, offset_hz))
        while self._offsets[0][0] < now - OFFSET_HISTORY_SECONDS:
            self._offsets.popleft()
        self._dirty = True

    def render(self) -> None:
        if not self._dirty or not self._offsets:
            return
        self._dirty = False
        history = np.array(self._offsets)
        self.curve.setData(history[:, 0] - time.monotonic(), history[:, 1])
        low, high = float(history[:, 1].min()), float(history[:, 1].max())
        widen = max(OFFSET_MIN_SPAN_HZ - (high - low), 0.0) / 2
        self._range.apply(self.plot, low - widen, high + widen)


class LevelPanel(SignalPanel):
    """The peak level of the decoded sound, left and right, over the time the frame
    layout of the signal page spans."""

    def __init__(self, controller: ReceiverController, title: str | None = None) -> None:
        plot = pg.PlotWidget()
        super().__init__(tr("Level of the decoded audio") if title is None else title, plot)
        self.plot = plot
        self.plot.setLabel("bottom", tr("Time (0 = now)"), units="s")
        self.plot.setLabel("left", tr("Peak level"), units="dBFS")
        self.plot.enableAutoRange(x=False, y=False)
        self.plot.setXRange(-HISTORY_SECONDS, 0, padding=0.02)
        self._range = SmoothedRange()
        # Bottom left, without a background: it never covers a curve.
        self._legend = self.plot.addLegend(offset=(10, -10))
        # Translators: the two audio channels in the legend of the level.
        self.curves = [self.plot.plot(name=tr("left")), self.plot.plot(name=tr("right"))]
        self._levels: deque[np.ndarray] = deque(maxlen=AUDIO_LEVEL_HISTORY)
        self._dirty = False
        controller.audio_player.level_ready.connect(self._on_level)
        controller.service_reset.connect(self._reset)

    def apply_theme(self, theme: Theme) -> None:
        style(self.plot, theme)
        colour_legend(self._legend, theme.text)
        for curve, colour in zip(self.curves, (theme.accent, theme.notice)):
            curve.setPen(pg.mkPen(colour, width=2))

    def _reset(self) -> None:
        self._levels.clear()
        self._dirty = False
        for curve in self.curves:
            curve.clear()

    def _on_level(self, levels: np.ndarray) -> None:
        self._levels.append(levels)
        self._dirty = True

    def render(self) -> None:
        if not self._dirty or not self._levels:
            return
        self._dirty = False
        history = np.array(self._levels)
        # As in the frame layout: 0 is now, the past extends to the left.
        times = (np.arange(len(history)) - (len(history) - 1)) * LEVEL_INTERVAL_SECONDS
        for channel, curve in enumerate(self.curves):
            if channel < history.shape[1]:
                curve.setData(times, history[:, channel])
            else:
                curve.clear()
        self._range.apply(self.plot, float(history.min()), float(history.max()))


# --- Figures ----------------------------------------------------------------------------

_RECEPTION_FIGURES = (
    ("offset", tr_noop("Frequency offset")),
    # Translators: how much of the power comes over other paths of the single frequency
    # network than the strongest.
    ("sfn", tr_noop("SFN share")),
    ("clipping", tr_noop("Overload")),
    ("frames", tr_noop("Audio frames")),
    ("correction", tr_noop("Error correction")),
    ("load", tr_noop("Load")),
)


def _figure_rows(rows: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    """The rows of a table of figures with their names translated."""
    return tuple((key, tr(name)) for key, name in rows)


class ReceptionFiguresPanel(SignalPanel):
    """The figures that tell how well the ensemble is decoded, from the frequency offset
    to the load of the computer."""

    def __init__(
        self,
        controller: ReceiverController,
        tally: FrameTally,
        title: str | None = None,
    ) -> None:
        figures = Figures(_figure_rows(_RECEPTION_FIGURES))
        super().__init__(tr("Figures") if title is None else title, figures)
        self.figures = figures
        self._controller = controller
        self._tally = tally
        dsp = controller.dsp_worker
        dsp.frequency_offset_ready.connect(self._on_frequency_offset)
        dsp.sfn_share_ready.connect(self._on_sfn_share)
        dsp.clipping_ready.connect(self._on_clipping)
        controller.overloaded_changed.connect(self._on_overloaded)
        self._on_overloaded(controller.overloaded)
        controller.load_changed.connect(self._on_load)
        controller.live_reset.connect(self._reset)
        tally.changed.connect(self._show_frames)
        self._show_frames()

    def _reset(self) -> None:
        for key in ("offset", "sfn", "clipping"):
            self.figures.set(key, "-")

    def _show_frames(self) -> None:
        tally = self._tally
        if tally.dab_plus is None:
            self.figures.set("frames", "-")
            self.figures.set("correction", "-")
            return
        self.figures.set(
            "frames",
            tr("{unit} {synced}/{total} in sync").format(
                unit=tally.unit, synced=tally.synced, total=tally.total
            ),
        )
        if tally.dab_plus:
            correction = tr("RS {corrected} corrected, {lost} uncorrectable").format(
                corrected=tally.rs_corrections, lost=tally.lost
            )
        else:
            correction = tr("Header CRC errors {lost}").format(lost=tally.lost)
        self.figures.set("correction", correction)

    def _on_frequency_offset(self, offset_hz: float) -> None:
        center_freq_hz = self._controller.center_frequency_hz
        ppm = offset_hz / center_freq_hz * 1e6 if center_freq_hz else 0.0
        self.figures.set("offset", f"{offset_hz:+.0f} Hz ({ppm:+.2f} ppm)")

    def _on_sfn_share(self, others: float, outside_guard: float) -> None:
        self.figures.set(
            "sfn",
            tr("{others:.1f} %, outside the guard {outside_guard:.1f} %").format(
                others=others * 100, outside_guard=outside_guard * 100
            ),
        )

    def _on_clipping(self, fraction: float) -> None:
        text = f"{fraction * 100:.2f} %"
        if self._controller.adc_overload:
            # The words of the synchronisation page: "– RSPduo reports overload" was too
            # long for half the overview in German.
            text = tr("{clipping}, ADC overloaded").format(clipping=text)
        self.figures.set("clipping", text)

    def _on_overloaded(self, overloaded: bool) -> None:
        self.figures.set_alert("clipping", overloaded)

    def _on_load(self, dsp_load: float, dsp_waiting: float, process_load: float) -> None:
        self.figures.set(
            "load",
            # Short, as in the rail: in the place of the slideshow the line is cut.
            tr("DSP {dsp:.0f} %, waiting {waiting:.0f} %, CPU {cpu:.0f} %").format(
                dsp=dsp_load * 100, waiting=dsp_waiting * 100, cpu=process_load * 100
            ),
        )


def describe_format(audio_format: AudioFormat | Mp2Format) -> tuple[str, str]:
    """The codec of the station playing, and the sample rate and channels of its sound.
    Two values instead of the one line of `describe()`, which is too wide for a figure."""
    rate = f"{audio_format.sample_rate // 1000} kHz"
    if isinstance(audio_format, Mp2Format):
        version = "MPEG-2" if audio_format.is_lsf else "MPEG-1"
        return f"{version} Layer II", f"{rate}, {audio_format.channel_mode_name}"
    if audio_format.parametric_stereo:
        codec = "HE-AAC v2"
    elif audio_format.sbr:
        codec = "HE-AAC"
    else:
        codec = "AAC-LC"
    channels = tr("Stereo") if audio_format.channels == 2 else tr("Mono")
    return codec, f"{rate}, {channels}"


_SOUND_FIGURES = (
    ("frames", tr_noop("In sync")),
    ("corrected", tr_noop("Corrected")),
    ("lost", tr_noop("Uncorrectable")),
    ("codec", tr_noop("Codec")),
    # Translators: sample rate and channels of the audio, e.g. "48 kHz, Stereo".
    ("audio", tr_noop("Audio format")),
    ("buffer", tr_noop("Buffer")),
)


class SoundFiguresPanel(SignalPanel):
    """The station playing from its audio frames to the sound: how many frames were in
    sync, repaired and lost, the codec, and how much sound the player holds."""

    def __init__(
        self,
        controller: ReceiverController,
        tally: FrameTally,
        title: str | None = None,
    ) -> None:
        figures = Figures(_figure_rows(_SOUND_FIGURES))
        super().__init__(tr("Figures") if title is None else title, figures)
        self.figures = figures
        self._tally = tally
        dsp = controller.dsp_worker
        dsp.superframe_ready.connect(self._on_superframe)
        dsp.mp2_frame_ready.connect(self._on_mp2_frame)
        controller.audio_player.buffer_changed.connect(self._on_buffer)
        controller.service_reset.connect(self._reset)
        tally.changed.connect(self._show_frames)
        self._show_frames()

    def _reset(self) -> None:
        for key in ("codec", "audio", "buffer"):
            self.figures.set(key, "-")

    def _on_superframe(self, result: SuperframeResult) -> None:
        if result.audio_format is not None:
            self._show_format(*describe_format(result.audio_format))

    def _on_mp2_frame(self, result: Mp2FrameResult) -> None:
        if result.sync_ok:
            self._show_format(*describe_format(result.audio_format))

    def _show_format(self, codec: str, sound: str) -> None:
        if self.figures.text("codec") != codec:
            self.figures.set("codec", codec)
        if self.figures.text("audio") != sound:
            self.figures.set("audio", sound)

    def _on_buffer(self, seconds: float, underruns: int) -> None:
        self.figures.set(
            "buffer",
            tr_plural(
                "{milliseconds:.0f} ms, {count} dropout",
                "{milliseconds:.0f} ms, {count} dropouts",
                underruns,
            ).format(milliseconds=seconds * 1000, count=underruns),
        )

    def _show_frames(self) -> None:
        tally = self._tally
        if tally.dab_plus is None:
            for key in ("frames", "corrected", "lost"):
                self.figures.set(key, "-")
            return
        self.figures.set("frames", f"{tally.synced}/{tally.total} {tally.unit}")
        if tally.dab_plus:
            corrected = tally.rs_corrections
            self.figures.set(
                "corrected",
                tr_plural(
                    "{count} byte (Reed-Solomon)", "{count} bytes (Reed-Solomon)", corrected
                ).format(count=corrected),
            )
            self.figures.set(
                "lost",
                tr_plural("{count} superframe", "{count} superframes", tally.lost).format(
                    count=tally.lost
                ),
            )
        else:
            self.figures.set("corrected", tr("does not apply to DAB"))
            self.figures.set(
                "lost",
                tr_plural(
                    "{count} frame (header CRC)", "{count} frames (header CRC)", tally.lost
                ).format(count=tally.lost),
            )
