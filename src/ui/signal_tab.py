"""The "Signal" tab: spectrum, power envelope, constellation and channel impulse response."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget

from dab.constants import MODE_I, SAMPLE_RATE
from i18n import tr
from ui.plots import PathDifferenceAxis, SmoothedRange

# Upper limit for the number of constellation points drawn at once. A data block can
# deliver up to 16*1536 = 24576 points, and pyqtgraph's ScatterPlotItem scales
# noticeably badly per point (profiling: redrawing the constellation alone was the
# biggest single item of the GUI load). For the bare visualisation of the four D-QPSK
# clusters a much smaller sample is visually more than enough.
MAX_CONSTELLATION_POINTS = 3000

# The constellation axes reach to this percentile of the point magnitudes, not to the
# largest one. At low SNR the D-QPSK product of two noisy symbols has long tails:
# measured on the weak 9D, the largest of the points shown lay at 4x the median
# magnitude (1.8x on 5C), and an axis fitted to it zoomed out so far that the cloud
# shrank into the middle. The few points beyond simply fall outside the view.
CONSTELLATION_RANGE_PERCENTILE = 99.0

# Density view: bins per axis, and how far the histogram reaches in multiples of the
# range above. It reaches past the view (range plus 10 % padding on each side, see
# `SmoothedRange`) so that no edge of the image shows inside it.
DENSITY_BINS = 160
DENSITY_EXTENT = 1.5

# Maximum number of frame markers shown at once in the power envelope. Displayed is
# the frame buffer (see `dsp.pipeline.DspWorker._emit_power_envelope`), which spans
# two to three frames depending on its fill level; measured on real hardware that is
# 1.99 markers per update on average. The pool sits above that with a safety margin.
MAX_FRAME_MARKERS = 4


def constellation_limit(points: np.ndarray) -> float:
    """Half the width of the constellation view: the magnitude that
    `CONSTELLATION_RANGE_PERCENTILE` percent of the points stay within."""
    if len(points) == 0:
        return 1.0
    return float(np.percentile(np.abs(points), CONSTELLATION_RANGE_PERCENTILE)) or 1.0


def constellation_density(points: np.ndarray, extent: float, bins: int) -> np.ndarray:
    """How many points fall into each of `bins` x `bins` squares covering
    [-extent, extent] in I and Q, indexed [I bin, Q bin] as `pg.ImageItem` expects.
    Points outside are left out."""
    scale = bins / (2 * extent)
    i_bins = np.floor((points.real + extent) * scale).astype(np.int64)
    q_bins = np.floor((points.imag + extent) * scale).astype(np.int64)
    inside = (i_bins >= 0) & (i_bins < bins) & (q_bins >= 0) & (q_bins < bins)
    counts = np.bincount(i_bins[inside] * bins + q_bins[inside], minlength=bins * bins)
    return counts.reshape(bins, bins)


class SignalTab(QWidget):
    """Spectrum and power envelope on top, constellation and impulse response below.

    Spectrum and power envelope are cheap and drawn on arrival. Constellation and
    impulse response are only noted by `update_*` and drawn in `render()`, which the
    main window calls at the shared refresh rate and only while this tab is visible.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pending_constellation: np.ndarray | None = None
        # The constellation last drawn, to redraw it at once when the view is switched.
        self._shown_constellation: np.ndarray | None = None
        self._pending_impulse_response: tuple[np.ndarray, np.ndarray] | None = None

        self._spectrum_plot = pg.PlotWidget(title=tr("Spectrum"))
        self._spectrum_plot.setLabel("bottom", tr("Frequency"), units="Hz")
        self._spectrum_plot.setLabel("left", tr("Power"), units="dB")
        self._spectrum_plot.enableAutoRange(x=True, y=False)
        self._spectrum_curve = self._spectrum_plot.plot(pen="c")
        self._spectrum_range = SmoothedRange()

        self._power_plot = pg.PlotWidget(title=tr("Power envelope / frame synchronisation"))
        self._power_plot.setLabel("bottom", tr("Time"), units="s")
        self._power_plot.setLabel("left", tr("Power"), units="dB")
        self._power_plot.enableAutoRange(x=True, y=False)
        self._power_curve = self._power_plot.plot(pen="y")
        self._power_range = SmoothedRange()
        self._power_plot.setToolTip(
            tr(
                "Power of the IQ samples being decoded at the moment (two to three frames). "
                "The red markers sit on the null symbols the frame synchronisation has locked "
                "onto - so they show the frame position actually decoded, not a search of its "
                "own.\n\n"
                "When locked there are two markers exactly one frame length apart: they "
                "enclose the frame decoded in this pass. The third null symbol at the right "
                "edge stays unmarked - the frame behind it has not fully arrived yet and is "
                "decoded only in the next pass. No markers means: no frame synchronised yet."
            )
        )
        self._null_markers = [
            pg.InfiniteLine(angle=90, pen="r", movable=False) for _ in range(MAX_FRAME_MARKERS)
        ]
        for marker in self._null_markers:
            marker.setVisible(False)
            self._power_plot.addItem(marker)

        self._constellation_plot = pg.PlotWidget(title=tr("Constellation (D-QPSK)"))
        self._constellation_plot.setLabel("bottom", "I")
        self._constellation_plot.setLabel("left", "Q")
        self._constellation_plot.setAspectLocked(True)
        self._constellation_plot.enableAutoRange(x=False, y=False)
        self._constellation_curve = self._constellation_plot.plot(
            pen=None, symbol="o", symbolSize=2, symbolPen=None, symbolBrush="c"
        )
        self._constellation_density = pg.ImageItem()
        self._constellation_density.setColorMap(pg.colormap.get("inferno"))
        self._constellation_density.setVisible(False)
        self._constellation_plot.addItem(self._constellation_density)
        self._constellation_range = SmoothedRange()
        # Points by default; the density is switched on here.
        self._density_checkbox = QCheckBox(tr("Density view"))
        self._density_checkbox.setToolTip(
            tr(
                "Shows, instead of single points, how densely the points of a frame lie, and "
                "that for all carriers of all symbols instead of a sample of {points} points. "
                "With weak reception the four clusters overlap so much that the point cloud "
                "looks like a single blot. In the density the four phases can still be made "
                "out then."
            ).format(points=MAX_CONSTELLATION_POINTS)
        )
        self._density_checkbox.toggled.connect(self._on_density_toggled)

        guard_seconds = MODE_I.guard_length / SAMPLE_RATE
        self._impulse_plot = pg.PlotWidget(
            title=tr("Impulse response of the channel"),
            axisItems={"top": PathDifferenceAxis("top")},
        )
        self._impulse_plot.setLabel("bottom", tr("Delay (0 = synchronisation)"), units="s")
        self._impulse_plot.setLabel("top", tr("Path difference (km)"))
        self._impulse_plot.setLabel("left", tr("Power rel. to strongest path"), units="dB")
        self._impulse_plot.enableAutoRange(x=True, y=False)
        # Shaded: the guard interval. Paths inside it do no harm.
        guard_region = pg.LinearRegionItem(
            (0.0, guard_seconds),
            movable=False,
            brush=pg.mkBrush(255, 255, 255, 25),
            pen=pg.mkPen(255, 255, 255, 80),
        )
        guard_region.setZValue(-10)
        self._impulse_plot.addItem(guard_region)
        self._impulse_curve = self._impulse_plot.plot(pen="c")
        self._impulse_range = SmoothedRange()
        self._impulse_plot.setToolTip(
            tr(
                "Each peak is a path on which the signal arrives: in the single frequency "
                "network one transmitter each, plus reflections. Its position is the delay "
                "against the path the receiver synchronises to (0); on top as a path "
                "difference, 1 µs corresponds to 300 m. Estimated from the phase reference "
                "symbol of each frame.\n\n"
                "Shaded is the guard interval ({guard:.0f} µs): paths inside it do no harm, "
                "their cyclic prefix covers the FFT window. Paths before or after it mix in "
                "neighbouring symbols.\n\n"
                "The bandwidth resolves paths only from about 0.7 µs apart; a reflection "
                "nearby merges with its transmitter. The carriers are weighted with a Hann "
                "window, otherwise every peak would have sidelobes of -13 dB that would look "
                "like further paths."
            ).format(guard=guard_seconds * 1e6)
        )

        layout = QVBoxLayout()
        layout.addWidget(self._spectrum_plot)
        layout.addWidget(self._power_plot)
        # The constellation is roughly square, the impulse response a time axis: widths
        # in the ratio 1:2.
        channel_row = QHBoxLayout()
        constellation_column = QVBoxLayout()
        constellation_column.addWidget(self._constellation_plot)
        constellation_column.addWidget(self._density_checkbox)
        channel_row.addLayout(constellation_column, stretch=1)
        channel_row.addWidget(self._impulse_plot, stretch=2)
        layout.addLayout(channel_row)
        self.setLayout(layout)

    def reset(self) -> None:
        """Drops data noted for drawing (on start and on a service change)."""
        self._pending_constellation = None
        self._pending_impulse_response = None

    def update_spectrum(self, freqs: np.ndarray, power_db: np.ndarray) -> None:
        self._spectrum_curve.setData(freqs, power_db)
        self._spectrum_range.apply(
            self._spectrum_plot, float(power_db.min()), float(power_db.max())
        )

    def update_power_envelope(
        self, time_axis: np.ndarray, power_db: np.ndarray, frame_times: list[float]
    ) -> None:
        self._power_curve.setData(time_axis, power_db)
        self._power_range.apply(self._power_plot, float(power_db.min()), float(power_db.max()))

        for marker, position in zip(self._null_markers, frame_times):
            marker.setPos(position)
            marker.setVisible(True)
        for marker in self._null_markers[len(frame_times) :]:
            marker.setVisible(False)

    def update_constellation(self, constellation: np.ndarray) -> None:
        # Only note it; drawing happens at the shared refresh rate.
        self._pending_constellation = constellation

    def update_impulse_response(self, delays: np.ndarray, power_db: np.ndarray) -> None:
        # Only note it; drawing happens at the shared refresh rate.
        self._pending_impulse_response = (delays, power_db)

    def render(self) -> None:
        self._render_constellation()
        self._render_impulse_response()

    def _on_density_toggled(self, density: bool) -> None:
        self._constellation_curve.setVisible(not density)
        self._constellation_density.setVisible(density)
        if self._shown_constellation is not None:
            self._draw_constellation(self._shown_constellation)

    def _render_constellation(self) -> None:
        constellation = self._pending_constellation
        if constellation is None:
            return
        self._pending_constellation = None
        self._shown_constellation = constellation
        self._draw_constellation(constellation)

    def _draw_constellation(self, constellation: np.ndarray) -> None:
        sample = constellation
        if len(sample) > MAX_CONSTELLATION_POINTS:
            sample = sample[:: len(sample) // MAX_CONSTELLATION_POINTS]
        # The sample suffices for a percentile and costs a fraction of the full frame.
        limit = constellation_limit(sample)

        if self._density_checkbox.isChecked():
            # All points: the histogram costs a few milliseconds even for a whole frame,
            # and drawing one image is cheaper than drawing the sample as points.
            extent = DENSITY_EXTENT * limit
            counts = constellation_density(constellation, extent, DENSITY_BINS)
            # The square root keeps the sparse tails visible next to the dense clusters.
            density = np.sqrt(counts)
            self._constellation_density.setImage(
                density, autoLevels=False, levels=(0.0, max(float(density.max()), 1.0))
            )
            self._constellation_density.setRect(-extent, -extent, 2 * extent, 2 * extent)
        else:
            self._constellation_curve.setData(sample.real, sample.imag)
        self._constellation_range.apply(self._constellation_plot, -limit, limit, axes="xy")

    def _render_impulse_response(self) -> None:
        if self._pending_impulse_response is None:
            return
        delays, power_db = self._pending_impulse_response
        self._pending_impulse_response = None

        self._impulse_curve.setData(delays, power_db)
        self._impulse_range.apply(self._impulse_plot, float(power_db.min()), float(power_db.max()))
