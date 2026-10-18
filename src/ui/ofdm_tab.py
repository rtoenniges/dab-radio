"""The "OFDM" tab: input and output of the FFT per symbol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from dab.constants import TransmissionMode
from dsp.pipeline import SymbolWindows
from dsp.tii import null_symbol_window_offset
from i18n import tr
from ui.plots import SmoothedRange

Colour = tuple[int, ...] | QColor | str


@dataclass(frozen=True)
class OfdmColours:
    """The colours of `OfdmPlots`, each anything `pg.mkColor` takes."""

    guard: Colour  # the guard intervals in the time signal
    window: Colour  # the FFT windows
    null_window: Colour  # the window in the null symbol
    magnitude: Colour  # the time signal
    guard_copy: Colour  # the guard intervals shifted over the end of their window
    boundary: Colour  # the sections of the frame in the FFT output
    minor_boundary: Colour  # the CIFs within the MSC
    carrier_edge: Colour  # the limits of the used carriers
    label_fill: Colour  # behind the names of the sections


DESKTOP_COLOURS = OfdmColours(
    guard=(255, 255, 255, 25),  # like the guard interval in the impulse response
    window=(70, 175, 185, 60),
    null_window=(190, 110, 200, 60),
    magnitude=(200, 200, 200),
    guard_copy=(255, 140, 0),
    # Red does not occur in the colour map (viridis), so the lines stay visible over
    # light and dark image areas alike.
    boundary=(255, 90, 90),
    minor_boundary=(255, 150, 150, 170),
    carrier_edge=(255, 255, 255, 140),
    label_fill=(0, 0, 0, 150),
)


def _opaque(colour: Colour) -> QColor:
    opaque = pg.mkColor(colour)
    opaque.setAlpha(255)
    return opaque


class OfdmPlots:
    """The two plots of the "FFT per symbol" block, without titles or explanations: the
    desktop's tab (`OfdmTab`) and a page of the touch surface each label them their own
    way.

    `time_plot` shows the time signal at the start of the frame and the sections the
    demodulator cuts it into, `spectrum_plot` the FFT output, one row per symbol.
    """

    # Labels of the symbol axis in the FFT output (ETSI EN 300 401, 14.4.1). With all
    # 76 rows the rows are too narrow to label PRS and FIC individually; then only
    # every fifth or tenth.
    _SYMBOL_TICKS_FINE: ClassVar[list[tuple[int, str]]] = [
        (0, "0 Null"),
        (1, "1 PRS"),
        (2, "2 FIC"),
        (5, "5 MSC"),
        *((symbol, str(symbol)) for symbol in range(10, 80, 5)),
    ]
    _SYMBOL_TICKS_COARSE: ClassVar[list[tuple[int, str]]] = [
        (0, "0 Null"),
        (5, "5 MSC"),
        *((symbol, str(symbol)) for symbol in range(10, 80, 10)),
    ]
    _MIN_ROW_PIXELS_FOR_FINE_TICKS = 14
    # Smallest span of the colour scale in dB (see `_levels`).
    _MIN_LEVEL_SPAN_DB = 10.0

    def __init__(
        self,
        mode: TransmissionMode,
        sample_rate: float,
        excerpt_symbols: int,
        colours: OfdmColours = DESKTOP_COLOURS,
    ) -> None:
        self._mode = mode
        self._sample_rate = sample_rate
        self._pending: SymbolWindows | None = None
        self._rows: tuple[int, int] | None = None  # (first symbol l, number of rows)
        self._symbol_ticks: list[tuple[int, str]] | None = None

        self.time_plot = pg.PlotWidget()
        self.time_plot.setLabel(
            "bottom", tr("Time from the start of the PRS (frame position)"), units="s"
        )
        self.time_plot.setLabel("left", tr("Magnitude |x|"))
        # Show the raw values, without a factor like "x0.001" on the axis.
        self.time_plot.getAxis("left").enableAutoSIPrefix(False)
        self.time_plot.enableAutoRange(x=False, y=False)
        self.time_plot.setXRange(
            -mode.null_symbol_length / sample_rate,
            excerpt_symbols * mode.symbol_length / sample_rate,
            padding=0.01,
        )
        self._time_range = SmoothedRange()

        # The sections are fixed: measured from the PRS, exactly as the demodulator
        # cut up the frame after the fine alignment. Each region with the name of its
        # colour.
        self._regions: list[tuple[pg.LinearRegionItem, str]] = []
        ticks = []
        null_window = null_symbol_window_offset(mode)
        self._add_region(null_window, null_window + mode.fft_length, "null_window")
        ticks.append((null_window + mode.fft_length / 2, tr("Null symbol · TII")))
        for symbol in range(1, excerpt_symbols + 1):
            start = (symbol - 1) * mode.symbol_length
            window = start + mode.guard_length
            self._add_region(start, window, "guard")
            self._add_region(window, start + mode.symbol_length, "window")
            name = "PRS" if symbol == 1 else "FIC" if symbol <= 4 else "MSC"
            ticks.append((start + mode.guard_length / 2, "Guard"))
            ticks.append((window + mode.fft_length / 2, f"l = {symbol} · {name}"))
        top_axis = self.time_plot.getAxis("top")
        top_axis.setTicks([[(position / sample_rate, text) for position, text in ticks]])
        self.time_plot.showAxis("top")

        self._magnitude_curve = self.time_plot.plot()
        self._guard_copy_curve = self.time_plot.plot(connect="finite")

        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setLabel("bottom", tr("FFT bin k (spacing 1 kHz, 0 = channel centre)"))
        self.spectrum_plot.setLabel("left", tr("Symbol l"))
        self.spectrum_plot.invertY(True)
        self.spectrum_plot.enableAutoRange(x=False, y=False)
        self.spectrum_plot.setXRange(
            -mode.fft_length / 2 - 0.5, mode.fft_length / 2 - 0.5, padding=0
        )
        self._image = pg.ImageItem(axisOrder="row-major")
        self.spectrum_plot.addItem(self._image)
        self.colour_bar = pg.ColorBarItem(
            colorMap=pg.colormap.get("viridis"), label=tr("Power |Z|² (dB)"), interactive=False
        )
        self.colour_bar.setImageItem(self._image, insert_in=self.spectrum_plot.getPlotItem())
        self._level_range = SmoothedRange(padding=0.0)
        # Limits of the used carriers k = -768 ... 768.
        half = mode.num_carriers // 2
        self._carrier_edges = [
            pg.InfiniteLine(edge, angle=90, movable=False) for edge in (-half - 0.5, half + 0.5)
        ]
        for edge in self._carrier_edges:
            self.spectrum_plot.addItem(edge)

        # Boundaries of the frame sections (ETSI EN 300 401, 14.4): null symbol, PRS
        # (l = 1), FIC (l = 2-4) and MSC (l = 5-76). Each line is the top edge of the
        # named section. The labels sit in the left margin, where only out-of-band
        # noise is, instead of over the carriers. With all 76 symbols PRS and FIC are
        # only one and three rows high, so their texts sit side by side.
        self._boundaries: list[tuple[pg.InfiniteLine, bool]] = []
        cif_symbols = (mode.symbols_per_frame - 4) // 4
        sections = ((0.5, "PRS", 0.01), (1.5, "FIC", 0.08), (4.5, "MSC · CIF 0", 0.01))
        for row, name, position in sections:
            self._add_section_boundary(row, name, position)
        # The four CIFs of the MSC, `cif_symbols` symbols each (ETSI EN 300 401, 14.4.2).
        # CIF 0 starts with the MSC and already has its line.
        for cif in range(1, 4):
            self._add_section_boundary(4.5 + cif * cif_symbols, f"CIF {cif}", 0.01, minor=True)
        self.set_colours(colours)

    def set_colours(self, colours: OfdmColours) -> None:
        for region, name in self._regions:
            region.setBrush(pg.mkBrush(getattr(colours, name)))
        self._magnitude_curve.setPen(pg.mkPen(colours.magnitude))
        self._guard_copy_curve.setPen(pg.mkPen(colours.guard_copy))
        for edge in self._carrier_edges:
            edge.setPen(pg.mkPen(colours.carrier_edge, style=Qt.PenStyle.DashLine))
        for line, minor in self._boundaries:
            colour = colours.minor_boundary if minor else colours.boundary
            style = Qt.PenStyle.DashLine if minor else Qt.PenStyle.SolidLine
            line.setPen(pg.mkPen(colour, style=style))
            line.label.setColor(_opaque(colour))
            line.label.fill = pg.mkBrush(colours.label_fill)
            line.label.update()

    def _add_section_boundary(
        self, row: float, name: str, position: float, minor: bool = False
    ) -> None:
        """Horizontal boundary line with the section name beneath it.

        `position` is the horizontal position of the label as a fraction of the plot
        width; `minor` draws the line weaker and dashed (the CIF boundaries inside the
        MSC).
        """
        line = pg.InfiniteLine(
            row,
            angle=0,
            movable=False,
            label=name,
            labelOpts={
                "position": position,
                # Text corner top left on the line: the label thereby sits inside the
                # section it names (the axis is inverted, larger l lies further down).
                "anchors": [(0, 0), (0, 0)],
            },
        )
        self.spectrum_plot.addItem(line)
        self._boundaries.append((line, minor))

    def _add_region(self, start: int, end: int, colour: str) -> None:
        region = pg.LinearRegionItem(
            (start / self._sample_rate, end / self._sample_rate),
            movable=False,
            pen=pg.mkPen(None),
        )
        region.setZValue(-10)
        self.time_plot.addItem(region)
        self._regions.append((region, colour))

    def update(self, windows: SymbolWindows) -> None:
        """Notes the newest frame; drawing happens in `render()`."""
        self._pending = windows

    def clear(self) -> None:
        """Drops the frame noted for drawing."""
        self._pending = None

    def render(self) -> None:
        windows = self._pending
        if windows is None:
            return
        self._pending = None
        self._render_time(windows)
        self._render_spectra(windows)

    def _render_time(self, windows: SymbolWindows) -> None:
        mode = self._mode
        magnitude = windows.magnitude
        positions = windows.excerpt_start + np.arange(len(magnitude))
        self._magnitude_curve.setData(positions / self._sample_rate, magnitude)

        # Every guard interval twice: in its place and shifted by the useful symbol
        # duration, over the end of the FFT window it repeats. NaN separates the
        # sections.
        copy_x: list[np.ndarray] = []
        copy_y: list[np.ndarray] = []
        gap = np.array([np.nan])
        excerpt_end = windows.excerpt_start + len(magnitude)
        for start in range(0, excerpt_end - mode.symbol_length + 1, mode.symbol_length):
            guard = np.arange(start, start + mode.guard_length)
            values = magnitude[guard - windows.excerpt_start]
            for shift in (0, mode.fft_length):
                copy_x += [(guard + shift) / self._sample_rate, gap]
                copy_y += [values, gap]
        if copy_x:
            self._guard_copy_curve.setData(np.concatenate(copy_x), np.concatenate(copy_y))

        if len(magnitude):
            self._time_range.apply(self.time_plot, 0.0, float(magnitude.max()))

    def _levels(self, power_db: np.ndarray, first_symbol: int) -> tuple[float, float]:
        """Limits of the colour scale, aligned to the used carriers.

        Taken over the whole image the scale would be useless: the empty bins and the
        null symbol lie 30 to 45 dB below the carriers, so the scale would be spent
        almost entirely on noise and all carriers would get the same colour. Only the
        data symbols on the used carriers count, therefore; whatever lies below stays
        visible in the darkest colour.

        The lower limit follows their 1st percentile but keeps at least
        `_MIN_LEVEL_SPAN_DB` below the upper limit. Without that minimum span a flat
        channel, the normal case with good reception, would be stretched over the full
        colour scale and a noise ripple of one or two dB would look like a deep notch.
        """
        center = self._mode.fft_length // 2
        half = self._mode.num_carriers // 2
        # Without the null symbol (row 0): it carries no data.
        carriers = power_db[1 if first_symbol == 0 else 0 :, center - half : center + half + 1]
        high = float(carriers.max())
        return min(float(np.percentile(carriers, 1)), high - self._MIN_LEVEL_SPAN_DB), high

    def _render_spectra(self, windows: SymbolWindows) -> None:
        rows = windows.spectra
        first_symbol = 1
        if windows.null_spectrum is not None:
            rows = np.vstack((windows.null_spectrum, rows))
            first_symbol = 0
        # In float32: accurate enough for the display (deviation below 1e-5 dB) and a
        # good three times faster than in complex128.
        rows = rows.astype(np.complex64)
        power_db = rows.real * rows.real + rows.imag * rows.imag
        np.log10(np.maximum(power_db, 1e-30, out=power_db), out=power_db)
        power_db *= 10.0

        new_levels = self._level_range.update(*self._levels(power_db, first_symbol))
        self._image.setImage(power_db, autoLevels=False)
        if new_levels is not None:
            self.colour_bar.setLevels(new_levels)

        # After fftshift bin i sits at k = i - fft_length/2; row j is symbol first_symbol + j.
        fft_length = self._mode.fft_length
        self._image.setRect(
            QRectF(-fft_length / 2 - 0.5, first_symbol - 0.5, fft_length, len(power_db))
        )
        rows_shape = (first_symbol, len(power_db))
        if rows_shape != self._rows:
            self._rows = rows_shape
            self.spectrum_plot.setYRange(
                first_symbol - 0.5, first_symbol + len(power_db) - 0.5, padding=0
            )
        row_pixels = self.spectrum_plot.getViewBox().height() / len(power_db)
        ticks = (
            self._SYMBOL_TICKS_FINE
            if row_pixels >= self._MIN_ROW_PIXELS_FOR_FINE_TICKS
            else self._SYMBOL_TICKS_COARSE
        )
        if ticks is not self._symbol_ticks:
            self._symbol_ticks = ticks
            self.spectrum_plot.getAxis("left").setTicks([ticks])


class OfdmTab(QWidget):
    """Shows what the "FFT per symbol" block receives and what it delivers.

    On top the time signal at the start of the frame and the sections the demodulator
    cuts the frame into: per symbol a guard interval, which is discarded, and an FFT
    window of exactly `fft_length` samples. The symbols are thus separated in the time
    domain before any FFT runs.

    Below the FFT output: one row per symbol, each the FFT of exactly one window from
    above, one column per bin. Here the FFT separates the carriers.
    """

    def __init__(
        self,
        mode: TransmissionMode,
        sample_rate: float,
        excerpt_symbols: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._plots = OfdmPlots(mode, sample_rate, excerpt_symbols)
        cif_symbols = (mode.symbols_per_frame - 4) // 4
        half = mode.num_carriers // 2

        time_plot = self._plots.time_plot
        time_plot.setTitle(
            tr("Time signal at the start of the frame: guard intervals and FFT windows")
        )
        time_plot.setToolTip(
            tr(
                "The start of the frame as the demodulator cuts it up: the null symbol, then "
                "the phase reference symbol (PRS, l = 1) and the first FIC symbols. Each "
                "symbol consists of {guard} samples of guard interval (grey) and {fft} samples "
                "of useful part (turquoise). Only the useful part goes into the FFT - exactly "
                "{milliseconds:g} ms of signal -, the guard interval is discarded.\n\n"
                "So the symbols are separated here in the time domain, before the FFT, on the "
                "grid the frame synchronisation delivers. The signal itself does not show the "
                "boundaries, it looks like noise.\n\n"
                "Orange: the samples of the guard interval, additionally shifted {fft} samples "
                "to the right. There they coincide with the end of the FFT window, for the "
                "transmitter repeated exactly that end at the front as the guard interval "
                "(cyclic prefix). Where the two diverge, noise, echoes of the previous symbol "
                "or a shifted frame position are at work.\n\n"
                "In the null symbol (violet) the window lies in the middle; its FFT serves the "
                "transmitter identification (TII). It is not entirely quiet there: what remains "
                "is what does not belong to the symbols of this ensemble - noise, the "
                "transmitter identification and the skirts of occupied neighbouring channels, "
                "which lie at the edges in the FFT output below."
            ).format(
                guard=mode.guard_length,
                fft=mode.fft_length,
                milliseconds=mode.fft_length / sample_rate * 1e3,
            )
        )

        spectrum_plot = self._plots.spectrum_plot
        spectrum_plot.setTitle(tr("FFT output: one row per symbol, one column per bin"))
        spectrum_plot.setToolTip(
            tr(
                "Each row is the FFT of exactly one FFT window from the diagram above: row 0 "
                "the null symbol, 1 the PRS, 2 to 4 the FIC, from 5 on the MSC. Each column is "
                "a bin, a frequency slice of 1 kHz. After the frequency correction carrier k "
                "lies exactly in bin k.\n\n"
                "Occupied are the {carriers} carriers between the dashed lines (k = -{half} to "
                "{half}), without k = 0 in the middle. Outside lies only noise; at the edges an "
                "occupied neighbouring channel may reach in.\n\n"
                "The power of a carrier stays almost the same across the frame but changes "
                "across the frequency: darker vertical stripes are notches from multipath "
                "reception, the counterpart of the peaks of the impulse response. The null "
                "symbol is empty but for the carrier pairs of the transmitter identification, "
                "if it carries one.\n\n"
                "The red lines separate the sections of the frame: null symbol, phase "
                "reference symbol, FIC and MSC. Dashed within them the four CIFs of the MSC, "
                "{cif_symbols} symbols each.\n\n"
                "Without a selected station only the first 16 data symbols are demodulated, "
                "with a station all {symbols}. Power unsmoothed per frame; only the colour "
                "scale follows the level slowly."
            ).format(
                carriers=mode.num_carriers,
                half=half,
                cif_symbols=cif_symbols,
                symbols=mode.symbols_per_frame - 1,
            )
        )

        # Input and output of the FFT per symbol one above the other: the windows in
        # the time signal on top, one row of FFT output per window below.
        layout = QVBoxLayout()
        layout.addWidget(time_plot, stretch=2)
        layout.addWidget(spectrum_plot, stretch=3)
        self.setLayout(layout)

    def update(self, windows: SymbolWindows) -> None:
        """Notes the newest frame; drawing happens in `render()`."""
        self._plots.update(windows)

    def render(self) -> None:
        self._plots.render()
