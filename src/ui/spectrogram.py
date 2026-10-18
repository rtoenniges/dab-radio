"""What the detail view shows of a captured stretch of signal (`dsp.signal_capture`): a
window of it in time and frequency as a field of levels, without widgets.

The rows of the field are spectra one after another in time, cut in one of two ways:

- `Rows.SLIDING`: an FFT every few microseconds, wherever it falls. The null symbols
  show as trenches, the TII carriers some transmitters send in them as a comb. Short
  FFTs resolve time and blur frequency; one as long as the useful part of a symbol
  (2048 samples, 1 kHz per bin, without a window function, as the demodulator takes
  it) gives each carrier a bin of its own, but only where it lies within one symbol:
  there the spectrum is the channel, level to within a decibel; where it reaches into
  the next symbol, the two interfere and it turns grainy (some 5 dB spread over the
  carriers).
- `Rows.SYMBOLS`: one row per OFDM symbol, its FFT exactly where the demodulator takes
  it (the useful part; in the null symbol the window of the TII). Each column is one
  carrier, k kHz from the centre of the channel; the one at the centre is never sent.

Levels are in dB relative to the plateau of the ensemble block, so the field looks the
same at any gain. With either, the field tells where in the window the null symbols,
the guard intervals and the useful parts of the symbols lie (`Part`): a sliding FFT of
2048 is level only around the middle of a symbol, where its window holds that symbol
alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from dsp.signal_capture import SignalCapture
from dsp.tii import null_symbol_window_offset

FFT_SIZES = (256, 512, 1024, 2048)
DEFAULT_FFT_SIZE = 512
# The stretch of time shown (ms) and the width of frequency (kHz; the whole sampled
# band is 2048 kHz, the ensemble 1536).
WINDOWS_MS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 96.0, 128.0)
DEFAULT_WINDOW_MS = 4.0
SPANS_KHZ = (2048, 1024, 512, 256, 128, 64, 32)
DEFAULT_SPAN_KHZ = 2048
# A null symbol jumped to stands this far into the window: its trench in the back
# quarter, what follows it in front.
NULL_LEAD = 0.2

# At most this many rows and columns, unless a model is given others (`Spectrogram`):
# the graphics card of the Raspberry Pi draws them twice a second's worth of times
# over, and each takes an FFT.
MAX_ROWS = 200
MAX_COLUMNS = 1024
# Rows of the sliding FFT come at least this many samples apart (7.8 µs).
MIN_HOP = 16
# Up to this many columns (or symbol rows) are drawn as blocks of their own, so that a
# single carrier can be told from its neighbours; more would be thinner than a pixel.
DISCRETE_COLUMNS = 160
DISCRETE_ROWS = 40

# The levels drawn, relative to the plateau: from the floor of the field, a range of
# LEVEL_RANGES_DB below TOP_DB, to its top. The largest puts the floor a little above the
# noise beside a strong ensemble (FLOOR_DB); the smaller ones show how level the plateau
# is: a sliding FFT of 2048 within a symbol spreads 0.6 to 0.9 dB over the carriers,
# reaching into the next one 4 to 6 dB, which 50 dB of height hide.
TOP_DB = 5.0
LEVEL_RANGES_DB = (10.0, 20.0, 30.0, 50.0)
DEFAULT_LEVEL_RANGE_DB = 50.0
FLOOR_DB = TOP_DB - DEFAULT_LEVEL_RANGE_DB

# How many FFTs at random places of the first frame give the level of the plateau.
_REFERENCE_FFTS = 48
# Of each column of the overview, at most this many FFTs are averaged.
_OVERVIEW_FFTS = 6
_OVERVIEW_FFT_SIZE = 256


class Rows(Enum):
    SLIDING = "sliding"
    SYMBOLS = "symbols"


class Part(Enum):
    """A stretch of the frame: the null symbol, the guard interval of a symbol (a copy
    of the end of its useful part), the useful part (the window of its FFT)."""

    NULL = "null"
    GUARD = "guard"
    USEFUL = "useful"


@dataclass(frozen=True)
class Heightfield:
    """Levels over time and frequency, ready to be drawn."""

    levels_db: np.ndarray  # (rows, columns), relative to the plateau, float32
    frequencies_khz: np.ndarray  # middle of each column, from the centre of the channel
    column_khz: float  # width of a column
    row_starts_ms: np.ndarray  # where each row's time begins, from the start of the capture
    row_ends_ms: np.ndarray
    discrete_columns: bool  # each column a block of its own
    discrete_rows: bool  # each row a slab of its own (one per symbol)
    window_ms: tuple[float, float]  # the time shown
    span_khz: tuple[float, float]  # the frequency shown
    # The symbol of each row (0 for the null symbol, 1 for the PRS ...); None for the
    # sliding FFT.
    symbols: np.ndarray | None = None
    # Of symbol rows: where each FFT takes its samples (ms of the capture), the useful
    # part, in the null symbol the window of the TII; None for the sliding FFT.
    fft_starts_ms: np.ndarray | None = None
    fft_ends_ms: np.ndarray | None = None
    # The frame structure in and around the window: where each stretch begins and
    # ends (ms of the capture) and what it is; before the first null symbol only as
    # far as the capture holds the FFT of a symbol.
    parts: tuple[tuple[float, float, Part], ...] = ()
    # Where the figures of the time axis have their 0 (ms of the capture).
    time_origin_ms: float = 0.0


@dataclass(frozen=True)
class Position:
    """Where in the frame structure a moment of the capture lies."""

    frame: int  # 0-based, in the capture
    symbol: int  # 0 = null symbol, 1 = PRS, 2..76 the data symbols


class Spectrogram:
    """A capture and the window onto it: the rows, the FFT, the time and frequency shown.

    Setting a value keeps it within what the capture holds; `heightfield` computes what
    to draw from the current values, with at most `max_rows` rows and `max_columns`
    columns.
    """

    def __init__(self, max_rows: int = MAX_ROWS, max_columns: int = MAX_COLUMNS) -> None:
        self.max_rows = max_rows
        self.max_columns = max_columns
        self._capture: SignalCapture | None = None
        self._samples: np.ndarray | None = None  # the frequency offset rotated out
        self._references: dict[tuple[Rows, int], float] = {}
        self.rows = Rows.SLIDING
        self.fft_size = DEFAULT_FFT_SIZE
        self.window_ms = DEFAULT_WINDOW_MS
        self.span_khz = float(DEFAULT_SPAN_KHZ)
        self.centre_khz = 0.0
        self.start_ms = 0.0
        # Where the time axis has its 0, in ms of the capture.
        self.time_origin_ms = 0.0

    @property
    def capture(self) -> SignalCapture | None:
        return self._capture

    def clear(self) -> None:
        """Forgets the capture: there is nothing to show until the next one."""
        self._capture = None
        self._samples = None
        self._references.clear()

    @property
    def duration_ms(self) -> float:
        if self._capture is None:
            return 0.0
        return 1000 * self._capture.duration_s

    def set_capture(self, capture: SignalCapture) -> None:
        """Shows a new capture. The first opens on its first null symbol; a new one
        keeps the window where it was, and that is the same place in the frames: every
        capture begins the same margin before a null symbol (`dsp.signal_capture`).
        Only the frame with the TII may be the other one."""
        first = self._capture is None
        self._capture = capture
        n = np.arange(len(capture.samples))
        rotation = np.exp(
            (-2j * np.pi * capture.frequency_offset_hz / capture.sample_rate) * n
        ).astype(np.complex64)
        self._samples = capture.samples * rotation
        self._references.clear()
        if first:
            self.start_ms = 0.0
            self.jump_to_null(1)
        else:
            self._clamp()

    # --- moving the window ----------------------------------------------------------

    def _ms(self, samples: float) -> float:
        return 1000 * samples / self._capture.sample_rate

    def _samples_of(self, ms: float) -> int:
        return round(ms * self._capture.sample_rate / 1000)

    def _clamp(self) -> None:
        self.start_ms = min(max(self.start_ms, 0.0), max(self.duration_ms - self.window_ms, 0.0))
        half = self.span_khz / 2
        limit = max(1024.0 - half, 0.0)
        self.centre_khz = min(max(self.centre_khz, -limit), limit)

    def scroll(self, ms: float) -> None:
        """Moves the window on in time (back for negative `ms`)."""
        self.start_ms += ms
        self._clamp()

    def scroll_to(self, start_ms: float) -> None:
        self.start_ms = start_ms
        self._clamp()

    def set_window_ms(self, window_ms: float) -> None:
        """Shows this much time, around the same middle."""
        middle = self.start_ms + self.window_ms / 2
        self.window_ms = window_ms
        self.start_ms = middle - window_ms / 2
        self._clamp()

    def set_span_khz(self, span_khz: float) -> None:
        """Shows this much frequency, around the same centre."""
        self.span_khz = float(span_khz)
        self._clamp()

    def set_centre_khz(self, centre_khz: float) -> None:
        self.centre_khz = centre_khz
        self._clamp()

    def jump_to_null(self, direction: int) -> None:
        """Puts the next null symbol after (1) or before (-1) the one in view, or the
        place a null symbol stands at when jumped to, at `NULL_LEAD` of the window."""
        if self._capture is None:
            return
        nulls = [self._ms(s) for s in self._capture.null_starts]
        anchor = self.start_ms + NULL_LEAD * self.window_ms
        if direction > 0:
            later = [t for t in nulls if t > anchor + 0.01]
            target = later[0] if later else None
        else:
            earlier = [t for t in nulls if t < anchor - 0.01]
            target = earlier[-1] if earlier else None
        if target is not None:
            self.start_ms = target - NULL_LEAD * self.window_ms
            self._clamp()

    def position(self, ms: float) -> Position | None:
        """The frame and symbol at a moment of the capture; None before the first frame."""
        capture = self._capture
        if capture is None:
            return None
        sample = self._samples_of(ms)
        mode = capture.mode
        frames = np.flatnonzero(capture.null_starts <= sample)
        if not len(frames):
            return None
        frame = int(frames[-1])
        into = sample - int(capture.frame_starts[frame])
        if into < 0:
            return Position(frame, 0)
        symbol = min(into // mode.symbol_length + 1, mode.symbols_per_frame)
        return Position(frame, int(symbol))

    # --- the field --------------------------------------------------------------------

    def heightfield(self) -> Heightfield | None:
        if self._capture is None:
            return None
        if self.rows is Rows.SYMBOLS:
            return self._symbol_field()
        return self._sliding_field()

    def _columns(self, size: int) -> tuple[np.ndarray, np.ndarray]:
        """The bins of an FFT of `size` (after fftshift) within the frequency shown, and
        their frequencies in kHz."""
        khz = (np.arange(size) - size // 2) * self._capture.sample_rate / size / 1000
        low, high = self.centre_khz - self.span_khz / 2, self.centre_khz + self.span_khz / 2
        inside = np.flatnonzero((khz >= low - 1e-9) & (khz < high - 1e-9))
        return inside, khz[inside]

    def _spectra_db(self, starts: np.ndarray, size: int, average: int = 1) -> np.ndarray:
        """Power (dB, fftshifted) of FFTs of `size` from each of `starts`, each the mean
        of `average` consecutive ones; rectangular at the length of the useful part, as
        the demodulator, else with a Hann window."""
        samples = self._samples
        mode = self._capture.mode
        starts = np.clip(starts, 0, len(samples) - size * average)
        offsets = np.arange(size * average).reshape(average, size)
        cut = samples[starts[:, None, None] + offsets[None, :, :]]
        if size != mode.fft_length:
            cut = cut * np.hanning(size).astype(np.float32)
        power = np.abs(np.fft.fft(cut, axis=-1)) ** 2
        power = np.fft.fftshift(power.mean(axis=1), axes=-1)
        return 10 * np.log10(np.maximum(power, 1e-20))

    def _reference_db(self, rows: Rows, size: int) -> float:
        """The level of the plateau in this kind of row: the median over the carriers
        of the ensemble of the mean spectrum of data symbols in the first frame, or of
        those of its symbols a short stretch holds (`NullSnapshotter`)."""
        key = (rows, size)
        if key not in self._references:
            mode = self._capture.mode
            first = int(self._capture.frame_starts[0])
            if rows is Rows.SYMBOLS:
                starts = first + mode.guard_length + mode.symbol_length * np.arange(4, 4 + 32)
            else:
                rng = np.random.default_rng(0)
                low = first + 5 * mode.symbol_length
                high = first + (mode.symbols_per_frame - 1) * mode.symbol_length
                if high + size > len(self._samples):
                    low, high = first, len(self._samples) - size
                starts = low + rng.integers(0, max(high - low, 1), _REFERENCE_FFTS)
            power = 10 ** (self._spectra_db(starts, size) / 10)
            khz = (np.arange(size) - size // 2) * self._capture.sample_rate / size / 1000
            inner = (np.abs(khz) < 0.9 * mode.num_carriers / 2) & (khz != 0)
            self._references[key] = float(10 * np.log10(np.median(power.mean(axis=0)[inner])))
        return self._references[key]

    def _fit_columns(
        self, levels: np.ndarray, khz: np.ndarray, column_khz: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """At most `max_columns`: neighbouring columns merged, keeping the highest (a
        single TII carrier must not vanish)."""
        factor = math.ceil(levels.shape[1] / self.max_columns)
        if factor <= 1:
            return levels, khz, column_khz
        keep = levels.shape[1] // factor * factor
        merged = levels[:, :keep].reshape(levels.shape[0], -1, factor).max(axis=2)
        return merged, khz[:keep].reshape(-1, factor).mean(axis=1), column_khz * factor

    def _sliding_rows(self) -> tuple[int, float]:
        """How many rows the sliding FFT has and how many samples apart they are."""
        length = max(self._samples_of(self.window_ms), 2 * MIN_HOP)
        rows = int(min(self.max_rows, max(2, length // MIN_HOP)))
        return rows, length / rows

    @property
    def row_step_us(self) -> float:
        """How far apart the rows of the sliding FFT are (µs); 0 without a capture."""
        if self._capture is None:
            return 0.0
        return 1e6 * self._sliding_rows()[1] / self._capture.sample_rate

    def _averaged(self, hop: float) -> int:
        """How many FFTs one row of the sliding FFT averages: zoomed out, a row covers
        more than one, and as many as fit into it are taken, so that hardly a sample is
        skipped."""
        return max(1, int(hop // self.fft_size))

    @property
    def row_length_us(self) -> float:
        """How much of the signal one row of the sliding FFT is computed from (µs): the
        stretch it smears over in time. 0 without a capture."""
        if self._capture is None:
            return 0.0
        samples = self.fft_size * self._averaged(self._sliding_rows()[1])
        return 1e6 * samples / self._capture.sample_rate

    def _sliding_field(self) -> Heightfield:
        capture = self._capture
        size = self.fft_size
        start = self._samples_of(self.start_ms)
        rows, hop = self._sliding_rows()
        average = self._averaged(hop)
        # The rows stand evenly over the window, each for the middle of its stretch, and
        # each FFT is centred on it: begun there instead, every row would show what lies
        # half an FFT later, and the landscape would leave the back of the window empty
        # and push the rest over its front edge, the more the shorter the window (half of
        # it with an FFT of 2048 over 1 ms).
        middle = start + (np.arange(rows) + 0.5) * hop
        starts = np.round(middle - size * average / 2).astype(np.int64)
        columns, khz = self._columns(size)
        levels = self._spectra_db(starts, size, average)[:, columns]
        levels -= self._reference_db(Rows.SLIDING, size)
        column_khz = capture.sample_rate / size / 1000
        levels, khz, column_khz = self._fit_columns(levels, khz, column_khz)
        row_ms = 1000 * middle / capture.sample_rate
        half = 1000 * hop / capture.sample_rate / 2
        return Heightfield(
            levels_db=levels.astype(np.float32),
            frequencies_khz=khz,
            column_khz=column_khz,
            row_starts_ms=row_ms - half,
            row_ends_ms=row_ms + half,
            discrete_columns=levels.shape[1] <= DISCRETE_COLUMNS,
            discrete_rows=False,
            window_ms=(self.start_ms, self.start_ms + self.window_ms),
            time_origin_ms=self.time_origin_ms,
            span_khz=(self.centre_khz - self.span_khz / 2, self.centre_khz + self.span_khz / 2),
            parts=self._parts(),
        )

    def _parts(self) -> tuple[tuple[float, float, Part], ...]:
        """The stretches of the frames that reach into the window, see `Heightfield`."""
        capture = self._capture
        guard = capture.mode.guard_length
        begins, ends, _windows, symbols = self._symbol_rows()
        low = self._samples_of(self.start_ms)
        high = self._samples_of(self.start_ms + self.window_ms)
        parts: list[tuple[float, float, Part]] = []
        for i in np.flatnonzero((ends > low) & (begins < high)):
            begin, end = int(begins[i]), int(ends[i])
            if symbols[i] == 0:
                parts.append((self._ms(begin), self._ms(end), Part.NULL))
            else:
                parts.append((self._ms(begin), self._ms(begin + guard), Part.GUARD))
                parts.append((self._ms(begin + guard), self._ms(end), Part.USEFUL))
        return tuple(parts)

    def _symbol_rows(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Every symbol of the capture: where its time begins and ends, where its FFT
        window begins, and its number in the frame. Before the first null symbol, the
        end of the frame before, as far as the capture holds the FFT of a symbol."""
        capture = self._capture
        mode = capture.mode
        null_offset = null_symbol_window_offset(mode)
        begins, ends, windows, symbols = [], [], [], []
        end = int(capture.frame_starts[0]) - mode.null_symbol_length
        for symbol in range(mode.symbols_per_frame, 0, -1):
            begin = end - mode.symbol_length
            if begin + mode.guard_length < 0:
                break
            begins.insert(0, begin)
            ends.insert(0, end)
            windows.insert(0, begin + mode.guard_length)
            symbols.insert(0, symbol)
            end = begin
        for prs in capture.frame_starts:
            prs = int(prs)
            begins.append(prs - mode.null_symbol_length)
            ends.append(prs)
            windows.append(prs + null_offset)
            symbols.append(0)
            for symbol in range(1, mode.symbols_per_frame + 1):
                begin = prs + (symbol - 1) * mode.symbol_length
                begins.append(begin)
                ends.append(begin + mode.symbol_length)
                windows.append(begin + mode.guard_length)
                symbols.append(symbol)
        return np.array(begins), np.array(ends), np.array(windows), np.array(symbols)

    def _symbol_field(self) -> Heightfield:
        capture = self._capture
        size = capture.mode.fft_length
        begins, ends, windows, symbols = self._symbol_rows()
        low, high = (
            self._samples_of(self.start_ms),
            self._samples_of(self.start_ms + self.window_ms),
        )
        shown = np.flatnonzero((ends > low) & (begins < high))
        if len(shown) < 2:
            # Fewer than two symbols in the window: the one in it and the next.
            first = int(np.searchsorted(ends, low, side="right"))
            first = min(first, len(begins) - 2)
            shown = np.arange(first, first + 2)
        shown = shown[: self.max_rows]
        columns, khz = self._columns(size)
        levels = self._spectra_db(windows[shown], size)[:, columns]
        levels -= self._reference_db(Rows.SYMBOLS, size)
        column_khz = capture.sample_rate / size / 1000
        levels, khz, column_khz = self._fit_columns(levels, khz, column_khz)
        return Heightfield(
            levels_db=levels.astype(np.float32),
            frequencies_khz=khz,
            column_khz=column_khz,
            row_starts_ms=1000 * begins[shown] / capture.sample_rate,
            row_ends_ms=1000 * ends[shown] / capture.sample_rate,
            discrete_columns=levels.shape[1] <= DISCRETE_COLUMNS,
            discrete_rows=len(shown) <= DISCRETE_ROWS,
            window_ms=(self.start_ms, self.start_ms + self.window_ms),
            time_origin_ms=self.time_origin_ms,
            span_khz=(self.centre_khz - self.span_khz / 2, self.centre_khz + self.span_khz / 2),
            symbols=symbols[shown],
            fft_starts_ms=1000 * windows[shown] / capture.sample_rate,
            fft_ends_ms=1000 * (windows[shown] + size) / capture.sample_rate,
            parts=self._parts(),
        )

    # --- the overview -------------------------------------------------------------------

    def overview(self, columns: int) -> np.ndarray | None:
        """The whole capture at a glance: levels (dB relative to the plateau) of
        `_OVERVIEW_FFT_SIZE` bins, the lowest frequency first, for `columns` stretches of
        time, (bins, columns)."""
        if self._capture is None or columns < 1:
            return None
        size = _OVERVIEW_FFT_SIZE
        total = len(self._samples) - size
        edges = np.linspace(0, total, columns + 1)
        # Of each stretch, FFTs spread over it, at most one per `size` samples.
        each = max(1, min(_OVERVIEW_FFTS, int((edges[1] - edges[0]) // size)))
        offsets = (np.arange(each) + 0.5) / each
        starts = (edges[:-1, None] + offsets[None, :] * (edges[1] - edges[0])).astype(np.int64)
        levels = self._spectra_db(starts.ravel(), size).reshape(columns, each, size)
        power = (10 ** (levels / 10)).mean(axis=1)
        levels = 10 * np.log10(np.maximum(power, 1e-20))
        return (levels - self._reference_db(Rows.SLIDING, size)).T.astype(np.float32)
