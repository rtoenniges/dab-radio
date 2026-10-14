"""The detail view of the signal: the frames the signal processing keeps when asked
(`dsp.signal_capture`), the field of levels computed from them (`ui.spectrogram`) and the
landscape built from that for OpenGL (`ui.waterfall_gl`), on a synthetic transmitter."""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dab.constants import MODE_I, SAMPLE_RATE
from dsp.pipeline import DspWorker
from dsp.signal_capture import (
    MARGIN_SAMPLES,
    SNAPSHOT_AFTER,
    SNAPSHOT_BEFORE,
    NullSnapshotter,
    SignalCapture,
    SignalCaptureRecorder,
)
from tests.dab_signal import build_stream, fib
from ui.spectrogram import (
    DISCRETE_COLUMNS,
    LEVEL_RANGES_DB,
    MAX_COLUMNS,
    MAX_ROWS,
    NULL_LEAD,
    TOP_DB,
    Part,
    Rows,
    Spectrogram,
)
from ui.waterfall_gl import (
    HALF_DEPTH,
    HALF_WIDTH,
    HEIGHT,
    STRIP_DEPTH,
    WaterfallCamera,
    WaterfallColours,
    frequency_ticks,
    grid_lines,
    landscape_grid,
    level_ticks,
    level_z,
    part_strip,
    time_ticks,
    time_x,
    triangle_strip,
)

FRAME = MODE_I.frame_length
NULL = MODE_I.null_symbol_length
SYMBOL = MODE_I.symbol_length
BLOCK = 2 * FRAME  # as the SDR worker reads


@pytest.fixture(scope="module")
def stream() -> np.ndarray:
    """Eight frames of a clean transmitter, 150 Hz off, starting 1000 samples into the
    first frame."""
    rng = np.random.default_rng(7)
    return build_stream([fib()] * 12, 8, rng, start_offset=1000, snr_db=40, freq_offset_hz=150)


def _feed(worker: DspWorker, stream: np.ndarray, block: int) -> None:
    for start in range(0, len(stream), block):
        worker.process_samples(stream[start : start + block].astype(np.complex64))


@pytest.fixture(scope="module")
def capture(stream: np.ndarray) -> SignalCapture:
    """Three frames, as the signal processing keeps them."""
    worker = DspWorker(SAMPLE_RATE, MODE_I)
    captures: list[SignalCapture] = []
    worker.signal_captured.connect(captures.append)
    worker.capture_signal(3)
    _feed(worker, stream, BLOCK // 2)
    assert len(captures) == 1
    return captures[0]


def _power(samples: np.ndarray) -> float:
    return float(np.mean(np.abs(samples) ** 2))


# --- what the signal processing keeps -------------------------------------------------------


def test_the_frames_come_whole_with_where_the_demodulator_found_them(capture) -> None:
    assert len(capture.frame_starts) == 3
    assert np.all(np.diff(capture.frame_starts) == FRAME)
    # The margin before the first null symbol, then the null symbol, then the PRS.
    assert capture.null_starts[0] == MARGIN_SAMPLES
    first = int(capture.frame_starts[0])
    assert _power(capture.samples[first - NULL + 100 : first - 100]) < 0.01 * _power(
        capture.samples[first : first + SYMBOL]
    )
    end = int(capture.frame_starts[-1]) + MODE_I.symbols_per_frame * SYMBOL
    assert len(capture.samples) == end + MARGIN_SAMPLES
    assert capture.samples.dtype == np.complex64
    assert capture.frequency_offset_hz == pytest.approx(150, abs=5)
    assert capture.duration_s == pytest.approx(len(capture.samples) / SAMPLE_RATE)


def test_the_recorder_waits_for_all_frames_and_their_samples() -> None:
    recorder = SignalCaptureRecorder(2, 10_000, MODE_I, SAMPLE_RATE)
    block = np.arange(BLOCK, dtype=np.complex64)
    recorder.add_block(0, block[:5_000])  # before the request: not kept
    recorder.add_block(5_000, block[5_000:])
    # Began before the request (its null symbol and margin reach in front of it).
    recorder.add_frame(10_000 + NULL, 0.0)
    recorder.add_frame(20_000, 1.0)
    assert recorder.capture() is None  # one frame of two
    recorder.add_frame(20_000 + FRAME, 3.0)
    assert recorder.frames_found == 2
    assert recorder.capture() is None  # the end of the second frame has not come
    recorder.add_block(BLOCK, np.arange(BLOCK, 2 * BLOCK, dtype=np.complex64))
    capture = recorder.capture()

    begin = 20_000 - NULL - MARGIN_SAMPLES
    assert capture.samples[0] == begin
    assert list(capture.frame_starts) == [20_000 - begin, 20_000 + FRAME - begin]
    assert capture.frequency_offset_hz == 2.0


def test_without_frames_the_recorder_keeps_two_frames_at_most() -> None:
    recorder = SignalCaptureRecorder(1, 0, MODE_I, SAMPLE_RATE)
    for k in range(20):
        recorder.add_block(k * 100_000, np.zeros(100_000, dtype=np.complex64))
    kept = sum(len(block) for _, block in recorder._blocks)
    assert kept <= 4 * FRAME + 100_000


def test_a_new_frequency_starts_the_capture_over(stream) -> None:
    worker = DspWorker(SAMPLE_RATE, MODE_I)
    captures: list[SignalCapture] = []
    worker.signal_captured.connect(captures.append)
    worker.capture_signal(1)
    worker.process_samples(stream[: BLOCK // 4])
    worker.restart_stream(1)
    _feed(worker, stream, BLOCK)
    assert len(captures) == 1
    assert len(captures[0].frame_starts) == 1


def test_the_snapshotter_cuts_the_stretch_around_a_null_symbol() -> None:
    snapshotter = NullSnapshotter(MODE_I, SAMPLE_RATE)
    block = np.arange(3 * FRAME, dtype=np.complex64)
    snapshotter.add_block(0, block[:FRAME])
    assert snapshotter.add_frame(FRAME + NULL, 2.0) is None  # its end has not come
    snapshotter.add_block(FRAME, block[FRAME:])
    snapshot = snapshotter.add_frame(FRAME + NULL, 2.0)

    begin = FRAME - SNAPSHOT_BEFORE
    assert len(snapshot.samples) == SNAPSHOT_BEFORE + SNAPSHOT_AFTER
    assert snapshot.samples[0] == begin and np.all(np.diff(snapshot.samples.real) == 1)
    assert snapshot.null_starts[0] == SNAPSHOT_BEFORE
    assert snapshot.start_sample == begin and snapshot.first_frame_number == 1
    assert snapshot.frequency_offset_hz == 2.0
    # Blocks of more than a few frames ago are let go.
    for k in range(3, 10):
        snapshotter.add_block(k * FRAME, np.zeros(FRAME, dtype=np.complex64))
    assert snapshotter.add_frame(FRAME + NULL, 0.0) is None
    assert sum(len(b) for _, b in snapshotter._blocks) <= 5 * FRAME


def test_the_signal_processing_cuts_a_stretch_from_every_frame_while_asked(stream) -> None:
    worker = DspWorker(SAMPLE_RATE, MODE_I)
    snapshots: list[SignalCapture] = []
    worker.null_snapshot_ready.connect(snapshots.append)
    worker.set_null_snapshots(True)
    _feed(worker, stream, BLOCK // 2)

    numbers = [s.first_frame_number for s in snapshots]
    assert len(numbers) >= 5 and np.all(np.diff(numbers) == 1)
    for snapshot in snapshots:
        null, prs = int(snapshot.null_starts[0]), int(snapshot.frame_starts[0])
        assert null == SNAPSHOT_BEFORE
        quiet = _power(snapshot.samples[null + 100 : prs - 100])
        assert quiet < 0.01 * _power(snapshot.samples[prs : prs + SYMBOL])
        assert snapshot.frequency_offset_hz == pytest.approx(150, abs=5)
    worker.set_null_snapshots(False)
    count = len(snapshots)
    _feed(worker, stream, BLOCK // 2)
    assert len(snapshots) == count


def test_a_short_stretch_has_the_plateau_and_the_end_of_the_frame_before(capture) -> None:
    snapshotter = NullSnapshotter(MODE_I, SAMPLE_RATE)
    snapshotter.add_block(0, capture.samples)
    snapshot = snapshotter.add_frame(int(capture.frame_starts[1]), capture.frequency_offset_hz)
    model = Spectrogram(max_rows=100, max_columns=512)
    model.fft_size = 1024
    model.set_capture(snapshot)
    field = model.heightfield()
    assert field.levels_db.shape == (100, 512)
    # The plateau from the few symbols the stretch holds, as from a whole frame.
    whole = Spectrogram()
    whole.set_capture(capture)
    assert model._reference_db(Rows.SLIDING, 1024) == pytest.approx(
        whole._reference_db(Rows.SLIDING, 1024), abs=0.5
    )
    # Before the null symbol, the last symbol of the frame before.
    assert [p[2] for p in field.parts[:3]] == [Part.GUARD, Part.USEFUL, Part.NULL]
    assert field.parts[1][0] < field.window_ms[0] < field.parts[1][1]


# --- the window and the field -----------------------------------------------------------------


@pytest.fixture
def model(capture) -> Spectrogram:
    model = Spectrogram()
    model.set_capture(capture)
    return model


def _ms(samples: float) -> float:
    return 1000 * samples / SAMPLE_RATE


def test_a_new_capture_keeps_the_place(model, capture) -> None:
    model.jump_to_null(1)
    model.scroll(0.7)
    start = model.start_ms
    model.set_capture(capture)
    assert model.start_ms == start
    model.scroll_to(1e6)
    shorter = SignalCapture(
        capture.samples[: len(capture.samples) // 2],
        SAMPLE_RATE,
        MODE_I,
        capture.frame_starts[:1],
        capture.frequency_offsets_hz[:1],
    )
    model.set_capture(shorter)  # beyond its end: at its end
    assert model.start_ms == pytest.approx(model.duration_ms - model.window_ms)


def test_it_opens_on_the_first_null_symbol_and_jumps_from_frame_to_frame(model, capture) -> None:
    nulls = [_ms(s) for s in capture.null_starts]
    lead = NULL_LEAD * model.window_ms
    assert model.start_ms == pytest.approx(nulls[0] - lead)
    model.jump_to_null(1)
    assert model.start_ms == pytest.approx(nulls[1] - lead)
    model.jump_to_null(1)
    model.jump_to_null(1)  # there is no fourth
    assert model.start_ms == pytest.approx(nulls[2] - lead)
    model.jump_to_null(-1)
    assert model.start_ms == pytest.approx(nulls[1] - lead)


def test_the_window_stays_within_the_capture(model) -> None:
    model.scroll(-1e6)
    assert model.start_ms == 0.0
    model.scroll(1e6)
    assert model.start_ms == pytest.approx(model.duration_ms - model.window_ms)
    model.scroll_to(100.0)
    model.set_window_ms(32.0)
    assert model.start_ms + 16.0 == pytest.approx(100.0 + 2.0)  # around the same middle
    model.set_span_khz(64)
    model.set_centre_khz(5000)
    assert model.centre_khz == 1024 - 32
    model.set_span_khz(2048)
    assert model.centre_khz == 0


def test_a_moment_is_placed_in_frame_and_symbol(model, capture) -> None:
    prs = _ms(capture.frame_starts[1])
    assert model.position(prs - 0.5).symbol == 0
    assert model.position(prs - 0.5).frame == 1
    assert model.position(prs + 0.1).symbol == 1
    assert model.position(prs + _ms(5 * SYMBOL) + 0.1).symbol == 6
    assert model.position(0.0) is None  # before the first null symbol


def test_the_sliding_fft_shows_the_null_symbol_as_a_trench(model, capture) -> None:
    field = model.heightfield()
    rows, columns = field.levels_db.shape
    assert rows <= MAX_ROWS and columns == model.fft_size  # all of the band
    assert field.window_ms == (model.start_ms, model.start_ms + model.window_ms)
    inner = np.abs(field.frequencies_khz) < 700
    middles = (field.row_starts_ms + field.row_ends_ms) / 2
    null = _ms(capture.null_starts[0])
    in_null = (middles > null + 0.3) & (middles < null + _ms(NULL) - 0.3)
    after = middles > null + _ms(NULL) + 0.3
    # The plateau at 0 dB, the trench far below it.
    assert np.median(field.levels_db[after][:, inner]) == pytest.approx(0, abs=2)
    assert np.median(field.levels_db[in_null][:, inner]) < -25
    assert field.symbols is None and not field.discrete_rows


def _trench(model: Spectrogram, middle_ms: float) -> tuple[float, float]:
    """Width (ms) and middle of the rows more than 10 dB below the plateau, the window
    centred on `middle_ms`."""
    model.scroll_to(middle_ms - model.window_ms / 2)
    field = model.heightfield()
    inner = np.abs(field.frequencies_khz) < 700
    deep = np.median(field.levels_db[:, inner], axis=1) < -10
    middles = (field.row_starts_ms + field.row_ends_ms) / 2
    return deep.sum() * (middles[1] - middles[0]), float(middles[deep].mean())


@pytest.mark.parametrize("size", [256, 2048])
@pytest.mark.parametrize("window_ms", [1.0, 4.0])
def test_the_rows_fill_the_window_and_the_trench_sits_on_the_null_symbol(
    model, capture, size, window_ms
) -> None:
    """Each FFT is centred on its row: the rows cover the window from end to end however
    short it is, and the trench lies in the middle of the null symbol."""
    model.fft_size = size
    model.set_window_ms(window_ms)
    middle = _ms(capture.null_starts[1] + NULL / 2)
    _width, trench = _trench(model, middle)
    field = model.heightfield()
    assert field.row_starts_ms[0] == pytest.approx(field.window_ms[0])
    assert field.row_ends_ms[-1] == pytest.approx(field.window_ms[1])
    assert np.all(np.abs(time_x(field, field.row_starts_ms)) <= HALF_WIDTH)
    assert trench == pytest.approx(middle, abs=0.05)


def test_the_longer_the_fft_the_narrower_the_trench(model, capture) -> None:
    """A row is down in the trench only while its FFT lies (nearly) all in the null
    symbol, 2656 samples: an FFT of 256 with a Hann window fits for most of it, one of
    2048 without a window (as the demodulator) for 608 samples, the room the null symbol
    leaves around the window of the TII."""
    model.set_window_ms(4.0)
    middle = _ms(capture.null_starts[1] + NULL / 2)
    widths = {}
    for size in (256, 2048):
        model.fft_size = size
        widths[size], _middle = _trench(model, middle)
    assert widths[256] > 1.1
    assert 0.3 < widths[2048] < 0.6


def test_an_fft_as_long_as_a_symbol_is_smooth_only_within_one(model, capture) -> None:
    """A window within the guard interval and the useful part of one symbol gives each
    carrier its bin and the level of the channel; one across two symbols is grainy."""
    model.fft_size = 2048
    model.set_window_ms(8.0)
    model.set_span_khz(1024)  # at most `MAX_COLUMNS`: one bin per column
    model.scroll_to(_ms(capture.frame_starts[0] + 3 * SYMBOL) - 0.5)
    field = model.heightfield()
    carriers = (np.abs(field.frequencies_khz) <= 700) & (field.frequencies_khz != 0)
    spread = field.levels_db[:, carriers].std(axis=1)
    assert spread.min() < 0.5  # within one symbol: flat to a fraction of a dB
    assert spread.max() > 3.0  # across two: grainy
    # Each column a carrier, 1 kHz apart.
    assert field.column_khz == 1.0


def test_one_row_per_symbol_one_column_per_carrier(model, capture) -> None:
    model.rows = Rows.SYMBOLS
    model.set_window_ms(8.0)
    model.set_span_khz(64)
    field = model.heightfield()
    # The end of the frame before the capture (its FFT lies in the margin), then the
    # first null symbol.
    assert list(field.symbols[:4]) == [76, 0, 1, 2]
    assert field.discrete_rows and field.discrete_columns
    assert np.allclose(field.frequencies_khz, np.arange(-32, 32))
    null_row = list(field.symbols).index(0)
    data_rows = field.symbols >= 2
    centre = list(field.frequencies_khz).index(0)
    others = field.frequencies_khz != 0
    # Every carrier of a data symbol at the level of the plateau, the centre one never
    # sent, nothing in the null symbol.
    assert np.all(np.abs(field.levels_db[data_rows][:, others]) < 3)
    assert np.all(field.levels_db[data_rows, centre] < -20)
    assert np.all(field.levels_db[null_row] < -20)
    # The rows follow the symbols in time.
    assert np.all(np.diff(field.row_starts_ms) > 0)


def test_the_whole_band_is_merged_into_fewer_columns_keeping_peaks(model) -> None:
    model.fft_size = 2048
    field = model.heightfield()
    assert field.levels_db.shape[1] == MAX_COLUMNS
    assert field.column_khz == 2.0
    assert not field.discrete_columns


def test_the_overview_covers_the_capture(model) -> None:
    overview = model.overview(300)
    assert overview.shape == (256, 300)
    # The ensemble in the middle of the band, the noise at its edges.
    assert np.median(overview[128]) > np.median(overview[5]) + 20


# --- the landscape ---------------------------------------------------------------------------


def test_blocks_stand_apart_on_the_floor_and_the_field_stands_on_it(model) -> None:
    model.rows = Rows.SYMBOLS
    model.set_span_khz(32)
    field = model.heightfield()
    assert field.levels_db.shape[1] <= DISCRETE_COLUMNS
    grid = landscape_grid(field)
    rows, columns = field.levels_db.shape
    assert grid.shape == (4 * rows, 4 * columns, 3)
    # Down at the edges of every block, up on its top.
    assert np.all(grid[:, 0::4, 2] == 0) and np.all(grid[:, 3::4, 2] == 0)
    assert grid[1, 1, 2] == pytest.approx(level_z(field.levels_db[0, 0]))
    # From the front left corner: the time rising to the right, the frequency into
    # the depth.
    assert grid[-1, 0, 0] > grid[0, 0, 0]
    assert grid[0, -1, 1] > grid[0, 0, 1]


def test_a_symbol_stands_over_the_samples_of_its_fft(model) -> None:
    """Not over the whole symbol: between two blocks lies the guard interval, as the
    strip marks it, and each block ends where the strip's useful part does."""
    model.rows = Rows.SYMBOLS
    model.jump_to_null(1)
    model.scroll(2.5)  # past the null symbol
    field = model.heightfield()
    assert field.discrete_rows and 0 not in field.symbols
    starts, ends = field.fft_starts_ms, field.fft_ends_ms
    assert ends - starts == pytest.approx(np.full(len(starts), _ms(MODE_I.fft_length)))
    assert starts[1:] - ends[:-1] == pytest.approx(
        np.full(len(starts) - 1, _ms(MODE_I.guard_length))
    )
    useful = [(begin, end) for begin, end, part in field.parts if part is Part.USEFUL]
    for start, end in zip(starts, ends, strict=True):
        assert (start, end) in [pytest.approx(u) for u in useful]
    grid = landscape_grid(field)
    inside = (starts > field.window_ms[0]) & (ends < field.window_ms[1])
    assert grid[0::4, 0, 0][inside] == pytest.approx(time_x(field, starts)[inside], abs=1e-6)
    assert grid[3::4, 0, 0][inside] == pytest.approx(time_x(field, ends)[inside], abs=1e-6)
    # The figures stand beneath the blocks.
    times, _labels = time_ticks(field)
    middles = (starts + ends) / 2
    assert inside.any() and all(np.min(np.abs(times - m)) < 1e-9 for m in middles[inside])


def test_a_smooth_field_gets_a_skirt_and_one_strip(model) -> None:
    field = model.heightfield()
    grid = landscape_grid(field)
    rows, columns = field.levels_db.shape
    assert grid.shape == (rows + 2, columns + 2, 3)
    assert np.all(grid[0, :, 2] == 0) and np.all(grid[:, -1, 2] == 0)
    strip = triangle_strip(grid)
    assert strip.shape == ((rows + 1) * (2 * (columns + 2) + 2), 3)
    assert strip.dtype == np.float32
    assert grid[..., 2].max() <= HEIGHT


@pytest.mark.parametrize("rows", [Rows.SLIDING, Rows.SYMBOLS])
def test_the_field_knows_the_frame_structure_in_the_window(model, capture, rows) -> None:
    """Each null symbol where the capture has it; each other symbol its guard interval,
    then its useful part; one after the other without a gap, over the whole window."""
    model.rows = rows
    model.jump_to_null(1)
    field = model.heightfield()
    parts = field.parts
    assert parts[0][0] <= field.window_ms[0] and parts[-1][1] >= field.window_ms[1]
    for (_, end, _), (begin, _, _) in itertools.pairwise(parts):
        assert begin == pytest.approx(end)
    nulls = [begin for begin, _, part in parts if part is Part.NULL]
    assert nulls == pytest.approx([_ms(capture.null_starts[1])])
    kinds = [part for _, _, part in parts]
    after = kinds.index(Part.NULL) + 1
    assert kinds[after : after + 4] == [Part.GUARD, Part.USEFUL, Part.GUARD, Part.USEFUL]
    guard, useful = parts[after], parts[after + 1]
    assert guard[1] - guard[0] == pytest.approx(_ms(MODE_I.guard_length))
    assert useful[1] - useful[0] == pytest.approx(_ms(MODE_I.fft_length))


def test_the_strip_lies_in_front_of_the_floor_cut_at_the_window(model) -> None:
    model.jump_to_null(1)
    field = model.heightfield()
    colours = WaterfallColours(
        background=(0.0, 0.0, 0.0), stops=((0.0, 0.0, 0.0),) * 5, grid=(1.0, 1.0, 1.0, 1.0)
    )
    strip = part_strip(field, colours)
    low, high = field.window_ms
    visible = [p for p in field.parts if p[1] > low and p[0] < high]
    assert len(visible) < len(field.parts)  # a guard interval before the window
    assert strip.shape == (6 * len(visible), 7)
    assert np.all(np.abs(strip[:, 0]) <= HALF_WIDTH)
    assert np.all((strip[:, 1] >= -HALF_DEPTH - STRIP_DEPTH) & (strip[:, 1] <= -HALF_DEPTH))
    assert np.all(strip[:, 2] == 0)
    null = visible.index(next(p for p in visible if p[2] is Part.NULL))
    assert tuple(strip[6 * null, 3:]) == pytest.approx(colours.part(Part.NULL))
    # Scrolled before the first null symbol, the margin shows the end of the frame
    # before: the last symbol, as far as the capture holds its FFT.
    model.scroll_to(0.0)
    parts = model.heightfield().parts
    null = _ms(model.capture.null_starts[0])
    assert [p[2] for p in parts[:3]] == [Part.GUARD, Part.USEFUL, Part.NULL]
    assert parts[1][1] == pytest.approx(null) and parts[0][0] < 0.0


def test_a_row_is_computed_from_one_fft_or_as_many_as_fit_into_it(model) -> None:
    assert model.row_length_us == pytest.approx(250.0)  # 512 samples
    model.set_window_ms(128.0)  # 200 rows, 1311 samples apart
    model.fft_size = 256
    assert model.row_length_us == pytest.approx(1e6 * 5 * 256 / SAMPLE_RATE)
    model.fft_size = 2048
    assert model.row_length_us == pytest.approx(1000.0)


def test_ticks_and_labels(model) -> None:
    field = model.heightfield()
    assert list(frequency_ticks(field)) == [-1000.0, -500.0, 0.0, 500.0, 1000.0]
    times, labels = time_ticks(field)
    assert len(times) == len(labels) >= 3
    assert np.all(time_x(field, times) <= HALF_WIDTH)
    model.rows = Rows.SYMBOLS
    field = model.heightfield()
    _times, labels = time_ticks(field)
    assert "N" in labels


@pytest.mark.parametrize("range_db", LEVEL_RANGES_DB)
def test_the_height_spans_the_range_chosen(model, range_db) -> None:
    """From the floor, `range_db` below the top, to the top; the back wall marks three to
    five levels in between, the plateau among them."""
    assert level_z(np.array(TOP_DB - range_db), range_db) == 0.0
    assert level_z(np.array(TOP_DB), range_db) == pytest.approx(HEIGHT)
    assert level_z(np.array(-100.0), range_db) == 0.0
    ticks = level_ticks(range_db)
    assert 3 <= len(ticks) <= 5 and 0.0 in ticks
    assert np.all((ticks > TOP_DB - range_db) & (ticks <= TOP_DB))
    field = model.heightfield()
    grid = landscape_grid(field, range_db)  # smooth: within its skirt, a point per level
    assert np.allclose(grid[1:-1, 1:-1, 2], level_z(field.levels_db, range_db))
    lines = grid_lines(field, (1.0, 1.0, 1.0, 1.0), range_db)
    wall = lines[(lines[:, 2] > 0) & (lines[:, 0] == HALF_WIDTH) & (lines[:, 2] < HEIGHT)]
    assert len(wall) == np.count_nonzero((ticks > TOP_DB - range_db) & (ticks < TOP_DB))


def test_a_window_shorter_than_a_symbol_still_names_its_symbols(model, capture) -> None:
    """From the first null symbol on (before it lies only the margin, without symbols)."""
    model.rows = Rows.SYMBOLS
    model.set_window_ms(1.0)
    for start in np.arange(_ms(capture.null_starts[0]), model.duration_ms - 1.0, 0.05):
        model.scroll_to(start)
        field = model.heightfield()
        times, labels = time_ticks(field)
        assert labels, start
        assert np.all((times >= field.window_ms[0]) & (times <= field.window_ms[1]))


def test_the_camera_turns_and_zooms_within_bounds() -> None:
    camera = WaterfallCamera()
    camera.rotate(0, 200)
    assert camera.elevation_deg == 89.0
    camera.zoom(100)
    assert camera.distance == 9.0
    camera.rotate(400, -500)
    assert -180 <= camera.azimuth_deg < 180 and camera.elevation_deg == 4.0
    # The viewer stands in front (lower y) at azimuth 0, with the frequency to its right.
    front = WaterfallCamera(azimuth_deg=0.0)
    assert front.position()[1] < 0
    assert front.axes()[0][0] > 0.99


# --- OpenGL ---------------------------------------------------------------------------------

_RENDER_SCRIPT = r"""
import json
import numpy as np
from PyQt6.QtGui import QGuiApplication
app = QGuiApplication([])
from ui.offscreen_gl import project
from ui.spectrogram import Heightfield
from ui.waterfall_gl import (
    HEIGHT, WaterfallCamera, WaterfallColours, WaterfallRenderer, WaterfallScene, level_z,
)
levels = np.full((40, 60), -60.0, dtype=np.float32)
levels[10:30, 20:40] = 0.0  # a block at the level of the plateau in the middle
field = Heightfield(
    levels_db=levels,
    frequencies_khz=np.arange(-30, 30) + 0.5,
    column_khz=1.0,
    row_starts_ms=np.arange(40) * 0.1,
    row_ends_ms=np.arange(1, 41) * 0.1,
    discrete_columns=False,
    discrete_rows=False,
    window_ms=(0.0, 4.0),
    span_khz=(-30.0, 30.0),
)
colours = WaterfallColours(
    background=(0.0, 0.0, 0.5),
    stops=((0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.4, 0.4, 0.4), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
    grid=(1.0, 1.0, 1.0, 1.0),
)
renderer = WaterfallRenderer()
camera = WaterfallCamera(azimuth_deg=0.0, elevation_deg=60.0)
width, height = 640, 400
image = renderer.render(camera, width, height, WaterfallScene(field, colours))
result = {"error": renderer.error}
if image is not None:
    matrix = camera.matrix(width / height)
    top = float(level_z(np.array(0.0)))
    places = {
        "block": (0.0, 0.0, top),
        "floor": (-0.85, 0.6, 0.0),
    }
    for name, point in places.items():
        x, y = project(matrix, np.array([point]), width, height)[0]
        result[name] = image.pixelColor(round(x), round(y)).getRgb()[:3]
    result["corner"] = image.pixelColor(2, 2).getRgb()[:3]
renderer.release()
print(json.dumps(result))
"""


@pytest.mark.skipif(sys.platform != "win32", reason="needs a desktop to draw on")
def test_opengl_draws_the_block_in_the_colour_of_its_level() -> None:
    source = Path(__file__).parent.parent / "src"
    environment = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    environment["PYTHONPATH"] = str(source)
    run = subprocess.run(
        [sys.executable, "-c", _RENDER_SCRIPT],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout.strip().splitlines()[-1])
    if result["error"]:
        pytest.skip(f"no OpenGL on this machine: {result['error']}")
    red, green, blue = result["block"]
    assert green > 100 and green > 2 * red and green > 2 * blue  # the top, lit
    assert max(result["floor"]) < 100  # the floor of the field, dark
    assert result["corner"][2] > 100 > result["corner"][1]  # the background
