"""Regression tests for the frame tracking in `dsp.pipeline.DspWorker`.

Background: the FIC is completely insensitive to skipped frames (every frame is
decoded independently), the MSC is not: the time interleaver spreads every logical
frame over 16 consecutive CIFs. A single dropped frame therefore permanently breaks
audio decoding without any other display giving it away. Exactly these gaps are what
the tests here check.
"""

from __future__ import annotations

import numpy as np

from dab.constants import MODE_I
from dsp import ofdm
from dsp.ofdm import DemodulatedFrame
from dsp.pipeline import (
    MAX_UNRELIABLE_FRAMES,
    NULL_TRACKING_WINDOW,
    SYMBOL_WINDOW_EXCERPT_SYMBOLS,
    DspWorker,
    SymbolWindows,
    _DataSubchannelChain,
)
from sdr.worker import DEFAULT_READ_SIZE

_MARKER_OFFSET = 100  # distance of the frame number marker behind the null symbol
_MARKER_HALF_WIDTH = 20  # marker as a short block so that it survives sample jitter


def _synthetic_stream(
    num_frames: int, null_power: list[float], offset: int = 5000, seed: int = 1
) -> np.ndarray:
    """Builds a sample stream of `num_frames` frames.

    Every frame consists of a quiet null symbol and a loud "useful part". `null_power`
    gives the amplitude of the null symbol per frame; in real reception it varies
    considerably (by some 16 dB, among other things because of the TII signal).

    The signal is zero-mean complex noise, not a constant level:
    `DspWorker.process_samples` can correct DC offset and IQ imbalance before the frame
    search (`dsp.iq_correction`), and only a zero-mean signal behaves like a real OFDM
    signal there.

    A short block of constant samples in the useful part carries the frame number, so
    that a test can attribute the delivered frames unambiguously. It sits
    `_MARKER_OFFSET` samples behind the null symbol and has an amplitude >= 1, so it
    can only steer the minimum search away from itself, never towards it.
    """
    rng = np.random.default_rng(seed)
    total = offset + num_frames * MODE_I.frame_length
    stream = (rng.normal(size=total) + 1j * rng.normal(size=total)) / np.sqrt(2)
    for i in range(num_frames):
        start = offset + i * MODE_I.frame_length
        stream[start : start + MODE_I.null_symbol_length] *= null_power[i]
        marker = start + MODE_I.null_symbol_length + _MARKER_OFFSET
        stream[marker - _MARKER_HALF_WIDTH : marker + _MARKER_HALF_WIDTH + 1] = complex(i + 1, 0)
    return stream


def _drain(worker: DspWorker, stream: np.ndarray) -> list[int]:
    """Feeds `stream` in block by block and returns the numbers of the frames that
    `_process_frame` actually got to see."""
    seen: list[int] = []

    def record(samples: np.ndarray, null_index: int) -> bool:
        marker = samples[null_index + MODE_I.null_symbol_length + _MARKER_OFFSET].real
        seen.append(round(float(marker)) - 1)
        return True

    worker._process_frame = record  # type: ignore[method-assign]
    for off in range(0, len(stream) - DEFAULT_READ_SIZE + 1, DEFAULT_READ_SIZE):
        worker.process_samples(stream[off : off + DEFAULT_READ_SIZE])
    return seen


def _fake_frame(
    correction: int = 0, frequency_offset_hz: float = 0.0, reliable: bool = True, spectra=None
) -> DemodulatedFrame:
    """A demodulation result with three data symbols, standing in for the real one."""
    return DemodulatedFrame(
        symbols=[np.ones(MODE_I.num_carriers, dtype=complex) for _ in range(3)],
        frequency_offset_hz=frequency_offset_hz,
        sample_correction=correction,
        reliable=reliable,
        channel_estimate=np.ones(MODE_I.num_carriers, dtype=complex),
        spectra=np.ones((4, MODE_I.fft_length), dtype=complex) if spectra is None else spectra,
    )


# Start offset of the test stream. It determines how many samples remain after every
# emptying of the frame buffer and is chosen here so that the remainder exceeds one
# frame length: exactly the condition under which an unlimited minimum search has TWO
# null symbols in its window and returns the quieter one. Without this offset the
# remainder stays below one frame length and the bug does not occur at all.
_STREAM_OFFSET_WITH_TWO_NULLS_IN_RANGE = 196_000


def test_drain_yields_every_frame_exactly_once_despite_varying_null_power():
    """No frame may be skipped or delivered twice.

    Once the frame buffer grows beyond two frame lengths, the search span contains TWO
    null symbols. A plain minimum search returns the quieter of them, here deliberately
    alternating the later one, so every second frame would be lost (this set-up would
    deliver only frames 1, 3, 5, 7, 9).
    """
    num_frames = 12
    # Every second frame has a markedly quieter null symbol than its predecessor; a
    # search across two frames jumps ahead exactly then.
    null_power = [0.05 if i % 2 else 0.2 for i in range(num_frames)]
    worker = DspWorker(2_048_000.0, MODE_I)

    stream = _synthetic_stream(
        num_frames, null_power, offset=_STREAM_OFFSET_WITH_TWO_NULLS_IN_RANGE
    )
    seen = _drain(worker, stream)

    assert len(seen) >= num_frames - 3, "practically all frames must be delivered"
    assert seen == list(range(seen[0], seen[0] + len(seen))), (
        f"frames must be delivered without gaps and in ascending order, but were {seen}"
    )


def test_drain_stays_locked_when_null_power_of_later_frame_is_lowest():
    """Even a single, particularly quiet later frame must not make the tracking jump
    there."""
    num_frames = 10
    null_power = [0.2] * num_frames
    null_power[6] = 0.001  # an extremely quiet null symbol far back in the buffer
    worker = DspWorker(2_048_000.0, MODE_I)

    seen = _drain(worker, _synthetic_stream(num_frames, null_power))

    assert seen == list(range(seen[0], seen[0] + len(seen))), (
        f"frames must remain gapless, but were {seen}"
    )


def test_tracking_window_cannot_contain_two_null_symbols():
    """Guards the central invariant: the tracking window is much narrower than the
    distance between two null symbols (one frame length)."""
    assert 2 * NULL_TRACKING_WINDOW < MODE_I.frame_length


def test_unusable_frame_invalidates_msc_chain_and_reacquires():
    """A frame that cannot be demodulated completely is a gap in the CIF sequence:
    the time-dependent stages must be reset and the frame search restarted instead of
    silently carrying on.
    """
    worker = DspWorker(2_048_000.0, MODE_I)
    worker._superframe_buffer = [np.zeros(8, dtype=np.uint8)] * 3
    worker._superframe_locked = True
    worker._superframe_misses = 2
    worker._audio_subchannel._fill = 9
    data_subchannel = worker._data_subchannels[30] = _DataSubchannelChain()
    data_subchannel.subchannel._fill = 9
    worker._process_frame = lambda samples, null_index: False  # type: ignore[method-assign]

    stream = _synthetic_stream(4, [0.2] * 4)
    for off in range(0, len(stream) - DEFAULT_READ_SIZE + 1, DEFAULT_READ_SIZE):
        worker.process_samples(stream[off : off + DEFAULT_READ_SIZE])

    assert worker._expected_null is None, "frame tracking must re-acquire"
    assert worker._superframe_buffer == []
    assert worker._superframe_locked is False
    assert worker._superframe_misses == 0
    assert worker._audio_subchannel._fill == 0
    assert data_subchannel.subchannel._fill == 0


def test_persistently_unreliable_frames_trigger_reacquisition():
    """Frame tracking must free itself from a wrong lock.

    Unlike a search over the whole buffer, a grid once locked does not correct itself.
    If acquisition locks onto a wrong position (as when reception restarts while the
    gain control is still settling), it would otherwise stay there forever and deliver
    nothing but unusable frames.
    """
    worker = DspWorker(2_048_000.0, MODE_I)
    worker._expected_null = 0  # locked
    reacquisitions = []

    def unreliable(samples: np.ndarray, null_index: int) -> bool:
        # Behaves like a frame that demodulates but is unreliable.
        reacquisitions.append(worker._expected_null is None)
        worker._unreliable_frames += 1
        return True

    worker._process_frame = unreliable  # type: ignore[method-assign]
    stream = _synthetic_stream(20, [0.2] * 20)
    for off in range(0, len(stream) - DEFAULT_READ_SIZE + 1, DEFAULT_READ_SIZE):
        worker.process_samples(stream[off : off + DEFAULT_READ_SIZE])

    assert any(reacquisitions), (
        "after persistently unreliable frames the frame position must be searched afresh"
    )
    assert sum(reacquisitions) <= len(reacquisitions) // MAX_UNRELIABLE_FRAMES + 1, (
        "it must not re-acquire after every frame"
    )


def test_reliable_frames_keep_the_lock():
    """A reliable frame resets the counter, so isolated dropouts (normal with intact
    reception) do not discard the lock."""
    worker = DspWorker(2_048_000.0, MODE_I)
    worker._expected_null = 0
    calls = {"n": 0}

    def mostly_reliable(samples: np.ndarray, null_index: int) -> bool:
        calls["n"] += 1
        # Every third frame unreliable, never MAX_UNRELIABLE_FRAMES in a row.
        worker._unreliable_frames = worker._unreliable_frames + 1 if calls["n"] % 3 == 0 else 0
        return True

    worker._process_frame = mostly_reliable  # type: ignore[method-assign]
    stream = _synthetic_stream(20, [0.2] * 20)
    for off in range(0, len(stream) - DEFAULT_READ_SIZE + 1, DEFAULT_READ_SIZE):
        worker.process_samples(stream[off : off + DEFAULT_READ_SIZE])

    assert calls["n"] > 10, "enough frames must have been processed"
    assert worker._expected_null is not None, "the lock must not be discarded"


def test_constellation_is_shown_for_unreliable_frames_too(monkeypatch):
    """The constellation diagram is meant to show the signal actually received, also
    (and especially) when it is bad. The same goes for the channel impulse response,
    the SFN share, the transmitter identification and the FFT-per-symbol display.
    Freezing them for frames rated unreliable would hide a briefly "collapsed" point
    cloud, exactly the information the diagram exists for.
    """
    spectra = np.ones((4, MODE_I.fft_length), dtype=complex)
    monkeypatch.setattr(
        ofdm,
        "demodulate_frame_symbols",
        lambda *args, **kwargs: _fake_frame(
            frequency_offset_hz=123.0, reliable=False, spectra=spectra
        ),
    )

    worker = DspWorker(2_048_000.0, MODE_I)
    shown: list[np.ndarray] = []
    offsets: list[float] = []
    worker.constellation_ready.connect(shown.append)
    worker.frequency_offset_ready.connect(offsets.append)
    responses: list[np.ndarray] = []
    shares: list[tuple[float, float]] = []
    worker.impulse_response_ready.connect(lambda delays, power: responses.append(power))
    worker.sfn_share_ready.connect(lambda others, outside: shares.append((others, outside)))
    tii_results: list[object] = []
    worker.tii_ready.connect(tii_results.append)
    windows: list[SymbolWindows] = []
    worker.symbol_windows_ready.connect(windows.append)

    stream = _synthetic_stream(6, [0.2] * 6)
    for off in range(0, len(stream) - DEFAULT_READ_SIZE + 1, DEFAULT_READ_SIZE):
        worker.process_samples(stream[off : off + DEFAULT_READ_SIZE])

    assert shown, "unreliable frames must appear in the constellation as well"
    assert len(shown[0]) == 3 * MODE_I.num_carriers
    assert offsets == [123.0] * len(shown)
    assert len(responses) == len(shares) == len(tii_results) == len(windows) == len(shown)
    assert windows[0].spectra is spectra


def test_symbol_windows_show_the_null_symbol_and_the_first_symbols(monkeypatch):
    """The time excerpt starts at the null symbol before the demodulated frame and
    spans `SYMBOL_WINDOW_EXCERPT_SYMBOLS` symbols, measured from the frame position
    actually used for demodulation."""
    correction = 7
    spectra = np.zeros((4, MODE_I.fft_length), dtype=complex)
    monkeypatch.setattr(
        ofdm,
        "demodulate_frame_symbols",
        lambda *args, **kwargs: _fake_frame(correction=correction, spectra=spectra),
    )
    worker = DspWorker(2_048_000.0, MODE_I)
    windows: list[SymbolWindows] = []
    worker.symbol_windows_ready.connect(windows.append)
    null_indices: list[int] = []
    decode = worker._process_frame

    def record(samples: np.ndarray, null_index: int) -> bool:
        null_indices.append(null_index)
        excerpt_end = (
            null_index
            + MODE_I.null_symbol_length
            + correction
            + (SYMBOL_WINDOW_EXCERPT_SYMBOLS * MODE_I.symbol_length)
        )
        expected.append(np.abs(samples[excerpt_end - 10 : excerpt_end]))
        return decode(samples, null_index)

    expected: list[np.ndarray] = []
    worker._process_frame = record  # type: ignore[method-assign]
    stream = _synthetic_stream(4, [0.2] * 4)
    for off in range(0, len(stream) - DEFAULT_READ_SIZE + 1, DEFAULT_READ_SIZE):
        worker.process_samples(stream[off : off + DEFAULT_READ_SIZE])

    assert windows
    for shown, tail in zip(windows, expected, strict=True):
        assert shown.excerpt_start == -MODE_I.null_symbol_length
        assert len(shown.magnitude) == (
            MODE_I.null_symbol_length + SYMBOL_WINDOW_EXCERPT_SYMBOLS * MODE_I.symbol_length
        )
        np.testing.assert_array_equal(shown.magnitude[-10:], tail)
        assert shown.null_spectrum is not None
        assert shown.null_spectrum.shape == (MODE_I.fft_length,)
        assert shown.spectra is spectra


def test_power_envelope_markers_follow_the_decoded_frame_positions():
    """The markers in the power envelope must show exactly the frame position the
    decoder locked onto.

    A separate minimum search over the read block finds a real null symbol too, but at
    most ONE per block (and with varying null symbol power the quieter of two), while
    the decoder processes two frames per block. The test therefore holds markers and
    actually decoded frame starts against each other.
    """
    num_frames = 12
    # Strongly varying null symbol power: exactly here an independent minimum search
    # and the frame tracking diverge.
    null_power = [0.05 if i % 2 else 0.2 for i in range(num_frames)]
    worker = DspWorker(2_048_000.0, MODE_I)

    decoded: list[int] = []

    def record(samples: np.ndarray, null_index: int) -> bool:
        decoded.append(worker._buffer_start + null_index)
        return True

    worker._process_frame = record  # type: ignore[method-assign]

    marked: list[int] = []
    displayed_start = 0

    def on_envelope(time_axis, power_db, frame_times):
        marked.extend(round(displayed_start + t * 2_048_000.0) for t in frame_times)

    worker.power_envelope_ready.connect(on_envelope)

    stream = _synthetic_stream(
        num_frames, null_power, offset=_STREAM_OFFSET_WITH_TWO_NULLS_IN_RANGE
    )
    for block_start in range(0, len(stream) - DEFAULT_READ_SIZE + 1, DEFAULT_READ_SIZE):
        # The start of the displayed window is the frame buffer start as it stands
        # BEFORE this block is processed (see DspWorker._emit_power_envelope).
        displayed_start = worker._buffer_start if worker._frame_buffer is not None else block_start
        worker.process_samples(stream[block_start : block_start + DEFAULT_READ_SIZE])

    assert decoded, "frames must have been decoded"
    assert marked, "frame markers must be reported"
    assert set(marked) == set(decoded), (
        "the markers must match the decoded frame starts exactly; "
        f"decoded only: {sorted(set(decoded) - set(marked))}, "
        f"marked only: {sorted(set(marked) - set(decoded))}"
    )
    assert len(marked) == len(set(marked)), "no frame start may be marked twice"


def test_power_envelope_markers_stay_inside_the_displayed_window():
    """Frame starts still lying in the previous read block (the frame buffer trails
    the display by its remainder) must not be drawn outside the displayed time
    axis."""
    worker = DspWorker(2_048_000.0, MODE_I)
    worker._process_frame = lambda samples, null_index: True  # type: ignore[method-assign]
    seen: list[tuple[float, list[float]]] = []
    worker.power_envelope_ready.connect(
        lambda time_axis, power_db, frame_times: seen.append((float(time_axis[-1]), frame_times))
    )

    stream = _synthetic_stream(10, [0.2] * 10)
    for start in range(0, len(stream) - DEFAULT_READ_SIZE + 1, DEFAULT_READ_SIZE):
        worker.process_samples(stream[start : start + DEFAULT_READ_SIZE])

    assert seen
    for last_time, frame_times in seen:
        for position in frame_times:
            assert 0.0 <= position <= last_time
