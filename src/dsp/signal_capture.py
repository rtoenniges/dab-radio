"""A stretch of the received signal kept whole for the detail view of the signal
page: the samples of several consecutive transmission frames as the demodulator saw
them (after the DC and IQ correction), with the frame starts it aligned to and the
frequency offset it corrected. The view computes its spectra from these itself, as
often as it is scrolled; the signal processing only collects them, once, when asked
(`SignalCaptureRecorder`). Ten frames are 0.96 s of signal and 15.7 MB as complex64.

For the null symbol on the radio page, the signal processing cuts a short stretch
around the null symbol of every frame instead, as long as it is asked to
(`NullSnapshotter`): 6 ms, 96 kB.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from dab.constants import TransmissionMode

# Frames kept by default: enough to scroll from frame to frame and to see the TII,
# which some transmitters send in every other null symbol only (5C, 2026-10-12).
CAPTURE_FRAMES = 10

# Samples kept before the first null symbol and after the end of the last frame (1 ms):
# the view can show the first null symbol with what precedes it.
MARGIN_SAMPLES = 2048

# Frames' worth of blocks kept while no frame has been found (see `SignalCaptureRecorder`).
_KEPT_WITHOUT_FRAMES = 4

# The stretch around a null symbol (`NullSnapshotter`): 2 ms before it, the end of the
# frame before, and 4 ms from its start, the null symbol and the first symbols after it.
SNAPSHOT_BEFORE = 4096
SNAPSHOT_AFTER = 8192


@dataclass(frozen=True)
class SignalCapture:
    """Consecutive frames of the signal, with where the demodulator found them."""

    samples: np.ndarray  # complex64
    sample_rate: float
    mode: TransmissionMode
    # Position of each frame's phase reference symbol in `samples`, as aligned by the
    # demodulator (after the fine alignment); int64, rising.
    frame_starts: np.ndarray
    # The frequency offset the demodulator rotated out of each frame (Hz).
    frequency_offsets_hz: np.ndarray
    # Where `samples` begin in the stream received since the last tuning.
    start_sample: int = 0

    @property
    def null_starts(self) -> np.ndarray:
        """Position of each frame's null symbol in `samples`."""
        return self.frame_starts - self.mode.null_symbol_length

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def first_frame_number(self) -> int:
        """The number of the first frame in the stream received since the last tuning,
        counted by its position (frame_length samples per frame): two frames one after
        the other differ by one, whichever of them the demodulator reported."""
        return round((self.start_sample + int(self.frame_starts[0])) / self.mode.frame_length)

    @property
    def frequency_offset_hz(self) -> float:
        """One offset for the whole stretch: it drifts by a few hertz at most."""
        return float(np.median(self.frequency_offsets_hz))


class SignalCaptureRecorder:
    """Collects the incoming blocks and the frames the demodulator finds in them until
    `frames` frames lie complete in the blocks, from the sample `first_sample` on.

    `add_block` takes the blocks in the order they arrive, with their absolute position
    in the stream; `add_frame` the frames as they are demodulated. Frames are found a
    block or so after their samples came, and the first ones may begin before
    `first_sample` (they are skipped). Until a frame is found, only the newest
    `_KEPT_WITHOUT_FRAMES` frames' worth of blocks are kept: without reception the
    recorder must not grow. The demodulator holds two to three frames before it reports
    the oldest (`dsp.pipeline.DspWorker._emit_power_envelope`); with fewer kept, the
    frame it reports would already have been dropped.
    """

    def __init__(
        self, frames: int, first_sample: int, mode: TransmissionMode, sample_rate: float
    ) -> None:
        if frames < 1:
            raise ValueError("At least one frame must be captured.")
        self.frames = frames
        self._first = first_sample
        self._mode = mode
        self._sample_rate = sample_rate
        self._blocks: list[tuple[int, np.ndarray]] = []
        self._pending = 0  # samples in `_blocks`
        self._starts: list[int] = []
        self._offsets: list[float] = []

    @property
    def frames_found(self) -> int:
        return len(self._starts)

    def add_block(self, start: int, samples: np.ndarray) -> None:
        end = start + len(samples)
        if end <= self._first:
            return
        if start < self._first:
            samples, start = samples[self._first - start :], self._first
        self._blocks.append((start, samples))
        self._pending += len(samples)
        if not self._starts:
            while self._pending - len(self._blocks[0][1]) >= _KEPT_WITHOUT_FRAMES * (
                self._mode.frame_length
            ):
                _, dropped = self._blocks.pop(0)
                self._pending -= len(dropped)
                self._first = self._blocks[0][0]

    def add_frame(self, prs_start: int, frequency_offset_hz: float) -> None:
        """A frame demodulated: its phase reference symbol at the absolute position
        `prs_start`, with the frequency offset rotated out of it."""
        if len(self._starts) >= self.frames:
            return
        if prs_start - self._mode.null_symbol_length - MARGIN_SAMPLES < self._first:
            return  # began before the recording
        self._starts.append(prs_start)
        self._offsets.append(frequency_offset_hz)

    def capture(self) -> SignalCapture | None:
        """The frames with their margins, once all are found and their samples are in;
        None until then."""
        if len(self._starts) < self.frames or not self._blocks:
            return None
        mode = self._mode
        begin = self._starts[0] - mode.null_symbol_length - MARGIN_SAMPLES
        end = self._starts[-1] + mode.symbols_per_frame * mode.symbol_length + MARGIN_SAMPLES
        received = self._blocks[-1][0] + len(self._blocks[-1][1])
        if received < end:
            return None
        offset = self._blocks[0][0]
        samples = np.concatenate([block for _, block in self._blocks])
        return SignalCapture(
            samples=samples[begin - offset : end - offset].astype(np.complex64),
            sample_rate=self._sample_rate,
            mode=mode,
            frame_starts=np.array(self._starts, dtype=np.int64) - begin,
            frequency_offsets_hz=np.array(self._offsets, dtype=np.float64),
            start_sample=begin,
        )


class NullSnapshotter:
    """Cuts the stretch from `before` samples ahead of a frame's null symbol to `after`
    samples from its start, for every frame demodulated (`add_frame`), out of the
    blocks as they came (`add_block`).

    It keeps the newest `_KEPT_WITHOUT_FRAMES` frames' worth of blocks, by reference:
    a frame is reported two to three frames after its samples came (see
    `SignalCaptureRecorder`), and the stretch reaches into the frame before.
    """

    def __init__(
        self,
        mode: TransmissionMode,
        sample_rate: float,
        before: int = SNAPSHOT_BEFORE,
        after: int = SNAPSHOT_AFTER,
    ) -> None:
        self._mode = mode
        self._sample_rate = sample_rate
        self._before = before
        self._after = after
        self._blocks: deque[tuple[int, np.ndarray]] = deque()
        self._kept = 0  # samples in `_blocks`

    def add_block(self, start: int, samples: np.ndarray) -> None:
        self._blocks.append((start, samples))
        self._kept += len(samples)
        limit = _KEPT_WITHOUT_FRAMES * self._mode.frame_length
        while self._kept - len(self._blocks[0][1]) >= limit:
            _, dropped = self._blocks.popleft()
            self._kept -= len(dropped)

    def add_frame(self, prs_start: int, frequency_offset_hz: float) -> SignalCapture | None:
        """The stretch around the null symbol before the phase reference symbol at the
        absolute position `prs_start`, as a capture of one frame; None where the
        blocks kept do not hold all of it."""
        begin = prs_start - self._mode.null_symbol_length - self._before
        end = begin + self._before + self._after
        if not self._blocks or self._blocks[0][0] > begin:
            return None
        last_start, last = self._blocks[-1]
        if last_start + len(last) < end:
            return None
        pieces = []
        for start, block in self._blocks:
            low, high = max(begin, start), min(end, start + len(block))
            if low < high:
                pieces.append(block[low - start : high - start])
        return SignalCapture(
            samples=np.concatenate(pieces).astype(np.complex64),
            sample_rate=self._sample_rate,
            mode=self._mode,
            frame_starts=np.array([prs_start - begin], dtype=np.int64),
            frequency_offsets_hz=np.array([frequency_offset_hz], dtype=np.float64),
            start_sample=begin,
        )
