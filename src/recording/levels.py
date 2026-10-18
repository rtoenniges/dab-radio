"""The level of a stretch for the cutter (`recording.cutter`): RMS and peak in dBFS of
every eighth of a unit (`STEPS_PER_UNIT`, 5 ms with DAB+ at 48 kHz and SBR), decoded
with the decoders of the player (FAAD2, libmpg123).

The level is not in the file: only decoding the audio tells it. A Raspberry Pi 5 reads
the level some 34 times faster than real time, a quarter of an hour in under half a
minute; the level of the decoded audio costs a tenth of the decoding, whether of whole
units or of their eighths (one numpy call either way). `LevelReader` lives in a thread
of its own and works in slices of `SLICE_S`, so that a stretch can be given up for
another at once; what it has read so far goes out every `REPORT_S`, and the surface
draws the level as it comes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from audio.aac import AacDecoder, AacDecoderError
from audio.mp2 import Mp2Decoder, Mp2DecoderError
from dsp.mp2_frame import Mp2Format
from recording.playback import RecordingAudio

logger = logging.getLogger(__name__)

# The level of a unit that gave no audio, and of digital silence.
FLOOR_DB = -120.0
# How long a slice decodes before the thread's event loop has its turn.
SLICE_S = 0.03
# How often at most the levels read so far are handed on.
REPORT_S = 0.2
# How many levels a unit has: a unit of 40 ms (960 samples of the core, doubled by SBR)
# in steps of 5 ms. With one level a unit, the level of the cutter comes in steps two or
# three pixels wide where a view shows 20 s; the samples of a unit (1920, 960 or 1152)
# part evenly in eight.
STEPS_PER_UNIT = 8
_FULL_SCALE = 32768.0
_FLOOR = 10 ** (FLOOR_DB / 20)


def unit_levels(
    pcm: np.ndarray, channels: int = 1, steps: int = STEPS_PER_UNIT
) -> tuple[np.ndarray, np.ndarray]:
    """RMS and peak in dBFS of `steps` equal parts of a unit's samples (int16,
    interleaved), of the sum of its channels: the silence between two titles can carry
    a signal of some 20 Hz at −26 dBFS with left the negative of right, and the sum
    there is digital silence, as a pause should be."""
    if not len(pcm):
        return np.full(steps, FLOOR_DB, dtype=np.float32), np.full(
            steps, FLOOR_DB, dtype=np.float32
        )
    samples = pcm.astype(np.float32) / _FULL_SCALE
    if channels > 1 and len(samples) % channels == 0:
        samples = samples.reshape(-1, channels).mean(axis=1)
    usable = len(samples) // steps * steps
    if usable:
        parts = samples[:usable].reshape(steps, -1)
    else:  # fewer samples than steps: all of them for each
        parts = np.broadcast_to(samples, (steps, len(samples)))
    rms = np.sqrt(np.mean(parts * parts, axis=1))
    peak = np.max(np.abs(parts), axis=1)
    return (
        (20 * np.log10(np.maximum(rms, _FLOOR))).astype(np.float32),
        (20 * np.log10(np.maximum(peak, _FLOOR))).astype(np.float32),
    )


class LevelReader(QObject):
    """Reads the level of a stretch (`start`), slice by slice."""

    # The levels of steps `first` on of the stretch `id` (`STEPS_PER_UNIT` a unit; float
    # arrays, dBFS; NaN where the decoder refused a unit), and the end of a stretch.
    read = pyqtSignal(str, int, object, object)
    finished = pyqtSignal(str)

    def __init__(
        self, parent: QObject | None = None, clock: Callable[[], float] = time.monotonic
    ) -> None:
        super().__init__(parent)
        self._clock = clock
        self._audio: RecordingAudio | None = None
        self._decoder: AacDecoder | Mp2Decoder | None = None
        # Every stretch begun counts one up: a slice of one given up meanwhile stops.
        self._generation = 0
        self._index = 0  # the next unit to decode
        # The decoder gives the audio of a unit with the next call (FAAD2 gives nothing
        # for the first): the units it gave nothing for at first, then each output is
        # the audio of the unit that many before.
        self._delay = 0
        self._started = False
        self._first = 0  # the first step of what is not handed on yet
        self._rms: list[np.ndarray] = []  # the levels of a unit each
        self._peak: list[np.ndarray] = []
        self._reported_at = 0.0
        self._refused = 0

    @pyqtSlot(object)
    def start(self, audio: RecordingAudio) -> None:
        """Reads the level of `audio` from its beginning, giving up what it read."""
        self.cancel()
        try:
            if isinstance(audio.audio_format, Mp2Format):
                self._decoder = Mp2Decoder(audio.audio_format)
            else:
                self._decoder = AacDecoder(audio.audio_format)
        except (AacDecoderError, Mp2DecoderError) as error:
            logger.warning("No level for %s: %s", audio.recording_id, error)
            self.finished.emit(audio.recording_id)
            return
        self._audio = audio
        self._index = 0
        self._delay = 0
        self._started = False
        self._first = 0
        self._rms = []
        self._peak = []
        self._reported_at = self._clock()
        self._refused = 0
        self._schedule()

    @pyqtSlot()
    def cancel(self) -> None:
        """Gives up the stretch being read."""
        self._generation += 1
        self._audio = None
        if self._decoder is not None:
            self._decoder.close()
            self._decoder = None

    def _schedule(self) -> None:
        generation = self._generation
        QTimer.singleShot(0, lambda: self._slice(generation))

    def _slice(self, generation: int) -> None:
        if generation != self._generation or self._audio is None or self._decoder is None:
            return
        audio, decoder = self._audio, self._decoder
        began = self._clock()
        while self._index < len(audio.units) and self._clock() - began < SLICE_S:
            unit = audio.units[self._index]
            self._index += 1
            try:
                pcm = decoder.decode(unit)
            except AacDecoderError, Mp2DecoderError:
                self._refused += 1
                pcm = None
            if pcm is not None and not len(pcm) and not self._started:
                self._delay += 1
                continue
            self._started = True
            if pcm is None or not len(pcm):
                # Refused, or no audio later on: unknown, not silence (no pause).
                unknown = np.full(STEPS_PER_UNIT, np.nan, dtype=np.float32)
                self._rms.append(unknown)
                self._peak.append(unknown)
            else:
                rms, peak = unit_levels(pcm, decoder.channels)
                self._rms.append(rms)
                self._peak.append(peak)
        done = self._index >= len(audio.units)
        if done or self._clock() - self._reported_at >= REPORT_S:
            self._report(audio.recording_id)
        if not done:
            self._schedule()
            return
        if self._refused:
            logger.info("Level of %s: %d units refused", audio.recording_id, self._refused)
        self.cancel()
        self.finished.emit(audio.recording_id)

    def _report(self, stretch_id: str) -> None:
        self._reported_at = self._clock()
        if not self._rms:
            return
        rms = np.concatenate(self._rms)
        peak = np.concatenate(self._peak)
        self.read.emit(stretch_id, self._first, rms, peak)
        self._first += len(rms)
        self._rms = []
        self._peak = []
