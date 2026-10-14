"""The level of a stretch for the cutter (`recording.levels`): an eighth of a unit at a
time, of the sum of the channels, in slices, each unit given the audio the decoder
hands back for it."""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import QApplication

from dsp.superframe import AudioFormat
from recording import levels as levels_module
from recording.levels import FLOOR_DB, STEPS_PER_UNIT, LevelReader, unit_levels
from recording.playback import RecordingAudio

FORMAT = AudioFormat(
    dac_rate_48k=True, sbr=True, stereo_core=True, parametric_stereo=False, mpeg_surround_config=0
)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def test_the_level_is_that_of_the_sum_of_the_channels():
    """Left the negative of right (seen between two titles of Absolut relax) sums to
    silence; the same on both channels keeps its level."""
    tone = (np.sin(np.arange(1920) / 2) * 16384).astype(np.int16)
    both = np.column_stack([tone, tone]).ravel()
    opposed = np.column_stack([tone, -tone]).ravel()
    rms, peak = unit_levels(both, 2)
    assert len(rms) == len(peak) == STEPS_PER_UNIT
    np.testing.assert_allclose(rms, 20 * np.log10(0.5 / np.sqrt(2)), atol=0.1)
    np.testing.assert_allclose(peak, 20 * np.log10(0.5), atol=0.1)
    np.testing.assert_array_equal(unit_levels(opposed, 2)[0], FLOOR_DB)
    np.testing.assert_array_equal(unit_levels(np.empty(0, dtype=np.int16), 2)[1], FLOOR_DB)


def test_a_unit_has_a_level_for_each_eighth_of_it():
    """A note that sets in halfway through the unit: silent first, loud after."""
    samples = np.zeros(1920, dtype=np.int16)
    samples[960:] = 16384
    rms, _peak = unit_levels(samples, 1)
    np.testing.assert_array_equal(rms[:4], FLOOR_DB)
    np.testing.assert_allclose(rms[4:], 20 * np.log10(0.5), atol=0.01)
    # Fewer samples than steps: all of them for each.
    np.testing.assert_allclose(
        unit_levels(np.full(3, 16384, dtype=np.int16), 1)[0], -6.02, atol=0.01
    )


class _LaggingDecoder:
    """Like FAAD2: nothing for the first unit, then each call gives the audio of the
    unit before. A unit's first byte is its amplitude; b"bad" is refused."""

    def __init__(self, audio_format: AudioFormat) -> None:
        self.channels = 2
        self._previous: bytes | None = None
        self.closed = False

    def decode(self, unit: bytes) -> np.ndarray:
        if unit == b"bad":
            raise levels_module.AacDecoderError("broken")
        previous, self._previous = self._previous, unit
        if previous is None:
            return np.empty(0, dtype=np.int16)
        return np.full(1920 * 2, previous[0] * 100, dtype=np.int16)

    def close(self) -> None:
        self.closed = True


def _read(reader: LevelReader, audio: RecordingAudio) -> tuple[np.ndarray, list]:
    """The level of each unit (the first of its steps) and what was handed on."""
    rms = np.full(len(audio.units) * STEPS_PER_UNIT, np.nan)
    reads: list[tuple[int, int]] = []
    finished: list[str] = []

    def on_read(stretch_id: str, first: int, values: np.ndarray, _peak: np.ndarray) -> None:
        reads.append((first, len(values)))
        rms[first : first + len(values)] = values

    reader.read.connect(on_read)
    reader.finished.connect(finished.append)
    reader.start(audio)
    deadline = time.monotonic() + 5
    while not finished and time.monotonic() < deadline:
        QCoreApplication.processEvents()
    assert finished == [audio.recording_id]
    steps = rms.reshape(-1, STEPS_PER_UNIT)
    assert all(
        np.array_equal(row, np.full(STEPS_PER_UNIT, row[0]), equal_nan=True) for row in steps
    ), "a unit of one level throughout"
    return steps[:, 0], reads


def test_each_unit_gets_the_audio_the_decoder_gives_for_it(monkeypatch):
    monkeypatch.setattr(levels_module, "AacDecoder", _LaggingDecoder)
    amplitudes = [10, 20, 0, 40]
    audio = RecordingAudio("cut:x", FORMAT, tuple(bytes([a]) for a in amplitudes))
    rms, _reads = _read(LevelReader(), audio)
    expected = [20 * np.log10(a * 100 / 32768) if a else FLOOR_DB for a in amplitudes[:3]]
    np.testing.assert_allclose(rms[:3], expected, atol=0.01)
    assert np.isnan(rms[3]), "the last unit's audio would come with a call after it"


def test_a_long_stretch_is_read_in_slices_and_a_new_one_gives_the_old_up(monkeypatch):
    monkeypatch.setattr(levels_module, "AacDecoder", _LaggingDecoder)
    ticks = iter(np.arange(0, 1000, 0.01))  # every call of the clock 10 ms later
    reader = LevelReader(clock=lambda: float(next(ticks)))
    audio = RecordingAudio("cut:x", FORMAT, (bytes([30]),) * 50 + (b"bad",) + (bytes([30]),) * 49)
    rms, reads = _read(reader, audio)
    assert len(reads) > 1, "handed on as it is read"
    assert all(first % STEPS_PER_UNIT == 0 for first, _count in reads)
    assert [first for first, _count in reads] == sorted(first for first, _count in reads)
    assert np.isnan(rms[49]) and not np.isnan(rms[48]) and not np.isnan(rms[51])

    started: list[str] = []
    reader.read.connect(lambda stretch_id, *_args: started.append(stretch_id))
    reader.start(RecordingAudio("cut:a", FORMAT, (bytes([1]),) * 400))
    reader.start(RecordingAudio("cut:b", FORMAT, (bytes([1]),) * 4))
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
    assert "cut:b" in started and "cut:a" not in started
