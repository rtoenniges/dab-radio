"""Tests for the spectrum of the audio (`audio.spectrum`): a tone shows where it is and
as loud as it is, and the spectra come at a steady rate whatever the block size."""

from __future__ import annotations

import numpy as np
import pytest

from audio.spectrum import CENTRES_HZ, FFT_SIZE, FLOOR_DB, INTERVAL_SECONDS, SpectrumAnalyzer

RATE = 48_000


def _sine(frequency: float, amplitude: float, seconds: float, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    tone = np.round(amplitude * 32767 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
    return np.repeat(tone, 2)


def test_a_tone_at_full_scale_reads_0_dbfs_in_its_band_and_little_elsewhere():
    analyzer = SpectrumAnalyzer()

    spectra = analyzer.add(_sine(1000.0, 1.0, 0.2), 2, RATE)

    centres, levels = spectra[-1]
    band = int(np.argmin(np.abs(np.log2(centres / 1000.0))))
    assert levels[band] == pytest.approx(0.0, abs=0.5)
    two_octaves = np.abs(np.log2(centres / 1000.0)) >= 2
    assert levels[two_octaves].max() < -40


def test_spectra_come_ten_times_a_second_of_audio_in_blocks_of_any_size():
    analyzer = SpectrumAnalyzer()
    audio = _sine(440.0, 0.5, 1.0)
    blocks = np.array_split(audio, len(audio) // (2 * 960))  # 20 ms

    count = sum(len(analyzer.add(block, 2, RATE)) for block in blocks)

    assert INTERVAL_SECONDS == pytest.approx(0.1)
    assert FFT_SIZE / RATE < INTERVAL_SECONDS  # the first one after 0.1 s
    assert count == 10


def test_bands_above_half_the_sample_rate_are_left_out():
    analyzer = SpectrumAnalyzer()

    centres, levels = analyzer.add(_sine(1000.0, 0.5, 0.2, rate=32_000), 2, 32_000)[-1]

    assert centres.max() < 16_000
    assert len(centres) == len(levels) < len(CENTRES_HZ)


def test_silence_reads_the_floor():
    analyzer = SpectrumAnalyzer()

    _centres, levels = analyzer.add(np.zeros(2 * RATE // 5, dtype=np.int16), 2, RATE)[-1]

    assert (levels == FLOOR_DB).all()
