"""The spectrum of the decoded audio, for the display behind the curve of the equalizer:
the level in bands of a sixth of an octave from 20 Hz to 20 kHz, ten times a second of
audio. Taken before the equalizer, so the curve shows what it does to what the station
sends."""

from __future__ import annotations

import numpy as np

FFT_SIZE = 4096
INTERVAL_SECONDS = 0.1
BANDS_PER_OCTAVE = 6
LOWEST_HZ = 20.0
HIGHEST_HZ = 20_000.0
# Below this a band reads as silence; also what an empty one reads.
FLOOR_DB = -120.0

# The centres of the bands; those above half the sample rate are left out.
CENTRES_HZ = LOWEST_HZ * 2.0 ** (
    np.arange(int(np.log2(HIGHEST_HZ / LOWEST_HZ) * BANDS_PER_OCTAVE) + 1) / BANDS_PER_OCTAVE
)

_WINDOW = np.hanning(FFT_SIZE)
# A sine at full scale reads 0 dBFS: the window takes the sum of its weights off the
# amplitude, and spreads the power of a sine over its equivalent noise bandwidth.
_AMPLITUDE_SCALE = 2.0 / _WINDOW.sum()
_NOISE_BANDWIDTH_BINS = FFT_SIZE * float((_WINDOW**2).sum()) / float(_WINDOW.sum()) ** 2


def _band_bins(sample_rate: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """The centres below half the sample rate and the FFT bins of each band, at least
    the one at its centre: at 48 kHz a bin is 11.7 Hz wide, more than a low band."""
    resolution = sample_rate / FFT_SIZE
    centres = CENTRES_HZ[CENTRES_HZ < sample_rate / 2]
    half_band = 2.0 ** (1.0 / (2 * BANDS_PER_OCTAVE))
    ranges = []
    for centre in centres:
        start = int(np.ceil(centre / half_band / resolution))
        stop = int(np.ceil(centre * half_band / resolution))
        if stop <= start:
            start = round(centre / resolution)
            stop = start + 1
        ranges.append((start, min(stop, FFT_SIZE // 2 + 1)))
    return centres, ranges


class SpectrumAnalyzer:
    """Collects decoded PCM and makes a spectrum of the latest `FFT_SIZE` samples
    every `INTERVAL_SECONDS` of audio, of both channels together."""

    def __init__(self) -> None:
        self._sample_rate = 0
        self._samples = np.zeros(0)
        self._since = 0  # samples since the last spectrum
        self._centres = np.zeros(0)
        self._ranges: list[tuple[int, int]] = []

    def reset(self) -> None:
        self._samples = np.zeros(0)
        self._since = 0

    def add(
        self, pcm: np.ndarray, channels: int, sample_rate: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Takes decoded PCM (interleaved int16); returns the spectra it completes,
        each as the band centres (Hz) and their levels (dBFS), mostly none or one."""
        if channels <= 0 or len(pcm) < channels or sample_rate <= 0:
            return []
        if sample_rate != self._sample_rate:
            self._sample_rate = sample_rate
            self._centres, self._ranges = _band_bins(sample_rate)
            self.reset()
        mono = pcm[: len(pcm) // channels * channels].reshape(-1, channels).mean(axis=1)
        self._samples = np.concatenate([self._samples, mono / 32768.0])[-FFT_SIZE:]
        self._since += len(mono)
        interval = int(INTERVAL_SECONDS * sample_rate)
        if len(self._samples) < FFT_SIZE or self._since < interval:
            return []
        self._since %= interval  # several intervals in one block make one spectrum
        return [(self._centres, self._levels())]

    def _levels(self) -> np.ndarray:
        amplitude = np.abs(np.fft.rfft(self._samples * _WINDOW)) * _AMPLITUDE_SCALE
        power = amplitude**2
        sums = np.array([power[start:stop].sum() for start, stop in self._ranges])
        levels = 10.0 * np.log10(np.maximum(sums / _NOISE_BANDWIDTH_BINS, 1e-30))
        return np.maximum(levels, FLOOR_DB)
