"""Power spectrum and clipping estimate of the raw IQ stream, for the display."""

from __future__ import annotations

from functools import cache

import numpy as np

# Segments averaged into one spectrum at most, spread evenly over the samples. A read
# block holds 192 of 2048 samples: all of them would cost as many FFTs as demodulating
# the two frames in it, for a display. With 32 the curve varies a little more from one
# block to the next (0.7 instead of 0.4 dB in the band) at a sixth of the FFTs.
MAX_SEGMENTS = 32


@cache
def _hann_window(fft_size: int) -> np.ndarray:
    """In single precision, so that complex64 samples stay complex64."""
    return np.hanning(fft_size).astype(np.float32)


def compute_power_spectrum(
    samples: np.ndarray,
    sample_rate: float,
    fft_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimates the power spectrum with an averaged, windowed periodogram.

    The sample buffer is cut into blocks of `fft_size`, weighted with a Hann window,
    transformed by FFT and the power averaged over the blocks (Welch's method), which
    reduces the spectral noise compared to a single FFT; over at most `MAX_SEGMENTS`
    of them, evenly spaced.

    Returns:
        Tuple of frequency axis (Hz, centred on 0) and power (dB, relative).

    Raises:
        ValueError: if fewer than one block of `fft_size` samples is available.
    """
    num_segments = len(samples) // fft_size
    if num_segments == 0:
        raise ValueError("Too few samples for the chosen FFT size.")

    window = _hann_window(fft_size)
    window_power = float(np.sum(window.astype(np.float64) ** 2))

    segments = samples[: num_segments * fft_size].reshape(num_segments, fft_size)
    segments = segments[:: -(-num_segments // MAX_SEGMENTS)]
    windowed = segments * window
    # Centred only after averaging: once on the mean instead of on every segment.
    spectra = np.fft.fft(windowed, axis=1)
    magnitude = spectra.real**2 + spectra.imag**2
    power = np.fft.fftshift(np.mean(magnitude, axis=0)) / (window_power * sample_rate)

    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate))
    power_db = 10.0 * np.log10(np.maximum(power, 1e-20))
    return freqs, power_db


def detect_clipping(samples: np.ndarray, threshold: float = 0.99) -> float:
    """Estimates the fraction of overdriven samples (clipped at ADC full scale).

    pyrtlsdr normalises the raw 8-bit ADC values of each I/Q channel to roughly
    [-1, 1], and `sdr.rspduo_reader` scales the RSPduo's 14-bit values the same way; a
    channel value close to +/-1 means full scale. A gain that overdrives the tuner
    input produces a clearly measurable share of such full-scale samples (a clean
    signal typically < 0.2 %, an overdriven one > 10 %), unlike a mere level change,
    which does not push the distribution against the limits. Useful in practice to
    explain to users why raising the gain further no longer has any visible effect
    (see `dsp.pipeline`).
    """
    return float(np.mean((np.abs(samples.real) > threshold) | (np.abs(samples.imag) > threshold)))
