"""Power spectrum and clipping estimate of the raw IQ stream, for the display."""

from __future__ import annotations

import numpy as np


def compute_power_spectrum(
    samples: np.ndarray,
    sample_rate: float,
    fft_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimates the power spectrum with an averaged, windowed periodogram.

    The sample buffer is cut into blocks of `fft_size`, weighted with a Hann window,
    transformed by FFT and the power averaged over all blocks (Welch's method), which
    reduces the spectral noise compared to a single FFT.

    Returns:
        Tuple of frequency axis (Hz, centred on 0) and power (dB, relative).

    Raises:
        ValueError: if fewer than one block of `fft_size` samples is available.
    """
    num_segments = len(samples) // fft_size
    if num_segments == 0:
        raise ValueError("Too few samples for the chosen FFT size.")

    window = np.hanning(fft_size)
    window_power = np.sum(window**2)

    segments = samples[: num_segments * fft_size].reshape(num_segments, fft_size)
    windowed = segments * window
    # Centred only after averaging: once on the mean instead of on every segment.
    spectra = np.fft.fft(windowed, axis=1)
    power = np.fft.fftshift(np.mean(np.abs(spectra) ** 2, axis=0)) / (window_power * sample_rate)

    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate))
    power_db = 10.0 * np.log10(np.maximum(power, 1e-20))
    return freqs, power_db


def detect_clipping(samples: np.ndarray, threshold: float = 0.99) -> float:
    """Estimates the fraction of overdriven samples (clipped at ADC full scale).

    pyrtlsdr normalises the raw 8-bit ADC values of each I/Q channel to roughly
    [-1, 1], and `sdr.rspduo_reader` scales the RSPduo's 14-bit values the same way; a
    channel value close to +/-1 means full scale. A gain that overdrives the
    tuner input produces a clearly measurable share of such full-scale samples
    (empirically on real hardware: a clean signal typically < 0.2 %, an overdriven one
    typically > 10 %), unlike a mere level change, which does not push the distribution
    against the limits. Useful in practice to explain to users why raising the gain
    further no longer has any visible effect (see `dsp.pipeline`).
    """
    return float(np.mean((np.abs(samples.real) > threshold) | (np.abs(samples.imag) > threshold)))
