import numpy as np
import pytest

from dsp.spectrum import compute_power_spectrum, detect_clipping


def test_compute_power_spectrum_peaks_at_tone_frequency():
    sample_rate = 2_048_000.0
    fft_size = 1024
    tone_freq = 200_000.0
    num_samples = fft_size * 8

    t = np.arange(num_samples) / sample_rate
    tone = np.exp(2j * np.pi * tone_freq * t).astype(np.complex64)

    freqs, power_db = compute_power_spectrum(tone, sample_rate, fft_size=fft_size)

    peak_freq = freqs[np.argmax(power_db)]
    bin_width = sample_rate / fft_size
    assert abs(peak_freq - tone_freq) <= bin_width


def test_compute_power_spectrum_raises_when_too_few_samples():
    samples = np.zeros(10, dtype=np.complex64)
    with pytest.raises(ValueError, match="Too few samples"):
        compute_power_spectrum(samples, sample_rate=2_048_000.0, fft_size=1024)


def test_detect_clipping_returns_zero_for_clean_signal():
    rng = np.random.default_rng(5)
    # Small amplitude so that even the rarest random outliers stay safely below the
    # clipping threshold (default 0.99).
    samples = 0.1 * (rng.standard_normal(10_000) + 1j * rng.standard_normal(10_000))

    assert detect_clipping(samples) == 0.0


def test_detect_clipping_detects_saturated_samples():
    samples = np.full(1000, 1.0 + 1.0j, dtype=np.complex128)
    samples[:100] = 0.1 + 0.1j  # 10 % below the threshold, the rest overdriven

    fraction = detect_clipping(samples, threshold=0.99)

    assert fraction == pytest.approx(0.9)
