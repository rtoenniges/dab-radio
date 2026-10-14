"""Tests for the DC offset and IQ imbalance correction."""

from __future__ import annotations

import math

import numpy as np

from dsp.iq_correction import IqCorrector


def _dab_like_signal(num_samples: int, seed: int = 0) -> np.ndarray:
    """Circularly symmetric noise: statistically like an OFDM signal with many
    independent subcarriers (I and Q uncorrelated, equally strong)."""
    rng = np.random.default_rng(seed)
    return (rng.normal(size=num_samples) + 1j * rng.normal(size=num_samples)) / np.sqrt(2)


def _apply_imbalance(samples: np.ndarray, gain_db: float, phase_deg: float) -> np.ndarray:
    """Distorts an IQ stream the way a real direct-conversion mixer does: the Q branch
    has a different gain and is not exactly 90° apart."""
    gain = 10.0 ** (gain_db / 20.0)
    phase = np.radians(phase_deg)
    in_phase = samples.real
    quadrature = gain * (samples.imag * np.cos(phase) + samples.real * np.sin(phase))
    return in_phase + 1j * quadrature


def _image_rejection_db(samples: np.ndarray, bin_index: int) -> float:
    """Ratio of the wanted tone to its mirror image in dB."""
    spectrum = np.abs(np.fft.fft(samples))
    return 20.0 * np.log10(spectrum[bin_index] / spectrum[-bin_index])


def test_disabled_correction_returns_input_untouched():
    samples = _dab_like_signal(4096) + (0.3 + 0.2j)
    result = IqCorrector().process(samples, remove_dc=False, correct_iq=False)
    assert result is samples


def test_removes_dc_offset():
    offset = 0.25 - 0.1j
    num_samples = 1 << 16
    samples = _dab_like_signal(num_samples) + offset
    corrector = IqCorrector()

    # Several blocks, since the estimate is smoothed exponentially.
    for _ in range(40):
        result = corrector.process(samples, remove_dc=True, correct_iq=False)

    # A mean estimator cannot get more accurate than the mean of the noise itself:
    # its standard deviation is 1/sqrt(N) per component. Tolerance with a safety
    # margin above that.
    noise_floor = 3.0 / math.sqrt(num_samples)
    assert abs(complex(result.mean())) < noise_floor
    assert abs(corrector.dc_offset - offset) < noise_floor


def test_dc_offset_is_not_removed_when_disabled():
    offset = 0.25 - 0.1j
    samples = _dab_like_signal(1 << 15) + offset
    result = IqCorrector().process(samples, remove_dc=False, correct_iq=True)
    assert abs(complex(result.mean()) - offset) < 0.05 * abs(offset)


def test_restores_orthogonality_and_equal_power():
    samples = _apply_imbalance(_dab_like_signal(1 << 16), gain_db=1.5, phase_deg=4.0)
    corrector = IqCorrector()
    for _ in range(40):
        result = corrector.process(samples, remove_dc=True, correct_iq=True)

    power_i = float(np.mean(result.real**2))
    power_q = float(np.mean(result.imag**2))
    correlation = float(np.mean(result.real * result.imag)) / np.sqrt(power_i * power_q)

    assert abs(correlation) < 1e-6, "I and Q must be uncorrelated afterwards"
    assert abs(power_q / power_i - 1.0) < 1e-6, "I and Q must be equally strong afterwards"


def test_suppresses_the_mirror_image_of_a_tone():
    """The actual effect: the mirror image disappears.

    A pure complex tone only has power at +f. An IQ imbalance additionally creates an
    image at -f, exactly what falls onto the wanted signal in DAB (signal symmetric
    around the centre frequency).
    """
    n = 1 << 14
    bin_index = 137
    tone = np.exp(2j * np.pi * bin_index * np.arange(n) / n)
    distorted = _apply_imbalance(tone, gain_db=1.5, phase_deg=4.0)

    before = _image_rejection_db(distorted, bin_index)
    corrector = IqCorrector()
    for _ in range(40):
        after = _image_rejection_db(
            corrector.process(distorted, remove_dc=True, correct_iq=True), bin_index
        )

    assert before < 35.0, "the test set-up must produce a noticeable mirror image at all"
    assert after > before + 40.0, (
        f"image rejection must rise clearly (before {before:.1f} dB, after {after:.1f} dB)"
    )


def test_reports_estimated_imbalance():
    samples = _apply_imbalance(_dab_like_signal(1 << 16), gain_db=1.5, phase_deg=4.0)
    corrector = IqCorrector()
    for _ in range(40):
        corrector.process(samples, remove_dc=True, correct_iq=True)

    # In the test set-up the gain acts on the already rotated Q branch, so the
    # measured value is the sum of both effects; what matters is that magnitude and
    # sign are right.
    assert 3.0 < corrector.quadrature_error_deg < 5.0
    assert -2.0 < corrector.gain_imbalance_db < -1.0


def test_silent_block_keeps_previous_estimate():
    samples = _apply_imbalance(_dab_like_signal(1 << 15), gain_db=1.5, phase_deg=4.0)
    corrector = IqCorrector()
    for _ in range(40):
        corrector.process(samples)
    before = (corrector.quadrature_error_deg, corrector.gain_imbalance_db)

    corrector.process(np.zeros(1024, dtype=np.complex128))

    assert (corrector.quadrature_error_deg, corrector.gain_imbalance_db) == before


def test_reset_discards_estimates():
    corrector = IqCorrector()
    corrector.process(_dab_like_signal(4096) + 0.3)
    corrector.reset()
    assert corrector.dc_offset == 0j
    assert corrector.quadrature_error_deg == 0.0
    assert corrector.gain_imbalance_db == 0.0
