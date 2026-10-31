import numpy as np
import pytest

from dab.constants import MODE_I
from dsp.sync import NULL_CANDIDATES, find_null_symbol, null_symbol_candidates

SAMPLE_RATE = 2_048_000.0
NULL = MODE_I.null_symbol_length


def _noise(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(length) + 1j * rng.standard_normal(length)).astype(np.complex64)


def _band(samples: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    """Only what lies between `low_hz` and `high_hz` of the samples."""
    spectrum = np.fft.fft(samples)
    frequencies = np.fft.fftfreq(len(samples), 1 / SAMPLE_RATE)
    spectrum[(frequencies < low_hz) | (frequencies >= high_hz)] = 0
    return np.fft.ifft(spectrum)


def test_find_null_symbol_locates_low_power_segment():
    null_start = 7333
    samples = _noise(20_000, seed=42)
    samples[null_start : null_start + NULL] *= 0.01

    detected = find_null_symbol(samples, MODE_I)

    assert abs(detected - null_start) <= 2


def test_find_null_symbol_raises_when_buffer_too_short():
    samples = np.zeros(NULL - 1, dtype=np.complex64)
    with pytest.raises(ValueError, match="shorter than a null symbol"):
        find_null_symbol(samples, MODE_I)


def test_find_null_symbol_max_start_ignores_later_null_symbol():
    # Two null symbols in the buffer (simulating a buffer spanning several frames);
    # without the limit the later (deeper) one might be found.
    early_null_start = 3000
    late_null_start = 14_000
    samples = _noise(20_000, seed=3)
    samples[early_null_start : early_null_start + NULL] *= 0.05
    samples[late_null_start : late_null_start + NULL] *= 0.01  # deeper than the early one

    # Without the limit the deeper (later) null symbol is found.
    assert abs(find_null_symbol(samples, MODE_I) - late_null_start) <= 2

    # With the search limited to the range before the late null symbol, the early
    # one is found reliably instead.
    detected = find_null_symbol(samples, MODE_I, max_start=late_null_start - 50)
    assert abs(detected - early_null_start) <= 2


def test_find_null_symbol_ignores_the_pauses_of_a_stronger_neighbour_beside_the_band():
    """The edge of a neighbouring channel within the sampled band, four times as strong
    as the ensemble and with null symbols at a frame clock of its own: in the total
    power its pause is the deeper minimum, but it lies beside the ensemble's carriers."""
    null_start, neighbour_pause = 7000, 19_000
    ensemble = _band(_noise(30_000, seed=5), -700e3, 700e3)
    ensemble[null_start : null_start + NULL] = 0
    neighbour = _band(_noise(30_000, seed=6), 950e3, 1020e3)
    neighbour *= np.sqrt(4 * np.mean(np.abs(ensemble) ** 2) / np.mean(np.abs(neighbour) ** 2))
    neighbour[neighbour_pause : neighbour_pause + NULL] = 0
    samples = (ensemble + neighbour).astype(np.complex64)

    total = np.cumsum(np.insert(np.abs(samples) ** 2, 0, 0.0))
    assert abs(int(np.argmin(total[NULL:] - total[:-NULL])) - neighbour_pause) <= 2
    assert abs(find_null_symbol(samples, MODE_I) - null_start) <= 2


def test_candidates_list_the_deepest_minimum_first_and_then_further_dips():
    shallow, deep = 5000, 20_000
    samples = _noise(40_000, seed=7)
    samples[shallow : shallow + NULL] *= 0.3
    samples[deep : deep + NULL] *= 0.1

    candidates = null_symbol_candidates(samples, MODE_I)

    assert len(candidates) == 2
    assert abs(candidates[0] - deep) <= 2
    assert abs(candidates[1] - shallow) <= 2


def test_candidates_leave_out_the_ripple_of_the_noise():
    null_start = 12_000
    samples = _noise(40_000, seed=8)
    samples[null_start : null_start + NULL] *= 0.1

    assert len(null_symbol_candidates(_noise(40_000, seed=9), MODE_I)) == 1
    candidates = null_symbol_candidates(samples, MODE_I)
    assert len(candidates) == 1 and abs(candidates[0] - null_start) <= 2


def test_candidates_are_limited_to_the_deepest_dips():
    starts = [3000, 12_000, 21_000, 30_000, 39_000]
    samples = _noise(50_000, seed=10)
    for depth, start in enumerate(starts):
        samples[start : start + NULL] *= 0.05 * (depth + 1)

    candidates = null_symbol_candidates(samples, MODE_I)

    assert len(candidates) == NULL_CANDIDATES
    for candidate, start in zip(candidates, starts, strict=False):
        assert abs(candidate - start) <= 2
