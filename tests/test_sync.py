import numpy as np
import pytest

from dsp.sync import find_null_symbol


def test_find_null_symbol_locates_low_power_segment():
    rng = np.random.default_rng(42)
    null_length = 100
    total_length = 2000
    null_start = 733

    noise = (rng.standard_normal(total_length) + 1j * rng.standard_normal(total_length)).astype(
        np.complex64
    )
    samples = noise.copy()
    samples[null_start : null_start + null_length] *= 0.01

    detected = find_null_symbol(samples, null_length)

    assert abs(detected - null_start) <= 2


def test_find_null_symbol_raises_when_buffer_too_short():
    samples = np.zeros(50, dtype=np.complex64)
    with pytest.raises(ValueError, match="shorter than a null symbol"):
        find_null_symbol(samples, null_symbol_length=100)


def test_find_null_symbol_max_start_ignores_later_null_symbol():
    # Two null symbols in the buffer (simulating a buffer spanning several frames);
    # without the limit the later (deeper) one might be found.
    rng = np.random.default_rng(3)
    null_length = 100
    total_length = 2000
    early_null_start = 300
    late_null_start = 1600

    noise = (rng.standard_normal(total_length) + 1j * rng.standard_normal(total_length)).astype(
        np.complex64
    )
    samples = noise.copy()
    samples[early_null_start : early_null_start + null_length] *= 0.05
    samples[late_null_start : late_null_start + null_length] *= 0.01  # deeper than the early one

    # Without the limit the deeper (later) null symbol is found.
    assert abs(find_null_symbol(samples, null_length) - late_null_start) <= 2

    # With the search limited to the range before the late null symbol, the early
    # one is found reliably instead.
    detected = find_null_symbol(samples, null_length, max_start=late_null_start - 50)
    assert abs(detected - early_null_start) <= 2
