import numpy as np

from dsp.timing import (
    find_integer_symbol_offset,
    generate_reference_symbol,
    physical_carrier_indices,
)


def _reference_useful_part(num_carriers: int, fft_length: int) -> np.ndarray:
    known = generate_reference_symbol(num_carriers)
    half = num_carriers // 2
    ref_raw = np.zeros(fft_length, dtype=complex)
    ref_raw[1 : half + 1] = known[half:]
    ref_raw[fft_length - half :] = known[:half]
    return np.fft.ifft(ref_raw)


def test_generate_reference_symbol_has_unit_magnitude():
    prs = generate_reference_symbol(num_carriers=1536)
    assert len(prs) == 1536
    np.testing.assert_allclose(np.abs(prs), 1.0, atol=1e-10)


def test_physical_carrier_indices_excludes_dc_and_spans_full_range():
    k = physical_carrier_indices(num_carriers=1536)
    assert len(k) == 1536
    assert k.min() == -768
    assert k.max() == 768
    assert 0 not in k


def test_find_integer_symbol_offset_recovers_known_shift():
    fft_length = 2048
    num_carriers = 1536
    useful = _reference_useful_part(num_carriers, fft_length)

    true_shift = 37
    window = np.roll(useful, true_shift)

    offset, confidence = find_integer_symbol_offset(window, num_carriers, fft_length)

    assert offset == true_shift
    assert confidence > 10


def test_find_integer_symbol_offset_handles_negative_shift():
    fft_length = 2048
    num_carriers = 1536
    useful = _reference_useful_part(num_carriers, fft_length)

    true_shift = -120
    window = np.roll(useful, true_shift)

    offset, confidence = find_integer_symbol_offset(window, num_carriers, fft_length)

    assert offset == true_shift
    assert confidence > 10


def test_find_integer_symbol_offset_low_confidence_for_unrelated_signal():
    fft_length = 2048
    num_carriers = 1536
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(fft_length) + 1j * rng.standard_normal(fft_length)

    _, confidence = find_integer_symbol_offset(noise, num_carriers, fft_length)

    assert confidence < 5
