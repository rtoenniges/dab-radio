import numpy as np

from dsp.deinterleave import (
    deinterleave_frequency,
    frequency_deinterleave_index,
    logical_carrier_order,
)

# Reference values from ETSI EN 300 401, table 25 (frequency interleaving for
# transmission mode I): k values for n=0..8.
EXPECTED_K_FIRST_9 = [-513, -14, 329, 692, -733, 13, 680, 273, -36]


def test_logical_carrier_order_matches_spec_table_25():
    k = logical_carrier_order(num_carriers=1536, fft_length=2048)

    assert len(k) == 1536
    np.testing.assert_array_equal(k[:9], EXPECTED_K_FIRST_9)


def test_logical_carrier_order_is_permutation_of_used_carriers():
    k = logical_carrier_order(num_carriers=1536, fft_length=2048)

    expected_carriers = np.concatenate([np.arange(-768, 0), np.arange(1, 769)])
    np.testing.assert_array_equal(np.sort(k), expected_carriers)


def test_deinterleave_frequency_is_invertible():
    rng = np.random.default_rng(0)
    physical = rng.standard_normal(1536) + 1j * rng.standard_normal(1536)
    index = frequency_deinterleave_index(num_carriers=1536, fft_length=2048)

    logical = deinterleave_frequency(physical, index)

    # Every physical position must appear exactly once in the logical order.
    assert len(np.unique(index)) == 1536
    reconstructed = np.empty_like(physical)
    reconstructed[index] = logical
    np.testing.assert_array_equal(reconstructed, physical)
