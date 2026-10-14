import numpy as np
import pytest

from dsp.time_interleave import TimeDeinterleaver


def test_time_deinterleaver_returns_zeros_before_any_history_exists():
    deinterleaver = TimeDeinterleaver(fragment_size=32)

    output = deinterleaver.push(np.full(32, 7, dtype=np.uint8))

    np.testing.assert_array_equal(output, np.zeros(32, dtype=np.uint8))


def test_time_deinterleaver_applies_known_per_position_delays():
    """Bit position i%16==0 has the longest delay (16 CIFs), i%16==15 the shortest
    (1 CIF), derived from welle.io's `interleaveMap` (see the docstring of
    `TimeDeinterleaver`)."""
    fragment_size = 32
    deinterleaver = TimeDeinterleaver(fragment_size)

    outputs = [
        deinterleaver.push(np.full(fragment_size, cif_index, dtype=np.uint8))
        for cif_index in range(20)
    ]

    for cif_index in range(16, 20):
        assert outputs[cif_index][0] == cif_index - 16  # longest delay: 16 CIFs
        assert outputs[cif_index][15] == cif_index - 1  # shortest delay: 1 CIF


def test_time_deinterleaver_raises_on_wrong_fragment_size():
    deinterleaver = TimeDeinterleaver(fragment_size=32)

    with pytest.raises(ValueError, match="fragment size"):
        deinterleaver.push(np.zeros(16, dtype=np.uint8))
