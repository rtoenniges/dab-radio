import numpy as np

from dsp.qpsk import demap_soft_bits, hard_decisions


def test_demap_soft_bits_matches_etsi_qpsk_mapper_convention():
    # q = (1/sqrt(2)) * [(1-2p_n) + j(1-2p_{n+K})]: p=0 -> +1, p=1 -> -1.
    scale = 1 / np.sqrt(2)
    symbols = np.array(
        [
            scale * (1 + 1j),  # p_n=0, p_{n+K}=0
            scale * (-1 + 1j),  # p_n=1, p_{n+K}=0
            scale * (1 - 1j),  # p_n=0, p_{n+K}=1
            scale * (-1 - 1j),  # p_n=1, p_{n+K}=1
        ]
    )

    bits = hard_decisions(demap_soft_bits(symbols))

    # First K=4 bits: real-part bits (p_n); last 4: imaginary-part bits (p_{n+K}).
    np.testing.assert_array_equal(bits[:4], [0, 1, 0, 1])
    np.testing.assert_array_equal(bits[4:], [0, 0, 1, 1])


def test_demap_soft_bits_output_length_is_double_input():
    symbols = np.array([1 + 1j, -1 - 1j, 1 - 1j])
    assert len(demap_soft_bits(symbols)) == 6


def test_demap_soft_bits_keeps_the_amplitude_as_confidence():
    """A carrier with a small amplitude (channel notch, noise) must yield soft bits of
    small magnitude, not be normalised to the same weight as a strong one."""
    soft = demap_soft_bits(np.array([2.0 - 2.0j, 0.1 + 0.05j]))

    np.testing.assert_allclose(soft, [2.0, 0.1, -2.0, 0.05], rtol=1e-6)
