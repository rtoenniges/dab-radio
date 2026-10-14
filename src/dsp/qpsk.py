"""QPSK demapping of the differentially demodulated subcarriers."""

from __future__ import annotations

import numpy as np


def demap_soft_bits(logical_symbols: np.ndarray) -> np.ndarray:
    """Demaps complex QPSK symbols (in logical carrier order) to soft bits.

    ETSI EN 300 401, clause 14.5 (QPSK symbol mapper):
        q[n] = (1/sqrt(2)) * [(1 - 2*p[n]) + j*(1 - 2*p[n+K])]

    A bit value of 0 therefore corresponds to a positive, a bit value of 1 to a negative
    real or imaginary part. For `K = len(logical_symbols)` symbols the result has length
    `2*K`: first all K "real part bits" p[0..K-1], then all K "imaginary part bits"
    p[K..2K-1], not interleaved bit by bit.

    The soft bit is the real or imaginary part itself: its sign is the bit, its
    magnitude the confidence (see `dsp.convolutional.viterbi_decode`). For D-QPSK
    (`z(l) * conj(z(l-1))`) that magnitude grows with the square of the carrier's
    channel gain, so carriers in a notch of the channel or drowned in noise
    automatically count for little. Deliberately not normalised per symbol or carrier:
    that would throw exactly this weighting away.

    Args:
        logical_symbols: complex (differentially demodulated) QPSK symbols in logical
            order n=0..K-1 (see `dsp.deinterleave`).

    Returns:
        float32 array of length `2*len(logical_symbols)`.
    """
    return np.concatenate([logical_symbols.real, logical_symbols.imag]).astype(np.float32)


def hard_decisions(soft_bits: np.ndarray) -> np.ndarray:
    """The bits the soft bits stand for (negative = 1), as uint8."""
    return (soft_bits < 0).astype(np.uint8)
