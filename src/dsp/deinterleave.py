"""Frequency deinterleaving: mapping the OFDM subcarriers back to logical bit order."""

from __future__ import annotations

import functools

import numpy as np


def _generate_permutation(size: int) -> np.ndarray:
    """Generates the DAB frequency interleaving permutation Π (ETSI EN 300 401, 14.6.1).

    Π(i) = 13*Π(i-1) + 511 (mod `size`), Π(0) = 0.
    """
    perm = np.zeros(size, dtype=np.int64)
    for i in range(1, size):
        perm[i] = (13 * perm[i - 1] + 511) % size
    return perm


def logical_carrier_order(num_carriers: int, fft_length: int) -> np.ndarray:
    """Returns the carrier index k for every logical bit position n=0..num_carriers-1.

    Corresponds to table 25 (ETSI EN 300 401): k[n] is the physical carrier index
    (negative = below the centre frequency, positive = above, DC at k=0 excluded) that
    the logical QPSK symbol position n is mapped to.
    """
    perm = _generate_permutation(fft_length)
    center = fft_length // 2
    half = num_carriers // 2
    mask = (perm >= center - half) & (perm <= center + half) & (perm != center)
    d = perm[mask]
    return d - center


@functools.cache
def frequency_deinterleave_index(num_carriers: int, fft_length: int) -> np.ndarray:
    """Maps logical bit positions n=0..num_carriers-1 to the physical array positions
    as delivered by `dsp.ofdm.extract_used_carriers` (carrier index k=-half..-1 at
    positions 0..half-1, k=1..half at positions half..num_carriers-1).

    Constant for a transmission mode, but the permutation behind it is a Python loop
    over `fft_length` steps, and FIC and MSC ask for it in every frame. So it is
    cached, and the returned array is read-only.
    """
    k = logical_carrier_order(num_carriers, fft_length)
    half = num_carriers // 2
    index = np.where(k < 0, k + half, k + half - 1).astype(np.int64)
    index.setflags(write=False)
    return index


def deinterleave_frequency(physical_order: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Applies the frequency deinterleaving permutation.

    `physical_order` must be in the order delivered by `extract_used_carriers`; `index`
    comes from `frequency_deinterleave_index`. The result is in logical bit order
    n=0..num_carriers-1.
    """
    return physical_order[index]
