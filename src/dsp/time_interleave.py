"""Time deinterleaving of a subchannel (ETSI EN 300 401, clause 11.2)."""

from __future__ import annotations

import numpy as np

# ETSI EN 300 401, clause 11.2.1: bit position mod 16 -> "branch" of the convolutional
# time interleaver. Cross-checked against welle.io (dab-audio.cpp, `interleaveMap`).
_INTERLEAVE_MAP = (0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15)
INTERLEAVE_DEPTH = 16


class TimeDeinterleaver:
    """Undoes the DAB time interleaver for a single subchannel (ETSI EN 300 401,
    clause 11.2: convolutional interleaver across 16 CIFs).

    At the transmitter, a bit at position i (0-based within a CIF fragment, i.e. the
    `subchannel.size * 64` bits a subchannel occupies per CIF) is delayed by
    `_INTERLEAVE_MAP[i mod 16]` CIFs (0 to 15). `push()` therefore delivers the
    matching output `16 - _INTERLEAVE_MAP[i mod 16]` calls later, see
    `test_time_deinterleaver_applies_known_per_position_delays` and the full
    cross-layer round trip in `tests/test_msc_eep_integration.py`, which models exactly
    this transmit direction. The deinterleaver therefore needs a ring buffer over the
    last 16 received CIFs; the first 16 calls of `push()` do not yet deliver fully
    valid output (the delay stages not yet filled contain zeros instead of real bits),
    which corresponds to the usual DAB start-up delay of about 384 ms until a service
    is first decoded completely.

    It carries soft bits (see `dsp.qpsk.demap_soft_bits`), so that their confidence
    reaches the Viterbi decoder; the permutation is the same as for hard bits.
    """

    def __init__(self, fragment_size: int) -> None:
        self._fragment_size = fragment_size
        self._history = np.zeros((INTERLEAVE_DEPTH, fragment_size), dtype=np.float32)
        self._index = 0
        self._delay = np.array(
            [_INTERLEAVE_MAP[i % INTERLEAVE_DEPTH] for i in range(fragment_size)], dtype=np.int64
        )
        self._bit_positions = np.arange(fragment_size)

    def push(self, cif_bits: np.ndarray) -> np.ndarray:
        """Takes the raw (still interleaved) soft bits of one CIF fragment and returns
        the soft bits deinterleaved at this point (length `fragment_size`, as passed to
        the constructor)."""
        if len(cif_bits) != self._fragment_size:
            raise ValueError("CIF fragment size does not match the deinterleaver.")

        slot_indices = (self._index + self._delay) & (INTERLEAVE_DEPTH - 1)
        output = self._history[slot_indices, self._bit_positions]

        self._history[self._index] = cif_bits
        self._index = (self._index + 1) & (INTERLEAVE_DEPTH - 1)
        return output
