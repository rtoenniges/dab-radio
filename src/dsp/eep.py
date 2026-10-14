"""Equal Error Protection (EEP): depuncturing and Viterbi decoding of MSC logical frames.

`decode_punctured_logical_frame` is the part shared with UEP (`dsp.uep`): both only
differ in how the logical frame is split into differently punctured segments.
"""

from __future__ import annotations

import numpy as np

from dab.fig import Subchannel
from dsp.convolutional import TAIL_PUNCTURE_PATTERN, depuncture, viterbi_decode
from dsp.scrambling import apply_energy_dispersal

# ETSI EN 300 401, table 13: the 24 puncturing patterns (32 bits each), cross-checked
# against welle.io (protTables.cpp, `p_codes`). Index 0 is ETSI pattern PI_1, index 23
# is PI_24.
# fmt: off
_P_CODES = np.array(
    [
        [1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=np.uint8,
)
# fmt: on

# Information bits per "L block": the block holds 128 bits of the rate-1/4 mother code
# and is punctured with the same 32-bit pattern four times over (ETSI EN 300 401,
# clause 11.3).
BLOCK_INFO_BITS = 32


def compute_eep_bitrate(subchannel: Subchannel) -> int:
    """Computes the effective bit rate (in kbit/s) of an EEP subchannel from its size
    in CU (ETSI EN 300 401, clause 11.3.2; see `dab.fig.Subchannel.bitrate`)."""
    if subchannel.is_uep:
        raise ValueError("compute_eep_bitrate is only defined for EEP subchannels.")
    return subchannel.bitrate


def _eep_a_puncturing(level: int, bitrate: int) -> tuple[int, int, int, int]:
    """Returns (L1, L2, PI1 index, PI2 index) for EEP profile A, level 0..3 (internal
    numbering; ETSI EEP levels 1..4). Cross-checked against welle.io
    (eep-protection.cpp)."""
    if level == 0:
        return 6 * bitrate // 8 - 3, 3, 23, 22
    if level == 1:
        if bitrate == 8:
            return 5, 1, 12, 11
        return 2 * bitrate // 8 - 3, 4 * bitrate // 8 + 3, 13, 12
    if level == 2:
        return 6 * bitrate // 8 - 3, 3, 7, 6
    if level == 3:
        return 4 * bitrate // 8 - 3, 2 * bitrate // 8 + 3, 2, 1
    raise ValueError(f"Invalid EEP-A level: {level}")


def _eep_b_puncturing(level: int, bitrate: int) -> tuple[int, int, int, int]:
    """Like `_eep_a_puncturing`, for EEP profile B."""
    l1 = 24 * bitrate // 32 - 3
    l2 = 3
    if level == 3:
        return l1, l2, 1, 0
    if level == 2:
        return l1, l2, 3, 2
    if level == 1:
        return l1, l2, 5, 4
    if level == 0:
        return l1, l2, 9, 8
    raise ValueError(f"Invalid EEP-B level: {level}")


def eep_segments(subchannel: Subchannel) -> list[tuple[int, int]]:
    """The puncturing of an EEP subchannel as (L, pattern index) pairs, see
    `decode_punctured_logical_frame`."""
    puncturing = _eep_a_puncturing if subchannel.eep_option == 0 else _eep_b_puncturing
    l1, l2, pi1_index, pi2_index = puncturing(subchannel.protection_index, subchannel.bitrate)
    return [(l1, pi1_index), (l2, pi2_index)]


def _segment_patterns(segments: list[tuple[int, int]]) -> list[tuple[int, np.ndarray]]:
    """(L, 128-bit puncturing pattern) per segment, followed by the tail with `PI_X`."""
    patterns = [(num_blocks, np.tile(_P_CODES[index], 4)) for num_blocks, index in segments]
    patterns.append((1, TAIL_PUNCTURE_PATTERN))
    return patterns


def punctured_length(segments: list[tuple[int, int]]) -> int:
    """Number of coded bits that the segments and the tail occupy after puncturing."""
    return sum(
        int(pattern.sum()) * num_blocks for num_blocks, pattern in _segment_patterns(segments)
    )


def decode_punctured_logical_frame(
    coded_bits: np.ndarray, segments: list[tuple[int, int]], padding_bits: int = 0
) -> np.ndarray:
    """Depunctures, Viterbi-decodes and descrambles one MSC logical frame codeword
    (ETSI EN 300 401, clauses 11.1-11.3).

    The codeword is a sequence of segments, each `L` blocks of 128 mother code bits
    punctured with one pattern `PI` (repeated four times per block), followed by the
    24 tail bits punctured with `PI_X`. EEP uses two such segments, UEP up to four;
    UEP may additionally append a few padding bits that carry nothing.

    Args:
        coded_bits: the time-deinterleaved soft bits of one CIF share (see
            `dsp.convolutional.viterbi_decode`).
        segments: (L, 0-based index into the 24 puncturing patterns) per segment.
        padding_bits: number of padding bits after the tail.

    Returns:
        The information bits: 32 per block, i.e. `24 * bitrate` per logical frame.
    """
    patterns = _segment_patterns(segments)
    lengths = [int(pattern.sum()) * num_blocks for num_blocks, pattern in patterns]
    expected = sum(lengths) + padding_bits
    if len(coded_bits) != expected:
        raise ValueError(
            f"Expected {expected} coded bits for this protection profile, "
            f"but got {len(coded_bits)}."
        )

    received_parts = []
    offset = 0
    for (_, pattern), length in zip(patterns, lengths, strict=True):
        received_parts.append(depuncture(coded_bits[offset : offset + length], pattern))
        offset += length

    num_info_bits = sum(num_blocks for num_blocks, _ in segments) * BLOCK_INFO_BITS
    scrambled = viterbi_decode(np.concatenate(received_parts), num_info_bits)
    return apply_energy_dispersal(scrambled)


def decode_eep_logical_frame(coded_bits: np.ndarray, subchannel: Subchannel) -> np.ndarray:
    """Decodes an EEP-protected MSC logical frame codeword (the time-deinterleaved bits
    of one CIF share) to the information bits it carries (ETSI EN 300 401, clause
    11.3.2).

    A DAB "logical frame" here is exactly one CIF (24 ms): one independent Viterbi pass
    per CIF.
    """
    if subchannel.is_uep:
        raise ValueError("decode_eep_logical_frame is only defined for EEP subchannels.")
    return decode_punctured_logical_frame(coded_bits, eep_segments(subchannel))
