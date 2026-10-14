"""Unequal Error Protection (UEP): depuncturing and Viterbi decoding of MSC logical frames.

UEP is the protection scheme of classic DAB audio (MPEG Layer II). Unlike EEP it
protects the logical frame unevenly, following the layout of an MP2 frame: the start
(header, bit allocation, scale factors) gets the strongest code rate, the sample data
in the middle the weakest, since errors there are merely audible as noise, and the end
(scale factor CRC and PAD) again a stronger one. That is why a UEP logical frame
consists of up to four differently punctured segments instead of two.
"""

from __future__ import annotations

import numpy as np

from dab.fig import Subchannel
from dsp.eep import decode_punctured_logical_frame, punctured_length

# ETSI EN 300 401, clause 11.3.1: puncturing per (bit rate in kbit/s, protection level)
# as (L1, L2, L3, L4, PI1, PI2, PI3, PI4). PI numbers are the ETSI numbering 1..24;
# L4 = 0 means only three segments.
#
# Taken from welle.io (uep-protection.cpp, `profileTable`) and compared entry by entry
# with the transmitter side, ODR-DabMod (SubchannelSource.cpp). The two agree except for
# (80 kbit/s, level 1), where welle.io has PI2 = 7 and ODR-DabMod PI2 = 17. 17 is right:
# with it the codeword fills the subchannel size from `dab.fig.UEP_TABLE` up to 4
# padding bits, like every other entry (0, 4 or 8), while 7 would leave 404 bits unused.
# fmt: off
_UEP_PUNCTURING = {
    (32, 5): (3, 4, 17, 0, 5, 3, 2, 0),
    (32, 4): (3, 3, 18, 0, 11, 6, 5, 0),
    (32, 3): (3, 4, 14, 3, 15, 9, 6, 8),
    (32, 2): (3, 4, 14, 3, 22, 13, 8, 13),
    (32, 1): (3, 5, 13, 3, 24, 17, 12, 17),
    (48, 5): (4, 3, 26, 3, 5, 4, 2, 3),
    (48, 4): (3, 4, 26, 3, 9, 6, 4, 6),
    (48, 3): (3, 4, 26, 3, 15, 10, 6, 9),
    (48, 2): (3, 4, 26, 3, 24, 14, 8, 15),
    (48, 1): (3, 5, 25, 3, 24, 18, 13, 18),
    (56, 5): (6, 10, 23, 3, 5, 4, 2, 3),
    (56, 4): (6, 10, 23, 3, 9, 6, 4, 5),
    (56, 3): (6, 12, 21, 3, 16, 7, 6, 9),
    (56, 2): (6, 10, 23, 3, 23, 13, 8, 13),
    (64, 5): (6, 9, 31, 2, 5, 3, 2, 3),
    (64, 4): (6, 9, 33, 0, 11, 6, 5, 0),
    (64, 3): (6, 12, 27, 3, 16, 8, 6, 9),
    (64, 2): (6, 10, 29, 3, 23, 13, 8, 13),
    (64, 1): (6, 11, 28, 3, 24, 18, 12, 18),
    (80, 5): (6, 10, 41, 3, 6, 3, 2, 3),
    (80, 4): (6, 10, 41, 3, 11, 6, 5, 6),
    (80, 3): (6, 11, 40, 3, 16, 8, 6, 7),
    (80, 2): (6, 10, 41, 3, 23, 13, 8, 13),
    (80, 1): (6, 10, 41, 3, 24, 17, 12, 18),
    (96, 5): (7, 9, 53, 3, 5, 4, 2, 4),
    (96, 4): (7, 10, 52, 3, 9, 6, 4, 6),
    (96, 3): (6, 12, 51, 3, 16, 9, 6, 10),
    (96, 2): (6, 10, 53, 3, 22, 12, 9, 12),
    (96, 1): (6, 13, 50, 3, 24, 18, 13, 19),
    (112, 5): (14, 17, 50, 3, 5, 4, 2, 5),
    (112, 4): (11, 21, 49, 3, 9, 6, 4, 8),
    (112, 3): (11, 23, 47, 3, 16, 8, 6, 9),
    (112, 2): (11, 21, 49, 3, 23, 12, 9, 14),
    (128, 5): (12, 19, 62, 3, 5, 3, 2, 4),
    (128, 4): (11, 21, 61, 3, 11, 6, 5, 7),
    (128, 3): (11, 22, 60, 3, 16, 9, 6, 10),
    (128, 2): (11, 21, 61, 3, 22, 12, 9, 14),
    (128, 1): (11, 20, 62, 3, 24, 17, 13, 19),
    (160, 5): (11, 19, 87, 3, 5, 4, 2, 4),
    (160, 4): (11, 23, 83, 3, 11, 6, 5, 9),
    (160, 3): (11, 24, 82, 3, 16, 8, 6, 11),
    (160, 2): (11, 21, 85, 3, 22, 11, 9, 13),
    (160, 1): (11, 22, 84, 3, 24, 18, 12, 19),
    (192, 5): (11, 20, 110, 3, 6, 4, 2, 5),
    (192, 4): (11, 22, 108, 3, 10, 6, 4, 9),
    (192, 3): (11, 24, 106, 3, 16, 10, 6, 11),
    (192, 2): (11, 20, 110, 3, 22, 13, 9, 13),
    (192, 1): (11, 21, 109, 3, 24, 20, 13, 24),
    (224, 5): (12, 22, 131, 3, 8, 6, 2, 6),
    (224, 4): (12, 26, 127, 3, 12, 8, 4, 11),
    (224, 3): (11, 20, 134, 3, 16, 10, 7, 9),
    (224, 2): (11, 22, 132, 3, 24, 16, 10, 15),
    (224, 1): (11, 24, 130, 3, 24, 20, 12, 20),
    (256, 5): (11, 24, 154, 3, 6, 5, 2, 5),
    (256, 4): (11, 24, 154, 3, 12, 9, 5, 10),
    (256, 3): (11, 27, 151, 3, 16, 10, 7, 10),
    (256, 2): (11, 22, 156, 3, 24, 14, 10, 13),
    (256, 1): (11, 26, 152, 3, 24, 19, 14, 18),
    (320, 5): (11, 26, 200, 3, 8, 5, 2, 6),
    (320, 4): (11, 25, 201, 3, 13, 9, 5, 10),
    (320, 2): (11, 26, 200, 3, 24, 17, 9, 17),
    (384, 5): (11, 27, 247, 3, 8, 6, 2, 7),
    (384, 3): (11, 24, 250, 3, 16, 9, 7, 10),
    (384, 1): (12, 28, 245, 3, 24, 20, 14, 23),
}
# fmt: on


def uep_segments(subchannel: Subchannel) -> list[tuple[int, int]]:
    """The puncturing of a UEP subchannel as (L, 0-based pattern index) pairs, see
    `dsp.eep.decode_punctured_logical_frame`."""
    profile = subchannel.uep_profile
    row = _UEP_PUNCTURING[(profile.bitrate, profile.protection_level)]
    blocks, pattern_numbers = row[:4], row[4:]
    return [
        (num_blocks, number - 1)
        for num_blocks, number in zip(blocks, pattern_numbers, strict=True)
        if num_blocks
    ]


def uep_padding_bits(subchannel: Subchannel) -> int:
    """Number of padding bits at the end of the codeword: what remains of the
    subchannel after the punctured segments and the tail (0, 4 or 8)."""
    return subchannel.size * 64 - punctured_length(uep_segments(subchannel))


def decode_uep_logical_frame(coded_bits: np.ndarray, subchannel: Subchannel) -> np.ndarray:
    """Decodes a UEP-protected MSC logical frame codeword (the time-deinterleaved bits
    of one CIF share) to its `24 * bitrate` information bits (ETSI EN 300 401, clause
    11.3.1)."""
    if not subchannel.is_uep:
        raise ValueError("decode_uep_logical_frame is only defined for UEP subchannels.")
    return decode_punctured_logical_frame(
        coded_bits, uep_segments(subchannel), padding_bits=uep_padding_bits(subchannel)
    )
