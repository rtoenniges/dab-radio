"""Tests for the Unequal Error Protection of classic DAB subchannels."""

from __future__ import annotations

import numpy as np
import pytest

from dab.fig import UEP_TABLE, Subchannel
from dsp.convolutional import soft_bits
from dsp.uep import decode_uep_logical_frame, uep_padding_bits, uep_segments
from tests.mp2_test_utils import encode_uep_logical_frame


def _uep_subchannel(table_index: int) -> Subchannel:
    return Subchannel(
        subchannel_id=1,
        start_address=0,
        size=UEP_TABLE[table_index].size,
        is_uep=True,
        protection_index=table_index,
    )


@pytest.mark.parametrize("table_index", range(len(UEP_TABLE)))
def test_every_uep_profile_fills_its_subchannel(table_index):
    """For all 64 table entries the punctured codeword must fit the subchannel size
    from the table exactly, up to the few padding bits the standard provides (0, 4 or
    8). Size table and puncturing table come from different places in the standard; a
    typo in either one shows up here as a mismatch."""
    subchannel = _uep_subchannel(table_index)
    profile = UEP_TABLE[table_index]

    blocks = sum(num_blocks for num_blocks, _ in uep_segments(subchannel))
    assert blocks * 32 == 24 * profile.bitrate
    assert uep_padding_bits(subchannel) in (0, 4, 8)

    decoded = decode_uep_logical_frame(np.zeros(profile.size * 64, dtype=np.float32), subchannel)
    assert len(decoded) == 24 * profile.bitrate


def test_80_kbits_level_1_uses_the_corrected_puncturing():
    """welle.io has PI2 = 7 at this one entry; ODR-DabMod, the transmitter side, has 17.
    With 7 the codeword would leave 404 of the 5376 subchannel bits unused."""
    table_index = 23
    assert UEP_TABLE[table_index] == (84, 1, 80)

    segments = uep_segments(_uep_subchannel(table_index))

    assert segments[1] == (10, 17 - 1)


@pytest.mark.parametrize(
    "table_index",
    [
        0,  # 32 kbit/s, level 5: only three segments (L4 = 0)
        23,  # 80 kbit/s, level 1: the corrected entry
        35,  # 128 kbit/s, level 3: a common profile for MP2 services, 4 padding bits
        46,  # 192 kbit/s, level 2: 8 padding bits
    ],
)
def test_decode_uep_logical_frame_recovers_bits(table_index):
    subchannel = _uep_subchannel(table_index)
    rng = np.random.default_rng(table_index)
    info_bits = rng.integers(0, 2, size=24 * subchannel.bitrate).astype(np.uint8)

    coded = encode_uep_logical_frame(info_bits, subchannel)
    assert len(coded) == subchannel.size * 64

    np.testing.assert_array_equal(decode_uep_logical_frame(soft_bits(coded), subchannel), info_bits)


def test_decode_uep_logical_frame_tolerates_bit_errors():
    subchannel = _uep_subchannel(35)
    rng = np.random.default_rng(7)
    info_bits = rng.integers(0, 2, size=24 * subchannel.bitrate).astype(np.uint8)

    coded = encode_uep_logical_frame(info_bits, subchannel).copy()
    coded[rng.choice(len(coded), size=20, replace=False)] ^= 1

    np.testing.assert_array_equal(decode_uep_logical_frame(soft_bits(coded), subchannel), info_bits)


def test_decode_uep_logical_frame_rejects_eep_subchannels():
    eep = Subchannel(
        subchannel_id=1, start_address=0, size=32, is_uep=False, protection_index=1, eep_option=0
    )
    with pytest.raises(ValueError, match="UEP"):
        decode_uep_logical_frame(np.zeros(32 * 64, dtype=np.uint8), eep)
