import numpy as np
import pytest

from dab.fig import Subchannel
from dsp.convolutional import TAIL_PUNCTURE_PATTERN, encode_mother_code, puncture, soft_bits
from dsp.eep import (
    _P_CODES,
    _eep_a_puncturing,
    _eep_b_puncturing,
    compute_eep_bitrate,
    decode_eep_logical_frame,
)
from dsp.scrambling import apply_energy_dispersal


def test_compute_eep_bitrate_matches_known_live_configurations():
    # Subchannel configurations observed live on real hardware (see the FIG 0/1
    # validation): plausible DAB+ bit rates common in practice confirm the formula
    # independently of the puncturing tables.
    asa_de = Subchannel(
        subchannel_id=1, start_address=0, size=32, is_uep=False, protection_index=1, eep_option=0
    )
    assert compute_eep_bitrate(asa_de) == 32  # EEP-A level 2

    dlf = Subchannel(
        subchannel_id=10,
        start_address=512,
        size=104,
        is_uep=False,
        protection_index=1,
        eep_option=0,
    )
    assert compute_eep_bitrate(dlf) == 104  # EEP-A level 2

    schlagerparadies = Subchannel(
        subchannel_id=20, start_address=320, size=42, is_uep=False, protection_index=1, eep_option=1
    )
    assert compute_eep_bitrate(schlagerparadies) == 64  # EEP-B level 2


@pytest.mark.parametrize(
    "eep_option,protection_index,size",
    [
        (0, 1, 32),  # EEP-A level 2, e.g. "ASA DE"
        (0, 2, 54),  # EEP-A level 3, e.g. "Absolut relax"
        (0, 1, 104),  # EEP-A level 2, e.g. "Dlf"
        (1, 1, 42),  # EEP-B level 2, e.g. "SCHLAGERPARADIES"
    ],
)
def test_decode_eep_logical_frame_matches_declared_subchannel_size(
    eep_option, protection_index, size
):
    """For subchannel configurations observed live (see the FIG 0/1 validation) the
    coded bit count computed from bit rate and level must match the subchannel size
    declared in FIG 0/1 exactly, a strong indication that the formulas (bit rate,
    L1/L2, puncturing patterns) fit together. If the coded bit count did not match,
    decode_eep_logical_frame would raise a ValueError."""
    subchannel = Subchannel(
        subchannel_id=1,
        start_address=0,
        size=size,
        is_uep=False,
        protection_index=protection_index,
        eep_option=eep_option,
    )
    coded_bits = np.zeros(size * 64, dtype=np.float32)

    decoded = decode_eep_logical_frame(coded_bits, subchannel)

    assert len(decoded) == 24 * compute_eep_bitrate(subchannel)


def _encode_eep_logical_frame(info_bits: np.ndarray, subchannel: Subchannel) -> np.ndarray:
    """Mirror image of `decode_eep_logical_frame` for the round-trip test below:
    scrambles, convolutionally encodes and punctures `info_bits` according to the
    protection profile of `subchannel`."""
    bitrate = compute_eep_bitrate(subchannel)
    puncturing = _eep_a_puncturing if subchannel.eep_option == 0 else _eep_b_puncturing
    l1, l2, pi1_index, pi2_index = puncturing(subchannel.protection_index, bitrate)
    pi1 = np.tile(_P_CODES[pi1_index], 4)
    pi2 = np.tile(_P_CODES[pi2_index], 4)

    scrambled = apply_energy_dispersal(info_bits)
    mother = encode_mother_code(scrambled)
    n1_mother, n2_mother = l1 * 128, l2 * 128
    pi1_part = mother[:n1_mother]
    pi2_part = mother[n1_mother : n1_mother + n2_mother]
    tail_part = mother[n1_mother + n2_mother :]

    return np.concatenate(
        [
            puncture(pi1_part, pi1),
            puncture(pi2_part, pi2),
            puncture(tail_part, TAIL_PUNCTURE_PATTERN),
        ]
    )


def test_decode_eep_logical_frame_recovers_bits_without_noise():
    subchannel = Subchannel(
        subchannel_id=1, start_address=0, size=32, is_uep=False, protection_index=1, eep_option=0
    )
    bitrate = compute_eep_bitrate(subchannel)
    rng = np.random.default_rng(1)
    info_bits = rng.integers(0, 2, size=24 * bitrate).astype(np.uint8)

    coded = _encode_eep_logical_frame(info_bits, subchannel)
    decoded = decode_eep_logical_frame(soft_bits(coded), subchannel)

    np.testing.assert_array_equal(decoded, info_bits)


def test_decode_eep_logical_frame_tolerates_bit_errors():
    subchannel = Subchannel(
        subchannel_id=1, start_address=0, size=32, is_uep=False, protection_index=1, eep_option=0
    )
    bitrate = compute_eep_bitrate(subchannel)
    rng = np.random.default_rng(2)
    info_bits = rng.integers(0, 2, size=24 * bitrate).astype(np.uint8)

    coded = _encode_eep_logical_frame(info_bits, subchannel).copy()
    error_positions = rng.choice(len(coded), size=5, replace=False)
    coded[error_positions] ^= 1

    decoded = decode_eep_logical_frame(soft_bits(coded), subchannel)

    np.testing.assert_array_equal(decoded, info_bits)


def test_decode_eep_logical_frame_raises_for_wrong_length():
    subchannel = Subchannel(
        subchannel_id=1, start_address=0, size=32, is_uep=False, protection_index=1, eep_option=0
    )

    with pytest.raises(ValueError, match="Expected"):
        decode_eep_logical_frame(np.zeros(10, dtype=np.uint8), subchannel)
