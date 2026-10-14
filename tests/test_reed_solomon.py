import numpy as np
import pytest

from dsp.reed_solomon import (
    CODEWORD_BYTES,
    DATA_BYTES,
    PACKET_CODEWORD_BYTES,
    PACKET_DATA_BYTES,
    PACKET_NROOTS,
    decode_packet_rs_codeword,
    decode_rs_codeword,
)
from tests.rs_test_utils import encode_rs_codeword


def test_generator_polynomial_produces_codeword_with_zero_syndromes():
    """An (error-free) codeword formed by `encode_rs_codeword` must be recognised by
    the decoder as already correct (0 corrections), which confirms that the Galois
    field tables, the generator polynomial construction and the decoder syndromes are
    consistent with one another."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=DATA_BYTES, dtype=np.uint8)

    codeword = encode_rs_codeword(data)
    corrected, num_corrections = decode_rs_codeword(codeword)

    assert num_corrections == 0
    np.testing.assert_array_equal(corrected, codeword)


@pytest.mark.parametrize("num_errors", [1, 2, 3, 4, 5])
def test_decode_rs_codeword_corrects_up_to_five_byte_errors(num_errors):
    rng = np.random.default_rng(num_errors)
    data = rng.integers(0, 256, size=DATA_BYTES, dtype=np.uint8)
    codeword = encode_rs_codeword(data)

    corrupted = codeword.copy()
    error_positions = rng.choice(CODEWORD_BYTES, size=num_errors, replace=False)
    for pos in error_positions:
        corrupted[pos] ^= rng.integers(1, 256, dtype=np.uint8)

    corrected, num_corrections = decode_rs_codeword(corrupted)

    assert num_corrections == num_errors
    np.testing.assert_array_equal(corrected, codeword)
    np.testing.assert_array_equal(corrected[:DATA_BYTES], data)


def test_decode_rs_codeword_reports_uncorrectable_for_six_errors():
    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, size=DATA_BYTES, dtype=np.uint8)
    codeword = encode_rs_codeword(data)

    corrupted = codeword.copy()
    error_positions = rng.choice(CODEWORD_BYTES, size=6, replace=False)
    for pos in error_positions:
        corrupted[pos] ^= rng.integers(1, 256, dtype=np.uint8)

    _corrected, num_corrections = decode_rs_codeword(corrupted)

    # With more errors than the code can correct, the decoder MUST either report a
    # failure (num_corrections == -1) or, in the (with random errors rare) case of
    # confusion with another valid codeword, a wrong result. Since the two are
    # practically indistinguishable, only the absence of a false "corrected
    # successfully, with a wrong result" is checked here.
    if num_corrections != -1:
        pytest.fail(
            f"6 errors were wrongly reported as correctable ({num_corrections} corrections)"
        )


def test_decode_rs_codeword_raises_for_wrong_length():
    with pytest.raises(ValueError, match="120 bytes"):
        decode_rs_codeword(np.zeros(10, dtype=np.uint8))


@pytest.mark.parametrize("num_errors", [0, 1, 4, 8])
def test_decode_packet_rs_codeword_corrects_up_to_eight_byte_errors(num_errors):
    """RS(204,188) of the packet mode FEC (EN 300 401, clause 5.3.5.1): t = 8."""
    rng = np.random.default_rng(100 + num_errors)
    data = rng.integers(0, 256, size=PACKET_DATA_BYTES, dtype=np.uint8)
    codeword = encode_rs_codeword(data, nroots=PACKET_NROOTS)
    assert len(codeword) == PACKET_CODEWORD_BYTES

    corrupted = codeword.copy()
    for pos in rng.choice(PACKET_CODEWORD_BYTES, size=num_errors, replace=False):
        corrupted[pos] ^= rng.integers(1, 256, dtype=np.uint8)

    corrected, num_corrections = decode_packet_rs_codeword(corrupted)

    assert num_corrections == num_errors
    np.testing.assert_array_equal(corrected, codeword)


def test_decode_packet_rs_codeword_reports_nine_errors_as_uncorrectable():
    rng = np.random.default_rng(7)
    data = rng.integers(0, 256, size=PACKET_DATA_BYTES, dtype=np.uint8)
    corrupted = encode_rs_codeword(data, nroots=PACKET_NROOTS)
    for pos in rng.choice(PACKET_CODEWORD_BYTES, size=9, replace=False):
        corrupted[pos] ^= rng.integers(1, 256, dtype=np.uint8)

    _corrected, num_corrections = decode_packet_rs_codeword(corrupted)

    assert num_corrections == -1


def test_decode_packet_rs_codeword_raises_for_wrong_length():
    with pytest.raises(ValueError, match="204 bytes"):
        decode_packet_rs_codeword(np.zeros(120, dtype=np.uint8))
