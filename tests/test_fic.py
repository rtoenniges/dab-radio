import numpy as np

from dab.crc import crc16_ccitt
from dsp.convolutional import soft_bits
from dsp.deinterleave import frequency_deinterleave_index
from dsp.fic import (
    FIC_CODEWORD_BITS,
    FIC_INFO_BITS,
    check_fib_crc,
    decode_fic_logical_frame,
    encode_fic_logical_frame,
    extract_fibs_from_symbols,
    split_fibs,
)


def test_encode_fic_logical_frame_has_correct_length():
    rng = np.random.default_rng(0)
    info_bits = rng.integers(0, 2, size=FIC_INFO_BITS).astype(np.uint8)

    coded = encode_fic_logical_frame(info_bits)

    assert len(coded) == FIC_CODEWORD_BITS


def test_decode_fic_logical_frame_recovers_bits_without_noise():
    rng = np.random.default_rng(1)
    info_bits = rng.integers(0, 2, size=FIC_INFO_BITS).astype(np.uint8)

    coded = encode_fic_logical_frame(info_bits)
    decoded = decode_fic_logical_frame(soft_bits(coded))

    np.testing.assert_array_equal(decoded, info_bits)


def test_decode_fic_logical_frame_tolerates_bit_errors():
    rng = np.random.default_rng(2)
    info_bits = rng.integers(0, 2, size=FIC_INFO_BITS).astype(np.uint8)

    coded = encode_fic_logical_frame(info_bits).copy()
    # Sprinkle in a few scattered bit errors (the convolutional code should fix them).
    error_positions = rng.choice(len(coded), size=5, replace=False)
    coded[error_positions] ^= 1

    decoded = decode_fic_logical_frame(soft_bits(coded))

    np.testing.assert_array_equal(decoded, info_bits)


def test_check_fib_crc_accepts_valid_fib_and_rejects_corrupted_one():
    rng = np.random.default_rng(3)
    data_bits = rng.integers(0, 2, size=240).astype(np.uint8)
    crc = crc16_ccitt(np.packbits(data_bits).tobytes()) ^ 0xFFFF
    crc_bits = np.array([(crc >> shift) & 1 for shift in range(15, -1, -1)], dtype=np.uint8)
    fib_bits = np.concatenate([data_bits, crc_bits])

    assert check_fib_crc(fib_bits) is True

    corrupted = fib_bits.copy()
    corrupted[0] ^= 1
    assert check_fib_crc(corrupted) is False


def test_split_fibs_returns_three_blocks_of_256_bits():
    info_bits = np.arange(FIC_INFO_BITS, dtype=np.uint8) % 2

    fibs = split_fibs(info_bits)

    assert len(fibs) == 3
    assert all(len(fib) == 256 for fib in fibs)
    np.testing.assert_array_equal(np.concatenate(fibs), info_bits)


def _logical_symbols_from_bits(bits: np.ndarray) -> np.ndarray:
    """Inverts `dsp.qpsk.demap_soft_bits`: rebuilds K complex QPSK symbols from 2*K bits
    (for the synthetic frame round-trip test below)."""
    k = len(bits) // 2
    real = (1 - 2 * bits[:k].astype(np.float64)) / np.sqrt(2)
    imag = (1 - 2 * bits[k:].astype(np.float64)) / np.sqrt(2)
    return real + 1j * imag


def test_extract_fibs_from_symbols_recovers_fibs_from_synthetic_frame():
    rng = np.random.default_rng(4)
    num_carriers = 1536
    fft_length = 2048
    index = frequency_deinterleave_index(num_carriers, fft_length)

    info_bits_per_frame = [
        rng.integers(0, 2, size=FIC_INFO_BITS).astype(np.uint8) for _ in range(4)
    ]
    bitstream = np.concatenate([encode_fic_logical_frame(bits) for bits in info_bits_per_frame])
    assert len(bitstream) == 3 * 3072

    fic_symbols = []
    for i in range(3):
        chunk = bitstream[i * 3072 : (i + 1) * 3072]
        logical = _logical_symbols_from_bits(chunk)
        physical = np.empty(num_carriers, dtype=complex)
        physical[index] = logical
        fic_symbols.append(physical)

    fibs = extract_fibs_from_symbols(fic_symbols, num_carriers, fft_length)

    expected_fibs = [fib for bits in info_bits_per_frame for fib in split_fibs(bits)]
    assert len(fibs) == len(expected_fibs) == 12
    for actual, expected in zip(fibs, expected_fibs, strict=True):
        np.testing.assert_array_equal(actual, expected)
