import numpy as np
import pytest

from dsp.deinterleave import frequency_deinterleave_index
from dsp.msc import (
    CIF_BITS,
    NUM_CIFS_PER_FRAME,
    NUM_MSC_SYMBOLS_PER_FRAME,
    assemble_msc_cifs,
    extract_subchannel_bits,
)
from dsp.qpsk import hard_decisions


def _logical_symbols_from_bits(bits: np.ndarray) -> np.ndarray:
    """Inverts `dsp.qpsk.demap_soft_bits`: rebuilds K complex QPSK symbols from 2*K bits
    (for the synthetic frame round-trip test below)."""
    k = len(bits) // 2
    real = (1 - 2 * bits[:k].astype(np.float64)) / np.sqrt(2)
    imag = (1 - 2 * bits[k:].astype(np.float64)) / np.sqrt(2)
    return real + 1j * imag


def test_assemble_msc_cifs_recovers_known_bits_from_synthetic_symbols():
    rng = np.random.default_rng(9)
    num_carriers = 1536
    fft_length = 2048
    index = frequency_deinterleave_index(num_carriers, fft_length)

    total_bits = NUM_CIFS_PER_FRAME * CIF_BITS
    assert total_bits == NUM_MSC_SYMBOLS_PER_FRAME * 2 * num_carriers
    info_bits = rng.integers(0, 2, size=total_bits).astype(np.uint8)

    msc_symbols = []
    for i in range(NUM_MSC_SYMBOLS_PER_FRAME):
        chunk = info_bits[i * 2 * num_carriers : (i + 1) * 2 * num_carriers]
        logical = _logical_symbols_from_bits(chunk)
        physical = np.empty(num_carriers, dtype=complex)
        physical[index] = logical
        msc_symbols.append(physical)

    cifs = assemble_msc_cifs(msc_symbols, num_carriers, fft_length)

    assert len(cifs) == NUM_CIFS_PER_FRAME
    assert all(len(cif) == CIF_BITS for cif in cifs)
    np.testing.assert_array_equal(hard_decisions(np.concatenate(cifs)), info_bits)


def test_assemble_msc_cifs_raises_for_wrong_symbol_count():
    with pytest.raises(ValueError, match="72 OFDM symbols"):
        assemble_msc_cifs([np.zeros(1536, dtype=complex)], num_carriers=1536, fft_length=2048)


def test_extract_subchannel_bits_returns_correct_slice():
    cif_bits = np.arange(CIF_BITS, dtype=np.uint8) % 2

    subchannel_bits = extract_subchannel_bits(cif_bits, start_address=10, size=3)

    assert len(subchannel_bits) == 3 * 64
    np.testing.assert_array_equal(subchannel_bits, cif_bits[10 * 64 : 13 * 64])


def test_extract_subchannel_bits_raises_when_out_of_range():
    cif_bits = np.zeros(CIF_BITS, dtype=np.uint8)

    with pytest.raises(ValueError, match="outside the CIF"):
        extract_subchannel_bits(cif_bits, start_address=860, size=10)
