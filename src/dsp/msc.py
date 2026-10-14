"""Main Service Channel: splitting the MSC symbols into CIFs and subchannels."""

from __future__ import annotations

import numpy as np

from dsp.deinterleave import deinterleave_frequency, frequency_deinterleave_index
from dsp.qpsk import demap_soft_bits

# ETSI EN 300 401, clause 5.3.1: a CIF (Common Interleaved Frame) has 864 capacity
# units (CU) of 64 bits each.
CU_BITS = 64
CIF_BITS = 864 * CU_BITS
NUM_CIFS_PER_FRAME = 4
NUM_MSC_SYMBOLS_PER_FRAME = 72  # OFDM symbols l=5..76 (transmission mode I)


def assemble_msc_cifs(
    msc_symbols: list[np.ndarray], num_carriers: int, fft_length: int
) -> list[np.ndarray]:
    """Demaps the 72 MSC OFDM symbols of a frame (l=5..76) into the 4 CIFs they carry
    (ETSI EN 300 401, clauses 5.3.1 and 14.4).

    The MSC counterpart of `dsp.fic.extract_fibs_from_symbols`: `msc_symbols` must be
    the already differentially demodulated D-QPSK constellation points of these 72
    symbols in physical carrier order (as delivered by
    `dsp.ofdm.extract_used_carriers`), `num_carriers` points each. The CIFs hold soft
    bits (see `dsp.qpsk.demap_soft_bits`).
    """
    if len(msc_symbols) != NUM_MSC_SYMBOLS_PER_FRAME:
        raise ValueError(
            f"MSC decoding needs exactly {NUM_MSC_SYMBOLS_PER_FRAME} OFDM symbols (l=5..76)."
        )

    index = frequency_deinterleave_index(num_carriers, fft_length)
    symbol_bits = [demap_soft_bits(deinterleave_frequency(symbol, index)) for symbol in msc_symbols]
    bitstream = np.concatenate(symbol_bits)

    return [bitstream[i * CIF_BITS : (i + 1) * CIF_BITS] for i in range(NUM_CIFS_PER_FRAME)]


def extract_subchannel_bits(cif_bits: np.ndarray, start_address: int, size: int) -> np.ndarray:
    """Extracts the capacity units (CU) assigned to a subchannel from a CIF (ETSI EN
    300 401, clause 6.2.1: consecutive CU addressing within a CIF).

    The returned bits are still time-interleaved (ETSI clause 11.2), convolutionally
    coded (clause 11.1) and scrambled (clause 10); all of that has to be undone before
    the actual audio decoding.
    """
    start_bit = start_address * CU_BITS
    end_bit = (start_address + size) * CU_BITS
    if end_bit > len(cif_bits):
        raise ValueError("Subchannel range lies outside the CIF.")
    return cif_bits[start_bit:end_bit]
