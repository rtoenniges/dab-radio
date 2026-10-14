"""Fast Information Channel: from the three FIC symbols of a frame to CRC-checked FIBs."""

from __future__ import annotations

import numpy as np

from dab.crc import check_crc
from dsp.convolutional import (
    TAIL_PUNCTURE_PATTERN,
    depuncture,
    encode_mother_code,
    puncture,
    viterbi_decode,
)
from dsp.deinterleave import deinterleave_frequency, frequency_deinterleave_index
from dsp.qpsk import demap_soft_bits
from dsp.scrambling import apply_energy_dispersal

# ETSI EN 300 401, clause 11.2 (coding in the Fast Information Channel, transmission
# mode I): fixed puncturing, independent of the MSC configuration.
FIC_INFO_BITS = 768  # I
FIC_CODEWORD_BITS = 2304  # length of the punctured codeword per logical frame

_BLOCK_BITS = 128
_NUM_BLOCKS_PI16 = 21
_NUM_BLOCKS_PI15 = 3

_PI16 = np.array([int(b) for b in "1110" * 8], dtype=np.uint8)  # table 13, PI=16
_PI15 = np.array([int(b) for b in "1110" * 7 + "1100"], dtype=np.uint8)  # table 13, PI=15


def encode_fic_logical_frame(info_bits: np.ndarray) -> np.ndarray:
    """Encodes 768 FIC information bits with the fixed FIC puncturing (the transmit
    side, used by the tests' synthetic transmitter)."""
    if len(info_bits) != FIC_INFO_BITS:
        raise ValueError(f"A FIC logical frame must be {FIC_INFO_BITS} bits long.")

    scrambled = apply_energy_dispersal(info_bits)
    mother = encode_mother_code(scrambled)
    block_part = mother[: 4 * FIC_INFO_BITS]
    tail_part = mother[4 * FIC_INFO_BITS :]

    pi16_part = block_part[: _NUM_BLOCKS_PI16 * _BLOCK_BITS]
    pi15_part = block_part[_NUM_BLOCKS_PI16 * _BLOCK_BITS :]

    return np.concatenate(
        [
            puncture(pi16_part, _PI16),
            puncture(pi15_part, _PI15),
            puncture(tail_part, TAIL_PUNCTURE_PATTERN),
        ]
    )


def decode_fic_logical_frame(coded_bits: np.ndarray) -> np.ndarray:
    """Decodes a FIC codeword (2 304 soft bits, see `dsp.convolutional.viterbi_decode`)
    back to 768 information bits."""
    if len(coded_bits) != FIC_CODEWORD_BITS:
        raise ValueError(f"A FIC codeword must be {FIC_CODEWORD_BITS} bits long.")

    n_pi16 = _NUM_BLOCKS_PI16 * _BLOCK_BITS // len(_PI16) * int(_PI16.sum())
    n_pi15 = _NUM_BLOCKS_PI15 * _BLOCK_BITS // len(_PI15) * int(_PI15.sum())

    pi16_coded = coded_bits[:n_pi16]
    pi15_coded = coded_bits[n_pi16 : n_pi16 + n_pi15]
    tail_coded = coded_bits[n_pi16 + n_pi15 :]

    received = np.concatenate(
        [
            depuncture(pi16_coded, _PI16),
            depuncture(pi15_coded, _PI15),
            depuncture(tail_coded, TAIL_PUNCTURE_PATTERN),
        ]
    )

    scrambled = viterbi_decode(received, FIC_INFO_BITS)
    return apply_energy_dispersal(scrambled)


def assemble_fic_bitstream(symbol_bits: list[np.ndarray]) -> np.ndarray:
    """Joins the demapped (soft) bit vectors of the three FIC OFDM symbols (l=2,3,4,
    3072 bits each) into the multiplex bit stream B' (ETSI 14.4.1)."""
    return np.concatenate(symbol_bits)


def split_fic_codewords(bitstream: np.ndarray) -> list[np.ndarray]:
    """Splits the 9 216-bit FIC multiplex bit stream into the 4 convolutional codewords
    (2 304 bits each) of CIF counts 0..3 (ETSI 14.4.1)."""
    return [bitstream[i * FIC_CODEWORD_BITS : (i + 1) * FIC_CODEWORD_BITS] for i in range(4)]


def split_fibs(info_bits: np.ndarray) -> list[np.ndarray]:
    """Splits 768 decoded FIC information bits into the 3 FIBs they contain (256 bits
    each, ETSI 5.2.1)."""
    return [info_bits[i * 256 : (i + 1) * 256] for i in range(3)]


def extract_fibs_from_symbols(
    fic_symbols: list[np.ndarray], num_carriers: int, fft_length: int
) -> list[np.ndarray]:
    """Decodes the 3 FIC OFDM symbols (l=2,3,4) of a frame to the 12 FIBs they contain
    (ETSI EN 300 401, clause 14.4.1).

    `fic_symbols` must be the already differentially demodulated D-QPSK constellation
    points of these three symbols in physical carrier order (as delivered by
    `dsp.ofdm.extract_used_carriers`), `num_carriers` points each.
    """
    if len(fic_symbols) != 3:
        raise ValueError("FIC decoding needs exactly 3 OFDM symbols (l=2,3,4).")

    index = frequency_deinterleave_index(num_carriers, fft_length)
    symbol_bits = [demap_soft_bits(deinterleave_frequency(symbol, index)) for symbol in fic_symbols]
    bitstream = assemble_fic_bitstream(symbol_bits)

    fibs: list[np.ndarray] = []
    for codeword in split_fic_codewords(bitstream):
        info_bits = decode_fic_logical_frame(codeword)
        fibs.extend(split_fibs(info_bits))
    return fibs


def check_fib_crc(fib_bits: np.ndarray) -> bool:
    """Checks a 256-bit FIB (240 data bits + 16 CRC bits) for correctness."""
    if len(fib_bits) != 256:
        raise ValueError("A FIB must be 256 bits long.")
    return check_crc(np.packbits(fib_bits).tobytes())
