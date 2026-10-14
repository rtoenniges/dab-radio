import numpy as np

from dab.fig import Subchannel
from dsp.convolutional import TAIL_PUNCTURE_PATTERN, encode_mother_code, puncture
from dsp.deinterleave import frequency_deinterleave_index
from dsp.eep import _P_CODES, _eep_a_puncturing, compute_eep_bitrate, decode_eep_logical_frame
from dsp.msc import (
    CIF_BITS,
    NUM_CIFS_PER_FRAME,
    NUM_MSC_SYMBOLS_PER_FRAME,
    assemble_msc_cifs,
    extract_subchannel_bits,
)
from dsp.scrambling import apply_energy_dispersal
from dsp.time_interleave import _INTERLEAVE_MAP, INTERLEAVE_DEPTH, TimeDeinterleaver


def _encode_eep_logical_frame(info_bits: np.ndarray, subchannel: Subchannel) -> np.ndarray:
    bitrate = compute_eep_bitrate(subchannel)
    l1, l2, pi1_index, pi2_index = _eep_a_puncturing(subchannel.protection_index, bitrate)
    pi1 = np.tile(_P_CODES[pi1_index], 4)
    pi2 = np.tile(_P_CODES[pi2_index], 4)
    scrambled = apply_energy_dispersal(info_bits)
    mother = encode_mother_code(scrambled)
    n1_mother, n2_mother = l1 * 128, l2 * 128
    return np.concatenate(
        [
            puncture(mother[:n1_mother], pi1),
            puncture(mother[n1_mother : n1_mother + n2_mother], pi2),
            puncture(mother[n1_mother + n2_mother :], TAIL_PUNCTURE_PATTERN),
        ]
    )


def _bits_to_qpsk_symbols(cif_bits_arrays: list[np.ndarray], index: np.ndarray) -> list[np.ndarray]:
    """Inverts `assemble_msc_cifs`: builds the 72 physically ordered QPSK symbols
    (l=5..76) of a frame from its 4 CIF bit arrays."""
    full_bitstream = np.concatenate(cif_bits_arrays)
    bits_per_symbol = 2 * 1536
    symbols = []
    for s in range(NUM_MSC_SYMBOLS_PER_FRAME):
        chunk = full_bitstream[s * bits_per_symbol : (s + 1) * bits_per_symbol]
        k = len(chunk) // 2
        real = (1 - 2 * chunk[:k].astype(np.float64)) / np.sqrt(2)
        imag = (1 - 2 * chunk[k:].astype(np.float64)) / np.sqrt(2)
        logical = real + 1j * imag
        physical = np.empty(1536, dtype=complex)
        physical[index] = logical
        symbols.append(physical)
    return symbols


def test_full_msc_chain_recovers_info_bits_across_layers():
    """Round-trip test across ALL MSC decoding layers together (time interleaving,
    CIF placement, QPSK mapping/frequency interleaving, EEP decoding), unlike
    test_eep.py, test_msc.py and test_time_interleave.py, which check each layer
    INDIVIDUALLY (each with its own synthetic encoder/decoder on BOTH sides). A
    mistake at the INTERFACE between two layers (e.g. a wrong delay direction in the
    time interleaving) would be invisible to any of the individual tests but would
    show up here.
    """
    rng = np.random.default_rng(123)
    subchannel = Subchannel(
        subchannel_id=12,
        start_address=720,
        size=104,
        is_uep=False,
        protection_index=1,
        eep_option=0,
    )
    bitrate = compute_eep_bitrate(subchannel)
    fragment_size = subchannel.size * 64
    num_info_bits = 24 * bitrate

    n_logical_frames = 24  # >= 16 (deinterleaver depth) + a few outputs to check
    info_bits_seq = [
        rng.integers(0, 2, size=num_info_bits).astype(np.uint8) for _ in range(n_logical_frames)
    ]
    coded = [_encode_eep_logical_frame(b, subchannel) for b in info_bits_seq]
    assert all(len(c) == fragment_size for c in coded)

    # Encoder-side time interleaving: Tx[T][i] = coded[T - delay[i]][i]
    # (delay[i] = _INTERLEAVE_MAP[i%16], see the TimeDeinterleaver docstring and
    # test_time_deinterleaver_applies_known_per_position_delays: position i is delayed
    # by _INTERLEAVE_MAP[i%16] CIFs when SENDING, so that the receiver gets back the
    # data from `16 - _INTERLEAVE_MAP[i%16]` CIFs ago at position i on `push()`).
    delay = np.array([_INTERLEAVE_MAP[i % INTERLEAVE_DEPTH] for i in range(fragment_size)])
    tx = []
    for t in range(n_logical_frames):
        frame = np.zeros(fragment_size, dtype=np.uint8)
        for i in range(fragment_size):
            src = t - delay[i]
            if src >= 0:
                frame[i] = coded[src][i]
        tx.append(frame)

    index = frequency_deinterleave_index(1536, 2048)
    deinterleaver = TimeDeinterleaver(fragment_size)
    checked = 0
    for frame_no in range(n_logical_frames // NUM_CIFS_PER_FRAME):
        full_cifs = []
        for c in range(NUM_CIFS_PER_FRAME):
            t = frame_no * NUM_CIFS_PER_FRAME + c
            full_cif = rng.integers(0, 2, size=CIF_BITS).astype(
                np.uint8
            )  # filler bits of other subchannels
            full_cif[
                subchannel.start_address * 64 : (subchannel.start_address + subchannel.size) * 64
            ] = tx[t]
            full_cifs.append(full_cif)

        msc_symbols = _bits_to_qpsk_symbols(full_cifs, index)
        cifs_out = assemble_msc_cifs(msc_symbols, 1536, 2048)

        for c in range(NUM_CIFS_PER_FRAME):
            t = frame_no * NUM_CIFS_PER_FRAME + c
            coded_bits = extract_subchannel_bits(
                cifs_out[c], subchannel.start_address, subchannel.size
            )
            deinterleaved = deinterleaver.push(coded_bits)
            if t >= INTERLEAVE_DEPTH:
                decoded = decode_eep_logical_frame(deinterleaved, subchannel)
                np.testing.assert_array_equal(decoded, info_bits_seq[t - INTERLEAVE_DEPTH])
                checked += 1

    assert checked == n_logical_frames - INTERLEAVE_DEPTH
