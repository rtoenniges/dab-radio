"""Synthetic DAB transmitter (mode I) for tests of the receive chain.

Builds complete transmission frames from FIGs: FIBs with CRC, FIC convolutional code,
frequency interleaving, QPSK, differential modulation against the phase reference
symbol and OFDM. The MSC carries random data.
"""

from __future__ import annotations

import numpy as np

from dab.constants import MODE_I
from dab.crc import crc16_ccitt
from dsp.deinterleave import frequency_deinterleave_index
from dsp.fic import FIC_INFO_BITS, encode_fic_logical_frame
from dsp.timing import generate_reference_symbol

SAMPLE_RATE = 2_048_000.0
FIBS_PER_FRAME = 12  # 4 CIFs of 3 FIBs each


def fig1_ensemble_label(eid: int, label: str) -> bytes:
    """FIG 1/0 (ensemble label), EBU Latin/ASCII."""
    payload = bytes([0x00]) + eid.to_bytes(2, "big") + label.encode("ascii").ljust(16) + b"\xff\x00"
    return bytes([(1 << 5) | len(payload)]) + payload


def fig1_service_label(sid: int, label: str) -> bytes:
    """FIG 1/1 (programme service label), EBU Latin/ASCII."""
    payload = bytes([0x01]) + sid.to_bytes(2, "big") + label.encode("ascii").ljust(16) + b"\xff\x00"
    return bytes([(1 << 5) | len(payload)]) + payload


def fig0_2_audio_services(services: list[tuple[int, int]]) -> bytes:
    """FIG 0/2 (service organisation): one DAB+ audio component (ASCTy 63) as the
    primary component per programme service (SId, SubChId)."""
    payload = bytes([0x02])
    for sid, subchannel_id in services:
        component = (63 << 8) | (subchannel_id << 2) | 0b10
        payload += sid.to_bytes(2, "big") + bytes([0x01]) + component.to_bytes(2, "big")
    return bytes([len(payload)]) + payload  # FIG type 0


def fig0_17_programme_types(programme_types: list[tuple[int, int]]) -> bytes:
    """FIG 0/17 (programme type): one entry (SId, PTy code) per service, static."""
    payload = bytes([0x11])
    for sid, code in programme_types:
        payload += sid.to_bytes(2, "big") + bytes([0x00, code])
    return bytes([len(payload)]) + payload  # FIG type 0


def fib(*figs: bytes) -> np.ndarray:
    """One FIB (256 bits) from the FIGs, padded with 0xFF and with a valid CRC."""
    data = b"".join(figs)
    if len(data) > 30:
        raise ValueError("The FIGs do not fit into one FIB.")
    payload = data.ljust(30, b"\xff")
    crc = crc16_ccitt(payload) ^ 0xFFFF
    return np.unpackbits(np.frombuffer(payload + crc.to_bytes(2, "big"), dtype=np.uint8))


def _ofdm_symbol(carriers: np.ndarray) -> np.ndarray:
    mode = MODE_I
    center = mode.fft_length // 2
    half = mode.num_carriers // 2
    spectrum = np.zeros(mode.fft_length, dtype=complex)
    spectrum[center - half : center] = carriers[:half]
    spectrum[center + 1 : center + 1 + half] = carriers[half:]
    useful = np.fft.ifft(np.fft.ifftshift(spectrum))
    return np.concatenate([useful[-mode.guard_length :], useful])


def _qpsk(bits: np.ndarray) -> np.ndarray:
    """Inverse of `dsp.qpsk.demap_soft_bits`: first all real-part bits, then all imaginary-part bits."""
    k = len(bits) // 2
    return ((1 - 2 * bits[:k].astype(float)) + 1j * (1 - 2 * bits[k:].astype(float))) / np.sqrt(2)


def build_frame(fibs: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """One transmission frame (null symbol + PRS + 75 data symbols) carrying the 12
    FIBs in the FIC."""
    mode = MODE_I
    if len(fibs) != FIBS_PER_FRAME:
        raise ValueError(f"A frame carries exactly {FIBS_PER_FRAME} FIBs.")
    info_bits = [np.concatenate(fibs[i : i + 3]) for i in range(0, FIBS_PER_FRAME, 3)]
    assert all(len(bits) == FIC_INFO_BITS for bits in info_bits)
    fic_bits = np.concatenate([encode_fic_logical_frame(bits) for bits in info_bits])

    index = frequency_deinterleave_index(mode.num_carriers, mode.fft_length)
    bits_per_symbol = 2 * mode.num_carriers
    data_symbols = []
    for i in range(mode.symbols_per_frame - 1):
        if i < 3:
            bits = fic_bits[i * bits_per_symbol : (i + 1) * bits_per_symbol]
        else:
            bits = rng.integers(0, 2, size=bits_per_symbol).astype(np.uint8)
        physical = np.empty(mode.num_carriers, dtype=complex)
        physical[index] = _qpsk(bits)
        data_symbols.append(physical)

    carriers = generate_reference_symbol(mode.num_carriers)
    symbols = [_ofdm_symbol(carriers)]
    for data in data_symbols:
        carriers = data * carriers
        symbols.append(_ofdm_symbol(carriers))
    return np.concatenate([np.zeros(mode.null_symbol_length, dtype=complex), *symbols])


def build_stream(
    fibs: list[np.ndarray],
    num_frames: int,
    rng: np.random.Generator,
    start_offset: int = 0,
    snr_db: float = 20.0,
    freq_offset_hz: float = 0.0,
) -> np.ndarray:
    """A sequence of `num_frames` frames all carrying the same FIBs, with noise and
    frequency offset; starts `start_offset` samples into the first frame."""
    frame = build_frame(fibs, rng)
    stream = np.tile(frame, num_frames)[start_offset:]
    signal_power = np.mean(np.abs(frame[MODE_I.null_symbol_length :]) ** 2)
    noise_power = signal_power / 10 ** (snr_db / 10)
    noise = rng.normal(size=len(stream)) + 1j * rng.normal(size=len(stream))
    stream = stream + noise * np.sqrt(noise_power / 2)
    n = np.arange(len(stream))
    return stream * np.exp(2j * np.pi * freq_offset_hz * n / SAMPLE_RATE)


def paused_carrier(
    num_samples: int, power: float, pause_start: int, frequency_hz: float = 300_000.0
) -> np.ndarray:
    """An interferer in the band that falls silent for a null symbol length once per
    frame, from `pause_start` on: its pauses make power minima that are not the
    ensemble's null symbols. On a whole subcarrier spacing it disturbs only that one
    subcarrier, so the ensemble stays decodable next to a carrier stronger than itself."""
    n = np.arange(num_samples)
    carrier = np.sqrt(power) * np.exp(2j * np.pi * frequency_hz * n / SAMPLE_RATE)
    carrier[(n - pause_start) % MODE_I.frame_length < MODE_I.null_symbol_length] = 0
    return carrier


def noise(num_samples: int, rng: np.random.Generator) -> np.ndarray:
    """An empty channel: noise only."""
    return (rng.normal(size=num_samples) + 1j * rng.normal(size=num_samples)) * 0.01
