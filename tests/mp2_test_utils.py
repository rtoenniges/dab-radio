"""Shared helpers for the tests of classic DAB (MPEG Audio Layer II)."""

from __future__ import annotations

import numpy as np

from dab.fig import Subchannel
from dsp.convolutional import TAIL_PUNCTURE_PATTERN, encode_mother_code, puncture
from dsp.eep import _P_CODES
from dsp.mp2_frame import CHANNEL_MODE_MONO, _crc16, _crc_protected_bit_count, parse_header
from dsp.scrambling import apply_energy_dispersal
from dsp.uep import uep_padding_bits, uep_segments

# A real frame, encoded with libtwolame (48 kHz, mono, 32 kbit/s, with CRC) from a 1 kHz
# sine. It is the touchstone for the CRC: the encoder computed it independently of
# this project. A frame holds exactly 24 periods of the tone, so every frame of the
# stream looks the same.
REAL_FRAME = bytes.fromhex(
    "fffc14c432f88a00002981fbc7dbd4a383d3ef1f6f528e0f4fbc7dbd4a383d3e"
    "f1f6f528e0f4fbc7dbd4a383d3ef1f6f528e0f4fbc7dbd4a383d3ef1f6f528e0"
    "f4fbc7dbd4a383d3ef1f6f528e0f4fbc7dbd4a383d3ef1f6f528e0f4c0000000"
)

_BITRATE_INDEX_MPEG1 = {32: 1, 48: 2, 56: 3, 64: 4, 96: 6, 128: 8, 192: 10}
_BITRATE_INDEX_LSF = {32: 4, 64: 8, 96: 10}


def mp2_header(
    bitrate: int,
    *,
    lsf: bool = False,
    channel_mode: int = CHANNEL_MODE_MONO,
    layer: int = 0b10,
    sampling_index: int = 0b01,
) -> bytes:
    bitrate_index = (_BITRATE_INDEX_LSF if lsf else _BITRATE_INDEX_MPEG1)[bitrate]
    word = (
        (0xFFF << 20)
        | ((0 if lsf else 1) << 19)
        | (layer << 17)
        | (0 << 16)  # protection_bit 0: CRC present
        | (bitrate_index << 12)
        | (sampling_index << 10)
        | (channel_mode << 6)
    )
    return word.to_bytes(4, "big")


def silent_mp2_frame(
    bitrate: int, *, lsf: bool = False, channel_mode: int = CHANNEL_MODE_MONO
) -> bytes:
    """A valid frame of silence: header, correct CRC, all bit allocations zero."""
    header_bytes = mp2_header(bitrate, lsf=lsf, channel_mode=channel_mode)
    header = parse_header(header_bytes)
    body = np.zeros(header.frame_length - 6, dtype=np.uint8)
    body_bits = np.unpackbits(body)
    count = _crc_protected_bit_count(header, body_bits)
    crc = _crc16(
        np.concatenate(
            [np.unpackbits(np.frombuffer(header_bytes[2:], np.uint8)), body_bits[:count]]
        )
    )
    return header_bytes + crc.to_bytes(2, "big") + body.tobytes()


def logical_frames(stream: bytes, bitrate: int) -> list[np.ndarray]:
    """Cuts a byte stream into the information bits of logical frames (3 bytes per
    kbit/s, i.e. 24 ms)."""
    size = 3 * bitrate
    return [
        np.unpackbits(np.frombuffer(stream[i : i + size], dtype=np.uint8))
        for i in range(0, len(stream) - size + 1, size)
    ]


def with_pad(frame: bytes, x_pad: bytes, f_pad: bytes, scf_crc_length: int) -> bytes:
    """Writes a PAD into the end of a frame as DAB lays it out: X-PAD in reversed byte
    order, then the ScF-CRC (left zero here), then the F-PAD."""
    end = len(frame) - len(f_pad) - scf_crc_length
    return frame[: end - len(x_pad)] + x_pad[::-1] + bytes(scf_crc_length) + f_pad


def encode_uep_logical_frame(info_bits: np.ndarray, subchannel: Subchannel) -> np.ndarray:
    """Mirror image of `decode_uep_logical_frame`: scrambles, convolutionally encodes,
    punctures segment by segment and appends the padding bits."""
    mother = encode_mother_code(apply_energy_dispersal(info_bits))
    parts = []
    offset = 0
    for num_blocks, index in uep_segments(subchannel):
        length = num_blocks * 128
        parts.append(puncture(mother[offset : offset + length], np.tile(_P_CODES[index], 4)))
        offset += length
    parts.append(puncture(mother[offset:], TAIL_PUNCTURE_PATTERN))
    parts.append(np.zeros(uep_padding_bits(subchannel), dtype=np.uint8))
    return np.concatenate(parts)
