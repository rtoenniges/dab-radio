"""DAB+ superframes: Reed-Solomon correction, firecode sync and audio unit extraction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dab.crc import check_crc
from dsp.reed_solomon import CODEWORD_BYTES, DATA_BYTES, decode_rs_codeword
from i18n import tr

# ETSI TS 102 563: a superframe is 5 logical frames of 24 ms, i.e. 120 ms of audio.
NUM_LOGICAL_FRAMES_PER_SUPERFRAME = 5
SUPERFRAME_SECONDS = 0.120

# ETSI TS 102 563, clause 5.2 / tools.cpp `CalcCRC_FIRE_CODE`: 16-bit CRC without
# inversion of initial or final value. It protects the first 11 bytes of the
# superframe (2 bytes CRC + 9 bytes header) and lets the receiver lock onto the start
# of a valid superframe.
_FIRE_CODE_POLY = 0x782F


def fire_code_crc(data: bytes) -> int:
    """Computes the DAB+ firecode over `data` (ETSI TS 102 563, clause 5.2).

    Cross-checked against welle.io/DABlin (tools.cpp, `CalcCRC_FIRE_CODE`): polynomial
    0x782F, register initialised to 0x0000, no final inversion, unlike the CRC-16/CCITT
    variant from `dab.crc` (not reflected, but with 0xFFFF init/inversion).
    """
    crc = 0x0000
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ _FIRE_CODE_POLY) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


@dataclass(frozen=True)
class AudioFormat:
    """Audio format of a DAB+ superframe (ETSI TS 102 563, clause 5.2).

    The format byte of the superframe header describes how the contained AAC bit
    stream is to be decoded. The values are needed for the AudioSpecificConfig of the
    AAC decoder (see `audio.aac`).
    """

    dac_rate_48k: bool  # True: 48 kHz output, False: 32 kHz
    sbr: bool  # spectral band replication (HE-AAC): the core runs at half the rate
    stereo_core: bool  # the AAC core is two-channel (aac_channel_mode)
    parametric_stereo: bool  # PS: stereo is generated from a mono core
    mpeg_surround_config: int  # 0 = no MPEG Surround

    @property
    def sample_rate(self) -> int:
        """Sample rate of the decoded audio (after SBR upsampling)."""
        return 48_000 if self.dac_rate_48k else 32_000

    @property
    def core_sample_rate(self) -> int:
        """Sample rate of the AAC core; half the output rate with SBR."""
        return self.sample_rate // 2 if self.sbr else self.sample_rate

    @property
    def channels(self) -> int:
        """Channel count of the decoded audio (PS turns mono into stereo)."""
        return 2 if self.stereo_core or self.parametric_stereo else 1

    @property
    def num_audio_units(self) -> int:
        """Number of audio units per superframe (ETSI TS 102 563, table 3).

        A superframe always lasts 120 ms and an AAC access unit covers 960 core
        samples in DAB+, so the core rate and the 120 ms directly give how many fit.
        """
        return self.core_sample_rate * 120 // 1000 // 960

    def describe(self) -> str:
        """Short description for the GUI."""
        parts = [
            f"{self.sample_rate // 1000} kHz",
            tr("Stereo") if self.channels == 2 else tr("Mono"),
        ]
        if self.sbr:
            parts.append("SBR")
        if self.parametric_stereo:
            parts.append("PS")
        return ", ".join(parts)


@dataclass(frozen=True)
class SuperframeHeader:
    """The evaluated superframe header (ETSI TS 102 563, clause 5.2)."""

    audio_format: AudioFormat
    # Byte offsets of the audio units within the data part, followed by the end of
    # the data as a final sentinel: audio unit i spans `au_start[i]:au_start[i + 1]`.
    au_start: list[int]


@dataclass
class SuperframeResult:
    """Result of decoding one DAB+ superframe (ETSI TS 102 563)."""

    sync_ok: bool = False
    rs_corrections: int = 0  # number of bytes corrected by Reed-Solomon
    rs_uncorrectable_blocks: int = 0  # number of RS codewords with too many errors
    audio_units: list[bytes] = field(default_factory=list)  # CRC-valid AUs only
    num_audio_units: int = 0  # total according to the superframe header (invalid ones included)
    audio_format: AudioFormat | None = None  # None until a firecode matched
    # Layout of the superframe in bytes, for display: per audio unit (start byte, end
    # byte, CRC valid). Everything before the first is the header; everything past
    # `data_length` are the Reed-Solomon parity bytes.
    au_segments: list[tuple[int, int, bool]] = field(default_factory=list)
    data_length: int = 0  # payload bytes (without RS parity)
    total_length: int = 0  # total length including RS parity


def assemble_superframe_bytes(logical_frames: list[np.ndarray]) -> np.ndarray:
    """Packs the 5 EEP-decoded logical frames (`num_info_bits` bits each) of a
    superframe (ETSI TS 102 563, clause 5.1) into a byte array.

    The bits of each logical frame are packed into bytes MSB first (as elsewhere in
    this project, e.g. `dab.fig.parse_fib`); the 5 byte blocks are concatenated in
    transmission order.
    """
    if len(logical_frames) != NUM_LOGICAL_FRAMES_PER_SUPERFRAME:
        raise ValueError(
            f"A superframe needs exactly {NUM_LOGICAL_FRAMES_PER_SUPERFRAME} logical frames."
        )
    return np.concatenate([np.packbits(frame) for frame in logical_frames])


def correct_superframe(raw_bytes: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Corrects the Reed-Solomon protected bytes of a superframe.

    `raw_bytes` (length `15 * bitRate`, ETSI TS 102 563 clause 5.1) is implicitly
    treated as a `(120, subch_index)` matrix in row-major order; each of the
    `subch_index` columns is an independent RS(120,110) codeword. Since this
    arrangement is exactly the transmission order (see `dsp.reed_solomon`), the matrix
    is simply `raw_bytes.reshape(120, subch_index)` without any prior reordering,
    cross-checked against welle.io/DABlin (`RSDecoder::DecodeSuperframe`).

    Returns:
        (corrected, total_corrections, num_uncorrectable_blocks): `corrected` has the
        same shape as `raw_bytes` (still including the RS parity bytes of each column,
        see `extract_data_bytes`).
    """
    subch_index = len(raw_bytes) // CODEWORD_BYTES
    matrix = raw_bytes.reshape(CODEWORD_BYTES, subch_index).copy()

    total_corrections = 0
    num_uncorrectable = 0
    for column in range(subch_index):
        corrected, num_corrections = decode_rs_codeword(matrix[:, column])
        if num_corrections == -1:
            num_uncorrectable += 1
        else:
            matrix[:, column] = corrected
            total_corrections += num_corrections

    return matrix.reshape(-1), total_corrections, num_uncorrectable


def extract_data_bytes(corrected_bytes: np.ndarray) -> np.ndarray:
    """Strips the Reed-Solomon parity bytes and returns the pure payload.

    Since the columns (RS codewords) are interleaved row by row (see
    `correct_superframe`), the first `DATA_BYTES` rows of all columns together are
    exactly the leading `110/120` share of `corrected_bytes`, in unchanged order.
    """
    subch_index = len(corrected_bytes) // CODEWORD_BYTES
    return corrected_bytes[: DATA_BYTES * subch_index]


def _parse_header(data: np.ndarray) -> SuperframeHeader | None:
    """Checks the firecode and reads the superframe header (ETSI TS 102 563, clause
    6.1: format byte + AU start table). Cross-checked against welle.io/DABlin
    (dabplus_decoder.cpp, `SuperframeFilter::CheckSync`).

    Returns:
        The header, or None if the firecode or the plausibility check fails.
    """
    if len(data) < 11:
        return None
    if data[3] == 0 and data[4] == 0:
        return None

    crc_stored = (int(data[0]) << 8) | int(data[1])
    if fire_code_crc(bytes(data[2:11])) != crc_stored:
        return None

    # Format byte (ETSI TS 102 563, clause 5.2): bit 7 rfa, bit 6 dac_rate, bit 5
    # sbr_flag, bit 4 aac_channel_mode, bit 3 ps_flag, bits 2..0 mpeg_surround_config.
    format_byte = int(data[2])
    audio_format = AudioFormat(
        dac_rate_48k=bool(format_byte & 0x40),
        sbr=bool(format_byte & 0x20),
        stereo_core=bool(format_byte & 0x10),
        parametric_stereo=bool(format_byte & 0x08),
        mpeg_surround_config=format_byte & 0x07,
    )
    dac_rate = audio_format.dac_rate_48k
    sbr_flag = audio_format.sbr

    num_aus = audio_format.num_audio_units
    au_start = [0] * (num_aus + 1)
    au_start[0] = (6 if sbr_flag else 11) if dac_rate else (5 if sbr_flag else 8)
    au_start[num_aus] = len(data)

    au_start[1] = (int(data[3]) << 4) | (int(data[4]) >> 4)
    if num_aus >= 3:
        au_start[2] = ((int(data[4]) & 0x0F) << 8) | int(data[5])
    if num_aus >= 4:
        au_start[3] = (int(data[6]) << 4) | (int(data[7]) >> 4)
    if num_aus == 6:
        au_start[4] = ((int(data[7]) & 0x0F) << 8) | int(data[8])
        au_start[5] = (int(data[9]) << 4) | (int(data[10]) >> 4)

    if any(au_start[i] >= au_start[i + 1] for i in range(num_aus)):
        return None

    return SuperframeHeader(audio_format=audio_format, au_start=au_start)


def decode_superframe(logical_frames: list[np.ndarray]) -> SuperframeResult:
    """Decodes a complete DAB+ superframe from 5 EEP-decoded logical frames down to
    the audio units it contains (ETSI TS 102 563).

    Every successfully extracted audio unit has passed its own CRC-16 (last 2 bytes)
    and holds the raw, untouched AAC "RAW_DATA_BLOCK" content, not yet AAC-decoded.
    """
    raw_bytes = assemble_superframe_bytes(logical_frames)
    corrected, total_corrections, num_uncorrectable = correct_superframe(raw_bytes)
    data = extract_data_bytes(corrected)

    header = _parse_header(data)
    if header is None:
        return SuperframeResult(
            sync_ok=False,
            rs_corrections=total_corrections,
            rs_uncorrectable_blocks=num_uncorrectable,
        )

    audio_units = []
    au_segments: list[tuple[int, int, bool]] = []
    for i in range(header.audio_format.num_audio_units):
        start, end = header.au_start[i], header.au_start[i + 1]
        au_data = data[start:end]
        valid = check_crc(au_data.tobytes())
        if valid:
            audio_units.append(bytes(au_data[:-2]))
        au_segments.append((start, end, valid))

    return SuperframeResult(
        sync_ok=True,
        rs_corrections=total_corrections,
        rs_uncorrectable_blocks=num_uncorrectable,
        audio_units=audio_units,
        num_audio_units=header.audio_format.num_audio_units,
        audio_format=header.audio_format,
        au_segments=au_segments,
        data_length=len(data),
        total_length=len(corrected),
    )
