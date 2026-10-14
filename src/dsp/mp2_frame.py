"""DAB audio frames (MPEG Audio Layer II): frame sync, header CRC and frame layout.

Classic DAB (ASCTy 0) carries MPEG Audio Layer II, "MP2", instead of the AAC
superframes of DAB+ (ETSI EN 300 401, clause 7). There is no Reed-Solomon code on top:
the only error correction is the convolutional code (Viterbi) of the MSC. The CRC in
the MPEG header merely detects errors, and only in the part of the frame where an
error would be really audible (bit allocation and scale factor selection).

Layout of a DAB audio frame, from front to back:

    header (4) | CRC (2) | audio data ... | X-PAD (reversed) | ScF-CRC (2/4) | F-PAD (2)

A frame always holds 1152 samples. At 48 kHz (MPEG-1) that is 24 ms and exactly one
logical frame; at 24 kHz (MPEG-2 "low sampling frequency", LSF) it is 48 ms, spread
over two logical frames. The synchroniser below therefore treats the logical frames as
one continuous byte stream and searches the frames in it, instead of assuming any
alignment to the logical frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from i18n import tr, tr_noop

# ISO/IEC 11172-3, clause 2.4.2.3: the fields of the 32-bit header.
_SYNC_WORD = 0xFFF
_LAYER_II = 0b10
# sampling_frequency index 01 is 48 kHz for MPEG-1 and 24 kHz for MPEG-2 LSF, the only
# two rates DAB allows (ETSI EN 300 401, clause 7).
_SAMPLING_INDEX_DAB = 0b01
_SAMPLE_RATE_MPEG1 = 48_000
_SAMPLE_RATE_LSF = 24_000

# Layer II bit rates in kbit/s by bitrate_index (0 = free format, 15 = forbidden):
# ISO/IEC 11172-3 for MPEG-1, ISO/IEC 13818-3 for the low sampling frequencies.
_BITRATES_MPEG1 = (0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 0)
_BITRATES_LSF = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0)

SAMPLES_PER_FRAME = 1152
HEADER_BYTES = 4
CRC_BYTES = 2
F_PAD_BYTES = 2

# Channel modes of the header (ISO/IEC 11172-3, clause 2.4.2.3).
CHANNEL_MODE_STEREO = 0
CHANNEL_MODE_JOINT_STEREO = 1
CHANNEL_MODE_DUAL_CHANNEL = 2
CHANNEL_MODE_MONO = 3
_CHANNEL_MODE_NAMES = {
    CHANNEL_MODE_STEREO: tr_noop("Stereo"),
    CHANNEL_MODE_JOINT_STEREO: tr_noop("Joint Stereo"),
    CHANNEL_MODE_DUAL_CHANNEL: tr_noop("Dual channel"),
    CHANNEL_MODE_MONO: tr_noop("Mono"),
}

# Number of bits of the bit allocation per subband ("nbal"). Which table applies
# depends on sample rate and bit rate per channel; its length is the number of
# subbands actually used (sblimit). Cross-checked against DABlin (dab_decoder.cpp).
# ISO/IEC 11172-3, table B.2a: 48 kHz, 56 kbit/s per channel and above.
_NBAL_48K_HIGH_RATE = (4,) * 11 + (3,) * 12 + (2,) * 4
# ISO/IEC 11172-3, table B.2c: 48 kHz, 32 and 48 kbit/s per channel.
_NBAL_48K_LOW_RATE = (4,) * 2 + (3,) * 6
# ISO/IEC 13818-3, table B.1: all low sampling frequencies.
_NBAL_LSF = (4,) * 4 + (3,) * 7 + (2,) * 19

# ISO/IEC 11172-3, clause 2.4.3.1: CRC-16 with generator x^16 + x^15 + x^2 + 1,
# register initialised to all ones, no final inversion.
_CRC_POLY = 0x8005
_CRC_INIT = 0xFFFF

# This many consecutive frame positions without a valid header count as a loss of the
# frame grid; after that the frames are searched afresh. Same reasoning as for the
# superframes of DAB+ (`dsp.pipeline.MAX_SUPERFRAME_MISSES`).
MAX_FRAME_MISSES = 5


@dataclass(frozen=True)
class Mp2Format:
    """What a DAB audio frame header states about the audio, i.e. what the decoder and
    the display need to know. Fixed for a service as long as the broadcaster does not
    switch, e.g. from stereo to mono."""

    sample_rate: int  # 48 000 (MPEG-1) or 24 000 (MPEG-2 LSF)
    bitrate: int  # kbit/s, equal to the bit rate of the subchannel
    channel_mode: int  # CHANNEL_MODE_*

    @property
    def is_lsf(self) -> bool:
        """MPEG-2 with half the sample rate ("low sampling frequency")."""
        return self.sample_rate == _SAMPLE_RATE_LSF

    @property
    def channels(self) -> int:
        return 1 if self.channel_mode == CHANNEL_MODE_MONO else 2

    @property
    def frame_length(self) -> int:
        """Frame length in bytes: 3 per kbit/s at 48 kHz, 6 per kbit/s at 24 kHz."""
        return SAMPLES_PER_FRAME // 8 * self.bitrate * 1000 // self.sample_rate

    @property
    def duration_seconds(self) -> float:
        return SAMPLES_PER_FRAME / self.sample_rate

    @property
    def scf_crc_length(self) -> int:
        """Length of the scale factor CRC that DAB places in front of the F-PAD (ETSI EN
        300 401, clause 7). Two bytes where only 8 subbands are used (48 kHz below 56
        kbit/s per channel), four otherwise. Cross-checked against DABlin
        (dab_decoder.cpp, `scf_crc_len`)."""
        return 2 if _nbal_table(self) is _NBAL_48K_LOW_RATE else 4

    @property
    def channel_mode_name(self) -> str:
        """The channel mode as the GUI names it."""
        return tr(_CHANNEL_MODE_NAMES[self.channel_mode])

    def describe(self) -> str:
        """Short description for the GUI."""
        version = "MPEG-2" if self.is_lsf else "MPEG-1"
        return (
            f"{version} Layer II, {self.sample_rate // 1000} kHz, "
            f"{self.channel_mode_name}, {self.bitrate} kbit/s"
        )


def _nbal_table(audio_format: Mp2Format) -> tuple[int, ...]:
    if audio_format.is_lsf:
        return _NBAL_LSF
    if audio_format.bitrate // audio_format.channels >= 56:
        return _NBAL_48K_HIGH_RATE
    return _NBAL_48K_LOW_RATE


@dataclass(frozen=True)
class Mp2Header:
    """The evaluated 32-bit header of one frame."""

    audio_format: Mp2Format
    mode_extension: int  # joint stereo: from which subband on both channels are shared
    protected: bool  # a CRC follows the header (always the case in DAB)
    padding: bool  # one extra byte; never set at the DAB sample rates

    @property
    def frame_length(self) -> int:
        return self.audio_format.frame_length + (1 if self.padding else 0)


def parse_header(data: bytes) -> Mp2Header | None:
    """Reads a Layer II header at 48 or 24 kHz from the first 4 bytes of `data`.

    Returns:
        The header, or None if the bytes are no valid DAB audio frame header.
    """
    if len(data) < HEADER_BYTES:
        return None
    word = int.from_bytes(data[:HEADER_BYTES], "big")
    if word >> 20 != _SYNC_WORD:
        return None
    is_mpeg1 = bool((word >> 19) & 0x1)
    layer = (word >> 17) & 0x3
    protection_bit = (word >> 16) & 0x1
    bitrate_index = (word >> 12) & 0xF
    sampling_index = (word >> 10) & 0x3
    if layer != _LAYER_II or sampling_index != _SAMPLING_INDEX_DAB:
        return None
    bitrate = (_BITRATES_MPEG1 if is_mpeg1 else _BITRATES_LSF)[bitrate_index]
    if bitrate == 0:
        return None
    return Mp2Header(
        audio_format=Mp2Format(
            sample_rate=_SAMPLE_RATE_MPEG1 if is_mpeg1 else _SAMPLE_RATE_LSF,
            bitrate=bitrate,
            channel_mode=(word >> 6) & 0x3,
        ),
        mode_extension=(word >> 4) & 0x3,
        protected=protection_bit == 0,  # 0 means: CRC present
        padding=bool((word >> 9) & 0x1),
    )


def _crc16(bits: np.ndarray) -> int:
    crc = _CRC_INIT
    for bit in bits:
        feedback = ((crc >> 15) & 1) ^ int(bit)
        crc = (crc << 1) & 0xFFFF
        if feedback:
            crc ^= _CRC_POLY
    return crc


def _crc_protected_bit_count(header: Mp2Header, body_bits: np.ndarray) -> int | None:
    """Number of bits after the CRC field that the header CRC covers: the complete bit
    allocation followed by the scale factor selection information (scfsi), 2 bits per
    subband and channel with a nonzero allocation (ISO/IEC 11172-3, clause 2.4.3.1).

    The allocation has to be read to know that, since it determines how many scfsi
    fields follow. Cross-checked against DABlin (dab_decoder.cpp, `CheckCRC`).

    Returns:
        The bit count, or None if the frame is too short for the allocation.
    """
    audio_format = header.audio_format
    nbal = _nbal_table(audio_format)
    sblimit = len(nbal)
    channels = audio_format.channels
    # Joint stereo: above the "bound" both channels share one allocation.
    bound = (
        (header.mode_extension + 1) * 4
        if audio_format.channel_mode == CHANNEL_MODE_JOINT_STEREO
        else sblimit
    )

    position = 0
    scfsi_bits = 0
    for subband in range(sblimit):
        for _ in range(channels if subband < bound else 1):
            width = nbal[subband]
            if position + width > len(body_bits):
                return None
            allocation = 0
            for bit in body_bits[position : position + width]:
                allocation = (allocation << 1) | int(bit)
            position += width
            if allocation:
                scfsi_bits += 2 * (1 if subband < bound else channels)
    return position + scfsi_bits


def check_header_crc(frame: bytes, header: Mp2Header) -> bool:
    """Checks the CRC after the header against the last 16 header bits and the bit
    allocation and scfsi it protects.

    DAB prescribes the CRC, so a frame without one counts as corrupted as well.
    """
    if not header.protected or len(frame) < HEADER_BYTES + CRC_BYTES:
        return False
    body_bits = np.unpackbits(np.frombuffer(frame[HEADER_BYTES + CRC_BYTES :], dtype=np.uint8))
    count = _crc_protected_bit_count(header, body_bits)
    if count is None:
        return False
    header_bits = np.unpackbits(np.frombuffer(frame[2:HEADER_BYTES], dtype=np.uint8))
    stored = int.from_bytes(frame[HEADER_BYTES : HEADER_BYTES + CRC_BYTES], "big")
    return _crc16(np.concatenate([header_bits, body_bits[:count]])) == stored


@dataclass
class Mp2FrameResult:
    """One position of the frame grid, as reported to the display and the audio output.

    A result is only reported while the grid is locked, so `audio_format` is always the
    format the grid was locked with, even if the header at this position is broken.
    """

    audio_format: Mp2Format
    sync_ok: bool = False  # a valid header sits at the expected position
    crc_ok: bool = False  # bit allocation and scfsi intact (header CRC)
    frame: bytes = b""  # the complete frame including header, input for the decoder
    # X-PAD bytes used in this frame; only known after the PAD evaluation
    # (`dab.pad.PadReader.read_mp2_frame`) and filled in by the caller.
    x_pad_length: int = 0

    def segments(self) -> list[int]:
        """Byte boundaries of the frame parts for the layout display: end of header
        with CRC, end of the audio data, end of the X-PAD, end of the ScF-CRC, end of
        the frame (after the F-PAD)."""
        length = len(self.frame) or self.audio_format.frame_length
        pad_start = length - F_PAD_BYTES
        scf_crc_start = pad_start - self.audio_format.scf_crc_length
        audio_end = max(HEADER_BYTES + CRC_BYTES, scf_crc_start - self.x_pad_length)
        return [HEADER_BYTES + CRC_BYTES, audio_end, scf_crc_start, pad_start, length]


class Mp2FrameSynchronizer:
    """Finds the DAB audio frames in the sequence of decoded logical frames.

    Searching means: the first position in the byte stream with a valid header for the
    subchannel's bit rate whose header CRC also checks out. From there the grid is
    followed by the frame length alone, like the superframes of DAB+: a corrupted
    frame does not shift the grid but is reported as such, and only after
    `MAX_FRAME_MISSES` positions in a row without a valid header is the search started
    again.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._audio_format: Mp2Format | None = None  # None = searching
        self._misses = 0

    @property
    def locked(self) -> bool:
        return self._audio_format is not None

    def reset(self) -> None:
        self._buffer.clear()
        self._audio_format = None
        self._misses = 0

    def push(self, frame_bits: np.ndarray) -> list[Mp2FrameResult]:
        """Takes the information bits of one logical frame (`24 * bitrate` bits) and
        returns the frames that are complete with it (none, one, or rarely two)."""
        bitrate = len(frame_bits) // 24
        self._buffer.extend(np.packbits(frame_bits).tobytes())
        results: list[Mp2FrameResult] = []
        while True:
            if self._audio_format is None and not self._search(bitrate):
                return results
            length = self._audio_format.frame_length
            if len(self._buffer) < length:
                return results
            results.append(self._take_frame(length))

    def _search(self, bitrate: int) -> bool:
        """Discards bytes up to the next frame start. False: none found (yet)."""
        position = 0
        while True:
            position = self._buffer.find(0xFF, position)
            if position < 0:
                # Keep the last bytes: a header may begin there and continue in the
                # next logical frame.
                del self._buffer[: max(0, len(self._buffer) - (HEADER_BYTES - 1))]
                return False
            header = parse_header(self._buffer[position : position + HEADER_BYTES])
            if header is not None and header.audio_format.bitrate == bitrate:
                end = position + header.frame_length
                if end > len(self._buffer):
                    del self._buffer[:position]  # wait for the rest of the frame
                    return False
                if check_header_crc(bytes(self._buffer[position:end]), header):
                    del self._buffer[:position]
                    self._audio_format = header.audio_format
                    self._misses = 0
                    return True
            position += 1

    def _take_frame(self, length: int) -> Mp2FrameResult:
        frame = bytes(self._buffer[:length])
        header = parse_header(frame)
        # A header with a different channel mode but the same frame length is a genuine
        # format change of the broadcaster, not a broken frame.
        if header is not None and header.frame_length == length:
            self._audio_format = header.audio_format
            self._misses = 0
            del self._buffer[:length]
            return Mp2FrameResult(
                audio_format=header.audio_format,
                sync_ok=True,
                crc_ok=check_header_crc(frame, header),
                frame=frame,
            )

        audio_format = self._audio_format
        self._misses += 1
        del self._buffer[:length]
        if self._misses >= MAX_FRAME_MISSES:
            self._audio_format = None
            self._misses = 0
        return Mp2FrameResult(audio_format=audio_format)
