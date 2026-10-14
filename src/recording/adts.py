"""ADTS, the self-synchronising stream format of AAC (ISO/IEC 14496-3, clause 1.A.3):
every access unit behind a 7-byte header with a sync word and the unit's length.

A title records as such a stream: the file grows unit by unit, and whatever is on the
drive when the power goes is readable up to the last complete unit. It is only a
transport, not the final form: ADTS cannot say that DAB+ access units span 960 instead
of 1024 core samples (ETSI TS 102 563), and a player reading the stream as it is would
decode noise. Once a title is complete, `recording.mp4` packs the units into an MP4
container whose AudioSpecificConfig carries the 960.

SBR and parametric stereo are not signalled in ADTS either; the header names the AAC
core, its sample rate and its channels.
"""

from __future__ import annotations

HEADER_LENGTH = 7
# The sampling frequency index of the AudioSpecificConfig (ISO/IEC 14496-3, table 1.18);
# DAB+ cores run at 48, 32, 24 and 16 kHz.
SAMPLING_FREQUENCY_INDEX = {
    96_000: 0,
    88_200: 1,
    64_000: 2,
    48_000: 3,
    44_100: 4,
    32_000: 5,
    24_000: 6,
    22_050: 7,
    16_000: 8,
    12_000: 9,
    11_025: 10,
    8_000: 11,
}
SAMPLE_RATE_OF_INDEX = {index: rate for rate, index in SAMPLING_FREQUENCY_INDEX.items()}
# The profile field: the audio object type less one; AAC LC is object type 2.
_PROFILE_AAC_LC = 1
# The longest unit a header can describe (13 bits for header and unit together).
MAX_UNIT_LENGTH = (1 << 13) - 1 - HEADER_LENGTH


def adts_header(core_sample_rate: int, channel_config: int, unit_length: int) -> bytes:
    """The header for one access unit of `unit_length` bytes: MPEG-4, AAC LC, no CRC,
    the buffer fullness marked variable."""
    if not 0 <= unit_length <= MAX_UNIT_LENGTH:
        raise ValueError(f"access unit of {unit_length} bytes does not fit an ADTS header")
    index = SAMPLING_FREQUENCY_INDEX[core_sample_rate]
    length = HEADER_LENGTH + unit_length
    return bytes(
        (
            0xFF,
            0xF1,  # sync word, MPEG-4, layer 0, protection absent
            (_PROFILE_AAC_LC << 6) | (index << 2) | ((channel_config >> 2) & 1),
            ((channel_config & 3) << 6) | ((length >> 11) & 3),
            (length >> 3) & 0xFF,
            ((length & 7) << 5) | 0x1F,  # buffer fullness 0x7FF: variable bit rate
            0xFC,  # the rest of it, and one raw data block
        )
    )


def split_adts(data: bytes) -> list[bytes]:
    """The access units of an ADTS stream. Bytes that are no header are skipped until
    the next sync word, and a unit cut short at the end (the power went) is left out."""
    units: list[bytes] = []
    position = 0
    end = len(data)
    while position + HEADER_LENGTH <= end:
        if data[position] != 0xFF or (data[position + 1] & 0xF6) != 0xF0:
            position += 1
            continue
        protection_absent = data[position + 1] & 1
        header_length = HEADER_LENGTH if protection_absent else HEADER_LENGTH + 2
        length = (
            ((data[position + 3] & 3) << 11) | (data[position + 4] << 3) | (data[position + 5] >> 5)
        )
        if length < header_length:
            position += 1
            continue
        if position + length > end:
            break
        units.append(data[position + header_length : position + length])
        position += length
    return units


def read_adts_format(data: bytes) -> tuple[int, int] | None:
    """Core sample rate and channel configuration from the first header; None where
    there is none."""
    position = 0
    end = len(data)
    while position + HEADER_LENGTH <= end:
        if data[position] == 0xFF and (data[position + 1] & 0xF6) == 0xF0:
            index = (data[position + 2] >> 2) & 0xF
            channels = ((data[position + 2] & 1) << 2) | (data[position + 3] >> 6)
            rate = SAMPLE_RATE_OF_INDEX.get(index)
            if rate is not None:
                return rate, channels
        position += 1
    return None
