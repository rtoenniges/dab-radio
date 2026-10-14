import numpy as np
import pytest

from dab.crc import crc16_ccitt
from dsp.superframe import AudioFormat, decode_superframe, fire_code_crc
from tests.rs_test_utils import encode_rs_codeword


def _build_au(payload: bytes) -> bytes:
    """Builds an audio unit: payload + its own CRC-16 (ETSI TS 102 563)."""
    crc = crc16_ccitt(payload) ^ 0xFFFF
    return payload + bytes([(crc >> 8) & 0xFF, crc & 0xFF])


def _build_synthetic_superframe_logical_frames(
    au_payloads: list[bytes],
) -> tuple[list[np.ndarray], list[bytes]]:
    """Builds 5 synthetic EEP-decoded logical frames (bits) for a valid DAB+
    superframe with `bitrate=8` (subch_index=1, i.e. a single RS codeword across the
    whole superframe, the simplest case).

    Format: dac_rate=0, sbr_flag=0 -> num_aus=4, header length 8 bytes.
    """
    assert len(au_payloads) == 4
    aus = [_build_au(p) for p in au_payloads]

    au_start = [8]
    offset = 8
    for au in aus:
        offset += len(au)
        au_start.append(offset)
    data_len = au_start[-1]
    assert data_len == 110, "the test set-up must yield exactly 110 data bytes (subch_index=1)."

    header = bytearray(8)
    header[2] = 0x00  # dac_rate=0, sbr_flag=0 -> num_aus=4
    header[3] = (au_start[1] >> 4) & 0xFF
    header[4] = ((au_start[1] & 0x0F) << 4) | ((au_start[2] >> 8) & 0x0F)
    header[5] = au_start[2] & 0xFF
    header[6] = (au_start[3] >> 4) & 0xFF
    header[7] = (au_start[3] & 0x0F) << 4

    # The firecode protects data[2:11], which includes the first 3 bytes of the AU
    # payload (au_start[0]=8), not only the header bytes.
    body = bytes(header[2:8]) + b"".join(aus)
    crc = fire_code_crc(body[:9])
    data = bytes([(crc >> 8) & 0xFF, crc & 0xFF]) + body
    assert len(data) == 110

    codeword = encode_rs_codeword(np.frombuffer(data, dtype=np.uint8))
    assert len(codeword) == 120

    logical_frames = [np.unpackbits(codeword[i * 24 : (i + 1) * 24]) for i in range(5)]
    return logical_frames, au_payloads


def test_decode_superframe_extracts_valid_audio_units_without_errors():
    au_payloads = [b"A" * 28, b"B" * 28, b"C" * 28, b"D" * 10]
    logical_frames, expected_payloads = _build_synthetic_superframe_logical_frames(au_payloads)

    result = decode_superframe(logical_frames)

    assert result.sync_ok is True
    assert result.rs_corrections == 0
    assert result.rs_uncorrectable_blocks == 0
    assert result.num_audio_units == 4
    assert [bytes(au) for au in result.audio_units] == expected_payloads


def test_decode_superframe_corrects_byte_errors_in_rs_codeword():
    au_payloads = [b"A" * 28, b"B" * 28, b"C" * 28, b"D" * 10]
    logical_frames, expected_payloads = _build_synthetic_superframe_logical_frames(au_payloads)

    # Inject one byte error into one of the logical frames (< 5 per RS codeword, so
    # it must be fully correctable).
    corrupted_frame = logical_frames[2].copy()
    byte_bits = corrupted_frame[8:16]
    byte_bits[:] = 1 - byte_bits
    corrupted_frame[8:16] = byte_bits
    logical_frames[2] = corrupted_frame

    result = decode_superframe(logical_frames)

    assert result.sync_ok is True
    assert result.rs_corrections == 1
    assert result.rs_uncorrectable_blocks == 0
    assert [bytes(au) for au in result.audio_units] == expected_payloads


def test_decode_superframe_fails_sync_on_garbage_input():
    rng = np.random.default_rng(5)
    logical_frames = [rng.integers(0, 2, size=192).astype(np.uint8) for _ in range(5)]

    result = decode_superframe(logical_frames)

    assert result.sync_ok is False
    assert result.audio_units == []


def test_decode_superframe_raises_for_wrong_frame_count():
    with pytest.raises(ValueError, match="5 logical frames"):
        decode_superframe([np.zeros(192, dtype=np.uint8)])


def test_decode_superframe_reports_the_audio_format():
    """The format byte of the superframe header (ETSI TS 102 563, clause 5.2)
    determines how the contained AAC bit stream is to be decoded. The test set-up
    uses format byte 0x00: dac_rate=0, sbr=0, mono, no PS."""
    logical_frames, _ = _build_synthetic_superframe_logical_frames(
        [b"A" * 28, b"B" * 28, b"C" * 28, b"D" * 10]
    )
    audio_format = decode_superframe(logical_frames).audio_format

    assert audio_format is not None
    assert audio_format.dac_rate_48k is False
    assert audio_format.sbr is False
    assert audio_format.stereo_core is False
    assert audio_format.parametric_stereo is False
    assert audio_format.sample_rate == 32_000
    assert audio_format.core_sample_rate == 32_000
    assert audio_format.channels == 1
    assert audio_format.num_audio_units == 4
    assert audio_format.describe() == "32 kHz, Mono"


@pytest.mark.parametrize(
    ("format_byte", "expected"),
    [
        (0b0100_0000, "48 kHz, Mono"),
        (0b0110_0000, "48 kHz, Mono, SBR"),
        (0b0111_0000, "48 kHz, Stereo, SBR"),
        (0b0110_1000, "48 kHz, Stereo, SBR, PS"),
        (0b0001_0000, "32 kHz, Stereo"),
    ],
)
def test_format_byte_is_decoded_bit_by_bit(format_byte, expected):
    """Bit 6 dac_rate, bit 5 sbr_flag, bit 4 aac_channel_mode, bit 3 ps_flag."""
    audio_format = AudioFormat(
        dac_rate_48k=bool(format_byte & 0x40),
        sbr=bool(format_byte & 0x20),
        stereo_core=bool(format_byte & 0x10),
        parametric_stereo=bool(format_byte & 0x08),
        mpeg_surround_config=format_byte & 0x07,
    )
    assert audio_format.describe() == expected
