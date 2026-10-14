"""Tests for the DAB audio frames (MPEG Audio Layer II) of classic DAB."""

from __future__ import annotations

import numpy as np
import pytest

from dsp.mp2_frame import (
    CHANNEL_MODE_JOINT_STEREO,
    CHANNEL_MODE_MONO,
    CHANNEL_MODE_STEREO,
    MAX_FRAME_MISSES,
    Mp2Format,
    Mp2FrameResult,
    Mp2FrameSynchronizer,
    check_header_crc,
    parse_header,
)
from tests.mp2_test_utils import REAL_FRAME, logical_frames, mp2_header, silent_mp2_frame


def _push_all(
    synchronizer: Mp2FrameSynchronizer, stream: bytes, bitrate: int
) -> list[Mp2FrameResult]:
    results = []
    for bits in logical_frames(stream, bitrate):
        results += synchronizer.push(bits)
    return results


def test_real_frame_header():
    header = parse_header(REAL_FRAME)

    assert header is not None
    assert header.audio_format == Mp2Format(48_000, 32, CHANNEL_MODE_MONO)
    assert header.protected
    assert header.frame_length == len(REAL_FRAME) == 96
    assert header.audio_format.duration_seconds == pytest.approx(0.024)


def test_header_crc_of_real_frame():
    assert check_header_crc(REAL_FRAME, parse_header(REAL_FRAME))


def test_header_crc_only_covers_the_sensitive_part():
    """A flipped bit in the bit allocation must fail the CRC; one in the samples
    further back does not, the CRC does not reach that far."""
    header = parse_header(REAL_FRAME)
    in_allocation = bytearray(REAL_FRAME)
    in_allocation[7] ^= 0x10
    in_samples = bytearray(REAL_FRAME)
    in_samples[60] ^= 0x10

    assert not check_header_crc(bytes(in_allocation), header)
    assert check_header_crc(bytes(in_samples), header)


@pytest.mark.parametrize(
    "header",
    [
        mp2_header(64, layer=0b01),  # Layer III
        mp2_header(64, sampling_index=0b00),  # 44.1 kHz, not allowed in DAB
        b"\x00" + mp2_header(64)[1:],  # no sync word
        mp2_header(64)[:3],  # too short
    ],
)
def test_parse_header_rejects_what_dab_does_not_use(header):
    assert parse_header(header) is None


@pytest.mark.parametrize(
    "audio_format,frame_length,scf_crc_length",
    [
        (Mp2Format(48_000, 32, CHANNEL_MODE_MONO), 96, 2),
        (Mp2Format(48_000, 56, CHANNEL_MODE_MONO), 168, 4),
        (Mp2Format(48_000, 96, CHANNEL_MODE_STEREO), 288, 2),  # 48 kbit/s per channel
        (Mp2Format(48_000, 128, CHANNEL_MODE_JOINT_STEREO), 384, 4),
        (Mp2Format(24_000, 64, CHANNEL_MODE_MONO), 384, 4),
    ],
)
def test_frame_length_and_scale_factor_crc(audio_format, frame_length, scf_crc_length):
    """3 bytes per kbit/s at 48 kHz, 6 at 24 kHz; the ScF-CRC is only 2 bytes where
    just 8 subbands are coded (48 kHz below 56 kbit/s per channel)."""
    assert audio_format.frame_length == frame_length
    assert audio_format.scf_crc_length == scf_crc_length


def test_synchroniser_finds_the_frames_behind_arbitrary_bytes():
    junk = np.random.default_rng(3).integers(0, 256, 37, dtype=np.uint8).tobytes()
    stream = junk + REAL_FRAME * 10

    results = _push_all(Mp2FrameSynchronizer(), stream, bitrate=32)

    # The last frame is cut off by the logical frame grid shifted by the junk.
    assert len(results) == 9
    assert all(r.sync_ok and r.crc_ok and r.frame == REAL_FRAME for r in results)


def test_synchroniser_assembles_24_khz_frames_from_two_logical_frames():
    frame = silent_mp2_frame(64, lsf=True)
    assert len(frame) == 2 * 3 * 64

    synchronizer = Mp2FrameSynchronizer()
    counts = [len(synchronizer.push(bits)) for bits in logical_frames(frame * 4, 64)]

    assert counts == [0, 1, 0, 1, 0, 1, 0, 1]


def test_synchroniser_ignores_frames_of_another_bit_rate():
    """The bit rate follows from the size of the logical frames; a 32 kbit/s header in
    a 64 kbit/s subchannel is a random bit pattern, not a frame."""
    results = _push_all(Mp2FrameSynchronizer(), REAL_FRAME * 8, bitrate=64)

    assert results == []


def test_crc_error_is_reported_without_losing_the_grid():
    broken = bytearray(REAL_FRAME)
    broken[7] ^= 0x10
    stream = REAL_FRAME * 2 + bytes(broken) + REAL_FRAME * 2

    results = _push_all(Mp2FrameSynchronizer(), stream, bitrate=32)

    assert [(r.sync_ok, r.crc_ok) for r in results] == [(True, True)] * 2 + [(True, False)] + [
        (True, True)
    ] * 2


def test_broken_header_keeps_the_grid_until_too_many_misses():
    no_header = b"\x00" * len(REAL_FRAME)
    synchronizer = Mp2FrameSynchronizer()

    one_miss = _push_all(synchronizer, REAL_FRAME * 2 + no_header + REAL_FRAME, bitrate=32)
    assert [r.sync_ok for r in one_miss] == [True, True, False, True]
    assert one_miss[2].audio_format == one_miss[0].audio_format, "the lost frame keeps the format"

    lost = _push_all(synchronizer, no_header * MAX_FRAME_MISSES, bitrate=32)
    assert [r.sync_ok for r in lost] == [False] * MAX_FRAME_MISSES
    assert not synchronizer.locked, "the grid is searched afresh"

    assert all(r.sync_ok for r in _push_all(synchronizer, REAL_FRAME * 3, bitrate=32))


def test_segments_split_the_frame_for_the_layout_display():
    result = Mp2FrameResult(
        audio_format=parse_header(REAL_FRAME).audio_format,
        sync_ok=True,
        crc_ok=True,
        frame=REAL_FRAME,
        x_pad_length=10,
    )

    # header + CRC | audio data | X-PAD (10) | ScF-CRC (2) | F-PAD (2)
    assert result.segments() == [6, 82, 92, 94, 96]
