"""Tests for the RTCM 3 stream of GEPOS: frames, CRC-24Q, ephemerides, SSRZ message
types and the stream across data groups."""

from __future__ import annotations

import pytest

from dab.crc import crc16_ccitt
from dab.data_group import parse_data_group
from dab.rtcm import (
    GNSS_BEIDOU,
    GNSS_GALILEO,
    GNSS_GPS,
    GlonassEphemeris,
    KeplerEphemeris,
    RtcmFramer,
    RtcmStreamDecoder,
    crc24q,
    decode_ephemeris,
    message_kind,
    message_label,
    message_name,
    message_type,
    ssrz_message_number,
)
from tests.rtcm_vectors import (
    BEIDOU_C08_FRAME,
    DATA_GROUPS,
    GALILEO_E02_FRAME,
    GLONASS_R11_FRAME,
    GPS_G12_FRAME,
    SSRZ_ZM001_FRAME,
    SSRZ_ZM003_START,
    SSRZ_ZM012_START,
)
from tests.ssrz_vectors import EPHEMERIS_FRAMES


def _message(frame: bytes) -> bytes:
    return frame[3:-3]


def test_crc24q_check_value() -> None:
    # The usual check value of CRC-24Q (also known as CRC-24/LTE-A).
    assert crc24q(b"123456789") == 0xCDE703


def test_crc24q_of_real_frames() -> None:
    for frame in (GPS_G12_FRAME, GLONASS_R11_FRAME, SSRZ_ZM001_FRAME):
        assert crc24q(frame[:-3]) == int.from_bytes(frame[-3:], "big")


def test_framer_finds_frames_behind_garbage_and_across_pushes() -> None:
    framer = RtcmFramer()
    stream = b"\x00\xd3\x12" + GPS_G12_FRAME + SSRZ_ZM001_FRAME
    messages = []
    for byte in stream:
        messages += framer.push(bytes([byte]))
    assert messages == [_message(GPS_G12_FRAME), _message(SSRZ_ZM001_FRAME)]
    assert framer.frames == 2
    assert framer.frame_bytes == len(GPS_G12_FRAME) + len(SSRZ_ZM001_FRAME)
    assert framer.discarded_bytes == 3
    # A false preamble while searching is no CRC error.
    assert framer.crc_errors == 0


def test_framer_counts_a_broken_frame_right_after_a_good_one() -> None:
    framer = RtcmFramer()
    broken = bytearray(GLONASS_R11_FRAME)
    broken[10] ^= 0x01
    messages = framer.push(GPS_G12_FRAME + bytes(broken) + SSRZ_ZM001_FRAME)
    assert messages == [_message(GPS_G12_FRAME), _message(SSRZ_ZM001_FRAME)]
    assert framer.crc_errors == 1
    assert framer.discarded_bytes == len(broken)


def test_framer_reset_drops_the_frame_under_way() -> None:
    framer = RtcmFramer()
    assert framer.push(GPS_G12_FRAME[:30]) == []
    framer.reset()
    assert framer.discarded_bytes == 30
    assert framer.push(GPS_G12_FRAME[30:] + SSRZ_ZM001_FRAME) == [_message(SSRZ_ZM001_FRAME)]


def test_gps_ephemeris() -> None:
    ephemeris = decode_ephemeris(_message(GPS_G12_FRAME))
    assert isinstance(ephemeris, KeplerEphemeris)
    assert (ephemeris.system, ephemeris.prn, ephemeris.name) == (GNSS_GPS, 12, "G12")
    assert (ephemeris.week, ephemeris.iod, ephemeris.toe) == (390, 61, 158400.0)
    assert ephemeris.sqrt_a == pytest.approx(5153.680686950684, abs=1e-9)
    assert ephemeris.eccentricity == pytest.approx(0.008823461132124066, abs=1e-15)
    assert ephemeris.crc == 126.09375
    assert ephemeris.healthy


def test_galileo_ephemeris() -> None:
    ephemeris = decode_ephemeris(_message(GALILEO_E02_FRAME))
    assert isinstance(ephemeris, KeplerEphemeris)
    assert (ephemeris.system, ephemeris.name) == (GNSS_GALILEO, "E02")
    assert (ephemeris.week, ephemeris.iod, ephemeris.toe) == (1414, 5, 155400.0)
    assert ephemeris.sqrt_a == pytest.approx(5440.639200210571, abs=1e-9)
    assert ephemeris.crs == 177.4375
    assert ephemeris.healthy


def test_beidou_ephemeris() -> None:
    ephemeris = decode_ephemeris(_message(BEIDOU_C08_FRAME))
    assert isinstance(ephemeris, KeplerEphemeris)
    assert (ephemeris.system, ephemeris.name) == (GNSS_BEIDOU, "C08")
    assert (ephemeris.week, ephemeris.iod, ephemeris.toe) == (1082, 1, 151200.0)
    # 18 bit harmonic terms with a resolution of 2^-6 m.
    assert ephemeris.crs == 345.015625
    assert ephemeris.crc == 695.90625
    assert ephemeris.healthy


def test_glonass_ephemeris() -> None:
    ephemeris = decode_ephemeris(_message(GLONASS_R11_FRAME))
    assert isinstance(ephemeris, GlonassEphemeris)
    assert (ephemeris.name, ephemeris.slot, ephemeris.frequency_channel) == ("R11", 11, 0)
    assert ephemeris.tb == 80100  # 22:15 Moscow time = 19:15 UTC
    assert ephemeris.iod == 89
    assert ephemeris.position == (12666517.08984375, 7347463.37890625, 20890970.703125)
    assert ephemeris.velocity[2] == pytest.approx(-1511.2581253051758)
    # Sign and magnitude: -1 * 2^-30 km/s^2.
    assert ephemeris.acceleration[0] == pytest.approx(-(2.0**-30) * 1000)
    # The most significant bit of Bn is set in the broadcast.
    assert not ephemeris.healthy


def test_other_and_truncated_messages_are_no_ephemeris() -> None:
    assert decode_ephemeris(_message(SSRZ_ZM001_FRAME)) is None
    assert decode_ephemeris(_message(GPS_G12_FRAME)[:-1]) is None
    assert decode_ephemeris(b"") is None


def test_ssrz_message_numbers() -> None:
    assert message_type(_message(SSRZ_ZM001_FRAME)) == 4090
    assert ssrz_message_number(_message(SSRZ_ZM001_FRAME)) == 1
    assert ssrz_message_number(SSRZ_ZM003_START) == 3
    assert ssrz_message_number(SSRZ_ZM012_START) == 12
    assert ssrz_message_number(_message(GPS_G12_FRAME)) is None
    # Another Geo++ sub-type is no SSRZ.
    other = bytearray(SSRZ_ZM003_START)
    other[1] = (other[1] & 0xF0) | 0x06
    assert ssrz_message_number(bytes(other)) is None


def test_message_names() -> None:
    assert message_kind(_message(GPS_G12_FRAME)) == (1019, None)
    assert message_name((1019, None)) == "GPS ephemerides"
    assert message_label((4090, 3)) == "4090 · ZM003"
    assert message_name((4090, 3)) == "SSRZ: gridded ionosphere"
    assert message_name((4090, 10)) == "SSRZ: stream identification"
    assert message_name((4090, 20)) == "SSRZ: not described in version 1.1.3"
    assert message_name((4090, None)) == "Geo++ proprietary (no SSRZ)"
    assert message_name((1234, None)) == "unknown"


def _groups():
    groups = [parse_data_group(raw) for raw in DATA_GROUPS]
    assert [group.continuity_index for group in groups] == [14, 15, 0]
    return groups


def test_stream_across_data_groups() -> None:
    decoder = RtcmStreamDecoder()
    new = []
    for group in _groups():
        new += decoder.push(group)
    names = [ephemeris.name for ephemeris in new]
    # The E02 frame starts in the first data group and ends in the second.
    assert "E02" in names
    statistics = decoder.statistics(1.0)
    assert statistics.data_groups == 3
    assert statistics.lost_data_groups == 0
    assert statistics.crc_errors == 0
    assert statistics.frames == sum(s.count for s in statistics.messages.values())
    assert statistics.ephemerides == len(set(names))
    # Only the bytes of the frame still under way at the end are outside a frame.
    assert statistics.discarded_bytes == 0


def test_repeated_data_group_and_repeated_ephemeris_are_ignored() -> None:
    decoder = RtcmStreamDecoder()
    first, second, _ = _groups()
    decoder.push(first)
    assert decoder.push(first) == []  # same continuity index: a repetition
    assert decoder.statistics(1.0).data_groups == 1
    assert [e.name for e in decoder.push(second)] == ["E02", "E03"]

    decoder = RtcmStreamDecoder()
    assert decoder.push(parse_data_group(_wrap(GPS_G12_FRAME, 0)))[0].name == "G12"
    # The same ephemeris again, in the next data group: counted, not reported again.
    assert decoder.push(parse_data_group(_wrap(GPS_G12_FRAME, 1))) == []
    assert decoder.statistics(1.0).messages[(1019, None)].count == 2


def test_lost_data_group_drops_the_frame_under_way() -> None:
    decoder = RtcmStreamDecoder()
    first, _, third = _groups()
    decoder.push(first)
    new = decoder.push(third)
    statistics = decoder.statistics(1.0)
    assert statistics.lost_data_groups == 1
    assert statistics.crc_errors == 0
    assert "E02" not in [e.name for e in new]


def _wrap(data: bytes, continuity: int) -> bytes:
    """A TDC data group (type 0, with CRC) around `data`."""
    raw = bytes([0x40, continuity << 4]) + data
    return raw + (crc16_ccitt(raw) ^ 0xFFFF).to_bytes(2, "big")


def test_frame_without_message_is_skipped() -> None:
    empty = b"\xd3\x00\x00"
    empty += crc24q(empty).to_bytes(3, "big")
    decoder = RtcmStreamDecoder()
    decoder.push(parse_data_group(_wrap(empty + GPS_G12_FRAME, 0)))
    statistics = decoder.statistics(1.0)
    assert statistics.frames == 2
    assert list(statistics.messages) == [(1019, None)]


def test_clock_parameters_and_ssr_iod() -> None:
    """The broadcast clock and the IOD by which SSR corrections name an ephemeris (the
    values were checked against pyrtcm for all 4 280 ephemerides of a recording)."""
    g06, r02, e03, c11 = (
        decode_ephemeris(EPHEMERIS_FRAMES[name][3:-3]) for name in ("G06", "R02", "E03", "C11")
    )
    assert (g06.toc, g06.ssr_iod) == (244800.0, 81)
    assert g06.af0 == pytest.approx(-5.072266794741154e-4, rel=1e-12)
    assert g06.af1 == pytest.approx(1.2391865311656147e-11, rel=1e-12)
    # GLONASS: tb in quarter hours, tau_n and the frequency channel.
    assert (r02.tb, r02.ssr_iod, r02.frequency_channel) == (78300, 87, -4)
    assert r02.tau_n == pytest.approx(-1.1774711310863495e-5, rel=1e-12)
    assert (e03.iod, e03.ssr_iod, e03.toc) == (17, 17, 238800.0)
    # BeiDou: the AODE is 1, the SSR IOD comes from toc: 237600 / 720 = 330, mod 240 = 90.
    assert (c11.iod, c11.ssr_iod, c11.toc) == (1, 90, 237600.0)
    assert c11.af0 == pytest.approx(-9.263607207685709e-4, rel=1e-12)
