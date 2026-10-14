"""Tests for the extraction of the programme associated data (PAD)."""

from __future__ import annotations

import pytest

from dab.crc import crc16_ccitt
from dab.dls import DynamicLabelDecoder
from dab.pad import (
    XPAD_APP_DLS_CONTINUATION,
    XPAD_APP_DLS_START,
    XPAD_APP_JOURNALINE_CONTINUATION,
    XPAD_APP_JOURNALINE_START,
    XPAD_APP_MOT_CONTINUATION,
    PadReader,
    extract_mp2_pad,
    extract_pad,
    parse_x_pad,
)
from tests.mp2_test_utils import silent_mp2_frame, with_pad

# A real access unit from receiving bigFM (channel 8C). It serves as the touchstone
# for the byte order: this is not constructed but measured, how a station actually
# lays out the PAD.
REAL_ACCESS_UNIT = bytes.fromhex(
    "8028"  # DSE: id_syn_ele=4, tag=0, align=0, 40 bytes of payload
    "56dc4911bdad86f0"
    "f8cc2e8e60a491a8"
    "ce9da8ab8fa92329"
    "a3cf9668551bf2b7"
    "726d002d0dad2002"
)


def _access_unit(payload: bytes) -> bytes:
    """Wraps a PAD field in a Data Stream Element."""
    return bytes([0x80, len(payload)]) + payload


def _pad_field(x_pad: bytes, f_pad: bytes) -> bytes:
    """Lays out a PAD field as it is transmitted: X-PAD reversed, F-PAD at the end."""
    return x_pad[::-1] + f_pad


def test_access_unit_without_data_stream_element_has_no_pad():
    # The first element is a Channel Pair Element (id_syn_ele=1), not a DSE.
    assert extract_pad(bytes([0x20, 0x10, 0x00])) is None


def test_extract_pad_splits_fixed_and_extended_part():
    x_pad = bytes(range(10))
    f_pad = bytes([0x20, 0x02])
    access_unit = _access_unit(_pad_field(x_pad, f_pad))

    extracted = extract_pad(access_unit)

    assert extracted is not None
    assert extracted[0] == f_pad
    assert extracted[1] == x_pad, "the X-PAD must be turned into reading order"


def test_contents_indicators_describe_the_following_subfields():
    """A contents indicator is one byte: upper 3 bits length index, lower 5 bits
    application type (ETSI EN 300 401, clause 7.4.2.2)."""
    # Length index 1 -> 6 bytes, application 2 (DLS start)
    x_pad = bytes([0x22]) + b"HALLO!"
    f_pad = bytes([0x20, 0x02])  # variable X-PAD, contents indicators present

    subfields, last = parse_x_pad(f_pad, x_pad)

    assert len(subfields) == 1
    assert subfields[0].application_type == XPAD_APP_DLS_START
    assert subfields[0].data == b"HALLO!"
    assert last == XPAD_APP_DLS_START


def test_without_contents_indicators_the_previous_application_continues():
    """If the F-PAD announces no contents indicators, the whole field continues the
    subfield started most recently, as the continuation of its application."""
    f_pad = bytes([0x20, 0x00])  # variable, but without contents indicators

    subfields, last = parse_x_pad(f_pad, b"ABCD", last_application_type=XPAD_APP_DLS_START)

    assert [(s.application_type, s.data) for s in subfields] == [
        (XPAD_APP_DLS_CONTINUATION, b"ABCD")
    ]
    assert last == XPAD_APP_DLS_START


def test_journaline_continues_with_its_own_continuation_type():
    """Journaline (Deutschlandfunk: start 4, continuation 5, announced in FIG 0/13) must
    not repeat its start type either, or every X-PAD without contents indicators would
    restart the group."""
    f_pad = bytes([0x20, 0x00])

    subfields, last = parse_x_pad(
        f_pad,
        b"ABCD",
        last_application_type=XPAD_APP_JOURNALINE_START,
        continuation_types={XPAD_APP_JOURNALINE_START: XPAD_APP_JOURNALINE_CONTINUATION},
    )

    assert [s.application_type for s in subfields] == [XPAD_APP_JOURNALINE_CONTINUATION]
    assert last == XPAD_APP_JOURNALINE_START


def test_unsignalled_user_types_have_no_continuation_pair():
    """Types 4..11 carry whatever FIG 0/13 announces; without that, it is unknown
    whether type 4 is the start of an application (continued with 5) or itself the
    continuation of one on type 3."""
    f_pad = bytes([0x20, 0x00])

    subfields, _ = parse_x_pad(f_pad, b"ABCD", last_application_type=XPAD_APP_JOURNALINE_START)

    assert [s.application_type for s in subfields] == [XPAD_APP_JOURNALINE_START]


def test_pad_reader_uses_the_signalled_continuation_types():
    reader = PadReader()
    reader.set_signalled_applications({6: 7})
    started = bytes([0x20, 0x02])  # indicators present
    continued = bytes([0x20, 0x00])
    x_pad = bytes([(0 << 5) | 6]) + b"ABCD"

    reader._read(started, x_pad, continuation_length=None)
    subfields = reader._read(continued, b"EFGH", continuation_length=None)

    assert [(s.application_type, s.data) for s in subfields] == [(7, b"EFGH")]


def test_without_contents_indicators_and_without_history_nothing_is_read():
    subfields, last = parse_x_pad(bytes([0x20, 0x00]), b"ABCD")
    assert subfields == []
    assert last is None


def test_no_x_pad_announced():
    assert parse_x_pad(bytes([0x00, 0x02]), b"ABCD") == ([], None)


def test_short_x_pad_is_limited_to_four_bytes():
    f_pad = bytes([0x10, 0x02])  # X-PAD indicator 1 = short X-PAD
    # Length index 0 -> 4 bytes, but only 3 fit behind the indicator.
    subfields, _ = parse_x_pad(f_pad, bytes([0x02]) + b"ABCDEFGH")

    assert len(subfields) == 1
    assert subfields[0].data == b"ABC", "only 3 of the 4 bytes remain behind the indicator"


def test_real_access_unit_from_broadcast():
    """Checked against real reception: the lengths announced in the X-PAD must add up
    to the length of the field to the byte. If the byte order were wrong, this sum
    would not work out."""
    extracted = extract_pad(REAL_ACCESS_UNIT)
    assert extracted is not None
    f_pad, x_pad = extracted

    assert f_pad == bytes([0x20, 0x02])  # variable X-PAD with contents indicators
    assert len(x_pad) == 38

    subfields, _ = parse_x_pad(f_pad, x_pad)

    assert [s.application_type for s in subfields] == [XPAD_APP_MOT_CONTINUATION] * 3
    assert [len(s.data) for s in subfields] == [24, 4, 6]
    # 4 bytes of contents indicators (three plus the end marker) + 34 bytes of data = 38
    assert 4 + sum(len(s.data) for s in subfields) == len(x_pad)


def test_pad_reader_remembers_the_application_across_access_units():
    reader = PadReader()
    first = _access_unit(_pad_field(bytes([0x22]) + b"START!", bytes([0x20, 0x02])))
    second = _access_unit(_pad_field(b"MEHR", bytes([0x20, 0x00])))

    assert reader.read(first)[0].application_type == XPAD_APP_DLS_START
    continued = reader.read(second)

    assert [(s.application_type, s.data) for s in continued] == [
        (XPAD_APP_DLS_CONTINUATION, b"MEHR")
    ]

    reader.reset()
    assert reader.read(second) == [], "after reset() the history is gone"


def test_x_pad_without_contents_indicators_does_not_restart_the_data_group():
    """A data group begins in a field with contents indicators and ends in one without.
    The second field continues the group; were it read as a new start, the bytes
    collected so far would be discarded and the label would never complete."""
    body = bytes([0x60 | (len(b"Radio Test") - 1), 0x00]) + b"Radio Test"
    crc = crc16_ccitt(body) ^ 0xFFFF
    group = body + bytes([crc >> 8, crc & 0xFF])  # 14 bytes: prefix, text, CRC
    # Length index 2 -> 8 bytes of DLS start, the remaining 6 bytes follow without CI.
    first = _access_unit(_pad_field(bytes([0x42]) + group[:8], bytes([0x20, 0x02])))
    second = _access_unit(_pad_field(group[8:], bytes([0x20, 0x00])))
    reader = PadReader()
    decoder = DynamicLabelDecoder()

    assert decoder.process(reader.read(first)) is None
    assert decoder.process(reader.read(second)).text == "Radio Test"


@pytest.mark.parametrize("access_unit", [b"", b"\x80", bytes([0x80, 0x01, 0x00])])
def test_truncated_input_is_rejected(access_unit):
    assert extract_pad(access_unit) is None


# --- Classic DAB: PAD at the end of the MP2 frame ------------------------------------

# 48 kHz, 128 kbit/s joint stereo would be more typical, but 64 kbit/s mono keeps the
# frames short; what matters is the 4-byte ScF-CRC of this format.
MP2_BITRATE = 64
MP2_SCF_CRC_LENGTH = 4


def test_extract_mp2_pad_reads_backwards_from_the_scale_factor_crc():
    x_pad = bytes(range(1, 11))
    f_pad = bytes([0x20, 0x02])
    frame = with_pad(silent_mp2_frame(MP2_BITRATE), x_pad, f_pad, MP2_SCF_CRC_LENGTH)

    extracted = extract_mp2_pad(frame, MP2_SCF_CRC_LENGTH)

    assert extracted is not None
    assert extracted[0] == f_pad
    # The X-PAD length is unknown here, so the area reaches back to the frame start.
    assert extracted[1][: len(x_pad)] == x_pad
    assert len(extracted[1]) == len(frame) - 2 - MP2_SCF_CRC_LENGTH


def test_mp2_x_pad_ends_where_the_contents_indicators_say():
    """In the MP2 frame nothing but the contents indicators delimit the X-PAD: one
    indicator (6 bytes DLS), the end marker, then the data. The audio bytes in front
    must not leak into the subfield."""
    x_pad = bytes([0x22, 0x00]) + b"START!"
    frame = with_pad(silent_mp2_frame(MP2_BITRATE), x_pad, bytes([0x20, 0x02]), MP2_SCF_CRC_LENGTH)
    reader = PadReader()

    subfields = reader.read_mp2_frame(frame, MP2_SCF_CRC_LENGTH)

    assert [(s.application_type, s.data) for s in subfields] == [(XPAD_APP_DLS_START, b"START!")]
    assert reader.x_pad_length == 8


def test_mp2_x_pad_without_contents_indicators_is_as_long_as_the_previous_one():
    """Without contents indicators the X-PAD continues the previous subfield with the
    same length as the previous X-PAD field (ETSI EN 300 401, clause 7.4.2.1)."""
    reader = PadReader()
    first = with_pad(
        silent_mp2_frame(MP2_BITRATE),
        bytes([0x22, 0x00]) + b"START!",
        bytes([0x20, 0x02]),
        MP2_SCF_CRC_LENGTH,
    )
    second = with_pad(
        silent_mp2_frame(MP2_BITRATE), b"WEITER..", bytes([0x20, 0x00]), MP2_SCF_CRC_LENGTH
    )

    reader.read_mp2_frame(first, MP2_SCF_CRC_LENGTH)
    continued = reader.read_mp2_frame(second, MP2_SCF_CRC_LENGTH)

    assert [(s.application_type, s.data) for s in continued] == [
        (XPAD_APP_DLS_CONTINUATION, b"WEITER..")
    ]
    assert reader.x_pad_length == 8


def test_mp2_frame_without_x_pad():
    reader = PadReader()
    frame = with_pad(silent_mp2_frame(MP2_BITRATE), b"", bytes([0x00, 0x00]), MP2_SCF_CRC_LENGTH)

    assert reader.read_mp2_frame(frame, MP2_SCF_CRC_LENGTH) == []
    assert reader.x_pad_length == 0
