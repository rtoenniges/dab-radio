"""Tests for MSC data groups and their assembly from the X-PAD."""

from __future__ import annotations

from dab.crc import crc16_ccitt
from dab.data_group import XPadDataGroupAssembler, parse_data_group, read_length_indicator
from dab.pad import (
    XPAD_APP_DATA_GROUP_LENGTH,
    XPAD_APP_JOURNALINE_CONTINUATION,
    XPAD_APP_JOURNALINE_START,
    XPAD_APP_MOT_CONTINUATION,
    XPAD_APP_MOT_START,
    XPadSubfield,
)

# The Journaline main menu of Deutschlandfunk, received on channel 5C.
JOURNALINE_MENU_GROUP = bytes.fromhex(
    "40d0000031011a15004a4576312e382e325f463630453137354430443041446575747363686c616e64"
    "66756e6b0217704e6163687269636874656e021f4050726f6772616d6d766f7273636861750200014b"
    "6f6e74616b74020002496d7072657373756d2a03"
)


def _with_crc(data: bytes) -> bytes:
    crc = crc16_ccitt(data) ^ 0xFFFF
    return data + bytes([crc >> 8, crc & 0xFF])


def _length_indicator(length: int) -> XPadSubfield:
    return XPadSubfield(XPAD_APP_DATA_GROUP_LENGTH, _with_crc(bytes([length >> 8, length & 0xFF])))


def _mot_group(transport_id: int, payload: bytes) -> bytes:
    """A data group with segment field, transport identifier and CRC, as MOT uses them."""
    header = bytes([0x40 | 0x20 | 0x10 | 4, 0x00])
    segment_field = bytes([0x80 | 0x00, 5])
    user_access = bytes([0x10 | 2, transport_id >> 8, transport_id & 0xFF])
    return _with_crc(header + segment_field + user_access + payload)


def test_parse_data_group_reads_segment_and_transport_id():
    parsed = parse_data_group(_mot_group(4711, b"abc"))

    assert parsed is not None
    assert parsed.group_type == 4
    assert parsed.has_segment_field and parsed.has_user_access
    assert parsed.transport_id == 4711
    assert parsed.segment_number == 5
    assert parsed.last_segment is True
    assert parsed.data == b"abc"


def test_data_group_with_broken_crc_is_rejected():
    group = bytearray(_mot_group(1, b"abc"))
    group[-1] ^= 0xFF
    assert parse_data_group(bytes(group)) is None


def test_parse_data_group_reads_the_header_of_a_journaline_group():
    """Journaline uses the plain form: no extension, segment or user access field,
    only the CRC (ETSI TS 102 979, clause 8.1.1)."""
    parsed = parse_data_group(JOURNALINE_MENU_GROUP)

    assert parsed is not None
    assert parsed.group_type == 0
    assert parsed.has_crc
    assert not (parsed.has_extension or parsed.has_segment_field or parsed.has_user_access)
    assert parsed.continuity_index == 0xD
    assert parsed.repetition_index == 0
    assert parsed.transport_id is None
    assert parsed.data == JOURNALINE_MENU_GROUP[2:-2]


def test_length_indicator_carries_its_own_crc():
    indicator = _length_indicator(102).data
    assert read_length_indicator(indicator) == 102

    broken = bytearray(indicator)
    broken[1] ^= 0x01
    assert read_length_indicator(bytes(broken)) is None


def test_group_is_assembled_from_start_and_continuations():
    assembler = XPadDataGroupAssembler(XPAD_APP_JOURNALINE_START, XPAD_APP_JOURNALINE_CONTINUATION)
    group = JOURNALINE_MENU_GROUP
    subfields = [_length_indicator(len(group)), XPadSubfield(XPAD_APP_JOURNALINE_START, group[:16])]
    subfields += [
        XPadSubfield(XPAD_APP_JOURNALINE_CONTINUATION, group[offset : offset + 16])
        for offset in range(16, len(group), 16)
    ]

    completed = []
    for subfield in subfields:  # one subfield per access unit
        completed += assembler.process([subfield])

    assert completed == [group]


def test_padding_behind_the_announced_length_is_cut_off():
    assembler = XPadDataGroupAssembler(XPAD_APP_MOT_START, XPAD_APP_MOT_CONTINUATION)
    group = _mot_group(3, b"payload")

    completed = assembler.process(
        [_length_indicator(len(group)), XPadSubfield(XPAD_APP_MOT_START, group + bytes(5))]
    )

    assert completed == [group]


def test_group_without_length_indicator_is_skipped():
    """Without the length its end cannot be found; the group is dropped instead of
    being collected until the next one starts."""
    assembler = XPadDataGroupAssembler(XPAD_APP_MOT_START, XPAD_APP_MOT_CONTINUATION)
    group = _mot_group(3, b"payload")

    assert assembler.process([XPadSubfield(XPAD_APP_MOT_START, group)]) == []
    # The next indicator belongs to the next group, not to the one already running.
    assert assembler.process([_length_indicator(len(group))]) == []
    assert assembler.process([XPadSubfield(XPAD_APP_MOT_CONTINUATION, group)]) == []


def test_continuations_without_start_are_ignored():
    """Joined in the middle of a group: nothing until the next start."""
    assembler = XPadDataGroupAssembler(XPAD_APP_MOT_START, XPAD_APP_MOT_CONTINUATION)
    group = _mot_group(3, b"payload")

    assert (
        assembler.process(
            [_length_indicator(len(group)), XPadSubfield(XPAD_APP_MOT_CONTINUATION, group)]
        )
        == []
    )
    assert assembler.process([XPadSubfield(XPAD_APP_MOT_START, group)]) == [group]


def test_indicator_for_the_next_group_does_not_cut_the_running_one_short():
    """The length indicator applies to the next group that starts. An indicator that
    arrives while a group is collected, here for a group of another application,
    must leave the running group alone."""
    assembler = XPadDataGroupAssembler(XPAD_APP_JOURNALINE_START, XPAD_APP_JOURNALINE_CONTINUATION)
    group = JOURNALINE_MENU_GROUP
    other = _mot_group(9, b"x" * 40)

    assert (
        assembler.process(
            [_length_indicator(len(group)), XPadSubfield(XPAD_APP_JOURNALINE_START, group[:48])]
        )
        == []
    )
    assert (
        assembler.process(
            [_length_indicator(len(other)), XPadSubfield(XPAD_APP_MOT_START, other[:24])]
        )
        == []
    )
    completed = assembler.process([XPadSubfield(XPAD_APP_JOURNALINE_CONTINUATION, group[48:])])

    assert completed == [group]


def test_two_applications_share_the_length_indicators():
    """Deutschlandfunk sends slideshow and Journaline in the same X-PAD: every
    application gets its own groups from the common stream of subfields."""
    mot = XPadDataGroupAssembler(XPAD_APP_MOT_START, XPAD_APP_MOT_CONTINUATION)
    journaline = XPadDataGroupAssembler(XPAD_APP_JOURNALINE_START, XPAD_APP_JOURNALINE_CONTINUATION)
    image_group = _mot_group(7, b"y" * 30)
    stream = [
        [_length_indicator(len(image_group)), XPadSubfield(XPAD_APP_MOT_START, image_group[:20])],
        [XPadSubfield(XPAD_APP_MOT_CONTINUATION, image_group[20:])],
        [
            _length_indicator(len(JOURNALINE_MENU_GROUP)),
            XPadSubfield(XPAD_APP_JOURNALINE_START, JOURNALINE_MENU_GROUP[:60]),
        ],
        [XPadSubfield(XPAD_APP_JOURNALINE_CONTINUATION, JOURNALINE_MENU_GROUP[60:])],
    ]

    mot_groups, journaline_groups = [], []
    for subfields in stream:
        mot_groups += mot.process(subfields)
        journaline_groups += journaline.process(subfields)

    assert mot_groups == [image_group]
    assert journaline_groups == [JOURNALINE_MENU_GROUP]


def test_reset_discards_a_partial_group():
    assembler = XPadDataGroupAssembler(XPAD_APP_MOT_START, XPAD_APP_MOT_CONTINUATION)
    group = _mot_group(3, b"payload")
    assembler.process([_length_indicator(len(group)), XPadSubfield(XPAD_APP_MOT_START, group[:4])])
    assembler.reset()

    assert assembler.process([XPadSubfield(XPAD_APP_MOT_CONTINUATION, group[4:])]) == []
