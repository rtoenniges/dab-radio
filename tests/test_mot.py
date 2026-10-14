"""Tests for the MOT decoding (slideshow images)."""

from __future__ import annotations

from dab.crc import crc16_ccitt
from dab.mot import (
    CONTENT_TYPE_IMAGE,
    CONTENT_TYPE_MOT_TRANSPORT,
    DATA_GROUP_MOT_BODY,
    DATA_GROUP_MOT_HEADER,
    MAX_PARTIAL_OBJECTS,
    MOT_TRANSPORT_HEADER_UPDATE,
    MotDecoder,
    decode_content_name,
    parse_mot_header,
)
from dab.pad import (
    XPAD_APP_DATA_GROUP_LENGTH,
    XPAD_APP_MOT_CONTINUATION,
    XPAD_APP_MOT_START,
    XPadSubfield,
)

JPEG = b"\xff\xd8\xff\xe0" + bytes(range(256)) * 2 + b"\xff\xd9"


def _with_crc(data: bytes) -> bytes:
    crc = crc16_ccitt(data) ^ 0xFFFF
    return data + bytes([crc >> 8, crc & 0xFF])


def _segment(payload: bytes, repetition: int = 7) -> bytes:
    """Prepends the segment header: 3 bits repetition, 13 bits size."""
    return bytes([(repetition << 5) | (len(payload) >> 8), len(payload) & 0xFF]) + payload


def _data_group(
    group_type: int, transport_id: int, segment_number: int, last: bool, payload: bytes
) -> bytes:
    """Builds an MSC data group with segment field, transport identifier and CRC."""
    header = bytes([0x40 | 0x20 | 0x10 | group_type, 0x00])
    segment_field = bytes([(0x80 if last else 0) | (segment_number >> 8), segment_number & 0xFF])
    user_access = bytes([0x10 | 2, transport_id >> 8, transport_id & 0xFF])
    return _with_crc(header + segment_field + user_access + _segment(payload))


def _mot_header(
    body_size: int,
    subtype: int = 1,
    name: str | None = None,
    *,
    content_type: int = CONTENT_TYPE_IMAGE,
    extra: bytes = b"",
) -> bytes:
    extension = b""
    if name is not None:
        value = bytes([0x00]) + name.encode()  # character set 0 (ASCII part)
        extension = bytes([0xCC, len(value)]) + value  # PLI=3 variable, ParamId=0x0C
    extension += extra
    header_size = 7 + len(extension)
    core = (body_size << 28) | (header_size << 15) | (content_type << 9) | subtype
    return core.to_bytes(7, "big") + extension


def _header_update(name: str, extra: bytes = b"") -> bytes:
    """Header of a header update: content type "MOT transport", subtype 0, body size
    0 (ETSI EN 301 234, clause 7.1.3, figure 29)."""
    return _mot_header(
        0,
        MOT_TRANSPORT_HEADER_UPDATE,
        name,
        content_type=CONTENT_TYPE_MOT_TRANSPORT,
        extra=extra,
    )


def _length_indicator(length: int) -> bytes:
    return _with_crc(bytes([length >> 8, length & 0xFF]))


def _feed_group(decoder: MotDecoder, group: bytes, chunk: int = 0) -> list:
    """Feeds a data group in through the X-PAD route: first the length indicator,
    then start and continuations."""
    decoder.process([XPadSubfield(XPAD_APP_DATA_GROUP_LENGTH, _length_indicator(len(group)))])
    if chunk == 0:
        return decoder.process([XPadSubfield(XPAD_APP_MOT_START, group)])
    result = decoder.process([XPadSubfield(XPAD_APP_MOT_START, group[:chunk])])
    for offset in range(chunk, len(group), chunk):
        result = decoder.process(
            [XPadSubfield(XPAD_APP_MOT_CONTINUATION, group[offset : offset + chunk])]
        )
    return result


def test_parse_mot_header_reads_size_type_and_name():
    header = _mot_header(body_size=34769, subtype=1, name="12.JPG")

    parsed = parse_mot_header(header)

    assert parsed is not None
    assert parsed.body_size == 34769
    assert parsed.content_type == CONTENT_TYPE_IMAGE
    assert parsed.content_subtype == 1
    assert parsed.content_name == "12.JPG"


def test_unknown_header_parameters_are_kept_unchanged():
    """Unknown parameters must not get lost; their interpretation happens one layer
    up (see `dab.slideshow`)."""
    # Appended cleanly: PLI 3 (variable), ParamId 0x25, length 2
    header = _mot_header(body_size=10, name="x.jpg")
    extra = bytes([0xC0 | 0x25, 2, 0x07, 0x2A])
    core = int.from_bytes(header[:7], "big")
    header_size = ((core >> 15) & 0x1FFF) + len(extra)
    core = (core & ~(0x1FFF << 15)) | (header_size << 15)
    header = core.to_bytes(7, "big") + header[7:] + extra

    parsed = parse_mot_header(header)

    assert parsed is not None
    assert parsed.parameters[0x25] == bytes([0x07, 0x2A])
    assert parsed.content_name == "x.jpg"


def test_complete_image_is_assembled():
    decoder = MotDecoder()
    header = _mot_header(len(JPEG), name="bild.jpg")

    assert _feed_group(decoder, _data_group(DATA_GROUP_MOT_HEADER, 7, 0, True, header)) == []
    finished = _feed_group(decoder, _data_group(DATA_GROUP_MOT_BODY, 7, 0, True, JPEG))

    assert len(finished) == 1
    obj = finished[0]
    assert obj.is_image
    assert obj.content_name == "bild.jpg"
    assert obj.body == JPEG
    assert obj.transport_id == 7


def test_image_assembled_from_several_body_segments():
    decoder = MotDecoder()
    _feed_group(decoder, _data_group(DATA_GROUP_MOT_HEADER, 9, 0, True, _mot_header(len(JPEG))))

    third = len(JPEG) // 3
    parts = [JPEG[:third], JPEG[third : 2 * third], JPEG[2 * third :]]
    finished: list = []
    for index, part in enumerate(parts):
        finished = _feed_group(
            decoder,
            _data_group(DATA_GROUP_MOT_BODY, 9, index, index == len(parts) - 1, part),
            chunk=17,  # spread over several access units
        )

    assert len(finished) == 1
    assert finished[0].body == JPEG


def test_missing_body_segment_yields_nothing():
    decoder = MotDecoder()
    _feed_group(decoder, _data_group(DATA_GROUP_MOT_HEADER, 3, 0, True, _mot_header(len(JPEG))))

    # Segment 0 is missing, only the last one arrives.
    finished = _feed_group(decoder, _data_group(DATA_GROUP_MOT_BODY, 3, 1, True, JPEG))

    assert finished == []


def test_body_without_header_yields_nothing():
    """Joining in the middle of a transmission, the header is missing: then there is
    no object until the next pass comes."""
    decoder = MotDecoder()
    assert _feed_group(decoder, _data_group(DATA_GROUP_MOT_BODY, 5, 0, True, JPEG)) == []


def test_segment_header_is_stripped():
    """Every MOT segment starts with two bytes of segment header. Left in place they
    would precede the image file and the header would be unreadable; exactly that is
    where the first version failed on real reception."""
    decoder = MotDecoder()
    _feed_group(decoder, _data_group(DATA_GROUP_MOT_HEADER, 11, 0, True, _mot_header(len(JPEG))))
    finished = _feed_group(decoder, _data_group(DATA_GROUP_MOT_BODY, 11, 0, True, JPEG))

    assert finished[0].body.startswith(b"\xff\xd8\xff"), "the image file must start directly"


def test_length_indicator_with_broken_crc_blocks_the_group():
    decoder = MotDecoder()
    broken = bytearray(_length_indicator(20))
    broken[-1] ^= 0xFF
    decoder.process([XPadSubfield(XPAD_APP_DATA_GROUP_LENGTH, bytes(broken))])

    group = _data_group(DATA_GROUP_MOT_HEADER, 1, 0, True, _mot_header(len(JPEG)))
    assert decoder.process([XPadSubfield(XPAD_APP_MOT_START, group)]) == []


def test_reset_discards_partial_objects():
    decoder = MotDecoder()
    _feed_group(decoder, _data_group(DATA_GROUP_MOT_HEADER, 2, 0, True, _mot_header(len(JPEG))))
    decoder.reset()

    assert _feed_group(decoder, _data_group(DATA_GROUP_MOT_BODY, 2, 0, True, JPEG)) == []


def test_unfinished_objects_touched_longest_ago_are_dropped():
    """An object that lost a segment to a CRC error stays unfinished. The decoder used
    to keep it until the next reset, for hours on one station; now the objects of
    later transport identifiers push it out."""
    decoder = MotDecoder()
    _feed_group(decoder, _data_group(DATA_GROUP_MOT_HEADER, 2, 0, True, _mot_header(len(JPEG))))
    for transport_id in range(100, 100 + MAX_PARTIAL_OBJECTS):
        header = _data_group(DATA_GROUP_MOT_HEADER, transport_id, 0, True, _mot_header(len(JPEG)))
        assert _feed_group(decoder, header) == []

    # The body of object 2 comes too late: its header is gone, nothing is completed.
    assert _feed_group(decoder, _data_group(DATA_GROUP_MOT_BODY, 2, 0, True, JPEG)) == []
    # The newest unfinished object is untouched by that.
    newest = 100 + MAX_PARTIAL_OBJECTS - 1
    finished = _feed_group(decoder, _data_group(DATA_GROUP_MOT_BODY, newest, 0, True, JPEG))
    assert [obj.transport_id for obj in finished] == [newest]


def test_a_segment_keeps_its_unfinished_object_from_being_dropped():
    """Not the oldest by first segment goes, but the one without a segment for the
    longest: an object still being transmitted survives the header updates and
    objects inserted in between."""
    decoder = MotDecoder()
    half = len(JPEG) // 2
    _feed_group(decoder, _data_group(DATA_GROUP_MOT_HEADER, 2, 0, True, _mot_header(len(JPEG))))
    for transport_id in range(100, 100 + MAX_PARTIAL_OBJECTS - 1):
        header = _data_group(DATA_GROUP_MOT_HEADER, transport_id, 0, True, _mot_header(len(JPEG)))
        _feed_group(decoder, header)
    assert _feed_group(decoder, _data_group(DATA_GROUP_MOT_BODY, 2, 0, False, JPEG[:half])) == []
    # One more object: the limit is exceeded, and object 100 goes, not object 2.
    _feed_group(decoder, _data_group(DATA_GROUP_MOT_HEADER, 200, 0, True, _mot_header(len(JPEG))))

    finished = _feed_group(decoder, _data_group(DATA_GROUP_MOT_BODY, 2, 1, True, JPEG[half:]))
    assert [obj.transport_id for obj in finished] == [2]
    assert finished[0].body == JPEG
    assert _feed_group(decoder, _data_group(DATA_GROUP_MOT_BODY, 100, 0, True, JPEG)) == []


def test_content_name_starts_with_a_charset_byte():
    """Unlike the slideshow's text parameters, the ContentName carries a character set
    byte (ETSI EN 301 234, 6.2.2.1.1). Both values were received exactly like this:
    0x00 at ffn, 0x40 at NDR."""
    assert decode_content_name(bytes([0x00]) + b"12.JPG") == "12.JPG"
    assert (
        decode_content_name(bytes([0x40]) + b"20260919_015338Z_NEWS_52400.jpg")
        == "20260919_015338Z_NEWS_52400.jpg"
    )
    assert decode_content_name(None) is None


# --- Header updates (ETSI EN 301 234, clause 7.1.3; TS 101 499, 6.3) ------------


def test_header_update_is_complete_without_a_body():
    """A header update has body size 0; body segments never arrive. The first
    version of the decoder waited for them anyway, so updates never reached the
    slideshow."""
    decoder = MotDecoder()
    finished = _feed_group(
        decoder,
        _data_group(DATA_GROUP_MOT_HEADER, 40, 0, True, _header_update("bild.jpg")),
    )

    assert len(finished) == 1
    update = finished[0]
    assert update.is_header_update
    assert not update.is_image
    assert update.content_name == "bild.jpg"
    assert update.body == b""


def test_header_update_keeps_its_parameters():
    decoder = MotDecoder()
    # PLI=3 (length byte follows), ParamId 0x25, two bytes: category 3, slide 7.
    category_slide = bytes([0xC0 | 0x25, 2, 3, 7])
    finished = _feed_group(
        decoder,
        _data_group(DATA_GROUP_MOT_HEADER, 41, 0, True, _header_update("bild.jpg", category_slide)),
    )

    assert finished[0].parameters[0x25] == bytes([3, 7])


def test_header_update_in_the_middle_of_a_body_is_processed_and_the_body_continues():
    """TS 101 499, clause 6.1: header updates may be inserted between the segments of
    another object. The update is to be processed immediately, the interrupted object
    assembled further."""
    decoder = MotDecoder()
    half = len(JPEG) // 2
    _feed_group(decoder, _data_group(DATA_GROUP_MOT_HEADER, 50, 0, True, _mot_header(len(JPEG))))
    assert _feed_group(decoder, _data_group(DATA_GROUP_MOT_BODY, 50, 0, False, JPEG[:half])) == []

    interleaved = _feed_group(
        decoder, _data_group(DATA_GROUP_MOT_HEADER, 51, 0, True, _header_update("alt.jpg"))
    )
    assert [obj.is_header_update for obj in interleaved] == [True]

    finished = _feed_group(decoder, _data_group(DATA_GROUP_MOT_BODY, 50, 1, True, JPEG[half:]))
    assert len(finished) == 1
    assert finished[0].body == JPEG


def test_image_is_not_a_header_update():
    decoder = MotDecoder()
    _feed_group(decoder, _data_group(DATA_GROUP_MOT_HEADER, 7, 0, True, _mot_header(len(JPEG))))
    finished = _feed_group(decoder, _data_group(DATA_GROUP_MOT_BODY, 7, 0, True, JPEG))

    assert not finished[0].is_header_update
