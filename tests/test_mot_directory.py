"""Tests for the MOT directory mode (ETSI EN 301 234, clause 7.2)."""

from __future__ import annotations

import gzip

import pytest

from dab.crc import crc16_ccitt
from dab.data_group import DataGroup, parse_data_group
from dab.mot_directory import (
    DATA_GROUP_MOT_DIRECTORY,
    DATA_GROUP_MOT_DIRECTORY_COMPRESSED,
    MotCarousel,
    MotObjectStatus,
    decompress_mot_directory,
    parse_mot_directory,
)
from tests.spi_vectors import (
    NDR_DIRECTORY_GROUP_0,
    NDR_DIRECTORY_GROUP_1,
    NDR_DIRECTORY_GROUP_2,
    NDR_SI_GROUP_0,
    NDR_SI_GROUP_1,
    NDR_SI_GROUP_2,
)

BODY = 4


def _parameter(parameter_id: int, value: bytes) -> bytes:
    """A header parameter with a length byte (PLI 3, clause 6.2)."""
    return bytes([0xC0 | parameter_id, len(value)]) + value


def _header(
    body_size: int, content_type: int, subtype: int, name: str, extra: bytes = b""
) -> bytes:
    """A MOT header: core (clause 6.1) and ContentName with character set byte."""
    extension = _parameter(0x0C, b"\x40" + name.encode("latin-1")) + extra
    header_size = 7 + len(extension)
    core = (body_size << 28) | (header_size << 15) | (content_type << 9) | subtype
    return core.to_bytes(7, "big") + extension


def _directory(entries: list[tuple[int, bytes]], period: int = 600) -> bytes:
    """An uncompressed MOT directory (clause 7.2.3): header, extension, entries."""
    extension = bytes([0x00])  # SortedHeaderInformation, without data field
    body = b"".join(tid.to_bytes(2, "big") + header for tid, header in entries)
    size = 13 + len(extension) + len(body)
    head = size.to_bytes(4, "big") + len(entries).to_bytes(2, "big") + period.to_bytes(3, "big")
    head += (0).to_bytes(2, "big") + len(extension).to_bytes(2, "big")
    return head + extension + body


def _groups(
    group_type: int, transport_id: int, entity: bytes, segment: int = 100
) -> list[DataGroup]:
    """The data groups of one MOT entity: segments with segmentation header, session
    header with segment number and transport ID, CRC (clause 5.1)."""
    parts = [entity[i : i + segment] for i in range(0, len(entity), segment)] or [b""]
    groups = []
    for number, part in enumerate(parts):
        last = number == len(parts) - 1
        raw = bytes([0x40 | 0x20 | 0x10 | group_type, 0x00])
        raw += ((last << 15) | number).to_bytes(2, "big")
        raw += bytes([0x12]) + transport_id.to_bytes(2, "big")
        raw += len(part).to_bytes(2, "big") + part
        raw += (crc16_ccitt(raw) ^ 0xFFFF).to_bytes(2, "big")
        groups.append(parse_data_group(raw))
    return groups


def _feed(carousel: MotCarousel, groups: list[DataGroup]) -> list:
    completed = []
    for group in groups:
        completed += carousel.process(group)
    return completed


def test_real_ndr_carousel_yields_directory_and_service_information():
    """NDR EPG on 7A: the SI segments arrive before the directory is complete, as on
    air; the object is put together as soon as the directory describes it."""
    carousel = MotCarousel()
    groups = [
        parse_data_group(raw)
        for raw in (
            NDR_SI_GROUP_0,
            NDR_DIRECTORY_GROUP_0,
            NDR_SI_GROUP_1,
            NDR_DIRECTORY_GROUP_1,
            NDR_SI_GROUP_2,
            NDR_DIRECTORY_GROUP_2,
        )
    ]

    completed = _feed(carousel, groups)

    directory = carousel.directory
    assert directory.transport_id == 0x7A06
    assert len(directory.entries) == 33
    assert 0x00 in directory.parameters  # SortedHeaderInformation
    assert directory.carousel_period is None
    names = [entry.header.content_name for entry in directory.entries]
    assert names[0] == "e0_d354_112x32_1.png"
    assert "spilogos95661E67.EIB" in names
    assert [(o.transport_id, o.content_type, o.content_subtype) for o in completed] == [
        (0x7A05, 7, 0)
    ]
    assert completed[0].content_name == "spilogos95661E67.EIB"
    assert len(completed[0].body) == 1488
    status = carousel.status()
    assert (status.directory_objects, status.complete_objects) == (33, 1)
    assert status.received_bytes == 1488


def test_body_before_directory_completes_when_the_directory_arrives():
    body = bytes(range(250))
    directory = _directory([(0x0101, _header(len(body), 2, 3, "logo.png"))])
    carousel = MotCarousel()

    assert _feed(carousel, _groups(BODY, 0x0101, body)) == []
    completed = _feed(carousel, _groups(DATA_GROUP_MOT_DIRECTORY, 0x0F00, directory))

    assert [(o.content_name, o.body, o.is_image) for o in completed] == [("logo.png", body, True)]


def test_an_object_is_reported_once_per_version():
    body = b"x" * 30
    carousel = MotCarousel()
    directory = _directory([(0x0101, _header(len(body), 7, 1, "a.EHB"))])
    _feed(carousel, _groups(DATA_GROUP_MOT_DIRECTORY, 0x0F00, directory))

    first = _feed(carousel, _groups(BODY, 0x0101, body))
    again = _feed(carousel, _groups(BODY, 0x0101, body))  # next turn of the carousel
    repeated_directory = _feed(carousel, _groups(DATA_GROUP_MOT_DIRECTORY, 0x0F00, directory))

    assert len(first) == 1
    assert again == repeated_directory == []


def test_a_new_directory_drops_what_it_no_longer_lists():
    carousel = MotCarousel()
    old = _directory([(0x0101, _header(3, 7, 1, "today")), (0x0102, _header(3, 7, 1, "gone"))])
    _feed(carousel, _groups(DATA_GROUP_MOT_DIRECTORY, 0x0F00, old))
    _feed(carousel, _groups(BODY, 0x0101, b"old"))
    _feed(carousel, _groups(BODY, 0x0102, b"bye"))

    new = _directory([(0x0101, _header(3, 7, 1, "today")), (0x0103, _header(3, 7, 1, "new"))])
    _feed(carousel, _groups(DATA_GROUP_MOT_DIRECTORY, 0x0F01, new))

    assert carousel.directory.transport_id == 0x0F01
    assert sorted(o.content_name for o in carousel.objects()) == ["today"]
    assert carousel.status().directories == 2


def test_compressed_directory():
    directory = _directory([(0x0101, _header(4, 7, 0, "si"))])
    packed = gzip.compress(directory)
    entity_size = 9 + len(packed)
    header = (0x80000000 | entity_size).to_bytes(4, "big") + bytes([0x01])
    header += len(directory).to_bytes(4, "big")
    entity = header + packed
    carousel = MotCarousel()

    assert decompress_mot_directory(entity) == directory
    _feed(carousel, _groups(DATA_GROUP_MOT_DIRECTORY_COMPRESSED, 0x0F00, entity))
    completed = _feed(carousel, _groups(BODY, 0x0101, b"data"))

    assert carousel.directory.compressed
    assert [o.content_name for o in completed] == ["si"]


def test_gzip_compressed_body_is_unpacked():
    content = b"advanced profile " * 20
    packed = gzip.compress(content)
    header = _header(len(packed), 7, 1, "adv", extra=bytes([0x40 | 0x11, 0x01]))
    carousel = MotCarousel()
    _feed(carousel, _groups(DATA_GROUP_MOT_DIRECTORY, 0x0F00, _directory([(0x0101, header)])))

    completed = _feed(carousel, _groups(BODY, 0x0101, packed))

    assert completed[0].body == content


def test_scrambled_object_is_skipped():
    header = _header(4, 2, 3, "secret.png", extra=_parameter(0x23, b"\x00\x01"))
    carousel = MotCarousel()
    _feed(carousel, _groups(DATA_GROUP_MOT_DIRECTORY, 0x0F00, _directory([(0x0101, header)])))

    assert _feed(carousel, _groups(BODY, 0x0101, b"abcd")) == []


def test_an_incomplete_body_waits_for_its_missing_segment():
    body = bytes(range(256)) * 2
    carousel = MotCarousel()
    _feed(
        carousel,
        _groups(DATA_GROUP_MOT_DIRECTORY, 0x0F00, _directory([(1, _header(512, 2, 1, "a"))])),
    )
    segments = _groups(BODY, 1, body)

    assert _feed(carousel, segments[:2] + segments[3:]) == []
    assert 0 < carousel.status().received_bytes < 512
    assert len(_feed(carousel, segments[2:3])) == 1


def test_the_status_tells_each_file_of_the_directory_apart():
    body = bytes(range(256)) * 2
    directory = _directory([(1, _header(512, 2, 1, "a.jpg")), (2, _header(300, 7, 1, "b.EHB"))])
    carousel = MotCarousel()
    _feed(carousel, _groups(DATA_GROUP_MOT_DIRECTORY, 0x0F00, directory))
    segments = _groups(BODY, 1, body)  # six of 100 bytes, the last of 12
    _feed(carousel, segments[:2] + segments[3:])
    _feed(carousel, _groups(BODY, 9, b"z" * 50))  # not in the directory

    status = carousel.status()

    assert (status.directory_transport_id, status.directories) == (0x0F00, 1)
    assert not status.directory_compressed
    assert [
        (o.content_name, o.segments, o.total_segments, o.received_bytes, o.complete)
        for o in status.objects
    ] == [("a.jpg", 5, 6, 412, False), ("b.EHB", 0, None, 0, False)]
    assert (status.received_bytes, status.complete_objects) == (412, 0)
    assert (status.unlisted_objects, status.unlisted_bytes) == (1, 50)

    _feed(carousel, segments[2:3])
    status = carousel.status()
    assert status.objects[0] == MotObjectStatus(1, "a.jpg", 2, 1, 512, 512, 6, 6, True)
    assert (status.received_bytes, status.complete_objects) == (512, 1)


@pytest.mark.parametrize(
    "mangle",
    [
        lambda d: bytes([d[0] | 0x80]) + d[1:],  # compression flag in a type 6 directory
        lambda d: (len(d) + 10).to_bytes(4, "big") + d[4:],  # size beyond the data
        lambda d: d[:-3],  # truncated header
    ],
)
def test_parse_mot_directory_rejects_broken_directories(mangle):
    directory = _directory([(0x0101, _header(4, 7, 0, "si"))])
    assert parse_mot_directory(directory, 1) is not None
    assert parse_mot_directory(mangle(directory), 1) is None
