"""MOT directory mode: a data carousel of many objects (ETSI EN 301 234, clause 7.2).

The slideshow sends one image at a time, each with its own MOT header (header mode, see
`dab.mot`). A data service such as the programme guide (SPI) instead broadcasts a whole
set of files over and over again, a data carousel. Its table of contents, the MOT
directory, lists every object with its transport identifier and its complete MOT
header: size, content type, ContentName and further parameters. The bodies travel
separately, cut into segments, in any order and repeated at will.

A receiver therefore:

1. collects the segments of the directory (data group type 6, or 7 if compressed) and
   evaluates it once complete. A directory with a new transport identifier replaces
   the previous one; objects it no longer lists are dropped (clause 7.2.7.3).
2. collects the body segments (data group type 4) by transport identifier. Segments of
   objects not described by the current directory are kept as well, in case the
   directory is simply still missing (clause C.3.4.1), within a memory budget.
3. puts an object together as soon as the directory describes it and all its segments
   are there. Objects compressed with gzip (MOT parameter CompressionType) are
   unpacked; scrambled ones (CAInfo) are skipped.

Since everything is repeated, a missing segment simply arrives with the next turn of
the carousel.
"""

from __future__ import annotations

import dataclasses
import logging
import zlib
from dataclasses import dataclass, field

from dab.data_group import DataGroup
from dab.mot import (
    DATA_GROUP_MOT_BODY,
    MotHeader,
    MotObject,
    parse_header_parameters,
    parse_mot_header,
    strip_segment_header,
)

logger = logging.getLogger(__name__)

# EN 300 401, table 13a / EN 301 234, clause 5.1.3: data group types of the directory.
DATA_GROUP_MOT_DIRECTORY = 6
DATA_GROUP_MOT_DIRECTORY_COMPRESSED = 7

# EN 301 234, clause 6.3: MOT parameters evaluated by the carousel itself.
PARAM_COMPRESSION_TYPE = 0x11
PARAM_CA_INFO = 0x23
_COMPRESSION_GZIP = 0x01  # TS 101 756, table 18

_DIRECTORY_HEADER_BYTES = 13
_COMPRESSED_DIRECTORY_HEADER_BYTES = 9
_HEADER_CORE_BYTES = 7
_TRANSPORT_ID_BYTES = 2
_MAX_DIRECTORY_BYTES = 1 << 20  # far above anything sensible (SPI: at most 8 kB)
_MAX_OBJECT_BYTES = 8 << 20
# Body segments of objects the current directory does not (yet) list are kept up to this
# many bytes; the oldest are dropped first.
_MAX_UNLISTED_BYTES = 4 << 20


@dataclass(frozen=True)
class MotDirectoryEntry:
    transport_id: int
    header: MotHeader


@dataclass(frozen=True)
class MotDirectory:
    """An evaluated MOT directory (EN 301 234, clause 7.2.3)."""

    transport_id: int
    # Maximum time for one turn of the carousel in seconds, None if undefined.
    carousel_period: float | None
    segment_size: int  # 0: objects may use different segment sizes
    parameters: dict[int, bytes]  # directory extension (e.g. SortedHeaderInformation)
    entries: tuple[MotDirectoryEntry, ...]
    compressed: bool = False  # received as data group type 7

    def entry(self, transport_id: int) -> MotDirectoryEntry | None:
        for entry in self.entries:
            if entry.transport_id == transport_id:
                return entry
        return None

    @property
    def total_body_bytes(self) -> int:
        return sum(entry.header.body_size for entry in self.entries)


def parse_mot_directory(data: bytes, transport_id: int) -> MotDirectory | None:
    """Evaluates an uncompressed MOT directory (clause 7.2.3, figure 30).

    Returns:
        The directory, or None if the data do not form one: compression flag or Rfu
        bits set, sizes that do not add up, a header that cannot be read.
    """
    if len(data) < _DIRECTORY_HEADER_BYTES or data[0] & 0xC0:
        return None
    directory_size = int.from_bytes(data[:4], "big") & 0x3FFFFFFF
    if directory_size > len(data) or directory_size < _DIRECTORY_HEADER_BYTES:
        return None
    data = data[:directory_size]
    number_of_objects = int.from_bytes(data[4:6], "big")
    period = int.from_bytes(data[6:9], "big")
    if data[9] & 0x80:  # Rfu
        return None
    segment_size = int.from_bytes(data[9:11], "big") & 0x1FFF
    extension_length = int.from_bytes(data[11:13], "big")
    offset = _DIRECTORY_HEADER_BYTES + extension_length
    if offset > len(data):
        return None
    parameters = parse_header_parameters(data[_DIRECTORY_HEADER_BYTES:offset])

    entries = []
    for _ in range(number_of_objects):
        if offset + _TRANSPORT_ID_BYTES + _HEADER_CORE_BYTES > len(data):
            return None
        entry_id = int.from_bytes(data[offset : offset + _TRANSPORT_ID_BYTES], "big")
        offset += _TRANSPORT_ID_BYTES
        core = int.from_bytes(data[offset : offset + _HEADER_CORE_BYTES], "big")
        header_size = (core >> 15) & 0x1FFF
        if offset + header_size > len(data):
            return None
        header = parse_mot_header(data[offset : offset + header_size])
        if header is None:
            return None
        entries.append(MotDirectoryEntry(entry_id, header))
        offset += header_size

    return MotDirectory(
        transport_id=transport_id,
        carousel_period=period / 10 if period else None,
        segment_size=segment_size,
        parameters=parameters,
        entries=tuple(entries),
    )


def decompress_mot_directory(entity: bytes) -> bytes | None:
    """Unpacks a compressed MOT directory (clause 7.2.8, figure 31): 9 bytes of header
    with compression flag, entity size, compression method and uncompressed length,
    then the directory compressed with gzip."""
    if len(entity) < _COMPRESSED_DIRECTORY_HEADER_BYTES or entity[0] & 0xC0 != 0x80:
        return None
    if entity[4] != _COMPRESSION_GZIP:
        return None
    return _gunzip(entity[_COMPRESSED_DIRECTORY_HEADER_BYTES:], _MAX_DIRECTORY_BYTES)


def _gunzip(data: bytes, limit: int) -> bytes | None:
    """Unpacks gzip data, None if they are broken or larger than `limit`."""
    decompressor = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
    try:
        result = decompressor.decompress(data, limit)
    except zlib.error:
        return None
    if decompressor.unconsumed_tail or not decompressor.eof:
        return None
    return result


@dataclass
class _Segments:
    """The segments of one directory or body received so far."""

    parts: dict[int, bytes] = field(default_factory=dict)
    last: int | None = None

    @property
    def size(self) -> int:
        return sum(len(part) for part in self.parts.values())

    def add(self, group: DataGroup, segment: bytes) -> None:
        self.parts[group.segment_number] = segment
        if group.last_segment:
            self.last = group.segment_number

    def joined(self) -> bytes | None:
        """All segments in order, None while any is missing."""
        if self.last is None or len(self.parts) < self.last + 1:
            return None
        if any(number not in self.parts for number in range(self.last + 1)):
            return None
        return b"".join(self.parts[number] for number in range(self.last + 1))


@dataclass(frozen=True)
class MotObjectStatus:
    """One object of the directory and how far its body has been received."""

    transport_id: int
    content_name: str | None
    content_type: int
    content_subtype: int
    body_size: int
    received_bytes: int  # of the body, at most `body_size`
    segments: int  # body segments received
    # Segments the body consists of; None until its last segment has come, unless the
    # directory fixes one segment size for all objects.
    total_segments: int | None
    complete: bool


@dataclass(frozen=True)
class MotCarouselStatus:
    """How far the carousel has been received, for the display."""

    directory_objects: int | None  # objects in the directory, None without a directory
    complete_objects: int  # of those, completely received
    directory_bytes: int  # total body size according to the directory
    received_bytes: int  # of that, received so far (complete and partial bodies)
    carousel_period: float | None  # seconds per turn according to the directory
    directories: int  # directories evaluated so far (versions of the carousel)
    directory_transport_id: int | None = None
    directory_compressed: bool = False
    objects: tuple[MotObjectStatus, ...] = ()  # in the order of the directory
    # Bodies of objects the directory does not list (of one still to come, or of an
    # older one): how many, and their bytes so far.
    unlisted_objects: int = 0
    unlisted_bytes: int = 0


class MotCarousel:
    """Reassembles the objects of a MOT data carousel from its data groups."""

    def __init__(self) -> None:
        self._directory: MotDirectory | None = None
        self._directories_received = 0
        self._directory_segments: dict[tuple[int, int], _Segments] = {}
        self._bodies: dict[int, _Segments] = {}
        self._objects: dict[int, MotObject] = {}  # completed, by transport identifier
        self._segment_counts: dict[int, int] = {}  # segments each completed object had

    @property
    def directory(self) -> MotDirectory | None:
        return self._directory

    def objects(self) -> list[MotObject]:
        """The completely received objects of the current directory."""
        return list(self._objects.values())

    def reset(self) -> None:
        self._directory = None
        self._directories_received = 0
        self._directory_segments.clear()
        self._bodies.clear()
        self._objects.clear()
        self._segment_counts.clear()

    def process(self, group: DataGroup) -> list[MotObject]:
        """Takes one data group of the carousel's service component.

        Returns:
            The objects completed by it: usually none, after a new directory possibly
            many whose bodies were already there.
        """
        if group.transport_id is None:
            return []
        if group.group_type == DATA_GROUP_MOT_BODY:
            return self._add_body_segment(group)
        if group.group_type in (DATA_GROUP_MOT_DIRECTORY, DATA_GROUP_MOT_DIRECTORY_COMPRESSED):
            return self._add_directory_segment(group)
        return []

    def status(self) -> MotCarouselStatus:
        directory = self._directory
        if directory is None:
            partial = sum(segments.size for segments in self._bodies.values())
            return MotCarouselStatus(
                None,
                0,
                0,
                partial,
                None,
                0,
                unlisted_objects=len(self._bodies),
                unlisted_bytes=partial,
            )
        objects = tuple(
            self._object_status(entry, directory.segment_size) for entry in directory.entries
        )
        listed = {entry.transport_id for entry in directory.entries}
        unlisted = [segments for tid, segments in self._bodies.items() if tid not in listed]
        return MotCarouselStatus(
            directory_objects=len(directory.entries),
            complete_objects=len(self._objects),
            directory_bytes=directory.total_body_bytes,
            received_bytes=sum(o.received_bytes for o in objects),
            carousel_period=directory.carousel_period,
            directories=self._directories_received,
            directory_transport_id=directory.transport_id,
            directory_compressed=directory.compressed,
            objects=objects,
            unlisted_objects=len(unlisted),
            unlisted_bytes=sum(segments.size for segments in unlisted),
        )

    def _object_status(self, entry: MotDirectoryEntry, segment_size: int) -> MotObjectStatus:
        header = entry.header
        transport_id = entry.transport_id
        complete = transport_id in self._objects
        segments = self._bodies.get(transport_id)
        if complete:
            received = header.body_size
            count = total = self._segment_counts[transport_id]
        elif segments is not None:
            received = min(segments.size, header.body_size)
            count = len(segments.parts)
            total = None if segments.last is None else segments.last + 1
        else:
            received, count, total = 0, 0, None
        if total is None and segment_size:
            total = max(1, -(-header.body_size // segment_size))
        return MotObjectStatus(
            transport_id=transport_id,
            content_name=header.content_name,
            content_type=header.content_type,
            content_subtype=header.content_subtype,
            body_size=header.body_size,
            received_bytes=received,
            segments=count,
            total_segments=total,
            complete=complete,
        )

    def _add_directory_segment(self, group: DataGroup) -> list[MotObject]:
        if self._directory is not None and group.transport_id == self._directory.transport_id:
            return []  # a repetition of the directory in use
        segment = strip_segment_header(group.data)
        if segment is None:
            return []
        key = (group.group_type, group.transport_id)
        segments = self._directory_segments.setdefault(key, _Segments())
        segments.add(group, segment)
        entity = segments.joined()
        if entity is None:
            return []
        # Segments of older directories will never complete any more.
        self._directory_segments.clear()

        compressed = group.group_type == DATA_GROUP_MOT_DIRECTORY_COMPRESSED
        if compressed:
            entity = decompress_mot_directory(entity)
            if entity is None:
                logger.debug("Dropped a compressed MOT directory that could not be unpacked")
                return []
        directory = parse_mot_directory(entity, group.transport_id)
        if directory is None:
            logger.debug("Dropped a MOT directory that could not be read")
            return []
        if compressed:
            directory = dataclasses.replace(directory, compressed=True)
        return self._use_directory(directory)

    def _use_directory(self, directory: MotDirectory) -> list[MotObject]:
        self._directory = directory
        self._directories_received += 1
        listed = {entry.transport_id for entry in directory.entries}
        # A changed object gets a new transport identifier (clause 7.2.7.3): whatever the
        # new directory does not list is outdated.
        for transport_id in [tid for tid in self._objects if tid not in listed]:
            del self._objects[transport_id]
            del self._segment_counts[transport_id]
        for transport_id in [tid for tid in self._bodies if tid not in listed]:
            del self._bodies[transport_id]
        completed = []
        for entry in directory.entries:
            mot_object = self._try_complete(entry.transport_id)
            if mot_object is not None:
                completed.append(mot_object)
        return completed

    def _add_body_segment(self, group: DataGroup) -> list[MotObject]:
        transport_id = group.transport_id
        if transport_id in self._objects:
            return []  # already complete, this is the next turn of the carousel
        if self._directory is None or self._directory.entry(transport_id) is None:
            # Not (yet) in the directory: kept for a directory that may list it, within
            # the budget.
            self._limit_unlisted()
        segment = strip_segment_header(group.data)
        if segment is None:
            return []
        self._bodies.setdefault(transport_id, _Segments()).add(group, segment)
        mot_object = self._try_complete(transport_id)
        return [] if mot_object is None else [mot_object]

    def _limit_unlisted(self) -> None:
        directory = self._directory
        unlisted = [
            tid for tid in self._bodies if directory is None or directory.entry(tid) is None
        ]
        total = sum(self._bodies[tid].size for tid in unlisted)
        for transport_id in unlisted:  # dicts keep insertion order: oldest first
            if total <= _MAX_UNLISTED_BYTES:
                break
            total -= self._bodies.pop(transport_id).size

    def _try_complete(self, transport_id: int) -> MotObject | None:
        directory = self._directory
        entry = None if directory is None else directory.entry(transport_id)
        segments = self._bodies.get(transport_id)
        if entry is None or segments is None:
            return None
        header = entry.header
        body = segments.joined()
        if body is None or len(body) < header.body_size:
            return None
        del self._bodies[transport_id]
        body = body[: header.body_size]

        if PARAM_CA_INFO in header.parameters:
            return None  # scrambled on MOT level; not for a receiver without CA
        compression = header.parameters.get(PARAM_COMPRESSION_TYPE)
        if compression:
            if compression[0] != _COMPRESSION_GZIP:
                return None
            body = _gunzip(body, _MAX_OBJECT_BYTES)
            if body is None:
                logger.debug("Dropped MOT object %04X: broken gzip data", transport_id)
                return None

        mot_object = MotObject(
            transport_id=transport_id,
            content_type=header.content_type,
            content_subtype=header.content_subtype,
            content_name=header.content_name,
            body=body,
            parameters=header.parameters,
        )
        self._objects[transport_id] = mot_object
        self._segment_counts[transport_id] = segments.last + 1
        return mot_object
