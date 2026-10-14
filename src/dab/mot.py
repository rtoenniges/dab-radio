"""MOT objects (Multimedia Object Transfer) from the X-PAD: the slideshow.

Many stations send images alongside the audio: the programme logo, the cover of the
current track, traffic information. These images are transported as MOT objects (ETSI
EN 301 234), packed into MSC data groups (ETSI EN 300 401, clause 5.3.3), which in turn
sit in the X-PAD subfields of applications 12 and 13 (see `dab.pad`).

The path of an image through these layers:

1. An MSC data group is assembled from the X-PAD subfields (see `dab.data_group`). Its
   length is announced by a separate subfield of application 1.
2. The data group carries a header with its type (3 = MOT header, 4 = MOT body), a
   transport identifier and a segment number.
3. The segments sharing a transport identifier form the header or the body of one
   object. The header states size, content type and file name.
4. Once the header and the complete body are present, the image is finished. Objects
   with body size 0, above all header updates that retroactively change parameters of
   an image sent earlier, are complete with the header alone.

Since an image arrives spread over many seconds and is repeated cyclically, joining in
the middle of a transmission is the normal case: the partial pass is discarded and the
next one is received completely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from dab.charset import decode_charset
from dab.data_group import DataGroup, XPadDataGroupAssembler, parse_data_group
from dab.pad import XPAD_APP_MOT_CONTINUATION, XPAD_APP_MOT_START, XPadSubfield
from i18n import tr, tr_plural

logger = logging.getLogger(__name__)

# ETSI EN 300 401, table 13a: MSC data group types.
DATA_GROUP_MOT_HEADER = 3
DATA_GROUP_MOT_BODY = 4

# ETSI TS 101 756, table 17: content types. The slideshow uses images and the subtypes
# of "MOT transport" (ETSI TS 101 499, clause 6.2.3).
CONTENT_TYPE_IMAGE = 2
IMAGE_SUBTYPES = {0: "GIF", 1: "JPEG", 2: "BMP", 3: "PNG"}
CONTENT_TYPE_MOT_TRANSPORT = 5
MOT_TRANSPORT_HEADER_UPDATE = 0
MOT_TRANSPORT_HEADER_ONLY = 1

# ETSI EN 301 234, table 3: identifier of the "ContentName" parameter.
_PARAM_CONTENT_NAME = 0x0C

_HEADER_CORE_LENGTH = 7
_SEGMENT_HEADER_LENGTH = 2
_MAX_OBJECT_BYTES = 2 * 1024 * 1024  # guard against absurd size fields

# Unfinished objects kept at a time, the most recently touched ones. In X-PAD the
# slideshow sends its objects one after another; only header updates are inserted
# between the segments of another object (ETSI TS 101 499, clause 6.1). An object that
# lost a segment to a CRC error would otherwise sit in `MotDecoder` until the next
# reset, i.e. for hours on one station, and once the 16-bit transport identifier wraps
# around, the segments of a later object would land in it.
MAX_PARTIAL_OBJECTS = 8


@dataclass
class MotObject:
    """A completely received MOT object."""

    transport_id: int
    content_type: int
    content_subtype: int
    content_name: str | None
    body: bytes
    # All header parameters unchanged, keyed by identifier. Deliberately kept raw: the
    # interpretation of individual parameters (e.g. the slideshow extensions from ETSI
    # TS 101 499) happens one layer up, and unknown parameters are not lost on the way.
    parameters: dict[int, bytes] = field(default_factory=dict)

    @property
    def is_image(self) -> bool:
        return self.content_type == CONTENT_TYPE_IMAGE

    @property
    def is_header_update(self) -> bool:
        """A header update changes the parameters of an object sent earlier, identified
        by the same ContentName; it has no body (ETSI EN 301 234, clause 7.1.3)."""
        return (
            self.content_type == CONTENT_TYPE_MOT_TRANSPORT
            and self.content_subtype == MOT_TRANSPORT_HEADER_UPDATE
        )

    def describe(self) -> str:
        """Short description for the GUI."""
        kind = IMAGE_SUBTYPES.get(self.content_subtype)
        if kind is None:
            kind = tr("Type {subtype}").format(subtype=self.content_subtype)
        size = len(self.body)
        text = tr_plural("{kind}, {size} byte", "{kind}, {size} bytes", size).format(
            kind=kind, size=size
        )
        return f"{text} - {self.content_name}" if self.content_name else text


@dataclass
class _PartialObject:
    """The segments of an object received so far."""

    header_segments: dict[int, bytes] = field(default_factory=dict)
    header_last: int | None = None
    body_segments: dict[int, bytes] = field(default_factory=dict)
    body_last: int | None = None


@dataclass(frozen=True)
class MotHeader:
    """The evaluated header of a MOT object."""

    body_size: int
    content_type: int
    content_subtype: int
    parameters: dict[int, bytes]

    @property
    def content_name(self) -> str | None:
        return decode_content_name(self.parameters.get(_PARAM_CONTENT_NAME))


def decode_content_name(value: bytes | None) -> str | None:
    """Turns the ContentName parameter into text.

    In the ContentName the first byte is a character set indicator (4 bits character
    set, 4 bits reserved) and only then follows the name (ETSI EN 301 234, clause
    6.2.2.1.1). Confirmed on air: 0x00 before "12.JPG" at ffn, 0x40 before the file
    names of NDR.

    For the ContentName only! The slideshow's text parameters (CategoryTitle,
    ClickThroughURL, AlternativeLocationURL) are plain UTF-8 without a character set
    byte (ETSI TS 101 499, clauses 5.3.3 and 5.3.5.4) and are decoded separately in
    `dab.slideshow`. An earlier version of this function was called
    "decode_parameter_text" and was wrongly used for them as well: every link became
    garbage and every title lost its first letter.
    """
    if not value:
        return None
    return decode_charset(value[1:], value[0] >> 4)


def parse_mot_header(header: bytes) -> MotHeader | None:
    """Reads the MOT header (ETSI EN 301 234, clause 6.1).

    The fixed part is seven bytes: 28 bits body size, 13 bits header size, 6 bits
    content type, 9 bits content subtype. Parameters of variable length follow; they are
    all collected here unchanged.
    """
    if len(header) < _HEADER_CORE_LENGTH:
        return None

    core = int.from_bytes(header[:_HEADER_CORE_LENGTH], "big")
    body_size = (core >> 28) & 0x0FFFFFFF
    header_size = (core >> 15) & 0x1FFF
    content_type = (core >> 9) & 0x3F
    content_subtype = core & 0x1FF
    if body_size > _MAX_OBJECT_BYTES or header_size < _HEADER_CORE_LENGTH:
        return None

    return MotHeader(
        body_size=body_size,
        content_type=content_type,
        content_subtype=content_subtype,
        parameters=parse_header_parameters(header[_HEADER_CORE_LENGTH:header_size]),
    )


def parse_header_parameters(extension: bytes) -> dict[int, bytes]:
    """Splits the header extension into its parameters.

    Every parameter starts with two bits of length indicator (PLI) and six bits of
    parameter identifier; the PLI determines how the data length is found (ETSI EN 301
    234, clause 6.2).
    """
    parameters: dict[int, bytes] = {}
    offset = 0
    while offset < len(extension):
        indicator = extension[offset]
        length_indicator = indicator >> 6
        parameter_id = indicator & 0x3F
        offset += 1

        if length_indicator == 0:
            length = 0
        elif length_indicator == 1:
            length = 1
        elif length_indicator == 2:
            length = 4
        else:
            if offset >= len(extension):
                break
            if extension[offset] & 0x80:  # extended length field (15 bits)
                if offset + 1 >= len(extension):
                    break
                length = ((extension[offset] & 0x7F) << 8) | extension[offset + 1]
                offset += 2
            else:
                length = extension[offset] & 0x7F
                offset += 1

        if offset + length > len(extension):
            break
        parameters[parameter_id] = extension[offset : offset + length]
        offset += length
    return parameters


def strip_segment_header(data: bytes) -> bytes | None:
    """Removes the segment header that every MOT segment starts with.

    Two bytes: 3 bits repetition counter, 13 bits segment size (ETSI EN 301 234, clause
    5.1.1). Without this step the two bytes end up in the payload: the header can then
    not be read, and the body would carry two foreign bytes in front of the image file.
    """
    if len(data) < _SEGMENT_HEADER_LENGTH:
        return None
    size = ((data[0] & 0x1F) << 8) | data[1]
    segment = data[_SEGMENT_HEADER_LENGTH : _SEGMENT_HEADER_LENGTH + size]
    return segment if len(segment) == size else None


class MotDecoder:
    """Assembles complete MOT objects from the X-PAD subfields."""

    def __init__(self) -> None:
        self._assembler = XPadDataGroupAssembler(XPAD_APP_MOT_START, XPAD_APP_MOT_CONTINUATION)
        self._objects: dict[int, _PartialObject] = {}  # in order of the last segment

    def reset(self) -> None:
        self._assembler.reset()
        self._objects.clear()

    def process(self, subfields: list[XPadSubfield]) -> list[MotObject]:
        """Processes the subfields of one access unit.

        Returns:
            The objects completed in this step (usually none).
        """
        finished: list[MotObject] = []
        for group in self._assembler.process(subfields):
            parsed = parse_data_group(group)
            if parsed is None or parsed.transport_id is None:
                continue
            completed = self._add_segment(parsed)
            if completed is not None:
                finished.extend(completed)
        return finished

    def _add_segment(self, group: DataGroup) -> list[MotObject] | None:
        if group.group_type not in (DATA_GROUP_MOT_HEADER, DATA_GROUP_MOT_BODY):
            return None
        segment = strip_segment_header(group.data)
        if segment is None:
            return None

        # The object just touched goes last; the ones touched longest ago make room.
        partial = self._objects.pop(group.transport_id, None)
        if partial is None:
            partial = _PartialObject()
        self._objects[group.transport_id] = partial
        while len(self._objects) > MAX_PARTIAL_OBJECTS:
            del self._objects[next(iter(self._objects))]

        if group.group_type == DATA_GROUP_MOT_HEADER:
            partial.header_segments[group.segment_number] = segment
            if group.last_segment:
                partial.header_last = group.segment_number
        else:
            partial.body_segments[group.segment_number] = segment
            if group.last_segment:
                partial.body_last = group.segment_number

        completed = self._try_complete(group.transport_id, partial)
        return [completed] if completed else None

    def _try_complete(self, transport_id: int, partial: _PartialObject) -> MotObject | None:
        if partial.header_last is None:
            return None
        if any(i not in partial.header_segments for i in range(partial.header_last + 1)):
            return None

        header = b"".join(partial.header_segments[i] for i in range(partial.header_last + 1))
        parsed = parse_mot_header(header)
        if parsed is None:
            del self._objects[transport_id]
            return None

        # Objects with body size 0 (header updates and "header only") are complete with
        # the header. Body segments never arrive for them; without this exception they
        # would stay half-finished forever.
        if parsed.body_size == 0:
            body = b""
        else:
            if partial.body_last is None:
                return None
            if any(i not in partial.body_segments for i in range(partial.body_last + 1)):
                return None
            body = b"".join(partial.body_segments[i] for i in range(partial.body_last + 1))
            if len(body) < parsed.body_size:
                return None  # not all body segments have arrived yet
        del self._objects[transport_id]

        return MotObject(
            transport_id=transport_id,
            content_type=parsed.content_type,
            content_subtype=parsed.content_subtype,
            content_name=parsed.content_name,
            body=body[: parsed.body_size],
            parameters=parsed.parameters,
        )
