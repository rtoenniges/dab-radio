"""Journaline: text pages of the station, navigable through menus (ETSI TS 102 979).

Journaline is teletext for digital radio: menus lead to news, programme previews and
contact details. Deutschlandfunk sends it in the X-PAD of its audio service, in
applications 4 and 5 (see `dab.pad`), announced in FIG 0/13 as user application 0x44A.
FIG 0/13 also names the X-PAD application type, the length of the extended header and,
optionally, a point of entry other than the root object (`JournalineParameters`).

The path of a page through the layers:

1. Every page, a "JML object", travels in an MSC data group of its own (clause 8.1.1),
   assembled from the X-PAD like the groups of the slideshow (see `dab.data_group`).
   Journaline forbids segmentation, user access fields and header extensions; the CRC
   is mandatory. Data group type 0 carries a JML object, type 6 management data (a
   table of contents meant for cache management, not evaluated here).
2. The object starts with a three-byte standard header: object ID, then object type,
   static flag, compress flag and revision index (clause 5.2). An extended header of
   fixed length may follow; FIG 0/13 signals its length, which is 0 on air.
3. If the object is compressed, one byte names the method (0x08, deflate, is the only
   one defined), and the content section follows as a raw deflate stream (RFC 1951)
   without zlib header. This byte is why on air the deflate stream only ends cleanly
   when started one byte after the header.
4. The content section is UTF-8 text, divided into blocks by the JML codes 0x00..0x0F
   (title, link item, body text, list item, ...) and interspersed with escape
   sequences 0x10..0x1F (line breaks, highlighting, data sections carrying links and
   timeouts; clause 5.3). Control codes cannot occur inside a UTF-8 character, so the
   content can be split byte by byte before decoding the text.

Objects are repeated in a carousel and are self-contained: a receiver needs no index to
show one. It keeps what it has received by object ID (`JournalineStore`) and lets the
user walk the menus from the point of entry onwards (`NavigationHistory`), the root
object 0x0000 unless FIG 0/13 names another.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto

from dab.data_group import XPadDataGroupAssembler, parse_data_group
from dab.pad import XPAD_APP_JOURNALINE_START, XPadSubfield

# Clause 8.1.1: data group types.
DATA_GROUP_JML_OBJECT = 0
DATA_GROUP_MANAGEMENT = 6

# Clause 4.3: the default point of entry, the only object every service must carry.
ROOT_OBJECT_ID = 0x0000

# Clause 5.1: header section plus uncompressed content section. The same limit applies
# to the data group field (clause 8.1.1).
MAX_OBJECT_BYTES = 4092

_STANDARD_HEADER_LENGTH = 3
_COMPRESSION_DEFLATE = 0x08  # clause 5.2; all other method values are reserved

# JML codes (clause 5.3.1, table 1). 0x06..0x0F are reserved.
_JML_END = 0x00
_JML_TITLE = 0x01
_JML_LINK_ITEM = 0x02
_JML_BODY_TEXT = 0x03
_JML_LIST_ITEM = 0x04
_JML_LIST_COLUMN = 0x05

# Escape sequence codes (clause 5.3.2, table 2). 0x15..0x19, 0x1E and 0x1F are reserved.
_ESC_LINE_BREAK = 0x10
_ESC_HIGHLIGHT_START = 0x12
_ESC_HIGHLIGHT_STOP = 0x13
_ESC_END_OF_INTRODUCTION = 0x14
_ESC_DATA_SECTION_START = 0x1A
_ESC_DATA_SECTION_CONTINUATION = 0x1B
_ESC_EXTENDED_CODE_BEGIN = 0x1C
_ESC_EXTENDED_CODE_END = 0x1D
_FIRST_TEXT_BYTE = 0x20

_DATA_SECTION_BLOCK = 256  # payload of one data section escape at most

# DS types (clause 5.3.2.2, table 4) evaluated here; padding, macros, keywords, speech
# and location data are skipped.
_DS_ABSOLUTE_TIMEOUT = 0x01
_DS_RELATIVE_TIMEOUT = 0x02
_DS_GENERAL_LINK = 0x03

# Absolute timeouts count quarter hours since 2000-01-01 00:00 UTC.
_TIMEOUT_EPOCH = 946_684_800  # POSIX time of that moment
_TIMEOUT_UNIT_SECONDS = 15 * 60


class ObjectType(IntEnum):
    """Object types (clause 5.2). 0 and 5..7 are reserved; such objects are ignored."""

    MENU = 1
    PLAIN_TEXT = 2
    TITLE_ONLY = 3
    LIST = 4


class LinkType(IntEnum):
    """Link types of a general link target (clause 5.3.2.3, table 5)."""

    JML_OBJECT = 0
    DAB_URI = 1
    URL = 2
    TELEPHONE = 3
    SMS = 4


@dataclass(frozen=True)
class MenuItem:
    """A link item of a menu: the visible text and the object it leads to."""

    target: int
    text: str


@dataclass(frozen=True)
class LinkTarget:
    """A general link target (clause 5.3.2.3): an action offered in addition to the
    content, e.g. a web page or a telephone number."""

    link_type: LinkType
    address: str  # empty for JML_OBJECT, see `target`
    label: str  # may be empty
    target: int | None = None  # object ID, JML_OBJECT only


@dataclass(frozen=True)
class JmlObject:
    """A received JML object, decoded down to its visible text.

    Of the escape sequences only line breaks change the text; a list column code
    outside a list becomes a space. Highlighting is kept for the body text, where
    Deutschlandfunk uses it for subheadings, and dropped elsewhere, as are preferred
    word breaks; the standard permits a receiver to ignore both (clause 5.3.2).
    """

    object_id: int
    object_type: ObjectType
    static: bool  # the ID keeps carrying the same kind of content (for bookmarks)
    revision: int  # 0..7, counted up with every change of content
    compressed: bool
    title: str
    menu_items: tuple[MenuItem, ...] = ()  # menus only
    body: str = ""  # plain text messages only
    body_highlights: tuple[tuple[int, int], ...] = ()  # highlighted (start, end) in `body`
    list_items: tuple[tuple[str, ...], ...] = ()  # list messages only, one entry per column
    links: tuple[LinkTarget, ...] = ()
    absolute_timeout: float | None = None  # POSIX time from which the object is void
    relative_timeout_minutes: int | None = None  # void this long after each reception
    transmitted_size: int = 0  # bytes of the data group field
    content_size: int = 0  # bytes of the uncompressed content section


def decode_data_group(group: bytes, extended_header_length: int = 0) -> JmlObject | None:
    """Decodes a complete MSC data group of the Journaline application.

    Returns:
        The JML object, or None if the group is corrupt, breaks the transport rules of
        clause 8.1.1 (which say to discard it), carries management data or an object
        that cannot be decoded.
    """
    parsed = parse_data_group(group)
    if parsed is None:
        return None
    if (
        parsed.has_extension
        or not parsed.has_crc
        or parsed.has_segment_field
        or parsed.has_user_access
    ):
        return None
    if parsed.group_type != DATA_GROUP_JML_OBJECT or len(parsed.data) > MAX_OBJECT_BYTES:
        return None
    return parse_jml_object(parsed.data, extended_header_length)


def parse_jml_object(data: bytes, extended_header_length: int = 0) -> JmlObject | None:
    """Decodes a JML object from the data group field (clause 5).

    Args:
        extended_header_length: length of the extended header in bytes, from the user
            application data in FIG 0/13 (clause 8.1.2). Its content is not defined
            yet and is skipped.

    Returns:
        The object, or None for a reserved object type, an unknown compression method,
        a broken deflate stream or an object beyond the size limit.
    """
    header_end = _STANDARD_HEADER_LENGTH + extended_header_length
    if len(data) < header_end:
        return None
    description = data[2]
    try:
        object_type = ObjectType(description >> 5)
    except ValueError:
        return None
    compressed = bool(description & 0x08)

    content = data[header_end:]
    if compressed:
        content = _inflate(content)
        if content is None:
            return None
    if header_end + len(content) > MAX_OBJECT_BYTES:
        return None

    blocks, sections = _parse_content(content)
    absolute_timeout, relative_timeout, links = _evaluate_data_sections(sections)
    title = next((block.text for block in blocks if block.code == _JML_TITLE), "")
    menu_items: tuple[MenuItem, ...] = ()
    body = ""
    body_highlights: tuple[tuple[int, int], ...] = ()
    list_items: tuple[tuple[str, ...], ...] = ()
    # Every block type applies to certain object types only (table 1); strays are
    # ignored.
    if object_type is ObjectType.MENU:
        menu_items = tuple(
            MenuItem(block.target, block.text)
            for block in blocks
            if block.code == _JML_LINK_ITEM and block.target is not None
        )
    elif object_type is ObjectType.PLAIN_TEXT:
        body, body_highlights = _join_body(
            [block.highlighted_text() for block in blocks if block.code == _JML_BODY_TEXT]
        )
    elif object_type is ObjectType.LIST:
        list_items = tuple(block.columns() for block in blocks if block.code == _JML_LIST_ITEM)

    return JmlObject(
        object_id=(data[0] << 8) | data[1],
        object_type=object_type,
        static=bool(description & 0x10),
        revision=description & 0x07,
        compressed=compressed,
        title=title,
        menu_items=menu_items,
        body=body,
        body_highlights=body_highlights,
        list_items=list_items,
        links=tuple(links),
        absolute_timeout=absolute_timeout,
        relative_timeout_minutes=relative_timeout,
        transmitted_size=len(data),
        content_size=len(content),
    )


def _inflate(section: bytes) -> bytes | None:
    """Unpacks a compressed content section: the method byte, then raw deflate.

    The standard fixes the deflate window at 4 096 bytes. Decompressing with the
    largest window (wbits -15) accepts such streams as well and does not fail on an
    encoder that takes the limit less strictly. The output is capped so that a
    malformed stream cannot inflate beyond the object size limit.
    """
    if not section or section[0] != _COMPRESSION_DEFLATE:
        return None
    decompressor = zlib.decompressobj(-15)
    try:
        content = decompressor.decompress(section[1:], MAX_OBJECT_BYTES + 1)
    except zlib.error:
        return None
    if not decompressor.eof or len(content) > MAX_OBJECT_BYTES:
        return None
    return content


@dataclass
class _Block:
    """A block of the content section while it is being read: its JML code and the
    raw text, split into columns by list column codes."""

    code: int
    target: int | None = None  # link items only
    raw_columns: list[bytearray] = field(default_factory=lambda: [bytearray()])

    def add(self, text: bytes) -> None:
        self.raw_columns[-1] += text

    def next_column(self) -> None:
        # Table 1: a receiver that cannot show columns replaces the code at least with
        # a space, which is all it can mean outside a list item.
        if self.code == _JML_LIST_ITEM:
            self.raw_columns.append(bytearray())
        else:
            self.add(b" ")

    def columns(self) -> tuple[str, ...]:
        """The visible text of every column, without highlighting."""
        return tuple(_visible_text(column)[0] for column in self.raw_columns)

    @property
    def text(self) -> str:
        return " ".join(self.columns())

    def highlighted_text(self) -> tuple[str, tuple[tuple[int, int], ...]]:
        """The visible text and its highlighted passages; see `_visible_text`."""
        return _visible_text(b" ".join(self.raw_columns))


def _decode_text(raw: bytes) -> str:
    return bytes(raw).decode("utf-8", errors="replace").strip()


def _visible_text(raw: bytes) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Decodes the text of a block without surrounding white space.

    `_parse_content` leaves the highlight codes in the text; here they are removed and
    turned into character ranges. A highlight still open at the end of the block ends
    there.

    Returns:
        `(text, ((start, end), ...))`.
    """
    marked = bytes(raw).decode("utf-8", errors="replace")
    chars: list[str] = []
    ranges: list[tuple[int, int]] = []
    start = None
    for char in marked:
        if char == chr(_ESC_HIGHLIGHT_START):
            if start is None:
                start = len(chars)
        elif char == chr(_ESC_HIGHLIGHT_STOP):
            if start is not None:
                ranges.append((start, len(chars)))
                start = None
        else:
            chars.append(char)
    if start is not None:
        ranges.append((start, len(chars)))

    text = "".join(chars)
    stripped = text.strip()
    lead = len(text) - len(text.lstrip())
    clipped = ((max(a - lead, 0), min(b - lead, len(stripped))) for a, b in ranges)
    return stripped, tuple((a, b) for a, b in clipped if a < b)


def _join_body(
    parts: list[tuple[str, tuple[tuple[int, int], ...]]],
) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Joins the body text blocks (exactly one is expected) with a blank line and
    moves their highlight ranges along."""
    texts: list[str] = []
    ranges: list[tuple[int, int]] = []
    offset = 0
    for text, spans in parts:
        if texts:
            offset += 2  # the "\n\n" in between
        ranges.extend((offset + a, offset + b) for a, b in spans)
        texts.append(text)
        offset += len(text)
    return "\n\n".join(texts), tuple(ranges)


def _parse_content(
    content: bytes, open_block: int | None = None
) -> tuple[list[_Block], list[bytes]]:
    """Splits a content section into its blocks and data sections (clause 5.3).

    A block starts with its JML code and ends with the "end" code, the next JML code or
    the end of the content. Text outside any block is dropped, as is every escape
    sequence without visible effect. Unknown escape codes are taken to be one byte
    long; the standard defines no length for them.

    Args:
        open_block: JML code of a block considered open from the start, for texts that
            stand on their own (the label of a link).

    Returns:
        `(blocks, data_section_payloads)`, the payloads in order of appearance.
    """
    blocks: list[_Block] = []
    sections: list[bytes] = []
    block = None
    if open_block is not None:
        block = _Block(open_block)
        blocks.append(block)

    index = 0
    while index < len(content):
        code = content[index]
        if code >= _FIRST_TEXT_BYTE:
            end = index + 1
            while end < len(content) and content[end] >= _FIRST_TEXT_BYTE:
                end += 1
            if block is not None:
                block.add(content[index:end])
            index = end
            continue

        index += 1
        if code in (_ESC_DATA_SECTION_START, _ESC_DATA_SECTION_CONTINUATION):
            payload, index = _read_data_section(content, index - 1)
            # A continuation is only valid right behind a full block, where the start
            # has already collected it; a stray one is skipped.
            if payload is not None and code == _ESC_DATA_SECTION_START:
                sections.append(payload)
        elif code in (_ESC_EXTENDED_CODE_BEGIN, _ESC_EXTENDED_CODE_END):
            index += 1  # the extended code itself; none is defined yet (clause 5.3.2.1)
        elif code in (_ESC_LINE_BREAK, _ESC_END_OF_INTRODUCTION):
            # The end of the introductory section separates a teaser from the details;
            # a new line shows that without hiding anything.
            if block is not None:
                block.add(b"\n")
        elif code in (_ESC_HIGHLIGHT_START, _ESC_HIGHLIGHT_STOP):
            # Kept as a mark in the text; `_visible_text` turns it into a range.
            if block is not None:
                block.add(bytes([code]))
        elif code >= _ESC_LINE_BREAK:
            pass  # preferred word break, reserved codes
        elif code == _JML_LIST_COLUMN:
            if block is not None:
                block.next_column()
        elif code == _JML_END:
            block = None
        else:
            target = None
            if code == _JML_LINK_ITEM:
                # The target ID is binary and may well contain bytes below 0x20.
                if index + 2 > len(content):
                    break
                target = (content[index] << 8) | content[index + 1]
                index += 2
            block = _Block(code, target)
            blocks.append(block)
    return blocks, sections


def _read_data_section(content: bytes, index: int) -> tuple[bytes | None, int]:
    """Reads the data section whose escape code sits at `index`.

    Each escape is followed by a length byte (number of bytes - 1) and the data. A
    payload of more than 256 bytes continues in escapes with code 0x1B, each directly
    behind a completely filled block (clause 5.3.2).

    Returns:
        `(payload, index behind the data section)`; the payload is None if the content
        section ends inside it.
    """
    payload = bytearray()
    while True:
        if index + 2 > len(content):
            return None, len(content)
        length = content[index + 1] + 1
        end = index + 2 + length
        if end > len(content):
            return None, len(content)
        payload += content[index + 2 : end]
        index = end
        if (
            length < _DATA_SECTION_BLOCK
            or index >= len(content)
            or content[index] != _ESC_DATA_SECTION_CONTINUATION
        ):
            return bytes(payload), index


def _evaluate_data_sections(
    sections: list[bytes],
) -> tuple[float | None, int | None, list[LinkTarget]]:
    """Picks timeouts and general link targets out of the data section payloads (DS
    type byte, then DS data; clause 5.3.2.2).

    Returns:
        `(absolute_timeout, relative_timeout_minutes, links)`. Of the timeouts, at most
        one each is allowed; should there be more, the first counts.
    """
    absolute_timeout: float | None = None
    relative_timeout: int | None = None
    links: list[LinkTarget] = []
    for payload in sections:
        ds_type, ds_data = payload[0], payload[1:]
        if ds_type == _DS_ABSOLUTE_TIMEOUT and len(ds_data) >= 3:
            if absolute_timeout is None:
                quarter_hours = int.from_bytes(ds_data[:3], "big")
                absolute_timeout = float(_TIMEOUT_EPOCH + quarter_hours * _TIMEOUT_UNIT_SECONDS)
        elif ds_type == _DS_RELATIVE_TIMEOUT and len(ds_data) >= 2:
            # A timeout of 0 minutes would make the object void on arrival; like the
            # TOC timeout (clause 6.2.2) it is taken as "no timeout".
            minutes = int.from_bytes(ds_data[:2], "big")
            if relative_timeout is None and minutes:
                relative_timeout = minutes
        elif ds_type == _DS_GENERAL_LINK:
            link = _parse_link(ds_data)
            if link is not None:
                links.append(link)
    return absolute_timeout, relative_timeout, links


def _parse_link(data: bytes) -> LinkTarget | None:
    """A general link target: link type, address, then optionally an end marker 0x00
    and a label (clause 5.3.2.3). Unknown link types are to be ignored."""
    if not data:
        return None
    try:
        link_type = LinkType(data[0])
    except ValueError:
        return None
    rest = data[1:]

    target = None
    if link_type is LinkType.JML_OBJECT:
        # Binary object ID: it may contain 0x00, so the end marker is found by position.
        if len(rest) < 2:
            return None
        target = (rest[0] << 8) | rest[1]
        address = ""
        raw_label = rest[3:]
    else:
        raw_address, _, raw_label = rest.partition(b"\x00")
        address = _decode_text(raw_address)
        if not address:
            return None

    # The label is formatted like any visible text and may carry escape sequences.
    label_blocks, _ = _parse_content(raw_label, open_block=_JML_TITLE)
    return LinkTarget(link_type, address, label_blocks[0].text, target)


@dataclass(frozen=True)
class JournalineParameters:
    """The Journaline specific user application data of FIG 0/13 (clause 8.1.2)."""

    version: int  # 0x00 today; later versions stay backwards compatible
    extended_header_length: int  # bytes, the same for every object of the service
    entry_object: int = ROOT_OBJECT_ID  # the default point of entry


def parse_user_application_data(data: bytes) -> JournalineParameters | None:
    """Version (8 bits), length of the extended header (8 bits), optionally the object
    ID of the point of entry (16 bits); later additions are ignored (clause 8.1.2).

    Deutschlandfunk sends `00 00 00 00`: version 0, no extended header, entry 0x0000.
    """
    if len(data) < 2:
        return None
    entry = int.from_bytes(data[2:4], "big") if len(data) >= 4 else ROOT_OBJECT_ID
    return JournalineParameters(version=data[0], extended_header_length=data[1], entry_object=entry)


class JournalineDecoder:
    """Decodes the JML objects in the X-PAD subfields of the Journaline application.

    `xpad_app_type` is the lowest X-PAD application type that FIG 0/13 announces for
    Journaline; its data groups continue in the next type.
    """

    def __init__(
        self, xpad_app_type: int = XPAD_APP_JOURNALINE_START, extended_header_length: int = 0
    ) -> None:
        self._assembler = XPadDataGroupAssembler(xpad_app_type, xpad_app_type + 1)
        self._extended_header_length = extended_header_length

    def reset(self) -> None:
        self._assembler.reset()

    def process(self, subfields: list[XPadSubfield]) -> list[JmlObject]:
        """Processes the subfields of one access unit or audio frame.

        Returns:
            The objects completed in this step (usually none).
        """
        objects = []
        for group in self._assembler.process(subfields):
            jml_object = decode_data_group(group, self._extended_header_length)
            if jml_object is not None:
                objects.append(jml_object)
        return objects


class Change(Enum):
    """What a received object means for the store."""

    NEW = auto()
    UPDATED = auto()
    UNCHANGED = auto()


@dataclass
class _StoredObject:
    jml_object: JmlObject
    received_at: float  # POSIX time of the most recent reception

    @property
    def expires_at(self) -> float | None:
        """Clause 5.3.2.2: the absolute timeout takes precedence over the relative one,
        which restarts with every reception."""
        jml_object = self.jml_object
        if jml_object.absolute_timeout is not None:
            return jml_object.absolute_timeout
        if jml_object.relative_timeout_minutes is not None:
            return self.received_at + jml_object.relative_timeout_minutes * 60
        return None


class JournalineStore:
    """The received objects of one service by object ID, the receiver's cache
    (clause 7.2.5). Objects past their timeout are no longer handed out."""

    def __init__(self) -> None:
        self._objects: dict[int, _StoredObject] = {}

    def __len__(self) -> int:
        return len(self._objects)

    def clear(self) -> None:
        self._objects.clear()

    def add(self, jml_object: JmlObject, now: float) -> Change:
        """Stores a received object.

        Clause 5.2 suggests detecting a change by comparing header and size; comparing
        the decoded object is stricter and just as cheap here.
        """
        stored = self._objects.get(jml_object.object_id)
        self._objects[jml_object.object_id] = _StoredObject(jml_object, now)
        if stored is None:
            return Change.NEW
        return Change.UNCHANGED if stored.jml_object == jml_object else Change.UPDATED

    def get(self, object_id: int, now: float) -> JmlObject | None:
        stored = self._objects.get(object_id)
        if stored is None or _expired(stored, now):
            return None
        return stored.jml_object

    def received_at(self, object_id: int) -> float | None:
        stored = self._objects.get(object_id)
        return None if stored is None else stored.received_at

    def expires_at(self, object_id: int) -> float | None:
        stored = self._objects.get(object_id)
        return None if stored is None else stored.expires_at

    def objects(self, now: float) -> list[JmlObject]:
        """All valid objects in ascending order of their IDs."""
        return [
            stored.jml_object
            for _, stored in sorted(self._objects.items())
            if not _expired(stored, now)
        ]

    def expire(self, now: float) -> list[int]:
        """Removes the objects past their timeout and returns their IDs."""
        expired = [
            object_id for object_id, stored in self._objects.items() if _expired(stored, now)
        ]
        for object_id in expired:
            del self._objects[object_id]
        return expired


def _expired(stored: _StoredObject, now: float) -> bool:
    expires_at = stored.expires_at
    return expires_at is not None and now >= expires_at


class NavigationHistory:
    """The history path (clauses 4.3 and 7.2.1): the object IDs from the point of entry
    to the object on display, so that the user can go back."""

    def __init__(self, entry: int = ROOT_OBJECT_ID) -> None:
        self._entry = entry
        self._path = [entry]

    @property
    def entry(self) -> int:
        return self._entry

    def set_entry(self, entry: int) -> None:
        """Takes the point of entry FIG 0/13 names (clause 8.1.2). If the user has not
        navigated away from the previous one yet, the new one is shown right away."""
        if self._path == [self._entry]:
            self._path = [entry]
        self._entry = entry

    @property
    def path(self) -> tuple[int, ...]:
        return tuple(self._path)

    @property
    def current(self) -> int:
        return self._path[-1]

    def follow(self, object_id: int) -> None:
        """Goes to a linked object. A link to an object already on the path, typically
        back to the main menu, shortens the path to it instead of lengthening it."""
        if object_id in self._path:
            del self._path[self._path.index(object_id) + 1 :]
        else:
            self._path.append(object_id)

    def back(self) -> None:
        """One step back. With nothing left to go back to, the root object follows."""
        if len(self._path) > 1:
            self._path.pop()
        else:
            self._path = [ROOT_OBJECT_ID]

    def home(self) -> None:
        self._path = [self._entry]
