"""Dynamic Label (DLS): the scrolling text with station and title information.

The text arrives in segments of at most 16 characters, spread over the X-PAD subfields
of applications 2 (segment start) and 3 (continuation), see `dab.pad`. A complete text
consists of up to eight such segments (ETSI EN 300 401, clause 7.4.5).

Two things make the assembly robust:

* Every segment carries its own CRC. A corrupted segment is discarded instead of
  displaying garbage.
* A toggle bit flips whenever a new text begins. It reveals that the segments collected
  so far belong to an old text and must be discarded.

DL Plus (ETSI TS 102 980) travels in the same data groups, as a DL command in place of
a text segment. It tags parts of the text with a content type (title, artist,
programme, ...), so that a receiver can show them separately. A tag only holds the
position of its part within the text; the link bit of the command ties it to the text
whose toggle bit has the same value. The parts cut out this way, the DL Plus objects,
outlive the text: the station alternates texts, and each object stays valid until one
of the same content type replaces it (clause 5.3). Objects of the Item category (title,
artist, ...) belong to the programme item on air and are dropped together when a new
item starts or none is running (clause 5.2.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from dab.charset import CHARSET_UTF8, CHARSET_UTF16, decode_charset
from dab.crc import check_crc
from dab.pad import XPAD_APP_DLS_CONTINUATION, XPAD_APP_DLS_START, XPadSubfield

_PREFIX_LENGTH = 2
_CRC_LENGTH = 2
_MAX_SEGMENTS = 8

# ETSI EN 300 401, table 32: commands sent in place of a text segment.
_COMMAND_CLEAR_DISPLAY = 1
_COMMAND_DL_PLUS = 2  # ETSI TS 102 980, clause 7.1

# ETSI TS 102 980, clause 7.2: the only DL Plus command defined so far.
_DL_PLUS_TAGS_COMMAND = 0
_DL_PLUS_TAG_LENGTH = 3


class DlPlusContentType(IntEnum):
    """DL Plus content types (ETSI TS 102 980, annex A, table A.1).

    Codes 0 to 63 are the same as in RDS RadioText Plus; 54 and 55 as well as 64 to
    127 are reserved.
    """

    DUMMY = 0
    ITEM_TITLE = 1
    ITEM_ALBUM = 2
    ITEM_TRACKNUMBER = 3
    ITEM_ARTIST = 4
    ITEM_COMPOSITION = 5
    ITEM_MOVEMENT = 6
    ITEM_CONDUCTOR = 7
    ITEM_COMPOSER = 8
    ITEM_BAND = 9
    ITEM_COMMENT = 10
    ITEM_GENRE = 11
    INFO_NEWS = 12
    INFO_NEWS_LOCAL = 13
    INFO_STOCKMARKET = 14
    INFO_SPORT = 15
    INFO_LOTTERY = 16
    INFO_HOROSCOPE = 17
    INFO_DAILY_DIVERSION = 18
    INFO_HEALTH = 19
    INFO_EVENT = 20
    INFO_SCENE = 21
    INFO_CINEMA = 22
    INFO_TV = 23
    INFO_DATE_TIME = 24
    INFO_WEATHER = 25
    INFO_TRAFFIC = 26
    INFO_ALARM = 27
    INFO_ADVERTISEMENT = 28
    INFO_URL = 29
    INFO_OTHER = 30
    STATIONNAME_SHORT = 31
    STATIONNAME_LONG = 32
    PROGRAMME_NOW = 33
    PROGRAMME_NEXT = 34
    PROGRAMME_PART = 35
    PROGRAMME_HOST = 36
    PROGRAMME_EDITORIAL_STAFF = 37
    PROGRAMME_FREQUENCY = 38
    PROGRAMME_HOMEPAGE = 39
    PROGRAMME_SUBCHANNEL = 40
    PHONE_HOTLINE = 41
    PHONE_STUDIO = 42
    PHONE_OTHER = 43
    SMS_STUDIO = 44
    SMS_OTHER = 45
    EMAIL_HOTLINE = 46
    EMAIL_STUDIO = 47
    EMAIL_OTHER = 48
    MMS_OTHER = 49
    CHAT = 50
    CHAT_CENTER = 51
    VOTE_QUESTION = 52
    VOTE_CENTRE = 53
    PRIVATE_1 = 56
    PRIVATE_2 = 57
    PRIVATE_3 = 58
    DESCRIPTOR_PLACE = 59
    DESCRIPTOR_APPOINTMENT = 60
    DESCRIPTOR_IDENTIFIER = 61
    DESCRIPTOR_PURCHASE = 62
    DESCRIPTOR_GET_DATA = 63

    @property
    def is_item(self) -> bool:
        """Belongs to the programme item on air (clause 5.1, category Item)."""
        return DlPlusContentType.ITEM_TITLE <= self <= DlPlusContentType.ITEM_GENRE


# Table A.1, note 6: only meant for RT+ receivers, a DL Plus receiver ignores them.
_RT_PLUS_ONLY = frozenset(
    {
        DlPlusContentType.INFO_DATE_TIME,
        DlPlusContentType.PROGRAMME_FREQUENCY,
        DlPlusContentType.PROGRAMME_SUBCHANNEL,
    }
)


@dataclass(frozen=True)
class DlPlus:
    """The DL Plus objects currently valid for the service, together with the item
    flags of the most recent tags command (ETSI TS 102 980, clauses 5.2.1 and 5.3)."""

    item_toggle: bool  # flips at the start of every programme item
    item_running: bool  # False: no item on air (e.g. news or the presenter talking)
    objects: dict[DlPlusContentType, str]

    @property
    def title(self) -> str | None:
        return self.objects.get(DlPlusContentType.ITEM_TITLE)

    @property
    def artist(self) -> str | None:
        return self.objects.get(DlPlusContentType.ITEM_ARTIST)


@dataclass(frozen=True)
class DynamicLabel:
    """The current dynamic label: the text and, once the station has sent DL Plus for
    the service, the DL Plus objects."""

    text: str
    dl_plus: DlPlus | None = None


@dataclass(frozen=True)
class _Segment:
    index: int
    last: bool
    charset: int  # only meaningful in the first segment of a text
    data: bytes


class DynamicLabelDecoder:
    """Assembles the dynamic label and its DL Plus objects from the X-PAD subfields.

    `process()` only reports a text once all segments of a label have arrived and their
    checksums are correct.
    """

    def __init__(self) -> None:
        self._group = bytearray()
        self._collecting = False
        # Segments of the text being received, and of the DL Plus command that is
        # waiting for (or has been applied to) the text its link bit names.
        self._segments: dict[int, _Segment] = {}
        self._toggle: bool | None = None
        self._command_segments: dict[int, _Segment] = {}
        self._link: bool | None = None
        # The most recently completed text, split into characters for the DL Plus tags.
        self._label: str | None = None
        self._label_toggle: bool | None = None
        self._characters: list[str] = []
        self._dl_plus: DlPlus | None = None

    @property
    def label(self) -> str | None:
        """The most recently completed text."""
        return self._label

    @property
    def dl_plus(self) -> DlPlus | None:
        """The current DL Plus objects, None until the first DL Plus command."""
        return self._dl_plus

    def reset(self) -> None:
        self._group.clear()
        self._collecting = False
        self._segments.clear()
        self._toggle = None
        self._command_segments.clear()
        self._link = None
        self._label = None
        self._label_toggle = None
        self._characters = []
        self._dl_plus = None

    def process(self, subfields: list[XPadSubfield]) -> DynamicLabel | None:
        """Processes the subfields of one access unit.

        Returns:
            The current label as soon as a text is complete or DL Plus tags have been
            applied to it, otherwise None.
        """
        updated = False
        for subfield in subfields:
            if subfield.application_type == XPAD_APP_DLS_START:
                self._group = bytearray(subfield.data)
                self._collecting = True
            elif subfield.application_type == XPAD_APP_DLS_CONTINUATION:
                if not self._collecting:
                    continue  # joined in the middle of a group
                self._group.extend(subfield.data)
            else:
                continue

            if self._take_complete_group():
                updated = True
        if not updated:
            return None
        return DynamicLabel(self._label or "", self._dl_plus)

    def _take_complete_group(self) -> bool:
        """Checks whether the collected bytes form a complete data group and evaluates it.

        Returns:
            True if the label or its DL Plus objects were updated.
        """
        if len(self._group) < _PREFIX_LENGTH:
            return False

        prefix = self._group[0]
        is_command = bool(prefix & 0x10)
        field = prefix & 0x0F
        if not is_command:
            field_length = field + 1
        elif field == _COMMAND_CLEAR_DISPLAY:
            field_length = 0
        elif field == _COMMAND_DL_PLUS:
            # Field 3 in the second prefix byte (ETSI TS 102 980, clause 7.1).
            field_length = (self._group[1] & 0x0F) + 1
        else:
            self._collecting = False  # unknown command, discard the group
            return False

        total = _PREFIX_LENGTH + field_length + _CRC_LENGTH
        if len(self._group) < total:
            return False

        group = bytes(self._group[:total])
        self._group.clear()
        self._collecting = False
        if not check_crc(group):
            return False
        data = group[_PREFIX_LENGTH : _PREFIX_LENGTH + field_length]
        if not is_command:
            return self._add_text_segment(group, data)
        if field == _COMMAND_CLEAR_DISPLAY:
            return self._clear_display()
        return self._add_command_segment(group, data)

    def _add_text_segment(self, group: bytes, text: bytes) -> bool:
        toggle = bool(group[0] & 0x80)
        first = bool(group[0] & 0x40)
        last = bool(group[0] & 0x20)
        # In the first segment the second byte holds the character set bits, in all
        # others the segment number.
        index = 0 if first else (group[1] & 0x70) >> 4
        charset = (group[1] & 0xF0) >> 4 if first else 0

        if self._toggle is not None and toggle != self._toggle:
            self._segments.clear()  # new text, discard the old fragments
        if self._link is not None and toggle != self._link:
            self._command_segments.clear()  # DL Plus tags of an older text
            self._link = None
        self._toggle = toggle
        self._segments[index] = _Segment(index, last, charset, text)

        raw = _join(self._segments)
        if raw is None:
            return False  # at least one segment is still missing
        self._characters = _split_characters(raw, self._segments[0].charset)
        self._label = "".join(self._characters)
        self._label_toggle = toggle
        # The tags may have come first, e.g. when reception started between the text
        # and its command, which the station repeats in turn.
        self._apply_dl_plus()
        return True

    def _clear_display(self) -> bool:
        """Removes the text. The DL Plus objects outlive it, but its tags must not be
        applied to the next text."""
        self._segments.clear()
        self._toggle = None
        self._command_segments.clear()
        self._link = None
        self._label = ""
        self._label_toggle = None
        self._characters = []
        return True

    def _add_command_segment(self, group: bytes, data: bytes) -> bool:
        """Collects a DL Plus command, which may be spread over several segments like a
        text (ETSI TS 102 980, clause 7.1)."""
        first = bool(group[0] & 0x40)
        last = bool(group[0] & 0x20)
        link = bool(group[1] & 0x80)
        # Field 2 holds the segment number in all but the first segment.
        index = 0 if first else (group[1] & 0x70) >> 4

        if first or link != self._link:
            self._command_segments.clear()  # a new command begins
        self._link = link
        self._command_segments[index] = _Segment(index, last, 0, data)
        return self._apply_dl_plus()

    def _apply_dl_plus(self) -> bool:
        """Creates DL Plus objects from the collected command and the text it links to.

        Returns:
            True if a complete tags command could be applied.
        """
        command = _join(self._command_segments)
        if command is None or self._label_toggle is None or self._link != self._label_toggle:
            return False
        if command[0] >> 4 != _DL_PLUS_TAGS_COMMAND:
            return False  # reserved command
        item_toggle = bool(command[0] & 0x08)
        item_running = bool(command[0] & 0x04)
        tag_count = (command[0] & 0x03) + 1
        if len(command) < 1 + tag_count * _DL_PLUS_TAG_LENGTH:
            return False

        previous = self._dl_plus
        objects = {} if previous is None else dict(previous.objects)
        # Clause 5.3: the Item objects live only as long as their programme item.
        if not item_running or (previous is not None and previous.item_toggle != item_toggle):
            _drop_item_objects(objects)
        for offset in range(1, 1 + tag_count * _DL_PLUS_TAG_LENGTH, _DL_PLUS_TAG_LENGTH):
            self._apply_tag(command[offset : offset + _DL_PLUS_TAG_LENGTH], objects, item_running)
        self._dl_plus = DlPlus(item_toggle, item_running, objects)
        return True

    def _apply_tag(
        self, tag: bytes, objects: dict[DlPlusContentType, str], item_running: bool
    ) -> None:
        """Creates, replaces or deletes one object (clauses 5.3 and 6)."""
        try:
            content_type = DlPlusContentType(tag[0] & 0x7F)
        except ValueError:
            return  # reserved content type
        if content_type == DlPlusContentType.DUMMY or content_type in _RT_PLUS_ONLY:
            return
        start = tag[1] & 0x7F
        length = (tag[2] & 0x7F) + 1  # the length marker counts the characters after the first
        if start + length > len(self._characters):
            return  # does not fit the text, so it cannot belong to it
        text = "".join(self._characters[start : start + length])

        # Clause 6.2: a single blank character deletes the object of that type.
        if length == 1 and text.isspace():
            if content_type.is_item:
                _drop_item_objects(objects)  # clause 8.1: all Item objects end together
            else:
                objects.pop(content_type, None)
            return
        if content_type.is_item and not item_running:
            return  # clause 5.2.1: no Item objects while no item is running
        if text:
            objects[content_type] = text


def _join(segments: dict[int, _Segment]) -> bytes | None:
    """The data of all segments in order, or None while one is still missing."""
    last_index = next((segment.index for segment in segments.values() if segment.last), None)
    if last_index is None or last_index >= _MAX_SEGMENTS:
        return None
    if any(index not in segments for index in range(last_index + 1)):
        return None
    return b"".join(segments[index].data for index in range(last_index + 1))


def _split_characters(raw: bytes, charset: int) -> list[str]:
    """Splits a text into its characters, one entry per character position.

    DL Plus tags count characters, not bytes (ETSI TS 102 980, clause 6.0), which
    differ in UTF-8 and UTF-16. The single-byte character sets are decoded byte by
    byte, so that characters which are not displayed (EBU Latin 0x0A, 0x0B, 0x1F)
    still occupy their position.
    """
    if charset in (CHARSET_UTF8, CHARSET_UTF16):
        return list(decode_charset(raw, charset))
    return [decode_charset(bytes([byte]), charset) for byte in raw]


def _drop_item_objects(objects: dict[DlPlusContentType, str]) -> None:
    for content_type in [content_type for content_type in objects if content_type.is_item]:
        del objects[content_type]
