"""Tests for the dynamic label decoder and DL Plus."""

from __future__ import annotations

import pytest

from dab.charset import CHARSET_EBU_LATIN, CHARSET_UTF8, EBU_LATIN
from dab.crc import crc16_ccitt
from dab.dls import DlPlusContentType, DynamicLabel, DynamicLabelDecoder
from dab.pad import XPAD_APP_DLS_CONTINUATION, XPAD_APP_DLS_START, XPadSubfield

TITLE = DlPlusContentType.ITEM_TITLE
ARTIST = DlPlusContentType.ITEM_ARTIST
ALBUM = DlPlusContentType.ITEM_ALBUM
PROGRAMME_NOW = DlPlusContentType.PROGRAMME_NOW
PROGRAMME_HOST = DlPlusContentType.PROGRAMME_HOST
STATIONNAME_LONG = DlPlusContentType.STATIONNAME_LONG
INFO_URL = DlPlusContentType.INFO_URL

_EBU_LATIN_CODES = {character: code for code, character in enumerate(EBU_LATIN) if character}


def _segment(
    text: bytes,
    *,
    first: bool = True,
    last: bool = True,
    index: int = 0,
    toggle: bool = False,
    charset: int = 0,
) -> bytes:
    """Builds a DLS data group: two bytes of prefix, the text, two bytes of CRC."""
    prefix = bytes(
        [
            (0x80 if toggle else 0)
            | (0x40 if first else 0)
            | (0x20 if last else 0)
            | (len(text) - 1),
            (charset << 4) if first else (index << 4),
        ]
    )
    body = prefix + text
    crc = crc16_ccitt(body) ^ 0xFFFF
    return body + bytes([crc >> 8, crc & 0xFF])


def _command(command: int, *, toggle: bool = False) -> bytes:
    prefix = bytes([(0x80 if toggle else 0) | 0x40 | 0x20 | 0x10 | command, 0])
    crc = crc16_ccitt(prefix) ^ 0xFFFF
    return prefix + bytes([crc >> 8, crc & 0xFF])


def _feed_group(decoder: DynamicLabelDecoder, group: bytes, chunk: int = 0) -> DynamicLabel | None:
    """Feeds in a data group, either in one piece or in chunks as it would arrive
    spread over several access units."""
    if chunk == 0:
        return decoder.process([XPadSubfield(XPAD_APP_DLS_START, group)])
    result = decoder.process([XPadSubfield(XPAD_APP_DLS_START, group[:chunk])])
    for offset in range(chunk, len(group), chunk):
        result = decoder.process(
            [XPadSubfield(XPAD_APP_DLS_CONTINUATION, group[offset : offset + chunk])]
        )
    return result


def _feed(decoder: DynamicLabelDecoder, group: bytes, chunk: int = 0) -> str | None:
    """Like `_feed_group`, but only the text of the result."""
    result = _feed_group(decoder, group, chunk)
    return None if result is None else result.text


def _send_text(
    decoder: DynamicLabelDecoder, text: str, *, toggle: bool, charset: int = CHARSET_EBU_LATIN
) -> str | None:
    """Sends a whole text in segments of at most 16 bytes, as a station does."""
    if charset == CHARSET_EBU_LATIN:
        raw = bytes(_EBU_LATIN_CODES[character] for character in text)
    else:
        raw = text.encode()
    chunks = [raw[offset : offset + 16] for offset in range(0, len(raw), 16)]
    result = None
    for index, chunk in enumerate(chunks):
        group = _segment(
            chunk,
            first=index == 0,
            last=index == len(chunks) - 1,
            index=index,
            toggle=toggle,
            charset=charset,
        )
        result = _feed(decoder, group)
    return result


def _dl_plus_group(
    command: bytes, *, link: bool, first: bool = True, last: bool = True, index: int = 0
) -> bytes:
    """Builds a DL Plus command data group (ETSI TS 102 980, clause 7.1). Like the
    stations here, the toggle bit of the prefix repeats the link bit."""
    prefix = bytes(
        [
            (0x80 if link else 0) | (0x40 if first else 0) | (0x20 if last else 0) | 0x10 | 0x02,
            (0x80 if link else 0) | (0 if first else index << 4) | (len(command) - 1),
        ]
    )
    body = prefix + command
    crc = crc16_ccitt(body) ^ 0xFFFF
    return body + bytes([crc >> 8, crc & 0xFF])


def _tags(
    *tags: tuple[int, int, int], item_toggle: bool = False, item_running: bool = True
) -> bytes:
    """A DL Plus tags command; one (content type, start, number of characters) per tag."""
    head = (0x08 if item_toggle else 0) | (0x04 if item_running else 0) | (len(tags) - 1)
    return bytes([head]) + b"".join(
        bytes([kind, start, length - 1]) for kind, start, length in tags
    )


def test_single_segment_label():
    decoder = DynamicLabelDecoder()
    assert _feed(decoder, _segment(b"Radio Test")) == "Radio Test"
    assert decoder.label == "Radio Test"


def test_label_spread_over_several_access_units():
    """A data group arrives in pieces: first the piece marked "start", then
    continuations."""
    decoder = DynamicLabelDecoder()
    assert _feed(decoder, _segment(b"Stueckweise"), chunk=4) == "Stueckweise"


def test_label_assembled_from_multiple_segments():
    decoder = DynamicLabelDecoder()

    assert _feed(decoder, _segment(b"Erster Teil ", first=True, last=False)) is None
    result = _feed(decoder, _segment(b"zweiter Teil", first=False, last=True, index=1))

    assert result == "Erster Teil zweiter Teil"


def test_incomplete_label_is_not_reported():
    """If a segment in the middle is missing, no text may result."""
    decoder = DynamicLabelDecoder()

    _feed(decoder, _segment(b"Anfang..", first=True, last=False))
    result = _feed(decoder, _segment(b"..Ende..", first=False, last=True, index=2))

    assert result is None, "segment 1 is missing, the text is incomplete"


def test_corrupted_segment_is_discarded():
    """Every data group carries its own CRC: a flipped bit must not display garbage."""
    decoder = DynamicLabelDecoder()
    group = bytearray(_segment(b"Kaputt"))
    group[3] ^= 0x01

    assert _feed(decoder, bytes(group)) is None
    assert decoder.label is None


def test_new_label_discards_leftovers_of_the_previous_one():
    """The toggle bit flips with every new text. Fragments of the old one must not
    mix with the new one."""
    decoder = DynamicLabelDecoder()
    _feed(decoder, _segment(b"Alter Text.", first=True, last=False, toggle=False))

    result = _feed(decoder, _segment(b"Neuer Text", first=True, last=True, toggle=True))

    assert result == "Neuer Text"


def test_clear_display_command_empties_the_label():
    decoder = DynamicLabelDecoder()
    _feed(decoder, _segment(b"Etwas"))

    assert _feed(decoder, _command(1)) == ""
    assert decoder.label == ""


def test_utf8_charset_is_decoded():
    decoder = DynamicLabelDecoder()
    text = "Grüße".encode()
    assert _feed(decoder, _segment(text, charset=15)) == "Grüße"


def test_ebu_latin_charset_is_decoded():
    decoder = DynamicLabelDecoder()
    assert _feed(decoder, _segment(b"Gr\x99\x8de", charset=0)) == "Grüße"


def test_continuation_without_start_is_ignored():
    """Joining in the middle of a data group, the start is missing: the fragments
    must not pass as text."""
    decoder = DynamicLabelDecoder()
    group = _segment(b"Mittendrin")

    result = decoder.process([XPadSubfield(XPAD_APP_DLS_CONTINUATION, group[4:])])

    assert result is None


def test_reset_clears_everything():
    decoder = DynamicLabelDecoder()
    _feed(decoder, _segment(b"Vorher"))
    decoder.reset()
    assert decoder.label is None


# --- DL Plus (ETSI TS 102 980) --------------------------------------------------------

# Complete DL Plus data groups (prefix, command, CRC) as received on air in September
# 2026, with the text they link to.
ABSOLUT_RELAX = (
    "72060d010709040003dd50",
    "A-HA - TAKE ON ME",
)
DLF_ITEM = (
    "72060501004b044e0f4592",
    (
        "Andruck, 28.09.'26: Niewiedervereinigung / Nostalgie / Deutsche Diskrepanzen, "
        "Catrin Stövesand"
    ),
)
DLF_PROGRAMME = (
    "f2860521002d24330fb421",
    "Andruck - das Magazin für Politische Literatur mit Catrin Stövesand",
)
BALLERMANN_RADIO = (
    "f2890601080d041d2520490fa42b",
    "Jetzt: «Hardcore Breit» von «Honk! x Calvin Kleinen x Emanuel Bento» auf Ballermann Radio",
)


def _send_on_air(decoder: DynamicLabelDecoder, recording: tuple[str, str]) -> DynamicLabel | None:
    """Sends a recorded text and its DL Plus command in the order the station does:
    first the text, then the tags (clause 7.4)."""
    group_hex, text = recording
    group = bytes.fromhex(group_hex)
    _send_text(decoder, text, toggle=bool(group[1] & 0x80))
    return _feed_group(decoder, group)


@pytest.mark.parametrize(
    ("recording", "item_toggle", "expected"),
    [
        (ABSOLUT_RELAX, True, {TITLE: "TAKE ON ME", ARTIST: "A-HA"}),
        (
            DLF_ITEM,
            False,
            {
                TITLE: "Andruck, 28.09.'26: Niewiedervereinigung / Nostalgie / Deutsche "
                "Diskrepanzen",
                ARTIST: "Catrin Stövesand",
            },
        ),
        (
            DLF_PROGRAMME,
            False,
            {
                PROGRAMME_NOW: "Andruck - das Magazin für Politische Literatur",
                PROGRAMME_HOST: "Catrin Stövesand",
            },
        ),
        (
            BALLERMANN_RADIO,
            False,
            {
                TITLE: "Hardcore Breit",
                ARTIST: "Honk! x Calvin Kleinen x Emanuel Bento",
                STATIONNAME_LONG: "Ballermann Radio",
            },
        ),
    ],
    ids=["absolut-relax", "dlf-item", "dlf-programme", "ballermann-radio"],
)
def test_dl_plus_objects_from_recorded_commands(recording, item_toggle, expected):
    decoder = DynamicLabelDecoder()

    result = _send_on_air(decoder, recording)

    assert result is not None
    assert result.text == recording[1]
    assert result.dl_plus == decoder.dl_plus
    assert decoder.dl_plus.objects == expected
    assert decoder.dl_plus.item_toggle is item_toggle
    assert decoder.dl_plus.item_running is True


def test_dl_plus_title_and_artist_shortcuts():
    decoder = DynamicLabelDecoder()
    _send_on_air(decoder, ABSOLUT_RELAX)

    assert decoder.dl_plus.title == "TAKE ON ME"
    assert decoder.dl_plus.artist == "A-HA"


def test_dl_plus_markers_count_characters_not_bytes():
    """Clause 6.0: the markers count characters. In UTF-8 the "ü" takes two bytes;
    counting bytes would shift everything behind it by one."""
    group_hex, text = DLF_PROGRAMME
    decoder = DynamicLabelDecoder()
    _send_text(decoder, text, toggle=True, charset=CHARSET_UTF8)

    _feed_group(decoder, bytes.fromhex(group_hex))

    assert decoder.dl_plus.objects == {
        PROGRAMME_NOW: "Andruck - das Magazin für Politische Literatur",
        PROGRAMME_HOST: "Catrin Stövesand",
    }


def test_dl_plus_objects_outlive_the_text():
    """The Dlf alternates a text about the item and one about the programme. Each
    object stays valid until one of the same content type replaces it (clause 5.3)."""
    decoder = DynamicLabelDecoder()
    _send_on_air(decoder, DLF_ITEM)

    result = _send_on_air(decoder, DLF_PROGRAMME)

    assert result.text == DLF_PROGRAMME[1]
    assert set(result.dl_plus.objects) == {TITLE, ARTIST, PROGRAMME_NOW, PROGRAMME_HOST}
    assert result.dl_plus.artist == "Catrin Stövesand"


def test_new_text_reports_the_objects_still_valid():
    """A new text arrives before its tags. Until they follow, the label carries the
    objects of the previous text."""
    decoder = DynamicLabelDecoder()
    _send_on_air(decoder, ABSOLUT_RELAX)

    result = _feed_group(decoder, _segment(b"Neuer Text", toggle=True))

    assert result.text == "Neuer Text"
    assert result.dl_plus.title == "TAKE ON ME"


def test_dl_plus_tags_wait_for_the_text_they_link_to():
    """Reception may start between a text and its tags; the station repeats both. Tags
    whose text is still missing are applied as soon as it is complete."""
    group_hex, text = ABSOLUT_RELAX
    decoder = DynamicLabelDecoder()

    assert _feed_group(decoder, bytes.fromhex(group_hex)) is None
    assert decoder.dl_plus is None

    _send_text(decoder, text, toggle=False)
    assert decoder.dl_plus.objects == {TITLE: "TAKE ON ME", ARTIST: "A-HA"}


def test_dl_plus_tags_of_another_text_are_not_applied():
    """The link bit must equal the toggle bit of the text, otherwise the positions
    refer to a different text (clause 8.1)."""
    decoder = DynamicLabelDecoder()
    _send_text(decoder, "Etwas ganz anderes", toggle=True)

    assert _feed_group(decoder, bytes.fromhex(ABSOLUT_RELAX[0])) is None
    assert decoder.dl_plus is None


def test_waiting_tags_are_dropped_when_another_text_begins():
    decoder = DynamicLabelDecoder()
    _feed_group(decoder, bytes.fromhex(ABSOLUT_RELAX[0]))  # link bit 0, text still missing
    _send_text(decoder, "Dazwischen", toggle=True)  # a newer text

    _send_text(decoder, ABSOLUT_RELAX[1], toggle=False)  # the toggle bit has flipped back

    assert decoder.dl_plus is None


def test_item_toggle_ends_the_item_objects():
    """A flipped item toggle bit means a new programme item: its predecessor's title,
    artist etc. are gone, objects of other categories stay (clause 5.3)."""
    decoder = DynamicLabelDecoder()
    _send_text(decoder, "A-HA - TAKE ON ME / Radio X", toggle=False)
    _feed_group(
        decoder,
        _dl_plus_group(
            _tags((TITLE, 7, 10), (ARTIST, 0, 4), (STATIONNAME_LONG, 20, 7)), link=False
        ),
    )

    _send_text(decoder, "Nena - 99 Luftballons", toggle=True)
    _feed_group(decoder, _dl_plus_group(_tags((TITLE, 7, 14), item_toggle=True), link=True))

    assert decoder.dl_plus.objects == {TITLE: "99 Luftballons", STATIONNAME_LONG: "Radio X"}


def test_same_item_collects_objects_from_several_texts():
    """While the item toggle bit stays, objects of the same item add up (clause 5.2.1)."""
    decoder = DynamicLabelDecoder()
    _send_text(decoder, "Jetzt: TAKE ON ME", toggle=False)
    _feed_group(decoder, _dl_plus_group(_tags((TITLE, 7, 10)), link=False))

    _send_text(decoder, "von A-HA", toggle=True)
    _feed_group(decoder, _dl_plus_group(_tags((ARTIST, 4, 4)), link=True))

    assert decoder.dl_plus.objects == {TITLE: "TAKE ON ME", ARTIST: "A-HA"}


def test_no_item_running_clears_the_item_objects():
    """Item running bit 0: news or the presenter, no title on air. Item objects are
    dropped and not created, the others remain (clause 5.2.1)."""
    decoder = DynamicLabelDecoder()
    _send_on_air(decoder, BALLERMANN_RADIO)

    _send_text(decoder, "Die Nachrichten: TITEL", toggle=False)
    result = _feed_group(
        decoder,
        _dl_plus_group(_tags((TITLE, 17, 5), item_running=False), link=False),
    )

    assert result.dl_plus.item_running is False
    assert result.dl_plus.title is None
    assert result.dl_plus.objects == {STATIONNAME_LONG: "Ballermann Radio"}


def test_dummy_tag_carries_only_the_item_flags():
    decoder = DynamicLabelDecoder()
    _send_text(decoder, "Hallo", toggle=False)

    _feed_group(decoder, _dl_plus_group(_tags((DlPlusContentType.DUMMY, 0, 1)), link=False))

    assert decoder.dl_plus.objects == {}
    assert decoder.dl_plus.item_running is True


def test_delete_object_removes_that_content_type():
    """Clause 6.2: a tag of length 1 on a blank deletes the object of its content type.
    For the Item category all its objects end together (clause 8.1)."""
    decoder = DynamicLabelDecoder()
    _send_text(decoder, "A-HA - TAKE ON ME / www.radio.de", toggle=False)
    _feed_group(
        decoder,
        _dl_plus_group(_tags((TITLE, 7, 10), (ARTIST, 0, 4), (INFO_URL, 20, 12)), link=False),
    )

    _send_text(decoder, "Hotline: 0123", toggle=True)
    _feed_group(decoder, _dl_plus_group(_tags((INFO_URL, 8, 1)), link=True))
    assert decoder.dl_plus.objects == {TITLE: "TAKE ON ME", ARTIST: "A-HA"}

    _feed_group(decoder, _dl_plus_group(_tags((ALBUM, 8, 1)), link=True))
    assert decoder.dl_plus.objects == {}


@pytest.mark.parametrize(
    "tag",
    [
        (54, 0, 4),  # reserved content type
        (64, 0, 4),  # reserved for future DL Plus additions
        (DlPlusContentType.INFO_DATE_TIME, 0, 4),  # meant for RT+ receivers only
        (TITLE, 10, 10),  # reaches beyond the end of the text
    ],
)
def test_unusable_tags_create_no_object(tag):
    decoder = DynamicLabelDecoder()
    _send_text(decoder, "A-HA - TAKE", toggle=False)

    _feed_group(decoder, _dl_plus_group(_tags(tag), link=False))

    assert decoder.dl_plus.objects == {}


def test_reserved_dl_plus_command_is_ignored():
    decoder = DynamicLabelDecoder()
    _send_text(decoder, "A-HA - TAKE ON ME", toggle=False)
    command = bytearray(_tags((TITLE, 7, 10)))
    command[0] |= 0x10  # command identifier 1 is not defined

    assert _feed_group(decoder, _dl_plus_group(bytes(command), link=False)) is None
    assert decoder.dl_plus is None


def test_corrupted_dl_plus_command_is_discarded():
    decoder = DynamicLabelDecoder()
    _send_text(decoder, ABSOLUT_RELAX[1], toggle=False)
    group = bytearray.fromhex(ABSOLUT_RELAX[0])
    group[4] ^= 0x01

    assert _feed_group(decoder, bytes(group)) is None
    assert decoder.dl_plus is None


def test_dl_plus_command_spread_over_two_segments():
    """Clause 7.1: a command may be segmented like a text; from the second segment on,
    field 2 carries the segment number."""
    decoder = DynamicLabelDecoder()
    _send_text(decoder, ABSOLUT_RELAX[1], toggle=False)
    command = _tags((TITLE, 7, 10), (ARTIST, 0, 4))

    _feed_group(decoder, _dl_plus_group(command[:4], link=False, last=False))
    assert decoder.dl_plus is None
    _feed_group(decoder, _dl_plus_group(command[4:], link=False, first=False, index=1))

    assert decoder.dl_plus.objects == {TITLE: "TAKE ON ME", ARTIST: "A-HA"}


def test_dl_plus_command_arriving_in_chunks():
    decoder = DynamicLabelDecoder()
    _send_text(decoder, ABSOLUT_RELAX[1], toggle=False)

    result = _feed_group(decoder, bytes.fromhex(ABSOLUT_RELAX[0]), chunk=3)

    assert result.dl_plus.title == "TAKE ON ME"


def test_clear_display_keeps_the_dl_plus_objects():
    decoder = DynamicLabelDecoder()
    _send_on_air(decoder, ABSOLUT_RELAX)

    result = _feed_group(decoder, _command(1))

    assert result.text == ""
    assert result.dl_plus.title == "TAKE ON ME"


def test_reset_forgets_the_dl_plus_objects():
    decoder = DynamicLabelDecoder()
    _send_on_air(decoder, ABSOLUT_RELAX)

    decoder.reset()

    assert decoder.dl_plus is None
