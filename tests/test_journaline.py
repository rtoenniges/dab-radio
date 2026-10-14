"""Tests for the Journaline decoding (ETSI TS 102 979)."""

from __future__ import annotations

import zlib
from types import SimpleNamespace

import pytest

from dab.constants import MODE_I, SAMPLE_RATE
from dab.crc import crc16_ccitt
from dab.fig import parse_fib
from dab.journaline import (
    DATA_GROUP_MANAGEMENT,
    MAX_OBJECT_BYTES,
    ROOT_OBJECT_ID,
    Change,
    JmlObject,
    JournalineDecoder,
    JournalineParameters,
    JournalineStore,
    LinkTarget,
    LinkType,
    MenuItem,
    NavigationHistory,
    ObjectType,
    decode_data_group,
    parse_jml_object,
    parse_user_application_data,
)
from dab.pad import (
    XPAD_APP_DATA_GROUP_LENGTH,
    XPAD_APP_JOURNALINE_CONTINUATION,
    XPAD_APP_JOURNALINE_START,
    XPadSubfield,
)
from dsp.pipeline import DspWorker
from tests.dab_signal import fib

# Complete MSC data groups (CRC included) received from Deutschlandfunk on channel 5C.
# The main menu, uncompressed:
REAL_MENU_GROUP = bytes.fromhex(
    "40d0000031011a15004a4576312e382e325f463630453137354430443041446575747363686c616e64"
    "66756e6b0217704e6163687269636874656e021f4050726f6772616d6d766f7273636861750200014b"
    "6f6e74616b74020002496d7072657373756d2a03"
)
# Object 0x1775, a title-only message of the programme preview, compressed:
REAL_COMPRESSED_GROUP = bytes.fromhex(
    "402017756e08938a6466ca28292928b6d2d72f2f2fd74b492d2d294ecec849cc4b492bcdcb06f2f50b"
    "8af2d38b127373cbf28b803289a57a4616864071bd8c92dc1c8600344985cc5c05cfbc92d4a2bcd412"
    "4623032b030305bfc4e48ca2cce48c92d43c0040a4"
)

TITLE, LINK_ITEM, BODY, LIST_ITEM, COLUMN, END = 0x01, 0x02, 0x03, 0x04, 0x05, 0x00


def _with_crc(data: bytes) -> bytes:
    crc = crc16_ccitt(data) ^ 0xFFFF
    return data + bytes([crc >> 8, crc & 0xFF])


def _object(
    object_id: int,
    object_type: int,
    content: bytes,
    *,
    static: bool = False,
    revision: int = 0,
    compress: bool = False,
) -> bytes:
    """A JML object: standard header, then the (possibly compressed) content section."""
    description = (object_type << 5) | (static << 4) | (compress << 3) | revision
    if compress:
        packer = zlib.compressobj(wbits=-12)
        content = bytes([0x08]) + packer.compress(content) + packer.flush()
    return bytes([object_id >> 8, object_id & 0xFF, description]) + content


def _group(jml_object: bytes, first_byte: int = 0x40) -> bytes:
    """An MSC data group as Journaline sends it: only the CRC flag set, type 0."""
    return _with_crc(bytes([first_byte, 0x00]) + jml_object)


def _data_section(payload: bytes) -> bytes:
    """Data section escapes (clause 5.3.2) for a payload of any length."""
    out = bytearray()
    for index, offset in enumerate(range(0, len(payload), 256)):
        chunk = payload[offset : offset + 256]
        out += bytes([0x1A if index == 0 else 0x1B, len(chunk) - 1]) + chunk
    return bytes(out)


# --- Real objects -------------------------------------------------------------------


def test_real_main_menu():
    menu = decode_data_group(REAL_MENU_GROUP)

    assert menu is not None
    assert menu.object_id == ROOT_OBJECT_ID
    assert menu.object_type is ObjectType.MENU
    assert menu.static and not menu.compressed
    assert menu.revision == 1
    # A data section with the encoder version sits inside the title; it is not shown.
    assert menu.title == "Deutschlandfunk"
    assert menu.menu_items == (
        MenuItem(0x1770, "Nachrichten"),
        MenuItem(0x1F40, "Programmvorschau"),
        MenuItem(0x0001, "Kontakt"),
        MenuItem(0x0002, "Impressum"),
    )
    assert menu.transmitted_size == len(REAL_MENU_GROUP) - 4


def test_real_compressed_object():
    """The byte between header and deflate stream is the compression method 0x08
    (clause 5.2); the general link target precedes the title (clause 5.3.2.2)."""
    message = decode_data_group(REAL_COMPRESSED_GROUP)

    assert message is not None
    assert message.object_id == 0x1775
    assert message.object_type is ObjectType.TITLE_ONLY
    assert message.compressed and not message.static
    assert message.revision == 6
    assert message.title == "20:00 Nachrichten"
    assert message.links == (
        LinkTarget(
            LinkType.URL,
            "https://www.deutschlandfunk.de/programmvorschau.281.de.html",
            "Programmvorschau im Internet",
        ),
    )
    assert message.content_size == 110


def test_decoder_takes_the_object_from_the_x_pad():
    decoder = JournalineDecoder()
    group = REAL_MENU_GROUP
    length = XPadSubfield(XPAD_APP_DATA_GROUP_LENGTH, _with_crc(bytes([0, len(group)])))

    objects = decoder.process([length, XPadSubfield(XPAD_APP_JOURNALINE_START, group[:40])])
    objects += decoder.process([XPadSubfield(XPAD_APP_JOURNALINE_CONTINUATION, group[40:])])

    assert [jml_object.title for jml_object in objects] == ["Deutschlandfunk"]


# --- Header and transport -------------------------------------------------------------


@pytest.mark.parametrize("object_type", [0, 5, 7])
def test_reserved_object_types_are_ignored(object_type):
    assert parse_jml_object(_object(1, object_type, bytes([TITLE]) + b"x")) is None


def test_extended_header_is_skipped():
    jml_object = bytes([0x00, 0x07, 0x60]) + b"EXT!" + bytes([TITLE]) + b"Kurz"

    parsed = parse_jml_object(jml_object, extended_header_length=4)

    assert parsed is not None
    assert parsed.object_type is ObjectType.TITLE_ONLY
    assert parsed.title == "Kurz"


def test_extended_header_before_the_compression_method():
    compressed = _object(3, ObjectType.TITLE_ONLY, bytes([TITLE]) + b"Ticker", compress=True)
    with_extension = compressed[:3] + b"\xaa\xbb" + compressed[3:]

    parsed = parse_jml_object(with_extension, extended_header_length=2)

    assert parsed is not None and parsed.title == "Ticker"


def test_unknown_compression_method_is_rejected():
    jml_object = bytearray(_object(3, ObjectType.TITLE_ONLY, bytes([TITLE]) + b"x", compress=True))
    jml_object[3] = 0x09
    assert parse_jml_object(bytes(jml_object)) is None


def test_broken_deflate_stream_is_rejected():
    jml_object = _object(3, ObjectType.TITLE_ONLY, bytes([TITLE]) + b"x" * 50, compress=True)
    assert parse_jml_object(jml_object[:-4]) is None


def test_objects_beyond_the_size_limit_are_rejected():
    """Clause 5.1: header plus uncompressed content at most 4 092 bytes, also when the
    content only reaches that size after decompression."""
    fitting = bytes([TITLE]) + b"a" * (MAX_OBJECT_BYTES - 4)
    assert parse_jml_object(_object(1, ObjectType.TITLE_ONLY, fitting)) is not None
    assert parse_jml_object(_object(1, ObjectType.TITLE_ONLY, fitting + b"a")) is None
    assert (
        parse_jml_object(_object(1, ObjectType.TITLE_ONLY, fitting + b"a", compress=True)) is None
    )


@pytest.mark.parametrize(
    "first_byte",
    [
        0x40 | 0x80,  # extension field
        0x40 | 0x20,  # segment field
        0x40 | 0x10,  # user access field
        0x00,  # no CRC
    ],
)
def test_groups_breaking_the_transport_rules_are_discarded(first_byte):
    """Clause 8.1.1: such groups are to be discarded, however their content looks."""
    jml_object = _object(1, ObjectType.TITLE_ONLY, bytes([TITLE]) + b"x")
    group = bytes([first_byte, 0x00]) + bytes(4) + jml_object
    if first_byte & 0x40:
        group = _with_crc(group)
    assert decode_data_group(group) is None


def test_management_data_is_not_an_object():
    toc = bytes([0x54, 0, 1, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0, 0x31])
    assert decode_data_group(_group(toc, 0x40 | DATA_GROUP_MANAGEMENT)) is None


def test_group_with_broken_crc_is_discarded():
    group = bytearray(REAL_MENU_GROUP)
    group[10] ^= 0x01
    assert decode_data_group(bytes(group)) is None


# --- Content section ----------------------------------------------------------------


def test_plain_text_message_with_escape_sequences():
    """Line breaks and the end of the introduction start a new line; word breaks,
    highlighting and extended codes leave no trace in the text."""
    content = (
        bytes([TITLE])
        + b"Neue Studie (17:13)"
        + bytes([BODY])
        + "Donaudampf\x11schiff fährt.".encode()
        + b"\x14Die \x12Details\x13 folgen.\x10Ende\x1c\x07."
        + bytes([END])
    )

    message = parse_jml_object(_object(0x0100, ObjectType.PLAIN_TEXT, content))

    assert message is not None
    assert message.object_type is ObjectType.PLAIN_TEXT
    assert message.title == "Neue Studie (17:13)"
    assert message.body == "Donaudampfschiff fährt.\nDie Details folgen.\nEnde."
    start = message.body.index("Details")
    assert message.body_highlights == ((start, start + len("Details")),)


def test_highlighting_is_kept_for_the_body_text_only():
    """Laid out like the contact page of Deutschlandfunk, which highlights its
    subheadings. In the title the highlighting is dropped; one still open at the end of
    a block ends there."""
    content = (
        bytes([TITLE])
        + b"\x12Deutschlandfunk\x13"
        + bytes([BODY])
        + b"Fax: 0221 345-4802\x10\x10\x12"
        + "Hörerservice".encode()
        + b"\x13\x10Haben Sie Fragen?\x10\x10\x12  Offen bis zum Ende "
    )

    message = parse_jml_object(_object(0x0001, ObjectType.PLAIN_TEXT, content))

    assert message.title == "Deutschlandfunk"
    assert message.body == (
        "Fax: 0221 345-4802\n\nHörerservice\nHaben Sie Fragen?\n\n  Offen bis zum Ende"
    )
    assert [message.body[a:b] for a, b in message.body_highlights] == [
        "Hörerservice",
        "  Offen bis zum Ende",
    ]


def test_highlights_of_several_body_blocks_keep_their_position():
    content = (
        bytes([TITLE])
        + b"T"
        + bytes([BODY])
        + b"eins \x12A\x13"
        + bytes([BODY])
        + b"\x12B\x13 zwei"
    )

    message = parse_jml_object(_object(1, ObjectType.PLAIN_TEXT, content))

    assert message.body == "eins A\n\nB zwei"
    assert [message.body[a:b] for a, b in message.body_highlights] == ["A", "B"]


def test_list_message_is_split_into_columns():
    content = (
        bytes([TITLE])
        + b"Bundesliga"
        + bytes([LIST_ITEM])
        + b"Dortmund - N\xc3\xbcrnberg"
        + bytes([COLUMN])
        + b"4:1"
        + bytes([LIST_ITEM])
        + b"Hertha - Bayern"
        + bytes([COLUMN, COLUMN])
        + b"(abgesagt)"
        + bytes([LIST_ITEM])
        + b"Tabelle folgt"
    )

    table = parse_jml_object(_object(0x0200, ObjectType.LIST, content))

    assert table is not None
    assert table.list_items == (
        ("Dortmund - Nürnberg", "4:1"),
        ("Hertha - Bayern", "", "(abgesagt)"),
        ("Tabelle folgt",),
    )


def test_blocks_only_count_for_their_object_type():
    """Table 1: link items belong to menus, body text to text messages. Text before
    the first block and blocks with reserved codes are dropped."""
    content = (
        b"verirrt"
        + bytes([TITLE])
        + b"Titel"
        + bytes([BODY])
        + "kein Menü".encode()
        + bytes([0x06])
        + b"reserviert"
        + bytes([LINK_ITEM, 0x00, 0x05])
        + b"Ziel"
    )

    menu = parse_jml_object(_object(1, ObjectType.MENU, content))
    message = parse_jml_object(_object(1, ObjectType.PLAIN_TEXT, content))

    assert menu.title == "Titel" and menu.body == ""
    assert menu.menu_items == (MenuItem(5, "Ziel"),)
    assert message.body == "kein Menü" and message.menu_items == ()


def test_link_target_bytes_are_not_codes():
    """The two bytes of a target ID are binary; 0x1A10 must not start a data section
    or a line break."""
    content = bytes([TITLE]) + b"Menue" + bytes([LINK_ITEM, 0x1A, 0x10]) + b"Seite"

    menu = parse_jml_object(_object(0, ObjectType.MENU, content))

    assert menu.menu_items == (MenuItem(0x1A10, "Seite"),)


def test_long_data_section_continues_over_several_escapes():
    """A payload of more than 256 bytes continues in "data section continuation"
    escapes; the whole payload is skipped and the text around it stays intact."""
    padding = _data_section(bytes([0xFF]) + bytes(range(256)) + b"rest" * 10)
    link = _data_section(bytes([0x03, 0x02]) + b"https://example.org\x00Mehr")
    content = bytes([TITLE]) + b"Vor" + padding + b"her" + link

    parsed = parse_jml_object(_object(9, ObjectType.TITLE_ONLY, content))

    assert parsed.title == "Vorher"
    assert parsed.links == (LinkTarget(LinkType.URL, "https://example.org", "Mehr"),)


def test_truncated_data_section_is_dropped():
    content = bytes([TITLE]) + b"Titel" + bytes([0x1A, 0x20, 0x03, 0x02]) + b"http"

    parsed = parse_jml_object(_object(9, ObjectType.TITLE_ONLY, content))

    assert parsed.title == "Titel"
    assert parsed.links == ()


def test_general_link_targets():
    sections = b"".join(
        _data_section(bytes([0x03]) + link)
        for link in (
            bytes([0x00, 0x00, 0x05, 0x00]) + b"Zur \x12Seite\x13",  # object ID with 0x00
            bytes([0x00, 0x12, 0x34]),  # no label, no end marker
            bytes([0x03]) + b"+49 221 345-0\x00H\xc3\xb6rerservice",
            bytes([0x04]) + b"+491234+Hallo",
            bytes([0x01]) + b"dab://de0.d210.0",
            bytes([0x02]) + b"\x00ohne Adresse",  # address must not be empty
            bytes([0x09]) + b"unbekannt",  # unknown link type
        )
    )
    content = sections + bytes([TITLE]) + b"Kontakt"

    parsed = parse_jml_object(_object(1, ObjectType.TITLE_ONLY, content))

    assert parsed.links == (
        LinkTarget(LinkType.JML_OBJECT, "", "Zur Seite", target=5),
        LinkTarget(LinkType.JML_OBJECT, "", "", target=0x1234),
        LinkTarget(LinkType.TELEPHONE, "+49 221 345-0", "Hörerservice"),
        LinkTarget(LinkType.SMS, "+491234+Hallo", ""),
        LinkTarget(LinkType.DAB_URI, "dab://de0.d210.0", ""),
    )


def test_timeouts():
    """Absolute timeout in quarter hours since 2000-01-01 00:00 UTC, relative timeout
    in minutes (clause 5.3.2.2)."""
    quarter_hours = 922_000  # 2026-04-18 04:00 UTC
    content = (
        _data_section(bytes([0x01]) + quarter_hours.to_bytes(3, "big"))
        + _data_section(bytes([0x02, 0x00, 90]))
        + bytes([TITLE])
        + b"Eilmeldung"
    )

    parsed = parse_jml_object(_object(1, ObjectType.TITLE_ONLY, content))

    assert parsed.absolute_timeout == 946_684_800 + quarter_hours * 900
    assert parsed.relative_timeout_minutes == 90


def test_relative_timeout_of_zero_means_none():
    content = _data_section(bytes([0x02, 0x00, 0x00])) + bytes([TITLE]) + b"x"
    parsed = parse_jml_object(_object(1, ObjectType.TITLE_ONLY, content))
    assert parsed.relative_timeout_minutes is None


def test_invalid_utf8_does_not_break_the_object():
    content = bytes([TITLE]) + b"St\xf6vesand"
    parsed = parse_jml_object(_object(1, ObjectType.TITLE_ONLY, content))
    assert parsed.title == "St�vesand"


# --- Store and navigation -------------------------------------------------------------


def _ticker(object_id: int, text: str, *, timeouts: bytes = b"") -> JmlObject:
    content = timeouts + bytes([TITLE]) + text.encode()
    return parse_jml_object(_object(object_id, ObjectType.TITLE_ONLY, content))


def test_store_tells_new_changed_and_repeated_objects_apart():
    store = JournalineStore()

    assert store.add(_ticker(7, "eins"), now=0.0) is Change.NEW
    assert store.add(_ticker(7, "eins"), now=1.0) is Change.UNCHANGED
    assert store.add(_ticker(7, "zwei"), now=2.0) is Change.UPDATED
    assert store.get(7, now=3.0).title == "zwei"
    assert store.received_at(7) == 2.0


def test_store_lists_objects_by_id():
    store = JournalineStore()
    for object_id in (0x1775, 0x0000, 0x0002):
        store.add(_ticker(object_id, "x"), now=0.0)

    assert [jml_object.object_id for jml_object in store.objects(now=0.0)] == [0, 2, 0x1775]


def test_relative_timeout_restarts_with_every_reception():
    store = JournalineStore()
    ticker = _ticker(7, "x", timeouts=_data_section(bytes([0x02, 0x00, 10])))

    store.add(ticker, now=0.0)
    store.add(ticker, now=300.0)  # the same object again

    assert store.get(7, now=899.0) is not None
    assert store.get(7, now=900.0) is None
    assert store.expire(now=900.0) == [7]
    assert len(store) == 0


def test_absolute_timeout_takes_precedence():
    store = JournalineStore()
    deadline = 946_684_800 + 900  # 2000-01-01 00:15 UTC
    timeouts = _data_section(bytes([0x01, 0x00, 0x00, 0x01])) + _data_section(
        bytes([0x02, 0x00, 0x01])
    )
    store.add(_ticker(7, "x", timeouts=timeouts), now=deadline - 600.0)

    assert store.expires_at(7) == deadline
    assert store.get(7, now=deadline - 1.0) is not None
    assert store.objects(now=deadline) == []


def test_history_follows_links_and_goes_back():
    history = NavigationHistory()
    history.follow(0x1770)
    history.follow(0x1771)
    assert history.path == (0x0000, 0x1770, 0x1771)

    history.back()
    assert history.current == 0x1770


def test_link_to_an_object_on_the_path_shortens_it():
    """Clause 4.3: e.g. a link back to the main menu from deep inside."""
    history = NavigationHistory()
    for object_id in (0x1770, 0x1771, 0x1772):
        history.follow(object_id)

    history.follow(0x1770)

    assert history.path == (0x0000, 0x1770)


def test_back_from_the_point_of_entry_leads_to_the_root_object():
    history = NavigationHistory(entry=0x0042)
    history.back()
    assert history.path == (ROOT_OBJECT_ID,)

    history.follow(5)
    history.home()
    assert history.path == (0x0042,)


# --- Through the signal processing ------------------------------------------------------


def _access_unit(contents: list[tuple[int, int, bytes]]) -> bytes:
    """A DAB+ access unit whose DSE carries an X-PAD with contents indicators: one
    `(length index, application type, data)` per subfield."""
    indicators = bytes((index << 5) | application for index, application, _ in contents)
    x_pad = indicators + b"".join(data for _, _, data in contents)
    pad_field = x_pad[::-1] + bytes([0x20, 0x02])  # variable X-PAD, indicators present
    return bytes([0x80, len(pad_field)]) + pad_field


# FIGs of Deutschlandfunk's service as received on 5C (from the FIG header byte on):
# FIG 0/2 with its four programme services, the Dlf (D210) on subchannel 10, and FIG 0/13
# with slideshow (X-PAD type 12) and Journaline (X-PAD type 4, user data 00 00 00 00).
REAL_FIG0_2 = bytes.fromhex("1502d210013f2ad220013f2ed230013f32d240013f36")
REAL_FIG0_13 = bytes.fromhex("170dd2100200420c3c8946040500000000d2200100420c3c")
DLF = 0xD210


def _worker_for_dlf(fig0_13: bytes = REAL_FIG0_13) -> tuple[DspWorker, list, list]:
    """A DspWorker that has received the FIGs of the Dlf and has it selected; returns
    it with the lists of reported objects and Journaline parameters."""
    worker = DspWorker(SAMPLE_RATE, MODE_I)
    objects: list = []
    parameters: list = []
    worker.journaline_object_ready.connect(objects.append)
    worker.journaline_parameters_ready.connect(parameters.append)
    for fig in (REAL_FIG0_2, fig0_13):
        parse_fib(fib(fig), worker._ensemble)
    worker.select_service(DLF)
    return worker, objects, parameters


def _menu_in_three_access_units(start_type: int) -> list[bytes]:
    """Real bytes spread over three access units as a station would send them: the
    length indicator first, then start and continuations."""
    group = REAL_MENU_GROUP
    length = _with_crc(bytes([0, len(group)]))
    return [
        _access_unit([(0, 1, length), (7, start_type, group[:48])]),
        _access_unit([(7, start_type + 1, group[48:96])]),
        _access_unit([(1, start_type + 1, group[96:])]),
    ]


def test_parse_user_application_data_of_deutschlandfunk():
    assert parse_user_application_data(bytes(4)) == JournalineParameters(0, 0, ROOT_OBJECT_ID)
    # Without the optional point of entry, the root object (clause 8.1.2).
    assert parse_user_application_data(bytes([0, 2])) == JournalineParameters(0, 2)
    assert parse_user_application_data(bytes([1, 0, 0x12, 0x34, 0xFF])).entry_object == 0x1234
    assert parse_user_application_data(bytes([0])) is None


def test_dsp_worker_reports_the_objects_from_the_pad():
    worker, received, parameters = _worker_for_dlf()

    audio_units = _menu_in_three_access_units(XPAD_APP_JOURNALINE_START)
    worker._decode_programme_data(SimpleNamespace(sync_ok=True, audio_units=audio_units))

    assert parameters == [JournalineParameters(0, 0, ROOT_OBJECT_ID)]
    assert [jml_object.object_id for jml_object in received] == [ROOT_OBJECT_ID]
    assert received[0].title == "Deutschlandfunk"


def test_dsp_worker_ignores_journaline_without_fig0_13():
    """Types 4 and 5 are free for any application; only FIG 0/13 says it is Journaline."""
    worker = DspWorker(SAMPLE_RATE, MODE_I)
    received = []
    worker.journaline_object_ready.connect(received.append)
    parse_fib(fib(REAL_FIG0_2), worker._ensemble)
    worker.select_service(DLF)

    audio_units = _menu_in_three_access_units(XPAD_APP_JOURNALINE_START)
    worker._decode_programme_data(SimpleNamespace(sync_ok=True, audio_units=audio_units))

    assert received == []


def test_dsp_worker_takes_x_pad_type_and_point_of_entry_from_fig0_13():
    """The same service, but Journaline announced on X-PAD type 6 with point of entry
    0x0042: the groups on 6/7 are decoded, the tab learns the entry."""
    fig0_13 = bytes.fromhex("0c0dd210018946060500000042")
    worker, received, parameters = _worker_for_dlf(fig0_13)

    audio_units = _menu_in_three_access_units(6)
    worker._decode_programme_data(SimpleNamespace(sync_ok=True, audio_units=audio_units))

    assert parameters == [JournalineParameters(0, 0, 0x0042)]
    assert [jml_object.object_id for jml_object in received] == [ROOT_OBJECT_ID]
