"""Programme Associated Data (PAD) from the audio stream of DAB+ and DAB.

Besides the audio, a service carries accompanying data in the same subchannel: the
dynamic label with station and title information, the images of the MOT slideshow and
the text pages of Journaline. All of them live in the PAD field. DAB+ carries it in a
Data Stream Element (DSE) of the AAC access unit (ETSI TS 102 563, clause 5.3), classic
DAB at the end of the MPEG Layer II frame (ETSI EN 300 401, clause 7).

Structure, from the outside in:

1. DAB+: the access unit starts with a DSE whose payload is the PAD field (ETSI EN 300
   401, clause 7.4). DAB: the PAD field fills the end of the frame, followed only by
   the scale factor CRC and the F-PAD.
2. The PAD field ends with two bytes of F-PAD ("fixed PAD") announcing whether, and in
   which form, an X-PAD precedes it.
3. Before it lies the X-PAD, **in reversed byte order**. Read from the back, the
   "contents indicators" come first, giving length and kind of the data subfields that
   follow, and then the subfields themselves.
4. Across several access units the subfields assemble into X-PAD data groups, from
   which the dynamic label, an image or a Journaline page is built.

The byte order and the split of the contents indicators were verified on air: the
lengths given there add up to the length of the X-PAD field to the byte.
"""

from __future__ import annotations

from dataclasses import dataclass

# ETSI EN 300 401, clause 7.4.3: X-PAD application types. Every multi-part application
# has one type for the start of a data group and one for its continuation.
XPAD_APP_END_MARKER = 0
XPAD_APP_DATA_GROUP_LENGTH = 1
XPAD_APP_DLS_START = 2
XPAD_APP_DLS_CONTINUATION = 3
XPAD_APP_MOT_START = 12
XPAD_APP_MOT_CONTINUATION = 13
# Journaline (ETSI TS 102 979) has no fixed type of its own; FIG 0/13 names the one a
# service uses (see `dab.data_services.xpad_applications`). Deutschlandfunk announces 4
# there and continues its groups with 5.
XPAD_APP_JOURNALINE_START = 4
XPAD_APP_JOURNALINE_CONTINUATION = 5

# Start type -> continuation type of the same application for the types EN 300 401 fixes
# (table 11; 14/15 are the MOT CA messages). An X-PAD without contents indicators
# continues the data group of the previous subfield, so it must not repeat a start
# type: the decoders would take it as a new group and discard what they collected so
# far. The other types only get their pair once FIG 0/13 announces an application on
# them (`PadReader.set_signalled_applications`).
_CONTINUATION_TYPE = {
    XPAD_APP_DLS_START: XPAD_APP_DLS_CONTINUATION,
    XPAD_APP_MOT_START: XPAD_APP_MOT_CONTINUATION,
    14: 15,
}

# ETSI EN 300 401, table 27: the eight possible lengths of an X-PAD data subfield,
# addressed by the 3-bit length index of the contents indicator.
_LENGTH_BY_INDEX = (4, 6, 8, 12, 16, 24, 32, 48)

_AAC_ELEMENT_DSE = 4  # id_syn_ele of a Data Stream Element
_FPAD_LENGTH = 2
_SHORT_XPAD_LENGTH = 4
_MAX_CONTENTS_INDICATORS = 4


@dataclass(frozen=True)
class XPadSubfield:
    """One X-PAD data subfield together with the application it belongs to."""

    application_type: int
    data: bytes


def extract_pad(access_unit: bytes) -> tuple[bytes, bytes] | None:
    """Extracts (F-PAD, X-PAD) from a DAB+ access unit.

    The PAD sits in the Data Stream Element, which, if present, is the first element of
    the AAC access unit (ETSI TS 102 563, clause 5.3). Not every access unit carries
    one; then there is nothing to fetch for these 40 ms.

    Returns:
        `(f_pad, x_pad)` with the two F-PAD bytes and the X-PAD already turned into
        reading order (contents indicators first), or None if the access unit carries
        no PAD.
    """
    if len(access_unit) < 2 or (access_unit[0] >> 5) != _AAC_ELEMENT_DSE:
        return None

    # DSE: 3 bits id_syn_ele, 4 bits element_instance_tag, 1 bit data_byte_align_flag,
    # 8 bits count (plus 8 bits of extension when count == 255).
    count = access_unit[1]
    start = 2
    if count == 255:
        if len(access_unit) < 3:
            return None
        count += access_unit[2]
        start = 3

    payload = access_unit[start : start + count]
    if len(payload) < _FPAD_LENGTH:
        return None

    # The F-PAD sits at the end, byte L before byte L-1; everything before it is the
    # X-PAD in reversed order.
    f_pad = payload[-_FPAD_LENGTH:]
    x_pad = payload[:-_FPAD_LENGTH][::-1]
    return f_pad, x_pad


def extract_mp2_pad(frame: bytes, scf_crc_length: int) -> tuple[bytes, bytes] | None:
    """Extracts (F-PAD, X-PAD area) from a DAB audio frame (MPEG Audio Layer II).

    The F-PAD occupies the last two bytes of the frame; before it lies the scale factor
    CRC (`scf_crc_length` bytes, see `dsp.mp2_frame.Mp2Format.scf_crc_length`) and
    before that the X-PAD, again in reversed byte order.

    Unlike in DAB+, nothing states where the X-PAD begins: it grows from the back into
    the frame as far as needed. The returned area therefore reaches back to the start
    of the frame, and only the contents indicators (or, without them, the length of the
    previous X-PAD) tell how much of it belongs to the X-PAD (see `PadReader`).

    Returns:
        `(f_pad, x_pad_area)` with the area in reading order, or None if the frame is
        too short.
    """
    x_pad_end = len(frame) - _FPAD_LENGTH - scf_crc_length
    if x_pad_end < 0:
        return None
    return frame[-_FPAD_LENGTH:], frame[:x_pad_end][::-1]


def parse_x_pad(
    f_pad: bytes,
    x_pad: bytes,
    last_application_type: int | None = None,
    continuation_types: dict[int, int] | None = None,
) -> tuple[list[XPadSubfield], int | None]:
    """Splits an X-PAD field into its data subfields.

    Args:
        f_pad: the two F-PAD bytes from `extract_pad`.
        x_pad: the X-PAD in reading order from `extract_pad`.
        last_application_type: the application of the most recent subfield. It is
            needed when the F-PAD announces no contents indicators: the whole X-PAD
            then continues the previous subfield without naming itself, and is
            returned with the continuation type of that application.
        continuation_types: start type -> continuation type of further applications,
            as FIG 0/13 announces them, in addition to the ones EN 300 401 fixes.

    Returns:
        `(subfields, last_application_type)`: the subfields of this X-PAD and the
        application to remember for the next call.
    """
    subfields, last_application_type, _ = _parse_x_pad(
        f_pad,
        x_pad,
        last_application_type,
        continuation_types={**_CONTINUATION_TYPE, **(continuation_types or {})},
    )
    return subfields, last_application_type


def _parse_x_pad(
    f_pad: bytes,
    x_pad: bytes,
    last_application_type: int | None,
    continuation_length: int | None = None,
    continuation_types: dict[int, int] = _CONTINUATION_TYPE,
) -> tuple[list[XPadSubfield], int | None, int]:
    """`parse_x_pad` that also reports the X-PAD length used.

    Args:
        continuation_length: None if `x_pad` is exactly the X-PAD field (DAB+).
            Otherwise (DAB) the length of the previous X-PAD field: a field without
            contents indicators is as long as its predecessor (ETSI EN 300 401, clause
            7.4.2.1; cross-checked against DABlin, pad_decoder.cpp).
        continuation_types: start type -> continuation type of all known applications.

    Returns:
        `(subfields, last_application_type, length)` with the length of the X-PAD field
        in bytes, contents indicators included.
    """
    if len(f_pad) < _FPAD_LENGTH:
        return [], last_application_type, 0

    x_pad_indicator = (f_pad[0] & 0x30) >> 4
    contents_indicators_present = bool(f_pad[1] & 0x02)

    if x_pad_indicator == 1:
        available = x_pad[:_SHORT_XPAD_LENGTH]
    elif x_pad_indicator == 2:
        available = x_pad
    else:
        return [], last_application_type, 0  # no X-PAD (0) or reserved (3)

    if not contents_indicators_present:
        # Without contents indicators the whole field continues the previous subfield,
        # as the continuation of its application (see `_CONTINUATION_TYPE`).
        if x_pad_indicator == 2 and continuation_length is not None:
            available = available[:continuation_length]
        if last_application_type is None or not available:
            return [], last_application_type, 0
        application_type = continuation_types.get(last_application_type, last_application_type)
        subfield = XPadSubfield(application_type, bytes(available))
        return [subfield], last_application_type, len(available)

    return _split_with_contents_indicators(available, last_application_type)


def _split_with_contents_indicators(
    available: bytes, last_application_type: int | None
) -> tuple[list[XPadSubfield], int | None, int]:
    """Reads the contents indicators at the start and then cuts out the subfields in
    the announced lengths.

    A contents indicator is one byte: the upper 3 bits index the length table, the
    lower 5 bits are the application type (ETSI EN 300 401, clause 7.4.2.2). The list
    ends with application type 0 or as soon as the announced lengths fill the field.

    Returns:
        Like `_parse_x_pad`; the length is that of the contents indicators plus the
        announced subfields.
    """
    indicators: list[tuple[int, int]] = []
    offset = 0
    while offset < len(available) and len(indicators) < _MAX_CONTENTS_INDICATORS:
        indicator = available[offset]
        application_type = indicator & 0x1F
        offset += 1
        if application_type == XPAD_APP_END_MARKER:
            break
        indicators.append((application_type, _LENGTH_BY_INDEX[indicator >> 5]))
        if offset + sum(length for _, length in indicators) >= len(available):
            break

    subfields: list[XPadSubfield] = []
    for application_type, length in indicators:
        data = available[offset : offset + length]
        if not data:
            break
        subfields.append(XPadSubfield(application_type, bytes(data)))
        offset += length
        last_application_type = application_type

    return subfields, last_application_type, min(offset, len(available))


class PadReader:
    """Continuously reads the X-PAD subfields out of the access units (DAB+) or audio
    frames (DAB).

    Remembers the most recent application, and for DAB the length of the most recent
    X-PAD field, so that fields without contents indicators are attributed correctly
    (see `_parse_x_pad`).
    """

    def __init__(self) -> None:
        self._last_application_type: int | None = None
        self._last_length = 0
        self._continuation_types = dict(_CONTINUATION_TYPE)
        # Length of the X-PAD field in the most recently read frame (bytes, contents
        # indicators included), for the display of the frame layout.
        self.x_pad_length = 0

    def set_signalled_applications(self, continuation_types: dict[int, int]) -> None:
        """Start type -> continuation type of the applications FIG 0/13 announces for
        the X-PAD of the service, in addition to the fixed ones. Survives `reset`."""
        self._continuation_types = {**_CONTINUATION_TYPE, **continuation_types}

    def reset(self) -> None:
        self._last_application_type = None
        self._last_length = 0
        self.x_pad_length = 0

    def read(self, access_unit: bytes) -> list[XPadSubfield]:
        """Reads the PAD of a DAB+ access unit."""
        pad = extract_pad(access_unit)
        if pad is None:
            self.x_pad_length = 0
            return []
        return self._read(*pad, continuation_length=None)

    def read_mp2_frame(self, frame: bytes, scf_crc_length: int) -> list[XPadSubfield]:
        """Reads the PAD of a DAB audio frame (see `extract_mp2_pad`)."""
        pad = extract_mp2_pad(frame, scf_crc_length)
        if pad is None:
            self.x_pad_length = 0
            return []
        return self._read(*pad, continuation_length=self._last_length)

    def _read(
        self, f_pad: bytes, x_pad: bytes, continuation_length: int | None
    ) -> list[XPadSubfield]:
        subfields, self._last_application_type, self.x_pad_length = _parse_x_pad(
            f_pad, x_pad, self._last_application_type, continuation_length, self._continuation_types
        )
        if self.x_pad_length:
            self._last_length = self.x_pad_length
        return subfields
