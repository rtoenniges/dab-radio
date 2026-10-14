"""MSC data groups (ETSI EN 300 401, clause 5.3.3) and their assembly from the X-PAD.

Applications that transport whole objects alongside the audio, the MOT slideshow and
Journaline, pack them into MSC data groups: a header of at least two bytes, the data
group data field and, as a rule, a CRC. In the X-PAD a data group is spread over the
subfields of two application types, one for its start and one for its continuation
(see `dab.pad`). How long it is, is announced beforehand by a subfield of its own,
application type 1, the data group length indicator (clause 7.4.5).

The length indicator names no application; it applies to the next data group that
starts. Each application assembles its groups with an `XPadDataGroupAssembler` of its
own, so every assembler sees every length indicator. It keeps the most recent one until
a group of its own application starts and binds it to that group. An indicator that
arrives while a group is still being collected (it announces the next group, possibly
one of another application) therefore does not cut the running group short.
"""

from __future__ import annotations

from dataclasses import dataclass

from dab.crc import check_crc
from dab.pad import XPAD_APP_DATA_GROUP_LENGTH, XPadSubfield

_LENGTH_INDICATOR_BYTES = 4  # 2 bits Rfa, 14 bits length, 16 bits CRC


@dataclass(frozen=True)
class DataGroup:
    """An MSC data group whose CRC, if it carries one, was correct.

    The four flags are those of the data group header; applications restrict which
    combinations they accept (Journaline, for instance, forbids segmentation).
    """

    group_type: int
    continuity_index: int
    repetition_index: int
    has_extension: bool
    has_crc: bool
    has_segment_field: bool
    has_user_access: bool
    segment_number: int  # 0 without a segment field
    last_segment: bool  # True without a segment field
    transport_id: int | None
    data: bytes  # the data group data field, without header, session header and CRC


def parse_data_group(group: bytes) -> DataGroup | None:
    """Splits an MSC data group (ETSI EN 300 401, clause 5.3.3.1).

    Returns:
        The data group, or None if the data are too short or the CRC is wrong.
    """
    if len(group) < 2:
        return None

    has_extension = bool(group[0] & 0x80)
    has_crc = bool(group[0] & 0x40)
    has_segment_field = bool(group[0] & 0x20)
    has_user_access = bool(group[0] & 0x10)
    group_type = group[0] & 0x0F

    if has_crc and not check_crc(group):
        return None
    body = group[:-2] if has_crc else group

    offset = 2
    if has_extension:
        offset += 2

    segment_number = 0
    last_segment = True
    if has_segment_field:
        if len(body) < offset + 2:
            return None
        last_segment = bool(body[offset] & 0x80)
        segment_number = ((body[offset] & 0x7F) << 8) | body[offset + 1]
        offset += 2

    transport_id: int | None = None
    if has_user_access:
        if len(body) < offset + 1:
            return None
        has_transport_id = bool(body[offset] & 0x10)
        length_indicator = body[offset] & 0x0F
        offset += 1
        if has_transport_id:
            if len(body) < offset + 2:
                return None
            transport_id = (body[offset] << 8) | body[offset + 1]
        offset += length_indicator

    if offset > len(body):
        return None
    return DataGroup(
        group_type=group_type,
        continuity_index=group[1] >> 4,
        repetition_index=group[1] & 0x0F,
        has_extension=has_extension,
        has_crc=has_crc,
        has_segment_field=has_segment_field,
        has_user_access=has_user_access,
        segment_number=segment_number,
        last_segment=last_segment,
        transport_id=transport_id,
        data=body[offset:],
    )


def read_length_indicator(data: bytes) -> int | None:
    """The subfield of application type 1: 2 reserved bits, 14 bits with the length of
    the following data group in bytes, then a CRC.

    Returns:
        The announced length, or None if the CRC is wrong.
    """
    if len(data) < _LENGTH_INDICATOR_BYTES or not check_crc(data[:_LENGTH_INDICATOR_BYTES]):
        return None
    return ((data[0] & 0x3F) << 8) | data[1]


class XPadDataGroupAssembler:
    """Assembles the MSC data groups of one X-PAD application from its subfields.

    Since an image or a text arrives spread over many access units, joining in the
    middle of a group is the normal case: its continuations are skipped until the next
    group of the application starts.
    """

    def __init__(self, start_type: int, continuation_type: int) -> None:
        self._start_type = start_type
        self._continuation_type = continuation_type
        self._announced_length: int | None = None  # for the next group that starts
        self._expected_length: int | None = None  # None: no group being collected
        self._group = bytearray()

    def reset(self) -> None:
        self._announced_length = None
        self._expected_length = None
        self._group.clear()

    def process(self, subfields: list[XPadSubfield]) -> list[bytes]:
        """Processes the subfields of one access unit or audio frame.

        Returns:
            The data groups completed in this step (usually none), exactly as long as
            announced and not yet checked; see `parse_data_group`.
        """
        completed: list[bytes] = []
        for subfield in subfields:
            application_type = subfield.application_type
            if application_type == XPAD_APP_DATA_GROUP_LENGTH:
                self._announced_length = read_length_indicator(subfield.data)
                continue
            if application_type == self._start_type:
                # Without a valid length indicator the end of the group cannot be
                # found; it is skipped.
                self._expected_length = self._announced_length
                self._announced_length = None
                self._group = bytearray(subfield.data)
            elif application_type == self._continuation_type:
                if self._expected_length is None:
                    continue  # joined in the middle of a group
                self._group.extend(subfield.data)
            else:
                continue

            if self._expected_length is not None and len(self._group) >= self._expected_length:
                # The last subfield may be padded beyond the end of the group.
                completed.append(bytes(self._group[: self._expected_length]))
                self._expected_length = None
                self._group.clear()
        return completed
