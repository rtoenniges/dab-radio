"""TPEG in a transparent data channel: the traffic information of ARD TPEG on channel 7A.

ARD TPEG is a packet mode data service (DSCTy 5, TDC, ETSI TS 101 759) whose FIG 0/13
announces user application 0x004, TPEG. The data fields of its data groups form a byte
stream of TPEG transport frames (ISO/TS 21219-5, TPEG2 service framework), nested in
three layers:

1. Transport frame: sync word FF0F, field length (16 bits, the bytes after the 7-byte
   header), header CRC, frame type. The header CRC covers sync word, field length,
   frame type and the first 11 bytes of the frame, 16 bytes in all; it is how a frame
   start is told apart from FF0F inside the data.
2. Frame type 0 is the stream directory (the service IDs in the stream, with a CRC of
   its own); frame type 1 a service frame: service ID SID-A.SID-B.SID-C, encryption
   indicator, then one component frame per service component.
3. Component frame: service component ID (SCID), field length, a header CRC over SCID,
   length and the first 13 bytes of data, then the data, which end in a CRC over all
   of them. SCID 0 is always the service and network information (SNI): a count and
   the SNI tables. Its fast tuning table says which application each other SCID
   carries (application ID, AID); TEC and TFP frames then hold a group priority, a
   message count and the messages as TPEG2 component trees (`dab.tpeg_messages`).

All CRCs are the CRC-16 of DAB (CCITT, register preset to ones, result inverted).

ARD TPEG sends one transport frame per data group, alternately a stream directory and
one or two service frames of about 3 900 bytes: SNI, TEC (SCID 1) or TFP (SCID 2). Its
continuity index counts modulo 4 (0, 1, 2, 3, 0, ...) where EN 300 401 has it count
modulo 16; the decoder learns the modulus from the largest index it sees. In 240 s the
SNI announced 655 TEC and 730 TFP messages, and 657 distinct TEC messages arrived.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from dab.crc import crc16_ccitt
from dab.data_group import DataGroup
from dab.tpeg_binary import TpegDecodeError, TpegReader, parse_components
from dab.tpeg_messages import TecMessage, TfpMessage, decode_tec_message, decode_tfp_message

SYNC_WORD = b"\xff\x0f"
FRAME_HEADER_BYTES = 7  # sync word, field length, header CRC, frame type
# Bytes of the frame (after the header) that the transport header CRC covers.
_FRAME_CRC_SPAN = 11
# Bytes of the data that the header CRC of a component frame covers.
_COMPONENT_CRC_SPAN = 13
_CRC_BYTES = 2

FRAME_TYPE_STREAM_DIRECTORY = 0
FRAME_TYPE_SERVICE = 1

# Application IDs (AID, allocated by TISA).
AID_SNI = 0
AID_TEC = 5
AID_TFP = 7
AID_EAW = 15
_MESSAGE_DECODERS = {AID_TEC: decode_tec_message, AID_TFP: decode_tfp_message}

# SNI tables (ISO/TS 21219-9).
SNI_SERVICE_NAME = 0
SNI_FAST_TUNING = 1
SNI_CONTENT = 3
SNI_LOGO = 7
SNI_VERSIONS = 14
SNI_MESSAGE_COUNTS = 33
# Selector of a fast tuning table entry: which optional fields follow.
_FAST_TUNING_ORIGINATOR = 0x01
_FAST_TUNING_OPERATING_TIME = 0x04
_FAST_TUNING_ENCRYPTION = 0x08
_FAST_TUNING_SAFETY = 0x10
_FAST_TUNING_KNOWN = (
    _FAST_TUNING_ORIGINATOR
    | _FAST_TUNING_OPERATING_TIME
    | _FAST_TUNING_ENCRYPTION
    | _FAST_TUNING_SAFETY
)

ServiceId = tuple[int, int, int]
Message = TecMessage | TfpMessage


def service_id_text(sid: ServiceId) -> str:
    """The usual notation of a TPEG service ID: SID-A.SID-B.SID-C in decimal."""
    return ".".join(str(part) for part in sid)


def _crc_ok(data: bytes, transmitted: bytes) -> bool:
    return crc16_ccitt(data) ^ 0xFFFF == int.from_bytes(transmitted, "big")


@dataclass(frozen=True)
class ServiceComponentEntry:
    """A line of the fast tuning table: which application a service component carries."""

    scid: int
    application: int  # AID
    content_id: int
    originator: ServiceId | None = None
    encryption: int | None = None
    safety: bool = False
    operating_time: tuple[int, int] | None = None  # POSIX seconds


@dataclass(frozen=True)
class ServiceInformation:
    """What the SNI tables of the service said so far (the latest of each table)."""

    name: str | None = None
    description: str | None = None
    components: tuple[ServiceComponentEntry, ...] = ()
    versions: dict[int, tuple[int, int]] = field(default_factory=dict)  # SCID: spec version
    message_counts: dict[int, int] = field(default_factory=dict)  # SCID: messages on air
    content: dict[int, str] = field(default_factory=dict)  # SCID: content description
    logo: bytes | None = None  # image file (PNG as sent by ARD)
    tables: tuple[int, ...] = ()  # identifiers of all tables received

    def application(self, scid: int) -> int | None:
        if scid == 0:
            return AID_SNI
        entry = next((e for e in self.components if e.scid == scid), None)
        return None if entry is None else entry.application


def _decode_sni_table(info: ServiceInformation, table: int, data: bytes) -> ServiceInformation:
    """Takes one SNI table into `info`; tables not shown here are only listed."""
    reader = TpegReader(data)
    if table == SNI_SERVICE_NAME:
        return replace(info, name=reader.short_string(), description=reader.short_string())
    if table == SNI_FAST_TUNING:
        reader.u8()  # table version
        reader.u8()  # character table
        entries = []
        while not reader.at_end():
            scid, selector = reader.u8(), reader.u8()
            if selector & ~_FAST_TUNING_KNOWN:
                raise TpegDecodeError(f"fast tuning table: unknown selector {selector:#04x}")
            originator = tuple(reader.bytes(3)) if selector & _FAST_TUNING_ORIGINATOR else None
            content_id, application = reader.u8(), reader.u16()
            operating_time = None
            if selector & _FAST_TUNING_OPERATING_TIME:
                operating_time = (reader.u32(), reader.u32())
            encryption = reader.u8() if selector & _FAST_TUNING_ENCRYPTION else None
            entries.append(
                ServiceComponentEntry(
                    scid,
                    application,
                    content_id,
                    originator,
                    encryption,
                    bool(selector & _FAST_TUNING_SAFETY),
                    operating_time,
                )
            )
        return replace(info, components=tuple(entries))
    if table == SNI_CONTENT:
        reader.u8()
        content = {}
        while not reader.at_end():
            scid = reader.u8()
            content[scid] = reader.short_string()
        return replace(info, content=content)
    if table == SNI_LOGO:
        reader.u8()  # graphic type
        return replace(info, logo=reader.rest())
    if table == SNI_VERSIONS:
        reader.u8()
        versions = {}
        while not reader.at_end():
            scid = reader.u8()
            versions[scid] = (reader.u8(), reader.u8())
        return replace(info, versions=versions)
    if table == SNI_MESSAGE_COUNTS:
        reader.u8()
        counts = {}
        while not reader.at_end():
            scid = reader.u8()
            counts[scid] = reader.u32()
        return replace(info, message_counts=counts)
    return info


@dataclass(frozen=True)
class ComponentFrame:
    """A component frame of a service frame, as far as it could be checked."""

    scid: int
    size: int  # bytes of data (field length)
    header_crc_ok: bool
    application: int | None  # AID as the SNI announces it; None: not (yet) known
    data_crc_ok: bool | None = None  # None: not checked, the application is unknown
    group_priority: int | None = None
    message_count: int | None = None  # as announced in the frame (SNI: tables)
    decoded: int = 0  # messages (SNI: tables) decoded
    errors: int = 0  # messages (SNI: tables) that did not decode


@dataclass(frozen=True)
class TransportFrame:
    """A transport frame whose header CRC was correct."""

    frame_type: int
    size: int  # field length: bytes after the header
    service_id: ServiceId | None = None  # service frame
    encryption: int | None = None
    components: tuple[ComponentFrame, ...] = ()
    directory: tuple[ServiceId, ...] = ()  # stream directory
    directory_crc_ok: bool | None = None


@dataclass(frozen=True)
class ComponentStatistics:
    """What arrived for one service component (SCID)."""

    frames: int = 0
    bytes: int = 0
    crc_errors: int = 0  # component frames with a wrong header or data CRC
    messages: int = 0  # messages (SNI: tables) decoded
    errors: int = 0  # messages (SNI: tables) that did not decode


@dataclass(frozen=True)
class TpegStatistics:
    data_groups: int = 0
    repeated_data_groups: int = 0  # repetitions, skipped
    lost_data_groups: int = 0  # jumps of the continuity index
    stream_bytes: int = 0
    discarded_bytes: int = 0  # bytes outside transport frames
    stream_directories: int = 0
    service_frames: int = 0
    other_frames: int = 0  # unknown frame types
    encrypted_frames: int = 0
    components: dict[int, ComponentStatistics] = field(default_factory=dict)


@dataclass(frozen=True)
class TpegFrame:
    """A transport frame and what it carried, with the state of the service after it."""

    frame: TransportFrame
    messages: tuple[Message, ...]
    service: ServiceInformation
    statistics: TpegStatistics


class TpegDecoder:
    """Joins the data groups of a TPEG component into the TPEG stream and decodes its
    frames: SNI into `ServiceInformation`, TEC and TFP into messages."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._continuity: int | None = None
        self._modulus = 4
        self._service = ServiceInformation()
        self._statistics = TpegStatistics()
        self._components: dict[int, ComponentStatistics] = {}

    @property
    def service(self) -> ServiceInformation:
        return self._service

    def interrupt(self) -> None:
        """The data groups have a gap the continuity index cannot reveal (the subchannel
        chain started afresh): a frame under way is lost."""
        self._buffer.clear()
        self._continuity = None

    def push(self, group: DataGroup) -> list[TpegFrame]:
        """Takes the next data group; returns the transport frames completed with it."""
        index = group.continuity_index
        if index == self._continuity:
            self._count(repeated_data_groups=1)
            return []
        if index > 3:
            self._modulus = 16
        if self._continuity is not None and index != (self._continuity + 1) % self._modulus:
            self._count(lost_data_groups=(index - self._continuity - 1) % self._modulus)
            self._buffer.clear()
        self._continuity = index
        self._count(data_groups=1, stream_bytes=len(group.data))
        self._buffer += group.data
        return [self._decode_frame(raw) for raw in self._take_frames()]

    def statistics(self) -> TpegStatistics:
        return replace(self._statistics, components=dict(self._components))

    def _count(self, **increments: int) -> None:
        stats = self._statistics
        self._statistics = replace(
            stats, **{name: getattr(stats, name) + value for name, value in increments.items()}
        )

    def _take_frames(self) -> list[bytes]:
        """Cuts the complete transport frames off the front of the buffer."""
        buffer = self._buffer
        frames = []
        while True:
            start = buffer.find(SYNC_WORD)
            if start < 0:
                # A last 0xFF may be the first half of the next sync word.
                keep = 1 if buffer[-1:] == b"\xff" else 0
                self._discard(len(buffer) - keep)
                return frames
            self._discard(start)
            if len(buffer) < FRAME_HEADER_BYTES:
                return frames
            length = int.from_bytes(buffer[2:4], "big")
            covered = FRAME_HEADER_BYTES + min(_FRAME_CRC_SPAN, length)
            if len(buffer) < covered:
                return frames
            if not _crc_ok(bytes(buffer[0:4] + buffer[6:covered]), buffer[4:6]):
                self._discard(1)  # FF0F inside the data, not a frame start
                continue
            if len(buffer) < FRAME_HEADER_BYTES + length:
                return frames
            frames.append(bytes(buffer[: FRAME_HEADER_BYTES + length]))
            del buffer[: FRAME_HEADER_BYTES + length]

    def _discard(self, count: int) -> None:
        if count > 0:
            del self._buffer[:count]
            self._count(discarded_bytes=count)

    def _decode_frame(self, raw: bytes) -> TpegFrame:
        frame_type = raw[6]
        body = raw[FRAME_HEADER_BYTES:]
        messages: list[Message] = []
        if frame_type == FRAME_TYPE_STREAM_DIRECTORY:
            self._count(stream_directories=1)
            frame = self._stream_directory(body)
        elif frame_type == FRAME_TYPE_SERVICE and len(body) >= 4:
            self._count(service_frames=1)
            frame = self._service_frame(body, messages)
        else:
            self._count(other_frames=1)
            frame = TransportFrame(frame_type, len(body))
        return TpegFrame(frame, tuple(messages), self._service, self.statistics())

    @staticmethod
    def _stream_directory(body: bytes) -> TransportFrame:
        """Frame type 0: number of services, their service IDs, CRC."""
        size = 1 + 3 * body[0] if body else 0
        if not body or len(body) != size + _CRC_BYTES:
            return TransportFrame(FRAME_TYPE_STREAM_DIRECTORY, len(body), directory_crc_ok=False)
        sids = tuple(tuple(body[i : i + 3]) for i in range(1, size, 3))
        ok = _crc_ok(body[:size], body[size:])
        return TransportFrame(
            FRAME_TYPE_STREAM_DIRECTORY, len(body), directory=sids, directory_crc_ok=ok
        )

    def _service_frame(self, body: bytes, messages: list[Message]) -> TransportFrame:
        sid: ServiceId = tuple(body[0:3])
        encryption = body[3]
        # Component frames: SCID, field length, header CRC, data.
        raw_frames: list[tuple[int, bytes, bool]] = []
        reader = TpegReader(body, 4)
        try:
            while not reader.at_end():
                header = reader.bytes(3)
                crc = reader.bytes(_CRC_BYTES)
                data = reader.bytes(int.from_bytes(header[1:3], "big"))
                ok = _crc_ok(header + data[:_COMPONENT_CRC_SPAN], crc)
                raw_frames.append((header[0], data, ok))
                if not ok:
                    break  # the length cannot be trusted, nor anything after it
        except TpegDecodeError:
            pass
        if encryption != 0:
            self._count(encrypted_frames=1)
            components = tuple(
                ComponentFrame(scid, len(data), ok, None) for scid, data, ok in raw_frames
            )
            return TransportFrame(FRAME_TYPE_SERVICE, len(body), sid, encryption, components)
        # The SNI first: it says what the other components carry.
        decoded: dict[int, ComponentFrame] = {}
        for position, (scid, data, ok) in enumerate(raw_frames):
            if scid == 0:
                decoded[position] = self._sni_frame(data, ok)
        for position, (scid, data, ok) in enumerate(raw_frames):
            if scid != 0:
                decoded[position] = self._application_frame(scid, data, ok, messages)
        components = tuple(decoded[i] for i in range(len(raw_frames)))
        return TransportFrame(FRAME_TYPE_SERVICE, len(body), sid, encryption, components)

    def _checked(self, scid: int, data: bytes, header_ok: bool) -> bool | None:
        """Counts a component frame and checks its data CRC: True if the data can be
        used, False if the data CRC is wrong, None if the header CRC already was."""
        data_ok = None
        if header_ok:
            data_ok = len(data) >= 2 + _CRC_BYTES and _crc_ok(data[:-2], data[-2:])
        stats = self._components.get(scid, ComponentStatistics())
        self._components[scid] = replace(
            stats,
            frames=stats.frames + 1,
            bytes=stats.bytes + len(data),
            crc_errors=stats.crc_errors + (not data_ok),
        )
        return data_ok

    def _count_messages(self, scid: int, decoded: int, errors: int) -> None:
        stats = self._components[scid]
        self._components[scid] = replace(
            stats, messages=stats.messages + decoded, errors=stats.errors + errors
        )

    def _sni_frame(self, data: bytes, header_ok: bool) -> ComponentFrame:
        data_ok = self._checked(0, data, header_ok)
        if not data_ok:
            return ComponentFrame(0, len(data), header_ok, AID_SNI, data_ok)
        reader = TpegReader(data, 0, len(data) - _CRC_BYTES)
        count = reader.u8()
        decoded = errors = 0
        service = self._service
        try:
            while not reader.at_end():
                table = reader.u8()
                content = reader.bytes(reader.u16())
                try:
                    service = _decode_sni_table(service, table, content)
                    decoded += 1
                except TpegDecodeError:
                    errors += 1
                if table not in service.tables:
                    service = replace(service, tables=tuple(sorted(service.tables + (table,))))
        except TpegDecodeError:
            errors += 1
        self._service = service
        self._count_messages(0, decoded, errors)
        return ComponentFrame(0, len(data), True, AID_SNI, True, None, count, decoded, errors)

    def _application_frame(
        self, scid: int, data: bytes, header_ok: bool, messages: list[Message]
    ) -> ComponentFrame:
        application = self._service.application(scid)
        decode = _MESSAGE_DECODERS.get(application)
        if decode is None:
            # Without knowing the application, not even the frame layout is known.
            stats = self._components.get(scid, ComponentStatistics())
            self._components[scid] = replace(
                stats, frames=stats.frames + 1, bytes=stats.bytes + len(data)
            )
            return ComponentFrame(scid, len(data), header_ok, application)
        data_ok = self._checked(scid, data, header_ok)
        if not data_ok:
            return ComponentFrame(scid, len(data), header_ok, application, data_ok)
        priority, count = data[0], data[1]
        decoded = errors = 0
        try:
            trees = parse_components(data[2:-_CRC_BYTES])
        except TpegDecodeError:
            trees = ()
            errors += 1
        for tree in trees:
            try:
                messages.append(decode(scid, tree))
                decoded += 1
            except TpegDecodeError:
                errors += 1
        self._count_messages(scid, decoded, errors)
        return ComponentFrame(
            scid, len(data), True, application, True, priority, count, decoded, errors
        )


@dataclass(frozen=True)
class StoredMessage:
    message: Message
    frame: TransportFrame  # the frame it arrived in last
    received: float  # POSIX seconds of the last reception


class TpegMessageStore:
    """The messages currently on air, by service component and message ID.

    A message is replaced by every newer reception (the version changes with the
    content, the expiry time is extended without a new version), removed by a message
    with the cancel flag, and dropped once its expiry time has passed. What the service
    stopped repeating while the receiver was away is dropped by `drop_received_before`.
    """

    def __init__(self) -> None:
        self._messages: dict[tuple[int, int], StoredMessage] = {}

    def __len__(self) -> int:
        return len(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def add(self, message: Message, frame: TransportFrame, received: float) -> bool:
        """Takes a received message; whether the set of messages or a content changed
        (new message, new version, cancellation)."""
        key = (message.scid, message.management.message_id)
        if message.management.cancel:
            return self._messages.pop(key, None) is not None
        previous = self._messages.get(key)
        self._messages[key] = StoredMessage(message, frame, received)
        return previous is None or previous.message.management.version != message.management.version

    def expire(self, now: float) -> bool:
        """Drops the messages that expired before `now`; whether any did."""
        expired = [k for k, s in self._messages.items() if s.message.management.expiry < now]
        for key in expired:
            del self._messages[key]
        return bool(expired)

    def drop_received_before(self, received: float) -> bool:
        """Drops the messages last received before `received`; whether any were."""
        stale = [k for k, s in self._messages.items() if s.received < received]
        for key in stale:
            del self._messages[key]
        return bool(stale)

    def get(self, key: tuple[int, int]) -> StoredMessage | None:
        return self._messages.get(key)

    def messages(self) -> list[StoredMessage]:
        return list(self._messages.values())
