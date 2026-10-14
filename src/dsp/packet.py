"""Packet mode: packets, their outer error protection and the data groups they carry
(ETSI EN 300 401, clauses 5.3.2 to 5.3.5).

Data services such as the programme guide (SPI/EPG) or TPEG do not travel in the X-PAD of
an audio service but in a subchannel of their own. Several of them can share it: the
subchannel carries a sequence of packets of 24, 48, 72 or 96 bytes, each with a 10-bit
address naming the service component it belongs to, and a CRC of its own.

On top of the convolutional code every packet mode subchannel carries a second, outer
error protection (clause 5.3.5, signalled in FIG 0/14): the packets are written column by
column into a table of 12 rows and 188 columns, and each row is protected by a
Reed-Solomon code RS(204,188) that corrects up to 8 wrong bytes. The 192 parity bytes
follow the table in nine FEC packets of their own (address 1022). Because the table is
filled by columns but coded by rows, an error burst in the byte stream is spread over
all twelve rows, each of which only sees a twelfth of it.

The path through this module:

1. `PacketFecDecoder` finds the FEC frames in the byte stream of the subchannel (2 256
   bytes of packets, then the nine FEC packets) and corrects them. Without FEC the
   packets are cut directly out of each logical frame, which always starts with one.
2. `split_packets` cuts the packets out of the corrected data and checks their CRC.
3. `PacketDataGroupAssembler` joins the packets of one address into MSC data groups
   (clause 5.3.4) by their first/last flags and continuity index.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dab.crc import check_crc
from dab.data_group import DataGroup, parse_data_group
from dsp.reed_solomon import PACKET_CODEWORD_BYTES, PACKET_DATA_BYTES, decode_packet_rs_codeword

# Clause 5.3.2.1, table 6: the packet lengths selected by the two top bits of the header.
PACKET_LENGTHS = (24, 48, 72, 96)
_HEADER_BYTES = 3
_CRC_BYTES = 2

PADDING_ADDRESS = 0  # clause 5.3.2.1: padding packets
FEC_ADDRESS = 1022  # clause 5.3.5.2: packets carrying the RS data table

# Clause 5.3.5.1: the FEC frame. The packets fill the application data table column by
# column; each of its 12 rows is one RS(204,188) codeword together with the same row of
# the RS data table.
FEC_ROWS = 12
FEC_COLUMNS = PACKET_CODEWORD_BYTES  # 204
ADT_COLUMNS = PACKET_DATA_BYTES  # 188
ADT_BYTES = FEC_ROWS * ADT_COLUMNS  # 2 256
RS_TABLE_BYTES = FEC_ROWS * (FEC_COLUMNS - ADT_COLUMNS)  # 192
# Clause 5.3.5.2: nine 24-byte FEC packets with a 2-byte header and 22 bytes of RS data;
# the last 6 bytes of the ninth are padding.
FEC_PACKETS = 9
FEC_PACKET_BYTES = 24
_FEC_PACKET_HEADER_BYTES = 2
FEC_SET_BYTES = FEC_PACKETS * FEC_PACKET_BYTES  # 216
FEC_FRAME_BYTES = ADT_BYTES + FEC_SET_BYTES  # 2 472

# How many of the nine FEC packet headers must be intact. The headers are not covered by
# the RS code; they only mark where the frame is. Seven of nine correct 16-bit patterns
# are practically impossible by chance, so a search with this threshold does not lock onto
# payload bytes. Once locked, the position follows from the frame length and a few
# intact headers only confirm it.
_FEC_HEADERS_TO_LOCK = 7
_FEC_HEADERS_TO_CONFIRM = 3
# Frames in a row without confirmation after which the frame grid is searched afresh.
_MAX_FEC_MISSES = 3


@dataclass(frozen=True)
class Packet:
    """One packet (clause 5.3.2)."""

    address: int
    length: int  # 24, 48, 72 or 96 bytes, header and CRC included
    continuity_index: int  # modulo 4 per address
    first: bool
    last: bool
    command: bool  # 1 = CA command packet, 0 = data
    data: bytes  # the useful data field, without padding
    crc_ok: bool

    @property
    def is_padding(self) -> bool:
        return self.address == PADDING_ADDRESS


def packet_length(first_header_byte: int) -> int:
    return PACKET_LENGTHS[first_header_byte >> 6]


def parse_packet(raw: bytes) -> Packet:
    """Reads a packet of exactly its length (see `packet_length`).

    The useful data length is limited to the data field, so that even a packet with a
    wrong CRC never yields more bytes than it has.
    """
    header = int.from_bytes(raw[:_HEADER_BYTES], "big")
    useful_length = header & 0x7F
    field_length = len(raw) - _HEADER_BYTES - _CRC_BYTES
    return Packet(
        address=(header >> 8) & 0x3FF,
        length=len(raw),
        continuity_index=(header >> 20) & 0x03,
        first=bool(header & 0x080000),
        last=bool(header & 0x040000),
        command=bool(header & 0x80),
        data=bytes(raw[_HEADER_BYTES : _HEADER_BYTES + min(useful_length, field_length)]),
        crc_ok=check_crc(raw),
    )


def split_packets(data: bytes) -> tuple[list[Packet], list[tuple[int, int]]]:
    """Cuts consecutive packets out of `data`, which must start with a packet.

    FEC packets (address 1022) are skipped: they carry no service data, and a receiver
    that does not evaluate them simply discards them (clause 5.3.5.2, note).

    Returns:
        `(packets, spans)`: the data packets and, for all packets including FEC packets,
        their (start, length) in `data`. A wrong length in a corrupted header shifts all
        following packets; the caller notices by their CRCs.
    """
    packets: list[Packet] = []
    spans: list[tuple[int, int]] = []
    offset = 0
    while offset < len(data):
        length = packet_length(data[offset])
        if offset + length > len(data):
            break
        raw = data[offset : offset + length]
        spans.append((offset, length))
        if ((raw[0] & 0x03) << 8 | raw[1]) != FEC_ADDRESS:
            packets.append(parse_packet(raw))
        offset += length
    return packets, spans


@dataclass(frozen=True)
class FecFrameResult:
    """One FEC frame after the Reed-Solomon decoding, for the packets it contains and
    for the display of the table.

    The table is indexed like the standard's figure 15: `[row, column]`, the application
    data table in columns 0..187, the RS data table in 188..203.
    """

    # (12, 204) bool: the bytes the decoder changed.
    corrected: np.ndarray
    # Per row: number of corrected bytes, -1 if the row had more than 8 errors.
    row_corrections: tuple[int, ...]
    # (188,) address of the packet occupying each column of the application data table
    # (every packet covers whole columns), -1 where no packet could be read.
    column_addresses: np.ndarray
    packets: list[Packet]
    crc_errors_before: int  # packets with a wrong CRC without the RS correction
    crc_errors_after: int

    @property
    def corrected_bytes(self) -> int:
        return sum(n for n in self.row_corrections if n > 0)

    @property
    def uncorrectable_rows(self) -> int:
        return sum(1 for n in self.row_corrections if n < 0)


def decode_fec_frame(frame: bytes) -> FecFrameResult:
    """Corrects one FEC frame of 2 472 bytes: the application data table followed by the
    nine FEC packets (clause 5.3.5)."""
    if len(frame) != FEC_FRAME_BYTES:
        raise ValueError(f"An FEC frame must be {FEC_FRAME_BYTES} bytes long.")
    stream = np.frombuffer(frame, dtype=np.uint8)

    rs_data = np.concatenate(
        [
            stream[start + _FEC_PACKET_HEADER_BYTES : start + FEC_PACKET_BYTES]
            for start in range(ADT_BYTES, FEC_FRAME_BYTES, FEC_PACKET_BYTES)
        ]
    )[:RS_TABLE_BYTES]
    table = np.empty((FEC_ROWS, FEC_COLUMNS), dtype=np.uint8)
    # Both tables are filled column by column: byte k lies in row k % 12, column k // 12.
    table[:, :ADT_COLUMNS] = stream[:ADT_BYTES].reshape(ADT_COLUMNS, FEC_ROWS).T
    table[:, ADT_COLUMNS:] = rs_data.reshape(-1, FEC_ROWS).T
    received = table.copy()

    row_corrections = []
    for row in range(FEC_ROWS):
        corrected, count = decode_packet_rs_codeword(table[row])
        row_corrections.append(count)
        if count > 0:
            table[row] = corrected

    adt_before = stream[:ADT_BYTES].tobytes()
    adt_after = table[:, :ADT_COLUMNS].T.reshape(-1).tobytes()
    packets_before, _ = split_packets(adt_before)
    packets, spans = split_packets(adt_after)

    column_addresses = np.full(ADT_COLUMNS, -1, dtype=np.int16)
    for start, length in spans:
        address = (adt_after[start] & 0x03) << 8 | adt_after[start + 1]
        column_addresses[start // FEC_ROWS : (start + length) // FEC_ROWS] = address

    return FecFrameResult(
        corrected=table != received,
        row_corrections=tuple(row_corrections),
        column_addresses=column_addresses,
        packets=packets,
        crc_errors_before=sum(not p.crc_ok for p in packets_before),
        crc_errors_after=sum(not p.crc_ok for p in packets),
    )


def _fec_header_matches(stream: np.ndarray, start: int, stop: int) -> np.ndarray:
    """For every candidate start of a set of FEC packets in `start..stop-1`: how many of
    its nine headers are intact (packet length 24, counter 0..8, address 1022)."""
    matches = np.zeros(stop - start, dtype=np.int8)
    for counter in range(FEC_PACKETS):
        offset = start + counter * FEC_PACKET_BYTES
        first = stream[offset : offset + len(matches)]
        second = stream[offset + 1 : offset + 1 + len(matches)]
        matches += (first == ((counter << 2) | 0x03)) & (second == (FEC_ADDRESS & 0xFF))
    return matches


class PacketFecDecoder:
    """Finds the FEC frames in the continuous byte stream of a subchannel and corrects
    them.

    The frames are not aligned with the logical frames (2 472 bytes against 3 bytes per
    kbit/s every 24 ms); what marks them is the set of nine FEC packets at their end. It
    is searched for once, after that every frame follows exactly 2 472 bytes after the
    previous one, as long as the stream has no gap (the caller then calls `reset`).
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._locked = False
        self._misses = 0

    @property
    def locked(self) -> bool:
        return self._locked

    def reset(self) -> None:
        self._buffer.clear()
        self._locked = False
        self._misses = 0

    def push(self, data: bytes) -> list[FecFrameResult]:
        """Adds the bytes of one logical frame.

        Returns:
            The FEC frames completed with it (usually none, at most a few).
        """
        self._buffer.extend(data)
        results = []
        while True:
            if not self._locked and not self._acquire():
                return results
            if len(self._buffer) < FEC_FRAME_BYTES:
                return results
            frame = bytes(self._buffer[:FEC_FRAME_BYTES])
            stream = np.frombuffer(frame, dtype=np.uint8)
            if _fec_header_matches(stream, ADT_BYTES, ADT_BYTES + 1)[0] < _FEC_HEADERS_TO_CONFIRM:
                self._misses += 1
                if self._misses >= _MAX_FEC_MISSES:
                    # The grid is lost; search again from the byte after the frame start.
                    self._locked = False
                    self._misses = 0
                    del self._buffer[:1]
                    continue
            else:
                self._misses = 0
            results.append(decode_fec_frame(frame))
            del self._buffer[:FEC_FRAME_BYTES]

    def _acquire(self) -> bool:
        """Searches the buffer for a set of FEC packets with a complete application data
        table in front of it and drops everything before that table."""
        stream = np.frombuffer(bytes(self._buffer), dtype=np.uint8)
        last_start = len(stream) - FEC_SET_BYTES
        if last_start >= ADT_BYTES:
            matches = _fec_header_matches(stream, ADT_BYTES, last_start + 1)
            found = np.flatnonzero(matches >= _FEC_HEADERS_TO_LOCK)
            if found.size:
                del self._buffer[: int(found[0])]  # the table starts ADT_BYTES before the set
                self._locked = True
                self._misses = 0
                return True
        # Keep just enough for a set that is only partly received yet.
        keep = FEC_FRAME_BYTES - 1
        if len(self._buffer) > keep:
            del self._buffer[: len(self._buffer) - keep]
        return False


@dataclass
class AddressStatistics:
    """What arrived for one packet address so far."""

    packets: int = 0
    crc_errors: int = 0  # packets with a wrong CRC (after the FEC, if any)
    useful_bytes: int = 0  # useful data of the CRC-valid packets
    data_groups: int = 0  # complete data groups with a valid CRC
    data_group_errors: int = 0  # complete data groups with a wrong CRC
    continuity_errors: int = 0  # data groups abandoned because packets were missing
    last_data_group: bytes = b""  # the most recent valid data group, for a look inside


class PacketDataGroupAssembler:
    """Joins the useful data of the packets of each address into MSC data groups (clause
    5.3.4): the first packet of a group has the first flag, the last one the last flag,
    and the continuity index counts on by one from packet to packet.

    A packet with a wrong CRC or a gap in the continuity index abandons the group, whose
    data group CRC could not be right anyway; it arrives again with the next pass of the
    carousel.
    """

    def __init__(self) -> None:
        # Address -> (bytes so far, continuity index of the most recent packet).
        self._partial: dict[int, tuple[bytearray, int]] = {}

    def reset(self) -> None:
        self._partial.clear()

    def push(self, packet: Packet, statistics: AddressStatistics) -> bytes | None:
        """Adds a data packet; returns a data group when it completes one."""
        address = packet.address
        if not packet.crc_ok:
            self._partial.pop(address, None)
            return None
        if packet.first:
            if address in self._partial:
                statistics.continuity_errors += 1  # the previous group never ended
            group = bytearray(packet.data)
        else:
            partial = self._partial.pop(address, None)
            if partial is None:
                return None  # joined in the middle of a group
            group, index = partial
            if packet.continuity_index != (index + 1) % 4:
                statistics.continuity_errors += 1
                return None
            group.extend(packet.data)
        if packet.last:
            self._partial.pop(address, None)
            return bytes(group)
        self._partial[address] = (group, packet.continuity_index)
        return None


@dataclass(frozen=True)
class PacketStatistics:
    """Snapshot of what the decoding of a packet mode subchannel has seen so far."""

    seconds: float  # signal duration covered (24 ms per logical frame)
    fec: bool  # whether the outer FEC is decoded
    fec_locked: bool
    fec_frames: int
    fec_corrected_bytes: int
    fec_uncorrectable_rows: int
    crc_errors_before_fec: int  # packets that the FEC repaired or not, see `addresses`
    padding_packets: int
    addresses: dict[int, AddressStatistics] = field(default_factory=dict)


@dataclass
class PacketModeOutput:
    """What one logical frame of a packet mode subchannel yielded."""

    fec_frames: list[FecFrameResult] = field(default_factory=list)
    # (address, data group) for every complete data group with a valid CRC, in order.
    data_groups: list[tuple[int, DataGroup]] = field(default_factory=list)


class PacketModeDecoder:
    """Everything between the logical frames of a packet mode subchannel and its data
    groups: FEC (if signalled in FIG 0/14), packets, data groups, and the statistics.

    Addresses whose components do not use data groups (DG flag in FIG 0/3) can be passed
    in `raw_addresses`; their packets are only counted.
    """

    def __init__(self, fec: bool, raw_addresses: frozenset[int] = frozenset()) -> None:
        self._fec = fec
        self._raw_addresses = raw_addresses
        self._fec_decoder = PacketFecDecoder()
        self._assembler = PacketDataGroupAssembler()
        self._logical_frames = 0
        self._fec_frames = 0
        self._fec_corrected_bytes = 0
        self._fec_uncorrectable_rows = 0
        self._crc_errors_before_fec = 0
        self._padding_packets = 0
        self._addresses: dict[int, AddressStatistics] = {}

    @property
    def fec(self) -> bool:
        return self._fec

    def reset(self) -> None:
        """After a gap in the logical frames: packets and data groups under way are lost,
        the statistics stay."""
        self._fec_decoder.reset()
        self._assembler.reset()

    def push(self, logical_frame: bytes) -> PacketModeOutput:
        self._logical_frames += 1
        output = PacketModeOutput()
        if self._fec:
            for frame in self._fec_decoder.push(logical_frame):
                output.fec_frames.append(frame)
                self._fec_frames += 1
                self._fec_corrected_bytes += frame.corrected_bytes
                self._fec_uncorrectable_rows += frame.uncorrectable_rows
                self._crc_errors_before_fec += frame.crc_errors_before
                self._handle_packets(frame.packets, output)
        else:
            packets, _ = split_packets(logical_frame)
            self._handle_packets(packets, output)
        return output

    def _handle_packets(self, packets: list[Packet], output: PacketModeOutput) -> None:
        for packet in packets:
            if packet.is_padding:
                self._padding_packets += 1
                continue
            statistics = self._addresses.setdefault(packet.address, AddressStatistics())
            statistics.packets += 1
            if not packet.crc_ok:
                statistics.crc_errors += 1
            else:
                statistics.useful_bytes += len(packet.data)
            if packet.address in self._raw_addresses:
                continue
            group = self._assembler.push(packet, statistics)
            if group is None:
                continue
            parsed = parse_data_group(group)
            if parsed is None:
                statistics.data_group_errors += 1
                continue
            statistics.data_groups += 1
            statistics.last_data_group = group
            output.data_groups.append((packet.address, parsed))

    def statistics(self) -> PacketStatistics:
        return PacketStatistics(
            seconds=self._logical_frames * 0.024,
            fec=self._fec,
            fec_locked=self._fec_decoder.locked,
            fec_frames=self._fec_frames,
            fec_corrected_bytes=self._fec_corrected_bytes,
            fec_uncorrectable_rows=self._fec_uncorrectable_rows,
            crc_errors_before_fec=self._crc_errors_before_fec,
            padding_packets=self._padding_packets,
            addresses={
                address: AddressStatistics(**vars(stats))
                for address, stats in sorted(self._addresses.items())
            },
        )
