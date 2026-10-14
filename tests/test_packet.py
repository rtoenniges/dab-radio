"""Tests for packet mode: packets, packet FEC and data groups (ETSI EN 300 401,
clauses 5.3.2 to 5.3.5)."""

from __future__ import annotations

import numpy as np
import pytest

from dab.crc import crc16_ccitt
from dsp.packet import (
    ADT_BYTES,
    ADT_COLUMNS,
    FEC_FRAME_BYTES,
    FEC_ROWS,
    PACKET_LENGTHS,
    AddressStatistics,
    PacketDataGroupAssembler,
    PacketFecDecoder,
    PacketModeDecoder,
    decode_fec_frame,
    parse_packet,
    split_packets,
)
from dsp.reed_solomon import PACKET_NROOTS
from tests.rs_test_utils import encode_rs_codeword

# Real packets from channel 5C, CRC included. EPG Deutschland (address 3): the last
# packet of a data group, 48 bytes, with 23 bytes of useful data from a programme text.
REAL_EPG_PACKET = bytes.fromhex(
    "540316697420696e746572657373696572742073696368ef4e00000000000000000000000000000000000000"
    "00003779"
)
# GEPOS (address 2): the last packet of a data group with a single useful byte.
REAL_GEPOS_PACKET = bytes.fromhex("3402013f0000000000000000000000000000000000006459")


def _packet(
    address: int,
    data: bytes,
    *,
    first: bool = True,
    last: bool = True,
    continuity: int = 0,
    length: int = 24,
) -> bytes:
    """A packet with a valid CRC (clause 5.3.2)."""
    field = length - 5
    header = (
        (PACKET_LENGTHS.index(length) << 22)
        | (continuity << 20)
        | (first << 19)
        | (last << 18)
        | (address << 8)
        | len(data)
    )
    body = header.to_bytes(3, "big") + data + bytes(field - len(data))
    return body + (crc16_ccitt(body) ^ 0xFFFF).to_bytes(2, "big")


def _data_group(payload: bytes, group_type: int = 0) -> bytes:
    """An MSC data group with CRC and nothing but the two header bytes."""
    body = bytes([0x40 | group_type, 0x00]) + payload
    return body + (crc16_ccitt(body) ^ 0xFFFF).to_bytes(2, "big")


def _padding_packets(count: int) -> bytes:
    return _packet(0, b"") * count


def _fec_frame(packets: bytes) -> bytes:
    """Appends the nine FEC packets to an application data table of packets, as the
    transmitter does (clause 5.3.5): the packets fill the table column by column, each
    row gets its 16 RS parity bytes, and the parity table is sent column by column."""
    assert len(packets) == ADT_BYTES
    table = np.frombuffer(packets, dtype=np.uint8).reshape(ADT_COLUMNS, FEC_ROWS).T
    parity = np.array(
        [encode_rs_codeword(table[row], nroots=PACKET_NROOTS)[ADT_COLUMNS:] for row in range(12)]
    )
    rs_stream = parity.T.reshape(-1).tobytes() + bytes(6)
    fec = b"".join(
        bytes([(counter << 2) | 0x03, 0xFE]) + rs_stream[counter * 22 : (counter + 1) * 22]
        for counter in range(9)
    )
    return packets + fec


def _application_data(seed: int = 0) -> tuple[bytes, list[bytes]]:
    """2 256 bytes of packets: address 7 with random data in 72-byte packets, filled up
    with padding. Returns the table and the data packets."""
    rng = np.random.default_rng(seed)
    data_packets = [
        _packet(7, rng.integers(0, 256, 67, dtype=np.uint8).tobytes(), length=72, continuity=i % 4)
        for i in range(20)
    ]
    table = b"".join(data_packets)
    table += _padding_packets((ADT_BYTES - len(table)) // 24)
    return table, data_packets


def test_parse_packet_reads_the_header():
    raw = _packet(0x155, b"hello", first=True, last=False, continuity=2, length=48)

    packet = parse_packet(raw)

    assert packet.address == 0x155
    assert packet.length == 48
    assert packet.continuity_index == 2
    assert packet.first and not packet.last
    assert not packet.command
    assert packet.data == b"hello"
    assert packet.crc_ok


def test_parse_packet_of_real_packets():
    epg = parse_packet(REAL_EPG_PACKET)
    assert (epg.address, epg.length, epg.continuity_index) == (3, 48, 1)
    assert (epg.first, epg.last, epg.crc_ok) == (False, True, True)
    assert epg.data.startswith(b"it interessiert sich")

    gepos = parse_packet(REAL_GEPOS_PACKET)
    assert (gepos.address, gepos.length, gepos.data, gepos.crc_ok) == (2, 24, b"?", True)


def test_parse_packet_detects_a_wrong_crc():
    raw = bytearray(REAL_GEPOS_PACKET)
    raw[5] ^= 0x01
    assert not parse_packet(bytes(raw)).crc_ok


def test_split_packets_skips_fec_packets():
    fec_packet = bytes([0x03, 0xFE]) + bytes(22)
    data = _packet(5, b"a") + fec_packet + _packet(6, b"b", length=48)

    packets, spans = split_packets(data)

    assert [p.address for p in packets] == [5, 6]
    assert spans == [(0, 24), (24, 24), (48, 48)]


def test_fec_frame_without_errors_needs_no_correction():
    table, _ = _application_data()

    result = decode_fec_frame(_fec_frame(table))

    assert result.row_corrections == (0,) * FEC_ROWS
    assert not result.corrected.any()
    assert [p.address for p in result.packets if not p.is_padding] == [7] * 20
    assert result.crc_errors_before == result.crc_errors_after == 0


def test_fec_frame_spreads_an_error_burst_over_all_rows():
    """96 consecutive wrong bytes (a burst in the byte stream) occupy 8 columns of the
    table: every row sees 8 errors, just what RS(204,188) can correct."""
    table, data_packets = _application_data()
    frame = bytearray(_fec_frame(table))
    for position in range(288, 384):  # packets 4 and 5 (72 bytes each), partly
        frame[position] ^= 0xA5

    result = decode_fec_frame(bytes(frame))

    assert result.row_corrections == (8,) * FEC_ROWS
    assert result.corrected_bytes == 96
    assert result.corrected[:, 24:32].all()
    assert result.corrected.sum() == 96
    # Without the FEC at least the two packets hit are lost; since the burst also hits a
    # header, the length read from it misplaces the following packet as well.
    assert result.crc_errors_before >= 2
    assert result.crc_errors_after == 0
    received = [p for p in result.packets if not p.is_padding]
    assert [p.data for p in received] == [parse_packet(raw).data for raw in data_packets]


def test_fec_frame_reports_a_row_beyond_repair():
    table, _ = _application_data(seed=3)
    frame = bytearray(_fec_frame(table))
    for column in range(9):  # 9 errors in row 0, the first byte of each column
        frame[column * FEC_ROWS] ^= 0xFF

    result = decode_fec_frame(bytes(frame))

    assert result.row_corrections[0] == -1
    assert result.row_corrections[1:] == (0,) * (FEC_ROWS - 1)
    assert result.uncorrectable_rows == 1
    assert result.crc_errors_after > 0


def test_fec_frame_maps_packets_to_columns():
    table = _packet(5, b"x", length=48) + _padding_packets((ADT_BYTES - 48) // 24)

    result = decode_fec_frame(_fec_frame(table))

    assert list(result.column_addresses[:4]) == [5] * 4
    assert set(result.column_addresses[4:]) == {0}


def test_decode_fec_frame_raises_for_wrong_length():
    with pytest.raises(ValueError, match="2472 bytes"):
        decode_fec_frame(bytes(100))


def test_fec_decoder_finds_the_frames_in_a_stream_of_logical_frames():
    """The frames are not aligned with the logical frames; the decoder finds them by the
    nine FEC packets, wherever the stream starts."""
    rng = np.random.default_rng(9)
    frames = [_fec_frame(_application_data(seed)[0]) for seed in range(3)]
    stream = rng.integers(0, 256, 1000, dtype=np.uint8).tobytes() + b"".join(frames)
    decoder = PacketFecDecoder()

    results = []
    for start in range(0, len(stream), 72):
        results += decoder.push(stream[start : start + 72])

    assert len(results) == 3
    assert decoder.locked
    assert all(r.crc_errors_after == 0 and len(r.packets) > 0 for r in results)


def test_fec_decoder_searches_again_after_losing_the_grid():
    frames = [_fec_frame(_application_data(seed)[0]) for seed in range(8)]
    # 24 bytes go missing without the caller noticing: the grid is shifted from there on.
    stream = frames[0] + frames[1][24:] + b"".join(frames[2:])
    decoder = PacketFecDecoder()

    results = decoder.push(stream)

    assert results[0].crc_errors_after == 0
    assert results[-1].crc_errors_after == 0  # locked onto the new grid
    assert results[-1].row_corrections == (0,) * FEC_ROWS


def test_assembler_joins_packets_into_a_data_group():
    group = _data_group(bytes(range(50)))
    raw = [
        _packet(9, group[:19], first=True, last=False, continuity=3),
        _packet(9, group[19:38], first=False, last=False, continuity=0),
        _packet(9, group[38:], first=False, last=True, continuity=1),
    ]
    assembler = PacketDataGroupAssembler()
    statistics = AddressStatistics()

    results = [assembler.push(parse_packet(p), statistics) for p in raw]

    assert results == [None, None, group]


def test_assembler_abandons_a_group_with_a_missing_packet():
    group = _data_group(bytes(range(50)))
    raw = [
        _packet(9, group[:19], first=True, last=False, continuity=0),
        _packet(9, group[38:], first=False, last=True, continuity=2),  # 1 is missing
    ]
    assembler = PacketDataGroupAssembler()
    statistics = AddressStatistics()

    results = [assembler.push(parse_packet(p), statistics) for p in raw]

    assert results == [None, None]
    assert statistics.continuity_errors == 1


def test_assembler_keeps_addresses_apart():
    group_a = _data_group(b"A" * 30)
    group_b = _data_group(b"B" * 5)
    assembler = PacketDataGroupAssembler()
    statistics = AddressStatistics()

    first = assembler.push(parse_packet(_packet(1, group_a[:19], last=False)), statistics)
    other = assembler.push(parse_packet(_packet(2, group_b)), statistics)
    rest = assembler.push(
        parse_packet(_packet(1, group_a[19:], first=False, continuity=1)), statistics
    )

    assert (first, other, rest) == (None, group_b, group_a)


def test_packet_mode_decoder_with_fec_counts_and_delivers_data_groups():
    groups = [_data_group(bytes([i]) * 10, group_type=4) for i in range(3)]
    packets = b"".join(_packet(3, g, continuity=i) for i, g in enumerate(groups))
    packets += _padding_packets((ADT_BYTES - len(packets)) // 24)
    stream = _fec_frame(packets) * 2
    decoder = PacketModeDecoder(fec=True)

    delivered = []
    for start in range(0, len(stream), 72):
        delivered += decoder.push(stream[start : start + 72]).data_groups

    assert [(address, group.data) for address, group in delivered] == [
        (3, bytes([i]) * 10) for i in range(3)
    ] * 2
    statistics = decoder.statistics()
    assert statistics.fec_frames == 2
    assert statistics.padding_packets == 2 * (ADT_BYTES // 24 - 3)
    assert statistics.addresses[3].data_groups == 6
    assert statistics.addresses[3].packets == 6
    logical_frames = -(-len(stream) // 72)
    assert statistics.seconds == pytest.approx(logical_frames * 0.024)


def test_packet_mode_decoder_without_fec_reads_each_logical_frame():
    group = _data_group(b"payload")
    frame = _packet(4, group) + _packet(0, b"")  # 48 bytes: 16 kbit/s
    decoder = PacketModeDecoder(fec=False)

    output = decoder.push(frame)

    assert [(address, g.data) for address, g in output.data_groups] == [(4, b"payload")]
    assert output.fec_frames == []


def test_packet_mode_decoder_only_counts_components_without_data_groups():
    frame = _packet(4, b"raw bytes") + _packet(5, _data_group(b"x"))
    decoder = PacketModeDecoder(fec=False, raw_addresses=frozenset({4}))

    output = decoder.push(frame)

    assert [address for address, _ in output.data_groups] == [5]
    assert decoder.statistics().addresses[4].packets == 1
    assert decoder.statistics().addresses[4].useful_bytes == len(b"raw bytes")


def test_fec_frame_bytes_constant():
    assert FEC_FRAME_BYTES == 2472
