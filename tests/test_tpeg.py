"""Tests for the TPEG decoding of ARD TPEG: binary types, frames, SNI, TEC and TFP."""

from __future__ import annotations

import pytest

from dab.crc import crc16_ccitt
from dab.data_group import parse_data_group
from dab.tpeg import (
    AID_EAW,
    AID_SNI,
    AID_TEC,
    AID_TFP,
    FRAME_TYPE_SERVICE,
    FRAME_TYPE_STREAM_DIRECTORY,
    SNI_LOGO,
    TpegDecoder,
    TpegMessageStore,
    service_id_text,
)
from dab.tpeg_binary import TpegDecodeError, TpegReader, coordinate_degrees, parse_components
from dab.tpeg_messages import (
    LOCATION_OPENLR,
    LOCATION_TMC,
    OPENLR_LINEAR,
    OPENLR_POINT_ALONG_LINE,
    TecMessage,
    TfpMessage,
    decode_tec_message,
    decode_tfp_message,
)
from dab.tpeg_tables import (
    advice_name,
    cause_name,
    component_name,
    effect_name,
    level_of_service_name,
    restriction_text,
)
from tests.tpeg_vectors import (
    DIRECTORY_GROUP,
    SNI_GROUP,
    TEC_CANCEL,
    TEC_LANE_CLOSED,
    TEC_ROADWORKS,
    TEC_STATIONARY_VEHICLE,
    TEC_TRAFFIC_LIGHTS,
    TEC_WEIGHT_LIMIT,
    TFP_CANCEL,
    TFP_FREE,
)

ARD_TPEG = (22, 98, 211)


def _crc(data: bytes) -> bytes:
    return (crc16_ccitt(data) ^ 0xFFFF).to_bytes(2, "big")


def _tec(data: bytes) -> TecMessage:
    (component,) = parse_components(data)
    return decode_tec_message(1, component)


def _tfp(data: bytes) -> TfpMessage:
    (component,) = parse_components(data)
    return decode_tfp_message(2, component)


def _component_frame(scid: int, data: bytes) -> bytes:
    header = bytes([scid]) + len(data).to_bytes(2, "big")
    return header + _crc(header + data[:13]) + data


def _application_frame(scid: int, messages: list[bytes], priority: int = 0) -> bytes:
    """A component frame of TEC or TFP: priority, count, messages, data CRC."""
    data = bytes([priority, len(messages)]) + b"".join(messages)
    return _component_frame(scid, data + _crc(data))


def _transport_frame(frame_type: int, body: bytes) -> bytes:
    header = b"\xff\x0f" + len(body).to_bytes(2, "big")
    return header + _crc(header + bytes([frame_type]) + body[:11]) + bytes([frame_type]) + body


def _service_frame(*component_frames: bytes, encryption: int = 0) -> bytes:
    return _transport_frame(
        FRAME_TYPE_SERVICE, bytes(ARD_TPEG) + bytes([encryption]) + b"".join(component_frames)
    )


def _group(continuity: int, data: bytes) -> bytes:
    raw = bytes([0x40, continuity << 4]) + data
    return raw + _crc(raw)


def _decoder_with_sni() -> TpegDecoder:
    decoder = TpegDecoder()
    decoder.push(parse_data_group(SNI_GROUP))
    return decoder


# --- Binary types -------------------------------------------------------------------


def test_multibyte_integers():
    reader = TpegReader(bytes.fromhex("7f 8c4c 81bb40 7f 8100 00"))
    assert reader.multibyte() == 127
    assert reader.multibyte() == 1612
    assert reader.multibyte() == 24000
    assert reader.signed_multibyte() == -1
    assert reader.signed_multibyte() == 128
    assert reader.signed_multibyte() == 0
    assert reader.at_end()


def test_selector_counts_from_bit_6_and_continues_in_bit_7():
    selector = TpegReader(bytes([0xC1, 0x40])).selector()
    assert [selector[i] for i in range(9)] == [
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
    ]
    assert not selector[20]
    assert selector.highest() == 7


def test_reading_past_the_end_raises():
    with pytest.raises(TpegDecodeError):
        TpegReader(b"\x01").u16()
    with pytest.raises(TpegDecodeError):
        TpegReader(b"\x80\x80").multibyte()


def test_coordinates():
    assert coordinate_degrees(0x06F714) == 9.79473
    assert coordinate_degrees(0x25333C) == 52.31268
    assert coordinate_degrees(-0x06F714) == -9.79473
    assert coordinate_degrees(0) == 0.0


def test_component_tree():
    (message,) = parse_components(TEC_ROADWORKS)
    assert message.size == len(TEC_ROADWORKS)
    assert [c.cid for c in message.children] == [1, 3, 2]
    location = message.child(2)
    assert [c.cid for c in location.children] == [LOCATION_OPENLR, 254]
    assert [c.cid for c in location.child(LOCATION_OPENLR).child(OPENLR_LINEAR).children] == [12]


def test_component_longer_than_its_parent_raises():
    with pytest.raises(TpegDecodeError):
        parse_components(TEC_ROADWORKS[:-1])


# --- TEC ----------------------------------------------------------------------------


def test_roadworks_with_openlr_line():
    message = _tec(TEC_ROADWORKS)
    management = message.management
    assert (management.message_id, management.version, management.cancel) == (1745, 0, False)
    assert management.expiry == 1790631844  # 28.09.2026 21:44:04 UTC
    assert management.priority == 1
    assert (message.effect, message.start, message.stop) == (7, 1781474400, 1798646400)
    (cause,) = message.causes
    assert (cause.main, cause.sub, cause.warning_level) == (3, None, 2)
    assert [(a.code, a.sub) for a in message.advice] == [(8, None)]
    location = message.location
    assert location.methods == (LOCATION_OPENLR, 254)
    assert location.tmc is None
    line = location.openlr
    assert (line.kind, line.version) == (OPENLR_LINEAR, (1, 0))
    assert [p.coordinate for p in line.points] == [(9.79473, 52.31268), (9.79867, 52.30842)]
    first = line.points[0]
    assert (first.frc, first.fow, first.lowest_frc, first.distance) == (2, 3, 2, 557)
    assert first.bearing == pytest.approx(151.875)
    assert (line.positive_offset, line.negative_offset) == (0, 140)
    assert line.length() == 557 - 140
    assert line.shape == ((9.79473, 52.31268), (9.79868, 52.30841))
    assert line.path() == line.shape


def test_lane_closed_with_precise_tmc_location():
    message = _tec(TEC_LANE_CLOSED)
    (cause,) = message.causes
    assert (cause.main, cause.lane_restriction) == (3, 4)
    tmc = message.location.tmc
    assert (tmc.location, tmc.country, tmc.table, tmc.extended_country_code) == (45524, 13, 1, 0xE0)
    assert (tmc.positive_direction, tmc.both_directions, tmc.extent) == (True, False, 1)
    assert tmc.table_version == (22, 0)
    assert (tmc.hazard_distance, tmc.problem_length) == (10100, 10100)
    openlr = message.location.openlr
    assert [(p.frc, p.fow) for p in openlr.points] == [(0, 1), (0, 1)]  # motorway
    assert openlr.length() == 10108


def test_point_along_line():
    message = _tec(TEC_TRAFFIC_LIGHTS)
    (cause,) = message.causes
    assert (cause.main, cause.sub) == (31, 3)
    assert cause_name(cause.main, cause.sub) == "traffic light failure"
    point = message.location.openlr
    assert point.kind == OPENLR_POINT_ALONG_LINE
    assert (point.side_of_road, point.orientation, point.positive_offset) == (0, 1, 0)
    assert point.points[0].coordinate == (8.80068, 53.07865)
    assert point.length() is None
    assert message.location.tmc.hazard_distance == 0


def test_vehicle_restriction():
    message = _tec(TEC_WEIGHT_LIMIT)
    assert (message.effect, message.start, message.stop) == (7, None, 1830272400)
    assert message.causes == ()
    (restriction,) = message.restrictions
    assert restriction.vehicle_type == 11
    assert restriction.restrictions == ((6, 24000),)
    assert restriction_text(6, 24000) == "weight over 24 t"
    assert message.location.openlr.positive_offset == 333


def test_intermediate_points_are_chained_in_order():
    message = _tec(TEC_STATIONARY_VEHICLE)
    (cause,) = message.causes
    assert (cause.main, cause.sub, cause.warning_level) == (10, 7, 3)
    assert cause_name(10, 7) == "stationary vehicle"
    line = message.location.openlr
    assert len(line.points) == 4
    # The last point is sent before the intermediate ones but is relative to the third.
    for point, shape_point in zip(line.points, line.shape, strict=True):
        assert point.coordinate == pytest.approx(shape_point, abs=5e-5)
    assert line.length() == 14269 + 850 + 6534


def test_cancellation_has_no_event():
    message = _tec(TEC_CANCEL)
    assert message.management.cancel
    assert message.management.message_id == 2703
    assert (message.effect, message.location) == (None, None)


def test_truncated_message_raises():
    component = parse_components(TEC_ROADWORKS)[0]
    management = component.child(1)
    broken = type(management)(1, management.attributes[:-1], (), management.size - 1)
    with pytest.raises(TpegDecodeError):
        decode_tec_message(1, type(component)(0, b"", (broken,), 0))


# --- TFP ----------------------------------------------------------------------------


def test_flow_matrix_with_tmc_location():
    message = _tfp(TFP_FREE)
    assert (message.management.message_id, message.management.version) == (2242, 92)
    assert (message.start, message.duration, message.spatial_resolution) == (1790623166, None, 0)
    (vector,) = message.vectors
    assert vector.time_offset == 0
    assert [(s.offset, s.level_of_service, s.average_speed) for s in vector.sections] == [
        (7, 1, None)
    ]
    tmc = message.location.tmc
    assert (tmc.location, tmc.positive_direction, tmc.extent) == (21731, False, 7)
    assert message.location.methods == (LOCATION_TMC,)


def test_tfp_cancellation():
    message = _tfp(TFP_CANCEL)
    assert message.management.cancel
    assert message.vectors == ()


# --- Transport, service frames and SNI ----------------------------------------------


def test_stream_directory():
    (frame,) = TpegDecoder().push(parse_data_group(DIRECTORY_GROUP))
    assert frame.frame.frame_type == FRAME_TYPE_STREAM_DIRECTORY
    assert frame.frame.directory == (ARD_TPEG,)
    assert frame.frame.directory_crc_ok
    assert service_id_text(ARD_TPEG) == "22.98.211"


def test_service_information():
    (frame,) = TpegDecoder().push(parse_data_group(SNI_GROUP))
    assert frame.frame.service_id == ARD_TPEG
    assert frame.frame.encryption == 0
    (sni,) = frame.frame.components
    assert (sni.scid, sni.application, sni.header_crc_ok, sni.data_crc_ok) == (
        0,
        AID_SNI,
        True,
        True,
    )
    assert (sni.message_count, sni.decoded, sni.errors) == (4, 4, 0)
    service = frame.service
    assert service.name == "ARD TPEG"
    assert service.description == "Aktuelle Verkehrsinformationen der ARD"
    assert [(e.scid, e.application, e.content_id) for e in service.components] == [
        (1, AID_TEC, 1),
        (2, AID_TFP, 2),
        (3, AID_EAW, 3),
    ]
    assert service.versions == {1: (3, 2), 2: (1, 0), 3: (1, 0)}
    assert service.message_counts == {1: 655, 2: 730, 3: 0}
    assert service.tables == (0, 1, 14, 33)
    assert frame.messages == ()


def test_messages_of_a_service_frame():
    decoder = _decoder_with_sni()
    raw = _service_frame(
        _application_frame(1, [TEC_ROADWORKS, TEC_LANE_CLOSED]),
        _application_frame(2, [TFP_FREE], priority=2),
    )
    (frame,) = decoder.push(parse_data_group(_group(1, raw)))
    tec, tfp = frame.frame.components
    assert (tec.scid, tec.application, tec.data_crc_ok, tec.message_count, tec.decoded) == (
        1,
        AID_TEC,
        True,
        2,
        2,
    )
    assert (tfp.application, tfp.group_priority, tfp.decoded) == (AID_TFP, 2, 1)
    assert [type(m).__name__ for m in frame.messages] == ["TecMessage", "TecMessage", "TfpMessage"]
    assert [m.management.message_id for m in frame.messages] == [1745, 1371, 2242]
    statistics = frame.statistics
    assert statistics.service_frames == 2
    assert statistics.components[1].messages == 2
    assert statistics.components[2].bytes == len(_application_frame(2, [TFP_FREE])) - 5


def test_wrong_data_crc_drops_the_messages():
    decoder = _decoder_with_sni()
    component = bytearray(_application_frame(1, [TEC_ROADWORKS]))
    component[-3] ^= 0x01  # inside the last message, after the 13 bytes of the header CRC
    (frame,) = decoder.push(parse_data_group(_group(1, _service_frame(bytes(component)))))
    (tec,) = frame.frame.components
    assert (tec.header_crc_ok, tec.data_crc_ok) == (True, False)
    assert frame.messages == ()
    assert frame.statistics.components[1].crc_errors == 1


def test_components_are_not_decoded_before_the_sni_names_them():
    raw = _service_frame(_application_frame(1, [TEC_ROADWORKS]))
    (frame,) = TpegDecoder().push(parse_data_group(_group(0, raw)))
    (tec,) = frame.frame.components
    assert (tec.application, tec.data_crc_ok) == (None, None)
    assert frame.messages == ()


def test_encrypted_service_frame_is_not_decoded():
    decoder = _decoder_with_sni()
    raw = _service_frame(_application_frame(1, [TEC_ROADWORKS]), encryption=1)
    (frame,) = decoder.push(parse_data_group(_group(1, raw)))
    assert frame.frame.encryption == 1
    assert frame.messages == ()
    assert frame.statistics.encrypted_frames == 1


def test_undecodable_message_is_counted_and_the_others_kept():
    decoder = _decoder_with_sni()
    broken = TEC_CANCEL[:3] + b"\x09" + TEC_CANCEL[4:]  # MMC component 1 renamed to 9
    raw = _service_frame(_application_frame(1, [broken, TEC_ROADWORKS]))
    (frame,) = decoder.push(parse_data_group(_group(1, raw)))
    (tec,) = frame.frame.components
    assert (tec.decoded, tec.errors) == (1, 1)
    assert [m.management.message_id for m in frame.messages] == [1745]


def test_logo_and_unknown_tables():
    decoder = TpegDecoder()
    png = b"\x89PNG\r\n\x1a\nrest"
    tables = bytes([SNI_LOGO]) + (len(png) + 1).to_bytes(2, "big") + b"\x01" + png
    tables += b"\x63\x00\x02ab"  # table 99: only listed
    data = b"\x02" + tables
    (frame,) = decoder.push(
        parse_data_group(_group(0, _service_frame(_component_frame(0, data + _crc(data)))))
    )
    assert frame.service.logo == png
    assert frame.service.tables == (SNI_LOGO, 99)


# --- Stream -------------------------------------------------------------------------


def test_frames_are_found_behind_garbage_and_across_data_groups():
    data = parse_data_group(SNI_GROUP).data
    stream = b"\x00\xff\x0f\x12\xff" + data  # a false sync word, then a half one
    decoder = TpegDecoder()
    assert decoder.push(parse_data_group(_group(0, stream[:40]))) == []
    (frame,) = decoder.push(parse_data_group(_group(1, stream[40:])))
    assert frame.service.name == "ARD TPEG"
    assert frame.statistics.discarded_bytes == 5


def test_repetition_is_skipped_and_continuity_counts_modulo_4():
    decoder = TpegDecoder()
    data = parse_data_group(DIRECTORY_GROUP).data
    assert len(decoder.push(parse_data_group(_group(3, data)))) == 1
    assert decoder.push(parse_data_group(_group(3, data))) == []  # repetition
    assert len(decoder.push(parse_data_group(_group(0, data)))) == 1  # 3 -> 0: no gap
    decoder.push(parse_data_group(_group(2, data)))  # 0 -> 2: one lost
    statistics = decoder.statistics()
    assert (statistics.data_groups, statistics.repeated_data_groups) == (3, 1)
    assert statistics.lost_data_groups == 1


def test_continuity_modulo_16_once_an_index_above_3_appears():
    decoder = TpegDecoder()
    data = parse_data_group(DIRECTORY_GROUP).data
    for index in (3, 4, 5, 7):
        decoder.push(parse_data_group(_group(index, data)))
    assert decoder.statistics().lost_data_groups == 1


def test_lost_data_group_drops_the_frame_under_way():
    data = parse_data_group(SNI_GROUP).data
    decoder = TpegDecoder()
    decoder.push(parse_data_group(_group(0, data[:40])))
    assert decoder.push(parse_data_group(_group(2, data[40:]))) == []
    (frame,) = decoder.push(parse_data_group(_group(3, data)))
    assert frame.service.name == "ARD TPEG"


def test_interrupt_drops_the_frame_under_way():
    data = parse_data_group(SNI_GROUP).data
    decoder = TpegDecoder()
    decoder.push(parse_data_group(_group(0, data[:40])))
    decoder.interrupt()
    assert decoder.push(parse_data_group(_group(1, data[40:]))) == []


# --- Message store ------------------------------------------------------------------


def test_message_store_replaces_cancels_and_expires():
    store = TpegMessageStore()
    frame = TpegDecoder().push(parse_data_group(SNI_GROUP))[0].frame
    roadworks = _tec(TEC_ROADWORKS)
    assert store.add(roadworks, frame, 100.0)
    assert not store.add(roadworks, frame, 130.0)  # same version: no change
    assert store.get((1, 1745)).received == 130.0
    cancel = _tec(TEC_CANCEL)
    assert not store.add(cancel, frame, 140.0)  # message 2703 is not stored
    tfp = _tfp(TFP_FREE)
    assert store.add(tfp, frame, 140.0)
    assert len(store) == 2
    assert store.expire(roadworks.management.expiry - 1)  # the TFP message expires first
    assert [s.message.management.message_id for s in store.messages()] == [1745]
    assert not store.expire(roadworks.management.expiry)
    assert store.expire(roadworks.management.expiry + 1)
    assert len(store) == 0


def test_message_store_drops_what_was_not_received_since():
    store = TpegMessageStore()
    frame = TpegDecoder().push(parse_data_group(SNI_GROUP))[0].frame
    store.add(_tec(TEC_ROADWORKS), frame, 100.0)
    store.add(_tfp(TFP_FREE), frame, 200.0)
    assert not store.drop_received_before(100.0)
    assert store.drop_received_before(150.0)
    assert [s.message.management.message_id for s in store.messages()] == [2242]


def test_cancellation_removes_a_stored_message():
    store = TpegMessageStore()
    frame = TpegDecoder().push(parse_data_group(SNI_GROUP))[0].frame
    tfp = _tfp(TFP_FREE)
    store.add(tfp, frame, 0.0)
    cancel = _tfp(TFP_CANCEL)
    cancel_of_stored = type(cancel)(
        2, type(cancel.management)(2242, 93, cancel.management.expiry, True), cancel.component
    )
    assert store.add(cancel_of_stored, frame, 1.0)
    assert len(store) == 0


# --- Tables -------------------------------------------------------------------------


def test_texts():
    assert effect_name(7) == "road closed"
    assert effect_name(None) == "-"
    assert cause_name(3) == "roadworks"
    assert cause_name(3, 1) == "major roadworks"
    assert cause_name(3, 99) == "roadworks"  # unknown sub-cause: the main cause
    assert cause_name(None) == "linked message"
    assert advice_name(8, 1) == "follow the diversion: follow the diversion signs"
    assert restriction_text(6, 3500) == "weight over 3.5 t"
    assert restriction_text(26, None) == "through traffic"
    assert level_of_service_name(1) == "free flow"
    assert level_of_service_name(17) == "free flow, increasing"
    assert level_of_service_name(45) == "stationary, decreasing strongly"
    assert component_name("tec", 0) == ("TEC message", "tec_message")
    assert component_name("location", 2) == ("TMC location", None)
    assert component_name("event", 2) == ("Component 2", None)
