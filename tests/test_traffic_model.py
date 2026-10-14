"""The `TrafficModel` with real ARD TPEG messages (`tests.tpeg_vectors`): the messages by
distance with what DAB brings, named and placed by a (made-up) TMC location table, the
traffic flow on its stretches, the counts of the service, and how it follows the
controller."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import dataclasses

import pytest
from PyQt6.QtWidgets import QApplication

from dab.gnss import Observer
from dab.tmc_locations import TMC_LOCATIONS_FILENAME
from dab.tpeg import AID_TEC, FRAME_TYPE_SERVICE
from dab.tpeg_messages import FlowSection, FlowVector, TecMessage, TfpMessage
from tests import tmc_table
from tests.test_tpeg import _application_frame, _service_frame, _transport_frame
from tests.test_traffic_tab import HANOVER, NOW, _frames
from tests.tpeg_vectors import TEC_LANE_CLOSED, TEC_ROADWORKS, TEC_TRAFFIC_LIGHTS, TFP_FREE
from ui.traffic_model import CYCLE_SECONDS, SILENT_SECONDS, TrafficModel, flow_stretches

APP = QApplication.instance() or QApplication([])
BREMEN = Observer(53.08, 8.80)


def _model(locations=None) -> TrafficModel:
    model = TrafficModel(HANOVER)
    model.now = lambda: NOW
    if locations is not None:
        model.set_locations(locations)
    frames = _frames(
        _service_frame(
            _application_frame(1, [TEC_LANE_CLOSED, TEC_TRAFFIC_LIGHTS, TEC_ROADWORKS]),
            _application_frame(2, [TFP_FREE]),
        )
    )
    for frame in frames:
        model.add_frame(8, frame)
    return model


def _keys(model: TrafficModel) -> set[int]:
    return {e.key[1] for e in model.entries()}


def _at(model: TrafficModel, now: float, *frames: bytes) -> None:
    """Service frames arriving at `now`."""
    model.now = lambda: now
    for frame in _frames(*frames):
        model.add_frame(8, frame)


def _add(model: TrafficModel, message) -> None:
    """Another message, as if it came with the frame of the others."""
    frame = next(iter(model._store.messages())).frame
    model._store.add(message, frame, NOW)
    model._changed()


def test_without_the_location_table_only_what_dab_brings() -> None:
    model = _model()
    entries = model.entries()
    assert [e.key[1] for e in entries] == [1745, 2554, 1371]
    assert all(e.where() == "" for e in entries)
    assert entries[2].road_kind() == "motorway"
    assert entries[0].headline() == "Road closed · roadworks"
    assert entries[0].distance_text() == "8 km SE"
    # The traffic flow has only TMC: no place without the table.
    assert model.flows() == []


def test_the_location_table_names_road_and_junctions(tmp_path) -> None:
    model = _model(tmc_table.table(tmp_path))
    by_id = {e.key[1]: e for e in model.entries()}
    assert by_id[1371].where() == "A 71 between Testheim-Süd junction and Testkreuz Nord junction"
    assert by_id[1371].short_where() == "A 71 at Testkreuz Nord junction"
    assert by_id[2554].where() == "L 123 at Am Testwall"
    assert by_id[1745].where() == ""  # OpenLR only


def test_a_message_with_tmc_alone_gets_its_place_from_the_table(tmp_path) -> None:
    model = _model(tmc_table.table(tmp_path))
    original = next(
        s.message for s in model.stored(TecMessage) if s.message.management.message_id == 1371
    )
    location = dataclasses.replace(original.location, openlr=None)
    management = dataclasses.replace(original.management, message_id=9999)
    _add(model, dataclasses.replace(original, location=location, management=management))
    entry = next(e for e in model.entries() if e.key[1] == 9999)
    # In the direction of traffic: from the far end towards the primary point.
    assert [p for point in entry.path for p in point] == pytest.approx(
        [11.13, 51.114, 11.066, 51.041]
    )
    assert entry.distance is not None
    assert entry.road_kind() == ""


def test_the_flow_on_the_points_of_its_stretch(tmp_path) -> None:
    model = _model(tmc_table.table(tmp_path))
    (flow,) = model.flows()
    assert flow.where() == "B 213 between Teststadt 7 and Teststadt 0 junction"
    assert len(flow.points) == 8
    assert [(s.start, s.end, s.level) for s in flow.stretches] == [(0, 7, 1)]
    assert flow.worst == 1
    assert flow.path()[0] == pytest.approx((8.59 - 7 * 0.06, 53.04 - 7 * 0.02))

    # A queue at the head: the worst first.
    original = next(s.message for s in model.stored(TfpMessage))
    vector = FlowVector(0, (FlowSection(2, 5, 12), FlowSection(5, 3, 35), FlowSection(9, 1, 90)))
    management = dataclasses.replace(original.management, message_id=7777)
    _add(model, dataclasses.replace(original, vectors=(vector,), management=management))
    flows = model.flows()
    assert [f.key[1] for f in flows] == [7777, original.management.message_id]
    queue = flows[0]
    assert [(s.start, s.end, s.level, s.speed) for s in queue.stretches] == [
        (0, 2, 5, 12),
        (2, 5, 3, 35),
        (5, 7, 1, 90),
    ]
    assert queue.speeds() == "12–90 km/h"


def test_sections_end_at_their_offset() -> None:
    message = TfpMessage(
        scid=2,
        management=None,
        component=None,
        spatial_resolution=0,
        vectors=(FlowVector(0, (FlowSection(3, 2, 50), FlowSection(1, 1, 90))),),
    )
    assert [(s.start, s.end) for s in flow_stretches(message, 10)] == [(0, 1), (1, 3)]
    # Another resolution than TMC locations is not placed.
    metres = dataclasses.replace(message, spatial_resolution=1)
    assert flow_stretches(metres, 10) == ()


def test_a_new_position_sorts_anew() -> None:
    model = _model()
    model.set_observer(BREMEN)
    assert [e.key[1] for e in model.entries()] == [2554, 1745, 1371]


def test_counts_and_reset() -> None:
    model = _model()
    assert model.counts(AID_TEC) == (3, 655)
    model.reset()
    assert model.entries() == []
    assert model.counts(AID_TEC) == (0, None)
    assert model.service is None


def test_messages_stay_through_a_channel_change() -> None:
    model = _model()
    model.update_components([])
    model.leave()
    assert model.tpeg is None and model.decoding is None
    assert _keys(model) == {1745, 2554, 1371}
    assert model.counts(AID_TEC) == (3, 655)
    assert model.held_since() == NOW
    assert model.service is not None


def test_what_the_service_no_longer_repeats_goes_after_one_cycle() -> None:
    model = _model()
    model.leave()
    # Back: the lane closure and the roadworks come again. The traffic lights wait for
    # their turn ...
    back = NOW + 60.0
    _at(model, back, _service_frame(_application_frame(1, [TEC_LANE_CLOSED])))
    _at(model, back + 10.0, _service_frame(_application_frame(1, [TEC_ROADWORKS])))
    assert _keys(model) == {1745, 2554, 1371}
    assert model.held_since() is None
    model.expire(back + CYCLE_SECONDS - 1.0)
    assert _keys(model) == {1745, 2554, 1371}
    # ... until the carousel has been round once: then they were withdrawn meanwhile.
    model.expire(back + CYCLE_SECONDS)
    assert _keys(model) == {1371, 1745}
    # The check runs once per return; what came since stays.
    model.expire(back + 2 * CYCLE_SECONDS)
    assert _keys(model) == {1371, 1745}


def test_silence_counts_as_an_absence() -> None:
    model = _model()
    later = NOW + SILENT_SECONDS + 1.0
    _at(model, later, _service_frame(_application_frame(1, [TEC_TRAFFIC_LIGHTS])))
    assert model.held_since() is None
    _at(model, later + 10.0, _service_frame(_application_frame(1, [TEC_TRAFFIC_LIGHTS])))
    model.expire(later + CYCLE_SECONDS)
    assert _keys(model) == {2554}


def test_another_tpeg_service_starts_afresh() -> None:
    model = _model()
    other = _transport_frame(
        FRAME_TYPE_SERVICE, bytes((22, 98, 1, 0)) + _application_frame(1, [TEC_ROADWORKS])
    )
    _at(model, NOW + 1.0, other)
    assert _keys(model) == {1745}
    # Its carousel has nothing to confirm: the message stays after a cycle.
    model.expire(NOW + 1.0 + CYCLE_SECONDS)
    assert _keys(model) == {1745}


def test_it_reads_the_table_of_the_controllers_data_directory(
    data_directory, make_controller, tmp_path_factory
) -> None:
    table = tmc_table.table(tmp_path_factory.mktemp("lcl"))
    table.save(data_directory / TMC_LOCATIONS_FILENAME)
    controller = make_controller(data_directory, start=False)
    model = TrafficModel.for_controller(controller)
    assert model.locations is not None
    assert model.roads is None  # not built here
    controller.set_observer(BREMEN)
    assert model.observer == BREMEN
    model.now = lambda: NOW
    for frame in _frames(_service_frame(_application_frame(1, [TEC_ROADWORKS]))):
        model.add_frame(8, frame)
    controller.live_reset.emit()
    assert _keys(model) == {1745}
