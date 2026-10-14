"""The "Traffic" tab with real ARD TPEG messages through the `TrafficModel`: list, map,
profile, details and structure."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from dab.data_group import parse_data_group
from dab.gnss import Observer
from dab.tpeg import AID_TEC, AID_TFP, TpegDecoder
from tests.test_tpeg import _application_frame, _group, _service_frame
from tests.tpeg_vectors import (
    SNI_GROUP,
    TEC_CANCEL,
    TEC_LANE_CLOSED,
    TEC_ROADWORKS,
    TEC_TRAFFIC_LIGHTS,
    TFP_FREE,
)
from ui.traffic_model import TrafficModel
from ui.traffic_tab import TrafficTab

# 28.09.2026 19:35 UTC, a minute after the recording: every message is still valid.
NOW = 1790624100.0
HANOVER = Observer(52.37, 9.73)


def _frames(*service_frames: bytes):
    decoder = TpegDecoder()
    frames = decoder.push(parse_data_group(SNI_GROUP))
    for index, raw in enumerate(service_frames):
        frames += decoder.push(parse_data_group(_group(index, raw)))
    return frames


@pytest.fixture
def tab() -> TrafficTab:
    _app = QApplication.instance() or QApplication([])
    model = TrafficModel(HANOVER)
    model.now = lambda: NOW
    tab = TrafficTab(model)
    frames = _frames(
        _service_frame(
            _application_frame(1, [TEC_LANE_CLOSED, TEC_TRAFFIC_LIGHTS, TEC_ROADWORKS]),
            _application_frame(2, [TFP_FREE]),
        )
    )
    for frame in frames:
        model.add_frame(8, frame)
    tab.render()
    yield tab
    tab.deleteLater()


def _column(table, column: int) -> list[str]:
    return [table.item(row, column).text() for row in range(table.rowCount())]


def test_messages_are_listed_nearest_first(tab):
    table = tab._tec_table
    # Laatzen (Hanover), Bremen, the motorway north of Erfurt.
    assert [key[1] for key in tab._tec_keys] == [1745, 2554, 1371]
    assert _column(table, 0) == [
        "road closed",
        "traffic situation unknown",
        "traffic situation unknown",
    ]
    assert _column(table, 1) == ["roadworks", "traffic light failure", "roadworks"]
    assert _column(table, 2)[2] == "motorway"
    assert _column(table, 3)[0] == "8 km SE"
    assert "3 of 655 traffic messages" in tab._status_label.text()
    assert "1 of 730 flow messages" in tab._status_label.text()


def test_selected_message_in_words_and_structure(tab):
    tab._select_row(AID_TEC, 2)
    details = tab._details.text()
    assert "Cause: roadworks, danger level 1, left lane closed" in details
    assert "Location (TMC): location 45524 of table D/1 (version 22.0)" in details
    assert "hazard 10.1 km before the location" in details
    assert "motorway (road class FRC 0)" in details
    top = tab._structure.topLevelItem(0)
    assert top.text(0) == "Transport frame"
    service = top.child(0)
    assert service.text(0) == "Service frame 22.98.211"
    assert [service.child(i).text(0) for i in range(service.childCount())] == [
        "Component frame SCID 1: TEC (traffic events)",
        "Component frame SCID 2: TFP (traffic flow)",
    ]
    message = service.child(0).child(0)
    assert message.text(0) == "TEC message (ID 0)"
    assert message.text(1) == str(len(TEC_LANE_CLOSED))
    names = [message.child(i).text(0) for i in range(message.childCount())]
    assert names == [
        "Message management (MMC) (ID 1)",
        "Event (ID 3)",
        "Location referencing (LRC) (ID 2)",
    ]
    location = message.child(2)
    assert [location.child(i).text(0) for i in range(location.childCount())] == [
        "TMC location (ID 2)",
        "OpenLR (ID 8)",
        "Method 254 (not public) (ID 254)",
    ]


def test_map_draws_the_openlr_lines(tab):
    closures, _stationary, _slow, hints = tab._markers
    assert len(closures.data) == 1
    assert len(hints.data) == 2
    tab._select_row(AID_TEC, 0)
    xs, _ys = tab._selected_line.getData()
    assert list(xs) == [9.79473, 9.79868]


def test_flow_profile(tab):
    tab._views.setCurrentIndex(1)
    assert _column(tab._tfp_table, 0) == ["free flow"]
    assert _column(tab._tfp_table, 2) == ["TMC 21731 (+7)"]
    tab._select_row(AID_TFP, 0)
    assert "Sections (now, positions in TMC locations): +7: free flow" in tab._details.text()
    assert len(tab._profile_points.data) == 1


def test_cancellation_and_expiry_remove_messages(tab):
    cancel = bytearray(TEC_CANCEL)
    cancel[6:8] = TEC_ROADWORKS[6:8]  # the message ID of the roadworks (1745)
    for frame in _frames(_service_frame(_application_frame(1, [bytes(cancel)]))):
        tab._model.add_frame(8, frame)
    tab._drawn_at = None
    tab.render()
    assert [key[1] for key in tab._tec_keys] == [2554, 1371]
    tab._model.now = lambda: 1790631844.5  # after the expiry of the TFP message and of 1371
    tab._drawn_at = None
    tab.render()
    assert [key[1] for key in tab._tec_keys] == [2554]
    assert tab._tfp_keys == []


def test_reset_forgets_everything(tab):
    tab._model.reset()
    tab._drawn_at = None
    tab.render()
    assert tab._tec_table.rowCount() == 0
    assert "No TPEG in the ensemble" in tab._status_label.text()


def test_the_location_table_names_the_stretches(tab, tmp_path):
    from tests import tmc_table

    tab._model.set_locations(tmc_table.table(tmp_path))
    tab._drawn_at = None
    tab.render()
    row = tab._tec_keys.index((1, 1371))
    assert tab._tec_table.item(row, 2).text() == "A 71"
    assert (
        tab._tec_table.item(row, 5).text()
        == "A 71 between Testheim-Süd junction and Testkreuz Nord junction"
    )
    assert _column(tab._tfp_table, 2) == ["B 213 between Teststadt 7 and Teststadt 0 junction"]
    tab._views.setCurrentIndex(1)
    tab._select_row(AID_TFP, 0)
    ticks = tab._profile.getAxis("bottom")._tickLevels
    assert ticks[0] == [(0, "Teststadt 0 junction"), (3, "Teststadt 3"), (7, "Teststadt 7")]
