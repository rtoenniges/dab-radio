"""The sub-page "Recording" of the touch settings (`ui.touch.recording_view`): the switch,
the size of the ring, and the start and stop delays, per station with a default."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication, QWidget

from tests.fake_receiver import SERVICES_5C
from ui.touch.recording_view import RecordingView, delay_scope_text, start_delay_text
from ui.touch.settings_page import LABEL_WIDTH
from ui.touch.window import SCREEN_HEIGHT, SCREEN_WIDTH


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def view(make_controller, data_directory):
    window = QWidget()
    window.resize(SCREEN_WIDTH, SCREEN_HEIGHT)
    window.show()
    controller = make_controller(data_directory)
    view = RecordingView(controller, LABEL_WIDTH, window)
    view.resize(SCREEN_WIDTH - 200, 520)
    view.show()
    yield view, controller
    window.deleteLater()


def test_the_texts_say_when_the_file_begins_and_whom_the_delay_applies_to():
    assert start_delay_text(0) == "At Dyn. Label"
    assert start_delay_text(30) == "30 s before Dyn. Label"
    assert start_delay_text(-10) == "10 s after Dyn. Label"
    assert delay_scope_text(None, False) == "Default for all stations"
    assert delay_scope_text("Dlf", False) == "Dlf, default"
    assert delay_scope_text("Dlf", True) == "Dlf, own value"


def test_the_switch_turns_the_recording_off_and_on(view):
    view, controller = view
    assert view._switch.isChecked() and view._switch.text() == "On"

    view._switch.click()
    assert not controller.recording_enabled and view._switch.text() == "Off"
    controller.set_recording_enabled(True)
    assert view._switch.isChecked() and view._switch.text() == "On"


def test_the_ring_size_is_set_with_a_stepper_and_the_button_leads_to_the_recordings(view):
    view, controller = view
    asked: list[bool] = []
    view.recordings_requested.connect(lambda: asked.append(True))
    assert view._ring_size.value() == controller.ring_size == 10

    view._ring_size._plus.click()
    assert controller.ring_size == 11
    view._show_recordings.click()
    assert asked == [True]


def test_the_start_delay_is_set_for_the_station_playing_with_a_default_for_all(view):
    view, controller = view
    start = view._start
    assert start.stepper.value() == controller.label_delay_s == 0
    # No station selected: the stepper sets the default of every station.
    assert start.scope.text() == "Default for all stations"
    assert not start.default_button.isVisibleTo(view)
    start.stepper._plus.click()
    assert controller.default_label_delay_s == 5 and controller.label_delay_s == 5

    # A station selected: its own value from the first step on.
    controller.select_service(SERVICES_5C[0][0])
    assert start.scope.text() == "Dlf, default" and start.stepper.value() == 5
    assert start.default_button.isVisibleTo(view)
    start.stepper._plus.click()
    assert controller.label_delay_s == 10 and controller.default_label_delay_s == 5
    assert start.scope.text() == "Dlf, own value"
    controller.select_service(SERVICES_5C[1][0])
    assert start.stepper.value() == 5 and start.scope.text() == "Dlf Kultur, default"

    start.stepper._minus.click()
    start.stepper._minus.click()
    assert controller.label_delay_s == -5
    start.default_button.click()
    assert controller.default_label_delay_s == -5
    controller.select_service(SERVICES_5C[0][0])
    assert controller.label_delay_s == 10  # its own value stands


def test_the_stop_delay_is_set_the_same_way(view):
    view, controller = view
    stop = view._stop
    assert stop.stepper.value() == controller.stop_delay_s == 0
    assert stop.scope.text() == "Default for all stations"
    stop.stepper._minus.click()
    assert controller.default_stop_delay_s == -5 and controller.label_delay_s == 0

    controller.select_service(SERVICES_5C[0][0])
    assert stop.scope.text() == "Dlf, default" and view._start.scope.text() == "Dlf, default"
    stop.stepper._plus.click()
    stop.stepper._plus.click()
    assert controller.stop_delay_s == 5 and controller.stop_delay_is_station_own
    assert stop.scope.text() == "Dlf, own value" and view._start.scope.text() == "Dlf, default"
    stop.default_button.click()
    assert controller.default_stop_delay_s == 5
    controller.select_service(SERVICES_5C[1][0])
    assert stop.stepper.value() == 5 and stop.scope.text() == "Dlf Kultur, default"
