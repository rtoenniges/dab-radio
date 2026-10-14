"""Tests for the equalizer of the touch surface (`ui.touch.equalizer_view`): a finger on a
point drags the band and turns no page, a finger beside the points does; the width, the
steppers, the presets with the settings made by hand, the line in the header and the
spectrum while the page is seen."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import QApplication, QPushButton, QWidget

from audio.equalizer import (
    BASS,
    HIGH_SHELF,
    LINEAR,
    PEAK,
    SPEECH,
    Band,
    EqualizerSettings,
)
from tests.fake_receiver import SERVICES_5C
from ui.touch.equalizer_view import (
    CUSTOM,
    FREQUENCIES_HZ,
    EqualizerPage,
    frequency_text,
    gain_text,
    summary_text,
)

PAGE_WIDTH, PAGE_HEIGHT = 1168, 720


class _Recorder:
    def __init__(self, signal) -> None:
        self.calls: list[tuple] = []
        signal.connect(lambda *args: self.calls.append(args))


@pytest.fixture
def page(data_directory, make_controller):
    controller = make_controller(data_directory, start=False)
    page = EqualizerPage(controller)
    page.resize(PAGE_WIDTH, PAGE_HEIGHT)
    page.show()
    QApplication.processEvents()
    yield page, controller
    page.hide()
    page.deleteLater()


def _mouse(widget: QWidget, kind: QEvent.Type, point: QPointF, held: bool, origin: QPointF) -> None:
    """A mouse event at `point` (on `widget`), as the pager sees a finger on a PC;
    `origin` is where the widget stood on the screen when the drag began (a page it
    turns moves the widget, not the finger)."""
    moving = kind == QEvent.Type.MouseMove
    button = Qt.MouseButton.NoButton if moving else Qt.MouseButton.LeftButton
    buttons = Qt.MouseButton.LeftButton if held else Qt.MouseButton.NoButton
    event = QMouseEvent(
        kind,
        point,
        origin + point,
        button,
        buttons,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(widget, event)


def _drag(widget: QWidget, start: QPointF, end: QPointF) -> None:
    origin = QPointF(widget.mapToGlobal(QPoint(0, 0)))
    _mouse(widget, QEvent.Type.MouseButtonPress, start, True, origin)
    for step in range(1, 11):
        _mouse(widget, QEvent.Type.MouseMove, start + (end - start) * (step / 10), True, origin)
    _mouse(widget, QEvent.Type.MouseButtonRelease, end, False, origin)


def test_a_finger_on_a_point_drags_the_band_and_beside_the_points_turns_the_page(page):
    page, controller = page
    curve = page.curve
    start = curve.handle_centre(2)  # Mids: 1 kHz, 0 dB
    end = QPointF(curve.x_of(2000.0), curve.y_of(4.0))

    _drag(curve, start, end)

    assert controller.equalizer.bands[2] == Band(PEAK, 2000.0, 4.0, 1.4)
    assert controller.equalizer.preset is None
    assert page.selected_band() == 2
    assert page.pager.current_index() == 0

    beside = QPointF(curve.x_of(8000.0), curve.y_of(-10.0))
    _drag(curve, beside, beside - QPointF(700, 0))

    assert page.pager.current_index() == 1
    assert controller.equalizer.bands[2] == Band(PEAK, 2000.0, 4.0, 1.4)


def test_a_dragged_band_keeps_to_its_steps_and_ranges(page):
    page, controller = page
    curve = page.curve
    start = curve.handle_centre(0)

    _drag(curve, start, QPointF(curve.x_of(71.0), curve.y_of(30.0)))

    band = controller.equalizer.bands[0]
    assert band.frequency_hz in FREQUENCIES_HZ
    assert band.gain_db == 12.0


def test_the_wheel_or_two_fingers_set_the_width_of_a_peaking_band_but_not_of_a_shelf(page):
    page, controller = page
    curve = page.curve
    page._choose_band(3)  # Presence
    centre = curve.handle_centre(3)

    def wheel(steps: int) -> None:
        event = QWheelEvent(
            centre,
            curve.mapToGlobal(centre),
            QPoint(0, 0),
            QPoint(0, 120 * steps),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        QApplication.sendEvent(curve, event)

    wheel(3)  # 1.4 octaves times 1.1 three times
    assert controller.equalizer.bands[3].width_octaves == 1.9
    curve.set_width(10.0)
    assert controller.equalizer.bands[3].width_octaves == 4.0

    page._choose_band(4)  # Brillanz, a shelf
    before = controller.equalizer
    wheel(-2)
    assert controller.equalizer == before


def test_the_fine_page_steps_the_chosen_band_and_a_shelf_has_no_width(page):
    page, controller = page
    fine = page.fine
    chip = next(b for b in fine.bands.findChildren(QPushButton) if b.text() == "Presence")

    chip.click()
    assert page.selected_band() == 3
    assert page.curve.selected() == 3
    fine.gain_stepper._plus.click()
    fine.frequency_stepper._plus.click()

    assert controller.equalizer.bands[3] == Band(PEAK, 4250.0, 0.5, 1.4)
    assert fine.range_label.text() == "2.6 kHz to 6.9 kHz"
    assert fine.width_stepper.value() == 1.4

    page._choose_band(4)
    assert fine.width_stepper.value() is None  # "–"
    assert not fine.width_stepper._plus.isEnabled()
    assert not fine.width_stepper._minus.isEnabled()
    assert fine.kind_label.text() == "High shelf"
    assert controller.equalizer.bands[4].kind == HIGH_SHELF


def test_presets_and_the_settings_made_by_hand_come_back(page):
    page, controller = page
    fine = page.fine
    assert fine.preset_button(LINEAR).isChecked()
    assert not fine.preset_button(CUSTOM).isEnabled()  # nothing made by hand yet

    fine.preset_button(SPEECH).click()
    assert controller.equalizer == EqualizerSettings.from_preset(SPEECH)
    fine.gain_stepper._plus.click()  # Lows: -6.0 to -5.5 dB
    own = controller.equalizer
    assert own.preset is None
    assert fine.preset_button(CUSTOM).isChecked()

    fine.preset_button(BASS).click()
    assert controller.equalizer == EqualizerSettings.from_preset(BASS)
    assert fine.preset_button(CUSTOM).isEnabled()
    fine.preset_button(CUSTOM).click()
    assert controller.equalizer == own


def test_the_header_says_what_is_set_for_whom_and_switches(page):
    page, controller = page
    assert page.summary.text() == "Flat · for all stations"
    assert page.switch.text() == "Off"

    page.switch.click()
    controller.set_equalizer(EqualizerSettings.from_preset(BASS))
    assert controller.equalizer_enabled
    assert page.switch.text() == "On"
    assert page.summary.text() == "Bass · for all stations · Level −6 dB"

    controller.set_equalizer_per_station(True)
    controller.select_service(SERVICES_5C[1][0])
    assert page.summary.text() == "Bass · like all stations · Level −6 dB"
    page.switch.click()  # off for this station only
    assert page.summary.text() == "Bass · for this station"
    assert page.switch.text() == "Off"


def test_the_spectrum_is_asked_for_only_while_the_page_is_seen(page):
    page, controller = page
    asked = _Recorder(controller._audio_spectrum_toggled)
    page.hide()
    page.show()
    centres = 20.0 * 2.0 ** (np.arange(60) / 6)
    controller.audio_player.spectrum_ready.emit(centres, np.full(60, -30.0))
    QApplication.processEvents()
    assert page.curve._spectrum is not None

    page.hide()

    assert asked.calls == [(False,), (True,), (False,)]
    assert page.curve._spectrum is None


@pytest.mark.parametrize(
    ("hz", "text"),
    [
        (20.0, "20 Hz"),
        (31.5, "31.5 Hz"),
        (1060.0, "1.06 kHz"),
        (4000.0, "4.0 kHz"),
        (12500.0, "12.5 kHz"),
    ],
)
def test_frequencies_read_as_on_a_mixer(hz, text):
    assert frequency_text(hz) == text


def test_gains_have_a_sign_and_a_true_minus():
    assert [gain_text(db) for db in (-3.5, 0.0, 2.0)] == ["−3.5 dB", "0.0 dB", "+2.0 dB"]


def test_the_summary_leaves_out_the_level_while_off():
    from audio.equalizer import EqualizerState

    state = EqualizerState(False, EqualizerSettings.from_preset(BASS))
    assert summary_text(state, per_station=False, own=False) == "Bass · for all stations"
