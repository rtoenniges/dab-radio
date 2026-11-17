"""The emergency warnings on the touch surface: the dialog over the window while one
plays, the symbol in the rail, the card on the settings page "Audio" and the number pad
typing a location code."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from dab.dls import DynamicLabel
from dab.ensemble_cache import SOURCE_SCAN, EnsembleCache, EnsembleInfo
from dab.ews import EwsAlert, EwsState, Stage
from dab.gnss import Observer
from dab.location_code import LocationCode, location_code
from net import wifi
from tests.fake_receiver import SERVICES_5C, SERVICES_11D
from ui import receiver_controller
from ui.touch import backlight
from ui.touch.icons import symbol_pixmap
from ui.touch.number_pad import BACKSPACE, NumberPad
from ui.touch.window import STATUS_ICON, TouchWindow

EID_5C = 0x10BC
ASA_DE = 0x10C4
POSITION = Observer(37.5, 18.0, 0.0)
SERVICES = (*SERVICES_5C, (ASA_DE, "ASA DE"))


def _alert(stage: Stage = Stage.LEVEL_1_START, *, other: bool = False) -> EwsAlert:
    if other:
        return EwsAlert(True, 0x1188, stage, 2, whole_area=True)
    return EwsAlert(
        False, EID_5C, stage, 2, whole_area=True, subchannel_id=9, sid=ASA_DE, receivable=True
    )


def _playing(*alerts: EwsAlert) -> EwsState:
    running = frozenset({9}) if any(not alert.other_ensemble for alert in alerts) else frozenset()
    return EwsState(True, EID_5C, alerts, running, ((9, ASA_DE),) if running else ())


QUIET = EwsState(participating=True, eid=EID_5C)


@pytest.fixture
def window(data_directory, make_controller, fake_workers, monkeypatch):
    """The touch window over a controller playing Dlf Kultur on 5C, which takes part;
    11D is stored as ensemble 1188. The buttons of the warning take a tap at once."""
    _app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(wifi, "find_nmcli", lambda: None)
    monkeypatch.setattr(backlight, "BACKLIGHT_CLASS", Path("nowhere") / "backlight")
    cache = EnsembleCache(data_directory / receiver_controller.ENSEMBLE_CACHE_FILENAME)
    cache.load()
    cache.put("5C", EnsembleInfo("DR Deutschland", SERVICES, 0.0, SOURCE_SCAN, eid=EID_5C))
    cache.put("11D", EnsembleInfo("NDR NDS", SERVICES_11D, 0.0, SOURCE_SCAN, eid=0x1188))
    cache.save()
    controller = make_controller(data_directory, start=False)
    window = TouchWindow(controller)
    controller.start()
    window.show()
    window._alert_dialog._guard_ms = 0
    controller.set_observer(POSITION)
    controller.play_service("5C", SERVICES_5C[1][0])
    worker = fake_workers[-1]
    worker.connected.emit("Connected.")
    dsp = controller.dsp_worker
    dsp.stream_restarted.emit(worker.tuning)
    dsp.services_updated.emit("DR Deutschland", list(SERVICES), {})
    dsp.ews_ready.emit(QUIET)
    yield window, controller
    window.close()


# --- The dialog ----------------------------------------------------------------------------


def test_a_warning_shows_over_the_window_and_wakes_the_screen(window, monkeypatch):
    window, controller = window
    woken = []
    monkeypatch.setattr(window._screen, "touched", lambda: woken.append(True))
    dialog = window._alert_dialog

    controller.dsp_worker.ews_ready.emit(_playing(_alert()))
    controller.dsp_worker.alert_label_ready.emit(DynamicLabel("Probealarm <b>Hafen</b>"))

    assert dialog.isVisible()
    assert dialog.geometry() == window.rect()
    assert dialog.headline.text() == "Warning"
    assert dialog.station.text() == "ASA DE"
    assert dialog.text.text() == "Probealarm <b>Hafen</b>"  # plain text, as sent
    assert woken  # the screen is on for it

    dialog.end_button.click()

    assert controller.alert is None
    assert not dialog.isVisible()


def test_a_warning_without_a_dynamic_label_shows_its_panel(window):
    """As the ASA function test: "ASA DE" sends no dynamic label, the one text in the
    panel that wraps."""
    window, controller = window
    dialog = window._alert_dialog

    controller.dsp_worker.ews_ready.emit(_playing(_alert()))

    panel = dialog.panel.geometry()
    assert dialog.text.isHidden()
    assert panel.height() == dialog.panel.layout().totalSizeHint().height() > 100
    assert dialog.rect().contains(panel)
    assert panel.center().y() == pytest.approx(dialog.rect().center().y(), abs=1)


def test_skipping_the_incident_lets_only_its_critical_warnings_through(window):
    window, controller = window
    dsp = controller.dsp_worker
    dsp.ews_ready.emit(_playing(_alert()))

    window._alert_dialog.incident_button.click()
    dsp.ews_ready.emit(_playing(_alert(Stage.LEVEL_1_UPDATE)))
    assert controller.alert is None
    dsp.ews_ready.emit(_playing(_alert(Stage.LEVEL_1_CRITICAL)))

    assert window._alert_dialog.headline.text() == "Critical warning"


def test_a_warning_of_another_ensemble_is_looked_for_there(window):
    window, controller = window

    controller.dsp_worker.ews_ready.emit(_playing(_alert(other=True)))

    assert window._alert_dialog.station.text() == "Looking for it on channel 11D …"


def test_the_buttons_wait_for_the_pause(window):
    window, controller = window
    window._alert_dialog._guard_ms = 60_000
    controller.dsp_worker.ews_ready.emit(_playing(_alert()))

    window._alert_dialog.end_button.click()

    assert controller.alert is not None


def test_a_warning_stays_above_a_notification(window):
    window, controller = window
    controller.dsp_worker.ews_ready.emit(_playing(_alert()))

    controller.remote_state.execute("notify", {"title": "Front door"})

    children = window.children()
    assert window._notification_dialog.isVisible()
    assert children.index(window._alert_dialog) > children.index(window._notification_dialog)


# --- The rail ------------------------------------------------------------------------------


def _shows(window: TouchWindow, symbol: str, color: str) -> bool:
    icon = window._ews_icon
    expected = symbol_pixmap(
        symbol, window._theme.color(color), STATUS_ICON, window.devicePixelRatioF()
    )
    return icon.isVisible() and icon.pixmap().toImage() == expected.toImage()


def test_the_rail_shows_whether_warnings_reach_the_radio(window):
    window, controller = window
    assert _shows(window, "alert", "text")  # 5C takes part

    controller.dsp_worker.ews_ready.emit(_playing(_alert()))
    assert _shows(window, "alert", "danger")
    controller.dsp_worker.ews_ready.emit(QUIET)

    window._ews_settle_timer.timeout.emit()
    controller.dsp_worker.ews_ready.emit(EwsState(eid=EID_5C))  # no FIG 0/15 for 10 s
    assert _shows(window, "alert_off", "muted")
    controller.set_ews_enabled(False)
    assert not window._ews_icon.isVisible()


def test_after_tuning_the_rail_waits_before_it_says_no_warnings(window):
    window, controller = window
    controller.select_channel("11D")
    worker = controller._worker
    controller.dsp_worker.stream_restarted.emit(worker.tuning)

    assert not window._ews_icon.isVisible()  # 11D has not had its time yet
    window._ews_settle_timer.timeout.emit()
    assert _shows(window, "alert_off", "muted")


# --- The settings --------------------------------------------------------------------------


def test_the_card_shows_the_code_of_the_location(window):
    window, _controller = window
    page = window._settings_page

    assert page._location_code.value.text() == location_code(37.5, 18.0).presentation
    assert page._location_code.subtitle.text() == "From the location under Display"
    assert page._ews_row.subtitle.text() == "The ensemble received takes part."
    assert page._alarm_volume.value() == receiver_controller.DEFAULT_ALARM_VOLUME_PERCENT


def test_a_code_entered_and_the_location_again(window):
    window, controller = window
    page = window._settings_page

    page._edit_location_code()
    pad = page.number_pad()
    for key in "125544671251":
        pad.press(key)
    pad._accept.click()
    assert controller.ews_location == LocationCode(1, "91BB81")
    assert page._location_code.value.text() == "1255-4467-1251"
    assert page._location_code.subtitle.text() == "Entered"

    page._ask_location_code()
    sheet = page.action_sheet()
    assert [button.text() for button in sheet.actions] == ["Use location", "Enter code"]
    sheet.actions[0].click()
    assert not controller.ews_location_entered


def test_switch_and_alarm_volume(window):
    window, controller = window
    page = window._settings_page

    page._ews_switch.click()
    page._alarm_volume._step(2)

    assert not controller.ews_enabled
    assert controller.alarm_volume_percent == receiver_controller.DEFAULT_ALARM_VOLUME_PERCENT + 10


# --- The number pad --------------------------------------------------------------------------


def test_the_number_pad_types_a_location_code_in_groups(window):
    window, _controller = window
    pad = NumberPad(window)
    taken = []
    pad.open_code("Location code", "1255-4467-1251", taken.append, lambda text: True, "12")
    assert not pad._keys["0"].isEnabled() and not pad._keys["9"].isEnabled()

    for key in "1253093513":  # the first replaces; 0 and 9 write nothing
        pad.press(key)
    assert pad.text == "1253-3513"
    pad.press(BACKSPACE)
    for key in "336689":
        pad.press(key)
    assert pad.text == "1253-3513-3668"
    pad.press("1")  # twelve at most
    assert pad.text == "1253-3513-3668"
    pad._accept.click()

    assert taken == ["1253-3513-3668"]
    # Numbers again: every digit writes.
    pad.open("Latitude", 51.0, 4, "°", (-90, 90), lambda _value: None)
    assert pad._keys["0"].isEnabled() and pad._keys["9"].isEnabled()
