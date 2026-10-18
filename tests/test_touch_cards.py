"""The rows on cards of the touch settings (`ui.touch.cards`), the sheet with the
choices (`ui.touch.overlay.ChoiceSheet`), the list of categories with its states and
the texts of those states (`ui.touch.settings_page`)."""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget

from net.wifi import BAND_5_GHZ
from ui.receiver_controller import DeviceSettings
from ui.touch import cards
from ui.touch.cards import (
    ROW_MARGIN,
    Card,
    ChoiceRow,
    Row,
    Toggle,
    action_row,
    switch_row,
    value_row,
)
from ui.touch.settings_page import (
    AUDIO,
    DISPLAY,
    SDR,
    CategoryList,
    bluetooth_state,
    mqtt_state,
    recording_state,
    sdr_state,
    wifi_state,
)
from ui.touch.theme import DARK, LIGHT, THEMES, TOUCH_TARGET, stylesheet


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window():
    window = QWidget()
    window.setStyleSheet(stylesheet(THEMES[DARK]))
    window.resize(1280, 720)
    window.show()
    yield window
    window.deleteLater()


def _shown(window: QWidget, *widgets: QWidget) -> None:
    layout = QVBoxLayout(window)
    for widget in widgets:
        layout.addWidget(widget)
    layout.addStretch()
    QApplication.processEvents()


# --- Rows -------------------------------------------------------------------------------


def test_a_toggle_is_a_switch_that_is_tall_enough_for_a_finger(window):
    toggle = Toggle()
    _shown(window, toggle)
    seen: list[bool] = []
    toggle.clicked.connect(seen.append)
    assert not toggle.isChecked() and toggle.height() >= 56 and toggle.width() >= TOUCH_TARGET

    toggle.click()
    assert toggle.isChecked() and seen == [True]
    QTest.mouseClick(toggle, Qt.MouseButton.LeftButton)
    assert not toggle.isChecked() and seen == [True, False]


def test_a_row_shows_title_subtitle_and_value_and_reports_a_tap(window):
    row = value_row("Server", "pi", "The rtl_tcp server.")
    _shown(window, row)
    taps: list[bool] = []
    row.tapped.connect(lambda: taps.append(True))
    assert row.title.text() == "Server" and row.text() == "pi"
    assert row.subtitle.isVisible() and row.subtitle.text() == "The rtl_tcp server."
    assert row.height() >= TOUCH_TARGET

    row.setText("radio-pi")
    assert row.text() == "radio-pi"
    row.set_subtitle("")
    assert not row.subtitle.isVisible()
    row.set_placeholder(True)
    assert row.value.property("role") == "muted"
    row.set_placeholder(False)
    assert row.value.property("role") == ""

    row.click()
    QTest.mouseClick(row, Qt.MouseButton.LeftButton, pos=QPoint(40, row.height() // 2))
    assert taps == [True, True]
    # A row without a value answers with its title.
    assert Row("Only a title").text() == "Only a title"


def test_a_tap_on_the_text_of_a_switch_row_flips_the_switch(window):
    toggle = Toggle()
    row = switch_row("AGC", toggle, "The stick sets its gain itself.")
    _shown(window, row)

    QTest.mouseClick(row, Qt.MouseButton.LeftButton, pos=QPoint(40, row.height() // 2))
    assert toggle.isChecked()
    QTest.mouseClick(toggle, Qt.MouseButton.LeftButton)
    assert not toggle.isChecked()


def test_an_action_row_colours_its_title(window):
    leads_on = action_row("Adjust sound")
    deletes = action_row("Delete all sound settings", danger=True)
    _shown(window, leads_on, deletes)
    assert leads_on.title.property("role") == "accent"
    assert deletes.title.property("role") == "danger"


def test_a_disabled_row_dims_its_texts(window):
    row = value_row("Device", "RTL-SDR")
    _shown(window, row)
    assert not row.property("dimmed")
    row.setEnabled(False)
    assert row.property("dimmed") is True
    row.setEnabled(True)
    assert row.property("dimmed") is False


# --- Cards ------------------------------------------------------------------------------


def _divider_count(card: Card) -> int:
    """How many lines the panel draws between its rows: the pixels in a column at the
    right end of the lines that are not the panel's colour (the rows there are empty)."""
    panel = card.panel
    image = panel.grab().toImage()
    surface = THEMES[cards.theme().key].color("surface")
    x = panel.width() - ROW_MARGIN - 2
    lines = 0
    for y in range(panel.height()):
        if QColor(image.pixel(x, y)) != surface:
            lines += 1
    return lines


def test_a_card_draws_a_line_between_the_rows_shown(window):
    rows = [Row("One"), Row("Two"), Row("Three")]
    card = Card("Heading", rows)
    _shown(window, card)
    assert card.heading.text() == "Heading" and card.rows() == rows
    assert _divider_count(card) == 2

    rows[1].hide()
    QApplication.processEvents()
    assert _divider_count(card) == 1
    rows[0].hide()
    QApplication.processEvents()
    assert _divider_count(card) == 0
    # The lines take the colours of the scheme set for the cards.
    cards.set_theme(THEMES[LIGHT])
    try:
        rows[0].show()
        rows[1].show()
        QApplication.processEvents()
        window.setStyleSheet(stylesheet(THEMES[LIGHT]))
        assert _divider_count(card) == 2
    finally:
        cards.set_theme(THEMES[DARK])


def test_a_card_without_a_heading_hides_the_label(window):
    card = Card("", [Row("One")])
    _shown(window, card)
    assert card.heading.isHidden()


# --- The sheet with the choices ----------------------------------------------------------


def test_a_choice_row_shows_the_choice_and_opens_the_sheet(window):
    row = ChoiceRow(
        "Device",
        [("rtlsdr", "RTL-SDR"), ("rspduo", "RSPduo"), ("rtltcp", "RTL_TCP")],
        subtitle="A change restarts the reception.",
        hint="The stick, or a stick elsewhere.",
    )
    _shown(window, row)
    chosen: list[object] = []
    row.chosen.connect(chosen.append)
    row.set_current("rspduo")
    assert row.text() == "RSPduo" and row.current() == "rspduo"
    assert chosen == []

    row.click()
    sheet = row.sheet()
    assert sheet.isVisible()
    assert sheet.title.text() == "Device"
    assert sheet.hint.text() == "The stick, or a stick elsewhere."
    assert [r.name.text() for r in sheet.rows] == ["RTL-SDR", "RSPduo", "RTL_TCP"]
    assert [bool(r.property("selected")) for r in sheet.rows] == [False, True, False]

    sheet.rows[2].click()
    assert not sheet.isVisible()
    assert chosen == ["rtltcp"] and row.text() == "RTL_TCP"

    # "Cancel", and a tap beside the panel, choose nothing.
    row.click()
    sheet._cancel.click()
    assert not sheet.isVisible() and chosen == ["rtltcp"]
    row.click()
    QTest.mouseClick(sheet, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    assert not sheet.isVisible() and chosen == ["rtltcp"]


def test_the_choices_of_a_row_can_change_with_details(window):
    row = ChoiceRow("Output", [("", "Automatic")])
    _shown(window, row)
    row.set_current("")
    row.set_choices([("", "Automatic"), ("bt", "WH-1000XM4")], {"bt": "Bluetooth"})
    assert row.choices() == [("", "Automatic"), ("bt", "WH-1000XM4")]
    assert row.text() == "Automatic"

    row.open()
    sheet = row.sheet()
    assert sheet.rows[1].detail.text() == "Bluetooth" and sheet.rows[0].detail.isHidden()
    sheet.dismiss()
    # A choice that is gone leaves the value empty, as nothing is in effect.
    row.set_choices([("", "Automatic")])
    row.set_current("bt")
    assert row.text() == ""


# --- The list of categories -------------------------------------------------------------


def test_the_list_marks_the_category_chosen_and_shows_its_state(window):
    categories = (
        (SDR, "SDR", "antenna"),
        (DISPLAY, "Display", "screen"),
        (AUDIO, "Audio", "loudspeaker"),
    )
    lst = CategoryList(categories)
    _shown(window, lst)
    chosen: list[str] = []
    lst.chosen.connect(chosen.append)

    lst.set_current(DISPLAY)
    assert lst.current() == DISPLAY
    assert [bool(r.property("selected")) for r in lst.rows.values()] == [False, True, False]
    lst.set_state(SDR, "RTL-SDR")
    assert lst.rows[SDR].state.isVisible() and lst.rows[SDR].state.text() == "RTL-SDR"
    assert not lst.rows[DISPLAY].state.isVisible()
    lst.set_state(SDR, "")
    assert not lst.rows[SDR].state.isVisible()

    lst.rows[AUDIO].click()
    assert chosen == [AUDIO]
    assert lst.current() == DISPLAY  # marked by the page, which shows the sub-page


def test_the_states_name_what_each_category_stands_at(make_controller, data_directory):
    assert sdr_state(DeviceSettings(device="rtlsdr")) == "RTL-SDR"
    assert sdr_state(DeviceSettings(device="rspduo", rspduo_tuner=2)) == "RSPduo · Tuner 2"
    assert sdr_state(DeviceSettings(device="rtltcp", rtl_tcp_host="pi")) == "RTL_TCP · pi"
    assert sdr_state(DeviceSettings(device="rtltcp", rtl_tcp_host="")) == "RTL_TCP · Not set"

    on = SimpleNamespace(
        available=True,
        activity="",
        connected_ssid="Home",
        connected_band=BAND_5_GHZ,
        device=SimpleNamespace(connecting=False),
        enabled=True,
    )
    assert wifi_state(on) == "Home · 5 GHz"
    assert wifi_state(SimpleNamespace(available=False)) == ""
    off = SimpleNamespace(
        available=True, activity="", connected_ssid="", device=None, enabled=False
    )
    assert wifi_state(off) == "Off"
    busy = SimpleNamespace(available=True, activity="Connecting to “Keller” …")
    assert wifi_state(busy) == "Connecting to “Keller” …"

    jabra = SimpleNamespace(name="Jabra Elite 10", battery=50)
    connected = SimpleNamespace(available=True, activity="", powered=True, connected_device=jabra)
    assert bluetooth_state(connected) == "Jabra Elite 10 · Battery 50 %"
    no_battery = SimpleNamespace(name="JBL Flip", battery=None)
    connected.connected_device = no_battery
    assert bluetooth_state(connected) == "JBL Flip"
    assert bluetooth_state(SimpleNamespace(available=True, activity="", powered=False)) == "Off"
    assert bluetooth_state(SimpleNamespace(available=False)) == ""

    controller = make_controller(data_directory)
    try:
        assert recording_state(controller) == "On · 10 titles"
        controller.set_recording_enabled(False)
        assert recording_state(controller) == "Off"
        assert mqtt_state(controller.mqtt_bridge) == "Off"
        assert mqtt_state(None) == ""
    finally:
        controller.shutdown()
