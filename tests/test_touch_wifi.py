"""The keyboard (`ui.touch.keyboard`) and the sub-page "Wi-Fi" of the touch settings
(`ui.touch.wifi_view`), on an nmcli that answers from a table."""

from __future__ import annotations

import os
import string

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import QApplication, QWidget

from net import wifi
from net.wifi import (
    BACKGROUND_POLL_INTERVAL_MS,
    BAND_2_4_GHZ,
    BAND_5_GHZ,
    POLL_INTERVAL_MS,
    Ipv4Settings,
    Network,
    SavedNetwork,
    WifiManager,
    ipv4_arguments,
    valid_dns_list,
    valid_ipv4,
)
from tests.fake_nmcli import (
    PERMISSIONS,
    PERMISSIONS_DENIED,
    UUID_HIDDEN,
    UUID_HOME,
    FakeNmcli,
    profile,
    profile_command,
)
from ui.touch.ip_settings import IpSettingsDialog
from ui.touch.keyboard import BACKSPACE, DOT, LAYOUT, MAX_BYTES, SHIFT, Keyboard, Shift
from ui.touch.number_pad import DECIMAL, SIGN, NumberPad
from ui.touch.settings_page import LABEL_WIDTH
from ui.touch.wifi_view import WifiEntry, WifiView, bands_text, wifi_entries
from ui.touch.window import SCREEN_HEIGHT, SCREEN_WIDTH


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window():
    window = QWidget()
    window.resize(SCREEN_WIDTH, SCREEN_HEIGHT)
    window.show()
    yield window
    window.deleteLater()


# --- Keyboard -------------------------------------------------------------------------


@pytest.fixture
def keyboard(window):
    keyboard = Keyboard(window)
    taken = []
    keyboard.open("Password", taken.append, accept_text="Connect", check=lambda t: len(t) >= 3)
    return keyboard, taken


def _type(keyboard: Keyboard, text: str) -> None:
    for character in text:
        keyboard.press(character)


def test_the_keyboard_fits_the_screen(keyboard):
    keyboard, _taken = keyboard
    assert keyboard.panel.width() <= SCREEN_WIDTH
    assert keyboard.panel.height() <= SCREEN_HEIGHT
    assert keyboard.panel.geometry().top() >= 0


def test_shift_holds_for_one_letter_or_until_tapped_again(keyboard):
    keyboard, _taken = keyboard
    assert "q" in keyboard.characters()

    keyboard.press(SHIFT)
    assert "Q" in keyboard.characters() and "1" in keyboard.characters()
    keyboard.press("Q")
    assert keyboard.shift is Shift.OFF
    assert "q" in keyboard.characters()

    keyboard.press(SHIFT)
    keyboard.press(SHIFT)
    _type(keyboard, "AB")
    assert keyboard.shift is Shift.LOCKED
    keyboard.press(SHIFT)
    assert keyboard.shift is Shift.OFF
    assert keyboard.text == "QAB"


def test_letters_and_symbols_hold_every_printable_ascii_character(keyboard):
    keyboard, _taken = keyboard
    reachable = set(keyboard.characters())
    keyboard.press(SHIFT)
    reachable |= set(keyboard.characters())
    keyboard.press(SHIFT)
    keyboard.press(SHIFT)
    keyboard.press(LAYOUT)
    reachable |= set(keyboard.characters())
    reachable.add(" ")  # the space bar

    printable = set(string.ascii_letters + string.digits + string.punctuation + " ")
    assert printable <= reachable
    assert {"ä", "ö", "ü", "Ä", "Ö", "Ü", "ß"} <= reachable


def test_an_ampersand_is_typed_and_shown_as_such(keyboard):
    keyboard, _taken = keyboard
    keyboard.press(LAYOUT)
    key = next(k for k, _row, _index in keyboard._character_keys if keyboard._characters[k] == "&")
    assert key.text() == "&&"  # one "&" would mark a shortcut
    key.click()
    assert keyboard.text == "&"


def test_accept_waits_for_the_check_and_hands_over_the_text(keyboard):
    keyboard, taken = keyboard
    _type(keyboard, "ab")
    assert not keyboard._accept.isEnabled()
    assert keyboard._hint.property("role") == "error"

    _type(keyboard, "c")
    keyboard.press(BACKSPACE)
    keyboard.press("d")
    keyboard._accept.click()

    assert taken == ["abd"]
    assert not keyboard.isVisible()
    assert keyboard.text == ""


def test_no_more_than_the_longest_name_or_key(keyboard):
    keyboard, _taken = keyboard
    _type(keyboard, "x" * (MAX_BYTES + 5))
    assert len(keyboard.text) == MAX_BYTES


def test_a_secret_shows_as_dots_until_the_eye_is_tapped(window):
    keyboard = Keyboard(window)
    keyboard.open("Password", lambda _text: None, secret=True)
    _type(keyboard, "geheim")
    assert keyboard._field.text() == DOT * 6

    keyboard._eye.click()
    assert keyboard._field.text() == "geheim"

    keyboard.dismiss()
    keyboard.open("Password", lambda _text: None, secret=True)
    assert not keyboard._eye.isChecked()
    assert keyboard.text == ""


def test_a_real_keyboard_types_too(keyboard):
    keyboard, taken = keyboard
    for text in ("A", "b", "!"):
        keyboard.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier, text)
        )
    keyboard.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
    )
    assert taken == ["Ab!"]


def test_a_tap_beside_the_keyboard_cancels(keyboard):
    keyboard, taken = keyboard
    _type(keyboard, "abc")
    keyboard.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(2, 2),
            QPointF(2, 2),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    assert not keyboard.isVisible()
    assert taken == []


# --- The list of networks -------------------------------------------------------------


def test_connected_first_then_saved_then_the_others_and_saved_out_of_reach_last():
    networks = [
        Network("Nachbar", 90, "WPA2"),
        Network("Heim", 60, "WPA2", in_use=True),
        Network("Gast", 50, ""),
        Network("Büro", 40, "WPA3"),
        Network("Firma", 30, "WPA2 802.1X"),
    ]
    saved = [
        SavedNetwork("1", "preconfigured", "Heim"),
        SavedNetwork("2", "Büro", "Büro"),
        SavedNetwork("3", "Keller", "Keller"),
    ]

    entries = wifi_entries(networks, saved, "Heim")

    assert [e.ssid for e in entries] == ["Heim", "Büro", "Nachbar", "Gast", "Firma", "Keller"]
    assert entries[0] == WifiEntry("Heim", 3, True, True, True)
    assert entries[0].detail == "Connected"
    assert entries[1].detail == "Saved"
    assert not entries[3].locked and entries[3].detail == ""
    assert entries[4].enterprise and entries[4].detail == "WPA-Enterprise"
    assert entries[5] == WifiEntry("Keller", None, False, False, True)
    assert entries[5].detail == "Saved, out of reach"


# --- The sub-page ---------------------------------------------------------------------


@pytest.fixture
def nmcli() -> FakeNmcli:
    return FakeNmcli()


@pytest.fixture
def view(window, nmcli):
    manager = WifiManager(runner=nmcli)
    view = WifiView(manager, LABEL_WIDTH, window)
    view.resize(SCREEN_WIDTH - 200, 520)
    view.show()  # reads everything
    yield view
    manager.shutdown()


def _row(view: WifiView, ssid: str):
    return next(row for row in view.rows() if row.entry.ssid == ssid)


def test_the_sub_page_shows_the_connection_and_the_networks(view):
    assert view._switch.isChecked() and view._switch.text() == "On"
    assert view._connection.text() == "FRITZ!Box 7590 XY"
    assert view._address.text() == "192.168.178.42"
    assert view._band.text() == "2.4 GHz" and not view._band.isHidden()
    assert [e.ssid for e in view.entries()] == [
        "FRITZ!Box 7590 XY",
        "Nachbar:WLAN",
        "Gast",
        "Firma",
        "Keller",
    ]
    assert view.message.text() == ""


def test_the_sub_page_polls_faster_while_shown(view, nmcli):
    timer = view._wifi._poll_timer
    assert timer.isActive() and timer.interval() == POLL_INTERVAL_MS
    view.hide()
    assert not timer.isActive()  # nobody watching in the background

    view._wifi.watch()
    view.show()
    assert timer.interval() == POLL_INTERVAL_MS
    view.hide()
    assert timer.isActive() and timer.interval() == BACKGROUND_POLL_INTERVAL_MS


def test_a_new_network_asks_for_its_password(view, nmcli):
    _row(view, "Nachbar:WLAN").tapped.emit()
    keyboard = view.keyboard()
    assert keyboard.isVisible()
    assert keyboard._title.text() == "Password for “Nachbar:WLAN”"

    _type(keyboard, "geheim1")
    assert not keyboard._accept.isEnabled()  # 7 characters
    keyboard.press("2")
    keyboard._accept.click()

    connect = nmcli.commands_starting("-w", "60", "device", "wifi", "connect")
    assert connect[0][5:8] == ["Nachbar:WLAN", "password", "geheim12"]


def test_an_open_network_is_connected_at_once(view, nmcli):
    _row(view, "Gast").tapped.emit()
    assert not view.keyboard().isVisible()
    assert nmcli.commands_starting("-w", "60", "device", "wifi", "connect")[0][5] == "Gast"


def test_a_saved_network_offers_to_connect_or_forget(view, nmcli):
    _row(view, "Keller").tapped.emit()
    sheet = view.sheet()
    assert sheet.isVisible()
    assert [b.text() for b in sheet.actions] == [
        "Forget",
        "IP settings",
        "Frequency band",
        "Connect",
    ]

    sheet.actions[3].click()

    assert not sheet.isVisible()
    assert ["-w", "30", "connection", "up", "uuid", UUID_HIDDEN] in nmcli.commands


def test_the_network_connected_to_offers_to_disconnect_or_forget(view, nmcli):
    _row(view, "FRITZ!Box 7590 XY").tapped.emit()
    sheet = view.sheet()
    assert sheet.detail.text() == "Connected, 192.168.178.42 by DHCP"
    assert [b.text() for b in sheet.actions] == [
        "Forget",
        "IP settings",
        "Frequency band",
        "Disconnect",
    ]

    sheet.actions[0].click()
    assert ["connection", "delete", "uuid", UUID_HOME] in nmcli.commands


def test_the_network_connected_to_is_disconnected_from_its_sheet(view, nmcli):
    _row(view, "FRITZ!Box 7590 XY").tapped.emit()
    view.sheet().actions[3].click()
    assert ["device", "disconnect", "wlan0"] in nmcli.commands


def test_a_hidden_network_is_entered_by_name_and_may_be_open(view, nmcli):
    view._hidden.click()
    keyboard = view.keyboard()
    assert keyboard._accept.text() == "Continue"
    assert not keyboard._accept.isEnabled()
    _type(keyboard, "Keller 2")
    keyboard._accept.click()

    assert keyboard.isVisible()
    assert keyboard._title.text() == "Password for “Keller 2”"
    assert keyboard._accept.isEnabled()  # empty: an open network
    _type(keyboard, "kurz")
    assert not keyboard._accept.isEnabled()
    for _ in range(4):
        keyboard.press(BACKSPACE)
    keyboard._accept.click()

    connect = nmcli.commands_starting("-w", "60", "device", "wifi", "connect")
    assert connect == [
        ["-w", "60", "device", "wifi", "connect", "Keller 2", "hidden", "yes", "ifname", "wlan0"]
    ]


def test_a_company_network_is_not_offered(view, nmcli):
    _row(view, "Firma").tapped.emit()
    assert not view.keyboard().isVisible()
    assert view.message.text().startswith("WPA-Enterprise")
    assert view.message.property("role") == "error"


def test_an_action_in_progress_shows_and_holds_the_list(view, nmcli):
    nmcli.holding = True
    _row(view, "Keller").tapped.emit()
    view.sheet().actions[3].click()

    assert view._connection.text() == "Connecting to “Keller” …"
    assert not view._list_panel.isEnabled()

    nmcli.release()
    assert view._list_panel.isEnabled()
    assert view._connection.text() == "FRITZ!Box 7590 XY"


def test_switching_the_wlan_off(view, nmcli):
    nmcli.answers[("radio", "wifi")] = "disabled\n"
    view._switch.click()

    assert ["radio", "wifi", "off"] in nmcli.commands
    assert not view._switch.isChecked() and view._switch.text() == "Off"
    assert view.entries() == []
    assert view._empty.text() == "Wi-Fi is off."
    assert not view._scan.isEnabled()


def test_missing_permissions_and_errors_are_shown_beneath(window, nmcli):
    nmcli.answers[PERMISSIONS] = PERMISSIONS_DENIED
    manager = WifiManager(runner=nmcli)
    view = WifiView(manager, LABEL_WIDTH, window)
    view.show()
    assert "polkit" in view.message.text()
    assert view.message.property("role") == "small"

    manager.failed.emit("Wrong password.")
    assert view.message.text() == "Wrong password."
    assert view.message.property("role") == "error"
    manager.shutdown()


def test_without_nmcli_the_sub_page_says_where_the_wlan_is_set(window, monkeypatch):
    monkeypatch.setattr(wifi, "find_nmcli", lambda: None)
    view = WifiView(WifiManager(), LABEL_WIDTH, window)
    view.show()
    assert view._unavailable.isVisible()
    assert not view._content.isVisible()


# --- IP settings ----------------------------------------------------------------------


def _press(pad: NumberPad, keys: str) -> None:
    for key in keys:
        pad.press(DECIMAL if key == "." else key)


def test_the_number_pad_types_a_list_of_addresses_with_dots_and_commas(window):
    pad = NumberPad(window)
    taken = []
    pad.open_address(
        "DNS servers", "192.168.2.1", taken.append, valid_dns_list, "Several", several=True
    )
    assert pad._keys[DECIMAL].text() == "." and pad._keys[SIGN].text() == ","
    assert pad._keys[SIGN].isEnabled()

    # A comma adds to the list it opened with; backspace takes ", " off at once.
    pad.press(SIGN)
    assert pad.text == "192.168.2.1, "
    pad.press(BACKSPACE)
    assert pad.text == "192.168.2.1"
    pad.press(SIGN)
    _press(pad, "1.1.1")
    assert not pad._accept.isEnabled() and pad._range.property("role") == "error"
    _press(pad, ".1")
    assert pad._accept.isEnabled() and pad._range.property("role") == "small"
    pad._accept.click()

    assert taken == ["192.168.2.1, 1.1.1.1"]
    # Numbers again: the keys write the decimal separator and the sign.
    pad.open("Latitude", 51.0, 4, "°", (-90, 90), lambda _value: None)
    assert pad._keys[DECIMAL].text() == "." and pad._keys[SIGN].text() == "±"


def test_on_an_address_the_first_digit_replaces_and_backspace_takes_one(window):
    pad = NumberPad(window)
    pad.open_address("Address", "192.168.2.47", lambda _text: None, valid_ipv4, "IPv4")
    assert not pad._keys[SIGN].isEnabled()  # one address
    pad.press(BACKSPACE)
    pad.press(BACKSPACE)
    _press(pad, "20")
    assert pad.text == "192.168.2.20"

    pad.open_address("Address", "192.168.2.47", lambda _text: None, valid_ipv4, "IPv4")
    _press(pad, "10")
    assert pad.text == "10"
    assert not pad._accept.isEnabled()


def test_rows_tell_addresses_set_by_hand_and_networks_without_dhcp():
    networks = [Network("Heim", 60, "WPA2", in_use=True), Network("Labor", 50, "WPA2")]
    manual = Ipv4Settings(True, "10.0.0.5", 8)
    saved = [
        SavedNetwork("1", "Heim", "Heim", manual),
        SavedNetwork("2", "Labor", "Labor"),
        SavedNetwork("3", "Keller", "Keller", manual),
    ]
    entries = wifi_entries(networks, saved, "Heim", frozenset({"2"}))
    assert [e.detail for e in entries] == [
        "Connected, IP manual",
        "Saved, no address by DHCP",
        "Saved, out of reach",
    ]


def _open_ip_settings(view: WifiView, ssid: str) -> IpSettingsDialog:
    _row(view, ssid).tapped.emit()
    sheet = view.sheet()
    next(b for b in sheet.actions if b.text() == "IP settings").click()
    dialog = view.ip_settings()
    assert dialog.isVisible() and not sheet.isVisible()
    return dialog


def _choose_manual(dialog: IpSettingsDialog, manual: bool) -> None:
    dialog._mode._group.button(1 if manual else 0).click()


def test_ip_settings_with_dhcp_show_what_the_router_gave(view):
    dialog = _open_ip_settings(view, "FRITZ!Box 7590 XY")

    assert dialog.panel.width() <= SCREEN_WIDTH and dialog.panel.height() <= SCREEN_HEIGHT
    assert dialog._network.text() == "FRITZ!Box 7590 XY"
    assert dialog._mode.current() is False
    assert dialog._address_value.text() == "192.168.178.42"
    assert dialog._subnet_value.text() == "/24" and dialog._mask.text() == "255.255.255.0"
    assert dialog._gateway_value.text() == "192.168.178.1"
    assert not dialog._address_button.isVisibleTo(dialog)
    assert dialog._dns_button.text() == "From the router: 192.168.178.1, 192.168.178.2"
    assert dialog._dns_button.property("placeholder")
    assert dialog._error.text() == ""
    assert not dialog._apply.isEnabled()  # nothing changed


def test_dns_servers_of_its_own_with_dhcp(view, nmcli):
    dialog = _open_ip_settings(view, "FRITZ!Box 7590 XY")
    dialog._dns_button.click()
    pad = dialog.number_pad()
    assert pad.isVisible()
    _press(pad, "9.9.9.9")
    pad.press(SIGN)
    _press(pad, "1.1.1.1")
    pad._accept.click()

    assert dialog._dns_button.text() == "9.9.9.9, 1.1.1.1"
    assert not dialog._dns_button.property("placeholder")
    dialog._apply.click()

    settings = Ipv4Settings(dns=("9.9.9.9", "1.1.1.1"))
    modify = ["connection", "modify", "uuid", UUID_HOME, *ipv4_arguments(settings)]
    assert modify in nmcli.commands
    assert ["device", "reapply", "wlan0"] in nmcli.commands
    assert not dialog.isVisible()


def test_switching_to_manual_starts_from_what_the_router_gave(view):
    dialog = _open_ip_settings(view, "FRITZ!Box 7590 XY")
    _choose_manual(dialog, True)

    assert dialog._address_button.isVisibleTo(dialog)
    assert dialog._address_button.text() == "192.168.178.42"
    assert dialog._prefix_stepper.value() == 24
    assert dialog._gateway_button.text() == "192.168.178.1"
    assert dialog._dns_button.text() == "192.168.178.1, 192.168.178.2"
    assert dialog._apply.isEnabled()

    # Back to DHCP, the router's servers filled in go again.
    _choose_manual(dialog, False)
    assert dialog._dns_button.property("placeholder")
    assert not dialog._apply.isEnabled()


def test_a_gateway_outside_the_subnet_is_marked_until_the_subnet_takes_it_in(view, nmcli):
    dialog = _open_ip_settings(view, "FRITZ!Box 7590 XY")
    _choose_manual(dialog, True)
    dialog._address_button.click()
    pad = dialog.number_pad()
    for _ in range(len("178.42")):
        pad.press(BACKSPACE)
    _press(pad, "2.20")
    pad._accept.click()

    assert dialog._address_button.text() == "192.168.2.20"
    assert dialog._error.text() != ""
    assert dialog._error.text() == "The gateway is not in the subnet 192.168.2.0/24."
    assert dialog._gateway_button.property("invalid")
    assert not dialog._address_button.property("invalid")
    assert not dialog._apply.isEnabled()

    for _ in range(8):
        dialog._prefix_stepper._minus.click()
    assert dialog._prefix_stepper.value() == 16
    assert dialog._mask.text() == "255.255.0.0"
    assert dialog._error.text() == ""
    assert not dialog._gateway_button.property("invalid")
    dialog._apply.click()

    settings = Ipv4Settings(
        True, "192.168.2.20", 16, "192.168.178.1", ("192.168.178.1", "192.168.178.2")
    )
    modify = ["connection", "modify", "uuid", UUID_HOME, *ipv4_arguments(settings)]
    assert modify in nmcli.commands


def test_a_profile_set_by_hand_opens_so_and_its_network_need_not_be_up(view):
    dialog = _open_ip_settings(view, "Keller")
    assert dialog._mode.current() is True
    assert dialog._address_button.text() == "192.168.178.60"
    assert dialog._gateway_button.text() == "192.168.178.1"
    assert dialog._dns_button.text() == "192.168.178.1, 1.1.1.1"
    assert not dialog._apply.isEnabled()

    _choose_manual(dialog, False)
    # Not connected: nothing to show of a router; its own DNS servers stay.
    assert dialog._address_value.text() == "–"
    assert not dialog._mask.isVisibleTo(dialog)
    assert dialog._dns_button.text() == "192.168.178.1, 1.1.1.1"
    assert dialog._apply.isEnabled()


# --- The frequency band -----------------------------------------------------------------


def test_rows_tell_a_network_locked_to_one_band():
    networks = [Network("Heim", 60, "WPA2", in_use=True), Network("Labor", 50, "WPA2")]
    saved = [
        SavedNetwork("1", "Heim", "Heim", Ipv4Settings(True, "10.0.0.5", 8), BAND_5_GHZ),
        SavedNetwork("2", "Labor", "Labor", band=BAND_2_4_GHZ),
        SavedNetwork("3", "Keller", "Keller"),
    ]
    entries = wifi_entries(networks, saved, "Heim")
    assert [e.detail for e in entries] == [
        "Connected, IP manual, 5 GHz only",
        "Saved, 2.4 GHz only",
        "Saved, out of reach",
    ]
    assert bands_text(frozenset({BAND_5_GHZ, BAND_2_4_GHZ})) == "2.4 GHz and 5 GHz"
    assert bands_text(frozenset({BAND_5_GHZ})) == "5 GHz"


def _open_band_sheet(view: WifiView, ssid: str):
    _row(view, ssid).tapped.emit()
    sheet = view.sheet()
    next(b for b in sheet.actions if b.text() == "Frequency band").click()
    assert sheet.isVisible()  # the same sheet, now with the bands
    return sheet


def test_the_band_sheet_tells_the_band_in_use_and_those_in_reach(view, nmcli):
    sheet = _open_band_sheet(view, "FRITZ!Box 7590 XY")

    assert sheet.title.text() == "Frequency band for “FRITZ!Box 7590 XY”"
    assert sheet.detail.text() == "Connected over 2.4 GHz. In reach on 2.4 GHz and 5 GHz."
    assert [b.text() for b in sheet.actions] == ["Automatic", "2.4 GHz", "5 GHz"]
    assert [b.property("role") for b in sheet.actions] == ["accent", None, None]

    sheet.actions[2].click()

    assert not sheet.isVisible()
    modify = ["connection", "modify", "uuid", UUID_HOME, "802-11-wireless.band", "a"]
    assert modify in nmcli.commands
    assert ["-w", "30", "connection", "up", "uuid", UUID_HOME] in nmcli.commands


def test_the_band_locked_to_is_marked_and_a_network_out_of_reach_says_so(view, nmcli):
    nmcli.answers[profile_command(UUID_HIDDEN)] = profile("Keller", band="bg")
    view._wifi.read(saved=True)
    assert _row(view, "Keller").entry.detail == "Saved, out of reach, 2.4 GHz only"

    sheet = _open_band_sheet(view, "Keller")

    assert sheet.detail.text() == "Out of reach."
    assert [b.property("role") for b in sheet.actions] == [None, "accent", None]
    sheet.actions[0].click()
    modify = ["connection", "modify", "uuid", UUID_HIDDEN, "802-11-wireless.band", ""]
    assert modify in nmcli.commands
    assert nmcli.commands_starting("-w", "30", "connection", "up") == []
