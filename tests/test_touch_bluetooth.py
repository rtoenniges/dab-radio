"""The sub-page "Bluetooth" of the touch settings (`ui.touch.bluetooth_view`), on a BlueZ
that answers from a table (`tests.fake_bluez`)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication, QWidget

from net.bluetooth import ADAPTER, DEVICE, BluetoothManager, Reply
from tests.fake_bluez import BOSE, HCI0, JBL, SONY, FakeBus
from ui.touch.bluetooth_view import BluetoothView, DeviceEntry, device_entries
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


@pytest.fixture
def bus() -> FakeBus:
    return FakeBus()


@pytest.fixture
def view(window, bus):
    manager = BluetoothManager(bus=bus)
    view = BluetoothView(manager, window)
    view.resize(SCREEN_WIDTH - 200, 520)
    view.show()  # reads everything
    yield view
    manager.shutdown()


def _row(view: BluetoothView, name: str):
    return next(row for row in view.rows() if row.entry.name == name)


def test_the_entries_tell_connected_and_paired():
    manager = BluetoothManager(bus=FakeBus())
    manager.start()
    entries = device_entries(manager.device_list)

    assert entries[0] == DeviceEntry(SONY, "WH-1000XM4", "headphones", True, True, 80)
    assert entries[0].detail == "Connected, battery 80 %"
    assert DeviceEntry(SONY, "WH-1000XM4", "headphones", True, True).detail == "Connected"
    assert entries[1].detail == "Paired" and entries[1].kind == "speaker"
    assert entries[2].kind == "phone"
    assert entries[3].detail == "" and entries[3].kind == "headphones"


def test_the_sub_page_shows_the_connection_and_the_devices(view):
    assert view._switch.isChecked()
    assert view._connection.text() == "WH-1000XM4"
    assert view._battery.text() == "Battery 80 %"
    assert view._scan.isEnabled()
    assert [e.name for e in view.entries()] == [
        "WH-1000XM4",
        "JBL Flip 6",
        "Pixel 8",
        "LE-Bose QC45",
    ]
    assert view.message.text().startswith("Put the audio device in pairing mode")


def test_without_a_system_bus_the_page_says_so(window):
    manager = BluetoothManager(bus=FakeBus(connected=False))
    view = BluetoothView(manager, window)
    view.show()

    assert view._unavailable.isVisibleTo(view)
    assert not view._content.isVisibleTo(view)
    assert view.message.text() == ""


def test_without_bluetoothd_or_switched_off_the_list_says_why(view, bus):
    bus.service_changed.emit(False)
    assert view._empty.text().startswith("The Bluetooth service")
    assert not view._switch.isEnabled()

    bus.service_changed.emit(True)
    bus.properties_changed.emit(HCI0, ADAPTER, {"Powered": False})
    assert view._empty.text() == "Bluetooth is off."
    assert view._connection.text() == "Bluetooth off"
    assert not view._scan.isEnabled() and view.entries() == []


def test_a_new_device_offers_to_pair_or_forget(view, bus):
    _row(view, "LE-Bose QC45").tapped.emit()
    sheet = view.sheet()
    assert sheet.isVisible()
    assert sheet.title.text() == "LE-Bose QC45" and sheet.detail.text() == "Not paired"
    assert [b.text() for b in sheet.actions] == ["Forget", "Pair"]
    assert bus.calls_of("Pair") == []

    sheet.actions[1].click()

    assert bus.calls_of("Pair") == [(BOSE, DEVICE, "Pair", [])]
    assert bus.calls_of("Connect") == [(BOSE, DEVICE, "Connect", [])]
    assert _row(view, "LE-Bose QC45").entry.connected


def test_a_device_not_paired_can_be_forgotten(view, bus):
    _row(view, "LE-Bose QC45").tapped.emit()
    view.sheet().actions[0].click()

    assert bus.calls_of("RemoveDevice") == [(HCI0, ADAPTER, "RemoveDevice", [BOSE])]
    assert bus.calls_of("Pair") == []
    assert "LE-Bose QC45" not in [e.name for e in view.entries()]


def test_a_paired_device_offers_to_connect_or_forget(view, bus):
    _row(view, "JBL Flip 6").tapped.emit()
    sheet = view.sheet()
    assert sheet.isVisible()
    assert sheet.title.text() == "JBL Flip 6" and sheet.detail.text() == "Paired"
    assert [b.text() for b in sheet.actions] == ["Forget", "Connect"]

    sheet.actions[1].click()

    assert not sheet.isVisible()
    assert bus.calls_of("Connect") == [(JBL, DEVICE, "Connect", [])]


def test_the_device_connected_offers_to_disconnect_or_forget(view, bus):
    _row(view, "WH-1000XM4").tapped.emit()
    sheet = view.sheet()
    assert sheet.detail.text() == "Connected, battery 80 %"
    assert [b.text() for b in sheet.actions] == ["Forget", "Disconnect"]

    sheet.actions[0].click()
    assert bus.calls_of("RemoveDevice") == [(HCI0, ADAPTER, "RemoveDevice", [SONY])]
    assert "WH-1000XM4" not in [e.name for e in view.entries()]


def test_the_device_connected_is_disconnected_from_its_sheet(view, bus):
    _row(view, "WH-1000XM4").tapped.emit()
    view.sheet().actions[1].click()
    assert bus.calls_of("Disconnect") == [(SONY, DEVICE, "Disconnect", [])]
    assert view._connection.text() == "Not connected"


def test_an_action_in_progress_shows_and_holds_the_list(view, bus):
    bus.holding = True
    _row(view, "LE-Bose QC45").tapped.emit()
    view.sheet().actions[1].click()

    assert view._connection.text() == "Pairing with “LE-Bose QC45” …"
    assert not view._list_panel.isEnabled()
    assert not view._scan.isEnabled()

    bus.release()
    assert view._list_panel.isEnabled()


def test_a_failure_is_shown_until_the_next_tap(view, bus):
    bus.answers[(DEVICE, "Connect")] = Reply(
        None, "org.bluez.Error.Failed", "br-connection-page-timeout"
    )
    _row(view, "JBL Flip 6").tapped.emit()
    view.sheet().actions[1].click()

    assert view.message.text().startswith("The device does not answer")
    assert view.message.property("role") == "error"

    view._scan.click()
    assert view.message.property("role") == "small"


def test_searching_disables_the_button_and_says_so(view, bus):
    view._scan.click()

    assert bus.calls_of("StartDiscovery") == [(HCI0, ADAPTER, "StartDiscovery", [])]
    assert not view._scan.isEnabled()
    assert view.message.text() == "Searching …"

    view.hide()  # leaving the page ends the discovery
    assert bus.calls_of("StopDiscovery") == [(HCI0, ADAPTER, "StopDiscovery", [])]


def test_a_tap_on_a_device_ends_the_search(view, bus):
    view._scan.click()
    _row(view, "LE-Bose QC45").tapped.emit()

    # The search takes radio time from headphones playing; the device has been found.
    assert bus.calls_of("StopDiscovery") == [(HCI0, ADAPTER, "StopDiscovery", [])]
    assert view.sheet().isVisible()
    assert view._scan.isEnabled()

    # Without a search of its own, a tap stops nothing.
    view.sheet().dismiss()
    _row(view, "LE-Bose QC45").tapped.emit()
    assert len(bus.calls_of("StopDiscovery")) == 1
