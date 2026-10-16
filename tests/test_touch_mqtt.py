"""The sub-page "MQTT" of the touch settings (`ui.touch.mqtt_view`), on a bridge whose
paho client connects to nothing (`tests.fake_mqtt`)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication, QWidget

from net.mqtt import MqttBridge, MqttSettings
from net.radio_state import RadioStateModel
from tests.fake_mqtt import FakeMqttClient
from ui.touch.mqtt_view import (
    NO_HOST,
    NO_PASSWORD,
    NO_USER,
    PASSWORD_DOTS,
    MqttView,
    valid_host,
    valid_prefix,
)
from ui.touch.settings_page import SettingsPage
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
    model = RadioStateModel(controller)
    bridge = MqttBridge(
        model, controller.settings, client_factory=FakeMqttClient, node="dabradio_test"
    )
    bridge.start()
    FakeMqttClient.instances = []
    view = MqttView(bridge, window)
    view.resize(SCREEN_WIDTH - 200, 520)
    view.show()
    yield view, bridge
    window.deleteLater()


def _type(keyboard, text: str) -> None:
    for character in text:
        keyboard.press(character)


def test_the_checks():
    assert valid_host("broker.local") and valid_host(" 192.168.1.10 ")
    assert not valid_host("") and not valid_host("my broker")
    assert valid_prefix("dabradio") and valid_prefix("haus/radio")
    assert not valid_prefix("") and not valid_prefix("dab radio") and not valid_prefix("a/+")


def test_the_rows_show_the_settings(view):
    view, bridge = view
    assert bridge.settings == MqttSettings()
    assert not view._switch.isChecked()
    assert view._status.text() == "Off"
    assert view._host.text() == NO_HOST
    assert view._port.text() == "1883"
    assert view._username.text() == NO_USER
    assert view._password.text() == NO_PASSWORD
    assert not view._tls.isChecked() and not view._tls_verify.isVisible()
    assert view._prefix.text() == "dabradio"
    assert view._discovery.isChecked() and view._discovery_prefix.text() == "homeassistant"


def test_the_keyboard_takes_the_broker_and_the_switch_connects(view):
    view, bridge = view
    view._host.click()
    keyboard = view.keyboard()
    assert keyboard.isVisible()
    _type(keyboard, "my broker")
    keyboard.accept()  # a blank: not taken
    assert keyboard.isVisible()
    assert bridge.settings.host == ""
    for _ in range(len("my broker")):
        keyboard.press("⌫")
    _type(keyboard, "broker")
    keyboard.accept()
    assert not keyboard.isVisible()
    assert bridge.settings.host == "broker"
    assert view._host.text() == "broker"
    assert FakeMqttClient.instances == []  # still off

    view._switch.click()
    assert bridge.settings.enabled
    assert view._switch.isChecked()
    client = FakeMqttClient.instances[-1]
    assert client.connect_args == ("broker", 1883, 60)
    assert view._status.text() == "Connecting to broker:1883 …"
    client.connect_now()
    assert view._status.text() == "Connected to broker:1883"
    client.connect_now("Not authorized")
    assert view._status.text() == "The broker broker:1883 refuses: Not authorized"

    view._switch.click()
    assert not bridge.settings.enabled
    assert client.disconnected
    assert view._status.text() == "Off"


def test_login_and_prefixes_come_over_the_keyboard(view):
    view, bridge = view
    view._username.click()
    _type(view.keyboard(), "radio")
    view.keyboard().accept()
    assert bridge.settings.username == "radio"
    assert view._username.text() == "radio"

    view._password.click()
    _type(view.keyboard(), "geheim")
    view.keyboard().accept()
    assert bridge.settings.password == "geheim"
    assert view._password.text() == PASSWORD_DOTS
    # Emptied again: no password.
    view._password.click()
    for _ in range(len("geheim")):
        view.keyboard().press("⌫")
    view.keyboard().accept()
    assert bridge.settings.password == ""
    assert view._password.text() == NO_PASSWORD

    view._prefix.click()
    _type(view.keyboard(), "/haus")
    view.keyboard().accept()
    assert bridge.settings.prefix == "dabradio/haus"
    assert view._prefix.text() == "dabradio/haus"

    view._discovery.click()
    assert not bridge.settings.discovery
    assert not view._discovery_prefix.isVisible()
    view._discovery.click()
    assert bridge.settings.discovery and view._discovery_prefix.isVisible()


def test_tls_moves_the_port_and_the_pad_sets_another(view):
    view, bridge = view
    view._tls.click()
    assert bridge.settings.tls and bridge.settings.port == 8883
    assert view._port.text() == "8883"
    assert view._tls_verify.isVisible() and view._tls_verify.isChecked()
    view._tls_verify.click()
    assert not bridge.settings.tls_verify
    view._tls.click()
    assert not bridge.settings.tls and bridge.settings.port == 1883

    view._port.click()
    pad = view.number_pad()
    assert pad.isVisible()
    for key in "1884":
        pad.press(key)
    pad._on_accept_clicked()
    assert bridge.settings.port == 1884
    assert view._port.text() == "1884"
    # Another port is left alone by the TLS switch.
    view._tls.click()
    assert bridge.settings.tls and bridge.settings.port == 1884


def test_the_page_follows_settings_applied_elsewhere(view):
    view, bridge = view
    bridge.apply(MqttSettings(enabled=True, host="broker", username="radio", password="x"))
    assert view._switch.isChecked()
    assert view._host.text() == "broker"
    assert view._username.text() == "radio"
    assert view._password.text() == PASSWORD_DOTS
    assert view._status.text() == "Connecting to broker:1883 …"


def test_the_settings_page_has_the_sub_page(make_controller, data_directory):
    controller = make_controller(data_directory)
    page = SettingsPage(controller)
    try:
        page.show_subpage("mqtt")
        assert page.current_subpage() == "mqtt"
        assert page._mqtt_view._bridge is controller.mqtt_bridge
    finally:
        page.shutdown()
        page.deleteLater()
