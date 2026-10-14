"""The radio on MQTT (`net.mqtt`): settings, the connection, what is published and
how the topics drive the commands, against a paho client that connects to nothing
(`tests.fake_mqtt`) on a controller without a device (`tests.fake_receiver`)."""

from __future__ import annotations

import base64
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QBuffer, QByteArray, QEventLoop, QIODevice, QTimer
from PyQt6.QtGui import QImage

import i18n
from dab.ensemble_cache import SOURCE_SCAN, EnsembleCache, EnsembleInfo
from dab.mot import CONTENT_TYPE_IMAGE, MotObject
from dab.slideshow import PARAM_TRIGGER_TIME
from net.mqtt import (
    DEFAULT_PREFIX,
    SET_KEYS,
    STATUS_OFF,
    MqttBridge,
    MqttSettings,
    clean_topic_part,
)
from net.radio_state import SECTIONS, RadioStateModel
from tests.fake_mqtt import FakeMqttClient
from tests.fake_receiver import ENSEMBLE_5C, SERVICES_5C, SERVICES_11D
from ui import receiver_controller

DLF, DLF_KULTUR = (sid for sid, _ in SERVICES_5C)
NDR_2 = SERVICES_11D[0][0]
ENABLED = MqttSettings(enabled=True, host="broker", username="radio", password="geheim")
NODE = "dabradio_test"


def _spin(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


@pytest.fixture
def setup(data_directory, make_controller):
    """A bridge on a model on a started controller with 5C stored."""
    controller = make_controller(data_directory, start=False)
    model = RadioStateModel(controller, throttle_ms=50)
    bridge = MqttBridge(model, controller.settings, client_factory=FakeMqttClient, node=NODE)
    controller.start()
    FakeMqttClient.instances = []
    return bridge, model, controller


def _png() -> bytes:
    """A picture of noise: too large for the echo of a command."""
    pixels = os.urandom(64 * 48 * 4)
    image = QImage(pixels, 64, 48, QImage.Format.Format_RGB32).copy()
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return bytes(data)


def _connect(bridge: MqttBridge, settings: MqttSettings = ENABLED) -> FakeMqttClient:
    """Switches the bridge on and lets the broker accept it."""
    bridge.start()
    bridge.apply(settings)
    client = FakeMqttClient.instances[-1]
    client.connect_now()
    return client


def _receive(controller, fake_workers):
    controller.start_reception()
    worker = fake_workers[-1]
    worker.connected.emit("Connected.")
    dsp = controller.dsp_worker
    dsp.stream_restarted.emit(worker.tuning)
    dsp.services_updated.emit(ENSEMBLE_5C, list(SERVICES_5C), {})
    dsp.service_list_completed.emit()
    return worker


# --- Settings -----------------------------------------------------------------------


def test_settings_are_cleaned_stored_and_loaded(setup):
    bridge, _model, controller = setup
    assert bridge.settings == MqttSettings()
    assert clean_topic_part(" /haus/#/radio/ ", "x") == "haus/radio"
    assert clean_topic_part("+", "x") == "x"

    bridge.apply(
        MqttSettings(
            enabled=True, host=" broker.local ", port=99999, prefix="/radio+/", discovery=False
        )
    )
    expected = MqttSettings(
        enabled=True, host="broker.local", port=65535, prefix="radio", discovery=False
    )
    assert bridge.settings == expected
    assert MqttSettings.load(controller.settings) == expected
    controller.settings.setValue("mqtt/port", "kaputt")
    assert MqttSettings.load(controller.settings).port == 1883


# --- The connection -----------------------------------------------------------------


def test_switched_off_the_bridge_connects_to_nothing(setup):
    bridge, _model, controller = setup
    bridge.start()
    assert FakeMqttClient.instances == []
    assert bridge.status.enabled is False
    assert bridge.status.message == STATUS_OFF
    # The controller's own bridge is off as well until the user switches it on.
    assert controller.mqtt_bridge.status.enabled is False


def test_connecting_sets_up_the_client(setup):
    bridge, _model, _controller = setup
    statuses = []
    bridge.status_changed.connect(statuses.append)
    bridge.start()
    bridge.apply(MqttSettings(enabled=True, host="broker", username="radio", password="geheim"))

    client = FakeMqttClient.instances[-1]
    assert client.client_id == NODE
    assert (client.username, client.password) == ("radio", "geheim")
    assert client.tls is None
    assert client.will == (f"{DEFAULT_PREFIX}/available", "offline", True)
    assert client.connect_args == ("broker", 1883, 60)
    assert client.loop_running
    assert bridge.status.connected is False
    assert statuses[-1].message == "Connecting to broker:1883 …"

    bridge.apply(MqttSettings(enabled=True, host="broker", port=8883, tls=True, tls_verify=False))
    assert client.disconnected and not client.loop_running
    client = FakeMqttClient.instances[-1]
    assert client.tls is not None and client.insecure
    assert client.connect_args == ("broker", 8883, 60)

    bridge.apply(MqttSettings(enabled=True))
    assert bridge.status.message == "No broker entered."
    assert len(FakeMqttClient.instances) == 2


def test_the_broker_may_refuse_or_drop(setup):
    bridge, _model, _controller = setup
    bridge.start()
    bridge.apply(ENABLED)
    client = FakeMqttClient.instances[-1]

    client.fail()
    assert bridge.status.message == "No connection to broker:1883, still trying …"
    client.connect_now("Not authorized")
    assert bridge.status.connected is False
    assert bridge.status.message == "The broker broker:1883 refuses: Not authorized"
    client.connect_now()
    assert bridge.status.connected is True
    assert bridge.status.message == "Connected to broker:1883"
    client.drop()
    assert bridge.status.connected is False
    assert bridge.status.message.startswith("Connection to broker:1883 lost")

    bridge.stop()
    assert client.disconnected
    assert bridge.status.message == STATUS_OFF


def test_stopping_says_offline(setup):
    bridge, _model, _controller = setup
    client = _connect(bridge)
    bridge.stop()
    assert client.published[-1] == (f"{DEFAULT_PREFIX}/available", "offline", True)
    assert client.disconnected and not client.loop_running
    assert bridge.status == bridge.status.__class__(True, False, STATUS_OFF)


# --- What goes out ------------------------------------------------------------------


def test_on_connection_everything_is_published(setup):
    bridge, _model, controller = setup
    controller.set_volume_percent(55)
    client = _connect(bridge)

    assert client.subscribed == [[("dabradio/command/#", 1), ("dabradio/set/#", 1)]]
    assert client.last("dabradio/available") == "online"
    for name in SECTIONS:
        assert f"dabradio/state/{name}" in client.topics()
    assert client.last_json("dabradio/state/receiver")["channel"] == "5C"
    assert client.last_json("dabradio/state/audio")["volume_percent"] == 55
    assert client.last("dabradio/station") == ""
    assert client.last("dabradio/output") == "Automatic"
    assert client.last("dabradio/slide") == b""
    assert all(retain for topic, _payload, retain in client.published if topic != "dabradio/result")

    # Home Assistant: the entities under the discovery prefix, the state read from
    # the sections.
    station = client.last_json(f"homeassistant/select/{NODE}/station/config")
    assert station["options"] == ["Dlf", "Dlf Kultur"]
    assert station["command_topic"] == "dabradio/set/station"
    assert station["device"]["identifiers"] == [NODE]
    assert station["device"]["sw_version"] == receiver_controller.APPLICATION_VERSION
    assert station["availability_topic"] == "dabradio/available"
    volume = client.last_json(f"homeassistant/number/{NODE}/volume/config")
    assert volume["state_topic"] == "dabradio/state/audio"
    assert volume["value_template"] == "{{ value_json.volume_percent }}"
    muted = client.last_json(f"homeassistant/switch/{NODE}/muted/config")
    assert muted["value_template"] == "{{ 'ON' if value_json.muted else 'OFF' }}"
    assert station["name"] == "Station"
    assert station["device"]["name"] == "DAB radio"
    title = client.last_json(f"homeassistant/sensor/{NODE}/title/config")
    assert title["name"] == "Title"
    assert title["state_topic"] == "dabradio/state/now_playing"
    assert client.last_json(f"homeassistant/image/{NODE}/slide/config")["image_topic"] == (
        "dabradio/slide"
    )
    for key in SET_KEYS:
        assert any(f"/{key}/config" in topic for topic in client.topics()), key

    # The configs go out before the state, so that the entities exist when it comes.
    topics = client.topics()
    assert topics.index(f"homeassistant/select/{NODE}/station/config") < topics.index(
        "dabradio/state/receiver"
    )


def test_changes_are_published_while_connected(setup, fake_workers):
    bridge, _model, controller = setup
    controller.set_volume_percent(10)
    client = _connect(bridge)
    published = len(client.published)

    controller.set_volume_percent(20)
    assert client.last_json("dabradio/state/audio")["volume_percent"] == 20
    _receive(controller, fake_workers)
    controller.select_service(DLF_KULTUR)
    assert client.last("dabradio/station") == "Dlf Kultur"
    assert client.last_json("dabradio/state/service")["name"] == "Dlf Kultur"

    # The slide as bytes; a new image format is announced to Home Assistant anew.
    now = {PARAM_TRIGGER_TIME: bytes(4)}
    controller.dsp_worker.slide_ready.emit(
        MotObject(1, CONTENT_TYPE_IMAGE, 3, "a.png", b"png", now)
    )
    assert client.last("dabradio/slide") == b"png"
    config = client.last_json(f"homeassistant/image/{NODE}/slide/config")
    assert config["content_type"] == "image/png"
    assert len(client.published) > published

    client.drop()
    published = len(client.published)
    controller.set_volume_percent(30)
    assert len(client.published) == published  # nothing while disconnected
    client.connect_now()
    assert client.last_json("dabradio/state/audio")["volume_percent"] == 30


def test_station_names_merge_favourites_and_the_ensemble(data_directory, make_controller):
    cache = EnsembleCache(data_directory / receiver_controller.ENSEMBLE_CACHE_FILENAME)
    cache.load()
    cache.put("11D", EnsembleInfo("NDR NDS", ((DLF, "Dlf"), (NDR_2, "NDR 2")), 0.0, SOURCE_SCAN))
    cache.save()
    controller = make_controller(data_directory, start=False)
    model = RadioStateModel(controller)
    bridge = MqttBridge(model, controller.settings, client_factory=FakeMqttClient, node=NODE)
    controller.start()
    controller.set_favourite("11D", DLF, True)
    controller.set_favourite("11D", NDR_2, True)
    client = _connect(bridge)

    station = client.last_json(f"homeassistant/select/{NODE}/station/config")
    assert station["options"] == ["Dlf (11D)", "NDR 2", "Dlf (5C)", "Dlf Kultur"]
    controller.select_service(DLF)
    assert client.last("dabradio/station") == "Dlf (5C)"

    client.deliver("dabradio/set/station", "Dlf (11D)")
    assert controller.channel == "11D"
    assert controller.selected_sid == DLF
    result = client.last_json("dabradio/result")
    assert result == {
        "command": "play",
        "arguments": {"channel": "11D", "sid": "D210"},
        "ok": True,
        "message": None,
    }
    # On 11D now: its stored list is on display, and "Dlf" is unambiguous again.
    controller.set_favourite("11D", NDR_2, False)
    station = client.last_json(f"homeassistant/select/{NODE}/station/config")
    assert station["options"] == ["Dlf", "NDR 2"]
    assert client.last("dabradio/station") == "Dlf"


# --- What comes in ------------------------------------------------------------------


def test_set_topics_drive_the_commands(setup, fake_workers):
    bridge, _model, controller = setup
    client = _connect(bridge)

    client.deliver("dabradio/set/volume", "42")
    assert controller.volume_percent == 42
    assert client.last_json("dabradio/result")["ok"] is True
    client.deliver("dabradio/set/muted", "ON")
    assert controller.audio_enabled is False
    client.deliver("dabradio/set/muted", "off")
    assert controller.audio_enabled is True
    client.deliver("dabradio/set/station", "Dlf Kultur")
    assert controller.selected_sid == DLF_KULTUR
    assert fake_workers[-1].started
    client.deliver("dabradio/set/reception", "OFF")
    assert fake_workers[-1].stop_requested
    client.deliver("dabradio/set/equalizer", "ON")
    assert controller.equalizer_enabled is True
    client.deliver("dabradio/set/output", "Automatic")
    assert controller.audio_output_choice == ""

    client.deliver("dabradio/set/station", "WDR 2")
    assert client.last_json("dabradio/result") == {
        "command": "set/station",
        "arguments": {"value": "WDR 2"},
        "ok": False,
        "message": "Unknown station “WDR 2”.",
    }
    client.deliver("dabradio/set/scan", "maybe")
    assert client.last_json("dabradio/result")["message"] == "“maybe” is neither ON nor OFF."
    client.deliver("dabradio/set/volume", "loud")
    assert client.last_json("dabradio/result")["message"] == (
        "“percent” must be a number, not 'loud'."
    )
    client.deliver("dabradio/set/colour", "red")
    assert client.last_json("dabradio/result")["message"] == "Unknown key “colour”."
    results = client.payloads("dabradio/result")
    assert all(not retain for topic, _p, retain in client.published if topic == "dabradio/result")
    assert len(results) == 11


def test_command_topic_takes_json_arguments(setup):
    bridge, _model, controller = setup
    client = _connect(bridge)

    client.deliver("dabradio/command/set_volume", '{"percent": 7}')
    assert controller.volume_percent == 7
    client.deliver("dabradio/command/select_service", '{"name": "Dlf"}')
    assert controller.selected_sid == DLF
    client.deliver("dabradio/command/stop_reception", "")
    assert client.last_json("dabradio/result") == {
        "command": "stop_reception",
        "arguments": {},
        "ok": False,
        "message": "Reception is not running.",
    }
    client.deliver("dabradio/command/set_volume", "[1, 2]")
    assert client.last_json("dabradio/result")["message"] == (
        "The arguments must be a JSON object."
    )
    client.deliver("dabradio/command/dance", "{}")
    assert client.last_json("dabradio/result")["message"] == "Unknown command “dance”."
    _spin(10)


def test_withdrawing_entities_and_moving_the_prefix(setup):
    bridge, _model, _controller = setup
    client = _connect(bridge)
    configs = [topic for topic in client.topics() if topic.endswith("/config")]
    assert len(configs) == 23

    bridge.apply(MqttSettings(enabled=True, host="broker", discovery=False))
    # The old client cleared every config before the connection was made anew.
    cleared = [topic for topic, payload, retain in client.published if payload == b"" and retain]
    assert set(configs) <= set(cleared)
    assert "dabradio/state/receiver" not in cleared
    successor = FakeMqttClient.instances[-1]
    assert successor is not client
    successor.connect_now()
    assert not any(topic.endswith("/config") for topic in successor.topics())
    assert successor.last("dabradio/available") == "online"

    bridge.apply(MqttSettings(enabled=True, host="broker", discovery=False, prefix="radio"))
    cleared = [topic for topic, payload, retain in successor.published if payload == b"" and retain]
    assert "dabradio/state/receiver" in cleared
    assert "dabradio/available" in cleared
    third = FakeMqttClient.instances[-1]
    third.connect_now()
    assert third.last("radio/available") == "online"
    assert third.subscribed == [[("radio/command/#", 1), ("radio/set/#", 1)]]


def test_applied_settings_are_reported(setup):
    bridge, _model, _controller = setup
    applied = []
    bridge.settings_changed.connect(applied.append)
    bridge.apply(MqttSettings(host=" broker "))
    bridge.apply(MqttSettings(host="broker"))  # the same again: nothing to report
    assert applied == [MqttSettings(host="broker")]


def test_everything_on_mqtt_is_english_when_the_surfaces_are_german(
    data_directory, make_controller, fake_workers
):
    i18n.set_language("de")
    try:
        controller = make_controller(data_directory, start=False)
        model = RadioStateModel(controller, throttle_ms=50)
        bridge = MqttBridge(model, controller.settings, client_factory=FakeMqttClient, node=NODE)
        controller.start()
        FakeMqttClient.instances = []
        client = _connect(bridge)

        assert client.last("dabradio/output") == "Automatic"
        station = client.last_json(f"homeassistant/select/{NODE}/station/config")
        assert station["name"] == "Station"
        assert station["device"]["name"] == "DAB radio"
        volume = client.last_json(f"homeassistant/number/{NODE}/volume/config")
        assert volume["name"] == "Volume"
        output = client.last_json(f"homeassistant/select/{NODE}/output/config")
        assert output["options"][0] == "Automatic"

        client.deliver("dabradio/set/output", "Automatic")
        assert controller.audio_output_choice == ""
        client.deliver("dabradio/set/station", "WDR 2")
        assert client.last_json("dabradio/result")["message"] == "Unknown station “WDR 2”."
        client.deliver("dabradio/set/muted", "perhaps")
        assert client.last_json("dabradio/result")["message"] == (
            "“muted” must be true or false, not 'perhaps'."
        )
        # The status line of the touch page "MQTT" is the surface's own: German.
        assert bridge.status.message.startswith("Verbunden")
    finally:
        i18n.set_language(i18n.SOURCE_LANGUAGE)


def test_notifications_and_their_answers(setup):
    bridge, _model, controller = setup
    client = _connect(bridge)
    image = base64.b64encode(_png()).decode()

    client.deliver(
        "dabradio/command/notify",
        json.dumps(
            {
                "id": "door",
                "title": "Front door",
                "message": "It rang.",
                "buttons": [{"id": "open", "label": "Open", "style": "accent"}, "Ignore"],
                "image": image,
            }
        ),
    )

    result = client.last_json("dabradio/result")
    assert result["ok"] is True
    assert result["command"] == "notify"
    # Not the whole image back.
    assert result["arguments"]["image"].endswith(f"({len(image)} characters)")
    assert controller.notifications.shown.id == "door"
    state = client.last_json("dabradio/state/notifications")
    assert state["shown"] == "door"
    assert state["open"][0]["image"] is True

    controller.notifications.press("door", "2")
    assert client.last_json("dabradio/notification/answer") == {
        "id": "door",
        "reason": "pressed",
        "button": "2",
        "label": "Ignore",
    }
    assert client.qos["dabradio/notification/answer"] == 1
    assert all(
        not retain for topic, _p, retain in client.published if topic.startswith("dabradio/notif")
    )
    assert client.last_json("dabradio/state/notifications") == {"shown": None, "open": []}

    # The notify entity of Home Assistant sends the text alone.
    client.deliver("dabradio/set/notify", "The washing is done.")
    shown = controller.notifications.shown
    assert (shown.title, shown.message, shown.buttons) == ("", "The washing is done.", ())
    client.deliver("dabradio/command/dismiss_notification", json.dumps({"id": shown.id}))
    assert client.last_json("dabradio/notification/answer")["reason"] == "dismissed"
    client.deliver("dabradio/set/notify", "")
    assert client.last_json("dabradio/result")["message"] == (
        "A notification needs a “title” or a “message”."
    )


def test_notification_entities_for_home_assistant(setup):
    bridge, _model, _controller = setup
    client = _connect(bridge)

    notify = client.last_json(f"homeassistant/notify/{NODE}/notify/config")
    assert notify["command_topic"] == "dabradio/set/notify"
    assert notify["qos"] == 1
    assert notify["name"] == "Notification"
    event = client.last_json(f"homeassistant/event/{NODE}/notification_answer/config")
    assert event["state_topic"] == "dabradio/notification/answer"
    assert event["event_types"] == ["pressed", "timeout", "dismissed", "replaced", "closed"]
    assert "value_json.reason" in event["value_template"]
    assert event["unique_id"] == f"{NODE}_notification_answer"
