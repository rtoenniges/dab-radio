"""The radio on MQTT, for a smart home: its state as retained topics, commands from
topics, and entities for Home Assistant through MQTT Discovery. What is published comes
from `net.radio_state.RadioStateModel`; this module only maps it onto topics.

The radio is the client: it connects to the broker of the smart home (Mosquitto, the
one of Home Assistant, ioBroker …) and opens no port of its own; whom the broker lets
in, and whether over TLS, the broker decides. paho-mqtt runs the connection in a thread
of its own (`loop_start`) and reconnects by itself; its callbacks only hand over to the
GUI thread through signals, where the state is read and the commands go to the model.

Topics, under the prefix of the settings (`MqttSettings.prefix`, "dabradio"):
  <prefix>/available          online | offline (retained; the will says offline)
  <prefix>/state/<section>    the section as JSON (retained), see `net.radio_state.SECTIONS`
  <prefix>/slide              the image of the slideshow, its bytes (retained; empty: none)
  <prefix>/station            the station playing, named as `set/station` takes it (retained)
  <prefix>/output             the audio output chosen, named as `set/output` takes it (retained)
  <prefix>/command/<name>     a command of `net.radio_state.COMMANDS`; the payload its
                              arguments as a JSON object, or empty
  <prefix>/set/<key>          a value for one entity (`SET_KEYS`): the station or output by
                              name, the volume in percent, ON or OFF for the switches,
                              the text of a notification
  <prefix>/result             what became of a command (JSON: command, arguments, ok,
                              message), not retained
  <prefix>/notification/answer  how a notification ended (`notify`; JSON: id, reason,
                              button, label), not retained

Commands and the answers of notifications go at least once (QoS 1): a lost answer leaves
an automation waiting. Everything else goes at most once (QoS 0).

The smart home finds the radio through MQTT Discovery, the scheme of Home Assistant
that openHAB, Domoticz and others read as well (configs retained under
`<discovery prefix>/<component>/<node>/<key>/config`): selects for the station (the
favourites and the services of the ensemble) and the audio output, a number for the
volume, switches for mute, reception, scan, recording, announcement switching and
equalizer, sensors for station, ensemble, channel, text, title, artist, reception,
status, recorder, codec and bit rate, the slideshow as an image, a notify entity for
notifications of text alone and an event entity for their answers. The names, the
`result` texts and the name of the default output among the options are English,
whatever the surfaces show; the status of the bridge, for the settings page, is
translated.

The settings (`MqttSettings`) live in the QSettings of the controller under "mqtt/",
the password in plain text: the file is the user's own, and unlike the WLAN there is
no keeper such as NetworkManager to hand it to.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import ssl
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

from paho.mqtt.client import CallbackAPIVersion, Client
from PyQt6.QtCore import QObject, QSettings, pyqtSignal

from i18n import tr, tr_noop
from net.radio_state import SECTIONS, RadioStateModel, brief_arguments
from ui.notifications import REASONS

logger = logging.getLogger(__name__)

DEFAULT_PORT = 1883
DEFAULT_TLS_PORT = 8883
DEFAULT_PREFIX = "dabradio"
DEFAULT_DISCOVERY_PREFIX = "homeassistant"
KEEPALIVE_S = 60
# paho waits this long before trying to reconnect, doubling up to the maximum.
RECONNECT_DELAY_S = (1, 60)
QOS = 0
# Commands, and the answers of notifications: at least once.
COMMAND_QOS = 1
SET_KEYS = (
    "station",
    "volume",
    "muted",
    "reception",
    "scan",
    "recording",
    "announcement_switching",
    "equalizer",
    "output",
    "notify",
)
# The audio output that follows the default of the system, an option of the output select
# in the smart home. Everything on MQTT is English, whatever the surfaces show.
OUTPUT_AUTOMATIC = "Automatic"
# Translators: the status of the MQTT connection while it is switched off.
STATUS_OFF = tr_noop("Off")
_SETTINGS_GROUP = "mqtt"
_ON = ("on", "true", "1", "an", "ein")
_OFF = ("off", "false", "0", "aus")


def clean_topic_part(text: str, default: str) -> str:
    """`text` as a part of a topic: without wildcards and blanks, without slashes at
    the ends; `default` if nothing remains."""
    return re.sub(r"/+", "/", re.sub(r"[+#\s]", "", text)).strip("/") or default


@dataclass(frozen=True)
class MqttSettings:
    enabled: bool = False
    host: str = ""
    port: int = DEFAULT_PORT
    username: str = ""
    password: str = ""
    tls: bool = False
    tls_verify: bool = True  # check the broker's certificate against the system's CAs
    prefix: str = DEFAULT_PREFIX
    discovery: bool = True  # announce the entities through MQTT Discovery
    discovery_prefix: str = DEFAULT_DISCOVERY_PREFIX

    @classmethod
    def load(cls, settings: QSettings) -> MqttSettings:
        """The stored settings; a value that cannot be read takes its default."""
        values = {}
        for field in fields(cls):
            default = field.default
            try:
                values[field.name] = settings.value(
                    f"{_SETTINGS_GROUP}/{field.name}", default, type=type(default)
                )
            except TypeError, ValueError:
                values[field.name] = default
        return cls(**values).cleaned()

    def store(self, settings: QSettings) -> None:
        for field in fields(self):
            settings.setValue(f"{_SETTINGS_GROUP}/{field.name}", getattr(self, field.name))

    def cleaned(self) -> MqttSettings:
        """With the host trimmed, the port in range and the prefixes fit for topics."""
        return MqttSettings(
            enabled=bool(self.enabled),
            host=self.host.strip(),
            port=min(max(int(self.port), 1), 65535),
            username=self.username.strip(),
            password=self.password,
            tls=bool(self.tls),
            tls_verify=bool(self.tls_verify),
            prefix=clean_topic_part(self.prefix, DEFAULT_PREFIX),
            discovery=bool(self.discovery),
            discovery_prefix=clean_topic_part(self.discovery_prefix, DEFAULT_DISCOVERY_PREFIX),
        )

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass(frozen=True)
class MqttStatus:
    """What the bridge does, for the settings page."""

    enabled: bool
    connected: bool
    message: str  # translated, for the settings page


def node_id() -> str:
    """The client id at the broker and the node of the discovery topics: from the
    host's name, so that two radios in one house tell apart."""
    name = re.sub(r"[^a-z0-9]+", "_", socket.gethostname().lower()).strip("_")
    return f"dabradio_{name}" if name else "dabradio"


def paho_client(client_id: str) -> Client:
    return Client(CallbackAPIVersion.VERSION2, client_id=client_id)


def _switched_on(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered in _ON:
        return True
    if lowered in _OFF:
        return False
    # The message goes to the user as the answer under `result`.
    raise ValueError(f"“{text}” is neither ON nor OFF.")


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode()


class MqttBridge(QObject):
    """See the module docstring. `start` connects if the settings say so, `apply` takes
    new settings (stored, and the connection made anew), `stop` says goodbye."""

    status_changed = pyqtSignal(object)  # MqttStatus
    settings_changed = pyqtSignal(object)  # MqttSettings, once `apply` took them
    # From paho's thread into the GUI thread.
    _connected = pyqtSignal(object)  # the reason code of the CONNACK
    _connect_failed = pyqtSignal()
    _disconnected = pyqtSignal(object)  # the reason code
    _message_received = pyqtSignal(str, bytes)  # topic, payload

    def __init__(
        self,
        model: RadioStateModel,
        settings: QSettings,
        client_factory: Callable[[str], Client] = paho_client,
        node: str | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._qsettings = settings
        self._settings = MqttSettings.load(settings)
        self._factory = client_factory
        self._node = node or node_id()
        self._client: Client | None = None
        self._started = False
        self._status = MqttStatus(self._settings.enabled, False, tr(STATUS_OFF))
        self._stations: dict[str, tuple[str, str]] = {}  # label: (channel, SId as hex)
        self._outputs: dict[str, str] = {}  # label: id of the output ("" = automatic)
        self._slide_type: str | None = None  # the MIME type announced for the image

        self._connected.connect(self._on_connected)
        self._connect_failed.connect(self._on_connect_failed)
        self._disconnected.connect(self._on_disconnected)
        self._message_received.connect(self._on_message)
        model.changed.connect(self._on_changed)
        model.command_finished.connect(self._on_command_finished)
        model.notification_answered.connect(self._on_notification_answered)

    # --- Control ------------------------------------------------------------------

    @property
    def settings(self) -> MqttSettings:
        return self._settings

    @property
    def status(self) -> MqttStatus:
        return self._status

    @property
    def node(self) -> str:
        return self._node

    def start(self) -> None:
        """Connects, if switched on; from then on `apply` makes the connection anew."""
        self._started = True
        self._connect()

    def stop(self) -> None:
        """Says offline and disconnects; for the application's exit."""
        self._started = False
        self._disconnect(tr(STATUS_OFF))

    def apply(self, settings: MqttSettings) -> None:
        """Stores the settings and, once started, connects with them anew (or
        disconnects, if switched off). Entities announced under a prefix no longer in
        use are withdrawn first."""
        settings = settings.cleaned()
        before = self._settings
        if settings == before:
            return
        settings.store(self._qsettings)
        if self._is_connected():
            self._withdraw(before, settings)
        self._settings = settings
        if self._started:
            self._disconnect(tr(STATUS_OFF))
            self._connect()
        self.settings_changed.emit(settings)

    def _connect(self) -> None:
        settings = self._settings
        if not settings.enabled:
            self._set_status(False, False, tr(STATUS_OFF))
            return
        if not settings.host:
            self._set_status(True, False, tr("No broker entered."))
            return
        client = self._factory(self._node)
        if settings.username:
            client.username_pw_set(settings.username, settings.password or None)
        if settings.tls:
            client.tls_set(cert_reqs=ssl.CERT_REQUIRED if settings.tls_verify else ssl.CERT_NONE)
            if not settings.tls_verify:
                client.tls_insecure_set(True)
        client.will_set(self._topic("available"), "offline", qos=QOS, retain=True)
        client.reconnect_delay_set(*RECONNECT_DELAY_S)
        client.on_connect = self._paho_on_connect
        client.on_connect_fail = self._paho_on_connect_fail
        client.on_disconnect = self._paho_on_disconnect
        client.on_message = self._paho_on_message
        self._client = client
        try:
            client.connect_async(settings.host, settings.port, keepalive=KEEPALIVE_S)
        except (OSError, ValueError) as exc:
            self._client = None
            message = tr("Cannot connect to {address}: {error}")
            self._set_status(True, False, message.format(address=settings.address, error=exc))
            return
        client.loop_start()
        message = tr("Connecting to {address} …").format(address=settings.address)
        self._set_status(True, False, message)

    def _disconnect(self, message: str) -> None:
        client, self._client = self._client, None
        if client is not None:
            if self._status.connected:
                client.publish(self._topic("available"), "offline", qos=QOS, retain=True)
            client.disconnect()
            client.loop_stop()
        self._set_status(self._settings.enabled, False, message)

    def _set_status(self, enabled: bool, connected: bool, message: str) -> None:
        status = MqttStatus(enabled, connected, message)
        if status == self._status:
            return
        self._status = status
        logger.info("MQTT: %s", message)
        self.status_changed.emit(status)

    def _is_connected(self) -> bool:
        return self._client is not None and self._status.connected

    # --- paho's thread ------------------------------------------------------------

    def _paho_on_connect(self, _client, _userdata, _flags, reason_code, _properties) -> None:
        self._connected.emit(reason_code)

    def _paho_on_connect_fail(self, _client, _userdata) -> None:
        self._connect_failed.emit()

    def _paho_on_disconnect(self, _client, _userdata, _flags, reason_code, _properties) -> None:
        self._disconnected.emit(reason_code)

    def _paho_on_message(self, _client, _userdata, message) -> None:
        self._message_received.emit(message.topic, bytes(message.payload))

    # --- The GUI thread -----------------------------------------------------------

    def _on_connected(self, reason_code: object) -> None:
        client = self._client
        if client is None:
            return  # stopped meanwhile
        settings = self._settings
        if getattr(reason_code, "is_failure", False):
            message = tr("The broker {address} refuses: {reason}")
            self._set_status(
                True, False, message.format(address=settings.address, reason=reason_code)
            )
            return
        self._set_status(True, True, tr("Connected to {address}").format(address=settings.address))
        client.subscribe(
            [(self._topic("command/#"), COMMAND_QOS), (self._topic("set/#"), COMMAND_QOS)]
        )
        self._publish_all()

    def _on_connect_failed(self) -> None:
        if self._client is None:
            return
        message = tr("No connection to {address}, still trying …")
        self._set_status(True, False, message.format(address=self._settings.address))

    def _on_disconnected(self, reason_code: object) -> None:
        if self._client is None:
            return
        message = tr("Connection to {address} lost ({reason}), trying again …")
        self._set_status(
            True, False, message.format(address=self._settings.address, reason=reason_code)
        )

    def _on_message(self, topic: str, payload: bytes) -> None:
        command_prefix = self._topic("command/")
        set_prefix = self._topic("set/")
        text = payload.decode("utf-8", "replace").strip()
        if topic.startswith(command_prefix):
            name = topic[len(command_prefix) :]
            arguments: Any = {}
            if text:
                try:
                    arguments = json.loads(text)
                except ValueError:
                    arguments = None
            if not isinstance(arguments, dict):
                self._publish_result(
                    name, {"payload": text}, "The arguments must be a JSON object."
                )
                return
            self._model.submit(name, arguments)
        elif topic.startswith(set_prefix):
            key = topic[len(set_prefix) :]
            try:
                name, arguments = self._set_command(key, text)
            except ValueError as exc:
                self._publish_result(f"set/{key}", {"value": text}, str(exc))
                return
            self._model.submit(name, arguments)

    def _set_command(self, key: str, text: str) -> tuple[str, dict]:
        """The command behind a `set/<key>` topic; the model checks the values. The
        message of a ValueError goes to the user as the answer under `result`."""
        match key:
            case "station":
                station = self._stations.get(text)
                if station is None:
                    raise ValueError(f"Unknown station “{text}”.")
                channel, sid = station
                return "play", {"channel": channel, "sid": sid}
            case "volume":
                return "set_volume", {"percent": text}
            case "muted":
                return "set_muted", {"muted": text}
            case "reception":
                return ("start_reception" if _switched_on(text) else "stop_reception"), {}
            case "scan":
                return ("start_scan" if _switched_on(text) else "cancel_scan"), {}
            case "recording":
                return "set_recording_enabled", {"enabled": text}
            case "announcement_switching":
                return "set_announcement_switching", {"enabled": text}
            case "equalizer":
                return "set_equalizer_enabled", {"enabled": text}
            case "output":
                if text not in self._outputs:
                    raise ValueError(f"Unknown output “{text}”.")
                return "set_output", {"id": self._outputs[text]}
            case "notify":
                # The notify entity of Home Assistant sends the message alone.
                return "notify", {"message": text}
        raise ValueError(f"Unknown key “{key}”.")

    def _on_command_finished(self, name: str, arguments: object, answer: object) -> None:
        if self._is_connected():
            self._publish_result(name, arguments, answer)

    def _on_notification_answered(self, answer: dict) -> None:
        if self._is_connected():
            topic = self._topic("notification/answer")
            self._publish(topic, _json(answer), retain=False, qos=COMMAND_QOS)

    def _on_changed(self, section: str) -> None:
        if not self._is_connected():
            return  # everything goes out anew on the next connection
        if section == "slide":
            self._publish_slide()
            return
        self._publish_section(section)
        discovery = self._settings.discovery
        if section in ("favourites", "services", "service"):
            if self._refresh_stations() and discovery:
                self._publish_config(self._station_config())
            self._publish(self._topic("station"), self._station_label())
        elif section == "audio":
            if self._refresh_outputs() and discovery:
                self._publish_config(self._output_config())
            self._publish(self._topic("output"), self._output_label())

    # --- Publishing ---------------------------------------------------------------

    def _topic(self, suffix: str, settings: MqttSettings | None = None) -> str:
        return f"{(settings or self._settings).prefix}/{suffix}"

    def _publish(
        self, topic: str, payload: bytes | str, retain: bool = True, qos: int = QOS
    ) -> None:
        client = self._client
        if client is not None:
            client.publish(topic, payload, qos=qos, retain=retain)

    def _publish_all(self) -> None:
        self._refresh_stations()
        self._refresh_outputs()
        image = self._model.slide
        self._slide_type = None if image is None else image.content_type
        if self._settings.discovery:
            for config in self._configs(self._settings):
                self._publish_config(config)
        self._publish(self._topic("available"), "online")
        for name in SECTIONS:
            self._publish_section(name)
        self._publish(self._topic("station"), self._station_label())
        self._publish(self._topic("output"), self._output_label())
        self._publish_slide()

    def _publish_section(self, name: str) -> None:
        self._publish(self._topic(f"state/{name}"), _json(self._model.section(name)))

    def _publish_slide(self) -> None:
        image = self._model.slide
        if image is not None and image.content_type != self._slide_type:
            # Home Assistant takes the MIME type from the config, not from the bytes.
            self._slide_type = image.content_type
            if self._settings.discovery:
                self._publish_config(self._slide_config())
        self._publish(self._topic("slide"), b"" if image is None else image.data)

    def _publish_result(self, command: str, arguments: object, message: object) -> None:
        result = {
            "command": command,
            # Not the whole of an image sent with a notification.
            "arguments": brief_arguments(arguments),
            "ok": message is None,
            "message": message,
        }
        self._publish(self._topic("result"), _json(result), retain=False)

    def _publish_config(self, config: tuple[str, dict]) -> None:
        topic, payload = config
        self._publish(topic, _json(payload))

    def _withdraw(self, before: MqttSettings, after: MqttSettings) -> None:
        """Clears what stays retained at the broker under settings no longer in use:
        the entities, if discovery goes off or its topics move, and the state, if the
        prefix changes."""
        topics_move = (
            before.prefix != after.prefix or before.discovery_prefix != after.discovery_prefix
        )
        if before.discovery and (not after.discovery or topics_move):
            for topic, _payload in self._configs(before):
                self._publish(topic, b"")
        if before.prefix != after.prefix:
            for name in SECTIONS:
                self._publish(self._topic(f"state/{name}", before), b"")
            for suffix in ("slide", "station", "output", "available"):
                self._publish(self._topic(suffix, before), b"")

    # --- The stations and outputs by name -----------------------------------------

    def _refresh_stations(self) -> bool:
        """The stations `set/station` takes by name: the favourites, then the services
        of the ensemble on display; a name used twice carries its channel. Whether the
        names changed."""
        model = self._model
        entries: list[tuple[str, str, str]] = [
            (entry["name"], entry["channel"], entry["sid"])
            for entry in model.section("favourites")["list"]
        ]
        known = {(channel, sid) for _, channel, sid in entries}
        services = model.section("services")
        channel = services["channel"]
        if channel is not None:
            for service in services["list"]:
                if (channel, service["sid"]) not in known:
                    entries.append((service["name"], channel, service["sid"]))
                    known.add((channel, service["sid"]))
        counts = Counter(name for name, _, _ in entries)
        stations: dict[str, tuple[str, str]] = {}
        for name, channel, sid in entries:
            label = name if counts[name] == 1 else f"{name} ({channel})"
            if label in stations:
                label = f"{name} ({channel} {sid})"
            stations[label] = (channel, sid)
        changed = list(stations) != list(self._stations)
        self._stations = stations
        return changed

    def _station_label(self) -> str:
        service = self._model.section("service")
        if service["sid"] is None:
            return ""
        wanted = (self._model.section("services")["channel"], service["sid"])
        for label, station in self._stations.items():
            if station == wanted:
                return label
        return ""

    def _refresh_outputs(self) -> bool:
        outputs = {OUTPUT_AUTOMATIC: ""}
        for output in self._model.section("audio")["outputs"]:
            label = output["name"]
            number = 2
            while label in outputs:
                label = f"{output['name']} ({number})"
                number += 1
            outputs[label] = output["id"]
        changed = list(outputs) != list(self._outputs)
        self._outputs = outputs
        return changed

    def _output_label(self) -> str:
        choice = self._model.section("audio")["output_choice"]
        for label, output_id in self._outputs.items():
            if output_id == choice:
                return label
        return ""

    # --- MQTT Discovery -----------------------------------------------------------

    def _configs(self, settings: MqttSettings) -> list[tuple[str, dict]]:
        """Every entity's discovery config with its topic, under `settings`."""
        state = {name: self._topic(f"state/{name}", settings) for name in SECTIONS}
        return [
            self._station_config(settings),
            self._output_config(settings),
            self._entity(
                "number",
                "volume",
                # The names of the entities in the smart home (MQTT Discovery), English.
                "Volume",
                settings,
                command_topic=self._topic("set/volume", settings),
                state_topic=state["audio"],
                value_template="{{ value_json.volume_percent }}",
                min=0,
                max=100,
                step=1,
                unit_of_measurement="%",
                mode="slider",
                icon="mdi:volume-high",
            ),
            self._switch("muted", "Mute", "audio", "value_json.muted", settings, "mdi:volume-off"),
            self._switch(
                "reception",
                "Reception",
                "receiver",
                "value_json.state in ['connecting', 'receiving']",
                settings,
                "mdi:radio-tower",
            ),
            self._switch(
                "scan",
                "Scan",
                "receiver",
                "value_json.state == 'scanning'",
                settings,
                "mdi:magnify",
            ),
            self._switch(
                "recording",
                "Recording",
                "recorder",
                "value_json.enabled",
                settings,
                "mdi:record-rec",
            ),
            self._switch(
                "announcement_switching",
                "Switch to announcements",
                "announcement",
                "value_json.switching",
                settings,
                "mdi:bullhorn",
            ),
            self._switch(
                "equalizer",
                "Equalizer",
                "audio",
                "value_json.equalizer.enabled",
                settings,
                "mdi:equalizer",
            ),
            self._sensor("service", "Station", "service", "name", settings, "mdi:radio"),
            self._sensor("ensemble", "Ensemble", "receiver", "ensemble", settings),
            self._sensor("channel", "Channel", "receiver", "channel", settings),
            self._sensor("text", "Dynamic Label", "now_playing", "text", settings, "mdi:text"),
            self._sensor("title", "Title", "now_playing", "title", settings, "mdi:music"),
            self._sensor(
                "artist", "Artist", "now_playing", "artist", settings, "mdi:account-music"
            ),
            self._sensor("quality", "Reception", "signal", "quality.text", settings, "mdi:signal"),
            self._sensor("status", "Status", "receiver", "status", settings),
            self._sensor("recorder", "Recorder", "recorder", "message", settings, "mdi:record-rec"),
            self._sensor("codec", "Codec", "service", "codec", settings),
            self._sensor(
                "bitrate",
                "Bit rate",
                "service",
                "bitrate_kbps",
                settings,
                unit_of_measurement="kbit/s",
            ),
            self._slide_config(settings),
            self._entity(
                "notify",
                "notify",
                "Notification",
                settings,
                command_topic=self._topic("set/notify", settings),
                qos=COMMAND_QOS,
                icon="mdi:message-text",
            ),
            self._entity(
                "event",
                "notification_answer",
                "Notification answer",
                settings,
                state_topic=self._topic("notification/answer", settings),
                event_types=list(REASONS),
                # Home Assistant wants the kind as "event_type"; the rest become
                # attributes.
                value_template=(
                    "{{ {'event_type': value_json.reason, 'id': value_json.id, "
                    "'button': value_json.button, 'label': value_json.label} | to_json }}"
                ),
                qos=COMMAND_QOS,
                icon="mdi:gesture-tap-button",
            ),
        ]

    def _station_config(self, settings: MqttSettings | None = None) -> tuple[str, dict]:
        settings = settings or self._settings
        return self._entity(
            "select",
            "station",
            "Station",
            settings,
            command_topic=self._topic("set/station", settings),
            state_topic=self._topic("station", settings),
            options=list(self._stations) or [""],
            icon="mdi:radio",
        )

    def _output_config(self, settings: MqttSettings | None = None) -> tuple[str, dict]:
        settings = settings or self._settings
        return self._entity(
            "select",
            "output",
            "Output",
            settings,
            command_topic=self._topic("set/output", settings),
            state_topic=self._topic("output", settings),
            options=list(self._outputs) or [OUTPUT_AUTOMATIC],
            icon="mdi:speaker",
        )

    def _slide_config(self, settings: MqttSettings | None = None) -> tuple[str, dict]:
        settings = settings or self._settings
        return self._entity(
            "image",
            "slide",
            "Slideshow",
            settings,
            image_topic=self._topic("slide", settings),
            content_type=self._slide_type or "image/jpeg",
        )

    def _switch(
        self, key: str, name: str, section: str, condition: str, settings: MqttSettings, icon: str
    ) -> tuple[str, dict]:
        return self._entity(
            "switch",
            key,
            name,
            settings,
            command_topic=self._topic(f"set/{key}", settings),
            state_topic=self._topic(f"state/{section}", settings),
            value_template="{{ 'ON' if " + condition + " else 'OFF' }}",
            payload_on="ON",
            payload_off="OFF",
            icon=icon,
        )

    def _sensor(
        self,
        key: str,
        name: str,
        section: str,
        field: str,
        settings: MqttSettings,
        icon: str | None = None,
        **extra: Any,
    ) -> tuple[str, dict]:
        if icon is not None:
            extra["icon"] = icon
        return self._entity(
            "sensor",
            key,
            name,
            settings,
            state_topic=self._topic(f"state/{section}", settings),
            value_template="{{ value_json." + field + " or '' }}",
            **extra,
        )

    def _entity(
        self, component: str, key: str, name: str, settings: MqttSettings, **extra: Any
    ) -> tuple[str, dict]:
        topic = f"{settings.discovery_prefix}/{component}/{self._node}/{key}/config"
        payload = {
            "name": name,
            "has_entity_name": True,
            "unique_id": f"{self._node}_{key}",
            "availability_topic": self._topic("available", settings),
            "device": {
                "identifiers": [self._node],
                # The radio as a device in the smart home.
                "name": "DAB radio",
                "manufacturer": "dab-radio",
                "model": "DAB+ Radio",
                "sw_version": self._model.section("receiver")["version"],
            },
            **extra,
        }
        return topic, payload
