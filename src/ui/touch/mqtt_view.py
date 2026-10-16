"""The sub-page "MQTT" of the settings: the radio at the broker of a smart home
(`net.mqtt`), as rows on three cards (`ui.touch.cards`): on or off with the connection
as it stands beneath, the broker's address and port, user and password; TLS and whether
the certificate is verified; the topic prefix, and whether the entities are announced
through MQTT Discovery (the scheme of Home Assistant, which openHAB, Domoticz and others
read as well) and under which prefix.

Texts come in over the keyboard (`ui.touch.keyboard`), the port over the number pad.
Every change is applied at once (`MqttBridge.apply`); the connection is then made anew.
Switching TLS on moves the port from 1883 to 8883 and back, unless another is set.
"""

from __future__ import annotations

from dataclasses import replace

from PyQt6.QtWidgets import QVBoxLayout, QWidget

from i18n import tr, tr_noop
from net.mqtt import (
    DEFAULT_PORT,
    DEFAULT_TLS_PORT,
    MqttBridge,
    MqttStatus,
    clean_topic_part,
)
from ui.touch.cards import Card, Toggle, switch_row, value_row
from ui.touch.keyboard import Keyboard
from ui.touch.number_pad import NumberPad
from ui.touch.theme import DARK, THEMES, Theme

PORT_RANGE = (1, 65535)
# Translators: no broker address entered yet.
NO_HOST = tr_noop("Not set")
NO_USER = tr_noop("No login")
NO_PASSWORD = tr_noop("No password")
PASSWORD_DOTS = "••••••••"
_BROKER_HINT = tr_noop(
    "The radio connects to the broker and opens no port itself; who may connect and "
    "whether it is encrypted is set at the broker."
)
_PASSWORD_HINT = tr_noop("Stored in plain text in the radio's settings.")
_DISCOVERY_HINT = tr_noop(
    "Discovery announces the radio in the scheme of Home Assistant, which openHAB, "
    "Domoticz and others read as well: stations, volume, Dynamic Label, slideshow and "
    "more appear there by themselves."
)


def valid_host(text: str) -> bool:
    """A host name or address: something, without blanks."""
    stripped = text.strip()
    return bool(stripped) and " " not in stripped


def valid_prefix(text: str) -> bool:
    """A topic prefix as `net.mqtt.clean_topic_part` would leave it."""
    stripped = text.strip()
    return bool(stripped) and clean_topic_part(stripped, "") == stripped


class MqttView(QWidget):
    """The cards "MQTT" (the switch with the connection beneath, broker, port, user name,
    password), "Encryption" (TLS, "Verify certificate") and "Topics" (the prefix,
    Discovery with its prefix)."""

    def __init__(self, bridge: MqttBridge, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bridge = bridge
        self._theme = THEMES[DARK]
        self._keyboard: Keyboard | None = None
        self._number_pad: NumberPad | None = None

        self._switch = Toggle()
        self._switch.clicked.connect(lambda on: self._set(enabled=on))
        self._mqtt_row = switch_row("MQTT", self._switch)
        self._status = self._mqtt_row.subtitle  # the connection as it stands

        self._host = value_row(tr("Broker"), subtitle=tr(_BROKER_HINT))
        self._host.tapped.connect(self._edit_host)
        self._port = value_row(tr("Port"))
        self._port.tapped.connect(self._edit_port)
        self._username = value_row(tr("User name"))
        self._username.tapped.connect(self._edit_username)
        self._password = value_row(tr("Password"), subtitle=tr(_PASSWORD_HINT))
        self._password.tapped.connect(self._edit_password)

        self._tls = Toggle()
        self._tls.clicked.connect(self._on_tls_clicked)
        self._tls_verify = Toggle()
        self._tls_verify.clicked.connect(lambda on: self._set(tls_verify=on))
        self._tls_verify_row = switch_row(tr("Verify certificate"), self._tls_verify)

        self._prefix = value_row(tr("Topic prefix"))
        self._prefix.tapped.connect(self._edit_prefix)
        self._discovery = Toggle()
        self._discovery.clicked.connect(lambda on: self._set(discovery=on))
        self._discovery_prefix = value_row(tr("Discovery prefix"))
        self._discovery_prefix.tapped.connect(self._edit_discovery_prefix)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(
            Card(
                "MQTT",
                [self._mqtt_row, self._host, self._port, self._username, self._password],
            )
        )
        layout.addWidget(
            Card(tr("Encryption"), [switch_row("TLS", self._tls), self._tls_verify_row])
        )
        layout.addWidget(
            Card(
                "Topics",
                [
                    self._prefix,
                    switch_row("Discovery", self._discovery, tr(_DISCOVERY_HINT)),
                    self._discovery_prefix,
                ],
            )
        )
        layout.addStretch()

        bridge.status_changed.connect(self._show_status)
        # Settings may change elsewhere too (a transport of its own one day).
        bridge.settings_changed.connect(lambda _settings: self._show_settings())
        self._show_settings()
        self._show_status(bridge.status)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        if self._keyboard is not None:
            self._keyboard.apply_theme(theme)

    def keyboard(self) -> Keyboard:
        """The keyboard over the whole window, made on first use."""
        if self._keyboard is None:
            self._keyboard = Keyboard(self.window())
            self._keyboard.apply_theme(self._theme)
        return self._keyboard

    def number_pad(self) -> NumberPad:
        """The pad over the whole window, made on first use."""
        if self._number_pad is None:
            self._number_pad = NumberPad(self.window())
        return self._number_pad

    # --- Showing ------------------------------------------------------------------

    def _show_settings(self) -> None:
        settings = self._bridge.settings
        self._switch.setChecked(settings.enabled)
        self._host.setText(settings.host or tr(NO_HOST))
        self._host.set_placeholder(not settings.host)
        self._port.setText(str(settings.port))
        self._username.setText(settings.username or tr(NO_USER))
        self._username.set_placeholder(not settings.username)
        self._password.setText(PASSWORD_DOTS if settings.password else tr(NO_PASSWORD))
        self._password.set_placeholder(not settings.password)
        self._tls.setChecked(settings.tls)
        self._tls_verify.setChecked(settings.tls_verify)
        self._tls_verify_row.setVisible(settings.tls)
        self._prefix.setText(settings.prefix)
        self._discovery.setChecked(settings.discovery)
        self._discovery_prefix.setText(settings.discovery_prefix)
        self._discovery_prefix.setVisible(settings.discovery)

    def _show_status(self, status: MqttStatus) -> None:
        self._mqtt_row.set_subtitle(status.message)

    # --- Changing -----------------------------------------------------------------

    def _set(self, **changes: object) -> None:
        self._bridge.apply(replace(self._bridge.settings, **changes))
        self._show_settings()

    def _on_tls_clicked(self, on: bool) -> None:
        """TLS brokers listen on 8883 as a rule: the default port follows the switch."""
        port = self._bridge.settings.port
        if on and port == DEFAULT_PORT:
            port = DEFAULT_TLS_PORT
        elif not on and port == DEFAULT_TLS_PORT:
            port = DEFAULT_PORT
        self._set(tls=on, port=port)

    def _edit_host(self) -> None:
        self.keyboard().open(
            tr("Broker address"),
            lambda text: self._set(host=text.strip()),
            text=self._bridge.settings.host,
            check=valid_host,
            hint=tr("Name or IP address, without port"),
        )

    def _edit_port(self) -> None:
        self.number_pad().open(
            tr("Broker port"),
            self._bridge.settings.port,
            0,
            "",
            PORT_RANGE,
            lambda value: self._set(port=int(value)),
        )

    def _edit_username(self) -> None:
        self.keyboard().open(
            tr("User name at the broker"),
            lambda text: self._set(username=text.strip()),
            text=self._bridge.settings.username,
            check=lambda _text: True,
            hint=tr("Leave empty if the broker needs no login"),
        )

    def _edit_password(self) -> None:
        self.keyboard().open(
            tr("Password at the broker"),
            lambda text: self._set(password=text),
            text=self._bridge.settings.password,
            secret=True,
            check=lambda _text: True,
            hint=tr("Leave empty if the broker needs none"),
        )

    def _edit_prefix(self) -> None:
        self.keyboard().open(
            tr("Topic prefix of the radio"),
            lambda text: self._set(prefix=text.strip()),
            text=self._bridge.settings.prefix,
            check=valid_prefix,
            hint=tr("Without spaces, + and #; default dabradio"),
        )

    def _edit_discovery_prefix(self) -> None:
        self.keyboard().open(
            tr("Discovery prefix"),
            lambda text: self._set(discovery_prefix=text.strip()),
            text=self._bridge.settings.discovery_prefix,
            check=valid_prefix,
            hint=tr("As set in the smart home; default homeassistant"),
        )
