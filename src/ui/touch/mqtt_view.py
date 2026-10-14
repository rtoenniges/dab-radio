"""The sub-page "MQTT" of the settings: the radio at the broker of a smart home
(`net.mqtt`): on or off, the connection as it stands, the broker's address and port,
user and password, TLS, the topic prefix, and whether the entities are announced
through MQTT Discovery (the scheme of Home Assistant, which openHAB, Domoticz and
others read as well) and under which prefix.

Texts come in over the keyboard (`ui.touch.keyboard`), the port over the number pad.
Every change is applied at once (`MqttBridge.apply`); the connection is then made anew.
Switching TLS on moves the port from 1883 to 8883 and back, unless another is set.
"""

from __future__ import annotations

from dataclasses import replace

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QWidget

from i18n import tr, tr_noop
from net.mqtt import (
    DEFAULT_PORT,
    DEFAULT_TLS_PORT,
    MqttBridge,
    MqttStatus,
    clean_topic_part,
)
from ui.touch.keyboard import Keyboard
from ui.touch.number_pad import NumberPad
from ui.touch.theme import DARK, THEMES, Theme
from ui.touch.widgets import ElidedLabel, set_role

PORT_RANGE = (1, 65535)
# Translators: no broker address entered yet.
NO_HOST = tr_noop("Not set")
NO_USER = tr_noop("No login")
NO_PASSWORD = tr_noop("No password")
PASSWORD_DOTS = "••••••••"
_HINT = tr_noop(
    "The radio connects to the broker and opens no port itself; who may connect and "
    "whether it is encrypted is set at the broker. The password is stored in plain text "
    "in the radio's settings."
)
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


def _switch() -> QPushButton:
    switch = set_role(QPushButton(tr("Off")), "switch")
    switch.setCheckable(True)
    switch.toggled.connect(lambda on: switch.setText(tr("On") if on else tr("Off")))
    return switch


class MqttView(QWidget):
    """Rows like the other sub-pages: "MQTT", "Connection", "Broker", "Login",
    "TLS", "Discovery" and "Topic prefix", each hint beneath its row; what the
    page cannot show at once lies below the fold, the essentials above."""

    def __init__(self, bridge: MqttBridge, label_width: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bridge = bridge
        self._theme = THEMES[DARK]
        self._keyboard: Keyboard | None = None
        self._number_pad: NumberPad | None = None

        self._switch = _switch()
        self._switch.clicked.connect(lambda on: self._set(enabled=on))
        self._status = set_role(ElidedLabel(), "large")

        self._host = set_role(QPushButton(), "value")
        self._host.clicked.connect(self._edit_host)
        self._port = set_role(QPushButton(), "value")
        self._port.clicked.connect(self._edit_port)
        broker = QHBoxLayout()
        broker.setSpacing(8)
        broker.addWidget(self._host)
        broker.addWidget(self._port)
        broker.addStretch()

        self._username = set_role(QPushButton(), "value")
        self._username.clicked.connect(self._edit_username)
        self._password = set_role(QPushButton(), "value")
        self._password.clicked.connect(self._edit_password)
        login = QHBoxLayout()
        login.setSpacing(8)
        login.addWidget(self._username)
        login.addWidget(self._password)
        login.addStretch()

        self._tls = _switch()
        self._tls.clicked.connect(self._on_tls_clicked)
        self._tls_verify = QPushButton(tr("Verify certificate"))
        self._tls_verify.setCheckable(True)
        self._tls_verify.clicked.connect(lambda on: self._set(tls_verify=on))
        tls = QHBoxLayout()
        tls.setSpacing(8)
        tls.addWidget(self._tls)
        tls.addWidget(self._tls_verify)
        tls.addStretch()

        self._prefix = set_role(QPushButton(), "value")
        self._prefix.clicked.connect(self._edit_prefix)

        self._discovery = _switch()
        self._discovery.clicked.connect(lambda on: self._set(discovery=on))
        self._discovery_prefix = set_role(QPushButton(), "value")
        self._discovery_prefix.clicked.connect(self._edit_discovery_prefix)
        discovery = QHBoxLayout()
        discovery.setSpacing(8)
        discovery.addWidget(self._discovery)
        discovery.addWidget(self._discovery_prefix)
        discovery.addStretch()

        hint = set_role(QLabel(tr(_HINT)), "small")
        hint.setWordWrap(True)
        discovery_hint = set_role(QLabel(tr(_DISCOVERY_HINT)), "small")
        discovery_hint.setWordWrap(True)

        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(12)
        rows = (
            ("MQTT", self._switch),
            (tr("Connection"), self._status),
            (tr("Broker"), broker),
            # Translators: user name and password at the broker.
            (tr("Login"), login),
            ("TLS", tls),
            ("Discovery", discovery),
            ("", discovery_hint),
            (tr("Topic prefix"), self._prefix),
            ("", hint),
        )
        for row, (name, content) in enumerate(rows):
            label = set_role(QLabel(name), "muted")
            label.setFixedWidth(label_width)
            grid.addWidget(label, row, 0)
            if isinstance(content, QHBoxLayout):
                grid.addLayout(content, row, 1)
            elif isinstance(content, QPushButton):
                grid.addWidget(content, row, 1, alignment=Qt.AlignmentFlag.AlignLeft)
            else:
                grid.addWidget(content, row, 1)
        grid.setRowStretch(len(rows), 1)
        grid.setColumnStretch(1, 1)

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
        self._port.setText(str(settings.port))
        self._username.setText(settings.username or tr(NO_USER))
        self._password.setText(PASSWORD_DOTS if settings.password else tr(NO_PASSWORD))
        self._tls.setChecked(settings.tls)
        self._tls_verify.setChecked(settings.tls_verify)
        self._tls_verify.setVisible(settings.tls)
        self._prefix.setText(settings.prefix)
        self._discovery.setChecked(settings.discovery)
        self._discovery_prefix.setText(settings.discovery_prefix)
        self._discovery_prefix.setVisible(settings.discovery)

    def _show_status(self, status: MqttStatus) -> None:
        self._status.setText(status.message)

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
