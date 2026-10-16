"""The sub-page "Bluetooth" of the settings: switch the adapter on and off, see the
device connected, pair headphones (or loudspeakers) seen in a search, connect or forget
paired ones.

A tap on a device opens a sheet that offers to forget it and besides: for a device not
paired (just found, or kept by BlueZ without a pairing) to pair and connect it
(headphones pair without a code, see `net.bluetooth`), for a paired one to connect it,
for the one connected to disconnect it. Where the radio plays to is chosen under
"Audio" (`ui.touch.settings_page`): with "Automatic" the headphones connected last.
"Connect" goes through the controller where it decides which device plays
(`ReceiverController.connect_bluetooth`): the adapter carries the audio to one device
at a time, and a loudspeaker without the hands-free profile cannot connect without it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QHideEvent, QMouseEvent, QShowEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QScroller,
    QVBoxLayout,
    QWidget,
)

from i18n import tr, tr_noop
from net.bluetooth import HEADPHONES, PHONE, SPEAKER, BluetoothManager, Device
from ui.touch.cards import Card, Row, Toggle, heading_label, switch_row
from ui.touch.icons import symbol_icon, symbol_pixmap
from ui.touch.overlay import ActionSheet
from ui.touch.theme import DARK, THEMES, Theme
from ui.touch.widgets import ElidedLabel, TapWidget, repolish, set_role

ROW_HEIGHT = 64
KIND_ICON = 28
REFRESH_ICON = QSize(32, 32)
CHECK_MARK = "✓"

_UNAVAILABLE = tr_noop(
    "Bluetooth devices can only be paired here on the Raspberry Pi: it needs BlueZ on the "
    "system bus. On the PC the system pairs them; headphones connected then appear under "
    "“Audio” as an output."
)
_HINT = tr_noop("Put the audio device in pairing mode, then search and tap it.")
_SEARCHING = tr_noop("Searching …")
_NO_SERVICE = tr_noop("The Bluetooth service (bluetoothd) is not running.")
_NO_ADAPTER = tr_noop("No Bluetooth adapter found.")
_OFF = tr_noop("Bluetooth is off.")
_NO_DEVICES = tr_noop("No devices found.")
_KIND_SYMBOLS = {HEADPHONES: "headphones", SPEAKER: "loudspeaker", PHONE: "phone"}


@dataclass(frozen=True)
class DeviceEntry:
    """A line of the list of devices."""

    path: str
    name: str
    kind: str
    connected: bool
    paired: bool
    battery: int | None = None  # percent, where BlueZ tells (`org.bluez.Battery1`)

    @property
    def detail(self) -> str:
        return device_detail(self.connected, self.paired, self.battery)


def device_detail(connected: bool, paired: bool, battery: int | None) -> str:
    """The line beneath a device's name: "Connected, battery 80 %", "Paired" or "" for
    a device just seen."""
    if connected:
        if battery is not None:
            return tr("Connected, battery {percent} %").format(percent=battery)
        return tr("Connected")
    if paired:
        if battery is not None:
            return tr("Paired, battery {percent} %").format(percent=battery)
        return tr("Paired")
    return ""


def device_entries(devices: list[Device]) -> list[DeviceEntry]:
    """The lines of the list, in the order `BluetoothManager.device_list` gives."""
    return [DeviceEntry(d.path, d.name, d.kind, d.connected, d.paired, d.battery) for d in devices]


def kind_symbol(kind: str) -> str:
    """The symbol (`ui.touch.icons`) for what a device is; the Bluetooth rune where it
    is none of headphones, loudspeaker or phone."""
    return _KIND_SYMBOLS.get(kind, "bluetooth")


def battery_text(device: Device | None) -> str:
    if device is None or device.battery is None:
        return ""
    return tr("Battery {percent} %").format(percent=device.battery)


class _DeviceRow(TapWidget):
    """✓ for the device connected, the name with a line beneath, and the symbol of what
    it is. It lights up while pressed."""

    def __init__(self, entry: DeviceEntry, theme: Theme) -> None:
        super().__init__()
        set_role(self, "row")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setMinimumHeight(ROW_HEIGHT)
        self.entry = entry
        mark = set_role(QLabel(), "large")
        mark.setFixedWidth(28)
        if entry.connected:
            mark.setText(f'<span style="color: {theme.accent}">{CHECK_MARK}</span>')
        name = ElidedLabel()
        name.setText(entry.name)
        texts = QVBoxLayout()
        texts.setContentsMargins(0, 0, 0, 0)
        texts.setSpacing(0)
        texts.addWidget(name)
        if entry.detail:
            texts.addWidget(set_role(QLabel(entry.detail), "small"))
        kind = QLabel()
        kind.setFixedWidth(KIND_ICON)
        kind.setPixmap(
            symbol_pixmap(
                kind_symbol(entry.kind), theme.color("muted"), KIND_ICON, self.devicePixelRatioF()
            )
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 12, 4)
        layout.setSpacing(12)
        layout.addWidget(mark)
        layout.addLayout(texts, stretch=1)
        layout.addWidget(kind)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._set_pressed(True)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._set_pressed(False)
        super().mouseReleaseEvent(event)

    def _set_pressed(self, pressed: bool) -> None:
        self.setProperty("pressed", pressed)
        repolish(self)


class BluetoothView(QWidget):
    """The card "Bluetooth" (the switch, the connection with the device's symbol and
    battery) and the list "Devices" beneath, which fills what the page leaves, with the
    button to search under it. `message` (errors, the hint how to pair) is for the
    settings page to place in its bottom row. Reads BlueZ only while shown."""

    def __init__(
        self,
        bluetooth: BluetoothManager,
        parent: QWidget | None = None,
        connect: Callable[[Device], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._bluetooth = bluetooth
        # What "Connect" does: the controller's, else the manager's own.
        self._connect = connect if connect is not None else bluetooth.connect_device
        self._theme = THEMES[DARK]
        self._sheet: ActionSheet | None = None
        self._rows: list[_DeviceRow] = []
        self._shown_entries: list[DeviceEntry] | None = None
        self._error = ""

        self.message = set_role(QLabel(), "small")

        self._switch = Toggle()
        self._switch.clicked.connect(self._on_switch_clicked)

        # The connection: the device's name as the value, its symbol beside, the
        # battery beneath the title.
        self._kind = QLabel()
        self._kind.setFixedWidth(KIND_ICON)
        self._connection_row = Row(tr("Connection"), control=self._kind, value="")
        self._connection = self._connection_row.value
        self._battery = self._connection_row.subtitle

        self._list_layout = QVBoxLayout()
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(0)
        self._list_layout.addStretch()
        rows = set_role(QWidget(), "transparent")
        rows.setLayout(self._list_layout)
        self._area = set_role(QScrollArea(), "transparent")
        self._area.setWidget(rows)
        self._area.setWidgetResizable(True)
        self._area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        set_role(self._area.viewport(), "transparent")
        QScroller.grabGesture(
            self._area.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture
        )
        self._empty = set_role(QLabel(), "muted")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._list_panel = set_role(QWidget(), "panel")
        self._list_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        panel_layout = QVBoxLayout(self._list_panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.addWidget(self._area, stretch=1)
        panel_layout.addWidget(self._empty, stretch=1)
        devices = set_role(QWidget(), "transparent")
        devices_layout = QVBoxLayout(devices)
        devices_layout.setContentsMargins(0, 0, 0, 0)
        devices_layout.setSpacing(6)
        devices_layout.addWidget(heading_label(tr("Devices")))
        devices_layout.addWidget(self._list_panel, stretch=1)

        self._scan = QPushButton()
        self._scan.setIconSize(REFRESH_ICON)
        self._scan.clicked.connect(self._on_scan_clicked)
        buttons = QHBoxLayout()
        buttons.setSpacing(12)
        buttons.addWidget(self._scan)
        buttons.addStretch()

        self._content = set_role(QWidget(), "transparent")
        content = QVBoxLayout(self._content)
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(12)
        content.addWidget(
            Card("Bluetooth", [switch_row("Bluetooth", self._switch), self._connection_row])
        )
        content.addWidget(devices, stretch=1)
        content.addLayout(buttons)
        self._unavailable = set_role(QLabel(tr(_UNAVAILABLE)), "muted")
        self._unavailable.setWordWrap(True)
        self._unavailable.setAlignment(Qt.AlignmentFlag.AlignTop)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._content, stretch=1)
        layout.addWidget(self._unavailable, stretch=1)

        bluetooth.changed.connect(self._refresh)
        bluetooth.failed.connect(self._on_failed)
        self.apply_theme(self._theme)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        self._scan.setIcon(symbol_icon("refresh", theme.color("text"), REFRESH_ICON))
        self._shown_entries = None  # the rows draw in the colours of the scheme
        self._refresh()

    def sheet(self) -> ActionSheet:
        if self._sheet is None:
            self._sheet = ActionSheet(self.window())
        return self._sheet

    def entries(self) -> list[DeviceEntry]:
        return list(self._shown_entries or [])

    def rows(self) -> list[_DeviceRow]:
        return list(self._rows)

    # --- Showing ----------------------------------------------------------------------

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._bluetooth.start()

    def hideEvent(self, event: QHideEvent) -> None:
        super().hideEvent(event)
        self._bluetooth.stop()

    def _refresh(self) -> None:
        bluetooth = self._bluetooth
        self._content.setVisible(bluetooth.available)
        self._unavailable.setVisible(not bluetooth.available)
        busy = bool(bluetooth.activity)
        ready = bluetooth.service_present and bluetooth.adapter is not None
        self._switch.setChecked(bluetooth.powered)
        self._switch.setEnabled(ready and not busy)
        self._scan.setEnabled(
            ready and bluetooth.powered and not busy and not bluetooth.discovering
        )
        self._list_panel.setEnabled(not busy)
        self._show_connection()

        entries = device_entries(bluetooth.device_list) if ready and bluetooth.powered else []
        if entries != self._shown_entries:
            self._fill_list(entries)
        if not bluetooth.service_present:
            self._empty.setText(tr(_NO_SERVICE))
        elif bluetooth.adapter is None:
            self._empty.setText(tr(_NO_ADAPTER))
        elif not bluetooth.powered:
            self._empty.setText(tr(_OFF))
        elif bluetooth.discovering:
            self._empty.setText(tr(_SEARCHING))
        else:
            self._empty.setText(tr(_NO_DEVICES))
        self._area.setVisible(bool(entries))
        self._empty.setVisible(not entries)
        self._show_message()

    def _show_connection(self) -> None:
        bluetooth = self._bluetooth
        # Off, BlueZ drops every connection; the objects may still say otherwise.
        device = bluetooth.connected_device if bluetooth.powered else None
        connected = device is not None and not bluetooth.activity
        if bluetooth.activity:
            text = bluetooth.activity
        elif device is not None:
            text = device.name
        elif not bluetooth.powered:
            text = tr("Bluetooth off")
        else:
            text = tr("Not connected")
        self._connection.setText(text)
        set_role(self._connection, "" if connected or bluetooth.activity else "muted")
        repolish(self._connection)
        if connected:
            self._kind.setPixmap(
                symbol_pixmap(
                    kind_symbol(device.kind),
                    self._theme.color("muted"),
                    KIND_ICON,
                    self.devicePixelRatioF(),
                )
            )
        self._kind.setVisible(connected)
        self._battery.setText(battery_text(device))
        self._battery.setVisible(connected and bool(self._battery.text()))

    def _fill_list(self, entries: list[DeviceEntry]) -> None:
        for row in self._rows:
            self._list_layout.removeWidget(row)
            row.deleteLater()
        self._rows = []
        for entry in entries:
            row = _DeviceRow(entry, self._theme)
            row.tapped.connect(lambda r=row: self._on_row_tapped(r.entry))
            self._list_layout.insertWidget(self._list_layout.count() - 1, row)
            self._rows.append(row)
        self._shown_entries = entries

    def _show_message(self) -> None:
        bluetooth = self._bluetooth
        if self._error:
            text, role = self._error, "error"
        elif bluetooth.discovering:
            text, role = tr(_SEARCHING), "small"
        elif bluetooth.available and bluetooth.powered:
            text, role = tr(_HINT), "small"
        else:
            text, role = "", "small"
        self.message.setText(text)
        if self.message.property("role") != role:
            set_role(self.message, role)
            repolish(self.message)

    def _on_failed(self, message: str) -> None:
        self._error = message
        self._show_message()

    def _clear_error(self) -> None:
        self._error = ""
        self._show_message()

    # --- Actions ----------------------------------------------------------------------

    def _on_switch_clicked(self, on: bool) -> None:
        self._clear_error()
        self._bluetooth.set_powered(on)

    def _on_scan_clicked(self) -> None:
        self._clear_error()
        self._bluetooth.scan()

    def _on_row_tapped(self, entry: DeviceEntry) -> None:
        bluetooth = self._bluetooth
        device = bluetooth.device(entry.path)
        if bluetooth.activity or device is None:
            return
        self._clear_error()
        # The device has been found: the search ends, as it takes radio time from the
        # headphones playing (`BluetoothManager.stop_scan`).
        bluetooth.stop_scan()
        # Every sheet offers "Forget": BlueZ keeps a device it was connected to even
        # without a pairing (one paired without bonding, as without an agent), and such a
        # device would otherwise stay in the list for good.
        forget = (tr("Forget"), "danger", lambda: bluetooth.forget(device))
        if device.connected:
            detail = device_detail(True, device.paired, device.battery)
            action = (tr("Disconnect"), "accent", lambda: bluetooth.disconnect_device(device))
        elif device.paired:
            detail = tr("Paired")
            action = (tr("Connect"), "accent", lambda: self._connect(device))
        else:
            detail = tr("Not paired")
            action = (tr("Pair"), "accent", lambda: bluetooth.pair(device))
        self.sheet().open(device.name, detail, [forget, action])
