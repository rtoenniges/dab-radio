"""The sub-page "Wi-Fi" of the settings: switch the Wi-Fi on and off, see the connection,
choose a network from those in reach or enter a hidden one by name.

A tap on a new network asks for its password on the keyboard (`ui.touch.keyboard`); one
on a saved network offers to connect or to forget it, one on the network connected to
offers to disconnect or to forget it, and both its IP settings (`ui.touch.ip_settings`)
and its frequency band: "Automatic" (the Wi-Fi chooses) or 2.4 GHz or 5 GHz only, chosen
on a sheet like the output of the sound. Saved networks out of reach (a hidden network
is always so) are listed last, so that they can be chosen and forgotten all the same."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QHideEvent, QMouseEvent, QShowEvent
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QScroller,
    QVBoxLayout,
    QWidget,
)

from i18n import join, tr, tr_noop
from net.wifi import (
    BAND_AUTOMATIC,
    BANDS,
    Ipv4Settings,
    Network,
    SavedNetwork,
    WifiManager,
    band_name,
    signal_bars,
    valid_password,
    valid_ssid,
)
from ui.touch.icons import symbol_icon, symbol_pixmap
from ui.touch.ip_settings import IpSettingsDialog
from ui.touch.keyboard import Keyboard
from ui.touch.overlay import ActionSheet
from ui.touch.theme import DARK, THEMES, Theme
from ui.touch.widgets import ElidedLabel, SignalBars, TapWidget, repolish, set_role

ROW_HEIGHT = 64
LOCK_SIZE = 24
REFRESH_ICON = QSize(32, 32)
CHECK_MARK = "✓"

_UNAVAILABLE = tr_noop(
    "Wi-Fi can only be set up here on the Raspberry Pi: it needs NetworkManager with nmcli."
)
_NO_PERMISSION = tr_noop(
    "The radio lacks the permission for Wi-Fi: install the polkit rule from the folder pi/."
)
_PASSWORD_HINT = tr_noop("8 to 63 characters")
_HIDDEN_PASSWORD_HINT = tr_noop("Leave empty if the network is open; otherwise 8 to 63 characters")


@dataclass(frozen=True)
class WifiEntry:
    """A line of the list of networks."""

    ssid: str
    bars: int | None  # None: saved, but not in reach
    locked: bool  # needs a password
    in_use: bool
    saved: bool
    enterprise: bool = False
    manual_ip: bool = False  # its address set by hand
    without_dhcp: bool = False  # took the password, gave no address
    band: str = BAND_AUTOMATIC  # the band its profile is locked to

    @property
    def detail(self) -> str:
        if self.in_use:
            detail = tr("Connected, IP manual") if self.manual_ip else tr("Connected")
        elif self.enterprise:
            return "WPA-Enterprise"
        elif self.bars is None:
            detail = tr("Saved, out of reach")
        elif not self.saved:
            return ""
        elif self.without_dhcp:
            detail = tr("Saved, no address by DHCP")
        else:
            detail = tr("Saved, IP manual") if self.manual_ip else tr("Saved")
        if self.band == BAND_AUTOMATIC:
            return detail
        # Translators: in the list of networks, after "Saved" or "Connected": the network
        # is used on this band only ("5 GHz").
        return join(", ", [detail, tr("{band} only").format(band=band_name(self.band))])


def bands_text(bands: frozenset[str]) -> str:
    """The bands a network is in reach on, in the order of `BANDS`."""
    names = [band_name(band) for band in BANDS if band in bands]
    if len(names) == 2:
        # Translators: two frequency bands, as in "2.4 GHz and 5 GHz".
        return tr("{first} and {second}").format(first=names[0], second=names[1])
    return names[0] if names else ""


def wifi_entries(
    networks: list[Network],
    saved: list[SavedNetwork],
    connected_ssid: str,
    without_dhcp: set[str] | frozenset[str] = frozenset(),
) -> list[WifiEntry]:
    """The network connected to first, then the saved ones in reach, then the others,
    each group by strength (as `networks` comes); saved ones out of reach last, by name.
    `without_dhcp` holds the profiles (UUIDs) of networks that gave no address."""
    profiles = {profile.ssid: profile for profile in saved}

    def entry(ssid: str, bars: int | None, locked: bool, enterprise: bool = False) -> WifiEntry:
        profile = profiles.get(ssid)
        return WifiEntry(
            ssid,
            bars,
            locked,
            # `connected_ssid` rather than the list's mark: the list may be older.
            ssid == connected_ssid,
            profile is not None,
            enterprise,
            profile is not None and profile.ipv4.manual,
            profile is not None and profile.uuid in without_dhcp,
            profile.band if profile is not None else BAND_AUTOMATIC,
        )

    in_reach = [
        entry(n.ssid, signal_bars(n.signal), n.needs_password, n.enterprise) for n in networks
    ]
    in_reach.sort(key=lambda e: (not e.in_use, not e.saved))
    reached = {e.ssid for e in in_reach}
    away = sorted(profiles.keys() - reached, key=str.casefold)
    return in_reach + [entry(ssid, None, False) for ssid in away]


class _NetworkRow(TapWidget):
    """✓ for the network connected to, the name with a line beneath, a padlock where a
    password is needed, and the bars. It lights up while pressed."""

    def __init__(self, entry: WifiEntry, theme: Theme) -> None:
        super().__init__()
        set_role(self, "row")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setMinimumHeight(ROW_HEIGHT)
        self.entry = entry
        mark = set_role(QLabel(), "large")
        mark.setFixedWidth(28)
        if entry.in_use:
            mark.setText(f'<span style="color: {theme.accent}">{CHECK_MARK}</span>')
        name = ElidedLabel()
        name.setText(entry.ssid)
        if entry.bars is None:
            set_role(name, "muted")
        texts = QVBoxLayout()
        texts.setContentsMargins(0, 0, 0, 0)
        texts.setSpacing(0)
        texts.addWidget(name)
        if entry.detail:
            texts.addWidget(set_role(QLabel(entry.detail), "small"))
        lock = QLabel()
        lock.setFixedWidth(LOCK_SIZE)
        if entry.locked:
            lock.setPixmap(
                symbol_pixmap("lock", theme.color("muted"), LOCK_SIZE, self.devicePixelRatioF())
            )
        bars = SignalBars()
        bars.set_theme(theme)
        bars.set_bars(entry.bars)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 12, 4)
        layout.setSpacing(12)
        layout.addWidget(mark)
        layout.addLayout(texts, stretch=1)
        layout.addWidget(lock)
        layout.addWidget(bars)

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


class WifiView(QWidget):
    """Rows like the other sub-pages: "Wi-Fi", "Connection", "Networks"; the list fills
    what the page leaves. `message` (errors, the missing polkit rule) is for the settings
    page to place in its bottom row. Reads the Wi-Fi faster while shown, scanning once
    when opened (`WifiManager.start`); the state it opens with was read at the start of
    the radio (`WifiManager.watch`)."""

    def __init__(self, wifi: WifiManager, label_width: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._wifi = wifi
        self._theme = THEMES[DARK]
        self._keyboard: Keyboard | None = None
        self._sheet: ActionSheet | None = None
        self._ip_settings: IpSettingsDialog | None = None
        self._rows: list[_NetworkRow] = []
        self._shown_entries: list[WifiEntry] | None = None
        self._error = ""

        self.message = set_role(QLabel(), "small")

        self._switch = set_role(QPushButton(tr("Off")), "switch")
        self._switch.setCheckable(True)
        self._switch.toggled.connect(lambda on: self._switch.setText(tr("On") if on else tr("Off")))
        self._switch.clicked.connect(self._on_switch_clicked)

        self._connection = set_role(ElidedLabel(), "large")
        self._connection_bars = SignalBars()
        self._band = set_role(QLabel(), "muted")
        self._address = set_role(QLabel(), "muted")
        connection = QHBoxLayout()
        connection.setSpacing(16)
        connection.addWidget(self._connection, stretch=1)
        connection.addWidget(self._connection_bars)
        connection.addWidget(self._band)
        connection.addWidget(self._address)

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

        self._scan = QPushButton()
        self._scan.setIconSize(REFRESH_ICON)
        self._scan.clicked.connect(self._on_scan_clicked)
        # Translators: the button to enter a hidden network by its name.
        self._hidden = QPushButton(tr("Manual SSID"))
        self._hidden.clicked.connect(self._add_hidden)
        side = QVBoxLayout()
        side.setSpacing(12)
        side.addWidget(self._scan, alignment=Qt.AlignmentFlag.AlignLeft)
        side.addWidget(self._hidden)
        side.addStretch()

        self._content = set_role(QWidget(), "transparent")
        grid = QGridLayout(self._content)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(12)
        for row, name in enumerate((tr("Wi-Fi"), tr("Connection"), tr("Networks"))):
            label = set_role(QLabel(name), "muted")
            label.setFixedWidth(label_width)
            top = Qt.AlignmentFlag.AlignTop if row == 2 else Qt.AlignmentFlag.AlignVCenter
            grid.addWidget(label, row, 0, alignment=top)
        left = Qt.AlignmentFlag.AlignLeft
        grid.addWidget(self._switch, 0, 1, alignment=left)
        grid.addLayout(connection, 1, 1)
        grid.addWidget(self._list_panel, 2, 1)
        grid.addLayout(side, 2, 2)
        grid.setRowStretch(2, 1)
        grid.setColumnStretch(1, 1)
        self._unavailable = set_role(QLabel(tr(_UNAVAILABLE)), "muted")
        self._unavailable.setWordWrap(True)
        self._unavailable.setAlignment(Qt.AlignmentFlag.AlignTop)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._content, stretch=1)
        layout.addWidget(self._unavailable, stretch=1)

        wifi.changed.connect(self._refresh)
        wifi.failed.connect(self._on_failed)
        self.apply_theme(self._theme)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        self._scan.setIcon(symbol_icon("refresh", theme.color("text"), REFRESH_ICON))
        self._connection_bars.set_theme(theme)
        if self._keyboard is not None:
            self._keyboard.apply_theme(theme)
        self._shown_entries = None  # the rows draw in the colours of the scheme
        self._refresh()

    def keyboard(self) -> Keyboard:
        """The keyboard over the whole window, made on first use."""
        if self._keyboard is None:
            self._keyboard = Keyboard(self.window())
            self._keyboard.apply_theme(self._theme)
        return self._keyboard

    def sheet(self) -> ActionSheet:
        if self._sheet is None:
            self._sheet = ActionSheet(self.window())
        return self._sheet

    def ip_settings(self) -> IpSettingsDialog:
        """The dialog over the whole window, made on first use."""
        if self._ip_settings is None:
            self._ip_settings = IpSettingsDialog(self.window())
        return self._ip_settings

    def entries(self) -> list[WifiEntry]:
        return list(self._shown_entries or [])

    def rows(self) -> list[_NetworkRow]:
        return list(self._rows)

    # --- Showing ----------------------------------------------------------------------

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._wifi.start()

    def hideEvent(self, event: QHideEvent) -> None:
        super().hideEvent(event)
        self._wifi.stop()

    def _refresh(self) -> None:
        wifi = self._wifi
        self._content.setVisible(wifi.available)
        self._unavailable.setVisible(not wifi.available)
        busy = bool(wifi.activity)
        self._switch.setChecked(wifi.enabled)
        self._switch.setEnabled(not busy and wifi.device is not None)
        self._scan.setEnabled(not busy and wifi.enabled)
        self._hidden.setEnabled(not busy and wifi.enabled and wifi.device is not None)
        self._list_panel.setEnabled(not busy)
        self._show_connection()

        if not wifi.enabled or wifi.device is None:
            entries = []
        else:
            entries = wifi_entries(
                wifi.networks, wifi.saved, wifi.connected_ssid, frozenset(wifi.without_dhcp)
            )
        if entries != self._shown_entries:
            self._fill_list(entries)
        if wifi.device is None:
            self._empty.setText(tr("No Wi-Fi device found."))
        elif not wifi.enabled:
            self._empty.setText(tr("Wi-Fi is off."))
        else:
            self._empty.setText(tr("No networks found."))
        self._area.setVisible(bool(entries))
        self._empty.setVisible(not entries)
        self._show_message()

    def _show_connection(self) -> None:
        wifi = self._wifi
        ssid = wifi.connected_ssid
        connected = bool(ssid) and not wifi.activity
        if wifi.activity:
            text = wifi.activity
        elif ssid:
            text = ssid
        elif wifi.device is not None and wifi.device.connecting:
            text = (
                tr("Connecting to “{name}” …").format(name=wifi.device.connection)
                if wifi.device.connection
                else tr("Connecting …")
            )
        elif not wifi.enabled:
            text = tr("Wi-Fi off")
        else:
            text = tr("Not connected")
        self._connection.setText(text)
        set_role(self._connection, "large" if connected or wifi.activity else "muted")
        repolish(self._connection)
        signal = next((n.signal for n in wifi.networks if n.ssid == ssid), None)
        self._connection_bars.set_bars(signal_bars(signal) if signal is not None else None)
        self._connection_bars.setVisible(connected)
        band = wifi.connected_band
        self._band.setText(band_name(band) if band != BAND_AUTOMATIC else "")
        self._band.setVisible(connected and band != BAND_AUTOMATIC)
        self._address.setText(wifi.address)
        self._address.setVisible(connected and bool(wifi.address))

    def _fill_list(self, entries: list[WifiEntry]) -> None:
        for row in self._rows:
            self._list_layout.removeWidget(row)
            row.deleteLater()
        self._rows = []
        for entry in entries:
            row = _NetworkRow(entry, self._theme)
            row.tapped.connect(lambda r=row: self._on_row_tapped(r.entry))
            self._list_layout.insertWidget(self._list_layout.count() - 1, row)
            self._rows.append(row)
        self._shown_entries = entries

    def _show_message(self) -> None:
        if self._error:
            text, role = self._error, "error"
        elif self._wifi.permissions_missing:
            text, role = tr(_NO_PERMISSION), "small"
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
        self._wifi.set_enabled(on)

    def _on_scan_clicked(self) -> None:
        self._clear_error()
        self._wifi.scan()

    def _on_row_tapped(self, entry: WifiEntry) -> None:
        wifi = self._wifi
        if wifi.activity:
            return
        self._clear_error()
        profile = wifi.saved_network(entry.ssid)
        if entry.in_use:
            actions = [(tr("Disconnect"), "accent", wifi.disconnect)]
            if profile is not None:
                actions[:0] = self._profile_actions(profile)
            if not wifi.address:
                detail = tr("Connected")
            elif profile is not None and profile.ipv4.manual:
                detail = tr("Connected, {address}, IP manual").format(address=wifi.address)
            else:
                detail = tr("Connected, {address} by DHCP").format(address=wifi.address)
            self.sheet().open(entry.ssid, detail, actions)
        elif profile is not None:
            self.sheet().open(
                entry.ssid,
                entry.detail,
                [
                    *self._profile_actions(profile),
                    (tr("Connect"), "accent", lambda: wifi.connect_saved(profile)),
                ],
            )
        elif entry.enterprise:
            self._on_failed(tr("WPA-Enterprise (802.1X) cannot be set up here."))
        elif entry.locked:
            self.keyboard().open(
                tr("Password for “{ssid}”").format(ssid=entry.ssid),
                lambda password: wifi.connect_new(entry.ssid, password),
                accept_text=tr("Connect"),
                secret=True,
                check=valid_password,
                hint=tr(_PASSWORD_HINT),
            )
        else:
            wifi.connect_new(entry.ssid)

    def _profile_actions(self, profile: SavedNetwork) -> list[tuple[str, str, Callable[[], None]]]:
        """What the sheet of a saved network offers besides connecting or disconnecting."""
        wifi = self._wifi
        return [
            (tr("Forget"), "danger", lambda: wifi.forget(profile)),
            (tr("IP settings"), "", lambda: self._open_ip_settings(profile)),
            (tr("Frequency band"), "", lambda: self._choose_band(profile)),
        ]

    def _choose_band(self, profile: SavedNetwork) -> None:
        """Offers "Automatic" or one band only, the choice marked, as for the output of
        the sound; beneath, the band connected over and those the network is in reach
        on, so that a band it does not send on is not locked to by mistake."""
        wifi = self._wifi
        network = next((n for n in wifi.networks if n.ssid == profile.ssid), None)
        active = wifi.active_profile()
        lines = []
        if active is not None and active.uuid == profile.uuid and wifi.connected_band:
            lines.append(tr("Connected over {band}.").format(band=band_name(wifi.connected_band)))
        if network is None:
            lines.append(tr("Out of reach."))
        elif network.bands:
            lines.append(tr("In reach on {bands}.").format(bands=bands_text(network.bands)))
        actions = [
            (
                band_name(band),
                "accent" if band == profile.band else "",
                lambda chosen=band: self._set_band(profile, chosen),
            )
            for band in BANDS
        ]
        title = tr("Frequency band for “{name}”").format(name=profile.ssid)
        self.sheet().open(title, join(" ", lines), actions)

    def _set_band(self, profile: SavedNetwork, band: str) -> None:
        self._clear_error()
        self._wifi.set_band(profile, band)

    def _open_ip_settings(self, profile: SavedNetwork) -> None:
        """What its connection got, from the router or as set, goes along where the
        profile is the one up now."""
        active = self._wifi.active_profile()
        is_active = active is not None and active.uuid == profile.uuid
        self.ip_settings().open(
            profile,
            self._wifi.active_ipv4 if is_active else None,
            lambda settings: self._apply_ip_settings(profile, settings),
        )

    def _apply_ip_settings(self, profile: SavedNetwork, settings: Ipv4Settings) -> None:
        self._clear_error()
        self._wifi.apply_ipv4(profile, settings)

    def _add_hidden(self) -> None:
        """First the name, then the password: a hidden network does not say whether it
        needs one."""
        self._clear_error()
        self.keyboard().open(
            tr("Name (SSID) of the hidden network"),
            self._ask_hidden_password,
            # Translators: on to the next step, the password.
            accept_text=tr("Continue"),
            check=valid_ssid,
            hint=tr("1 to 32 characters"),
        )

    def _ask_hidden_password(self, ssid: str) -> None:
        self.keyboard().open(
            tr("Password for “{ssid}”").format(ssid=ssid),
            lambda password: self._wifi.connect_new(ssid, password, hidden=True),
            accept_text=tr("Connect"),
            secret=True,
            check=lambda password: not password or valid_password(password),
            hint=tr(_HIDDEN_PASSWORD_HINT),
        )
