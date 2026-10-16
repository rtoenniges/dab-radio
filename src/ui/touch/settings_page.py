"""The settings page: the categories as a list on the left, each with its symbol and,
where there is one, its state in a line beneath (the device, the output playing, the
network, the headphones, the folder of the recordings, whether the recorder runs, the
broker); on the right the rows of the chosen one on cards (`ui.touch.cards`).

SDR (device with the tuner of the RSPduo or the server of rtl_tcp, gain, frequency
correction, corrections), Display (language, colour scheme, the load in the navigation
rail, the level beside the volume slider of the radio page, the constellation as points
or as a density, when the screen dims and goes off, `ui.touch.backlight`; the
listener's position), Audio (the output, `audio.outputs`; the ceiling of the volume,
`audio.volume`; the equalizer: on or off, the way to it, for all stations or per
station, deleting all its settings; switching to announcements), Wi-Fi
(`ui.touch.wifi_view`), Bluetooth (`ui.touch.bluetooth_view`), Storage (the USB drives
and the directory for the recordings, `ui.touch.storage_view`), Recording
(`ui.touch.recording_view`) and MQTT (the radio at the broker of a smart home,
`ui.touch.mqtt_view`). A setting with several choices shows the one in effect, a tap
opens the sheet with all of them. Sub-pages of rows are swiped up and down where they
are taller than the page. The way out of full screen is in the bottom row, the version
of the radio beside it, and what a sub-page has to say (errors, hints)."""

from __future__ import annotations

from dataclasses import replace

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPainter, QPaintEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QScroller,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from audio.volume import CEILING_RANGE_DB
from dab.gnss import Observer
from i18n import available_languages, join, tr, tr_noop, tr_plural
from net.bluetooth import BluetoothManager
from net.mqtt import MqttBridge
from net.wifi import BAND_AUTOMATIC, WifiManager, band_name
from sdr.rspduo import FREQ_CORRECTION_RANGE_PPM, IF_GAIN_REDUCTION_RANGE_DB, LNA_GAIN_REDUCTION_DB
from sdr.rtl_tcp import PORT_RANGE as RTL_TCP_PORT_RANGE
from storage.volumes import VolumeManager, free_space
from ui.language import restart_question, store_language, stored_language
from ui.receiver_controller import (
    APPLICATION_VERSION,
    DEVICES,
    HEIGHT_RANGE_M,
    LATITUDE_RANGE,
    LONGITUDE_RANGE,
    RSPDUO_TUNERS,
    RTLSDR_PPM_RANGE,
    DeviceSettings,
    ReceiverController,
    ReceptionState,
    rspduo_tuner_name,
)
from ui.touch import cards
from ui.touch.backlight import TIMEOUT_CHOICES_S, ScreenDimmer
from ui.touch.bluetooth_view import BluetoothView, battery_text
from ui.touch.cards import (
    Card,
    ChoiceRow,
    Row,
    Toggle,
    action_row,
    stepper_row,
    switch_row,
    value_row,
)
from ui.touch.icons import symbol_pixmap
from ui.touch.keyboard import Keyboard
from ui.touch.mqtt_view import NO_HOST, MqttView, valid_host
from ui.touch.number_pad import NumberPad, format_number
from ui.touch.overlay import ActionSheet
from ui.touch.recording_view import RecordingView
from ui.touch.storage_view import StorageView, directory_text, free_text
from ui.touch.theme import DARK, LIGHT, THEMES, Theme
from ui.touch.widgets import EdgeFades, ElidedLabel, Stepper, TapWidget, repolish, set_role
from ui.touch.wifi_view import WifiView

# Neither the stick nor the RSPduo takes a new gain while running, so a change during
# reception restarts it (`ReceiverController.restart_reception`); only once the stepper
# has rested this long, not for every step of a held button.
RESTART_DELAY_MS = 1200
# Restyling the whole window for the other colour scheme holds the GUI thread for a
# moment (0.3 s on the Pi, 0.16 s on the PC, measured 2026-10-13 off screen); the
# sheet with the choice closes first, and the scheme follows this much later.
THEME_FEEDBACK_MS = 50

# The list of categories on the left.
LIST_WIDTH = 300
CATEGORY_ICON = 28
CATEGORY_ICON_LEFT = 16
CATEGORY_ROW_HEIGHT = 68
# Between the parts of a state in the list ("WLAN-0815 · 5 GHz").
STATE_SEPARATOR = " · "

_HINT = tr_noop("Changes to the device briefly restart a running reception.")
_AGC_HINT = tr_noop("The stick sets its gain itself.")
_DC_HINT = tr_noop("Removes the DC offset of the samples.")
_IQ_HINT = tr_noop("Balances amplitude and phase of I and Q.")
_LANGUAGE_HINT = tr_noop("Takes effect with the next start.")
_SCREEN_HINT = tr_noop("After the last touch.")
_LOCATION_HINT = tr_noop("For the sky and the corrections of GEPOS.")
_PER_STATION_HINT = tr_noop(
    "Per station, every station keeps its own on/off and sound; "
    "one without a setting of its own sounds like all stations."
)
# The ceiling of the volume (`audio.volume`): what 100 % of the slider gives.
_CEILING_HINT = tr_noop(
    "What 100 % of the slider gives; the whole scale moves with it. "
    "For loudspeakers that take less than the amplifier delivers."
)
# "Automatic" under "Output" (`audio.outputs`): the default of the system.
_OUTPUT_HINT = tr_noop(
    "Automatic follows the system, so headphones as soon as they are connected. "
    "Volume and maximum apply per output."
)
_ANNOUNCEMENT_HINT = tr_noop(
    "For an announcement from another ensemble the radio tunes there and returns afterwards."
)
_NO_BACKLIGHT_HINT = tr_noop(
    "There is no backlight to switch here; "
    "on the Raspberry Pi the setting applies to the Touch Display 2."
)
_BACKLIGHT_PERMISSION_HINT = tr_noop(
    "The radio lacks the permission for the backlight: install the udev rule from the folder pi/."
)
# What the density of the constellation is good for; no tooltip can tell on a touch
# screen (see `ui.touch.signal_panels.ConstellationPanel`).
_DENSITY_HINT = tr_noop(
    "The density counts all carriers of all symbols; at weak reception the four "
    "phases can still be made out in it."
)

# The categories in the order of the list: key, name, symbol (`ui.touch.icons`).
SDR = "sdr"
DISPLAY = "display"
AUDIO = "audio"
WIFI = "wifi"
BLUETOOTH = "bluetooth"
STORAGE = "storage"
RECORDING = "recording"
MQTT = "mqtt"
_CATEGORIES = (
    (SDR, "SDR", "antenna"),
    (DISPLAY, tr_noop("Display"), "screen"),
    (AUDIO, tr_noop("Audio"), "loudspeaker"),
    (WIFI, tr_noop("Wi-Fi"), "wifi"),
    (BLUETOOTH, "Bluetooth", "bluetooth"),
    (STORAGE, tr_noop("Storage"), "usb"),
    (RECORDING, tr_noop("Recording"), "record"),
    (MQTT, "MQTT", "home"),
)
_SUBPAGES = tuple((key, name) for key, name, _symbol in _CATEGORIES)


def _ppm(value: float) -> str:
    return f"{value:+.2f} ppm" if isinstance(value, float) else f"{value:+d} ppm"


def _lna(state: int) -> str:
    reduction = LNA_GAIN_REDUCTION_DB[state]
    return f"{state} (−{reduction} dB)" if reduction else "0 (0 dB)"


def timeout_text(seconds: float) -> str:
    """When the screen dims or goes off, after the row's label: "never", "after 30 s",
    "after 2 min", "after 1 h"."""
    if seconds <= 0:
        # Translators: the screen is never dimmed or switched off.
        return tr("never")
    if seconds < 60:
        return tr("after {seconds:.0f} s").format(seconds=seconds)
    if seconds < 3600:
        return tr("after {minutes:.0f} min").format(minutes=seconds / 60)
    return tr("after {hours:.0f} h").format(hours=seconds / 3600)


def backlight_hint(screen: ScreenDimmer) -> str:
    """Why the times would do nothing here; "" where they work."""
    if screen.backlight is None:
        return tr(_NO_BACKLIGHT_HINT)
    if not screen.backlight.writable:
        return tr(_BACKLIGHT_PERMISSION_HINT)
    return ""


# The position to 0.0001 degrees (some 10 m), the height to the metre.
COORDINATE_DECIMALS = 4


def coordinate_text(value: float, positive: str, negative: str) -> str:
    """A latitude or longitude with its hemisphere: "51.1634° N", "3.7000° W" (the
    decimal separator as the language writes it, `format_number`)."""
    hemisphere = positive if value >= 0 else negative
    return f"{format_number(abs(value), COORDINATE_DECIMALS)}° {hemisphere}"


# --- The states in the list ---------------------------------------------------------------


def sdr_state(settings: DeviceSettings) -> str:
    """The device, with the tuner of the RSPduo or the server of rtl_tcp:
    "RSPduo · Tuner 2", "RTL_TCP · 192.168.2.55"."""
    name = DEVICES[settings.device]
    rspduo = settings.rspduo()
    if rspduo is not None:
        return join(STATE_SEPARATOR, [name, rspduo_tuner_name(rspduo.tuner)])
    if settings.rtl_tcp() is not None:
        return join(STATE_SEPARATOR, [name, settings.rtl_tcp_host or tr(NO_HOST)])
    return name


def audio_state(controller: ReceiverController) -> str:
    """The output playing; "" while none is."""
    in_use = controller.audio_output_in_use
    return in_use.name if in_use is not None else ""


def wifi_state(wifi: WifiManager) -> str:
    """The network and the band: "WLAN-0815 · 5 GHz"; "" where the Wi-Fi cannot be set
    here (a PC)."""
    if not wifi.available:
        return ""
    if wifi.activity:
        return wifi.activity
    ssid = wifi.connected_ssid
    if ssid:
        band = wifi.connected_band
        parts = [ssid, band_name(band)] if band and band != BAND_AUTOMATIC else [ssid]
        return join(STATE_SEPARATOR, parts)
    if wifi.device is not None and wifi.device.connecting:
        return tr("Connecting …")
    if not wifi.enabled:
        return tr("Off")
    return tr("Not connected")


def bluetooth_state(bluetooth: BluetoothManager) -> str:
    """The device connected and its battery: "Jabra Elite 10 · Battery 50 %"; "" where
    Bluetooth cannot be set here (a PC)."""
    if not bluetooth.available:
        return ""
    if bluetooth.activity:
        return bluetooth.activity
    device = bluetooth.connected_device if bluetooth.powered else None
    if device is not None:
        battery = battery_text(device)
        return join(STATE_SEPARATOR, [device.name, battery]) if battery else device.name
    if not bluetooth.powered:
        return tr("Off")
    return tr("Not connected")


def storage_state(controller: ReceiverController, volumes: VolumeManager) -> str:
    """The folder of the recordings and the space left on its drive: "USB-STICK ›
    Aufnahmen · 12 GB free of 31 GB"."""
    directory = controller.recording_directory
    text = directory_text(directory, volumes.volume_of(directory))
    if directory is None:
        return text
    if not controller.recording_directory_available:
        return join(STATE_SEPARATOR, [text, tr("Drive not mounted")])
    space = free_text(free_space(directory))
    return join(STATE_SEPARATOR, [text, space]) if space else text


def recording_state(controller: ReceiverController) -> str:
    """Whether the recorder runs and how many titles the ring keeps: "On · 10 titles"."""
    if not controller.recording_enabled:
        return tr("Off")
    count = controller.ring_size
    titles = tr_plural("{count} title", "{count} titles", count).format(count=count)
    return join(STATE_SEPARATOR, [tr("On"), titles])


def mqtt_state(bridge: MqttBridge | None) -> str:
    """The connection as it stands, as the sub-page tells it."""
    return bridge.status.message if bridge is not None else ""


# --- The list ----------------------------------------------------------------------------


class _CategoryRow(TapWidget):
    """A category: its symbol (painted, in the accent colour while chosen), its name and
    its state beneath. It lights up while pressed."""

    def __init__(self, key: str, name: str, symbol: str) -> None:
        super().__init__()
        self.key = key
        self.symbol = symbol
        set_role(self, "row")
        self.setProperty("selected", False)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setMinimumHeight(CATEGORY_ROW_HEIGHT)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(CATEGORY_ICON_LEFT + CATEGORY_ICON + 14, 0, 12, 0)
        texts = set_role(QWidget(), "transparent")
        column = QVBoxLayout(texts)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        self.name = QLabel(name)
        column.addWidget(self.name)
        self.state = set_role(ElidedLabel(), "small")
        self.state.hide()
        column.addWidget(self.state)
        layout.addWidget(texts, stretch=1)

    def set_state(self, text: str) -> None:
        self.state.setText(text)
        self.state.setVisible(bool(text))

    def set_selected(self, selected: bool) -> None:
        if bool(self.property("selected")) == selected:
            return
        self.setProperty("selected", selected)
        for widget in (self, self.name, self.state):
            repolish(widget)
        self.update()

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

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        color = cards.theme().color("accent" if self.property("selected") else "text")
        pixmap = symbol_pixmap(self.symbol, color, CATEGORY_ICON, self.devicePixelRatioF())
        painter = QPainter(self)
        painter.drawPixmap(CATEGORY_ICON_LEFT, (self.height() - CATEGORY_ICON) // 2, pixmap)
        painter.end()


class CategoryList(QWidget):
    """The categories one below the other; a tap reports the key (`chosen`)."""

    chosen = pyqtSignal(str)

    def __init__(
        self, categories: tuple[tuple[str, str, str], ...], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        set_role(self, "transparent")
        self.setFixedWidth(LIST_WIDTH)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.rows: dict[str, _CategoryRow] = {}
        for key, name, symbol in categories:
            row = _CategoryRow(key, tr(name), symbol)
            row.tapped.connect(lambda k=key: self.chosen.emit(k))
            layout.addWidget(row)
            self.rows[key] = row
        layout.addStretch()
        self._current = ""

    def current(self) -> str:
        return self._current

    def set_current(self, key: str) -> None:
        """Marks `key` as chosen without reporting it."""
        self._current = key
        for row_key, row in self.rows.items():
            row.set_selected(row_key == key)

    def set_state(self, key: str, text: str) -> None:
        self.rows[key].set_state(text)

    def apply_theme(self, theme: Theme) -> None:
        self.update()  # the symbols take their colour from `cards.theme()`


class _OutputRow(ChoiceRow):
    """The outputs to choose from, "Automatic" first (the default of the system); a
    line beneath names the output in use where it is not the one chosen. Made anew as
    outputs come and go (`ReceiverController.audio_outputs_changed`)."""

    def __init__(self, controller: ReceiverController, parent: QWidget | None = None) -> None:
        super().__init__(tr("Output"), [], hint=tr(_OUTPUT_HINT), parent=parent)
        self._controller = controller
        self.chosen.connect(controller.set_audio_output)
        controller.audio_outputs_changed.connect(self.refresh)
        self.refresh()

    def keys(self) -> list[str]:
        """The ids to choose from, "" for "Automatic"."""
        return [key for key, _name in self.choices()]

    def refresh(self) -> None:
        controller = self._controller
        outputs = controller.audio_outputs
        choices = [("", tr("Automatic")), *((output.id, output.name) for output in outputs)]
        details = {output.id: "Bluetooth" for output in outputs if output.bluetooth}
        self.set_choices(choices, details)
        choice = controller.audio_output_choice
        self.set_current(choice)
        in_use = controller.audio_output_in_use
        shown = in_use is not None and (choice == "" or choice != in_use.id)
        self.set_subtitle(tr("currently {name}").format(name=in_use.name) if shown else "")


class SettingsPage(QWidget):
    theme_chosen = pyqtSignal(str)  # DARK or LIGHT
    load_shown_chosen = pyqtSignal(bool)
    volume_figure_shown_chosen = pyqtSignal(bool)
    constellation_density_chosen = pyqtSignal(bool)  # True: density, False: points
    equalizer_requested = pyqtSignal()  # the row "Adjust sound"
    recordings_requested = pyqtSignal()  # "Show recordings" under "Recording"
    quit_requested = pyqtSignal()
    # Start anew in the language chosen (`main.restart`).
    restart_requested = pyqtSignal()

    def __init__(
        self,
        controller: ReceiverController,
        parent: QWidget | None = None,
        wifi: WifiManager | None = None,
        bluetooth: BluetoothManager | None = None,
        screen: ScreenDimmer | None = None,
        volumes: VolumeManager | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        settings = controller.device_settings
        # Only the Raspberry Pi has one to set; elsewhere the sub-page says so.
        self._wifi = wifi if wifi is not None else WifiManager(self)
        # Read right away and kept current in the background, so that the sub-page has
        # everything when it is opened and the window can show whether the Wi-Fi is on.
        self._wifi.watch()
        # BlueZ on the system bus, likewise only on the Raspberry Pi.
        self._bluetooth = bluetooth if bluetooth is not None else BluetoothManager(self)
        # Read right away, not only once the page is shown: the window follows devices
        # connecting (`ui.touch.window.rehide_cursor`), and BlueZ's changes are only
        # taken for devices known. The controller decides which of them plays.
        self._bluetooth.start()
        controller.attach_bluetooth(self._bluetooth)
        # Likewise the backlight; the page made without one only keeps the times.
        self._screen = (
            screen if screen is not None else ScreenDimmer(None, controller.settings, self)
        )
        # The USB drives through UDisks2, likewise only on the Raspberry Pi; read right
        # away, so that a drive plugged in is mounted without a visit to the page.
        self._volumes = volumes if volumes is not None else VolumeManager(self)
        self._volumes.watch()

        # --- SDR: the device, its own rows (only those of the chosen one are shown),
        # and what applies to every device.
        self._device = ChoiceRow(tr("Device"), list(DEVICES.items()), subtitle=tr(_HINT))
        self._device.set_current(settings.device)
        self._device.chosen.connect(self._on_device_chosen)

        # The tuner of the RSPduo with the antenna.
        self._tuner = ChoiceRow(
            tr("Tuner"), [(tuner, rspduo_tuner_name(tuner)) for tuner in RSPDUO_TUNERS]
        )
        self._tuner.set_current(settings.rspduo_tuner)
        self._tuner.chosen.connect(self._on_device_settings_changed)

        # The rtl_tcp server: name or address over the keyboard, the port over the pad.
        self._keyboard: Keyboard | None = None
        self._rtl_tcp_host = settings.rtl_tcp_host
        self._rtl_tcp_port = settings.rtl_tcp_port
        self._server_host = value_row(tr("Server"))
        self._server_host.tapped.connect(self._edit_server_host)
        self._server_port = value_row(tr("Port"))
        self._server_port.tapped.connect(self._edit_server_port)
        self._rtl_tcp_rows = (self._server_host, self._server_port)
        self._show_server()

        self._gain = Stepper(controller.gain_steps_db, lambda gain: f"{gain:.1f} dB")
        self._gain.set_value(settings.gain_db)
        self._agc = Toggle()
        self._agc.setChecked(settings.agc)
        self._agc.toggled.connect(self._gain.setDisabled)
        self._gain.setDisabled(settings.agc)
        self._ppm = Stepper(range(RTLSDR_PPM_RANGE[0], RTLSDR_PPM_RANGE[1] + 1), _ppm)
        self._ppm.set_value(settings.ppm)

        self._lna = Stepper(range(len(LNA_GAIN_REDUCTION_DB)), _lna)
        self._lna.set_value(settings.lna_state)
        low, high = IF_GAIN_REDUCTION_RANGE_DB
        self._if_gain_reduction = Stepper(range(low, high + 1), lambda db: f"−{db} dB")
        self._if_gain_reduction.set_value(settings.if_gain_reduction_db)
        low, high = FREQ_CORRECTION_RANGE_PPM
        ppm_steps = [
            round(hundredths / 100, 2)
            for hundredths in range(round(low * 100), round(high * 100) + 1)
        ]
        self._rspduo_ppm = Stepper(ppm_steps, _ppm)
        self._rspduo_ppm.set_value(settings.rspduo_ppm)

        self._rtlsdr_rows = (
            stepper_row(tr("Gain"), self._gain),
            switch_row("AGC", self._agc, tr(_AGC_HINT)),
            stepper_row(tr("PPM correction"), self._ppm),
        )
        self._rspduo_rows = (
            stepper_row(tr("LNA state"), self._lna),
            stepper_row(tr("IF gain reduction"), self._if_gain_reduction),
            stepper_row(tr("PPM correction"), self._rspduo_ppm),
        )
        for stepper in (
            self._gain,
            self._ppm,
            self._lna,
            self._if_gain_reduction,
            self._rspduo_ppm,
        ):
            stepper.value_changed.connect(self._on_device_settings_changed)
        self._agc.toggled.connect(self._on_device_settings_changed)

        self._dc = Toggle()
        self._dc.setChecked(controller.dc_correction)
        self._dc.toggled.connect(controller.set_dc_correction)
        self._iq = Toggle()
        self._iq.setChecked(controller.iq_correction)
        self._iq.toggled.connect(controller.set_iq_correction)
        sdr_page = self._subpage(
            Card(tr("Device"), [self._device, self._tuner, *self._rtl_tcp_rows]),
            Card(tr("Tuner"), [*self._rtlsdr_rows, *self._rspduo_rows]),
            Card(
                tr("Correction"),
                [
                    switch_row(tr("DC correction"), self._dc, tr(_DC_HINT)),
                    switch_row(tr("IQ correction"), self._iq, tr(_IQ_HINT)),
                ],
            ),
        )

        # --- Display. Each language by its own name; it takes effect in a new start,
        # which the page offers right away (`ui.language`).
        self._language = ChoiceRow(
            tr("Language"), list(available_languages().items()), subtitle=tr(_LANGUAGE_HINT)
        )
        self._language.set_current(stored_language(controller.settings))
        self._language.chosen.connect(self._on_language_chosen)
        self._theme = ChoiceRow(tr("Appearance"), [(DARK, tr("Dark")), (LIGHT, tr("Light"))])
        self._theme.chosen.connect(self._on_theme_chosen)
        self._theme_shown = DARK  # the scheme of the window, `set_theme`
        # The scheme is reported only once the sheet has closed (`THEME_FEEDBACK_MS`);
        # a second choice in between counts instead.
        self._theme_delay = QTimer(self)
        self._theme_delay.setSingleShot(True)
        self._theme_delay.setInterval(THEME_FEEDBACK_MS)
        self._theme_delay.timeout.connect(self._report_theme)
        # The figures above the clock in the navigation rail; `clicked`, not `toggled`,
        # so that the window setting the stored choice is not reported back.
        self._load_shown = Toggle()
        self._load_shown.clicked.connect(self.load_shown_chosen)
        # The level beside the volume slider on the radio page, likewise.
        self._volume_figure_shown = Toggle()
        self._volume_figure_shown.clicked.connect(self.volume_figure_shown_chosen)
        # How the constellation is drawn, on the signal page and on the radio page.
        self._constellation = ChoiceRow(
            tr("Constellation"),
            [(False, tr("Point cloud")), (True, tr("Density"))],
            subtitle=tr(_DENSITY_HINT),
        )
        self._constellation.chosen.connect(
            lambda density: self.constellation_density_chosen.emit(density)
        )
        # When the screen dims and when it goes off, each after the last touch; where
        # neither can happen (a PC, the udev rule missing) the line says why.
        self._dim_after = Stepper(TIMEOUT_CHOICES_S, timeout_text)
        self._dim_after.set_value(self._screen.dim_after_s)
        self._dim_after.value_changed.connect(self._screen.set_dim_after)
        self._off_after = Stepper(TIMEOUT_CHOICES_S, timeout_text)
        self._off_after.set_value(self._screen.off_after_s)
        self._off_after.value_changed.connect(self._screen.set_off_after)
        # Translators: when the screen dims, under the heading "Screen".
        dim_row = stepper_row(tr("Dim"), self._dim_after)
        dim_row.set_subtitle(backlight_hint(self._screen) or tr(_SCREEN_HINT))
        self._backlight_hint = dim_row.subtitle
        # Translators: when the screen goes off, under the heading "Screen".
        off_row = stepper_row(tr("Switch off"), self._off_after)

        # The listener's position, for the sky and the corrections of GEPOS; each value
        # opens the number pad.
        self._number_pad: NumberPad | None = None
        self._latitude = value_row(tr("Latitude"), subtitle=tr(_LOCATION_HINT))
        self._latitude.tapped.connect(self._edit_latitude)
        self._longitude = value_row(tr("Longitude"))
        self._longitude.tapped.connect(self._edit_longitude)
        self._height = value_row(tr("Height"))
        self._height.tapped.connect(self._edit_height)
        self._show_observer(controller.observer)
        controller.observer_changed.connect(self._show_observer)
        display_page = self._subpage(
            # Translators: the heading over language and colour scheme.
            Card(tr("Interface"), [self._language, self._theme]),
            Card(
                tr("Radio page"),
                [
                    switch_row(tr("Load in the menu bar"), self._load_shown),
                    switch_row(tr("Level beside the volume slider"), self._volume_figure_shown),
                    self._constellation,
                ],
            ),
            Card(tr("Screen"), [dim_row, off_row]),
            Card(tr("Location"), [self._latitude, self._longitude, self._height]),
        )

        # --- Audio. The ceiling of the volume: what 100 % of the slider gives, for
        # loudspeakers that take less than the amplifier delivers.
        low, high = CEILING_RANGE_DB
        self._volume_ceiling = Stepper(
            range(low, high + 1), lambda db: f"{db} dB".replace("-", "−")
        )
        self._volume_ceiling.set_value(controller.volume_ceiling_db)
        self._volume_ceiling.value_changed.connect(controller.set_volume_ceiling_db)
        controller.volume_changed.connect(
            lambda: self._volume_ceiling.set_value(self._controller.volume_ceiling_db)
        )
        # The equalizer: on or off for the station playing (or all of them), for all
        # stations or per station, the way to it, and deleting all its settings.
        self._equalizer_switch = Toggle()
        self._equalizer_switch.clicked.connect(controller.set_equalizer_enabled)
        self._equalizer_scope = ChoiceRow(
            tr("Applies to"),
            [(False, tr("All stations")), (True, tr("Per station"))],
            subtitle=tr(_PER_STATION_HINT),
        )
        self._equalizer_scope.chosen.connect(controller.set_equalizer_per_station)
        self._equalizer_button = action_row(tr("Adjust sound"))
        self._equalizer_button.tapped.connect(self.equalizer_requested)
        self._equalizer_reset = action_row(tr("Delete all sound settings"), danger=True)
        self._equalizer_reset.tapped.connect(self._ask_equalizer_reset)
        # Where the audio goes: the default of the system or one output for good.
        self._output_row = _OutputRow(controller)
        self._sheet: ActionSheet | None = None
        self._show_equalizer()
        controller.equalizer_changed.connect(self._show_equalizer)
        # Only does something when an announcement for the selected service runs in
        # another subchannel; the service itself stays selected throughout.
        self._announcement_switch = Toggle()
        self._announcement_switch.setChecked(controller.announcement_switching)
        self._announcement_switch.toggled.connect(controller.set_announcement_switching)
        audio_page = self._subpage(
            Card(
                tr("Output"),
                [
                    self._output_row,
                    stepper_row(tr("Maximum volume"), self._volume_ceiling, tr(_CEILING_HINT)),
                ],
            ),
            Card(
                tr("Sound"),
                [
                    switch_row(tr("Equalizer"), self._equalizer_switch),
                    self._equalizer_scope,
                    self._equalizer_button,
                    self._equalizer_reset,
                ],
            ),
            Card(
                tr("Announcements"),
                [
                    switch_row(
                        tr("Switch to the station of the announcement"),
                        self._announcement_switch,
                        tr(_ANNOUNCEMENT_HINT),
                    )
                ],
            ),
        )

        # --- The sub-pages with lists keep their height; the rest scroll.
        self._wifi_view = WifiView(self._wifi)
        self._bluetooth_view = BluetoothView(self._bluetooth, connect=controller.connect_bluetooth)
        self._storage_view = StorageView(controller, self._volumes)
        self._recording_view = RecordingView(controller)
        self._recording_view.recordings_requested.connect(self.recordings_requested)
        self._recording_view.changed.connect(self._show_states)
        self._mqtt_view = MqttView(controller.mqtt_bridge)
        self._stack = QStackedWidget()
        for page in (
            sdr_page,
            display_page,
            audio_page,
            self._wifi_view,
            self._bluetooth_view,
            self._storage_view,
            self._subpage(self._recording_view),
            self._subpage(self._mqtt_view),
        ):
            self._stack.addWidget(page)

        self._categories = CategoryList(_CATEGORIES)
        self._categories.chosen.connect(self.show_subpage)
        body = QHBoxLayout()
        body.setSpacing(24)
        body.addWidget(self._categories)
        body.addWidget(self._stack, stretch=1)

        # The version of the radio, small in the bottom row beside the way out; before
        # them what the sub-page shown has to say.
        self._version = set_role(
            QLabel(tr("Version {version}").format(version=APPLICATION_VERSION)), "small"
        )
        quit_button = set_role(QPushButton(tr("Quit radio")), "danger")
        quit_button.clicked.connect(self.quit_requested)
        bottom = QHBoxLayout()
        bottom.addWidget(self._wifi_view.message)
        bottom.addWidget(self._bluetooth_view.message)
        bottom.addWidget(self._storage_view.message)
        bottom.addStretch()  # keeps the button on the right where the messages are hidden
        bottom.addWidget(self._version)
        bottom.addSpacing(16)
        bottom.addWidget(quit_button)

        # The margins of the services page, so that the lists of both stand alike.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)
        layout.addLayout(body, stretch=1)
        layout.addLayout(bottom)
        self.show_subpage(SDR)

        self._restart_timer = QTimer(self)
        self._restart_timer.setSingleShot(True)
        self._restart_timer.setInterval(RESTART_DELAY_MS)
        self._restart_timer.timeout.connect(self._restart_if_receiving)

        controller.state_changed.connect(self._on_state_changed)
        self._show_device_rows()
        self._on_state_changed(controller.state)

        # The states in the list follow what they tell of.
        self._wifi.changed.connect(self._show_states)
        self._bluetooth.changed.connect(self._show_states)
        self._volumes.changed.connect(self._show_states)
        controller.audio_outputs_changed.connect(self._show_states)
        controller.recording_directory_changed.connect(self._show_states)
        controller.recording_enabled_changed.connect(self._show_states)
        if controller.mqtt_bridge is not None:
            controller.mqtt_bridge.status_changed.connect(self._show_states)
        self._show_states()

    @staticmethod
    def _subpage(*parts: QWidget) -> QScrollArea:
        """The parts one below the other, swiped up and down where they are taller than
        the page; the bottom edge fades while more lies below."""
        content = set_role(QWidget(), "transparent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        for part in parts:
            layout.addWidget(part)
        layout.addStretch()
        area = set_role(QScrollArea(), "transparent")
        area.setWidget(content)
        area.setWidgetResizable(True)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        set_role(area.viewport(), "transparent")
        QScroller.grabGesture(area.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture)
        EdgeFades(area, Qt.Orientation.Vertical)
        return area

    def show_subpage(self, key: str) -> None:
        index = next(i for i, (subpage, _name) in enumerate(_SUBPAGES) if subpage == key)
        self._stack.setCurrentIndex(index)
        self._categories.set_current(key)
        self._wifi_view.message.setVisible(key == WIFI)
        self._bluetooth_view.message.setVisible(key == BLUETOOTH)
        self._storage_view.message.setVisible(key == STORAGE)

    def current_subpage(self) -> str:
        return _SUBPAGES[self._stack.currentIndex()][0]

    def _show_states(self, *_args: object) -> None:
        """The line beneath each category in the list."""
        controller = self._controller
        states = {
            SDR: sdr_state(controller.device_settings),
            AUDIO: audio_state(controller),
            WIFI: wifi_state(self._wifi),
            BLUETOOTH: bluetooth_state(self._bluetooth),
            STORAGE: storage_state(controller, self._volumes),
            RECORDING: recording_state(controller),
            MQTT: mqtt_state(controller.mqtt_bridge),
        }
        for key, text in states.items():
            self._categories.set_state(key, text)

    def _on_theme_chosen(self, _key: str) -> None:
        # Not now, when the event loop gets to it: the sheet closes first.
        self._theme_delay.start()

    def _report_theme(self) -> None:
        """Reports the scheme chosen last, unless the window shows it already."""
        key = self._theme.current()
        if key != self._theme_shown:
            self.theme_chosen.emit(key)

    def set_theme(self, key: str) -> None:
        self._theme_shown = key
        self._theme.set_current(key)
        theme = THEMES[key]
        cards.set_theme(theme)
        self._categories.apply_theme(theme)
        self._wifi_view.apply_theme(theme)
        self._bluetooth_view.apply_theme(theme)
        self._storage_view.apply_theme(theme)
        self._mqtt_view.apply_theme(theme)
        if self._keyboard is not None:
            self._keyboard.apply_theme(theme)
        self.update()  # the switches, chevrons and lines take the new colours

    def set_load_shown(self, shown: bool) -> None:
        """Marks the choice without reporting it."""
        self._load_shown.setChecked(shown)

    def set_volume_figure_shown(self, shown: bool) -> None:
        """Marks the choice without reporting it."""
        self._volume_figure_shown.setChecked(shown)

    def set_constellation_density(self, density: bool) -> None:
        """Marks the choice without reporting it."""
        self._constellation.set_current(density)

    def shutdown(self) -> None:
        """Ends what still runs for the Wi-Fi, Bluetooth and the storage (on closing
        the window)."""
        self._wifi.shutdown()
        self._bluetooth.shutdown()
        self._volumes.shutdown()

    @property
    def wifi(self) -> WifiManager:
        """The Wi-Fi as the sub-page "Wi-Fi" shows it (`net.wifi`)."""
        return self._wifi

    @property
    def bluetooth(self) -> BluetoothManager:
        """BlueZ as the sub-page "Bluetooth" shows it (`net.bluetooth`)."""
        return self._bluetooth

    @property
    def volumes(self) -> VolumeManager:
        """The USB drives as the sub-page "Storage" shows them (`storage.volumes`)."""
        return self._volumes

    # --- Equalizer --------------------------------------------------------------------

    def _show_equalizer(self) -> None:
        self._equalizer_switch.setChecked(self._controller.equalizer_enabled)
        self._equalizer_scope.set_current(self._controller.equalizer_per_station)

    def action_sheet(self) -> ActionSheet:
        """The question before deleting, over the whole window, made on first use."""
        if self._sheet is None:
            self._sheet = ActionSheet(self.window())
        return self._sheet

    def _on_language_chosen(self, chosen: str) -> None:
        """Keeps the language chosen and offers the new start it takes, asked in that
        language; back to the language running, there is nothing to restart."""
        store_language(self._controller.settings, chosen)
        question = restart_question(chosen)
        if question is None:
            return
        self.action_sheet().open(
            question.title,
            question.detail,
            [(question.restart, "accent", self.restart_requested.emit)],
            cancel=question.later,
        )

    def _ask_equalizer_reset(self) -> None:
        self.action_sheet().open(
            tr("Delete all sound settings?"),
            tr(
                "The equalizer of every station and the one for all stations are switched "
                "off and set to Flat."
            ),
            [(tr("Delete"), "danger", self._controller.reset_equalizers)],
        )

    # --- Position ---------------------------------------------------------------------

    def number_pad(self) -> NumberPad:
        """The pad over the whole window, made on first use."""
        if self._number_pad is None:
            self._number_pad = NumberPad(self.window())
        return self._number_pad

    def _show_observer(self, observer: Observer) -> None:
        # Translators: the hemispheres of a latitude, North and South.
        north, south = tr("N"), tr("S")
        # Translators: the hemispheres of a longitude, East and West.
        east, west = tr("E"), tr("W")
        self._latitude.setText(coordinate_text(observer.latitude, north, south))
        self._longitude.setText(coordinate_text(observer.longitude, east, west))
        self._height.setText(f"{format_number(observer.height, 0)} m")

    def _set_observer(self, **changes: float) -> None:
        self._controller.set_observer(replace(self._controller.observer, **changes))

    def _edit_latitude(self) -> None:
        self.number_pad().open(
            tr("Latitude (south negative)"),
            self._controller.observer.latitude,
            COORDINATE_DECIMALS,
            "°",
            LATITUDE_RANGE,
            lambda value: self._set_observer(latitude=value),
        )

    def _edit_longitude(self) -> None:
        self.number_pad().open(
            tr("Longitude (west negative)"),
            self._controller.observer.longitude,
            COORDINATE_DECIMALS,
            "°",
            LONGITUDE_RANGE,
            lambda value: self._set_observer(longitude=value),
        )

    def _edit_height(self) -> None:
        self.number_pad().open(
            tr("Height above the ellipsoid"),
            self._controller.observer.height,
            0,
            " m",
            HEIGHT_RANGE_M,
            lambda value: self._set_observer(height=value),
        )

    def _on_device_chosen(self, _device: str) -> None:
        self._show_device_rows()
        self._on_device_settings_changed()

    def _show_device_rows(self) -> None:
        settings = DeviceSettings(device=self._device.current())
        self._tuner.setVisible(settings.rspduo() is not None)
        for row in self._rtl_tcp_rows:
            row.setVisible(settings.rtl_tcp() is not None)
        for row in self._rtlsdr_rows:
            row.setVisible(settings.is_rtlsdr)
        for row in self._rspduo_rows:
            row.setVisible(not settings.is_rtlsdr)

    # --- rtl_tcp ----------------------------------------------------------------------

    def keyboard(self) -> Keyboard:
        """The keyboard over the whole window, made on first use."""
        if self._keyboard is None:
            self._keyboard = Keyboard(self.window())
            self._keyboard.apply_theme(THEMES[self._theme_shown])
        return self._keyboard

    def _show_server(self) -> None:
        self._server_host.setText(self._rtl_tcp_host or tr(NO_HOST))
        self._server_host.set_placeholder(not self._rtl_tcp_host)
        self._server_port.setText(str(self._rtl_tcp_port))

    def _edit_server_host(self) -> None:
        self.keyboard().open(
            tr("rtl_tcp server"),
            self._set_server_host,
            text=self._rtl_tcp_host,
            check=valid_host,
            hint=tr("Name or IP address, without port"),
        )

    def _edit_server_port(self) -> None:
        self.number_pad().open(
            tr("rtl_tcp port"),
            self._rtl_tcp_port,
            0,
            "",
            RTL_TCP_PORT_RANGE,
            self._set_server_port,
        )

    def _set_server_host(self, text: str) -> None:
        self._rtl_tcp_host = text.strip()
        self._show_server()
        self._on_device_settings_changed()

    def _set_server_port(self, value: float) -> None:
        self._rtl_tcp_port = int(value)
        self._show_server()
        self._on_device_settings_changed()

    def _on_device_settings_changed(self, *_args: object) -> None:
        # Asked first: after an error another device starts reception at once
        # (`ReceiverController.set_device_settings`), which needs no restart.
        receiving = self._controller.is_receiving
        self._controller.set_device_settings(
            DeviceSettings(
                device=self._device.current(),
                gain_db=self._gain.value(),
                agc=self._agc.isChecked(),
                ppm=self._ppm.value(),
                rspduo_tuner=self._tuner.current(),
                lna_state=self._lna.value(),
                if_gain_reduction_db=self._if_gain_reduction.value(),
                rspduo_ppm=self._rspduo_ppm.value(),
                rtl_tcp_host=self._rtl_tcp_host,
                rtl_tcp_port=self._rtl_tcp_port,
            )
        )
        self._show_states()
        if receiving:
            self._restart_timer.start()

    def _restart_if_receiving(self) -> None:
        if self._controller.state is ReceptionState.RECEIVING:
            self._controller.restart_reception()

    def _on_state_changed(self, state: ReceptionState) -> None:
        """The device can be changed except while it is being opened or scans."""
        free = state in (ReceptionState.IDLE, ReceptionState.RECEIVING)
        rows: tuple[Row, ...] = (
            self._device,
            self._tuner,
            *self._rtl_tcp_rows,
            *self._rtlsdr_rows,
            *self._rspduo_rows,
        )
        for row in rows:
            row.setEnabled(free)
        if free:
            self._gain.setDisabled(self._agc.isChecked())
