"""The settings page, in sub-pages chosen by chips at the top: SDR (device, gain,
frequency correction, corrections), Display (colour scheme, the load in the navigation
rail, the level beside the volume slider of the radio page, the constellation as points
or as a density, when the screen dims and goes off,
`ui.touch.backlight`; the listener's position), Audio (the output, `audio.outputs`; the
ceiling of the volume, `audio.volume`; the equalizer: on or off, the way to it, for all
stations or per station, deleting all its settings; switching to announcements), Wi-Fi
(`ui.touch.wifi_view`), Bluetooth (`ui.touch.bluetooth_view`), Storage (the USB
drives and the directory for the recordings, `ui.touch.storage_view`), Recording
(`ui.touch.recording_view`) and MQTT (the radio at the broker of a smart home,
`ui.touch.mqtt_view`). The sub-pages of
rows are swiped up and down where they are taller than the page. The way out of full
screen is on every sub-page, the version of the radio beside it in the bottom row."""

from __future__ import annotations

from dataclasses import replace

from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QScroller,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from audio.volume import CEILING_RANGE_DB
from dab.gnss import Observer
from i18n import available_languages, tr, tr_noop
from net.bluetooth import BluetoothManager
from net.wifi import WifiManager
from sdr.rspduo import FREQ_CORRECTION_RANGE_PPM, IF_GAIN_REDUCTION_RANGE_DB, LNA_GAIN_REDUCTION_DB
from storage.volumes import VolumeManager
from ui.language import restart_question, store_language, stored_language
from ui.receiver_controller import (
    APPLICATION_VERSION,
    DEVICES,
    HEIGHT_RANGE_M,
    LATITUDE_RANGE,
    LONGITUDE_RANGE,
    RTLSDR_PPM_RANGE,
    DeviceSettings,
    ReceiverController,
    ReceptionState,
)
from ui.touch.backlight import TIMEOUT_CHOICES_S, ScreenDimmer
from ui.touch.bluetooth_view import BluetoothView
from ui.touch.icons import symbol_icon
from ui.touch.mqtt_view import MqttView
from ui.touch.number_pad import NumberPad, format_number
from ui.touch.overlay import ActionSheet
from ui.touch.recording_view import RecordingView
from ui.touch.storage_view import StorageView
from ui.touch.theme import DARK, LIGHT, THEMES, TOUCH_TARGET, Theme
from ui.touch.widgets import ChipRow, ChoiceButtons, EdgeFades, Stepper, set_role
from ui.touch.wifi_view import WifiView

# Neither the stick nor the RSPduo takes a new gain while running, so a change during
# reception restarts it (`ReceiverController.restart_reception`); only once the stepper
# has rested this long, not for every step of a held button.
RESTART_DELAY_MS = 1200
# Restyling the whole window for the other colour scheme holds the GUI thread for a
# moment (0.3 s on the Pi, 0.16 s on the PC, measured 2026-10-13 off screen), and the
# tapped button would only light up with the new scheme, as if the tap had been lost.
# It is drawn first, and the scheme follows this much later.
THEME_FEEDBACK_MS = 50

LABEL_WIDTH = 220
_HINT = tr_noop("Changes to the device briefly restart a running reception.")
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
OUTPUT_ICON = QSize(28, 28)
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
EQUALIZER_ICON = QSize(40, 40)

# The sub-pages in the order of their chips: key, name.
SDR = "sdr"
DISPLAY = "display"
AUDIO = "audio"
WIFI = "wifi"
BLUETOOTH = "bluetooth"
STORAGE = "storage"
RECORDING = "recording"
MQTT = "mqtt"
_SUBPAGES = (
    (SDR, "SDR"),
    (DISPLAY, tr_noop("Display")),
    (AUDIO, tr_noop("Audio")),
    (WIFI, tr_noop("Wi-Fi")),
    (BLUETOOTH, "Bluetooth"),
    (STORAGE, tr_noop("Storage")),
    (RECORDING, tr_noop("Recording")),
    (MQTT, "MQTT"),
)


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


class _OutputRow(QWidget):
    """The outputs to choose from, "Automatic" first (the default of the system);
    beside them the output in use, where it is not the one chosen. Made anew as outputs
    come and go (`ReceiverController.audio_outputs_changed`)."""

    def __init__(self, controller: ReceiverController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._theme = THEMES[DARK]
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: list[QPushButton] = []
        self._keys: list[str] = []
        self._bluetooth: list[bool] = []
        self._in_use = set_role(QLabel(), "muted")
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._layout.addWidget(self._in_use)
        self._layout.addStretch()
        self._group.idClicked.connect(self._on_clicked)
        controller.audio_outputs_changed.connect(self.refresh)
        self.refresh()

    def keys(self) -> list[str]:
        """The ids of the buttons, "" for "Automatic"."""
        return list(self._keys)

    def buttons(self) -> list[QPushButton]:
        return list(self._buttons)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        self._show_icons()

    def refresh(self) -> None:
        controller = self._controller
        outputs = controller.audio_outputs
        keys = ["", *(output.id for output in outputs)]
        names = [tr("Automatic"), *(output.name for output in outputs)]
        bluetooth = [False, *(output.bluetooth for output in outputs)]
        if keys != self._keys or [b.text() for b in self._buttons] != names:
            self._rebuild(keys, names, bluetooth)
        choice = controller.audio_output_choice
        for key, button in zip(self._keys, self._buttons, strict=True):
            button.setChecked(key == choice)
        in_use = controller.audio_output_in_use
        shown = in_use is not None and (choice == "" or choice != in_use.id)
        self._in_use.setText(tr("currently {name}").format(name=in_use.name) if shown else "")
        self._in_use.setVisible(shown)

    def _rebuild(self, keys: list[str], names: list[str], bluetooth: list[bool]) -> None:
        for button in self._buttons:
            self._group.removeButton(button)
            self._layout.removeWidget(button)
            button.deleteLater()
        self._buttons = []
        self._keys = keys
        self._bluetooth = bluetooth
        for index, name in enumerate(names):
            button = QPushButton(name)
            button.setCheckable(True)
            self._group.addButton(button, index)
            self._layout.insertWidget(index, button)
            self._buttons.append(button)
        self._show_icons()

    def _show_icons(self) -> None:
        """The Bluetooth rune on the buttons of Bluetooth devices, in the colour of the
        scheme (the chosen button draws its text in the colour on the accent)."""
        text, on_accent = self._theme.color("text"), self._theme.color("on_accent")
        for button, is_bluetooth in zip(self._buttons, self._bluetooth, strict=True):
            if is_bluetooth:
                button.setIconSize(OUTPUT_ICON)
                button.setIcon(symbol_icon("bluetooth", text, OUTPUT_ICON, on_accent))

    def _on_clicked(self, index: int) -> None:
        self._controller.set_audio_output(self._keys[index])


class SettingsPage(QWidget):
    theme_chosen = pyqtSignal(str)  # DARK or LIGHT
    load_shown_chosen = pyqtSignal(bool)
    volume_figure_shown_chosen = pyqtSignal(bool)
    constellation_density_chosen = pyqtSignal(bool)  # True: density, False: points
    equalizer_requested = pyqtSignal()  # the button with the symbol of the equalizer
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

        # The stick apart from the two tuners of the RSPduo: two devices, not three.
        self._device = ChoiceButtons(list(DEVICES.items()), gaps=("rspduo1",))
        self._device.set_current(settings.device)
        self._device.chosen.connect(self._on_device_chosen)

        self._gain = Stepper(controller.gain_steps_db, lambda gain: f"{gain:.1f} dB")
        self._gain.set_value(settings.gain_db)
        self._agc = QPushButton("AGC")
        self._agc.setCheckable(True)
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

        gain_row = QHBoxLayout()
        gain_row.setSpacing(24)
        gain_row.addWidget(self._gain)
        gain_row.addWidget(self._agc)
        gain_row.addStretch()
        self._rtlsdr_rows = self._rows((tr("Gain"), gain_row), (tr("PPM correction"), self._ppm))
        self._rspduo_rows = self._rows(
            (tr("LNA state"), self._lna),
            (tr("IF gain reduction"), self._if_gain_reduction),
            (tr("PPM correction"), self._rspduo_ppm),
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

        self._dc = QPushButton(tr("DC correction"))
        self._dc.setCheckable(True)
        self._dc.setChecked(controller.dc_correction)
        self._dc.toggled.connect(controller.set_dc_correction)
        self._iq = QPushButton(tr("IQ correction"))
        self._iq.setCheckable(True)
        self._iq.setChecked(controller.iq_correction)
        self._iq.toggled.connect(controller.set_iq_correction)
        corrections = QHBoxLayout()
        corrections.setSpacing(8)
        corrections.addWidget(self._dc)
        corrections.addWidget(self._iq)
        corrections.addStretch()

        # On the sub-page "Audio". Only does something when an announcement for the
        # selected service runs in another subchannel; the service itself stays
        # selected throughout.
        self._announcement_switch = QPushButton(tr("Switch to the station of the announcement"))
        self._announcement_switch.setCheckable(True)
        self._announcement_switch.setChecked(controller.announcement_switching)
        self._announcement_switch.toggled.connect(controller.set_announcement_switching)

        # Each language by its own name; it takes effect in a new start, which the page
        # offers right away (`ui.language`).
        self._language = ChoiceButtons(list(available_languages().items()))
        self._language.set_current(stored_language(controller.settings))
        self._language.chosen.connect(self._on_language_chosen)
        self._theme = ChoiceButtons([(DARK, tr("Dark")), (LIGHT, tr("Light"))])
        self._theme.chosen.connect(self._on_theme_chosen)
        self._theme_shown = DARK  # the scheme of the window, `set_theme`
        # The scheme is reported only once the tap shows (`THEME_FEEDBACK_MS`); a
        # second tap in between counts instead.
        self._theme_delay = QTimer(self)
        self._theme_delay.setSingleShot(True)
        self._theme_delay.setInterval(THEME_FEEDBACK_MS)
        self._theme_delay.timeout.connect(self._report_theme)
        # The figures above the clock in the navigation rail; `clicked`, not `toggled`,
        # so that the window setting the stored choice is not reported back.
        self._load_shown = QPushButton(tr("Show in the menu bar"))
        self._load_shown.setCheckable(True)
        self._load_shown.clicked.connect(self.load_shown_chosen)
        # The level beside the volume slider on the radio page, likewise.
        self._volume_figure_shown = QPushButton(tr("Show beside the slider"))
        self._volume_figure_shown.setCheckable(True)
        self._volume_figure_shown.clicked.connect(self.volume_figure_shown_chosen)
        # How the constellation is drawn, on the signal page and on the radio page.
        self._constellation = ChoiceButtons([(False, tr("Point cloud")), (True, tr("Density"))])
        self._constellation.chosen.connect(
            lambda density: self.constellation_density_chosen.emit(density)
        )
        # When the screen dims and when it goes off, each after the last touch; where
        # neither can happen (a PC, the udev rule missing) a line says why.
        self._dim_after = Stepper(TIMEOUT_CHOICES_S, timeout_text)
        self._dim_after.set_value(self._screen.dim_after_s)
        self._dim_after.value_changed.connect(self._screen.set_dim_after)
        self._off_after = Stepper(TIMEOUT_CHOICES_S, timeout_text)
        self._off_after.set_value(self._screen.off_after_s)
        self._off_after.value_changed.connect(self._screen.set_off_after)
        self._backlight_hint = set_role(QLabel(backlight_hint(self._screen)), "small")
        self._backlight_hint.setVisible(bool(self._backlight_hint.text()))

        # The ceiling of the volume: what 100 % of the slider gives, for loudspeakers
        # that take less than the amplifier delivers.
        low, high = CEILING_RANGE_DB
        self._volume_ceiling = Stepper(
            range(low, high + 1), lambda db: f"{db} dB".replace("-", "−")
        )
        self._volume_ceiling.set_value(controller.volume_ceiling_db)
        self._volume_ceiling.value_changed.connect(controller.set_volume_ceiling_db)
        controller.volume_changed.connect(
            lambda: self._volume_ceiling.set_value(self._controller.volume_ceiling_db)
        )

        # The equalizer: on or off for the station playing (or all of them), the way to
        # it beside; for all stations or per station on a row of its own.
        self._equalizer_switch = set_role(QPushButton(tr("Off")), "switch")
        self._equalizer_switch.setCheckable(True)
        self._equalizer_switch.clicked.connect(controller.set_equalizer_enabled)
        self._equalizer_button = QToolButton()
        self._equalizer_button.setFixedSize(TOUCH_TARGET, TOUCH_TARGET)
        self._equalizer_button.setIconSize(EQUALIZER_ICON)
        self._equalizer_button.setToolTip(tr("Open the equalizer"))
        self._equalizer_button.clicked.connect(self.equalizer_requested)
        equalizer = QHBoxLayout()
        equalizer.setSpacing(8)
        equalizer.addWidget(self._equalizer_switch)
        equalizer.addWidget(self._equalizer_button)
        equalizer.addStretch()
        self._equalizer_scope = ChoiceButtons(
            [(False, tr("All stations")), (True, tr("Per station"))]
        )
        self._equalizer_scope.chosen.connect(controller.set_equalizer_per_station)
        self._equalizer_reset = QPushButton(tr("Delete all sound settings"))
        self._equalizer_reset.clicked.connect(self._ask_equalizer_reset)
        # Where the audio goes: the default of the system or one output for good.
        self._output_row = _OutputRow(controller)
        self._sheet: ActionSheet | None = None
        self._show_equalizer()
        controller.equalizer_changed.connect(self._show_equalizer)

        # The listener's position, for the sky and the corrections of GEPOS; each value
        # opens the number pad.
        self._number_pad: NumberPad | None = None
        self._latitude = set_role(QPushButton(), "value")
        self._latitude.clicked.connect(self._edit_latitude)
        self._longitude = set_role(QPushButton(), "value")
        self._longitude.clicked.connect(self._edit_longitude)
        self._height = set_role(QPushButton(), "value")
        self._height.clicked.connect(self._edit_height)
        position = QHBoxLayout()
        position.setSpacing(8)
        for button in (self._latitude, self._longitude, self._height):
            position.addWidget(button)
        position.addStretch()
        self._show_observer(controller.observer)
        controller.observer_changed.connect(self._show_observer)

        # The device on top, then its own rows (only those of the chosen one are shown),
        # then what applies to every device.
        self._stack = QStackedWidget()
        self._stack.addWidget(
            self._subpage(
                self._rows((tr("Device"), self._device)),
                self._rtlsdr_rows,
                self._rspduo_rows,
                self._rows((tr("Correction"), corrections)),
            )
        )
        self._stack.addWidget(
            self._subpage(
                self._rows(
                    (tr("Language"), self._language),
                    (tr("Appearance"), self._theme),
                    (tr("Load"), self._load_shown),
                    (tr("Volume"), self._volume_figure_shown),
                    (tr("Constellation"), self._constellation),
                    ("", set_role(QLabel(tr(_DENSITY_HINT)), "small")),
                    (tr("Dim screen"), self._dim_after),
                    (tr("Screen off"), self._off_after),
                ),
                self._backlight_hint,
                self._rows((tr("Location"), position)),
            )
        )
        self._stack.addWidget(
            self._subpage(
                self._rows(
                    (tr("Output"), self._output_row),
                    ("", set_role(QLabel(tr(_OUTPUT_HINT)), "small")),
                    (tr("Maximum volume"), self._volume_ceiling),
                    ("", set_role(QLabel(tr(_CEILING_HINT)), "small")),
                    (tr("Equalizer"), equalizer),
                    (tr("Applies to"), self._equalizer_scope),
                    ("", set_role(QLabel(tr(_PER_STATION_HINT)), "small")),
                    (tr("Reset"), self._equalizer_reset),
                    (tr("Announcements"), self._announcement_switch),
                )
            )
        )
        self._wifi_view = WifiView(self._wifi, LABEL_WIDTH)
        self._stack.addWidget(self._wifi_view)
        self._bluetooth_view = BluetoothView(
            self._bluetooth, LABEL_WIDTH, connect=controller.connect_bluetooth
        )
        self._stack.addWidget(self._bluetooth_view)
        self._storage_view = StorageView(controller, self._volumes, LABEL_WIDTH)
        self._stack.addWidget(self._storage_view)
        self._recording_view = RecordingView(controller, LABEL_WIDTH)
        self._recording_view.recordings_requested.connect(self.recordings_requested)
        self._stack.addWidget(self._recording_view)
        # Seven rows and two hints: more than the page is high, so it scrolls.
        self._mqtt_view = MqttView(controller.mqtt_bridge, LABEL_WIDTH)
        self._stack.addWidget(self._subpage(self._mqtt_view))
        self._subpage_chips = ChoiceButtons(
            [(key, tr(name)) for key, name in _SUBPAGES], role="chip"
        )
        self._subpage_chips.chosen.connect(self.show_subpage)
        # Swiped sideways where it is wider than the page, as more sub-pages come.
        self._chips_area = ChipRow(self._subpage_chips)
        # The version of the radio, small in the bottom row beside the way out: the
        # eight chips need the whole width (seen in a render, 10.10.2026).
        self._version = set_role(
            QLabel(tr("Version {version}").format(version=APPLICATION_VERSION)), "small"
        )
        top = QHBoxLayout()
        top.addWidget(self._chips_area, stretch=1)

        self._hint = set_role(QLabel(tr(_HINT)), "small")
        quit_button = set_role(QPushButton(tr("Quit radio")), "danger")
        quit_button.clicked.connect(self.quit_requested)
        bottom = QHBoxLayout()
        bottom.addWidget(self._hint)
        bottom.addWidget(self._wifi_view.message)
        bottom.addWidget(self._bluetooth_view.message)
        bottom.addWidget(self._storage_view.message)
        bottom.addStretch()  # keeps the button on the right where the hint is hidden
        bottom.addWidget(self._version)
        bottom.addSpacing(16)
        bottom.addWidget(quit_button)

        # The margins of the services page, so that the chips of both stand alike.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)
        layout.addLayout(top)
        layout.addWidget(self._stack, stretch=1)
        layout.addLayout(bottom)
        self.show_subpage(SDR)

        self._restart_timer = QTimer(self)
        self._restart_timer.setSingleShot(True)
        self._restart_timer.setInterval(RESTART_DELAY_MS)
        self._restart_timer.timeout.connect(self._restart_if_receiving)

        controller.state_changed.connect(self._on_state_changed)
        self._show_device_rows()
        self._on_state_changed(controller.state)

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

    @staticmethod
    def _rows(*rows: tuple[str, QWidget | QHBoxLayout]) -> QWidget:
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(12)
        for row, (name, content) in enumerate(rows):
            label = set_role(QLabel(name), "muted")
            label.setFixedWidth(LABEL_WIDTH)
            grid.addWidget(label, row, 0)
            if isinstance(content, QHBoxLayout):
                grid.addLayout(content, row, 1)
            else:
                grid.addWidget(content, row, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        return widget

    def show_subpage(self, key: str) -> None:
        index = next(i for i, (subpage, _name) in enumerate(_SUBPAGES) if subpage == key)
        self._stack.setCurrentIndex(index)
        self._subpage_chips.set_current(key)
        # Only the device restarts the reception.
        self._hint.setVisible(key == SDR)
        self._wifi_view.message.setVisible(key == WIFI)
        self._bluetooth_view.message.setVisible(key == BLUETOOTH)
        self._storage_view.message.setVisible(key == STORAGE)

    def current_subpage(self) -> str:
        return _SUBPAGES[self._stack.currentIndex()][0]

    def _on_theme_chosen(self, _key: str) -> None:
        # Drawn now, not when the event loop gets to it: the restyling may come first.
        self._theme.repaint()
        self._theme_delay.start()

    def _report_theme(self) -> None:
        """Reports the scheme chosen last, unless the window shows it already."""
        key = self._theme.current()
        if key != self._theme_shown:
            self.theme_chosen.emit(key)

    def set_theme(self, key: str) -> None:
        self._theme_shown = key
        self._theme.set_current(key)
        self._wifi_view.apply_theme(THEMES[key])
        self._bluetooth_view.apply_theme(THEMES[key])
        self._storage_view.apply_theme(THEMES[key])
        self._mqtt_view.apply_theme(THEMES[key])
        self._output_row.apply_theme(THEMES[key])
        text = THEMES[key].color("text")
        self._equalizer_button.setIcon(symbol_icon("equalizer", text, EQUALIZER_ICON))

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
        enabled = self._controller.equalizer_enabled
        self._equalizer_switch.setChecked(enabled)
        self._equalizer_switch.setText(tr("On") if enabled else tr("Off"))
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
        rtlsdr = self._device.current() == "rtlsdr"
        self._rtlsdr_rows.setVisible(rtlsdr)
        self._rspduo_rows.setVisible(not rtlsdr)

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
                lna_state=self._lna.value(),
                if_gain_reduction_db=self._if_gain_reduction.value(),
                rspduo_ppm=self._rspduo_ppm.value(),
            )
        )
        if receiving:
            self._restart_timer.start()

    def _restart_if_receiving(self) -> None:
        if self._controller.state is ReceptionState.RECEIVING:
            self._controller.restart_reception()

    def _on_state_changed(self, state: ReceptionState) -> None:
        """The device can be changed except while it is being opened or scans."""
        free = state in (ReceptionState.IDLE, ReceptionState.RECEIVING)
        for widget in (self._device, self._rtlsdr_rows, self._rspduo_rows):
            widget.setEnabled(free)
        if free:
            self._gain.setDisabled(self._agc.isChecked())
