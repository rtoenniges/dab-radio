"""The window of the touch surface: exactly one screen of the Touch Display 2 in
landscape, a navigation rail on the left and one page at a time beside it."""

from __future__ import annotations

import time

from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QCursor, QMouseEvent, QResizeEvent
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from dab.timecode import ReferenceClock
from i18n import format_date, tr, tr_noop, weekday
from net.wifi import BAND_5_GHZ
from ui.globe_gl import GlobeRenderThread
from ui.gnss_model import GnssModel
from ui.guide_model import GuideModel
from ui.journaline_model import JournalineModel
from ui.offscreen_gl import RenderThread
from ui.receiver_controller import LOAD_WARNING_THRESHOLD, ReceiverController, ReceptionState
from ui.touch.backlight import Backlight, Level, ScreenDimmer
from ui.touch.glow import GlowingSymbol
from ui.touch.icons import badged_pixmap, symbol_icon
from ui.touch.notification_view import NotificationDialog
from ui.touch.radio_page import RadioPage
from ui.touch.services_page import ServicesPage
from ui.touch.settings_page import SettingsPage
from ui.touch.signal_page import SignalPage
from ui.touch.signal_panels import FrameTally
from ui.touch.stations import StationDirectory
from ui.touch.stations_page import StationsPage
from ui.touch.theme import DARK, NAV_WIDTH, SETTINGS_THEME, THEMES, Theme, stylesheet
from ui.touch.widgets import repolish, set_role
from ui.traffic_model import TrafficModel
from ui.waterfall_gl import WaterfallRenderer

# The Touch Display 2 rotated to landscape (see `ui.touch`).
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

# The pages in the order of the navigation rail: key, symbol, name.
RADIO = "radio"
STATIONS = "stations"
SERVICES = "services"
SIGNAL = "signal"
SETTINGS = "settings"
_PAGES = (
    (RADIO, "radio", tr_noop("Radio")),
    (STATIONS, "stations", tr_noop("Stations")),
    (SERVICES, "services", tr_noop("Services")),
    (SIGNAL, "signal", tr_noop("Signal")),
    (SETTINGS, "settings", tr_noop("Settings")),
)
NAV_BUTTON_HEIGHT = 96
NAV_ICON = QSize(36, 36)
# The symbols above the clock (`TouchWindow._show_connectivity`): the Wi-Fi and
# Bluetooth, each while it is on.
STATUS_ICON = 24

# Key of the setting whether the rail shows the load (settings page, "Display"); hidden
# by default.
SETTINGS_LOAD_SHOWN = "touch/rail_load_shown"
# Key of the setting whether the constellation is drawn as a density rather than as
# points (settings page, "Display"), on the signal page and on the radio page alike.
SETTINGS_CONSTELLATION_DENSITY = "touch/constellation_density"
# Key of the setting whether the radio page shows the level beside the volume slider
# (settings page, "Display"); hidden by default.
SETTINGS_VOLUME_FIGURE_SHOWN = "touch/volume_figure_shown"

# Rate at which the plots of the signal page are drawn, as in the desktop window
# (`ui.main_window.PLOT_REFRESH_INTERVAL_MS`, where the reason is given).
PLOT_REFRESH_INTERVAL_MS = 200
CLOCK_INTERVAL_MS = 1000


def date_text(timestamp: float) -> str:
    """The weekday above the date, e.g. "Tuesday\\n2026-10-06", in the language of the
    surface (`i18n.weekday`), not in the locale of the Raspberry Pi. The longest weekday
    ("Wednesday", "Donnerstag") is narrower than "Settings" ("Einstellungen") above it,
    so the date fits the rail on two lines."""
    local = time.localtime(timestamp)
    return f"{weekday(local)}\n{format_date(local)}"


def band_badge(band: str) -> str:
    """The figure at the Wi-Fi above the clock for the band it is connected on: "5" on
    5 GHz; "" on 2.4 GHz, the usual band, and where it is not known."""
    return "5" if band == BAND_5_GHZ else ""


class _Banner(QLabel):
    """An error across the top of the page; a tap puts it away."""

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.hide()


# The pointer is hidden by an override cursor (`main.py`, full screen on the Pi). cage
# puts its default arrow back whenever an input device comes or goes (it updates the
# seat's capabilities and sets the cursor image anew), which Bluetooth headphones do:
# BlueZ gives them an input device for their buttons (AVRCP) a moment after they
# connect and takes it away when they go. Qt learns nothing of it, so the window sets
# the hidden cursor again, in steps, as the device may follow late (seen 09.10.2026).
REHIDE_CURSOR_DELAYS_MS = (500, 1500, 3000, 6000)


def rehide_cursor() -> None:
    """Tells the compositor the override cursor anew; nothing without one (a window on
    the PC). Qt skips setting the cursor it believes is set, so it goes through another
    shape first, both in one round of the event loop, before anything is drawn."""
    app = QApplication.instance()
    override = app.overrideCursor() if app is not None else None
    if override is None:
        return
    other = Qt.CursorShape.ArrowCursor
    if override.shape() == other:
        other = Qt.CursorShape.BlankCursor
    app.changeOverrideCursor(QCursor(other))
    app.changeOverrideCursor(QCursor(override.shape()))


class TouchWindow(QMainWindow):
    """The surface for the touch display. Call `controller.start()` once it is
    constructed, then `start_playing()`."""

    # The user asked to start anew in the language chosen; `main` does it once the window
    # has closed.
    restart_requested = pyqtSignal()

    def __init__(self, controller: ReceiverController) -> None:
        super().__init__()
        self.setWindowTitle("DAB+ Radio")
        self._controller = controller
        # Laid out for exactly this size. On a PC the window keeps it, so that what is
        # tested there is what the display shows.
        self.setFixedSize(SCREEN_WIDTH, SCREEN_HEIGHT)

        # The broadcast time of the ensemble (FIG 0/10): a Raspberry Pi without network
        # does not know the time. Also the clock of the slideshow's time stamps.
        self._clock = ReferenceClock()
        self._directory = StationDirectory(controller, self)
        self._guide_model = GuideModel(controller, self)
        # Built before the radio page: it empties itself on a change of station before
        # the page leaves its Journaline.
        self._journaline_model = JournalineModel(controller, self)
        # The audio frames counted for the figures of the radio and the signal page.
        tally = FrameTally(controller, self)
        self._gnss_model = GnssModel.for_controller(controller, self)
        self._traffic_model = TrafficModel.for_controller(controller, self)
        # One OpenGL context and one thread drawing for the 3D view of the GNSS menu and
        # the one on the radio page, set up when the first of them is shown.
        self._globe_renderer = GlobeRenderThread()
        # Likewise for the landscapes of the signal: the detail view of the signal page and
        # the null symbol on the radio page.
        self._waterfall_renderer = RenderThread(WaterfallRenderer())
        self._radio_page = RadioPage(
            controller,
            self._directory,
            self._guide_model,
            self._journaline_model,
            self._clock,
            tally,
            self._gnss_model,
            self._globe_renderer,
            self._waterfall_renderer,
        )
        self._stations_page = StationsPage(controller, self._directory)
        self._services_page = ServicesPage(
            controller,
            self._guide_model,
            self._gnss_model,
            self._traffic_model,
            self._clock,
            globe_renderer=self._globe_renderer,
        )
        self._signal_page = SignalPage(controller, tally, self._waterfall_renderer)
        # Dims the screen and switches it off when nothing has been touched for the
        # times set on the settings page; on a PC, without a backlight, it does nothing.
        self._screen = ScreenDimmer(Backlight.find(), controller.settings, self)
        self._settings_page = SettingsPage(controller, screen=self._screen)
        self._theme: Theme = THEMES[DARK]
        # Devices connecting and going bring the pointer back (see `rehide_cursor`).
        self._bluetooth_connected = self._connected_bluetooth_devices()
        self._settings_page.bluetooth.changed.connect(self._on_bluetooth_changed)
        # The symbols above the clock follow the Wi-Fi and Bluetooth, both read at the
        # start and kept current (`net.wifi.WifiManager.watch`, `net.bluetooth`).
        self._settings_page.wifi.changed.connect(self._show_connectivity)
        self._settings_page.bluetooth.changed.connect(self._show_connectivity)
        pages = {
            RADIO: self._radio_page,
            STATIONS: self._stations_page,
            SERVICES: self._services_page,
            SIGNAL: self._signal_page,
            SETTINGS: self._settings_page,
        }

        self._stack = QStackedWidget()
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        self._nav_buttons: dict[str, QToolButton] = {}
        self._page_keys: list[str] = []
        nav_layout = QVBoxLayout()
        nav_layout.setContentsMargins(8, 16, 8, 16)
        nav_layout.setSpacing(8)
        for key, _symbol, name in _PAGES:
            self._stack.addWidget(pages[key])
            button = set_role(QToolButton(), "nav")
            button.setText(tr(name))
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setIconSize(NAV_ICON)
            button.setFixedSize(NAV_WIDTH - 16, NAV_BUTTON_HEIGHT)
            self._nav_group.addButton(button, len(self._page_keys))
            self._nav_buttons[key] = button
            self._page_keys.append(key)
            nav_layout.addWidget(button)
        nav_layout.addStretch()
        # The symbol of the signal page glows red while the receiver is overloaded, seen
        # from every page.
        self._overload_glow = GlowingSymbol(
            self._nav_buttons[SIGNAL], "signal", NAV_ICON, self._screen_seen
        )
        # Visible from every page unless switched off; tight like the two lines of the
        # date. Names to the left and figures to the right, so that nothing moves when
        # a figure gains a digit. "wait" is the part of the DSP figure its thread spent
        # waiting rather than computing, for the GIL above all (`DspWorker._report_load`);
        # the row is the widest, "wartet: 100 %" (German) takes 88 of the 92 px within
        # the margins.
        self._load_panel = set_role(QWidget(), "transparent")
        load_layout = QGridLayout(self._load_panel)
        load_layout.setContentsMargins(2, 0, 2, 0)
        load_layout.setSpacing(0)
        self._dsp_load_name = set_role(QLabel("DSP:"), "small")
        self._dsp_load_label = set_role(QLabel("-"), "small")
        # Translators: the share of time the signal processing spent waiting, in the
        # narrow rail beside "DSP:" and "CPU:"; at most as wide as "wartet:".
        self._waiting_name = set_role(QLabel(tr("wait:")), "small")
        self._waiting_label = set_role(QLabel("-"), "small")
        self._cpu_load_name = set_role(QLabel("CPU:"), "small")
        self._cpu_load_label = set_role(QLabel("-"), "small")
        rows = (
            (self._dsp_load_name, self._dsp_load_label),
            (self._waiting_name, self._waiting_label),
            (self._cpu_load_name, self._cpu_load_label),
        )
        for row, (name, value) in enumerate(rows):
            load_layout.addWidget(name, row, 0)
            load_layout.addWidget(value, row, 1)
        for value in (self._dsp_load_label, self._waiting_label, self._cpu_load_label):
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        load_layout.setColumnStretch(1, 1)
        nav_layout.addWidget(self._load_panel)
        # Wi-Fi and Bluetooth, each only while it is on (`_show_connectivity`); the row
        # keeps its height with both off, so that nothing above it moves.
        self._wifi_icon = QLabel()
        self._bluetooth_icon = QLabel()
        self._status_row = set_role(QWidget(), "transparent")
        self._status_row.setFixedHeight(STATUS_ICON)
        status_layout = QHBoxLayout(self._status_row)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(12)
        status_layout.addStretch()
        for icon in (self._wifi_icon, self._bluetooth_icon):
            icon.setFixedSize(STATUS_ICON, STATUS_ICON)
            status_layout.addWidget(icon)
        status_layout.addStretch()
        nav_layout.addWidget(self._status_row)
        self._clock_label = set_role(QLabel(), "large")
        self._clock_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self._clock_label)
        self._date_label = set_role(QLabel(), "small")
        self._date_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self._date_label)
        self._nav_group.idClicked.connect(
            lambda index: self._on_nav_clicked(self._page_keys[index])
        )
        nav_rail = QWidget()
        nav_rail.setObjectName("navRail")
        nav_rail.setFixedWidth(NAV_WIDTH)
        nav_rail.setLayout(nav_layout)

        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(self._stack)
        self._banner = _Banner(self._content)
        self._banner.setObjectName("errorBanner")
        self._banner.setWordWrap(True)
        self._banner.hide()

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(nav_rail)
        root_layout.addWidget(self._content, stretch=1)
        self.setCentralWidget(root)

        # Notifications from outside (MQTT) over everything, the rail included; one coming
        # wakes the screen, at night as much as by day.
        self._notification_dialog = NotificationDialog(self, controller.notifications)
        controller.notifications.posted.connect(lambda _notification: self._screen.touched())

        self._radio_page.stations_requested.connect(lambda: self.show_page(STATIONS))
        self._radio_page.guide_requested.connect(self._show_guide)
        self._settings_page.theme_chosen.connect(self.set_theme)
        self._settings_page.load_shown_chosen.connect(self.set_load_shown)
        self._settings_page.volume_figure_shown_chosen.connect(self.set_volume_figure_shown)
        self._settings_page.constellation_density_chosen.connect(self.set_constellation_density)
        self._settings_page.equalizer_requested.connect(self._show_equalizer)
        self._settings_page.recordings_requested.connect(self._show_recordings)
        self._settings_page.quit_requested.connect(self.close)
        self._settings_page.restart_requested.connect(self.restart_requested)
        controller.dsp_worker.date_time_ready.connect(self._clock.synchronise)
        # The time comes from the ensemble; another ensemble sets it afresh.
        controller.live_reset.connect(self._clock.reset)
        controller.error_occurred.connect(self._show_error)
        controller.state_changed.connect(self._on_state_changed)
        controller.load_changed.connect(self._show_load)
        controller.overloaded_changed.connect(self._overload_glow.set_glowing)

        stored = controller.settings.value(SETTINGS_THEME, DARK)
        self.set_theme(stored if stored in THEMES else DARK)
        try:
            load_shown = controller.settings.value(SETTINGS_LOAD_SHOWN, False, type=bool)
        except TypeError, ValueError:  # edited by hand
            load_shown = False
        self.set_load_shown(load_shown)
        try:
            volume_figure_shown = controller.settings.value(
                SETTINGS_VOLUME_FIGURE_SHOWN, False, type=bool
            )
        except TypeError, ValueError:  # edited by hand
            volume_figure_shown = False
        self.set_volume_figure_shown(volume_figure_shown)
        try:
            density = controller.settings.value(SETTINGS_CONSTELLATION_DENSITY, False, type=bool)
        except TypeError, ValueError:  # edited by hand
            density = False
        self.set_constellation_density(density)
        self.show_page(RADIO)

        self._plot_timer = QTimer(self)
        self._plot_timer.setInterval(PLOT_REFRESH_INTERVAL_MS)
        self._plot_timer.timeout.connect(self._render_plots)
        self._plot_timer.start()
        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(CLOCK_INTERVAL_MS)
        self._clock_timer.timeout.connect(self._show_time)
        self._clock_timer.start()
        self._show_time()

    def _connected_bluetooth_devices(self) -> set[str]:
        bluetooth = self._settings_page.bluetooth
        return {d.path for d in bluetooth.devices.values() if d.connected}

    def _on_bluetooth_changed(self) -> None:
        connected = self._connected_bluetooth_devices()
        if connected == self._bluetooth_connected:
            return
        self._bluetooth_connected = connected
        for delay in REHIDE_CURSOR_DELAYS_MS:
            QTimer.singleShot(delay, rehide_cursor)

    def start_playing(self) -> None:
        """Plays the station of the last session, as a radio does when switched on;
        without one, the station page opens to choose one."""
        if self._controller.selected_sid is None:
            self.show_page(STATIONS)
            return
        self.show_page(RADIO)
        self._controller.start_reception()

    def _on_nav_clicked(self, key: str) -> None:
        """A tap on the button of the signal page while it is shown takes it back to
        its first page: its pages are swiped, with no other way straight back. The
        services page and the settings have chips for theirs (the chips of the views
        with a pager do the same there); the radio page shows its player again anyway
        (`show_page`)."""
        if key == self.current_page() and key == SIGNAL:
            self._signal_page.show_first_page()
        self.show_page(key)

    def show_page(self, key: str) -> None:
        if key == RADIO:
            # Also from the choice of the display, as from a sub-page.
            self._radio_page.show_player()
        self._stack.setCurrentIndex(self._page_keys.index(key))
        self._nav_buttons[key].setChecked(True)
        self._render_plots()

    def _show_guide(self, sid: int) -> None:
        """The guide card of the radio page: the schedule of the station playing."""
        self._services_page.show_guide(sid)
        self.show_page(SERVICES)

    def _show_equalizer(self) -> None:
        """The way from the settings to the equalizer, a sub-page of the radio page."""
        self.show_page(RADIO)
        self._radio_page.show_equalizer()

    def _show_recordings(self) -> None:
        """The way from the settings to the recordings, a sub-page of the radio page."""
        self.show_page(RADIO)
        self._radio_page.show_recordings()

    def current_page(self) -> str:
        return self._page_keys[self._stack.currentIndex()]

    def set_theme(self, key: str) -> None:
        """Switches between the dark and the light scheme and keeps the choice."""
        theme = THEMES[key]
        self._theme = theme
        self._controller.settings.setValue(SETTINGS_THEME, key)
        self.setStyleSheet(stylesheet(theme))
        muted, accent = theme.color("muted"), theme.color("accent")
        for page_key, symbol, _name in _PAGES:
            self._nav_buttons[page_key].setIcon(symbol_icon(symbol, muted, NAV_ICON, accent))
        self._overload_glow.set_theme(theme)
        self._show_connectivity()
        self._radio_page.apply_theme(theme)
        self._stations_page.apply_theme(theme)
        self._services_page.apply_theme(theme)
        self._signal_page.apply_theme(theme)
        self._settings_page.set_theme(key)

    def set_load_shown(self, shown: bool) -> None:
        """Shows or hides the load in the rail and keeps the choice."""
        self._controller.settings.setValue(SETTINGS_LOAD_SHOWN, shown)
        self._load_panel.setVisible(shown)
        self._settings_page.set_load_shown(shown)

    def set_volume_figure_shown(self, shown: bool) -> None:
        """Shows or hides the level beside the volume slider of the radio page and
        keeps the choice."""
        self._controller.settings.setValue(SETTINGS_VOLUME_FIGURE_SHOWN, shown)
        self._radio_page.set_volume_figure_shown(shown)
        self._settings_page.set_volume_figure_shown(shown)

    def set_constellation_density(self, density: bool) -> None:
        """Draws the constellation as a density or as points, on the signal page and
        on the radio page, and keeps the choice."""
        self._controller.settings.setValue(SETTINGS_CONSTELLATION_DENSITY, density)
        self._signal_page.set_constellation_density(density)
        self._radio_page.set_constellation_density(density)
        self._settings_page.set_constellation_density(density)

    # --- Status -----------------------------------------------------------------------

    def _show_error(self, message: str) -> None:
        self._banner.setText(tr("Error: {message}").format(message=message))
        self._place_banner()
        self._banner.show()
        self._banner.raise_()

    def _on_state_changed(self, state: ReceptionState) -> None:
        # A new attempt: the error belongs to the last one.
        if state in (ReceptionState.CONNECTING, ReceptionState.SCANNING):
            self._banner.hide()

    def _place_banner(self) -> None:
        width = self._content.width() - 48
        self._banner.setFixedWidth(width)
        self._banner.adjustSize()
        self._banner.move(24, 16)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._place_banner()

    def _show_load(self, dsp_load: float, dsp_waiting: float, process_load: float) -> None:
        """The figures of the signal page, shortened: share of real time the signal
        processing needs, how much of that it spent waiting, load of the program
        (100 % = one core)."""
        self._dsp_load_label.setText(f"{dsp_load * 100:.0f} %")
        self._waiting_label.setText(f"{dsp_waiting * 100:.0f} %")
        self._cpu_load_label.setText(f"{process_load * 100:.0f} %")
        # Red as in the status bar of the desktop: the audio is about to drop out.
        role = "error" if dsp_load > LOAD_WARNING_THRESHOLD else "small"
        for label in (self._dsp_load_name, self._dsp_load_label):
            set_role(label, role)
            repolish(label)

    def _show_connectivity(self) -> None:
        """The symbols above the clock: the Wi-Fi while it is on (and there is a Wi-Fi
        device), connected with the band it is connected on (`band_badge`), Bluetooth
        while the adapter is on; each in the colour of the text while connected, muted
        otherwise, and gone while off."""
        wifi = self._settings_page.wifi
        bluetooth = self._settings_page.bluetooth
        connected = bool(wifi.connected_ssid)
        self._show_status_icon(
            self._wifi_icon,
            "wifi",
            wifi.enabled and wifi.device is not None,
            connected,
            band_badge(wifi.connected_band) if connected else "",
        )
        # Off, BlueZ drops every connection; its objects may still say otherwise.
        self._show_status_icon(
            self._bluetooth_icon,
            "bluetooth",
            bluetooth.powered,
            bluetooth.powered and bluetooth.connected_device is not None,
        )

    def _show_status_icon(
        self, label: QLabel, symbol: str, on: bool, connected: bool, badge: str = ""
    ) -> None:
        if on:
            color = self._theme.color("text" if connected else "muted")
            pixmap = badged_pixmap(
                symbol, color, STATUS_ICON, badge, label.font(), self.devicePixelRatioF()
            )
            # As wide as the symbol with its badge; the row keeps both symbols centred.
            label.setFixedWidth(round(pixmap.deviceIndependentSize().width()))
            label.setPixmap(pixmap)
        label.setVisible(on)

    def _show_time(self) -> None:
        now = self._clock.now() or time.time()
        self._clock_label.setText(time.strftime("%H:%M", time.localtime(now)))
        self._date_label.setText(date_text(now))

    def _screen_seen(self) -> bool:
        return self._screen.level is not Level.OFF

    def _render_plots(self) -> None:
        if not self._screen_seen():
            return  # nobody sees them, and the GUI thread of the Pi has the time
        if self.current_page() == RADIO:
            self._radio_page.render()
        elif self.current_page() == SIGNAL:
            self._signal_page.render()
        elif self.current_page() == SERVICES:
            self._services_page.render()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._screen.shutdown()
        self._settings_page.shutdown()
        self._globe_renderer.stop()
        self._waterfall_renderer.stop()
        self._controller.shutdown()
        event.accept()
