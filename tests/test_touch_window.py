"""The touch surface (`ui.touch`) and how `main.py` chooses it: that its pages hand what
the user does to the controller and follow what the controller reports."""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import QBuffer, QEvent, QEventLoop, QIODevice, QObject, QPoint, Qt, QTimer
from PyQt6.QtGui import QFontMetrics, QImage
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton

from audio.equalizer import BASS, EqualizerSettings, EqualizerState
from dab.announcement import ALARM_CLUSTER, AnnouncementStatus
from dab.dls import DlPlus, DlPlusContentType, DynamicLabel
from dab.ensemble_cache import SOURCE_SCAN, EnsembleCache, EnsembleInfo
from dab.fig import (
    Announcement,
    AnnouncementSupport,
    AudioComponent,
    OeAnnouncement,
    Subchannel,
)
from dab.mot import CONTENT_TYPE_IMAGE, MotObject
from dab.slideshow import PARAM_TRIGGER_TIME
from dsp.pipeline import SubchannelStatus
from main import parse_arguments
from net import wifi
from tests.fake_receiver import ENSEMBLE_5C, SERVICES_5C, SERVICES_11D
from ui import receiver_controller
from ui.receiver_controller import SETTINGS_FILENAME, AnnouncementVisit, ReceptionState
from ui.signal_tab import DENSITY_BINS, MAX_CONSTELLATION_POINTS
from ui.touch import backlight, settings_page, stations_page
from ui.touch.backlight import SETTINGS_DIM_AFTER, SETTINGS_OFF_AFTER
from ui.touch.display_choice import CONSTELLATION
from ui.touch.gnss_view import GLOBE, SKY
from ui.touch.guide_view import CAROUSEL, SCHEDULE
from ui.touch.icons import symbol_pixmap
from ui.touch.radio_page import (
    FALLBACK_LOGO_SIZE,
    ICON_SIZE,
    SLIDE_SIZE,
    Notice,
    announcement_notice,
    visit_notice,
)
from ui.touch.services_page import GNSS as GNSS_MENU
from ui.touch.services_page import GUIDE as GUIDE_MENU
from ui.touch.services_page import TRAFFIC as TRAFFIC_MENU
from ui.touch.settings_page import AUDIO, DISPLAY, MQTT, SDR, WIFI
from ui.touch.signal_page import OFDM, OVERVIEW
from ui.touch.signal_panels import ConstellationPanel
from ui.touch.stations import FAVOURITES
from ui.touch.theme import (
    DARK,
    FONT_FAMILY,
    LIGHT,
    SETTINGS_THEME,
    THEMES,
    TILE_HEIGHT,
    TILE_WIDTH,
    touch_font,
)
from ui.touch.traffic_view import MAP, MESSAGES
from ui.touch.widgets import (
    LONG_PRESS_MS,
    MARQUEE_REST_MS,
    MARQUEE_SPEED_PX_PER_S,
    EdgeFades,
    marquee_offset,
    one_line,
)
from ui.touch.window import (
    RADIO,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SERVICES,
    SETTINGS,
    SETTINGS_CONSTELLATION_DENSITY,
    SETTINGS_LOAD_SHOWN,
    SETTINGS_VOLUME_FIGURE_SHOWN,
    SIGNAL,
    STATIONS,
    TouchWindow,
    band_badge,
    date_text,
)


def test_the_options_of_the_radio_are_taken_and_the_rest_left_to_qt():
    arguments, qt_arguments = parse_arguments(
        ["main.py", "--touch", "-platform", "offscreen", "--fullscreen", "--system-volume"]
    )

    assert arguments.touch
    assert arguments.fullscreen
    assert arguments.system_volume
    assert qt_arguments == ["main.py", "-platform", "offscreen"]


def test_without_options_the_desktop_window_is_chosen():
    arguments, qt_arguments = parse_arguments(["main.py"])

    assert not arguments.touch
    assert not arguments.fullscreen
    assert not arguments.system_volume  # the system's volume stays with the user
    assert qt_arguments == ["main.py"]


def _spin(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def _store_service(data_directory, sid: int) -> None:
    from PyQt6.QtCore import QSettings

    settings = QSettings(str(data_directory / SETTINGS_FILENAME), QSettings.Format.IniFormat)
    settings.setValue("service/sid", sid)
    settings.sync()


@pytest.fixture
def touch_window(make_controller, monkeypatch):
    """Creates a touch window on a fresh controller and starts the controller, as `main`
    does; the restart after a device change follows at once."""
    monkeypatch.setattr(settings_page, "RESTART_DELAY_MS", 0)
    # Never the nmcli or the backlight of the computer the tests run on.
    monkeypatch.setattr(wifi, "find_nmcli", lambda: None)
    monkeypatch.setattr(backlight, "BACKLIGHT_CLASS", Path("nowhere") / "backlight")
    windows = []

    def make(data_directory) -> TouchWindow:
        controller = make_controller(data_directory, start=False)
        window = TouchWindow(controller)
        controller.start()
        windows.append(window)
        return window

    yield make
    for window in windows:
        window.deleteLater()


def test_the_window_is_one_screen_of_the_display_in_landscape(data_directory, touch_window):
    window = touch_window(data_directory)
    window.show()

    assert (window.width(), window.height()) == (SCREEN_WIDTH, SCREEN_HEIGHT)
    assert window.minimumSize() == window.maximumSize()


def test_the_rail_shows_the_weekday_above_the_date():
    monday = time.mktime((2026, 10, 5, 12, 0, 0, 0, 0, -1))

    assert date_text(monday) == "Monday\n2026-10-05"
    assert date_text(monday + 3 * 24 * 3600) == "Thursday\n2026-10-08"


def test_the_wifi_in_the_rail_marks_5_ghz_only():
    assert band_badge(wifi.BAND_5_GHZ) == "5"
    assert band_badge(wifi.BAND_2_4_GHZ) == band_badge(wifi.BAND_AUTOMATIC) == ""


def test_every_weekday_fits_the_rail(data_directory, touch_window, shipped_font):
    # In the typeface the display draws: the offscreen platform without it gives every
    # character the full font size.
    window = touch_window(data_directory)
    window.show()
    label = window._date_label
    button = window._nav_buttons[SETTINGS]
    monday = time.mktime((2026, 10, 5, 12, 0, 0, 0, 0, -1))
    for day in range(7):
        label.setText(date_text(monday + day * 24 * 3600))
        assert label.sizeHint().width() <= button.width(), label.text()


@pytest.fixture
def shipped_font():
    """The typeface of the touch surface as the application's font, as `main` sets it."""
    app = QApplication.instance()
    previous = app.font()
    app.setFont(touch_font())
    yield
    app.setFont(previous)


def test_every_name_in_the_rail_fits_its_button(data_directory, touch_window, shipped_font):
    # The offscreen platform does load the shipped typeface. In DejaVu Sans, which
    # Raspberry Pi OS draws otherwise, "Einstellungen" came out 106 px wide and was cut.
    window = touch_window(data_directory)
    window.show()

    for button in window._nav_buttons.values():
        assert button.font().family() == FONT_FAMILY
        assert QFontMetrics(button.font()).horizontalAdvance(button.text()) <= button.width()


def test_the_rail_shows_the_load_and_warns_when_decoding_falls_behind(data_directory, touch_window):
    window = touch_window(data_directory)

    window._controller.load_changed.emit(0.45, 0.03, 0.8)
    assert window._dsp_load_label.text() == "45 %"
    assert window._waiting_label.text() == "3 %"
    assert window._cpu_load_label.text() == "80 %"
    assert window._dsp_load_name.property("role") == "small"

    window._controller.load_changed.emit(0.95, 0.5, 1.2)
    assert window._dsp_load_name.property("role") == "error"
    assert window._dsp_load_label.property("role") == "error"


def test_the_load_fits_the_rail_with_all_its_cores_busy(data_directory, touch_window, shipped_font):
    window = touch_window(data_directory)
    window.set_load_shown(True)
    window.show()
    # A Raspberry Pi has four cores; the GUI can keep the GIL for all of a second.
    window._controller.load_changed.emit(1.5, 1.0, 4.0)
    _spin(0)  # the layouts settle with the pending events
    # Within the margins of the panel, in the typeface the display draws.
    widest = window._load_panel.layout().contentsRect().width()
    assert widest > 0
    rows = (
        (window._dsp_load_name, window._dsp_load_label),
        (window._waiting_name, window._waiting_label),
        (window._cpu_load_name, window._cpu_load_label),
    )

    assert all(name.sizeHint().width() + value.sizeHint().width() <= widest for name, value in rows)
    assert window.centralWidget().minimumSizeHint().height() <= SCREEN_HEIGHT


def test_without_a_station_of_the_last_session_the_station_page_opens(
    data_directory, touch_window, fake_workers
):
    window = touch_window(data_directory)

    window.start_playing()

    assert window.current_page() == STATIONS
    assert fake_workers == []


def test_the_station_of_the_last_session_plays_at_once(data_directory, touch_window, fake_workers):
    _store_service(data_directory, SERVICES_5C[1][0])
    window = touch_window(data_directory)

    window.start_playing()

    assert window.current_page() == RADIO
    assert window._controller.state is ReceptionState.CONNECTING
    assert window._radio_page._name_label.text() == "Dlf Kultur"


def test_the_radio_page_names_the_other_channels_of_the_station(dlf_kultur_elsewhere, touch_window):
    _store_service(dlf_kultur_elsewhere, SERVICES_5C[1][0])
    window = touch_window(dlf_kultur_elsewhere)
    label = window._radio_page._ensemble_label

    assert label.text() == f"5C [11D, 12D?] · {ENSEMBLE_5C}"

    window._controller.select_service(SERVICES_5C[0][0])  # Dlf: only here
    assert label.text() == f"5C · {ENSEMBLE_5C}"


def test_a_tapped_tile_plays_its_station(data_directory, touch_window, fake_workers):
    window = touch_window(data_directory)
    window.show_page(STATIONS)

    window._stations_page._tiles[("5C", SERVICES_5C[0][0])].click()

    assert window.current_page() == STATIONS
    assert window._controller.selected_sid == SERVICES_5C[0][0]
    assert len(fake_workers) == 1
    assert window._stations_page._tiles[("5C", SERVICES_5C[0][0])].property("current")


def test_next_and_previous_step_through_the_filtered_stations(
    data_directory, touch_window, fake_workers
):
    cache = EnsembleCache(data_directory / receiver_controller.ENSEMBLE_CACHE_FILENAME)
    cache.load()
    cache.put("11D", EnsembleInfo("NDR NDS", SERVICES_11D, 0.0, SOURCE_SCAN))
    cache.save()
    window = touch_window(data_directory)
    controller = window._controller
    controller.play_service("5C", SERVICES_5C[1][0])
    fake_workers[-1].connected.emit("Verbunden.")
    radio = window._radio_page

    radio._next_button.click()
    assert (controller.channel, controller.selected_sid) == ("11D", SERVICES_11D[0][0])
    assert fake_workers[-1].retuned_to  # one reception, retuned to 11D

    window._directory.set_filter("5C")
    radio._previous_button.click()
    assert (controller.channel, controller.selected_sid) == ("5C", SERVICES_5C[1][0])


def test_the_radio_page_names_the_filter_the_steps_follow(data_directory, touch_window):
    cache = EnsembleCache(data_directory / receiver_controller.ENSEMBLE_CACHE_FILENAME)
    cache.load()
    cache.put("11D", EnsembleInfo("NDR NDS", SERVICES_11D, 0.0, SOURCE_SCAN))
    cache.save()
    window = touch_window(data_directory)
    label = window._radio_page._filter_label

    assert label.text() == "All stations (3)"
    assert not label.isHidden()

    window._directory.set_filter("5C")
    assert label.text() == f"5C · {ENSEMBLE_5C} (2)"

    window._directory.set_filter(FAVOURITES)
    assert label.text() == "Favourites (0)"
    window._controller.set_favourite("5C", SERVICES_5C[1][0], True)
    assert label.text() == "Favourites (1)"


def test_without_a_known_station_the_radio_page_names_no_filter(tmp_path, touch_window):
    window = touch_window(tmp_path)

    assert window._radio_page._filter_label.isHidden()
    assert window._radio_page._ensemble_label.text() == "Tap a station on the “Stations” page."


def test_the_filter_chips_narrow_the_tiles(data_directory, touch_window):
    cache = EnsembleCache(data_directory / receiver_controller.ENSEMBLE_CACHE_FILENAME)
    cache.load()
    cache.put("11D", EnsembleInfo("NDR NDS", SERVICES_11D, 0.0, SOURCE_SCAN))
    cache.save()
    window = touch_window(data_directory)
    page = window._stations_page

    chip = next(b for b in page._chip_group.buttons() if b.property("channel") == "11D")
    chip.click()

    assert list(page._tiles) == [("11D", SERVICES_11D[0][0])]


def test_the_favourites_come_first_among_the_chips(data_directory, touch_window):
    window = touch_window(data_directory)
    page = window._stations_page

    chips = page._chip_group.buttons()
    assert [chip.text() for chip in chips] == [
        "Favourites (0)",
        "All (2)",
        f"5C · {ENSEMBLE_5C} (2)",
    ]
    assert not chips[0].icon().isNull()

    chips[0].click()
    # None marked yet: the page says how to mark one.
    assert page._grid_area.isHidden()
    assert not page._empty_label.isHidden()
    assert page._empty_label.text().startswith("No favourites yet.")


def test_a_long_press_on_a_tile_marks_its_station(data_directory, touch_window):
    window = touch_window(data_directory)
    controller = window._controller
    page = window._stations_page
    dlf, dlf_kultur = (("5C", sid) for sid, _label in SERVICES_5C)
    assert not page._tiles[dlf_kultur].has_star()

    page._tiles[dlf_kultur].long_pressed.emit()
    assert controller.is_favourite(*dlf_kultur)
    assert page._tiles[dlf_kultur].has_star()
    assert not page._tiles[dlf].has_star()
    favourites_chip = page._chip_group.buttons()[0]
    assert favourites_chip.text() == "Favourites (1)"
    # Wider by the count: the row is measured again.
    assert favourites_chip.width() == favourites_chip.sizeHint().width()

    window._directory.set_filter(FAVOURITES)
    assert list(page._tiles) == [dlf_kultur]
    assert page._empty_label.isHidden()

    # Taken out again, it leaves the favourites.
    page._tiles[dlf_kultur].long_pressed.emit()
    assert not controller.is_favourite(*dlf_kultur)
    assert page._tiles == {}
    assert not page._empty_label.isHidden()
    assert favourites_chip.text() == "Favourites (0)"


def test_a_finger_resting_on_a_tile_is_no_tap(data_directory, touch_window):
    """Pressed on the tile itself, not through the scroll area: QTest into a viewport
    grabbed by QScroller crashed later tests now and then."""
    tile = stations_page._StationTile()
    tile.resize(TILE_WIDTH, TILE_HEIGHT)
    tile.show()
    clicks, presses = [], []
    tile.clicked.connect(lambda: clicks.append(True))
    tile.long_pressed.connect(lambda: presses.append(True))
    centre = QPoint(TILE_WIDTH // 2, TILE_HEIGHT // 2)
    # The windows earlier tests let go of are deleted in the next event loop, which would
    # be the one the long press is waited for in: after other modules that took up to a
    # second, and the press came too late (2026-10-12).
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)

    QTest.mousePress(tile, Qt.MouseButton.LeftButton, pos=centre)
    _spin(LONG_PRESS_MS + 200)
    QTest.mouseMove(tile, centre + QPoint(4, 0))
    QTest.mouseRelease(tile, Qt.MouseButton.LeftButton, pos=centre + QPoint(4, 0))
    assert (presses, clicks) == ([True], [])

    QTest.mouseClick(tile, Qt.MouseButton.LeftButton, pos=centre)
    assert (presses, clicks) == ([True], [True])

    # Moved, the finger scrolls: no long press.
    QTest.mousePress(tile, Qt.MouseButton.LeftButton, pos=centre)
    QTest.mouseMove(tile, centre + QPoint(0, 40))
    _spin(LONG_PRESS_MS + 200)
    QTest.mouseRelease(tile, Qt.MouseButton.LeftButton, pos=centre + QPoint(0, 40))
    assert presses == [True]
    tile.deleteLater()


def test_the_star_on_the_radio_page_marks_the_station_playing(data_directory, touch_window):
    window = touch_window(data_directory)
    controller = window._controller
    radio = window._radio_page
    button = radio._favourite_button
    assert not button.isEnabled()  # no station chosen

    sid = SERVICES_5C[1][0]
    controller.play_service("5C", sid)
    assert button.isEnabled()
    assert not button.isChecked()

    button.click()
    assert controller.is_favourite("5C", sid)
    assert window._stations_page._tiles[("5C", sid)].has_star()

    # Taken out on the station page, the star goes out here as well.
    window._stations_page._tiles[("5C", sid)].long_pressed.emit()
    assert not button.isChecked()


def test_the_star_sits_on_the_corner_of_the_logo(data_directory, touch_window):
    window = touch_window(data_directory)
    window.show()
    radio = window._radio_page
    window._controller.play_service("5C", SERVICES_5C[1][0])
    _spin(50)

    logo = radio._logo_label.geometry()
    badge = radio._favourite_button.geometry()
    assert radio._favourite_button.isVisible()
    assert logo.contains(badge.center())
    assert badge.right() > logo.right() and badge.bottom() > logo.bottom()


def test_next_and_previous_step_through_the_favourites(data_directory, touch_window, fake_workers):
    cache = EnsembleCache(data_directory / receiver_controller.ENSEMBLE_CACHE_FILENAME)
    cache.load()
    cache.put("11D", EnsembleInfo("NDR NDS", SERVICES_11D, 0.0, SOURCE_SCAN))
    cache.save()
    window = touch_window(data_directory)
    controller = window._controller
    controller.set_favourite("5C", SERVICES_5C[0][0], True)
    controller.set_favourite("11D", SERVICES_11D[0][0], True)
    window._directory.set_filter(FAVOURITES)
    controller.play_service("5C", SERVICES_5C[0][0])
    fake_workers[-1].connected.emit("Verbunden.")

    window._radio_page._next_button.click()
    assert (controller.channel, controller.selected_sid) == ("11D", SERVICES_11D[0][0])
    window._radio_page._next_button.click()
    assert (controller.channel, controller.selected_sid) == ("5C", SERVICES_5C[0][0])


def test_many_ensembles_keep_their_full_names_and_scroll_sideways(data_directory, touch_window):
    cache = EnsembleCache(data_directory / receiver_controller.ENSEMBLE_CACHE_FILENAME)
    cache.load()
    for index, channel in enumerate(("6A", "7B", "9C", "10B", "11B", "12A", "12D")):
        label = f"Ensemble mit langem Namen {index}"
        cache.put(channel, EnsembleInfo(label, ((0xE000 + index, "X"),), 0.0, SOURCE_SCAN))
    cache.save()
    window = touch_window(data_directory)
    window.show()
    window.show_page(STATIONS)
    page = window._stations_page

    chips = page._chip_group.buttons()
    assert len(chips) == 10  # Favoriten, Alle and eight ensembles
    assert all(chip.width() == chip.sizeHint().width() for chip in chips)
    assert page._chips_area.horizontalScrollBar().maximum() > 0

    chips[-1].click()
    assert page._chips_area.horizontalScrollBar().value() > 0  # the chosen one in view


def test_the_chips_fade_out_where_more_follow(data_directory, touch_window):
    cache = EnsembleCache(data_directory / receiver_controller.ENSEMBLE_CACHE_FILENAME)
    cache.load()
    for index, channel in enumerate(("6A", "7B", "9C", "10B")):
        label = f"Ensemble mit langem Namen {index}"
        cache.put(channel, EnsembleInfo(label, ((0xE000 + index, "X"),), 0.0, SOURCE_SCAN))
    cache.save()
    window = touch_window(data_directory)
    window.show()
    window.show_page(STATIONS)
    page = window._stations_page
    fades = page._chip_fades
    area = page._chips_area
    bar = area.horizontalScrollBar()

    assert (fades.start.isVisible(), fades.end.isVisible()) == (False, True)
    assert fades.end.geometry().right() == area.width() - 1
    assert fades.end.height() == area.height()
    # Taps go through to the chip beneath.
    assert fades.end.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    bar.setValue(bar.maximum() // 2)
    assert (fades.start.isVisible(), fades.end.isVisible()) == (True, True)

    page._chip_group.buttons()[-1].click()
    assert bar.value() == bar.maximum()
    assert (fades.start.isVisible(), fades.end.isVisible()) == (True, False)
    # A chip chosen between others stands clear of the fades.
    chip = page._chip_group.buttons()[4]
    chip.click()
    assert (fades.start.isVisible(), fades.end.isVisible()) == (True, True)
    assert chip.mapTo(area, QPoint(0, 0)).x() >= fades.start.width()
    assert chip.mapTo(area, QPoint(chip.width(), 0)).x() <= fades.end.x()


def test_chips_that_all_fit_do_not_fade(data_directory, touch_window):
    window = touch_window(data_directory)
    window.show()
    window.show_page(STATIONS)
    fades = window._stations_page._chip_fades

    assert window._stations_page._chips_area.horizontalScrollBar().maximum() == 0
    assert not fades.start.isVisible()
    assert not fades.end.isVisible()


def test_the_station_page_opens_at_the_station_playing(data_directory, touch_window, fake_workers):
    window = touch_window(data_directory)
    window.show()
    window.show_page(STATIONS)
    window.show_page(RADIO)
    # Many more stations come while the page is hidden, the one playing far down.
    controller = window._controller
    for index, channel in enumerate(("6A", "7B", "9C", "10B", "11B", "12A", "12D")):
        services = tuple((0xE000 + 16 * index + n, f"Sender {n}") for n in range(8))
        controller.ensemble_cache.put(
            channel, EnsembleInfo(f"Ensemble {index}", services, 0.0, SOURCE_SCAN)
        )
        controller.channel_stored.emit(channel)
    playing = ("12A", 0xE000 + 16 * 5 + 3)
    controller.play_service(*playing)

    window.show_page(STATIONS)
    _spin(0)

    page = window._stations_page
    bar = page._grid_area.verticalScrollBar()
    viewport = page._grid_area.viewport()
    tile = page._tiles[playing]
    top = tile.mapTo(viewport, QPoint(0, 0)).y()
    assert bar.value() > 0
    assert 0 <= top and top + tile.height() <= viewport.height()


class _Paints(QObject):
    """Counts the paint events of the widgets it filters."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Paint:
            self.count += 1
        return False


def test_the_chosen_scheme_is_kept(data_directory, touch_window):
    window = touch_window(data_directory)
    window.show()
    window.show_page(SETTINGS)
    window._settings_page.show_subpage(DISPLAY)
    QApplication.processEvents()

    row = window._settings_page._theme
    row.open()
    sheet = row.sheet()
    assert sheet.isVisible()
    next(r for r in sheet.rows if r.key == LIGHT).click()

    # The sheet closes and the row shows the choice at once, the window restyles
    # itself a moment later.
    assert not sheet.isVisible() and row.text() == "Light"
    assert window._theme.key == DARK
    _spin(settings_page.THEME_FEEDBACK_MS + 50)
    assert window._theme.key == LIGHT
    assert window._controller.settings.value(SETTINGS_THEME) == LIGHT


def test_a_second_tap_on_the_scheme_settles_it(data_directory, touch_window, monkeypatch):
    window = touch_window(data_directory)
    window.show_page(SETTINGS)
    restyled = []
    monkeypatch.setattr(window, "setStyleSheet", restyled.append)
    row = window._settings_page._theme

    # Back to the scheme showing before it was switched: nothing to restyle.
    row.choose(LIGHT)
    row.choose(DARK)
    _spin(settings_page.THEME_FEEDBACK_MS + 50)
    assert window._theme.key == DARK
    assert restyled == []

    row.choose(LIGHT)
    row.choose(DARK)
    row.choose(LIGHT)
    _spin(settings_page.THEME_FEEDBACK_MS + 50)
    assert window._theme.key == LIGHT
    assert len(restyled) == 1


def test_the_list_of_the_settings_page_chooses_its_sub_page(data_directory, touch_window):
    window = touch_window(data_directory)
    window.show()
    window.show_page(SETTINGS)
    page = window._settings_page
    rows = page._categories.rows
    assert page.current_subpage() == SDR
    assert rows[SDR].property("selected") and not rows[DISPLAY].property("selected")
    assert page._device.isVisible()
    assert page._device.subtitle.isVisible()  # a change restarts the reception

    rows[DISPLAY].click()
    assert page.current_subpage() == DISPLAY
    assert rows[DISPLAY].property("selected") and not rows[SDR].property("selected")
    assert page._theme.isVisible() and page._load_shown.isVisible()
    assert page._constellation.isVisible()
    assert page._dim_after.isVisible() and page._off_after.isVisible()
    assert page._backlight_hint.isVisible()  # no backlight on the PC
    assert page._latitude.isVisible() and page._height.isVisible()
    assert not page._device.isVisible()

    rows[AUDIO].click()
    assert page.current_subpage() == AUDIO
    assert page._equalizer_switch.isVisible() and page._equalizer_scope.isVisible()
    assert page._equalizer_reset.isVisible()
    assert page._announcement_switch.isVisible()
    assert not page._latitude.isVisible()
    assert not page._wifi_view.message.isVisible()

    rows[WIFI].click()
    assert page.current_subpage() == WIFI
    assert page._wifi_view.isVisible()
    assert page._wifi_view.message.isVisible()


def test_a_settings_sub_page_taller_than_the_page_scrolls_and_fades_at_the_edge(
    data_directory, touch_window
):
    window = touch_window(data_directory)
    window.show()
    window.show_page(SETTINGS)
    page = window._settings_page
    page.show_subpage(AUDIO)
    area = page._stack.currentWidget()
    fades = area.findChild(EdgeFades)
    bar = area.verticalScrollBar()
    switch = page._announcement_switch

    # The last row lies below the edge; the fade there says that there is more.
    assert bar.maximum() > 0
    bottom = switch.mapTo(area.viewport(), QPoint(0, switch.height())).y()
    assert bottom > area.viewport().height()
    assert (fades.start.isVisible(), fades.end.isVisible()) == (False, True)
    assert fades.end.geometry().bottom() == area.height() - 1

    bar.setValue(bar.maximum())
    assert (fades.start.isVisible(), fades.end.isVisible()) == (True, False)
    bottom = switch.mapTo(area.viewport(), QPoint(0, switch.height())).y()
    assert bottom <= area.viewport().height()

    # Never deeper than what lies beyond: a few pixels too many dim no whole row.
    bar.setValue(bar.maximum() - 3)
    assert fades.end.isVisible() and fades.end.height() == 3

    # The SDR rows fit: nothing to scroll, nothing faded.
    page.show_subpage(SDR)
    area = page._stack.currentWidget()
    fades = area.findChild(EdgeFades)
    assert area.verticalScrollBar().maximum() == 0
    assert not fades.start.isVisible() and not fades.end.isVisible()


def test_the_settings_page_shows_the_version_bottom_right_on_every_sub_page(
    data_directory, touch_window, shipped_font
):
    window = touch_window(data_directory)
    window.show()
    window.show_page(SETTINGS)
    page = window._settings_page
    rows = list(page._categories.rows.values())
    for key in (SDR, DISPLAY, AUDIO, WIFI, MQTT):
        page.show_subpage(key)
        assert page._version.isVisible()
        assert page._version.text() == f"Version {receiver_controller.APPLICATION_VERSION}"
    # Below the list of categories, left of the way out of the radio, which keeps the
    # right margin.
    version = page._version.geometry()
    quit_button = next(b for b in page.findChildren(QPushButton) if b.text() == "Quit radio")
    assert version.top() > max(row.mapTo(page, row.rect().bottomLeft()).y() for row in rows)
    assert version.right() < quit_button.geometry().left()
    # The eight categories have their names in full in the shipped typeface.
    for row in rows:
        name = row.name
        assert QFontMetrics(name.font()).horizontalAdvance(name.text()) <= name.width(), name.text()


def test_the_equalizer_is_switched_and_chosen_for_all_or_each_station_in_the_settings(
    data_directory, touch_window
):
    window = touch_window(data_directory)
    page = window._settings_page
    controller = window._controller
    assert not page._equalizer_switch.isChecked()

    page._equalizer_switch.click()
    assert controller.equalizer_enabled
    assert page._equalizer_switch.isChecked()
    # Also when switched on the equalizer page itself.
    window._radio_page.equalizer.switch.click()
    assert not page._equalizer_switch.isChecked()

    page._equalizer_scope.choose(True)
    assert controller.equalizer_per_station
    window._controller.shutdown()
    assert touch_window(data_directory)._settings_page._equalizer_scope.current() is True


def test_the_ceiling_of_the_volume_in_the_settings_moves_the_whole_slider(
    data_directory, touch_window
):
    window = touch_window(data_directory)
    page = window._settings_page
    radio = window._radio_page
    controller = window._controller
    assert page._volume_ceiling.value() == 0
    assert radio._volume_slider.value() == 70
    assert radio._volume_label.text() == "−18 dB"

    radio._volume_slider.setValue(80)
    assert radio._volume_label.text() == "−12 dB"
    page._volume_ceiling._minus.click()
    assert controller.volume_ceiling_db == -1
    assert radio._volume_slider.value() == 80
    assert radio._volume_label.text() == "−13 dB"
    radio._volume_slider.setValue(0)
    assert radio._volume_label.text() == "muted"
    window._controller.shutdown()
    assert touch_window(data_directory)._settings_page._volume_ceiling.value() == -1


def test_the_rows_of_the_settings_lead_to_the_equalizer_and_the_recordings_and_back(
    data_directory, touch_window
):
    window = touch_window(data_directory)
    window.show_page(SETTINGS)
    radio = window._radio_page

    window._settings_page._equalizer_button.click()
    assert window.current_page() == RADIO and radio.in_equalizer()
    # "Back" leads to the settings the equalizer was opened from.
    radio.equalizer.back_button.click()
    assert window.current_page() == SETTINGS and not radio.in_equalizer()

    window._settings_page._recording_view._show_recordings.click()
    assert window.current_page() == RADIO and radio.in_recordings()
    radio.recordings.back_button.click()
    assert window.current_page() == SETTINGS and not radio.in_recordings()

    # The page of the navigation leads back to the player, and "Back" on the
    # equalizer opened there stays on the radio page.
    window._settings_page._equalizer_button.click()
    window._on_nav_clicked(RADIO)
    assert window.current_page() == RADIO and not radio.in_equalizer()
    radio.show_equalizer()
    radio.equalizer.back_button.click()
    assert window.current_page() == RADIO and not radio.in_equalizer()


def test_deleting_all_equalizer_settings_asks_first(data_directory, touch_window):
    window = touch_window(data_directory)
    window.show()
    controller = window._controller
    controller.set_equalizer(EqualizerSettings.from_preset(BASS))
    controller.set_equalizer_enabled(True)
    page = window._settings_page

    page._equalizer_reset.click()
    sheet = page.action_sheet()
    assert sheet.isVisible()
    sheet.dismiss()  # "Cancel"
    assert controller.equalizer_enabled

    page._equalizer_reset.click()
    sheet.actions[0].click()  # "Delete"
    assert not sheet.isVisible()
    assert controller.equalizer_state == EqualizerState()
    assert not page._equalizer_switch.isChecked()


def test_the_button_beside_the_loudspeaker_opens_the_equalizer(data_directory, touch_window):
    window = touch_window(data_directory)
    radio = window._radio_page

    radio._equalizer_button.click()
    assert radio.in_equalizer()
    radio.equalizer.back_button.click()
    assert not radio.in_equalizer()


def test_the_button_beside_the_loudspeaker_shows_whether_the_equalizer_is_on(
    data_directory, touch_window
):
    window = touch_window(data_directory)
    controller = window._controller
    radio = window._radio_page
    button = radio._equalizer_button

    def symbol(color: str) -> QImage:
        return symbol_pixmap("equalizer", radio._theme.color(color), ICON_SIZE.width()).toImage()

    assert not button.property("active")
    assert button.icon().pixmap(ICON_SIZE).toImage() == symbol("text")

    controller.set_equalizer_enabled(True)
    assert button.property("active")
    assert button.icon().pixmap(ICON_SIZE).toImage() == symbol("accent")
    # Not a switch: a tap opens the equalizer and leaves it on.
    button.click()
    assert radio.in_equalizer() and controller.equalizer_enabled and button.property("active")
    radio.equalizer.back_button.click()

    # Per station: the station playing switches its own off, another follows the one of
    # all stations.
    controller.select_service(SERVICES_5C[0][0])
    controller.set_equalizer_per_station(True)
    controller.set_equalizer_enabled(False)
    assert not button.property("active")
    controller.select_service(SERVICES_5C[1][0])
    assert button.property("active")
    controller.select_service(SERVICES_5C[0][0])
    assert not button.property("active")
    # The theme changes the colour of the symbol, not what it tells.
    window.set_theme(LIGHT)
    assert button.icon().pixmap(ICON_SIZE).toImage() == symbol("text")


def test_the_load_in_the_rail_is_hidden_until_chosen_and_stays_so(data_directory, touch_window):
    window = touch_window(data_directory)
    page = window._settings_page
    assert not window._load_panel.isVisibleTo(window)
    assert not page._load_shown.isChecked()

    page._load_shown.click()
    assert window._load_panel.isVisibleTo(window)
    assert window._controller.settings.value(SETTINGS_LOAD_SHOWN, type=bool) is True

    window._controller.shutdown()
    window = touch_window(data_directory)
    assert window._load_panel.isVisibleTo(window)
    assert window._settings_page._load_shown.isChecked()

    window._settings_page._load_shown.click()
    window._controller.shutdown()
    window = touch_window(data_directory)
    assert not window._load_panel.isVisibleTo(window)


def test_the_level_beside_the_volume_slider_is_hidden_until_chosen_and_stays_so(
    data_directory, touch_window
):
    window = touch_window(data_directory)
    page = window._settings_page
    label = window._radio_page._volume_label
    assert not label.isVisibleTo(window)
    assert not page._volume_figure_shown.isChecked()

    page._volume_figure_shown.click()
    assert label.isVisibleTo(window)
    assert window._controller.settings.value(SETTINGS_VOLUME_FIGURE_SHOWN, type=bool) is True

    window._controller.shutdown()
    window = touch_window(data_directory)
    assert window._radio_page._volume_label.isVisibleTo(window)
    assert window._settings_page._volume_figure_shown.isChecked()


def test_the_constellation_is_drawn_as_points_or_as_a_density_on_both_pages(
    data_directory, touch_window
):
    window = touch_window(data_directory)
    page = window._settings_page
    on_signal = window._signal_page.page(OVERVIEW).constellation
    dsp = window._controller.dsp_worker
    rng = np.random.default_rng(0)
    # A frame of noisy D-QPSK, more points than the sample of the point cloud draws.
    count = 2 * MAX_CONSTELLATION_POINTS
    points = (rng.choice([-1.0, 1.0], count) + 1j * rng.choice([-1.0, 1.0], count)) / np.sqrt(2)
    points += 0.1 * (rng.normal(size=count) + 1j * rng.normal(size=count))
    points = points.astype(np.complex64)

    # Points at first, a sample of them.
    assert page._constellation.current() is False
    window.show_page(SIGNAL)
    dsp.constellation_ready.emit(points)
    window._render_plots()
    assert on_signal.curve.isVisible() and not on_signal.image.isVisible()
    assert len(on_signal.curve.getData()[0]) == MAX_CONSTELLATION_POINTS

    # The density counts every point; what is on show is redrawn at once.
    page._constellation.choose(True)
    assert window._controller.settings.value(SETTINGS_CONSTELLATION_DENSITY, type=bool) is True
    assert on_signal.image.isVisible() and not on_signal.curve.isVisible()
    assert on_signal.image.image.shape == (DENSITY_BINS, DENSITY_BINS)
    assert np.sum(on_signal.image.image**2) == pytest.approx(count)
    # No point stands in the colour of the panel, the densest square in that of the text.
    window.set_theme(LIGHT)
    surface, text = THEMES[LIGHT].color("surface"), THEMES[LIGHT].color("text")
    assert tuple(on_signal.image.lut[0]) == (surface.red(), surface.green(), surface.blue())
    assert tuple(on_signal.image.lut[-1]) == (text.red(), text.green(), text.blue())

    # The panel on the radio page, made only now, draws the density too.
    radio = window._radio_page
    radio.show_display(CONSTELLATION)
    on_radio = radio._slide_area.currentWidget()
    assert isinstance(on_radio, ConstellationPanel) and on_radio.density
    window.show_page(RADIO)
    dsp.constellation_ready.emit(points)
    window._render_plots()
    assert on_radio.image.isVisible() and on_radio.image.image is not None

    # Back to points, on both pages.
    page._constellation.choose(False)
    assert window._controller.settings.value(SETTINGS_CONSTELLATION_DENSITY, type=bool) is False
    assert on_radio.curve.isVisible() and not on_radio.image.isVisible()
    assert len(on_radio.curve.getData()[0]) == MAX_CONSTELLATION_POINTS
    assert on_signal.curve.isVisible() and not on_signal.image.isVisible()

    # The choice outlasts a restart, also for the panel made with the radio page.
    page._constellation.choose(True)
    window._controller.shutdown()
    window = touch_window(data_directory)
    assert window._settings_page._constellation.current() is True
    assert window._signal_page.page(OVERVIEW).constellation.density
    assert window._radio_page._slide_area.currentWidget().density


def test_the_times_of_the_screen_are_set_on_the_settings_page_and_kept(
    data_directory, touch_window, shipped_font
):
    window = touch_window(data_directory)
    window.show()
    window.show_page(SETTINGS)
    page = window._settings_page
    assert page._dim_after.value() == 0 and page._off_after.value() == 0
    assert {label.text() for label in page.findChildren(QLabel)} >= {"Dim", "Switch off"}

    page._dim_after._plus.click()
    page._off_after._plus.click()
    page._off_after._plus.click()
    assert window._screen.dim_after_s == 15 and window._screen.off_after_s == 30
    assert window._controller.settings.value(SETTINGS_DIM_AFTER, type=float) == 15
    assert window._controller.settings.value(SETTINGS_OFF_AFTER, type=float) == 30

    window._controller.shutdown()
    window = touch_window(data_directory)
    assert window._settings_page._dim_after.value() == 15
    assert window._settings_page._off_after.value() == 30


def test_switching_to_announcements_is_set_on_the_settings_page_and_kept(
    data_directory, touch_window
):
    window = touch_window(data_directory)
    switch = window._settings_page._announcement_switch
    assert not switch.isChecked()

    switch.click()
    assert window._controller.announcement_switching

    window._controller.shutdown()
    assert touch_window(data_directory)._settings_page._announcement_switch.isChecked()


@pytest.mark.parametrize("device", list(receiver_controller.DEVICES))
def test_the_settings_page_fits_the_screen_with_every_device(data_directory, touch_window, device):
    window = touch_window(data_directory)
    window.show()
    window.show_page(SETTINGS)
    page = window._settings_page

    page._device.set_current(device)
    page._show_device_rows()

    assert page.minimumSizeHint().height() <= SCREEN_HEIGHT


def test_the_wlan_sub_page_fits_the_screen_with_a_long_list(data_directory, touch_window):
    window = touch_window(data_directory)
    window.show()
    window.show_page(SETTINGS)
    page = window._settings_page
    page.show_subpage(WIFI)

    assert page.minimumSizeHint().height() <= SCREEN_HEIGHT


def test_an_error_is_shown_until_tapped_away(data_directory, touch_window, fake_workers):
    window = touch_window(data_directory)
    window.show()
    window._controller.start_reception()

    fake_workers[-1].error.emit("No RTL-SDR found.")
    assert window._banner.isVisible()
    assert window._banner.text() == "Error: No RTL-SDR found."

    window._banner.mousePressEvent(None)
    assert not window._banner.isVisible()


def test_a_device_change_during_reception_restarts_it(data_directory, touch_window, fake_workers):
    window = touch_window(data_directory)
    controller = window._controller
    controller.start_reception()
    fake_workers[-1].connected.emit("Verbunden.")

    window._settings_page._gain._plus.click()
    _spin(50)

    assert controller.device_settings.gain_db > receiver_controller.DEFAULT_GAIN_DB
    assert fake_workers[0].stop_requested
    fake_workers[0].stopped.emit()
    assert len(fake_workers) == 2
    assert controller.state is ReceptionState.CONNECTING


def test_the_rspduo_is_one_device_with_its_tuner_beneath(data_directory, touch_window):
    window = touch_window(data_directory)
    window.show()
    window.show_page(SETTINGS)
    page = window._settings_page
    page._device.open()
    sheet = page._device.sheet()
    assert sheet.isVisible() and sheet.panel.width() <= window.width()
    assert [row.name.text() for row in sheet.rows] == ["RTL-SDR", "RSPduo", "RTL_TCP"]
    assert not page._tuner.isVisible()

    sheet.rows[1].click()
    assert page._tuner.isVisible() and page._tuner.text() == "Tuner 1"
    assert page._lna.isVisible() and not page._gain.isVisible()
    page._tuner.open()
    sheet = page._tuner.sheet()
    assert [row.name.text() for row in sheet.rows] == ["Tuner 1", "Tuner 2"]
    sheet.rows[1].click()

    settings = window._controller.device_settings
    assert (settings.device, settings.rspduo_tuner) == ("rspduo", 2)
    assert page._tuner.text() == "Tuner 2"
    assert page._categories.rows[SDR].state.text() == "RSPduo · Tuner 2"

    page._device.choose("rtlsdr")
    assert not page._tuner.isVisible()


def test_rtl_tcp_takes_its_server_and_the_rows_of_the_stick(data_directory, touch_window):
    window = touch_window(data_directory)
    window.show()
    window.show_page(SETTINGS)
    page = window._settings_page
    assert not page._server_host.isVisible()

    page._device.choose("rtltcp")
    assert page._server_host.isVisible() and page._server_port.isVisible()
    assert page._gain.isVisible()
    assert not page._lna.isVisible()
    assert page._server_host.text() == "Not set"

    page._server_host.click()
    keyboard = page.keyboard()
    for character in "radio-pi":
        keyboard.press(character)
    keyboard.accept()
    page._server_port.click()
    pad = page.number_pad()
    for key in "1235":
        pad.press(key)
    pad._on_accept_clicked()

    settings = window._controller.device_settings
    assert settings.rtl_tcp_host == "radio-pi"
    assert settings.rtl_tcp_port == 1235
    assert (page._server_host.text(), page._server_port.text()) == ("radio-pi", "1235")
    page._device.choose("rspduo")
    assert not page._server_host.isVisible()
    assert not page._gain.isVisible()
    assert page._lna.isVisible()


def test_back_from_a_missing_rspduo_the_stick_plays_on(data_directory, touch_window, fake_workers):
    """The error of the RSPduo goes with the next attempt, without a tap on the banner
    or on the play button; the settings page does not restart that attempt again."""
    window = touch_window(data_directory)
    window.show()
    controller = window._controller
    page = window._settings_page
    controller.start_reception()
    fake_workers[-1].connected.emit("Verbunden.")

    page._device.choose("rspduo")
    _spin(50)
    fake_workers[0].stopped.emit()
    assert fake_workers[-1].rspduo.tuner == 1
    fake_workers[-1].error.emit("Kein RSPduo gefunden.")
    assert window._banner.isVisible()

    page._device.choose("rtlsdr")
    assert not window._banner.isVisible()
    assert len(fake_workers) == 3
    assert controller.state is ReceptionState.CONNECTING
    fake_workers[-1].connected.emit("Verbunden.")
    _spin(50)
    assert not fake_workers[-1].stop_requested


def test_a_scan_during_reception_stops_it_and_resumes_after(
    data_directory, touch_window, fake_workers, fake_scanners
):
    window = touch_window(data_directory)
    controller = window._controller
    controller.play_service("5C", SERVICES_5C[0][0])
    fake_workers[-1].connected.emit("Verbunden.")
    page = window._stations_page

    page._scan_button.click()
    assert fake_workers[0].stop_requested
    fake_workers[0].stopped.emit()
    assert controller.state is ReceptionState.SCANNING

    info = EnsembleInfo(ENSEMBLE_5C, SERVICES_5C, 0.0, SOURCE_SCAN, fib_success=1.0)
    fake_scanners[0].channel_scanned.emit("5C", info)
    fake_scanners[0].scan_finished.emit(True)
    assert page._status_label.text().startswith("Scan finished")
    _spin(50)

    assert len(fake_workers) == 2
    assert controller.selected_sid == SERVICES_5C[0][0]


def test_closing_the_window_shuts_the_controller_down(data_directory, touch_window):
    window = touch_window(data_directory)
    audio_thread = window._controller._audio_thread

    window.close()

    assert not audio_thread.isRunning()


def _png(width: int, height: int) -> bytes:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(0x3366AA)
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return bytes(buffer.data())


def test_a_slide_is_shown_at_twice_its_size_until_the_station_changes(data_directory, touch_window):
    window = touch_window(data_directory)
    radio = window._radio_page
    untriggered = MotObject(1, CONTENT_TYPE_IMAGE, 3, "alt.png", _png(320, 240))
    # TriggerTime "NOW": a validity bit of 0 (ETSI TS 101 499, clause 6.2.4).
    triggered = MotObject(
        2, CONTENT_TYPE_IMAGE, 3, "bild.png", _png(320, 240), {PARAM_TRIGGER_TIME: bytes(4)}
    )

    # Without a TriggerTime an image is kept, not shown (table 2), as on the desktop.
    window._controller.dsp_worker.slide_ready.emit(untriggered)
    assert radio._slide_label.pixmap().width() == FALLBACK_LOGO_SIZE
    window._controller.dsp_worker.slide_ready.emit(triggered)
    assert radio._slide_label.pixmap().size() == SLIDE_SIZE

    window._controller.select_service(SERVICES_5C[0][0])
    assert radio._slide_label.pixmap().width() == FALLBACK_LOGO_SIZE


# Traffic news for cluster 2 in subchannel 5, which carries the service 0xD382.
TRAFFIC = Announcement(2, 0x0002, new=True, subchannel_id=5)
TRAFFIC_STATUS = AnnouncementStatus(
    AnnouncementSupport(0x0002, (2,)),
    alarm_enabled=False,
    active=TRAFFIC,
    source_sid=0xD382,
    source_component=AudioComponent(5, 63),
)


def test_the_announcement_notice_says_where_the_announcement_is_heard():
    def notice(selected_sid=0xD381, source="NDR 2", switched=False, switching_enabled=True):
        return announcement_notice(
            TRAFFIC_STATUS,
            source,
            selected_sid,
            switched=switched,
            switching_enabled=switching_enabled,
        )

    assert notice(selected_sid=0xD382) == Notice(
        "Announcement: Road Traffic flash", "running on this station", alarm=False
    )
    assert notice(switched=True).detail == "audio comes from NDR 2"
    assert notice().detail == "running on NDR 2"
    assert notice(switching_enabled=False).detail == "running on NDR 2 · switching is off"
    assert notice(source=None).detail == "running on Subchannel 5"


# Traffic news of another ensemble, carried there by NDR 2 (FIG 0/26).
OE_TRAFFIC = OeAnnouncement(5, 0x0002, new=True, scids=0, target_sid=0xD382)
OE_TRAFFIC_STATUS = AnnouncementStatus(
    None, alarm_enabled=False, oe_active=OE_TRAFFIC, oe_eid=0x1188
)


def test_the_notice_on_an_announcement_of_another_ensemble():
    def notice(source="NDR 2", switching_enabled=True):
        return announcement_notice(
            OE_TRAFFIC_STATUS,
            source,
            0xD220,
            switched=False,
            switching_enabled=switching_enabled,
        )

    assert notice() == Notice("Announcement: Road Traffic flash", "running on NDR 2", alarm=False)
    assert notice(source=None).detail == "running in another ensemble"
    assert notice(switching_enabled=False).detail == "running on NDR 2 · switching is off"


VISIT = AnnouncementVisit(OE_TRAFFIC, "11D", 0xD382, "NDR 2", "5C", 0xD220, "Dlf Kultur")


def test_the_notice_while_gone_to_another_ensemble_says_where_it_goes_back_to():
    assert visit_notice(VISIT) == Notice(
        "Announcement: Road Traffic flash",
        "then back to Dlf Kultur · tap to return now",
        alarm=False,
    )


def test_without_a_running_announcement_there_is_no_notice():
    idle = AnnouncementStatus(AnnouncementSupport(0x0002, (2,)), alarm_enabled=False)
    for status in (None, idle):
        assert (
            announcement_notice(status, None, 0xD381, switched=False, switching_enabled=True)
            is None
        )


def test_an_alarm_announcement_is_named_as_such():
    alarm = AnnouncementStatus(
        None, alarm_enabled=True, active=Announcement(ALARM_CLUSTER, 0x0001, False, 9)
    )
    notice = announcement_notice(alarm, None, 0xD381, switched=False, switching_enabled=True)
    assert (notice.title, notice.alarm) == ("Alarm announcement", True)


def test_the_radio_page_shows_a_running_announcement_until_it_ends(data_directory, touch_window):
    window = touch_window(data_directory)
    window.show()
    controller = window._controller
    controller.select_service(SERVICES_5C[0][0])
    radio = window._radio_page
    assert not radio._notice.isVisibleTo(radio)

    controller.announcement_changed.emit(TRAFFIC_STATUS, "NDR 2")
    assert radio._notice.isVisibleTo(radio)
    assert radio._notice_title.text() == "Announcement: Road Traffic flash"
    assert radio._notice_detail.text() == "running on NDR 2 · switching is off"

    # The signal processing moves the audio to the announcement's subchannel.
    subchannel = Subchannel(5, 0, 72, is_uep=False, protection_index=2, eep_option=0)
    controller.dsp_worker.subchannel_info_ready.emit(
        SubchannelStatus(AudioComponent(5, 63), subchannel, announcement=TRAFFIC)
    )
    assert radio._notice_detail.text() == "audio comes from NDR 2"

    controller.announcement_changed.emit(None, None)
    assert not radio._notice.isVisibleTo(radio)


def test_a_tap_on_the_notice_goes_back_from_another_ensemble(data_directory, touch_window):
    window = touch_window(data_directory)
    window.show()
    controller = window._controller
    radio = window._radio_page
    returned = []
    controller.return_from_announcement = lambda: returned.append(True)

    controller.announcement_visit_changed.emit(VISIT)
    assert radio._notice.isVisibleTo(radio)
    assert radio._notice_detail.text().startswith("then back to Dlf Kultur")

    radio._notice.tapped.emit()
    assert returned == [True]

    controller.announcement_visit_changed.emit(None)
    assert not radio._notice.isVisibleTo(radio)


def test_an_announcement_goes_with_its_station(data_directory, touch_window):
    window = touch_window(data_directory)
    controller = window._controller
    controller.select_service(SERVICES_5C[0][0])
    radio = window._radio_page
    controller.announcement_changed.emit(TRAFFIC_STATUS, "NDR 2")

    controller.select_service(SERVICES_5C[1][0])

    assert not radio._notice.isVisibleTo(radio)


def test_dl_plus_is_shown_even_where_the_label_says_it_already(data_directory, touch_window):
    window = touch_window(data_directory)
    radio = window._radio_page
    dsp = window._controller.dsp_worker
    tags = {DlPlusContentType.ITEM_ARTIST: "Mantar", DlPlusContentType.ITEM_TITLE: "Hang 'Em Low"}

    dsp.dynamic_label_ready.emit(DynamicLabel("Mantar - Hang 'Em Low", DlPlus(False, True, tags)))
    assert radio._dl_plus_label.text() == "Mantar – Hang 'Em Low"
    assert not radio._dl_plus_label.isHidden()

    dsp.dynamic_label_ready.emit(DynamicLabel("Jetzt auf BOB!", DlPlus(False, True, tags)))
    assert radio._dl_plus_label.text() == "Mantar – Hang 'Em Low"

    dsp.dynamic_label_ready.emit(DynamicLabel("Jetzt auf BOB!", DlPlus(False, False, tags)))
    assert radio._dl_plus_label.isHidden()


def test_dl_plus_stands_right_beneath_the_dynamic_label(data_directory, touch_window):
    window = touch_window(data_directory)
    window.show()
    radio = window._radio_page
    tags = {DlPlusContentType.ITEM_ARTIST: "Mantar", DlPlusContentType.ITEM_TITLE: "Hang 'Em Low"}
    assert not radio._dl_plus_label.isVisibleTo(radio)

    window._controller.dsp_worker.dynamic_label_ready.emit(
        DynamicLabel("Jetzt auf BOB!", DlPlus(False, True, tags))
    )
    _spin(0)

    dls, dl_plus = radio._dls_label.geometry(), radio._dl_plus_label.geometry()
    assert 0 <= dl_plus.top() - dls.bottom() <= 8


def test_texts_of_one_line():
    assert one_line("Dlf\nKommentar\x0bDie Debatte  um\x1fdie") == "Dlf Kommentar Die Debatte umdie"
    distance = 1000
    run_ms = distance * 1000 / MARQUEE_SPEED_PX_PER_S
    assert marquee_offset(0, distance) == 0
    assert marquee_offset(MARQUEE_REST_MS, distance) == 0
    assert marquee_offset(MARQUEE_REST_MS + 1000, distance) == MARQUEE_SPEED_PX_PER_S
    # Once the copy stands where the text began, it rests again.
    assert marquee_offset(MARQUEE_REST_MS + run_ms, distance) == 0
    assert marquee_offset(2 * MARQUEE_REST_MS + run_ms + 500, distance) == pytest.approx(
        MARQUEE_SPEED_PX_PER_S / 2
    )


def test_the_dynamic_label_runs_through_only_when_it_does_not_fit(data_directory, touch_window):
    window = touch_window(data_directory)
    window.show()
    label = window._radio_page._dls_label
    show = window._controller.dsp_worker.dynamic_label_ready.emit
    long_text = "Deutschlandfunk - Kommentar - Die Debatte um die Wehrpflicht geht weiter"

    show(DynamicLabel("Dlf", None))
    assert not label.running
    show(DynamicLabel(long_text, None))
    assert label.overflows() and label.running
    assert label.sizeHint().height() == label.height()

    # Stations repeat their label: it runs on rather than starting from rest again.
    started = label._since_set.msecsSinceReference()
    _spin(20)
    show(DynamicLabel(long_text, None))
    assert label._since_set.msecsSinceReference() == started
    show(DynamicLabel(long_text + "!", None))
    assert label._since_set.msecsSinceReference() > started

    # Not on screen, nothing to animate.
    window.show_page(SETTINGS)
    assert not label.running
    window.show_page(RADIO)
    assert label.running


def test_a_spinner_turns_over_the_tiles_while_the_scan_runs(
    data_directory, touch_window, fake_scanners
):
    window = touch_window(data_directory)
    window.show()
    window.show_page(STATIONS)
    page = window._stations_page

    page._scan_button.click()
    assert page._spinner.running and page._spinner.isVisible()
    fake_scanners[0].channel_started.emit("5B", 1, 38)
    assert (page._spinner._title, page._spinner._detail) == ("2/38", "Channel 5B")

    fake_scanners[0].scan_finished.emit(True)
    assert not page._spinner.isVisible()


def test_the_first_scan_keeps_the_filters_on_top_and_the_scan_button_at_the_bottom(
    tmp_path, touch_window, fake_scanners
):
    """No stations known yet, and the spinner stands in for the note that says so: the
    empty grid keeps the space between them, else the layout spread both rows apart
    evenly and the filters came down onto the spinner."""
    window = touch_window(tmp_path)
    window.show()
    window.show_page(STATIONS)
    page = window._stations_page
    _spin(20)
    chips_top = page._chips_area.geometry().top()
    bottom_row_bottom = page._bottom_row.geometry().bottom()

    page._scan_button.click()
    _spin(20)
    assert page._spinner.isVisible() and page._empty_label.isHidden()
    assert page._chips_area.geometry().top() == chips_top
    assert page._bottom_row.geometry().bottom() == bottom_row_bottom
    spinner = page._spinner.geometry()
    assert spinner.top() > page._chips_area.geometry().bottom()
    assert spinner.bottom() < page._bottom_row.geometry().top()


def test_a_second_tap_on_the_menu_shown_turns_back_to_its_first_page(data_directory, touch_window):
    window = touch_window(data_directory)
    window.show()
    window._nav_buttons[SIGNAL].click()
    signal = window._signal_page
    signal.show_page(OFDM)
    window._nav_buttons[SIGNAL].click()
    assert window.current_page() == SIGNAL and signal.current_page() == OVERVIEW

    services = window._services_page
    window._nav_buttons[SERVICES].click()
    services.menu_button(TRAFFIC_MENU).click()
    traffic, gnss = services.traffic_view, services.gnss_view
    traffic.show_page(MESSAGES)
    gnss.show_page(GLOBE)
    # Another chip shows its view as it was left.
    services.menu_button(GNSS_MENU).click()
    assert services.current_view() == GNSS_MENU and gnss.current_page() == GLOBE
    # The chip of the view shown: its first page.
    services.menu_button(GNSS_MENU).click()
    assert services.current_view() == GNSS_MENU and gnss.current_page() == SKY
    services.menu_button(TRAFFIC_MENU).click()
    assert traffic.current_page() == MESSAGES
    # The button in the rail changes nothing: the services page has chips for its
    # views, as the settings have for their sections.
    window._nav_buttons[SERVICES].click()
    assert services.current_view() == TRAFFIC_MENU and traffic.current_page() == MESSAGES
    services.menu_button(TRAFFIC_MENU).click()
    assert traffic.current_page() == MAP
    # The guide has pages too: its chip takes the carousel back to the schedule.
    guide = services._guide_view
    services.menu_button(GUIDE_MENU).click()
    guide.show_page(CAROUSEL)
    services.menu_button(GUIDE_MENU).click()
    assert services.current_view() == GUIDE_MENU and guide.current_page() == SCHEDULE
    window._nav_buttons[SETTINGS].click()
    settings = window._settings_page
    settings.show_subpage(WIFI)
    window._nav_buttons[SETTINGS].click()
    assert settings.current_subpage() == WIFI


# --- The output -----------------------------------------------------------------------


def _fake_outputs(monkeypatch):
    """Outputs stand in for those of the system, for the controller and the player
    alike: the loudspeaker and headphones, the headphones the default (as PipeWire
    makes them once they are connected)."""
    from audio import outputs, player
    from tests.test_audio_outputs import HEADPHONES, SPEAKER, _FakeMediaDevices
    from ui import receiver_controller

    _FakeMediaDevices.instances.clear()
    _FakeMediaDevices.change([SPEAKER, HEADPHONES], HEADPHONES)
    for module in (outputs, player, receiver_controller):
        monkeypatch.setattr(module, "QMediaDevices", _FakeMediaDevices)
    return outputs.audio_outputs()


def test_the_audio_settings_offer_the_outputs_automatic_first(
    data_directory, touch_window, monkeypatch
):
    outputs = _fake_outputs(monkeypatch)
    window = touch_window(data_directory)
    window.show()
    window.show_page(SETTINGS)
    page = window._settings_page
    page.show_subpage(AUDIO)
    row = page._output_row
    controller = window._controller
    _spin(50)  # the player tells where the audio goes: the default, the headphones

    assert row.isVisible()
    names = ["Automatic", "Built-in Audio Stereo", "WH-1000XM4"]
    assert [name for _key, name in row.choices()] == names
    assert row.keys() == ["", outputs[0].id, outputs[1].id]
    assert row.current() == "" and row.text() == "Automatic"
    assert row.subtitle.isVisible() and row.subtitle.text() == "currently WH-1000XM4"

    # The sheet: the one in effect marked, the Bluetooth device told as such.
    row.open()
    sheet = row.sheet()
    assert sheet.isVisible()
    assert [r.name.text() for r in sheet.rows] == names
    assert sheet.rows[0].property("selected") and not sheet.rows[2].property("selected")
    assert sheet.rows[2].detail.text() == "Bluetooth" and sheet.rows[1].detail.isHidden()
    sheet.rows[2].click()
    assert not sheet.isVisible()
    assert controller.audio_output_choice == outputs[1].id
    assert row.text() == "WH-1000XM4"
    assert not row.subtitle.isVisible()  # the one chosen is the one in use

    row.choose(outputs[0].id)
    assert controller.audio_output_choice == outputs[0].id
    assert row.subtitle.text() == "currently WH-1000XM4"  # until the player has moved
    _spin(50)
    assert not row.subtitle.isVisible()


def test_the_loudspeaker_button_shows_headphones_and_offers_the_outputs_when_held(
    data_directory, touch_window, monkeypatch
):
    from PyQt6.QtTest import QTest

    from ui.touch.widgets import LONG_PRESS_MS

    outputs = _fake_outputs(monkeypatch)
    window = touch_window(data_directory)
    window.show()
    page = window._radio_page
    controller = window._controller
    button = page._mute_button
    _spin(50)  # the player tells where the audio goes: the headphones
    assert page.mute_symbol(False) == "headphones"
    assert page.mute_symbol(True) == "headphones_muted"

    # Held: the sheet, and no toggling of the mute when the finger lifts.
    QTest.mousePress(button, Qt.MouseButton.LeftButton)
    _spin(LONG_PRESS_MS + 150)
    QTest.mouseRelease(button, Qt.MouseButton.LeftButton)
    sheet = page.output_sheet()
    assert sheet.isVisible()
    assert not button.isChecked() and controller.audio_enabled
    assert sheet.title.text() == "Output"
    assert sheet.detail.text() == "Automatic, currently WH-1000XM4"
    assert [b.text() for b in sheet.actions] == ["Automatic", *(o.name for o in outputs)]
    assert sheet.actions[0].property("role") == "accent"

    sheet.actions[1].click()
    assert not sheet.isVisible()
    assert controller.audio_output_choice == outputs[0].id
    _spin(50)  # the player has moved to the loudspeaker
    assert page.mute_symbol(False) == "volume"

    # A tap still mutes.
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert button.isChecked() and not controller.audio_enabled
    assert page.mute_symbol(True) == "muted"

    # Muted or unmuted from elsewhere (MQTT): the button follows, and one tap undoes
    # it (it used to take two, the button keeping a state of its own, 10.10.2026).
    controller.set_audio_enabled(True)
    assert not button.isChecked()
    controller.set_audio_enabled(False)
    assert button.isChecked()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert not button.isChecked() and controller.audio_enabled


def test_the_volume_slider_of_the_radio_page_follows_another_output(
    data_directory, touch_window, monkeypatch
):
    outputs = _fake_outputs(monkeypatch)
    window = touch_window(data_directory)
    controller = window._controller
    slider = window._radio_page._volume_slider
    report = controller._audio_player.output_changed

    report.emit(outputs[0].id.encode(), outputs[0].name)
    slider.setValue(85)
    report.emit(outputs[1].id.encode(), outputs[1].name)
    slider.setValue(35)
    report.emit(outputs[0].id.encode(), outputs[0].name)

    assert slider.value() == 85 and controller.volume_percent == 85


def test_the_settings_have_a_bluetooth_category(data_directory, touch_window):
    window = touch_window(data_directory)
    window.show()
    window.show_page(SETTINGS)
    page = window._settings_page

    page._categories.rows["bluetooth"].click()

    assert page.current_subpage() == "bluetooth"
    assert page._bluetooth_view.isVisible()
    # No system bus on the PC: the page says where pairing happens instead.
    assert page._bluetooth_view._unavailable.isVisible()
    assert page._bluetooth_view.message.isVisible()
    assert not page._wifi_view.message.isVisible()


# --- WLAN and Bluetooth in the rail ---------------------------------------------------


def test_the_rail_shows_wlan_and_bluetooth_while_they_are_on(
    data_directory, touch_window, monkeypatch
):
    from net.bluetooth import ADAPTER, DEVICE, BluetoothManager
    from net.wifi import WifiManager
    from tests.fake_bluez import HCI0, SONY, FakeBus
    from tests.fake_nmcli import DEVICE_STATUS, LIST, RADIO, WIFI_LIST, FakeNmcli
    from ui.touch import settings_page
    from ui.touch.icons import badged_pixmap
    from ui.touch.window import STATUS_ICON

    nmcli = FakeNmcli()
    bus = FakeBus()
    monkeypatch.setattr(
        settings_page, "WifiManager", lambda parent=None: WifiManager(parent, runner=nmcli)
    )
    monkeypatch.setattr(
        settings_page, "BluetoothManager", lambda parent=None: BluetoothManager(parent, bus=bus)
    )
    window = touch_window(data_directory)
    window.set_load_shown(True)  # the symbols sit between it and the clock
    window.show()
    wifi_icon, bluetooth_icon = window._wifi_icon, window._bluetooth_icon

    def drawn(label: QLabel, symbol: str, color: str, badge: str = "") -> bool:
        color = THEMES[window._theme.key].color(color)
        expected = badged_pixmap(symbol, color, STATUS_ICON, badge, label.font())
        return label.pixmap().toImage() == expected.toImage()

    # Read at the start, both connected: in the colour of the text, above the clock; the
    # Wi-Fi on 2.4 GHz without a figure.
    assert wifi_icon.isVisible() and bluetooth_icon.isVisible()
    assert drawn(wifi_icon, "wifi", "text") and drawn(bluetooth_icon, "bluetooth", "text")
    assert wifi_icon.width() == STATUS_ICON == bluetooth_icon.width()
    assert window._status_row.geometry().bottom() < window._clock_label.geometry().top()
    assert window._status_row.geometry().top() > window._load_panel.geometry().bottom()

    # Over to the access point on 5 GHz: the figure, the symbol wider for it.
    on_5_ghz = WIFI_LIST.replace(" :82:5180", "*:82:5180").replace("*:64:2437", " :64:2437")
    nmcli.answers[(*LIST, "no")] = on_5_ghz
    window._settings_page.wifi._poll()
    assert drawn(wifi_icon, "wifi", "text", "5")
    assert wifi_icon.width() > STATUS_ICON

    # The headphones go: Bluetooth stays on, muted; off, the rune goes.
    bus.properties_changed.emit(SONY, DEVICE, {"Connected": False})
    assert bluetooth_icon.isVisible() and drawn(bluetooth_icon, "bluetooth", "muted")
    bus.properties_changed.emit(HCI0, ADAPTER, {"Powered": False})
    assert not bluetooth_icon.isVisible()

    # The WLAN loses its network, then is switched off elsewhere: the readings in the
    # background notice. Without a connection no band, though the list still marks one.
    nmcli.answers[DEVICE_STATUS] = "wlan0:wifi:disconnected:\n"
    window._settings_page.wifi._poll()
    assert wifi_icon.isVisible() and drawn(wifi_icon, "wifi", "muted")
    assert wifi_icon.width() == STATUS_ICON
    nmcli.answers[RADIO] = "disabled\n"
    window._settings_page.wifi._poll()
    assert not wifi_icon.isVisible()

    # Back on, in the colours of the light scheme.
    nmcli.answers[RADIO] = "enabled\n"
    window._settings_page.wifi._poll()
    window.set_theme(LIGHT)
    assert drawn(wifi_icon, "wifi", "muted")


def test_the_rail_keeps_its_place_for_the_symbols_while_both_are_off(data_directory, touch_window):
    window = touch_window(data_directory)
    window.show()
    _spin(0)

    # Neither nmcli nor a system bus on the PC: no symbol, but the row stays.
    assert not window._wifi_icon.isVisible() and not window._bluetooth_icon.isVisible()
    assert window._status_row.isVisible()
    assert window._status_row.height() == window._wifi_icon.height()


# --- The pointer ----------------------------------------------------------------------


def test_the_hidden_cursor_is_set_anew_when_a_bluetooth_device_connects_or_goes(
    data_directory, touch_window, monkeypatch
):
    from net.bluetooth import DEVICE, BluetoothManager
    from tests.fake_bluez import SONY, FakeBus
    from ui.touch import settings_page
    from ui.touch import window as window_module

    bus = FakeBus()
    monkeypatch.setattr(
        settings_page, "BluetoothManager", lambda parent=None: BluetoothManager(parent, bus=bus)
    )
    monkeypatch.setattr(window_module, "REHIDE_CURSOR_DELAYS_MS", (0, 0))
    hidden = []
    monkeypatch.setattr(window_module, "rehide_cursor", lambda: hidden.append(True))
    window = touch_window(data_directory)
    assert window._bluetooth_connected == {SONY}  # read at the start, not only on the page

    bus.properties_changed.emit(SONY, DEVICE, {"Connected": False})
    bus.properties_changed.emit(SONY, DEVICE, {"RSSI": -50})  # no change of connections
    _spin(20)
    assert hidden == [True, True]

    bus.properties_changed.emit(SONY, DEVICE, {"Connected": True})
    _spin(20)
    assert hidden == [True] * 4


def test_rehiding_the_cursor_keeps_the_override_and_does_nothing_without_one():
    from PyQt6.QtGui import QCursor
    from PyQt6.QtWidgets import QApplication

    from ui.touch.window import rehide_cursor

    app = QApplication.instance() or QApplication([])
    assert app.overrideCursor() is None
    rehide_cursor()
    assert app.overrideCursor() is None

    app.setOverrideCursor(QCursor(Qt.CursorShape.BlankCursor))
    try:
        rehide_cursor()
        assert app.overrideCursor().shape() == Qt.CursorShape.BlankCursor
    finally:
        app.restoreOverrideCursor()
    assert app.overrideCursor() is None
