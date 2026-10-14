"""The language of the surfaces (`ui.language`): English unless German was chosen, in the
desktop's menu "View › Language" or on the touch settings page, one setting for both;
each offers the new start the change takes, and builds itself in German."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QLocale, QSettings, QTranslator
from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton

import i18n
import main
from net import wifi
from ui.language import (
    SETTINGS_LANGUAGE,
    configure_language,
    restart_command,
    restart_question,
    store_language,
    stored_language,
)
from ui.main_window import MainWindow
from ui.receiver_controller import SETTINGS_FILENAME
from ui.touch import backlight
from ui.touch.settings_page import DISPLAY
from ui.touch.window import SETTINGS, TouchWindow


@pytest.fixture(autouse=True)
def english():
    """Back to English after each test, as the other tests expect."""
    yield
    i18n.set_language(i18n.SOURCE_LANGUAGE)


def _settings(data_directory: Path) -> QSettings:
    return QSettings(str(data_directory / SETTINGS_FILENAME), QSettings.Format.IniFormat)


@pytest.fixture
def touch_window(make_controller, monkeypatch):
    """A touch window on a fresh controller, as `main` builds it."""
    monkeypatch.setattr(wifi, "find_nmcli", lambda: None)
    monkeypatch.setattr(backlight, "BACKLIGHT_CLASS", Path("nowhere") / "backlight")
    windows = []

    def make(data_directory) -> TouchWindow:
        controller = make_controller(data_directory, start=False)
        window = TouchWindow(controller)
        controller.start()
        window.show()
        windows.append(window)
        return window

    yield make
    for window in windows:
        window.deleteLater()


@pytest.fixture
def desktop_window(make_controller):
    """A desktop window on a fresh controller, as `main` builds it."""
    windows = []

    def make(data_directory) -> MainWindow:
        controller = make_controller(data_directory, start=False)
        window = MainWindow(controller)
        controller.start()
        window.show()
        windows.append(window)
        return window

    yield make
    for window in windows:
        window.deleteLater()


def _button(parent, text: str) -> QPushButton:
    return next(b for b in parent.findChildren(QPushButton) if b.text() == text)


def _language_actions(window: MainWindow) -> dict[str, object]:
    """The entries of "View › Language" by their text."""
    view = next(a.menu() for a in window.menuBar().actions() if a.menu() is not None)
    languages = next(a.menu() for a in view.actions() if a.menu() is not None)
    return {action.text(): action for action in languages.actions()}


# --- The setting -----------------------------------------------------------------------


def test_english_is_the_language_until_another_is_chosen(tmp_path):
    settings = _settings(tmp_path)
    assert stored_language(settings) == "en"
    settings.setValue(SETTINGS_LANGUAGE, "de")
    assert stored_language(settings) == "de"
    # A catalog gone, or a value written by hand: English rather than nothing.
    settings.setValue(SETTINGS_LANGUAGE, "xx")
    assert stored_language(settings) == "en"


def test_the_language_stored_is_set_before_the_surface_is_built(tmp_path):
    app = QApplication.instance() or QApplication([])
    settings = _settings(tmp_path)
    settings.setValue(SETTINGS_LANGUAGE, "de")
    system_locale = QLocale()

    try:
        assert configure_language(app, settings) == "de"
        assert i18n.language() == "de"
        assert i18n.tr("Settings") == "Einstellungen"
        assert stored_language(_settings(tmp_path)) == "de"
        # Qt's widgets write numbers as the language does, not as the system's locale.
        assert QLocale().decimalPoint() == ","
        store_language(settings, "en")
        assert configure_language(app, settings) == "en"
        assert QLocale().decimalPoint() == "."
    finally:
        # Qt's own German and the locale stay out of the other tests.
        QLocale.setDefault(system_locale)
        for translator in app.findChildren(QTranslator):
            app.removeTranslator(translator)
            translator.deleteLater()


def test_the_question_is_asked_in_the_language_chosen_and_not_for_the_one_running():
    question = restart_question("de")
    assert question is not None
    assert question.title == "Das Radio auf Deutsch neu starten?"
    assert (question.restart, question.later) == ("Jetzt neu starten", "Später")
    assert restart_question("en") is None


# --- Touch -------------------------------------------------------------------------------


def test_choosing_a_language_on_the_touch_surface_offers_the_restart(data_directory, touch_window):
    window = touch_window(data_directory)
    page = window._settings_page
    page.show_subpage(DISPLAY)
    restarts = []
    window.restart_requested.connect(lambda: restarts.append(True))

    _button(page._language, "Deutsch").click()

    assert stored_language(window._controller.settings) == "de"
    sheet = page.action_sheet()
    assert sheet.isVisible()
    # Asked in German while the surface is still English, so that the one who chose it
    # can read it.
    assert sheet.title.text() == "Das Radio auf Deutsch neu starten?"
    assert _button(sheet, "Später")
    _button(sheet, "Jetzt neu starten").click()
    assert restarts == [True]


def test_back_to_the_language_running_there_is_nothing_to_restart(data_directory, touch_window):
    window = touch_window(data_directory)
    page = window._settings_page
    _button(page._language, "Deutsch").click()
    page.action_sheet().dismiss()

    _button(page._language, "English").click()

    assert stored_language(window._controller.settings) == "en"
    assert not page.action_sheet().isVisible()


def test_any_other_sheet_keeps_its_way_back_in_the_language_running(data_directory, touch_window):
    window = touch_window(data_directory)
    page = window._settings_page
    _button(page._language, "Deutsch").click()
    page.action_sheet().dismiss()

    page._equalizer_reset.click()

    assert _button(page.action_sheet(), "Cancel")


def test_the_touch_surface_is_built_in_german_when_german_is_set(data_directory, touch_window):
    i18n.set_language("de")
    window = touch_window(data_directory)

    assert window._nav_buttons[SETTINGS].text() == "Einstellungen"
    page = window._settings_page
    assert _button(page, "Radio beenden")
    assert _button(page._subpage_chips, "Anzeige")
    assert _button(page._language, "English") and _button(page._language, "Deutsch")


# --- Desktop -----------------------------------------------------------------------------


def test_the_desktop_lists_each_language_by_its_own_name_and_marks_the_one_chosen(
    data_directory, desktop_window
):
    _settings(data_directory).setValue(SETTINGS_LANGUAGE, "de")
    window = desktop_window(data_directory)

    actions = _language_actions(window)

    assert list(actions) == ["English", "Deutsch"]
    assert actions["Deutsch"].isChecked() and not actions["English"].isChecked()


def test_choosing_a_language_on_the_desktop_offers_the_restart(data_directory, desktop_window):
    window = desktop_window(data_directory)
    restarts = []
    window.restart_requested.connect(lambda: restarts.append(True))

    _language_actions(window)["Deutsch"].trigger()

    assert stored_language(window._controller.settings) == "de"
    box = window._restart_box
    assert isinstance(box, QMessageBox) and box.isVisible()
    assert box.text() == "Das Radio auf Deutsch neu starten?"
    assert {button.text() for button in box.buttons()} == {"Jetzt neu starten", "Später"}
    _button(box, "Jetzt neu starten").click()
    assert restarts == [True]


def test_on_the_desktop_the_language_running_needs_no_restart(data_directory, desktop_window):
    window = desktop_window(data_directory)

    _language_actions(window)["English"].trigger()

    assert stored_language(window._controller.settings) == "en"
    assert window._restart_box is None


def test_the_desktop_is_built_in_german_when_german_is_set(data_directory, desktop_window):
    i18n.set_language("de")
    window = desktop_window(data_directory)

    menus = [action.text() for action in window.menuBar().actions()]
    assert menus == ["&Ansicht"]
    tabs = [window._tabs.tabText(index) for index in range(window._tabs.count())]
    assert "Datendienste" in tabs and "Programmführer" in tabs and "Verkehr" in tabs


# --- The new start -----------------------------------------------------------------------


def test_the_radio_starts_anew_as_it_was_started(monkeypatch):
    command = restart_command()
    assert command[0] == sys.executable
    assert command[1:] == sys.orig_argv[1:]

    calls = []
    monkeypatch.setattr(main.os, "execv", lambda path, args: calls.append(("execv", path, args)))
    monkeypatch.setattr(main.subprocess, "Popen", lambda args: calls.append(("Popen", args)))
    main.restart(["python", "src/main.py", "--touch"])
    if sys.platform == "win32":
        assert calls == [("Popen", ["python", "src/main.py", "--touch"])]
    else:
        # In place of this process: cage keeps its client, systemd its unit.
        assert calls == [("execv", "python", ["python", "src/main.py", "--touch"])]


def test_qt_takes_its_pulseaudio_backend_on_linux_unless_one_is_named():
    environ: dict[str, str] = {}
    main.choose_audio_backend(environ, "linux")
    assert environ == {"QT_AUDIO_BACKEND": "pulseaudio"}
    environ = {"QT_AUDIO_BACKEND": "pipewire"}
    main.choose_audio_backend(environ, "linux")
    assert environ == {"QT_AUDIO_BACKEND": "pipewire"}
    environ = {}
    main.choose_audio_backend(environ, "win32")
    assert environ == {}


def test_the_stream_is_pinned_to_its_sink_where_the_radio_sets_the_volume():
    """Otherwise PipeWire moves it, at the gain of headphones, to the loudspeaker
    pinned at 100 % when their sink goes."""
    pinned = main.PINNED_STREAM_PROPERTIES
    environ: dict[str, str] = {}
    main.pin_audio_stream(environ, "linux", system_volume=True)
    assert environ == {"PULSE_PROP": pinned}
    main.pin_audio_stream(environ, "linux", system_volume=True)  # a restart
    assert environ == {"PULSE_PROP": pinned}
    environ = {"PULSE_PROP": "media.role=music"}
    main.pin_audio_stream(environ, "linux", system_volume=True)
    assert environ == {"PULSE_PROP": f"media.role=music {pinned}"}
    for platform, system_volume in (("linux", False), ("win32", True)):
        environ = {}
        main.pin_audio_stream(environ, platform, system_volume)
        assert environ == {}


def test_info_messages_of_the_radio_are_kept_and_those_of_the_receivers_are_not():
    import logging

    names = (*main.INFO_LOGGERS, "sdr", "dsp")
    levels = {name: logging.getLogger(name).level for name in names}
    try:
        main.configure_logging()
        assert logging.getLogger("audio.system_volume").isEnabledFor(logging.INFO)
        assert logging.getLogger("net.bluetooth").isEnabledFor(logging.INFO)
        assert logging.getLogger("ui.receiver_controller").isEnabledFor(logging.INFO)
        assert not logging.getLogger("sdr.rspduo_reader").isEnabledFor(logging.INFO)
        assert logging.getLogger("sdr.rspduo_reader").isEnabledFor(logging.WARNING)
    finally:
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)
