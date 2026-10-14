"""Dimming the screen and switching it off (`ui.touch.backlight`): the backlight through
the files of a backlight class in a temporary directory, the times, and the touch that
wakes the screen."""

from __future__ import annotations

import os
import stat
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QEventLoop, QSettings, Qt, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QPushButton, QWidget

from ui.touch.backlight import (
    DIM_FRACTION,
    SETTINGS_DIM_AFTER,
    WAKE_GRACE_MS,
    Backlight,
    Level,
    ScreenDimmer,
)

DEVICE = "10-0045"
DIMMED = round(255 * DIM_FRACTION)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _spin(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


@pytest.fixture
def sysfs(tmp_path: Path) -> Path:
    """A backlight class with one backlight, as the Touch Display 2 registers it: lit,
    at 200 of 255."""
    device = tmp_path / "backlight" / DEVICE
    device.mkdir(parents=True)
    (device / "max_brightness").write_text("255\n")
    (device / "brightness").write_text("200\n")
    (device / "bl_power").write_text("0\n")
    return device.parent


def _read(sysfs: Path, name: str) -> int:
    return int((sysfs / DEVICE / name).read_text())


@pytest.fixture
def settings(tmp_path: Path) -> QSettings:
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def window():
    """A window with a button that counts its clicks."""
    window = QWidget()
    window.resize(400, 300)
    button = QPushButton("Tippen", window)
    button.setGeometry(100, 100, 200, 72)
    window.clicks = []
    button.clicked.connect(lambda: window.clicks.append(1))
    window.button = button
    window.show()
    yield window
    window.deleteLater()


@pytest.fixture
def dimmer(sysfs, settings, window):
    dimmer = ScreenDimmer(Backlight.find(sysfs), settings, window)
    yield dimmer
    dimmer.shutdown()


# --- Backlight ------------------------------------------------------------------------


def test_the_backlight_is_found_with_what_it_shows(sysfs):
    backlight = Backlight.find(sysfs)

    assert backlight is not None
    assert backlight.name == DEVICE
    assert backlight.max_brightness == 255
    assert backlight.full_brightness == 200
    assert backlight.dim_brightness == DIMMED
    assert backlight.writable


def test_without_a_backlight_there_is_none(tmp_path):
    assert Backlight.find(tmp_path / "missing") is None
    (tmp_path / "empty").mkdir()
    assert Backlight.find(tmp_path / "empty") is None


def test_a_reading_as_dark_as_the_dimmed_level_is_taken_for_a_crash_while_dimmed(sysfs):
    (sysfs / DEVICE / "brightness").write_text(str(DIMMED))

    assert Backlight.find(sysfs).full_brightness == 255


def test_brightness_and_power_are_written_to_the_files(sysfs):
    backlight = Backlight.find(sysfs)

    backlight.set_brightness(DIMMED)
    assert _read(sysfs, "brightness") == DIMMED
    backlight.set_power(False)
    assert _read(sysfs, "bl_power") == 1
    backlight.set_power(True)
    assert _read(sysfs, "bl_power") == 0


def test_without_bl_power_switching_off_darkens_the_brightness(sysfs):
    (sysfs / DEVICE / "bl_power").unlink()
    backlight = Backlight.find(sysfs)

    backlight.set_power(False)
    assert _read(sysfs, "brightness") == 0
    backlight.set_brightness(200)
    backlight.set_power(True)
    assert _read(sysfs, "brightness") == 200


def test_a_backlight_only_root_may_write_is_reported_once(sysfs, caplog):
    brightness = sysfs / DEVICE / "brightness"
    brightness.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    try:
        backlight = Backlight.find(sysfs)
        assert not backlight.writable
        backlight.set_brightness(DIMMED)
        backlight.set_brightness(DIMMED + 1)
    finally:
        brightness.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

    assert _read(sysfs, "brightness") == 200
    assert sum("Cannot write" in record.message for record in caplog.records) == 1


# --- ScreenDimmer ---------------------------------------------------------------------


def test_the_screen_dims_and_goes_off_after_the_times_set(sysfs, dimmer):
    dimmer.set_dim_after(0.05)
    dimmer.set_off_after(0.15)
    assert dimmer.level is Level.AWAKE

    _spin(100)
    assert dimmer.level is Level.DIMMED
    assert _read(sysfs, "brightness") == DIMMED
    assert _read(sysfs, "bl_power") == 0

    _spin(100)
    assert dimmer.level is Level.OFF
    assert _read(sysfs, "bl_power") == 1

    dimmer.shutdown()
    assert dimmer.level is Level.AWAKE
    assert _read(sysfs, "brightness") == 200
    assert _read(sysfs, "bl_power") == 0


def test_the_screen_goes_straight_off_where_that_time_is_the_shorter(sysfs, dimmer):
    dimmer.set_dim_after(0.1)
    dimmer.set_off_after(0.05)

    _spin(75)
    assert dimmer.level is Level.OFF
    _spin(75)
    assert dimmer.level is Level.OFF  # not dimmed afterwards
    assert _read(sysfs, "bl_power") == 1


def test_a_touch_on_a_dimmed_screen_lights_it_and_reaches_what_it_hit(sysfs, dimmer, window):
    dimmer.set_dim_after(0.05)
    _spin(100)
    assert dimmer.level is Level.DIMMED

    QTest.mouseClick(window.button, Qt.MouseButton.LeftButton)
    assert dimmer.level is Level.AWAKE
    assert _read(sysfs, "brightness") == 200
    assert window.clicks == [1]
    # And the wait starts afresh.
    _spin(100)
    assert dimmer.level is Level.DIMMED


def test_the_touch_that_lights_a_screen_that_was_off_does_nothing_else(sysfs, dimmer, window):
    dimmer.set_off_after(0.05)
    _spin(100)
    assert dimmer.level is Level.OFF

    QTest.mouseClick(window.button, Qt.MouseButton.LeftButton)
    assert dimmer.level is Level.AWAKE
    assert _read(sysfs, "bl_power") == 0
    assert window.clicks == []

    dimmer.set_off_after(0)  # stays lit from now on
    _spin(WAKE_GRACE_MS + 50)
    QTest.mouseClick(window.button, Qt.MouseButton.LeftButton)
    assert window.clicks == [1]


def test_a_start_lights_a_screen_left_dark(sysfs, settings, window):
    (sysfs / DEVICE / "bl_power").write_text("1")

    dimmer = ScreenDimmer(Backlight.find(sysfs), settings, window)
    assert _read(sysfs, "bl_power") == 0
    assert _read(sysfs, "brightness") == 200
    dimmer.shutdown()


def test_the_times_are_kept_in_the_settings(settings, window):
    dimmer = ScreenDimmer(None, settings, window)
    assert dimmer.dim_after_s == 0 and dimmer.off_after_s == 0

    dimmer.set_dim_after(120)
    dimmer.set_off_after(600)
    settings.sync()
    assert settings.value(SETTINGS_DIM_AFTER, type=float) == 120

    again = ScreenDimmer(None, QSettings(settings.fileName(), QSettings.Format.IniFormat), window)
    assert again.dim_after_s == 120
    assert again.off_after_s == 600


def test_without_a_backlight_the_screen_stays_as_it_is(settings, window):
    dimmer = ScreenDimmer(None, settings, window)
    assert not dimmer.available

    dimmer.set_dim_after(0.01)
    dimmer.set_off_after(0.02)
    _spin(50)
    assert dimmer.level is Level.AWAKE
    QTest.mouseClick(window.button, Qt.MouseButton.LeftButton)
    assert window.clicks == [1]
