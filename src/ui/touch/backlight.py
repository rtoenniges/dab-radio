"""Dimming the screen when nothing has been touched for a while and switching it off
after longer, as a phone does, through the backlight of the display.

On the Raspberry Pi the Touch Display 2 registers its backlight with the kernel's
backlight class: `/sys/class/backlight/<name>/` with `brightness`, `max_brightness` and
`bl_power` (0 lights the backlight, 1 switches it off). A PC has none; the surface then
keeps the setting and does nothing else. The compositor is no help here: cage offers
its clients no idle management, and `wlr-randr --off` would take the output away from
the window. The touch panel keeps working with the backlight off, so the first touch
wakes the screen; when the screen was off, that touch does nothing else, since what it
hit could not be seen.

Only root may write the two files unless a udev rule hands them to the group video:
`pi/99-dab-radio-backlight.rules`, installed once per Pi."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from enum import Enum
from pathlib import Path

from PyQt6.QtCore import QEvent, QObject, QSettings, Qt, QTimer
from PyQt6.QtWidgets import QApplication

logger = logging.getLogger(__name__)

BACKLIGHT_CLASS = Path("/sys/class/backlight")
# The brightness of a dimmed screen, as a share of the brightest.
DIM_FRACTION = 0.1
# What the two steppers of the settings page offer, in seconds; 0 is never.
TIMEOUT_CHOICES_S = (0, 15, 30, 60, 120, 300, 600, 900, 1800, 3600)
SETTINGS_DIM_AFTER = "touch/dim_after_s"
SETTINGS_OFF_AFTER = "touch/off_after_s"
# Once the finger that woke a screen that was off has lifted, anything within this time
# is still swallowed: Qt may follow the touch with a mouse event synthesised from it.
WAKE_GRACE_MS = 250

_POWER_ON = "0"
_POWER_OFF = "1"

_INPUT_EVENTS = frozenset(
    {
        QEvent.Type.TouchBegin,
        QEvent.Type.TouchUpdate,
        QEvent.Type.TouchEnd,
        QEvent.Type.TouchCancel,
        QEvent.Type.MouseButtonPress,
        QEvent.Type.MouseButtonDblClick,
        QEvent.Type.MouseMove,
        QEvent.Type.MouseButtonRelease,
        QEvent.Type.Wheel,
        QEvent.Type.KeyPress,
        QEvent.Type.KeyRelease,
    }
)
# The end of the touch (or click, or key) that woke the screen.
_RELEASES = frozenset(
    {
        QEvent.Type.TouchEnd,
        QEvent.Type.TouchCancel,
        QEvent.Type.MouseButtonRelease,
        QEvent.Type.KeyRelease,
    }
)


class Backlight:
    """One backlight of the kernel's backlight class, read and written through its files.

    `full_brightness` is what the display showed when the radio started, restored on
    waking. A reading no brighter than the dimmed level is taken for the radio's own
    dimming, left behind by a crash, and the brightest is used instead."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._brightness = directory / "brightness"
        self._power = directory / "bl_power"
        self.max_brightness = int((directory / "max_brightness").read_text())
        self.dim_brightness = max(1, round(self.max_brightness * DIM_FRACTION))
        brightness = int(self._brightness.read_text())
        self.full_brightness = (
            brightness if brightness > self.dim_brightness else self.max_brightness
        )
        self._failed = False

    @classmethod
    def find(cls, root: Path | None = None) -> Backlight | None:
        """The first backlight registered (on the Pi the Touch Display 2 is the only
        one); None without one, or where its files cannot be read (a PC)."""
        root = BACKLIGHT_CLASS if root is None else root
        try:
            directories = sorted(p for p in root.iterdir() if (p / "brightness").is_file())
        except OSError:
            return None
        for directory in directories:
            try:
                backlight = cls(directory)
            except (OSError, ValueError) as error:
                logger.warning("Cannot read the backlight %s: %s", directory, error)
                continue
            logger.info(
                "Backlight %s: brightness %d of %d%s",
                backlight.name,
                backlight.full_brightness,
                backlight.max_brightness,
                "" if backlight.writable else ", not writable (udev rule from pi/ missing?)",
            )
            return backlight
        return None

    @property
    def name(self) -> str:
        return self._directory.name

    @property
    def writable(self) -> bool:
        """Whether the radio may set it (the udev rule from pi/ is installed)."""
        files = [self._brightness]
        if self._power.exists():
            files.append(self._power)
        return all(os.access(path, os.W_OK) for path in files)

    def set_brightness(self, value: int) -> None:
        self._write(self._brightness, str(value))

    def set_power(self, on: bool) -> None:
        """Lights the backlight or switches it off, the brightness kept for the next
        time; a backlight without `bl_power` is darkened to 0 instead (and lit again by
        the brightness written before `set_power(True)`)."""
        if self._power.exists():
            self._write(self._power, _POWER_ON if on else _POWER_OFF)
        elif not on:
            self._write(self._brightness, "0")

    def _write(self, path: Path, text: str) -> None:
        try:
            path.write_text(text)
        except OSError as error:
            if not self._failed:  # once, not at every timeout
                logger.warning("Cannot write %s to %s: %s", text, path, error)
            self._failed = True


class Level(Enum):
    AWAKE = "awake"
    DIMMED = "dimmed"
    OFF = "off"


class ScreenDimmer(QObject):
    """Dims the screen when nothing has been touched for `dim_after_s` and switches it
    off after `off_after_s` (each 0 for never), and lights it again at the first touch;
    keeps both times in the settings. It sees every touch as the event filter of the
    application, so the pages need not report them. Without a backlight it keeps the
    settings and does nothing else."""

    def __init__(
        self,
        backlight: Backlight | None,
        settings: QSettings,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._backlight = backlight
        self._settings = settings
        self._dim_after_s = self._stored(SETTINGS_DIM_AFTER)
        self._off_after_s = self._stored(SETTINGS_OFF_AFTER)
        self._level = Level.AWAKE
        self._swallowing = False
        self._dim_timer = self._timer(self._dim)
        self._off_timer = self._timer(self._switch_off)
        self._grace = QTimer(self)
        self._grace.setSingleShot(True)
        self._grace.setInterval(WAKE_GRACE_MS)
        self._app: QApplication | None = None
        if backlight is None:
            logger.debug("No backlight to dim: the screen stays on")
            return
        # A crash may have left the screen dimmed or off.
        self._light()
        self._restart_timers()
        self._app = QApplication.instance()
        if self._app is not None:
            self._app.installEventFilter(self)

    def _timer(self, on_timeout: Callable[[], None]) -> QTimer:
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(on_timeout)
        return timer

    def _stored(self, key: str) -> float:
        try:
            value = self._settings.value(key, 0.0, type=float)
        except TypeError, ValueError:  # edited by hand
            return 0.0
        return max(value, 0.0)

    @property
    def available(self) -> bool:
        """Whether there is a backlight to dim (on the Pi)."""
        return self._backlight is not None

    @property
    def backlight(self) -> Backlight | None:
        return self._backlight

    @property
    def level(self) -> Level:
        return self._level

    @property
    def dim_after_s(self) -> float:
        return self._dim_after_s

    @property
    def off_after_s(self) -> float:
        return self._off_after_s

    def set_dim_after(self, seconds: float) -> None:
        """Dims the screen this long after the last touch; 0 never. Kept."""
        self._dim_after_s = max(float(seconds), 0.0)
        self._settings.setValue(SETTINGS_DIM_AFTER, self._dim_after_s)
        self._restart_timers()

    def set_off_after(self, seconds: float) -> None:
        """Switches the screen off this long after the last touch; 0 never. Kept."""
        self._off_after_s = max(float(seconds), 0.0)
        self._settings.setValue(SETTINGS_OFF_AFTER, self._off_after_s)
        self._restart_timers()

    def touched(self) -> None:
        """Counts as a touch: lights the screen and starts the wait afresh."""
        if self._level is not Level.AWAKE:
            self._light()
        self._restart_timers()

    def shutdown(self) -> None:
        """Lights the screen for whatever comes next and stops watching."""
        self._dim_timer.stop()
        self._off_timer.stop()
        if self._app is not None:
            self._app.removeEventFilter(self)
            self._app = None
        if self._backlight is not None and self._level is not Level.AWAKE:
            self._light()

    # --- Timing -----------------------------------------------------------------------

    def _restart_timers(self) -> None:
        stages = ((self._dim_timer, self._dim_after_s), (self._off_timer, self._off_after_s))
        for timer, seconds in stages:
            timer.stop()
            if self._backlight is not None and seconds > 0 and self._level is Level.AWAKE:
                timer.start(round(seconds * 1000))

    def _dim(self) -> None:
        # Not a screen that went off first, where that time is the shorter.
        if self._level is Level.AWAKE and self._backlight is not None:
            self._backlight.set_brightness(self._backlight.dim_brightness)
            self._level = Level.DIMMED

    def _switch_off(self) -> None:
        if self._level is not Level.OFF and self._backlight is not None:
            self._backlight.set_power(False)
            self._level = Level.OFF

    def _light(self) -> None:
        assert self._backlight is not None
        self._backlight.set_brightness(self._backlight.full_brightness)
        self._backlight.set_power(True)
        self._level = Level.AWAKE

    # --- Touches ----------------------------------------------------------------------

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        kind = event.type()
        if kind not in _INPUT_EVENTS:
            return False
        if kind == QEvent.Type.MouseMove and event.buttons() == Qt.MouseButton.NoButton:
            return False  # the pointer alone: a mouse beside the display, synthesised moves
        if self._level is not Level.AWAKE:
            # On a screen that was off the touch could not see what it hit.
            self._swallowing = self._level is Level.OFF
            self._light()
        self._restart_timers()
        if self._swallowing:
            if kind in _RELEASES:
                self._swallowing = False
                self._grace.start()
            return True
        return self._grace.isActive()
