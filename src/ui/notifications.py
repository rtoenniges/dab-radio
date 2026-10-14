"""Notifications from outside the radio, a smart home over MQTT above all
(`net.radio_state`, command `notify`): a title, a message, up to `MAX_BUTTONS` buttons
and perhaps an image, shown by the surface until a button is pressed, their time runs
out or the sender takes them back. Every notification ends with exactly one answer for
its sender (`NotificationAnswer`), so that an automation waiting for one never waits for
ever.

`NotificationCenter` holds the open ones without a widget, as the controller holds
whatever a surface needs: one at a time is shown (`shown`), the others wait, alarms
before warnings before the rest, and within a level in the order they came. A
notification with the id of an open one takes its place, and the old one is answered
`replaced`; an alarm coming while another one is shown is shown at once, and the other
waits again.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QImage


class Level(Enum):
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"


# The order of the queue: alarms first.
_PRIORITY = {Level.ALARM: 0, Level.WARNING: 1, Level.INFO: 2}

# What the sender may ask; `net.radio_state` checks the arguments against them.
MAX_BUTTONS = 4
MAX_ID_LENGTH = 64
MAX_TITLE_LENGTH = 120
MAX_MESSAGE_LENGTH = 1000
MAX_LABEL_LENGTH = 40
MAX_TIMEOUT_S = 24 * 3600
# Notifications open at once, the one shown included.
MAX_OPEN = 20
# The image as sent, and its longest side once read: larger ones are scaled down, the
# screen is smaller anyway.
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_SIDE = 1280
BUTTON_STYLES = ("", "accent", "danger")

# How a notification ended (`NotificationAnswer.reason`).
PRESSED = "pressed"  # a button was pressed
TIMEOUT = "timeout"  # its time ran out
DISMISSED = "dismissed"  # the sender took it back
REPLACED = "replaced"  # a notification with the same id took its place
CLOSED = "closed"  # the radio quit
REASONS = (PRESSED, TIMEOUT, DISMISSED, REPLACED, CLOSED)

# The one button of a notification that names none; its label is translated where it is
# shown and English in the answer.
DEFAULT_BUTTON_ID = "ok"
DEFAULT_BUTTON_LABEL = "OK"


@dataclass(frozen=True)
class NotificationButton:
    id: str  # what the answer names, for the automation to key on
    label: str  # what the button says, as sent
    style: str = ""  # one of `BUTTON_STYLES`: plain, "accent" or "danger"


@dataclass(frozen=True)
class Notification:
    id: str
    title: str
    message: str
    # None of them: the surface shows "OK" alone (`DEFAULT_BUTTON_ID`).
    buttons: tuple[NotificationButton, ...] = ()
    level: Level = Level.INFO
    timeout_s: float | None = None  # from its arrival; None: until answered
    image: QImage | None = None
    received: float = 0.0  # Unix time, set by `NotificationCenter.post`

    @property
    def expires(self) -> float | None:
        return None if self.timeout_s is None else self.received + self.timeout_s

    def button(self, button_id: str) -> NotificationButton | None:
        """The button with this id; the default one for a notification without."""
        if not self.buttons:
            if button_id == DEFAULT_BUTTON_ID:
                return NotificationButton(DEFAULT_BUTTON_ID, DEFAULT_BUTTON_LABEL)
            return None
        return next((button for button in self.buttons if button.id == button_id), None)


@dataclass(frozen=True)
class NotificationAnswer:
    """How a notification ended: the button pressed (`button`, `label`) or why not."""

    id: str
    reason: str  # one of `REASONS`
    button: str | None = None
    label: str | None = None

    def json(self) -> dict:
        return {"id": self.id, "reason": self.reason, "button": self.button, "label": self.label}


@dataclass
class _Entry:
    notification: Notification
    sequence: int  # the order of arrival; a replacement keeps the place of the old one
    timer: QTimer | None


class NotificationCenter(QObject):
    """See the module docstring. `post` adds, `press` answers with a button, `dismiss`
    takes one back, `close_all` ends them all as the radio quits."""

    # The open notifications or the one shown changed (`open`, `shown`).
    changed = pyqtSignal()
    # A notification came, new or in place of an open one; the surface wakes the screen.
    posted = pyqtSignal(object)  # Notification
    answered = pyqtSignal(object)  # NotificationAnswer

    def __init__(
        self, parent: QObject | None = None, clock: Callable[[], float] = time.time
    ) -> None:
        super().__init__(parent)
        self._clock = clock
        self._entries: list[_Entry] = []
        self._sequence = 0

    @property
    def open(self) -> tuple[Notification, ...]:
        """The open notifications in the order they are shown, the one shown first."""
        return tuple(entry.notification for entry in self._entries)

    @property
    def shown(self) -> Notification | None:
        return self._entries[0].notification if self._entries else None

    def get(self, notification_id: str) -> Notification | None:
        entry = self._entry(notification_id)
        return None if entry is None else entry.notification

    def remaining_s(self, notification: Notification) -> float | None:
        """Seconds left until the time of `notification` runs out; None without one."""
        expires = notification.expires
        return None if expires is None else max(expires - self._clock(), 0.0)

    def post(self, notification: Notification) -> Notification:
        """Opens `notification`, received now; in place of an open one with its id,
        which is answered `replaced`. Raises ValueError while `MAX_OPEN` are open."""
        old = self._entry(notification.id)
        if old is None and len(self._entries) >= MAX_OPEN:
            # The message goes to the sender as the answer of the command.
            raise ValueError(f"{MAX_OPEN} notifications are open already.")
        notification = replace(notification, received=self._clock())
        if old is None:
            self._sequence += 1
            sequence = self._sequence
        else:
            self._remove(old)
            sequence = old.sequence
            self.answered.emit(NotificationAnswer(notification.id, REPLACED))
        self._entries.append(_Entry(notification, sequence, self._timer(notification)))
        self._entries.sort(key=lambda e: (_PRIORITY[e.notification.level], e.sequence))
        self.posted.emit(notification)
        self.changed.emit()
        return notification

    def press(self, notification_id: str, button_id: str) -> None:
        """The user pressed a button of an open notification; it closes."""
        entry = self._entry(notification_id)
        if entry is None:
            return  # answered meanwhile
        button = entry.notification.button(button_id)
        if button is None:
            return
        self._end(entry, NotificationAnswer(notification_id, PRESSED, button.id, button.label))

    def dismiss(self, notification_id: str) -> bool:
        """Takes an open notification back, answered `dismissed`; whether it was open."""
        entry = self._entry(notification_id)
        if entry is None:
            return False
        self._end(entry, NotificationAnswer(notification_id, DISMISSED))
        return True

    def close_all(self) -> None:
        """Ends every open notification, answered `closed`; as the radio quits."""
        entries, self._entries = self._entries, []
        for entry in entries:
            self._stop(entry)
        if entries:
            self.changed.emit()
        for entry in entries:
            self.answered.emit(NotificationAnswer(entry.notification.id, CLOSED))

    def _entry(self, notification_id: str) -> _Entry | None:
        return next((e for e in self._entries if e.notification.id == notification_id), None)

    def _timer(self, notification: Notification) -> QTimer | None:
        if notification.timeout_s is None:
            return None
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda n=notification: self._on_timeout(n))
        timer.start(round(notification.timeout_s * 1000))
        return timer

    def _on_timeout(self, notification: Notification) -> None:
        entry = self._entry(notification.id)
        if entry is not None and entry.notification is notification:
            self._end(entry, NotificationAnswer(notification.id, TIMEOUT))

    def _end(self, entry: _Entry, answer: NotificationAnswer) -> None:
        self._remove(entry)
        self.changed.emit()
        self.answered.emit(answer)

    def _remove(self, entry: _Entry) -> None:
        self._entries.remove(entry)
        self._stop(entry)

    @staticmethod
    def _stop(entry: _Entry) -> None:
        if entry.timer is not None:
            entry.timer.stop()
            entry.timer.deleteLater()
