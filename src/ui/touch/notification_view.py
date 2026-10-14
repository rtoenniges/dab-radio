"""A notification from outside the radio (`ui.notifications`, sent over MQTT) over the
whole window: a bar in the colour of its level, the title and the message, an image
beside them if it brings one, its buttons at the bottom right and, at the bottom left,
how many more wait. With a timeout the bar shrinks from the right as the time runs out.

It stays until a button is pressed or the centre closes it (its time ran out, the
sender took it back); a tap beside it does nothing, unlike a sheet. Its buttons take a
tap only `GUARD_MS` after it appeared: it may come up under a finger on its way to
something else, and that finger is not to open the door.

What the sender wrote is shown as plain text: a label would take HTML from the network
for rich text, and a button an ampersand for a shortcut.
"""

from __future__ import annotations

from PyQt6.QtCore import QElapsedTimer, Qt, QTimer
from PyQt6.QtGui import QMouseEvent, QPixmap, QResizeEvent
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from i18n import tr, tr_plural
from ui.notifications import DEFAULT_BUTTON_ID, Notification, NotificationCenter
from ui.touch.overlay import Overlay
from ui.touch.widgets import repolish, set_role

GUARD_MS = 600
PANEL_WIDTH = 840
# With an image beside the texts: a camera's 4:3 picture fills its box 420 × 315.
PANEL_WIDTH_WITH_IMAGE = 1120
PANEL_MARGIN = 24
IMAGE_BOX = 420
LEVEL_BAR_HEIGHT = 6
# The bar of a timeout is redrawn once per pixel it loses, but no more often than this
# (about 30 a second: a short timeout loses several pixels a frame) nor more seldom.
COUNTDOWN_INTERVAL_MS = (33, 1000)
# The longest message (`ui.notifications.MAX_MESSAGE_LENGTH`) takes about 15 lines beside
# an image; more lines than that, from many line breaks, are cut.
MESSAGE_MAX_HEIGHT = 420


def button_text(label: str) -> str:
    """`label` as a button shows it: a single "&" would mark a shortcut."""
    return label.replace("&", "&&")


def countdown_interval_ms(timeout_s: float, width: int) -> int:
    """How often the bar of a timeout of `timeout_s` over `width` pixels is redrawn: once
    per pixel, within `COUNTDOWN_INTERVAL_MS`."""
    low, high = COUNTDOWN_INTERVAL_MS
    return min(max(round(timeout_s * 1000 / max(width, 1)), low), high)


class _LevelTrack(QWidget):
    """The place of the bar, the width of the panel; the bar in it (`bar`, coloured by
    the style sheet) as long as the share of the time left, from the left."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(LEVEL_BAR_HEIGHT)
        self.bar = set_role(QWidget(self), "level")
        self.bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self._share = 1.0

    @property
    def share(self) -> float:
        return self._share

    def set_share(self, share: float) -> None:
        self._share = min(max(share, 0.0), 1.0)
        self._place()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._place()

    def _place(self) -> None:
        width = round(self.width() * self._share)
        self.bar.setGeometry(0, 0, width, self.height())
        self.bar.setVisible(width > 0)


class NotificationDialog(Overlay):
    """Shows `center.shown` while there is one; see the module docstring."""

    def __init__(
        self, parent: QWidget, center: NotificationCenter, guard_ms: int = GUARD_MS
    ) -> None:
        super().__init__(parent)
        self._center = center
        self._guard_ms = guard_ms
        self._notification: Notification | None = None
        self._since = QElapsedTimer()
        self._countdown = QTimer(self)
        self._countdown.timeout.connect(self._show_time_left)

        self.level_track = _LevelTrack()
        self.level_bar = self.level_track.bar
        self.image = QLabel()
        self.image.setFixedWidth(IMAGE_BOX)
        self.image.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.title = set_role(QLabel(), "headline")
        self.title.setWordWrap(True)
        self.title.setTextFormat(Qt.TextFormat.PlainText)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setTextFormat(Qt.TextFormat.PlainText)
        self.message.setMaximumHeight(MESSAGE_MAX_HEIGHT)
        self.message.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.waiting = set_role(QLabel(), "muted")
        self.buttons: list[QPushButton] = []
        self._button_row = QHBoxLayout()
        self._button_row.setSpacing(12)
        self._button_row.addWidget(self.waiting)
        self._button_row.addStretch()

        texts = QVBoxLayout()
        texts.setSpacing(12)
        texts.addWidget(self.title)
        texts.addWidget(self.message)
        texts.addStretch()
        content = QHBoxLayout()
        content.setSpacing(24)
        content.addWidget(self.image)
        content.addLayout(texts, stretch=1)
        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(PANEL_MARGIN, 16, PANEL_MARGIN, 20)
        layout.setSpacing(16)
        layout.addWidget(self.level_track)
        layout.addLayout(content)
        layout.addLayout(self._button_row)

        center.changed.connect(self._follow)
        self._follow()

    @property
    def notification(self) -> Notification | None:
        """The notification on display."""
        return self._notification

    @property
    def counting_down(self) -> bool:
        """Whether the bar is shrinking with a timeout."""
        return self._countdown.isActive()

    def _follow(self) -> None:
        shown = self._center.shown
        if shown is None:
            self._notification = None
            self._countdown.stop()
            self.hide()
            return
        if shown is not self._notification:
            # New, or another in its place: the finger gets its pause again.
            self._notification = shown
            self._fill(shown)
            self._since.start()
        waiting = len(self._center.open) - 1
        self.waiting.setText(
            tr_plural("{count} more waiting", "{count} more waiting", waiting).format(count=waiting)
        )
        self.waiting.setVisible(waiting > 0)
        self.show_over()

    def _fill(self, notification: Notification) -> None:
        self.level_bar.setProperty("level", notification.level.value)
        repolish(self.level_bar)
        self.title.setText(notification.title)
        self.title.setVisible(bool(notification.title))
        self.message.setText(notification.message)
        self.message.setVisible(bool(notification.message))
        image = notification.image
        if image is None:
            self.image.clear()
            self.image.hide()
            self.panel.setFixedWidth(PANEL_WIDTH)
        else:
            ratio = self.devicePixelRatioF()
            side = round(IMAGE_BOX * ratio)
            pixmap = QPixmap.fromImage(
                image.scaled(
                    side,
                    side,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            pixmap.setDevicePixelRatio(ratio)
            self.image.setPixmap(pixmap)
            self.image.show()
            self.panel.setFixedWidth(PANEL_WIDTH_WITH_IMAGE)
        for button in self.buttons:
            # Out of the layout at once: deleted only later, it would still take room.
            self._button_row.removeWidget(button)
            button.hide()
            button.deleteLater()
        self.buttons = []
        if notification.buttons:
            choices = [(b.id, button_text(b.label), b.style) for b in notification.buttons]
        else:
            choices = [(DEFAULT_BUTTON_ID, tr("OK"), "accent")]
        for button_id, text, style in choices:
            button = QPushButton(text)
            if style:
                set_role(button, style)
            button.clicked.connect(lambda _checked=False, b=button_id: self._press(b))
            self._button_row.addWidget(button)
            # At once: in a panel already shown Qt would show it only later, after the
            # height was measured without the row of buttons (a notification replaced).
            button.show()
            self.buttons.append(button)
        self._show_time_left()
        if notification.timeout_s is None:
            self._countdown.stop()
        else:
            # Back after an alarm, it goes on from where its time stands.
            width = self.panel.width() - 2 * PANEL_MARGIN
            self._countdown.start(countdown_interval_ms(notification.timeout_s, width))

    def _show_time_left(self) -> None:
        notification = self._notification
        left = None if notification is None else self._center.remaining_s(notification)
        share = 1.0 if left is None else left / notification.timeout_s
        self.level_track.set_share(share)

    def _place_panel(self) -> None:
        """As high as its texts need at its fixed width: `adjustSize` measured the
        wrapped message at a narrower one and left a line of room beneath it."""
        width = self.panel.width()
        self.panel.resize(width, self.panel.layout().totalHeightForWidth(width))
        self.panel.move((self.width() - width) // 2, (self.height() - self.panel.height()) // 2)

    def _press(self, button_id: str) -> None:
        notification = self._notification
        if notification is None or self._since.elapsed() < self._guard_ms:
            return
        self._center.press(notification.id, button_id)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """A tap beside the panel: it stays."""
        event.accept()
