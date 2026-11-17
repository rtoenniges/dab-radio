"""The emergency warning playing (`ReceiverController.alert`, ASA) over the whole window:
a red bar, what it is (its stage), the station carrying it, its dynamic label, and the
buttons to break it off: once, with the repeats of its incident, or with all of its
incident but a critical warning (ETSI TS 104 089, clauses 7.4.2 and 7.6.4).

Like a notification it stays until it ends or is broken off; a tap beside it does
nothing, and its buttons take a tap only `GUARD_MS` after it appeared. The dynamic label
comes from the broadcast and is shown as plain text.
"""

from __future__ import annotations

from PyQt6.QtCore import QElapsedTimer, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from dab.ews import stage_text
from i18n import tr
from ui.receiver_controller import AlertPlayback, ReceiverController
from ui.touch.notification_view import GUARD_MS, LEVEL_BAR_HEIGHT, PANEL_MARGIN, PANEL_WIDTH
from ui.touch.overlay import Overlay
from ui.touch.widgets import repolish, set_role

# The dynamic label of a warning has 128 characters at most: four lines at the width of
# the panel.
TEXT_MAX_HEIGHT = 240


class AlertDialog(Overlay):
    """Shows the emergency warning playing while there is one; see the module
    docstring."""

    def __init__(
        self, parent: QWidget, controller: ReceiverController, guard_ms: int = GUARD_MS
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._guard_ms = guard_ms
        self._playback: AlertPlayback | None = None
        self._since = QElapsedTimer()
        self.panel.setFixedWidth(PANEL_WIDTH)

        self.level_bar = set_role(QWidget(), "level")
        self.level_bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.level_bar.setProperty("level", "alarm")
        self.level_bar.setFixedHeight(LEVEL_BAR_HEIGHT)
        repolish(self.level_bar)
        self.headline = set_role(QLabel(), "headline")
        self.station = set_role(QLabel(), "muted")
        self.station.setTextFormat(Qt.TextFormat.PlainText)
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setTextFormat(Qt.TextFormat.PlainText)
        self.text.setMaximumHeight(TEXT_MAX_HEIGHT)
        self.text.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # Translators: breaks off the emergency warning playing; its repeats would play
        # again.
        self.end_button = set_role(QPushButton(tr("End")), "danger")
        self.end_button.clicked.connect(lambda: self._end())
        # Translators: breaks off the emergency warning and skips the repeats of its
        # incident; updates still play.
        self.repeats_button = QPushButton(tr("Skip repeats"))
        self.repeats_button.clicked.connect(lambda: self._end(repeats=True))
        # Translators: breaks off the emergency warning and skips the rest of its
        # incident but a critical warning.
        self.incident_button = QPushButton(tr("Skip incident"))
        self.incident_button.clicked.connect(lambda: self._end(incident=True))
        buttons = QHBoxLayout()
        buttons.setSpacing(12)
        buttons.addWidget(self.repeats_button)
        buttons.addWidget(self.incident_button)
        buttons.addStretch()
        buttons.addWidget(self.end_button)

        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(PANEL_MARGIN, 16, PANEL_MARGIN, 20)
        layout.setSpacing(12)
        layout.addWidget(self.level_bar)
        layout.addWidget(self.headline)
        layout.addWidget(self.station)
        layout.addWidget(self.text)
        layout.addSpacing(4)
        layout.addLayout(buttons)

        controller.alert_changed.connect(self._follow)
        self._follow(controller.alert)

    @property
    def playback(self) -> AlertPlayback | None:
        """The warning on display."""
        return self._playback

    def _follow(self, playback: AlertPlayback | None) -> None:
        if playback is None:
            self._playback = None
            self.hide()
            return
        if self._playback is None or playback.alert.key != self._playback.alert.key:
            self._since.start()  # a new warning: the finger gets its pause again
        self._playback = playback
        self.headline.setText(stage_text(playback.alert.stage))
        if playback.subchannel_id is None:
            station = tr("Looking for it on channel {channel} …").format(channel=playback.channel)
        else:
            station = playback.label or ""
        self.station.setText(station)
        self.station.setVisible(bool(station))
        self.text.setText(playback.text or "")
        self.text.setVisible(bool(playback.text))
        self.show_over()

    def keep_on_top(self) -> None:
        """Raises the warning above what came up over it (a notification)."""
        if self.isVisible():
            self.raise_()

    def _place_panel(self) -> None:
        """As high as its texts need at its fixed width (see
        `NotificationDialog._place_panel`)."""
        width = self.panel.width()
        self.panel.resize(width, self.panel.layout().totalHeightForWidth(width))
        self.panel.move((self.width() - width) // 2, (self.height() - self.panel.height()) // 2)

    def _end(self, *, repeats: bool = False, incident: bool = False) -> None:
        if self._playback is None or self._since.elapsed() < self._guard_ms:
            return
        self._controller.end_alert(dismiss_repeats=repeats, dismiss_incident=incident)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """A tap beside the panel: it stays."""
        event.accept()
