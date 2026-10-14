"""What lies over the whole window for a moment: a dimmed scrim with a panel in its
middle (the number pad, the keyboard, the choices for a network or a device). A tap on
the scrim beside the panel cancels."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from i18n import tr
from ui.touch.widgets import set_role

SHEET_WIDTH = 640
# What the screen leaves for a wider sheet, 40 px to either edge.
SHEET_MAX_WIDTH = 1200


class Overlay(QWidget):
    """Covers its parent (the window); fill `panel` and call `show_over` to show it."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        set_role(self, "scrim")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.panel = set_role(QWidget(self), "panel")
        self.panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        parent.installEventFilter(self)
        self.hide()

    def show_over(self) -> None:
        self.setGeometry(self.parentWidget().rect())
        self._place_panel()
        self.show()
        self.raise_()

    def dismiss(self) -> None:
        """Hides the overlay; a subclass forgets here what it was opened for."""
        self.hide()

    def _place_panel(self) -> None:
        self.panel.adjustSize()
        self.panel.move(
            (self.width() - self.panel.width()) // 2, (self.height() - self.panel.height()) // 2
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self.panel.geometry().contains(event.position().toPoint()):
            self.dismiss()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        resized = event.type() == QEvent.Type.Resize
        if watched is self.parentWidget() and resized and self.isVisible():
            self.setGeometry(self.parentWidget().rect())
            self._place_panel()
        return False


class ActionSheet(Overlay):
    """What can be done with something (a network, a device, the output): its name, a
    line beneath, "Cancel" on the left and the actions on the right."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        # Wider where its actions need it (the outputs of the system have long names).
        self.panel.setMinimumWidth(SHEET_WIDTH)
        self.panel.setMaximumWidth(SHEET_MAX_WIDTH)
        self.title = set_role(QLabel(), "large")
        self.title.setWordWrap(True)
        self.detail = set_role(QLabel(), "muted")
        self.detail.setWordWrap(True)
        self._cancel = QPushButton()
        self._cancel.clicked.connect(self.dismiss)
        self._buttons = QHBoxLayout()
        self._buttons.setSpacing(12)
        self._buttons.addWidget(self._cancel)
        self._buttons.addStretch()
        self.actions: list[QPushButton] = []
        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)
        layout.addWidget(self.title)
        layout.addWidget(self.detail)
        layout.addLayout(self._buttons)

    def open(
        self,
        title: str,
        detail: str,
        actions: list[tuple[str, str, Callable[[], None]]],
        cancel: str | None = None,
    ) -> None:
        """`actions`: text, role of the button ("accent", "danger", "" for a plain one)
        and what it does. `cancel`: the text of the way back, "Cancel" unless given."""
        self.title.setText(title)
        self._cancel.setText(cancel if cancel is not None else tr("Cancel"))
        self.detail.setText(detail)
        self.detail.setVisible(bool(detail))
        for button in self.actions:
            # Out of the layout at once: deleted only later, it would still take room.
            self._buttons.removeWidget(button)
            button.hide()
            button.deleteLater()
        self.actions = []
        for text, role, action in actions:
            button = QPushButton(text)
            if role:
                set_role(button, role)
            button.clicked.connect(lambda _checked=False, a=action: self._choose(a))
            self._buttons.addWidget(button)
            self.actions.append(button)
        self.show_over()

    def _choose(self, action: Callable[[], None]) -> None:
        self.dismiss()
        action()
