"""What lies over the whole window for a moment: a dimmed scrim with a panel in its
middle (the number pad, the keyboard, the choices for a network or a device). A tap on
the scrim beside the panel cancels."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from i18n import tr
from ui.touch.theme import TOUCH_TARGET
from ui.touch.widgets import TapWidget, repolish, set_role

SHEET_WIDTH = 640
# What the screen leaves for a wider sheet, 40 px to either edge.
SHEET_MAX_WIDTH = 1200
# The sheet with the choices of a setting: wide enough for the names of the outputs.
CHOICE_SHEET_WIDTH = 760
# Added before a choice that begins a group (the stick, the tuners of the RSPduo).
CHOICE_GROUP_GAP = 16
CHECK_MARK = "✓"


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


class _ChoiceRow(TapWidget):
    """A choice on the sheet: its name with a line beneath, ✓ on the one in effect,
    which also stands in the accent colour. It lights up while pressed."""

    def __init__(self, key: object, text: str, detail: str, chosen: bool) -> None:
        super().__init__()
        self.key = key
        set_role(self, "row")
        self.setProperty("selected", chosen)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setMinimumHeight(TOUCH_TARGET)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(16)
        texts = set_role(QWidget(), "transparent")
        column = QVBoxLayout(texts)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        self.name = QLabel(text)
        column.addWidget(self.name)
        self.detail = set_role(QLabel(detail), "small")
        self.detail.setVisible(bool(detail))
        column.addWidget(self.detail)
        layout.addWidget(texts, stretch=1)
        if chosen:
            layout.addWidget(set_role(QLabel(CHECK_MARK), "large"))

    def click(self) -> None:
        """As a tap on the choice."""
        self.tapped.emit()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._set_pressed(True)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._set_pressed(False)
        super().mouseReleaseEvent(event)

    def _set_pressed(self, pressed: bool) -> None:
        self.setProperty("pressed", pressed)
        repolish(self)


class ChoiceSheet(Overlay):
    """The choices of a setting (the device, the language, the output): the setting's
    name, the choices as rows with the one in effect marked, a hint beneath where there
    is one, and "Cancel". A tap on a choice closes the sheet and reports it."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.panel.setFixedWidth(CHOICE_SHEET_WIDTH)
        self.title = set_role(QLabel(), "large")
        self.title.setWordWrap(True)
        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(4)
        self.hint = set_role(QLabel(), "small")
        self.hint.setWordWrap(True)
        self._cancel = QPushButton()
        self._cancel.clicked.connect(self.dismiss)
        buttons = QHBoxLayout()
        buttons.addWidget(self._cancel)
        buttons.addStretch()
        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        layout.addWidget(self.title)
        layout.addLayout(self._rows_layout)
        layout.addWidget(self.hint)
        layout.addLayout(buttons)
        self.rows: list[_ChoiceRow] = []
        self._choose: Callable[[object], None] | None = None

    def open(
        self,
        title: str,
        choices: Sequence[tuple[object, str]],
        current: object,
        choose: Callable[[object], None],
        hint: str = "",
        details: Mapping[object, str] | None = None,
        gaps: Sequence[object] = (),
    ) -> None:
        """`choices`: key and name of each; `current`: the key in effect; `choose`: what
        a tap on a choice does, with its key; `details`: a line beneath a name; `gaps`:
        the keys that begin a group, set apart by a wider gap."""
        self.title.setText(title)
        self._cancel.setText(tr("Cancel"))
        self.hint.setText(hint)
        self.hint.setVisible(bool(hint))
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()  # out of sight at once: deleted only later
                widget.deleteLater()
        self.rows = []
        details = details or {}
        for key, text in choices:
            if key in gaps:
                self._rows_layout.addSpacing(CHOICE_GROUP_GAP)
            row = _ChoiceRow(key, text, details.get(key, ""), key == current)
            row.tapped.connect(lambda k=key: self._pick(k))
            self._rows_layout.addWidget(row)
            self.rows.append(row)
        self._choose = choose
        self.show_over()

    def _pick(self, key: object) -> None:
        choose, self._choose = self._choose, None
        self.dismiss()
        if choose is not None:
            choose(key)
