"""Rows on cards for the settings (`ui.touch.settings_page` and its sub-pages).

A row names a setting on the left, with a line beneath where it needs explaining, and
shows what is set on the right: a switch (`Toggle`), a stepper, or the value with a
chevron where a tap opens something, for one of several choices the sheet with all of
them (`ChoiceRow`, `ui.touch.overlay.ChoiceSheet`). Rows that belong together stand on
one card under a small heading (`Card`), a line between them.

What the style sheet cannot draw (the switch, the chevron, the lines between the rows)
takes its colours from `theme()`, which the settings page sets with the scheme.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from PyQt6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import QAbstractButton, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ui.touch.icons import symbol_pixmap
from ui.touch.overlay import ChoiceSheet
from ui.touch.theme import DARK, THEMES, TOUCH_TARGET, Theme
from ui.touch.widgets import ElidedLabel, TapWidget, repolish, set_role

# The margin within a row, and the inset of the lines between the rows of a card.
ROW_MARGIN = 20
ROW_VERTICAL_MARGIN = 6
CHEVRON = 24
# The pill of a switch, drawn in the middle of an area to tap (the row around it
# takes the tap too, see `switch_row`).
TOGGLE_WIDTH = 64
TOGGLE_HEIGHT = 36
TOGGLE_TARGET_WIDTH = 88
TOGGLE_TARGET_HEIGHT = 56
DIVIDER_ALPHA = 120

_theme: Theme = THEMES[DARK]


def set_theme(theme: Theme) -> None:
    """The scheme the switches, chevrons and lines are drawn in from now on; what is on
    show is repainted by whoever calls (`QWidget.update`)."""
    global _theme
    _theme = theme


def theme() -> Theme:
    return _theme


class Toggle(QAbstractButton):
    """A switch: a pill with a knob, in the accent colour while on. A button like any
    other (`isChecked`, `setChecked`, `clicked`, `toggled`)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(TOGGLE_TARGET_WIDTH, TOGGLE_TARGET_HEIGHT)
        self.toggled.connect(lambda _on: self.update())

    def paintEvent(self, event: QPaintEvent) -> None:
        t = theme()
        on = self.isChecked()
        if not self.isEnabled():
            pill, knob = t.color("outline"), t.color("surface")
        elif on:
            pill, knob = t.color("accent"), t.color("on_accent")
        else:
            pill, knob = t.color("outline"), t.color("muted")
        left = (self.width() - TOGGLE_WIDTH) / 2
        top = (self.height() - TOGGLE_HEIGHT) / 2
        radius = TOGGLE_HEIGHT / 2
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(pill)
        painter.drawRoundedRect(QRectF(left, top, TOGGLE_WIDTH, TOGGLE_HEIGHT), radius, radius)
        painter.setBrush(knob)
        x = left + TOGGLE_WIDTH - radius if on else left + radius
        painter.drawEllipse(QPointF(x, top + radius), radius - 5, radius - 5)
        painter.end()


class Row(TapWidget):
    """A setting: its name (`title`, elided where it does not fit) with a line beneath
    (`subtitle`, or a widget of the caller's), and on the right the value (`value`, a
    label, where one is given), the control (a switch, a stepper) and the chevron where
    a tap opens something. A tappable row lights up while pressed, as a list row does;
    `click` is the tap. Disabled, its texts dim like the buttons."""

    def __init__(
        self,
        title: str = "",
        subtitle: str | QWidget = "",
        control: QWidget | None = None,
        value: str | None = None,
        chevron: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        set_role(self, "row")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setMinimumHeight(TOUCH_TARGET)
        self._chevron = chevron
        self._tappable = chevron
        layout = QHBoxLayout(self)
        right = ROW_MARGIN + CHEVRON + 8 if chevron else ROW_MARGIN
        layout.setContentsMargins(ROW_MARGIN, ROW_VERTICAL_MARGIN, right, ROW_VERTICAL_MARGIN)
        layout.setSpacing(16)
        texts = set_role(QWidget(), "transparent")
        column = QVBoxLayout(texts)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        self.title = ElidedLabel()
        self.title.setText(title)
        column.addWidget(self.title)
        self.subtitle: QLabel | None
        if isinstance(subtitle, QWidget):
            self.subtitle = None
            column.addWidget(subtitle)
        else:
            self.subtitle = set_role(QLabel(), "small")
            self.subtitle.setWordWrap(True)
            column.addWidget(self.subtitle)
            self.set_subtitle(subtitle)
        layout.addWidget(texts, stretch=1)
        self.value: QLabel | None = None
        if value is not None:
            self.value = QLabel(value)
            layout.addWidget(self.value)
        self.control = control
        if control is not None:
            layout.addWidget(control)

    def set_tappable(self, tappable: bool) -> None:
        """Whether a tap on the row does something (it then lights up while pressed)."""
        self._tappable = tappable

    def set_subtitle(self, text: str) -> None:
        if self.subtitle is not None:
            self.subtitle.setText(text)
            self.subtitle.setVisible(bool(text))

    def text(self) -> str:
        """The value shown, or the title of a row without one."""
        return self.value.text() if self.value is not None else self.title.text()

    def setText(self, text: str) -> None:
        """Shows `text` as the value."""
        if self.value is not None:
            self.value.setText(text)

    def set_placeholder(self, placeholder: bool) -> None:
        """Dims the value where it only says that nothing is set."""
        if self.value is not None and self.value.property("role") != (
            "muted" if placeholder else ""
        ):
            set_role(self.value, "muted" if placeholder else "")
            repolish(self.value)

    def click(self) -> None:
        """As a tap on the row."""
        self.tapped.emit()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._tappable and event.button() == Qt.MouseButton.LeftButton:
            self._set_pressed(True)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._set_pressed(False)
        super().mouseReleaseEvent(event)

    def _set_pressed(self, pressed: bool) -> None:
        if bool(self.property("pressed")) != pressed:
            self.setProperty("pressed", pressed)
            repolish(self)

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.EnabledChange:
            self.setProperty("dimmed", not self.isEnabled())
            repolish(self)
            for label in self.findChildren(QLabel):
                repolish(label)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if not self._chevron:
            return
        ratio = self.devicePixelRatioF()
        pixmap = symbol_pixmap("chevron", theme().color("muted"), CHEVRON, ratio)
        painter = QPainter(self)
        painter.drawPixmap(
            self.width() - ROW_MARGIN - CHEVRON, (self.height() - CHEVRON) // 2, pixmap
        )
        painter.end()


def switch_row(title: str, toggle: Toggle, subtitle: str = "") -> Row:
    """A row with a switch; a tap on its text flips the switch too."""
    row = Row(title, subtitle, control=toggle)
    row.set_tappable(True)
    row.tapped.connect(toggle.click)
    return row


def stepper_row(title: str, stepper: QWidget, subtitle: str = "") -> Row:
    return Row(title, subtitle, control=stepper)


def value_row(title: str, value: str = "", subtitle: str = "") -> Row:
    """Shows a value a tap lets one type (on the keyboard or the number pad)."""
    return Row(title, subtitle, value=value, chevron=True)


def action_row(title: str, subtitle: str = "", danger: bool = False) -> Row:
    """A row that leads on (in the accent colour, with a chevron) or does something at
    once that cannot be undone (`danger`)."""
    row = Row(title, subtitle, chevron=not danger)
    row.set_tappable(True)
    set_role(row.title, "danger" if danger else "accent")
    return row


def heading_label(text: str) -> QLabel:
    """The heading over a card, or over a list that stands in for one."""
    label = set_role(QLabel(text), "heading")
    label.setContentsMargins(ROW_MARGIN, 0, 0, 0)
    label.setVisible(bool(text))
    return label


class _Panel(QWidget):
    """The panel of a card: the rows one below the other, a line between those shown."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        set_role(self, "panel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 4, 0, 4)
        self._layout.setSpacing(0)
        self._rows: list[QWidget] = []

    def add(self, row: QWidget) -> None:
        self._rows.append(row)
        self._layout.addWidget(row)
        row.installEventFilter(self)

    def rows(self) -> list[QWidget]:
        return list(self._rows)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() in (QEvent.Type.ShowToParent, QEvent.Type.HideToParent):
            self.update()  # the lines follow the rows shown
        return False

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        shown = [row for row in self._rows if not row.isHidden()]
        if len(shown) < 2:
            return
        color = theme().color("outline")
        color.setAlpha(DIVIDER_ALPHA)
        painter = QPainter(self)
        painter.setPen(QPen(color, 1))
        for row in shown[1:]:
            y = row.geometry().top()
            painter.drawLine(ROW_MARGIN, y, self.width() - ROW_MARGIN - 1, y)
        painter.end()


class Card(QWidget):
    """Rows that belong together on one panel, a heading above it."""

    def __init__(
        self, heading: str = "", rows: Sequence[QWidget] = (), parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        set_role(self, "transparent")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.heading = heading_label(heading)
        layout.addWidget(self.heading)
        self.panel = _Panel()
        layout.addWidget(self.panel)
        for row in rows:
            self.add(row)

    def add(self, row: QWidget) -> None:
        self.panel.add(row)

    def rows(self) -> list[QWidget]:
        return self.panel.rows()


class ChoiceRow(Row):
    """One of several choices, in place of a row of buttons: the row shows the one in
    effect, a tap opens the sheet with all of them (`ui.touch.overlay.ChoiceSheet`).
    `current`, `set_current` and `chosen` as `ui.touch.widgets.ChoiceButtons` has them;
    `choose` is what a tap on a choice does."""

    chosen = pyqtSignal(object)

    def __init__(
        self,
        title: str,
        choices: Sequence[tuple[object, str]],
        subtitle: str = "",
        hint: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(title, subtitle, value="", chevron=True, parent=parent)
        self._keys: list[object] = []
        self._names: dict[object, str] = {}
        self._details: dict[object, str] = {}
        self._current: object = None
        self._hint = hint
        self._sheet: ChoiceSheet | None = None
        self.set_choices(choices)
        self.tapped.connect(self.open)

    def choices(self) -> list[tuple[object, str]]:
        return [(key, self._names[key]) for key in self._keys]

    def set_choices(
        self, choices: Sequence[tuple[object, str]], details: Mapping[object, str] | None = None
    ) -> None:
        """The choices from now on, each with a line beneath it on the sheet where
        `details` has one; the one in effect stays, even if it is gone from them."""
        self._keys = [key for key, _name in choices]
        self._names = dict(choices)
        self._details = dict(details or {})
        self._show()

    def current(self) -> object:
        return self._current

    def set_current(self, key: object) -> None:
        """Marks `key` as the choice in effect without reporting it."""
        self._current = key
        self._show()

    def choose(self, key: object) -> None:
        """As a tap on `key` in the sheet: marks it and reports it."""
        self.set_current(key)
        self.chosen.emit(key)

    def sheet(self) -> ChoiceSheet:
        """The sheet over the whole window, made on first use."""
        if self._sheet is None:
            self._sheet = ChoiceSheet(self.window())
        return self._sheet

    def open(self) -> None:
        self.sheet().open(
            self.title.text(),
            self.choices(),
            self._current,
            self.choose,
            self._hint,
            self._details,
        )

    def _show(self) -> None:
        self.setText(self._names.get(self._current, ""))
