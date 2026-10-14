"""A keyboard over the whole window, for text the touch display has no other way to take
(name and password of a Wi-Fi). German layout (QWERTZ) with umlauts; a second layout
holds every other printable ASCII character, any of which a WPA password may contain.

On a PC the keys of a real keyboard work as well."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from i18n import tr
from ui.touch.icons import symbol_icon
from ui.touch.overlay import Overlay
from ui.touch.theme import DARK, KEY_WIDTH, THEMES, Theme
from ui.touch.widgets import repolish, set_role

KEY_SPACING = 8
COLUMNS = 11
# Longer than any SSID (32 bytes) or WPA key (64 hex digits) needs.
MAX_BYTES = 64
FIELD_WIDTH = COLUMNS * KEY_WIDTH + (COLUMNS - 1) * KEY_SPACING
EYE_ICON = QSize(32, 32)

SHIFT = "⇧"
BACKSPACE = "⌫"
SPACE = " "
# The key that switches between letters and symbols, and what it reads on either.
LAYOUT = "layout"
_TO_SYMBOLS = "?123"
_TO_LETTERS = "ABC"
DOT = "•"

# Four rows of eleven characters; the last row leaves the first column to shift and the
# last to backspace. The digits stay in both layouts.
# fmt: off
_LETTERS = (
    "1234567890ß",
    "qwertzuiopü",
    "asdfghjklöä",
    "yxcvbnm-.",
)
_SYMBOLS = (
    "1234567890~",
    "!\"#$%&'()*+",
    ",-./:;<=>?@",
    "[\\]^_`{|}",
)
# fmt: on


class Shift(Enum):
    OFF = "off"
    ONCE = "once"  # for the next letter
    LOCKED = "locked"  # until tapped again


class Keyboard(Overlay):
    """`open` shows it with a title, the text to start from and what to do with the text
    taken over. "Apply" (or what `accept_text` says) is disabled while `check`
    rejects the text, and the hint beneath the text then shows in red. A secret text
    shows as dots until the eye is tapped."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._text = ""
        self._secret = False
        self._symbols = False
        self._shift = Shift.OFF
        self._check: Callable[[str], bool] = bool
        self._hint_text = ""
        self._on_accept: Callable[[str], None] | None = None

        self._title = set_role(QLabel(), "muted")
        self._eye = QPushButton()
        self._eye.setCheckable(True)
        self._eye.setIconSize(EYE_ICON)
        self._eye.toggled.connect(lambda _shown: self._show())
        cancel = QPushButton(tr("Cancel"))
        cancel.clicked.connect(self.dismiss)
        header = QHBoxLayout()
        header.setSpacing(KEY_SPACING)
        header.addWidget(self._title, stretch=1)
        header.addWidget(self._eye)
        header.addWidget(cancel)

        self._field = set_role(QLabel(), "field")
        self._field.setFixedWidth(FIELD_WIDTH)
        self._hint = set_role(QLabel(), "small")

        # One button per place; what it types changes with the layout and shift.
        keys = QGridLayout()
        keys.setSpacing(KEY_SPACING)
        self._character_keys: list[tuple[QPushButton, int, int]] = []
        self._characters: dict[QPushButton, str] = {}
        for row, characters in enumerate(_LETTERS):
            first = 1 if row == len(_LETTERS) - 1 else 0
            for index in range(len(characters)):
                key = set_role(QPushButton(), "key")
                key.clicked.connect(lambda _checked=False, k=key: self.press(self._characters[k]))
                keys.addWidget(key, row, first + index)
                self._character_keys.append((key, row, index))
        last_row = len(_LETTERS) - 1
        self._shift_key = set_role(QPushButton(SHIFT), "key")
        self._shift_key.setCheckable(True)
        self._shift_key.clicked.connect(lambda: self.press(SHIFT))
        # Keeps its place in the symbols, which have no capitals.
        policy = self._shift_key.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self._shift_key.setSizePolicy(policy)
        keys.addWidget(self._shift_key, last_row, 0)
        backspace = set_role(QPushButton(BACKSPACE), "key")
        backspace.setAutoRepeat(True)
        backspace.clicked.connect(lambda: self.press(BACKSPACE))
        keys.addWidget(backspace, last_row, COLUMNS - 1)
        self._layout_key = QPushButton(_TO_SYMBOLS)
        self._layout_key.clicked.connect(lambda: self.press(LAYOUT))
        # Translators: the space bar of the on-screen keyboard.
        space = QPushButton(tr("Space"))
        space.clicked.connect(lambda: self.press(SPACE))
        self._accept = set_role(QPushButton(), "accent")
        self._accept.clicked.connect(self.accept)
        bottom = last_row + 1
        keys.addWidget(self._layout_key, bottom, 0, 1, 2)
        keys.addWidget(space, bottom, 2, 1, 6)
        keys.addWidget(self._accept, bottom, 8, 1, 3)

        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(24, 16, 24, 20)
        layout.setSpacing(12)
        layout.addLayout(header)
        layout.addWidget(self._field)
        layout.addWidget(self._hint)
        layout.addLayout(keys)
        self.apply_theme(THEMES[DARK])

    def apply_theme(self, theme: Theme) -> None:
        self._eye.setIcon(
            symbol_icon("eye", theme.color("text"), EYE_ICON, theme.color("on_accent"))
        )

    def open(
        self,
        title: str,
        on_accept: Callable[[str], None],
        *,
        accept_text: str | None = None,
        text: str = "",
        secret: bool = False,
        check: Callable[[str], bool] = bool,
        hint: str = "",
    ) -> None:
        """Shows the keyboard with `text`; `on_accept` gets the text taken over, with the
        button `accept_text` ("Apply" without one)."""
        self._title.setText(title)
        self._on_accept = on_accept
        self._accept.setText(tr("Apply") if accept_text is None else accept_text)
        self._text = text
        self._secret = secret
        self._check = check
        self._hint_text = hint
        self._symbols = False
        self._shift = Shift.OFF
        self._eye.setVisible(secret)
        self._eye.setChecked(False)
        self._show()
        self.show_over()
        self.setFocus()

    def dismiss(self) -> None:
        self._on_accept = None
        # Nothing of a password stays behind in the hidden keyboard.
        self._text = ""
        self._show()
        super().dismiss()

    @property
    def text(self) -> str:
        return self._text

    @property
    def shift(self) -> Shift:
        return self._shift

    def characters(self) -> list[str]:
        """What the character keys type now, row by row."""
        return [self._characters[key] for key, _row, _index in self._character_keys]

    def press(self, key: str) -> None:
        """What a key does: a character is typed, the others as their names say."""
        if key == SHIFT:
            following = {Shift.OFF: Shift.ONCE, Shift.ONCE: Shift.LOCKED, Shift.LOCKED: Shift.OFF}
            self._shift = following[self._shift]
        elif key == LAYOUT:
            self._symbols = not self._symbols
        elif key == BACKSPACE:
            self._text = self._text[:-1]
        elif len((self._text + key).encode()) <= MAX_BYTES:
            self._text += key
            if self._shift is Shift.ONCE:
                self._shift = Shift.OFF
        self._show()

    def accept(self) -> None:
        if not self._check(self._text):
            return
        text, on_accept = self._text, self._on_accept
        self.dismiss()
        if on_accept is not None:
            on_accept(text)

    def _show(self) -> None:
        rows = _SYMBOLS if self._symbols else _LETTERS
        capitals = self._shift is not Shift.OFF and not self._symbols
        for key, row, index in self._character_keys:
            character = rows[row][index]
            # Only letters have capitals; "ß" would become "SS".
            if capitals and row > 0:
                character = character.upper()
            self._characters[key] = character
            key.setText(character.replace("&", "&&"))  # "&" alone marks a shortcut
        self._layout_key.setText(_TO_LETTERS if self._symbols else _TO_SYMBOLS)
        self._shift_key.setVisible(not self._symbols)
        self._shift_key.setChecked(self._shift is not Shift.OFF)
        locked = self._shift is Shift.LOCKED
        if self._shift_key.property("locked") != locked:
            self._shift_key.setProperty("locked", locked)
            repolish(self._shift_key)

        shown = DOT * len(self._text) if self._secret and not self._eye.isChecked() else self._text
        metrics = self._field.fontMetrics()
        margins = self._field.contentsMargins()
        width = FIELD_WIDTH - margins.left() - margins.right() - 8
        # The end of a long text, where the next character goes.
        self._field.setText(metrics.elidedText(shown, Qt.TextElideMode.ElideLeft, width) or " ")

        valid = self._check(self._text)
        self._accept.setEnabled(valid)
        self._hint.setText(self._hint_text)
        self._hint.setVisible(bool(self._hint_text))
        role = "small" if valid else "error"
        if self._hint.property("role") != role:
            set_role(self._hint, role)
            repolish(self._hint)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_Backspace:
            self.press(BACKSPACE)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.accept()
        elif key == Qt.Key.Key_Escape:
            self.dismiss()
        elif event.text() and event.text().isprintable():
            # Already in capitals where the shift of the real keyboard was held.
            for character in event.text():
                self.press(character)
        else:
            super().keyPressEvent(event)
