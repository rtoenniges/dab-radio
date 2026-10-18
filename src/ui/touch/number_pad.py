"""A number pad over the whole window, for numbers a stepper would take too many steps
to reach (the listener's position) and for IPv4 addresses: the touch display has no
keyboard."""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from i18n import tr, tr_context
from ui.touch.overlay import Overlay
from ui.touch.widgets import repolish, set_role

PANEL_WIDTH = 520
PANEL_MARGIN = 24
KEY_SPACING = 12
# More digits than a coordinate to 0.0001 degrees or a height in metres need.
MAX_DIGITS = 9
# Room for a few DNS servers.
MAX_ADDRESS_CHARS = 64
MINUS = "−"
# The key of the decimal separator, which reads as the language writes the separator
# (`decimal_separator`); `press` takes either separator for it.
DECIMAL = "decimal"
_SEPARATORS = (".", ",")
BACKSPACE = "⌫"
SIGN = "±"
# For addresses the two keys write these, whatever the language.
ADDRESS_DOT = "."
LIST_COMMA = ","
LIST_SEPARATOR = ", "

# The keys: digits like a telephone turned upside down (as on a calculator), with the
# decimal separator and zero beneath; backspace and sign on the right.
_KEYS = (
    ("7", 0, 0),
    ("8", 0, 1),
    ("9", 0, 2),
    (BACKSPACE, 0, 3),
    ("4", 1, 0),
    ("5", 1, 1),
    ("6", 1, 2),
    (SIGN, 1, 3),
    ("1", 2, 0),
    ("2", 2, 1),
    ("3", 2, 2),
    (DECIMAL, 3, 0),
    ("0", 3, 1),
)


def decimal_separator() -> str:
    """The decimal separator of the pad and of the values it edits, as the language
    writes it: "." in English, "," in German."""
    # Translators: the decimal separator of the number pad and of the position on the
    # settings page ("51.1634° N").
    return tr_context("decimal separator", ".")


def format_number(value: float, decimals: int) -> str:
    """With the decimal separator of the language and a true minus, as the pad shows
    it."""
    return f"{value:.{decimals}f}".replace(".", decimal_separator()).replace("-", MINUS)


def parse_number(text: str) -> float | None:
    """The value of the text entered, None while it is no number yet."""
    try:
        return float(text.replace(MINUS, "-").replace(decimal_separator(), "."))
    except ValueError:
        return None


class NumberPad(Overlay):
    """Covers its parent (the window) with a dimmed scrim and a panel of keys.

    `open` starts with the current value; the first digit replaces it, as on a
    calculator, the sign stays. Outside the range "Apply" is disabled and the range
    shows in red. A tap on the scrim cancels.

    `open_address` edits an IPv4 address, or a list of them separated by commas, as
    text: the key of the decimal separator writes ".", the key of the sign ",". The
    first digit replaces the text there too, but backspace takes off one character
    (", " at once), so that the end of an address is changed quickly; a comma adds to
    the list as it stands.
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._decimals = 0
        self._unit = ""
        self._limits = (0.0, 0.0)
        self._range_text = ""
        self._text = ""
        self._fresh = True
        self._on_accept: Callable[[float], None] | None = None
        # Editing an address (`open_address`).
        self._address = False
        self._several = False
        self._check: Callable[[str], bool] = bool
        self._on_text: Callable[[str], None] | None = None

        self.panel.setFixedWidth(PANEL_WIDTH)
        self._title = set_role(QLabel(), "muted")
        self._value = set_role(QLabel(), "title")
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._range = set_role(QLabel(), "small")

        self._keys: dict[str, QPushButton] = {}
        keys = QGridLayout()
        keys.setSpacing(KEY_SPACING)
        for text, row, column in _KEYS:
            caption = decimal_separator() if text == DECIMAL else text
            key = set_role(QPushButton(caption), "key")
            key.clicked.connect(lambda _checked=False, k=text: self.press(k))
            if column == 3:
                # Apart from the digits.
                keys.addWidget(key, row, column, alignment=Qt.AlignmentFlag.AlignRight)
            else:
                keys.addWidget(key, row, column)
            self._keys[text] = key
        keys.setColumnStretch(3, 1)

        self._cancel = QPushButton(tr("Cancel"))
        self._cancel.clicked.connect(self.dismiss)
        self._accept = set_role(QPushButton(tr("Apply")), "accent")
        self._accept.clicked.connect(self._on_accept_clicked)
        buttons = QHBoxLayout()
        buttons.addWidget(self._cancel)
        buttons.addStretch()
        buttons.addWidget(self._accept)

        self._header = QHBoxLayout()
        self._header.addWidget(self._title, stretch=1)
        self._header.addWidget(self._value)
        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(PANEL_MARGIN, 20, PANEL_MARGIN, 20)
        layout.setSpacing(16)
        layout.addLayout(self._header)
        layout.addLayout(keys)
        layout.addWidget(self._range)
        layout.addLayout(buttons)

    def open(
        self,
        title: str,
        value: float,
        decimals: int,
        unit: str,
        limits: tuple[float, float],
        on_accept: Callable[[float], None],
    ) -> None:
        """Shows the pad with `value`; `on_accept` gets the number taken over."""
        self._set_address_mode(False)
        self._title.setText(title)
        self._decimals = decimals
        self._unit = unit
        self._limits = limits
        self._on_accept = on_accept
        self._text = format_number(value, decimals)
        self._fresh = True
        self._keys[DECIMAL].setEnabled(decimals > 0)
        self._keys[SIGN].setEnabled(limits[0] < 0)
        low, high = (format_number(v, 0) for v in limits)
        self._range_text = tr("Range {low} to {high}{unit}").format(low=low, high=high, unit=unit)
        self._show()
        self.show_over()

    def open_address(
        self,
        title: str,
        text: str,
        on_accept: Callable[[str], None],
        check: Callable[[str], bool],
        hint: str,
        several: bool = False,
    ) -> None:
        """Shows the pad with the address `text`, or with `several` the list of them;
        "Apply" waits for `check`, `hint` beneath the keys turns red while it fails, and
        `on_accept` gets the text taken over."""
        self._set_address_mode(True)
        self._title.setText(title)
        self._several = several
        self._check = check
        self._range_text = hint
        self._on_text = on_accept
        self._text = text
        self._fresh = True
        self._keys[DECIMAL].setEnabled(True)
        self._keys[SIGN].setEnabled(several)
        self._show()
        self.show_over()

    def _set_address_mode(self, address: bool) -> None:
        self._address = address
        self._keys[DECIMAL].setText(ADDRESS_DOT if address else decimal_separator())
        self._keys[SIGN].setText(LIST_COMMA if address else SIGN)

    def dismiss(self) -> None:
        self._on_accept = None
        self._on_text = None
        super().dismiss()

    @property
    def text(self) -> str:
        return self._text

    def value(self) -> float | None:
        """The number entered if it lies in the range, else None."""
        value = parse_number(self._text)
        if value is None or not self._limits[0] <= value <= self._limits[1]:
            return None
        return value

    def press(self, key: str) -> None:
        """What a key does to the text entered; the decimal separator either as
        `DECIMAL` or as a character, "." or ","."""
        if self._address:
            self._press_address(key)
            self._show()
            return
        if key in _SEPARATORS:
            key = DECIMAL
        separator = decimal_separator()
        negative = self._text.startswith(MINUS)
        digits = self._text.removeprefix(MINUS)
        if key == SIGN:
            negative = not negative
        elif key == BACKSPACE:
            digits = "" if self._fresh else digits[:-1]
            self._fresh = False
        elif key == DECIMAL:
            if self._fresh:
                digits, self._fresh = "0", False
            if separator not in digits:
                digits = (digits or "0") + separator
        else:
            if self._fresh:
                digits, self._fresh = "", False
            _whole, point, fraction = digits.partition(separator)
            # No more decimals than the value has, no more digits than it can need.
            full = point and len(fraction) >= self._decimals
            if not full and sum(c.isdigit() for c in digits) < MAX_DIGITS:
                digits = key if digits == "0" else digits + key
        self._text = (MINUS if negative else "") + digits
        self._show()

    def _press_address(self, key: str) -> None:
        if key == BACKSPACE:
            cut = len(LIST_SEPARATOR) if self._text.endswith(LIST_SEPARATOR) else 1
            self._text = self._text[:-cut]
        elif key in (SIGN, LIST_COMMA):
            if self._several and self._text and not self._text.endswith(LIST_SEPARATOR):
                self._text += LIST_SEPARATOR
        else:
            character = ADDRESS_DOT if key == DECIMAL else key
            if self._fresh:
                self._text = ""
            if len(self._text) < MAX_ADDRESS_CHARS:
                self._text += character
        self._fresh = False

    def _show(self) -> None:
        if self._address:
            self._value.setText(self._fitting(self._text) or " ")
            valid = self._check(self._text)
        else:
            self._value.setText(f"{self._text or ' '}{self._unit}")
            valid = self.value() is not None
        self._accept.setEnabled(valid)
        self._range.setText(self._range_text)
        role = "small" if valid else "error"
        if self._range.property("role") != role:
            set_role(self._range, role)
            repolish(self._range)

    def _fitting(self, text: str) -> str:
        """A list of addresses may be wider than the panel: its start gives way, the
        end being where the typing is."""
        self._title.ensurePolished()
        self._value.ensurePolished()
        title = self._title.fontMetrics().horizontalAdvance(self._title.text())
        spacing = max(self._header.spacing(), 0)
        room = PANEL_WIDTH - 2 * PANEL_MARGIN - spacing - title
        return self._value.fontMetrics().elidedText(text, Qt.TextElideMode.ElideLeft, room)

    def _on_accept_clicked(self) -> None:
        if self._address:
            text, on_text = self._text, self._on_text
            valid = self._check(text)
            self.dismiss()
            if valid and on_text is not None:
                on_text(text)
            return
        value = self.value()
        on_accept = self._on_accept
        self.dismiss()
        if value is not None and on_accept is not None:
            on_accept(value)
