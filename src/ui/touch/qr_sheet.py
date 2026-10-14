"""A QR code over the whole window, for what the radio cannot open itself: a web page or
a telephone number a station gives in its Journaline. The Raspberry Pi has no browser;
the phone in the listener's hand has.

Encoded by segno (ISO/IEC 18004) and drawn dark on light in either colour scheme, as
scanners expect, with the quiet zone of four modules the standard asks for. Every module
is a whole number of pixels, so that its edges stay sharp.
"""

from __future__ import annotations

import segno
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from i18n import tr
from ui.touch.overlay import Overlay
from ui.touch.widgets import set_role

# The side of the code with its quiet zone, at most: on the Touch Display 2 about 45 mm,
# a module of a long link (version 6, 49 modules with the zone) 7 px or almost 1 mm.
QR_SIZE = 360
QUIET_ZONE = 4  # modules, ISO/IEC 18004 clause 6.3.8
PANEL_WIDTH = 920
# Where a long address may break: after these, as a browser breaks a URL.
_BREAK_AFTER = "/.-_?&=#"
_ZERO_WIDTH_SPACE = "\u200b"


def qr_image(text: str, size: int = QR_SIZE) -> QImage:
    """The QR code of `text` with its quiet zone, at most `size` pixels square."""
    code = segno.make(text, error="m", micro=False)
    count = code.symbol_size(border=QUIET_ZONE)[0]
    module = max(1, size // count)
    side = count * module
    image = QImage(side, side, QImage.Format.Format_RGB32)
    image.fill(QColor("white"))
    painter = QPainter(image)
    dark = QColor("black")
    for row, line in enumerate(code.matrix):
        for column, value in enumerate(line):
            if value:
                x = (column + QUIET_ZONE) * module
                y = (row + QUIET_ZONE) * module
                painter.fillRect(x, y, module, module, dark)
    painter.end()
    return image


def breakable(address: str) -> str:
    """The address with room to break after slashes, dots and dashes: a label wraps a
    text only where it may, and a URL has no spaces."""
    return "".join(
        character + _ZERO_WIDTH_SPACE if character in _BREAK_AFTER else character
        for character in address
    )


class QrSheet(Overlay):
    """The code on the left; on the right what it holds, a name and the address beneath,
    and "Close"."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.payload = ""  # what the code shown holds
        self.panel.setFixedWidth(PANEL_WIDTH)
        self.code = QLabel()
        self.code.setFixedSize(QR_SIZE, QR_SIZE)
        self.code.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title = set_role(QLabel(), "large")
        self.title.setWordWrap(True)
        self.title.setTextFormat(Qt.TextFormat.PlainText)
        self.detail = set_role(QLabel(), "muted")
        self.detail.setWordWrap(True)
        self.detail.setTextFormat(Qt.TextFormat.PlainText)
        self.close_button = QPushButton(tr("Close"))
        self.close_button.clicked.connect(self.dismiss)
        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.close_button)
        texts = QVBoxLayout()
        texts.setSpacing(12)
        texts.addWidget(self.title)
        texts.addWidget(self.detail)
        texts.addStretch()
        texts.addLayout(buttons)
        layout = QHBoxLayout(self.panel)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(24)
        layout.addWidget(self.code)
        layout.addLayout(texts, stretch=1)

    def open(self, title: str, address: str, payload: str) -> None:
        """Shows the code of `payload` (a URL, "tel:…"), named by `title`, with the
        address as the station gives it."""
        self.payload = payload
        self.code.setPixmap(QPixmap.fromImage(qr_image(payload)))
        self.title.setText(title)
        self.detail.setText(breakable(address))
        self.detail.setVisible(bool(address) and address != title)
        self.show_over()
