"""The QR code over the window (`ui.touch.qr_sheet`): the code as segno encodes it,
drawn with its quiet zone and whole pixels per module, and the sheet that shows it."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import segno
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QWidget

from ui.touch.qr_sheet import QR_SIZE, QUIET_ZONE, QrSheet, breakable, qr_image
from ui.touch.window import SCREEN_HEIGHT, SCREEN_WIDTH

URL = "https://www.deutschlandfunk.de/selenskyj-wehrt-sich-gegen-trumps-schuldzuweisung-100.html"


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def test_the_code_is_drawn_module_by_module_inside_its_quiet_zone():
    matrix = segno.make(URL, error="m", micro=False).matrix
    count = len(matrix) + 2 * QUIET_ZONE

    image = qr_image(URL)

    module = image.width() // count
    assert image.width() == image.height() == count * module <= QR_SIZE
    assert module >= 7  # about 1 mm on the Touch Display 2, for a phone at arm's length

    def dark(row: int, column: int) -> bool:
        x = (column + QUIET_ZONE) * module + module // 2
        y = (row + QUIET_ZONE) * module + module // 2
        return image.pixelColor(x, y) == QColor("black")

    assert [[dark(r, c) for c in range(len(matrix))] for r in range(len(matrix))] == [
        [bool(value) for value in line] for line in matrix
    ]
    # The quiet zone is light all round.
    edge = QUIET_ZONE * module
    assert all(image.pixelColor(x, edge - 1) == QColor("white") for x in range(image.width()))
    assert all(image.pixelColor(edge - 1, y) == QColor("white") for y in range(image.height()))


def test_a_long_address_may_break_after_slashes_and_dashes():
    shown = breakable("https://www.dlf.de/a-b")

    assert shown.replace("\u200b", "") == "https://www.dlf.de/a-b"
    assert "/\u200b" in shown and "-\u200b" in shown and ".\u200b" in shown


def test_the_sheet_shows_the_code_and_what_it_holds():
    window = QWidget()
    window.resize(SCREEN_WIDTH, SCREEN_HEIGHT)
    window.show()
    sheet = QrSheet(window)

    sheet.open("Hörerservice", "+492213451831", "tel:+492213451831")

    assert sheet.isVisible()
    assert sheet.payload == "tel:+492213451831"
    assert not sheet.code.pixmap().isNull()
    assert sheet.title.text() == "Hörerservice"
    assert sheet.detail.text() == "+492213451831"
    # The panel lies within the window.
    assert window.rect().contains(sheet.panel.geometry())

    # A name that is the address says it once.
    sheet.open("www.deutschlandfunk.de", "www.deutschlandfunk.de", "http://www.deutschlandfunk.de")
    assert sheet.detail.isHidden()

    sheet.close_button.click()
    assert not sheet.isVisible()
    window.deleteLater()
