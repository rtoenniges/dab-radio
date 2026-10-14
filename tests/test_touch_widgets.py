"""Widgets of the touch surface (`ui.touch.widgets`): the row of chips that is swiped
sideways where it is wider than the page."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget

from ui.touch.theme import DARK, THEMES, stylesheet, touch_font
from ui.touch.widgets import CHIP_ROW_HEIGHT, ChipRow, ChoiceButtons


@pytest.fixture(scope="module", autouse=True)
def qapp():
    app = QApplication.instance() or QApplication([])
    previous = app.font()
    app.setFont(touch_font())
    yield app
    app.setFont(previous)


def _row(names: list[str], width: int) -> tuple[QWidget, ChipRow, ChoiceButtons]:
    window = QWidget()
    window.setStyleSheet(stylesheet(THEMES[DARK]))
    window.resize(width, 200)
    chips = ChoiceButtons([(name, name) for name in names], role="chip")
    area = ChipRow(chips)
    layout = QVBoxLayout(window)
    layout.addWidget(area)
    window.show()
    QApplication.processEvents()
    return window, area, chips


def test_a_row_wider_than_the_page_scrolls_instead_of_squeezing_its_chips():
    names = [f"Seite {number}" for number in range(1, 16)]
    window, area, chips = _row(names, 800)
    try:
        buttons = chips.findChildren(QPushButton)
        assert len(buttons) == 15
        assert area.height() == CHIP_ROW_HEIGHT
        # Every chip at its natural width, the row as wide as all of them together.
        for button in buttons:
            assert button.width() == button.sizeHint().width(), button.text()
        assert chips.width() >= sum(button.width() for button in buttons)
        assert chips.width() > area.viewport().width()
        bar = area.horizontalScrollBar()
        assert bar.maximum() == chips.width() - area.viewport().width()
        # More beyond the right edge: the fade there shows it, none on the left.
        assert area.fades.end.isVisible()
        assert not area.fades.start.isVisible()
        bar.setValue(bar.maximum())
        assert area.fades.start.isVisible()
        assert not area.fades.end.isVisible()
    finally:
        window.deleteLater()


def test_a_row_that_fits_needs_no_scrolling():
    window, area, chips = _row(["SDR", "Display", "Audio"], 800)
    try:
        assert area.horizontalScrollBar().maximum() == 0
        assert not area.fades.start.isVisible()
        assert not area.fades.end.isVisible()
        assert chips.set_current("Audio") is None
        assert chips.current() == "Audio"
    finally:
        window.deleteLater()
