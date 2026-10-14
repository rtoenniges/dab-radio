"""`ui.plots.MapLabel`: a name on a map, above all drawn on it, with a halo in the
colour of the ground."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import QApplication

from ui.plots import HALO_WIDTH, MAP_LABEL_Z, MapLabel

APP = QApplication.instance() or QApplication([])


def _halo_format(label: MapLabel) -> QTextCharFormat:
    """The format of the halo's last letter."""
    cursor = QTextCursor(label._halo.document())
    cursor.movePosition(QTextCursor.MoveOperation.End)
    return cursor.charFormat()


def test_the_halo_follows_the_text_its_font_and_place() -> None:
    label = MapLabel("Köln", color=(170, 170, 180))
    assert label.zValue() == MAP_LABEL_Z
    font = QFont()
    font.setPixelSize(16)
    label.setFont(font)
    label.setText("Frankfurt am Main")

    halo = label._halo
    assert halo.toPlainText() == "Frankfurt am Main"
    assert halo.font().pixelSize() == 16
    # Beneath the letters and in their place, which the anchor moved up.
    assert halo.zValue() < label.textItem.zValue()
    assert halo.pos() == label.textItem.pos()
    assert label.textItem.pos().y() < 0
    outline = _halo_format(label).textOutline()
    assert outline.color() == QColor(0, 0, 0)  # the desktop's black plots
    assert outline.widthF() == HALO_WIDTH


def test_the_halo_takes_the_colour_of_the_ground() -> None:
    label = MapLabel("Köln", color="#5b6875", halo="#182027")
    label.set_halo("#ffffff")
    assert label.halo == QColor("#ffffff")
    assert _halo_format(label).textOutline().color() == QColor("#ffffff")
    assert _halo_format(label).foreground().color() == QColor("#ffffff")
    assert label.color == QColor("#5b6875")
