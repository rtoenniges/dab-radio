"""Entries of the service list in the side panel (`ui.service_list`): logos from the
programme guide and the programme type as a second line."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtGui import QColor, QIcon, QPixmap
from PyQt6.QtWidgets import QApplication, QListWidget, QStyleOptionViewItem

from ui.service_list import LOGO_SIZE, ServiceListDelegate, service_item, show_logos

POP_MUSIC = 10  # PTy code


@pytest.fixture
def service_list() -> QListWidget:
    _app = QApplication.instance() or QApplication([])
    widget = QListWidget()
    widget.setIconSize(LOGO_SIZE)
    widget.setItemDelegate(ServiceListDelegate(widget))
    widget.addItem(service_item(0x1057, "80s80s", POP_MUSIC))
    widget.addItem(service_item(0xD230, "Dlf Nova", None))
    yield widget
    widget.deleteLater()


def _logo() -> QIcon:
    pixmap = QPixmap(128, 128)
    pixmap.fill(QColor("red"))
    return QIcon(pixmap)


def _icons(service_list: QListWidget) -> list[bool]:
    """Per entry: has an icon (a logo or the blank space)."""
    return [not service_list.item(row).icon().isNull() for row in range(service_list.count())]


def test_without_logos_the_entries_have_no_icon(service_list: QListWidget):
    show_logos(service_list, {0xABCD: _logo()})  # a service that is not on the list

    assert _icons(service_list) == [False, False]


def test_entries_without_a_logo_keep_the_names_in_line(service_list: QListWidget):
    logo = _logo()
    show_logos(service_list, {0x1057: logo})

    assert _icons(service_list) == [True, True]
    assert service_list.item(0).icon().cacheKey() == logo.cacheKey()
    blank = service_list.item(1).icon().pixmap(LOGO_SIZE).toImage()
    assert blank.pixelColor(0, 0).alpha() == 0

    show_logos(service_list, {})
    assert _icons(service_list) == [False, False]


def _height(service_list: QListWidget, row: int) -> int:
    option = QStyleOptionViewItem()
    option.initFrom(service_list)
    option.decorationSize = service_list.iconSize()  # as the view sets it
    index = service_list.model().index(row, 0)
    return service_list.itemDelegate().sizeHint(option, index).height()


def test_an_entry_is_as_tall_as_its_logo_and_its_two_lines(service_list: QListWidget):
    two_lines = _height(service_list, 0)

    show_logos(service_list, {0x1057: _logo()})

    with_logo = _height(service_list, 1)  # one line beside the blank space
    assert with_logo >= LOGO_SIZE.height()
    # The second line goes beside the logo, not below it.
    assert _height(service_list, 0) == max(two_lines, with_logo)
