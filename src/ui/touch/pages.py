"""What the swiped menus share (`gnss_view` and `traffic_view` of the services page, and
`signal_page`): a page with its name and a status line, tables for a finger, and the
note that a data service is switched off."""

from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from i18n import tr
from ui.tables import make_table
from ui.touch.theme import SMALL_FONT_PX, Theme
from ui.touch.widgets import set_role

ROW_HEIGHT = 44


def pixel_font(pixels: int) -> QFont:
    font = QFont()
    font.setPixelSize(pixels)
    return font


def rgb(color: str) -> tuple[int, int, int]:
    c = QColor(color)
    return c.red(), c.green(), c.blue()


def touch_table(headers: list[str], stretch: int | None = None) -> QTableWidget:
    """A read-only table for a finger: rows of `ROW_HEIGHT`, scrolled by the pager per
    pixel, no focus frame. The column `stretch` (the last one by default) takes the
    width left, also after `fill_table` fitted the others to their content."""
    table = make_table(headers)
    stretch = len(headers) - 1 if stretch is None else stretch
    header = table.horizontalHeader()
    header.setStretchLastSection(False)
    header.setSectionResizeMode(stretch, QHeaderView.ResizeMode.Stretch)
    table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
    table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    table.setShowGrid(False)
    table.setWordWrap(False)
    table.horizontalHeader().setHighlightSections(False)
    return table


def style_axes(plot: pg.PlotWidget) -> None:
    for name in ("left", "bottom", "top", "right"):
        plot.getAxis(name).setStyle(tickFont=pixel_font(SMALL_FONT_PX))


class Page(QWidget):
    """A page: its name on the left, a status line on the right, the content beneath."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = set_role(QLabel(title), "large")
        self.status = set_role(QLabel(), "muted")
        self.status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header = QHBoxLayout()
        header.addWidget(self.title)
        header.addStretch()
        header.addWidget(self.status)
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(12)
        self.body.addLayout(header)

    def render(self) -> None:
        raise NotImplementedError

    def apply_theme(self, theme: Theme) -> None:
        pass


class SwitchedOffNote(QWidget):
    """ "… is switched off under Data services." with a button to switch it on."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        set_role(self, "panel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.button = set_role(QPushButton(tr("Switch on")), "accent")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 8, 12, 8)
        layout.addWidget(QLabel(text), stretch=1)
        layout.addWidget(self.button)
