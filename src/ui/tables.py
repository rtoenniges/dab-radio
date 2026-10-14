"""Tables shared by the tabs: read-only, refreshed in place."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
)

_GREYED = QColor(140, 140, 140)
# Item data role noting whether an item is greyed out, so that its brush is only set
# when that changes.
_GREY_ROLE = Qt.ItemDataRole.UserRole + 1


def fill_table(
    table: QTableWidget,
    rows: list[tuple[str, ...]],
    right_aligned: range = range(0),
    greyed: set[int] = frozenset(),
) -> None:
    """Shows `rows` in `table`, refreshed every second.

    The items stay and only take a new text where it changed: replacing them costs a
    relayout each. The columns are fitted to their content when the number of rows
    changes, which is when new content appears.
    """
    resize = table.rowCount() != len(rows)
    if resize:
        table.setRowCount(len(rows))
    grey = QBrush(_GREYED)
    for row, cells in enumerate(rows):
        for column, text in enumerate(cells):
            item = table.item(row, column)
            if item is None:
                item = QTableWidgetItem(text)
                if column in right_aligned:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                table.setItem(row, column, item)
            elif item.text() != text:
                item.setText(text)
            is_grey = row in greyed
            if item.data(_GREY_ROLE) != is_grey:
                item.setData(_GREY_ROLE, is_grey)
                item.setForeground(grey if is_grey else QBrush())
    if resize:
        # Column by column: `resizeColumnsToContents` would shrink a stretched column
        # to its content too, and it would stay so until the table is resized.
        header = table.horizontalHeader()
        for column in range(table.columnCount()):
            if header.sectionResizeMode(column) != QHeaderView.ResizeMode.Stretch:
                table.resizeColumnToContents(column)


def make_table(headers: list[str]) -> QTableWidget:
    """A read-only table with these column headers, to be filled by `fill_table`."""
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    # Not ResizeToContents: with it, every changed cell measures the whole table
    # again (1.3 ms per cell with 46 rows, while holding the GIL that the signal
    # processing needs). `fill_table` fits the columns once instead.
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    header.setStretchLastSection(True)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    return table
