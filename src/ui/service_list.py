"""Entries of the service list in the side panel: the service logo from the programme
guide, the service name and, in a second, smaller line, its programme type (FIG 0/17)."""

from __future__ import annotations

from PyQt6.QtCore import QModelIndex, QRect, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QIcon, QPainter, QPalette, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QListWidget,
    QListWidgetItem,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

from dab.programme_type import programme_type_name
from i18n import tr

SID_ROLE = Qt.ItemDataRole.UserRole
# The second line of an entry, or None for a single-line entry.
DETAIL_ROLE = Qt.ItemDataRole.UserRole + 1

_DETAIL_FONT_SCALE = 0.85
_DETAIL_OPACITY = 0.65
_TWO_LINE_PADDING = 4  # extra height of a two-line entry, so that entries stay apart

LOGO_SIZE = QSize(32, 32)  # icon size of the list


def service_item(sid: int, name: str, programme_type: int | None) -> QListWidgetItem:
    """A list entry for one service; `programme_type` is the PTy code, None while it is
    unknown (not received yet, or a list stored before programme types were kept)."""
    item = QListWidgetItem(name)
    item.setData(SID_ROLE, sid)
    lines = [tr("Service ID (SId): {sid:04X}").format(sid=sid)]
    if programme_type is None:
        lines.append(tr("Programme type: unknown (FIG 0/17 not received)"))
    else:
        names = programme_type_name(programme_type)
        short = f"PTy {programme_type}" if names is None else names.short
        full = tr("not assigned in DAB") if names is None else names.full
        item.setData(DETAIL_ROLE, short)
        lines.append(
            tr("Programme type: {name} (PTy {code}, FIG 0/17)").format(
                name=full, code=programme_type
            )
        )
    item.setToolTip("\n".join(lines))
    return item


def show_logos(service_list: QListWidget, logos: dict[int, QIcon]) -> None:
    """Puts the logo of each service (SId -> logo) in front of its name. Once one
    service of the list has a logo, the others get an empty space of the same size, so
    that the names stay in line; without any logo the list looks as before."""
    items = [service_list.item(row) for row in range(service_list.count())]
    blank = QIcon()
    if any(item.data(SID_ROLE) in logos for item in items):
        pixmap = QPixmap(service_list.iconSize())
        pixmap.fill(Qt.GlobalColor.transparent)
        blank = QIcon(pixmap)
    for item in items:
        item.setIcon(logos.get(item.data(SID_ROLE), blank))


class ServiceListDelegate(QStyledItemDelegate):
    """Draws the text of an entry with the smaller, dimmer second line from
    `DETAIL_ROLE` below it. Entries without one look as usual."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        detail = index.data(DETAIL_ROLE)
        if not detail:
            super().paint(painter, option, index)
            return
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        widget = opt.widget
        style = widget.style() if widget is not None else QApplication.style()
        # Same text position as the style uses for single-line entries, beside the logo.
        rect = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, opt, widget)
        margin = style.pixelMetric(QStyle.PixelMetric.PM_FocusFrameHMargin, None, widget) + 1
        rect = rect.adjusted(margin, 0, -margin, 0)
        name = opt.text
        # The style draws background, selection, focus and logo, the text comes below.
        opt.text = ""
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)

        detail_font = _detail_font(opt.font)
        name_metrics = QFontMetrics(opt.font)
        detail_metrics = QFontMetrics(detail_font)
        top = rect.top() + (rect.height() - name_metrics.height() - detail_metrics.height()) // 2

        selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        enabled = bool(opt.state & QStyle.StateFlag.State_Enabled)
        color = opt.palette.color(
            QPalette.ColorGroup.Normal if enabled else QPalette.ColorGroup.Disabled,
            QPalette.ColorRole.HighlightedText if selected else QPalette.ColorRole.Text,
        )
        dimmed = QColor(color)
        dimmed.setAlphaF(_DETAIL_OPACITY)
        align = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter

        painter.save()
        painter.setFont(opt.font)
        painter.setPen(color)
        name_rect = QRect(rect.left(), top, rect.width(), name_metrics.height())
        elide = Qt.TextElideMode.ElideRight
        painter.drawText(name_rect, align, name_metrics.elidedText(name, elide, rect.width()))
        painter.setFont(detail_font)
        painter.setPen(dimmed)
        detail_rect = QRect(
            rect.left(), name_rect.bottom() + 1, rect.width(), detail_metrics.height()
        )
        painter.drawText(detail_rect, align, detail_metrics.elidedText(detail, elide, rect.width()))
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        size = super().sizeHint(option, index)
        if index.data(DETAIL_ROLE):
            # Both lines of text; the logo beside them may be taller still.
            opt = QStyleOptionViewItem(option)
            self.initStyleOption(opt, index)
            opt.features &= ~QStyleOptionViewItem.ViewItemFeature.HasDecoration
            widget = opt.widget
            style = widget.style() if widget is not None else QApplication.style()
            text = style.sizeFromContents(QStyle.ContentsType.CT_ItemViewItem, opt, QSize(), widget)
            detail_height = QFontMetrics(_detail_font(option.font)).height()
            size.setHeight(max(size.height(), text.height() + detail_height + _TWO_LINE_PADDING))
        return size


def _detail_font(font: QFont) -> QFont:
    detail = QFont(font)
    if font.pointSizeF() > 0:
        detail.setPointSizeF(font.pointSizeF() * _DETAIL_FONT_SCALE)
    else:
        detail.setPixelSize(round(font.pixelSize() * _DETAIL_FONT_SCALE))
    return detail
