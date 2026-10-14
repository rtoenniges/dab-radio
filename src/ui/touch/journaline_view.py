"""Journaline on the touch surface (ETSI TS 102 979): the card on the radio page and the
page it opens, a sub-page of the radio page like the equalizer (`ui.touch.radio_page`).

The card shows while FIG 0/13 announces Journaline for the station playing and the
column beside the slide has room for it (an announcement, DL Plus and the programme
come first): the menus of its main menu and the number of pages received. The page
shows two columns: on the left a menu, on the right what is chosen in it, a menu again
or a message. A tap on an entry of the menu on the right moves that menu to the left;
the arrow above goes back up. Beside the arrow the path from the main menu, named as
the menus name their entries.

As the programme guide marks the programmes with a description, an arrow marks the
entries with more behind them than their headline (`has_more`): a menu, a text, a list
or a link to another page. A short message with only a link to the web has none (most
of the Dlf's programme preview: "20:00 Nachrichten"); it can be tapped all the same.
Pages not received yet are listed muted (clause 7.2.3); they can be chosen all the same
and appear as soon as the station sends them. A text message the station changes while
it is shown stays as it is until "New version received - show" is tapped (clause
4.2.2.1); menus and short messages follow at once.

The Dlf heads its news "topic - headline - quote": the topic before the first " - "
stands small above the headline (`split_topic`).

A link to the web, a telephone number or one for an SMS below a message shows its QR
code on a tap (`ui.touch.qr_sheet`, `link_payload`): the Raspberry Pi has no browser
and no telephone, the phone in the listener's hand has both.
"""

from __future__ import annotations

import html
import time

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QFont, QFontMetrics, QHideEvent, QMouseEvent, QShowEvent
from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QScroller,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from dab.journaline import JmlObject, LinkTarget, LinkType, ObjectType
from i18n import tr, tr_plural
from ui.journaline_model import JournalineModel
from ui.touch.icons import symbol_icon, symbol_pixmap
from ui.touch.qr_sheet import QrSheet
from ui.touch.theme import DARK, FONT_PX, SMALL_FONT_PX, THEMES, Theme
from ui.touch.widgets import ElidedLabel, TapWidget, repolish, set_role

# The page beside the navigation: 1168 px, less 24 px margin on either side.
PAGE_WIDTH = 1120
LEFT_WIDTH = 400
COLUMN_GAP = 24
# The margins of a row, the gap to its arrow and the arrow.
ROW_MARGIN_LEFT = 16
ROW_MARGIN_RIGHT = 8
ROW_SPACING = 8
ROW_CHEVRON = 24
ROW_HEIGHT = 64
# The menu in the right column sits this far inside its panel.
PANEL_INSET = 8
# The width of the text in a row of either column.
LEFT_TEXT_WIDTH = LEFT_WIDTH - ROW_MARGIN_LEFT - ROW_MARGIN_RIGHT - ROW_SPACING - ROW_CHEVRON
RIGHT_TEXT_WIDTH = (
    PAGE_WIDTH
    - LEFT_WIDTH
    - COLUMN_GAP
    - 2 * PANEL_INSET
    - ROW_MARGIN_LEFT
    - ROW_MARGIN_RIGHT
    - ROW_SPACING
    - ROW_CHEVRON
)
HEADLINE_LINES = 2
CARD_ICON = 32
CARD_CHEVRON = 28
LINK_ICON = 24
BUTTON_ICON = QSize(28, 28)

_TOPIC_SEPARATOR = " - "


def split_topic(text: str) -> tuple[str, str]:
    """The topic before the first " - " and the headline after it, as the Dlf heads its
    news ("Formel 1 - Verstappen gewinnt Grand Prix in Singapur"); without one, no
    topic and the whole text."""
    topic, separator, headline = text.partition(_TOPIC_SEPARATOR)
    if not separator or not topic.strip() or not headline.strip():
        return "", text
    return topic.strip(), headline.strip()


def one_line(text: str) -> str:
    """A text of the station on one line: its line breaks and runs of spaces as one."""
    return " ".join(text.split())


def elided_lines(text: str, metrics: QFontMetrics, width: int, lines: int) -> str:
    """The text broken into at most `lines` lines of `width` pixels, the last one
    elided if the text goes on. A word longer than a line stands on a line of its own,
    elided."""
    words = text.split()
    result: list[str] = []
    while words and len(result) < lines:
        line = words.pop(0)
        while words and metrics.horizontalAdvance(f"{line} {words[0]}") <= width:
            line = f"{line} {words.pop(0)}"
        result.append(line)
    if words:
        result[-1] = f"{result[-1]} {' '.join(words)}"
    return "\n".join(
        metrics.elidedText(line, Qt.TextElideMode.ElideRight, width) for line in result
    )


def has_more(jml_object: JmlObject) -> bool:
    """Whether a page has more to show than its headline: a menu, a text, a list or a
    link to another page. A link to the web alone does not count: the Dlf gives every
    entry of its programme preview the same one."""
    if jml_object.object_type is ObjectType.MENU:
        return bool(jml_object.menu_items)
    return (
        bool(jml_object.body.strip())
        or bool(jml_object.list_items)
        or any(link.link_type is LinkType.JML_OBJECT for link in jml_object.links)
    )


def link_payload(link: LinkTarget) -> str | None:
    """What the QR code of a link holds: the web address (with "http://" where the
    station leaves it out), "tel:" or "sms:" and the number to dial; None for what a
    phone can do nothing with (a DAB link, a link to another page, an empty one)."""
    address = link.address.strip()
    if not address:
        return None
    if link.link_type is LinkType.URL:
        if "://" in address or address.lower().startswith("mailto:"):
            return address
        return f"http://{address}"
    if link.link_type in (LinkType.TELEPHONE, LinkType.SMS):
        digits = "".join(c for c in address if c.isdigit())
        if not digits:
            return None
        scheme = "tel" if link.link_type is LinkType.TELEPHONE else "sms"
        prefix = "+" if address.startswith("+") else ""
        return f"{scheme}:{prefix}{digits}"
    return None


def page_count_text(count: int) -> str:
    return tr_plural("{count} page", "{count} pages", count).format(count=count)


def received_text(timestamp: float) -> str:
    moment = time.strftime("%H:%M", time.localtime(timestamp))
    return tr("Received at {time}").format(time=moment)


def _font(pixel_size: int) -> QFont:
    """The font the style sheet gives a label of this size. Set on the label from the
    start: a scroll area lays its rows out before the style sheet reaches them."""
    font = QFont(QApplication.font())
    font.setPixelSize(pixel_size)
    return font


def _html_text(text: str) -> str:
    """Escapes a text of the station; its line breaks are kept."""
    return html.escape(text).replace("\n", "<br>")


def _highlighted_html(text: str, highlights: tuple[tuple[int, int], ...]) -> str:
    """Like `_html_text`, the highlighted passages in bold."""
    parts = []
    position = 0
    for start, end in highlights:
        parts.append(_html_text(text[position:start]))
        parts.append(f"<b>{_html_text(text[start:end])}</b>")
        position = end
    parts.append(_html_text(text[position:]))
    return "".join(parts)


def _scroll(widget: QWidget) -> QScrollArea:
    area = set_role(QScrollArea(), "transparent")
    area.setWidget(widget)
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    set_role(area.viewport(), "transparent")
    # Swiped, as a finger expects; taps still reach the rows.
    QScroller.grabGesture(area.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture)
    return area


class JournalineCard(TapWidget):
    """The menus of the station's main menu and the number of pages received; a tap
    opens the Journaline page. Whether it shows, the radio page decides: only while
    the station announces Journaline, and only where there is room."""

    def __init__(self, model: JournalineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        set_role(self, "panel")
        # A plain QWidget subclass draws the background of the style sheet only so.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self._icon = QLabel()
        self._icon.setFixedSize(CARD_ICON, CARD_ICON)
        self._chevron = QLabel()
        self._chevron.setFixedSize(CARD_CHEVRON, CARD_CHEVRON)
        title = set_role(QLabel("Journaline"), "large")
        self.count = set_role(QLabel(), "small")
        self.detail = set_role(ElidedLabel(), "muted")
        layout = QGridLayout(self)
        layout.setContentsMargins(16, 12, 12, 12)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(4)
        layout.addWidget(self._icon, 0, 0, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(title, 0, 1)
        layout.addWidget(self.count, 0, 2, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.detail, 1, 1, 1, 2)
        layout.addWidget(self._chevron, 0, 3, 2, 1, alignment=Qt.AlignmentFlag.AlignVCenter)
        layout.setColumnStretch(1, 1)
        model.changed.connect(self._on_changed)
        self.refresh()

    def apply_theme(self, theme: Theme) -> None:
        self._icon.setPixmap(symbol_pixmap("journaline", theme.color("accent"), CARD_ICON))
        self._chevron.setPixmap(symbol_pixmap("chevron", theme.color("muted"), CARD_CHEVRON))

    def _on_changed(self, _ids: frozenset[int] | None) -> None:
        self.refresh()

    def refresh(self) -> None:
        model = self._model
        count = len(model.objects())
        self.count.setText(page_count_text(count) if count else "")
        entry = model.get(model.entry)
        names = [one_line(item.text) for item in entry.menu_items] if entry else []
        names = [name for name in names if name]
        self.detail.setText(" · ".join(names) if names else tr("Receiving …"))


class _PressRow(TapWidget):
    """A row to tap, shaded while the finger is on it."""

    def __init__(self) -> None:
        super().__init__()
        set_role(self, "row")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._set_pressed(True)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._set_pressed(False)
        super().mouseReleaseEvent(event)

    def _set_pressed(self, pressed: bool) -> None:
        self.setProperty("pressed", pressed)
        repolish(self)


class _Row(_PressRow):
    """An entry of a menu: the topic small above, the headline on up to two lines, an
    arrow if there is more behind it (`has_more`); muted while the page behind it has
    not arrived, marked while chosen."""

    def __init__(
        self,
        target: int,
        text: str,
        *,
        more: bool,
        received: bool,
        selected: bool,
        text_width: int,
        theme: Theme,
    ) -> None:
        super().__init__()
        self.target = target
        self.setProperty("selected", selected)
        # Never squeezed below what its text needs: a minimum height set explicitly
        # would replace that, and the scroll area would shrink the rows to it.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        topic, headline = split_topic(text)
        font = _font(FONT_PX)
        self.headline = QLabel(
            elided_lines(headline, QFontMetrics(font), text_width, HEADLINE_LINES)
        )
        self.headline.setTextFormat(Qt.TextFormat.PlainText)
        self.headline.setFont(font)
        if not received:
            set_role(self.headline, "muted")
        texts = QVBoxLayout()
        texts.setSpacing(0)
        if topic:
            self.topic = set_role(ElidedLabel(), "small")
            self.topic.setFont(_font(SMALL_FONT_PX))
            self.topic.setText(topic)
            texts.addWidget(self.topic)
        texts.addWidget(self.headline)
        chevron = QLabel()
        chevron.setFixedSize(ROW_CHEVRON, ROW_CHEVRON)
        self.more = more
        if more:
            color = theme.color("accent" if selected else "muted")
            chevron.setPixmap(symbol_pixmap("chevron", color, ROW_CHEVRON))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(ROW_MARGIN_LEFT, 8, ROW_MARGIN_RIGHT, 8)
        layout.setSpacing(ROW_SPACING)
        layout.addStrut(ROW_HEIGHT - 16)
        layout.addLayout(texts, stretch=1)
        layout.addWidget(chevron)


class JournalinePage(QWidget):
    """The Journaline of the station playing in two columns (see the module)."""

    def __init__(self, model: JournalineModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._theme = THEMES[DARK]
        self._station = ""
        # The menus from the point of entry to the one in the left column, and the
        # entry chosen in it (None: the first, once the menu is there).
        self._path: list[int] = [model.entry]
        self._selected: int | None = None
        # The version of the chosen page on display: a text message changed by the
        # station is only replaced on request.
        self._shown: JmlObject | None = None
        # Changes while hidden are shown when the page shows again.
        self._stale = False
        self._left: QScrollArea | None = None
        self._right: QWidget | None = None
        self._right_scroll: QScrollArea | None = None
        self.rows: list[_Row] = []  # of the left column
        self.right_rows: list[_Row] = []  # of a menu in the right column
        self.link_rows: list[TapWidget] = []  # links to other pages, below a message
        self.qr_rows: list[TapWidget] = []  # links with a QR code, below a message
        self._qr_sheet: QrSheet | None = None
        self.update_button: QPushButton | None = None

        title = set_role(QLabel("Journaline"), "large")
        self._summary = set_role(ElidedLabel(), "muted")
        self.back_button = QPushButton(tr("Back"))
        header = QHBoxLayout()
        header.setSpacing(16)
        header.addWidget(title)
        header.addWidget(self._summary, stretch=1)
        header.addWidget(self.back_button)

        self.up_button = QPushButton()
        self.up_button.setIconSize(BUTTON_ICON)
        self.up_button.setToolTip(tr("Up one level"))
        self.up_button.clicked.connect(self.go_up)
        self.path_label = set_role(ElidedLabel(), "large")
        path = QHBoxLayout()
        path.setSpacing(12)
        path.addWidget(self.up_button)
        path.addWidget(self.path_label, stretch=1)

        self._columns = QHBoxLayout()
        self._columns.setSpacing(COLUMN_GAP)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)
        layout.addLayout(header)
        layout.addLayout(path)
        layout.addLayout(self._columns, stretch=1)

        model.changed.connect(self._on_changed)
        self.apply_theme(self._theme)

    # --- Outside ---------------------------------------------------------------------

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.up_button.setIcon(symbol_icon("arrow_up", theme.color("text"), BUTTON_ICON))
        self._render(keep_scroll=True)

    def set_station(self, name: str) -> None:
        self._station = name
        self._show_summary()

    @property
    def path(self) -> tuple[int, ...]:
        return tuple(self._path)

    @property
    def selected(self) -> int | None:
        return self._selected

    @property
    def shown(self) -> JmlObject | None:
        """The page on display in the right column, as shown."""
        return self._shown

    def showEvent(self, event: QShowEvent) -> None:
        if self._stale:
            self._render(keep_scroll=True)
        super().showEvent(event)

    def hideEvent(self, event: QHideEvent) -> None:
        """A code still shown belongs to the page: it goes with it (another station)."""
        if self._qr_sheet is not None and self._qr_sheet.isVisible():
            self._qr_sheet.dismiss()
        super().hideEvent(event)

    def qr_sheet(self) -> QrSheet:
        """The QR code of a link over the whole window, made on first use."""
        if self._qr_sheet is None:
            self._qr_sheet = QrSheet(self.window())
        return self._qr_sheet

    def show_code(self, link: LinkTarget) -> None:
        """The QR code of a link below a message, named as the station names it."""
        payload = link_payload(link)
        if payload is not None:
            title = one_line(link.label) or link.address.strip()
            self.qr_sheet().open(title, link.address.strip(), payload)

    # --- Navigation ------------------------------------------------------------------

    def choose(self, object_id: int) -> None:
        """An entry of the menu on the left: shown on the right."""
        self._selected = object_id
        self._render(keep_scroll=True, right_scroll=False)

    def enter(self, object_id: int) -> None:
        """An entry of the menu on the right: that menu moves to the left, the entry is
        shown on the right."""
        if self._selected is not None:
            self._path.append(self._selected)
        self._selected = object_id
        self._render(keep_scroll=False)

    def follow(self, object_id: int) -> None:
        """A link below a message to another page: a menu on the path is returned to,
        anything else shown on the right."""
        if object_id in self._path:
            index = self._path.index(object_id)
            if index > 0:
                self._selected = object_id
                del self._path[index:]
            else:
                self._selected = None
                del self._path[1:]
        else:
            self._selected = object_id
        self._render(keep_scroll=False)

    def go_up(self) -> None:
        """Back one level: the menu on the left moves to the right, chosen in its own."""
        if len(self._path) > 1:
            self._selected = self._path.pop()
            self._render(keep_scroll=False)

    def show_new_version(self) -> None:
        self._render(keep_scroll=False, right_scroll=False)

    # --- Following the model ---------------------------------------------------------

    def _on_changed(self, ids: frozenset[int] | None) -> None:
        model = self._model
        self._show_summary()
        if ids is None:
            if self._path[0] != model.entry or not model.objects():
                # Another station, or the point of entry FIG 0/13 names: from the start.
                self._path = [model.entry]
                self._selected = None
                self._shown = None
        elif not ids & self._in_view():
            return
        if not self.isVisible():
            self._stale = True
            return
        self._render(keep_scroll=True, keep_shown=self._reading())

    def _reading(self) -> bool:
        """Whether a text message is shown that the station still sends: a new version
        of it waits for "show" (clause 4.2.2.1)."""
        shown = self._shown
        return (
            shown is not None
            and shown.object_type is ObjectType.PLAIN_TEXT
            and shown.object_id == self._selected
            and self._model.get(shown.object_id) is not None
        )

    def _in_view(self) -> set[int]:
        """The pages whose change shows: the path, the entries of both columns, the
        page chosen and the pages its links lead to."""
        ids = set(self._path)
        for menu_id in (self._path[-1], self._selected):
            menu = self._model.get(menu_id) if menu_id is not None else None
            if menu is not None and menu.object_type is ObjectType.MENU:
                ids.update(item.target for item in menu.menu_items)
        if self._selected is not None:
            ids.add(self._selected)
        if self._shown is not None:
            ids.update(link.target for link in self._shown.links if link.target is not None)
        return ids

    # --- Display ---------------------------------------------------------------------

    def _show_summary(self) -> None:
        count = len(self._model.objects())
        parts = [part for part in (self._station, page_count_text(count) if count else "") if part]
        self._summary.setText(" · ".join(parts))

    def _title(self, object_id: int) -> str:
        jml_object = self._model.get(object_id)
        return one_line(jml_object.title) if jml_object and jml_object.title else "…"

    def _path_names(self) -> list[str]:
        """The point of entry by its title, every menu after it as its menu names it."""
        names = [self._title(self._path[0])]
        for parent, child in zip(self._path, self._path[1:]):
            menu = self._model.get(parent)
            item = next((i for i in menu.menu_items if i.target == child), None) if menu else None
            text = one_line(item.text) if item is not None else ""
            names.append(text or self._title(child))
        return names

    def _render(
        self, *, keep_scroll: bool, right_scroll: bool | None = None, keep_shown: bool = False
    ) -> None:
        """Builds both columns afresh; with `keep_scroll` each where it was."""
        self._stale = False
        self._show_summary()
        self.path_label.setText(" › ".join(self._path_names()))
        self.up_button.setEnabled(len(self._path) > 1)
        if right_scroll is None:
            right_scroll = keep_scroll

        left_menu = self._model.get(self._path[-1])
        if self._selected is None and left_menu is not None and left_menu.menu_items:
            self._selected = left_menu.menu_items[0].target
        if not keep_shown:
            self._shown = self._model.get(self._selected) if self._selected is not None else None
        self.update_button = None

        left_position = self._left.verticalScrollBar().value() if self._left else 0
        right_position = self._right_scroll.verticalScrollBar().value() if self._right_scroll else 0
        for column in (self._left, self._right):
            if column is not None:
                # Hidden at once: deleted later, a removed column would still be painted.
                column.hide()
                self._columns.removeWidget(column)
                column.deleteLater()

        self._left, self.rows = self._menu_column(left_menu, self._selected)
        self._left.setFixedWidth(LEFT_WIDTH)
        self._columns.addWidget(self._left)
        self._right, self._right_scroll = self._content_column()
        self._columns.addWidget(self._right, stretch=1)
        if keep_scroll:
            self._left.verticalScrollBar().setValue(left_position)
        if right_scroll and self._right_scroll is not None:
            self._right_scroll.verticalScrollBar().setValue(right_position)

    def _rows(
        self, menu: JmlObject, selected: int | None, text_width: int, on_tap
    ) -> tuple[QWidget, list[_Row]]:
        rows_widget = set_role(QWidget(), "transparent")
        layout = QVBoxLayout(rows_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        rows = []
        for item in menu.menu_items:
            target = self._model.get(item.target)
            row = _Row(
                item.target,
                one_line(item.text) or self._title(item.target),
                more=target is not None and has_more(target),
                received=target is not None,
                selected=item.target == selected,
                text_width=text_width,
                theme=self._theme,
            )
            row.tapped.connect(lambda target=item.target: on_tap(target))
            layout.addWidget(row)
            rows.append(row)
        layout.addStretch(1)
        return rows_widget, rows

    def _missing(self) -> QLabel:
        text = tr("Receiving …") if self._model.announced else tr("Not received yet")
        label = set_role(QLabel(text), "muted")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def _menu_column(
        self, menu: JmlObject | None, selected: int | None
    ) -> tuple[QScrollArea, list[_Row]]:
        if menu is None or menu.object_type is not ObjectType.MENU:
            holder = set_role(QWidget(), "transparent")
            QVBoxLayout(holder).addWidget(self._missing())
            return _scroll(holder), []
        rows_widget, rows = self._rows(menu, selected, LEFT_TEXT_WIDTH, self.choose)
        return _scroll(rows_widget), rows

    def _content_column(self) -> tuple[QWidget, QScrollArea | None]:
        panel = set_role(QWidget(), "panel")
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        self.right_rows = []
        self.link_rows = []
        self.qr_rows = []
        shown = self._shown
        if self._selected is None:
            return panel, None
        if shown is None:
            outer.addWidget(self._missing())
            return panel, None
        if shown.object_type is ObjectType.MENU:
            rows_widget, self.right_rows = self._rows(shown, None, RIGHT_TEXT_WIDTH, self.enter)
            rows_widget.layout().setContentsMargins(
                PANEL_INSET, PANEL_INSET, PANEL_INSET, PANEL_INSET
            )
            area = _scroll(rows_widget)
            outer.addWidget(area)
            return panel, area
        area = _scroll(self._message(shown))
        outer.addWidget(area)
        return panel, area

    def _message(self, jml_object: JmlObject) -> QWidget:
        """A text message, a short message or a list, with its links below."""
        content = set_role(QWidget(), "transparent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        newest = self._model.get(jml_object.object_id)
        if newest is not None and newest != jml_object:
            self.update_button = set_role(QPushButton(tr("New version received - show")), "accent")
            self.update_button.clicked.connect(self.show_new_version)
            layout.addWidget(self.update_button)
        topic, headline = split_topic(one_line(jml_object.title))
        if topic:
            layout.addWidget(set_role(QLabel(topic), "small"))
        title = set_role(QLabel(headline), "headline")
        title.setTextFormat(Qt.TextFormat.PlainText)
        title.setWordWrap(True)
        layout.addWidget(title)
        received = self._model.received_at(jml_object.object_id)
        if received is not None:
            layout.addWidget(set_role(QLabel(received_text(received)), "small"))
        if jml_object.object_type is ObjectType.PLAIN_TEXT and jml_object.body:
            body = QLabel(_highlighted_html(jml_object.body, jml_object.body_highlights))
            body.setTextFormat(Qt.TextFormat.RichText)
            body.setWordWrap(True)
            layout.addWidget(body)
        if jml_object.object_type is ObjectType.LIST and jml_object.list_items:
            layout.addLayout(self._list(jml_object))
        if jml_object.links:
            layout.addSpacing(8)
            for link in jml_object.links:
                layout.addWidget(self._link_row(link))
        layout.addStretch(1)
        return content

    @staticmethod
    def _list(jml_object: JmlObject) -> QGridLayout:
        """The rows of a list, its columns aligned; rows with fewer columns padded
        (table 1, code 0x05). The last column takes what room is left."""
        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(8)
        width = max(len(row) for row in jml_object.list_items)
        for row, columns in enumerate(jml_object.list_items):
            for column, text in enumerate(columns):
                label = QLabel(one_line(text))
                label.setTextFormat(Qt.TextFormat.PlainText)
                label.setWordWrap(column == width - 1)
                grid.addWidget(label, row, column, alignment=Qt.AlignmentFlag.AlignTop)
        grid.setColumnStretch(width - 1, 1)
        return grid

    def _link_row(self, link: LinkTarget) -> QWidget:
        """A link below a message: to another page a row to tap; a web page or a number
        a row to tap for its QR code, marked as such; a DAB link as text."""
        if link.link_type is LinkType.JML_OBJECT and link.target is not None:
            text = one_line(link.label) or self._title(link.target)
            row = _Row(
                link.target,
                text,
                more=True,
                received=self._model.get(link.target) is not None,
                selected=False,
                text_width=RIGHT_TEXT_WIDTH,
                theme=self._theme,
            )
            row.tapped.connect(lambda target=link.target: self.follow(target))
            self.link_rows.append(row)
            return row
        coded = link_payload(link) is not None
        if coded:
            row = _PressRow()
            row.tapped.connect(lambda link=link: self.show_code(link))
            self.qr_rows.append(row)
        else:
            row = set_role(QWidget(), "transparent")
        layout = QHBoxLayout(row)
        if coded:
            # Room for the shade of a press around the symbols and the text.
            layout.setContentsMargins(8, 6, 8, 6)
        else:
            layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        icon = QLabel()
        icon.setFixedSize(LINK_ICON, LINK_ICON)
        symbol = "phone" if link.link_type in (LinkType.TELEPHONE, LinkType.SMS) else "web"
        icon.setPixmap(symbol_pixmap(symbol, self._theme.color("muted"), LINK_ICON))
        address = html.escape(link.address)
        text = address
        if link.label:
            muted = self._theme.muted
            text = f"{_html_text(link.label)}<br><span style='color: {muted}'>{address}</span>"
        label = QLabel(text)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setWordWrap(True)
        layout.addWidget(icon, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(label, stretch=1)
        if coded:
            mark = QLabel()
            mark.setFixedSize(LINK_ICON, LINK_ICON)
            mark.setPixmap(symbol_pixmap("qr", self._theme.color("muted"), LINK_ICON))
            layout.addWidget(mark, alignment=Qt.AlignmentFlag.AlignVCenter)
        return row
