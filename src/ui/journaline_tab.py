"""The "Journaline" tab: the station's text pages (ETSI TS 102 979)."""

from __future__ import annotations

import html
import time

from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QFontDatabase
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from dab.journaline import (
    ROOT_OBJECT_ID,
    Change,
    JmlObject,
    JournalineStore,
    LinkTarget,
    LinkType,
    NavigationHistory,
    ObjectType,
)
from i18n import format_day_month, tr, tr_noop, tr_plural

# Interval at which objects past their timeout are removed. Timeouts count minutes or
# quarter hours (clause 5.3.2.2); ten seconds are ample.
EXPIRY_TICK_MS = 10_000

# Links between objects are rendered as anchors of this scheme with the object ID.
_OBJECT_SCHEME = "jml"
# Link targets opened in the browser or mail program. Everything else a station might
# send (other schemes, telephone numbers, DAB URIs) is shown as text only.
_EXTERNAL_SCHEMES = ("http", "https", "mailto")

_TYPE_NAMES = {
    # Translators: the types of Journaline objects (ETSI TS 102 979, clause 4.2).
    ObjectType.MENU: tr_noop("Menu"),
    ObjectType.PLAIN_TEXT: tr_noop("Text message"),
    ObjectType.TITLE_ONLY: tr_noop("Short message"),
    ObjectType.LIST: tr_noop("List"),
}

_NOTHING_RECEIVED = tr_noop(
    "<h3>No Journaline data received yet.</h3>"
    "<p>Journaline is a text service with news, programme notes and contact details that "
    "a station transmits alongside the audio in the PAD field (ETSI TS 102 979). Only a "
    "few stations offer it. The pages arrive one after another in an endless loop; it "
    "takes a while until all of them are there.</p>"
)


def _object_label(object_id: int) -> str:
    """How an object ID is shown."""
    return f"0x{object_id:04X}"


def _object_name(object_id: int) -> str:
    """An object named by its ID, for links and menu items without a text."""
    return tr("Object {object}").format(object=_object_label(object_id))


def _html_text(text: str) -> str:
    """Escapes a text for the display; line breaks of the station are kept."""
    return html.escape(text).replace("\n", "<br>")


def _highlighted_html(text: str, highlights: tuple[tuple[int, int], ...]) -> str:
    """Like `_html_text`, with the highlighted passages in bold."""
    parts = []
    position = 0
    for start, end in highlights:
        parts.append(_html_text(text[position:start]))
        parts.append(f"<b>{_html_text(text[start:end])}</b>")
        position = end
    parts.append(_html_text(text[position:]))
    return "".join(parts)


def _single_line(text: str) -> str:
    return " ".join(text.split())


class JournalineTab(QWidget):
    """On the left all received objects, on the right the object on display with its
    links, above it the navigation.

    Owns the object store and the history path. The main window only feeds in decoded
    objects and clears everything on a change of service.
    """

    object_count_changed = pyqtSignal(int)  # number of valid objects in the store

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._store = JournalineStore()
        self._history = NavigationHistory()
        # The version on display. It can differ from the one in the store: an updated
        # text message is only shown on request (see `add_object`).
        self._shown: JmlObject | None = None

        self._back_button = QPushButton(tr("Back"))
        self._back_button.setToolTip(
            tr(
                "Back to the previous page. If there is none, to the main menu "
                "(ETSI TS 102 979, clause 4.3)."
            )
        )
        self._back_button.clicked.connect(self._on_back)
        self._home_button = QPushButton(tr("Main menu"))
        self._home_button.clicked.connect(self._on_home)
        self._update_home_tooltip()
        self._path_label = QLabel()
        self._path_label.setTextFormat(Qt.TextFormat.PlainText)
        self._path_label.setWordWrap(True)
        self._path_label.setToolTip(
            tr(
                "The path from the main menu to the page on display. A link to a page "
                "already on the path (such as back to the main menu) shortens it."
            )
        )

        self._list_heading = QLabel()
        self._object_list = QListWidget()
        self._object_list.setMinimumWidth(240)
        self._object_list.setToolTip(
            tr(
                "All objects received so far, ordered by their ID. Each page comes as an "
                "object of its own in an MSC data group of its own with CRC. A click shows "
                "the page, even if no menu links to it yet."
            )
        )
        self._object_list.itemClicked.connect(self._on_object_clicked)

        self._update_button = QPushButton(tr("New version received - show"))
        self._update_button.setToolTip(
            tr(
                "The station has changed the message on display. So that the text does not "
                "change while you read it, the new version only appears on a click (clause "
                "4.2.2.1). Menus, lists and short messages, by contrast, are renewed at once."
            )
        )
        self._update_button.clicked.connect(lambda: self._show_current(keep_scroll=False))
        self._update_button.setVisible(False)

        self._view = QTextBrowser()
        self._view.setOpenLinks(False)
        self._view.anchorClicked.connect(self._on_anchor_clicked)
        self._view.setToolTip(
            tr(
                "The page on display. Menu entries in square brackets point to pages not "
                "received yet; they appear as soon as the station broadcasts them the next "
                "time."
            )
        )

        self._info_label = QLabel("-")
        self._info_label.setWordWrap(True)
        self._info_label.setToolTip(
            tr(
                "Header of the object: ID, type, revision (counts every change of content, "
                "modulo 8) and the static flag (“static”: the same kind of content always "
                "comes under this ID). Compressed objects are packed with Deflate; then the "
                "transmitted and the unpacked size are given."
            )
        )

        navigation = QHBoxLayout()
        navigation.addWidget(self._back_button)
        navigation.addWidget(self._home_button)
        navigation.addWidget(self._path_label, stretch=1)

        list_side = QVBoxLayout()
        list_side.addWidget(self._list_heading)
        list_side.addWidget(self._object_list, stretch=1)

        page_side = QVBoxLayout()
        page_side.addWidget(self._update_button)
        page_side.addWidget(self._view, stretch=1)
        page_side.addWidget(self._info_label)

        pages = QHBoxLayout()
        pages.addLayout(list_side, stretch=1)
        pages.addLayout(page_side, stretch=3)

        layout = QVBoxLayout()
        layout.addLayout(navigation)
        layout.addLayout(pages, stretch=1)
        self.setLayout(layout)

        self._expiry_timer = QTimer(self)
        self._expiry_timer.setInterval(EXPIRY_TICK_MS)
        self._expiry_timer.timeout.connect(self._on_expiry_tick)
        self._expiry_timer.start()

        self._refresh_object_list()
        self._show_current(keep_scroll=False)

    # --- Input from the main window ----------------------------------------------

    def reset(self) -> None:
        """Forgets all objects; they belong to the previously selected service. The
        point of entry returns to the root object until FIG 0/13 names another."""
        self._store.clear()
        self._history = NavigationHistory()
        self._update_home_tooltip()
        self._refresh_object_list()
        self._show_current(keep_scroll=False)

    def set_entry_object(self, object_id: int) -> None:
        """Takes the point of entry from the user application data in FIG 0/13 (clause
        8.1.2); usually the root object 0x0000."""
        if object_id == self._history.entry:
            return
        self._history.set_entry(object_id)
        self._update_home_tooltip()
        self._show_current(keep_scroll=False)

    def add_object(self, jml_object: JmlObject) -> None:
        """Takes a received object.

        How an update of the object on display appears depends on its type (clause
        4.2): menus, lists and short messages are renewed immediately at the same
        scroll position, a text message is only replaced on request.
        """
        change = self._store.add(jml_object, time.time())
        if change is Change.UNCHANGED:
            return
        self._refresh_object_list()
        if jml_object.object_id != self._history.current:
            # Menu entries whose target has just arrived lose their brackets.
            self._render(keep_scroll=True)
        elif (
            change is Change.UPDATED
            and self._shown is not None
            and self._shown.object_type is ObjectType.PLAIN_TEXT
        ):
            self._update_button.setVisible(True)
        else:
            self._show_current(keep_scroll=True)

    # --- Navigation -----------------------------------------------------------------

    def _on_back(self) -> None:
        self._history.back()
        self._show_current(keep_scroll=False)

    def _on_home(self) -> None:
        self._history.home()
        self._show_current(keep_scroll=False)

    def _on_object_clicked(self, item: QListWidgetItem) -> None:
        self._follow(item.data(Qt.ItemDataRole.UserRole))

    def _on_anchor_clicked(self, url: QUrl) -> None:
        if url.scheme() == _OBJECT_SCHEME:
            try:
                self._follow(int(url.path()))
            except ValueError:
                pass
        elif url.scheme() in _EXTERNAL_SCHEMES:
            QDesktopServices.openUrl(url)

    def _follow(self, object_id: int) -> None:
        self._history.follow(object_id)
        self._show_current(keep_scroll=False)

    def _update_home_tooltip(self) -> None:
        entry = self._history.entry
        if entry == ROOT_OBJECT_ID:
            tooltip = tr("To the entry point of the service, object {object}.")
        else:
            tooltip = tr("To the entry point of the service, object {object}, as FIG 0/13 says.")
        self._home_button.setToolTip(tooltip.format(object=_object_label(entry)))

    def _on_expiry_tick(self) -> None:
        """Clause 5.3.2.2: an object past its timeout is no longer shown."""
        expired = self._store.expire(time.time())
        if not expired:
            return
        self._refresh_object_list()
        if self._history.current in expired:
            self._show_current(keep_scroll=False)
        else:
            self._render(keep_scroll=True)

    # --- Display --------------------------------------------------------------------

    def _show_current(self, *, keep_scroll: bool) -> None:
        """Shows the object at the end of the history path in its newest version."""
        self._shown = self._store.get(self._history.current, time.time())
        self._update_button.setVisible(False)
        self._render(keep_scroll=keep_scroll)

    def _render(self, *, keep_scroll: bool) -> None:
        scrollbar = self._view.verticalScrollBar()
        position = scrollbar.value()
        if self._shown is None:
            self._view.setHtml(self._missing_html())
            self._info_label.setText("-")
        else:
            self._view.setHtml(self._object_html(self._shown))
            self._info_label.setText(self._describe(self._shown))
        if keep_scroll:
            scrollbar.setValue(position)

        path = self._history.path
        self._path_label.setText(" › ".join(self._title_of(object_id) for object_id in path))
        self._back_button.setEnabled(path != (ROOT_OBJECT_ID,))
        self._home_button.setEnabled(len(path) > 1)
        self._select_in_list(self._history.current)

    def _title_of(self, object_id: int) -> str:
        jml_object = self._store.get(object_id, time.time())
        if jml_object is None or not jml_object.title:
            return _object_label(object_id)
        return _single_line(jml_object.title)

    def _missing_html(self) -> str:
        object_id = self._history.current
        if not len(self._store):
            return tr(_NOTHING_RECEIVED)
        if object_id == self._history.entry:
            return tr(
                "<h3>The main menu has not arrived yet.</h3>"
                "<p>Pages received so far are in the list on the left. The main menu "
                "(object {object}) appears here as soon as the station broadcasts it the "
                "next time.</p>"
            ).format(object=_object_label(object_id))
        return tr(
            "<h3>Object {object} is not available - please wait.</h3>"
            "<p>It has not been received yet or has expired. The station repeats its pages "
            "all the time; it appears here as soon as it comes the next time.</p>"
        ).format(object=_object_label(object_id))

    def _object_html(self, jml_object: JmlObject) -> str:
        title = _html_text(jml_object.title)
        if not title:
            untitled = html.escape(tr("untitled"))
            title = f"<i>{untitled}</i>"
        if jml_object.object_type is ObjectType.TITLE_ONLY:
            # Meant as a news ticker: the text alone, large (clause 4.2.2.2).
            parts = [f'<p align="center" style="font-size: 18pt;">{title}</p>']
        else:
            parts = [f"<h2>{title}</h2>"]

        if jml_object.object_type is ObjectType.MENU:
            parts.append(self._menu_html(jml_object))
        elif jml_object.object_type is ObjectType.PLAIN_TEXT:
            body = _highlighted_html(jml_object.body, jml_object.body_highlights)
            parts.append(f"<p>{body}</p>")
        elif jml_object.object_type is ObjectType.LIST:
            parts.append(self._list_html(jml_object))

        links = [self._link_html(link) for link in jml_object.links]
        if links:
            # Translators: heading of the links below a Journaline page.
            heading = html.escape(tr("See also:"))
            parts.append(f"<p><b>{heading}</b><br>" + "<br>".join(links) + "</p>")
        return "".join(parts)

    def _menu_html(self, jml_object: JmlObject) -> str:
        """Link items; those whose target has not arrived yet in brackets and grey
        (clause 7.2.3). They can be selected all the same."""
        now = time.time()
        rows = []
        for item in jml_object.menu_items:
            text = _html_text(item.text) or _object_name(item.target)
            if self._store.get(item.target, now) is None:
                anchor = (
                    f'<a href="{_OBJECT_SCHEME}:{item.target}" style="color: gray;">[{text}]</a>'
                )
            else:
                anchor = f'<a href="{_OBJECT_SCHEME}:{item.target}">{text}</a>'
            rows.append(f"<tr><td>›</td><td>{anchor}</td></tr>")
        return (
            '<table cellspacing="0" cellpadding="4" style="font-size: 13pt;">'
            f"{''.join(rows)}</table>"
        )

    @staticmethod
    def _list_html(jml_object: JmlObject) -> str:
        """List items as a table: columns start at the same position in every row, and
        rows with fewer columns are padded with empty ones (table 1, code 0x05)."""
        width = max((len(row) for row in jml_object.list_items), default=0)
        rows = []
        for row in jml_object.list_items:
            cells = [*row, *([""] * (width - len(row)))]
            rows.append(
                "<tr>" + "".join(f"<td>{_html_text(cell)}</td>" for cell in cells) + "</tr>"
            )
        # A monospaced font, as the standard recommends for lists; the generic CSS name
        # "monospace" is not resolved reliably by Qt.
        family = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()
        return (
            f'<table cellspacing="0" cellpadding="3" style="font-family: \'{family}\';">'
            f"{''.join(rows)}</table>"
        )

    @staticmethod
    def _link_html(link: LinkTarget) -> str:
        """A general link target (clause 5.3.2.3): clickable where it leads to another
        page, the web or an e-mail address, otherwise as text."""
        label = _html_text(link.label)
        address = html.escape(link.address)
        if link.link_type is LinkType.JML_OBJECT:
            text = label or _object_name(link.target)
            return f'<a href="{_OBJECT_SCHEME}:{link.target}">{text}</a>'
        if link.link_type is LinkType.URL:
            if QUrl(link.address).scheme().lower() in _EXTERNAL_SCHEMES:
                anchor = f'<a href="{html.escape(link.address, quote=True)}">{address}</a>'
            else:
                anchor = address
            return f"{label}: {anchor}" if label else anchor
        if link.link_type is LinkType.DAB_URI:
            # Translators: a Journaline link to a DAB service or object (a DAB URI).
            kind = tr("DAB link")
        elif link.link_type is LinkType.TELEPHONE:
            kind = tr("Phone")
        else:
            kind = "SMS"
        return f"{label} ({kind}: {address})" if label else f"{kind}: {address}"

    def _describe(self, jml_object: JmlObject) -> str:
        """Header data, size and validity of the object on display."""
        parts = [
            _object_name(jml_object.object_id),
            tr(_TYPE_NAMES[jml_object.object_type]),
            tr("Revision {revision}").format(revision=jml_object.revision),
            # Translators: the static flag of a Journaline object: the same kind of
            # content always comes under its ID, or not.
            tr("static") if jml_object.static else tr("changing"),
        ]
        if jml_object.compressed:
            parts.append(
                tr(
                    "Deflate-compressed: {transmitted} bytes transmitted, {size} bytes of "
                    "content unpacked"
                ).format(transmitted=jml_object.transmitted_size, size=jml_object.content_size)
            )
        else:
            size = jml_object.transmitted_size
            parts.append(tr_plural("{size} byte", "{size} bytes", size).format(size=size))
        received = self._store.received_at(jml_object.object_id)
        if received is not None:
            moment = time.strftime("%H:%M:%S", time.localtime(received))
            parts.append(tr("received {time}").format(time=moment))
        expires = self._store.expires_at(jml_object.object_id)
        if expires is not None:
            moment = time.localtime(expires)
            parts.append(
                tr("valid until {day} {time}").format(
                    day=format_day_month(moment), time=time.strftime("%H:%M", moment)
                )
            )
        return " · ".join(parts)

    def _refresh_object_list(self) -> None:
        """Rebuilds the list. It changes with every newly arriving object, so the
        scroll position is kept: the list must not jump while the user browses it."""
        objects = self._store.objects(time.time())
        self._list_heading.setText(tr("Received objects ({count}):").format(count=len(objects)))
        scrollbar = self._object_list.verticalScrollBar()
        position = scrollbar.value()
        self._object_list.clear()
        for jml_object in objects:
            title = _single_line(jml_object.title) or tr("untitled")
            kind = tr(_TYPE_NAMES[jml_object.object_type])
            item = QListWidgetItem(f"{_object_label(jml_object.object_id)}  {kind}: {title}")
            item.setData(Qt.ItemDataRole.UserRole, jml_object.object_id)
            self._object_list.addItem(item)
        self._select_in_list(self._history.current)
        scrollbar.setValue(position)
        self.object_count_changed.emit(len(objects))

    def _select_in_list(self, object_id: int) -> None:
        self._object_list.clearSelection()
        for row in range(self._object_list.count()):
            item = self._object_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == object_id:
                self._object_list.setCurrentItem(item)
                return
