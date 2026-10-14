"""The "Programme guide" menu of the services page, on two pages to swipe through
(`ui.touch.pager`): the guide itself, and the MOT carousel it comes in.

The guide: on the left the services of all guides by channel, on the right the
schedule of the one chosen, day by day. A tap on a programme opens its description,
"Listen" plays the service.

The carousel: the files its directory lists (the service information, a schedule per
service and day, logos), each with how many of its segments are in, and beneath them
how the data groups and packets were checked, as the pages "RTCM stream" and
"TPEG stream" show for their streams. Everything comes from `ui.guide_model.GuideModel`.
"""

from __future__ import annotations

import time

from PyQt6.QtCore import QModelIndex, QPointF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QIcon, QPainter, QPixmap, QShowEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QScroller,
    QSizePolicy,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from dab.combined_guide import GuideEntry, GuideGroup
from dab.mot import CONTENT_TYPE_IMAGE, IMAGE_SUBTYPES
from dab.mot_directory import MotCarouselStatus, MotObjectStatus
from dab.spi import (
    CONTENT_TYPE_SPI,
    SUBTYPE_GROUP_INFORMATION,
    SUBTYPE_PROGRAMME_INFORMATION,
    SUBTYPE_SERVICE_INFORMATION,
    Programme,
)
from dab.timecode import ReferenceClock
from dsp.packet import PacketStatistics
from i18n import format_day_month, tr, tr_noop, tr_plural, weekday
from ui.guide_model import GuideModel
from ui.receiver_controller import ReceiverController, ReceptionState
from ui.tables import fill_table
from ui.touch.icons import symbol_icon, symbol_pixmap
from ui.touch.pager import Pager
from ui.touch.pages import Page, SwitchedOffNote, pixel_font, touch_table
from ui.touch.stations import placeholder_pixmap
from ui.touch.theme import DARK, EDGE_FADE_WIDTH, FONT_PX, LARGE_FONT_PX, THEMES, Theme
from ui.touch.widgets import EdgeFades, TapWidget, repolish, set_role

# The pages in the order of swiping.
SCHEDULE = "schedule"
CAROUSEL = "carousel"
PAGES = (SCHEDULE, CAROUSEL)

LIST_WIDTH = 360
LIST_LOGO = 40
# The calendar at the right of a service whose schedule the guide carries.
SCHEDULE_MARK = 24
# The arrow at the right of a programme whose description opens with a tap.
EXPAND_MARK = 24
HEADER_LOGO = 64
# The programme running is marked anew this often while the guide is on display.
RUNNING_REFRESH_MS = 30_000
# Above the programme running, the end of the one before it stays in view.
RUNNING_TOP_MARGIN = 60
# The longest time left of a programme running, whose text is the widest (see
# `_ProgrammeRow.set_now`).
RUNNING_WIDEST_S = 23 * 3600 + 59 * 60

# What a list entry stands for: (channel or None, SId, can be tuned to).
_ENTRY_ROLE = Qt.ItemDataRole.UserRole
# Whether the guide carries a schedule of the service (for the calendar).
_SCHEDULE_ROLE = Qt.ItemDataRole.UserRole + 1

_EMPTY = tr_noop(
    "No programme guide stored yet.\n\n"
    "If an ensemble received carries one (such as EPG Deutschland on 5C), it is "
    "received along with it. The first schedules take a few minutes to arrive."
)
_NO_SPI = tr_noop("No programme guide in the ensemble.")
# What the SPI objects of the carousel are (ETSI TS 102 371, table 11).
_SPI_CONTENT = {
    SUBTYPE_SERVICE_INFORMATION: tr_noop("Service Information (SI)"),
    SUBTYPE_PROGRAMME_INFORMATION: tr_noop("Programme Information (PI)"),
    SUBTYPE_GROUP_INFORMATION: tr_noop("Group Information (GI)"),
}


def day_key(timestamp: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(timestamp))


def day_title(key: str, now: float) -> str:
    """ "Today", "Tomorrow", else e.g. "Wed 7 Oct"."""
    if key == day_key(now):
        return tr("Today")
    if key == day_key(now + 24 * 3600):
        return tr("Tomorrow")
    moment = time.strptime(key, "%Y-%m-%d")
    return f"{weekday(moment, short=True)} {format_day_month(moment)}"


def duration_text(seconds: int) -> str:
    minutes = round(seconds / 60)
    if minutes < 60:
        return tr("{minutes} min").format(minutes=minutes)
    hours, minutes = divmod(minutes, 60)
    if minutes:
        return tr("{hours} h {minutes} min").format(hours=hours, minutes=minutes)
    return tr("{hours} h").format(hours=hours)


def running_text(seconds_left: int) -> str:
    """The time left of a programme running: "running, 30 min left"."""
    return tr("running, {duration} left").format(duration=duration_text(seconds_left))


def group_title(group: GuideGroup) -> str:
    if group.eid is not None:
        return tr("Ensemble {eid}, on no known channel").format(eid=f"{group.eid:04X}")
    if group.channel is None:
        return tr("Manual frequency")
    return f"{group.channel} · {group.label}" if group.label else group.channel


def content_label(content_type: int, subtype: int) -> str:
    """What a file of the carousel holds, by its MOT content type."""
    if content_type == CONTENT_TYPE_SPI:
        if subtype in _SPI_CONTENT:
            return tr(_SPI_CONTENT[subtype])
        return tr("SPI type {subtype}").format(subtype=subtype)
    if content_type == CONTENT_TYPE_IMAGE:
        if subtype in IMAGE_SUBTYPES:
            # Translators: {format} is GIF, JPEG, BMP or PNG.
            return tr("Image {format}").format(format=IMAGE_SUBTYPES[subtype])
        return tr("Image type {subtype}").format(subtype=subtype)
    return tr("Type {content_type}/{subtype}").format(content_type=content_type, subtype=subtype)


def size_text(size: int) -> str:
    if size < 1000:
        return tr_plural("{size} byte", "{size} bytes", size).format(size=size)
    return f"{size / 1000:.1f} kB"


def files_text(count: int) -> str:
    return tr_plural("{count} file", "{count} files", count).format(count=count)


def object_row(status: MotObjectStatus) -> tuple[str, str, str, str, str]:
    """A file of the carousel: name, content, size, segments in, how far it is."""
    name = status.content_name or tr("Transport ID {transport_id}").format(
        transport_id=f"{status.transport_id:04X}"
    )
    if status.total_segments is None:
        segments = str(status.segments)
    else:
        # Translators: segments of a file received, of all it has.
        segments = tr("{received} of {total}").format(
            received=status.segments, total=status.total_segments
        )
    if status.complete:
        # Translators: a file of the carousel received in full.
        received = tr("complete")
    elif status.received_bytes and status.body_size:
        received = f"{status.received_bytes / status.body_size * 100:.0f} %"
    else:
        received = "-"
    return (
        name,
        content_label(status.content_type, status.content_subtype),
        size_text(status.body_size),
        segments,
        received,
    )


def describe_carousel(
    status: MotCarouselStatus, packets: PacketStatistics | None, address: int | None
) -> str:
    """How far the carousel is in, and how the data groups and packets of the guide
    (those of its packet `address` in the subchannel's `packets`) were checked."""
    if status.directory_objects is None:
        text = tr(
            "Directory not received yet · {size} kB of segments collected from {files}"
        ).format(
            size=f"{status.unlisted_bytes / 1000:.0f}", files=files_text(status.unlisted_objects)
        )
        lines = [text]
    else:
        share = status.received_bytes / status.directory_bytes if status.directory_bytes else 1.0
        if status.directory_compressed:
            text = tr(
                "Directory {transport_id} (no. {number} since the start, compressed) · {files}, "
                "{received} of {total} kB ({share} %)"
            )
        else:
            text = tr(
                "Directory {transport_id} (no. {number} since the start) · {files}, "
                "{received} of {total} kB ({share} %)"
            )
        text = text.format(
            transport_id=f"{status.directory_transport_id:04X}",
            number=status.directories,
            files=files_text(status.directory_objects),
            received=f"{status.received_bytes / 1000:.0f}",
            total=f"{status.directory_bytes / 1000:.0f}",
            share=f"{share * 100:.0f}",
        )
        if status.carousel_period is not None:
            text += " · " + tr("Cycle time {minutes} min according to the directory").format(
                minutes=f"{status.carousel_period / 60:.0f}"
            )
        if status.unlisted_objects:
            text += " · " + tr("{files} not listed in the directory ({size} kB)").format(
                files=files_text(status.unlisted_objects),
                size=f"{status.unlisted_bytes / 1000:.0f}",
            )
        lines = [text]
    own = None if packets is None or address is None else packets.addresses.get(address)
    if own is not None:
        rate = own.useful_bytes * 8 / packets.seconds if packets.seconds else 0.0
        text = tr(
            "Data groups {data_groups}, {data_group_errors} with CRC error, {continuity_errors} "
            "interrupted · Packets {packets}, {crc_errors} with CRC error, {rate} kbit/s payload"
        ).format(
            data_groups=own.data_groups,
            data_group_errors=own.data_group_errors,
            continuity_errors=own.continuity_errors,
            packets=own.packets,
            crc_errors=own.crc_errors,
            rate=f"{rate / 1000:.1f}",
        )
        if packets.fec:
            if packets.fec_locked:
                fec = tr(
                    "Packet FEC in sync: {frames} frames, {corrected} bytes corrected, "
                    "{rows} rows uncorrectable"
                )
            else:
                fec = tr(
                    "Packet FEC not in sync: {frames} frames, {corrected} bytes corrected, "
                    "{rows} rows uncorrectable"
                )
            text += " · " + fec.format(
                frames=packets.fec_frames,
                corrected=packets.fec_corrected_bytes,
                rows=packets.fec_uncorrectable_rows,
            )
        else:
            text += " · " + tr("no packet FEC")
        lines.append(text)
    return "\n".join(lines)


class _ScheduleMarkDelegate(QStyledItemDelegate):
    """Draws the entry as usual and, for a service with a schedule, a calendar at its
    right edge."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.mark = QPixmap()

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        super().paint(painter, option, index)
        if not index.data(_SCHEDULE_ROLE) or self.mark.isNull():
            return
        rect = option.rect
        painter.drawPixmap(
            rect.right() - 12 - SCHEDULE_MARK,
            rect.center().y() - SCHEDULE_MARK // 2,
            self.mark,
        )


class _DaysArea(QScrollArea):
    """The row of days, swiped sideways by QScroller. While it is wider than its view,
    a finger coming down on it is its own, not the pager's (`claims_drag` of
    `ui.touch.pager`): a move sideways scrolls the days instead of turning the page."""

    def claims_drag(self, position: QPointF) -> bool:
        bar = self.horizontalScrollBar()
        return bar.maximum() > bar.minimum() and self.rect().contains(
            self.mapFromGlobal(position.toPoint())
        )


class _ProgrammeRow(TapWidget):
    """Time, title and duration; the description below opens with a tap. An arrow at the
    right shows which rows have one, pointing up while it is open."""

    opened = pyqtSignal()  # the description was opened by a tap

    def __init__(self, programme: Programme, theme: Theme) -> None:
        super().__init__()
        set_role(self, "programme")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)  # see `_Card`
        self.programme = programme
        self._theme = theme
        self.setProperty("running", False)
        self._time = set_role(
            QLabel(time.strftime("%H:%M", time.localtime(programme.start))), "large"
        )
        self._time.setFixedWidth(80)
        self._title = set_role(QLabel(programme.names.best), "large")
        self._title.setWordWrap(True)
        self._title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._duration = set_role(QLabel(), "muted")
        self._duration.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        # As wide as the longest text it shows, so the durations of all rows stay in line;
        # measured, as the font on the Pi is not the one on Windows.
        self._duration.setFixedWidth(
            QFontMetrics(pixel_font(FONT_PX)).horizontalAdvance(running_text(RUNNING_WIDEST_S))
        )
        self._description = set_role(QLabel(programme.description), "muted")
        self._description.setWordWrap(True)
        self._description.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._description.hide()
        # As high as the first line of the title, the arrow in its middle; without a
        # description empty, so the durations of all rows stay in line.
        self._mark = QLabel()
        self._mark.setFixedSize(EXPAND_MARK, QFontMetrics(pixel_font(LARGE_FONT_PX)).height())
        layout = QGridLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setHorizontalSpacing(16)
        layout.addWidget(self._time, 0, 0, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._title, 0, 1)
        layout.addWidget(self._duration, 0, 2)
        layout.addWidget(self._mark, 0, 3, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._description, 1, 1, 1, 3)
        layout.setColumnStretch(1, 1)
        self._show_mark()
        self.tapped.connect(self._toggle)

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        self._show_mark()

    def _show_mark(self) -> None:
        if not self.programme.description:
            return
        color = self._theme.color("accent" if self.property("running") else "muted")
        symbol = "collapse" if self.is_open else "expand"
        self._mark.setPixmap(symbol_pixmap(symbol, color, EXPAND_MARK))

    def set_now(self, now: float) -> bool:
        """Marks the programme if it is running; True if it is."""
        programme = self.programme
        running = programme.start <= now < programme.end
        if running:
            self._duration.setText(running_text(round(programme.end - now)))
        else:
            self._duration.setText(duration_text(programme.duration))
        if self.property("running") != running:
            self.setProperty("running", running)
            repolish(self)
            for label in (self._time, self._title, self._duration, self._description):
                repolish(label)
            self._show_mark()
        return running

    @property
    def is_open(self) -> bool:
        return self._description.isVisibleTo(self)

    def set_open(self, open_: bool) -> None:
        self._description.setVisible(open_)
        self._show_mark()

    def _toggle(self) -> None:
        if self.programme.description:
            opening = not self.is_open
            self.set_open(opening)
            if opening:
                self.opened.emit()


class _SchedulePage(QWidget):
    """The guide: the services on the left, the schedule of the one chosen on the right.
    It keeps itself up to date from the model's signals; the pager scrolls its lists."""

    def __init__(
        self,
        controller: ReceiverController,
        model: GuideModel,
        clock: ReferenceClock,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._model = model
        self._clock = clock
        self._theme = THEMES[DARK]
        self._groups: list[GuideGroup] | None = None
        self._sid: int | None = None
        self._chosen_by_hand = False
        self._day: str | None = None
        self._shown_days: list[str] = []
        self._shown_programmes: tuple[int | None, str | None, list[Programme]] = (None, None, [])
        self._rows: list[_ProgrammeRow] = []
        # The row to scroll to once the new rows are laid out (see `_scroll_to_pending`).
        self._pending_scroll: _ProgrammeRow | None = None
        self._dirty = True
        # The next refresh scrolls to the programme running (see `show_running`).
        self._to_running = False

        self._list = QListWidget()
        self._list.setFixedWidth(LIST_WIDTH)
        self._list.setIconSize(QSize(LIST_LOGO, LIST_LOGO))
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._list.itemClicked.connect(self._on_item_clicked)
        self._mark_delegate = _ScheduleMarkDelegate(self._list)
        self._list.setItemDelegate(self._mark_delegate)

        self._logo = QLabel()
        self._logo.setFixedSize(HEADER_LOGO, HEADER_LOGO)
        self._logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name = set_role(QLabel(), "title")
        self._where = set_role(QLabel(), "muted")
        # Translators: the button that tunes to the service whose schedule is shown.
        self._play = QPushButton(tr("Listen"))
        self._play.setIconSize(QSize(28, 28))
        self._play.clicked.connect(self._on_play_clicked)
        names = QVBoxLayout()
        names.setSpacing(2)
        names.addWidget(self._name)
        names.addWidget(self._where)
        header = QHBoxLayout()
        header.setSpacing(16)
        header.addWidget(self._logo)
        header.addLayout(names, stretch=1)
        header.addWidget(self._play)

        # The days in a row that keeps its natural width and is swiped sideways, as
        # the ensemble filter of the station page (see there why not resizable).
        self._day_row = QWidget()
        self._day_layout = QHBoxLayout(self._day_row)
        self._day_layout.setContentsMargins(0, 0, 0, 0)
        self._day_layout.setSpacing(8)
        self._day_group = QButtonGroup(self)
        self._day_group.setExclusive(True)
        self._days_area = _DaysArea()
        self._days_area.setWidget(self._day_row)
        self._days_area.setWidgetResizable(False)
        self._days_area.setFixedHeight(64)
        self._days_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        QScroller.grabGesture(
            self._days_area.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture
        )
        # Days may end right at the edge with more beyond: the fade shows there are.
        self._day_fades = EdgeFades(self._days_area)

        self._row_layout = QVBoxLayout()
        self._row_layout.setContentsMargins(0, 0, 0, 0)
        self._row_layout.setSpacing(4)
        self._row_layout.addStretch()
        rows = QWidget()
        rows.setLayout(self._row_layout)
        self._area = QScrollArea()
        self._area.setWidget(rows)
        self._area.setWidgetResizable(True)
        self._area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._area.verticalScrollBar().rangeChanged.connect(self._scroll_to_pending)
        self._no_schedule = set_role(QLabel(), "muted")
        self._no_schedule.setWordWrap(True)
        self._no_schedule.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        schedule = QVBoxLayout()
        schedule.setSpacing(12)
        schedule.addLayout(header)
        schedule.addWidget(self._days_area)
        schedule.addWidget(self._area, stretch=1)
        schedule.addWidget(self._no_schedule, stretch=1)
        self._schedule = QWidget()
        self._schedule.setLayout(schedule)

        columns = QHBoxLayout()
        columns.setSpacing(16)
        columns.addWidget(self._list)
        columns.addWidget(self._schedule, stretch=1)
        self._columns = QWidget()
        self._columns.setLayout(columns)
        self._empty = set_role(QLabel(tr(_EMPTY)), "large")
        self._empty.setWordWrap(True)
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status = set_role(QLabel(), "small")
        self._status.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._columns, stretch=1)
        layout.addWidget(self._empty, stretch=1)
        layout.addWidget(self._status)

        self._running_timer = QTimer(self)
        self._running_timer.setInterval(RUNNING_REFRESH_MS)
        self._running_timer.timeout.connect(self._mark_running)
        self._running_timer.start()

        model.changed.connect(self._on_model_changed)
        model.carousel_changed.connect(self._update_status)
        model.decoding_changed.connect(self._update_status)
        controller.state_changed.connect(self._update_play_button)

    # --- Showing ----------------------------------------------------------------------

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        self._play.setIcon(symbol_icon("play", theme.color("text"), QSize(28, 28)))
        self._mark_delegate.mark = symbol_pixmap("calendar", theme.color("accent"), SCHEDULE_MARK)
        for row in self._rows:
            row.set_theme(theme)
        self._groups = None  # the placeholders take the scheme's colours
        self._dirty = True
        if self.isVisible():
            self.refresh()

    def render(self) -> None:
        """Nothing at the plot rate: the page follows the model's signals."""

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if self._dirty:
            self.refresh()

    def refresh(self) -> None:
        """Reads the guides (the first time from the store) and shows them."""
        self._dirty = False
        groups = self._model.groups()
        empty = not any(group.entries for group in groups)
        self._columns.setVisible(not empty)
        self._empty.setVisible(empty)
        if groups != self._groups:
            self._groups = groups
            self._fill_list(groups)
        if self._find(self._sid) is None or not self._chosen_by_hand:
            self._sid = self._default_sid()
        self._mark_chosen()
        self._show_service()
        self._update_status()
        if self._to_running:
            self._to_running = False
            # The guide follows the station playing on, as when it opens by itself,
            # unless it would choose another service then.
            self._chosen_by_hand = self._default_sid() != self._sid
            running = self._mark_running()
            if running is not None and self._pending_scroll is None:
                # The rows stayed as they were and have their places: scrolled at once.
                self._pending_scroll = running
                self._scroll_to_pending(0, self._area.verticalScrollBar().maximum())

    def show_running(self, sid: int) -> None:
        """Chooses the service and today, the programme running at the top (the guide
        card of the radio page leads here). A hidden view does so when it is shown: rows
        laid out while it is hidden have no places yet to scroll to."""
        self._sid = sid
        self._chosen_by_hand = True
        self._day = None
        self._to_running = True
        if self.isVisible():
            self.refresh()
        else:
            self._dirty = True

    def _now(self) -> float:
        return self._clock.now() or time.time()

    def _fill_list(self, groups: list[GuideGroup]) -> None:
        """Services with a schedule stand out: bold and with a calendar; those the
        guide only names (often with their logo) are muted."""
        self._list.clear()
        muted = QColor(self._theme.muted)
        bold = QFont(self._list.font())
        bold.setPixelSize(FONT_PX)
        bold.setBold(True)
        for group in groups:
            header = QListWidgetItem(group_title(group))
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            self._list.addItem(header)
            tunable = group.channel is not None and group.eid is None
            for entry in group.entries:
                item = QListWidgetItem(entry.name)
                item.setIcon(QIcon(self._logo_pixmap(entry, LIST_LOGO)))
                item.setData(_ENTRY_ROLE, (group.channel, entry.sid, tunable and entry.listed))
                item.setData(_SCHEDULE_ROLE, entry.has_schedule)
                if entry.has_schedule:
                    item.setFont(bold)
                else:
                    item.setForeground(muted)
                self._list.addItem(item)

    def _logo_pixmap(self, entry: GuideEntry, size: int) -> QPixmap:
        if entry.logo is not None:
            pixmap = QPixmap()
            if pixmap.loadFromData(entry.logo):
                return pixmap.scaled(
                    size,
                    size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
        return placeholder_pixmap(
            entry.name, size, self._theme.color("pressed"), self._theme.color("text")
        )

    def _find(self, sid: int | None) -> tuple[GuideGroup, GuideEntry] | None:
        """The service in the group of the channel playing if it is there, else in the
        first group listing it."""
        if sid is None or not self._groups:
            return None
        found = [(g, e) for g in self._groups for e in g.entries if e.sid == sid]
        if not found:
            return None
        playing = self._controller.channel
        return next(((g, e) for g, e in found if g.channel == playing), found[0])

    def _default_sid(self) -> int | None:
        """The service playing if a guide has its schedule, else the first service
        with one (many guides carry only logos for most of their services)."""
        playing = self._find(self._controller.selected_sid)
        if playing is not None and playing[1].has_schedule:
            return self._controller.selected_sid
        entries = [entry for group in self._groups or [] for entry in group.entries]
        candidates = [entry for entry in entries if entry.has_schedule] or entries
        return candidates[0].sid if candidates else None

    def _mark_chosen(self) -> None:
        found = self._find(self._sid)
        for row in range(self._list.count()):
            item = self._list.item(row)
            data = item.data(_ENTRY_ROLE)
            if found is not None and data is not None and data[:2] == (found[0].channel, self._sid):
                self._list.setCurrentItem(item)
                return

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        data = item.data(_ENTRY_ROLE)
        if data is None:
            return
        self._sid = data[1]
        self._chosen_by_hand = True
        self._day = None
        self._show_service()

    def _show_service(self) -> None:
        found = self._find(self._sid)
        if found is None:
            self._name.setText("")
            self._where.setText("")
            self._logo.clear()
            self._fill_rows(None, [])
            return
        group, entry = found
        self._name.setText(entry.name)
        self._where.setText(group_title(group))
        self._logo.setPixmap(self._logo_pixmap(entry, HEADER_LOGO))
        self._update_play_button()
        now = self._now()
        programmes = self._model.programmes(entry.sid)
        today = day_key(now)
        days = sorted({day_key(p.start) for p in programmes})
        shown_days = [day for day in days if day >= today] or days
        if self._day not in shown_days:
            self._day = today if today in shown_days else (shown_days[0] if shown_days else None)
        self._fill_days(shown_days, now)
        on_day = [p for p in programmes if day_key(p.start) == self._day]
        self._fill_rows(entry.sid, on_day)
        if not programmes:
            self._no_schedule.setText(
                tr(
                    "The programme guide carries no schedule for this station, or it has not "
                    "been received yet."
                )
            )
        self._area.setVisible(bool(programmes))
        self._days_area.setVisible(bool(programmes))
        self._no_schedule.setVisible(not programmes)

    def _fill_days(self, days: list[str], now: float) -> None:
        if days != self._shown_days:
            self._shown_days = days
            for button in self._day_group.buttons():
                self._day_group.removeButton(button)
                button.hide()
                button.deleteLater()
            while self._day_layout.count():
                self._day_layout.takeAt(0)
            for day in days:
                chip = set_role(QPushButton(day_title(day, now)), "chip")
                chip.setCheckable(True)
                chip.setProperty("day", day)
                chip.clicked.connect(lambda _checked, d=day: self._choose_day(d))
                self._day_group.addButton(chip)
                self._day_layout.addWidget(chip)
                # Shown at once, not only with the next event: the row is measured
                # right after this, and hidden buttons count for nothing.
                chip.show()
            self._day_layout.addStretch()
            self._day_row.adjustSize()
        for chip in self._day_group.buttons():
            if chip.property("day") == self._day:
                chip.setChecked(True)
                # Clear of the fade at the edge.
                self._days_area.ensureWidgetVisible(chip, EDGE_FADE_WIDTH + 8, 0)

    def _choose_day(self, day: str) -> None:
        self._day = day
        self._show_service()

    def _fill_rows(self, sid: int | None, programmes: list[Programme]) -> None:
        if (sid, self._day, programmes) == self._shown_programmes:
            self._mark_running()
            return
        self._shown_programmes = (sid, self._day, programmes)
        for row in self._rows:
            self._row_layout.removeWidget(row)
            row.deleteLater()
        self._rows = []
        for index, programme in enumerate(programmes):
            row = _ProgrammeRow(programme, self._theme)
            row.opened.connect(lambda r=row: self._close_others(r))
            self._row_layout.insertWidget(index, row)
            self._rows.append(row)
        running = self._mark_running()
        self._area.verticalScrollBar().setValue(0)
        # The rows have no place yet: scrolled to once the area knows its new extent.
        self._pending_scroll = running

    def _close_others(self, opened: _ProgrammeRow) -> None:
        """One description open at a time: the one opened before closes."""
        for row in self._rows:
            if row is not opened and row.is_open:
                row.set_open(False)

    def _scroll_to_pending(self, _minimum: int = 0, maximum: int = 0) -> None:
        """The programme running at the top, with the end of the one before it."""
        row = self._pending_scroll
        if row is None or maximum <= 0:
            return
        self._pending_scroll = None
        self._area.verticalScrollBar().setValue(max(row.y() - RUNNING_TOP_MARGIN, 0))

    def _mark_running(self) -> _ProgrammeRow | None:
        now = self._now()
        running = None
        for row in self._rows:
            if row.set_now(now):
                running = row
        return running

    # --- Playing and status -----------------------------------------------------------

    def _update_play_button(self, *_args: object) -> None:
        found = self._find(self._sid)
        tunable = False
        if found is not None:
            group, entry = found
            tunable = group.channel is not None and group.eid is None and entry.listed
        free = self._controller.state in (ReceptionState.IDLE, ReceptionState.RECEIVING)
        self._play.setEnabled(tunable and free)

    def _on_play_clicked(self) -> None:
        found = self._find(self._sid)
        if found is None:
            return
        group, entry = found
        self._controller.play_service(group.channel, entry.sid)

    def _on_model_changed(self) -> None:
        if self.isVisible():
            self.refresh()
        else:
            self._dirty = True

    def _update_status(self) -> None:
        controller = self._controller
        if controller.guide_scid is None:
            self._status.setText("")
            return
        channel = controller.list_channel
        carousel = self._model.carousel
        if carousel is not None and carousel.directory_objects is not None:
            total = carousel.directory_objects
            if channel:
                text = tr_plural(
                    "Receiving the programme guide from {channel}: {complete} of {total} file.",
                    "Receiving the programme guide from {channel}: {complete} of {total} files.",
                    total,
                )
            else:
                text = tr_plural(
                    "Receiving the programme guide from this frequency: {complete} of {total} "
                    "file.",
                    "Receiving the programme guide from this frequency: {complete} of {total} "
                    "files.",
                    total,
                )
            text = text.format(channel=channel, complete=carousel.complete_objects, total=total)
        elif channel:
            text = tr("Receiving the programme guide from {channel}.").format(channel=channel)
        else:
            text = tr("Receiving the programme guide from this frequency.")
        self._status.setText(text)


class _CarouselPage(Page):
    """The files of the carousel as its directory lists them, how far each is in, and
    beneath them how the data groups and packets were checked."""

    def __init__(self, model: GuideModel) -> None:
        super().__init__(tr("MOT carousel"))
        self._model = model
        self._dirty = True
        self.table = touch_table(
            [tr("File"), tr("Content"), tr("Size"), tr("Segments"), tr("Received")], stretch=0
        )
        self.check = set_role(QLabel(), "muted")
        self.check.setWordWrap(True)
        self.body.addWidget(self.table, stretch=1)
        self.body.addWidget(self.check)
        model.carousel_changed.connect(self._on_changed)
        model.decoding_changed.connect(self._on_changed)

    def _on_changed(self) -> None:
        self._dirty = True

    def render(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        model = self._model
        status = model.carousel
        objects = () if status is None else status.objects
        # Files of which nothing has come yet are greyed.
        pending = {index for index, o in enumerate(objects) if not o.segments and not o.complete}
        fill_table(
            self.table, [object_row(o) for o in objects], right_aligned=range(2, 5), greyed=pending
        )
        decoding = model.decoding
        address = (
            None if decoding is None or decoding.packet is None else decoding.packet.packet_address
        )
        self.check.setText(
            "" if status is None else describe_carousel(status, model.packets, address)
        )
        self.status.setText(_status(model))


def _status(model: GuideModel) -> str:
    decoding = model.decoding
    if decoding is not None:
        return tr("Decoding “{label}”").format(label=decoding.label)
    if model.spi is not None:
        return tr("The programme guide is not being decoded.")
    return tr(_NO_SPI)


class GuideView(QWidget):
    """The two pages in a pager, and above them a note with a switch while the guide is
    switched off on the "Data services" menu."""

    def __init__(
        self,
        controller: ReceiverController,
        model: GuideModel,
        clock: ReferenceClock,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._model = model
        self._note = SwitchedOffNote(
            tr("The programme guide is switched off in the data services.")
        )
        self._switch_on = self._note.button
        self._switch_on.clicked.connect(self._on_switch_on)

        self._schedule = _SchedulePage(controller, model, clock)
        self._pages: dict[str, QWidget] = {
            SCHEDULE: self._schedule,
            CAROUSEL: _CarouselPage(model),
        }
        self._pager = Pager()
        for key in PAGES:
            self._pager.add_page(self._pages[key])
        # A page turned to is drawn at once, not only at the next tick of the window.
        self._pager.current_changed.connect(lambda _index: self.render())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self._note)
        layout.addWidget(self._pager, stretch=1)

        model.decoding_changed.connect(self._update_note)
        self._update_note()
        self.apply_theme(THEMES[DARK])

    @property
    def pager(self) -> Pager:
        return self._pager

    def page(self, key: str) -> QWidget:
        return self._pages[key]

    def current_page(self) -> str:
        return PAGES[self._pager.current_index()]

    def show_page(self, key: str) -> None:
        self._pager.set_current_index(PAGES.index(key))

    def show_running(self, sid: int) -> None:
        """The guide card of the radio page: the schedule of the service, today, the
        programme running at the top."""
        self.show_page(SCHEDULE)
        self._schedule.show_running(sid)

    def render(self) -> None:
        """Draws the page shown; called by the window at its plot rate."""
        self._pages[self.current_page()].render()

    def apply_theme(self, theme: Theme) -> None:
        self._pager.set_theme(theme)
        for page in self._pages.values():
            page.apply_theme(theme)

    def _spi_switched_off(self) -> bool:
        spi = self._model.spi
        return spi is not None and self._controller.is_data_component_switched_off(spi)

    def _update_note(self) -> None:
        self._note.setVisible(self._spi_switched_off())

    def _on_switch_on(self) -> None:
        spi = self._model.spi
        if spi is not None:
            self._controller.set_data_component_enabled(spi.scid, True)
