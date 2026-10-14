"""The radio page: what is playing, with slideshow, dynamic label, a running
announcement, the programme running if the programme guide has a schedule of the station,
the star on the corner of the station's logo that marks it as favourite, and the controls
for playing, stepping through the stations, the sound and the volume.

A long press on the slideshow leads to the choice of what to show in its place
(`ui.touch.display_choice`): the slideshow, a panel of the signal page, or the satellites
in space while the ensemble carries GEPOS, which two fingers bring nearer or farther. A
tap on the slideshow leads to the slides received from the station
(`ui.touch.slide_gallery`), the button beside the loudspeaker to the equalizer
(`ui.touch.equalizer_view`); it is tinted, its symbol in the accent colour, while the
equalizer is on. While the station sends Journaline, a card above the programme leads
to its pages (`ui.touch.journaline_view`); it gives way where the column lacks the room
for it, to an announcement, DL Plus and the programme."""

from __future__ import annotations

import math
import time
from typing import NamedTuple

from PyQt6.QtCore import (
    QEvent,
    QObject,
    QPointF,
    QRectF,
    QSignalBlocker,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QEventPoint,
    QIcon,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QTouchEvent,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from dab.announcement import ALARM_CLUSTER, AnnouncementStatus, announcement_types
from dab.dls import DynamicLabel
from dab.mot import MotObject
from dab.slideshow import Slide, SlideStore
from dab.spi import Programme
from dab.timecode import ReferenceClock
from dsp.mp2_frame import Mp2FrameResult
from dsp.pipeline import SubchannelStatus
from dsp.superframe import SuperframeResult
from i18n import tr
from ui.cut_model import CutModel
from ui.globe_gl import GlobeRenderThread
from ui.gnss_model import GnssModel
from ui.guide_model import GuideModel, now_and_next
from ui.journaline_model import JournalineModel
from ui.offscreen_gl import RenderThread
from ui.receiver_controller import (
    AnnouncementVisit,
    ReceiverController,
    ReceptionState,
    describe_other_channels,
    volume_text,
)
from ui.touch.cut_view import CutPage
from ui.touch.display_choice import (
    CONSTELLATION,
    DISPLAY_KEYS,
    SETTINGS_RADIO_DISPLAY,
    SLIDESHOW,
    DisplayChoicePage,
    make_panel,
    offered_displays,
)
from ui.touch.equalizer_view import EqualizerPage
from ui.touch.fine_slider import FineSlider
from ui.touch.globe_view import WHEEL_ZOOM
from ui.touch.guide_view import day_key, day_title, duration_text, running_text
from ui.touch.icons import symbol_icon, symbol_pixmap
from ui.touch.journaline_view import JournalineCard, JournalinePage
from ui.touch.overlay import ActionSheet
from ui.touch.pager import SWIPE_SLOP
from ui.touch.quality import ReceptionQuality
from ui.touch.recordings_view import RecordingsPage
from ui.touch.signal_panels import ConstellationPanel, FrameTally, SignalPanel
from ui.touch.slide_gallery import SlideGalleryPage, show_slide
from ui.touch.stations import Station, StationDirectory, placeholder_pixmap
from ui.touch.theme import BADGE_SIZE, DARK, THEMES, Theme
from ui.touch.widgets import (
    LONG_PRESS_MS,
    ElidedLabel,
    HoldButton,
    Marquee,
    SignalBars,
    TapWidget,
    repolish,
    set_role,
)

# The slideshow's standard size, 320 x 240, shown at twice its size: an integer factor
# keeps it sharp (and clause 9.2.3 of ETSI TS 101 499 permits it, see
# `permitted_scale_factor`).
SLIDE_SIZE = QSize(640, 480)
LOGO_SIZE = 96
# Shown in place of a slide while none has arrived: the station's logo or initials.
FALLBACK_LOGO_SIZE = 240
# The column beside the slide: what is left of the page (1168 px beside the navigation,
# less 24 px margin on either side) after the slide and the gap of 32 px.
DETAILS_WIDTH = 448
# The star of a favourite on the lower right corner of the logo: a round badge reaching
# `BADGE_OVERLAP` beyond it, in a square to tap (`BADGE_SIZE`) a little larger.
BADGE_DIAMETER = 46
BADGE_OVERLAP = 12
BADGE_STAR = 26
# The badge of the recordings on the lower right corner of the slide: a dot, red while
# a title records, and the count of titles in the ring.
BADGE_DOT = 5.0
BADGE_COUNT_FONT_PX = 17
BOTTOM_BAR_HEIGHT = 112
TRANSPORT_BUTTON = 88
ICON_SIZE = QSize(40, 40)
VOLUME_SLIDER_WIDTH = 360
# The spacings of the ticks on the scale of the volume's loupe, in percent.
VOLUME_TICKS = (1, 5, 10, 50, 100)
# The gain in dB beside the slider ("−59 dB", "muted").
VOLUME_LABEL_WIDTH = 72
# The calendar at the left of the guide card and the arrow at its right.
GUIDE_ICON = 32
# Between the Journaline card and the guide card: closer than the rest of the column,
# a group; so the Journaline card still fits beside an announcement.
CARD_GAP = 8
GUIDE_CHEVRON = 28
# A programme starting sooner is announced with the time to go, a later one by its day.
STARTS_SOON_SECONDS = 12 * 3600

# Interval at which the slideshow checks its time stamps, as in `ui.programme_tab`.
SLIDESHOW_TICK_MS = 500

# Stands for "the slide area must be drawn afresh" where a slide or None is expected.
_REFRESH = object()

_TOUCH_EVENTS = (
    QEvent.Type.TouchBegin,
    QEvent.Type.TouchUpdate,
    QEvent.Type.TouchEnd,
    QEvent.Type.TouchCancel,
)


def badge_pixmap(theme: Theme, favourite: bool) -> QPixmap:
    """The badge of the star: a disc in the colour of the panels, set off from the logo
    by a ring in that of the background, the star filled for a favourite."""
    pixmap = QPixmap(BADGE_SIZE, BADGE_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    ring = 3.0
    painter.setPen(QPen(QColor(theme.background), ring))
    painter.setBrush(QColor(theme.surface))
    radius = (BADGE_DIAMETER - ring) / 2
    centre = QPointF(BADGE_SIZE / 2, BADGE_SIZE / 2)
    painter.drawEllipse(centre, radius, radius)
    star = symbol_pixmap(
        "star" if favourite else "star_outline",
        theme.color("accent" if favourite else "text"),
        BADGE_STAR,
    )
    corner = (BADGE_SIZE - BADGE_STAR) // 2
    painter.drawPixmap(corner, corner, star)
    painter.end()
    return pixmap


def recordings_badge_pixmap(theme: Theme, recording: bool, count: int) -> QPixmap:
    """The badge of the recordings: the disc of the star's badge with a dot (red while a
    title records) and the count of titles in the ring ("9+" beyond nine)."""
    pixmap = QPixmap(BADGE_SIZE, BADGE_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    ring = 3.0
    painter.setPen(QPen(QColor(theme.background), ring))
    painter.setBrush(QColor(theme.surface))
    radius = (BADGE_DIAMETER - ring) / 2
    centre = QPointF(BADGE_SIZE / 2, BADGE_SIZE / 2)
    painter.drawEllipse(centre, radius, radius)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(theme.color("danger" if recording else "muted"))
    painter.drawEllipse(QPointF(centre.x() - 9, centre.y()), BADGE_DOT, BADGE_DOT)
    font = painter.font()
    font.setPixelSize(BADGE_COUNT_FONT_PX)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(theme.color("text"))
    text = str(count) if count <= 9 else "9+"
    painter.drawText(
        QRectF(centre.x() - 1, 0, BADGE_SIZE / 2, BADGE_SIZE),
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        text,
    )
    painter.end()
    return pixmap


def clock_text(timestamp: float) -> str:
    return time.strftime("%H:%M", time.localtime(timestamp))


def starts_text(start: float, now: float) -> str:
    """When a programme not yet running starts: "starts in 20 min" if soon (also after
    midnight), else the day as the guide names it ("Tomorrow", "Wed 7 Oct")."""
    if start - now < STARTS_SOON_SECONDS:
        return tr("starts in {duration}").format(duration=duration_text(round(start - now)))
    return day_title(day_key(start), now)


class Notice(NamedTuple):
    """What the notice on a running announcement says."""

    title: str
    detail: str
    alarm: bool  # shown in the colour of danger


def _notice_title(cluster_id: int, flags: int) -> str:
    if cluster_id == ALARM_CLUSTER:
        return tr("Alarm announcement")
    # The names come translated (`dab.announcement.announcement_types`).
    types = ", ".join(name.short for name in announcement_types(flags))
    # Translators: {types} are the names of the announcement types, e.g. "News flash".
    return tr("Announcement: {types}").format(types=types)


def announcement_notice(
    status: AnnouncementStatus | None,
    source: str | None,
    selected_sid: int | None,
    *,
    switched: bool,
    switching_enabled: bool,
) -> Notice | None:
    """The notice on an announcement running for the selected service, in this ensemble
    or another; None while none runs. `source` names the service carrying it, `switched`
    says whether the audio follows it into another subchannel (see
    `dsp.pipeline.SubchannelStatus`)."""
    if status is None:
        return None
    off = "" if switching_enabled else " · " + tr("switching is off")
    if status.oe_active is not None:
        announcement = status.oe_active
        if source:
            where = tr("running on {station}").format(station=source)
        else:
            where = tr("running in another ensemble")
        alarm = announcement.cluster_id == ALARM_CLUSTER
        return Notice(
            _notice_title(announcement.cluster_id, announcement.flags), where + off, alarm
        )
    if status.active is None:
        return None
    announcement = status.active
    title = _notice_title(announcement.cluster_id, announcement.flags)
    alarm = announcement.cluster_id == ALARM_CLUSTER
    if status.source_sid is not None and status.source_sid == selected_sid:
        return Notice(title, tr("running on this station"), alarm)
    where = source or tr("Subchannel {number}").format(number=announcement.subchannel_id)
    if switched:
        return Notice(title, tr("audio comes from {station}").format(station=where), alarm)
    return Notice(title, tr("running on {station}").format(station=where) + off, alarm)


def visit_notice(visit: AnnouncementVisit) -> Notice:
    """The notice while reception has gone to another ensemble for an announcement:
    where it goes back to, and that a tap goes back at once."""
    announcement = visit.announcement
    origin = visit.origin_label or tr("Station {sid:04X}").format(sid=visit.origin_sid)
    return Notice(
        _notice_title(announcement.cluster_id, announcement.flags),
        tr("then back to {station} · tap to return now").format(station=origin),
        announcement.cluster_id == ALARM_CLUSTER,
    )


class _GuideCard(TapWidget):
    """The programme running with the time left and the one after it, from the programme
    guide; without one running, the next. A tap leads to the guide."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        set_role(self, "panel")
        # A plain QWidget subclass draws the background of the style sheet only so.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self._icon = QLabel()
        self._icon.setFixedSize(GUIDE_ICON, GUIDE_ICON)
        self._chevron = QLabel()
        self._chevron.setFixedSize(GUIDE_CHEVRON, GUIDE_CHEVRON)
        self.time = set_role(QLabel(), "large")
        self.title = set_role(ElidedLabel(), "large")
        self.detail = set_role(QLabel(), "muted")
        self.next_time = set_role(QLabel(), "muted")
        self.next_title = set_role(ElidedLabel(), "muted")
        layout = QGridLayout(self)
        layout.setContentsMargins(16, 12, 12, 12)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(4)
        layout.addWidget(self._icon, 0, 0, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.time, 0, 1)
        layout.addWidget(self.title, 0, 2)
        layout.addWidget(self.detail, 1, 2)
        layout.addWidget(self.next_time, 2, 1)
        layout.addWidget(self.next_title, 2, 2)
        layout.addWidget(self._chevron, 0, 3, 3, 1, alignment=Qt.AlignmentFlag.AlignVCenter)
        layout.setColumnStretch(2, 1)

    def apply_theme(self, theme: Theme) -> None:
        self._icon.setPixmap(symbol_pixmap("calendar", theme.color("accent"), GUIDE_ICON))
        self._chevron.setPixmap(symbol_pixmap("chevron", theme.color("muted"), GUIDE_CHEVRON))

    def show_programmes(
        self, running: Programme | None, following: Programme | None, now: float
    ) -> None:
        first = running or following
        if first is None:
            return
        self.time.setText(clock_text(first.start))
        self.title.setText(first.names.best)
        if running is not None:
            self.detail.setText(running_text(round(running.end - now)))
            later = following
        else:
            self.detail.setText(starts_text(first.start, now))
            later = None
        self.next_time.setVisible(later is not None)
        self.next_title.setVisible(later is not None)
        if later is not None:
            self.next_time.setText(clock_text(later.start))
            self.next_title.setText(later.names.best)


class _SlideArea(QStackedWidget):
    """The slide, or the display chosen in its place. A finger resting on it asks for
    another display: the area is shaded while it rests, and `long_pressed` follows
    after `LONG_PRESS_MS` unless it moved or lifted first. Lifted earlier without moving,
    it was a tap (`tapped`). Two fingers spreading or closing make `pinched` with how
    much nearer (below 1) or farther (above 1) they ask the display to be, as the wheel
    does on a PC, and moving together `dragged` with how far their middle moved;
    they are no long press and no tap. What the area shows never takes the touch
    itself.

    The area takes the fingers as touch events, which Qt then does not turn into mouse
    events; the mouse of a PC comes as mouse events: the left button as a finger, the
    middle one held makes `turned` with how far it moved."""

    long_pressed = pyqtSignal()
    tapped = pyqtSignal()
    pinched = pyqtSignal(float)
    dragged = pyqtSignal(float, float)  # pixels across and down
    turned = pyqtSignal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(SLIDE_SIZE)
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents)
        self._pressed_at: QPointF | None = None
        # Two fingers or more: the first two by id, and how far apart they are.
        self._fingers: tuple[tuple[int, int], float, QPointF] | None = None
        self._turn_from: QPointF | None = None  # the middle button held
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(LONG_PRESS_MS)
        self._timer.timeout.connect(self._on_held)
        self._shade = set_role(QWidget(self), "shade")
        self._shade.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self._shade.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._shade.setGeometry(self.rect())
        self._shade.hide()

    def add(self, widget: QWidget) -> None:
        widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.addWidget(widget)

    def held(self) -> bool:
        """Whether a finger rests on the area, not yet long enough."""
        return self._timer.isActive()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        button = event.button()
        if button == Qt.MouseButton.LeftButton:
            self._press(event.position())
        elif button == Qt.MouseButton.MiddleButton:
            self._turn_from = event.position()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        at = event.position()
        if self._turn_from is not None:
            moved, self._turn_from = at - self._turn_from, at
            self.turned.emit(moved.x(), moved.y())
        else:
            self._move(at)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        button = event.button()
        if button == Qt.MouseButton.LeftButton:
            self._release()
        elif button == Qt.MouseButton.MiddleButton:
            self._turn_from = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        # A step of the wheel is 120 units; touchpads send less at a time.
        steps = event.angleDelta().y() / 120
        if steps:
            self.pinched.emit(WHEEL_ZOOM**steps)
        event.accept()

    def event(self, event: QEvent) -> bool:
        if event.type() in _TOUCH_EVENTS:
            self._on_touch(event)  # type: ignore[arg-type]
            event.accept()
            return True
        return super().event(event)

    def _on_touch(self, touch: QTouchEvent) -> None:
        kind = touch.type()
        down = [p for p in touch.points() if p.state() != QEventPoint.State.Released]
        if kind == QEvent.Type.TouchBegin:
            self._fingers = None
            if len(touch.points()) == 1:
                self._press(self._local(touch.points()[0]))
            else:
                self._pinch(down)
        elif kind == QEvent.Type.TouchUpdate:
            if len(down) >= 2:
                self._pinch(down)
            elif down:
                self._fingers = None
                self._move(self._local(down[0]))
        else:
            self._fingers = None
            if kind == QEvent.Type.TouchEnd:
                self._release()
            else:
                self._let_go()

    def _local(self, point: QEventPoint) -> QPointF:
        """Where a finger is on the area, from its position on the screen (the one Qt
        always fills in)."""
        return self.mapFromGlobal(point.globalPosition())

    def _pinch(self, down: list[QEventPoint]) -> None:
        """One step of two fingers or more; the first two by id count."""
        self._let_go()  # neither a long press nor a tap
        first, second = sorted(down, key=lambda p: p.id())[:2]
        a, b = self._local(first), self._local(second)
        ids = (first.id(), second.id())
        distance = math.hypot(a.x() - b.x(), a.y() - b.y())
        middle = (a + b) / 2
        previous, self._fingers = self._fingers, (ids, distance, middle)
        if previous is None or previous[0] != ids:
            return  # a finger came or went: the movement begins anew from here
        moved = middle - previous[2]
        if moved.x() or moved.y():
            self.dragged.emit(moved.x(), moved.y())
        if previous[1] >= 1 and distance >= 1:
            self.pinched.emit(previous[1] / distance)

    def _press(self, at: QPointF) -> None:
        self._pressed_at = at
        self._timer.start()
        self._shade.raise_()
        self._shade.show()

    def _move(self, at: QPointF) -> None:
        if self._pressed_at is None:
            return
        if (at - self._pressed_at).manhattanLength() > SWIPE_SLOP:
            self._let_go()

    def _release(self) -> None:
        # Still running: neither held long enough nor moved.
        tap = self._timer.isActive()
        self._let_go()
        if tap:
            self.tapped.emit()

    def _let_go(self) -> None:
        self._pressed_at = None
        self._timer.stop()
        self._shade.hide()

    def _on_held(self) -> None:
        self._let_go()
        self.long_pressed.emit()


class RadioPage(QWidget):
    stations_requested = pyqtSignal()  # Start pressed with no station chosen yet
    guide_requested = pyqtSignal(int)  # the guide card was tapped: the station's SId

    def __init__(
        self,
        controller: ReceiverController,
        directory: StationDirectory,
        guide: GuideModel,
        journaline: JournalineModel,
        clock: ReferenceClock,
        tally: FrameTally,
        gnss_model: GnssModel,
        globe_renderer: GlobeRenderThread | None = None,
        waterfall_renderer: RenderThread | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._directory = directory
        self._guide = guide
        self._journaline = journaline
        self._clock = clock
        self._tally = tally
        self._gnss_model = gnss_model
        self._globe_renderer = globe_renderer
        self._waterfall_renderer = waterfall_renderer
        # The schedule of the station playing, across all guides.
        self._programmes: list[Programme] = []
        self._theme = THEMES[DARK]
        self._slides = SlideStore()
        # The slide on display, None for the logo standing in for one.
        self._shown_slide: Slide | None | object = _REFRESH
        self._quality = ReceptionQuality()
        self._subchannel_text: str | None = None
        # The announcement running for the station (with the name of the service carrying
        # it), and whether the audio follows it there.
        self._announcement: tuple[AnnouncementStatus | None, str | None] = (None, None)
        self._switched_to_announcement = False
        # Reception gone to another ensemble for an announcement there.
        self._visit: AnnouncementVisit | None = controller.announcement_visit

        self._slide_label = QLabel()
        set_role(self._slide_label, "slide")
        self._slide_label.setFixedSize(SLIDE_SIZE)
        self._slide_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._slide_area = _SlideArea()
        self._slide_area.add(self._slide_label)
        self._slide_area.long_pressed.connect(self.show_choice)
        self._slide_area.tapped.connect(self._on_slide_tapped)
        self._slide_area.pinched.connect(self._on_pinched)
        self._slide_area.dragged.connect(self._on_dragged)
        self._slide_area.turned.connect(self._on_turned)
        # The panels shown so far, made when first shown.
        self._panels: dict[str, SignalPanel] = {}
        # How the constellation is drawn, for the panel made later (the window sets it
        # from the settings page, as on the signal page).
        self._constellation_density = False
        # The display chosen, and the one shown: the slideshow while the ensemble lacks
        # the service of the one chosen. None before anything is shown.
        self._display = SLIDESHOW
        self._shown: str | None = None

        # The filter of the station page, which ◀ ▶ step through, above logo and name.
        self._filter_label = set_role(ElidedLabel(), "heading")
        self._logo_label = QLabel()
        self._logo_label.setFixedSize(LOGO_SIZE, LOGO_SIZE)
        self._logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name_label = set_role(QLabel(), "title")
        self._name_label.setWordWrap(True)
        self._ensemble_label = set_role(QLabel(), "muted")
        self._ensemble_label.setWordWrap(True)
        # One line; a label too long for it runs through.
        self._dls_label = set_role(Marquee(), "large")
        self._dl_plus_label = set_role(QLabel(), "muted")
        self._dl_plus_label.setWordWrap(True)
        self._dl_plus_label.hide()
        self._guide_card = _GuideCard()
        self._guide_card.hide()
        self._guide_card.tapped.connect(self._on_guide_card_tapped)
        # Shown by `_fit_journaline_card`.
        self._journaline_card = JournalineCard(journaline)
        self._journaline_card.hide()
        self._journaline_card.tapped.connect(self.show_journaline)
        self._bars = SignalBars()
        self._reception_label = set_role(QLabel(), "muted")
        self._reception_label.setWordWrap(True)
        self._notice = set_role(TapWidget(), "notice")
        self._notice.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self._notice.tapped.connect(self._on_notice_tapped)
        self._notice.setProperty("alarm", False)
        self._notice_title = set_role(QLabel(), "large")
        self._notice_title.setWordWrap(True)  # several types make a long title
        self._notice_detail = QLabel()
        self._notice_detail.setWordWrap(True)
        notice = QVBoxLayout(self._notice)
        notice.setContentsMargins(16, 10, 16, 12)
        notice.setSpacing(2)
        notice.addWidget(self._notice_title)
        notice.addWidget(self._notice_detail)
        self._notice.hide()

        names = QVBoxLayout()
        names.setSpacing(4)
        names.addStretch()
        names.addWidget(self._name_label)
        names.addWidget(self._ensemble_label)
        names.addStretch()
        header = QHBoxLayout()
        header.setSpacing(16)
        header.addWidget(self._logo_label)
        header.addLayout(names, stretch=1)
        station = QVBoxLayout()
        station.setSpacing(8)
        station.addWidget(self._filter_label)
        station.addLayout(header)
        reception = QHBoxLayout()
        reception.setSpacing(12)
        reception.addWidget(self._bars)
        reception.addWidget(self._reception_label, stretch=1)

        details_widget = QWidget()
        details_widget.setFixedWidth(DETAILS_WIDTH)
        self._details = details_widget
        # On the corner of the logo, outside the layout: placed wherever the logo is
        # (`eventFilter`).
        self._favourite_button = set_role(QToolButton(details_widget), "badge")
        self._favourite_button.setCheckable(True)
        self._favourite_button.setIconSize(QSize(BADGE_SIZE, BADGE_SIZE))
        self._favourite_button.clicked.connect(self._on_favourite_clicked)
        self._logo_label.installEventFilter(self)
        details = QVBoxLayout(details_widget)
        details.setContentsMargins(0, 0, 0, 0)
        details.setSpacing(16)
        labels = QVBoxLayout()
        labels.setSpacing(4)
        labels.addWidget(self._dls_label)
        labels.addWidget(self._dl_plus_label)
        details.addLayout(station)
        # Right beneath the name, where it is not missed.
        details.addWidget(self._notice)
        details.addLayout(labels)
        # All the room left, so that the header stays put whatever the labels say.
        details.addStretch(1)
        # Above the reception line, where they stay put as well.
        cards = QVBoxLayout()
        cards.setSpacing(CARD_GAP)
        cards.addWidget(self._journaline_card)
        cards.addWidget(self._guide_card)
        details.addLayout(cards)
        details.addLayout(reception)
        # Whatever changes in the column may take the room of the Journaline card.
        details_widget.installEventFilter(self)

        top = QHBoxLayout()
        top.setContentsMargins(0, 24, 0, 16)
        top.setSpacing(32)
        top.addWidget(self._slide_area, alignment=Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(details_widget)

        self._previous_button = self._transport_button(lambda: self._step(-1))
        self._play_button = self._transport_button(self._on_play_clicked)
        self._next_button = self._transport_button(lambda: self._step(1))
        self._equalizer_button = self._transport_button(self.show_equalizer)
        # A tap mutes; held, it offers where the audio goes (`_choose_output`), and its
        # symbol tells: headphones while the audio goes to a Bluetooth device.
        self._mute_button = HoldButton()
        self._mute_button.setFixedSize(TRANSPORT_BUTTON, TRANSPORT_BUTTON)
        self._mute_button.setIconSize(ICON_SIZE)
        self._mute_button.setCheckable(True)
        self._mute_button.long_pressed.connect(self._choose_output)
        self._output_sheet: ActionSheet | None = None
        self._mute_button.setChecked(not controller.audio_enabled)
        # A tap tells the controller; the button follows the controller, which is also
        # muted from elsewhere (MQTT, playing a recording lifts the mute).
        self._mute_button.clicked.connect(lambda muted: controller.set_audio_enabled(not muted))
        self._mute_button.toggled.connect(self._show_mute_icon)
        controller.audio_enabled_changed.connect(
            lambda enabled: self._mute_button.setChecked(not enabled)
        )
        # Pulled up, the finger sets it finer, the level in a loupe above it
        # (`ui.touch.fine_slider`); a tap beside the handle pages, never jumps to full.
        self._volume_slider = FineSlider(
            lambda _percent: volume_text(controller.volume_db, controller.volume_device_percent),
            VOLUME_TICKS,
        )
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(controller.volume_percent)
        self._volume_slider.setFixedWidth(VOLUME_SLIDER_WIDTH)
        self._volume_slider.valueChanged.connect(controller.set_volume_percent)
        # The gain in dB (`ui.receiver_controller.volume_text`), also when the ceiling
        # of the volume moves it in the settings. Hidden unless chosen in the settings
        # (`set_volume_figure_shown`).
        self._volume_label = set_role(QLabel(), "small")
        self._volume_label.setFixedWidth(VOLUME_LABEL_WIDTH)
        self._volume_label.hide()
        self._show_volume()
        controller.volume_changed.connect(self._show_volume)
        controller.audio_outputs_changed.connect(self._show_output)

        bottom = QHBoxLayout()
        bottom.setContentsMargins(24, 0, 32, 0)
        bottom.setSpacing(16)
        bottom.addWidget(self._previous_button)
        bottom.addWidget(self._play_button)
        bottom.addWidget(self._next_button)
        bottom.addStretch()
        bottom.addWidget(self._equalizer_button)
        bottom.addWidget(self._mute_button)
        bottom.addWidget(self._volume_slider)
        bottom.addWidget(self._volume_label)
        bottom_bar = set_role(QWidget(), "panel")
        bottom_bar.setFixedHeight(BOTTOM_BAR_HEIGHT)
        bottom_bar.setLayout(bottom)

        player = QWidget()
        layout = QVBoxLayout(player)
        layout.setContentsMargins(24, 0, 24, 16)
        layout.setSpacing(0)
        layout.addLayout(top, stretch=1)
        layout.addWidget(bottom_bar)
        # On the corner of the slide, outside the layout like the star: the way to the
        # recordings, with the count of titles in the ring (`_place_recordings_badge`).
        self._recordings_button = set_role(QToolButton(player), "badge")
        self._recordings_button.setIconSize(QSize(BADGE_SIZE, BADGE_SIZE))
        self._recordings_button.setToolTip(tr("Recordings"))
        self._recordings_button.clicked.connect(self.show_recordings)
        self._slide_area.installEventFilter(self)

        self._choice = DisplayChoicePage()
        self._choice.chosen.connect(self.show_display)
        self._choice.back_requested.connect(self.show_player)
        self._gallery = SlideGalleryPage(self._slides)
        self._gallery.back_button.clicked.connect(self.show_player)
        self._equalizer = EqualizerPage(controller)
        self._equalizer.back_button.clicked.connect(self.show_player)
        self._recordings = RecordingsPage(controller)
        self._recordings.back_button.clicked.connect(self.show_player)
        self._recordings.cut_requested.connect(self.show_cut)
        self._cut_model = CutModel(controller, self)
        self._cut = CutPage(self._cut_model)
        self._cut.back_button.clicked.connect(self._leave_cut)
        self._journaline_page = JournalinePage(journaline)
        self._journaline_page.back_button.clicked.connect(self.show_player)
        self._stack = QStackedWidget()
        self._stack.addWidget(player)
        self._stack.addWidget(self._choice)
        self._stack.addWidget(self._gallery)
        self._stack.addWidget(self._equalizer)
        self._stack.addWidget(self._recordings)
        self._stack.addWidget(self._cut)
        self._stack.addWidget(self._journaline_page)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._stack)

        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(SLIDESHOW_TICK_MS)
        self._tick_timer.timeout.connect(self._on_tick)
        self._tick_timer.start()

        dsp = controller.dsp_worker
        dsp.dynamic_label_ready.connect(self._show_dynamic_label)
        dsp.slide_ready.connect(self._add_mot_object)
        dsp.subchannel_info_ready.connect(self._on_subchannel_info)
        dsp.superframe_ready.connect(self._on_superframe)
        dsp.mp2_frame_ready.connect(self._on_mp2_frame)
        controller.state_changed.connect(self._on_state_changed)
        controller.service_reset.connect(self._reset_service)
        controller.announcement_changed.connect(self._on_announcement_changed)
        controller.announcement_visit_changed.connect(self._on_visit_changed)
        controller.other_ensembles_changed.connect(self._show_ensemble_line)
        controller.data_components_changed.connect(self._show_display)
        controller.equalizer_changed.connect(self._show_equalizer_state)
        controller.recordings_changed.connect(self._show_recordings_badge)
        controller.recorder_status_changed.connect(self._on_recorder_status)
        directory.changed.connect(self._show_station)
        guide.changed.connect(self._load_schedule)
        journaline.changed.connect(self._on_journaline_changed)

        self._reset_service()
        stored = controller.settings.value(SETTINGS_RADIO_DISPLAY, SLIDESHOW)
        self._display = stored if stored in DISPLAY_KEYS else SLIDESHOW
        self._show_display()
        self.apply_theme(self._theme)

    def _transport_button(self, slot) -> QToolButton:
        button = QToolButton()
        button.setFixedSize(TRANSPORT_BUTTON, TRANSPORT_BUTTON)
        button.setIconSize(ICON_SIZE)
        if slot is not None:
            button.clicked.connect(slot)
        return button

    # --- Theme ------------------------------------------------------------------------

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        text = theme.color("text")
        self._previous_button.setIcon(symbol_icon("previous", text, ICON_SIZE))
        self._next_button.setIcon(symbol_icon("next", text, ICON_SIZE))
        self._show_equalizer_state()
        self._recordings.apply_theme(theme)
        self._cut.apply_theme(theme)
        self._show_recordings_badge()
        favourite = QIcon(badge_pixmap(theme, favourite=False))
        favourite.addPixmap(badge_pixmap(theme, favourite=True), QIcon.Mode.Normal, QIcon.State.On)
        self._favourite_button.setIcon(favourite)
        self._equalizer.apply_theme(theme)
        self._bars.set_theme(theme)
        self._volume_slider.set_theme(theme)
        self._guide_card.apply_theme(theme)
        self._journaline_card.apply_theme(theme)
        self._journaline_page.apply_theme(theme)
        self._show_output()
        self._on_state_changed(self._controller.state)
        self._show_station()
        self._shown_slide = _REFRESH
        self._update_slide()
        self._choice.apply_theme(theme)
        for panel in self._panels.values():
            panel.apply_theme(theme)

    def _show_output(self) -> None:
        """The loudspeaker button shows headphones while the audio goes to a Bluetooth
        device, either struck through while muted."""
        self._show_mute_icon(self._mute_button.isChecked())

    def mute_symbol(self, muted: bool) -> str:
        """The symbol of the loudspeaker button (`ui.touch.icons`): headphones while
        the audio goes to a Bluetooth device, either struck through while muted."""
        output = self._controller.audio_output_in_use
        bluetooth = output is not None and output.bluetooth
        if muted:
            return "headphones_muted" if bluetooth else "muted"
        return "headphones" if bluetooth else "volume"

    def _show_mute_icon(self, muted: bool) -> None:
        color = self._theme.color("on_accent" if muted else "text")
        self._mute_button.setIcon(symbol_icon(self.mute_symbol(muted), color, ICON_SIZE))

    def output_sheet(self) -> ActionSheet:
        """The choice of the output over the whole window, made on first use."""
        if self._output_sheet is None:
            self._output_sheet = ActionSheet(self.window())
        return self._output_sheet

    def _choose_output(self) -> None:
        """The loudspeaker button held: "Automatic" (the default of the system, which
        takes headphones as they connect) or one output for good, the choice marked."""
        controller = self._controller
        choice = controller.audio_output_choice
        in_use = controller.audio_output_in_use
        if in_use is None:
            detail = tr("Automatic") if choice == "" else ""
        elif choice == "":
            detail = tr("Automatic, currently {name}").format(name=in_use.name)
        else:
            detail = in_use.name
        actions = [
            (
                tr("Automatic"),
                "accent" if choice == "" else "",
                lambda: controller.set_audio_output(""),
            )
        ]
        for output in controller.audio_outputs:
            role = "accent" if output.id == choice else ""
            actions.append(
                (output.name, role, lambda oid=output.id: controller.set_audio_output(oid))
            )
        self.output_sheet().open(tr("Output"), detail, actions)

    def _show_volume(self) -> None:
        """Figure and slider; the slider moves by itself when another output brings
        its own volume."""
        with QSignalBlocker(self._volume_slider):
            self._volume_slider.setValue(self._controller.volume_percent)
        controller = self._controller
        self._volume_label.setText(
            volume_text(controller.volume_db, controller.volume_device_percent)
        )

    def set_volume_figure_shown(self, shown: bool) -> None:
        """Shows the level beside the slider, in dB or in percent of the device's own
        volume; hidden, the slider moves into its place."""
        self._volume_label.setVisible(shown)

    def _show_equalizer_state(self) -> None:
        """The button beside the loudspeaker tells whether the equalizer is on: tinted,
        its symbol in the accent colour (the dynamic property `active`, see
        `ui.touch.theme`). Not a switch: a tap opens the equalizer."""
        active = self._controller.equalizer_enabled
        color = self._theme.color("accent" if active else "text")
        self._equalizer_button.setIcon(symbol_icon("equalizer", color, ICON_SIZE))
        if self._equalizer_button.property("active") != active:
            self._equalizer_button.setProperty("active", active)
            repolish(self._equalizer_button)

    # --- In the place of the slideshow ------------------------------------------------

    def display(self) -> str:
        """The key of what is chosen to show in the place of the slideshow."""
        return self._display

    def shown_display(self) -> str:
        """The key of what is shown there: the slideshow while the ensemble lacks the
        service of the display chosen."""
        return self._shown or SLIDESHOW

    def set_constellation_density(self, density: bool) -> None:
        """Draws the constellation in the place of the slideshow as a density or as
        points, now and when its panel is made."""
        self._constellation_density = density
        panel = self._panels.get(CONSTELLATION)
        if panel is not None:
            panel.set_density(density)

    def choosing(self) -> bool:
        return self._stack.currentWidget() is self._choice

    def in_gallery(self) -> bool:
        return self._stack.currentWidget() is self._gallery

    def show_choice(self) -> None:
        """The choice, with the display shown marked and in sight."""
        self._choice.set_current(self.shown_display())
        self._stack.setCurrentWidget(self._choice)
        self._choice.reveal_current()

    def show_player(self) -> None:
        self._stack.setCurrentIndex(0)

    def show_gallery(self) -> None:
        """The slides received from the station, the one on display chosen."""
        self._stack.setCurrentWidget(self._gallery)
        self._gallery.open()

    def in_equalizer(self) -> bool:
        return self._stack.currentWidget() is self._equalizer

    def show_equalizer(self) -> None:
        """The equalizer, as the button beside the loudspeaker opens it."""
        self._stack.setCurrentWidget(self._equalizer)

    @property
    def equalizer(self) -> EqualizerPage:
        return self._equalizer

    def in_recordings(self) -> bool:
        return self._stack.currentWidget() is self._recordings

    def show_recordings(self) -> None:
        """The recordings, as the badge on the slide opens them."""
        self._stack.setCurrentWidget(self._recordings)

    @property
    def recordings(self) -> RecordingsPage:
        return self._recordings

    def show_cut(self, recording_id: str) -> None:
        """The cutter for a recording, as its sheet among the recordings opens it."""
        self._cut.open(recording_id)
        self._stack.setCurrentWidget(self._cut)

    def _leave_cut(self) -> None:
        """Back to the recordings: the session ends."""
        self._cut_model.close()
        self.show_recordings()

    @property
    def cut(self) -> CutPage:
        return self._cut

    def in_journaline(self) -> bool:
        return self._stack.currentWidget() is self._journaline_page

    def show_journaline(self) -> None:
        """The Journaline of the station, as its card opens it."""
        self._stack.setCurrentWidget(self._journaline_page)

    @property
    def journaline(self) -> JournalinePage:
        return self._journaline_page

    @property
    def journaline_card(self) -> JournalineCard:
        return self._journaline_card

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """The badges follow the logo and the slide wherever the layout puts them; the
        Journaline card shows where the column beside the slide has room for it."""
        if event.type() in (QEvent.Type.Move, QEvent.Type.Resize, QEvent.Type.Show):
            if watched is self._logo_label:
                self._place_badge()
            elif watched is self._slide_area:
                self._place_recordings_badge()
        if watched is self._details and event.type() in (
            QEvent.Type.LayoutRequest,
            QEvent.Type.Resize,
        ):
            self._fit_journaline_card()
        return False

    def _on_journaline_changed(self, _ids: frozenset[int] | None) -> None:
        self._fit_journaline_card()

    def _fit_journaline_card(self) -> None:
        """Shows the Journaline card while the station announces Journaline and the
        column has room for it beside all else it shows: an announcement, two lines of
        DL Plus and the programme together leave none."""
        card = self._journaline_card
        if not self._journaline.announced:
            card.hide()
            return
        layout = self._details.layout()
        width = self._details.width()
        needed = (
            layout.heightForWidth(width)
            if layout.hasHeightForWidth()
            else layout.sizeHint().height()
        )
        if card.isHidden():
            # Beside the guide card in their group, else on its own in the column.
            gap = layout.spacing() if self._guide_card.isHidden() else CARD_GAP
            needed += card.sizeHint().height() + gap
        card.setVisible(needed <= self._details.height())

    def _place_recordings_badge(self) -> None:
        slide = self._slide_area.geometry()
        reach = BADGE_DIAMETER // 2 - BADGE_OVERLAP
        centre_x = slide.x() + slide.width() - reach
        centre_y = slide.y() + slide.height() - reach
        self._recordings_button.setGeometry(
            centre_x - BADGE_SIZE // 2, centre_y - BADGE_SIZE // 2, BADGE_SIZE, BADGE_SIZE
        )
        self._recordings_button.raise_()

    def _on_recorder_status(self, _status: object) -> None:
        self._show_recordings_badge()

    def _show_recordings_badge(self) -> None:
        controller = self._controller
        recording = controller.recorder_status.recording
        count = len(controller.recordings)
        state = (recording, count, self._theme.key)
        if state == getattr(self, "_badge_state", None):
            return
        self._badge_state = state
        self._recordings_button.setIcon(
            QIcon(recordings_badge_pixmap(self._theme, recording, count))
        )

    def _place_badge(self) -> None:
        logo = self._logo_label.geometry()
        reach = BADGE_DIAMETER // 2 - BADGE_OVERLAP  # from the corner into the logo
        centre_x = logo.x() + logo.width() - reach
        centre_y = logo.y() + logo.height() - reach
        self._favourite_button.setGeometry(
            centre_x - BADGE_SIZE // 2, centre_y - BADGE_SIZE // 2, BADGE_SIZE, BADGE_SIZE
        )
        self._favourite_button.raise_()

    def _on_slide_tapped(self) -> None:
        if self.shown_display() == SLIDESHOW:
            self.show_gallery()

    def _on_pinched(self, factor: float) -> None:
        """Two fingers (or the wheel) zoom the panel shown, if it zooms at all."""
        panel = self._panels.get(self.shown_display())
        if panel is not None:
            panel.zoom(factor)

    def _on_dragged(self, dx: float, dy: float) -> None:
        """Two fingers moving together turn the panel shown, if fingers turn it."""
        panel = self._panels.get(self.shown_display())
        if panel is not None and panel.turned_by_fingers:
            panel.turn(dx, dy)

    def _on_turned(self, dx: float, dy: float) -> None:
        """The middle button of a PC turns the panel shown, if it turns at all."""
        panel = self._panels.get(self.shown_display())
        if panel is not None:
            panel.turn(dx, dy)

    def show_display(self, key: str) -> None:
        """Shows `key` in the place of the slideshow from now on, also after the next
        start (while the ensemble carries its service), and goes back to the player."""
        self._controller.settings.setValue(SETTINGS_RADIO_DISPLAY, key)
        self._display = key
        self._show_display()
        self.show_player()

    def _show_display(self) -> None:
        """Offers the displays the ensemble can fill and shows the one chosen, or the
        slideshow while the ensemble lacks its service."""
        offered = offered_displays(self._controller.data_components)
        self._choice.set_offered(offered)
        key = self._display if self._display in offered else SLIDESHOW
        self._choice.set_current(key)
        if key == self._shown:
            return
        self._shown = key
        if key == SLIDESHOW:
            self._slide_area.setCurrentWidget(self._slide_label)
            return
        panel = self._panels.get(key)
        if panel is None:
            panel = make_panel(
                key,
                self._controller,
                self._tally,
                self._gnss_model,
                self._globe_renderer,
                self._waterfall_renderer,
            )
            if isinstance(panel, ConstellationPanel):
                panel.set_density(self._constellation_density)
            panel.apply_theme(self._theme)
            self._slide_area.add(panel)
            self._panels[key] = panel
        self._slide_area.setCurrentWidget(panel)
        # Drawn at once, not only at the next tick of the window.
        panel.render()

    def render(self) -> None:
        """Draws the panel shown in the place of the slideshow. The window calls it at
        its plot rate while this page is shown; the panels not shown are not drawn,
        as on the signal page."""
        panel = self._panels.get(self.shown_display())
        if panel is not None and self._stack.currentIndex() == 0:
            panel.render()

    # --- Station ----------------------------------------------------------------------

    def _show_station(self) -> None:
        station = self._directory.current()
        sid = self._controller.selected_sid
        if station is not None:
            self._name_label.setText(station.label)
        elif sid is not None:
            self._name_label.setText(tr("Station {sid:04X}").format(sid=sid))
        else:
            self._name_label.setText(tr("No station chosen"))
        self._gallery.set_station("" if sid is None else self._name_label.text())
        self._journaline_page.set_station("" if sid is None else self._name_label.text())
        self._show_filter()
        self._show_ensemble_line()
        self._logo_label.setPixmap(self._logo(station, LOGO_SIZE))
        # Only a station of the station page: there is where the favourites are shown.
        self._favourite_button.setEnabled(station is not None)
        self._favourite_button.setVisible(station is not None)
        self._favourite_button.setChecked(
            station is not None and self._directory.is_favourite(station)
        )
        if self._shown_slide is None:
            self._show_fallback()
        self._load_schedule()

    def _show_filter(self) -> None:
        """The filter chosen on the station page, with the number of stations it passes,
        as ◀ ▶ step through them: "Favourites (3)", "All stations (42)" or
        "5C · DR Deutschland (14)". Nothing while no station is known: the ensemble
        line then says where to find some."""
        directory = self._directory
        chosen = directory.filter
        name = tr("All stations") if chosen is None else directory.filter_name(chosen)
        self._filter_label.setText(f"{name} ({directory.counts().get(chosen, 0)})")
        self._filter_label.setVisible(bool(directory.all_stations()))

    def _show_ensemble_line(self) -> None:
        """Channel and ensemble of the station, with the other channels carrying it in
        brackets (`describe_other_channels`)."""
        station = self._directory.current()
        if station is None and self._controller.selected_sid is None:
            self._ensemble_label.setText(tr("Tap a station on the “Stations” page."))
            return
        channel = self._controller.channel if station is None else station.channel
        text = channel or tr("Manual frequency")
        others = describe_other_channels(self._controller.other_ensembles)
        if others:
            text = f"{text} {others}"
        ensemble = None if station is None else self._directory.ensemble_label(station.channel)
        self._ensemble_label.setText(f"{text} · {ensemble}" if ensemble else text)

    def _logo(self, station: Station | None, size: int) -> QPixmap:
        logo = None if station is None else self._directory.logo(station.sid)
        if logo is not None:
            return logo.scaled(
                size,
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        label = "?" if station is None else station.label
        return placeholder_pixmap(
            label, size, self._theme.color("raised"), self._theme.color("muted")
        )

    def _step(self, offset: int) -> None:
        station = self._directory.neighbour(offset)
        if station is not None:
            self._controller.play_service(station.channel, station.sid)

    def _on_favourite_clicked(self, checked: bool) -> None:
        station = self._directory.current()
        if station is not None:
            self._controller.set_favourite(*station.key, checked)

    def _on_play_clicked(self) -> None:
        if self._controller.is_receiving:
            self._controller.stop_reception()
        elif self._controller.selected_sid is None:
            self.stations_requested.emit()
        else:
            self._controller.start_reception()

    # --- Reception --------------------------------------------------------------------

    def _on_state_changed(self, state: ReceptionState) -> None:
        receiving = state in (ReceptionState.CONNECTING, ReceptionState.RECEIVING)
        icon = "stop" if receiving else "play"
        self._play_button.setIcon(symbol_icon(icon, self._theme.color("text"), ICON_SIZE))
        self._play_button.setEnabled(state is not ReceptionState.SCANNING)
        stepping = state in (ReceptionState.IDLE, ReceptionState.RECEIVING)
        self._previous_button.setEnabled(stepping)
        self._next_button.setEnabled(stepping)
        self._show_reception()

    def _reset_service(self) -> None:
        """Everything of the previous station goes: label, images, reception figures,
        and its Journaline, which the next station may not send."""
        if self.in_journaline():
            self.show_player()
        self._slides.clear()
        self._shown_slide = _REFRESH
        self._quality.reset()
        self._subchannel_text = None
        self._announcement = (None, None)
        self._switched_to_announcement = False
        self._show_announcement()
        self._dls_label.setText("")
        self._show_dl_plus("")
        self._show_station()
        self._update_slide()
        self._show_reception()

    def _on_subchannel_info(self, status: SubchannelStatus | None) -> None:
        if status is None or status.component is None or status.subchannel is None:
            self._subchannel_text = None
        else:
            kind = "DAB+" if status.component.is_dab_plus else "DAB"
            self._subchannel_text = f"{kind} · {status.subchannel.bitrate} kbit/s"
        self._show_reception()
        self._switched_to_announcement = status is not None and status.announcement is not None
        self._show_announcement()

    def _on_superframe(self, result: SuperframeResult) -> None:
        self._quality.add_superframe(result)
        self._show_reception()

    def _on_mp2_frame(self, result: Mp2FrameResult) -> None:
        self._quality.add_mp2_frame(result)
        self._show_reception()

    def _show_reception(self) -> None:
        state = self._controller.state
        bars = None
        if state is ReceptionState.IDLE:
            text = tr("No reception")
        elif state is ReceptionState.SCANNING:
            text = tr("Scan running")
        elif state is ReceptionState.CONNECTING:
            text = tr("Connecting …")
        elif self._subchannel_text is None:
            text = tr("Searching for the station …")
        else:
            bars = self._quality.bars
            quality = self._quality.describe()
            text = self._subchannel_text
            if quality is not None:
                # Translators: {quality} as `ui.touch.quality` judges it, e.g. "good".
                text = tr("Reception {quality} · {details}").format(quality=quality, details=text)
        self._bars.set_bars(bars)
        self._reception_label.setText(text)

    # --- Announcements ----------------------------------------------------------------

    def _on_announcement_changed(
        self, status: AnnouncementStatus | None, source: str | None
    ) -> None:
        self._announcement = (status, source)
        self._show_announcement()

    def _on_visit_changed(self, visit: AnnouncementVisit | None) -> None:
        self._visit = visit
        self._show_announcement()

    def _show_announcement(self) -> None:
        """While reception has gone to another ensemble for an announcement, that one;
        else the one running for the station, if any."""
        if self._visit is not None:
            notice = visit_notice(self._visit)
        else:
            status, source = self._announcement
            notice = announcement_notice(
                status,
                source,
                self._controller.selected_sid,
                switched=self._switched_to_announcement,
                switching_enabled=self._controller.announcement_switching,
            )
        self._notice.setVisible(notice is not None)
        if notice is None:
            return
        self._notice_title.setText(notice.title)
        self._notice_detail.setText(notice.detail)
        if self._notice.property("alarm") != notice.alarm:
            self._notice.setProperty("alarm", notice.alarm)
            for widget in (self._notice, self._notice_title, self._notice_detail):
                repolish(widget)

    def _on_notice_tapped(self) -> None:
        """A tap on the notice of another ensemble's announcement goes back at once."""
        self._controller.return_from_announcement()

    # --- Dynamic label and slideshow --------------------------------------------------

    def _show_dynamic_label(self, label: DynamicLabel) -> None:
        self._dls_label.setText(label.text)
        dl_plus = label.dl_plus
        if dl_plus is None:
            return  # keep what DL Plus said, see `ui.programme_tab.show_dynamic_label`
        parts = [dl_plus.artist, dl_plus.title] if dl_plus.item_running else []
        # Also where the label says it already: it soon moves on, the tags stay.
        self._show_dl_plus(" – ".join(part for part in parts if part))

    def _show_dl_plus(self, text: str) -> None:
        """Right beneath the dynamic label; without anything to say it takes no room."""
        self._dl_plus_label.setText(text)
        self._dl_plus_label.setVisible(bool(text))

    def _add_mot_object(self, mot_object: MotObject) -> None:
        now = self._clock.now()
        if mot_object.is_header_update:
            self._slides.apply_update(mot_object, now=now)
        else:
            # Received at the broadcast time, as the rail tells the time: a Raspberry Pi
            # without network does not know it. Until FIG 0/10 came, the computer's.
            self._slides.add(mot_object, received_at=now, now=now)
        self._update_slide()

    def _on_tick(self) -> None:
        now = self._clock.now()
        self._slides.expire(now)
        self._slides.tick(now)
        self._update_slide()
        self._show_guide_card()

    def _update_slide(self) -> None:
        """Shows the slide the station presents now (normal mode, clause 5.2), and
        what the store holds on the slide gallery while it is shown."""
        self._slides.take_presented()
        if self.in_gallery():
            self._gallery.refresh()
        slide = self._slides.live_slide()
        if slide == self._shown_slide:
            return
        self._shown_slide = slide
        if slide is None:
            self._show_fallback()
            return
        pixmap = QPixmap()
        if not pixmap.loadFromData(slide.data):
            # Clause 9.3: images that cannot be decoded are ignored.
            self._show_fallback()
            return
        show_slide(self._slide_label, pixmap, SLIDE_SIZE)

    def _show_fallback(self) -> None:
        """Without a slide, the station's logo (or initials) stands in for it."""
        self._slide_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._slide_label.setPixmap(self._logo(self._directory.current(), FALLBACK_LOGO_SIZE))

    # --- Programme guide --------------------------------------------------------------

    def _load_schedule(self) -> None:
        """Takes the schedule of the station playing from the guides."""
        sid = self._controller.selected_sid
        self._programmes = [] if sid is None else self._guide.programmes(sid)
        self._show_guide_card()

    def _show_guide_card(self) -> None:
        """The card shows what runs and what follows; for a station whose schedule has
        nothing now or later (or that has none) it gives the room back."""
        now = self._clock.now() or time.time()
        running, following = now_and_next(self._programmes, now)
        shown = running is not None or following is not None
        if shown:
            self._guide_card.show_programmes(running, following, now)
        self._guide_card.setVisible(shown)

    def _on_guide_card_tapped(self) -> None:
        sid = self._controller.selected_sid
        if sid is not None:
            self.guide_requested.emit(sid)
