"""The slides of the station playing, received since it was chosen: a tap on the slideshow
of the radio page leads here (`ui.touch.radio_page`). Master and detail side by side: on
the left the slide chosen as large as on the radio page, on the right the slides as
thumbnails, the newest first, swiped upwards for older ones.

Nothing is kept for it: it shows the holding buffer of the radio page
(`dab.slideshow.SlideStore`, up to 64 slides), which goes with the station. That is
enough on air, where every slide carries a name of its own (e.g.
"20260919_0244Z_WETTER_2.jpg"): a slide sent again under the same name replaces the
older version (clause 6.2.2 of ETSI TS 101 499) and moves to the top.
"""

from __future__ import annotations

import time

from PyQt6.QtCore import QRectF, QSize, Qt, QTimer
from PyQt6.QtGui import QIcon, QImage, QPainter, QPainterPath, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QScroller,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from dab.slideshow import Presentation, Slide, SlideStore, permitted_scale_factor
from i18n import tr, tr_noop, tr_plural
from ui.touch.theme import PANEL_RADIUS
from ui.touch.widgets import ElidedLabel, repolish, set_role

# The slide chosen: the size of the slide on the radio page, twice the standard 320 x 240.
DETAIL_SIZE = QSize(640, 480)
COLUMNS = 2
THUMBNAIL_SIZE = QSize(200, 150)

_EMPTY = tr_noop("No picture has arrived from this station yet.")
_UNREADABLE = tr_noop("The picture cannot be read.")


def show_slide(label: QLabel, pixmap: QPixmap, size: QSize) -> None:
    """Shows a decoded slide on `label`, an area of `size`, at the scale clause 9.2.3 of
    ETSI TS 101 499 permits (`permitted_scale_factor`): an integer factor keeps it
    sharp. A slide too big even at half size is cropped at the right and bottom
    (clause 9.1.3)."""
    factor = permitted_scale_factor(pixmap.width(), pixmap.height(), size.width(), size.height())
    if factor != 1.0:
        pixmap = pixmap.scaled(
            round(pixmap.width() * factor),
            round(pixmap.height() * factor),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setPixmap(rounded_slide(pixmap, size))


def rounded_slide(pixmap: QPixmap, size: QSize) -> QPixmap:
    """`pixmap` on an area of `size`, centred if it fits and else from the top left, with
    the corners of a panel cut off. The style sheet rounds only the background of the
    label; a slide filling the area covered its corners."""
    image = QImage(size, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    if pixmap.width() <= size.width() and pixmap.height() <= size.height():
        painter.drawPixmap(
            (size.width() - pixmap.width()) // 2, (size.height() - pixmap.height()) // 2, pixmap
        )
    else:
        painter.drawPixmap(0, 0, pixmap)
    rect = QRectF(image.rect())
    corners = QPainterPath()  # filled odd-even: the rectangle less the rounded one
    corners.addRect(rect)
    corners.addRoundedRect(rect, PANEL_RADIUS, PANEL_RADIUS)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
    painter.fillPath(corners, Qt.GlobalColor.black)
    painter.end()
    return QPixmap.fromImage(image)


def _clock(timestamp: float, seconds: bool = False) -> str:
    return time.strftime("%H:%M:%S" if seconds else "%H:%M", time.localtime(timestamp))


def count_text(count: int) -> str:
    if count == 0:
        return tr("no picture")
    return tr_plural("{count} picture", "{count} pictures", count).format(count=count)


def describe_slide(slide: Slide, store: SlideStore) -> str:
    """The line beneath the slide chosen: when it arrived, its category (sent, or taken
    from the file name) and, if the station has not shown it, why not."""
    parts = [tr("Received {time}").format(time=_clock(slide.received_at, seconds=True))]
    if slide.is_emergency_warning:
        parts.append(tr("Warning"))
    category = store.category_title(slide.category_key)
    if category:
        parts.append(category)
    state = store.presentation(slide)
    trigger = None if slide.trigger is None else slide.trigger.timestamp
    if state in (Presentation.SCHEDULED, Presentation.AWAITING_CLOCK) and trigger is not None:
        parts.append(tr("scheduled for {time}").format(time=_clock(trigger)))
    elif state in (Presentation.NO_TRIGGER, Presentation.MISSED):
        parts.append(tr("not shown by the station"))
    return " · ".join(parts)


class SlideGalleryPage(QWidget):
    """Master and detail of the slides in `store`. `refresh` follows the store; the
    radio page calls it whenever the store may have changed while this page is shown,
    and costs nothing as long as nothing did."""

    def __init__(self, store: SlideStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._store = store
        self._shown: tuple | None = None  # what the thumbnails were made from
        self._selected: tuple | None = None  # the key of the slide chosen
        self._slides: list[Slide] = []
        self._buttons: list[QToolButton] = []
        # Thumbnails by slide and content: a slide sent again under its name may differ.
        self._thumbnails: dict[tuple, QPixmap] = {}

        title = set_role(QLabel(tr("Slideshow")), "large")
        self._summary = set_role(ElidedLabel(), "muted")
        self.back_button = QPushButton(tr("Back"))
        header = QHBoxLayout()
        header.setSpacing(16)
        header.addWidget(title)
        header.addWidget(self._summary, stretch=1)
        header.addWidget(self.back_button)

        self._detail = set_role(QLabel(), "slide")
        self._detail.setFixedSize(DETAIL_SIZE)
        self._detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._detail.setWordWrap(True)
        self._caption = set_role(ElidedLabel(), "muted")
        self._caption.setFixedWidth(DETAIL_SIZE.width())
        detail = QVBoxLayout()
        detail.setSpacing(8)
        detail.addWidget(self._detail)
        detail.addWidget(self._caption)
        detail.addStretch(1)

        self._group = QButtonGroup(self)
        self._group.idClicked.connect(self._on_thumbnail_clicked)
        grid_widget = set_role(QWidget(), "transparent")
        self._grid = QGridLayout(grid_widget)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(12)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._scroll = QScrollArea()
        self._scroll.setWidget(grid_widget)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Swiped, as a finger expects; taps still reach the thumbnails.
        QScroller.grabGesture(
            self._scroll.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture
        )

        body = QHBoxLayout()
        body.setSpacing(24)
        body.addLayout(detail)
        body.addWidget(self._scroll, stretch=1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)
        layout.addLayout(header)
        layout.addLayout(body, stretch=1)
        self.set_station("")

    def set_station(self, name: str) -> None:
        self._station = name
        self._show_summary()

    def selected(self) -> Slide | None:
        return next((slide for slide in self._slides if slide.key == self._selected), None)

    def slides(self) -> list[Slide]:
        """The slides as the thumbnails show them, the newest first."""
        return list(self._slides)

    def thumbnail(self, index: int) -> QToolButton:
        return self._buttons[index]

    def open(self) -> None:
        """Shows the slides with the one on the radio page chosen, or else the newest."""
        live = self._store.live_slide()
        self._selected = None if live is None else live.key
        self._shown = None
        self.refresh()
        # Once the thumbnails have their places.
        QTimer.singleShot(0, self._reveal_selected)

    def refresh(self) -> None:
        slides = self._store.all_slides()
        live = self._store.live_slide()
        live_key = None if live is None else live.key
        shown = (
            tuple((s.key, s.received_at, self._store.presentation(s)) for s in slides),
            live_key,
        )
        if shown == self._shown:
            return
        self._shown = shown
        self._slides = slides
        keys = [slide.key for slide in slides]
        if self._selected not in keys:
            self._selected = keys[0] if keys else None
        self._fill(live_key)
        self._show_selected()
        self._show_summary()

    def _show_summary(self) -> None:
        count = count_text(len(self._store))
        self._summary.setText(f"{self._station} · {count}" if self._station else count)

    def _fill(self, live_key: tuple | None) -> None:
        for button in self._buttons:
            self._group.removeButton(button)
            button.deleteLater()
        self._buttons = []
        current = set()
        for index, slide in enumerate(self._slides):
            button = set_role(QToolButton(), "thumb")
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setIconSize(THUMBNAIL_SIZE)
            thumbnail = self._thumbnail(slide)
            if thumbnail is not None:
                button.setIcon(QIcon(thumbnail))
            moment = _clock(slide.received_at)
            # The slide standing on the radio page now.
            button.setText(f"{moment} ●" if slide.key == live_key else moment)
            button.setProperty("current", False)
            self._group.addButton(button, index)
            self._grid.addWidget(button, index // COLUMNS, index % COLUMNS)
            self._buttons.append(button)
            current.add((slide.key, len(slide.data), hash(slide.data)))
        # Only the thumbnails of slides still held.
        self._thumbnails = {
            key: pixmap for key, pixmap in self._thumbnails.items() if key in current
        }

    def _thumbnail(self, slide: Slide) -> QPixmap | None:
        key = (slide.key, len(slide.data), hash(slide.data))
        thumbnail = self._thumbnails.get(key)
        if thumbnail is None:
            pixmap = QPixmap()
            if not pixmap.loadFromData(slide.data):
                return None
            thumbnail = pixmap.scaled(
                THUMBNAIL_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._thumbnails[key] = thumbnail
        return thumbnail

    def _on_thumbnail_clicked(self, index: int) -> None:
        self._selected = self._slides[index].key
        self._show_selected()

    def _show_selected(self) -> None:
        for slide, button in zip(self._slides, self._buttons):
            current = slide.key == self._selected
            if button.property("current") != current:
                button.setProperty("current", current)
                repolish(button)
        slide = self.selected()
        if slide is None:
            self._detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._detail.setText(tr(_EMPTY))
            self._caption.setText("")
            return
        pixmap = QPixmap()
        if pixmap.loadFromData(slide.data):
            show_slide(self._detail, pixmap, DETAIL_SIZE)
        else:
            # Clause 9.3: images that cannot be decoded are ignored.
            self._detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._detail.setText(tr(_UNREADABLE))
        self._caption.setText(describe_slide(slide, self._store))

    def _reveal_selected(self) -> None:
        keys = [slide.key for slide in self._slides]
        if self._selected in keys:
            self._scroll.ensureWidgetVisible(self._buttons[keys.index(self._selected)])
