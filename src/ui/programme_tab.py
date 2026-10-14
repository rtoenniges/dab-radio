"""The "Programme" tab: dynamic label with DL Plus and the MOT slideshow."""

from __future__ import annotations

import html
import time

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPalette, QPixmap, QResizeEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from dab.dls import DlPlus, DlPlusContentType, DynamicLabel
from dab.mot import MotObject
from dab.slideshow import Presentation, Slide, SlideStore, permitted_scale_factor
from dab.timecode import BroadcastTime, ReferenceClock
from i18n import format_day_month, tr, tr_context, tr_noop

# Entry of the category selection for the slideshow's normal mode (ETSI TS 101 499,
# clause 5.2): the display follows the station. All other entries are categories of
# interactive mode. -1 collides neither with a CategoryID (1..255) nor with a group
# derived from a file name.
LIVE_VIEW = -1

# Interval at which the slideshow checks its time stamps. ExpireTime and TriggerTime
# are accurate to one second (ETSI TS 101 499, clauses 5.3.2 and 5.3.4); half a second
# keeps the reaction safely below that.
SLIDESHOW_TICK_MS = 500

_LIVE_VIEW_LABEL = tr_noop("Live (normal mode)")

_DL_PLUS_ALWAYS_SHOWN = (DlPlusContentType.ITEM_TITLE, DlPlusContentType.ITEM_ARTIST)
_NO_DL_PLUS_YET = tr_noop("No DL Plus received yet.")
_NO_IMAGE_YET = tr_noop("No image received yet.")


def _dl_plus_names() -> dict[DlPlusContentType, str]:
    """Names of the DL Plus content types (ETSI TS 102 980, annex A) in display order,
    in the language set: title and artist first, always shown, then whatever else the
    station tags. DUMMY and the types meant for RT+ only never become objects."""
    # Translators: the content types of DL Plus (ETSI TS 102 980, annex A), shown in
    # front of the part of the Dynamic Label the station tagged with them.
    return {
        DlPlusContentType.ITEM_TITLE: tr_context("DL Plus", "Title"),
        DlPlusContentType.ITEM_ARTIST: tr_context("DL Plus", "Artist"),
        DlPlusContentType.ITEM_ALBUM: tr_context("DL Plus", "Album"),
        DlPlusContentType.ITEM_TRACKNUMBER: tr_context("DL Plus", "Track number"),
        DlPlusContentType.ITEM_COMPOSITION: tr_context("DL Plus", "Composition"),
        DlPlusContentType.ITEM_MOVEMENT: tr_context("DL Plus", "Movement"),
        DlPlusContentType.ITEM_CONDUCTOR: tr_context("DL Plus", "Conductor"),
        DlPlusContentType.ITEM_COMPOSER: tr_context("DL Plus", "Composer"),
        DlPlusContentType.ITEM_BAND: tr_context("DL Plus", "Band/Orchestra"),
        DlPlusContentType.ITEM_COMMENT: tr_context("DL Plus", "Comment"),
        DlPlusContentType.ITEM_GENRE: tr_context("DL Plus", "Genre"),
        DlPlusContentType.PROGRAMME_NOW: tr_context("DL Plus", "Programme"),
        DlPlusContentType.PROGRAMME_PART: tr_context("DL Plus", "Programme part"),
        DlPlusContentType.PROGRAMME_HOST: tr_context("DL Plus", "Host"),
        DlPlusContentType.PROGRAMME_EDITORIAL_STAFF: tr_context("DL Plus", "Editorial staff"),
        DlPlusContentType.PROGRAMME_NEXT: tr_context("DL Plus", "Next"),
        DlPlusContentType.STATIONNAME_LONG: tr_context("DL Plus", "Station"),
        DlPlusContentType.STATIONNAME_SHORT: tr_context("DL Plus", "Station (short)"),
        DlPlusContentType.PROGRAMME_HOMEPAGE: tr_context("DL Plus", "Website"),
        DlPlusContentType.INFO_NEWS: tr_context("DL Plus", "News"),
        DlPlusContentType.INFO_NEWS_LOCAL: tr_context("DL Plus", "Local news"),
        DlPlusContentType.INFO_STOCKMARKET: tr_context("DL Plus", "Stock market"),
        DlPlusContentType.INFO_SPORT: tr_context("DL Plus", "Sport"),
        DlPlusContentType.INFO_LOTTERY: tr_context("DL Plus", "Lottery"),
        DlPlusContentType.INFO_HOROSCOPE: tr_context("DL Plus", "Horoscope"),
        DlPlusContentType.INFO_DAILY_DIVERSION: tr_context("DL Plus", "Daily diversion"),
        DlPlusContentType.INFO_HEALTH: tr_context("DL Plus", "Health"),
        DlPlusContentType.INFO_EVENT: tr_context("DL Plus", "Event"),
        DlPlusContentType.INFO_SCENE: tr_context("DL Plus", "Scene"),
        DlPlusContentType.INFO_CINEMA: tr_context("DL Plus", "Cinema"),
        DlPlusContentType.INFO_TV: tr_context("DL Plus", "TV"),
        DlPlusContentType.INFO_WEATHER: tr_context("DL Plus", "Weather"),
        DlPlusContentType.INFO_TRAFFIC: tr_context("DL Plus", "Traffic"),
        DlPlusContentType.INFO_ALARM: tr_context("DL Plus", "Alarm"),
        DlPlusContentType.INFO_ADVERTISEMENT: tr_context("DL Plus", "Advertisement"),
        DlPlusContentType.INFO_URL: tr_context("DL Plus", "Link"),
        DlPlusContentType.INFO_OTHER: tr_context("DL Plus", "Info"),
        DlPlusContentType.PHONE_HOTLINE: tr_context("DL Plus", "Phone (hotline)"),
        DlPlusContentType.PHONE_STUDIO: tr_context("DL Plus", "Phone (studio)"),
        DlPlusContentType.PHONE_OTHER: tr_context("DL Plus", "Phone"),
        DlPlusContentType.SMS_STUDIO: tr_context("DL Plus", "SMS (studio)"),
        DlPlusContentType.SMS_OTHER: "SMS",
        DlPlusContentType.EMAIL_HOTLINE: tr_context("DL Plus", "E-mail (hotline)"),
        DlPlusContentType.EMAIL_STUDIO: tr_context("DL Plus", "E-mail (studio)"),
        DlPlusContentType.EMAIL_OTHER: tr_context("DL Plus", "E-mail"),
        DlPlusContentType.MMS_OTHER: "MMS",
        DlPlusContentType.CHAT: tr_context("DL Plus", "Chat"),
        DlPlusContentType.CHAT_CENTER: tr_context("DL Plus", "Chat address"),
        DlPlusContentType.VOTE_QUESTION: tr_context("DL Plus", "Vote"),
        DlPlusContentType.VOTE_CENTRE: tr_context("DL Plus", "Vote at"),
        DlPlusContentType.PRIVATE_1: tr_context("DL Plus", "Private 1"),
        DlPlusContentType.PRIVATE_2: tr_context("DL Plus", "Private 2"),
        DlPlusContentType.PRIVATE_3: tr_context("DL Plus", "Private 3"),
        DlPlusContentType.DESCRIPTOR_PLACE: tr_context("DL Plus", "Place"),
        DlPlusContentType.DESCRIPTOR_APPOINTMENT: tr_context("DL Plus", "Appointment"),
        DlPlusContentType.DESCRIPTOR_IDENTIFIER: tr_context("DL Plus", "Identifier"),
        DlPlusContentType.DESCRIPTOR_PURCHASE: tr_context("DL Plus", "Purchase"),
        DlPlusContentType.DESCRIPTOR_GET_DATA: tr_context("DL Plus", "More at"),
    }


class ProgrammeTab(QWidget):
    """Dynamic label on top, below it the slideshow gallery with category selection,
    image list, the image itself and its link.

    Owns the holding buffer (`SlideStore`) and the slideshow reference clock. The
    main window only feeds in decoded objects and the broadcast time.
    """

    alert_changed = pyqtSignal(bool)  # an emergency warning is (no longer) on display

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._slide_store = SlideStore()
        self._reference_clock = ReferenceClock()
        self._slide_pixmap: QPixmap | None = None
        self._visible_slides: list[Slide] = []
        self._shown_key: tuple | None = None
        self._alert_active = False

        self._label_display = QLabel("-")
        self._label_display.setWordWrap(True)
        self._label_display.setStyleSheet("font-size: 18pt;")
        self._label_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label_display.setToolTip(
            tr(
                "Dynamic Label: the station's scrolling text with details of the title and "
                "the programme. It comes in the PAD field of the audio units, so in the same "
                "data stream as the audio."
            )
        )

        self._dl_plus_display = QLabel(tr(_NO_DL_PLUS_YET))
        self._dl_plus_display.setTextFormat(Qt.TextFormat.RichText)
        self._dl_plus_display.setWordWrap(True)
        self._dl_plus_display.setStyleSheet("font-size: 12pt;")
        self._dl_plus_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._dl_plus_display.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._dl_plus_display.setToolTip(
            tr(
                "DL Plus (ETSI TS 102 980): the station tags parts of the Dynamic Label with "
                "their meaning, such as title, artist or programme. The tags come as a "
                "command of their own after the Dynamic Label; a link bit ties them to it. A "
                "detail stays until the station replaces it, even if the Dynamic Label shows "
                "something else in the meantime. Title, artist and so on only apply to the "
                "current item: when a new one begins (item toggle bit) or none is running, "
                "for instance during presentation or news (item running bit = 0), they are "
                "dropped."
            )
        )

        self._reference_time_label = QLabel()
        self._reference_time_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._reference_time_label.setToolTip(
            tr(
                "The clock against which the slideshow checks expiry and trigger times: the "
                "time the ensemble sends (FIG 0/10), not the clock of this computer (ETSI TS "
                "101 499, annex D). Until the first time has been received, no image "
                "expires. The deviation shows how far the computer clock is from it - "
                "including the processing time until evaluation."
            )
        )
        self._show_reference_time(None)

        self._category_combo = QComboBox()
        self._category_combo.setToolTip(
            tr(
                "“Live” is normal mode: the display follows the station. The other entries "
                "are categories the station assigns (ETSI TS 101 499); in them you browse "
                "yourself, in the order the station sets. Entries marked “from file name” "
                "are not transmitted categories but derived from the file names of the "
                "images - some stations encode the grouping only there."
            )
        )
        self._category_combo.addItem(tr(_LIVE_VIEW_LABEL), LIVE_VIEW)
        self._category_combo.currentIndexChanged.connect(self._on_view_changed)

        self._slide_list = QListWidget()
        self._slide_list.setMinimumWidth(240)
        self._slide_list.setToolTip(
            tr(
                "Live: all stored images, the most recently received at the top. The image "
                "shown is always the one the station triggered last (TriggerTime, ETSI TS "
                "101 499, clause 5.3.2). Greyed out: stored, but not triggered for display - "
                "the reason follows. A click shows these images too. In a category: in the "
                "order of the image number the station assigns."
            )
        )
        self._slide_list.currentRowChanged.connect(self._on_slide_selected)

        self._slide_link = QLabel()
        self._slide_link.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._slide_link.setOpenExternalLinks(True)
        self._slide_link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self._slide_link.setToolTip(
            tr(
                "Some stations send an address with further information along with an image "
                "(ClickThroughURL). A click opens it in the browser. Only http and https "
                "addresses are shown (ETSI TS 101 499, clause 9.4.1)."
            )
        )

        self._slide_display = QLabel(tr(_NO_IMAGE_YET))
        self._slide_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._slide_display.setMinimumHeight(240)
        # Black background around images that do not fill the area (clause 9.1.3),
        # and no growing with the image: at 50 % reduction an image that is still too
        # big may be cropped at the right and bottom instead of inflating the window.
        self._slide_display.setStyleSheet("background-color: black; color: #bbbbbb;")
        self._slide_display.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._slide_display.setToolTip(
            tr(
                "MOT slideshow: images from the station (station logo, cover of the current "
                "title). An image is transmitted spread over many seconds and repeated "
                "cyclically - after switching on it therefore takes a while until the first "
                "one is complete."
            )
        )
        self._slide_info = QLabel("-")
        self._slide_info.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._alert_banner = QLabel(tr("Emergency warning from the station"))
        self._alert_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._alert_banner.setStyleSheet(
            "background-color: #b00020; color: white; font-weight: bold; padding: 6px;"
        )
        self._alert_banner.setToolTip(
            tr(
                "The station has marked the image on display as an emergency warning "
                "(parameter Alert = 0x01, ETSI TS 101 499, clause 6.2.10). Such an image "
                "brings the display back from a category to normal mode (“Live”)."
            )
        )
        self._alert_banner.setVisible(False)

        gallery_side = QVBoxLayout()
        gallery_side.addWidget(QLabel(tr("Category:")))
        gallery_side.addWidget(self._category_combo)
        gallery_side.addWidget(self._slide_list, stretch=1)

        image_side = QVBoxLayout()
        image_side.addWidget(self._alert_banner)
        image_side.addWidget(self._slide_display, stretch=1)
        image_side.addWidget(self._slide_info)
        image_side.addWidget(self._slide_link)

        gallery_row = QHBoxLayout()
        gallery_row.addLayout(gallery_side, stretch=1)
        gallery_row.addLayout(image_side, stretch=3)

        layout = QVBoxLayout()
        layout.addWidget(QLabel(tr("Dynamic Label:")))
        layout.addWidget(self._label_display)
        layout.addWidget(self._dl_plus_display)
        layout.addSpacing(12)
        slideshow_heading = QHBoxLayout()
        slideshow_heading.addWidget(QLabel(tr("Slideshow (MOT):")))
        slideshow_heading.addStretch(1)
        slideshow_heading.addWidget(self._reference_time_label)
        layout.addLayout(slideshow_heading)
        layout.addLayout(gallery_row, stretch=1)
        self.setLayout(layout)

        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(SLIDESHOW_TICK_MS)
        self._tick_timer.timeout.connect(self._on_tick)
        self._tick_timer.start()

    # --- Input from the main window ----------------------------------------------

    def reset(self) -> None:
        """Forgets label, DL Plus and images; they belong to the previously selected
        service."""
        self._label_display.setText("-")
        self._dl_plus_display.setText(tr(_NO_DL_PLUS_YET))
        self._slide_store.clear()
        self._visible_slides = []
        self._reset_category_combo()
        self._slide_list.clear()
        self._show_no_slide()

    def reset_clock(self) -> None:
        """The time comes from the ensemble; another ensemble sets it afresh."""
        self._reference_clock.reset()

    def show_dynamic_label(self, label: DynamicLabel) -> None:
        """Shows the text and its DL Plus objects.

        A label without DL Plus leaves the objects on display: within a service that
        only means the decoder started afresh after a reception gap and has not seen
        the next tags yet. A change of service clears them via `reset`.
        """
        self._label_display.setText(label.text or "-")
        if label.dl_plus is not None:
            self._show_dl_plus(label.dl_plus)

    def add_mot_object(self, mot_object: MotObject) -> None:
        """Takes a received image or header update.

        Whether the image is shown right away is decided by its TriggerTime (ETSI TS
        101 499, clause 5.3.2), see `_update_slideshow`.
        """
        now = self._reference_clock.now()
        if mot_object.is_header_update:
            changed = self._slide_store.apply_update(mot_object, now=now) is not None
        else:
            changed = self._slide_store.add(mot_object, now=now) is not None
        self._update_slideshow(changed=changed)

    def synchronise_clock(self, broadcast_time: BroadcastTime) -> None:
        """Sets the reference clock from FIG 0/10. Called in the GUI thread, where the
        clock is also read."""
        self._reference_clock.synchronise(broadcast_time)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._show_slide()  # fit the image to the new width

    # --- DL Plus ----------------------------------------------------------------------

    def _show_dl_plus(self, dl_plus: DlPlus) -> None:
        """Title and artist always, below them every other object the station tagged,
        and the two item flags that decide whether title and artist are valid."""
        rows = []
        for content_type, name in _dl_plus_names().items():
            text = dl_plus.objects.get(content_type)
            if content_type in _DL_PLUS_ALWAYS_SHOWN:
                value = f"<b>{html.escape(text)}</b>" if text else "-"
            elif text:
                value = html.escape(text)
            else:
                continue
            rows.append(f'<tr><td align="right">{html.escape(name)}:</td><td>{value}</td></tr>')

        flags = (
            f"Item Running = {int(dl_plus.item_running)}, Item Toggle = {int(dl_plus.item_toggle)}"
        )
        if dl_plus.item_running:
            state = tr("Item running ({flags})").format(flags=flags)
        else:
            state = tr("No title at the moment, e.g. presentation or news ({flags})").format(
                flags=flags
            )
        self._dl_plus_display.setText(
            f'<table align="center" cellspacing="0" cellpadding="2">{"".join(rows)}</table>'
            f'<p align="center" style="font-size: 9pt;">DL Plus: {html.escape(state)}</p>'
        )

    # --- Slideshow logic ------------------------------------------------------------

    def _on_tick(self) -> None:
        """Removes expired images from store and display (clause 5.3.4), triggers due
        images (5.3.2) and updates the reference time display."""
        now = self._reference_clock.now()
        self._show_reference_time(now)
        expired = bool(self._slide_store.expire(now))
        triggered = self._slide_store.tick(now)
        self._update_slideshow(changed=expired or triggered)

    def _update_slideshow(self, *, changed: bool) -> None:
        """Updates categories, list and display.

        In normal mode ("Live") the display jumps to every triggered image; the
        station dictates the sequence. A stored but untriggered image does not change
        the display. In a category the user browses; the chosen image then stays
        (clause 5.2) unless an emergency warning is triggered (6.2.10).
        """
        presented = self._slide_store.take_presented()
        if presented is None and not changed:
            return
        if presented is not None and presented.is_emergency_warning:
            self._return_to_normal_mode()
        self._refresh_categories()
        self._refresh_slide_list(
            jump=presented is not None and self._category_combo.currentData() == LIVE_VIEW
        )

    def _show_reference_time(self, now: float | None) -> None:
        if now is None:
            self._reference_time_label.setText(tr("Station time (FIG 0/10): not received yet"))
            return
        offset = time.time() - now
        seconds = f"{abs(offset):.1f}"
        if abs(offset) < 0.05:
            deviation = tr("computer clock in step")
        elif offset > 0:
            deviation = tr("computer clock {seconds} s ahead").format(seconds=seconds)
        else:
            deviation = tr("computer clock {seconds} s behind").format(seconds=seconds)
        self._reference_time_label.setText(
            tr("Station time (FIG 0/10): {time}, {deviation}").format(
                time=time.strftime("%H:%M:%S", time.localtime(now)), deviation=deviation
            )
        )

    @staticmethod
    def _format_moment(timestamp: float) -> str:
        """Time of day in local time; with the date if it is not today."""
        moment = time.localtime(timestamp)
        if time.strftime("%Y%m%d", moment) == time.strftime("%Y%m%d"):
            return time.strftime("%H:%M:%S", moment)
        return f"{format_day_month(moment)} {time.strftime('%H:%M:%S', moment)}"

    def _reset_category_combo(self) -> None:
        """Back to only the "Live" entry, without triggering a view change."""
        self._category_combo.blockSignals(True)
        self._category_combo.clear()
        self._category_combo.addItem(tr(_LIVE_VIEW_LABEL), LIVE_VIEW)
        self._category_combo.blockSignals(False)

    def _return_to_normal_mode(self) -> None:
        """Clause 6.2.10: an emergency warning ends interactive mode; the display
        returns to "Live", where it shows the warning. The user may switch back into
        a category afterwards."""
        self._category_combo.blockSignals(True)
        self._category_combo.setCurrentIndex(self._category_combo.findData(LIVE_VIEW))
        self._category_combo.blockSignals(False)

    def _show_alert(self, active: bool) -> None:
        """Banner for an emergency warning while it is on display. The main window
        also flags the tab so that the warning is not missed from other tabs."""
        self._alert_banner.setVisible(active)
        if active != self._alert_active:
            self._alert_active = active
            self.alert_changed.emit(active)

    def _on_view_changed(self, _index: int) -> None:
        self._refresh_slide_list(jump=True)

    def _refresh_categories(self) -> None:
        """Fills the category selection; "Live" always comes first."""
        entries = [(LIVE_VIEW, tr(_LIVE_VIEW_LABEL))] + [
            (key, self._view_label(key, title)) for key, title in self._slide_store.categories()
        ]
        existing = [
            (self._category_combo.itemData(i), self._category_combo.itemText(i))
            for i in range(self._category_combo.count())
        ]
        if existing == entries:
            return

        current = self._category_combo.currentData()
        self._category_combo.blockSignals(True)
        self._category_combo.clear()
        for key, label in entries:
            self._category_combo.addItem(label, key)
        index = self._category_combo.findData(current)
        self._category_combo.setCurrentIndex(max(index, 0))
        self._category_combo.blockSignals(False)
        if index < 0:
            # The chosen category no longer exists: back to "Live".
            self._refresh_slide_list(jump=True)

    @staticmethod
    def _view_label(key: int | str, title: str) -> str:
        # Groups derived from file names are not transmitted categories and are
        # marked as such.
        if isinstance(key, str):
            return tr("{title} (from file name)").format(title=title)
        return title

    def _refresh_slide_list(self, *, jump: bool = False) -> None:
        """Rebuilds the image list and picks the image to display.

        `jump` shows the image the view dictates: in normal mode the most recently
        triggered one, in a category the first. Otherwise the displayed image stays
        selected as long as it is still in the list. An image already on display is
        not redrawn: a re-sent image with the same ContentName must not affect the
        display (clause 6.2.2).
        """
        view = self._category_combo.currentData()
        live_view = view is None or view == LIVE_VIEW
        if live_view:
            self._visible_slides = self._slide_store.all_slides()
        else:
            self._visible_slides = self._slide_store.slides(view)

        held_color = self._slide_list.palette().color(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text
        )
        self._slide_list.blockSignals(True)
        self._slide_list.clear()
        for slide in self._visible_slides:
            moment = time.strftime("%H:%M:%S", time.localtime(slide.received_at))
            label = slide.name or tr("unnamed")
            if slide.is_emergency_warning:
                label = tr("[Warning] {name}").format(name=label)
            if slide.is_categorised and not live_view:
                label = f"{slide.slide_id:3d}  {label}"
            if slide.expire_time is not None:
                label = tr("{name}  (until {time})").format(
                    name=label, time=self._format_moment(slide.expire_time)
                )
            item = QListWidgetItem(f"{moment}  {label}")
            if live_view:
                # In normal mode what counts is whether and when the image is
                # triggered; whatever was not shown is greyed out with the reason.
                note = self._presentation_note(slide)
                if note:
                    short, explanation = note
                    item.setText(f"{item.text()}  - {short}")
                    item.setToolTip(explanation)
                    item.setForeground(held_color)
            self._slide_list.addItem(item)
        self._slide_list.blockSignals(False)

        if not self._visible_slides:
            self._show_no_slide()
            return

        keys = [slide.key for slide in self._visible_slides]
        live = self._slide_store.live_slide()
        default = (None if live is None else live.key) if live_view else keys[0]
        target = self._shown_key if not jump and self._shown_key in keys else default
        if target not in keys:
            self._slide_list.blockSignals(True)
            self._slide_list.setCurrentRow(-1)
            self._slide_list.blockSignals(False)
            self._show_no_slide(tr("No image is triggered for display at the moment."))
            return

        row = keys.index(target)
        self._slide_list.blockSignals(True)
        self._slide_list.setCurrentRow(row)
        self._slide_list.blockSignals(False)
        if target == self._shown_key:
            self._update_slide_position()
        else:
            self._on_slide_selected(row)

    def _presentation_note(self, slide: Slide) -> tuple[str, str] | None:
        """Why an image is not (yet) shown in normal mode, short for the list and in
        full for the tooltip, or None if it was triggered (clause 5.3.2, table 2)."""
        state = self._slide_store.presentation(slide)
        when = (
            ""
            if slide.trigger is None or slide.trigger.timestamp is None
            else self._format_moment(slide.trigger.timestamp)
        )
        if state is Presentation.SCHEDULED:
            return (
                tr("from {time}").format(time=when),
                tr("The station has scheduled the display for {time} (TriggerTime).").format(
                    time=when
                ),
            )
        if state is Presentation.AWAITING_CLOCK:
            return (
                tr("from {time}?").format(time=when),
                tr(
                    "The station has scheduled the display for {time}. Whether that moment is "
                    "still to come only shows once the station time (FIG 0/10) has been "
                    "received (clause 8.2)."
                ).format(time=when),
            )
        if state is Presentation.NO_TRIGGER:
            return tr("no trigger time"), tr(
                "The image came without a TriggerTime: stored, but not to be shown until the "
                "station triggers it with a header update."
            )
        if state is Presentation.MISSED:
            return (
                tr("missed ({time})").format(time=when),
                tr(
                    "The TriggerTime {time} had already passed on reception: stored, but not "
                    "to be shown."
                ).format(time=when),
            )
        return None

    def _on_slide_selected(self, row: int) -> None:
        if not 0 <= row < len(self._visible_slides):
            return
        slide = self._visible_slides[row]
        self._shown_key = slide.key
        self._show_alert(slide.is_emergency_warning)
        pixmap = QPixmap()
        if not pixmap.loadFromData(slide.data):
            # Clause 9.3: images that cannot be decoded are ignored.
            self._slide_pixmap = None
            self._slide_display.setPixmap(QPixmap())
            self._slide_display.setText(
                tr("Image cannot be displayed ({description})").format(description=slide.describe())
            )
            self._slide_info.setText("")
            self._slide_link.setText("")
            return

        self._slide_pixmap = pixmap
        self._show_slide()
        self._update_slide_position()

        # Only the ClickThroughURL is a page with further information. The
        # AlternativeLocationURL, by contrast, is an image source on the web (clause
        # 6.2.9) and does not belong here.
        url = slide.click_through_url
        if url:
            escaped = html.escape(url, quote=True)
            self._slide_link.setText(
                tr("More information: {link}").format(link=f'<a href="{escaped}">{escaped}</a>')
            )
        else:
            self._slide_link.setText("")

    def _update_slide_position(self) -> None:
        """Description plus "image x of y": when browsing a category it must always be
        clear which image is shown (clause 5.2)."""
        row = self._slide_list.currentRow()
        if not 0 <= row < len(self._visible_slides) or self._slide_pixmap is None:
            return
        slide = self._visible_slides[row]
        view = self._category_combo.currentData()
        # Translators: the slideshow's normal mode, in which the display follows the station.
        where = tr("Live") if view == LIVE_VIEW else self._slide_store.category_title(view)
        values = {
            "description": slide.describe(),
            "width": self._slide_pixmap.width(),
            "height": self._slide_pixmap.height(),
            "view": where,
            "number": row + 1,
            "count": len(self._visible_slides),
        }
        if slide.expire_time is None:
            text = tr("{description}, {width}x{height} pixels - {view}, image {number} of {count}")
        else:
            values["time"] = self._format_moment(slide.expire_time)
            text = tr(
                "{description}, {width}x{height} pixels - {view}, image {number} of {count}, "
                "valid until {time}"
            )
        self._slide_info.setText(text.format(**values))

    def _show_no_slide(self, message: str | None = None) -> None:
        self._slide_pixmap = None
        self._shown_key = None
        self._show_alert(False)
        self._slide_display.setPixmap(QPixmap())
        self._slide_display.setText(tr(_NO_IMAGE_YET) if message is None else message)
        self._slide_info.setText("-")
        self._slide_link.setText("")

    def _show_slide(self) -> None:
        """Shows the image at a scale permitted by clause 9.2.3.

        The aspect ratio is always preserved. Enlarging only from 150 % on, shrinking
        only to 50 %; if an image still does not fit, it is cropped at the right and
        bottom (clause 9.1.3), hence the top-left instead of centred alignment then.
        """
        if self._slide_pixmap is None:
            return
        pixmap = self._slide_pixmap
        area = self._slide_display.contentsRect().size()
        factor = permitted_scale_factor(
            pixmap.width(), pixmap.height(), area.width(), area.height()
        )
        if factor != 1.0:
            pixmap = pixmap.scaled(
                round(pixmap.width() * factor),
                round(pixmap.height() * factor),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        fits = pixmap.width() <= area.width() and pixmap.height() <= area.height()
        self._slide_display.setAlignment(
            Qt.AlignmentFlag.AlignCenter
            if fits
            else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._slide_display.setPixmap(pixmap)
