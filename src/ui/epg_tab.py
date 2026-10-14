"""The "Programme guide" tab: the electronic programme guides (SPI) of all channels."""

from __future__ import annotations

import html
import logging
import time
from dataclasses import replace

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPalette, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dab.channels import BAND_III_CHANNELS, channel_for_frequency
from dab.combined_guide import (
    Channel,
    CombinedGuide,
    EnsembleServices,
    GuideEntry,
    GuideGroup,
    channel_order,
    service_logos,
)
from dab.data_services import USER_APPLICATION_SPI, DataComponent
from dab.ensemble_cache import EnsembleCache
from dab.guide_cache import GuideCache
from dab.mot_directory import MotCarouselStatus, MotDirectory
from dab.spi import Programme, ProgrammeGuide, decode_spi_object, profile_names
from dsp.pipeline import CarouselObject, DataComponentStatus
from i18n import format_date, tr, tr_context, tr_plural, weekday

logger = logging.getLogger(__name__)

# Interval at which the highlight of the running programme moves on.
NOW_TICK_MS = 30_000

# Schedules that ended longer ago than this are no longer loaded from the stored guide.
STORED_SCHEDULE_MAX_AGE_SECONDS = 24 * 3600

_LOGO_SIZE = QSize(40, 40)
_SID_ROLE = Qt.ItemDataRole.UserRole
_PROGRAMME_ROLE = Qt.ItemDataRole.UserRole + 1
_GROUP_ROLE = Qt.ItemDataRole.UserRole + 2
_PLAYABLE_ROLE = Qt.ItemDataRole.UserRole + 3

# Columns of the service tree: name with logo, and the play button.
_NAME_COLUMN = 0
_PLAY_COLUMN = 1
_PLAY_COLUMN_WIDTH = 48
_PLAY_CIRCLE_DIAMETER = 26


def _day_title(start: float) -> str:
    local = time.localtime(start)
    return f"{weekday(local)}, {format_date(local)}"


def _duration_text(seconds: int) -> str:
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"


def _manual_frequency() -> str:
    # Translators: the channel of a manually set frequency, inside sentences such as
    # "from the programme guide of the manual frequency" (German: "der manuellen Frequenz").
    return tr("the manual frequency")


def _channel_names(channels: list[Channel]) -> str:
    """ "5C", "5C and 7A", "5C, 5D and 7A"; None is the manual frequency."""
    names = [channel or _manual_frequency() for channel in channels]
    if len(names) <= 1:
        return "".join(names)
    # Translators: the end of a list of channels: "5C, 5D and 7A".
    return tr("{first} and {last}").format(first=", ".join(names[:-1]), last=names[-1])


def _frequency_names(frequencies: tuple[int, ...], *, with_megahertz: bool = False) -> list[str]:
    """Frequencies in kHz as channels ("11D", with `with_megahertz` "11D (222.064
    MHz)"), outside Band III in MHz."""
    names = []
    for frequency in frequencies:
        megahertz = f"{frequency / 1000:.3f} MHz"
        channel = channel_for_frequency(frequency)
        if channel is None:
            names.append(megahertz)
        else:
            names.append(f"{channel} ({megahertz})" if with_megahertz else channel)
    return names


def _play_icon(color: QColor, size: QSize, pixel_ratio: float) -> QIcon:
    """A play triangle in a circle, centred in `size` (the icon size of the tree, which
    the logos need)."""
    pixmap = QPixmap(round(size.width() * pixel_ratio), round(size.height() * pixel_ratio))
    pixmap.setDevicePixelRatio(pixel_ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    x, y = size.width() / 2, size.height() / 2
    radius = _PLAY_CIRCLE_DIAMETER / 2
    painter.setPen(QPen(color, 1.5))
    painter.drawEllipse(QRectF(x - radius, y - radius, 2 * radius, 2 * radius))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    # Shifted to the right so that the triangle looks centred.
    painter.drawPolygon(
        QPolygonF([QPointF(x - 4, y - 6.5), QPointF(x - 4, y + 6.5), QPointF(x + 7, y)])
    )
    painter.end()
    return QIcon(pixmap)


def _logo_pixmap(data: bytes | None) -> QPixmap | None:
    if data is None:
        return None
    pixmap = QPixmap()
    if not pixmap.loadFromData(data):
        return None
    return pixmap


def spi_component(components: list[DataComponent]) -> DataComponent | None:
    """The data component carrying the programme guide, if the ensemble has one."""
    return next(
        (c for c in components if USER_APPLICATION_SPI in c.app_types and c.decodable), None
    )


class EpgTab(QWidget):
    """On the left the services of all programme guides with their logos, grouped by
    the channel they are broadcast on; on the right the programmes of the chosen one,
    grouped by day, below the description.

    A guide may cover more than its own ensemble (EPG Deutschland on 5C also lists the
    services of 5D), so the tab shows the guides of all channels together
    (`dab.combined_guide`). During reception the guide of the received channel grows
    with the completed objects of the MOT carousel while the SPI service is decoded
    (see the "Data services" tab); the button at the top starts that decoding.

    Like the service list, every guide is stored per channel (`cache`); the controller
    stores what is received (`ui.receiver_controller`, so that every surface collects
    guides), this tab loads them again whenever a channel is shown and only reports what
    was stored (`note_stored`). The channel lists (`ensembles`) tell which channel a
    service belongs to. With a manual frequency nothing is stored.
    """

    component_requested = pyqtSignal(object)  # SCId of the SPI component to decode
    # Play button: channel (None = the manual frequency) and SId of the service.
    service_requested = pyqtSignal(object, int)
    # dict[int, QIcon]: SId -> logo for the service list of the channel on display (see
    # `dab.combined_guide.service_logos`); for every channel shown and on every change.
    logos_changed = pyqtSignal(object)

    def __init__(
        self,
        cache: GuideCache | None = None,
        ensembles: EnsembleCache | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cache = cache
        self._ensembles = ensembles
        self._channel: Channel = None  # channel received or selected, None = manual
        self._live = False  # reception is running on that channel
        self._stored_at: dict[str, float] = {}  # last change of the stored guide by channel
        self._store_error: str | None = None
        self._guides = CombinedGuide()
        # The ensemble on the received channel as the FIC describes it; it may not be
        # stored yet.
        self._ensemble: EnsembleServices | None = None
        # The service labels of the FIC stored with each guide, by channel.
        self._stored_labels: dict[str, dict[int, str]] = {}
        self._preferred_sid: int | None = None  # the service selected in the side panel
        self._shown_sid: int | None = None
        self._collapsed: set[tuple[Channel, int | None]] = set()  # keys of closed groups
        # What the lists currently show, so that they are only rebuilt when it changes:
        # every rebuild loses the scroll position, and during reception new objects of
        # the carousel keep arriving.
        self._listed_groups: list[GuideGroup] = []
        self._listed_programmes: tuple[int | None, list[Programme]] = (None, [])
        self._announced_logos: dict[int, bytes] | None = None  # last sent by `logos_changed`
        self._spi: DataComponent | None = None  # SPI component of the ensemble
        # SCId of the SPI component being decoded, whose carousel the guide takes.
        self._decoded_spi: int | None = None
        self._carousel: MotCarouselStatus | None = None

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._receive_button = QPushButton(tr("Receive programme guide"))
        self._receive_button.setToolTip(
            tr(
                "Decodes the data service with the programme guide (SPI, ETSI TS 102 818/102 "
                "371) in addition to the audio. It comes in packet mode in a subchannel of its "
                "own as an MOT carousel: station logos, service information and a schedule per "
                "station and day. Until everything is there, at least one cycle of the "
                "carousel passes, often a few minutes."
            )
        )
        self._receive_button.clicked.connect(self._on_receive_clicked)

        self._service_tree = QTreeWidget()
        self._service_tree.setColumnCount(2)
        self._service_tree.setHeaderHidden(True)
        header = self._service_tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(_NAME_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_PLAY_COLUMN, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(_PLAY_COLUMN, _PLAY_COLUMN_WIDTH)
        self._service_tree.setIconSize(_LOGO_SIZE)
        self._service_tree.setIndentation(8)  # the logos set the services apart already
        self._service_tree.setMinimumWidth(260)
        self._service_tree.setMouseTracking(True)  # for the hand over the play button
        self._service_tree.currentItemChanged.connect(self._on_service_changed)
        self._service_tree.itemCollapsed.connect(self._on_group_collapsed)
        self._service_tree.itemExpanded.connect(self._on_group_expanded)
        self._service_tree.itemClicked.connect(self._on_service_clicked)
        self._service_tree.itemEntered.connect(self._on_item_entered)
        self._service_tree.viewportEntered.connect(self._service_tree.viewport().unsetCursor)
        ratio = self.devicePixelRatioF()
        palette = self.palette()
        self._play_icon = _play_icon(palette.color(QPalette.ColorRole.Highlight), _LOGO_SIZE, ratio)
        self._play_icon_disabled = _play_icon(
            palette.color(QPalette.ColorRole.PlaceholderText), _LOGO_SIZE, ratio
        )
        self._service_tree.setToolTip(
            tr(
                "The stations of all stored programme guides with their logos, ordered by the "
                "channel they run on. A programme guide can cover more than its own ensemble: "
                "EPG Deutschland also names stations of other ensembles. Stations without a "
                "schedule are grey; some ensembles only send logos and names. The play "
                "symbol tunes to the channel and plays the station."
            )
        )

        self._heading = QLabel()
        heading_font = QFont(self._heading.font())
        heading_font.setPointSizeF(heading_font.pointSizeF() * 1.3)
        heading_font.setBold(True)
        self._heading.setFont(heading_font)
        self._logo_label = QLabel()

        self._programme_tree = QTreeWidget()
        self._programme_tree.setHeaderLabels(
            [
                # Translators: the columns of the schedule in the programme guide.
                tr_context("schedule column", "Start"),
                tr_context("schedule column", "Duration"),
                tr_context("schedule column", "Programme"),
            ]
        )
        self._programme_tree.setRootIsDecorated(True)
        self._programme_tree.setUniformRowHeights(True)
        self._programme_tree.currentItemChanged.connect(self._on_programme_changed)
        self._programme_tree.setToolTip(
            tr(
                "The schedule by day, times in the time zone of this computer. The current "
                "programme is in bold. The stations transmit the times in UTC with the offset "
                "to local time (ETSI TS 102 371, clause 5.4.5.2)."
            )
        )
        self._description = QTextBrowser()
        self._description.setOpenExternalLinks(True)

        top = QHBoxLayout()
        top.addWidget(self._status_label, stretch=1)
        top.addWidget(self._receive_button)

        heading_row = QHBoxLayout()
        heading_row.addWidget(self._logo_label)
        heading_row.addWidget(self._heading, stretch=1)

        schedule_side = QWidget()
        schedule_layout = QVBoxLayout(schedule_side)
        schedule_layout.setContentsMargins(0, 0, 0, 0)
        schedule_layout.addLayout(heading_row)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._programme_tree)
        splitter.addWidget(self._description)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        schedule_layout.addWidget(splitter, stretch=1)

        columns = QSplitter(Qt.Orientation.Horizontal)
        columns.addWidget(self._service_tree)
        columns.addWidget(schedule_side)
        columns.setStretchFactor(0, 1)
        columns.setStretchFactor(1, 3)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(columns, stretch=1)
        self.setLayout(layout)

        self._now_timer = QTimer(self)
        self._now_timer.setInterval(NOW_TICK_MS)
        self._now_timer.timeout.connect(self._update_running_programme)
        self._now_timer.start()

        self.show_channel(None, live=False)

    # --- Input from the main window ----------------------------------------------

    def show_channel(self, channel: Channel, *, live: bool) -> None:
        """Shows the stored guides of all channels, and with `live` from now on also
        what the reception on `channel` (None = manual frequency) delivers."""
        self._channel = channel
        self._live = live
        self._ensemble = None
        self._spi = None
        self._decoded_spi = None
        self._carousel = None
        self._store_error = None
        # Which guide's logo comes first depends on the channel.
        self._announced_logos = None
        self._load_stored_guides()
        self._refresh_services()
        self._update_status()

    def reception_stopped(self) -> None:
        """The guide on display stays; it is stored already."""
        self._live = False
        self._decoded_spi = None
        self._carousel = None
        self._update_status()

    def set_ensemble(self, label: str | None, services: dict[int, str]) -> None:
        """The ensemble on the received channel as the FIC describes it: which services
        belong to the channel, and their names where the guides do not give any."""
        received = self._ensemble or EnsembleServices(None)
        self._ensemble = replace(received, label=label, services=dict(services))
        if not self._guides.empty:
            self._refresh_services()

    def set_frequency_information(
        self, eid: int | None, frequencies: dict[int, tuple[int, ...]]
    ) -> None:
        """The EId of the received ensemble (which ensemble the received channel
        carries) and the frequencies it announces for ensembles (FIG 0/21)."""
        received = self._ensemble or EnsembleServices(None)
        self._ensemble = replace(received, eid=eid, frequencies=dict(frequencies))
        if not self._guides.empty:
            self._refresh_services()

    def set_selected_service(self, sid: int | None) -> None:
        """Shows the service selected in the side panel, if a guide knows it."""
        self._preferred_sid = sid
        item = None if sid is None else self._find_service_item(sid)
        if item is not None:
            self._show_service(sid)
            self._service_tree.scrollToItem(item)

    def update_components(self, components: list[DataComponent]) -> None:
        self._spi = spi_component(components)
        self._update_status()

    def set_data_decoding(self, statuses: tuple[DataComponentStatus, ...]) -> None:
        """Which data components are being decoded: the guide only grows while one of
        them is an SPI component, and only from that one."""
        decoded = next(
            (
                status.scid
                for status in statuses
                if status.component is not None
                and USER_APPLICATION_SPI in status.component.app_types
            ),
            None,
        )
        if decoded != self._decoded_spi:
            self._carousel = None
        self._decoded_spi = decoded
        self._update_status()

    def update_carousel(self, scid: int, status: MotCarouselStatus) -> None:
        if scid == self._decoded_spi:
            self._carousel = status
            self._update_status()

    def set_directory(self, scid: int, directory: MotDirectory) -> None:
        """A new version of the carousel: objects it no longer lists are outdated (the
        controller removes them from the stored guide as well)."""
        if scid != self._decoded_spi:
            return
        names = {entry.header.content_name for entry in directory.entries}
        if self._received_guide().retain({name for name in names if name}):
            self._refresh_services()

    def add_object(self, scid: int, carousel_object: CarouselObject) -> None:
        if scid == self._decoded_spi and self._received_guide().add(
            carousel_object.mot_object, carousel_object.spi
        ):
            self._refresh_services()
            self._update_status()

    def note_stored(self, channel: str, when: float, _services: bool = True) -> None:
        """The controller stored an object of the guide received on `channel`."""
        self._stored_at[channel] = when
        self._store_error = None
        self._update_status()

    def show_store_error(self, message: str) -> None:
        """The controller could not store the guide received."""
        self._store_error = message
        self._update_status()

    def _received_guide(self) -> ProgrammeGuide:
        """The guide of the channel received (or last received)."""
        return self._guides.guide(self._channel)

    # --- Stored guides --------------------------------------------------------------

    def _load_stored_guides(self) -> None:
        """Takes over the stored guides of all channels; schedules that ended more than
        a day ago are left out and deleted."""
        self._guides.clear()
        self._stored_at = {}
        self._stored_labels = {}
        if self._cache is None:
            return
        for channel in self._cache.channels():
            try:
                stored = self._cache.load(channel)
            except (OSError, ValueError) as exc:
                logger.warning("Stored guide of %s not loaded: %s", channel, exc)
                continue
            if stored is None:
                continue
            guide = self._guides.guide(channel)
            for mot_object in stored.objects:
                guide.add(mot_object, decode_spi_object(mot_object))
            self._stored_at[channel] = stored.updated_at
            self._stored_labels[channel] = stored.labels
            if guide.drop_schedules_ended_before(time.time() - STORED_SCHEDULE_MAX_AGE_SECONDS):
                self._store_retained(channel)

    def _store_retained(self, channel: Channel) -> None:
        """Deletes from the stored guide of a channel what the guide on display no
        longer holds."""
        if self._cache is None or channel is None:
            return
        now = time.time()
        try:
            self._cache.retain(channel, self._guides.guide(channel).content_names(), now)
        except OSError as exc:
            self._storing_failed(exc)

    def _storing_failed(self, exc: OSError) -> None:
        if self._store_error is None:
            logger.warning("Guide of %s not stored: %s", self._channel, exc)
        self._store_error = str(exc)
        self._update_status()

    # --- Display --------------------------------------------------------------------

    def _on_receive_clicked(self) -> None:
        if self._spi is not None:
            self.component_requested.emit(self._spi.scid)

    def _update_status(self) -> None:
        self._receive_button.setVisible(
            self._live and self._spi is not None and self._decoded_spi is None
        )
        if self._decoded_spi is not None:
            text = self._receiving_text()
        else:
            parts = []
            if self._stored_at:
                parts.append(self._stored_text())
            if self._live and self._spi is not None:
                parts.append(self._announcement_text())
            elif self._live and self._received_guide().empty:
                parts.append(
                    tr(
                        "This ensemble announces no programme guide (FIG 0/13 with SPI), or it "
                        "has not appeared in the FIC yet."
                    )
                )
            elif not self._stored_at:
                parts.append(tr("No programme guide has been stored yet."))
            text = " ".join(parts)
        if self._store_error is not None:
            text += " " + tr("Storing not possible: {error}").format(error=self._store_error)
        self._status_label.setText(text)

    def _stored_text(self) -> str:
        """Which channels a guide is stored for, and when each was last added to."""
        stored = []
        for channel, when in sorted(
            self._stored_at.items(), key=lambda item: channel_order(item[0])
        ):
            moment = time.localtime(when)
            stored.append(f"{channel} ({format_date(moment)} {time.strftime('%H:%M', moment)})")
        return tr_plural(
            "Stored programme guide, last added to: {channels}.",
            "Stored programme guides, last added to: {channels}.",
            len(stored),
        ).format(channels=_channel_names(stored))

    def _announcement_text(self) -> str:
        spi_data = next(
            app.data for app in self._spi.applications if app.app_type == USER_APPLICATION_SPI
        )
        profiles = ", ".join(profile_names(spi_data))
        if profiles:
            text = tr("The ensemble broadcasts the programme guide “{name}” (profile: {profiles}).")
        else:
            text = tr("The ensemble broadcasts the programme guide “{name}”.")
        return text.format(name=self._spi.label, profiles=profiles)

    def _receiving_text(self) -> str:
        parts = [tr("Receiving the programme guide.")]
        carousel = self._carousel
        if carousel is not None and carousel.directory_objects is not None:
            parts.append(
                tr("Carousel: {complete} of {files} files ({received} of {total} kB).").format(
                    complete=carousel.complete_objects,
                    files=carousel.directory_objects,
                    received=f"{carousel.received_bytes / 1000:.0f}",
                    total=f"{carousel.directory_bytes / 1000:.0f}",
                )
            )
        elif carousel is not None:
            parts.append(tr("Waiting for the MOT directory..."))
        guide = self._received_guide()
        if not guide.empty:
            if guide.has_service_information:
                text = tr("Schedules: {count}, service information: yes.")
            else:
                text = tr("Schedules: {count}, service information: not yet.")
            parts.append(text.format(count=guide.schedule_count))
        if self._channel is None:
            parts.append(tr("Nothing is stored with a manual frequency."))
        else:
            parts.append(tr("Stored for channel {channel}.").format(channel=self._channel))
        return " ".join(parts)

    def _known_ensembles(self) -> dict[Channel, EnsembleServices]:
        """The ensembles of all stored channels; for the received channel what the FIC
        delivers, each part as soon as it has arrived. A channel whose service list was
        never stored is known from the labels stored with its guide."""
        ensembles: dict[Channel, EnsembleServices] = {
            channel: EnsembleServices(None, labels)
            for channel, labels in self._stored_labels.items()
        }
        if self._ensembles is not None:
            for channel, info in self._ensembles.entries().items():
                ensembles[channel] = EnsembleServices(
                    info.label, dict(info.services), info.eid, dict(info.frequencies)
                )
        received = self._ensemble
        if received is not None:
            stored = ensembles.get(self._channel) or EnsembleServices(None)
            ensembles[self._channel] = EnsembleServices(
                received.label or stored.label,
                received.services or stored.services,
                stored.eid if received.eid is None else received.eid,
                received.frequencies or stored.frequencies,
            )
        return ensembles

    def _refresh_services(self) -> None:
        groups = self._guides.groups(self._known_ensembles())
        rebuilt = groups != self._listed_groups
        scroll_position = self._service_tree.verticalScrollBar().value()
        if rebuilt:
            self._listed_groups = groups
            self._fill_service_tree(groups)
        self._announce_logos(groups)

        previous = self._shown_sid
        wanted = previous
        if wanted is None or self._find_service_item(wanted) is None:
            wanted = self._preferred_sid
        if wanted is None or self._find_service_item(wanted) is None:
            wanted = self._default_sid(groups)
        if wanted is not None:
            self._show_service(wanted)
        else:
            self._shown_sid = None
            self._refresh_programmes()
        if rebuilt and wanted == previous:
            # Selecting the new item scrolled to it; the list stays where it was.
            self._service_tree.verticalScrollBar().setValue(scroll_position)

    def _announce_logos(self, groups: list[GuideGroup]) -> None:
        logos = service_logos(groups, self._channel)
        if logos == self._announced_logos:
            return
        self._announced_logos = logos
        pixmaps = {sid: _logo_pixmap(data) for sid, data in logos.items()}
        self.logos_changed.emit(
            {sid: QIcon(pixmap) for sid, pixmap in pixmaps.items() if pixmap is not None}
        )

    def _default_sid(self, groups: list[GuideGroup]) -> int | None:
        """The first service with a schedule, preferably one of the channel on
        display; else the first one at all."""
        own = [group for group in groups if group.key == (self._channel, None)]
        entries = [entry for group in own + groups for entry in group.entries]
        if not entries:
            return None
        return next((entry.sid for entry in entries if entry.has_schedule), entries[0].sid)

    def _fill_service_tree(self, groups: list[GuideGroup]) -> None:
        tree = self._service_tree
        tree.blockSignals(True)
        tree.clear()
        for group in groups:
            parent = QTreeWidgetItem([self._group_title(group)])
            parent.setData(0, _GROUP_ROLE, group.key)
            parent.setFlags(Qt.ItemFlag.ItemIsEnabled)
            parent.setToolTip(0, self._group_tooltip(group))
            font = parent.font(0)
            font.setBold(True)
            parent.setFont(0, font)
            tree.addTopLevelItem(parent)
            parent.setFirstColumnSpanned(True)
            for entry in group.entries:
                parent.addChild(self._service_item(group, entry))
            parent.setExpanded(group.key not in self._collapsed)
        tree.blockSignals(False)

    @staticmethod
    def _group_title(group: GuideGroup) -> str:
        if group.eid is not None and group.frequencies:
            where = ", ".join(_frequency_names(group.frequencies))
            return tr("Ensemble {eid} (announced on {channels})").format(
                eid=f"{group.eid:04X}", channels=where
            )
        if group.eid is not None:
            return tr("Ensemble {eid} (no channel known)").format(eid=f"{group.eid:04X}")
        channel = tr("Manual frequency") if group.channel is None else group.channel
        return f"{channel} - {group.label}" if group.label else channel

    @staticmethod
    def _group_tooltip(group: GuideGroup) -> str:
        origins = _channel_names(list(group.origins))
        if group.eid is not None:
            text = tr(
                "According to the programme guide of {origins}, these stations run in the "
                "ensemble with the identifier (EId) {eid}. It was not received here on any "
                "channel."
            ).format(origins=origins, eid=f"{group.eid:04X}")
            if group.frequencies:
                frequencies = ", ".join(_frequency_names(group.frequencies, with_megahertz=True))
                text += " " + tr(
                    "According to the FIC of {channels} (FIG 0/21) it broadcasts on "
                    "{frequencies}. That is an announcement, not a measurement: the coverage "
                    "area may be out of reach, and the same channel may carry another ensemble "
                    "here."
                ).format(channels=_channel_names(list(group.announced_by)), frequencies=frequencies)
            else:
                text += " " + tr("No ensemble received announces its frequency (FIG 0/21).")
            return text
        if group.channel is None:
            return tr(
                "The stations of the manually set frequency, from the programme guide of {origins}."
            ).format(origins=origins)
        frequency = BAND_III_CHANNELS.get(group.channel)
        if frequency is None:
            return tr(
                "The stations on channel {channel}, from the programme guide of {origins}."
            ).format(channel=group.channel, origins=origins)
        return tr(
            "The stations on channel {channel} ({frequency} MHz), from the programme guide of "
            "{origins}."
        ).format(channel=group.channel, frequency=f"{frequency:.3f}", origins=origins)

    def _service_item(self, group: GuideGroup, entry: GuideEntry) -> QTreeWidgetItem:
        item = QTreeWidgetItem([entry.name])
        item.setData(0, _SID_ROLE, entry.sid)
        pixmap = _logo_pixmap(entry.logo)
        if pixmap is not None:
            item.setIcon(0, QIcon(pixmap))
        if not entry.has_schedule:
            item.setForeground(0, self.palette().brush(QPalette.ColorRole.PlaceholderText))
        if entry.has_schedule:
            tooltip = tr("Service ID (SId): {sid}, with schedule")
        else:
            tooltip = tr("Service ID (SId): {sid}, without schedule")
        tooltip = tooltip.format(sid=f"{entry.sid:04X}")
        if entry.origins != (group.channel,) or group.eid is not None:
            tooltip += "\n" + tr("From the programme guide of {origins}").format(
                origins=_channel_names(list(entry.origins))
            )
        item.setToolTip(0, tooltip)
        self._set_play_button(item, group, entry)
        return item

    def _set_play_button(self, item: QTreeWidgetItem, group: GuideGroup, entry: GuideEntry) -> None:
        """The play button of a service: active where it can be tuned to, i.e. on the
        service list of its channel; disabled with the reason where the channel is known
        but the list does not have the service; missing for an unknown ensemble.

        The group of the manual frequency only exists while it is set: its guide is
        not stored, and showing another channel reloads the stored guides."""
        if group.eid is not None:
            return
        item.setData(_PLAY_COLUMN, _PLAYABLE_ROLE, entry.listed)
        where = _manual_frequency() if group.channel is None else group.channel
        if not entry.listed:
            item.setIcon(_PLAY_COLUMN, self._play_icon_disabled)
            tooltip = tr(
                "Playback not possible: not in the station list of {channel}. The programme "
                "guide names the station, but the ensemble does not broadcast it (or the "
                "stored list is out of date)."
            ).format(channel=where)
        else:
            item.setIcon(_PLAY_COLUMN, self._play_icon)
            if group.channel is None:
                tooltip = tr("Play {name}").format(name=entry.name)
            else:
                tooltip = tr("Play {name}: tunes to channel {channel}").format(
                    name=entry.name, channel=group.channel
                )
        item.setToolTip(_PLAY_COLUMN, tooltip)

    def _find_service_item(self, sid: int) -> QTreeWidgetItem | None:
        """The item of a service. A service can be listed under several channels: the
        current item comes first, then the one under the channel on display."""
        current = self._service_tree.currentItem()
        if current is not None and current.data(0, _SID_ROLE) == sid:
            return current
        found = None
        for index in range(self._service_tree.topLevelItemCount()):
            group = self._service_tree.topLevelItem(index)
            for row in range(group.childCount()):
                item = group.child(row)
                if item.data(0, _SID_ROLE) != sid:
                    continue
                if group.data(0, _GROUP_ROLE) == (self._channel, None):
                    return item
                found = found or item
        return found

    def _show_service(self, sid: int) -> None:
        item = self._find_service_item(sid)
        if item is None:
            return
        if self._service_tree.currentItem() is not item:
            self._service_tree.blockSignals(True)
            self._service_tree.setCurrentItem(item)
            self._service_tree.blockSignals(False)
        self._shown_sid = sid
        self._refresh_programmes()

    def _on_service_changed(self, current: QTreeWidgetItem | None, _previous) -> None:
        sid = None if current is None else current.data(0, _SID_ROLE)
        if sid is not None:
            self._shown_sid = sid
            self._refresh_programmes()

    def _on_service_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        if column != _PLAY_COLUMN or not item.data(_PLAY_COLUMN, _PLAYABLE_ROLE):
            return
        channel, _ = item.parent().data(0, _GROUP_ROLE)
        sid = item.data(0, _SID_ROLE)
        # Only after the click has been handled: tuning rebuilds this tree.
        QTimer.singleShot(0, lambda: self.service_requested.emit(channel, sid))

    def _on_item_entered(self, item: QTreeWidgetItem, column: int) -> None:
        viewport = self._service_tree.viewport()
        if column == _PLAY_COLUMN and item.data(_PLAY_COLUMN, _PLAYABLE_ROLE):
            viewport.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            viewport.unsetCursor()

    def _on_group_collapsed(self, item: QTreeWidgetItem) -> None:
        self._collapsed.add(item.data(0, _GROUP_ROLE))

    def _on_group_expanded(self, item: QTreeWidgetItem) -> None:
        self._collapsed.discard(item.data(0, _GROUP_ROLE))

    def _refresh_programmes(self) -> None:
        """Shows the programmes of the chosen service, rebuilding the list only if they
        changed. For the same service the scroll position and selection stay; a new
        service opens at the running programme."""
        sid = self._shown_sid
        item = None if sid is None else self._find_service_item(sid)
        self._heading.setText("" if item is None else item.text(0))
        icon = None if item is None else item.icon(0)
        pixmap = None if icon is None or icon.isNull() else icon.pixmap(_LOGO_SIZE)
        self._logo_label.setPixmap(pixmap if pixmap is not None else QPixmap())

        programmes = [] if sid is None else self._guides.programmes(sid)
        listed_sid, listed = self._listed_programmes
        if sid == listed_sid and programmes == listed and self._programme_tree.topLevelItemCount():
            return
        self._listed_programmes = (sid, programmes)
        self._show_programmes(programmes, same_service=sid == listed_sid)

    def _show_programmes(self, programmes: list[Programme], *, same_service: bool) -> None:
        tree = self._programme_tree
        scroll_position = tree.verticalScrollBar().value()
        selected = tree.currentItem()
        selected_programme = None if selected is None else selected.data(0, _PROGRAMME_ROLE)

        tree.blockSignals(True)
        tree.clear()
        now = time.time()
        day_items: dict[str, QTreeWidgetItem] = {}  # by date, "YYYY-MM-DD"
        restore_item = None
        for programme in programmes:
            day = time.strftime("%Y-%m-%d", time.localtime(programme.start))
            parent = day_items.get(day)
            if parent is None:
                parent = QTreeWidgetItem([_day_title(programme.start)])
                parent.setFirstColumnSpanned(True)
                font = parent.font(0)
                font.setBold(True)
                parent.setFont(0, font)
                tree.addTopLevelItem(parent)
                day_items[day] = parent
            row = QTreeWidgetItem(
                [
                    time.strftime("%H:%M", time.localtime(programme.start)),
                    _duration_text(programme.duration),
                    programme.names.best,
                ]
            )
            row.setData(0, _PROGRAMME_ROLE, programme)
            if programme.description:
                row.setToolTip(2, programme.description)
            if selected_programme is not None and (programme.start, programme.names) == (
                selected_programme.start,
                selected_programme.names,
            ):
                restore_item = row
            parent.addChild(row)

        # Past days collapsed, today and later open.
        today = time.strftime("%Y-%m-%d", time.localtime(now))
        for day, parent in day_items.items():
            parent.setExpanded(day >= today)
        running = self._update_running_programme()
        for column in range(3):
            tree.resizeColumnToContents(column)

        if same_service and restore_item is not None:
            tree.setCurrentItem(restore_item)
            tree.verticalScrollBar().setValue(scroll_position)
        elif running is not None:
            tree.setCurrentItem(running)
            tree.scrollToItem(running)
        tree.blockSignals(False)
        current = tree.currentItem()
        self._show_description(None if current is None else current.data(0, _PROGRAMME_ROLE))
        if not programmes:
            self._description.setHtml(self._empty_html())

    def _update_running_programme(self) -> QTreeWidgetItem | None:
        """Sets the running programme in bold (and only that one) and fits the columns
        to it; returns its row."""
        now = time.time()
        running = None
        changed = False
        for day in range(self._programme_tree.topLevelItemCount()):
            parent = self._programme_tree.topLevelItem(day)
            for index in range(parent.childCount()):
                row = parent.child(index)
                programme = row.data(0, _PROGRAMME_ROLE)
                is_running = programme.start <= now < programme.end
                if is_running:
                    running = row
                for column in range(3):
                    font = row.font(column)
                    if font.bold() != is_running:
                        font.setBold(is_running)
                        row.setFont(column, font)
                        changed = True
        if changed:
            for column in range(3):
                self._programme_tree.resizeColumnToContents(column)
        return running

    def _on_programme_changed(self, current: QTreeWidgetItem | None, _previous) -> None:
        programme = None if current is None else current.data(0, _PROGRAMME_ROLE)
        self._show_description(programme)

    def _show_description(self, programme: Programme | None) -> None:
        if programme is None:
            self._description.setHtml("")
            return
        start = time.localtime(programme.start)
        end = time.localtime(programme.end)
        when = tr("{day}, {start} to {end} ({duration})").format(
            day=_day_title(programme.start),
            start=time.strftime("%H:%M", start),
            end=time.strftime("%H:%M", end),
            duration=_duration_text(programme.duration),
        )
        parts = [f"<h3>{html.escape(programme.names.best)}</h3>", f"<p>{html.escape(when)}</p>"]
        if programme.description:
            text = html.escape(programme.description).replace("\n", "<br>")
            parts.append(f"<p>{text}</p>")
        self._description.setHtml("".join(parts))

    def _empty_html(self) -> str:
        if self._shown_sid is None:
            if self._decoded_spi is not None:
                return tr(
                    "<p>No entries yet. The carousel of the programme guide runs in an endless "
                    "loop; it takes a few minutes until service information and schedules have "
                    "come by once.</p>"
                )
            return tr(
                "<p>The programme guide (SPI) comes as a data service of its own. With “Receive "
                "programme guide” above or in the “Data services” tab it is decoded in addition "
                "to the audio and stored for the channel; next time it is there at once.</p>"
            )
        return tr("<p>For this station the programme guide contains no schedule.</p>")
