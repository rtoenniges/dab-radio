"""All stations the radio knows, across every stored ensemble, for the station page and
for stepping through them with ◀ ▶.

The order is that of the programme guide (`dab.combined_guide`): channels by frequency,
the stations of a channel by name; the favourites (`ReceiverController.favourites`) keep
it as well. The stored service list of a channel (scan or
"Update station list") counts until the channel is received and its FIC has
delivered the whole list; from then on what the FIC delivers, until the channel is
stored again. Taken over before it is complete, the list would grow by single stations
and reorder the page each time; complete, it changes the catalogue only where it differs
from the stored one.

Logos come from the stored programme guides (`dab.guide_cache`), which the controller
fills while a guide is received; they are read again whenever it stores a logo or a
service information. A station whose guide has not been received has none; the
surface then shows its initials.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap

from dab.combined_guide import CombinedGuide, EnsembleServices, channel_order, service_logos
from dab.ensemble_cache import EnsembleInfo
from dab.guide_cache import GuideCache
from dab.spi import CONTENT_TYPE_SPI, SUBTYPE_PROGRAMME_INFORMATION, decode_spi_object
from i18n import tr
from ui.receiver_controller import ReceiverController

logger = logging.getLogger(__name__)

# The logos are read again this long after the last logo or service information was
# stored: on the first turn of a carousel they arrive one after another for minutes.
LOGO_RELOAD_DELAY_MS = 3000

# The filter that passes the favourites, in place of a channel.
FAVOURITES = "favourites"
# Key of the setting holding the filter chosen last: a channel, FAVOURITES, or missing
# for all stations.
SETTINGS_STATION_FILTER = "touch/station_filter"


@dataclass(frozen=True)
class Station:
    channel: str
    sid: int
    label: str
    programme_type: int | None = None

    @property
    def key(self) -> tuple[str, int]:
        """A station is a service on a channel: the same SId may be on several."""
        return self.channel, self.sid


@dataclass(frozen=True)
class EnsembleListing:
    """The services of one channel, stored or received."""

    label: str | None
    services: tuple[tuple[int, str], ...]
    programme_types: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_info(cls, info: EnsembleInfo) -> EnsembleListing:
        return cls(info.label, tuple(info.services), dict(info.programme_types))


def build_catalogue(listings: dict[str, EnsembleListing]) -> list[Station]:
    """The stations of all channels in the order of the programme guide."""
    stations = [
        Station(channel, sid, label, listing.programme_types.get(sid))
        for channel, listing in listings.items()
        for sid, label in listing.services
    ]
    return sorted(stations, key=lambda s: (channel_order(s.channel), s.label.casefold(), s.sid))


def step_station(
    stations: list[Station], current: tuple[str, int] | None, offset: int
) -> Station | None:
    """The station `offset` places from `current`, wrapping round at either end. If
    `current` is not among `stations`, the first (forwards) or last (backwards)."""
    if not stations:
        return None
    keys = [station.key for station in stations]
    if current in keys:
        return stations[(keys.index(current) + offset) % len(stations)]
    return stations[0] if offset > 0 else stations[-1]


def initials(label: str) -> str:
    """Up to two characters standing for a station without a logo: the first letters of
    its first two words ("Dlf Kultur" -> "DK"), else the first two of the one word."""
    words = ["".join(c for c in word if c.isalnum()) for word in label.split()]
    words = [word for word in words if word]
    if not words:
        return "?"
    if len(words) == 1:
        return words[0][:2].upper()
    return (words[0][0] + words[1][0]).upper()


def placeholder_pixmap(label: str, size: int, background: QColor, foreground: QColor) -> QPixmap:
    """A round badge with the initials of the station, in place of a logo."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(background)
    painter.drawEllipse(0, 0, size, size)
    font = QFont()
    font.setPixelSize(round(size * 0.38))
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(foreground)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, initials(label))
    painter.end()
    return pixmap


def stored_logos(guide_cache: GuideCache, ensembles: dict[str, EnsembleInfo]) -> dict[int, bytes]:
    """SId -> logo from the programme guides stored for any channel. The schedules are
    left undecoded: they hold no logos, and they make up most of a guide."""
    combined = CombinedGuide()
    for channel in guide_cache.channels():
        try:
            stored = guide_cache.load(channel)
        except (OSError, ValueError) as exc:
            logger.warning("Stored guide of %s not loaded: %s", channel, exc)
            continue
        if stored is None:
            continue
        guide = combined.guide(channel)
        for mot_object in stored.objects:
            if (
                mot_object.content_type == CONTENT_TYPE_SPI
                and mot_object.content_subtype == SUBTYPE_PROGRAMME_INFORMATION
            ):
                continue
            guide.add(mot_object, decode_spi_object(mot_object))
    known = {
        channel: EnsembleServices(info.label, dict(info.services), info.eid, dict(info.frequencies))
        for channel, info in ensembles.items()
    }
    return service_logos(combined.groups(known), None)


class StationDirectory(QObject):
    """The catalogue of `build_catalogue`, kept up to date from the controller, with the
    filter chosen on the station page and the logos. The filter is kept for the next
    start, so that ◀ ▶ go on stepping through the same stations."""

    changed = pyqtSignal()  # the stations, the filter or the station playing changed
    logos_changed = pyqtSignal()  # a stored guide brought new or changed logos

    def __init__(self, controller: ReceiverController, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._live: dict[str, EnsembleListing] = {}
        # A channel, FAVOURITES, or None for all.
        self._filter: str | None = controller.settings.value(SETTINGS_STATION_FILTER, None)
        if not isinstance(self._filter, str):  # edited by hand
            self._filter = None
        self._stations: list[Station] = []
        self._logos: dict[int, QPixmap] = {}
        self._logo_data: dict[int, bytes] | None = None  # what `_logos` was made from
        self._load_logos()
        self._rebuild()
        self._logo_timer = QTimer(self)
        self._logo_timer.setSingleShot(True)
        self._logo_timer.setInterval(LOGO_RELOAD_DELAY_MS)
        self._logo_timer.timeout.connect(self._reload_logos)
        controller.channel_stored.connect(self._on_channel_stored)
        controller.service_list_changed.connect(self._on_service_list_changed)
        controller.service_list_completed.connect(self._on_service_list_changed)
        controller.selected_service_changed.connect(self.changed)
        controller.guide_stored.connect(self._on_guide_stored)
        controller.favourites_changed.connect(self.changed)

    # --- What the pages read ----------------------------------------------------------

    def stations(self) -> list[Station]:
        """The stations passing the filter, in order."""
        if self._filter is None:
            return list(self._stations)
        if self._filter == FAVOURITES:
            favourites = self._controller.favourites
            return [station for station in self._stations if station.key in favourites]
        return [station for station in self._stations if station.channel == self._filter]

    def all_stations(self) -> list[Station]:
        return list(self._stations)

    @property
    def filter(self) -> str | None:
        return self._filter

    def set_filter(self, channel: str | None) -> None:
        """A channel, FAVOURITES, or None for all stations."""
        if channel == self._filter:
            return
        self._filter = channel
        if channel is None:
            self._controller.settings.remove(SETTINGS_STATION_FILTER)
        else:
            self._controller.settings.setValue(SETTINGS_STATION_FILTER, channel)
        self.changed.emit()

    def is_favourite(self, station: Station) -> bool:
        return self._controller.is_favourite(*station.key)

    def toggle_favourite(self, station: Station) -> None:
        self._controller.set_favourite(*station.key, not self.is_favourite(station))

    def filters(self) -> list[tuple[str, str]]:
        """(channel, "5C · DR Deutschland") for every channel with stations, in order."""
        channels = dict.fromkeys(station.channel for station in self._stations)
        return [(channel, self.filter_name(channel)) for channel in channels]

    def filter_name(self, channel: str | None) -> str:
        """The name of a filter as the chips of the station page read: "Favourites",
        "All", or the channel with its ensemble, "5C · DR Deutschland"."""
        if channel == FAVOURITES:
            return tr("Favourites")
        if channel is None:
            # Translators: the chip of all stations, beside "Favourites" and the channels.
            return tr("All")
        label = self.ensemble_label(channel)
        return f"{channel} · {label}" if label else channel

    def counts(self) -> dict[str | None, int]:
        """How many stations each filter passes: None (all), FAVOURITES (those known)
        and every channel."""
        favourites = self._controller.favourites
        counts: dict[str | None, int] = {None: len(self._stations), FAVOURITES: 0}
        for station in self._stations:
            counts[station.channel] = counts.get(station.channel, 0) + 1
            if station.key in favourites:
                counts[FAVOURITES] += 1
        return counts

    def current(self) -> Station | None:
        """The station selected in the controller, if the catalogue has it."""
        sid = self._controller.selected_sid
        channel = self._controller.channel
        if sid is None:
            return None
        return next((s for s in self._stations if s.key == (channel, sid)), None)

    def neighbour(self, offset: int) -> Station | None:
        """The station `offset` places from the one playing, within the filter."""
        current = self.current()
        return step_station(self.stations(), None if current is None else current.key, offset)

    def ensemble_label(self, channel: str | None) -> str | None:
        listing = self._listings().get(channel) if channel is not None else None
        return None if listing is None else listing.label

    def logo(self, sid: int) -> QPixmap | None:
        return self._logos.get(sid)

    # --- Keeping up to date -----------------------------------------------------------

    def _listings(self) -> dict[str, EnsembleListing]:
        listings = {
            channel: EnsembleListing.from_info(info)
            for channel, info in self._controller.ensemble_cache.entries().items()
        }
        listings.update(self._live)
        return listings

    def _rebuild(self) -> None:
        stations = build_catalogue(self._listings())
        if stations == self._stations:
            return
        self._stations = stations
        if self._filter not in (None, FAVOURITES) and all(
            s.channel != self._filter for s in stations
        ):
            self._filter = None
        self.changed.emit()

    def _on_channel_stored(self, channel: str) -> None:
        # Stored now (scan, or the received list taken over): the stored list counts.
        self._live.pop(channel, None)
        self._rebuild()

    def _on_service_list_changed(
        self,
        ensemble_label: str | None,
        services: list[tuple[int, str]],
        programme_types: dict[int, int],
    ) -> None:
        channel = self._controller.list_channel
        if not self._controller.live_list_complete or channel is None or not services:
            return
        self._live[channel] = EnsembleListing(ensemble_label, tuple(services), programme_types)
        self._rebuild()

    def _on_guide_stored(self, _channel: str, _when: float, services: bool) -> None:
        if services:
            self._logo_timer.start()

    def _reload_logos(self) -> None:
        if self._load_logos():
            self.logos_changed.emit()
            self.changed.emit()

    def _load_logos(self) -> bool:
        """Reads the logos of all stored guides; True if any changed."""
        controller = self._controller
        logos = stored_logos(controller.guide_cache, controller.ensemble_cache.entries())
        if logos == self._logo_data:
            return False
        self._logo_data = logos
        self._logos = {}
        for sid, data in logos.items():
            pixmap = QPixmap()
            if pixmap.loadFromData(data):
                self._logos[sid] = pixmap
        return True
