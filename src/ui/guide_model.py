"""The programme guides of all channels for a surface, without a widget: the stored ones
(`dab.guide_cache`) and, while a guide is received, what its carousel completes, and how
far the carousel and the packets beneath it have come (for a page on the transport).

The controller stores what is received (`ui.receiver_controller`); this model only
keeps it in memory, combined across channels (`dab.combined_guide`), so that a surface
can show services and schedules. It reads the stored guides on first use rather than at
start: decoding every stored schedule takes a moment (60 ms for the guides of 5C and 7A
on a PC), and a surface that never shows the guide should not pay for it. The touch
surface does use it at start: its radio page shows what runs on the station playing.
The desktop's "Programme guide" tab (`ui.epg_tab`) still keeps its own.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from dab.combined_guide import CombinedGuide, EnsembleServices, GuideGroup
from dab.data_services import USER_APPLICATION_SPI, DataComponent
from dab.mot_directory import MotCarouselStatus, MotDirectory
from dab.spi import Programme, decode_spi_object
from dsp.packet import PacketStatistics
from dsp.pipeline import CarouselObject
from ui.receiver_controller import ReceiverController

logger = logging.getLogger(__name__)

# Schedules that ended longer ago than this are left out when the stored guides are read
# (as in `ui.epg_tab`, which also deletes them from the store).
STORED_SCHEDULE_MAX_AGE_SECONDS = 24 * 3600

# `changed` is reported at most this often: on the first turn of a carousel its objects
# arrive several per second for minutes.
CHANGED_DELAY_MS = 1000


def now_and_next(
    programmes: list[Programme], now: float
) -> tuple[Programme | None, Programme | None]:
    """The programme running at `now` and the first to start after it, each None if
    there is none; `programmes` in order of time (as `GuideModel.programmes` gives them).
    """
    running = None
    for programme in programmes:
        if programme.start > now:
            return running, programme
        if now < programme.end:
            running = programme
    return running, None


def spi_component(components: list[DataComponent]) -> DataComponent | None:
    """The data component carrying the programme guide (SPI), if the ensemble has one."""
    return next(
        (c for c in components if USER_APPLICATION_SPI in c.app_types and c.decodable), None
    )


class GuideModel(QObject):
    changed = pyqtSignal()  # services, logos or schedules changed
    # The guide appeared in or vanished from the ensemble received, or its decoding
    # started or stopped.
    decoding_changed = pyqtSignal()
    # The carousel or the packets of the guide being decoded were reported anew, or
    # there is nothing any more.
    carousel_changed = pyqtSignal()

    def __init__(self, controller: ReceiverController, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._guides = CombinedGuide()
        self._loaded = False
        # The service labels of the FIC stored with each guide, by channel.
        self._stored_labels: dict[str, dict[int, str]] = {}
        # The ensemble received as the FIC describes it; it may not be stored yet.
        self._ensemble: EnsembleServices | None = None
        self._spi: DataComponent | None = None
        self._carousel: MotCarouselStatus | None = None
        self._packets: PacketStatistics | None = None
        self._changed_timer = QTimer(self)
        self._changed_timer.setSingleShot(True)
        self._changed_timer.setInterval(CHANGED_DELAY_MS)
        self._changed_timer.timeout.connect(self.changed)

        dsp = controller.dsp_worker
        dsp.data_object_ready.connect(self._on_object)
        dsp.mot_directory_ready.connect(self._on_directory)
        dsp.mot_carousel_ready.connect(self._on_carousel)
        dsp.packet_statistics_ready.connect(self._on_packet_statistics)
        controller.live_reset.connect(self._on_live_reset)
        controller.ensemble_received.connect(self._on_ensemble)
        controller.frequency_information_received.connect(self._on_frequency_information)
        controller.channel_stored.connect(self._notify)
        controller.data_components_changed.connect(self._on_components)

    # --- What a surface reads ---------------------------------------------------------

    def groups(self) -> list[GuideGroup]:
        """The services of all guides by channel (see `CombinedGuide.groups`)."""
        self.load()
        return self._guides.groups(self._known_ensembles())

    def programmes(self, sid: int) -> list[Programme]:
        """The programmes of a service across all guides, in order of time."""
        self.load()
        return self._guides.programmes(sid)

    @property
    def empty(self) -> bool:
        self.load()
        return self._guides.empty

    # --- The guide being received ---------------------------------------------------

    @property
    def spi(self) -> DataComponent | None:
        """The guide (SPI) component of the ensemble received, if it has one."""
        return self._spi

    @property
    def decoding(self) -> DataComponent | None:
        """The guide component being decoded, None while none is."""
        scid = self._controller.guide_scid
        if scid is None:
            return None
        return next((c for c in self._controller.data_components if c.scid == scid), None)

    @property
    def carousel(self) -> MotCarouselStatus | None:
        """The carousel of the guide being decoded, as last reported."""
        return self._carousel

    @property
    def packets(self) -> PacketStatistics | None:
        """The packets of the subchannel carrying the guide being decoded, as last
        reported; the guide's own are those of its packet address."""
        return self._packets

    def load(self) -> None:
        """Reads the stored guides of all channels, once."""
        if self._loaded:
            return
        self._loaded = True
        cache = self._controller.guide_cache
        oldest = time.time() - STORED_SCHEDULE_MAX_AGE_SECONDS
        for channel in cache.channels():
            try:
                stored = cache.load(channel)
            except (OSError, ValueError) as exc:
                logger.warning("Stored guide of %s not loaded: %s", channel, exc)
                continue
            if stored is None:
                continue
            guide = self._guides.guide(channel)
            for mot_object in stored.objects:
                guide.add(mot_object, decode_spi_object(mot_object))
            guide.drop_schedules_ended_before(oldest)
            self._stored_labels[channel] = stored.labels

    # --- Keeping up to date -----------------------------------------------------------

    def _notify(self, *_args: object) -> None:
        if not self._changed_timer.isActive():
            self._changed_timer.start()

    def _known_ensembles(self) -> dict[str | None, EnsembleServices]:
        """The ensembles of all stored channels; for the channel received what the FIC
        delivers, each part as soon as it has arrived (as `ui.epg_tab` does)."""
        ensembles: dict[str | None, EnsembleServices] = {
            channel: EnsembleServices(None, labels)
            for channel, labels in self._stored_labels.items()
        }
        for channel, info in self._controller.ensemble_cache.entries().items():
            ensembles[channel] = EnsembleServices(
                info.label, dict(info.services), info.eid, dict(info.frequencies)
            )
        received = self._ensemble
        if received is not None:
            channel = self._controller.list_channel
            stored = ensembles.get(channel) or EnsembleServices(None)
            ensembles[channel] = EnsembleServices(
                received.label or stored.label,
                received.services or stored.services,
                stored.eid if received.eid is None else received.eid,
                received.frequencies or stored.frequencies,
            )
        return ensembles

    def _on_live_reset(self) -> None:
        """Another frequency: what its FIC says comes afresh."""
        self._ensemble = None
        self._notify()
        self._forget_transport()

    def _on_components(self) -> None:
        self._spi = spi_component(self._controller.data_components)
        if self._controller.guide_scid is None:
            self._forget_transport()
        self.decoding_changed.emit()

    def _forget_transport(self) -> None:
        if self._carousel is None and self._packets is None:
            return
        self._carousel = None
        self._packets = None
        self.carousel_changed.emit()

    def _on_carousel(self, scid: int, status: MotCarouselStatus) -> None:
        if scid == self._controller.guide_scid:
            self._carousel = status
            self.carousel_changed.emit()

    def _on_packet_statistics(self, subchannel_id: int, statistics: PacketStatistics) -> None:
        decoding = self.decoding
        if decoding is not None and decoding.subchannel_id == subchannel_id:
            self._packets = statistics
            self.carousel_changed.emit()

    def _on_ensemble(self, label: str | None, services: dict[int, str]) -> None:
        received = self._ensemble or EnsembleServices(None)
        self._ensemble = replace(received, label=label, services=dict(services))
        self._notify()

    def _on_frequency_information(
        self, eid: int | None, frequencies: dict[int, tuple[int, ...]]
    ) -> None:
        received = self._ensemble or EnsembleServices(None)
        self._ensemble = replace(received, eid=eid, frequencies=dict(frequencies))
        self._notify()

    def _on_object(self, scid: int, carousel_object: CarouselObject) -> None:
        # Before the first use nothing is kept: reading the store then brings it, as the
        # controller has stored it (only with a manual frequency it is not).
        if not self._loaded or not self._controller.is_received_guide(scid):
            return
        guide = self._guides.guide(self._controller.list_channel)
        if guide.add(carousel_object.mot_object, carousel_object.spi):
            self._notify()

    def _on_directory(self, scid: int, directory: MotDirectory) -> None:
        """A new version of the carousel: objects it no longer lists are outdated."""
        if not self._loaded or not self._controller.is_received_guide(scid):
            return
        names = {entry.header.content_name for entry in directory.entries}
        guide = self._guides.guide(self._controller.list_channel)
        if guide.retain({name for name in names if name}):
            self._notify()
