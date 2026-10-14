"""What GEPOS carries, for a surface, without a widget: the broadcast ephemerides and
where they put the satellites in the sky and in space, the statistics of the RTCM stream, the SSRZ
corrections evaluated at the listener's position with their course over half an hour,
and what they say about the atmosphere (`ui.atmosphere_model`).

The desktop's tabs under "GNSS" (`ui.gnss_tab`, `ui.atmosphere_tab`,
`ui.corrections_tab`) and the touch surface's "GNSS" menu (`ui.touch.gnss_view`) draw
from one `GnssModel`; they only decide when to draw. It keeps collecting while nothing
of it is shown. The listener's position is the controller's
(`ReceiverController.observer`).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from dab.corrections import CorrectionState, SatelliteCorrections
from dab.data_services import (
    DSCTY_TDC,
    USER_APPLICATION_GEPOS,
    USER_APPLICATION_TPEG,
    DataComponent,
)
from dab.gnss import (
    MAX_GLONASS_AGE_SECONDS,
    MAX_KEPLER_AGE_SECONDS,
    Observer,
    ephemeris_age,
    look_angles,
    orbit_ring,
    satellite_positions,
)
from dab.rtcm import Ephemeris, GlonassEphemeris, RtcmStatistics
from dab.ssrz import HighRateCorrections
from dab.timecode import BroadcastTime, ReferenceClock
from dsp.pipeline import DataComponentStatus
from i18n import tr
from ui.atmosphere_model import AtmosphereModel
from ui.receiver_controller import ReceiverController

# Positions are computed at most this often (seconds of the clock): a satellite moves
# by 0.01 degrees per second across the sky.
POSITION_INTERVAL_SECONDS = 1.0
# The path of each satellite over the next hour, in steps of three minutes; recomputed
# every 30 s, or at once when an ephemeris or the position changes.
TRACK_MINUTES = 60
TRACK_STEP_MINUTES = 3
TRACK_INTERVAL_SECONDS = 30.0
# Points of an orbit in space; a straight piece of 2° runs 4 km inside the ellipse.
ORBIT_POINTS = 180

# The corrections are evaluated with every high rate message (every 5 s) and kept this
# long per satellite.
HISTORY_MINUTES = 30
_HISTORY_SAMPLES = HISTORY_MINUTES * 12 + 1
# Evaluations at most this often (seconds of GPS time): one takes some 20 ms.
EVALUATION_SECONDS = 4.5


def gepos_component(components: list[DataComponent]) -> DataComponent | None:
    """The data component carrying GEPOS, if the ensemble has one."""
    return next(
        (c for c in components if USER_APPLICATION_GEPOS in c.app_types and c.decodable), None
    )


def decoded_tdc_component(statuses: tuple[DataComponentStatus, ...]) -> DataComponent | None:
    """Among the data components being decoded, the transparent data channel whose RTCM
    stream is shown: GEPOS if it is among them, otherwise the first TDC component that
    is not TPEG (that one goes to the "Traffic" tab)."""
    tdc = [
        status.component
        for status in statuses
        if status.component is not None
        and status.component.packet is not None
        and status.component.packet.dscty == DSCTY_TDC
        and USER_APPLICATION_TPEG not in status.component.app_types
    ]
    return next((c for c in tdc if USER_APPLICATION_GEPOS in c.app_types), next(iter(tdc), None))


def _max_age(ephemeris: Ephemeris) -> float:
    if isinstance(ephemeris, GlonassEphemeris):
        return MAX_GLONASS_AGE_SECONDS
    return MAX_KEPLER_AGE_SECONDS


def sky_xy(azimuth: np.ndarray, elevation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Polar projection of the sky plot: the zenith in the centre, the horizon on the
    circle of radius 90, north at the top and east on the right, as on a map."""
    radius = 90.0 - elevation
    angle = np.radians(azimuth)
    return radius * np.sin(angle), radius * np.cos(angle)


def describe_age(age: float) -> str:
    """When the ephemeris refers to, from its age in seconds ("3 min ago")."""
    minutes = round(age / 60)
    if minutes > 0:
        # Translators: the reference time of an ephemeris, before now.
        return tr("{minutes} min ago").format(minutes=minutes)
    if minutes < 0:
        # Translators: the reference time of an ephemeris, after now.
        return tr("in {minutes} min").format(minutes=-minutes)
    # Translators: the reference time of an ephemeris is the present minute.
    return tr("now")


@dataclass(frozen=True)
class SatelliteView:
    """Where a satellite stands, seen from the listener."""

    ephemeris: Ephemeris
    azimuth: float
    elevation: float
    distance: float
    age: float  # seconds from the reference time of the ephemeris to now


# A path over the next hour: system, azimuth and elevation in degrees.
Track = tuple[str, np.ndarray, np.ndarray]


@dataclass(frozen=True, eq=False)
class SatelliteOrbit:
    """Where a satellite stands in space and the ellipse it runs on, Earth-fixed."""

    ephemeris: Ephemeris
    # (ORBIT_POINTS + 1, 3) in metres, from where the satellite stands round to there
    # again (`dab.gnss.orbit_ring`).
    ring: np.ndarray
    elevation: float  # seen from the listener, degrees

    @property
    def position(self) -> np.ndarray:
        return self.ring[0]


@dataclass(frozen=True)
class CorrectionSample:
    time: float  # GPS seconds
    clock: float  # m
    clock_low_rate: float  # m, C0 (+ C1 dt)
    orbit: tuple[float, float, float]  # m
    orbit_range: float  # m
    ionosphere: float  # m on the reference signal
    troposphere: float  # m


class CorrectionHistory:
    """The last half hour of one satellite, for the plots."""

    def __init__(self) -> None:
        self.samples: deque[CorrectionSample] = deque(maxlen=_HISTORY_SAMPLES)

    def add(self, sample: CorrectionSample) -> None:
        if self.samples and sample.time <= self.samples[-1].time:
            return
        self.samples.append(sample)

    def column(self, name: str) -> np.ndarray:
        return np.array([getattr(s, name) for s in self.samples], dtype=float)


class GnssModel(QObject):
    """Takes what the signal processing finds in GEPOS (the `add_*`/`update_*` methods,
    connected by `for_controller`) and keeps it for the surfaces to draw."""

    # GEPOS appeared in or vanished from the ensemble, or the decoded TDC component changed.
    decoding_changed = pyqtSignal()
    sky_changed = pyqtSignal()  # an ephemeris came or went, or the position changed
    stream_changed = pyqtSignal()  # new statistics of the RTCM stream, or none any more
    atmosphere_changed = pyqtSignal()  # a map of `atmosphere` would change
    corrections_changed = pyqtSignal()  # an epoch was evaluated, or the position changed

    def __init__(self, observer: Observer, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._observer = observer
        # The clock of the ensemble (FIG 0/10), like the slideshow's; until it is
        # received, the computer's.
        self._clock = ReferenceClock()
        self._gepos: DataComponent | None = None
        self._decoding: DataComponent | None = None  # the TDC component being decoded
        # The latest ephemeris per satellite, with the time it was received.
        self._ephemerides: dict[str, tuple[Ephemeris, float]] = {}
        self._statistics: RtcmStatistics | None = None
        self.atmosphere = AtmosphereModel()
        self._corrections = CorrectionState()
        self._latest: list[SatelliteCorrections] = []
        self._latest_time: float | None = None
        self._history: dict[str, CorrectionHistory] = {}

    @classmethod
    def for_controller(
        cls, controller: ReceiverController, parent: QObject | None = None
    ) -> GnssModel:
        """A model fed by the controller's signal processing, at its position."""
        model = cls(controller.observer, parent)
        dsp = controller.dsp_worker
        dsp.data_services_updated.connect(model.update_components)
        dsp.data_decoding_ready.connect(model.set_data_decoding)
        # Through the model rather than directly to the clock: the clock is then set in
        # the GUI thread, where it is also read.
        dsp.date_time_ready.connect(model.synchronise_clock)
        dsp.rtcm_statistics_ready.connect(model.update_statistics)
        dsp.ephemeris_ready.connect(model.add_ephemeris)
        dsp.ssrz_ready.connect(model.add_ssrz)
        controller.observer_changed.connect(model.set_observer)
        # The statistics belong to the ensemble; the ephemerides and corrections describe
        # the satellites and stay.
        controller.live_reset.connect(model.reset_stream)
        return model

    # --- Input --------------------------------------------------------------------------

    def update_components(self, components: list[DataComponent]) -> None:
        gepos = gepos_component(components)
        if gepos != self._gepos:
            self._gepos = gepos
            self.decoding_changed.emit()

    def set_data_decoding(self, statuses: tuple[DataComponentStatus, ...]) -> None:
        decoding = decoded_tdc_component(statuses)
        if decoding == self._decoding:
            return
        previous = None if self._decoding is None else self._decoding.scid
        self._decoding = decoding
        if (None if decoding is None else decoding.scid) != previous:
            self.reset_stream()
        self.decoding_changed.emit()

    def synchronise_clock(self, broadcast_time: BroadcastTime) -> None:
        self._clock.synchronise(broadcast_time)

    def set_observer(self, observer: Observer) -> None:
        """A new position: the course of the corrections so far belonged to the old one."""
        if observer == self._observer:
            return
        self._observer = observer
        self._history.clear()
        if self._latest_time is not None:
            self._evaluate(self._latest_time)
        self.sky_changed.emit()
        self.corrections_changed.emit()

    def reset_stream(self) -> None:
        """Forgets the statistics of the stream (new ensemble or another component)."""
        if self._statistics is not None:
            self._statistics = None
            self.stream_changed.emit()

    def add_ephemeris(self, ephemeris: Ephemeris) -> None:
        self._ephemerides[ephemeris.name] = (ephemeris, self.now())
        self._corrections.add_ephemeris(ephemeris)
        self.sky_changed.emit()

    def update_statistics(self, scid: int, statistics: RtcmStatistics) -> None:
        if self._decoding is None or scid != self._decoding.scid:
            return
        self._statistics = statistics
        self.stream_changed.emit()

    def add_ssrz(self, result: object) -> None:
        """Takes what an SSRZ message said (`dsp.pipeline.DspWorker.ssrz_ready`). Every
        high rate message (5 s) is an epoch of the corrections."""
        if self.atmosphere.add(result, self.now()):
            self.atmosphere_changed.emit()
        self._corrections.add_ssrz(result)
        if not isinstance(result, HighRateCorrections) or result.gps_time is None:
            return
        if self._latest_time is None or (
            abs(result.gps_time - self._latest_time) >= EVALUATION_SECONDS
        ):
            self._evaluate(result.gps_time)
            self.corrections_changed.emit()

    # --- State --------------------------------------------------------------------------

    def now(self) -> float:
        """Seconds since the epoch by the ensemble clock, else by the computer's."""
        synchronised = self._clock.now()
        return time.time() if synchronised is None else synchronised

    @property
    def clock_synchronised(self) -> bool:
        return self._clock.now() is not None

    @property
    def observer(self) -> Observer:
        return self._observer

    @property
    def gepos(self) -> DataComponent | None:
        """The GEPOS component of the ensemble, if it has one."""
        return self._gepos

    @property
    def decoding(self) -> DataComponent | None:
        """The TDC component whose RTCM stream is decoded, normally GEPOS."""
        return self._decoding

    @property
    def gepos_decoded(self) -> bool:
        return (
            self._gepos is not None
            and self._decoding is not None
            and self._decoding.scid == self._gepos.scid
        )

    # --- Sky ----------------------------------------------------------------------------

    @property
    def ephemeris_count(self) -> int:
        return len(self._ephemerides)

    def drop_stale(self, now: float) -> None:
        """An ephemeris no longer renewed is dropped once it is too old. Its age cannot be
        told from the ephemeris alone after a while: the week of GPS and the day of
        GLONASS are ambiguous, so it is measured from when it arrived."""
        stale = [
            name
            for name, (ephemeris, received) in self._ephemerides.items()
            if now - received > _max_age(ephemeris)
        ]
        for name in stale:
            del self._ephemerides[name]
        if stale:
            self.sky_changed.emit()

    def satellites(self, now: float) -> list[SatelliteView]:
        """Every satellite with an ephemeris, highest first; those below the horizon
        have a negative elevation."""
        satellites = []
        for ephemeris, _received in self._ephemerides.values():
            position = satellite_positions(ephemeris, now)
            if not np.all(np.isfinite(position)):
                continue
            azimuth, elevation, distance = look_angles(self._observer, position)
            satellites.append(
                SatelliteView(
                    ephemeris,
                    float(azimuth[0]),
                    float(elevation[0]),
                    float(distance[0]),
                    ephemeris_age(ephemeris, now),
                )
            )
        satellites.sort(key=lambda s: -s.elevation)
        return satellites

    def orbits(self, now: float) -> list[SatelliteOrbit]:
        """Every satellite with an ephemeris where it stands in space, with its orbit as
        it lies there now."""
        orbits = []
        for ephemeris, _received in self._ephemerides.values():
            ring = orbit_ring(ephemeris, now, ORBIT_POINTS)
            if not np.all(np.isfinite(ring)):
                continue
            _, elevation, _ = look_angles(self._observer, ring[:1])
            orbits.append(SatelliteOrbit(ephemeris, ring, float(elevation[0])))
        return orbits

    def tracks(self, now: float) -> dict[str, Track]:
        """The path over the next hour of every satellite above the horizon in it."""
        times = now + np.arange(0, TRACK_MINUTES + 1, TRACK_STEP_MINUTES) * 60.0
        tracks = {}
        for ephemeris, _received in self._ephemerides.values():
            azimuth, elevation, _ = look_angles(
                self._observer, satellite_positions(ephemeris, times)
            )
            if np.any(elevation >= 0):
                tracks[ephemeris.name] = (ephemeris.system, azimuth, elevation)
        return tracks

    # --- Stream -------------------------------------------------------------------------

    @property
    def statistics(self) -> RtcmStatistics | None:
        """What the RTCM stream of the decoded component has yielded so far."""
        return self._statistics

    # --- Corrections --------------------------------------------------------------------

    @property
    def corrections(self) -> list[SatelliteCorrections]:
        """The corrections of the latest epoch per satellite, highest first."""
        return self._latest

    @property
    def corrections_time(self) -> float | None:
        """GPS time of the latest epoch evaluated."""
        return self._latest_time

    def history(self, name: str) -> CorrectionHistory | None:
        return self._history.get(name)

    def missing_ephemerides(self) -> list[str]:
        """Satellites of the corrections whose ephemeris (by its IOD) has not come yet."""
        low = self._corrections.low_rate
        if low is None:
            return []
        return [
            name
            for name in low.satellites
            if self._corrections.ephemeris(name, low.ephemeris_iods.get(name, -1)) is None
        ]

    def _evaluate(self, gps_time: float) -> None:
        self._latest = self._corrections.evaluate(self._observer, gps_time)
        self._latest_time = gps_time
        for c in self._latest:
            history = self._history.setdefault(c.name, CorrectionHistory())
            history.add(
                CorrectionSample(
                    gps_time,
                    c.clock,
                    c.clock - (0.0 if math.isnan(c.clock_high_rate) else c.clock_high_rate),
                    c.orbit,
                    c.orbit_range,
                    c.ionosphere,
                    c.troposphere,
                )
            )
