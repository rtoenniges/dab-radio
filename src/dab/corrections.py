"""What the SSRZ corrections of GEPOS mean for a receiver at one place.

SSRZ describes the errors of GNSS signals in state space, independent of any receiver:
per satellite a correction to its broadcast clock and orbit and its signal biases, and
the ionosphere and troposphere as models over the service area (Geo++ SSRZ format
1.1.3, chapter 11; decoded in `dab.ssrz`). A receiver turns them into corrections of
its own measurements by evaluating everything along the line of sight to each
satellite. That is done here, per satellite and error source:

- Clock: dC = C0 + C1 (t - t0) + C_HR (eq. 11.18), in metres, as sent. The satellite
  clock is then t_broadcast - dC / c (eq. 11.17).
- Orbit: the radial, along-track and cross-track correction (eqs. 11.8 to 11.13),
  turned into the change of the range from the receiver: positive if the corrected
  satellite stands farther away.
- Signal biases as sent: code biases in metres, phase biases in cycles.
- Ionosphere: slant TEC of each stage (global VTEC, regional per satellite, gridded
  residuals; eq. 11.24) at the pierce points of the line of sight, and the delay it
  causes on the reference signal of the satellite (L1, G1, E1, B1-2).
- Troposphere: model delay in the slant (MOPS weather, Saastamoinen, Global Mapping
  Function; eqs. 11.73 to 11.75) and what the regional and gridded scale factors and
  the mapping function improvement add to it (eqs. 11.56, 11.78, 11.80).

The satellite position comes from the broadcast ephemeris the correction names by its
IOD (`dab.rtcm`, `dab.gnss`), at the time of transmission, rotated by the Earth's
rotation during the signal's travel (Sagnac). The ionosphere uses a spherical Earth of
6370 km radius (clause 11.4.1), the troposphere the ellipsoid.

Checked against the reference decoder of Geo++ (Geopp-SSRZ-Python-Demo, "state space
influence"); where it deviates from the format description, this module says so at the
place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from dab.atmosphere import day_of_year, model_zenith_delays
from dab.gmf import global_mapping_function
from dab.gnss import (
    BEIDOU_GPS_OFFSET_SECONDS,
    GPS_EPOCH_POSIX,
    GPS_UTC_LEAP_SECONDS,
    MOSCOW_UTC_OFFSET_SECONDS,
    SECONDS_PER_DAY,
    SECONDS_PER_WEEK,
    Observer,
    satellite_positions,
)
from dab.rtcm import GNSS_BEIDOU, GNSS_GLONASS, Ephemeris, GlonassEphemeris, KeplerEphemeris
from dab.ssrz import (
    TROPOSPHERE_DRY,
    TROPOSPHERE_TOTAL,
    TROPOSPHERE_WET,
    GlobalVtec,
    Grid,
    GridIonosphere,
    GridTroposphere,
    HighRateCorrections,
    LowRateCorrections,
    RegionalIonosphere,
    RegionalTroposphere,
    SsrzMetadata,
    SsrzTime,
    chebyshev,
)

SPEED_OF_LIGHT = 299_792_458.0
EARTH_ROTATION = 7.2921151467e-5  # rad/s (eq. 11.16)
# Ionospheric delay per TECU on frequency f: 40.3e16 / f^2 metres (eq. 11.24).
IONOSPHERE_CONSTANT = 40.3e16
# Spherical Earth of the ionosphere models (clause 11.4.1), m.
SPHERICAL_EARTH_RADIUS = 6_370_000.0
# Layer height of the gridded ionosphere (eq. 11.54), m.
GRID_IONOSPHERE_LAYER = 400_000.0
# Scale of the transversal Mercator projection of the regional ionosphere (eqs. 11.51,
# 11.52): the Earth radius in units of 1000 km.
_RSI_SCALE = 6.37

# Reference signals (the satellite reference point, table 11.1) and their frequencies.
_GPS_L1 = 1_575.42e6
_BEIDOU_B1 = 1_561.098e6
REFERENCE_SIGNALS = {"G": "1C", "R": "1C", "E": "1C", "J": "1C", "C": "2I"}

# How long a correction message stays usable after its time tag: clock and orbit two
# low rate intervals, the high rate clock two of its intervals.
_MAX_LOW_RATE_AGE = 60.0
_MAX_HIGH_RATE_AGE = 10.0
_MAX_ATMOSPHERE_AGE = 120.0
_EPHEMERIDES_PER_SATELLITE = 4

MJD_UNIX_EPOCH = 40_587.0  # 1970-01-01


def reference_frequency(name: str, frequency_channel: int = 0) -> float:
    """Frequency of the reference signal of a satellite in Hz: GPS, Galileo and QZSS
    L1/E1, GLONASS G1 of its frequency channel, BeiDou B1-2."""
    system = name[0]
    if system == GNSS_GLONASS:
        return 1_602.0e6 + frequency_channel * 0.5625e6
    if system == GNSS_BEIDOU:
        return _BEIDOU_B1
    return _GPS_L1


@dataclass(frozen=True)
class SatelliteCorrections:
    """The corrections of one satellite, evaluated at one place and time. Metres unless
    noted; NaN where the stream has nothing (yet)."""

    name: str
    elevation: float  # degrees, ellipsoidal
    azimuth: float  # degrees
    ephemeris_iod: int
    clock: float  # dC = C0 + C1 (t - t0) + C_HR
    clock_high_rate: float  # C_HR
    orbit: tuple[float, float, float]  # radial, along-track, cross-track
    orbit_range: float  # change of the range by the orbit correction
    code_biases: dict[str, float]
    phase_biases: dict[str, float]  # cycles
    frequency: float  # Hz, of the reference signal
    stec: dict[str, float]  # TECU per stage: "GVI", "RSI", "GRI"
    troposphere_model: float  # slant delay of the model troposphere
    # What each stage adds to it: "RT" (regional scale factors), "GRT" (gridded scale
    # factors), "MFI" (mapping function improvement).
    troposphere_parts: dict[str, float]

    @property
    def total_stec(self) -> float:
        values = [v for v in self.stec.values() if not math.isnan(v)]
        return sum(values) if values else math.nan

    @property
    def ionosphere(self) -> float:
        """Delay of the reference signal's code by the ionosphere (the carrier phase is
        advanced by as much)."""
        return IONOSPHERE_CONSTANT / self.frequency**2 * self.total_stec

    @property
    def troposphere_correction(self) -> float:
        """What the scale factors and the mapping function improvement add to the
        model delay."""
        values = [v for v in self.troposphere_parts.values() if not math.isnan(v)]
        return sum(values) if values else math.nan

    @property
    def troposphere(self) -> float:
        """Slant delay of the troposphere."""
        correction = self.troposphere_correction
        return self.troposphere_model + (0.0 if math.isnan(correction) else correction)


@dataclass
class _Stored:
    ephemerides: dict[str, dict[int, Ephemeris]] = field(default_factory=dict)
    metadata: SsrzMetadata | None = None
    grids: dict[int, Grid] = field(default_factory=dict)
    low_rate: LowRateCorrections | None = None
    high_rate: HighRateCorrections | None = None
    global_vtec: GlobalVtec | None = None
    regional_ionosphere: RegionalIonosphere | None = None
    grid_ionosphere: dict[int, GridIonosphere] = field(default_factory=dict)
    regional_troposphere: RegionalTroposphere | None = None
    grid_troposphere: dict[int, GridTroposphere] = field(default_factory=dict)
    gps_time: float | None = None  # of the last ZM009


class CorrectionState:
    """Collects the SSRZ results and broadcast ephemeris of the stream and evaluates
    them for a place (`evaluate`)."""

    def __init__(self) -> None:
        self._stored = _Stored()
        # Pierce point origins of the regional ionosphere, by satellite, ephemeris IOD
        # and the time they were set: fixed for a minute, so computed once.
        self._origins: dict[tuple[str, int, float], tuple[float, float] | None] = {}

    # --- Input ---------------------------------------------------------------------

    def add_ephemeris(self, ephemeris: Ephemeris) -> None:
        """Corrections name their ephemeris by IOD and may still refer to the previous
        one for a while after a new one arrived: a few are kept per satellite."""
        per_iod = self._stored.ephemerides.setdefault(ephemeris.name, {})
        per_iod.pop(ephemeris.ssr_iod, None)
        per_iod[ephemeris.ssr_iod] = ephemeris
        while len(per_iod) > _EPHEMERIDES_PER_SATELLITE:
            del per_iod[next(iter(per_iod))]

    def add_ssrz(self, result: object) -> None:
        s = self._stored
        if isinstance(result, SsrzTime):
            if result.gps_time is not None:
                s.gps_time = result.gps_time
        elif isinstance(result, SsrzMetadata):
            s.metadata = result
            for grid in result.grids:
                s.grids[grid.grid_id] = grid
        elif isinstance(result, LowRateCorrections):
            if result.clock is not None:
                s.low_rate = result
        elif isinstance(result, HighRateCorrections):
            s.high_rate = result
        elif isinstance(result, GlobalVtec):
            s.global_vtec = result
        elif isinstance(result, RegionalIonosphere):
            s.regional_ionosphere = result
        elif isinstance(result, GridIonosphere):
            s.grid_ionosphere[result.grid_id] = result
        elif isinstance(result, RegionalTroposphere):
            s.regional_troposphere = result
        elif isinstance(result, GridTroposphere):
            s.grid_troposphere[result.grid_id] = result
        elif hasattr(result, "grids"):  # GridDefinitions
            for grid in result.grids:
                s.grids[grid.grid_id] = grid

    @property
    def low_rate(self) -> LowRateCorrections | None:
        return self._stored.low_rate

    @property
    def high_rate(self) -> HighRateCorrections | None:
        return self._stored.high_rate

    @property
    def gps_time(self) -> float | None:
        """GPS time of the last time tag of the stream (ZM009)."""
        return self._stored.gps_time

    def ephemeris(self, name: str, iod: int) -> Ephemeris | None:
        return self._stored.ephemerides.get(name, {}).get(iod)

    # --- Evaluation ----------------------------------------------------------------

    def evaluate(self, observer: Observer, gps_time: float) -> list[SatelliteCorrections]:
        """The corrections of every satellite of the last low rate message whose
        ephemeris is known, at `observer` and `gps_time` (seconds since the GPS
        epoch), sorted by elevation, including those below the horizon."""
        s = self._stored
        low = s.low_rate
        if low is None or low.gps_time is None or low.clock is None:
            return []
        if abs(gps_time - low.gps_time) > _MAX_LOW_RATE_AGE:
            return []
        receiver = observer.ecef()
        mjd = (gps_time - GPS_UTC_LEAP_SECONDS + GPS_EPOCH_POSIX) / SECONDS_PER_DAY + (
            MJD_UNIX_EPOCH
        )
        doy = day_of_year(gps_time - GPS_UTC_LEAP_SECONDS + GPS_EPOCH_POSIX)
        zenith = model_zenith_delays(observer.latitude, observer.longitude, observer.height, doy)
        tropo_factors = self._troposphere_factors(observer, gps_time)
        grid_iono = self._closest(s.grid_ionosphere, observer, gps_time)
        high = s.high_rate
        if high is not None and (
            high.gps_time is None or abs(gps_time - high.gps_time) > _MAX_HIGH_RATE_AGE
        ):
            high = None
        high_clock = {} if high is None else dict(zip(high.satellites, high.clock, strict=True))
        high_radial: dict[str, float] = {}
        if high is not None and high.radial is not None:
            high_radial = dict(zip(high.satellites, high.radial, strict=True))

        results = []
        for row, name in enumerate(low.satellites):
            iod = low.ephemeris_iods.get(name)
            ephemeris = None if iod is None else self.ephemeris(name, iod)
            if ephemeris is None:
                continue
            state = _satellite_state(ephemeris, gps_time, receiver)
            if state is None:
                continue
            position, velocity = state
            elevation, azimuth = _look_angles(observer, receiver, position)

            dt = gps_time - low.gps_time
            c0 = low.clock[row, 0]
            c1 = low.clock[row, 1] if low.clock.shape[1] > 1 else 0.0
            c_hr = float(high_clock.get(name, math.nan))
            clock = c0 + (0.0 if math.isnan(c1) else c1 * dt)
            clock += 0.0 if math.isnan(c_hr) else c_hr
            orbit = np.array(low.orbit[row], dtype=float)
            if low.velocity is not None:
                orbit = orbit + np.nan_to_num(low.velocity[row]) * dt
            orbit[0] += high_radial.get(name, 0.0)
            orbit_range = _orbit_range(position, velocity, receiver, orbit)

            channel = ephemeris.frequency_channel if isinstance(ephemeris, GlonassEphemeris) else 0
            stec = self._stec(name, receiver, position, gps_time, grid_iono, observer, elevation)
            model = zenith.hydrostatic, zenith.wet
            mapping = global_mapping_function(
                mjd, observer.latitude, observer.longitude, observer.height, elevation
            )
            slant_dry = float(model[0] * mapping[0])
            slant_wet = float(model[1] * mapping[1])
            parts = self._troposphere_parts(
                tropo_factors, slant_dry, slant_wet, elevation, observer
            )
            results.append(
                SatelliteCorrections(
                    name,
                    elevation,
                    azimuth,
                    iod,
                    float(clock),
                    c_hr,
                    (float(orbit[0]), float(orbit[1]), float(orbit[2])),
                    orbit_range,
                    dict(low.code_biases.get(name, {})),
                    {sig: b.cycles for sig, b in low.phase_biases.get(name, {}).items()},
                    reference_frequency(name, channel),
                    stec,
                    slant_dry + slant_wet,
                    parts,
                )
            )
        results.sort(key=lambda r: -r.elevation)
        return results

    # --- Ionosphere ----------------------------------------------------------------

    def _stec(
        self,
        name: str,
        receiver: np.ndarray,
        position: np.ndarray,
        gps_time: float,
        grid_iono: GridIonosphere | None,
        observer: Observer,
        elevation: float,
    ) -> dict[str, float]:
        s = self._stored
        stec = {"GVI": math.nan, "RSI": math.nan, "GRI": math.nan}
        vtec = s.global_vtec
        if vtec is not None and vtec.valid and _fresh(vtec.gps_time, gps_time):
            pierce = pierce_point(receiver, position, vtec.layer_height_km * 1000.0)
            if pierce is not None:
                lat, lon, slant = pierce
                value = vtec.vtec(np.degrees(lat), np.degrees(lon), gps_time)
                stec["GVI"] = float(slant * value)
        regional = s.regional_ionosphere
        meta = None if s.metadata is None else s.metadata.regional_ionosphere
        if (
            regional is not None
            and meta is not None
            and name in regional.satellites
            and _fresh(regional.gps_time, gps_time)
        ):
            stec["RSI"] = self._regional_stec(name, regional, meta, receiver, position, gps_time)
        if grid_iono is not None and name in grid_iono.satellites:
            pierce = pierce_point(receiver, position, GRID_IONOSPHERE_LAYER)
            grid = s.grids.get(grid_iono.grid_id)
            if pierce is not None and grid is not None:
                values = grid_iono.values[grid_iono.satellites.index(name)]
                vertical = inverse_distance(grid, values, observer)
                stec["GRI"] = float(pierce[2] * vertical)
        return stec

    def _regional_stec(self, name, regional, meta, receiver, position, gps_time) -> float:
        """Eqs. 11.35 to 11.53. The pierce point origin is the pierce point of the line
        from the ground point origin to the satellite, fixed for a minute (at every
        full minute of GPS time, see `RegionalIonosphereMetadata.origin_interval`).
        The polynomial lives in a sun-fixed frame: as time goes on, the client's pierce
        point is turned east by dt / 43200 s * pi before it is compared with the origin.
        (Eq. 11.39 turns the origin east instead; the stream agrees with the reference
        decoder, which turns the pierce point.)"""
        origin = meta.origin
        layer = (meta.layer_height_km or 400.0) * 1000.0
        if origin is None:
            return math.nan
        ephemeris_iod = self._stored.low_rate.ephemeris_iods.get(name)
        ephemeris = self.ephemeris(name, ephemeris_iod)
        origin_time = gps_time - gps_time % meta.origin_interval
        key = (name, ephemeris_iod, origin_time)
        if key not in self._origins:
            if len(self._origins) > 500:
                self._origins.clear()
            gpo = Observer(*origin).ecef()
            origin_state = _satellite_state(ephemeris, origin_time, gpo)
            origin_pierce = None
            if origin_state is not None:
                origin_pierce = pierce_point(gpo, origin_state[0], layer)
            self._origins[key] = None if origin_pierce is None else origin_pierce[:2]
        origin_pierce = self._origins[key]
        client = pierce_point(receiver, position, layer)
        if origin_pierce is None or client is None:
            return math.nan
        shift = (gps_time - origin_time) / 43_200.0 * math.pi
        north, east = transversal_offsets(
            origin_pierce[0], origin_pierce[1], client[0], client[1] + shift
        )
        coefficients = regional.coefficients[regional.satellites.index(name)]
        order = max(l + m for l, m in regional.orders)
        cheb_north = chebyshev(order, np.array(north * _RSI_SCALE))
        cheb_east = chebyshev(order, np.array(east * _RSI_SCALE))
        vertical = 0.0
        for value, (l, m) in zip(coefficients, regional.orders, strict=True):
            if not math.isnan(value):
                vertical += value * float(cheb_north[l]) * float(cheb_east[m])
        return client[2] * vertical

    # --- Troposphere ---------------------------------------------------------------

    def _troposphere_factors(
        self, observer: Observer, gps_time: float
    ) -> dict[str, dict[str, float]]:
        """Scale factors per stage and component at the observer: "RT" from the
        regional polynomial (eq. 11.78) of the closest region, "GRT" from the closest
        grid, interpolated."""
        s = self._stored
        stages: dict[str, dict[str, float]] = {}
        regional = s.regional_troposphere
        if regional is not None and _fresh(regional.gps_time, gps_time) and regional.regions:
            region = min(
                regional.regions,
                key=lambda r: _distance_km(
                    observer, r.definition.origin[0], r.definition.origin[1]
                ),
            )
            factors = stages.setdefault("RT", {})
            for component in region.components:
                value = region.scale_factor(
                    component, observer.latitude, observer.longitude, observer.height
                )
                if value is not None:
                    factors[component] = float(value)
        gridded = self._closest(s.grid_troposphere, observer, gps_time)
        grid = None if gridded is None else s.grids.get(gridded.grid_id)
        if gridded is not None and grid is not None:
            factors = stages.setdefault("GRT", {})
            for component, values in gridded.components.items():
                value = inverse_distance(grid, values[0], observer)
                if not math.isnan(value):
                    factors[component] = value
        return stages

    def _troposphere_parts(
        self,
        stages: dict[str, dict[str, float]],
        slant_dry: float,
        slant_wet: float,
        elevation: float,
        observer: Observer,
    ) -> dict[str, float]:
        """What each stage adds to the model delay: the scale factors times the slant
        model delay of their component (eq. 11.56), and the mapping function
        improvement (eq. 11.80)."""
        total = slant_dry + slant_wet
        parts = {}
        for stage, factors in stages.items():
            parts[stage] = (
                factors.get(TROPOSPHERE_DRY, 0.0) * slant_dry
                + factors.get(TROPOSPHERE_WET, 0.0) * slant_wet
                + factors.get(TROPOSPHERE_TOTAL, 0.0) * total
            )
        regional = self._stored.regional_troposphere
        if "RT" in stages and regional is not None:
            for region in regional.regions:
                mapping = region.definition.mapping
                if mapping is None or region.mapping is None:
                    continue
                improvement = 0.0
                if elevation <= mapping.max_elevation:
                    epsilon = (mapping.max_elevation - elevation) / mapping.max_elevation
                    dh = (observer.height - region.definition.origin[2]) / 1000.0
                    shape = 1.0 - math.cos(epsilon * math.pi / 2.0)
                    orders = _parameter_orders(len(region.mapping))
                    for value, (p, q) in zip(region.mapping, orders, strict=True):
                        if not math.isnan(value):
                            improvement += total * value * shape * epsilon**q * dh**p
                parts["MFI"] = parts.get("MFI", 0.0) + improvement
        return parts

    # --- Grids ---------------------------------------------------------------------

    def _closest(self, per_grid: dict, observer: Observer, gps_time: float):
        """The gridded message of the grid with the station closest to the observer."""
        best = None
        best_distance = math.inf
        for grid_id, message in per_grid.items():
            grid = self._stored.grids.get(grid_id)
            if grid is None or grid.iod != message.grid_iod:
                continue
            if not _fresh(message.gps_time, gps_time):
                continue
            distance = float(np.min(_distances_km(observer, grid.latitudes, grid.longitudes)))
            if distance < best_distance:
                best, best_distance = message, distance
        return best


def _fresh(message_time: float | None, gps_time: float) -> bool:
    return message_time is not None and abs(gps_time - message_time) <= _MAX_ATMOSPHERE_AGE


def _parameter_orders(count: int) -> list[tuple[int, int]]:
    """(p, q) of the mapping function improvement parameters m_pq in transmission
    order: along the anti-diagonals, m00, m01, m10, m02, m11, m20, ..."""
    orders = []
    diagonal = 0
    while len(orders) < count:
        orders += [(p, diagonal - p) for p in range(diagonal + 1)]
        diagonal += 1
    return orders[:count]


# --- Satellites -------------------------------------------------------------------


def _satellite_clock(ephemeris: Ephemeris, gps_time: float) -> float:
    """Broadcast satellite clock offset in seconds (without the relativistic term)."""
    if isinstance(ephemeris, KeplerEphemeris):
        t = gps_time - (BEIDOU_GPS_OFFSET_SECONDS if ephemeris.system == GNSS_BEIDOU else 0.0)
        half = SECONDS_PER_WEEK / 2
        dt = (t - ephemeris.toc + half) % SECONDS_PER_WEEK - half
        return ephemeris.af0 + ephemeris.af1 * dt + ephemeris.af2 * dt * dt
    utc = gps_time - GPS_UTC_LEAP_SECONDS
    time_of_day = (utc + MOSCOW_UTC_OFFSET_SECONDS) % SECONDS_PER_DAY
    half = SECONDS_PER_DAY / 2
    dt = (time_of_day - ephemeris.tb + half) % SECONDS_PER_DAY - half
    return -ephemeris.tau_n + ephemeris.gamma_n * dt


def _satellite_state(
    ephemeris: Ephemeris, gps_time: float, receiver: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Position of the satellite when it sent the signal received at `gps_time`, in the
    Earth-fixed frame of the reception (Sagnac), and its Earth-fixed velocity."""
    travel = 0.075
    position = None
    for _ in range(2):  # the second pass is exact to nanoseconds
        transmit = gps_time - travel - _satellite_clock(ephemeris, gps_time - travel)
        utc = transmit - GPS_UTC_LEAP_SECONDS + GPS_EPOCH_POSIX
        positions = satellite_positions(ephemeris, np.array([utc - 0.5, utc, utc + 0.5]))
        if not np.all(np.isfinite(positions)):
            return None
        position = positions[1]
        travel = float(np.linalg.norm(position - receiver)) / SPEED_OF_LIGHT
    velocity = positions[2] - positions[0]
    angle = EARTH_ROTATION * travel
    rotation = np.array(
        [
            [math.cos(angle), math.sin(angle), 0.0],
            [-math.sin(angle), math.cos(angle), 0.0],
            [0, 0, 1],
        ]
    )
    return rotation @ position, rotation @ velocity


def _orbit_range(
    position: np.ndarray, velocity: np.ndarray, receiver: np.ndarray, orbit: np.ndarray
) -> float:
    """Eqs. 11.9 to 11.14: the change of the range by the orbit correction."""
    along = velocity / np.linalg.norm(velocity)
    cross = np.cross(position, velocity)
    cross /= np.linalg.norm(cross)
    radial = np.cross(along, cross)
    shift = orbit[0] * radial + orbit[1] * along + orbit[2] * cross
    return float(np.linalg.norm(position + shift - receiver) - np.linalg.norm(position - receiver))


def _look_angles(
    observer: Observer, receiver: np.ndarray, position: np.ndarray
) -> tuple[float, float]:
    lat, lon = math.radians(observer.latitude), math.radians(observer.longitude)
    east, north, up = _local_frame(lat, lon)
    d = position - receiver
    elevation = math.degrees(math.atan2(up @ d, math.hypot(east @ d, north @ d)))
    azimuth = math.degrees(math.atan2(east @ d, north @ d)) % 360.0
    return elevation, azimuth


def _local_frame(lat: float, lon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    east = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    return east, north, up


def pierce_point(
    receiver: np.ndarray, satellite: np.ndarray, layer_height: float
) -> tuple[float, float, float] | None:
    """Eqs. 11.27 to 11.33 on the spherical Earth: geocentric latitude and longitude of
    the pierce point (radians) of the line of sight through a layer `layer_height`
    metres above the sphere, and the slant factor. None below the horizon."""
    radius = float(np.linalg.norm(receiver))
    lat = math.asin(receiver[2] / radius)
    lon = math.atan2(receiver[1], receiver[0])
    height = radius - SPHERICAL_EARTH_RADIUS
    east, north, up = _local_frame(lat, lon)
    d = satellite - receiver
    elevation = math.atan2(up @ d, math.hypot(east @ d, north @ d))
    if elevation < 0:
        return None
    azimuth = math.atan2(east @ d, north @ d)
    ratio = (SPHERICAL_EARTH_RADIUS + height) / (SPHERICAL_EARTH_RADIUS + layer_height)
    psi = math.pi / 2 - elevation - math.asin(ratio * math.cos(elevation))
    lat_pp = math.asin(
        math.sin(lat) * math.cos(psi) + math.cos(lat) * math.sin(psi) * math.cos(azimuth)
    )
    angle = math.asin(max(-1.0, min(1.0, math.sin(psi) * math.sin(azimuth) / math.cos(lat_pp))))
    across_pole = math.tan(psi) * math.cos(azimuth)
    if (lat >= 0 and across_pole > math.tan(math.pi / 2 - lat)) or (
        lat < 0 and -across_pole > math.tan(math.pi / 2 + lat)
    ):
        lon_pp = lon + math.pi - angle
    else:
        lon_pp = lon + angle
    return lat_pp, lon_pp, 1.0 / math.sin(elevation + psi)


def _unit(lat: float, lon: float) -> np.ndarray:
    return np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])


def transversal_offsets(
    lat_origin: float, lon_origin: float, lat: float, lon: float
) -> tuple[float, float]:
    """Eqs. 11.37 to 11.48: north and east offset (radians) of a pierce point from the
    pierce point origin in the transversal system of the origin."""
    origin = _unit(lat_origin, lon_origin)
    point = _unit(lat, lon)
    pole = np.array([0.0, 0.0, 1.0])
    normal = np.cross(pole, origin)
    normal /= np.linalg.norm(normal)
    distance = float(normal @ point)
    east = math.asin(max(-1.0, min(1.0, distance)))
    in_plane = point - distance * normal
    in_plane /= np.linalg.norm(in_plane)
    cross = np.cross(in_plane, origin)
    size = float(np.linalg.norm(cross))
    north = math.asin(min(1.0, size))
    if size > 0 and float(cross @ normal) < 0:
        north = -north
    return north, east


_WGS84_A = 6_378_137.0
_WGS84_F = 1 / 298.257223563


def _distances_km(observer: Observer, latitudes, longitudes) -> np.ndarray:
    """Horizontal distances from the observer, in the plane of its horizon."""
    lat0 = math.radians(observer.latitude)
    north = np.radians(np.asarray(latitudes, dtype=float) - observer.latitude)
    east = np.radians(np.asarray(longitudes, dtype=float) - observer.longitude) * math.cos(lat0)
    return np.hypot(north, east) * _WGS84_A / 1000.0


def _distance_km(observer: Observer, latitude: float, longitude: float) -> float:
    return float(_distances_km(observer, [latitude], [longitude])[0])


def inverse_distance(grid: Grid, values: np.ndarray, observer: Observer) -> float:
    """The grid values interpolated to the observer with inverse distance squared
    weights over the stations with a value (SSRZ leaves the interpolation open,
    clause 11.5.4; the reference decoder uses the same)."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return math.nan
    distances = _distances_km(observer, grid.latitudes[valid], grid.longitudes[valid])
    if np.min(distances) < 1e-3:
        return float(values[valid][int(np.argmin(distances))])
    weights = 1.0 / distances**2
    return float(np.sum(weights * values[valid]) / np.sum(weights))
