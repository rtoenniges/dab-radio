"""Satellite orbits from broadcast ephemerides, and where the satellites stand in the sky.

GEPOS sends the broadcast ephemerides of all satellites its corrections cover (see
`dab.rtcm`). From them the position of every satellite at any moment close to their
reference time follows, and with the position of the listener its direction in the sky.
No GNSS receiver is involved: the DAB+ signal alone says which satellites are above the
horizon.

GPS, Galileo and BeiDou describe their orbits the same way, by Keplerian elements with
harmonic corrections (IS-GPS-200, table 20-IV; Galileo OS SIS ICD, 5.1.1; BeiDou ICD
B1I, 5.2.4.12). They differ in their constants, in their time scale and in the
geostationary BeiDou satellites, whose orbit plane is tilted by 5 degrees before the
elements are applied. GLONASS instead sends position, velocity and luni-solar
acceleration at one instant; the orbit around it follows by integrating the equations
of motion in the Earth-fixed frame (GLONASS ICD, edition 5.1, appendix A.3.1.2).

All times are POSIX seconds in UTC. The difference between the frames (WGS 84,
Galileo's GTRF, BeiDou's CGCS2000, GLONASS' PZ-90.11) is a few centimetres and is
ignored here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dab.rtcm import (
    GNSS_BEIDOU,
    GNSS_GALILEO,
    GNSS_GPS,
    Ephemeris,
    GlonassEphemeris,
    KeplerEphemeris,
)

GPS_EPOCH_POSIX = 315_964_800  # 1980-01-06 00:00 UTC
# GPS time runs ahead of UTC by the leap seconds since 1980: 18 s since 1 January 2017
# (IERS Bulletin C). Galileo system time runs with GPS time, BeiDou time 14 s behind it.
GPS_UTC_LEAP_SECONDS = 18
BEIDOU_GPS_OFFSET_SECONDS = 14
SECONDS_PER_WEEK = 604_800
SECONDS_PER_DAY = 86_400
# GLONASS counts its time of day in Moscow time.
MOSCOW_UTC_OFFSET_SECONDS = 3 * 3600

# Gravitational constant (m^3/s^2) and rotation rate of the Earth (rad/s), as each
# system defines them for its orbit computation.
_KEPLER_CONSTANTS = {
    GNSS_GPS: (3.986005e14, 7.2921151467e-5),
    GNSS_GALILEO: (3.986004418e14, 7.2921151467e-5),
    GNSS_BEIDOU: (3.986004418e14, 7.292115e-5),
}

# PZ-90 constants of the GLONASS orbit integration.
_GLONASS_GM = 3.9860044e14
_GLONASS_EARTH_RADIUS = 6_378_136.0
_GLONASS_J2 = 1.0826257e-3
_GLONASS_EARTH_ROTATION = 7.292115e-5
# Runge-Kutta step. The ICD's example uses a few minutes; one minute keeps the
# integration error far below the metre over an hour.
GLONASS_INTEGRATION_STEP_SECONDS = 60.0

# How far from its reference time an ephemeris is still used. GPS ephemerides are
# fitted to four hours around toe and renewed every two; the others are renewed more
# often. A day-old ephemeris would still put a satellite roughly in the right part of
# the sky, but that is no longer the orbit it was sent for. GLONASS ephemerides are
# renewed every 30 minutes; the integration drifts by some kilometres per hour, which
# is invisible in the sky but no reason to go further.
MAX_KEPLER_AGE_SECONDS = 4 * 3600
MAX_GLONASS_AGE_SECONDS = 2 * 3600

# The orbit plane of the geostationary BeiDou satellites is rotated by -5 degrees
# about the x axis before the elements are applied (BeiDou ICD B1I, table 5-11).
_BEIDOU_GEO_TILT = math.radians(-5.0)
_KEPLER_ITERATIONS = 10  # eccentricities below 0.03: converges to 1e-15 in a few steps

# WGS 84 ellipsoid.
WGS84_A = 6_378_137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)

# The ellipse of `orbit_ring` follows from the satellite's velocity, taken from its
# positions this long before and after. The gravitational constant and the Earth's
# rotation are those of GPS; the other systems' differ in the seventh digit.
_RING_VELOCITY_SECONDS = 1.0
_RING_GM = 3.986005e14
_EARTH_ROTATION = 7.2921151467e-5


def gps_time(utc: float | np.ndarray) -> float | np.ndarray:
    """GPS seconds since 6 January 1980 for POSIX seconds in UTC."""
    return utc - GPS_EPOCH_POSIX + GPS_UTC_LEAP_SECONDS


def is_beidou_geo(prn: int) -> bool:
    """The geostationary BeiDou satellites: C01 to C05 (BDS-2) and C59 to C63 (BDS-3)."""
    return prn <= 5 or prn >= 59


def ephemeris_age(ephemeris: Ephemeris, utc: float) -> float:
    """Seconds from the reference time of the ephemeris to `utc` (negative before it)."""
    if isinstance(ephemeris, KeplerEphemeris):
        return float(_kepler_tk(ephemeris, np.array([utc]))[0])
    return float(_glonass_dt(ephemeris, np.array([utc]))[0])


def satellite_positions(ephemeris: Ephemeris, utc: float | np.ndarray) -> np.ndarray:
    """Earth-fixed positions of the satellite (metres) at the given times.

    Returns:
        Shape (N, 3) for N times (N = 1 for a single time). Rows are NaN where the time
        lies too far from the reference time of the ephemeris (`MAX_KEPLER_AGE_SECONDS`,
        `MAX_GLONASS_AGE_SECONDS`).
    """
    times = np.atleast_1d(np.asarray(utc, dtype=np.float64))
    if isinstance(ephemeris, KeplerEphemeris):
        return _kepler_positions(ephemeris, times)
    return _glonass_positions(ephemeris, times)


def orbit_ring(ephemeris: Ephemeris, utc: float, points: int = 180) -> np.ndarray:
    """The ellipse the satellite runs on, as it lies in space at `utc`: `points` + 1
    Earth-fixed positions (metres) at equal steps of angle, beginning where the satellite
    stands at `utc` and ending there again. As if the Earth stood still: in the hours the
    satellite takes to go round, the Earth turns on below it, so the satellite does not
    follow this line over the ground.

    The ellipse is the Keplerian orbit through the satellite's position and velocity at
    `utc` (the osculating orbit). That is the same for every system, also for GLONASS,
    which sends no elements, and for the geostationary BeiDou satellites with their
    tilted plane. The Earth's flattening takes the satellites away from it by 0.3 km in
    an hour. NaN throughout where `satellite_positions` has no position.
    """
    step = _RING_VELOCITY_SECONDS
    before, position, after = satellite_positions(ephemeris, utc + np.array([-step, 0.0, step]))
    if not np.isfinite([before, position, after]).all():
        return np.full((points + 1, 3), np.nan)
    # The velocity in the frame that does not turn and coincides with the Earth-fixed
    # one at `utc`.
    velocity = (after - before) / (2 * step) + np.cross([0.0, 0.0, _EARTH_ROTATION], position)
    radius = np.linalg.norm(position)
    momentum = np.cross(position, velocity)
    eccentricity = np.cross(velocity, momentum) / _RING_GM - position / radius
    # Angles in the plane of the orbit count from the satellite in the direction it moves.
    first = position / radius
    second = np.cross(momentum / np.linalg.norm(momentum), first)
    perigee = math.atan2(eccentricity @ second, eccentricity @ first)
    angle = np.linspace(0.0, 2 * np.pi, points + 1)
    distance = (momentum @ momentum / _RING_GM) / (
        1 + np.linalg.norm(eccentricity) * np.cos(angle - perigee)
    )
    return distance[:, None] * (np.cos(angle)[:, None] * first + np.sin(angle)[:, None] * second)


def _kepler_tk(e: KeplerEphemeris, utc: np.ndarray) -> np.ndarray:
    """Time from toe in the system's own time scale, folded into +-half a week: the
    transmitted week is ambiguous, but an ephemeris is never older than a few hours."""
    t = gps_time(utc)
    if e.system == GNSS_BEIDOU:
        t = t - BEIDOU_GPS_OFFSET_SECONDS
    half_week = SECONDS_PER_WEEK / 2
    return (t - e.toe + half_week) % SECONDS_PER_WEEK - half_week


def _kepler_positions(e: KeplerEphemeris, utc: np.ndarray) -> np.ndarray:
    """IS-GPS-200, table 20-IV, for all times at once."""
    gm, earth_rotation = _KEPLER_CONSTANTS[e.system]
    tk = _kepler_tk(e, utc)
    a = e.sqrt_a**2
    mean_motion = math.sqrt(gm / a**3) + e.mean_motion_difference
    mean_anomaly = e.mean_anomaly + mean_motion * tk
    eccentric_anomaly = mean_anomaly
    for _ in range(_KEPLER_ITERATIONS):
        eccentric_anomaly = mean_anomaly + e.eccentricity * np.sin(eccentric_anomaly)
    true_anomaly = np.arctan2(
        math.sqrt(1 - e.eccentricity**2) * np.sin(eccentric_anomaly),
        np.cos(eccentric_anomaly) - e.eccentricity,
    )
    latitude = true_anomaly + e.perigee
    sin2, cos2 = np.sin(2 * latitude), np.cos(2 * latitude)
    u = latitude + e.cus * sin2 + e.cuc * cos2
    r = a * (1 - e.eccentricity * np.cos(eccentric_anomaly)) + e.crs * sin2 + e.crc * cos2
    inclination = e.inclination + e.inclination_rate * tk + e.cis * sin2 + e.cic * cos2
    x_plane, y_plane = r * np.cos(u), r * np.sin(u)
    sin_i, cos_i = np.sin(inclination), np.cos(inclination)

    geo = e.system == GNSS_BEIDOU and is_beidou_geo(e.prn)
    if geo:
        # Node in the inertial frame; the Earth's rotation follows after the tilt.
        node = e.right_ascension + e.right_ascension_rate * tk - earth_rotation * e.toe
    else:
        node = (
            e.right_ascension
            + (e.right_ascension_rate - earth_rotation) * tk
            - earth_rotation * e.toe
        )
    sin_node, cos_node = np.sin(node), np.cos(node)
    x = x_plane * cos_node - y_plane * cos_i * sin_node
    y = x_plane * sin_node + y_plane * cos_i * cos_node
    z = y_plane * sin_i
    if geo:
        cos_tilt, sin_tilt = math.cos(_BEIDOU_GEO_TILT), math.sin(_BEIDOU_GEO_TILT)
        y, z = y * cos_tilt + z * sin_tilt, -y * sin_tilt + z * cos_tilt
        sin_rot, cos_rot = np.sin(earth_rotation * tk), np.cos(earth_rotation * tk)
        x, y = x * cos_rot + y * sin_rot, -x * sin_rot + y * cos_rot

    positions = np.stack([x, y, z], axis=1)
    positions[np.abs(tk) > MAX_KEPLER_AGE_SECONDS] = np.nan
    return positions


def _glonass_dt(e: GlonassEphemeris, utc: np.ndarray) -> np.ndarray:
    """Time from tb, folded into +-half a day (tb is a time of day)."""
    time_of_day = (utc + MOSCOW_UTC_OFFSET_SECONDS) % SECONDS_PER_DAY
    half_day = SECONDS_PER_DAY / 2
    return (time_of_day - e.tb + half_day) % SECONDS_PER_DAY - half_day


def _glonass_derivative(
    state: tuple[float, ...], acceleration: tuple[float, float, float]
) -> tuple[float, ...]:
    """Equations of motion in the rotating PZ-90 frame: central body with J2,
    centrifugal and Coriolis terms, and the luni-solar acceleration taken as constant
    (GLONASS ICD, appendix A.3.1.2)."""
    x, y, z, vx, vy, vz = state
    r2 = x * x + y * y + z * z
    r = math.sqrt(r2)
    j2_term = 1.5 * _GLONASS_J2 * _GLONASS_GM * _GLONASS_EARTH_RADIUS**2 / (r2 * r2 * r)
    z_term = 5.0 * z * z / r2
    central = -_GLONASS_GM / (r2 * r) - j2_term * (1.0 - z_term)
    omega = _GLONASS_EARTH_ROTATION
    omega2 = omega * omega
    return (
        vx,
        vy,
        vz,
        (central + omega2) * x + 2.0 * omega * vy + acceleration[0],
        (central + omega2) * y - 2.0 * omega * vx + acceleration[1],
        (central - 2.0 * j2_term) * z + acceleration[2],
    )


def _runge_kutta_step(
    state: tuple[float, ...], step: float, acceleration: tuple[float, float, float]
) -> tuple[float, ...]:
    k1 = _glonass_derivative(state, acceleration)
    k2 = _glonass_derivative(
        tuple(s + 0.5 * step * k for s, k in zip(state, k1, strict=True)), acceleration
    )
    k3 = _glonass_derivative(
        tuple(s + 0.5 * step * k for s, k in zip(state, k2, strict=True)), acceleration
    )
    k4 = _glonass_derivative(
        tuple(s + step * k for s, k in zip(state, k3, strict=True)), acceleration
    )
    return tuple(
        s + step / 6.0 * (a + 2.0 * b + 2.0 * c + d)
        for s, a, b, c, d in zip(state, k1, k2, k3, k4, strict=True)
    )


def _glonass_positions(e: GlonassEphemeris, utc: np.ndarray) -> np.ndarray:
    """Integrates from tb forwards to the later times and backwards to the earlier ones,
    each time continuing from the previous target, so that a whole track costs no more
    steps than its farthest point."""
    dt = _glonass_dt(e, utc)
    positions = np.full((len(dt), 3), np.nan)
    order = np.argsort(dt)
    forward = [i for i in order if dt[i] >= 0]
    backward = [i for i in order[::-1] if dt[i] < 0]
    for indices in (forward, backward):
        state = (*e.position, *e.velocity)
        elapsed = 0.0
        for index in indices:
            target = float(dt[index])
            if abs(target) > MAX_GLONASS_AGE_SECONDS:
                break  # sorted by distance from tb: all further ones are too far as well
            while abs(target - elapsed) > 1e-6:
                step = max(
                    -GLONASS_INTEGRATION_STEP_SECONDS,
                    min(GLONASS_INTEGRATION_STEP_SECONDS, target - elapsed),
                )
                state = _runge_kutta_step(state, step, e.acceleration)
                elapsed += step
            positions[index] = state[:3]
    return positions


@dataclass(frozen=True)
class Observer:
    """A place on the Earth: geodetic latitude and longitude in degrees (WGS 84), height
    above the ellipsoid in metres."""

    latitude: float
    longitude: float
    height: float = 0.0

    def ecef(self) -> np.ndarray:
        lat, lon = math.radians(self.latitude), math.radians(self.longitude)
        n = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat) ** 2)
        return np.array(
            [
                (n + self.height) * math.cos(lat) * math.cos(lon),
                (n + self.height) * math.cos(lat) * math.sin(lon),
                (n * (1 - WGS84_E2) + self.height) * math.sin(lat),
            ]
        )


def look_angles(
    observer: Observer, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direction and distance from the observer to Earth-fixed positions (N, 3).

    Returns:
        Azimuth in degrees (0 = north, 90 = east, 0..360), elevation above the horizon
        in degrees, distance in metres; NaN where the position is NaN.
    """
    lat, lon = math.radians(observer.latitude), math.radians(observer.longitude)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    d = np.atleast_2d(positions) - observer.ecef()
    east = -sin_lon * d[:, 0] + cos_lon * d[:, 1]
    north = -sin_lat * cos_lon * d[:, 0] - sin_lat * sin_lon * d[:, 1] + cos_lat * d[:, 2]
    up = cos_lat * cos_lon * d[:, 0] + cos_lat * sin_lon * d[:, 1] + sin_lat * d[:, 2]
    azimuth = np.degrees(np.arctan2(east, north)) % 360.0
    elevation = np.degrees(np.arctan2(up, np.hypot(east, north)))
    return azimuth, elevation, np.sqrt(east**2 + north**2 + up**2)
