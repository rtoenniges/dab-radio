"""Tests for the orbit computation from broadcast ephemerides and the look angles."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from dab.gnss import (
    MAX_GLONASS_AGE_SECONDS,
    MAX_KEPLER_AGE_SECONDS,
    Observer,
    ephemeris_age,
    gps_time,
    look_angles,
    orbit_ring,
    satellite_positions,
)
from dab.rtcm import GNSS_BEIDOU, KeplerEphemeris, decode_ephemeris
from tests.rtcm_vectors import (
    BEIDOU_C08_FRAME,
    BEIDOU_C08_PRECISE_KM,
    GALILEO_E02_FRAME,
    GALILEO_E02_PRECISE_KM,
    GLONASS_R11_FRAME,
    GLONASS_R11_PRECISE_KM,
    GPS_G12_FRAME,
    GPS_G12_PRECISE_KM,
    REFERENCE_EPOCH_UTC,
)

HANNOVER = Observer(52.3759, 9.7320, 60.0)


def _ephemeris(frame: bytes):
    return decode_ephemeris(frame[3:-3])


@pytest.mark.parametrize(
    ("frame", "precise_km", "tolerance_m"),
    [
        (GPS_G12_FRAME, GPS_G12_PRECISE_KM, 3.0),
        (GALILEO_E02_FRAME, GALILEO_E02_PRECISE_KM, 3.0),
        (BEIDOU_C08_FRAME, BEIDOU_C08_PRECISE_KM, 3.0),
        # 15 minutes of integration from tb.
        (GLONASS_R11_FRAME, GLONASS_R11_PRECISE_KM, 5.0),
    ],
)
def test_broadcast_orbit_matches_precise_orbit(frame, precise_km, tolerance_m) -> None:
    position = satellite_positions(_ephemeris(frame), REFERENCE_EPOCH_UTC)[0]
    assert np.linalg.norm(position - np.array(precise_km) * 1000) < tolerance_m


def test_gps_time() -> None:
    # 2026-09-28 19:30:00 GPS time: week 2438, Monday 19:30.
    week, second = divmod(gps_time(REFERENCE_EPOCH_UTC), 604_800)
    assert (week, second) == (2438, 86_400 + 19.5 * 3600)


def test_ephemeris_age() -> None:
    gps = _ephemeris(GPS_G12_FRAME)
    assert ephemeris_age(gps, REFERENCE_EPOCH_UTC) == pytest.approx(-1800.0)
    # BeiDou time runs 14 s behind GPS time.
    assert ephemeris_age(_ephemeris(BEIDOU_C08_FRAME), REFERENCE_EPOCH_UTC) == pytest.approx(5386.0)
    glonass = _ephemeris(GLONASS_R11_FRAME)
    assert ephemeris_age(glonass, REFERENCE_EPOCH_UTC) == pytest.approx(882.0)
    # tb is a time of day in Moscow time: 00:15 there is 21:15 UTC the day before.
    early = dataclasses.replace(glonass, tb=900)
    utc_21_20 = REFERENCE_EPOCH_UTC + 6618  # 19:29:42 UTC + 1:50:18
    assert ephemeris_age(early, utc_21_20) == pytest.approx(300.0)
    assert ephemeris_age(early, utc_21_20 - 1800) == pytest.approx(-1500.0)


def test_stale_ephemeris_gives_no_position() -> None:
    gps = _ephemeris(GPS_G12_FRAME)
    toe_utc = REFERENCE_EPOCH_UTC + 1800
    times = np.array([toe_utc, toe_utc + MAX_KEPLER_AGE_SECONDS + 60])
    positions = satellite_positions(gps, times)
    assert np.all(np.isfinite(positions[0]))
    assert np.all(np.isnan(positions[1]))

    glonass = _ephemeris(GLONASS_R11_FRAME)
    tb_utc = REFERENCE_EPOCH_UTC - 882
    positions = satellite_positions(glonass, tb_utc + np.array([0.0, MAX_GLONASS_AGE_SECONDS + 60]))
    assert np.allclose(positions[0], glonass.position)
    assert np.all(np.isnan(positions[1]))


def test_glonass_track_equals_single_positions() -> None:
    """The track integrates on from target to target, forwards and backwards."""
    glonass = _ephemeris(GLONASS_R11_FRAME)
    tb_utc = REFERENCE_EPOCH_UTC - 882
    times = tb_utc + np.array([1500.0, -700.0, 0.0, 90.0, -3000.0, 3600.0])
    track = satellite_positions(glonass, times)
    for time, position in zip(times, track, strict=True):
        assert np.allclose(position, satellite_positions(glonass, time)[0], atol=1e-3)
    # The orbit radius stays that of GLONASS (25 500 km).
    assert np.all(np.abs(np.linalg.norm(track, axis=1) - 25.5e6) < 0.1e6)


def test_beidou_geo_stays_above_the_equator() -> None:
    """A geostationary orbit, described the BeiDou way: elements of an orbit inclined by
    5 degrees with its ascending node at 180 degrees, which the rotation by -5 degrees
    about the x axis brings back to the equator. With the wrong sign of the rotation the
    inclination would double and the satellite swing 7 000 km north and south."""
    earth_rotation = 7.292115e-5
    radius = (3.986004418e14 / earth_rotation**2) ** (1 / 3)
    geo = KeplerEphemeris(
        system=GNSS_BEIDOU, prn=3, week=1082, iod=1, toe=0.0, sqrt_a=math.sqrt(radius),
        eccentricity=0.0, inclination=math.radians(5.0), inclination_rate=0.0,
        right_ascension=math.pi, right_ascension_rate=0.0, perigee=0.0, mean_anomaly=0.0,
        mean_motion_difference=0.0, cuc=0.0, cus=0.0, crc=0.0, crs=0.0, cic=0.0, cis=0.0,
        healthy=True,
    )  # fmt: skip
    # toe = 0 is the start of the BeiDou week: GPS week start plus 14 s.
    gps_week_start = REFERENCE_EPOCH_UTC - (86_400 + 19.5 * 3600)
    times = gps_week_start + 14 + np.arange(-3, 4) * 1200.0
    positions = satellite_positions(geo, times)
    assert np.all(np.abs(positions[:, 2]) < 1000)
    longitudes = np.degrees(np.arctan2(positions[:, 1], positions[:, 0])) % 360
    assert np.allclose(longitudes, 180.0, atol=0.01)
    assert np.allclose(np.linalg.norm(positions, axis=1), radius)


def _distance_to_ring(ring: np.ndarray, point: np.ndarray) -> float:
    """From a point to the nearest of the straight pieces between the ring's points."""
    start, way = ring[:-1], np.diff(ring, axis=0)
    t = np.clip(np.einsum("ij,ij->i", point - start, way) / np.einsum("ij,ij->i", way, way), 0, 1)
    return float(np.min(np.linalg.norm(start + way * t[:, None] - point, axis=1)))


@pytest.mark.parametrize(
    "frame", [GPS_G12_FRAME, GALILEO_E02_FRAME, BEIDOU_C08_FRAME, GLONASS_R11_FRAME]
)
def test_the_orbit_ring_holds_the_satellite_for_an_hour(frame) -> None:
    """The ellipse begins where the satellite stands. As it lies in space, the satellite
    runs along it: its positions over the next hour, turned back by the Earth's rotation
    since, stay within a kilometre (the Earth's flattening moves the true orbit away from
    the ellipse by 0.3 km in an hour, measured 07.10.2026)."""
    ephemeris = _ephemeris(frame)
    ring = orbit_ring(ephemeris, REFERENCE_EPOCH_UTC, 720)
    assert ring.shape == (721, 3)
    position = satellite_positions(ephemeris, REFERENCE_EPOCH_UTC)[0]
    assert np.allclose(ring[0], position, atol=1e-3)
    assert np.allclose(ring[-1], position, atol=1e-3)
    for seconds in range(300, 3601, 300):
        later = satellite_positions(ephemeris, REFERENCE_EPOCH_UTC + seconds)[0]
        turn = 7.2921151467e-5 * seconds
        cos, sin = math.cos(turn), math.sin(turn)
        turned = np.array(
            [later[0] * cos - later[1] * sin, later[0] * sin + later[1] * cos, later[2]]
        )
        assert _distance_to_ring(ring, turned) < 1000.0
    # The satellite moves on along the ring, not back.
    assert np.linalg.norm(ring[1] - later) < np.linalg.norm(ring[-2] - later)


def test_the_orbit_ring_of_a_geostationary_beidou_satellite_is_the_equator() -> None:
    """Elements of an orbit inclined by 5 degrees, turned back into the equator (see
    `test_beidou_geo_stays_above_the_equator`): the ring is a circle in the equator, as
    far out as the satellite."""
    earth_rotation = 7.292115e-5
    radius = (3.986004418e14 / earth_rotation**2) ** (1 / 3)
    geo = KeplerEphemeris(
        system=GNSS_BEIDOU, prn=3, week=1082, iod=1, toe=0.0, sqrt_a=math.sqrt(radius),
        eccentricity=0.0, inclination=math.radians(5.0), inclination_rate=0.0,
        right_ascension=math.pi, right_ascension_rate=0.0, perigee=0.0, mean_anomaly=0.0,
        mean_motion_difference=0.0, cuc=0.0, cus=0.0, crc=0.0, crs=0.0, cic=0.0, cis=0.0,
        healthy=True,
    )  # fmt: skip
    gps_week_start = REFERENCE_EPOCH_UTC - (86_400 + 19.5 * 3600)
    ring = orbit_ring(geo, gps_week_start + 14 + 600, 90)
    assert np.all(np.abs(ring[:, 2]) < 1000)
    # Two-body orbit with GPS' gravitational constant: a few metres from BeiDou's radius.
    assert np.allclose(np.linalg.norm(ring, axis=1), radius, atol=50)
    # Eastwards, as the Earth turns.
    assert np.cross(ring[0], ring[1])[2] > 0


def test_a_stale_ephemeris_gives_no_orbit_ring() -> None:
    gps = _ephemeris(GPS_G12_FRAME)
    late = REFERENCE_EPOCH_UTC + 1800 + MAX_KEPLER_AGE_SECONDS + 60
    assert np.all(np.isnan(orbit_ring(gps, late, 10)))


def test_look_angles() -> None:
    at_origin = Observer(0.0, 0.0)
    ground = 6_378_137.0
    positions = np.array(
        [
            [ground + 20e6, 0.0, 0.0],  # overhead
            [ground, 0.0, 1000.0],  # north on the horizon
            [ground, 1000.0, 0.0],  # east on the horizon
            [ground + 1000.0, -1000.0, 0.0],  # 45 degrees high in the west
            [np.nan, np.nan, np.nan],
        ]
    )
    azimuth, elevation, distance = look_angles(at_origin, positions)
    assert elevation[0] == pytest.approx(90.0)
    assert distance[0] == pytest.approx(20e6)
    assert (azimuth[1], elevation[1]) == (pytest.approx(0.0), pytest.approx(0.0, abs=1e-3))
    assert (azimuth[2], elevation[2]) == (pytest.approx(90.0), pytest.approx(0.0, abs=1e-3))
    assert (azimuth[3], elevation[3]) == (pytest.approx(270.0), pytest.approx(45.0))
    assert np.isnan(azimuth[4]) and np.isnan(elevation[4])


def test_satellites_over_hannover() -> None:
    """The broadcast orbit puts the satellites where the precise orbit has them."""
    for frame, precise_km in (
        (GPS_G12_FRAME, GPS_G12_PRECISE_KM),
        (GALILEO_E02_FRAME, GALILEO_E02_PRECISE_KM),
    ):
        broadcast = satellite_positions(_ephemeris(frame), REFERENCE_EPOCH_UTC)
        azimuth, elevation, _ = look_angles(HANNOVER, broadcast)
        reference = look_angles(HANNOVER, np.array([precise_km]) * 1000)
        assert azimuth[0] == pytest.approx(reference[0][0], abs=0.001)
        assert elevation[0] == pytest.approx(reference[1][0], abs=0.001)
        assert elevation[0] > 30
