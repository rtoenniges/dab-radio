"""The corrections of GEPOS at a place (`dab.corrections`), against the "state space
influence" of the reference decoder of Geo++ for the test epoch of `tests.ssrz_vectors`
(18:39:55 GPS time, 52.5° N 9.5° E, 100 m)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dab.atmosphere import day_of_year, model_zenith_delays
from dab.corrections import (
    GRID_IONOSPHERE_LAYER,
    IONOSPHERE_CONSTANT,
    CorrectionState,
    SatelliteCorrections,
    _satellite_state,
    inverse_distance,
    pierce_point,
    reference_frequency,
    transversal_offsets,
)
from dab.gmf import global_mapping_function
from dab.gnss import GPS_EPOCH_POSIX, Observer
from dab.rtcm import decode_ephemeris
from dab.ssrz import SsrzDecoder
from tests.ssrz_vectors import (
    EPHEMERIS_FRAMES,
    ZM001_HIGH_RATE,
    ZM002_LOW_RATE,
    ZM003_GRID_IONOSPHERE,
    ZM004_GRID_TROPOSPHERE,
    ZM005_REGIONAL_IONOSPHERE,
    ZM006_GLOBAL_VTEC,
    ZM007_REGIONAL_TROPOSPHERE,
    ZM009_TIME,
    ZM010_STREAM_ID,
    ZM011_SATELLITE_GROUPS,
    ZM012_METADATA,
    ZM013_GRID_3,
    ZM024_LEAP_SECOND,
)

EPOCH = 2438 * 604800 + 239995.0  # GPS time of ZM009
OBSERVER = Observer(52.5, 9.5, 100.0)

# In stream order: metadata, then the corrections of time tag 600 and the high rate clock
# of 605.
FRAMES = (
    ZM009_TIME,
    ZM010_STREAM_ID,
    ZM024_LEAP_SECOND,
    ZM011_SATELLITE_GROUPS,
    ZM012_METADATA,
    ZM013_GRID_3,
    ZM002_LOW_RATE,
    ZM006_GLOBAL_VTEC,
    ZM007_REGIONAL_TROPOSPHERE,
    ZM005_REGIONAL_IONOSPHERE,
    ZM004_GRID_TROPOSPHERE,
    ZM003_GRID_IONOSPHERE,
    ZM001_HIGH_RATE,
)

# The reference decoder at this epoch and place: elevation, azimuth (degrees), orbit
# correction along the line from satellite to receiver (m), clock dC (m), slant TEC of
# the global and regional stage (TECU), regional troposphere (m), model troposphere with
# mapping function improvement (m), phase bias of the reference signal (m).
REFERENCE = {
    "G06": (22.96571, 43.84934, 0.00235, -3.1754, 15.44257, -2.83068, 0.05642, 6.13992, 1.71121),
    "R02": (15.44499, 243.14937, -0.43941, 2.2796, 27.74522, -0.84176, 0.08193, 8.91857, 0.64533),
    "E03": (34.50972, 299.74724, -0.10586, -0.9905, 14.81616, -2.00675, 0.03902, 4.24575, 1.79753),
    "C11": (18.92635, 41.96288, -0.1702, -0.6384, 16.7574, -3.56045, 0.06763, 7.36106, 0.2603),
}


@pytest.fixture(scope="module")
def state() -> CorrectionState:
    state = CorrectionState()
    for frame in EPHEMERIS_FRAMES.values():
        state.add_ephemeris(decode_ephemeris(frame[3:-3]))
    decoder = SsrzDecoder()
    for frame in FRAMES:
        result = decoder.push(frame[3:-3])
        assert result is not None
        state.add_ssrz(result)
    return state


@pytest.fixture(scope="module")
def corrections(state: CorrectionState) -> dict[str, SatelliteCorrections]:
    return {c.name: c for c in state.evaluate(OBSERVER, EPOCH)}


def test_only_satellites_with_the_named_ephemeris(corrections) -> None:
    assert set(corrections) == set(EPHEMERIS_FRAMES)


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_agrees_with_the_reference_decoder(corrections, name: str) -> None:
    elevation, azimuth, orbit, clock, gvi, rsi, rt, model, phase = REFERENCE[name]
    c = corrections[name]
    assert c.elevation == pytest.approx(elevation, abs=1e-4)
    assert c.azimuth == pytest.approx(azimuth, abs=1e-3)
    # The reference projects onto the line from the satellite to the receiver.
    assert c.orbit_range == pytest.approx(-orbit, abs=1e-4)
    assert c.clock == pytest.approx(clock, abs=1e-4)
    assert c.stec["GVI"] == pytest.approx(gvi, abs=1e-3)
    assert c.stec["RSI"] == pytest.approx(rsi, abs=1e-3)
    assert c.troposphere_parts["RT"] == pytest.approx(rt, abs=1e-4)
    assert c.troposphere_model + c.troposphere_parts["MFI"] == pytest.approx(model, abs=1e-3)
    signal = "2I" if name[0] == "C" else "1C"
    wavelength = 299_792_458.0 / c.frequency
    assert c.phase_biases[signal] * wavelength == pytest.approx(phase, abs=1e-4)


def test_clock_is_low_plus_high_rate(corrections) -> None:
    g06 = corrections["G06"]
    assert g06.clock_high_rate == pytest.approx(0.0146)
    assert g06.clock == pytest.approx(-3.19 + 0.0146)


def test_gridded_stages_interpolate_the_grid_at_the_place(state, corrections) -> None:
    """Grid 3 is the only one here. The ionosphere residual and the wet scale factor are
    interpolated to the place and mapped to the slant."""
    stored = state._stored
    grid = stored.grids[3]
    receiver = OBSERVER.ecef()
    posix = EPOCH - 18 + GPS_EPOCH_POSIX
    zenith = model_zenith_delays(52.5, 9.5, 100.0, day_of_year(posix))
    for name, c in corrections.items():
        gridded = stored.grid_ionosphere[3]
        values = gridded.values[gridded.satellites.index(name)]
        ephemeris = state.ephemeris(name, c.ephemeris_iod)
        position = _satellite_state(ephemeris, EPOCH, receiver)[0]
        slant = pierce_point(receiver, position, GRID_IONOSPHERE_LAYER)[2]
        assert c.stec["GRI"] == pytest.approx(slant * inverse_distance(grid, values, OBSERVER))

        factor = inverse_distance(grid, stored.grid_troposphere[3].components["wet"][0], OBSERVER)
        mjd = posix / 86400 + 40587
        wet = global_mapping_function(mjd, 52.5, 9.5, 100.0, c.elevation)[1]
        assert c.troposphere_parts["GRT"] == pytest.approx(factor * zenith.wet * wet)


def test_ionosphere_delay_on_the_reference_signal(corrections) -> None:
    g06 = corrections["G06"]
    assert g06.frequency == 1575.42e6
    assert g06.ionosphere == pytest.approx(IONOSPHERE_CONSTANT / 1575.42e6**2 * g06.total_stec)
    # GLONASS: G1 of the satellite's frequency channel (R02: -4).
    assert corrections["R02"].frequency == pytest.approx(1602e6 - 4 * 0.5625e6)
    assert reference_frequency("C11") == 1561.098e6


def test_troposphere_is_model_plus_stages(corrections) -> None:
    for c in corrections.values():
        assert c.troposphere == pytest.approx(
            c.troposphere_model + sum(c.troposphere_parts.values())
        )
        assert 2.3 < c.troposphere_model < 10.0


def test_nothing_when_the_corrections_are_old(state) -> None:
    assert state.evaluate(OBSERVER, EPOCH + 3600) == []


def test_inverse_distance_at_a_station_and_without_values(state) -> None:
    grid = state._stored.grids[3]
    values = np.arange(grid.point_count, dtype=float)
    at_station = Observer(grid.latitudes[4], grid.longitudes[4])
    assert inverse_distance(grid, values, at_station) == 4.0
    values[:] = math.nan
    assert math.isnan(inverse_distance(grid, values, OBSERVER))


def test_transversal_offsets_north_and_east() -> None:
    lat, lon = math.radians(51.5), math.radians(10.0)
    north, east = transversal_offsets(lat, lon, lat + 0.01, lon)
    assert north == pytest.approx(0.01, rel=1e-6)
    assert east == pytest.approx(0.0, abs=1e-12)
    north, east = transversal_offsets(lat, lon, lat, lon + 0.01)
    assert east == pytest.approx(0.01 * math.cos(lat), rel=1e-3)
    assert north > 0  # a small circle of latitude bends north of the great circle
    north, _ = transversal_offsets(lat, lon, lat - 0.01, lon)
    assert north == pytest.approx(-0.01, rel=1e-6)


def test_pierce_point_straight_up() -> None:
    receiver = Observer(0.0, 0.0, 0.0).ecef()
    satellite = receiver * 4
    lat, lon, slant = pierce_point(receiver, satellite, 400e3)
    assert (lat, lon) == pytest.approx((0.0, 0.0), abs=1e-12)
    assert slant == pytest.approx(1.0)
    assert pierce_point(receiver, -satellite, 400e3) is None
