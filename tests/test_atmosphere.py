"""Tests for the model troposphere of SSRZ (`dab.atmosphere`)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dab.atmosphere import (
    ZenithDelays,
    corrected_zenith_delays,
    day_of_year,
    geoid_height,
    model_zenith_delays,
    mops_weather,
    summed_scale_factors,
)
from dab.ssrz import TROPOSPHERE_DRY, TROPOSPHERE_TOTAL, TROPOSPHERE_WET, SsrzDecoder
from tests.ssrz_vectors import (
    ZM004_GRID_TROPOSPHERE,
    ZM007_REGIONAL_TROPOSPHERE,
    ZM009_TIME,
    ZM011_SATELLITE_GROUPS,
    ZM012_METADATA,
    ZM013_GRID_3,
)

# The day on which the seasonal terms vanish in the north: 28 + 365.25 / 4.
NO_SEASON = 28 + 365.25 / 4


def test_geoid_heights_at_the_nodes_and_between() -> None:
    assert geoid_height(50.0, 10.0) == pytest.approx(48.0)
    assert geoid_height(60.0, 20.0) == pytest.approx(21.0)
    # Halfway between 50 N 10 E (48 m) and 60 N 10 E (41 m).
    assert geoid_height(55.0, 10.0) == pytest.approx(44.5)
    np.testing.assert_allclose(geoid_height([50.0, 51.0], [10.0, 10.0]), [48.0, 47.3])


def test_mops_weather_is_the_table_without_season() -> None:
    weather = mops_weather(45.0, NO_SEASON)
    assert weather.pressure == pytest.approx(1015.75)
    assert weather.temperature == pytest.approx(283.15)
    assert weather.water_vapour == pytest.approx(11.66)
    assert weather.lapse_rate == pytest.approx(0.00558)
    assert weather.vapour_lapse == pytest.approx(2.57)


def test_mops_weather_between_latitudes_and_in_winter() -> None:
    # Halfway between 45 and 60 degrees, on 28 January: the seasonal minimum.
    weather = mops_weather(52.5, 28.0)
    assert weather.temperature == pytest.approx((283.15 + 272.15) / 2 - (11.0 + 15.0) / 2)
    assert weather.water_vapour == pytest.approx((11.66 + 6.78) / 2 - (7.24 + 5.36) / 2)
    # The south has its minimum half a year later.
    assert mops_weather(-45.0, 211.0).temperature == pytest.approx(283.15 - 11.0)


def test_model_zenith_delays_at_sea_level() -> None:
    # On the geoid (h = N) at 45 degrees the gravity correction is 1.
    height = geoid_height(45.0, 10.0)
    delays = model_zenith_delays(45.0, 10.0, height, NO_SEASON)
    assert delays.hydrostatic == pytest.approx(0.0022768 * 1015.75)
    assert delays.wet == pytest.approx(0.0022768 * (1255.0 / 283.15 + 0.05) * 11.66)
    assert delays.total == pytest.approx(delays.hydrostatic + delays.wet)


def test_model_zenith_delays_decrease_with_height() -> None:
    height = geoid_height(45.0, 10.0) + 1000.0
    delays = model_zenith_delays(45.0, 10.0, height, NO_SEASON)
    theta = 1.0 - 0.00558 * 1000.0 / 283.15
    exponent = 9.80665 / (287.0537625 * 0.00558)
    pressure = 1015.75 * theta**exponent
    assert delays.hydrostatic == pytest.approx(0.0022768 * pressure / (1.0 - 0.00028))
    vapour = 11.66 * theta ** (3.57 * exponent)
    temperature = 283.15 - 5.58
    assert delays.wet == pytest.approx(0.0022768 * (1255.0 / temperature + 0.05) * vapour)
    # About 12 % less pressure per kilometre.
    assert 0.87 < pressure / 1015.75 < 0.90


def test_water_vapour_below_the_geoid_is_not_raised() -> None:
    height = geoid_height(45.0, 10.0) - 100.0
    below = model_zenith_delays(45.0, 10.0, height, NO_SEASON)
    at_sea_level = model_zenith_delays(45.0, 10.0, height + 100.0, NO_SEASON)
    assert below.hydrostatic > at_sea_level.hydrostatic
    # Only the temperature term still changes.
    assert below.wet == pytest.approx(at_sea_level.wet, rel=5e-3)


def test_scale_factors_apply_per_component() -> None:
    model = ZenithDelays(np.array([2.3]), np.array([0.1]))
    corrected = corrected_zenith_delays(
        model, {TROPOSPHERE_DRY: np.array([0.01]), TROPOSPHERE_WET: np.array([0.5])}
    )
    assert corrected.hydrostatic[0] == pytest.approx(2.323)
    assert corrected.wet[0] == pytest.approx(0.15)
    total = corrected_zenith_delays(model, {TROPOSPHERE_TOTAL: np.array([0.1])})
    assert total.total[0] == pytest.approx(2.4 * 1.1)


def test_summed_scale_factors_of_gepos() -> None:
    decoder = SsrzDecoder()
    for frame in (ZM009_TIME, ZM011_SATELLITE_GROUPS, ZM012_METADATA, ZM013_GRID_3):
        decoder.push(frame[3:-3])
    gridded = decoder.push(ZM004_GRID_TROPOSPHERE[3:-3])
    regional = decoder.push(ZM007_REGIONAL_TROPOSPHERE[3:-3])
    grid = decoder.grids[3]
    factors = summed_scale_factors(
        grid.latitudes, grid.longitudes, grid.heights, gridded=gridded, regional=regional
    )
    # The regional troposphere sends the dry part, the grid the wet part.
    np.testing.assert_allclose(factors[TROPOSPHERE_DRY], 0.00968)
    np.testing.assert_allclose(factors[TROPOSPHERE_WET], gridded.components[TROPOSPHERE_WET][0])
    delays = corrected_zenith_delays(
        model_zenith_delays(grid.latitudes, grid.longitudes, grid.heights, 272.8), factors
    )
    wet = delays.wet[np.isfinite(delays.wet)]
    # Lowland stations on an evening in September: 2.3 m dry, some 15 cm wet.
    assert np.all((2.25 < delays.hydrostatic) & (delays.hydrostatic < 2.35))
    assert np.all((0.10 < wet) & (wet < 0.20))


def test_day_of_year() -> None:
    # 29.09.2026, 18:00 UTC: day 272.
    assert day_of_year(1790704800.0) == pytest.approx(272.75)
    assert math.floor(day_of_year(1767225600.0)) == 1  # 1 January 2026
