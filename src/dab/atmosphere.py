"""The model troposphere of SSRZ and what the corrections of GEPOS make of it.

SSRZ does not send tropospheric delays but scale factors to a model troposphere that
sender and receiver compute alike (Geo++ SSRZ format 1.1.3, clause 11.5): average
weather after RTCA MOPS DO-229 for latitude and day of the year, carried to the height
of the station, and from it the zenith delays after Saastamoinen. A scale factor of
+0.3 for the wet component means 30 % more water vapour delay than the model has.

In the zenith every mapping function is 1, so the delay there is (eq. 11.56)

    ZTD = ZHD_model (1 + sum of dry factors) + ZWD_model (1 + sum of wet factors)
          + (ZHD_model + ZWD_model) * sum of total factors,

summed over the stages (regional and gridded). The mapping function improvement only
acts on satellites below 20 degrees and plays no part in the zenith.

The format description has two constants off by a factor of ten in eqs. 11.70 and
11.72; used here are those of Saastamoinen (1972) as everywhere else, 0.00266 and
0.0022768 (the reference decoder of Geo++ does the same).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from dab.ssrz import (
    TROPOSPHERE_DRY,
    TROPOSPHERE_TOTAL,
    TROPOSPHERE_WET,
    GridTroposphere,
    RegionalTroposphere,
)

# MOPS meteorological parameters at mean sea level (tables 11.3 and 11.4): latitude
# nodes, averages and seasonal amplitudes of pressure (hPa), temperature (K), water
# vapour pressure (hPa), temperature lapse rate (K/m) and water vapour lapse rate.
# fmt: off
_LATITUDES = np.array([15.0, 30.0, 45.0, 60.0, 75.0])
_AVERAGE = np.array([
    # P       T       e      beta     lambda
    [1013.25, 299.65, 26.31, 0.00630, 2.77],
    [1017.25, 294.15, 21.79, 0.00605, 3.15],
    [1015.75, 283.15, 11.66, 0.00558, 2.57],
    [1011.75, 272.15,  6.78, 0.00539, 1.81],
    [1013.00, 263.65,  4.11, 0.00453, 1.55],
])
_SEASONAL = np.array([
    [ 0.00,  0.00, 0.00, 0.00000, 0.00],
    [-3.75,  7.00, 8.85, 0.00025, 0.33],
    [-2.25, 11.00, 7.24, 0.00032, 0.46],
    [-1.75, 15.00, 5.36, 0.00081, 0.74],
    [-0.50, 14.50, 3.39, 0.00062, 0.30],
])
# fmt: on
# Day of the year of the minimum of the seasonal term: 28 in the north, 211 in the south.
_MINIMUM_DAY_NORTH = 28.0
_MINIMUM_DAY_SOUTH = 211.0
_DAYS_PER_YEAR = 365.25

_GAS_CONSTANT_DRY = 287.0537625  # R_d in J/(kg K)
_GRAVITY = 9.80665  # g_0 in m/s^2

# Height of the WGS 84 geoid above the ellipsoid in 10 degree steps (DMA TR 8350.2),
# the grid eq. 11.57 names, cut out for Central Europe: latitudes 40, 50, 60 degrees
# north, longitudes 10 W to 30 E.
_GEOID_LATITUDES = np.array([40.0, 50.0, 60.0])
_GEOID_LONGITUDES = np.array([-10.0, 0.0, 10.0, 20.0, 30.0])
# fmt: off
_GEOID_HEIGHTS = np.array([
    [51.0, 52.0, 48.0, 35.0, 40.0],
    [59.0, 47.0, 48.0, 42.0, 28.0],
    [57.0, 47.0, 41.0, 21.0, 18.0],
])
# fmt: on


def geoid_height(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Geoid height N in m, bilinear in the 10 degree grid. Outside Central Europe the
    values at its edge (the service area of GEPOS lies well inside)."""
    latitude = np.clip(np.asarray(latitude, dtype=float), 40.0, 60.0)
    longitude = np.clip(np.asarray(longitude, dtype=float), -10.0, 30.0)
    row = np.clip(((latitude - 40.0) // 10.0).astype(int), 0, 1)
    column = np.clip(((longitude + 10.0) // 10.0).astype(int), 0, 3)
    u = (latitude - _GEOID_LATITUDES[row]) / 10.0
    v = (longitude - _GEOID_LONGITUDES[column]) / 10.0
    heights = _GEOID_HEIGHTS
    return (
        (1 - u) * (1 - v) * heights[row, column]
        + (1 - u) * v * heights[row, column + 1]
        + u * (1 - v) * heights[row + 1, column]
        + u * v * heights[row + 1, column + 1]
    )


@dataclass(frozen=True)
class Weather:
    """MOPS weather at mean sea level (eqs. 11.58 to 11.65)."""

    pressure: np.ndarray  # hPa
    temperature: np.ndarray  # K
    water_vapour: np.ndarray  # hPa
    lapse_rate: np.ndarray  # K/m
    vapour_lapse: np.ndarray  # dimensionless


def mops_weather(latitude: np.ndarray, day_of_year: float) -> Weather:
    """The five parameters for a latitude (degrees) and day of the year, linear in
    latitude between the nodes and constant beyond 15 and 75 degrees."""
    latitude = np.asarray(latitude, dtype=float)
    absolute = np.abs(latitude)
    minimum_day = np.where(latitude >= 0, _MINIMUM_DAY_NORTH, _MINIMUM_DAY_SOUTH)
    season = np.cos(2.0 * math.pi * (day_of_year - minimum_day) / _DAYS_PER_YEAR)
    values = [
        np.interp(absolute, _LATITUDES, _AVERAGE[:, column])
        - np.interp(absolute, _LATITUDES, _SEASONAL[:, column]) * season
        for column in range(5)
    ]
    return Weather(*values)


@dataclass(frozen=True)
class ZenithDelays:
    """Zenith delays in m."""

    hydrostatic: np.ndarray  # dry part
    wet: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.hydrostatic + self.wet


def model_zenith_delays(
    latitude: np.ndarray, longitude: np.ndarray, height: np.ndarray, day_of_year: float
) -> ZenithDelays:
    """The model troposphere of SSRZ at ellipsoidal positions (degrees, m): MOPS weather
    carried to the orthometric height (eqs. 11.66 to 11.69), zenith delays after
    Saastamoinen (eqs. 11.70 to 11.72)."""
    latitude = np.asarray(latitude, dtype=float)
    orthometric = np.asarray(height, dtype=float) - geoid_height(latitude, longitude)
    weather = mops_weather(latitude, day_of_year)
    beta = weather.lapse_rate
    temperature = weather.temperature - beta * orthometric
    theta = temperature / weather.temperature
    exponent = _GRAVITY / (_GAS_CONSTANT_DRY * beta)
    pressure = weather.pressure * theta**exponent
    # Below the geoid the water vapour pressure is not raised above its sea level value
    # (eq. 11.69).
    vapour = weather.water_vapour * np.minimum(
        theta ** ((weather.vapour_lapse + 1) * exponent), 1.0
    )
    gravity = 1.0 - 0.00266 * np.cos(2.0 * np.radians(latitude)) - 0.00000028 * orthometric
    hydrostatic = 0.0022768 * pressure / gravity
    wet = 0.0022768 * (1255.0 / temperature + 0.05) * vapour
    return ZenithDelays(hydrostatic, wet)


def corrected_zenith_delays(
    model: ZenithDelays,
    scale_factors: dict[str, np.ndarray],
) -> ZenithDelays:
    """Zenith delays after applying the summed scale factors per component (dry, wet,
    total); a total factor is split onto both parts in proportion (eq. 11.56)."""
    shape = np.shape(model.hydrostatic)
    dry = np.broadcast_to(scale_factors.get(TROPOSPHERE_DRY, 0.0), shape)
    wet = np.broadcast_to(scale_factors.get(TROPOSPHERE_WET, 0.0), shape)
    total = np.broadcast_to(scale_factors.get(TROPOSPHERE_TOTAL, 0.0), shape)
    return ZenithDelays(model.hydrostatic * (1.0 + dry + total), model.wet * (1.0 + wet + total))


def summed_scale_factors(
    latitude: np.ndarray,
    longitude: np.ndarray,
    height: np.ndarray,
    gridded: GridTroposphere | None = None,
    regional: RegionalTroposphere | None = None,
) -> dict[str, np.ndarray]:
    """Scale factors per component at the given positions, summed over the regional
    stage (evaluated there) and the gridded stage (whose points the positions must
    be). A missing gridded value counts as NaN."""
    factors: dict[str, np.ndarray] = {}
    shape = np.broadcast(latitude, longitude, height).shape
    if regional is not None:
        for region in regional.regions:
            for component in region.components:
                value = region.scale_factor(component, latitude, longitude, height)
                if value is not None:
                    factors[component] = factors.get(component, np.zeros(shape)) + value
    if gridded is not None:
        for component, values in gridded.components.items():
            factors[component] = factors.get(component, np.zeros(shape)) + values[0]
    return factors


def day_of_year(posix_time: float) -> float:
    """Day of the year (1.0 = 1 January, 0:00 UTC) of a POSIX time."""
    return time.gmtime(posix_time).tm_yday + (posix_time % 86400.0) / 86400.0
