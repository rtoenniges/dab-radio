"""Global Mapping Function (GMF) of Böhm et al. (2006) for the troposphere.

SSRZ maps its model zenith delays to the slant with the GMF (Geo++ SSRZ format 1.1.3,
eqs. 11.73 and 11.74). The GMF has the continued fraction form of Marini, as Niell (1996)
uses it, for the hydrostatic and the wet delay each:

    m(E) = (1 + a / (1 + b / (1 + c))) / (sin E + a / (sin E + b / (sin E + c)))

b and c are constants (c of the hydrostatic part varies with latitude and season, as in
VMF1). The coefficient a varies with place and season: its mean and its annual amplitude
are each expanded in spherical harmonics up to degree and order 9 over the globe, fitted
to numerical weather model data of the ECMWF; the annual term has its reference day on
28 January (Niell 1996). The hydrostatic function is corrected for the height of the
station after Niell (1996).

The 440 coefficients of the expansion and the constants of the continued fractions are
the result of that fit and cannot be derived from the paper; they are taken from the
author's own routine `gmf.f` (Johannes Böhm, TU Wien, revision of 2011-07-21,
https://vmf.geo.tuwien.ac.at/codes/gmf.f, read 2026-10-11), who permitted their use in
this program and their publication under GPL-3.0-or-later (October 2026). The code of
this module is written from the papers and is not a translation of that routine; see
`THIRD-PARTY.md`.

References:
    Böhm, J., A. E. Niell, P. Tregoning, H. Schuh (2006): Global Mapping Function (GMF):
    A new empirical mapping function based on numerical weather model data. Geophysical
    Research Letters 33, L07304, doi:10.1029/2005GL025545.

    Niell, A. E. (1996): Global mapping functions for the atmosphere delay at radio
    wavelengths. Journal of Geophysical Research 101 (B2), 3227-3246.
"""

from __future__ import annotations

import math

import numpy as np

# Reference day of the annual term, 28 January 1980 (Niell 1996), as Modified Julian Date.
_REFERENCE_MJD = 44266.0
_DAYS_PER_YEAR = 365.25
# Degree and order of the spherical harmonic expansion of the coefficient a.
_DEGREE = 9

# Coefficients of the expansion of a, in units of 1e-5: 55 terms in the order n = 0..9
# outer, m = 0..n inner, for the cosine (A) and sine (B) terms of the mean (MEAN) and of
# the annual amplitude (AMP), for the hydrostatic (H) and the wet (W) function. The
# associated Legendre functions are the unnormalised ones without the Condon-Shortley
# phase (Heiskanen and Moritz 1967).
# fmt: off
_AH_MEAN = np.array((
    +1.2517e+02, +8.5030e-01, +6.9360e-02, -6.7600e+00, +1.7710e-01,
    +1.1300e-02, +5.9630e-01, +1.8080e-02, +2.8010e-03, -1.4140e-03,
    -1.2120e+00, +9.3000e-02, +3.6830e-03, +1.0950e-03, +4.6710e-05,
    +3.9590e-01, -3.8670e-02, +5.4130e-03, -5.2890e-04, +3.2290e-04,
    +2.0670e-05, +3.0000e-01, +2.0310e-02, +5.9000e-03, +4.5730e-04,
    -7.6190e-05, +2.3270e-06, +3.8450e-06, +1.1820e-01, +1.1580e-02,
    +5.4450e-03, +6.2190e-05, +4.2040e-06, -2.0930e-06, +1.5400e-07,
    -4.2800e-08, -4.7510e-01, -3.4900e-02, +1.7580e-03, +4.0190e-04,
    -2.7990e-06, -1.2870e-06, +5.4680e-07, +7.5800e-08, -6.3000e-09,
    -1.1600e-01, +8.3010e-03, +8.7710e-04, +9.9550e-05, -1.7180e-06,
    -2.0120e-06, +1.1700e-08, +1.7900e-08, -1.3000e-09, +1.0000e-10,
))
_BH_MEAN = np.array((
    +0.0000e+00, +0.0000e+00, +3.2490e-02, +0.0000e+00, +3.3240e-02,
    +1.8500e-02, +0.0000e+00, -1.1150e-01, +2.5190e-02, +4.9230e-03,
    +0.0000e+00, +2.7370e-02, +1.5950e-02, -7.3320e-04, +1.9330e-04,
    +0.0000e+00, -4.7960e-02, +6.3810e-03, -1.5990e-04, -3.6850e-04,
    +1.8150e-05, +0.0000e+00, +7.0330e-02, +2.4260e-03, -1.1110e-03,
    -1.3570e-04, -7.8280e-06, +2.5470e-06, +0.0000e+00, +5.7790e-03,
    +3.1330e-03, -5.3120e-04, -2.0280e-05, +2.3230e-07, -9.1000e-08,
    -1.6500e-08, +0.0000e+00, +3.6880e-02, -8.6380e-04, -8.5140e-05,
    -2.8280e-05, +5.4030e-07, +4.3900e-07, +1.3500e-08, +1.8000e-09,
    +0.0000e+00, -2.7360e-02, -2.9770e-04, +8.1130e-05, +2.3290e-07,
    +8.4510e-07, +4.4900e-08, -8.1000e-09, -1.5000e-09, +2.0000e-10,
))
_AH_AMP = np.array((
    -2.7380e-01, -2.8370e+00, +1.2980e-02, -3.5880e-01, +2.4130e-02,
    +3.4270e-02, -7.6240e-01, +7.2720e-02, +2.1600e-02, -3.3850e-03,
    +4.4240e-01, +3.7220e-02, +2.1950e-02, -1.5030e-03, +2.4260e-04,
    +3.0130e-01, +5.7620e-02, +1.0190e-02, -4.4760e-04, +6.7900e-05,
    +3.2270e-05, +3.1230e-01, -3.5350e-02, +4.8400e-03, +3.0250e-06,
    -4.3630e-05, +2.8540e-07, -1.2860e-06, -6.7250e-01, -3.7300e-02,
    +8.9640e-04, +1.3990e-04, -3.9900e-06, +7.4310e-06, -2.7960e-07,
    -1.6010e-07, +4.0680e-02, -1.3520e-02, +7.2820e-04, +9.5940e-05,
    +2.0700e-06, -9.6200e-08, -2.7420e-07, -6.3700e-08, -6.3000e-09,
    +8.6250e-02, -5.9710e-03, +4.7050e-04, +2.3350e-05, +4.2260e-06,
    +2.4750e-07, -8.8500e-08, -3.6000e-08, -2.9000e-09, +0.0000e+00,
))
_BH_AMP = np.array((
    +0.0000e+00, +0.0000e+00, -1.1360e-01, +0.0000e+00, -1.8680e-01,
    -1.3990e-02, +0.0000e+00, -1.0430e-01, +1.1750e-02, -2.2400e-03,
    +0.0000e+00, -3.2220e-02, +1.3330e-02, -2.6470e-03, -2.3160e-05,
    +0.0000e+00, +5.3390e-02, +1.1070e-02, -3.1160e-03, -1.0790e-04,
    -1.2990e-05, +0.0000e+00, +4.8610e-03, +8.8910e-03, -6.4480e-04,
    -1.2790e-05, +6.3580e-06, -1.4170e-07, +0.0000e+00, +3.0410e-02,
    +1.1500e-03, -8.7430e-04, -2.7810e-05, +6.3670e-07, -1.1400e-08,
    -4.2000e-08, +0.0000e+00, -2.9820e-02, -3.0000e-03, +1.3940e-05,
    -3.2900e-05, -1.7050e-07, +7.4400e-08, +2.7200e-08, -6.6000e-09,
    +0.0000e+00, +1.2360e-02, -9.9810e-04, -3.7920e-05, -1.3550e-05,
    +1.1620e-06, -1.7890e-07, +1.4700e-08, -2.4000e-09, -4.0000e-10,
))
_AW_MEAN = np.array((
    +5.6400e+01, +1.5550e+00, -1.0110e+00, -3.9750e+00, +3.1710e-02,
    +1.0650e-01, +6.1750e-01, +1.3760e-01, +4.2290e-02, +3.0280e-03,
    +1.6880e+00, -1.6920e-01, +5.4780e-02, +2.4730e-02, +6.0590e-04,
    +2.2780e+00, +6.6140e-03, -3.5050e-04, -6.6970e-03, +8.4020e-04,
    +7.0330e-04, -3.2360e+00, +2.1840e-01, -4.6110e-02, -1.6130e-02,
    -1.6040e-03, +5.4200e-05, +7.9220e-05, -2.7110e-01, -4.4060e-01,
    -3.3760e-02, -2.8010e-03, -4.0900e-04, -2.0560e-05, +6.8940e-06,
    +2.3170e-06, +1.9410e+00, -2.5620e-01, +1.5980e-02, +5.4490e-03,
    +3.5440e-04, +1.1480e-05, +7.5030e-06, -5.6670e-07, -3.6600e-08,
    +8.6830e-01, -5.9310e-02, -1.8640e-03, -1.2770e-04, +2.0290e-04,
    +1.2690e-05, +1.6290e-06, +9.6600e-08, -1.0150e-07, -5.0000e-10,
))
_BW_MEAN = np.array((
    +0.0000e+00, +0.0000e+00, +2.5920e-01, +0.0000e+00, +2.9740e-02,
    -5.4710e-01, +0.0000e+00, -5.9260e-01, -1.0300e-01, -1.5670e-02,
    +0.0000e+00, +1.7100e-01, +9.0250e-02, +2.6890e-02, +2.2430e-03,
    +0.0000e+00, +3.4390e-01, +2.4020e-02, +5.4100e-03, +1.6010e-03,
    +9.6690e-05, +0.0000e+00, +9.5020e-02, -3.0630e-02, -1.0550e-03,
    -1.0670e-04, -1.1300e-04, +2.1240e-05, +0.0000e+00, -3.1290e-01,
    +8.4630e-03, +2.2530e-04, +7.4130e-05, -9.3760e-05, -1.6060e-06,
    +2.0600e-06, +0.0000e+00, +2.7390e-01, +1.1670e-03, -2.2460e-05,
    -1.2870e-04, -2.4380e-05, -7.5610e-07, +1.1580e-06, +4.9500e-08,
    +0.0000e+00, -1.3440e-01, +5.3420e-03, +3.7750e-04, -6.7560e-05,
    -1.6860e-06, -1.1840e-06, +2.7680e-07, +2.7300e-08, +5.7000e-09,
))
_AW_AMP = np.array((
    +1.0230e-01, -2.6950e+00, +3.4170e-01, -1.4050e-01, +3.1750e-01,
    +2.1160e-01, +3.5360e+00, -1.5050e-01, -1.6600e-02, +2.9670e-02,
    +3.8190e-01, -1.6950e-01, -7.4440e-02, +7.4090e-03, -6.2620e-03,
    -1.8360e+00, -1.7590e-02, -6.2560e-02, -2.3710e-03, +7.9470e-04,
    +1.5010e-04, -8.6030e-01, -1.3600e-01, -3.6290e-02, -3.7060e-03,
    -2.9760e-04, +1.8570e-05, +3.0210e-05, +2.2480e+00, -1.1780e-01,
    +1.2550e-02, +1.1340e-03, -2.1610e-04, -5.8170e-06, +8.8360e-07,
    -1.7690e-07, +7.3130e-01, -1.1880e-01, +1.1450e-02, +1.0110e-03,
    +1.0830e-04, +2.5700e-06, -2.1400e-06, -5.7100e-08, +2.0000e-08,
    -1.6320e+00, -6.9480e-03, -3.8930e-03, +8.5920e-04, +7.5770e-05,
    +4.5390e-06, -3.8520e-07, -2.2130e-07, -1.3700e-08, +5.8000e-09,
))
_BW_AMP = np.array((
    +0.0000e+00, +0.0000e+00, -8.8650e-02, +0.0000e+00, -4.3090e-01,
    +6.3400e-02, +0.0000e+00, +1.1620e-01, +6.1760e-02, -4.2340e-03,
    +0.0000e+00, +2.5300e-01, +4.0170e-02, -6.2040e-03, +4.9770e-03,
    +0.0000e+00, -1.7370e-01, -5.6380e-03, +1.4880e-04, +4.8570e-04,
    -1.8090e-04, +0.0000e+00, -1.5140e-01, -1.6850e-02, +5.3330e-03,
    -7.6110e-05, +2.3940e-05, +8.1950e-06, +0.0000e+00, +9.3260e-02,
    -1.2750e-02, -3.0710e-04, +5.3740e-05, -3.3910e-05, -7.4360e-06,
    +6.7470e-07, +0.0000e+00, -8.6370e-02, -3.8070e-03, -6.8330e-04,
    -3.8610e-05, -2.2680e-05, +1.4540e-06, +3.8600e-07, -1.0680e-07,
    +0.0000e+00, -2.6580e-02, -1.9470e-03, +7.1310e-04, -3.5060e-05,
    +1.8850e-07, +5.7920e-07, +3.9900e-08, +2.0000e-08, -5.7000e-09,
))
# fmt: on
_SCALE = 1e-5

# b and c of the continued fractions. The hydrostatic c grows from the equator to the
# poles and swings with the season, in antiphase on the southern hemisphere.
_B_HYDROSTATIC = 0.0029
_C0_HYDROSTATIC = 0.062
_C10_HYDROSTATIC = {"north": 0.001, "south": 0.002}
_C11_HYDROSTATIC = {"north": 0.005, "south": 0.007}
_B_WET = 0.00146
_C_WET = 0.04391
# Height correction of the hydrostatic function (Niell 1996), per kilometre of height.
_A_HEIGHT, _B_HEIGHT, _C_HEIGHT = 2.53e-5, 5.49e-3, 1.14e-3


def _legendre(t: float) -> list[list[float]]:
    """The unnormalised associated Legendre functions P_nm(t) up to degree and order
    `_DEGREE`, as a table [n][m], from the standard recurrences in n and m (Heiskanen
    and Moritz 1967, section 1-11), without the Condon-Shortley phase."""
    u = math.sqrt(max(0.0, 1.0 - t * t))
    p = [[0.0] * (_DEGREE + 1) for _ in range(_DEGREE + 1)]
    p[0][0] = 1.0
    for m in range(1, _DEGREE + 1):
        p[m][m] = (2 * m - 1) * u * p[m - 1][m - 1]
    for m in range(_DEGREE):
        p[m + 1][m] = (2 * m + 1) * t * p[m][m]
    for m in range(_DEGREE + 1):
        for n in range(m + 2, _DEGREE + 1):
            p[n][m] = ((2 * n - 1) * t * p[n - 1][m] - (n + m - 1) * p[n - 2][m]) / (n - m)
    return p


def _harmonics(latitude: float, longitude: float) -> tuple[np.ndarray, np.ndarray]:
    """The 55 surface spherical harmonics P_nm(sin lat) cos(m lon) and
    P_nm(sin lat) sin(m lon), in the order of the coefficient tables."""
    p = _legendre(math.sin(latitude))
    terms = [(n, m) for n in range(_DEGREE + 1) for m in range(n + 1)]
    cosines = np.array([p[n][m] * math.cos(m * longitude) for n, m in terms])
    sines = np.array([p[n][m] * math.sin(m * longitude) for n, m in terms])
    return cosines, sines


def _coefficient_a(
    cosines: np.ndarray,
    sines: np.ndarray,
    season: float,
    mean: tuple[np.ndarray, np.ndarray],
    amplitude: tuple[np.ndarray, np.ndarray],
) -> float:
    """a at the place of the harmonics and the time of `season` (the cosine of the
    annual phase): the mean plus the annual amplitude times the season."""
    a_mean = float(mean[0] @ cosines + mean[1] @ sines)
    a_amplitude = float(amplitude[0] @ cosines + amplitude[1] @ sines)
    return (a_mean + a_amplitude * season) * _SCALE


def _marini(sine: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """The continued fraction in sin E, normalised to 1 at the zenith."""
    zenith = 1.0 + a / (1.0 + b / (1.0 + c))
    return zenith / (sine + a / (sine + b / (sine + c)))


def global_mapping_function(
    mjd: float, latitude: float, longitude: float, height: float, elevation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Hydrostatic and wet mapping function.

    Args:
        mjd: Modified Julian Date.
        latitude, longitude: of the station, degrees (ellipsoidal latitude).
        height: of the station above sea level, m.
        elevation: of the satellites, degrees.
    """
    lat, lon = math.radians(latitude), math.radians(longitude)
    cosines, sines = _harmonics(lat, lon)
    phase = 2.0 * math.pi * (mjd - _REFERENCE_MJD) / _DAYS_PER_YEAR
    season = math.cos(phase)
    sine = np.sin(np.radians(np.asarray(elevation, dtype=float)))

    hemisphere = "south" if lat < 0 else "north"
    # On the southern hemisphere the seasons are half a year apart.
    season_c = math.cos(phase + (math.pi if hemisphere == "south" else 0.0))
    c_hydrostatic = _C0_HYDROSTATIC + (
        (season_c + 1.0) * _C11_HYDROSTATIC[hemisphere] / 2.0 + _C10_HYDROSTATIC[hemisphere]
    ) * (1.0 - math.cos(lat))
    a_hydrostatic = _coefficient_a(cosines, sines, season, (_AH_MEAN, _BH_MEAN), (_AH_AMP, _BH_AMP))
    hydrostatic = _marini(sine, a_hydrostatic, _B_HYDROSTATIC, c_hydrostatic)
    # Niell's height correction: the difference between 1 / sin E and a mapping function
    # of fixed coefficients, per kilometre of height.
    per_km = 1.0 / sine - _marini(sine, _A_HEIGHT, _B_HEIGHT, _C_HEIGHT)
    hydrostatic = hydrostatic + per_km * height / 1000.0

    a_wet = _coefficient_a(cosines, sines, season, (_AW_MEAN, _BW_MEAN), (_AW_AMP, _BW_AMP))
    wet = _marini(sine, a_wet, _B_WET, _C_WET)
    return hydrostatic, wet
