"""The Global Mapping Function (`dab.gmf`): the published check case, the Legendre
functions of the rewritten expansion, and the shape of the function."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dab.gmf import _legendre, global_mapping_function


def test_published_check_case() -> None:
    """The check case published with the routine (the header of GMF.F in the IERS
    Conventions software collection): MJD 55055, NRAO Green Bank, zenith distance
    1.278564131 rad. With the coefficients of Böhm's gmf.f both functions agree to the
    last digit; the IERS copy, whose first coefficient reads 125.2 instead of 125.17,
    put the hydrostatic one 1.1e-5 low."""
    hydrostatic, wet = global_mapping_function(
        55055.0,
        math.degrees(0.6708665767),
        math.degrees(-1.393397187),
        844.715,
        90.0 - math.degrees(1.278564131),
    )
    assert hydrostatic == pytest.approx(3.425245519339138678, abs=1e-12)
    assert wet == pytest.approx(3.449589116182419554, abs=1e-12)


@pytest.mark.parametrize("t", [-0.9, -0.3, 0.0, 0.4, 0.75, 1.0])
def test_legendre_functions_match_their_closed_forms(t: float) -> None:
    """The recurrences against the closed forms of the first degrees (unnormalised,
    without the Condon-Shortley phase)."""
    u = math.sqrt(1.0 - t * t)
    p = _legendre(t)
    assert p[0][0] == 1.0
    assert p[1][0] == pytest.approx(t)
    assert p[1][1] == pytest.approx(u)
    assert p[2][0] == pytest.approx((3 * t * t - 1) / 2)
    assert p[2][1] == pytest.approx(3 * t * u)
    assert p[2][2] == pytest.approx(3 * u * u)
    assert p[3][0] == pytest.approx((5 * t**3 - 3 * t) / 2)
    assert p[3][1] == pytest.approx(1.5 * (5 * t * t - 1) * u)
    assert p[3][3] == pytest.approx(15 * u**3)
    assert p[9][9] == pytest.approx(math.prod(range(1, 18, 2)) * u**9)


def test_zenith_is_one_and_low_elevations_grow() -> None:
    elevations = np.array([90.0, 30.0, 10.0, 5.0])
    hydrostatic, wet = global_mapping_function(61312.0, 52.4, 9.7, 60.0, elevations)
    assert hydrostatic[0] == pytest.approx(1.0, abs=1e-3)
    assert wet[0] == pytest.approx(1.0, abs=1e-3)
    assert np.all(np.diff(hydrostatic) > 0)
    # At 30 degrees close to 1 / sin(E).
    assert hydrostatic[1] == pytest.approx(2.0, rel=0.01)
