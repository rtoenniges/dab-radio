"""The scaling and density of the constellation view (`ui.signal_tab`)."""

import numpy as np

from ui.signal_tab import constellation_density, constellation_limit


def test_constellation_limit_ignores_rare_outliers():
    """A few far-out points (the tails of a noisy D-QPSK product) must not zoom the
    view out; the maximum would put the axis at 50 here."""
    rng = np.random.default_rng(1)
    points = (rng.choice([-1.0, 1.0], 1000) + 1j * rng.choice([-1.0, 1.0], 1000)) / np.sqrt(2)
    points[:5] = 50.0

    assert constellation_limit(points) == np.float64(1.0)


def test_constellation_limit_of_no_points_is_a_usable_range():
    assert constellation_limit(np.array([], dtype=complex)) == 1.0
    assert constellation_limit(np.zeros(10, dtype=complex)) == 1.0


def test_constellation_density_counts_points_per_square_in_i_q_order():
    points = np.array([0.5 + 0.5j, 0.6 + 0.4j, -0.5 + 0.5j, 0.5 - 0.5j, 3.0 + 0.0j])

    counts = constellation_density(points, extent=1.0, bins=2)

    # [I bin, Q bin]: bin 0 is negative, bin 1 positive; 3.0 lies outside.
    np.testing.assert_array_equal(counts, [[0, 1], [1, 2]])
