import numpy as np

from dsp.scrambling import apply_energy_dispersal, generate_prbs

# ETSI EN 300 401, table 12: the first 16 bits of the PRBS.
EXPECTED_PRBS_FIRST_16 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]


def test_generate_prbs_matches_spec_table_12():
    prbs = generate_prbs(16)
    np.testing.assert_array_equal(prbs, EXPECTED_PRBS_FIRST_16)


def test_apply_energy_dispersal_is_involutory():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=768).astype(np.uint8)

    scrambled = apply_energy_dispersal(bits)
    descrambled = apply_energy_dispersal(scrambled)

    np.testing.assert_array_equal(descrambled, bits)
    assert not np.array_equal(scrambled, bits)
