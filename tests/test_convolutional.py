import numpy as np

from dsp.convolutional import (
    NUM_OUTPUTS,
    TAIL_BITS,
    depuncture,
    encode_mother_code,
    puncture,
    soft_bits,
    viterbi_decode,
)


def test_encode_mother_code_output_length():
    bits = np.zeros(20, dtype=np.uint8)
    encoded = encode_mother_code(bits)
    assert len(encoded) == NUM_OUTPUTS * (len(bits) + TAIL_BITS)


def test_viterbi_decode_recovers_bits_without_puncturing_or_noise():
    rng = np.random.default_rng(1)
    info_bits = rng.integers(0, 2, size=100).astype(np.uint8)

    mother = encode_mother_code(info_bits)

    decoded = viterbi_decode(soft_bits(mother), num_info_bits=len(info_bits))

    np.testing.assert_array_equal(decoded, info_bits)


def test_viterbi_decode_corrects_single_bit_error():
    rng = np.random.default_rng(2)
    info_bits = rng.integers(0, 2, size=50).astype(np.uint8)

    mother = encode_mother_code(info_bits).copy()
    mother[10] ^= 1  # a single bit error in the codeword

    decoded = viterbi_decode(soft_bits(mother), num_info_bits=len(info_bits))

    np.testing.assert_array_equal(decoded, info_bits)


def test_puncture_and_depuncture_roundtrip_with_viterbi():
    rng = np.random.default_rng(3)
    info_bits = rng.integers(0, 2, size=40).astype(np.uint8)
    pattern = np.array([1, 1, 0, 1], dtype=np.uint8)  # a rate-3/4-like pattern

    mother = encode_mother_code(info_bits)
    punctured = puncture(soft_bits(mother), pattern)
    received = depuncture(punctured, pattern)

    decoded = viterbi_decode(received, num_info_bits=len(info_bits))

    np.testing.assert_array_equal(decoded, info_bits)


def test_depuncture_fills_punctured_positions_with_zero():
    """A soft bit of 0 carries no information, which is exactly what a punctured
    position is."""
    pattern = np.array([1, 0, 1, 1], dtype=np.uint8)

    received = depuncture(np.array([0.5, -2.0, 1.0, -1.0, 3.0, 0.25]), pattern)

    np.testing.assert_array_equal(received, [0.5, 0, -2.0, 1.0, -1.0, 0, 3.0, 0.25])


def test_viterbi_decode_trusts_confident_soft_bits_over_weak_noisy_ones():
    """What soft decisions are for: bits from weak carriers (in a notch of the
    channel, drowned in noise) count for little, so the confident bits around them
    decide.

    40 % of the coded bits are replaced by weak noise of random sign, scattered as the
    frequency and time interleaving scatter a bad carrier. As soft bits they are
    almost erasures, which the rate-1/4 code easily bridges; as hard decisions half of
    them are errors of full weight (20 % bit errors), too many for the code.
    """
    rng = np.random.default_rng(0)
    info_bits = rng.integers(0, 2, size=200).astype(np.uint8)
    received = soft_bits(encode_mother_code(info_bits))
    weak = rng.random(len(received)) < 0.4
    received[weak] = 0.1 * rng.choice([-1.0, 1.0], weak.sum())

    soft_decoded = viterbi_decode(received, num_info_bits=len(info_bits))
    hard_decoded = viterbi_decode(np.sign(received), num_info_bits=len(info_bits))

    np.testing.assert_array_equal(soft_decoded, info_bits)
    assert not np.array_equal(hard_decoded, info_bits)


def test_viterbi_decode_ignores_the_scale_of_the_soft_bits():
    rng = np.random.default_rng(5)
    info_bits = rng.integers(0, 2, size=60).astype(np.uint8)
    received = soft_bits(encode_mother_code(info_bits)) * rng.uniform(0.1, 2.0, 264)
    received[::7] *= -0.3  # some weak wrong bits

    np.testing.assert_array_equal(
        viterbi_decode(received * 1e4, len(info_bits)),
        viterbi_decode(received, len(info_bits)),
    )


def test_puncture_removes_marked_positions():
    mother_bits = np.arange(8, dtype=np.uint8) % 2
    pattern = np.array([1, 0, 1, 1], dtype=np.uint8)

    result = puncture(mother_bits, pattern)

    expected_mask = np.tile(pattern.astype(bool), 2)
    np.testing.assert_array_equal(result, mother_bits[expected_mask])
