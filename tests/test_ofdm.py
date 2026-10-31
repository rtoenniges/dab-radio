import numpy as np
import pytest

from dab.constants import MODE_I, TransmissionMode
from dsp.ofdm import (
    FrameDemodulator,
    correct_frequency_offset,
    demodulate_frame_symbols,
    demodulate_symbol,
    differential_demodulate,
    estimate_frequency_offset,
    extract_used_carriers,
)
from dsp.sync import find_null_symbol
from dsp.timing import generate_reference_symbol, physical_carrier_indices
from tests.dab_signal import build_stream, fib, noise, paused_carrier

SAMPLE_RATE = 2_048_000.0
QPSK_ALPHABET = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)


def _make_ofdm_symbol(
    carriers: np.ndarray, fft_length: int, guard_length: int, num_carriers: int
) -> np.ndarray:
    """Builds a synthetic OFDM symbol (including the cyclic prefix) from carrier values."""
    center = fft_length // 2
    half = num_carriers // 2
    spectrum_shifted = np.zeros(fft_length, dtype=complex)
    spectrum_shifted[center - half : center] = carriers[:half]
    spectrum_shifted[center + 1 : center + 1 + half] = carriers[half:]
    useful = np.fft.ifft(np.fft.ifftshift(spectrum_shifted))
    return np.concatenate([useful[-guard_length:], useful])


def _make_test_mode() -> TransmissionMode:
    return TransmissionMode(
        name="test",
        fft_length=64,
        guard_length=16,
        num_carriers=48,
        symbols_per_frame=5,
        null_symbol_length=80,
    )


def _build_synthetic_frame(
    mode: TransmissionMode,
    num_data_symbols: int,
    rng: np.random.Generator,
    timing_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds null symbol + PRS + differentially encoded data symbols.

    `timing_offset` delays the content of every OFDM symbol by that many samples, a
    fraction of a sample too: each carrier k turns by -2πkτ/N, the same in every
    symbol, exactly what an FFT window that far off sees.

    Returns:
        (samples, expected_constellation)
    """
    ramp = np.exp(
        -2j * np.pi * physical_carrier_indices(mode.num_carriers) * timing_offset / mode.fft_length
    )
    # The real PRS pattern (ETSI 14.3.2), so that the PRS cross-correlation built
    # into demodulate_frame_symbols (dsp.timing) finds a genuine match.
    prs_carriers = generate_reference_symbol(mode.num_carriers)
    symbols = [
        _make_ofdm_symbol(
            prs_carriers * ramp, mode.fft_length, mode.guard_length, mode.num_carriers
        )
    ]

    expected_symbols = []
    reference_carriers = prs_carriers
    for _ in range(num_data_symbols):
        data = rng.choice(QPSK_ALPHABET, size=mode.num_carriers)
        expected_symbols.append(data)
        # current * conj(reference) should give `data` (reference has magnitude 1).
        current_carriers = data * reference_carriers
        symbols.append(
            _make_ofdm_symbol(
                current_carriers * ramp, mode.fft_length, mode.guard_length, mode.num_carriers
            )
        )
        reference_carriers = current_carriers

    frame_samples = np.concatenate(symbols)
    null_samples = (
        0.001
        * (
            rng.standard_normal(mode.null_symbol_length)
            + 1j * rng.standard_normal(mode.null_symbol_length)
        )
    ).astype(complex)
    samples = np.concatenate([null_samples, frame_samples])

    return samples, np.concatenate(expected_symbols)


def _constellation(samples: np.ndarray, mode: TransmissionMode, num_data_symbols: int):
    """All constellation points of a frame as one array, or None."""
    frame = demodulate_frame_symbols(
        samples, null_start=0, mode=mode, sample_rate=SAMPLE_RATE, max_data_symbols=num_data_symbols
    )
    return None if frame is None else np.concatenate(frame.symbols)


def test_extract_used_carriers_slices_symmetrically_around_dc():
    spectrum = np.arange(8, dtype=complex)  # centre index = 4 (DC)
    result = extract_used_carriers(spectrum, num_carriers=4)
    # lower 2 carriers: indices 2,3; upper 2 carriers: indices 5,6 (DC=4 excluded)
    np.testing.assert_array_equal(result, [2, 3, 5, 6])


def test_demodulate_symbol_recovers_known_spectrum():
    fft_length, guard_length = 16, 4
    spectrum = np.zeros(fft_length, dtype=complex)
    spectrum[3] = 2.0 + 1.0j
    useful = np.fft.ifft(np.fft.ifftshift(spectrum))
    symbol_samples = np.concatenate([useful[-guard_length:], useful])

    result = demodulate_symbol(symbol_samples, fft_length, guard_length)

    np.testing.assert_allclose(result, spectrum, atol=1e-10)


def test_demodulate_symbol_transforms_each_row_on_its_own():
    """Several symbols at once: one FFT per row, identical to the single call."""
    fft_length, guard_length = 16, 4
    rng = np.random.default_rng(0)
    rows = rng.standard_normal((3, fft_length + guard_length)) + 1j * rng.standard_normal(
        (3, fft_length + guard_length)
    )

    result = demodulate_symbol(rows, fft_length, guard_length)

    assert result.shape == (3, fft_length)
    for row, spectrum in zip(rows, result, strict=True):
        np.testing.assert_allclose(spectrum, demodulate_symbol(row, fft_length, guard_length))


def test_demodulate_frame_symbols_returns_the_raw_fft_of_every_symbol():
    """The FFT output goes to the display unchanged: one row per symbol from the PRS
    onwards, all bins. The PRS sits on the used carriers; DC and the edge bins stay
    empty."""
    mode = _make_test_mode()
    rng = np.random.default_rng(3)
    samples, _ = _build_synthetic_frame(mode, num_data_symbols=3, rng=rng)

    frame = demodulate_frame_symbols(samples, 0, mode, SAMPLE_RATE, max_data_symbols=3)

    assert frame is not None
    spectra = frame.spectra
    assert spectra.shape == (4, mode.fft_length)
    np.testing.assert_allclose(
        extract_used_carriers(spectra[0], mode.num_carriers),
        generate_reference_symbol(mode.num_carriers),
        atol=1e-9,
    )
    used = np.zeros(mode.fft_length, dtype=bool)
    center, half = mode.fft_length // 2, mode.num_carriers // 2
    used[center - half : center + half + 1] = True
    used[center] = False
    np.testing.assert_allclose(spectra[:, ~used], 0, atol=1e-9)


def test_differential_demodulate_recovers_original_data():
    reference = QPSK_ALPHABET
    data = QPSK_ALPHABET[::-1]
    current = data * reference  # |reference| == 1

    result = differential_demodulate(current, reference)

    np.testing.assert_allclose(result, data, atol=1e-10)


def test_estimate_frequency_offset_recovers_known_offset():
    mode = _make_test_mode()
    rng = np.random.default_rng(1)
    samples, _ = _build_synthetic_frame(mode, num_data_symbols=3, rng=rng)
    frame_samples = samples[mode.null_symbol_length :]

    true_offset_hz = 5000.0
    n = np.arange(len(frame_samples))
    rotated = frame_samples * np.exp(2j * np.pi * true_offset_hz * n / SAMPLE_RATE)

    estimated = estimate_frequency_offset(
        rotated, mode.fft_length, mode.guard_length, mode.symbol_length, SAMPLE_RATE
    )

    assert abs(estimated - true_offset_hz) < 50.0


def test_correct_frequency_offset_undoes_rotation():
    n = np.arange(100)
    samples = np.ones(100, dtype=complex)
    offset_hz = 1000.0
    rotated = samples * np.exp(2j * np.pi * offset_hz * n / SAMPLE_RATE)

    corrected = correct_frequency_offset(rotated, offset_hz, SAMPLE_RATE)

    np.testing.assert_allclose(corrected, samples, atol=1e-10)


def test_demodulate_frame_symbols_recovers_dqpsk_symbols():
    mode = _make_test_mode()
    rng = np.random.default_rng(42)
    num_data_symbols = 3
    samples, expected = _build_synthetic_frame(mode, num_data_symbols, rng)

    constellation = _constellation(samples, mode, num_data_symbols)

    assert constellation is not None
    assert len(constellation) == num_data_symbols * mode.num_carriers
    np.testing.assert_allclose(constellation, expected, atol=1e-6)


def test_demodulate_frame_symbols_corrects_frequency_offset():
    mode = _make_test_mode()
    rng = np.random.default_rng(7)
    num_data_symbols = 3
    samples, expected = _build_synthetic_frame(mode, num_data_symbols, rng)

    n = np.arange(len(samples))
    rotated = samples * np.exp(2j * np.pi * 5000.0 * n / SAMPLE_RATE)

    constellation = _constellation(rotated, mode, num_data_symbols)

    assert constellation is not None
    np.testing.assert_allclose(constellation, expected, atol=1e-3)


@pytest.mark.parametrize("timing_offset", [0.4, -0.3])
def test_fractional_timing_offset_cancels_in_differential_demodulation(timing_offset):
    """Why there is no fine timing correction: the phase ramp across the carriers is the
    same in every symbol and drops out of z(l) · z*(l-1). Uncancelled it would turn the
    outermost carriers here by up to 54°."""
    mode = _make_test_mode()
    rng = np.random.default_rng(11)
    num_data_symbols = 3
    samples, expected = _build_synthetic_frame(mode, num_data_symbols, rng, timing_offset)

    constellation = _constellation(samples, mode, num_data_symbols)

    assert constellation is not None
    np.testing.assert_allclose(constellation, expected, atol=1e-6)


def test_weak_but_aligned_frame_counts_as_reliable():
    """`reliable` is about synchronisation, not signal quality. At 6 dB SNR the D-QPSK
    points scatter widely (a z^4 clustering check would reject such frames and restart
    the MSC chain every second), yet the frame is aligned and the error correction can
    decode it."""
    rng = np.random.default_rng(3)
    stream = build_stream([fib()] * 12, 2, rng, snr_db=6.0)

    frame = demodulate_frame_symbols(stream, 0, MODE_I, SAMPLE_RATE, max_data_symbols=75)

    assert frame.reliable


def test_frame_without_dab_signal_is_unreliable():
    rng = np.random.default_rng(4)

    frame = demodulate_frame_symbols(
        noise(3 * MODE_I.frame_length, rng), 0, MODE_I, SAMPLE_RATE, max_data_symbols=75
    )

    assert not frame.reliable


def test_acquire_takes_the_null_symbol_behind_which_the_prs_lies():
    """The pauses of a carrier in the band, twice as strong as the ensemble, make the
    deepest power minimum; the check for the PRS passes over them."""
    null_start = MODE_I.frame_length - 50_000
    pause = null_start - 23_900
    stream = build_stream([fib()] * 12, 3, np.random.default_rng(12), start_offset=50_000)
    stream = stream + paused_carrier(len(stream), 2 * np.mean(np.abs(stream) ** 2), pause)
    assert abs(find_null_symbol(stream, MODE_I, MODE_I.frame_length) - pause) < 50

    null_index, frame = FrameDemodulator(MODE_I, SAMPLE_RATE).acquire(
        stream, MODE_I.frame_length, 16
    )

    assert frame.reliable
    assert abs(null_index + frame.sample_correction - null_start) <= 1


def test_acquire_without_a_prs_takes_the_deepest_minimum():
    quiet = 70_000
    samples = noise(2 * MODE_I.frame_length, np.random.default_rng(13))
    samples[quiet : quiet + MODE_I.null_symbol_length] *= 0.1

    null_index, frame = FrameDemodulator(MODE_I, SAMPLE_RATE).acquire(
        samples, MODE_I.frame_length, 16
    )

    assert abs(null_index - quiet) <= 2
    assert not frame.reliable


def test_demodulate_frame_symbols_returns_none_when_no_data_symbol_fits():
    mode = _make_test_mode()
    samples = np.zeros(mode.null_symbol_length + mode.symbol_length, dtype=complex)

    assert _constellation(samples, mode, num_data_symbols=16) is None


def test_estimate_frequency_offset_raises_when_no_full_symbol():
    mode = _make_test_mode()
    samples = np.zeros(mode.symbol_length - 1, dtype=complex)

    with pytest.raises(ValueError, match="Too few samples"):
        estimate_frequency_offset(
            samples, mode.fft_length, mode.guard_length, mode.symbol_length, SAMPLE_RATE
        )
