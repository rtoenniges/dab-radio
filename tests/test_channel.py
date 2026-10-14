import numpy as np
import pytest

from dab.constants import MODE_I, SAMPLE_RATE
from dsp.channel import impulse_response, sfn_shares
from dsp.ofdm import demodulate_frame_symbols
from dsp.timing import generate_reference_symbol, physical_carrier_indices

CARRIERS = physical_carrier_indices(MODE_I.num_carriers)


def _channel(paths: list[tuple[float, int]]) -> np.ndarray:
    """Channel per subcarrier from paths (amplitude, delay in samples)."""
    return sum(
        amplitude * np.exp(-2j * np.pi * CARRIERS * delay / MODE_I.fft_length)
        for amplitude, delay in paths
    )


def _shares(channel: np.ndarray) -> tuple[float, float]:
    delays, power = impulse_response(channel, MODE_I, SAMPLE_RATE)
    return sfn_shares(delays, power, MODE_I, SAMPLE_RATE)


def test_single_path_lies_at_zero_and_has_no_sfn_share():
    delays, power = impulse_response(_channel([(1.0, 0)]), MODE_I, SAMPLE_RATE)

    assert delays[np.argmax(power)] == 0.0
    others, outside = sfn_shares(delays, power, MODE_I, SAMPLE_RATE)
    assert others < 0.005
    assert outside < 0.005


def test_echo_appears_at_its_delay_and_level():
    delay = 205  # 100.1 µs, 30 km of path difference
    delays, power = impulse_response(_channel([(1.0, 0), (0.5, delay)]), MODE_I, SAMPLE_RATE)

    echo = np.argmin(np.abs(delays - delay / SAMPLE_RATE))
    assert power[echo] / power.max() == pytest.approx(0.25, rel=0.02)


def test_echo_inside_the_guard_interval_is_harmless():
    """Power share of the echo: 0.5² / (1 + 0.5²) = 20 %."""
    others, outside = _shares(_channel([(1.0, 0), (0.5, 205)]))

    assert others == pytest.approx(0.2, abs=0.01)
    assert outside < 0.005


def test_echo_beyond_the_guard_interval_counts_as_harmful():
    others, outside = _shares(_channel([(1.0, 0), (0.5, 614)]))  # 300 µs

    assert others == pytest.approx(0.2, abs=0.01)
    assert outside == pytest.approx(0.2, abs=0.01)


def test_strongest_path_beyond_the_guard_interval_counts_as_harmful():
    """The receiver locks to the earliest path that is at least half as strong as the
    strongest. If the strongest arrives 300 µs later, it is itself a disturbance and
    must not pass as the harmless main path."""
    others, outside = _shares(_channel([(0.8, 0), (1.0, 614)]))

    assert others == pytest.approx(0.64 / 1.64, abs=0.01)
    assert outside == pytest.approx(1.0 / 1.64, abs=0.01)


def test_noise_does_not_count_as_sfn_share():
    """Noise spreads over all delays. Without subtracting it, a 10 dB SNR per
    carrier would show as roughly 10 % of apparent SFN share."""
    rng = np.random.default_rng(5)
    noise = (
        rng.standard_normal(len(CARRIERS)) + 1j * rng.standard_normal(len(CARRIERS))
    ) / np.sqrt(2 * 10)

    others, outside = _shares(_channel([(1.0, 0)]) + noise)

    assert others < 0.02
    assert outside < 0.02


def _ofdm_symbol(carriers: np.ndarray) -> np.ndarray:
    n, g, k = MODE_I.fft_length, MODE_I.guard_length, MODE_I.num_carriers
    spectrum = np.zeros(n, dtype=complex)
    spectrum[n // 2 - k // 2 : n // 2] = carriers[: k // 2]
    spectrum[n // 2 + 1 : n // 2 + 1 + k // 2] = carriers[k // 2 :]
    useful = np.fft.ifft(np.fft.ifftshift(spectrum))
    return np.concatenate([useful[-g:], useful])


def test_demodulation_reports_the_channel_at_its_own_window():
    """Through the real demodulation: the path the receiver synchronises to sits at
    0, an echo of half amplitude 200 samples behind it."""
    rng = np.random.default_rng(3)
    carriers = generate_reference_symbol(MODE_I.num_carriers)
    parts = [np.zeros(MODE_I.null_symbol_length, dtype=complex), _ofdm_symbol(carriers)]
    for _ in range(4):
        carriers = carriers * np.exp(1j * np.pi / 4 * rng.choice([1, 3, 5, 7], len(carriers)))
        parts.append(_ofdm_symbol(carriers))
    signal = np.concatenate(parts)
    signal[200:] += 0.5 * signal[:-200]

    frame = demodulate_frame_symbols(signal, 0, MODE_I, SAMPLE_RATE, max_data_symbols=3)
    assert frame is not None
    delays, power = impulse_response(frame.channel_estimate, MODE_I, SAMPLE_RATE)

    assert delays[np.argmax(power)] == 0.0
    echo = np.argmin(np.abs(delays - 200 / SAMPLE_RATE))
    assert power[echo] / power.max() == pytest.approx(0.25, rel=0.05)
