import numpy as np
import pytest

from dab.constants import MODE_I, SAMPLE_RATE
from dsp.ofdm import demodulate_frame_symbols
from dsp.tii import (
    PATTERNS,
    TiiDetector,
    detect_transmitters,
    null_symbol_carrier_power,
)
from dsp.timing import generate_reference_symbol, physical_carrier_indices

K = MODE_I.num_carriers
N = MODE_I.fft_length
CARRIERS = physical_carrier_indices(K)
PRS = generate_reference_symbol(K)


def _tii_carriers(main_id: int, sub_id: int) -> np.ndarray:
    """Carriers k whose pair k, k+1 the transmitter occupies, per ETSI EN 300 401,
    clause 14.8 (as in ODR-DabMod, src/TII.cpp)."""
    return np.array(
        [
            start + 2 * sub_id + 48 * block
            for start in (-768, -384, 1, 385)
            for block in range(8)
            if PATTERNS[main_id][block]
        ]
    )


def _spectrum(transmitters: list[tuple[int, int, float]]) -> np.ndarray:
    """Null symbol per carrier (physical order): both carriers of a pair with the
    phase of the reference symbol at the first one (z_k from clause 14.8)."""
    spectrum = np.zeros(K, dtype=complex)
    position = {int(k): i for i, k in enumerate(CARRIERS)}
    for main_id, sub_id, amplitude in transmitters:
        for k in _tii_carriers(main_id, sub_id):
            phase = PRS[position[k]]
            spectrum[position[k]] += amplitude * phase
            spectrum[position[k + 1]] += amplitude * phase
    return spectrum


def _to_time(carriers: np.ndarray, guard: int) -> np.ndarray:
    grid = np.zeros(N, dtype=complex)
    grid[CARRIERS % N] = carriers
    useful = np.fft.ifft(grid)
    return np.concatenate([useful[-guard:], useful])


def _frame(transmitters, rng, noise=0.01, data_symbols=4) -> np.ndarray:
    """Null symbol with TII, phase reference and D-QPSK data symbols."""
    null = _to_time(_spectrum(transmitters), MODE_I.null_symbol_length - N)
    parts = [null, _to_time(PRS, MODE_I.guard_length)]
    carriers = PRS
    for _ in range(data_symbols):
        carriers = carriers * np.exp(1j * np.pi / 4 * rng.choice([1, 3, 5, 7], K))
        parts.append(_to_time(carriers, MODE_I.guard_length))
    signal = np.concatenate(parts)
    level = noise / np.sqrt(N)
    return signal + level * (
        rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
    )


def _null_power(transmitters, rng, noise=0.01) -> np.ndarray:
    power = null_symbol_carrier_power(
        _frame(transmitters, rng, noise), MODE_I.null_symbol_length, 0.0, SAMPLE_RATE, MODE_I
    )
    assert power is not None
    return power


def test_pattern_table_matches_the_reference():
    """Spot checks against the table in ODR-DabMod (src/TII.cpp)."""
    assert len(PATTERNS) == len(set(PATTERNS)) == 70
    assert all(sum(pattern) == 4 for pattern in PATTERNS)
    assert PATTERNS[0] == (0, 0, 0, 0, 1, 1, 1, 1)
    assert PATTERNS[1] == (0, 0, 0, 1, 0, 1, 1, 1)
    assert PATTERNS[35] == (1, 0, 0, 0, 0, 1, 1, 1)
    assert PATTERNS[69] == (1, 1, 1, 1, 0, 0, 0, 0)


def test_transmitter_occupies_16_pairs_within_the_band():
    carriers = _tii_carriers(69, 23)
    assert len(carriers) == 16
    assert carriers.min() >= -768 and carriers.max() + 1 <= 768
    assert 0 not in carriers and -1 not in carriers  # no pair across the centre


def test_single_transmitter_is_identified():
    rng = np.random.default_rng(1)
    result = detect_transmitters(_null_power([(19, 4, 1.0)], rng), frames=1)

    assert [(t.main_id, t.sub_id) for t in result.transmitters] == [(19, 4)]
    assert result.transmitters[0].level_db == 0.0
    assert result.transmitters[0].share == 1.0


def test_two_transmitters_with_their_levels_and_shares():
    """The second transmitter at half amplitude lies 6 dB lower and has a fifth of
    the power of both transmitters."""
    rng = np.random.default_rng(2)
    detector = TiiDetector()
    for _ in range(8):
        result = detector.add(_null_power([(19, 4, 1.0), (19, 11, 0.5)], rng))

    ids = [(t.main_id, t.sub_id) for t in result.transmitters]
    assert ids == [(19, 4), (19, 11)]
    assert result.transmitters[1].level_db == pytest.approx(-6.0, abs=0.5)
    assert result.transmitters[1].share == pytest.approx(0.2, abs=0.02)
    assert result.detection_limit_db < -6.0


def test_tii_in_every_second_null_symbol_is_found():
    """That is how ODR-DabMod and the transmitters on 5C, 5D and 8C send it."""
    rng = np.random.default_rng(3)
    detector = TiiDetector()
    for frame in range(32):
        transmitters = [(51, 4, 1.0)] if frame % 2 else []
        result = detector.add(_null_power(transmitters, rng, noise=0.1))

    assert [(t.main_id, t.sub_id) for t in result.transmitters] == [(51, 4)]
    assert result.frames == 32


def test_noise_alone_yields_no_transmitter():
    rng = np.random.default_rng(4)
    detector = TiiDetector()
    for _ in range(32):
        result = detector.add(_null_power([], rng))

    assert result.transmitters == []
    assert result.detection_limit_db is None


def test_detector_averages_only_the_latest_null_symbols():
    rng = np.random.default_rng(5)
    detector = TiiDetector(averaged_frames=4)
    for _ in range(4):
        detector.add(_null_power([(19, 4, 1.0)], rng))
    for _ in range(4):
        result = detector.add(_null_power([(35, 13, 1.0)], rng))

    assert [(t.main_id, t.sub_id) for t in result.transmitters] == [(35, 13)]


def test_null_symbol_is_found_through_the_real_demodulation():
    """With a frequency offset of 3.7 kHz (three whole carrier spacings plus a
    fraction): frame position and frequency correction come from
    `demodulate_frame_symbols`, as in the worker."""
    rng = np.random.default_rng(6)
    signal = _frame([(35, 13, 1.0)], rng)
    signal = signal * np.exp(2j * np.pi * 3700.0 * np.arange(len(signal)) / SAMPLE_RATE)

    frame = demodulate_frame_symbols(signal, 0, MODE_I, SAMPLE_RATE, max_data_symbols=3)
    assert frame is not None
    frame_start = MODE_I.null_symbol_length + frame.sample_correction
    power = null_symbol_carrier_power(
        signal, frame_start, frame.frequency_offset_hz, SAMPLE_RATE, MODE_I
    )

    assert power is not None
    found = detect_transmitters(power, frames=1).transmitters
    assert [(t.main_id, t.sub_id) for t in found] == [(35, 13)]


def test_null_symbol_before_the_buffer_is_skipped():
    rng = np.random.default_rng(7)
    signal = _frame([(19, 4, 1.0)], rng)
    assert null_symbol_carrier_power(signal, 100, 0.0, SAMPLE_RATE, MODE_I) is None
