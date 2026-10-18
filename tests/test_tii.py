import numpy as np
import pytest

from dab.constants import MODE_I, SAMPLE_RATE
from dsp import tii
from dsp.channel import impulse_response
from dsp.ofdm import demodulate_frame_symbols
from dsp.tii import (
    PATTERNS,
    PathTracker,
    TiiDetector,
    code_of_path,
    code_positions,
    detect_transmitters,
    null_symbol_carriers,
    null_symbol_spectrum,
    tii_delay_offset,
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


def _null_carriers(transmitters, rng, noise=0.01) -> np.ndarray:
    spectrum = null_symbol_spectrum(
        _frame(transmitters, rng, noise), MODE_I.null_symbol_length, 0.0, SAMPLE_RATE, MODE_I
    )
    assert spectrum is not None
    return null_symbol_carriers(spectrum, MODE_I)


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
    result = detect_transmitters(np.abs(_null_carriers([(19, 4, 1.0)], rng)) ** 2, frames=1)

    assert [(t.main_id, t.sub_id) for t in result.transmitters] == [(19, 4)]
    assert result.transmitters[0].level_db == 0.0
    assert result.transmitters[0].share == 1.0


def test_two_transmitters_with_their_levels_and_shares():
    """The second transmitter at half amplitude lies 6 dB lower and has a fifth of
    the power of both transmitters."""
    rng = np.random.default_rng(2)
    detector = TiiDetector()
    for _ in range(8):
        result = detector.add(_null_carriers([(19, 4, 1.0), (19, 11, 0.5)], rng))

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
        result = detector.add(_null_carriers(transmitters, rng, noise=0.1))

    assert [(t.main_id, t.sub_id) for t in result.transmitters] == [(51, 4)]
    assert result.frames == 32


def test_noise_alone_yields_no_transmitter():
    rng = np.random.default_rng(4)
    detector = TiiDetector()
    for _ in range(32):
        result = detector.add(_null_carriers([], rng))

    assert result.transmitters == []
    assert result.detection_limit_db is None


def test_detector_averages_only_the_latest_null_symbols():
    rng = np.random.default_rng(5)
    detector = TiiDetector(averaged_frames=4)
    for _ in range(4):
        detector.add(_null_carriers([(19, 4, 1.0)], rng))
    for _ in range(4):
        result = detector.add(_null_carriers([(35, 13, 1.0)], rng))

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
    spectrum = null_symbol_spectrum(
        signal, frame_start, frame.frequency_offset_hz, SAMPLE_RATE, MODE_I
    )

    assert spectrum is not None
    power = np.abs(null_symbol_carriers(spectrum, MODE_I)) ** 2
    found = detect_transmitters(power, frames=1).transmitters
    assert [(t.main_id, t.sub_id) for t in found] == [(35, 13)]


def test_null_symbol_before_the_buffer_is_skipped():
    rng = np.random.default_rng(7)
    signal = _frame([(19, 4, 1.0)], rng)
    assert null_symbol_spectrum(signal, 100, 0.0, SAMPLE_RATE, MODE_I) is None


# --- Paths of the impulse response ----------------------------------------------------

OFFSET = tii_delay_offset(MODE_I)
_POSITIONS = np.arange(K)
# The phase both carriers of a pair carry: that of the first, at an even position.
_PAIR_REFERENCE = PRS[_POSITIONS - _POSITIONS % 2]


def _carriers_of_frame(paths, rng, tii=True, noise=0.003):
    """Channel estimate and null symbol of one frame, built on the carriers.

    `paths`: (delay in samples, amplitude, code or None). Every path carries the TII of
    its code with the amplitude of its signal, OFFSET samples later in the null symbol
    window (see `tii_delay_offset`); each frame has a phase of its own.
    """
    phase = np.exp(2j * np.pi * rng.random())
    channel = np.zeros(K, dtype=complex)
    null = np.zeros(K, dtype=complex)
    for delay, amplitude, code in paths:
        channel += amplitude * np.exp(-2j * np.pi * CARRIERS * delay / N)
        if tii and code is not None:
            positions = code_positions(*code)
            null[positions] += (
                amplitude
                * _PAIR_REFERENCE[positions]
                * np.exp(-2j * np.pi * CARRIERS[positions] * (delay + OFFSET) / N)
            )

    def noisy(values):
        return phase * values + noise * (rng.standard_normal(K) + 1j * rng.standard_normal(K))

    return noisy(null), noisy(channel)


def _detect(paths, frames, rng, **options):
    detector = TiiDetector()
    for frame in range(frames):
        result = detector.add(*_carriers_of_frame(paths, rng, tii=frame % 2 == 1, **options))
    return result


def _described(paths):
    return [(round(path.delay_s * 1e6, 1), path.transmitter, path.echo) for path in paths]


def test_tii_lies_304_samples_behind_the_impulse_response_in_mode_i():
    """As measured on air on four ensembles, to a quarter sample."""
    assert OFFSET == 304


def test_paths_are_named_by_the_code_they_carry():
    """Two transmitters, a reflection of the first (its code, later) and a path without
    TII, at fractional delays; TII in every second null symbol, as on air."""
    rng = np.random.default_rng(11)
    paths = [
        (0.4, 1.0, (19, 4)),
        (40.3, 0.18, (19, 4)),  # reflection, 19.7 µs
        (194.2, 0.1, (24, 3)),  # a further transmitter, 94.8 µs
        (313.6, 0.18, None),  # without TII, 153.1 µs
    ]
    result = _detect(paths, 40, rng)

    assert _described(result.paths) == [
        (0.2, (19, 4), False),
        (19.7, (19, 4), True),
        (94.8, (24, 3), False),
        (153.1, None, False),
    ]
    levels = [round(path.level_db) for path in result.paths]
    assert levels == [0, -15, -20, -15]


def test_a_transmitter_below_the_detection_limit_is_found_at_its_path():
    """24-03 lies 30 dB below 19-04, with the detection limit of the power averaging
    at -20 dB: the comparison at its path finds it (in 40 of 40 noise patterns over
    128 frames); the shares cover both."""
    rng = np.random.default_rng(12)
    paths = [(0.0, 1.0, (19, 4)), (194.0, 0.0316, (24, 3))]
    result = _detect(paths, 128, rng, noise=0.05)

    assert result.detection_limit_db > -25.0
    codes = [(t.main_id, t.sub_id) for t in result.transmitters]
    assert codes == [(19, 4), (24, 3)]
    weak = result.transmitters[1]
    assert weak.level_db == pytest.approx(-30.0, abs=0.5)
    assert weak.delay_s == pytest.approx(194.0 / SAMPLE_RATE, abs=0.5 / SAMPLE_RATE)
    assert sum(t.share for t in result.transmitters) == pytest.approx(1.0)
    assert result.transmitters[0].delay_s == pytest.approx(0.0, abs=0.5 / SAMPLE_RATE)


def test_a_path_close_to_a_stronger_one_without_a_code_of_its_own_is_left_out():
    """A peak 4 µs behind the main path that carries its code: on air on all four
    channels, likely the filters."""
    rng = np.random.default_rng(13)
    paths = [(0.0, 1.0, (35, 13)), (8.0, 0.05, (35, 13)), (257.0, 0.05, (35, 1))]
    result = _detect(paths, 40, rng)

    assert _described(result.paths) == [(0.0, (35, 13), False), (125.5, (35, 1), False)]


def test_without_tii_there_are_no_paths():
    rng = np.random.default_rng(14)
    paths = [(0.0, 1.0, None), (100.0, 0.3, None)]
    result = _detect(paths, 20, rng)

    assert result.transmitters == [] and result.paths == []


def test_without_channel_estimates_only_the_transmitters_are_detected():
    rng = np.random.default_rng(15)
    detector = TiiDetector()
    for _ in range(20):
        null, _channel = _carriers_of_frame([(0.0, 1.0, (19, 4))], rng)
        result = detector.add(null)

    assert [(t.main_id, t.sub_id) for t in result.transmitters] == [(19, 4)]
    assert result.paths == [] and result.transmitters[0].delay_s is None


def test_reset_forgets_the_paths():
    rng = np.random.default_rng(16)
    detector = TiiDetector()
    for frame in range(20):
        detector.add(*_carriers_of_frame([(0.0, 1.0, (19, 4))], rng, tii=frame % 2 == 1))
    detector.reset()
    result = detector.add(*_carriers_of_frame([(0.0, 1.0, (19, 4))], rng))

    assert result.paths == []


def test_a_code_of_its_own_wins_over_the_code_of_a_stronger_path():
    """On 8C the path at 125 µs carried 35-13, the code of the strongest path, as well
    as its own 35-01."""
    gain = np.zeros((24, 8), dtype=complex)
    deviation = np.full((24, 8), 0.1)
    for comb in (1, 13):
        gain[comb, [b for b, used in enumerate(PATTERNS[35]) if used]] = 0.9

    assert code_of_path(gain, deviation, taken={(35, 13)}) == (35, 1)
    assert code_of_path(gain, deviation, taken=set()) in {(35, 1), (35, 13)}


def test_a_gain_far_above_one_is_no_code():
    """Leakage from a stronger path that the block's carriers barely tell apart: on 7A
    31-03 showed gains of 2.3 to 3 at the path of 31-01."""
    gain = np.zeros((24, 8), dtype=complex)
    gain[3, [b for b, used in enumerate(PATTERNS[31]) if used]] = 2.8
    deviation = np.full((24, 8), 0.1)

    assert code_of_path(gain, deviation, taken=set()) is None


def test_a_code_found_before_holds_with_less_margin():
    gain = np.zeros((24, 8), dtype=complex)
    gain[3, [b for b, used in enumerate(PATTERNS[24]) if used]] = 0.9
    deviation = np.full((24, 8), 0.3)  # 3 standard deviations

    assert code_of_path(gain, deviation, taken=set()) is None
    assert code_of_path(gain, deviation, taken=set(), previous=(24, 3)) == (24, 3)


def test_a_code_no_path_carries_is_not_listed():
    """A spur on the carriers of 24-03 with a phase of its own in every null symbol:
    the power averaging takes it for a transmitter, but it belongs to no path."""
    rng = np.random.default_rng(18)
    detector = TiiDetector()
    spur = code_positions(24, 3)
    for frame in range(40):
        null, channel = _carriers_of_frame([(0.0, 1.0, (19, 4))], rng, tii=frame % 2 == 1)
        null[spur] += 0.5 * np.exp(2j * np.pi * rng.random(len(spur)))
        result = detector.add(null, channel)

    averaged = detect_transmitters(detector._power_sum / len(detector._powers), 32)
    assert (24, 3) in [(t.main_id, t.sub_id) for t in averaged.transmitters]
    assert [(t.main_id, t.sub_id) for t in result.transmitters] == [(19, 4)]


def test_a_weak_new_code_shows_once_two_identifications_agree(monkeypatch):
    """At a weak path the code found may change from one identification to the next;
    it shows once two in a row agree, and until then the path keeps what it showed."""
    rng = np.random.default_rng(19)
    tracker = PathTracker()
    main = code_positions(19, 4)
    for frame in range(40):
        null, channel = _carriers_of_frame(
            [(0.0, 1.0, (19, 4)), (194.0, 0.1, None)], rng, tii=frame % 2 == 1
        )
        power = impulse_response(channel, MODE_I, SAMPLE_RATE)[1]
        tracker.add(null, channel, power, main)

    found = iter([(24, 3), None, (24, 3), (24, 3), (56, 3), (56, 3)])
    real = tii.code_of_path

    def scripted(gain, deviation, taken, previous=None):
        code = real(gain, deviation, taken, previous)
        return code if code == (19, 4) else next(found)

    monkeypatch.setattr(tii, "code_of_path", scripted)
    monkeypatch.setattr(
        tii, "code_strength", lambda gain, deviation, code: 100.0 if code == (19, 4) else 5.0
    )

    shown = []
    for _ in range(6):
        weak = [path for path in tracker.paths() if path.delay_s > 50e-6]
        shown.append(weak[0].transmitter)
    # 24-03 once, gone, then twice; 56-03 waits a round while 24-03 is still carried.
    assert shown == [None, None, None, (24, 3), (24, 3), (56, 3)]


def _delayed(signal: np.ndarray, samples: int) -> np.ndarray:
    return np.concatenate([np.zeros(samples, dtype=complex), signal[: len(signal) - samples]])


def test_paths_through_the_real_demodulation():
    """The time base of TII and impulse response as the worker sees them: a reflection
    100 samples (48.8 µs) behind its transmitter carries its code."""
    rng = np.random.default_rng(17)
    detector = TiiDetector()
    for _ in range(12):
        direct = _frame([(19, 4, 1.0)], rng, noise=0.0)
        signal = direct + 0.3 * _delayed(direct, 100)
        signal = signal + 0.002 * (
            rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
        )
        frame = demodulate_frame_symbols(signal, 0, MODE_I, SAMPLE_RATE, max_data_symbols=3)
        assert frame is not None
        frame_start = MODE_I.null_symbol_length + frame.sample_correction
        spectrum = null_symbol_spectrum(
            signal, frame_start, frame.frequency_offset_hz, SAMPLE_RATE, MODE_I
        )
        result = detector.add(null_symbol_carriers(spectrum, MODE_I), frame.channel_estimate)

    assert _described(result.paths) == [(0.0, (19, 4), False), (48.8, (19, 4), True)]
