"""Soft bits weighted by the reliability of each carrier (`dsp.qpsk.CarrierReliability`):
a narrow interferer in the band must not break the error correction."""

from __future__ import annotations

import numpy as np

from dab.constants import MODE_I
from dsp.ofdm import FrameDemodulator
from dsp.pipeline import DspWorker
from dsp.qpsk import CarrierReliability
from dsp.scan import EnsembleProbe
from sdr.scanner import SCAN_BLOCK_SAMPLES
from sdr.worker import DEFAULT_READ_SIZE
from tests.dab_signal import (
    FIBS_PER_FRAME,
    SAMPLE_RATE,
    build_stream,
    fib,
    fig1_ensemble_label,
    interfering_carrier,
    noise,
)

K = MODE_I.num_carriers

# A spur a little off carrier k = -635, 20 dB above the power of one carrier: without
# the weighting, about a third of the FIBs fail.
SPUR_HZ = -635_300.0
SPUR_TO_CARRIER = 100.0


def _dqpsk_points(rng: np.random.Generator, symbols: int, noise: np.ndarray, spur: np.ndarray):
    """D-QPSK points z(l)·z*(l-1) over a flat channel: complex noise of the given level
    per carrier, plus an interferer of random phase with the given amplitude."""
    shape = (symbols + 1, K)
    # Each symbol turns the previous one by ±45° or ±135°.
    phases = np.cumsum(rng.integers(0, 4, shape) * (np.pi / 2) + np.pi / 4, axis=0)
    received = (
        np.exp(1j * phases)
        + noise * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)
        + spur * np.exp(2j * np.pi * rng.random(shape))
    )
    return received[1:] * np.conj(received[:-1])


def test_an_interfered_carrier_and_a_noisy_one_get_little_weight():
    rng = np.random.default_rng(1)
    level = np.full(K, 0.1)
    level[200] = 0.4
    spur = np.zeros(K)
    spur[100] = 3.0
    reliability = CarrierReliability()

    for _ in range(10):
        reliability.update(_dqpsk_points(rng, 16, level, spur))

    weights = reliability.weights
    assert np.median(weights) == 1.0
    assert weights[100] < 0.02
    assert weights[200] < 0.2
    # The clean carriers scatter with the estimate from 160 points, but far above.
    assert np.delete(weights, [100, 200]).min() > 0.4


def test_the_estimate_starts_empty_and_is_forgotten_on_reset():
    reliability = CarrierReliability()
    assert reliability.weights is None

    reliability.update(_dqpsk_points(np.random.default_rng(2), 16, np.full(K, 0.1), np.zeros(K)))
    assert reliability.weights is not None

    reliability.reset()
    assert reliability.weights is None


def test_noise_free_points_give_finite_weights():
    """Synthetic signals without noise put every point on its ideal position."""
    reliability = CarrierReliability()
    reliability.update(_dqpsk_points(np.random.default_rng(3), 16, np.zeros(K), np.zeros(K)))

    assert np.all(np.isfinite(reliability.weights))


def test_the_demodulator_carries_the_estimate_over_reliable_frames_only():
    rng = np.random.default_rng(4)
    stream = build_stream([fib()] * FIBS_PER_FRAME, 2, rng)
    demodulator = FrameDemodulator(MODE_I, SAMPLE_RATE)

    frame = demodulator.demodulate(stream, 0, 16)
    assert frame.reliable
    assert frame.carrier_weights is not None and len(frame.carrier_weights) == K

    # A frame without a PRS neither feeds the estimate nor loses it.
    lost = demodulator.demodulate(noise(2 * MODE_I.frame_length, rng), 0, 16)
    assert not lost.reliable
    np.testing.assert_array_equal(lost.carrier_weights, frame.carrier_weights)

    demodulator.reset()
    assert (
        demodulator.demodulate(noise(2 * MODE_I.frame_length, rng), 0, 16).carrier_weights is None
    )


def _stream_with_spur(seed: int, num_frames: int) -> np.ndarray:
    fibs = [fib(fig1_ensemble_label(0x10BC, "Test Mux"))] + [fib()] * (FIBS_PER_FRAME - 1)
    stream = build_stream(fibs, num_frames, np.random.default_rng(seed), start_offset=50_000)
    power = SPUR_TO_CARRIER * np.mean(np.abs(stream) ** 2) / K
    return stream + interfering_carrier(len(stream), power, SPUR_HZ)


def test_reception_reads_every_fib_next_to_a_strong_spur():
    stream = _stream_with_spur(5, 8)
    worker = DspWorker(SAMPLE_RATE, MODE_I)
    frames = []
    decode = worker._process_frame

    def count(samples: np.ndarray, null_index: int) -> bool:
        frames.append(null_index)
        return decode(samples, null_index)

    worker._process_frame = count  # type: ignore[method-assign]
    for off in range(0, len(stream) - DEFAULT_READ_SIZE + 1, DEFAULT_READ_SIZE):
        worker.process_samples(stream[off : off + DEFAULT_READ_SIZE])

    assert frames
    assert worker._valid_fibs == FIBS_PER_FRAME * len(frames)


def test_the_scan_reads_every_fib_next_to_a_strong_spur():
    probe = EnsembleProbe(MODE_I, SAMPLE_RATE)

    probe.process(_stream_with_spur(6, 6)[:SCAN_BLOCK_SAMPLES])

    assert probe.total_fibs > 0
    assert probe.valid_fibs == probe.total_fibs
