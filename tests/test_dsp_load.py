"""The load report of `DspWorker`: how much of real time the signal processing needs,
and how much of that its thread spent computing rather than waiting."""

from __future__ import annotations

import numpy as np
import pytest

from dab.constants import MODE_I, SAMPLE_RATE
from dsp import pipeline
from dsp.pipeline import LOAD_REPORT_INTERVAL_SECONDS, DspWorker


def test_the_load_is_reported_by_the_wall_clock_and_by_the_cpu_time_of_the_thread(monkeypatch):
    clock = {"wall": 100.0, "cpu": 5.0}
    monkeypatch.setattr(pipeline, "_wall_clock", lambda: clock["wall"])
    monkeypatch.setattr(pipeline, "_cpu_clock", lambda: clock["cpu"])
    worker = DspWorker(SAMPLE_RATE, MODE_I)
    reports: list[tuple[float, float]] = []
    worker.load_ready.connect(lambda demand, compute: reports.append((demand, compute)))
    cost = {"wall": 0.3, "cpu": 0.1}  # a block of 0.25 s of signal: 0.3 s, computing 0.1 s of it

    def work(_samples: np.ndarray) -> None:
        clock["wall"] += cost["wall"]
        clock["cpu"] += cost["cpu"]

    monkeypatch.setattr(worker, "_process_samples", work)
    block = np.zeros(int(SAMPLE_RATE * LOAD_REPORT_INTERVAL_SECONDS / 4), dtype=np.complex64)

    for _ in range(3):
        worker.process_samples(block)
    assert reports == []  # not before a second of signal
    worker.process_samples(block)
    assert reports == [(pytest.approx(1.2), pytest.approx(0.4))]

    # The next second starts afresh: keeping up now, and hardly waiting.
    cost["wall"], cost["cpu"] = 0.1, 0.09
    for _ in range(4):
        worker.process_samples(block)
    assert reports[1:] == [(pytest.approx(0.4), pytest.approx(0.36))]


def test_the_clocks_are_the_wall_clock_and_the_cpu_time_of_the_calling_thread():
    import time

    assert pipeline._wall_clock is time.perf_counter
    assert pipeline._cpu_clock is time.thread_time
