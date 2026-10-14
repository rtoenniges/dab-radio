import numpy as np
import pytest

import dsp.scan
from dab.constants import MODE_I
from dab.ensemble_cache import SOURCE_SCAN
from dsp.scan import MIN_LISTEN_SECONDS, NO_SIGNAL_SECONDS, EnsembleProbe
from sdr.scanner import SCAN_BLOCK_SAMPLES
from tests.dab_signal import (
    SAMPLE_RATE,
    build_stream,
    fib,
    fig0_2_audio_services,
    fig0_17_programme_types,
    fig1_ensemble_label,
    fig1_service_label,
    noise,
)

_BLOCK_SECONDS = SCAN_BLOCK_SAMPLES / SAMPLE_RATE
_PROGRAMME_TYPES = {0xD001: 10, 0xD002: 3}


def _ensemble_fibs(labelled_sids: list[int], typed_sids: list[int] = ()) -> list[np.ndarray]:
    """FIBs of an ensemble "Test Mux" with two audio services, of which only
    `labelled_sids` get a name and only `typed_sids` a programme type."""
    names = {0xD001: "Radio Eins", 0xD002: "Radio Zwei"}
    fibs = [
        fib(fig1_ensemble_label(0x10BC, "Test Mux")),
        fib(fig0_2_audio_services([(0xD001, 1), (0xD002, 2)])),
    ]
    fibs += [fib(fig1_service_label(sid, names[sid])) for sid in labelled_sids]
    if typed_sids:
        types = [(sid, _PROGRAMME_TYPES[sid]) for sid in typed_sids]
        fibs.append(fib(fig0_17_programme_types(types)))
    return fibs + [fib()] * (12 - len(fibs))


def _feed_until_finished(probe: EnsembleProbe, stream: np.ndarray) -> int:
    blocks = 0
    while not probe.finished:
        start = blocks * SCAN_BLOCK_SAMPLES
        assert start + SCAN_BLOCK_SAMPLES <= len(stream), "the probe did not finish in time"
        probe.process(stream[start : start + SCAN_BLOCK_SAMPLES])
        blocks += 1
    return blocks


def test_probe_finds_ensemble_and_complete_service_list():
    rng = np.random.default_rng(1)
    # An arbitrary frame start and a frequency offset as an inaccurate RTL-SDR
    # crystal causes (more than one subcarrier spacing).
    stream = build_stream(
        _ensemble_fibs([0xD001, 0xD002]),
        num_frames=24,
        rng=rng,
        start_offset=70_000,
        snr_db=15.0,
        freq_offset_hz=12_345.0,
    )
    probe = EnsembleProbe(MODE_I, SAMPLE_RATE)

    blocks = _feed_until_finished(probe, stream)

    # Done as soon as the minimum time is up, not only at the maximum time.
    assert blocks * _BLOCK_SECONDS < MIN_LISTEN_SECONDS + 2 * _BLOCK_SECONDS
    info = probe.result()
    assert info is not None
    assert info.label == "Test Mux"
    assert info.services == ((0xD001, "Radio Eins"), (0xD002, "Radio Zwei"))
    assert info.source == SOURCE_SCAN
    assert info.fib_success == pytest.approx(1.0)
    assert info.eid == 0x10BC
    assert info.frequencies == ()  # the ensemble sends no FIG 0/21
    assert info.service_ensembles == ()  # nor FIG 0/24


def test_probe_rejects_empty_channel_after_no_signal_time():
    rng = np.random.default_rng(2)
    probe = EnsembleProbe(MODE_I, SAMPLE_RATE)

    blocks = _feed_until_finished(probe, noise(10 * SCAN_BLOCK_SAMPLES, rng))

    assert blocks == int(np.ceil(NO_SIGNAL_SECONDS / _BLOCK_SECONDS))
    assert probe.total_fibs > 0
    assert probe.valid_fibs == 0
    assert probe.result() is None


def test_probe_waits_for_missing_service_labels_up_to_max_listen_time(monkeypatch):
    """As long as a service from FIG 0/2 has no name, listening continues up to the
    maximum time; whatever has arrived by then is stored."""
    monkeypatch.setattr(dsp.scan, "MAX_LISTEN_SECONDS", 3 * _BLOCK_SECONDS)
    rng = np.random.default_rng(3)
    stream = build_stream(_ensemble_fibs([0xD001]), num_frames=24, rng=rng, start_offset=1_000)
    probe = EnsembleProbe(MODE_I, SAMPLE_RATE)

    blocks = _feed_until_finished(probe, stream)

    # One block until the ensemble is confirmed, then three blocks of maximum time.
    assert blocks == 4
    info = probe.result()
    assert info is not None
    assert info.services == ((0xD001, "Radio Eins"),)


def test_probe_stores_the_programme_types():
    rng = np.random.default_rng(4)
    both = [0xD001, 0xD002]
    stream = build_stream(_ensemble_fibs(both, both), num_frames=24, rng=rng, start_offset=1_000)
    probe = EnsembleProbe(MODE_I, SAMPLE_RATE)

    blocks = _feed_until_finished(probe, stream)

    assert blocks * _BLOCK_SECONDS < MIN_LISTEN_SECONDS + 2 * _BLOCK_SECONDS
    assert probe.result().programme_types == ((0xD001, 10), (0xD002, 3))


def test_probe_waits_for_missing_programme_types_once_the_ensemble_sends_them(monkeypatch):
    """An ensemble without FIG 0/17 does not delay the scan (see the first test); one
    that sends it gets time until every service has its programme type."""
    monkeypatch.setattr(dsp.scan, "MAX_LISTEN_SECONDS", 3 * _BLOCK_SECONDS)
    rng = np.random.default_rng(5)
    stream = build_stream(
        _ensemble_fibs([0xD001, 0xD002], [0xD001]), num_frames=24, rng=rng, start_offset=1_000
    )
    probe = EnsembleProbe(MODE_I, SAMPLE_RATE)

    blocks = _feed_until_finished(probe, stream)

    assert blocks == 4
    assert probe.result().programme_types == ((0xD001, 10),)
