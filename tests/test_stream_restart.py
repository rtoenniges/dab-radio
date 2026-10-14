"""Channel change during reception: `DspWorker.restart_stream` separates the samples
of the old frequency from those of the new one."""

from __future__ import annotations

import numpy as np

from dab.constants import MODE_I, SAMPLE_RATE
from dsp.pipeline import DspWorker
from dsp.scan import MIN_LISTEN_SECONDS
from sdr.worker import DEFAULT_READ_SIZE
from tests.dab_signal import (
    build_stream,
    fib,
    fig0_2_audio_services,
    fig1_ensemble_label,
    fig1_service_label,
)


def _ensemble_stream(
    eid: int,
    label: str,
    sid: int,
    service: str,
    num_frames: int = 6,
    extra_figs: tuple[bytes, ...] = (),
) -> np.ndarray:
    fibs = [
        fib(fig1_ensemble_label(eid, label)),
        fib(fig1_service_label(sid, service)),
        fib(fig0_2_audio_services([(sid, 1)])),
    ] + [fib(fig) for fig in extra_figs]
    rng = np.random.default_rng(eid)
    return build_stream(
        fibs + [fib()] * (12 - len(fibs)), num_frames=num_frames, rng=rng, start_offset=30_000
    )


def _feed(worker: DspWorker, stream: np.ndarray) -> None:
    for start in range(0, len(stream), DEFAULT_READ_SIZE):
        worker.process_samples(stream[start : start + DEFAULT_READ_SIZE])


def test_restart_stream_separates_the_ensembles_of_two_frequencies():
    """Results before the confirmation belong to the old frequency, those after it
    only to the new one. The GUI relies on exactly this to keep services of the
    previous ensemble out of the new channel's list."""
    worker = DspWorker(SAMPLE_RATE, MODE_I)
    events = []
    worker.services_updated.connect(
        lambda label, services, _programme_types: events.append((label, services))
    )
    worker.stream_restarted.connect(events.append)

    worker.restart_stream(1)
    _feed(worker, _ensemble_stream(0x10BC, "Mux A", 0xD001, "Radio A"))
    worker.restart_stream(2)
    _feed(worker, _ensemble_stream(0x10BD, "Mux B", 0xD002, "Radio B"))

    boundary = events.index(2)
    # The reset empties the list right before it is confirmed.
    assert events[boundary - 1] == (None, [])
    assert events[boundary - 2] == ("Mux A", [(0xD001, "Radio A")])
    after = events[boundary + 1 :]
    assert after[-1] == ("Mux B", [(0xD002, "Radio B")])
    assert all(0xD001 not in dict(services) for _label, services in after)


def test_frequency_information_is_reset_with_the_stream():
    """The EId comes with the ensemble label; the reset clears it before the new
    frequency is confirmed, like the service list."""
    worker = DspWorker(SAMPLE_RATE, MODE_I)
    events = []
    worker.frequency_information_updated.connect(
        lambda eid, frequencies: events.append((eid, frequencies))
    )
    worker.stream_restarted.connect(events.append)

    worker.restart_stream(1)
    _feed(worker, _ensemble_stream(0x10BC, "Mux A", 0xD001, "Radio A"))
    worker.restart_stream(2)
    _feed(worker, _ensemble_stream(0x10BD, "Mux B", 0xD002, "Radio B"))

    boundary = events.index(2)
    assert events[boundary - 2 : boundary] == [(0x10BC, {}), (None, {})]
    assert events[-1] == (0x10BD, {})


def test_the_ensembles_of_the_services_are_reported_once_and_reset_with_the_stream():
    """FIG 0/24 as 5D sends it, repeated in every frame: reported when it first comes,
    cleared by the reset before the new frequency is confirmed."""
    worker = DspWorker(SAMPLE_RATE, MODE_I)
    events = []
    worker.service_ensembles_updated.connect(events.append)
    worker.stream_restarted.connect(events.append)
    rtl_radio = bytes.fromhex("061812290111f7")

    worker.restart_stream(1)
    _feed(
        worker, _ensemble_stream(0x11F7, "Antenne DE", 0x1229, "RTL RADIO", extra_figs=(rtl_radio,))
    )
    worker.restart_stream(2)

    assert events == [{}, 1, {0x1229: (0x11F7,)}, {}, 2]


def test_the_complete_service_list_is_reported_once_per_stream():
    """By the rule of the scan: not before `MIN_LISTEN_SECONDS` after the ensemble is
    confirmed, right after the list it completes, and anew after a restart."""
    worker = DspWorker(SAMPLE_RATE, MODE_I)
    events = []
    worker.services_updated.connect(lambda _label, services, _types: events.append(services))
    worker.service_list_completed.connect(lambda: events.append("complete"))
    worker.stream_restarted.connect(events.append)

    worker.restart_stream(1)
    stream = _ensemble_stream(0x10BC, "Mux A", 0xD001, "Radio A", num_frames=18)
    early = int(0.6 * MIN_LISTEN_SECONDS * SAMPLE_RATE)
    _feed(worker, stream[:early])
    assert [(0xD001, "Radio A")] in events
    assert "complete" not in events

    _feed(worker, stream[early:])
    assert events.count("complete") == 1
    assert events[events.index("complete") - 1] == [(0xD001, "Radio A")]

    worker.restart_stream(2)
    _feed(worker, _ensemble_stream(0x10BD, "Mux B", 0xD002, "Radio B", num_frames=18))
    after = events[events.index(2) + 1 :]
    assert after.count("complete") == 1
    assert after[after.index("complete") - 1] == [(0xD002, "Radio B")]
