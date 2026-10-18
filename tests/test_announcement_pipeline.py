"""Announcements through `dsp.pipeline.DspWorker`: from FIG 0/18 and FIG 0/19 in the FIC
of a synthetic ensemble to the status of the selected service and, if enabled, to the
audio of the announcement's subchannel and back."""

from __future__ import annotations

import numpy as np

from dab.constants import MODE_I, SAMPLE_RATE
from dsp.pipeline import DspWorker, SubchannelStatus
from sdr.worker import DEFAULT_READ_SIZE
from tests.dab_signal import FIBS_PER_FRAME, build_stream, fib, fig0_2_audio_services

NDR_INFO = 0xD384  # takes road traffic announcements, cluster 3
NDR_2 = 0xD382  # carries them in its subchannel
NDR_INFO_SUBCHANNEL = 1
NDR_2_SUBCHANNEL = 2

# FIG 0/1: both subchannels EEP-3A with 36 CU, one after the other.
FIG0_1 = bytes.fromhex("0901" + "0400" + "8824" + "0824" + "8824")
FIG0_18 = bytes.fromhex("0712d38400020103")  # as on 7A: NDR Info, road traffic, cluster 3
FIG0_19_RUNNING = bytes.fromhex("051303000282")  # cluster 3, road traffic, new, subchannel 2
FIG0_19_ENDED = bytes.fromhex("051303000002")  # ASw flags zero


def _stream(fig0_19: bytes, seed: int, start_offset: int = 0) -> np.ndarray:
    figs = [
        FIG0_1,
        fig0_2_audio_services([(NDR_INFO, NDR_INFO_SUBCHANNEL), (NDR_2, NDR_2_SUBCHANNEL)]),
        FIG0_18,
        fig0_19,
    ]
    fibs = [fib(fig) for fig in figs]
    fibs += [fib()] * (FIBS_PER_FRAME - len(fibs))
    return build_stream(
        fibs, num_frames=4, rng=np.random.default_rng(seed), start_offset=start_offset
    )


def _feed(worker: DspWorker, stream: np.ndarray) -> None:
    for start in range(0, len(stream), DEFAULT_READ_SIZE):
        worker.process_samples(stream[start : start + DEFAULT_READ_SIZE])


def _worker(*, switching: bool) -> tuple[DspWorker, list, list[SubchannelStatus]]:
    worker = DspWorker(SAMPLE_RATE, MODE_I)
    statuses = []
    subchannels: list[SubchannelStatus] = []
    worker.announcement_ready.connect(statuses.append)
    worker.subchannel_info_ready.connect(subchannels.append)
    worker.set_announcement_switching_enabled(switching)
    worker.select_service(NDR_INFO)
    return worker, statuses, subchannels


def test_a_running_announcement_is_reported_for_the_selected_service():
    worker, statuses, subchannels = _worker(switching=False)

    _feed(worker, _stream(FIG0_19_RUNNING, seed=1, start_offset=30_000))

    status = statuses[-1]
    assert status.support.flags == 0x0002
    assert status.support.clusters == (3,)
    assert status.active.subchannel_id == NDR_2_SUBCHANNEL
    assert status.active.new
    assert status.source_sid == NDR_2
    # Without switching the audio stays with the service.
    assert subchannels[-1].subchannel.subchannel_id == NDR_INFO_SUBCHANNEL
    assert subchannels[-1].announcement is None


def test_the_audio_follows_the_announcement_and_returns_after_its_end():
    worker, statuses, subchannels = _worker(switching=True)

    _feed(worker, _stream(FIG0_19_RUNNING, seed=2, start_offset=30_000))
    during = subchannels[-1]
    _feed(worker, _stream(FIG0_19_ENDED, seed=3))

    assert during.subchannel.subchannel_id == NDR_2_SUBCHANNEL
    assert during.announcement.subchannel_id == NDR_2_SUBCHANNEL
    assert statuses[-1].active is None
    assert subchannels[-1].subchannel.subchannel_id == NDR_INFO_SUBCHANNEL
    assert subchannels[-1].announcement is None


def test_switching_can_be_turned_off_during_an_announcement():
    worker, _statuses, subchannels = _worker(switching=True)
    _feed(worker, _stream(FIG0_19_RUNNING, seed=4, start_offset=30_000))

    worker.set_announcement_switching_enabled(False)

    assert subchannels[-1].subchannel.subchannel_id == NDR_INFO_SUBCHANNEL


def test_a_service_selection_is_always_answered_with_its_status():
    """The GUI clears its displays on a service change and relies on a fresh report."""
    worker, statuses, _subchannels = _worker(switching=False)
    _feed(worker, _stream(FIG0_19_RUNNING, seed=5, start_offset=30_000))
    count = len(statuses)

    worker.select_service(NDR_2)
    worker.select_service(NDR_2)

    assert len(statuses) == count + 2
    assert statuses[-1].support is None
    assert statuses[-1].active is None


# FIG 0/25: NDR Info also takes road traffic from OE cluster 5, served by ensemble 1188;
# FIG 0/26: road traffic in OE cluster 5, carried there by D581 (built after ETSI EN 300
# 401 V2.1.1, clauses 8.1.6.3 and 8.1.6.4: no transmitter received sends either).
FIG0_25 = bytes.fromhex("0919d38400020005" + "1188")
FIG0_26_RUNNING = bytes.fromhex("071a05000280d581")


def test_an_announcement_of_another_ensemble_is_reported_but_heard_there():
    worker, statuses, subchannels = _worker(switching=True)
    figs = [
        FIG0_1,
        fig0_2_audio_services([(NDR_INFO, NDR_INFO_SUBCHANNEL), (NDR_2, NDR_2_SUBCHANNEL)]),
        FIG0_25,
        FIG0_26_RUNNING,
    ]
    fibs = [fib(fig) for fig in figs] + [fib()] * (FIBS_PER_FRAME - len(figs))
    stream = build_stream(fibs, num_frames=4, rng=np.random.default_rng(6), start_offset=30_000)

    _feed(worker, stream)

    status = statuses[-1]
    assert status.oe_support.targets == ((5, 0x1188),)
    assert (status.oe_active.target_sid, status.oe_eid) == (0xD581, 0x1188)
    assert status.active is None
    # Retuning is up to the controller; the audio stays with the service.
    assert subchannels[-1].subchannel.subchannel_id == NDR_INFO_SUBCHANNEL


def test_the_service_carrying_an_announcement_knows_it():
    """NDR 2 is in no cluster, but cluster 3 is served from its subchannel."""
    worker, statuses, _subchannels = _worker(switching=False)
    worker.select_service(NDR_2)

    _feed(worker, _stream(FIG0_19_RUNNING, seed=7, start_offset=30_000))

    assert statuses[-1].active is None
    assert statuses[-1].carried.subchannel_id == NDR_2_SUBCHANNEL
