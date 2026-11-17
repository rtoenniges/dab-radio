"""Emergency warnings through `dsp.pipeline.DspWorker`: from FIG 0/15 in the FIC of a
synthetic ensemble to the reported state, and with `play_alert` to the audio of the
warning's subchannel and back."""

from __future__ import annotations

import numpy as np

from dab.constants import MODE_I, SAMPLE_RATE
from dab.ews import EwsState, Stage
from dsp.pipeline import DspWorker, SubchannelStatus
from sdr.worker import DEFAULT_READ_SIZE
from tests.dab_signal import FIBS_PER_FRAME, build_stream, fib, fig0_2_audio_services

EID = 0x10BC
SERVICE = 0x10C3  # the one listened to
SERVICE_SUBCHANNEL = 1
ALERT_SERVICE = 0x10C4  # carries the warnings
ALERT_SUBCHANNEL = 2

FIG0_0 = bytes.fromhex("050010bc2342")  # EId 10BC, Al flag set, as on 5C
# FIG 0/1: both subchannels EEP-3A with 36 CU, one after the other.
FIG0_1 = bytes.fromhex("0901" + "0400" + "8824" + "0824" + "8824")
HEARTBEAT = bytes.fromhex("018f")
# Trigger in subchannel 2, last of its group, Level 1 Start, incident 7, no codes.
TRIGGER = bytes.fromhex("030f" + "42" + "87")


def _stream(fig0_15: bytes, seed: int, start_offset: int = 0) -> np.ndarray:
    figs = [
        FIG0_0,
        FIG0_1,
        fig0_2_audio_services([(SERVICE, SERVICE_SUBCHANNEL), (ALERT_SERVICE, ALERT_SUBCHANNEL)]),
        fig0_15,
    ]
    fibs = [fib(fig) for fig in figs]
    fibs += [fib()] * (FIBS_PER_FRAME - len(fibs))
    return build_stream(
        fibs, num_frames=4, rng=np.random.default_rng(seed), start_offset=start_offset
    )


def _feed(worker: DspWorker, stream: np.ndarray) -> None:
    for start in range(0, len(stream), DEFAULT_READ_SIZE):
        worker.process_samples(stream[start : start + DEFAULT_READ_SIZE])


def _worker() -> tuple[DspWorker, list[EwsState], list[SubchannelStatus]]:
    worker = DspWorker(SAMPLE_RATE, MODE_I)
    states: list[EwsState] = []
    subchannels: list[SubchannelStatus] = []
    worker.ews_ready.connect(states.append)
    worker.subchannel_info_ready.connect(subchannels.append)
    worker.select_service(SERVICE)
    return worker, states, subchannels


def test_the_heartbeat_tells_that_the_ensemble_takes_part():
    worker, states, _ = _worker()

    _feed(worker, _stream(HEARTBEAT, seed=1, start_offset=30_000))

    assert states[-1] == EwsState(participating=True, eid=EID)


def test_a_trigger_is_reported_with_the_service_carrying_it():
    worker, states, subchannels = _worker()

    _feed(worker, _stream(TRIGGER, seed=2, start_offset=30_000))

    (alert,) = states[-1].alerts
    assert (alert.eid, alert.stage, alert.incident) == (EID, Stage.LEVEL_1_START, 7)
    assert (alert.subchannel_id, alert.sid, alert.receivable) == (
        ALERT_SUBCHANNEL,
        ALERT_SERVICE,
        True,
    )
    assert alert.whole_area
    assert states[-1].running == {ALERT_SUBCHANNEL}
    # Reporting is all: whether to play it is the receiver's decision.
    assert subchannels[-1].subchannel.subchannel_id == SERVICE_SUBCHANNEL


def test_the_audio_goes_to_the_warning_and_back():
    worker, _, subchannels = _worker()
    _feed(worker, _stream(TRIGGER, seed=3, start_offset=30_000))

    worker.play_alert(ALERT_SUBCHANNEL)
    during = subchannels[-1]
    worker.play_alert(None)
    after = subchannels[-1]

    assert during.subchannel.subchannel_id == ALERT_SUBCHANNEL
    assert during.alert and during.announcement is None
    assert after.subchannel.subchannel_id == SERVICE_SUBCHANNEL
    assert not after.alert


def test_a_restart_of_the_stream_ends_the_warning():
    worker, states, subchannels = _worker()
    _feed(worker, _stream(TRIGGER, seed=4, start_offset=30_000))
    worker.play_alert(ALERT_SUBCHANNEL)

    worker.restart_stream(1)
    _feed(worker, _stream(HEARTBEAT, seed=5, start_offset=30_000))

    assert EwsState() in states
    assert states[-1].alerts == ()
    assert subchannels[-1].subchannel.subchannel_id == SERVICE_SUBCHANNEL
    assert not subchannels[-1].alert
