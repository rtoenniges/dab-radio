"""`RspDuo` against a fake of the SDRplay API: choice of the tuner, configuration for DAB,
callbacks and the German messages for everything the user can fix."""

import ctypes

import numpy as np
import pytest

from dab.constants import SAMPLE_RATE
from sdr import rspduo as rspduo_module
from sdr import sdrplay_api as sp
from sdr.exceptions import SdrplayError
from sdr.packet_ring import FREQUENCY_CHANGED
from sdr.rspduo import GainReport, RspDuo, RspDuoSettings
from sdr.service_state import ServiceState
from tests.fake_sdrplay import FakeApi, make_device

FREQ_5C = 178_352_000.0


def _open(api: FakeApi, tuner: int = 1) -> RspDuo:
    device = RspDuo(api_factory=lambda: api)
    device.open(tuner)
    return device


def _start(device: RspDuo, settings: RspDuoSettings | None = None, **callbacks) -> None:
    device.start(
        settings or RspDuoSettings(),
        FREQ_5C,
        on_gain=callbacks.get("on_gain", lambda g: None),
        on_overload=callbacks.get("on_overload", lambda o: None),
        on_failure=callbacks.get("on_failure", lambda m: None),
    )


@pytest.mark.parametrize(("tuner", "api_tuner"), [(1, sp.TUNER_A), (2, sp.TUNER_B)])
def test_open_claims_the_chosen_tuner_in_single_tuner_mode(tuner, api_tuner):
    api = FakeApi()
    _open(api, tuner)
    assert api.selected.tuner == api_tuner
    assert api.selected.rspDuoMode == sp.RSPDUO_MODE_SINGLE_TUNER
    assert not api.locked


def test_open_reports_serial_and_api_version():
    api = FakeApi()
    info = RspDuo(api_factory=lambda: api).open(1)
    assert info.serial == "2301007634"
    assert info.api_version == 3.15


def test_there_is_no_tuner_3():
    with pytest.raises(ValueError):
        RspDuo(api_factory=FakeApi).open(3)


@pytest.mark.parametrize("tuner", [1, 2])
def test_start_configures_the_tuner_for_dab(tuner):
    api = FakeApi()
    device = _open(api, tuner)
    # As another program might leave them in the service.
    api.channel.rspDuoTunerParams.biasTEnable = 1
    api.channel.rspDuoTunerParams.rfDabNotchEnable = 1
    settings = RspDuoSettings(
        tuner=tuner, lna_state=3, if_gain_reduction_db=45, freq_correction_ppm=-0.61
    )
    _start(device, settings)

    tuner_params = api.channel.tunerParams
    assert api._dev_params.fsFreq.fsHz == SAMPLE_RATE
    assert api._dev_params.ppm == -0.61
    assert api._dev_params.mode == rspduo_module.USB_TRANSFER_MODE
    assert tuner_params.rfFreq.rfHz == FREQ_5C
    assert tuner_params.bwType == sp.BW_1_536
    assert tuner_params.ifType == sp.IF_ZERO
    assert tuner_params.gain.LNAstate == 3
    assert tuner_params.gain.gRdB == 45
    assert api.channel.ctrlParams.agc.enable == sp.AGC_DISABLE
    assert api.channel.ctrlParams.decimation.enable == 0
    assert api.channel.rspDuoTunerParams.biasTEnable == 0
    assert api.channel.rspDuoTunerParams.rfDabNotchEnable == 0


def test_the_lna_table_has_ten_states_for_band_iii():
    assert len(rspduo_module.LNA_GAIN_REDUCTION_DB) == 10
    assert rspduo_module.LNA_GAIN_REDUCTION_DB[0] == 0
    assert list(rspduo_module.LNA_GAIN_REDUCTION_DB) == sorted(rspduo_module.LNA_GAIN_REDUCTION_DB)


def test_stream_callback_fills_the_ring_and_drain_hands_the_packets_over():
    api = FakeApi()
    device = _open(api)
    _start(device)
    assert api.context is not None  # the ring, passed back with every callback
    packets = []

    def sink(samples, num, reset, changed):
        packets.append((samples.copy(), num, reset, changed))

    api.push(4242, [1, 2, 3, 4], reset=True)
    assert device.drain(sink) == 0  # the newest packet waits for the next
    api.push(4246, [5, 6, 7, 8], rf_changed=True)
    api.push(4250, [9])
    assert device.drain(sink) == 2
    assert device.packets_written == 3

    (samples, num, reset, changed), flagged = packets
    np.testing.assert_array_equal(samples, [1, -1, 2, -2, 3, -3, 4, -4])
    assert (num, reset, changed) == (4242, True, 0)
    assert flagged[3] == FREQUENCY_CHANGED
    assert device.overflows == 0
    device.close()
    assert device.drain(sink) == 0  # the ring is gone with the stream


def test_events_report_gain_and_acknowledge_overload():
    api = FakeApi()
    device = _open(api, 2)
    gains, overloads, failures = [], [], []
    _start(device, on_gain=gains.append, on_overload=overloads.append, on_failure=failures.append)

    params = sp.EventParamsT()
    params.gainParams = sp.GainCbParamT(40, 20, 50.7)
    api.callbacks.EventCbFn(sp.EVENT_GAIN_CHANGE, sp.TUNER_B, ctypes.pointer(params), None)
    params.powerOverloadParams.powerOverloadChangeType = sp.OVERLOAD_DETECTED
    api.callbacks.EventCbFn(
        sp.EVENT_POWER_OVERLOAD_CHANGE, sp.TUNER_B, ctypes.pointer(params), None
    )
    api.callbacks.EventCbFn(sp.EVENT_DEVICE_REMOVED, sp.TUNER_B, ctypes.pointer(params), None)

    assert gains == [GainReport(40, 20, pytest.approx(50.7))]
    assert overloads == [True]
    assert (sp.TUNER_B, sp.UPDATE_CTRL_OVERLOAD_MSG_ACK) in api.updates
    assert failures == ["The RSPduo was disconnected from USB."]


@pytest.mark.parametrize(("tuner", "api_tuner"), [(1, sp.TUNER_A), (2, sp.TUNER_B)])
def test_retuning_is_sent_to_the_chosen_tuner(tuner, api_tuner):
    api = FakeApi()
    device = _open(api, tuner)
    _start(device)
    device.set_frequency(227_360_000.0)
    assert api.updates == [(api_tuner, sp.UPDATE_TUNER_FRF)]
    assert api.channel.tunerParams.rfFreq.rfHz == 227_360_000.0


def test_a_retune_the_api_refuses_is_reported():
    api = FakeApi(update_err=sp.OUT_OF_RANGE)
    device = _open(api)
    _start(device)
    with pytest.raises(SdrplayError, match="frequency not applied"):
        device.set_frequency(227_360_000.0)


def test_close_releases_in_order_and_is_idempotent():
    api = FakeApi()
    device = _open(api)
    _start(device)
    device.close()
    device.close()
    assert api.calls == ["open", "select", "init", "uninit", "release", "close"]


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (ServiceState.STOPPED, "is not running"),
        (ServiceState.NOT_INSTALLED, "is not installed"),
        (ServiceState.RUNNING, "restart the service"),
    ],
)
def test_open_failure_names_the_service_state(monkeypatch, state, message):
    monkeypatch.setattr(rspduo_module, "query_service_state", lambda: state)
    api = FakeApi(open_err=sp.FAIL)
    with pytest.raises(SdrplayError, match=message):
        RspDuo(api_factory=lambda: api).open(1)
    assert "close" not in api.calls  # never opened


def test_other_api_version_is_refused():
    api = FakeApi(version=3.16)
    with pytest.raises(SdrplayError, match="3.16 found"):
        RspDuo(api_factory=lambda: api).open(1)
    assert api.calls[-1] == "close"


def test_no_device():
    api = FakeApi(devices=[])
    with pytest.raises(SdrplayError, match="No free SDRplay device"):
        RspDuo(api_factory=lambda: api).open(1)
    assert not api.locked


def test_other_rsp_model():
    api = FakeApi(devices=[make_device(hw=255)])  # RSP1A
    with pytest.raises(SdrplayError, match="No RSPduo"):
        RspDuo(api_factory=lambda: api).open(1)


def test_a_tuner_another_program_leaves_as_slave_is_busy():
    # Another program holds tuner B as master; tuner A is offered only as its slave.
    api = FakeApi(devices=[make_device(tuner=sp.TUNER_A, modes=sp.RSPDUO_MODE_SLAVE)])
    with pytest.raises(SdrplayError, match="is busy"):
        RspDuo(api_factory=lambda: api).open(1)
    assert "select" not in api.calls


def test_select_failure_is_reported_as_busy():
    api = FakeApi(select_err=sp.FAIL)
    with pytest.raises(SdrplayError, match="is busy"):
        RspDuo(api_factory=lambda: api).open(2)
    assert not api.locked
    assert api.calls[-1] == "close"
