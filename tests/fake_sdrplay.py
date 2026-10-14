"""A fake of the SDRplay API for tests: behaves like sdrplay_api.dll as measured on the
RSPduo in the GNSS receiver, with switchable faults and packets pushed by the test.

Taken over from the GNSS receiver; here either tuner can be selected, and the test
pushes the packets itself instead of a streaming thread.
"""

import ctypes

import numpy as np

from sdr import sdrplay_api as sp

PACKET = 1344  # samples per stream callback at 2.048 MS/s, as measured

HANDLE = 0x1234


def make_device(hw=sp.RSPDUO_HW_VERSION, tuner=sp.TUNER_BOTH, modes=7, serial=b"2301007634"):
    device = sp.DeviceT()
    device.SerNo = serial
    device.hwVer = hw
    device.tuner = tuner
    device.rspDuoMode = modes
    device.valid = 1
    return device


class FakeApi:
    """Behaves like sdrplay_api.dll as measured on the RSPduo, with switchable faults."""

    def __init__(
        self,
        devices=None,
        open_err=sp.SUCCESS,
        version=3.15,
        select_err=sp.SUCCESS,
        update_err=sp.SUCCESS,
    ):
        self.devices = [make_device()] if devices is None else devices
        self.open_err = open_err
        self.version = version
        self.select_err = select_err
        self.update_err = update_err
        self.calls: list[str] = []
        self.updates: list[tuple[int, int]] = []
        self.selected: sp.DeviceT | None = None
        self.callbacks: sp.CallbackFnsT | None = None
        self.context: int | None = None
        self.locked = False
        # Called inside `update`, before it returns: packets the device delivers while
        # the API applies a change.
        self.on_update = None
        # Parameter memory as the API owns it; in single tuner mode only the channel of
        # the selected tuner is filled in.
        self._dev_params = sp.DevParamsT()
        self._channel_a = sp.RxChannelParamsT()
        self._channel_b = sp.RxChannelParamsT()
        self._params = sp.DeviceParamsT()

    def open(self):
        self.calls.append("open")
        return self.open_err

    def close(self):
        self.calls.append("close")
        return sp.SUCCESS

    def api_version(self):
        return sp.SUCCESS, self.version

    def lock_device_api(self):
        self.locked = True
        return sp.SUCCESS

    def unlock_device_api(self):
        self.locked = False
        return sp.SUCCESS

    def get_devices(self):
        return sp.SUCCESS, [sp.DeviceT.from_buffer_copy(d) for d in self.devices]

    def select_device(self, device):
        self.calls.append("select")
        if self.select_err == sp.SUCCESS:
            device.dev = HANDLE
            self.selected = device
        return self.select_err

    def release_device(self, device):
        self.calls.append("release")
        return sp.SUCCESS

    def get_error_string(self, err):
        return f"err{err}"

    def get_last_error(self, device):
        return ""

    def get_device_params(self, handle):
        assert handle == HANDLE
        self._params.devParams = ctypes.pointer(self._dev_params)
        if self.selected.tuner == sp.TUNER_A:
            self._params.rxChannelA = ctypes.pointer(self._channel_a)
        else:
            self._params.rxChannelB = ctypes.pointer(self._channel_b)
        return sp.SUCCESS, self._params

    def init(self, handle, callbacks, context=None):
        self.calls.append("init")
        self.callbacks = callbacks
        self.context = context
        return sp.SUCCESS

    def uninit(self, handle):
        self.calls.append("uninit")
        return sp.SUCCESS

    def update(self, handle, tuner, reason, reason_ext1=0):
        self.updates.append((tuner, reason))
        if self.on_update is not None:
            self.on_update()
        return self.update_err

    def push(self, first_sample_num, i, q=None, reset=False, rf_changed=False):
        """Calls the stream callback as the API would, with I (and Q, by default -I);
        `rf_changed` flags the packet with which a new frequency took effect."""
        i = np.asarray(i, np.int16)
        q = -i if q is None else np.asarray(q, np.int16)
        n = len(i)
        xi = (ctypes.c_short * n).from_buffer_copy(i.tobytes())
        xq = (ctypes.c_short * n).from_buffer_copy(q.tobytes())
        params = sp.StreamCbParamsT(
            firstSampleNum=first_sample_num & 0xFFFFFFFF, numSamples=n, rfChanged=int(rf_changed)
        )
        self.callbacks.StreamACbFn(xi, xq, ctypes.pointer(params), n, int(reset), self.context)

    @property
    def channel(self) -> sp.RxChannelParamsT:
        """The channel of the selected tuner."""
        return self._channel_a if self.selected.tuner == sp.TUNER_A else self._channel_b
