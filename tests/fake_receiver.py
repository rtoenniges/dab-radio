"""A receiver without a device for the tests of `ui.receiver_controller` and the surfaces
on top of it: an RTL-SDR worker that only records what it is asked to do, and stored
service lists to start from."""

from __future__ import annotations

from typing import ClassVar

from PyQt6.QtCore import QObject, pyqtSignal

ENSEMBLE_5C = "DR Deutschland"
SERVICES_5C = ((0xD210, "Dlf"), (0xD220, "Dlf Kultur"))
SERVICES_11D = ((0xD311, "NDR 2"),)


class FakeRtlSdrWorker(QObject):
    """Stands in for `sdr.worker.RtlSdrWorker`: the same signals, no device. The test
    emits `connected`, `error` and `stopped` itself."""

    samples_ready = pyqtSignal(object)
    tuned = pyqtSignal(int)
    connected = pyqtSignal(object)
    error = pyqtSignal(object)
    stopped = pyqtSignal()

    instances: ClassVar[list[FakeRtlSdrWorker]] = []

    def __init__(self, center_freq, sample_rate, gain, freq_correction_ppm) -> None:
        super().__init__()
        self.center_freq = center_freq
        self.gain = gain
        self.freq_correction_ppm = freq_correction_ppm
        self.tuning = 100
        self.retuned_to: list[float] = []
        self.started = False
        self.stop_requested = False
        FakeRtlSdrWorker.instances.append(self)

    def start(self) -> None:
        self.started = True

    def retune(self, center_freq: float) -> int:
        self.retuned_to.append(center_freq)
        self.tuning += 1
        return self.tuning

    def stop(self) -> None:
        self.stop_requested = True


class FakeRspDuoWorker(FakeRtlSdrWorker):
    """Stands in for `sdr.worker.RspDuoWorker`, among the instances of
    `FakeRtlSdrWorker`; `rspduo` holds the tuner and gain it was given."""

    overload_changed = pyqtSignal(bool)

    def __init__(self, center_freq, settings) -> None:
        super().__init__(center_freq, None, None, None)
        self.rspduo = settings


class FakeRtlTcpWorker(FakeRtlSdrWorker):
    """Stands in for `sdr.worker.RtlTcpWorker`, among the instances of
    `FakeRtlSdrWorker`; `rtl_tcp` holds the server, gain and correction it was given."""

    def __init__(self, center_freq, settings) -> None:
        super().__init__(center_freq, None, settings.gain, settings.freq_correction_ppm)
        self.rtl_tcp = settings


class FakeChannelScanner(QObject):
    """Stands in for `sdr.scanner.ChannelScanner`: the same signals, no device. The test
    reports channels and the end itself."""

    channel_started = pyqtSignal(str, int, int)
    channel_progress = pyqtSignal(str, int, int, object, int)
    channel_scanned = pyqtSignal(str, object)
    spectrum_ready = pyqtSignal(object, object)
    error = pyqtSignal(object)
    scan_finished = pyqtSignal(bool)

    instances: ClassVar[list[FakeChannelScanner]] = []

    def __init__(self, channels, sample_rate, gain, freq_correction_ppm, rspduo, rtl_tcp) -> None:
        super().__init__()
        self.rspduo = rspduo
        self.rtl_tcp = rtl_tcp
        self.started = False
        self.cancelled = False
        FakeChannelScanner.instances.append(self)

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def stop(self) -> None:
        self.cancelled = True

    def wait(self, _timeout: int = 0) -> bool:
        return True
