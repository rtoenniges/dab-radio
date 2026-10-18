"""One tuner of the SDRplay RSPduo, streaming through the SDRplay API.

Either tuner can receive Band III. The chosen one runs in single tuner mode, which
claims the whole device: while the radio receives, no other program gets the second
tuner. Both tuners together would need master/slave mode (one program per tuner, 6 or
8 MS/s on a low IF) and are not supported.

The radio needs no bias-T and switches it off.
"""

from __future__ import annotations

import atexit
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass

from dab.constants import SAMPLE_RATE
from i18n import tr, tr_noop
from sdr import sdrplay_api as sp
from sdr.exceptions import SdrplayError
from sdr.packet_ring import PacketRing, PacketSink
from sdr.service_state import ServiceState, query_service_state

logger = logging.getLogger(__name__)

# The RSPduo's tuners, as the RSPduo labels them, and as the API calls them.
TUNERS: dict[int, int] = {1: sp.TUNER_A, 2: sp.TUNER_B}

# IF bandwidth at the DAB sample rate of 2.048 MS/s with zero IF: the API's 1.536 MHz
# filter is exactly the width of a DAB ensemble.
BANDWIDTH_KHZ = sp.BW_1_536

# Gain reduction of the LNA states from 60 to 420 MHz (Band III: 174 to 240 MHz), from
# the gain reduction table of the RSPduo in the SDRplay API specification; the API's
# gain event reports exactly these on both tuners.
LNA_GAIN_REDUCTION_DB = (0, 6, 12, 18, 20, 26, 32, 38, 57, 62)

# IF gain reduction the API accepts in normal mode (sdrplay_api_NORMAL_MIN_GR .. MAX_BB_GR).
IF_GAIN_REDUCTION_RANGE_DB = (20, 59)

# Fixed gain, no AGC, as with the RTL-SDR (see `DEFAULT_GAIN_DB` in
# `ui.receiver_controller`): a control loop would move the level within the frame. The
# values are not yet tuned for a Band III antenna. Without an antenna the noise lies at
# about -43 dBFS, 10 dB above the floor of the stages after the LNA (-53.5 dBFS from LNA
# state 4 on), and the level follows the IF reduction dB for dB from 20 to 50 dB.
DEFAULT_LNA_STATE = 0
DEFAULT_IF_GAIN_REDUCTION_DB = 40

# Range of the frequency correction offered, in ppm (`devParams.ppm` of the API, a
# double, set before `Init`). The RSPduo's TCXO is specified to 0.5 ppm. The API
# corrects mixer and sample clock alike, and a positive frequency error needs a
# negative correction.
FREQ_CORRECTION_RANGE_PPM = (-10.0, 10.0)

# USB transfer mode (`devParams.mode`, set before `Init`). Isochronous, the API's
# default, reserves bandwidth but never repeats a transfer; bulk repeats it. On Linux
# (in a VM at least) isochronous delivers packets out of order and stale ones from the
# start of the stream again, bulk none; on Windows isochronous loses nothing and stays.
USB_TRANSFER_MODE = sp.TRANSFER_ISOCH if sys.platform == "win32" else sp.TRANSFER_BULK


@dataclass(frozen=True)
class RspDuoSettings:
    tuner: int = 1  # 1 or 2, as labelled on the RSPduo
    lna_state: int = DEFAULT_LNA_STATE
    if_gain_reduction_db: int = DEFAULT_IF_GAIN_REDUCTION_DB
    freq_correction_ppm: float = 0.0


@dataclass(frozen=True)
class GainReport:
    """Gain as the API reports it after every change."""

    if_gain_reduction_db: int
    lna_gain_reduction_db: int
    system_gain_db: float  # the API's estimate of the total gain from antenna port to ADC


@dataclass(frozen=True)
class DeviceInfo:
    serial: str
    api_version: float


# The messages when the API service cannot be reached, with how the user starts or
# restarts it; translated when shown.
if sys.platform == "win32":
    _SERVICE_STOPPED = tr_noop(
        "The “SDRplay API Service” is not running. Please start it in the Services "
        "console (services.msc) or restart the computer."
    )
    _SERVICE_UNREACHABLE = tr_noop(
        "No connection to the “SDRplay API Service” ({error}). Please restart the "
        "service in the Services console (services.msc)."
    )
else:
    _SERVICE_STOPPED = tr_noop(
        "The “SDRplay API Service” is not running. Please start it with "
        "“sudo systemctl start sdrplay”."
    )
    _SERVICE_UNREACHABLE = tr_noop(
        "No connection to the “SDRplay API Service” ({error}). Please restart the "
        "service with “sudo systemctl restart sdrplay”."
    )

_BUSY_MESSAGE = tr_noop(
    "The RSPduo is busy. Is SDRuno, SDRconnect or another SDRplay program using it? "
    "Please release the device there or quit the program."
)


class RspDuo:
    """Opens one tuner of an RSPduo, configures it for DAB and streams its samples.

    Lifecycle: `open()`, `start()`, any number of `set_frequency` calls, `close()`.
    `close()` is safe to call at any point and more than once. Every error the user can
    cause or fix (service not running, no device, device busy) raises `SdrplayError`
    with a message for the user.

    The samples do not pass through Python on the API's thread: its stream callback is
    the compiled C function of `sdr.packet_ring`, which needs no GIL. The caller fetches
    the packets with `drain()` from a thread of its own, as often as it likes.
    """

    def __init__(self, api_factory: Callable[[], sp.SdrplayApi] = sp.SdrplayApi) -> None:
        self._api_factory = api_factory
        self._api: sp.SdrplayApi | None = None
        self._api_opened = False
        self._device: sp.DeviceT | None = None
        self._handle: int | None = None
        self._tuner = sp.TUNER_A
        self._device_params: sp.DeviceParamsT | None = None
        self._channel: sp.RxChannelParamsT | None = None
        self._streaming = False
        # The ctypes callback objects and the ring must live as long as the API may call.
        self._callbacks: sp.CallbackFnsT | None = None
        self._ring: PacketRing | None = None
        self._on_gain: Callable[[GainReport], None] | None = None
        self._on_overload: Callable[[bool], None] | None = None
        self._on_failure: Callable[[str], None] | None = None

    def open(self, tuner: int) -> DeviceInfo:
        """Connects to the API service and claims `tuner` (1 or 2) of the first RSPduo."""
        if tuner not in TUNERS:
            raise ValueError(f"The RSPduo has tuners 1 and 2, not {tuner}")
        self._tuner = TUNERS[tuner]
        api = self._api_factory()
        self._api = api
        err = api.open()
        if err != sp.SUCCESS:
            raise SdrplayError(_describe_open_failure(api, err))
        self._api_opened = True

        err, version = api.api_version()
        if err != sp.SUCCESS:
            self.close()
            raise SdrplayError(
                tr(
                    "SDRplay API: cannot read the version ({error}). Do the service and the "
                    "API library match? Reinstalling usually helps."
                ).format(error=api.get_error_string(err))
            )
        if version != sp.SUPPORTED_API_VERSION:
            self.close()
            raise SdrplayError(
                tr(
                    "SDRplay API {found} found, this program is written for {supported}. "
                    "Please install the “Hardware API” {supported}."
                ).format(found=f"{version:.2f}", supported=f"{sp.SUPPORTED_API_VERSION:.2f}")
            )

        api.lock_device_api()
        try:
            err, devices = api.get_devices()
            if err != sp.SUCCESS:
                raise SdrplayError(
                    tr("Could not query the SDRplay devices: {error}").format(
                        error=api.get_error_string(err)
                    )
                )
            device = _choose_rspduo(devices, self._tuner)
            device.tuner = self._tuner
            device.rspDuoMode = sp.RSPDUO_MODE_SINGLE_TUNER
            err = api.select_device(device)
            if err != sp.SUCCESS:
                detail = api.get_last_error(device)
                logger.warning("SelectDevice failed: %s (%s)", api.get_error_string(err), detail)
                raise SdrplayError(tr(_BUSY_MESSAGE))
        except SdrplayError:
            api.unlock_device_api()
            self.close()
            raise
        api.unlock_device_api()
        self._device = device
        self._handle = device.dev

        err, params = api.get_device_params(self._handle)
        if err != sp.SUCCESS or params is None:
            self.close()
            raise SdrplayError(
                tr("RSPduo: cannot read the parameters ({error}).").format(
                    error=api.get_error_string(err)
                )
            )
        # In single tuner mode the API fills in the channel of the chosen tuner only; the
        # samples arrive through the callback for stream A either way.
        channel = params.rxChannelA if self._tuner == sp.TUNER_A else params.rxChannelB
        if not channel:
            self.close()
            raise SdrplayError(tr("RSPduo: tuner {tuner} is not available.").format(tuner=tuner))
        self._device_params = params
        self._channel = channel.contents
        return DeviceInfo(serial=device.SerNo.decode("ascii", "replace"), api_version=version)

    def start(
        self,
        settings: RspDuoSettings,
        center_freq_hz: float,
        on_gain: Callable[[GainReport], None],
        on_overload: Callable[[bool], None],
        on_failure: Callable[[str], None],
    ) -> None:
        """Configures the tuner for DAB and starts streaming into the ring (see `drain()`).

        The event callbacks run in the API's thread; `on_failure` reports a removed or
        failed device with a message for the user.
        """
        if self._api is None or self._channel is None or self._handle is None:
            raise RuntimeError("open() must succeed before start()")
        self._on_gain = on_gain
        self._on_overload = on_overload
        self._on_failure = on_failure

        dev_params = self._device_params.devParams.contents
        dev_params.fsFreq.fsHz = float(SAMPLE_RATE)
        dev_params.ppm = settings.freq_correction_ppm
        dev_params.mode = USB_TRANSFER_MODE
        tuner = self._channel.tunerParams
        tuner.rfFreq.rfHz = center_freq_hz
        tuner.bwType = BANDWIDTH_KHZ
        tuner.ifType = sp.IF_ZERO
        tuner.gain.LNAstate = settings.lna_state
        tuner.gain.gRdB = settings.if_gain_reduction_db
        self._channel.ctrlParams.agc.enable = sp.AGC_DISABLE
        self._channel.ctrlParams.decimation.enable = 0
        duo = self._channel.rspDuoTunerParams
        duo.biasTEnable = 0
        # The DAB notch suppresses Band III, the whole point here. Off by default; set
        # anyway, since another program may leave the service with other values.
        duo.rfDabNotchEnable = 0
        duo.rfNotchEnable = 0

        self._ring = PacketRing(float(SAMPLE_RATE))
        self._callbacks = sp.CallbackFnsT(
            sp.StreamCallback(self._ring.callback_address),
            sp.StreamCallback(self._ring.callback_address),
            sp.EventCallback(self._event_callback),
        )
        err = self._api.init(self._handle, self._callbacks, self._ring.context)
        if err != sp.SUCCESS:
            detail = self._api.get_last_error(self._device)
            error = self._api.get_error_string(err) + (f" ({detail})" if detail else "")
            raise SdrplayError(tr("Could not start the RSPduo: {error}").format(error=error))
        self._streaming = True
        # Should the program end without close() (an uncaught exception), the API would
        # keep calling the stream callback into a Python that is shutting down, and the
        # service would keep the device claimed.
        atexit.register(self.close)

    def drain(self, sink: PacketSink) -> int:
        """Hands the packets received so far to `sink(samples, first_sample_num, reset,
        changed)`, in order, from the caller's thread; returns how many. `samples` (int16,
        I and Q interleaved) are valid during the call only; `changed` carries the API's
        flags for a change of gain, frequency or rate that took effect with the packet.
        May be called from any one thread, as often as wanted: the ring holds
        `sdr.packet_ring.RING_SECONDS` of packets."""
        if self._ring is None:
            return 0
        return self._ring.drain(sink)

    @property
    def overflows(self) -> int:
        """Packets the ring dropped because `drain()` came too late."""
        return self._ring.overflows if self._ring is not None else 0

    @property
    def packets_written(self) -> int:
        """Packets the ring has taken so far; `drain()` hands them over in this order."""
        return self._ring.written if self._ring is not None else 0

    def set_frequency(self, center_freq_hz: float) -> None:
        self._require_streaming()
        self._channel.tunerParams.rfFreq.rfHz = center_freq_hz
        # Translators: the setting in "RSPduo: {setting} not applied ({error}).".
        self._update(sp.UPDATE_TUNER_FRF, tr("frequency"))

    def close(self) -> None:
        """Stops streaming, releases the device and disconnects from the service."""
        api = self._api
        if api is None:
            return
        if self._streaming and self._handle is not None:
            self._streaming = False
            atexit.unregister(self.close)
            err = api.uninit(self._handle)
            if err != sp.SUCCESS:
                logger.warning("Uninit failed: %s", api.get_error_string(err))
            self._ring = None  # after Uninit the API calls no more
        if self._device is not None:
            err = api.release_device(self._device)
            if err != sp.SUCCESS:
                logger.warning("ReleaseDevice failed: %s", api.get_error_string(err))
            self._device = None
            self._handle = None
            self._device_params = None
            self._channel = None
        if self._api_opened:
            self._api_opened = False
            api.close()
        self._api = None

    def _require_streaming(self) -> None:
        if not self._streaming or self._channel is None:
            raise RuntimeError("the device is not streaming")

    def _update(self, reason: int, what: str) -> None:
        err = self._api.update(self._handle, self._tuner, reason, sp.UPDATE_EXT1_NONE)
        if err != sp.SUCCESS:
            raise SdrplayError(
                tr("RSPduo: {setting} not applied ({error}).").format(
                    setting=what, error=self._api.get_error_string(err)
                )
            )

    def _event_callback(self, event_id, tuner, params, _context) -> None:
        # Exceptions must not escape into the API's thread; ctypes would only print them.
        try:
            if event_id == sp.EVENT_GAIN_CHANGE:
                gain = params.contents.gainParams
                self._on_gain(GainReport(gain.gRdB, gain.lnaGRdB, gain.currGain))
            elif event_id == sp.EVENT_POWER_OVERLOAD_CHANGE:
                change = params.contents.powerOverloadParams.powerOverloadChangeType
                # The API keeps reporting until the message is acknowledged. Done here in
                # the API's thread, as in the example of the API specification; no lock
                # against `_update` from the caller's thread, which could deadlock if the
                # API waited for this callback inside `Update`.
                self._api.update(
                    self._handle, tuner, sp.UPDATE_CTRL_OVERLOAD_MSG_ACK, sp.UPDATE_EXT1_NONE
                )
                self._on_overload(change == sp.OVERLOAD_DETECTED)
            elif event_id == sp.EVENT_DEVICE_REMOVED:
                self._on_failure(tr("The RSPduo was disconnected from USB."))
            elif event_id == sp.EVENT_DEVICE_FAILURE:
                self._on_failure(
                    tr("The RSPduo reports a device failure (check the USB connection).")
                )
        except Exception:
            logger.exception("Error in the event callback")


def _choose_rspduo(devices: list[sp.DeviceT], tuner: int) -> sp.DeviceT:
    """The first RSPduo whose `tuner` (the API's TUNER_A or TUNER_B) can be used in
    single tuner mode."""
    if not devices:
        raise SdrplayError(
            tr(
                "No free SDRplay device found. Is the RSPduo connected? Is SDRuno or "
                "SDRconnect using it?"
            )
        )
    duos = [device for device in devices if device.hwVer == sp.RSPDUO_HW_VERSION]
    if not duos:
        raise SdrplayError(tr("No RSPduo found, only other SDRplay devices."))
    for device in duos:
        # While another program uses one tuner, the other is offered only in a reduced
        # mode (slave to that program's master).
        if device.tuner & tuner and device.rspDuoMode & sp.RSPDUO_MODE_SINGLE_TUNER:
            return device
    raise SdrplayError(tr(_BUSY_MESSAGE))


def _describe_open_failure(api: sp.SdrplayApi, err: int) -> str:
    """Why `sdrplay_api_Open` failed, as far as the service's state tells."""
    state = query_service_state()
    if state is ServiceState.NOT_INSTALLED:
        return tr(
            "The “SDRplay API Service” is not installed. Please install the "
            "“Hardware API” 3.15 from sdrplay.com."
        )
    if state in (ServiceState.STOPPED, ServiceState.PAUSED):
        return tr(_SERVICE_STOPPED)
    if state is ServiceState.PENDING:
        return tr("The “SDRplay API Service” is starting. Please try again in a moment.")
    return tr(_SERVICE_UNREACHABLE).format(error=api.get_error_string(err))
