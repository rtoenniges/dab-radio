"""ctypes binding of the SDRplay API 3 (`sdrplay_api.dll`, on Linux `libsdrplay_api.so.3`).

SDRplay's website now calls it the "Hardware API", as opposed to the WebSocket API of
SDRconnect; it is the same library, still version 3.15. Unlike librtlsdr it does not
talk to the USB device itself but to the "SDRplay API Service", which owns all RSPs and
hands out one tuner at a time.

The structures mirror the C headers of API 3.15 (`<Install_Dir>/inc/sdrplay_api*.h`,
on Linux `/usr/local/include`) field by field, including those of other RSP models:
`sdrplay_api_GetDeviceParams` returns pointers into the service's own memory, and a
field in the wrong place would write into a neighbouring parameter. Sizes and offsets
are checked against what MSVC reports for the original headers
(tests/test_sdrplay_api.py); gcc on Linux x64 reports the same, since the headers use
no `long`. The headers declare no `#pragma pack`, so ctypes' natural alignment matches,
and enums are 4-byte ints.

Error codes are returned, not raised: what a code means for the user depends on the
call (`sdrplay_api_Fail` from `Open` means the service is not running, from
`SelectDevice` that another program holds the device), so `sdr.rspduo` translates them.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    Union,
    byref,
    c_char,
    c_char_p,
    c_double,
    c_float,
    c_int,
    c_short,
    c_ubyte,
    c_uint,
    c_ushort,
    c_void_p,
)
from pathlib import Path

from i18n import tr
from sdr.exceptions import SdrplayError

# The layouts below are those of this version. Another version may move fields, so it is
# refused until the layout test has been rerun against its headers.
SUPPORTED_API_VERSION = 3.15

# Default install location on Windows; the installer records the actual one in the
# registry.
DEFAULT_INSTALL_DIR = Path(r"C:\Program Files\SDRplay\API")
_REGISTRY_KEY = r"SOFTWARE\SDRplay\Service\API"
# Where the installer of API 3.15 for Linux puts the library (the service goes to
# /opt/sdrplay_api).
LINUX_LIBRARY = Path("/usr/local/lib/libsdrplay_api.so.3")

MAX_DEVICES = 16
RSPDUO_HW_VERSION = 3  # SDRPLAY_RSPduo_ID

# sdrplay_api_ErrT
SUCCESS = 0
FAIL = 1
INVALID_PARAM = 2
OUT_OF_RANGE = 3
ALREADY_INITIALISED = 9
NOT_INITIALISED = 10
SERVICE_NOT_RESPONDING = 14
INVALID_SERVICE_VERSION = 24

# sdrplay_api_TunerSelectT; tuner A is labelled "Tuner 1" on the RSPduo, B "Tuner 2".
TUNER_NEITHER = 0
TUNER_A = 1
TUNER_B = 2
TUNER_BOTH = 3

# sdrplay_api_RspDuoModeT (in `DeviceT.rspDuoMode` of `GetDevices` a bit mask of the
# modes still available)
RSPDUO_MODE_UNKNOWN = 0
RSPDUO_MODE_SINGLE_TUNER = 1
RSPDUO_MODE_DUAL_TUNER = 2
RSPDUO_MODE_MASTER = 4
RSPDUO_MODE_SLAVE = 8

# sdrplay_api_Bw_MHzT (IF bandwidth in kHz)
BW_0_200 = 200
BW_0_300 = 300
BW_0_600 = 600
BW_1_536 = 1536
BW_5_000 = 5000
BW_6_000 = 6000
BW_7_000 = 7000
BW_8_000 = 8000

# sdrplay_api_If_kHzT
IF_ZERO = 0

# sdrplay_api_AgcControlT
AGC_DISABLE = 0

# sdrplay_api_TransferModeT (USB)
TRANSFER_ISOCH = 0
TRANSFER_BULK = 1

# sdrplay_api_ReasonForUpdateT
UPDATE_NONE = 0x00000000
UPDATE_DEV_FS = 0x00000001
UPDATE_TUNER_GR = 0x00008000
UPDATE_TUNER_FRF = 0x00020000
UPDATE_CTRL_AGC = 0x01000000
UPDATE_CTRL_OVERLOAD_MSG_ACK = 0x04000000
UPDATE_RSPDUO_BIAS_T_CONTROL = 0x08000000
# sdrplay_api_ReasonForUpdateExtension1T
UPDATE_EXT1_NONE = 0x00000000

# sdrplay_api_EventT
EVENT_GAIN_CHANGE = 0
EVENT_POWER_OVERLOAD_CHANGE = 1
EVENT_DEVICE_REMOVED = 2
EVENT_RSPDUO_MODE_CHANGE = 3
EVENT_DEVICE_FAILURE = 4

# sdrplay_api_PowerOverloadCbEventIdT
OVERLOAD_DETECTED = 0
OVERLOAD_CORRECTED = 1


class DeviceT(Structure):
    _fields_ = [
        ("SerNo", c_char * 64),
        ("hwVer", c_ubyte),
        ("tuner", c_int),
        ("rspDuoMode", c_int),
        ("valid", c_ubyte),
        ("rspDuoSampleFreq", c_double),
        ("dev", c_void_p),
    ]


class FsFreqT(Structure):
    _fields_ = [("fsHz", c_double), ("syncUpdate", c_ubyte), ("reCal", c_ubyte)]


class SyncUpdateT(Structure):
    _fields_ = [("sampleNum", c_uint), ("period", c_uint)]


class ResetFlagsT(Structure):
    _fields_ = [
        ("resetGainUpdate", c_ubyte),
        ("resetRfUpdate", c_ubyte),
        ("resetFsUpdate", c_ubyte),
    ]


class Rsp1aParamsT(Structure):
    _fields_ = [("rfNotchEnable", c_ubyte), ("rfDabNotchEnable", c_ubyte)]


class Rsp2ParamsT(Structure):
    _fields_ = [("extRefOutputEn", c_ubyte)]


class RspDuoParamsT(Structure):
    _fields_ = [("extRefOutputEn", c_int)]


class RspDxParamsT(Structure):
    _fields_ = [
        ("hdrEnable", c_ubyte),
        ("biasTEnable", c_ubyte),
        ("antennaSel", c_int),
        ("rfNotchEnable", c_ubyte),
        ("rfDabNotchEnable", c_ubyte),
    ]


class DevParamsT(Structure):
    _fields_ = [
        ("ppm", c_double),
        ("fsFreq", FsFreqT),
        ("syncUpdate", SyncUpdateT),
        ("resetFlags", ResetFlagsT),
        ("mode", c_int),
        ("samplesPerPkt", c_uint),
        ("rsp1aParams", Rsp1aParamsT),
        ("rsp2Params", Rsp2ParamsT),
        ("rspDuoParams", RspDuoParamsT),
        ("rspDxParams", RspDxParamsT),
    ]


class GainValuesT(Structure):
    _fields_ = [("curr", c_float), ("max", c_float), ("min", c_float)]


class GainT(Structure):
    _fields_ = [
        ("gRdB", c_int),
        ("LNAstate", c_ubyte),
        ("syncUpdate", c_ubyte),
        ("minGr", c_int),
        ("gainVals", GainValuesT),
    ]


class RfFreqT(Structure):
    _fields_ = [("rfHz", c_double), ("syncUpdate", c_ubyte)]


class DcOffsetTunerT(Structure):
    _fields_ = [
        ("dcCal", c_ubyte),
        ("speedUp", c_ubyte),
        ("trackTime", c_int),
        ("refreshRateTime", c_int),
    ]


class TunerParamsT(Structure):
    _fields_ = [
        ("bwType", c_int),
        ("ifType", c_int),
        ("loMode", c_int),
        ("gain", GainT),
        ("rfFreq", RfFreqT),
        ("dcOffsetTuner", DcOffsetTunerT),
    ]


class DcOffsetT(Structure):
    _fields_ = [("DCenable", c_ubyte), ("IQenable", c_ubyte)]


class DecimationT(Structure):
    _fields_ = [
        ("enable", c_ubyte),
        ("decimationFactor", c_ubyte),
        ("wideBandSignal", c_ubyte),
    ]


class AgcT(Structure):
    _fields_ = [
        ("enable", c_int),
        ("setPoint_dBfs", c_int),
        ("attack_ms", c_ushort),
        ("decay_ms", c_ushort),
        ("decay_delay_ms", c_ushort),
        ("decay_threshold_dB", c_ushort),
        ("syncUpdate", c_int),
    ]


class ControlParamsT(Structure):
    _fields_ = [
        ("dcOffset", DcOffsetT),
        ("decimation", DecimationT),
        ("agc", AgcT),
        ("adsbMode", c_int),
    ]


class Rsp1aTunerParamsT(Structure):
    _fields_ = [("biasTEnable", c_ubyte)]


class Rsp2TunerParamsT(Structure):
    _fields_ = [
        ("biasTEnable", c_ubyte),
        ("amPortSel", c_int),
        ("antennaSel", c_int),
        ("rfNotchEnable", c_ubyte),
    ]


class RspDuoResetSlaveFlagsT(Structure):
    _fields_ = [("resetGainUpdate", c_ubyte), ("resetRfUpdate", c_ubyte)]


class RspDuoTunerParamsT(Structure):
    _fields_ = [
        ("biasTEnable", c_ubyte),
        ("tuner1AmPortSel", c_int),
        ("tuner1AmNotchEnable", c_ubyte),
        ("rfNotchEnable", c_ubyte),
        ("rfDabNotchEnable", c_ubyte),
        ("resetSlaveFlags", RspDuoResetSlaveFlagsT),
    ]


class RspDxTunerParamsT(Structure):
    _fields_ = [("hdrBw", c_int)]


class RxChannelParamsT(Structure):
    _fields_ = [
        ("tunerParams", TunerParamsT),
        ("ctrlParams", ControlParamsT),
        ("rsp1aTunerParams", Rsp1aTunerParamsT),
        ("rsp2TunerParams", Rsp2TunerParamsT),
        ("rspDuoTunerParams", RspDuoTunerParamsT),
        ("rspDxTunerParams", RspDxTunerParamsT),
    ]


class DeviceParamsT(Structure):
    _fields_ = [
        ("devParams", POINTER(DevParamsT)),
        ("rxChannelA", POINTER(RxChannelParamsT)),
        ("rxChannelB", POINTER(RxChannelParamsT)),
    ]


class ErrorInfoT(Structure):
    _fields_ = [
        ("file", c_char * 256),
        ("function", c_char * 256),
        ("line", c_int),
        ("message", c_char * 1024),
    ]


class StreamCbParamsT(Structure):
    _fields_ = [
        ("firstSampleNum", c_uint),
        ("grChanged", c_int),
        ("rfChanged", c_int),
        ("fsChanged", c_int),
        ("numSamples", c_uint),
    ]


class GainCbParamT(Structure):
    _fields_ = [("gRdB", c_uint), ("lnaGRdB", c_uint), ("currGain", c_double)]


class PowerOverloadCbParamT(Structure):
    _fields_ = [("powerOverloadChangeType", c_int)]


class RspDuoModeCbParamT(Structure):
    _fields_ = [("modeChangeType", c_int)]


class EventParamsT(Union):
    _fields_ = [
        ("gainParams", GainCbParamT),
        ("powerOverloadParams", PowerOverloadCbParamT),
        ("rspDuoModeParams", RspDuoModeCbParamT),
    ]


# The headers declare no calling convention, i.e. cdecl (the only one on x64 anyway).
StreamCallback = CFUNCTYPE(
    None, POINTER(c_short), POINTER(c_short), POINTER(StreamCbParamsT), c_uint, c_uint, c_void_p
)
EventCallback = CFUNCTYPE(None, c_int, c_int, POINTER(EventParamsT), c_void_p)


class CallbackFnsT(Structure):
    _fields_ = [
        ("StreamACbFn", StreamCallback),
        ("StreamBCbFn", StreamCallback),
        ("EventCbFn", EventCallback),
    ]


def find_api_library() -> Path:
    """Path of the API library for this platform and Python's architecture.

    Windows: the installer records its directory under
    HKLM\\SOFTWARE\\SDRplay\\Service\\API (value `Install_Dir`); the DLLs lie in `x64`,
    `x86` and `arm64` below it. Linux: the installer puts the library into
    `/usr/local/lib`, which not every distribution's loader searches; a library
    installed elsewhere (a distribution package) is left to the loader.
    """
    if sys.platform != "win32":
        return LINUX_LIBRARY if LINUX_LIBRARY.exists() else Path(LINUX_LIBRARY.name)
    import winreg

    install_dir = DEFAULT_INSTALL_DIR
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _REGISTRY_KEY) as key:
            install_dir = Path(winreg.QueryValueEx(key, "Install_Dir")[0])
    except OSError:
        pass
    arch = "x64" if ctypes.sizeof(c_void_p) == 8 else "x86"
    return install_dir / arch / "sdrplay_api.dll"


class SdrplayApi:
    """The functions of the API library the receiver uses, with their C signatures.

    Thin on purpose: every method returns the API's error code (or the requested value),
    so that `sdr.rspduo.RspDuo` decides what an error means. Tests replace this class by
    a fake with the same methods.
    """

    def __init__(self, library_path: Path | None = None) -> None:
        path = library_path or find_api_library()
        try:
            self._dll = ctypes.CDLL(str(path))
        except OSError as exc:
            raise SdrplayError(
                tr(
                    "SDRplay API not found ({path}). Please install the “Hardware API” 3.15 "
                    "from sdrplay.com."
                ).format(path=path)
            ) from exc
        self._bind()

    def _bind(self) -> None:
        dll = self._dll
        signatures = {
            "sdrplay_api_Open": (c_int, []),
            "sdrplay_api_Close": (c_int, []),
            "sdrplay_api_ApiVersion": (c_int, [POINTER(c_float)]),
            "sdrplay_api_LockDeviceApi": (c_int, []),
            "sdrplay_api_UnlockDeviceApi": (c_int, []),
            "sdrplay_api_GetDevices": (c_int, [POINTER(DeviceT), POINTER(c_uint), c_uint]),
            "sdrplay_api_SelectDevice": (c_int, [POINTER(DeviceT)]),
            "sdrplay_api_ReleaseDevice": (c_int, [POINTER(DeviceT)]),
            "sdrplay_api_GetErrorString": (c_char_p, [c_int]),
            "sdrplay_api_GetLastError": (POINTER(ErrorInfoT), [POINTER(DeviceT)]),
            "sdrplay_api_GetDeviceParams": (c_int, [c_void_p, POINTER(POINTER(DeviceParamsT))]),
            "sdrplay_api_Init": (c_int, [c_void_p, POINTER(CallbackFnsT), c_void_p]),
            "sdrplay_api_Uninit": (c_int, [c_void_p]),
            "sdrplay_api_Update": (c_int, [c_void_p, c_int, c_uint, c_uint]),
        }
        for name, (restype, argtypes) in signatures.items():
            function = getattr(dll, name)
            function.restype = restype
            function.argtypes = argtypes

    def open(self) -> int:
        return self._dll.sdrplay_api_Open()

    def close(self) -> int:
        return self._dll.sdrplay_api_Close()

    def api_version(self) -> tuple[int, float]:
        version = c_float()
        err = self._dll.sdrplay_api_ApiVersion(byref(version))
        # Returned as a float (3.1500000953...), compared to two decimals.
        return err, round(version.value, 2)

    def lock_device_api(self) -> int:
        return self._dll.sdrplay_api_LockDeviceApi()

    def unlock_device_api(self) -> int:
        return self._dll.sdrplay_api_UnlockDeviceApi()

    def get_devices(self) -> tuple[int, list[DeviceT]]:
        devices = (DeviceT * MAX_DEVICES)()
        count = c_uint()
        err = self._dll.sdrplay_api_GetDevices(devices, byref(count), MAX_DEVICES)
        # Copies, so that the caller may keep and modify them (see `select_device`).
        return err, [DeviceT.from_buffer_copy(device) for device in devices[: count.value]]

    def select_device(self, device: DeviceT) -> int:
        """Claims the device; sets `device.dev`, the handle for all device calls.

        `device.tuner` and `device.rspDuoMode` choose tuner and mode of an RSPduo.
        """
        return self._dll.sdrplay_api_SelectDevice(byref(device))

    def release_device(self, device: DeviceT) -> int:
        return self._dll.sdrplay_api_ReleaseDevice(byref(device))

    def get_error_string(self, err: int) -> str:
        text = self._dll.sdrplay_api_GetErrorString(err)
        return text.decode("ascii", "replace") if text else f"error {err}"

    def get_last_error(self, device: DeviceT) -> str:
        """The API's detailed message for the last error on `device`, "" if none."""
        info = self._dll.sdrplay_api_GetLastError(byref(device))
        if not info:
            return ""
        return info.contents.message.decode("ascii", "replace").strip()

    def get_device_params(self, handle: int) -> tuple[int, DeviceParamsT | None]:
        params = POINTER(DeviceParamsT)()
        err = self._dll.sdrplay_api_GetDeviceParams(handle, byref(params))
        return err, (params.contents if params else None)

    def init(self, handle: int, callbacks: CallbackFnsT, context: int | None = None) -> int:
        """Starts streaming; `context` (an address) comes back with every callback."""
        return self._dll.sdrplay_api_Init(handle, byref(callbacks), context)

    def uninit(self, handle: int) -> int:
        return self._dll.sdrplay_api_Uninit(handle)

    def update(self, handle: int, tuner: int, reason: int, reason_ext1: int = 0) -> int:
        return self._dll.sdrplay_api_Update(handle, tuner, reason, reason_ext1)
