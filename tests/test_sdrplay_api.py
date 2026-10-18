"""The ctypes structures against the layout of the original C headers.

The expected values were printed by a small C program against the headers of API 3.15
(`sizeof`/`offsetof` of every structure the binding writes into or reads from); built
with MSVC for Windows x64 and with gcc for Linux x64, it printed the same values. A
mismatch means the binding would read or write the wrong field in the API's memory.
"""

import ctypes

import pytest

from sdr import sdrplay_api as sp

SIZES = [
    (sp.DeviceT, 96),
    (sp.DevParamsT, 64),
    (sp.TunerParamsT, 72),
    (sp.GainT, 24),
    (sp.ControlParamsT, 32),
    (sp.AgcT, 20),
    (sp.RxChannelParamsT, 144),
    (sp.RspDuoTunerParamsT, 16),
    (sp.RspDxParamsT, 12),
    (sp.DeviceParamsT, 24),
    (sp.StreamCbParamsT, 20),
    (sp.GainCbParamT, 16),
    (sp.EventParamsT, 16),
    (sp.CallbackFnsT, 24),
    (sp.ErrorInfoT, 1540),
]

OFFSETS = [
    (sp.DeviceT, "hwVer", 64),
    (sp.DeviceT, "tuner", 68),
    (sp.DeviceT, "rspDuoMode", 72),
    (sp.DeviceT, "valid", 76),
    (sp.DeviceT, "rspDuoSampleFreq", 80),
    (sp.DeviceT, "dev", 88),
    (sp.DevParamsT, "fsFreq", 8),
    (sp.DevParamsT, "syncUpdate", 24),
    (sp.DevParamsT, "resetFlags", 32),
    (sp.DevParamsT, "mode", 36),
    (sp.DevParamsT, "samplesPerPkt", 40),
    (sp.DevParamsT, "rsp1aParams", 44),
    (sp.DevParamsT, "rsp2Params", 46),
    (sp.DevParamsT, "rspDuoParams", 48),
    (sp.DevParamsT, "rspDxParams", 52),
    (sp.TunerParamsT, "gain", 12),
    (sp.TunerParamsT, "rfFreq", 40),
    (sp.TunerParamsT, "dcOffsetTuner", 56),
    (sp.GainT, "LNAstate", 4),
    (sp.GainT, "minGr", 8),
    (sp.GainT, "gainVals", 12),
    (sp.ControlParamsT, "decimation", 2),
    (sp.ControlParamsT, "agc", 8),
    (sp.ControlParamsT, "adsbMode", 28),
    (sp.RxChannelParamsT, "ctrlParams", 72),
    (sp.RxChannelParamsT, "rsp1aTunerParams", 104),
    (sp.RxChannelParamsT, "rsp2TunerParams", 108),
    (sp.RxChannelParamsT, "rspDuoTunerParams", 124),
    (sp.RxChannelParamsT, "rspDxTunerParams", 140),
    (sp.RspDuoTunerParamsT, "tuner1AmPortSel", 4),
    (sp.RspDuoTunerParamsT, "resetSlaveFlags", 11),
    (sp.ErrorInfoT, "line", 512),
    (sp.ErrorInfoT, "message", 516),
]


@pytest.mark.skipif(ctypes.sizeof(ctypes.c_void_p) != 8, reason="layout measured for x64")
@pytest.mark.parametrize(("structure", "size"), SIZES, ids=lambda v: getattr(v, "__name__", v))
def test_structure_size_matches_msvc(structure, size):
    assert ctypes.sizeof(structure) == size


@pytest.mark.skipif(ctypes.sizeof(ctypes.c_void_p) != 8, reason="layout measured for x64")
@pytest.mark.parametrize(("structure", "field", "offset"), OFFSETS)
def test_field_offset_matches_msvc(structure, field, offset):
    assert getattr(structure, field).offset == offset
