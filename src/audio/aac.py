"""AAC decoding of the DAB+ audio units through FAAD2.

The audio units from `dsp.superframe` contain raw AAC access units ("RAW_DATA_BLOCK")
without any container header, neither ADTS nor ADIF. What they consist of is stated
in the format byte of the superframe header instead and has to be handed to the
decoder as an MPEG-4 `AudioSpecificConfig`.

Two DAB+ peculiarities on which a naive integration fails:

* An access unit spans **960** instead of the usual 1 024 core samples (ETSI TS 102
  563, clause 5.1). This is signalled through the `frameLengthFlag` of the
  AudioSpecificConfig; a decoder that does not support it delivers either nothing or
  noise.
* SBR is signalled "explicitly hierarchical": the AAC core runs at half the output
  rate, and the output rate is stated separately in the configuration.
"""

from __future__ import annotations

import ctypes
from typing import Self

import numpy as np

from audio.native import find_library
from dsp.superframe import AudioFormat
from i18n import tr

# ISO/IEC 14496-3, table 1.16 (samplingFrequencyIndex). Only the rates reachable in
# DAB+; nothing else can appear in the format byte.
_SAMPLING_FREQUENCY_INDEX = {48_000: 3, 32_000: 5, 24_000: 6, 16_000: 8}

_AOT_AAC_LC = 2
_AOT_SBR = 5


class _BitWriter:
    """Writes bit fields MSB first, like all MPEG-4 configurations."""

    def __init__(self) -> None:
        self._bits: list[int] = []

    def write(self, value: int, num_bits: int) -> None:
        if not 0 <= value < (1 << num_bits):
            raise ValueError(f"{value} does not fit into {num_bits} bits.")
        self._bits.extend((value >> shift) & 1 for shift in range(num_bits - 1, -1, -1))

    def to_bytes(self) -> bytes:
        padding = (-len(self._bits)) % 8
        return bytes(np.packbits(self._bits + [0] * padding))


def audio_specific_config(audio_format: AudioFormat) -> bytes:
    """Builds the MPEG-4 AudioSpecificConfig for a DAB+ audio format.

    Layout according to ISO/IEC 14496-3, clause 1.6.2.1. Without SBR this is plain
    AAC LC; with SBR the "explicit hierarchical" signalling is used, in which the SBR
    object type with the CORE sample rate and the output rate comes first and the
    actual core object type follows.

    Parametric stereo (PS) is deliberately NOT signalled separately (object type 29):
    DAB+ transmits PS as part of the SBR payload, and FAAD2 detects it from there by
    itself. An explicitly signalled PS that then turns out to be absent from the bit
    stream would lead to a wrong channel count.

    Returns:
        2 bytes without SBR, 4 bytes with SBR (each padded with zero bits).
    """
    core_index = _SAMPLING_FREQUENCY_INDEX[audio_format.core_sample_rate]
    # With PS the AAC core itself is single-channel; the two channels only arise in the
    # decoder from the PS parameters.
    channel_config = 2 if audio_format.stereo_core else 1

    writer = _BitWriter()
    if audio_format.sbr:
        output_index = _SAMPLING_FREQUENCY_INDEX[audio_format.sample_rate]
        writer.write(_AOT_SBR, 5)
        writer.write(core_index, 4)
        writer.write(channel_config, 4)
        writer.write(output_index, 4)
        writer.write(_AOT_AAC_LC, 5)
    else:
        writer.write(_AOT_AAC_LC, 5)
        writer.write(core_index, 4)
        writer.write(channel_config, 4)

    # GASpecificConfig (ISO/IEC 14496-3, clause 4.4.1)
    writer.write(1, 1)  # frameLengthFlag: 960 instead of 1024 core samples (DAB+)
    writer.write(0, 1)  # dependsOnCoreCoder
    writer.write(0, 1)  # extensionFlag
    return writer.to_bytes()


def audio_format_from_config(config: bytes) -> AudioFormat:
    """The DAB+ audio format an AudioSpecificConfig of `audio_specific_config` describes
    (reading a recording back, `recording.playback`). Parametric stereo is not
    signalled there and comes out False; the decoder detects it from the stream as it
    does live.

    Raises:
        ValueError: for a configuration this radio did not write (another object type,
        1024-sample frames, a rate DAB+ does not use).
    """
    bits = int.from_bytes(config, "big")
    total = len(config) * 8
    position = 0

    def take(count: int) -> int:
        nonlocal position
        if position + count > total:
            raise ValueError("AudioSpecificConfig too short")
        position += count
        return (bits >> (total - position)) & ((1 << count) - 1)

    rates = {index: rate for rate, index in _SAMPLING_FREQUENCY_INDEX.items()}
    object_type = take(5)
    core_rate = rates.get(take(4))
    channel_config = take(4)
    sbr = object_type == _AOT_SBR
    if sbr:
        output_rate = rates.get(take(4))
        object_type = take(5)
    else:
        output_rate = core_rate
    if object_type != _AOT_AAC_LC:
        raise ValueError(f"AudioSpecificConfig {config.hex()}: not AAC LC")
    if core_rate is None or output_rate not in (48_000, 32_000):
        raise ValueError(f"AudioSpecificConfig {config.hex()}: not a DAB+ sample rate")
    if sbr and output_rate != 2 * core_rate:
        raise ValueError(f"AudioSpecificConfig {config.hex()}: SBR rates do not match")
    if channel_config not in (1, 2):
        raise ValueError(f"AudioSpecificConfig {config.hex()}: channel configuration")
    if take(1) != 1:
        raise ValueError(f"AudioSpecificConfig {config.hex()}: not 960-sample frames")
    return AudioFormat(
        dac_rate_48k=output_rate == 48_000,
        sbr=sbr,
        stereo_core=channel_config == 2,
        parametric_stereo=False,
        mpeg_surround_config=0,
    )


class AacDecoderError(RuntimeError):
    """The AAC decoder is unavailable or rejects the configuration. Shown to the user."""


class _FaadConfiguration(ctypes.Structure):
    """`NeAACDecConfiguration` from neaacdec.h."""

    _fields_ = [
        ("defObjectType", ctypes.c_ubyte),
        ("defSampleRate", ctypes.c_ulong),
        ("outputFormat", ctypes.c_ubyte),
        ("downMatrix", ctypes.c_ubyte),
        ("useOldADTSFormat", ctypes.c_ubyte),
        ("dontUpSampleImplicitSBR", ctypes.c_ubyte),
    ]


class _FaadFrameInfo(ctypes.Structure):
    """`NeAACDecFrameInfo` from neaacdec.h."""

    _fields_ = [
        ("bytesconsumed", ctypes.c_ulong),
        ("samples", ctypes.c_ulong),
        ("channels", ctypes.c_ubyte),
        ("error", ctypes.c_ubyte),
        ("samplerate", ctypes.c_ulong),
        ("sbr", ctypes.c_ubyte),
        ("object_type", ctypes.c_ubyte),
        ("header_type", ctypes.c_ubyte),
        ("num_front_channels", ctypes.c_ubyte),
        ("num_side_channels", ctypes.c_ubyte),
        ("num_back_channels", ctypes.c_ubyte),
        ("num_lfe_channels", ctypes.c_ubyte),
        ("channel_position", ctypes.c_ubyte * 64),
        ("ps", ctypes.c_ubyte),
    ]


_FAAD_FMT_16BIT = 1


def _load_faad() -> ctypes.CDLL:
    # faad.dll as built by native\build_codecs.ps1, libfaad-2.dll as in MSYS2.
    name = find_library("faad", "libfaad-2") or "libfaad.so.2"
    try:
        library = ctypes.CDLL(name)
    except OSError as exc:  # pragma: no cover - depends on the installation
        raise AacDecoderError(
            tr(
                "FAAD2 (libfaad2) was not found - without this library the DAB+ audio "
                "cannot be decoded. On Debian/Ubuntu: 'apt install libfaad2', on Windows: "
                "build it with native\\build_codecs.ps1 or put faad.dll into the folder "
                "native."
            )
        ) from exc

    library.NeAACDecOpen.restype = ctypes.c_void_p
    library.NeAACDecOpen.argtypes = []
    library.NeAACDecGetCurrentConfiguration.restype = ctypes.POINTER(_FaadConfiguration)
    library.NeAACDecGetCurrentConfiguration.argtypes = [ctypes.c_void_p]
    library.NeAACDecSetConfiguration.restype = ctypes.c_ubyte
    library.NeAACDecSetConfiguration.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_FaadConfiguration),
    ]
    library.NeAACDecInit2.restype = ctypes.c_byte
    library.NeAACDecInit2.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ubyte),
    ]
    library.NeAACDecDecode.restype = ctypes.c_void_p
    library.NeAACDecDecode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_FaadFrameInfo),
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_ulong,
    ]
    library.NeAACDecClose.restype = None
    library.NeAACDecClose.argtypes = [ctypes.c_void_p]
    library.NeAACDecGetErrorMessage.restype = ctypes.c_char_p
    library.NeAACDecGetErrorMessage.argtypes = [ctypes.c_ubyte]
    return library


_faad: ctypes.CDLL | None = None


def _faad_library() -> ctypes.CDLL:
    global _faad
    if _faad is None:
        _faad = _load_faad()
    return _faad


def is_available() -> bool:
    """True if FAAD2 can be loaded."""
    try:
        _faad_library()
    except AacDecoderError:
        return False
    return True


class AacDecoder:
    """Decodes DAB+ audio units to PCM samples (16 bit, interleaved).

    Meant for one fixed audio format: if the format changes (another service), a new
    decoder has to be created, since the AudioSpecificConfig can only be passed once
    when opening.

    Usable as a context manager; otherwise call `close()` so that the memory held by
    FAAD2 is released.
    """

    def __init__(self, audio_format: AudioFormat) -> None:
        # Set first: `__del__` closes it even when loading the library fails.
        self._handle: int | None = None
        self._faad = _faad_library()
        self.audio_format = audio_format

        handle = self._faad.NeAACDecOpen()
        if not handle:
            raise AacDecoderError(tr("Could not initialise FAAD2 (NeAACDecOpen)."))
        self._handle = handle

        config = self._faad.NeAACDecGetCurrentConfiguration(handle)
        config.contents.outputFormat = _FAAD_FMT_16BIT
        # 0 = upsample SBR output to the full rate. Otherwise implicitly detected SBR
        # would come out at half the sample rate, which would not match the rate
        # announced in the superframe header.
        config.contents.dontUpSampleImplicitSBR = 0
        config.contents.downMatrix = 0
        if not self._faad.NeAACDecSetConfiguration(handle, config):
            self.close()
            raise AacDecoderError(tr("FAAD2 rejected the decoder configuration."))

        asc = audio_specific_config(audio_format)
        buffer = (ctypes.c_ubyte * len(asc)).from_buffer_copy(asc)
        sample_rate = ctypes.c_ulong(0)
        channels = ctypes.c_ubyte(0)
        if self._faad.NeAACDecInit2(handle, buffer, len(asc), sample_rate, channels) != 0:
            self.close()
            raise AacDecoderError(
                tr("FAAD2 rejects the AudioSpecificConfig {config} (format: {format}).").format(
                    config=asc.hex(), format=audio_format.describe()
                )
            )
        self.sample_rate = int(sample_rate.value)
        self.channels = int(channels.value)

    def decode(self, audio_unit: bytes) -> np.ndarray:
        """Decodes one audio unit to interleaved int16 samples.

        Returns:
            1D array with `samples * channels` values (L,R,L,R,... for stereo); empty
            if FAAD2 delivers no output for this unit yet. That is normal while the
            decoder is settling on SBR/PS.

        Raises:
            AacDecoderError: if FAAD2 reports the unit as faulty.
        """
        if self._handle is None:
            raise AacDecoderError(tr("The decoder has already been closed."))
        if not audio_unit:
            return np.empty(0, dtype=np.int16)

        info = _FaadFrameInfo()
        buffer = (ctypes.c_ubyte * len(audio_unit)).from_buffer_copy(audio_unit)
        pointer = self._faad.NeAACDecDecode(self._handle, info, buffer, len(audio_unit))

        if info.error:
            message = self._faad.NeAACDecGetErrorMessage(info.error)
            raise AacDecoderError(
                f"FAAD2: {message.decode('utf-8', 'replace') if message else info.error}"
            )
        if not pointer or info.samples == 0:
            return np.empty(0, dtype=np.int16)

        # `NeAACDecDecode` returns a pointer to a decoder-internal buffer that is
        # overwritten on the next call, hence the copy.
        self.channels = int(info.channels)
        self.sample_rate = int(info.samplerate)
        array_type = ctypes.c_int16 * int(info.samples)
        return np.frombuffer(array_type.from_address(pointer), dtype=np.int16).copy()

    def close(self) -> None:
        if self._handle is not None:
            self._faad.NeAACDecClose(self._handle)
            self._handle = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
