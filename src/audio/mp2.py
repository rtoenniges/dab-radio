"""MPEG Audio Layer II decoding of classic DAB services through libmpg123.

The counterpart of `audio.aac` for DAB services without "+". Unlike AAC in DAB+, the
frames need no configuration of their own: every MP2 frame starts with a complete
header, so mpg123 is fed the frames exactly as `dsp.mp2_frame` found them. Layer II
has no bit reservoir either, so leaving out a corrupted frame does not affect its
neighbours.
"""

from __future__ import annotations

import ctypes
from typing import Self

import numpy as np

from audio.native import find_library
from dsp.mp2_frame import SAMPLES_PER_FRAME, Mp2Format
from i18n import tr

# From mpg123.h / fmt123.h (API version 49; the values are stable across versions).
_MPG123_OK = 0
_MPG123_NEW_FORMAT = -11
_MPG123_NEED_MORE = -10
_MPG123_ADD_FLAGS = 2
_MPG123_QUIET = 0x20
_MPG123_MONO = 1
_MPG123_STEREO = 2
_MPG123_ENC_SIGNED_16 = 0xD0

# Room for the samples of one frame: 1152 samples, at most two channels, 2 bytes each.
_OUTPUT_BUFFER_BYTES = SAMPLES_PER_FRAME * 2 * 2


class Mp2DecoderError(RuntimeError):
    """The MP2 decoder is unavailable or rejects a frame. Shown to the user."""


def _load_mpg123() -> ctypes.CDLL:
    # mpg123.dll as built by native\build_codecs.ps1, libmpg123-0.dll as distributed by
    # the mpg123 project and MSYS2.
    name = find_library("mpg123", "libmpg123-0") or "libmpg123.so.0"
    try:
        library = ctypes.CDLL(name)
    except OSError as exc:  # pragma: no cover - depends on the installation
        raise Mp2DecoderError(
            tr(
                "libmpg123 was not found - without this library the audio of classic DAB "
                "stations cannot be decoded. On Debian/Ubuntu: 'apt install libmpg123-0', "
                "on Windows: build it with native\\build_codecs.ps1 or put mpg123.dll into "
                "the folder native."
            )
        ) from exc

    library.mpg123_init.restype = ctypes.c_int
    library.mpg123_init.argtypes = []
    library.mpg123_new.restype = ctypes.c_void_p
    library.mpg123_new.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    library.mpg123_delete.restype = None
    library.mpg123_delete.argtypes = [ctypes.c_void_p]
    library.mpg123_param.restype = ctypes.c_int
    library.mpg123_param.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long, ctypes.c_double]
    library.mpg123_format_none.restype = ctypes.c_int
    library.mpg123_format_none.argtypes = [ctypes.c_void_p]
    library.mpg123_format.restype = ctypes.c_int
    library.mpg123_format.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_int]
    library.mpg123_open_feed.restype = ctypes.c_int
    library.mpg123_open_feed.argtypes = [ctypes.c_void_p]
    # `mpg123_decode` rather than `mpg123_decode_frame`: it has no `off_t` parameter,
    # whose size (and hence the exported symbol name) depends on how libmpg123 was
    # built.
    library.mpg123_decode.restype = ctypes.c_int
    library.mpg123_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.mpg123_getformat.restype = ctypes.c_int
    library.mpg123_getformat.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_long),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    library.mpg123_plain_strerror.restype = ctypes.c_char_p
    library.mpg123_plain_strerror.argtypes = [ctypes.c_int]

    # Required before libmpg123 1.27, a harmless no-op since.
    library.mpg123_init()
    return library


_mpg123: ctypes.CDLL | None = None


def _mpg123_library() -> ctypes.CDLL:
    global _mpg123
    if _mpg123 is None:
        _mpg123 = _load_mpg123()
    return _mpg123


def is_available() -> bool:
    """True if libmpg123 can be loaded."""
    try:
        _mpg123_library()
    except Mp2DecoderError:
        return False
    return True


class Mp2Decoder:
    """Decodes DAB audio frames to PCM samples (16 bit, interleaved).

    Same interface as `audio.aac.AacDecoder`, so that the player can use either. A new
    decoder is created per format, like for AAC; mpg123 would cope with a format change
    by itself, but this way both decoders behave alike in the player.

    Usable as a context manager; otherwise call `close()` so that the memory held by
    libmpg123 is released.
    """

    def __init__(self, audio_format: Mp2Format) -> None:
        # Set first: `__del__` closes it even when loading the library fails.
        self._handle: int | None = None
        self._mpg123 = _mpg123_library()
        self.audio_format = audio_format
        # Until mpg123 reports the format of the first decoded frame, the header tells.
        self.sample_rate = audio_format.sample_rate
        self.channels = audio_format.channels
        self._output = ctypes.create_string_buffer(_OUTPUT_BUFFER_BYTES)

        error = ctypes.c_int(0)
        handle = self._mpg123.mpg123_new(None, ctypes.byref(error))
        if not handle:
            raise Mp2DecoderError(
                tr("Could not initialise libmpg123 ({code}).").format(code=error.value)
            )
        self._handle = handle

        # Only 16-bit output at the two DAB rates, in the native channel count; mpg123
        # would otherwise be free to pick another encoding. QUIET keeps its own
        # warnings off stderr, errors are reported through the return codes.
        steps = (
            self._mpg123.mpg123_param(handle, _MPG123_ADD_FLAGS, _MPG123_QUIET, 0.0),
            self._mpg123.mpg123_format_none(handle),
            *(
                self._mpg123.mpg123_format(
                    handle, rate, _MPG123_MONO | _MPG123_STEREO, _MPG123_ENC_SIGNED_16
                )
                for rate in (48_000, 24_000)
            ),
            self._mpg123.mpg123_open_feed(handle),
        )
        failed = next((code for code in steps if code != _MPG123_OK), None)
        if failed is not None:
            self.close()
            raise Mp2DecoderError(
                tr("libmpg123 rejects the configuration: {error}").format(
                    error=self._message(failed)
                )
            )

    def decode(self, frame: bytes) -> np.ndarray:
        """Decodes one complete frame (header included) to interleaved int16 samples.

        Returns:
            1D array with `samples * channels` values (L,R,L,R,... for stereo); empty if
            mpg123 delivers no output for this frame yet.

        Raises:
            Mp2DecoderError: if libmpg123 reports an error.
        """
        if self._handle is None:
            raise Mp2DecoderError(tr("The decoder has already been closed."))
        if not frame:
            return np.empty(0, dtype=np.int16)

        chunks: list[bytes] = []
        done = ctypes.c_size_t(0)
        data: bytes | None = frame
        while True:
            code = self._mpg123.mpg123_decode(
                self._handle,
                data,
                len(data) if data else 0,
                self._output,
                _OUTPUT_BUFFER_BYTES,
                ctypes.byref(done),
            )
            data = None  # the input is only handed over once, then drained
            if done.value:
                chunks.append(self._output.raw[: done.value])
            if code == _MPG123_NEW_FORMAT:
                self._read_format()
            elif code == _MPG123_NEED_MORE:
                break
            elif code != _MPG123_OK:
                raise Mp2DecoderError(f"libmpg123: {self._message(code)}")
            elif not done.value:
                break
        return np.frombuffer(b"".join(chunks), dtype=np.int16).copy()

    def _read_format(self) -> None:
        rate = ctypes.c_long(0)
        channels = ctypes.c_int(0)
        encoding = ctypes.c_int(0)
        code = self._mpg123.mpg123_getformat(
            self._handle, ctypes.byref(rate), ctypes.byref(channels), ctypes.byref(encoding)
        )
        if code == _MPG123_OK:
            self.sample_rate = int(rate.value)
            self.channels = int(channels.value)

    def _message(self, code: int) -> str:
        message = self._mpg123.mpg123_plain_strerror(code)
        return message.decode("utf-8", "replace") if message else str(code)

    def close(self) -> None:
        if self._handle is not None:
            self._mpg123.mpg123_delete(self._handle)
            self._handle = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
