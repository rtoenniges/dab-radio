"""Choosing the librtlsdr that pyrtlsdr loads: the radio's own build with RAW_IO.

On Windows librtlsdr reaches the stick through libusb and the WinUSB driver. Without the
pipe policy RAW_IO, WinUSB passes on one read at a time however many USB transfers
librtlsdr keeps queued (libusb issue 490). Between two of them the stick is not polled
for a moment, its buffer overflows, and samples are lost without any notice. In the
stick's test mode (a byte counter instead of samples) at 2.048 MS/s that makes a few
jumps of about 0.9 ms in ten minutes on an otherwise idle machine, more and longer ones
when the machine stutters; with RAW_IO none. For DAB every such loss shifts the symbol
grid in the middle of a frame (see `sdr.worker.RtlSdrWorker.run`). With the radio's
transfers of 786 432 bytes (pyrtlsdr) there are three times fewer transfer boundaries
and the losses are rarer, but the cause is the same.

The library bundled with `pyrtlsdrlib` (rtl-sdr 0.8.0 of the librtlsdr/librtlsdr fork)
comes with a libusb that cannot set RAW_IO. `native/build_rtlsdr.ps1` therefore builds
`native/rtlsdr.dll`: rtl-sdr 2.0.3 of osmocom with libusb 1.0.30 linked in, patched to
switch RAW_IO on when it opens the stick. Where it has not been built, the bundled
library serves, and the log warns.

pyrtlsdr asks `pyrtlsdrlib.load_librtlsdr()` for its library the moment `rtlsdr` is
first imported. `install()` puts `load_librtlsdr()` of this module in its place; the
package `sdr` calls it on import, so it is in effect for everything that reaches
`rtlsdr` through `sdr`. A script that imports `rtlsdr` before `sdr` gets the bundled
library; the log warns then.

What the own build changes besides RAW_IO:

- With RAW_IO, WinUSB rejects every read that is not a multiple of the USB packet size
  of 512 bytes. All block sizes in `sdr` are such multiples.
- rtl-sdr 2.0.3 lacks the fork's functions for PLL dithering and the GPIOs, which
  pyrtlsdr binds on import without checking for them. They are filled in here, see
  `_FORK_ONLY_FUNCTIONS`.

What it must not change: the fork leaves the IF amplifier (VGA) of the R820T/2 to the
RTL2832U's IF AGC, rtl-sdr 2.0.3 sets it to a fixed 16.3 dB. The radio's gain setting
relies on the former (see `DEFAULT_GAIN_DB` in `ui.receiver_controller`), so
`native/rtlsdr-r82xx-if-agc.patch` restores it.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from pathlib import Path

import pyrtlsdrlib

logger = logging.getLogger(__name__)

# The radio's own build with RAW_IO, next to its build script.
NATIVE_LIBRARY = Path(__file__).resolve().parents[2] / "native" / "rtlsdr.dll"

# Functions of the librtlsdr/librtlsdr fork that pyrtlsdr 0.5 binds on import without
# checking for them and that osmocom's rtl-sdr does not have. Without a stand-in,
# `import rtlsdr` fails with an AttributeError. The radio uses none of them itself.
_FORK_ONLY_FUNCTIONS = (
    "rtlsdr_set_dithering",
    "rtlsdr_set_gpio_output",
    "rtlsdr_set_gpio_input",
    "rtlsdr_set_gpio_bit",
    "rtlsdr_get_gpio_bit",
    "rtlsdr_set_gpio_byte",
    "rtlsdr_get_gpio_byte",
    "rtlsdr_set_gpio_status",
)

_load_bundled = pyrtlsdrlib.load_librtlsdr


def _set_dithering(_dev: object, enabled: int) -> int:
    """Stands in for `rtlsdr_set_dithering`, which pyrtlsdr calls on every open.

    osmocom's rtl-sdr has no such switch, the PLL of the R820T/2 always dithers. That
    is what pyrtlsdr asks for unless told otherwise, so switching it on succeeds and
    switching it off fails.
    """
    return 0 if enabled else -1


def _unsupported(*_args: object) -> int:
    """Stands in for the GPIO functions: fails like a library call would."""
    return -1


def _add_fork_only_functions(dll: object) -> None:
    for name in _FORK_ONLY_FUNCTIONS:
        if not hasattr(dll, name):
            stand_in = _set_dithering if name == "rtlsdr_set_dithering" else _unsupported
            setattr(dll, name, stand_in)


def load_librtlsdr() -> ctypes.CDLL | None:
    """The radio's own build if it has been built, else the library of `pyrtlsdrlib`.

    Takes the place of `pyrtlsdrlib.load_librtlsdr`, see `install`.
    """
    if sys.platform != "win32":
        # RAW_IO is a setting of WinUSB; elsewhere the bundled library has no such gap.
        return _load_bundled()
    if NATIVE_LIBRARY.is_file():
        try:
            dll = ctypes.CDLL(str(NATIVE_LIBRARY))
        except OSError:
            logger.exception("Failed to load %s", NATIVE_LIBRARY)
        else:
            _add_fork_only_functions(dll)
            return dll
    logger.warning(
        "Using the librtlsdr of pyrtlsdrlib, which cannot set RAW_IO: the stick will "
        "lose samples between USB transfers every few minutes. Build %s with "
        "native/build_rtlsdr.ps1",
        NATIVE_LIBRARY,
    )
    return _load_bundled()


def install() -> None:
    """Makes pyrtlsdr load its library through `load_librtlsdr`.

    Must run before `rtlsdr` is imported for the first time; afterwards pyrtlsdr has
    already bound the functions of the library it found.
    """
    too_late = "rtlsdr.librtlsdr" in sys.modules
    if too_late and sys.platform == "win32" and pyrtlsdrlib.load_librtlsdr is not load_librtlsdr:
        logger.warning(
            "rtlsdr was imported before sdr: pyrtlsdr keeps the library of pyrtlsdrlib "
            "it has already loaded, which cannot set RAW_IO"
        )
    pyrtlsdrlib.load_librtlsdr = load_librtlsdr
