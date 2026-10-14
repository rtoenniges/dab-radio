"""Opening and configuring the RTL-SDR stick, shared by reception and channel scan."""

from __future__ import annotations

import sys

from rtlsdr import RtlSdr
from rtlsdr.librtlsdr import librtlsdr
from rtlsdr.rtlsdr import LibUSBError

from i18n import tr, tr_noop
from sdr.exceptions import RtlSdrError

# After opening, the stick briefly delivers samples from the frequency it was tuned to
# before. Measured on real hardware: after receiving on 5D and reopening on the empty
# channel 5A, the first 0.1-0.4 s still contained one or two fully decodable frames
# from 5D, even in a freshly started program. Unfiltered, their services end up in the
# service list of the new channel (on a channel change just as on the first channel of
# a scan), on an empty channel complete with the ensemble name. Retuning an already
# open device does not show this. The first 0.5 s after every open are therefore
# discarded. 1 024 000 samples = 4000 * 512 bytes, a multiple of the USB packet size
# for the synchronous reads in the scanner.
DISCARD_AFTER_OPEN_SAMPLES = 1_024_000

# Samples discarded after every retune of an open device (50 ms) to let oscillator and
# gain control settle. On real hardware that is enough: during a scan, 5D right after
# the ensemble on 5C contained not a single service from there, and during reception a
# block starting 17 ms after the retune already decoded without errors. 102 400 samples
# = 400 * 512 bytes.
RETUNE_SETTLE_SAMPLES = 102_400

_BUSY_MESSAGE = tr_noop("RTL-SDR device busy. Is another application already using it?")
_NOT_FOUND_MESSAGE = tr_noop("RTL-SDR device not found. Is the stick connected?")

# The messages for the error codes of libusb, translated when shown
# (`describe_libusb_error`).
_LIBUSB_ERROR_MESSAGES: dict[int, str] = {
    -3: tr_noop("Access to the RTL-SDR device denied (missing permissions/udev rule?)."),
    -4: _NOT_FOUND_MESSAGE,
    -5: _NOT_FOUND_MESSAGE,
    -6: _BUSY_MESSAGE,
}
if sys.platform == "win32":
    # WinUSB lets only one program at a time open the stick and reports every further
    # attempt as LIBUSB_ERROR_ACCESS, not as LIBUSB_ERROR_BUSY like Linux does. Measured
    # with a second process holding the stick: both reading its USB strings and opening
    # it fail with -3. There are no udev rules to point to there either.
    _LIBUSB_ERROR_MESSAGES[-3] = _BUSY_MESSAGE


def open_rtlsdr(
    center_freq: float,
    sample_rate: float,
    gain: str | float,
    freq_correction_ppm: int = 0,
    device_index: int = 0,
) -> RtlSdr:
    """Opens and configures the RTL-SDR stick.

    Shared by reception (`sdr.worker.RtlSdrWorker`) and the channel scan
    (`sdr.scanner.ChannelScanner`) so that both produce the same understandable error
    messages.

    Raises:
        RtlSdrError: if no device is connected, it is busy or inaccessible, or it
            cannot be configured.
    """
    try:
        serials = RtlSdr.get_device_serial_addresses()
    except LibUSBError as exc:
        # Reading the serial numbers already opens every stick. If another program is
        # using it, that fails here and not only at `RtlSdr()` below. Uncaught, the
        # exception would end the thread without `error`, and the GUI would wait for
        # the connection forever.
        raise RtlSdrError(describe_libusb_error(exc)) from exc
    if not serials:
        raise RtlSdrError(tr("No RTL-SDR device found. Is the stick connected?"))

    try:
        sdr = RtlSdr(device_index=device_index)
    except LibUSBError as exc:
        raise RtlSdrError(describe_libusb_error(exc)) from exc
    except Exception as exc:
        raise RtlSdrError(
            tr("Could not initialise the RTL-SDR: {error}").format(error=exc)
        ) from exc

    try:
        sdr.sample_rate = sample_rate
        if freq_correction_ppm:
            sdr.freq_correction = freq_correction_ppm
        sdr.center_freq = center_freq
        # Besides the tuner's own gain control (which `sdr.gain` drives), the RTL2832U
        # has a second, separate digital AGC stage in the chip itself. pyrtlsdr does
        # NOT switch it along when `sdr.gain` is set; its state otherwise persists
        # across (re)opening the device at whatever the driver set last time. Keep it
        # explicitly in step with the requested gain mode, otherwise this second AGC
        # stage can keep overriding a manually chosen gain.
        sdr.set_agc_mode(gain == "auto")
        sdr.gain = gain
    except Exception as exc:
        sdr.close()
        raise RtlSdrError(tr("RTL-SDR configuration failed: {error}").format(error=exc)) from exc

    return sdr


def retune_rtlsdr(sdr: RtlSdr, center_freq: float) -> None:
    """Retunes an open stick, also while an asynchronous read is running.

    Calls librtlsdr directly instead of pyrtlsdr's `center_freq` setter: on an error
    the setter closes the device itself, which during a running `read_samples_async`
    would pull the device away from under the reading thread.

    Raises:
        RtlSdrError: if the tuner could not be set to the frequency; the device stays
            open.
    """
    result = librtlsdr.rtlsdr_set_center_freq(sdr.dev_p, round(center_freq))
    if result < 0:
        raise RtlSdrError(
            tr("Retuning to {frequency} MHz failed (error code {code}).").format(
                frequency=f"{center_freq / 1e6:.3f}", code=result
            )
        )


def describe_libusb_error(exc: LibUSBError) -> str:
    """Turns a libusb error code into the message shown to the user."""
    message = _LIBUSB_ERROR_MESSAGES.get(exc.errno)
    if message is None:
        return tr("Could not open the RTL-SDR device: {error}").format(error=exc)
    return tr(message)
