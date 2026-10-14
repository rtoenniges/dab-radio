"""Reading the IQ stream from the RTL-SDR stick or the RSPduo in a thread of its own.

`RtlSdrWorker` and `RspDuoWorker` offer the GUI the same signals and methods
(`samples_ready`, `tuned`, `connected`, `error`, `stopped`, `tuning`, `retune`, `stop`),
so that everything downstream runs unchanged with either device.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections.abc import Callable

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from rtlsdr import RtlSdr

from dab.constants import MODE_I, SAMPLE_RATE
from i18n import exception_text, tr
from sdr.device import (
    DISCARD_AFTER_OPEN_SAMPLES,
    RETUNE_SETTLE_SAMPLES,
    open_rtlsdr,
    retune_rtlsdr,
)
from sdr.exceptions import RtlSdrError, SdrplayError
from sdr.rspduo import RspDuoSettings
from sdr.rspduo_reader import RspDuoReader

logger = logging.getLogger(__name__)

# Must span at least two transmission frames: since read blocks are not synchronised
# to the frame clock, the null symbol can lie anywhere in the buffer. Only if a
# complete frame is guaranteed to follow it can all 75 data symbols (FIC + MSC) be
# demodulated; with a buffer of only one frame the null symbol would occasionally be
# found too late and the rest of the frame cut off (see the `max_start` parameter of
# dsp.sync.find_null_symbol, which makes sure only null symbols with enough frame left
# after them are candidates). 2 * 196 608 samples = 786 432 bytes is also a multiple of
# 512, the USB packet size: librtlsdr requires that for the asynchronous USB transfers
# (it silently falls back to its default length otherwise), and so does WinUSB with
# RAW_IO (see `sdr.library`).
DEFAULT_READ_SIZE = 2 * MODE_I.frame_length

# Wait between the end of the asynchronous read and closing the device.
# `rtlsdr_cancel_async` only initiates the cancellation of the running USB transfers;
# closing the device too early leaves it streaming and on the next open it either
# delivers nothing at all (blocking read) or constant filler bytes, fixable only by a
# USB reset or unplugging the stick. Measured on real hardware: without the wait the
# NEXT start fails, 50 ms are not enough, from 100 ms on it works reliably, so 250 ms
# with a safety margin. A channel change does not close the device (see
# `RtlSdrWorker.retune`), so this only applies to stopping.
_CLOSE_SETTLE_SECONDS = 0.25

# Every tuning of a stick, be it opening it or retuning it while running, gets a number
# unique within the process. The worker announces it in the sample stream itself (see
# `RtlSdrWorker.tuned`), so that everything downstream can tell where the samples of
# the new frequency begin, even across several workers.
_tuning_numbers = itertools.count()


class RtlSdrWorker(QThread):
    """Reads IQ samples from an RTL-SDR stick in a thread of its own.

    All communication with the GUI goes through Qt signals so that the signal
    processing never blocks the UI thread.
    """

    samples_ready = pyqtSignal(object)  # numpy.ndarray[complex64]
    # Number of the tuning (see `tuning`) whose samples follow. Emitted from the reading
    # thread right before the first block received on a new frequency, after opening and
    # after every `retune`, so that in queued connections it lines up exactly at the
    # boundary between the samples of the old and the new frequency.
    tuned = pyqtSignal(int)
    # Texts for the status line; objects, not `str`, so that a translated text keeps its
    # English for the remote interfaces (`i18n.Translated`).
    connected = pyqtSignal(object)
    error = pyqtSignal(object)
    stopped = pyqtSignal()

    def __init__(
        self,
        center_freq: float,
        sample_rate: float = SAMPLE_RATE,
        gain: str | float = "auto",
        freq_correction_ppm: int = 0,
        device_index: int = 0,
        read_size: int = DEFAULT_READ_SIZE,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._center_freq = center_freq
        self._sample_rate = sample_rate
        self._gain = gain
        self._freq_correction_ppm = freq_correction_ppm
        self._device_index = device_index
        self._read_size = read_size
        self._running = False
        self._read_cancelled = False
        self._sdr: RtlSdr | None = None
        # Number of the current tuning, as announced by `tuned`.
        self.tuning = next(_tuning_numbers)
        # Keeps `retune()`, which runs in the GUI thread, and closing the device apart.
        # Never taken in the read callback, see `retune`.
        self._device_lock = threading.Lock()
        # Guards the discard bookkeeping below, shared by `retune()` and the read
        # callback. Only ever held briefly.
        self._stream_lock = threading.Lock()
        self._retuning = False  # the tuner is being switched right now
        self._discard_remaining = 0  # samples still to be discarded
        self._unannounced_tuning: int | None = None  # to be announced with the next block

    def run(self) -> None:
        """Reads the IQ stream **asynchronously** (librtlsdr `rtlsdr_read_async`).

        Important and not interchangeable with a loop over the synchronous
        `read_samples()`: the synchronous interface starts a single USB bulk transfer
        per call and reads nothing between two calls. Everything the stick produces in
        the gap is lost, measured on real hardware at about 1 % of the samples, in
        bursts of a few thousand samples every few frames. For the bare
        spectrum/constellation display that goes unnoticed, and even the FIC stays
        error-free (every frame is decoded independently there), but for the MSC it is
        fatal: the time deinterleaver (see `dsp.time_interleave.TimeDeinterleaver`)
        requires a gapless sequence of CIFs, and every sample loss shifts the symbol
        grid in the middle of a frame, so audio decoding fails permanently. The
        asynchronous path keeps several USB transfers queued at all times and delivers
        a gapless stream (verified: the distance between consecutive phase reference
        symbols is exactly constant over a whole recording).

        On Windows that holds only with the radio's own build of librtlsdr, which
        switches on WinUSB's RAW_IO: without it the queued transfers are passed on one
        at a time, and every few minutes samples are lost between two of them (see
        `sdr.library`).
        """
        try:
            self._sdr = self._open_device()
        except RtlSdrError as exc:
            self.error.emit(exception_text(exc))
            return

        # The first blocks still come from the previously tuned frequency (see
        # DISCARD_AFTER_OPEN_SAMPLES).
        with self._stream_lock:
            self._discard_remaining = DISCARD_AFTER_OPEN_SAMPLES
            self._unannounced_tuning = self.tuning
        # Before `connected`: from then on the GUI may call `retune`.
        self._running = True
        self._read_cancelled = False
        self.connected.emit(
            tr("RTL-SDR ready @ {frequency} MHz, {rate} MSps").format(
                frequency=f"{self._center_freq / 1e6:.3f}", rate=f"{self._sample_rate / 1e6:.3f}"
            )
        )

        try:
            self._sdr.read_samples_async(self._on_samples, self._read_size)
        except Exception as exc:  # noqa: BLE001 - hardware errors must not crash the thread
            if self._running:
                self.error.emit(tr("Error reading the IQ samples: {error}").format(error=exc))

        time.sleep(_CLOSE_SETTLE_SECONDS)  # see _CLOSE_SETTLE_SECONDS
        self._close_device()
        self.stopped.emit()

    def _on_samples(self, samples, context) -> None:
        """Callback of `read_samples_async` (runs in this thread).

        Must stay short: while it runs, no USB transfers are collected.
        `samples_ready` is connected across thread boundaries and therefore only
        queues the block in the DSP thread's event loop instead of processing it here.
        """
        if not self._running:
            self._cancel_read()
            return
        with self._stream_lock:
            if self._retuning:
                return
            if self._discard_remaining > 0:
                # Whole blocks only, see `retune`.
                self._discard_remaining -= len(samples)
                return
            tuning = self._unannounced_tuning
            self._unannounced_tuning = None
        if tuning is not None:
            self.tuned.emit(tuning)
        self.samples_ready.emit(samples)

    def _cancel_read(self) -> None:
        """Cancels the asynchronous read, exactly once.

        `rtlsdr_cancel_async` only sets a cancel flag; the USB transfers still in
        flight are reported back one after another afterwards, so the callback runs
        several more times. A repeated cancel would leave the device in a state where
        it keeps streaming and on the next open blocks or aborts with
        LIBUSB_ERROR_TIMEOUT (fixable only by a USB reset).
        """
        sdr = self._sdr
        if sdr is None or self._read_cancelled:
            return
        self._read_cancelled = True
        try:
            sdr.cancel_read_async()
        except Exception:  # an error while cancelling no longer matters
            logger.debug("cancel_read_async failed", exc_info=True)

    def retune(self, center_freq: float) -> int | None:
        """Tunes the running stick to another frequency without interrupting reception.

        Called from the GUI thread while `run()` reads. librtlsdr allows that: tuning
        goes through USB control transfers, independent of the bulk transfers of the
        running read (rtl_tcp retunes from a thread of its own the same way). Unlike a
        stop and restart, the device stays open, so neither the wait before closing
        (`_CLOSE_SETTLE_SECONDS`) nor the leftovers after opening
        (`DISCARD_AFTER_OPEN_SAMPLES`) apply. Measured on real hardware (R820T): the
        call takes about 42 ms and the sample stream continues without a gap.

        Discarded are the blocks arriving while the tuner switches and afterwards
        `read_size + RETUNE_SETTLE_SAMPLES` samples in whole blocks, i.e. two blocks
        (~0.4 s). The first of them was already being filled at the switch and on real
        hardware still contained decodable frames of the old channel. The second
        covers the settling, and the case that a block completed during the switch
        only reaches the callback after it and takes the place of the first. The
        first block kept is preceded by `tuned`.

        The read callback must never wait for the duration of the switch: the calling
        thread waits for its control transfers to complete, and librtlsdr's event
        handling in the reading thread is what completes them. Hence the separate,
        briefly held `_stream_lock`.

        Returns:
            The number of the new tuning, as `tuned` will announce it; None if the
            worker is not reading (not yet connected or already stopping).

        Raises:
            RtlSdrError: if the tuner could not be switched. No samples are passed on
                from then on; the caller should stop the worker.
        """
        with self._device_lock:
            sdr = self._sdr
            if sdr is None or not self._running:
                return None
            tuning = next(_tuning_numbers)
            with self._stream_lock:
                self._retuning = True
                self._unannounced_tuning = tuning
            retune_rtlsdr(sdr, center_freq)
            with self._stream_lock:
                self._retuning = False
                self._discard_remaining = self._read_size + RETUNE_SETTLE_SAMPLES
            self._center_freq = center_freq
            self.tuning = tuning
        return tuning

    def stop(self) -> None:
        """Asks the thread to finish and waits until it has.

        `rtlsdr_cancel_async` is expressly meant to be called from a thread other than
        the reading one and cancels the `read_samples_async` running in `run()`. The
        wait must cover the draining of the USB transfers plus `_CLOSE_SETTLE_SECONDS`
        so that the device is really closed before the caller carries on.
        """
        self._running = False
        self._cancel_read()
        self.wait(5000)

    def _open_device(self) -> RtlSdr:
        return open_rtlsdr(
            center_freq=self._center_freq,
            sample_rate=self._sample_rate,
            gain=self._gain,
            freq_correction_ppm=self._freq_correction_ppm,
            device_index=self._device_index,
        )

    def _close_device(self) -> None:
        with self._device_lock:
            if self._sdr is not None:
                try:
                    self._sdr.close()
                except Exception:
                    logger.exception("Failed to close the RTL-SDR device")
                self._sdr = None


class RspDuoWorker(QThread):
    """Reads IQ samples from one tuner of the RSPduo in a thread of its own.

    Same signals and methods as `RtlSdrWorker`, plus `overload_changed`. The samples
    reach Python only here: the API's thread puts its packets into a ring without the
    GIL (`sdr.packet_ring`), and `run()` takes them out in blocks of `read_size`
    (`sdr.rspduo_reader.RspDuoReader`), which the API's sample numbers keep gapless.
    """

    samples_ready = pyqtSignal(object)  # numpy.ndarray[complex64]
    tuned = pyqtSignal(int)  # see `RtlSdrWorker.tuned`
    connected = pyqtSignal(object)  # see `RtlSdrWorker.connected`
    error = pyqtSignal(object)
    stopped = pyqtSignal()
    # True while the RSPduo reports an overload of its ADC (from the API's thread).
    overload_changed = pyqtSignal(bool)

    def __init__(
        self,
        center_freq: float,
        settings: RspDuoSettings,
        read_size: int = DEFAULT_READ_SIZE,
        reader_factory: Callable[..., RspDuoReader] = RspDuoReader,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._center_freq = center_freq
        self._settings = settings
        self._read_size = read_size
        self._reader_factory = reader_factory
        self._running = False
        self._stop_requested = False
        self._reader: RspDuoReader | None = None
        # Number of the current tuning, as announced by `tuned`.
        self.tuning = next(_tuning_numbers)
        # Keeps `retune()` and `stop()`, which run in the GUI thread, and opening and
        # closing the device apart.
        self._device_lock = threading.Lock()

    def run(self) -> None:
        reader = self._reader_factory(
            self._settings,
            self._center_freq,
            tag=self.tuning,
            on_overload=self.overload_changed.emit,
        )
        try:
            info = reader.open()
        except SdrplayError as exc:
            self.error.emit(exception_text(exc))
            return

        with self._device_lock:
            if self._stop_requested:
                # Stopped while opening (closing the window), which takes about 0.4 s.
                reader.close()
                stopped_early = True
            else:
                stopped_early = False
                self._reader = reader
                # Before `connected`: from then on the GUI may call `retune`.
                self._running = True
        if stopped_early:
            self.stopped.emit()
            return
        self.connected.emit(
            tr("RSPduo {serial}, tuner {tuner} ready @ {frequency} MHz, {rate} MSps").format(
                serial=info.serial,
                tuner=self._settings.tuner,
                frequency=f"{self._center_freq / 1e6:.3f}",
                rate=f"{SAMPLE_RATE / 1e6:.3f}",
            )
        )

        failure: str | None = None
        announced: int | None = None
        try:
            while self._running:
                block = reader.read_block(self._read_size)
                if block is None:
                    break
                tuning, samples = block
                if tuning != announced:
                    # Right before the first block of the new frequency, see
                    # `RtlSdrWorker.tuned`.
                    self.tuned.emit(tuning)
                    announced = tuning
                self.samples_ready.emit(samples)
        except SdrplayError as exc:
            failure = exception_text(exc)
        except Exception as exc:  # hardware errors must not crash the thread
            logger.exception("Reading from the RSPduo failed")
            failure = tr("Error reading the IQ samples: {error}").format(error=exc)

        with self._device_lock:
            self._running = False
            self._reader = None
            reader.close()
        if failure is not None:
            # Like a failed open: the message stays instead of "Stopped.".
            self.error.emit(failure)
        else:
            self.stopped.emit()

    def retune(self, center_freq: float) -> int | None:
        """Tunes the running RSPduo to another frequency without interrupting reception.

        Called from the GUI thread while `run()` reads. Everything the device delivered
        before the API applied the frequency is dropped, plus the settling time (see
        `RspDuoReader.retune`); the first block kept is preceded by `tuned`.

        Returns:
            The number of the new tuning, as `tuned` will announce it; None if the
            worker is not reading (not yet connected or already stopping).

        Raises:
            SdrplayError: if the tuner could not be switched. No samples are passed on
                from then on; the caller should stop the worker.
        """
        with self._device_lock:
            reader = self._reader
            if reader is None or not self._running:
                return None
            tuning = next(_tuning_numbers)
            reader.retune(center_freq, tuning)
            self._center_freq = center_freq
            self.tuning = tuning
        return tuning

    def stop(self) -> None:
        """Asks the thread to finish and waits until the device is released."""
        with self._device_lock:
            self._stop_requested = True
            self._running = False
            if self._reader is not None:
                self._reader.cancel()
        self.wait(5000)
