"""Blocks of IQ samples from the RSPduo, read like pyrtlsdr's synchronous reads.

The RSPduo pushes its packets into a ring (`sdr.packet_ring`); `RspDuoReader` drains it
from the calling thread and hands out blocks of any length as complex64 with full scale
1.0, like pyrtlsdr. Reception (`sdr.worker.RspDuoWorker`) reads blocks of two frames in a
loop, the channel scan (`sdr.scanner.ChannelScanner`) uses the same calls as on the
stick (`read_samples`, `center_freq`, `close`).

The API numbers its samples (`firstSampleNum`, 32 bits, wrapping) and flags the packet
with which a new frequency took effect (`rfChanged`), which pyrtlsdr does not. Two things
follow from it:

- A gap in the numbers is a lost run of samples of known length. It is filled with
  zeros, so that the OFDM symbol grid continues where it was: the frame
  synchronisation keeps its place, and only the frames in the gap fail, instead of the
  whole chain resynchronising and the time deinterleaver starting from scratch.
- Retuning drops exactly what is of the old frequency (see `retune`). The API delivers
  its packets in bursts, and a change shows in the stream 15 to 78 ms after
  `sdrplay_api_Update` has returned (measured 04.10.2026 on both tuners, 24 to 119
  packets later, for gain and frequency alike): cutting where `Update` returns would
  keep up to 78 ms of the old frequency. The flagged packet was in every case the one
  in which the level stepped, and from the next packet on the level was steady.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import numpy as np

from dab.constants import SAMPLE_RATE
from i18n import tr
from sdr.exceptions import SdrplayError
from sdr.packet_ring import API_PACKET_SAMPLES, FREQUENCY_CHANGED
from sdr.rspduo import DeviceInfo, GainReport, RspDuo, RspDuoSettings

logger = logging.getLogger(__name__)

# The API scales the 14-bit ADC to the full int16 range; full scale is 2**15.
FULL_SCALE = 32768.0

# Discarded after the start (0.5 s). The API's sample counter jumps right after `Init`:
# measured in the GNSS receiver (RSPduo, Tuner 2, 2 MS/s) five times by 9000 to 15000
# samples within the first 0.2 s, afterwards not once in 10 s; in the radio on
# 04.10.2026 (both tuners, 2.048 MS/s) five jumps by 8592 to 18924 samples within the
# first 0.2 s, afterwards none in 9.5 s and none across eight retunes. 1 024 000 samples
# are also what the radio discards after opening the RTL-SDR
# (`sdr.device.DISCARD_AFTER_OPEN_SAMPLES`).
START_DISCARD_SAMPLES = 1_024_000

# Discarded after the packet with which a new frequency took effect (50 ms), for the
# tuner's PLL to settle; what suffices for the RTL-SDR (`sdr.device.RETUNE_SETTLE_SAMPLES`).
# On the RSPduo the level was steady from the next packet on (0.7 ms, see above).
RETUNE_SETTLE_SAMPLES = 102_400

# Should the API not flag a new frequency, it counts as applied this many packets after
# `Update` has returned (0.3 s; at most 119 packets were seen, see above).
FLAG_TIMEOUT_PACKETS = round(0.3 * SAMPLE_RATE / API_PACKET_SAMPLES)

# Largest gap in the API's sample counter that is filled with zeros (1 s). Anything
# larger is not a lost packet but a restart; its length cannot be trusted.
MAX_FILL_SAMPLES = SAMPLE_RATE

# How often a waiting read looks into the ring. A block of two frames takes 192 ms; four
# visits per block keep the hold on the GIL short.
POLL_SECONDS = 0.05

# Without a single packet for this long the stream counts as broken. The API reports a
# removed device itself; this catches a stream that stalls without a word.
NO_DATA_TIMEOUT_SECONDS = 2.0

_MASK = 0xFFFFFFFF


class RspDuoReader:
    """One tuner of the RSPduo as a blocking source of IQ blocks.

    `read_block` runs in one thread (the reading one); `retune` and `cancel` may be
    called from any other. Each block belongs to one tuning, identified by the `tag`
    given to `retune` (the initial one at construction): a block never mixes the
    samples of two frequencies.
    """

    def __init__(
        self,
        settings: RspDuoSettings,
        center_freq_hz: float,
        tag: int = 0,
        on_overload: Callable[[bool], None] | None = None,
        device_factory: Callable[[], RspDuo] = RspDuo,
        poll_seconds: float = POLL_SECONDS,
        no_data_timeout_seconds: float = NO_DATA_TIMEOUT_SECONDS,
    ) -> None:
        self._settings = settings
        self._center_freq_hz = center_freq_hz
        self._on_overload = on_overload
        self._device = device_factory()
        self._poll_seconds = poll_seconds
        self._no_data_timeout = no_data_timeout_seconds
        self._cancelled = threading.Event()
        self._failure: str | None = None
        # Keeps `retune` and `close` apart.
        self._device_lock = threading.Lock()
        # What `retune` tells the reading thread: (tag of the samples from the latest
        # retune on, frequency updates sent to the API so far, packets written when the
        # latest one returned or None while it runs). Replaced as a whole and read once
        # per packet, without a lock: `retune` runs in the GUI thread and must never wait
        # for a reading thread that waits for the GIL.
        self._tuning: tuple[int, int, int | None] = (tag, 0, 0)
        # The reading thread's own state.
        self._tag = tag  # of the samples in `_chunks`
        self._updates_seen = 0  # frequency updates the reading thread knows of
        self._flags_awaited = 0  # updates whose flagged packet has not come yet
        self._discard_remaining = START_DISCARD_SAMPLES
        self._packet = 0  # index of the next packet from the ring
        self._expected: int | None = None  # number of the next sample, as the API counts
        self._chunks: list[np.ndarray] = []  # complex64, not yet scaled
        self._pending = 0  # samples in `_chunks`
        self.lost_samples = 0  # zeros filled in so far
        self._reported_overflows = 0

    @property
    def center_freq(self) -> float:
        return self._center_freq_hz

    @center_freq.setter
    def center_freq(self, center_freq_hz: float) -> None:
        """Retunes, as pyrtlsdr's setter of the same name; for the channel scan."""
        self.retune(center_freq_hz, self._tuning[0] + 1)

    def open(self) -> DeviceInfo:
        """Claims the tuner and starts streaming.

        Raises:
            SdrplayError: if the service, the device or the tuner is not available. The
                device is released again.
        """
        try:
            info = self._device.open(self._settings.tuner)
            self._device.start(
                self._settings,
                self._center_freq_hz,
                on_gain=_log_gain,
                on_overload=self._overload_changed,
                on_failure=self._failed,
            )
        except SdrplayError:
            self._device.close()
            raise
        return info

    def read_block(self, num_samples: int) -> tuple[int, np.ndarray] | None:
        """The next `num_samples` samples of one tuning, with its tag; waits for them.

        Returns:
            (tag, complex64 samples), or None once `cancel` has been called.

        Raises:
            SdrplayError: if the device failed, was removed or delivers nothing.
        """
        deadline = time.monotonic() + self._no_data_timeout
        while True:
            if self._failure is not None:
                raise SdrplayError(self._failure)
            if self._cancelled.is_set():
                return None
            drained = self._device.drain(self._push)
            if self._pending >= num_samples:
                return self._tag, self._take(num_samples)
            self._check_overflows()
            now = time.monotonic()
            if drained:
                deadline = now + self._no_data_timeout
            elif now > deadline:
                raise SdrplayError(
                    tr(
                        "The RSPduo no longer delivers samples. Please check the USB "
                        "connection and restart reception."
                    )
                )
            self._cancelled.wait(self._poll_seconds)

    def read_samples(self, num_samples: int) -> np.ndarray:
        """The next `num_samples` samples, as pyrtlsdr's method of the same name.

        Raises:
            SdrplayError: as `read_block`, and once `cancel` has been called.
        """
        block = self.read_block(num_samples)
        if block is None:
            raise SdrplayError(tr("Reading from the RSPduo cancelled."))
        return block[1]

    def retune(self, center_freq_hz: float, tag: int) -> None:
        """Tunes to another frequency without interrupting the stream; the samples of the
        new frequency carry `tag`.

        May be called from any thread while another one reads; returns as soon as the API
        has taken the frequency (about 1 ms), without waiting for the reading thread.
        That thread drops the samples it has not handed out yet and every packet up to
        the one the API flags as the first of the new frequency, then
        RETUNE_SETTLE_SAMPLES. Updates sent in quick succession are counted, so that
        the flag of an earlier one does not pass for that of the latest.

        Raises:
            SdrplayError: if the API did not take the frequency. No samples are handed
                out from then on; the caller should stop reading.
        """
        with self._device_lock:
            _, updates, _ = self._tuning
            if center_freq_hz == self._center_freq_hz:
                # Nothing to switch; the samples from here on carry the new tag.
                self._tuning = (tag, updates, self._device.packets_written)
                return
            self._tuning = (tag, updates + 1, None)
            self._device.set_frequency(center_freq_hz)
            self._tuning = (tag, updates + 1, self._device.packets_written)
            self._center_freq_hz = center_freq_hz

    def cancel(self) -> None:
        """Makes a waiting `read_block` return None, from any thread."""
        self._cancelled.set()

    def close(self) -> None:
        """Stops streaming and releases the device; safe to call more than once."""
        self._cancelled.set()
        with self._device_lock:
            self._device.close()

    def _push(self, samples: np.ndarray, first_sample_num: int, reset: bool, changed: int) -> None:
        """Takes one packet from the ring (int16, I and Q interleaved)."""
        packet = self._packet
        self._packet += 1
        n = len(samples) // 2
        expected = self._expected
        self._expected = (first_sample_num + n) & _MASK
        if not self._is_new_frequency(packet, changed):
            return
        if expected is not None and not reset:
            gap = (first_sample_num - expected) & _MASK
            if gap:
                self._on_gap(gap)
        elif reset and expected is not None:
            logger.warning("The RSPduo's stream restarted; samples lost in unknown number")

        start = 0
        if self._discard_remaining:
            start = min(n, self._discard_remaining)
            self._discard_remaining -= start
        if start < n:
            chunk = np.empty(n - start, np.complex64)
            chunk.view(np.float32)[:] = samples[2 * start :]
            self._chunks.append(chunk)
            self._pending += n - start

    def _is_new_frequency(self, packet: int, changed: int) -> bool:
        """Whether the packet is of the frequency of the latest retune, past the switch.

        Follows what `retune` has published. A new tag drops the samples not handed out
        yet; a new frequency update is awaited until the API flags the packet with which
        it took effect, or for FLAG_TIMEOUT_PACKETS after `Update` returned. The flagged
        packet itself holds the switch and is dropped.
        """
        tag, updates, cut = self._tuning
        if tag != self._tag:
            self._tag = tag
            self._chunks = []
            self._pending = 0
        if updates != self._updates_seen:
            self._flags_awaited += updates - self._updates_seen
            self._updates_seen = updates
        if not self._flags_awaited:
            return True
        if changed & FREQUENCY_CHANGED:
            self._flags_awaited -= 1
            if not self._flags_awaited:
                self._discard_remaining = max(self._discard_remaining, RETUNE_SETTLE_SAMPLES)
            return False
        if cut is not None and packet >= cut + FLAG_TIMEOUT_PACKETS:
            logger.warning(
                "The RSPduo did not flag the new frequency within %d packets; taking it as applied",
                FLAG_TIMEOUT_PACKETS,
            )
            self._flags_awaited = 0
            self._discard_remaining = max(self._discard_remaining, RETUNE_SETTLE_SAMPLES)
            return True
        return False

    def _on_gap(self, gap: int) -> None:
        if self._discard_remaining:
            return  # still in what is discarded anyway
        if gap > MAX_FILL_SAMPLES:
            logger.warning("The RSPduo's sample counter jumped by %d samples", gap)
            return
        logger.warning("The RSPduo lost %d samples; filled with zeros", gap)
        self.lost_samples += gap
        self._chunks.append(np.zeros(gap, np.complex64))
        self._pending += gap

    def _take(self, num_samples: int) -> np.ndarray:
        samples = np.concatenate(self._chunks)
        self._chunks = [samples[num_samples:]] if len(samples) > num_samples else []
        self._pending = len(samples) - num_samples
        block = samples[:num_samples]
        block *= np.float32(1.0 / FULL_SCALE)
        return block

    def _check_overflows(self) -> None:
        overflows = self._device.overflows
        if overflows > self._reported_overflows:
            self._reported_overflows = overflows
            logger.warning("The packet ring overflowed: %d packets dropped so far", overflows)

    def _overload_changed(self, overloaded: bool) -> None:
        logger.info("RSPduo ADC overload %s", "detected" if overloaded else "corrected")
        if self._on_overload is not None:
            self._on_overload(overloaded)

    def _failed(self, message: str) -> None:
        # From the API's thread; the reading thread raises it.
        self._failure = message
        self._cancelled.set()


def _log_gain(report: GainReport) -> None:
    logger.info(
        "RSPduo gain: IF reduction %d dB, LNA reduction %d dB, system gain %.1f dB",
        report.if_gain_reduction_db,
        report.lna_gain_reduction_db,
        report.system_gain_db,
    )
