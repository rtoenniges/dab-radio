"""Channel scan: tuning through Band III and probing every channel for an ensemble."""

from __future__ import annotations

import logging
from collections.abc import Callable

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from rtlsdr import RtlSdr

from dab.constants import MODE_I, SAMPLE_RATE
from dab.ensemble_cache import EnsembleInfo
from dsp.scan import EnsembleProbe
from dsp.spectrum import compute_power_spectrum
from i18n import exception_text, tr
from sdr.device import DISCARD_AFTER_OPEN_SAMPLES, RETUNE_SETTLE_SAMPLES, open_rtlsdr
from sdr.exceptions import DeviceError
from sdr.rspduo import RspDuoSettings
from sdr.rspduo_reader import RspDuoReader

logger = logging.getLogger(__name__)

# Read block of the scan: four frames (~384 ms). Three frames are decoded from it (see
# `dsp.scan.EnsembleProbe.process`); with two frames as in reception
# (`sdr.worker.DEFAULT_READ_SIZE`) it would be only one, and every empty channel
# would hold up the scan twice as long. 4 * 196 608 samples are 1 572 864 bytes, a
# multiple of the 512-byte USB packet size, so no read ends in the middle of a packet.
# With RAW_IO (see `sdr.library`) WinUSB rejects any other length. The RSPduo takes any.
SCAN_BLOCK_SAMPLES = 4 * MODE_I.frame_length


class ChannelScanner(QThread):
    """Channel scan: tunes through all channels one after another and probes each for
    a DAB ensemble (see `dsp.scan.EnsembleProbe`).

    Reads synchronously, unlike reception (`sdr.worker.RtlSdrWorker`). Reception needs
    a gapless sample stream, the scan does not: it evaluates every frame on its own.
    Synchronous reads allow retuning safely between two blocks, and cancelling avoids
    the delicate shutdown of the asynchronous read (see `RtlSdrWorker._cancel_read`).

    With `rspduo` set it scans with that tuner of the RSPduo instead of the stick, read
    through `sdr.rspduo_reader.RspDuoReader` with the same calls; `gain`,
    `freq_correction_ppm` and `device_index` then do not apply.
    """

    channel_started = pyqtSignal(
        str, int, int
    )  # channel, running number (from 0), number of channels
    # channel, valid FIBs, checked FIBs, ensemble label (str|None), number of named services
    channel_progress = pyqtSignal(str, int, int, object, int)
    channel_scanned = pyqtSignal(str, object)  # channel, EnsembleInfo or None (no ensemble)
    spectrum_ready = pyqtSignal(object, object)  # freqs (Hz), power (dB) of the most recent block
    error = pyqtSignal(object)  # a text, see `RtlSdrWorker.connected`
    scan_finished = pyqtSignal(bool)  # True if all channels were scanned

    def __init__(
        self,
        channels: dict[str, float],
        sample_rate: float = SAMPLE_RATE,
        gain: str | float = "auto",
        freq_correction_ppm: int = 0,
        device_index: int = 0,
        rspduo: RspDuoSettings | None = None,
        rspduo_reader_factory: Callable[..., RspDuoReader] = RspDuoReader,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._channels = dict(channels)  # channel -> centre frequency in MHz
        self._sample_rate = sample_rate
        self._gain = gain
        self._freq_correction_ppm = freq_correction_ppm
        self._device_index = device_index
        self._rspduo = rspduo
        self._rspduo_reader_factory = rspduo_reader_factory
        self._running = False

    def run(self) -> None:
        self._running = True
        first_freq_mhz = next(iter(self._channels.values()))
        try:
            sdr = self._open(round(first_freq_mhz * 1e6))
        except DeviceError as exc:
            self.error.emit(exception_text(exc))
            self.scan_finished.emit(False)
            return

        completed = False
        try:
            # Otherwise services of the channel received last would end up in the
            # service list of the first one (see DISCARD_AFTER_OPEN_SAMPLES). The RSPduo
            # has discarded its start already; another 0.5 s cost nothing in a scan.
            sdr.read_samples(DISCARD_AFTER_OPEN_SAMPLES)
            completed = self._scan_all(sdr)
        except Exception as exc:  # noqa: BLE001 - hardware errors must not crash the thread
            self.error.emit(tr("Error during the scan: {error}").format(error=exc))
        finally:
            try:
                sdr.close()
            except Exception:
                logger.exception("Failed to close the device")
        self.scan_finished.emit(completed)

    def _open(self, center_freq: int) -> RtlSdr | RspDuoReader:
        """Opens the stick or the RSPduo, tuned to `center_freq`.

        Raises:
            DeviceError: if the device is missing, busy or cannot be configured.
        """
        if self._rspduo is None:
            return open_rtlsdr(
                center_freq=center_freq,
                sample_rate=self._sample_rate,
                gain=self._gain,
                freq_correction_ppm=self._freq_correction_ppm,
                device_index=self._device_index,
            )
        reader = self._rspduo_reader_factory(self._rspduo, center_freq)
        reader.open()
        return reader

    def cancel(self) -> None:
        """Cancels the scan after the block currently being read, without waiting."""
        self._running = False

    def stop(self) -> None:
        """Cancels the scan and waits until the device is closed."""
        self.cancel()
        self.wait(5000)

    def _scan_all(self, sdr: RtlSdr | RspDuoReader) -> bool:
        total = len(self._channels)
        for index, (channel, freq_mhz) in enumerate(self._channels.items()):
            if not self._running:
                return False
            self.channel_started.emit(channel, index, total)
            info = self._scan_channel(sdr, channel, round(freq_mhz * 1e6))
            if not self._running:
                # Cancelled in the middle of a channel: the result is incomplete and
                # would corrupt a stored entry.
                return False
            self.channel_scanned.emit(channel, info)
        return True

    def _scan_channel(
        self, sdr: RtlSdr | RspDuoReader, channel: str, center_freq: int
    ) -> EnsembleInfo | None:
        sdr.center_freq = center_freq
        # Across all 38 channels this costs less than two seconds.
        sdr.read_samples(RETUNE_SETTLE_SAMPLES)
        probe = EnsembleProbe(MODE_I, self._sample_rate)
        while self._running and not probe.finished:
            samples = sdr.read_samples(SCAN_BLOCK_SAMPLES)
            self.spectrum_ready.emit(*compute_power_spectrum(samples, self._sample_rate))
            probe.process(samples)
            self.channel_progress.emit(
                channel,
                probe.valid_fibs,
                probe.total_fibs,
                probe.ensemble.label,
                len(probe.ensemble.services),
            )
        return probe.result()
