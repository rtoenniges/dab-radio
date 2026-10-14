"""Probing a channel for a DAB ensemble: the signal processing side of the scanner."""

from __future__ import annotations

import time

import numpy as np

from dab.constants import TransmissionMode
from dab.ensemble_cache import SOURCE_SCAN, EnsembleInfo
from dab.fig import Ensemble, parse_fib
from dsp.fic import check_fib_crc, extract_fibs_from_symbols
from dsp.ofdm import FrameDemodulator
from dsp.sync import find_null_symbol

# This many CRC-valid FIBs make a channel count as receivable. Pure noise passes the
# 16-bit CRC with a probability of 1/65536 per FIB. On an empty channel about 70 FIBs
# are checked within `NO_SIGNAL_SECONDS`, so a good 2700 across the 38 channels of a
# scan: with a single hit as the criterion, roughly every 25th scan would contain a
# ghost channel. Two hits on the same empty channel are practically impossible, while
# a real ensemble delivers up to 12 valid FIBs in a single frame.
MIN_VALID_FIBS = 2

# If no ensemble shows up within this much signal, the channel counts as empty. That is
# enough for six decoded frames (see `sdr.scanner.SCAN_BLOCK_SAMPLES`), so even a weak
# channel on which only isolated frames come through error-free is noticed. This wait
# is spent on every empty channel and thus almost alone determines the scan duration.
NO_SIGNAL_SECONDS = 0.75

# Minimum time a found ensemble is listened to before its service list may count as
# complete. The check is whether every service from the service organisation
# (FIG 0/2) has a name (FIG 1/1), which presupposes that FIG 0/2 itself is complete,
# and for large ensembles that one is spread over several FIBs.
MIN_LISTEN_SECONDS = 1.0

# Maximum time an ensemble is listened to. If names are still missing after that,
# whatever has arrived by then is stored.
MAX_LISTEN_SECONDS = 10.0

# Demodulated data symbols per frame, as in reception without a selected service
# (`dsp.pipeline.DspWorker`). Only the first three carry the FIC; the rest improve
# the estimate of the integer frequency offset, which averages over the symbols at
# the start of the frame (see `dsp.ofdm.demodulate_frame_symbols`).
PROBE_DATA_SYMBOLS = 16


def service_list_complete(ensemble: Ensemble) -> bool:
    """Every service has a name and, if the ensemble signals programme types at all,
    a programme type. FIG 0/17 is optional; an ensemble without it must not hold up
    the scan until `MAX_LISTEN_SECONDS`."""
    return (
        ensemble.label is not None
        and bool(ensemble.audio_components)
        and all(sid in ensemble.services for sid in ensemble.audio_components)
        and (
            not ensemble.programme_types
            or all(sid in ensemble.programme_types for sid in ensemble.audio_components)
        )
    )


def service_list_settled(ensemble: Ensemble, listened_seconds: float) -> bool:
    """Whether the service list counts as complete after `listened_seconds` of signal
    since the ensemble was confirmed (`MIN_VALID_FIBS`): for the scan, and during
    reception for the surfaces that wait for the whole list (`dsp.pipeline`)."""
    return listened_seconds >= MAX_LISTEN_SECONDS or (
        listened_seconds >= MIN_LISTEN_SECONDS and service_list_complete(ensemble)
    )


class EnsembleProbe:
    """Checks a channel for a DAB ensemble and collects its service list.

    The core of the channel scan (`sdr.scanner.ChannelScanner`). Decodes only the FIC
    of each frame, with the same functions as reception (`dsp.pipeline.DspWorker`): a
    channel therefore counts as receivable exactly when reception there could read the
    ensemble and its services. The criterion is CRC-valid FIBs and not the signal
    level: a strong neighbouring channel or an interferer raises the level just the
    same, and only decodable data prove an ensemble.

    Every block is evaluated on its own. Gaps between blocks therefore do no harm,
    unlike for the MSC time deinterleaver during reception.
    """

    def __init__(self, mode: TransmissionMode, sample_rate: float) -> None:
        self._mode = mode
        self._sample_rate = sample_rate
        self._demodulator = FrameDemodulator(mode, sample_rate)
        self.ensemble = Ensemble()
        self.valid_fibs = 0
        self.total_fibs = 0
        self._seconds = 0.0  # signal duration evaluated so far
        self._found_at: float | None = None  # signal duration at which the ensemble was confirmed

    @property
    def receivable(self) -> bool:
        return self.valid_fibs >= MIN_VALID_FIBS

    @property
    def finished(self) -> bool:
        """True once it is settled that the channel is empty, or the service list is
        complete (or the maximum listening time has been reached)."""
        if self._found_at is None:
            return self._seconds >= NO_SIGNAL_SECONDS
        return service_list_settled(self.ensemble, self._seconds - self._found_at)

    def process(self, samples: np.ndarray) -> None:
        """Evaluates one block; it should span several frame lengths.

        One frame is decoded from every excerpt of two frame lengths: the null symbol
        is only searched within the first frame length, so a complete frame is
        guaranteed to follow (as during acquisition in
        `dsp.pipeline.DspWorker._locate_next_null`). The excerpts advance by one frame
        length each, so a block of n frame lengths yields n-1 frames.
        """
        frame_length = self._mode.frame_length
        for start in range(0, len(samples) - 2 * frame_length + 1, frame_length):
            self._decode_frame(samples[start : start + 2 * frame_length])
        self._seconds += len(samples) / self._sample_rate
        if self._found_at is None and self.receivable:
            self._found_at = self._seconds

    def result(self) -> EnsembleInfo | None:
        """The ensemble found, or None if the channel is empty."""
        if not self.receivable:
            return None
        services = sorted(self.ensemble.services.items(), key=lambda item: item[1])
        return EnsembleInfo(
            label=self.ensemble.label,
            services=tuple(services),
            updated_at=time.time(),
            source=SOURCE_SCAN,
            fib_success=self.valid_fibs / self.total_fibs,
            programme_types=tuple(sorted(self.ensemble.programme_types.items())),
            eid=self.ensemble.eid,
            # FIG 0/21 and 0/24 are repeated slowly: what the scan does not wait for is
            # missing.
            frequencies=tuple(sorted(self.ensemble.frequencies.items())),
            service_ensembles=tuple(sorted(self.ensemble.service_ensembles.items())),
        )

    def _decode_frame(self, samples: np.ndarray) -> None:
        mode = self._mode
        null_index = find_null_symbol(samples, mode.null_symbol_length, max_start=mode.frame_length)
        frame = self._demodulator.demodulate(samples, null_index, PROBE_DATA_SYMBOLS)
        if frame is None or len(frame.symbols) < 3:
            return

        for fib in extract_fibs_from_symbols(frame.symbols[:3], mode.num_carriers, mode.fft_length):
            self.total_fibs += 1
            if check_fib_crc(fib):
                self.valid_fibs += 1
                parse_fib(fib, self.ensemble)
