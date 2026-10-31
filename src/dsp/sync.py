"""Null symbol search: locating the start of a transmission frame."""

from __future__ import annotations

import functools

import numpy as np

from dab.constants import TransmissionMode

# Low-pass in front of the search, so that it sees only the band of the ensemble's
# carriers (+/- 768 kHz in mode I). The 2.048 MHz sampled also hold the edges of the
# neighbouring channels, whose carriers begin 944 kHz from the centre, and those can be
# stronger than the wanted ensemble. In the total power their null symbols, at a frame
# clock of their own, can then make the deeper minimum, and the search would lock onto a
# neighbour. A windowed sinc (Kaiser): flat to 614 kHz, -6 dB at the outermost carriers,
# at least 54 dB down from 880 kHz on.
_BAND_FILTER_TAPS = 31
_BAND_FILTER_BETA = 5.0

# Number of null symbol candidates tried during acquisition (`null_symbol_candidates`):
# the wanted ensemble's plus the pauses of up to two interferers with a frame clock of
# their own, such as the neighbours below and above.
NULL_CANDIDATES = 3

# Besides the deepest power minimum, only minima at least this far below the mean are
# candidates: noise and OFDM symbols alone dip by up to about 0.4 dB within a frame, and
# the acquisition need not check each of those for a PRS. A null symbol is shallower
# than this only where the ensemble makes up less than an eighth of the power in its
# band.
NULL_CANDIDATE_MIN_DIP_DB = 0.6


@functools.cache
def _band_filter(mode: TransmissionMode) -> np.ndarray:
    """Taps of the low-pass that keeps only the band of the carriers (see
    `_BAND_FILTER_TAPS`). Single precision, so that complex64 samples stay complex64."""
    edge = mode.num_carriers / (2 * mode.fft_length)  # outermost carrier, cycles per sample
    n = np.arange(_BAND_FILTER_TAPS) - (_BAND_FILTER_TAPS - 1) / 2
    taps = 2 * edge * np.sinc(2 * edge * n) * np.kaiser(_BAND_FILTER_TAPS, _BAND_FILTER_BETA)
    return (taps / taps.sum()).astype(np.float32)


def _window_power(samples: np.ndarray, mode: TransmissionMode, max_start: int | None) -> np.ndarray:
    """Power within the band of the carriers, summed over one null symbol length, for
    every candidate start position (at most `max_start` of them)."""
    length = mode.null_symbol_length
    if len(samples) < length:
        raise ValueError("Sample buffer is shorter than a null symbol.")
    if max_start is not None:
        samples = samples[: max(max_start, 1) + length - 1]

    power = np.abs(np.convolve(samples, _band_filter(mode), mode="same")) ** 2
    cumulative = np.cumsum(np.insert(power, 0, 0.0))
    return cumulative[length:] - cumulative[:-length]


def find_null_symbol(
    samples: np.ndarray, mode: TransmissionMode, max_start: int | None = None
) -> int:
    """Finds the start index of the DAB null symbol by searching for the power minimum.

    The null symbol marks the start of a transmission frame with a brief transmission
    pause (markedly lower power than the OFDM symbols that follow). A sliding window
    looks for the position with the lowest mean power in the sample buffer. Only the
    band of the ensemble's carriers counts (see `_BAND_FILTER_TAPS`).

    `max_start` restricts the search to the first `max_start` candidate start
    positions. This matters for a buffer spanning more than one frame (e.g. to
    guarantee enough samples for a complete MSC decode): without the limit a plain
    global minimum search would occasionally pick the LATER of the (then several) null
    symbols in the buffer, leaving too few samples for a complete frame after it.

    Raises:
        ValueError: if the buffer is shorter than a null symbol.
    """
    return int(np.argmin(_window_power(samples, mode, max_start)))


def null_symbol_candidates(
    samples: np.ndarray, mode: TransmissionMode, max_start: int | None = None
) -> list[int]:
    """Start indices of the power minima that may be the null symbol, deepest first.

    The first is always the one `find_null_symbol` returns. Up to `NULL_CANDIDATES - 1`
    further minima follow if they lie at least `NULL_CANDIDATE_MIN_DIP_DB` below the
    mean, each with its window clear of the windows before it. Even within the band, the
    deepest minimum need not belong to the wanted ensemble: an interferer with pauses of
    its own that is stronger there makes a deeper one. The acquisition
    (`dsp.ofdm.FrameDemodulator.acquire`) therefore checks the candidates for the PRS.

    Raises:
        ValueError: if the buffer is shorter than a null symbol.
    """
    window_sum = _window_power(samples, mode, max_start).astype(np.float64)
    threshold = window_sum.mean() * 10 ** (-NULL_CANDIDATE_MIN_DIP_DB / 10)
    length = mode.null_symbol_length
    candidates: list[int] = []
    while len(candidates) < NULL_CANDIDATES:
        index = int(np.argmin(window_sum))
        if candidates and window_sum[index] > threshold:
            break
        candidates.append(index)
        window_sum[max(0, index - length + 1) : index + length] = np.inf
    return candidates
