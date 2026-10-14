"""Transmitter Identification Information (TII).

ETSI EN 300 401, clause 14.8. In the null symbol at the start of the frame, every
transmitter of a single-frequency network switches on a few carrier pairs whose
positions reveal its identity. In mode I the 1536 carriers form 4 groups of 8 blocks
with 48 carriers each:

* The comb number c (sub-identifier, 0..23) selects in every block the pair k, k+1
  with k = group start + 2c + 48b.
* The pattern number p (main identifier, 0..69) selects which 4 of the 8 blocks are
  occupied; 70 is the number of ways to choose 4 out of 8.

The pattern repeats in all 4 groups, so a transmitter occupies 16 pairs with 32
carriers. Each carrier has the same power as in the other symbols, so the signal as a
whole lies 17 dB below them. Structure and pattern table are checked against
ODR-DabMod (src/TII.cpp), which generates the signal on the transmitter side.

It is only sent in every second null symbol (as ODR-DabMod does; confirmed on air on
5C). Detection here works on the power per carrier, averaged over the last null
symbols: a transmitter's signal sits on 32 of 1536 carriers, and a single null symbol
is not enough for weaker transmitters. All null symbols are averaged, including those
without TII. That costs 3 dB of sensitivity but does not depend on telling the two
kinds apart correctly after a loss of synchronisation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from dab.constants import TransmissionMode
from dsp.ofdm import correct_frequency_offset, demodulate_symbol, extract_used_carriers

# Number of averaged null symbols (32 frames = 3 s, 16 of them carrying TII).
TII_AVERAGED_FRAMES = 32

# A block counts as occupied when its power exceeds the noise by this much. Without a
# transmitter, all blocks on real reception stayed within 1 dB of the noise after
# averaging; the detected transmitter was 10 dB above it.
TII_THRESHOLD_DB = 3.0

# Mode I: start of the 4 carrier groups (k = 0 is the unused centre).
_GROUP_STARTS = (-768, -384, 1, 385)
_NUM_COMBS = 24
_NUM_BLOCKS = 8
_BLOCK_CARRIERS = 48

# The 70 patterns: pattern number -> occupied blocks b = 0..7. The standard lists them
# as a table in ascending order, with block 0 as the most significant bit.
PATTERNS: tuple[tuple[int, ...], ...] = tuple(
    tuple((value >> (7 - b)) & 1 for b in range(_NUM_BLOCKS))
    for value in range(256)
    if value.bit_count() == 4
)
_PATTERN_NUMBERS = {pattern: number for number, pattern in enumerate(PATTERNS)}


def _carrier_position(k: int) -> int:
    """Position of carrier k in the order of `extract_used_carriers`."""
    return k + 768 if k < 0 else k + 767


# Positions of all possible pair carriers, shape (comb, block, group, pair).
_PAIR_POSITIONS = np.array(
    [
        [
            [
                [_carrier_position(start + 2 * comb + _BLOCK_CARRIERS * block + j) for j in (0, 1)]
                for start in _GROUP_STARTS
            ]
            for block in range(_NUM_BLOCKS)
        ]
        for comb in range(_NUM_COMBS)
    ]
)


@dataclass(frozen=True)
class TiiTransmitter:
    main_id: int  # pattern number p (0..69)
    sub_id: int  # comb number c (0..23)
    level_db: float  # power relative to the strongest detected transmitter
    share: float  # share of the power of all detected transmitters (0..1)


@dataclass(frozen=True)
class TiiResult:
    transmitters: list[TiiTransmitter]  # by power, descending
    # Weakest transmitter still detectable, relative to the strongest detected one
    # (None as long as none is detected).
    detection_limit_db: float | None
    frames: int  # number of averaged null symbols


def null_symbol_window_offset(mode: TransmissionMode) -> int:
    """Start of the FFT window in the null symbol, relative to the start of the PRS.

    The window sits centred in the null symbol: the symbol is 608 samples longer than
    one useful symbol duration, which leaves 304 samples of distance to echoes of the
    neighbouring symbols at the front and at the back.
    """
    return -mode.null_symbol_length + (mode.null_symbol_length - mode.fft_length) // 2


def null_symbol_spectrum(
    samples: np.ndarray,
    frame_start: int,
    freq_offset_hz: float,
    sample_rate: float,
    mode: TransmissionMode,
) -> np.ndarray | None:
    """FFT of the null symbol preceding the frame, all `fft_length` bins, DC in the middle.

    Args:
        frame_start: position of the phase reference symbol in `samples`, as used by
            the demodulation after the fine alignment.
        freq_offset_hz: the frequency offset that was rotated out there.

    Returns:
        The spectrum, or None if the null symbol does not lie entirely in the buffer.
    """
    start = frame_start + null_symbol_window_offset(mode)
    if start < 0 or start + mode.fft_length > len(samples):
        return None
    window = correct_frequency_offset(
        samples[start : start + mode.fft_length], freq_offset_hz, sample_rate
    )
    return demodulate_symbol(window, mode.fft_length, 0)


def null_symbol_carrier_power(
    samples: np.ndarray,
    frame_start: int,
    freq_offset_hz: float,
    sample_rate: float,
    mode: TransmissionMode,
) -> np.ndarray | None:
    """Power per subcarrier in the null symbol preceding the frame (see `null_symbol_spectrum`).

    Returns:
        Power per carrier in the order of `extract_used_carriers`, or None if the null
        symbol does not lie entirely in the buffer.
    """
    spectrum = null_symbol_spectrum(samples, frame_start, freq_offset_hz, sample_rate, mode)
    if spectrum is None:
        return None
    return np.abs(extract_used_carriers(spectrum, mode.num_carriers)) ** 2


def detect_transmitters(carrier_power: np.ndarray, frames: int) -> TiiResult:
    """Detects the transmitters from the averaged power per carrier.

    Per comb the 4 strongest blocks count; if even the weakest of them lies
    `TII_THRESHOLD_DB` above the noise, they form the pattern. The noise is the median
    over all blocks of all combs, of which only few are occupied. Limitation: two
    transmitters with the same sub-identifier and different main identifiers overlap
    in the same comb; only the pattern of the 4 strongest blocks is detected then.
    """
    block_power = carrier_power[_PAIR_POSITIONS].mean(axis=(2, 3))  # (comb, block)
    noise = float(np.median(block_power))
    threshold = noise * 10 ** (TII_THRESHOLD_DB / 10)

    found: list[tuple[float, int, int]] = []
    for comb in range(_NUM_COMBS):
        strongest = np.argsort(block_power[comb])[::-1][:4]
        if block_power[comb, strongest].min() <= threshold:
            continue
        pattern = tuple(int(b in strongest) for b in range(_NUM_BLOCKS))
        power = float(block_power[comb, strongest].mean()) - noise
        found.append((power, _PATTERN_NUMBERS[pattern], comb))

    if not found:
        return TiiResult([], None, frames)
    found.sort(reverse=True)
    strongest_power = found[0][0]
    total = sum(power for power, _, _ in found)
    transmitters = [
        TiiTransmitter(
            main_id, sub_id, float(10 * np.log10(power / strongest_power)), power / total
        )
        for power, main_id, sub_id in found
    ]
    limit = 10 * np.log10((threshold - noise) / strongest_power)
    return TiiResult(transmitters, float(limit), frames)


class TiiDetector:
    """Averages the null symbols of the last frames and detects the transmitters."""

    def __init__(self, averaged_frames: int = TII_AVERAGED_FRAMES) -> None:
        self._history: deque[np.ndarray] = deque(maxlen=averaged_frames)

    def reset(self) -> None:
        self._history.clear()

    def add(self, carrier_power: np.ndarray) -> TiiResult:
        self._history.append(carrier_power)
        return detect_transmitters(np.mean(self._history, axis=0), len(self._history))
