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

It is only sent in every second null symbol (as ODR-DabMod does). Detection here works
on the power per carrier, averaged over the last null symbols: a transmitter's signal
sits on 32 of 1536 carriers, and a single null symbol is not enough for weaker
transmitters. All null symbols are averaged, including those without TII. That costs
3 dB of sensitivity but does not depend on telling the two kinds apart correctly after
a loss of synchronisation.

Paths of the impulse response (`PathTracker`): both carriers of a pair carry the phase
that the phase reference symbol has at the first of them, so the TII carriers of a
transmitter sample its own channel, while the phase reference symbol gives the sum of
all channels. Each peak of the impulse response (`dsp.channel`) is cut out of the
response of the same frame, transformed back to the carriers and compared with the TII
carriers: a path that carries a code shows a gain of about 1 on the 4 blocks of its
pattern and none on the other 4. That names the transmitter behind each path, tells a
further transmitter from a reflection of a known one (the reflection carries the code
of its transmitter), and finds transmitters well below the detection limit of the
power averaging, because the comparison adds up the carriers coherently at a known
delay: a transmitter 34 dB below the strongest is named where the power averaging
stops at -24 dB. A path without a code is a reflection or a transmitter that sends no
TII or too weak a one; the signal cannot tell those two apart.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from dab.constants import MODE_I, SAMPLE_RATE, TransmissionMode
from dsp.channel import PRE_ECHO_SAMPLES, impulse_response
from dsp.ofdm import correct_frequency_offset, demodulate_symbol, extract_used_carriers
from dsp.timing import generate_reference_symbol, physical_carrier_indices

# Number of averaged null symbols (32 frames = 3 s, 16 of them carrying TII).
TII_AVERAGED_FRAMES = 32

# A block counts as occupied when its power exceeds the noise by this much. Without a
# transmitter, all blocks stay within 1 dB of the noise after averaging; a detected
# transmitter lies some 10 dB above it.
TII_THRESHOLD_DB = 3.0

# Mode I: start of the 4 carrier groups (k = 0 is the unused centre).
_GROUP_STARTS = (-768, -384, 1, 385)
_NUM_COMBS = 24
_NUM_BLOCKS = 8
_BLOCK_CARRIERS = 48

# --- Paths of the impulse response ---------------------------------------------------

# Frames over which the paths and their codes are compared (25 s, 128 of them with
# TII). Longer than `TII_AVERAGED_FRAMES`, because a weak transmitter stands out only
# slowly: one at -25 dB lies above `PATH_MIN_Z` in 45 % of the identifications over 128
# frames and in 92 % over 256, where transmitters at -30 dB turn up as well. A strong
# transmitter is named after a few frames all the same.
PATH_AVERAGED_FRAMES = 256

# A peak of the averaged impulse response becomes a path when it lies this far above
# its noise (12 dB) and stays one down to `PATH_KEEP_SNR` (9 dB); at most `MAX_PATHS`
# of them, the strongest.
PATH_MIN_SNR = 16.0
PATH_KEEP_SNR = 8.0
MAX_PATHS = 8

# Half width of the gate that cuts a path out of the impulse response, in samples:
# ±1.5 µs, the main lobe of a single path (`dsp.channel.MAIN_PATH_HALF_WIDTH_SECONDS`).
# Peaks closer together than the gate count as one path.
PATH_GATE_SAMPLES = 3

# A block of a comb carries the path's code when its gain lies this many standard
# deviations of the noise above 0 (`PATH_KEEP_Z` for the code the path carried at the
# last identification) and within `PATH_GAIN_RANGE`: the TII carriers have the
# amplitude of the phase reference symbol, so a path carrying the code shows about 1
# (0.75 to 1.18 on air). Larger gains come from a stronger path that the carriers of
# the block cannot separate from this one, not from a transmitter. In null symbols
# without TII, the weakest of the 4 best blocks of a comb stays below 2.1, and below
# 2.8 for the combs without a code at the paths of a real channel. A path at the limit,
# around 3, showed its code in some identifications and not in others; from 4.5 on
# such a path stays without one, while a transmitter once named stays above
# `PATH_KEEP_Z`.
PATH_MIN_Z = 4.5
PATH_KEEP_Z = 2.0
PATH_GAIN_RANGE = (0.5, 2.0)

# A new code below this many standard deviations shows only once two identifications in
# a row found it at the path: a weak path can stand out with one code in one
# identification and with another in the next. A strong one is named at once.
PATH_SURE_Z = 8.0

# The other 4 blocks of the comb must stay below this share of the weakest of the 4,
# else the pattern, and with it the main identifier, is not certain: a comb can stand
# out at a path with one pattern in one minute and another in the next.
PATH_PATTERN_CONTRAST = 0.5

# Besides the path itself, the fit of a block takes this many of the strongest other
# paths into account. A block has only 8 carriers, and over the frames a static channel
# changes only its common phase, so more unknowns than carriers cannot be told apart;
# the strongest paths are the ones whose leakage matters.
PATH_OTHER_REGRESSORS = 3

# The paths are identified anew every this many frames (about every 2 s), from at
# least `PATH_MIN_FRAMES` frames.
PATH_INTERVAL_FRAMES = 20
PATH_MIN_FRAMES = 8

# A path this close to a stronger one is reported only if it carries a code of its own:
# a peak 2 to 5 µs behind the main path, 25 to 32 dB below it, commonly carries the
# main path's code (likely the filters, not an echo).
NEAR_PATH_SECONDS = 6e-6

# A null symbol carries TII when the carriers of the strongest transmitter hold this
# much more than the median power of all carriers (6 dB).
_TII_FRAME_FACTOR = 4.0

# Frames per part when the sums are built anew (`PathTracker._rebuild`).
_REBUILD_FRAMES = 32

# The cut-out paths are taken back to the carriers through a Tukey window (flat over
# 70 % of the carriers): the Hann window of the display would leave too few carriers
# with enough weight to divide by. Carriers with less weight than `_USABLE_WEIGHT` stay
# out of the comparison.
_TUKEY_ALPHA = 0.3
_USABLE_WEIGHT = 0.3

# The 70 patterns: pattern number -> occupied blocks b = 0..7. The standard lists them
# as a table in ascending order, with block 0 as the most significant bit.
PATTERNS: tuple[tuple[int, ...], ...] = tuple(
    tuple((value >> (7 - b)) & 1 for b in range(_NUM_BLOCKS))
    for value in range(256)
    if value.bit_count() == 4
)
_PATTERN_NUMBERS = {pattern: number for number, pattern in enumerate(PATTERNS)}
# The same by the value of the pattern as a byte (block 0 the most significant bit).
_PATTERN_BY_VALUE = np.zeros(256, dtype=int)
for _number, _pattern in enumerate(PATTERNS):
    _PATTERN_BY_VALUE[
        sum(bit << (_NUM_BLOCKS - 1 - block) for block, bit in enumerate(_pattern))
    ] = _number


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
# The same per block, shape (comb * block, carrier).
_BLOCK_POSITIONS = _PAIR_POSITIONS.reshape(_NUM_COMBS * _NUM_BLOCKS, -1)


@dataclass(frozen=True)
class TiiTransmitter:
    main_id: int  # pattern number p (0..69)
    sub_id: int  # comb number c (0..23)
    level_db: float  # power relative to the strongest detected transmitter
    share: float  # share of the power of all detected transmitters (0..1)
    # Delay of its first path in the impulse response, relative to the path the
    # receiver synchronises to (`dsp.channel`); None while no path carries its code.
    delay_s: float | None = None


@dataclass(frozen=True)
class TiiPath:
    """A peak of the impulse response and the transmitter whose TII it carries."""

    delay_s: float  # relative to the path the receiver synchronises to
    level_db: float  # relative to the strongest path
    transmitter: tuple[int, int] | None  # (main_id, sub_id), None: no code found
    # A later path with the code of an earlier one: a reflection of that transmitter.
    echo: bool = False


@dataclass(frozen=True)
class TiiResult:
    # By power, descending; with those found only at a path of the impulse response.
    transmitters: list[TiiTransmitter]
    # Weakest transmitter still detectable, relative to the strongest detected one
    # (None as long as none is detected).
    detection_limit_db: float | None
    frames: int  # number of averaged null symbols
    # The paths of the impulse response, by delay; empty while no transmitter is known.
    paths: list[TiiPath] = field(default_factory=list)


def null_symbol_window_offset(mode: TransmissionMode) -> int:
    """Start of the FFT window in the null symbol, relative to the start of the PRS.

    The window sits centred in the null symbol: the symbol is 608 samples longer than
    one useful symbol duration, which leaves 304 samples of distance to echoes of the
    neighbouring symbols at the front and at the back.
    """
    return -mode.null_symbol_length + (mode.null_symbol_length - mode.fft_length) // 2


def tii_delay_offset(mode: TransmissionMode) -> int:
    """How many samples later the TII carriers place a path than the impulse response.

    The transmitter builds the null symbol like any other symbol, with the surplus over
    one useful symbol duration as cyclic prefix (ODR-DabMod), so the phase of its
    carriers refers to the end of that prefix; the window starts halfway into it
    (`null_symbol_window_offset`). Mode I: 304 samples, which matches the signal on air
    to a quarter sample.
    """
    surplus = mode.null_symbol_length - mode.fft_length
    return surplus - surplus // 2


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


def null_symbol_carriers(spectrum: np.ndarray, mode: TransmissionMode) -> np.ndarray:
    """The subcarriers of a null symbol spectrum (`null_symbol_spectrum`), in the order
    of `extract_used_carriers`."""
    return extract_used_carriers(spectrum, mode.num_carriers)


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

    strongest = np.argsort(block_power, axis=1)[:, ::-1][:, :4]  # (comb, 4)
    chosen = np.take_along_axis(block_power, strongest, axis=1)
    patterns = _PATTERN_BY_VALUE[(1 << (_NUM_BLOCKS - 1 - strongest)).sum(axis=1)]
    found = [
        (float(chosen[comb].mean()) - noise, int(patterns[comb]), int(comb))
        for comb in np.flatnonzero(chosen.min(axis=1) > threshold)
    ]

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


def code_positions(main_id: int, sub_id: int) -> np.ndarray:
    """Positions of the 32 carriers of a transmitter, in the order of `extract_used_carriers`."""
    blocks = [block for block, used in enumerate(PATTERNS[main_id]) if used]
    return _PAIR_POSITIONS[sub_id][blocks].ravel()


# --- Paths of the impulse response ---------------------------------------------------


@dataclass(frozen=True)
class _TiiFrame:
    """A frame whose null symbol carries TII, as the comparison needs it."""

    number: int  # counted by `PathTracker.add`
    tii: np.ndarray  # null symbol on the time base of the response (`_Geometry`)
    channel: np.ndarray  # channel estimate, usable carriers only (`_Geometry`)
    response: np.ndarray  # `_Geometry.response`
    median_power: float  # median power of the null symbol's carriers


class _Geometry:
    """What the comparison of paths and TII carriers needs (mode I). The carriers are
    held in the order of the blocks (`_BLOCK_POSITIONS`, all 1536 once), so that a
    block is 8 neighbours."""

    def __init__(self, mode: TransmissionMode) -> None:
        k = physical_carrier_indices(mode.num_carriers)
        self.fft_length = mode.fft_length
        self.bins = k % mode.fft_length
        x = np.linspace(0.0, 1.0, mode.num_carriers)
        weight = np.ones(mode.num_carriers)
        edge = _TUKEY_ALPHA / 2
        rising, falling = x < edge, x > 1 - edge
        weight[rising] = 0.5 * (1 + np.cos(np.pi * (x[rising] - edge) / edge))
        weight[falling] = 0.5 * (1 + np.cos(np.pi * (x[falling] - 1 + edge) / edge))
        self.weight = weight
        usable = weight >= _USABLE_WEIGHT
        prs = generate_reference_symbol(mode.num_carriers)
        # Both carriers of a pair carry the phase of the first; the first sits at an
        # even position (the groups start at even positions and a comb steps by 2).
        positions = np.arange(mode.num_carriers)
        reference = prs[positions - positions % 2]
        offset = tii_delay_offset(mode)

        order = _BLOCK_POSITIONS.ravel()
        self.order = order
        self.k = k[order]
        self.usable = usable[order]
        # Undoes the window on the usable carriers and drops the others.
        self.unweight = np.where(usable, 1 / np.maximum(weight, _USABLE_WEIGHT), 0.0)[order]
        # Takes the TII carriers to the time base of the impulse response; drops the
        # carriers the paths leave out.
        self.to_response = (
            np.conj(reference) * np.exp(2j * np.pi * k * offset / mode.fft_length) * usable
        )[order]
        self.gate = np.arange(-PATH_GATE_SAMPLES, PATH_GATE_SAMPLES + 1)
        # A tap of the gate back on the carriers, relative to the path: (gate, carrier).
        self.gate_phases = np.exp(-2j * np.pi * np.outer(self.gate, self.k) / mode.fft_length)

    def response(self, channel_estimate: np.ndarray) -> np.ndarray:
        """Complex impulse response through the Tukey window, cyclic, index = delay."""
        grid = np.zeros(self.fft_length, dtype=complex)
        grid[self.bins] = channel_estimate * self.weight
        return np.fft.ifft(grid)

    def ramps(self, delays: list[int]) -> np.ndarray:
        """Per path its delay on the carriers, window undone: (path, carrier)."""
        return np.exp(-2j * np.pi * np.outer(delays, self.k) / self.fft_length) * self.unweight


_GEOMETRY = _Geometry(MODE_I)


def _block_sums(
    frames: list[_TiiFrame], delays: list[int], ramps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The normal equations of the fit per block, summed over the frames given.

    The unknowns are the paths and, last, the rest of the channel: what no gate takes
    in, above all the side lobes of the strongest path beyond its gate. Without it, a
    path 30 dB weaker can take up the strongest one's code with a gain of 0.8.

    Returns:
        (sum of conj(a) x a, shape (block, unknown, unknown); sum of conj(a) x TII, shape
        (block, unknown)), where `a` is each unknown's share of the channel on the
        block's carriers and TII the carriers of the null symbol.
    """
    geometry = _GEOMETRY
    responses = np.array([frame.response for frame in frames])
    indices = np.add.outer(np.asarray(delays), geometry.gate) % geometry.fft_length
    # Each path's share of the channel per carrier: (frame, path, carrier).
    channels = (responses[:, indices] @ geometry.gate_phases) * ramps
    rest = np.array([frame.channel for frame in frames]) - channels.sum(axis=1)
    unknowns = np.concatenate([channels, rest[:, None]], axis=1)
    blocks = len(_BLOCK_POSITIONS)
    shape = (len(frames), len(delays) + 1, blocks, -1)
    a = unknowns.reshape(shape).transpose(2, 1, 0, 3).reshape(blocks, len(delays) + 1, -1)
    tii = np.array([frame.tii for frame in frames])
    y = tii.reshape(len(frames), blocks, -1).transpose(1, 0, 2).reshape(blocks, -1)
    conjugate = np.conj(a)
    return conjugate @ a.transpose(0, 2, 1), (conjugate @ y[:, :, None])[:, :, 0]


def _path_gains(
    normal: np.ndarray, projected: np.ndarray, noise: float, order: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Gain of every path on the TII carriers of every block, and its noise.

    `order`: the paths strongest first. Per block and path a least-squares fit onto
    this path, the strongest others (`PATH_OTHER_REGRESSORS`) and the rest of the
    channel (`_block_sums`).

    Returns:
        (gain, standard deviation of its noise), each of shape (path, comb, block).
    """
    paths = normal.shape[1] - 1
    rest = paths
    # The unknowns of each path's fit, all of the same number: (path, unknown).
    chosen = np.array(
        [
            [path, *[other for other in order if other != path][:PATH_OTHER_REGRESSORS], rest]
            for path in range(paths)
        ]
    )
    sub = normal[:, chosen[:, :, None], chosen[:, None, :]].transpose(1, 0, 2, 3)
    right = projected[:, chosen].transpose(1, 0, 2)[..., None]
    # A tiny ridge keeps a block without signal (all zero) invertible.
    scale = np.abs(np.einsum("pbii->pb", sub)).max(axis=1)
    sub = sub + np.eye(chosen.shape[1]) * (np.maximum(scale, 1e-300) * 1e-9)[:, None, None, None]
    inverse = np.linalg.inv(sub)
    gains = (inverse @ right)[..., 0, 0]
    deviations = np.sqrt(noise * np.maximum(inverse[..., 0, 0].real, 0.0))
    deviations[scale == 0.0] = np.inf
    shape = (paths, _NUM_COMBS, _NUM_BLOCKS)
    return gains.reshape(shape), deviations.reshape(shape)


def code_strength(gain: np.ndarray, deviation: np.ndarray, code: tuple[int, int]) -> float:
    """How far the 4 blocks of `code` stand above the noise at a path: the least of
    their standard deviations, 0 if a gain lies outside `PATH_GAIN_RANGE`."""
    main_id, comb = code
    blocks = [block for block, used in enumerate(PATTERNS[main_id]) if used]
    magnitude = np.abs(gain[comb, blocks])
    low, high = PATH_GAIN_RANGE
    if not np.all((magnitude >= low) & (magnitude <= high)):
        return 0.0
    with np.errstate(divide="ignore"):
        return float(np.min(magnitude / deviation[comb, blocks]))


def code_of_path(
    gain: np.ndarray,
    deviation: np.ndarray,
    taken: set[tuple[int, int]],
    previous: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """The code a path carries, from its gains per comb and block, or None.

    Per comb the 4 blocks furthest above the noise form the pattern, if every one of
    them is `PATH_MIN_Z` above it and in `PATH_GAIN_RANGE`, and the other 4 stay well
    below (`PATH_PATTERN_CONTRAST`); for the code the path
    carried before (`previous`) `PATH_KEEP_Z` suffices. Should several combs qualify, a
    code that no stronger path carries (`taken`) wins, then the code the path carried
    before. A weaker path can show the code of a stronger one besides its own, about as
    strongly (as from a transmitter fed off air that passes the signal on with its own
    TII added), and leakage from a stronger path whose delay the 8 carriers of a block
    barely tell apart shows its code too.
    """
    magnitude = np.abs(gain)
    z = np.divide(magnitude, deviation, out=np.zeros_like(magnitude), where=deviation > 0)
    ranked = np.argsort(z, axis=1)[:, ::-1]
    strongest = ranked[:, :4]  # (comb, 4)
    least = np.take_along_axis(z, strongest, axis=1).min(axis=1)
    fifth = np.take_along_axis(z, ranked[:, 4:5], axis=1)[:, 0]
    chosen = np.take_along_axis(magnitude, strongest, axis=1)
    low, high = PATH_GAIN_RANGE
    plausible = ((chosen >= low) & (chosen <= high)).all(axis=1) & (
        fifth < least * PATH_PATTERN_CONTRAST
    )
    patterns = _PATTERN_BY_VALUE[(1 << (_NUM_BLOCKS - 1 - strongest)).sum(axis=1)]
    candidates: list[tuple[bool, bool, float, tuple[int, int]]] = []
    for comb in np.flatnonzero(plausible & (least >= PATH_KEEP_Z)):
        code = (int(patterns[comb]), int(comb))
        if least[comb] >= PATH_MIN_Z or code == previous:
            candidates.append((code in taken, code != previous, -float(least[comb]), code))
    if not candidates:
        return None
    return min(candidates)[-1]


def find_paths(power: np.ndarray, kept: Sequence[int] = ()) -> list[int]:
    """Peaks of an averaged impulse response (`dsp.channel.impulse_response`), strongest
    first, as indices into `power`, at least a gate apart.

    A new peak must lie `PATH_MIN_SNR` above the noise; one of `kept` (the paths so far)
    stays while its gate holds `PATH_KEEP_SNR`, at the index it had.
    """
    noise = float(np.median(power)) / np.log(2)
    strength = {
        index: float(power[max(index - 1, 0) : index + 2].max())
        for index in kept
        if power[max(index - 1, 0) : index + 2].max() > noise * PATH_KEEP_SNR
    }
    inner = power[1:-1]
    peaks = (
        np.flatnonzero(
            (inner >= power[:-2]) & (inner >= power[2:]) & (inner > noise * PATH_MIN_SNR)
        )
        + 1
    )
    # The strongest few suffice; side lobes of strong paths make many small peaks.
    for index in peaks[np.argsort(power[peaks])[::-1][: 4 * MAX_PATHS]]:
        if all(abs(int(index) - other) > 2 * PATH_GATE_SAMPLES for other in strength):
            strength[int(index)] = float(power[index])
    chosen: list[int] = []
    for index in sorted(strength, key=lambda index: -strength[index]):
        if all(abs(index - other) > 2 * PATH_GATE_SAMPLES for other in chosen):
            chosen.append(index)
            if len(chosen) == MAX_PATHS:
                break
    return chosen


def _peak_offset(power: np.ndarray, index: int) -> float:
    """Fraction of a sample by which the peak lies beside `index` (parabola in dB)."""
    if index == 0 or index == len(power) - 1:
        return 0.0
    left, centre, right = 10 * np.log10(np.maximum(power[index - 1 : index + 2], 1e-30))
    curvature = left - 2 * centre + right
    if curvature >= 0:
        return 0.0
    return float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))


class PathTracker:
    """The paths of the impulse response over the last `PATH_AVERAGED_FRAMES` frames and
    the codes they carry.

    The normal equations of the fit per block are kept as sums over the null symbols
    with TII in the window: each frame is added when it comes and taken out when it
    leaves, which costs about as much as one frame's share. Only when the set of paths
    changes are the sums built anew from the frames held, which takes tens of
    milliseconds; paths and codes therefore hold with a margin (`PATH_KEEP_SNR`,
    `PATH_KEEP_Z`).
    """

    def __init__(self, sample_rate: float = SAMPLE_RATE) -> None:
        self._sample_rate = sample_rate
        self.reset()

    def reset(self) -> None:
        # The impulse responses as displayed, and their sum.
        self._powers: deque[np.ndarray] = deque()
        self._power_sum: np.ndarray | None = None
        self._frames: deque[_TiiFrame] = deque()
        self._count = 0
        self._indices: list[int] = []  # the paths: indices into the displayed response
        self._ramps: np.ndarray | None = None
        self._normal: np.ndarray | None = None
        self._projected: np.ndarray | None = None
        # The code each path shows, and the one it would show if the next identification
        # agrees (`PATH_SURE_Z`).
        self._codes: dict[int, tuple[int, int]] = {}
        self._candidates: dict[int, tuple[int, int]] = {}
        # Gains of the last identification, (path, comb, block), for `carries`.
        self._gains: np.ndarray | None = None
        self._deviations: np.ndarray | None = None

    def add(
        self,
        null_carriers: np.ndarray,
        channel_estimate: np.ndarray,
        response_power: np.ndarray,
        tii_positions: np.ndarray | None,
    ) -> None:
        """Adds a frame: its null symbol, channel estimate and impulse response as
        displayed (`dsp.channel.impulse_response`). `tii_positions`: the carriers of the
        strongest transmitter detected; the frame counts as one with TII when they stand
        out."""
        self._count += 1
        power = response_power
        self._powers.append(power)
        if self._power_sum is None:
            self._power_sum = power.copy()
        else:
            self._power_sum += power
        if len(self._powers) > PATH_AVERAGED_FRAMES:
            self._power_sum -= self._powers.popleft()
        if tii_positions is not None:
            power = np.abs(null_carriers) ** 2
            median = float(np.median(power))
            if power[tii_positions].mean() > _TII_FRAME_FACTOR * median:
                geometry = _GEOMETRY
                frame = _TiiFrame(
                    self._count,
                    null_carriers[geometry.order] * geometry.to_response,
                    channel_estimate[geometry.order] * geometry.usable,
                    geometry.response(channel_estimate),
                    median,
                )
                self._frames.append(frame)
                self._accumulate([frame], 1)
        while self._frames and self._frames[0].number <= self._count - PATH_AVERAGED_FRAMES:
            self._accumulate([self._frames.popleft()], -1)

    def paths(self) -> list[TiiPath]:
        """The paths by delay, with their codes. Left out is a path near a stronger one
        (`NEAR_PATH_SECONDS`) without a code of its own."""
        if len(self._powers) < PATH_MIN_FRAMES or not self._frames:
            return []
        power = self._power_sum / len(self._powers)
        indices = find_paths(power, self._indices)
        if sorted(indices) != sorted(self._indices):
            self._rebuild(indices)
        if not indices or self._normal is None:
            return []
        # Strongest first.
        order = sorted(range(len(self._indices)), key=lambda path: -power[self._indices[path]])
        noise = float(np.mean([frame.median_power for frame in self._frames])) / np.log(2)
        gains, deviations = _path_gains(self._normal, self._projected, noise, order)

        self._gains, self._deviations = gains, deviations

        codes: dict[int, tuple[int, int] | None] = {}
        candidates: dict[int, tuple[int, int]] = {}
        taken: set[tuple[int, int]] = set()
        for path in order:
            index = self._indices[path]
            previous = self._codes.get(index)
            code = code_of_path(gains[path], deviations[path], taken, previous)
            if (
                code is not None
                and code != previous
                and code_strength(gains[path], deviations[path], code) < PATH_SURE_Z
                and self._candidates.get(index) != code
            ):
                # New and weak: wait for the next identification, and meanwhile keep what
                # the path showed while it still carries that.
                candidates[index] = code
                if previous is None or (
                    code_strength(gains[path], deviations[path], previous) < PATH_KEEP_Z
                ):
                    previous = None
                code = previous
            codes[index] = code
            if code is not None:
                taken.add(code)
        self._codes = {index: code for index, code in codes.items() if code is not None}
        self._candidates = candidates
        return self._label(power, [self._indices[path] for path in order], codes)

    def carries(self, code: tuple[int, int]) -> bool | None:
        """Whether some path carries `code` (`PATH_KEEP_Z`), by the last identification;
        None before the first one."""
        if self._gains is None:
            return None
        return any(
            code_strength(gain, deviation, code) >= PATH_KEEP_Z
            for gain, deviation in zip(self._gains, self._deviations, strict=True)
        )

    def _label(
        self, power: np.ndarray, indices: list[int], codes: dict[int, tuple[int, int] | None]
    ) -> list[TiiPath]:
        strongest = float(power[indices[0]])
        kept: list[tuple[float, float, tuple[int, int] | None]] = []
        for rank, index in enumerate(indices):
            code = codes[index]
            near = [
                other
                for other in indices[:rank]
                if abs(other - index) / self._sample_rate <= NEAR_PATH_SECONDS
            ]
            if any(code is None or code == codes[other] for other in near):
                continue
            delay = (index - PRE_ECHO_SAMPLES + _peak_offset(power, index)) / self._sample_rate
            kept.append((delay, float(10 * np.log10(power[index] / strongest)), code))
        paths: list[TiiPath] = []
        seen: set[tuple[int, int]] = set()
        for delay, level, code in sorted(kept, key=lambda item: item[0]):
            paths.append(TiiPath(delay, level, code, echo=code in seen))
            if code is not None:
                seen.add(code)
        return paths

    def _delays(self) -> list[int]:
        return [index - PRE_ECHO_SAMPLES for index in self._indices]

    def _rebuild(self, indices: list[int]) -> None:
        self._indices = indices
        self._ramps = _GEOMETRY.ramps(self._delays())
        self._normal = self._projected = None
        self._gains = self._deviations = None
        # In parts, which keeps the arrays of a part small.
        frames = list(self._frames)
        for start in range(0, len(frames), _REBUILD_FRAMES):
            self._accumulate(frames[start : start + _REBUILD_FRAMES], 1)

    def _accumulate(self, frames: list[_TiiFrame], sign: int) -> None:
        if not self._indices:
            return
        normal, projected = _block_sums(frames, self._delays(), self._ramps)
        if self._normal is None:
            self._normal, self._projected = normal * sign, projected * sign
        else:
            self._normal += sign * normal
            self._projected += sign * projected


def _with_paths(result: TiiResult, paths: list[TiiPath]) -> TiiResult:
    """The transmitters with the delay of their first path, plus those that only a path
    revealed, and the shares over all of them."""
    first = {path.transmitter: path for path in paths if path.transmitter and not path.echo}
    transmitters = [
        replace(transmitter, delay_s=first[code].delay_s)
        if (code := (transmitter.main_id, transmitter.sub_id)) in first
        else transmitter
        for transmitter in result.transmitters
    ]
    known = {(transmitter.main_id, transmitter.sub_id) for transmitter in transmitters}
    # A path's level counts from the strongest path; the detection levels count from the
    # strongest transmitter, mostly the owner of that path.
    strongest = max(paths, key=lambda path: path.level_db)
    reference = next(
        (
            transmitter.level_db
            for transmitter in transmitters
            if (transmitter.main_id, transmitter.sub_id) == strongest.transmitter
        ),
        0.0,
    )
    for code, path in first.items():
        if code not in known:
            transmitters.append(
                TiiTransmitter(code[0], code[1], reference + path.level_db, 0.0, path.delay_s)
            )
    weights = [10 ** (transmitter.level_db / 10) for transmitter in transmitters]
    total = sum(weights)
    transmitters = sorted(
        (
            replace(transmitter, share=weight / total)
            for transmitter, weight in zip(transmitters, weights, strict=True)
        ),
        key=lambda transmitter: -transmitter.level_db,
    )
    return replace(result, transmitters=transmitters, paths=paths)


class TiiDetector:
    """Averages the null symbols of the last frames and detects the transmitters; with
    the channel estimates of the frames also the transmitter behind each path of the
    impulse response (`PathTracker`)."""

    def __init__(
        self, averaged_frames: int = TII_AVERAGED_FRAMES, sample_rate: float = SAMPLE_RATE
    ) -> None:
        self._averaged_frames = averaged_frames
        self._powers: deque[np.ndarray] = deque()
        self._power_sum: np.ndarray | None = None
        self._sample_rate = sample_rate
        self._tracker = PathTracker(sample_rate)
        self._paths: list[TiiPath] = []
        self._since_paths = 0

    def reset(self) -> None:
        self._powers.clear()
        self._power_sum = None
        self._tracker.reset()
        self._paths = []
        self._since_paths = 0

    def add(
        self,
        null_carriers: np.ndarray,
        channel_estimate: np.ndarray | None = None,
        response_power: np.ndarray | None = None,
    ) -> TiiResult:
        """Adds the null symbol of a frame (`null_symbol_carriers`) and, if given, the
        frame's channel estimate (`dsp.ofdm.DemodulatedFrame.channel_estimate`) and its
        impulse response as displayed (`dsp.channel.impulse_response`, worked out here
        if not given)."""
        power = np.abs(null_carriers) ** 2
        self._powers.append(power)
        if self._power_sum is None:
            self._power_sum = power.copy()
        else:
            self._power_sum += power
        if len(self._powers) > self._averaged_frames:
            self._power_sum -= self._powers.popleft()
        frames = len(self._powers)
        result = detect_transmitters(self._power_sum / frames, frames)
        if channel_estimate is None:
            return result
        strongest = result.transmitters[0] if result.transmitters else None
        if response_power is None:
            response_power = impulse_response(channel_estimate, MODE_I, self._sample_rate)[1]
        self._tracker.add(
            null_carriers,
            channel_estimate,
            response_power,
            None if strongest is None else code_positions(strongest.main_id, strongest.sub_id),
        )
        self._since_paths += 1
        if strongest is None:
            self._paths = []
        elif not self._paths or self._since_paths >= PATH_INTERVAL_FRAMES:
            self._paths = self._tracker.paths()
            self._since_paths = 0
        if not self._paths:
            return result
        # A transmitter the power averaging detects stands out in the impulse response
        # by far (the phase reference adds up all 1536 carriers), so a path carries its
        # code; one that no path carries is not a transmitter (an image or a spur in
        # the null symbol).
        confirmed = [
            transmitter
            for transmitter in result.transmitters
            if self._tracker.carries((transmitter.main_id, transmitter.sub_id)) is not False
        ]
        return _with_paths(replace(result, transmitters=confirmed), self._paths)
