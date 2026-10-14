"""Phase reference symbol (PRS) and the timing corrections derived from it."""

from __future__ import annotations

import functools

import numpy as np

# ETSI EN 300 401, table 24: h_i,j values (period 16). Cross-checked against the
# reference implementation welle.io (AlbrechtL/welle.io, phasetable.cpp).
_H = (
    (0, 2, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 2, 2, 1, 1),
    (0, 3, 2, 3, 0, 1, 3, 0, 2, 1, 2, 3, 2, 3, 3, 0),
    (0, 0, 0, 2, 0, 2, 1, 3, 2, 2, 0, 2, 2, 0, 1, 3),
    (0, 1, 2, 1, 0, 3, 3, 2, 2, 3, 2, 1, 2, 1, 3, 2),
)

# ETSI EN 300 401, table 23 (transmission mode I): (k_min, k_max, i, n). Cross-checked
# against welle.io; one entry (385..416) differed after text recognition of the ETSI
# PDF (it read n=3 instead of n=0 there) and was corrected against the reference
# implementation.
# fmt: off
_MODE_I_PHASE_TABLE = (
    (-768, -737, 0, 1), (-736, -705, 1, 2), (-704, -673, 2, 0), (-672, -641, 3, 1),
    (-640, -609, 0, 3), (-608, -577, 1, 2), (-576, -545, 2, 2), (-544, -513, 3, 3),
    (-512, -481, 0, 2), (-480, -449, 1, 1), (-448, -417, 2, 2), (-416, -385, 3, 3),
    (-384, -353, 0, 1), (-352, -321, 1, 2), (-320, -289, 2, 3), (-288, -257, 3, 3),
    (-256, -225, 0, 2), (-224, -193, 1, 2), (-192, -161, 2, 2), (-160, -129, 3, 1),
    (-128, -97, 0, 1), (-96, -65, 1, 3), (-64, -33, 2, 1), (-32, -1, 3, 2),
    (1, 32, 0, 3), (33, 64, 3, 1), (65, 96, 2, 1), (97, 128, 1, 1),
    (129, 160, 0, 2), (161, 192, 3, 2), (193, 224, 2, 1), (225, 256, 1, 0),
    (257, 288, 0, 2), (289, 320, 3, 2), (321, 352, 2, 3), (353, 384, 1, 3),
    (385, 416, 0, 0), (417, 448, 3, 2), (449, 480, 2, 1), (481, 512, 1, 3),
    (513, 544, 0, 3), (545, 576, 3, 3), (577, 608, 2, 3), (609, 640, 1, 0),
    (641, 672, 0, 3), (673, 704, 3, 0), (705, 736, 2, 1), (737, 768, 1, 1),
)
# fmt: on


def _phase_angle(k: int) -> float:
    for k_min, k_max, i, n in _MODE_I_PHASE_TABLE:
        if k_min <= k <= k_max:
            j = (k - k_min) % 16
            return (np.pi / 2) * (_H[i][j] + n)
    raise ValueError(f"Carrier index {k} lies outside the PRS definition range.")


@functools.cache
def physical_carrier_indices(num_carriers: int) -> np.ndarray:
    """Carrier indices k in physical order (as `extract_used_carriers` delivers them).

    For a fixed number of carriers (always `num_carriers=1536` in DAB mode I) this
    returns the same result for the whole run time, so it is cached. The returned
    array must therefore not be modified by the caller.
    """
    half = num_carriers // 2
    return np.concatenate([np.arange(-half, 0), np.arange(1, half + 1)])


@functools.cache
def generate_reference_symbol(num_carriers: int = 1536) -> np.ndarray:
    """Generates the known phase reference symbol (PRS, l=1) in physical carrier order
    (ETSI EN 300 401, clause 14.3.2, transmission mode I).

    Cached for the same reason as `physical_carrier_indices`: the result depends only
    on `num_carriers`, and profiling showed it being recomputed from scratch several
    times per frame (including 1536 table scans in `_phase_angle` each time). The
    returned array must not be modified by the caller.
    """
    k_values = physical_carrier_indices(num_carriers)
    phases = np.array([_phase_angle(int(k)) for k in k_values])
    return np.exp(1j * phases)


@functools.cache
def _reference_symbol_fft_order(num_carriers: int, fft_length: int) -> np.ndarray:
    """The known PRS in native (not fftshift-centred) FFT bin order: bin k for
    k=1..K/2, bin (N-k) for k=-K/2..-1."""
    known = generate_reference_symbol(num_carriers)
    half = num_carriers // 2
    ref = np.zeros(fft_length, dtype=complex)
    ref[1 : half + 1] = known[half:]
    ref[fft_length - half :] = known[:half]
    return ref


def find_integer_symbol_offset(
    window: np.ndarray, num_carriers: int, fft_length: int
) -> tuple[int, float]:
    """Finds the integer sample offset to the start of the useful part of an OFDM
    symbol by FFT/IFFT cross-correlation with the known PRS.

    The channel estimate (spectrum times conjugate known PRS) is transformed back into
    the time domain by IFFT: a pure sample timing error then shows as a sharp peak at
    the position of the error (Fourier shift theorem). Unlike a plain phase slope
    estimate in the frequency domain, this resolves even large (multi-sample) errors
    unambiguously, with no phase wrap-around ambiguity. Analogous to welle.io's
    `PhaseReference::findIndex`.

    `window` must hold at least `fft_length` samples, starting close to (within
    +/- fft_length/2 samples of) the true start of the useful part.

    With multipath propagation (e.g. in an SFN such as the German Bundesmux) the
    correlation can show several peaks, and the strongest does not necessarily
    correspond to the first (correct) path: a later, stronger echo would lead a plain
    maximum search to a wrong but "certain"-looking alignment. Therefore, following
    welle.io's `ThresholdBeforePeak` strategy, the earliest peak whose correlation
    value (not necessarily the largest) exceeds half of the global maximum is taken
    instead of the globally strongest peak.

    Returns:
        (offset, confidence): `offset` is the sample position of the start of the
        useful part relative to the window start; `confidence` is the ratio of peak
        height to mean correlation height (well above 1 for a real match).
    """
    ref = _reference_symbol_fft_order(num_carriers, fft_length)
    spectrum = np.fft.fft(window[:fft_length])
    correlation = np.fft.ifft(spectrum * np.conj(ref))
    magnitude = np.abs(correlation)

    global_max_index = int(np.argmax(magnitude))
    confidence = float(magnitude[global_max_index] / magnitude.mean())

    peak = global_max_index
    if confidence >= 3:
        threshold = magnitude[global_max_index] / 2
        exceeds = np.flatnonzero(magnitude > threshold)
        if len(exceeds) > 0:
            peak = int(exceeds[0])

    offset = peak if peak <= fft_length // 2 else peak - fft_length
    return offset, confidence
