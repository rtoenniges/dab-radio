"""QPSK demapping of the differentially demodulated subcarriers."""

from __future__ import annotations

import numpy as np


def demap_soft_bits(logical_symbols: np.ndarray) -> np.ndarray:
    """Demaps complex QPSK symbols (in logical carrier order) to soft bits.

    ETSI EN 300 401, clause 14.5 (QPSK symbol mapper):
        q[n] = (1/sqrt(2)) * [(1 - 2*p[n]) + j*(1 - 2*p[n+K])]

    A bit value of 0 therefore corresponds to a positive, a bit value of 1 to a negative
    real or imaginary part. For `K = len(logical_symbols)` symbols the result has length
    `2*K`: first all K "real part bits" p[0..K-1], then all K "imaginary part bits"
    p[K..2K-1], not interleaved bit by bit.

    The soft bit is the real or imaginary part itself: its sign is the bit, its
    magnitude the confidence (see `dsp.convolutional.viterbi_decode`). For D-QPSK
    (`z(l) * conj(z(l-1))`) that magnitude grows with the square of the carrier's
    channel gain, so carriers in a notch of the channel or drowned in noise
    automatically count for little. Deliberately not normalised per symbol or carrier:
    that would throw exactly this weighting away. It assumes the same noise on every
    carrier, though; `CarrierReliability` supplies the factor per carrier where that
    does not hold, applied to the points before demapping.

    Args:
        logical_symbols: complex (differentially demodulated) QPSK symbols in logical
            order n=0..K-1 (see `dsp.deinterleave`).

    Returns:
        float32 array of length `2*len(logical_symbols)`.
    """
    return np.concatenate([logical_symbols.real, logical_symbols.imag]).astype(np.float32)


def hard_decisions(soft_bits: np.ndarray) -> np.ndarray:
    """The bits the soft bits stand for (negative = 1), as uint8."""
    return (soft_bits < 0).astype(np.uint8)


# Share of each new frame in the running estimate of `CarrierReliability`: about the
# last 10 frames (1 s) count.
RELIABILITY_SMOOTHING = 0.1

# Symbols per frame that go into the estimate: the FIC and the first MSC symbols, as
# many as the scan demodulates. With the smoothing that is some 160 points per carrier,
# and all 75 data symbols would cost 20 times as much (they no longer fit the cache).
RELIABILITY_SYMBOLS = 16


class CarrierReliability:
    """How far the soft bits of each carrier can be trusted, estimated from the D-QPSK
    points of the frames so far.

    `demap_soft_bits` takes the magnitude of a point as its confidence, which assumes the
    same noise on every carrier. A narrow interferer in the band breaks that: its
    carrier comes out strong while its phase is random, so its bits look surer than any
    other, and a single one of them can outweigh the free distance of the convolutional
    code. Its sidelobes spill over onto the neighbouring carriers as well.

    Turned onto its nearest ideal point, a D-QPSK point keeps the signal in its real part
    and nothing but noise in its imaginary part. Per carrier, the mean real part over the
    mean square of the imaginary part is the factor that turns its soft bits into
    log-likelihood ratios: for a strong carrier simply one over its noise, a weak one
    counts less still. The estimate comes from the data symbols, so the TII carriers of
    the null symbol do not disturb it.
    """

    def __init__(self) -> None:
        self._signal: np.ndarray | None = None
        self._noise: np.ndarray | None = None
        self._weights: np.ndarray | None = None

    def reset(self) -> None:
        """Forgets the estimate, e.g. after a channel change."""
        self._signal = None
        self._noise = None
        self._weights = None

    def update(self, points: np.ndarray) -> None:
        """Adds the D-QPSK points of a frame: one row per symbol, physical carrier order.

        The nearest ideal point of a + jb is sign(a) + j·sign(b); turned onto it, the
        point has the real part |a| + |b| and the imaginary part ±(|a| - |b|). Written
        that way it takes a third of the time of the complex product.
        """
        a = np.abs(points.real)
        b = np.abs(points.imag)
        signal = (a + b).mean(axis=0)
        noise = np.square(a - b).mean(axis=0)
        if self._signal is None or self._noise is None:
            self._signal, self._noise = signal, noise
        else:
            self._signal += RELIABILITY_SMOOTHING * (signal - self._signal)
            self._noise += RELIABILITY_SMOOTHING * (noise - self._noise)
        # More than 60 dB counts as 60 dB, so that noise-free points (synthetic signals)
        # do not divide by zero.
        floor = (1e-3 * float(self._signal.mean())) ** 2
        if floor == 0:
            self._weights = None
            return
        weights = self._signal / np.maximum(self._noise, floor)
        # Scaled to a median of 1, so that the soft bits keep their size.
        self._weights = weights / np.median(weights)

    @property
    def weights(self) -> np.ndarray | None:
        """The factor per carrier (physical order, median 1), or None while there is no
        estimate yet."""
        return self._weights
