"""Channel impulse response and single-frequency network share.

A single-frequency network (SFN) radiates the same signal from several transmitters on
the same frequency. The receiver therefore gets several copies, each delayed by the
propagation time of its path. The channel impulse response shows these copies as
individual peaks over delay: 1 µs corresponds to 300 m of path difference.

It is estimated from the phase reference symbol (PRS): received times conjugate known
gives the channel per subcarrier, and the inverse FFT of that gives the impulse
response. This costs one 2048-point FFT per frame.
"""

from __future__ import annotations

import functools

import numpy as np

from dab.constants import TransmissionMode
from dsp.timing import physical_carrier_indices

# Displayed delay range in samples, relative to the path the receiver synchronises to
# (0). The inverse FFT cyclically covers one useful symbol duration (2048 samples =
# 1000 µs); it is split so that pre-echoes up to 250 µs and echoes up to 750 µs are
# visible. The receiver locks to the earliest strong path, so later paths are the
# normal case.
PRE_ECHO_SAMPLES = 512

# Half width of the main path in seconds (±1.5 µs, ±450 m). A single path is not
# arbitrarily narrow in the impulse response: the 1.536 MHz bandwidth only resolves
# about 0.7 µs, and the Hann window widens the peak further. With this width less than
# 0.2 % of a single path's energy lies outside (simulated for arbitrary fractions of a
# sample). Paths closer together than this count as part of the main path.
MAIN_PATH_HALF_WIDTH_SECONDS = 1.5e-6


@functools.cache
def _fft_bins(num_carriers: int, fft_length: int) -> np.ndarray:
    """FFT bin per subcarrier in physical order. Cached: do not modify."""
    return physical_carrier_indices(num_carriers) % fft_length


@functools.cache
def _window(num_carriers: int) -> np.ndarray:
    """Hann window across the subcarriers. Cached: do not modify.

    Without a window every peak would have -13 dB side lobes that look like further
    paths, and 3-5 % of a single path's energy would lie outside its main lobe. With
    the Hann window the side lobes drop to -31 dB.
    """
    return np.hanning(num_carriers)


def impulse_response(
    channel_estimate: np.ndarray, mode: TransmissionMode, sample_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Channel impulse response from the per-subcarrier channel estimate.

    Args:
        channel_estimate: channel per subcarrier in physical order, as delivered by
            `dsp.ofdm.demodulate_frame_symbols`.

    Returns:
        (delay in seconds, linear power), ascending in delay. The path the receiver
        synchronised to sits at 0.
    """
    n = mode.fft_length
    grid = np.zeros(n, dtype=complex)
    grid[_fft_bins(mode.num_carriers, n)] = channel_estimate * _window(mode.num_carriers)
    power = np.abs(np.fft.ifft(grid)) ** 2
    delays = np.arange(-PRE_ECHO_SAMPLES, n - PRE_ECHO_SAMPLES)
    return delays / sample_rate, np.roll(power, PRE_ECHO_SAMPLES)


def sfn_shares(
    delays: np.ndarray, power: np.ndarray, mode: TransmissionMode, sample_rate: float
) -> tuple[float, float]:
    """Share of the received power that does not arrive over the strongest path.

    Returns:
        (share of all paths except the strongest, share outside the guard interval),
        both relative to the total signal power (0..1).

    Harmless are paths from 0 up to one guard interval behind the path the receiver
    synchronised to: their cyclic prefix still covers the start of the FFT window.
    Everything before and after mixes neighbouring symbols into the window. At both
    limits the resolution of the main path serves as tolerance. The second share
    includes the strongest path if it lies there itself: the receiver locks to the
    earliest path that is at least half as strong (`dsp.timing.find_integer_symbol_offset`),
    so the strongest may come later.

    The noise spreads evenly over all delays and is subtracted, otherwise it would
    count as SFN share. It is estimated from the median of the impulse response, which
    consists mostly of noise: for complex Gaussian noise the median of the power is
    ln(2) times its mean.
    """
    noise = float(np.median(power)) / np.log(2)
    signal = power - noise

    half_width = MAIN_PATH_HALF_WIDTH_SECONDS
    guard = mode.guard_length / sample_rate
    main = np.abs(delays - delays[np.argmax(power)]) <= half_width
    harmless = (delays >= -half_width) & (delays <= guard + half_width)

    # After subtracting the estimated noise individual sums can go slightly negative,
    # so the shares are clamped to 0..1.
    total = float(signal.sum())
    if total <= 0.0:
        return 0.0, 0.0
    others = float(signal[~main].sum()) / total
    outside = float(signal[~harmless].sum()) / total
    return float(np.clip(others, 0.0, 1.0)), float(np.clip(outside, 0.0, 1.0))
