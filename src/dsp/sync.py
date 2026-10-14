"""Null symbol search: locating the start of a transmission frame."""

from __future__ import annotations

import numpy as np


def find_null_symbol(
    samples: np.ndarray, null_symbol_length: int, max_start: int | None = None
) -> int:
    """Finds the start index of the DAB null symbol by searching for the power minimum.

    The null symbol marks the start of a transmission frame with a brief transmission
    pause (markedly lower power than the OFDM symbols that follow). A sliding window
    looks for the position with the lowest mean power in the sample buffer.

    `max_start` restricts the search to the first `max_start` candidate start
    positions. This matters for a buffer spanning more than one frame (e.g. to
    guarantee enough samples for a complete MSC decode): without the limit a plain
    global minimum search would occasionally pick the LATER of the (then several) null
    symbols in the buffer, leaving too few samples for a complete frame after it.

    Raises:
        ValueError: if the buffer is shorter than a null symbol.
    """
    if len(samples) < null_symbol_length:
        raise ValueError("Sample buffer is shorter than a null symbol.")

    power = np.abs(samples) ** 2
    cumulative = np.cumsum(np.insert(power, 0, 0.0))
    window_sum = cumulative[null_symbol_length:] - cumulative[:-null_symbol_length]
    if max_start is not None:
        window_sum = window_sum[: max(max_start, 1)]
    return int(np.argmin(window_sum))
