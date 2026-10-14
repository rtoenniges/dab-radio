"""Querying the gains the connected RTL-SDR tuner supports."""

from __future__ import annotations

from rtlsdr import RtlSdr

# The gain steps of the common R820T/2 tuner found in most current RTL-SDR sticks
# (`r82xx_gains` in librtlsdr, there in tenths of a dB). Only a fallback for when no
# device is available for a real query at start-up.
FALLBACK_GAINS_DB = (
    0.0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7, 16.6, 19.7, 20.7, 22.9, 25.4,
    28.0, 29.7, 32.8, 33.8, 36.4, 37.2, 38.6, 40.2, 42.1, 43.4, 43.9, 44.5, 48.0, 49.6,
)  # fmt: skip
FALLBACK_GAIN_RANGE_DB = (FALLBACK_GAINS_DB[0], FALLBACK_GAINS_DB[-1])


def query_gains(device_index: int = 0) -> tuple[float, ...] | None:
    """Determines the gains the connected RTL-SDR stick supports, in ascending order.

    Briefly opens the device, reads the gain values the tuner supports and closes it
    again.

    Returns:
        The gains in dB, or None if no device is available or no gain values could be
        read.
    """
    try:
        sdr = RtlSdr(device_index=device_index)
    except OSError:
        return None

    try:
        gains = sdr.valid_gains_db
    except OSError:
        return None
    finally:
        sdr.close()

    if not gains:
        return None

    return tuple(sorted(gains))


def query_gain_range(device_index: int = 0) -> tuple[float, float] | None:
    """The gain range supported by the connected RTL-SDR stick as (min_db, max_db), None
    if no device is available or no gain values could be read (see `query_gains`)."""
    gains = query_gains(device_index)
    if gains is None:
        return None
    return gains[0], gains[-1]
