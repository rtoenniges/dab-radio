"""The samples of an RTL2832U: unsigned 8-bit I/Q bytes as complex64, from the stick
(`sdr.device`) as from an rtl_tcp server (`sdr.rtl_tcp`)."""

from __future__ import annotations

from functools import cache

import numpy as np


@cache
def _sample_table() -> np.ndarray:
    """complex64 for each pair of bytes read as a little-endian uint16: I in the low
    byte, Q in the high one, scaled to full scale 1.0 as pyrtlsdr does."""
    levels = (np.arange(256, dtype=np.float32) - np.float32(127.5)) / np.float32(127.5)
    table = np.empty(65536, np.complex64)
    table.real = np.tile(levels, 256)
    table.imag = np.repeat(levels, 256)
    return table


def to_samples(data: bytes | bytearray | memoryview) -> np.ndarray:
    """The samples in `data` (I and Q interleaved, unsigned 8 bit; an even length), as
    complex64: one look-up per sample. pyrtlsdr's own conversion goes through float64 and
    delivers complex128, twice the memory for every stage of the signal processing to
    go through, for 8 bits of resolution."""
    return _sample_table()[np.frombuffer(data, dtype="<u2")]
