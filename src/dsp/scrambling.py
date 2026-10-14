"""Energy dispersal: the PRBS scrambling applied to every FIC and MSC logical frame."""

from __future__ import annotations

import functools

import numpy as np


@functools.cache
def generate_prbs(length: int) -> np.ndarray:
    """Generates the DAB energy dispersal PRBS (ETSI EN 300 401, clause 10.1).

    Feedback shift register with 9 stages and polynomial x^9+x^5+1, initialised to all
    ones; the output of each step is the feedback (XOR of stages 5 and 9) before it is
    shifted into the register.

    The sequence depends only on the length and restarts from the same initial value
    for every scrambled block, so it is constant for the whole run time and is cached.
    Without the cache this function was by far the most expensive item of the whole
    signal processing (20 % of the compute time according to profiling), because it
    ran a few thousand shift register steps in pure Python again for EVERY FIC and MSC
    logical frame. The returned array is therefore read-only and must not be modified
    by the caller.
    """
    register = np.ones(9, dtype=np.uint8)
    output = np.empty(length, dtype=np.uint8)
    for i in range(length):
        feedback = register[8] ^ register[4]
        output[i] = feedback
        register = np.concatenate(([feedback], register[:-1]))
    output.setflags(write=False)
    return output


def apply_energy_dispersal(bits: np.ndarray) -> np.ndarray:
    """Scrambles or descrambles `bits` with the PRBS (XOR is its own inverse).

    The PRBS restarts from its initial value for every scrambled block, e.g. every
    768-bit FIC logical frame (ETSI 10.2/10.3).
    """
    prbs = generate_prbs(len(bits))
    return (bits.astype(np.uint8) ^ prbs).astype(np.uint8)
