"""Helpers for the Reed-Solomon tests (not a test module itself, hence no `test_`
prefix; shared by test_reed_solomon.py, test_superframe.py and test_packet.py).
"""

import numpy as np

from dsp.reed_solomon import _ALPHA_TO, _INDEX_OF, DATA_BYTES, FCR, NROOTS, PRIM, _modnn


def gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return int(_ALPHA_TO[_modnn(int(_INDEX_OF[a]) + int(_INDEX_OF[b]))])


def build_generator_poly(nroots: int = NROOTS) -> list[int]:
    """Builds the RS generator polynomial g(x) = prod_(k=0..nroots-1) (x - alpha^(FCR+k*PRIM))
    from its roots (the standard construction, independent of libfec's LFSR encoder
    code). It only serves as an independent test encoder against which
    `decode_rs_codeword` can be checked with real, valid codewords. Coefficients in
    descending degree, g[0] is always 1 (monic).
    """
    g = [1]
    for k in range(nroots):
        root = int(_ALPHA_TO[_modnn(FCR + k * PRIM)])
        new_g = [0] * (len(g) + 1)
        for i, c in enumerate(g):
            new_g[i] ^= c
            new_g[i + 1] ^= gf_mul(c, root)
        g = new_g
    return g


_GENERATOR_POLYS: dict[int, list[int]] = {}


def encode_rs_codeword(data_bytes: np.ndarray, nroots: int = NROOTS) -> np.ndarray:
    """Forms a valid RS codeword from the data bytes by explicit polynomial division by
    the generator polynomial (the standard construction for systematic RS codes:
    codeword = data + remainder(data * x^nroots / g(x))). Deliberately written as a
    plain, directly traceable long division instead of a more error-prone LFSR
    register trick, since this encoder only serves the tests.

    With the default `nroots` it builds RS(120,110) codewords from 110 data bytes (DAB+);
    with `nroots=16` and 188 data bytes the RS(204,188) rows of the packet mode FEC.
    Leading zero bytes of the shortened code do not change the remainder, so the
    shortening needs no special treatment.
    """
    if nroots == NROOTS and len(data_bytes) != DATA_BYTES:
        raise ValueError(f"Expected {DATA_BYTES} data bytes.")
    generator = _GENERATOR_POLYS.setdefault(nroots, build_generator_poly(nroots))
    data_length = len(data_bytes)

    remainder = [int(b) for b in data_bytes] + [0] * nroots
    for i in range(data_length):
        coef = remainder[i]
        if coef != 0:
            for j in range(nroots + 1):
                remainder[i + j] ^= gf_mul(coef, generator[j])

    parity = remainder[data_length:]
    return np.concatenate(
        [np.asarray(data_bytes, dtype=np.uint8), np.array(parity, dtype=np.uint8)]
    )
