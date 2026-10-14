"""Reed-Solomon decoding: DAB+ superframes (ETSI TS 102 563, clause 5.1) and the FEC of
packet mode subchannels (ETSI EN 300 401, clause 5.3.5)."""

from __future__ import annotations

import numba
import numpy as np

# ETSI TS 102 563, clause 5.1 (Reed-Solomon protection of DAB+ superframes): shortened
# RS(120,110) code over GF(256), derived from the full RS(255,245) code by padding with
# 135 virtual zero bytes at the front. The parameters (field polynomial, first root,
# root spacing, number of roots, padding) and the complete decoding algorithm
# (syndromes, Berlekamp-Massey, Chien search, Forney) are cross-checked byte for byte
# against the reference library libfec used by welle.io (Phil Karn, KA9Q;
# src/libs/fec/{init_rs.h,decode_rs.h} in AlbrechtL/welle.io), because a mistake in
# any of these constants could make the error correction silently insert wrong bytes
# or discard correct data.
SYMBOL_SIZE = 8
FIELD_GENERATOR_POLY = 0x11D  # x^8+x^4+x^3+x^2+1
NN = (1 << SYMBOL_SIZE) - 1  # 255: symbols in the full (unshortened) code
NROOTS = 10  # number of parity bytes (corrects up to 5 byte errors per codeword)
FCR = 0  # first root of the generator polynomial (index form)
PRIM = 1  # spacing between the roots (index form)
# IPRIM: PRIM-th root of 1 (index form), needed by libfec for the Chien search. Derived
# as in init_rs.h: smallest iprim>=1 with iprim%PRIM==0, then IPRIM=iprim/PRIM; with
# PRIM=1 that is immediately iprim=1, IPRIM=1.
IPRIM = 1
PAD = 135  # padding bytes: NN - PAD = 120 bytes actually transmitted
CODEWORD_BYTES = NN - PAD  # 120
DATA_BYTES = CODEWORD_BYTES - NROOTS  # 110

# ETSI EN 300 401, clause 5.3.5.1 (FEC for packet mode): shortened RS(204,188) code,
# t = 8, derived from RS(255,239) by 51 virtual zero bytes at the front. Field
# polynomial, first root and root spacing are the same as for DAB+ (g(x) =
# (x+λ^0)(x+λ^1)...(x+λ^15), λ = 0x02), only with 16 roots instead of 10.
PACKET_NROOTS = 16
PACKET_CODEWORD_BYTES = 204
PACKET_DATA_BYTES = PACKET_CODEWORD_BYTES - PACKET_NROOTS  # 188

_A0 = NN  # sentinel for "log(0)" (the Galois field element 0 has no logarithm)


@numba.njit(cache=True)
def _modnn(x: int) -> int:
    """Reduces `x` modulo NN=255 (ETSI/libfec bit trick: NN=2^8-1)."""
    while x >= NN:
        x -= NN
        x = (x >> SYMBOL_SIZE) + (x & NN)
    return x


def _build_gf_tables() -> tuple[np.ndarray, np.ndarray]:
    """Builds the log/antilog tables (ALPHA_TO/INDEX_OF) of the Galois field GF(256),
    exactly like libfec's `init_rs.h`."""
    alpha_to = np.zeros(NN + 1, dtype=np.int64)
    index_of = np.zeros(NN + 1, dtype=np.int64)
    index_of[0] = _A0
    alpha_to[_A0] = 0
    sr = 1
    for i in range(NN):
        index_of[sr] = i
        alpha_to[i] = sr
        sr <<= 1
        if sr & (1 << SYMBOL_SIZE):
            sr ^= FIELD_GENERATOR_POLY
        sr &= NN
    return alpha_to, index_of


_ALPHA_TO, _INDEX_OF = _build_gf_tables()


def decode_rs_codeword(codeword: np.ndarray) -> tuple[np.ndarray, int]:
    """Decodes/corrects a 120-byte RS(120,110) codeword (ETSI TS 102 563).

    Byte-for-byte port of libfec's generic Berlekamp-Massey decoder (`decode_rs.h`),
    specialised to the DAB+ parameters (see the module comment). `codeword[0]` is the
    first transmitted (most significant) byte, `codeword[110:120]` are the 10 parity
    bytes.

    Returns:
        (corrected, num_corrections): `corrected` is the 120-byte codeword, corrected
        where the errors were correctable; `num_corrections` is the number of corrected
        bytes (0 for an error-free codeword), or -1 if there are more errors than the
        code can correct (then `corrected` is `codeword` unchanged).
    """
    if len(codeword) != CODEWORD_BYTES:
        raise ValueError(f"An RS codeword must be {CODEWORD_BYTES} bytes long.")
    return _decode(codeword, NROOTS)


def decode_packet_rs_codeword(codeword: np.ndarray) -> tuple[np.ndarray, int]:
    """Decodes/corrects a 204-byte RS(204,188) codeword of the packet mode FEC (ETSI EN
    300 401, clause 5.3.5.1): one row of the FEC frame, 188 bytes of the application
    data table followed by the 16 bytes of the RS data table.

    Returns:
        Like `decode_rs_codeword`; up to 8 byte errors are correctable.
    """
    if len(codeword) != PACKET_CODEWORD_BYTES:
        raise ValueError(f"A packet mode RS codeword must be {PACKET_CODEWORD_BYTES} bytes long.")
    return _decode(codeword, PACKET_NROOTS)


def _decode(codeword: np.ndarray, nroots: int) -> tuple[np.ndarray, int]:
    # Contiguous copy: the callers pass a column or row of a matrix, i.e. possibly a
    # strided view.
    corrected = np.array(codeword, dtype=np.uint8)
    num_corrections = _decode_rs_in_place(corrected, nroots)
    if num_corrections == -1:
        return codeword, -1
    return corrected, num_corrections


@numba.njit(cache=True)
def _decode_rs_in_place(data: np.ndarray, nroots: int) -> int:
    """Core of the two decoders above, JIT-compiled with numba, for a shortened code
    with `nroots` parity bytes; the number of padding bytes follows from the length of
    `data`.

    As pure Python one superframe (12 codewords at 96 kbit/s) cost 3.7 ms without and
    14 ms with 3 errors per codeword, i.e. slower the worse the reception. Writes the
    corrections straight into `data` and returns their number; on -1 (uncorrectable)
    `data` stays unchanged, because nothing is written before that check.

    The lists and `np.roll` calls of the libfec template are fixed arrays and loops
    here so that nothing is allocated inside the loops.

    No `nogil`: a codeword takes only microseconds, and every release of the GIL would
    have to be fought for again afterwards; if the GUI happens to be computing, that
    can take up to the switch interval (5 ms).
    """
    pad = NN - len(data)
    # Form the syndromes: evaluate data(x) at the roots of g(x).
    syndromes = np.empty(nroots, dtype=np.int64)
    for i in range(nroots):
        syndromes[i] = data[0]
    for j in range(1, len(data)):
        for i in range(nroots):
            if syndromes[i] == 0:
                syndromes[i] = data[j]
            else:
                syndromes[i] = (
                    data[j] ^ _ALPHA_TO[_modnn(_INDEX_OF[syndromes[i]] + (FCR + i) * PRIM)]
                )

    syn_error = False
    syndromes_index = np.empty(nroots, dtype=np.int64)
    for i in range(nroots):
        syn_error = syn_error or syndromes[i] != 0
        syndromes_index[i] = _INDEX_OF[syndromes[i]]

    if not syn_error:
        return 0

    lambda_ = np.zeros(nroots + 1, dtype=np.int64)
    lambda_[0] = 1
    b = np.empty(nroots + 1, dtype=np.int64)
    for i in range(nroots + 1):
        b[i] = _INDEX_OF[lambda_[i]]
    t = np.empty(nroots + 1, dtype=np.int64)

    # Berlekamp-Massey: determines the error locator polynomial lambda(x).
    el = 0
    r = 0
    while r < nroots:
        r += 1
        discrepancy = 0
        for i in range(r):
            if lambda_[i] != 0 and syndromes_index[r - i - 1] != _A0:
                discrepancy ^= _ALPHA_TO[_modnn(_INDEX_OF[lambda_[i]] + syndromes_index[r - i - 1])]
        discrepancy = _INDEX_OF[discrepancy]

        if discrepancy == _A0:
            # b(x) = x * b(x), like `memmove(&b[1], b, nroots)` in libfec.
            for i in range(nroots, 0, -1):
                b[i] = b[i - 1]
            b[0] = _A0
        else:
            t[:] = lambda_
            for i in range(nroots):
                if b[i] != _A0:
                    t[i + 1] = lambda_[i + 1] ^ _ALPHA_TO[_modnn(discrepancy + b[i])]
            if 2 * el <= r - 1:
                el = r - el
                for i in range(nroots + 1):
                    if lambda_[i] == 0:
                        b[i] = _A0
                    else:
                        b[i] = _modnn(_INDEX_OF[lambda_[i]] - discrepancy + NN)
            else:
                for i in range(nroots, 0, -1):
                    b[i] = b[i - 1]
                b[0] = _A0
            lambda_, t = t, lambda_

    deg_lambda = 0
    lambda_index = np.empty(nroots + 1, dtype=np.int64)
    for i in range(nroots + 1):
        lambda_index[i] = _INDEX_OF[lambda_[i]]
        if lambda_index[i] != _A0:
            deg_lambda = i

    # Chien search: the roots of lambda(x) are the error locations. `k` must be used
    # with its value BEFORE the increment (the C for loop
    # `for(i=1,k=IPRIM-1; ...; i++, k=MODNN(k+IPRIM))` increments k only AFTER the
    # loop body); IPRIM is 1 as well for our PRIM=1. At most `deg_lambda` <= nroots
    # roots, since the search stops there.
    reg = lambda_index.copy()
    roots = np.empty(nroots, dtype=np.int64)
    locations = np.empty(nroots, dtype=np.int64)
    count = 0
    k = IPRIM - 1
    for i in range(1, NN + 1):
        q = 1
        for j in range(deg_lambda, 0, -1):
            if reg[j] != _A0:
                reg[j] = _modnn(reg[j] + j)
                q ^= _ALPHA_TO[reg[j]]
        if q == 0:
            roots[count] = i
            locations[count] = k
            count += 1
            if count == deg_lambda:
                break
        k = _modnn(k + IPRIM)

    if deg_lambda != count:
        return -1  # more errors than correctable

    # Error evaluator polynomial omega(x) = s(x)*lambda(x) mod x^nroots.
    deg_omega = deg_lambda - 1
    omega_index = np.full(deg_omega + 1, _A0, dtype=np.int64)
    for i in range(deg_omega + 1):
        acc = 0
        for j in range(i, -1, -1):
            if syndromes_index[i - j] != _A0 and lambda_index[j] != _A0:
                acc ^= _ALPHA_TO[_modnn(syndromes_index[i - j] + lambda_index[j])]
        omega_index[i] = _INDEX_OF[acc]

    # Forney algorithm: compute the error values from the error locations and apply them.
    for j in range(count - 1, -1, -1):
        root = roots[j]
        num1 = 0
        for i in range(deg_omega, -1, -1):
            if omega_index[i] != _A0:
                num1 ^= _ALPHA_TO[_modnn(omega_index[i] + i * root)]
        num2 = _ALPHA_TO[_modnn(root * (FCR - 1) + NN)]
        den = 0
        i = min(deg_lambda, nroots - 1) & ~1
        while i >= 0:
            if lambda_index[i + 1] != _A0:
                den ^= _ALPHA_TO[_modnn(lambda_index[i + 1] + i * root)]
            i -= 2

        loc = locations[j]
        if num1 != 0 and loc >= pad:
            data[loc - pad] ^= _ALPHA_TO[
                _modnn(_INDEX_OF[num1] + _INDEX_OF[num2] + NN - _INDEX_OF[den])
            ]

    return count
