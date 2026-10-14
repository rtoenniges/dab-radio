"""The DAB mother convolutional code: encoder, puncturing and the Viterbi decoder."""

from __future__ import annotations

import numba
import numpy as np

# DAB mother convolutional code, rate 1/4, constraint length 7 (ETSI EN 300 401, clause
# 11.1.1). Generator polynomials (octal): 133, 171, 145, 133.
CONSTRAINT_LENGTH = 7
MEMORY_BITS = CONSTRAINT_LENGTH - 1  # 6
NUM_STATES = 1 << MEMORY_BITS  # 64
NUM_OUTPUTS = 4
TAIL_BITS = MEMORY_BITS  # 6 zero bits that flush the shift register

# ETSI EN 300 401, table 13, puncturing pattern "PI_X": puncturing of the 24-bit tail
# block (the 6 zero bits of the shift register, rate-1/4 coded). Identical for the FIC
# (dsp.fic) and the MSC error protection (dsp.eep), cross-checked against welle.io
# (fic-handler.cpp, `PI_X`).
TAIL_PUNCTURE_PATTERN = np.array([int(b) for b in "1100" * 6], dtype=np.uint8)


def _output_bits(state: int, input_bit: int) -> tuple[int, int, int, int]:
    """Computes (x0,x1,x2,x3) for a given state and input bit.

    `state` bit j (j=0..5) corresponds to a[i-(j+1)] (a[i-1]..a[i-6]).
    """
    b0 = state & 1
    b1 = (state >> 1) & 1
    b2 = (state >> 2) & 1
    b3 = (state >> 3) & 1
    b4 = (state >> 4) & 1
    b5 = (state >> 5) & 1
    x0 = input_bit ^ b1 ^ b2 ^ b4 ^ b5
    x1 = input_bit ^ b0 ^ b1 ^ b2 ^ b5
    x2 = input_bit ^ b0 ^ b3 ^ b5
    x3 = x0
    return x0, x1, x2, x3


def _next_state(state: int, input_bit: int) -> int:
    return ((state << 1) | input_bit) & (NUM_STATES - 1)


def _build_trellis() -> tuple[np.ndarray, np.ndarray]:
    """Builds the complete trellis transition table.

    Returns:
        next_state[state, bit] -> successor state
        outputs[state, bit] -> (4,) coded output bits
    """
    next_state = np.zeros((NUM_STATES, 2), dtype=np.int64)
    outputs = np.zeros((NUM_STATES, 2, NUM_OUTPUTS), dtype=np.uint8)
    for state in range(NUM_STATES):
        for bit in (0, 1):
            outputs[state, bit] = _output_bits(state, bit)
            next_state[state, bit] = _next_state(state, bit)
    return next_state, outputs


_NEXT_STATE, _OUTPUTS = _build_trellis()


def encode_mother_code(bits: np.ndarray) -> np.ndarray:
    """Convolutionally encodes `bits` (a_i) with the DAB mother code (rate 1/4).

    Appends `TAIL_BITS` zero bits to flush the shift register (ETSI 11.1.1: "all-zero
    final state") and returns the serialised mother codeword sequence U of length
    `4 * (len(bits) + TAIL_BITS)`: the four partial streams x0..x3 are interleaved per
    time step.
    """
    state = 0
    output = np.empty((len(bits) + TAIL_BITS) * NUM_OUTPUTS, dtype=np.uint8)
    extended = np.concatenate([bits, np.zeros(TAIL_BITS, dtype=bits.dtype)])
    for i, bit in enumerate(extended):
        bit = int(bit)
        output[i * NUM_OUTPUTS : (i + 1) * NUM_OUTPUTS] = _OUTPUTS[state, bit]
        state = _NEXT_STATE[state, bit]
    return output


def soft_bits(bits: np.ndarray) -> np.ndarray:
    """Hard bits as soft bits of full confidence: +1 for a 0, -1 for a 1.

    The receiver chain carries soft bits from the QPSK demapper to the Viterbi decoder
    (see `viterbi_decode`); this is the transmit-side counterpart for the synthetic
    signals of the tests.
    """
    return (1.0 - 2.0 * np.asarray(bits, dtype=np.float32)).astype(np.float32)


def puncture(mother_bits: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Punctures `mother_bits` (length a multiple of `len(pattern)`) with `pattern`.

    `pattern` is a 0/1 array; positions with 0 are removed.
    """
    tiled_pattern = np.tile(pattern.astype(bool), len(mother_bits) // len(pattern))
    return mother_bits[tiled_pattern]


def depuncture(punctured_bits: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Re-inserts the punctured positions as soft bits of value 0.

    Returns an array of length `len(punctured_bits) / weight(pattern) * len(pattern)`.
    A soft bit of 0 says nothing about the bit, so the Viterbi decoder weighs both
    hypotheses equally there (see `viterbi_decode`); no separate mask is needed.
    """
    pattern_bool = pattern.astype(bool)
    weight = int(pattern_bool.sum())
    num_repeats = len(punctured_bits) // weight
    tiled_mask = np.tile(pattern_bool, num_repeats)

    result = np.zeros(len(tiled_mask), dtype=np.float32)
    result[tiled_mask] = punctured_bits
    return result


_TRELLIS_INF = 1e30  # far above any reachable path metric

# The 4 output bits of a trellis transition packed into one number 0..15 (x0 most
# significant), so that the Viterbi decoder can look up its branch metric.
_OUTPUT_CODES = (_OUTPUTS.astype(np.int64) << np.arange(NUM_OUTPUTS - 1, -1, -1)).sum(axis=2)

# A state has exactly two predecessors: the shift register step
# `((state << 1) | bit) & 63` discards the top bit of the old state, and the two
# predecessors differ only in that bit. `_OLDEST_BIT` is that bit.
_OLDEST_BIT = NUM_STATES >> 1


@numba.njit(cache=True)
def _viterbi(received: np.ndarray, output_codes: np.ndarray) -> np.ndarray:
    """Trellis pass and traceback (the hot path of `viterbi_decode`), JIT-compiled with
    numba.

    Plain scalar loops instead of NumPy vector operations: with only 4 bits per step,
    the Python/NumPy call overhead per trellis transition exceeds the actual arithmetic
    by orders of magnitude (profiling showed 2.36 million `np.count_nonzero` calls for
    a single FIC logical frame, ~1 s instead of ~10 ms).

    The branch metric of a transition sums the soft bits whose sign contradicts its
    output bits and subtracts those that agree (-r for an output bit 0, +r for a 1).
    Minimising it maximises the correlation between the received soft bits and the
    codeword. Per step it depends only on the 4 output bits, so it is computed once
    for all 16 combinations.

    Add-compare-select per target state: of its two predecessors the one with the
    smaller path metric wins, on a tie the one with the smaller index. Only that
    decision (the top bit of the predecessor) is stored; the transmitted bit is the
    bottom bit of the target state anyway.

    Deliberately keeps the GIL (no `nogil=True`): the DSP thread would have to win it
    back afterwards, and if the GUI happens to be running Python it only gets it after
    the 5 ms switch interval. Measured next to a second thread computing continuously:
    6.2 instead of 0.7 ms per logical frame. A call only takes about 0.35 ms, and the
    GUI can happily wait that long.

    Returns:
        All decoded bits including the tail bits.
    """
    num_steps = received.shape[0]

    path_metric = np.full(NUM_STATES, _TRELLIS_INF)
    path_metric[0] = 0.0  # guaranteed to start in the zero state (register reset per logical frame)
    new_metric = np.empty(NUM_STATES)
    decisions = np.empty((num_steps, NUM_STATES), dtype=np.uint8)
    branch_metric = np.empty(1 << NUM_OUTPUTS)

    for t in range(num_steps):
        for code in range(1 << NUM_OUTPUTS):
            metric = 0.0
            for k in range(NUM_OUTPUTS):
                if (code >> (NUM_OUTPUTS - 1 - k)) & 1:
                    metric += received[t, k]
                else:
                    metric -= received[t, k]
            branch_metric[code] = metric

        for state in range(NUM_STATES):
            bit = state & 1
            newer = state >> 1
            older = newer | _OLDEST_BIT
            metric_newer = path_metric[newer] + branch_metric[output_codes[newer, bit]]
            metric_older = path_metric[older] + branch_metric[output_codes[older, bit]]
            if metric_older < metric_newer:
                new_metric[state] = metric_older
                decisions[t, state] = 1
            else:
                new_metric[state] = metric_newer
                decisions[t, state] = 0
        path_metric, new_metric = new_metric, path_metric

    # The final state is guaranteed to be zero as well (tail flush).
    decoded = np.empty(num_steps, dtype=np.uint8)
    state = 0
    for t in range(num_steps - 1, -1, -1):
        decoded[t] = state & 1
        state = (state >> 1) | (_OLDEST_BIT if decisions[t, state] else 0)
    return decoded


def viterbi_decode(received_soft_bits: np.ndarray, num_info_bits: int) -> np.ndarray:
    """Soft-decision Viterbi decoding of the DAB mother code.

    `received_soft_bits` has length `4 * (num_info_bits + TAIL_BITS)`. A soft bit's
    sign carries the bit (positive = 0, negative = 1, as the QPSK mapper sends it), its
    magnitude how reliable it is; 0 carries no information (punctured positions, see
    `depuncture`). Only the ratios between the soft bits of one call matter, not their
    scale.

    Unlike a hard decision, a weak carrier (in a notch of the channel, in noise)
    contributes little to the metric instead of counting as much as a strong one.
    Measured on channel 9D at about 9 dB SNR: the Reed-Solomon decoder behind it had
    to correct 1-2 instead of ~300 bytes in 8 s, and decoding kept working with added
    noise where hard decisions already lost every superframe.

    Returns:
        The decoded information bits (length `num_info_bits`, without the tail bits).
    """
    num_steps = num_info_bits + TAIL_BITS
    received = np.asarray(received_soft_bits, dtype=np.float32).reshape(num_steps, NUM_OUTPUTS)
    return _viterbi(received, _OUTPUT_CODES)[:num_info_bits]
