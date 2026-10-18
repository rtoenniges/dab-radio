"""The packets of the SDRplay API, taken over without the GIL.

The API calls its stream callback from a thread of its own, about 1500 times a second at
2.048 MS/s (1344 samples each). A Python callback needs the GIL for every call,
and gets it at most once per switch interval while another thread computes in Python:
with the default 5 ms and two such threads, some 250 times a second, a tenth of what
the stream needs, and the API drops most of the samples. The radio's signal processing
is such a thread.

Here the callback is a C function compiled with numba (`cfunc`): no Python objects, no
GIL. It copies the packet with its sample number and reset flag into a ring of slots of
SLOT_SAMPLES samples; a packet longer than a slot is cut into several. The ring lives in
one NumPy buffer, which the callback reaches through the context pointer the API hands
back with every call. `drain()` passes the packets on in order, from a Python thread that
may wait for the GIL as long as the ring holds. Should it lag longer, the newest packets
are dropped and counted; the sample counter then jumps, and `sdr.rspduo_reader` fills
the gap with zeros.

The ring holds RING_SECONDS of signal in slots of one packet each (25 MB at 2.048 MS/s).

The newest packet always stays in the ring until the next one has arrived: the callback
writes samples, then metadata, then the count, and nothing orders these stores for a
reader on another core; a packet is only read once the callback that wrote it has
returned and the next one has run.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numba import carray, cfunc, njit, types

# Signal the ring holds, in seconds, at API_PACKET_SAMPLES per slot. The reading thread
# (`sdr.worker.RspDuoWorker`) visits it every 50 ms; this covers the GIL being held
# elsewhere for a long time, as the RTL-SDR's 15 USB transfers of 192 ms each do for
# pyrtlsdr's reading thread.
RING_SECONDS = 3.0

# Samples per packet of the SDRplay API at the DAB rate of 2.048 MS/s, on both tuners of
# the RSPduo; other rates give other sizes (1008 at 2 and 6 MS/s, 2016 at 10 MS/s). The
# ring is sized for it.
API_PACKET_SAMPLES = 1344

# Samples a slot holds: one packet. A longer packet is cut into several slots.
SLOT_SAMPLES = API_PACKET_SAMPLES

# The buffer: the state words, then the metadata of every slot, then the samples of
# every slot (I and Q interleaved), all int64-aligned.
_STATE_WORDS = 8
WRITTEN, READ, OVERFLOWS, SLOTS = 0, 1, 2, 3
_META_WORDS = 4  # per slot: samples, first sample number, reset flag, change flags
NUM_SAMPLES, FIRST_SAMPLE_NUM, RESET, CHANGED = 0, 1, 2, 3
# Bits of CHANGED: the API's grChanged, rfChanged and fsChanged of the packet.
GAIN_CHANGED, FREQUENCY_CHANGED, RATE_CHANGED = 1, 2, 4
_STATE_BYTES = 8 * _STATE_WORDS
_META_BYTES_PER_SLOT = 8 * _META_WORDS
_SAMPLE_BYTES_PER_SLOT = 2 * 2 * SLOT_SAMPLES

# The API's callback: short *xi, short *xq, sdrplay_api_StreamCbParamsT *params (five
# 32-bit words, firstSampleNum first), unsigned numSamples, unsigned reset, void *context.
_SIGNATURE = types.void(
    types.CPointer(types.int16),
    types.CPointer(types.int16),
    types.CPointer(types.uint32),
    types.uint32,
    types.uint32,
    types.voidptr,
)

# Takes the samples (int16, I and Q interleaved, a view into the ring valid during the
# call), the API's number of the first sample, its reset flag and its change flags
# (GAIN_CHANGED, FREQUENCY_CHANGED, RATE_CHANGED).
PacketSink = Callable[[np.ndarray, int, bool, int], None]


def ring_slots(sample_rate: float) -> int:
    """Slots for RING_SECONDS of signal at `sample_rate`, in packets of the API."""
    return max(2, math.ceil(RING_SECONDS * sample_rate / API_PACKET_SAMPLES))


@njit(nogil=True, cache=True, inline="always")
def _ring(context):  # pragma: no cover
    """The state words, the metadata and the samples of the ring at `context`, and its
    number of slots, which the state carries."""
    state = carray(context, (_STATE_WORDS,), np.int64)
    slots = state[SLOTS]
    meta_bytes = _META_BYTES_PER_SLOT * slots
    total = _STATE_BYTES + meta_bytes + _SAMPLE_BYTES_PER_SLOT * slots
    base = carray(context, (total,), np.int8)
    meta = base[_STATE_BYTES : _STATE_BYTES + meta_bytes].view(np.int64)
    samples = base[_STATE_BYTES + meta_bytes :].view(np.int16)
    return state, meta, samples, slots


@cfunc(_SIGNATURE, nogil=True, cache=True)
def _stream_callback(xi, xq, params, num_samples, reset, context):  # pragma: no cover
    state, meta, samples, slots = _ring(context)
    total = np.int64(num_samples)
    first = np.int64(params[0])
    done = 0
    # One slot per packet; a packet longer than a slot fills several, numbered on.
    while done < total:
        count = min(total - done, SLOT_SAMPLES)
        written = state[WRITTEN]
        if written - state[READ] >= slots:
            state[OVERFLOWS] += 1
        else:
            slot = written % slots
            offset = slot * 2 * SLOT_SAMPLES
            for k in range(count):
                samples[offset + 2 * k] = xi[done + k]
                samples[offset + 2 * k + 1] = xq[done + k]
            meta[slot * _META_WORDS + NUM_SAMPLES] = count
            meta[slot * _META_WORDS + FIRST_SAMPLE_NUM] = (first + done) & 0xFFFFFFFF
            meta[slot * _META_WORDS + RESET] = reset if done == 0 else 0
            changed = 0
            if done == 0:
                if params[1] != 0:
                    changed |= GAIN_CHANGED
                if params[2] != 0:
                    changed |= FREQUENCY_CHANGED
                if params[3] != 0:
                    changed |= RATE_CHANGED
            meta[slot * _META_WORDS + CHANGED] = changed
            state[WRITTEN] = written + 1
        done += count


class PacketRing:
    """The ring and the compiled callback that fills it.

    `slots` overrides the number of slots (`ring_slots` of the sample rate otherwise):
    for tests that want a small ring.
    """

    def __init__(self, sample_rate: float, slots: int | None = None) -> None:
        self._sample_rate = sample_rate
        self._slots = ring_slots(sample_rate) if slots is None else slots
        if self._slots < 2:
            raise ValueError("the ring needs at least two slots: the newest packet waits")
        meta_bytes = _META_BYTES_PER_SLOT * self._slots
        total = _STATE_BYTES + meta_bytes + _SAMPLE_BYTES_PER_SLOT * self._slots
        self._buffer = np.zeros(total, np.int8)
        self._state = self._buffer[:_STATE_BYTES].view(np.int64)
        self._state[SLOTS] = self._slots
        self._meta = (
            self._buffer[_STATE_BYTES : _STATE_BYTES + meta_bytes]
            .view(np.int64)
            .reshape(self._slots, _META_WORDS)
        )
        self._samples = (
            self._buffer[_STATE_BYTES + meta_bytes :]
            .view(np.int16)
            .reshape(self._slots, 2 * SLOT_SAMPLES)
        )

    @property
    def callback_address(self) -> int:
        """Address of the C callback, for `sdrplay_api_CallbackFnsT`."""
        return _stream_callback.address

    @property
    def context(self) -> int:
        """Address of the buffer, to be passed to `sdrplay_api_Init` as the context."""
        return self._buffer.ctypes.data

    @property
    def slots(self) -> int:
        return self._slots

    @property
    def seconds(self) -> float:
        """Signal the ring holds, in packets of the API."""
        return self._slots * API_PACKET_SAMPLES / self._sample_rate

    @property
    def overflows(self) -> int:
        """Packets dropped because the ring was full."""
        return int(self._state[OVERFLOWS])

    @property
    def written(self) -> int:
        """Packets written into the ring so far (dropped ones not counted). The n-th
        packet `drain()` hands over, counted from 0, is the n-th written."""
        return int(self._state[WRITTEN])

    @property
    def pending(self) -> int:
        """Packets in the ring, the newest (which waits for the next) included."""
        return int(self._state[WRITTEN] - self._state[READ])

    def drain(self, sink: PacketSink) -> int:
        """Hands every packet but the newest to `sink(samples, first_sample_num, reset,
        changed)`, in order; `samples` are views into the ring, valid during the call.
        Returns how many were handed over."""
        state = self._state
        read = int(state[READ])
        last = int(state[WRITTEN]) - 1
        for index in range(read, last):
            slot = index % self._slots
            count = int(self._meta[slot, NUM_SAMPLES])
            sink(
                self._samples[slot, : 2 * count],
                int(self._meta[slot, FIRST_SAMPLE_NUM]),
                bool(self._meta[slot, RESET]),
                int(self._meta[slot, CHANGED]),
            )
            state[READ] = index + 1
        return max(0, last - read)
