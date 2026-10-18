"""The ring the API's compiled callback fills without the GIL: order, wrap-around,
overflow, packets longer than a slot and the packet that waits for the next."""

import ctypes
import threading

import numpy as np
import pytest

from sdr import sdrplay_api as sp
from sdr.packet_ring import (
    API_PACKET_SAMPLES,
    FREQUENCY_CHANGED,
    GAIN_CHANGED,
    RATE_CHANGED,
    RING_SECONDS,
    SLOT_SAMPLES,
    PacketRing,
)

FS = 2_000_000.0


class _Pusher:
    """Calls the ring's C callback as the API would."""

    def __init__(self, ring: PacketRing) -> None:
        self.ring = ring
        self.callback = sp.StreamCallback(ring.callback_address)

    def push(self, number: int, count: int = 1008, reset: int = 0, **changed: int) -> None:
        xi = (ctypes.c_short * count)(*((number + k) % 32768 for k in range(count)))
        xq = (ctypes.c_short * count)(*((-number - k) % 32768 for k in range(count)))
        params = sp.StreamCbParamsT(firstSampleNum=number * count, numSamples=count, **changed)
        self.callback(xi, xq, ctypes.pointer(params), count, reset, self.ring.context)


def _collect(ring: PacketRing) -> list:
    packets = []
    ring.drain(
        lambda iq, num, reset, _changed: packets.append(
            (iq[0::2].copy(), iq[1::2].copy(), num, reset)
        )
    )
    return packets


def test_packets_come_out_in_order_and_the_newest_waits_for_the_next():
    ring = PacketRing(FS)
    pusher = _Pusher(ring)
    for number in range(3):
        pusher.push(number, reset=int(number == 0))
    assert ring.pending == 3
    packets = _collect(ring)
    assert [num for _, _, num, _ in packets] == [0, 1008]
    assert [reset for _, _, _, reset in packets] == [True, False]
    i, q, _, _ = packets[1]
    np.testing.assert_array_equal(i, (1 + np.arange(1008)) % 32768)
    np.testing.assert_array_equal(q, (-1 - np.arange(1008)) % 32768)
    assert ring.pending == 1
    pusher.push(3, count=7)
    (packet,) = _collect(ring)
    assert packet[2] == 2 * 1008 and len(packet[0]) == 1008
    assert ring.overflows == 0


def test_the_ring_wraps_around():
    ring = PacketRing(FS, slots=64)
    pusher = _Pusher(ring)
    seen = []
    for number in range(3 * ring.slots + 5):
        pusher.push(number, count=16)
        if number % 7 == 0:
            seen += _collect(ring)
    seen += _collect(ring)
    assert [num for _, _, num, _ in seen] == [16 * n for n in range(3 * ring.slots + 4)]
    for i, _, num, _ in seen[-3:]:
        np.testing.assert_array_equal(i, (num // 16 + np.arange(16)) % 32768)
    assert ring.overflows == 0


def test_a_late_reader_loses_the_newest_packets_which_are_counted():
    ring = PacketRing(FS, slots=64)
    pusher = _Pusher(ring)
    for number in range(ring.slots + 5):
        pusher.push(number, count=16)
    assert ring.overflows == 5 and ring.pending == ring.slots
    assert ring.written == ring.slots  # the dropped ones are not counted
    packets = _collect(ring)
    assert [num for _, _, num, _ in packets] == [16 * n for n in range(ring.slots - 1)]
    pusher.push(ring.slots + 5, count=16)
    (packet,) = _collect(ring)
    assert packet[2] == 16 * (ring.slots - 1)  # the one that waited; 64 to 68 are gone


def test_a_packet_longer_than_a_slot_is_cut_into_several():
    ring = PacketRing(FS)
    pusher = _Pusher(ring)
    pusher.push(0, count=2 * SLOT_SAMPLES + 1, reset=1)  # as 2016 at 10 MS/s, and more
    pusher.push(1, count=1)
    packets = _collect(ring)
    assert [len(i) for i, _, _, _ in packets] == [SLOT_SAMPLES, SLOT_SAMPLES, 1]
    assert [num for _, _, num, _ in packets] == [0, SLOT_SAMPLES, 2 * SLOT_SAMPLES]
    assert [reset for _, _, _, reset in packets] == [True, False, False]
    i = np.concatenate([i for i, _, _, _ in packets])
    np.testing.assert_array_equal(i, np.arange(2 * SLOT_SAMPLES + 1) % 32768)
    assert ring.overflows == 0


def test_the_ring_holds_three_seconds_in_packets_of_one_slot_each():
    # At 2.048 MS/s the API sends 1344 samples per packet; with slots of 1024 every packet
    # would take two and the ring would hold 2 s.
    ring = PacketRing(2_048_000.0)
    assert SLOT_SAMPLES == API_PACKET_SAMPLES
    assert ring.seconds >= RING_SECONDS
    assert ring.slots * API_PACKET_SAMPLES < (RING_SECONDS + 0.001) * 2_048_000.0
    assert ring.slots == 4572
    with pytest.raises(ValueError):
        PacketRing(FS, slots=1)


def test_the_callback_runs_from_a_foreign_thread_while_python_drains():
    ring = PacketRing(FS)
    pusher = _Pusher(ring)
    total = 4000
    done = threading.Event()

    def produce():
        for number in range(total):
            pusher.push(number, count=64)
        done.set()

    threading.Thread(target=produce).start()
    seen = []
    while not done.is_set() or ring.pending > 1:
        seen += _collect(ring)
    assert [num for _, _, num, _ in seen] == [64 * n for n in range(total - 1)]
    assert ring.overflows == 0


def test_the_change_flags_of_the_api_are_kept_with_the_first_slot_of_a_packet():
    ring = PacketRing(FS)
    pusher = _Pusher(ring)
    pusher.push(0, count=16)
    pusher.push(1, count=16, rfChanged=1)
    pusher.push(2, count=SLOT_SAMPLES + 1, grChanged=1, fsChanged=1)
    pusher.push(3, count=16)
    flags = []
    ring.drain(lambda iq, num, reset, changed: flags.append(changed))
    assert flags == [0, FREQUENCY_CHANGED, GAIN_CHANGED | RATE_CHANGED, 0]
