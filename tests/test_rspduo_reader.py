"""`RspDuoReader` on the packets of a fake SDRplay API: blocks after the discarded start,
gaps filled with zeros, retuning that drops exactly what came before, and the ways a
read ends."""

import ctypes
import logging
import threading
import time

import numpy as np
import pytest

from sdr import rspduo_reader as reader_module
from sdr import sdrplay_api as sp
from sdr.exceptions import SdrplayError
from sdr.rspduo import RspDuo, RspDuoSettings
from sdr.rspduo_reader import FULL_SCALE, RspDuoReader
from tests.fake_sdrplay import PACKET, FakeApi

START_DISCARD = 2 * PACKET
SETTLE = PACKET // 2


@pytest.fixture(autouse=True)
def _short_discards(monkeypatch):
    monkeypatch.setattr(reader_module, "START_DISCARD_SAMPLES", START_DISCARD)
    monkeypatch.setattr(reader_module, "RETUNE_SETTLE_SAMPLES", SETTLE)


def _values(first: int, count: int) -> np.ndarray:
    """I of the samples numbered `first` to `first + count`; Q is -I."""
    return ((np.arange(first, first + count) % 2000) - 1000).astype(np.int16)


def _expected(first: int, count: int) -> np.ndarray:
    values = _values(first, count).astype(np.float32) / FULL_SCALE
    return (values - 1j * values).astype(np.complex64)


class _Stream:
    """Pushes consecutively numbered packets, as the device delivers them."""

    def __init__(self, api: FakeApi, first: int = 0) -> None:
        self.api = api
        self.next = first

    def packets(self, count: int, skip: int = 0, rf_changed: bool = False) -> None:
        """`count` packets, after `skip` lost samples; `rf_changed` flags them all."""
        self.next += skip
        for _ in range(count):
            self.api.push(self.next, _values(self.next, PACKET), rf_changed=rf_changed)
            self.next += PACKET


def _reader(api: FakeApi, **kwargs) -> RspDuoReader:
    kwargs.setdefault("poll_seconds", 0.001)
    reader = RspDuoReader(
        RspDuoSettings(tuner=1),
        178_352_000.0,
        tag=7,
        device_factory=lambda: RspDuo(api_factory=lambda: api),
        **kwargs,
    )
    reader.open()
    return reader


def test_blocks_begin_after_the_discarded_start_at_full_scale_one():
    api = FakeApi()
    reader = _reader(api)
    _Stream(api).packets(6)  # the sixth waits for the next

    tag, samples = reader.read_block(3000)

    assert tag == 7
    assert samples.dtype == np.complex64
    np.testing.assert_allclose(samples, _expected(START_DISCARD, 3000))


def test_consecutive_blocks_continue_without_a_gap():
    api = FakeApi()
    reader = _reader(api)
    _Stream(api).packets(8)

    first = reader.read_samples(1000)
    second = reader.read_samples(2500)

    np.testing.assert_allclose(np.concatenate([first, second]), _expected(START_DISCARD, 3500))


def test_a_gap_in_the_numbers_is_filled_with_zeros():
    api = FakeApi()
    reader = _reader(api)
    stream = _Stream(api)
    stream.packets(4)
    stream.packets(4, skip=500)

    samples = reader.read_samples(5 * PACKET)

    kept = 2 * PACKET  # the two packets after the discarded start
    np.testing.assert_allclose(samples[:kept], _expected(START_DISCARD, kept))
    np.testing.assert_array_equal(samples[kept : kept + 500], 0)
    np.testing.assert_allclose(samples[kept + 500 :], _expected(4 * PACKET + 500, 3 * PACKET - 500))
    assert reader.lost_samples == 500


def test_a_jump_longer_than_a_second_is_not_filled():
    api = FakeApi()
    reader = _reader(api)
    stream = _Stream(api)
    stream.packets(4)
    stream.packets(4, skip=3_000_000)

    samples = reader.read_samples(4 * PACKET)

    np.testing.assert_allclose(samples[2 * PACKET :], _expected(4 * PACKET + 3_000_000, 2 * PACKET))
    assert reader.lost_samples == 0


def test_the_sample_counter_wraps_around_without_a_gap():
    api = FakeApi()
    reader = _reader(api)
    first = 2**32 - 5 * PACKET + 17
    stream = _Stream(api, first=first)
    stream.packets(9)

    reader.read_samples(6 * PACKET)

    assert reader.lost_samples == 0


def test_retuning_drops_the_old_frequency_up_to_the_packet_the_api_flags():
    api = FakeApi()
    reader = _reader(api)
    stream = _Stream(api)
    stream.packets(6)
    # Leaves 3 * PACKET - 1000 samples of the old frequency waiting in the reader.
    assert reader.read_block(1000)[0] == 7
    # The device keeps delivering while the API applies the frequency.
    api.on_update = lambda: stream.packets(2)

    reader.retune(227_360_000.0, tag=8)
    stream.packets(5)  # still the old frequency: the change shows later in the stream
    switch = stream.next
    stream.packets(1, rf_changed=True)
    stream.packets(6)
    tag, samples = reader.read_block(3000)

    assert api.updates == [(sp.TUNER_A, sp.UPDATE_TUNER_FRF)]
    assert tag == 8
    # The flagged packet holds the switch; the settling follows it.
    np.testing.assert_allclose(samples, _expected(switch + PACKET + SETTLE, 3000))
    assert reader.center_freq == 227_360_000.0


def test_retunes_in_quick_succession_wait_for_the_flag_of_the_latest():
    api = FakeApi()
    reader = _reader(api)
    stream = _Stream(api)
    stream.packets(6)
    reader.read_block(1000)

    reader.retune(180_064_000.0, tag=8)
    reader.retune(227_360_000.0, tag=9)
    stream.packets(3)
    stream.packets(1, rf_changed=True)  # the first frequency
    stream.packets(3)
    switch = stream.next
    stream.packets(1, rf_changed=True)  # the second
    stream.packets(6)
    tag, samples = reader.read_block(3000)

    assert tag == 9
    np.testing.assert_allclose(samples, _expected(switch + PACKET + SETTLE, 3000))


def test_without_a_flag_the_new_frequency_counts_as_applied_after_the_timeout(monkeypatch, caplog):
    monkeypatch.setattr(reader_module, "FLAG_TIMEOUT_PACKETS", 4)
    api = FakeApi()
    reader = _reader(api)
    stream = _Stream(api)
    stream.packets(6)
    reader.read_block(1000)

    reader.retune(227_360_000.0, tag=8)  # 6 packets written so far
    stream.packets(12)
    with caplog.at_level(logging.WARNING):
        tag, samples = reader.read_block(3000)

    assert tag == 8
    np.testing.assert_allclose(samples, _expected(10 * PACKET + SETTLE, 3000))
    assert "did not flag the new frequency" in caplog.text


def test_a_flag_without_a_retune_is_ignored():
    api = FakeApi()
    reader = _reader(api)
    stream = _Stream(api)
    stream.packets(3)
    stream.packets(1, rf_changed=True)
    stream.packets(4)

    samples = reader.read_samples(5 * PACKET)

    np.testing.assert_allclose(samples, _expected(START_DISCARD, 5 * PACKET))


def test_retuning_to_the_same_frequency_only_starts_a_new_tag():
    api = FakeApi()
    reader = _reader(api)
    stream = _Stream(api)
    stream.packets(6)
    reader.read_block(1000)

    reader.retune(178_352_000.0, tag=8)
    stream.packets(3)
    tag, samples = reader.read_block(1000)

    assert api.updates == []
    assert tag == 8
    # The samples waiting from before belong to the old tag; the sixth packet, which
    # waited in the ring, is the first of the new one.
    np.testing.assert_allclose(samples, _expected(5 * PACKET, 1000))


def test_the_center_freq_setter_retunes_with_the_next_tag():
    api = FakeApi()
    reader = _reader(api)
    stream = _Stream(api)
    reader.center_freq = 180_064_000.0
    stream.packets(1, rf_changed=True)
    stream.packets(5)

    tag, _ = reader.read_block(1000)

    assert tag == 8
    assert reader.center_freq == 180_064_000.0


def test_after_a_failed_retune_nothing_more_is_handed_out():
    api = FakeApi(update_err=sp.FAIL)
    reader = _reader(api, no_data_timeout_seconds=0.05)
    stream = _Stream(api)

    with pytest.raises(SdrplayError, match="not applied"):
        reader.retune(227_360_000.0, tag=8)
    stream.packets(6)

    with pytest.raises(SdrplayError, match="no longer delivers samples"):
        reader.read_block(1000)


def test_blocks_never_mix_two_frequencies_while_another_thread_retunes():
    """Packets keep coming from a thread of their own, as from the API, each one with the
    frequency it was received on. As on the RSPduo, a new frequency shows some packets
    after `Update` has returned, in a packet the API flags; the reading thread retunes
    between its reads."""
    api = FakeApi()
    reader = _reader(api)
    requested = {"mhz": 7}  # the tags are the frequencies in MHz
    api.on_update = lambda: requested.update(mhz=round(api.channel.tunerParams.rfFreq.rfHz / 1e6))
    stop = threading.Event()

    def deliver():
        number = 0
        applied = 7
        delay = None
        while not stop.is_set():
            switch = False
            if requested["mhz"] != applied:
                delay = 5 if delay is None else delay - 1
                if delay == 0:
                    applied = requested["mhz"]
                    delay = None
                    switch = True
            api.push(number, np.full(PACKET, applied, np.int16), rf_changed=switch)
            number += PACKET
            time.sleep(0.0001)

    thread = threading.Thread(target=deliver)
    thread.start()
    try:
        for tag in range(8, 14):
            block_tag, samples = reader.read_block(3000)
            assert np.all(samples.real * FULL_SCALE == block_tag)
            reader.retune(tag * 1e6, tag)
    finally:
        stop.set()
        thread.join()


def test_cancel_ends_a_waiting_read():
    api = FakeApi()
    reader = _reader(api)
    threading.Timer(0.05, reader.cancel).start()

    assert reader.read_block(1000) is None
    with pytest.raises(SdrplayError, match="cancelled"):
        reader.read_samples(1000)


def test_a_removed_device_ends_the_read_with_its_message():
    api = FakeApi()
    reader = _reader(api)
    params = sp.EventParamsT()
    api.callbacks.EventCbFn(sp.EVENT_DEVICE_REMOVED, sp.TUNER_A, ctypes.pointer(params), None)

    with pytest.raises(SdrplayError, match="disconnected from USB"):
        reader.read_block(1000)


def test_a_stream_without_packets_counts_as_broken():
    api = FakeApi()
    reader = _reader(api, no_data_timeout_seconds=0.05)

    with pytest.raises(SdrplayError, match="no longer delivers samples"):
        reader.read_block(1000)


def test_overload_reports_are_passed_on():
    api = FakeApi()
    overloads = []
    reader = _reader(api, on_overload=overloads.append)
    params = sp.EventParamsT()
    params.powerOverloadParams.powerOverloadChangeType = sp.OVERLOAD_DETECTED
    api.callbacks.EventCbFn(
        sp.EVENT_POWER_OVERLOAD_CHANGE, sp.TUNER_A, ctypes.pointer(params), None
    )
    params.powerOverloadParams.powerOverloadChangeType = sp.OVERLOAD_CORRECTED
    api.callbacks.EventCbFn(
        sp.EVENT_POWER_OVERLOAD_CHANGE, sp.TUNER_A, ctypes.pointer(params), None
    )

    assert overloads == [True, False]
    reader.close()


def test_a_failed_open_releases_the_device():
    api = FakeApi(select_err=sp.FAIL)

    with pytest.raises(SdrplayError, match="is busy"):
        _reader(api)

    assert api.calls[-1] == "close"


def test_close_releases_the_device_once():
    api = FakeApi()
    reader = _reader(api)

    reader.close()
    reader.close()

    assert api.calls == ["open", "select", "init", "uninit", "release", "close"]
    assert reader.read_block(1000) is None
