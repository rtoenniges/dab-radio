"""`RspDuoWorker` with a scripted reader: the same signals as `RtlSdrWorker`, `tuned`
before the first block of every tuning, and how reception ends."""

import numpy as np

from sdr.exceptions import SdrplayError
from sdr.rspduo import DeviceInfo, RspDuoSettings
from sdr.worker import RspDuoWorker


class _Reader:
    """Hands out the blocks of a script; an entry may also be a callable, run between two
    reads (a call from the GUI thread while the worker reads)."""

    def __init__(self, script, open_error=None, on_open=None):
        self.script = list(script)
        self.open_error = open_error
        self.on_open = on_open
        self.tag = None
        self.retuned = []
        self.closed = False
        self.cancelled = False

    def __call__(self, settings, center_freq, tag=0, on_overload=None):
        self.settings = settings
        self.center_freq = center_freq
        self.tag = tag
        self.on_overload = on_overload
        return self

    def open(self):
        if self.open_error is not None:
            raise self.open_error
        if self.on_open is not None:
            self.on_open()
        return DeviceInfo(serial="2301007634", api_version=3.15)

    def read_block(self, num_samples):
        while self.script:
            entry = self.script.pop(0)
            if isinstance(entry, Exception):
                raise entry
            if callable(entry):
                entry()
                continue
            return self.tag, np.full(num_samples, entry, np.complex64)
        return None

    def retune(self, center_freq, tag):
        self.retuned.append(center_freq)
        self.tag = tag

    def cancel(self):
        self.cancelled = True

    def close(self):
        self.closed = True


def _worker(reader: _Reader) -> RspDuoWorker:
    return RspDuoWorker(
        center_freq=178_352_000.0,
        settings=RspDuoSettings(tuner=2),
        read_size=16,
        reader_factory=reader,
    )


def _events(worker: RspDuoWorker) -> list:
    events = []
    worker.connected.connect(lambda message: events.append(("connected", message)))
    worker.tuned.connect(lambda tuning: events.append(("tuned", tuning)))
    worker.samples_ready.connect(lambda samples: events.append(("samples", samples[0].real)))
    worker.error.connect(lambda message: events.append(("error", message)))
    worker.stopped.connect(lambda: events.append(("stopped",)))
    return events


def test_the_first_block_is_preceded_by_its_tuning():
    reader = _Reader([1, 2])
    worker = _worker(reader)
    events = _events(worker)

    worker.run()

    assert events[0][0] == "connected"
    assert "RSPduo 2301007634, tuner 2" in events[0][1]
    assert events[1:] == [
        ("tuned", worker.tuning),
        ("samples", 1.0),
        ("samples", 2.0),
        ("stopped",),
    ]
    assert reader.tag == worker.tuning
    assert reader.settings == RspDuoSettings(tuner=2)
    assert reader.closed


def test_a_retune_announces_the_new_tuning_before_its_first_block():
    tunings = []
    reader = _Reader([1])
    worker = _worker(reader)
    reader.script += [lambda: tunings.append(worker.retune(227_360_000.0)), 2, 3]
    events = _events(worker)
    first = worker.tuning

    worker.run()

    (second,) = tunings
    assert second != first and worker.tuning == second
    assert reader.retuned == [227_360_000.0]
    assert [event for event in events if event[0] != "connected"] == [
        ("tuned", first),
        ("samples", 1.0),
        ("tuned", second),
        ("samples", 2.0),
        ("samples", 3.0),
        ("stopped",),
    ]


def test_retune_before_the_connection_does_nothing():
    reader = _Reader([])
    worker = _worker(reader)

    assert worker.retune(227_360_000.0) is None
    assert reader.retuned == []


def test_an_open_failure_is_reported_without_stopped():
    reader = _Reader([], open_error=SdrplayError("The RSPduo is busy."))
    worker = _worker(reader)
    events = _events(worker)

    worker.run()

    assert events == [("error", "The RSPduo is busy.")]


def test_a_read_failure_ends_reception_with_its_message():
    reader = _Reader([1, SdrplayError("The RSPduo was disconnected from USB.")])
    worker = _worker(reader)
    events = _events(worker)

    worker.run()

    assert events[-1] == ("error", "The RSPduo was disconnected from USB.")
    assert ("stopped",) not in events
    assert reader.closed


def test_stopping_while_opening_releases_the_device_unread():
    reader = _Reader([1, 2])
    worker = _worker(reader)
    reader.on_open = worker.stop  # e.g. the window is closed while connecting
    events = _events(worker)

    worker.run()

    assert events == [("stopped",)]
    assert reader.closed
    assert reader.script == [1, 2]


def test_stop_cancels_the_waiting_read():
    reader = _Reader([1])
    worker = _worker(reader)
    reader.script.append(worker.stop)
    events = _events(worker)

    worker.run()

    assert reader.cancelled
    assert events[-1] == ("stopped",)
