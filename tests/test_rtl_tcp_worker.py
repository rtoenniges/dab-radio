"""`RtlTcpWorker` with a scripted reader: the same course as `RspDuoWorker` (shared in
`_BlockReaderWorker`, tested in `test_rspduo_worker`), here what is its own."""

import numpy as np

from sdr.exceptions import RtlTcpError
from sdr.rtl_tcp import RtlTcpSettings, ServerInfo
from sdr.worker import RtlTcpWorker


class _Reader:
    """Hands out one block per entry of the script; an exception is raised instead."""

    def __init__(self, script, open_error=None):
        self.script = list(script)
        self.open_error = open_error
        self.closed = False

    def __call__(self, settings, center_freq, tag=0):
        self.settings = settings
        self.center_freq = center_freq
        self.tag = tag
        return self

    def open(self):
        if self.open_error is not None:
            raise self.open_error
        return ServerInfo(tuner_type=6, gain_count=29)

    def read_block(self, num_samples):
        if not self.script:
            return None
        entry = self.script.pop(0)
        if isinstance(entry, Exception):
            raise entry
        return self.tag, np.full(num_samples, entry, np.complex64)

    def retune(self, center_freq, tag):
        self.tag = tag

    def cancel(self):
        pass

    def close(self):
        self.closed = True


def _run(reader: _Reader) -> tuple[RtlTcpWorker, list]:
    worker = RtlTcpWorker(
        center_freq=178_352_000.0,
        settings=RtlTcpSettings("radio-pi", gain=29.7),
        read_size=16,
        reader_factory=reader,
    )
    events = []
    worker.connected.connect(lambda message: events.append(("connected", message)))
    worker.tuned.connect(lambda tuning: events.append(("tuned", tuning)))
    worker.samples_ready.connect(lambda samples: events.append(("samples", samples[0].real)))
    worker.error.connect(lambda message: events.append(("error", message)))
    worker.stopped.connect(lambda: events.append(("stopped",)))
    worker.run()
    return worker, events


def test_reception_names_server_and_tuner_and_announces_its_tuning():
    reader = _Reader([1, 2])

    worker, events = _run(reader)

    assert reader.settings == RtlTcpSettings("radio-pi", gain=29.7)
    assert reader.center_freq == 178_352_000.0
    assert events == [
        ("connected", "rtl_tcp radio-pi:1234 (R828D) ready @ 178.352 MHz, 2.048 MSps"),
        ("tuned", worker.tuning),
        ("samples", 1.0),
        ("samples", 2.0),
        ("stopped",),
    ]
    assert reader.closed


def test_a_server_out_of_reach_ends_reception_with_its_message():
    reader = _Reader([], open_error=RtlTcpError("radio-pi:1234 refuses the connection."))

    _worker, events = _run(reader)

    assert events == [("error", "radio-pi:1234 refuses the connection.")]


def test_a_lost_connection_ends_reception_with_its_message():
    reader = _Reader([1, RtlTcpError("The rtl_tcp server closed the connection.")])

    _worker, events = _run(reader)

    assert events[-2:] == [
        ("samples", 1.0),
        ("error", "The rtl_tcp server closed the connection."),
    ]
    assert reader.closed
