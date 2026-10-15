"""`RtlTcpReader` against an rtl_tcp server on localhost that the test plays: the header,
the commands on connecting and retuning, the samples, what is dropped, and what goes
wrong. `StreamLag` on its own."""

import socket
import struct
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from dab.constants import SAMPLE_RATE
from sdr import rtl_tcp
from sdr.exceptions import RtlTcpError
from sdr.rtl_tcp import (
    CMD_SET_AGC_MODE,
    CMD_SET_FREQ_CORRECTION,
    CMD_SET_FREQUENCY,
    CMD_SET_GAIN,
    CMD_SET_GAIN_MODE,
    CMD_SET_SAMPLE_RATE,
    RtlTcpReader,
    RtlTcpSettings,
    ServerInfo,
    StreamLag,
    open_connection,
    receive_buffer_bytes,
    to_samples,
)

HEADER_R820T = struct.pack(">4sII", b"RTL0", 5, 29)


class _Server:
    """Listens on localhost and takes the reader's connection in a thread (`open` waits
    for the header meanwhile), sending `header` right away; None sends nothing, b"" closes
    at once."""

    def __init__(self, header: bytes | None = HEADER_R820T) -> None:
        self._listener = socket.create_server(("127.0.0.1", 0))
        self.port = self._listener.getsockname()[1]
        self._header = header
        self._connection: socket.socket | None = None
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def _accept(self) -> None:
        self._connection, _ = self._listener.accept()
        if self._header == b"":
            self._connection.close()
        elif self._header is not None:
            self._connection.sendall(self._header)

    @property
    def connection(self) -> socket.socket:
        self._thread.join(5)
        assert self._connection is not None
        return self._connection

    def send(self, data: bytes) -> None:
        self.connection.sendall(data)

    def commands(self, count: int) -> list[tuple[int, int]]:
        """The next `count` commands from the reader."""
        connection = self.connection
        connection.settimeout(5)
        data = b""
        while len(data) < 5 * count:
            data += connection.recv(5 * count - len(data))
        return [struct.unpack(">BI", data[i : i + 5]) for i in range(0, len(data), 5)]

    def nothing_more(self) -> bool:
        """Whether the reader sent nothing further within a moment."""
        connection = self.connection
        connection.settimeout(0.2)
        try:
            return connection.recv(5) == b""
        except TimeoutError:
            return True

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
        self._listener.close()


class _Clock:
    """A clock the test sets; the reader reads it on every arrival and retune."""

    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def server():
    server = _Server()
    yield server
    server.close()


@pytest.fixture
def no_margins(monkeypatch):
    """Nothing dropped after opening and none of the margin after retuning, so that the
    tests see exactly what the lag estimate drops."""
    monkeypatch.setattr(rtl_tcp, "DISCARD_AFTER_OPEN_SAMPLES", 0)
    monkeypatch.setattr(rtl_tcp, "RETUNE_MARGIN_SAMPLES", 0)


def _reader(server: _Server, clock=None, **settings) -> RtlTcpReader:
    options = {} if clock is None else {"clock": clock}
    return RtlTcpReader(
        RtlTcpSettings("127.0.0.1", server.port, **settings), 178_352_000, tag=7, **options
    )


def _bytes(count: int, first: int = 0) -> bytes:
    """`count` samples whose I counts up from `first` and Q down from 255."""
    data = bytearray()
    for index in range(count):
        data += bytes([(first + index) % 256, 255 - (first + index) % 256])
    return bytes(data)


def test_samples_are_scaled_like_pyrtlsdr():
    samples = to_samples(bytes([0, 255, 255, 0, 127, 128]))

    assert samples.dtype == np.complex64
    np.testing.assert_allclose(samples, [-1 + 1j, 1 - 1j, (-0.5 + 0.5j) / 127.5], atol=1e-7)


def test_the_server_is_written_with_ipv6_in_brackets():
    assert RtlTcpSettings("radio-pi").server == "radio-pi:1234"
    assert RtlTcpSettings("::1", 1235).server == "[::1]:1235"


def test_open_reads_the_header_and_sends_every_setting(server):
    reader = _reader(server, gain=29.7, freq_correction_ppm=-3)

    info = reader.open()

    assert info == ServerInfo(tuner_type=5, gain_count=29)
    assert info.tuner == "R820T"
    assert server.commands(6) == [
        (CMD_SET_SAMPLE_RATE, round(SAMPLE_RATE)),
        (CMD_SET_FREQ_CORRECTION, 2**32 - 3),
        (CMD_SET_FREQUENCY, 178_352_000),
        (CMD_SET_AGC_MODE, 0),
        (CMD_SET_GAIN_MODE, 1),
        (CMD_SET_GAIN, 297),
    ]
    reader.close()


def test_automatic_gain_leaves_it_to_both_agcs(server):
    reader = _reader(server, gain="auto")

    reader.open()

    assert server.commands(5)[3:] == [(CMD_SET_AGC_MODE, 1), (CMD_SET_GAIN_MODE, 0)]
    assert server.nothing_more()
    reader.close()


def test_the_start_after_opening_is_dropped(server, no_margins, monkeypatch):
    monkeypatch.setattr(rtl_tcp, "DISCARD_AFTER_OPEN_SAMPLES", 4)
    clock = _Clock()
    reader = _reader(server, clock)
    reader.open()
    server.send(_bytes(10))
    clock.now += 0.5

    tag, block = reader.read_block(6)

    assert tag == 7
    np.testing.assert_array_equal(block, to_samples(_bytes(10)[8:]))
    reader.close()


def test_an_odd_byte_waits_for_its_partner(server, no_margins):
    reader = _reader(server)
    reader.open()
    data = _bytes(3)

    server.send(data[:3])
    first = reader.read_samples(1)
    server.send(data[3:])
    rest = reader.read_samples(2)

    np.testing.assert_array_equal(np.concatenate([first, rest]), to_samples(data))
    reader.close()


def test_retuning_drops_what_the_server_produced_before_the_command(server, no_margins):
    clock = _Clock()
    reader = _reader(server, clock)
    reader.open()
    server.commands(5)
    server.send(_bytes(8))
    clock.now += 0.5
    assert reader.read_block(8)[0] == 7

    # Four samples' time after the last arrival: by then the server has produced four
    # more, still on the old frequency.
    clock.now += 4 / SAMPLE_RATE
    reader.retune(180_064_000, tag=8)
    server.send(_bytes(8, first=8))
    clock.now += 4 / SAMPLE_RATE

    tag, block = reader.read_block(4)

    assert server.commands(1) == [(CMD_SET_FREQUENCY, 180_064_000)]
    assert tag == 8
    np.testing.assert_array_equal(block, to_samples(_bytes(8, first=8)[8:]))
    assert reader.center_freq == 180_064_000
    reader.close()


def test_the_margin_is_dropped_after_retuning(server, no_margins, monkeypatch):
    monkeypatch.setattr(rtl_tcp, "RETUNE_MARGIN_SAMPLES", 3)
    clock = _Clock()
    reader = _reader(server, clock)
    reader.open()
    server.send(_bytes(8))
    clock.now += 0.5
    reader.read_block(8)

    reader.retune(180_064_000, tag=8)
    server.send(_bytes(8, first=8))
    clock.now += 8 / SAMPLE_RATE

    tag, block = reader.read_block(5)

    assert tag == 8
    np.testing.assert_array_equal(block, to_samples(_bytes(8, first=8)[6:]))
    reader.close()


def test_samples_not_handed_out_are_dropped_with_a_new_tag(server, no_margins):
    clock = _Clock()
    reader = _reader(server, clock)
    reader.open()
    server.commands(5)
    server.send(_bytes(8))
    clock.now += 0.5
    reader.read_block(4)  # four more of the old tuning wait

    reader.retune(178_352_000, tag=8)  # the same frequency: nothing to switch
    server.send(_bytes(4, first=8))
    clock.now += 4 / SAMPLE_RATE

    tag, block = reader.read_block(4)

    assert tag == 8
    np.testing.assert_array_equal(block, to_samples(_bytes(4, first=8)))
    assert server.nothing_more()
    reader.close()


def test_the_scan_retunes_through_the_setter(server, no_margins):
    reader = _reader(server)
    reader.open()
    server.commands(5)

    reader.center_freq = 180_064_000

    assert server.commands(1) == [(CMD_SET_FREQUENCY, 180_064_000)]
    reader.close()


def test_no_server_set():
    connect = Mock()
    reader = RtlTcpReader(RtlTcpSettings(" "), 178_352_000, connect=connect)

    with pytest.raises(RtlTcpError, match="No rtl_tcp server set"):
        reader.open()
    connect.assert_not_called()


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (socket.gaierror(11001, "getaddrinfo failed"), "rtl_tcp server radio-pi:1234 not found"),
        (ConnectionRefusedError(), "radio-pi:1234 refuses the connection"),
        (TimeoutError(), "No answer from the rtl_tcp server radio-pi:1234"),
        (OSError(10065, "unreachable"), "Could not connect to the rtl_tcp server radio-pi:1234"),
    ],
)
def test_a_server_out_of_reach_is_reported(failure, message):
    reader = RtlTcpReader(
        RtlTcpSettings("radio-pi"), 178_352_000, connect=Mock(side_effect=failure)
    )

    with pytest.raises(RtlTcpError, match=message):
        reader.open()


def test_a_server_busy_with_another_client_sends_nothing():
    server = _Server(header=None)
    reader = RtlTcpReader(
        RtlTcpSettings("127.0.0.1", server.port), 178_352_000, header_timeout_seconds=0.2
    )

    with pytest.raises(RtlTcpError, match="sends nothing"):
        reader.open()
    server.close()


def test_another_protocol_is_refused():
    server = _Server(header=b"HTTP/1.1 400 Bad")
    reader = RtlTcpReader(RtlTcpSettings("127.0.0.1", server.port), 178_352_000)

    with pytest.raises(RtlTcpError, match="is not an rtl_tcp server"):
        reader.open()
    server.close()


def test_a_server_closing_before_the_header():
    server = _Server(header=b"")
    reader = RtlTcpReader(RtlTcpSettings("127.0.0.1", server.port), 178_352_000)

    with pytest.raises(RtlTcpError):
        reader.open()
    server.close()


def test_a_server_closing_while_reading(server, no_margins):
    reader = _reader(server)
    reader.open()
    server.commands(5)  # read, or Windows resets the connection instead of closing it
    server.send(_bytes(2))
    server.connection.close()

    with pytest.raises(RtlTcpError, match="closed the connection"):
        reader.read_block(4)
    reader.close()


def test_a_stalled_stream_is_reported(server):
    reader = RtlTcpReader(
        RtlTcpSettings("127.0.0.1", server.port),
        178_352_000,
        poll_seconds=0.02,
        no_data_timeout_seconds=0.2,
    )
    reader.open()

    with pytest.raises(RtlTcpError, match="no longer delivers samples"):
        reader.read_block(4)
    reader.close()


def test_cancel_ends_a_waiting_read(server):
    reader = RtlTcpReader(RtlTcpSettings("127.0.0.1", server.port), 178_352_000, poll_seconds=0.02)
    reader.open()
    threading.Timer(0.1, reader.cancel).start()

    assert reader.read_block(4) is None
    with pytest.raises(RtlTcpError, match="cancelled"):
        reader.read_samples(4)
    reader.close()


def test_the_receive_buffer_is_asked_for_where_the_system_grants_it(tmp_path):
    rmem_max = tmp_path / "rmem_max"
    assert receive_buffer_bytes("win32", rmem_max) == rtl_tcp.RECEIVE_BUFFER_BYTES
    # Linux would cap it and leave its own sizing off: better not asked for.
    rmem_max.write_text("212992", encoding="ascii")
    assert receive_buffer_bytes("linux", rmem_max) is None
    rmem_max.write_text("16777216", encoding="ascii")
    assert receive_buffer_bytes("linux", rmem_max) == rtl_tcp.RECEIVE_BUFFER_BYTES
    assert receive_buffer_bytes("linux", tmp_path / "missing") is None


def test_the_connection_gets_its_receive_buffer_before_connecting(server):
    connection = open_connection(("127.0.0.1", server.port), 5.0)

    granted = connection.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    connection.close()
    if receive_buffer_bytes() is not None:
        assert granted >= rtl_tcp.RECEIVE_BUFFER_BYTES


def test_a_refused_connection_raises_like_create_connection():
    listener = socket.create_server(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()

    with pytest.raises(OSError):
        open_connection(("127.0.0.1", port), 5.0)


def test_no_lag_without_arrivals():
    lag = StreamLag()

    assert lag.produced(10.0) is None
    assert lag.samples(10.0, 0) == 0.0


def test_lag_grows_while_nothing_arrives():
    lag = StreamLag()
    lag.add(10.0, 100)

    assert lag.samples(10.0, 100) == 0.0
    assert lag.samples(10.5, 100) == pytest.approx(SAMPLE_RATE / 2)


def test_lag_counts_what_arrives_late():
    lag = StreamLag()
    lag.add(10.0, 100)
    lag.add(10.5, 100 + SAMPLE_RATE / 2)  # in step
    lag.add(11.0, 100 + SAMPLE_RATE / 2)  # nothing for half a second

    assert lag.samples(11.0, 100 + SAMPLE_RATE / 2) == pytest.approx(SAMPLE_RATE / 2)
    assert lag.produced(11.0) == pytest.approx(100 + SAMPLE_RATE)


def test_lag_forgets_minima_out_of_its_window():
    lag = StreamLag(window_seconds=60.0)
    lag.add(0.0, 0)
    # A queue that never emptied since: five seconds behind for good.
    lag.add(100.0, round(95 * SAMPLE_RATE))

    assert lag.samples(100.0, round(95 * SAMPLE_RATE)) == 0.0
