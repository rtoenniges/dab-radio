"""IQ samples from an rtl_tcp server, read like pyrtlsdr's synchronous reads.

rtl_tcp (part of osmocom rtl-sdr) serves one RTL-SDR stick over TCP. Its protocol is
nowhere written down as a standard, but it is the one in use: besides rtl_tcp itself,
rsp_tcp (SDRplay) and the Android app "RTL-SDR driver" serve it, and SDR#, SDR++, GQRX,
welle.io and Qt-DAB read it. As implemented in `rtl_tcp.c`:

- Right after the connection the server sends 12 bytes: the magic "RTL0", then the type
  of the tuner and the number of its gain steps, each a big-endian uint32. After that
  nothing but the samples as they come from the stick: I and Q interleaved, unsigned
  8 bit around 127.5.
- The client sends commands of 5 bytes: the command, then a big-endian uint32. Nothing is
  answered or acknowledged.
- The server serves one client at a time; a second one is accepted by the system but
  hears nothing until the first has gone. The stick keeps its settings from one client
  to the next, so all of them are sent on connecting.

`RtlTcpReader` hands out blocks like `sdr.rspduo_reader.RspDuoReader`: reception
(`sdr.worker.RtlTcpWorker`) reads blocks of two frames in a loop, the channel scan
(`sdr.scanner.ChannelScanner`) uses the same calls as on the stick (`read_samples`,
`center_freq`, `close`).

Retuning is the delicate part, since nothing tells where the samples of the new frequency
begin. Behind the command lies whatever is on its way: the USB transfer being filled in
the server (rtl_tcp's are 64 ms long), the queue of the server (up to 500 transfers, half
a minute, should the client fall behind), and the buffers of both systems and the
network in between. How much that is, the stream itself tells: the server produces at
the sample rate, so the sample clock run since some moment, less the samples received,
grows by every sample still on the way (`StreamLag`). Retuning drops every sample that
the server can have produced before the command reached the stick, by that estimate,
plus a margin for what it cannot see (`RETUNE_MARGIN_SAMPLES`).
"""

from __future__ import annotations

import logging
import socket
import struct
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from dab.constants import SAMPLE_RATE
from i18n import tr
from sdr.device import DISCARD_AFTER_OPEN_SAMPLES
from sdr.exceptions import RtlTcpError

logger = logging.getLogger(__name__)

DEFAULT_PORT = 1234
PORT_RANGE = (1, 65535)

HEADER_MAGIC = b"RTL0"
_HEADER = struct.Struct(">4sII")
_COMMAND = struct.Struct(">BI")

# The commands of rtl_tcp used here (the others set test mode, direct sampling, offset
# tuning, the crystals, single IF stages, gain by index and the bias tee).
CMD_SET_FREQUENCY = 0x01  # Hz
CMD_SET_SAMPLE_RATE = 0x02  # samples per second
CMD_SET_GAIN_MODE = 0x03  # 0 = the tuner's AGC, 1 = manual gain
CMD_SET_GAIN = 0x04  # tenths of a dB; librtlsdr takes the nearest step
CMD_SET_FREQ_CORRECTION = 0x05  # ppm, signed
CMD_SET_AGC_MODE = 0x08  # the RTL2832U's digital AGC, 0 or 1

# The tuner types of librtlsdr (`enum rtlsdr_tuner`), as rtl_tcp reports them.
TUNER_NAMES = {1: "E4000", 2: "FC0012", 3: "FC0013", 4: "FC2580", 5: "R820T", 6: "R828D"}

CONNECT_TIMEOUT_SECONDS = 5.0
# rtl_tcp sends its header right after accepting. A server serving another client
# accepts the connection all the same (the backlog of `listen`) and sends nothing.
HEADER_TIMEOUT_SECONDS = 3.0
# How often a waiting read looks whether it was cancelled.
POLL_SECONDS = 0.1
# Without a single byte for this long the stream counts as broken.
NO_DATA_TIMEOUT_SECONDS = 3.0
# Room in the system for the samples the reading thread has not taken yet (4 s at
# 2.048 MS/s): the system fills it without the GIL, as the API's thread fills the ring of
# the RSPduo (`sdr.packet_ring`, 3 s). When the GUI holds the GIL for long, turning the
# detail view, the reading thread falls behind by seconds; with only a second of room the
# rest piled up in rtl_tcp's queue (2.2 s, its "ll+" lines), and came back only as fast as
# the network carries more than the stream, little over Wi-Fi: the audio ran dry while
# the RSPduo played on (15.10.2026). Asked for before connecting, since the window scale
# is agreed on in the handshake; see `receive_buffer_bytes` for Linux.
RECEIVE_BUFFER_BYTES = 16 * 1024 * 1024
# Read up to this much at a time (1 s): a reading thread that has fallen behind takes
# all that waits at its next turn, as `RspDuoReader` drains the whole ring. With rtl_tcp's
# 64 ms per read it caught up with nothing while the GIL was scarce, needing several turns
# for each.
READ_BYTES = 4 * 1024 * 1024
# After a read of less than this (16 ms), the reading thread has caught up and pauses
# (20 ms; the receive buffer holds seconds).
SHORT_READ_BYTES = 64 * 1024
SHORT_READ_PAUSE_SECONDS = 0.02
# Where Linux keeps the most a program may ask for as receive buffer.
_RMEM_MAX = Path("/proc/sys/net/core/rmem_max")

# Dropped after the samples the server can have produced before a retune reached the
# stick (0.25 s): what the lag estimate cannot see. The USB transfer being filled when
# the command arrives (64 ms with rtl_tcp), the command's way to the server, the tuner
# switching and settling (`sdr.device.RETUNE_SETTLE_SAMPLES`, 50 ms), and the jitter of
# the network.
RETUNE_MARGIN_SAMPLES = 512_000

# The lag is measured against the least seen within this window. Longer windows see a
# queue that never empties as no lag at all less often; shorter ones follow the drift
# between the clocks of server and receiver more closely (100 ppm are 6 ms a minute).
LAG_WINDOW_SECONDS = 60.0
# From this lag on the log says so (once per rise): the network does not keep up, or
# this computer did not let the reading thread read.
LAG_WARNING_SAMPLES = SAMPLE_RATE


class StreamLag:
    """How many samples the server has produced beyond those received, above the least
    such number seen lately.

    The server produces samples at a steady rate. At time `t`, after `n` samples received
    in all, `t * rate - n` is the time the stream started, in samples, plus the samples
    on their way. Its least value over the last `LAG_WINDOW_SECONDS` is the start plus
    the least on their way, so the difference to it is what is on the way beyond that:
    in a queue of the server or in the buffers of the network.
    """

    def __init__(
        self, rate: float = SAMPLE_RATE, window_seconds: float = LAG_WINDOW_SECONDS
    ) -> None:
        self._rate = rate
        self._window = window_seconds
        # (second, least offset within it), oldest first.
        self._minima: deque[tuple[int, float]] = deque()

    def add(self, now: float, received: int) -> None:
        """Notes that `received` samples have arrived in all by `now`."""
        offset = now * self._rate - received
        second = int(now)
        minima = self._minima
        if minima and minima[-1][0] == second:
            if offset < minima[-1][1]:
                minima[-1] = (second, offset)
        else:
            minima.append((second, offset))
        while minima and minima[0][0] < now - self._window:
            minima.popleft()

    def produced(self, at: float) -> float | None:
        """How many samples the server had produced by `at`, as far as the arrivals
        show: those received while nothing more was on the way. None before the first."""
        if not self._minima:
            return None
        return at * self._rate - min(offset for _, offset in self._minima)

    def samples(self, now: float, received: int) -> float:
        """The samples on their way at `now` beyond the least seen."""
        produced = self.produced(now)
        return 0.0 if produced is None else max(0.0, produced - received)


@dataclass(frozen=True)
class RtlTcpSettings:
    """Server and stick settings for rtl_tcp; gain and correction as for the stick here."""

    host: str
    port: int = DEFAULT_PORT
    gain: str | float = "auto"  # dB, or "auto" for the tuner's AGC
    freq_correction_ppm: int = 0

    @property
    def server(self) -> str:
        """The server as "host:port", an IPv6 address in brackets."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{host}:{self.port}"


@dataclass(frozen=True)
class ServerInfo:
    """What the server tells about its stick on connecting."""

    tuner_type: int
    gain_count: int

    @property
    def tuner(self) -> str:
        """The tuner's name; that of librtlsdr's type number for one it does not know."""
        return TUNER_NAMES.get(self.tuner_type, f"#{self.tuner_type}")


def receive_buffer_bytes(platform: str = sys.platform, rmem_max: Path = _RMEM_MAX) -> int | None:
    """The receive buffer to ask for: RECEIVE_BUFFER_BYTES, or None to leave the size to
    the system.

    Linux caps what is asked for at net.core.rmem_max (208 KiB on Raspberry Pi OS), and
    whoever asks switches its automatic sizing off, which grows the buffer up to the
    third value of net.ipv4.tcp_rmem (6 MiB there); asked for in vain, the buffer would
    end up smaller than without asking. Raised as root with `sysctl -w
    net.core.rmem_max=16777216`, it is asked for.
    """
    if not platform.startswith("linux"):
        return RECEIVE_BUFFER_BYTES
    try:
        allowed = int(rmem_max.read_text(encoding="ascii"))
    except OSError, ValueError:
        return None
    return RECEIVE_BUFFER_BYTES if allowed >= RECEIVE_BUFFER_BYTES else None


def open_connection(address: tuple[str, int], timeout: float) -> socket.socket:
    """Connects like `socket.create_connection`, with the receive buffer set before
    connecting (`receive_buffer_bytes`).

    Raises:
        OSError: as `socket.create_connection`, the error of the last address tried.
    """
    host, port = address
    size = receive_buffer_bytes()
    failure: OSError | None = None
    for family, kind, protocol, _name, target in socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM
    ):
        sock = socket.socket(family, kind, protocol)
        try:
            if size is not None:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, size)
            sock.settimeout(timeout)
            sock.connect(target)
        except OSError as exc:
            sock.close()
            failure = exc
            continue
        return sock
    raise failure if failure is not None else OSError(f"no address for {host}")


@cache
def _sample_table() -> np.ndarray:
    """complex64 for each pair of bytes read as a little-endian uint16: I in the low
    byte, Q in the high one, scaled to full scale 1.0 as pyrtlsdr does."""
    levels = (np.arange(256, dtype=np.float32) - np.float32(127.5)) / np.float32(127.5)
    table = np.empty(65536, np.complex64)
    table.real = np.tile(levels, 256)
    table.imag = np.repeat(levels, 256)
    return table


def to_samples(data: bytes | bytearray | memoryview) -> np.ndarray:
    """The samples in `data` (I and Q interleaved, unsigned 8 bit; an even length)."""
    return _sample_table()[np.frombuffer(data, dtype="<u2")]


class RtlTcpReader:
    """An rtl_tcp server as a blocking source of IQ blocks.

    `read_block` runs in one thread (the reading one); `retune` and `cancel` may be
    called from any other. Each block belongs to one tuning, identified by the `tag`
    given to `retune` (the initial one at construction): a block never mixes the
    samples of two frequencies.
    """

    def __init__(
        self,
        settings: RtlTcpSettings,
        center_freq_hz: float,
        tag: int = 0,
        connect: Callable[..., socket.socket] = open_connection,
        clock: Callable[[], float] = time.monotonic,
        poll_seconds: float = POLL_SECONDS,
        no_data_timeout_seconds: float = NO_DATA_TIMEOUT_SECONDS,
        header_timeout_seconds: float = HEADER_TIMEOUT_SECONDS,
    ) -> None:
        self._settings = settings
        self._center_freq_hz = center_freq_hz
        self._connect = connect
        self._clock = clock
        self._poll_seconds = poll_seconds
        self._no_data_timeout = no_data_timeout_seconds
        self._header_timeout = header_timeout_seconds
        self._socket: socket.socket | None = None
        self._cancelled = threading.Event()
        # Keeps the commands of `retune` and `close` apart.
        self._send_lock = threading.Lock()
        # What `retune` tells the reading thread: (tag of the samples from the latest
        # retune on, when its command left, or None if nothing was switched). Replaced
        # as a whole, read once per arrival, without a lock; see
        # `RspDuoReader._tuning`. `open` publishes the first one.
        self._tuning: tuple[int, float | None] = (tag, None)
        # The reading thread's own state.
        self._tag = tag  # of the samples in `_chunks`
        self._resolved: tuple[int, float | None] | None = None  # the tuning applied
        self._keep_from = 0  # samples before this one are dropped
        self._received = 0  # samples received in all
        self._lag = StreamLag()
        self._lag_reported = False
        self._buffer = bytearray(READ_BYTES)
        self._carry = 0  # an odd byte left at the start of `_buffer`
        self._chunks: list[np.ndarray] = []
        self._pending = 0  # samples in `_chunks`

    @property
    def center_freq(self) -> float:
        return self._center_freq_hz

    @center_freq.setter
    def center_freq(self, center_freq_hz: float) -> None:
        """Retunes, as pyrtlsdr's setter of the same name; for the channel scan."""
        self.retune(center_freq_hz, self._tuning[0] + 1)

    @property
    def lag_samples(self) -> float:
        """The samples on their way beyond the least seen (`StreamLag`)."""
        return self._lag.samples(self._clock(), self._received)

    def open(self) -> ServerInfo:
        """Connects, reads the header and sends all settings.

        Raises:
            RtlTcpError: if the server cannot be reached, sends nothing or is no
                rtl_tcp server. The connection is closed again.
        """
        settings = self._settings
        server = settings.server
        if not settings.host.strip():
            raise RtlTcpError(tr("No rtl_tcp server set."))
        try:
            sock = self._connect((settings.host.strip(), settings.port), CONNECT_TIMEOUT_SECONDS)
        except socket.gaierror as exc:
            raise RtlTcpError(
                tr("rtl_tcp server {server} not found.").format(server=server)
            ) from exc
        except ConnectionRefusedError as exc:
            raise RtlTcpError(
                tr("{server} refuses the connection. Is rtl_tcp running there?").format(
                    server=server
                )
            ) from exc
        except TimeoutError as exc:
            raise RtlTcpError(
                tr("No answer from the rtl_tcp server {server}.").format(server=server)
            ) from exc
        except OSError as exc:
            raise RtlTcpError(
                tr("Could not connect to the rtl_tcp server {server}: {error}").format(
                    server=server, error=exc
                )
            ) from exc
        self._socket = sock
        try:
            info = self._read_header()
            self._configure()
        except BaseException:
            self.close()
            raise
        return info

    def _read_header(self) -> ServerInfo:
        sock = self._socket
        assert sock is not None
        server = self._settings.server
        header = bytearray()
        sock.settimeout(self._header_timeout)
        try:
            while len(header) < _HEADER.size:
                data = sock.recv(_HEADER.size - len(header))
                if not data:
                    raise RtlTcpError(
                        tr("The rtl_tcp server {server} closed the connection.").format(
                            server=server
                        )
                    )
                header += data
        except TimeoutError as exc:
            raise RtlTcpError(
                tr(
                    "The rtl_tcp server {server} sends nothing. It serves one program at a "
                    "time; is another one connected to it?"
                ).format(server=server)
            ) from exc
        except OSError as exc:
            raise RtlTcpError(
                tr("The connection to the rtl_tcp server was lost: {error}").format(error=exc)
            ) from exc
        magic, tuner_type, gain_count = _HEADER.unpack(header)
        if magic != HEADER_MAGIC:
            raise RtlTcpError(tr("{server} is not an rtl_tcp server.").format(server=server))
        sock.settimeout(self._poll_seconds)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            logger.debug("Could not set TCP_NODELAY", exc_info=True)
        info = ServerInfo(tuner_type, gain_count)
        logger.info(
            "rtl_tcp server %s: tuner %s with %d gain steps; receive buffer %d KiB",
            server,
            info.tuner,
            gain_count,
            sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF) // 1024,
        )
        return info

    def _configure(self) -> None:
        """Sends every setting, in the order of `sdr.device.open_rtlsdr`: the stick keeps
        what the previous client set."""
        settings = self._settings
        gain = settings.gain
        manual = gain != "auto"
        commands = [
            (CMD_SET_SAMPLE_RATE, round(SAMPLE_RATE)),
            (CMD_SET_FREQ_CORRECTION, settings.freq_correction_ppm & 0xFFFFFFFF),
            (CMD_SET_FREQUENCY, round(self._center_freq_hz)),
            # The digital AGC follows the gain mode, see `open_rtlsdr`.
            (CMD_SET_AGC_MODE, 0 if manual else 1),
            (CMD_SET_GAIN_MODE, 1 if manual else 0),
        ]
        if manual:
            commands.append((CMD_SET_GAIN, round(float(gain) * 10)))
        with self._send_lock:
            # The samples before the frequency took effect, and the stick's own leftovers
            # of the frequency before (`DISCARD_AFTER_OPEN_SAMPLES`), are dropped.
            self._tuning = (self._tuning[0], self._clock())
            for command, value in commands:
                self._send(command, value)

    def read_block(self, num_samples: int) -> tuple[int, np.ndarray] | None:
        """The next `num_samples` samples of one tuning, with its tag; waits for them.

        Returns:
            (tag, complex64 samples), or None once `cancel` has been called.

        Raises:
            RtlTcpError: if the connection broke or the server delivers nothing.
        """
        sock = self._socket
        if sock is None:
            return None
        deadline = self._clock() + self._no_data_timeout
        while True:
            if self._cancelled.is_set():
                return None
            if self._tuning != self._resolved and self._resolved is not None:
                self._apply_tuning(self._received)
            if self._pending >= num_samples:
                return self._tag, self._take(num_samples)
            try:
                count = sock.recv_into(memoryview(self._buffer)[self._carry :])
            except TimeoutError:
                if self._clock() > deadline:
                    raise RtlTcpError(
                        tr(
                            "The rtl_tcp server no longer delivers samples. Please check "
                            "the network and restart reception."
                        )
                    ) from None
                continue
            except OSError as exc:
                if self._cancelled.is_set():
                    return None
                raise RtlTcpError(
                    tr("The connection to the rtl_tcp server was lost: {error}").format(error=exc)
                ) from exc
            if not count:
                if self._cancelled.is_set():
                    return None
                raise RtlTcpError(tr("The rtl_tcp server closed the connection."))
            now = self._clock()
            deadline = now + self._no_data_timeout
            self._push(count, now)
            if count < SHORT_READ_BYTES and self._pending < num_samples:
                # Caught up with the stream: wait for more instead of taking it a few
                # network packets at a time, each costing a turn at the GIL.
                self._cancelled.wait(SHORT_READ_PAUSE_SECONDS)

    def read_samples(self, num_samples: int) -> np.ndarray:
        """The next `num_samples` samples, as pyrtlsdr's method of the same name.

        Raises:
            RtlTcpError: as `read_block`, and once `cancel` has been called.
        """
        block = self.read_block(num_samples)
        if block is None:
            raise RtlTcpError(tr("Reading from the rtl_tcp server cancelled."))
        return block[1]

    def retune(self, center_freq_hz: float, tag: int) -> None:
        """Tunes to another frequency without interrupting the stream; the samples of the
        new frequency carry `tag`.

        May be called from any thread while another one reads; returns as soon as the
        command is sent, without waiting for the reading thread. That thread drops the
        samples it has not handed out yet and everything the server can have produced
        before the command reached it (see the module's docstring).

        Raises:
            RtlTcpError: if the command could not be sent. No samples are handed out from
                then on; the caller should stop reading.
        """
        with self._send_lock:
            if center_freq_hz == self._center_freq_hz:
                # Nothing to switch; the samples from here on carry the new tag.
                self._tuning = (tag, None)
                return
            self._tuning = (tag, self._clock())
            self._send(CMD_SET_FREQUENCY, round(center_freq_hz))
            self._center_freq_hz = center_freq_hz

    def cancel(self) -> None:
        """Makes a waiting `read_block` return None, from any thread."""
        self._cancelled.set()

    def close(self) -> None:
        """Closes the connection; the server then waits for the next client. Safe to call
        more than once."""
        self._cancelled.set()
        with self._send_lock:
            sock, self._socket = self._socket, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                logger.debug("Closing the connection failed", exc_info=True)

    def _send(self, command: int, value: int) -> None:
        sock = self._socket
        if sock is None:
            raise RtlTcpError(tr("The rtl_tcp server closed the connection."))
        try:
            sock.sendall(_COMMAND.pack(command, value))
        except OSError as exc:
            raise RtlTcpError(
                tr("Sending to the rtl_tcp server failed: {error}").format(error=exc)
            ) from exc

    def _push(self, count: int, now: float) -> None:
        """Takes the bytes just received into `_buffer` behind an odd one carried over."""
        total = self._carry + count
        even = total & ~1
        samples = to_samples(memoryview(self._buffer)[:even])
        if total & 1:
            self._buffer[0] = self._buffer[total - 1]
        self._carry = total & 1
        first = self._received
        self._received += len(samples)
        self._lag.add(now, self._received)
        self._check_lag(now)
        if self._tuning != self._resolved:
            self._apply_tuning(first)
        start = max(0, self._keep_from - first)
        if start < len(samples):
            self._chunks.append(samples[start:])
            self._pending += len(samples) - start

    def _apply_tuning(self, received_before: int) -> None:
        """Takes up what `retune` (or `open`) published, before the samples from
        `received_before` on: a new tag drops the samples not handed out yet, and a
        command sent drops what can have been produced before it reached the stick."""
        tuning = self._tuning
        tag, sent_at = tuning
        if self._resolved is None:
            # Opening: the stick's leftovers of the frequency it was tuned to before.
            self._keep_from = DISCARD_AFTER_OPEN_SAMPLES
        if tag != self._tag:
            self._tag = tag
            self._chunks = []
            self._pending = 0
        if sent_at is not None:
            produced = self._lag.produced(sent_at)
            if produced is None:
                produced = self._received  # nothing arrived yet
            self._keep_from = max(
                self._keep_from, received_before, round(produced) + RETUNE_MARGIN_SAMPLES
            )
        self._resolved = tuning

    def _check_lag(self, now: float) -> None:
        lag = self._lag.samples(now, self._received)
        if lag > LAG_WARNING_SAMPLES and not self._lag_reported:
            self._lag_reported = True
            logger.warning(
                "The rtl_tcp stream lags %.1f s behind: the network does not keep up, or "
                "the reading thread did not get the GIL",
                lag / SAMPLE_RATE,
            )
        elif lag < LAG_WARNING_SAMPLES / 2:
            self._lag_reported = False

    def _take(self, num_samples: int) -> np.ndarray:
        samples = np.concatenate(self._chunks) if len(self._chunks) > 1 else self._chunks[0]
        self._chunks = [samples[num_samples:]] if len(samples) > num_samples else []
        self._pending = len(samples) - num_samples
        return samples[:num_samples]
