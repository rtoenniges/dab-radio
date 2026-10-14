"""A small NTRIP caster: passes the RTCM 3 stream of GEPOS on to GNSS receivers.

NTRIP (Networked Transport of RTCM via Internet Protocol, RTCM 10410.1) is how RTK
receivers and programs like RTKLIB get their corrections: a client asks a caster over
HTTP for a mountpoint and then receives the RTCM byte stream. Here the caster is the
radio, and the stream is the one GEPOS broadcasts over DAB+, frame by frame exactly as
it came out of the data groups (see `dab.rtcm`): nothing is decoded or re-encoded on
the way, so every client sees what a receiver of the broadcast would.

Understood are:

- NTRIP 1.0: ``GET /MOUNT HTTP/1.0``, answered with ``ICY 200 OK`` and the bare stream.
- NTRIP 2.0: ``GET /MOUNT HTTP/1.1`` with ``Ntrip-Version: Ntrip/2.0``, answered with
  ``HTTP/1.1 200 OK`` and the stream in chunked transfer encoding.
- ``GET /`` or an unknown mountpoint: the sourcetable, with one STR record per
  mountpoint.
- Plain TCP: a client that sends no request within `RAW_WAIT_SECONDS` gets the stream
  of the first mountpoint without any header (RTKLIB's "TCP Client", netcat).

A client may send its position as NMEA GGA sentences, as rovers do for virtual
reference stations (in NTRIP 2.0 also in the ``Ntrip-GGA`` header). The last one is
kept per client; the radio can take it as its own position.

There is no authentication: the caster is meant for the local network. Every client
has a queue of its own; the signal processing only appends to it and never waits for
the network. A client that falls more than `QUEUE_BATCHES` batches behind is dropped.
"""

from __future__ import annotations

import queue
import re
import select
import socket
import threading
import time
from dataclasses import dataclass, field

from i18n import tr

DEFAULT_PORT = 2101  # registered for NTRIP (IANA)
# How long a new connection may stay silent before it counts as a plain TCP client.
RAW_WAIT_SECONDS = 1.0
# How long a request may take to arrive completely.
REQUEST_TIMEOUT_SECONDS = 5.0
MAX_REQUEST_BYTES = 8192
# Batches of frames a client may lag behind (GEPOS sends about 7 per second).
QUEUE_BATCHES = 256
# A client that does not take its data within this time is dropped.
SEND_TIMEOUT_SECONDS = 10.0
# Longest NMEA sentence kept while waiting for its line end (82 by the standard).
_MAX_NMEA_LINE = 256
_POLL_SECONDS = 0.2
_SERVER_NAME = "DAB+ Radio NTRIP-Caster/1.0"

PROTOCOL_NTRIP1 = "NTRIP 1.0"
PROTOCOL_NTRIP2 = "NTRIP 2.0"
PROTOCOL_TCP = "TCP"

_RTCM_PREAMBLE = 0xD3


class NtripCasterError(OSError):
    """The caster cannot start. The message is shown in the GUI and therefore translated."""


@dataclass(frozen=True)
class GgaPosition:
    """A position a client reported in an NMEA GGA sentence."""

    latitude: float  # degrees, north positive
    longitude: float  # degrees, east positive
    height: float | None  # above the ellipsoid (altitude plus geoid separation), m
    quality: int  # 1 GNSS, 2 DGNSS, 4 RTK fixed, 5 RTK float, ...
    satellites: int | None
    received_at: float  # POSIX time


def _nmea_checksum_ok(sentence: str) -> bool:
    if "*" not in sentence:
        return True  # the checksum is optional in NMEA 0183
    body, _, checksum = sentence[1:].partition("*")
    value = 0
    for character in body:
        value ^= ord(character)
    try:
        return value == int(checksum[:2], 16)
    except ValueError:
        return False


def _nmea_angle(value: str, hemisphere: str, degree_digits: int) -> float:
    angle = float(value[:degree_digits]) + float(value[degree_digits:]) / 60.0
    return -angle if hemisphere in ("S", "W") else angle


def parse_gga(sentence: str, received_at: float | None = None) -> GgaPosition | None:
    """A GGA sentence of any talker ($GPGGA, $GNGGA, ...), or None if the line is no
    valid GGA or reports no fix (quality 0)."""
    sentence = sentence.strip()
    if not re.match(r"^\$[A-Z]{2}GGA,", sentence) or not _nmea_checksum_ok(sentence):
        return None
    fields = sentence.split("*")[0].split(",")
    if len(fields) < 10:
        return None
    try:
        quality = int(fields[6] or 0)
        if quality == 0 or not fields[2] or not fields[4]:
            return None
        latitude = _nmea_angle(fields[2], fields[3], 2)
        longitude = _nmea_angle(fields[4], fields[5], 3)
        satellites = int(fields[7]) if fields[7] else None
        height = None
        if fields[9]:
            separation = float(fields[11]) if len(fields) > 11 and fields[11] else 0.0
            height = float(fields[9]) + separation
    except ValueError:
        return None
    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        return None
    return GgaPosition(
        latitude,
        longitude,
        height,
        quality,
        satellites,
        time.time() if received_at is None else received_at,
    )


def mountpoint_for(scid: int, label: str | None) -> str:
    """The mountpoint of a data component: its service label with letters, digits, '-'
    and '_' only ("GEPOS" stays "GEPOS"); without a label its SCId."""
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", (label or "").strip()).strip("_")
    return name or f"SCID{scid:03X}"


@dataclass(frozen=True)
class ClientStatus:
    address: str
    mountpoint: str
    protocol: str
    connected_at: float  # POSIX time
    bytes_sent: int
    position: GgaPosition | None


@dataclass(frozen=True)
class MountpointStatus:
    name: str
    message_types: tuple[int, ...]  # the RTCM message types seen so far
    bytes_published: int
    seconds: float  # since the first frame


@dataclass(frozen=True)
class CasterStatus:
    running: bool
    port: int | None
    mountpoints: tuple[MountpointStatus, ...] = ()
    clients: tuple[ClientStatus, ...] = ()

    @property
    def latest_position(self) -> GgaPosition | None:
        """The most recent position any client reported."""
        positions = [c.position for c in self.clients if c.position is not None]
        return max(positions, key=lambda p: p.received_at, default=None)


@dataclass
class _Mountpoint:
    name: str
    message_types: set[int] = field(default_factory=set)
    bytes_published: int = 0
    first_frame_at: float | None = None


class _Client:
    def __init__(self, sock: socket.socket, address: str) -> None:
        self.sock = sock
        self.address = address
        self.mountpoint = ""
        self.protocol = PROTOCOL_TCP
        self.connected_at = time.time()
        self.bytes_sent = 0
        self.position: GgaPosition | None = None
        self.batches: queue.Queue[bytes] = queue.Queue(QUEUE_BATCHES)
        self.dropped = False
        self._nmea = bytearray()

    def feed_nmea(self, data: bytes) -> None:
        """Bytes the client sent after its request: GGA sentences, line by line."""
        self._nmea.extend(data)
        *lines, rest = self._nmea.split(b"\n")
        self._nmea = bytearray(rest[-_MAX_NMEA_LINE:])
        for line in lines:
            self.take_gga(line.decode("ascii", "replace"))

    def take_gga(self, line: str) -> None:
        position = parse_gga(line)
        if position is not None:
            self.position = position

    def status(self) -> ClientStatus:
        return ClientStatus(
            self.address,
            self.mountpoint,
            self.protocol,
            self.connected_at,
            self.bytes_sent,
            self.position,
        )


class NtripCaster:
    """Serves RTCM streams under mountpoints. `publish` may be called from any thread;
    each client is served by a thread of its own."""

    def __init__(self, raw_wait_seconds: float = RAW_WAIT_SECONDS) -> None:
        self._raw_wait = raw_wait_seconds
        self._lock = threading.Lock()
        self._mountpoints: dict[str, _Mountpoint] = {}
        self._clients: list[_Client] = []
        self._server: socket.socket | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._port: int | None = None

    # --- Control -------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._server is not None

    @property
    def port(self) -> int | None:
        """The port listened on (the one chosen by the system if started with 0)."""
        return self._port

    def start(self, port: int = DEFAULT_PORT, host: str = "") -> None:
        """Listens on `host`, by default on all interfaces (IPv4 and IPv6).

        Raises:
            NtripCasterError: The port is taken or may not be used.
        """
        self.stop()
        try:
            if not host and socket.has_dualstack_ipv6():
                server = socket.create_server(
                    ("", port), family=socket.AF_INET6, dualstack_ipv6=True
                )
            else:
                server = socket.create_server((host, port))
        except OSError as error:
            raise NtripCasterError(
                tr(
                    "The NTRIP caster cannot open port {port}: {error}. Is another "
                    "program using it?"
                ).format(port=port, error=error.strerror or error)
            ) from error
        server.settimeout(_POLL_SECONDS)
        self._stop.clear()
        self._server = server
        self._port = server.getsockname()[1]
        thread = threading.Thread(target=self._accept_loop, name="ntrip-accept", daemon=True)
        self._threads = [thread]
        thread.start()

    def stop(self) -> None:
        """Closes the port and all connections."""
        server = self._server
        if server is None:
            return
        self._stop.set()
        self._server = None
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            _close(client.sock)
        for thread in list(self._threads):
            thread.join(timeout=2.0)
        _close(server)
        with self._lock:
            self._clients.clear()
        self._threads = []
        self._port = None

    def add_mountpoint(self, name: str) -> None:
        with self._lock:
            self._mountpoints.setdefault(name, _Mountpoint(name))

    def publish(self, mountpoint: str, frames: list[bytes]) -> None:
        """Hands complete RTCM frames to all clients of the mountpoint (made on first
        use). Never blocks."""
        if not frames:
            return
        data = b"".join(frames)
        with self._lock:
            entry = self._mountpoints.get(mountpoint)
            if entry is None:
                entry = self._mountpoints[mountpoint] = _Mountpoint(mountpoint)
            if entry.first_frame_at is None:
                entry.first_frame_at = time.time()
            entry.bytes_published += len(data)
            for frame in frames:
                if len(frame) >= 5 and frame[0] == _RTCM_PREAMBLE:
                    entry.message_types.add(frame[3] << 4 | frame[4] >> 4)
            clients = [c for c in self._clients if c.mountpoint == mountpoint]
        for client in clients:
            try:
                client.batches.put_nowait(data)
            except queue.Full:
                client.dropped = True

    def status(self) -> CasterStatus:
        now = time.time()
        with self._lock:
            mountpoints = tuple(
                MountpointStatus(
                    m.name,
                    tuple(sorted(m.message_types)),
                    m.bytes_published,
                    0.0 if m.first_frame_at is None else now - m.first_frame_at,
                )
                for m in self._mountpoints.values()
            )
            clients = tuple(c.status() for c in self._clients if c.mountpoint)
        return CasterStatus(self.running, self._port, mountpoints, clients)

    # --- Connections ---------------------------------------------------------------

    def _accept_loop(self) -> None:
        server = self._server
        while server is not None and not self._stop.is_set():
            try:
                sock, address = server.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            thread = threading.Thread(
                target=self._serve, args=(sock, _address_text(address)), daemon=True
            )
            self._threads = [t for t in self._threads if t.is_alive()] + [thread]
            thread.start()

    def _serve(self, sock: socket.socket, address: str) -> None:
        client = _Client(sock, address)
        with self._lock:
            self._clients.append(client)
        try:
            if self._answer_request(client):
                self._stream(client)
        except OSError, ValueError:
            pass  # connection lost, or closed by `stop` (select then sees fd -1)
        finally:
            with self._lock:
                if client in self._clients:
                    self._clients.remove(client)
            _close(sock)

    def _answer_request(self, client: _Client) -> bool:
        """Reads the request and answers it. True if the stream is to follow."""
        request = _read_request(client.sock, self._raw_wait)
        if request is None or not request.startswith(b"GET "):
            # Plain TCP: whatever the client sent may be NMEA.
            client.feed_nmea(request or b"")
            with self._lock:
                first = next(iter(self._mountpoints), None)
                if first is None:
                    return False
                client.mountpoint = first
            return True

        head, _, rest = request.partition(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        parts = lines[0].split()
        path = parts[1] if len(parts) > 1 else "/"
        version = parts[2] if len(parts) > 2 else "HTTP/1.0"
        headers = {}
        for line in lines[1:]:
            name, _, value = line.partition(":")
            headers[name.strip().lower()] = value.strip()
        ntrip2 = headers.get("ntrip-version", "").lower() == "ntrip/2.0"
        http11 = ntrip2 or version.upper() == "HTTP/1.1"
        gga = headers.get("ntrip-gga")
        if gga:
            client.take_gga(gga)
        client.feed_nmea(rest)

        name = path.lstrip("/").split("?")[0]
        with self._lock:
            known = name in self._mountpoints
        if not known:
            client.sock.sendall(self._sourcetable_response(http11, ntrip2))
            return False
        if ntrip2:
            client.sock.sendall(
                (
                    "HTTP/1.1 200 OK\r\n"
                    "Ntrip-Version: Ntrip/2.0\r\n"
                    f"Server: {_SERVER_NAME}\r\n"
                    "Content-Type: gnss/data\r\n"
                    "Cache-Control: no-store, no-cache, max-age=0\r\n"
                    "Transfer-Encoding: chunked\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("ascii")
            )
            client.protocol = PROTOCOL_NTRIP2
        else:
            client.sock.sendall(b"ICY 200 OK\r\n\r\n")
            client.protocol = PROTOCOL_NTRIP1
        with self._lock:
            client.mountpoint = name
        return True

    def _stream(self, client: _Client) -> None:
        sock = client.sock
        sock.settimeout(SEND_TIMEOUT_SECONDS)
        chunked = client.protocol == PROTOCOL_NTRIP2
        while not self._stop.is_set() and not client.dropped:
            try:
                data = client.batches.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                data = None
            if data:
                if chunked:
                    sock.sendall(f"{len(data):X}\r\n".encode("ascii") + data + b"\r\n")
                else:
                    sock.sendall(data)
                client.bytes_sent += len(data)
            readable, _, _ = select.select([sock], [], [], 0)
            if readable:
                incoming = sock.recv(4096)
                if not incoming:
                    return  # the client closed the connection
                client.feed_nmea(incoming)

    def _sourcetable_response(self, http11: bool, ntrip2: bool) -> bytes:
        body = self._sourcetable().encode("ascii")
        if http11:
            content_type = "gnss/sourcetable" if ntrip2 else "text/plain"
            status = "HTTP/1.1 200 OK\r\n" + ("Ntrip-Version: Ntrip/2.0\r\n" if ntrip2 else "")
        else:
            content_type = "text/plain"
            status = "SOURCETABLE 200 OK\r\n"
        head = (
            f"{status}Server: {_SERVER_NAME}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        )
        return head.encode("ascii") + body

    def _sourcetable(self) -> str:
        """One STR record per mountpoint (RTCM 10410.1, table 5). Format details: the
        message types seen so far; carrier 0 (no observations), network solution."""
        lines = []
        for status in self.status().mountpoints:
            types = ",".join(str(t) for t in status.message_types)
            bitrate = (
                round(status.bytes_published * 8 / status.seconds) if status.seconds >= 1 else 0
            )
            lines.append(
                f"STR;{status.name};{status.name} (DAB+);RTCM 3.3;{types};0;"
                "GPS+GLO+GAL+BDS;;DEU;51.16;10.45;0;1;DAB+ Radio;none;N;N;"
                f"{bitrate};"
            )
        lines.append("ENDSOURCETABLE")
        return "\r\n".join(lines) + "\r\n"


def _read_request(sock: socket.socket, raw_wait: float) -> bytes | None:
    """The request head (up to the empty line, with whatever followed it), or what a
    client sent that is no request; None if it sent nothing within `raw_wait`."""
    data = bytearray()
    sock.settimeout(raw_wait)
    deadline = time.monotonic() + REQUEST_TIMEOUT_SECONDS
    while b"\r\n\r\n" not in data and len(data) < MAX_REQUEST_BYTES:
        try:
            chunk = sock.recv(4096)
        except TimeoutError:
            if not data:
                return None
            if time.monotonic() > deadline or not data.startswith(b"GET "):
                break
            continue
        if not chunk:
            break
        data.extend(chunk)
        if not b"GET ".startswith(bytes(data[:4])):
            break  # not a request: a plain TCP client sending NMEA
        sock.settimeout(max(0.05, deadline - time.monotonic()))
    return bytes(data)


def _address_text(address: tuple) -> str:
    host = str(address[0])
    return host.removeprefix("::ffff:")


def _close(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


class RtcmRelay:
    """Connects the RTCM frames of the decoded data components (by SCId) to the
    mountpoints of the caster, named after the service label ("GEPOS").

    `set_components` runs in the GUI thread, `publish` in the thread of the signal
    processing; the mapping is replaced as a whole, never changed in place.
    """

    def __init__(self, caster: NtripCaster) -> None:
        self._caster = caster
        self._mountpoints: dict[int, str] = {}

    def set_components(self, components) -> None:
        """Data components with `scid` and `label` (dab.data_services.DataComponent)."""
        self._mountpoints = {c.scid: mountpoint_for(c.scid, c.label) for c in components}

    def publish(self, scid: int, frames: list[bytes]) -> None:
        if not self._caster.running:
            return
        mountpoint = self._mountpoints.get(scid) or mountpoint_for(scid, None)
        self._caster.publish(mountpoint, frames)
