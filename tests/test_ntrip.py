"""The NTRIP caster that passes the RTCM stream of GEPOS on: NTRIP 1.0 and 2.0, the
sourcetable, plain TCP, positions by GGA, and frames passed on byte for byte."""

from __future__ import annotations

import socket
import time
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from dab.data_group import parse_data_group
from dab.rtcm import RtcmFramer, RtcmStreamDecoder
from net.ntrip import (
    PROTOCOL_NTRIP1,
    PROTOCOL_NTRIP2,
    PROTOCOL_TCP,
    NtripCaster,
    NtripCasterError,
    RtcmRelay,
    mountpoint_for,
    parse_gga,
)
from tests.rtcm_vectors import DATA_GROUPS, GLONASS_R11_FRAME, GPS_G12_FRAME, SSRZ_ZM001_FRAME

# Only the loopback interface: listening on all would ask for a firewall rule.
LOOPBACK = "127.0.0.1"
# A GGA of a u-blox receiver near Hannover: 52.37 N, 9.74 E, RTK fixed.
GGA = "$GNGGA,183000.00,5222.2720,N,00944.1060,E,4,18,0.6,56.3,M,46.1,M,1.0,0000*"


def _with_checksum(sentence: str) -> str:
    value = 0
    for character in sentence[1:-1]:
        value ^= ord(character)
    return f"{sentence}{value:02X}"


def _wait_for(condition: Callable[[], bool], seconds: float = 3.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.01)


def _receive(sock: socket.socket, count: int) -> bytes:
    data = bytearray()
    sock.settimeout(3.0)
    while len(data) < count:
        chunk = sock.recv(count - len(data))
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


def _receive_until(sock: socket.socket, end: bytes) -> bytes:
    data = bytearray()
    sock.settimeout(3.0)
    while end not in data:
        chunk = sock.recv(1)
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


@pytest.fixture
def caster():
    caster = NtripCaster(raw_wait_seconds=0.2)
    caster.start(0, LOOPBACK)
    caster.add_mountpoint("GEPOS")
    yield caster
    caster.stop()


def _connect(caster: NtripCaster) -> socket.socket:
    return socket.create_connection((LOOPBACK, caster.port), timeout=3.0)


def _clients(caster: NtripCaster) -> int:
    return len(caster.status().clients)


# --- GGA ------------------------------------------------------------------------------


def test_gga_gives_position_and_ellipsoidal_height() -> None:
    position = parse_gga(_with_checksum(GGA), received_at=5.0)
    assert position is not None
    assert position.latitude == pytest.approx(52 + 22.272 / 60)
    assert position.longitude == pytest.approx(9 + 44.106 / 60)
    assert position.height == pytest.approx(56.3 + 46.1)  # altitude plus geoid separation
    assert position.quality == 4
    assert position.satellites == 18
    assert position.received_at == 5.0


def test_gga_south_west_and_without_checksum() -> None:
    sentence = "$GPGGA,000000,3354.000,S,07015.000,W,1,05,1.2,500.0,M,,M,,"
    position = parse_gga(sentence)
    assert position is not None
    assert position.latitude == pytest.approx(-33.9)
    assert position.longitude == pytest.approx(-70.25)
    assert position.height == pytest.approx(500.0)


def test_gga_without_fix_or_with_wrong_checksum_is_ignored() -> None:
    assert parse_gga("$GPGGA,000000,,,,,0,00,,,M,,M,,*66") is None
    assert parse_gga(GGA + "00") is None
    assert parse_gga("$GPRMC,000000,A,5222.2720,N,00944.1060,E,0,0,010126,,,A") is None


def test_mountpoint_from_the_service_label() -> None:
    assert mountpoint_for(0x020, "GEPOS") == "GEPOS"
    assert mountpoint_for(0x020, " EPG Deutschland ") == "EPG_Deutschland"
    assert mountpoint_for(0x020, None) == "SCID020"


# --- Caster ---------------------------------------------------------------------------


def test_ntrip1_client_gets_the_frames_unchanged_and_reports_gga(caster: NtripCaster) -> None:
    with _connect(caster) as sock:
        sock.sendall(b"GET /GEPOS HTTP/1.0\r\nUser-Agent: NTRIP RTKLIB/2.4.3\r\n\r\n")
        assert _receive(sock, 14) == b"ICY 200 OK\r\n\r\n"
        _wait_for(lambda: _clients(caster) == 1)
        frames = [GPS_G12_FRAME, SSRZ_ZM001_FRAME]
        caster.publish("GEPOS", frames)
        caster.publish("GEPOS", [GLONASS_R11_FRAME])
        expected = b"".join(frames) + GLONASS_R11_FRAME
        assert _receive(sock, len(expected)) == expected

        sock.sendall((_with_checksum(GGA) + "\r\n").encode("ascii"))
        _wait_for(lambda: caster.status().latest_position is not None)
        client = caster.status().clients[0]
        assert client.protocol == PROTOCOL_NTRIP1
        assert client.mountpoint == "GEPOS"
        assert client.bytes_sent == len(expected)
        assert client.position.quality == 4
    _wait_for(lambda: _clients(caster) == 0)


def test_ntrip2_client_gets_chunks_and_gga_from_the_header(caster: NtripCaster) -> None:
    with _connect(caster) as sock:
        sock.sendall(
            (
                "GET /GEPOS HTTP/1.1\r\nHost: localhost\r\nNtrip-Version: Ntrip/2.0\r\n"
                f"User-Agent: NTRIP test\r\nNtrip-GGA: {_with_checksum(GGA)}\r\n\r\n"
            ).encode("ascii")
        )
        head = _receive_until(sock, b"\r\n\r\n").decode("ascii")
        assert head.startswith("HTTP/1.1 200 OK\r\n")
        assert "Ntrip-Version: Ntrip/2.0" in head
        assert "Transfer-Encoding: chunked" in head
        _wait_for(lambda: _clients(caster) == 1)
        caster.publish("GEPOS", [GPS_G12_FRAME])
        size = int(_receive_until(sock, b"\r\n")[:-2], 16)
        assert size == len(GPS_G12_FRAME)
        assert _receive(sock, size + 2) == GPS_G12_FRAME + b"\r\n"
        client = caster.status().clients[0]
        assert client.protocol == PROTOCOL_NTRIP2
        assert client.position is not None


def test_sourcetable_lists_the_mountpoints_with_their_messages(caster: NtripCaster) -> None:
    caster.publish("GEPOS", [GPS_G12_FRAME, SSRZ_ZM001_FRAME])
    with _connect(caster) as sock:
        sock.sendall(b"GET / HTTP/1.0\r\n\r\n")
        response = _receive_until(sock, b"ENDSOURCETABLE\r\n").decode("ascii")
    assert response.startswith("SOURCETABLE 200 OK\r\n")
    record = next(line for line in response.split("\r\n") if line.startswith("STR;"))
    fields = record.split(";")
    assert fields[1] == "GEPOS"
    assert fields[3] == "RTCM 3.3"
    assert fields[4] == "1019,4090"


def test_unknown_mountpoint_gets_the_sourcetable(caster: NtripCaster) -> None:
    with _connect(caster) as sock:
        sock.sendall(b"GET /NOPE HTTP/1.1\r\nNtrip-Version: Ntrip/2.0\r\n\r\n")
        response = _receive_until(sock, b"ENDSOURCETABLE\r\n").decode("ascii")
    assert response.startswith("HTTP/1.1 200 OK\r\n")
    assert "Content-Type: gnss/sourcetable" in response
    assert "STR;GEPOS;" in response


def test_plain_tcp_client_gets_the_bare_stream(caster: NtripCaster) -> None:
    with _connect(caster) as sock:
        _wait_for(lambda: _clients(caster) == 1)  # after the silent wait
        caster.publish("GEPOS", [GPS_G12_FRAME])
        assert _receive(sock, len(GPS_G12_FRAME)) == GPS_G12_FRAME
        assert caster.status().clients[0].protocol == PROTOCOL_TCP


def test_taken_port_is_reported(caster: NtripCaster) -> None:
    other = NtripCaster()
    with pytest.raises(NtripCasterError, match=f"port {caster.port}"):
        other.start(caster.port, LOOPBACK)
    assert not other.running


def test_stop_closes_the_connections(caster: NtripCaster) -> None:
    with _connect(caster) as sock:
        sock.sendall(b"GET /GEPOS HTTP/1.0\r\n\r\n")
        _receive(sock, 14)
        _wait_for(lambda: _clients(caster) == 1)
        caster.stop()
        assert not caster.running
        sock.settimeout(3.0)
        try:
            rest = sock.recv(100)
        except OSError:
            rest = b""
        assert rest == b""


def test_relay_names_the_mountpoint_after_the_service() -> None:
    published = []
    fake = SimpleNamespace(running=True, publish=lambda m, f: published.append((m, f)))
    relay = RtcmRelay(fake)
    relay.set_components([SimpleNamespace(scid=0x020, label="GEPOS")])
    relay.publish(0x020, [GPS_G12_FRAME])
    relay.publish(0x021, [GPS_G12_FRAME])
    assert published == [("GEPOS", [GPS_G12_FRAME]), ("SCID021", [GPS_G12_FRAME])]
    fake.running = False
    relay.publish(0x020, [GPS_G12_FRAME])
    assert len(published) == 2


# --- Frames as received ---------------------------------------------------------------


def test_stream_decoder_passes_the_frames_on_byte_for_byte() -> None:
    groups = [parse_data_group(raw) for raw in DATA_GROUPS]
    decoder = RtcmStreamDecoder()
    frames = []
    for group in groups:
        decoder.push(group)
        frames += decoder.take_frames()
    assert frames
    stream = b"".join(group.data for group in groups)
    # Every frame is a piece of the stream, and together they are the stream up to the
    # frame still under way at the end.
    joined = b"".join(frames)
    assert stream.startswith(joined)
    assert decoder.statistics(1.0).frames == len(frames)
    assert decoder.take_frames() == []


def test_framer_keeps_no_frames_unless_asked() -> None:
    framer = RtcmFramer()
    framer.push(GPS_G12_FRAME)
    assert framer.take_frames() == []
    keeping = RtcmFramer(keep_frames=True)
    keeping.push(b"\x00\x01" + GPS_G12_FRAME + GPS_G12_FRAME[:10])
    assert keeping.take_frames() == [GPS_G12_FRAME]
