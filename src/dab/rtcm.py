"""RTCM 3 messages in a transparent data channel: the GNSS data of GEPOS.

GEPOS, the satellite positioning service of the Bundesamt für Kartographie und Geodäsie
(BKG), is a data service in packet mode on channel 5C: DSCTy 5 (TDC, ETSI TS 101 759)
with data groups, user application type 0x5DC. TDC leaves the content to the
application. Here the data fields of the data groups, joined in the order of their
continuity index, form an RTCM 3 byte stream (RTCM 10403.3, clause 4). Each frame is

    preamble 0xD3 | 6 reserved bits | 10 bit length n | n bytes of message | CRC-24Q

and frames run across data group boundaries. Every message starts with its 12 bit
message type number.

The frames follow each other without a byte between them. They carry the broadcast
ephemerides of GPS (1019), GLONASS (1020), BeiDou (1042) and Galileo I/NAV (1046),
together a good half of the 4.3 kbit/s, and the Geo++ proprietary message 4090 with
sub-type 7: SSRZ, the PPP-RTK corrections themselves (clock and orbit corrections,
biases, ionosphere and troposphere; Geo++ SSRZ format, version 1.1.3). The ephemerides
are decoded here, the orbits they describe are computed in `dab.gnss`; SSRZ is decoded
in `dab.ssrz`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dab.data_group import DataGroup
from dab.ssrz import SsrzDecoder, SsrzResult, SsrzStatistics
from i18n import tr, tr_noop

PREAMBLE = 0xD3
_HEADER_BYTES = 3
_CRC_BYTES = 3
MAX_MESSAGE_BYTES = 1023  # 10 bit length field

# GNSS, by their RINEX letters (also used in the satellite names: G05, E12, ...).
GNSS_GPS = "G"
GNSS_GLONASS = "R"
GNSS_GALILEO = "E"
GNSS_BEIDOU = "C"

MESSAGE_GPS_EPHEMERIS = 1019
MESSAGE_GLONASS_EPHEMERIS = 1020
MESSAGE_BEIDOU_EPHEMERIS = 1042
MESSAGE_GALILEO_INAV_EPHEMERIS = 1046
MESSAGE_GEOPP_PROPRIETARY = 4090
GEOPP_SUBTYPE_SSRZ = 7  # Geo++ SSRZ format, chapter 4

# The names of the message types shown, after RTCM 10403.3, table 3.5-1.
_MESSAGE_NAMES = {
    1005: tr_noop("Reference station: position"),
    1006: tr_noop("Reference station: position and antenna height"),
    1033: tr_noop("Reference station: antenna and receiver"),
    MESSAGE_GPS_EPHEMERIS: tr_noop("GPS ephemerides"),
    MESSAGE_GLONASS_EPHEMERIS: tr_noop("GLONASS ephemerides"),
    MESSAGE_BEIDOU_EPHEMERIS: tr_noop("BeiDou ephemerides"),
    1044: tr_noop("QZSS ephemerides"),
    1045: tr_noop("Galileo ephemerides (F/NAV)"),
    MESSAGE_GALILEO_INAV_EPHEMERIS: tr_noop("Galileo ephemerides (I/NAV)"),
    1230: tr_noop("GLONASS: code-phase biases"),
    MESSAGE_GEOPP_PROPRIETARY: tr_noop("Geo++ proprietary"),
}

# SSRZ message types ZM001 to ZM025 (Geo++ SSRZ format v1.1.3, table 5.1).
_SSRZ_NAMES = {
    1: tr_noop("SSRZ: clock and radial orbit correction, high rate"),
    2: tr_noop("SSRZ: clock, orbit and signal biases, low rate"),
    3: tr_noop("SSRZ: gridded ionosphere"),
    4: tr_noop("SSRZ: gridded troposphere"),
    5: tr_noop("SSRZ: satellite dependent regional ionosphere"),
    6: tr_noop("SSRZ: global ionosphere (VTEC)"),
    7: tr_noop("SSRZ: regional troposphere"),
    8: tr_noop("SSRZ: QIX biases"),
    9: tr_noop("SSRZ: time tag"),
    10: tr_noop("SSRZ: stream identification"),
    11: tr_noop("SSRZ metadata: satellite groups"),
    12: tr_noop("SSRZ metadata"),
    13: tr_noop("SSRZ metadata: grid definition"),
    24: tr_noop("SSRZ: leap seconds"),
    25: tr_noop("SSRZ metadata: expected messages"),
}

# Scale factors of the ephemeris fields (RTCM 10403.3, clause 3.4). Angles come in
# semicircles; RTCM (like IS-GPS-200) fixes pi to this value for the conversion.
_SEMICIRCLE = 3.1415926535898
_P5, _P6, _P19, _P20 = 2.0**-5, 2.0**-6, 2.0**-19, 2.0**-20
_P11, _P29, _P30, _P31 = 2.0**-11, 2.0**-29, 2.0**-30, 2.0**-31
_P33, _P34, _P40, _P43 = 2.0**-33, 2.0**-34, 2.0**-40, 2.0**-43
_P46, _P50, _P55, _P59, _P66 = 2.0**-46, 2.0**-50, 2.0**-55, 2.0**-59, 2.0**-66

# CRC-24Q (RTCM 10403.3, clause 4.2): polynomial 0x1864CFB, register initialised to 0,
# not reflected, not inverted.
_CRC24Q_POLYNOMIAL = 0x1864CFB


def _crc24q_table() -> tuple[int, ...]:
    table = []
    for byte in range(256):
        crc = byte << 16
        for _ in range(8):
            crc <<= 1
            if crc & 0x1000000:
                crc ^= _CRC24Q_POLYNOMIAL
        table.append(crc & 0xFFFFFF)
    return tuple(table)


_CRC24Q_TABLE = _crc24q_table()


def crc24q(data: bytes) -> int:
    """CRC-24Q over `data`, as transmitted after every RTCM 3 frame."""
    crc = 0
    for byte in data:
        crc = ((crc << 8) & 0xFFFFFF) ^ _CRC24Q_TABLE[(crc >> 16) ^ byte]
    return crc


class _BitReader:
    """Reads the bit fields of an RTCM message, most significant bit first."""

    def __init__(self, data: bytes) -> None:
        self._value = int.from_bytes(data, "big")
        self._bits = len(data) * 8
        self._position = 0

    def remaining(self) -> int:
        return self._bits - self._position

    def uint(self, bits: int) -> int:
        self._position += bits
        return (self._value >> (self._bits - self._position)) & ((1 << bits) - 1)

    def int(self, bits: int) -> int:
        """Two's complement (RTCM data types intN)."""
        value = self.uint(bits)
        return value - (1 << bits) if value >> (bits - 1) else value

    def sign_magnitude(self, bits: int) -> int:
        """Sign and magnitude (RTCM data types intSN, used for the GLONASS fields)."""
        value = self.uint(bits)
        magnitude = value & ((1 << (bits - 1)) - 1)
        return -magnitude if value >> (bits - 1) else magnitude


def message_type(message: bytes) -> int | None:
    """The 12 bit message type number at the start of a message."""
    if len(message) < 2:
        return None
    return (message[0] << 4) | (message[1] >> 4)


def _prefix_code(reader: _BitReader, first_bits: int, increment: int, max_bits: int) -> int:
    """An SSRZ prefix code PC(N0, db, Nmax) (Geo++ SSRZ format, section 6.1): blocks of
    N0, N0 + db, ... up to Nmax bits, a block of all ones announces the next one, and
    the value is the sum of the blocks."""
    total = 0
    bits = first_bits
    while True:
        if reader.remaining() < bits:
            return -1
        block = reader.uint(bits)
        total += block
        if block != (1 << bits) - 1:
            return total
        bits = min(bits + increment, max_bits)


def ssrz_message_number(message: bytes) -> int | None:
    """The SSRZ message type (1 for ZM001 ...) of a Geo++ message 4090 with sub-type 7,
    None for any other message.

    After the RTCM message type come 4 bits of Geo++ sub-type, then the SSRZ message
    type indicator ZDF002 as prefix code PC(2,1,5): values 0 to 24 stand for ZM001 to
    ZM025 (Geo++ SSRZ format, table 7.1).
    """
    if message_type(message) != MESSAGE_GEOPP_PROPRIETARY or len(message) < 3:
        return None
    reader = _BitReader(message)
    reader.uint(12)
    if reader.uint(4) != GEOPP_SUBTYPE_SSRZ:
        return None
    indicator = _prefix_code(reader, 2, 1, 5)
    return None if indicator < 0 else indicator + 1


# What a message is, for the statistics: (message type, SSRZ message number or None).
MessageKind = tuple[int, int | None]


def message_kind(message: bytes) -> MessageKind | None:
    number = message_type(message)
    if number is None:
        return None
    return number, ssrz_message_number(message)


def message_name(kind: MessageKind) -> str:
    """Name of a message kind, translated for the display."""
    number, ssrz = kind
    if ssrz is not None:
        name = _SSRZ_NAMES.get(ssrz)
        return tr("SSRZ: not described in version 1.1.3") if name is None else tr(name)
    if number == MESSAGE_GEOPP_PROPRIETARY:
        return tr("Geo++ proprietary (no SSRZ)")
    name = _MESSAGE_NAMES.get(number)
    # Translators: a message type of the RTCM stream not known to the radio.
    return tr("unknown") if name is None else tr(name)


def message_label(kind: MessageKind) -> str:
    """Short label of a message kind: "1019" or "4090 · ZM003"."""
    number, ssrz = kind
    return f"{number}" if ssrz is None else f"{number} · ZM{ssrz:03d}"


@dataclass(frozen=True)
class KeplerEphemeris:
    """The broadcast ephemeris of a GPS, Galileo or BeiDou satellite: Keplerian orbit
    elements with their perturbation terms (IS-GPS-200, table 20-III), converted to SI
    units, angles in radians.

    The three systems use the same orbit model; they differ in the resolution of some
    fields, in their time scale (GPS and Galileo agree, BeiDou time runs 14 s behind)
    and in their constants (see `dab.gnss`).
    """

    system: str  # GNSS_GPS, GNSS_GALILEO or GNSS_BEIDOU
    prn: int
    week: int  # as transmitted: GPS modulo 1024, Galileo modulo 4096, BeiDou modulo 8192
    iod: int  # issue of data: IODE (GPS), IODnav (Galileo), AODE (BeiDou)
    toe: float  # reference time, seconds of the week in the system's own time
    sqrt_a: float  # square root of the semi-major axis in m^0.5
    eccentricity: float
    inclination: float  # i0
    inclination_rate: float  # IDOT, rad/s
    right_ascension: float  # Omega0, longitude of the ascending node at the start of the week
    right_ascension_rate: float  # Omega dot, rad/s
    perigee: float  # argument of perigee omega
    mean_anomaly: float  # M0
    mean_motion_difference: float  # delta n, rad/s
    cuc: float  # harmonic corrections: argument of latitude (rad) ...
    cus: float
    crc: float  # ... orbit radius (m) ...
    crs: float
    cic: float  # ... and inclination (rad)
    cis: float
    healthy: bool
    # Satellite clock: t_sv - t = af0 + af1 (t - toc) + af2 (t - toc)^2 (seconds), toc in
    # seconds of the week in the system's own time.
    toc: float = 0.0
    af0: float = 0.0
    af1: float = 0.0
    af2: float = 0.0

    @property
    def name(self) -> str:
        return f"{self.system}{self.prn:02d}"

    @property
    def ssr_iod(self) -> int:
        """The issue of data by which SSR corrections name this ephemeris (Geo++ SSRZ
        1.1.3, table 11.2): the IODE of GPS, the 8 low bits of the Galileo IODnav, and
        for BeiDou mod(toc / 720, 240)."""
        if self.system == GNSS_GALILEO:
            return self.iod & 0xFF
        if self.system == GNSS_BEIDOU:
            return int(self.toc // 720) % 240
        return self.iod


@dataclass(frozen=True)
class GlonassEphemeris:
    """The broadcast ephemeris of a GLONASS satellite: position, velocity and the
    luni-solar acceleration at the reference time tb, in the Earth-fixed frame PZ-90
    (GLONASS ICD, edition 5.1, section 4.4), converted to metres and seconds. Positions
    at other times follow by integrating the equations of motion (`dab.gnss`)."""

    slot: int  # orbital slot 1..24
    frequency_channel: int  # -7..+13
    tb: int  # reference time, seconds of the day in Moscow time (UTC + 3 h)
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    acceleration: tuple[float, float, float]
    healthy: bool
    # Satellite clock: t_GLONASS = t_sv + tau_n - gamma_n (t_sv - tb) (GLONASS ICD 5.1,
    # clause 3.3.3), tau_n in seconds.
    tau_n: float = 0.0
    gamma_n: float = 0.0

    system = GNSS_GLONASS

    @property
    def prn(self) -> int:
        return self.slot

    @property
    def iod(self) -> int:
        """GLONASS has no issue of data; a new ephemeris comes with a new tb (every
        30 minutes). tb counts in quarter hours."""
        return self.tb // 900

    @property
    def ssr_iod(self) -> int:
        """SSR corrections name a GLONASS ephemeris by the 7 bits of tb."""
        return self.iod

    @property
    def name(self) -> str:
        return f"{self.system}{self.slot:02d}"


Ephemeris = KeplerEphemeris | GlonassEphemeris

_EPHEMERIS_BITS = {
    MESSAGE_GPS_EPHEMERIS: 488,
    MESSAGE_GLONASS_EPHEMERIS: 360,
    MESSAGE_BEIDOU_EPHEMERIS: 511,
    MESSAGE_GALILEO_INAV_EPHEMERIS: 504,
}


def decode_ephemeris(message: bytes) -> Ephemeris | None:
    """Decodes a message 1019, 1020, 1042 or 1046 (RTCM 10403.3, clause 3.5.10).

    Returns:
        The ephemeris, or None for another message type or a message that is too short.
    """
    number = message_type(message)
    bits = _EPHEMERIS_BITS.get(number)
    if bits is None or len(message) * 8 < bits:
        return None
    reader = _BitReader(message)
    reader.uint(12)
    if number == MESSAGE_GPS_EPHEMERIS:
        return _decode_gps(reader)
    if number == MESSAGE_GALILEO_INAV_EPHEMERIS:
        return _decode_galileo_inav(reader)
    if number == MESSAGE_BEIDOU_EPHEMERIS:
        return _decode_beidou(reader)
    return _decode_glonass(reader)


def _read_orbit(
    r: _BitReader,
    harmonic_bits: int,
    harmonic_scale: float,
    radius_scale: float,
    toe_bits: int,
    toe_scale: float,
) -> dict[str, float]:
    """The orbit block from Crs to Omega dot, in the same order in 1019, 1042 and 1046.

    BeiDou transmits the harmonic corrections with 18 instead of 16 bits and a finer
    resolution; the time of ephemeris has a different resolution in each system.
    """
    orbit = {"crs": r.int(harmonic_bits) * radius_scale}
    orbit["mean_motion_difference"] = r.int(16) * _P43 * _SEMICIRCLE
    orbit["mean_anomaly"] = r.int(32) * _P31 * _SEMICIRCLE
    orbit["cuc"] = r.int(harmonic_bits) * harmonic_scale
    orbit["eccentricity"] = r.uint(32) * _P33
    orbit["cus"] = r.int(harmonic_bits) * harmonic_scale
    orbit["sqrt_a"] = r.uint(32) * _P19
    orbit["toe"] = r.uint(toe_bits) * toe_scale
    orbit["cic"] = r.int(harmonic_bits) * harmonic_scale
    orbit["right_ascension"] = r.int(32) * _P31 * _SEMICIRCLE
    orbit["cis"] = r.int(harmonic_bits) * harmonic_scale
    orbit["inclination"] = r.int(32) * _P31 * _SEMICIRCLE
    orbit["crc"] = r.int(harmonic_bits) * radius_scale
    orbit["perigee"] = r.int(32) * _P31 * _SEMICIRCLE
    orbit["right_ascension_rate"] = r.int(24) * _P43 * _SEMICIRCLE
    return orbit


def _decode_gps(r: _BitReader) -> KeplerEphemeris:
    """Message 1019 (DF009 to DF137)."""
    prn = r.uint(6)
    week = r.uint(10)
    r.uint(4)  # URA
    r.uint(2)  # code on L2
    idot = r.int(14) * _P43 * _SEMICIRCLE
    iode = r.uint(8)
    clock = {"toc": r.uint(16) * 16.0, "af2": r.int(8) * _P55, "af1": r.int(16) * _P43}
    clock["af0"] = r.int(22) * _P31
    r.uint(10)  # IODC
    orbit = _read_orbit(r, 16, _P29, _P5, 16, 16.0)
    r.int(8)  # TGD
    health = r.uint(6)
    return KeplerEphemeris(
        GNSS_GPS, prn, week, iode, inclination_rate=idot, healthy=health == 0, **orbit, **clock
    )


def _decode_galileo_inav(r: _BitReader) -> KeplerEphemeris:
    """Message 1046 (DF252 to DF288)."""
    prn = r.uint(6)
    week = r.uint(12)
    iodnav = r.uint(10)
    r.uint(8)  # SISA
    idot = r.int(14) * _P43 * _SEMICIRCLE
    clock = {"toc": r.uint(14) * 60.0, "af2": r.int(6) * _P59, "af1": r.int(21) * _P46}
    clock["af0"] = r.int(31) * _P34
    orbit = _read_orbit(r, 16, _P29, _P5, 14, 60.0)
    r.int(10)  # BGD E5a/E1
    r.int(10)  # BGD E5b/E1
    # Signal health and data validity of E5b and E1-B: 0 means OK for all four.
    status = (r.uint(2), r.uint(1), r.uint(2), r.uint(1))
    return KeplerEphemeris(
        GNSS_GALILEO,
        prn,
        week,
        iodnav,
        inclination_rate=idot,
        healthy=not any(status),
        **orbit,
        **clock,
    )


def _decode_beidou(r: _BitReader) -> KeplerEphemeris:
    """Message 1042 (DF488 to DF515)."""
    prn = r.uint(6)
    week = r.uint(13)
    r.uint(4)  # URAI
    idot = r.int(14) * _P43 * _SEMICIRCLE
    aode = r.uint(5)
    clock = {"toc": r.uint(17) * 8.0, "af2": r.int(11) * _P66, "af1": r.int(22) * _P50}
    clock["af0"] = r.int(24) * _P33
    r.uint(5)  # AODC
    orbit = _read_orbit(r, 18, _P31, _P6, 17, 8.0)
    r.int(10)  # TGD1
    r.int(10)  # TGD2
    health = r.uint(1)
    return KeplerEphemeris(
        GNSS_BEIDOU, prn, week, aode, inclination_rate=idot, healthy=health == 0, **orbit, **clock
    )


def _decode_glonass(r: _BitReader) -> GlonassEphemeris:
    """Message 1020 (DF038 to DF136). Position, velocity and acceleration come in km,
    km/s and km/s^2 as sign and magnitude."""
    slot = r.uint(6)
    frequency_channel = r.uint(5) - 7
    r.uint(1)  # almanac health
    r.uint(1)  # almanac health availability
    r.uint(2)  # P1
    r.uint(12)  # tk
    unhealthy = r.uint(1)  # most significant bit of Bn
    r.uint(1)  # P2
    tb = r.uint(7) * 900
    state = []
    for _axis in range(3):
        velocity = r.sign_magnitude(24) * _P20 * 1000.0
        position = r.sign_magnitude(27) * _P11 * 1000.0
        acceleration = r.sign_magnitude(5) * _P30 * 1000.0
        state.append((position, velocity, acceleration))
    positions, velocities, accelerations = zip(*state, strict=True)
    r.uint(1)  # P3
    gamma_n = r.sign_magnitude(11) * _P40
    r.uint(2)  # P
    r.uint(1)  # ln
    tau_n = r.sign_magnitude(22) * _P30
    return GlonassEphemeris(
        slot,
        frequency_channel,
        tb,
        positions,
        velocities,
        accelerations,
        not unhealthy,
        tau_n,
        gamma_n,
    )


class RtcmFramer:
    """Finds the RTCM 3 frames in a byte stream and checks their CRC-24Q.

    Bytes that belong to no correct frame are skipped. Frames follow each other without
    a gap in the stream, so once a frame was correct the next one has to start right
    after it; only a frame failing there counts as a CRC error. While searching, a
    preamble byte inside other data is no error.

    With `keep_frames`, the complete frames (header, message and CRC, byte for byte as
    received) are kept for `take_frames`, to pass the stream on unchanged.
    """

    def __init__(self, keep_frames: bool = False) -> None:
        self._buffer = bytearray()
        self._in_sync = False
        self._keep_frames = keep_frames
        self._frames: list[bytes] = []
        self.frames = 0
        self.frame_bytes = 0
        self.crc_errors = 0
        self.discarded_bytes = 0

    def reset(self) -> None:
        """After a gap in the stream: the frame under way is lost."""
        self.discarded_bytes += len(self._buffer)
        self._buffer.clear()
        self._in_sync = False

    def push(self, data: bytes) -> list[bytes]:
        """Appends bytes of the stream.

        Returns:
            The messages (without frame header and CRC) of all frames completed with
            them.
        """
        buffer = self._buffer
        buffer.extend(data)
        messages = []
        while True:
            start = buffer.find(PREAMBLE)
            if start < 0:
                self._skip(len(buffer))
                break
            if start:
                self._skip(start)
            if len(buffer) < _HEADER_BYTES:
                break
            if buffer[1] & 0xFC:  # the 6 reserved bits are 0
                self._skip(1)
                continue
            length = ((buffer[1] & 0x03) << 8) | buffer[2]
            end = _HEADER_BYTES + length + _CRC_BYTES
            if len(buffer) < end:
                break
            if crc24q(buffer[: end - _CRC_BYTES]) != int.from_bytes(
                buffer[end - _CRC_BYTES : end], "big"
            ):
                if self._in_sync:
                    self.crc_errors += 1
                self._skip(1)
                continue
            messages.append(bytes(buffer[_HEADER_BYTES : end - _CRC_BYTES]))
            if self._keep_frames:
                self._frames.append(bytes(buffer[:end]))
            self.frames += 1
            self.frame_bytes += end
            self._in_sync = True
            del buffer[:end]
        return messages

    def take_frames(self) -> list[bytes]:
        """The complete frames found since the last call (only with `keep_frames`)."""
        frames, self._frames = self._frames, []
        return frames

    def _skip(self, count: int) -> None:
        del self._buffer[:count]
        self.discarded_bytes += count
        self._in_sync = False


@dataclass(frozen=True)
class MessageStatistics:
    count: int
    frame_bytes: int  # including frame header and CRC


@dataclass(frozen=True)
class RtcmStatistics:
    """What the RTCM stream of the decoded data component has yielded so far."""

    seconds: float  # signal duration covered
    data_groups: int
    lost_data_groups: int  # jumps of the continuity index
    frames: int  # frames with a correct CRC
    crc_errors: int
    discarded_bytes: int  # bytes outside correct frames
    stream_bytes: int  # all bytes of the stream (data group data fields)
    messages: dict[MessageKind, MessageStatistics] = field(default_factory=dict)
    ephemerides: int = 0  # satellites with a decoded ephemeris
    ssrz: SsrzStatistics = field(default_factory=SsrzStatistics)


class RtcmStreamDecoder:
    """Joins the data groups of a TDC component into the RTCM stream, counts its
    message types, decodes the ephemerides in it and passes SSRZ on to `dab.ssrz`.

    The continuity index (4 bits) increments with every new data group; a data group
    that arrives again with the same index is a repetition (ETSI TS 101 759, clause
    4.1.2) and is skipped. If the index jumps, data groups are missing and the frame
    under way is lost.
    """

    def __init__(self) -> None:
        self._framer = RtcmFramer(keep_frames=True)
        self._continuity: int | None = None
        self._data_groups = 0
        self._lost_data_groups = 0
        self._stream_bytes = 0
        self._messages: dict[MessageKind, list[int]] = {}
        self._ephemerides: dict[str, Ephemeris] = {}
        self._ssrz = SsrzDecoder()
        self._ssrz_results: list[SsrzResult] = []

    def interrupt(self) -> None:
        """The data groups have a gap the continuity index cannot reveal (the subchannel
        chain started afresh)."""
        self._framer.reset()
        self._continuity = None

    def push(self, group: DataGroup) -> list[Ephemeris]:
        """Takes the next data group of the component.

        Returns:
            The ephemerides that are new with it: first seen for their satellite, or
            different from the one decoded before. The same ephemeris is repeated every
            few seconds and only reported once.
        """
        if group.continuity_index == self._continuity:
            return []
        if self._continuity is not None and group.continuity_index != (self._continuity + 1) % 16:
            self._lost_data_groups += (group.continuity_index - self._continuity - 1) % 16
            self._framer.reset()
        self._continuity = group.continuity_index
        self._data_groups += 1
        self._stream_bytes += len(group.data)

        new = []
        for message in self._framer.push(group.data):
            kind = message_kind(message)
            if kind is None:  # a frame too short for a message type
                continue
            counts = self._messages.setdefault(kind, [0, 0])
            counts[0] += 1
            counts[1] += len(message) + _HEADER_BYTES + _CRC_BYTES
            if kind[1] is not None:
                result = self._ssrz.push(message)
                if result is not None:
                    self._ssrz_results.append(result)
                continue
            ephemeris = decode_ephemeris(message)
            if ephemeris is not None and self._ephemerides.get(ephemeris.name) != ephemeris:
                self._ephemerides[ephemeris.name] = ephemeris
                new.append(ephemeris)
        return new

    def take_ssrz(self) -> list[SsrzResult]:
        """What the SSRZ messages since the last call said (see `dab.ssrz`)."""
        results, self._ssrz_results = self._ssrz_results, []
        return results

    def take_frames(self) -> list[bytes]:
        """The frames completed since the last call, byte for byte as received (with
        header and CRC), for passing the stream on (see `net.ntrip`)."""
        return self._framer.take_frames()

    def statistics(self, seconds: float) -> RtcmStatistics:
        framer = self._framer
        return RtcmStatistics(
            seconds=seconds,
            data_groups=self._data_groups,
            lost_data_groups=self._lost_data_groups,
            frames=framer.frames,
            crc_errors=framer.crc_errors,
            discarded_bytes=framer.discarded_bytes,
            stream_bytes=self._stream_bytes,
            messages={
                kind: MessageStatistics(count, size)
                for kind, (count, size) in sorted(self._messages.items(), key=_sort_key)
            },
            ephemerides=len(self._ephemerides),
            ssrz=self._ssrz.statistics(),
        )


def _sort_key(item: tuple[MessageKind, list[int]]) -> tuple[int, int]:
    (number, ssrz), _counts = item
    return number, -1 if ssrz is None else ssrz
