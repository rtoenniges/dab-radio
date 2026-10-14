"""SSRZ, the compressed state space representation of Geo++, in the RTCM stream of GEPOS.

GEPOS sends its PPP-RTK corrections as Geo++ proprietary RTCM messages 4090 with
sub-type 7 (see `dab.rtcm`). Their content follows the Geo++ SSRZ format, version 1.1.3
(Geo++ "State Space Representation Format (SSRZ)", 2025-04-15; chapter and table
numbers below refer to it). SSRZ describes every error source of a GNSS signal on its
own: satellite clocks and orbits, signal biases, the ionosphere and the troposphere.

SSRZ is built for a small, even data rate. Everything that rarely changes is moved
into metadata sent every 30 s (ZM011, ZM012, ZM013): the satellite groups, resolutions,
model parameters and the grids of ground points. The correction messages then carry
only numbers, compressed with a Rice code (clause 3.7): a sign bit, the quotient in
unary and the remainder in p bits, where p and the resolution adapt per block. Short
fields use prefix codes PC(N0, dN, Nmax) (clause 6.1). A negative zero marks a missing
value.

Decoded here are all messages GEPOS sends, and what it takes to read them:

- ZM002, low rate (30 s): per satellite the IOD of the broadcast ephemeris the
  corrections refer to, the clock correction C0, the orbit correction (radial,
  along-track, cross-track), code biases in metres and phase biases in cycles (modulo
  10) with their continuity indicator. Its header also says which satellites the other
  satellite dependent messages list.
- ZM001, high rate (5 s): a clock correction on top of that of ZM002.
- ZM003, gridded ionosphere: per satellite and grid point the remaining slant TEC,
  mapped to the vertical, after the functional models below.
- ZM004, gridded troposphere: per grid point a scale factor to the model troposphere.
- ZM005, regional ionosphere: per satellite a small polynomial over the service area.
- ZM006, global ionosphere: vertical TEC as spherical harmonics.
- ZM007, regional troposphere: a polynomial scale factor around a ground point origin.
- ZM013, the grids: chains of triangles between the reference stations (chapter 12).
- ZM011 (satellite groups) and ZM012 (metadata, blocks ZMB001 to ZMB007): the
  satellite dependent messages only list values, for exactly the satellites of the
  groups that the last ZM002 has, in SSRZ order.
- ZM009 (time), ZM010 (stream identification), ZM024 (leap seconds).

What the corrections mean at a place is computed in `dab.corrections`.

Where the format description and the stream disagree, the stream wins; checked against
the reference decoder of Geo++ (Geopp-SSRZ-Python-Demo, AGPL, used only for comparison)
on a 15 minute recording of 5C:

- ZDF092 (order of the grid coordinate resolution) is PC(2,0), not PC(2,1,0).
- "sint" fields (ZDF168 to ZDF170, ZDF252, ZDF254) are sign and magnitude, like the
  ranges in table 6.1 say (-0 exists).
- Table 6.2 lists the prefix codes of 23 and 24 one too high, and table 9.28 gives
  ZM024 as 11 111 1110; the stream follows the rule of clause 6.1: 11 111 1101.
- The code biases of a Rice block (ZDB004) come per GNSS signal by signal, each for all
  satellites; clause 7.2.4 describes the satellites as the outer order.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

import numpy as np

# The RTCM message carrying SSRZ: Geo++ proprietary, sub-type 7 (chapter 4). The same
# numbers as in `dab.rtcm`, which uses this module.
_GEOPP_MESSAGE = 4090
_SSRZ_SUBTYPE = 7

# The SSRZ GNSS in the order of their IDs (table 10.1), as RINEX letters. The first
# transmitted bit of a GNSS bit mask (ZDF012) stands for GPS.
SSRZ_GNSS = ("G", "R", "E", "J", "S", "C", "I")
_GNSS_MASK_BITS = 16

# Signals per GNSS by their bit in the signal bit mask ZDF019 (table 10.3), as RINEX 3
# observation codes (band and attribute); "" is reserved. The first transmitted bit is
# bit 0.
# fmt: off
_SIGNALS = {
    "G": ("1C", "1P", "1W", "", "", "2C", "2D", "2S", "2L", "2X", "2P", "2W", "", "",
          "5I", "5Q", "5X", "1S", "1L", "1X"),
    "R": ("1C", "1P", "2P", "2C", "4A", "4B", "4X", "6A", "6B", "6X", "3I", "3Q", "3X"),
    "E": ("1A", "1B", "1C", "1X", "1Z", "5I", "5Q", "5X", "7I", "7Q", "7X", "8I", "8Q",
          "8X", "6A", "6B", "6C", "6X", "6Z"),
    "J": ("1C", "1S", "1L", "2S", "2L", "2X", "5I", "5Q", "5X", "6S", "6L", "6X", "1X"),
    "S": ("1C", "5I", "5Q", "5X"),
    "C": ("2I", "2Q", "2X", "6I", "6Q", "6X", "7I", "7Q", "7X", "1D", "1P", "1X", "1A",
          "1N", "5D", "5P", "5X", "7D", "7P", "7Z", "8D", "8P", "8X"),
}
# fmt: on
_SIGNAL_MASK_BITS = 32

# SSRZ message numbers (ZM001 = 1, ...).
ZM_HIGH_RATE = 1
ZM_LOW_RATE = 2
ZM_GRID_IONOSPHERE = 3
ZM_GRID_TROPOSPHERE = 4
ZM_REGIONAL_IONOSPHERE = 5
ZM_GLOBAL_VTEC = 6
ZM_REGIONAL_TROPOSPHERE = 7
ZM_TIME_TAG = 9
ZM_STREAM_ID = 10
ZM_SATELLITE_GROUPS = 11
ZM_METADATA = 12
ZM_GRID_DEFINITION = 13
ZM_LEAP_SECOND = 24

# Metadata block types (ZDF003) as they appear in ZM012 (table 9.23).
_BLOCK_HIGH_RATE = 1
_BLOCK_LOW_RATE = 2
_BLOCK_GRID_IONOSPHERE = 3
_BLOCK_GRID_TROPOSPHERE = 4
_BLOCK_REGIONAL_IONOSPHERE = 5
_BLOCK_GLOBAL_VTEC = 6
_BLOCK_REGIONAL_TROPOSPHERE = 7
_BLOCK_SATELLITE_GROUPS = 11
_BLOCK_GRID_DEFINITION = 13

# The 15 minute time tag (ZDF050) of the correction messages.
TIME_TAG_PERIOD = 900
# Update interval assumed for the low rate satellites until the metadata tells.
_DEFAULT_LOW_RATE_INTERVAL = 30

# The tropospheric components in the order of their bits in ZDF107.
TROPOSPHERE_DRY = "dry"
TROPOSPHERE_WET = "wet"
TROPOSPHERE_TOTAL = "total"
_TROPOSPHERE_COMPONENTS = (TROPOSPHERE_DRY, TROPOSPHERE_WET, TROPOSPHERE_TOTAL)
_MAPPING_IMPROVEMENT_BIT = 3

# Satellite group definition modes (ZDF014).
_GROUP_MODE_LIST = 0
_GROUP_MODE_GNSS = 6

# Seconds between the GPS epoch (1980-01-06) and the POSIX epoch.
GPS_EPOCH_POSIX = 315964800


class SsrzError(ValueError):
    """A message is cut short or does not match its metadata."""


class _Bits:
    """Reads the bit fields of an SSRZ message, most significant bit first."""

    def __init__(self, data: bytes, position: int = 0) -> None:
        self._bits = format(int.from_bytes(data, "big"), f"0{len(data) * 8}b") if data else ""
        self.position = position
        self.end = len(self._bits)

    def remaining(self) -> int:
        return self.end - self.position

    def uint(self, bits: int) -> int:
        if bits == 0:
            return 0
        end = self.position + bits
        if end > self.end:
            raise SsrzError("message too short")
        value = int(self._bits[self.position : end], 2)
        self.position = end
        return value

    def flags(self, count: int) -> tuple[bool, ...]:
        return tuple(self.uint(1) == 1 for _ in range(count))

    def int32(self) -> int:
        value = self.uint(32)
        return value - (1 << 32) if value >> 31 else value

    def float32(self) -> float:
        return struct.unpack(">f", self.uint(32).to_bytes(4, "big"))[0]

    def sign_magnitude(self, bits: int) -> int:
        value = self.uint(bits)
        magnitude = value & ((1 << (bits - 1)) - 1)
        return -magnitude if value >> (bits - 1) else magnitude

    def prefix(self, first: int, step: int = 0, limit: int = 0) -> int:
        """Prefix code PC(first, step, limit) (clause 6.1): blocks of `first`, then
        `first + step` ... bits (at most `limit`, 0 for no limit); a block of all ones
        announces another one, the value is the sum of all blocks."""
        total = 0
        bits = first
        while True:
            block = self.uint(bits)
            total += block
            if block != (1 << bits) - 1:
                return total
            bits += step
            if limit:
                bits = min(bits, limit)

    def unary(self) -> int:
        """Ones up to the next zero (the Rice quotient, PC(1,0))."""
        stop = self._bits.find("0", self.position, self.end)
        if stop < 0:
            raise SsrzError("message too short")
        count = stop - self.position
        self.position = stop + 1
        return count

    def unary_limited(self, limit: int) -> int:
        """Ones up to the next zero, but at most `limit` ones; the zero after them is
        only sent below the limit (the continuity indicator ZDF312)."""
        count = 0
        while count < limit and self.uint(1):
            count += 1
        return count

    def rice(self, p: int) -> int | None:
        """A Rice-encoded integer value (ZDB001): sign, quotient, remainder. None for the
        negative zero, the mark of a missing value."""
        negative = self.uint(1)
        value = (self.unary() << p) + self.uint(p)
        if negative:
            return -value if value else None
        return value

    def rice_quotient_first(self, p: int) -> int | None:
        """The variant of the global VTEC coefficients (ZDB062): quotient, sign,
        remainder."""
        quotient = self.unary()
        negative = self.uint(1)
        value = (quotient << p) + self.uint(p)
        if negative:
            return -value if value else None
        return value


def satellite_name(gnss: str, satellite_id: int) -> str:
    """RINEX name of a satellite from its SSRZ satellite ID (table 10.2): the PRN for
    most systems, QZSS counts from PRN 193 (J01) and SBAS from PRN 120 (S20)."""
    if gnss == "S":
        return f"S{satellite_id + 19:02d}"
    return f"{gnss}{satellite_id:02d}"


def _satellite_order(name: str) -> tuple[int, int]:
    return SSRZ_GNSS.index(name[0]), int(name[1:])


def sort_satellites(names) -> tuple[str, ...]:
    """Satellites in SSRZ order: by GNSS ID, then by satellite ID."""
    return tuple(sorted(names, key=_satellite_order))


def _gnss_mask(bits: _Bits) -> tuple[str, ...]:
    """ZDF012: the GNSS of a definition, in ascending GNSS ID."""
    mask = bits.flags(_GNSS_MASK_BITS)
    if any(mask[len(SSRZ_GNSS) :]):
        raise SsrzError("reserved GNSS set")
    return tuple(gnss for gnss, present in zip(SSRZ_GNSS, mask) if present)


# --- Results ------------------------------------------------------------------------


@dataclass(frozen=True)
class SsrzTime:
    """ZM009: the reference time of the corrections. `week` and `seconds` are GPS time;
    `week` is None if only a time of week or day is sent."""

    definition: int
    week: int | None
    seconds: int  # within the period of the definition
    period: int  # seconds

    @property
    def gps_time(self) -> float | None:
        """Seconds since the GPS epoch, if the week is known."""
        if self.week is None:
            return None
        return self.week * 604800 + self.seconds


@dataclass(frozen=True)
class StreamIdentification:
    """ZM010."""

    version: int
    provider_id: int  # 33: BKG (table 15.6)
    stream_id: int


@dataclass(frozen=True)
class LeapSecond:
    """ZM024: GPS time runs ahead of UTC by `leap_seconds` until the announced month;
    then by `leap_seconds + change` (None: a stored announcement is withdrawn)."""

    year: int
    month: int
    leap_seconds: int
    change: int | None


@dataclass(frozen=True)
class SatelliteGroups:
    """ZM011 or ZMB011: the satellites of the low rate and high rate satellite groups,
    each in SSRZ order."""

    low_rate: tuple[tuple[str, ...], ...]
    high_rate: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class MetadataBlock:
    """One block of ZM012, as it arrived."""

    tag: int  # ZDF020: which variant of the block
    type_number: int  # ZDF003: which message it describes
    bits: int
    decoded: bool


@dataclass(frozen=True)
class Timing:
    """Update interval and offset of a correction (ZDB018, ZDB040), in seconds."""

    interval: int
    offset: int


@dataclass(frozen=True)
class ModelParameters:
    """ZDB060: model parameters, interpreted per model in the appendix (clause 15.1)."""

    model_id: int
    version: int
    integers: tuple[int, ...]
    floats: tuple[float, ...]


@dataclass(frozen=True)
class RiceBlockDefinition:
    """The attribute of a Rice block (which orders or parameters it carries, as a bit
    mask) and its default bin size p0."""

    mask: tuple[bool, ...]
    p0: int


@dataclass(frozen=True)
class GridIonosphereMetadata:
    """ZMB003."""

    tag: int
    resolution: float  # TECU
    p0: int
    timing: Timing | None


@dataclass(frozen=True)
class GridTroposphereMetadata:
    """ZMB004: the transmitted components with their resolution and default bin size."""

    tag: int
    timing: Timing | None
    model: ModelParameters
    components: tuple[tuple[str, float, int], ...]  # (component, resolution, p0)

    @property
    def parameters_per_point(self) -> int:
        """N_GRT (table 15.2): corrections per grid point and component."""
        return max(1, self.model.integers[0]) if self.model.integers else 1


@dataclass(frozen=True)
class RegionalIonosphereMetadata:
    """ZMB005: a polynomial in north and east per satellite around the pierce point of
    the ground point origin (clause 11.4.2), model 8 (table 15.3)."""

    tag: int
    timing: Timing | None
    model: ModelParameters
    resolution: float  # TECU
    coefficients: tuple[tuple[int, int], ...]  # (l, m) of a_lm in transmission order
    rice_blocks: tuple[RiceBlockDefinition, ...]  # attribute: orders l + m

    @property
    def layer_height_km(self) -> float | None:
        return self.model.floats[0] if len(self.model.floats) > 0 else None

    @property
    def origin(self) -> tuple[float, float, float] | None:
        """Ground point origin: latitude, longitude (degrees), height (m)."""
        if len(self.model.floats) < 4:
            return None
        return self.model.floats[1], self.model.floats[2], self.model.floats[3]

    @property
    def origin_interval(self) -> int:
        """Seconds for which a pierce point origin stays fixed; it is set anew at every
        multiple of them in GPS time. Table 15.3 has the third integer parameter only
        as "always 60"; the reference decoder of Geo++ reads it so, and the stream
        agrees."""
        integers = self.model.integers
        return integers[2] if len(integers) > 2 and integers[2] > 0 else 60


@dataclass(frozen=True)
class GlobalVtecMetadata:
    """ZMB006: spherical harmonics, model 4 (table 15.4)."""

    tag: int
    timing: Timing | None
    model: ModelParameters
    resolution: float  # TECU

    @property
    def coefficient_count(self) -> int:
        return self.model.integers[0]

    @property
    def degree(self) -> int:
        return self.model.integers[2]

    @property
    def order(self) -> int:
        return self.model.integers[3]

    @property
    def phase_time(self) -> float:
        """Seconds of the day of the TEC maximum the sun-fixed longitude refers to
        (eq. 11.26: 50400 s, 14:00); sent as the first float parameter."""
        floats = self.model.floats
        return floats[0] if floats and floats[0] > 0 else 50400.0

    @property
    def layer_height_km(self) -> float:
        """Height of the ionospheric layer; 400 km, the height of the gridded stage
        (eq. 11.54), if not sent."""
        floats = self.model.floats
        return floats[1] if len(floats) > 1 else 400.0


@dataclass(frozen=True)
class TroposphereComponentDefinition:
    """ZDB050: a basic component of the regional troposphere."""

    component: str
    max_horizontal_order: int
    max_vertical_order: int
    resolution: float
    rice_blocks: tuple[RiceBlockDefinition, ...]  # attribute: horizontal orders l + m
    separate_height_orders: bool

    @property
    def coefficients(self) -> tuple[tuple[int, int, int], ...]:
        """(k, l, m) of a_klm in transmission order: per height order k the triangle of
        horizontal orders along its anti-diagonals (clause 3.7.3.4)."""
        return tuple(
            (k, l, m)
            for k in range(self.max_vertical_order + 1)
            for l, m in _triangle(self.max_horizontal_order)
        )


@dataclass(frozen=True)
class MappingImprovementDefinition:
    """ZDB051: the mapping function improvement for low satellites (eq. 11.80)."""

    max_horizontal_order: int
    max_vertical_order: int
    max_elevation: int  # degrees
    resolution: float
    rice_blocks: tuple[RiceBlockDefinition, ...]  # attribute: parameters

    @property
    def parameter_count(self) -> int:
        return (self.max_horizontal_order + 1) * (self.max_vertical_order + 1)


@dataclass(frozen=True)
class TroposphereRegionDefinition:
    """ZMB007-3: one region of the regional troposphere."""

    region_id: int
    origin: tuple[float, float, float]  # latitude, longitude (degrees), height (m)
    horizontal_scale: float  # D_RT in m
    vertical_scale: float  # H_RT in m
    components: tuple[TroposphereComponentDefinition, ...]
    mapping: MappingImprovementDefinition | None


@dataclass(frozen=True)
class RegionalTroposphereMetadata:
    """ZMB007 (tag 3)."""

    tag: int
    timing: Timing | None
    regions: tuple[TroposphereRegionDefinition, ...]


@dataclass(frozen=True)
class ParameterBlockDefinition:
    """ZDB021: which GNSS one Rice block of a satellite parameter covers, and its default
    bin size p0 per component."""

    gnss: tuple[str, ...]
    p0: tuple[int, ...]


@dataclass(frozen=True)
class SatelliteParameter:
    """A satellite parameter of ZM001 or ZM002 (clock, orbit, velocity): the default
    resolution per component and the Rice blocks each component is sent in."""

    resolutions: tuple[float, ...]  # m, m/s
    blocks: tuple[ParameterBlockDefinition, ...]

    @property
    def components(self) -> int:
        return len(self.resolutions)


@dataclass(frozen=True, eq=False)
class SignalBlockDefinition:
    """ZDB022: the signals one Rice block of signal biases carries, per GNSS, and its
    default bin size."""

    signals: dict[str, tuple[str, ...]]  # GNSS -> RINEX signal codes
    p0: int


@dataclass(frozen=True)
class HighRateMetadata:
    """ZMB001: how ZM001 sends the clock (and possibly the radial orbit) every few
    seconds."""

    tag: int
    timing: Timing | None
    clock: SatelliteParameter
    orbit: SatelliteParameter | None  # radial component only


@dataclass(frozen=True, eq=False)
class LowRateMetadata:
    """ZMB002: how ZM002 sends clock, orbit, velocity and signal biases."""

    tag: int
    timing: Timing | None
    iod_bits: dict[str, int]  # length of the broadcast ephemeris IOD per GNSS (ZDB030)
    iod_tag: int
    clock: SatelliteParameter  # C0 (m), and C1 (m/s) if two components
    orbit: SatelliteParameter  # radial, along-track, cross-track (m)
    velocity: SatelliteParameter | None  # their rates (m/s)
    code_bias_resolution: float  # m
    code_bias_blocks: tuple[SignalBlockDefinition, ...]
    # Per GNSS the signal whose code bias is zero by definition and not sent.
    code_bias_reference: dict[str, tuple[str, ...]]
    phase_bias_signals: dict[str, tuple[str, ...]]
    phase_bias_cycle_range: int  # phase biases are sent modulo this many cycles
    phase_bias_bits: int
    continuity_bits: int  # longest continuity/overflow indicator
    satellite_ionosphere: bool  # ZM002 ends with the GSI coefficients (not decoded)

    @property
    def phase_bias_resolution(self) -> float:
        """Cycles per step of a phase bias value (ZDF071 / 2^ZDF072)."""
        return self.phase_bias_cycle_range / 2.0**self.phase_bias_bits


@dataclass(frozen=True)
class SsrzMetadata:
    """ZM012: what it takes to read the correction messages."""

    iod: int
    blocks: tuple[MetadataBlock, ...]
    low_rate_tag: int | None = None
    low_rate_timing: Timing | None = None
    low_rate: LowRateMetadata | None = None
    high_rate: HighRateMetadata | None = None
    grid_ionosphere: GridIonosphereMetadata | None = None
    grid_troposphere: GridTroposphereMetadata | None = None
    regional_ionosphere: RegionalIonosphereMetadata | None = None
    global_vtec: GlobalVtecMetadata | None = None
    regional_troposphere: RegionalTroposphereMetadata | None = None
    satellite_groups: SatelliteGroups | None = None
    grids: tuple[Grid, ...] = ()


@dataclass(frozen=True)
class GridChain:
    """The points of one chain of a grid (ZDB010), in chain order."""

    latitudes: tuple[float, ...]  # degrees
    longitudes: tuple[float, ...]
    heights: tuple[float, ...] | None  # ellipsoidal, m
    # (a, b, c) for every point from the fourth on, for the gridded data predictor.
    predictor_points: tuple[tuple[int, int, int], ...] | None


@dataclass(frozen=True)
class Grid:
    """A grid of ground points (ZDB012). GEPOS sends 14 of them, roughly one per federal
    state, 181 points in all: the reference stations of the network."""

    grid_id: int
    iod: int
    resolution: float  # radians
    chains: tuple[GridChain, ...]

    @property
    def point_count(self) -> int:
        return sum(len(chain.latitudes) for chain in self.chains)

    @property
    def latitudes(self) -> np.ndarray:
        return np.array([lat for chain in self.chains for lat in chain.latitudes])

    @property
    def longitudes(self) -> np.ndarray:
        return np.array([lon for chain in self.chains for lon in chain.longitudes])

    @property
    def heights(self) -> np.ndarray:
        """Heights of all points, NaN where a chain has none."""
        return np.array(
            [
                height
                for chain in self.chains
                for height in (chain.heights or (math.nan,) * len(chain.latitudes))
            ]
        )


@dataclass(frozen=True)
class GridDefinitions:
    """ZM013: one or more grids."""

    grids: tuple[Grid, ...]


@dataclass(frozen=True)
class PhaseBias:
    """A phase bias of ZM002 in cycles, sent modulo the cycle range, and how long the
    provider has kept it free of cycle slips: `continuity` counts down from the longest
    indicator (0: no slip for the longest time the indicator can tell)."""

    cycles: float
    continuity: int


@dataclass(frozen=True, eq=False)
class LowRateCorrections:
    """ZM002: per satellite the corrections to its broadcast clock and orbit and its
    signal biases, every 30 s (clause 11.1 to 11.3). The satellites listed here are the
    ones the other satellite dependent messages refer to.

    `clock` holds C0 (m) and, if sent, C1 (m/s); `orbit` the radial, along-track and
    cross-track correction (m) to the position from the broadcast ephemeris named by
    `ephemeris_iods`; `velocity` their rates (m/s). Values are NaN where not sent.
    """

    time_tag: int
    iod: int  # of the metadata
    satellites: tuple[str, ...]
    gps_time: float | None
    ephemeris_iods: dict[str, int] = field(default_factory=dict)
    clock: np.ndarray | None = None  # (satellites, 1 or 2)
    orbit: np.ndarray | None = None  # (satellites, 3)
    velocity: np.ndarray | None = None  # (satellites, 3)
    code_biases: dict[str, dict[str, float]] = field(default_factory=dict)  # m
    phase_biases: dict[str, dict[str, PhaseBias]] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class HighRateCorrections:
    """ZM001: every 5 s a clock correction (m) per satellite on top of the low rate one
    (eq. 11.18), and possibly a radial orbit correction (m)."""

    time_tag: int
    gps_time: float | None
    satellites: tuple[str, ...]
    clock: np.ndarray
    radial: np.ndarray | None


@dataclass(frozen=True, eq=False)
class GridIonosphere:
    """ZM003: per satellite and grid point the slant TEC left after the functional
    ionosphere models, mapped to the vertical, in TECU (clause 11.4.3). NaN: none."""

    time_tag: int
    gps_time: float | None
    grid_id: int
    grid_iod: int
    satellites: tuple[str, ...]
    values: np.ndarray  # (satellites, points)
    predicted: bool  # the gridded data predictor was used for some chain


@dataclass(frozen=True, eq=False)
class GridTroposphere:
    """ZM004: per component and grid point the scale factor to the model troposphere
    (clause 11.5.4). NaN: none."""

    time_tag: int
    gps_time: float | None
    grid_id: int
    grid_iod: int
    components: dict[str, np.ndarray]  # component -> (parameters, points)
    predicted: bool


@dataclass(frozen=True, eq=False)
class RegionalIonosphere:
    """ZM005: per satellite the coefficients a_lm of the regional polynomial in TECU."""

    time_tag: int
    gps_time: float | None
    satellites: tuple[str, ...]
    coefficients: np.ndarray  # (satellites, coefficients), NaN: none
    orders: tuple[tuple[int, int], ...]  # (l, m) per column


@dataclass(frozen=True, eq=False)
class GlobalVtec:
    """ZM006: vertical TEC as spherical harmonics in a sun-fixed frame (eq. 11.25)."""

    time_tag: int
    gps_time: float | None
    valid: bool
    degree: int
    order: int
    cosine: np.ndarray  # C_nm in TECU, (degree + 1, order + 1), 0 where m > n
    sine: np.ndarray  # S_nm
    layer_height_km: float
    phase_time: float  # seconds of the day

    def vtec(self, latitude: np.ndarray, longitude: np.ndarray, gps_time: float) -> np.ndarray:
        """Vertical TEC in TECU at pierce points (geocentric latitude and longitude in
        degrees on the spherical Earth model) at `gps_time` (seconds since the GPS
        epoch)."""
        latitude = np.radians(np.asarray(latitude, dtype=float))
        longitude = np.radians(np.asarray(longitude, dtype=float))
        seconds = gps_time % 86400.0
        sun = longitude + (seconds - self.phase_time) / 86400.0 * 2.0 * math.pi
        legendre = normalised_legendre(self.degree, self.order, np.sin(latitude))
        total = np.zeros(np.broadcast(latitude, longitude).shape)
        for n in range(self.degree + 1):
            for m in range(min(n, self.order) + 1):
                term = self.cosine[n, m] * np.cos(m * sun)
                if m:
                    term = term + self.sine[n, m] * np.sin(m * sun)
                total = total + term * legendre[n][m]
        return total


@dataclass(frozen=True, eq=False)
class TroposphereRegion:
    """ZM007-3: the coefficients of one region."""

    definition: TroposphereRegionDefinition
    components: dict[str, np.ndarray]  # component -> a_klm in the definition's order
    mapping: np.ndarray | None  # m_pq, NaN where not sent

    def scale_factor(
        self, component: str, latitude: np.ndarray, longitude: np.ndarray, height: np.ndarray
    ) -> np.ndarray | None:
        """The scale factor of `component` at ellipsoidal positions (degrees, m) after
        eq. 11.78, None if the region does not send it."""
        coefficients = self.components.get(component)
        if coefficients is None:
            return None
        definition = next(c for c in self.definition.components if c.component == component)
        lat0, lon0, h0 = self.definition.origin
        scale = EARTH_RADIUS_TROPOSPHERE / self.definition.horizontal_scale
        dphi = np.radians(np.asarray(latitude, dtype=float) - lat0) * scale
        dlam = (
            np.radians(np.asarray(longitude, dtype=float) - lon0)
            * math.cos(math.radians(lat0))
            * scale
        )
        dh = (np.asarray(height, dtype=float) - h0) / self.definition.vertical_scale
        order = definition.max_horizontal_order
        cheb_phi = chebyshev(order, dphi)
        cheb_lam = chebyshev(order, dlam)
        total = np.zeros(np.broadcast(dphi, dlam, dh).shape)
        for value, (k, l, m) in zip(coefficients, definition.coefficients):
            if not math.isnan(value):
                total = total + value * dh**k * cheb_phi[l] * cheb_lam[m]
        return total


@dataclass(frozen=True, eq=False)
class RegionalTroposphere:
    """ZM007."""

    time_tag: int
    gps_time: float | None
    regions: tuple[TroposphereRegion, ...]


SsrzResult = (
    SsrzTime
    | StreamIdentification
    | LeapSecond
    | SatelliteGroups
    | SsrzMetadata
    | GridDefinitions
    | LowRateCorrections
    | HighRateCorrections
    | GridIonosphere
    | GridTroposphere
    | RegionalIonosphere
    | GlobalVtec
    | RegionalTroposphere
)

# Earth radius of the regional troposphere (eq. 11.78).
EARTH_RADIUS_TROPOSPHERE = 6378135.0


# --- Models -------------------------------------------------------------------------


def _triangle(max_order: int) -> list[tuple[int, int]]:
    """(l, m) of an upper left triangle of coefficients up to l + m = max_order, along
    the anti-diagonals: a00, a01, a10, a02, a11, a20, ... (figure 3)."""
    return [(l, order - l) for order in range(max_order + 1) for l in range(order + 1)]


def chebyshev(order: int, x: np.ndarray) -> list[np.ndarray]:
    """Chebyshev polynomials T_0(x) ... T_order(x) (eq. 11.36)."""
    x = np.asarray(x, dtype=float)
    values = [np.ones_like(x), x]
    for _ in range(2, order + 1):
        values.append(2.0 * x * values[-1] - values[-2])
    return values[: order + 1]


def normalised_legendre(degree: int, order: int, x: np.ndarray) -> list[list[np.ndarray]]:
    """Fully normalised associated Legendre functions P_nm(x) for n <= degree and
    m <= min(n, order), with x = sin(latitude); geodetic convention (no Condon-Shortley
    phase): P_nm = sqrt((2 - delta_m0) (2n + 1) (n - m)! / (n + m)!) * P'_nm."""
    x = np.asarray(x, dtype=float)
    cosine = np.sqrt(np.clip(1.0 - x * x, 0.0, None))
    plain: list[list[np.ndarray]] = [[np.zeros_like(x)] * (degree + 1) for _ in range(degree + 1)]
    plain[0][0] = np.ones_like(x)
    for m in range(1, degree + 1):
        plain[m][m] = (2 * m - 1) * cosine * plain[m - 1][m - 1]
    for m in range(degree + 1):
        for n in range(m + 1, degree + 1):
            previous = plain[n - 2][m] if n - 2 >= m else np.zeros_like(x)
            plain[n][m] = ((2 * n - 1) * x * plain[n - 1][m] - (n + m - 1) * previous) / (n - m)
    result = []
    for n in range(degree + 1):
        row = []
        for m in range(min(n, order) + 1):
            factor = (2 if m else 1) * (2 * n + 1) * math.factorial(n - m) / math.factorial(n + m)
            row.append(math.sqrt(factor) * plain[n][m])
        result.append(row)
    return result


def predict_chain(values: list[float], points: tuple[tuple[int, int, int], ...]) -> list[float]:
    """The gridded data predictor (clause 12.1.3, table 12.1): `values` are the
    decoded differences of one chain (NaN: invalid), `points` the predictor points
    (a, b, c) from the fourth point on. Returns the grid values."""
    result = [math.nan] * len(values)
    start = next((i for i, value in enumerate(values) if not math.isnan(value)), None)
    if start is None:
        return result
    used = [math.nan] * len(values)  # x_i
    result[start] = used[start] = values[start]
    for i in range(start + 1, len(values)):
        if i <= start + 2 or i < 3:
            predicted = used[start]
        else:
            a, b, c = points[i - 3]
            predicted = used[b] + used[c] - used[a]
        if math.isnan(values[i]):
            used[i] = predicted
        else:
            result[i] = used[i] = predicted + values[i]
    return result


# --- Decoder ------------------------------------------------------------------------


@dataclass
class SsrzStatistics:
    """Per SSRZ message number: decoded, still lacking metadata, broken."""

    decoded: dict[int, int] = field(default_factory=dict)
    waiting: dict[int, int] = field(default_factory=dict)
    errors: dict[int, int] = field(default_factory=dict)


class SsrzDecoder:
    """Decodes the SSRZ messages of a stream and keeps what later messages refer to:
    metadata, satellite groups, grids, the satellites of the last low rate message and
    the time.

    A correction message that arrives before its metadata cannot be read (its length
    is not even known) and counts as waiting; within 30 s all metadata come round.
    """

    def __init__(self) -> None:
        self.metadata: SsrzMetadata | None = None
        self.satellite_groups: SatelliteGroups | None = None
        self.grids: dict[int, Grid] = {}
        self._gps_time: float | None = None
        self._leap_seconds: int | None = None
        # The satellites of the last ZM002 per low rate group, with its time tag.
        self._low_rate: dict[int, tuple[int, frozenset[str]]] = {}
        self._statistics = SsrzStatistics()
        self._handlers = {
            ZM_HIGH_RATE: self._high_rate_message,
            ZM_LOW_RATE: self._low_rate_message,
            ZM_GRID_IONOSPHERE: self._grid_ionosphere,
            ZM_GRID_TROPOSPHERE: self._grid_troposphere,
            ZM_REGIONAL_IONOSPHERE: self._regional_ionosphere,
            ZM_GLOBAL_VTEC: self._global_vtec,
            ZM_REGIONAL_TROPOSPHERE: self._regional_troposphere,
            ZM_TIME_TAG: self._time_tag,
            ZM_STREAM_ID: self._stream_id,
            ZM_SATELLITE_GROUPS: self._satellite_groups,
            ZM_METADATA: self._metadata,
            ZM_GRID_DEFINITION: self._grid_definition,
            ZM_LEAP_SECOND: self._leap_second,
        }

    def push(self, message: bytes) -> SsrzResult | None:
        """Takes an RTCM message (starting with its message type).

        Returns:
            What the message says, or None for another message, one not decoded here,
            one still lacking its metadata, or a broken one.
        """
        if len(message) < 3 or (message[0] << 4 | message[1] >> 4) != _GEOPP_MESSAGE:
            return None
        bits = _Bits(message, 12)
        if bits.uint(4) != _SSRZ_SUBTYPE:
            return None
        try:
            number = bits.prefix(2, 1, 5) + 1
        except SsrzError:
            return None
        handler = self._handlers.get(number)
        if handler is None:
            return None
        statistics = self._statistics
        try:
            result = handler(bits)
        except SsrzError, IndexError, KeyError, ValueError, ZeroDivisionError:
            statistics.errors[number] = statistics.errors.get(number, 0) + 1
            return None
        counter = statistics.waiting if result is None else statistics.decoded
        counter[number] = counter.get(number, 0) + 1
        return result

    def statistics(self) -> SsrzStatistics:
        s = self._statistics
        return SsrzStatistics(dict(s.decoded), dict(s.waiting), dict(s.errors))

    @property
    def leap_seconds(self) -> int | None:
        return self._leap_seconds

    # --- Time ----------------------------------------------------------------------

    def resolve_time_tag(self, time_tag: int) -> float | None:
        """GPS time (seconds since the GPS epoch) of a 15 minute time tag, taken as the
        one closest to the last ZM009."""
        now = self._gps_time
        if now is None:
            return None
        time = now - now % TIME_TAG_PERIOD + time_tag
        if time - now > TIME_TAG_PERIOD / 2:
            time -= TIME_TAG_PERIOD
        elif now - time > TIME_TAG_PERIOD / 2:
            time += TIME_TAG_PERIOD
        return time

    def _time_tag(self, bits: _Bits) -> SsrzTime:
        definition = bits.prefix(2, 1)
        week = None
        if definition == 0:
            seconds, period = bits.uint(7) * 30, 3600
        elif definition == 1:
            seconds, period = bits.uint(10) * 5, 3600
        elif definition == 2:
            seconds, period = bits.uint(10), TIME_TAG_PERIOD
        elif definition == 3:
            seconds, period = bits.uint(12), 3600
        elif definition == 4:
            seconds, period = bits.uint(17), 86400
        elif definition == 5:
            seconds, period = bits.uint(20), 604800
        elif definition == 6:
            week = bits.uint(12)
            seconds, period = bits.uint(20), 604800
        else:
            raise SsrzError("unknown time tag definition")
        result = SsrzTime(definition, week, seconds, period)
        if result.gps_time is not None:
            self._gps_time = result.gps_time
        return result

    def _stream_id(self, bits: _Bits) -> StreamIdentification:
        version = bits.uint(3) + 1
        provider = bits.prefix(8)
        stream = bits.prefix(8)
        # Further blocks (ZM010-X) are only announced so far; each says its size.
        while bits.prefix(8):
            bits.position += bits.prefix(8) * 8
        return StreamIdentification(version, provider, stream)

    def _leap_second(self, bits: _Bits) -> LeapSecond | None:
        if bits.prefix(8) != 1:
            return None
        year = 2023 + bits.prefix(5)
        month = bits.uint(4) + 1
        leap = 18 + bits.sign_magnitude(4)
        # ZDF254: -1, 0 or +1 second; -0 withdraws a stored announcement.
        raw = bits.uint(2)
        change = None if raw == 0b10 else (-1 if raw == 0b11 else raw)
        self._leap_seconds = leap
        return LeapSecond(year, month, leap, change)

    # --- Metadata ------------------------------------------------------------------

    def _satellite_groups(self, bits: _Bits) -> SatelliteGroups | None:
        tag = bits.prefix(8)
        if tag not in (1, 2):
            return None
        groups = _read_satellite_groups(bits, tag)
        self.satellite_groups = groups
        return groups

    def _grid_definition(self, bits: _Bits) -> GridDefinitions | None:
        tag = bits.prefix(8)
        if tag not in (3, 4):
            return None
        grids = _read_grids(bits, tag)
        for grid in grids:
            self.grids[grid.grid_id] = grid
        return GridDefinitions(grids)

    def _metadata(self, bits: _Bits) -> SsrzMetadata:
        iod = bits.prefix(2, 2)
        blocks = []
        decoded: dict[str, object] = {}
        while True:
            tag = bits.prefix(8)
            if tag == 0:
                break
            type_number = bits.uint(8)
            size = bits.uint(13)
            start = bits.position
            end = start + size
            if end > bits.end:
                raise SsrzError("metadata block beyond the message")
            ok = False
            reader = _METADATA_BLOCKS.get((type_number, tag))
            if reader is not None:
                bits.end, whole = end, bits.end
                try:
                    fields = reader(bits, tag)
                    ok = bits.position == end
                except SsrzError:
                    ok = False
                bits.end = whole
                if ok:
                    decoded.update(fields)
            blocks.append(MetadataBlock(tag, type_number, size, ok))
            bits.position = end
        metadata = SsrzMetadata(iod, tuple(blocks), **decoded)
        self.metadata = metadata
        if metadata.satellite_groups is not None:
            self.satellite_groups = metadata.satellite_groups
        for grid in metadata.grids:
            self.grids[grid.grid_id] = grid
        return metadata

    # --- Satellites of the low rate message ----------------------------------------

    def _group_count(self) -> int | None:
        groups = self.satellite_groups
        return None if groups is None else len(groups.low_rate)

    def _selected_groups(self, bits: _Bits) -> list[int] | None:
        count = self._group_count()
        if count is None:
            return None
        return [index for index, selected in enumerate(bits.flags(count)) if selected]

    def _low_rate_message(self, bits: _Bits) -> LowRateCorrections | None:
        """ZM002. The header (which satellites) is what the other satellite dependent
        messages need; the corrections follow only with the metadata of the same IOD."""
        metadata = self.metadata
        if metadata is None or metadata.low_rate_tag is None or self.satellite_groups is None:
            return None
        if metadata.low_rate_tag == 2:
            time_tag = bits.uint(10)
            iod = bits.prefix(2, 2)
            bits.uint(1)  # announcement of new metadata
        else:
            iod = bits.prefix(2, 2)
            bits.uint(1)
            time_tag = bits.uint(10)
        groups = self._selected_groups(bits)
        if groups is None:
            return None
        candidates = sort_satellites(
            {name for group in groups for name in self.satellite_groups.low_rate[group]}
        )
        available = frozenset(
            name for name, present in zip(candidates, bits.flags(len(candidates))) if present
        )
        for group in groups:
            members = set(self.satellite_groups.low_rate[group])
            self._low_rate[group] = (time_tag, available & members)
        satellites = sort_satellites(available)
        gps_time = self.resolve_time_tag(time_tag)
        meta = metadata.low_rate
        if meta is None or iod != metadata.iod:
            return LowRateCorrections(time_tag, iod, satellites, gps_time)
        # The broadcast ephemeris each correction refers to (ZDF309).
        iods = {name: bits.uint(meta.iod_bits.get(name[0], 0)) for name in satellites}
        clock = _read_parameter(bits, satellites, meta.clock)
        orbit = _read_parameter(bits, satellites, meta.orbit)
        velocity = None
        if meta.velocity is not None:
            velocity = _read_parameter(bits, satellites, meta.velocity)
        code_biases = _read_signal_biases(
            bits, satellites, meta.code_bias_blocks, meta.code_bias_resolution
        )
        for name in satellites:
            for signal in meta.code_bias_reference.get(name[0], ()):
                code_biases[name].setdefault(signal, 0.0)
        phase_biases = _read_phase_biases(bits, satellites, meta)
        return LowRateCorrections(
            time_tag,
            iod,
            satellites,
            gps_time,
            iods,
            clock,
            orbit,
            velocity,
            code_biases,
            phase_biases,
        )

    def _high_rate_message(self, bits: _Bits) -> HighRateCorrections | None:
        """ZM001: for the satellites of the chosen high rate groups that the still valid
        low rate message has."""
        metadata = self.metadata
        meta = None if metadata is None else metadata.high_rate
        groups = self.satellite_groups
        time_tag = bits.uint(10)
        if meta is None or groups is None:
            return None
        selected = [i for i, flag in enumerate(bits.flags(len(groups.high_rate))) if flag]
        members = {name for i in selected for name in groups.high_rate[i]}
        available = self._valid_low_rate_satellites(time_tag)
        if available is None:
            return None
        satellites = sort_satellites(members & available)
        clock = _read_parameter(bits, satellites, meta.clock)[:, 0]
        radial = None
        if meta.orbit is not None:
            radial = _read_parameter(bits, satellites, meta.orbit)[:, 0]
        return HighRateCorrections(
            time_tag, self.resolve_time_tag(time_tag), satellites, clock, radial
        )

    def _valid_low_rate_satellites(self, time_tag: int) -> set[str] | None:
        """All satellites of the low rate messages still valid at `time_tag`."""
        metadata = self.metadata
        timing = None if metadata is None else metadata.low_rate_timing
        interval = timing.interval if timing is not None else _DEFAULT_LOW_RATE_INTERVAL
        valid = [
            available
            for low_rate_tag, available in self._low_rate.values()
            if (time_tag - low_rate_tag) % TIME_TAG_PERIOD < max(interval, 1)
        ]
        if not valid:
            return None
        return set().union(*valid)

    def _satellites_for(self, groups: list[int], time_tag: int) -> tuple[str, ...] | None:
        """The satellites a satellite dependent message with this group list lists:
        those of the groups that the still valid low rate message had."""
        metadata = self.metadata
        timing = None if metadata is None else metadata.low_rate_timing
        interval = timing.interval if timing is not None else _DEFAULT_LOW_RATE_INTERVAL
        satellites: set[str] = set()
        for group in groups:
            entry = self._low_rate.get(group)
            if entry is None:
                return None
            low_rate_tag, available = entry
            if (time_tag - low_rate_tag) % TIME_TAG_PERIOD >= max(interval, 1):
                return None
            satellites |= available
        return sort_satellites(satellites)

    # --- Correction messages --------------------------------------------------------

    def _grid_for(self, grid_id: int, iod: int) -> Grid | None:
        grid = self.grids.get(grid_id)
        return grid if grid is not None and grid.iod == iod else None

    def _grid_ionosphere(self, bits: _Bits) -> GridIonosphere | None:
        metadata = self.metadata
        meta = None if metadata is None else metadata.grid_ionosphere
        time_tag = bits.uint(10)
        grid_id = bits.prefix(4) + 1
        iod = bits.uint(3)
        groups = self._selected_groups(bits)
        grid = self._grid_for(grid_id, iod)
        if meta is None or groups is None or grid is None:
            return None
        satellites = self._satellites_for(groups, time_tag)
        if satellites is None:
            return None
        values = np.empty((len(satellites), grid.point_count))
        predicted = False
        for row in range(len(satellites)):
            values[row], used = _read_gridded(bits, grid, meta.resolution, meta.p0)
            predicted |= used
        return GridIonosphere(
            time_tag,
            self.resolve_time_tag(time_tag),
            grid_id,
            iod,
            satellites,
            values,
            predicted,
        )

    def _grid_troposphere(self, bits: _Bits) -> GridTroposphere | None:
        metadata = self.metadata
        meta = None if metadata is None else metadata.grid_troposphere
        time_tag = bits.uint(10)
        grid_id = bits.prefix(4) + 1
        iod = bits.uint(3)
        grid = self._grid_for(grid_id, iod)
        if meta is None or grid is None:
            return None
        components = {}
        predicted = False
        for component, resolution, p0 in meta.components:
            rows = []
            for _ in range(meta.parameters_per_point):
                row, used = _read_gridded(bits, grid, resolution, p0)
                rows.append(row)
                predicted |= used
            components[component] = np.array(rows)
        return GridTroposphere(
            time_tag, self.resolve_time_tag(time_tag), grid_id, iod, components, predicted
        )

    def _regional_ionosphere(self, bits: _Bits) -> RegionalIonosphere | None:
        metadata = self.metadata
        meta = None if metadata is None else metadata.regional_ionosphere
        time_tag = bits.uint(10)
        groups = self._selected_groups(bits)
        if meta is None or groups is None or meta.tag != 1:
            return None
        satellites = self._satellites_for(groups, time_tag)
        if satellites is None:
            return None
        coefficients = np.full((len(satellites), len(meta.coefficients)), math.nan)
        for block in meta.rice_blocks:
            columns = [
                column
                for column, (l, m) in enumerate(meta.coefficients)
                if l + m < len(block.mask) and block.mask[l + m]
            ]
            if not columns or not satellites:
                continue
            values = _read_rice_block(
                bits, len(columns) * len(satellites), meta.resolution, block.p0
            )
            for index, column in enumerate(columns):
                coefficients[:, column] = values[
                    index * len(satellites) : (index + 1) * len(satellites)
                ]
        return RegionalIonosphere(
            time_tag, self.resolve_time_tag(time_tag), satellites, coefficients, meta.coefficients
        )

    def _global_vtec(self, bits: _Bits) -> GlobalVtec | None:
        metadata = self.metadata
        meta = None if metadata is None else metadata.global_vtec
        time_tag = bits.uint(10)
        valid = bits.uint(1) == 1
        if meta is None or meta.tag not in (2, 3):
            return None
        degree, order = meta.degree, meta.order
        cosine = np.zeros((degree + 1, order + 1))
        sine = np.zeros((degree + 1, order + 1))
        if valid:
            p = 10 + bits.prefix(3, 1)
            for m in range(order + 1):
                for n in range(m, degree + 1):
                    cosine[n, m] = _scaled(bits.rice_quotient_first(p), meta.resolution)
            for m in range(1, order + 1):
                for n in range(m, degree + 1):
                    sine[n, m] = _scaled(bits.rice_quotient_first(p), meta.resolution)
        return GlobalVtec(
            time_tag,
            self.resolve_time_tag(time_tag),
            valid,
            degree,
            order,
            cosine,
            sine,
            meta.layer_height_km,
            meta.phase_time,
        )

    def _regional_troposphere(self, bits: _Bits) -> RegionalTroposphere | None:
        metadata = self.metadata
        meta = None if metadata is None else metadata.regional_troposphere
        time_tag = bits.uint(10)
        if meta is None or meta.tag != 3:
            return None
        definitions = {region.region_id: region for region in meta.regions}
        regions = []
        while True:
            region_id = bits.prefix(8) + 1
            definition = definitions.get(region_id)
            if definition is None:
                return None
            components = {}
            for component in definition.components:
                components[component.component] = _read_coefficients(bits, component)
            mapping = None
            if definition.mapping is not None:
                mapping = _read_parameters(bits, definition.mapping)
            regions.append(TroposphereRegion(definition, components, mapping))
            # Each ZM007-3 block ends with the multi-region flag (ZDF030, table 8.11);
            # a message ending right after the last coefficient has no room for it.
            if not (bits.remaining() and bits.uint(1)):
                break
        return RegionalTroposphere(time_tag, self.resolve_time_tag(time_tag), tuple(regions))


def _scaled(value: int | None, resolution: float) -> float:
    return math.nan if value is None else value * resolution


def _read_rice_block(bits: _Bits, count: int, resolution: float, p0: int) -> list[float]:
    """An SSRZ Rice block (ZDB002): scale factor and bin size indicators, then `count`
    values, scaled to physical units (NaN: none)."""
    scale = bits.prefix(2)
    p = p0 + bits.prefix(2)
    step = resolution * 2.0**scale
    return [_scaled(bits.rice(p), step) for _ in range(count)]


def _read_parameter(
    bits: _Bits, satellites: tuple[str, ...], parameter: SatelliteParameter
) -> np.ndarray:
    """ZDB003 per component: the Rice blocks of a satellite parameter, each for the
    satellites of its GNSS in SSRZ order (a block without satellites is not sent).
    Returns (satellites, components), NaN where not sent."""
    values = np.full((len(satellites), parameter.components), math.nan)
    for component, resolution in enumerate(parameter.resolutions):
        for block in parameter.blocks:
            rows = [row for row, name in enumerate(satellites) if name[0] in block.gnss]
            if rows:
                read = _read_rice_block(bits, len(rows), resolution, block.p0[component])
                values[rows, component] = read
    return values


def _read_signal_biases(
    bits: _Bits,
    satellites: tuple[str, ...],
    blocks: tuple[SignalBlockDefinition, ...],
    resolution: float,
) -> dict[str, dict[str, float]]:
    """ZDB004: per Rice block the biases of its signals; per GNSS signal by signal in
    table order, each for all satellites of the GNSS. (Clause 7.2.4 describes the
    satellites as the outer order; the stream and the reference decoder have the
    signals outside.)"""
    biases: dict[str, dict[str, float]] = {name: {} for name in satellites}
    for block in blocks:
        entries = [
            (name, signal)
            for gnss in SSRZ_GNSS
            for signal in block.signals.get(gnss, ())
            for name in satellites
            if name[0] == gnss
        ]
        if not entries:
            continue
        for (name, signal), value in zip(
            entries, _read_rice_block(bits, len(entries), resolution, block.p0), strict=True
        ):
            if not math.isnan(value):
                biases[name][signal] = value
    return biases


def _read_phase_biases(
    bits: _Bits, satellites: tuple[str, ...], meta: LowRateMetadata
) -> dict[str, dict[str, PhaseBias]]:
    """ZDB032: per satellite and phase bias signal a flag, then the value (unsigned,
    modulo the cycle range) and the continuity indicator."""
    result: dict[str, dict[str, PhaseBias]] = {}
    for name in satellites:
        entries = {}
        for signal in meta.phase_bias_signals.get(name[0], ()):
            if bits.uint(1):
                cycles = bits.uint(meta.phase_bias_bits) * meta.phase_bias_resolution
                entries[signal] = PhaseBias(cycles, bits.unary_limited(meta.continuity_bits))
        result[name] = entries
    return result


def _read_gridded(bits: _Bits, grid: Grid, resolution: float, p0: int) -> tuple[np.ndarray, bool]:
    """ZDB006: one value per grid point, chain by chain (ZDB005). Returns the values
    and whether the predictor was used."""
    values: list[float] = []
    predicted = False
    for chain in grid.chains:
        count = len(chain.latitudes)
        if bits.uint(1):  # predictor flag (ZDF315)
            if chain.predictor_points is None:
                raise SsrzError("predictor without predictor points")
            first = _read_rice_block(bits, 1, resolution, p0)
            rest = _read_rice_block(bits, count - 1, resolution, p0) if count > 1 else []
            values += predict_chain(first + rest, chain.predictor_points)
            predicted = True
        else:
            values += _read_rice_block(bits, count, resolution, p0)
    return np.array(values), predicted


def _read_coefficients(bits: _Bits, component: TroposphereComponentDefinition) -> np.ndarray:
    """ZDB007 for a regional troposphere component: the Rice blocks carry the
    coefficients of the horizontal orders flagged in their attribute, per height order
    in separate blocks or together. How several height orders share one block is not
    spelled out; taken here in the order of `coefficients`. GEPOS sends order 0 only."""
    coefficients = component.coefficients
    values = np.full(len(coefficients), math.nan)
    height_passes = (
        range(component.max_vertical_order + 1) if component.separate_height_orders else [None]
    )
    for height in height_passes:
        for block in component.rice_blocks:
            columns = [
                index
                for index, (k, l, m) in enumerate(coefficients)
                if (height is None or k == height) and l + m < len(block.mask) and block.mask[l + m]
            ]
            if columns:
                read = _read_rice_block(bits, len(columns), component.resolution, block.p0)
                values[columns] = read
    return values


def _read_parameters(bits: _Bits, mapping: MappingImprovementDefinition) -> np.ndarray:
    """ZDB009: the parameters flagged in each Rice block's parameter list."""
    values = np.full(mapping.parameter_count, math.nan)
    for block in mapping.rice_blocks:
        columns = [index for index, flagged in enumerate(block.mask) if flagged]
        if columns:
            values[columns] = _read_rice_block(bits, len(columns), mapping.resolution, block.p0)
    return values


# --- Metadata blocks ------------------------------------------------------------------


def _read_timing(bits: _Bits) -> Timing:
    """ZDB018."""
    return Timing(bits.uint(6), bits.uint(6))


def _read_update_offset(bits: _Bits) -> Timing:
    """ZDB040."""
    return Timing(bits.prefix(6) + 1, bits.prefix(6))


def _read_satellite_timing(bits: _Bits) -> Timing | None:
    """ZDF010, ZDF055 and the ZDB019 blocks: the timing of satellite dependent
    corrections per satellite group; returns the first."""
    groups = bits.uint(5) + 1
    count = bits.uint(5) + 1
    first = None
    for _ in range(count):
        timing = _read_timing(bits)
        bits.flags(groups)
        first = first or timing
    return first


def _read_id_timing(bits: _Bits) -> Timing | None:
    """ZDB044 or ZDB046: update and offset blocks per grid or region ID; returns the
    first."""
    max_id = bits.prefix(8) + 1
    count = bits.prefix(3) + 1
    first = None
    for _ in range(count):
        timing = _read_update_offset(bits)
        bits.flags(max_id)
        first = first or timing
    return first


def _read_model(bits: _Bits) -> ModelParameters:
    """ZDB060."""
    model_id = bits.uint(8)
    version = bits.uint(8)
    integers = tuple(bits.int32() for _ in range(bits.uint(8)))
    floats = tuple(bits.float32() for _ in range(bits.uint(8)))
    return ModelParameters(model_id, version, integers, floats)


def _read_rice_definitions(bits: _Bits, mask_bits: int) -> tuple[RiceBlockDefinition, ...]:
    """ZDB023 with its ZDB024 blocks (order bit mask, then p0), or ZDB027 with its ZDB028
    blocks (parameter list bit mask, then p0)."""
    count = bits.prefix(2, 1, 5)
    return tuple(
        RiceBlockDefinition(bits.flags(mask_bits), bits.prefix(2, 1, 5)) for _ in range(count)
    )


def _read_parameter_blocks(bits: _Bits, components: int) -> tuple[ParameterBlockDefinition, ...]:
    """ZDF042 and its ZDB021 blocks: GNSS bit mask, then p0 per component."""
    return tuple(
        ParameterBlockDefinition(
            _gnss_mask(bits), tuple(bits.prefix(2, 1, 5) for _ in range(components))
        )
        for _ in range(bits.prefix(2, 1, 5))
    )


def _read_signal_mask(bits: _Bits) -> dict[str, tuple[str, ...]]:
    """ZDB020: the GNSS, and per GNSS its signals (ZDF019) in table order."""
    signals = {}
    for gnss in _gnss_mask(bits):
        names = _SIGNALS.get(gnss, ())
        signals[gnss] = tuple(
            names[bit] if bit < len(names) and names[bit] else f"#{bit}"
            for bit, flagged in enumerate(bits.flags(_SIGNAL_MASK_BITS))
            if flagged
        )
    return signals


def _read_high_rate(bits: _Bits, tag: int) -> dict[str, object]:
    """ZMB001 (tag 1)."""
    if tag != 1:
        raise SsrzError("high rate metadata only with tag 1")
    timing = _read_satellite_timing(bits)
    count = bits.uint(2)
    resolution = bits.prefix(10) * 1e-4
    clock = SatelliteParameter((resolution,) * count, _read_parameter_blocks(bits, count))
    orbit = None
    count = bits.uint(2)
    if count:
        resolution = bits.prefix(10) * 1e-4
        orbit = SatelliteParameter((resolution,) * count, _read_parameter_blocks(bits, count))
    return {"high_rate": HighRateMetadata(tag, timing, clock, orbit)}


def _read_low_rate(bits: _Bits, tag: int) -> dict[str, object]:
    """ZMB002 (tag 1). Resolutions: clock C0 0.1 mm, C1 0.01 mm/s, orbit 0.1 mm,
    velocity 0.001 mm/s, code biases 0.1 mm (ZDF062 to ZDF070)."""
    if tag != 1:
        raise SsrzError("low rate metadata only with tag 1")
    timing = _read_satellite_timing(bits)
    # ZDB030: length of the broadcast ephemeris IOD per GNSS, and how to use it.
    iod_bits = {gnss: bits.uint(4) for gnss in _gnss_mask(bits)}
    iod_tag = bits.prefix(2)

    count = bits.uint(2)
    resolutions = [bits.prefix(10) * 1e-4]
    if count == 2:
        resolutions.append(bits.prefix(10) * 1e-5)
    clock = SatelliteParameter(tuple(resolutions[:count]), _read_parameter_blocks(bits, count))

    count = bits.uint(2)
    resolutions = [bits.prefix(10) * 1e-4, bits.prefix(8) * 1e-4, bits.prefix(8) * 1e-4]
    orbit = SatelliteParameter(tuple(resolutions[:count]), _read_parameter_blocks(bits, count))

    velocity = None
    count = bits.uint(2)
    if count:
        resolutions = [bits.prefix(10) * 1e-6 for _ in range(3)]
        velocity = SatelliteParameter(
            tuple(resolutions[:count]), _read_parameter_blocks(bits, count)
        )

    code_resolution = bits.prefix(11) * 1e-4
    code_blocks = tuple(
        SignalBlockDefinition(_read_signal_mask(bits), bits.prefix(2))
        for _ in range(bits.prefix(2))
    )
    code_reference = _read_signal_mask(bits)
    phase_signals = _read_signal_mask(bits)
    cycle_range = bits.prefix(5)
    phase_bits = bits.prefix(5)
    continuity_bits = bits.prefix(5)
    satellite_ionosphere = bits.uint(1) == 1
    if satellite_ionosphere:
        # Model, resolution and Rice blocks of the GSI coefficients: not sent by GEPOS
        # and not decoded; they end the block.
        bits.position = bits.end
    return {
        "low_rate_tag": tag,
        "low_rate_timing": timing,
        "low_rate": LowRateMetadata(
            tag,
            timing,
            iod_bits,
            iod_tag,
            clock,
            orbit,
            velocity,
            code_resolution,
            code_blocks,
            code_reference,
            phase_signals,
            cycle_range,
            phase_bits,
            continuity_bits,
            satellite_ionosphere,
        ),
    }


def _read_grid_ionosphere(bits: _Bits, tag: int) -> dict[str, object]:
    timing = _read_satellite_timing(bits) if tag == 1 else _read_id_timing(bits)
    resolution = bits.prefix(4) * 0.001
    p0 = bits.prefix(2, 1, 5)
    return {"grid_ionosphere": GridIonosphereMetadata(tag, resolution, p0, timing)}


def _read_grid_troposphere(bits: _Bits, tag: int) -> dict[str, object]:
    timing = _read_timing(bits) if tag == 2 else _read_id_timing(bits)
    model = _read_model(bits)
    mask = bits.flags(16)
    components = []
    for component, present in zip(_TROPOSPHERE_COMPONENTS, mask):
        if present:
            resolution = bits.prefix(8) * 1e-5
            components.append((component, resolution, bits.prefix(2, 1, 5)))
    return {"grid_troposphere": GridTroposphereMetadata(tag, timing, model, tuple(components))}


def _rsi_coefficients(count: int) -> tuple[tuple[int, int], ...]:
    """The coefficients a_lm of the regional ionosphere by their number (table 15.3):
    a00; a00, a01, a10; a00, a01, a10, a11; otherwise the triangle in its order."""
    if count == 4:
        return ((0, 0), (0, 1), (1, 0), (1, 1))
    triangle = []
    order = 0
    while len(triangle) < count:
        triangle += _triangle(order)[len(triangle) :]
        order += 1
    return tuple(triangle[:count])


def _read_regional_ionosphere(bits: _Bits, tag: int) -> dict[str, object]:
    timing = _read_satellite_timing(bits)
    model = _read_model(bits)
    resolution = bits.prefix(4) * 0.001
    coefficients = _rsi_coefficients(model.integers[0] if model.integers else 1)
    orders = max(l + m for l, m in coefficients) + 1
    blocks = _read_rice_definitions(bits, orders)
    return {
        "regional_ionosphere": RegionalIonosphereMetadata(
            tag, timing, model, resolution, coefficients, blocks
        )
    }


def _read_global_vtec(bits: _Bits, tag: int) -> dict[str, object]:
    timing = _read_timing(bits)
    model = _read_model(bits)
    bits.prefix(2)  # encoder type (ZDF129): 1 = Rice
    resolution = bits.prefix(4) * 1e-3 if tag == 2 else bits.prefix(8) * 1e-5
    return {"global_vtec": GlobalVtecMetadata(tag, timing, model, resolution)}


def _read_troposphere_component(bits: _Bits, component: str) -> TroposphereComponentDefinition:
    """ZDB050. The flag for separate blocks per height order follows only for the dry
    component, as the notes of ZDB050 say (bit 0 of ZDF107)."""
    horizontal = bits.prefix(2)
    vertical = bits.prefix(2)
    resolution = bits.prefix(8) * 1e-5
    blocks = _read_rice_definitions(bits, horizontal + 1)
    separate = bits.uint(1) == 1 if component == TROPOSPHERE_DRY else False
    return TroposphereComponentDefinition(
        component, horizontal, vertical, resolution, blocks, separate
    )


def _read_mapping(bits: _Bits) -> MappingImprovementDefinition:
    """ZDB051."""
    horizontal = bits.prefix(2)
    vertical = bits.prefix(2)
    elevation = bits.uint(6)
    resolution = bits.prefix(8) * 1e-6
    blocks = _read_rice_definitions(bits, (horizontal + 1) * (vertical + 1))
    return MappingImprovementDefinition(horizontal, vertical, elevation, resolution, blocks)


def _read_regional_troposphere(bits: _Bits, tag: int) -> dict[str, object]:
    if tag != 3:
        raise SsrzError("regional troposphere metadata only with tag 3")
    timing = _read_id_timing(bits)
    regions = []
    while True:
        region_id = bits.prefix(8) + 1
        latitude = bits.sign_magnitude(11) * 0.1
        longitude = bits.sign_magnitude(12) * 0.1
        height = float(bits.sign_magnitude(14))
        horizontal = 100e3 + bits.prefix(8) * 10e3
        vertical = 100.0 + bits.prefix(8) * 10.0
        mask = bits.flags(16)
        components = tuple(
            _read_troposphere_component(bits, component)
            for component, present in zip(_TROPOSPHERE_COMPONENTS, mask)
            if present
        )
        mapping = _read_mapping(bits) if mask[_MAPPING_IMPROVEMENT_BIT] else None
        regions.append(
            TroposphereRegionDefinition(
                region_id,
                (latitude, longitude, height),
                horizontal,
                vertical,
                components,
                mapping,
            )
        )
        if not bits.uint(1):  # multi-region flag (ZDF030)
            break
    return {"regional_troposphere": RegionalTroposphereMetadata(tag, timing, tuple(regions))}


def _read_satellite_groups(bits: _Bits, tag: int) -> SatelliteGroups:
    """ZMB011: NG(LR) and NG(HR) are sent as they are with tag 1, minus one with
    tag 2."""
    extra = 1 if tag == 2 else 0
    low_rate = tuple(_read_group(bits) for _ in range(bits.uint(5) + extra))
    high_rate = tuple(_read_group(bits) for _ in range(bits.uint(5) + extra))
    return SatelliteGroups(low_rate, high_rate)


def _read_group(bits: _Bits) -> tuple[str, ...]:
    """ZDB017."""
    systems = _gnss_mask(bits)
    maxima = [bits.uint(7) for _ in systems]
    mode = bits.uint(4)
    satellites = []
    for gnss, maximum in zip(systems, maxima):
        if mode == _GROUP_MODE_LIST:
            members = [i + 1 for i, flag in enumerate(bits.flags(maximum)) if flag]
        elif mode == _GROUP_MODE_GNSS:
            members = list(range(1, maximum + 1))
        else:
            raise SsrzError("reserved satellite group mode")
        satellites += [satellite_name(gnss, member) for member in members]
    return tuple(satellites)


def _read_satellite_group_block(bits: _Bits, tag: int) -> dict[str, object]:
    return {"satellite_groups": _read_satellite_groups(bits, tag)}


def _read_grid_block(bits: _Bits, tag: int) -> dict[str, object]:
    return {"grids": _read_grids(bits, tag)}


def _read_grids(bits: _Bits, tag: int) -> tuple[Grid, ...]:
    """ZMB013 with its ZDB012 blocks. With tag 4 the coordinate and height resolution
    are common to all grids, with tag 3 each grid has its own."""
    count = bits.prefix(4)
    resolution = height_resolution = None
    if tag == 4:
        resolution = _coordinate_resolution(bits)
        height_resolution = bits.prefix(5)
    grids = []
    for _ in range(count):
        grid_id = bits.prefix(4) + 1
        iod = bits.uint(3)
        grid_resolution = _coordinate_resolution(bits) if tag == 3 else resolution
        chains = tuple(
            _read_chain(bits, tag, grid_resolution, height_resolution)
            for _ in range(bits.prefix(4) + 1)
        )
        grids.append(Grid(grid_id, iod, grid_resolution, chains))
    return tuple(grids)


def _coordinate_resolution(bits: _Bits) -> float:
    """ZDF092 and ZDF093: n * 10^-(2 + m) radians. ZDF092 is PC(2,0) in the stream
    (format description: PC(2,1,0))."""
    order = bits.prefix(2)
    return bits.uint(8) * 10.0 ** -(2 + order)


def _read_chain(
    bits: _Bits, tag: int, resolution: float, height_resolution: int | None
) -> GridChain:
    """ZDB010: a chain of grid points, reconstructed triangle by triangle (clause 12.1)."""
    count = bits.prefix(5) + 1
    p = bits.prefix(4)
    first = (_required(bits.rice(p)) * resolution, _required(bits.rice(p)) * resolution)
    pairs: list[tuple[float, float]] = []
    flags: list[tuple[bool, ...]] = []
    if count > 1:
        p = bits.prefix(4)
        values = [_required(bits.rice(p)) * resolution for _ in range(2 * (count - 1))]
        pairs = list(zip(values[0::2], values[1::2]))
        flags = [bits.flags(count - 2) for _ in range(4)]
    heights = None
    has_heights = tag == 4 and bool(height_resolution)
    if tag == 3 and bits.uint(1):
        height_resolution = bits.prefix(5)
        has_heights = True
    if has_heights:
        p = bits.prefix(4)
        height = _required(bits.rice(p))
        steps = [height]
        if count > 1:
            p = bits.prefix(4)
            for _ in range(count - 1):
                height += _required(bits.rice(p))
                steps.append(height)
        heights = tuple(float(h * height_resolution) for h in steps)
    predictor = None
    if bits.uint(1):
        predictor = tuple(
            (i - bits.prefix(2) - 1, i - bits.prefix(2) - 1, i - bits.prefix(2) - 1)
            for i in range(3, count)
        )
    points = reconstruct_chain(first, pairs, flags)
    return GridChain(
        tuple(math.degrees(lat) for lat, _ in points),
        tuple(math.degrees(lon) for _, lon in points),
        heights,
        predictor,
    )


def _required(value: int | None) -> int:
    if value is None:
        raise SsrzError("missing value in a grid definition")
    return value


def reconstruct_chain(
    first: tuple[float, float],
    pairs: list[tuple[float, float]],
    flags: list[tuple[bool, ...]],
) -> list[tuple[float, float]]:
    """The points of a chain (radians) from its first point, (dphi, dlambda) of the
    second and (ds, ss) of every further one (clause 12.1.1).

    Every further point is the third corner of a triangle over a baseline, computed in
    the (phi, lambda) plane (eqs. 12.7 to 12.13). The first baseline runs from the
    first to the second point. `flags` are, per point from the third on: use the
    current baseline again (else take the next one of the list), third point left of
    the baseline, add the baseline to the left base point, add the one to the right
    base point; seen from the middle of the baseline towards the third point.
    """
    points = [first]
    if not pairs:
        return points
    dphi, dlam = pairs[0]
    points.append((first[0] + dphi, first[1] + dlam))
    baselines = [(0, 1)]
    current = 0
    use_current, left_side, add_left, add_right = flags or ([], [], [], [])
    for index, (ds, ss) in enumerate(pairs[1:]):
        if not use_current[index]:
            current += 1
        if current >= len(baselines):
            raise SsrzError("grid chain runs out of baselines")
        a, b = baselines[current]
        (phi1, lam1), (phi2, lam2) = points[a], points[b]
        s12 = math.hypot(phi2 - phi1, lam2 - lam1)
        t12 = math.atan2(lam2 - lam1, phi2 - phi1)
        s13 = ss + s12 - ds / 2
        s23 = s13 + ds
        if s12 <= 0 or s13 <= 0:
            raise SsrzError("degenerate grid triangle")
        cos_alpha = (s13 * s13 + s12 * s12 - s23 * s23) / (2 * s13 * s12)
        alpha = math.acos(max(-1.0, min(1.0, cos_alpha)))
        left = left_side[index]
        t3 = t12 - alpha if left else t12 + alpha
        third = len(points)
        points.append((phi1 + s13 * math.cos(t3), lam1 + s13 * math.sin(t3)))
        # Seen from the middle of the baseline towards the new point, the base point on
        # the left is the first one if the point lies left of the baseline.
        left_point, right_point = (a, b) if left else (b, a)
        if add_left[index]:
            baselines.append((left_point, third))
        if add_right[index]:
            baselines.append((third, right_point))
    return points


_METADATA_BLOCKS = {
    (_BLOCK_HIGH_RATE, 1): _read_high_rate,
    (_BLOCK_LOW_RATE, 1): _read_low_rate,
    (_BLOCK_GRID_IONOSPHERE, 1): _read_grid_ionosphere,
    (_BLOCK_GRID_IONOSPHERE, 2): _read_grid_ionosphere,
    (_BLOCK_GRID_TROPOSPHERE, 2): _read_grid_troposphere,
    (_BLOCK_GRID_TROPOSPHERE, 3): _read_grid_troposphere,
    (_BLOCK_REGIONAL_IONOSPHERE, 1): _read_regional_ionosphere,
    (_BLOCK_GLOBAL_VTEC, 2): _read_global_vtec,
    (_BLOCK_GLOBAL_VTEC, 3): _read_global_vtec,
    (_BLOCK_REGIONAL_TROPOSPHERE, 3): _read_regional_troposphere,
    (_BLOCK_SATELLITE_GROUPS, 1): _read_satellite_group_block,
    (_BLOCK_SATELLITE_GROUPS, 2): _read_satellite_group_block,
    (_BLOCK_GRID_DEFINITION, 3): _read_grid_block,
    (_BLOCK_GRID_DEFINITION, 4): _read_grid_block,
}
