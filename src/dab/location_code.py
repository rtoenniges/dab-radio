"""DAB location codes: the places the Emergency Warning System addresses (ETSI TS 104 089,
annexes A and F).

A location code names a spherical rectangle: a zone of 36° × 36° (or one of the two polar
caps), divided again and again into 4 × 4 sub-areas, each division one hexadecimal digit.
Six digits, the most a code has, give about 1 km north-south. An alert names its area as
a set of codes, often with fewer digits; a receiver is inside if its own code begins with
one of them (clause 7.5.4).

The listener enters the code of a place in the presentation format of annex A, twelve
symbols 1 to 8 in three groups (`1255-4467-1251`), with a check sum against typing
errors; in Germany it is known as the ASA code. Nothing but arithmetic connects it with
latitude and longitude: no table and no network is needed either way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The digits of a code at the finest resolution, and of one in the presentation format.
MAX_DIGITS = 6
NORTH_POLAR_ZONE = 0
SOUTH_POLAR_ZONE = 41
ZONES = 42

ZONE_SIZE_DEG = 36.0  # the banded zones, in latitude and longitude
POLAR_CAP_DEG = 18.0  # the polar zones reach this far from the pole
POLAR_RING_DEG = 9.0  # the outer ring of a polar zone, the inner cap the rest
CHECKSUM_MODULUS = 61  # annex A.2
PRESENTATION_SYMBOLS = 12

_PRESENTATION = re.compile(r"[1-8]{12}")


@dataclass(frozen=True, order=True)
class LocationCode:
    """A zone and the hexadecimal digits within it, most significant first; no digits
    stands for the whole zone."""

    zone: int
    digits: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.zone < ZONES:
            raise ValueError(f"No such zone: {self.zone}")
        if len(self.digits) > MAX_DIGITS or any(c not in "0123456789ABCDEF" for c in self.digits):
            raise ValueError(f"Not the digits of a location code: {self.digits!r}")

    def __str__(self) -> str:
        """The notation of the standard, e.g. "Z10:B736BB"."""
        return f"Z{self.zone}:{self.digits}"

    def contains(self, other: LocationCode) -> bool:
        """Whether `other` lies in this area: same zone, and the digits of this code
        begin those of `other` (clause 7.5.4)."""
        return self.zone == other.zone and other.digits.startswith(self.digits)

    @property
    def presentation(self) -> str | None:
        """The code in the presentation format of annex A ("2366-7443-8484"); None
        unless it has all six digits."""
        if len(self.digits) != MAX_DIGITS:
            return None
        value = self.zone << 24 | int(self.digits, 16)
        full = value << 6 | value % CHECKSUM_MODULUS
        symbols = "".join(str(int(octal) + 1) for octal in f"{full:012o}")
        return f"{symbols[0:4]}-{symbols[4:8]}-{symbols[8:12]}"


def parse_presentation(text: str) -> LocationCode | None:
    """The location code a listener entered in the presentation format of annex A;
    blanks and hyphens between the symbols are allowed. None if it is not one: a symbol
    outside 1 to 8, the wrong number, a wrong check sum, or a place that does not exist."""
    symbols = re.sub(r"[\s-]", "", text)
    if not _PRESENTATION.fullmatch(symbols):
        return None
    full = int("".join(str(int(symbol) - 1) for symbol in symbols), 8)
    value, checksum = full >> 6, full & 0x3F
    if value % CHECKSUM_MODULUS != checksum:
        return None
    zone, digits = value >> 24, f"{value & 0xFFFFFF:06X}"
    if zone >= ZONES or (_is_polar(zone) and digits[0] == "0"):
        return None
    return LocationCode(zone, digits)


def location_code(latitude: float, longitude: float, digits: int = MAX_DIGITS) -> LocationCode:
    """The code of the spherical rectangle with this WGS84 position in it (annex F),
    with `digits` digits (1 to 6)."""
    if not 1 <= digits <= MAX_DIGITS:
        raise ValueError(f"A location code has 1 to {MAX_DIGITS} digits, not {digits}")
    southerly, easterly = _extents(latitude, longitude)
    if southerly < POLAR_CAP_DEG or southerly >= 180.0 - POLAR_CAP_DEG:
        zone, full = _polar_code(southerly, easterly)
    else:
        row = int((southerly - POLAR_CAP_DEG) / ZONE_SIZE_DEG)
        column = int(easterly / ZONE_SIZE_DEG)
        zone = 10 * row + column + 1
        sc = int(_frac((southerly - POLAR_CAP_DEG) / ZONE_SIZE_DEG) * 4096)
        ec = int(_frac(easterly / ZONE_SIZE_DEG) * 4096)
        full = _interleave(sc, ec, MAX_DIGITS)
    return LocationCode(zone, f"{full:06X}"[:digits])


def cell(code: LocationCode) -> tuple[float, float, float, float]:
    """The area of a code as (south, north, west, east) in WGS84 degrees; west and east
    between -180 and 180, west greater than east where it crosses 180°."""
    if _is_polar(code.zone):
        top, height, left, width, rest = _polar_cell(code)
    else:
        row, column = divmod(code.zone - 1, 10)
        top, height = POLAR_CAP_DEG + ZONE_SIZE_DEG * row, ZONE_SIZE_DEG
        left, width = ZONE_SIZE_DEG * column, ZONE_SIZE_DEG
        rest = code.digits
    for digit in rest:
        value = int(digit, 16)
        height, width = height / 4, width / 4
        top += (value >> 2) * height
        left += (value & 3) * width
    north = 90.0 - top
    return north - height, north, _longitude(left), _longitude(left + width)


def centre(code: LocationCode) -> tuple[float, float]:
    """The middle of the area of a code as (latitude, longitude); its own code again
    at the same number of digits."""
    south, north, west, east = cell(code)
    if east < west:
        east += 360.0
    return (south + north) / 2, _longitude((west + east) / 2)


def _extents(latitude: float, longitude: float) -> tuple[float, float]:
    """Annex F.2: the southerly extent, 0 at the north pole to 180 at the south pole, and
    the easterly extent, 0 to under 360 eastwards from Greenwich."""
    southerly = 90.0 - min(max(latitude, -90.0), 90.0)
    easterly = longitude % 360.0
    if easterly >= 360.0:  # a tiny negative longitude rounds up to 360
        easterly = 0.0
    return southerly, easterly


def _polar_code(southerly: float, easterly: float) -> tuple[int, int]:
    """Annex F.5: the zone and the 24 bits of the digits in a polar zone, whose first
    digit names a sector of the inner cap (11 to 15) or of the outer ring (1 to 10)."""
    north = southerly < POLAR_CAP_DEG
    # From the edge of the zone nearer the north pole: 0 to 18. The first half is the
    # inner cap in the north and the outer ring in the south.
    offset = southerly if north else southerly - (180.0 - POLAR_CAP_DEG)
    first_half = offset < POLAR_RING_DEG
    inner = first_half if north else not first_half
    sector = 72.0 if inner else ZONE_SIZE_DEG
    first = int(easterly / sector) + (11 if inner else 1)
    sc = int(_frac(offset / POLAR_RING_DEG) * 1024)
    ec = int(_frac(easterly / sector) * 1024)
    zone = NORTH_POLAR_ZONE if north else SOUTH_POLAR_ZONE
    return zone, first << 20 | _interleave(sc, ec, MAX_DIGITS - 1)


def _polar_cell(code: LocationCode) -> tuple[float, float, float, float, str]:
    """The rectangle (in southerly and easterly extents) a polar code's first digit
    names, and the digits that divide it further; digit 0 or none is the whole zone."""
    north = code.zone == NORTH_POLAR_ZONE
    cap_top = 0.0 if north else 180.0 - POLAR_CAP_DEG
    first = int(code.digits[0], 16) if code.digits else 0
    if first == 0:
        return cap_top, POLAR_CAP_DEG, 0.0, 360.0, ""
    inner = first >= 11
    if inner:
        left, width = (first - 11) * 72.0, 72.0
        top = cap_top if north else cap_top + POLAR_RING_DEG
    else:
        left, width = (first - 1) * ZONE_SIZE_DEG, ZONE_SIZE_DEG
        top = cap_top + POLAR_RING_DEG if north else cap_top
    return top, POLAR_RING_DEG, left, width, code.digits[1:]


def _interleave(sc: int, ec: int, digits: int) -> int:
    """The digits of a code from the southerly and easterly codes: each digit takes two
    bits of either, the southerly ones first (annex F.4)."""
    value = 0
    for shift in range(2 * (digits - 1), -1, -2):
        value = value << 4 | ((sc >> shift) & 3) << 2 | ((ec >> shift) & 3)
    return value


def _is_polar(zone: int) -> bool:
    return zone in (NORTH_POLAR_ZONE, SOUTH_POLAR_ZONE)


def _frac(value: float) -> float:
    return value % 1.0


def _longitude(easterly: float) -> float:
    """An easterly extent as a WGS84 longitude, -180 to 180."""
    easterly %= 360.0
    return easterly - 360.0 if easterly > 180.0 else easterly
