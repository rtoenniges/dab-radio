"""The binary encoding of TPEG2 (ISO/TS 21219-4 and -5): the primitive data types and the
component tree that every TPEG2 application message is built from.

A TPEG2 message is a tree of components. Each component starts with its identifier
(one byte, meaningful only within its parent), the length of the rest of the component
and the length of its attribute block, both as `IntUnLoMB`::

    id | lengthComp | lengthAttr | attributes ... | child components ...

The attribute block holds the component's own fields; the child components follow it
up to the end given by `lengthComp`. Because both lengths are given, a receiver can skip
any component or attribute it does not know, and `parse_component` builds the tree
without knowing the application.

Inside an attribute block, optional fields are announced by a selector (`BitArray`):
bytes whose most significant bit says that another byte follows, the other seven bits
flag the optional fields in order, the first one in bit 6 (0x40). Booleans are flags of
the same selector. Integers are big-endian; `IntUnLoMB` and `IntSiLoMB` carry seven
bits per byte, the most significant group first, with the most significant bit as
continuation flag. The interpretation follows the TPEG2 evaluation kit of TISA (pyTPEG,
BSD licence; no code of it is used) and was checked against the ARD TPEG service on
channel 7A: every length and CRC of 1 159 TEC and 1 606 TFP messages works out.
"""

from __future__ import annotations

from dataclasses import dataclass

# Coordinates in TPEG are 24-bit integers: a full circle of 360 degrees in 2^24 steps
# (about 2.1e-5 degrees, 2.4 m in latitude).
_COORDINATE_STEPS = 1 << 24


class TpegDecodeError(ValueError):
    """The data end before a field, or a field holds a value the format rules out."""


class Selector:
    """A decoded `BitArray`: which optional fields (and boolean flags) follow."""

    def __init__(self, flags: tuple[bool, ...]) -> None:
        self._flags = flags

    def __getitem__(self, index: int) -> bool:
        """Flag `index` (0: bit 6 of the first byte); beyond the end: absent."""
        return index < len(self._flags) and self._flags[index]

    def highest(self) -> int:
        """The index of the last set flag, -1 if none is set."""
        return max((i for i, flag in enumerate(self._flags) if flag), default=-1)


class TpegReader:
    """Reads the TPEG data types from a byte string, front to back.

    Every read raises `TpegDecodeError` if the data end first, so that a decoder can
    read a structure field by field and treat a short structure as one error.
    """

    def __init__(self, data: bytes, start: int = 0, end: int | None = None) -> None:
        self._data = data
        self.position = start
        self._end = len(data) if end is None else end

    @property
    def remaining(self) -> int:
        return self._end - self.position

    def at_end(self) -> bool:
        return self.position >= self._end

    def _take(self, count: int) -> bytes:
        if count < 0 or self.position + count > self._end:
            raise TpegDecodeError(
                f"{count} bytes wanted at offset {self.position}, {self.remaining} left"
            )
        value = self._data[self.position : self.position + count]
        self.position += count
        return value

    def bytes(self, count: int) -> bytes:
        return self._take(count)

    def rest(self) -> bytes:
        return self._take(self.remaining)

    def limited(self, count: int) -> TpegReader:
        """A reader over the next `count` bytes, which this one skips."""
        if count < 0 or count > self.remaining:
            raise TpegDecodeError(f"{count} bytes wanted at offset {self.position}")
        inner = TpegReader(self._data, self.position, self.position + count)
        self.position += count
        return inner

    def u8(self) -> int:
        """IntUnTi."""
        return self._take(1)[0]

    def u16(self) -> int:
        """IntUnLi."""
        return int.from_bytes(self._take(2), "big")

    def s16(self) -> int:
        """IntSiLi."""
        return int.from_bytes(self._take(2), "big", signed=True)

    def s24(self) -> int:
        """IntSi24."""
        return int.from_bytes(self._take(3), "big", signed=True)

    def u32(self) -> int:
        """IntUnLo; also DateTime (seconds since 1970-01-01 00:00 UTC)."""
        return int.from_bytes(self._take(4), "big")

    def multibyte(self) -> int:
        """IntUnLoMB: seven bits per byte, most significant group first."""
        value = 0
        for _ in range(5):
            byte = self.u8()
            value = (value << 7) | (byte & 0x7F)
            if not byte & 0x80:
                return value
        raise TpegDecodeError("IntUnLoMB longer than five bytes")

    def signed_multibyte(self) -> int:
        """IntSiLoMB: like IntUnLoMB, bit 6 of the first byte is the sign (two's
        complement over all groups)."""
        start = self.position
        value = self.multibyte()
        bits = 7 * (self.position - start)
        if self._data[start] & 0x40:
            value -= 1 << bits
        return value

    def selector(self) -> Selector:
        """BitArray: continuation in bit 7, the flags in bits 6 to 0 of each byte."""
        flags: list[bool] = []
        for _ in range(8):
            byte = self.u8()
            flags.extend(bool(byte & (0x40 >> bit)) for bit in range(7))
            if not byte & 0x80:
                return Selector(tuple(flags))
        raise TpegDecodeError("BitArray longer than eight bytes")

    def short_string(self) -> str:
        """ShortString: one length byte, UTF-8."""
        return self._take(self.u8()).decode("utf-8", errors="replace")

    def localised_short_string(self) -> tuple[int, str]:
        """LocalisedShortString: language (typ001), then a ShortString."""
        language = self.u8()
        return language, self.short_string()

    def absolute_coordinate(self) -> tuple[float, float]:
        """WGS 84 longitude and latitude as two IntSi24, rounded to five decimals as
        the TPEG location references define them."""
        return coordinate_degrees(self.s24()), coordinate_degrees(self.s24())


def coordinate_degrees(value: int) -> float:
    """Degrees of a 24-bit TPEG coordinate (the encoder rounded half away from zero)."""
    if value == 0:
        return 0.0
    half = 0.5 if value > 0 else -0.5
    return round((value - half) * 360.0 / _COORDINATE_STEPS, 5)


@dataclass(frozen=True)
class Component:
    """One component of a TPEG2 message, with its attribute block still as bytes."""

    cid: int
    attributes: bytes
    children: tuple[Component, ...]
    size: int  # bytes of the whole component, identifier and lengths included

    def child(self, cid: int) -> Component | None:
        return next((c for c in self.children if c.cid == cid), None)

    def all(self, cid: int) -> list[Component]:
        return [c for c in self.children if c.cid == cid]


def parse_component(reader: TpegReader) -> Component:
    """Reads one component and, recursively, its children."""
    start = reader.position
    cid = reader.u8()
    length = reader.multibyte()
    if length > reader.remaining:
        raise TpegDecodeError(f"component {cid}: {length} bytes, only {reader.remaining} left")
    inner = reader.limited(length)
    attributes = inner.bytes(inner.multibyte())
    children = []
    while not inner.at_end():
        children.append(parse_component(inner))
    return Component(cid, attributes, tuple(children), reader.position - start)


def parse_components(data: bytes) -> tuple[Component, ...]:
    """The components that fill `data` completely, one after the other."""
    reader = TpegReader(data)
    components = []
    while not reader.at_end():
        components.append(parse_component(reader))
    return tuple(components)
