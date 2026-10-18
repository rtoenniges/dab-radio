"""The TPEG2 applications ARD TPEG carries: traffic events (TEC, ISO/TS 21219-15) and
traffic flow (TFP, ISO/TS 21219-18), with their message management (MMC, ISO/TS
21219-6) and location references (LRC, ISO/TS 21219-7: TMC and OpenLR).

Every message is a component tree (see `dab.tpeg_binary`) with the identifier 0 at its
root and three kinds of children:

* 1, message management: message ID, version, expiry time, cancel flag,
* the application data: for TEC the event (3) with its causes and advice, for TFP a
  flow matrix (6) with the level of service along the location,
* 2, the location referencing container, which may describe the same place in several
  methods at once. ARD sends TMC (2: location code of the ALERT-C table, direction and
  extent), OpenLR (8: WGS 84 coordinates of reference points along the road, with the
  road geometry as a shape) and a method 254 that no public specification describes.

TMC codes can only be resolved with the location table of the country (for Germany the
LCL of the BASt, licensed on request and not to be published); OpenLR carries its own
coordinates. Field order and component identifiers follow the TPEG2 evaluation kit of
TISA, the TPEG2 protobuf definitions it publishes and, for MMC, TMC and OpenLR, the Java
library tpeg-item (Apache 2.0); no code of them is used. With this field order the
attribute blocks of every message of ARD TPEG are used up exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

from dab.tpeg_binary import Component, Selector, TpegDecodeError, TpegReader, parse_component

# Children of a message (TEC and TFP).
COMPONENT_MESSAGE = 0
COMPONENT_MANAGEMENT = 1
COMPONENT_LOCATION = 2
# TEC (TISA SP20012, TEC 3.2 as announced by the SNI of ARD TPEG).
COMPONENT_EVENT = 3
COMPONENT_DIRECT_CAUSE = 4
COMPONENT_LINKED_CAUSE = 5
COMPONENT_ADVICE = 6
COMPONENT_VEHICLE_RESTRICTION = 7
COMPONENT_DIVERSION = 8
COMPONENT_SPEED_LIMIT = 11
# TFP.
COMPONENT_FLOW_MATRIX = 6
COMPONENT_FLOW_VECTOR = 7

# Location referencing methods, children of the location container.
LOCATION_TMC = 2
LOCATION_OPENLR = 8
# OpenLR: the location reference inside the OpenLR component, and its parts.
OPENLR_LINEAR = 0
OPENLR_POINT_ALONG_LINE = 2
OPENLR_LINE_PROPERTIES = 9
OPENLR_PATH_PROPERTIES = 10
OPENLR_SHAPE = 12

# TEC codes the display has to recognise (tec001, tec002).
EFFECT_UNKNOWN = 1
EFFECT_NO_TRAFFIC_FLOW = 7

# Coordinates: (longitude, latitude) in degrees, WGS 84.
Coordinate = tuple[float, float]

# Relative coordinates of OpenLR count in steps of 1e-5 degrees.
_RELATIVE_COORDINATE_STEP = 1e-5
# Precise TMC distances count in steps of 100 m.
_TMC_DISTANCE_STEP = 100


@dataclass(frozen=True)
class MessageManagement:
    """The message management container (MMC): identity and lifetime of a message."""

    message_id: int
    version: int
    expiry: int  # POSIX seconds (UTC); after it the message must no longer be shown
    cancel: bool  # the message is withdrawn, the rest of it is empty
    generation_time: int | None = None
    priority: int | None = None  # typ007: 1 low, 2 medium, 3 high


@dataclass(frozen=True)
class TmcLocation:
    """A location of the ALERT-C table (ISO 14819-3): code, direction, extent."""

    location: int
    country: int  # country code of the table (Germany: 13 = 0xD)
    table: int
    positive_direction: bool  # towards the successors in the location list
    both_directions: bool
    extent: int  # locations beyond the primary one, in `positive_direction`
    extended_country_code: int | None = None
    table_version: tuple[int, int] | None = None  # major, minor
    distance_accuracy: int | None = None  # 0: 100 m, 1: 500 m, 2: 1 km, 3: more
    hazard_distance: int | None = None  # metres upstream of the primary location
    problem_length: int | None = None  # metres


@dataclass(frozen=True)
class ReferencePoint:
    """A location reference point of OpenLR: where it is and what road leaves it."""

    coordinate: Coordinate
    frc: int  # functional road class, 0: most important
    fow: int  # form of way (olr002)
    bearing: float  # degrees clockwise from north, in steps of 360/256
    lowest_frc: int | None = None  # on the path to the next point
    distance: int | None = None  # metres along the road to the next point


@dataclass(frozen=True)
class OpenLrLocation:
    """An OpenLR location reference: a line along the road network or a point on it."""

    kind: int  # OPENLR_LINEAR or OPENLR_POINT_ALONG_LINE
    version: tuple[int, int]
    points: tuple[ReferencePoint, ...]
    positive_offset: int = 0  # metres from the first point to the start
    negative_offset: int = 0  # metres from the end to the last point
    side_of_road: int | None = None  # olr004, point along line only
    orientation: int | None = None  # olr003, point along line only
    shape: tuple[Coordinate, ...] = ()  # road geometry, if sent

    def path(self) -> tuple[Coordinate, ...]:
        """The line to draw: the shape if there is one, else the reference points."""
        if len(self.shape) >= 2:
            return self.shape
        return tuple(point.coordinate for point in self.points)

    def length(self) -> int | None:
        """Metres between start and end of a linear location."""
        distances = [p.distance for p in self.points[:-1]]
        if self.kind != OPENLR_LINEAR or None in distances:
            return None
        return max(sum(distances) - self.positive_offset - self.negative_offset, 0)


@dataclass(frozen=True)
class Location:
    """The location referencing container, with the methods decoded here."""

    methods: tuple[int, ...]  # identifiers of every method sent, in order
    tmc: TmcLocation | None = None
    openlr: OpenLrLocation | None = None


@dataclass(frozen=True)
class TecCause:
    """A direct cause (the reason, with details) or a link to a message about it."""

    main: int | None  # tec002; None for a linked cause
    sub: int | None = None  # tec1xx, the table chosen by `main`
    warning_level: int | None = None  # tec003
    unverified: bool = False
    length_affected: int | None = None  # metres
    lane_restriction: int | None = None  # tec004
    lanes: int | None = None
    free_text: tuple[str, ...] = ()
    offset: int | None = None  # metres from the start of the cause to the end of the location
    linked_message: int | None = None


@dataclass(frozen=True)
class TecAdvice:
    code: int | None  # tec005
    sub: int | None = None  # tec2xx, the table chosen by `code`
    free_text: tuple[str, ...] = ()


@dataclass(frozen=True)
class VehicleRestriction:
    """For which vehicles the event (or an advice) applies."""

    vehicle_type: int | None  # tec009; None: all
    restrictions: tuple[tuple[int, int | None], ...] = ()  # tec007 and its value


@dataclass(frozen=True)
class TecMessage:
    """A traffic event: what happens to the traffic flow, why, and where."""

    scid: int
    management: MessageManagement
    component: Component  # the whole message, for the display of its structure
    effect: int | None = None  # tec001; None in a cancellation
    start: int | None = None  # POSIX seconds
    stop: int | None = None
    tendency: int | None = None  # tec006
    length_affected: int | None = None  # metres
    average_speed: int | None = None  # m/s
    delay: int | None = None  # minutes
    causes: tuple[TecCause, ...] = ()
    advice: tuple[TecAdvice, ...] = ()
    restrictions: tuple[VehicleRestriction, ...] = ()
    speed_limits: tuple[int, ...] = ()  # temporary limits, km/h
    diversion_segments: int = 0
    location: Location | None = None


@dataclass(frozen=True)
class FlowSection:
    """The traffic flow on one stretch of a TFP location."""

    offset: int  # end of the section, in units of the spatial resolution
    level_of_service: int | None = None  # tfp003
    average_speed: int | None = None  # km/h
    free_flow_travel_time: int | None = None
    delay: int | None = None
    spatial_resolution: int | None = None  # tfp004, overrides the vector's
    section_type: int | None = None  # tfp007


@dataclass(frozen=True)
class FlowVector:
    time_offset: int  # minutes after the start time (0: now; later: prediction)
    sections: tuple[FlowSection, ...]
    spatial_resolution: int | None = None


@dataclass(frozen=True)
class TfpMessage:
    """Traffic flow along a location: level of service and speed per section."""

    scid: int
    management: MessageManagement
    component: Component
    start: int | None = None  # POSIX seconds
    duration: int | None = None
    spatial_resolution: int | None = None  # tfp004: 0 counts in TMC locations
    vectors: tuple[FlowVector, ...] = ()
    location: Location | None = None


def _attributes(component: Component) -> TpegReader:
    return TpegReader(component.attributes)


def _done(reader: TpegReader, what: str) -> None:
    """An attribute block has to be used up exactly: leftover bytes mean that a field
    was read as something it is not."""
    if not reader.at_end():
        raise TpegDecodeError(f"{what}: {reader.remaining} bytes of attributes left over")


def decode_management(component: Component) -> MessageManagement:
    """MMC (ISO/TS 21219-6): messageID, versionID, messageExpiryTime, then a selector
    with the cancel flag and the optional generation time and priority."""
    reader = _attributes(component)
    message_id = reader.multibyte()
    version = reader.u8()
    expiry = reader.u32()
    selector = reader.selector()
    generation_time = reader.u32() if selector[1] else None
    priority = reader.u8() if selector[2] else None
    _done(reader, "MMC")
    return MessageManagement(message_id, version, expiry, selector[0], generation_time, priority)


# --- Location references ------------------------------------------------------------


def _tmc_location(component: Component) -> TmcLocation:
    reader = _attributes(component)
    location = reader.u16()
    country = reader.u8()
    table = reader.u8()
    selector = reader.selector()
    extent = reader.u8() if selector[2] else 0
    ecc = reader.u8() if selector[3] else None
    version = None
    if selector[4]:
        start = reader.position
        value = reader.multibyte()
        # One byte: 4 bits major, 3 bits minor; two bytes: 7 bits each.
        version = (
            (value >> 3, value & 0x07)
            if reader.position - start == 1
            else (value >> 7, value & 0x7F)
        )
    accuracy = hazard = problem = None
    if selector[5]:
        precise = reader.selector()
        accuracy = reader.u8() if precise[0] else None
        near = reader.u8() if precise[1] else None
        far = reader.u16() if precise[2] else None
        short = reader.u8() if precise[3] else None
        long = reader.u16() if precise[4] else None
        hazard = _hundreds(far if far is not None else near)
        problem = _hundreds(long if long is not None else short)
    _done(reader, "TMC")
    return TmcLocation(
        location,
        country,
        table,
        selector[0],
        selector[1],
        extent,
        ecc,
        version,
        accuracy,
        hazard,
        problem,
    )


def _hundreds(value: int | None) -> int | None:
    return None if value is None else value * _TMC_DISTANCE_STEP


def _embedded(reader: TpegReader, cid: int) -> TpegReader:
    """A component inside an attribute block (line and path properties of OpenLR):
    returns a reader over its attributes."""
    component = parse_component(reader)
    if component.cid != cid:
        raise TpegDecodeError(f"component {cid} expected, {component.cid} found")
    return _attributes(component)


def _line_properties(reader: TpegReader) -> tuple[int, int, float]:
    inner = _embedded(reader, OPENLR_LINE_PROPERTIES)
    frc, fow, bearing = inner.u8(), inner.u8(), inner.u8()
    sides = inner.selector()  # bearings of side roads left and right
    if sides[0]:
        inner.u8()
    if sides[1]:
        inner.u8()
    _done(inner, "OpenLR line properties")
    return frc, fow, bearing * 360.0 / 256


def _path_properties(reader: TpegReader) -> tuple[int, int]:
    inner = _embedded(reader, OPENLR_PATH_PROPERTIES)
    lowest_frc, distance = inner.u8(), inner.multibyte()
    inner.selector()  # against the driving direction
    _done(inner, "OpenLR path properties")
    return lowest_frc, distance


def _absolute(reader: TpegReader) -> Coordinate:
    coordinate = reader.absolute_coordinate()
    if reader.selector()[0]:
        reader.signed_multibyte()  # altitude
    return coordinate


def _relative(reader: TpegReader) -> tuple[int, int]:
    longitude, latitude = reader.s16(), reader.s16()
    if reader.selector()[0]:
        reader.signed_multibyte()  # altitude
    return longitude, latitude


def _shift(coordinate: Coordinate, step: tuple[int, int]) -> Coordinate:
    return (
        round(coordinate[0] + step[0] * _RELATIVE_COORDINATE_STEP, 5),
        round(coordinate[1] + step[1] * _RELATIVE_COORDINATE_STEP, 5),
    )


def _openlr_location(component: Component) -> OpenLrLocation | None:
    """The OpenLR component (ISO/TS 21219-22): version, then one location reference."""
    reader = _attributes(component)
    version_byte = reader.u8()
    version = (version_byte >> 4, version_byte & 0x0F)
    reference = next(
        (c for c in component.children if c.cid in (OPENLR_LINEAR, OPENLR_POINT_ALONG_LINE)), None
    )
    if reference is None:
        return None
    reader = _attributes(reference)
    first = _absolute(reader)
    first_line = _line_properties(reader)
    first_path = _path_properties(reader)
    last_step = _relative(reader)
    last_line = _line_properties(reader)
    points = [ReferencePoint(first, *first_line, *first_path)]
    positive = negative = 0
    side = orientation = None
    if reference.cid == OPENLR_LINEAR:
        selector = reader.selector()
        intermediates = []
        if selector[0]:
            for _ in range(reader.multibyte()):
                intermediates.append(
                    (_relative(reader), _line_properties(reader), _path_properties(reader))
                )
        positive = reader.multibyte() if selector[1] else 0
        negative = reader.multibyte() if selector[2] else 0
        # Each point is relative to the one before it in the sequence; the last point
        # comes before the intermediate ones in the data.
        for step, line, path in intermediates:
            points.append(ReferencePoint(_shift(points[-1].coordinate, step), *line, *path))
    else:
        side, orientation = reader.u8(), reader.u8()
        positive = reader.multibyte() if reader.selector()[0] else 0
    points.append(ReferencePoint(_shift(points[-1].coordinate, last_step), *last_line))
    _done(reader, "OpenLR")
    shape: tuple[Coordinate, ...] = ()
    shape_component = reference.child(OPENLR_SHAPE)
    if shape_component is not None:
        reader = _attributes(shape_component)
        shape = tuple(_absolute(reader) for _ in range(reader.multibyte()))
        _done(reader, "OpenLR shape")
    return OpenLrLocation(
        reference.cid, version, tuple(points), positive, negative, side, orientation, shape
    )


def decode_location(component: Component) -> Location:
    """The location referencing container: every method is listed, TMC and OpenLR are
    decoded. A method that does not decode is left out rather than losing the message."""
    tmc = openlr = None
    for method in component.children:
        try:
            if method.cid == LOCATION_TMC and tmc is None:
                tmc = _tmc_location(method)
            elif method.cid == LOCATION_OPENLR and openlr is None:
                openlr = _openlr_location(method)
        except TpegDecodeError:
            pass
    return Location(tuple(method.cid for method in component.children), tmc, openlr)


# --- TEC ----------------------------------------------------------------------------


def _free_text(reader: TpegReader) -> tuple[str, ...]:
    return tuple(reader.localised_short_string()[1] for _ in range(reader.multibyte()))


def _direct_cause(component: Component) -> TecCause:
    reader = _attributes(component)
    main, warning_level = reader.u8(), reader.u8()
    selector = reader.selector()
    sub = reader.u8() if selector[1] else None
    length = reader.multibyte() if selector[2] else None
    lane_restriction = reader.u8() if selector[3] else None
    lanes = reader.u8() if selector[4] else None
    free_text = _free_text(reader) if selector[5] else ()
    offset = reader.multibyte() if selector[6] else None
    if selector.highest() > 6:
        reader.rest()  # the lanes affected (TEC 3.3 and later): not shown
    _done(reader, "TEC direct cause")
    return TecCause(
        main,
        sub,
        warning_level,
        selector[0],
        length,
        lane_restriction,
        lanes,
        free_text,
        offset,
    )


def _linked_cause(component: Component) -> TecCause:
    reader = _attributes(component)
    linked = reader.multibyte()
    reader.rest()  # content ID and service of the linked message
    return TecCause(None, linked_message=linked)


def _advice(component: Component) -> TecAdvice:
    reader = _attributes(component)
    selector = reader.selector()
    code = reader.u8() if selector[0] else None
    sub = reader.u8() if selector[1] else None
    free_text = _free_text(reader) if selector[2] else ()
    _done(reader, "TEC advice")
    return TecAdvice(code, sub, free_text)


def _vehicle_restriction(component: Component) -> VehicleRestriction:
    reader = _attributes(component)
    selector = reader.selector()
    vehicle_type = reader.u8() if selector[0] else None
    restrictions = []
    if selector[1]:
        for _ in range(reader.multibyte()):
            kind = reader.u8()
            details = reader.selector()
            value = reader.multibyte() if details[0] else None
            if details[1]:
                parse_component(reader)  # destination area: not shown
            restrictions.append((kind, value))
    _done(reader, "TEC vehicle restriction")
    return VehicleRestriction(vehicle_type, tuple(restrictions))


def _speed_limits(component: Component) -> tuple[int, ...]:
    reader = _attributes(component)
    limits = []
    for _ in range(reader.multibyte()):
        limits.append(reader.u8())
        details = reader.selector()
        if details[0]:
            reader.u8()  # limit when wet
        if details[1]:
            reader.multibyte()  # length of the section
    reader.rest()  # unit (km/h unless flagged mph) and offset
    return tuple(limits)


def _diversion_segments(component: Component) -> int:
    return TpegReader(component.attributes).multibyte()


def _event(component: Component) -> dict:
    reader = _attributes(component)
    effect = reader.u8()
    selector: Selector = reader.selector()
    fields = {
        "effect": effect,
        "start": reader.u32() if selector[0] else None,
        "stop": reader.u32() if selector[1] else None,
        "tendency": reader.u8() if selector[2] else None,
        "length_affected": reader.multibyte() if selector[3] else None,
        "average_speed": reader.u8() if selector[4] else None,
        "delay": reader.multibyte() if selector[5] else None,
    }
    if selector[6]:
        reader.u8()  # segment speed limit, for routing only
    if selector[7]:
        reader.u8()  # expected speed
    if selector.highest() > 7:
        reader.rest()  # later additions (junction closure)
    _done(reader, "TEC event")
    causes, advice, restrictions, limits, diversions = [], [], [], [], 0
    for child in component.children:
        if child.cid == COMPONENT_DIRECT_CAUSE:
            causes.append(_direct_cause(child))
        elif child.cid == COMPONENT_LINKED_CAUSE:
            causes.append(_linked_cause(child))
        elif child.cid == COMPONENT_ADVICE:
            advice.append(_advice(child))
        elif child.cid == COMPONENT_VEHICLE_RESTRICTION:
            restrictions.append(_vehicle_restriction(child))
        elif child.cid == COMPONENT_SPEED_LIMIT:
            limits.extend(_speed_limits(child))
        elif child.cid == COMPONENT_DIVERSION:
            diversions += _diversion_segments(child)
    fields.update(
        causes=tuple(causes),
        advice=tuple(advice),
        restrictions=tuple(restrictions),
        speed_limits=tuple(limits),
        diversion_segments=diversions,
    )
    return fields


def _message_parts(component: Component) -> tuple[MessageManagement, Location | None]:
    if component.cid != COMPONENT_MESSAGE:
        raise TpegDecodeError(f"message component expected, {component.cid} found")
    management = component.child(COMPONENT_MANAGEMENT)
    if management is None:
        raise TpegDecodeError("message without management container")
    location = component.child(COMPONENT_LOCATION)
    return decode_management(management), None if location is None else decode_location(location)


def decode_tec_message(scid: int, component: Component) -> TecMessage:
    """A TEC message from its component tree.

    Raises:
        TpegDecodeError: if the management container or the event does not decode.
    """
    management, location = _message_parts(component)
    event = component.child(COMPONENT_EVENT)
    fields = {} if event is None else _event(event)
    return TecMessage(scid, management, component, location=location, **fields)


# --- TFP ----------------------------------------------------------------------------


def _flow_section(reader: TpegReader) -> FlowSection:
    offset = reader.multibyte()
    status = reader.selector()
    level = reader.u8() if status[0] else None
    speed = reader.u8() if status[1] else None
    free_flow = reader.multibyte() if status[2] else None
    delay = reader.multibyte() if status[3] else None
    if status.highest() > 3:
        raise TpegDecodeError("TFP status with extensions")
    selector = reader.selector()
    resolution = reader.u8() if selector[0] else None
    section_type = reader.u8() if selector[1] else None
    if selector.highest() > 1:
        # Restrictions, statistics and causes per section: not decoded.
        raise TpegDecodeError("TFP section with restrictions, statistics or cause")
    return FlowSection(offset, level, speed, free_flow, delay, resolution, section_type)


def _flow_vector(component: Component) -> FlowVector:
    reader = _attributes(component)
    time_offset = reader.multibyte()
    sections = tuple(_flow_section(reader) for _ in range(reader.multibyte()))
    resolution = reader.u8() if reader.selector()[0] else None
    _done(reader, "TFP flow vector")
    return FlowVector(time_offset, sections, resolution)


def decode_tfp_message(scid: int, component: Component) -> TfpMessage:
    """A TFP message from its component tree (flow matrix only; flow status and flow
    polygons are not sent by ARD).

    Raises:
        TpegDecodeError: if the management container or the flow matrix does not decode.
    """
    management, location = _message_parts(component)
    matrix = component.child(COMPONENT_FLOW_MATRIX)
    if matrix is None:
        return TfpMessage(scid, management, component, location=location)
    reader = _attributes(matrix)
    start = reader.u32()
    duration = reader.multibyte() if reader.selector()[0] else None
    resolution = reader.u8()
    _done(reader, "TFP flow matrix")
    vectors = tuple(_flow_vector(c) for c in matrix.all(COMPONENT_FLOW_VECTOR))
    return TfpMessage(scid, management, component, start, duration, resolution, vectors, location)
