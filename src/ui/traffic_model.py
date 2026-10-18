"""What ARD TPEG broadcasts, for a surface, without a widget: the traffic events (TEC)
by their distance from the listener, the traffic flow (TFP) on its stretches, and the
state of the service (SNI, statistics).

Where a message lies comes from what DAB brings: the OpenLR line, and the TMC location
resolved by the TMC location table (`dab.tmc_locations`) if it was imported; that also
names the road and its junctions. Without the table a message has no name, only what
OpenLR says of its road, and the traffic flow (TMC only) has no place.

The desktop's "Traffic" tab (`ui.traffic_tab`) and the touch surface's "Traffic" menu
(`ui.touch.traffic_view`) draw from one `TrafficModel`. The texts both show of a message
are here too (`describe_tec` and the helpers before it).

The messages outlive a channel change (an announcement from another ensemble, a look at
another channel): every message carries its expiry time, so the store cleans itself. Only
a withdrawal sent while the receiver was away is missed; so once the service is received
again, whatever it has not repeated within one cycle of its carousel is dropped.
"""

from __future__ import annotations

import html
import math
import time
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from dab.data_services import USER_APPLICATION_TPEG, DataComponent
from dab.gnss import Observer
from dab.road_network import RoadNetwork
from dab.road_network import load_if_present as load_roads_in
from dab.timecode import BroadcastTime, ReferenceClock
from dab.tmc_locations import TmcLocationTable, TmcPoint
from dab.tmc_locations import load_if_present as load_locations_in
from dab.tpeg import (
    AID_TEC,
    ServiceId,
    ServiceInformation,
    StoredMessage,
    TpegFrame,
    TpegMessageStore,
    TpegStatistics,
)
from dab.tpeg_messages import (
    EFFECT_NO_TRAFFIC_FLOW,
    LOCATION_OPENLR,
    LOCATION_TMC,
    OPENLR_LINEAR,
    Coordinate,
    Location,
    TecMessage,
    TfpMessage,
)
from dab.tpeg_tables import (
    LANE_RESTRICTIONS,
    TENDENCIES,
    VEHICLE_TYPES,
    WARNING_LEVELS,
    advice_name,
    cause_name,
    decimal,
    effect_name,
    form_of_way_name,
    location_method_name,
    restriction_text,
)
from dsp.pipeline import DataComponentStatus
from i18n import format_date, format_day_month, tr, tr_noop, tr_plural
from ui.receiver_controller import ReceiverController

Key = tuple[int, int]  # SCId, message ID

# Severity of a traffic event, drawn in the status colours (critical, serious, warning)
# plus blue for messages without a traffic flow effect (hazards, lane closures). Warning
# and serious are close for colour-blind readers: every class has a symbol of its own
# and a legend entry. The names are translated where they are shown.
SEVERITIES = (
    (tr_noop("Closure"), (0xD0, 0x3B, 0x3B), "s"),
    (tr_noop("Stationary traffic"), (0xEC, 0x83, 0x5A), "t"),
    (tr_noop("Queuing, slow"), (0xFA, 0xB2, 0x19), "d"),
    (tr_noop("Notice"), (0x39, 0x87, 0xE5), "o"),
)
# The symbols as characters, for lists and legends beside the map (pyqtgraph's "t" is
# a triangle pointing down).
SEVERITY_MARKS = ("■", "▼", "◆", "●")

_EARTH_RADIUS_KM = 6371.0
# Translators: the points of the compass, for the direction of a traffic message from
# the listener ("12 km SE").
_COMPASS = (
    tr_noop("N"),
    tr_noop("NE"),
    tr_noop("E"),
    tr_noop("SE"),
    tr_noop("S"),
    tr_noop("SW"),
    tr_noop("W"),
    tr_noop("NW"),
)
# Translators: the priority of a traffic message.
_PRIORITIES = {1: tr_noop("low"), 2: tr_noop("medium"), 3: tr_noop("high")}
# No frame for this long and the service counts as silent.
SILENT_SECONDS = 30.0
# Received again after an absence (another channel, or silence), the service has repeated
# everything still on air within this time: all 655 TEC messages the SNI announced came
# within 240 s, the TFP messages twice. What was held from before and did not come again
# is dropped then, since its withdrawal may have been missed.
CYCLE_SECONDS = 300.0

# Levels of service of the traffic flow (tfp003 without its tendency) with their colours,
# in the scale of the traffic events: yellow for slow, orange for queues, red for closed.
# The names are those of `dab.tpeg_tables.level_of_service_name`, translated where shown.
FLOW_LEVELS = {
    0: (tr_noop("unknown"), (0x89, 0x87, 0x81)),
    1: (tr_noop("free flow"), (0x0C, 0xA3, 0x0C)),
    2: (tr_noop("heavy"), (0xF2, 0xD2, 0x6B)),
    3: (tr_noop("slow"), (0xFA, 0xB2, 0x19)),
    4: (tr_noop("queuing"), (0xFA, 0xB2, 0x19)),
    5: (tr_noop("stationary"), (0xEC, 0x83, 0x5A)),
    6: (tr_noop("closed"), (0xD0, 0x3B, 0x3B)),
}
# The flow is disturbed from this level on; the map draws only those stretches.
DISTURBED = 2


def load_roads(data_directory: Path | None) -> RoadNetwork | None:
    return None if data_directory is None else load_roads_in(data_directory)


def load_locations(data_directory: Path | None) -> TmcLocationTable | None:
    return None if data_directory is None else load_locations_in(data_directory)


def tpeg_component(components: list[DataComponent]) -> DataComponent | None:
    """The data component carrying TPEG, if the ensemble has one."""
    return next(
        (c for c in components if USER_APPLICATION_TPEG in c.app_types and c.decodable), None
    )


def local(timestamp: float | None, spec: str = "%H:%M") -> str:
    """The time of day of `timestamp` by `spec`; "-" without one."""
    return "-" if timestamp is None else time.strftime(spec, time.localtime(timestamp))


def local_date(timestamp: float | None, with_time: bool = True) -> str:
    """The date of `timestamp` as the language writes it, with the time of day; "-"
    without one."""
    if timestamp is None:
        return "-"
    moment = time.localtime(timestamp)
    date = format_date(moment)
    return f"{date} {time.strftime('%H:%M', moment)}" if with_time else date


def local_day(timestamp: float) -> str:
    """Day and month of `timestamp` as the language writes them."""
    return format_day_month(time.localtime(timestamp))


def kilometres(metres: int) -> str:
    return f"{metres} m" if metres < 1000 else f"{decimal(metres / 1000, '.1f')} km"


def distance_and_direction(observer: Observer, point: Coordinate) -> tuple[float, str]:
    """Great-circle distance in km from the listener and the compass direction."""
    lat1, lon1 = math.radians(observer.latitude), math.radians(observer.longitude)
    lon2, lat2 = math.radians(point[0]), math.radians(point[1])
    a = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    distance = 2 * _EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))
    bearing = math.degrees(
        math.atan2(
            math.sin(lon2 - lon1) * math.cos(lat2),
            math.cos(lat1) * math.sin(lat2)
            - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1),
        )
    )
    return distance, tr(_COMPASS[round((bearing % 360) / 45) % 8])


def severity(message: TecMessage) -> int:
    """Index into `SEVERITIES` by the effect (tec001): 7 no flow, 6 stationary, 3 to 5
    heavy, slow and queuing traffic, everything else a hint."""
    if message.effect == EFFECT_NO_TRAFFIC_FLOW:
        return 0
    if message.effect == 6:
        return 1
    if message.effect in (3, 4, 5):
        return 2
    return 3


def path(message: TecMessage) -> tuple[Coordinate, ...]:
    """The line of the message's OpenLR location; empty without one."""
    location = message.location
    if location is None or location.openlr is None:
        return ()
    return location.openlr.path()


def nearest(observer: Observer, points: tuple[Coordinate, ...]) -> tuple[float, str] | None:
    if not points:
        return None
    return min((distance_and_direction(observer, p) for p in points), key=lambda d: d[0])


def cause_text(message: TecMessage) -> str:
    if not message.causes:
        return "-"
    first = message.causes[0]
    text = cause_name(first.main, first.sub)
    if len(message.causes) > 1:
        text += f" (+{len(message.causes) - 1})"
    return text


def period(message: TecMessage | TfpMessage) -> str:
    if message.start is not None and message.stop is not None:
        # Translators: the period of a traffic message, from one date to another.
        return tr("{start} to {end}").format(
            start=local_date(message.start, with_time=False),
            end=local_date(message.stop, with_time=False),
        )
    if message.stop is not None:
        return tr("until {time}").format(time=local_date(message.stop))
    if message.start is not None:
        return tr("since {time}").format(time=local_date(message.start))
    return "-"


def short_period(message: TecMessage, now: float) -> str:
    """For a list row: "until 24 Oct", "until 18:00" today, "since 14:05"."""
    today = time.localtime(now)[:3]
    if message.stop is not None:
        same_day = time.localtime(message.stop)[:3] == today
        when = local(message.stop) if same_day else local_day(message.stop)
        return tr("until {time}").format(time=when)
    if message.start is not None:
        same_day = time.localtime(message.start)[:3] == today
        when = local(message.start) if same_day else local_day(message.start)
        return tr("since {time}").format(time=when)
    return ""


def place_text(location: Location | None) -> str:
    """Where the location begins: coordinates, else the TMC code, else its methods."""
    if location is None:
        return "-"
    if location.openlr is not None:
        lon, lat = location.openlr.points[0].coordinate
        # Translators: a position north and east; German "{latitude}° N, {longitude}° O".
        return tr("{latitude}° N, {longitude}° E").format(
            latitude=decimal(lat, ".4f"), longitude=decimal(lon, ".4f")
        )
    if location.tmc is not None:
        return f"TMC {location.tmc.location}"
    return ", ".join(location_method_name(m) for m in location.methods) or "-"


def describe_tec(message: TecMessage) -> list[str]:
    """The message in words, a line each (rich text)."""
    lines = [f"<b>{effect_name(message.effect)}</b>"]
    for cause in message.causes:
        if cause.linked_message is not None:
            lines.append(tr("Cause: see message {message}").format(message=cause.linked_message))
            continue
        parts = [cause_name(cause.main, cause.sub)]
        if cause.warning_level in WARNING_LEVELS:
            parts.append(tr(WARNING_LEVELS[cause.warning_level]))
        if cause.unverified:
            parts.append(tr("unconfirmed"))
        if cause.lane_restriction in LANE_RESTRICTIONS:
            lanes = tr(LANE_RESTRICTIONS[cause.lane_restriction])
            parts.append(f"{lanes} ({cause.lanes})" if cause.lanes else lanes)
        if cause.length_affected is not None:
            # Translators: the length of road a cause affects, "over 2.5 km".
            parts.append(tr("over {length}").format(length=kilometres(cause.length_affected)))
        lines.append(tr("Cause: {cause}").format(cause=", ".join(parts)))
        lines.extend(_quoted(text) for text in cause.free_text)
    for advice in message.advice:
        lines.append(tr("Advice: {advice}").format(advice=advice_name(advice.code, advice.sub)))
        lines.extend(_quoted(text) for text in advice.free_text)
    for restriction in message.restrictions:
        vehicles = VEHICLE_TYPES.get(restriction.vehicle_type)
        parts = [tr("all vehicles") if vehicles is None else tr(vehicles)]
        parts += [restriction_text(kind, value) for kind, value in restriction.restrictions]
        lines.append(tr("Applies to: {vehicles}").format(vehicles=", ".join(parts)))
    if message.speed_limits:
        limits = ", ".join(f"{limit} km/h" for limit in message.speed_limits)
        lines.append(tr("Speed limit: {limits}").format(limits=limits))
    if message.diversion_segments:
        count = message.diversion_segments
        lines.append(
            tr_plural(
                "Diversion over {count} section", "Diversion over {count} sections", count
            ).format(count=count)
        )
    extra = []
    if message.length_affected is not None:
        extra.append(tr("length {length}").format(length=kilometres(message.length_affected)))
    if message.delay is not None:
        extra.append(tr("delay {minutes} min").format(minutes=message.delay))
    if message.average_speed is not None:
        speed = f"{message.average_speed * 3.6:.0f}"
        extra.append(tr("speed {speed} km/h").format(speed=speed))
    if message.tendency in TENDENCIES:
        extra.append(tr("trend: {trend}").format(trend=tr(TENDENCIES[message.tendency])))
    if extra:
        line = ", ".join(extra)
        lines.append(line[:1].upper() + line[1:])
    if message.start is not None or message.stop is not None:
        lines.append(
            tr("Period: {start} to {end}").format(
                start=local_date(message.start), end=local_date(message.stop)
            )
        )
    lines += describe_location(message.location)
    return lines


def _quoted(text: str) -> str:
    """A text the service sent, in quotation marks (rich text)."""
    # Translators: quotation marks around a text a traffic message carries.
    return tr("“{text}”").format(text=html.escape(text))


def describe_location(location: Location | None) -> list[str]:
    if location is None:
        return []
    lines = []
    openlr = location.openlr
    if openlr is not None:
        first = openlr.points[0]
        count = len(openlr.points)
        if openlr.kind == OPENLR_LINEAR:
            shape = tr_plural("line through {count} point", "line through {count} points", count)
        else:
            shape = tr_plural(
                "point on a line through {count} point",
                "point on a line through {count} points",
                count,
            )
        parts = [shape.format(count=count)]
        length = openlr.length()
        if length is not None:
            parts.append(kilometres(length))
        parts.append(
            tr("{form_of_way} (road class FRC {frc})").format(
                form_of_way=form_of_way_name(first.fow), frc=first.frc
            )
        )
        lon, lat = first.coordinate
        # Translators: where an OpenLR location begins; German "Anfang {latitude}° N,
        # {longitude}° O".
        parts.append(
            tr("start {latitude}° N, {longitude}° E").format(
                latitude=decimal(lat, ".5f"), longitude=decimal(lon, ".5f")
            )
        )
        lines.append(tr("Location (OpenLR): {details}").format(details=", ".join(parts)))
    tmc = location.tmc
    if tmc is not None:
        if tmc.both_directions:
            direction = tr("both directions")
        elif tmc.positive_direction:
            direction = tr("positive direction")
        else:
            direction = tr("negative direction")
        table = f"{tmc.country:X}/{tmc.table}"
        if tmc.table_version is not None:
            version = f"{tmc.table_version[0]}.{tmc.table_version[1]}"
            place = tr("location {location} of table {table} (version {version})").format(
                location=tmc.location, table=table, version=version
            )
        else:
            place = tr("location {location} of table {table}").format(
                location=tmc.location, table=table
            )
        parts = [place, direction]
        if tmc.extent:
            parts.append(
                tr_plural(
                    "{count} further location", "{count} further locations", tmc.extent
                ).format(count=tmc.extent)
            )
        if tmc.hazard_distance:
            # Translators: where the hazard of a TMC message lies, before its location.
            parts.append(
                tr("hazard {distance} before the location").format(
                    distance=kilometres(tmc.hazard_distance)
                )
            )
        if tmc.problem_length:
            parts.append(tr("length {length}").format(length=kilometres(tmc.problem_length)))
        lines.append(tr("Location (TMC): {details}").format(details=", ".join(parts)))
    others = [m for m in location.methods if m not in (LOCATION_TMC, LOCATION_OPENLR)]
    if others:
        names = ", ".join(location_method_name(m) for m in others)
        lines.append(
            tr("Further location reference: {methods} (not publicly described)").format(
                methods=names
            )
        )
    return lines


def describe_management(stored: StoredMessage) -> str:
    """ID, version, priority, expiry and last reception (plain text)."""
    management = stored.message.management
    if management.priority in _PRIORITIES:
        text = tr("Message {message}, version {version}, priority {priority}").format(
            message=management.message_id,
            version=management.version,
            priority=tr(_PRIORITIES[management.priority]),
        )
    else:
        text = tr("Message {message}, version {version}").format(
            message=management.message_id, version=management.version
        )
    return tr("{message}; valid until {expiry}, last received {received}").format(
        message=text,
        expiry=local(management.expiry),
        received=local(stored.received, "%H:%M:%S"),
    )


def message_key(message: TecMessage | TfpMessage) -> Key:
    return message.scid, message.management.message_id


def describe_stretch(points: list[TmcPoint] | tuple[TmcPoint, ...]) -> str:
    """A TMC location in words, in the direction of traffic (towards the primary point):
    "B 6 between Bremen-Neustadt junction and Bremen-Überseestadt junction", "A 4 at
    Magdala junction"."""
    if not points:
        return ""
    road = points[0].road
    if len(points) > 1:
        start, end = points[-1].label(), points[0].label()
        if road:
            return tr("{road} between {start} and {end}").format(road=road, start=start, end=end)
        return tr("between {start} and {end}").format(start=start, end=end)
    if road:
        return tr("{road} at {place}").format(road=road, place=points[0].label())
    return tr("at {place}").format(place=points[0].label())


def _coordinates(points: tuple[TmcPoint, ...]) -> tuple[Coordinate, ...]:
    """The points of a TMC location as a line in the direction of traffic."""
    return tuple((p.longitude, p.latitude) for p in reversed(points))


@dataclass(frozen=True)
class TrafficEntry:
    """A traffic event as the surfaces list it."""

    key: Key
    stored: StoredMessage
    severity: int  # index into SEVERITIES
    path: tuple[Coordinate, ...]  # OpenLR, else from the TMC location; empty: no place
    distance: float | None  # km from the listener to its nearest point
    direction: str | None  # compass direction of that point
    tmc: tuple[TmcPoint, ...] = ()  # the TMC location resolved, primary point first

    @property
    def message(self) -> TecMessage:
        return self.stored.message

    def headline(self) -> str:
        """Effect and cause: "Road closed · roadworks"."""
        effect = effect_name(self.message.effect)
        return f"{effect[:1].upper()}{effect[1:]} · {cause_text(self.message)}"

    def where(self) -> str:
        """The TMC location in words; empty without the location table."""
        return describe_stretch(self.tmc)

    def short_where(self) -> str:
        """For a single line: road and primary point ("B 6 at Laatzen/Sehnde junction")."""
        return describe_stretch(self.tmc[:1])

    def road_kind(self) -> str:
        """What OpenLR says of the road ("motorway"); empty without OpenLR."""
        location = self.message.location
        if location is None or location.openlr is None:
            return ""
        return form_of_way_name(location.openlr.points[0].fow)

    def distance_text(self) -> str:
        if self.distance is None:
            return tr("no location")
        return f"{self.distance:.0f} km {self.direction}"


@dataclass(frozen=True)
class FlowStretch:
    """A section of a traffic flow message on its TMC location: from point `start` to
    point `end` (indices from the primary point), with its level and speed."""

    start: int
    end: int
    level: int  # 0 to 6, `FLOW_LEVELS`
    speed: int | None  # km/h


@dataclass(frozen=True)
class FlowEntry:
    """A traffic flow message on the points of its TMC location."""

    key: Key
    stored: StoredMessage
    points: tuple[TmcPoint, ...]  # primary point first
    stretches: tuple[FlowStretch, ...]
    worst: int  # the worst level along it, 0 if none is known
    distance: float  # km from the listener to its nearest point
    direction: str

    @property
    def message(self) -> TfpMessage:
        return self.stored.message

    def where(self) -> str:
        return describe_stretch(self.points)

    def path(self) -> tuple[Coordinate, ...]:
        return _coordinates(self.points)

    def speeds(self) -> str:
        found = [s.speed for s in self.stretches if s.speed]
        if not found:
            return ""
        low, high = min(found), max(found)
        return f"{low} km/h" if low == high else f"{low}–{high} km/h"

    def distance_text(self) -> str:
        return f"{self.distance:.0f} km {self.direction}"


def flow_stretches(message: TfpMessage, count: int) -> tuple[FlowStretch, ...]:
    """The sections of the current flow vector on a location of `count` points. A
    section ends at its offset (in TMC locations from the primary point) and begins
    where the one before ended."""
    if message.spatial_resolution not in (0, None) or not message.vectors:
        return ()
    stretches = []
    begin = 0
    for section in sorted(message.vectors[0].sections, key=lambda s: s.offset):
        end = min(section.offset, count - 1)
        if end > begin:
            level = 0 if section.level_of_service is None else section.level_of_service & 0x07
            stretches.append(FlowStretch(begin, end, level, section.average_speed))
        begin = max(begin, end)
    return tuple(stretches)


class TrafficModel(QObject):
    """Takes the TPEG frames of the signal processing (`add_frame`, connected by
    `for_controller`) and keeps the messages until they expire or are cancelled, also
    while another channel is received (`leave`); a frame of another TPEG service (by the
    service ID of its transport frame) starts afresh."""

    decoding_changed = pyqtSignal()  # TPEG appeared in or vanished from the ensemble
    messages_changed = pyqtSignal()  # messages came, went, or the position changed
    service_changed = pyqtSignal()  # a frame brought the SNI and statistics anew
    roads_changed = pyqtSignal()  # the road network was read
    locations_changed = pyqtSignal()  # the TMC location table was read

    def __init__(
        self,
        observer: Observer,
        data_directory: Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._observer = observer
        # Road network and TMC location table are read from here on first use.
        self._data_directory = data_directory
        self._roads: RoadNetwork | None = None
        self._roads_read = data_directory is None
        self._locations: TmcLocationTable | None = None
        self._locations_read = data_directory is None
        self._clock = ReferenceClock()
        self._tpeg: DataComponent | None = None
        self._decoding: DataComponent | None = None
        self._store = TpegMessageStore()
        self._service_id: ServiceId | None = None  # whose messages the store holds
        self._service: ServiceInformation | None = None
        self._statistics: TpegStatistics | None = None
        self._last_frame_at: float | None = None
        # The current spell of reception: when its first frame came, and whether
        # messages held from before it still await their repetition.
        self._visit_started: float | None = None
        self._visit_pending = False
        self._entries: list[TrafficEntry] | None = None
        self._flows: list[FlowEntry] | None = None

    @classmethod
    def for_controller(
        cls, controller: ReceiverController, parent: QObject | None = None
    ) -> TrafficModel:
        """A model fed by the controller's signal processing, at its position, with the
        road network and location table of its data directory."""
        model = cls(controller.observer, controller.data_directory, parent)
        dsp = controller.dsp_worker
        dsp.data_services_updated.connect(model.update_components)
        dsp.data_decoding_ready.connect(model.set_data_decoding)
        dsp.date_time_ready.connect(model.synchronise_clock)
        dsp.tpeg_frame_ready.connect(model.add_frame)
        controller.observer_changed.connect(model.set_observer)
        # The messages stay through a channel change; the ensemble's components do not.
        controller.live_reset.connect(model.leave)
        return model

    # --- Input --------------------------------------------------------------------------

    def update_components(self, components: list[DataComponent]) -> None:
        tpeg = tpeg_component(components)
        if tpeg != self._tpeg:
            self._tpeg = tpeg
            self.decoding_changed.emit()

    def set_data_decoding(self, statuses: tuple[DataComponentStatus, ...]) -> None:
        decoding = next(
            (
                status.component
                for status in statuses
                if status.component is not None
                and USER_APPLICATION_TPEG in status.component.app_types
            ),
            None,
        )
        if decoding != self._decoding:
            self._decoding = decoding
            self.decoding_changed.emit()

    def synchronise_clock(self, broadcast_time: BroadcastTime) -> None:
        self._clock.synchronise(broadcast_time)

    def set_observer(self, observer: Observer) -> None:
        if observer != self._observer:
            self._observer = observer
            self._changed()

    def set_roads(self, roads: RoadNetwork | None) -> None:
        """Uses this road network instead of reading the file."""
        self._roads = roads
        self._roads_read = True
        self.roads_changed.emit()

    def set_locations(self, locations: TmcLocationTable | None) -> None:
        """Uses this TMC location table instead of reading the file."""
        self._locations = locations
        self._locations_read = True
        self._changed()
        self.locations_changed.emit()

    def reset(self) -> None:
        """Forgets service, statistics and messages."""
        self._forget()
        self._last_frame_at = None
        self._changed()
        self.service_changed.emit()

    def leave(self) -> None:
        """Reception moved to another frequency (`ReceiverController.live_reset`): the
        ensemble's TPEG component is void, the messages and what the SNI said stay."""
        self._visit_started = None
        if self._tpeg is not None or self._decoding is not None:
            self._tpeg = None
            self._decoding = None
            self.decoding_changed.emit()

    def _forget(self) -> None:
        self._store.clear()
        self._service_id = None
        self._service = None
        self._statistics = None
        self._visit_started = None
        self._visit_pending = False

    def add_frame(self, scid: int, frame: TpegFrame) -> None:
        """Takes a decoded transport frame (`dsp.pipeline.DspWorker.tpeg_frame_ready`)."""
        now = self.now()
        changed = False
        service_id = frame.frame.service_id
        if service_id is not None and service_id != self._service_id:
            # Another TPEG service: the messages held are not its own.
            changed = len(self._store) > 0
            self._forget()
            self._service_id = service_id
        if self._visit_started is None or now - self._last_frame_at > SILENT_SECONDS:
            # Back after an absence: what is held has to come again within one cycle.
            self._visit_started = now
            self._visit_pending = len(self._store) > 0
        self._service = frame.service
        self._statistics = frame.statistics
        self._last_frame_at = now
        for message in frame.messages:
            changed |= self._store.add(message, frame.frame, now)
        changed |= self._prune(now)
        if changed:
            self._changed()
        self.service_changed.emit()

    def expire(self, now: float | None = None) -> None:
        """Drops the messages past their expiry, and those the service has not repeated
        since the receiver came back (called before drawing)."""
        if self._prune(self.now() if now is None else now):
            self._changed()

    def _prune(self, now: float) -> bool:
        changed = self._store.expire(now)
        if (
            self._visit_pending
            and self._visit_started is not None
            and now - self._visit_started >= CYCLE_SECONDS
        ):
            self._visit_pending = False
            changed |= self._store.drop_received_before(self._visit_started)
        return changed

    def _changed(self) -> None:
        self._entries = None
        self._flows = None
        self.messages_changed.emit()

    # --- State --------------------------------------------------------------------------

    def now(self) -> float:
        synchronised = self._clock.now()
        return time.time() if synchronised is None else synchronised

    @property
    def observer(self) -> Observer:
        return self._observer

    @property
    def tpeg(self) -> DataComponent | None:
        return self._tpeg

    @property
    def decoding(self) -> DataComponent | None:
        return self._decoding

    @property
    def service(self) -> ServiceInformation | None:
        return self._service

    @property
    def statistics(self) -> TpegStatistics | None:
        return self._statistics

    @property
    def silent(self) -> bool:
        """Frames came, but none for `SILENT_SECONDS`."""
        return self._last_frame_at is not None and self.now() - self._last_frame_at > SILENT_SECONDS

    def held_since(self) -> float | None:
        """The time of the last frame while messages are held but the service is not
        being received (reception left its channel, or it fell silent); None while they
        are live, or without messages."""
        if self._last_frame_at is None or not len(self._store):
            return None
        if self._visit_started is None or self.silent:
            return self._last_frame_at
        return None

    @property
    def roads(self) -> RoadNetwork | None:
        """The road network, read on first use; None if it was not built."""
        if not self._roads_read:
            self._roads_read = True
            self._roads = load_roads(self._data_directory)
            if self._roads is not None:
                self.roads_changed.emit()
        return self._roads

    @property
    def locations(self) -> TmcLocationTable | None:
        """The TMC location table, read on first use; None if it was not imported."""
        if not self._locations_read:
            self._locations_read = True
            self._locations = load_locations(self._data_directory)
            if self._locations is not None:
                self._entries = self._flows = None
                self.locations_changed.emit()
        return self._locations

    def get(self, key: Key) -> StoredMessage | None:
        return self._store.get(key)

    def stored(self, kind: type) -> list[StoredMessage]:
        return [s for s in self._store.messages() if isinstance(s.message, kind)]

    def counts(self, aid: int) -> tuple[int, int | None]:
        """Messages of an application held, and how many the SNI says are on air."""
        kind = TecMessage if aid == AID_TEC else TfpMessage
        have = len(self.stored(kind))
        service = self._service
        announced = None
        if service is not None:
            scid = next((e.scid for e in service.components if e.application == aid), None)
            announced = service.message_counts.get(scid)
        return have, announced

    def _resolve(self, location: Location | None) -> tuple[TmcPoint, ...]:
        """The points of a message's TMC location, if the table knows it."""
        table = self.locations
        if table is None or location is None or location.tmc is None:
            return ()
        if not table.serves(location.tmc):
            return ()
        return tuple(table.stretch(location.tmc))

    def entries(self) -> list[TrafficEntry]:
        """The traffic events, the nearest first; those without a place at the end."""
        if self._entries is None:
            entries = []
            for stored in self.stored(TecMessage):
                message: TecMessage = stored.message
                points = self._resolve(message.location)
                line = path(message) or _coordinates(points)
                found = nearest(self._observer, line)
                entries.append(
                    TrafficEntry(
                        message_key(message),
                        stored,
                        severity(message),
                        line,
                        None if found is None else found[0],
                        None if found is None else found[1],
                        points,
                    )
                )
            entries.sort(
                key=lambda e: (math.inf if e.distance is None else e.distance, e.severity, e.key)
            )
            self._entries = entries
        return self._entries

    def entry(self, key: Key) -> TrafficEntry | None:
        return next((e for e in self.entries() if e.key == key), None)

    def flows(self) -> list[FlowEntry]:
        """The traffic flow messages whose location the TMC table knows, the worst first,
        then the nearest. Empty without the table."""
        if self._flows is None:
            flows = []
            for stored in self.stored(TfpMessage):
                message: TfpMessage = stored.message
                points = self._resolve(message.location)
                if not points:
                    continue
                stretches = flow_stretches(message, len(points))
                found = nearest(self._observer, _coordinates(points))
                flows.append(
                    FlowEntry(
                        message_key(message),
                        stored,
                        points,
                        stretches,
                        max((s.level for s in stretches if s.level < 7), default=0),
                        found[0],
                        found[1],
                    )
                )
            flows.sort(key=lambda f: (-f.worst, f.distance, f.key))
            self._flows = flows
        return self._flows

    def flow(self, key: Key) -> FlowEntry | None:
        return next((f for f in self.flows() if f.key == key), None)
