"""Service and Programme Information (SPI): the electronic programme guide, in the binary
encoding used for broadcast (ETSI TS 102 371, on the XML model of ETSI TS 102 818).

An SPI service is a MOT data carousel (see `dab.mot_directory`) in a packet mode
subchannel. It carries:

* one Service Information object (SI, MOT content type 7/0) per ensemble: the services
  with their names and the file names of their logos,
* Programme Information objects (PI, 7/1), one per service and day: the schedule,
* the logos themselves as PNG or JPEG images.

The SI and PI objects are XML documents compressed into a tree of tag-length-value
objects (clause 5): elements hold attributes (tags from 0x80), child elements and
character data (tag 0x01). Frequent strings can be replaced by one-byte tokens defined
in a token table at the top of the object (clause 5.5). Times are given in UTC like in
the FIC (MJD and time of day, clause 5.4.5.2), durations in seconds.

`decode_binary_object` turns the bytes into a generic `Element` tree, `decode_spi_object`
turns that into the few things a receiver displays (in the signal processing thread,
like all decoding), and `ProgrammeGuide` collects them across the objects of the
carousel.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from dab.mot import MotObject
from dab.timecode import mjd_to_timestamp
from i18n import tr, tr_noop

# MOT content type/subtype of the SPI objects (clause 6.4.2, table 11).
CONTENT_TYPE_SPI = 7
SUBTYPE_SERVICE_INFORMATION = 0
SUBTYPE_PROGRAMME_INFORMATION = 1
SUBTYPE_GROUP_INFORMATION = 2

# MOT parameters of SPI objects (clause 6.4, table 10).
PARAM_PROFILE_SUBSET = 0x21
PARAM_SCOPE_START = 0x25
PARAM_SCOPE_END = 0x26
PARAM_SCOPE_ID = 0x27

# Profile IDs in the user application data of FIG 0/13 (clause 7.1.1, table 12).
# Translators: the profiles of the programme guide (SPI): the basic and the advanced one.
PROFILE_NAMES = {1: tr_noop("Basic"), 2: tr_noop("Advanced")}

# Top-level and special tags (clause 5.3, annex D).
TAG_CDATA = 0x01
TAG_EPG = 0x02
TAG_SERVICE_INFORMATION = 0x03
TAG_TOKEN_TABLE = 0x04
TAG_DEFAULT_LANGUAGE = 0x06
# Element tags (annex D).
TAG_SHORT_NAME = 0x10
TAG_MEDIUM_NAME = 0x11
TAG_LONG_NAME = 0x12
TAG_MEDIA_DESCRIPTION = 0x13
TAG_LOCATION = 0x19
TAG_SHORT_DESCRIPTION = 0x1A
TAG_LONG_DESCRIPTION = 0x1B
TAG_PROGRAMME = 0x1C
TAG_SCHEDULE = 0x21
TAG_SCOPE = 0x24
TAG_SERVICE_SCOPE = 0x25
TAG_ENSEMBLE = 0x26
TAG_SERVICE = 0x28
TAG_BEARER = 0x29
TAG_MULTIMEDIA = 0x2B
TAG_TIME = 0x2C
# Attribute tags (annex E); their meaning depends on the element.
ATTR_ID = 0x80  # ensemble, bearer, serviceScope, programme
ATTR_SHORT_ID = 0x81  # programme
ATTR_START_TIME = 0x80  # scope
ATTR_STOP_TIME = 0x81  # scope
ATTR_TIME = 0x80  # time
ATTR_DURATION = 0x81  # time
ATTR_MULTIMEDIA_MIME = 0x80
ATTR_MULTIMEDIA_URL = 0x82
ATTR_MULTIMEDIA_TYPE = 0x83
ATTR_MULTIMEDIA_WIDTH = 0x84
ATTR_MULTIMEDIA_HEIGHT = 0x85

# Clause 5.5.2: the only byte values that can be tokens.
_TOKEN_TAGS = frozenset(
    (0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x0B, 0x0C, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13)
)


@dataclass
class Element:
    """One element of the binary object tree, with its attributes still as raw bytes."""

    tag: int
    attributes: dict[int, bytes] = field(default_factory=dict)
    children: list[Element] = field(default_factory=list)
    text: str | None = None  # CDATA content

    def child(self, tag: int) -> Element | None:
        return next((child for child in self.children if child.tag == tag), None)

    def all(self, tag: int) -> list[Element]:
        return [child for child in self.children if child.tag == tag]


def _read_tlv(data: bytes, offset: int) -> tuple[int, bytes, int] | None:
    """Reads a tag-length-value object (clauses 5.2.3 and 5.2.4): tag, one length byte,
    for 0xFE and 0xFF followed by a 16- or 24-bit length.

    Returns:
        `(tag, value, offset after the object)`, or None if it runs past the end.
    """
    if offset + 2 > len(data):
        return None
    tag, length = data[offset], data[offset + 1]
    offset += 2
    if length in (0xFE, 0xFF):
        size = 2 if length == 0xFE else 3
        if offset + size > len(data):
            return None
        length = int.from_bytes(data[offset : offset + size], "big")
        offset += size
    if offset + length > len(data):
        return None
    return tag, data[offset : offset + length], offset + length


def _parse_token_table(data: bytes) -> dict[int, bytes]:
    tokens: dict[int, bytes] = {}
    offset = 0
    while offset + 2 <= len(data):
        tag, length = data[offset], data[offset + 1]
        if tag in _TOKEN_TAGS:
            tokens[tag] = data[offset + 2 : offset + 2 + length]
        offset += 2 + length
    return tokens


def expand_text(raw: bytes, tokens: dict[int, bytes]) -> str:
    """Character data with the tokens replaced by their strings, decoded as UTF-8
    (clauses 5.3.1 and 5.5). Token tags are control characters, so they can never be
    part of a UTF-8 sequence and are replaced byte by byte."""
    if tokens and any(byte in tokens for byte in raw):
        raw = b"".join(tokens.get(byte, bytes((byte,))) for byte in raw)
    return raw.decode("utf-8", errors="replace")


def _parse_element(tag: int, data: bytes, tokens: dict[int, bytes]) -> Element:
    element = Element(tag)
    offset = 0
    while offset < len(data):
        tlv = _read_tlv(data, offset)
        if tlv is None:
            break  # a truncated object keeps what could be read
        child_tag, value, offset = tlv
        if child_tag == TAG_CDATA:
            element.text = expand_text(value, tokens)
        elif child_tag == TAG_TOKEN_TABLE:
            # Only in the top-level element, before any character data (clause 5.5.1).
            tokens.update(_parse_token_table(value))
        elif child_tag == TAG_DEFAULT_LANGUAGE:
            pass  # language selection is not offered
        elif child_tag >= 0x80:
            element.attributes[child_tag] = value
        else:
            element.children.append(_parse_element(child_tag, value, tokens))
    return element


def decode_binary_object(data: bytes) -> tuple[Element, dict[int, bytes]] | None:
    """Decodes an SPI binary object into its element tree (clause 5.2.2).

    Returns:
        `(top-level element, token table)`, or None if the object does not start with
        one of the two top-level elements (epg, serviceInformation).
    """
    tlv = _read_tlv(data, 0)
    if tlv is None or tlv[0] not in (TAG_EPG, TAG_SERVICE_INFORMATION):
        return None
    tokens: dict[int, bytes] = {}
    return _parse_element(tlv[0], tlv[1], tokens), tokens


def decode_timepoint(data: bytes) -> float | None:
    """A timePoint attribute as POSIX time (clause 5.4.5.2).

    Layout like the time in FIG 0/10: 1 bit Rfa, 17 bits MJD, 1 bit Rfa, then the LTO flag
    and the UTC flag, hours and minutes (short form) or additionally seconds and 10 bits
    Rfa (long form), and, with the LTO flag, a byte of local time offset. The offset
    only states the local time of the service; the moment itself is the UTC.

    The length of the attribute decides whether the LTO byte is there: the example in
    annex C of the standard sets the LTO flag in a 4-byte short form, whereas the
    Deutschlandfunk sends 5 bytes with the offset.
    """
    if len(data) < 4:
        return None
    value = int.from_bytes(data[:4], "big")
    mjd = (value >> 14) & 0x1FFFF
    long_form = bool(value & 0x0800)
    hours = (value >> 6) & 0x1F
    minutes = value & 0x3F
    seconds = 0
    if long_form:
        if len(data) < 6:
            return None
        seconds = data[4] >> 2
    if hours > 23 or minutes > 59 or seconds > 59:
        return None
    return mjd_to_timestamp(mjd, hours, minutes, seconds)


def decode_duration(data: bytes) -> int | None:
    """A duration attribute: seconds as a 16-bit number (clause 5.4.5.3)."""
    return int.from_bytes(data[:2], "big") if len(data) >= 2 else None


@dataclass(frozen=True)
class Bearer:
    """A service in the dab: domain (clause 5.4.5.1.2): where it is received."""

    ecc: int | None
    eid: int | None
    sid: int
    scids: int


def decode_bearer(data: bytes) -> Bearer | None:
    """A bearerURI in the dab: domain (clause 5.4.5.1.2, figure 3).

    First byte: Rfa, Ens flag, X-PAD flag, SId flag and the 4-bit SCIdS; then ECC and EId
    (Ens flag, always 1 today), then the SId with 16 bits, or 32 with the SId flag. On
    air (Deutschlandfunk): `40 e0 10bc d210` for dab:de0.10bc.d210.0.
    """
    if not data:
        return None
    flags = data[0]
    with_ensemble = bool(flags & 0x40)
    sid_bytes = 4 if flags & 0x10 else 2
    offset = 1
    ecc = eid = None
    if with_ensemble:
        if len(data) < offset + 3:
            return None
        ecc = data[offset]
        eid = int.from_bytes(data[offset + 1 : offset + 3], "big")
        offset += 3
    if len(data) < offset + sid_bytes:
        return None
    sid = int.from_bytes(data[offset : offset + sid_bytes], "big")
    return Bearer(ecc=ecc, eid=eid, sid=sid, scids=flags & 0x0F)


@dataclass(frozen=True)
class Names:
    """The three names of a service or programme: up to 8, 16 and 128 characters."""

    short: str = ""
    medium: str = ""
    long: str = ""

    @property
    def best(self) -> str:
        """The most informative name there is."""
        return self.long or self.medium or self.short


def _names(element: Element) -> Names:
    def text(tag: int) -> str:
        child = element.child(tag)
        return (child.text or "").strip() if child is not None else ""

    return Names(text(TAG_SHORT_NAME), text(TAG_MEDIUM_NAME), text(TAG_LONG_NAME))


def _descriptions(element: Element) -> tuple[str, str]:
    """(short, long) description from the mediaDescription children."""
    short = long = ""
    for media in element.all(TAG_MEDIA_DESCRIPTION):
        for tag in (TAG_SHORT_DESCRIPTION, TAG_LONG_DESCRIPTION):
            child = media.child(tag)
            if child is not None and child.text:
                if tag == TAG_SHORT_DESCRIPTION:
                    short = short or child.text.strip()
                else:
                    long = long or child.text.strip()
    return short, long


def _integer(value: bytes | None) -> int | None:
    return int.from_bytes(value, "big") if value else None


@dataclass(frozen=True)
class Logo:
    """A logo named in the SI; `content_name` is the ContentName of its MOT object."""

    content_name: str
    width: int | None
    height: int | None


@dataclass(frozen=True)
class SpiService:
    names: Names
    bearers: tuple[Bearer, ...]
    logos: tuple[Logo, ...]
    description: str


@dataclass(frozen=True)
class ServiceInformation:
    """The SI object (content type 7/0): the services of an ensemble."""

    ensemble_names: Names
    ecc: int | None
    eid: int | None
    services: tuple[SpiService, ...]


@dataclass(frozen=True)
class Programme:
    """One programme of a schedule."""

    names: Names
    start: float  # POSIX time
    duration: int  # seconds
    short_description: str = ""
    long_description: str = ""
    short_id: int | None = None

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def description(self) -> str:
        return self.long_description or self.short_description


@dataclass(frozen=True)
class Schedule:
    """A PI object (content type 7/1): the programmes of one service, usually one day."""

    service: Bearer | None
    start: float | None  # scope of the object
    stop: float | None
    programmes: tuple[Programme, ...]


def _service_information(root: Element, tokens: dict[int, bytes]) -> ServiceInformation:
    ensemble = root.child(TAG_ENSEMBLE)
    ecc = eid = None
    ensemble_names = Names()
    # DAB: the services are children of the ensemble element (clause 5.3.2.3).
    service_parent = root
    if ensemble is not None:
        ensemble_names = _names(ensemble)
        identifier = ensemble.attributes.get(ATTR_ID, b"")
        if len(identifier) >= 3:
            ecc, eid = identifier[0], int.from_bytes(identifier[1:3], "big")
        service_parent = ensemble
    services = []
    for service in service_parent.all(TAG_SERVICE):
        bearers = []
        for bearer in service.all(TAG_BEARER):
            decoded = decode_bearer(bearer.attributes.get(ATTR_ID, b""))
            if decoded is not None:
                bearers.append(decoded)
        logos = []
        for media in service.all(TAG_MEDIA_DESCRIPTION):
            for multimedia in media.all(TAG_MULTIMEDIA):
                url = multimedia.attributes.get(ATTR_MULTIMEDIA_URL)
                if url:
                    logos.append(
                        Logo(
                            content_name=expand_text(url, tokens),
                            width=_integer(multimedia.attributes.get(ATTR_MULTIMEDIA_WIDTH)),
                            height=_integer(multimedia.attributes.get(ATTR_MULTIMEDIA_HEIGHT)),
                        )
                    )
        short, long = _descriptions(service)
        services.append(SpiService(_names(service), tuple(bearers), tuple(logos), long or short))
    return ServiceInformation(ensemble_names, ecc, eid, tuple(services))


def _schedule(root: Element, scope_id: bytes | None) -> Schedule | None:
    schedule = root.child(TAG_SCHEDULE)
    if schedule is None:
        return None
    service = None
    start = stop = None
    scope = schedule.child(TAG_SCOPE)
    if scope is not None:
        start = decode_timepoint(scope.attributes.get(ATTR_START_TIME, b""))
        stop = decode_timepoint(scope.attributes.get(ATTR_STOP_TIME, b""))
        service_scope = scope.child(TAG_SERVICE_SCOPE)
        if service_scope is not None:
            service = decode_bearer(service_scope.attributes.get(ATTR_ID, b""))
    if service is None and scope_id:
        service = decode_bearer(scope_id)

    programmes = []
    for element in schedule.all(TAG_PROGRAMME):
        programme = _programme(element)
        if programme is not None:
            programmes.append(programme)
    programmes.sort(key=lambda programme: programme.start)
    return Schedule(service, start, stop, tuple(programmes))


def _programme(element: Element) -> Programme | None:
    for location in element.all(TAG_LOCATION):
        time = location.child(TAG_TIME)
        if time is None:
            continue
        start = decode_timepoint(time.attributes.get(ATTR_TIME, b""))
        duration = decode_duration(time.attributes.get(ATTR_DURATION, b""))
        if start is None or duration is None:
            continue
        short, long = _descriptions(element)
        return Programme(
            names=_names(element),
            start=start,
            duration=duration,
            short_description=short,
            long_description=long,
            short_id=_integer(element.attributes.get(ATTR_SHORT_ID)),
        )
    return None  # only on demand, or without a usable time


def decode_spi_object(mot_object: MotObject) -> ServiceInformation | Schedule | None:
    """Decodes an SI or PI object of the basic profile.

    Objects of the advanced profile (MOT parameter ProfileSubset) only complement the
    basic ones and are skipped, as are group information (7/2) and anything that is
    not SPI at all.
    """
    if mot_object.content_type != CONTENT_TYPE_SPI:
        return None
    if PARAM_PROFILE_SUBSET in mot_object.parameters:
        return None
    decoded = decode_binary_object(mot_object.body)
    if decoded is None:
        return None
    root, tokens = decoded
    subtype = mot_object.content_subtype
    if subtype == SUBTYPE_SERVICE_INFORMATION and root.tag == TAG_SERVICE_INFORMATION:
        return _service_information(root, tokens)
    if subtype == SUBTYPE_PROGRAMME_INFORMATION and root.tag == TAG_EPG:
        return _schedule(root, mot_object.parameters.get(PARAM_SCOPE_ID))
    return None


def profile_names(user_application_data: bytes) -> list[str]:
    """The SPI profiles announced in FIG 0/13 (clause 7.1.1): profile IDs up to the end
    or up to a delimiting 0x00; their names in the language set."""
    names = []
    for profile in user_application_data:
        if profile == 0:
            break
        name = PROFILE_NAMES.get(profile)
        if name is None:
            names.append(tr("Profile {profile}").format(profile=profile))
        else:
            names.append(tr(name))
    return names


def merge_programmes(programmes: Iterable[Programme]) -> list[Programme]:
    """Programmes from several schedules in order of time; a programme listed twice
    (in the objects of two days around midnight, or in two guides) appears once."""
    unique: dict[tuple[float, str], Programme] = {}
    for programme in programmes:
        unique.setdefault((programme.start, programme.names.best), programme)
    return sorted(unique.values(), key=lambda programme: programme.start)


@dataclass(frozen=True)
class GuideService:
    """A service as the programme guide knows it."""

    sid: int
    name: str
    logo: bytes | None  # image data of the best logo for a list, if received
    has_schedule: bool
    # EIds of the ensembles its bearers name, i.e. where the service is broadcast.
    ensembles: frozenset[int] = frozenset()


class ProgrammeGuide:
    """Collects the SI, PI and logo objects of an SPI carousel.

    Objects are kept by ContentName: a new version of an object replaces the old one,
    and when a new MOT directory no longer lists an object (e.g. yesterday's schedule),
    `retain` drops it.
    """

    def __init__(self) -> None:
        self._information: dict[str, ServiceInformation] = {}
        self._schedules: dict[str, Schedule] = {}
        self._images: dict[str, bytes] = {}

    def clear(self) -> None:
        self._information.clear()
        self._schedules.clear()
        self._images.clear()

    @property
    def empty(self) -> bool:
        return not (self._information or self._schedules)

    @property
    def schedule_count(self) -> int:
        return len(self._schedules)

    @property
    def has_service_information(self) -> bool:
        return bool(self._information)

    @property
    def ensemble_ids(self) -> frozenset[int]:
        """The EIds the SI objects give for their ensemble: the guide's own ensemble."""
        return frozenset(
            information.eid
            for information in self._information.values()
            if information.eid is not None
        )

    def add(self, mot_object: MotObject, decoded: ServiceInformation | Schedule | None) -> bool:
        """Takes a completed object of the carousel: an image, or an SI or PI object
        together with what `decode_spi_object` made of it.

        Returns:
            True if the guide changed.
        """
        name = mot_object.content_name
        if not name:
            return False
        if mot_object.is_image:
            self._images[name] = mot_object.body
            return True
        if isinstance(decoded, ServiceInformation):
            self._information[name] = decoded
            return True
        if isinstance(decoded, Schedule):
            self._schedules[name] = decoded
            return True
        return False

    def retain(self, content_names: set[str]) -> bool:
        """Drops every object whose ContentName is not in `content_names`."""
        changed = False
        for store in (self._information, self._schedules, self._images):
            for name in [name for name in store if name not in content_names]:
                del store[name]
                changed = True
        return changed

    def content_names(self) -> set[str]:
        """The ContentNames of all objects the guide holds."""
        return set(self._information) | set(self._schedules) | set(self._images)

    def drop_schedules_ended_before(self, moment: float) -> bool:
        """Drops the schedules whose scope (or, without one, last programme) ended
        before `moment`: a stored guide from last week is of no use.

        Returns:
            True if the guide changed.
        """
        outdated = []
        for name, schedule in self._schedules.items():
            end = schedule.stop
            if end is None and schedule.programmes:
                end = max(programme.end for programme in schedule.programmes)
            if end is not None and end < moment:
                outdated.append(name)
        for name in outdated:
            del self._schedules[name]
        return bool(outdated)

    def services(self, labels: dict[int, str]) -> list[GuideService]:
        """The services the SI lists or a schedule covers, named from the SI or else
        from `labels` (service labels of the FIC), sorted by name. Some carousels only
        carry the SI with the logos and no schedules at all (NDR on 7A)."""
        scheduled = [
            schedule.service
            for schedule in self._schedules.values()
            if schedule.service is not None
        ]
        listed = [
            bearer
            for information in self._information.values()
            for service in information.services
            for bearer in service.bearers
        ]
        ensembles: dict[int, set[int]] = {}
        for bearer in scheduled + listed:
            eids = ensembles.setdefault(bearer.sid, set())
            if bearer.eid is not None:
                eids.add(bearer.eid)
        with_schedule = {bearer.sid for bearer in scheduled}
        result = []
        for sid, eids in ensembles.items():
            service = self._spi_service(sid)
            name = service.names.best if service is not None else ""
            name = name or labels.get(sid) or f"{sid:04X}"
            result.append(
                GuideService(sid, name, self._logo(service), sid in with_schedule, frozenset(eids))
            )
        return sorted(result, key=lambda service: service.name.casefold())

    def programmes(self, sid: int) -> list[Programme]:
        """All programmes of a service, across the objects of all days, in order of
        time."""
        return merge_programmes(
            programme
            for schedule in self._schedules.values()
            if schedule.service is not None and schedule.service.sid == sid
            for programme in schedule.programmes
        )

    def _spi_service(self, sid: int) -> SpiService | None:
        for information in self._information.values():
            for service in information.services:
                if any(bearer.sid == sid for bearer in service.bearers):
                    return service
        return None

    def _logo(self, service: SpiService | None) -> bytes | None:
        """The received logo that suits a list entry best: the square one closest to
        128 pixels, else any."""
        if service is None:
            return None
        received = [logo for logo in service.logos if logo.content_name in self._images]
        if not received:
            return None

        def unsuitability(logo: Logo) -> tuple[int, int]:
            square = logo.width is not None and logo.width == logo.height
            return (0 if square else 1, abs((logo.width or 0) - 128))

        return self._images[min(received, key=unsuitability).content_name]
