"""Announcements: traffic news, news flashes and the like, for which a receiver may
leave the selected service for a while (ETSI EN 300 401, clause 8.1.6).

FIG 0/18 names, per service, the announcement types that may interrupt it and the
announcement clusters it belongs to. While an announcement runs, FIG 0/19 reports it
for a cluster, with its types and the subchannel carrying its audio. It applies to a
service if the service is in that cluster and permits one of its types. Alarm
announcements are different: they come in the reserved cluster 0xFF and interrupt
every service, but only while the Al flag of FIG 0/0 enables them.

Announcements carried in other ensembles work alike (clauses 8.1.6.3 and 8.1.6.4):
FIG 0/25 names, per service, the types it permits from other ensembles, with OE
clusters and the ensemble serving each; FIG 0/26 reports a running one for an OE
cluster, with the service carrying it there. Hearing it means retuning, which is up to
`ui.receiver_controller`.

The type names are the English ones of ETSI TS 101 756, table 14, and the German
catalog has the abbreviations of table B.4; the longer descriptions follow table 14.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from dab.fig import (
    TMID_STREAM_AUDIO,
    Announcement,
    AnnouncementSupport,
    AudioComponent,
    Ensemble,
    OeAnnouncement,
    OeAnnouncementSupport,
)
from i18n import tr, tr_context

ALARM_CLUSTER = 0xFF  # clause 8.1.6.2: used exclusively for alarm announcements
ALARM_FLAG = 0x0001  # b0 of the ASu/ASw flags

# FIG 0/19 is repeated once per second while an announcement runs. If it has not come
# for this long, the announcement counts as over even though its end was not received,
# e.g. because reception failed at that moment; otherwise a receiver switched to the
# announcement would never switch back. Five repetitions missed in a row mean that the
# FIC is hardly decodable anyway.
ANNOUNCEMENT_TIMEOUT_SECONDS = 5.0


class AnnouncementTypeName(NamedTuple):
    short: str  # name (TS 101 756, table 14), for compact displays
    full: str


def _names() -> tuple[AnnouncementTypeName, ...]:
    """ETSI TS 101 756 V2.4.1, indexed by the bit of the ASu/ASw flags, in the language
    set. b11 to b15 are reserved."""
    return (
        AnnouncementTypeName(
            # Translators: the announcement types (ETSI TS 101 756, table 14), short for
            # lists such as "Announcement: Road Traffic flash, News flash", each followed by
            # a longer description; German: the abbreviations of table B.4 ("Service" for
            # the event announcement is the standard's choice).
            tr_context("announcement type", "Alarm"),
            tr_context("announcement type", "Alarm: serious or life-threatening situation"),
        ),
        AnnouncementTypeName(
            tr_context("announcement type", "Road Traffic flash"),
            tr_context("announcement type", "Road traffic flash: problems on the road"),
        ),
        AnnouncementTypeName(
            tr_context("announcement type", "Transport flash"),
            tr_context("announcement type", "Transport flash: public transport, timetables"),
        ),
        AnnouncementTypeName(
            tr_context("announcement type", "Warning/Service"),
            tr_context("announcement type", "Warning/Service: e.g. power or water outage"),
        ),
        AnnouncementTypeName(
            tr_context("announcement type", "News flash"),
            tr_context("announcement type", "News flash: short news"),
        ),
        AnnouncementTypeName(
            tr_context("announcement type", "Area weather flash"),
            tr_context("announcement type", "Area weather flash: weather report"),
        ),
        AnnouncementTypeName(
            tr_context("announcement type", "Event announcement"),
            tr_context("announcement type", "Event announcement: notice of an event"),
        ),
        AnnouncementTypeName(
            tr_context("announcement type", "Special event"),
            tr_context("announcement type", "Special event: unforeseen event"),
        ),
        AnnouncementTypeName(
            tr_context("announcement type", "Programme Information"),
            tr_context("announcement type", "Programme Information: programme notice"),
        ),
        AnnouncementTypeName(
            tr_context("announcement type", "Sport report"),
            tr_context("announcement type", "Sport report: sports news"),
        ),
        AnnouncementTypeName(
            tr_context("announcement type", "Financial report"),
            tr_context("announcement type", "Financial report: business and finance news"),
        ),
    )


def announcement_types(flags: int) -> list[AnnouncementTypeName]:
    """The names of the types whose bits are set in ASu or ASw flags, in bit order and
    in the language set. Reserved bits get a placeholder, so that nothing the station
    sends goes missing."""
    known = _names()
    names = []
    for bit in range(16):
        if flags & (1 << bit):
            if bit < len(known):
                names.append(known[bit])
            else:
                names.append(
                    AnnouncementTypeName(
                        tr("Type {bit}").format(bit=bit),
                        tr("Type {bit} (reserved)").format(bit=bit),
                    )
                )
    return names


@dataclass(frozen=True)
class AnnouncementStatus:
    """What the FIC says about announcements for one service."""

    # FIG 0/18 of the service; None if it sent none, then the service may only be
    # interrupted by alarm announcements.
    support: AnnouncementSupport | None
    alarm_enabled: bool  # Al flag of FIG 0/0
    # The running announcement that applies to the service, None if there is none.
    active: Announcement | None = None
    # The service whose audio is in the subchannel of `active`, if FIG 0/2 names one:
    # its SId and its audio component there.
    source_sid: int | None = None
    source_component: AudioComponent | None = None
    # FIG 0/25 of the service; None if it sent none.
    oe_support: OeAnnouncementSupport | None = None
    # The running announcement of another ensemble that applies to the service, and the
    # EId of that ensemble. Only while none of the tuned ensemble applies, unless it is
    # an alarm and the one here is not; `active` is None then.
    oe_active: OeAnnouncement | None = None
    oe_eid: int | None = None
    # A running announcement in the service's own subchannel, for whichever cluster: the
    # service is making it. How a receiver gone to another ensemble for an announcement
    # tells that it has ended.
    carried: Announcement | None = None


class AnnouncementMonitor:
    """Follows the running announcements of an ensemble over time.

    `dab.fig` records what FIG 0/19 and FIG 0/26 report; this adds the timeout for
    announcements whose end was lost (see `ANNOUNCEMENT_TIMEOUT_SECONDS`). A report
    counts as confirmed when `Ensemble.announcements` (or `oe_announcements`) holds a new
    object for its key, so `update` has to see the ensemble after every decoded FIB
    batch.
    """

    def __init__(self) -> None:
        # Cluster Id -> (the latest report, the time it arrived).
        self._reports: dict[int, tuple[Announcement, float]] = {}
        # (OE Cluster Id, target SId) -> (the latest report, the time it arrived).
        self._oe_reports: dict[tuple[int, int], tuple[OeAnnouncement, float]] = {}

    def reset(self) -> None:
        self._reports.clear()
        self._oe_reports.clear()

    def update(self, ensemble: Ensemble, now: float) -> None:
        """Notes which announcements FIG 0/19 and FIG 0/26 have just confirmed; `now`
        in seconds, on any monotonic time base."""
        self._reports = _confirmed(self._reports, ensemble.announcements, now)
        self._oe_reports = _confirmed(self._oe_reports, ensemble.oe_announcements, now)

    def running(self, now: float) -> list[Announcement]:
        """The announcements of the tuned ensemble running at `now`, ordered by cluster."""
        return _running(self._reports, now)

    def running_oe(self, now: float) -> list[OeAnnouncement]:
        """The announcements of other ensembles running at `now`, ordered by OE cluster
        and target."""
        return _running(self._oe_reports, now)

    def status(self, ensemble: Ensemble, sid: int, now: float) -> AnnouncementStatus:
        """The announcement situation of one service at `now`, as of the last
        `update`."""
        alarm_enabled = ensemble.alarm_announcements
        support = ensemble.announcement_support.get(sid)
        oe_support = ensemble.oe_announcement_support.get(sid)
        running = self.running(now)
        active = _applicable(running, support, alarm_enabled=alarm_enabled)
        oe = _applicable_oe(self.running_oe(now), oe_support, alarm_enabled=alarm_enabled)
        oe_active = oe_eid = None
        # An alarm before everything else, then an announcement of the tuned ensemble.
        if oe is not None and (active is None or (_is_alarm(oe[0]) and not _is_alarm(active))):
            active = None
            oe_active, oe_eid = oe
        source = None if active is None else audio_in_subchannel(ensemble, active.subchannel_id)
        own = ensemble.audio_components.get(sid)
        carried = None
        if own is not None:
            carried = next(
                (
                    announcement
                    for announcement in running
                    if announcement.subchannel_id == own.subchannel_id
                    and (alarm_enabled or not _is_alarm(announcement))
                ),
                None,
            )
        return AnnouncementStatus(
            support=support,
            alarm_enabled=alarm_enabled,
            active=active,
            source_sid=None if source is None else source[0],
            source_component=None if source is None else source[1],
            oe_support=oe_support,
            oe_active=oe_active,
            oe_eid=oe_eid,
            carried=carried,
        )


def _confirmed[K, A](
    previous: dict[K, tuple[A, float]], current: dict[K, A], now: float
) -> dict[K, tuple[A, float]]:
    """The reports of `current` with the time each was confirmed: kept from `previous`
    while the object is the same, `now` for a new one."""
    reports = {}
    for key, announcement in current.items():
        seen = previous.get(key)
        reports[key] = seen if seen is not None and seen[0] is announcement else (announcement, now)
    return reports


def _running[A](reports: dict, now: float) -> list[A]:
    """The reported announcements confirmed within the timeout, ordered by key."""
    return [
        announcement
        for _key, (announcement, seen) in sorted(reports.items())
        if now - seen <= ANNOUNCEMENT_TIMEOUT_SECONDS
    ]


def _is_alarm(announcement: Announcement | OeAnnouncement) -> bool:
    return announcement.cluster_id == ALARM_CLUSTER


def _applicable(
    running: list[Announcement], support: AnnouncementSupport | None, *, alarm_enabled: bool
) -> Announcement | None:
    """The running announcement that may interrupt a service with this support, an
    alarm before everything else. Alarm announcements are only valid while the Al flag
    enables them (clause 8.1.6.0)."""
    if alarm_enabled:
        for announcement in running:
            if announcement.cluster_id == ALARM_CLUSTER and announcement.flags & ALARM_FLAG:
                return announcement
    if support is None:
        return None
    for announcement in running:
        if (
            announcement.cluster_id != ALARM_CLUSTER
            and announcement.cluster_id in support.clusters
            and announcement.flags & support.flags
        ):
            return announcement
    return None


def _applicable_oe(
    running: list[OeAnnouncement], support: OeAnnouncementSupport | None, *, alarm_enabled: bool
) -> tuple[OeAnnouncement, int] | None:
    """The running announcement of another ensemble that may interrupt a service with
    this support, with the EId of that ensemble; an alarm before everything else. An
    alarm needs no support, its target names the ensemble (clause 8.1.6.4); any other
    needs an OE cluster of the service and a common type."""
    if alarm_enabled:
        for announcement in running:
            if _is_alarm(announcement) and announcement.flags & ALARM_FLAG:
                return announcement, announcement.target_sid
    if support is None:
        return None
    for announcement in running:
        if _is_alarm(announcement) or not announcement.flags & support.flags:
            continue
        for cluster_id, eid in support.targets:
            if cluster_id == announcement.cluster_id:
                return announcement, eid
    return None


def audio_in_subchannel(
    ensemble: Ensemble, subchannel_id: int
) -> tuple[int, AudioComponent] | None:
    """The service with an audio component in this subchannel and that component, as
    FIG 0/2 lists them; a primary component before a secondary one. None if FIG 0/2
    names no audio in the subchannel (yet)."""
    found = None
    for sid, components in ensemble.service_components.items():
        for component in components:
            if component.tmid != TMID_STREAM_AUDIO or component.subchannel_id != subchannel_id:
                continue
            audio = (sid, AudioComponent(subchannel_id, component.type_code))
            if component.primary:
                return audio
            found = found or audio
    return found
