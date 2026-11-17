"""The Emergency Warning System of DAB (ETSI TS 104 089), in Germany "ASA" (Automatic
Safety Alert): alert messages that a receiver plays on its own when they concern the
place it is at.

Ensembles that take part send FIG 0/15 about once per second: empty as a heartbeat while
there is nothing to say, else describing alert messages. One carried in the ensemble goes
through the phases trigger (at least 5 s: stage, incident and the alert area as location
codes), sustain (once per second while it plays) and end. One carried in another ensemble
is signalled during its trigger phase only, with that ensemble's EId; the receiver tunes
there to hear it. A pre-trigger before the trigger serves the transmitters among each
other and is not for receivers.

`EwsMonitor` follows the FIG 0/15 of the tuned ensemble over time and says which alerts
are in their trigger phase and which alert messages are under way. `AlertSelector`
decides, as a receiver that is playing audio (clause 7.2.2.4, "audio mode"), which of
them to play: by stage, by what the listener dismissed (clause 7.4.2) and by location
(clause 7.5). Playing it, and tuning to another ensemble for it, is up to
`ui.receiver_controller`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import IntEnum

from dab.announcement import audio_in_subchannel
from dab.fig import Ensemble, EwsSignal
from dab.location_code import LocationCode
from i18n import tr_context

# Clause 7.2.2.4: without a FIG 0/15 for this long an ensemble no longer takes part.
EWS_TIMEOUT_SECONDS = 10.0
# An alert group is sent completely at least once per second (clause 6.6.1), more
# slowly only when it is very long; an alert not repeated for this long has left its
# trigger phase.
TRIGGER_TIMEOUT_SECONDS = 3.0
# The sustain phase comes once per second (clause 6.6.2). If neither it nor the trigger
# came for this long, the alert message counts as over even though its end was not
# received, as for announcements (`dab.announcement.ANNOUNCEMENT_TIMEOUT_SECONDS`).
ALERT_TIMEOUT_SECONDS = 5.0
# An alert played or broken off is not played again while it is still signalled, which
# it may be for a while from other ensembles (clause 6.5.1, note); one not signalled for
# this long that comes again is a new alert message.
FORGET_SECONDS = 60.0
# Clause 7.4.2: incident identifiers are reused, so the listener's choices for one lapse
# when it has not been signalled for a while; the standard suggests a week.
DISMISSAL_LIFETIME_SECONDS = 7 * 24 * 3600.0


class Phase(IntEnum):
    """Of an alert message, in the Id field of FIG 0/15 (annex E)."""

    PRE_TRIGGER = 0
    TRIGGER = 1
    SUSTAIN = 2
    END = 3


class Stage(IntEnum):
    """Of an incident, in the status field of FIG 0/15 (annex E, clause 6.2.3). Level 1
    targets every receiver, level 2 only those already playing audio."""

    LEVEL_1_START = 0
    LEVEL_1_UPDATE = 1
    LEVEL_1_REPEAT = 2
    LEVEL_1_CRITICAL = 3
    LEVEL_2_START = 4
    LEVEL_2_UPDATE = 5
    LEVEL_2_REPEAT = 6
    TEST = 7


_STARTS = (Stage.LEVEL_1_START, Stage.LEVEL_2_START)
_UPDATES = (Stage.LEVEL_1_UPDATE, Stage.LEVEL_2_UPDATE)
_REPEATS = (Stage.LEVEL_1_REPEAT, Stage.LEVEL_2_REPEAT)


def stage_text(stage: Stage) -> str:
    """What an alert of this stage is, for the listener, in the language set; the level
    is left out, since an alert that is played concerns the listener either way."""
    if stage in _STARTS:
        return tr_context("alert stage", "Warning")
    if stage in _UPDATES:
        return tr_context("alert stage", "Update")
    if stage in _REPEATS:
        return tr_context("alert stage", "Repeat")
    if stage == Stage.LEVEL_1_CRITICAL:
        return tr_context("alert stage", "Critical warning")
    return tr_context("alert stage", "Test")


@dataclass(frozen=True)
class EwsAlert:
    """An alert in its trigger phase, from the FIG 0/15 of its alert set seen so far."""

    other_ensemble: bool
    # The ensemble carrying the alert message: the tuned one (from FIG 0/0) or the one
    # named in the Id field.
    eid: int | None
    stage: Stage
    incident: int  # IId: the alerts of one incident share it (clause 6.2.2)
    # The alert area: the location codes received so far, and whether a FIG 0/15 of the
    # set came without any, which makes it the whole coverage area of the ensemble.
    locations: frozenset[LocationCode] = frozenset()
    whole_area: bool = False
    # In the tuned ensemble: the subchannel of the alert audio, the service with its
    # audio there (FIG 0/2) and whether FIG 0/1 configures the subchannel. In another
    # ensemble whether it can be heard is up to the receiver, so None.
    subchannel_id: int | None = None
    sid: int | None = None
    receivable: bool | None = None

    @property
    def key(self) -> tuple[int | None, Stage, int]:
        """The same in the ensemble carrying the alert as in the ensembles signalling it
        as another ensemble's: one ensemble carries one alert message at a time."""
        return self.eid, self.stage, self.incident

    @property
    def incident_key(self) -> tuple[int | None, int]:
        """What dismissing applies to (clause 7.4.2): EId and IId."""
        return self.eid, self.incident


@dataclass(frozen=True)
class EwsState:
    """What the tuned ensemble signals for the Emergency Warning System."""

    # FIG 0/15 came within `EWS_TIMEOUT_SECONDS`: the ensemble takes part.
    participating: bool = False
    eid: int | None = None  # of the tuned ensemble, from FIG 0/0
    # In their trigger phase: those carried in the tuned ensemble first, then those of
    # other ensembles, each in the order they began.
    alerts: tuple[EwsAlert, ...] = ()
    # The subchannels whose alert message is under way (trigger or sustain phase).
    running: frozenset[int] = frozenset()


@dataclass
class _Trigger:
    first_seen: float
    seen: float
    locations: frozenset[LocationCode]
    whole_area: bool


class EwsMonitor:
    """Follows the FIG 0/15 of the tuned ensemble over time; `state` says how things
    stand. Pre-trigger signalling is ignored, as by every consumer receiver."""

    def __init__(self) -> None:
        self._last_signal: float | None = None
        # (OE flag, SubChId or EId, stage, IId) -> what its alert set said so far.
        self._triggers: dict[tuple[bool, int, int, int], _Trigger] = {}
        self._running: dict[int, float] = {}  # SubChId -> last trigger or sustain

    def reset(self) -> None:
        self._last_signal = None
        self._triggers.clear()
        self._running.clear()

    def update(self, signals: Iterable[EwsSignal], now: float) -> None:
        """Takes the FIG 0/15 received at `now` (seconds, any monotonic time base)."""
        for signal in signals:
            self._last_signal = now
            if signal.heartbeat or signal.phase == Phase.PRE_TRIGGER:
                continue
            if not signal.other_ensemble:
                subchannel = signal.subchannel_id
                if signal.phase == Phase.END:
                    self._running.pop(subchannel, None)
                    continue
                self._running[subchannel] = now
                if signal.phase != Phase.TRIGGER:
                    continue
            key = (
                signal.other_ensemble,
                signal.eid if signal.other_ensemble else signal.subchannel_id,
                signal.stage,
                signal.incident,
            )
            trigger = self._triggers.get(key)
            if trigger is None or now - trigger.seen > TRIGGER_TIMEOUT_SECONDS:
                trigger = self._triggers[key] = _Trigger(now, now, frozenset(), False)
            trigger.seen = now
            trigger.locations |= frozenset(signal.locations)
            trigger.whole_area = trigger.whole_area or not signal.locations

    def state(self, ensemble: Ensemble, now: float) -> EwsState:
        """How things stand at `now`, with what FIG 0/0, 0/1 and 0/2 of `ensemble` say
        about the alerts carried in it."""
        self._triggers = {
            key: trigger
            for key, trigger in self._triggers.items()
            if now - trigger.seen <= TRIGGER_TIMEOUT_SECONDS
        }
        self._running = {
            subchannel: seen
            for subchannel, seen in self._running.items()
            if now - seen <= ALERT_TIMEOUT_SECONDS
        }
        ordered = sorted(self._triggers.items(), key=lambda item: (item[0][0], item[1].first_seen))
        alerts = []
        for (other, identifier, stage, incident), trigger in ordered:
            alert = EwsAlert(
                other_ensemble=other,
                eid=identifier if other else ensemble.eid,
                stage=Stage(stage),
                incident=incident,
                locations=trigger.locations,
                whole_area=trigger.whole_area,
            )
            if not other:
                source = audio_in_subchannel(ensemble, identifier)
                alert = replace(
                    alert,
                    subchannel_id=identifier,
                    sid=None if source is None else source[0],
                    receivable=identifier in ensemble.subchannels,
                )
            alerts.append(alert)
        return EwsState(
            participating=(
                self._last_signal is not None and now - self._last_signal <= EWS_TIMEOUT_SECONDS
            ),
            eid=ensemble.eid,
            alerts=tuple(alerts),
            running=frozenset(self._running),
        )


def stage_matches(stage: Stage, *, repeats_dismissed: bool, incident_dismissed: bool) -> bool:
    """Table 1 of clause 7.5.3 for a receiver playing audio: the start of an incident
    and a critical alert always, an update unless the listener dismissed the incident,
    a repeat unless the listener dismissed it or its repeats, a test never."""
    if stage in _UPDATES:
        return not incident_dismissed
    if stage in _REPEATS:
        return not (repeats_dismissed or incident_dismissed)
    return stage != Stage.TEST


def location_matches(alert: EwsAlert, receiver: LocationCode | None) -> bool:
    """Clause 7.5.4: an alert without location codes concerns the whole coverage area,
    one with codes the receiver inside one of them. Without a location of its own a
    receiver only takes those for the whole area (clause 7.2.3)."""
    if alert.whole_area:
        return True
    return receiver is not None and any(code.contains(receiver) for code in alert.locations)


@dataclass
class _Dismissal:
    repeats: bool = False
    incident: bool = False
    seen: float = 0.0


class AlertSelector:
    """Which alert to play, for a receiver that is playing audio; and what the listener
    dismissed. Lives as long as the receiver, across tunings, so that an alert played
    in one ensemble is not played again when another one still signals it."""

    def __init__(self) -> None:
        self._sightings: dict[tuple, float] = {}  # EwsAlert.key -> last seen
        self._done: set[tuple] = set()  # played or broken off, while still signalled
        self._dismissals: dict[tuple, _Dismissal] = {}  # EwsAlert.incident_key -> choice

    def observe(self, state: EwsState, now: float) -> None:
        """Notes the alerts signalled at `now` (seconds, monotonic). The start of an
        incident clears what the listener dismissed for its identifier (clause 7.4.2)."""
        for alert in state.alerts:
            if alert.key not in self._sightings and alert.stage in _STARTS:
                self._dismissals.pop(alert.incident_key, None)
            self._sightings[alert.key] = now
            dismissal = self._dismissals.get(alert.incident_key)
            if dismissal is not None:
                dismissal.seen = now
        self._sightings = {
            key: seen for key, seen in self._sightings.items() if now - seen <= FORGET_SECONDS
        }
        self._done &= self._sightings.keys()
        self._dismissals = {
            key: dismissal
            for key, dismissal in self._dismissals.items()
            if now - dismissal.seen <= DISMISSAL_LIFETIME_SECONDS
        }

    def choose(
        self,
        state: EwsState,
        receiver: LocationCode | None,
        receivable: Callable[[EwsAlert], bool],
    ) -> EwsAlert | None:
        """The alert to play: the first that matches by stage, location and
        receivability (clause 7.5) and has not been played already, those of the tuned
        ensemble before those of others (clause 6.1). `receivable` decides for alerts of
        other ensembles whether that ensemble can be tuned to."""
        for alert in state.alerts:
            if alert.key in self._done:
                continue
            dismissal = self._dismissals.get(alert.incident_key, _Dismissal())
            if not stage_matches(
                alert.stage,
                repeats_dismissed=dismissal.repeats,
                incident_dismissed=dismissal.incident,
            ):
                continue
            if not location_matches(alert, receiver):
                continue
            if alert.other_ensemble and not receivable(alert):
                continue
            if not alert.other_ensemble and not alert.receivable:
                continue
            return alert
        return None

    def finished(self, alert: EwsAlert, now: float) -> None:
        """The alert was played to its end or broken off: not again while it is still
        signalled."""
        self._sightings[alert.key] = now
        self._done.add(alert.key)

    def dismiss(self, alert: EwsAlert, now: float, *, repeats: bool, incident: bool) -> None:
        """The listener asks not to hear the repeats of this incident, or nothing more of
        it but a critical alert or its next start (clause 7.4.2)."""
        dismissal = self._dismissals.setdefault(alert.incident_key, _Dismissal())
        dismissal.repeats = dismissal.repeats or repeats
        dismissal.incident = dismissal.incident or incident
        dismissal.seen = now
