"""The programme guides of all channels as one list, grouped by the channel each service
is broadcast on.

A programme guide is not confined to its own ensemble. The SI of EPG Deutschland on 5C
lists 42 services: 13 of its own ensemble, the 16 of Antenne DE on 5D and 13 of an
ensemble from North Rhine-Westphalia (EId 1205) that is not receivable here. Each
service comes with its bearers (`dab.spi.Bearer`), which name the ensembles carrying it;
EPG Deutschland gives every service a bearer in its own ensemble 10BC as well, including
the services that are not on 5C at all.

A bearer names an ensemble, not a frequency. Which channel an ensemble is on is known
from reception: the EId each channel of the channel list was received with (FIG 1/0),
and the ensemble a guide gives for itself in its SI, on the channel it came from. That
is measured. What the FIC says about other ensembles (FIG 0/21) is only announced: 5D
names 8C (199.360 MHz) for its own ensemble 11F7, while here 8C carries NI HANNOVER.
So an announcement never files a service under a channel; it only describes an
ensemble that was received nowhere.

A service is filed under the channel whose service list contains it (the stored
channel list or the running reception). Failing that, it goes under the ensembles its
bearers name other than the guide's own one: on the channels where each was received,
otherwise as an ensemble of its own, with the frequencies announced for it. A service
with no other ensemble stays with the channel its guide came from, and so do all
services of a guide whose own ensemble is unknown (neither its SI has come round yet
nor the EId of its channel). Where several guides know a service, they are merged by
SId, their schedules included.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from dab.channels import BAND_III_CHANNELS
from dab.spi import GuideService, Programme, ProgrammeGuide, merge_programmes

# The channel a guide came from, or a group belongs to; None stands for the manual
# frequency.
Channel = str | None


@dataclass(frozen=True)
class EnsembleServices:
    """The ensemble on a channel as far as it is known, from the stored channel list or
    the running reception."""

    label: str | None
    services: dict[int, str] = field(default_factory=dict)  # SId -> service label
    eid: int | None = None  # FIG 1/0: which ensemble was received there
    # FIG 0/21: EId -> centre frequencies in kHz, as this ensemble announces them.
    frequencies: dict[int, tuple[int, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class GuideEntry:
    """A service in the combined list."""

    sid: int
    name: str
    logo: bytes | None
    has_schedule: bool
    origins: tuple[Channel, ...]  # the channels whose guide lists the service
    # The service list of the group's channel contains the service: it can be tuned to
    # there. False in the group of an unknown ensemble.
    listed: bool = False


@dataclass(frozen=True)
class GuideGroup:
    """The services of one channel, or of an ensemble on none of the known channels."""

    channel: Channel  # None: manual frequency, or the unknown ensemble `eid`
    eid: int | None  # only for an unknown ensemble
    label: str | None  # ensemble label, if known
    entries: tuple[GuideEntry, ...]
    # Only for an unknown ensemble: its frequencies in kHz as announced in FIG 0/21,
    # and the channels whose ensembles announce them.
    frequencies: tuple[int, ...] = ()
    announced_by: tuple[Channel, ...] = ()

    @property
    def key(self) -> tuple[Channel, int | None]:
        return self.channel, self.eid

    @property
    def origins(self) -> tuple[Channel, ...]:
        """The channels whose guides name the services of this group."""
        return tuple(dict.fromkeys(origin for entry in self.entries for origin in entry.origins))


def channel_order(channel: Channel) -> tuple[int, float, str]:
    """Sort key for channels: Band III by frequency, then others, the manual frequency
    last."""
    if channel is None:
        return 2, 0.0, ""
    if channel in BAND_III_CHANNELS:
        return 0, BAND_III_CHANNELS[channel], channel
    return 1, 0.0, channel


class CombinedGuide:
    """The programme guides received on several channels, one `ProgrammeGuide` each."""

    def __init__(self) -> None:
        self._guides: dict[Channel, ProgrammeGuide] = {}

    def guide(self, origin: Channel) -> ProgrammeGuide:
        """The guide received on a channel, created empty on first use."""
        return self._guides.setdefault(origin, ProgrammeGuide())

    def clear(self) -> None:
        self._guides.clear()

    @property
    def empty(self) -> bool:
        return all(guide.empty for guide in self._guides.values())

    def groups(
        self, ensembles: dict[Channel, EnsembleServices], labels: dict[int, str] | None = None
    ) -> list[GuideGroup]:
        """The services of all guides, grouped as described in the module docstring:
        channels in order of frequency, then the manual frequency, then ensembles on
        no known channel; the services of a group by name.

        Args:
            ensembles: the known ensembles by channel.
            labels: further service labels (SId -> label) for services that neither
                the guides nor `ensembles` name.
        """
        names = dict(labels or {})
        for ensemble in ensembles.values():
            names.update(ensemble.services)
        located = self._received_ensembles(ensembles)

        found: dict[tuple[Channel, int | None], dict[int, list[tuple[Channel, GuideService]]]] = {}
        for origin, guide in self._guides.items():
            own = set(guide.ensemble_ids)
            received = ensembles.get(origin)
            if received is not None and received.eid is not None:
                own.add(received.eid)
            for service in guide.services(names):
                for key in _group_keys(service, origin, own, ensembles, located):
                    found.setdefault(key, {}).setdefault(service.sid, []).append((origin, service))

        announced = _announcements(ensembles)
        groups = []
        for (channel, eid), services in found.items():
            ensemble = ensembles.get(channel) if eid is None else None
            listed = set() if ensemble is None else set(ensemble.services)
            entries = sorted(
                (_entry(channel, candidates, listed) for candidates in services.values()),
                key=lambda entry: (entry.name.casefold(), entry.sid),
            )
            if eid is None:
                label = ensemble.label if ensemble is not None else None
                groups.append(GuideGroup(channel, None, label, tuple(entries)))
                continue
            frequencies = announced.get(eid, {})
            sources = {source for sources in frequencies.values() for source in sources}
            groups.append(
                GuideGroup(
                    None,
                    eid,
                    None,
                    tuple(entries),
                    frequencies=tuple(sorted(frequencies)),
                    announced_by=tuple(sorted(sources, key=channel_order)),
                )
            )
        return sorted(groups, key=_group_order)

    def _received_ensembles(
        self, ensembles: dict[Channel, EnsembleServices]
    ) -> dict[int, list[Channel]]:
        """EId -> the channels the ensemble was received on: with its EId in the FIC,
        or as the ensemble a guide from that channel gives for itself."""
        received = [(ensemble.eid, channel) for channel, ensemble in ensembles.items()]
        received += [
            (eid, origin) for origin, guide in self._guides.items() for eid in guide.ensemble_ids
        ]
        located: dict[int, list[Channel]] = {}
        for eid, channel in received:
            if eid is not None and channel not in located.setdefault(eid, []):
                located[eid].append(channel)
        return located

    def programmes(self, sid: int) -> list[Programme]:
        """All programmes of a service across the guides of all channels, in order of
        time."""
        return merge_programmes(
            programme for guide in self._guides.values() for programme in guide.programmes(sid)
        )


def service_logos(groups: Iterable[GuideGroup], channel: Channel) -> dict[int, bytes]:
    """SId -> logo of every service in `groups` that has one, for the service list of
    `channel`. A service filed under several channels takes its logo from the group of
    `channel`, where the guide received on that channel comes first."""
    logos: dict[int, bytes] = {}
    for group in sorted(groups, key=lambda group: group.key == (channel, None)):
        logos.update((entry.sid, entry.logo) for entry in group.entries if entry.logo is not None)
    return logos


def _group_keys(
    service: GuideService,
    origin: Channel,
    own: set[int],
    ensembles: dict[Channel, EnsembleServices],
    located: dict[int, list[Channel]],
) -> list[tuple[Channel, int | None]]:
    """The groups a service of the guide from `origin` belongs to."""
    channels = [
        channel for channel, ensemble in ensembles.items() if service.sid in ensemble.services
    ]
    if channels:
        return [(channel, None) for channel in channels]
    keys = []
    for eid in sorted(service.ensembles - own):
        if eid in located:
            keys += [(channel, None) for channel in located[eid]]
        elif own:
            keys.append((None, eid))
        # With the guide's own ensemble unknown (its SI may simply not have come round
        # yet), the bearer most likely names just that.
    return list(dict.fromkeys(keys)) or [(origin, None)]


def _announcements(
    ensembles: dict[Channel, EnsembleServices],
) -> dict[int, dict[int, set[Channel]]]:
    """EId -> frequency in kHz -> the channels whose ensembles announce it (FIG 0/21)."""
    announced: dict[int, dict[int, set[Channel]]] = {}
    for channel, ensemble in ensembles.items():
        for eid, frequencies in ensemble.frequencies.items():
            for frequency in frequencies:
                announced.setdefault(eid, {}).setdefault(frequency, set()).add(channel)
    return announced


def _entry(
    channel: Channel, candidates: Iterable[tuple[Channel, GuideService]], listed: set[int]
) -> GuideEntry:
    """Merges what several guides know about a service; the guide received on the
    service's own channel comes first. `listed` are the SIds on the channel's service
    list."""
    ordered = sorted(candidates, key=lambda candidate: candidate[0] != channel)
    services = [service for _, service in ordered]
    return GuideEntry(
        sid=services[0].sid,
        name=services[0].name,
        logo=next((service.logo for service in services if service.logo is not None), None),
        has_schedule=any(service.has_schedule for service in services),
        origins=tuple(dict.fromkeys(origin for origin, _ in ordered)),
        listed=services[0].sid in listed,
    )


def _group_order(group: GuideGroup) -> tuple:
    if group.eid is not None:
        return 1, (), group.eid
    return 0, channel_order(group.channel), 0
