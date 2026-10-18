"""Where a service can be heard besides the tuned ensemble: the groundwork of service
following (ETSI EN 300 401 clause 8.1.10, ETSI TS 103 176).

FIG 0/24 names the ensembles carrying a service by their EId, FIG 0/21 the frequencies
an ensemble is broadcast on. Which channel carries an ensemble here is only known from
reception: the EId each channel of the channel list was received with (FIG 1/0). What
FIG 0/21 says is announced, and may hold elsewhere in the country: 5D announces its own
ensemble 11F7 on 8C as well, while 8C carries a different ensemble in some regions. So
an announced channel only counts while no other ensemble was received on it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from dab.channels import channel_for_frequency
from dab.ensemble_cache import EnsembleInfo


@dataclass(frozen=True)
class EnsembleLocation:
    """Where an ensemble can be heard, as far as is known."""

    eid: int
    channel: str | None  # None: no channel known
    # True: the channel list received the ensemble on `channel`; False: FIG 0/21 only
    # announces it there (or, without a channel, nothing is known).
    received: bool
    label: str | None = None  # the ensemble label from the channel list, if received


def locate_ensemble(
    eid: int,
    stored: Mapping[str, EnsembleInfo],
    announced: Mapping[int, tuple[int, ...]],
) -> EnsembleLocation:
    """The channel of an ensemble: one whose stored list came from it, else the first
    channel `announced` (FIG 0/21: EId -> frequencies in kHz) names for it on which no
    other ensemble was received. Stored lists without an EId (from before it was kept)
    rule out no channel."""
    for channel, info in stored.items():
        if info.eid == eid:
            return EnsembleLocation(eid, channel, received=True, label=info.label)
    for frequency in announced.get(eid, ()):
        channel = channel_for_frequency(frequency)
        info = None if channel is None else stored.get(channel)
        if channel is not None and (info is None or info.eid is None):
            return EnsembleLocation(eid, channel, received=False)
    return EnsembleLocation(eid, None, received=False)


def other_ensembles(
    eids: Iterable[int],
    own_eid: int,
    stored: Mapping[str, EnsembleInfo],
    announced: Mapping[int, tuple[int, ...]],
) -> tuple[EnsembleLocation, ...]:
    """The ensembles of `eids` (FIG 0/24 for one service) other than the tuned one
    `own_eid`, located: those received first, then those announced, then those with no
    channel known, each in the order sent."""
    locations = [
        locate_ensemble(eid, stored, announced) for eid in dict.fromkeys(eids) if eid != own_eid
    ]
    return tuple(
        sorted(locations, key=lambda location: (not location.received, location.channel is None))
    )
