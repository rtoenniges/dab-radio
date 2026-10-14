"""Persistent per-channel list of ensembles found by the scanner or during reception."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path

logger = logging.getLogger(__name__)

# Where a stored entry came from.
SOURCE_SCAN = "scan"  # channel scan
SOURCE_LIVE = "live"  # taken over from live reception

_FORMAT_VERSION = 1


@dataclass(frozen=True)
class EnsembleInfo:
    """What is stored about the ensemble of one channel."""

    label: str | None  # ensemble label (FIG 1/0); None if never received
    services: tuple[tuple[int, str], ...]  # (SId, name) of the programme services, by name
    updated_at: float  # Unix time of the capture
    source: str  # SOURCE_SCAN or SOURCE_LIVE
    # Share of CRC-valid FIBs during the scan (0..1): an honest measure of how well the
    # channel could be received. Only measured by the scanner; None when the entry was
    # taken over from live reception.
    fib_success: float | None = None
    # (SId, PTy code) from FIG 0/17, by SId. Missing in lists stored before programme
    # types were evaluated; they load with none.
    programme_types: tuple[tuple[int, int], ...] = ()
    # Ensemble identifier (FIG 1/0): which ensemble is received on this channel. None
    # in lists stored before it was kept.
    eid: int | None = None
    # (EId, centre frequencies in kHz) from FIG 0/21, by EId: where this ensemble and
    # others are broadcast according to this one; announced, not measured.
    frequencies: tuple[tuple[int, tuple[int, ...]], ...] = ()
    # (SId, EIds of the ensembles carrying it) from FIG 0/24, by SId; see
    # `keep_service_ensembles` for how it is collected. Missing in lists stored before.
    service_ensembles: tuple[tuple[int, tuple[int, ...]], ...] = ()

    def to_json(self) -> dict:
        return {
            "label": self.label,
            "services": [[sid, name] for sid, name in self.services],
            "updated_at": self.updated_at,
            "source": self.source,
            "fib_success": self.fib_success,
            "programme_types": [[sid, code] for sid, code in self.programme_types],
            "eid": self.eid,
            "frequencies": [[eid, list(frequencies)] for eid, frequencies in self.frequencies],
            "service_ensembles": [[sid, list(eids)] for sid, eids in self.service_ensembles],
        }

    @classmethod
    def from_json(cls, data: dict) -> EnsembleInfo:
        """Raises KeyError, TypeError or ValueError for a malformed entry."""
        label = data["label"]
        fib_success = data.get("fib_success")
        eid = data.get("eid")
        return cls(
            label=None if label is None else str(label),
            services=tuple((int(sid), str(name)) for sid, name in data["services"]),
            updated_at=float(data["updated_at"]),
            source=str(data["source"]),
            fib_success=None if fib_success is None else float(fib_success),
            programme_types=tuple(
                (int(sid), int(code)) for sid, code in data.get("programme_types", ())
            ),
            eid=None if eid is None else int(eid),
            frequencies=tuple(
                (int(eid), tuple(int(frequency) for frequency in frequencies))
                for eid, frequencies in data.get("frequencies", ())
            ),
            service_ensembles=tuple(
                (int(sid), tuple(int(eid) for eid in eids))
                for sid, eids in data.get("service_ensembles", ())
            ),
        )


def keep_service_ensembles(info: EnsembleInfo, previous: EnsembleInfo | None) -> EnsembleInfo:
    """`info` with the FIG 0/24 entries of the entry it replaces that it lacks itself, if
    both come from the same ensemble.

    FIG 0/24 comes round complete only every two minutes or so, while a scan listens for
    seconds: what one capture misses, an earlier one may have. An entry of `info`
    replaces the one for the same SId.
    """
    if previous is None or info.eid is None or previous.eid != info.eid:
        return info
    kept = dict(previous.service_ensembles) | dict(info.service_ensembles)
    return replace(info, service_ensembles=tuple(sorted(kept.items())))


class EnsembleCache:
    """Stored ensembles per channel, kept as a JSON file.

    A channel is listed here exactly when an ensemble was found on it the last time it
    was looked at, so a scan that finds nothing on a channel removes it again.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, EnsembleInfo] = {}

    def load(self) -> None:
        """Reads the file. A missing file leaves the cache empty; so does a corrupt one
        (with a warning), and the next scan rebuilds it."""
        self._entries = {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            logger.warning("Channel list %s is not readable: %s", self._path, exc)
            return

        if not isinstance(data, dict) or data.get("version") != _FORMAT_VERSION:
            logger.warning("Channel list %s has an unknown format", self._path)
            return
        for channel, entry in data.get("channels", {}).items():
            try:
                self._entries[channel] = EnsembleInfo.from_json(entry)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipped entry for channel %s: %s", channel, exc)

    def save(self) -> None:
        """Writes the file, first to a temporary file that is then renamed, so that an
        interruption in the middle of writing cannot destroy the old list.

        Raises:
            OSError: if the file cannot be written.
        """
        data = {
            "version": _FORMAT_VERSION,
            "channels": {channel: info.to_json() for channel, info in self._entries.items()},
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(self._path.name + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(temporary, self._path)

    def get(self, channel: str) -> EnsembleInfo | None:
        return self._entries.get(channel)

    def entries(self) -> dict[str, EnsembleInfo]:
        """All stored ensembles by channel."""
        return dict(self._entries)

    def put(self, channel: str, info: EnsembleInfo) -> None:
        self._entries[channel] = info

    def remove(self, channel: str) -> None:
        self._entries.pop(channel, None)
