"""The road network beneath the traffic map: roads from motorways down to district
roads, borders, coast and town names of Germany, taken from OpenStreetMap by
`tools/build_road_network.py` and stored as one file in the data directory.

Data © OpenStreetMap contributors, under the Open Database License (ODbL); a map that
shows it must say so (`ATTRIBUTION`). The file is built on each computer and never
committed: the extract of Germany alone has some 5 GB.

ARD TPEG locates its messages by OpenLR on roads of every class from 0 (motorway) to 4
(district road) in about equal shares (measured on 7A, 06.10.2026: of 572 messages 18 %
on class 0, 26 % on 1, 19 % on 2, 14 % on 3, 22 % on 4, 1 % on 5), so the network goes
down to tertiary roads. It is only the background: a message is not named after the
road nearest to it, since that is in some 8 % the road crossing it (compared with the
TMC location of the messages that carry both, same day); names come from the TMC
location table (`dab.tmc_locations`).

The lines are kept per layer as all their points (longitude, latitude) in one array
with the start of each line; `lines` cuts out those in view, with NaN between lines
as pyqtgraph draws them (`connect="finite"`). OSM splits a road at every junction into
pieces of mostly two points; the build joins them end to end, since drawing costs by
the point and a piece that short leaves nothing to simplify. Besides the fine lines the
roads down to the Landesstraßen, borders and coast have a medium copy for a region,
the main roads a coarse copy for the whole country (`FINE`, `MEDIUM`, `COARSE`): the
fine ones would cost the Raspberry Pi too much to draw there (measured 06.10.2026 on
the PC, a view 4° wide: 660 000 points in pieces took 184 ms per frame).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from i18n import tr_noop

ROAD_NETWORK_FILENAME = "road_network.npz"
# Raised whenever the layout of the file changes; an older file must be built anew.
FORMAT_VERSION = 3
# As OpenStreetMap asks for it, in the language of the surface (the wording of
# openstreetmap.org/copyright in that language); shown with `tr`.
ATTRIBUTION = tr_noop("© OpenStreetMap contributors")

# The layers, from the most important road down; then borders and coast.
MOTORWAY = "motorway"  # Autobahn
MAJOR = "major"  # trunk and primary roads: Bundesstraßen, some Landesstraßen
SECONDARY = "secondary"  # mostly Landesstraßen
TERTIARY = "tertiary"  # mostly Kreisstraßen
LINK = "link"  # slip roads of all of them
BORDER = "border"  # the borders of Germany on land
STATE = "state"  # the borders of the federal states
COAST = "coast"
ROAD_LAYERS = (MOTORWAY, MAJOR, SECONDARY, TERTIARY, LINK)
LAYERS = (*ROAD_LAYERS, BORDER, STATE, COAST)
# The copies of a layer by detail; every layer has the fine one.
FINE = "fine"
MEDIUM = "medium"  # for a region, as wide as a few districts
COARSE = "coarse"  # for the whole country
# The layers that also have a medium or a coarse copy.
MEDIUM_LAYERS = (MOTORWAY, MAJOR, SECONDARY, BORDER, STATE, COAST)
COARSE_LAYERS = (MOTORWAY, MAJOR, BORDER, STATE, COAST)
_COPIES = {FINE: LAYERS, MEDIUM: MEDIUM_LAYERS, COARSE: COARSE_LAYERS}

# OSM highway=* by layer.
OSM_HIGHWAYS = {
    "motorway": MOTORWAY,
    "trunk": MAJOR,
    "primary": MAJOR,
    "secondary": SECONDARY,
    "tertiary": TERTIARY,
    "motorway_link": LINK,
    "trunk_link": LINK,
    "primary_link": LINK,
    "secondary_link": LINK,
    "tertiary_link": LINK,
}

# Places: cities (place=city) and towns (place=town).
CITY = 0
TOWN = 1


Bounds = tuple[float, float, float, float]  # west, south, east, north


@dataclass(frozen=True)
class Place:
    name: str
    longitude: float
    latitude: float
    rank: int  # CITY or TOWN
    population: int  # 0: unknown


@dataclass
class _Layer:
    xy: np.ndarray  # (points, 2) float32: longitude, latitude
    start: np.ndarray  # (lines + 1,) int64: first point of each line, then the end
    bounds: np.ndarray  # (lines, 4) float32: west, south, east, north

    @classmethod
    def make(cls, xy: np.ndarray, start: np.ndarray) -> _Layer:
        if len(start) > 1:
            first = start[:-1]
            bounds = np.column_stack(
                [
                    np.minimum.reduceat(xy[:, 0], first),
                    np.minimum.reduceat(xy[:, 1], first),
                    np.maximum.reduceat(xy[:, 0], first),
                    np.maximum.reduceat(xy[:, 1], first),
                ]
            )
        else:
            bounds = np.zeros((0, 4), np.float32)
        return cls(xy, start, bounds)

    def __len__(self) -> int:
        return len(self.start) - 1

    def in_view(self, view: Bounds) -> np.ndarray:
        west, south, east, north = view
        b = self.bounds
        return np.flatnonzero(
            (b[:, 2] >= west) & (b[:, 0] <= east) & (b[:, 3] >= south) & (b[:, 1] <= north)
        )

    def lines(self, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The points of the lines `selected`, a NaN after each."""
        if not len(selected):
            return np.empty(0), np.empty(0)
        first = self.start[selected]
        lengths = self.start[selected + 1] - first
        total = int(lengths.sum()) + len(selected)
        # Position of each output point in `xy`: runs of each line, one more for the gap.
        ends = np.cumsum(lengths + 1)
        index = np.arange(total) - np.repeat(ends - lengths - 1, lengths + 1)
        index += np.repeat(first, lengths + 1)
        gaps = ends - 1
        index[gaps] = 0
        x = self.xy[index, 0].astype(np.float64)
        y = self.xy[index, 1].astype(np.float64)
        x[gaps] = np.nan
        y[gaps] = np.nan
        return x, y


def array_name(layer: str, detail: str, part: str) -> str:
    """The name of an array of the file: `part` "xy" or "start" of a copy of a layer."""
    return f"{layer}_{part}" if detail == FINE else f"{layer}_{detail}_{part}"


class RoadNetwork:
    """The copies of the layers by detail (all fine, some also medium or coarse) and
    the places."""

    def __init__(
        self,
        copies: dict[str, dict[str, _Layer]],
        places: list[Place],
        source: str = "",
    ) -> None:
        self._copies = copies
        self.places = places
        self.source = source

    @classmethod
    def load(cls, path: Path) -> RoadNetwork:
        """Reads a file of `tools/build_road_network.py`; ValueError if it has another
        format version."""
        with np.load(path, allow_pickle=False) as data:
            version = int(data["version"])
            if version != FORMAT_VERSION:
                raise ValueError(f"road network format {version}, expected {FORMAT_VERSION}")
            copies = {
                detail: {
                    name: _Layer.make(
                        data[array_name(name, detail, "xy")],
                        data[array_name(name, detail, "start")],
                    )
                    for name in names
                }
                for detail, names in _COPIES.items()
            }
            places = [
                Place(str(name), float(xy[0]), float(xy[1]), int(rank), int(population))
                for name, xy, rank, population in zip(
                    data["place_name"],
                    data["place_xy"],
                    data["place_rank"],
                    data["place_population"],
                )
            ]
            return cls(copies, places, str(data["source"]))

    def line_count(self, layer: str, detail: str = FINE) -> int:
        return len(self._copies[detail][layer])

    def lines(self, layer: str, view: Bounds, detail: str = FINE) -> tuple[np.ndarray, np.ndarray]:
        """The lines of a layer in view (degrees), with NaN between them, from the copy
        of `detail`; the fine one where the layer has no such copy."""
        source = self._copies[detail].get(layer)
        if source is None:
            source = self._copies[FINE][layer]
        return source.lines(source.in_view(view))

    def places_in(self, view: Bounds, rank: int = TOWN) -> list[Place]:
        """The places in view up to `rank` (CITY: cities only), largest first."""
        west, south, east, north = view
        found = [
            p
            for p in self.places
            if p.rank <= rank and west <= p.longitude <= east and south <= p.latitude <= north
        ]
        found.sort(key=lambda p: (p.rank, -p.population))
        return found


def default_path(data_directory: Path) -> Path:
    return data_directory / ROAD_NETWORK_FILENAME


def load_if_present(data_directory: Path) -> RoadNetwork | None:
    """The network in the data directory, None if it was not built (or is outdated)."""
    path = default_path(data_directory)
    if not path.exists():
        return None
    try:
        return RoadNetwork.load(path)
    except OSError, ValueError, KeyError:
        return None
