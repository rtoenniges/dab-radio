"""Builds the road network of the traffic map (`dab.road_network`) from the OpenStreetMap
extract of Germany, once per computer:

    uv run python tools/build_road_network.py

Downloads `germany-latest.osm.pbf` from Geofabrik (some 5 GB) into the build directory
outside OneDrive (Windows: %LOCALAPPDATA%\\dab-radio\\build\\osm, else
~/.cache/dab-radio/osm) unless it is there already (`--update` fetches a new one, `--pbf`
takes another file). Reads it in three passes with pyosmium: the ways of the roads,
borders and coast, then their nodes, then the cities and towns. Joins the pieces of each
layer end to end (`merge`), simplifies the lines (Douglas-Peucker, `FINE_TOLERANCE_M`, a
medium copy with `MEDIUM_TOLERANCE_M` and a coarse one of the main roads with
`COARSE_TOLERANCE_M`) and writes `road_network.npz` into the data directory of the radio
(where `ReceiverController` keeps its settings; `--output` writes elsewhere). Takes a few
minutes and some GB of memory; the file can also be built on a PC and copied to the
Raspberry Pi.

The data are © OpenStreetMap contributors under the ODbL; the file is not committed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from array import array
from pathlib import Path

import numpy as np
import osmium
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dab.road_network import (
    BORDER,
    CITY,
    COARSE,
    COARSE_LAYERS,
    COAST,
    FINE,
    FORMAT_VERSION,
    LAYERS,
    MEDIUM,
    MEDIUM_LAYERS,
    OSM_HIGHWAYS,
    ROAD_NETWORK_FILENAME,
    STATE,
    TOWN,
    array_name,
)

EXTRACT_URL = "https://download.geofabrik.de/europe/germany-latest.osm.pbf"
EXTRACT_NAME = "germany-latest.osm.pbf"
# A point of a line may lie this far from the simplified line: about the width of a road.
FINE_TOLERANCE_M = 8.0
# The medium copy for a region (`ui.road_map`), where a pixel is 100 m or more.
MEDIUM_TOLERANCE_M = 30.0
# The coarse copy for the whole country, where a pixel is some 500 m.
COARSE_TOLERANCE_M = 150.0
# The copies of the layers: detail, tolerance, layers.
_COPIES = (
    (FINE, FINE_TOLERANCE_M, LAYERS),
    (MEDIUM, MEDIUM_TOLERANCE_M, MEDIUM_LAYERS),
    (COARSE, COARSE_TOLERANCE_M, COARSE_LAYERS),
)
# Latitude at which degrees of longitude are converted to metres for the simplification.
REFERENCE_LATITUDE = 51.0
_M_PER_DEGREE = 111_195.0
_PLACES = {"city": CITY, "town": TOWN}
_DOWNLOAD_CHUNK = 1 << 20


def build_directory() -> Path:
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / "dab-radio" / "build" / "osm"
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "dab-radio" / "osm"


def data_directory() -> Path:
    """The radio's data directory, as the controller finds it."""
    from PyQt6.QtCore import QCoreApplication

    from ui.receiver_controller import APPLICATION_NAME, app_data_directory

    _app = QCoreApplication.instance() or QCoreApplication([sys.argv[0]])
    QCoreApplication.setApplicationName(APPLICATION_NAME)
    return app_data_directory()


def download(url: str, target: Path) -> None:
    """Fetches `url` into `target`, continuing an interrupted download."""
    partial = target.with_suffix(target.suffix + ".part")
    have = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "dab-radio road network build"})
    if have:
        request.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(request) as response, open(partial, "ab" if have else "wb") as file:
        total = have + int(response.headers.get("Content-Length", 0))
        done, shown = have, time.monotonic()
        while chunk := response.read(_DOWNLOAD_CHUNK):
            file.write(chunk)
            done += len(chunk)
            if time.monotonic() - shown > 5:
                print(f"  {done / 1e9:.2f} of {total / 1e9:.2f} GB", flush=True)
                shown = time.monotonic()
    partial.replace(target)


class _Lines:
    """The ways of one layer as node ids, until the nodes are known."""

    def __init__(self) -> None:
        self.refs = array("q")
        self.start = array("q", [0])

    def add(self, refs: list[int]) -> None:
        self.refs.extend(refs)
        self.start.append(len(self.refs))


def merge(lines: _Lines) -> _Lines:
    """Joins the lines that meet end to end where no third one meets them: OSM splits a
    road into pieces of a few points each (on average 2.5 for the motorways), and a piece
    that short leaves nothing to simplify."""
    refs = np.frombuffer(lines.refs, dtype=np.int64)
    start = np.frombuffer(lines.start, dtype=np.int64)
    count = len(start) - 1
    first, last = refs[start[:-1]], refs[start[1:] - 1]
    ends: dict[int, list[int]] = {}
    for line in range(count):
        ends.setdefault(int(first[line]), []).append(line)
        ends.setdefault(int(last[line]), []).append(line)
    used = np.zeros(count, dtype=bool)
    merged = _Lines()

    def follow(node: int, line: int) -> tuple[int, bool] | None:
        """The other line at `node` if exactly two meet there and it is still free;
        with whether it starts there."""
        meeting = ends.get(node, ())
        if len(meeting) != 2:
            return None
        other = meeting[0] if meeting[1] == line else meeting[1]
        if other == line or used[other]:
            return None
        return other, int(first[other]) == node

    for line in range(count):
        if used[line]:
            continue
        used[line] = True
        chain = list(refs[start[line] : start[line + 1]])
        for forward in (True, False):
            current = line
            while True:
                node = int(chain[-1] if forward else chain[0])
                found = follow(node, current)
                if found is None:
                    break
                other, starts_here = found
                used[other] = True
                piece = list(refs[start[other] : start[other + 1]])
                if forward:
                    chain.extend(piece[1:] if starts_here else piece[-2::-1])
                else:
                    piece = piece if not starts_here else piece[::-1]
                    chain[:0] = piece[:-1]
                current = other
        merged.add(chain)
    return merged


def read_ways(pbf: Path) -> tuple[dict[str, _Lines], osmium.IdTracker]:
    """Pass 1: the roads by layer, borders and coast."""
    lines = {name: _Lines() for name in LAYERS}
    tracker = osmium.IdTracker()
    wanted = [("highway", value) for value in OSM_HIGHWAYS]
    wanted += [("natural", "coastline"), ("boundary", "administrative")]
    processor = osmium.FileProcessor(str(pbf), osmium.osm.WAY).with_filter(
        osmium.filter.TagFilter(*wanted)
    )
    for way in processor:
        tags = way.tags
        layers = []
        layer = OSM_HIGHWAYS.get(tags.get("highway", ""))
        if layer is not None:
            layers.append(layer)
        if tags.get("natural") == "coastline":
            layers.append(COAST)
        if tags.get("boundary") == "administrative" and tags.get("maritime") != "yes":
            level = tags.get("admin_level")
            if level == "2":
                layers.append(BORDER)
            elif level == "4":
                layers.append(STATE)
        if not layers:
            continue
        refs = [node.ref for node in way.nodes]
        if len(refs) < 2:
            continue
        tracker.add_references(way)
        for layer in layers:
            lines[layer].add(refs)
    return lines, tracker


def read_nodes(pbf: Path, tracker: osmium.IdTracker) -> tuple[np.ndarray, np.ndarray]:
    """Pass 2: the coordinates of the nodes the ways use, by sorted id."""
    ids = array("q")
    coordinates = array("d")
    processor = osmium.FileProcessor(str(pbf), osmium.osm.NODE).with_filter(tracker.id_filter())
    for node in processor:
        location = node.location
        ids.append(node.id)
        coordinates.append(location.lon)
        coordinates.append(location.lat)
    ids_array = np.frombuffer(ids, dtype=np.int64)
    xy = np.frombuffer(coordinates, dtype=np.float64).reshape(-1, 2)
    order = np.argsort(ids_array, kind="stable")
    return ids_array[order], xy[order]


def read_places(pbf: Path) -> list[tuple[str, float, float, int, int]]:
    """Pass 3: cities and towns with their names and population."""
    places = []
    processor = osmium.FileProcessor(str(pbf), osmium.osm.NODE).with_filter(
        osmium.filter.TagFilter(*[("place", value) for value in _PLACES])
    )
    for node in processor:
        name = node.tags.get("name")
        if not name or not node.location.valid():
            continue
        digits = "".join(c for c in node.tags.get("population", "") if c.isdigit())
        population = int(digits) if digits and len(digits) < 10 else 0
        rank = _PLACES[node.tags["place"]]
        places.append((name, node.location.lon, node.location.lat, rank, population))
    return places


@njit(cache=True)
def _simplify(x: np.ndarray, y: np.ndarray, start: np.ndarray, tolerance: float) -> np.ndarray:
    """Douglas-Peucker for each line (`start` as in `dab.road_network`): which points stay."""
    keep = np.zeros(len(x), dtype=np.bool_)
    stack = np.empty((len(x) + 2, 2), dtype=np.int64)
    for line in range(len(start) - 1):
        first, last = start[line], start[line + 1] - 1
        keep[first] = True
        keep[last] = True
        stack[0, 0], stack[0, 1] = first, last
        depth = 1
        while depth:
            depth -= 1
            a, b = stack[depth, 0], stack[depth, 1]
            if b - a < 2:
                continue
            dx, dy = x[b] - x[a], y[b] - y[a]
            length = np.sqrt(dx * dx + dy * dy)
            worst, at = -1.0, -1
            for i in range(a + 1, b):
                if length > 0:
                    d = abs(dy * (x[i] - x[a]) - dx * (y[i] - y[a])) / length
                else:
                    d = np.sqrt((x[i] - x[a]) ** 2 + (y[i] - y[a]) ** 2)
                if d > worst:
                    worst, at = d, i
            if worst > tolerance:
                keep[at] = True
                stack[depth, 0], stack[depth, 1] = a, at
                stack[depth + 1, 0], stack[depth + 1, 1] = at, b
                depth += 2
    return keep


def assemble(
    lines: _Lines, ids: np.ndarray, xy: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """The lines of a layer as points (float32 lon/lat) and starts, simplified; lines
    with a node outside the extract keep only the points that are known."""
    refs = np.frombuffer(lines.refs, dtype=np.int64)
    start = np.frombuffer(lines.start, dtype=np.int64)
    position = np.searchsorted(ids, refs)
    position = np.minimum(position, len(ids) - 1)
    known = ids[position] == refs
    points = xy[position]
    scale = np.cos(np.radians(REFERENCE_LATITUDE))
    keep = _simplify(
        points[:, 0] * scale * _M_PER_DEGREE, points[:, 1] * _M_PER_DEGREE, start, tolerance
    )
    keep &= known
    # Each line keeps the count of its points that stay; lines left with fewer than two go.
    if len(start) < 2:
        return np.zeros((0, 2), np.float32), np.zeros(1, np.int64)
    counts = np.add.reduceat(keep.astype(np.int64), start[:-1])
    whole = counts >= 2
    line_of_point = np.repeat(np.arange(len(start) - 1), np.diff(start))
    keep &= whole[line_of_point]
    new_start = np.concatenate([[0], np.cumsum(counts[whole])]).astype(np.int64)
    return points[keep].astype(np.float32), new_start


def build(pbf: Path, output: Path) -> None:
    started = time.monotonic()
    print(f"Reading the roads of {pbf.name} ...", flush=True)
    lines, tracker = read_ways(pbf)
    print(f"  {sum(len(x.start) - 1 for x in lines.values())} lines")
    print("Reading their nodes ...", flush=True)
    ids, xy = read_nodes(pbf, tracker)
    print(f"  {len(ids)} nodes")
    print("Reading cities and towns ...", flush=True)
    places = read_places(pbf)
    print(f"  {len(places)} places")

    arrays: dict[str, np.ndarray] = {}
    for name in LAYERS:
        pieces = len(lines[name].start) - 1
        merged = merge(lines[name])
        line = f"  {name}: {pieces} pieces in {len(merged.start) - 1} lines"
        for detail, tolerance, names in _COPIES:
            if name in names:
                points, start = assemble(merged, ids, xy, tolerance)
                arrays[array_name(name, detail, "xy")] = points
                arrays[array_name(name, detail, "start")] = start
                line += f", {detail} {len(points)} points"
        print(line, flush=True)
    arrays["place_name"] = np.array([p[0] for p in places], dtype=str)
    arrays["place_xy"] = np.array([(p[1], p[2]) for p in places], dtype=np.float32).reshape(-1, 2)
    arrays["place_rank"] = np.array([p[3] for p in places], dtype=np.int8)
    arrays["place_population"] = np.array([p[4] for p in places], dtype=np.int32)
    arrays["version"] = np.array(FORMAT_VERSION)
    modified = time.strftime("%d.%m.%Y", time.localtime(pbf.stat().st_mtime))
    arrays["source"] = np.array(f"{pbf.name} vom {modified}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".part.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)
    size = output.stat().st_size / 1e6
    print(f"Wrote {output}: {size:.1f} MB in {time.monotonic() - started:.0f} s")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pbf", type=Path, help="an OSM extract already at hand instead of the download"
    )
    parser.add_argument("--update", action="store_true", help="download the extract anew")
    parser.add_argument("--output", type=Path, help="file to write instead of the data directory")
    arguments = parser.parse_args(argv[1:])
    pbf = arguments.pbf
    if pbf is None:
        pbf = build_directory() / EXTRACT_NAME
        if arguments.update or not pbf.exists():
            pbf.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {EXTRACT_URL} ...", flush=True)
            download(EXTRACT_URL, pbf)
    output = arguments.output or data_directory() / ROAD_NETWORK_FILENAME
    build(pbf, output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
