"""A road network for the tests of the traffic surfaces, built by
`tools/build_road_network.py` from a small OSM file: roads under the three TEC messages
of `tests.tpeg_vectors` (an L 389 at Laatzen, a street in Bremen, the A 71 north of
Erfurt) and the towns beside them."""

from __future__ import annotations

import sys
from pathlib import Path

import osmium
from osmium.osm import mutable

from dab.road_network import ROAD_NETWORK_FILENAME, RoadNetwork

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import build_road_network

_ROADS = (
    ({"highway": "secondary", "ref": "L 389"}, [(9.7940, 52.3135), (9.7990, 52.3080)]),
    ({"highway": "tertiary", "name": "Am Wall"}, [(8.8000, 53.0790), (8.8010, 53.0780)]),
    (
        {"highway": "motorway", "ref": "A 71"},
        [(11.1310, 51.1150), (11.0990, 51.0780), (11.0660, 51.0400)],
    ),
)
_PLACES = (
    ("Laatzen", "town", (9.8130, 52.3150)),
    ("Bremen", "city", (8.8070, 53.0760)),
    ("Erfurt", "city", (11.0300, 50.9800)),
    ("Hannover", "city", (9.7350, 52.3740)),
)


def build(directory: Path) -> Path:
    """Writes the OSM file and builds the network into `directory`; its path."""
    osm = directory / "roads.osm"
    with osmium.SimpleWriter(str(osm)) as writer:
        node = 1
        ways = []
        for tags, points in _ROADS:
            refs = []
            for point in points:
                writer.add_node(mutable.Node(id=node, location=point))
                refs.append(node)
                node += 1
            ways.append((tags, refs))
        for name, kind, point in _PLACES:
            writer.add_node(
                mutable.Node(id=node, location=point, tags={"place": kind, "name": name})
            )
            node += 1
        for way_id, (tags, refs) in enumerate(ways, start=1):
            writer.add_way(mutable.Way(id=way_id, nodes=refs, tags=tags))
    output = directory / ROAD_NETWORK_FILENAME
    build_road_network.build(osm, output)
    return output


def network(directory: Path) -> RoadNetwork:
    return RoadNetwork.load(build(directory))
