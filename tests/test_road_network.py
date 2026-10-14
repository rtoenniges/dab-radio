"""The road network of the traffic map: built by `tools/build_road_network.py` from a
small OSM file written here, then read by `dab.road_network`: the layers, the simplified
lines, the lines in view and the places."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import osmium
import pytest
from osmium.osm import mutable

from dab.road_network import (
    BORDER,
    CITY,
    COARSE,
    COAST,
    LINK,
    MAJOR,
    MEDIUM,
    MOTORWAY,
    SECONDARY,
    STATE,
    TERTIARY,
    TOWN,
    RoadNetwork,
    load_if_present,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import build_road_network

# Metres per degree of latitude; at 52° N a degree of longitude is 0.616 of it.
M = 111_195.0


def _write_osm(path: Path) -> None:
    """A motorway west to east at 52° N with a kink of 3 m (simplified away) and one of
    500 m (kept), a Bundesstraße crossing it, a Landesstraße in two pieces, a district
    road, a slip road without number, coast, borders on land and at sea, and a city and
    a town."""
    nodes = {
        1: (9.00, 52.0),
        2: (9.05, 52.0 + 3 / M),
        3: (9.10, 52.0),
        4: (9.15, 52.0 + 500 / M),
        5: (9.20, 52.0),
        11: (9.10, 51.9),
        12: (9.10, 52.1),
        21: (9.30, 52.3),
        22: (9.35, 52.3),
        31: (9.20, 52.0),
        32: (9.21, 52.01),
        41: (8.0, 54.0),
        42: (8.5, 54.1),
        51: (6.0, 51.0),
        52: (6.0, 52.0),
        61: (7.0, 54.5),
        62: (8.0, 54.5),
        71: (10.0, 52.0),
        72: (10.0, 53.0),
        81: (9.40, 52.2),
        82: (9.45, 52.2 + 500 / M),
        83: (9.50, 52.2),
    }
    with osmium.SimpleWriter(str(path)) as writer:
        for node_id, location in nodes.items():
            writer.add_node(mutable.Node(id=node_id, location=location))
        writer.add_node(
            mutable.Node(
                id=100,
                location=(9.72, 52.37),
                tags={"place": "city", "name": "Hannover", "population": "535.061"},
            )
        )
        writer.add_node(
            mutable.Node(id=101, location=(10.08, 52.62), tags={"place": "town", "name": "Celle"})
        )
        writer.add_node(
            mutable.Node(id=102, location=(9.5, 52.5), tags={"place": "village", "name": "Dorf"})
        )
        ways = [
            (1000, [1, 2, 3, 4, 5], {"highway": "motorway", "ref": "A 7"}),
            (1001, [11, 3, 12], {"highway": "primary", "ref": "B  3;B 6"}),
            (1002, [21, 22], {"highway": "tertiary", "name": "Hauptstraße"}),
            (1003, [31, 32], {"highway": "motorway_link"}),
            (1004, [41, 42], {"natural": "coastline"}),
            (1005, [51, 52], {"boundary": "administrative", "admin_level": "2"}),
            (1006, [61, 62], {"boundary": "administrative", "admin_level": "2", "maritime": "yes"}),
            (1007, [71, 72], {"boundary": "administrative", "admin_level": "4"}),
            (1008, [1, 5], {"highway": "residential", "name": "Gasse"}),
            (1009, [81, 82], {"highway": "secondary", "ref": "L 190"}),
            (1010, [82, 83], {"highway": "secondary", "ref": "L 190"}),
        ]
        for way_id, refs, tags in ways:
            writer.add_way(mutable.Way(id=way_id, nodes=refs, tags=tags))


@pytest.fixture(scope="module")
def network(tmp_path_factory) -> RoadNetwork:
    directory = tmp_path_factory.mktemp("roads")
    osm = directory / "roads.osm"
    _write_osm(osm)
    output = directory / "road_network.npz"
    build_road_network.build(osm, output)
    assert load_if_present(directory) is not None
    return RoadNetwork.load(output)


def test_the_layers_take_roads_borders_and_coast_by_their_tags(network) -> None:
    assert network.line_count(MOTORWAY) == 1
    assert network.line_count(MAJOR) == 1
    assert network.line_count(SECONDARY) == 1  # its two pieces joined
    assert network.line_count(TERTIARY) == 1
    assert network.line_count(LINK) == 1
    assert network.line_count(COAST) == 1
    assert network.line_count(BORDER) == 1  # not the one at sea
    assert network.line_count(STATE) == 1


def test_lines_are_simplified_and_cut_to_the_view(network) -> None:
    everywhere = (0.0, 40.0, 20.0, 60.0)
    x, y = network.lines(MOTORWAY, everywhere)
    # The kink of 3 m goes, the one of 500 m stays; a NaN ends the line.
    assert list(x[:-1]) == pytest.approx([9.0, 9.1, 9.15, 9.2])
    assert np.isnan(x[-1]) and np.isnan(y[-1])
    # The medium (30 m) and coarse copies (150 m) keep the 500 m too.
    for detail in (MEDIUM, COARSE):
        x, _y = network.lines(MOTORWAY, everywhere, detail)
        assert len(x) == 5
    # A layer without such a copy gives its fine lines.
    x, _y = network.lines(TERTIARY, everywhere, COARSE)
    assert len(x) == 3

    x, _y = network.lines(MOTORWAY, (10.0, 51.0, 11.0, 53.0))  # east of it
    assert len(x) == 0
    x, _y = network.lines(TERTIARY, everywhere)
    assert len(x) == 3


def test_places_are_cities_and_towns_largest_first(network) -> None:
    assert [(p.name, p.rank) for p in network.places_in((0, 40, 20, 60))] == [
        ("Hannover", CITY),
        ("Celle", TOWN),
    ]
    assert [p.name for p in network.places_in((0, 40, 20, 60), rank=CITY)] == ["Hannover"]
    assert network.places[0].population == 535061


def test_an_outdated_file_is_not_used(tmp_path) -> None:
    np.savez(tmp_path / "road_network.npz", version=np.array(0))
    assert load_if_present(tmp_path) is None
    assert load_if_present(tmp_path / "missing") is None


def test_pieces_meeting_end_to_end_are_joined() -> None:
    lines = build_road_network._Lines()
    for refs in ([1, 2, 3], [5, 4, 3], [5, 6], [3, 7], [8, 9]):
        lines.add(refs)
    merged = build_road_network.merge(lines)
    refs = np.frombuffer(merged.refs, dtype=np.int64)
    start = np.frombuffer(merged.start, dtype=np.int64)
    chains = [refs[start[i] : start[i + 1]].tolist() for i in range(len(start) - 1)]
    # Three lines meet at node 3: none is joined there; 5-4-3 and 5-6 meet alone at 5.
    assert chains == [[1, 2, 3], [6, 5, 4, 3], [3, 7], [8, 9]]


def test_the_map_draws_more_classes_closer_in(network) -> None:
    import pyqtgraph as pg
    from PyQt6.QtWidgets import QApplication

    from ui.road_map import LOCAL, OVERVIEW, REGION, RoadMap, level_for

    assert [level_for(w) for w in (11.0, 2.0, 0.5)] == [OVERVIEW, REGION, LOCAL]
    _app = QApplication.instance() or QApplication([])
    plot = pg.PlotWidget()
    roads = RoadMap(plot)
    plot.setXRange(5.0, 16.0, padding=0)
    plot.setYRange(47.0, 55.5, padding=0)
    roads.set_network(network)
    assert roads.level == OVERVIEW
    assert len(roads.lines(TERTIARY)[0]) == 0
    plot.setXRange(9.0, 9.5, padding=0)
    plot.setYRange(52.0, 52.4, padding=0)
    roads.redraw()
    assert roads.level == LOCAL
    x, _y = roads.lines(TERTIARY)
    assert len(x) == 3  # the Hauptstraße and its gap


def _shown_map(network):
    """A plot 400 x 300 pixels over the motorway at 52° N, with the network beneath."""
    import pyqtgraph as pg
    from PyQt6.QtWidgets import QApplication

    from ui.road_map import RoadMap

    app = QApplication.instance() or QApplication([])
    plot = pg.PlotWidget()
    plot.hideAxis("left")
    plot.hideAxis("bottom")
    plot.resize(400, 300)
    plot.show()
    roads = RoadMap(plot)
    plot.getViewBox().setRange(xRange=(8.9, 9.3), yRange=(51.85, 52.35), padding=0)
    app.processEvents()
    roads.set_network(network)
    _finish(app, roads)
    return app, plot, roads


def _finish(app, roads) -> None:
    """Runs the event loop until the picture is painted."""
    for _ in range(1000):
        if not roads.painting:
            break
        app.processEvents()
    assert not roads.painting


def _colour_at(plot, longitude: float, latitude: float) -> tuple[int, int, int]:
    """The brightest colour within a pixel of a place on the plot."""
    from PyQt6.QtCore import QPointF

    image = plot.grab().toImage()
    scene = plot.getViewBox().mapViewToScene(QPointF(longitude, latitude))
    point = plot.mapFromScene(scene)
    ratio = image.devicePixelRatio()
    colours = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            colour = image.pixelColor(
                round((point.x() + dx) * ratio), round((point.y() + dy) * ratio)
            )
            colours.append((colour.red(), colour.green(), colour.blue()))
    return max(colours, key=sum)


def test_the_roads_are_painted_once_into_a_picture_where_they_lie(network) -> None:
    from ui.road_map import RoadColours

    _app, plot, roads = _shown_map(network)
    assert roads.picture is not None
    # The motorway at 52° N is where it lies, not mirrored about the middle of the view.
    assert _colour_at(plot, 9.05, 52.0) == RoadColours().motorway
    assert _colour_at(plot, 9.05, 52.2) == (0, 0, 0)
    plot.close()


def test_moving_the_view_a_little_paints_nothing_anew_zooming_in_does(network) -> None:
    from ui.road_map import RESCALE_FACTOR

    app, plot, roads = _shown_map(network)
    picture = roads.picture
    view_box = plot.getViewBox()
    (x0, x1), (y0, y1) = view_box.viewRange()
    view_box.setRange(xRange=(x0 + 0.02, x1 + 0.02), yRange=(y0, y1), padding=0)
    app.processEvents()
    assert not roads._timer.isActive()
    assert roads.picture is picture
    # The motorway moved with the view.
    assert _colour_at(plot, 9.05, 52.0) != (0, 0, 0)

    # Closer in by more than the factor: painted anew once the view rests.
    view_box.scaleBy((1 / (RESCALE_FACTOR * 1.1),) * 2)
    app.processEvents()
    assert roads._timer.isActive()
    roads.redraw()
    _finish(app, roads)
    assert roads.picture is not picture
    plot.close()


def test_the_picture_is_painted_in_slices_and_a_new_view_begins_it_anew(
    network, monkeypatch
) -> None:
    import ui.road_map
    from ui.road_map import RoadColours

    monkeypatch.setattr(ui.road_map, "SLICE_POINTS", 2)
    app, plot, roads = _shown_map(network)
    picture = roads.picture
    roads.redraw()
    roads._paint_slice()
    # The picture shown stays until the new one is done.
    assert roads.painting
    assert roads.picture is picture
    roads.redraw()
    _finish(app, roads)
    assert roads.picture is not picture
    # The slices meet: the motorway runs on between 9.10° and 9.15° E, where one
    # slice of two points ends and the next begins.
    assert _colour_at(plot, 9.125, 52.0 + 250 / M) == RoadColours().motorway
    plot.close()
