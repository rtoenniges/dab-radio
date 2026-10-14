"""The traffic flow on the map: each direction of a road drawn a few pixels to the right of
its direction of travel, all stretches at once, levels left out on request."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from PyQt6.QtWidgets import QApplication

from ui.flow_map import FlowLayer

APP = QApplication.instance() or QApplication([])


@pytest.fixture
def layer():
    """A layer moving lines 4 pixels, on a scale of one unit per pixel."""
    layer = FlowLayer(pg.PlotWidget(), offset=4.0)
    layer._scale = (1.0, 1.0)
    return layer


def _points(x: np.ndarray, y: np.ndarray) -> list[tuple[float, float]]:
    return [(float(a), float(b)) for a, b in zip(x, y)]


def test_a_line_moves_to_the_right_of_its_direction(layer) -> None:
    east = np.array([(0.0, 0.0), (10.0, 0.0)])
    north = np.array([(20.0, 0.0), (20.0, 10.0)])
    x, y = layer._joined([east, north])
    # Right of east is south, right of north is east (y points up); NaN after each line.
    assert _points(x[:2], y[:2]) == [(0.0, -4.0), (10.0, -4.0)]
    assert _points(x[3:5], y[3:5]) == [(24.0, 0.0), (24.0, 10.0)]
    assert np.isnan(x[2]) and np.isnan(x[5])
    assert len(x) == 6


def test_a_corner_moves_along_the_mean_of_its_two_segments(layer) -> None:
    corner = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
    x, y = layer._joined([corner])
    side = 4.0 / np.sqrt(2.0)
    assert (x[1], y[1]) == pytest.approx((10.0 + side, -side))
    # The ends keep the normal of their one segment.
    assert (x[0], y[0]) == pytest.approx((0.0, -4.0))
    assert (x[2], y[2]) == pytest.approx((14.0, 10.0))


def test_a_single_point_and_a_repeated_one_stay_where_they_are(layer) -> None:
    single = np.array([(5.0, 5.0)])
    repeated = np.array([(0.0, 0.0), (0.0, 0.0)])
    x, y = layer._joined([single, repeated])
    assert _points(x[:1], y[:1]) == [(5.0, 5.0)]
    assert _points(x[2:4], y[2:4]) == [(0.0, 0.0), (0.0, 0.0)]


def test_the_offset_is_in_pixels_whatever_the_scale(layer) -> None:
    layer._scale = (0.01, 0.02)  # degrees per pixel across and up
    x, y = layer._joined([np.array([(9.0, 52.0), (9.1, 52.0)])])
    assert y[0] == pytest.approx(52.0 - 4 * 0.02)
    assert x[0] == pytest.approx(9.0)


def test_no_lines_give_empty_arrays(layer) -> None:
    x, y = layer._joined([])
    assert len(x) == 0 and len(y) == 0
