"""The names of the towns on the road map (`ui.road_map._Labels`): one item paints them
all, each with a halo, where the `MapLabel`s stood; and the pixel size of a view box
without pyqtgraph's inverse transforms."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QTransform
from PyQt6.QtWidgets import QApplication

from dab.road_network import TOWN, Place
from ui.plots import MAP_LABEL_Z
from ui.road_map import _Labels, pixel_size

APP = QApplication.instance() or QApplication([])


def _painted(labels: _Labels, size: tuple[int, int], transform: QTransform) -> QImage:
    image = QImage(*size, QImage.Format.Format_ARGB32)
    image.fill(QColor(0, 0, 0, 0))
    painter = QPainter(image)
    painter.setWorldTransform(transform)
    labels.paint(painter, None)
    painter.end()
    return image


def _colours(image: QImage) -> set[tuple[int, int, int]]:
    return {
        image.pixelColor(x, y).getRgb()[:3]
        for x in range(image.width())
        for y in range(image.height())
        if image.pixelColor(x, y).alpha() == 255
    }


def test_the_names_stand_at_their_towns_with_a_halo_and_follow_the_view() -> None:
    labels = _Labels(QFont("Arial", 12))
    labels.set_colours(QColor(10, 200, 30), QColor(200, 10, 30))
    labels.set_places([Place("Celle", 10.0, 52.6, TOWN, 70_000)])
    assert labels.names == ["Celle"] and labels.zValue() == MAP_LABEL_Z
    assert labels.boundingRect().contains(10.0, 52.6)

    # Degrees to pixels: the town at (100, 150) of the image, y downwards.
    to_pixels = QTransform(10.0, 0.0, 0.0, -10.0, 0.0, 676.0)
    assert to_pixels.map(10.0, 52.6) == (100.0, 150.0)
    image = _painted(labels, (300, 200), to_pixels)
    rows = np.array([[image.pixelColor(x, y).alpha() for x in range(300)] for y in range(200)])
    ys, xs = np.nonzero(rows)
    # Up and to the right of the town, as the MapLabels' anchor (0, 1) placed them.
    assert xs.min() >= 98 and xs.min() <= 104 and xs.max() < 200
    assert 138 <= ys.max() <= 152 and ys.min() > 110  # no descenders in "Celle"
    assert {(10, 200, 30), (200, 10, 30)} <= _colours(image)

    # Zoomed in: the name moves with the town but keeps its size.
    closer = QTransform(20.0, 0.0, 0.0, -20.0, -100.0, 1202.0)
    assert closer.map(10.0, 52.6) == (100.0, 150.0)
    assert _painted(labels, (300, 200), closer) == image
    aside = QTransform(10.0, 0.0, 0.0, -10.0, 50.0, 676.0)
    moved = _painted(labels, (300, 200), aside)
    assert np.array_equal(
        np.array([[moved.pixelColor(x + 50, y).alpha() for x in range(200)] for y in range(200)]),
        rows[:, :200],
    )


def test_new_colours_paint_the_names_anew() -> None:
    labels = _Labels(None)
    labels.set_places([Place("Celle", 10.0, 52.6, TOWN, 70_000)])
    transform = QTransform(10.0, 0.0, 0.0, -10.0, 0.0, 676.0)
    labels.set_colours(QColor(10, 200, 30), QColor(200, 10, 30))
    first = _colours(_painted(labels, (300, 200), transform))
    labels.set_colours(QColor(1, 2, 3), QColor(4, 5, 6))
    second = _colours(_painted(labels, (300, 200), transform))
    assert {(10, 200, 30), (200, 10, 30)} <= first
    assert {(1, 2, 3), (4, 5, 6)} <= second and not {(10, 200, 30), (200, 10, 30)} & second
    assert labels.colour == QColor(1, 2, 3) and labels.halo == QColor(4, 5, 6)


def test_the_pixel_size_is_the_range_over_the_size_of_the_view() -> None:
    plot = pg.PlotWidget()
    plot.resize(400, 300)
    plot.show()
    APP.processEvents()
    view_box = plot.getViewBox()
    view_box.setRange(xRange=(6.0, 14.0), yRange=(47.0, 55.0), padding=0)
    APP.processEvents()

    width, height = pixel_size(view_box)
    assert width == pytest.approx(8.0 / view_box.width())
    assert height == pytest.approx(8.0 / view_box.height())
    assert (width, height) == pytest.approx(view_box.viewPixelSize(), rel=1e-6)
    plot.close()
