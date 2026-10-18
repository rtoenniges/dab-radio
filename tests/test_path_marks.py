"""Marks of the transmitters on the impulse response (`ui.path_marks`)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
import pytest
from PyQt6.QtGui import QColor, QFont, QFontMetricsF
from PyQt6.QtWidgets import QApplication

from dsp.tii import TiiPath, TiiTransmitter
from ui.path_marks import (
    LABEL_GAP,
    LABEL_PADDING_Y,
    TRIANGLE_GAP,
    TRIANGLE_HEIGHT,
    PathMarks,
    TransmitterColours,
)

APP = QApplication.instance() or QApplication([])

PALETTE = [QColor("#f0a020"), QColor("#5bd17a"), QColor("#c77dd0")]


@pytest.fixture
def plot():
    widget = pg.PlotWidget()
    widget.resize(800, 300)
    widget.setXRange(-250e-6, 750e-6, padding=0)
    widget.setYRange(-50.0, 5.0, padding=0)
    widget.show()
    APP.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()


def _names(colours: TransmitterColours, codes) -> list[str]:
    return [colours.colour(code).name() for code in codes]


def test_the_strongest_transmitter_takes_the_first_colour_and_keeps_it():
    colours = TransmitterColours(PALETTE)
    colours.note([TiiTransmitter(19, 4, 0.0, 0.9), TiiTransmitter(24, 3, -10.0, 0.1)])
    colours.note([TiiTransmitter(24, 3, 0.0, 0.6), TiiTransmitter(19, 4, -2.0, 0.4)])

    assert _names(colours, [(19, 4), (24, 3)]) == ["#f0a020", "#5bd17a"]
    colours.reset()
    colours.note([TiiTransmitter(24, 3, 0.0, 1.0)])
    assert _names(colours, [(24, 3)]) == ["#f0a020"]


def test_transmitters_are_filled_with_their_code_reflections_outlined(plot):
    colours = TransmitterColours(PALETTE)
    colours.note([TiiTransmitter(19, 4, 0.0, 0.99), TiiTransmitter(24, 3, -24.7, 0.01)])
    marks = PathMarks(plot, colours)
    marks.show(
        [
            TiiPath(0.0, 0.0, (19, 4)),
            TiiPath(19.7e-6, -15.0, (19, 4), echo=True),
            TiiPath(94.7e-6, -24.7, (24, 3)),
            TiiPath(152.8e-6, -30.5, None),
        ]
    )

    shown = [(label, filled, colour.name()) for label, filled, colour, _tier in marks.shown()]
    assert shown == [
        ("19-04", True, "#f0a020"),
        (None, False, "#f0a020"),
        ("24-03", True, "#5bd17a"),
        ("?", False, "#808080"),
    ]
    marks.clear()
    assert marks.shown() == []


def test_a_label_that_would_cover_another_moves_up(plot):
    """Two transmitters 2 µs apart at the same level: on 800 pixels for 1 ms
    their labels would overlap."""
    marks = PathMarks(plot, TransmitterColours(PALETTE))
    marks.show([TiiPath(0.0, 0.0, (19, 4)), TiiPath(2e-6, 0.0, (24, 3))])
    APP.processEvents()

    assert [tier for *_rest, tier in marks.shown()] == [0, 1]


def test_far_apart_labels_stay_on_their_paths(plot):
    marks = PathMarks(plot, TransmitterColours(PALETTE))
    marks.show([TiiPath(0.0, 0.0, (19, 4)), TiiPath(300e-6, 0.0, (24, 3))])
    APP.processEvents()

    assert [tier for *_rest, tier in marks.shown()] == [0, 0]


def test_the_axis_leaves_room_for_the_label_of_the_strongest_path(plot):
    marks = PathMarks(plot, TransmitterColours(PALETTE))
    assert marks.headroom_db(50.0) == 0.0  # nothing marked
    marks.show([TiiPath(0.0, 0.0, (19, 4))])

    headroom = marks.headroom_db(50.0)
    # As `SmoothedRange` sets it: the data range padded by a tenth on either side.
    low, high = -50.0, 0.0 + headroom
    pad = (high - low) * 0.1
    plot.setYRange(low - pad, high + pad, padding=0)
    APP.processEvents()

    view = plot.getViewBox()
    top = view.mapViewToScene(pg.Point(0.0, high + pad)).y()
    peak = view.mapViewToScene(pg.Point(0.0, 0.0)).y()
    bold = QFont()
    bold.setBold(True)
    label = QFontMetricsF(bold).height() + 2 * LABEL_PADDING_Y
    assert peak - top >= TRIANGLE_GAP + TRIANGLE_HEIGHT + LABEL_GAP + label
