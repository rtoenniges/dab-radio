"""The "Corrections" tab with the real messages of `tests.ssrz_vectors` through the
`GnssModel`: the table per satellite, the course over two epochs and a new position."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from dab.gnss import Observer
from dab.rtcm import decode_ephemeris
from dab.ssrz import SsrzDecoder
from tests.ssrz_vectors import EPHEMERIS_FRAMES, ZM001_HIGH_RATE
from tests.test_corrections import FRAMES
from ui.corrections_tab import CorrectionsTab
from ui.gnss_model import GnssModel

HANNOVER = Observer(52.5, 9.5, 100.0)


@pytest.fixture
def tab():
    _app = QApplication.instance() or QApplication([])
    model = GnssModel(HANNOVER)
    tab = CorrectionsTab(model)
    for frame in EPHEMERIS_FRAMES.values():
        model.add_ephemeris(decode_ephemeris(frame[3:-3]))
    decoder = SsrzDecoder()
    for frame in FRAMES:
        model.add_ssrz(decoder.push(frame[3:-3]))
    tab.render()
    yield tab, model, decoder
    tab.deleteLater()


def _column(tab: CorrectionsTab, column: int) -> list[str]:
    table = tab._table
    return [table.item(row, column).text() for row in range(table.rowCount())]


def test_table_lists_the_satellites_by_elevation(tab) -> None:
    tab, _model, _decoder = tab
    # Only the four whose ephemeris is known, highest first. Evaluated at the epoch of the
    # high rate message (18:40:05), 10 s after the reference epoch: G06 22.96° then 22.9°.
    assert _column(tab, 0) == ["E03", "G06", "C11", "R02"]
    row = _column(tab, 0).index("G06")
    assert tab._table.item(row, 1).text() == "22.9°"
    assert tab._table.item(row, 2).text() == "81"
    assert tab._table.item(row, 3).text() == "−3.175 m"
    assert tab._table.item(row, 8).text() == "2W −2.05 · 5Q −0.09"
    assert "Still without a matching ephemeris: G11" in tab._status_label.text()


def test_course_of_the_highest_satellite(tab) -> None:
    tab, _model, _decoder = tab
    assert tab._selected == "E03"
    x, y = tab._clock_plot.items["δC with fine correction"].getData()
    assert len(x) == 1
    assert y[0] == pytest.approx((-0.99 - 0.0005) * 100)


def test_new_position_restarts_the_course(tab) -> None:
    tab, model, _decoder = tab
    model.set_observer(Observer(48.14, 11.58, 560.0))  # Munich
    tab.render()
    history = model.history("E03")
    assert len(history.samples) == 1
    assert "48.1400° N" in tab._status_label.text()


def test_repeated_high_rate_epoch_is_not_evaluated_twice(tab) -> None:
    _tab, model, decoder = tab
    before = model.corrections_time
    model.add_ssrz(decoder.push(ZM001_HIGH_RATE[3:-3]))
    assert model.corrections_time == before
    assert len(model.history("G06").samples) == 1
