"""The "Atmosphere" tab with real SSRZ messages through the `GnssModel`: maps,
satellites and texts."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication

from dab.gnss import Observer
from dab.ssrz import SsrzDecoder
from tests.ssrz_vectors import (
    ZM002_LOW_RATE,
    ZM003_GRID_IONOSPHERE,
    ZM004_GRID_TROPOSPHERE,
    ZM005_REGIONAL_IONOSPHERE,
    ZM006_GLOBAL_VTEC,
    ZM007_REGIONAL_TROPOSPHERE,
    ZM009_TIME,
    ZM010_STREAM_ID,
    ZM011_SATELLITE_GROUPS,
    ZM012_METADATA,
    ZM013_GRID_3,
    ZM024_LEAP_SECOND,
)
from ui.atmosphere_model import CITIES
from ui.atmosphere_tab import AtmosphereTab, MapPanel
from ui.gnss_model import GnssModel

FRAMES = (
    ZM009_TIME,
    ZM010_STREAM_ID,
    ZM024_LEAP_SECOND,
    ZM011_SATELLITE_GROUPS,
    ZM012_METADATA,
    ZM013_GRID_3,
    ZM002_LOW_RATE,
    ZM006_GLOBAL_VTEC,
    ZM007_REGIONAL_TROPOSPHERE,
    ZM005_REGIONAL_IONOSPHERE,
    ZM004_GRID_TROPOSPHERE,
    ZM003_GRID_IONOSPHERE,
)


@pytest.fixture(scope="module")
def tab() -> AtmosphereTab:
    _app = QApplication.instance() or QApplication([])
    model = GnssModel(Observer(51.0, 10.0))
    tab = AtmosphereTab(model)
    decoder = SsrzDecoder()
    for frame in FRAMES:
        result = decoder.push(frame[3:-3])
        assert result is not None
        model.add_ssrz(result)
    tab.render()
    yield tab
    tab.deleteLater()


def test_every_satellite_can_be_chosen_and_is_listed(tab: AtmosphereTab) -> None:
    combo = tab._ionosphere_view
    # The global model and the 34 satellites of the low rate message.
    assert combo.count() == 35
    assert combo.itemData(1) == "G06"
    assert tab._satellite_table.rowCount() == 34
    assert tab._satellite_table.item(0, 0).text() == "G06"
    # Regional a00 of G06 and the mean of its 12 gridded values.
    assert tab._satellite_table.item(0, 1).text() == "-1.35"
    assert tab._satellite_table.item(0, 6).text() == "12/13"


def test_ionosphere_fields(tab: AtmosphereTab) -> None:
    field = tab._ionosphere_field()
    # The global model: some 8 TECU over Germany in the evening.
    assert 5.0 < np.nanmin(field.raster) < np.nanmax(field.raster) < 12.0
    tab._ionosphere_view.setCurrentIndex(tab._ionosphere_view.findData("E31"))
    field = tab._ionosphere_field()
    assert np.nanmax(field.values) == pytest.approx(0.36)
    assert field.levels == pytest.approx((-0.36, 0.36))
    # Only grid 3 is known: the raster covers north-western Germany.
    assert 0.0 < np.mean(np.isfinite(field.raster)) < 0.3
    tab.render()


def test_troposphere_fields(tab: AtmosphereTab) -> None:
    views = tab._troposphere_view
    wet = tab._troposphere_field()
    assert np.sum(np.isfinite(wet.values)) == 12
    assert 10.0 < np.nanmin(wet.values) < np.nanmax(wet.values) < 20.0  # cm
    views.setCurrentIndex(1)
    total = tab._troposphere_field()
    assert 240.0 < np.nanmin(total.values) < 250.0
    views.setCurrentIndex(2)
    factor = tab._troposphere_field()
    assert np.nanmax(factor.values) == pytest.approx(38.908)  # %
    tab.render()


def test_description(tab: AtmosphereTab) -> None:
    text = tab._info_label.text()
    assert "Provider 33 (BKG)" in text
    assert "1 grid with 13 stations in total" in text
    assert "dry component +0.97 % relative to the model" in text
    assert "18:39:37 UTC" in text  # 18:39:55 GPS time
    # Decimal separators only in numbers, every line a sentence.
    assert all(line.endswith(".") for line in text.splitlines())


def test_the_names_of_the_cities_lie_above_the_stations_in_a_halo() -> None:
    _app = QApplication.instance() or QApplication([])
    panel = MapPanel(None)
    labels = panel._city_labels
    assert [label.toPlainText() for label in labels] == [name for name, _lat, _lon in CITIES]
    assert all(label.zValue() > panel._points.zValue() for label in labels)
    assert all(label.halo == QColor(0, 0, 0) for label in labels)  # the desktop's black plots
    # The light scheme of the touch surface.
    panel.set_city_colour(QColor("#5b6875"), QColor("#ffffff"))
    assert all(label.color == QColor("#5b6875") for label in labels)
    assert all(label.halo == QColor("#ffffff") for label in labels)
