"""The TMC location table (`dab.tmc_locations`) read from a made-up download in the
exchange format (`tests.tmc_table`): points, names, roads, the extent of a location in
either direction, the own file and the import script."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from dab.tmc_locations import (
    TMC_LOCATIONS_FILENAME,
    TmcLocationTable,
    load_if_present,
    road_text,
)
from dab.tpeg_messages import TmcLocation
from tests import tmc_table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import import_tmc_locations


def _location(code: int, extent: int, positive: bool, **changes) -> TmcLocation:
    values = {
        "location": code,
        "country": 0xD,
        "table": 1,
        "positive_direction": positive,
        "both_directions": False,
        "extent": extent,
        "extended_country_code": 0xE0,
    }
    values.update(changes)
    return TmcLocation(**values)


@pytest.fixture
def table(tmp_path) -> TmcLocationTable:
    return tmc_table.table(tmp_path)


def test_points_with_road_name_and_kind(table) -> None:
    assert (table.country, table.extended_country, table.table, table.version) == (
        "D",
        "E0",
        1,
        "22.0",
    )
    point = table.point(tmc_table.TEC_A71_PRIMARY)
    assert point.road == "A 71"
    assert point.label() == "Testkreuz Nord junction"
    assert (point.longitude, point.latitude) == pytest.approx((11.066, 51.041))
    # The road of a point on a segment comes from the segment.
    assert table.point(tmc_table.TFP_PRIMARY).road == "B 213"
    # A crossing keeps its name without a short form.
    assert table.point(tmc_table.TEC_TOWN).label() == "Am Testwall"
    assert road_text("A7") == "A 7" and road_text("B 245A") == "B 245A"


def test_the_extent_follows_the_direction_bit(table) -> None:
    forward = table.stretch(_location(tmc_table.TEC_A71_PRIMARY, 1, positive=True))
    assert [p.code for p in forward] == [tmc_table.TEC_A71_PRIMARY, 90002]
    backward = table.stretch(_location(tmc_table.TFP_PRIMARY, 7, positive=False))
    assert [p.code for p in backward] == list(tmc_table.TFP_CHAIN)
    # Where the table ends, the extent ends.
    short = table.stretch(_location(tmc_table.TEC_A71_PRIMARY, 3, positive=True))
    assert len(short) == 2
    assert table.stretch(_location(12345, 1, positive=True)) == []


def test_only_locations_of_its_own_table(table) -> None:
    assert table.serves(_location(1, 0, True))
    assert table.serves(_location(1, 0, True, extended_country_code=None))
    assert not table.serves(_location(1, 0, True, table=2))
    assert not table.serves(_location(1, 0, True, country=0xA))
    assert not table.serves(_location(1, 0, True, extended_country_code=0xE1))


def test_from_a_folder_and_through_the_own_file(tmp_path) -> None:
    folder = tmp_path / "LCL" / "inner"
    folder.mkdir(parents=True)
    for name, text in tmc_table.files().items():
        (folder / name).write_text(text, encoding="utf-8")
    table = TmcLocationTable.read_exchange(tmp_path / "LCL")
    table.save(tmp_path / TMC_LOCATIONS_FILENAME)
    loaded = load_if_present(tmp_path)
    assert len(loaded) == len(table) == 11
    assert loaded.point(90002) == table.point(90002)
    location = _location(tmc_table.TFP_PRIMARY, 7, positive=False)
    assert loaded.stretch(location) == table.stretch(location)
    with pytest.raises(ValueError):
        TmcLocationTable.read_exchange(tmp_path / "LCL" / "missing")


def test_the_import_script_writes_the_file(tmp_path) -> None:
    source = tmc_table.write_zip(tmp_path / "download.zip")
    output = tmp_path / "data" / TMC_LOCATIONS_FILENAME
    assert import_tmc_locations.main(["import", str(source), "--output", str(output)]) == 0
    assert load_if_present(output.parent) is not None
    assert load_if_present(tmp_path / "nowhere") is None
