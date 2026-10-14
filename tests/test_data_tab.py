"""The "Data services" tab: check boxes that start and stop the decoding, several
services at once, and the details of the one clicked."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from dab.data_services import data_components
from dsp.packet import AddressStatistics, PacketStatistics
from dsp.pipeline import DataComponentStatus
from tests.test_data_services import _ensemble
from ui.data_tab import DataTab

EPG_ROW, GEPOS_ROW = 0, 1


@pytest.fixture
def tab() -> DataTab:
    _app = QApplication.instance() or QApplication([])
    tab = DataTab()
    tab.update_components(data_components(_ensemble()))
    yield tab
    tab.deleteLater()


def _statuses(tab: DataTab, *rows: int) -> tuple[DataComponentStatus, ...]:
    components = [tab._components[row] for row in rows]
    return tuple(DataComponentStatus(c.scid, c) for c in components)


def _statistics(address: int) -> PacketStatistics:
    return PacketStatistics(
        seconds=1.0,
        fec=True,
        fec_locked=True,
        fec_frames=1,
        fec_corrected_bytes=0,
        fec_uncorrectable_rows=0,
        crc_errors_before_fec=0,
        padding_packets=0,
        addresses={address: AddressStatistics(packets=10)},
    )


def test_check_box_starts_and_stops_the_decoding(tab: DataTab) -> None:
    toggled = []
    tab.component_toggled.connect(lambda scid, on: toggled.append((scid, on)))
    item = tab._table.item(GEPOS_ROW, 0)

    item.setCheckState(Qt.CheckState.Checked)
    item.setCheckState(Qt.CheckState.Unchecked)

    assert toggled == [(0x020, True), (0x020, False)]


def test_several_services_are_decoded_and_the_last_started_is_shown(tab: DataTab) -> None:
    tab.set_decoding(_statuses(tab, EPG_ROW))
    tab.set_decoding(_statuses(tab, EPG_ROW, GEPOS_ROW))

    checked = [tab._table.item(row, 0).checkState() for row in (EPG_ROW, GEPOS_ROW)]
    assert checked == [Qt.CheckState.Checked, Qt.CheckState.Checked]
    assert tab._shown_scid == 0x020
    assert "Decoding “GEPOS”" in tab._status_label.text()
    assert "Also decoding: “EPG Deutschland”." in tab._status_label.text()


def test_details_follow_the_service_clicked(tab: DataTab) -> None:
    tab.set_decoding(_statuses(tab, EPG_ROW, GEPOS_ROW))
    tab.update_statistics(30, _statistics(3))
    tab.update_statistics(32, _statistics(2))
    assert tab._address_table.item(0, 1).text() == "GEPOS (GEPOS (GNSS, RTCM 3))"

    tab._on_cell_clicked(EPG_ROW, 3)

    assert tab._shown_scid == 0x022
    assert tab._address_table.item(0, 1).text() == "EPG Deutschland (SPI (programme guide))"


def test_stopping_the_service_shown_shows_another(tab: DataTab) -> None:
    toggled = []
    tab.component_toggled.connect(lambda scid, on: toggled.append((scid, on)))
    tab.set_decoding(_statuses(tab, EPG_ROW, GEPOS_ROW))

    tab.set_decoding(_statuses(tab, EPG_ROW))

    assert tab._table.item(GEPOS_ROW, 0).checkState() == Qt.CheckState.Unchecked
    assert tab._shown_scid == 0x022
    assert toggled == []  # confirming the state is no request

    tab._stop_button.click()

    assert toggled == [(0x022, False)]
