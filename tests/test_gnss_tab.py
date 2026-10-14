"""The NTRIP caster as the "GNSS" tab controls it: switched on, it asks for GEPOS once,
names the mountpoint after it and takes the position a receiver reports by GGA."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import socket
import time
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QApplication

from dab.data_services import USER_APPLICATION_GEPOS
from dab.gnss import Observer
from net.ntrip import NtripCaster
from ui.gnss_model import GnssModel
from ui.gnss_tab import GnssTab

LOOPBACK = "127.0.0.1"
GGA = "$GNGGA,183000.00,5222.2720,N,00944.1060,E,4,18,0.6,56.3,M,46.1,M,1.0,0000"


class _LoopbackCaster(NtripCaster):
    """Listens on the loopback interface only: all interfaces would ask for a firewall
    rule."""

    def start(self, port: int = 0, host: str = "") -> None:
        super().start(port, LOOPBACK)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind((LOOPBACK, 0))
        return sock.getsockname()[1]


def _gepos() -> SimpleNamespace:
    return SimpleNamespace(
        scid=0x020, label="GEPOS", app_types=(USER_APPLICATION_GEPOS,), decodable=True
    )


@pytest.fixture
def tab():
    app = QApplication.instance() or QApplication([])
    caster = _LoopbackCaster(raw_wait_seconds=0.2)
    model = GnssModel(Observer(51.0, 10.0))
    tab = GnssTab(model, None, caster)
    tab._port_spin.setValue(_free_port())
    yield tab, model, caster, app
    caster.stop()
    tab.deleteLater()


def test_caster_asks_for_gepos_once_and_names_the_mountpoint(tab) -> None:
    tab, model, caster, _app = tab
    requested = []
    tab.component_requested.connect(requested.append)
    model.update_components([_gepos()])
    assert requested == []  # the caster is off

    tab._caster_check.click()
    assert caster.running
    assert requested == [0x020]
    assert [m.name for m in caster.status().mountpoints] == ["GEPOS"]
    assert "Running on port" in tab._caster_label.text()

    # Stopped by hand, GEPOS stays stopped while the ensemble stays the same.
    model.update_components([_gepos()])
    assert requested == [0x020]

    tab._caster_check.click()
    assert not caster.running
    assert tab._caster_label.text() == "Off."


def test_position_from_gga_of_a_connected_receiver(tab) -> None:
    tab, model, caster, _app = tab
    observers = []
    tab.observer_changed.connect(observers.append)
    model.update_components([_gepos()])
    tab._caster_check.click()
    tab._gga_check.setChecked(True)
    with socket.create_connection((LOOPBACK, caster.port), timeout=3.0) as sock:
        sock.sendall(f"GET /GEPOS HTTP/1.0\r\n\r\n{GGA}\r\n".encode("ascii"))
        deadline = time.monotonic() + 3.0
        while caster.status().latest_position is None and time.monotonic() < deadline:
            time.sleep(0.01)
        tab._poll_caster()
    observer = tab.observer()
    assert observer.latitude == pytest.approx(52.3712, abs=1e-4)
    assert observer.longitude == pytest.approx(9.7351, abs=1e-4)
    assert observer.height == pytest.approx(102.4, abs=0.5)
    assert observers and observers[-1] == observer
    assert "RTK fixed" in tab._caster_label.text()
