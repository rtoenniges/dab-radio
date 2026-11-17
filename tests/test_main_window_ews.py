"""The emergency warning playing on the desktop: a box that is not modal, follows the
warning and breaks it off."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

from dab.dls import DynamicLabel
from dab.ensemble_cache import SOURCE_SCAN, EnsembleCache, EnsembleInfo
from dab.ews import EwsAlert, EwsState, Stage
from tests.fake_receiver import SERVICES_5C
from ui import receiver_controller
from ui.main_window import MainWindow

EID_5C = 0x10BC
ASA_DE = 0x10C4
SERVICES = (*SERVICES_5C, (ASA_DE, "ASA DE"))
ALERT = EwsAlert(
    False, EID_5C, Stage.LEVEL_1_START, 1, whole_area=True, subchannel_id=9, sid=ASA_DE,
    receivable=True,
)  # fmt: skip
PLAYING = EwsState(True, EID_5C, (ALERT,), frozenset({9}), ((9, ASA_DE),))
QUIET = EwsState(True, EID_5C)


@pytest.fixture
def window(data_directory, make_controller, fake_workers):
    _app = QApplication.instance() or QApplication([])
    cache = EnsembleCache(data_directory / receiver_controller.ENSEMBLE_CACHE_FILENAME)
    cache.load()
    cache.put("5C", EnsembleInfo("DR Deutschland", SERVICES, 0.0, SOURCE_SCAN, eid=EID_5C))
    cache.save()
    controller = make_controller(data_directory, start=False)
    controller.set_ews_enabled(True)
    window = MainWindow(controller)
    controller.start()
    window.show()
    controller.play_service("5C", SERVICES_5C[1][0])
    worker = fake_workers[-1]
    worker.connected.emit("Connected.")
    controller.dsp_worker.stream_restarted.emit(worker.tuning)
    controller.dsp_worker.services_updated.emit("DR Deutschland", list(SERVICES), {})
    yield window, controller
    window.close()


def test_a_box_shows_the_warning_and_closes_with_it(window):
    window, controller = window
    controller.dsp_worker.ews_ready.emit(PLAYING)
    controller.dsp_worker.alert_label_ready.emit(DynamicLabel("Rauch <über> dem Hafen"))

    box = window._ews_box
    assert box.isVisible() and box.windowModality() == Qt.WindowModality.NonModal
    assert box.text() == "Warning"
    assert "ASA DE<br>Rauch &lt;über&gt; dem Hafen" in box.informativeText()

    controller.dsp_worker.ews_ready.emit(QUIET)
    assert window._ews_box is None


def test_its_buttons_break_the_warning_off(window):
    window, controller = window
    controller.dsp_worker.ews_ready.emit(PLAYING)
    box = window._ews_box
    skip = next(b for b in box.buttons() if b.text() == "Skip incident")

    skip.click()
    update = replace(ALERT, stage=Stage.LEVEL_1_UPDATE)
    controller.dsp_worker.ews_ready.emit(replace(PLAYING, alerts=(update,)))

    assert controller.alert is None
    assert window._ews_box is None
    assert box.buttonRole(box.defaultButton()) == QMessageBox.ButtonRole.RejectRole
