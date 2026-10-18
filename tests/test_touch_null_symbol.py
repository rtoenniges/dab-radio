"""The null symbol on the radio page (`ui.touch.null_symbol_view`): it asks for the
stretches around the null symbols only while shown, draws them at the plot rate taking
turns between frames of even and odd number, with fixed settings and camera, and forgets
them on another channel.

OpenGL is stood in for as on the detail page (`tests/test_touch_detail_view.py`)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QApplication

from dab.constants import MODE_I, SAMPLE_RATE
from dsp.signal_capture import NullSnapshotter, SignalCapture
from tests.dab_signal import build_stream, fib
from tests.test_touch_detail_view import _FakeRenderer
from ui.spectrogram import NULL_LEAD, Part
from ui.touch.display_choice import NULL_SYMBOL, make_panel, preview_pixmap
from ui.touch.null_symbol_view import (
    CAMERA,
    COLUMNS,
    FFT_SIZE,
    RANGE_DB,
    ROWS,
    WINDOW_MS,
    NullSymbolPanel,
)
from ui.touch.signal_panels import FrameTally
from ui.touch.theme import DARK, LIGHT, THEMES
from ui.waterfall_gl import time_ticks

APP = QApplication.instance() or QApplication([])
FRAME = MODE_I.frame_length
NULL = MODE_I.null_symbol_length


@pytest.fixture(scope="module")
def snapshots() -> list[SignalCapture]:
    """The stretches around the null symbols of ten frames one after the other, cut as
    the signal processing cuts them; frame k has the number k."""
    rng = np.random.default_rng(5)
    stream = build_stream([fib()] * 12, 11, rng, snr_db=40).astype(np.complex64)
    snapshotter = NullSnapshotter(MODE_I, SAMPLE_RATE)
    snapshotter.add_block(0, stream)
    return [snapshotter.add_frame(NULL + k * FRAME, 0.0) for k in range(1, 11)]


@pytest.fixture
def panel(data_directory, make_controller, monkeypatch):
    controller = make_controller(data_directory)
    switched: list[bool] = []
    monkeypatch.setattr(controller, "set_null_snapshots", switched.append)
    renderer = _FakeRenderer()
    panel = NullSymbolPanel(controller, renderer)
    panel.apply_theme(THEMES[DARK])
    panel.resize(640, 480)
    panel.show()
    APP.processEvents()
    panel.switched = switched
    panel.renderer = renderer
    panel.controller = controller
    yield panel
    panel.close()
    panel.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


def _shown(panel) -> int | None:
    capture = panel.model.capture
    return None if capture is None else capture.first_frame_number


def test_it_asks_for_the_null_symbols_only_while_shown(panel) -> None:
    assert panel.switched == [True]
    panel.hide()
    assert panel.switched == [True, False]
    assert panel.view.scene is None  # nothing came yet
    assert panel.view._waiting_text == "Waiting for the null symbols …"


def test_it_takes_turns_between_even_and_odd_frames(panel, snapshots) -> None:
    emit = panel.controller.dsp_worker.null_snapshot_ready.emit
    # Two frames come for every tick of the plot rate (96 ms against 200 ms).
    shown = []
    for first in range(0, 10, 2):
        emit(snapshots[first])
        emit(snapshots[first + 1])
        panel.render()
        shown.append(_shown(panel))
    parities = [number % 2 for number in shown]
    assert parities == [0, 1, 0, 1, 0]
    # Always one of the two newest.
    assert all(number in (2 * k + 1, 2 * k + 2) for k, number in enumerate(shown))
    # No new frames: the older one of the last pair once, then nothing anew.
    scenes = len(panel.renderer.scenes)
    panel.render()
    panel.render()
    assert len(panel.renderer.scenes) == scenes + 1  # the older one of the last pair
    assert _shown(panel) == 9


def test_the_landscape_has_the_settings_of_the_picture(panel, snapshots) -> None:
    snapshot = snapshots[3]
    panel.controller.dsp_worker.null_snapshot_ready.emit(snapshot)
    panel.render()
    scene = panel.view.scene
    field = scene.field
    null_ms = 1000 * float(snapshot.null_starts[0]) / SAMPLE_RATE
    assert panel.model.fft_size == FFT_SIZE == 1024
    assert scene.range_db == RANGE_DB == 30.0
    assert field.window_ms == pytest.approx((null_ms - NULL_LEAD * WINDOW_MS, null_ms + 3.2))
    assert field.span_khz == (-1024.0, 1024.0)
    assert field.levels_db.shape == (ROWS, COLUMNS)
    # The time counted from the start of the null symbol.
    times, labels = time_ticks(field)
    assert labels[:4] == ["0", "1", "2", "3"] and times[0] == pytest.approx(null_ms)
    nulls = [begin for begin, _end, part in field.parts if part is Part.NULL]
    assert nulls == pytest.approx([null_ms])
    assert panel.view.camera == CAMERA


def test_another_channel_forgets_the_null_symbols(panel, snapshots) -> None:
    emit = panel.controller.dsp_worker.null_snapshot_ready.emit
    emit(snapshots[0])
    panel.render()
    assert panel.view.scene is not None
    panel.controller.live_reset.emit()
    assert panel.view.scene is None and panel.model.capture is None
    panel.render()
    assert panel.view.scene is None  # what came before is gone
    emit(snapshots[1])
    panel.render()
    assert _shown(panel) == 2


def test_the_radio_page_gets_it_with_the_shared_renderer(data_directory, make_controller):
    controller = make_controller(data_directory)
    renderer = _FakeRenderer()
    tally = FrameTally(controller)
    panel = make_panel(NULL_SYMBOL, controller, tally, None, None, renderer)
    assert isinstance(panel, NullSymbolPanel) and panel.view._renderer is renderer
    for theme in (DARK, LIGHT):
        assert not preview_pixmap(NULL_SYMBOL, THEMES[theme]).isNull()
    panel.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
