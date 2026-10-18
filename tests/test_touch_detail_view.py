"""The page "Detail view" of the signal page: a capture asked for when it is first turned
to, the landscape scrolled in time with one finger and turned with two, the overview
that moves the window, and the controls.

As for the globe (`tests/test_touch_globe.py`), OpenGL is stood in for by a render
thread that hands back an image of one colour at once; the landscape's numbers are
tested in `tests/test_spectrogram.py`.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QObject, QPoint, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QEventPoint,
    QImage,
    QPainter,
    QPointingDevice,
    QTouchEvent,
    QWheelEvent,
)
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from dab.constants import MODE_I, SAMPLE_RATE
from dsp.signal_capture import MARGIN_SAMPLES, SignalCapture
from tests.dab_signal import build_stream, fib, noise
from ui.offscreen_gl import RenderFrame
from ui.spectrogram import LEVEL_RANGES_DB, WINDOWS_MS, Rows
from ui.touch.detail_view import SCROLL_PER_HEIGHT, WHEEL_SCROLL, DetailPage
from ui.touch.theme import DARK, THEMES
from ui.waterfall_gl import (
    DEFAULT_AZIMUTH_DEG,
    DEFAULT_DISTANCE,
    DEFAULT_LOOK_AT,
    LOOK_AT_LIMITS,
    WaterfallScene,
)

APP = QApplication.instance() or QApplication([])
FRAME = MODE_I.frame_length
NULL = MODE_I.null_symbol_length

PRESSED = QEventPoint.State.Pressed
MOVED = QEventPoint.State.Updated
RELEASED = QEventPoint.State.Released
_DEVICE = QPointingDevice(
    "test touch screen",
    1,
    QPointingDevice.DeviceType.TouchScreen,
    QPointingDevice.PointerType.Finger,
    QPointingDevice.Capability.Position,
    5,
    0,
)


class _FakeRenderer(QObject):
    """Stands in for the render thread: notes the scenes it is asked to draw and hands
    back an image of one colour at once."""

    frame_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.ready = True
        self.error = ""
        self.scenes: list[WaterfallScene] = []

    def request(self, view, key, camera, width, height, scene=None, line_width=1.0) -> None:
        self.scenes.append(scene)
        image = QImage(width, height, QImage.Format.Format_RGB32)
        image.fill(QColor(1, 2, 3))
        self.frame_ready.emit(RenderFrame(view, key, camera, image))


def _capture() -> SignalCapture:
    rng = np.random.default_rng(3)
    frames = build_stream([fib()] * 12, 3, rng, snr_db=40)
    samples = np.concatenate(
        [noise(MARGIN_SAMPLES + 0, rng), frames, noise(MARGIN_SAMPLES, rng)]
    ).astype(np.complex64)
    starts = MARGIN_SAMPLES + NULL + FRAME * np.arange(3)
    return SignalCapture(samples, SAMPLE_RATE, MODE_I, starts, np.zeros(3))


@pytest.fixture
def page(data_directory, make_controller, monkeypatch):
    controller = make_controller(data_directory)
    asked = []
    monkeypatch.setattr(controller, "capture_signal", lambda: asked.append(1))
    renderer = _FakeRenderer()
    page = DetailPage(controller, renderer)
    page.apply_theme(THEMES[DARK])
    page.resize(1100, 620)
    page.show()
    APP.processEvents()
    page.asked = asked
    page.renderer = renderer
    page.controller = controller
    yield page
    page.close()
    page.deleteLater()
    # Now, not in whichever later test first runs the event loop: there the deletions
    # of whole modules add up (up to a second, enough to hold up a long press).
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


def _captured(page) -> SignalCapture:
    page.render()
    capture = _capture()
    page.controller.dsp_worker.signal_captured.emit(capture)
    APP.processEvents()
    return capture


def _touch(widget, kind: QEvent.Type, points) -> None:
    origin = QPointF(widget.mapToGlobal(QPoint(0, 0)))
    event_points = [
        QEventPoint(ident, state, position, origin + position) for ident, state, position in points
    ]
    QApplication.sendEvent(
        widget, QTouchEvent(kind, _DEVICE, Qt.KeyboardModifier.NoModifier, event_points)
    )


def _fingers(widget, starts: list[QPointF], ends: list[QPointF], steps: int = 10) -> None:
    ids = range(len(starts))
    _touch(widget, QEvent.Type.TouchBegin, [(i, PRESSED, p) for i, p in zip(ids, starts)])
    for step in range(1, steps + 1):
        places = [a + (b - a) * step / steps for a, b in zip(starts, ends)]
        _touch(widget, QEvent.Type.TouchUpdate, [(i, MOVED, p) for i, p in zip(ids, places)])
    _touch(widget, QEvent.Type.TouchEnd, [(i, RELEASED, p) for i, p in zip(ids, ends)])


def _wheel(widget, steps: int, modifiers=Qt.KeyboardModifier.NoModifier) -> None:
    at = QPointF(widget.width() / 2, widget.height() / 2)
    event = QWheelEvent(
        at,
        QPointF(widget.mapToGlobal(at.toPoint())),
        QPoint(),
        QPoint(0, 120 * steps),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)


def test_the_first_turn_to_the_page_asks_for_frames_and_shows_them(page) -> None:
    assert page.view.scene is None
    page.render()
    page.render()
    assert page.asked == [1]  # once, until they come
    assert page.status.text() == "Capturing …"
    assert not page.capture_button.isEnabled()

    capture = _captured(page)
    assert page.model.capture is capture
    assert page.status.text().startswith("3 frames of 5C, captured at ")
    assert page.capture_button.isEnabled()
    scene = page.renderer.scenes[-1]
    assert scene is page.view.scene and scene.field.levels_db.shape[0] > 10
    assert page.overview._image is not None
    assert page.view.caption == (
        "Frame 1, null symbol · a row every 20 µs from 250 µs each, 4 kHz per column"
    )

    # Frames nobody here asked for are left alone.
    page.controller.dsp_worker.signal_captured.emit(_capture())
    assert page.model.capture is capture


def test_one_finger_up_or_down_scrolls_through_time_across_does_not(page) -> None:
    _captured(page)
    model, view = page.model, page.view
    model.scroll_to(model.duration_ms / 2)
    start = model.start_ms

    # Down: the landscape comes nearer, later rows come in at the far end of the time.
    _fingers(view, [QPointF(300, 150)], [QPointF(300, 250)])
    expected = start + 100 * SCROLL_PER_HEIGHT * model.window_ms / view.height()
    assert model.start_ms == pytest.approx(expected, abs=1e-6)

    start = model.start_ms
    _fingers(view, [QPointF(200, 200)], [QPointF(400, 210)])  # across: the pager's
    assert model.start_ms == start

    # The field follows, computed a moment later (a finger gives many steps); other
    # tests' windows deleted meanwhile may hold the event loop up longer than that.
    for _ in range(200):
        if not view._compute_timer.isActive():
            break
        QTest.qWait(10)
    assert page.view.scene.field.window_ms[0] == pytest.approx(model.start_ms)


def test_two_fingers_turn_and_zoom_and_a_double_tap_returns(page) -> None:
    _captured(page)
    view = page.view
    camera = view.camera
    left, right = QPointF(300, 300), QPointF(400, 300)
    _fingers(view, [left, right], [left + QPointF(100, 0), right + QPointF(100, 0)])
    assert camera.azimuth_deg == pytest.approx(DEFAULT_AZIMUTH_DEG - 100 * 180 / view.width())
    _fingers(view, [left, right], [left - QPointF(50, 0), right + QPointF(50, 0)])
    assert camera.distance == pytest.approx(DEFAULT_DISTANCE / 2)

    middle = QPointF(300, 200)
    for _ in range(2):
        _touch(view, QEvent.Type.TouchBegin, [(0, PRESSED, middle)])
        _touch(view, QEvent.Type.TouchEnd, [(0, RELEASED, middle)])
    assert view.camera.distance == DEFAULT_DISTANCE
    assert view.camera.azimuth_deg == DEFAULT_AZIMUTH_DEG


def test_three_fingers_shift_the_view_within_bounds(page) -> None:
    _captured(page)
    view = page.view
    camera = view.camera
    starts = [QPointF(300, 300), QPointF(400, 300), QPointF(350, 380)]
    requested = len(page.renderer.scenes)
    _fingers(view, starts, [p + QPointF(60, 0) for p in starts])
    # Neither turned nor zoomed, nor scrolled in time; drawn anew.
    assert (camera.azimuth_deg, camera.distance) == (DEFAULT_AZIMUTH_DEG, DEFAULT_DISTANCE)
    assert len(page.renderer.scenes) > requested
    # The landscape follows the fingers: the point looked at went the other way.
    moved = np.array(camera.look_at) - np.array(DEFAULT_LOOK_AT)
    assert moved @ camera.axes()[0] == pytest.approx(-60 * camera.units_per_pixel(view.height()))
    assert moved @ camera.axes()[1] == pytest.approx(0.0, abs=1e-9)
    # Far beyond: it stops where the landscape would leave the view.
    _fingers(view, starts, [p + QPointF(5000, -5000) for p in starts])
    for value, (low, high) in zip(camera.look_at, LOOK_AT_LIMITS, strict=True):
        assert low <= value <= high
    middle = QPointF(300, 200)
    for _ in range(2):
        _touch(view, QEvent.Type.TouchBegin, [(0, PRESSED, middle)])
        _touch(view, QEvent.Type.TouchEnd, [(0, RELEASED, middle)])
    assert view.camera.look_at == DEFAULT_LOOK_AT


def test_the_wheel_scrolls_and_with_ctrl_zooms(page) -> None:
    _captured(page)
    model, view = page.model, page.view
    model.scroll_to(10.0)
    _wheel(view, 2)
    assert model.start_ms == pytest.approx(10.0 + 2 * WHEEL_SCROLL * model.window_ms)
    _wheel(view, 1, Qt.KeyboardModifier.ControlModifier)
    assert view.camera.distance < DEFAULT_DISTANCE
    QTest.mouseDClick(view, Qt.MouseButton.LeftButton, pos=QPoint(100, 100))
    assert view.camera.distance == DEFAULT_DISTANCE


def test_the_left_button_dragged_down_scrolls_on(page) -> None:
    _captured(page)
    model, view = page.model, page.view
    model.scroll_to(10.0)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=QPoint(300, 200))
    QTest.mouseMove(view, QPoint(300, 220))
    QTest.mouseMove(view, QPoint(300, 300))
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=QPoint(300, 300))
    assert model.start_ms > 10.0


def test_the_middle_button_turns_and_the_right_one_shifts_as_on_the_globe(page) -> None:
    _captured(page)
    view = page.view
    QTest.mousePress(view, Qt.MouseButton.MiddleButton, pos=QPoint(300, 200))
    QTest.mouseMove(view, QPoint(400, 200))
    QTest.mouseRelease(view, Qt.MouseButton.MiddleButton, pos=QPoint(400, 200))
    camera = view.camera
    assert camera.azimuth_deg == pytest.approx(DEFAULT_AZIMUTH_DEG - 100 * 180 / view.width())
    assert camera.look_at == DEFAULT_LOOK_AT
    azimuth = camera.azimuth_deg
    QTest.mousePress(view, Qt.MouseButton.RightButton, pos=QPoint(300, 200))
    QTest.mouseMove(view, QPoint(360, 200))
    QTest.mouseRelease(view, Qt.MouseButton.RightButton, pos=QPoint(360, 200))
    moved = np.array(camera.look_at) - np.array(DEFAULT_LOOK_AT)
    assert moved @ camera.axes()[0] == pytest.approx(-60 * camera.units_per_pixel(view.height()))
    assert camera.azimuth_deg == azimuth
    # Released, a move of the mouse does nothing.
    QTest.mouseMove(view, QPoint(100, 100))
    assert camera.azimuth_deg == azimuth


def test_the_overview_puts_the_window_where_it_is_tapped_and_drags_it(page) -> None:
    _captured(page)
    model, overview = page.model, page.overview
    assert overview.claims_drag(QPointF(0, 0))
    width, height = overview.width(), overview.height()

    QTest.mousePress(overview, Qt.MouseButton.LeftButton, pos=QPoint(width // 2, height // 2))
    assert model.start_ms + model.window_ms / 2 == pytest.approx(
        model.duration_ms * (width // 2) / width, abs=0.01
    )
    QTest.mouseMove(overview, QPoint(width // 2 + 50, height // 2))
    QTest.mouseRelease(
        overview, Qt.MouseButton.LeftButton, pos=QPoint(width // 2 + 50, height // 2)
    )
    assert model.start_ms + model.window_ms / 2 == pytest.approx(
        model.duration_ms * (width // 2 + 50) / width, abs=0.01
    )
    rect = overview.window_rect()
    assert rect.center().x() == pytest.approx(width // 2 + 50, abs=1)


def test_the_controls(page) -> None:
    capture = _captured(page)
    model = page.model

    page.rows.chosen.emit(Rows.SYMBOLS)
    assert model.rows is Rows.SYMBOLS
    assert not page.fft.isEnabled() and page.fft.value() == 2048
    assert page.view.scene.field.symbols is not None
    assert "one row per symbol, one column per carrier" in page.view.caption
    page.rows.chosen.emit(Rows.SLIDING)
    assert page.fft.isEnabled() and page.fft.value() == 512

    page.window.value_changed.emit(16.0)
    assert model.window_ms == 16.0
    page.span.value_changed.emit(256)
    assert page.view.scene.field.span_khz == (-128.0, 128.0)

    # The height: a new scene of the same field, not computed again.
    field, drawn = page.view.scene.field, len(page.renderer.scenes)
    assert page.level_range.value() == 50.0 and page.view.scene.range_db == 50.0
    page.level_range.value_changed.emit(20.0)
    assert page.view.scene.range_db == 20.0 and page.view.scene.field is field
    assert len(page.renderer.scenes) == drawn + 1

    page.next_null.click()
    lead = 0.2 * model.window_ms
    second = 1000 * capture.null_starts[1] / SAMPLE_RATE
    assert model.start_ms == pytest.approx(second - lead)
    page.previous_null.click()
    assert model.start_ms < second - lead

    page.capture_button.click()
    assert page.asked == [1, 1]
    assert page.status.text() == "Capturing …"
    assert page.model.capture is capture  # the landscape stays until the new one comes

    # The new one is shown where the old one was, as turned and zoomed as it was.
    start, span, azimuth = model.start_ms, model.span_khz, page.view.camera.azimuth_deg
    page.controller.dsp_worker.signal_captured.emit(_capture())
    assert page.model.capture is not capture
    assert (model.start_ms, model.span_khz) == (start, span)
    assert page.view.camera.azimuth_deg == azimuth
    assert page.view.scene.field.window_ms[0] == start
    assert page.view.scene.range_db == 20.0


def test_without_opengl_the_view_says_so(page) -> None:
    _captured(page)
    page.renderer.error = "The 3D view needs OpenGL 2.1"
    page.view.repaint()
    assert not page.view.available


@pytest.mark.parametrize("rows", [Rows.SLIDING, Rows.SYMBOLS])
def test_the_labels_are_drawn_for_every_window_and_place(page, rows) -> None:
    """Drawn by the paint event, where an error only reaches the terminal: called here
    itself, for every length of the window at places across the capture (a window of
    1 ms per symbol can hold no symbol's middle and leave the axis empty)."""
    _captured(page)
    model, view = page.model, page.view
    model.rows = rows
    image = QImage(view.width(), view.height(), QImage.Format.Format_RGB32)
    for window_ms in WINDOWS_MS:
        model.set_window_ms(window_ms)
        for share in np.linspace(0.0, 1.0, 23):
            model.scroll_to(share * (model.duration_ms - window_ms))
            view.refresh(at_once=True)
            painter = QPainter(image)
            try:
                view._draw_labels(painter, view.camera, view.scene)
                view._draw_legend(painter, view.scene)
            finally:
                painter.end()
    for range_db in LEVEL_RANGES_DB:
        view.set_range_db(range_db)
        painter = QPainter(image)
        try:
            view._draw_labels(painter, view.camera, view.scene)
        finally:
            painter.end()
