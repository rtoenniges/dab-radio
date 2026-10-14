"""The GNSS menu of the touch surface and what it is built from: the pager that a
finger swipes through, the number pad for the listener's position, and the five pages
drawing from the `GnssModel`.

Swipes are sent as mouse events to the widget beneath the finger, as Qt delivers those
it synthesises from touch; the clock of the pager is replaced so that their speed is
known.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
import pytest
from PyQt6.QtCore import QAbstractAnimation, QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from dab.data_services import data_components
from dab.gnss import Observer
from dab.rtcm import decode_ephemeris
from dab.ssrz import SsrzDecoder
from dsp.pipeline import DataComponentStatus
from tests.ssrz_vectors import EPHEMERIS_FRAMES
from tests.test_corrections import FRAMES
from tests.test_data_services import FIGS_5C, _ensemble
from ui.touch import pager as pager_module
from ui.touch.gnss_view import ATMOSPHERE, CORRECTIONS, GLOBE, SKY, STREAM
from ui.touch.number_pad import NumberPad
from ui.touch.pager import DOTS_FADE_MS, SETTLE_MS, Pager
from ui.touch.services_page import GNSS
from ui.touch.window import SERVICES, SETTINGS, TouchWindow

WIDTH, HEIGHT = 600, 400

# One application for the module: one destroyed and made anew would take the widgets
# of the fixtures with it.
APP = QApplication.instance() or QApplication([])


class _Clock:
    """Stands in for `time.monotonic` in the pager."""

    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(pager_module.time, "monotonic", clock.monotonic)
    return clock


class _Finger:
    """Presses, drags and lifts on a widget, in coordinates of the pager's window."""

    def __init__(self, clock: _Clock) -> None:
        self._clock = clock
        self._widget: QWidget | None = None

    def _send(self, kind: QEvent.Type, x: float, y: float, buttons) -> None:
        widget = self._widget
        point = QPointF(x, y)
        local = widget.mapFromGlobal(point)
        moving = kind == QEvent.Type.MouseMove
        button = Qt.MouseButton.NoButton if moving else Qt.MouseButton.LeftButton
        event = QMouseEvent(kind, local, point, button, buttons, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(widget, event)

    def down(self, widget: QWidget, x: float, y: float) -> None:
        self._widget = widget
        self._send(QEvent.Type.MouseButtonPress, x, y, Qt.MouseButton.LeftButton)

    def move(self, x: float, y: float, seconds: float) -> None:
        self._clock.now += seconds
        self._send(QEvent.Type.MouseMove, x, y, Qt.MouseButton.LeftButton)

    def up(self, x: float, y: float, seconds: float = 0.01) -> None:
        self._clock.now += seconds
        self._send(QEvent.Type.MouseButtonRelease, x, y, Qt.MouseButton.NoButton)

    def drag(
        self,
        widget: QWidget,
        start: tuple[float, float],
        end: tuple[float, float],
        seconds: float,
    ) -> None:
        """From `start` to `end` in ten steps over `seconds`."""
        self.down(widget, *start)
        for step in range(1, 11):
            x = start[0] + (end[0] - start[0]) * step / 10
            y = start[1] + (end[1] - start[1]) * step / 10
            self.move(x, y, seconds / 10)
        self.up(*end, seconds=0.0)


@pytest.fixture
def pager():
    pager = Pager()
    pager.resize(WIDTH, HEIGHT)
    pages = [QLabel(f"Seite {index}") for index in range(3)]
    for page in pages:
        pager.add_page(page)
    pager.show()
    APP.processEvents()
    yield pager, pages
    pager.hide()
    pager.deleteLater()


def _centre(widget: QWidget) -> tuple[float, float]:
    point = widget.mapToGlobal(widget.rect().center()).toPointF()
    return point.x(), point.y()


def test_a_long_or_quick_swipe_turns_the_page_a_short_slow_one_does_not(pager, clock) -> None:
    pager, pages = pager
    finger = _Finger(clock)
    turned = []
    pager.current_changed.connect(turned.append)
    x, y = _centre(pages[0])

    finger.drag(pages[0], (x, y), (x - 200, y), seconds=1.0)  # 200 px/s, a third of the width
    assert pager.current_index() == 0

    finger.drag(pages[0], (x, y), (x - 400, y), seconds=1.0)  # past the middle
    assert pager.current_index() == 1

    x, y = _centre(pages[1])
    finger.drag(pages[1], (x, y), (x - 60, y), seconds=0.05)  # a flick: 1200 px/s
    assert pager.current_index() == 2
    assert turned == [1, 2]

    # Beyond the last page there is none.
    x, y = _centre(pages[2])
    finger.drag(pages[2], (x, y), (x - 400, y), seconds=0.1)
    assert pager.current_index() == 2


def test_a_finger_stops_the_turning_page_a_tap_lets_it_go_on_a_swipe_moves_it_on(
    pager, clock
) -> None:
    pager, pages = pager
    finger = _Finger(clock)
    strip = pages[0].parentWidget()
    settle = pager._settle
    # Where on the pager the finger comes down does not matter: the strip moves.
    x, y = _centre(pager)

    finger.drag(pages[0], (x, y), (x - 400, y), seconds=1.0)
    assert pager.current_index() == 1
    # On its way from -400 to -600, driven by hand (see the dots below).
    settle.setCurrentTime(SETTLE_MS // 5)
    caught = strip.x()
    assert -WIDTH < caught < -400

    finger.down(pages[1], x, y)
    assert settle.state() == QAbstractAnimation.State.Stopped
    assert strip.x() == caught
    finger.up(x, y)
    assert settle.state() == QAbstractAnimation.State.Running
    assert settle.endValue() == QPoint(-WIDTH, 0)
    settle.setCurrentTime(SETTLE_MS)
    assert strip.pos() == QPoint(-WIDTH, 0)

    # Caught on the way to the third page, the strip moves on from where it stands
    # and settles on the page nearer to where the finger lets go of it.
    finger.drag(pages[1], (x, y), (x - 400, y), seconds=1.0)
    assert pager.current_index() == 2
    settle.setCurrentTime(SETTLE_MS // 5)
    caught = strip.x()
    finger.down(pages[2], x, y)
    finger.move(x + 50, y, 0.2)
    assert strip.x() == caught + 50
    finger.move(x + 250, y, 0.8)
    finger.up(x + 250, y)
    assert pager.current_index() == 1
    assert settle.endValue() == QPoint(-WIDTH, 0)


def test_a_page_hidden_under_the_finger_stands_on_its_page_again(pager, clock) -> None:
    pager, pages = pager
    finger = _Finger(clock)
    strip = pages[0].parentWidget()
    x, y = _centre(pager)
    finger.down(pages[0], x, y)
    finger.move(x - 100, y, 0.5)
    assert strip.x() == -100

    pager.hide()
    pager.show()
    assert strip.pos() == QPoint(0, 0)
    assert pager.current_index() == 0


def test_the_dots_show_while_swiping_and_fade_out(pager, clock) -> None:
    pager, pages = pager
    assert pager.dots_opacity == 1.0  # shown
    pager._dots_timer.timeout.emit()
    fade = pager._fade
    assert fade.state() == QAbstractAnimation.State.Running
    assert fade.endValue() == 0.0
    # To its end without waiting: after `test_main_window` in the same process Qt's
    # animation timer no longer ticks; the fade itself runs (checked by hand).
    fade.setCurrentTime(DOTS_FADE_MS)
    assert pager.dots_opacity == 0.0

    finger = _Finger(clock)
    x, y = _centre(pages[0])
    finger.down(pages[0], x, y)
    finger.move(x - 100, y, 0.1)
    assert pager.dots_opacity == 1.0
    finger.up(x - 100, y)


def test_a_swipe_over_a_button_does_not_click_it_a_tap_does(clock) -> None:
    pager = Pager()
    pager.resize(WIDTH, HEIGHT)
    buttons = [QPushButton("Knopf"), QPushButton("Zwei")]
    for button in buttons:
        pager.add_page(button)
    pager.show()
    APP.processEvents()
    clicks = []
    buttons[0].clicked.connect(lambda: clicks.append(True))
    finger = _Finger(clock)
    x, y = _centre(buttons[0])

    finger.drag(buttons[0], (x, y), (x - 400, y), seconds=1.0)
    assert pager.current_index() == 1
    assert not clicks
    assert not buttons[0].isDown()

    pager.set_current_index(0)
    finger.down(buttons[0], x, y)
    finger.up(x, y)
    assert clicks == [True]
    pager.hide()


def test_a_drag_up_or_down_scrolls_a_list_on_the_page(clock) -> None:
    pager = Pager()
    pager.resize(WIDTH, HEIGHT)
    table = QTableWidget(100, 1)
    table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    for row in range(100):
        table.setItem(row, 0, QTableWidgetItem(f"Zeile {row}"))
    pager.add_page(table)
    pager.add_page(QLabel("Zwei"))
    pager.show()
    APP.processEvents()
    viewport = table.viewport()
    finger = _Finger(clock)
    x, y = _centre(viewport)

    finger.drag(viewport, (x, y), (x, y - 120), seconds=1.0)  # slowly: no run-out

    assert table.verticalScrollBar().value() == 120
    assert pager.current_index() == 0
    pager.hide()


@pytest.fixture
def pad():
    window = QWidget()
    window.resize(1280, 720)
    window.show()
    pad = NumberPad(window)
    taken = []
    pad.open("Breite", -12.5, 4, "°", (-90.0, 90.0), taken.append)
    yield pad, taken
    window.deleteLater()


def test_the_first_digit_replaces_the_value_the_sign_stays(pad) -> None:
    pad, _taken = pad
    assert pad.text == "−12.5000"

    for key in "3,2":
        pad.press(key)
    assert pad.text == "−3.2"
    pad.press("±")
    assert pad.value() == pytest.approx(3.2)


def test_no_more_decimals_than_the_value_has(pad) -> None:
    pad, _taken = pad
    for key in "1,234567":
        pad.press(key)
    assert pad.text == "−1.2345"
    pad.press("⌫")
    assert pad.text == "−1.234"


def test_outside_the_range_nothing_can_be_taken(pad) -> None:
    pad, taken = pad
    for key in "95":
        pad.press(key)
    assert pad.value() is None
    assert not pad._accept.isEnabled()
    assert pad._range.property("role") == "error"

    pad.press("⌫")
    pad._accept.click()
    assert taken == [-9.0]
    assert not pad.isVisible()


def test_a_tap_beside_the_pad_cancels(pad) -> None:
    pad, taken = pad
    pad.press("7")
    pad.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(5, 5),
            QPointF(5, 5),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    assert not pad.isVisible()
    assert taken == []


def test_the_position_is_entered_on_the_settings_page(data_directory, make_controller) -> None:
    controller = make_controller(data_directory)
    window = TouchWindow(controller)
    window.show()
    window.show_page(SETTINGS)
    page = window._settings_page

    page._latitude.click()
    pad = page.number_pad()
    assert pad.isVisible()
    for key in "52,3759":
        pad.press(key)
    pad._accept.click()

    assert controller.observer.latitude == pytest.approx(52.3759)
    assert page._latitude.text() == "52.3759° N"
    page._longitude.click()
    pad.press("±")
    pad._accept.click()
    assert page._longitude.text().endswith("° W")
    window.close()


def _receive_gepos(controller, window, workers):
    """Fake reception of 5C with GEPOS decoded and the messages of the test vectors."""
    controller.start_reception()
    worker = workers[-1]
    worker.connected.emit("Verbunden.")
    dsp = controller.dsp_worker
    dsp.stream_restarted.emit(worker.tuning)
    epg, gepos = data_components(_ensemble(FIGS_5C))
    dsp.data_services_updated.emit([epg, gepos])
    dsp.data_decoding_ready.emit((DataComponentStatus(gepos.scid, gepos),))
    model = window._gnss_model
    for frame in EPHEMERIS_FRAMES.values():
        dsp.ephemeris_ready.emit(decode_ephemeris(frame[3:-3]))
    decoder = SsrzDecoder()
    for frame in FRAMES:
        dsp.ssrz_ready.emit(decoder.push(frame[3:-3]))
    return model, gepos


def test_the_gnss_menu_shows_what_gepos_carries(
    data_directory, make_controller, fake_workers
) -> None:
    controller = make_controller(data_directory, start=False)
    controller.set_observer(Observer(52.5, 9.5, 100.0))
    window = TouchWindow(controller)
    controller.start()
    window.show()
    page = window._services_page
    assert not page.menu_button(GNSS).isVisibleTo(page)

    _receive_gepos(controller, window, fake_workers)
    assert page.menu_button(GNSS).isVisibleTo(page)
    window.show_page(SERVICES)
    page.show_view(GNSS)
    view = page.gnss_view

    view.show_page(CORRECTIONS)
    corrections = view.page(CORRECTIONS)
    names = [corrections.table.item(row, 0).text() for row in range(4)]
    assert names == ["E03", "G06", "C11", "R02"]
    assert corrections.selected == "E03"
    corrections.table.cellClicked.emit(1, 0)
    assert corrections.selected == "G06"
    assert "G06" in corrections.course.text()

    view.show_page(STREAM)
    assert "GEPOS" in view.page(STREAM).status.text()
    view.show_page(ATMOSPHERE)
    assert "Corrections up to date" in view.page(ATMOSPHERE).status.text()
    view.show_page(SKY)
    assert view.current_page() == SKY
    view.show_page(GLOBE)
    assert view.current_page() == GLOBE
    assert view.page(GLOBE).status.text()

    # No tooltips: a finger has nothing to hover with.
    assert all(not widget.toolTip() for widget in view.findChildren(QWidget))
    points = [
        item
        for plot in view.findChildren(pg.PlotWidget)
        for item in plot.getPlotItem().items
        if isinstance(item, pg.ScatterPlotItem)
    ]
    assert points
    assert not any(item.opts["hoverable"] for item in points)
    window.close()


def test_a_note_switches_gepos_on_again(data_directory, make_controller, fake_workers) -> None:
    controller = make_controller(data_directory, start=False)
    window = TouchWindow(controller)
    controller.start()
    window.show()
    _model, gepos = _receive_gepos(controller, window, fake_workers)
    view = window._services_page.gnss_view
    note = view._note
    assert not note.isVisibleTo(view)

    controller.set_data_component_enabled(gepos.scid, False)
    assert note.isVisibleTo(view)
    view._switch_on.click()
    assert not controller.is_data_component_switched_off(gepos)
    assert not note.isVisibleTo(view)
    window.close()
