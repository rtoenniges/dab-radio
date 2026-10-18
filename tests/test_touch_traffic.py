"""The "Traffic" menu of the touch surface with real ARD TPEG messages on a small road
network and a made-up TMC location table: the map with its gestures (two fingers zoom,
one turns the page, a tap shows a message) and its legend (a tap hides a layer), the
list, the traffic flow, the stream, and the note while TPEG is switched off.

Fingers are sent as touch events, as Qt delivers them to a widget that accepts them.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import dataclasses

import pytest
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QEventPoint, QMouseEvent, QPointingDevice, QTouchEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from dab.data_services import data_components
from dab.gnss import Observer
from dab.tpeg_messages import FlowSection, FlowVector, TfpMessage
from dsp.pipeline import DataComponentStatus
from tests import tmc_table, traffic_roads
from tests.test_data_services import _ensemble
from tests.test_tpeg import _application_frame, _service_frame
from tests.test_traffic_tab import NOW, _frames
from tests.tpeg_vectors import (
    FIGS_7A_TPEG,
    TEC_LANE_CLOSED,
    TEC_ROADWORKS,
    TEC_TRAFFIC_LIGHTS,
    TFP_FREE,
)
from ui.receiver_controller import ReceiverController
from ui.touch import traffic_view
from ui.touch.services_page import TRAFFIC
from ui.touch.theme import DARK, LIGHT, THEMES
from ui.touch.traffic_view import (
    BUILD_PAUSE_MS,
    FLOW,
    MAP,
    MESSAGES,
    SETTINGS_HIDDEN,
    SHOW_WIDTH,
    STREAM,
    _MapPage,
    _MessageRow,
    _RowList,
)
from ui.touch.window import RADIO, SERVICES, TouchWindow

APP = QApplication.instance() or QApplication([])
HANOVER = (52.37, 9.73)
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


def _touch(
    widget,
    kind: QEvent.Type,
    points: list[tuple[int, QEventPoint.State, QPointF]],
    origin: QPointF | None = None,
):
    """Fingers (id, state, position on `widget`). Their screen position counts from
    where the widget stood when the gesture began (`origin`, else now): a page the
    gesture turns moves the widget, not the fingers."""
    if origin is None:
        origin = _origin(widget)
    event_points = [
        QEventPoint(ident, state, position, origin + QPointF(position.toPoint()))
        for ident, state, position in points
    ]
    QApplication.sendEvent(
        widget, QTouchEvent(kind, _DEVICE, Qt.KeyboardModifier.NoModifier, event_points)
    )


def _origin(widget) -> QPointF:
    """Where the top left corner of `widget` is on the screen."""
    return QPointF(widget.mapToGlobal(QPoint(0, 0)))


def _one_finger(widget, start: QPointF, end: QPointF, steps: int = 8) -> None:
    origin = _origin(widget)
    _touch(widget, QEvent.Type.TouchBegin, [(0, PRESSED, start)], origin)
    for step in range(1, steps + 1):
        point = start + (end - start) * step / steps
        _touch(widget, QEvent.Type.TouchUpdate, [(0, MOVED, point)], origin)
    _touch(widget, QEvent.Type.TouchEnd, [(0, RELEASED, end)], origin)


def _pinch(widget, centre: QPointF, start: float, end: float, steps: int = 8) -> None:
    def fingers(state0, state1, spread):
        offset = QPointF(spread / 2, 0)
        return [(0, state0, centre - offset), (1, state1, centre + offset)]

    _touch(widget, QEvent.Type.TouchBegin, [(0, PRESSED, centre - QPointF(start / 2, 0))])
    _touch(widget, QEvent.Type.TouchUpdate, fingers(MOVED, PRESSED, start))
    for step in range(1, steps + 1):
        _touch(
            widget,
            QEvent.Type.TouchUpdate,
            fingers(MOVED, MOVED, start + (end - start) * step / steps),
        )
    _touch(widget, QEvent.Type.TouchEnd, fingers(RELEASED, RELEASED, end))


def _tap(widget, point: QPointF) -> None:
    _touch(widget, QEvent.Type.TouchBegin, [(0, PRESSED, point)])
    _touch(widget, QEvent.Type.TouchEnd, [(0, RELEASED, point)])


def _mouse(widget, start: QPointF, end: QPointF, steps: int = 8) -> None:
    """A finger as Qt hands it to most widgets, as the mouse: down on `widget` at
    `start`, moved to `end` and lifted there."""

    origin = _origin(widget)

    def send(kind: QEvent.Type, point: QPointF, button, buttons) -> None:
        global_point = origin + QPointF(point.toPoint())
        event = QMouseEvent(
            kind, point, global_point, button, buttons, Qt.KeyboardModifier.NoModifier
        )
        QApplication.sendEvent(widget, event)

    left = Qt.MouseButton.LeftButton
    send(QEvent.Type.MouseButtonPress, start, left, left)
    if end != start:
        for step in range(1, steps + 1):
            point = start + (end - start) * (step / steps)
            send(QEvent.Type.MouseMove, point, Qt.MouseButton.NoButton, left)
    send(QEvent.Type.MouseButtonRelease, end, left, Qt.MouseButton.NoButton)


@pytest.fixture(autouse=True)
def whole_lists(monkeypatch):
    """The lists filled in one slice, however slow the machine; the tests of the
    slices set their own."""
    monkeypatch.setattr(traffic_view, "BUILD_SLICE_SECONDS", 60.0)


@pytest.fixture(scope="module")
def roads(tmp_path_factory):
    return traffic_roads.network(tmp_path_factory.mktemp("roads"))


@pytest.fixture
def window(data_directory, make_controller, fake_workers, roads, tmp_path):
    controller: ReceiverController = make_controller(data_directory, start=False)
    controller.set_observer(Observer(*HANOVER))
    window = TouchWindow(controller)
    controller.start()
    window.show()
    model = window._traffic_model
    model.now = lambda: NOW
    model.set_roads(roads)
    model.set_locations(tmc_table.table(tmp_path))
    controller.start_reception()
    worker = fake_workers[-1]
    worker.connected.emit("Verbunden.")
    dsp = controller.dsp_worker
    dsp.stream_restarted.emit(worker.tuning)
    (tpeg,) = data_components(_ensemble(FIGS_7A_TPEG))
    dsp.data_services_updated.emit([tpeg])
    dsp.data_decoding_ready.emit((DataComponentStatus(tpeg.scid, tpeg),))
    frames = _frames(
        _service_frame(
            _application_frame(1, [TEC_LANE_CLOSED, TEC_TRAFFIC_LIGHTS, TEC_ROADWORKS]),
            _application_frame(2, [TFP_FREE]),
        )
    )
    for frame in frames:
        dsp.tpeg_frame_ready.emit(tpeg.scid, frame)
    # A queue at the head of the B 213, besides its free flow.
    flow = next(s.message for s in model.stored(TfpMessage))
    vector = FlowVector(0, (FlowSection(2, 5, 12), FlowSection(5, 3, 35), FlowSection(9, 1, 90)))
    management = dataclasses.replace(flow.management, message_id=7777)
    model._store.add(
        dataclasses.replace(flow, vectors=(vector,), management=management), frames[-1].frame, NOW
    )
    model._changed()
    window.show_page(SERVICES)
    page = window._services_page
    page.show_view(TRAFFIC)
    view = page.traffic_view
    yield window, view, tpeg
    window.close()


def _draw(window) -> None:
    view = window._services_page.traffic_view
    for page in (view.page(MAP), view.page(MESSAGES)):
        page._drawn_at = None
    window._render_plots()
    APP.processEvents()


def _pixel(map_page, longitude: float, latitude: float) -> QPointF:
    """Where a place of the map is on its viewport."""
    plot = map_page.plot
    scene = plot.getViewBox().mapViewToScene(QPointF(longitude, latitude))
    return QPointF(plot.mapFromScene(scene))


def test_the_menu_is_offered_and_the_map_shows_the_messages(window) -> None:
    window, view, _tpeg = window
    assert window._services_page.menu_button(TRAFFIC).isVisibleTo(window)
    _draw(window)
    map_page = view.page(MAP)
    assert map_page.status.text() == "3 messages, 3 with a location · 1 stretch disturbed"
    assert map_page.flow.count() == 2  # the queue and the slow stretch of the B 213
    assert map_page.roads.network is not None
    closures, _stationary, _slow, hints = map_page._markers
    assert len(closures.data) == 1 and len(hints.data) == 2


def test_the_names_of_the_towns_lie_above_the_messages_in_a_halo(window) -> None:
    window, view, _tpeg = window
    _draw(window)
    map_page = view.page(MAP)
    beneath = [
        *map_page._lines,
        *map_page._markers,
        map_page._selected_ring,
        map_page._observer,
        *map_page.flow._items.values(),
    ]
    top = max(item.zValue() for item in beneath)
    for theme in (THEMES[LIGHT], THEMES[DARK]):
        view.apply_theme(theme)
        labels = map_page.roads.labels
        assert labels.names
        assert labels.zValue() > top
        assert labels.halo == theme.color("surface")


def test_two_fingers_zoom_one_turns_the_page(window) -> None:
    window, view, _tpeg = window
    _draw(window)
    plot = view.page(MAP).plot
    viewport = plot.viewport()
    (x0, x1), _ = plot.getViewBox().viewRange()
    centre = QPointF(viewport.width() / 2, viewport.height() / 2)

    _pinch(viewport, centre, 100, 300)
    (y0, y1), _ = plot.getViewBox().viewRange()
    assert (y1 - y0) == pytest.approx((x1 - x0) / 3, rel=0.05)
    assert view.current_page() == MAP

    # One finger across the map: the next page, the view as it was.
    _one_finger(viewport, centre + QPointF(150, 0), centre - QPointF(250, 0))
    assert view.current_page() == MESSAGES
    (z0, z1), _ = plot.getViewBox().viewRange()
    assert (z0, z1) == pytest.approx((y0, y1))


def test_a_tap_on_a_symbol_shows_its_message_and_more_opens_it(window) -> None:
    window, view, _tpeg = window
    _draw(window)
    map_page = view.page(MAP)
    viewport = map_page.plot.viewport()
    # The roadworks at Laatzen, at the start of its line.
    _tap(viewport, _pixel(map_page, 9.79473, 52.31268))
    assert map_page.selected[1] == 1745
    assert "Road closed · roadworks" in map_page.card_text.text()

    map_page.more.click()
    assert view.current_page() == MESSAGES
    messages = view.page(MESSAGES)
    assert messages.rows[map_page.selected].is_open
    assert sum(row.is_open for row in messages.rows.values()) == 1

    # A message with a TMC location is named after it.
    view.show_page(MAP)
    _tap(viewport, _pixel(map_page, 11.12996, 51.11431))
    assert map_page.selected[1] == 1371
    assert "A 71 at Testkreuz Nord junction" in map_page.card_text.text()

    # A tap beside every symbol shows none.
    _tap(viewport, QPointF(5, 5))
    assert map_page.selected is None
    assert not map_page.more.isVisible()


def test_the_list_shows_a_message_on_the_map(window) -> None:
    window, view, _tpeg = window
    view.show_page(MESSAGES)
    _draw(window)
    messages = view.page(MESSAGES)
    keys = messages.order
    assert [key[1] for key in keys] == [1745, 2554, 1371]
    assert messages.rows[keys[1]].below.text() == "L 123 at Am Testwall"

    messages.rows[keys[2]].map_requested.emit(keys[2])
    assert view.current_page() == MAP
    map_page = view.page(MAP)
    assert map_page.selected == keys[2]
    (x0, x1), _ = map_page.plot.getViewBox().viewRange()
    assert x1 - x0 <= SHOW_WIDTH + 1e-6
    assert x0 < 11.1 < x1  # the A 71 north of Erfurt


def test_an_open_message_holds_the_list_until_it_is_closed(window) -> None:
    window, view, _tpeg = window
    view.show_page(MESSAGES)
    _draw(window)
    messages = view.page(MESSAGES)
    keys = list(messages.order)
    row = messages.rows[keys[1]]
    QTest.mouseClick(row, Qt.MouseButton.LeftButton)
    assert row.is_open

    # Another message comes: the list stays as it is while one is read.
    model = window._traffic_model
    stored = model.get(keys[0])
    management = dataclasses.replace(stored.message.management, message_id=8888)
    model._store.add(dataclasses.replace(stored.message, management=management), stored.frame, NOW)
    model._changed()
    _draw(window)
    assert messages.order == keys
    assert row.is_open
    assert messages.status.text() == "List paused while a message is open"

    # Closed, the list takes it in.
    QTest.mouseClick(row, Qt.MouseButton.LeftButton)
    _draw(window)
    assert (keys[0][0], 8888) in messages.order
    assert messages.status.text() == "by distance from the location"

    # A message opened from the map fills the list anew, also while another is open.
    QTest.mouseClick(row, Qt.MouseButton.LeftButton)
    model._store.add(
        dataclasses.replace(
            stored.message, management=dataclasses.replace(management, message_id=9999)
        ),
        stored.frame,
        NOW,
    )
    model._changed()
    view.show_message((keys[0][0], 9999))
    assert messages.rows[(keys[0][0], 9999)].is_open
    assert not row.is_open


def test_scrolling_or_swiping_over_a_row_does_not_open_it(window) -> None:
    window, view, _tpeg = window
    view.show_page(MESSAGES)
    _draw(window)
    messages = view.page(MESSAGES)
    # Enough messages for the list to scroll.
    model = window._traffic_model
    stored = model.get(messages.order[0])
    for message_id in range(9000, 9020):
        management = dataclasses.replace(stored.message.management, message_id=message_id)
        model._store.add(
            dataclasses.replace(stored.message, management=management), stored.frame, NOW
        )
    model._changed()
    _draw(window)
    bar = messages.list.verticalScrollBar()
    assert bar.maximum() > 0
    row = messages.rows[messages.order[4]]
    title = row.title
    centre = QPointF(title.width() / 2, title.height() / 2)

    # Up the list: it scrolls, and no row opens.
    _mouse(title, centre, centre - QPointF(0, 120))
    assert bar.value() > 0
    assert not any(other.is_open for other in messages.rows.values())

    # Across: the next page, and the row stays shut.
    _mouse(title, centre, centre - QPointF(300, 0))
    assert view.current_page() == FLOW
    assert not row.is_open

    # A tap opens it.
    view.show_page(MESSAGES)
    _mouse(title, centre, centre)
    assert row.is_open


def test_the_stream_page_lists_the_applications(window) -> None:
    window, view, _tpeg = window
    view.show_page(STREAM)
    window._render_plots()
    stream = view.page(STREAM)
    table = stream.table
    names = [table.item(row, 0).text() for row in range(table.rowCount())]
    assert names[:3] == [
        "SNI (service information)",
        "TEC (traffic events)",
        "TFP (traffic flow)",
    ]
    assert table.item(1, 2).text() == "3 of 655"
    # The shares of the stream add up to all component frames, the SNI among them.
    shares = [table.item(row, 5).text() for row in range(table.rowCount())]
    assert all(share.endswith(" %") for share in shares[:3])
    assert shares[3:] == ["-"]  # the EAW component sent nothing
    assert 99 <= sum(int(share.removesuffix(" %")) for share in shares[:3]) <= 101
    assert "ARD TPEG" in stream.name.text()
    assert "CRC errors" in stream.check.text()


def test_a_note_switches_tpeg_on_again(window) -> None:
    window, view, tpeg = window
    controller = window._controller
    note = view._note
    assert not note.isVisibleTo(view)
    controller.set_data_component_enabled(tpeg.scid, False)
    assert note.isVisibleTo(view)
    view._switch_on.click()
    assert not note.isVisibleTo(view)


def test_a_mouse_click_on_a_symbol_is_a_tap(window) -> None:
    window, view, _tpeg = window
    _draw(window)
    map_page = view.page(MAP)
    viewport = map_page.plot.viewport()
    point = _pixel(map_page, 9.79473, 52.31268)
    for kind, buttons in (
        (QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton),
        (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton),
    ):
        event = QMouseEvent(
            kind,
            point,
            QPointF(viewport.mapToGlobal(point.toPoint())),
            Qt.MouseButton.LeftButton,
            buttons,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(viewport, event)
    assert map_page.selected[1] == 1745
    assert view.current_page() == MAP


# Where the hint on the A 71 north of Erfurt starts.
_A71_HINT = (11.12996, 51.11431)


def _tap_entry(entry) -> None:
    centre = QPointF(entry.width() / 2, entry.height() / 2)
    _mouse(entry, centre, centre)


def test_a_tap_on_the_legend_hides_its_messages_and_the_settings_keep_it(window) -> None:
    window, view, _tpeg = window
    _draw(window)
    map_page = view.page(MAP)
    viewport = map_page.plot.viewport()
    closures, _stationary, _slow, hints = map_page._markers
    _tap(viewport, _pixel(map_page, *_A71_HINT))
    assert map_page.selected[1] == 1371

    entry = map_page.legend["severity3"]
    _tap_entry(entry)
    assert not entry.shown
    assert "○" in entry.mark.text()
    assert not entry.property("pressed")
    assert len(hints.data) == 0 and len(closures.data) == 1
    # The message shown beneath the map went with its layer; its symbol answers no tap.
    assert map_page.selected is None
    assert not map_page.more.isVisible()
    _tap(viewport, _pixel(map_page, *_A71_HINT))
    assert map_page.selected is None
    # The status still counts every message with a place.
    assert map_page.status.text() == "3 messages, 3 with a location · 1 stretch disturbed"

    settings = window._controller.settings
    assert settings.value(SETTINGS_HIDDEN) == "severity3"
    again = _MapPage(window._traffic_model, settings)
    assert not again.legend["severity3"].shown
    assert again.legend["severity0"].shown

    _tap_entry(entry)
    assert entry.shown
    assert len(hints.data) == 2
    assert settings.value(SETTINGS_HIDDEN) == ""


def test_a_message_shown_on_the_map_shows_its_layer_again(window) -> None:
    window, view, _tpeg = window
    view.show_page(MESSAGES)
    _draw(window)
    map_page = view.page(MAP)
    map_page.legend["severity3"].set_shown(False)
    key = next(key for key in view.page(MESSAGES).order if key[1] == 1371)

    view.show_on_map(key)
    assert map_page.legend["severity3"].shown
    assert map_page.selected == key
    assert len(map_page._markers[3].data) == 2


def test_the_legend_hides_levels_of_the_flow(window) -> None:
    window, view, _tpeg = window
    _draw(window)
    map_page = view.page(MAP)
    map_page.legend["flow5"].set_shown(False)  # the queue
    assert map_page.flow.count() == 1
    map_page.legend["flow3"].set_shown(False)  # slow, and stop and go alike
    assert map_page.flow.count() == 0

    # A stretch shown on the map shows the levels it has.
    view.show_page(FLOW)
    flow_page = view.page(FLOW)
    flow_page._drawn_at = None
    window._render_plots()
    (row,) = flow_page.rows.values()
    row.map_requested.emit(row.key)
    assert map_page.legend["flow5"].shown and map_page.legend["flow3"].shown
    assert map_page.flow.count() == 2


def test_without_the_road_network_cities_show_where_the_messages_are() -> None:
    from ui.traffic_model import TrafficModel

    page = _MapPage(TrafficModel(Observer(*HANOVER)))
    page.render()
    assert page.roads.network is None
    assert page._cities.isVisible()
    assert "road network missing" in page.status.text()


def test_the_flow_page_lists_the_disturbed_stretches(window) -> None:
    window, view, _tpeg = window
    view.show_page(FLOW)
    flow_page = view.page(FLOW)
    flow_page._drawn_at = None
    window._render_plots()
    assert flow_page.status.text() == "1 stretch disturbed, 1 free"
    (row,) = flow_page.rows.values()
    assert row.title.text() == "B 213 between Teststadt 7 and Teststadt 0 junction"
    assert row.figure.text() == "stationary · 12–90 km/h"
    assert not flow_page.note.isVisible()

    row.map_requested.emit(row.key)
    assert view.current_page() == MAP
    map_page = view.page(MAP)
    assert map_page.selected_flow == row.key
    (x0, x1), (y0, y1) = map_page.plot.getViewBox().viewRange()
    assert x0 < 8.17 and 8.59 < x1 and y0 < 52.9 and 53.04 < y1


def test_without_the_location_table_the_flow_page_says_how_to_import_it() -> None:
    from ui.touch.traffic_view import _FlowPage
    from ui.traffic_model import TrafficModel

    page = _FlowPage(TrafficModel(Observer(*HANOVER)))
    page.show()
    page.render()
    assert page.note.isVisible()
    assert "import_tmc_locations" in page.note.text()
    assert not page.list.isVisible()
    assert page.status.text() == "TMC location table missing"


def _wait_until(condition, seconds: float = 2.0) -> bool:
    for _ in range(round(seconds * 200)):
        if condition():
            return True
        QTest.qWait(5)
    return condition()


def _row_list() -> _RowList:
    rows = _RowList(_MessageRow)
    rows.resize(600, 400)
    rows.show()
    return rows


def _show_index(row, index: int) -> None:
    row.title.setText(f"Message {index}")


def _laid_out(rows: _RowList) -> list:
    """The keys of the rows in the layout, top to bottom."""
    layout = rows._layout
    widgets = [layout.itemAt(i).widget() for i in range(layout.count())]
    return [w.key for w in widgets if isinstance(w, _MessageRow)]


def test_a_long_list_fills_in_slices_the_first_rows_first(monkeypatch) -> None:
    monkeypatch.setattr(traffic_view, "BUILD_SLICE_SECONDS", 0.0)  # a row a slice
    rows = _row_list()
    keys = [("m", i) for i in range(5)]
    rows.fill(keys, _show_index, "… and 3 more")
    assert rows.order == keys[:1] and not rows.complete
    assert not rows.rest.isVisibleTo(rows)
    assert _wait_until(lambda: rows.complete)
    assert rows.order == keys == _laid_out(rows)
    assert rows.rest.isVisibleTo(rows) and rows.rest.text() == "… and 3 more"
    APP.processEvents()
    tops = [rows.rows[key].y() for key in keys]
    assert tops == sorted(set(tops))

    # A new order with a new row first: the rows there move at once, and the slices
    # show them anew.
    again = [("m", 9), *reversed(keys)]
    rows.fill(again, _show_index, "")
    assert rows.order == again == _laid_out(rows)
    assert not rows.complete
    assert _wait_until(lambda: rows.complete)
    assert rows.rows[keys[0]].title.text() == "Message 5"
    assert not rows.rest.isVisibleTo(rows)


def test_a_row_not_made_yet_opens_and_the_rest_waits_for_it(monkeypatch) -> None:
    monkeypatch.setattr(traffic_view, "BUILD_SLICE_SECONDS", 0.0)
    rows = _row_list()
    keys = [("m", i) for i in range(6)]
    rows.fill(keys, _show_index, "")
    # Made with those before it.
    rows.open(keys[3])
    assert rows.order == keys[:4] == _laid_out(rows)
    assert rows.rows[keys[3]].is_open
    QTest.qWait(3 * BUILD_PAUSE_MS)
    assert rows.order == keys[:4] and rows.paused
    # Closed, the next filling makes the rest.
    rows.rows[keys[3]].set_open(False)
    rows.fill(keys, _show_index, "")
    assert _wait_until(lambda: rows.complete)
    assert rows.order == keys


def test_the_lists_fill_while_the_map_is_shown(window) -> None:
    window, view, _tpeg = window
    assert view.current_page() == MAP
    _draw(window)
    assert [key[1] for key in view.page(MESSAGES).order] == [1745, 2554, 1371]
    assert len(view.page(FLOW).rows) == 1


def test_rows_made_while_the_menu_is_hidden_have_their_height_when_shown(
    window, monkeypatch
) -> None:
    """The slices go on while another page of the window is shown; back on the list,
    the rows have the height they need instead of sharing the height the list had."""
    window, view, _tpeg = window
    model = window._traffic_model
    queue = next(s for s in model.stored(TfpMessage) if s.message.management.message_id == 7777)
    for message_id in range(8000, 8012):
        management = dataclasses.replace(queue.message.management, message_id=message_id)
        model._store.add(
            dataclasses.replace(queue.message, management=management), queue.frame, NOW
        )
    model._changed()
    monkeypatch.setattr(traffic_view, "BUILD_SLICE_SECONDS", 0.0)  # a row a slice
    flow = view.page(FLOW)
    flow._drawn_at = None
    window._render_plots()
    assert not flow.list.complete
    window.show_page(RADIO)
    assert _wait_until(lambda: flow.list.complete)
    assert len(flow.rows) == 13

    window.show_page(SERVICES)
    view.show_page(FLOW)
    QTest.qWait(50)
    for row in flow.rows.values():
        assert row.height() >= row.heightForWidth(row.width())
    assert flow.list.verticalScrollBar().maximum() > 0


def test_an_open_message_holds_the_rows_still_to_come(window, monkeypatch) -> None:
    window, view, _tpeg = window
    view.show_page(MESSAGES)
    _draw(window)
    messages = view.page(MESSAGES)
    model = window._traffic_model
    stored = model.get(messages.order[0])
    for message_id in range(9000, 9010):
        management = dataclasses.replace(stored.message.management, message_id=message_id)
        model._store.add(
            dataclasses.replace(stored.message, management=management), stored.frame, NOW
        )
    model._changed()
    monkeypatch.setattr(traffic_view, "BUILD_SLICE_SECONDS", 0.0)
    _draw(window)
    assert not messages.list.complete
    row = messages.rows[messages.order[0]]
    QTest.mouseClick(row, Qt.MouseButton.LeftButton)
    assert row.is_open
    QTest.qWait(3 * BUILD_PAUSE_MS)
    assert messages.list.paused
    made = len(messages.rows)
    _draw(window)
    assert len(messages.rows) == made
    assert messages.status.text() == "List paused while a message is open"

    QTest.mouseClick(row, Qt.MouseButton.LeftButton)
    _draw(window)
    assert _wait_until(lambda: messages.list.complete)
    assert len(messages.order) == 13
    assert messages.status.text() == "by distance from the location"
