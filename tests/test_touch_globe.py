"""The page "3D view" of the GNSS menu: the satellites in space with their orbits in
the colours of their systems, the listener's position, and the fingers that turn, zoom
and shift the view.

OpenGL needs a graphics driver, which the offscreen platform these tests run on does not
offer. The page is therefore tested with a stand-in for the render thread that only
notes what it is asked to draw and hands an image back at once, and as it behaves without
OpenGL; the real renderer draws in a process of its own on the real platform, without a
window, and is skipped where that has no OpenGL either.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QObject, QPoint, QPointF, QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QEventPoint, QImage, QPainter, QPointingDevice, QTouchEvent
from PyQt6.QtWidgets import QApplication, QLabel

from dab.data_services import data_components
from dab.gnss import Observer
from dab.rtcm import decode_ephemeris
from tests.rtcm_vectors import (
    BEIDOU_C08_FRAME,
    GALILEO_E02_FRAME,
    GLONASS_R11_FRAME,
    GPS_G12_FRAME,
    REFERENCE_EPOCH_UTC,
)
from tests.test_data_services import FIGS_5C, _ensemble
from ui.globe_gl import CAMERA_DISTANCE_KM, EARTH_RADIUS_KM
from ui.gnss_model import ORBIT_POINTS, GnssModel, SatelliteOrbit
from ui.gnss_tab import system_style
from ui.offscreen_gl import RenderFrame
from ui.touch import globe_view
from ui.touch.globe_view import POSITION_LABEL, VIEW_DISTANCE_KM, GlobeView
from ui.touch.gnss_view import (
    GLOBE,
    PAGES,
    RADIO_GLOBE_DISTANCE_KM,
    SETTINGS_RADIO_GLOBE_DISTANCE,
    SKY,
    GlobePanel,
    _GlobePage,
)
from ui.touch.pager import Pager
from ui.touch.radio_page import SLIDE_SIZE
from ui.touch.theme import DARK, LIGHT, THEMES

APP = QApplication.instance() or QApplication([])
HANNOVER = Observer(52.5, 9.5, 100.0)
SPACE = QColor(1, 2, 3)  # what the made-up renderer fills its image with
HOLLOW = QColor(8, 12, 22)  # inside the mark of a satellite out of order
WIDTH, HEIGHT = 1200, 560

PRESSED = QEventPoint.State.Pressed
MOVED = QEventPoint.State.Updated
STILL = QEventPoint.State.Stationary
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
    """Stands in for `GlobeRenderThread`: notes what it is asked to draw and hands back
    an image of one colour at once, from the GUI thread."""

    frame_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.ready = True
        self.error = ""
        self.images = 0
        self.lines: np.ndarray | None = None

    def request(self, view, key, camera, width, height, lines=None, line_width=1.0) -> None:
        self.lines = lines
        self.images += 1
        image = QImage(width, height, QImage.Format.Format_RGB32)
        image.fill(SPACE)
        self.frame_ready.emit(RenderFrame(view, key, camera, image))


class _Clock:
    """Stands in for `time.monotonic` in the view, which tells a double tap by it."""

    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(globe_view.time, "monotonic", clock.monotonic)
    return clock


def _touch(widget, kind: QEvent.Type, points, origin: QPointF | None = None) -> None:
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


def _fingers(widget, starts: list[QPointF], ends: list[QPointF], steps: int = 10) -> None:
    """Fingers that come down together, move from `starts` to `ends` and are lifted.
    Their screen positions are whole pixels: the steps should be too, or the rounding
    moves the fingers' centre to and fro while they spread."""
    ids = range(len(starts))
    origin = _origin(widget)
    pressed = [(i, PRESSED, p) for i, p in zip(ids, starts)]
    _touch(widget, QEvent.Type.TouchBegin, pressed, origin)
    for step in range(1, steps + 1):
        places = [a + (b - a) * step / steps for a, b in zip(starts, ends)]
        moved = [(i, MOVED, p) for i, p in zip(ids, places)]
        _touch(widget, QEvent.Type.TouchUpdate, moved, origin)
    released = [(i, RELEASED, p) for i, p in zip(ids, ends)]
    _touch(widget, QEvent.Type.TouchEnd, released, origin)


def _tap(widget, at: QPointF, clock: _Clock, after: float = 2.0) -> None:
    """A finger put down `after` seconds after the last and lifted 80 ms later."""
    clock.now += after
    _touch(widget, QEvent.Type.TouchBegin, [(0, PRESSED, at)])
    clock.now += 0.08
    _touch(widget, QEvent.Type.TouchEnd, [(0, RELEASED, at)])


def _model(observer: Observer = HANNOVER) -> GnssModel:
    model = GnssModel(observer)
    model.now = lambda: REFERENCE_EPOCH_UTC
    for frame in (GPS_G12_FRAME, GALILEO_E02_FRAME, BEIDOU_C08_FRAME, GLONASS_R11_FRAME):
        model.add_ephemeris(decode_ephemeris(frame[3:-3]))
    return model


@pytest.fixture
def page():
    model = _model()
    renderer = _FakeRenderer()
    page = _GlobePage(model, renderer)
    page.resize(WIDTH, HEIGHT)
    page.show()
    APP.processEvents()  # until then the window is not exposed and paints nothing
    page.render()
    page.view.repaint()
    yield page, model, renderer
    page.close()
    page.deleteLater()


def _marks(view: GlobeView) -> dict[str, QPointF]:
    """Where the marks are drawn, by the satellite's name or "Location"."""
    return {
        what if isinstance(what, str) else what.ephemeris.name: place
        for place, _radius, what in view._marks
    }


def _orbits(view: GlobeView) -> dict[str, SatelliteOrbit]:
    return {s.ephemeris.name: s for s in view._satellites}


def test_the_satellites_and_their_orbits_in_the_colours_of_their_systems(page) -> None:
    page, _model, renderer = page
    assert page.title.text() == "3D view"
    assert page.status.text() == "4 satellites, 4 above the horizon"
    view = page.view

    # The orbits and nothing else: no lines of sight down to the listener's position.
    lines = renderer.lines
    assert lines.shape == (4 * 2 * ORBIT_POINTS, 7)
    assert np.all(np.linalg.norm(lines[:, :3], axis=1) > 20_000)
    for index, satellite in enumerate(view._satellites):
        piece = lines[index * 2 * ORBIT_POINTS : (index + 1) * 2 * ORBIT_POINTS]
        colour = np.array(system_style(satellite.ephemeris.system).colour) / 255
        assert np.allclose(piece[:, 3:6], colour)
        assert np.allclose(piece[:, 6], 0.55)
        assert np.allclose(piece[0, :3], satellite.position / 1000, atol=1e-2)

    # The marks of those in front of the Earth and in the view, in the colour of their
    # system, R11 hollow (out of order by its ephemeris, as in the sky); the listener's
    # position in yellow.
    marks = _marks(view)
    assert {"G12", "E02", "R11", POSITION_LABEL} <= set(marks)
    image = page.view.grab().toImage()
    ratio = image.devicePixelRatio()

    def colour(name: str) -> QColor:
        place = marks[name]
        return image.pixelColor(round(place.x() * ratio), round(place.y() * ratio))

    for name in ("G12", "E02"):
        satellite = _orbits(view)[name]
        assert colour(name).getRgb()[:3] == system_style(satellite.ephemeris.system).colour
    assert colour("R11") == HOLLOW
    assert colour(POSITION_LABEL) == QColor(253, 216, 53)


def test_below_the_horizon_they_are_paler_and_the_view_turns_to_a_new_position(page) -> None:
    page, model, renderer = page
    view = page.view
    # The first view: from the south-west of the listener.
    assert (view.camera.latitude_deg, view.camera.longitude_deg) == (52.5 - 24, 9.5 - 40)
    assert view.camera.distance_km == VIEW_DISTANCE_KM

    model.set_observer(Observer(-52.5, -170.5, 0.0))  # the other side of the Earth
    page.render()
    view.repaint()
    assert page.status.text() == "4 satellites, 0 above the horizon"
    assert np.allclose(renderer.lines[:, 6], 0.18)
    # From the south-west again: 40° west of 170.5° W is 149.5° E.
    assert (view.camera.latitude_deg, view.camera.longitude_deg) == (-52.5 - 24, 149.5)


def test_two_fingers_turn_the_view_and_zoom_it_three_shift_it(page) -> None:
    page, _model, _renderer = page
    view = page.view
    camera = view.camera
    latitude, longitude = camera.latitude_deg, camera.longitude_deg
    left, right = QPointF(500, 300), QPointF(600, 300)

    # Together to the right: the ground follows, the view turns westwards.
    shift = QPointF(150, 0)
    _fingers(view, [left, right], [left + shift, right + shift])
    km_per_pixel = camera.km_per_pixel(view.height())
    turned = math.degrees(150 * km_per_pixel / EARTH_RADIUS_KM) / math.cos(math.radians(latitude))
    assert camera.longitude_deg == pytest.approx(longitude - turned)
    assert camera.latitude_deg == pytest.approx(latitude)
    assert camera.distance_km == pytest.approx(VIEW_DISTANCE_KM)

    # Spread to twice the distance: twice as close.
    _fingers(view, [left, right], [left - QPointF(50, 0), right + QPointF(50, 0)])
    assert camera.distance_km == pytest.approx(VIEW_DISTANCE_KM / 2)
    assert camera.longitude_deg == pytest.approx(longitude - turned)
    # Closed again, far beyond: no farther than the camera goes.
    for _ in range(4):
        _fingers(view, [left - QPointF(200, 0), right + QPointF(200, 0)], [left, right])
    assert camera.distance_km == CAMERA_DISTANCE_KM[1]

    # Three fingers shift the centre of the view; the ground follows them.
    longitude = camera.longitude_deg
    km_per_pixel = camera.km_per_pixel(view.height())
    three = [QPointF(400, 200), QPointF(500, 260), QPointF(450, 330)]
    _fingers(view, three, [p + QPointF(-30, 40) for p in three])  # 3 and 4 px a step
    assert camera.right_km == pytest.approx(30 * km_per_pixel)
    assert camera.up_km == pytest.approx(40 * km_per_pixel)
    assert camera.longitude_deg == longitude


def test_a_finger_coming_or_going_does_not_make_the_view_jump(page) -> None:
    page, _model, _renderer = page
    view = page.view
    camera = view.camera
    before = (camera.latitude_deg, camera.longitude_deg, camera.distance_km, camera.right_km)
    a, b, c = QPointF(400, 300), QPointF(500, 300), QPointF(900, 100)
    _touch(view, QEvent.Type.TouchBegin, [(0, PRESSED, a), (1, PRESSED, b)])
    _touch(view, QEvent.Type.TouchUpdate, [(0, STILL, a), (1, STILL, b), (2, PRESSED, c)])
    _touch(view, QEvent.Type.TouchUpdate, [(0, STILL, a), (1, STILL, b), (2, RELEASED, c)])
    _touch(view, QEvent.Type.TouchUpdate, [(0, STILL, a), (1, STILL, b)])
    _touch(view, QEvent.Type.TouchEnd, [(0, RELEASED, a), (1, RELEASED, b)])
    after = (camera.latitude_deg, camera.longitude_deg, camera.distance_km, camera.right_km)
    assert after == before


def test_a_tap_turns_the_view_around_a_satellite_and_a_double_tap_returns(page, clock) -> None:
    page, model, _renderer = page
    view = page.view
    _tap(view, _marks(view)["G12"], clock)
    assert view.followed == "G12"
    view.finish_animations()
    g12 = _orbits(view)["G12"]
    assert np.allclose(view.camera.pivot_km, g12.position / 1000)

    # It follows the satellite as it moves on.
    model.now = lambda: REFERENCE_EPOCH_UTC + 60
    page.feed.invalidate()
    page.render()
    view.repaint()
    moved = _orbits(view)["G12"].position / 1000
    assert np.linalg.norm(moved - g12.position / 1000) > 100
    assert np.allclose(view.camera.pivot_km, moved)

    # A tap on the listener's position: around the centre of the Earth again.
    _tap(view, _marks(view)[POSITION_LABEL], clock)
    view.finish_animations()
    assert view.followed is None
    assert view.camera.around_earth

    # Twice in quick succession: back to the first view.
    _fingers(view, [QPointF(500, 300), QPointF(600, 300)], [QPointF(600, 350), QPointF(800, 350)])
    _tap(view, QPointF(100, 100), clock)
    _tap(view, QPointF(110, 105), clock, after=0.2)
    camera = view.camera
    assert (camera.latitude_deg, camera.longitude_deg) == (52.5 - 24, 9.5 - 40)
    assert camera.distance_km == VIEW_DISTANCE_KM
    assert camera.around_earth


def test_the_button_resets_the_view(page, clock) -> None:
    page, _model, _renderer = page
    view = page.view
    assert page.reset_button.text() == "Reset view"
    _tap(view, _marks(view)["G12"], clock)
    view.finish_animations()
    _fingers(view, [QPointF(500, 300), QPointF(600, 300)], [QPointF(600, 350), QPointF(800, 350)])
    three = [QPointF(400, 200), QPointF(500, 260), QPointF(450, 330)]
    _fingers(view, three, [p + QPointF(-30, 40) for p in three])
    assert view.followed == "G12"

    page.reset_button.click()
    camera = view.camera
    assert (camera.latitude_deg, camera.longitude_deg) == (52.5 - 24, 9.5 - 40)
    assert camera.distance_km == VIEW_DISTANCE_KM
    assert (camera.right_km, camera.up_km) == (0.0, 0.0)
    assert camera.around_earth
    assert view.followed is None


def test_one_finger_turns_the_page_not_the_view() -> None:
    pager = Pager()
    pager.resize(WIDTH, HEIGHT)
    page = _GlobePage(_model(), _FakeRenderer())
    pager.add_page(page)
    pager.add_page(QLabel("Nächste Seite"))
    pager.show()
    page.render()
    APP.processEvents()
    view = page.view
    camera = view.camera
    before = (camera.latitude_deg, camera.longitude_deg)

    start = QPointF(view.width() / 2, view.height() / 2)
    _fingers(view, [start], [start - QPointF(700, 0)])
    assert pager.current_index() == 1
    assert (camera.latitude_deg, camera.longitude_deg) == before
    assert view.followed is None
    pager.close()
    pager.deleteLater()


def test_the_page_is_the_second_of_the_gnss_menu() -> None:
    assert PAGES[:2] == (SKY, GLOBE)


def test_on_the_radio_page_the_view_is_cut_to_the_slide_and_nearer() -> None:
    model = _model()
    renderer = _FakeRenderer()
    panel = GlobePanel(model, renderer)
    panel.resize(SLIDE_SIZE)
    panel.show()
    APP.processEvents()
    panel.render()
    view = panel.view
    view.repaint()

    # The view fills the slide; the first view of the page, but farther.
    assert view.size() == SLIDE_SIZE
    assert not panel.heading.isVisible()
    camera = view.camera
    assert (camera.latitude_deg, camera.longitude_deg) == (52.5 - 24, 9.5 - 40)
    assert camera.distance_km == RADIO_GLOBE_DISTANCE_KM > VIEW_DISTANCE_KM
    # What the page says in its header, in the corner.
    assert view.caption == "4 satellites, 4 above the horizon"

    # The corners in the colour of the page around it, in either scheme; space is dark.
    for theme in (THEMES[DARK], THEMES[LIGHT]):
        panel.apply_theme(theme)
        image = view.grab().toImage()
        assert image.pixelColor(0, 0) == theme.color("background")
        assert image.pixelColor(image.width() - 1, image.height() - 1) == theme.color("background")
        assert image.pixelColor(image.width() // 2, 4) == SPACE
    panel.close()
    panel.deleteLater()


def test_on_the_radio_page_the_view_keeps_its_distance(tmp_path) -> None:
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    model = _model()
    panel = GlobePanel(model, _FakeRenderer(), settings)
    panel.render()
    assert panel.view.camera.distance_km == RADIO_GLOBE_DISTANCE_KM
    panel.zoom(0.5)
    nearer = RADIO_GLOBE_DISTANCE_KM / 2
    assert settings.value(SETTINGS_RADIO_GLOBE_DISTANCE, type=float) == pytest.approx(nearer)

    # A new listener's position turns the view there, from the same distance.
    model.set_observer(Observer(48.0, 11.5, 500.0))
    panel.render()
    camera = panel.view.camera
    assert (camera.latitude_deg, camera.longitude_deg) == (48.0 - 24, 11.5 - 40)
    assert camera.distance_km == pytest.approx(nearer)
    panel.deleteLater()

    # The next start begins there.
    panel = GlobePanel(_model(), _FakeRenderer(), settings)
    panel.render()
    assert panel.view.camera.distance_km == pytest.approx(nearer)
    panel.deleteLater()

    # Edited by hand: the first distance, or the nearest the camera allows.
    low, high = CAMERA_DISTANCE_KM
    for stored, expected in (
        ("weit", RADIO_GLOBE_DISTANCE_KM),
        ("nan", RADIO_GLOBE_DISTANCE_KM),
        (1.0, low),
        (1e9, high),
    ):
        settings.setValue(SETTINGS_RADIO_GLOBE_DISTANCE, stored)
        panel = GlobePanel(_model(), _FakeRenderer(), settings)
        panel.render()
        assert panel.view.camera.distance_km == expected
        panel.deleteLater()


def test_the_radio_page_says_what_the_view_waits_for() -> None:
    model = GnssModel(HANNOVER)
    panel = GlobePanel(model, _FakeRenderer())
    panel.render()
    assert panel.view.caption == "No GEPOS in the ensemble."
    model.update_components(data_components(_ensemble(FIGS_5C)))  # no ephemeris yet
    panel.feed.invalidate()
    panel.render()
    assert panel.view.caption == "Waiting for ephemerides …"
    model.now = lambda: REFERENCE_EPOCH_UTC
    model.add_ephemeris(decode_ephemeris(GPS_G12_FRAME[3:-3]))
    panel.render()
    assert panel.view.caption == "1 satellite, 1 above the horizon"
    panel.deleteLater()


def test_a_label_that_would_run_off_the_view_stands_left_of_its_mark() -> None:
    view = GlobeView(_FakeRenderer())
    view.resize(WIDTH, HEIGHT)

    def drawn_columns(x: float) -> np.ndarray:
        image = QImage(view.size(), QImage.Format.Format_ARGB32)
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        view._draw_label(painter, QPointF(x, 100), 5.0, "G12", strong=True)
        painter.end()
        return np.array(
            [c for c in range(image.width()) if image.pixelColor(c, 100).alpha() > 0]
            + [c for c in range(image.width()) if image.pixelColor(c, 96).alpha() > 0]
        )

    beside = drawn_columns(200)
    assert beside.min() > 200 + 5
    edge = drawn_columns(WIDTH - 8)
    assert edge.max() < WIDTH - 8 - 5


def test_without_opengl_the_page_says_so() -> None:
    page = _GlobePage(_model())
    page.resize(WIDTH, HEIGHT)
    page.show()
    page.render()
    deadline = time.monotonic() + 10
    while page.view.available and time.monotonic() < deadline:
        page.view.repaint()
        APP.processEvents()
    assert "OpenGL 2.1" in page.view.error
    page.render()
    assert page.status.text() == "No 3D view without OpenGL"
    page.close()
    page.deleteLater()


# Draws with the real renderer and prints the colours of some pixels: seen from above the
# Gulf of Guinea, an orbit-like line along the Earth's axis in green.
_RENDER_SCRIPT = """
import json, sys
import numpy as np
from PyQt6.QtWidgets import QApplication
from ui.earth import ellipsoid_km
from ui.globe_gl import Camera, GlobeRenderer
from ui.offscreen_gl import project

app = QApplication(sys.argv)
renderer = GlobeRenderer()
steps = 0
while renderer.prepare():
    steps += 1
camera = Camera(0.0, 0.0, 40_000.0)
width, height = 800, 600
matrix = camera.matrix(width / height)
places = {
    "sea": ellipsoid_km(np.array(-5.0), np.array(-5.0)),
    "sahara": ellipsoid_km(np.array(26.0), np.array(4.0)),
    "congo": ellipsoid_km(np.array(-3.0), np.array(22.0)),
    "line beside the earth": np.array([0.0, 0.0, 10_000.0]),
    "line inside the earth": np.array([0.0, 0.0, 1_500.0]),
}
line = np.array([[0, 0, -20_000, 0, 1, 0, 1], [0, 0, 20_000, 0, 1, 0, 1]], dtype=np.float32)
result = {"error": renderer.error, "steps": steps, "ready": renderer.ready}
for name, lines in (("plain", None), ("scene", line)):
    image = renderer.render(camera, width, height, lines, line_width=3.0)
    if image is None:
        break
    pixels = {"corner": image.pixelColor(3, 3).getRgb()[:3], "size": [image.width(), image.height()]}
    for place, position in places.items():
        x, y = project(matrix, np.asarray(position)[None, :], width, height)[0]
        pixels[place] = image.pixelColor(round(x), round(y)).getRgb()[:3]
    result[name] = pixels
renderer.release()
result["released"] = not renderer.ready
print(json.dumps(result))
"""


@pytest.mark.skipif(sys.platform != "win32", reason="needs a desktop to draw on")
def test_opengl_draws_the_earth_and_an_orbit_hidden_behind_it() -> None:
    source = Path(__file__).parent.parent / "src"
    environment = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    environment["PYTHONPATH"] = str(source)
    run = subprocess.run(
        [sys.executable, "-c", _RENDER_SCRIPT],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout.strip().splitlines()[-1])
    if result["error"]:
        pytest.skip(f"no OpenGL on this machine: {result['error']}")
    assert result["ready"] and result["steps"] == 4 and result["released"]

    plain, scene = result["plain"], result["scene"]
    assert plain["size"] == [800, 600]
    assert sum(plain["corner"]) < 40  # space
    red, green, blue = plain["sea"]
    assert blue > 1.5 * green > red
    for land in ("sahara", "congo"):
        red, green, blue = plain[land]
        assert green > blue and green > 80
    # The line shows beside the Earth and is hidden where it runs through it.
    red, green, blue = scene["line beside the earth"]
    assert green > 200 and green > 2 * red and green > 2 * blue
    assert sum(plain["line beside the earth"]) < 120
    assert (
        np.abs(np.array(scene["line inside the earth"]) - plain["line inside the earth"]).max() < 40
    )
    assert np.abs(np.array(scene["sea"]) - plain["sea"]).max() < 10
