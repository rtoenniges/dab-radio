"""What the radio page of the touch surface shows in the place of the slideshow: a long
press on the slide leads to the choice, a tile there chooses a panel of the signal page
or the 3D view of GNSS, which is drawn only while it is shown and kept for the next
start. The 3D view is offered only while the ensemble carries GEPOS."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QEventLoop, QPoint, QPointF, QSettings, Qt, QTimer
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel

from dab.data_services import USER_APPLICATION_GEPOS, data_components
from dsp.superframe import AudioFormat, SuperframeResult
from tests.test_data_services import FIGS_5C, _ensemble
from tests.test_touch_globe import MOVED, PRESSED, RELEASED, STILL, _touch
from ui.touch import gnss_view
from ui.touch.display_choice import (
    CONSTELLATION,
    DISPLAYS,
    GLOBE,
    NULL_SYMBOL,
    SETTINGS_RADIO_DISPLAY,
    SLIDESHOW,
    SOUND_FIGURES,
    SPECTRUM,
    offered_displays,
    preview_pixmap,
)
from ui.touch.globe_view import WHEEL_ZOOM
from ui.touch.gnss_view import GlobePanel
from ui.touch.null_symbol_view import (
    CAMERA,
    SETTINGS_RADIO_NULL_SYMBOL_CAMERA,
    stored_camera,
)
from ui.touch.radio_page import LONG_PRESS_MS, SLIDE_SIZE
from ui.touch.services_page import GNSS
from ui.touch.signal_page import OVERVIEW
from ui.touch.signal_panels import ConstellationPanel, SignalPanel
from ui.touch.theme import DARK, LIGHT, THEMES
from ui.touch.window import RADIO, SIGNAL, TouchWindow

APP = QApplication.instance() or QApplication([])

HE_AAC = AudioFormat(True, True, True, False, 0)


@pytest.fixture
def touch_window(make_controller):
    """Creates a touch window on a fresh controller, on the radio page."""
    windows = []

    def make(data_directory) -> TouchWindow:
        controller = make_controller(data_directory, start=False)
        window = TouchWindow(controller)
        controller.start()
        window.show()
        window.show_page(RADIO)
        windows.append(window)
        return window

    yield make
    for window in windows:
        window.close()
        window.deleteLater()


def _spin(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def _receive_5c(window, workers) -> list:
    """Fake reception of 5C, whose data components are the programme guide and GEPOS;
    returns them."""
    controller = window._controller
    controller.start_reception()
    worker = workers[-1]
    worker.connected.emit("Verbunden.")
    controller.dsp_worker.stream_restarted.emit(worker.tuning)
    components = data_components(_ensemble(FIGS_5C))
    controller.dsp_worker.data_services_updated.emit(components)
    return components


def _without_gepos(components: list) -> list:
    return [c for c in components if USER_APPLICATION_GEPOS not in c.app_types]


def _superframe() -> SuperframeResult:
    return SuperframeResult(
        sync_ok=True,
        rs_corrections=3,
        rs_uncorrectable_blocks=0,
        num_audio_units=3,
        audio_format=HE_AAC,
        au_segments=[],
        data_length=990,
        total_length=1080,
    )


def test_a_finger_resting_on_the_slide_leads_to_the_choice(data_directory, touch_window):
    window = touch_window(data_directory)
    radio = window._radio_page
    area = radio._slide_area
    centre = QPoint(SLIDE_SIZE.width() // 2, SLIDE_SIZE.height() // 2)

    QTest.mousePress(area, Qt.MouseButton.LeftButton, pos=centre)
    assert area.held()
    assert area._shade.isVisible()
    assert not radio.choosing()

    _spin(LONG_PRESS_MS + 200)
    assert radio.choosing()
    assert not area._shade.isVisible()
    # The display shown is marked.
    assert radio._choice.current() == SLIDESHOW
    QTest.mouseRelease(area, Qt.MouseButton.LeftButton, pos=centre)


def test_a_tap_or_a_swipe_on_the_slide_is_no_long_press(data_directory, touch_window):
    window = touch_window(data_directory)
    radio = window._radio_page
    area = radio._slide_area
    centre = QPoint(SLIDE_SIZE.width() // 2, SLIDE_SIZE.height() // 2)

    QTest.mousePress(area, Qt.MouseButton.LeftButton, pos=centre)
    QTest.mouseRelease(area, Qt.MouseButton.LeftButton, pos=centre)
    assert not area.held()

    QTest.mousePress(area, Qt.MouseButton.LeftButton, pos=centre)
    QTest.mouseMove(area, centre + QPoint(80, 0))
    assert not area.held()
    assert not area._shade.isVisible()

    _spin(LONG_PRESS_MS + 200)
    assert not radio.choosing()
    QTest.mouseRelease(area, Qt.MouseButton.LeftButton, pos=centre + QPoint(80, 0))


def test_a_tile_shows_its_panel_in_the_place_of_the_slideshow(data_directory, touch_window):
    window = touch_window(data_directory)
    radio = window._radio_page
    radio.show_choice()

    radio._choice.tile(CONSTELLATION).click()

    assert not radio.choosing()
    assert radio.display() == CONSTELLATION
    panel = radio._slide_area.currentWidget()
    assert isinstance(panel, ConstellationPanel)
    assert panel.size() == SLIDE_SIZE
    # The panel never takes the touch: the press is the slide area's.
    assert panel.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    rng = np.random.default_rng(0)
    points = (rng.normal(size=500) + 1j * rng.normal(size=500)).astype(np.complex64)
    window._controller.dsp_worker.constellation_ready.emit(points)
    window._render_plots()
    assert len(panel.curve.getData()[0]) == 500

    radio.show_choice()
    radio._choice.tile(SLIDESHOW).click()
    assert radio._slide_area.currentWidget() is radio._slide_label


def test_every_display_can_be_chosen(data_directory, touch_window, fake_workers):
    window = touch_window(data_directory)
    radio = window._radio_page
    _receive_5c(window, fake_workers)

    for display in DISPLAYS:
        radio.show_display(display.key)
        shown = radio._slide_area.currentWidget()
        assert radio.shown_display() == display.key
        if display.key == SLIDESHOW:
            assert shown is radio._slide_label
        else:
            assert isinstance(shown, SignalPanel)
            assert shown.isVisibleTo(window)
            assert shown.size() == SLIDE_SIZE
            assert shown.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)


def test_the_tiles_run_on_downwards_without_a_hint(data_directory, touch_window, fake_workers):
    window = touch_window(data_directory)
    radio = window._radio_page
    choice = radio._choice
    _receive_5c(window, fake_workers)
    radio.show_choice()
    APP.processEvents()

    # Only the title is text; the 3D view of GNSS is the last tile, in a third row
    # whose top shows beneath the two that fit.
    assert [label.text() for label in choice.findChildren(QLabel)] == ["Display on the radio page"]
    assert DISPLAYS[-1].key == GLOBE
    assert choice.tile(GLOBE).text() == "GNSS 3D view"
    viewport = choice.area.viewport()
    top = choice.tile(GLOBE).mapTo(viewport, QPoint(0, 0)).y()
    assert 0 < top < viewport.height() < top + choice.tile(GLOBE).height()

    # Chosen, it is scrolled into sight when the choice opens again.
    choice.tile(GLOBE).click()
    radio.show_choice()
    top = choice.tile(GLOBE).mapTo(viewport, QPoint(0, 0)).y()
    assert 0 <= top and top + choice.tile(GLOBE).height() <= viewport.height()
    choice.tile(SLIDESHOW).click()
    radio.show_choice()
    assert choice.area.verticalScrollBar().value() == 0


def test_the_globe_is_offered_while_the_ensemble_carries_gepos(
    data_directory, touch_window, make_controller, fake_workers
):
    window = touch_window(data_directory)
    radio = window._radio_page
    choice = radio._choice
    controller = window._controller

    # Before the ensemble's data services are known: every tile but the globe's.
    radio.show_choice()
    assert not choice.tile(GLOBE).isEnabled()
    assert all(choice.tile(d.key).isEnabled() for d in DISPLAYS if d.key != GLOBE)

    components = _receive_5c(window, fake_workers)
    assert choice.tile(GLOBE).isEnabled()
    choice.tile(GLOBE).click()
    assert not radio.choosing()
    assert radio.display() == radio.shown_display() == GLOBE
    panel = radio._slide_area.currentWidget()
    assert isinstance(panel, GlobePanel)
    assert panel.view.camera.distance_km == gnss_view.RADIO_GLOBE_DISTANCE_KM
    # One OpenGL context for both 3D views.
    window._services_page.show_view(GNSS)
    other = window._services_page.gnss_view.page(gnss_view.GLOBE).view
    assert panel.view._renderer is other._renderer is window._globe_renderer

    # Without GEPOS the slideshow stands in for it; the choice stays and is kept.
    controller.dsp_worker.data_services_updated.emit(_without_gepos(components))
    assert radio.display() == GLOBE
    assert radio.shown_display() == SLIDESHOW
    assert radio._slide_area.currentWidget() is radio._slide_label
    assert controller.settings.value(SETTINGS_RADIO_DISPLAY) == GLOBE
    radio.show_choice()
    assert not choice.tile(GLOBE).isEnabled()
    assert choice.current() == SLIDESHOW
    radio.show_player()

    # With GEPOS back, so is the globe.
    controller.dsp_worker.data_services_updated.emit(components)
    assert radio.shown_display() == GLOBE
    assert radio._slide_area.currentWidget() is panel

    # After the next start: chosen, shown once GEPOS is received again.
    controller.shutdown()
    window = touch_window(data_directory)
    assert window._radio_page.display() == GLOBE
    assert window._radio_page.shown_display() == SLIDESHOW
    _receive_5c(window, fake_workers)
    assert window._radio_page.shown_display() == GLOBE


def test_a_finger_resting_on_the_globe_leads_to_the_choice(
    data_directory, touch_window, fake_workers
):
    window = touch_window(data_directory)
    radio = window._radio_page
    _receive_5c(window, fake_workers)
    radio.show_display(GLOBE)
    area = radio._slide_area
    centre = QPoint(SLIDE_SIZE.width() // 2, SLIDE_SIZE.height() // 2)

    # Qt gives a finger (and the mouse) to the widget found there: the area, not the
    # view, which would take the touch for turning it.
    assert window.childAt(area.mapTo(window, centre)) is area
    _touch(area, QEvent.Type.TouchBegin, [(0, PRESSED, QPointF(centre))])
    assert area.held()
    _spin(LONG_PRESS_MS + 200)
    assert radio.choosing()
    _touch(area, QEvent.Type.TouchEnd, [(0, RELEASED, QPointF(centre))])


def test_two_fingers_bring_the_globe_nearer_one_still_taps(
    data_directory, touch_window, fake_workers
):
    window = touch_window(data_directory)
    radio = window._radio_page
    _receive_5c(window, fake_workers)
    radio.show_display(GLOBE)
    window._render_plots()
    area = radio._slide_area
    camera = radio._slide_area.currentWidget().view.camera
    left, right = QPointF(270, 240), QPointF(370, 240)

    # Spread to twice the distance: twice as near; no long press, however long it took.
    _touch(area, QEvent.Type.TouchBegin, [(0, PRESSED, left), (1, PRESSED, right)])
    for step in range(1, 11):
        shift = QPointF(5 * step, 0)
        _touch(area, QEvent.Type.TouchUpdate, [(0, MOVED, left - shift), (1, MOVED, right + shift)])
    _spin(LONG_PRESS_MS + 200)
    _touch(area, QEvent.Type.TouchEnd, [(0, RELEASED, left), (1, RELEASED, right)])
    assert camera.distance_km == pytest.approx(gnss_view.RADIO_GLOBE_DISTANCE_KM / 2)
    assert not radio.choosing()
    assert not area._shade.isVisible()
    # Turned or shifted it is not, also by two fingers moving together.
    _touch(area, QEvent.Type.TouchBegin, [(0, PRESSED, left), (1, PRESSED, right)])
    for step in range(1, 11):
        shift = QPointF(6 * step, 3 * step)
        _touch(area, QEvent.Type.TouchUpdate, [(0, MOVED, left + shift), (1, MOVED, right + shift)])
    _touch(area, QEvent.Type.TouchEnd, [(0, RELEASED, left), (1, RELEASED, right)])
    assert (camera.right_km, camera.up_km) == (0.0, 0.0)
    assert (camera.latitude_deg, camera.longitude_deg) == pytest.approx(
        (
            window._controller.observer.latitude - 24,
            window._controller.observer.longitude - 40,
        )
    )

    # Nor by the mouse of a PC: neither the middle button turns it nor the right one
    # shifts it.
    longitude = camera.longitude_deg
    _drag(area, Qt.MouseButton.MiddleButton, QPoint(300, 240), QPoint(360, 240))
    _drag(area, Qt.MouseButton.RightButton, QPoint(300, 240), QPoint(360, 260))
    assert camera.longitude_deg == longitude
    assert (camera.right_km, camera.up_km) == (0.0, 0.0)
    assert not radio.choosing() and not area.held()

    # A second finger coming late ends the long press; the wheel of a PC zooms too.
    _touch(area, QEvent.Type.TouchBegin, [(0, PRESSED, left)])
    _touch(area, QEvent.Type.TouchUpdate, [(0, STILL, left), (1, PRESSED, right)])
    assert not area.held()
    _touch(area, QEvent.Type.TouchEnd, [(0, RELEASED, left), (1, RELEASED, right)])
    distance = camera.distance_km
    QApplication.sendEvent(area, _wheel(area, 120))
    assert camera.distance_km == pytest.approx(distance * WHEEL_ZOOM)

    # On the slideshow two fingers do nothing, one finger's tap opens the gallery.
    radio.show_display(SLIDESHOW)
    _touch(area, QEvent.Type.TouchBegin, [(0, PRESSED, left), (1, PRESSED, right)])
    _touch(area, QEvent.Type.TouchUpdate, [(0, MOVED, left - QPointF(40, 0)), (1, STILL, right)])
    _touch(area, QEvent.Type.TouchEnd, [(0, RELEASED, left), (1, RELEASED, right)])
    assert not radio.in_gallery()
    _touch(area, QEvent.Type.TouchBegin, [(0, PRESSED, left)])
    _touch(area, QEvent.Type.TouchEnd, [(0, RELEASED, left)])
    assert radio.in_gallery()


def test_two_fingers_turn_and_zoom_the_null_symbol_and_it_keeps_its_view(
    data_directory, touch_window
):
    window = touch_window(data_directory)
    radio = window._radio_page
    radio.show_display(NULL_SYMBOL)
    area = radio._slide_area
    view = area.currentWidget().view
    assert (view.camera.azimuth_deg, view.camera.distance) == (CAMERA.azimuth_deg, CAMERA.distance)
    left, right = QPointF(270, 240), QPointF(370, 240)

    # Moving together, they turn it, as on the detail page; no long press.
    _touch(area, QEvent.Type.TouchBegin, [(0, PRESSED, left), (1, PRESSED, right)])
    for step in range(1, 11):
        shift = QPointF(8 * step, 0)
        _touch(area, QEvent.Type.TouchUpdate, [(0, MOVED, left + shift), (1, MOVED, right + shift)])
    _spin(LONG_PRESS_MS + 200)
    _touch(area, QEvent.Type.TouchEnd, [(0, RELEASED, left), (1, RELEASED, right)])
    assert view.camera.azimuth_deg == pytest.approx(CAMERA.azimuth_deg - 80 * 180 / view.width())
    assert view.camera.distance == pytest.approx(CAMERA.distance)
    assert not radio.choosing()
    # Spreading to twice the distance: twice as near.
    _touch(area, QEvent.Type.TouchBegin, [(0, PRESSED, left), (1, PRESSED, right)])
    for step in range(1, 11):
        shift = QPointF(5 * step, 0)
        _touch(area, QEvent.Type.TouchUpdate, [(0, MOVED, left - shift), (1, MOVED, right + shift)])
    _touch(area, QEvent.Type.TouchEnd, [(0, RELEASED, left), (1, RELEASED, right)])
    assert view.camera.distance == pytest.approx(CAMERA.distance / 2)

    # Kept for the next start, as the globe keeps its distance.
    kept = stored_camera(window._controller.settings)
    assert (kept.azimuth_deg, kept.distance) == pytest.approx(
        (view.camera.azimuth_deg, view.camera.distance), abs=0.01
    )
    # On a PC the middle button turns it, and that stays; the right one shifts it not.
    azimuth = view.camera.azimuth_deg
    _drag(area, Qt.MouseButton.MiddleButton, QPoint(300, 240), QPoint(340, 240))
    assert view.camera.azimuth_deg == pytest.approx(azimuth - 40 * 180 / view.width())
    kept = stored_camera(window._controller.settings)
    assert kept.azimuth_deg == pytest.approx(view.camera.azimuth_deg, abs=0.01)
    _drag(area, Qt.MouseButton.RightButton, QPoint(300, 240), QPoint(360, 240))
    assert view.camera.look_at == CAMERA.look_at
    assert not radio.choosing() and not area.held()

    # Taps change nothing there, as on the globe: no double tap back to the first view.
    turned = (view.camera.azimuth_deg, view.camera.distance)
    for _ in range(2):
        _touch(area, QEvent.Type.TouchBegin, [(0, PRESSED, left)])
        _touch(area, QEvent.Type.TouchEnd, [(0, RELEASED, left)])
    assert (view.camera.azimuth_deg, view.camera.distance) == turned
    assert not radio.in_gallery()


def test_the_stored_camera_of_the_null_symbol_stays_within_bounds(data_directory):
    settings = QSettings(str(data_directory / "camera.ini"), QSettings.Format.IniFormat)
    assert stored_camera(settings) == CAMERA  # none stored
    for edited in ("nonsense", "10 20", "nan 30 4", "1 2 3 4"):
        settings.setValue(SETTINGS_RADIO_NULL_SYMBOL_CAMERA, edited)
        assert stored_camera(settings) == CAMERA, edited
    settings.setValue(SETTINGS_RADIO_NULL_SYMBOL_CAMERA, "250 120 100")
    camera = stored_camera(settings)
    assert (camera.azimuth_deg, camera.elevation_deg, camera.distance) == (-110.0, 89.0, 9.0)
    assert camera.look_at == CAMERA.look_at


def _drag(widget, button: Qt.MouseButton, start: QPoint, end: QPoint) -> None:
    QTest.mousePress(widget, button, pos=start)
    QTest.mouseMove(widget, end)
    QTest.mouseRelease(widget, button, pos=end)


def _wheel(widget, delta: int) -> QWheelEvent:
    at = QPointF(100, 100)
    return QWheelEvent(
        at,
        QPointF(widget.mapToGlobal(at.toPoint())),
        QPoint(),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def test_the_globe_keeps_its_distance_for_the_next_start(
    data_directory, touch_window, fake_workers
):
    window = touch_window(data_directory)
    _receive_5c(window, fake_workers)
    window._radio_page.show_display(GLOBE)
    area = window._radio_page._slide_area
    QApplication.sendEvent(area, _wheel(area, 240))
    distance = gnss_view.RADIO_GLOBE_DISTANCE_KM * WHEEL_ZOOM**2
    assert area.currentWidget().view.camera.distance_km == pytest.approx(distance)

    window._controller.shutdown()
    window = touch_window(data_directory)
    _receive_5c(window, fake_workers)
    panel = window._radio_page._slide_area.currentWidget()
    assert isinstance(panel, GlobePanel)
    assert panel.view.camera.distance_km == pytest.approx(distance)


def test_offered_displays_follow_the_user_applications():
    components = data_components(_ensemble(FIGS_5C))
    every = {display.key for display in DISPLAYS}
    assert offered_displays(components) == every
    assert offered_displays(_without_gepos(components)) == every - {GLOBE}
    assert offered_displays([]) == every - {GLOBE}


def test_the_choice_is_kept_for_the_next_start(data_directory, touch_window):
    window = touch_window(data_directory)
    window._radio_page.show_display(SPECTRUM)
    assert window._controller.settings.value(SETTINGS_RADIO_DISPLAY) == SPECTRUM

    window._controller.shutdown()
    window = touch_window(data_directory)
    assert window._radio_page.display() == SPECTRUM
    assert not window._radio_page.choosing()


def test_an_unknown_display_in_the_settings_shows_the_slideshow(data_directory, touch_window):
    window = touch_window(data_directory)
    window._controller.settings.setValue(SETTINGS_RADIO_DISPLAY, "wetter")

    window._controller.shutdown()
    window = touch_window(data_directory)
    assert window._radio_page.display() == SLIDESHOW


def test_the_panel_is_drawn_only_while_the_radio_page_shows_it(data_directory, touch_window):
    window = touch_window(data_directory)
    radio = window._radio_page
    radio.show_display(SPECTRUM)
    panel = radio._slide_area.currentWidget()
    dsp = window._controller.dsp_worker
    freqs = np.linspace(-1e6, 1e6, 64)

    window.show_page(SIGNAL)
    dsp.spectrum_ready.emit(freqs, np.full(64, -40.0))
    window._render_plots()
    assert panel.curve.getData()[0] is None
    # The signal page has a spectrum of its own.
    overview = window._signal_page.page(OVERVIEW)
    assert len(overview._panels[0].curve.getData()[0]) == 64

    radio.show_choice()
    window._render_plots()
    assert panel.curve.getData()[0] is None

    # The rail's "Radio" leads back from the choice, as from a sub-page, and draws.
    window.show_page(RADIO)
    assert not radio.choosing()
    assert len(panel.curve.getData()[0]) == 64


def test_back_leads_to_the_player_without_changing_the_display(data_directory, touch_window):
    window = touch_window(data_directory)
    radio = window._radio_page
    radio.show_display(CONSTELLATION)
    radio.show_choice()

    radio._choice.back_requested.emit()

    assert not radio.choosing()
    assert radio.display() == CONSTELLATION


def test_the_figures_chosen_later_count_the_frames_since_the_station(data_directory, touch_window):
    window = touch_window(data_directory)
    radio = window._radio_page
    dsp = window._controller.dsp_worker
    dsp.superframe_ready.emit(_superframe())
    dsp.superframe_ready.emit(_superframe())

    radio.show_display(SOUND_FIGURES)

    figures = radio._slide_area.currentWidget().figures
    assert figures.text("frames") == "2/2 Superframes"
    assert figures.text("corrected") == "6 bytes (Reed-Solomon)"
    window._controller.service_reset.emit()
    assert figures.text("frames") == "-"


def test_every_display_has_a_picture_of_its_own_in_both_schemes():
    pictures = {
        (display.key, key): preview_pixmap(display.key, THEMES[key]).toImage()
        for display in DISPLAYS
        for key in (DARK, LIGHT)
    }

    assert all(not image.isNull() for image in pictures.values())
    for key in (DARK, LIGHT):
        images = [pictures[(display.key, key)] for display in DISPLAYS]
        assert all(a != b for i, a in enumerate(images) for b in images[i + 1 :])
    assert pictures[(SLIDESHOW, DARK)] != pictures[(SLIDESHOW, LIGHT)]
