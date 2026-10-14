"""The slides of the station playing on the touch surface: a tap on the slideshow of the
radio page shows those received since the station was chosen, the chosen one large
beside the thumbnails, without keeping anything beyond the holding buffer."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QBuffer, QIODevice, QPoint, Qt
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from dab.mot import CONTENT_TYPE_IMAGE, MotObject
from dab.slideshow import PARAM_TRIGGER_TIME
from tests.fake_receiver import SERVICES_5C
from ui.touch.display_choice import CONSTELLATION
from ui.touch.radio_page import SLIDE_SIZE
from ui.touch.slide_gallery import _EMPTY, DETAIL_SIZE, count_text, rounded_slide
from ui.touch.theme import PANEL_RADIUS
from ui.touch.window import RADIO, TouchWindow

APP = QApplication.instance() or QApplication([])

# TriggerTime "NOW": a validity bit of 0 (ETSI TS 101 499, clause 6.2.4).
NOW = {PARAM_TRIGGER_TIME: bytes(4)}


@pytest.fixture
def window(data_directory, make_controller):
    controller = make_controller(data_directory, start=False)
    window = TouchWindow(controller)
    controller.start()
    window.show()
    window.show_page(RADIO)
    yield window
    window.close()
    window.deleteLater()


def _png(colour: int) -> bytes:
    image = QImage(320, 240, QImage.Format.Format_RGB32)
    image.fill(colour)
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return bytes(buffer.data())


def _receive(window, transport_id: int, name: str, at: float, parameters=NOW) -> None:
    """A slide arriving at broadcast time `at`."""
    radio = window._radio_page
    radio._clock.now = lambda: at
    data = _png(0x101010 * transport_id)
    slide = MotObject(transport_id, CONTENT_TYPE_IMAGE, 3, name, data, dict(parameters or {}))
    window._controller.dsp_worker.slide_ready.emit(slide)


def _tap(area) -> None:
    centre = QPoint(SLIDE_SIZE.width() // 2, SLIDE_SIZE.height() // 2)
    QTest.mousePress(area, Qt.MouseButton.LeftButton, pos=centre)
    QTest.mouseRelease(area, Qt.MouseButton.LeftButton, pos=centre)


def test_a_tap_on_the_slideshow_shows_the_slides_of_the_station(window):
    radio = window._radio_page
    _receive(window, 1, "20261007_1400Z_NEWS_1.jpg", 1_791_381_600.0)
    _receive(window, 2, "20261007_1402Z_WETTER_2.jpg", 1_791_381_720.0)

    _tap(radio._slide_area)

    assert radio.in_gallery()
    gallery = radio._gallery
    # The newest first; the one on the radio page chosen and marked.
    assert [slide.name for slide in gallery.slides()] == [
        "20261007_1402Z_WETTER_2.jpg",
        "20261007_1400Z_NEWS_1.jpg",
    ]
    assert gallery.selected().name == "20261007_1402Z_WETTER_2.jpg"
    assert gallery.thumbnail(0).text().endswith("●")
    assert not gallery.thumbnail(1).text().endswith("●")
    assert gallery.thumbnail(0).property("current")
    assert gallery._detail.pixmap().size() == DETAIL_SIZE
    assert gallery._caption.text().endswith(" · Wetter")
    assert gallery._summary.text().endswith("2 pictures")


def test_a_tap_on_a_panel_in_the_place_of_the_slideshow_does_nothing(window):
    radio = window._radio_page
    radio.show_display(CONSTELLATION)

    _tap(radio._slide_area)

    assert not radio.in_gallery()


def test_a_thumbnail_shows_its_slide_large(window):
    radio = window._radio_page
    _receive(window, 1, "eins.jpg", 1_791_381_600.0)
    _receive(window, 2, "zwei.jpg", 1_791_381_720.0)
    radio.show_gallery()
    gallery = radio._gallery

    gallery.thumbnail(1).click()

    assert gallery.selected().name == "eins.jpg"
    assert gallery.thumbnail(1).property("current")
    assert not gallery.thumbnail(0).property("current")
    # The radio page goes on with the station's slide.
    assert radio._slides.live_slide().name == "zwei.jpg"


def test_slides_arriving_meanwhile_join_the_thumbnails(window):
    radio = window._radio_page
    _receive(window, 1, "eins.jpg", 1_791_381_600.0)
    radio.show_gallery()
    gallery = radio._gallery
    first = gallery.thumbnail(0)

    # Nothing new: the thumbnails stay as they are.
    radio._on_tick()
    assert gallery.thumbnail(0) is first

    _receive(window, 2, "zwei.jpg", 1_791_381_720.0)
    assert [slide.name for slide in gallery.slides()] == ["zwei.jpg", "eins.jpg"]
    # The slide chosen stays chosen; the mark follows the radio page.
    assert gallery.selected().name == "eins.jpg"
    assert gallery.thumbnail(0).text().endswith("●")


def test_the_slides_go_with_the_station(window):
    radio = window._radio_page
    _receive(window, 1, "eins.jpg", 1_791_381_600.0)
    radio.show_gallery()

    window._controller.select_service(SERVICES_5C[0][0])

    gallery = radio._gallery
    assert gallery.slides() == []
    assert gallery.selected() is None
    assert gallery._detail.text() == _EMPTY
    assert gallery._summary.text().endswith("no picture")


def test_a_slide_is_received_at_the_broadcast_time(window):
    _receive(window, 1, "eins.jpg", 1_791_381_600.0)

    assert window._radio_page._slides.live_slide().received_at == 1_791_381_600.0


def test_a_slide_the_station_has_not_shown_says_so(window):
    radio = window._radio_page
    _receive(window, 1, "vorab.jpg", 1_791_381_600.0, parameters=None)

    radio.show_gallery()

    assert radio._gallery._caption.text().endswith("not shown by the station")


def test_back_and_the_rail_lead_to_the_player(window):
    radio = window._radio_page
    radio.show_gallery()
    radio._gallery.back_button.click()
    assert not radio.in_gallery()

    radio.show_gallery()
    window.show_page(RADIO)
    assert not radio.in_gallery()


def test_the_count_is_said_in_words():
    assert count_text(0) == "no picture"
    assert count_text(1) == "1 picture"
    assert count_text(12) == "12 pictures"


def test_a_slide_filling_the_area_has_the_rounded_corners_of_a_panel():
    filling = QPixmap(DETAIL_SIZE)
    filling.fill(QColor("red"))
    image = rounded_slide(filling, DETAIL_SIZE).toImage()
    right, bottom = DETAIL_SIZE.width() - 1, DETAIL_SIZE.height() - 1
    for x, y in ((0, 0), (right, 0), (0, bottom), (right, bottom)):
        assert image.pixelColor(x, y).alpha() == 0
    assert image.pixelColor(PANEL_RADIUS, 0).alpha() == 255
    assert image.pixelColor(DETAIL_SIZE.width() // 2, DETAIL_SIZE.height() // 2) == QColor("red")


def test_a_small_slide_stays_centred_and_a_big_one_is_cropped_at_right_and_bottom():
    small = QPixmap(100, 50)
    small.fill(QColor("red"))
    image = rounded_slide(small, DETAIL_SIZE).toImage()
    assert image.pixelColor(320, 240) == QColor("red")
    assert image.pixelColor(260, 240).alpha() == 0

    big = QPixmap(800, 600)
    big.fill(QColor("blue"))
    big_image = big.toImage()
    big_image.setPixelColor(PANEL_RADIUS, PANEL_RADIUS, QColor("red"))
    image = rounded_slide(QPixmap.fromImage(big_image), DETAIL_SIZE).toImage()
    assert image.pixelColor(PANEL_RADIUS, PANEL_RADIUS) == QColor("red")
