"""The symbol of the signal page in the rail glows up red and fades again while the
receiver is overloaded."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QToolButton

from ui.touch.glow import FADE_S, GLOW_LEVELS, Glow, GlowingSymbol, glow_pixmap
from ui.touch.icons import symbol_pixmap
from ui.touch.theme import DARK, THEMES
from ui.touch.window import SIGNAL, TouchWindow

APP = QApplication.instance() or QApplication([])


def test_the_glow_fades_up_and_down_in_the_same_time() -> None:
    glow = Glow()
    glow.start()
    assert glow.brightness == 0.0

    glow.advance(FADE_S / 2)
    half_up = glow.brightness
    glow.advance(FADE_S / 2)
    assert glow.brightness == pytest.approx(1.0)
    glow.advance(FADE_S / 2)
    assert glow.brightness == pytest.approx(half_up)
    glow.advance(FADE_S / 2)
    assert glow.brightness == pytest.approx(0.0)
    glow.advance(FADE_S / 4)  # and up again
    assert 0.0 < glow.brightness < half_up
    assert glow.glowing


def test_stopped_while_fading_up_it_fades_out_from_there() -> None:
    glow = Glow()
    glow.start()
    glow.advance(FADE_S * 0.3)
    brightness = glow.brightness

    glow.stop()
    assert glow.brightness == pytest.approx(brightness)
    glow.advance(FADE_S * 0.1)
    assert glow.brightness < brightness
    glow.advance(FADE_S * 0.3)
    assert not glow.glowing
    assert glow.brightness == 0.0


def test_dark_the_symbol_is_drawn_as_ever() -> None:
    theme = THEMES[DARK]
    plain = symbol_pixmap("signal", theme.color("muted"), 36).toImage()

    assert glow_pixmap("signal", theme.color("muted"), theme, 0.0, 36).toImage() == plain
    glowing = glow_pixmap("signal", theme.color("muted"), theme, 1.0, 36).toImage()
    assert glowing != plain
    # The middle of the wave, at its full glow, in the danger colour of the scheme.
    assert glowing.pixelColor(15, 18).name() == theme.danger


def test_the_button_follows_the_glow_and_goes_back_to_its_symbol() -> None:
    button = QToolButton()
    button.setIcon(QIcon())
    symbol = GlowingSymbol(button, "signal", QSize(36, 36), lambda: True)
    symbol.set_theme(THEMES[DARK])
    plain = button.icon().pixmap(36, 36).toImage()

    symbol.set_glowing(True)
    symbol.glow.advance(FADE_S)
    symbol._advance()
    assert symbol.level == GLOW_LEVELS - 1
    assert button.icon().pixmap(36, 36).toImage() != plain

    symbol.set_glowing(False)
    symbol.glow.advance(FADE_S)
    symbol._advance()
    assert symbol.level == 0
    assert not symbol._timer.isActive()
    assert button.icon().pixmap(36, 36).toImage() == plain


def test_nobody_seeing_the_screen_the_symbol_is_left_as_it_is() -> None:
    button = QToolButton()
    seen = [False]
    symbol = GlowingSymbol(button, "signal", QSize(36, 36), lambda: seen[0])
    symbol.set_theme(THEMES[DARK])

    symbol.set_glowing(True)
    symbol.glow.advance(FADE_S)
    symbol._advance()
    assert symbol.level == 0

    seen[0] = True
    symbol._advance()
    assert symbol.level == GLOW_LEVELS - 1


@pytest.fixture
def window(data_directory, make_controller, fake_workers):
    controller = make_controller(data_directory, start=False)
    window = TouchWindow(controller)
    controller.start()
    yield window
    window.close()
    window.deleteLater()


def test_the_rail_glows_while_the_receiver_is_overloaded(window) -> None:
    controller = window._controller
    glow = window._overload_glow.glow
    controller.start_reception()
    controller.dsp_worker.stream_restarted.emit(controller._worker.tuning)

    controller.dsp_worker.clipping_ready.emit(0.2)
    assert glow.blinking
    assert window._overload_glow._timer.isActive()

    controller.stop_reception()
    controller._worker.stopped.emit()
    assert not glow.blinking


def test_a_new_scheme_keeps_the_glow_of_the_signal_symbol(window) -> None:
    symbol = window._overload_glow
    symbol.set_glowing(True)
    symbol.glow.advance(FADE_S)
    symbol._advance()
    glowing = window._nav_buttons[SIGNAL].icon().pixmap(36, 36).toImage()

    window.set_theme("light")
    window.set_theme(DARK)

    assert window._nav_buttons[SIGNAL].icon().pixmap(36, 36).toImage() == glowing
