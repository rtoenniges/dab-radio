"""The picture of the radio starting (`ui.touch.splash`), the boot splash written from it
(`pi/plymouth/dab-radio`) and that `main.py` shows the picture before it imports the
rest of the radio."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor, QFont, QFontInfo, QImage, QPainter
from PyQt6.QtWidgets import QApplication, QWidget

from ui.touch import splash
from ui.touch.theme import DARK, FONT_FILES, THEMES

REPOSITORY = Path(__file__).resolve().parent.parent
THEME = REPOSITORY / "pi" / "plymouth" / "dab-radio"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _render(width: int, height: int) -> QImage:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    painter = QPainter(image)
    splash.paint_splash(painter, QRectF(0, 0, width, height))
    painter.end()
    return image


def _wait_until(condition, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while not condition() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)
    return condition()


def test_colours_and_typeface_are_those_of_the_dark_theme() -> None:
    dark = THEMES[DARK]
    assert splash.BACKGROUND == dark.background
    assert splash.ACCENT == dark.accent
    assert splash.TEXT == dark.text
    assert splash.MUTED == dark.muted
    assert splash.FONT_FILES == FONT_FILES


def test_dots_gather_round_the_four_points_on_the_diagonals() -> None:
    dots = splash.constellation()
    assert len(dots) == 4 * splash.DOTS_PER_POINT
    ring = splash.POINT_OFFSET * math.sqrt(2)
    for point in range(4):
        theta = math.pi / 4 + point * math.pi / 2
        group = dots[point * splash.DOTS_PER_POINT : (point + 1) * splash.DOTS_PER_POINT]
        xs = [dot.radius * math.cos(dot.angle) for dot in group]
        ys = [dot.radius * math.sin(dot.angle) for dot in group]
        assert sum(xs) / len(xs) == pytest.approx(ring * math.cos(theta), abs=3)
        assert sum(ys) / len(ys) == pytest.approx(ring * math.sin(theta), abs=3)
        for x, y in zip(xs, ys, strict=True):
            assert math.hypot(x - ring * math.cos(theta), y - ring * math.sin(theta)) < (
                4 * splash.SCATTER
            )
        # Before the lock the dots of a point fill its quarter of the ring.
        spreads = sorted(dot.spread for dot in group)
        assert spreads[0] < -0.9 * math.pi / 4
        assert spreads[-1] > 0.9 * math.pi / 4


def test_dots_are_the_same_every_time() -> None:
    splash.constellation.cache_clear()
    assert splash.constellation() == splash.constellation.__wrapped__()


def test_picture_shows_dots_and_name_on_the_background(qapp) -> None:
    image = _render(splash.WIDTH, splash.HEIGHT)
    background = QColor(splash.BACKGROUND).rgb()
    assert image.pixel(0, 0) == background
    assert image.pixel(splash.WIDTH - 1, splash.HEIGHT - 1) == background
    dot = splash.constellation()[0]
    x = round(splash.WIDTH / 2 + dot.radius * math.cos(dot.angle))
    y = round(splash.CENTRE_Y + dot.radius * math.sin(dot.angle))
    assert QColor(image.pixel(x, y)).blue() > 200
    # Something of the name is drawn on its line.
    title_row = [image.pixel(x, splash.TITLE_CENTRE_Y) for x in range(splash.WIDTH)]
    assert any(pixel != background for pixel in title_row)


def test_picture_is_scaled_into_the_middle_of_another_screen(qapp) -> None:
    image = _render(720, 1280)
    background = QColor(splash.BACKGROUND).rgb()
    # Scaled by 720 / 1280 into the middle: the top quarter stays empty.
    assert all(image.pixel(x, y) == background for x in range(0, 720, 8) for y in range(0, 400, 8))
    scale = 720 / splash.WIDTH
    top = (1280 - splash.HEIGHT * scale) / 2
    dot = splash.constellation()[0]
    x = round(360 + dot.radius * math.cos(dot.angle) * scale)
    y = round(top + (splash.CENTRE_Y + dot.radius * math.sin(dot.angle)) * scale)
    assert QColor(image.pixel(x, y)).blue() > 150


def test_boot_splash_carries_the_table_of_the_picture() -> None:
    """tools/build_boot_splash.py has run since the picture last changed."""
    script = (THEME / "dab-radio.script").read_text(encoding="utf-8")
    assert splash.with_plymouth_table(script) == script
    assert f"dots.count = {len(splash.constellation())};" in script


def test_boot_splash_has_the_images_its_script_loads(qapp) -> None:
    script = (THEME / "dab-radio.script").read_text(encoding="utf-8")
    for name, image in splash.splash_images().items():
        assert f'Image("{name}")' in script
        stored = QImage(str(THEME / name))
        assert not stored.isNull(), name
        if name != splash.TITLE_IMAGE:
            # The name's image depends on how the font is drawn; the others do not.
            assert stored.size() == image.size(), name
    theme = (THEME / "dab-radio.plymouth").read_text(encoding="utf-8")
    assert "ModuleName=script" in theme
    assert "ScriptFile=/usr/share/plymouth/themes/dab-radio/dab-radio.script" in theme


def test_picture_leaves_the_regular_typeface_to_the_surface(qapp) -> None:
    """The picture registers the typeface before the surface does, its regular weight
    as well: the surface then registers nothing, and its text would be bold."""
    splash.title_font()
    regular = QFont(splash.FONT_FAMILY)
    regular.setPixelSize(20)
    bold = QFont(regular)
    bold.setWeight(QFont.Weight.Bold)
    assert QFontInfo(regular).family() == splash.FONT_FAMILY
    assert QFontInfo(regular).weight() < QFontInfo(bold).weight()


def test_table_without_its_marks_is_refused() -> None:
    with pytest.raises(ValueError):
        splash.with_plymouth_table("Plymouth.SetRefreshFunction(refresh);\n")


def test_startup_splash_closes_once_the_window_has_come_up_over_it(qapp) -> None:
    startup = splash.StartupSplash()
    closed = []
    startup.destroyed.connect(lambda: closed.append(True))
    started = time.monotonic()
    assert startup.show_now()
    assert time.monotonic() - started < splash.SHOW_TIMEOUT_MS / 1000
    window = QWidget()
    window.show()
    startup.close_over(window)
    assert _wait_until(lambda: bool(closed))
    window.close()


def test_startup_splash_closes_over_a_window_already_shown(qapp) -> None:
    startup = splash.StartupSplash()
    closed = []
    startup.destroyed.connect(lambda: closed.append(True))
    startup.show_now()
    window = QWidget()
    window.show()
    assert _wait_until(lambda: window.windowHandle().isExposed())
    startup.close_over(window)
    assert _wait_until(lambda: bool(closed))
    window.close()


def test_main_imports_nothing_heavy_before_the_picture() -> None:
    """The picture is to come up at once: importing `main` brings neither the controller
    nor numpy, pyqtgraph or the signal processing along."""
    code = (
        "import sys; import main; "
        "heavy = [m for m in ('numpy', 'pyqtgraph', 'numba', 'ui.receiver_controller', "
        "'ui.touch.window', 'ui.main_window', 'ui.touch.theme') if m in sys.modules]; "
        "print(','.join(heavy))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY / "src",
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY / "src")},
    )
    assert result.stdout.strip() == ""
