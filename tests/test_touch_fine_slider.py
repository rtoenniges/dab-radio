"""The slider a finger sets finer by pulling up (`ui.touch.fine_slider`): the speed by
the height of the finger, the ticks of the loupe's scale, and a finger dragging."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication, QWidget

from ui.touch.fine_slider import (
    FACTOR_TEXTS,
    FINE_LEVELS,
    FINGER_CLEARANCE,
    LEVEL_HYSTERESIS,
    LOUPE_MARGIN,
    FineSlider,
    fine_level,
    scale_ticks,
)

STEPS = (1, 5, 10, 50, 100)
# A slider of 360 px: 320 px for the middle of its handle, 100 units over them.
WIDTH = 360
UNITS_PER_PX = 100 / 320


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window():
    window = QWidget()
    window.resize(1280, 720)
    window.show()
    yield window
    window.deleteLater()


def _slider(window: QWidget, jump: bool = False) -> FineSlider:
    slider = FineSlider(lambda value: f"{value} %", STEPS, jump=jump, parent=window)
    slider.setRange(0, 100)
    slider.setValue(50)
    slider.setGeometry(1280 - WIDTH - 24, 620, WIDTH, 72)
    slider.show()
    return slider


def _send(slider: FineSlider, kind: QEvent.Type, x: float, y: float) -> None:
    """A finger as Qt hands it to a widget: a mouse at (x, y) on the slider."""
    local = QPointF(x, y)
    left = Qt.MouseButton.LeftButton
    button = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseMove else left
    buttons = Qt.MouseButton.NoButton if kind == QEvent.Type.MouseButtonRelease else left
    event = QMouseEvent(
        kind, local, slider.mapToGlobal(local), button, buttons, Qt.KeyboardModifier.NoModifier
    )
    QApplication.sendEvent(slider, event)


def _press(slider, x, y=36):
    _send(slider, QEvent.Type.MouseButtonPress, x, y)


def _move(slider, x, y):
    _send(slider, QEvent.Type.MouseMove, x, y)


def _release(slider, x, y):
    _send(slider, QEvent.Type.MouseButtonRelease, x, y)


# --- Speeds and ticks -------------------------------------------------------------------


def test_the_speed_follows_the_height_of_the_finger_without_flickering():
    assert [factor for _rise, factor in FINE_LEVELS] == [1.0, 0.5, 0.25, 0.125]
    assert len(FACTOR_TEXTS) == len(FINE_LEVELS)
    assert fine_level(-36, 0) == 0  # on the slider
    assert fine_level(39, 0) == 0 and fine_level(40, 0) == 1
    assert fine_level(120, 1) == 2 and fine_level(500, 0) == 3
    # Back down, faster again only well below the threshold.
    assert fine_level(40 - LEVEL_HYSTERESIS, 1) == 1
    assert fine_level(40 - LEVEL_HYSTERESIS - 1, 1) == 0
    assert fine_level(0, 3) == 0


def test_the_ticks_of_the_scale_spread_apart_at_a_slower_speed():
    # At full speed 1 % is 3.2 px, too close: ticks every 5 %, taller at 50 %.
    ticks = scale_ticks(50, UNITS_PER_PX, 150, STEPS, 0, 100)
    assert [tick.value for tick in ticks] == list(range(5, 100, 5))
    middle = next(tick for tick in ticks if tick.value == 50)
    beside = next(tick for tick in ticks if tick.value == 55)
    assert middle.offset == 0 and middle.size == 1 and middle.label == 1
    assert beside.offset == pytest.approx(16) and beside.alpha == 1
    assert beside.size < 0.1 and beside.label == 0

    # At an eighth, 1 % is 25.6 px: a tick for every percent, 5 % with a label.
    ticks = scale_ticks(50, UNITS_PER_PX / 8, 150, STEPS, 0, 100)
    assert [tick.value for tick in ticks] == list(range(45, 56))
    assert ticks[6].offset == pytest.approx(25.6) and ticks[6].label == 0
    assert ticks[0].label == 1  # 45 %

    # None beyond the ends.
    assert scale_ticks(2, UNITS_PER_PX, 150, STEPS, 0, 100)[0].value == 0
    assert scale_ticks(98, UNITS_PER_PX, 150, STEPS, 0, 100)[-1].value == 100


# --- The finger -------------------------------------------------------------------------


def test_a_finger_on_the_handle_drags_it_and_moves_it_finer_the_higher_it_rises(window):
    slider = _slider(window)
    values, released = [], []
    slider.valueChanged.connect(values.append)
    slider.sliderReleased.connect(lambda: released.append(slider.value()))
    handle = 20 + 50 * 3.2

    _press(slider, handle + 10)  # beside the handle's middle, still on it
    assert slider.isSliderDown() and slider.value() == 50
    _move(slider, handle + 42, 36)  # at full speed, as far as the finger
    assert slider.value() == 60
    loupe = slider.loupe()
    assert loupe is not None and loupe.isVisible()
    assert loupe.text == "60 %" and loupe.level == 0

    # Pulled up, the value stays; then it follows an eighth of the finger.
    _move(slider, handle + 42, -250)
    assert slider.value() == 60 and slider.level() == 3 and loupe.level == 3
    _move(slider, handle + 74, -250)
    assert slider.value() == 61 and loupe.text == "61 %"
    _move(slider, handle + 202, -250)  # 128 px: 5 %
    assert slider.value() == 66
    # The loupe stands above the finger, within the window.
    finger = slider.mapTo(window, QPointF(handle + 202, -250).toPoint())
    assert loupe.geometry().bottom() < finger.y() - FINGER_CLEARANCE
    assert loupe.geometry().right() <= window.width() - LOUPE_MARGIN

    # Back on the slider at full speed: no jump, the handle keeps its distance.
    _move(slider, handle + 202, 36)
    assert slider.value() == 66 and slider.level() == 0
    _move(slider, handle + 170, 36)
    assert slider.value() == 56

    _release(slider, handle + 170, 36)
    assert released == [56] and not slider.isSliderDown()
    assert not loupe.isVisible()
    assert values == [60, 61, 66, 56]


def test_a_tap_beside_the_handle_pages_and_the_player_jumps_there(window):
    slider = _slider(window)
    _press(slider, 340)  # at the right end, where 100 % would be
    _move(slider, 340, -250)
    _release(slider, 340, -250)
    assert slider.value() == 60  # a page, never full volume
    assert slider.loupe() is None

    player = _slider(window, jump=True)
    released = []
    player.sliderReleased.connect(lambda: released.append(player.value()))
    _press(player, 340)
    assert player.value() == 100 and player.isSliderDown()
    _move(player, 340, -250)
    _move(player, 212, -250)  # 128 px at an eighth: 5
    _release(player, 212, -250)
    assert released == [95]


def test_a_tap_shows_no_loupe_and_hiding_ends_a_drag(window):
    player = _slider(window, jump=True)
    released = []
    player.sliderReleased.connect(lambda: released.append(player.value()))
    _press(player, 100)
    _move(player, 100, 33)  # a finger is never quite still
    _release(player, 100, 33)
    assert released == [25] and player.loupe() is None

    _press(player, 100)
    _move(player, 100, -250)
    assert player.loupe().isVisible()
    player.hide()
    assert not player.isSliderDown() and not player.loupe().isVisible()
    assert released == [25, 25]
