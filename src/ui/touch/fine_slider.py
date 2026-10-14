"""A slider for a finger that sets it finer than its width allows: on the handle the
finger drags it as usual; pulled up off the slider while still down, it moves the value
slower, by a half, a quarter or an eighth of its own movement, the higher the slower
(`FINE_LEVELS`). A change of speed never moves the value: the finger moves it on from
where it is. Back on the slider, the handle keeps its distance to the finger.

The sliders lie at the foot of the screen, so the finger goes up, and the hand then
covers the slider. While the finger drags, a loupe above the fingertip shows the value,
the stretch of the scale around it beneath a handle that stays in the middle, and the
speed, with a ladder of the speeds beside it. The scale in the loupe moves with the
finger pixel for pixel: at an eighth of the speed it is drawn eight times wider, so its
ticks spread apart as the finger rises.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from PyQt6.QtCore import QEasingCurve, QPoint, QPointF, QRectF, Qt, QVariantAnimation
from PyQt6.QtGui import (
    QColor,
    QFont,
    QHideEvent,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
)
from PyQt6.QtWidgets import QSlider, QStyle, QWidget

from ui.touch.pager import SWIPE_SLOP
from ui.touch.theme import (
    DARK,
    LARGE_FONT_PX,
    PANEL_RADIUS,
    SMALL_FONT_PX,
    THEMES,
    TOUCH_TARGET,
    Theme,
)

# The handle of a slider as the style sheet draws it (`ui.touch.theme`).
SLIDER_HANDLE_PX = 40
# How far above the top edge of the slider the finger rises for each speed (px), and
# the speed: the share of the finger's movement the value follows.
FINE_LEVELS = ((0, 1.0), (40, 0.5), (120, 0.25), (200, 0.125))
FACTOR_TEXTS = ("1×", "½×", "¼×", "⅛×")
# How far below the threshold of its speed the finger goes back before it is faster.
LEVEL_HYSTERESIS = 16
# Either side of the handle's middle a finger coming down takes it, wider than the
# handle: a target for a finger.
GRAB_HALF_WIDTH = TOUCH_TARGET / 2
# The loupe: its width, its height with and without labels beneath the scale, the room
# between fingertip and its lower edge, and to the window's edges.
LOUPE_WIDTH = 400
LOUPE_HEIGHT = 132
LOUPE_HEIGHT_UNLABELLED = 112
FINGER_CLEARANCE = 72
LOUPE_MARGIN = 8
LOUPE_PADDING = 16
VALUE_FONT_PX = 30
# How long the scale takes to stretch to a new speed.
ZOOM_MS = 160
# A kind of tick (its spacing) fades in as its spacing on the screen grows from the
# first value to the second, its label from the third to the fourth; ticks grow taller
# between the second and the third.
TICK_MIN_PX = 6
TICK_FULL_PX = 14
LABEL_MIN_PX = 56
LABEL_FULL_PX = 72
# The ladder of speeds beside the scale, finest on top.
LADDER_WIDTH = 24
LADDER_BAR = 6
LADDER_GAP = 6
FADE_PX = 32


def fine_level(rise: float, level: int) -> int:
    """The speed (an index of `FINE_LEVELS`) for a finger `rise` px above the top edge
    of the slider (negative on it), coming from `level`: finer once past a threshold,
    faster again only `LEVEL_HYSTERESIS` below it, so that it does not flicker."""
    new = level
    while new + 1 < len(FINE_LEVELS) and rise >= FINE_LEVELS[new + 1][0]:
        new += 1
    while new > 0 and rise < FINE_LEVELS[new][0] - LEVEL_HYSTERESIS:
        new -= 1
    return new


def _ramp(x: float, start: float, end: float) -> float:
    return min(max((x - start) / (end - start), 0.0), 1.0)


@dataclass(frozen=True)
class Tick:
    """A tick of the loupe's scale: where it stands from the handle (px), how opaque
    and how tall it is (0 to 1), how opaque its label."""

    value: int
    offset: float
    alpha: float
    size: float
    label: float


def scale_ticks(
    value: float,
    units_per_px: float,
    half_width: float,
    steps: Sequence[int],
    low: float,
    high: float,
) -> list[Tick]:
    """The ticks within `half_width` px either side of `value` on a scale of
    `units_per_px`, only between `low` and `high`. `steps` are the spacings of ticks,
    each dividing the next (1, 5, 10, 30, ...); a tick is of the widest that divides it,
    and its kind shows as its spacing on the screen grows (`TICK_MIN_PX` ...)."""
    if units_per_px <= 0:
        return []
    finest = next((step for step in steps if step / units_per_px >= TICK_MIN_PX), None)
    if finest is None:
        return []
    start = max(low, value - half_width * units_per_px)
    end = min(high, value + half_width * units_per_px)
    ticks = []
    for index in range(math.ceil(start / finest), math.floor(end / finest) + 1):
        tick = index * finest
        widest = max(step for step in steps if step >= finest and tick % step == 0)
        spacing = widest / units_per_px
        ticks.append(
            Tick(
                tick,
                (tick - value) / units_per_px,
                _ramp(spacing, TICK_MIN_PX, TICK_FULL_PX),
                _ramp(spacing, TICK_FULL_PX, LABEL_MIN_PX),
                _ramp(spacing, LABEL_MIN_PX, LABEL_FULL_PX),
            )
        )
    return ticks


class _Loupe(QWidget):
    """What a slider shows above the finger while it drags (see the module). Taps go
    through to what lies beneath."""

    def __init__(self, parent: QWidget, labelled: bool) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFixedSize(LOUPE_WIDTH, LOUPE_HEIGHT if labelled else LOUPE_HEIGHT_UNLABELLED)
        self._theme = THEMES[DARK]
        self.text = ""
        self.value = 0.0
        self.low = 0.0
        self.high = 0.0
        self.units_per_px = 1.0  # at full speed
        self.steps: Sequence[int] = ()
        self.tick_text: Callable[[int], str] | None = None
        self.level = 0
        # The speed the scale is drawn at, as its power of two, on its way to the level's.
        self._shown_log = 0.0
        self._zoom = QVariantAnimation(self)
        self._zoom.setDuration(ZOOM_MS)
        self._zoom.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._zoom.valueChanged.connect(self._on_zoom)
        self.hide()

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.update()

    def set_level(self, level: int, animated: bool = True) -> None:
        target = math.log2(FINE_LEVELS[level][1])
        if level != self.level or not animated:
            self._zoom.stop()
            if animated:
                self._zoom.setStartValue(self._shown_log)
                self._zoom.setEndValue(target)
                self._zoom.start()
            else:
                self._shown_log = target
        self.level = level
        self.update()

    def shown_units_per_px(self) -> float:
        return self.units_per_px * 2.0**self._shown_log

    def place(self, finger: QPoint) -> None:
        """Above the fingertip (in the window), within the window."""
        rightmost = self.parentWidget().width() - self.width() - LOUPE_MARGIN
        x = min(max(finger.x() - self.width() // 2, LOUPE_MARGIN), rightmost)
        y = max(LOUPE_MARGIN, finger.y() - FINGER_CLEARANCE - self.height())
        self.move(x, y)

    def _on_zoom(self, value: float) -> None:
        self._shown_log = value
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        theme = self._theme
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(theme.color("outline"), 2))
        painter.setBrush(theme.color("surface"))
        painter.drawRoundedRect(
            QRectF(self.rect()).adjusted(1, 1, -1, -1), PANEL_RADIUS, PANEL_RADIUS
        )

        inner = QRectF(self.rect()).adjusted(
            LOUPE_PADDING, LOUPE_PADDING - 4, -LOUPE_PADDING, -LOUPE_PADDING
        )
        header = QRectF(inner.left(), inner.top(), inner.width(), 40)
        font = QFont(self.font())
        font.setPixelSize(VALUE_FONT_PX)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(theme.color("text"))
        align_left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        painter.drawText(header, align_left, self.text)
        font.setPixelSize(LARGE_FONT_PX)
        painter.setFont(font)
        painter.setPen(theme.color("accent"))
        align_right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        painter.drawText(header, align_right, FACTOR_TEXTS[self.level])

        tape = QRectF(inner.left(), header.bottom() + 8, inner.width() - LADDER_WIDTH - 16, 0)
        tape.setBottom(inner.bottom())
        self._paint_tape(painter, tape)
        self._paint_ladder(
            painter, QRectF(tape.right() + 16, tape.top(), LADDER_WIDTH, tape.height())
        )
        painter.end()

    def _paint_tape(self, painter: QPainter, tape: QRectF) -> None:
        theme = self._theme
        middle = tape.center().x()
        units_per_px = self.shown_units_per_px()
        track_y = tape.top() + 30
        painter.save()
        painter.setClipRect(tape)
        start = max(tape.left(), middle + (self.low - self.value) / units_per_px)
        end = min(tape.right(), middle + (self.high - self.value) / units_per_px)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(theme.color("outline"))
        painter.drawRoundedRect(QRectF(start, track_y - 5, max(0.0, end - start), 10), 5, 5)
        painter.setBrush(theme.color("accent"))
        painter.drawRoundedRect(QRectF(start, track_y - 5, max(0.0, middle - start), 10), 5, 5)

        label_font = QFont(self.font())
        label_font.setPixelSize(SMALL_FONT_PX)
        painter.setFont(label_font)
        muted = theme.color("muted")
        for tick in scale_ticks(
            self.value, units_per_px, tape.width() / 2, self.steps, self.low, self.high
        ):
            x = middle + tick.offset
            color = QColor(muted)
            color.setAlphaF(tick.alpha)
            painter.setPen(QPen(color, 2))
            height = 6 + 12 * tick.size
            painter.drawLine(QPointF(x, track_y - 10 - height), QPointF(x, track_y - 10))
            if self.tick_text is not None and tick.label > 0:
                color.setAlphaF(tick.label)
                painter.setPen(color)
                painter.drawText(
                    QRectF(x - 50, track_y + 10, 100, 24),
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                    self.tick_text(tick.value),
                )
        # The tape runs out into the panel at either edge.
        surface = theme.color("surface")
        clear = QColor(surface)
        clear.setAlpha(0)
        for edge, inward in ((tape.left(), FADE_PX), (tape.right(), -FADE_PX)):
            fade = QLinearGradient(edge, 0, edge + inward, 0)
            fade.setColorAt(0, surface)
            fade.setColorAt(1, clear)
            painter.fillRect(
                QRectF(min(edge, edge + inward), tape.top(), FADE_PX, tape.height()), fade
            )

        painter.setPen(QPen(theme.color("surface"), 3))
        painter.setBrush(theme.color("accent"))
        painter.drawEllipse(QPointF(middle, track_y), 14, 14)
        painter.restore()

    def _paint_ladder(self, painter: QPainter, area: QRectF) -> None:
        """A bar for each speed, the slowest on top, lit up to the speed in use: there
        is more above while the finger is still low."""
        painter.setPen(Qt.PenStyle.NoPen)
        bottom = area.top() + 30 + LADDER_BAR / 2
        for index in range(len(FINE_LEVELS)):
            role = "accent" if index <= self.level else "outline"
            painter.setBrush(self._theme.color(role))
            top = bottom - LADDER_BAR - index * (LADDER_BAR + LADDER_GAP)
            painter.drawRoundedRect(QRectF(area.left(), top, area.width(), LADDER_BAR), 3, 3)


class FineSlider(QSlider):
    """A horizontal slider a finger sets finer by pulling up (see the module).

    `value_text` gives the value for the loupe, `steps` the spacings of the ticks of its
    scale (each dividing the next), `tick_text` their labels (none without). With
    `jump`, a finger coming down beside the handle moves the handle there and drags on
    from there (the player); without, it pages as QSlider does (the volume, where a tap
    at the right end would jump to full volume)."""

    def __init__(
        self,
        value_text: Callable[[int], str],
        steps: Sequence[int],
        tick_text: Callable[[int], str] | None = None,
        jump: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._value_text = value_text
        self._steps = tuple(steps)
        self._tick_text = tick_text
        self._jump = jump
        self._theme = THEMES[DARK]
        self._loupe: _Loupe | None = None
        # The finger dragging: the value with its fraction, where it went down, where it
        # was last, its speed, and at full speed its distance to the handle's middle.
        self._dragging = False
        self._exact = 0.0
        self._pressed_at = QPointF()
        self._last_x = 0.0
        self._level = 0
        self._offset = 0.0

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        if self._loupe is not None:
            self._loupe.set_theme(theme)

    def loupe(self) -> _Loupe | None:
        """The loupe, once a finger has dragged (for the tests)."""
        return self._loupe

    def level(self) -> int:
        """The speed of the finger dragging, an index of `FINE_LEVELS`."""
        return self._level

    # --- Positions ----------------------------------------------------------------------

    def _span(self) -> int:
        return max(1, self.width() - SLIDER_HANDLE_PX)

    def _units_per_px(self) -> float:
        return (self.maximum() - self.minimum()) / self._span()

    def _x_of(self, value: float) -> float:
        """The middle of the handle at `value`."""
        return SLIDER_HANDLE_PX / 2 + (value - self.minimum()) / self._units_per_px()

    def _value_at(self, x: float) -> float:
        value = self.minimum() + (x - SLIDER_HANDLE_PX / 2) * self._units_per_px()
        return min(max(value, self.minimum()), self.maximum())

    # --- The finger ---------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self.maximum() <= self.minimum():
            super().mousePressEvent(event)
            return
        x = event.position().x()
        if self._jump:
            span = self._span()
            position = round(x - SLIDER_HANDLE_PX / 2)
            self.setValue(
                QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), position, span)
            )
        elif abs(x - self._x_of(self.value())) > GRAB_HALF_WIDTH:
            super().mousePressEvent(event)
            return
        self._dragging = True
        self._exact = float(self.value())
        self._pressed_at = event.position()
        self._last_x = x
        self._level = 0
        self._offset = x - self._x_of(self._exact)
        self.setSliderDown(True)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._dragging:
            super().mouseMoveEvent(event)
            return
        position = event.position()
        x = position.x()
        level = fine_level(-position.y(), self._level)
        if level == 0:
            if self._level != 0:
                # Back at full speed: the handle keeps its distance to the finger.
                self._offset = x - self._x_of(self._exact)
            self._exact = self._value_at(x - self._offset)
        else:
            moved = (x - self._last_x) * FINE_LEVELS[level][1] * self._units_per_px()
            self._exact = min(max(self._exact + moved, self.minimum()), self.maximum())
        self._last_x = x
        self._level = level
        self.setSliderPosition(round(self._exact))
        far = position - self._pressed_at
        if self._loupe_shown() or max(abs(far.x()), abs(far.y())) >= SWIPE_SLOP:
            self._show_loupe(position)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if not self._dragging or event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        self._end_drag()
        event.accept()

    def hideEvent(self, event: QHideEvent) -> None:
        if self._dragging:
            self._end_drag()
        super().hideEvent(event)

    def _end_drag(self) -> None:
        self._dragging = False
        self._level = 0
        if self._loupe is not None:
            self._loupe.hide()
        self.setSliderDown(False)

    # --- The loupe ----------------------------------------------------------------------

    def _loupe_shown(self) -> bool:
        return self._loupe is not None and self._loupe.isVisible()

    def _show_loupe(self, position: QPointF) -> None:
        window = self.window()
        if self._loupe is None or self._loupe.parentWidget() is not window:
            self._loupe = _Loupe(window, self._tick_text is not None)
            self._loupe.set_theme(self._theme)
        loupe = self._loupe
        loupe.text = self._value_text(self.value())
        loupe.value = float(self.value())
        loupe.low = float(self.minimum())
        loupe.high = float(self.maximum())
        loupe.units_per_px = self._units_per_px()
        loupe.steps = self._steps
        loupe.tick_text = self._tick_text
        loupe.set_level(self._level, animated=loupe.isVisible())
        loupe.place(self.mapTo(window, position.toPoint()))
        if not loupe.isVisible():
            loupe.show()
            loupe.raise_()
