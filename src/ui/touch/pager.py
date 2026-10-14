"""Pages side by side that a finger swipes through, as on the home screen of a phone,
with dots beneath them that tell how many there are and which one is shown. The dots
appear while swiping and when the pager is shown, and fade out again soon."""

from __future__ import annotations

import math
import time
from collections import deque
from enum import Enum, auto

from PyQt6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QEvent,
    QObject,
    QPoint,
    QPointF,
    QPropertyAnimation,
    QRectF,
    Qt,
    QTimer,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QEventPoint,
    QHideEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QResizeEvent,
    QShowEvent,
    QTouchEvent,
)
from PyQt6.QtWidgets import QAbstractScrollArea, QApplication, QScrollBar, QWidget

from ui.touch.theme import Theme

# The strip beneath the pages that holds the dots.
DOTS_HEIGHT = 28
DOT_SIZE = 10
DOT_GAP = 14
# The dots stay this long after the last swipe, then fade out.
DOTS_VISIBLE_MS = 1500
DOTS_FADE_MS = 300
# A finger has to move this far before the pager tells a swipe from a scroll or a tap.
SWIPE_SLOP = 16
# Let go, the page turns if it was dragged more than half its width, or flicked faster
# than this (pixels per second) however short; the velocity is taken over the last
# `VELOCITY_WINDOW_MS`.
FLING_SPEED = 600
VELOCITY_WINDOW_MS = 100
SETTLE_MS = 250
# Beyond the first and the last page the strip follows the finger only this much.
EDGE_RESISTANCE = 0.3
# A list let go faster than this runs on and slows down within `RUN_OUT_MS`, as far as
# it would at that speed in a third of the time.
RUN_OUT_SPEED = 300
RUN_OUT_MS = 800

_MOUSE_EVENTS = (
    QEvent.Type.MouseButtonPress,
    QEvent.Type.MouseMove,
    QEvent.Type.MouseButtonRelease,
)
_TOUCH_EVENTS = (
    QEvent.Type.TouchBegin,
    QEvent.Type.TouchUpdate,
    QEvent.Type.TouchEnd,
    QEvent.Type.TouchCancel,
)


class _Drag(Enum):
    UNDECIDED = auto()  # not yet moved far enough
    SWIPE = auto()  # across: turns the page
    SCROLL = auto()  # up or down over a list: scrolls it
    PASS = auto()  # up or down over something else: left to it


class _Dots(QWidget):
    """One dot per page, the current one in the accent colour."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._count = 0
        self._current = 0
        self._opacity = 0.0
        self._on = QColor(Qt.GlobalColor.white)
        self._off = QColor(Qt.GlobalColor.gray)

    def set_theme(self, theme: Theme) -> None:
        self._on = theme.color("accent")
        self._off = theme.color("outline")
        self.update()

    def set_pages(self, count: int, current: int) -> None:
        self._count = count
        self._current = current
        self.update()

    def get_opacity(self) -> float:
        return self._opacity

    def set_opacity(self, opacity: float) -> None:
        self._opacity = opacity
        self.update()

    opacity = pyqtProperty(float, fget=get_opacity, fset=set_opacity)

    def paintEvent(self, event: QPaintEvent) -> None:
        if self._count < 2 or self._opacity <= 0.0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(self._opacity)
        painter.setPen(Qt.PenStyle.NoPen)
        width = self._count * DOT_SIZE + (self._count - 1) * DOT_GAP
        left = (self.width() - width) / 2
        top = (self.height() - DOT_SIZE) / 2
        for index in range(self._count):
            painter.setBrush(self._on if index == self._current else self._off)
            x = left + index * (DOT_SIZE + DOT_GAP)
            painter.drawEllipse(QRectF(x, top, DOT_SIZE, DOT_SIZE))
        painter.end()


class Pager(QWidget):
    """Swiped horizontally anywhere over its pages, also over their buttons, lists and
    plots. The pages lie side by side on a strip that moves with the finger and settles
    on the nearest page when let go.

    The pager sees the finger before the widget beneath it does (an event filter on the
    application) and decides once it has moved `SWIPE_SLOP`: across, it turns the page;
    up or down over a list, it scrolls the list itself and lets it run out. Either way
    the widget beneath is told the press is over without a click. A tap stays with the
    widget. Lists on the pages therefore need no QScroller (it would take the finger
    first); they should scroll per pixel.

    Qt hands a finger to most widgets as mouse events; a widget that takes touch events
    itself (a map with `ui.touch.gestures.PlotGestures`) gets touch events, which the
    pager follows the same way. When a second finger comes, the gesture is the widget's:
    a swipe begun snaps back. A widget that drags with one finger itself from where it
    came down (a point on the curve of the equalizer) says so by `claims_drag`; the
    pager then leaves that finger alone.

    A finger that comes down while the strip settles stops it where it is. A swipe
    moves it on from there; anything else (a tap, a scroll, a second finger) lets it go
    on to its page.
    """

    current_changed = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._strip = QWidget(self)
        self._pages: list[QWidget] = []
        self._current = 0
        self._dots = _Dots(self)
        self._fade = QPropertyAnimation(self._dots, b"opacity", self)
        self._fade.setDuration(DOTS_FADE_MS)
        self._fade.setEndValue(0.0)
        self._dots_timer = QTimer(self)
        self._dots_timer.setSingleShot(True)
        self._dots_timer.setInterval(DOTS_VISIBLE_MS)
        self._dots_timer.timeout.connect(self._fade.start)
        self._settle = QPropertyAnimation(self._strip, b"pos", self)
        self._settle.setDuration(SETTLE_MS)
        self._settle.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._run_out = QPropertyAnimation(self)
        self._run_out.setPropertyName(b"value")
        self._run_out.setDuration(RUN_OUT_MS)
        self._run_out.setEasingCurve(QEasingCurve.Type.OutCubic)

        # The finger: where it came down (global), on which widget, what the drag is,
        # and the list it scrolls with the value of its scroll bar at the start.
        self._press: QPointF | None = None
        self._target: QWidget | None = None
        self._drag = _Drag.UNDECIDED
        self._scroll_bar: QScrollBar | None = None
        self._scroll_start = 0
        # Where the strip stood when the finger came down, as the offset a finger would
        # have dragged it to (`_unresisted`).
        self._strip_start = 0.0
        self._samples: deque[tuple[float, QPointF]] = deque()  # (seconds, global position)
        # A mouse event reaches the filter again for every parent it propagates to.
        self._last_event: tuple | None = None
        self._cancelling = False
        self._filtering = False
        self._touch = False  # the finger comes as touch events, not as a mouse

    # --- Pages --------------------------------------------------------------------------

    def add_page(self, page: QWidget) -> None:
        page.setParent(self._strip)
        self._pages.append(page)
        self._layout_pages()
        page.show()

    def count(self) -> int:
        return len(self._pages)

    def current_index(self) -> int:
        return self._current

    def current_page(self) -> QWidget | None:
        return self._pages[self._current] if self._pages else None

    def set_current_index(self, index: int, animated: bool = False) -> None:
        index = min(max(index, 0), len(self._pages) - 1)
        changed = index != self._current
        self._current = index
        self._dots.set_pages(len(self._pages), index)
        target = QPoint(-index * self.width(), 0)
        self._settle.stop()
        if animated:
            self._settle.setStartValue(self._strip.pos())
            self._settle.setEndValue(target)
            self._settle.start()
        else:
            self._strip.move(target)
        if changed:
            self.current_changed.emit(index)

    def set_theme(self, theme: Theme) -> None:
        self._dots.set_theme(theme)

    def flash_dots(self) -> None:
        """Shows the dots and lets them fade out after a while."""
        self._fade.stop()
        self._dots.set_opacity(1.0)
        self._dots_timer.start()

    @property
    def dots_opacity(self) -> float:
        return self._dots.get_opacity()

    def _layout_pages(self) -> None:
        width, height = self.width(), max(self.height() - DOTS_HEIGHT, 0)
        for index, page in enumerate(self._pages):
            page.setGeometry(index * width, 0, width, height)
        self._strip.resize(max(len(self._pages), 1) * width, height)
        self._settle.stop()
        self._strip.move(-self._current * width, 0)
        self._dots.setGeometry(0, height, width, DOTS_HEIGHT)
        self._dots.set_pages(len(self._pages), self._current)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._layout_pages()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not self._filtering:
            QApplication.instance().installEventFilter(self)
            self._filtering = True
        self.flash_dots()

    def hideEvent(self, event: QHideEvent) -> None:
        super().hideEvent(event)
        if self._filtering:
            QApplication.instance().removeEventFilter(self)
            self._filtering = False
        self._dots_timer.stop()
        self._fade.stop()
        self._dots.set_opacity(0.0)
        self._press = None
        # A finger on the strip loses it here; nobody sees it settle.
        self._settle.stop()
        self._strip.move(-self._current * self.width(), 0)

    # --- The finger -------------------------------------------------------------------

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        kind = event.type()
        if kind not in _MOUSE_EVENTS and kind not in _TOUCH_EVENTS or self._cancelling:
            return False
        if not isinstance(watched, QWidget) or not (watched is self or self.isAncestorOf(watched)):
            return False
        if kind in _TOUCH_EVENTS:
            return self._on_touch(watched, event)  # type: ignore[arg-type]
        mouse: QMouseEvent = event  # type: ignore[assignment]
        position = mouse.globalPosition()
        key = (event.type(), mouse.timestamp(), position.x(), position.y())
        if key == self._last_event:
            # The same event, propagated to a parent: decided already.
            return self._drag in (_Drag.SWIPE, _Drag.SCROLL)
        self._last_event = key
        if event.type() == QEvent.Type.MouseButtonPress:
            return self._on_press(watched, mouse)
        if event.type() == QEvent.Type.MouseMove:
            return self._on_move(mouse)
        return self._on_release(mouse)

    def _on_press(self, watched: QWidget, mouse: QMouseEvent) -> bool:
        if mouse.button() != Qt.MouseButton.LeftButton:
            return False
        self._begin(watched, mouse.globalPosition(), touch=False)
        return False

    def _on_move(self, mouse: QMouseEvent) -> bool:
        if self._press is None or not mouse.buttons() & Qt.MouseButton.LeftButton:
            return False
        return self._drag_to(mouse.globalPosition())

    def _on_release(self, mouse: QMouseEvent) -> bool:
        if self._press is None or mouse.button() != Qt.MouseButton.LeftButton:
            return False
        return self._finish(mouse.globalPosition())

    def _on_touch(self, watched: QWidget, touch: QTouchEvent) -> bool:
        kind = touch.type()
        points = touch.points()
        if kind == QEvent.Type.TouchBegin:
            if len(points) == 1:
                self._begin(watched, points[0].globalPosition(), touch=True)
            return False
        if self._press is None or not self._touch:
            return False
        down = [p for p in points if p.state() != QEventPoint.State.Released]
        if kind == QEvent.Type.TouchUpdate and len(down) < 2:
            return self._drag_to(down[0].globalPosition()) if down else False
        if kind == QEvent.Type.TouchEnd and len(down) < 2:
            return self._finish(points[0].globalPosition())
        # A second finger, or the touch is cancelled: the gesture is not a swipe.
        self._press = None
        self._go_on()
        self._drag = _Drag.UNDECIDED
        return False

    def _begin(self, watched: QWidget, position: QPointF, touch: bool) -> None:
        self._run_out.stop()
        self._press = position
        self._target = watched
        self._touch = touch
        self._drag = _Drag.PASS if self._claimed(watched, position) else _Drag.UNDECIDED
        if self._drag is _Drag.UNDECIDED:
            self._settle.stop()  # caught on its way to a page
        self._strip_start = self._unresisted(self._strip.x())
        self._samples.clear()
        self._samples.append((time.monotonic(), position))

    def _claimed(self, watched: QWidget, position: QPointF) -> bool:
        """Whether a widget under the finger drags with it itself from there (a point
        on the curve of the equalizer): one with a method `claims_drag(position)`,
        position on the screen, that says so. Then this finger turns no page."""
        widget: QWidget | None = watched
        while widget is not None and widget is not self:
            claims = getattr(widget, "claims_drag", None)
            if claims is not None and claims(position):
                return True
            widget = widget.parentWidget()
        return False

    def _drag_to(self, position: QPointF) -> bool:
        dx = position.x() - self._press.x()
        dy = position.y() - self._press.y()
        if self._drag is _Drag.UNDECIDED:
            if abs(dx) < SWIPE_SLOP and abs(dy) < SWIPE_SLOP:
                return False
            self._decide(dx, dy)
        if self._drag is _Drag.PASS:
            return False
        self._record(position)
        if self._drag is _Drag.SWIPE:
            self._follow(dx)
        else:
            self._scroll_bar.setValue(round(self._scroll_start - dy))
        return True

    def _finish(self, position: QPointF) -> bool:
        drag = self._drag
        dx = position.x() - self._press.x()
        self._press = None
        self._target = None
        self._drag = _Drag.UNDECIDED
        if drag is _Drag.SWIPE:
            self._record(position)
            vx, _vy = self._velocity()
            self.set_current_index(self._landing(dx, vx), animated=True)
            self.flash_dots()
        elif drag is _Drag.SCROLL:
            self._record(position)
            _vx, vy = self._velocity()
            self._let_run_out(vy)
        else:
            self._go_on()  # a tap, or a drag left to the widget
            return False
        # The key lets a release propagated to a parent pass the filter as handled too.
        self._drag = drag
        QTimer.singleShot(0, self._forget_drag)
        return True

    def _forget_drag(self) -> None:
        if self._press is None:
            self._drag = _Drag.UNDECIDED

    def _decide(self, dx: float, dy: float) -> None:
        if abs(dx) > abs(dy):
            self._drag = _Drag.SWIPE if len(self._pages) > 1 else _Drag.PASS
        else:
            self._scroll_bar = self._scroll_bar_under(self._target)
            if self._scroll_bar is None:
                self._drag = _Drag.PASS
            else:
                self._drag = _Drag.SCROLL
                self._scroll_start = self._scroll_bar.value()
        if self._drag is not _Drag.SWIPE:
            self._go_on()
        # A widget taking touch events sees no press to cancel; it ignores one finger.
        if self._drag is not _Drag.PASS and not self._touch:
            self._cancel_target()

    def _follow(self, dx: float) -> None:
        """Moves the strip with the finger, against resistance beyond the ends."""
        offset = self._strip_start + dx
        lowest = -(len(self._pages) - 1) * self.width()
        if offset > 0:
            offset *= EDGE_RESISTANCE
        elif offset < lowest:
            offset = lowest + (offset - lowest) * EDGE_RESISTANCE
        self._strip.move(round(offset), 0)
        self._fade.stop()
        self._dots_timer.stop()
        self._dots.set_opacity(1.0)

    def _unresisted(self, offset: float) -> float:
        """The offset a finger drags the strip to for `_follow` to show it at `offset`:
        beyond the ends the resistance taken back, so that a finger catching the strip
        there moves it on without a jump."""
        lowest = -(len(self._pages) - 1) * self.width()
        if offset > 0:
            return offset / EDGE_RESISTANCE
        if offset < lowest:
            return lowest + (offset - lowest) / EDGE_RESISTANCE
        return offset

    def _landing(self, dx: float, velocity: float) -> int:
        """The page a swipe settles on, from where the finger lets go of the strip: after
        a flick the next one in its direction, else the nearer one."""
        pages = -(self._strip_start + dx) / max(self.width(), 1)
        if abs(velocity) >= FLING_SPEED:
            return math.floor(pages) if velocity > 0 else math.ceil(pages)
        return round(pages)

    def _go_on(self) -> None:
        """Lets the strip go on to the current page when a finger stopped it on its way
        there and does not swipe."""
        stopped = self._settle.state() != QAbstractAnimation.State.Running
        if stopped and self._strip.pos() != QPoint(-self._current * self.width(), 0):
            self.set_current_index(self._current, animated=True)

    def _let_run_out(self, velocity: float) -> None:
        bar = self._scroll_bar
        if bar is None or abs(velocity) < RUN_OUT_SPEED:
            return
        # OutCubic starts at three times its mean speed: over a third of the time at
        # the speed of the finger.
        distance = velocity * RUN_OUT_MS / 1000 / 3
        end = min(max(round(bar.value() - distance), bar.minimum()), bar.maximum())
        self._run_out.setTargetObject(bar)
        self._run_out.setStartValue(bar.value())
        self._run_out.setEndValue(end)
        self._run_out.start()

    def _record(self, position: QPointF) -> None:
        now = time.monotonic()
        self._samples.append((now, position))
        while len(self._samples) > 2 and now - self._samples[0][0] > VELOCITY_WINDOW_MS / 1000:
            self._samples.popleft()

    def _velocity(self) -> tuple[float, float]:
        """Pixels per second across and down, over the last moments of the drag."""
        (t0, p0), (t1, p1) = self._samples[0], self._samples[-1]
        if t1 <= t0:
            return 0.0, 0.0
        return (p1.x() - p0.x()) / (t1 - t0), (p1.y() - p0.y()) / (t1 - t0)

    def _scroll_bar_under(self, widget: QWidget | None) -> QScrollBar | None:
        """The vertical scroll bar of the list the finger is on, if it can scroll."""
        while widget is not None and widget is not self:
            if isinstance(widget, QAbstractScrollArea):
                bar = widget.verticalScrollBar()
                if bar.maximum() > bar.minimum():
                    return bar
            widget = widget.parentWidget()
        return None

    def _cancel_target(self) -> None:
        """The finger swipes or scrolls: the widget it came down on must not take the
        gesture for its own. It gets a release far outside itself, which does not click
        a button or choose a row."""
        target = self._target
        if target is None:
            return
        release = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            QPointF(-1e6, -1e6),
            QPointF(-1e6, -1e6),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self._cancelling = True
        try:
            QApplication.sendEvent(target, release)
        finally:
            self._cancelling = False
