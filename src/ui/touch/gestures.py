"""Gestures of more than one finger on a pyqtgraph plot. The Touch Display 2 reports up to
five fingers; Qt hands them to a widget as touch events once it accepts them.

Two fingers zoom about the point between them and move the view with it, both at once
as on a phone's map. One finger is left to the pager (`ui.touch.pager`), which turns the
page with it; only a tap and a double tap of one finger are reported here. For work on
a PC, a click counts as a tap, a double click as a double tap, and the wheel zooms.

While two fingers are down, the plot itself stands still: a picture of the view as it
was when they came down (`_PinchPreview`) is scaled and moved as they move, one copy of
a picture per frame (`PINCH_INTERVAL_MS`), and the view's range is set once when they
are lifted. Every change of the range runs pyqtgraph's Python code for all the items of
the plot and paints them all anew, some 20 of them on the traffic map; followed step by
step, that kept the Raspberry Pi's GUI thread, and with it the GIL, busy for 60 % of
real time, and the signal processing fell behind. Lines and marks scale with the picture
meanwhile; the names of the towns are drawn on it crisp and at their size by the plot's
`overlay`, with the items it names in `hidden_in_snapshot` left out of the picture.

Zoom limits are the view box's own (`ViewBox.setLimits`), applied when the fingers are
lifted.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QEventPoint,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPixmap,
    QTouchEvent,
    QTransform,
    QWheelEvent,
)
from PyQt6.QtWidgets import QGraphicsItem, QWidget

from ui.touch.pager import SWIPE_SLOP

# A second tap within this time and distance makes a double tap.
DOUBLE_TAP_MS = 350
DOUBLE_TAP_DISTANCE = 40
# A wheel step of 15 degrees zooms by this factor.
WHEEL_ZOOM = 0.8
# Two fingers move the picture of the view no more often than this, see the module.
PINCH_INTERVAL_MS = 33

# Paints over the picture of the view while two fingers move it: the painter in the
# viewport's pixels, and the transform from the plot's coordinates to those pixels.
Overlay = Callable[[QPainter, QTransform], None]


class _PinchPreview(QWidget):
    """Over the plot's viewport while two fingers are down: the picture of the view as
    it was, scaled and moved as they are (`transform`, pixels to pixels)."""

    def __init__(self, viewport: QWidget) -> None:
        super().__init__(viewport)
        # The fingers go on to the viewport beneath.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.snapshot = QPixmap()
        self.background = QBrush()
        self.transform = QTransform()
        self.to_pixels = QTransform()  # the plot's coordinates to pixels, at the start
        self.overlay: Overlay | None = None
        self.hide()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.background)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setTransform(self.transform)
        painter.drawPixmap(0, 0, self.snapshot)
        painter.resetTransform()
        if self.overlay is not None:
            self.overlay(painter, self.to_pixels * self.transform)


class PlotGestures(QObject):
    """Attached to `plot`'s viewport. `tapped` and `double_tapped` carry the point in
    the plot's coordinates. `overlay` paints over the picture of the view while two
    fingers move it, `hidden_in_snapshot` names the items left out of that picture."""

    tapped = pyqtSignal(QPointF)
    double_tapped = pyqtSignal(QPointF)

    def __init__(self, plot: pg.PlotWidget) -> None:
        super().__init__(plot)
        self._plot = plot
        viewport = plot.viewport()
        viewport.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents)
        viewport.installEventFilter(self)
        self._start: QPointF | None = None  # where the first finger came down (viewport)
        self._moved = False
        self._several = False  # a second finger came: no tap
        self.overlay: Overlay | None = None
        self.hidden_in_snapshot: list[QGraphicsItem] = []
        self._preview = _PinchPreview(viewport)
        # Two fingers down: their centre and distance when the picture was taken.
        self._pinch: tuple[QPointF, float] | None = None
        self._pending = False  # the picture has moved since it was last painted
        self._throttle = QTimer(self)
        self._throttle.setSingleShot(True)
        self._throttle.setInterval(PINCH_INTERVAL_MS)
        self._throttle.timeout.connect(self._paint_pending)
        self._last_tap: tuple[int, QPointF] | None = None  # timestamp and place
        # The mouse: where the left button went down; a double click's second release
        # is no tap.
        self._click: QPointF | None = None
        self._double_click = False

    # --- The view --------------------------------------------------------------------

    def _local(self, point: QEventPoint) -> QPointF:
        """Where a finger is on the viewport, from its position on the screen (the one
        Qt always fills in)."""
        return self._plot.viewport().mapFromGlobal(point.globalPosition())

    def to_view(self, point: QPointF) -> QPointF:
        """A point of the viewport in the plot's coordinates."""
        scene = self._plot.mapToScene(point.toPoint())
        return self._plot.getViewBox().mapSceneToView(scene)

    def zoom(self, factor: float, centre: QPointF) -> None:
        """Scales the view by `factor` (below 1 zooms in) about a point of the plot."""
        self._plot.getViewBox().scaleBy((factor, factor), center=centre)

    @property
    def pinching(self) -> bool:
        """Two fingers are moving the picture of the view."""
        return self._pinch is not None

    def pinch(self, first: QPointF, second: QPointF) -> None:
        """One step of two fingers at these viewport positions. The first takes the
        picture of the view; the others scale and move it, so that the point that was
        under the fingers' centre follows it and the view grows as they spread."""
        centre = (first + second) / 2
        distance = math.hypot(first.x() - second.x(), first.y() - second.y())
        if self._pinch is None:
            if distance >= 1:
                self._begin_pinch(centre, distance)
            return
        centre0, distance0 = self._pinch
        scale = max(distance, 1.0) / distance0
        transform = QTransform()
        transform.translate(centre.x(), centre.y())
        transform.scale(scale, scale)
        transform.translate(-centre0.x(), -centre0.y())
        self._preview.transform = transform
        if self._throttle.isActive():
            self._pending = True
        else:
            self._preview.update()
            self._throttle.start()

    def _paint_pending(self) -> None:
        if self._pending:
            self._pending = False
            self._preview.update()
            self._throttle.start()

    def _begin_pinch(self, centre: QPointF, distance: float) -> None:
        plot, viewport, preview = self._plot, self._plot.viewport(), self._preview
        view_box = plot.getViewBox()
        view_box.updateMatrix()
        preview.to_pixels = view_box.childGroup.sceneTransform() * plot.viewportTransform()
        for item in self.hidden_in_snapshot:
            item.setVisible(False)
        preview.snapshot = viewport.grab()
        for item in self.hidden_in_snapshot:
            item.setVisible(True)
        preview.background = plot.backgroundBrush()
        preview.transform = QTransform()
        preview.overlay = self.overlay
        preview.setGeometry(viewport.rect())
        preview.show()
        preview.raise_()
        self._pinch = (centre, distance)

    def _end_pinch(self) -> None:
        """The fingers are lifted: the view takes the range the picture showed last."""
        self._throttle.stop()
        self._pending = False
        if self._pinch is None:
            return
        self._pinch = None
        preview = self._preview
        inverse, invertible = (preview.to_pixels * preview.transform).inverted()
        if invertible:
            shown = inverse.mapRect(QRectF(self._plot.viewport().rect()))
            self._plot.getViewBox().setRange(shown, padding=0)
        preview.hide()
        preview.snapshot = QPixmap()

    # --- Events ----------------------------------------------------------------------

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        kind = event.type()
        if kind == QEvent.Type.TouchBegin:
            touch: QTouchEvent = event  # type: ignore[assignment]
            self._start = self._local(touch.points()[0])
            self._moved = False
            self._several = len(touch.points()) > 1
            self._pinch = None
            event.accept()
            return True
        if kind == QEvent.Type.TouchUpdate:
            self._on_update(event)  # type: ignore[arg-type]
            return True
        if kind == QEvent.Type.Resize and self._preview.isVisible():
            self._preview.setGeometry(self._plot.viewport().rect())
            return False
        if kind in (QEvent.Type.TouchEnd, QEvent.Type.TouchCancel):
            touch = event  # type: ignore[assignment]
            if kind == QEvent.Type.TouchEnd and not (self._moved or self._several):
                self._on_tap(self._local(touch.points()[0]), touch.timestamp())
            self._start = None
            self._end_pinch()
            return True
        if kind == QEvent.Type.Wheel:
            wheel: QWheelEvent = event  # type: ignore[assignment]
            steps = wheel.angleDelta().y() / 120
            if steps:
                self.zoom(WHEEL_ZOOM**steps, self.to_view(wheel.position()))
            return True
        if kind == QEvent.Type.MouseButtonPress:
            mouse: QMouseEvent = event  # type: ignore[assignment]
            if mouse.button() == Qt.MouseButton.LeftButton:
                self._click = mouse.position()
                self._double_click = False
            return False
        if kind == QEvent.Type.MouseButtonRelease:
            self._on_release(event)  # type: ignore[arg-type]
            return False
        if kind == QEvent.Type.MouseButtonDblClick:
            mouse = event  # type: ignore[assignment]
            self._double_click = True
            self.double_tapped.emit(self.to_view(mouse.position()))
            return True
        return False

    def _on_release(self, mouse: QMouseEvent) -> None:
        """A click without dragging is a tap. A drag across is the pager's, which then
        keeps the release from reaching here."""
        start, self._click = self._click, None
        if mouse.button() != Qt.MouseButton.LeftButton or start is None or self._double_click:
            return
        offset = mouse.position() - start
        if abs(offset.x()) <= SWIPE_SLOP and abs(offset.y()) <= SWIPE_SLOP:
            self.tapped.emit(self.to_view(mouse.position()))

    def _on_update(self, touch: QTouchEvent) -> None:
        down = [p for p in touch.points() if p.state() != QEventPoint.State.Released]
        if len(down) >= 2:
            self._several = True
            self.pinch(self._local(down[0]), self._local(down[1]))
            return
        self._end_pinch()
        if down and self._start is not None:
            offset = self._local(down[0]) - self._start
            if abs(offset.x()) > SWIPE_SLOP or abs(offset.y()) > SWIPE_SLOP:
                self._moved = True

    def _on_tap(self, position: QPointF, timestamp: int) -> None:
        last = self._last_tap
        if last is not None:
            elapsed = timestamp - last[0]
            near = math.hypot(position.x() - last[1].x(), position.y() - last[1].y())
            if 0 <= elapsed <= DOUBLE_TAP_MS and near <= DOUBLE_TAP_DISTANCE:
                self._last_tap = None
                self.double_tapped.emit(self.to_view(position))
                return
        self._last_tap = (timestamp, position)
        self.tapped.emit(self.to_view(position))
