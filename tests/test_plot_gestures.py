"""Two fingers on a pyqtgraph plot (`ui.touch.gestures.PlotGestures`): while they are
down, a picture of the view is scaled and moved with them and the view itself stands
still; when they are lifted, the view takes the range the picture showed."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
import pytest
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QEventPoint, QPointingDevice, QTouchEvent, QTransform
from PyQt6.QtWidgets import QApplication

from ui.touch.gestures import PlotGestures

APP = QApplication.instance() or QApplication([])
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


def _touch(widget, kind: QEvent.Type, points) -> None:
    event_points = [
        QEventPoint(ident, state, position, QPointF(widget.mapToGlobal(position.toPoint())))
        for ident, state, position in points
    ]
    QApplication.sendEvent(
        widget, QTouchEvent(kind, _DEVICE, Qt.KeyboardModifier.NoModifier, event_points)
    )


@pytest.fixture
def plot():
    plot = pg.PlotWidget()
    plot.resize(400, 300)
    plot.hideAxis("left")
    plot.hideAxis("bottom")
    plot.show()
    APP.processEvents()
    plot.getViewBox().setRange(xRange=(0.0, 40.0), yRange=(0.0, 30.0), padding=0)
    APP.processEvents()
    yield plot
    plot.close()


def test_the_picture_follows_the_fingers_and_the_view_takes_over_when_they_lift(plot) -> None:
    gestures = PlotGestures(plot)
    painted: list[QTransform] = []
    gestures.overlay = lambda painter, to_pixels: painted.append(QTransform(to_pixels))
    viewport = plot.viewport()
    view_box = plot.getViewBox()
    before = view_box.viewRange()
    preview = gestures._preview

    # Two fingers 100 px apart around (150, 150) of the viewport...
    _touch(viewport, QEvent.Type.TouchBegin, [(0, PRESSED, QPointF(100, 150))])
    _touch(
        viewport,
        QEvent.Type.TouchUpdate,
        [(0, MOVED, QPointF(100, 150)), (1, PRESSED, QPointF(200, 150))],
    )
    assert gestures.pinching and preview.isVisible()
    assert preview.geometry() == viewport.rect()
    # ...spread to 200 px around (250, 150): the picture doubles about the point that
    # was under their centre, which moves 100 px to the right.
    _touch(
        viewport,
        QEvent.Type.TouchUpdate,
        [(0, MOVED, QPointF(150, 150)), (1, MOVED, QPointF(350, 150))],
    )
    assert view_box.viewRange() == before  # the view stands still meanwhile
    assert preview.transform.map(QPointF(150, 150)) == QPointF(250, 150)
    assert preview.transform.map(QPointF(200, 150)) == QPointF(350, 150)
    preview.repaint()
    assert painted
    # The overlay gets the plot's coordinates mapped as the picture is: what was under
    # the fingers' centre is now under their new centre.
    under_centre = gestures.to_view(QPointF(150, 150))
    assert painted[-1].map(under_centre).x() == pytest.approx(250, abs=1)
    assert painted[-1].map(under_centre).y() == pytest.approx(150, abs=1)

    _touch(
        viewport,
        QEvent.Type.TouchEnd,
        [(0, RELEASED, QPointF(150, 150)), (1, RELEASED, QPointF(350, 150))],
    )
    APP.processEvents()
    assert not gestures.pinching and not preview.isVisible()
    (x0, x1), (y0, y1) = view_box.viewRange()
    (b0, b1), (c0, c1) = before
    assert (x1 - x0) == pytest.approx((b1 - b0) / 2, rel=0.02)
    assert (y1 - y0) == pytest.approx((c1 - c0) / 2, rel=0.02)
    # The point that was under the fingers' first centre lies under their last.
    now = gestures.to_view(QPointF(250, 150))
    assert now.x() == pytest.approx(under_centre.x(), abs=0.2)
    assert now.y() == pytest.approx(under_centre.y(), abs=0.2)


def test_a_finger_lifted_ends_the_pinch_and_the_picture_moves_at_most_per_frame(plot) -> None:
    gestures = PlotGestures(plot)
    viewport = plot.viewport()
    preview = gestures._preview
    paints = []
    preview.paintEvent = lambda event: paints.append(1)  # type: ignore[method-assign]

    _touch(viewport, QEvent.Type.TouchBegin, [(0, PRESSED, QPointF(100, 150))])
    _touch(
        viewport,
        QEvent.Type.TouchUpdate,
        [(0, MOVED, QPointF(100, 150)), (1, PRESSED, QPointF(200, 150))],
    )
    for step in range(1, 11):  # ten steps at once: painted once now, once per frame after
        _touch(
            viewport,
            QEvent.Type.TouchUpdate,
            [(0, MOVED, QPointF(100 - step, 150)), (1, MOVED, QPointF(200 + step, 150))],
        )
    APP.processEvents()
    assert len(paints) == 1
    assert gestures._throttle.isActive() and gestures._pending

    _touch(viewport, QEvent.Type.TouchUpdate, [(0, MOVED, QPointF(90, 150))])  # one lifted
    assert not gestures.pinching and not preview.isVisible()
    (x0, x1), _ = plot.getViewBox().viewRange()
    assert (x1 - x0) == pytest.approx(40.0 / 1.2, rel=0.02)
