"""The Earth in space with the satellites GEPOS sends ephemerides of, for the page
"3D view" of the "GNSS" menu (`ui.touch.gnss_view`) and, cut to the shape of the
slide, for the radio page (`ui.touch.display_choice`). A radio has no GNSS receiver:
there are no range spheres, no lines of sight and no fix, only the place the listener
entered ("Location"). Satellites and orbits take the colours and symbols of their
systems, as on the page "Sky"; those below the listener's horizon are paler.

OpenGL draws the Earth and the orbits into an image (`ui.globe_gl`), in a thread of its
own: the view asks for a new image whenever the camera or the scene changes and shows the
last one finished, with the marks and their labels painted on it here, from the camera
that image was drawn with.

Fingers: one is left to the pager (`ui.touch.pager`), which turns the page with it, and
taps. Two turn the view as they move together and bring it closer or farther as they
spread or close, both at once; three shift the centre of the view. A tap on a satellite
lets the view glide until the satellite is in the middle and turns it around it from
then on, following it; a tap on "Location" turns it around the centre of the Earth
again, and a double tap returns to the first view. On a PC: the wheel zooms, the right
button held shifts, the middle button held turns, a click is a tap and a double click a
double tap.
"""

from __future__ import annotations

import math
import time

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QEventPoint,
    QFontMetricsF,
    QHideEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPixmap,
    QPolygonF,
    QRadialGradient,
    QTouchEvent,
    QWheelEvent,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget

from i18n import tr, tr_noop
from ui.globe_gl import EARTH_RADIUS_KM, Camera, GlobeRenderThread, hidden_by_earth
from ui.gnss_model import SatelliteOrbit
from ui.gnss_tab import system_style
from ui.offscreen_gl import RenderFrame, project
from ui.touch.gestures import DOUBLE_TAP_DISTANCE, DOUBLE_TAP_MS
from ui.touch.pager import SWIPE_SLOP
from ui.touch.pages import pixel_font
from ui.touch.theme import SMALL_FONT_PX

# The view before the user moves it: not straight down at the listener but from the
# south-west, so that the satellites above them stand beside the Earth, not in front of
# it; from this far, the orbits of GPS fit the height of the page.
VIEW_DISTANCE_KM = 64_000.0
_VIEW_OFFSET_DEG = (-24.0, -40.0)  # latitude, longitude

# How long the view glides until the satellite tapped is in its middle.
CENTRE_SECONDS = 0.6
_FRAME_MS = 33
# Each step of the wheel brings the view this much closer.
WHEEL_ZOOM = 0.87

# The marks of the satellites: the symbol's size in pixels, how much smaller the farthest
# and larger the nearest are drawn, and how pale they and their orbits are below the
# listener's horizon.
_MARK_SIZE = 13.0
_MARK_SCALE = (0.75, 1.35)
# The sizes between those come in steps of this, so that each mark is a small image
# painted once and blitted, not scaled smoothly at every frame (a few milliseconds per
# frame on the Raspberry Pi with 40 satellites); a step is well under a pixel.
_MARK_SIZE_STEP = 0.05
_BELOW_OPACITY = 0.45
_ORBIT_OPACITY = 0.55
_BELOW_ORBIT_OPACITY = 0.18
_POSITION_RADIUS = 5.0
_POSITION_GLOW = 16.0
# A finger is broader than a mouse: a tap counts on a mark this many pixels beside it.
_TAP_REACH = 14.0
# Between a mark and its label, and between the caption and the edges of the view.
_LABEL_GAP = 4.0
_CAPTION_MARGIN = 16

_BACKGROUND = QColor(8, 12, 22)
_TEXT = QColor(225, 228, 235)
_DIM_TEXT = QColor(150, 155, 168)
_CAPTION_BACKING = QColor(8, 12, 22, 200)
_POSITION_COLOUR = QColor(253, 216, 53)
# The label of the listener's position, which also stands for it among the marks.
# Translators: the listener's position on the globe of the 3D view.
POSITION_LABEL = tr_noop("Location")


def _ease(t: float) -> float:
    """Fast at first, slowing into the end."""
    return 1 - (1 - min(max(t, 0.0), 1.0)) ** 3


def _symbol_path(symbol: str, size: float) -> QPainterPath:
    """The symbol of a system as on the page "Sky" (● ■ ▲ ◆), `size` wide, around the
    origin."""
    half = size / 2
    path = QPainterPath()
    if symbol == "o":
        path.addEllipse(QPointF(), half, half)
        return path
    corners = {
        "s": [(-0.85, -0.85), (0.85, -0.85), (0.85, 0.85), (-0.85, 0.85)],
        "t1": [(0.0, -1.1), (1.0, 0.75), (-1.0, 0.75)],
        "d": [(0.0, -1.15), (0.9, 0.0), (0.0, 1.15), (-0.9, 0.0)],
    }[symbol]
    path.addPolygon(QPolygonF([QPointF(x * half, y * half) for x, y in corners]))
    path.closeSubpath()
    return path


class GlobeView(QWidget):
    """The scene, turned, zoomed and shifted with fingers (see the module)."""

    prepared = pyqtSignal()  # OpenGL is set up, or has turned out to be missing

    def __init__(
        self,
        renderer: GlobeRenderThread | None = None,
        parent: QWidget | None = None,
        distance_km: float = VIEW_DISTANCE_KM,
    ):
        """`renderer` may be shared with other views; `distance_km` is that of the
        first view (`reset_view`)."""
        super().__init__(parent)
        self._renderer = renderer if renderer is not None else GlobeRenderThread()
        self._renderer.frame_ready.connect(self._on_frame_ready)
        self._renderer.failed.connect(self._on_failed)
        self._distance_km = distance_km
        self._satellites: list[SatelliteOrbit] = []
        self._position_km: np.ndarray | None = None  # the listener's
        # The satellite (by name) the view turns around and follows; None: the centre of
        # the Earth.
        self._followed: str | None = None
        # How far the view has glided to the point it turns around: from that point and
        # the shift from it (right and up, km), to the new one, without a shift.
        self._glide = 1.0
        self._glide_from: tuple[np.ndarray, float, float] = (np.zeros(3), 0.0, 0.0)
        self._glide_to = np.zeros(3)
        self._animations: dict[str, tuple[float, float, float, float]] = {}
        self._home = (0.0, 0.0)
        self._camera = Camera(0.0, 0.0, distance_km)
        # A line in the lower left corner; the corners rounded off in the colour around.
        self._caption = ""
        self._corner_radius = 0.0
        self._corner_colour = QColor()
        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_MS)
        self._timer.timeout.connect(self._on_frame)
        self._lines: np.ndarray | None = None  # for OpenGL; None: to be built
        # The frame shown, the last one the thread finished for this view, and what was
        # asked for last: painted again as it is while the pager moves the page, asked
        # for anew only when the camera, the scene or the size changes.
        self._frame: RenderFrame | None = None
        self._requested: tuple | None = None
        self._painting = False
        self._prepared_emitted = False  # `prepared` goes out once
        # Place and radius of every mark drawn, and what it stands for (a satellite or
        # the listener's position), for taps.
        self._marks: list[tuple[QPointF, float, SatelliteOrbit | str]] = []
        # The marks and labels as small images, painted once each.
        self._sprites: dict[tuple, QPixmap] = {}
        self._font = pixel_font(SMALL_FONT_PX)
        # One finger down: where, and on which mark; None once it moved or another came.
        self._tap: tuple[QPointF, SatelliteOrbit | str | None] | None = None
        self._last_tap: tuple[float, QPointF] | None = None  # monotonic seconds and place
        # Two fingers or more: which ones, their centre and how far apart they are.
        self._fingers: tuple[tuple[int, ...], QPointF, float] | None = None
        # The mouse: where the left button went down and on which mark, and where the
        # right (shifting) or middle (turning) button was last.
        self._press: tuple[QPointF, SatelliteOrbit | str | None] | None = None
        self._pan_from: QPointF | None = None
        self._turn_from: QPointF | None = None
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents)
        # Every pixel is painted here: Qt need not paint the panel beneath first.
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # --- what is shown --------------------------------------------------------------

    @property
    def camera(self) -> Camera:
        return self._camera

    @property
    def available(self) -> bool:
        """False once OpenGL has turned out to be missing."""
        return not self._renderer.error

    @property
    def error(self) -> str:
        return self._renderer.error

    @property
    def followed(self) -> str | None:
        """The satellite the view turns around, if any."""
        return self._followed

    @property
    def caption(self) -> str:
        return self._caption

    def set_caption(self, text: str) -> None:
        """A line written in the lower left corner over the scene; empty for none."""
        if text != self._caption:
            self._caption = text
            self.update()

    def set_corners(self, radius: float, colour: QColor) -> None:
        """Rounds the corners off, in the colour of what lies around the view."""
        self._corner_radius = radius
        self._corner_colour = QColor(colour)
        self.update()

    def set_scene(self, satellites: list[SatelliteOrbit], position_km: np.ndarray | None) -> None:
        """The satellites in space and the listener's position, Earth-fixed km."""
        self._satellites = list(satellites)
        self._position_km = position_km
        self._lines = None
        if self._followed is not None:
            self._follow()
        self._refresh()

    def _follow(self) -> None:
        """The view moves along with the satellite it turns around; it stays where it is
        when the satellite is gone."""
        position = next(
            (s.position / 1000 for s in self._satellites if s.ephemeris.name == self._followed),
            None,
        )
        if position is None:
            self._followed = None
        elif "glide" in self._animations:
            self._glide_to = position
        else:
            self._camera.pivot_km = tuple(position.tolist())

    def look_at(self, latitude_deg: float, longitude_deg: float) -> None:
        """Turns the view to a place on the ground and remembers it for `reset_view`."""
        self._home = (latitude_deg, longitude_deg)
        self.reset_view()

    def reset_view(self) -> None:
        self._animations.pop("glide", None)
        self._followed = None
        d_latitude, d_longitude = _VIEW_OFFSET_DEG
        self._camera = Camera(
            self._home[0] + d_latitude, self._home[1] + d_longitude, self._distance_km
        )
        self._refresh()

    # --- animation ------------------------------------------------------------------

    def centre_on(self, satellite: SatelliteOrbit | None) -> None:
        """Lets the view glide sideways until a satellite is in its middle, and turns it
        around that satellite from then on, following it; None does the same for the
        centre of the Earth. The direction and the distance stay."""
        camera = self._camera
        self._followed = satellite.ephemeris.name if satellite is not None else None
        self._glide_from = (np.array(camera.pivot_km), camera.right_km, camera.up_km)
        self._glide_to = satellite.position / 1000 if satellite is not None else np.zeros(3)
        self._animate("glide", 0.0, 1.0, CENTRE_SECONDS)

    @property
    def glide(self) -> float:
        return self._glide

    @glide.setter
    def glide(self, t: float) -> None:
        self._glide = t
        pivot, right, up = self._glide_from
        self._camera.pivot_km = tuple((pivot * (1 - t) + self._glide_to * t).tolist())
        self._camera.pan_to(right * (1 - t), up * (1 - t))

    def _end_glide(self) -> None:
        if self._animations.pop("glide", None) is not None:
            self.glide = 1.0

    def _animate(self, name: str, start: float, end: float, seconds: float) -> None:
        setattr(self, name, start)
        self._animations[name] = (time.monotonic(), seconds, start, end)
        if self.isVisible():
            self._timer.start()
            self._refresh()
        else:
            self.finish_animations()

    def _on_frame(self) -> None:
        self.advance(time.monotonic())

    def advance(self, now: float) -> None:
        """Moves the running animations on to the time `now` (monotonic seconds)."""
        for name, (began, seconds, start, end) in list(self._animations.items()):
            t = (now - began) / seconds
            setattr(self, name, start + (end - start) * _ease(t))
            if t >= 1.0:
                del self._animations[name]
        if not self._animations:
            self._timer.stop()
        self._refresh()

    def finish_animations(self) -> None:
        for name, (_, _, _, end) in self._animations.items():
            setattr(self, name, end)
        self._animations.clear()
        self._timer.stop()
        self._refresh()

    def hideEvent(self, event: QHideEvent) -> None:
        # Nothing moves while nobody watches.
        self.finish_animations()
        super().hideEvent(event)

    # --- moving the view --------------------------------------------------------------

    def turn(self, dx: float, dy: float) -> None:
        """Turns the view as if it were dragged by (dx, dy) pixels: around the Earth, the
        ground below the viewer follows; around a satellite, half a turn across the
        height of the view."""
        if self._camera.around_earth:
            km_per_pixel = self._camera.km_per_pixel(self.height())
            per_pixel = math.degrees(km_per_pixel / EARTH_RADIUS_KM)
            narrowing = max(math.cos(math.radians(self._camera.latitude_deg)), 0.2)
        else:
            per_pixel, narrowing = 180.0 / self.height(), 1.0
        self._camera.rotate(-dx * per_pixel / narrowing, dy * per_pixel)
        self._refresh()

    def shift(self, dx: float, dy: float) -> None:
        """Shifts the view by (dx, dy) pixels: the ground follows, the centre of the view
        goes the other way."""
        km_per_pixel = self._camera.km_per_pixel(self.height())
        self._camera.pan(-dx * km_per_pixel, dy * km_per_pixel)
        self._refresh()

    def zoom(self, factor: float) -> None:
        """Brings the view closer (below 1) or farther (above 1)."""
        self._camera.zoom(factor)
        self._refresh()

    def keep_distance(self) -> None:
        """Makes the distance of the view now that of the first view (`reset_view`)."""
        self._distance_km = self._camera.distance_km

    # --- fingers ----------------------------------------------------------------------

    def event(self, event: QEvent) -> bool:
        kind = event.type()
        if kind in (
            QEvent.Type.TouchBegin,
            QEvent.Type.TouchUpdate,
            QEvent.Type.TouchEnd,
            QEvent.Type.TouchCancel,
        ):
            self._on_touch(event)  # type: ignore[arg-type]
            event.accept()
            return True
        return super().event(event)

    def _local(self, point: QEventPoint) -> QPointF:
        """Where a finger is on the view, from its position on the screen (the one Qt
        always fills in)."""
        return self.mapFromGlobal(point.globalPosition())

    def _on_touch(self, touch: QTouchEvent) -> None:
        kind = touch.type()
        down = [p for p in touch.points() if p.state() != QEventPoint.State.Released]
        if kind == QEvent.Type.TouchBegin:
            # A glide still under way ends at once: the fingers take the view over.
            self._end_glide()
            self._fingers = None
            self._tap = None
            if len(touch.points()) == 1:
                at = self._local(touch.points()[0])
                self._tap = (at, self._mark_at(at))
            else:
                self._move_fingers(down)
            return
        if kind == QEvent.Type.TouchUpdate:
            if len(down) >= 2:
                self._tap = None
                self._move_fingers(down)
                return
            self._fingers = None
            if self._tap is not None and down:
                offset = self._local(down[0]) - self._tap[0]
                if abs(offset.x()) > SWIPE_SLOP or abs(offset.y()) > SWIPE_SLOP:
                    self._tap = None
            return
        tap, self._tap = self._tap, None
        self._fingers = None
        if kind == QEvent.Type.TouchEnd and tap is not None:
            self._on_tap(tap[1], self._local(touch.points()[0]))

    def _move_fingers(self, down: list[QEventPoint]) -> None:
        """One step of two fingers or more: two turn and zoom, more shift."""
        ids = tuple(sorted(p.id() for p in down))
        places = [self._local(p) for p in down]
        centre = sum(places, QPointF()) / len(places)
        spread = sum(math.hypot(p.x() - centre.x(), p.y() - centre.y()) for p in places)
        spread /= len(places)
        previous = self._fingers
        self._fingers = (ids, centre, spread)
        if previous is None or previous[0] != ids:
            return  # a finger came or went: the movement begins anew from here
        moved = centre - previous[1]
        if len(ids) == 2:
            self.turn(moved.x(), moved.y())
            if previous[2] >= 1.0 and spread >= 1.0:
                self.zoom(previous[2] / spread)
        else:
            self.shift(moved.x(), moved.y())

    def _on_tap(self, what: SatelliteOrbit | str | None, at: QPointF) -> None:
        now = time.monotonic()
        last = self._last_tap
        if last is not None:
            elapsed_ms = (now - last[0]) * 1000
            near = math.hypot(at.x() - last[1].x(), at.y() - last[1].y())
            if elapsed_ms <= DOUBLE_TAP_MS and near <= DOUBLE_TAP_DISTANCE:
                self._last_tap = None
                self.reset_view()
                return
        self._last_tap = (now, at)
        self._tapped(what)

    def _tapped(self, what: SatelliteOrbit | str | None) -> None:
        if isinstance(what, SatelliteOrbit):
            self.centre_on(what)
        elif what == POSITION_LABEL:
            self.centre_on(None)

    # --- the mouse --------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._end_glide()
        button = event.button()
        if button == Qt.MouseButton.LeftButton:
            self._press = (event.position(), self._mark_at(event.position()))
        elif button == Qt.MouseButton.RightButton:
            self._pan_from = event.position()
        elif button == Qt.MouseButton.MiddleButton:
            self._turn_from = event.position()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._pan_from is not None:
            moved = event.position() - self._pan_from
            self._pan_from = event.position()
            self.shift(moved.x(), moved.y())
        elif self._turn_from is not None:
            moved = event.position() - self._turn_from
            self._turn_from = event.position()
            self.turn(moved.x(), moved.y())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        button = event.button()
        if button == Qt.MouseButton.LeftButton:
            # A drag across is the pager's, which then sends a release far away.
            press, self._press = self._press, None
            if press is not None:
                offset = event.position() - press[0]
                if abs(offset.x()) <= SWIPE_SLOP and abs(offset.y()) <= SWIPE_SLOP:
                    self._tapped(press[1])
        elif button == Qt.MouseButton.RightButton:
            self._pan_from = None
        elif button == Qt.MouseButton.MiddleButton:
            self._turn_from = None

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_view()

    def wheelEvent(self, event: QWheelEvent) -> None:
        # A step of the wheel is 120 units; touchpads send less at a time.
        self.zoom(WHEEL_ZOOM ** (event.angleDelta().y() / 120))
        event.accept()

    def _mark_at(self, at: QPointF) -> SatelliteOrbit | str | None:
        """What the mark at a place of the view stands for; None beside all marks."""
        for place, radius, what in reversed(self._marks):  # the nearest was drawn last
            if math.hypot(at.x() - place.x(), at.y() - place.y()) <= radius + _TAP_REACH:
                return what
        return None

    # --- what OpenGL draws ----------------------------------------------------------

    def _line_vertices(self) -> np.ndarray:
        """The orbits for `GlobeRenderer.render`, in the colours of their systems."""
        if self._lines is not None:
            return self._lines
        parts = []
        for s in self._satellites:
            colour = [c / 255 for c in system_style(s.ephemeris.system).colour]
            opacity = _ORBIT_OPACITY if s.elevation >= 0 else _BELOW_ORBIT_OPACITY
            ring = np.repeat(s.ring / 1000, 2, axis=0)[1:-1]  # every point ends a piece
            parts.append(np.column_stack([ring, np.tile([*colour, opacity], (len(ring), 1))]))
        self._lines = (
            np.concatenate(parts).astype(np.float32)
            if parts
            else np.empty((0, 7), dtype=np.float32)
        )
        return self._lines

    # --- drawing --------------------------------------------------------------------

    def _refresh(self) -> None:
        """The camera or the scene changed: asks for a new frame, and the view is
        painted when it comes. Until then there is nothing new to paint, and painting
        on every touch event cost the Raspberry Pi more than the frames themselves."""
        if self.isVisible():
            self._request_frame()

    def _request_frame(self) -> None:
        """Asks the thread for the scene as it is now, unless that has been asked for."""
        ratio = self.devicePixelRatioF()
        width, height = round(self.width() * ratio), round(self.height() * ratio)
        c = self._camera
        lines = self._line_vertices()
        key = (
            c.latitude_deg,
            c.longitude_deg,
            c.distance_km,
            c.right_km,
            c.up_km,
            c.pivot_km,
            id(lines),
            width,
            height,
        )
        if key != self._requested:
            self._requested = key
            self._renderer.request(self, key, c, width, height, lines, 1.5 * ratio)

    def _on_frame_ready(self, frame: RenderFrame) -> None:
        if frame.view is not self:
            return  # another view's, from the shared thread
        frame.image.setDevicePixelRatio(self.devicePixelRatioF())
        previous, self._frame = self._frame, frame
        if previous is not None:
            previous.release()
        self._emit_prepared()
        if not self._painting:  # a stand-in may hand the frame over while painting
            self.update()

    def _on_failed(self, _message: str) -> None:
        self._emit_prepared()
        self.update()

    def _emit_prepared(self) -> None:
        if not self._prepared_emitted:
            self._prepared_emitted = True
            self.prepared.emit()

    def _paint_text(self, painter: QPainter, text: str) -> None:
        painter.fillRect(self.rect(), _BACKGROUND)
        painter.setPen(_TEXT)
        flags = Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap
        painter.drawText(self.rect().adjusted(24, 24, -24, -24), flags, text)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        self._marks = []
        self._painting = True
        try:
            self._paint_scene(painter)
        finally:
            self._painting = False
        self._round_corners(painter)

    def _paint_scene(self, painter: QPainter) -> None:
        if self._renderer.error:
            # Also when the shared thread failed for another view before this one asked.
            self._emit_prepared()
            self._paint_text(painter, self._renderer.error)
            return
        self._request_frame()
        frame = self._frame
        if frame is None:
            # The first time the view is shown: the thread sets OpenGL up and draws.
            self._paint_text(painter, tr("Preparing the 3D view …"))
            return
        image = frame.image
        ratio = self.devicePixelRatioF()
        if image.size() == QSize(round(self.width() * ratio), round(self.height() * ratio)):
            painter.drawImage(0, 0, image)
        else:
            # Resized since: stretched until the frame of the new size comes.
            painter.drawImage(self.rect(), image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self._draw_marks(painter, frame.camera)
        self._draw_caption(painter)

    def _draw_caption(self, painter: QPainter) -> None:
        """The caption on a dark backing, which keeps it legible over orbits and labels."""
        if not self._caption:
            return
        metrics = QFontMetricsF(self._font)
        pad_x, pad_y = 10.0, 4.0
        width = metrics.horizontalAdvance(self._caption) + 2 * pad_x
        height = metrics.height() + 2 * pad_y
        margin = _CAPTION_MARGIN - pad_x  # the text itself keeps the margin
        backing = QRectF(margin, self.height() - margin - height, width, height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_CAPTION_BACKING)
        painter.drawRoundedRect(backing, height / 2, height / 2)
        painter.setFont(self._font)
        painter.setPen(_TEXT)
        painter.drawText(backing, Qt.AlignmentFlag.AlignCenter, self._caption)

    def _round_corners(self, painter: QPainter) -> None:
        """Covers the corners outside a rounded rectangle (`set_corners`)."""
        radius = self._corner_radius
        if radius <= 0:
            return
        rect = QRectF(self.rect())
        corners = QPainterPath()  # filled odd-even: the rectangle less the rounded one
        corners.addRect(rect)
        corners.addRoundedRect(rect, radius, radius)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillPath(corners, self._corner_colour)

    def _draw_marks(self, painter: QPainter, camera: Camera) -> None:
        """The satellites, the farthest first, then the listener's position, as seen by
        the camera the image shown was drawn with."""
        matrix = camera.matrix(self.width() / self.height())
        viewer = camera.position()
        satellites = self._satellites
        if satellites:
            positions = np.array([s.position for s in satellites]) / 1000
            places = project(matrix, positions, self.width(), self.height())
            distances = np.linalg.norm(positions - viewer, axis=1)
            hidden = hidden_by_earth(viewer, positions) | np.isnan(places[:, 0])
            for k in np.argsort(-distances):
                if not hidden[k]:
                    # Nearer satellites a little larger, as a hint of depth.
                    low, high = _MARK_SCALE
                    size = min(max(self._camera.distance_km / distances[k], low), high)
                    self._draw_satellite(painter, satellites[k], QPointF(*places[k]), size)
        position = self._position_km
        if position is not None:
            place = project(matrix, position[None, :], self.width(), self.height())[0]
            if not hidden_by_earth(viewer, position[None, :])[0] and not math.isnan(place[0]):
                at = QPointF(*place)
                self._draw_position(painter, at)
                self._draw_label(painter, at, _POSITION_RADIUS, tr(POSITION_LABEL), strong=True)
                self._marks.append((at, _POSITION_RADIUS, POSITION_LABEL))

    def _sprite(self, key: tuple, half_width: float, half_height: float, paint) -> QPixmap:
        """The image kept under `key`, painted by `paint(painter)` around the origin the
        first time it is asked for."""
        ratio = self.devicePixelRatioF()
        pixmap = self._sprites.get((key, ratio))
        if pixmap is None:
            pixmap = QPixmap(round(2 * half_width * ratio), round(2 * half_height * ratio))
            pixmap.setDevicePixelRatio(ratio)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.translate(half_width, half_height)
            paint(painter)
            painter.end()
            self._sprites[(key, ratio)] = pixmap
        return pixmap

    def _draw_satellite(
        self, painter: QPainter, satellite: SatelliteOrbit, at: QPointF, size: float
    ) -> None:
        ephemeris = satellite.ephemeris
        style = system_style(ephemeris.system)
        above = satellite.elevation >= 0
        steps = round(size / _MARK_SIZE_STEP)
        size = steps * _MARK_SIZE_STEP

        def paint_mark(p: QPainter) -> None:
            colour = QColor(*style.colour)
            if not above:
                p.setOpacity(_BELOW_OPACITY)
            if ephemeris.healthy:
                p.setPen(QPen(QColor(0, 0, 0, 170), 1.2))
                p.setBrush(colour)
            else:
                # Out of order: hollow, as on the page "Sky"; the ring keeps the system.
                p.setPen(QPen(colour, 2.0))
                p.setBrush(_BACKGROUND)
            p.drawPath(_symbol_path(style.symbol, _MARK_SIZE * size))

        half = (_MARK_SIZE * 0.6 + 2) * size
        key = ("mark", ephemeris.system, ephemeris.healthy, above, steps)
        mark = self._sprite(key, half, half, paint_mark)
        # On whole pixels, as the labels: blitted, not resampled.
        painter.drawPixmap(QPointF(round(at.x() - half), round(at.y() - half)), mark)
        radius = _MARK_SIZE / 2 * size
        self._draw_label(painter, at, radius, ephemeris.name, strong=above)
        self._marks.append((at, radius, satellite))

    def _draw_label(
        self, painter: QPainter, at: QPointF, radius: float, text: str, strong: bool
    ) -> None:
        """Text to the right of a mark, to its left where it would run off the view, with
        a shadow for where the Earth is behind it."""
        font = self._font
        metrics = QFontMetricsF(font)
        half_width = metrics.horizontalAdvance(text) / 2 + 2
        half_height = metrics.height() / 2 + 1

        def paint_label(p: QPainter) -> None:
            p.setFont(font)
            flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            box = QRectF(-half_width, -half_height, 2 * half_width - 1, 2 * half_height - 1)
            p.setPen(QColor(0, 0, 0, 170))
            p.drawText(box.translated(1, 1), flags, text)
            p.setPen(_TEXT if strong else _DIM_TEXT)
            p.drawText(box, flags, text)

        label = self._sprite(("label", text, strong), half_width, half_height, paint_label)
        left = at.x() + radius + _LABEL_GAP
        if left + 2 * half_width > self.width() and at.x() <= self.width():
            left = at.x() - radius - _LABEL_GAP - 2 * half_width
        painter.drawPixmap(QPointF(round(left), round(at.y() - half_height)), label)

    def _draw_position(self, painter: QPainter, at: QPointF) -> None:
        def paint_position(p: QPainter) -> None:
            glow = QRadialGradient(QPointF(), _POSITION_GLOW)
            glow.setColorAt(0.0, QColor(255, 255, 255, 170))
            glow.setColorAt(1.0, QColor(255, 255, 255, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(glow)
            p.drawEllipse(QPointF(), _POSITION_GLOW, _POSITION_GLOW)
            p.setPen(QPen(Qt.GlobalColor.black, 1.5))
            p.setBrush(_POSITION_COLOUR)
            p.drawEllipse(QPointF(), _POSITION_RADIUS, _POSITION_RADIUS)

        half = _POSITION_GLOW + 1
        mark = self._sprite(("position",), half, half, paint_position)
        painter.drawPixmap(QPointF(round(at.x() - half), round(at.y() - half)), mark)
