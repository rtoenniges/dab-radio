"""The page "Detail view" of the signal page: about a second of the received signal,
captured whole (`dsp.signal_capture`), as a landscape of levels over time and frequency
that one scrolls through (`ui.spectrogram` computes it, `ui.waterfall_gl` draws it).

Fingers on the landscape: one up or down scrolls through time (the landscape follows
the finger), across it turns the page as everywhere in the pager; two turn the view as
they move together and bring it closer or farther as they spread or close; three shift
it, as on the 3D view of the GNSS menu; a double tap returns to the first view. On a
PC: the left button dragged up or down scrolls, the wheel scrolls, with Ctrl held it
zooms, the middle button held turns and the right one shifts, as on the globe, a double
click returns.

Along the time axis a strip shows the null symbols, guard intervals and useful parts,
in the colours of the page "OFDM", with a legend at the bottom left. Beneath the
landscape the whole capture at a glance (`CaptureOverview`), the null symbols as dark
lines across it, and on it the window shown; dragging it moves the window in time and
frequency, a tap puts it there. To the right: one row per FFT wherever it falls or one
per symbol, the length of the FFT, how much time and frequency the landscape shows, how
many dB its height spans, the null symbol before or after, and a new capture.
"""

from __future__ import annotations

import dataclasses
import math
import time

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QEventPoint,
    QFontMetricsF,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QResizeEvent,
    QTouchEvent,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from dsp.signal_capture import SignalCapture
from i18n import tr, tr_context
from ui.offscreen_gl import RenderFrame, RenderThread, project
from ui.receiver_controller import ReceiverController
from ui.spectrogram import (
    DEFAULT_LEVEL_RANGE_DB,
    FFT_SIZES,
    FLOOR_DB,
    LEVEL_RANGES_DB,
    SPANS_KHZ,
    TOP_DB,
    WINDOWS_MS,
    Part,
    Rows,
    Spectrogram,
)
from ui.touch.gestures import DOUBLE_TAP_DISTANCE, DOUBLE_TAP_MS
from ui.touch.pager import SWIPE_SLOP
from ui.touch.pages import Page, pixel_font
from ui.touch.theme import PANEL_RADIUS, SMALL_FONT_PX, TOUCH_TARGET, Theme
from ui.touch.widgets import ChoiceButtons, Stepper, set_role
from ui.waterfall_gl import (
    HALF_DEPTH,
    HALF_WIDTH,
    WaterfallCamera,
    WaterfallColours,
    WaterfallRenderer,
    WaterfallScene,
    frequency_ticks,
    frequency_y,
    level_ticks,
    level_z,
    time_ticks,
    time_x,
)

# A finger moved across the whole height of the landscape scrolls this many windows.
SCROLL_PER_HEIGHT = 1.2
# A step of the wheel scrolls this share of the window.
WHEEL_SCROLL = 0.1
WHEEL_ZOOM = 0.9
# The field is computed anew at most this often while a finger scrolls.
_COMPUTE_MS = 30
OVERVIEW_HEIGHT = 72
CONTROLS_WIDTH = 290
_STEPPER_VALUE_WIDTH = 120
# The null symbol in the strip along the time axis, as its window on the page "OFDM"
# (`ui.touch.signal_page`).
_NULL_RGB = (190, 110, 200)
# Labels stand this far beyond the edges of the floor, in units of the landscape.
_LABEL_OFFSET = 0.11


def _decimal(value: float, spec: str) -> str:
    # Translators: the decimal separator of the figures of the detail view.
    return format(value, spec).replace(".", tr_context("decimal separator", "."))


def _rgb(colour: str) -> tuple[float, float, float]:
    c = QColor(colour)
    return c.redF(), c.greenF(), c.blueF()


def colour_stops(theme: Theme) -> tuple[str, ...]:
    """From the floor of the field to its top: the surface of the panel, the soft and the
    full accent, the text; between the accents a step of the same hue."""
    middle = QColor(theme.accent_soft)
    accent = QColor(theme.accent)
    blend = QColor(
        (middle.red() + 2 * accent.red()) // 3,
        (middle.green() + 2 * accent.green()) // 3,
        (middle.blue() + 2 * accent.blue()) // 3,
    )
    return (theme.surface, theme.accent_soft, blend.name(), theme.accent, theme.text)


def colour_table(theme: Theme) -> np.ndarray:
    """The colours of `colour_stops` at 256 levels, as 0xffRRGGBB for a QImage."""
    positions = (0.0, 0.30, 0.62, 0.86, 1.0)
    colours = np.array([QColor(c).getRgb()[:3] for c in colour_stops(theme)], dtype=float)
    t = np.linspace(0.0, 1.0, 256)
    channels = [np.interp(t, positions, colours[:, i]) for i in range(3)]
    r, g, b = (np.round(c).astype(np.uint32) for c in channels)
    return (0xFF000000 | (r << 16) | (g << 8) | b).astype(np.uint32)


class WaterfallView(QWidget):
    """The landscape, scrolled, turned and shifted with fingers (see the module)."""

    moved = pyqtSignal()  # the window moved in time through a gesture

    def __init__(
        self,
        model: Spectrogram,
        renderer: RenderThread | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._renderer = renderer if renderer is not None else RenderThread(WaterfallRenderer())
        self._renderer.frame_ready.connect(self._on_frame_ready)
        self._renderer.failed.connect(lambda _message: self.update())
        self._camera = WaterfallCamera()
        self._colours: WaterfallColours | None = None
        self._label_colour = QColor()
        self._shadow_colour = QColor()
        self._corner_colour = QColor()
        self._scene: WaterfallScene | None = None
        self._range_db = DEFAULT_LEVEL_RANGE_DB
        self._frame: RenderFrame | None = None
        self._requested: tuple | None = None
        self._painting = False
        self._waiting_text = ""
        self.caption = ""
        self._font = pixel_font(SMALL_FONT_PX)
        self._compute_timer = QTimer(self)
        self._compute_timer.setSingleShot(True)
        self._compute_timer.setInterval(_COMPUTE_MS)
        self._compute_timer.timeout.connect(self._compute)
        # One finger: where it came down and whether it scrolls (None: not yet decided).
        self._touch_from: QPointF | None = None
        self._scrolling: bool | None = None
        self._last_tap: tuple[float, QPointF] | None = None
        self._fingers: tuple[tuple[int, ...], QPointF, float] | None = None
        self._press: QPointF | None = None
        self._turn_from: QPointF | None = None
        self._shift_from: QPointF | None = None
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @property
    def camera(self) -> WaterfallCamera:
        return self._camera

    @property
    def scene(self) -> WaterfallScene | None:
        return self._scene

    @property
    def available(self) -> bool:
        return not self._renderer.error

    def set_theme(self, theme: Theme) -> None:
        self._colours = WaterfallColours(
            background=_rgb(theme.surface),
            stops=tuple(_rgb(c) for c in colour_stops(theme)),
            grid=(*_rgb(theme.muted), 0.55),
            parts=(
                (*(c / 255 for c in _NULL_RGB), 0.85),
                (*_rgb(theme.muted), 0.55),
                (*_rgb(theme.accent), 0.85),
            ),
        )
        self._label_colour = QColor(theme.muted)
        self._shadow_colour = QColor(theme.background)
        self._corner_colour = QColor(theme.background)
        self.refresh(at_once=True)

    def set_waiting_text(self, text: str) -> None:
        """Shown in place of the landscape while there is none."""
        self._waiting_text = text
        self.update()

    def refresh(self, at_once: bool = False) -> None:
        """The model changed: the field is computed anew, soon (scrolling gives many
        changes in a row) or `at_once`."""
        if at_once:
            self._compute_timer.stop()
            self._compute()
        elif not self._compute_timer.isActive():
            self._compute_timer.start()

    def _compute(self) -> None:
        field = self._model.heightfield()
        if field is None or self._colours is None:
            self._scene = None
        else:
            self._scene = WaterfallScene(field, self._colours, self._range_db)
        self._request_frame()
        self.update()

    @property
    def range_db(self) -> float:
        return self._range_db

    def set_range_db(self, range_db: float) -> None:
        """How many dB the height of the landscape spans, down from its top; the field
        stays as it is."""
        self._range_db = range_db
        if self._scene is not None:
            self._scene = dataclasses.replace(self._scene, range_db=range_db)
        self._request_frame()
        self.update()

    def reset_view(self) -> None:
        self._camera = WaterfallCamera()
        self._request_frame()

    def set_camera(self, camera: WaterfallCamera) -> None:
        self._camera = camera
        self._request_frame()

    def turn(self, dx: float, dy: float) -> None:
        """Turns the view as if dragged by (dx, dy) pixels: half a turn across the width."""
        self._camera.rotate(-dx * 180.0 / max(self.width(), 1), dy * 90.0 / max(self.height(), 1))
        self._request_frame()

    def zoom(self, factor: float) -> None:
        self._camera.zoom(factor)
        self._request_frame()

    def shift(self, dx: float, dy: float) -> None:
        """Shifts the view as if dragged by (dx, dy) pixels: the landscape follows."""
        per_pixel = self._camera.units_per_pixel(max(self.height(), 1))
        self._camera.pan(-dx * per_pixel, dy * per_pixel)
        self._request_frame()

    def scroll_pixels(self, dy: float) -> None:
        """Scrolls as a finger moved down by `dy` pixels: the landscape comes towards the
        viewer, and later rows come into view at the far end of the time axis."""
        model = self._model
        model.scroll(dy * SCROLL_PER_HEIGHT * model.window_ms / max(self.height(), 1))
        self.refresh()
        self.moved.emit()

    # --- frames from the render thread --------------------------------------------------

    def _request_frame(self) -> None:
        if not self.isVisible() or self._scene is None:
            return
        ratio = self.devicePixelRatioF()
        width, height = round(self.width() * ratio), round(self.height() * ratio)
        c = self._camera
        key = (c.azimuth_deg, c.elevation_deg, c.distance, self._scene, width, height, c.look_at)
        if key != self._requested:
            self._requested = key
            self._renderer.request(self, key, c, width, height, self._scene, 1.5 * ratio)

    def _on_frame_ready(self, frame: RenderFrame) -> None:
        if frame.view is not self:
            return
        frame.image.setDevicePixelRatio(self.devicePixelRatioF())
        previous, self._frame = self._frame, frame
        if previous is not None:
            previous.release()
        if not self._painting:
            self.update()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._request_frame()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._request_frame()

    # --- painting -----------------------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        self._painting = True
        try:
            self._paint_content(painter)
        finally:
            self._painting = False
        self._round_corners(painter)

    def _paint_text(self, painter: QPainter, text: str) -> None:
        background = self._colours.background if self._colours else (0.0, 0.0, 0.0)
        painter.fillRect(self.rect(), QColor.fromRgbF(*background))
        painter.setPen(self._label_colour)
        flags = Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap
        painter.drawText(self.rect().adjusted(24, 24, -24, -24), flags, text)

    def _paint_content(self, painter: QPainter) -> None:
        if self._renderer.error:
            self._paint_text(painter, self._renderer.error)
            return
        if self._scene is None:
            self._paint_text(painter, self._waiting_text)
            return
        self._request_frame()
        frame = self._frame
        if frame is None:
            self._paint_text(painter, tr("Preparing the 3D view …"))
            return
        ratio = self.devicePixelRatioF()
        if frame.image.size() == QSize(round(self.width() * ratio), round(self.height() * ratio)):
            painter.drawImage(0, 0, frame.image)
        else:
            painter.drawImage(self.rect(), frame.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        scene = frame.key[3]
        self._draw_labels(painter, frame.camera, scene)
        self._draw_caption(painter)
        self._draw_legend(painter, scene)

    def _draw_labels(
        self, painter: QPainter, camera: WaterfallCamera, scene: WaterfallScene
    ) -> None:
        """The figures along the edges of the floor, as seen by the camera the image shown
        was drawn with: time in front, frequency on the left, levels at the back right,
        each axis rising from the corner nearest the viewer."""
        field = scene.field
        width, height = self.width(), self.height()
        matrix = camera.matrix(width / height)
        painter.setFont(self._font)
        painter.setPen(self._label_colour)
        metrics = QFontMetricsF(self._font)
        w, d, gap = HALF_WIDTH, HALF_DEPTH, _LABEL_OFFSET

        def place(points: list[tuple[float, float, float]]) -> np.ndarray:
            if not points:
                return np.empty((0, 2))
            return project(matrix, np.array(points, dtype=float), width, height)

        def write(at: np.ndarray, text: str, align: str) -> None:
            if np.isnan(at[0]):
                return
            advance = metrics.horizontalAdvance(text)
            x = {"centre": at[0] - advance / 2, "left": at[0], "right": at[0] - advance}[align]
            baseline = QPointF(x, at[1] + metrics.ascent() / 2)
            # A shadow in the colour of the background keeps it legible over the field.
            painter.setPen(self._shadow_colour)
            painter.drawText(baseline + QPointF(1, 1), text)
            painter.setPen(self._label_colour)
            painter.drawText(baseline, text)

        ticks = frequency_ticks(field)
        places = place([(-w - gap, float(y), 0.0) for y in frequency_y(field, ticks)])
        for k, (at, khz) in enumerate(zip(places, ticks, strict=True)):
            text = _decimal(khz, ".0f") if abs(khz - round(khz)) < 1e-6 else _decimal(khz, ".1f")
            text = ("+" + text if khz > 0 else text).replace("-", "−")
            if k == len(ticks) - 1:
                text += " kHz"
            write(at, text, "right")

        times, labels = time_ticks(field)
        places = place([(float(x), -d - gap, 0.0) for x in time_x(field, times)])
        for k, (at, label) in enumerate(zip(places, labels, strict=True)):
            if field.symbols is None:
                label = label.replace(".", tr_context("decimal separator", "."))
                label = label.replace("-", "−")
                if k == len(labels) - 1:
                    label += " ms"
            write(at, label, "centre")

        levels = level_ticks(scene.range_db)
        heights = [float(level_z(np.array(v), scene.range_db)) for v in levels]
        places = place([(w + 0.03, d, z) for z in heights])
        # The wall is seen foreshortened, more so the smaller the range: from the plateau
        # outwards, a level whose figure would touch one already written goes without.
        written: list[float] = []
        for index in np.argsort(np.abs(levels), kind="stable"):
            at, level = places[index], levels[index]
            if any(abs(at[1] - y) < metrics.height() for y in written):
                continue
            write(at, f"{level:+.0f} dB".replace("+0", "0").replace("-", "−"), "left")
            written.append(at[1])

    def _draw_legend(self, painter: QPainter, scene: WaterfallScene) -> None:
        """What the colours of the strip along the time axis stand for, at the bottom
        left."""
        if not scene.field.parts:
            return
        painter.setFont(self._font)
        painter.setPen(self._label_colour)
        metrics = QFontMetricsF(self._font)
        box = round(0.8 * metrics.ascent())
        x, baseline = 16.0, self.height() - 14.0 - metrics.descent()
        for part, text in (
            (Part.NULL, tr("Null symbol")),
            (Part.GUARD, tr("Guard interval")),
            (Part.USEFUL, tr("Useful part")),
        ):
            painter.fillRect(
                QRectF(x, baseline - box, box, box), QColor.fromRgbF(*scene.colours.part(part))
            )
            x += box + 6
            painter.drawText(QPointF(x, baseline), text)
            x += metrics.horizontalAdvance(text) + 18

    def _draw_caption(self, painter: QPainter) -> None:
        if not self.caption:
            return
        painter.setFont(self._font)
        painter.setPen(self._label_colour)
        metrics = QFontMetricsF(self._font)
        # At the top: the corner nearest the viewer, with the lowest figures of both
        # axes, lies at the bottom.
        painter.drawText(QPointF(16, 14 + metrics.ascent()), self.caption)

    def _round_corners(self, painter: QPainter) -> None:
        rect = QRectF(self.rect())
        corners = QPainterPath()
        corners.addRect(rect)
        corners.addRoundedRect(rect, PANEL_RADIUS, PANEL_RADIUS)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillPath(corners, self._corner_colour)

    # --- fingers ------------------------------------------------------------------------

    def event(self, event: QEvent) -> bool:
        if event.type() in (
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
        return self.mapFromGlobal(point.globalPosition())

    def _on_touch(self, touch: QTouchEvent) -> None:
        kind = touch.type()
        down = [p for p in touch.points() if p.state() != QEventPoint.State.Released]
        if kind == QEvent.Type.TouchBegin:
            self._fingers = None
            self._scrolling = None
            self._touch_from = self._local(touch.points()[0]) if len(down) == 1 else None
            if len(down) > 1:
                self._move_fingers(down)
            return
        if kind == QEvent.Type.TouchUpdate:
            if len(down) >= 2:
                self._touch_from = None
                self._scrolling = False
                self._move_fingers(down)
                return
            self._fingers = None
            if self._touch_from is None or not down:
                return
            at = self._local(down[0])
            moved = at - self._touch_from
            if self._scrolling is None:
                if abs(moved.x()) < SWIPE_SLOP and abs(moved.y()) < SWIPE_SLOP:
                    return
                # Across is the pager's: it turns the page.
                self._scrolling = abs(moved.y()) > abs(moved.x())
            if self._scrolling:
                self._touch_from = at
                self.scroll_pixels(moved.y())
            return
        start, scrolling = self._touch_from, self._scrolling
        self._touch_from = None
        self._scrolling = None
        self._fingers = None
        if kind == QEvent.Type.TouchEnd and start is not None and scrolling is None:
            self._tapped(self._local(touch.points()[0]))

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
            return
        moved = centre - previous[1]
        if len(ids) > 2:
            self.shift(moved.x(), moved.y())
            return
        self.turn(moved.x(), moved.y())
        if previous[2] >= 1.0 and spread >= 1.0:
            self.zoom(previous[2] / spread)

    def _tapped(self, at: QPointF) -> None:
        """A double tap returns to the first view."""
        now = time.monotonic()
        last = self._last_tap
        if last is not None:
            near = math.hypot(at.x() - last[1].x(), at.y() - last[1].y())
            if (now - last[0]) * 1000 <= DOUBLE_TAP_MS and near <= DOUBLE_TAP_DISTANCE:
                self._last_tap = None
                self.reset_view()
                return
        self._last_tap = (now, at)

    # --- the mouse ----------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._press = event.position()
            self._scrolling = None
        elif event.button() == Qt.MouseButton.RightButton:
            self._shift_from = event.position()
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._turn_from = event.position()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._shift_from is not None:
            moved = event.position() - self._shift_from
            self._shift_from = event.position()
            self.shift(moved.x(), moved.y())
            return
        if self._turn_from is not None:
            moved = event.position() - self._turn_from
            self._turn_from = event.position()
            self.turn(moved.x(), moved.y())
            return
        if self._press is None:
            return
        moved = event.position() - self._press
        if self._scrolling is None:
            if abs(moved.x()) < SWIPE_SLOP and abs(moved.y()) < SWIPE_SLOP:
                return
            self._scrolling = abs(moved.y()) > abs(moved.x())
        if self._scrolling:
            self._press = event.position()
            self.scroll_pixels(moved.y())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._press = None
            self._scrolling = None
        elif event.button() == Qt.MouseButton.RightButton:
            self._shift_from = None
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._turn_from = None

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_view()

    def wheelEvent(self, event: QWheelEvent) -> None:
        steps = event.angleDelta().y() / 120
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom(WHEEL_ZOOM**steps)
        else:
            self._model.scroll(steps * WHEEL_SCROLL * self._model.window_ms)
            self.refresh()
            self.moved.emit()
        event.accept()


class CaptureOverview(QWidget):
    """The whole capture at a glance with the window shown on it: time across, frequency
    up; dragged, the window follows, tapped, it goes there (see the module)."""

    moved = pyqtSignal()

    def __init__(self, model: Spectrogram, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._image: QImage | None = None
        self._table: np.ndarray | None = None
        self._frame_colour = QColor()
        self._corner_colour = QColor()
        self._background = QColor()
        self._drag_from: QPointF | None = None
        self.setFixedHeight(OVERVIEW_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_theme(self, theme: Theme) -> None:
        self._table = colour_table(theme)
        self._frame_colour = QColor(theme.accent)
        self._corner_colour = QColor(theme.background)
        self._background = QColor(theme.surface)
        self.refresh_image()

    def refresh_image(self) -> None:
        """The capture changed (or the colours, or the width)."""
        levels = self._model.overview(max(self.width(), 1))
        if levels is None or self._table is None:
            self._image = None
        else:
            # Always the largest range: the strip is for finding one's way.
            t = np.clip((levels[::-1] - FLOOR_DB) / (TOP_DB - FLOOR_DB), 0.0, 1.0)
            pixels = np.ascontiguousarray(self._table[np.round(t * 255).astype(np.int64)])
            rows, columns = pixels.shape
            image = QImage(pixels.data, columns, rows, 4 * columns, QImage.Format.Format_RGB32)
            self._image = image.copy()
        self.update()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.refresh_image()

    def claims_drag(self, position: QPointF) -> bool:
        """For the pager: a finger on the overview moves the window, not the page."""
        return self._model.capture is not None

    def window_rect(self) -> QRectF:
        """The window shown, in pixels of this widget."""
        model = self._model
        duration = max(model.duration_ms, 1e-9)
        x = model.start_ms / duration * self.width()
        width = max(model.window_ms / duration * self.width(), 3.0)
        top = (1024.0 - (model.centre_khz + model.span_khz / 2)) / 2048 * self.height()
        return QRectF(x, top, width, model.span_khz / 2048 * self.height())

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        if self._image is None:
            painter.fillRect(self.rect(), self._background)
        else:
            painter.drawImage(QRectF(self.rect()), self._image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(self._frame_colour, 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.window_rect().adjusted(1, 1, -1, -1))
        rect = QRectF(self.rect())
        corners = QPainterPath()
        corners.addRect(rect)
        corners.addRoundedRect(rect, PANEL_RADIUS / 2, PANEL_RADIUS / 2)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillPath(corners, self._corner_colour)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._model.capture is None:
            return
        at = event.position()
        if not self.window_rect().adjusted(-12, -12, 12, 12).contains(at):
            self._centre_on(at)
        self._drag_from = at

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_from is None:
            return
        moved = event.position() - self._drag_from
        self._drag_from = event.position()
        model = self._model
        model.scroll(moved.x() / max(self.width(), 1) * model.duration_ms)
        model.set_centre_khz(model.centre_khz - moved.y() / max(self.height(), 1) * 2048)
        self.update()
        self.moved.emit()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_from = None

    def _centre_on(self, at: QPointF) -> None:
        model = self._model
        model.scroll_to(at.x() / max(self.width(), 1) * model.duration_ms - model.window_ms / 2)
        model.set_centre_khz(1024.0 - at.y() / max(self.height(), 1) * 2048)
        self.update()
        self.moved.emit()


class DetailPage(Page):
    """The page of the signal page (see the module). The first capture is taken when the
    page is first turned to; a new one on request."""

    def __init__(
        self, controller: ReceiverController, renderer: RenderThread | None = None
    ) -> None:
        super().__init__(tr("Detail view"))
        self._controller = controller
        self.model = Spectrogram()
        self.view = WaterfallView(self.model, renderer)
        self.overview = CaptureOverview(self.model)
        self._capturing = False
        self._captured: tuple[str, str] | None = None  # channel and time of the capture

        self.rows = ChoiceButtons(
            [(Rows.SLIDING, tr("Sliding")), (Rows.SYMBOLS, tr("Per symbol"))], role="chip"
        )
        self.rows.set_current(self.model.rows)
        self.rows.chosen.connect(self._choose_rows)
        self.fft = Stepper(
            FFT_SIZES,
            lambda size: tr("FFT {size}").format(size=size),
            value_width=_STEPPER_VALUE_WIDTH,
        )
        self.fft.set_value(self.model.fft_size)
        self.fft.value_changed.connect(self._choose_fft)
        self.window = Stepper(
            WINDOWS_MS,
            lambda ms: _decimal(ms, "g") + " ms",
            value_width=_STEPPER_VALUE_WIDTH,
        )
        self.window.set_value(self.model.window_ms)
        self.window.value_changed.connect(self._choose_window)
        self.span = Stepper(SPANS_KHZ, lambda khz: f"{khz} kHz", value_width=_STEPPER_VALUE_WIDTH)
        self.span.set_value(self.model.span_khz)
        self.span.value_changed.connect(self._choose_span)
        self.level_range = Stepper(
            LEVEL_RANGES_DB,
            lambda range_db: _decimal(range_db, "g") + " dB",
            value_width=_STEPPER_VALUE_WIDTH,
        )
        self.level_range.set_value(self.view.range_db)
        self.level_range.value_changed.connect(self.view.set_range_db)
        self.previous_null = QPushButton(tr("‹ Null"))
        self.next_null = QPushButton(tr("Null ›"))
        for button, direction in ((self.previous_null, -1), (self.next_null, 1)):
            button.setFixedHeight(TOUCH_TARGET)
            button.clicked.connect(lambda _checked=False, d=direction: self._jump(d))
        self.capture_button = set_role(QPushButton(tr("Capture again")), "accent")
        self.capture_button.setFixedHeight(TOUCH_TARGET)
        self.capture_button.clicked.connect(self.capture)

        nulls = QHBoxLayout()
        nulls.setSpacing(8)
        nulls.addWidget(self.previous_null)
        nulls.addWidget(self.next_null)
        controls = QVBoxLayout()
        controls.setSpacing(10)
        controls.addWidget(self.rows)
        controls.addWidget(self.fft)
        controls.addWidget(self.window)
        controls.addWidget(self.span)
        controls.addWidget(self.level_range)
        controls.addLayout(nulls)
        controls.addStretch()
        controls.addWidget(self.capture_button)
        side = QWidget()
        side.setFixedWidth(CONTROLS_WIDTH)
        side.setLayout(controls)
        controls.setContentsMargins(0, 0, 0, 0)

        left = QVBoxLayout()
        left.setSpacing(10)
        left.addWidget(self.view, stretch=1)
        left.addWidget(self.overview)
        row = QHBoxLayout()
        row.setSpacing(16)
        row.addLayout(left, stretch=1)
        row.addWidget(side)
        self.body.addLayout(row, stretch=1)

        self.view.moved.connect(self._moved)
        self.overview.moved.connect(self._moved_by_overview)
        controller.dsp_worker.signal_captured.connect(self._on_capture)
        self._show_state()

    # --- capturing ------------------------------------------------------------------------

    def capture(self) -> None:
        """Asks for the next frames of the signal; the landscape stays until they come."""
        self._capturing = True
        self._controller.capture_signal()
        self._show_state()

    def _on_capture(self, capture: SignalCapture) -> None:
        if not self._capturing:
            return  # asked for by someone else
        self._capturing = False
        channel = self._controller.channel or _decimal(self._controller.frequency_mhz, ".3f")
        self._captured = (channel, time.strftime("%H:%M:%S"))
        self.model.set_capture(capture)
        self.overview.refresh_image()
        self.view.refresh(at_once=True)
        self._show_state()

    def render(self) -> None:
        """Turned to for the first time: the first capture."""
        if self.model.capture is None and not self._capturing:
            self.capture()

    def _show_state(self) -> None:
        capture = self.model.capture
        frames = len(capture.frame_starts) if capture is not None else 0
        if self._capturing:
            status = tr("Capturing …")
        elif capture is not None and self._captured is not None:
            channel, at = self._captured
            status = tr("{frames} frames of {channel}, captured at {time}").format(
                frames=frames, channel=channel, time=at
            )
        else:
            status = ""
        self.status.setText(status)
        self.view.set_waiting_text(tr("Waiting for the frames of the signal …"))
        self.capture_button.setEnabled(not self._capturing)
        # Per symbol, the FFT is always as long as the useful part of a symbol.
        sliding = self.model.rows is Rows.SLIDING
        self.fft.set_value(self.model.fft_size if sliding else FFT_SIZES[-1])
        self.fft.setEnabled(sliding)
        self._show_caption()

    def _show_caption(self) -> None:
        model = self.model
        position = model.position(model.start_ms + model.window_ms / 2)
        parts = []
        if position is not None:
            if position.symbol == 0:
                parts.append(tr("Frame {frame}, null symbol").format(frame=position.frame + 1))
            else:
                parts.append(
                    tr("Frame {frame}, symbol {symbol}").format(
                        frame=position.frame + 1, symbol=position.symbol
                    )
                )
        if model.capture is not None:
            if model.rows is Rows.SYMBOLS:
                parts.append(tr("one row per symbol, one column per carrier"))
            else:
                parts.append(
                    tr(
                        "a row every {step} µs from {length} µs each, {width} kHz per column"
                    ).format(
                        step=_decimal(model.row_step_us, ".0f"),
                        length=_decimal(model.row_length_us, ".0f"),
                        width=_decimal(2048 / model.fft_size, "g"),
                    )
                )
        self.view.caption = " · ".join(parts)
        self.view.update()

    # --- the controls -----------------------------------------------------------------------

    def _choose_rows(self, rows: Rows) -> None:
        self.model.rows = rows
        self._changed()

    def _choose_fft(self, size: int) -> None:
        self.model.fft_size = size
        self._changed()

    def _choose_window(self, window_ms: float) -> None:
        self.model.set_window_ms(window_ms)
        self._changed()

    def _choose_span(self, span_khz: int) -> None:
        self.model.set_span_khz(span_khz)
        self._changed()

    def _jump(self, direction: int) -> None:
        self.model.jump_to_null(direction)
        self._changed()

    def _changed(self) -> None:
        self.view.refresh(at_once=True)
        self.overview.update()
        self._show_state()

    def _moved(self) -> None:
        self.overview.update()
        self._show_caption()

    def _moved_by_overview(self) -> None:
        self.view.refresh()
        self._show_caption()

    def apply_theme(self, theme: Theme) -> None:
        self.view.set_theme(theme)
        self.overview.set_theme(theme)
