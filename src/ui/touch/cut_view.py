"""The cutter (`ui.cut_model`): a sub-page of the radio page, reached from the sheet of
a recording ("Cut", `ui.touch.recordings_view`).

From the top: the whole stretch, the recording with its neighbours, each file named,
with the part kept, the part shown below as a frame and titles kept from it before as a
green bar (a tap or a drag moves the part shown); the part shown, large: the names of
the files along the top with their boundaries dashed, the dynamic labels where they
appear first, the level mirrored about its middle, accent where kept and dimmed beyond,
the pauses marked, the time of day beneath, and the two handles with their tabs at the
bottom, where the hand does not cover the level. A finger on a handle drags it, one
elsewhere taps where it plays or, moved, scrolls the time; two zoom and scroll. Beneath:
play and pause, what is shown (the title, its beginning, its end; the latter two choose
the handle the steps move), the steps and the jumps to the pause before or after; the
name of the title, with the names the station sent, and "Keep" or "Save".
"""

from __future__ import annotations

import math
import time

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QEventPoint,
    QFont,
    QFontMetricsF,
    QHideEvent,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPolygonF,
    QTouchEvent,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from i18n import tr, tr_context, tr_noop, tr_plural
from recording.cutter import Stretch
from ui.cut_model import END, START, CutModel
from ui.touch.icons import symbol_icon
from ui.touch.keyboard import Keyboard
from ui.touch.overlay import ActionSheet, Overlay
from ui.touch.theme import DARK, PANEL_RADIUS, SMALL_FONT_PX, THEMES, Theme
from ui.touch.widgets import ElidedLabel, repolish, set_role

OVERVIEW_HEIGHT = 64
# The bands of the large view: the names of the files above the level, the time of
# day and the tabs of the handles beneath it.
NAME_STRIP = 34
AXIS_BAND = 26
TAB_BAND = 54
TAB_WIDTH = 104
TAB_HEIGHT = 44
# How far beside its line a finger takes a handle.
HANDLE_REACH = 40
# A finger that moved less than this did not scroll, it tapped.
TAP_SLOP = 12
TICK_SPACINGS_S = (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600)
MIN_TICK_PX = 120
# The shortest part of a stretch shown, and how much room it may leave either side.
MIN_VIEW_S = 2.0
VIEW_MARGIN = 0.05
# What is shown: the title with a quarter of its length either side; its beginning
# 6 s before to 14 s after; its end 14 s before to 6 s after.
TITLE_MARGIN = 0.25
START_VIEW_S = (6.0, 14.0)
END_VIEW_S = (14.0, 6.0)
STEPS_S = (-1.0, -0.1, 0.1, 1.0)
# How often at most the views of the stretch are painted anew: a finger reports some
# sixty moves a second, and every picture holds the GUI thread, with the GIL the signal
# processing needs, for a few milliseconds.
PAINT_INTERVAL_MS = 33
# While it plays, the view turns over as the position leaves it on the right (within
# this share of its span: not a view scrolled away by hand), the position then this far
# into it.
FOLLOW_REACH = 0.5
FOLLOW_LEAD = 0.1
PLAY_ICON = QSize(32, 32)
# The level is drawn as the square root of the amplitude: between linear (a solid block
# for radio, which is compressed) and dB (where quiet passages fill the height).
LEVEL_EXPONENT = 0.5
MARK_MAX_WIDTH = 280
NAME_ROW_HEIGHT = 56
NAME_SHEET_WIDTH = 760
TITLE_VIEW = "title"

_TOUCH_EVENTS = (
    QEvent.Type.TouchBegin,
    QEvent.Type.TouchUpdate,
    QEvent.Type.TouchEnd,
    QEvent.Type.TouchCancel,
)
_CANNOT_CUT = tr_noop("Cannot cut")
_CANNOT_KEEP = tr_noop("Cannot keep")
_READING_LEVEL = tr_noop("Reading the level … {percent} %")


def _alpha(color: str, alpha: int) -> QColor:
    result = QColor(color)
    result.setAlpha(alpha)
    return result


def _decimal(text: str) -> str:
    return text.replace(".", tr_context("decimal separator", "."))


def clock_text(moment: float, tenths: bool = False) -> str:
    """ "21:27:41", with tenths "21:27:41.4"."""
    text = time.strftime("%H:%M:%S", time.localtime(moment))
    if tenths:
        text += _decimal(f".{int(moment * 10 + 1e-6) % 10}")
    return text


def length_text(seconds: float) -> str:
    """ "2:11.6"."""
    minutes, rest = divmod(max(0.0, seconds), 60)
    return _decimal(f"{int(minutes)}:{rest:04.1f}")


def name_text(artist: str, title: str) -> str:
    return " – ".join(part for part in (artist, title) if part)


def view_preset(model: CutModel, kind: str) -> tuple[float, float]:
    """The part shown of the stretch for the title, its beginning or its end."""
    if kind == START:
        before, after = START_VIEW_S
        return model.start - before, model.start + after
    if kind == END:
        before, after = END_VIEW_S
        return model.end - before, model.end + after
    margin = TITLE_MARGIN * (model.end - model.start)
    return model.start - margin, model.end + margin


def level_columns(levels: np.ndarray, times: np.ndarray, seconds_per_level: float) -> np.ndarray:
    """The highest of the levels under each pixel column: `times` are the
    `columns + 1` edges in seconds; NaN where none was read."""
    count = len(levels)
    if count == 0 or len(times) < 2:
        return np.full(max(0, len(times) - 1), np.nan)
    first = np.floor(times / seconds_per_level).astype(int)
    inside = (first[:-1] >= 0) & (first[:-1] < count)
    edges = np.clip(first, 0, count)
    # A pad at the end makes `count` a valid index for the last column's edge.
    padded = np.append(levels.astype(float), np.nan)
    with np.errstate(invalid="ignore"):
        columns = np.fmax.reduceat(padded, edges)[:-1]
    columns[~inside] = np.nan
    return columns


def _height(levels_db: np.ndarray, half: float) -> np.ndarray:
    amplitude = np.power(10.0, np.nan_to_num(levels_db, nan=-200.0) / 20.0)
    return np.clip(amplitude**LEVEL_EXPONENT, 0.0, 1.0) * half


def level_images(
    layers: list[np.ndarray],
    times: np.ndarray,
    seconds_per_level: float,
    height: int,
    half: float,
    pictures: list[tuple[QColor, tuple[int, ...]]],
) -> list[QImage]:
    """The level as pictures of `len(times) - 1` columns: the layers (levels in dBFS
    every `seconds_per_level`, each drawn over the one before) mirrored about the
    middle, their edges smoothed; one picture for each colour with its alpha for each
    layer.

    Made with numpy in a few milliseconds, much of it without the GIL, and only when
    the part shown or the level changes (`_StretchView`): painted with QPainter at
    every move, the level held the GUI thread 40 to 80 ms each time, and the signal
    processing waited for the GIL beyond real time. The layers of a picture share their
    colour, so the picture is that colour times their alpha, looked up from a table."""
    width = len(times) - 1
    if width <= 0 or height <= 0:
        return [QImage() for _ in pictures]
    distance = np.abs(np.arange(height, dtype=np.float32) + 0.5 - height / 2)[:, None]
    covers = []
    for levels in layers:
        heights = _height(level_columns(levels, times, seconds_per_level), half).astype(np.float32)
        covers.append(np.clip(heights[None, :] - distance + 0.5, 0.0, 1.0))
    alphas = np.arange(256, dtype=np.uint32)
    images = []
    for color, layer_alphas in pictures:
        total = np.zeros((height, width), dtype=np.float32)
        for cover, alpha in zip(covers, layer_alphas, strict=True):
            layer = cover * (alpha / 255.0)
            total = layer + total * (1.0 - layer)  # this layer over those before
        # Premultiplied ARGB for every alpha in the colour.
        table = (
            (alphas << 24)
            | ((color.red() * alphas // 255) << 16)
            | ((color.green() * alphas // 255) << 8)
            | (color.blue() * alphas // 255)
        )
        pixels = table[(total * 255.0 + 0.5).astype(np.uint8)]
        image = QImage(
            pixels.tobytes(), width, height, 4 * width, QImage.Format.Format_ARGB32_Premultiplied
        )
        images.append(image.copy())  # the array goes, the picture stays
    return images


class _StretchView(QWidget):
    """What both views of the stretch share: the model, the colours and the level,
    drawn once for the part shown and kept until that, the level or the size change;
    a handle that moves only cuts the two pictures (accent and muted) differently."""

    def __init__(self, model: CutModel) -> None:
        super().__init__()
        self._model = model
        self._theme = THEMES[DARK]
        self._level_key: tuple | None = None
        self._level_pictures: tuple[QImage, QImage] | None = None
        self._paint_timer = QTimer(self)
        self._paint_timer.setSingleShot(True)
        self._paint_timer.setInterval(PAINT_INTERVAL_MS)
        self._paint_timer.timeout.connect(self.update)

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.update()

    def request_paint(self) -> None:
        """Paints anew within `PAINT_INTERVAL_MS`, once for all asked for meanwhile."""
        if not self._paint_timer.isActive():
            self._paint_timer.start()

    def _panel(self, painter: QPainter) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self._theme.surface))
        painter.drawRoundedRect(QRectF(self.rect()), PANEL_RADIUS, PANEL_RADIUS)

    def _paint_level(
        self,
        painter: QPainter,
        rect: QRectF,
        view: tuple[float, float],
        selection: tuple[float, float],
        half: float,
        peaks: bool,
    ) -> None:
        """The level of `view` (seconds) in `rect`, about its middle: accent between
        the x of `selection`, muted beyond."""
        model = self._model
        stretch = model.stretch
        if stretch is None:
            return
        left, top = round(rect.left()), round(rect.top())
        width, height = round(rect.width()), round(rect.height())
        key = (view, width, height, half, peaks, model.levels_version, self._theme.key)
        if key != self._level_key or self._level_pictures is None:
            v0, v1 = view
            times = v0 + np.arange(width + 1) / max(width, 1) * (v1 - v0)
            theme = self._theme
            # The peaks faint beneath, the RMS over them; accent and muted.
            layers, accent_alphas, muted_alphas = [model.rms_db], (255,), (120,)
            if peaks:
                layers, accent_alphas, muted_alphas = (
                    [model.peak_db, model.rms_db],
                    (90, 255),
                    (60, 120),
                )
            accent, muted = level_images(
                layers,
                times,
                model.level_step_seconds,
                height,
                half,
                [(QColor(theme.accent), accent_alphas), (QColor(theme.muted), muted_alphas)],
            )
            self._level_pictures = (accent, muted)
            self._level_key = key
        accent, muted = self._level_pictures
        start, end = (min(max(x, left), left + width) for x in selection)
        for picture, x0, x1 in (
            (muted, left, start),
            (accent, start, end),
            (muted, end, left + width),
        ):
            span = round(x1) - round(x0)
            if span > 0:
                source = QRectF(round(x0) - left, 0, span, height)
                painter.drawImage(QRectF(round(x0), top, span, height), picture, source)


class OverviewStrip(_StretchView):
    """The whole stretch: its files with their names, the level, the part kept, the
    part shown in the large view (a frame) and the titles kept from it (a green bar). A
    tap or a drag puts the part shown there."""

    view_requested = pyqtSignal(float)  # the middle of the part to show, in seconds

    def __init__(self, model: CutModel) -> None:
        super().__init__(model)
        self.setFixedHeight(OVERVIEW_HEIGHT)
        self.view = (0.0, 1.0)

    def _rect(self) -> QRectF:
        return QRectF(16, 8, self.width() - 32, self.height() - 16)

    def x_of(self, seconds: float) -> float:
        stretch = self._model.stretch
        rect = self._rect()
        duration = stretch.duration if stretch is not None and stretch.duration else 1.0
        return rect.left() + seconds / duration * rect.width()

    def t_of(self, x: float) -> float:
        stretch = self._model.stretch
        rect = self._rect()
        duration = stretch.duration if stretch is not None else 0.0
        return (x - rect.left()) / max(rect.width(), 1.0) * duration

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._request(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._request(event.position().x())

    def _request(self, x: float) -> None:
        if self._model.stretch is not None:
            self.view_requested.emit(self.t_of(x))

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        self._panel(painter)
        model = self._model
        stretch = model.stretch
        if stretch is None:
            return
        theme = self._theme
        rect = self._rect()
        names = QRectF(rect.left(), rect.top(), rect.width(), 18)
        wave = QRectF(rect.left(), rect.top() + 20, rect.width(), rect.height() - 20)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_alpha(theme.accent, 46))
        left, right = self.x_of(model.start), self.x_of(model.end)
        painter.drawRect(QRectF(left, wave.top(), right - left, wave.height()))
        selection = (self.x_of(model.start), self.x_of(model.end))
        self._paint_level(
            painter, wave, (0.0, stretch.duration), selection, wave.height() / 2, peaks=False
        )
        font = QFont(self.font())
        font.setPixelSize(14)
        painter.setFont(font)
        metrics = QFontMetricsF(font)
        clips = stretch.clips
        for index, clip in enumerate(clips):
            left = self.x_of(clip.start)
            following = clips[index + 1].start if index + 1 < len(clips) else stretch.duration
            right = self.x_of(following)
            if clip.start > 0:
                painter.setPen(QPen(QColor(theme.outline), 2))
                painter.drawLine(QPointF(left, rect.top()), QPointF(left, rect.bottom()))
            opened = clip.recording.id == stretch.opened.id
            painter.setPen(QColor(theme.text if opened else theme.muted))
            text = metrics.elidedText(
                clip.recording.name, Qt.TextElideMode.ElideRight, max(0.0, right - left - 12)
            )
            painter.drawText(
                QRectF(left + 6, names.top(), max(0.0, right - left - 12), names.height()),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                text,
            )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.success))
        for start, end in model.kept:
            left, right = self.x_of(start), self.x_of(end)
            painter.drawRoundedRect(QRectF(left, rect.bottom() - 5, right - left, 5), 2, 2)
        v0, v1 = self.view
        left = max(self.x_of(v0), rect.left() - 4)
        right = min(self.x_of(v1), rect.right() + 4)
        painter.setPen(QPen(_alpha(theme.text, 220), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(left, wave.top() - 3, right - left, wave.height() + 6), 6, 6)
        painter.end()


class WaveView(_StretchView):
    """The part of the stretch shown, large; see the module docstring. Takes the
    fingers as touch events; on a PC the mouse drags and taps, the wheel zooms."""

    view_changed = pyqtSignal()  # zoomed or scrolled by hand

    def __init__(self, model: CutModel) -> None:
        super().__init__(model)
        self.view = (0.0, 1.0)
        # One finger: the handle it drags (and how far beside its line it came down),
        # or where it came down and the part shown then, until it moves (a scroll) or
        # lifts (a tap).
        self._handle: str | None = None
        self._offset = 0.0
        self._pressed_x: float | None = None
        self._pressed_view = self.view
        self._scrolling = False
        # Two fingers: their first distance, the time under their middle and the span.
        self._pinch: tuple[float, float, float] | None = None
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents)

    # --- Geometry ---------------------------------------------------------------------

    def plot_rect(self) -> QRectF:
        return QRectF(
            16, NAME_STRIP, self.width() - 32, self.height() - NAME_STRIP - AXIS_BAND - TAB_BAND
        )

    def x_of(self, seconds: float) -> float:
        rect = self.plot_rect()
        v0, v1 = self.view
        return rect.left() + (seconds - v0) / (v1 - v0) * rect.width()

    def t_of(self, x: float) -> float:
        rect = self.plot_rect()
        v0, v1 = self.view
        return v0 + (x - rect.left()) / max(rect.width(), 1.0) * (v1 - v0)

    def set_view(self, start: float, end: float) -> None:
        """Shows `start` to `end` seconds, kept to a sensible part of the stretch."""
        stretch = self._model.stretch
        duration = stretch.duration if stretch is not None else 1.0
        span = min(max(end - start, MIN_VIEW_S), duration * (1 + 2 * VIEW_MARGIN))
        margin = VIEW_MARGIN * span
        start = min(max(start, -margin), duration + margin - span)
        self.view = (start, start + span)
        self.request_paint()

    def ensure_visible(self, seconds: float) -> bool:
        """Scrolls `seconds` into view where it is out of it; whether it scrolled."""
        v0, v1 = self.view
        span = v1 - v0
        if v0 <= seconds <= v1:
            return False
        start = seconds - 0.25 * span if seconds < v0 else seconds - 0.75 * span
        self.set_view(start, start + span)
        return True

    def handle_at(self, point: QPointF) -> str | None:
        """The handle a finger at `point` takes: on its tab, or near its line."""
        model = self._model
        if model.stretch is None:
            return None
        tab_top = self.plot_rect().bottom() + AXIS_BAND + 4
        for handle in (model.selected, *(h for h in (START, END) if h != model.selected)):
            if self._tab_rect(handle, tab_top).contains(point):
                return handle
        nearest, distance = None, float(HANDLE_REACH)
        for handle, seconds in ((START, model.start), (END, model.end)):
            gap = abs(self.x_of(seconds) - point.x())
            if gap < distance or (gap == distance and handle == model.selected):
                nearest, distance = handle, gap
        return nearest

    def _tab_rect(self, handle: str, top: float) -> QRectF:
        x = self.x_of(self._model.start if handle == START else self._model.end)
        left = x if handle == START else x - TAB_WIDTH
        return QRectF(left, top, TAB_WIDTH, TAB_HEIGHT)

    # --- Fingers and mouse ------------------------------------------------------------

    def event(self, event: QEvent) -> bool:
        if event.type() in _TOUCH_EVENTS:
            self._on_touch(event)  # type: ignore[arg-type]
            event.accept()
            return True
        return super().event(event)

    def _on_touch(self, touch: QTouchEvent) -> None:
        kind = touch.type()
        down = [p for p in touch.points() if p.state() != QEventPoint.State.Released]
        if kind == QEvent.Type.TouchBegin:
            self._pinch = None
            if len(down) >= 2:
                self._cancel_press()
                self._pinch_to(down)
            elif down:
                self.press(self._local(down[0]))
        elif kind == QEvent.Type.TouchUpdate:
            if len(down) >= 2:
                self._cancel_press()
                self._pinch_to(down)
            elif down and self._pinch is None:
                self.drag_to(self._local(down[0]))
        elif kind == QEvent.Type.TouchEnd:
            if self._pinch is None and touch.points():
                self.release(self._local(touch.points()[0]))
            self._pinch = None
            self._cancel_press()
        else:
            self._pinch = None
            self._cancel_press()

    def _local(self, point: QEventPoint) -> QPointF:
        return self.mapFromGlobal(point.globalPosition())

    def _pinch_to(self, down: list[QEventPoint]) -> None:
        first, second = sorted(down, key=lambda p: p.id())[:2]
        a, b = self._local(first), self._local(second)
        distance = max(math.hypot(a.x() - b.x(), a.y() - b.y()), 1.0)
        centre = (a.x() + b.x()) / 2
        if self._pinch is None:
            self._pinch = (distance, self.t_of(centre), self.view[1] - self.view[0])
            return
        start_distance, anchor, start_span = self._pinch
        span = start_span * start_distance / distance
        rect = self.plot_rect()
        fraction = (centre - rect.left()) / max(rect.width(), 1.0)
        begin = anchor - fraction * span
        self.set_view(begin, begin + span)
        self.view_changed.emit()

    def _cancel_press(self) -> None:
        if self._handle is not None:
            self._model.let_go(self._handle)
        self._handle = None
        self._pressed_x = None
        self._scrolling = False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.press(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self.drag_to(event.position())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.release(event.position())

    def wheelEvent(self, event: QWheelEvent) -> None:
        steps = event.angleDelta().y() / 120
        if steps and self._model.stretch is not None:
            anchor = self.t_of(event.position().x())
            v0, v1 = self.view
            scale = 0.85**steps
            self.set_view(anchor - (anchor - v0) * scale, anchor + (v1 - anchor) * scale)
            self.view_changed.emit()
        event.accept()

    def press(self, point: QPointF) -> None:
        """A finger came down: on a handle it takes it, elsewhere it may tap or
        scroll."""
        if self._model.stretch is None:
            return
        handle = self.handle_at(point)
        if handle is not None:
            self._handle = handle
            seconds = self._model.start if handle == START else self._model.end
            self._offset = self.x_of(seconds) - point.x()
            self._model.select(handle)
            return
        self._pressed_x = point.x()
        self._pressed_view = self.view
        self._scrolling = False

    def drag_to(self, point: QPointF) -> None:
        if self._handle is not None:
            rect = self.plot_rect()
            x = min(max(point.x() + self._offset, rect.left()), rect.right())
            self._model.move(self._handle, self.t_of(x), preview=False)
            return
        if self._pressed_x is None:
            return
        moved = point.x() - self._pressed_x
        if not self._scrolling and abs(moved) < TAP_SLOP:
            return
        self._scrolling = True
        v0, v1 = self._pressed_view
        shift = -moved / max(self.plot_rect().width(), 1.0) * (v1 - v0)
        self.set_view(v0 + shift, v1 + shift)
        self.view_changed.emit()

    @property
    def dragging(self) -> bool:
        """A finger or the mouse drags a handle."""
        return self._handle is not None

    def release(self, point: QPointF) -> None:
        if self._handle is not None:
            handle, self._handle = self._handle, None
            self._model.let_go(handle)
            return
        if self._pressed_x is not None and not self._scrolling:
            self._model.tap(self.t_of(point.x()))
        self._pressed_x = None
        self._scrolling = False

    # --- Painting ---------------------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        self._panel(painter)
        model = self._model
        if model.stretch is None:
            painter.end()
            return
        rect = self.plot_rect()
        painter.save()
        painter.setClipRect(QRectF(rect.left(), 0, rect.width(), self.height()))
        self._paint_selection(painter, rect)
        self._paint_pauses(painter, rect)
        selection = (self.x_of(model.start), self.x_of(model.end))
        self._paint_level(painter, rect, self.view, selection, rect.height() / 2 - 14, peaks=True)
        painter.setPen(QPen(_alpha(self._theme.outline, 160), 1))
        middle = rect.center().y()
        painter.drawLine(QPointF(rect.left(), middle), QPointF(rect.right(), middle))
        self._paint_files(painter, rect)
        self._paint_marks(painter, rect)
        self._paint_axis(painter, rect)
        self._paint_position(painter, rect)
        painter.restore()
        self._paint_handles(painter, rect)
        self._paint_reading(painter, rect)
        painter.end()

    def _paint_selection(self, painter: QPainter, rect: QRectF) -> None:
        model = self._model
        left, right = self.x_of(model.start), self.x_of(model.end)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_alpha(self._theme.accent, 26))
        painter.drawRect(QRectF(left, rect.top(), right - left, rect.height()))

    def _paint_pauses(self, painter: QPainter, rect: QRectF) -> None:
        theme = self._theme
        for start, end in self._model.pauses:
            left, right = self.x_of(start), self.x_of(end)
            if right < rect.left() or left > rect.right():
                continue
            painter.setPen(Qt.PenStyle.NoPen)
            if right - left >= 6:
                painter.setBrush(_alpha(theme.notice, 34))
                painter.drawRect(QRectF(left, rect.top(), right - left, rect.height()))
            centre = (left + right) / 2
            tip = rect.bottom() - 2
            painter.setBrush(QColor(theme.notice))
            painter.drawPolygon(
                QPolygonF(
                    [QPointF(centre, tip - 12), QPointF(centre - 8, tip), QPointF(centre + 8, tip)]
                )
            )

    def _paint_files(self, painter: QPainter, rect: QRectF) -> None:
        """The names along the top, each staying in view while its file is; the
        boundary of a file dashed through the level."""
        model = self._model
        stretch = model.stretch
        assert stretch is not None
        theme = self._theme
        font = QFont(self.font())
        font.setPixelSize(SMALL_FONT_PX)
        painter.setFont(font)
        metrics = QFontMetricsF(font)
        clips = stretch.clips
        for index, clip in enumerate(clips):
            following = clips[index + 1].start if index + 1 < len(clips) else stretch.duration
            left = max(self.x_of(clip.start), rect.left())
            right = min(self.x_of(following), rect.right())
            if right <= left:
                continue
            x = self.x_of(clip.start)
            if clip.start > 0 and rect.left() < x < rect.right():
                pen = QPen(QColor(theme.text), 2)
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawLine(QPointF(x, 6), QPointF(x, rect.bottom()))
            opened = clip.recording.id == stretch.opened.id
            painter.setPen(QColor(theme.text if opened else theme.muted))
            width = max(0.0, right - left - 16)
            text = metrics.elidedText(clip.recording.name, Qt.TextElideMode.ElideRight, width)
            painter.drawText(
                QRectF(left + 8, 6, width, NAME_STRIP - 10),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                text,
            )

    def _paint_marks(self, painter: QPainter, rect: QRectF) -> None:
        """Where a dynamic label appears first: a mark at the top, the text beside it
        as far as the next."""
        marks = [(self.x_of(at), text) for at, text in self._model.marks]
        if not marks:
            return
        theme = self._theme
        font = QFont(self.font())
        font.setPixelSize(14)
        painter.setFont(font)
        metrics = QFontMetricsF(font)
        top = rect.top() + 4
        for index, (x, text) in enumerate(marks):
            following = marks[index + 1][0] if index + 1 < len(marks) else rect.right()
            if following < rect.left() or x > rect.right():
                continue
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(theme.muted))
            painter.drawPolygon(
                QPolygonF([QPointF(x - 6, top), QPointF(x + 6, top), QPointF(x, top + 8)])
            )
            width = min(following - x - 12, MARK_MAX_WIDTH)
            if width < 40:
                continue
            shown = metrics.elidedText(text, Qt.TextElideMode.ElideRight, width - 12)
            pill = QRectF(x + 8, top, metrics.horizontalAdvance(shown) + 12, 20)
            painter.setBrush(_alpha(theme.raised, 220))
            painter.drawRoundedRect(pill, 6, 6)
            painter.setPen(QColor(theme.text))
            painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, shown)

    def _paint_axis(self, painter: QPainter, rect: QRectF) -> None:
        theme = self._theme
        model = self._model
        v0, v1 = self.view
        per_px = (v1 - v0) / max(rect.width(), 1.0)
        spacing = next(
            (s for s in TICK_SPACINGS_S if s / per_px >= MIN_TICK_PX), TICK_SPACINGS_S[-1]
        )
        font = QFont(self.font())
        font.setPixelSize(SMALL_FONT_PX)
        painter.setFont(font)
        origin = model.clock(0.0)
        tick = math.ceil((origin + v0) / spacing) * spacing - origin
        while tick <= v1:
            x = self.x_of(tick)
            if rect.left() + 50 <= x <= rect.right() - 50:
                painter.setPen(QPen(QColor(theme.outline), 1))
                painter.drawLine(QPointF(x, rect.bottom()), QPointF(x, rect.bottom() + 6))
                painter.setPen(QColor(theme.muted))
                painter.drawText(
                    QRectF(x - 60, rect.bottom() + 4, 120, AXIS_BAND - 4),
                    Qt.AlignmentFlag.AlignCenter,
                    clock_text(origin + tick, tenths=spacing < 1),
                )
            tick += spacing

    def _paint_position(self, painter: QPainter, rect: QRectF) -> None:
        theme = self._theme
        x = self.x_of(self._model.position)
        painter.setPen(QPen(QColor(theme.text), 2))
        painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.text))
        painter.drawPolygon(
            QPolygonF(
                [QPointF(x - 7, rect.top()), QPointF(x + 7, rect.top()), QPointF(x, rect.top() + 9)]
            )
        )

    def _paint_handles(self, painter: QPainter, rect: QRectF) -> None:
        theme = self._theme
        model = self._model
        font = QFont(self.font())
        font.setPixelSize(18)
        font.setBold(True)
        painter.setFont(font)
        tab_top = rect.bottom() + AXIS_BAND + 4
        for handle, seconds, text in (
            (START, model.start, tr_context("cut handle", "Start")),
            (END, model.end, tr_context("cut handle", "End")),
        ):
            x = self.x_of(seconds)
            if not rect.left() - 1 <= x <= rect.right() + 1:
                continue
            chosen = handle == model.selected
            painter.setPen(QPen(QColor(theme.accent), 3))
            painter.drawLine(QPointF(x, rect.top() - 4), QPointF(x, tab_top))
            # The tab beside the line, on the side of what is kept, so that the two
            # never cover each other.
            tab = self._tab_rect(handle, tab_top)
            path = QPainterPath()
            path.addRoundedRect(tab, 12, 12)
            corner = QPainterPath()
            corner.addRect(
                QRectF(tab.left() if handle == START else tab.right() - 12, tab.top(), 12, 12)
            )
            path = path.united(corner)
            if chosen:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(theme.accent))
            else:
                painter.setPen(QPen(QColor(theme.accent), 3))
                painter.setBrush(QColor(theme.raised))
            painter.drawPath(path)
            painter.setPen(QColor(theme.on_accent if chosen else theme.accent))
            painter.drawText(tab, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_reading(self, painter: QPainter, rect: QRectF) -> None:
        """While the level is read: how far, where it is still missing."""
        model = self._model
        stretch = model.stretch
        assert stretch is not None
        read = model.levels_read
        if read >= 1.0:
            return
        x = max(self.x_of(read * stretch.duration), rect.left())
        if x >= rect.right() - 80:
            return
        font = QFont(self.font())
        font.setPixelSize(SMALL_FONT_PX)
        painter.setFont(font)
        painter.setPen(QColor(self._theme.muted))
        painter.drawText(
            QRectF(x, rect.top(), rect.right() - x, rect.height()),
            Qt.AlignmentFlag.AlignCenter,
            tr(_READING_LEVEL).format(percent=round(100 * read)),
        )


class NameSheet(Overlay):
    """The names the station sent during the stretch, to take one with a tap."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.panel.setFixedWidth(NAME_SHEET_WIDTH)
        self.title = set_role(QLabel(), "large")
        self._rows = QVBoxLayout()
        self._rows.setSpacing(4)
        self._cancel = QPushButton()
        self._cancel.clicked.connect(self.dismiss)
        self.buttons: list[QPushButton] = []
        buttons = QHBoxLayout()
        buttons.addWidget(self._cancel)
        buttons.addStretch()
        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        layout.addWidget(self.title)
        layout.addLayout(self._rows)
        layout.addLayout(buttons)

    def open(self, names: list[tuple[str, str]], chosen) -> None:
        """`chosen` gets artist and title of the name tapped."""
        # Translators: the title of the sheet with the names a station sent.
        self.title.setText(tr("Names the station sent"))
        self._cancel.setText(tr("Cancel"))
        for button in self.buttons:
            self._rows.removeWidget(button)
            button.hide()
            button.deleteLater()
        self.buttons = []
        for artist, title in names:
            button = QPushButton(name_text(artist, title))
            button.setStyleSheet(
                f"min-height: {NAME_ROW_HEIGHT}px; max-height: {NAME_ROW_HEIGHT}px; text-align: left;"
            )
            button.clicked.connect(
                lambda _checked=False, a=artist, t=title: self._choose(chosen, a, t)
            )
            self._rows.addWidget(button)
            self.buttons.append(button)
        self.show_over()

    def _choose(self, chosen, artist: str, title: str) -> None:
        self.dismiss()
        chosen(artist, title)


class CutPage(QWidget):
    """Title, what is open and the way back; the whole stretch; the part shown with
    the handles; play, what is shown and the steps of the handle chosen; the name and
    "Keep" or "Save"."""

    def __init__(self, model: CutModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._theme = THEMES[DARK]
        self._keyboard: Keyboard | None = None
        self._sheet: ActionSheet | None = None
        self._names: NameSheet | None = None
        self._shown = TITLE_VIEW  # what the view follows: TITLE_VIEW, START, END or ""
        self._handles = (0.0, 0.0)

        title = set_role(QLabel(tr("Cut")), "large")
        self.summary = set_role(ElidedLabel(), "muted")
        self.back_button = QPushButton(tr("Back"))
        header = QHBoxLayout()
        header.setSpacing(16)
        header.addWidget(title)
        header.addWidget(self.summary, stretch=1)
        header.addWidget(self.back_button)

        self.overview = OverviewStrip(model)
        self.overview.view_requested.connect(self._centre_view)
        self.wave = WaveView(model)
        self.wave.view_changed.connect(self._on_view_changed)

        self.play_button = QToolButton()
        self.play_button.setStyleSheet("min-width: 72px; max-width: 72px; padding: 0px;")
        self.play_button.setIconSize(PLAY_ICON)
        self.play_button.clicked.connect(model.toggle)
        self.views = QButtonGroup(self)
        views = QHBoxLayout()
        views.setSpacing(8)
        self.view_buttons: dict[str, QPushButton] = {}
        for key, text in (
            (TITLE_VIEW, tr("Title")),
            (START, tr_context("cut handle", "Start")),
            (END, tr_context("cut handle", "End")),
        ):
            button = set_role(QPushButton(text), "chip")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, k=key: self.show_part(k))
            self.views.addButton(button)
            views.addWidget(button)
            self.view_buttons[key] = button
        steps = QHBoxLayout()
        steps.setSpacing(8)
        self.step_buttons: list[QPushButton] = []
        for seconds in STEPS_S:
            sign = "+" if seconds > 0 else "−"
            button = QPushButton(_decimal(f"{sign}{abs(seconds):g} s"))
            button.setStyleSheet("padding: 0px 8px;")
            button.clicked.connect(lambda _checked=False, s=seconds: model.step(s))
            steps.addWidget(button)
            self.step_buttons.append(button)
        pauses = QHBoxLayout()
        pauses.setSpacing(8)
        # Translators: jumps the handle chosen to the pause before it.
        self.earlier_pause = QPushButton(tr("‹ Pause"))
        # Translators: jumps the handle chosen to the pause after it.
        self.later_pause = QPushButton(tr("Pause ›"))
        for button, forward in ((self.earlier_pause, False), (self.later_pause, True)):
            button.setStyleSheet("padding: 0px 12px;")
            button.clicked.connect(lambda _checked=False, f=forward: model.jump_to_pause(f))
            pauses.addWidget(button)
        controls = QHBoxLayout()
        controls.setSpacing(16)
        controls.addWidget(self.play_button)
        controls.addLayout(views)
        controls.addStretch()
        controls.addLayout(steps)
        controls.addLayout(pauses)

        # Translators: opens the names the station sent, to take one for the title cut.
        self.names_button = QPushButton(tr("Names"))
        self.names_button.clicked.connect(self._open_names)
        self.artist_button = set_role(QPushButton(), "value")
        self.artist_button.clicked.connect(self._edit_artist)
        dash = QLabel("–")
        self.title_button = set_role(QPushButton(), "value")
        self.title_button.setMinimumWidth(260)
        self.title_button.clicked.connect(self._edit_title)
        self.times = set_role(QLabel(), "muted")
        self.keep_button = set_role(QPushButton(), "accent")
        self.keep_button.setMinimumWidth(180)
        self.keep_button.clicked.connect(self._save)
        # What the button did last in this session, for its tick, and what it does now.
        self._done_label = ""
        self._saving_label = ""
        self._saving_replaces = False
        bottom = QHBoxLayout()
        bottom.setSpacing(12)
        bottom.addWidget(self.names_button)
        bottom.addWidget(self.artist_button)
        bottom.addWidget(dash)
        bottom.addWidget(self.title_button)
        bottom.addStretch()
        bottom.addWidget(self.times)
        bottom.addSpacing(8)
        bottom.addWidget(self.keep_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(14)
        layout.addLayout(header)
        layout.addWidget(self.overview)
        layout.addWidget(self.wave, stretch=1)
        layout.addLayout(controls)
        layout.addLayout(bottom)

        model.opened.connect(self._on_opened)
        model.failed.connect(self._on_failed)
        model.changed.connect(self._show)
        model.levels_changed.connect(self._on_levels)
        model.title_saved.connect(self._on_saved)
        model.save_failed.connect(self._on_save_failed)
        self._summary_override = ""
        self._show()

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.overview.set_theme(theme)
        self.wave.set_theme(theme)
        if self._keyboard is not None:
            self._keyboard.apply_theme(theme)
        self._show_icon()

    def open(self, recording_id: str) -> None:
        """Opens a recording for cutting."""
        self._summary_override = ""
        self._done_label = ""
        self._model.open(recording_id)

    def keyboard(self) -> Keyboard:
        if self._keyboard is None:
            self._keyboard = Keyboard(self.window())
            self._keyboard.apply_theme(self._theme)
        return self._keyboard

    def sheet(self) -> ActionSheet:
        if self._sheet is None:
            self._sheet = ActionSheet(self.window())
        return self._sheet

    def name_sheet(self) -> NameSheet:
        if self._names is None:
            self._names = NameSheet(self.window())
        return self._names

    def hideEvent(self, event: QHideEvent) -> None:
        """Leaving the page ends what plays of the stretch: the station is heard again.
        The session stays until "Back" or another recording."""
        self._model.stop_playback()
        super().hideEvent(event)

    # --- What is shown ----------------------------------------------------------------

    def show_part(self, part: str) -> None:
        """Shows the title, its beginning or its end; the latter two choose the handle
        the steps move."""
        self._shown = part
        if part in (START, END):
            self._model.select(part)
        self._check(part)
        if self._model.stretch is not None:
            self._set_view(*view_preset(self._model, part))

    def _check(self, part: str) -> None:
        button = self.view_buttons.get(part)
        if button is not None:
            button.setChecked(True)
            return
        self.views.setExclusive(False)
        for other in self.view_buttons.values():
            other.setChecked(False)
        self.views.setExclusive(True)

    def _set_view(self, start: float, end: float) -> None:
        self.wave.set_view(start, end)
        self.overview.view = self.wave.view
        self.overview.request_paint()

    def _centre_view(self, seconds: float) -> None:
        v0, v1 = self.wave.view
        span = v1 - v0
        self._set_view(seconds - span / 2, seconds + span / 2)
        self._on_view_changed()

    def _on_view_changed(self) -> None:
        """Zoomed or scrolled by hand: the view follows no handle any more."""
        self._shown = ""
        self._check("")
        self.overview.view = self.wave.view
        self.overview.request_paint()

    def _on_opened(self) -> None:
        self._handles = (self._model.start, self._model.end)
        self.show_part(TITLE_VIEW)
        self._show()

    def _on_failed(self, message: str) -> None:
        self.sheet().open(tr(_CANNOT_CUT), message, [], cancel=tr("OK"))

    def _on_levels(self) -> None:
        self.wave.request_paint()
        self.overview.request_paint()
        self._show_pause_buttons()

    def _show(self) -> None:
        model = self._model
        stretch = model.stretch
        loaded = stretch is not None
        handles = (model.start, model.end)
        if loaded and handles != self._handles:
            moved = START if handles[0] != self._handles[0] else END
            self._handles = handles
            if self.wave.dragging:
                # The finger holds the handle where it is: a view following it would
                # pull the level away beneath the finger, and draw it anew each time.
                pass
            elif self._shown in (START, END):
                self._set_view(*view_preset(model, self._shown))
            elif self.wave.ensure_visible(model.start if moved == START else model.end):
                self._on_view_changed()
            self._summary_override = ""
        if loaded and model.playing:
            self._follow(model.position)
        self.summary.setText(self._summary_override or self._summary(stretch))
        for button in (*self.view_buttons.values(), *self.step_buttons, self.play_button):
            button.setEnabled(loaded)
        self._show_icon()
        self._show_pause_buttons()
        self.names_button.setEnabled(loaded and bool(model.suggestions))
        for button, text, placeholder in (
            (self.artist_button, model.artist, tr("Artist")),
            (self.title_button, model.title, tr("Title")),
        ):
            button.setEnabled(loaded)
            button.setText(text or placeholder)
            if button.property("placeholder") != (not text):
                # Styled anew only when it changes: a repolish costs, and this runs
                # with every move of a handle.
                button.setProperty("placeholder", not text)
                repolish(button)
        if loaded:
            self.times.setText(
                f"{clock_text(model.clock(model.start), True)} – "
                f"{clock_text(model.clock(model.end), True)} · {length_text(model.end - model.start)}"
            )
        else:
            self.times.setText("")
        label = tr("Save") if model.replacing else tr("Keep")
        # The tick only after keeping or saving in this session: a kept title opened
        # as it is needs no saving, but nothing was done.
        done = model.saved and self._done_label
        self.keep_button.setText(f"{self._done_label} ✓" if done else label)
        self.keep_button.setEnabled(loaded and not model.saved and not model.saving)
        self.wave.request_paint()
        self.overview.request_paint()

    def _follow(self, position: float) -> None:
        """While it plays, the view turns over where the position runs out of it on the
        right; a view scrolled away by hand stays."""
        v0, v1 = self.wave.view
        span = v1 - v0
        if v1 <= position < v1 + FOLLOW_REACH * span:
            begin = position - FOLLOW_LEAD * span
            self._set_view(begin, begin + span)
            self._on_view_changed()

    def _summary(self, stretch: Stretch | None) -> str:
        if stretch is None:
            return tr("Reading the recording …") if self._model.loading else ""
        clips = stretch.clips
        first = clock_text(stretch.origin)[:5]
        last = clock_text(stretch.origin + stretch.duration)[:5]
        return " · ".join(
            (
                stretch.opened.station,
                tr("{first} to {last}").format(first=first, last=last),
                tr_plural("{count} recording", "{count} recordings", len(clips)).format(
                    count=len(clips)
                ),
            )
        )

    def _show_icon(self) -> None:
        name = "pause" if self._model.playing else "play"
        self.play_button.setIcon(symbol_icon(name, self._theme.color("text"), PLAY_ICON))

    def _show_pause_buttons(self) -> None:
        self.earlier_pause.setEnabled(self._model.can_jump(False))
        self.later_pause.setEnabled(self._model.can_jump(True))

    # --- The name and keeping ---------------------------------------------------------

    def _edit_artist(self) -> None:
        model = self._model
        self.keyboard().open(
            tr("Artist"),
            lambda text: model.set_name(text, model.title),
            text=model.artist,
            check=lambda _text: True,
        )

    def _edit_title(self) -> None:
        model = self._model
        self.keyboard().open(
            tr("Title"),
            lambda text: model.set_name(model.artist, text),
            text=model.title,
            check=lambda _text: True,
        )

    def _open_names(self) -> None:
        self.name_sheet().open(self._model.suggestions, self._model.set_name)

    def _save(self) -> None:
        self._saving_label = tr("Save") if self._model.replacing else tr("Keep")
        self._saving_replaces = self._model.replacing
        self._model.save()

    def _on_saved(self, recording: object) -> None:
        self._done_label = self._saving_label
        if self._saving_replaces:
            text = tr("Saved: {name} · {length}")
        else:
            text = tr("Kept: {name} · {length}")
        self._summary_override = text.format(
            name=recording.name, length=length_text(recording.duration)
        )
        self._show()

    def _on_save_failed(self, message: str) -> None:
        self.sheet().open(tr(_CANNOT_KEEP), message, [], cancel=tr("OK"))
