"""The equalizer ("Sound"): a sub-page of the radio page, reached by the button beside the
loudspeaker (`ui.touch.radio_page`) or from the settings (`ui.touch.settings_page`).
Two pages to swipe (`ui.touch.pager`):

- the curve: the response of the five bands over the spectrum of the audio
  (`audio.spectrum`), each band a point at its frequency and gain. A finger on a point
  drags it (frequency across, gain up and down), two fingers spread or close the width
  of the chosen band (only a peaking band has one). A finger beside the points turns
  the page.
- fine adjustment: the band by its name, steppers for frequency, gain and width, the
  curve small, and the presets.

What it shows and changes is the controller's equalizer in effect
(`ReceiverController.equalizer_state`): the one of all stations, or while per station
the one of the station playing.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QEventPoint,
    QFont,
    QFontMetricsF,
    QHideEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPolygonF,
    QShowEvent,
    QTouchEvent,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from audio.equalizer import (
    BASS,
    FREQUENCY_RANGE_HZ,
    GAIN_RANGE_DB,
    HIGH_SHELF,
    LINEAR,
    LOW_SHELF,
    PEAK,
    SPEECH,
    TREBLE,
    WIDTH_RANGE_OCTAVES,
    Band,
    EqualizerSettings,
    EqualizerState,
    headroom_db,
    response_db,
)
from i18n import tr, tr_noop
from ui.receiver_controller import ReceiverController
from ui.touch.pager import Pager
from ui.touch.theme import DARK, FONT_PX, PANEL_RADIUS, THEMES, Theme
from ui.touch.widgets import ChoiceButtons, ElidedLabel, Stepper, set_role

# The names of the five bands, from low to high (marked, translated where shown).
BAND_NAMES = (
    tr_noop("Lows"),
    tr_noop("Low mids"),
    tr_noop("Mids"),
    tr_noop("Presence"),
    tr_noop("Brilliance"),
)
KIND_NAMES = {
    LOW_SHELF: tr_noop("Low shelf"),
    # Translators: a peaking filter, the bell-shaped band of an equalizer.
    PEAK: tr_noop("Bell"),
    HIGH_SHELF: tr_noop("High shelf"),
}
# The presets in the order of their chips, and the settings made by hand.
CUSTOM = "custom"
PRESET_NAMES = {
    # Translators: the preset of the equalizer that leaves the sound as it is.
    LINEAR: tr_noop("Flat"),
    SPEECH: tr_noop("Speech"),
    BASS: tr_noop("Bass"),
    TREBLE: tr_noop("Treble"),
    # Translators: the preset chip for the settings made by hand.
    CUSTOM: tr_noop("Custom"),
}

# The preferred numbers R40 (ISO 3), some twelfth of an octave apart: the frequencies a
# band is set to, by stepper and by finger alike.
_R40 = (
    1.0, 1.06, 1.12, 1.18, 1.25, 1.32, 1.4, 1.5, 1.6, 1.7,
    1.8, 1.9, 2.0, 2.12, 2.24, 2.36, 2.5, 2.65, 2.8, 3.0,
    3.15, 3.35, 3.55, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.3,
    5.6, 6.0, 6.3, 6.7, 7.1, 7.5, 8.0, 8.5, 9.0, 9.5,
)  # fmt: skip


def _r40_frequencies() -> tuple[float, ...]:
    low, high = FREQUENCY_RANGE_HZ
    frequencies = (round(m * decade, 6) for decade in (10, 100, 1000, 10_000) for m in _R40)
    return tuple(f for f in frequencies if low <= f <= high)


FREQUENCIES_HZ = _r40_frequencies()
GAIN_STEP_DB = 0.5
GAINS_DB = tuple(
    step * GAIN_STEP_DB
    for step in range(
        round(GAIN_RANGE_DB[0] / GAIN_STEP_DB), round(GAIN_RANGE_DB[1] / GAIN_STEP_DB) + 1
    )
)
WIDTHS_OCTAVES = tuple(
    round(tenth / 10, 1)
    for tenth in range(round(WIDTH_RANGE_OCTAVES[0] * 10), round(WIDTH_RANGE_OCTAVES[1] * 10) + 1)
)

# The curves are drawn for the usual output rate of DAB+.
RESPONSE_RATE = 48_000
_CURVE_FREQUENCIES = np.geomspace(*FREQUENCY_RANGE_HZ, 240)
_DECADES = math.log10(FREQUENCY_RANGE_HZ[1] / FREQUENCY_RANGE_HZ[0])
_FREQUENCY_TICKS = (50, 100, 200, 500, 1000, 2000, 5000, 10_000)
_SMALL_TICKS = (100, 1000, 10_000)
# The spectrum from this level (dBFS) at the bottom to 0 dBFS at the top.
SPECTRUM_FLOOR_DB = -90.0
# From one spectrum to the next a band falls at most this much, so the display calms
# down; it rises at once.
SPECTRUM_FALL_DB = 3.0
HANDLE_RADIUS = 22
# A finger this near to a point takes it, more than the point: a fingertip is wide.
HANDLE_REACH = 40
AXIS_FONT_PX = 14
HINT = tr_noop("Drag a point · spread two fingers: width")
FINE_CURVE_HEIGHT = 188
PRESETS_WIDTH = 288
STEPPER_LABEL_WIDTH = 140


def frequency_text(hz: float) -> str:
    """ "60 Hz", "1.06 kHz", "4.0 kHz", "12.5 kHz"."""
    if hz < 1000:
        return f"{hz:g} Hz"
    khz = hz / 1000
    if khz >= 10 or round(khz, 1) == round(khz, 2):
        return f"{khz:.1f} kHz"
    return f"{khz:.2f} kHz"


def gain_text(db: float) -> str:
    return "0.0 dB" if db == 0 else f"{db:+.1f} dB".replace("-", "−")


def width_text(octaves: float) -> str:
    # Translators: the width of a band in octaves, shortened.
    return tr("{octaves:.1f} oct.").format(octaves=octaves)


def _rough_frequency(hz: float) -> str:
    return f"{hz:.0f} Hz" if hz < 1000 else f"{hz / 1000:.1f} kHz"


def band_range_text(band: Band) -> str:
    """Between which frequencies a peaking band acts (its width); a shelf from where."""
    if band.kind == LOW_SHELF:
        return tr("below {frequency}").format(frequency=_rough_frequency(band.frequency_hz))
    if band.kind == HIGH_SHELF:
        return tr("above {frequency}").format(frequency=_rough_frequency(band.frequency_hz))
    low = band.frequency_hz * 2.0 ** (-band.width_octaves / 2)
    high = band.frequency_hz * 2.0 ** (band.width_octaves / 2)
    return tr("{low} to {high}").format(low=_rough_frequency(low), high=_rough_frequency(high))


def summary_text(state: EqualizerState, per_station: bool, own: bool) -> str:
    """The line beside the title: the preset (or "Custom setting"), for whom it is,
    and by how much the audio is lowered for the boosts."""
    preset = state.settings.preset
    name = tr(PRESET_NAMES[preset]) if preset in PRESET_NAMES else tr("Custom setting")
    if own:
        scope = tr("for this station")
    elif per_station:
        scope = tr("like all stations")
    else:
        scope = tr("for all stations")
    parts = [name, scope]
    headroom = headroom_db(state.settings, RESPONSE_RATE)
    if state.enabled and headroom >= GAIN_STEP_DB:
        parts.append(tr("Level −{headroom:.0f} dB").format(headroom=headroom))
    return " · ".join(parts)


def _nearest_frequency(hz: float) -> float:
    return min(FREQUENCIES_HZ, key=lambda value: abs(math.log(value / hz)))


def _nearest(values: tuple[float, ...], value: float) -> float:
    return min(values, key=lambda candidate: abs(candidate - value))


def _alpha(color: str, alpha: int) -> QColor:
    result = QColor(color)
    result.setAlpha(alpha)
    return result


_TOUCH_EVENTS = (
    QEvent.Type.TouchBegin,
    QEvent.Type.TouchUpdate,
    QEvent.Type.TouchEnd,
    QEvent.Type.TouchCancel,
)


class CurveView(QWidget):
    """The response of the bands over the spectrum of the audio, on a panel of its own.
    `interactive`: large, the bands points to drag, the chosen band's name and figures
    above; else small, the chosen band marked.

    Takes the fingers as touch events (two of them for the width); on a PC the mouse
    drags and the wheel sets the width."""

    band_chosen = pyqtSignal(int)
    band_moved = pyqtSignal(int, object)  # index, the band as the finger has it (Band)

    def __init__(self, interactive: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._interactive = interactive
        self._settings = EqualizerSettings()
        self._enabled = False
        self._selected = 0
        self._curve = np.zeros(len(_CURVE_FREQUENCIES))
        self._spectrum: tuple[np.ndarray, np.ndarray] | None = None
        self._theme = THEMES[DARK]
        # The band being dragged and where its point lies from the finger.
        self._drag: tuple[int, QPointF] | None = None
        # Two fingers: their first distance and the width of the band then.
        self._pinch: tuple[float, float] | None = None
        if interactive:
            self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents)

    # --- What is shown ----------------------------------------------------------------

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.update()

    def set_settings(self, settings: EqualizerSettings, enabled: bool) -> None:
        self._settings = settings
        self._enabled = enabled
        self._curve = response_db(settings, _CURVE_FREQUENCIES, RESPONSE_RATE)
        self.update()

    def set_selected(self, index: int) -> None:
        self._selected = index
        self.update()

    def selected(self) -> int:
        return self._selected

    def set_spectrum(self, centres: np.ndarray, levels: np.ndarray) -> None:
        previous = self._spectrum
        if previous is not None and previous[0].shape == centres.shape:
            levels = np.maximum(levels, previous[1] - SPECTRUM_FALL_DB)
        self._spectrum = (centres, levels)
        self.update()

    def clear_spectrum(self) -> None:
        self._spectrum = None
        self.update()

    # --- Geometry ---------------------------------------------------------------------

    def plot_rect(self) -> QRectF:
        if self._interactive:
            left, top, right, bottom = 56, 60, 24, 44
        else:
            left, top, right, bottom = 48, 12, 16, 24
        return QRectF(left, top, self.width() - left - right, self.height() - top - bottom)

    def x_of(self, hz: float) -> float:
        rect = self.plot_rect()
        return rect.left() + math.log10(hz / FREQUENCY_RANGE_HZ[0]) / _DECADES * rect.width()

    def frequency_at(self, x: float) -> float:
        rect = self.plot_rect()
        fraction = min(max((x - rect.left()) / rect.width(), 0.0), 1.0)
        return FREQUENCY_RANGE_HZ[0] * 10.0 ** (fraction * _DECADES)

    def y_of(self, db: float) -> float:
        rect = self.plot_rect()
        return rect.center().y() - db / GAIN_RANGE_DB[1] * rect.height() / 2

    def gain_at(self, y: float) -> float:
        rect = self.plot_rect()
        db = (rect.center().y() - y) / (rect.height() / 2) * GAIN_RANGE_DB[1]
        return min(max(db, GAIN_RANGE_DB[0]), GAIN_RANGE_DB[1])

    def handle_centre(self, index: int) -> QPointF:
        """Where the point of a band is: at its frequency and its own gain (for a shelf
        the gain it reaches)."""
        band = self._settings.bands[index]
        return QPointF(self.x_of(band.frequency_hz), self.y_of(band.gain_db))

    def handle_at(self, point: QPointF) -> int | None:
        """The band whose point is nearest to `point` within reach of a finger."""
        best, best_distance = None, float(HANDLE_REACH)
        for index in range(len(self._settings.bands)):
            centre = self.handle_centre(index)
            distance = math.hypot(centre.x() - point.x(), centre.y() - point.y())
            if distance <= best_distance:
                best, best_distance = index, distance
        return best

    def claims_drag(self, position: QPointF) -> bool:
        """For the pager: a finger coming down on a point drags the point, not the
        page (`position` on the screen)."""
        return self._interactive and self.handle_at(self.mapFromGlobal(position)) is not None

    # --- Fingers and mouse ------------------------------------------------------------

    def event(self, event: QEvent) -> bool:
        if self._interactive and event.type() in _TOUCH_EVENTS:
            self._on_touch(event)  # type: ignore[arg-type]
            event.accept()
            return True
        return super().event(event)

    def _on_touch(self, touch: QTouchEvent) -> None:
        kind = touch.type()
        down = [p for p in touch.points() if p.state() != QEventPoint.State.Released]
        if kind == QEvent.Type.TouchBegin:
            self._drag = None
            self._pinch = None
            if len(touch.points()) == 1:
                self.press(self._local(touch.points()[0]))
            else:
                self._pinch_to(down)
        elif kind == QEvent.Type.TouchUpdate:
            if len(down) >= 2:
                self._drag = None
                self._pinch_to(down)
            elif down:
                self._pinch = None
                self.drag_to(self._local(down[0]))
        else:
            self._drag = None
            self._pinch = None

    def _local(self, point: QEventPoint) -> QPointF:
        return self.mapFromGlobal(point.globalPosition())

    def _pinch_to(self, down: list[QEventPoint]) -> None:
        first, second = sorted(down, key=lambda p: p.id())[:2]
        a, b = self._local(first), self._local(second)
        distance = math.hypot(a.x() - b.x(), a.y() - b.y())
        band = self._settings.bands[self._selected]
        if self._pinch is None or self._pinch[0] < 1:
            self._pinch = (distance, band.width_octaves)
            return
        start_distance, start_width = self._pinch
        self.set_width(start_width * distance / start_distance)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._interactive and event.button() == Qt.MouseButton.LeftButton:
            self.press(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self.drag_to(event.position())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self._interactive:
            return
        steps = event.angleDelta().y() / 120
        if steps:
            band = self._settings.bands[self._selected]
            self.set_width(band.width_octaves * 1.1**steps)
        event.accept()

    def press(self, point: QPointF) -> None:
        """A finger or the mouse came down: on a point, that band is chosen and
        dragged."""
        index = self.handle_at(point)
        if index is None:
            return
        if index != self._selected:
            self._selected = index
            self.band_chosen.emit(index)
            self.update()
        self._drag = (index, self.handle_centre(index) - point)

    def drag_to(self, point: QPointF) -> None:
        """The band dragged follows: frequency and gain in their steps."""
        if self._drag is None:
            return
        index, offset = self._drag
        centre = point + offset
        band = self._settings.bands[index]
        moved = replace(
            band,
            frequency_hz=_nearest_frequency(self.frequency_at(centre.x())),
            gain_db=_nearest(GAINS_DB, self.gain_at(centre.y())),
        )
        if moved != band:
            self.band_moved.emit(index, moved)

    def set_width(self, octaves: float) -> None:
        """The width of the chosen band, in its steps; a shelf has none."""
        band = self._settings.bands[self._selected]
        if band.kind != PEAK:
            return
        width = _nearest(WIDTHS_OCTAVES, octaves)
        if width != band.width_octaves:
            self.band_moved.emit(self._selected, replace(band, width_octaves=width))

    # --- Painting ---------------------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:
        theme = self._theme
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.surface))
        painter.drawRoundedRect(QRectF(self.rect()), PANEL_RADIUS, PANEL_RADIUS)
        rect = self.plot_rect()
        self._paint_width(painter, rect)
        self._paint_grid(painter, rect)
        self._paint_spectrum(painter, rect)
        self._paint_curve(painter, rect)
        self._paint_handles(painter)
        self._paint_labels(painter, rect)
        painter.end()

    def _paint_width(self, painter: QPainter, rect: QRectF) -> None:
        band = self._settings.bands[self._selected]
        if band.kind != PEAK:
            return
        low, high = FREQUENCY_RANGE_HZ
        left = self.x_of(max(band.frequency_hz * 2.0 ** (-band.width_octaves / 2), low))
        right = self.x_of(min(band.frequency_hz * 2.0 ** (band.width_octaves / 2), high))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_alpha(self._theme.accent, 18))
        painter.drawRect(QRectF(left, rect.top(), right - left, rect.height()))

    def _paint_grid(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(_alpha(self._theme.outline, 150), 1))
        for hz in _FREQUENCY_TICKS:
            x = self.x_of(hz)
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        for db in (12, 6, -6, -12):
            y = self.y_of(db)
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        painter.setPen(QPen(QColor(self._theme.outline), 2))
        zero = self.y_of(0)
        painter.drawLine(QPointF(rect.left(), zero), QPointF(rect.right(), zero))

    def _paint_spectrum(self, painter: QPainter, rect: QRectF) -> None:
        if self._spectrum is None:
            return
        centres, levels = self._spectrum
        shown = (centres >= FREQUENCY_RANGE_HZ[0]) & (centres <= FREQUENCY_RANGE_HZ[1])
        if not shown.any():
            return
        polygon = QPolygonF()
        first = centres[shown][0]
        polygon.append(QPointF(self.x_of(first), rect.bottom()))
        for hz, level in zip(centres[shown], levels[shown]):
            fraction = min(max((level - SPECTRUM_FLOOR_DB) / -SPECTRUM_FLOOR_DB, 0.0), 1.0)
            polygon.append(QPointF(self.x_of(hz), rect.bottom() - fraction * rect.height()))
        polygon.append(QPointF(self.x_of(centres[shown][-1]), rect.bottom()))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_alpha(self._theme.muted, 40))
        painter.drawPolygon(polygon)

    def _paint_curve(self, painter: QPainter, rect: QRectF) -> None:
        color = self._theme.accent if self._enabled else self._theme.muted
        points = [
            QPointF(self.x_of(hz), self.y_of(min(max(db, GAIN_RANGE_DB[0]), GAIN_RANGE_DB[1])))
            for hz, db in zip(_CURVE_FREQUENCIES, self._curve)
        ]
        path = QPainterPath(points[0])
        for point in points[1:]:
            path.lineTo(point)
        area = QPainterPath(path)
        zero = self.y_of(0)
        area.lineTo(QPointF(points[-1].x(), zero))
        area.lineTo(QPointF(points[0].x(), zero))
        area.closeSubpath()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_alpha(color, 30))
        painter.drawPath(area)
        pen = QPen(QColor(color), 3)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def _paint_handles(self, painter: QPainter) -> None:
        theme = self._theme
        font = QFont(self.font())
        font.setPixelSize(18)
        font.setBold(True)
        painter.setFont(font)
        for index in range(len(self._settings.bands)):
            centre = self.handle_centre(index)
            chosen = index == self._selected
            if not self._interactive:
                radius = 10 if chosen else 6
                painter.setPen(QPen(QColor(theme.text if chosen else theme.surface), 2))
                painter.setBrush(QColor(theme.accent))
                painter.drawEllipse(centre, radius, radius)
                continue
            if chosen:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(_alpha(theme.accent, 56))
                painter.drawEllipse(centre, HANDLE_RADIUS + 16, HANDLE_RADIUS + 16)
                painter.setPen(QPen(QColor(theme.text), 3))
                painter.setBrush(QColor(theme.accent))
            else:
                painter.setPen(QPen(QColor(theme.accent), 3))
                painter.setBrush(QColor(theme.raised))
            painter.drawEllipse(centre, HANDLE_RADIUS, HANDLE_RADIUS)
            painter.setPen(QColor(theme.on_accent if chosen else theme.accent))
            box = QRectF(
                centre.x() - HANDLE_RADIUS,
                centre.y() - HANDLE_RADIUS,
                2 * HANDLE_RADIUS,
                2 * HANDLE_RADIUS,
            )
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, str(index + 1))

    def _paint_labels(self, painter: QPainter, rect: QRectF) -> None:
        theme = self._theme
        font = QFont(self.font())
        font.setPixelSize(AXIS_FONT_PX)
        painter.setFont(font)
        painter.setPen(QColor(theme.muted))
        right_aligned = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        if self._interactive:
            gains = ((12, "+12"), (6, "+6"), (0, "0 dB"), (-6, "−6"), (-12, "−12"))
            ticks = _FREQUENCY_TICKS
        else:
            gains = ((12, "+12"), (0, "0"), (-12, "−12"))
            ticks = _SMALL_TICKS
        for db, text in gains:
            box = QRectF(0, self.y_of(db) - 10, rect.left() - 10, 20)
            painter.drawText(box, right_aligned, text)
        top = rect.bottom() + 4
        for hz in ticks:
            text = f"{hz // 1000}k" if hz >= 1000 else str(hz)
            box = QRectF(self.x_of(hz) - 30, top, 60, 20)
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        painter.drawText(QRectF(rect.right() - 60, top, 60, 20), right_aligned, "Hz")
        if not self._interactive:
            return
        painter.drawText(QRectF(rect.right() - 560, 18, 560, 24), right_aligned, tr(HINT))
        band = self._settings.bands[self._selected]
        figures = [frequency_text(band.frequency_hz), gain_text(band.gain_db)]
        # Translators: a shelving band of the equalizer, which has no width.
        figures.append(width_text(band.width_octaves) if band.kind == PEAK else tr("Shelf"))
        name_font = QFont(self.font())
        name_font.setPixelSize(FONT_PX)
        painter.setFont(name_font)
        painter.setPen(QColor(theme.text))
        name = tr(BAND_NAMES[self._selected])
        left_aligned = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        painter.drawText(QRectF(24, 16, 400, 28), left_aligned, name)
        offset = 24 + QFontMetricsF(name_font).horizontalAdvance(name)
        painter.setPen(QColor(theme.muted))
        painter.drawText(QRectF(offset, 16, 480, 28), left_aligned, " · " + " · ".join(figures))


class _FinePage(QWidget):
    """The chosen band by its name with steppers for its figures and the curve small,
    beside the presets."""

    band_chosen = pyqtSignal(int)
    band_changed = pyqtSignal(int, object)  # index, Band
    preset_chosen = pyqtSignal(str)  # a key of PRESET_NAMES

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = EqualizerSettings()
        self._selected = 0
        self.bands = ChoiceButtons(
            [(index, tr(name)) for index, name in enumerate(BAND_NAMES)], role="chip"
        )
        self.bands.chosen.connect(self._on_band_chosen)
        self.frequency_stepper = Stepper(FREQUENCIES_HZ, frequency_text, value_width=140)
        self.gain_stepper = Stepper(GAINS_DB, gain_text, value_width=140)
        self.width_stepper = Stepper(WIDTHS_OCTAVES, width_text, value_width=140)
        self.frequency_stepper.value_changed.connect(self._on_frequency)
        self.gain_stepper.value_changed.connect(self._on_gain)
        self.width_stepper.value_changed.connect(self._on_width)
        self.kind_label = set_role(QLabel(), "small")
        self.range_label = set_role(QLabel(), "small")
        steppers = QGridLayout()
        steppers.setHorizontalSpacing(16)
        steppers.setVerticalSpacing(12)
        rows = (
            (tr("Frequency"), self.frequency_stepper, self.kind_label),
            (tr("Level"), self.gain_stepper, None),
            (tr("Width"), self.width_stepper, self.range_label),
        )
        for row, (name, stepper, note) in enumerate(rows):
            label = set_role(QLabel(name), "muted")
            label.setFixedWidth(STEPPER_LABEL_WIDTH)
            steppers.addWidget(label, row, 0)
            steppers.addWidget(stepper, row, 1)
            if note is not None:
                steppers.addWidget(note, row, 2)
        steppers.setColumnStretch(2, 1)
        self.curve = CurveView(interactive=False)
        self.curve.setFixedHeight(FINE_CURVE_HEIGHT)

        band_panel = set_role(QWidget(), "panel")
        band_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        band_layout = QVBoxLayout(band_panel)
        band_layout.setContentsMargins(24, 20, 24, 20)
        band_layout.setSpacing(20)
        band_layout.addWidget(self.bands)
        band_layout.addLayout(steppers)
        band_layout.addStretch()
        band_layout.addWidget(self.curve)

        self._presets = QButtonGroup(self)
        self._presets.setExclusive(True)
        self._preset_buttons: dict[str, QPushButton] = {}
        preset_grid = QGridLayout()
        preset_grid.setSpacing(12)
        for index, (key, name) in enumerate(PRESET_NAMES.items()):
            button = set_role(QPushButton(tr(name)), "chip")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, k=key: self.preset_chosen.emit(k))
            self._presets.addButton(button)
            self._preset_buttons[key] = button
            preset_grid.addWidget(button, index // 2, index % 2)
        preset_panel = set_role(QWidget(), "panel")
        preset_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        preset_panel.setFixedWidth(PRESETS_WIDTH)
        preset_layout = QVBoxLayout(preset_panel)
        preset_layout.setContentsMargins(20, 20, 20, 20)
        preset_layout.setSpacing(12)
        preset_layout.addWidget(set_role(QLabel(tr("Presets")), "heading"))
        preset_layout.addLayout(preset_grid)
        preset_layout.addStretch()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        layout.addWidget(band_panel, stretch=1)
        layout.addWidget(preset_panel)

    def preset_button(self, key: str) -> QPushButton:
        return self._preset_buttons[key]

    def show_state(self, state: EqualizerState, selected: int) -> None:
        """The figures of the chosen band and the preset of `state`, without reporting."""
        self._settings = state.settings
        self._selected = selected
        band = state.settings.bands[selected]
        self.bands.set_current(selected)
        self.frequency_stepper.set_value(band.frequency_hz)
        self.gain_stepper.set_value(band.gain_db)
        if band.kind == PEAK:
            self.width_stepper.set_values(WIDTHS_OCTAVES)
            self.width_stepper.set_value(band.width_octaves)
        else:
            self.width_stepper.set_values([])  # a shelf has no width: "–"
        self.kind_label.setText(tr(KIND_NAMES[band.kind]))
        self.range_label.setText(band_range_text(band))
        self.curve.set_settings(state.settings, state.enabled)
        self.curve.set_selected(selected)
        preset = state.settings.preset if state.settings.preset in PRESET_NAMES else CUSTOM
        self._preset_buttons[preset].setChecked(True)
        custom = self._preset_buttons[CUSTOM]
        custom.setEnabled(preset == CUSTOM or state.custom is not None)

    def _band(self) -> Band:
        return self._settings.bands[self._selected]

    def _on_band_chosen(self, index: object) -> None:
        self.band_chosen.emit(int(index))

    def _on_frequency(self, hz: float) -> None:
        self.band_changed.emit(self._selected, replace(self._band(), frequency_hz=hz))

    def _on_gain(self, db: float) -> None:
        self.band_changed.emit(self._selected, replace(self._band(), gain_db=db))

    def _on_width(self, octaves: float | None) -> None:
        if octaves is not None:
            self.band_changed.emit(self._selected, replace(self._band(), width_octaves=octaves))


class EqualizerPage(QWidget):
    """Title, what is set and for whom, the switch and the way back above two pages: the
    curve and the fine adjustment."""

    def __init__(self, controller: ReceiverController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._selected = 0

        title = set_role(QLabel(tr("Sound")), "large")
        self.summary = set_role(ElidedLabel(), "muted")
        self.switch = set_role(QPushButton(tr("Off")), "switch")
        self.switch.setCheckable(True)
        self.switch.clicked.connect(controller.set_equalizer_enabled)
        self.back_button = QPushButton(tr("Back"))
        header = QHBoxLayout()
        header.setSpacing(16)
        header.addWidget(title)
        header.addWidget(self.summary, stretch=1)
        header.addWidget(self.switch)
        header.addWidget(self.back_button)

        self.curve = CurveView(interactive=True)
        self.curve.band_chosen.connect(self._choose_band)
        self.curve.band_moved.connect(self._change_band)
        self.fine = _FinePage()
        self.fine.band_chosen.connect(self._choose_band)
        self.fine.band_changed.connect(self._change_band)
        self.fine.preset_chosen.connect(self._choose_preset)
        self.pager = Pager()
        self.pager.add_page(self.curve)
        self.pager.add_page(self.fine)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)
        layout.addLayout(header)
        layout.addWidget(self.pager, stretch=1)

        controller.equalizer_changed.connect(self._show)
        controller.audio_player.spectrum_ready.connect(self._show_spectrum)
        self._show()

    def apply_theme(self, theme: Theme) -> None:
        self.pager.set_theme(theme)
        self.curve.set_theme(theme)
        self.fine.curve.set_theme(theme)

    def selected_band(self) -> int:
        return self._selected

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        # The spectrum costs an FFT ten times a second: only while it is seen.
        self._controller.set_audio_spectrum_enabled(True)

    def hideEvent(self, event: QHideEvent) -> None:
        super().hideEvent(event)
        self._controller.set_audio_spectrum_enabled(False)
        self.curve.clear_spectrum()
        self.fine.curve.clear_spectrum()

    def _show(self) -> None:
        controller = self._controller
        state = controller.equalizer_state
        self.switch.setChecked(state.enabled)
        self.switch.setText(tr("On") if state.enabled else tr("Off"))
        self.summary.setText(
            summary_text(
                state, controller.equalizer_per_station, controller.equalizer_is_station_own
            )
        )
        self.curve.set_settings(state.settings, state.enabled)
        self.curve.set_selected(self._selected)
        self.fine.show_state(state, self._selected)

    def _show_spectrum(self, centres: np.ndarray, levels: np.ndarray) -> None:
        self.curve.set_spectrum(centres, levels)
        self.fine.curve.set_spectrum(centres, levels)

    def _choose_band(self, index: int) -> None:
        self._selected = index
        self._show()

    def _change_band(self, index: int, band: Band) -> None:
        self._selected = index
        self._controller.set_equalizer(self._controller.equalizer.with_band(index, band))

    def _choose_preset(self, key: str) -> None:
        if key == CUSTOM:
            custom = self._controller.equalizer_state.custom
            if custom is not None:
                self._controller.set_equalizer(custom)
            self._show()  # nothing to bring back: the chips as they were
        else:
            self._controller.set_equalizer(EqualizerSettings.from_preset(key))
