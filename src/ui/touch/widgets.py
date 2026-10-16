"""Controls made for a finger: steppers instead of spin boxes, rows of buttons instead of
drop-down lists, the reception bars and the spinner of the scan; labels of one line, one
running through like a ticker when its text does not fit; fading edges on a row or a page
swiped."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PyQt6.QtCore import (
    QElapsedTimer,
    QEvent,
    QObject,
    QPointF,
    QRectF,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QFont,
    QHideEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPalette,
    QPen,
    QResizeEvent,
    QShowEvent,
)
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QScroller,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from ui.touch.pager import SWIPE_SLOP
from ui.touch.theme import EDGE_FADE_WIDTH, TOUCH_TARGET, Theme

# How long a finger rests on something to do more than a tap (choose what the slide
# area shows, mark a station as favourite), as long as on a phone.
LONG_PRESS_MS = 500

# Holding − or + repeats the step: first after this delay, then at this interval.
REPEAT_DELAY_MS = 400
REPEAT_INTERVAL_MS = 120

# The spinner turns its arc by this angle at this interval: a turn in 1.2 s at 25
# frames per second, smooth enough without costing the Raspberry Pi much.
SPINNER_STEP_DEGREES = 12
SPINNER_STEP_MS = 40
SPINNER_ARC_DEGREES = 100

# A marquee rests at the start of its text, then runs at this speed until the copy that
# follows the text stands where the text began, and rests again. About 30 frames per
# second: smooth, and little work for the Raspberry Pi with a label of one line.
MARQUEE_REST_MS = 2500
MARQUEE_SPEED_PX_PER_S = 60
MARQUEE_FRAME_MS = 33
# Between the end of the text and the copy that follows it.
MARQUEE_GAP_PX = 80
_ONE_LINE_ALIGNMENT = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter


def set_role(widget: QWidget, role: str) -> QWidget:
    """Gives the widget its look from the style sheet (see `ui.touch.theme`)."""
    widget.setProperty("role", role)
    return widget


def repolish(widget: QWidget) -> None:
    """Applies the style sheet again after a dynamic property changed."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class TapWidget(QWidget):
    """Reports a tap on itself (`tapped`), and nothing else: a finger lifted beside it,
    or moved before it was lifted, did not tap. The pager (`ui.touch.pager`) and QScroller
    end a press that turned into a swipe or a scroll with a release far outside."""

    tapped = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pressed_at: QPointF | None = None

    def click(self) -> None:
        """As a tap on the widget."""
        self.tapped.emit()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed_at = event.position()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        pressed_at, self._pressed_at = self._pressed_at, None
        if event.button() == Qt.MouseButton.LeftButton and pressed_at is not None:
            moved = event.position() - pressed_at
            still = abs(moved.x()) < SWIPE_SLOP and abs(moved.y()) < SWIPE_SLOP
            if still and self.rect().contains(event.position().toPoint()):
                self.tapped.emit()
        super().mouseReleaseEvent(event)


class HoldButton(QToolButton):
    """A tool button that also reports a long press (`long_pressed`, after
    `LONG_PRESS_MS` with the finger still on it); a press held that long is no tap, so
    it neither clicks nor toggles when the finger lifts."""

    long_pressed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._held = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(LONG_PRESS_MS)
        self._timer.timeout.connect(self._on_held)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._held = False
            self._timer.start()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._timer.isActive() and not self.rect().contains(event.position().toPoint()):
            self._timer.stop()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._timer.stop()
        if self._held and event.button() == Qt.MouseButton.LeftButton:
            self._held = False
            self.setDown(False)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _on_held(self) -> None:
        if self.isDown():
            self._held = True
            self.long_pressed.emit()


class EdgeFades(QObject):
    """Lets the content of a scroll area fade into the background at either edge where
    more of it lies beyond: left and right for a row swiped sideways, top and bottom
    for a page swiped up and down. A row whose last chip in view ends right at the edge,
    or a page whose last row lies wholly below it, would not show otherwise that there
    is more. The fades (`start` left or at the top, `end` right or at the bottom) are
    drawn by the style sheet (roles "fadeLeft", "fadeRight", "fadeTop", "fadeBottom")
    and let every touch through to what lies beneath them."""

    def __init__(
        self, area: QScrollArea, orientation: Qt.Orientation = Qt.Orientation.Horizontal
    ) -> None:
        super().__init__(area)
        self._area = area
        self._vertical = orientation == Qt.Orientation.Vertical
        self.start = self._fade("fadeTop" if self._vertical else "fadeLeft")
        self.end = self._fade("fadeBottom" if self._vertical else "fadeRight")
        bar = self._bar()
        bar.valueChanged.connect(self._update)
        bar.rangeChanged.connect(self._update)
        area.installEventFilter(self)
        self._update()

    def _bar(self) -> QScrollBar:
        area = self._area
        return area.verticalScrollBar() if self._vertical else area.horizontalScrollBar()

    def _fade(self, role: str) -> QWidget:
        fade = set_role(QWidget(self._area), role)
        fade.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        fade.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        return fade

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self._area and event.type() == QEvent.Type.Resize:
            self._update()
        return False

    def _update(self) -> None:
        """Each fade no deeper than what lies beyond its edge: a page a few pixels too
        tall would otherwise look as if its whole last row were disabled."""
        bar = self._bar()
        before = min(EDGE_FADE_WIDTH, bar.value() - bar.minimum())
        beyond = min(EDGE_FADE_WIDTH, bar.maximum() - bar.value())
        width, height = self._area.width(), self._area.height()
        if self._vertical:
            self.start.setGeometry(0, 0, width, before)
            self.end.setGeometry(0, height - beyond, width, beyond)
        else:
            self.start.setGeometry(0, 0, before, height)
            self.end.setGeometry(width - beyond, 0, beyond, height)
        self.start.setVisible(before > 0)
        self.end.setVisible(beyond > 0)
        # Above the viewport, a sibling of theirs.
        self.start.raise_()
        self.end.raise_()


# The row of chips above a page: a little higher than the chips, so that the fades of
# `ChipRow` cover them whole.
CHIP_ROW_HEIGHT = 72


class ChipRow(QScrollArea):
    """A row of chips (`ChoiceButtons`, or a widget with a row of buttons) swiped
    sideways where it is wider than the page: the row keeps its natural width (`fit`),
    however many chips there are. A resizable row would be squeezed to the page
    instead: the style sheet gives every button a minimum width, and the layout shrinks
    the chips down to it, cutting their names (seen with the eighth chip of the
    settings, 10.10.2026). `fades` show where more lies beyond either edge."""

    def __init__(self, row: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._row = row
        set_role(self, "transparent")
        set_role(self.viewport(), "transparent")
        self.setWidget(row)
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.setFixedHeight(CHIP_ROW_HEIGHT)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Scrolls by swiping, as a finger expects; taps still reach the buttons.
        QScroller.grabGesture(self.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture)
        self.fades = EdgeFades(self)
        self.fit()

    def fit(self) -> None:
        """Every chip at its full width, the row as wide as they are together; to be
        called after the chips changed."""
        self._row.adjustSize()

    def showEvent(self, event: QEvent) -> None:
        super().showEvent(event)
        self.fit()  # the paddings of the style sheet apply once polished


class Stepper(QWidget):
    """− value +: steps through a list of values, holding a button repeats.

    Spin boxes need a keyboard or tiny arrows; on a touch screen two big buttons and
    the value between them are the usual way."""

    value_changed = pyqtSignal(object)

    def __init__(
        self,
        values: Sequence[object],
        formatter: Callable[[object], str] = str,
        value_width: int = 160,
        parent: QWidget | None = None,
        symbols: tuple[str, str] = ("−", "+"),
    ) -> None:
        super().__init__(parent)
        self._values = list(values)
        self._formatter = formatter
        self._index = 0
        self._minus = self._make_button(symbols[0], -1)
        self._plus = self._make_button(symbols[1], 1)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setFixedWidth(value_width)
        set_role(self._label, "large")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._minus)
        layout.addWidget(self._label)
        layout.addWidget(self._plus)
        self._show()

    def _make_button(self, text: str, step: int) -> QPushButton:
        button = set_role(QPushButton(text), "step")
        button.setFixedSize(TOUCH_TARGET, TOUCH_TARGET)
        button.setAutoRepeat(True)
        button.setAutoRepeatDelay(REPEAT_DELAY_MS)
        button.setAutoRepeatInterval(REPEAT_INTERVAL_MS)
        button.clicked.connect(lambda: self._step(step))
        return button

    def value(self) -> object:
        """The value shown; None while there are no values."""
        return self._values[self._index] if self._values else None

    def set_values(self, values: Sequence[object]) -> None:
        """Steps through these values from now on, keeping the one shown if it is among
        them (else the first); without reporting it."""
        current = self.value()
        self._values = list(values)
        self._index = self._values.index(current) if current in self._values else 0
        self._show()

    def set_value(self, value: object) -> None:
        """Shows `value`, or the nearest one for numbers that are not in the list;
        without reporting it."""
        if value in self._values:
            self._index = self._values.index(value)
        else:
            self._index = min(range(len(self._values)), key=lambda i: abs(self._values[i] - value))
        self._show()

    def _step(self, step: int) -> None:
        index = min(max(self._index + step, 0), len(self._values) - 1)
        if index == self._index:
            return
        self._index = index
        self._show()
        self.value_changed.emit(self.value())

    def _show(self) -> None:
        self._label.setText(self._formatter(self.value()) if self._values else "–")
        self._minus.setEnabled(self._index > 0)
        self._plus.setEnabled(self._index < len(self._values) - 1)


class ChoiceButtons(QWidget):
    """A row of buttons of which exactly one is chosen, in place of a drop-down list."""

    chosen = pyqtSignal(object)

    def __init__(
        self,
        choices: Sequence[tuple[object, str]],
        parent: QWidget | None = None,
        role: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._keys: list[object] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        for key, text in choices:
            button = QPushButton(text)
            if role is not None:
                set_role(button, role)  # e.g. "chip"
            button.setCheckable(True)
            self._group.addButton(button, len(self._keys))
            self._keys.append(key)
            layout.addWidget(button)
        layout.addStretch()
        self._group.idClicked.connect(lambda index: self.chosen.emit(self._keys[index]))

    def current(self) -> object:
        return self._keys[self._group.checkedId()]

    def set_current(self, key: object) -> None:
        """Marks `key` as chosen without reporting it."""
        self._group.button(self._keys.index(key)).setChecked(True)


class SignalBars(QWidget):
    """Four bars rising to the right, as many filled as the reception is good (0-4);
    None draws them all empty (nothing to judge yet)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bars: int | None = None
        self._on = Qt.GlobalColor.white
        self._off = Qt.GlobalColor.gray
        self.setFixedSize(QSize(40, 32))

    def set_theme(self, theme: Theme) -> None:
        self._on = theme.color("accent")
        self._off = theme.color("outline")
        self.update()

    def set_bars(self, bars: int | None) -> None:
        self._bars = bars
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        width = self.width() / 4
        for index in range(4):
            height = self.height() * (index + 1) / 4
            painter.setBrush(self._on if (self._bars or 0) > index else self._off)
            painter.drawRoundedRect(
                QRectF(index * width + 2, self.height() - height, width - 4, height), 2, 2
            )
        painter.end()


class Spinner(QWidget):
    """A ring with a turning arc, for work that takes a while (the scan), on a disc of
    the panel colour so that it stands out from what lies beneath. In its middle a
    short title (e.g. "12/38") and a line beneath it. Taps go through to what is below."""

    def __init__(self, diameter: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(diameter, diameter)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._angle = 0
        self._title = ""
        self._detail = ""
        self._disc = QColor(Qt.GlobalColor.darkGray)
        self._track = QColor(Qt.GlobalColor.gray)
        self._arc = QColor(Qt.GlobalColor.cyan)
        self._text = QColor(Qt.GlobalColor.white)
        self._muted = QColor(Qt.GlobalColor.lightGray)
        self._timer = QTimer(self)
        self._timer.setInterval(SPINNER_STEP_MS)
        self._timer.timeout.connect(self._turn)
        self.hide()

    def set_theme(self, theme: Theme) -> None:
        self._disc = theme.color("surface")
        self._track = theme.color("outline")
        self._arc = theme.color("accent")
        self._text = theme.color("text")
        self._muted = theme.color("muted")
        self.update()

    def set_text(self, title: str, detail: str = "") -> None:
        self._title = title
        self._detail = detail
        self.update()

    def start(self) -> None:
        self.show()
        self.raise_()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    @property
    def running(self) -> bool:
        return self._timer.isActive()

    def _turn(self) -> None:
        self._angle = (self._angle + SPINNER_STEP_DEGREES) % 360
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        size = self.width()
        line = max(size // 18, 4)
        ring = QRectF(line * 1.5, line * 1.5, size - 3 * line, size - 3 * line)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._disc)
        painter.drawEllipse(QRectF(0, 0, size, size))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(self._track, line)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawEllipse(ring)
        pen.setColor(self._arc)
        painter.setPen(pen)
        # Qt counts angles counter-clockwise in sixteenths of a degree: a falling start
        # angle turns the arc clockwise.
        painter.drawArc(ring, -self._angle * 16, SPINNER_ARC_DEGREES * 16)
        title_font = QFont()
        title_font.setPixelSize(round(size * 0.17))
        title_font.setBold(True)
        detail_font = QFont()
        detail_font.setPixelSize(round(size * 0.08))
        middle = size / 2
        painter.setFont(title_font)
        painter.setPen(self._text)
        painter.drawText(
            QRectF(0, middle - size * 0.17, size, size * 0.22),
            Qt.AlignmentFlag.AlignCenter,
            self._title,
        )
        painter.setFont(detail_font)
        painter.setPen(self._muted)
        painter.drawText(
            QRectF(0, middle + size * 0.07, size, size * 0.12),
            Qt.AlignmentFlag.AlignCenter,
            self._detail,
        )
        painter.end()


def one_line(text: str) -> str:
    """The text on one line: line breaks (and the end of a headline in a dynamic label)
    become spaces, a preferred word break (0x1F, ETSI EN 300 401, clause 7.4.5.2) goes."""
    return " ".join(text.replace("", "").split())


def marquee_offset(elapsed_ms: float, distance: float) -> float:
    """How far the text of a marquee has run `elapsed_ms` after it was set: it rests,
    runs `distance` pixels (the text and the gap) and starts again from rest, the copy
    that followed it now in its place."""
    run_ms = distance * 1000 / MARQUEE_SPEED_PX_PER_S
    phase = elapsed_ms % (MARQUEE_REST_MS + run_ms)
    return max(phase - MARQUEE_REST_MS, 0.0) * MARQUEE_SPEED_PX_PER_S / 1000


class _OneLineLabel(QLabel):
    """A label of one line that paints itself: as high as its font, and as narrow as the
    layout makes it, however long the text. Font and colour come from the style sheet
    like those of any label."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    def setText(self, text: str) -> None:
        super().setText(one_line(text))
        self.update()

    def text_width(self) -> int:
        return self.fontMetrics().horizontalAdvance(self.text())

    def sizeHint(self) -> QSize:
        margins = self.contentsMargins()
        return QSize(
            self.text_width() + margins.left() + margins.right(),
            self.fontMetrics().height() + margins.top() + margins.bottom(),
        )

    def minimumSizeHint(self) -> QSize:
        return QSize(0, self.sizeHint().height())

    def _painter(self) -> QPainter:
        painter = QPainter(self)
        painter.setFont(self.font())
        painter.setPen(self.palette().color(QPalette.ColorRole.WindowText))
        painter.setClipRect(self.contentsRect())
        return painter


class ElidedLabel(_OneLineLabel):
    """A label of one line that ends in "…" where its text does not fit."""

    def paintEvent(self, event: QPaintEvent) -> None:
        rect = self.contentsRect()
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, rect.width())
        painter = self._painter()
        painter.drawText(rect, _ONE_LINE_ALIGNMENT, text)
        painter.end()


class Marquee(_OneLineLabel):
    """A label of one line whose text, where it does not fit, runs through from right to
    left like a ticker, resting at its start each time round. Text that fits stands
    still, and the label only animates while it is on screen."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._since_set = QElapsedTimer()
        self._since_set.start()
        self._timer = QTimer(self)
        self._timer.setInterval(MARQUEE_FRAME_MS)
        self._timer.timeout.connect(self.update)

    def setText(self, text: str) -> None:
        if one_line(text) == self.text():
            return  # stations repeat their label: it runs on undisturbed
        super().setText(text)
        self._since_set.restart()
        self._run_if_needed()

    @property
    def running(self) -> bool:
        return self._timer.isActive()

    def overflows(self) -> bool:
        return self.text_width() > self.contentsRect().width()

    def _run_if_needed(self) -> None:
        if self.isVisible() and self.overflows():
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._run_if_needed()

    def hideEvent(self, event: QHideEvent) -> None:
        super().hideEvent(event)
        self._run_if_needed()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._run_if_needed()

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self.updateGeometry()
            self._run_if_needed()

    def paintEvent(self, event: QPaintEvent) -> None:
        rect = self.contentsRect()
        width = self.text_width()
        painter = self._painter()
        if width <= rect.width():
            painter.drawText(rect, _ONE_LINE_ALIGNMENT, self.text())
        else:
            distance = width + MARQUEE_GAP_PX
            left = rect.left() - marquee_offset(self._since_set.elapsed(), distance)
            for x in (left, left + distance):
                painter.drawText(
                    QRectF(x, rect.top(), width, rect.height()), _ONE_LINE_ALIGNMENT, self.text()
                )
        painter.end()
