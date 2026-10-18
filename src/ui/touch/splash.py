"""The picture of the radio starting on the touch display: the constellation of DAB's
π/4-DQPSK locked into its four points, the name beneath it.

It is the last picture of the boot splash as well (`pi/plymouth/dab-radio`, installed by
`pi/install.sh --splash`), whose script animates it: the dots run round a ring at first,
as the constellation does while the receiver is off in frequency, then slow down and
gather in the four points. On a Raspberry Pi 5 the display comes up about 5 s after
power on (its driver loads from the root file system), the radio's unit starts about 3 s
later and ends Plymouth, which has to let go of the display for cage, and the radio
needs 2 to 4 s more until its window shows. `StartupSplash` covers that time:
`main.py` shows it as soon as Qt is up, before it imports the rest of the radio, and
closes it once the radio's window has come up over it.

`tools/build_boot_splash.py` writes the images of the boot splash and the table of its
script (`with_plymouth_table`) from this module, so both draw the same dots; a fixed seed
makes them. Only PyQt6 is imported here, for the picture to come quickly: the colours of
the dark theme are repeated rather than taken from `ui.touch.theme`, which brings
pyqtgraph and numpy along (a test holds them equal).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from PyQt6.QtCore import QEvent, QEventLoop, QObject, QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QFontMetricsF,
    QImage,
    QLinearGradient,
    QPainter,
    QPaintEvent,
    QRadialGradient,
)
from PyQt6.QtWidgets import QWidget

# The picture is laid out for the 1280 × 720 of the touch display; another screen gets
# it scaled by the smaller of the two ratios, in the middle.
WIDTH = 1280
HEIGHT = 720
# The centre of the constellation, and how far each of its four points lies from it
# along either axis.
CENTRE_Y = 284
POINT_OFFSET = 112
# Half the length of either axis through the centre.
AXIS_LENGTH = 240
DOTS_PER_POINT = 24
# How far a dot lies from its point once locked (standard deviation, px), and off the
# ring before the lock (relative to its distance from the centre).
SCATTER = 11.0
RING_NOISE = 0.1
# Fixes the dots, so that the boot splash and the radio draw the same ones: the number
# of the DAB standard, EN 300 401.
SEED = 300401
# The image of a dot: a core with a glow about it.
DOT_SIZE = 16
DOT_CORE_RADIUS = 3.0
# The name, its first word in the accent colour.
TITLE = ("DAB+", " Radio")
TITLE_PX = 64
TITLE_CENTRE_Y = 572

# The colours of the dark theme of the touch surface (`ui.touch.theme.THEMES`).
BACKGROUND = "#101418"
ACCENT = "#4fc3f7"
TEXT = "#e8edf2"
MUTED = "#93a1af"
# The typeface of the touch surface (`ui.touch.theme.FONT_FILES`), both of its files:
# `ui.touch.theme.touch_font` registers none once Qt knows the family, and the surface
# would come out all bold.
FONT_FAMILY = "Source Sans 3"
FONT_FILES = tuple(
    Path(__file__).parent.parent / "assets" / "fonts" / name
    for name in ("SourceSans3-Regular.ttf", "SourceSans3-Bold.ttf")
)

# The images of the boot splash, by the names its script loads them by.
DOT_IMAGE = "dot.png"
AXES_IMAGE = "axes.png"
TITLE_IMAGE = "title.png"
# The marks in the boot splash's script between which `with_plymouth_table` writes.
TABLE_BEGIN = "# --- begin of the part written by tools/build_boot_splash.py ---"
TABLE_END = "# --- end of the part written by tools/build_boot_splash.py ---"

# Longest wait for the picture to be drawn before the radio goes on regardless.
SHOW_TIMEOUT_MS = 1000
# After the picture is drawn, the event loop waits this long before the radio goes on:
# Qt hands the picture to the compositor when its loop next waits, and the imports of
# the radio follow without one.
HANDOVER_MS = 50
# The picture stays this long after the radio's window has come up over it, for the
# window's first drawing: where no window is drawn, cage shows its empty background.
CLOSE_DELAY_MS = 300
# Closed regardless after this, should the radio's window never come up.
CLOSE_TIMEOUT_MS = 15000


@dataclass(frozen=True)
class Dot:
    """A dot of the constellation, in polar coordinates about its centre (screen
    coordinates, y downwards; angles in radians)."""

    radius: float  # locked
    angle: float  # locked
    spread: float  # how far round the ring from `angle` it runs before the lock
    noise: float  # how far off the ring then, relative to `radius`
    phase: float  # of its slow drift about its place once locked


@cache
def constellation() -> tuple[Dot, ...]:
    """The dots, `DOTS_PER_POINT` about each of the four points on the diagonals. So few
    drawn at random clump and stray; laid out instead on a sunflower spiral whose radii
    follow the normal distribution (the inverse of its radial distribution, Rayleigh's),
    each point gets an even, round cloud, and the dots' places round the ring before the
    lock are spread as evenly over the quarter of it each point fills."""
    rng = random.Random(SEED)
    ring = POINT_OFFSET * math.sqrt(2)
    golden_angle = math.pi * (3 - math.sqrt(5))
    dots = []
    for point in range(4):
        theta = math.pi / 4 + point * math.pi / 2
        turn = rng.uniform(0, 2 * math.pi)
        spreads = [(k + rng.random()) / DOTS_PER_POINT - 0.5 for k in range(DOTS_PER_POINT)]
        rng.shuffle(spreads)
        for k in range(DOTS_PER_POINT):
            distance = SCATTER * math.sqrt(-2 * math.log(1 - (k + 0.5) / DOTS_PER_POINT))
            direction = turn + k * golden_angle
            x = ring * math.cos(theta) + distance * math.cos(direction)
            y = ring * math.sin(theta) + distance * math.sin(direction)
            dots.append(
                Dot(
                    radius=math.hypot(x, y),
                    angle=math.atan2(y, x),
                    spread=spreads[k] * math.pi / 2,
                    noise=rng.gauss(0, RING_NOISE),
                    phase=rng.uniform(0, 2 * math.pi),
                )
            )
    return tuple(dots)


def _transparent(width: int, height: int) -> QImage:
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    return image


def _with_alpha(colour: str, alpha: float) -> QColor:
    result = QColor(colour)
    result.setAlphaF(alpha)
    return result


def dot_image() -> QImage:
    """A dot: a core in the accent colour, a glow about it fading out to the edge."""
    image = _transparent(DOT_SIZE, DOT_SIZE)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    centre = QPointF(DOT_SIZE / 2, DOT_SIZE / 2)
    core = DOT_CORE_RADIUS / (DOT_SIZE / 2)
    glow = QRadialGradient(centre, DOT_SIZE / 2)
    glow.setColorAt(0, QColor(ACCENT))
    glow.setColorAt(core, QColor(ACCENT))
    glow.setColorAt(core + 0.08, _with_alpha(ACCENT, 0.4))
    glow.setColorAt(1, _with_alpha(ACCENT, 0))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(glow)
    painter.drawEllipse(centre, DOT_SIZE / 2, DOT_SIZE / 2)
    painter.end()
    return image


def axes_image() -> QImage:
    """The two axes through the centre, which lies at (`AXIS_LENGTH`, `AXIS_LENGTH`);
    each fades out towards its ends."""
    size = 2 * AXIS_LENGTH
    image = _transparent(size, size)
    painter = QPainter(image)
    for horizontal in (True, False):
        if horizontal:
            line = QRectF(0, AXIS_LENGTH - 1, size, 2)
            fade = QLinearGradient(0, 0, size, 0)
        else:
            line = QRectF(AXIS_LENGTH - 1, 0, 2, size)
            fade = QLinearGradient(0, 0, 0, size)
        fade.setColorAt(0, _with_alpha(MUTED, 0))
        fade.setColorAt(0.35, _with_alpha(MUTED, 0.3))
        fade.setColorAt(0.65, _with_alpha(MUTED, 0.3))
        fade.setColorAt(1, _with_alpha(MUTED, 0))
        painter.fillRect(line, fade)
    painter.end()
    return image


def title_font() -> QFont:
    """Source Sans 3 Bold at `TITLE_PX`; registers the typeface with Qt where its
    database lacks it (see `ui.touch.theme.touch_font`)."""
    if FONT_FAMILY not in QFontDatabase.families():
        for path in FONT_FILES:
            QFontDatabase.addApplicationFont(str(path))
    font = QFont(FONT_FAMILY)
    font.setPixelSize(TITLE_PX)
    font.setWeight(QFont.Weight.Bold)
    return font


def title_image() -> QImage:
    """The name, as wide and high as the font's advance and line, with a pixel to
    spare all round."""
    font = title_font()
    metrics = QFontMetricsF(font)
    first, rest = TITLE
    width = math.ceil(metrics.horizontalAdvance(first + rest)) + 2
    height = math.ceil(metrics.height()) + 2
    image = _transparent(width, height)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    painter.setFont(font)
    baseline = 1 + metrics.ascent()
    painter.setPen(QColor(ACCENT))
    painter.drawText(QPointF(1, baseline), first)
    painter.setPen(QColor(TEXT))
    painter.drawText(QPointF(1 + metrics.horizontalAdvance(first), baseline), rest)
    painter.end()
    return image


@cache
def splash_images() -> dict[str, QImage]:
    """The images of the picture by the names of the boot splash's files."""
    return {DOT_IMAGE: dot_image(), AXES_IMAGE: axes_image(), TITLE_IMAGE: title_image()}


def paint_splash(painter: QPainter, rect: QRectF) -> None:
    """Draws the picture on `rect`: the background all over, the rest scaled to fit."""
    painter.fillRect(rect, QColor(BACKGROUND))
    scale = min(rect.width() / WIDTH, rect.height() / HEIGHT)
    images = splash_images()
    painter.save()
    painter.translate(rect.center().x() - WIDTH * scale / 2, rect.center().y() - HEIGHT * scale / 2)
    painter.scale(scale, scale)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, scale != 1)
    centre_x = WIDTH / 2
    painter.drawImage(QPointF(centre_x - AXIS_LENGTH, CENTRE_Y - AXIS_LENGTH), images[AXES_IMAGE])
    dot = images[DOT_IMAGE]
    for each in constellation():
        x = centre_x + each.radius * math.cos(each.angle) - DOT_SIZE / 2
        y = CENTRE_Y + each.radius * math.sin(each.angle) - DOT_SIZE / 2
        painter.drawImage(QPointF(x, y), dot)
    title = images[TITLE_IMAGE]
    painter.drawImage(
        QPointF(centre_x - title.width() / 2, TITLE_CENTRE_Y - title.height() / 2), title
    )
    painter.restore()


def plymouth_table() -> str:
    """The layout and the dots in the script language of Plymouth, for the boot
    splash's script, which animates them."""
    background = QColor(BACKGROUND)
    lines = [
        f"design.width = {WIDTH};",
        f"design.height = {HEIGHT};",
        f"design.centre_y = {CENTRE_Y};",
        f"design.axis_length = {AXIS_LENGTH};",
        f"design.dot_size = {DOT_SIZE};",
        f"design.title_centre_y = {TITLE_CENTRE_Y};",
        f"design.background_red = {background.redF():.4f};",
        f"design.background_green = {background.greenF():.4f};",
        f"design.background_blue = {background.blueF():.4f};",
        f"dots.count = {len(constellation())};",
    ]
    for i, dot in enumerate(constellation()):
        lines.append(
            f"dots.radius[{i}] = {dot.radius:.3f}; dots.angle[{i}] = {dot.angle:.5f}; "
            f"dots.spread[{i}] = {dot.spread:.5f}; dots.noise[{i}] = {dot.noise:.5f}; "
            f"dots.phase[{i}] = {dot.phase:.5f};"
        )
    return "\n".join(lines) + "\n"


def with_plymouth_table(script: str) -> str:
    """`script`, the boot splash's, with `plymouth_table` between its marks."""
    begin = script.find(TABLE_BEGIN)
    end = script.find(TABLE_END)
    if begin < 0 or end < begin:
        raise ValueError("the script lacks the marks of the table")
    head = script[: begin + len(TABLE_BEGIN)]
    return f"{head}\n{plymouth_table()}{script[end:]}"


class StartupSplash(QWidget):
    """The picture as a window of its own, full screen while the radio starts. The
    radio's window comes up over it, as cage shows the window mapped last."""

    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._drawn = False
        self._handover = QTimer(self)
        self._handover.setSingleShot(True)
        self._close_timer = QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.timeout.connect(self.close)

    def show_now(self) -> bool:
        """Shows the picture full screen and returns once it is on the screen, or after
        `SHOW_TIMEOUT_MS`; whether it was drawn. Under Wayland a window is drawn only
        once the compositor has answered, which takes a turn of the event loop."""
        loop = QEventLoop()
        self._handover.timeout.connect(loop.quit)
        self._handover.start(SHOW_TIMEOUT_MS)
        self.showFullScreen()
        loop.exec()
        self._handover.timeout.disconnect(loop.quit)
        return self._drawn

    def close_over(self, window: QWidget) -> None:
        """Closes the picture once `window`, shown, has come up over it."""
        handle = window.windowHandle()
        if handle is None or handle.isExposed():
            self._close_timer.start(CLOSE_DELAY_MS)
            return
        handle.installEventFilter(self)
        self._close_timer.start(CLOSE_TIMEOUT_MS)

    def eventFilter(self, watched: QObject | None, event: QEvent | None) -> bool:
        if (
            event is not None
            and event.type() == QEvent.Type.Expose
            and getattr(watched, "isExposed", lambda: False)()
        ):
            watched.removeEventFilter(self)
            self._close_timer.start(CLOSE_DELAY_MS)
        return False

    def paintEvent(self, event: QPaintEvent | None) -> None:
        painter = QPainter(self)
        paint_splash(painter, QRectF(self.rect()))
        painter.end()
        if not self._drawn:
            self._drawn = True
            if self._handover.isActive():
                self._handover.start(HANDOVER_MS)
