"""A symbol that glows up red and fades again, slowly and in turn: the warning of the
navigation rail that the receiver is overloaded (`ReceiverController.overloaded`)."""

from __future__ import annotations

import math
from collections.abc import Callable

from PyQt6.QtCore import QElapsedTimer, QObject, QSize, Qt, QTimer
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QToolButton

from ui.touch.icons import symbol_pixmap
from ui.touch.theme import Theme

# One fade up and one down, each this long.
FADE_S = 1.0

# The pictures between dark and the full glow; a step of 1/32 in brightness is not seen.
GLOW_LEVELS = 33
FRAME_MS = 33
# The most the glow moves on in one frame: a GUI thread held up for longer (the window
# restyled, `TouchWindow.set_theme`) does not make it jump.
MAX_STEP_S = 0.1

# The glow around the strokes, outermost first: its width in grid units of the symbol
# (`ui.touch.icons`, strokes 2 units wide) and its opacity at the full glow. The widest
# keeps within the grid but for half a unit at the ends of a symbol that reaches its edge
# (the wave of the signal).
HALO = ((5.0, 0.22), (3.5, 0.45))


class Glow:
    """The brightness of the glow (0 dark to 1 full), moved on by the time passed
    (`advance`): up and down in turn, each in `FADE_S`, along a cosine, so that it eases
    into the full glow and into the dark."""

    def __init__(self) -> None:
        self._phase = 0.0  # 0 dark, 1 the full glow, 2 dark again
        self._blinking = False

    @property
    def blinking(self) -> bool:
        return self._blinking

    @property
    def brightness(self) -> float:
        return (1.0 - math.cos(math.pi * self._phase)) / 2.0

    @property
    def glowing(self) -> bool:
        """Blinking, or still fading out."""
        return self._blinking or self._phase > 0.0

    def start(self) -> None:
        """Blinks on from where the glow stands."""
        self._blinking = True

    def stop(self) -> None:
        """Fades out from where the glow stands, at the same pace."""
        self._blinking = False
        if self._phase < 1.0:
            self._phase = 2.0 - self._phase  # as bright, but going down

    def advance(self, seconds: float) -> None:
        if not self.glowing:
            return  # gone out, it stays dark
        self._phase += seconds / FADE_S
        if self._blinking:
            self._phase %= 2.0
        elif self._phase >= 2.0:
            self._phase = 0.0


def _mix(a: QColor, b: QColor, share: float) -> QColor:
    """`a` moved by `share` (0 to 1) towards `b`."""
    return QColor.fromRgbF(
        a.redF() + (b.redF() - a.redF()) * share,
        a.greenF() + (b.greenF() - a.greenF()) * share,
        a.blueF() + (b.blueF() - a.blueF()) * share,
        a.alphaF() + (b.alphaF() - a.alphaF()) * share,
    )


def glow_pixmap(symbol: str, base: QColor, theme: Theme, brightness: float, size: int) -> QPixmap:
    """The symbol in `base`, turned towards the danger colour of the scheme and glowing
    in it as far as `brightness` (0 to 1) says; at 0 exactly `symbol_pixmap`."""
    if brightness <= 0.0:
        return symbol_pixmap(symbol, base, size)
    red = theme.color("danger")
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    # Each layer drawn whole and then laid on with its opacity: strokes overlapping at
    # their joints would otherwise add up.
    for width, opacity in HALO:
        painter.setOpacity(opacity * brightness)
        painter.drawPixmap(0, 0, symbol_pixmap(symbol, red, size, line=width))
    painter.setOpacity(1.0)
    painter.drawPixmap(0, 0, symbol_pixmap(symbol, _mix(base, red, brightness), size))
    painter.end()
    return pixmap


class GlowingSymbol(QObject):
    """Lets the symbol of a button of the navigation rail glow (`set_glowing`), from the
    colour `ui.touch.icons.symbol_icon` draws it in: muted while the button is not
    checked, the accent colour while it is. The pictures are drawn when first needed and
    kept for the scheme; a frame only sets the icon anew when the brightness has moved by
    a step, and not while `seen` says nobody sees the screen."""

    def __init__(
        self,
        button: QToolButton,
        symbol: str,
        size: QSize,
        seen: Callable[[], bool],
    ) -> None:
        super().__init__(button)
        self._button = button
        self._symbol = symbol
        self._size = size
        self._seen = seen
        self._glow = Glow()
        self._theme: Theme | None = None
        self._icons: dict[int, QIcon] = {}
        self._level = 0
        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self._advance)

    @property
    def glow(self) -> Glow:
        return self._glow

    @property
    def level(self) -> int:
        """The picture shown: 0 dark to `GLOW_LEVELS` - 1 the full glow."""
        return self._level

    def set_theme(self, theme: Theme) -> None:
        """Draws in the colours of `theme`; sets the icon of the button, with the glow
        it has right now."""
        self._theme = theme
        self._icons.clear()
        self._show(self._level)

    def set_glowing(self, glowing: bool) -> None:
        """Starts the blinking, or lets the glow fade out."""
        if not glowing:
            self._glow.stop()
            return
        self._glow.start()
        if not self._timer.isActive():
            self._clock.start()
            self._timer.start()

    def _advance(self) -> None:
        seconds = min(self._clock.restart() / 1000.0, MAX_STEP_S)
        self._glow.advance(seconds)
        if not self._glow.glowing:
            self._timer.stop()
            self._show(0)
        elif self._seen():
            level = round(self._glow.brightness * (GLOW_LEVELS - 1))
            if level != self._level:
                self._show(level)

    def _show(self, level: int) -> None:
        self._level = level
        if self._theme is not None:
            self._button.setIcon(self._icon(level))

    def _icon(self, level: int) -> QIcon:
        icon = self._icons.get(level)
        if icon is None:
            theme = self._theme
            brightness = level / (GLOW_LEVELS - 1)
            width = self._size.width()
            icon = QIcon(glow_pixmap(self._symbol, theme.color("muted"), theme, brightness, width))
            icon.addPixmap(
                glow_pixmap(self._symbol, theme.color("accent"), theme, brightness, width),
                QIcon.Mode.Normal,
                QIcon.State.On,
            )
            self._icons[level] = icon
        return icon
