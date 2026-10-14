"""What the radio page shows in the place of the slideshow (`ui.touch.radio_page`): the
slideshow, one of the panels of the signal page (`ui.touch.signal_panels`) that has
the shape of a tile there, or the view of a data service. A long press on the slide
opens `DisplayChoicePage`, where every display is a tile with a stylised picture of it;
a tap chooses one. The tiles run on downwards, a finger swipes them up.

A data service's display is only offered while the channel received carries the service
(its user application in FIG 0/13), as the service's menu on the services page; its tile
cannot be tapped otherwise. Chosen before, it gives way to the slideshow while the
service is missing, and comes back with it.

The pictures are drawn, not taken from the live plots: they cost no compute time, and
they show what a display looks like before anything has been received.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QScroller,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from dab.data_services import USER_APPLICATION_GEPOS, DataComponent
from i18n import tr, tr_noop
from ui.globe_gl import GlobeRenderThread
from ui.gnss_model import GnssModel
from ui.offscreen_gl import RenderThread
from ui.receiver_controller import ReceiverController
from ui.touch.gnss_view import GlobePanel
from ui.touch.null_symbol_view import NullSymbolPanel
from ui.touch.signal_panels import (
    ConstellationPanel,
    FrameTally,
    ImpulsePanel,
    LevelPanel,
    OffsetPanel,
    ReceptionFiguresPanel,
    SignalPanel,
    SoundFiguresPanel,
    SpectrumPanel,
)
from ui.touch.theme import DARK, THEMES, Theme
from ui.touch.widgets import repolish, set_role

# Key of the setting holding the display chosen.
SETTINGS_RADIO_DISPLAY = "touch/radio_display"

SLIDESHOW = "slideshow"
SPECTRUM = "spectrum"
CONSTELLATION = "constellation"
IMPULSE = "impulse"
OFFSET = "offset"
LEVEL = "level"
RECEPTION_FIGURES = "reception"
SOUND_FIGURES = "sound"
GLOBE = "globe"
NULL_SYMBOL = "null_symbol"


@dataclass(frozen=True)
class Display:
    key: str
    name: str  # on its tile, and above the figures on the radio page (marked, `tr`)
    # The user application (FIG 0/13) of the data service it shows; None: always offered.
    app_type: int | None = None


DISPLAYS = (
    Display(SLIDESHOW, tr_noop("Slideshow")),
    Display(SPECTRUM, tr_noop("Spectrum")),
    Display(CONSTELLATION, tr_noop("Constellation")),
    Display(IMPULSE, tr_noop("Impulse response")),
    Display(OFFSET, tr_noop("Frequency offset")),
    Display(LEVEL, tr_noop("Audio level")),
    Display(RECEPTION_FIGURES, tr_noop("Reception figures")),
    Display(SOUND_FIGURES, tr_noop("Audio figures")),
    Display(NULL_SYMBOL, tr_noop("Null symbol 3D")),
    Display(GLOBE, tr_noop("GNSS 3D view"), USER_APPLICATION_GEPOS),
)
DISPLAY_KEYS = tuple(display.key for display in DISPLAYS)

# Four tiles side by side fill the page (`ui.touch.theme.DISPLAY_TILE_HEIGHT`); the
# picture in one has the shape of the slide (4:3).
COLUMNS = 4
PREVIEW_SIZE = QSize(224, 168)

_TITLE = tr_noop("Display on the radio page")


def offered_displays(components: Iterable[DataComponent]) -> set[str]:
    """The keys of the displays offered for an ensemble with these data components."""
    carried = {app_type for component in components for app_type in component.app_types}
    return {
        display.key
        for display in DISPLAYS
        if display.app_type is None or display.app_type in carried
    }


def make_panel(
    key: str,
    controller: ReceiverController,
    tally: FrameTally,
    gnss_model: GnssModel,
    globe_renderer: GlobeRenderThread | None = None,
    waterfall_renderer: RenderThread | None = None,
) -> SignalPanel:
    """A panel of its own for the radio page; not for the slideshow. The 3D view draws
    with `globe_renderer`, which it may share with that of the services page, and keeps
    its distance in the settings; the null symbol with `waterfall_renderer`, which it
    may share with the detail view of the signal page."""
    if key == GLOBE:
        return GlobePanel(gnss_model, globe_renderer, controller.settings)
    if key == NULL_SYMBOL:
        return NullSymbolPanel(controller, waterfall_renderer)
    if key == SPECTRUM:
        return SpectrumPanel(controller)
    if key == CONSTELLATION:
        return ConstellationPanel(controller)
    if key == IMPULSE:
        return ImpulsePanel(controller)
    if key == OFFSET:
        return OffsetPanel(controller)
    if key == LEVEL:
        return LevelPanel(controller)
    # The width of the slide is fixed: a value too long for it ends in "…".
    if key == RECEPTION_FIGURES:
        panel = ReceptionFiguresPanel(controller, tally, tr("Reception figures"))
    elif key == SOUND_FIGURES:
        panel = SoundFiguresPanel(controller, tally, tr("Audio figures"))
    else:
        raise ValueError(f"no panel for display {key!r}")
    panel.figures.spread()
    return panel


# --- The pictures -----------------------------------------------------------------------
# Drawn on a grid of 120 x 90 units, the shape of the slide, and scaled to the size asked
# for. The noise is drawn from fixed seeds, so that a picture is the same every time.

_GRID_WIDTH = 120.0
_GRID_HEIGHT = 90.0
_LEFT, _RIGHT, _TOP, _BOTTOM = 14.0, 110.0, 10.0, 80.0  # the axes of a plot
_CURVE = 1.4  # width of a curve, in grid units


def _noise(seed: int, count: int, scale: float) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, scale, count)


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    return np.convolve(values, np.ones(width) / width, mode="same")


def _polyline(p: QPainter, xs: np.ndarray, ys: np.ndarray) -> None:
    p.drawPolyline(QPolygonF([QPointF(float(x), float(y)) for x, y in zip(xs, ys)]))


def _axes(p: QPainter, theme: Theme) -> None:
    p.setPen(QPen(theme.color("outline"), 0.8))
    p.drawLine(QPointF(_LEFT, _TOP), QPointF(_LEFT, _BOTTOM))
    p.drawLine(QPointF(_LEFT, _BOTTOM), QPointF(_RIGHT, _BOTTOM))


def _curve_pen(p: QPainter, theme: Theme, name: str = "accent") -> None:
    pen = QPen(theme.color(name), _CURVE)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)


def _slideshow(p: QPainter, theme: Theme) -> None:
    """A picture: a frame with mountains and the sun."""
    frame = QRectF(22, 15, 76, 60)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(theme.color("raised"))
    p.drawRoundedRect(frame, 4, 4)
    p.save()
    p.setClipRect(frame)
    p.setBrush(theme.color("accent"))
    p.drawPolygon(
        QPolygonF([QPointF(x, y) for x, y in ((18, 76), (46, 40), (62, 58), (74, 46), (102, 76))])
    )
    p.restore()
    p.setBrush(theme.color("notice"))
    p.drawEllipse(QPointF(82, 29), 6, 6)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(theme.color("muted"), 2.0))
    p.drawRoundedRect(frame, 4, 4)


def _spectrum(p: QPainter, theme: Theme) -> None:
    """The ensemble standing as a plateau above the noise floor."""
    _axes(p, theme)
    xs = np.linspace(_LEFT + 3, _RIGHT - 2, 120)
    inside = (xs > 32) & (xs < 92)
    ys = np.where(inside, 28.0, 68.0) + _noise(1, len(xs), 1.0)
    _curve_pen(p, theme)
    _polyline(p, xs, ys)


def _constellation(p: QPainter, theme: Theme) -> None:
    """Four clouds of points, one per phase step of D-QPSK."""
    p.setPen(QPen(theme.color("outline"), 0.8))
    p.drawLine(QPointF(24, 45), QPointF(96, 45))
    p.drawLine(QPointF(60, 9), QPointF(60, 81))
    rng = np.random.default_rng(2)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(theme.color("accent"))
    for cx, cy in ((40, 25), (80, 25), (40, 65), (80, 65)):
        for dx, dy in rng.normal(0.0, 3.6, (40, 2)):
            p.drawEllipse(QPointF(cx + dx, cy + dy), 1.1, 1.1)


def _impulse(p: QPainter, theme: Theme) -> None:
    """Two paths in the guard interval, shaded, above the noise."""
    guard = theme.color("accent")
    guard.setAlpha(45)
    p.fillRect(QRectF(28, _TOP, 44, _BOTTOM - _TOP), guard)
    _axes(p, theme)
    xs = np.linspace(_LEFT + 3, _RIGHT - 2, 120)
    ys = 70.0 + _noise(3, len(xs), 1.4)
    ys[np.argmin(np.abs(xs - 28))] = 16.0
    ys[np.argmin(np.abs(xs - 44))] = 36.0
    _curve_pen(p, theme)
    _polyline(p, xs, ys)


def _offset(p: QPainter, theme: Theme) -> None:
    """The offset wandering a little about its mean over the minutes."""
    _axes(p, theme)
    xs = np.linspace(_LEFT + 3, _RIGHT - 2, 90)
    ys = 45.0 + _smooth(np.cumsum(_noise(4, len(xs), 1.6)), 5)
    ys += 45.0 - ys.mean()
    _curve_pen(p, theme)
    _polyline(p, xs[3:-3], ys[3:-3])


def _level(p: QPainter, theme: Theme) -> None:
    """The peak level of the two channels."""
    _axes(p, theme)
    xs = np.linspace(_LEFT + 3, _RIGHT - 2, 70)
    for seed, mean, colour in ((5, 32.0, "accent"), (6, 44.0, "notice")):
        ys = mean + _smooth(_noise(seed, len(xs), 7.0), 2)
        _curve_pen(p, theme, colour)
        _polyline(p, xs[1:-1], ys[1:-1])


def _figures(names: tuple[str, ...], widths: tuple[float, ...]) -> Callable:
    """Rows of figures: a name (marked with `tr_noop`), and the value as a bar."""

    def draw(p: QPainter, theme: Theme) -> None:
        font = QFont()
        font.setPixelSize(8)
        p.setFont(font)
        for row, (name, width) in enumerate(zip(names, widths)):
            y = 9 + row * 15
            p.setPen(theme.color("muted"))
            p.drawText(QRectF(10, y, 64, 12), Qt.AlignmentFlag.AlignVCenter, tr(name))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(theme.color("accent"))
            p.drawRoundedRect(QRectF(76, y + 3.5, width, 5), 2.5, 2.5)

    return draw


def _globe(p: QPainter, theme: Theme) -> None:
    """The Earth with the listener's position, three orbits around it and a satellite on
    each."""
    centre = QPointF(60, 45)
    p.setPen(QPen(theme.color("muted"), 1.0))
    p.setBrush(Qt.BrushStyle.NoBrush)
    satellites = []
    # Tilt of the orbit (degrees) and where on it the satellite is (radians).
    for tilt, phase in ((-28.0, 0.7), (24.0, 3.5), (78.0, 2.2)):
        p.save()
        p.translate(centre)
        p.rotate(tilt)
        p.drawEllipse(QPointF(), 48, 17)
        p.restore()
        x, y = 48 * math.cos(phase), 17 * math.sin(phase)
        turn = math.radians(tilt)
        satellites.append(
            QPointF(
                centre.x() + x * math.cos(turn) - y * math.sin(turn),
                centre.y() + x * math.sin(turn) + y * math.cos(turn),
            )
        )
    p.setPen(QPen(theme.color("accent"), 1.6))
    p.setBrush(theme.color("accent_soft"))
    p.drawEllipse(centre, 15, 15)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(theme.color("notice"))
    p.drawEllipse(QPointF(66, 38), 2.6, 2.6)
    p.setBrush(theme.color("accent"))
    for place in satellites:
        p.drawEllipse(place, 3.2, 3.2)


def _null_symbol(p: QPainter, theme: Theme) -> None:
    """The landscape seen along the time axis: the ensemble as a slab before and after
    the trench of the null symbol, a few carriers of the TII in it, and the strip of the
    frame structure along the side."""

    def at(f: float, t: float, h: float = 0.0) -> QPointF:
        # f across the band from its low edge (right) to its high edge (left), t along
        # the time into the depth, h upwards; all from 0 to 1.
        return QPointF(86 - 66 * f + 22 * t, 84 - 12 * f - 52 * t - 12 * h)

    def face(points: list[QPointF], colour: QColor) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(colour)
        p.drawPolygon(QPolygonF(points))

    accent, soft = theme.color("accent"), theme.color("accent_soft")
    side = QColor(
        (accent.red() + soft.red()) // 2,
        (accent.green() + soft.green()) // 2,
        (accent.blue() + soft.blue()) // 2,
    )
    p.setPen(QPen(theme.color("outline"), 0.8))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolygon(QPolygonF([at(0, 0), at(1, 0), at(1, 1), at(0, 1)]))
    low, high, height = 0.125, 0.875, 1.0

    def slab(t0: float, t1: float) -> None:
        face(
            [at(low, t0, height), at(high, t0, height), at(high, t1, height), at(low, t1, height)],
            accent,
        )
        face([at(low, t0), at(high, t0), at(high, t0, height), at(low, t0, height)], soft)
        face([at(low, t0), at(low, t1), at(low, t1, height), at(low, t0, height)], side)

    slab(0.54, 1.0)  # the far one first
    p.setPen(QPen(accent, 1.0))
    for f in (0.24, 0.27, 0.43, 0.46, 0.66, 0.69):
        p.drawLine(at(f, 0.38), at(f, 0.38, 0.55))
    slab(0.0, 0.16)
    for t0, t1, name in (
        (0.0, 0.16, "accent"),
        (0.16, 0.54, None),
        (0.54, 0.6, "muted"),
        (0.6, 1.0, "accent"),
    ):
        colour = QColor(190, 110, 200) if name is None else theme.color(name)
        p.setPen(QPen(colour, 2.4, cap=Qt.PenCapStyle.FlatCap))
        p.drawLine(at(-0.05, t0), at(-0.05, t1))


_PICTURES: dict[str, Callable[[QPainter, Theme], None]] = {
    SLIDESHOW: _slideshow,
    SPECTRUM: _spectrum,
    CONSTELLATION: _constellation,
    IMPULSE: _impulse,
    OFFSET: _offset,
    LEVEL: _level,
    RECEPTION_FIGURES: _figures(
        (
            tr_noop("Offset"),
            # Translators: short for the share of the single frequency network, in a
            # small picture.
            tr_noop("SFN"),
            tr_noop("Overload"),
            tr_noop("Load"),
            tr_noop("Audio frames"),
        ),
        (24, 32, 14, 20, 28),
    ),
    SOUND_FIGURES: _figures(
        (
            tr_noop("In sync"),
            tr_noop("Corrected"),
            tr_noop("Codec"),
            tr_noop("Audio format"),
            tr_noop("Buffer"),
        ),
        (26, 18, 30, 22, 16),
    ),
    GLOBE: _globe,
    NULL_SYMBOL: _null_symbol,
}


def preview_pixmap(key: str, theme: Theme, size: QSize = PREVIEW_SIZE) -> QPixmap:
    """The stylised picture of the display `key`: a small panel in the colours of
    `theme`."""
    pixmap = QPixmap(size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(size.width() / _GRID_WIDTH, size.height() / _GRID_HEIGHT)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(theme.color("surface"))
    painter.drawRoundedRect(QRectF(0, 0, _GRID_WIDTH, _GRID_HEIGHT), 6, 6)
    _PICTURES[key](painter, theme)
    painter.end()
    return pixmap


# --- The page ---------------------------------------------------------------------------


class DisplayChoicePage(QWidget):
    """The displays as tiles, the one shown marked like the station playing; those not
    offered cannot be tapped."""

    chosen = pyqtSignal(str)  # the key of the display tapped
    back_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current: str | None = None
        title = set_role(QLabel(tr(_TITLE)), "large")
        back = QPushButton(tr("Back"))
        back.clicked.connect(self.back_requested)
        header = QHBoxLayout()
        header.addWidget(title)
        header.addStretch()
        header.addWidget(back)

        self._tiles: dict[str, QToolButton] = {}
        self._group = QButtonGroup(self)
        tiles = set_role(QWidget(), "transparent")
        grid = QGridLayout(tiles)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(16)
        grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        for index, display in enumerate(DISPLAYS):
            tile = set_role(QToolButton(), "display")
            tile.setText(tr(display.name))
            tile.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            tile.setIconSize(PREVIEW_SIZE)
            tile.setProperty("current", False)
            self._group.addButton(tile, index)
            grid.addWidget(tile, index // COLUMNS, index % COLUMNS)
            self._tiles[display.key] = tile
        self._group.idClicked.connect(self._on_tile_clicked)
        # Two rows fit the page, the third shows the top of its tiles: more lie below.
        self._area = QScrollArea()
        self._area.setWidget(tiles)
        self._area.setWidgetResizable(True)
        self._area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Swiped, as a finger expects; taps still reach the tiles.
        QScroller.grabGesture(
            self._area.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)
        layout.addLayout(header)
        layout.addWidget(self._area, stretch=1)
        self.apply_theme(THEMES[DARK])

    def apply_theme(self, theme: Theme) -> None:
        for key, tile in self._tiles.items():
            tile.setIcon(QIcon(preview_pixmap(key, theme)))

    def current(self) -> str | None:
        return self._current

    def set_current(self, key: str) -> None:
        """Marks the tile of the display shown."""
        self._current = key
        for tile_key, tile in self._tiles.items():
            current = tile_key == key
            if tile.property("current") != current:
                tile.setProperty("current", current)
                repolish(tile)

    def set_offered(self, keys: set[str]) -> None:
        """The tiles of the displays not in `keys` cannot be tapped."""
        for key, tile in self._tiles.items():
            tile.setEnabled(key in keys)

    def reveal_current(self) -> None:
        """Scrolls the tile of the display shown into sight."""
        if self._current is not None:
            self._area.ensureWidgetVisible(self._tiles[self._current], 0, 16)

    def tile(self, key: str) -> QToolButton:
        return self._tiles[key]

    @property
    def area(self) -> QScrollArea:
        return self._area

    def _on_tile_clicked(self, index: int) -> None:
        self.chosen.emit(DISPLAYS[index].key)
