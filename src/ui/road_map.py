"""The road network (`dab.road_network`) beneath a pyqtgraph map of longitude and
latitude: the layers that suit the zoom, cut to what is in view, town names, and the
attribution OpenStreetMap asks for.

Seen whole, Germany shows only the coarse motorways, main roads, borders and coast with
the cities; closer in, the main roads and the Landesstraßen in medium detail with the
towns; close up all of them fine, and the Kreisstraßen and slip roads (`level_for`).
Every level draws a margin around the view, so that moving the map a little draws
nothing anew.

The roads are painted once into a picture of that region (`_Picture`), which the map
then shows as an image: moving the view moves the image, and the messages above it,
which change every few seconds, no longer paint the network again (painted every frame,
all of Germany cost 84 ms when a message changed and 120 ms per step of a pinch). The
picture is painted anew when the view leaves it, the level changes, or the scale differs
by more than `RESCALE_FACTOR`; until then a pinch shows it scaled. Qt's own cache of an
item's painting (`QGraphicsItem.DeviceCoordinateCache`) left patches of the map empty
after zooming.

The picture is painted in slices (`SLICE_POINTS`) with the event loop between them,
since a call of QPainter holds the GIL for all its length: painted at once, the picture
kept the signal processing and the audio from running Python for as long as it took
(drawLines of 140 000 points took 67 ms), and the map from taking a touch. A thread
would not help, it would hold the GIL just the same. The picture shown stays until the
new one is done.

The names of the towns are one item too (`_Labels`), each name painted once as a small
image with its halo and put on whole pixels: as `MapLabel`s, forty of them set their
transforms anew one by one at every step of a pinch, in Python (3.7 ms per step before
anything was painted).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QTransform,
)
from PyQt6.QtWidgets import QGraphicsObject, QStyleOptionGraphicsItem, QWidget

from dab.road_network import (
    ATTRIBUTION,
    BORDER,
    CITY,
    COARSE,
    COAST,
    FINE,
    LINK,
    MAJOR,
    MEDIUM,
    MOTORWAY,
    SECONDARY,
    STATE,
    TERTIARY,
    TOWN,
    Bounds,
    Place,
    RoadNetwork,
)
from i18n import tr
from ui.plots import HALO_WIDTH, MAP_LABEL_Z

Colour = tuple[int, int, int]

# The levels by the width of the view in degrees of longitude.
OVERVIEW = 0  # wider than OVERVIEW_DEGREES
REGION = 1  # wider than LOCAL_DEGREES
LOCAL = 2
OVERVIEW_DEGREES = 4.0
LOCAL_DEGREES = 1.2
# Each level draws this share of the view's size again on every side.
MARGIN = 0.3
# Town names: at most this many, the largest first.
MAX_LABELS = 40
# Drawing waits this long after the view moved, so that a pinch draws once at its end
# rather than at every step.
REDRAW_DELAY_MS = 60
# The picture is painted anew once the scale of the view differs from the one it was
# painted at by more than this factor: closer in it would blur, farther out fray.
RESCALE_FACTOR = 1.25
# Neither side of the picture has more pixels (a big window on a screen scaled up, with
# the margin).
MAX_PICTURE_PX = 4096
# Points painted into the picture at a time, a few milliseconds each.
SLICE_POINTS = 10_000
# The margin a QGraphicsTextItem keeps around its text: the names stand where the
# `MapLabel`s stood, their lower left corner (with that margin) at the town.
_TEXT_MARGIN = 4.0


@dataclass(frozen=True)
class RoadColours:
    """The default suits the black background of the desktop's plots."""

    motorway: Colour = (150, 150, 160)
    major: Colour = (115, 115, 125)
    minor: Colour = (80, 80, 90)
    border: Colour = (140, 140, 150)
    coast: Colour = (70, 110, 150)
    text: Colour = (170, 170, 180)
    ground: Colour = (0, 0, 0)  # the plot's background, beneath the attribution and names


# Per layer: colour role and pen width in pixels (scaled by `RoadMap(width_scale=...)`).
_STYLE = {
    MOTORWAY: ("motorway", 2.5),
    MAJOR: ("major", 1.8),
    SECONDARY: ("minor", 1.3),
    TERTIARY: ("minor", 1.0),
    LINK: ("minor", 1.0),
    BORDER: ("border", 1.5),
    STATE: ("border", 1.0),
    COAST: ("coast", 1.5),
}
_DASHES = {BORDER: Qt.PenStyle.DashLine, STATE: Qt.PenStyle.DotLine}
# The layers of each level, the least important drawn first, and the copy drawn.
_LEVEL_LAYERS = {
    OVERVIEW: (STATE, COAST, BORDER, MAJOR, MOTORWAY),
    REGION: (STATE, COAST, BORDER, SECONDARY, MAJOR, MOTORWAY),
    LOCAL: (STATE, COAST, BORDER, TERTIARY, LINK, SECONDARY, MAJOR, MOTORWAY),
}
_LEVEL_DETAIL = {OVERVIEW: COARSE, REGION: MEDIUM, LOCAL: FINE}
_LEVEL_PLACES = {OVERVIEW: CITY, REGION: TOWN, LOCAL: TOWN}


def pixel_size(view_box: pg.ViewBox) -> tuple[float, float]:
    """Degrees of longitude and latitude per pixel of a view box; 0 before it has a
    size."""
    (west, east), (south, north) = view_box.viewRange()
    width, height = view_box.width(), view_box.height()
    if width <= 0 or height <= 0:
        return 0.0, 0.0
    return (east - west) / width, (north - south) / height


def level_for(width_degrees: float) -> int:
    if width_degrees > OVERVIEW_DEGREES:
        return OVERVIEW
    if width_degrees > LOCAL_DEGREES:
        return REGION
    return LOCAL


class _Picture(QGraphicsObject):
    """An image laid over a region of the map (in degrees), its first row in the south:
    the view turns the y axis upwards."""

    def __init__(self) -> None:
        super().__init__()
        self.image: QImage | None = None
        self._rect = QRectF()

    def set_image(self, image: QImage | None, rect: QRectF) -> None:
        self.prepareGeometryChange()
        self.image = image
        self._rect = rect
        self.update()

    def boundingRect(self) -> QRectF:
        return QRectF(self._rect)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem | None,
        widget: QWidget | None = None,
    ) -> None:
        if self.image is None:
            return
        # Scaled during a pinch it is smoothed; at its own scale smoothing would only
        # blur it by the fraction of a pixel the view stands off.
        shown = painter.worldTransform().mapRect(self._rect).width()
        scale = shown / self.image.width() * self.image.devicePixelRatio()
        smooth = abs(scale - 1.0) > 0.02
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)
        painter.drawImage(self._rect, self.image)


class _Labels(QGraphicsObject):
    """The names of the towns, one item for all of them, above everything else in the
    plot (`MAP_LABEL_Z`); see the module. Placed in pixels when painted, so its bounding
    rect is the whole world: it is painted whenever the view is."""

    def __init__(self, font: QFont | None) -> None:
        super().__init__()
        self._font = QFont(font) if font is not None else QFont()
        self._colour = QColor(200, 200, 200)
        self._halo = QColor(0, 0, 0)
        self._places: list[Place] = []
        self._sprites: dict[tuple[str, float], QPixmap] = {}
        self.setZValue(MAP_LABEL_Z)

    @property
    def names(self) -> list[str]:
        return [place.name for place in self._places]

    @property
    def colour(self) -> QColor:
        return QColor(self._colour)

    @property
    def halo(self) -> QColor:
        return QColor(self._halo)

    def set_places(self, places: list[Place]) -> None:
        self._places = list(places)
        self.update()

    def set_colours(self, text: QColor, halo: QColor) -> None:
        if text != self._colour or halo != self._halo:
            self._colour, self._halo = QColor(text), QColor(halo)
            self._sprites.clear()
            self.update()

    def boundingRect(self) -> QRectF:
        return QRectF(-180.0, -90.0, 360.0, 180.0)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem | None,
        widget: QWidget | None = None,
    ) -> None:
        self.paint_at(painter, painter.worldTransform())

    def paint_at(self, painter: QPainter, to_pixels: QTransform) -> None:
        """Paints the names with `to_pixels` taking degrees to the pixels of the painter's
        device; also over the picture of the view two fingers move (`ui.touch.gestures`)."""
        if not self._places:
            return
        ratio = painter.device().devicePixelRatioF()
        painter.save()
        painter.resetTransform()
        for place in self._places:
            sprite = self._sprite(place.name, ratio)
            at = to_pixels.map(QPointF(place.longitude, place.latitude))
            height = sprite.height() / ratio
            # The text's box (without the halo around it) has its lower left corner
            # at the town.
            painter.drawPixmap(
                QPointF(round(at.x() - HALO_WIDTH / 2), round(at.y() + HALO_WIDTH / 2 - height)),
                sprite,
            )
        painter.restore()

    def _sprite(self, name: str, ratio: float) -> QPixmap:
        """The name with its halo, painted once per name and screen scale."""
        key = (name, ratio)
        sprite = self._sprites.get(key)
        if sprite is None:
            metrics = QFontMetricsF(self._font)
            width = metrics.horizontalAdvance(name) + 2 * _TEXT_MARGIN + HALO_WIDTH
            height = metrics.height() + 2 * _TEXT_MARGIN + HALO_WIDTH
            sprite = QPixmap(math.ceil(width * ratio), math.ceil(height * ratio))
            sprite.setDevicePixelRatio(ratio)
            sprite.fill(Qt.GlobalColor.transparent)
            painter = QPainter(sprite)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            baseline = QPointF(
                _TEXT_MARGIN + HALO_WIDTH / 2, _TEXT_MARGIN + HALO_WIDTH / 2 + metrics.ascent()
            )
            # The halo as `MapLabel` paints it: the letters outlined wide, beneath them.
            outline = QPainterPath()
            outline.addText(baseline, self._font, name)
            pen = QPen(self._halo, HALO_WIDTH)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.strokePath(outline, pen)
            painter.setFont(self._font)
            painter.setPen(self._colour)
            painter.drawText(baseline, name)
            painter.end()
            self._sprites[key] = sprite
        return sprite


class RoadMap(QObject):
    """Draws a `RoadNetwork` into `plot` beneath everything else and follows its view;
    only the names of the towns lie above it all (`_Labels`)."""

    def __init__(
        self,
        plot: pg.PlotWidget,
        colours: RoadColours | None = None,
        font: QFont | None = None,
        width_scale: float = 1.0,
    ) -> None:
        super().__init__(plot)
        self._plot = plot
        self._colours = colours or RoadColours()
        self._font = font
        self._width_scale = width_scale
        self._network: RoadNetwork | None = None
        # The level, the region and the degrees of longitude per pixel drawn.
        self._drawn: tuple[int, Bounds, float] | None = None
        self._lines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._pens: dict[str, QPen] = {}
        self._picture = _Picture()
        self._picture.setZValue(-20)
        plot.addItem(self._picture, ignoreBounds=True)
        # Paints a layer into the picture as the plots draw a curve (wide lines as single
        # segments, which Qt draws faster than a path); never shown itself.
        self._painter = pg.PlotCurveItem(connect="finite")
        # The picture being painted, a slice per step of the event loop.
        self._painting: Iterator[None] | None = None
        self._slices = QTimer(self)
        self._slices.setInterval(0)
        self._slices.timeout.connect(self._paint_slice)
        self._places = pg.ScatterPlotItem(size=4, pen=None)
        self._places.setZValue(-9)
        plot.addItem(self._places, ignoreBounds=True)
        self._labels = _Labels(self._font)
        plot.addItem(self._labels, ignoreBounds=True)
        view_box = plot.getViewBox()
        self._attribution = pg.TextItem(tr(ATTRIBUTION), anchor=(1, 1))
        self._attribution.setParentItem(view_box)
        self._attribution.setVisible(False)
        view_box.sigResized.connect(self._place_attribution)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(REDRAW_DELAY_MS)
        self._timer.timeout.connect(self.redraw)
        view_box.sigRangeChanged.connect(self._on_range_changed)
        self._apply_style()

    @property
    def network(self) -> RoadNetwork | None:
        return self._network

    @property
    def level(self) -> int | None:
        """The level drawn last, None before the first drawing."""
        return None if self._drawn is None else self._drawn[0]

    def set_network(self, network: RoadNetwork | None) -> None:
        self._network = network
        self._attribution.setVisible(network is not None)
        self._drawn = None
        self.redraw()

    def set_colours(self, colours: RoadColours) -> None:
        self._colours = colours
        self._apply_style()
        self._drawn = None
        self.redraw()

    @property
    def picture(self) -> QImage | None:
        """The roads as painted last."""
        return self._picture.image

    @property
    def labels(self) -> _Labels:
        """The names of the towns shown."""
        return self._labels

    @property
    def painting(self) -> bool:
        """A new picture is being painted."""
        return self._painting is not None

    def lines(self, layer: str) -> tuple[np.ndarray, np.ndarray]:
        """The lines of a layer as painted last, with NaN between them."""
        return self._lines.get(layer, (np.empty(0), np.empty(0)))

    def _apply_style(self) -> None:
        for layer, (role, width) in _STYLE.items():
            pen = pg.mkPen(getattr(self._colours, role), width=width * self._width_scale)
            pen.setStyle(_DASHES.get(layer, Qt.PenStyle.SolidLine))
            self._pens[layer] = pen
        self._places.setBrush(pg.mkBrush(*self._colours.text))
        self._labels.set_colours(QColor(*self._colours.text), QColor(*self._colours.ground))
        self._attribution.setColor(QColor(*self._colours.text))
        self._attribution.fill = pg.mkBrush(*self._colours.ground, 210)
        self._attribution.update()
        if self._font is not None:
            self._attribution.setFont(self._font)

    def _view(self) -> Bounds:
        (west, east), (south, north) = self._plot.getViewBox().viewRange()
        return west, south, east, north

    def _on_range_changed(self, *_args: object) -> None:
        if self._network is None:
            return
        west, south, east, north = self._view()
        if self._drawn is not None:
            level, (dw, ds, de, dn), pixel = self._drawn
            inside = west >= dw and east <= de and south >= ds and north <= dn
            scale = self._pixel()[0] / pixel if pixel > 0 else 0.0
            rescaled = not 1 / RESCALE_FACTOR <= scale <= RESCALE_FACTOR
            if inside and level == level_for(east - west) and not rescaled:
                return
        self._timer.start()

    def _pixel(self) -> tuple[float, float]:
        """Degrees of longitude and latitude per pixel of the view; 0 before it has a
        size. From the range and the size of the view: pyqtgraph's `viewPixelSize`
        maps three points through the inverse transform instead, at every step of a
        pinch."""
        return pixel_size(self._plot.getViewBox())

    def redraw(self) -> None:
        """Draws the layers of the level that suits the view, with the margin; the
        picture of them follows slice by slice."""
        self._timer.stop()
        self._stop_painting()
        network = self._network
        if network is None:
            self._lines = {}
            self._picture.set_image(None, QRectF())
            self._places.setData([], [])
            self._labels.set_places([])
            self._drawn = None
            return
        west, south, east, north = self._view()
        level = level_for(east - west)
        mx, my = (east - west) * MARGIN, (north - south) * MARGIN
        region = (west - mx, south - my, east + mx, north + my)
        self._lines = {
            layer: network.lines(layer, region, _LEVEL_DETAIL[level])
            for layer in _LEVEL_LAYERS[level]
        }
        pixel = self._pixel()
        self._painting = self._paint(region, pixel)
        self._slices.start()
        places = network.places_in(region, _LEVEL_PLACES[level])[:MAX_LABELS]
        self._places.setData([p.longitude for p in places], [p.latitude for p in places])
        self._labels.set_places(places)
        self._drawn = (level, region, pixel[0])

    def _paint(self, region: Bounds, pixel: tuple[float, float]) -> Iterator[None]:
        """Paints the lines into a picture of `region` at the view's scale and the
        screen's pixel ratio, `SLICE_POINTS` per step, and shows it when done; none
        while the view has no size."""
        west, south, east, north = region
        if pixel[0] <= 0 or pixel[1] <= 0:
            self._picture.set_image(None, QRectF())
            return
        ratio = self._plot.devicePixelRatioF()
        columns = max(1, min(MAX_PICTURE_PX, round((east - west) / pixel[0] * ratio)))
        rows = max(1, min(MAX_PICTURE_PX, round((north - south) / pixel[1] * ratio)))
        image = QImage(columns, rows, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        sx, sy = columns / (east - west), rows / (north - south)
        painter = QPainter(image)
        try:
            painter.setTransform(QTransform(sx, 0, 0, sy, -west * sx, -south * sy))
            for layer in _STYLE:
                x, y = self._lines.get(layer, ((), ()))
                pen = QPen(self._pens[layer])
                pen.setWidthF(pen.widthF() * ratio)
                self._painter.setPen(pen)
                # Each slice begins with the last point of the one before: they meet.
                for start in range(0, len(x) - 1, SLICE_POINTS):
                    stop = start + SLICE_POINTS + 1
                    self._painter.setData(x[start:stop], y[start:stop])
                    self._painter.paint(painter, None, None)
                    yield
        finally:
            painter.end()
            self._painter.setData([], [])
        image.setDevicePixelRatio(ratio)
        self._picture.set_image(image, QRectF(west, south, east - west, north - south))

    def _paint_slice(self) -> None:
        if self._painting is not None:
            try:
                next(self._painting)
                return
            except StopIteration:
                self._painting = None
        self._slices.stop()

    def _stop_painting(self) -> None:
        if self._painting is not None:
            self._painting.close()
            self._painting = None
        self._slices.stop()

    def _place_attribution(self, *_args: object) -> None:
        rect = self._plot.getViewBox().boundingRect()
        self._attribution.setPos(rect.right() - 4, rect.bottom() - 2)
