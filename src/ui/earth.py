"""The Earth of the 3D view (`ui.globe_gl`): its outlines from Natural Earth, painted as
an image over longitude and latitude, and laid as lines on the WGS 84 ellipsoid. Taken
over from the gnss-receiver project.

The image only fills land and water; it is stretched over the whole globe, so that one
of its pixels is 10 km wide and a coast in it blurs when the view comes close. The
coasts, the borders and the grid are therefore drawn as lines on top, which stay sharp
at any distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPainterPath

from dab.gnss import WGS84_A, WGS84_E2

# Built by `tools/build_earth.py` from Natural Earth 1:50 million (public domain).
ASSET = Path(__file__).parent / "assets" / "earth.npz"
_UNITS_PER_DEGREE = 100  # of the coordinates in the file

# Longer steps of an outline are divided: a straight line between two points 10° apart
# (the border along the 49th parallel, the edge of Antarctica) would run 24 km below
# the surface, one over 1° runs 240 m below it.
MAX_STEP_DEG = 1.0
# Meridians and parallels of the grid. The meridians end short of the poles, where they
# would run into one point.
GRID_STEP_DEG = 15
_GRID_MAX_LATITUDE_DEG = 75

# The water darkens towards the poles; the land goes from ice over the green of the
# middle latitudes to the paler tropics and back.
_WATER = ((0.0, "#0a1a2e"), (0.5, "#0f2b4a"), (1.0, "#0a1a2e"))
_LAND = (
    (0.00, "#c9d6dc"),  # 90° north: ice
    (0.08, "#b4c4cb"),
    (0.15, "#5d7a6c"),  # 63° north
    (0.30, "#3f6b54"),
    (0.42, "#5e7a52"),
    (0.50, "#4d7350"),  # the equator
    (0.58, "#5e7a52"),
    (0.72, "#3f6b54"),
    (0.83, "#6c8478"),
    (0.87, "#c9d6dc"),  # 67° south: Antarctica
    (1.00, "#dde6ea"),
)
_LAKE = QColor("#12325a")


@dataclass(frozen=True)
class EarthOutlines:
    """Longitude and latitude in degrees, (N, 2) each."""

    land: list[np.ndarray]  # closed rings; a ring inside another is a hole
    lakes: list[np.ndarray]  # closed rings
    borders: list[np.ndarray]  # open lines


def load_outlines(path: Path = ASSET) -> EarthOutlines:
    with np.load(path) as data:

        def parts(points: str, lengths: str) -> list[np.ndarray]:
            degrees = data[points].astype(np.float64) / _UNITS_PER_DEGREE
            return np.split(degrees, np.cumsum(data[lengths])[:-1])

        return EarthOutlines(
            land=parts("land", "land_rings"),
            lakes=parts("lakes", "lake_rings"),
            borders=parts("borders", "border_lines"),
        )


def ellipsoid_km(latitude_deg: np.ndarray, longitude_deg: np.ndarray) -> np.ndarray:
    """Earth-fixed positions (..., 3) in km of places on the WGS 84 ellipsoid."""
    lat, lon = np.radians(latitude_deg), np.radians(longitude_deg)
    n = WGS84_A / 1000 / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    return np.stack(
        [
            n * np.cos(lat) * np.cos(lon),
            n * np.cos(lat) * np.sin(lon),
            n * (1 - WGS84_E2) * np.sin(lat),
        ],
        axis=-1,
    )


def paint_surface(outlines: EarthOutlines, width: int = 4096) -> QImage:
    """Land and water as an image over longitude (-180° on the left) and latitude (north
    at the top), twice as wide as high."""
    height = width // 2
    image = QImage(width, height, QImage.Format.Format_RGB32)
    painter = QPainter(image)
    painter.fillRect(image.rect(), _gradient(_WATER, height))
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(_gradient(_LAND, height))
    painter.drawPath(_path(outlines.land, width, height))
    painter.setBrush(_LAKE)
    painter.drawPath(_path(outlines.lakes, width, height))
    painter.end()
    return image


def _gradient(stops: tuple[tuple[float, str], ...], height: int) -> QLinearGradient:
    gradient = QLinearGradient(0.0, 0.0, 0.0, float(height))
    for at, colour in stops:
        gradient.setColorAt(at, QColor(colour))
    return gradient


def _path(rings: list[np.ndarray], width: int, height: int) -> QPainterPath:
    path = QPainterPath()  # odd-even fill: a ring inside another cuts a hole
    for ring in rings:
        x = (ring[:, 0] + 180.0) / 360.0 * width
        y = (90.0 - ring[:, 1]) / 180.0 * height
        path.addPolygon(pg.functions.arrayToQPolygonF(x, y))
        path.closeSubpath()
    return path


def outline_segments(lines: list[np.ndarray], closed: bool) -> np.ndarray:
    """The lines as pairs of points on the ellipsoid, (2 M, 3) in km as float32: every
    two rows are one straight piece. Of rings (`closed`), the edges that only close them
    in the file are left out: along the meridian of 180°, where the land is cut, and
    along the south pole."""
    if closed:
        lines = [
            line if np.all(line[0] == line[-1]) else np.vstack([line, line[:1]]) for line in lines
        ]
    if not lines:
        return np.empty((0, 3), dtype=np.float32)
    # All lines in one array (a loop over the 2200 rings took 270 ms, this takes 15 ms,
    # measured 02.10.2026): every point but the last of a line begins an edge.
    points = np.concatenate(lines)
    begins_edge = np.ones(len(points), dtype=bool)
    begins_edge[np.cumsum([len(line) for line in lines]) - 1] = False
    start, end = points[begins_edge], points[np.flatnonzero(begins_edge) + 1]
    cut = np.zeros(len(start), dtype=bool)
    if closed:
        cut = (np.abs(start[:, 0]) >= 180.0) & (np.abs(end[:, 0]) >= 180.0)
        cut |= (start[:, 1] <= -90.0) & (end[:, 1] <= -90.0)
    steps = np.abs(end - start).max(axis=1)
    counts = np.maximum(1, np.ceil(steps / MAX_STEP_DEG)).astype(int)
    # Every piece of every edge: which edge it belongs to and where on it it begins.
    edge = np.repeat(np.arange(len(counts)), counts)
    first = np.repeat(np.cumsum(counts) - counts, counts)
    fraction = (np.arange(len(edge)) - first) / counts[edge]
    span = end[edge] - start[edge]
    begin = start[edge] + span * fraction[:, None]
    finish = start[edge] + span * (fraction + 1 / counts[edge])[:, None]
    keep = ~cut[edge]
    pairs = np.stack([begin[keep], finish[keep]], axis=1).reshape(-1, 2)
    return ellipsoid_km(pairs[:, 1], pairs[:, 0]).astype(np.float32)


def grid_segments() -> np.ndarray:
    """Meridians and parallels every GRID_STEP_DEG degrees, as `outline_segments`."""
    lines = []
    latitudes = np.arange(-_GRID_MAX_LATITUDE_DEG, _GRID_MAX_LATITUDE_DEG + 1.0)
    for longitude in range(-180, 180, GRID_STEP_DEG):
        lines.append(np.column_stack([np.full_like(latitudes, longitude), latitudes]))
    longitudes = np.arange(-180.0, 181.0)
    for latitude in range(-_GRID_MAX_LATITUDE_DEG, _GRID_MAX_LATITUDE_DEG + 1, GRID_STEP_DEG):
        lines.append(np.column_stack([longitudes, np.full_like(longitudes, latitude)]))
    return outline_segments(lines, closed=False)
