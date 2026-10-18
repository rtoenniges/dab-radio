"""OpenGL for the detail view: a field of levels over time and frequency
(`ui.spectrogram.Heightfield`) drawn as a landscape into an image.

Drawn off screen in a thread of its own, as `ui.offscreen_gl` describes; the view has
a thread of its own (`RenderThread(WaterfallRenderer())`), apart from the globe's. What
the view hands over as the scene is a `WaterfallScene`; a new one is loaded into the
graphics card, the same one again is not.

The landscape lies in units of its own: time across (x) from -`HALF_WIDTH`, the
earliest row, to `HALF_WIDTH`, the latest (left to right for a viewer in front);
frequency into the depth (y) from -`HALF_DEPTH`, the lowest, to `HALF_DEPTH`; the
levels upwards from the floor of the field (0) to its top (`HEIGHT`). Seen from the
front left, the corner nearest the viewer holds the earliest time and the lowest
frequency; the time rises along the front edge to the right, the frequency along the
left edge into the depth. It is one triangle strip over a grid of points (the strips of
the rows joined by degenerate triangles). A column or row drawn as a block of its own
has four points across: down at the floor, up, along its top, down again; between
blocks the floor shows. A symbol stands over the samples its FFT takes, so that the
floor between two is the guard interval, as the strip below marks it. The faces are lit
flat, with the normal of each triangle taken from the screen-space derivatives of its
position in the fragment shader, so the grid needs no normals; a face turned away from
the viewer is lit from its other side, which leaves the winding (and the upside-down
drawing of the readback) out of it.

Along the front edge of the floor, before the figures of the time, a strip shows the
frame structure (`ui.spectrogram.Part`): null symbols, guard intervals and useful parts.
On the floor itself it would lie under the plateau.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from PyQt6.QtGui import QVector3D

from ui.offscreen_gl import (
    GL_BLEND,
    GL_CULL_FACE,
    GL_DEPTH_TEST,
    GL_LINES,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    GL_TRIANGLE_STRIP,
    GL_TRIANGLES,
    LINES_ATTRIBUTES,
    LINES_FRAGMENT,
    LINES_VERTEX,
    Mesh,
    OffscreenRenderer,
    perspective,
    view_axes,
)
from ui.spectrogram import DEFAULT_LEVEL_RANGE_DB, TOP_DB, Heightfield, Part

FIELD_OF_VIEW_DEG = 32.0
HALF_WIDTH = 1.0
HALF_DEPTH = 0.8
HEIGHT = 0.5
# A carrier's block leaves this share of its width to the floor on each side.
BLOCK_GAP = 0.12
# The strip of the frame structure, in front of the floor.
STRIP_DEPTH = 0.05
# The camera before the user moves it: from the front left, looking down onto the
# trenches of the null symbols, with the lowest frequency and the earliest time in the
# corner nearest to it (see the module).
DEFAULT_AZIMUTH_DEG = -30.0
DEFAULT_ELEVATION_DEG = 38.0
DEFAULT_DISTANCE = 4.7
DISTANCE_RANGE = (1.6, 9.0)
# The point the camera looks at and turns around: the middle of the floor, a little
# towards the front, a quarter of the height up.
DEFAULT_LOOK_AT = (0.0, -0.05, 0.25 * HEIGHT)
# How far that point may be shifted (`WaterfallCamera.pan`), per axis: the landscape
# stays in view.
LOOK_AT_LIMITS = ((-1.2, 1.2), (-1.0, 1.0), (-0.3, 0.8))
ELEVATION_RANGE = (4.0, 89.0)
# A line on the back wall every so many dB, by the range of levels the height spans:
# three to five lines.
_LEVEL_STEPS_DB = {10.0: 2.0, 20.0: 5.0, 30.0: 10.0, 50.0: 20.0}

_SURFACE_VERTEX = """
attribute vec3 a_position;
uniform mat4 u_matrix;
varying vec3 v_world;
void main() {
    v_world = a_position;
    gl_Position = u_matrix * vec4(a_position, 1.0);
}
"""
# The colour of the level, from the floor to the top of the field, in five stops as the
# density of the constellation (`ui.touch.signal_panels.density_colour_map`).
_SURFACE_FRAGMENT = """
uniform vec3 u_c0;
uniform vec3 u_c1;
uniform vec3 u_c2;
uniform vec3 u_c3;
uniform vec3 u_c4;
uniform vec3 u_light;
uniform vec3 u_camera;
uniform float u_height;
varying vec3 v_world;
vec3 gradient(float t) {
    if (t < 0.30) return mix(u_c0, u_c1, t / 0.30);
    if (t < 0.62) return mix(u_c1, u_c2, (t - 0.30) / 0.32);
    if (t < 0.86) return mix(u_c2, u_c3, (t - 0.62) / 0.24);
    return mix(u_c3, u_c4, (t - 0.86) / 0.14);
}
void main() {
    vec3 n = normalize(cross(dFdx(v_world), dFdy(v_world)));
    if (dot(n, u_camera - v_world) < 0.0) n = -n;
    float t = clamp(v_world.z / u_height, 0.0, 1.0);
    float light = 0.50 + 0.50 * max(dot(n, u_light), 0.0);
    gl_FragColor = vec4(gradient(t) * light, 1.0);
}
"""


@dataclass
class WaterfallCamera:
    """Looks at a point of the landscape (`look_at`, by default near its middle) from a
    direction (azimuth 0: from the front, positive: from the right) and height
    (elevation) at a distance."""

    azimuth_deg: float = DEFAULT_AZIMUTH_DEG
    elevation_deg: float = DEFAULT_ELEVATION_DEG
    distance: float = DEFAULT_DISTANCE
    look_at: tuple[float, float, float] = DEFAULT_LOOK_AT

    def rotate(self, azimuth_deg: float, elevation_deg: float) -> None:
        self.azimuth_deg = (self.azimuth_deg + azimuth_deg + 180.0) % 360.0 - 180.0
        low, high = ELEVATION_RANGE
        self.elevation_deg = min(max(self.elevation_deg + elevation_deg, low), high)

    def zoom(self, factor: float) -> None:
        low, high = DISTANCE_RANGE
        self.distance = min(max(self.distance * factor, low), high)

    def target(self) -> np.ndarray:
        return np.array(self.look_at)

    def pan(self, right: float, up: float) -> None:
        """Shifts the point looked at by `right` and `up` along the axes of the view,
        within `LOOK_AT_LIMITS`."""
        axes = self.axes()
        point = self.target() + right * axes[0] + up * axes[1]
        self.look_at = tuple(
            float(min(max(value, low), high))
            for value, (low, high) in zip(point, LOOK_AT_LIMITS, strict=True)
        )

    def units_per_pixel(self, height_px: float) -> float:
        """What a pixel of a view `height_px` high covers at the point looked at."""
        return 2 * self.distance * math.tan(math.radians(FIELD_OF_VIEW_DEG) / 2) / height_px

    def direction(self) -> np.ndarray:
        """From the target towards the viewer, unit length."""
        az, el = math.radians(self.azimuth_deg), math.radians(self.elevation_deg)
        return np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])

    def position(self) -> np.ndarray:
        return self.target() + self.distance * self.direction()

    def axes(self) -> np.ndarray:
        """Rows right, up and back (towards the viewer) of the view."""
        return view_axes(self.direction())

    def matrix(self, aspect: float) -> np.ndarray:
        """Landscape units to clip coordinates."""
        view = np.eye(4)
        view[:3, :3] = axes = self.axes()
        view[:3, 3] = -(axes @ self.position())
        return perspective(FIELD_OF_VIEW_DEG, aspect, 0.05, self.distance + 4.0) @ view


# The stretches of the frame in the order of `WaterfallColours.parts`.
PARTS = (Part.NULL, Part.GUARD, Part.USEFUL)


@dataclass(frozen=True)
class WaterfallColours:
    """Colours as (r, g, b) or (r, g, b, a) from 0 to 1."""

    background: tuple[float, float, float]
    stops: tuple[tuple[float, float, float], ...]  # five, from the floor to the top
    grid: tuple[float, float, float, float]
    # Of the strip: the null symbol, the guard interval, the useful part (`PARTS`).
    parts: tuple[tuple[float, float, float, float], ...] = (
        (0.75, 0.43, 0.78, 0.85),
        (0.58, 0.63, 0.69, 0.55),
        (0.31, 0.76, 0.97, 0.85),
    )

    def part(self, part: Part) -> tuple[float, float, float, float]:
        return self.parts[PARTS.index(part)]


@dataclass(frozen=True, eq=False)
class WaterfallScene:
    """What the view hands to the renderer; told apart by identity."""

    field: Heightfield
    colours: WaterfallColours
    # The levels the height spans, down from `ui.spectrogram.TOP_DB`.
    range_db: float = DEFAULT_LEVEL_RANGE_DB


# --- the landscape in numbers --------------------------------------------------------------


def frequency_y(field: Heightfield, khz: np.ndarray | float) -> np.ndarray:
    """Higher is deeper, the lowest at the front."""
    low, high = field.span_khz
    return -HALF_DEPTH + 2 * HALF_DEPTH * (np.asarray(khz) - low) / (high - low)


def time_x(field: Heightfield, ms: np.ndarray | float) -> np.ndarray:
    """Later is further right, the earliest row on the left."""
    low, high = field.window_ms
    x = -HALF_WIDTH + 2 * HALF_WIDTH * (np.asarray(ms) - low) / (high - low)
    return np.clip(x, -HALF_WIDTH, HALF_WIDTH)


def level_z(levels_db: np.ndarray, range_db: float = DEFAULT_LEVEL_RANGE_DB) -> np.ndarray:
    """The height of levels (dB relative to the plateau) when the height spans
    `range_db` down from `TOP_DB`: what lies below rests on the floor."""
    return HEIGHT * np.clip((levels_db - (TOP_DB - range_db)) / range_db, 0.0, 1.0)


def _blocks(
    lows: np.ndarray, highs: np.ndarray, gap: float = BLOCK_GAP
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Four points per block: down at its low edge, up, along its top, down at its high
    edge, `gap` of its width inside its edges. Positions, the block each point belongs
    to, and which are on the floor."""
    width = highs - lows
    low, high = lows + gap * width, highs - gap * width
    positions = np.stack([low, low, high, high], axis=1).ravel()
    index = np.repeat(np.arange(len(lows)), 4)
    floor = np.tile([True, False, False, True], len(lows))
    return positions, index, floor


def _steps(lows: np.ndarray, highs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two points per row, its start and end: flat rows with upright steps between."""
    positions = np.stack([lows, highs], axis=1).ravel()
    index = np.repeat(np.arange(len(lows)), 2)
    return positions, index, np.zeros(len(positions), dtype=bool)


def _smooth(middles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return middles, np.arange(len(middles)), np.zeros(len(middles), dtype=bool)


def _skirted(
    positions: np.ndarray, index: np.ndarray, floor: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The same with a point on the floor at either end: the landscape stands on the
    floor as a block, it does not float."""
    if floor[0] and floor[-1]:
        return positions, index, floor
    return (
        np.concatenate([[positions[0]], positions, [positions[-1]]]),
        np.concatenate([[index[0]], index, [index[-1]]]),
        np.concatenate([[True], floor, [True]]),
    )


def landscape_grid(field: Heightfield, range_db: float = DEFAULT_LEVEL_RANGE_DB) -> np.ndarray:
    """The points of the landscape, (rows, columns, 3), its height spanning
    `range_db`."""
    khz, half = field.frequencies_khz, field.column_khz / 2
    if field.discrete_columns:
        deep = _blocks(frequency_y(field, khz - half), frequency_y(field, khz + half))
    else:
        deep = _smooth(frequency_y(field, khz))
    starts, ends = time_x(field, field.row_starts_ms), time_x(field, field.row_ends_ms)
    if field.discrete_rows:
        # A symbol over the samples of its FFT: the floor between is the guard interval.
        windows = time_x(field, field.fft_starts_ms), time_x(field, field.fft_ends_ms)
        across = _blocks(*windows, gap=0.0)
    elif field.symbols is not None:
        across = _steps(starts, ends)
    else:
        across = _smooth((starts + ends) / 2)
    ys, columns, column_floor = _skirted(*deep)
    xs, rows, row_floor = _skirted(*across)
    z = level_z(field.levels_db, range_db)[np.ix_(rows, columns)]
    z[row_floor, :] = 0.0
    z[:, column_floor] = 0.0
    grid = np.empty((len(xs), len(ys), 3), dtype=np.float32)
    grid[..., 0] = xs[:, None]
    grid[..., 1] = ys[None, :]
    grid[..., 2] = z
    return grid


def triangle_strip(grid: np.ndarray) -> np.ndarray:
    """The grid as one triangle strip: each pair of neighbouring rows a strip, joined by
    repeating the last point of one and the first of the next. (N, 3) float32."""
    rows, columns, _ = grid.shape
    pairs = np.stack([grid[:-1], grid[1:]], axis=2).reshape(rows - 1, 2 * columns, 3)
    joined = np.concatenate([pairs[:, :1], pairs, pairs[:, -1:]], axis=1)
    return np.ascontiguousarray(joined.reshape(-1, 3), dtype=np.float32)


def nice_step(span: float, count: int) -> float:
    """A step of 1, 2 or 5 times a power of ten that cuts `span` into about `count`."""
    raw = span / max(count, 1)
    power = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 5, 10):
        if factor * power >= raw:
            return factor * power
    return 10 * power


def frequency_ticks(field: Heightfield) -> np.ndarray:
    low, high = field.span_khz
    step = nice_step(high - low, 5)
    return np.arange(math.ceil(low / step) * step, high + step / 1000, step)


def level_ticks(range_db: float = DEFAULT_LEVEL_RANGE_DB) -> np.ndarray:
    """The levels marked on the back wall, from the top down to just above the floor:
    multiples of a step, the plateau (0 dB) among them."""
    step = _LEVEL_STEPS_DB.get(range_db) or nice_step(range_db, 4)
    return np.arange(math.floor(TOP_DB / step) * step, TOP_DB - range_db, -step)


def grid_lines(
    field: Heightfield,
    colour: tuple[float, float, float, float],
    range_db: float = DEFAULT_LEVEL_RANGE_DB,
) -> np.ndarray:
    """The outline of the floor, ticks along its front (time) and left (frequency)
    edges, and on the back wall a line per level of `level_ticks`: (2 N, 7), position
    and colour."""
    w, d = HALF_WIDTH, HALF_DEPTH
    segments = [
        ((-w, d, 0), (w, d, 0)),
        ((w, d, 0), (w, -d, 0)),
        ((w, -d, 0), (-w, -d, 0)),
        ((-w, -d, 0), (-w, d, 0)),
        ((-w, d, 0), (-w, d, HEIGHT)),
        ((w, d, 0), (w, d, HEIGHT)),
    ]
    for x in time_x(field, time_ticks(field)[0]):
        segments.append(((x, -d, 0), (x, -d - 0.05, 0)))
    for y in frequency_y(field, frequency_ticks(field)):
        segments.append(((-w, y, 0), (-w - 0.05, y, 0)))
    for level in level_ticks(range_db):
        z = float(level_z(np.array(level), range_db))
        segments.append(((-w, d, z), (w, d, z)))
    ends = np.array(segments, dtype=np.float32).reshape(-1, 3)
    return np.column_stack([ends, np.tile(colour, (len(ends), 1))]).astype(np.float32)


def part_strip(field: Heightfield, colours: WaterfallColours) -> np.ndarray:
    """The frame structure as a strip along the front edge of the floor, a rectangle
    per stretch, cut at the window: triangles, (6 N, 7), position and colour."""
    inner, outer = -HALF_DEPTH, -HALF_DEPTH - STRIP_DEPTH
    vertices = []
    for begin, end, part in field.parts:
        left, right = (float(x) for x in time_x(field, np.array([begin, end])))
        if right <= left:
            continue  # outside the window
        colour = colours.part(part)
        for x, y in ((left, outer), (right, outer), (right, inner)):
            vertices.append((x, y, 0.0, *colour))
        for x, y in ((left, outer), (right, inner), (left, inner)):
            vertices.append((x, y, 0.0, *colour))
    return np.array(vertices, dtype=np.float32).reshape(-1, 7)


def time_ticks(field: Heightfield) -> tuple[np.ndarray, list[str]]:
    """Where along the time axis to mark, in ms of the capture, and what: ms for the
    sliding FFT, the number of the symbol (N for the null symbol) for symbol rows."""
    if field.symbols is None:
        origin = field.time_origin_ms
        low, high = (t - origin for t in field.window_ms)
        step = nice_step(high - low, 4)
        figures = np.arange(math.ceil(low / step) * step, high + step / 1000, step)
        decimals = max(0, -math.floor(math.log10(step)))
        labels = [f"{t:.{decimals}f}" for t in figures + 0.0]  # no "-0"
        return origin + figures, labels
    # Each symbol in the middle of what is seen of its FFT, beneath its block: a window
    # shorter than a symbol may hold no symbol's middle at all (1 ms against 1.25 ms).
    low, high = field.window_ms
    begins = np.maximum(field.fft_starts_ms, low)
    ends = np.minimum(field.fft_ends_ms, high)
    middles = (begins + ends) / 2
    inside = ends > begins
    step = max(1, int(nice_step(max(int(inside.sum()), 1), 6)))
    chosen = [
        i for i in np.flatnonzero(inside) if field.symbols[i] == 0 or field.symbols[i] % step == 0
    ]
    labels = ["N" if field.symbols[i] == 0 else str(int(field.symbols[i])) for i in chosen]
    return middles[chosen], labels


# --- drawing ---------------------------------------------------------------------------------


class WaterfallRenderer(OffscreenRenderer):
    """Draws a `WaterfallScene` seen from a `WaterfallCamera` into an image, see the
    module; nothing without a scene."""

    log_name = "Detail view"

    def _shaders(self) -> dict[str, tuple[str, str]]:
        return {
            "surface": (_SURFACE_VERTEX, _SURFACE_FRAGMENT),
            "lines": (LINES_VERTEX, LINES_FRAGMENT),
        }

    def _set_up_scene(self) -> Iterator[None]:
        with self._current():
            self._meshes["landscape"] = Mesh(np.empty((0, 3)), (("a_position", 3),))
            self._meshes["lines"] = Mesh(np.empty((0, 7)), LINES_ATTRIBUTES)
            self._meshes["parts"] = Mesh(np.empty((0, 7)), LINES_ATTRIBUTES)
        yield from ()  # all in one step

    def render(
        self,
        camera: WaterfallCamera,
        width: int,
        height: int,
        scene: WaterfallScene | None = None,
        line_width: float = 1.0,
        buffer: np.ndarray | None = None,
    ):
        if scene is None:
            return None
        return super().render(camera, width, height, scene, line_width, buffer)

    def _load(self, scene: WaterfallScene) -> None:
        drawn = self._scene
        if drawn is None or scene.field is not drawn.field or scene.range_db != drawn.range_db:
            grid = landscape_grid(scene.field, scene.range_db)
            self._meshes["landscape"].load(triangle_strip(grid))
        lines = grid_lines(scene.field, scene.colours.grid, scene.range_db)
        self._meshes["lines"].load(lines)
        self._meshes["parts"].load(part_strip(scene.field, scene.colours))

    def _background(self, scene: WaterfallScene) -> tuple[float, float, float]:
        return scene.colours.background

    def _paint(
        self,
        camera: WaterfallCamera,
        width: int,
        height: int,
        scene: WaterfallScene,
        line_width: float,
    ) -> None:
        gl = self._gl
        matrix = self._clip(camera.matrix(width / height))
        right, up, back = camera.axes()
        light = 0.55 * back + 0.65 * up - 0.52 * right  # from the upper left of the viewer
        stops = {f"u_c{i}": QVector3D(*colour) for i, colour in enumerate(scene.colours.stops)}

        gl.glDisable(GL_CULL_FACE)
        gl.glDisable(GL_BLEND)
        gl.glEnable(GL_DEPTH_TEST)
        self._draw(
            "surface",
            "landscape",
            GL_TRIANGLE_STRIP,
            u_matrix=matrix,
            u_light=QVector3D(*(light / np.linalg.norm(light))),
            u_camera=QVector3D(*camera.position()),
            u_height=HEIGHT,
            **stops,
        )
        gl.glEnable(GL_BLEND)
        gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self._draw("lines", "parts", GL_TRIANGLES, u_matrix=matrix)
        self._draw("lines", "lines", GL_LINES, u_matrix=matrix)
