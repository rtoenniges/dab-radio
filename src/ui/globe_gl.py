"""OpenGL for the 3D view: the Earth and the orbits, drawn into an image. Taken over from
the gnss-receiver project, without what belongs to a receiver there: the range spheres
around the satellites, their circles on the ground and the map of the ionosphere.

Drawn off screen in a thread of its own, as `ui.offscreen_gl` describes; the globe's
views share one thread (`GlobeRenderThread`) and so one context. Everything is in km,
Earth-fixed: x towards the meridian of Greenwich, z towards the north pole.

What is drawn, in this order:
- the background and the stars, which turn with the view;
- the Earth: the image of `ui.earth` on the WGS 84 ellipsoid, lit from beside the
  viewer, and on it the coasts, borders and grid as lines;
- the glow of the atmosphere around it;
- the orbits, hidden where the Earth is in front of them.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QObject
from PyQt6.QtGui import QVector3D, QVector4D
from PyQt6.QtOpenGL import QOpenGLTexture

from dab.gnss import WGS84_A, WGS84_F
from ui.earth import (
    EarthOutlines,
    ellipsoid_km,
    grid_segments,
    load_outlines,
    outline_segments,
    paint_surface,
)
from ui.offscreen_gl import (
    GL_BACK,
    GL_BLEND,
    GL_CULL_FACE,
    GL_DEPTH_TEST,
    GL_FRONT,
    GL_LINES,
    GL_ONE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINTS,
    GL_SRC_ALPHA,
    GL_TRIANGLES,
    LINES_ATTRIBUTES,
    LINES_FRAGMENT,
    LINES_VERTEX,
    Mesh,
    OffscreenRenderer,
    RenderThread,
    perspective,
    view_axes,
)

EARTH_RADIUS_KM = WGS84_A / 1000
# The ellipsoid stretched by this along its axis is a sphere.
_POLAR_STRETCH = 1 / (1 - WGS84_F)

FIELD_OF_VIEW_DEG = 40.0
# The view stays between 1100 km above the ground (the Earth's image blurs below) and
# six times the distance of the farthest satellites, from the point it turns around;
# around a satellite also at least 7500 km from the centre of the Earth.
CAMERA_DISTANCE_KM = (7_500.0, 260_000.0)
_MAX_CAMERA_LATITUDE_DEG = 89.0
# Nothing drawn is farther from the Earth's centre: the geostationary and inclined
# geosynchronous BeiDou satellites at 42 200 km, with room to spare.
_SCENE_RADIUS_KM = 45_000.0
# The centre of the view can be shifted this far off the centre of the Earth: as far
# as anything is drawn.
PAN_LIMIT_KM = _SCENE_RADIUS_KM

# The Earth's mesh has a vertex every 2°: a triangle's middle lies 1 km below the
# ellipsoid.
_EARTH_STEP_DEG = 2
# The Earth's image: one pixel is 20 km wide. Coasts and borders are lines on top and
# stay sharp; 4096 pixels would take twice as long to paint (0.16 s instead of 0.09 s on
# the PC, measured 07.10.2026), without letting go of the GIL.
SURFACE_WIDTH = 2048
# The glow of the atmosphere ends at this many Earth radii.
_ATMOSPHERE_SCALE = 1.05
_STARS = 1800

_BACKGROUND_VERTEX = """
attribute vec2 a_position;
varying vec2 v_position;
void main() {
    v_position = a_position;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""
_BACKGROUND_FRAGMENT = """
varying vec2 v_position;
void main() {
    float r = length(v_position * vec2(1.0, 0.75));
    vec3 colour = mix(vec3(0.050, 0.075, 0.130), vec3(0.008, 0.012, 0.024),
                      smoothstep(0.0, 1.25, r));
    gl_FragColor = vec4(colour, 1.0);
}
"""

_STARS_VERTEX = """
attribute vec4 a_star;  // direction and brightness
uniform mat4 u_matrix;
varying float v_brightness;
void main() {
    v_brightness = a_star.w;
    gl_Position = (u_matrix * vec4(a_star.xyz, 1.0)).xyww;
}
"""
_STARS_FRAGMENT = """
varying float v_brightness;
void main() {
    gl_FragColor = vec4(vec3(0.80, 0.88, 1.0) * v_brightness, 1.0);
}
"""

_EARTH_VERTEX = """
attribute vec3 a_position;
attribute vec3 a_normal;
attribute vec2 a_uv;
uniform mat4 u_matrix;
uniform float u_scale;
varying vec3 v_world;
varying vec3 v_normal;
varying vec2 v_uv;
void main() {
    v_world = a_position * u_scale;
    v_normal = a_normal;
    v_uv = a_uv;
    gl_Position = u_matrix * vec4(v_world, 1.0);
}
"""
_EARTH_FRAGMENT = """
uniform sampler2D u_surface;
uniform vec3 u_camera;
uniform vec3 u_light;
varying vec3 v_world;
varying vec3 v_normal;
varying vec2 v_uv;
void main() {
    vec3 n = normalize(v_normal);
    vec3 v = normalize(u_camera - v_world);
    vec3 colour = texture2D(u_surface, v_uv).rgb * (0.38 + 0.78 * max(dot(n, u_light), 0.0));
    // The haze of the atmosphere, thickest where the view grazes the ground.
    float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
    colour += vec3(0.20, 0.45, 0.90) * rim * 0.65;
    gl_FragColor = vec4(colour, 1.0);
}
"""

# The glow around the Earth: the far side of a slightly larger sphere, brightest where
# the line of sight just passes the ground, fading to nothing at that sphere's rim.
_ATMOSPHERE_FRAGMENT = """
uniform vec3 u_camera;
uniform float u_edge;
varying vec3 v_world;
varying vec3 v_normal;
varying vec2 v_uv;
void main() {
    float facing = abs(dot(normalize(v_normal), normalize(u_camera - v_world)));
    float t = clamp(facing / u_edge, 0.0, 1.0);
    gl_FragColor = vec4(vec3(0.22, 0.48, 1.0) * pow(t, 2.2) * 0.85, 1.0);
}
"""

# Lines on the ground. The Earth is convex: a place on it is seen exactly if its
# surface faces the viewer, so no depth test is needed (and none could tell a line on
# the surface from the surface).
_SURFACE_LINES_VERTEX = """
attribute vec3 a_position;
uniform mat4 u_matrix;
uniform vec3 u_camera;
varying float v_facing;
void main() {
    vec3 n = normalize(a_position * vec3(1.0, 1.0, POLAR_STRETCH * POLAR_STRETCH));
    v_facing = dot(n, normalize(u_camera - a_position));
    gl_Position = u_matrix * vec4(a_position, 1.0);
}
"""
_SURFACE_LINES_FRAGMENT = """
uniform vec4 u_colour;
varying float v_facing;
void main() {
    if (v_facing <= 0.0) discard;
    gl_FragColor = vec4(u_colour.rgb, u_colour.a * smoothstep(0.0, 0.25, v_facing));
}
"""


@dataclass
class Camera:
    """Looks, north up, at the point it turns around (`pivot_km`), from the direction of
    a place on the ground: at the centre of the Earth, then from above that place, or at a
    satellite. The view can be shifted from that point in its plane (`pan`). The shift
    stays in that plane when the view turns, and the point stays in the middle when the
    view comes closer."""

    latitude_deg: float
    longitude_deg: float
    distance_km: float  # from the centre of the view
    right_km: float = 0.0  # the centre of the view, from the pivot
    up_km: float = 0.0
    pivot_km: tuple[float, float, float] = (0.0, 0.0, 0.0)  # Earth-fixed

    def __post_init__(self) -> None:
        self.rotate(0.0, 0.0)
        self.zoom(1.0)
        self.pan(0.0, 0.0)

    @property
    def around_earth(self) -> bool:
        """Whether the view turns around the centre of the Earth."""
        return not any(self.pivot_km)

    def rotate(self, longitude_deg: float, latitude_deg: float) -> None:
        longitude, latitude, before = self.longitude_deg, self.latitude_deg, self.position()
        self.longitude_deg = (self.longitude_deg + longitude_deg + 180.0) % 360.0 - 180.0
        limit = _MAX_CAMERA_LATITUDE_DEG
        self.latitude_deg = min(max(self.latitude_deg + latitude_deg, -limit), limit)
        if not self._clear_of_earth(before):
            self.longitude_deg, self.latitude_deg = longitude, latitude

    def zoom(self, factor: float) -> None:
        distance, before = self.distance_km, self.position()
        low, high = CAMERA_DISTANCE_KM
        self.distance_km = min(max(self.distance_km * factor, low), high)
        if not self._clear_of_earth(before):
            self.distance_km = distance

    def _clear_of_earth(self, before_km: np.ndarray) -> bool:
        """Around a satellite, the viewer could turn or come closer into the Earth: it
        keeps the nearest distance from its centre, or at least does not come closer."""
        if self.around_earth:
            return True  # the distance keeps it there
        reach = np.linalg.norm(self.position())
        return reach >= min(CAMERA_DISTANCE_KM[0], np.linalg.norm(before_km))

    def pan(self, right_km: float, up_km: float) -> None:
        """Shifts the centre of the view along the right and up axes of the view."""
        self.pan_to(self.right_km + right_km, self.up_km + up_km)

    def pan_to(self, right: float, up: float) -> None:
        """Puts the centre of the view this far right of and above the pivot, in km along
        the axes of the view."""
        length = math.hypot(right, up)
        if length > PAN_LIMIT_KM:
            right, up = right * PAN_LIMIT_KM / length, up * PAN_LIMIT_KM / length
        self.right_km, self.up_km = right, up

    def direction(self) -> np.ndarray:
        """From the centre of the Earth towards the viewer, unit length."""
        lat, lon = math.radians(self.latitude_deg), math.radians(self.longitude_deg)
        return np.array(
            [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]
        )

    def centre(self) -> np.ndarray:
        """The point the view is centred on, Earth-fixed km."""
        right, up, _ = self.axes()
        return np.array(self.pivot_km) + self.right_km * right + self.up_km * up

    def position(self) -> np.ndarray:
        """Of the viewer, Earth-fixed km."""
        return self.centre() + self.distance_km * self.direction()

    def axes(self) -> np.ndarray:
        """Rows right, up and back (towards the viewer) of the view, Earth-fixed."""
        return view_axes(self.direction())

    def projection(self, aspect: float, near: float, far: float) -> np.ndarray:
        return perspective(FIELD_OF_VIEW_DEG, aspect, near, far)

    def matrix(self, aspect: float) -> np.ndarray:
        """Earth-fixed km to clip coordinates."""
        view = np.eye(4)
        view[:3, :3] = axes = self.axes()
        viewer = self.position()
        view[:3, 3] = -(axes @ viewer)
        # Near enough for an orbit right in front of the viewer or the satellite it turns
        # around, far enough for one behind the Earth: everything lies around the centre
        # of the Earth.
        near = 0.05 * min(self.distance_km, np.linalg.norm(viewer) - EARTH_RADIUS_KM)
        far = axes[2] @ viewer + _SCENE_RADIUS_KM
        return self.projection(aspect, near, far) @ view

    def sky_matrix(self, aspect: float) -> np.ndarray:
        """Directions to clip coordinates: the stars turn with the view but do not come
        closer."""
        view = np.eye(4)
        view[:3, :3] = self.axes()
        return self.projection(aspect, 0.1, 10.0) @ view

    def km_per_pixel(self, height_px: float) -> float:
        """What a pixel covers on the ground below the viewer, or around a satellite at
        the satellite."""
        visible = 2 * (self.distance_km - (EARTH_RADIUS_KM if self.around_earth else 0.0))
        return visible * math.tan(math.radians(FIELD_OF_VIEW_DEG) / 2) / height_px


def hidden_by_earth(viewer_km: np.ndarray, points_km: np.ndarray) -> np.ndarray:
    """Whether the Earth stands between the viewer and each of the points (N, 3). Points
    on the ground count as seen on the near side: the Earth is taken 200 m smaller, which
    shows them up to half a degree beyond the horizon."""
    stretch = np.array([1.0, 1.0, _POLAR_STRETCH])  # the ellipsoid becomes a sphere
    start = viewer_km * stretch
    way = points_km * stretch - start
    # The point of the way closest to the centre, within the way.
    t = np.clip(-(way @ start) / np.einsum("ij,ij->i", way, way), 0.0, 1.0)
    closest = start + way * t[:, None]
    return np.linalg.norm(closest, axis=1) < EARTH_RADIUS_KM - 0.2


def _triangles(grid: np.ndarray) -> np.ndarray:
    """Two triangles per cell of a grid of vertices (rows, columns, K), counter-clockwise
    seen from outside when rows run north and columns east: (N, K) as float32."""
    corners = [
        grid[:-1, :-1],
        grid[:-1, 1:],
        grid[1:, 1:],
        grid[:-1, :-1],
        grid[1:, 1:],
        grid[1:, :-1],
    ]
    return np.stack(corners, axis=2).reshape(-1, grid.shape[-1]).astype(np.float32)


def _directions(latitude_deg: np.ndarray, longitude_deg: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(latitude_deg), np.radians(longitude_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def earth_mesh(step_deg: int = _EARTH_STEP_DEG) -> np.ndarray:
    """The ellipsoid as triangles: position (km), normal and place in the image of
    `ui.earth.paint_surface` per vertex, (N, 8)."""
    lat, lon = np.meshgrid(
        np.arange(-90.0, 90.0 + step_deg, step_deg),
        np.arange(-180.0, 180.0 + step_deg, step_deg),
        indexing="ij",
    )
    uv = np.stack([(lon + 180.0) / 360.0, (90.0 - lat) / 180.0], axis=-1)
    return _triangles(np.concatenate([ellipsoid_km(lat, lon), _directions(lat, lon), uv], axis=-1))


def star_field(count: int = _STARS) -> np.ndarray:
    """Directions and brightness of made-up stars, (N, 4): many faint ones, few bright."""
    rng = np.random.default_rng(1575)
    directions = rng.standard_normal((count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    brightness = 0.12 + 0.88 * rng.random(count) ** 4
    return np.column_stack([directions, brightness]).astype(np.float32)


class GlobeRenderer(OffscreenRenderer):
    """Draws the Earth and the orbits into an image, see the module. Its scene is the
    orbits: position (km) and colour with opacity per end, (2 N, 7), hidden where the
    Earth is in front; its camera a `Camera`.

    Setting up has, after the context and the shaders (`OffscreenRenderer`), three steps
    of its own. Measured in the gnss-receiver on 02.10.2026: painting the Earth's image
    0.18 s (here 0.09 s, `SURFACE_WIDTH`), the meshes 0.10 s; Qt does not let go of the
    GIL in any of it.
    """

    log_name = "3D view"
    # The version of GLSL and the constants of this module.
    shader_header = f"""#version 120
#define POLAR_STRETCH {_POLAR_STRETCH:.9f}
"""

    def __init__(self, outlines: EarthOutlines | None = None) -> None:
        super().__init__()
        self._outlines = outlines
        self._line_ranges: dict[str, tuple[int, int]] = {}  # in the mesh "surface_lines"

    def _shaders(self) -> dict[str, tuple[str, str]]:
        return {
            "background": (_BACKGROUND_VERTEX, _BACKGROUND_FRAGMENT),
            "stars": (_STARS_VERTEX, _STARS_FRAGMENT),
            "earth": (_EARTH_VERTEX, _EARTH_FRAGMENT),
            "atmosphere": (_EARTH_VERTEX, _ATMOSPHERE_FRAGMENT),
            "surface_lines": (_SURFACE_LINES_VERTEX, _SURFACE_LINES_FRAGMENT),
            "lines": (LINES_VERTEX, LINES_FRAGMENT),
        }

    def _set_up_scene(self) -> Iterator[None]:
        outlines = self._outlines if self._outlines is not None else load_outlines()
        image = paint_surface(outlines, SURFACE_WIDTH)
        yield

        with self._current():
            self._textures["surface"] = texture = QOpenGLTexture(image)
            texture.setMinMagFilters(
                QOpenGLTexture.Filter.LinearMipMapLinear, QOpenGLTexture.Filter.Linear
            )
            texture.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
            texture.setMaximumAnisotropy(8.0)
        del image
        yield

        corners = [[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]]
        earth = (("a_position", 3), ("a_normal", 3), ("a_uv", 2))
        surface_lines = {
            "grid": grid_segments(),
            "borders": outline_segments(outlines.borders, closed=False),
            "coasts": outline_segments(outlines.land + outlines.lakes, closed=True),
        }
        first = 0
        for name, segments in surface_lines.items():
            self._line_ranges[name] = (first, len(segments))
            first += len(segments)
        with self._current():
            self._meshes["background"] = Mesh(np.array(corners), (("a_position", 2),))
            self._meshes["stars"] = Mesh(star_field(), (("a_star", 4),))
            self._meshes["earth"] = Mesh(earth_mesh(), earth)
            self._meshes["lines"] = Mesh(np.empty((0, 7)), LINES_ATTRIBUTES)
            self._meshes["surface_lines"] = Mesh(
                np.concatenate(list(surface_lines.values())), (("a_position", 3),)
            )

    def _load(self, lines: np.ndarray | None) -> None:
        self._meshes["lines"].load(lines if lines is not None else np.empty((0, 7)))

    def _paint(
        self, camera: Camera, width: int, height: int, lines: np.ndarray | None, line_width: float
    ) -> None:
        gl = self._gl
        aspect = width / height
        matrix = self._clip(camera.matrix(aspect))
        position = QVector3D(*camera.position())
        right, up, back = camera.axes()

        gl.glDisable(GL_DEPTH_TEST)
        gl.glDisable(GL_CULL_FACE)
        gl.glDisable(GL_BLEND)
        self._draw("background", "background", GL_TRIANGLES)

        gl.glEnable(GL_BLEND)
        gl.glBlendFunc(GL_ONE, GL_ONE)
        gl.glPointSize(1.5 * line_width)
        self._draw("stars", "stars", GL_POINTS, u_matrix=self._clip(camera.sky_matrix(aspect)))

        gl.glDisable(GL_BLEND)
        gl.glEnable(GL_DEPTH_TEST)
        gl.glEnable(GL_CULL_FACE)
        gl.glCullFace(GL_BACK)
        texture = self._textures["surface"]
        texture.bind(0)
        light = 0.72 * back + 0.50 * up - 0.48 * right  # from the upper left of the viewer
        self._draw(
            "earth",
            "earth",
            GL_TRIANGLES,
            u_matrix=matrix,
            u_scale=1.0,
            u_surface=0,
            u_camera=position,
            u_light=QVector3D(*(light / np.linalg.norm(light))),
        )
        texture.release()

        gl.glDisable(GL_DEPTH_TEST)
        gl.glEnable(GL_BLEND)
        gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        for name, colour in (
            ("grid", QVector4D(0.75, 0.85, 1.0, 0.10)),
            ("borders", QVector4D(0.85, 0.92, 0.88, 0.30)),
            ("coasts", QVector4D(0.80, 0.93, 0.90, 0.60)),
        ):
            self._draw(
                "surface_lines",
                "surface_lines",
                GL_LINES,
                self._line_ranges[name],
                u_matrix=matrix,
                u_camera=position,
                u_colour=colour,
            )

        gl.glEnable(GL_DEPTH_TEST)
        gl.glDepthMask(False)
        gl.glBlendFunc(GL_ONE, GL_ONE)
        gl.glCullFace(GL_FRONT)
        self._draw(
            "atmosphere",
            "earth",
            GL_TRIANGLES,
            u_matrix=matrix,
            u_scale=_ATMOSPHERE_SCALE,
            u_camera=position,
            u_edge=math.sqrt(1 - 1 / _ATMOSPHERE_SCALE**2),
        )

        gl.glDisable(GL_CULL_FACE)
        gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self._draw("lines", "lines", GL_LINES, u_matrix=matrix)


class GlobeRenderThread(RenderThread):
    """The render thread of the globe's views (`RenderThread` with a `GlobeRenderer`)."""

    def __init__(
        self, renderer: OffscreenRenderer | None = None, parent: QObject | None = None
    ) -> None:
        super().__init__(renderer if renderer is not None else GlobeRenderer(), parent)
