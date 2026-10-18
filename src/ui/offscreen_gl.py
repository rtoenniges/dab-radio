"""OpenGL off screen, in a thread of its own, for the 3D views: the globe (`ui.globe_gl`)
and the landscape of the detail view (`ui.waterfall_gl`). Each of them is a renderer
(`OffscreenRenderer`) that brings its shaders, its meshes and how it draws a scene;
everything around that is here.

The scene is drawn off screen into a framebuffer and handed on as a QImage, which the
view paints like any other picture and writes its marks and labels on. A QOpenGLWidget
would instead make Qt compose the whole window, every page of it, with OpenGL from the
moment it is created; this way OpenGL is first touched when the view is shown, and
nothing else changes if it is missing.

Drawing runs in a thread of its own (`RenderThread`), and the image comes back through
`glReadPixels` called by ctypes: that call waits for the graphics card to finish the
frame, and ctypes lets go of the GIL meanwhile, which PyQt never does (its `toImage` held
it for the whole wait; on the Raspberry Pi the signal processing then waited for the GIL
most of the time while the globe was turned, and the audio ran dry). The GUI thread only
hands over the camera and paints the last finished image, so a gesture never waits for
the graphics card either. OpenGL gives the pixels bottom row first; the scene is
therefore drawn upside down (`_FLIP`, with the winding of the faces turned around) and
comes out upright, without a copy.

OpenGL 2.1 with GLSL 1.20 is asked for, which every driver has, through Qt's own
classes: no PyOpenGL, and nothing of GLX or X11. The context comes from Qt's platform
plugin: under Wayland (the Raspberry Pi runs the radio in the compositor cage) through
EGL. The Qt of the PyQt6 wheel for aarch64 (PyQt6-Qt6 6.11.2) is built for desktop
OpenGL, its Wayland plugin binds EGL's desktop OpenGL API and makes the offscreen
surface an EGL pbuffer, or no surface at all where EGL_KHR_surfaceless_context is
offered (as Mesa does). Under XWayland (platform "xcb") Qt would use GLX. The log names
the platform with the driver.
"""

from __future__ import annotations

import copy
import ctypes
import logging
import math
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QObject, QSize, QThread, pyqtSignal
from PyQt6.QtGui import (
    QGuiApplication,
    QImage,
    QMatrix4x4,
    QOffscreenSurface,
    QOpenGLContext,
    QSurfaceFormat,
)
from PyQt6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLFramebufferObject,
    QOpenGLFramebufferObjectFormat,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLTexture,
    QOpenGLVersionFunctionsFactory,
    QOpenGLVersionProfile,
)

from i18n import tr

logger = logging.getLogger(__name__)

GL_POINTS = 0x0000
GL_LINES = 0x0001
GL_TRIANGLES = 0x0004
GL_TRIANGLE_STRIP = 0x0005
GL_DEPTH_BUFFER_BIT = 0x0100
GL_COLOR_BUFFER_BIT = 0x4000
GL_ONE = 1
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_FRONT = 0x0404
GL_BACK = 0x0405
GL_CULL_FACE = 0x0B44
GL_DEPTH_TEST = 0x0B71
GL_BLEND = 0x0BE2
_GL_FLOAT = 0x1406
_GL_UNSIGNED_BYTE = 0x1401
_GL_RENDERER = 0x1F01
_GL_VERSION = 0x1F02
_GL_CW = 0x0900
_GL_CCW = 0x0901
_GL_BGRA = 0x80E1
# Clip coordinates upside down, see the module.
_FLIP = np.diag([1.0, -1.0, 1.0, 1.0])
# OpenGL's functions take the calling convention of the system's libraries.
_FUNCTION_TYPE = ctypes.WINFUNCTYPE if sys.platform == "win32" else ctypes.CFUNCTYPE

# Lines with a position and a colour (with opacity) at each end, as both views draw them.
LINES_VERTEX = """
attribute vec3 a_position;
attribute vec4 a_colour;
uniform mat4 u_matrix;
varying vec4 v_colour;
void main() {
    v_colour = a_colour;
    gl_Position = u_matrix * vec4(a_position, 1.0);
}
"""
LINES_FRAGMENT = """
varying vec4 v_colour;
void main() {
    gl_FragColor = v_colour;
}
"""
LINES_ATTRIBUTES = (("a_position", 3), ("a_colour", 4))


# --- cameras -------------------------------------------------------------------------------


def view_axes(back: np.ndarray) -> np.ndarray:
    """Rows right, up and back (towards the viewer) of a view from the direction `back`
    (unit length), with z upwards on the screen."""
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    return np.array([right, np.cross(back, right), back])


def perspective(field_of_view_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """The projection of a camera that sees `field_of_view_deg` from top to bottom."""
    f = 1.0 / math.tan(math.radians(field_of_view_deg) / 2)
    return np.array(
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )


def project(matrix: np.ndarray, points: np.ndarray, width: float, height: float) -> np.ndarray:
    """Pixel coordinates (N, 2) of points of the scene (N, 3), seen through a camera's
    `matrix`; NaN behind the viewer."""
    clip = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    w = np.where(clip[:, 3] > 1e-9, clip[:, 3], np.nan)
    x = (clip[:, 0] / w + 1) / 2 * width
    y = (1 - clip[:, 1] / w) / 2 * height
    return np.column_stack([x, y])


# --- the renderer --------------------------------------------------------------------------


def _qmatrix(matrix: np.ndarray) -> QMatrix4x4:
    return QMatrix4x4(*(float(v) for v in matrix.flatten()))


def _read_pixels_function(context: QOpenGLContext, name: str):
    """`glReadPixels` of this (current) context as a ctypes function, which lets go of
    the GIL while it waits for the graphics card; None where the platform gives out no
    address, then Qt reads the image back and holds the GIL meanwhile."""
    address = int(context.getProcAddress(b"glReadPixels"))
    if not address:
        logger.warning("%s: no address of glReadPixels, the GIL is held while reading back", name)
        return None
    signature = _FUNCTION_TYPE(
        None,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
    )
    return signature(address)


class Mesh:
    """Vertices in a buffer of the graphics card, with what each of them consists of."""

    def __init__(self, vertices: np.ndarray, attributes: Sequence[tuple[str, int]]) -> None:
        self.attributes = attributes  # name in the shader and number of floats
        self.stride = vertices.shape[1] * 4
        self.count = 0
        self.buffer = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
        self.buffer.create()
        self.load(vertices)

    def load(self, vertices: np.ndarray) -> None:
        data = np.ascontiguousarray(vertices, dtype=np.float32)
        self.count = len(data)
        self.buffer.bind()
        self.buffer.allocate(data.tobytes(), data.nbytes)
        self.buffer.release()


class OffscreenRenderer:
    """Draws a scene into an image. A subclass says what is drawn: its shaders
    (`_shaders`), the rest of its setting up (`_set_up_scene`), how a new scene goes into
    the graphics card (`_load`) and how it is drawn (`_paint`); what a scene and a camera
    are is its own business.

    OpenGL is set up when first needed, in steps (`prepare`): creating the context can
    take almost half a second (it loads the graphics driver), and Qt does not let go of
    the GIL in that, nor in what a subclass adds. Done at once while a stream runs, the
    processing would stand still that long. Between the steps the caller lets the event
    loop run. If a step fails, `error` says why and `render` returns None from then on.

    Drawn from `RenderThread`, the context is made in the GUI thread (`open`) and handed
    to the thread (`move_to_thread`), which does the rest of the steps and the drawing;
    all in one thread, `prepare` does it all.
    """

    # Names the renderer in the log.
    log_name = "3D view"
    # Before every shader: the version of GLSL, and the constants a subclass adds.
    shader_header = "#version 120\n"

    def __init__(self) -> None:
        self._ready: bool | None = None  # None: not set up yet
        self._steps: Iterator[None] | None = None  # of the setting up, once begun
        self.error = ""  # shown in the view, translated
        self._context: QOpenGLContext | None = None
        self._surface: QOffscreenSurface | None = None
        self._gl = None
        self._programs: dict[str, QOpenGLShaderProgram] = {}
        self._meshes: dict[str, Mesh] = {}
        self._textures: dict[str, QOpenGLTexture] = {}
        self._scene: object | None = None  # what the meshes hold, see `render`
        self._framebuffer: QOpenGLFramebufferObject | None = None
        self._resolved: QOpenGLFramebufferObject | None = None  # without multisampling
        self._read_pixels = None  # glReadPixels through ctypes, see `_read_back`
        self._flip = False  # drawn upside down for the readback through ctypes

    # --- setting up -----------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready is True

    def prepare(self) -> bool:
        """Does the next step of setting OpenGL up. False once nothing is left to do:
        then the renderer is ready, or `error` is set."""
        if self._ready is not None:
            return False
        if self._steps is None:
            self._steps = self._set_up()
        if next(self._steps, self) is self:  # the generator has ended
            self._steps = None
            if self._ready is None:
                self._ready = True
        return self._ready is None

    def _fail(self, reason: str, shown: str) -> None:
        """Gives up for good: `reason` for the log, `shown` for the view."""
        logger.warning(
            "%s (Qt platform %s): %s", self.log_name, QGuiApplication.platformName(), reason
        )
        self.error = tr(
            "The 3D view needs OpenGL 2.1, which could not be set up on this computer ({reason})."
        ).format(reason=shown)
        self._ready = False

    def open(self) -> bool:
        """Makes the OpenGL context and the surface it draws on, in the GUI thread (Qt
        wants the surface made there); False with `error` set if that fails. The first
        step of `prepare` where everything is in one thread."""
        if self._context is not None:
            return True
        if self._ready is False:
            return False
        requested = QSurfaceFormat()
        requested.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)  # not OpenGL ES
        requested.setVersion(2, 1)
        context = QOpenGLContext()
        context.setFormat(requested)
        if not context.create():
            self._fail("no OpenGL context", tr("no OpenGL context"))
            return False
        surface = QOffscreenSurface()
        surface.setFormat(context.format())
        surface.create()
        if not surface.isValid() or not context.makeCurrent(surface):
            self._fail(
                "the OpenGL context cannot be made current",
                tr("the OpenGL context cannot be activated"),
            )
            return False
        context.doneCurrent()
        self._context, self._surface = context, surface
        return True

    def move_to_thread(self, thread: QThread) -> None:
        """Hands the context to the thread that draws from now on; to be called from
        the thread the context belongs to, the GUI thread after `open`."""
        if self._context is not None:
            self._context.moveToThread(thread)

    @contextmanager
    def _current(self) -> Iterator[None]:
        """Makes the context current for a step of setting up."""
        self._context.makeCurrent(self._surface)
        try:
            yield
        finally:
            self._context.doneCurrent()

    def _set_up(self) -> Iterator[None]:
        """The steps of setting up: the context, the functions of OpenGL and the shaders,
        then those of the subclass; the context is current only within a step."""
        if not self.open():
            return
        yield

        with self._current():
            profile = QOpenGLVersionProfile()
            profile.setVersion(2, 1)
            self._gl = gl = QOpenGLVersionFunctionsFactory.get(profile, self._context)
            if gl is None:
                return self._fail("no OpenGL 2.1 functions", tr("no functions of OpenGL 2.1"))
            self._read_pixels = _read_pixels_function(self._context, self.log_name)
            self._flip = self._read_pixels is not None
            logger.info(
                "%s: %s, OpenGL %s, Qt platform %s, read back by %s",
                self.log_name,
                gl.glGetString(_GL_RENDERER),
                gl.glGetString(_GL_VERSION),
                QGuiApplication.platformName(),
                "ctypes (GIL released)" if self._flip else "Qt (GIL held)",
            )
            for name, (vertex, fragment) in self._shaders().items():
                program = QOpenGLShaderProgram()
                if not (
                    program.addShaderFromSourceCode(
                        QOpenGLShader.ShaderTypeBit.Vertex, self.shader_header + vertex
                    )
                    and program.addShaderFromSourceCode(
                        QOpenGLShader.ShaderTypeBit.Fragment, self.shader_header + fragment
                    )
                    and program.link()
                ):
                    log = program.log().strip()
                    return self._fail(
                        f"shader '{name}': {log}",
                        tr("shader “{name}”: {log}").format(name=name, log=log),
                    )
                self._programs[name] = program
        yield

        yield from self._set_up_scene()

    def _shaders(self) -> dict[str, tuple[str, str]]:
        """The programs by name: the sources of their vertex and fragment shaders."""
        raise NotImplementedError

    def _set_up_scene(self) -> Iterator[None]:
        """The rest of setting up, textures and meshes, in steps of its own; a step that
        needs the context makes it current (`_current`)."""
        raise NotImplementedError

    def release(self) -> None:
        """Frees what the graphics card holds and drops the context; needed before the
        application ends, while the context still exists. All in one thread; the render
        thread does it in two halves (`release_gl`, `close`)."""
        self.release_gl()
        self.close()

    def release_gl(self) -> None:
        """Frees what the graphics card holds, in the thread the context belongs to, and
        hands the context back to the GUI thread."""
        context = self._context
        if context is not None and self._surface is not None:
            context.makeCurrent(self._surface)
            self._framebuffer = None
            self._resolved = None
            self._textures.clear()
            self._meshes.clear()
            self._programs.clear()
            context.doneCurrent()
            application = QCoreApplication.instance()
            if application is not None and context.thread() is not application.thread():
                context.moveToThread(application.thread())
        self._gl = None
        self._read_pixels = None
        self._scene = None
        self._steps = None

    def close(self) -> None:
        """Drops context and surface, in the GUI thread; set up again if asked to draw
        again."""
        self._context = None
        self._surface = None
        if self._ready:
            self._ready = None

    def _framebuffer_of(self, width: int, height: int) -> QOpenGLFramebufferObject | None:
        framebuffer = self._framebuffer
        if framebuffer is not None and framebuffer.size() == QSize(width, height):
            return framebuffer
        self._framebuffer = None  # freed before the new one is made
        # Multisampled only where it can be resolved into a plain one for reading.
        choices = (4, 0) if QOpenGLFramebufferObject.hasOpenGLFramebufferBlit() else (0,)
        for samples in choices:
            form = QOpenGLFramebufferObjectFormat()
            form.setAttachment(QOpenGLFramebufferObject.Attachment.Depth)
            form.setSamples(samples)
            framebuffer = QOpenGLFramebufferObject(QSize(width, height), form)
            if framebuffer.isValid():
                self._framebuffer = framebuffer
                return framebuffer
        return None

    def _resolved_of(self, width: int, height: int) -> QOpenGLFramebufferObject:
        """A framebuffer without multisampling to resolve the drawn one into: pixels can
        only be read from such a one."""
        resolved = self._resolved
        if resolved is None or resolved.size() != QSize(width, height):
            self._resolved = resolved = QOpenGLFramebufferObject(QSize(width, height))
        return resolved

    # --- drawing --------------------------------------------------------------------

    def render(
        self,
        camera,
        width: int,
        height: int,
        scene: object | None = None,
        line_width: float = 1.0,
        buffer: np.ndarray | None = None,
    ) -> QImage | None:
        """The scene seen from `camera` as an image of `width` by `height` pixels, or
        None without OpenGL. A scene other than the one drawn last (told apart by
        identity) goes into the graphics card first (`_load`); the same one again does
        not. What is left of setting OpenGL up is done first, at once. `buffer`, uint8 of
        (height, width, 4), receives the pixels: the image shows it without a copy, so it
        must outlive the image; without one the image keeps a buffer of its own."""
        while self.prepare():
            pass
        if not self._ready or width < 1 or height < 1:
            return None
        if not self._context.makeCurrent(self._surface):
            return self._fail("the OpenGL context was lost", tr("the OpenGL context was lost"))
        framebuffer = self._framebuffer_of(width, height)
        if framebuffer is None:
            self._context.doneCurrent()
            return self._fail("no framebuffer", tr("no framebuffer"))
        if scene is not self._scene:
            self._load(scene)
            self._scene = scene

        gl = self._gl
        framebuffer.bind()
        gl.glViewport(0, 0, width, height)
        gl.glDepthMask(True)
        gl.glColorMask(True, True, True, True)
        gl.glClearColor(*self._background(scene), 1.0)
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # The image stays opaque: blended, lines and glows would lower its alpha, and
        # whatever the view lies on would show through (white in the light scheme).
        gl.glColorMask(True, True, True, False)
        gl.glLineWidth(line_width)
        gl.glFrontFace(_GL_CW if self._flip else _GL_CCW)  # upside down they wind the other way
        self._paint(camera, width, height, scene, line_width)
        gl.glDepthMask(True)
        gl.glColorMask(True, True, True, True)
        gl.glDisable(GL_BLEND)
        gl.glDisable(GL_DEPTH_TEST)

        framebuffer.release()
        image = self._read_back(framebuffer, width, height, buffer)
        self._context.doneCurrent()
        return image

    def _load(self, scene: object | None) -> None:
        """Puts a new scene into the meshes; the context is current."""
        raise NotImplementedError

    def _background(self, scene: object | None) -> tuple[float, float, float]:
        """What the image is cleared to before drawing."""
        return (0.0, 0.0, 0.0)

    def _paint(self, camera, width: int, height: int, scene: object | None, line_width: float):
        """Draws the scene into the framebuffer, which is bound and cleared; the state
        of depth, blending and culling it leaves behind does not matter."""
        raise NotImplementedError

    def _clip(self, matrix: np.ndarray) -> QMatrix4x4:
        """A camera's matrix (to clip coordinates) for the shaders: upside down where the
        image is read back through ctypes (see the module)."""
        return _qmatrix(_FLIP @ matrix if self._flip else matrix)

    def _read_back(
        self,
        framebuffer: QOpenGLFramebufferObject,
        width: int,
        height: int,
        buffer: np.ndarray | None,
    ) -> QImage:
        """The pixels of the framebuffer as an image, see the module."""
        if self._read_pixels is None:
            return framebuffer.toImage()  # drawn the usual way up for this
        source = framebuffer
        if framebuffer.format().samples() > 0:
            source = self._resolved_of(width, height)
            QOpenGLFramebufferObject.blitFramebuffer(source, framebuffer)
        if buffer is None or buffer.shape != (height, width, 4):
            buffer = np.empty((height, width, 4), dtype=np.uint8)
        source.bind()
        # Waits for the graphics card without the GIL; BGRA is what Format_RGB32 holds.
        self._read_pixels(0, 0, width, height, _GL_BGRA, _GL_UNSIGNED_BYTE, buffer.ctypes.data)
        source.release()
        image = QImage(
            sip.voidptr(buffer.ctypes.data), width, height, width * 4, QImage.Format.Format_RGB32
        )
        image.buffer = buffer  # keeps the pixels as long as the image
        return image

    def _draw(
        self,
        program_name: str,
        mesh_name: str,
        mode: int,
        vertices: tuple[int, int] | None = None,
        **uniforms,
    ) -> None:
        """Draws the mesh (or `vertices` of it: the first and how many) with the program
        and these values of its uniforms."""
        program, mesh = self._programs[program_name], self._meshes[mesh_name]
        first, count = vertices if vertices is not None else (0, mesh.count)
        if not count:
            return
        program.bind()
        for name, value in uniforms.items():
            program.setUniformValue(name, value)
        mesh.buffer.bind()
        offset = 0
        locations = []
        for name, size in mesh.attributes:
            location = program.attributeLocation(name)
            if location >= 0:  # a shader that does not use it has optimised it away
                program.enableAttributeArray(location)
                program.setAttributeBuffer(location, _GL_FLOAT, offset, size, mesh.stride)
                locations.append(location)
            offset += size * 4
        self._gl.glDrawArrays(mode, first, count)
        for location in locations:
            program.disableAttributeArray(location)
        mesh.buffer.release()
        program.release()


# --- the render thread -------------------------------------------------------------------

# The thread draws no more often than this: the display shows 60 frames a second at
# most, and the GUI thread has to paint each one it is handed.
FRAME_INTERVAL_SECONDS = 1 / 30


class _BufferPool:
    """The pixel buffers of the frames: the thread draws into a free one while the view
    shows another. A frame gives its buffer back when the view lets it go
    (`RenderFrame.release`)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._free: list[np.ndarray] = []

    def take(self, width: int, height: int) -> np.ndarray:
        with self._lock:
            for index, buffer in enumerate(self._free):
                if buffer.shape == (height, width, 4):
                    return self._free.pop(index)
            self._free.clear()  # of another size: the view was resized
        return np.empty((height, width, 4), dtype=np.uint8)

    def give_back(self, buffer: np.ndarray) -> None:
        with self._lock:
            if len(self._free) < 3:
                self._free.append(buffer)


@dataclass(eq=False)
class RenderFrame:
    """A drawn scene for one view: the image and the camera it was seen from, which the
    view takes for its marks so that they stand on this image, not on a newer camera."""

    view: object  # the view that asked for it
    key: tuple  # what the view asked for, as it tells one frame from another
    camera: object  # a copy of the view's camera at the request
    image: QImage
    buffer: np.ndarray | None = None  # the image's pixels, from `pool`
    pool: _BufferPool | None = None

    def release(self) -> None:
        """Gives the pixels back to be drawn into again: the image is no longer shown."""
        if self.pool is not None and self.buffer is not None:
            self.pool.give_back(self.buffer)
        self.pool = self.buffer = None


class RenderThread(QThread):
    """Draws the scenes of one renderer for the views in a thread of its own, see the
    module; views that share it share the renderer's context.

    A view asks for a frame (`request`) and gets it as a `RenderFrame` through
    `frame_ready`, in its own thread; frames of other views it leaves alone. Of the
    requests that come while a frame is drawn only the newest is drawn afterwards: a
    gesture faster than the graphics card is not drawn step by step. The first request
    sets OpenGL up, the context in the GUI thread (`OffscreenRenderer.open`), the rest in
    this thread. If that fails, `failed` carries the reason (translated) and `error` keeps
    it. `stop` ends the thread before the application does.
    """

    frame_ready = pyqtSignal(object)  # RenderFrame, from this thread
    failed = pyqtSignal(str)  # OpenGL has turned out to be missing: `error`

    def __init__(self, renderer: OffscreenRenderer, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._renderer = renderer
        self._condition = threading.Condition()
        self._request: tuple | None = None  # the newest not yet drawn
        self._stopping = False
        self._pool = _BufferPool()
        application = QCoreApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self.stop)

    @property
    def error(self) -> str:
        return self._renderer.error

    @property
    def ready(self) -> bool:
        return self._renderer.ready

    def request(
        self,
        view: object,
        key: tuple,
        camera,
        width: int,
        height: int,
        scene: object | None,
        line_width: float,
    ) -> None:
        """Asks for `scene` seen from `camera` (copied here) as an image of `width` by
        `height` pixels; from the GUI thread. `key` comes back with the frame."""
        if self._renderer.error or self._stopping:
            return
        if not self.isRunning():
            if not self._renderer.open():
                self.failed.emit(self._renderer.error)
                return
            self._renderer.move_to_thread(self)
            self.start()
        with self._condition:
            self._request = (view, key, copy.copy(camera), width, height, scene, line_width)
            self._condition.notify()

    def stop(self) -> None:
        """Ends the thread, which frees what the graphics card holds, and drops the
        context; from the GUI thread."""
        with self._condition:
            self._stopping = True
            self._condition.notify()
        if self.isRunning():
            self.wait()
        self._renderer.close()

    def run(self) -> None:
        renderer = self._renderer
        try:
            while renderer.prepare():
                pass
            if renderer.error:
                self.failed.emit(renderer.error)
            drawn_at = -math.inf
            while True:
                with self._condition:
                    while self._request is None and not self._stopping:
                        self._condition.wait()
                    if self._stopping:
                        return
                # No more often than FRAME_INTERVAL_SECONDS; the newest request by then
                # is the one drawn.
                pause = drawn_at + FRAME_INTERVAL_SECONDS - time.monotonic()
                if pause > 0:
                    time.sleep(pause)
                with self._condition:
                    if self._stopping:
                        return
                    request, self._request = self._request, None
                if renderer.error:
                    continue
                drawn_at = time.monotonic()
                view, key, camera, width, height, scene, line_width = request
                buffer = self._pool.take(width, height)
                image = renderer.render(camera, width, height, scene, line_width, buffer)
                if image is None:
                    self._pool.give_back(buffer)
                    if renderer.error:
                        self.failed.emit(renderer.error)
                    continue
                self.frame_ready.emit(RenderFrame(view, key, camera, image, buffer, self._pool))
        except Exception:  # the views must learn of it, not meet a dead thread
            logger.exception("%s: drawing failed", type(renderer).__name__)
            renderer.error = tr("The 3D view has failed (see the log).")
            self.failed.emit(renderer.error)
        finally:
            renderer.release_gl()
