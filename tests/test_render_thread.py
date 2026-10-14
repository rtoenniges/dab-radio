"""The thread that draws the 3D views (`ui.offscreen_gl.RenderThread`), with a stand-in
for the OpenGL part: frames come back in the GUI thread, only the newest of several
requests is drawn, the pixels are drawn into again once a frame is let go of, and a
failure is reported instead of a frame."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import numpy as np
import pytest
from PyQt6 import sip
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication

from ui.globe_gl import Camera
from ui.offscreen_gl import FRAME_INTERVAL_SECONDS, RenderFrame, RenderThread

APP = QApplication.instance() or QApplication([])
MAIN_THREAD = threading.get_ident()


class _FakeGl:
    """Stands in for `OffscreenRenderer`: fills the buffer with one value and notes which
    thread did what."""

    def __init__(self, opens: bool = True, seconds: float = 0.0) -> None:
        self.ready = False
        self.error = ""
        self._opens = opens
        self._seconds = seconds  # how long a frame takes
        self.threads: dict[str, int] = {}
        self.rendered: list[tuple[float, int, int]] = []  # longitude, width, height
        self.closed = False

    def open(self) -> bool:
        self.threads["open"] = threading.get_ident()
        if not self._opens:
            self.error = "The 3D view needs OpenGL 2.1 (stand-in)."
        return self._opens

    def move_to_thread(self, thread) -> None:
        self.threads["move"] = threading.get_ident()

    def prepare(self) -> bool:
        self.threads["prepare"] = threading.get_ident()
        self.ready = True
        return False

    def render(self, camera, width, height, lines, line_width, buffer) -> QImage:
        self.threads["render"] = threading.get_ident()
        time.sleep(self._seconds)
        buffer[:] = len(self.rendered) + 1
        self.rendered.append((camera.longitude_deg, width, height))
        return QImage(
            sip.voidptr(buffer.ctypes.data), width, height, width * 4, QImage.Format.Format_RGB32
        )

    def release_gl(self) -> None:
        self.threads["release"] = threading.get_ident()

    def close(self) -> None:
        self.threads["close"] = threading.get_ident()
        self.closed = True


class _View:
    pass


def _spin_until(condition, seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        APP.processEvents()
        time.sleep(0.002)
    assert condition(), "the thread did not answer in time"


@pytest.fixture
def thread():
    threads = []

    def make(gl: _FakeGl) -> RenderThread:
        thread = RenderThread(gl)
        threads.append(thread)
        return thread

    yield make
    for thread in threads:
        thread.stop()


def test_a_frame_comes_back_in_the_gui_thread_drawn_in_the_other(thread) -> None:
    gl = _FakeGl()
    render = thread(gl)
    frames: list[RenderFrame] = []
    render.frame_ready.connect(lambda frame: frames.append(frame))
    view, camera = _View(), Camera(10.0, 20.0, 30_000.0)

    render.request(view, ("key",), camera, 64, 48, None, 1.5)
    _spin_until(lambda: frames)

    frame = frames[0]
    assert frame.view is view and frame.key == ("key",)
    assert frame.camera is not camera and frame.camera.longitude_deg == 20.0  # a copy
    assert frame.image.size().width() == 64 and frame.image.size().height() == 48
    assert frame.buffer.shape == (48, 64, 4) and frame.buffer[0, 0, 0] == 1
    assert render.ready and not render.error
    assert gl.threads["open"] == MAIN_THREAD and gl.threads["move"] == MAIN_THREAD
    assert gl.threads["prepare"] != MAIN_THREAD and gl.threads["render"] != MAIN_THREAD
    assert threading.get_ident() == MAIN_THREAD  # the frames were delivered here

    render.stop()
    assert gl.closed and gl.threads["close"] == MAIN_THREAD
    assert gl.threads["release"] == gl.threads["render"]


def test_of_requests_during_a_frame_only_the_newest_is_drawn(thread) -> None:
    gl = _FakeGl(seconds=0.05)
    render = thread(gl)
    frames: list[RenderFrame] = []
    render.frame_ready.connect(lambda frame: frames.append(frame))
    view = _View()

    for longitude in range(1, 7):
        render.request(view, (longitude,), Camera(0.0, float(longitude), 30_000.0), 8, 8, None, 1.0)
    _spin_until(lambda: frames and frames[-1].key == (6,))
    time.sleep(2 * FRAME_INTERVAL_SECONDS)  # nothing more is drawn afterwards
    APP.processEvents()

    assert [c[0] for c in gl.rendered] in ([6.0], [1.0, 6.0])
    assert frames[-1].key == (6,) and frames[-1].camera.longitude_deg == 6.0


def test_a_frame_let_go_of_lends_its_pixels_to_the_next(thread) -> None:
    gl = _FakeGl()
    render = thread(gl)
    frames: list[RenderFrame] = []
    render.frame_ready.connect(lambda frame: frames.append(frame))
    view = _View()

    render.request(view, (1,), Camera(0.0, 0.0, 30_000.0), 8, 8, None, 1.0)
    _spin_until(lambda: len(frames) == 1)
    first = frames[0].buffer
    render.request(view, (2,), Camera(0.0, 1.0, 30_000.0), 8, 8, None, 1.0)
    _spin_until(lambda: len(frames) == 2)
    assert frames[1].buffer is not first  # the first is still shown
    frames[0].release()
    frames[0].release()  # a second time does no harm
    render.request(view, (3,), Camera(0.0, 2.0, 30_000.0), 8, 8, None, 1.0)
    _spin_until(lambda: len(frames) == 3)
    assert frames[2].buffer is first
    assert np.all(first == 3)

    render.request(view, (4,), Camera(0.0, 3.0, 30_000.0), 4, 4, None, 1.0)  # resized
    _spin_until(lambda: len(frames) == 4)
    assert frames[3].buffer.shape == (4, 4, 4)


def test_without_opengl_the_failure_is_reported_at_once_and_nothing_starts(thread) -> None:
    gl = _FakeGl(opens=False)
    render = thread(gl)
    failures: list[str] = []
    render.failed.connect(failures.append)

    render.request(_View(), (1,), Camera(0.0, 0.0, 30_000.0), 8, 8, None, 1.0)

    assert failures == [gl.error] and render.error == gl.error
    assert not render.isRunning()
    render.request(_View(), (2,), Camera(0.0, 0.0, 30_000.0), 8, 8, None, 1.0)
    assert len(failures) == 1  # not again
