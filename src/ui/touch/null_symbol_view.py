"""The null symbol on the radio page, in the place of the slideshow
(`ui.touch.display_choice`): the landscape of the detail view (`ui.touch.detail_view`)
with fixed settings (a sliding FFT of 1024, 4 ms, the whole band, 30 dB high), seen at
first along the time axis from the front left, drawn from the stretch the signal
processing cuts around the null symbol of every frame
(`dsp.signal_capture.NullSnapshotter`).

It is drawn at the rate of the other plots (`render`, five times a second), from the
newest stretch, taking turns between frames of even and of odd number. Some transmitters
send their TII in every other null symbol only, so the comb comes and goes; the newest
stretch alone would nearly always be of the same kind, since two frames take 192 ms
against the 200 ms of the plot rate.

The stretches come only while the tile is shown. The landscape is coarser than in the
detail view (`ROWS`, `COLUMNS`): plenty for the 640 px of the slide, and little work for
the Raspberry Pi, five times a second.

Fingers come through the slide area of the radio page, which keeps one for its long
press: two turn the view as they move together and bring it closer or farther as they
spread or close; on a PC the wheel zooms and the middle button held turns. Unlike the
globe there, fingers turn it as well; like the globe, it is not shifted and has no
double tap to return to the first view, that is for the detail view itself. Where it
was turned to stays, also over the next start (`SETTINGS_RADIO_NULL_SYMBOL_CAMERA`), as
the globe keeps its distance.
"""

from __future__ import annotations

import dataclasses
import math

from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QHideEvent, QShowEvent

from dsp.signal_capture import SignalCapture
from i18n import tr
from ui.offscreen_gl import RenderThread
from ui.receiver_controller import ReceiverController
from ui.spectrogram import NULL_LEAD, Spectrogram
from ui.touch.detail_view import WaterfallView
from ui.touch.signal_panels import SignalPanel
from ui.touch.theme import DARK, THEMES, Theme
from ui.waterfall_gl import DISTANCE_RANGE, ELEVATION_RANGE, WaterfallCamera

# Key of the setting holding where the view was turned to: azimuth, elevation and
# distance, separated by spaces.
SETTINGS_RADIO_NULL_SYMBOL_CAMERA = "touch/radio_null_symbol_camera"

FFT_SIZE = 1024
WINDOW_MS = 4.0
RANGE_DB = 30.0
ROWS = 100
COLUMNS = 512
# Along the time axis, from a little before the band's low edge with the strip of the
# frame structure, so that the null symbol lies across the view. It looks at the point
# the detail view looks at, the middle of the landscape, which it turns around; looking
# at a point shifted so that landscape and figures together stand in the middle makes
# the landscape seem shifted to the right.
CAMERA = WaterfallCamera(azimuth_deg=-75.34, elevation_deg=35.38, distance=4.4)


def stored_camera(settings: QSettings) -> WaterfallCamera:
    """The camera the tile was last turned to (`SETTINGS_RADIO_NULL_SYMBOL_CAMERA`),
    within the bounds of the camera; the default (`CAMERA`) without one, or where the
    setting was edited by hand into something else."""
    camera = dataclasses.replace(CAMERA)
    try:
        azimuth, elevation, distance = (
            float(v) for v in str(settings.value(SETTINGS_RADIO_NULL_SYMBOL_CAMERA, "")).split()
        )
    except ValueError:
        return camera
    if not all(math.isfinite(v) for v in (azimuth, elevation, distance)):
        return camera
    camera.azimuth_deg = azimuth
    camera.elevation_deg = min(max(elevation, ELEVATION_RANGE[0]), ELEVATION_RANGE[1])
    camera.distance = min(max(distance, DISTANCE_RANGE[0]), DISTANCE_RANGE[1])
    camera.rotate(0.0, 0.0)  # the azimuth within ±180°
    return camera


class NullSymbolPanel(SignalPanel):
    """The tile (see the module); `renderer` may be shared with the detail view."""

    turned_by_fingers = True

    def __init__(
        self, controller: ReceiverController, renderer: RenderThread | None = None
    ) -> None:
        super().__init__(tr("Null symbol"))
        self.heading.hide()  # the view fills the panel
        self.body.setContentsMargins(0, 0, 0, 0)
        self._controller = controller
        self.model = Spectrogram(max_rows=ROWS, max_columns=COLUMNS)
        self.model.fft_size = FFT_SIZE
        self.model.window_ms = WINDOW_MS
        self.view = WaterfallView(self.model, renderer)
        self.view.set_camera(stored_camera(controller.settings))
        self.view.set_range_db(RANGE_DB)
        self.view.set_waiting_text(tr("Waiting for the null symbols …"))
        self.body.addWidget(self.view, stretch=1)
        # The newest stretch and the one drawn last, by the parity of the frame number.
        self._newest: dict[int, SignalCapture] = {}
        self._drawn: dict[int, SignalCapture] = {}
        self._parity: int | None = None  # of the stretch shown
        controller.dsp_worker.null_snapshot_ready.connect(self._note)
        controller.live_reset.connect(self._reset)
        self.apply_theme(THEMES[DARK])

    def apply_theme(self, theme: Theme) -> None:
        self.view.set_theme(theme)

    def turn(self, dx: float, dy: float) -> None:
        self.view.turn(dx, dy)
        self._keep_camera()

    def zoom(self, factor: float) -> None:
        self.view.zoom(factor)
        self._keep_camera()

    def _keep_camera(self) -> None:
        camera = self.view.camera
        self._controller.settings.setValue(
            SETTINGS_RADIO_NULL_SYMBOL_CAMERA,
            f"{camera.azimuth_deg:.2f} {camera.elevation_deg:.2f} {camera.distance:.3f}",
        )

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._controller.set_null_snapshots(True)

    def hideEvent(self, event: QHideEvent) -> None:
        super().hideEvent(event)
        self._controller.set_null_snapshots(False)

    def _note(self, snapshot: SignalCapture) -> None:
        self._newest[snapshot.first_frame_number % 2] = snapshot

    def _reset(self) -> None:
        """Another channel, or reception stopped: nothing to show until it comes."""
        self._newest.clear()
        self._drawn.clear()
        self._parity = None
        self.model.clear()
        self.view.refresh(at_once=True)

    def render(self) -> None:
        """The newest stretch of the other parity than the one shown, or of the same
        where frames are missing; nothing new, nothing drawn."""
        order = (0, 1) if self._parity is None else (1 - self._parity, self._parity)
        for parity in order:
            snapshot = self._newest.get(parity)
            if snapshot is not None and snapshot is not self._drawn.get(parity):
                self._show(snapshot, parity)
                return

    def _show(self, snapshot: SignalCapture, parity: int) -> None:
        self._drawn[parity] = snapshot
        self._parity = parity
        model = self.model
        model.set_capture(snapshot)
        null_ms = 1000 * float(snapshot.null_starts[0]) / snapshot.sample_rate
        model.time_origin_ms = null_ms  # the time from the start of the null symbol
        model.scroll_to(null_ms - NULL_LEAD * model.window_ms)
        self.view.refresh(at_once=True)
