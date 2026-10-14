"""The sub-page "Recording" of the settings: what the recorder of the last titles
(`recording.recorder`) is told.

- "Recording": on or off. Off, the recorder writes nothing and the title recording ends.
- "Buffer size": how many of the last titles the ring keeps, and the way to the
  recordings (`recordings_requested`, the button "Show recordings").
- "Start delay": when the file of a title begins against the dynamic label that
  names it (the label mostly runs late; the file then begins before it).
- "Stop delay": when the file of the title before ends against that label: cut
  short before it, or run on after it while the next title already records.

Both delays are set for the station playing, which gets a value of its own from the
first step on; "Apply to all" makes its value the default of every station
without one (`ReceiverController.label_delay_s`, `stop_delay_s`). Where the
recordings go is on the sub-page "Storage" (`ui.touch.storage_view`).
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QWidget

from i18n import tr, tr_plural
from recording.recorder import LABEL_DELAY_RANGE_S, RING_SIZE_RANGE
from ui.receiver_controller import ReceiverController
from ui.touch.widgets import ElidedLabel, Stepper, set_role

# The delays (`recording.recorder.LABEL_DELAY_RANGE_S`), in steps of this.
LABEL_DELAY_STEP_S = 5
# All steppers as wide as the longest text, so that their buttons line up.
VALUE_WIDTH = 260


def start_delay_text(seconds: int) -> str:
    """When the file of a title begins (or ends) against the dynamic label: "At Dyn.
    Label", "30 s before Dyn. Label" (the label runs late), "10 s after Dyn. Label"."""
    if seconds == 0:
        # Translators: the file of a title begins (or ends) right at the Dynamic Label
        # that names the title; "Dyn. Label" shortened to fit the stepper.
        return tr("At Dyn. Label")
    if seconds > 0:
        return tr("{seconds} s before Dyn. Label").format(seconds=seconds)
    return tr("{seconds} s after Dyn. Label").format(seconds=-seconds)


def delay_scope_text(station: str | None, own: bool) -> str:
    """Whom the delay set applies to: "Dlf, own value", "Dlf, default", or the
    default for every station while no station is selected."""
    if station is None:
        return tr("Default for all stations")
    if own:
        return tr("{station}, own value").format(station=station)
    return tr("{station}, default").format(station=station)


class _DelayRow:
    """A delay against the dynamic label: the stepper, whom it applies to, and the
    button that makes its value the default of every station."""

    def __init__(
        self,
        value: Callable[[], int],
        own: Callable[[], bool],
        set_value: Callable[[int], None],
        set_default: Callable[[int], None],
    ) -> None:
        self._value = value
        self._own = own
        low, high = LABEL_DELAY_RANGE_S
        self.stepper = Stepper(
            range(low, high + 1, LABEL_DELAY_STEP_S), start_delay_text, value_width=VALUE_WIDTH
        )
        self.stepper.value_changed.connect(set_value)
        self.scope = set_role(ElidedLabel(), "muted")
        self.default_button = QPushButton(tr("Apply to all"))
        self.default_button.clicked.connect(lambda: set_default(value()))
        self.layout = QHBoxLayout()
        self.layout.setSpacing(16)
        self.layout.addWidget(self.stepper)
        self.layout.addWidget(self.scope, stretch=1)

    def show(self, station: str | None) -> None:
        self.stepper.set_value(self._value())
        self.scope.setText(delay_scope_text(station, self._own()))
        self.default_button.setVisible(station is not None)


class RecordingView(QWidget):
    """Rows like the other sub-pages: the switch, "Buffer size", "Start delay" and
    "Stop delay", the steppers with their explanation and a button beside."""

    recordings_requested = pyqtSignal()  # the button "Show recordings"

    def __init__(
        self, controller: ReceiverController, label_width: int, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._controller = controller

        self._switch = set_role(QPushButton(tr("Off")), "switch")
        self._switch.setCheckable(True)
        self._switch.toggled.connect(lambda on: self._switch.setText(tr("On") if on else tr("Off")))
        self._switch.setChecked(controller.recording_enabled)
        self._switch.clicked.connect(controller.set_recording_enabled)

        low, high = RING_SIZE_RANGE
        self._ring_size = Stepper(
            range(low, high + 1),
            lambda n: tr_plural("{count} title", "{count} titles", n).format(count=n),
            value_width=VALUE_WIDTH,
        )
        self._ring_size.set_value(controller.ring_size)
        self._ring_size.value_changed.connect(controller.set_ring_size)
        self._show_recordings = QPushButton(tr("Show recordings"))
        self._show_recordings.clicked.connect(self.recordings_requested)
        ring = QHBoxLayout()
        ring.setSpacing(16)
        ring.addWidget(self._ring_size)
        ring.addStretch(1)

        self._start = _DelayRow(
            lambda: controller.label_delay_s,
            lambda: controller.label_delay_is_station_own,
            controller.set_label_delay_s,
            controller.set_default_label_delay_s,
        )
        self._stop = _DelayRow(
            lambda: controller.stop_delay_s,
            lambda: controller.stop_delay_is_station_own,
            controller.set_stop_delay_s,
            controller.set_default_stop_delay_s,
        )

        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(12)
        names = (tr("Recording"), tr("Buffer size"), tr("Start delay"), tr("Stop delay"))
        for row, name in enumerate(names):
            label = set_role(QLabel(name), "muted")
            label.setFixedWidth(label_width)
            grid.addWidget(label, row, 0, alignment=Qt.AlignmentFlag.AlignVCenter)
        left = Qt.AlignmentFlag.AlignLeft
        grid.addWidget(self._switch, 0, 1, alignment=left)
        grid.addLayout(ring, 1, 1)
        grid.addWidget(self._show_recordings, 1, 2, alignment=left)
        grid.addLayout(self._start.layout, 2, 1)
        grid.addWidget(self._start.default_button, 2, 2, alignment=left)
        grid.addLayout(self._stop.layout, 3, 1)
        grid.addWidget(self._stop.default_button, 3, 2, alignment=left)
        grid.setRowStretch(4, 1)
        grid.setColumnStretch(1, 1)

        controller.recording_enabled_changed.connect(self._switch.setChecked)
        controller.label_delay_changed.connect(self._show_delays)
        controller.stop_delay_changed.connect(self._show_delays)
        controller.selected_service_changed.connect(self._on_service_changed)
        self._show_delays()

    def _on_service_changed(self, _sid: object) -> None:
        self._show_delays()

    def _show_delays(self) -> None:
        station = self._controller.selected_service_label
        self._start.show(station)
        self._stop.show(station)
