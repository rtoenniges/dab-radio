"""The sub-page "Recording" of the settings: what the recorder of the last titles
(`recording.recorder`) is told, as rows on two cards (`ui.touch.cards`).

- "Recording": on or off. Off, the recorder writes nothing and the title recording ends.
- "Buffer size": how many of the last titles the ring keeps; beneath, the way to the
  recordings (`recordings_requested`, the row "Show recordings").
- "Start delay": when the file of a title begins against the dynamic label that
  names it (the label mostly runs late; the file then begins before it).
- "Stop delay": when the file of the title before ends against that label: cut
  short before it, or run on after it while the next title already records.

Both delays are set for the station playing, which gets a value of its own from the
first step on; "Apply to all stations" makes its values the default of every station
without one (`ReceiverController.label_delay_s`, `stop_delay_s`). Where the
recordings go is on the sub-page "Storage" (`ui.touch.storage_view`).
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from i18n import tr, tr_noop, tr_plural
from recording.recorder import LABEL_DELAY_RANGE_S, RING_SIZE_RANGE
from ui.receiver_controller import ReceiverController
from ui.touch.cards import Card, Toggle, action_row, stepper_row, switch_row
from ui.touch.widgets import Stepper

# The delays (`recording.recorder.LABEL_DELAY_RANGE_S`), in steps of this.
LABEL_DELAY_STEP_S = 5
# All steppers as wide as the longest text, so that their buttons line up.
VALUE_WIDTH = 240
_RING_HINT = tr_noop("How many of the last titles are kept; titles kept for good stay anyway.")
_DEFAULT_HINT = tr_noop(
    "Makes the delays of the station playing the default of every station without its own."
)


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
    """A delay against the dynamic label: the row with the stepper, and beneath its
    name whom the value applies to."""

    def __init__(
        self,
        title: str,
        value: Callable[[], int],
        own: Callable[[], bool],
        set_value: Callable[[int], None],
    ) -> None:
        self._value = value
        self._own = own
        low, high = LABEL_DELAY_RANGE_S
        self.stepper = Stepper(
            range(low, high + 1, LABEL_DELAY_STEP_S), start_delay_text, value_width=VALUE_WIDTH
        )
        self.stepper.value_changed.connect(set_value)
        self.row = stepper_row(title, self.stepper)
        self.scope = self.row.subtitle

    def show(self, station: str | None) -> None:
        self.stepper.set_value(self._value())
        self.row.set_subtitle(delay_scope_text(station, self._own()))


class RecordingView(QWidget):
    """The cards "Recording" (the switch, "Buffer size", "Show recordings") and "Delays"
    ("Start delay", "Stop delay", "Apply to all stations")."""

    recordings_requested = pyqtSignal()  # the row "Show recordings"
    # The size of the ring was changed here (the settings page shows it in its list).
    changed = pyqtSignal()

    def __init__(self, controller: ReceiverController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller

        self._switch = Toggle()
        self._switch.setChecked(controller.recording_enabled)
        self._switch.clicked.connect(controller.set_recording_enabled)

        low, high = RING_SIZE_RANGE
        self._ring_size = Stepper(
            range(low, high + 1),
            lambda n: tr_plural("{count} title", "{count} titles", n).format(count=n),
            value_width=VALUE_WIDTH,
        )
        self._ring_size.set_value(controller.ring_size)
        self._ring_size.value_changed.connect(self._on_ring_size_changed)
        self._show_recordings = action_row(tr("Show recordings"))
        self._show_recordings.tapped.connect(self.recordings_requested)

        self._start = _DelayRow(
            tr("Start delay"),
            lambda: controller.label_delay_s,
            lambda: controller.label_delay_is_station_own,
            controller.set_label_delay_s,
        )
        self._stop = _DelayRow(
            tr("Stop delay"),
            lambda: controller.stop_delay_s,
            lambda: controller.stop_delay_is_station_own,
            controller.set_stop_delay_s,
        )
        self._apply_defaults = action_row(tr("Apply to all stations"), tr(_DEFAULT_HINT))
        self._apply_defaults.tapped.connect(self._on_apply_defaults)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(
            Card(
                tr("Recording"),
                [
                    switch_row(tr("Recording"), self._switch),
                    stepper_row(tr("Buffer size"), self._ring_size, tr(_RING_HINT)),
                    self._show_recordings,
                ],
            )
        )
        layout.addWidget(
            Card(tr("Delays"), [self._start.row, self._stop.row, self._apply_defaults])
        )
        layout.addStretch()

        controller.recording_enabled_changed.connect(self._switch.setChecked)
        controller.label_delay_changed.connect(self._show_delays)
        controller.stop_delay_changed.connect(self._show_delays)
        controller.selected_service_changed.connect(self._on_service_changed)
        self._show_delays()

    def _on_ring_size_changed(self, size: int) -> None:
        self._controller.set_ring_size(size)
        self.changed.emit()

    def _on_service_changed(self, _sid: object) -> None:
        self._show_delays()

    def _show_delays(self) -> None:
        station = self._controller.selected_service_label
        self._start.show(station)
        self._stop.show(station)
        # Without a station the steppers set the defaults themselves.
        self._apply_defaults.setVisible(station is not None)

    def _on_apply_defaults(self) -> None:
        controller = self._controller
        controller.set_default_label_delay_s(controller.label_delay_s)
        controller.set_default_stop_delay_s(controller.stop_delay_s)
