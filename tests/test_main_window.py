"""The main window on its controller (`ui.main_window`): that the controls and the side
panel follow the controller, also where the change does not come from them, and hand
what the user does on to it. What the controller does with it is tested in
`test_receiver_controller`."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from dab.channels import BAND_III_CHANNELS
from tests.fake_receiver import ENSEMBLE_5C, SERVICES_5C, SERVICES_11D
from ui.main_window import CHANNEL_MANUAL, MainWindow


@pytest.fixture
def window_on(make_controller):
    """Creates a window on a fresh controller and starts the controller, as `main` does."""
    windows = []

    def make(data_directory) -> MainWindow:
        controller = make_controller(data_directory, start=False)
        window = MainWindow(controller)
        controller.start()
        windows.append(window)
        return window

    yield make
    for window in windows:
        window.deleteLater()


def _service_labels(window: MainWindow) -> list[str]:
    service_list = window._service_list
    return [service_list.item(row).text() for row in range(service_list.count())]


def test_the_stored_list_is_shown_after_the_start(data_directory, window_on):
    window = window_on(data_directory)

    assert window._channel_combo.currentData() == "5C"
    assert window._ensemble_label.text() == f"Ensemble: {ENSEMBLE_5C}"
    assert [label for _, label in SERVICES_5C] == _service_labels(window)
    assert window._service_source_label.text().startswith("Stored station list")
    assert not window._update_services_button.isEnabled()


def test_the_controls_follow_the_reception(data_directory, window_on, fake_workers):
    window = window_on(data_directory)
    dsp = window._controller.dsp_worker

    window._start_stop_button.click()
    assert not window._start_stop_button.isEnabled()
    assert not window._channel_combo.isEnabled()

    worker = fake_workers[-1]
    worker.connected.emit("RTL-SDR ready.")
    assert window._status_label.text() == "RTL-SDR ready."
    assert window._start_stop_button.text() == "Stop"
    assert window._start_stop_button.isEnabled()
    assert window._channel_combo.isEnabled()
    assert not window._scan_button.isEnabled()
    assert not window._device_combo.isEnabled()

    dsp.stream_restarted.emit(worker.tuning)
    dsp.services_updated.emit(ENSEMBLE_5C, list(SERVICES_5C), {})
    assert window._service_source_label.text().startswith("Live from the FIC.")
    assert window._update_services_button.isEnabled()

    window._start_stop_button.click()
    worker.stopped.emit()
    assert window._status_label.text() == "Stopped."
    assert window._start_stop_button.text() == "Start"
    assert window._scan_button.isEnabled()
    assert window._device_combo.isEnabled()


def test_the_channel_selection_follows_the_play_button(data_directory, window_on, fake_workers):
    window = window_on(data_directory)

    window._controller.play_service("11D", SERVICES_11D[0][0])

    assert window._channel_combo.currentData() == "11D"
    assert window._frequency_spin.value() == BAND_III_CHANNELS["11D"]
    assert not window._frequency_spin.isEnabled()
    # Moved without the combo box reporting it back: one reception, no retune.
    assert len(fake_workers) == 1
    assert fake_workers[0].retuned_to == []


def test_a_typed_frequency_is_handed_to_the_controller(data_directory, window_on):
    window = window_on(data_directory)

    window._channel_combo.setCurrentIndex(window._channel_combo.findText(CHANNEL_MANUAL))
    assert window._frequency_spin.isEnabled()
    window._frequency_spin.setValue(200.0)

    assert window._controller.channel is None
    assert window._controller.frequency_mhz == 200.0
    assert _service_labels(window) == []


def test_a_click_in_the_list_selects_the_service(data_directory, window_on):
    window = window_on(data_directory)

    window._service_list.itemClicked.emit(window._service_list.item(1))

    assert window._controller.selected_sid == SERVICES_5C[1][0]
    assert window._service_list.currentRow() == 1
    assert window._subchannel_info_label.text() == "Waiting for service information..."


def test_the_side_panel_names_the_other_channels_of_the_selected_service(
    dlf_kultur_elsewhere, window_on
):
    window = window_on(dlf_kultur_elsewhere)
    label = window._service_channel_label
    assert label.isHidden()  # no service selected

    window._service_list.itemClicked.emit(window._service_list.item(1))  # Dlf Kultur
    assert not label.isHidden()
    assert label.text() == "Channel 5C [11D, 12D?]"

    window._service_list.itemClicked.emit(window._service_list.item(0))  # Dlf: only here
    assert label.text() == "Channel 5C"


def test_device_settings_are_handed_to_the_controller(data_directory, window_on):
    window = window_on(data_directory)

    window._device_combo.setCurrentIndex(window._device_combo.findData("rspduo2"))
    window._lna_combo.setCurrentIndex(3)

    settings = window._controller.device_settings
    assert settings.device == "rspduo2"
    assert settings.lna_state == 3
    assert settings.rspduo().tuner == 2
    assert not window._rtlsdr_controls.isVisibleTo(window)
    assert window._rspduo_controls.isVisibleTo(window)


def test_closing_the_window_shuts_the_controller_down(data_directory, window_on):
    window = window_on(data_directory)
    audio_thread = window._controller._audio_thread

    window.close()

    assert not audio_thread.isRunning()


def test_the_audio_box_tells_and_follows_the_controller(data_directory, window_on):
    window = window_on(data_directory)
    controller = window._controller
    box = window._audio_checkbox
    assert box.isChecked() and controller.audio_enabled

    box.click()
    assert not controller.audio_enabled
    # Muted or unmuted from elsewhere (MQTT): the box follows.
    controller.set_audio_enabled(True)
    assert box.isChecked()
    controller.set_audio_enabled(False)
    assert not box.isChecked()
    box.click()
    assert controller.audio_enabled
