"""The receiver without a surface (`ui.receiver_controller`): stored settings, the service
list between stored and received, service selection, retuning and the play button of the
programme guide.

No device is needed: the RTL-SDR worker is replaced by one that only records what it is
asked to do (`tests.fake_receiver`), and the results of the signal processing are
emitted by the test.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import time
from dataclasses import replace
from pathlib import Path

import pytest
from PyQt6.QtCore import QEventLoop, QSettings, QTimer

from audio.equalizer import BASS, LOW_SHELF, SPEECH, Band, EqualizerSettings, EqualizerState
from dab.announcement import ALARM_CLUSTER, AnnouncementStatus
from dab.channels import BAND_III_CHANNELS
from dab.data_services import data_components
from dab.ensemble_cache import SOURCE_LIVE, SOURCE_SCAN, EnsembleCache, EnsembleInfo
from dab.fig import Announcement, OeAnnouncement
from dab.gnss import Observer
from dab.mot import CONTENT_TYPE_IMAGE, MotObject
from dab.service_following import EnsembleLocation
from dsp.pipeline import CarouselObject, DataComponentStatus
from sdr.gain import FALLBACK_GAIN_RANGE_DB, FALLBACK_GAINS_DB
from sdr.rtl_tcp import RtlTcpSettings
from tests.fake_receiver import ENSEMBLE_5C, SERVICES_5C, SERVICES_11D
from tests.test_data_services import FIGS_5C, _ensemble
from ui import receiver_controller
from ui.receiver_controller import (
    DEFAULT_GAIN_DB,
    DEFAULT_OBSERVER,
    SETTINGS_FILENAME,
    DeviceSettings,
    ReceptionState,
    describe_other_channels,
    describe_stored,
)


class _Recorder:
    """Collects what a signal emits."""

    def __init__(self, signal) -> None:
        self.calls: list[tuple] = []
        signal.connect(lambda *args: self.calls.append(args))

    @property
    def last(self) -> tuple:
        return self.calls[-1]


def _settings(data_directory: Path) -> QSettings:
    return QSettings(str(data_directory / SETTINGS_FILENAME), QSettings.Format.IniFormat)


def test_stored_settings_are_clamped_to_what_can_be_set(data_directory, make_controller):
    settings = _settings(data_directory)
    settings.setValue("controls/gain_db", 80.0)  # from a stick with another tuner
    settings.setValue("controls/rspduo_lna_state", 42)
    settings.setValue("controls/rspduo_tuner", 3)
    settings.setValue("controls/device", "hackrf")
    settings.setValue("controls/channel", "99Z")
    settings.sync()

    controller = make_controller(data_directory, start=False)

    assert controller.device_settings == DeviceSettings(
        gain_db=FALLBACK_GAIN_RANGE_DB[1], lna_state=9
    )
    assert controller.channel == "5C"
    assert controller.frequency_mhz == BAND_III_CHANNELS["5C"]


def test_defaults_without_stored_settings(tmp_path, make_controller):
    controller = make_controller(tmp_path, start=False)

    assert controller.device_settings.gain_db == DEFAULT_GAIN_DB
    assert controller.device_settings.gain() == DEFAULT_GAIN_DB
    assert controller.device_settings.rspduo() is None
    assert not controller.dc_correction
    assert controller.audio_enabled
    assert controller.volume_percent == 70
    assert controller.volume_ceiling_db == 0
    assert controller.volume_db == pytest.approx(-18.0)


def test_the_volume_and_its_ceiling_are_stored_and_the_player_gets_the_gain(
    data_directory, make_controller
):
    controller = make_controller(data_directory, start=False)
    gains = _Recorder(controller._volume_changed)
    changes = _Recorder(controller.volume_changed)

    controller.set_volume_percent(80)
    assert controller.volume_db == pytest.approx(-12.0)
    assert gains.last == (pytest.approx(10 ** (-12 / 20)),)
    controller.set_volume_ceiling_db(-6)
    assert controller.volume_percent == 80
    assert controller.volume_db == pytest.approx(-18.0)
    assert gains.last == (pytest.approx(10 ** (-18 / 20)),)
    assert len(changes.calls) == 2
    controller.set_volume_ceiling_db(-99)
    assert controller.volume_ceiling_db == -30
    controller.set_volume_percent(0)
    assert controller.volume_db is None
    assert gains.last == (0.0,)

    controller.settings.sync()
    again = make_controller(data_directory, start=False)
    assert again.volume_percent == 0
    assert again.volume_ceiling_db == -30


def test_the_listeners_position_is_stored_and_clamped(data_directory, make_controller):
    controller = make_controller(data_directory, start=False)
    assert controller.observer == DEFAULT_OBSERVER
    changes = _Recorder(controller.observer_changed)

    controller.set_observer(Observer(52.3759, 9.732, 60.0))
    controller.set_observer(Observer(52.3759, 9.732, 60.0))
    assert changes.calls == [(Observer(52.3759, 9.732, 60.0),)]

    controller.set_observer(Observer(95.0, -200.0, 20000.0))
    assert controller.observer == Observer(90.0, -180.0, 9000.0)
    controller.settings.sync()
    assert make_controller(data_directory, start=False).observer == Observer(90.0, -180.0, 9000.0)


def test_favourites_are_stations_and_kept_for_the_next_start(data_directory, make_controller):
    controller = make_controller(data_directory, start=False)
    assert controller.favourites == frozenset()
    changes = _Recorder(controller.favourites_changed)

    controller.set_favourite("5C", 0xD220, True)
    controller.set_favourite("5C", 0xD220, True)
    controller.set_favourite("11D", 0xD311, True)
    assert len(changes.calls) == 2
    # The same service on another channel is another station.
    assert controller.is_favourite("5C", 0xD220)
    assert not controller.is_favourite("5D", 0xD220)
    controller.settings.sync()
    assert make_controller(data_directory, start=False).favourites == {
        ("5C", 0xD220),
        ("11D", 0xD311),
    }

    controller.set_favourite("11D", 0xD311, False)
    controller.set_favourite("5C", 0xD220, False)
    assert len(changes.calls) == 4
    controller.settings.sync()
    assert make_controller(data_directory, start=False).favourites == frozenset()


def test_the_equalizer_is_off_at_first_and_kept_for_the_next_start(data_directory, make_controller):
    controller = make_controller(data_directory, start=False)
    assert not controller.equalizer_enabled
    assert not controller.equalizer_per_station
    assert controller.equalizer == EqualizerSettings()
    changes = _Recorder(controller.equalizer_changed)
    to_player = _Recorder(controller._equalizer_set)
    bass = EqualizerSettings.from_preset(BASS)

    controller.set_equalizer(bass)
    controller.set_equalizer_enabled(True)
    controller.set_equalizer_enabled(True)

    assert len(changes.calls) == 2
    assert to_player.calls == [(None,), (bass,)]  # off until switched on
    controller.settings.sync()
    again = make_controller(data_directory, start=False)
    assert again.equalizer_enabled
    assert again.equalizer == bass


def test_per_station_a_station_keeps_its_own_and_the_others_sound_like_all(
    data_directory, make_controller
):
    """Its own settings and its own switch: a station switched off stays off, the
    others go on as all stations."""
    controller = make_controller(data_directory, start=False)
    bass = EqualizerSettings.from_preset(BASS)
    speech = EqualizerSettings.from_preset(SPEECH)
    controller.set_equalizer(bass)
    controller.set_equalizer_enabled(True)
    controller.set_equalizer_per_station(True)
    controller.select_service(DLF_KULTUR)
    assert controller.equalizer == bass
    assert controller.equalizer_enabled
    assert not controller.equalizer_is_station_own
    to_player = _Recorder(controller._equalizer_set)
    other = SERVICES_5C[0][0]

    controller.set_equalizer(speech)
    assert controller.equalizer_is_station_own
    controller.select_service(other)
    controller.set_equalizer_enabled(False)  # from now on its own: off
    controller.select_service(DLF_KULTUR)
    controller.set_equalizer_per_station(False)

    assert to_player.calls == [(speech,), (bass,), (None,), (speech,), (bass,)]
    assert controller.equalizer_enabled  # all stations: still on
    controller.settings.sync()
    again = make_controller(data_directory, start=False)
    again.set_equalizer_per_station(True)
    again.select_service(DLF_KULTUR)
    assert (again.equalizer, again.equalizer_enabled) == (speech, True)
    again.select_service(other)
    assert (again.equalizer, again.equalizer_enabled) == (bass, False)


def test_settings_made_by_hand_come_back_after_a_preset(data_directory, make_controller):
    controller = make_controller(data_directory, start=False)
    own = EqualizerSettings().with_band(0, Band(LOW_SHELF, 80.0, 4.0))

    controller.set_equalizer(own)
    controller.set_equalizer(EqualizerSettings.from_preset(SPEECH))

    assert controller.equalizer_state.custom == own
    controller.settings.sync()
    assert make_controller(data_directory, start=False).equalizer_state.custom == own


def test_deleting_the_equalizers_leaves_every_station_off_and_flat(data_directory, make_controller):
    controller = make_controller(data_directory, start=False)
    controller.set_equalizer_per_station(True)
    controller.select_service(DLF_KULTUR)
    controller.set_equalizer(EqualizerSettings.from_preset(SPEECH))
    controller.set_equalizer_enabled(True)
    controller.set_equalizer_per_station(False)
    controller.set_equalizer(EqualizerSettings.from_preset(BASS))
    controller.set_equalizer_per_station(True)
    to_player = _Recorder(controller._equalizer_set)

    controller.reset_equalizers()

    assert to_player.calls == [(None,)]
    assert controller.equalizer_per_station  # stays as chosen
    assert controller.equalizer_state == EqualizerState()
    assert not controller.equalizer_is_station_own
    controller.settings.sync()
    again = make_controller(data_directory, start=False)
    again.select_service(DLF_KULTUR)
    assert again.equalizer_state == EqualizerState()
    controller.set_equalizer_per_station(False)
    assert controller.equalizer_state == EqualizerState()


def test_equalizers_edited_beyond_reading_are_left_out(data_directory, make_controller):
    speech = EqualizerState(True, EqualizerSettings.from_preset(SPEECH))
    settings = _settings(data_directory)
    settings.setValue("equalizer/all_stations", "{not json")
    settings.setValue(
        "equalizer/stations",
        json.dumps(
            {
                "D220": speech.to_dict(),
                "Dlf": speech.to_dict(),
                "D210": {"bands": 3},
                "D230": speech.settings.to_dict(),  # no switch
            }
        ),
    )
    settings.setValue("equalizer/per_station", True)
    settings.sync()

    controller = make_controller(data_directory, start=False)
    assert controller.equalizer_state == EqualizerState()
    controller.select_service(0xD220)
    assert controller.equalizer_state == speech
    for sid in (0xD210, 0xD230):
        controller.select_service(sid)
        assert controller.equalizer_state == EqualizerState()


def test_a_favourite_edited_beyond_reading_is_left_out(data_directory, make_controller):
    settings = _settings(data_directory)
    settings.setValue("service/favourites", ["5C/D220", "5C/Dlf", "11D"])
    settings.sync()

    assert make_controller(data_directory, start=False).favourites == {("5C", 0xD220)}


def test_a_stored_manual_frequency_is_restored(data_directory, make_controller):
    settings = _settings(data_directory)
    settings.setValue("controls/channel", "manual")
    settings.setValue("controls/frequency_mhz", 200.0)
    settings.sync()

    controller = make_controller(data_directory, start=False)

    assert controller.channel is None
    assert controller.frequency_mhz == 200.0


def test_start_shows_the_stored_list_and_selects_the_stored_service(
    data_directory, make_controller
):
    settings = _settings(data_directory)
    settings.setValue("service/sid", 0xD220)
    settings.sync()
    controller = make_controller(data_directory, start=False)
    services = _Recorder(controller.service_list_changed)
    selected = _Recorder(controller.selected_service_changed)
    shown = _Recorder(controller.stored_list_shown)

    controller.start()

    assert services.last == (ENSEMBLE_5C, list(SERVICES_5C), {})
    assert shown.last == ("5C",)
    assert selected.last == (0xD220,)
    assert controller.selected_sid == 0xD220


def test_a_stored_service_the_list_lacks_is_not_selected(data_directory, make_controller):
    settings = _settings(data_directory)
    settings.setValue("service/sid", 0xBEEF)
    settings.sync()

    controller = make_controller(data_directory)

    assert controller.selected_sid is None


def test_a_channel_without_the_selected_service_drops_it(data_directory, make_controller):
    controller = make_controller(data_directory)
    controller.select_service(0xD210)
    tuning = _Recorder(controller.tuning_changed)
    selected = _Recorder(controller.selected_service_changed)
    source = _Recorder(controller.service_source_changed)

    controller.select_channel("11D")

    assert tuning.last == ("11D", BAND_III_CHANNELS["11D"])
    assert selected.last == (None,)
    assert source.last == ("No station list stored for channel 11D.", False)
    assert _settings(data_directory).value("controls/channel") == "11D"
    assert _settings(data_directory).value("service/sid") is None


def test_a_manual_frequency_replaces_the_selected_channel(data_directory, make_controller):
    controller = make_controller(data_directory)
    tuning = _Recorder(controller.tuning_changed)

    controller.set_manual_frequency(222.064)

    assert controller.channel is None
    assert tuning.last == (None, 222.064)
    assert _settings(data_directory).value("controls/channel") == "manual"


def test_reception_shows_only_the_list_of_the_confirmed_tuning(
    data_directory, make_controller, fake_workers
):
    controller = make_controller(data_directory)
    states = _Recorder(controller.state_changed)
    services = _Recorder(controller.service_list_changed)
    dsp = controller.dsp_worker

    controller.start_reception()
    worker = fake_workers[-1]
    assert worker.started
    assert worker.center_freq == BAND_III_CHANNELS["5C"] * 1e6
    assert worker.gain == DEFAULT_GAIN_DB
    assert services.last == (None, [], {})

    worker.connected.emit("Connected.")
    assert states.calls == [(ReceptionState.CONNECTING,), (ReceptionState.RECEIVING,)]

    # Still the result of an earlier frequency: the signal processing has not
    # confirmed the tuning yet.
    dsp.services_updated.emit("Alt", [(1, "Alt")], {})
    assert services.last == (None, [], {})

    dsp.stream_restarted.emit(worker.tuning)
    dsp.services_updated.emit(ENSEMBLE_5C, list(SERVICES_5C), {0xD210: 3})
    assert services.last == (ENSEMBLE_5C, list(SERVICES_5C), {0xD210: 3})


def test_retuning_waits_for_the_new_tuning(data_directory, make_controller, fake_workers):
    controller = make_controller(data_directory)
    services = _Recorder(controller.service_list_changed)
    live_resets = _Recorder(controller.live_reset)
    dsp = controller.dsp_worker
    controller.start_reception()
    worker = fake_workers[-1]
    worker.connected.emit("Connected.")
    dsp.stream_restarted.emit(worker.tuning)

    controller.select_channel("11D")

    assert worker.retuned_to == [BAND_III_CHANNELS["11D"] * 1e6]
    assert controller.center_frequency_hz == BAND_III_CHANNELS["11D"] * 1e6
    dsp.services_updated.emit(ENSEMBLE_5C, list(SERVICES_5C), {})
    assert services.last == (None, [], {})

    resets = len(live_resets.calls)
    dsp.stream_restarted.emit(worker.tuning)
    assert len(live_resets.calls) == resets + 1
    dsp.services_updated.emit("NDR NDS", list(SERVICES_11D), {})
    assert services.last == ("NDR NDS", list(SERVICES_11D), {})
    assert controller.list_channel == "11D"


def test_the_complete_list_is_reported_once_the_tuning_is_confirmed(
    data_directory, make_controller, fake_workers
):
    """Surfaces that wait for the whole list get it with the completion, and only for
    the frequency on display: the old one's completion is a straggler."""
    controller = make_controller(data_directory)
    completed = _Recorder(controller.service_list_completed)
    dsp = controller.dsp_worker
    controller.start_reception()
    worker = fake_workers[-1]
    dsp.stream_restarted.emit(worker.tuning)
    dsp.services_updated.emit(ENSEMBLE_5C, list(SERVICES_5C), {0xD210: 3})
    assert not controller.live_list_complete

    dsp.service_list_completed.emit()
    assert controller.live_list_complete
    assert completed.last == (ENSEMBLE_5C, list(SERVICES_5C), {0xD210: 3})

    controller.select_channel("11D")
    assert not controller.live_list_complete
    dsp.service_list_completed.emit()  # still about 5C
    assert not controller.live_list_complete
    assert len(completed.calls) == 1

    dsp.stream_restarted.emit(worker.tuning)
    dsp.services_updated.emit("NDR NDS", list(SERVICES_11D), {})
    dsp.service_list_completed.emit()
    assert completed.last == ("NDR NDS", list(SERVICES_11D), {})


def test_the_received_list_can_be_stored(data_directory, make_controller, fake_workers):
    controller = make_controller(data_directory)
    stored = _Recorder(controller.channel_stored)
    source = _Recorder(controller.service_source_changed)
    dsp = controller.dsp_worker
    controller.select_channel("11D")
    controller.start_reception()
    worker = fake_workers[-1]
    dsp.stream_restarted.emit(worker.tuning)
    dsp.services_updated.emit("NDR NDS", list(SERVICES_11D), {0xD311: 10})
    dsp.frequency_information_updated.emit(0x1011, {0x1011: (222064,)})
    assert source.last[1]

    controller.store_live_services()

    assert stored.last == ("11D",)
    info = EnsembleCache(data_directory / receiver_controller.ENSEMBLE_CACHE_FILENAME)
    info.load()
    entry = info.get("11D")
    assert entry.label == "NDR NDS"
    assert entry.services == SERVICES_11D
    assert entry.source == SOURCE_LIVE
    assert entry.programme_types == ((0xD311, 10),)
    assert entry.eid == 0x1011
    assert entry.frequencies == ((0x1011, (222064,)),)


def test_stopping_keeps_a_received_list(data_directory, make_controller, fake_workers):
    controller = make_controller(data_directory)
    stopped = _Recorder(controller.reception_stopped)
    states = _Recorder(controller.state_changed)
    dsp = controller.dsp_worker
    controller.start_reception()
    worker = fake_workers[-1]
    dsp.stream_restarted.emit(worker.tuning)
    dsp.services_updated.emit(ENSEMBLE_5C, list(SERVICES_5C), {})

    controller.stop_reception()
    assert worker.stop_requested
    worker.stopped.emit()

    assert stopped.last == (True,)
    assert states.last == (ReceptionState.IDLE,)
    assert not controller.is_receiving


def test_stopping_drops_the_load_of_the_signal_processing(
    data_directory, make_controller, fake_workers
):
    """It is only reported per second of signal: without reception no report comes."""
    controller = make_controller(data_directory)
    loads = _Recorder(controller.load_changed)
    dsp = controller.dsp_worker
    controller.start_reception()
    worker = fake_workers[-1]
    dsp.load_ready.emit(0.45, 0.40)
    assert loads.last[:2] == (0.45, pytest.approx(0.05))

    worker.stopped.emit()
    assert loads.last[:2] == (0.0, 0.0)

    dsp.load_ready.emit(0.45, 0.40)  # from blocks still queued at the end
    assert loads.last[:2] == (0.0, 0.0)


def test_the_waiting_share_of_the_signal_processing_is_never_negative(
    data_directory, make_controller, fake_workers
):
    """The CPU time of its thread is counted in ticks on Windows and can exceed the
    wall clock by one."""
    controller = make_controller(data_directory)
    loads = _Recorder(controller.load_changed)
    controller.start_reception()

    controller.dsp_worker.load_ready.emit(0.45, 0.46)

    assert loads.last[:2] == (0.45, 0.0)


def test_the_receiver_counts_as_overloaded_a_while_after_a_block_clipped(
    data_directory, make_controller, fake_workers, monkeypatch
):
    monkeypatch.setattr(receiver_controller, "OVERLOAD_HOLD_MS", 50)
    controller = make_controller(data_directory)
    changes = _Recorder(controller.overloaded_changed)
    dsp = controller.dsp_worker
    controller.start_reception()
    worker = fake_workers[-1]
    dsp.clipping_ready.emit(0.2)  # still the frequency before: the tuning is not confirmed
    assert not controller.overloaded

    dsp.stream_restarted.emit(worker.tuning)
    dsp.clipping_ready.emit(0.005)  # below the threshold of 1 %
    assert not controller.overloaded
    dsp.clipping_ready.emit(0.02)
    assert controller.overloaded
    dsp.clipping_ready.emit(0.005)  # one block clean: still within the hold
    assert controller.overloaded
    assert changes.calls == [(True,)]

    _wait_for(lambda: not controller.overloaded)
    assert changes.calls == [(True,), (False,)]


def test_stopping_ends_the_overload_at_once(data_directory, make_controller, fake_workers):
    controller = make_controller(data_directory)
    dsp = controller.dsp_worker
    controller.start_reception()
    worker = fake_workers[-1]
    dsp.stream_restarted.emit(worker.tuning)
    dsp.clipping_ready.emit(0.3)
    assert controller.overloaded

    worker.stopped.emit()
    assert not controller.overloaded
    dsp.clipping_ready.emit(0.3)  # from blocks still queued at the end
    assert not controller.overloaded


def test_the_rspduo_reporting_overload_overloads_the_receiver(
    data_directory, make_controller, fake_workers
):
    controller = make_controller(data_directory)
    controller.set_device_settings(
        replace(controller.device_settings, device="rspduo", rspduo_tuner=2)
    )
    controller.start_reception()
    worker = fake_workers[-1]
    controller.dsp_worker.stream_restarted.emit(worker.tuning)

    worker.overload_changed.emit(True)
    assert controller.adc_overload
    assert controller.overloaded
    controller.dsp_worker.clipping_ready.emit(0.0)  # it clips none, and still reports it
    assert controller.overloaded

    # Another channel: everything of the one before is void.
    controller.select_channel("11D")
    controller.dsp_worker.stream_restarted.emit(worker.tuning)
    assert not controller.adc_overload
    assert not controller.overloaded


def test_a_failed_start_shows_the_stored_list_again(data_directory, make_controller, fake_workers):
    controller = make_controller(data_directory)
    status = _Recorder(controller.status_changed)
    shown = _Recorder(controller.stored_list_shown)
    controller.start_reception()

    fake_workers[-1].error.emit("No RTL-SDR found.")

    assert status.last == ("Error: No RTL-SDR found.",)
    assert shown.last == ("5C",)
    assert controller.state is ReceptionState.IDLE


def test_the_play_button_tunes_selects_and_starts(data_directory, make_controller, fake_workers):
    settings = _settings(data_directory)
    settings.setValue("controls/channel", "11D")
    settings.sync()
    controller = make_controller(data_directory)
    tuning = _Recorder(controller.tuning_changed)

    controller.play_service("5C", 0xD220)

    assert tuning.last == ("5C", BAND_III_CHANNELS["5C"])
    assert controller.selected_sid == 0xD220
    assert controller.state is ReceptionState.CONNECTING
    assert fake_workers[-1].center_freq == BAND_III_CHANNELS["5C"] * 1e6


def test_the_play_button_is_ignored_while_connecting(data_directory, make_controller, fake_workers):
    controller = make_controller(data_directory)
    controller.start_reception()

    controller.play_service("11D", 0xD311)

    assert controller.channel == "5C"
    assert len(fake_workers) == 1


def test_an_announcement_names_the_service_it_comes_from(
    data_directory, make_controller, fake_workers
):
    controller = make_controller(data_directory)
    announcements = _Recorder(controller.announcement_changed)
    dsp = controller.dsp_worker
    status = AnnouncementStatus(None, alarm_enabled=False, source_sid=0xD220)

    dsp.announcement_ready.emit(status)
    assert announcements.last == (None, None)  # no reception: nothing to announce

    controller.start_reception()
    worker = fake_workers[-1]
    dsp.stream_restarted.emit(worker.tuning)
    dsp.services_updated.emit(ENSEMBLE_5C, list(SERVICES_5C), {})
    dsp.announcement_ready.emit(status)
    assert announcements.last == (status, "Dlf Kultur")


def test_without_a_stick_the_gain_steps_of_the_r820t_apply(tmp_path, make_controller):
    controller = make_controller(tmp_path, start=False)

    assert controller.gain_steps_db == FALLBACK_GAINS_DB
    assert controller.gain_range_db == (0.0, 49.6)


def test_a_restart_opens_the_device_again_once_it_is_closed(
    data_directory, make_controller, fake_workers
):
    controller = make_controller(data_directory)
    controller.start_reception()
    fake_workers[-1].connected.emit("Connected.")

    controller.restart_reception()
    assert fake_workers[0].stop_requested
    assert len(fake_workers) == 1

    fake_workers[0].stopped.emit()
    assert len(fake_workers) == 2
    assert controller.state is ReceptionState.CONNECTING


def test_no_restart_after_an_error(data_directory, make_controller, fake_workers):
    controller = make_controller(data_directory)
    errors = _Recorder(controller.error_occurred)
    controller.start_reception()
    controller.restart_reception()

    fake_workers[0].error.emit("Device unplugged.")

    assert errors.calls == [("Device unplugged.",)]
    assert len(fake_workers) == 1
    assert controller.state is ReceptionState.IDLE


def test_another_device_after_an_error_starts_reception_again(
    data_directory, make_controller, fake_workers
):
    """Back from a tuner of an RSPduo not plugged in, the stick plays on without a tap
    on the play button; the gain alone does not start it, nor a second change."""
    controller = make_controller(data_directory)
    stick = controller.device_settings
    controller.set_device_settings(replace(stick, device="rspduo"))
    controller.start_reception()
    assert fake_workers[-1].rspduo.tuner == 1
    fake_workers[-1].error.emit("No RSPduo found.")

    controller.set_device_settings(replace(controller.device_settings, lna_state=3))
    assert len(fake_workers) == 1
    controller.set_device_settings(stick)

    assert len(fake_workers) == 2
    assert not hasattr(fake_workers[-1], "rspduo")
    assert controller.state is ReceptionState.CONNECTING
    fake_workers[-1].connected.emit("Connected.")
    controller.stop_reception()  # by hand: no error to recover from
    fake_workers[-1].stopped.emit()
    controller.set_device_settings(replace(stick, device="rspduo", rspduo_tuner=2))
    assert len(fake_workers) == 2


def test_another_device_after_a_failed_scan_starts_nothing(
    data_directory, make_controller, fake_workers, fake_scanners
):
    controller = make_controller(data_directory)
    controller.start_reception()
    fake_workers[-1].error.emit("No RTL-SDR found.")
    controller.start_scan()
    fake_scanners[-1].error.emit("No RTL-SDR found.")
    fake_scanners[-1].scan_finished.emit(False)

    controller.set_device_settings(replace(controller.device_settings, device="rspduo"))

    assert len(fake_workers) == 1
    assert controller.state is ReceptionState.IDLE


def test_rtl_tcp_receives_from_the_server_with_the_settings_of_the_stick(
    data_directory, make_controller, fake_workers, fake_scanners
):
    controller = make_controller(data_directory)
    statuses = _Recorder(controller.status_changed)
    controller.set_device_settings(
        replace(
            controller.device_settings,
            device="rtltcp",
            gain_db=25.4,
            ppm=3,
            rtl_tcp_host="radio-pi",
            rtl_tcp_port=1235,
        )
    )
    assert controller.device_settings.is_rtlsdr
    assert controller.device_settings.rspduo() is None

    controller.start_reception()

    assert fake_workers[-1].rtl_tcp == RtlTcpSettings("radio-pi", 1235, 25.4, 3)
    assert statuses.last == ("Connecting to RTL_TCP radio-pi:1235...",)
    fake_workers[-1].error.emit("No answer from the rtl_tcp server radio-pi:1235.")
    controller.start_scan()
    assert fake_scanners[-1].rtl_tcp == RtlTcpSettings("radio-pi", 1235, 25.4, 3)
    assert fake_scanners[-1].rspduo is None


def test_the_rtl_tcp_server_is_stored(data_directory, make_controller):
    controller = make_controller(data_directory, start=False)
    controller.set_device_settings(
        replace(controller.device_settings, device="rtltcp", rtl_tcp_host="10.0.0.7")
    )
    settings = _settings(data_directory)
    settings.setValue("controls/rtl_tcp_port", 70000)  # edited by hand
    settings.sync()

    stored = make_controller(data_directory, start=False).device_settings

    assert stored.device == "rtltcp"
    assert stored.rtl_tcp() == RtlTcpSettings("10.0.0.7", 65535, DEFAULT_GAIN_DB, 0)


def test_another_rtl_tcp_server_after_an_error_starts_reception_again(
    data_directory, make_controller, fake_workers
):
    controller = make_controller(data_directory)
    first = replace(controller.device_settings, device="rtltcp", rtl_tcp_host="radio-pi")
    controller.set_device_settings(first)
    controller.start_reception()
    fake_workers[-1].error.emit("radio-pi:1234 refuses the connection.")

    controller.set_device_settings(replace(first, gain_db=33.8))
    assert len(fake_workers) == 1
    controller.set_device_settings(replace(first, rtl_tcp_port=1235))

    assert len(fake_workers) == 2
    assert fake_workers[-1].rtl_tcp.port == 1235
    assert controller.state is ReceptionState.CONNECTING


def test_the_tuner_of_the_rspduo_is_stored(data_directory, make_controller):
    controller = make_controller(data_directory, start=False)
    controller.set_device_settings(
        replace(controller.device_settings, device="rspduo", rspduo_tuner=2)
    )

    stored = make_controller(data_directory, start=False).device_settings

    assert (stored.device, stored.rspduo_tuner) == ("rspduo", 2)
    assert stored.rspduo().tuner == 2


@pytest.mark.parametrize(("device", "tuner"), [("rspduo1", 1), ("rspduo2", 2)])
def test_a_tuner_stored_as_a_device_is_the_rspduo_with_that_tuner(
    data_directory, make_controller, device, tuner
):
    """Up to 1.3.1 each tuner of the RSPduo was a device of its own."""
    settings = _settings(data_directory)
    settings.setValue("controls/device", device)
    settings.sync()

    stored = make_controller(data_directory, start=False).device_settings

    assert (stored.device, stored.rspduo_tuner) == ("rspduo", tuner)


def test_another_tuner_after_an_error_starts_reception_again(
    data_directory, make_controller, fake_workers
):
    controller = make_controller(data_directory)
    statuses = _Recorder(controller.status_changed)
    first = replace(controller.device_settings, device="rspduo")
    controller.set_device_settings(first)
    controller.start_reception()
    assert statuses.last == ("Connecting to RSPduo Tuner 1...",)
    fake_workers[-1].error.emit("RSPduo: tuner 1 is not available.")

    controller.set_device_settings(replace(first, lna_state=3))
    assert len(fake_workers) == 1
    controller.set_device_settings(replace(first, rspduo_tuner=2))

    assert len(fake_workers) == 2
    assert fake_workers[-1].rspduo.tuner == 2
    assert statuses.last == ("Connecting to RSPduo Tuner 2...",)


def _receive_5c_data(controller, worker, figs=FIGS_5C) -> list:
    """Starts reception, confirms the tuning and lets the FIC describe the data services
    of 5C (EPG Deutschland and GEPOS)."""
    controller.dsp_worker.stream_restarted.emit(worker.tuning)
    components = data_components(_ensemble(figs))
    controller.dsp_worker.data_services_updated.emit(components)
    return components


def test_every_decodable_data_service_is_decoded_by_default(
    data_directory, make_controller, fake_workers
):
    controller = make_controller(data_directory)
    toggled = _Recorder(controller._data_component_toggled)
    controller.start_reception()

    epg, gepos = _receive_5c_data(controller, fake_workers[-1])
    controller.dsp_worker.data_services_updated.emit([epg, gepos])  # the FIC repeats

    assert toggled.calls == [(epg.scid, True), (gepos.scid, True)]
    assert controller.data_components == [epg, gepos]


def test_a_data_service_is_started_only_once_the_fic_describes_it(
    data_directory, make_controller, fake_workers
):
    controller = make_controller(data_directory)
    toggled = _Recorder(controller._data_component_toggled)
    controller.start_reception()

    _receive_5c_data(controller, fake_workers[-1], FIGS_5C[:1])

    assert toggled.calls == []


def test_a_data_service_switched_off_stays_off_also_after_a_restart(
    data_directory, make_controller, fake_workers
):
    controller = make_controller(data_directory)
    controller.start_reception()
    epg, gepos = _receive_5c_data(controller, fake_workers[-1])
    controller.set_data_component_enabled(epg.scid, False)
    assert controller.is_data_component_switched_off(epg)
    controller.shutdown()

    again = make_controller(data_directory)
    toggled = _Recorder(again._data_component_toggled)
    again.start_reception()
    _receive_5c_data(again, fake_workers[-1])

    assert toggled.calls == [(gepos.scid, True)]
    again.set_data_component_enabled(epg.scid, True)
    assert not again.is_data_component_switched_off(epg)


def test_the_received_guide_is_stored_for_its_channel(
    data_directory, make_controller, fake_workers
):
    controller = make_controller(data_directory)
    stored = _Recorder(controller.guide_stored)
    controller.start_reception()
    epg, _gepos = _receive_5c_data(controller, fake_workers[-1])
    controller.dsp_worker.data_decoding_ready.emit((DataComponentStatus(epg.scid, epg),))
    logo = MotObject(1, CONTENT_TYPE_IMAGE, 3, "logo.png", b"\x89PNG logo")

    controller.dsp_worker.data_object_ready.emit(epg.scid, CarouselObject(logo))
    controller.dsp_worker.data_object_ready.emit(epg.scid, CarouselObject(logo))  # next turn

    assert len(stored.calls) == 1
    assert stored.last[0] == "5C" and stored.last[2]
    assert controller.guide_cache.load("5C").objects[0].body == b"\x89PNG logo"


# --- Announcements of other ensembles (FIG 0/25, FIG 0/26) ------------------------------
# Dlf Kultur on 5C takes road traffic from OE cluster 5, served by ensemble 1188 on 11D,
# where NDR 2 carries it.

EID_11D = 0x1188
NDR_2 = SERVICES_11D[0][0]
DLF_KULTUR = SERVICES_5C[1][0]
TRAFFIC_ON_11D = OeAnnouncement(5, 0x0002, new=True, scids=0, target_sid=NDR_2)
GO = AnnouncementStatus(None, alarm_enabled=False, oe_active=TRAFFIC_ON_11D, oe_eid=EID_11D)
QUIET = AnnouncementStatus(None, alarm_enabled=False)
# On 11D: NDR 2 carries an announcement in its subchannel (for its own cluster 2).
RUNNING_THERE = AnnouncementStatus(
    None, alarm_enabled=False, carried=Announcement(2, 0x0002, new=False, subchannel_id=5)
)


def _spin(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def _wait_for(condition, timeout_ms: int = 2000) -> None:
    """Runs the event loop until `condition()` holds. Not a fixed time: after many tests
    Windows delivers timer messages late (only with an empty message queue)."""
    for _ in range(timeout_ms // 10):
        if condition():
            return
        _spin(10)


@pytest.fixture
def receiving_5c(data_directory, make_controller, fake_workers, monkeypatch):
    """A controller playing Dlf Kultur on 5C with switching to announcements on; 11D is
    stored with its EId. The searches and the bar on going back last 50 ms."""
    monkeypatch.setattr(receiver_controller, "ANNOUNCEMENT_SEARCH_MS", 50)
    cache = EnsembleCache(data_directory / receiver_controller.ENSEMBLE_CACHE_FILENAME)
    cache.load()
    cache.put("11D", EnsembleInfo("NDR NDS", SERVICES_11D, 0.0, SOURCE_SCAN, eid=EID_11D))
    cache.save()
    controller = make_controller(data_directory)
    controller.set_announcement_switching(True)
    controller.play_service("5C", DLF_KULTUR)
    worker = fake_workers[-1]
    worker.connected.emit("Connected.")
    controller.dsp_worker.stream_restarted.emit(worker.tuning)
    controller.dsp_worker.services_updated.emit(ENSEMBLE_5C, list(SERVICES_5C), {})
    return controller, worker


def _arrive(controller, worker, status=RUNNING_THERE) -> None:
    """The signal processing reaches the new frequency and reports from there."""
    controller.dsp_worker.stream_restarted.emit(worker.tuning)
    controller.dsp_worker.announcement_ready.emit(status)


def test_an_announcement_of_another_ensemble_is_heard_there_until_it_ends(receiving_5c):
    controller, worker = receiving_5c
    visits = _Recorder(controller.announcement_visit_changed)
    dsp = controller.dsp_worker

    dsp.announcement_ready.emit(GO)
    assert (controller.channel, controller.selected_sid) == ("11D", NDR_2)
    assert worker.retuned_to[-1] == BAND_III_CHANNELS["11D"] * 1e6
    visit = controller.announcement_visit
    assert (visit.channel, visit.label) == ("11D", "NDR 2")
    assert (visit.origin_channel, visit.origin_sid, visit.origin_label) == (
        "5C",
        DLF_KULTUR,
        "Dlf Kultur",
    )

    dsp.announcement_ready.emit(QUIET)  # a straggler of 5C: the tuning is not confirmed
    assert controller.channel == "11D"
    _arrive(controller, worker)
    dsp.announcement_ready.emit(QUIET)  # over

    assert (controller.channel, controller.selected_sid) == ("5C", DLF_KULTUR)
    assert controller.announcement_visit is None
    assert visits.calls == [(visit,), (None,)]


def test_the_announcement_names_the_service_of_the_other_ensemble(receiving_5c):
    controller, _worker = receiving_5c
    controller.set_announcement_switching(False)
    announcements = _Recorder(controller.announcement_changed)

    controller.dsp_worker.announcement_ready.emit(GO)

    assert announcements.last == (GO, "NDR 2")
    assert controller.channel == "5C"  # switching is off
    assert controller.announcement_visit is None


def test_an_announcement_not_found_there_leads_back(receiving_5c):
    controller, _worker = receiving_5c
    controller.dsp_worker.announcement_ready.emit(GO)

    _wait_for(lambda: controller.announcement_visit is None)

    assert (controller.channel, controller.selected_sid) == ("5C", DLF_KULTUR)


def test_the_users_choice_ends_the_visit_there(receiving_5c):
    controller, worker = receiving_5c
    controller.dsp_worker.announcement_ready.emit(GO)
    _arrive(controller, worker)

    controller.select_service(NDR_2)
    controller.dsp_worker.announcement_ready.emit(QUIET)
    _spin(100)

    assert (controller.channel, controller.selected_sid) == ("11D", NDR_2)
    assert controller.announcement_visit is None


def test_returning_early_by_hand(receiving_5c):
    controller, worker = receiving_5c
    controller.dsp_worker.announcement_ready.emit(GO)
    _arrive(controller, worker)

    controller.return_from_announcement()

    assert (controller.channel, controller.selected_sid) == ("5C", DLF_KULTUR)


def test_stopping_there_leaves_the_service_left_behind_selected(receiving_5c):
    controller, worker = receiving_5c
    controller.dsp_worker.announcement_ready.emit(GO)

    worker.stopped.emit()

    assert controller.state is ReceptionState.IDLE
    assert (controller.channel, controller.selected_sid) == ("5C", DLF_KULTUR)
    assert controller.announcement_visit is None


def test_closing_there_keeps_the_service_left_behind_for_the_next_start(
    data_directory, receiving_5c
):
    controller, _worker = receiving_5c
    controller.dsp_worker.announcement_ready.emit(GO)

    controller.shutdown()

    settings = _settings(data_directory)
    assert settings.value("controls/channel") == "5C"
    assert int(settings.value("service/sid")) == DLF_KULTUR


def test_an_announcement_gone_to_does_not_lead_there_again(receiving_5c):
    controller, worker = receiving_5c
    dsp = controller.dsp_worker
    dsp.announcement_ready.emit(GO)
    _arrive(controller, worker)
    dsp.announcement_ready.emit(QUIET)  # over there, back to 5C

    # 5C reports nothing at first (its reports start afresh), then the same announcement.
    _arrive(controller, worker, QUIET)
    dsp.announcement_ready.emit(GO)
    assert controller.channel == "5C"

    # Once 5C no longer reports it, the next one leads there again.
    dsp.announcement_ready.emit(QUIET)
    _wait_for(lambda: not controller._revisit_timer.isActive())
    dsp.announcement_ready.emit(GO)
    assert controller.channel == "11D"


def test_the_channel_of_an_ensemble_not_stored_comes_from_fig_0_21(receiving_5c):
    controller, _worker = receiving_5c
    elsewhere = AnnouncementStatus(None, False, oe_active=TRAFFIC_ON_11D, oe_eid=0x1189)
    frequency_12d = round(BAND_III_CHANNELS["12D"] * 1000)
    controller.dsp_worker.frequency_information_updated.emit(0x10BC, {0x1189: (frequency_12d,)})

    controller.dsp_worker.announcement_ready.emit(elsewhere)

    assert (controller.channel, controller.selected_sid) == ("12D", NDR_2)
    assert controller.announcement_visit.label is None


def test_an_alarm_of_another_ensemble_is_heard_on_any_service_there(receiving_5c):
    controller, worker = receiving_5c
    alarm = OeAnnouncement(ALARM_CLUSTER, 0x0001, new=True, scids=0, target_sid=EID_11D)
    controller.dsp_worker.announcement_ready.emit(
        AnnouncementStatus(None, True, oe_active=alarm, oe_eid=EID_11D)
    )
    assert (controller.channel, controller.selected_sid) == ("11D", NDR_2)

    alarm_there = Announcement(ALARM_CLUSTER, 0x0001, new=True, subchannel_id=9)
    _arrive(controller, worker, AnnouncementStatus(None, True, active=alarm_there))
    assert controller.announcement_visit is not None
    assert not controller._visit_timer.isActive()  # found: the search is over


def test_an_announced_channel_where_another_ensemble_was_received_is_not_gone_to(
    receiving_5c,
):
    """FIG 0/21 of 5C announces 1189 on 11D, where 1188 was received: it holds
    elsewhere in the country, not here."""
    controller, _worker = receiving_5c
    elsewhere = AnnouncementStatus(None, False, oe_active=TRAFFIC_ON_11D, oe_eid=0x1189)
    frequency_11d = round(BAND_III_CHANNELS["11D"] * 1000)
    controller.dsp_worker.frequency_information_updated.emit(0x10BC, {0x1189: (frequency_11d,)})

    controller.dsp_worker.announcement_ready.emit(elsewhere)

    assert controller.channel == "5C"
    assert controller.announcement_visit is None


# --- Other ensembles carrying the selected service (FIG 0/24) --------------------------
# Dlf Kultur on 5C (EId 10BC) is said to be carried in 1188 as well, which was received
# on 11D, in 1189, which FIG 0/21 announces on 12D, and in 1205, of which nothing is known.

EID_5C = 0x10BC
DLF = SERVICES_5C[0][0]


def test_the_other_channels_are_bracketed_with_those_only_announced_marked():
    """A channel named twice counts once, as it comes first (received before announced);
    an ensemble with no channel known is left out."""
    others = (
        EnsembleLocation(EID_11D, "11D", received=True, label="NDR NDS"),
        EnsembleLocation(0x1189, "12D", received=False),
        EnsembleLocation(0x1190, "11D", received=False),
        EnsembleLocation(0x1205, None, received=False),
    )

    assert describe_other_channels(others) == "[11D, 12D?]"
    assert describe_other_channels(others[3:]) == ""
    assert describe_other_channels(()) == ""


def test_a_stored_list_is_described_by_its_source_and_time():
    taken = time.mktime((2026, 10, 6, 14, 30, 0, 0, 0, -1))  # local time

    scanned = EnsembleInfo(ENSEMBLE_5C, SERVICES_5C, taken, SOURCE_SCAN)
    assert describe_stored(scanned) == "scan of 2026-10-06 14:30"
    received = replace(scanned, source=SOURCE_LIVE)
    assert describe_stored(received) == "taken from reception on 2026-10-06 14:30"


def test_the_other_ensembles_of_the_selected_service_are_located(receiving_5c):
    controller, _worker = receiving_5c
    changes = _Recorder(controller.other_ensembles_changed)
    dsp = controller.dsp_worker
    frequency_12d = round(BAND_III_CHANNELS["12D"] * 1000)

    dsp.service_ensembles_updated.emit({DLF_KULTUR: (EID_5C, EID_11D, 0x1189, 0x1205)})
    assert controller.other_ensembles == ()  # 10BC could be any of them yet
    dsp.frequency_information_updated.emit(EID_5C, {0x1189: (frequency_12d,)})

    expected = (
        EnsembleLocation(EID_11D, "11D", received=True, label="NDR NDS"),
        EnsembleLocation(0x1189, "12D", received=False),
        EnsembleLocation(0x1205, None, received=False),
    )
    assert controller.other_ensembles == expected
    assert changes.calls == [(expected,)]


def test_the_other_ensembles_follow_the_selection_and_end_with_the_channel(receiving_5c):
    controller, worker = receiving_5c
    dsp = controller.dsp_worker
    dsp.frequency_information_updated.emit(EID_5C, {})
    dsp.service_ensembles_updated.emit({DLF_KULTUR: (EID_5C, EID_11D)})
    assert [location.eid for location in controller.other_ensembles] == [EID_11D]

    controller.select_service(DLF)  # carried here only
    assert controller.other_ensembles == ()
    controller.select_service(DLF_KULTUR)
    assert [location.eid for location in controller.other_ensembles] == [EID_11D]

    controller.select_channel("11D")
    assert controller.other_ensembles == ()
    dsp.service_ensembles_updated.emit({DLF_KULTUR: (EID_5C, EID_11D)})  # a straggler of 5C
    dsp.stream_restarted.emit(worker.tuning)
    assert controller.other_ensembles == ()


def _store_5c(data_directory: Path, eid: int, service_ensembles) -> None:
    """Stores 5C with its EId and what FIG 0/24 said, and 11D with its EId."""
    cache = EnsembleCache(data_directory / receiver_controller.ENSEMBLE_CACHE_FILENAME)
    cache.load()
    cache.put(
        "5C",
        EnsembleInfo(
            ENSEMBLE_5C, SERVICES_5C, 0.0, SOURCE_SCAN, eid=eid, service_ensembles=service_ensembles
        ),
    )
    cache.put("11D", EnsembleInfo("NDR NDS", SERVICES_11D, 0.0, SOURCE_SCAN, eid=EID_11D))
    cache.save()


def _stored_5c(data_directory: Path) -> EnsembleInfo:
    cache = EnsembleCache(data_directory / receiver_controller.ENSEMBLE_CACHE_FILENAME)
    cache.load()
    return cache.get("5C")


def test_the_stored_list_names_the_other_ensembles_until_another_ensemble_turns_up(
    data_directory, make_controller, fake_workers
):
    """Without reception and right after the tuning from the stored list; once 5C turns
    out to carry another ensemble, that list no longer applies."""
    _store_5c(data_directory, EID_5C, ((DLF_KULTUR, (EID_5C, EID_11D)),))
    controller = make_controller(data_directory)
    in_11d = (EnsembleLocation(EID_11D, "11D", received=True, label="NDR NDS"),)

    controller.select_service(DLF_KULTUR)
    assert controller.other_ensembles == in_11d

    controller.start_reception()
    worker = fake_workers[-1]
    controller.dsp_worker.stream_restarted.emit(worker.tuning)
    assert controller.other_ensembles == in_11d

    controller.dsp_worker.frequency_information_updated.emit(0x10BD, {})
    assert controller.other_ensembles == ()


def test_what_the_ensemble_sends_is_added_to_its_stored_list(
    data_directory, make_controller, fake_workers
):
    """Dlf gets nothing new and keeps its entry; Dlf Kultur comes anew, then its entry is
    withdrawn (the change event indication). The service list stays as it is."""
    _store_5c(data_directory, EID_5C, ((DLF, (EID_5C, 0x1205)), (DLF_KULTUR, (EID_5C, 0x1189))))
    controller = make_controller(data_directory)
    stored = _Recorder(controller.channel_stored)
    controller.play_service("5C", DLF_KULTUR)
    dsp = controller.dsp_worker
    dsp.stream_restarted.emit(fake_workers[-1].tuning)

    dsp.service_ensembles_updated.emit({DLF_KULTUR: (EID_5C, EID_11D)})
    assert _stored_5c(data_directory).service_ensembles == (
        (DLF, (EID_5C, 0x1205)),
        (DLF_KULTUR, (EID_5C, 0x1189)),
    )  # whose ensemble that is, is not known yet
    dsp.frequency_information_updated.emit(EID_5C, {})
    assert _stored_5c(data_directory).service_ensembles == (
        (DLF, (EID_5C, 0x1205)),
        (DLF_KULTUR, (EID_5C, EID_11D)),
    )

    dsp.service_ensembles_updated.emit({})
    assert _stored_5c(data_directory).service_ensembles == ((DLF, (EID_5C, 0x1205)),)
    assert stored.calls == []


def test_the_stored_list_of_another_ensemble_is_not_added_to(
    data_directory, make_controller, fake_workers
):
    _store_5c(data_directory, 0x10BD, ())
    controller = make_controller(data_directory)
    controller.play_service("5C", DLF_KULTUR)
    dsp = controller.dsp_worker
    dsp.stream_restarted.emit(fake_workers[-1].tuning)

    dsp.frequency_information_updated.emit(EID_5C, {})
    dsp.service_ensembles_updated.emit({DLF_KULTUR: (EID_5C, EID_11D)})

    assert _stored_5c(data_directory).service_ensembles == ()


def test_a_scan_keeps_what_the_stored_list_knew_of_fig_0_24(
    data_directory, make_controller, fake_scanners
):
    """The scan heard FIG 0/24 only for Dlf; Dlf Kultur stays from before."""
    _store_5c(data_directory, EID_5C, ((DLF_KULTUR, (EID_5C, EID_11D)),))
    controller = make_controller(data_directory)
    controller.start_scan()

    scanned = EnsembleInfo(
        ENSEMBLE_5C,
        SERVICES_5C,
        1.0,
        SOURCE_SCAN,
        eid=EID_5C,
        service_ensembles=((DLF, (EID_5C, 0x1205)),),
    )
    fake_scanners[-1].channel_scanned.emit("5C", scanned)
    fake_scanners[-1].scan_finished.emit(True)

    assert _stored_5c(data_directory).service_ensembles == (
        (DLF, (EID_5C, 0x1205)),
        (DLF_KULTUR, (EID_5C, EID_11D)),
    )


def test_the_scan_reports_its_progress(data_directory, make_controller, fake_scanners):
    controller = make_controller(data_directory)
    status = _Recorder(controller.status_changed)
    controller.start_scan()
    assert status.last == ("Scan: connecting to RTL-SDR...",)

    scanner = fake_scanners[-1]
    channel = f"Scan 8/38: channel 5C ({BAND_III_CHANNELS['5C']:.3f} MHz)"
    scanner.channel_started.emit("5C", 7, 38)
    assert status.last == (f"{channel} - 0 found so far",)
    scanner.channel_progress.emit("5C", 38, 40, ENSEMBLE_5C, 2)
    assert status.last == (
        f"{channel} - 38/40 FIBs error-free, ensemble {ENSEMBLE_5C}, 2 stations - 0 found so far",
    )
    scanner.channel_scanned.emit("5C", EnsembleInfo(ENSEMBLE_5C, SERVICES_5C, 1.0, SOURCE_SCAN))
    scanner.channel_scanned.emit("5D", None)
    scanner.scan_finished.emit(True)
    assert status.last == (
        "Scan finished: 2 channels searched, 1 ensemble found (green in the channel selection).",
    )


# --- Recording directory --------------------------------------------------------------


def test_the_recording_directory_is_stored_and_restored(make_controller, data_directory, tmp_path):
    controller = make_controller(data_directory)
    assert controller.recording_directory is None
    assert not controller.recording_directory_available
    changes = _Recorder(controller.recording_directory_changed)

    recordings = tmp_path / "stick" / "Aufnahmen"
    controller.set_recording_directory(str(recordings))

    assert controller.recording_directory == recordings
    assert changes.calls == [(recordings,)]
    # The drive may be plugged in later: the choice stands, only not available.
    assert not controller.recording_directory_available
    recordings.mkdir(parents=True)
    assert controller.recording_directory_available
    controller.set_recording_directory(recordings)  # the same again: nothing happens
    assert len(changes.calls) == 1
    assert _settings(data_directory).value("recording/directory") == str(recordings)

    again = make_controller(data_directory)
    assert again.recording_directory == recordings

    again.set_recording_directory(None)
    assert again.recording_directory is None
    assert not _settings(data_directory).contains("recording/directory")


# --- Recording ------------------------------------------------------------------------


def test_the_recorder_follows_directory_station_and_ring_size(
    make_controller, data_directory, tmp_path
):
    from recording.recorder import NO_DIRECTORY, NO_STATION, WAITING

    controller = make_controller(data_directory)
    assert controller.ring_size == 10
    assert controller.recordings == () and controller.kept_recordings == ()
    assert controller.recorder_status.message == NO_DIRECTORY
    statuses = _Recorder(controller.recorder_status_changed)

    stick = tmp_path / "stick"
    stick.mkdir()
    controller.set_recording_directory(stick)
    _wait_for(lambda: controller.recorder_status.message == NO_STATION)
    assert controller.recorder_status.message == NO_STATION

    # The service chosen is named from the stored list; the recorder waits for audio.
    controller.select_service(SERVICES_5C[0][0])
    _wait_for(lambda: controller.recorder_status.message == WAITING)
    assert controller.recorder_status.message == WAITING
    assert statuses.calls[-1][0].message == WAITING

    controller.set_ring_size(5)
    assert _settings(data_directory).value("recording/ring_titles", type=int) == 5
    controller.set_ring_size(99)
    assert controller.ring_size == 30
    again = make_controller(data_directory)
    assert again.ring_size == 30

    # Nothing to keep or delete yet: nothing happens, nothing breaks.
    controller.keep_recording("x")
    controller.delete_recording("x")
    _spin(50)
    assert controller.recordings == ()


def test_the_label_delay_is_kept_per_station_with_a_default(make_controller, data_directory):
    controller = make_controller(data_directory)
    changes = _Recorder(controller.label_delay_changed)
    assert controller.label_delay_s == 0 and controller.selected_service_label is None

    controller.set_label_delay_s(30)  # no station: the default
    assert controller.default_label_delay_s == 30 and len(changes.calls) == 1

    controller.select_service(SERVICES_5C[0][0])
    assert controller.selected_service_label == "Dlf"
    assert controller.label_delay_s == 30 and not controller.label_delay_is_station_own
    controller.set_label_delay_s(45)
    assert controller.label_delay_s == 45 and controller.label_delay_is_station_own
    assert controller.default_label_delay_s == 30
    controller.set_label_delay_s(99)  # clamped
    assert controller.label_delay_s == 60

    controller.select_service(SERVICES_5C[1][0])
    assert controller.label_delay_s == 30
    controller.clear_station_label_delay()  # nothing of its own: nothing happens
    controller.select_service(SERVICES_5C[0][0])
    controller.clear_station_label_delay()
    assert controller.label_delay_s == 30 and not controller.label_delay_is_station_own

    controller.set_label_delay_s(-10)
    settings = _settings(data_directory)
    assert settings.value("recording/label_delay_s", type=int) == 30
    assert json.loads(settings.value("recording/station_label_delays")) == {"D210": -10}
    again = make_controller(data_directory)
    assert again.label_delay_s == -10 and again.default_label_delay_s == 30


def test_a_recording_plays_in_place_of_the_station_until_the_station_changes(
    make_controller, data_directory, tmp_path, monkeypatch
):
    from dataclasses import asdict

    from recording.mp4 import write_m4a
    from recording.playback import LOADING, PAUSED, PLAYING
    from recording.recorder import Recording, write_sidecar
    from tests.test_audio_player import FORMAT_48K_STEREO, _FakeDecoder

    monkeypatch.setattr("audio.player.AacDecoder", _FakeDecoder)
    controller = make_controller(data_directory, start=False)
    monkeypatch.setattr(controller.audio_player, "_setup_sink", lambda *args: None)
    controller.start()
    stick = tmp_path / "stick"
    stick.mkdir()
    path = stick / "2026-10-09 18.07 Dlf – Queen – Radio Ga Ga.m4a"
    write_m4a(path, [bytes([n]) * 60 for n in range(75)], FORMAT_48K_STEREO)  # 3 s
    write_sidecar(
        Recording(
            id=path.stem,
            path=path,
            kept=True,
            complete=True,
            codec="DAB+",
            station="Dlf",
            ensemble="DR Deutschland",
            channel="5C",
            sid=0xD210,
            title="Radio Ga Ga",
            artist="Queen",
            text="",
            start=1_791_000_000.0,
            duration=3.0,
            size=path.stat().st_size,
            lost=0,
            bitrate=96,
            audio_format=asdict(FORMAT_48K_STEREO),
        )
    )
    states = _Recorder(controller.playback_changed)
    failures = _Recorder(controller.playback_failed)
    controller.set_audio_enabled(False)
    controller.set_recording_directory(stick)
    _wait_for(lambda: len(controller.kept_recordings) == 1)
    recording = controller.kept_recordings[0]
    assert controller.playback is None and controller.playback_recording is None

    controller.play_recording(recording.id)
    assert controller.playback.state == LOADING and controller.playback_recording == recording
    assert controller.audio_enabled, "whoever asks to hear a recording wants to hear it"
    _wait_for(lambda: controller.playback is not None and controller.playback.state == PLAYING)
    assert controller.playback.recording_id == recording.id
    assert controller.playback.duration == pytest.approx(3.0)

    controller.pause_playback()
    _wait_for(lambda: controller.playback.state == PAUSED)
    controller.toggle_playback()
    _wait_for(lambda: controller.playback.state == PLAYING)
    controller.seek_playback(2.0)
    _wait_for(lambda: controller.playback.position >= 1.9)
    assert controller.playback.position == pytest.approx(2.0, abs=0.05)

    controller.select_service(SERVICES_5C[0][0])
    assert controller.playback is None and states.last == (None,)
    controller.play_recording("nothing")
    assert controller.playback is None

    # A file that cannot be read any more: the surface is told, nothing plays.
    path.unlink()
    controller.play_recording(recording.id)
    _wait_for(lambda: failures.calls != [])
    # The text comes from the recorder (`TitleRecorder.load`).
    assert len(failures.calls) == 1 and failures.last[0]
    assert controller.playback is None

    # Deleting the title playing stops it first.
    path.write_bytes(b"")
    write_m4a(path, [bytes([n]) * 60 for n in range(75)], FORMAT_48K_STEREO)
    controller.play_recording(recording.id)
    _wait_for(lambda: controller.playback is not None and controller.playback.state == PLAYING)
    controller.delete_recording(recording.id)
    assert controller.playback is None
    _wait_for(lambda: controller.kept_recordings == ())


def test_the_stop_delay_and_the_switch_are_kept_like_the_label_delay(
    make_controller, data_directory
):
    from recording.recorder import DISABLED

    controller = make_controller(data_directory)
    changes = _Recorder(controller.stop_delay_changed)
    assert controller.stop_delay_s == 0 and controller.recording_enabled

    controller.set_stop_delay_s(-10)  # no station: the default
    assert controller.default_stop_delay_s == -10 and len(changes.calls) == 1
    controller.select_service(SERVICES_5C[0][0])
    assert controller.stop_delay_s == -10 and not controller.stop_delay_is_station_own
    controller.set_stop_delay_s(15)
    assert controller.stop_delay_s == 15 and controller.stop_delay_is_station_own
    assert controller.default_stop_delay_s == -10
    controller.clear_station_stop_delay()
    assert controller.stop_delay_s == -10 and not controller.stop_delay_is_station_own
    controller.set_stop_delay_s(20)
    settings = _settings(data_directory)
    assert settings.value("recording/stop_delay_s", type=int) == -10
    assert json.loads(settings.value("recording/station_stop_delays")) == {"D210": 20}

    switched = _Recorder(controller.recording_enabled_changed)
    controller.set_recording_enabled(False)
    assert not controller.recording_enabled and switched.last == (False,)
    assert settings.value("recording/enabled", type=bool) is False
    stick = data_directory / "stick"
    stick.mkdir()
    controller.set_recording_directory(stick)
    _wait_for(lambda: controller.recorder_status.message == DISABLED)
    assert controller.recorder_status.message == DISABLED

    again = make_controller(data_directory)
    assert again.stop_delay_s == 20 and not again.recording_enabled
