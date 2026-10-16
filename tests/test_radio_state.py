"""The state of the radio for programs outside it (`net.radio_state`): the sections built
from what the controller, the signal processing and the audio output report, and the
commands, on a controller without a device (`tests.fake_receiver`)."""

from __future__ import annotations

import os
import threading
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QEventLoop, QTimer

import i18n
from dab.announcement import AnnouncementStatus, announcement_types
from dab.dls import DlPlus, DlPlusContentType, DynamicLabel
from dab.ensemble_cache import SOURCE_SCAN, EnsembleCache, EnsembleInfo
from dab.fig import Announcement, AudioComponent, Subchannel
from dab.mot import CONTENT_TYPE_IMAGE, MotObject
from dab.programme_type import programme_type_name
from dab.slideshow import PARAM_TRIGGER_TIME
from dsp.pipeline import SubchannelStatus
from dsp.superframe import SuperframeResult
from i18n import tr
from net.radio_state import COMMANDS, SECTIONS, RadioStateModel
from recording.playback import PAUSED, PLAYING, PlaybackState
from recording.recorder import RECORDING, RecorderStatus, Recording
from tests.fake_receiver import ENSEMBLE_5C, SERVICES_5C, SERVICES_11D
from ui import receiver_controller
from ui.receiver_controller import APPLICATION_VERSION, ReceptionState
from ui.touch.quality import ReceptionQuality

DLF, DLF_KULTUR = (sid for sid, _ in SERVICES_5C)
NDR_2 = SERVICES_11D[0][0]
START = 1_791_000_000.0


class _Log:
    """The sections the model reports as changed, in order."""

    def __init__(self, model: RadioStateModel) -> None:
        self.names: list[str] = []
        model.changed.connect(self.names.append)

    def count(self, name: str) -> int:
        return self.names.count(name)

    def clear(self) -> None:
        self.names.clear()


class _Calls:
    """Collects what a signal emits."""

    def __init__(self, signal) -> None:
        self.calls: list[tuple] = []
        signal.connect(lambda *args: self.calls.append(args))


def _spin(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def _wait_for(condition, timeout_ms: int = 2000) -> None:
    for _ in range(timeout_ms // 10):
        if condition():
            return
        _spin(10)


def _recording(
    number: int, title: str = "", *, kept=False, complete=True, duration=200.0
) -> Recording:
    stem = f"2026-10-09 18.{number:02d} Dlf"
    return Recording(
        id=stem,
        path=Path("/stick/Puffer") / (stem + ".m4a"),
        kept=kept,
        complete=complete,
        codec="DAB+",
        station="Dlf",
        ensemble=ENSEMBLE_5C,
        channel="5C",
        sid=DLF,
        title=title,
        artist="",
        text=title,
        start=START + number * 60,
        duration=duration,
        size=1000,
        lost=0,
        bitrate=96,
        audio_format={},
    )


def _receive(controller, fake_workers, label=ENSEMBLE_5C, services=SERVICES_5C):
    """Starts reception, confirms the tuning and lets the FIC deliver the list."""
    controller.start_reception()
    worker = fake_workers[-1]
    worker.connected.emit("Connected.")
    dsp = controller.dsp_worker
    dsp.stream_restarted.emit(worker.tuning)
    dsp.services_updated.emit(label, list(services), {})
    return worker


@pytest.fixture
def radio(data_directory, make_controller):
    """A model on a started controller with 5C stored; the throttle lasts 50 ms."""
    controller = make_controller(data_directory, start=False)
    model = RadioStateModel(controller, throttle_ms=50)
    controller.start()
    return model, controller


def test_the_device_names_the_tuner_of_the_rspduo_and_follows_a_change(radio):
    """Also without reception, which a change of the device does not start."""
    model, controller = radio
    changes: list[str] = []
    model.changed.connect(changes.append)

    controller.set_device_settings(replace(controller.device_settings, device="rspduo"))
    assert model.section("receiver")["device"] == "RSPduo(1)"
    controller.set_device_settings(replace(controller.device_settings, rspduo_tuner=2))
    assert model.section("receiver")["device"] == "RSPduo(2)"
    controller.set_device_settings(replace(controller.device_settings, lna_state=3))

    assert changes == ["receiver", "receiver"]
    assert controller.state is ReceptionState.IDLE


def test_sections_after_start_show_the_stored_list(radio):
    model, _controller = radio

    assert set(model.snapshot()) == set(SECTIONS)
    receiver = model.section("receiver")
    assert receiver["state"] == "idle"
    assert receiver["channel"] == "5C"
    assert receiver["ensemble"] == ENSEMBLE_5C
    assert receiver["device"] == "RTL-SDR"
    assert receiver["version"] == APPLICATION_VERSION
    services = model.section("services")
    assert services["channel"] == "5C"
    assert services["live"] is False
    assert [(s["sid"], s["name"]) for s in services["list"]] == [
        ("D210", "Dlf"),
        ("D220", "Dlf Kultur"),
    ]
    assert model.section("service")["sid"] is None
    assert model.section("now_playing")["text"] is None
    assert model.slide is None
    assert set(COMMANDS) == set(model._commands)


def test_reception_fills_receiver_and_live_services(radio, fake_workers):
    model, controller = radio
    log = _Log(model)

    controller.start_reception()
    assert model.section("receiver")["state"] == "connecting"
    worker = fake_workers[-1]
    worker.connected.emit("Connected.")
    assert model.section("receiver")["state"] == "receiving"
    assert model.section("receiver")["status"] == "Connected."

    dsp = controller.dsp_worker
    dsp.stream_restarted.emit(worker.tuning)
    dsp.services_updated.emit(ENSEMBLE_5C, list(SERVICES_5C), {DLF: 1})
    services = model.section("services")
    assert services["live"] is True
    assert services["complete"] is False
    # Until the received list is complete, the stored one stays on display, as on
    # the surfaces; it knows no programme types.
    assert services["list"][0]["programme_type"] is None
    dsp.service_list_completed.emit()
    services = model.section("services")
    assert services["complete"] is True
    # The name as the surfaces show it (code 1, news), translated.
    assert services["list"][0]["programme_type"] == programme_type_name(1).full
    dsp.frequency_information_updated.emit(0x10BC, {})
    assert model.section("receiver")["ensemble_id"] == "10BC"
    assert log.count("services") >= 2

    # A device error ends reception; the message stays until the next start.
    worker.error.emit("Stick unplugged")
    receiver = model.section("receiver")
    assert receiver["state"] == "idle"
    assert receiver["error"] == "Stick unplugged"
    assert receiver["status"] == "Error: Stick unplugged"
    controller.start_reception()
    assert model.section("receiver")["error"] is None


def test_selected_service_now_playing_and_slide(radio, fake_workers):
    model, controller = radio
    _receive(controller, fake_workers)
    controller.select_service(DLF)
    service = model.section("service")
    assert service["sid"] == "D210"
    assert service["name"] == "Dlf"
    assert service["codec"] is None

    dsp = controller.dsp_worker
    subchannel = Subchannel(5, 0, 72, is_uep=False, protection_index=2, eep_option=0)
    dsp.subchannel_info_ready.emit(SubchannelStatus(AudioComponent(5, 63), subchannel))
    controller.audio_player.format_changed.emit("48 kHz, Stereo, SBR")
    service = model.section("service")
    assert service["codec"] == "DAB+"
    assert service["bitrate_kbps"] == subchannel.bitrate
    assert service["audio_format"] == "48 kHz, Stereo, SBR"

    objects = {
        DlPlusContentType.ITEM_TITLE: "Take On Me",
        DlPlusContentType.ITEM_ARTIST: "a-ha",
    }
    dsp.dynamic_label_ready.emit(DynamicLabel("a-ha - Take On Me", DlPlus(False, True, objects)))
    assert model.section("now_playing") == {
        "text": "a-ha - Take On Me",
        "title": "Take On Me",
        "artist": "a-ha",
        "item_running": True,
    }
    # No item on air: the text stays, title and artist do not apply.
    dsp.dynamic_label_ready.emit(DynamicLabel("Nachrichten", DlPlus(False, False, objects)))
    assert model.section("now_playing")["title"] is None
    assert model.section("now_playing")["item_running"] is False

    log = _Log(model)
    # Stations trigger their images with "NOW" (four zero bytes, `dab.slideshow`).
    now = {PARAM_TRIGGER_TIME: bytes(4)}
    dsp.slide_ready.emit(MotObject(1, CONTENT_TYPE_IMAGE, 1, "slide.jpg", b"\xff\xd8jpeg", now))
    assert log.names == ["slide"]
    assert model.slide.data == b"\xff\xd8jpeg"
    assert model.slide.content_type == "image/jpeg"
    assert model.slide.name == "slide.jpg"

    # Another service: everything of the one before is void.
    controller.select_service(DLF_KULTUR)
    assert model.slide is None
    assert model.section("now_playing")["text"] is None
    assert model.section("service")["codec"] is None
    assert model.section("service")["name"] == "Dlf Kultur"


def test_signal_section_is_throttled(radio, fake_workers):
    model, controller = radio
    _receive(controller, fake_workers)
    controller.select_service(DLF)
    log = _Log(model)
    dsp = controller.dsp_worker

    frames = (
        SuperframeResult(sync_ok=True, rs_corrections=2),
        SuperframeResult(sync_ok=True, rs_uncorrectable_blocks=1),
    )
    dsp.superframe_ready.emit(frames[0])
    assert log.count("signal") == 1  # the first change at once
    dsp.superframe_ready.emit(frames[1])
    dsp.frequency_offset_ready.emit(-120.0)
    assert log.count("signal") == 1  # the rest waits for the throttle
    _wait_for(lambda: log.count("signal") == 2)
    signal = model.section("signal")
    assert signal["frames"] == {
        "unit": "Superframes",
        "synced": 2,
        "total": 2,
        "lost": 1,
        "rs_corrections": 2,
    }
    # The words of the radio page for the same frames.
    quality = ReceptionQuality()
    for frame in frames:
        quality.add_superframe(frame)
    assert signal["quality"] == {"bars": 1, "text": quality.describe()}
    assert signal["frequency_offset_hz"] == -120.0
    assert signal["load"] is None

    controller.select_service(DLF_KULTUR)
    _wait_for(lambda: model.section("signal")["frames"]["total"] == 0)
    assert model.section("signal")["quality"] == {"bars": None, "text": None}


def test_audio_section_and_commands(radio):
    model, controller = radio
    log = _Log(model)

    assert model.execute("set_volume", {"percent": 40}) is None
    assert model.section("audio")["volume_percent"] == 40
    assert model.section("audio")["volume_device_percent"] is None  # a loudspeaker
    assert model.execute("set_muted", {"muted": "true"}) is None
    assert model.section("audio")["muted"] is True
    assert controller.audio_enabled is False
    assert model.execute("set_muted", {"muted": False}) is None
    assert model.section("audio")["muted"] is False
    assert model.execute("set_equalizer_enabled", {"enabled": 1}) is None
    assert model.section("audio")["equalizer"]["enabled"] is True
    assert model.execute("set_output", {"id": ""}) is None
    assert log.count("audio") >= 4

    assert model.execute("set_output", {"id": "not-there"}) == "Unknown output “not-there”."
    assert model.execute("set_volume", {"percent": 101}) == ("“percent” must be between 0 and 100.")
    assert model.execute("set_volume", {"percent": "loud"}) == (
        "“percent” must be a number, not 'loud'."
    )
    assert model.execute("set_volume", {}) == "The argument “percent” is missing."
    assert model.execute("set_muted", {"muted": "maybe"}) == (
        "“muted” must be true or false, not 'maybe'."
    )
    assert controller.volume_percent == 40


def test_commands_follow_the_state(radio, fake_workers, fake_scanners):
    model, controller = radio

    assert model.execute("dance") == "Unknown command “dance”."
    assert model.execute("stop_reception") == "Reception is not running."
    assert model.execute("cancel_scan") == "No scan is running."
    assert model.execute("select_channel", {"channel": "99Z"}) == "Unknown channel “99Z”."
    assert model.execute("select_service", {"name": "dlf  KULTUR"}) is None
    assert controller.selected_sid == DLF_KULTUR
    assert model.execute("select_service", {"name": "WDR 2"}) == (
        "Station “WDR 2” is not in the station list."
    )
    assert model.execute("select_service", {"sid": "D311"}) == (
        "Station D311 is not in the station list."
    )
    assert model.execute("select_service", {"sid": "xyz"}) == "“sid” must be an SId, not 'xyz'."
    assert model.execute("select_service", {}) == "The argument “sid” or “name” is missing."

    assert model.execute("start_scan") is None
    assert controller.state is ReceptionState.SCANNING
    assert model.execute("select_channel", {"channel": "5a"}) == "The scan is running."
    assert model.execute("start_reception") == "The scan is running."
    assert model.execute("start_scan") == "The scan is running."
    scanner = fake_scanners[-1]
    scanner.channel_started.emit("5A", 0, 38)
    assert model.section("receiver")["scan"] == {"channel": "5A", "index": 0, "total": 38}
    assert model.execute("cancel_scan") is None
    assert scanner.cancelled
    scanner.scan_finished.emit(False)
    assert controller.state is ReceptionState.IDLE
    assert model.section("receiver")["scan"] is None

    assert model.execute("start_reception") is None
    assert model.execute("start_reception") == "Reception is just starting."
    fake_workers[-1].connected.emit("Connected.")
    assert model.execute("start_reception") == "Reception is already running."
    assert model.execute("start_scan") == "Reception is running; stop it before scanning."
    assert model.execute("select_channel", {"channel": "11d"}) is None
    assert controller.channel == "11D"
    assert model.execute("stop_reception") is None
    assert fake_workers[-1].stop_requested


def test_play_tunes_selects_and_starts(data_directory, make_controller, fake_workers):
    cache = EnsembleCache(data_directory / receiver_controller.ENSEMBLE_CACHE_FILENAME)
    cache.load()
    cache.put("11D", EnsembleInfo("NDR NDS", SERVICES_11D, 0.0, SOURCE_SCAN))
    cache.save()
    controller = make_controller(data_directory, start=False)
    model = RadioStateModel(controller)
    controller.start()

    assert model.execute("play", {"channel": "11D", "name": "NDR 2"}) is None
    assert controller.channel == "11D"
    assert controller.selected_sid == NDR_2
    assert fake_workers[-1].started
    # Reception started: until the received list is complete, the stored one of the
    # channel stays on display, as on the surfaces.
    assert model.section("services")["live"] is True
    assert model.section("services")["complete"] is False
    assert model.section("services")["list"][0]["name"] == "NDR 2"
    assert model.section("receiver")["ensemble"] == "NDR NDS"
    assert model.section("receiver")["channel"] == "11D"
    assert model.execute("play", {"name": "NDR 2"}) == "Reception is just starting."
    fake_workers[-1].connected.emit("Connected.")

    assert model.execute("play", {"channel": "11D", "name": "Dlf"}) == (
        "Station “Dlf” is not in the station list."
    )
    # A SId the stored list does not know may still be on air there.
    assert model.execute("play", {"channel": "11D", "sid": "0xABCD"}) is None
    assert controller.selected_sid == 0xABCD
    # Without a channel, the list on display is meant.
    assert model.execute("play", {"name": "NDR 2"}) is None
    assert controller.selected_sid == NDR_2
    assert model.execute("play", {"channel": "5C", "sid": "D210"}) is None
    assert controller.channel == "5C"
    assert controller.selected_sid == DLF


def test_favourites(radio):
    model, controller = radio

    assert model.execute("set_favourite", {"name": "Dlf", "favourite": True}) is None
    assert model.section("favourites") == {
        "list": [{"channel": "5C", "sid": "D210", "name": "Dlf", "ensemble": ENSEMBLE_5C}]
    }
    assert model.section("services")["list"][0]["favourite"] is True
    controller.select_service(DLF)
    assert model.section("service")["favourite"] is True

    # A station of a channel not stored: known by its SId only.
    assert (
        model.execute("set_favourite", {"channel": "11D", "sid": "D311", "favourite": True}) is None
    )
    assert [e["name"] for e in model.section("favourites")["list"]] == ["Dlf", "D311"]
    assert model.execute("set_favourite", {"sid": "D210", "favourite": False}) is None
    assert [e["sid"] for e in model.section("favourites")["list"]] == ["D311"]
    assert model.section("service")["favourite"] is False
    assert model.execute("set_favourite", {"sid": "D210"}) == "The argument “favourite” is missing."


def test_recorder_sections_and_commands(radio):
    model, controller = radio
    kept = _Calls(controller._recording_kept)
    deleted = _Calls(controller._recording_deleted)
    running = _recording(3, "Drei", complete=False)
    ring = (_recording(1, "Eins"), _recording(2, "Zwei"))
    controller._on_recordings_changed(ring, (_recording(0, "Null", kept=True),))
    controller._on_recorder_status(RecorderStatus(True, RECORDING, running))

    recorder = model.section("recorder")
    assert recorder["recording"] is True
    assert recorder["message"] == RECORDING
    assert recorder["current"]["name"] == "Drei"
    assert recorder["current"]["complete"] is False
    assert recorder["ring_size"] == controller.ring_size
    assert recorder["directory"] is None
    assert recorder["available"] is False
    recordings = model.section("recordings")
    assert [r["name"] for r in recordings["ring"]] == ["Eins", "Zwei"]
    assert [r["name"] for r in recordings["kept"]] == ["Null"]
    assert recordings["ring"][0]["sid"] == "D210"
    assert recordings["ring"][0]["start"] == START + 60

    assert model.execute("keep_recording", {"id": "nope"}) == "Unknown recording “nope”."
    assert model.execute("keep_recording", {"id": running.id}) is None
    assert model.execute("keep_recording", {"id": ring[0].id}) is None
    assert kept.calls == [(running.id,), (ring[0].id,)]
    # The title recording cannot be deleted, a kept one can.
    assert model.execute("delete_recording", {"id": running.id}) == (
        f"Unknown recording “{running.id}”."
    )
    assert model.execute("delete_recording", {"id": "2026-10-09 18.00 Dlf"}) is None
    assert deleted.calls == [("2026-10-09 18.00 Dlf",)]


def test_playback_section(radio):
    model, controller = radio
    recording = _recording(1, "Eins")
    controller._on_recordings_changed((recording,), ())
    assert model.execute("pause_playback") == "No recording is playing."
    assert model.execute("seek_playback", {"position": 5}) == "No recording is playing."
    assert model.execute("play_recording", {"id": "nope"}) == "Unknown recording “nope”."

    log = _Log(model)
    controller._playback_source_id = recording.id
    controller._playback_recording = recording
    controller._on_playback_changed(PlaybackState(recording.id, PLAYING, 0.0, 200.0))
    assert model.section("playback") == {
        "id": recording.id,
        "name": "Eins",
        "state": PLAYING,
        "position": 0.0,
        "duration": 200.0,
    }
    assert log.count("playback") == 1
    # Only the position moves on: the first at once, then the throttle holds them.
    controller._on_playback_changed(PlaybackState(recording.id, PLAYING, 0.25, 200.0))
    controller._on_playback_changed(PlaybackState(recording.id, PLAYING, 0.5, 200.0))
    assert log.count("playback") == 2
    # A pause goes out at once.
    controller._on_playback_changed(PlaybackState(recording.id, PAUSED, 0.75, 200.0))
    assert log.count("playback") == 3
    assert model.section("playback")["state"] == PAUSED
    assert model.section("playback")["position"] == 0.75
    controller._on_playback_changed(None)
    assert model.section("playback")["id"] is None


def test_submit_runs_in_the_gui_thread_and_reports(radio):
    model, controller = radio
    # The player's first report of its output comes queued after the start, also
    # without a sound card; a new output starts at 10 %, which would overrule the
    # volume the command sets if the report came after it.
    reported: list[bool] = []
    controller.audio_outputs_changed.connect(lambda: reported.append(True))
    _wait_for(lambda: reported)
    assert reported
    finished: list[tuple] = []
    threads: list[int] = []

    def note(*args) -> None:
        finished.append(args)
        threads.append(threading.get_ident())

    model.command_finished.connect(note)
    worker = threading.Thread(target=model.submit, args=("set_volume", {"percent": 33}))
    worker.start()
    worker.join()
    _wait_for(lambda: finished)
    assert finished == [("set_volume", {"percent": 33}, None)]
    assert threads == [threading.get_ident()]
    assert controller.volume_percent == 33

    model.submit("dance")
    _wait_for(lambda: len(finished) == 2)
    assert finished[1] == ("dance", {}, "Unknown command “dance”.")


def test_announcement_section(radio, fake_workers):
    model, controller = radio
    _receive(controller, fake_workers)
    controller.select_service(DLF)
    traffic = Announcement(1, 0x0002, new=True, subchannel_id=5)
    status = AnnouncementStatus(None, alarm_enabled=False, active=traffic, source_sid=DLF_KULTUR)

    controller.dsp_worker.announcement_ready.emit(status)
    announcement = model.section("announcement")
    assert announcement["active"] == {
        "types": [announcement_types(0x0002)[0].short],  # road traffic flash
        "alarm": False,
        "new": True,
        "source": "Dlf Kultur",
    }
    assert announcement["elsewhere"] is None
    assert announcement["visit"] is None

    assert model.execute("return_from_announcement") == (
        "Reception is not at an announcement of another ensemble."
    )
    assert model.execute("set_announcement_switching", {"enabled": True}) is None
    assert model.section("announcement")["switching"] is True
    assert model.execute("set_announcement_switching", {"enabled": "aus"}) is None
    assert model.section("announcement")["switching"] is False

    controller.dsp_worker.announcement_ready.emit(AnnouncementStatus(None, alarm_enabled=False))
    assert model.section("announcement")["active"] is None


@pytest.fixture
def german():
    """The surfaces in German, as chosen on the radio; English again after the test."""
    i18n.set_language("de")
    yield
    i18n.set_language(i18n.SOURCE_LANGUAGE)


def test_the_remote_interfaces_stay_english_when_the_surfaces_are_german(
    german, data_directory, make_controller, fake_workers
):
    controller = make_controller(data_directory, start=False)
    model = RadioStateModel(controller, throttle_ms=50)
    controller.start()
    # What the surfaces show is German, what goes out stays English.
    assert controller.recorder_status.message == "Kein Aufnahmeordner gewählt"
    assert model.section("recorder")["message"] == "No recording folder chosen"

    controller.start_reception()
    assert model.section("receiver")["status"] == "Connecting to RTL-SDR..."
    worker = fake_workers[-1]
    worker.connected.emit(
        tr("RTL-SDR ready @ {frequency} MHz, {rate} MSps").format(frequency="178.352", rate="2.048")
    )
    assert model.section("receiver")["status"] == "RTL-SDR ready @ 178.352 MHz, 2.048 MSps"
    dsp = controller.dsp_worker
    dsp.stream_restarted.emit(worker.tuning)
    dsp.services_updated.emit(ENSEMBLE_5C, list(SERVICES_5C), {DLF: 1})
    dsp.service_list_completed.emit()
    assert model.section("services")["list"][0]["programme_type"] == "News"
    dsp.superframe_ready.emit(SuperframeResult(sync_ok=True, rs_corrections=2))
    _wait_for(lambda: model.section("signal")["quality"]["text"] is not None)
    quality = model.section("signal")["quality"]["text"]
    assert quality == i18n.english(model._quality.describe()) != model._quality.describe()

    controller.select_service(DLF)
    traffic = Announcement(1, 0x0002, new=True, subchannel_id=5)
    dsp.announcement_ready.emit(
        AnnouncementStatus(None, alarm_enabled=False, active=traffic, source_sid=DLF_KULTUR)
    )
    assert model.section("announcement")["active"]["types"] == ["Road Traffic flash"]

    assert model.execute("dance") == "Unknown command “dance”."
    assert model.execute("set_volume", {"percent": "loud"}) == (
        "“percent” must be a number, not 'loud'."
    )

    # A device error, raised and caught as in the worker.
    raised = RuntimeError(tr("No RTL-SDR device found. Is the stick connected?"))
    worker.error.emit(i18n.exception_text(raised))
    receiver = model.section("receiver")
    assert receiver["error"] == "No RTL-SDR device found. Is the stick connected?"
    assert receiver["status"] == "Error: No RTL-SDR device found. Is the stick connected?"
