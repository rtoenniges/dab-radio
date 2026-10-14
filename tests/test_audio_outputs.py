"""The choice of the output (`audio.outputs`, `AudioPlayer.set_output`,
`ReceiverController.set_audio_output`), the volume remembered for each output, and the
level handed to the sound system where it takes it (`audio.system_volume` stood in
for), on outputs that stand in for those of the system."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from typing import ClassVar

import pytest
from PyQt6.QtCore import QObject, pyqtSignal

from audio import outputs, player
from audio.outputs import AudioOutput, audio_outputs, is_bluetooth
from audio.player import AudioPlayer
from audio.system_volume import VOLUME_NORM, raw_of_share, raw_volume
from ui import receiver_controller


class _FakeDevice:
    def __init__(self, device_id: bytes, description: str) -> None:
        self._id = device_id
        self._description = description

    def id(self) -> bytes:
        return self._id

    def description(self) -> str:
        return self._description

    def isNull(self) -> bool:
        return not self._id


class _FakeMediaDevices(QObject):
    """Stands in for `QMediaDevices`: the outputs and the default are set by the test,
    `change` sends what Qt sends when a device appears or goes."""

    audioOutputsChanged = pyqtSignal()
    instances: ClassVar[list[_FakeMediaDevices]] = []
    outputs: ClassVar[list[_FakeDevice]] = []
    default: ClassVar[_FakeDevice] = _FakeDevice(b"", "")

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        _FakeMediaDevices.instances.append(self)

    @classmethod
    def audioOutputs(cls) -> list[_FakeDevice]:
        return list(cls.outputs)

    @classmethod
    def defaultAudioOutput(cls) -> _FakeDevice:
        return cls.default

    @classmethod
    def change(cls, outputs: list[_FakeDevice], default: _FakeDevice) -> None:
        cls.outputs, cls.default = outputs, default
        for instance in cls.instances:
            instance.audioOutputsChanged.emit()


SPEAKER = _FakeDevice(b"alsa_output.platform-soc_sound.stereo-fallback", "Built-in Audio Stereo")
HEADPHONES = _FakeDevice(b"bluez_output.F8_4E_17_01_02_03.1", "WH-1000XM4")
EARBUDS = _FakeDevice(b"bluez_output.A0_B1_C2_D3_E4_F5.1", "Galaxy Buds")


@pytest.fixture
def media_devices(monkeypatch):
    _FakeMediaDevices.instances.clear()
    _FakeMediaDevices.change([SPEAKER], SPEAKER)
    monkeypatch.setattr(outputs, "QMediaDevices", _FakeMediaDevices)
    monkeypatch.setattr(player, "QMediaDevices", _FakeMediaDevices)
    monkeypatch.setattr(receiver_controller, "QMediaDevices", _FakeMediaDevices)
    return _FakeMediaDevices


class _Recorder:
    def __init__(self, signal) -> None:
        self.calls: list[tuple] = []
        signal.connect(lambda *args: self.calls.append(args))

    @property
    def last(self) -> tuple:
        return self.calls[-1]


# --- audio.outputs --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("device_id", "name", "expected"),
    [
        ("bluez_output.F8_4E_17_01_02_03.1", "WH-1000XM4", True),
        ("bluez_sink.F8_4E_17_01_02_03.a2dp_sink", "WH-1000XM4", True),
        ("{0.0.0.00000000}.{guid}", "Kopfhörer (WH-1000XM4 Stereo)", True),
        ("{0.0.0.00000000}.{guid}", "Headset (WH-1000XM4 Hands-Free)", True),
        ("{0.0.0.00000000}.{guid}", "Lautsprecher (Realtek(R) Audio)", False),
        ("alsa_output.platform-soc_sound.stereo-fallback", "Built-in Audio Stereo", False),
    ],
)
def test_bluetooth_outputs_are_told_by_sink_name_or_windows_description(device_id, name, expected):
    assert is_bluetooth(device_id, name) == expected


def test_the_outputs_of_the_system_are_listed_with_the_default_marked(media_devices):
    media_devices.change([SPEAKER, HEADPHONES], HEADPHONES)

    listed = audio_outputs()

    assert listed == [
        AudioOutput(SPEAKER.id().decode(), "Built-in Audio Stereo", False, False),
        AudioOutput(HEADPHONES.id().decode(), "WH-1000XM4", True, True),
    ]


# --- The player -----------------------------------------------------------------------


class _FakeSink:
    def __init__(self) -> None:
        self.stopped = False
        self.stateChanged = _FakeSignal()

    def stop(self) -> None:
        self.stopped = True

    def deleteLater(self) -> None:
        pass


class _FakeSignal:
    def disconnect(self, _slot) -> None:
        pass


@pytest.fixture
def started_player(media_devices, monkeypatch):
    instance = AudioPlayer()
    monkeypatch.setattr(instance, "_setup_sink", lambda *args, **kwargs: None)
    reports = _Recorder(instance.output_changed)
    instance.start()
    yield instance, reports
    instance.stop()


def test_the_player_reports_the_default_and_follows_the_choice(started_player, media_devices):
    player_, reports = started_player
    assert reports.calls == [(SPEAKER.id(), "Built-in Audio Stereo")]

    # Chosen but not there: the default keeps playing until it appears.
    player_.set_output(HEADPHONES.id())
    assert len(reports.calls) == 1

    media_devices.change([SPEAKER, HEADPHONES], SPEAKER)
    assert reports.last == (HEADPHONES.id(), "WH-1000XM4")

    player_.set_output(None)
    assert reports.last == (SPEAKER.id(), "Built-in Audio Stereo")

    # Automatic: the default of the system moves to the headphones and back.
    media_devices.change([SPEAKER, HEADPHONES], HEADPHONES)
    assert reports.last == (HEADPHONES.id(), "WH-1000XM4")
    media_devices.change([SPEAKER], SPEAKER)
    assert reports.last == (SPEAKER.id(), "Built-in Audio Stereo")


def test_a_change_of_output_tears_the_sink_down_for_the_next_audio(started_player, media_devices):
    player_, _reports = started_player
    sink = _FakeSink()
    player_._sink = sink
    player_._sink_device = SPEAKER
    player_._playing = True

    media_devices.change([SPEAKER, HEADPHONES], HEADPHONES)

    assert sink.stopped and player_._sink is None and not player_._playing

    # The same device again: nothing happens to a sink playing to it.
    sink = _FakeSink()
    player_._sink = sink
    player_._sink_device = HEADPHONES
    media_devices.change([SPEAKER, HEADPHONES], HEADPHONES)
    assert not sink.stopped and player_._sink is sink


def test_a_device_chosen_is_played_to_anew_when_the_default_moves_away_from_it(
    started_player, media_devices
):
    """WirePlumber moves a stream whose own target was the default along with the default
    (2026-10-11: the loudspeaker chosen, headphones connecting, the radio near silent in
    them); a stream made anew to the device chosen, no longer the default, stays."""
    player_, reports = started_player
    player_.set_output(SPEAKER.id())
    sink = _FakeSink()
    player_._sink, player_._sink_device, player_._playing = sink, SPEAKER, True

    # The headphones appear, then become the default: the stream is made anew, still
    # to the loudspeaker.
    media_devices.change([SPEAKER, HEADPHONES], SPEAKER)
    assert not sink.stopped
    media_devices.change([SPEAKER, HEADPHONES], HEADPHONES)
    assert sink.stopped and player_._sink is None
    assert reports.last == (SPEAKER.id(), "Built-in Audio Stereo")

    # That one was not made to the default: changes of the default leave it alone …
    sink = _FakeSink()
    player_._sink, player_._sink_device = sink, SPEAKER
    media_devices.change([SPEAKER, HEADPHONES, EARBUDS], EARBUDS)
    media_devices.change([SPEAKER], SPEAKER)
    assert not sink.stopped
    # … until the loudspeaker was the default again and the default leaves it.
    media_devices.change([SPEAKER, HEADPHONES], HEADPHONES)
    assert sink.stopped


class _EndedSink(_FakeSink):
    """A sound card whose stream PipeWire has ended: it takes nothing more, while Qt
    goes on calling it active."""

    def bytesFree(self) -> int:
        return 0

    def bufferSize(self) -> int:
        return 96_000


def _waiting(player_, sink, device) -> None:
    """`sink` plays to `device`, with audio waiting for it."""
    player_._sink, player_._device, player_._sink_device = sink, sink, device
    player_._playing = True
    player_._bytes_per_second = 192_000
    player_._buffer.extend(bytes(19_200))


def test_a_sound_card_taking_no_more_audio_has_lost_its_device(
    started_player, media_devices, monkeypatch
):
    """Headphones switched off: PipeWire ends the stream pinned to their sink
    (`main.PINNED_STREAM_PROPERTIES`), and Qt 6.11 reports neither that nor the sink
    going, it keeps listing the headphones (on the Pi, 2026-10-11). Audio waiting and
    none taken for `STALL_SECONDS`: the device counts as gone and the default plays,
    until Qt reports a device appearing."""
    now = [100.0]
    monkeypatch.setattr(player, "_clock", lambda: now[0])
    player_, reports = started_player
    media_devices.change([SPEAKER, HEADPHONES], SPEAKER)
    player_.set_output(HEADPHONES.id())
    sink = _EndedSink()
    _waiting(player_, sink, HEADPHONES)
    player_._taken_at = now[0]

    now[0] += player.STALL_SECONDS / 2
    player_._flush()
    assert player_._sink is sink
    now[0] += player.STALL_SECONDS / 2
    player_._flush()
    assert sink.stopped and player_._sink is None and not player_._buffer
    assert reports.last == (SPEAKER.id(), "Built-in Audio Stereo")

    player_._follow_output()  # the check every 2 s: Qt still lists them, they stay out
    assert reports.last == (SPEAKER.id(), "Built-in Audio Stereo")
    media_devices.change([SPEAKER, HEADPHONES], SPEAKER)  # they connect again
    assert reports.last == (HEADPHONES.id(), "WH-1000XM4")


def test_a_lost_default_leaves_the_player_without_output(
    started_player, media_devices, monkeypatch
):
    now = [100.0]
    monkeypatch.setattr(player, "_clock", lambda: now[0])
    player_, reports = started_player
    media_devices.change([SPEAKER, HEADPHONES], HEADPHONES)
    _waiting(player_, _EndedSink(), HEADPHONES)
    player_._taken_at = now[0]
    now[0] += player.STALL_SECONDS
    player_._flush()
    assert reports.last == (None, "")  # nothing else to play to
    player_._flushes = player._OUTPUT_CHECK_FLUSHES - 1
    player_._flush()  # the check every 2 s: tried again, it may have stalled otherwise
    assert reports.last == (HEADPHONES.id(), "WH-1000XM4") and not player_._lost


def test_a_dropout_or_a_pause_is_no_device_lost(started_player, media_devices, monkeypatch):
    """Without audio waiting (the reception stalled) or while a recording is paused, a
    sound card taking nothing is as it should be."""
    now = [100.0]
    monkeypatch.setattr(player, "_clock", lambda: now[0])
    player_, _reports = started_player
    sink = _EndedSink()
    _waiting(player_, sink, SPEAKER)
    player_._buffer.clear()
    player_._taken_at = now[0]
    now[0] += 3 * player.STALL_SECONDS
    player_._flush()
    assert player_._sink is sink

    player_._buffer.extend(bytes(19_200))
    player_._paused = True
    now[0] += 3 * player.STALL_SECONDS
    player_._flush()
    assert player_._sink is sink and not player_._lost


# --- The controller -------------------------------------------------------------------


def test_the_choice_is_stored_and_handed_to_the_player(
    media_devices, data_directory, make_controller
):
    controller = make_controller(data_directory, start=False)
    handed = _Recorder(controller._audio_output_set)
    changes = _Recorder(controller.audio_outputs_changed)
    assert controller.audio_output_choice == ""

    controller.set_audio_output(HEADPHONES.id().decode())

    assert handed.last == (HEADPHONES.id(),)
    assert len(changes.calls) == 1
    controller.settings.sync()
    assert (
        make_controller(data_directory, start=False).audio_output_choice == HEADPHONES.id().decode()
    )

    controller.set_audio_output("")
    assert handed.last == (None,)


def test_starting_hands_the_stored_choice_to_the_player(
    media_devices, data_directory, make_controller
):
    controller = make_controller(data_directory, start=False)
    controller.set_audio_output(HEADPHONES.id().decode())
    controller.settings.sync()

    again = make_controller(data_directory, start=False)
    handed = _Recorder(again._audio_output_set)
    again.start()
    assert handed.calls == [(HEADPHONES.id(),)]


def test_every_output_keeps_its_own_volume(media_devices, data_directory, make_controller):
    controller = make_controller(data_directory, start=False)
    gains = _Recorder(controller._volume_changed)
    changes = _Recorder(controller.volume_changed)
    report = controller._audio_player.output_changed

    report.emit(SPEAKER.id(), "Built-in Audio Stereo")
    assert controller.audio_output_in_use == AudioOutput(
        SPEAKER.id().decode(), "Built-in Audio Stereo", False
    )
    assert controller.volume_percent == 10 and controller.volume_ceiling_db == 0  # new
    controller.set_volume_percent(80)
    controller.set_volume_ceiling_db(-6)

    # Headphones not played to before start at 10 % without a ceiling, however loud the
    # loudspeaker plays (-18 dB here) …
    report.emit(HEADPHONES.id(), "WH-1000XM4")
    assert controller.audio_output_in_use.bluetooth
    assert controller.volume_percent == 10 and controller.volume_ceiling_db == 0
    assert controller.volume_db == pytest.approx(-54.0)
    assert gains.last == (pytest.approx(10 ** (-54 / 20)),)
    controller.set_volume_percent(40)

    # … as do other ones, however loud the headphones play …
    report.emit(EARBUDS.id(), "Galaxy Buds")
    assert controller.volume_percent == 10 and controller.volume_ceiling_db == 0
    report.emit(HEADPHONES.id(), "WH-1000XM4")
    assert controller.volume_percent == 40  # played to before: its own level

    # … and the loudspeaker gets its own back, slider and player told.
    before = len(changes.calls)
    report.emit(SPEAKER.id(), "Built-in Audio Stereo")
    assert controller.volume_percent == 80 and controller.volume_ceiling_db == -6
    assert len(changes.calls) == before + 1
    assert gains.last == (pytest.approx(10 ** (-18 / 20)),)

    controller.settings.sync()
    stored = json.loads(controller.settings.value("audio/volumes"))
    assert stored == {
        SPEAKER.id().decode(): [80, -6],
        HEADPHONES.id().decode(): [40, 0],
        EARBUDS.id().decode(): [10, 0],
    }

    again = make_controller(data_directory, start=False)
    again._audio_player.output_changed.emit(HEADPHONES.id(), "WH-1000XM4")
    assert again.volume_percent == 40 and again.volume_ceiling_db == 0


# --- The controller with a sound system that takes the volume ---------------------------


class _FakeSystemVolume(QObject):
    """Stands in for `audio.system_volume.SystemVolume`: pactl is there where the
    controller wants it (`enabled`), the raw values set are recorded, and the test sends
    what the system reports."""

    changed = pyqtSignal(str, int)
    applied = pyqtSignal(str)
    failed = pyqtSignal(str, str)
    software_volume = pyqtSignal(str)
    instances: ClassVar[list[_FakeSystemVolume]] = []

    def __init__(self, parent: QObject | None = None, enabled: bool = True) -> None:
        super().__init__(parent)
        self.available = enabled
        self.following: str | None = None
        self.sets: list[tuple[str, int]] = []
        _FakeSystemVolume.instances.append(self)

    def follow(self, sink: str | None) -> None:
        self.following = sink

    def set_volume(self, sink: str, raw: int) -> None:
        self.sets.append((sink, raw))

    def shutdown(self) -> None:
        pass


@pytest.fixture
def system_volume(monkeypatch) -> list[_FakeSystemVolume]:
    _FakeSystemVolume.instances.clear()
    monkeypatch.setattr(receiver_controller, "SystemVolume", _FakeSystemVolume)
    return _FakeSystemVolume.instances


def test_without_the_option_the_system_volume_is_left_alone(
    media_devices, system_volume, data_directory, make_controller
):
    """Next to other programs (the touch surface in a window on the desktop of a Pi) a
    loudspeaker pinned at 100 % would play their sounds at full volume: without
    `system_volume` the radio sets no sink and attenuates itself, on Bluetooth too."""
    controller = make_controller(data_directory, start=False)
    (system,) = system_volume
    gains = _Recorder(controller._volume_changed)
    report = controller._audio_player.output_changed

    report.emit(SPEAKER.id(), "Built-in Audio Stereo")
    report.emit(HEADPHONES.id(), "WH-1000XM4")
    assert system.sets == [] and system.following is None
    assert controller.volume_device_percent is None  # decibels, the player's scale
    assert gains.last == (pytest.approx(10 ** (-54 / 20)),)  # the first play's 10 %


def test_a_loudspeaker_stands_at_100_percent_and_the_player_attenuates(
    media_devices, system_volume, data_directory, make_controller
):
    """PipeWire applies the loudspeaker's volume in software, and after a reboot the
    value it shows need not be the one its node plays at (route -52.8 dB, node 1.0 on
    2026-10-11): the radio never leaves the level to it."""
    controller = make_controller(data_directory, start=False, system_volume=True)
    (system,) = system_volume
    gains = _Recorder(controller._volume_changed)
    report = controller._audio_player.output_changed
    speaker = SPEAKER.id().decode()

    report.emit(SPEAKER.id(), "Built-in Audio Stereo")
    assert system.following is None
    assert system.sets == [(speaker, VOLUME_NORM)]
    assert controller.volume_device_percent is None  # decibels
    assert gains.last == (pytest.approx(10 ** (-54 / 20)),)  # the first play's 10 %
    system.applied.emit(speaker)  # the 100 % arrived: the player still attenuates
    assert gains.last == (pytest.approx(10 ** (-54 / 20)),)

    controller.set_volume_percent(30)
    controller.set_volume_ceiling_db(-6)
    assert system.sets == [(speaker, VOLUME_NORM)]  # the slider stays with the player
    assert gains.last == (pytest.approx(10 ** (-48 / 20)),)
    system.changed.emit(speaker, raw_volume(-3.0))  # not followed: nothing moves
    assert controller.volume_percent == 30


def test_a_bluetooth_device_takes_the_level_itself(
    media_devices, system_volume, data_directory, make_controller
):
    controller = make_controller(data_directory, start=False, system_volume=True)
    (system,) = system_volume
    gains = _Recorder(controller._volume_changed)
    report = controller._audio_player.output_changed
    speaker = SPEAKER.id().decode()
    report.emit(SPEAKER.id(), "Built-in Audio Stereo")
    controller.set_volume_percent(30)

    # Not played to before: 10 % of its own range, the player attenuating until the
    # sink has it (a level from an output before could be far too loud).
    headphones = HEADPHONES.id().decode()
    report.emit(HEADPHONES.id(), "WH-1000XM4")
    assert system.following == headphones
    assert controller.volume_percent == 10 and controller.volume_ceiling_db == 0
    assert controller.volume_device_percent == 10
    assert system.sets[-1] == (headphones, raw_of_share(0.10))
    assert gains.last == (pytest.approx(10 ** (controller.volume_db / 20)),)
    system.applied.emit(speaker)  # the loudspeaker's 100 %, late: not this output's
    assert gains.last != (1.0,)
    system.applied.emit(headphones)
    assert gains.last == (1.0,)

    controller.set_volume_percent(50)
    assert system.sets[-1] == (headphones, raw_of_share(0.5))
    assert controller.volume_device_percent == 50
    assert controller.volume_db == pytest.approx(-18.06, abs=0.01)  # nominal, cubic
    controller.set_volume_ceiling_db(-6)  # 79 % of the range at 100 %
    assert system.sets[-1] == (headphones, raw_of_share(0.5 * 10 ** (-6 / 60)))
    assert controller.volume_device_percent == 40

    # Back at the loudspeaker: pinned again, the player attenuates (30 %, -42 dB).
    report.emit(SPEAKER.id(), "Built-in Audio Stereo")
    assert system.following is None
    assert system.sets[-1] == (speaker, VOLUME_NORM)
    assert gains.last == (pytest.approx(10 ** (-42 / 20)),)

    # A second Bluetooth device starts at 10 % as well, however loud the first plays.
    report.emit(HEADPHONES.id(), "WH-1000XM4")
    controller.set_volume_ceiling_db(0)
    controller.set_volume_percent(80)
    report.emit(EARBUDS.id(), "Galaxy Buds")
    assert controller.volume_percent == 10
    assert system.sets[-1] == (EARBUDS.id().decode(), raw_of_share(0.10))

    report.emit(None, "")
    assert system.following is None


def test_a_bluetooth_device_without_a_volume_of_its_own_is_pinned_too(
    media_devices, system_volume, data_directory, make_controller
):
    controller = make_controller(data_directory, start=False, system_volume=True)
    (system,) = system_volume
    gains = _Recorder(controller._volume_changed)
    report = controller._audio_player.output_changed
    headphones = HEADPHONES.id().decode()
    report.emit(HEADPHONES.id(), "WH-1000XM4")
    assert system.following == headphones

    system.software_volume.emit("bluez_output.00_11_22_33_44_55.1")  # not the one in use
    assert system.following == headphones
    system.software_volume.emit(headphones)
    assert system.following is None
    assert system.sets[-1] == (headphones, VOLUME_NORM)
    assert controller.volume_device_percent is None  # decibels, the player's scale
    assert gains.last == (pytest.approx(10 ** (-54 / 20)),)
    system.applied.emit(headphones)
    assert gains.last == (pytest.approx(10 ** (-54 / 20)),)


def test_a_volume_changed_on_the_device_moves_the_slider_and_stays_under_the_ceiling(
    media_devices, system_volume, data_directory, make_controller
):
    controller = make_controller(data_directory, start=False, system_volume=True)
    (system,) = system_volume
    changes = _Recorder(controller.volume_changed)
    report = controller._audio_player.output_changed
    headphones = HEADPHONES.id().decode()
    report.emit(HEADPHONES.id(), "WH-1000XM4")
    controller.set_volume_percent(50)
    told, sets = len(changes.calls), len(system.sets)

    # The same level: nothing. Another: the slider follows over the device's range,
    # the sink is not set again (it has that level), the output remembers it.
    system.changed.emit(headphones, raw_of_share(0.5))
    assert controller.volume_percent == 50 and len(changes.calls) == told
    system.changed.emit(headphones, round(65536 * 76 / 127))  # step 76 of 127
    assert controller.volume_percent == 60 and len(changes.calls) == told + 1
    assert len(system.sets) == sets
    system.changed.emit(headphones, round(65536 * 2 / 127))
    assert controller.volume_percent == 2
    system.changed.emit(headphones, 0)
    assert controller.volume_percent == 0 and controller.volume_db is None
    system.changed.emit("bluez_output.00_11_22_33_44_55.1", 65536)  # not in use
    assert controller.volume_percent == 0

    # Above the "Maximum volume": the slider at 100 %, the device back to the ceiling.
    controller.set_volume_ceiling_db(-6)
    system.changed.emit(headphones, 65536)
    assert controller.volume_percent == 100
    assert system.sets[-1] == (headphones, raw_of_share(10 ** (-6 / 60)))

    controller.settings.sync()
    stored = json.loads(controller.settings.value("audio/volumes"))
    assert stored[headphones] == [100, -6]


def test_a_sink_that_takes_no_volume_leaves_it_to_the_player_until_it_does(
    media_devices, system_volume, data_directory, make_controller
):
    controller = make_controller(data_directory, start=False, system_volume=True)
    (system,) = system_volume
    gains = _Recorder(controller._volume_changed)
    report = controller._audio_player.output_changed
    headphones = HEADPHONES.id().decode()
    report.emit(HEADPHONES.id(), "WH-1000XM4")

    # The player attenuates meanwhile, at the level wanted; the sink stays followed.
    system.failed.emit(headphones, "Failure: No such entity")
    assert system.following == headphones
    assert gains.last == (pytest.approx(10 ** (controller.volume_db / 20)),)
    controller.set_volume_percent(80)
    assert system.sets[-1] == (headphones, raw_of_share(0.8))  # tried again
    system.failed.emit(headphones, "Failure: No such entity")
    assert gains.last == (pytest.approx(10 ** (controller.volume_db / 20)),)

    # Once a set gets through, the device has the level and the player plays at 1.0.
    controller.set_volume_percent(70)
    system.applied.emit(headphones)
    assert gains.last == (1.0,)
    system.failed.emit("bluez_output.00_11_22_33_44_55.1", "Failure")  # not ours
    assert gains.last == (1.0,)
