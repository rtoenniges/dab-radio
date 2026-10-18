"""Moving the radio's A2DP stream between Bluetooth devices (`audio.bluetooth_route`)
and which device the controller gives it to (`ReceiverController._route_bluetooth`), on
a BlueZ that carries one stream at a time as the Pi does (`tests.fake_bluez`) and a
pactl that lists the sinks of PipeWire from a table."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from typing import ClassVar

import pytest
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from audio import bluetooth_route
from audio.bluetooth_route import (
    BluetoothRoute,
    a2dp_sink,
    sink_address,
    sink_name,
)
from net.bluetooth import A2DP_SINK_UUID, DEVICE, BluetoothManager
from net.command import CommandResult
from net.dbus import Reply
from tests.fake_bluez import BOSE, JBL, SONY, FakeBus
from ui import receiver_controller

SONY_SINK = "bluez_output.F8_4E_17_01_02_03.1"
JBL_SINK = "bluez_output.B8_F6_53_04_05_06.1"
BOSE_SINK = "bluez_output.2C_41_A1_07_08_09.1"
SPEAKER_SINK = "alsa_output.platform-soc_sound.stereo-fallback"
HANDS_FREE = "headset-head-unit"


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


class _Recorder:
    def __init__(self, signal) -> None:
        self.calls: list[tuple] = []
        signal.connect(lambda *args: self.calls.append(args))


class FakePactl:
    """Answers `pactl -f json list sinks` from the bus: a device connected has a sink,
    on A2DP while it holds the radio's stream, else on the hands-free profile; with
    `slow` the A2DP profile shows only from the next listing on, as WirePlumber
    switches the card a moment after BlueZ."""

    def __init__(self, bus: FakeBus) -> None:
        self.bus = bus
        self.commands: list[list[str]] = []
        self.slow = False
        self.never = False  # WirePlumber keeps the hands-free profile
        self._seen: set[str] = set()

    def run(self, arguments: list[str], timeout_ms: int, done) -> None:
        self.commands.append(list(arguments))
        assert arguments == ["-f", "json", "list", "sinks"]
        sinks = []
        for path, interfaces in self.bus.objects.items():
            device = interfaces.get(DEVICE)
            if device is None or not device.get("Connected"):
                continue
            a2dp = self.bus.stream_of(path) is not None and not self.never
            if a2dp and self.slow and path not in self._seen:
                self._seen.add(path)
                a2dp = False
            sinks.append(
                {
                    "name": sink_name(device["Address"]),
                    "properties": {
                        "api.bluez5.path": path,
                        "api.bluez5.profile": "a2dp-sink" if a2dp else HANDS_FREE,
                    },
                }
            )
        done(CommandResult(0, json.dumps(sinks)))

    def stop(self) -> None:
        pass


@pytest.fixture
def bus() -> FakeBus:
    bus = FakeBus()
    bus.one_stream = True
    return bus


@pytest.fixture
def bluetooth(bus) -> BluetoothManager:
    manager = BluetoothManager(bus=bus)
    manager.start()
    return manager


def _connect_hands_free(bus: FakeBus, path: str) -> None:
    """A device connecting by itself while another holds the stream: hands-free only."""
    bus._set(path, DEVICE, "Paired", True)
    bus._set(path, DEVICE, "Connected", True)


# --- Names of PipeWire --------------------------------------------------------------------


def test_the_sink_of_a_device_is_named_after_its_address():
    assert sink_name("f8:4e:17:01:02:03") == SONY_SINK
    assert sink_address(SONY_SINK) == "F8:4E:17:01:02:03"
    assert sink_address("bluez_output.f8_4e_17_01_02_03.1") == "F8:4E:17:01:02:03"
    assert sink_address(SPEAKER_SINK) is None
    assert sink_address("") is None


def test_the_a2dp_sink_of_a_device_is_found_in_the_listing():
    listing = json.dumps(
        [
            {"name": SPEAKER_SINK, "properties": {"device.api": "alsa"}},
            {
                "name": SONY_SINK,
                "properties": {"api.bluez5.path": SONY, "api.bluez5.profile": HANDS_FREE},
            },
            {
                "name": JBL_SINK,
                "properties": {"api.bluez5.path": JBL, "api.bluez5.profile": "a2dp-sink"},
            },
        ]
    )
    assert a2dp_sink(listing, JBL) == JBL_SINK
    assert a2dp_sink(listing, SONY) is None  # hands-free only
    assert a2dp_sink("Failure: Connection refused", JBL) is None
    assert a2dp_sink('{"name": "x"}', JBL) is None


# --- The move -------------------------------------------------------------------------------


def _route(bluetooth: BluetoothManager, bus: FakeBus) -> tuple[BluetoothRoute, FakePactl]:
    pactl = FakePactl(bus)
    return BluetoothRoute(bluetooth, runner=pactl), pactl


def test_the_stream_moves_to_a_device_on_the_hands_free_profile(bluetooth, bus):
    """Headphones on the hands-free profile next to a device holding the stream: after
    the player's lead the stream to that device ends, the one to the headphones is set
    up, and the move is over once PipeWire lists their sink with A2DP."""
    _connect_hands_free(bus, BOSE)
    route, pactl = _route(bluetooth, bus)
    finished = _Recorder(route.finished)
    pactl.slow = True

    assert route.move_to(bluetooth.device(BOSE))
    assert route.busy and route.target == BOSE
    assert bluetooth.activity == "Moving the audio to “LE-Bose QC45” …"
    assert not route.move_to(bluetooth.device(JBL))  # one at a time
    assert bus.calls_of("DisconnectProfile") == []  # the player is given its lead

    route._hold_timer.timeout.emit()
    assert bus.calls_of("DisconnectProfile") == [
        (SONY, DEVICE, "DisconnectProfile", [A2DP_SINK_UUID])
    ]
    assert bus.calls_of("ConnectProfile") == [(BOSE, DEVICE, "ConnectProfile", [A2DP_SINK_UUID])]
    assert bluetooth.a2dp_device.path == BOSE
    assert bluetooth.device(SONY).connected  # on the hands-free profile now
    assert finished.calls == [] and route._poll_timer.isActive()  # the card not yet

    route._poll_timer.timeout.emit()
    assert finished.calls == [(BOSE, True, "")]
    assert not route.busy and bluetooth.activity == ""


def test_a_device_not_connected_is_connected_once_the_stream_is_free(bluetooth, bus):
    """The loudspeaker cannot connect while another device holds the stream: it lacks
    the hands-free profile towards the Pi and drops."""
    route, _pactl = _route(bluetooth, bus)
    finished = _Recorder(route.finished)
    route.move_to(bluetooth.device(JBL))
    assert bluetooth.activity == ""  # the manager tells "Connecting …" itself

    route._hold_timer.timeout.emit()
    assert bus.calls_of("DisconnectProfile")[0][0] == SONY
    assert bus.calls_of("Connect") == [(JBL, DEVICE, "Connect", [])]
    assert bluetooth.device(JBL).connected and bluetooth.a2dp_device.path == JBL
    assert finished.calls == [(JBL, True, "")] and bluetooth.activity == ""


def test_a_stream_that_does_not_end_is_waited_for_a_while(bluetooth, bus):
    _connect_hands_free(bus, BOSE)
    bus.answers[(DEVICE, "DisconnectProfile")] = Reply()  # answered, but nothing ends
    route, _pactl = _route(bluetooth, bus)
    finished = _Recorder(route.finished)
    failures = _Recorder(bluetooth.failed)
    route.move_to(bluetooth.device(BOSE))
    route._hold_timer.timeout.emit()
    assert bus.calls_of("ConnectProfile") == [] and route._release_timer.isActive()

    route._release_timer.timeout.emit()
    assert bus.calls_of("ConnectProfile")[0][0] == BOSE  # refused: still held
    ((device, ok, why),) = finished.calls
    assert (device, ok) == (BOSE, False) and why.startswith("No audio connection free")
    assert failures.calls == []  # reporting it is the controller's, once it gives up
    assert bluetooth.activity == ""


def test_a_card_that_stays_hands_free_ends_the_wait(bluetooth, bus, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(bluetooth_route, "_clock", lambda: now[0])
    _connect_hands_free(bus, BOSE)
    route, pactl = _route(bluetooth, bus)
    pactl.never = True
    finished = _Recorder(route.finished)
    route.move_to(bluetooth.device(BOSE))
    route._hold_timer.timeout.emit()
    route._poll_timer.timeout.emit()
    assert finished.calls == []

    now[0] += bluetooth_route.SINK_WAIT_S
    route._poll_timer.timeout.emit()
    assert finished.calls == [(BOSE, False, "")] and not route.busy


def test_a_device_gone_before_the_move_begins_ends_it(bluetooth, bus):
    _connect_hands_free(bus, BOSE)
    route, _pactl = _route(bluetooth, bus)
    finished = _Recorder(route.finished)
    route.move_to(bluetooth.device(BOSE))
    bus.objects.pop(BOSE)
    bus.interfaces_removed.emit(BOSE, [DEVICE])
    route._hold_timer.timeout.emit()
    assert finished.calls == [(BOSE, False, "")]
    assert bus.calls_of("DisconnectProfile") == []


def test_without_pactl_the_stream_set_up_is_enough(bluetooth, bus, monkeypatch):
    monkeypatch.setattr(bluetooth_route, "find_pactl", lambda: None)
    _connect_hands_free(bus, BOSE)
    route = BluetoothRoute(bluetooth)
    finished = _Recorder(route.finished)
    route.move_to(bluetooth.device(BOSE))
    route._hold_timer.timeout.emit()
    assert finished.calls == [(BOSE, True, "")]


def test_a_move_asked_for_as_the_last_ends_keeps_the_activity(bluetooth, bus):
    """A listener of `finished` asking for the next move at once: the end of the move
    before must leave the new one's state alone, or its text stays in the activity and
    holds the Bluetooth page."""
    _connect_hands_free(bus, BOSE)
    route, _pactl = _route(bluetooth, bus)
    refuse = [True]
    bus.answers[(DEVICE, "ConnectProfile")] = lambda _path, _arguments: (
        Reply(None, "org.bluez.Error.Failed", "br-connection-refused") if refuse else None
    )
    route.finished.connect(
        lambda _path, ok, _why: None if ok else route.move_to(bluetooth.device(BOSE))
    )
    route.move_to(bluetooth.device(BOSE))
    route._hold_timer.timeout.emit()
    assert route.busy and bluetooth.activity == "Moving the audio to “LE-Bose QC45” …"

    refuse.clear()
    route._hold_timer.timeout.emit()
    assert not route.busy and bluetooth.a2dp_device.path == BOSE
    assert bluetooth.activity == ""


# --- The controller's choice -----------------------------------------------------------------


class _FakeSystemVolume(QObject):
    """pactl is there where the controller wants it (`enabled`); nothing is set."""

    changed = pyqtSignal(str, int)
    applied = pyqtSignal(str)
    failed = pyqtSignal(str, str)
    software_volume = pyqtSignal(str)
    instances: ClassVar[list[_FakeSystemVolume]] = []

    def __init__(self, parent: QObject | None = None, enabled: bool = True) -> None:
        super().__init__(parent)
        self.available = enabled

    def follow(self, sink: str | None) -> None:
        pass

    def set_volume(self, sink: str, raw: int) -> None:
        pass

    def shutdown(self) -> None:
        pass


@pytest.fixture
def system_volume(monkeypatch):
    monkeypatch.setattr(receiver_controller, "SystemVolume", _FakeSystemVolume)


@pytest.fixture
def radio(system_volume, data_directory, make_controller, bluetooth, bus):
    """A controller owning the sound system (`system_volume`) with the Bluetooth of
    the fake attached; its route answers at once (`settle`)."""
    controller = make_controller(data_directory, start=False, system_volume=True)
    route, _pactl = _route(bluetooth, bus)
    controller.attach_bluetooth(bluetooth, route)
    return controller


class _Player:
    """What the controller tells the player: the outputs and the hold, in order."""

    def __init__(self, controller) -> None:
        self.events: list[tuple[str, object]] = []
        controller._audio_output_set.connect(lambda out: self.events.append(("output", out)))
        controller._audio_held.connect(lambda held: self.events.append(("held", held)))

    @property
    def output(self) -> object:
        return next(value for kind, value in reversed(self.events) if kind == "output")


def _settle(controller) -> None:
    """Lets the moves of the controller's route run through, the pauses before a
    next try included."""
    route = controller._bluetooth_route
    for _ in range(20):
        if route._hold_timer.isActive():
            route._hold_timer.timeout.emit()
        elif route._poll_timer.isActive():
            route._poll_timer.timeout.emit()
        elif controller._bluetooth_retry.isActive():
            controller._bluetooth_retry.stop()
            controller._bluetooth_retry.timeout.emit()
        elif not route.busy:
            return


def test_automatic_plays_to_the_device_holding_the_stream(
    system_volume, data_directory, make_controller, bluetooth, bus
):
    controller = make_controller(data_directory, start=False, system_volume=True)
    player = _Player(controller)
    route, _pactl = _route(bluetooth, bus)
    controller.attach_bluetooth(bluetooth, route)
    assert player.output == SONY_SINK.encode()
    assert not route.busy


def test_automatic_gives_the_stream_to_the_device_connected_last(radio, bluetooth, bus):
    """With "Automatic" the headphones switched on last play, and with A2DP, not over
    the hands-free profile."""
    player = _Player(radio)
    _connect_hands_free(bus, BOSE)
    assert radio._bluetooth_route.target == BOSE
    assert player.events[0] == ("held", True)
    _settle(radio)

    assert bluetooth.a2dp_device.path == BOSE
    assert player.output == BOSE_SINK.encode()
    # The player is told where to play before it may play again.
    assert player.events[-2:] == [("output", BOSE_SINK.encode()), ("held", False)]
    # The headphones before stay connected on the hands-free profile, and stay silent.
    assert bluetooth.device(SONY).connected
    assert not radio._bluetooth_route.busy


def test_a_device_chosen_keeps_the_stream_until_another_is_chosen(radio, bluetooth, bus):
    player = _Player(radio)
    radio.set_audio_output(SONY_SINK)
    _connect_hands_free(bus, BOSE)
    assert not radio._bluetooth_route.busy
    assert bus.calls_of("DisconnectProfile") == []
    assert player.output == SONY_SINK.encode()

    radio.set_audio_output(BOSE_SINK)  # chosen under "Output": it gets A2DP
    _settle(radio)
    assert bluetooth.a2dp_device.path == BOSE
    assert player.output == BOSE_SINK.encode()
    assert player.events[-1] == ("held", False)

    radio.set_audio_output(SONY_SINK)  # and back
    _settle(radio)
    assert bluetooth.a2dp_device.path == SONY


def test_the_device_left_gets_the_stream_once_it_is_free(radio, bluetooth, bus):
    """The device holding the stream goes: under "Automatic" the one still connected
    over the hands-free profile gets it, without anyone to release."""
    player = _Player(radio)
    radio.set_audio_output(SONY_SINK)
    _connect_hands_free(bus, BOSE)
    radio.set_audio_output("")
    _settle(radio)  # newest: the Bose got it
    assert bluetooth.a2dp_device.path == BOSE
    bluetooth.disconnect_device(bluetooth.device(BOSE))
    assert bluetooth.a2dp_device is None
    _settle(radio)
    assert bluetooth.device(BOSE) is not None and not bluetooth.device(BOSE).connected
    assert bluetooth.a2dp_device.path == SONY
    assert len(bus.calls_of("DisconnectProfile")) == 1  # only the first move released
    assert player.output == SONY_SINK.encode()


def test_headphones_switched_off_while_playing_are_not_set_up_again(radio, bluetooth, bus):
    """BlueZ ends the stream of headphones going off before it reports them gone: set up
    again, the stream to them would fail and, with the tries of their connecting
    counted, show an error on the page. The ones connected before get the stream at
    once."""
    failures = _Recorder(bluetooth.failed)
    _connect_hands_free(bus, BOSE)
    _settle(radio)
    assert bluetooth.a2dp_device.path == BOSE
    assert BOSE not in radio._bluetooth_moves  # through: the tries start over

    bus.end_stream(BOSE)  # switched off: the stream goes first
    assert radio._bluetooth_route.target == SONY  # not the Bose again
    bus._set(BOSE, DEVICE, "Connected", False)
    _settle(radio)
    tried = [call[0] for call in bus.calls_of("ConnectProfile")]
    assert tried == [BOSE, SONY] and bluetooth.a2dp_device.path == SONY
    assert failures.calls == []


def test_a_stream_a_phone_took_is_left_to_it_until_chosen(radio, bluetooth, bus):
    """A multipoint box ends the radio's stream when a phone plays to it: the radio does
    not take it back by itself, only when the box is chosen."""
    bus.end_stream(SONY)
    assert not radio._bluetooth_route.busy
    assert bus.calls_of("ConnectProfile") == []

    radio.set_audio_output(SONY_SINK)
    _settle(radio)
    assert bluetooth.a2dp_device.path == SONY


def test_with_the_loudspeaker_chosen_a_device_falling_back_to_gets_the_stream(
    radio, bluetooth, bus
):
    """A device the player plays to (the default, the output chosen being gone) on the
    hands-free profile gets A2DP while no other device holds it."""
    radio.set_audio_output(SPEAKER_SINK)
    bluetooth.disconnect_device(bluetooth.device(SONY))
    _connect_hands_free(bus, SONY)
    assert not radio._bluetooth_route.busy  # the loudspeaker plays
    radio._audio_player.output_changed.emit(SONY_SINK.encode(), "WH-1000XM4")
    assert radio._bluetooth_route.target == SONY
    _settle(radio)
    assert bluetooth.a2dp_device.path == SONY


def test_a_device_refusing_at_first_gets_the_stream_at_the_next_try(radio, bluetooth, bus):
    """Headphones may refuse the stream just after connecting by themselves and take it
    at the next try. The player stays silent from the first try to the last, and the
    page is told of nothing."""
    player = _Player(radio)
    failures = _Recorder(bluetooth.failed)
    refusals = [1]

    def connect_profile(_path, _arguments):
        if refusals:
            refusals.pop()
            return Reply(None, "org.bluez.Error.Failed", "br-connection-refused")
        return None

    bus.answers[(DEVICE, "ConnectProfile")] = connect_profile
    _connect_hands_free(bus, BOSE)
    radio._bluetooth_route._hold_timer.timeout.emit()  # the first try, refused
    assert radio._bluetooth_retry.isActive() and not radio._bluetooth_route.busy
    assert player.events == [("held", True)]  # silent through the pause
    assert bluetooth.activity == ""  # the page takes taps meanwhile

    _settle(radio)
    assert bluetooth.a2dp_device.path == BOSE
    assert player.events == [("held", True), ("output", BOSE_SINK.encode()), ("held", False)]
    assert failures.calls == [] and bluetooth.activity == ""
    assert BOSE not in radio._bluetooth_moves


def test_a_device_refusing_the_stream_is_given_up_after_its_tries(radio, bluetooth, bus):
    """The headphones connected last would not take the stream: after their tries the
    ones before, whose stream was freed for them, get it back, and the page is told
    why."""
    refuse = {BOSE}
    bus.answers[(DEVICE, "ConnectProfile")] = lambda path, _arguments: (
        Reply(None, "org.bluez.Error.Failed", "br-connection-refused") if path in refuse else None
    )
    failures = _Recorder(bluetooth.failed)
    _connect_hands_free(bus, BOSE)
    _settle(radio)
    tried = [call[0] for call in bus.calls_of("ConnectProfile")]
    assert tried == [BOSE] * receiver_controller.BLUETOOTH_MOVES + [SONY]
    assert bluetooth.a2dp_device.path == SONY and not radio._bluetooth_route.busy
    assert failures.calls == [("The device refused the connection.",)]

    bus._set(BOSE, DEVICE, "Connected", False)
    refuse.clear()
    _connect_hands_free(bus, BOSE)  # connecting anew: tried again
    assert radio._bluetooth_route.target == BOSE


def test_connecting_by_hand_frees_the_stream_and_takes_the_choice(radio, bluetooth, bus):
    """ "Connect" on the Bluetooth page: the speaker drops without A2DP, so the stream
    is freed for it first; with a Bluetooth device chosen, the choice moves to it."""
    radio.set_audio_output(SONY_SINK)
    radio.connect_bluetooth(bluetooth.device(JBL))
    assert radio.audio_output_choice == JBL_SINK
    _settle(radio)
    assert bus.calls_of("Connect") == [(JBL, DEVICE, "Connect", [])]
    assert bluetooth.a2dp_device.path == JBL


def test_connecting_a_phone_by_hand_is_left_to_bluez(radio, bluetooth, bus):
    from tests.fake_bluez import PHONE

    radio.connect_bluetooth(bluetooth.device(PHONE))
    assert bus.calls_of("Connect") == [(PHONE, DEVICE, "Connect", [])]
    assert bus.calls_of("DisconnectProfile") == []


def test_without_the_sound_system_bluez_decides(
    system_volume, data_directory, make_controller, bluetooth, bus
):
    """Next to other programs (no `system_volume`) the radio moves no stream."""
    controller = make_controller(data_directory, start=False)
    controller.attach_bluetooth(bluetooth)
    assert controller._bluetooth_route is None
    _connect_hands_free(bus, BOSE)
    controller.connect_bluetooth(bluetooth.device(JBL))
    assert bus.calls_of("DisconnectProfile") == []
    assert bus.calls_of("Connect") == [(JBL, DEVICE, "Connect", [])]
