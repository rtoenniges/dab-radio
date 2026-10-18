"""The volume of the sink in the sound system (`audio.system_volume`): decibels and
the share of a device's range as pactl's raw value and back, the readback parsed, a sink
followed only once it is known to apply its volume in the device (waited for, as a
Bluetooth device brings its volume control up late), one command for the last of
several values, the device never left louder than set, the radio's own value not taken
for a change (also as a Bluetooth device rounds it, or answers late), changes made
elsewhere reported, and nothing without pactl."""

from __future__ import annotations

import json
import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from collections.abc import Callable

import pytest
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from audio import system_volume
from audio.system_volume import (
    HARDWARE_WAIT_S,
    SETTLE_S,
    VOLUME_NORM,
    SystemVolume,
    db_of_raw,
    parse_volume,
    raw_of_share,
    raw_volume,
    share_of_raw,
    sink_flags,
)
from net.command import CommandResult

# The timers of SystemVolume want an event dispatcher in the thread.
_app = QApplication.instance() or QApplication([])

# The loudspeaker of the Pi: PipeWire applies its volume in software. A USB sound card
# with a mixer of its own and Bluetooth headphones with AVRCP apply it in the device.
SPEAKER = "alsa_output.platform-soc_107c000000_sound.stereo-fallback"
USB_CARD = "alsa_output.usb-Generic_USB_Audio-00.analog-stereo"
HEADPHONES = "bluez_output.50_C2_75_65_E7_BE.1"
LIST_SINKS = ["-f", "json", "list", "sinks"]
FLAGS = {
    SPEAKER: ["HARDWARE", "DECIBEL_VOLUME", "LATENCY"],
    USB_CARD: ["HARDWARE", "HW_VOLUME_CTRL", "DECIBEL_VOLUME", "LATENCY"],
    HEADPHONES: ["HARDWARE", "HW_VOLUME_CTRL", "DECIBEL_VOLUME", "LATENCY"],
}


def avrcp(raw: int) -> int:
    """What a Bluetooth device makes of a raw value: one of 127 steps (AVRCP absolute
    volume), which lie evenly on pactl's cubic scale."""
    return round(VOLUME_NORM * round(127 * raw / VOLUME_NORM) / 127)


def volume_output(raw: int) -> str:
    """The output of `pactl get-sink-volume` for a stereo sink at this raw value."""
    percent = round(100 * raw / VOLUME_NORM)
    db = "-inf" if raw == 0 else f"{60 * math.log10(raw / VOLUME_NORM):.2f}"
    channel = f"{raw} / {percent:3d}% / {db} dB"
    return f"Volume: front-left: {channel},   front-right: {channel}\n        balance 0.00\n"


def sinks_output(names: list[str], flags: dict[str, list[str]] = FLAGS) -> str:
    """The output of `pactl -f json list sinks`, with only what is read of it."""
    return json.dumps(
        [{"index": 50 + i, "name": n, "flags": flags[n]} for i, n in enumerate(names)]
    )


class FakePactl:
    """Answers pactl at once: set-sink-volume stores the raw value (rounded as a
    Bluetooth device does for a bluez sink), get-sink-volume prints it as pactl does, and
    list sinks gives the flags. `holding` keeps the answers back until `release`, as a
    pactl still running."""

    def __init__(self) -> None:
        self.volumes: dict[str, int] = dict.fromkeys(FLAGS, VOLUME_NORM)
        self.listed = list(FLAGS)
        self.flags = {name: list(flags) for name, flags in FLAGS.items()}
        self.commands: list[list[str]] = []
        self.held: list[tuple[list[str], Callable[[CommandResult], None]]] = []
        self.holding = False
        self.refuse_setting: str | None = None  # the error pactl prints then
        self.stopped = False

    def run(self, arguments: list[str], timeout_ms: int, done) -> None:
        self.commands.append(list(arguments))
        if self.holding:
            self.held.append((list(arguments), done))
            return
        done(self.answer(arguments))

    def release(self) -> None:
        self.holding = False
        while self.held:
            arguments, done = self.held.pop(0)
            done(self.answer(arguments))

    def answer(self, arguments: list[str]) -> CommandResult:
        if arguments == LIST_SINKS:
            return CommandResult(0, sinks_output(self.listed, self.flags))
        command, sink = arguments[0], arguments[1]
        if command == "set-sink-volume":
            if self.refuse_setting is not None:
                return CommandResult(1, "", self.refuse_setting)
            raw = int(arguments[2])
            self.volumes[sink] = avrcp(raw) if sink.startswith("bluez") else raw
            return CommandResult(0, "")
        if command == "get-sink-volume":
            if sink not in self.volumes:
                return CommandResult(1, "", "Failure: No such entity\n")
            return CommandResult(0, volume_output(self.volumes[sink]))
        raise AssertionError(f"unexpected pactl command {arguments}")

    def stop(self) -> None:
        self.stopped = True

    def commands_starting(self, word: str) -> list[list[str]]:
        return [c for c in self.commands if c[0] == word]


class FakeMonitor(QObject):
    """Stands in for `pactl subscribe`: the test sends its lines."""

    line = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.starts = 0
        self.running = False

    def start(self) -> None:
        if not self.running:
            self.starts += 1
            self.running = True

    def stop(self) -> None:
        self.running = False

    def sink_changed(self, index: int = 55) -> None:
        self.line.emit(f"Event 'change' on sink #{index}")


class _Recorder:
    def __init__(self, signal) -> None:
        self.calls: list[tuple] = []
        signal.connect(lambda *args: self.calls.append(args))

    @property
    def last(self) -> tuple:
        return self.calls[-1]


@pytest.fixture
def pactl() -> FakePactl:
    return FakePactl()


@pytest.fixture
def monitor() -> FakeMonitor:
    return FakeMonitor()


@pytest.fixture
def clock(monkeypatch) -> list[float]:
    """The clock of the module, which the test moves."""
    now = [1000.0]
    monkeypatch.setattr(system_volume, "_clock", lambda: now[0])
    return now


@pytest.fixture
def volume(pactl, monitor, clock) -> SystemVolume:
    instance = SystemVolume(runner=pactl, monitor=monitor)
    yield instance
    instance.shutdown()


# --- The scale ------------------------------------------------------------------------


def test_decibels_and_the_share_of_a_range_become_pactls_raw_value_and_back():
    assert raw_volume(0.0) == VOLUME_NORM
    assert raw_volume(None) == 0
    assert raw_volume(60 * math.log10(0.4)) == 26214  # WirePlumber's 0.064: 0.4 of the scale
    assert raw_volume(-60.0) == pytest.approx(VOLUME_NORM / 10, abs=1)
    assert raw_volume(-30.0) == pytest.approx(VOLUME_NORM / math.sqrt(10), abs=1)
    assert db_of_raw(VOLUME_NORM) == pytest.approx(0.0)
    assert db_of_raw(0) is None
    assert db_of_raw(26214) == pytest.approx(-23.88, abs=0.01)
    for db in range(-60, 1):
        assert db_of_raw(raw_volume(float(db))) == pytest.approx(db, abs=0.01)

    # A Bluetooth device's 127 steps lie evenly on the raw scale: the share is linear.
    assert raw_of_share(0.0) == 0
    assert raw_of_share(1.0) == VOLUME_NORM
    assert raw_of_share(0.5) == VOLUME_NORM // 2
    assert raw_of_share(76 / 127) == 39218
    assert share_of_raw(39218) == pytest.approx(76 / 127, abs=1e-4)
    assert share_of_raw(0) == 0.0 and share_of_raw(-5) == 0.0


def test_the_readback_of_pactl_is_parsed_at_its_loudest_channel():
    assert parse_volume(volume_output(26214)) == 26214
    assert parse_volume(volume_output(0)) == 0
    assert parse_volume("") is None
    assert parse_volume("Failure: No such entity\n") is None
    unequal = (
        "Volume: front-left: 26214 /  40% / -23.88 dB,   front-right: 32768 /  50% / -18.06 dB\n"
    )
    assert parse_volume(unequal) == 32768


def test_the_flags_of_a_sink_are_read_from_the_json_of_pactl():
    output = sinks_output([SPEAKER, HEADPHONES])
    assert sink_flags(output, HEADPHONES) == FLAGS[HEADPHONES]
    assert sink_flags(output, SPEAKER) == FLAGS[SPEAKER]
    assert sink_flags(output, USB_CARD) is None  # not listed
    assert sink_flags("", SPEAKER) is None
    assert sink_flags("Connection failure", SPEAKER) is None
    assert sink_flags("{}", SPEAKER) is None


# --- Which sink takes the level --------------------------------------------------------


def test_a_sink_followed_is_asked_first_and_set_once_it_applies_the_volume_itself(volume, pactl):
    software = _Recorder(volume.software_volume)
    pactl.holding = True
    volume.follow(USB_CARD)
    volume.set_volume(USB_CARD, raw_volume(-30.0))
    assert pactl.commands == [LIST_SINKS]  # the volume waits for the answer
    pactl.release()
    assert pactl.commands[1] == ["set-sink-volume", USB_CARD, str(raw_volume(-30.0))]
    assert software.calls == []


def give_up_waiting(volume: SystemVolume, clock: list[float]) -> None:
    """Lets the time for the flag pass and the next query come."""
    clock[0] += HARDWARE_WAIT_S
    volume._on_probe_timer()


def test_a_sink_that_applies_its_volume_in_software_is_never_set_to_a_level(
    volume, pactl, monitor, clock
):
    """The loudspeaker of the Pi: after a reboot its route can show -52.8 dB while its
    node plays at 1.0, so the radio does not set its level."""
    software = _Recorder(volume.software_volume)
    applied = _Recorder(volume.applied)
    volume.follow(SPEAKER)
    volume.set_volume(SPEAKER, raw_volume(-52.8))
    assert software.calls == [] and pactl.commands_starting("set-sink-volume") == []
    monitor.sink_changed()  # asked again: still in software
    assert pactl.commands.count(LIST_SINKS) == 2 and software.calls == []
    give_up_waiting(volume, clock)
    assert software.calls == [(SPEAKER,)]
    assert pactl.commands_starting("set-sink-volume") == []
    assert applied.calls == []

    # Its changes are not followed either; a set while not followed goes through (the
    # controller pins it at 100 %).
    monitor.sink_changed()
    assert pactl.commands_starting("get-sink-volume") == []
    volume.follow(None)
    volume.set_volume(SPEAKER, VOLUME_NORM)
    assert pactl.commands_starting("set-sink-volume") == [
        ["set-sink-volume", SPEAKER, str(VOLUME_NORM)]
    ]
    assert applied.calls == [(SPEAKER,)]


def test_a_sink_pactl_does_not_list_counts_as_one_in_software(volume, pactl, clock):
    software = _Recorder(volume.software_volume)
    pactl.listed = [SPEAKER]
    volume.follow(HEADPHONES)
    volume.set_volume(HEADPHONES, raw_of_share(0.1))
    give_up_waiting(volume, clock)
    assert software.calls == [(HEADPHONES,)]
    assert pactl.commands_starting("set-sink-volume") == []


def test_a_bluetooth_device_whose_volume_control_comes_late_is_waited_for(
    volume, pactl, monitor, clock
):
    """A Bluetooth headset asked at once has no flag yet and would be set to 100 % of
    its own volume: its AVRCP comes up a second or two after its audio."""
    software = _Recorder(volume.software_volume)
    applied = _Recorder(volume.applied)
    pactl.flags[HEADPHONES] = ["HARDWARE", "DECIBEL_VOLUME", "LATENCY"]
    volume.follow(HEADPHONES)
    volume.set_volume(HEADPHONES, raw_of_share(0.1))
    clock[0] += 0.5
    volume._on_probe_timer()  # asked again, still without
    assert pactl.commands.count(LIST_SINKS) == 2
    assert pactl.commands_starting("set-sink-volume") == [] and software.calls == []

    # AVRCP is up: PipeWire changes the sink, and the radio asks at once.
    pactl.flags[HEADPHONES] = FLAGS[HEADPHONES]
    clock[0] += 1.0
    monitor.sink_changed()
    assert pactl.commands_starting("set-sink-volume") == [
        ["set-sink-volume", HEADPHONES, str(raw_of_share(0.1))]
    ]
    assert applied.calls == [(HEADPHONES,)] and software.calls == []
    give_up_waiting(volume, clock)  # too late to count any more
    assert software.calls == []


def test_a_level_held_for_a_sink_given_up_is_dropped(volume, pactl):
    """Following another sink before the answer drops a level held for the first one: it
    must not reach a sink not known to apply it in the device."""
    pactl.holding = True
    volume.follow(SPEAKER)
    volume.set_volume(SPEAKER, raw_volume(-52.8))
    volume.follow(None)
    volume.set_volume(SPEAKER, VOLUME_NORM)  # the controller's pin
    pactl.release()
    assert pactl.commands_starting("set-sink-volume") == [
        ["set-sink-volume", SPEAKER, str(VOLUME_NORM)]
    ]

    pactl.commands.clear()
    pactl.holding = True
    volume.follow(SPEAKER)
    volume.set_volume(SPEAKER, raw_volume(-52.8))
    volume.follow(HEADPHONES)
    pactl.release()
    assert pactl.commands_starting("set-sink-volume") == []


# --- Setting ----------------------------------------------------------------------------


def test_setting_runs_pactl_with_the_raw_value_once_for_the_last_of_several(volume, pactl):
    applied = _Recorder(volume.applied)
    volume.follow(USB_CARD)
    pactl.commands.clear()

    volume.set_volume(USB_CARD, raw_volume(-30.0))
    assert pactl.commands[0] == ["set-sink-volume", USB_CARD, str(raw_volume(-30.0))]
    assert pactl.volumes[USB_CARD] == raw_volume(-30.0)
    assert applied.calls == [(USB_CARD,)]
    # Read back once, so that the value as applied is known as the radio's own.
    assert pactl.commands[1] == ["get-sink-volume", USB_CARD]
    assert len(pactl.commands) == 2

    # While pactl still runs, a slider dragged sets only where it ends.
    pactl.holding = True
    volume.set_volume(USB_CARD, raw_volume(-29.4))
    volume.set_volume(USB_CARD, raw_volume(-28.8))
    volume.set_volume(USB_CARD, raw_volume(-28.2))
    assert len(pactl.commands) == 3
    pactl.release()
    assert pactl.commands_starting("set-sink-volume")[1:] == [
        ["set-sink-volume", USB_CARD, str(raw_volume(-29.4))],
        ["set-sink-volume", USB_CARD, str(raw_volume(-28.2))],
    ]
    assert pactl.volumes[USB_CARD] == raw_volume(-28.2)
    assert len(applied.calls) == 3

    volume.set_volume(USB_CARD, 0)
    assert pactl.commands_starting("set-sink-volume")[-1] == ["set-sink-volume", USB_CARD, "0"]


def test_a_sink_not_followed_is_set_without_asking_and_not_read_back(volume, pactl):
    volume.follow(HEADPHONES)
    pactl.commands.clear()
    volume.set_volume(SPEAKER, VOLUME_NORM)
    assert pactl.commands == [["set-sink-volume", SPEAKER, str(VOLUME_NORM)]]


def test_its_own_value_is_no_change_even_as_a_bluetooth_device_rounds_it(volume, pactl, monitor):
    changes = _Recorder(volume.changed)
    volume.follow(HEADPHONES)
    volume.set_volume(HEADPHONES, 26214)
    assert pactl.volumes[HEADPHONES] == avrcp(26214) != 26214
    assert changes.calls == []

    # The system reports the change the radio made, and the device its rounding.
    monitor.sink_changed()
    monitor.sink_changed()
    assert pactl.commands_starting("get-sink-volume") == [["get-sink-volume", HEADPHONES]] * 3
    assert changes.calls == []


def test_a_change_made_elsewhere_is_reported_as_the_raw_value(volume, pactl, monitor, clock):
    changes = _Recorder(volume.changed)
    volume.follow(HEADPHONES)
    volume.set_volume(HEADPHONES, 26214)
    clock[0] += SETTLE_S  # the radio's own set has settled

    # The button on the headphones: up to step 76 of 127.
    pactl.volumes[HEADPHONES] = round(VOLUME_NORM * 76 / 127)
    monitor.sink_changed()
    assert changes.calls == [(HEADPHONES, round(VOLUME_NORM * 76 / 127))]

    # The same value again: no change. Silence: 0.
    monitor.sink_changed()
    assert len(changes.calls) == 1
    pactl.volumes[HEADPHONES] = 0
    monitor.sink_changed()
    assert changes.last == (HEADPHONES, 0)

    # Set by the radio again: its own value, as before, also once settled.
    volume.set_volume(HEADPHONES, raw_volume(-12.0))
    monitor.sink_changed()
    clock[0] += SETTLE_S
    monitor.sink_changed()
    assert len(changes.calls) == 2


def test_what_a_device_answers_shortly_after_a_set_is_the_radios_own(volume, pactl, monitor, clock):
    """A device with 16 steps of its own answers a set with the step it took, further
    off than the tolerance; within the settle window that is no change, and the slider
    stays where the finger put it."""
    changes = _Recorder(volume.changed)
    volume.follow(HEADPHONES)
    volume.set_volume(HEADPHONES, VOLUME_NORM // 2)  # the device took step 8 of 16
    pactl.volumes[HEADPHONES] = round(VOLUME_NORM * 8 / 16) + 3000
    clock[0] += SETTLE_S / 2
    monitor.sink_changed()
    assert changes.calls == []

    # The same answer again, after the window: known already, no change.
    clock[0] += SETTLE_S
    monitor.sink_changed()
    assert changes.calls == []

    # A button on the device after the window: a change.
    pactl.volumes[HEADPHONES] = round(VOLUME_NORM * 12 / 16)
    monitor.sink_changed()
    assert changes.calls == [(HEADPHONES, round(VOLUME_NORM * 12 / 16))]


def test_a_change_before_any_setting_is_reported(volume, pactl, monitor):
    changes = _Recorder(volume.changed)
    volume.follow(USB_CARD)
    pactl.volumes[USB_CARD] = 26214
    monitor.sink_changed()
    assert changes.calls == [(USB_CARD, 26214)]


def test_events_of_other_objects_and_of_other_kinds_are_ignored(volume, pactl, monitor):
    volume.follow(USB_CARD)
    pactl.commands.clear()
    for line in (
        "Event 'change' on sink-input #12",
        "Event 'new' on sink #58",
        "Event 'remove' on sink #58",
        "Event 'change' on source #3",
        "Event 'change' on card #48",
        "Event 'change' on server #0",
    ):
        monitor.line.emit(line)
    assert pactl.commands == []


def test_a_sink_that_takes_no_volume_is_reported(volume, pactl):
    failures = _Recorder(volume.failed)
    applied = _Recorder(volume.applied)
    pactl.refuse_setting = "Failure: No such entity\n"
    volume.follow(USB_CARD)
    volume.set_volume(USB_CARD, raw_volume(-30.0))
    assert failures.calls == [(USB_CARD, "Failure: No such entity")]
    assert applied.calls == []
    assert pactl.commands_starting("get-sink-volume") == []


def test_following_starts_the_monitor_once_and_shutdown_stops_it(volume, pactl, monitor):
    assert volume.following is None and monitor.starts == 0
    volume.follow(SPEAKER)
    volume.follow(HEADPHONES)
    volume.follow(None)
    volume.follow(SPEAKER)
    assert monitor.starts == 1 and monitor.running
    assert volume.following == SPEAKER
    volume.shutdown()
    assert not monitor.running and pactl.stopped


def test_without_pactl_nothing_runs(monkeypatch):
    monkeypatch.setattr(system_volume, "find_pactl", lambda: None)
    volume = SystemVolume()
    assert not volume.available
    volume.follow(SPEAKER)
    volume.set_volume(SPEAKER, raw_volume(-30.0))
    volume.shutdown()


def test_disabled_it_does_not_even_look_for_pactl(monkeypatch):
    """The radio next to other programs (no `--system-volume`) leaves the system's
    volume alone, pactl or not."""
    looked = []
    monkeypatch.setattr(system_volume, "find_pactl", lambda: looked.append(1) or "pactl")
    volume = SystemVolume(enabled=False)
    assert not volume.available and looked == []
    volume.follow(SPEAKER)
    volume.set_volume(SPEAKER, raw_volume(-30.0))
    volume.shutdown()


# --- Never shown quieter than it plays ---------------------------------------------------


def test_a_device_left_quieter_than_set_moves_the_slider(volume, pactl, monitor, clock):
    """WirePlumber writes back the volume it stored for the device a moment after it
    connects (here step 76 while the radio set 73 %): the answer falls into the settle
    window, and only the read after it shows the truth."""
    changes = _Recorder(volume.changed)
    volume.follow(HEADPHONES)
    volume.set_volume(HEADPHONES, raw_of_share(0.73))
    pactl.volumes[HEADPHONES] = round(VOLUME_NORM * 76 / 127)  # WirePlumber's
    monitor.sink_changed()
    assert changes.calls == []  # within the window: taken for the radio's own

    clock[0] += SETTLE_S
    volume._on_verify_timer()
    assert changes.calls == [(HEADPHONES, round(VOLUME_NORM * 76 / 127))]
    assert pactl.commands_starting("set-sink-volume") == [
        ["set-sink-volume", HEADPHONES, str(raw_of_share(0.73))]
    ]


def test_a_device_left_louder_than_set_is_set_again_once(volume, pactl, monitor, clock):
    changes = _Recorder(volume.changed)
    volume.follow(HEADPHONES)
    volume.set_volume(HEADPHONES, raw_of_share(0.1))
    pactl.volumes[HEADPHONES] = VOLUME_NORM  # a stored 100 % written back
    monitor.sink_changed()
    clock[0] += SETTLE_S
    volume._on_verify_timer()
    assert (
        pactl.commands_starting("set-sink-volume")
        == [["set-sink-volume", HEADPHONES, str(raw_of_share(0.1))]] * 2
    )
    assert pactl.volumes[HEADPHONES] == avrcp(raw_of_share(0.1))
    assert changes.calls == []

    # Taken this time: nothing more.
    clock[0] += SETTLE_S
    volume._on_verify_timer()
    assert len(pactl.commands_starting("set-sink-volume")) == 2 and changes.calls == []


def test_a_device_that_stays_louder_is_shown_as_loud_as_it_plays(volume, pactl, monitor, clock):
    changes = _Recorder(volume.changed)
    volume.follow(HEADPHONES)
    volume.set_volume(HEADPHONES, raw_of_share(0.1))
    for _ in range(2):  # it goes back to its own level after every set
        pactl.volumes[HEADPHONES] = round(VOLUME_NORM * 40 / 127)
        monitor.sink_changed()
        clock[0] += SETTLE_S
        volume._on_verify_timer()
    assert len(pactl.commands_starting("set-sink-volume")) == 2
    assert changes.calls == [(HEADPHONES, round(VOLUME_NORM * 40 / 127))]

    # A new level from the radio is checked anew, with one more try.
    volume.set_volume(HEADPHONES, raw_of_share(0.2))
    pactl.volumes[HEADPHONES] = VOLUME_NORM
    clock[0] += SETTLE_S
    volume._on_verify_timer()
    assert pactl.commands_starting("set-sink-volume")[-1] == [
        "set-sink-volume",
        HEADPHONES,
        str(raw_of_share(0.2)),
    ]
    assert len(pactl.commands_starting("set-sink-volume")) == 4


def test_a_device_as_set_is_left_alone(volume, pactl, clock):
    changes = _Recorder(volume.changed)
    volume.follow(HEADPHONES)
    volume.set_volume(HEADPHONES, raw_of_share(0.5))  # rounded to a step: within tolerance
    clock[0] += SETTLE_S
    volume._on_verify_timer()
    assert changes.calls == [] and len(pactl.commands_starting("set-sink-volume")) == 1
