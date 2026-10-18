"""The volume of the output in the sound system, where the system has one to set.

With PipeWire (on the Raspberry Pi, through its PulseAudio interface and `pactl`) the
radio sets the volume of the sink it plays to instead of attenuating the samples itself
(`AudioPlayer.set_volume` stays at 1.0 then). The system then knows the level: PipeWire
passes it on to Bluetooth devices as their own volume (AVRCP absolute volume, in 127
steps), so that their beeps and tones come at that level and not at their full one, and
the buttons on such a device move the system's volume, which the radio follows
(`changed`). Where there is no `pactl` (Windows, a Linux without the PulseAudio tools)
`available` is False and the player attenuates itself.

The radio sets the system's volume only where it is the one program playing: the touch
radio started through `pi/start-touch.sh` (`--system-volume`). Next to other programs,
such as the touch surface in a window on the desktop of a Raspberry Pi, a loudspeaker
pinned at 100 % (below) would play their sounds at full volume; there the controller
builds this class disabled, `available` is False as without pactl, and the system's
volume stays with the user.

Only a sink that applies its volume in the device itself is left the level: PulseAudio's
flag HW_VOLUME_CTRL, which a Bluetooth device with AVRCP absolute volume carries.
`follow` asks for it first (`pactl -f json list sinks`, again for a few seconds, since
a Bluetooth device brings its volume control up after its audio) and holds the volume
until the answer; a sink without it is reported (`software_volume`) and never set to
a level. After a set the sink is read once more, and a device playing louder than the
radio set is set again: WirePlumber writes back the volume it stored for a device
right after it connects, racing the radio's own.
Where PipeWire applies the volume in software, the value lives on the device's route
and has to be copied to the node that processes the samples, and after a reboot that
copy can go missing: the route, which pactl reports, stands at a low volume while the
node plays at 1.0, at full volume. A set through pactl goes to the route as well and
changes nothing when the value is the same. Such a sink is set to 100 %
(`VOLUME_NORM`) and the player attenuates: a value lost on the way can only make it
quieter, never louder. A Bluetooth device with a volume of its own is not held so: it
would play at 100 % of that volume, its beeps and tones too.

The radio stays the source of the level: it writes the volume it remembers for an
output whenever that output starts to play (`ReceiverController`).

pactl takes the volume as PulseAudio's raw value (`VOLUME_NORM` = 65536 is 0 dB, cubic
below it: `CUBIC_DECADE_DB` per decade of the raw value, and a Bluetooth device's 127
steps lie evenly on it) rather than in dB: pactl takes a value with a leading minus as
relative to the volume set, and one with a decimal point as linear. `get-sink-volume`
gives the raw value per channel back the same way. This module deals in raw values;
which scale the slider follows (decibels for a loudspeaker, the share of its range for
a device with a volume of its own) is the controller's choice, `audio.volume`.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import time

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from audio.volume import CUBIC_DECADE_DB
from net.command import CommandResult, CommandRunner, LineMonitor

logger = logging.getLogger(__name__)

PACTL = "pactl"
COMMAND_TIMEOUT_MS = 5000
# PulseAudio's volume scale: this is 0 dB, 0 is silence.
VOLUME_NORM = 65536
# After the radio set a volume, a readback within this much of it (raw units) is its
# own value as the system applied it, not a change made elsewhere: a Bluetooth device
# rounds it to one of 127 steps (AVRCP), 516 raw units apart. So is any readback
# within `SETTLE_S` of the set: a device answers a set with the step it took, which
# may be further off (devices with 16 or 32 steps of their own), and while the slider
# is dragged the answers lag behind the sets; the slider must not jump under the
# finger. A button pressed on the device within that moment is missed, nothing more.
ECHO_TOLERANCE = VOLUME_NORM // 100
SETTLE_S = 0.5
_clock = time.monotonic

# The flag of a sink whose volume the device applies itself (`pactl list sinks`).
HARDWARE_VOLUME_FLAG = "HW_VOLUME_CTRL"
# A Bluetooth device brings its volume control (AVRCP) up a second or two after its
# audio, and until then PipeWire lists its sink without the flag: asked at once, it
# would be taken for a device without a volume of its own and set to 100 % of its own
# volume. The flag is asked again at every change of a sink and every
# `PROBE_RETRY_MS`, for `HARDWARE_WAIT_S`, before a sink counts as one that applies
# its volume in software.
HARDWARE_WAIT_S = 5.0
PROBE_RETRY_MS = 500
# When the sink is read once more after a set, the settle window being over: the
# radio never shows less than plays (`SystemVolume._verify`).
VERIFY_DELAY_MS = round(SETTLE_S * 1000) + 100

# One channel in the output of `pactl get-sink-volume`:
# "Volume: front-left: 26214 /  40% / -23.88 dB,   front-right: 26214 /  40% / -23.88 dB"
_CHANNEL = re.compile(r"(\d+)\s*/\s*\d+%\s*/\s*(?:-?[\d.]+|-inf)\s*dB")
# What `pactl subscribe` prints when a sink's volume (or anything else about it) changes;
# a sink input, a source or a card are not it.
_SINK_CHANGED = re.compile(r"^Event 'change' on sink #\d+$")


def find_pactl() -> str | None:
    """The path of pactl; None where there is none (Windows, a Linux without the
    PulseAudio tools)."""
    return shutil.which(PACTL)


def raw_volume(db: float | None) -> int:
    """pactl's raw value for a gain in dB; 0 for silence (None)."""
    if db is None:
        return 0
    return max(0, round(VOLUME_NORM * 10 ** (db / CUBIC_DECADE_DB)))


def db_of_raw(raw: int) -> float | None:
    """The gain in dB of pactl's raw value; None for silence (0)."""
    if raw <= 0:
        return None
    return CUBIC_DECADE_DB * math.log10(raw / VOLUME_NORM)


def raw_of_share(share: float) -> int:
    """pactl's raw value for a share of a device's own range (`audio.volume`): the raw
    scale is linear in a Bluetooth device's steps."""
    return max(0, round(VOLUME_NORM * share))


def share_of_raw(raw: int) -> float:
    """The share of a device's own range that pactl's raw value is."""
    return max(0, raw) / VOLUME_NORM


def sink_flags(output: str, sink: str) -> list[str] | None:
    """The flags of `sink` in the output of `pactl -f json list sinks`; None where it
    is not listed or the output is not that."""
    try:
        sinks = json.loads(output)
    except ValueError:
        return None
    if not isinstance(sinks, list):
        return None
    for entry in sinks:
        if isinstance(entry, dict) and entry.get("name") == sink:
            return [str(flag) for flag in entry.get("flags") or []]
    return None


def parse_volume(output: str) -> int | None:
    """The raw volume in the output of `pactl get-sink-volume`, of the loudest channel;
    None where there is none."""
    values = [int(match.group(1)) for match in _CHANNEL.finditer(output)]
    return max(values) if values else None


class SystemVolume(QObject):
    """Sets the volume of a sink of the system and follows the one of the output in use.

    `follow` first asks whether the sink applies its volume in the device (the flag
    HW_VOLUME_CTRL); a volume for it waits for the answer. A Bluetooth device brings its
    volume control up a moment after its audio, so the flag is asked again at every
    change of a sink and every `PROBE_RETRY_MS` for `HARDWARE_WAIT_S`; a sink still
    without it then is reported (`software_volume`) and never set to a level.
    `set_volume` queues one pactl command; several before it ran set only the last.
    After every volume set, the sink is read back, so that the value the system applied
    (a Bluetooth device rounds it) is known as the radio's own; a readback after a change
    event that differs from it, later than `SETTLE_S` after the set, is a change made
    elsewhere, `changed`. Once the settle window is over the sink is read once more
    (`VERIFY_DELAY_MS`), and the radio never shows less than plays: a device that ended
    quieter than set is reported (`changed`, the slider follows), one that ended louder is
    set again, once, and reported if it stays louder (WirePlumber writes back the volume
    it stored for a device right after it connects, racing the radio's own).
    """

    # The volume of a sink was changed elsewhere: its name, pactl's raw value (0 for
    # silence).
    changed = pyqtSignal(str, int)
    # The sink followed applies its volume in software, not in the device: its name. The
    # radio must not leave the level to it.
    software_volume = pyqtSignal(str)
    # A volume the radio set reached the sink (its name): the player can play at 1.0.
    applied = pyqtSignal(str)
    # The sink would not take the volume: its name and why (for the log).
    failed = pyqtSignal(str, str)

    def __init__(
        self,
        parent: QObject | None = None,
        runner: CommandRunner | None = None,
        monitor: LineMonitor | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(parent)
        if not enabled:  # the radio is not the one program playing: no pactl
            runner = monitor = None
        elif runner is None:
            program = find_pactl()
            if program:
                runner = CommandRunner(program, self)
                monitor = LineMonitor(program, ["subscribe"], self)
        self._runner = runner
        self._monitor = monitor
        if monitor is not None:
            monitor.line.connect(self._on_line)
        # The sink followed; the volume to set next (sink, raw); whether a command runs;
        # whether the sink followed is to be read, or verified after the settle window;
        # the raw value last set for it, which the readback is to confirm; the raw value
        # last read for it; whether the last value was set again for a device louder.
        self._sink: str | None = None
        self._pending: tuple[str, int] | None = None
        self._running = False
        self._read_wanted = False
        self._verify_wanted = False
        self._expected: int | None = None
        self._known: int | None = None
        self._reasserted = False
        self._settled_at = 0.0  # until when a readback is the radio's own
        # Whether the sink followed applies its volume in the device: None until known;
        # asked until `_probe_deadline`.
        self._hardware: bool | None = None
        self._probe_wanted = False
        self._probe_deadline = 0.0
        self._probe_timer = QTimer(self)
        self._probe_timer.setSingleShot(True)
        self._probe_timer.setInterval(PROBE_RETRY_MS)
        self._probe_timer.timeout.connect(self._on_probe_timer)
        self._verify_timer = QTimer(self)
        self._verify_timer.setSingleShot(True)
        self._verify_timer.setInterval(VERIFY_DELAY_MS)
        self._verify_timer.timeout.connect(self._on_verify_timer)

    @property
    def available(self) -> bool:
        """Whether there is a pactl to set the volume with."""
        return self._runner is not None

    @property
    def following(self) -> str | None:
        """The sink whose changes are reported."""
        return self._sink

    def follow(self, sink: str | None) -> None:
        """Takes the level of this sink from now on, once it is known to apply it in the
        device, and reports the changes of its volume; None for none."""
        if sink == self._sink:
            return
        if self._pending is not None and self._pending[0] == self._sink and not self._hardware:
            self._pending = None  # held for a sink not known to take it in the device
        self._sink = sink
        self._expected = None
        self._known = None
        self._read_wanted = False
        self._verify_wanted = False
        self._reasserted = False
        self._hardware = None
        self._probe_wanted = sink is not None
        self._probe_deadline = _clock() + HARDWARE_WAIT_S
        self._probe_timer.stop()
        self._verify_timer.stop()
        if sink is not None and self._monitor is not None:
            self._monitor.start()
        self._next()

    def set_volume(self, sink: str, raw: int) -> None:
        """Sets the sink's volume to pactl's raw value (`raw_volume`, `raw_of_share`;
        0 for silence); `applied` tells when it is there."""
        if self._runner is None:
            return
        self._pending = (sink, max(0, int(raw)))
        if sink == self._sink:
            self._reasserted = False
        self._next()

    def shutdown(self) -> None:
        self._probe_timer.stop()
        self._verify_timer.stop()
        if self._monitor is not None:
            self._monitor.stop()
        if self._runner is not None:
            self._runner.stop()

    def _on_line(self, line: str) -> None:
        if not _SINK_CHANGED.match(line) or self._sink is None:
            return
        if self._hardware is None:
            self._probe_wanted = True  # the volume control may have come up
        elif self._hardware:
            self._read_wanted = True
        self._next()

    def _on_probe_timer(self) -> None:
        if self._sink is not None and self._hardware is None:
            self._probe_wanted = True
            self._next()

    def _on_verify_timer(self) -> None:
        if self._sink is not None and self._hardware:
            self._verify_wanted = True
            self._next()

    def _next(self) -> None:
        if self._running or self._runner is None:
            return
        if self._probe_wanted and self._sink is not None:
            probed = self._sink
            self._probe_wanted = False
            self._run(
                ["-f", "json", "list", "sinks"], lambda result: self._on_sinks(probed, result)
            )
            return
        held = self._pending
        if held is not None and held[0] == self._sink and self._hardware is not True:
            if self._hardware is None:
                return  # until the answer
            self._pending = None  # applied in software: not the sink's to hold
        if self._pending is not None:
            sink, raw = self._pending
            self._pending = None
            self._run(
                ["set-sink-volume", sink, str(raw)],
                lambda result: self._on_set(sink, raw, result),
            )
        elif (self._verify_wanted or self._read_wanted) and self._sink and self._hardware:
            sink, verify = self._sink, self._verify_wanted
            self._verify_wanted = self._read_wanted = False
            self._run(
                ["get-sink-volume", sink],
                lambda result: self._on_read(sink, result, verify),
            )

    def _run(self, arguments: list[str], done) -> None:
        self._running = True
        self._runner.run(arguments, COMMAND_TIMEOUT_MS, done)  # type: ignore[union-attr]

    def _on_sinks(self, sink: str, result: CommandResult) -> None:
        self._running = False
        if sink == self._sink and self._hardware is None:
            flags = sink_flags(result.output, sink) if result.ok else None
            if flags is not None and HARDWARE_VOLUME_FLAG in flags:
                self._hardware = True
                logger.info("%s applies its volume in the device; it takes the level", sink)
            elif _clock() < self._probe_deadline:
                self._probe_timer.start()  # its volume control may come up yet
            else:
                self._hardware = False
                logger.info(
                    "%s applies its volume in software (flags %s); the player attenuates",
                    sink,
                    flags if flags is not None else _why(result) or "not listed",
                )
                self.software_volume.emit(sink)
        self._next()

    def _on_set(self, sink: str, raw: int, result: CommandResult) -> None:
        self._running = False
        if result.ok:
            if sink == self._sink:
                self._expected = raw
                self._settled_at = _clock() + SETTLE_S
                self._read_wanted = True
                self._verify_timer.start()
            self.applied.emit(sink)
        else:
            why = _why(result)
            logger.warning("pactl could not set the volume of %s: %s", sink, why)
            self.failed.emit(sink, why)
        self._next()

    def _on_read(self, sink: str, result: CommandResult, verify: bool) -> None:
        self._running = False
        raw = parse_volume(result.output) if result.ok else None
        if raw is None:
            logger.warning("pactl could not read the volume of %s: %s", sink, _why(result))
        elif sink == self._sink:
            expected = self._expected
            if verify and expected is not None:
                self._verify(sink, raw, expected)
            elif expected is not None and (
                abs(raw - expected) <= ECHO_TOLERANCE or _clock() < self._settled_at
            ):
                # As the system applied it; a Bluetooth device may report its rounding
                # a moment later, which is still ours.
                self._known = raw
            elif raw != self._known:
                self._expected = None
                self._known = raw
                self.changed.emit(sink, raw)
        self._next()

    def _verify(self, sink: str, raw: int, expected: int) -> None:
        """The settle window is over: the radio must not show less than plays."""
        self._known = raw
        if abs(raw - expected) <= ECHO_TOLERANCE:
            return
        if raw > expected and not self._reasserted:
            self._reasserted = True
            logger.info("%s plays at %d, louder than the %d set; set again", sink, raw, expected)
            if self._pending is None or self._pending[0] != sink:
                self._pending = (sink, expected)
            return
        logger.info("%s stays at %d after %d was set; the slider follows", sink, raw, expected)
        self._expected = None
        self.changed.emit(sink, raw)


def _why(result: CommandResult) -> str:
    text = (result.error or result.output).strip()
    return text or f"exit code {result.exit_code}"
