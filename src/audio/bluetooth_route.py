"""Moves the radio's audio connection (A2DP) from one Bluetooth device to another.

The adapter carries A2DP to one device at a time (`net.bluetooth`): a second device
connecting plays over the hands-free profile, mono at 16 kHz, which sounds tinny.
`BluetoothRoute.move_to` gives A2DP to the device that is to play: it ends the stream
to the device holding it (`BluetoothManager.release_a2dp`), sets one up to the new one
(`connect_a2dp`, or `connect_device` where it is not connected yet) and waits until
PipeWire lists the new device's sink with the A2DP profile (`pactl -f json list
sinks`). The old stream ends at once, the new one stands within about half a second,
and WirePlumber switches the card of the new device to A2DP by itself.

All the while the player is to stay silent, which the controller sees to: a device
changing profile takes its sink away and brings it back under the same name, and
PipeWire moves a stream whose sink goes to the default, which may be the loudspeaker,
pinned at 100 % for the player to attenuate, while the player plays at 1.0 to a device
that applies its volume itself. The move begins `HOLD_LEAD_MS` after it was asked for,
so that the player has let go of its sink by then.

Which device is to play is the controller's choice (`ReceiverController`), and so is
trying again: a device that has just connected may refuse the stream for a moment and
take it at the next try.
"""

from __future__ import annotations

import json
import logging
import re
import time

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from audio.system_volume import find_pactl
from i18n import tr
from net.bluetooth import BluetoothManager, Device, error_text
from net.command import CommandResult, CommandRunner
from net.dbus import Reply

logger = logging.getLogger(__name__)

# PipeWire names the sink of a Bluetooth device after its address, the same for A2DP
# and for the hands-free profile ("bluez_output.20_18_5B_8D_FD_52.1", 1 being its node
# for playback).
_SINK = re.compile(r"^bluez_output\.([0-9A-F]{2}(?:_[0-9A-F]{2}){5})\.\d+$", re.IGNORECASE)
_PLAYBACK_NODE = 1

# How long the player has to fall silent before the move begins: `holding` reaches it
# queued, in its own thread, and a sink must not go before it has let go of it.
HOLD_LEAD_MS = 150
# How long the stream to the device holding A2DP may take to end before the new one is
# set up all the same; how long PipeWire may take to list the new device's sink with
# the A2DP profile, asked every `SINK_POLL_MS`.
RELEASE_WAIT_MS = 3000
SINK_WAIT_S = 6.0
SINK_POLL_MS = 250
COMMAND_TIMEOUT_MS = 5000
_clock = time.monotonic


def sink_name(address: str) -> str:
    """The name PipeWire gives the sink of the Bluetooth device with this address."""
    return f"bluez_output.{address.upper().replace(':', '_')}.{_PLAYBACK_NODE}"


def sink_address(sink: str) -> str | None:
    """The address of the Bluetooth device a sink plays to; None for any other sink."""
    match = _SINK.match(sink)
    return match.group(1).replace("_", ":").upper() if match else None


def a2dp_sink(output: str, device_path: str) -> str | None:
    """The name of the sink that plays to the device with this path of BlueZ over A2DP,
    in the output of `pactl -f json list sinks`; None while it has none."""
    try:
        sinks = json.loads(output)
    except ValueError:
        return None
    if not isinstance(sinks, list):
        return None
    for entry in sinks:
        if not isinstance(entry, dict):
            continue
        properties = entry.get("properties") or {}
        profile = str(properties.get("api.bluez5.profile", ""))
        if properties.get("api.bluez5.path") == device_path and profile.startswith("a2dp"):
            return str(entry.get("name", ""))
    return None


class BluetoothRoute(QObject):
    """Moves A2DP to the device asked for (`move_to`), one move at a time.

    `finished` follows every move with the path of the device, whether it now plays
    over A2DP and, where BlueZ refused the stream, why (for the surface; else ""). The
    move is over by then: a listener may ask for the next one."""

    finished = pyqtSignal(str, bool, object)

    def __init__(
        self,
        bluetooth: BluetoothManager,
        parent: QObject | None = None,
        runner: CommandRunner | None = None,
    ) -> None:
        super().__init__(parent)
        if runner is None:
            program = find_pactl()
            if program:
                runner = CommandRunner(program, self)
        self._bluetooth = bluetooth
        self._runner = runner
        # The device the move goes to, None between moves; what the move waits for:
        # "hold" (the player to fall silent), "release" (the stream to `_releasing` to
        # end), "connect" (BlueZ to answer), "sink" (PipeWire to list the A2DP sink,
        # until `_sink_deadline`).
        self._target: str | None = None
        self._stage = ""
        self._releasing: str | None = None
        self._sink_deadline = 0.0
        # Whether the device is to be connected: only one that was not when the move was
        # asked for (the surface's "Connect"). One connected then that goes meanwhile is
        # leaving: BlueZ ends its stream before it reports it disconnected, and the
        # controller took it for one to give the stream to; connecting it would undo a
        # "Disconnect" of the user's.
        self._connecting = False
        # The text `activity` of the manager shows for the moves, while it is theirs;
        # a move following another takes it over.
        self._shown = ""
        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.setInterval(HOLD_LEAD_MS)
        self._hold_timer.timeout.connect(self._begin)
        self._release_timer = QTimer(self)
        self._release_timer.setSingleShot(True)
        self._release_timer.setInterval(RELEASE_WAIT_MS)
        self._release_timer.timeout.connect(self._on_release_timeout)
        self._poll_timer = QTimer(self)
        self._poll_timer.setSingleShot(True)
        self._poll_timer.setInterval(SINK_POLL_MS)
        self._poll_timer.timeout.connect(self._poll)
        bluetooth.changed.connect(self._on_bluetooth_changed)

    @property
    def busy(self) -> bool:
        return self._target is not None

    @property
    def target(self) -> str | None:
        """The path of the device a move in progress goes to."""
        return self._target

    def move_to(self, device: Device) -> bool:
        """Gives A2DP to this device, connecting it first where it is not; False while
        another move is in progress."""
        if self.busy:
            return False
        self._target = device.path
        self._connecting = not device.connected
        self._stage = "hold"
        if device.connected and self._bluetooth.activity in ("", self._shown):
            self._shown = tr("Moving the audio to “{name}” …").format(name=device.name)
            self._bluetooth.show_activity(self._shown)
        self._hold_timer.start()
        return True

    def _begin(self) -> None:
        """The player has had `HOLD_LEAD_MS` to fall silent: the move begins."""
        device = None if self._target is None else self._bluetooth.device(self._target)
        if self._stage != "hold" or device is None or not self._still_wanted(device):
            self._finish(False)
            return
        holder = self._bluetooth.a2dp_device
        if holder is not None and holder.path != device.path:
            logger.info("Moving the audio (A2DP) from %s to %s", holder.name, device.name)
            self._stage = "release"
            self._releasing = holder.path
            self._release_timer.start()
            self._bluetooth.release_a2dp(holder, self._on_released)
        else:
            logger.info("Setting up the audio (A2DP) to %s", device.name)
            self._connect()

    def shutdown(self) -> None:
        self._hold_timer.stop()
        self._release_timer.stop()
        self._poll_timer.stop()
        if self._runner is not None:
            self._runner.stop()

    # --- The steps ----------------------------------------------------------------------

    def _on_released(self, reply: Reply) -> None:
        if not reply.ok:
            logger.warning("BlueZ did not end the audio stream: %s %s", reply.error, reply.message)
        self._on_bluetooth_changed()

    def _on_bluetooth_changed(self) -> None:
        if self._stage != "release" or self._releasing is None:
            return
        holder = self._bluetooth.device(self._releasing)
        if holder is None or not holder.a2dp:
            self._connect()

    def _on_release_timeout(self) -> None:
        if self._stage == "release":
            logger.warning("The audio stream to %s did not end; going on", self._releasing)
            self._connect()

    def _connect(self) -> None:
        self._release_timer.stop()
        self._releasing = None
        device = None if self._target is None else self._bluetooth.device(self._target)
        if device is None or not self._still_wanted(device):
            self._finish(False)
            return
        self._stage = "connect"
        if device.a2dp:
            self._await_sink()
        elif device.connected:
            self._bluetooth.connect_a2dp(device, self._on_a2dp_connected)
        else:
            # The surface's "Connect": the manager shows it and reports a failure.
            self._clear_activity()
            if not self._bluetooth.connect_device(device, self._on_device_connected):
                self._finish(False)

    def _still_wanted(self, device: Device) -> bool:
        """Whether the device is still one to give the stream to: not one that has gone
        since the move was asked for."""
        if device.connected or self._connecting:
            return True
        logger.info("%s disconnected; the audio is not moved to it", device.name)
        return False

    def _on_a2dp_connected(self, reply: Reply) -> None:
        if self._stage != "connect":
            return
        if not reply.ok and "already connected" not in reply.message.lower():
            logger.warning(
                "BlueZ set up no audio stream to %s: %s %s",
                self._target,
                reply.error,
                reply.message,
            )
            self._finish(False, error_text(reply))
            return
        self._await_sink()

    def _on_device_connected(self, reply: Reply) -> None:
        if self._stage != "connect":
            return
        if not reply.ok:
            self._finish(False)  # the manager has reported it
            return
        self._await_sink()

    def _await_sink(self) -> None:
        self._stage = "sink"
        if self._runner is None:
            self._finish(True)
            return
        self._sink_deadline = _clock() + SINK_WAIT_S
        self._poll()

    def _poll(self) -> None:
        if self._stage != "sink" or self._runner is None:
            return
        self._runner.run(["-f", "json", "list", "sinks"], COMMAND_TIMEOUT_MS, self._on_sinks)

    def _on_sinks(self, result: CommandResult) -> None:
        if self._stage != "sink" or self._target is None:
            return
        sink = a2dp_sink(result.output, self._target) if result.ok else None
        if sink is not None:
            logger.info("The audio plays to %s over A2DP", sink)
            self._finish(True)
        elif _clock() >= self._sink_deadline:
            logger.warning(
                "PipeWire lists no A2DP sink of %s after %.0f s; playing on",
                self._target,
                SINK_WAIT_S,
            )
            self._finish(False)
        else:
            self._poll_timer.start()

    def _finish(self, ok: bool, why: str = "") -> None:
        """Ends the move, all of it before telling so: a listener of `finished` may
        ask for the next move at once, and ending this one afterwards would wipe the
        new one's state and leave its text in the activity, holding the Bluetooth page.
        The activity goes once no move follows."""
        path = self._target or ""
        self._target = None
        self._stage = ""
        self._releasing = None
        self._hold_timer.stop()
        self._release_timer.stop()
        self._poll_timer.stop()
        self.finished.emit(path, ok, why)
        if not self.busy:
            self._clear_activity()

    def _clear_activity(self) -> None:
        shown, self._shown = self._shown, ""
        if shown and self._bluetooth.activity == shown:
            self._bluetooth.show_activity("")
