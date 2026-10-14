"""Bluetooth on the Raspberry Pi: pairing and connecting headphones (and loudspeakers)
through BlueZ, the Bluetooth stack of Linux, over its D-Bus API.

BlueZ (`bluetoothd`) owns the adapter and publishes it and every device it knows as
objects on the system bus: `org.bluez.Adapter1` for the adapter (power, discovery),
`org.bluez.Device1` for each device (pair, connect, trust, forget) and `org.bluez.Battery1`
where a device tells its charge. The audio itself is not BlueZ's business: PipeWire
registers the A2DP profile with BlueZ and turns a connected device into an output of
the system, which the radio then plays to like any sound card (`audio.player`).

`BluetoothManager` keeps the state (adapter, devices, what is going on) and does what
the surface asks; it talks to the bus through a `Bus` (`net.dbus`, shared with the
storage), which `QtBus` implements with QtDBus and the tests replace. Nothing here
waits for the bus: every call is asynchronous, and BlueZ reports every change through
signals, so there is no polling.

Pairing goes through an agent the radio offers BlueZ (`BluetoothAgent`, capability
NoInputNoOutput), made the default one at the start. Without any agent BlueZ switches
the adapter's bondable mode off, and the kernel then pairs without bonding: the key is
not kept, and a device once disconnected cannot connect again, only be paired anew (a
JBL box showed `Paired: yes, Bonded: no` on 2026-10-11, and BlueZ dropped the pairing
of a Jabra, while the radio registered no agent). Headphones pair "Just Works",
which the kernel accepts by itself for such an agent; where BlueZ does ask (a
confirmation, an authorization, the PIN of an old headset, a service of a device not
trusted), the agent says yes only for the device the surface is pairing and refuses
everyone else. The adapter stays as BlueZ sets it: not discoverable, so that no other
device lists it, and taking connections only from devices it knows. A paired device is
marked trusted, so that BlueZ lets it use its services without asking the agent.

The adapter carries the audio (A2DP) to one device at a time: BlueZ lets each of its
local stream endpoints serve one stream, also one at rest, and PipeWire registers one
for SBC, the codec headphones and loudspeakers share (SBC-XQ shares it, AAC is not in
Debian's PipeWire). A second device connecting then gets no A2DP ("Unable to select
SEP" in the log of bluetoothd) and plays over the hands-free profile, mono at 16 kHz
(the Jabra Elite 10 next to a JBL box, 2026-10-11). The manager therefore tells which
devices take audio (`Device.audio`), which one holds A2DP (`Device.a2dp`, from the
transport BlueZ publishes for it) and in which order they connected; moving A2DP from
one to the other is `audio.bluetooth_route`'s.

On Windows there is no system bus; the surface says so and leaves pairing to the
system, where headphones are an output like any other.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace

from PyQt6.QtCore import QObject, QTimer, pyqtClassInfo, pyqtSignal, pyqtSlot

from i18n import tr
from net.dbus import (
    ERROR_ACCESS_DENIED,
    ERROR_NO_REPLY,
    ERROR_SERVICE_UNKNOWN,
    OBJECT_MANAGER,
    Bus,
    QDBusMessage,
    QDBusObjectPath,
    QtBus,
    Rejected,
    Reply,
    plain,
)

logger = logging.getLogger(__name__)

SERVICE = "org.bluez"
ADAPTER = "org.bluez.Adapter1"
DEVICE = "org.bluez.Device1"
BATTERY = "org.bluez.Battery1"
# The stream of A2DP to a device, while it is set up (`.../dev_…/sep1/fd0`).
MEDIA_TRANSPORT = "org.bluez.MediaTransport1"
# The two sides of A2DP: headphones and loudspeakers list the sink among their UUIDs,
# the transport of the radio's stream to them carries the source (the radio's side).
A2DP_SINK_UUID = "0000110b-0000-1000-8000-00805f9b34fb"
A2DP_SOURCE_UUID = "0000110a-0000-1000-8000-00805f9b34fb"
AGENT_MANAGER = "org.bluez.AgentManager1"
AGENT = "org.bluez.Agent1"
AGENT_MANAGER_PATH = "/org/bluez"
# Where the radio offers its agent on the bus.
AGENT_PATH = "/org/dabradio/bluetooth/agent"
# Headphones and loudspeakers have neither keys nor a display, and neither has the radio
# for pairing: they pair "Just Works".
AGENT_CAPABILITY = "NoInputNoOutput"
# What an old headset (before Secure Simple Pairing, Bluetooth 2.1) takes as its PIN.
LEGACY_PIN = "0000"

# How long a tap on "search" lets the adapter look for devices, at most: a tap on a
# device ends it (`BluetoothManager.stop_scan`). Discovery takes radio time from a
# device playing, so it is not left running.
DISCOVERY_SECONDS = 12
# Only classic Bluetooth (BR/EDR): A2DP headphones and loudspeakers are such, and a
# discovery over Low Energy too would list every beacon and tracker around.
DISCOVERY_TRANSPORT = "bredr"
# Errors of BlueZ and of the bus (the names of the D-Bus errors).
ERROR_ALREADY_EXISTS = "org.bluez.Error.AlreadyExists"
ERROR_IN_PROGRESS = "org.bluez.Error.InProgress"
ERROR_NOT_READY = "org.bluez.Error.NotReady"
ERROR_FAILED = "org.bluez.Error.Failed"
ERROR_REJECTED = "org.bluez.Error.Rejected"
ERROR_AUTHENTICATION = (
    "org.bluez.Error.AuthenticationCanceled",
    "org.bluez.Error.AuthenticationFailed",
    "org.bluez.Error.AuthenticationRejected",
    "org.bluez.Error.AuthenticationTimeout",
    "org.bluez.Error.ConnectionAttemptFailed",
)

# What a device is, from the icon name BlueZ derives from its class of device.
HEADPHONES = "headphones"
SPEAKER = "speaker"
PHONE = "phone"
OTHER = "other"
_KINDS = {
    "audio-headset": HEADPHONES,
    "audio-headphones": HEADPHONES,
    "audio-card": SPEAKER,
    "audio-speakers": SPEAKER,
    "phone": PHONE,
}


@dataclass(frozen=True)
class Device:
    """A device BlueZ knows: paired, or seen in the last discovery."""

    path: str  # "/org/bluez/hci0/dev_AA_BB_CC_DD_EE_FF"
    address: str
    name: str  # the alias, else the name, else the address
    icon: str  # BlueZ's icon name ("audio-headset", "phone" …), "" where unknown
    paired: bool = False
    trusted: bool = False
    connected: bool = False
    rssi: int | None = None  # only while discovering
    battery: int | None = None  # percent, where the device tells
    audio: bool = False  # takes audio from the radio (lists the A2DP sink)
    a2dp: bool = False  # holds the radio's A2DP stream, the one the adapter carries

    @property
    def kind(self) -> str:
        return _KINDS.get(self.icon, OTHER)

    @property
    def named(self) -> bool:
        """Whether BlueZ has a name for it; an unnamed device just seen is noise."""
        return bool(self.name) and self.name != self.address


def device_from_properties(path: str, properties: dict, battery: dict | None = None) -> Device:
    """A device from the properties of its `org.bluez.Device1` (and `org.bluez.Battery1`)."""
    address = str(properties.get("Address", ""))
    name = str(properties.get("Alias") or properties.get("Name") or address)
    percentage = None if battery is None else battery.get("Percentage")
    return Device(
        path=path,
        address=address,
        name=name,
        icon=str(properties.get("Icon", "")),
        paired=bool(properties.get("Paired", False)),
        trusted=bool(properties.get("Trusted", False)),
        connected=bool(properties.get("Connected", False)),
        rssi=_int_or_none(properties.get("RSSI")),
        battery=_percentage(percentage),
        audio=_takes_audio(properties.get("UUIDs")),
    )


def _takes_audio(uuids: object) -> bool:
    return isinstance(uuids, list | tuple) and A2DP_SINK_UUID in (str(u).lower() for u in uuids)


def transport_device(properties: dict) -> str | None:
    """The device a transport of BlueZ belongs to, where it is the radio's A2DP stream
    (not one a phone plays to the radio over)."""
    if str(properties.get("UUID", "")).lower() != A2DP_SOURCE_UUID:
        return None
    device = str(properties.get("Device", ""))
    return device or None


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except TypeError, ValueError:
        return None


def _percentage(value: object) -> int | None:
    """`Percentage` is a D-Bus byte (`y`), which PyQt6 hands over as `bytes` of length 1:
    50 % arrives as b"2", and int() would read that character as the digit 2."""
    if isinstance(value, bytes | bytearray) and len(value) == 1:
        return value[0]
    return _int_or_none(value)


def update_device(device: Device, interface: str, changed: dict) -> Device:
    """The device with the properties `changed` of `interface` applied."""
    if interface == BATTERY:
        if "Percentage" in changed:
            return replace(device, battery=_percentage(changed["Percentage"]))
        return device
    if interface != DEVICE:
        return device
    fields = {}
    for name, value in changed.items():
        match name:
            case "Alias":
                fields["name"] = str(value) or device.address
            case "Name":
                if device.name == device.address:
                    fields["name"] = str(value) or device.address
            case "Icon":
                fields["icon"] = str(value)
            case "Paired":
                fields["paired"] = bool(value)
            case "Trusted":
                fields["trusted"] = bool(value)
            case "Connected":
                fields["connected"] = bool(value)
            case "RSSI":
                fields["rssi"] = _int_or_none(value)
            case "UUIDs":
                fields["audio"] = _takes_audio(value)
    return replace(device, **fields) if fields else device


def parse_managed_objects(objects: dict) -> tuple[str | None, dict, dict[str, Device]]:
    """Adapter path and properties (None and {} without one) and the devices by path
    from `GetManagedObjects` of BlueZ, those with the radio's A2DP stream marked."""
    adapter: str | None = None
    adapter_properties: dict = {}
    devices: dict[str, Device] = {}
    for path in sorted(objects):
        interfaces = objects[path]
        if ADAPTER in interfaces and adapter is None:
            adapter, adapter_properties = path, dict(interfaces[ADAPTER])
        if DEVICE in interfaces:
            devices[path] = device_from_properties(
                path, interfaces[DEVICE], interfaces.get(BATTERY)
            )
    for device in parse_transports(objects).values():
        if device in devices:
            devices[device] = replace(devices[device], a2dp=True)
    return adapter, adapter_properties, devices


def parse_transports(objects: dict) -> dict[str, str]:
    """The radio's A2DP streams in `GetManagedObjects`: the device of each by the path
    of its transport."""
    transports = {}
    for path, interfaces in objects.items():
        if MEDIA_TRANSPORT in interfaces:
            device = transport_device(interfaces[MEDIA_TRANSPORT])
            if device is not None:
                transports[path] = device
    return transports


def sorted_devices(devices: dict[str, Device]) -> list[Device]:
    """Connected first, then paired, then the rest by strength; unnamed devices that
    are not paired are left out."""
    shown = [d for d in devices.values() if d.paired or d.named]
    shown.sort(
        key=lambda d: (
            not d.connected,
            not d.paired,
            -(d.rssi if d.rssi is not None else -1000),
            d.name.casefold(),
        )
    )
    return shown


def error_text(reply: Reply) -> str:
    """What went wrong, for the surface."""
    detail = reply.message.strip()
    lowered = detail.lower()
    if reply.error in ERROR_AUTHENTICATION:
        return tr("Pairing failed: the device refused or did not answer. Is it in pairing mode?")
    if reply.error == ERROR_ACCESS_DENIED:
        return tr("No permission: on the Pi, add the user to the group bluetooth.")
    if reply.error == ERROR_SERVICE_UNKNOWN:
        return tr("The Bluetooth service (bluetoothd) is not running.")
    if reply.error in ERROR_NO_REPLY:
        return tr("Timeout: the device did not answer in time.")
    if reply.error == ERROR_NOT_READY:
        return tr("Bluetooth is off.")
    if reply.error == ERROR_IN_PROGRESS:
        return tr("An operation is already in progress; please wait.")
    if "profile-unavailable" in lowered:
        return tr(
            "No audio profile: on the Pi, PipeWire lacks the package "
            "libspa-0.2-bluetooth, or PipeWire is not running."
        )
    if "page-timeout" in lowered or "host is down" in lowered:
        return tr("The device does not answer. Is it switched on and in range?")
    if "connection-refused" in lowered or "refused" in lowered:
        return tr("The device refused the connection.")
    if "already connected" in lowered:
        return tr("Already connected.")
    if "create-socket" in lowered:
        # BlueZ found no free stream endpoint: another device holds the audio.
        return tr("No audio connection free: another Bluetooth device holds it.")
    return tr("BlueZ reports: {detail}").format(detail=detail or reply.error)


AgentAnswer = Callable[[str, list], list]


@pyqtClassInfo("D-Bus Interface", AGENT)
class BluetoothAgent(QObject):
    """The methods of `org.bluez.Agent1` the radio offers BlueZ (`BluetoothManager`
    registers it). Each call goes to `answer` with its method name and plain arguments
    and is replied to with the values it returns or, where it raises `Rejected`, with
    that error. Qt delivers the calls in this object's thread and finds a slot by its
    name and D-Bus signature ("uint" is `u`, "ushort" `q`, a final `QDBusMessage` is the
    call); the reply goes out from here (`setDelayedReply`), since a refusal has to be a
    D-Bus error."""

    def __init__(self, bus: Bus, answer: AgentAnswer, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._bus = bus
        self._answer = answer

    def _reply(self, message: QDBusMessage, method: str, *arguments: object) -> None:
        message.setDelayedReply(True)
        try:
            values = self._answer(method, [plain(argument) for argument in arguments])
        except Rejected as refusal:
            self._bus.refuse(message, refusal.error, refusal.text)
        else:
            self._bus.reply(message, values)

    @pyqtSlot(QDBusMessage)
    def Release(self, message: QDBusMessage) -> None:
        self._reply(message, "Release")

    @pyqtSlot(QDBusObjectPath, QDBusMessage, result=str)
    def RequestPinCode(self, device: QDBusObjectPath, message: QDBusMessage) -> str:
        self._reply(message, "RequestPinCode", device)
        return ""  # the reply went out already

    @pyqtSlot(QDBusObjectPath, str, QDBusMessage)
    def DisplayPinCode(self, device: QDBusObjectPath, pincode: str, message: QDBusMessage) -> None:
        self._reply(message, "DisplayPinCode", device, pincode)

    @pyqtSlot(QDBusObjectPath, QDBusMessage, result="uint")
    def RequestPasskey(self, device: QDBusObjectPath, message: QDBusMessage) -> int:
        self._reply(message, "RequestPasskey", device)
        return 0  # the reply went out already

    @pyqtSlot(QDBusObjectPath, "uint", "ushort", QDBusMessage)
    def DisplayPasskey(
        self, device: QDBusObjectPath, passkey: int, entered: int, message: QDBusMessage
    ) -> None:
        self._reply(message, "DisplayPasskey", device, passkey, entered)

    @pyqtSlot(QDBusObjectPath, "uint", QDBusMessage)
    def RequestConfirmation(
        self, device: QDBusObjectPath, passkey: int, message: QDBusMessage
    ) -> None:
        self._reply(message, "RequestConfirmation", device, passkey)

    @pyqtSlot(QDBusObjectPath, QDBusMessage)
    def RequestAuthorization(self, device: QDBusObjectPath, message: QDBusMessage) -> None:
        self._reply(message, "RequestAuthorization", device)

    @pyqtSlot(QDBusObjectPath, str, QDBusMessage)
    def AuthorizeService(self, device: QDBusObjectPath, uuid: str, message: QDBusMessage) -> None:
        self._reply(message, "AuthorizeService", device, uuid)

    @pyqtSlot(QDBusMessage)
    def Cancel(self, message: QDBusMessage) -> None:
        self._reply(message, "Cancel")


class BluetoothManager(QObject):
    """Adapter and devices of BlueZ, and what can be done with them.

    `start` reads everything once BlueZ is there and registers the agent
    (`BluetoothAgent`, answered by `answer_agent`); from then on its signals keep the
    state current. `activity` tells what an action in progress does (pairing,
    connecting), `changed` follows every change, `failed` what went wrong,
    `device_connected` a device that has just connected.
    """

    changed = pyqtSignal()
    failed = pyqtSignal(str)  # what went wrong, for the surface
    device_connected = pyqtSignal(str)  # the path of the device

    def __init__(self, parent: QObject | None = None, bus: Bus | None = None) -> None:
        super().__init__(parent)
        self._bus = bus if bus is not None else QtBus(SERVICE, self)
        self.adapter: str | None = None  # the path of the adapter, None without one
        self.service_present = False  # bluetoothd answered on the bus
        self.powered = False
        self.discovering = False
        self.devices: dict[str, Device] = {}  # by path
        self.activity = ""  # e.g. "Pairing with “WH-1000XM4” …"
        # The radio's A2DP streams: the device of each by the path of its transport.
        self._transports: dict[str, str] = {}
        # When each device connected, as a running number: the newest is the highest.
        self._connected_order: dict[str, int] = {}
        self._sequence = 0
        self._reading = False
        self._discovery_started = False
        self._discovery_timer = QTimer(self)
        self._discovery_timer.setSingleShot(True)
        self._discovery_timer.setInterval(DISCOVERY_SECONDS * 1000)
        self._discovery_timer.timeout.connect(self._stop_discovery)
        # The agent: offered on the bus once, registered with BlueZ at the start and
        # whenever bluetoothd comes back; the device the surface is pairing, the only
        # one it says yes for.
        self._agent = BluetoothAgent(self._bus, self.answer_agent, self)
        self._agent_offered = False
        self._agent_requested = False
        self.agent_registered = False
        self._pairing: str | None = None

        self._bus.interfaces_added.connect(self._on_interfaces_added)
        self._bus.interfaces_removed.connect(self._on_interfaces_removed)
        self._bus.properties_changed.connect(self._on_properties_changed)
        self._bus.service_changed.connect(self._on_service_changed)

    @property
    def available(self) -> bool:
        """Whether there is a system bus to ask (not on Windows)."""
        return self._bus.connected

    @property
    def device_list(self) -> list[Device]:
        """The devices for the list: connected first, then paired, then those seen."""
        return sorted_devices(self.devices)

    @property
    def connected_device(self) -> Device | None:
        return next((d for d in self.device_list if d.connected), None)

    @property
    def a2dp_device(self) -> Device | None:
        """The device holding the radio's A2DP stream, None while none does."""
        return next((d for d in self.devices.values() if d.a2dp), None)

    @property
    def audio_devices(self) -> list[Device]:
        """The devices taking audio that are connected, the one connected last first."""
        connected = [d for d in self.devices.values() if d.connected and d.audio]
        connected.sort(key=lambda d: self._connected_order.get(d.path, 0), reverse=True)
        return connected

    @property
    def newest_audio_device(self) -> Device | None:
        """The device taking audio that connected last; None without one connected."""
        return next(iter(self.audio_devices), None)

    def device(self, path: str) -> Device | None:
        return self.devices.get(path)

    def device_by_address(self, address: str) -> Device | None:
        wanted = address.upper()
        return next((d for d in self.devices.values() if d.address.upper() == wanted), None)

    def start(self) -> None:
        """Reads adapter and devices and registers the agent; BlueZ's signals keep them
        current from then on."""
        if self.available:
            self._read()
            self._register_agent()

    def stop(self) -> None:
        """Ends a discovery still running (on leaving the page)."""
        self.stop_scan()

    def stop_scan(self) -> None:
        """Ends a discovery the radio started, before `DISCOVERY_SECONDS` are over: the
        surface does so as soon as a device is tapped, since the one looked for has been
        found then, and the inquiry takes the radio time a device playing needs (the
        sound of headphones stuttered while it ran, with the JBL and the Jabra on the
        Pi, 2026-10-11)."""
        if self._discovery_started:
            self._stop_discovery()

    def shutdown(self) -> None:
        self.stop()

    # --- Reading ----------------------------------------------------------------------

    def _read(self) -> None:
        if self._reading:
            return
        self._reading = True
        self._bus.call("/", OBJECT_MANAGER, "GetManagedObjects", [], self._on_managed_objects)

    def _on_managed_objects(self, reply: Reply) -> None:
        self._reading = False
        if not reply.ok or not isinstance(reply.value, dict):
            if reply.ok:
                logger.warning("GetManagedObjects answered with %r, not a dict", type(reply.value))
            self.service_present = reply.error != ERROR_SERVICE_UNKNOWN and reply.ok
            self._forget_all()
            if not reply.ok and reply.error != ERROR_SERVICE_UNKNOWN:
                self.failed.emit(error_text(reply))
            self.changed.emit()
            return
        self.service_present = True
        self.adapter, properties, self.devices = parse_managed_objects(reply.value)
        self._transports = parse_transports(reply.value)
        self.powered = bool(properties.get("Powered", False))
        self.discovering = bool(properties.get("Discovering", False))
        # Which of those connected came last is not known: the one holding A2DP counts
        # as the newest, so that the radio keeps playing where it plays.
        connected = sorted(
            (d for d in self.devices.values() if d.connected), key=lambda d: (d.a2dp, d.path)
        )
        self._connected_order = {}
        for device in connected:
            self._note_connected(device.path)
        self.changed.emit()

    def _forget_all(self) -> None:
        self.adapter = None
        self.powered = self.discovering = False
        self.devices = {}
        self._transports = {}
        self._connected_order = {}

    def _note_connected(self, path: str) -> None:
        self._sequence += 1
        self._connected_order[path] = self._sequence

    def _mark_a2dp(self, device: str) -> None:
        """Brings `Device.a2dp` of a device in line with the transports."""
        if device in self.devices:
            holds = device in self._transports.values()
            self.devices[device] = replace(self.devices[device], a2dp=holds)

    # --- Signals of BlueZ -------------------------------------------------------------

    def _on_interfaces_added(self, path: str, interfaces: dict) -> None:
        if ADAPTER in interfaces and self.adapter is None:
            self.adapter = path
            properties = interfaces[ADAPTER]
            self.powered = bool(properties.get("Powered", False))
            self.discovering = bool(properties.get("Discovering", False))
            self.service_present = True
        if DEVICE in interfaces:
            self.devices[path] = device_from_properties(
                path, interfaces[DEVICE], interfaces.get(BATTERY)
            )
            self._mark_a2dp(path)
            if self.devices[path].connected:
                self._note_connected(path)
        elif BATTERY in interfaces and path in self.devices:
            self.devices[path] = update_device(self.devices[path], BATTERY, interfaces[BATTERY])
        if MEDIA_TRANSPORT in interfaces:
            device = transport_device(interfaces[MEDIA_TRANSPORT])
            if device is not None:
                self._transports[path] = device
                self._mark_a2dp(device)
        self.changed.emit()

    def _on_interfaces_removed(self, path: str, names: list) -> None:
        if DEVICE in names:
            self.devices.pop(path, None)
            self._connected_order.pop(path, None)
            self._transports = {t: d for t, d in self._transports.items() if d != path}
        elif BATTERY in names and path in self.devices:
            self.devices[path] = replace(self.devices[path], battery=None)
        if MEDIA_TRANSPORT in names and path in self._transports:
            self._mark_a2dp(self._transports.pop(path))
        if ADAPTER in names and path == self.adapter:
            self._forget_all()
        self.changed.emit()

    def _on_properties_changed(self, path: str, interface: str, changed: dict) -> None:
        connected = False
        if interface == ADAPTER and path == self.adapter:
            if "Powered" in changed:
                self.powered = bool(changed["Powered"])
                if not self.powered:
                    self.discovering = False
            if "Discovering" in changed:
                self.discovering = bool(changed["Discovering"])
                if not self.discovering:
                    self._discovery_started = False
                    self._discovery_timer.stop()
        elif interface in (DEVICE, BATTERY) and path in self.devices:
            before = self.devices[path]
            self.devices[path] = update_device(before, interface, changed)
            if self.devices[path].connected != before.connected:
                if self.devices[path].connected:
                    self._note_connected(path)
                    connected = True
                else:
                    self._connected_order.pop(path, None)
        else:
            return
        # The connection first: a listener of both learns it before it acts on the change.
        if connected:
            self.device_connected.emit(path)
        self.changed.emit()

    def _on_service_changed(self, running: bool) -> None:
        if running:
            self._read()
            self._register_agent()
            return
        # bluetoothd went, and its agents with it.
        self._agent_requested = self.agent_registered = False
        self._pairing = None
        self.service_present = False
        self._forget_all()
        self.activity = ""
        self.changed.emit()

    # --- The agent --------------------------------------------------------------------

    def _register_agent(self) -> None:
        """Offers the agent on the bus and registers it with BlueZ as the default one:
        only while an agent is registered does BlueZ let the adapter bond, that is keep
        the key of a pairing."""
        if self._agent_requested:
            return
        if not self._agent_offered:
            self._agent_offered = self._bus.register_object(AGENT_PATH, self._agent)
            if not self._agent_offered:
                logger.warning(
                    "Could not offer the Bluetooth agent on the bus; pairings are not kept"
                )
                return
        self._agent_requested = True
        self._bus.call(
            AGENT_MANAGER_PATH,
            AGENT_MANAGER,
            "RegisterAgent",
            [self._bus.object_path(AGENT_PATH), AGENT_CAPABILITY],
            self._on_agent_registered,
        )

    def _on_agent_registered(self, reply: Reply) -> None:
        if not reply.ok and reply.error != ERROR_ALREADY_EXISTS:
            logger.warning("BlueZ refused the Bluetooth agent: %s %s", reply.error, reply.message)
            self._agent_requested = False
            return
        self._bus.call(
            AGENT_MANAGER_PATH,
            AGENT_MANAGER,
            "RequestDefaultAgent",
            [self._bus.object_path(AGENT_PATH)],
            self._on_default_agent,
        )

    def _on_default_agent(self, reply: Reply) -> None:
        if not reply.ok:
            logger.warning(
                "BlueZ did not make the Bluetooth agent the default: %s %s",
                reply.error,
                reply.message,
            )
            self._agent_requested = False
            return
        self.agent_registered = True
        logger.info("Bluetooth agent registered (%s); pairings are kept", AGENT_CAPABILITY)

    def answer_agent(self, method: str, arguments: list) -> list:
        """What the agent answers BlueZ: the return values of `method`, or `Rejected`.
        Yes only for the device the surface is pairing: a confirmation, an authorization,
        a service, the PIN of an old headset (a passkey to type in it cannot give). Any
        other device is refused: one the radio did not pair gets no service, and one it
        did is trusted, for which BlueZ does not ask."""
        device = str(arguments[0]) if arguments else ""
        known = self.devices.get(device)
        name = known.name if known is not None else device
        being_paired = bool(device) and device == self._pairing
        if method in ("Release", "Cancel"):
            logger.info("Bluetooth agent: %s", method)
            if method == "Release":
                self._agent_requested = self.agent_registered = False
            return []
        if method in ("DisplayPinCode", "DisplayPasskey"):
            logger.info("Bluetooth agent: %s for %s", method, name)
            return []
        if being_paired and method in (
            "RequestConfirmation",
            "RequestAuthorization",
            "AuthorizeService",
        ):
            logger.info("Bluetooth agent: %s for %s accepted", method, name)
            return []
        if being_paired and method == "RequestPinCode":
            logger.info("Bluetooth agent: PIN %s offered to %s", LEGACY_PIN, name)
            return [LEGACY_PIN]
        logger.warning(
            "Bluetooth agent: %s for %s refused (%s)",
            method,
            name,
            "no passkey to type" if being_paired else "not being paired from the radio",
        )
        raise Rejected(ERROR_REJECTED, "Refused by the radio")

    # --- Actions ----------------------------------------------------------------------

    def set_powered(self, on: bool) -> None:
        if self.adapter is None:
            return
        self._bus.set_property(self.adapter, ADAPTER, "Powered", on, self._on_action_done)

    def scan(self) -> None:
        """Lets the adapter look for classic Bluetooth devices for `DISCOVERY_SECONDS`."""
        if self.adapter is None or not self.powered or self.discovering:
            return
        self._bus.call(
            self.adapter,
            ADAPTER,
            "SetDiscoveryFilter",
            # A dict goes out as a{sv} with the value inside the variant already; wrapped
            # in a QDBusVariant too it became a variant in a variant, which BlueZ refused
            # as "Invalid arguments" (on the Pi, 09.10.2026).
            [{"Transport": DISCOVERY_TRANSPORT}],
            self._on_filter_set,
        )

    def _on_filter_set(self, reply: Reply) -> None:
        if not reply.ok:
            self.failed.emit(error_text(reply))
            return
        self._bus.call(self.adapter, ADAPTER, "StartDiscovery", [], self._on_discovery_started)

    def _on_discovery_started(self, reply: Reply) -> None:
        if not reply.ok:
            if reply.error != ERROR_IN_PROGRESS:
                self.failed.emit(error_text(reply))
            return
        self._discovery_started = True
        self._discovery_timer.start()

    def _stop_discovery(self) -> None:
        self._discovery_timer.stop()
        self._discovery_started = False
        if self.adapter is not None:
            self._bus.call(self.adapter, ADAPTER, "StopDiscovery", [], lambda _reply: None)

    def pair(self, device: Device) -> None:
        """Pairs a new device, marks it trusted and connects it."""
        if self.activity:
            return
        self._set_activity(tr("Pairing with “{name}” …").format(name=device.name))
        # The agent says yes for it until pairing and connecting are through.
        self._pairing = device.path
        self._bus.call(
            device.path, DEVICE, "Pair", [], lambda reply: self._on_paired(device, reply)
        )

    def _on_paired(self, device: Device, reply: Reply) -> None:
        if not reply.ok and reply.error != ERROR_ALREADY_EXISTS:
            self._fail(reply)
            return
        # Trusted: BlueZ lets it reconnect on its own, without asking an agent.
        self._bus.set_property(device.path, DEVICE, "Trusted", True, lambda _reply: None)
        self._set_activity(tr("Connecting to “{name}” …").format(name=device.name))
        self._bus.call(device.path, DEVICE, "Connect", [], self._on_action_done)

    def connect_device(self, device: Device, done: Callable[[Reply], None] | None = None) -> bool:
        """Connects a paired device; `done` hears how it went, after the manager. False
        while another action is in progress, which leaves it undone."""
        if self.activity:
            return False
        self._set_activity(tr("Connecting to “{name}” …").format(name=device.name))

        def connected(reply: Reply) -> None:
            self._on_action_done(reply)
            if done is not None:
                done(reply)

        self._bus.call(device.path, DEVICE, "Connect", [], connected)
        return True

    def release_a2dp(self, device: Device, done: Callable[[Reply], None]) -> None:
        """Ends the radio's A2DP stream to a device, which frees it for another. A device
        with the hands-free profile stays connected over that (the Jabra Elite 10), one
        without goes altogether a few seconds later (the JBL box, whose hands-free
        profile belongs to the phone; both 2026-10-11)."""
        self._bus.call(device.path, DEVICE, "DisconnectProfile", [A2DP_SINK_UUID], done)

    def connect_a2dp(self, device: Device, done: Callable[[Reply], None]) -> None:
        """Sets up the radio's A2DP stream to a device connected without it; WirePlumber
        switches its card from the hands-free profile to A2DP by itself (0.4 s with the
        Jabra Elite 10, 2026-10-11)."""
        self._bus.call(device.path, DEVICE, "ConnectProfile", [A2DP_SINK_UUID], done)

    def show_activity(self, text: str) -> None:
        """Tells what is going on (`activity`) for an action of someone else's, such as
        `audio.bluetooth_route` moving the audio; "" when it is over."""
        self._set_activity(text)

    def report_failure(self, text: str) -> None:
        self.failed.emit(text)

    def disconnect_device(self, device: Device) -> None:
        if self.activity:
            return
        self._set_activity(tr("Disconnecting “{name}” …").format(name=device.name))
        self._bus.call(device.path, DEVICE, "Disconnect", [], self._on_action_done)

    def forget(self, device: Device) -> None:
        """Removes the device from BlueZ, pairing and all."""
        if self.activity or self.adapter is None:
            return
        self._set_activity(tr("Removing “{name}” …").format(name=device.name))
        self._bus.call(
            self.adapter,
            ADAPTER,
            "RemoveDevice",
            [self._bus.object_path(device.path)],
            self._on_action_done,
        )

    def _on_action_done(self, reply: Reply) -> None:
        if reply.ok:
            self._pairing = None
            self._set_activity("")
        else:
            self._fail(reply)

    def _fail(self, reply: Reply) -> None:
        self._pairing = None
        self.activity = ""
        self.failed.emit(error_text(reply))
        self.changed.emit()

    def _set_activity(self, text: str) -> None:
        self.activity = text
        self.changed.emit()
