"""A BlueZ that answers from a table of objects, for `net.bluetooth.BluetoothManager`:
the adapter of a Raspberry Pi with headphones connected (and playing over A2DP), a
loudspeaker and a phone paired, headphones just seen in a discovery and an unnamed
device beside them. With `one_stream` it carries A2DP to one device at a time, as the
Pi does (`net.bluetooth`)."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

from net.bluetooth import (
    A2DP_SINK_UUID,
    A2DP_SOURCE_UUID,
    ADAPTER,
    BATTERY,
    DEVICE,
    MEDIA_TRANSPORT,
)
from net.dbus import OBJECT_MANAGER, PROPERTIES, Bus, Reply

HCI0 = "/org/bluez/hci0"
SONY = f"{HCI0}/dev_F8_4E_17_01_02_03"
JBL = f"{HCI0}/dev_B8_F6_53_04_05_06"
BOSE = f"{HCI0}/dev_2C_41_A1_07_08_09"
PHONE = f"{HCI0}/dev_A0_B1_C2_0A_0B_0C"
UNNAMED = f"{HCI0}/dev_00_11_22_33_44_55"
SONY_TRANSPORT = f"{SONY}/sep1/fd0"

HANDS_FREE_UUID = "0000111e-0000-1000-8000-00805f9b34fb"
AUDIO_GATEWAY_UUID = "0000111f-0000-1000-8000-00805f9b34fb"
# What headphones and a loudspeaker list; a phone plays to others, it takes no audio.
HEADPHONE_UUIDS = [A2DP_SINK_UUID.upper(), HANDS_FREE_UUID]
SPEAKER_UUIDS = [A2DP_SINK_UUID]
PHONE_UUIDS = [A2DP_SOURCE_UUID, AUDIO_GATEWAY_UUID]


def _device(address, name, icon, paired=False, trusted=False, connected=False, rssi=None, uuids=()):
    properties = {
        "Address": address,
        "Alias": name or address,
        "Icon": icon,
        "Paired": paired,
        "Bonded": paired,
        "Trusted": trusted,
        "Connected": connected,
        "Adapter": HCI0,
        "UUIDs": list(uuids),
    }
    if name:
        properties["Name"] = name
    if rssi is not None:
        properties["RSSI"] = rssi
    return properties


def pi_objects() -> dict:
    """`GetManagedObjects` of a Pi as described above."""
    return {
        "/org/bluez": {"org.freedesktop.DBus.Introspectable": {}, "org.bluez.AgentManager1": {}},
        HCI0: {
            ADAPTER: {
                "Address": "DC:A6:32:AA:BB:CC",
                "Name": "raspberrypi",
                "Alias": "raspberrypi",
                "Powered": True,
                "Discoverable": False,
                "Pairable": True,
                "Discovering": False,
            },
            "org.bluez.Media1": {},
        },
        SONY: {
            DEVICE: _device(
                "F8:4E:17:01:02:03",
                "WH-1000XM4",
                "audio-headset",
                paired=True,
                trusted=True,
                connected=True,
                uuids=HEADPHONE_UUIDS,
            ),
            BATTERY: {"Percentage": bytes([80]), "Source": "AVRCP"},  # `y`, as PyQt6 gives it
            "org.bluez.MediaControl1": {},
        },
        SONY_TRANSPORT: {MEDIA_TRANSPORT: _transport(SONY)},
        JBL: {
            DEVICE: _device(
                "B8:F6:53:04:05:06", "JBL Flip 6", "audio-card", paired=True, uuids=SPEAKER_UUIDS
            )
        },
        PHONE: {
            DEVICE: _device("A0:B1:C2:0A:0B:0C", "Pixel 8", "phone", paired=True, uuids=PHONE_UUIDS)
        },
        BOSE: {
            DEVICE: _device(
                "2C:41:A1:07:08:09",
                "LE-Bose QC45",
                "audio-headset",
                rssi=-62,
                uuids=HEADPHONE_UUIDS,
            )
        },
        UNNAMED: {DEVICE: _device("00:11:22:33:44:55", "", "", rssi=-80)},
    }


def _transport(device: str, uuid: str = A2DP_SOURCE_UUID) -> dict:
    """The properties of a transport of BlueZ: the radio's A2DP stream to `device`."""
    return {"Device": device, "UUID": uuid, "State": "idle", "Codec": 0}


class FakeBus(Bus):
    """Answers at once, changing its objects as bluetoothd would and sending the signals
    bluetoothd sends; with `holding` the answers wait for `release`. An entry in
    `answers` ((interface, method) → `Reply` or a function of path and arguments)
    overrides the usual answer.

    A device taking audio gets the radio's A2DP stream (a transport) as it connects.
    With `one_stream` only while no other device holds it, as on the Pi: then Connect
    fails as BlueZ's does ("br-connection-create-socket"), and so does ConnectProfile.
    A device in `drops_without_a2dp` disconnects as its stream ends (a loudspeaker
    whose hands-free profile belongs to a phone)."""

    def __init__(self, objects: dict | None = None, connected: bool = True) -> None:
        super().__init__()
        self.objects = objects if objects is not None else pi_objects()
        self._connected = connected
        self.calls: list[tuple[str, str, str, list]] = []
        self.answers: dict[tuple[str, str], object] = {}
        self.holding = False
        self.held: list[tuple[tuple[str, str, str, list], Callable[[Reply], None]]] = []
        # Objects the manager offers on the bus, and the replies it sent their callers.
        self.exported: dict[str, object] = {}
        self.answered: list[tuple[object, list]] = []
        self.refused: list[tuple[object, str, str]] = []
        self.one_stream = False
        self.drops_without_a2dp: set[str] = set()
        self._transports = 1

    @property
    def connected(self) -> bool:
        return self._connected

    def call(self, path, interface, method, arguments, done) -> None:
        call = (path, interface, method, list(arguments))
        self.calls.append(call)
        if self.holding:
            self.held.append((call, done))
            return
        done(self.answer(*call))

    def release(self) -> None:
        """Answers the calls held back, as bluetoothd would after a while."""
        self.holding = False
        while self.held:
            call, done = self.held.pop(0)
            done(self.answer(*call))

    def register_object(self, path, target) -> bool:
        if not self._connected:
            return False
        self.exported[path] = target
        return True

    def reply(self, message, values) -> None:
        self.answered.append((message, list(values)))

    def refuse(self, message, error, text) -> None:
        self.refused.append((message, error, text))

    def calls_of(self, method: str) -> list[tuple[str, str, str, list]]:
        return [c for c in self.calls if c[2] == method]

    def answer(self, path, interface, method, arguments) -> Reply:
        answer = self.answers.get((interface, method))
        if callable(answer):
            answer = answer(path, arguments)
        if isinstance(answer, Reply):
            return answer
        if interface == OBJECT_MANAGER and method == "GetManagedObjects":
            return Reply(deepcopy(self.objects))
        if interface == PROPERTIES and method == "Set":
            target, name, value = arguments
            self._set(path, target, name, value)
            return Reply()
        if interface == ADAPTER:
            if method == "StartDiscovery":
                self._set(path, ADAPTER, "Discovering", True)
            elif method == "StopDiscovery":
                self._set(path, ADAPTER, "Discovering", False)
            elif method == "RemoveDevice":
                removed = str(arguments[0])
                self.objects.pop(removed, None)
                self.interfaces_removed.emit(removed, [DEVICE])
            return Reply()
        if interface == DEVICE:
            if method == "Pair":
                self._set(path, DEVICE, "Paired", True)
            elif method in ("Connect", "ConnectProfile"):
                if self._takes_audio(path) and not self._stream_free(path):
                    return Reply(None, "org.bluez.Error.Failed", "br-connection-create-socket")
                if not self.objects[path][DEVICE].get("Connected"):
                    self._set(path, DEVICE, "Connected", True)
                if self._takes_audio(path):
                    self.add_stream(path)
            elif method == "Disconnect":
                self.end_stream(path)
                self._set(path, DEVICE, "Connected", False)
            elif method == "DisconnectProfile":
                self.end_stream(path)
                if path in self.drops_without_a2dp:
                    self._set(path, DEVICE, "Connected", False)
            return Reply()
        return Reply()

    def stream_of(self, device: str) -> str | None:
        """The path of the transport of the radio's A2DP stream to `device`."""
        for path, interfaces in self.objects.items():
            transport = interfaces.get(MEDIA_TRANSPORT)
            if transport is not None and transport["Device"] == device:
                return path
        return None

    def add_stream(self, device: str) -> None:
        """Sets up the radio's A2DP stream to `device`, as BlueZ does (InterfacesAdded)."""
        if self.stream_of(device) is not None:
            return
        path = f"{device}/sep1/fd{self._transports}"
        self._transports += 1
        self.objects[path] = {MEDIA_TRANSPORT: _transport(device)}
        self.interfaces_added.emit(path, deepcopy(self.objects[path]))

    def end_stream(self, device: str) -> None:
        path = self.stream_of(device)
        if path is not None:
            del self.objects[path]
            self.interfaces_removed.emit(path, [MEDIA_TRANSPORT])

    def _takes_audio(self, device: str) -> bool:
        uuids = self.objects.get(device, {}).get(DEVICE, {}).get("UUIDs", [])
        return A2DP_SINK_UUID in (u.lower() for u in uuids)

    def _stream_free(self, device: str) -> bool:
        if not self.one_stream:
            return True
        holders = {
            t[MEDIA_TRANSPORT]["Device"] for t in self.objects.values() if MEDIA_TRANSPORT in t
        }
        return not holders - {device}

    def _set(self, path: str, interface: str, name: str, value: object) -> None:
        self.objects.setdefault(path, {}).setdefault(interface, {})[name] = value
        self.properties_changed.emit(path, interface, {name: value})
