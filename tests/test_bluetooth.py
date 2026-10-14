"""BlueZ over D-Bus without a bus (`net.bluetooth`): what the objects of bluetoothd
become, the order of the list, the texts of its errors, and the manager on a BlueZ that
answers from a table (`tests.fake_bluez`)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QObject
from PyQt6.QtDBus import QDBusMessage, QDBusObjectPath
from PyQt6.QtWidgets import QApplication

from net.bluetooth import (
    A2DP_SINK_UUID,
    A2DP_SOURCE_UUID,
    ADAPTER,
    AGENT,
    AGENT_CAPABILITY,
    AGENT_MANAGER,
    AGENT_PATH,
    BATTERY,
    DEVICE,
    DISCOVERY_TRANSPORT,
    ERROR_ALREADY_EXISTS,
    ERROR_FAILED,
    ERROR_REJECTED,
    HEADPHONES,
    MEDIA_TRANSPORT,
    OTHER,
    PHONE,
    SERVICE,
    SPEAKER,
    BluetoothAgent,
    BluetoothManager,
    Device,
    device_from_properties,
    error_text,
    parse_managed_objects,
    parse_transports,
    sorted_devices,
    transport_device,
    update_device,
)
from net.dbus import (
    ERROR_ACCESS_DENIED,
    ERROR_SERVICE_UNKNOWN,
    OBJECT_MANAGER,
    PROPERTIES,
    QtBus,
    Rejected,
    Reply,
    plain,
)
from tests.fake_bluez import (
    BOSE,
    HCI0,
    JBL,
    SONY,
    SONY_TRANSPORT,
    UNNAMED,
    FakeBus,
    pi_objects,
)
from tests.fake_bluez import (
    PHONE as PIXEL,
)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


class _Recorder:
    def __init__(self, signal) -> None:
        self.calls: list[tuple] = []
        signal.connect(lambda *args: self.calls.append(args))


# --- Parsing --------------------------------------------------------------------------


def test_the_adapter_and_the_devices_are_read_from_the_managed_objects():
    adapter, properties, devices = parse_managed_objects(pi_objects())

    assert adapter == HCI0
    assert properties["Powered"] is True
    sony = devices[SONY]
    assert sony == Device(
        SONY,
        "F8:4E:17:01:02:03",
        "WH-1000XM4",
        "audio-headset",
        paired=True,
        trusted=True,
        connected=True,
        battery=80,
        audio=True,
        a2dp=True,
    )
    assert sony.kind == HEADPHONES and sony.named
    assert devices[JBL].kind == SPEAKER
    assert devices[BOSE].rssi == -62 and devices[BOSE].kind == HEADPHONES
    assert devices[UNNAMED].kind == OTHER and not devices[UNNAMED].named


def test_devices_taking_audio_and_the_one_holding_the_radios_stream_are_marked():
    """Headphones and loudspeakers list the A2DP sink (in any case of the letters);
    the radio's stream to one is the transport with the source: one a phone plays to
    the radio over carries the sink and is not the radio's."""
    objects = pi_objects()
    objects[f"{PIXEL}/sep1/fd5"] = {MEDIA_TRANSPORT: {"Device": PIXEL, "UUID": A2DP_SINK_UUID}}
    _adapter, _properties, devices = parse_managed_objects(objects)

    assert [path for path, d in devices.items() if d.audio] == [BOSE, JBL, SONY]
    assert [path for path, d in devices.items() if d.a2dp] == [SONY]
    assert parse_transports(objects) == {SONY_TRANSPORT: SONY}
    assert transport_device({"Device": SONY, "UUID": A2DP_SOURCE_UUID.upper()}) == SONY
    assert transport_device({"Device": SONY, "UUID": A2DP_SINK_UUID}) is None
    assert update_device(devices[UNNAMED], DEVICE, {"UUIDs": [A2DP_SINK_UUID]}).audio


def test_without_an_adapter_there_is_none():
    objects = {"/org/bluez": {"org.bluez.AgentManager1": {}}}
    assert parse_managed_objects(objects) == (None, {}, {})


def test_a_device_without_alias_takes_its_name_then_its_address():
    device = device_from_properties("/d", {"Address": "AA:BB", "Name": "Buds"})
    assert device.name == "Buds"
    assert device_from_properties("/d", {"Address": "AA:BB"}).name == "AA:BB"
    assert device_from_properties("/d", {"Address": "AA:BB", "Icon": "phone"}).kind == PHONE


def test_connected_first_then_paired_then_the_seen_by_strength_and_unnamed_left_out():
    _adapter, _properties, devices = parse_managed_objects(pi_objects())
    devices["/x"] = Device("/x", "01:02", "Anker", "audio-card", rssi=-40)

    names = [d.name for d in sorted_devices(devices)]

    assert names == ["WH-1000XM4", "JBL Flip 6", "Pixel 8", "Anker", "LE-Bose QC45"]


def test_changed_properties_are_applied():
    device = Device("/d", "AA:BB", "AA:BB", "")

    named = update_device(device, DEVICE, {"Name": "Buds", "Icon": "audio-headset", "RSSI": -50})
    assert named.name == "Buds" and named.kind == HEADPHONES and named.rssi == -50
    renamed = update_device(named, DEVICE, {"Alias": "Meine Buds"})
    assert renamed.name == "Meine Buds"
    assert update_device(renamed, DEVICE, {"Name": "Other"}).name == "Meine Buds"
    connected = update_device(device, DEVICE, {"Connected": True, "Paired": True})
    assert connected.connected and connected.paired
    assert update_device(device, BATTERY, {"Percentage": 55}).battery == 55
    assert update_device(device, "org.bluez.MediaControl1", {"Connected": True}) is device


def test_the_battery_byte_of_pyqt6_is_read_as_a_number():
    # PyQt6 hands the D-Bus byte over as bytes of length 1: BlueZ said 50 % for the
    # Jabra Elite 10, int(b"2") made it 2 % (10.10.2026), and 100 % (b"d") was no value.
    from PyQt6.QtCore import QMetaType, QVariant

    byte = QVariant(50)
    assert byte.convert(QMetaType(QMetaType.Type.UChar.value))
    assert plain(byte.value()) == b"2"

    device = Device("/d", "AA:BB", "Buds", "")
    for percent in (0, 50, 100):
        changed = {"Percentage": bytes([percent])}
        assert update_device(device, BATTERY, changed).battery == percent
        assert device_from_properties("/d", {}, changed).battery == percent


def test_replies_of_qtdbus_become_plain_python():
    from PyQt6.QtDBus import QDBusObjectPath, QDBusVariant

    value = {
        QDBusObjectPath(HCI0): {ADAPTER: {"Powered": QDBusVariant(True), "Name": "pi"}},
        "list": [QDBusVariant(1), QDBusObjectPath("/x")],
    }

    assert plain(value) == {HCI0: {ADAPTER: {"Powered": True, "Name": "pi"}}, "list": [1, "/x"]}


@pytest.mark.parametrize(
    ("reply", "start"),
    [
        (
            Reply(None, "org.bluez.Error.AuthenticationCanceled", "Authentication Canceled"),
            "Pairing failed",
        ),
        (Reply(None, ERROR_ACCESS_DENIED, "Rejected send message"), "No permission"),
        (
            Reply(None, ERROR_SERVICE_UNKNOWN, "The name org.bluez was not provided"),
            "The Bluetooth",
        ),
        (Reply(None, "org.freedesktop.DBus.Error.NoReply", "Did not receive a reply"), "Timeout"),
        (Reply(None, "org.bluez.Error.NotReady", "Resource Not Ready"), "Bluetooth is off"),
        (Reply(None, ERROR_FAILED, "br-connection-profile-unavailable"), "No audio profile"),
        (Reply(None, ERROR_FAILED, "br-connection-page-timeout"), "The device does not answer"),
        (Reply(None, ERROR_FAILED, "br-connection-refused"), "The device refused"),
        (Reply(None, ERROR_FAILED, "br-connection-create-socket"), "No audio connection free"),
        (Reply(None, ERROR_FAILED, "Something odd"), "BlueZ reports: Something odd"),
        (Reply(None, "org.bluez.Error.Odd", ""), "BlueZ reports: org.bluez.Error.Odd"),
    ],
)
def test_errors_are_explained_to_the_user(reply, start):
    assert error_text(reply).startswith(start)


# --- The manager ----------------------------------------------------------------------


@pytest.fixture
def bus() -> FakeBus:
    return FakeBus()


@pytest.fixture
def manager(bus):
    manager = BluetoothManager(bus=bus)
    manager.start()
    return manager


def test_starting_reads_adapter_and_devices(manager, bus):
    assert manager.available and manager.service_present
    assert manager.adapter == HCI0 and manager.powered and not manager.discovering
    assert [d.name for d in manager.device_list] == [
        "WH-1000XM4",
        "JBL Flip 6",
        "Pixel 8",
        "LE-Bose QC45",
    ]
    assert manager.connected_device.name == "WH-1000XM4"
    assert manager.connected_device.battery == 80
    assert bus.calls[0] == ("/", OBJECT_MANAGER, "GetManagedObjects", [])


def test_without_a_system_bus_nothing_is_asked():
    bus = FakeBus(connected=False)
    manager = BluetoothManager(bus=bus)
    manager.start()
    assert not manager.available
    assert bus.calls == []
    assert bus.exported == {} and not manager.agent_registered


def test_without_bluetoothd_the_service_is_marked_absent(bus):
    bus.answers[(OBJECT_MANAGER, "GetManagedObjects")] = Reply(
        None, ERROR_SERVICE_UNKNOWN, "not provided"
    )
    manager = BluetoothManager(bus=bus)
    failures = _Recorder(manager.failed)
    manager.start()

    assert manager.available and not manager.service_present
    assert manager.adapter is None and manager.device_list == []
    assert failures.calls == []  # the page says so itself

    bus.answers.clear()
    bus.service_changed.emit(True)
    assert manager.service_present and manager.adapter == HCI0


def test_pairing_trusts_and_connects(manager, bus):
    changes = _Recorder(manager.changed)
    bose = manager.device(BOSE)
    bus.holding = True

    manager.pair(bose)
    assert manager.activity == "Pairing with “LE-Bose QC45” …"
    assert bus.calls_of("Pair") == [(BOSE, DEVICE, "Pair", [])]
    assert manager.pair(bose) is None and len(bus.calls_of("Pair")) == 1  # one at a time

    bus.release()
    assert bus.calls_of("Set") == [(BOSE, PROPERTIES, "Set", [DEVICE, "Trusted", True])]
    assert bus.calls_of("Connect") == [(BOSE, DEVICE, "Connect", [])]
    assert manager.activity == ""
    assert manager.device(BOSE).paired and manager.device(BOSE).connected
    assert len(changes.calls) >= 3


def test_an_already_paired_device_is_connected_all_the_same(manager, bus):
    bus.answers[(DEVICE, "Pair")] = Reply(None, ERROR_ALREADY_EXISTS, "Already Exists")
    manager.pair(manager.device(BOSE))
    assert bus.calls_of("Connect") == [(BOSE, DEVICE, "Connect", [])]
    assert manager.activity == ""


def test_a_refused_pairing_is_reported_and_ends_the_activity(manager, bus):
    bus.answers[(DEVICE, "Pair")] = Reply(
        None, "org.bluez.Error.AuthenticationFailed", "Authentication Failed"
    )
    failures = _Recorder(manager.failed)
    manager.pair(manager.device(BOSE))

    assert manager.activity == ""
    assert failures.calls[0][0].startswith("Pairing failed")
    assert bus.calls_of("Connect") == []


def test_connecting_disconnecting_and_forgetting(manager, bus):
    jbl = manager.device(JBL)
    manager.connect_device(jbl)
    assert bus.calls_of("Connect") == [(JBL, DEVICE, "Connect", [])]
    assert manager.device(JBL).connected

    manager.disconnect_device(manager.device(JBL))
    assert bus.calls_of("Disconnect") == [(JBL, DEVICE, "Disconnect", [])]
    assert not manager.device(JBL).connected

    manager.forget(manager.device(JBL))
    assert bus.calls_of("RemoveDevice") == [(HCI0, ADAPTER, "RemoveDevice", [JBL])]
    assert manager.device(JBL) is None
    assert manager.activity == ""


def test_a_scan_filters_classic_bluetooth_and_ends_by_itself(manager, bus, monkeypatch):
    manager.scan()

    assert bus.calls_of("SetDiscoveryFilter") == [
        (HCI0, ADAPTER, "SetDiscoveryFilter", [{"Transport": DISCOVERY_TRANSPORT}])
    ]
    assert bus.calls_of("StartDiscovery") == [(HCI0, ADAPTER, "StartDiscovery", [])]
    assert manager.discovering and manager._discovery_timer.isActive()
    manager.scan()  # already discovering: nothing more
    assert len(bus.calls_of("StartDiscovery")) == 1

    manager._discovery_timer.timeout.emit()
    assert bus.calls_of("StopDiscovery") == [(HCI0, ADAPTER, "StopDiscovery", [])]
    assert not manager.discovering and not manager._discovery_timer.isActive()


def test_leaving_the_page_ends_a_discovery(manager, bus):
    manager.scan()
    manager.stop()
    assert bus.calls_of("StopDiscovery") == [(HCI0, ADAPTER, "StopDiscovery", [])]


def test_devices_seen_and_gone_follow_the_signals_of_bluez(manager, bus):
    bus.interfaces_added.emit(
        "/org/bluez/hci0/dev_11_22_33_44_55_66",
        {DEVICE: {"Address": "11:22:33:44:55:66", "Alias": "Buds", "Icon": "audio-headset"}},
    )
    assert "Buds" in [d.name for d in manager.device_list]

    bus.interfaces_added.emit(
        "/org/bluez/hci0/dev_11_22_33_44_55_66", {BATTERY: {"Percentage": bytes([42])}}
    )
    assert manager.device("/org/bluez/hci0/dev_11_22_33_44_55_66").battery == 42

    bus.properties_changed.emit(SONY, DEVICE, {"Connected": False})
    assert manager.connected_device is None

    bus.interfaces_removed.emit("/org/bluez/hci0/dev_11_22_33_44_55_66", [DEVICE])
    assert "Buds" not in [d.name for d in manager.device_list]

    bus.properties_changed.emit(HCI0, ADAPTER, {"Powered": False})
    assert not manager.powered

    bus.service_changed.emit(False)
    assert not manager.service_present and manager.adapter is None and manager.devices == {}


def test_switching_the_adapter_sets_its_property(manager, bus):
    manager.set_powered(False)
    assert bus.calls_of("Set") == [(HCI0, PROPERTIES, "Set", [ADAPTER, "Powered", False])]
    assert not manager.powered  # from the signal the fake sends


# --- QtDBus ---------------------------------------------------------------------------


def test_the_qt_bus_is_made_and_called_without_a_system_bus():
    """On the PC there is no system bus: subscribing and calling must still run
    through, so that the names of the QtDBus API are checked here (the enum of the
    service watcher once failed only on the Pi). On the Pi this reaches bluetoothd."""
    bus = QtBus(SERVICE)
    assert isinstance(bus.connected, bool)
    bus.call("/", OBJECT_MANAGER, "GetManagedObjects", [], lambda _reply: None)
    assert isinstance(bus.variant(True), object) and bus.object_path("/x") is not None
    assert isinstance(bus.register_object("/org/dabradio/test", QObject()), bool)
    message = QDBusMessage.createMethodCall(SERVICE, "/x", AGENT, "Cancel")
    bus.reply(message, [])
    bus.refuse(message, ERROR_REJECTED, "no")


# --- The agent ----------------------------------------------------------------------------


def test_starting_offers_the_agent_and_makes_it_the_default(manager, bus):
    assert isinstance(bus.exported[AGENT_PATH], BluetoothAgent)
    assert bus.calls_of("RegisterAgent") == [
        ("/org/bluez", AGENT_MANAGER, "RegisterAgent", [AGENT_PATH, "NoInputNoOutput"])
    ]
    assert bus.calls_of("RequestDefaultAgent") == [
        ("/org/bluez", AGENT_MANAGER, "RequestDefaultAgent", [AGENT_PATH])
    ]
    assert manager.agent_registered and AGENT_CAPABILITY == "NoInputNoOutput"

    manager.start()  # the page shown: no second registration
    assert len(bus.calls_of("RegisterAgent")) == 1


def test_bluez_coming_back_gets_the_agent_again(manager, bus):
    bus.service_changed.emit(False)
    assert not manager.agent_registered
    bus.service_changed.emit(True)
    assert len(bus.calls_of("RegisterAgent")) == 2 and manager.agent_registered
    assert list(bus.exported) == [AGENT_PATH]  # offered on the bus once


def test_an_agent_bluez_refuses_is_not_made_the_default_and_tried_again_later(bus):
    bus.answers[(AGENT_MANAGER, "RegisterAgent")] = Reply(None, ERROR_FAILED, "Failed")
    manager = BluetoothManager(bus=bus)
    manager.start()
    assert not manager.agent_registered and bus.calls_of("RequestDefaultAgent") == []

    # BlueZ still knows it (registered before): made the default all the same.
    bus.answers[(AGENT_MANAGER, "RegisterAgent")] = Reply(None, ERROR_ALREADY_EXISTS, "Exists")
    manager.start()
    assert len(bus.calls_of("RequestDefaultAgent")) == 1 and manager.agent_registered


def test_the_agents_slots_carry_the_signatures_bluez_calls(bus):
    meta = BluetoothAgent(bus, lambda _method, _arguments: []).metaObject()
    slots = {
        bytes(meta.method(i).methodSignature()).decode(): meta.method(i).typeName()
        for i in range(meta.methodOffset(), meta.methodCount())
    }
    # org.bluez.Agent1 (BlueZ doc/org.bluez.Agent.rst): o is the device, u a uint32, q a
    # uint16; the final QDBusMessage is the call itself and not part of the signature.
    assert slots == {
        "Release(QDBusMessage)": "void",
        "RequestPinCode(QDBusObjectPath,QDBusMessage)": "QString",
        "DisplayPinCode(QDBusObjectPath,QString,QDBusMessage)": "void",
        "RequestPasskey(QDBusObjectPath,QDBusMessage)": "uint",
        "DisplayPasskey(QDBusObjectPath,uint,ushort,QDBusMessage)": "void",
        "RequestConfirmation(QDBusObjectPath,uint,QDBusMessage)": "void",
        "RequestAuthorization(QDBusObjectPath,QDBusMessage)": "void",
        "AuthorizeService(QDBusObjectPath,QString,QDBusMessage)": "void",
        "Cancel(QDBusMessage)": "void",
    }
    assert meta.classInfo(meta.indexOfClassInfo("D-Bus Interface")).value() == AGENT


def test_the_agent_says_yes_only_to_the_device_being_paired(manager, bus):
    agent = bus.exported[AGENT_PATH]
    a2dp = "0000110d-0000-1000-8000-00805f9b34fb"

    def ask(method: str, *arguments) -> list | None:
        """Calls the slot as Qt would; the values replied, None for a refusal."""
        message = QDBusMessage.createMethodCall(SERVICE, AGENT_PATH, AGENT, method)
        answered, refused = len(bus.answered), len(bus.refused)
        getattr(agent, method)(*arguments, message)
        assert message.isDelayedReply()
        if len(bus.answered) > answered:
            return bus.answered[-1][1]
        assert len(bus.refused) == refused + 1 and bus.refused[-1][1] == ERROR_REJECTED
        return None

    bose, sony = QDBusObjectPath(BOSE), QDBusObjectPath(SONY)
    # Nobody is being paired: a device asking on its own gets nothing.
    assert ask("RequestAuthorization", bose) is None
    assert ask("RequestConfirmation", bose, 123456) is None
    assert ask("AuthorizeService", bose, a2dp) is None
    assert ask("RequestPinCode", bose) is None

    bus.holding = True
    manager.pair(manager.device(BOSE))
    assert ask("RequestConfirmation", bose, 123456) == []
    assert ask("RequestAuthorization", bose) == []
    assert ask("AuthorizeService", bose, a2dp) == []
    assert ask("RequestPinCode", bose) == ["0000"]
    assert ask("RequestPasskey", bose) is None  # no keyboard to type one
    assert ask("RequestAuthorization", sony) is None  # another device meanwhile
    assert ask("AuthorizeService", sony, a2dp) is None

    bus.release()
    assert manager.activity == "" and manager.device(BOSE).connected
    assert ask("RequestAuthorization", bose) is None  # paired and connected: done
    assert ask("DisplayPasskey", bose, 123456, 2) == []
    assert ask("DisplayPinCode", bose, "1234") == []
    assert ask("Cancel") == []
    assert ask("Release") == [] and not manager.agent_registered


def test_a_failed_pairing_ends_the_yes_of_the_agent(manager, bus):
    bus.answers[(DEVICE, "Pair")] = Reply(
        None, "org.bluez.Error.AuthenticationFailed", "Authentication Failed"
    )
    manager.pair(manager.device(BOSE))
    with pytest.raises(Rejected):
        manager.answer_agent("RequestAuthorization", [BOSE])


# --- The radio's A2DP stream ------------------------------------------------------------


def test_the_stream_and_the_order_of_connecting_follow_bluez(manager, bus):
    """The device holding the stream counts as connected last at the start, so that the
    radio plays on where it plays; every connection after it is newer."""
    connected = _Recorder(manager.device_connected)
    assert manager.a2dp_device.path == SONY
    assert manager.newest_audio_device.path == SONY

    manager.connect_device(manager.device(JBL))  # a second stream: not on the Pi
    assert connected.calls == [(JBL,)]
    assert manager.newest_audio_device.path == JBL
    assert manager.device(JBL).a2dp

    manager.disconnect_device(manager.device(JBL))
    assert not manager.device(JBL).a2dp
    assert manager.newest_audio_device.path == SONY
    bus.end_stream(SONY)
    assert manager.a2dp_device is None and not manager.device(SONY).a2dp
    bus.add_stream(SONY)
    assert manager.a2dp_device.path == SONY


def test_a_phone_connecting_is_no_audio_device(manager, bus):
    manager.connect_device(manager.device(PIXEL))
    assert manager.device(PIXEL).connected
    assert manager.newest_audio_device.path == SONY


def test_the_stream_is_ended_and_set_up_by_profile(manager, bus):
    """On the Pi a second device gets no stream while one holds it (`one_stream`)."""
    bus.one_stream = True
    results = []
    jbl = manager.device(JBL)
    failures = _Recorder(manager.failed)

    assert manager.connect_device(jbl)
    assert not manager.device(JBL).connected
    assert failures.calls[-1][0].startswith("No audio connection free")

    manager.release_a2dp(manager.device(SONY), results.append)
    assert bus.calls_of("DisconnectProfile") == [
        (SONY, DEVICE, "DisconnectProfile", [A2DP_SINK_UUID])
    ]
    assert results[-1].ok and manager.a2dp_device is None
    assert manager.device(SONY).connected  # over the hands-free profile still

    manager.connect_a2dp(manager.device(SONY), results.append)
    assert bus.calls_of("ConnectProfile") == [(SONY, DEVICE, "ConnectProfile", [A2DP_SINK_UUID])]
    assert results[-1].ok and manager.a2dp_device.path == SONY


def test_connecting_waits_for_an_action_in_progress(manager, bus):
    bus.holding = True
    assert manager.connect_device(manager.device(JBL))
    assert not manager.connect_device(manager.device(BOSE))
    assert len(bus.calls_of("Connect")) == 1


def test_the_device_found_by_address_and_bluez_gone_forget_the_stream(manager, bus):
    assert manager.device_by_address("f8:4e:17:01:02:03").path == SONY
    assert manager.device_by_address("00:00:00:00:00:00") is None
    bus.service_changed.emit(False)
    assert manager.a2dp_device is None and manager.newest_audio_device is None
