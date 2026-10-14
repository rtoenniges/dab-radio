"""USB storage through UDisks2 without a bus (`storage.volumes`): what the objects of
udisksd become, the order of the list, the texts, and the manager on a UDisks2 that
answers from a table (`tests.fake_udisks`) or without one."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from net.dbus import ERROR_SERVICE_UNKNOWN, OBJECT_MANAGER, QtBus, Reply
from storage.volumes import (
    BLOCK,
    DRIVE,
    ERROR_DEVICE_BUSY,
    ERROR_FAILED,
    FILESYSTEM,
    MANAGER_PATH,
    SERVICE,
    Volume,
    VolumeManager,
    decode_bytes,
    error_text,
    parse_managed_objects,
    size_text,
    sorted_volumes,
    volume_from_properties,
)
from tests.fake_udisks import (
    INTENSO,
    INTENSO_STICK,
    MOUNT_OPTIONS,
    SANDISK,
    STICK,
    STICK_MOUNT,
    UUID_MOUNT,
    FakeBus,
    pi_objects,
)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


class _Recorder:
    def __init__(self, signal) -> None:
        self.calls: list[tuple] = []
        signal.connect(lambda *args: self.calls.append(args))


# --- Parsing --------------------------------------------------------------------------


def test_only_file_systems_on_usb_drives_are_taken():
    volumes = parse_managed_objects(pi_objects())

    assert set(volumes) == {STICK, INTENSO_STICK}  # not the SD card, not the whole disk
    stick = volumes[STICK]
    assert stick == Volume(
        STICK,
        "/dev/sda1",
        "USB-STICK",
        "SanDisk Ultra",
        "vfat",
        30_990_000_000,
        STICK_MOUNT,
        False,
        SANDISK,
    )
    assert stick.mounted and stick.name == "USB-STICK"
    intenso = volumes[INTENSO_STICK]
    assert not intenso.mounted and intenso.name == "Intenso Speed Line"
    assert intenso.filesystem == "exfat" and intenso.drive_path == INTENSO


def test_a_volume_without_label_or_drive_is_named_after_its_device():
    volume = volume_from_properties("/b", {"Device": b"/dev/sdc1\0"}, {}, {})
    assert volume.name == "/dev/sdc1" and volume.size == 0 and not volume.mounted


def test_bytes_of_udisks_become_strings():
    assert decode_bytes(b"/media/pi/X\0") == "/media/pi/X"
    assert decode_bytes([47, 109, 0]) == "/m"
    assert decode_bytes(None) == ""


def test_mounted_first_then_by_name():
    volumes = parse_managed_objects(pi_objects())
    volumes["/c"] = Volume("/c", "/dev/sdc1", "aaa", "", "vfat", 1, "", False, "/d")
    assert [v.name for v in sorted_volumes(volumes)] == ["USB-STICK", "aaa", "Intenso Speed Line"]


def test_sizes_read_as_drives_are_sold():
    assert size_text(31_000_000_000) == "31 GB"
    assert size_text(1_500_000_000_000) == "1.5 TB"
    assert size_text(512_000_000) == "512 MB"
    assert size_text(2_000) == "2 kB"


@pytest.mark.parametrize(
    ("reply", "start"),
    [
        (Reply(None, "org.freedesktop.UDisks2.Error.NotAuthorized", "x"), "No permission"),
        (Reply(None, ERROR_SERVICE_UNKNOWN, "not provided"), "UDisks2 is not running"),
        (Reply(None, "org.freedesktop.DBus.Error.NoReply", ""), "Timeout"),
        (Reply(None, ERROR_DEVICE_BUSY, "target is busy"), "The drive is still"),
        (Reply(None, ERROR_FAILED, "unknown filesystem type 'exfat'"), "The system cannot mount"),
        (Reply(None, ERROR_FAILED, "Something odd"), "UDisks2 reports: Something odd"),
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
    manager = VolumeManager(bus=bus)
    manager.watch()
    return manager


def test_watching_reads_the_volumes_and_mounts_the_one_not_mounted(manager, bus):
    assert manager.available and manager.service_present and manager.can_mount
    assert bus.calls[0] == (MANAGER_PATH, OBJECT_MANAGER, "GetManagedObjects", [])
    # The Intenso stick was not mounted: mounted at once, where UDisks2 puts it.
    assert bus.calls_of("Mount") == [(INTENSO_STICK, FILESYSTEM, "Mount", [MOUNT_OPTIONS])]
    assert [v.name for v in manager.volume_list] == ["Intenso Speed Line", "USB-STICK"]
    assert manager.volume(INTENSO_STICK).mount_point == UUID_MOUNT
    assert manager.activity == ""


def test_mounting_shows_an_activity_until_udisks_answers(bus):
    manager = VolumeManager(bus=bus, auto_mount=False)
    manager.watch()
    changes = _Recorder(manager.changed)
    bus.holding = True

    manager.mount(manager.volume(INTENSO_STICK))
    assert manager.activity == "Mounting “Intenso Speed Line” …"
    assert len(changes.calls) == 1

    bus.release()
    assert manager.activity == ""
    assert manager.volume(INTENSO_STICK).mounted
    assert len(changes.calls) >= 2


def test_a_failed_mount_is_reported_and_not_tried_again(bus):
    bus.answers[(FILESYSTEM, "Mount")] = Reply(None, ERROR_FAILED, "unknown filesystem type")
    manager = VolumeManager(bus=bus)
    failures = _Recorder(manager.failed)
    manager.watch()

    assert failures.calls[0][0].startswith("The system cannot mount")
    assert manager.activity == ""
    assert not manager.volume(INTENSO_STICK).mounted
    bus.properties_changed.emit(INTENSO_STICK, BLOCK, {"IdLabel": "INTENSO"})
    assert len(bus.calls_of("Mount")) == 1  # a change of a property mounts nothing


def test_without_auto_mount_a_tap_mounts(bus):
    manager = VolumeManager(bus=bus, auto_mount=False)
    manager.watch()
    assert bus.calls_of("Mount") == []

    manager.mount(manager.volume(INTENSO_STICK))
    assert bus.calls_of("Mount") == [(INTENSO_STICK, FILESYSTEM, "Mount", [MOUNT_OPTIONS])]
    manager.mount(manager.volume(INTENSO_STICK))  # mounted: nothing more
    assert len(bus.calls_of("Mount")) == 1


def test_ejecting_unmounts_and_powers_the_drive_off(manager, bus):
    stick = manager.volume(STICK)
    manager.eject(stick)

    assert bus.calls_of("Unmount") == [(STICK, FILESYSTEM, "Unmount", [MOUNT_OPTIONS])]
    assert bus.calls_of("PowerOff") == [(SANDISK, DRIVE, "PowerOff", [MOUNT_OPTIONS])]
    assert manager.volume(STICK) is None  # gone with the drive
    assert manager.activity == ""


def test_a_drive_that_cannot_be_powered_off_is_unmounted_all_the_same(manager, bus):
    bus.answers[(DRIVE, "PowerOff")] = Reply(None, ERROR_DEVICE_BUSY, "busy")
    failures = _Recorder(manager.failed)
    manager.eject(manager.volume(STICK))

    assert not manager.volume(STICK).mounted
    assert failures.calls == []
    assert manager.activity == ""


def test_a_stick_plugged_in_appears_and_is_mounted_and_goes_when_pulled(manager, bus):
    drive = f"{MANAGER_PATH}/drives/Kingston_DataTraveler_0815"
    block = f"{MANAGER_PATH}/block_devices/sdc1"
    objects = pi_objects()
    objects[drive] = {
        DRIVE: {"Vendor": "Kingston", "Model": "DataTraveler", "ConnectionBus": "usb"}
    }
    objects[block] = {
        BLOCK: {
            "Device": b"/dev/sdc1\0",
            "Drive": drive,
            "IdUsage": "filesystem",
            "IdType": "vfat",
            "IdLabel": "KINGSTON",
            "Size": 16_000_000_000,
        },
        FILESYSTEM: {"MountPoints": []},
    }
    bus.plug_in({drive: objects[drive], block: objects[block]})

    assert bus.calls_of("Mount")[-1] == (block, FILESYSTEM, "Mount", [MOUNT_OPTIONS])
    assert manager.volume(block).mount_point == "/media/pi/KINGSTON"

    bus.remove_drive(drive)
    assert manager.volume(block) is None

    # Plugged in again: mounted again.
    bus.plug_in({drive: objects[drive], block: objects[block]})
    assert len([c for c in bus.calls_of("Mount") if c[0] == block]) == 2


def test_the_volume_a_directory_lies_on(manager):
    assert manager.volume_of(Path(STICK_MOUNT) / "Aufnahmen" / "2026").path == STICK
    assert manager.volume_of(Path("/media/pi/USB-STICK")).path == STICK
    assert manager.volume_of(Path("/media/pi/OTHER/x")) is None
    assert manager.volume_of(None) is None


def test_without_a_system_bus_the_volumes_of_the_system_are_listed():
    listed = [Volume("D:/", "D:", "Stick", "", "exFAT", 64_000_000_000, "D:/")]
    bus = FakeBus(connected=False)
    manager = VolumeManager(bus=bus, system=lambda: list(listed))
    changes = _Recorder(manager.changed)
    manager.start()

    assert not manager.available and not manager.can_mount
    assert bus.calls == []
    assert [v.name for v in manager.volume_list] == ["Stick"]
    assert len(changes.calls) == 1
    assert manager._poll_timer.isActive()

    listed.clear()
    manager._poll_timer.timeout.emit()
    assert manager.volume_list == []
    manager.stop()
    assert not manager._poll_timer.isActive()
    manager.mount(Volume("D:/", "D:", "Stick", "", "exFAT", 1, ""))  # nothing to ask
    assert bus.calls == []


def test_without_udisks_the_service_is_marked_absent_and_the_system_asked(bus):
    bus.answers[(OBJECT_MANAGER, "GetManagedObjects")] = Reply(
        None, ERROR_SERVICE_UNKNOWN, "not provided"
    )
    system = [Volume("/media/pi/X", "/dev/sda1", "X", "", "vfat", 1, "/media/pi/X")]
    manager = VolumeManager(bus=bus, system=lambda: list(system))
    failures = _Recorder(manager.failed)
    manager.watch()

    assert manager.available and not manager.service_present and not manager.can_mount
    assert [v.name for v in manager.volume_list] == ["X"]
    assert failures.calls == []  # the page says so itself

    bus.answers.clear()
    bus.service_changed.emit(True)
    assert manager.service_present
    assert {v.path for v in manager.volume_list} == {STICK, INTENSO_STICK}

    bus.service_changed.emit(False)
    assert not manager.service_present and [v.name for v in manager.volume_list] == ["X"]


# --- QtDBus ---------------------------------------------------------------------------


def test_the_qt_bus_for_udisks_is_made_and_called_without_a_system_bus():
    """On the PC there is no system bus: subscribing and calling must still run through
    (the names of the QtDBus API are checked here). On the Pi this reaches udisksd."""
    bus = QtBus(SERVICE)
    replies = []
    assert bus.service == SERVICE and isinstance(bus.connected, bool)
    bus.call(MANAGER_PATH, OBJECT_MANAGER, "GetManagedObjects", [], replies.append)
    assert bus.variant(True) is not None and bus.object_path("/x") is not None
