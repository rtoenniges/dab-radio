"""A UDisks2 that answers from a table of objects, for `storage.volumes.VolumeManager`:
a Raspberry Pi with its SD card, a USB stick mounted, a second one not mounted, and the
whole-disk devices beside the partitions."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

from net.dbus import OBJECT_MANAGER, PROPERTIES, Bus, Reply
from storage.volumes import BLOCK, DRIVE, FILESYSTEM, MANAGER_PATH

DRIVES = f"{MANAGER_PATH}/drives"
BLOCKS = f"{MANAGER_PATH}/block_devices"
SANDISK = f"{DRIVES}/SanDisk_Ultra_4C530001"
INTENSO = f"{DRIVES}/Intenso_Speed_Line_1902"
SD_CARD = f"{DRIVES}/SC32G_0x12345678"
STICK = f"{BLOCKS}/sda1"
STICK_DISK = f"{BLOCKS}/sda"
INTENSO_STICK = f"{BLOCKS}/sdb1"
BOOT = f"{BLOCKS}/mmcblk0p1"
ROOT = f"{BLOCKS}/mmcblk0p2"
STICK_MOUNT = "/media/pi/USB-STICK"
# Where UDisks2 mounts a file system without a label.
UUID_MOUNT = "/media/pi/1902-ABCD"

MOUNT_OPTIONS = {"auth.no_user_interaction": True}


def _bytes(text: str) -> bytes:
    return text.encode() + b"\0"


def _drive(vendor: str, model: str, bus: str, removable: bool = True) -> dict:
    return {
        "Vendor": vendor,
        "Model": model,
        "ConnectionBus": bus,
        "Removable": removable,
        "Ejectable": removable,
        "Size": 31_000_000_000,
    }


def _block(device: str, drive: str, usage: str, kind: str, label: str, size: int, **hints) -> dict:
    return {
        "Device": _bytes(device),
        "PreferredDevice": _bytes(device),
        "Drive": drive,
        "IdUsage": usage,
        "IdType": kind,
        "IdLabel": label,
        "Size": size,
        "ReadOnly": False,
        "HintIgnore": hints.get("ignore", False),
        "HintSystem": hints.get("system", False),
    }


def pi_objects() -> dict:
    """`GetManagedObjects` of a Pi as described above."""
    return {
        f"{MANAGER_PATH}/Manager": {"org.freedesktop.UDisks2.Manager": {"Version": "2.10.1"}},
        SANDISK: {DRIVE: _drive("SanDisk", "Ultra", "usb")},
        INTENSO: {DRIVE: _drive("Intenso", "Speed Line", "usb")},
        SD_CARD: {DRIVE: _drive("", "SC32G", "sdio")},
        STICK_DISK: {
            BLOCK: _block("/dev/sda", SANDISK, "", "", "", 31_000_000_000),
            "org.freedesktop.UDisks2.PartitionTable": {"Type": "dos"},
        },
        STICK: {
            BLOCK: _block("/dev/sda1", SANDISK, "filesystem", "vfat", "USB-STICK", 30_990_000_000),
            "org.freedesktop.UDisks2.Partition": {"Number": 1},
            FILESYSTEM: {"MountPoints": [_bytes(STICK_MOUNT)], "Size": 30_990_000_000},
        },
        INTENSO_STICK: {
            BLOCK: _block("/dev/sdb1", INTENSO, "filesystem", "exfat", "", 64_000_000_000),
            FILESYSTEM: {"MountPoints": [], "Size": 64_000_000_000},
        },
        BOOT: {
            BLOCK: _block(
                "/dev/mmcblk0p1", SD_CARD, "filesystem", "vfat", "bootfs", 536_870_912, system=True
            ),
            FILESYSTEM: {"MountPoints": [_bytes("/boot/firmware")]},
        },
        ROOT: {
            BLOCK: _block(
                "/dev/mmcblk0p2",
                SD_CARD,
                "filesystem",
                "ext4",
                "rootfs",
                31_000_000_000,
                system=True,
            ),
            FILESYSTEM: {"MountPoints": [_bytes("/")]},
        },
    }


class FakeBus(Bus):
    """Answers at once, changing its objects as udisksd would and sending the signals
    udisksd sends; with `holding` the answers wait for `release`. An entry in `answers`
    ((interface, method) → `Reply` or a function of path and arguments) overrides the
    usual answer."""

    def __init__(self, objects: dict | None = None, connected: bool = True) -> None:
        super().__init__()
        self.objects = objects if objects is not None else pi_objects()
        self._connected = connected
        self.calls: list[tuple[str, str, str, list]] = []
        self.answers: dict[tuple[str, str], object] = {}
        self.holding = False
        self.held: list[tuple[tuple[str, str, str, list], Callable[[Reply], None]]] = []

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
        """Answers the calls held back, as udisksd would after a while."""
        self.holding = False
        while self.held:
            call, done = self.held.pop(0)
            done(self.answer(*call))

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
        if interface == FILESYSTEM and method == "Mount":
            label = self.objects[path][BLOCK].get("IdLabel") or UUID_MOUNT.rsplit("/", 1)[1]
            mount = f"/media/pi/{label}"
            self._set(path, FILESYSTEM, "MountPoints", [_bytes(mount)])
            return Reply(mount)
        if interface == FILESYSTEM and method == "Unmount":
            self._set(path, FILESYSTEM, "MountPoints", [])
            return Reply()
        if interface == DRIVE and method == "PowerOff":
            self.remove_drive(path)
            return Reply()
        return Reply()

    def plug_in(self, objects: dict) -> None:
        """A drive plugged in: its objects appear, as udisksd announces them."""
        for path, interfaces in objects.items():
            self.objects[path] = deepcopy(interfaces)
            self.interfaces_added.emit(path, deepcopy(interfaces))

    def remove_drive(self, drive: str) -> None:
        """The drive and its block devices go, as when it is pulled or powered off."""
        gone = [
            path
            for path, interfaces in self.objects.items()
            if path == drive or interfaces.get(BLOCK, {}).get("Drive") == drive
        ]
        for path in gone:
            names = list(self.objects.pop(path))
            self.interfaces_removed.emit(path, names)

    def _set(self, path: str, interface: str, name: str, value: object) -> None:
        self.objects.setdefault(path, {}).setdefault(interface, {})[name] = value
        self.properties_changed.emit(path, interface, {name: deepcopy(value)})
