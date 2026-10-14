"""USB storage for the recordings: sticks and disks found, mounted and ejected through
UDisks2, the disk service of Linux, over its D-Bus API.

UDisks2 (`udisksd`) watches the block devices of the system and publishes each as an
object on the system bus: `org.freedesktop.UDisks2.Block` for the device itself (what it
is, its label, the drive it sits on), `org.freedesktop.UDisks2.Filesystem` where it holds
a file system (mount points, `Mount`, `Unmount`) and `org.freedesktop.UDisks2.Drive` for
the drive (vendor, model, how it is connected, `PowerOff`). The radio takes only file
systems on drives connected over USB: the SD card the system runs from is to be spared
(the ring of recordings writes all the time), and nothing else has a place in a radio.

A file system that appears is mounted at once (`AUTO_MOUNT`), as a car radio would do,
to `/media/<user>/<label>`; "Eject" unmounts it and powers the drive off, so that it
can be pulled. polkit decides whether the radio may do that:
`pi/50-dab-radio-udisks2.rules` lets the members of the group plugdev.

`VolumeManager` keeps the state (the volumes, what is going on) and does what the
surface asks; it talks to the bus through a `Bus` (`net.dbus`), which the tests replace.
Nothing here waits for the bus, and UDisks2 reports every change through signals.

On the PC there is no system bus (Windows) or no UDisks2: `QStorageInfo` then lists what
the system has mounted, read anew every few seconds while the sub-page is shown, and
nothing can be mounted or ejected from the radio.
"""

from __future__ import annotations

import logging
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QObject, QStorageInfo, QTimer, pyqtSignal

from i18n import tr, tr_context
from net.dbus import (
    ERROR_ACCESS_DENIED,
    ERROR_NO_REPLY,
    ERROR_SERVICE_UNKNOWN,
    OBJECT_MANAGER,
    Bus,
    QtBus,
    Reply,
)

logger = logging.getLogger(__name__)

SERVICE = "org.freedesktop.UDisks2"
MANAGER_PATH = "/org/freedesktop/UDisks2"
BLOCK = "org.freedesktop.UDisks2.Block"
FILESYSTEM = "org.freedesktop.UDisks2.Filesystem"
DRIVE = "org.freedesktop.UDisks2.Drive"
# How a drive is connected (`Drive.ConnectionBus`); only these are taken.
USB = "usb"
# What a block device holds (`Block.IdUsage`) when it can be mounted.
USAGE_FILESYSTEM = "filesystem"

# A file system that appears is mounted without a tap.
AUTO_MOUNT = True
# Options of Mount, Unmount and PowerOff: no authentication agent is asked (the radio has
# none; polkit either allows the action or not, see `pi/50-dab-radio-udisks2.rules`).
_OPTIONS = {"auth.no_user_interaction": True}

# Without UDisks2 (the PC), the volumes the system mounted are read anew this often
# while the sub-page is shown; QStorageInfo reports no changes by itself.
POLL_INTERVAL_MS = 5000
# On Linux without UDisks2, only what is mounted under these counts as removable.
_MEDIA_PREFIXES = ("/media/", "/run/media/", "/mnt/")

# Errors of UDisks2 (the names of the D-Bus errors, udisks/udiskserror.h).
ERROR_NOT_AUTHORIZED = (
    "org.freedesktop.UDisks2.Error.NotAuthorized",
    "org.freedesktop.UDisks2.Error.NotAuthorizedCanObtain",
    "org.freedesktop.UDisks2.Error.NotAuthorizedDismissed",
)
ERROR_ALREADY_MOUNTED = "org.freedesktop.UDisks2.Error.AlreadyMounted"
ERROR_NOT_MOUNTED = "org.freedesktop.UDisks2.Error.NotMounted"
ERROR_DEVICE_BUSY = "org.freedesktop.UDisks2.Error.DeviceBusy"
ERROR_NOT_SUPPORTED = "org.freedesktop.UDisks2.Error.NotSupported"
ERROR_FAILED = "org.freedesktop.UDisks2.Error.Failed"


@dataclass(frozen=True)
class Volume:
    """A file system on a USB drive, mounted or not."""

    path: str  # the object path of the block device; on the PC the root path of the volume
    device: str  # "/dev/sda1"; on the PC "D:"
    label: str  # the label of the file system, "" without one
    drive: str  # vendor and model of the drive, "" where unknown
    filesystem: str  # "vfat", "exfat", "ntfs", "ext4" …
    size: int  # bytes
    mount_point: str = ""  # "" while not mounted
    read_only: bool = False
    drive_path: str = ""  # the object path of the drive, for `PowerOff`

    @property
    def mounted(self) -> bool:
        return bool(self.mount_point)

    @property
    def name(self) -> str:
        """What the list calls it: the label, else the drive, else the device."""
        return self.label or self.drive or self.device


def decode_bytes(value: object) -> str:
    """A string UDisks2 sends as bytes (`ay`, NUL-terminated: device names and mount
    points), as a string."""
    if isinstance(value, bytes | bytearray):
        return bytes(value).rstrip(b"\0").decode("utf-8", errors="replace")
    if isinstance(value, list):  # the array as a list of ints
        raw = bytes(int(item) & 0xFF for item in value)
        return raw.rstrip(b"\0").decode("utf-8", errors="replace")
    return str(value or "")


def mount_points(value: object) -> list[str]:
    """The mount points of a file system (`Filesystem.MountPoints`, `aay`)."""
    if not isinstance(value, list):
        return []
    return [point for point in (decode_bytes(item) for item in value) if point]


def is_wanted(block: dict, drive: dict) -> bool:
    """Whether a block device is a file system on a USB drive the radio may use."""
    return (
        str(drive.get("ConnectionBus", "")) == USB
        and str(block.get("IdUsage", "")) == USAGE_FILESYSTEM
        and not bool(block.get("HintIgnore", False))
    )


def volume_from_properties(path: str, block: dict, filesystem: dict, drive: dict) -> Volume:
    """A volume from the properties of its `Block`, `Filesystem` and `Drive`."""
    vendor = str(drive.get("Vendor", "")).strip()
    model = str(drive.get("Model", "")).strip()
    points = mount_points(filesystem.get("MountPoints"))
    return Volume(
        path=path,
        device=decode_bytes(block.get("PreferredDevice") or block.get("Device")),
        label=str(block.get("IdLabel", "")),
        drive=" ".join(part for part in (vendor, model) if part),
        filesystem=str(block.get("IdType", "")),
        size=int(block.get("Size", 0) or 0),
        mount_point=points[0] if points else "",
        read_only=bool(block.get("ReadOnly", False)),
        drive_path=str(block.get("Drive", "")),
    )


def parse_managed_objects(objects: dict) -> dict[str, Volume]:
    """The volumes by object path from `GetManagedObjects` of UDisks2 (or from the
    objects kept current from its signals)."""
    volumes: dict[str, Volume] = {}
    for path in sorted(objects):
        interfaces = objects[path]
        if BLOCK not in interfaces or FILESYSTEM not in interfaces:
            continue
        block = interfaces[BLOCK]
        drive = objects.get(str(block.get("Drive", "")), {}).get(DRIVE, {})
        if is_wanted(block, drive):
            volumes[path] = volume_from_properties(path, block, interfaces[FILESYSTEM], drive)
    return volumes


def sorted_volumes(volumes: dict[str, Volume]) -> list[Volume]:
    """Mounted first, then by name."""
    return sorted(volumes.values(), key=lambda v: (not v.mounted, v.name.casefold(), v.device))


def system_volumes() -> list[Volume]:
    """The volumes the system has mounted, where there is no UDisks2 to ask: on Windows
    every drive but the system's, on Linux what is mounted under /media, /run/media or
    /mnt. They count as mounted; nothing can be mounted or ejected through them."""
    volumes = []
    for info in QStorageInfo.mountedVolumes():
        if not info.isValid() or not info.isReady() or info.isRoot():
            continue
        root = info.rootPath()
        if sys.platform == "win32":
            device = root.rstrip("/")
        elif root.startswith(_MEDIA_PREFIXES):
            device = bytes(info.device()).decode("utf-8", errors="replace")
        else:
            continue
        volumes.append(
            Volume(
                path=root,
                device=device,
                label=info.name(),
                drive="",
                filesystem=bytes(info.fileSystemType()).decode("utf-8", errors="replace"),
                size=max(info.bytesTotal(), 0),
                mount_point=root,
                read_only=info.isReadOnly(),
            )
        )
    return volumes


def free_space(directory: Path | str) -> tuple[int, int] | None:
    """Free and total bytes of the file system `directory` lies on; None where it cannot
    be asked (the stick pulled)."""
    try:
        usage = shutil.disk_usage(directory)
    except OSError:
        return None
    return usage.free, usage.total


def size_text(size: int) -> str:
    """ "32 GB", "1.5 TB", "512 MB": in powers of 1000, as drives are sold."""
    for unit, factor in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6)):
        if size >= factor:
            value = size / factor
            text = f"{value:.1f}" if value < 10 else f"{value:.0f}"
            # Translators: the decimal separator in a size like "1.5 TB".
            separator = tr_context("decimal separator", ".")
            return f"{text.replace('.', separator)} {unit}"
    return f"{size / 1e3:.0f} kB"


def error_text(reply: Reply) -> str:
    """What went wrong, for the surface."""
    detail = reply.message.strip()
    lowered = detail.lower()
    if reply.error in ERROR_NOT_AUTHORIZED:
        return tr(
            "No permission: on the Pi, install the polkit rule from the folder pi/ and add "
            "the user to the group plugdev."
        )
    if reply.error == ERROR_SERVICE_UNKNOWN:
        return tr("UDisks2 is not running: on the Pi “sudo apt install udisks2”.")
    if reply.error == ERROR_ACCESS_DENIED:
        return tr("No permission on the system bus.")
    if reply.error in ERROR_NO_REPLY:
        return tr("Timeout: UDisks2 did not answer.")
    if reply.error == ERROR_ALREADY_MOUNTED:
        return tr("Already mounted.")
    if reply.error == ERROR_NOT_MOUNTED:
        return tr("Not mounted.")
    if reply.error == ERROR_DEVICE_BUSY:
        return tr("The drive is still in use.")
    if reply.error == ERROR_NOT_SUPPORTED or "unknown filesystem type" in lowered:
        return tr(
            "The system cannot mount this file system; the Pi may lack the driver "
            "(exfatprogs, ntfs-3g)."
        )
    return tr("UDisks2 reports: {detail}").format(detail=detail or reply.error)


class VolumeManager(QObject):
    """The USB volumes and what can be done with them.

    `watch` reads everything once; from then on UDisks2's signals keep the state
    current. `activity` tells what an action in progress does (mounting, ejecting),
    `changed` follows every change, `failed` what went wrong.
    """

    changed = pyqtSignal()
    failed = pyqtSignal(str)  # what went wrong, for the surface

    def __init__(
        self,
        parent: QObject | None = None,
        bus: Bus | None = None,
        system: Callable[[], list[Volume]] = system_volumes,
        auto_mount: bool = AUTO_MOUNT,
    ) -> None:
        super().__init__(parent)
        self._bus = bus if bus is not None else QtBus(SERVICE, self)
        self._system = system
        self._auto_mount = auto_mount
        self.service_present = False  # UDisks2 answered on the bus
        self.volumes: dict[str, Volume] = {}  # by path
        self.activity = ""  # e.g. "Mounting “USB-Stick” …"
        self._objects: dict = {}  # what UDisks2 publishes, kept current from its signals
        self._reading = False
        self._busy = 0  # actions in progress
        self._mount_tried: set[str] = set()  # paths mounted by itself, once each
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll)

        self._bus.interfaces_added.connect(self._on_interfaces_added)
        self._bus.interfaces_removed.connect(self._on_interfaces_removed)
        self._bus.properties_changed.connect(self._on_properties_changed)
        self._bus.service_changed.connect(self._on_service_changed)

    @property
    def available(self) -> bool:
        """Whether there is a system bus to ask (not on Windows)."""
        return self._bus.connected

    @property
    def can_mount(self) -> bool:
        """Whether UDisks2 is there to mount and eject; else the system's volumes are
        only listed."""
        return self.available and self.service_present

    @property
    def volume_list(self) -> list[Volume]:
        """The volumes for the list: mounted first, then by name."""
        return sorted_volumes(self.volumes)

    @property
    def mounted_volumes(self) -> list[Volume]:
        return [volume for volume in self.volume_list if volume.mounted]

    def volume(self, path: str) -> Volume | None:
        return self.volumes.get(path)

    def volume_of(self, directory: Path | str | None) -> Volume | None:
        """The mounted volume `directory` lies on; None where it lies on none."""
        if directory is None:
            return None
        target = Path(directory)
        best: Volume | None = None
        for volume in self.mounted_volumes:
            mount = Path(volume.mount_point)
            if target.is_relative_to(mount) and (
                best is None or len(mount.parts) > len(Path(best.mount_point).parts)
            ):
                best = volume
        return best

    def watch(self) -> None:
        """Reads the volumes once; UDisks2's signals keep them current from then on.
        Without UDisks2 the system's volumes are read, once."""
        if self.available:
            self._read()
        else:
            self._read_system()

    def start(self) -> None:
        """Reads anew and, without UDisks2, keeps reading the system's volumes while
        the sub-page is shown."""
        self.watch()
        self._poll_timer.start()

    def stop(self) -> None:
        self._poll_timer.stop()

    def shutdown(self) -> None:
        self.stop()

    # --- Reading ----------------------------------------------------------------------

    def _read(self) -> None:
        if self._reading:
            return
        self._reading = True
        self._bus.call(MANAGER_PATH, OBJECT_MANAGER, "GetManagedObjects", [], self._on_objects)

    def _on_objects(self, reply: Reply) -> None:
        self._reading = False
        if not reply.ok or not isinstance(reply.value, dict):
            if reply.ok:
                logger.warning("GetManagedObjects answered with %r, not a dict", type(reply.value))
            self.service_present = False
            self._objects = {}
            if not reply.ok and reply.error != ERROR_SERVICE_UNKNOWN:
                self.failed.emit(error_text(reply))
            self._read_system()
            self.changed.emit()
            return
        self.service_present = True
        self._objects = reply.value
        self._rebuild()

    def _read_system(self) -> None:
        """The volumes the system mounted, where UDisks2 is not there to ask."""
        volumes = {volume.path: volume for volume in self._system()}
        if volumes != self.volumes:
            self.volumes = volumes
            self.changed.emit()

    def _poll(self) -> None:
        if not self.service_present:
            self._read_system()

    def _rebuild(self) -> None:
        """The volumes from the objects, mounting those that are new and not mounted."""
        before = set(self.volumes)
        volumes = parse_managed_objects(self._objects)
        changed = volumes != self.volumes
        self.volumes = volumes
        for path in list(self._mount_tried):
            if path not in volumes:
                self._mount_tried.discard(path)  # plugged in again: mounted again
        if changed:
            self.changed.emit()
        if not self._auto_mount:
            return
        for path, volume in volumes.items():
            if path in before or volume.mounted or path in self._mount_tried:
                continue
            self._mount_tried.add(path)
            self.mount(volume)

    # --- Signals of UDisks2 -----------------------------------------------------------

    def _on_interfaces_added(self, path: str, interfaces: dict) -> None:
        self._objects.setdefault(path, {}).update(interfaces)
        self.service_present = True
        self._rebuild()

    def _on_interfaces_removed(self, path: str, names: list) -> None:
        if path not in self._objects:
            return
        for name in names:
            self._objects[path].pop(name, None)
        if not self._objects[path]:
            del self._objects[path]
        self._rebuild()

    def _on_properties_changed(self, path: str, interface: str, changed: dict) -> None:
        if path not in self._objects or interface not in self._objects[path]:
            return
        self._objects[path][interface].update(changed)
        self._rebuild()

    def _on_service_changed(self, running: bool) -> None:
        if running:
            self._read()
            return
        self.service_present = False
        self._objects = {}
        self.volumes = {}
        self.activity = ""
        self._busy = 0
        self._read_system()
        self.changed.emit()

    # --- Actions ----------------------------------------------------------------------

    def mount(self, volume: Volume) -> None:
        """Mounts the file system, to where UDisks2 puts it (`/media/<user>/<label>`)."""
        if not self.can_mount or volume.mounted:
            return
        self._begin(tr("Mounting “{name}” …").format(name=volume.name))
        self._bus.call(
            volume.path,
            FILESYSTEM,
            "Mount",
            [_OPTIONS],
            lambda reply: self._on_mounted(volume, reply),
        )

    def _on_mounted(self, volume: Volume, reply: Reply) -> None:
        if not reply.ok:
            if reply.error == ERROR_ALREADY_MOUNTED:
                self._end()
            else:
                self._fail(reply)
            return
        # UDisks2 reports the mount point through PropertiesChanged as well; the reply
        # is there first.
        if isinstance(reply.value, str) and reply.value and volume.path in self._objects:
            filesystem = self._objects[volume.path].setdefault(FILESYSTEM, {})
            filesystem["MountPoints"] = [reply.value.encode() + b"\0"]
        self._end()

    def eject(self, volume: Volume) -> None:
        """Unmounts the file system and powers the drive off, so that it can be pulled."""
        if not self.can_mount:
            return
        self._begin(tr("Ejecting “{name}” …").format(name=volume.name))
        if volume.mounted:
            self._bus.call(
                volume.path,
                FILESYSTEM,
                "Unmount",
                [_OPTIONS],
                lambda reply: self._on_unmounted(volume, reply),
            )
        else:
            self._power_off(volume)

    def _on_unmounted(self, volume: Volume, reply: Reply) -> None:
        if not reply.ok and reply.error != ERROR_NOT_MOUNTED:
            self._fail(reply)
            return
        if volume.path in self._objects:
            self._objects[volume.path].setdefault(FILESYSTEM, {})["MountPoints"] = []
        self._power_off(volume)

    def _power_off(self, volume: Volume) -> None:
        if not volume.drive_path or volume.drive_path not in self._objects:
            self._end()
            return
        self._bus.call(volume.drive_path, DRIVE, "PowerOff", [_OPTIONS], self._on_powered_off)

    def _on_powered_off(self, reply: Reply) -> None:
        # Unmounted is what matters; a drive that cannot be powered off (another file
        # system of it still mounted, no right to) may be pulled all the same.
        if not reply.ok:
            logger.info("PowerOff refused: %s: %s", reply.error, reply.message)
        self._end()

    def _begin(self, activity: str) -> None:
        self._busy += 1
        self.activity = activity
        self.changed.emit()

    def _end(self) -> None:
        self._busy = max(self._busy - 1, 0)
        if self._busy == 0:
            self.activity = ""
        self._rebuild()
        self.changed.emit()

    def _fail(self, reply: Reply) -> None:
        self.failed.emit(error_text(reply))
        self._end()
