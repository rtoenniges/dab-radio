"""The system bus (D-Bus) of the Raspberry Pi, for the services the radio talks to there:
BlueZ for Bluetooth (`net.bluetooth`) and UDisks2 for USB storage (`storage.volumes`).
Both publish their objects through an ObjectManager and report every change through its
signals and `PropertiesChanged`, so nothing has to poll them.

`Bus` is what a manager needs of the bus; `QtBus` is the real one through QtDBus, bound
to one service, and the tests bring one that answers from a table. Nothing here waits for
the bus: every call is asynchronous and hands its reply to a callback.

PyQt6 demarshals replies and signals to Python objects by their signature (`QDBusArgument`
has no reading methods in PyQt6); `plain` unwraps what is left: variants, object paths
and byte arrays.

The radio can also offer an object of its own on the bus (`register_object`), whose
slots the service then calls, as BlueZ calls the agent of `net.bluetooth`; such a slot
takes the call with a delayed reply and answers it with `reply` or `refuse`.

On Windows there is no system bus: `QtBus.connected` is False, the calls only answer with
an error, and a manager says so or falls back to what the system offers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import QByteArray, QObject, pyqtSignal, pyqtSlot

try:
    from PyQt6.QtDBus import (
        QDBusConnection,
        QDBusMessage,
        QDBusObjectPath,
        QDBusPendingCallWatcher,
        QDBusPendingReply,
        QDBusServiceWatcher,
        QDBusVariant,
    )
except ImportError:  # a PyQt6 built without QtDBus
    QDBusConnection = None  # type: ignore[assignment,misc]
    QDBusMessage = object  # type: ignore[assignment,misc]
    # A type for the slots of an object offered on the bus (`net.bluetooth`), which is
    # then never registered.
    QDBusObjectPath = object  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

OBJECT_MANAGER = "org.freedesktop.DBus.ObjectManager"
PROPERTIES = "org.freedesktop.DBus.Properties"

# Errors of the bus itself (the names of the D-Bus errors); those of a service are its own.
ERROR_ACCESS_DENIED = "org.freedesktop.DBus.Error.AccessDenied"
ERROR_SERVICE_UNKNOWN = "org.freedesktop.DBus.Error.ServiceUnknown"
ERROR_NO_REPLY = ("org.freedesktop.DBus.Error.NoReply", "org.freedesktop.DBus.Error.Timeout")

# Connecting headphones or mounting a slow stick can take a good while; well within
# Qt's default of 25 s, which this sets explicitly.
CALL_TIMEOUT_MS = 30_000


@dataclass(frozen=True)
class Reply:
    """The answer to a call: its first return value, or the error."""

    value: object = None
    error: str = ""  # the name of the D-Bus error, "" on success
    message: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


Done = Callable[[Reply], None]


class Rejected(Exception):
    """An object the radio offers on the bus refuses a call: the caller gets the D-Bus
    error `error` with `text`."""

    def __init__(self, error: str, text: str = "") -> None:
        super().__init__(text or error)
        self.error = error
        self.text = text


def plain(value: object) -> object:
    """A reply of QtDBus as plain Python: variants unwrapped, object paths as strings,
    byte arrays as bytes, containers recursively."""
    if hasattr(value, "variant") and callable(value.variant):  # QDBusVariant
        return plain(value.variant())
    if hasattr(value, "path") and callable(value.path):  # QDBusObjectPath
        return value.path()
    if isinstance(value, QByteArray):
        return bytes(value)
    if isinstance(value, dict):
        return {plain(key): plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain(item) for item in value]
    return value


class Bus(QObject):
    """What a manager needs of the system bus for one service. `QtBus` is the real one;
    the tests bring one that answers from a table."""

    interfaces_added = pyqtSignal(str, object)  # path, {interface: {property: value}}
    interfaces_removed = pyqtSignal(str, object)  # path, [interface]
    properties_changed = pyqtSignal(str, str, object)  # path, interface, {property: value}
    service_changed = pyqtSignal(bool)  # the service appeared on the bus (True) or left it

    @property
    def connected(self) -> bool:
        """Whether there is a system bus (not on Windows)."""
        raise NotImplementedError

    def call(self, path: str, interface: str, method: str, arguments: list, done: Done) -> None:
        """Calls `method` of the service's object `path`; `done` gets the reply later."""
        raise NotImplementedError

    def set_property(self, path: str, interface: str, name: str, value: object, done: Done) -> None:
        self.call(path, PROPERTIES, "Set", [interface, name, self.variant(value)], done)

    def register_object(self, path: str, target: QObject) -> bool:
        """Offers the slots of `target` on the bus at `path`, under the interface its
        class info "D-Bus Interface" names; False where that is not possible."""
        raise NotImplementedError

    def reply(self, message: object, values: list) -> None:
        """Replies to a call taken with a delayed reply, with these return values."""
        raise NotImplementedError

    def refuse(self, message: object, error: str, text: str) -> None:
        """Replies to a call taken with a delayed reply with the D-Bus error `error`."""
        raise NotImplementedError

    @staticmethod
    def variant(value: object) -> object:
        """`value` as a D-Bus variant, where the API wants one."""
        return value

    @staticmethod
    def object_path(path: str) -> object:
        """`path` typed as a D-Bus object path, where the API wants one."""
        return path


class QtBus(Bus):
    """The system bus through QtDBus, for the objects of `service`. Replies and signals
    arrive as Python objects (PyQt6 demarshals them by their signature); `plain` unwraps
    what is left."""

    def __init__(self, service: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._service = service
        self._bus = QDBusConnection.systemBus() if QDBusConnection is not None else None
        self._watchers: set = set()  # pending calls, kept alive until they finish
        # Subscribed even where the bus is not connected (Windows): the calls then
        # only return False, and the tests on Windows run through this code.
        if self._bus is None:
            return
        bus = self._bus
        bus.connect(service, "", OBJECT_MANAGER, "InterfacesAdded", self._on_interfaces_added)
        bus.connect(service, "", OBJECT_MANAGER, "InterfacesRemoved", self._on_interfaces_removed)
        bus.connect(service, "", PROPERTIES, "PropertiesChanged", self._on_properties_changed)
        self._service_watcher = QDBusServiceWatcher(
            service, bus, QDBusServiceWatcher.WatchModeFlag.WatchForOwnerChange, self
        )
        self._service_watcher.serviceRegistered.connect(
            lambda _name: self.service_changed.emit(True)
        )
        self._service_watcher.serviceUnregistered.connect(
            lambda _name: self.service_changed.emit(False)
        )

    @property
    def service(self) -> str:
        return self._service

    @property
    def connected(self) -> bool:
        return self._bus is not None and self._bus.isConnected()

    def register_object(self, path: str, target: QObject) -> bool:
        if not self.connected:
            return False
        options = QDBusConnection.RegisterOption
        exported = options.ExportScriptableSlots | options.ExportNonScriptableSlots
        return bool(self._bus.registerObject(path, target, exported))

    def reply(self, message: object, values: list) -> None:
        if self._bus is not None:
            self._bus.send(message.createReply(list(values)))  # type: ignore[attr-defined]

    def refuse(self, message: object, error: str, text: str) -> None:
        if self._bus is not None:
            self._bus.send(message.createErrorReply(error, text))  # type: ignore[attr-defined]

    @staticmethod
    def variant(value: object) -> object:
        return QDBusVariant(value)

    @staticmethod
    def object_path(path: str) -> object:
        return QDBusObjectPath(path)

    def call(self, path: str, interface: str, method: str, arguments: list, done: Done) -> None:
        if self._bus is None:
            done(Reply(None, ERROR_SERVICE_UNKNOWN, "no system bus"))
            return
        message = QDBusMessage.createMethodCall(self._service, path, interface, method)
        message.setArguments(list(arguments))
        watcher = QDBusPendingCallWatcher(self._bus.asyncCall(message, CALL_TIMEOUT_MS), self)
        self._watchers.add(watcher)
        watcher.finished.connect(lambda _w, w=watcher, d=done: self._on_finished(w, d))

    def _on_finished(self, watcher: object, done: Done) -> None:
        self._watchers.discard(watcher)
        reply = QDBusPendingReply(watcher)
        if reply.isError():
            error = reply.error()
            result = Reply(None, error.name(), error.message())
            logger.debug("%s call failed: %s: %s", self._service, error.name(), error.message())
        else:
            arguments = reply.reply().arguments()
            result = Reply(plain(arguments[0]) if arguments else None)
        watcher.deleteLater()  # type: ignore[attr-defined]
        done(result)

    @pyqtSlot(QDBusMessage)
    def _on_interfaces_added(self, message: QDBusMessage) -> None:
        arguments = plain(message.arguments())
        if len(arguments) == 2:
            self.interfaces_added.emit(str(arguments[0]), arguments[1])

    @pyqtSlot(QDBusMessage)
    def _on_interfaces_removed(self, message: QDBusMessage) -> None:
        arguments = plain(message.arguments())
        if len(arguments) == 2:
            self.interfaces_removed.emit(str(arguments[0]), arguments[1])

    @pyqtSlot(QDBusMessage)
    def _on_properties_changed(self, message: QDBusMessage) -> None:
        arguments = plain(message.arguments())
        if len(arguments) >= 2:
            self.properties_changed.emit(message.path(), str(arguments[0]), arguments[1])
