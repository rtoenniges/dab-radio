"""State of the SDRplay API service, to tell why the SDRplay API cannot be opened.

On Windows the service control manager is asked, on Linux systemd.
"""

from __future__ import annotations

import ctypes
import enum
import subprocess
import sys
from ctypes import wintypes

SDRPLAY_SERVICE_NAME = "SDRplayAPIService"
# The unit the installer of API 3.15 for Linux sets up (`/opt/sdrplay_api/sdrplay_apiService`).
SDRPLAY_SYSTEMD_UNIT = "sdrplay.service"

_SC_MANAGER_CONNECT = 0x0001
_SERVICE_QUERY_STATUS = 0x0004
_ERROR_SERVICE_DOES_NOT_EXIST = 1060


class ServiceState(enum.Enum):
    NOT_INSTALLED = "not installed"
    STOPPED = "stopped"
    PENDING = "pending"  # starting, stopping, pausing or continuing
    RUNNING = "running"
    PAUSED = "paused"
    UNKNOWN = "unknown"  # the service manager could not be asked


# dwCurrentState of SERVICE_STATUS
_STATES = {
    1: ServiceState.STOPPED,
    2: ServiceState.PENDING,
    3: ServiceState.PENDING,
    4: ServiceState.RUNNING,
    5: ServiceState.PENDING,
    6: ServiceState.PENDING,
    7: ServiceState.PAUSED,
}


class _ServiceStatus(ctypes.Structure):
    _fields_ = [
        ("dwServiceType", wintypes.DWORD),
        ("dwCurrentState", wintypes.DWORD),
        ("dwControlsAccepted", wintypes.DWORD),
        ("dwWin32ExitCode", wintypes.DWORD),
        ("dwServiceSpecificExitCode", wintypes.DWORD),
        ("dwCheckPoint", wintypes.DWORD),
        ("dwWaitHint", wintypes.DWORD),
    ]


# ActiveState of a systemd unit
_SYSTEMD_STATES = {
    "active": ServiceState.RUNNING,
    "inactive": ServiceState.STOPPED,
    "failed": ServiceState.STOPPED,
    "activating": ServiceState.PENDING,
    "deactivating": ServiceState.PENDING,
    "reloading": ServiceState.PENDING,
    "refreshing": ServiceState.PENDING,
}


def query_service_state() -> ServiceState:
    """State of the SDRplay API service.

    Anything unexpected yields UNKNOWN rather than an exception: the result only refines
    an error message.
    """
    if sys.platform == "win32":
        return _query_windows_service(SDRPLAY_SERVICE_NAME)
    if sys.platform == "linux":
        return _query_systemd_unit(SDRPLAY_SYSTEMD_UNIT)
    return ServiceState.UNKNOWN


def _query_systemd_unit(unit: str) -> ServiceState:
    """Asks systemd for the state of a unit; needs no root rights."""
    try:
        output = subprocess.run(
            ["systemctl", "show", unit, "--property=LoadState", "--property=ActiveState"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except OSError, subprocess.SubprocessError:
        return ServiceState.UNKNOWN
    properties = dict(line.partition("=")[::2] for line in output.splitlines())
    if properties.get("LoadState") == "not-found":
        return ServiceState.NOT_INSTALLED
    return _SYSTEMD_STATES.get(properties.get("ActiveState", ""), ServiceState.UNKNOWN)


def _query_windows_service(name: str) -> ServiceState:
    """Asks the service control manager for the state of a service.

    Needs no administrator rights (SC_MANAGER_CONNECT and SERVICE_QUERY_STATUS are
    granted to every user).
    """
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    advapi32.OpenSCManagerW.restype = wintypes.HANDLE
    advapi32.OpenSCManagerW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
    advapi32.OpenServiceW.restype = wintypes.HANDLE
    advapi32.OpenServiceW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, wintypes.DWORD]
    advapi32.QueryServiceStatus.restype = wintypes.BOOL
    advapi32.QueryServiceStatus.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ServiceStatus)]
    advapi32.CloseServiceHandle.restype = wintypes.BOOL
    advapi32.CloseServiceHandle.argtypes = [wintypes.HANDLE]

    manager = advapi32.OpenSCManagerW(None, None, _SC_MANAGER_CONNECT)
    if not manager:
        return ServiceState.UNKNOWN
    try:
        service = advapi32.OpenServiceW(manager, name, _SERVICE_QUERY_STATUS)
        if not service:
            if ctypes.get_last_error() == _ERROR_SERVICE_DOES_NOT_EXIST:
                return ServiceState.NOT_INSTALLED
            return ServiceState.UNKNOWN
        try:
            status = _ServiceStatus()
            if not advapi32.QueryServiceStatus(service, ctypes.byref(status)):
                return ServiceState.UNKNOWN
            return _STATES.get(status.dwCurrentState, ServiceState.UNKNOWN)
        finally:
            advapi32.CloseServiceHandle(service)
    finally:
        advapi32.CloseServiceHandle(manager)
