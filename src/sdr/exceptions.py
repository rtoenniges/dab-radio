class DeviceError(RuntimeError):
    """Raised when a receiver cannot be accessed. The message is shown to the user."""


class RtlSdrError(DeviceError):
    """Raised when the RTL-SDR device cannot be accessed. The message is shown to the user."""


class SdrplayError(DeviceError):
    """Raised when the RSPduo cannot be accessed. The message is shown to the user."""
