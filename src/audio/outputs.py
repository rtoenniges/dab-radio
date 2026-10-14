"""The outputs of the system the radio can play to (`QMediaDevices`), as the surfaces
show them: the id Qt gives a device, its description, whether it is a Bluetooth device
and whether it is the system's default.

The default is what "Automatic" plays to (`ui.receiver_controller`): PipeWire on the
Raspberry Pi and Windows alike make headphones the default as soon as they connect and
go back to the loudspeaker when they are gone, so the radio need not keep that logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtMultimedia import QMediaDevices


@dataclass(frozen=True)
class AudioOutput:
    id: str  # `QAudioDevice.id` as text (a PipeWire node name, a Windows endpoint id)
    name: str  # the description of the device
    bluetooth: bool
    default: bool = False


def is_bluetooth(device_id: str, name: str) -> bool:
    """Whether an output is a Bluetooth device: PipeWire and PulseAudio name its sink
    "bluez_output.<address>.<n>" or "bluez_sink.<address>…"; Windows describes its
    endpoints as "Kopfhörer (<name> Stereo)" or "… Hands-Free)", the closing bracket
    included, which keeps PipeWire's "Built-in Audio Stereo" (a sound card) apart."""
    lowered = name.lower()
    return device_id.startswith("bluez") or lowered.endswith((" stereo)", " hands-free)"))


def audio_outputs() -> list[AudioOutput]:
    """The outputs of the system, in Qt's order, the default marked."""
    default = QMediaDevices.defaultAudioOutput()
    outputs = []
    for device in QMediaDevices.audioOutputs():
        device_id = bytes(device.id()).decode("utf-8", "replace")
        name = device.description()
        outputs.append(
            AudioOutput(device_id, name, is_bluetooth(device_id, name), device.id() == default.id())
        )
    return outputs
