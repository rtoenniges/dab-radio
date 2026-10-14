"""Reading a recording back for playback in the radio (`audio.player.AudioPlayer.play`):
the access units of an `.m4a` (`recording.mp4`) or the frames of an `.mp2`, with the
format the decoder needs, and the state of a playback for the surface.

The whole file goes into memory (a title of 4 min at 96 kbit/s is 2.9 MB, the longest
20 min at 192 kbit/s just under 30 MB): no handle stays open, so the drive may go and
the file may be renamed ("Keep") or deleted while it plays, and the audio thread
never waits for the stick. The recorder reads it (`recording.recorder.TitleRecorder.load`),
in its thread.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from audio.aac import audio_format_from_config
from dsp.mp2_frame import SAMPLES_PER_FRAME, Mp2Format
from dsp.superframe import AudioFormat
from recording.mp4 import SAMPLES_PER_UNIT, read_boxes

# The states of a playback (`PlaybackState.state`).
LOADING = "loading"  # the file is being read
PLAYING = "playing"
PAUSED = "paused"

# Descriptor tags of the elementary stream (ISO/IEC 14496-1, clause 8.3.3), as
# `recording.mp4.esds_box` writes them.
_TAG_ES_DESCRIPTOR = 0x03
_TAG_DECODER_CONFIG = 0x04
_TAG_DECODER_SPECIFIC_INFO = 0x05
# The fixed part of the `mp4a` sample entry before its child boxes (version 0).
_SAMPLE_ENTRY_LENGTH = 28
# The fixed part of the DecoderConfigDescriptor before its child descriptors.
_DECODER_CONFIG_LENGTH = 13


class PlaybackError(ValueError):
    """A file that cannot be read back as a recording."""


@dataclass(frozen=True)
class RecordingAudio:
    """A recording read for playback: its access units (DAB+) or frames (DAB), and the
    format the decoder is opened with."""

    recording_id: str
    audio_format: AudioFormat | Mp2Format
    units: tuple[bytes, ...]

    @property
    def unit_seconds(self) -> float:
        """How long one unit plays: 960 core samples of AAC or 1152 samples of MP2."""
        if isinstance(self.audio_format, Mp2Format):
            return SAMPLES_PER_FRAME / self.audio_format.sample_rate
        return SAMPLES_PER_UNIT / self.audio_format.core_sample_rate

    @property
    def duration(self) -> float:
        return len(self.units) * self.unit_seconds


@dataclass(frozen=True)
class PlaybackState:
    """What the player does with a recording, for the surface."""

    recording_id: str
    state: str  # LOADING, PLAYING or PAUSED
    position: float  # seconds into the title: what is heard right now
    duration: float


def read_mp2(path: Path, audio_format: Mp2Format) -> tuple[bytes, ...]:
    """The frames of a raw `.mp2` file: all of the same length, as DAB sends them at a
    constant bit rate; a last incomplete one is left out."""
    data = path.read_bytes()
    length = audio_format.frame_length
    if length <= 0:
        raise PlaybackError(f"no frame length for {audio_format}")
    return tuple(data[start : start + length] for start in range(0, len(data) - length + 1, length))


def read_m4a(path: Path) -> tuple[AudioFormat, tuple[bytes, ...]]:
    """The access units of an `.m4a` written by `recording.mp4.write_m4a` (one audio
    track, every unit a sample) and the DAB+ format from its AudioSpecificConfig.

    Raises:
        PlaybackError: for a file that is not such a recording.
    """
    data = path.read_bytes()
    moov = _child(data, b"moov", 0, len(data))
    stbl = moov
    for kind in (b"trak", b"mdia", b"minf", b"stbl"):
        stbl = _child(data, kind, *stbl)
    audio_format = _audio_format(data, _child(data, b"stsd", *stbl))
    sizes = _sample_sizes(data, _child(data, b"stsz", *stbl))
    offsets = _chunk_offsets(data, stbl)
    runs = _samples_per_chunk(data, _child(data, b"stsc", *stbl))

    units: list[bytes] = []
    for number, offset in enumerate(offsets, start=1):
        per_chunk = next((per for first, per in reversed(runs) if first <= number), 0)
        position = offset
        for _ in range(per_chunk):
            if len(units) >= len(sizes):
                break
            size = sizes[len(units)]
            if position + size > len(data):
                raise PlaybackError("a sample lies beyond the end of the file")
            units.append(data[position : position + size])
            position += size
    if len(units) != len(sizes):
        raise PlaybackError("the sample table and the chunks disagree")
    return audio_format, tuple(units)


def _child(data: bytes, kind: bytes, start: int, end: int) -> tuple[int, int]:
    """The content of the first box of `kind` among the boxes in `data[start:end]`."""
    for found, content_start, content_end in read_boxes(data, start, end):
        if found == kind:
            return content_start, content_end
    raise PlaybackError(f"no {kind.decode('latin-1')!r} box")


def _full_box_payload(start: int) -> int:
    """Where the content of a full box begins, after version and flags."""
    return start + 4


def _sample_sizes(data: bytes, stsz: tuple[int, int]) -> list[int]:
    position = _full_box_payload(stsz[0])
    sample_size, count = struct.unpack_from(">II", data, position)
    if sample_size:
        return [sample_size] * count
    position += 8
    if position + 4 * count > stsz[1]:
        raise PlaybackError("the sample sizes are cut short")
    return list(struct.unpack_from(f">{count}I", data, position))


def _chunk_offsets(data: bytes, stbl: tuple[int, int]) -> list[int]:
    for kind, width in ((b"stco", "I"), (b"co64", "Q")):
        try:
            start, end = _child(data, kind, *stbl)
        except PlaybackError:
            continue
        position = _full_box_payload(start)
        (count,) = struct.unpack_from(">I", data, position)
        position += 4
        if position + struct.calcsize(width) * count > end:
            raise PlaybackError("the chunk offsets are cut short")
        return list(struct.unpack_from(f">{count}{width}", data, position))
    raise PlaybackError("no chunk offsets")


def _samples_per_chunk(data: bytes, stsc: tuple[int, int]) -> list[tuple[int, int]]:
    """(first chunk, samples per chunk) of each run, in order."""
    position = _full_box_payload(stsc[0])
    (count,) = struct.unpack_from(">I", data, position)
    position += 4
    if position + 12 * count > stsc[1]:
        raise PlaybackError("the sample-to-chunk table is cut short")
    return [struct.unpack_from(">II", data, position + 12 * index)[:2] for index in range(count)]


def _audio_format(data: bytes, stsd: tuple[int, int]) -> AudioFormat:
    """The DAB+ format from the AudioSpecificConfig in the `esds` of the sample entry."""
    entries_start = _full_box_payload(stsd[0]) + 4  # after the entry count
    mp4a = _child(data, b"mp4a", entries_start, stsd[1])
    esds = _child(data, b"esds", mp4a[0] + _SAMPLE_ENTRY_LENGTH, mp4a[1])
    config = _decoder_specific_info(data, _full_box_payload(esds[0]), esds[1])
    try:
        return audio_format_from_config(config)
    except ValueError as error:
        raise PlaybackError(str(error)) from error


def _descriptor(data: bytes, position: int) -> tuple[int, int, int]:
    """A descriptor at `position`: tag, content start and end. The length takes one to
    four bytes, seven bits each, the high bit saying that another follows."""
    tag = data[position]
    position += 1
    length = 0
    for _ in range(4):
        byte = data[position]
        position += 1
        length = (length << 7) | (byte & 0x7F)
        if not byte & 0x80:
            break
    return tag, position, position + length


def _decoder_specific_info(data: bytes, start: int, end: int) -> bytes:
    tag, position, es_end = _descriptor(data, start)
    if tag != _TAG_ES_DESCRIPTOR:
        raise PlaybackError("no elementary stream descriptor")
    flags = data[position + 2]
    position += 3  # ES_ID and flags
    if flags & 0x80:
        position += 2  # dependsOn_ES_ID
    if flags & 0x40:
        position += 1 + data[position]  # URL
    if flags & 0x20:
        position += 2  # OCR_ES_ID
    while position < es_end:
        tag, content, content_end = _descriptor(data, position)
        if tag == _TAG_DECODER_CONFIG:
            inner = content + _DECODER_CONFIG_LENGTH
            while inner < content_end:
                tag, info, info_end = _descriptor(data, inner)
                if tag == _TAG_DECODER_SPECIFIC_INFO:
                    return data[info:info_end]
                inner = info_end
        position = content_end
    raise PlaybackError("no AudioSpecificConfig")
