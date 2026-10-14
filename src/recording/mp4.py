"""An MP4 audio file (`.m4a`, ISO/IEC 14496-12 and 14496-14) around the AAC access units
of a title, as they came off the air: no decoding, no re-encoding.

The container is what makes a DAB+ recording playable: its AudioSpecificConfig (the
`esds` box of the sample description) says that every access unit spans 960 core
samples and whether SBR is in it, which ADTS cannot (`recording.adts`). The same
configuration drives FAAD2 in the radio (`audio.aac.audio_specific_config`), so a
player with FAAD2 (VLC) decodes the file as the radio does; ffmpeg's own AAC decoder
does not take 960-sample frames (it reports the "960/120 MDCT window" as missing,
checked in its source, 09.10.2026).

One track, one chunk holding every unit in order, a sample table with their sizes and
the constant duration of 960 core samples, and the iTunes-style tags (`ilst`) for title,
artist, album (the station) and date, which every player reads.
"""

from __future__ import annotations

import struct
import time
from collections.abc import Iterable
from pathlib import Path

from audio.aac import audio_specific_config
from dsp.superframe import AudioFormat

# Core samples per DAB+ access unit (ETSI TS 102 563, clause 5.2).
SAMPLES_PER_UNIT = 960
# Seconds between the epoch of MP4 (1904) and that of Unix (1970).
_EPOCH_OFFSET = 2_082_844_800
_BRAND = b"M4A "
_COMPATIBLE_BRANDS = (b"M4A ", b"mp42", b"isom")
# MPEG-4 audio in an MPEG-4 elementary stream (ISO/IEC 14496-1, tables 5 and 6).
_OBJECT_TYPE_MPEG4_AUDIO = 0x40
_STREAM_TYPE_AUDIO = (0x05 << 2) | 0x01
_TAG_ES_DESCRIPTOR = 0x03
_TAG_DECODER_CONFIG = 0x04
_TAG_DECODER_SPECIFIC_INFO = 0x05
_TAG_SL_CONFIG = 0x06
# The unit matrix of the movie and track headers.
_UNITY_MATRIX = struct.pack(">9i", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)
# "und" packed as three 5-bit letters (ISO 639-2 in ISO/IEC 14496-12, clause 8.4.2.3).
_LANGUAGE_UNDETERMINED = 0x55C4

# The iTunes tags written, by the keys of `write_m4a`'s `tags`.
TAG_ATOMS = {
    "title": b"\xa9nam",
    "artist": b"\xa9ART",
    "album": b"\xa9alb",
    "date": b"\xa9day",
    "comment": b"\xa9cmt",
}


def box(kind: bytes, *payload: bytes) -> bytes:
    """A box: 32-bit size, four-letter type, content."""
    content = b"".join(payload)
    return struct.pack(">I4s", 8 + len(content), kind) + content


def full_box(kind: bytes, version: int, flags: int, *payload: bytes) -> bytes:
    """A box with version and flags in front of its content."""
    return box(kind, bytes((version,)) + flags.to_bytes(3, "big"), *payload)


def descriptor(tag: int, *payload: bytes) -> bytes:
    """A descriptor of the elementary stream (ISO/IEC 14496-1, clause 8.3.3): tag, the
    length on one byte (every one here is shorter than 128 bytes), content."""
    content = b"".join(payload)
    if len(content) >= 128:
        raise ValueError("descriptor too long for a one-byte length")
    return bytes((tag, len(content))) + content


def esds_box(config: bytes, bitrate: int) -> bytes:
    """The elementary stream descriptor with the AudioSpecificConfig."""
    decoder_config = descriptor(
        _TAG_DECODER_CONFIG,
        bytes((_OBJECT_TYPE_MPEG4_AUDIO, _STREAM_TYPE_AUDIO)),
        (0).to_bytes(3, "big"),  # buffer size
        struct.pack(">II", bitrate, bitrate),  # maximum and average bit rate
        descriptor(_TAG_DECODER_SPECIFIC_INFO, config),
    )
    es_descriptor = descriptor(
        _TAG_ES_DESCRIPTOR,
        struct.pack(">HB", 0, 0),  # ES_ID, no flags
        decoder_config,
        descriptor(_TAG_SL_CONFIG, b"\x02"),
    )
    return full_box(b"esds", 0, 0, es_descriptor)


def sample_entry(audio_format: AudioFormat, bitrate: int) -> bytes:
    """The `mp4a` sample description: the AAC core's channels and sample rate, and the
    `esds` with the configuration of the whole (SBR included)."""
    channels = 2 if audio_format.stereo_core else 1
    rate = audio_format.core_sample_rate
    return box(
        b"mp4a",
        bytes(6),  # reserved
        struct.pack(">H", 1),  # data reference index
        bytes(8),  # version, revision, vendor
        struct.pack(">HHHH", channels, 16, 0, 0),  # channels, sample size, compression, packet
        struct.pack(">I", rate << 16),  # sample rate, 16.16
        esds_box(audio_specific_config(audio_format), bitrate),
    )


def tag_box(atom: bytes, text: str) -> bytes:
    """One iTunes tag: the atom with a `data` box of UTF-8 text (type 1)."""
    return box(atom, full_box(b"data", 0, 1, bytes(4), text.encode("utf-8")))


def metadata_box(tags: dict[str, str]) -> bytes:
    """`udta` with the iTunes item list; empty tags are left out."""
    items = b"".join(
        tag_box(TAG_ATOMS[key], text) for key, text in tags.items() if key in TAG_ATOMS and text
    )
    handler = full_box(b"hdlr", 0, 0, bytes(4), b"mdir", b"appl", bytes(8), b"\0")
    return box(b"udta", full_box(b"meta", 0, 0, handler, box(b"ilst", items)))


def write_m4a(
    path: Path,
    units: Iterable[bytes],
    audio_format: AudioFormat,
    tags: dict[str, str] | None = None,
    bitrate: int = 0,
    created: float | None = None,
) -> int:
    """Writes the access units as an `.m4a` file; returns the duration in seconds.
    `bitrate` in bit/s goes into the stream descriptor (0: unknown), `created` is the
    time of the recording (Unix seconds, now by default)."""
    units = list(units)
    sizes = [len(unit) for unit in units]
    timescale = audio_format.core_sample_rate
    duration = len(units) * SAMPLES_PER_UNIT
    stamp = int((time.time() if created is None else created) + _EPOCH_OFFSET)
    tags = dict(tags or {})

    ftyp = box(b"ftyp", _BRAND, struct.pack(">I", 0), *_COMPATIBLE_BRANDS)
    mdat_header = struct.pack(">I4s", 8 + sum(sizes), b"mdat")
    data_offset = len(ftyp) + len(mdat_header)

    stbl = box(
        b"stbl",
        full_box(b"stsd", 0, 0, struct.pack(">I", 1), sample_entry(audio_format, bitrate)),
        full_box(b"stts", 0, 0, struct.pack(">III", 1, len(units), SAMPLES_PER_UNIT)),
        full_box(b"stsc", 0, 0, struct.pack(">IIII", 1, 1, len(units), 1)),
        full_box(
            b"stsz", 0, 0, struct.pack(">II", 0, len(units)), struct.pack(f">{len(units)}I", *sizes)
        ),
        full_box(b"stco", 0, 0, struct.pack(">II", 1, data_offset)),
    )
    minf = box(
        b"minf",
        full_box(b"smhd", 0, 0, bytes(4)),
        box(b"dinf", full_box(b"dref", 0, 0, struct.pack(">I", 1), full_box(b"url ", 0, 1))),
        stbl,
    )
    mdia = box(
        b"mdia",
        full_box(
            b"mdhd",
            0,
            0,
            struct.pack(">IIIIHH", stamp, stamp, timescale, duration, _LANGUAGE_UNDETERMINED, 0),
        ),
        full_box(b"hdlr", 0, 0, bytes(4), b"soun", bytes(12), b"SoundHandler\0"),
        minf,
    )
    tkhd = full_box(
        b"tkhd",
        0,
        3,  # enabled, in the movie
        struct.pack(">IIII", stamp, stamp, 1, 0),  # created, modified, track ID, reserved
        struct.pack(">I", duration),
        bytes(8),
        struct.pack(">HHHH", 0, 0, 0x0100, 0),  # layer, group, volume, reserved
        _UNITY_MATRIX,
        struct.pack(">II", 0, 0),  # width and height
    )
    mvhd = full_box(
        b"mvhd",
        0,
        0,
        struct.pack(">IIII", stamp, stamp, timescale, duration),
        struct.pack(">IH", 0x10000, 0x0100),  # rate, volume
        bytes(10),
        _UNITY_MATRIX,
        bytes(24),  # pre-defined
        struct.pack(">I", 2),  # next track ID
    )
    moov = box(b"moov", mvhd, box(b"trak", tkhd, mdia), metadata_box(tags))

    with open(path, "wb") as file:
        file.write(ftyp)
        file.write(mdat_header)
        file.writelines(units)
        file.write(moov)
    return duration / timescale


def read_boxes(data: bytes, start: int = 0, end: int | None = None) -> list[tuple[bytes, int, int]]:
    """The boxes in `data[start:end]` as (type, content start, content end); for the
    tests and for reading a file back."""
    end = len(data) if end is None else end
    boxes = []
    position = start
    while position + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, position)
        if size < 8 or position + size > end:
            break
        boxes.append((kind, position + 8, position + size))
        position += size
    return boxes
