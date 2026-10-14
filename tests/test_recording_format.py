"""The file formats of the recordings: the ADTS stream a title records as
(`recording.adts`) and the MP4 container a complete DAB+ title is packed into
(`recording.mp4`)."""

from __future__ import annotations

import struct

import pytest

from audio.aac import audio_specific_config
from dsp.superframe import AudioFormat
from recording.adts import (
    HEADER_LENGTH,
    MAX_UNIT_LENGTH,
    adts_header,
    read_adts_format,
    split_adts,
)
from recording.mp4 import SAMPLES_PER_UNIT, read_boxes, write_m4a

FORMAT_48K_SBR_STEREO = AudioFormat(
    dac_rate_48k=True, sbr=True, stereo_core=True, parametric_stereo=False, mpeg_surround_config=0
)
FORMAT_32K_PS = AudioFormat(
    dac_rate_48k=False, sbr=True, stereo_core=False, parametric_stereo=True, mpeg_surround_config=0
)
FORMAT_48K_LC = AudioFormat(
    dac_rate_48k=True, sbr=False, stereo_core=True, parametric_stereo=False, mpeg_surround_config=0
)


# --- ADTS -----------------------------------------------------------------------------


def test_the_header_names_the_core_and_the_length():
    header = adts_header(24_000, 2, 300)

    assert len(header) == HEADER_LENGTH
    assert header[0] == 0xFF and header[1] == 0xF1  # sync, MPEG-4, no CRC
    assert (header[2] >> 6) == 1  # AAC LC
    assert (header[2] >> 2) & 0xF == 6  # 24 kHz
    channels = ((header[2] & 1) << 2) | (header[3] >> 6)
    assert channels == 2
    length = ((header[3] & 3) << 11) | (header[4] << 3) | (header[5] >> 5)
    assert length == HEADER_LENGTH + 300
    assert header[5] & 0x1F == 0x1F and header[6] == 0xFC  # fullness 0x7FF, one block


def test_a_unit_too_long_for_the_header_is_refused():
    with pytest.raises(ValueError):
        adts_header(48_000, 2, MAX_UNIT_LENGTH + 1)


def test_the_stream_splits_back_into_its_units_and_a_cut_tail_is_left_out():
    units = [b"\x01\x02\x03", b"\xff\xf1" + bytes(10), b"", b"x" * 500]
    stream = b"".join(adts_header(48_000, 2, len(u)) + u for u in units)

    assert split_adts(stream) == units
    assert split_adts(b"junk" + stream) == units
    assert split_adts(stream[:-1]) == units[:-1]
    assert split_adts(b"") == []
    assert read_adts_format(stream) == (48_000, 2)
    assert read_adts_format(b"junk") is None


# --- MP4 ------------------------------------------------------------------------------


def _children(data: bytes, start: int, end: int) -> dict[bytes, tuple[int, int]]:
    return {kind: (a, b) for kind, a, b in read_boxes(data, start, end)}


def _find(data: bytes, path: list[bytes]) -> tuple[int, int]:
    """Content start and end of the box at `path` from the top."""
    start, end = 0, len(data)
    for kind in path:
        children = _children(data, start, end)
        start, end = children[kind]
        if kind == b"meta":
            start += 4  # a full box: version and flags before its children
    return start, end


@pytest.mark.parametrize("audio_format", [FORMAT_48K_SBR_STEREO, FORMAT_32K_PS, FORMAT_48K_LC])
def test_the_m4a_holds_the_units_the_configuration_and_the_tags(tmp_path, audio_format):
    units = [bytes([i]) * (10 + i) for i in range(30)]
    path = tmp_path / "title.m4a"

    duration = write_m4a(
        path,
        units,
        audio_format,
        {"title": "TAKE ON ME", "artist": "A-HA", "album": "Absolut relax", "date": "2026-10-09"},
        bitrate=96_000,
        created=1_791_000_000,
    )
    data = path.read_bytes()

    assert duration == pytest.approx(30 * SAMPLES_PER_UNIT / audio_format.core_sample_rate)
    top = [kind for kind, _a, _b in read_boxes(data)]
    assert top == [b"ftyp", b"mdat", b"moov"]
    mdat = _find(data, [b"mdat"])
    assert data[mdat[0] : mdat[1]] == b"".join(units)

    # The sample table: every unit's size, the constant duration, one chunk at the data.
    stbl = [b"moov", b"trak", b"mdia", b"minf", b"stbl"]
    stsz = _find(data, [*stbl, b"stsz"])
    count = struct.unpack_from(">I", data, stsz[0] + 8)[0]
    sizes = struct.unpack_from(f">{count}I", data, stsz[0] + 12)
    assert list(sizes) == [len(u) for u in units]
    stts = _find(data, [*stbl, b"stts"])
    assert struct.unpack_from(">III", data, stts[0] + 4) == (1, 30, SAMPLES_PER_UNIT)
    stco = _find(data, [*stbl, b"stco"])
    assert struct.unpack_from(">II", data, stco[0] + 4) == (1, mdat[0])

    # The time scale is the core rate, and the esds carries the radio's own
    # AudioSpecificConfig (960 samples per unit, SBR where there is).
    mdhd = _find(data, [b"moov", b"trak", b"mdia", b"mdhd"])
    timescale, total = struct.unpack_from(">II", data, mdhd[0] + 12)
    assert timescale == audio_format.core_sample_rate and total == 30 * SAMPLES_PER_UNIT
    stsd = _find(data, [*stbl, b"stsd"])
    mp4a = _children(data, stsd[0] + 8, stsd[1])[b"mp4a"]
    channels = struct.unpack_from(">H", data, mp4a[0] + 16)[0]
    assert channels == (2 if audio_format.stereo_core else 1)
    rate = struct.unpack_from(">I", data, mp4a[0] + 24)[0] >> 16
    assert rate == audio_format.core_sample_rate
    esds = _children(data, mp4a[0] + 28, mp4a[1])[b"esds"]
    assert audio_specific_config(audio_format) in data[esds[0] : esds[1]]

    # The tags, as iTunes writes them.
    ilst = _find(data, [b"moov", b"udta", b"meta", b"ilst"])
    items = _children(data, *ilst)
    assert set(items) == {b"\xa9nam", b"\xa9ART", b"\xa9alb", b"\xa9day"}
    name = items[b"\xa9nam"]
    assert data[name[0] : name[1]].endswith(b"TAKE ON ME")


def test_empty_tags_are_left_out_and_the_file_plays_without_any(tmp_path):
    path = tmp_path / "x.m4a"
    write_m4a(path, [b"a", b"b"], FORMAT_48K_LC, {"title": "", "artist": ""})
    data = path.read_bytes()
    ilst = _find(data, [b"moov", b"udta", b"meta", b"ilst"])
    assert _children(data, *ilst) == {}
