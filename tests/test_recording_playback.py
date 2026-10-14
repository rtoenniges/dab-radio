"""Reading a recording back for playback (`recording.playback`): the access units of an
`.m4a` as `recording.mp4.write_m4a` wrote them, with the format from the
AudioSpecificConfig, the frames of an `.mp2`, and what is not such a file."""

from __future__ import annotations

from dataclasses import replace

import pytest

from audio.aac import audio_format_from_config, audio_specific_config
from dsp.mp2_frame import CHANNEL_MODE_JOINT_STEREO, CHANNEL_MODE_MONO, Mp2Format
from dsp.superframe import AudioFormat
from recording.mp4 import write_m4a
from recording.playback import PlaybackError, RecordingAudio, read_m4a, read_mp2

SBR_STEREO = AudioFormat(
    dac_rate_48k=True, sbr=True, stereo_core=True, parametric_stereo=False, mpeg_surround_config=0
)
FORMATS = [
    SBR_STEREO,
    replace(SBR_STEREO, sbr=False),  # plain AAC LC at 48 kHz
    replace(SBR_STEREO, dac_rate_48k=False, stereo_core=False),  # 32 kHz, a mono core
    replace(SBR_STEREO, stereo_core=False, parametric_stereo=True),  # PS over a mono core
]


def test_the_audio_specific_config_reads_back_to_the_format():
    """PS is not in the configuration (the decoder finds it in the stream) and comes
    back False; everything else as written."""
    for audio_format in FORMATS:
        config = audio_specific_config(audio_format)
        assert audio_format_from_config(config) == replace(audio_format, parametric_stereo=False)


def test_a_configuration_this_radio_did_not_write_is_refused():
    with pytest.raises(ValueError, match="960"):
        audio_format_from_config(bytes.fromhex("1190"))  # AAC LC 48 kHz stereo, 1024 frames
    with pytest.raises(ValueError, match="AAC LC"):
        audio_format_from_config(bytes.fromhex("0a10"))  # AAC Main
    with pytest.raises(ValueError, match="sample rate"):
        audio_format_from_config(bytes.fromhex("1210"))  # 44.1 kHz
    with pytest.raises(ValueError, match="short"):
        audio_format_from_config(b"\x11")


def test_an_m4a_reads_back_to_its_units_and_format(tmp_path):
    units = [bytes([number]) * (50 + number) for number in range(7)]
    for audio_format in FORMATS:
        path = tmp_path / "title.m4a"
        write_m4a(path, units, audio_format, {"title": "Africa"}, 96_000, 1_791_000_000)

        read_format, read_units = read_m4a(path)

        assert read_units == tuple(units)
        assert read_format == replace(audio_format, parametric_stereo=False)

    audio = RecordingAudio("title", SBR_STEREO, read_units)
    assert audio.unit_seconds == pytest.approx(0.04)  # 960 core samples at 24 kHz
    assert audio.duration == pytest.approx(0.28)
    plain = RecordingAudio("title", FORMATS[1], read_units)
    assert plain.unit_seconds == pytest.approx(0.02)


def test_what_is_not_such_an_m4a_is_refused(tmp_path):
    path = tmp_path / "noise.m4a"
    path.write_bytes(bytes(range(256)) * 4)
    with pytest.raises(PlaybackError, match="moov"):
        read_m4a(path)

    write_m4a(path, [b"a" * 100] * 10, SBR_STEREO)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])  # cut in the audio, the tables lost
    with pytest.raises(PlaybackError, match="moov"):
        read_m4a(path)


def test_mp2_frames_split_by_their_fixed_length(tmp_path):
    stereo = Mp2Format(sample_rate=48_000, bitrate=128, channel_mode=CHANNEL_MODE_JOINT_STEREO)
    assert stereo.frame_length == 384
    path = tmp_path / "title.mp2"
    path.write_bytes(b"".join(bytes([number]) * 384 for number in range(5)) + b"\xff" * 100)

    frames = read_mp2(path, stereo)

    assert len(frames) == 5 and all(len(frame) == 384 for frame in frames)
    assert frames[3] == bytes([3]) * 384  # the incomplete tail is left out
    audio = RecordingAudio("title", stereo, frames)
    assert audio.unit_seconds == pytest.approx(0.024)
    assert audio.duration == pytest.approx(0.12)

    mono_lsf = Mp2Format(sample_rate=24_000, bitrate=64, channel_mode=CHANNEL_MODE_MONO)
    assert len(read_mp2(path, mono_lsf)) == 2020 // mono_lsf.frame_length
