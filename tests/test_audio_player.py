"""Tests for the playback logic (`audio.player`).

Deliberately without a real sound card: what is tested is the logic in front of it,
creating and switching the decoder, handling corrupted audio units and the cap on the
delay. Setting up the output (`QAudioSink`) is replaced for this, so no audio device
is needed while the buffer and decoder logic runs in full.
"""

from __future__ import annotations

import os
from typing import ClassVar

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtMultimedia import QAudio
from PyQt6.QtWidgets import QApplication

from audio.aac import AacDecoderError
from audio.equalizer import PEAK, Band, EqualizerSettings
from audio.player import (
    BUFFER_REPORT_INTERVAL_S,
    LEVEL_INTERVAL_SECONDS,
    MAX_QUEUED_SECONDS,
    PLAYBACK_LEAD_SECONDS,
    PLAYBACK_START_WARNING_S,
    PREBUFFER_SECONDS,
    SINK_BUFFER_SECONDS,
    AudioPlayer,
)
from dsp.mp2_frame import CHANNEL_MODE_JOINT_STEREO, Mp2Format, Mp2FrameResult
from dsp.superframe import AudioFormat, SuperframeResult
from recording.playback import PAUSED, PLAYING, PlaybackState, RecordingAudio

FORMAT_48K_STEREO = AudioFormat(
    dac_rate_48k=True, sbr=True, stereo_core=True, parametric_stereo=False, mpeg_surround_config=0
)
FORMAT_32K_MONO = AudioFormat(
    dac_rate_48k=False, sbr=True, stereo_core=False, parametric_stereo=False, mpeg_surround_config=0
)
MP2_FORMAT = Mp2Format(sample_rate=48_000, bitrate=128, channel_mode=CHANNEL_MODE_JOINT_STEREO)


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """The refill timer of the player needs an application with an event dispatcher."""
    return QApplication.instance() or QApplication([])


class _FakeDecoder:
    """Stands in for `audio.aac.AacDecoder` without needing FAAD2."""

    instances: ClassVar[list[_FakeDecoder]] = []

    def __init__(self, audio_format: AudioFormat) -> None:
        self.audio_format = audio_format
        self.sample_rate = audio_format.sample_rate
        self.channels = audio_format.channels
        self.closed = False
        self.fail_on: set[bytes] = set()
        self.decoded: list[bytes] = []
        _FakeDecoder.instances.append(self)

    def decode(self, audio_unit: bytes) -> np.ndarray:
        if audio_unit in self.fail_on:
            raise AacDecoderError("broken access unit")
        self.decoded.append(audio_unit)
        # 960 core samples, doubled with SBR, times the channel count
        return np.zeros(1920 * self.channels, dtype=np.int16)

    def close(self) -> None:
        self.closed = True


class _FakeMp2Decoder(_FakeDecoder):
    """Stands in for `audio.mp2.Mp2Decoder`: 1152 samples per frame."""

    def decode(self, audio_unit: bytes) -> np.ndarray:
        self.decoded.append(audio_unit)
        return np.zeros(1152 * self.channels, dtype=np.int16)


@pytest.fixture
def fake_decoder(monkeypatch):
    _FakeDecoder.instances.clear()
    monkeypatch.setattr("audio.player.AacDecoder", _FakeDecoder)
    monkeypatch.setattr("audio.player.Mp2Decoder", _FakeMp2Decoder)
    return _FakeDecoder


@pytest.fixture
def player(monkeypatch):
    instance = AudioPlayer()
    # The sound card stays out of it; decoder and buffer logic run in full as if the
    # output were switched on.
    monkeypatch.setattr(instance, "_setup_sink", lambda *args, **kwargs: None)
    yield instance
    instance.stop()


def _superframe(audio_format=FORMAT_48K_STEREO, units=(b"au",), sync_ok=True) -> SuperframeResult:
    return SuperframeResult(
        sync_ok=sync_ok,
        audio_units=list(units),
        num_audio_units=len(units),
        audio_format=audio_format if sync_ok else None,
    )


def _mp2_frame(frame=b"mp2", *, sync_ok=True, crc_ok=True) -> Mp2FrameResult:
    return Mp2FrameResult(audio_format=MP2_FORMAT, sync_ok=sync_ok, crc_ok=crc_ok, frame=frame)


def test_unsynchronised_superframe_is_ignored(player, fake_decoder):
    player.process_superframe(_superframe(sync_ok=False))
    assert fake_decoder.instances == []


def test_decoder_is_created_once_and_reused(player, fake_decoder):
    for _ in range(3):
        player.process_superframe(_superframe(units=(b"a", b"b", b"c")))

    assert len(fake_decoder.instances) == 1
    assert fake_decoder.instances[0].decoded == [b"a", b"b", b"c"] * 3


def test_format_change_creates_a_new_decoder(player, fake_decoder):
    """The AudioSpecificConfig can only be passed when opening, so a format change
    necessarily needs a fresh decoder."""
    player.process_superframe(_superframe(FORMAT_48K_STEREO))
    player.process_superframe(_superframe(FORMAT_32K_MONO))

    assert len(fake_decoder.instances) == 2
    assert fake_decoder.instances[0].closed, "the old decoder must be closed"
    assert fake_decoder.instances[1].audio_format == FORMAT_32K_MONO


def test_broken_audio_unit_does_not_abort_the_stream(player, fake_decoder):
    """A corrupted access unit must not break the audio; the next one is almost
    always valid again."""
    player.process_superframe(_superframe(units=(b"gut1",)))
    fake_decoder.instances[0].fail_on = {b"kaputt"}
    messages: list[str] = []
    player.status_changed.connect(messages.append)

    player.process_superframe(_superframe(units=(b"kaputt", b"gut2")))

    assert fake_decoder.instances[0].decoded == [b"gut1", b"gut2"]
    assert any("discarded" in m for m in messages)


def test_latency_is_capped_by_dropping_the_oldest_audio(player, fake_decoder):
    """The reserve never shrinks by itself; playback runs exactly in real time. A
    backlog (e.g. from the JIT compilation at start-up) would otherwise stay as
    permanent delay."""
    player.process_superframe(_superframe())
    seconds_per_unit = 1920 / FORMAT_48K_STEREO.sample_rate

    # Feed in so many units that the cap would be clearly exceeded
    units = int(3 * MAX_QUEUED_SECONDS / seconds_per_unit)
    player.process_superframe(_superframe(units=tuple(b"au" for _ in range(units))))

    assert player._queued_seconds() <= MAX_QUEUED_SECONDS + seconds_per_unit


def test_the_backlog_is_cut_in_whole_frames(player, fake_decoder, monkeypatch):
    """A cut through a frame shifts every later sample by a byte: loud noise, until
    another cut happens to shift it back."""
    # One byte more than a second of audio may stay: the cut would be odd.
    monkeypatch.setattr(
        "audio.player.MAX_QUEUED_SECONDS", 1.0 + 1 / FORMAT_48K_STEREO.sample_rate / 4
    )
    player.process_superframe(_superframe())
    units = int(3 / (1920 / FORMAT_48K_STEREO.sample_rate))

    player.process_superframe(_superframe(units=tuple(b"au" for _ in range(units))))

    assert len(player._buffer) % 4 == 0
    assert player._queued_seconds() <= 1.0 + 1920 / FORMAT_48K_STEREO.sample_rate


def test_disabling_the_sound_clears_the_buffer(player, fake_decoder):
    player.process_superframe(_superframe(units=(b"a", b"b")))
    assert player._queued_seconds() > 0

    player.set_enabled(False)

    assert player._queued_seconds() == 0


def test_muted_player_decodes_but_does_not_pile_up_audio(player, fake_decoder):
    """Muted means: keep decoding (so that format and errors are displayed) but
    buffer nothing. Otherwise the buffer would grow without bound, since nothing
    drains it without playback."""
    player.set_enabled(False)

    for _ in range(200):
        player.process_superframe(_superframe(units=(b"a", b"b", b"c")))

    assert len(fake_decoder.instances[0].decoded) == 600, "decoding must continue"
    assert player._queued_seconds() == 0


def test_a_held_player_lets_go_of_the_sound_card_without_saying_muted(playing_player, fake_decoder):
    """While the Bluetooth audio moves (`audio.bluetooth_route`) the sink is let go of,
    or PipeWire would move the stream to the default as the device's sink goes; the
    decoding goes on, and the next audio after the hold gets a new sink."""
    player, sinks = playing_player
    messages: list[str] = []
    player.status_changed.connect(messages.append)
    player.process_superframe(_superframe(units=(b"a", b"b")))
    assert len(sinks) == 1 and player._queued_seconds() > 0

    player.set_held(True)
    assert player._sink is None
    assert player._queued_seconds() == 0
    for _ in range(50):
        player.process_superframe(_superframe(units=(b"a", b"b", b"c")))
    assert len(sinks) == 1 and player._queued_seconds() == 0
    assert len(fake_decoder.instances[0].decoded) == 152, "decoding goes on"
    assert "muted" not in messages

    player.set_held(False)
    player.process_superframe(_superframe())
    assert len(sinks) == 2


class _Output:
    def __init__(self, device_id: bytes) -> None:
        self._id = device_id

    def id(self) -> bytes:
        return self._id

    def description(self) -> str:
        return self._id.decode()


def test_a_new_output_stays_silent_until_its_gain_comes(player):
    """The gain before may be the 1.0 of a Bluetooth device applying its volume itself,
    and the new output the loudspeaker at 100 %: silent until the controller answers."""
    reports: list[tuple] = []
    player.output_changed.connect(lambda *args: reports.append(args))
    player.set_volume(1.0)

    player._report_output(_Output(b"alsa_output.platform-soc_sound.stereo-fallback"))
    assert len(reports) == 1 and player._volume == 0.0
    player.set_volume(0.25)  # the controller's answer
    player._report_output(_Output(b"alsa_output.platform-soc_sound.stereo-fallback"))
    assert len(reports) == 1 and player._volume == 0.25  # the same output: kept


def test_stop_closes_the_decoder_and_reports_no_format(player, fake_decoder):
    player.process_superframe(_superframe())
    formats: list[object] = []
    player.format_changed.connect(formats.append)

    player.stop()

    assert fake_decoder.instances[0].closed
    assert formats == [None]
    # Afterwards everything starts over, including a fresh decoder.
    player.process_superframe(_superframe())
    assert len(fake_decoder.instances) == 2


def test_unavailable_decoder_is_reported_with_format_and_reason(player, monkeypatch):
    """Without FAAD2 the display must name the reason rather than show no format."""

    def unavailable(_audio_format):
        raise AacDecoderError("FAAD2 (libfaad2) was not found")

    monkeypatch.setattr("audio.player.AacDecoder", unavailable)
    formats: list[object] = []
    states: list[str] = []
    player.format_changed.connect(formats.append)
    player.status_changed.connect(states.append)

    player.process_superframe(_superframe())
    player.process_superframe(_superframe())

    assert formats == [FORMAT_48K_STEREO.describe()]
    assert states == ["FAAD2 (libfaad2) was not found"]


def test_mp2_frames_go_to_the_mp2_decoder(player, fake_decoder):
    for frame in (b"f1", b"f2", b"f3"):
        player.process_mp2_frame(_mp2_frame(frame))

    assert len(fake_decoder.instances) == 1
    assert isinstance(fake_decoder.instances[0], _FakeMp2Decoder)
    assert fake_decoder.instances[0].decoded == [b"f1", b"f2", b"f3"]


def test_mp2_frame_with_crc_error_is_replaced_by_silence(player, fake_decoder):
    """A frame whose header CRC failed may carry a corrupted bit allocation and is not
    decoded; its 24 ms are filled with silence so that the time axis does not shrink."""
    player.process_mp2_frame(_mp2_frame(b"gut"))
    queued = player._queued_seconds()

    player.process_mp2_frame(_mp2_frame(b"kaputt", crc_ok=False))
    player.process_mp2_frame(_mp2_frame(sync_ok=False, crc_ok=False))

    assert fake_decoder.instances[0].decoded == [b"gut"]
    assert player._queued_seconds() == pytest.approx(queued + 2 * MP2_FORMAT.duration_seconds)


def test_switching_from_dab_plus_to_dab_replaces_the_decoder(player, fake_decoder):
    player.process_superframe(_superframe())
    player.process_mp2_frame(_mp2_frame())

    aac, mp2 = fake_decoder.instances
    assert aac.closed, "the AAC decoder must be closed"
    assert isinstance(mp2, _FakeMp2Decoder)


def test_level_readings_are_evenly_spaced_whatever_the_frame_duration(player, fake_decoder):
    """The level display assumes one reading per LEVEL_INTERVAL_SECONDS. An MP2 frame
    covers 24 ms, so five frames (120 ms) must give three readings, not five."""
    levels: list[np.ndarray] = []
    player.level_ready.connect(levels.append)

    for _ in range(5):
        player.process_mp2_frame(_mp2_frame())

    assert LEVEL_INTERVAL_SECONDS == pytest.approx(0.04)
    assert len(levels) == 3
    assert all(level.shape == (MP2_FORMAT.channels,) for level in levels)


class _ToneDecoder(_FakeDecoder):
    """Decodes every access unit to 40 ms more of a steady 1 kHz tone at half scale."""

    def __init__(self, audio_format: AudioFormat) -> None:
        super().__init__(audio_format)
        self._position = 0

    def decode(self, audio_unit: bytes) -> np.ndarray:
        t = (self._position + np.arange(1920)) / self.sample_rate
        self._position += 1920
        tone = np.round(0.5 * 32767 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.int16)
        return np.repeat(tone, self.channels)


@pytest.fixture
def tone_decoder(monkeypatch):
    monkeypatch.setattr("audio.player.AacDecoder", _ToneDecoder)


CUT_AT_1KHZ = EqualizerSettings((Band(PEAK, 1000.0, -12.0),), None)


def _play_tone(player: AudioPlayer, superframes: int = 3) -> None:
    for _ in range(superframes):
        player.process_superframe(_superframe(units=(b"a", b"b", b"c")))  # 120 ms each


def test_the_equalizer_shapes_the_audio_and_the_level_shows_what_is_played(player, tone_decoder):
    levels: list[np.ndarray] = []
    player.level_ready.connect(levels.append)
    _play_tone(player)
    plain = levels[-1][0]

    player.set_equalizer(CUT_AT_1KHZ)
    _play_tone(player)
    cut = levels[-1][0]
    player.set_equalizer(None)
    _play_tone(player)

    assert plain == pytest.approx(-6.0, abs=0.1)
    assert cut - plain == pytest.approx(-12.0, abs=0.3)
    assert levels[-1][0] == pytest.approx(plain, abs=0.1)


def test_the_spectrum_is_reported_only_while_asked_for_and_before_the_equalizer(
    player, tone_decoder
):
    spectra: list[tuple[np.ndarray, np.ndarray]] = []
    player.spectrum_ready.connect(lambda centres, levels: spectra.append((centres, levels)))
    player.set_equalizer(CUT_AT_1KHZ)
    _play_tone(player)
    assert spectra == []

    player.set_spectrum_enabled(True)
    _play_tone(player)
    player.set_spectrum_enabled(False)
    _play_tone(player)

    assert len(spectra) == 3  # one per 100 ms of audio
    centres, levels = spectra[-1]
    band = int(np.argmin(np.abs(np.log2(centres / 1000.0))))
    assert levels[band] == pytest.approx(-6.0, abs=0.5)  # the tone as decoded


# --- A recording in place of the live audio ---------------------------------------------


class _Connectable:
    """Stands in for a Qt signal nobody listens to in the test."""

    def connect(self, *_args: object) -> None:
        pass

    def disconnect(self, *_args: object) -> None:
        pass


class _FakeSink:
    """Stands in for `QAudioSink` in push mode: holds up to `capacity` bytes, which the
    test drains as the sound card would; it is its own device (`start` returns it)."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.held = 0
        self.started = False
        self.suspended = False
        self.stateChanged = _Connectable()

    def bufferSize(self) -> int:
        return self.capacity

    def bytesFree(self) -> int:
        return self.capacity - self.held

    def start(self) -> _FakeSink:
        self.started = True
        return self

    def write(self, data: bytes) -> int:
        taken = min(len(data), self.bytesFree())
        self.held += taken
        return taken

    def suspend(self) -> None:
        self.suspended = True

    def resume(self) -> None:
        self.suspended = False

    def stop(self) -> None:
        pass

    def deleteLater(self) -> None:
        pass

    def setVolume(self, _volume: float) -> None:
        pass

    def setBufferSize(self, size: int) -> None:
        self.capacity = size

    def drain(self, seconds: float) -> None:
        """The sound card played this much."""
        self.held = max(0, self.held - int(seconds * BYTES_PER_SECOND))


BYTES_PER_SECOND = 48_000 * 2 * 2  # 48 kHz stereo, 16 bit
UNIT_SECONDS = 1920 / 48_000  # what the fake decoder makes of one unit: 40 ms


@pytest.fixture
def playing_player(monkeypatch):
    """A player whose sound card is a `_FakeSink`, made anew whenever the player sets
    one up; the sinks made so far, newest last."""
    instance = AudioPlayer()
    sinks: list[_FakeSink] = []

    def setup(_sample_rate: int, _channels: int) -> None:
        sink = _FakeSink(int(SINK_BUFFER_SECONDS * instance._bytes_per_second))
        instance._sink = sink
        instance._playing = False
        sinks.append(sink)

    monkeypatch.setattr(instance, "_setup_sink", setup)
    yield instance, sinks
    instance.stop()


def _recording(units: int, recording_id: str = "rec") -> RecordingAudio:
    return RecordingAudio(
        recording_id, FORMAT_48K_STEREO, tuple(f"u{n}".encode() for n in range(units))
    )


def test_a_recording_plays_in_place_of_the_live_audio_paced_by_the_sound_card(
    playing_player, fake_decoder
):
    player, sinks = playing_player
    states: list[object] = []
    player.playback_changed.connect(states.append)
    player.process_superframe(_superframe())  # the station, decoder 1

    player.play(_recording(100), 0.0)  # 4 s

    live, own = fake_decoder.instances  # a fresh decoder for the recording
    assert live.closed and own.audio_format == FORMAT_48K_STEREO
    # Decoded ahead only up to the lead: 0.8 s, 20 units; the position is what is heard.
    assert len(own.decoded) == 20 and player._queued_seconds() == pytest.approx(0.8)
    assert states[-1] == PlaybackState("rec", PLAYING, 0.0, 4.0)
    assert player._queued_seconds() <= MAX_QUEUED_SECONDS

    # The live audio is dropped undecoded meanwhile.
    player.process_superframe(_superframe(units=(b"live",)))
    assert b"live" not in own.decoded and len(fake_decoder.instances) == 2

    # The sound card started with `play` and took the queue; the file follows as it
    # plays, from the refill timer.
    sink = sinks[-1]
    assert sink.started and sink.held == sink.capacity
    sink.drain(0.5)
    player._flush()
    player._report_playback(force=True)
    assert len(own.decoded) == 33  # 0.5 s more, to the lead again
    assert states[-1].position == pytest.approx(0.5, abs=UNIT_SECONDS)
    assert player._queued_seconds() <= PLAYBACK_LEAD_SECONDS + UNIT_SECONDS

    # Played to the end: the live audio is back, decoder and card made anew.
    for _ in range(50):
        sink.drain(0.5)
        player._flush()
        if states[-1] is None:
            break
    assert states[-1] is None and len(own.decoded) == 100 and own.closed
    assert player._source is None
    player.process_superframe(_superframe(units=(b"live",)))
    assert fake_decoder.instances[-1].decoded == [b"live"] and len(fake_decoder.instances) == 3


def test_pause_holds_the_recording_and_a_seek_primes_the_decoder(playing_player, fake_decoder):
    player, sinks = playing_player
    states: list[object] = []
    player.playback_changed.connect(states.append)
    player.play(_recording(100), 0.0)
    own = fake_decoder.instances[-1]
    player._flush()
    sink = sinks[-1]

    player.pause()
    assert sink.suspended and states[-1].state == PAUSED
    decoded = len(own.decoded)
    sink.drain(0.5)
    player._flush()
    assert len(own.decoded) == decoded, "nothing is pulled while paused"
    player.resume()
    assert not sink.suspended and states[-1].state == PLAYING
    assert len(own.decoded) > decoded

    player.seek(2.0)  # unit 50: decoding begins two units before, their output dropped
    fresh = fake_decoder.instances[-1]
    assert fresh is not own and own.closed
    assert fresh.decoded[:3] == [b"u48", b"u49", b"u50"] and len(fresh.decoded) == 22
    assert player._queued_seconds() == pytest.approx(0.8)
    assert states[-1] == PlaybackState("rec", PLAYING, pytest.approx(2.0), 4.0)
    assert len(sinks) == 2, "the sound card is set up anew after a jump"


def test_stop_playback_and_stop_give_the_live_audio_back(player, fake_decoder):
    states: list[object] = []
    player.playback_changed.connect(states.append)
    player.play(_recording(10), 0.0)
    assert states[-1].state == PLAYING

    player.stop_playback()
    assert states[-1] is None and fake_decoder.instances[-1].closed
    player.process_superframe(_superframe(units=(b"live",)))
    assert fake_decoder.instances[-1].decoded == [b"live"]

    player.play(_recording(10), 0.0)
    player.stop()  # the station changes
    assert states[-1] is None and player._source is None


def test_a_stretch_of_the_cutter_halts_at_its_stop_and_goes_back(playing_player, fake_decoder):
    """The cutter plays up to the end mark: there the player pauses, back where it
    began, so the same passage plays again; the station stays silent meanwhile. The
    end of a playback forgets the stop."""
    player, sinks = playing_player
    states: list[object] = []
    player.playback_changed.connect(states.append)
    player.set_playback_stop(2.0, 1.0)  # halts before unit 50
    player.play(_recording(100), 1.0)
    own = fake_decoder.instances[-1]

    for _ in range(20):
        sinks[-1].drain(0.5)
        player._flush()
        if states[-1].state == PAUSED:
            break
    assert states[-1] == PlaybackState("rec", PAUSED, pytest.approx(1.0), 4.0)
    assert own.decoded[-1] == b"u49" and own.closed, "decoded up to the stop, then afresh"
    assert player._source is not None, "the station does not come back"

    player.resume()
    again = fake_decoder.instances[-1]
    assert again.decoded[:3] == [b"u23", b"u24", b"u25"]  # from 1.0 s, primed
    assert states[-1].state == PLAYING

    player.stop_playback()
    assert player._stop_seconds is None
    player.play(_recording(10), 0.0)
    assert player._source_end() == 10


def test_a_muted_player_does_not_run_through_the_recording(player, fake_decoder):
    player.set_enabled(False)
    player.play(_recording(100), 0.0)
    own = fake_decoder.instances[-1]
    assert own.decoded == []

    player.set_enabled(True)
    player._flush()
    assert len(own.decoded) == 20


def test_the_rest_of_a_recording_plays_however_short(playing_player, fake_decoder):
    """Less than the usual lead left after a jump near the end: the card starts anyway."""
    player, sinks = playing_player
    player.play(_recording(100), 3.9)
    player._flush()
    assert sinks[-1].started


def test_stop_leaves_the_refill_timer_running(player, fake_decoder):
    """`stop` comes with every change of station. A recording played back afterwards
    is refilled by the timer alone: the live audio refills with every superframe and
    never missed the timer, a recording starved after its first fill (10.10.2026)."""
    player.start()
    assert player._timer is not None and player._timer.isActive()

    player.stop()

    assert player._timer.isActive()
    player.play(_recording(100), 0.0)
    assert player._queued_seconds() == pytest.approx(PLAYBACK_LEAD_SECONDS)
    assert PLAYBACK_LEAD_SECONDS > PREBUFFER_SECONDS, "the card starts with the first fill"


def test_the_buffer_is_reported_twice_a_second_and_a_dropout_at_once(
    playing_player, fake_decoder, monkeypatch
):
    """Reported at every refill, every 20 ms and after every superframe, it made near
    60 calls a second into the GUI thread, for a figure read by eye."""
    player, sinks = playing_player
    clock = [1000.0]
    monkeypatch.setattr("audio.player.time.monotonic", lambda: clock[0])
    reports: list[tuple[float, int]] = []
    player.buffer_changed.connect(lambda seconds, underruns: reports.append((seconds, underruns)))

    for _ in range(20):
        player.process_superframe(_superframe())
        player._flush()
    assert sinks[-1].started
    assert len(reports) == 1

    clock[0] += BUFFER_REPORT_INTERVAL_S
    player._flush()
    assert len(reports) == 2

    player._on_sink_state_changed(QAudio.State.IdleState)  # the sound card ran dry
    player._flush()
    assert len(reports) == 3 and reports[-1][1] == 1

    player.stop()
    assert reports[-1] == (0.0, 1)


def test_a_playback_that_does_not_start_is_reported_once(player, fake_decoder, monkeypatch, caplog):
    """No sound card (`_setup_sink` does nothing here): after a while the log says what
    the player knows, once."""
    clock = [1000.0]
    monkeypatch.setattr("audio.player.time.monotonic", lambda: clock[0])
    player.play(_recording(100), 0.0)
    player._flush()
    assert not any("has not started" in r.message for r in caplog.records)

    clock[0] += PLAYBACK_START_WARNING_S + 1
    player._flush()
    player._flush()
    warnings = [r for r in caplog.records if "has not started" in r.message]
    assert len(warnings) == 1
    assert "decoder open, sink none, enabled True, 20 units decoded" in warnings[0].message
