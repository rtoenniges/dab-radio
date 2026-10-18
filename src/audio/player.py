"""Decoding and playback of the audio of DAB+ (AAC) and DAB (MP2) through QtMultimedia,
through the equalizer (`audio.equalizer`) on the way; and of a recording in place of the
live audio (`AudioPlayer.play`, `recording.playback`), through the same decoders and
the same path, so that volume, equalizer, output and spectrum hold for it alike."""

from __future__ import annotations

import logging
import time

import numpy as np
from PyQt6.QtCore import QIODevice, QObject, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtMultimedia import QAudio, QAudioDevice, QAudioFormat, QAudioSink, QMediaDevices

from audio.aac import AacDecoder, AacDecoderError
from audio.equalizer import Equalizer, EqualizerSettings, warm_up
from audio.mp2 import Mp2Decoder, Mp2DecoderError
from audio.spectrum import SpectrumAnalyzer
from dsp.mp2_frame import Mp2Format, Mp2FrameResult
from dsp.superframe import SUPERFRAME_SECONDS, AudioFormat, SuperframeResult
from i18n import tr, tr_noop
from recording.playback import PAUSED, PLAYING, PlaybackState, RecordingAudio

logger = logging.getLogger(__name__)

# A superframe carries 120 ms of audio but does not arrive evenly: the signal
# processing delivers the frames in bursts, one per read block. Without a reserve in
# the buffer the sound card would run dry between bursts (audible stuttering). 500 ms
# of lead bridge that comfortably.
PREBUFFER_SECONDS = 0.5

# Upper limit for the total audio queued ahead, i.e. the delay between reception and
# loudspeaker. When exceeded, the oldest data are dropped.
#
# Needed because the reserve never shrinks by itself: playback runs exactly in real
# time, so a backlog once built up stays forever as latency. And backlogs build up
# regularly: at start-up the JIT compilation of the Viterbi decoder (numba) takes a
# good second, during which samples pile up and are then processed faster than real
# time. Without the cap the delay stays above two seconds for good.
MAX_QUEUED_SECONDS = 1.0

# Buffer size of the sound card itself. Without an explicit value Qt picks very
# different sizes depending on platform and audio backend; a buffer that is too small
# runs dry between two refills.
SINK_BUFFER_SECONDS = 0.5

# Interval at which data are pushed to the sound card.
_FLUSH_INTERVAL_MS = 20
# Every so many flushes (2 s) the output is checked against the device chosen or the
# system's default: `QMediaDevices` reports devices appearing and going, but not a
# default changed by hand in the system.
_OUTPUT_CHECK_FLUSHES = 100
_UNREPORTED = object()

# A sound card that has taken no audio for this long while audio waits for it has lost
# its device: with the stream pinned to its sink (`main.PINNED_STREAM_PROPERTIES`)
# PipeWire ends the stream when the sink goes, and Qt 6.11 reports neither that nor
# the sink going, unless it was the default; it keeps listing the device. The sound
# card stops taking audio some 0.3 s after its sink went, while Qt goes on calling it
# active. A healthy one takes some every few tens of milliseconds; a dropout is no such
# thing, then no audio waits.
STALL_SECONDS = 0.5
_clock = time.monotonic

# Spacing of two readings of the level display. Measured over fixed windows of decoded
# audio rather than once per decoded unit: depending on the format an AAC access unit
# covers 20, 40 or 60 ms and an MP2 frame 24 or 48 ms, while the display assumes evenly
# spaced readings. 40 ms, i.e. 25 readings per second, are fine enough to see dynamics
# and dropouts and cheap enough not to burden the GUI.
LEVEL_INTERVAL_SECONDS = 0.04

# While a recording plays (`play`), it is decoded ahead only this far: below
# MAX_QUEUED_SECONDS, so that nothing is dropped, and far enough that the sound card
# never runs dry between two refills. The file thus runs exactly in real time.
PLAYBACK_LEAD_SECONDS = 0.8
# After a seek within a DAB+ recording, this many access units before the target are
# decoded and dropped: FAAD2 gives nothing for the first unit, and SBR needs a run-up.
PLAYBACK_PRIMING_UNITS = 2
# How often at most the position of a playback is reported (`playback_changed`).
PLAYBACK_REPORT_INTERVAL_S = 0.25
# How often at most the audio queued ahead is reported (`buffer_changed`); a dropout at
# once. A figure read by eye: reported at every refill and superframe, it would make
# near 60 calls a second into the GUI thread, each drawing a label anew, even with the
# page out of sight.
BUFFER_REPORT_INTERVAL_S = 0.5
# A recording whose sound card has not started this long after `play` is reported
# in the log, with what the player knows: the way to find out why it stays silent.
PLAYBACK_START_WARNING_S = 3.0


class AudioPlayer(QObject):
    """Decodes DAB+ superframes (AAC) or DAB audio frames (MP2) to PCM and plays them
    through the sound card.

    Lives in a QThread of its own: the audio decoding should hold up neither the GUI
    nor the signal processing, and the output to the sound card has to keep flowing
    evenly while the signal processing is busy with a block of samples.
    """

    # State reports for the display
    format_changed = pyqtSignal(object)  # description (str) or None
    status_changed = pyqtSignal(str)
    # Audio queued ahead (s) and underruns, at most every `BUFFER_REPORT_INTERVAL_S`.
    buffer_changed = pyqtSignal(float, int)
    level_ready = pyqtSignal(object)  # peak level per channel in dBFS (np.ndarray)
    # Band centres (Hz) and levels (dBFS) of the audio before the equalizer, while asked
    # for (`set_spectrum_enabled`); see `audio.spectrum`.
    spectrum_ready = pyqtSignal(object, object)
    # The output the audio goes (or would go) to: the id of the device (bytes) and its
    # description, None and "" without one. On every change: `set_output`, the default
    # of the system, a device appearing or going.
    output_changed = pyqtSignal(object, str)
    # A recording plays in place of the live audio (`play`): its state about four times
    # a second and at every change, None once it ended or was stopped.
    playback_changed = pyqtSignal(object)  # recording.playback.PlaybackState or None

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._decoder: AacDecoder | Mp2Decoder | None = None
        self._audio_format: AudioFormat | Mp2Format | None = None
        self._sink: QAudioSink | None = None
        self._device: QIODevice | None = None
        self._buffer = bytearray()
        self._bytes_per_second = 0
        self._playing = False
        self._enabled = True
        # Kept from the sound card while the Bluetooth audio moves (`set_held`).
        self._held = False
        self._volume = 1.0
        self._underruns = 0
        # When `buffer_changed` last went out, and the underruns it told.
        self._buffer_reported_at = 0.0
        self._underruns_reported = 0
        self._timer: QTimer | None = None
        # Decoded samples not yet covered by a level reading, as (frames, channels).
        self._level_pending = np.empty((0, 0), dtype=np.int16)
        self._equalizer = Equalizer()
        # Only while a display shows the spectrum: it costs an FFT ten times a second.
        self._spectrum: SpectrumAnalyzer | None = None
        # The output chosen (`set_output`): the id of a device, None for the default of
        # the system; the device the sink plays to; the default of the system at the
        # last look (`_follow_output`); what `output_changed` last told.
        self._output_id: bytes | None = None
        self._sink_device: QAudioDevice | None = None
        # When the sound card last took audio (`STALL_SECONDS`), and the devices whose
        # stream ended so, kept from `_chosen_device` until Qt reports a device
        # appearing or another output is chosen.
        self._taken_at = 0.0
        self._lost: set[bytes] = set()
        self._stall_warned = False  # until a sound card takes audio again
        self._default_id: bytes | None = None
        self._reported_output: object = _UNREPORTED
        self._media_devices: QMediaDevices | None = None
        self._flushes = 0
        # The recording playing instead of the live audio (`play`): the source, the next
        # unit to decode, how many were decoded (for the position), how many of the next
        # decoded are dropped (the run-up after a seek), and whether it is paused.
        self._source: RecordingAudio | None = None
        self._source_index = 0
        self._source_decoded = 0
        self._source_skip = 0
        self._paused = False
        self._playback_reported_at = 0.0
        # Where the recording halts and where it goes back to then (`set_playback_stop`,
        # the cutter): the seconds it stops at, None to play to the end.
        self._stop_seconds: float | None = None
        self._back_to = 0.0
        # For the log: when the recording was asked for, how many of its units the
        # decoder refused or gave nothing for, and whether the silence was reported.
        self._source_started_at = 0.0
        self._source_failed = 0
        self._source_silent = 0
        self._source_warned = False

    @pyqtSlot()
    def start(self) -> None:
        """Creates the refill timer; to be called in the player's own thread."""
        warm_up()
        if self._media_devices is None:
            # Made here, in the player's thread: its signal reports devices appearing
            # and going (headphones connecting), to follow or to return to the default.
            self._media_devices = QMediaDevices(self)
            self._media_devices.audioOutputsChanged.connect(self._on_outputs_changed)
        self._follow_output()
        if self._timer is None:
            self._timer = QTimer(self)
            self._timer.setInterval(_FLUSH_INTERVAL_MS)
            self._timer.timeout.connect(self._flush)
        self._timer.start()

    @pyqtSlot()
    def stop(self) -> None:
        """Ends playback and releases the sound card: the station changes, or the radio
        ends. The refill timer keeps running: `start` comes once, with the thread, and
        with the timer stopped a recording played back (`play`) would starve after its
        first fill, since only the superframes refill otherwise."""
        self._drop_source()
        self._teardown_sink()
        self._close_decoder()
        self._buffer.clear()
        self._audio_format = None
        self.format_changed.emit(None)
        self._report_buffer(0.0, force=True)

    @pyqtSlot(bool)
    def set_enabled(self, enabled: bool) -> None:
        """Switches the audio output. Decoding continues so that the display (format,
        errors) stays meaningful and the audio resumes without a start-up delay."""
        self._enabled = enabled
        if not enabled:
            self._teardown_sink()
            self._buffer.clear()
            self._report_buffer(0.0, force=True)
            self.status_changed.emit(tr("muted"))

    @pyqtSlot(bool)
    def set_held(self, held: bool) -> None:
        """Keeps the audio from the sound card while the radio moves its Bluetooth audio
        from one device to another (`audio.bluetooth_route`), as muting does, but without
        telling so: the sink of a device changing profile goes, and PipeWire would move
        the stream to the default, maybe the loudspeaker, at the gain meant for a device
        that applies its volume itself."""
        self._held = held
        if held:
            self._teardown_sink()
            self._buffer.clear()
            self._report_buffer(0.0, force=True)

    @property
    def _audible(self) -> bool:
        return self._enabled and not self._held

    @pyqtSlot(float)
    def set_volume(self, volume: float) -> None:
        """The linear factor of the sink, 0.0 to 1.0; the slider's scale is in
        `audio.volume`."""
        self._volume = max(0.0, min(1.0, volume))
        if self._sink is not None:
            self._sink.setVolume(self._volume)

    @pyqtSlot(object)
    def set_output(self, device_id: object) -> None:
        """Plays to the device with this id (bytes, as `QAudioDevice.id` gives it) from
        now on, or to the system's default with None. A device that is not there leaves
        the default playing until it appears."""
        self._output_id = bytes(device_id) if device_id else None
        self._lost.clear()
        self._follow_output()

    @pyqtSlot(object)
    def set_equalizer(self, settings: EqualizerSettings | None) -> None:
        """The equalizer from the next decoded block on; None switches it off."""
        self._equalizer.configure(settings)

    @pyqtSlot(bool)
    def set_spectrum_enabled(self, enabled: bool) -> None:
        """Whether `spectrum_ready` reports the spectrum of the audio."""
        if not enabled:
            self._spectrum = None
        elif self._spectrum is None:
            self._spectrum = SpectrumAnalyzer()

    @pyqtSlot(object)
    def process_superframe(self, result: SuperframeResult) -> None:
        """Takes a decoded superframe (slot for `dsp.pipeline.DspWorker.superframe_ready`).

        Missing audio is replaced by silence rather than simply left out. That matters
        more than it sounds: playback runs exactly in real time, so a skipped
        superframe permanently shortens the reserve in the output buffer. It can sink
        from 500 ms to below 160 ms within a minute, below the spacing of two data
        bursts from the signal processing (~192 ms), and then the sound card runs dry
        before every burst. With silence at the gap the time axis stands still and the
        reserve is preserved.
        """
        if self._source is not None:
            return  # a recording plays instead; the live audio is dropped undecoded
        if result.audio_format is None or not result.sync_ok:
            self._conceal(SUPERFRAME_SECONDS)
            self._flush()
            return

        if result.audio_format != self._audio_format:
            self._reconfigure(result.audio_format)
        if self._decoder is None:
            return

        decoded = 0
        for audio_unit in result.audio_units:
            try:
                pcm = self._decoder.decode(audio_unit)
            except AacDecoderError as exc:
                # A corrupted access unit must not break the stream; the next one is
                # almost always valid again.
                logger.debug("Audio unit discarded: %s", exc)
                self.status_changed.emit(tr("Audio unit discarded ({error})").format(error=exc))
                continue
            if len(pcm):
                self._append(pcm)
                decoded += 1

        # Individual access units that already failed their CRC leave a gap in the
        # time axis as well.
        missing = result.num_audio_units - decoded
        if missing > 0 and result.num_audio_units:
            self._conceal(missing * SUPERFRAME_SECONDS / result.num_audio_units)

        self._flush()

    @pyqtSlot(object)
    def process_mp2_frame(self, result: Mp2FrameResult) -> None:
        """Takes a DAB audio frame (slot for `dsp.pipeline.DspWorker.mp2_frame_ready`).

        A frame whose header CRC failed is not decoded but replaced by silence, for
        the same reason as a lost superframe (see `process_superframe`). Its bit
        allocation may be corrupted, and a decoder working from a wrong allocation
        produces loud artefacts rather than just noise.
        """
        if self._source is not None:
            return  # a recording plays instead
        if not result.sync_ok or not result.crc_ok:
            self._conceal(result.audio_format.duration_seconds)
            self._flush()
            return

        if result.audio_format != self._audio_format:
            self._reconfigure(result.audio_format)
        if self._decoder is None:
            return

        try:
            pcm = self._decoder.decode(result.frame)
        except Mp2DecoderError as exc:
            logger.debug("MP2 frame discarded: %s", exc)
            self.status_changed.emit(tr("MP2 frame discarded ({error})").format(error=exc))
            self._conceal(result.audio_format.duration_seconds)
        else:
            if len(pcm):
                self._append(pcm)
        self._flush()

    def _conceal(self, seconds: float) -> None:
        """Fills a gap in the audio stream with silence (see `process_superframe`)."""
        if self._decoder is None or self._bytes_per_second == 0 or not self._audible:
            return
        num_samples = int(seconds * self._decoder.sample_rate) * self._decoder.channels
        self._append(np.zeros(num_samples, dtype=np.int16))

    def _reconfigure(self, audio_format: AudioFormat | Mp2Format) -> None:
        self._close_decoder()
        self._teardown_sink()
        self._buffer.clear()
        self._level_pending = np.empty((0, 0), dtype=np.int16)
        self._equalizer.reset()
        if self._spectrum is not None:
            self._spectrum.reset()
        self._audio_format = audio_format
        self._open_decoder()

    def _open_decoder(self, announce: bool = True) -> None:
        """A decoder for `_audio_format`; `announce` reports format and state."""
        audio_format = self._audio_format
        assert audio_format is not None
        try:
            if isinstance(audio_format, Mp2Format):
                self._decoder = Mp2Decoder(audio_format)
            else:
                self._decoder = AacDecoder(audio_format)
        except (AacDecoderError, Mp2DecoderError) as exc:
            self._decoder = None
            # The format is known all the same. Reported as None, the display would
            # show "Audio: -" and swallow the reason, such as FAAD2 missing.
            self.format_changed.emit(audio_format.describe())
            self.status_changed.emit(str(exc))
            logger.warning("Audio decoder unavailable: %s", exc)
            return
        if announce:
            self.format_changed.emit(audio_format.describe())
            self.status_changed.emit(tr("Decoding audio..."))

    # --- A recording in place of the live audio ----------------------------------------

    @pyqtSlot(object, float)
    def play(self, audio: RecordingAudio, position: float = 0.0) -> None:
        """Plays a recording (`recording.playback.RecordingAudio`) from `position`
        seconds on, in place of the live audio, which is dropped undecoded meanwhile.
        The decoder is made anew for the recording's format, the sound card too, as at a
        change of station; `stop_playback`, the end of the recording or `stop` give the
        live audio back."""
        self._source = audio
        self._paused = False
        self._source_started_at = time.monotonic()
        self._source_failed = 0
        self._source_silent = 0
        self._source_warned = False
        logger.info(
            "Playback of %s: %d units, %.1f s, %s, from %.1f s",
            audio.recording_id,
            len(audio.units),
            audio.duration,
            audio.audio_format.describe(),
            position,
        )
        self._reconfigure(audio.audio_format)
        self.status_changed.emit(tr("Playing the recording"))
        self._seek_source(position, fresh_decoder=False)
        self._pull()
        self._flush()  # the sound card starts at once, not with the next timer tick
        self._report_playback(force=True)

    @pyqtSlot()
    def pause(self) -> None:
        """Holds the recording playing; the live audio stays silent."""
        if self._source is None or self._paused:
            return
        self._paused = True
        if self._sink is not None and self._playing:
            self._sink.suspend()
        self._report_playback(force=True)

    @pyqtSlot()
    def resume(self) -> None:
        if self._source is None or not self._paused:
            return
        self._paused = False
        if self._sink is not None and self._playing:
            self._sink.resume()
        self._pull()
        self._flush()
        self._report_playback(force=True)

    @pyqtSlot(float)
    def seek(self, position: float) -> None:
        """Goes on from `position` seconds into the recording playing."""
        if self._source is None:
            return
        self._seek_source(position, fresh_decoder=True)
        self._pull()
        self._flush()
        self._report_playback(force=True)

    @pyqtSlot(float, float)
    def set_playback_stop(self, stop: float, back_to: float) -> None:
        """Where the recording halts: having played up to `stop` seconds, it pauses and
        goes back to `back_to`, so that the same passage plays again (the cutter plays a
        title up to its end mark). Negative `stop`: to the end, then the station is back.
        Holds for the next `play` as well; the end of a playback forgets it."""
        self._stop_seconds = None if stop < 0 else stop
        self._back_to = max(0.0, back_to)

    def _source_end(self) -> int:
        """The unit the recording playing stops before."""
        source = self._source
        assert source is not None
        if self._stop_seconds is None:
            return len(source.units)
        return max(0, min(round(self._stop_seconds / source.unit_seconds), len(source.units)))

    def _halt(self) -> None:
        """Played up to the stop: paused, back where it began."""
        self._paused = True
        self._seek_source(self._back_to, fresh_decoder=True)
        self._report_playback(force=True)

    @pyqtSlot()
    def stop_playback(self) -> None:
        """Ends the recording playing: the live audio is back with the next superframe,
        decoder and sound card set up anew as after a change of station."""
        if self._source is not None:
            self._leave_playback(tr_noop("Back to the station"))

    def _leave_playback(self, message: str) -> None:
        """Back to the live audio; `message` (marked with `tr_noop`) goes to the log and,
        translated, to the status."""
        if self._source_failed or self._source_silent:
            logger.warning(
                "Playback: the decoder refused %d units and gave nothing for %d",
                self._source_failed,
                self._source_silent,
            )
        logger.info("Playback over: %s", message)
        self._drop_source()
        self._close_decoder()
        self._teardown_sink()
        self._buffer.clear()
        self._audio_format = None
        self.status_changed.emit(tr(message))

    def _drop_source(self) -> None:
        """Forgets the recording playing, if any, and says so."""
        if self._source is None:
            return
        self._source = None
        self._paused = False
        self._source_skip = 0
        self._stop_seconds = None
        self.playback_changed.emit(None)

    def _seek_source(self, position: float, fresh_decoder: bool) -> None:
        """Begins anew at `position`: what was queued goes, the sound card is set up
        anew with the next audio, and the decoder starts a few units before the target
        (`PLAYBACK_PRIMING_UNITS`), whose output is dropped."""
        source = self._source
        assert source is not None
        target = max(0, min(int(position / source.unit_seconds), len(source.units)))
        priming = PLAYBACK_PRIMING_UNITS if isinstance(source.audio_format, AudioFormat) else 0
        self._source_index = max(0, target - priming)
        self._source_skip = target - self._source_index
        self._source_decoded = target
        self._buffer.clear()
        self._teardown_sink()
        if fresh_decoder and self._decoder is not None:
            # FAAD2 carries state from unit to unit; after a jump it starts afresh.
            self._close_decoder()
            self._open_decoder(announce=False)

    def _source_exhausted(self) -> bool:
        return self._source is not None and self._source_index >= self._source_end()

    def _pull(self) -> None:
        """Decodes the recording playing as far ahead as `PLAYBACK_LEAD_SECONDS`, from
        the refill timer: the file thus runs exactly in real time, however fast it
        could be decoded. Not while paused or muted, when nothing drains the queue."""
        source = self._source
        if source is None or self._paused or not self._audible or self._decoder is None:
            return
        end = self._source_end()
        while self._source_index < end:
            if self._queued_seconds() >= PLAYBACK_LEAD_SECONDS:
                return
            unit = source.units[self._source_index]
            self._source_index += 1
            try:
                pcm = self._decoder.decode(unit)
            except (AacDecoderError, Mp2DecoderError) as exc:
                logger.debug("Unit of the recording discarded: %s", exc)
                self._source_failed += 1
                pcm = None
            if self._source_skip > 0:
                self._source_skip -= 1
                continue
            self._source_decoded += 1
            if pcm is not None and len(pcm):
                self._append(pcm)
            elif pcm is not None:
                self._source_silent += 1

    def _report_playback(self, force: bool = False) -> None:
        """Where the recording is: what was decoded less what is still queued ahead of
        the loudspeaker; at most every `PLAYBACK_REPORT_INTERVAL_S` unless forced."""
        source = self._source
        if source is None:
            return
        now = time.monotonic()
        if not force and now - self._playback_reported_at < PLAYBACK_REPORT_INTERVAL_S:
            return
        self._playback_reported_at = now
        played = self._source_decoded * source.unit_seconds - self._queued_seconds()
        position = max(0.0, min(source.duration, played))
        state = PAUSED if self._paused else PLAYING
        self.playback_changed.emit(
            PlaybackState(source.recording_id, state, position, source.duration)
        )

    def _append(self, pcm: np.ndarray) -> None:
        # The data rate hangs on the DECODER, not on the sound card: otherwise it
        # would be unknown while the output is muted and the delay cap (below) would
        # not take effect, so the buffer would grow without bound.
        self._bytes_per_second = self._decoder.sample_rate * self._decoder.channels * 2
        sample_rate, channels = self._decoder.sample_rate, self._decoder.channels
        if self._spectrum is not None:
            for centres, levels in self._spectrum.add(pcm, channels, sample_rate):
                self.spectrum_ready.emit(centres, levels)
        # The level after the equalizer: what goes to the loudspeaker.
        pcm = self._equalizer.process(pcm, channels, sample_rate)
        self._emit_level(pcm)
        if not self._audible:
            return  # muted or held: keep decoding, but do not queue anything

        # The definitive channel count is only known after the first decoded frame:
        # with parametric stereo a mono core yields two channels, which FAAD2 cannot
        # know before the first access unit. The sound card is therefore set up here
        # and not already from the superframe header.
        if self._sink is None:
            self._setup_sink(self._decoder.sample_rate, self._decoder.channels)
        self._buffer.extend(pcm.tobytes())

        # Drop the oldest first so that the delay stays capped. The yardstick is the
        # TOTAL audio queued ahead, not only this intermediate buffer: the lion's
        # share sits in the sound card's buffer. Whole frames only (every channel of a
        # sample, two bytes each): a cut through one shifts all later samples by a byte
        # or swaps the channels, and the sound card plays loud noise until another cut
        # happens to shift them back. rtl_tcp over Wi-Fi delivers in bursts and so
        # builds backlogs often.
        excess = self._queued_seconds() - MAX_QUEUED_SECONDS
        if excess > 0 and self._bytes_per_second:
            frame = 2 * channels
            drop = min(len(self._buffer), int(excess * self._bytes_per_second)) // frame * frame
            if drop > 0:
                del self._buffer[:drop]
                self.status_changed.emit(
                    tr("Backlog cleared, {milliseconds} ms dropped").format(
                        milliseconds=f"{drop / self._bytes_per_second * 1000:.0f}"
                    )
                )

    def _emit_level(self, pcm: np.ndarray) -> None:
        """Reports the peak level per channel (dBFS) for the level display, one reading
        per `LEVEL_INTERVAL_SECONDS` of decoded audio."""
        channels = self._decoder.channels
        if channels == 0 or len(pcm) < channels:
            return
        frames = pcm[: len(pcm) // channels * channels].reshape(-1, channels)
        # The channel count can change once after the start (parametric stereo, see
        # `_append`); samples of the old count cannot be combined with the new ones.
        if self._level_pending.shape[1] != channels:
            self._level_pending = np.empty((0, channels), dtype=np.int16)
        pending = np.concatenate([self._level_pending, frames])

        window = max(1, int(LEVEL_INTERVAL_SECONDS * self._decoder.sample_rate))
        complete = len(pending) // window * window
        for start in range(0, complete, window):
            block = pending[start : start + window]
            peak = np.abs(block.astype(np.float32)).max(axis=0) / 32768.0
            self.level_ready.emit(20.0 * np.log10(np.maximum(peak, 1e-6)))
        self._level_pending = pending[complete:]

    def _setup_sink(self, sample_rate: int, channels: int) -> None:
        audio_format = QAudioFormat()
        audio_format.setSampleRate(sample_rate)
        audio_format.setChannelCount(channels)
        audio_format.setSampleFormat(QAudioFormat.SampleFormat.Int16)

        device = self._chosen_device()
        if device is None:
            logger.warning("No audio output device")
            self.status_changed.emit(tr("No audio output found"))
            return
        if not device.isFormatSupported(audio_format):
            logger.warning(
                "%s does not support %d Hz, %d channels",
                device.description(),
                sample_rate,
                channels,
            )
            self.status_changed.emit(
                tr("Audio output does not support {rate} Hz / {channels} channels").format(
                    rate=sample_rate, channels=channels
                )
            )
            return

        self._sink = QAudioSink(device, audio_format, self)
        self._sink.setBufferSize(int(SINK_BUFFER_SECONDS * self._bytes_per_second))
        self._sink.setVolume(self._volume)
        self._sink.stateChanged.connect(self._on_sink_state_changed)
        self._playing = False
        self._sink_device = device
        self._default_id = _default_output_id()  # whether the new stream follows it
        self._report_output(device)

    def _chosen_device(self) -> QAudioDevice | None:
        """The device to play to: the one chosen if it is there, else the default of
        the system; None without any. A device whose stream ended (`_lost`) counts as
        not there, however Qt lists it."""
        if self._output_id is not None and self._output_id not in self._lost:
            for device in QMediaDevices.audioOutputs():
                if bytes(device.id()) == self._output_id:
                    return device
        device = QMediaDevices.defaultAudioOutput()
        if device.isNull() or bytes(device.id()) in self._lost:
            return None
        return device

    def _on_outputs_changed(self) -> None:
        """Qt reports a device appearing or the default changing: a device lost may be
        back, and is tried again (a stream for one still gone fails at once and ends
        as a stall)."""
        self._lost.clear()
        self._follow_output()

    def _on_stalled(self) -> None:
        """The sound card takes no more audio (`STALL_SECONDS`): its device is gone.
        The sink goes, and the audio goes on to the default where there is one, silent
        until the gain for it comes (`_report_output`)."""
        device = self._sink_device
        name = "the sound card" if device is None else device.description()
        log = logger.info if self._stall_warned else logger.warning
        self._stall_warned = True
        log(
            "%s has taken no audio for %.1f s; its device is gone, the output follows",
            name,
            STALL_SECONDS,
        )
        if device is not None:
            self._lost.add(bytes(device.id()))
        self._teardown_sink()
        self._buffer.clear()
        self._follow_output()

    def _follow_output(self) -> None:
        """Moves the output to the device chosen (or the default) once it differs from
        the one playing: the sink is torn down and made anew with the next audio.

        A device chosen is played to anew as well when the default of the system moves
        away from it: WirePlumber 0.5 marks a stream whose own target is the default as
        following the default (`checkFollowDefault`, as PulseAudio does) at its next
        rescan, which any node appearing sets off, and moves it along with the default.
        With the loudspeaker chosen, the audio would go to headphones as they connect,
        the player still attenuating for the loudspeaker: near silence. A new stream to
        the device, no longer the default, stays there."""
        device = self._chosen_device()
        default = _default_output_id()
        if self._sink is not None:
            playing = None if self._sink_device is None else bytes(self._sink_device.id())
            if device is None or bytes(device.id()) != playing:
                self._teardown_sink()
            elif playing == self._default_id and default != playing:
                logger.info(
                    "The default output moved away from %s; its stream is made anew",
                    self._sink_device.description(),
                )
                self._teardown_sink()
        self._default_id = default
        self._report_output(device)

    def _report_output(self, device: QAudioDevice | None) -> None:
        """Tells the new output, silent until the gain for it comes back (the controller
        answers every `output_changed` with one, `ReceiverController._apply_volume`):
        the gain of the output before may be 1.0, that of a Bluetooth device applying
        its volume itself, and the new one the loudspeaker, pinned at 100 % for the
        player to attenuate. A sink made with the next audio would play at that 1.0
        until the answer came."""
        device_id = None if device is None else bytes(device.id())
        if device_id != self._reported_output:
            self._reported_output = device_id
            self._volume = 0.0
            if self._sink is not None:
                self._sink.setVolume(0.0)
            self.output_changed.emit(device_id, "" if device is None else device.description())

    def _on_sink_state_changed(self, state: QAudio.State) -> None:
        # In push mode, IdleState means the sound card has run out of data, i.e. an
        # audible dropout. This is the only reliable place to detect that: the
        # intermediate buffer here is empty in normal operation anyway, because
        # everything is passed on to the sink immediately.
        if state == QAudio.State.IdleState and self._playing and not self._source_exhausted():
            self._underruns += 1
            self.status_changed.emit(
                tr("Dropout (buffer ran dry), {count} so far").format(count=self._underruns)
            )

    def _flush(self) -> None:
        self._flushes += 1
        if self._flushes % _OUTPUT_CHECK_FLUSHES == 0:
            if self._lost and self._sink is None and self._chosen_device() is None:
                # Nothing else to play to: the device lost is tried again, it may have
                # stalled for another reason.
                self._lost.clear()
            self._follow_output()
        if self._source is not None:
            self._pull()
            self._report_playback()
            self._warn_if_silent()
        if self._sink is None or self._bytes_per_second == 0:
            return

        buffered = self._queued_seconds()
        if not self._playing:
            # Only start once enough reserve has built up (see PREBUFFER_SECONDS),
            # otherwise the beginning stutters; the rest of a recording, however short,
            # plays. Not while a recording is paused before it even started.
            if (buffered < PREBUFFER_SECONDS and not self._source_exhausted()) or self._paused:
                self._report_buffer(buffered)
                return
            self._device = self._sink.start()
            self._playing = True
            self._taken_at = _clock()
            self.status_changed.emit(tr("Playback running"))

        now = _clock()
        if self._paused or not self._buffer:
            self._taken_at = now  # nothing for the sound card to take
        if self._device is not None and self._buffer:
            writable = min(len(self._buffer), self._sink.bytesFree())
            if writable > 0:
                written = self._device.write(bytes(self._buffer[:writable]))
                if written > 0:
                    del self._buffer[:written]
                    self._taken_at = now
                    self._stall_warned = False
        if now - self._taken_at >= STALL_SECONDS:
            self._on_stalled()
            return
        self._report_buffer(self._queued_seconds())
        if self._source_exhausted() and self._queued_seconds() <= 0:
            if self._stop_seconds is not None:
                self._halt()
            else:
                self._leave_playback(tr_noop("Recording ended, back to the station"))

    def _report_buffer(self, seconds: float, force: bool = False) -> None:
        """What is queued ahead and the dropouts so far: at most every
        `BUFFER_REPORT_INTERVAL_S`, a new dropout at once, unless forced."""
        now = time.monotonic()
        if (
            not force
            and self._underruns == self._underruns_reported
            and now - self._buffer_reported_at < BUFFER_REPORT_INTERVAL_S
        ):
            return
        self._buffer_reported_at = now
        self._underruns_reported = self._underruns
        self.buffer_changed.emit(seconds, self._underruns)

    def _warn_if_silent(self) -> None:
        """A recording whose sound card has not started `PLAYBACK_START_WARNING_S` after
        `play`: once into the log, with everything that could explain it."""
        if self._source_warned or self._playing or self._paused:
            return
        if time.monotonic() - self._source_started_at < PLAYBACK_START_WARNING_S:
            return
        self._source_warned = True
        logger.warning(
            "Playback has not started after %.0f s: decoder %s, sink %s, enabled %s, "
            "%d units decoded, %d refused, %d silent, %.2f s queued, %d bytes/s",
            PLAYBACK_START_WARNING_S,
            "open" if self._decoder is not None else "none",
            "set up" if self._sink is not None else "none",
            self._enabled,
            self._source_decoded,
            self._source_failed,
            self._source_silent,
            self._queued_seconds(),
            self._bytes_per_second,
        )

    def _queued_seconds(self) -> float:
        """How much audio is queued ahead in total: this intermediate buffer plus what
        the sound card has not played yet."""
        if self._bytes_per_second == 0:
            return 0.0
        queued = len(self._buffer)
        if self._sink is not None and self._playing:
            queued += max(0, self._sink.bufferSize() - self._sink.bytesFree())
        return queued / self._bytes_per_second

    def _teardown_sink(self) -> None:
        if self._sink is not None:
            self._sink.stateChanged.disconnect(self._on_sink_state_changed)
            self._sink.stop()
            self._sink.deleteLater()
        self._sink = None
        self._sink_device = None
        self._device = None
        self._playing = False

    def _close_decoder(self) -> None:
        if self._decoder is not None:
            self._decoder.close()
            self._decoder = None


def _default_output_id() -> bytes | None:
    """The id of the system's default output, None without one."""
    device = QMediaDevices.defaultAudioOutput()
    return None if device.isNull() else bytes(device.id())
