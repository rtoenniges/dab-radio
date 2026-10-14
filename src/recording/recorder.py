"""The ring of the last titles of the station playing, on the USB drive.

The radio records all the time: the raw stream behind the error correction, as it goes
to the decoder (the AAC access units of DAB+, `dsp.superframe.SuperframeResult`, or
the MPEG audio frames of DAB, `dsp.mp2_frame.Mp2FrameResult`), goes title by title
into files under `<directory>/Puffer/`. Where a title begins and ends says the dynamic
label (`recording.titles`), shifted by the label delay of the station (`set_label_delay`:
the audio of the last seconds is kept in memory, so that a title can begin before the
label that announced it; and `set_stop_delay`: the title before ends that much before
the label, cut short, or that much after it, running on in a file of its own while
the next already records, `_closing`; what lies between stands in both files or in
neither); a title also ends with the station, with
reception, after `MAX_TITLE_SECONDS`, and when the audio format changes. Of the complete titles the
ring keeps the last `ring_size`, the oldest goes; one shorter than `MIN_TITLE_SECONDS`
(a jingle, a snippet of the presenter) is not kept at all. "Keep" (`keep`) moves a
title out of the buffer into the directory itself, where it stays; asked of the title
recording, it is moved as soon as it is complete.

While a title records, it is an ADTS stream (`recording.adts`): robust, readable up to
the last complete unit should the power go. Complete, a DAB+ title becomes an `.m4a`
(`recording.mp4`, the container that makes it playable), a DAB title stays the raw
`.mp2` every player takes. Beside every file lies a `.json` with what the surface
shows (station, title, artist, time, duration, ...): the lists are read from those,
never from the audio. The sidecar also notes the dynamic labels seen while the title
recorded, and where the file lies in the unbroken stream it came from: the cutter
(`recording.cutter`) puts a title and its neighbours side by side by that, reads them
here (`load_stretch`) and has a part of them written as a kept title (`save_cut`).

`TitleRecorder` lives in a thread of its own (`ui.receiver_controller`): a write to a
USB stick can stall for a moment, and neither the signal processing nor the surface
must wait for it. Everything reaches it through queued signals; `changed` and
`status_changed` carry snapshots back.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from dab.dls import DynamicLabel
from dsp.mp2_frame import Mp2Format, Mp2FrameResult
from dsp.superframe import AudioFormat, SuperframeResult
from i18n import tr, tr_noop
from recording import cutter
from recording.adts import adts_header, split_adts
from recording.cutter import CutRequest
from recording.mp4 import SAMPLES_PER_UNIT, write_m4a
from recording.playback import PlaybackError, RecordingAudio, read_m4a, read_mp2
from recording.titles import TitleInfo, TitleTracker

logger = logging.getLogger(__name__)

RING_SIZE_RANGE = (3, 30)
DEFAULT_RING_SIZE = 10
# How late the dynamic label runs against the audio at a title boundary, in seconds:
# positive where the text comes after the audio changed (many stations delay their
# metadata to match their web streams, seen as some 30 s on 09.10.2026, and unevenly),
# negative where it comes before. A title then begins that much before (or after) its
# label; the title before keeps its end all the same, so the files overlap and nothing
# is cut off however much the delay varies.
LABEL_DELAY_RANGE_S = (-60, 60)
DEFAULT_LABEL_DELAY_S = 0
# How the title before ends against the label of the next, in seconds: positive = cut
# short that much before the label, negative = running on that much after it (its file
# then records on beside the next, `TitleRecorder._closing`); the same range.
DEFAULT_STOP_DELAY_S = 0
# A title longer than this is cut into files of this length (talk programmes).
MAX_TITLE_SECONDS = 20 * 60
# Shorter than this is no title worth keeping.
MIN_TITLE_SECONDS = 20
# Below this much free space the ring stops writing.
MIN_FREE_BYTES = 200_000_000
BUFFER_DIRECTORY = "Puffer"
# The file of the title recording is flushed to the system this often; the kernel
# writes it to the drive within its usual half minute.
FLUSH_INTERVAL_S = 5.0
SPACE_CHECK_INTERVAL_S = 60.0
# How often at most the status with the running title goes to the surface.
STATUS_INTERVAL_S = 1.0
# How many dynamic labels a file notes at most (`Recording.labels`): a station that
# alternates three texts every 8 s fills 450 in its 20 min.
MAX_LABELS = 1000
SIDECAR_SUFFIX = ".json"
CODEC_DAB_PLUS = "DAB+"
CODEC_DAB = "DAB"
MP2_SAMPLES_PER_FRAME = 1152
# What no file system among FAT, exFAT, NTFS and ext4 takes in a name, and the length
# a name part is cut to.
_FORBIDDEN = set('<>:"/\\|?*')
MAX_NAME_PART = 60

# What the recorder is doing, or why not; translated when set (`RecorderStatus.message`).
NO_DIRECTORY = tr_noop("No recording folder chosen")
DIRECTORY_MISSING = tr_noop("Recording folder unavailable")
NO_SPACE = tr_noop("Not enough space on the drive")
NO_STATION = tr_noop("No station chosen")
WAITING = tr_noop("Waiting for audio")
# Translators: the recorder's status, followed by the title it records:
# "Now recording: Artist – Title · 3:20".
RECORDING = tr_noop("Now recording")
DISABLED = tr_noop("Recording off")


@dataclass(frozen=True)
class Station:
    """The service recorded, as the surface names it."""

    channel: str
    sid: int
    label: str
    ensemble: str


@dataclass(frozen=True)
class Recording:
    """A title in the ring, kept, or recording right now (`complete` False)."""

    id: str  # the stem of the file, unique
    path: Path  # the audio file
    kept: bool
    complete: bool
    codec: str  # CODEC_DAB_PLUS or CODEC_DAB
    station: str
    ensemble: str
    channel: str
    sid: int
    title: str
    artist: str
    text: str  # the dynamic label last seen
    start: float  # Unix seconds
    duration: float  # seconds of audio in the file
    size: int  # bytes
    lost: int  # access units or frames that did not make it
    bitrate: int  # kbit/s
    audio_format: dict  # the fields of the AudioFormat or Mp2Format, for the remux
    # The unbroken stream the file is part of and the number of its first unit in it:
    # where it lies against its neighbours, to the unit (`recording.cutter`). Files of
    # one stream follow each other at a title boundary or after `MAX_TITLE_SECONDS`;
    # "" in files of earlier versions of the radio, which did not note it.
    session: str = ""
    first_unit: int = 0
    cut: bool = False  # cut out of recordings by the cutter
    # The dynamic labels seen while it recorded: (seconds into the file where the audio
    # changed, as far as the label delay says, text, DL Plus artist and title).
    labels: tuple[tuple[float, str, str, str], ...] = ()

    @property
    def name(self) -> str:
        """ "Artist – Title", one of them, the text, or "Untitled"."""
        if self.title and self.artist:
            return f"{self.artist} – {self.title}"
        return self.title or self.artist or self.text or tr("Untitled")

    @property
    def sidecar(self) -> Path:
        return self.path.with_suffix(SIDECAR_SUFFIX)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["path"] = self.path.name
        return data

    @classmethod
    def from_dict(cls, data: dict, directory: Path) -> Recording:
        """The recording a sidecar holds; what older sidecars lack takes its default,
        what a required field lacks raises `TypeError`."""
        fields = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        fields["path"] = directory / str(data.get("path", ""))
        fields["labels"] = tuple(
            (float(at), str(text), str(artist), str(title))
            for at, text, artist, title in data.get("labels", ())
        )
        return cls(**fields)


@dataclass(frozen=True)
class RecorderStatus:
    recording: bool
    message: str  # translated, for the surface
    current: Recording | None = None  # the title recording right now


def safe_name(text: str) -> str:
    """`text` as a part of a file name: what no file system takes replaced, cut to
    `MAX_NAME_PART` characters."""
    cleaned = "".join("_" if c in _FORBIDDEN or ord(c) < 32 else c for c in text)
    cleaned = " ".join(cleaned.split()).strip(". ")
    return cleaned[:MAX_NAME_PART].rstrip(". ")


def unique_path(path: Path) -> Path:
    """`path`, or with " (2)", " (3)" … where a file of the name exists."""
    candidate = path
    number = 2
    while candidate.exists() or candidate.with_suffix(SIDECAR_SUFFIX).exists():
        candidate = path.with_name(f"{path.stem} ({number}){path.suffix}")
        number += 1
    return candidate


def final_stem(start: float, station: str, info: TitleInfo) -> str:
    """ "2026-10-09 18.32 Dlf – A-HA – TAKE ON ME"."""
    parts = [time.strftime("%Y-%m-%d %H.%M", time.localtime(start)) + " " + safe_name(station)]
    for part in (info.artist, info.title):
        if part:
            parts.append(safe_name(part))
    if not info.named and info.text:
        parts.append(safe_name(info.text))
    return " – ".join(part for part in parts if part)


def free_bytes(directory: Path) -> int | None:
    try:
        return shutil.disk_usage(directory).free
    except OSError:
        return None


def load_recording(sidecar: Path, kept: bool) -> Recording | None:
    """The recording a sidecar describes; None where it cannot be read."""
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        recording = Recording.from_dict(data, sidecar.parent)
    except (OSError, ValueError, TypeError, KeyError) as error:
        logger.warning("Cannot read %s: %s", sidecar, error)
        return None
    return replace(recording, kept=kept)


def write_sidecar(recording: Recording) -> None:
    text = json.dumps(recording.to_dict(), ensure_ascii=False, indent=2)
    recording.sidecar.write_text(text, encoding="utf-8")


def remove_files(recording: Recording) -> None:
    for path in (recording.path, recording.sidecar):
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            logger.warning("Cannot remove %s: %s", path, error)


def _read_units(recording: Recording) -> tuple[AudioFormat | Mp2Format, tuple[bytes, ...]]:
    """The format and the units of a complete title, read whole (`recording.playback`).

    Raises:
        OSError, PlaybackError, TypeError, ValueError: where it cannot be read.
    """
    if recording.codec == CODEC_DAB_PLUS:
        return read_m4a(recording.path)
    audio_format = Mp2Format(**recording.audio_format)
    return audio_format, read_mp2(recording.path, audio_format)


class _Segment:
    """A title recording: its file, what is known of it and how much is in it."""

    def __init__(
        self,
        path: Path,
        codec: str,
        station: Station,
        start: float,
        info: TitleInfo,
        audio_format: AudioFormat | Mp2Format,
        bitrate: int,
        opened_at: float,
        session: str = "",
        first_unit: int = 0,
    ) -> None:
        self.path = path
        self.file = open(path, "wb")  # noqa: SIM115 - closed by `close`
        self.codec = codec
        self.station = station
        self.start = start
        self.info = info
        self.audio_format = audio_format
        self.bitrate = bitrate
        self.session = session
        self.first_unit = first_unit
        self.labels: list[tuple[float, str, str, str]] = []
        self.units = 0
        self.lost = 0
        self.size = 0
        self.keep = False  # to be kept as soon as it is complete
        self.last_flush = opened_at
        self.last_space_check = opened_at
        # The last seconds written, as (data, units), for a title that begins before
        # its label (`tail_copy`); and a boundary to come once this many units are written
        # (`pending`: the info of the next title, the units, the seconds waited), for
        # one moved forward.
        self.tail: deque[tuple[bytes, int]] = deque()
        self.tail_units = 0
        self.pending: tuple[TitleInfo, int, float] | None = None

    @property
    def units_per_second(self) -> float:
        if self.codec == CODEC_DAB_PLUS:
            return self.audio_format.core_sample_rate / SAMPLES_PER_UNIT
        return self.audio_format.sample_rate / MP2_SAMPLES_PER_FRAME

    @property
    def duration(self) -> float:
        return self.units / self.units_per_second

    def write(self, data: bytes, units: int = 0, keep_seconds: float = 0.0) -> None:
        """Writes `data` holding `units` access units or frames, keeping the last
        `keep_seconds` of what was written in memory."""
        self.file.write(data)
        self.size += len(data)
        self.units += units
        if keep_seconds <= 0:
            self.tail.clear()
            self.tail_units = 0
            return
        self.tail.append((data, units))
        self.tail_units += units
        kept = keep_seconds * self.units_per_second
        while self.tail and self.tail_units - self.tail[0][1] >= kept:
            self.tail_units -= self.tail.popleft()[1]

    def tail_copy(self, seconds: float) -> list[tuple[bytes, int]]:
        """The last `seconds` written (as far as kept), oldest first: they go to the
        beginning of the title that begins as well; this file keeps them."""
        wanted = seconds * self.units_per_second
        copied: list[tuple[bytes, int]] = []
        copied_units = 0
        for chunk in reversed(self.tail):
            if copied_units >= wanted:
                break
            copied.insert(0, chunk)
            copied_units += chunk[1]
        return copied

    def cut_tail(self, seconds: float) -> None:
        """Takes the last `seconds` (as far as kept) out of the file again: the title
        ends that much before the label that named the next (`set_stop_delay`)."""
        wanted = seconds * self.units_per_second
        cut_units = 0
        length = 0
        while self.tail and cut_units < wanted:
            data, units = self.tail.pop()
            cut_units += units
            length += len(data)
        self.tail_units -= cut_units
        if length:
            self.file.flush()
            self.file.truncate(self.size - length)
            self.file.seek(self.size - length)
            self.size -= length
            self.units -= cut_units

    def note_label(self, at: float, text: str, artist: str, title: str) -> None:
        """Notes a dynamic label where it differs from the last noted."""
        label = (round(at, 3), text, artist, title)
        if self.labels and self.labels[-1][1:] == label[1:]:
            return
        if len(self.labels) < MAX_LABELS:
            self.labels.append(label)

    def close(self) -> None:
        self.file.close()

    def recording(self, complete: bool = False) -> Recording:
        return Recording(
            id=self.path.stem,
            path=self.path,
            kept=self.keep,
            complete=complete,
            codec=self.codec,
            station=self.station.label,
            ensemble=self.station.ensemble,
            channel=self.station.channel,
            sid=self.station.sid,
            title=self.info.title,
            artist=self.info.artist,
            text=self.info.text,
            start=self.start,
            duration=self.duration,
            size=self.size,
            lost=self.lost,
            bitrate=self.bitrate,
            audio_format=asdict(self.audio_format),
            session=self.session,
            first_unit=self.first_unit,
            labels=tuple(self.labels),
        )


class TitleRecorder(QObject):
    """Records the station playing into the ring; see the module docstring."""

    changed = pyqtSignal(object, object)  # the ring and the kept titles, tuples of Recording
    status_changed = pyqtSignal(object)  # RecorderStatus
    # A title read for playback (`load`): the audio, or the id and why not (translated).
    loaded = pyqtSignal(object)  # recording.playback.RecordingAudio
    load_failed = pyqtSignal(str, str)
    # A title read with its neighbours for the cutter (`load_stretch`), or its id and
    # why not (translated).
    stretch_loaded = pyqtSignal(object)  # recording.cutter.Stretch
    stretch_failed = pyqtSignal(str, str)
    # A cut written (`save_cut`): the kept title it became, and the id of the one it
    # replaced ("" for none); or why not (translated).
    cut_saved = pyqtSignal(object, str)  # Recording, id replaced
    cut_failed = pyqtSignal(str)

    def __init__(
        self,
        parent: QObject | None = None,
        now: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(parent)
        self._now = now
        self._monotonic = monotonic
        self._directory: Path | None = None
        self._ring_size = DEFAULT_RING_SIZE
        self._label_delay = DEFAULT_LABEL_DELAY_S
        self._stop_delay = DEFAULT_STOP_DELAY_S
        self._enabled = True
        self._station: Station | None = None
        self._tracker = TitleTracker(monotonic)
        self._ring: list[Recording] = []  # complete, in the buffer, oldest first
        self._kept: list[Recording] = []
        # Whether the titles in the directory have been read (`_read_directory`).
        self._scanned = False
        self._segment: _Segment | None = None
        # The title that ends after its label (`set_stop_delay` negative): its file
        # records on beside the next for this many units more.
        self._closing: _Segment | None = None
        self._closing_units_left = 0
        # The unbroken stream recording (`Recording.session`) and how many units it
        # brought so far: a title boundary or `MAX_TITLE_SECONDS` keep it, anything that
        # breaks the audio (another station, a format, a stop) begins another.
        self._session = ""
        self._stream_units = 0
        self._error = ""
        self._status = RecorderStatus(False, tr(NO_DIRECTORY))
        self._status_sent_at = 0.0

    @property
    def directory(self) -> Path | None:
        return self._directory

    @property
    def ring(self) -> tuple[Recording, ...]:
        return tuple(self._ring)

    @property
    def kept(self) -> tuple[Recording, ...]:
        return tuple(self._kept)

    @property
    def status(self) -> RecorderStatus:
        return self._status

    @property
    def buffer_directory(self) -> Path | None:
        return None if self._directory is None else self._directory / BUFFER_DIRECTORY

    # --- Settings and station ---------------------------------------------------------

    @pyqtSlot(object)
    def set_directory(self, directory: object) -> None:
        """Records into this directory from now on; reads the titles already there, or
        as soon as it is there."""
        self._close_segment()
        self._directory = Path(directory) if directory else None
        self._error = ""
        self._read_directory()
        self._update_status()

    def _read_directory(self) -> None:
        """Reads the titles in the directory afresh, if it is there; otherwise the
        first title recorded after it came does (`_start_segment`). On the Pi the radio
        starts before its drive is mounted, and a drive pulled may come back with the
        files of a title it broke off."""
        self._ring = []
        self._kept = []
        self._scanned = self._directory is not None and self._directory.is_dir()
        if self._scanned:
            self._scan()
        self.changed.emit(self.ring, self.kept)

    @pyqtSlot(int)
    def set_ring_size(self, size: int) -> None:
        self._ring_size = max(RING_SIZE_RANGE[0], min(int(size), RING_SIZE_RANGE[1]))
        if self._trim_ring():
            self.changed.emit(self.ring, self.kept)

    @pyqtSlot(int)
    def set_label_delay(self, seconds: int) -> None:
        """How late the label runs against the audio (`LABEL_DELAY_RANGE_S`)."""
        low, high = LABEL_DELAY_RANGE_S
        self._label_delay = max(low, min(int(seconds), high))

    @pyqtSlot(int)
    def set_stop_delay(self, seconds: int) -> None:
        """How the title before ends against the label of the next: cut short that many
        seconds before it, or negative, running on that much after it
        (`LABEL_DELAY_RANGE_S`)."""
        low, high = LABEL_DELAY_RANGE_S
        self._stop_delay = max(low, min(int(seconds), high))

    @pyqtSlot(bool)
    def set_enabled(self, enabled: bool) -> None:
        """Switched off, nothing is written and the title recording ends."""
        self._enabled = bool(enabled)
        if not self._enabled:
            self._close_segment()
        self._update_status(force=True)

    @pyqtSlot(object)
    def set_station(self, station: object) -> None:
        """The service recorded from now on (a `Station`, None for none). A new name
        for the same service only renames what records."""
        if station is not None and not isinstance(station, Station):
            return
        current = self._station
        same = (
            station is not None
            and current is not None
            and (station.channel, station.sid) == (current.channel, current.sid)
        )
        self._station = station
        if same:
            if self._segment is not None:
                self._segment.station = station
                self._set_status(True, tr(RECORDING), force=True)
        else:
            self._close_segment()
            self._tracker.reset()
        self._update_status()

    @pyqtSlot()
    def reset(self) -> None:
        """Another service is coming: the title ends, the labels so far are void."""
        self._close_segment()
        self._tracker.reset()
        self._update_status()

    @pyqtSlot()
    def stop(self) -> None:
        """Ends the title recording and keeps it (on the exit of the radio)."""
        self._close_segment()
        self._update_status(force=True)

    # --- The stream -------------------------------------------------------------------

    @pyqtSlot(object)
    def process_superframe(self, result: SuperframeResult) -> None:
        if self._station is None or self._directory is None or not self._enabled:
            return
        audio_format = result.audio_format
        segment = self._segment
        if audio_format is None:
            if segment is not None and segment.codec == CODEC_DAB_PLUS:
                segment.lost += segment.audio_format.num_audio_units
                self._set_status(True, tr(RECORDING))
            return
        if segment is not None and (
            segment.codec != CODEC_DAB_PLUS or segment.audio_format != audio_format
        ):
            self._close_segment()
            segment = None
        if segment is None:
            bitrate = round(result.total_length * 8 / 120) if result.total_length else 0
            segment = self._start_segment(CODEC_DAB_PLUS, audio_format, bitrate)
            if segment is None:
                return
        segment.lost += max(result.num_audio_units - len(result.audio_units), 0)
        core_rate = audio_format.core_sample_rate
        channels = 2 if audio_format.stereo_core else 1
        data = b"".join(
            adts_header(core_rate, channels, len(unit)) + unit for unit in result.audio_units
        )
        try:
            segment.write(data, len(result.audio_units), self._tail_seconds())
            self._write_closing(data, len(result.audio_units))
        except OSError as error:
            self._fail(error)
            return
        self._stream_units += len(result.audio_units)
        self._after_write(segment)

    @pyqtSlot(object)
    def process_mp2_frame(self, result: Mp2FrameResult) -> None:
        if self._station is None or self._directory is None or not self._enabled:
            return
        segment = self._segment
        if not result.sync_ok or not result.frame:
            if segment is not None and segment.codec == CODEC_DAB:
                segment.lost += 1
                self._set_status(True, tr(RECORDING))
            return
        if segment is not None and (
            segment.codec != CODEC_DAB or segment.audio_format != result.audio_format
        ):
            self._close_segment()
            segment = None
        if segment is None:
            segment = self._start_segment(
                CODEC_DAB, result.audio_format, result.audio_format.bitrate
            )
            if segment is None:
                return
        try:
            segment.write(result.frame, 1, self._tail_seconds())
            self._write_closing(result.frame, 1)
        except OSError as error:
            self._fail(error)
            return
        self._stream_units += 1
        self._after_write(segment)

    @pyqtSlot(object)
    def process_dynamic_label(self, label: DynamicLabel) -> None:
        new_item = self._tracker.update(label)
        segment = self._segment
        if segment is None:
            return
        self._follow_label(segment, new_item)
        self._note_label(label)

    def _follow_label(self, segment: _Segment, new_item: TitleInfo | None) -> None:
        # The first label after the title began (the station was just chosen) names
        # what records rather than beginning anything: the audio came first.
        known = segment.info.named or bool(segment.info.text)
        if new_item is not None and known:
            if self._label_delay < 0:
                # The label runs ahead of the audio: the title begins that much later.
                waited = float(-self._label_delay)
                units = segment.units + round(waited * segment.units_per_second)
                segment.pending = (new_item, units, waited)
                return
            self._split(segment)
        elif segment.pending is None:
            current = self._tracker.current
            if current is not None and current != segment.info:
                segment.info = current
                self._set_status(True, tr(RECORDING), force=True)

    def _note_label(self, label: DynamicLabel) -> None:
        """Notes the label in the title recording (`Recording.labels`), where the audio
        it tells of lies as far as the label delay says: so many seconds before the
        label came, or after it."""
        segment = self._segment
        if segment is None:
            return
        text = " ".join(label.text.split())
        dl_plus = label.dl_plus
        running = dl_plus is not None and dl_plus.item_running
        artist = (dl_plus.artist or "").strip() if running else ""
        title = (dl_plus.title or "").strip() if running else ""
        segment.note_label(segment.duration - self._label_delay, text, artist, title)

    def _split(self, segment: _Segment, waited: float = 0.0) -> None:
        """Ends the title and begins the next at the boundary the label gave, `waited`
        seconds ago where the label ran ahead (`_Segment.pending`). The next begins
        `_label_delay` seconds before the label, with a copy of the audio since; the
        one that ends is cut short `_stop_delay` seconds before the label, or runs on
        that much after it in a file of its own (`_closing`)."""
        audio_format, codec, bitrate = segment.audio_format, segment.codec, segment.bitrate
        start_before = max(0.0, float(self._label_delay))
        head = segment.tail_copy(start_before) if start_before > 0 else []
        head_units = sum(units for _data, units in head)
        taken = head_units / segment.units_per_second
        cut = self._stop_delay + waited
        if cut < 0:
            self._finish_closing()
            self._segment = None
            self._closing = segment
            self._closing_units_left = round(-cut * segment.units_per_second)
        else:
            if cut > 0:
                try:
                    segment.cut_tail(cut)
                except OSError as error:
                    self._fail(error)
                    return
            self._close_segment()
        fresh = self._start_segment(
            codec, audio_format, bitrate, self._now() - taken, continues=True, back=head_units
        )
        if fresh is None:
            return
        try:
            for data, units in head:
                fresh.write(data, units, self._tail_seconds())
        except OSError as error:
            self._fail(error)
            return
        self._set_status(True, tr(RECORDING), force=True)

    def _tail_seconds(self) -> float:
        """How many of the last seconds written the title recording keeps in memory:
        what the next title takes over when it begins before its label, and what a cut
        at the end may take away, after a wait for a label running ahead."""
        return max(0.0, self._label_delay, self._stop_delay + max(0, -self._label_delay))

    def _write_closing(self, data: bytes, units: int) -> None:
        """The title that ends after its label runs on: the same audio goes into its
        file until its seconds are up."""
        closing = self._closing
        if closing is None:
            return
        closing.write(data, units)
        self._closing_units_left -= units
        if self._closing_units_left <= 0:
            self._finish_closing()

    # --- Keeping and deleting ---------------------------------------------------------

    @pyqtSlot(str)
    def keep(self, recording_id: str) -> None:
        """Moves a title of the ring into the directory, where the ring leaves it; the
        title recording is marked and moved once it is complete."""
        segment = self._segment
        if segment is not None and segment.path.stem == recording_id:
            segment.keep = True
            self._set_status(True, tr(RECORDING), force=True)
            return
        closing = self._closing
        if closing is not None and closing.path.stem == recording_id:
            closing.keep = True
            return
        recording = next((r for r in self._ring if r.id == recording_id), None)
        if recording is None:
            return
        kept = self._move_to_kept(recording)
        if kept is None:
            return
        self._ring.remove(recording)
        self._add_kept(kept)

    def _move_to_kept(self, recording: Recording) -> Recording | None:
        if self._directory is None:
            return None
        target = unique_path(self._directory / recording.path.name)
        kept = replace(recording, id=target.stem, path=target, kept=True)
        try:
            recording.path.rename(target)
            write_sidecar(kept)
            recording.sidecar.unlink(missing_ok=True)
        except OSError as error:
            self._fail(error)
            return None
        return kept

    def _add_kept(self, kept: Recording) -> None:
        self._kept.append(kept)
        self._kept.sort(key=lambda r: r.start)
        self.changed.emit(self.ring, self.kept)

    @pyqtSlot(str)
    def delete(self, recording_id: str) -> None:
        for recordings in (self._ring, self._kept):
            recording = next((r for r in recordings if r.id == recording_id), None)
            if recording is not None:
                remove_files(recording)
                recordings.remove(recording)
                self.changed.emit(self.ring, self.kept)
                return

    # --- Playback ---------------------------------------------------------------------

    @pyqtSlot(str)
    def load(self, recording_id: str) -> None:
        """Reads a complete title of the ring or a kept one for playback
        (`recording.playback`), the whole file at once, and hands it on with `loaded`;
        `load_failed` says why not. Here, in the recorder's thread, so that neither the
        surface nor the audio waits for the stick, and after a packing under way."""
        recording = self._find(recording_id)
        if recording is None:
            self.load_failed.emit(recording_id, tr("The recording no longer exists"))
            return
        try:
            audio_format, units = _read_units(recording)
        except (OSError, PlaybackError, TypeError, ValueError) as error:
            logger.warning("Cannot read %s for playback: %s", recording.path, error)
            self.load_failed.emit(recording_id, tr("The recording cannot be read"))
            return
        if not units:
            self.load_failed.emit(recording_id, tr("The recording is empty"))
            return
        self.loaded.emit(RecordingAudio(recording_id, audio_format, units))

    def _find(self, recording_id: str) -> Recording | None:
        return next((r for r in (*self._ring, *self._kept) if r.id == recording_id), None)

    # --- Cutting ----------------------------------------------------------------------

    @pyqtSlot(str)
    def load_stretch(self, recording_id: str) -> None:
        """Reads a complete title of the ring or a kept one with its neighbours for the
        cutter (`recording.cutter`) and hands the stretch on with `stretch_loaded`;
        `stretch_failed` says why not. A neighbour that cannot be read is left out."""
        opened = self._find(recording_id)
        if opened is None:
            self.stretch_failed.emit(recording_id, tr("The recording no longer exists"))
            return
        try:
            audio_format, units = _read_units(opened)
        except (OSError, PlaybackError, TypeError, ValueError) as error:
            logger.warning("Cannot read %s for cutting: %s", opened.path, error)
            self.stretch_failed.emit(recording_id, tr("The recording cannot be read"))
            return
        if not units:
            self.stretch_failed.emit(recording_id, tr("The recording is empty"))
            return
        seconds_per_unit = cutter.unit_seconds(audio_format)
        pieces = [(opened, units)]
        for neighbour in cutter.neighbours(opened, (*self._ring, *self._kept), seconds_per_unit):
            if neighbour is None:
                continue
            try:
                _format, neighbour_units = _read_units(neighbour)
            except (OSError, PlaybackError, TypeError, ValueError) as error:
                logger.warning("Cannot read the neighbour %s: %s", neighbour.path, error)
                continue
            if neighbour_units:
                pieces.append((neighbour, neighbour_units))
        self.stretch_loaded.emit(cutter.build_stretch(opened, pieces, audio_format))

    @pyqtSlot(object)
    def save_cut(self, request: CutRequest) -> None:
        """Writes what a cut keeps as a title of its own into the directory, among the
        kept titles: the units as they are, an `.m4a` or `.mp2` with its sidecar. The
        title `request.replaces` names goes, the ring stays as it is."""
        directory = self._directory
        if directory is None or not directory.is_dir():
            self.cut_failed.emit(tr(DIRECTORY_MISSING))
            return
        stretch = request.stretch
        opened = stretch.opened
        units = request.units
        if not units:
            self.cut_failed.emit(tr("Nothing to keep"))
            return
        info = TitleInfo(request.title, request.artist, request.text)
        suffix = ".m4a" if opened.codec == CODEC_DAB_PLUS else ".mp2"
        wanted = directory / (final_stem(request.start, opened.station, info) + suffix)
        old = next((r for r in self._kept if r.id == request.replaces), None)
        # A title replaced under the same name: the cut goes beside it until it is gone.
        same_name = old is not None and old.path == wanted
        path = wanted.with_name(wanted.stem + ".cut" + suffix) if same_name else unique_path(wanted)
        try:
            if opened.codec == CODEC_DAB_PLUS:
                tags = {
                    "title": request.title or request.text,
                    "artist": request.artist,
                    "album": opened.station,
                    "date": time.strftime("%Y-%m-%d", time.localtime(request.start)),
                }
                duration = write_m4a(
                    path,
                    units,
                    stretch.audio_format,
                    tags,
                    opened.bitrate * 1000,
                    request.start,
                )
            else:
                path.write_bytes(b"".join(units))
                duration = request.duration
            if old is not None:
                remove_files(old)
                self._kept.remove(old)
            if same_name:
                path = path.replace(wanted)
            kept = replace(
                opened,
                id=path.stem,
                path=path,
                kept=True,
                complete=True,
                title=request.title,
                artist=request.artist,
                text=request.text,
                start=request.start,
                duration=duration,
                size=path.stat().st_size,
                lost=0,
                session=stretch.session,
                first_unit=stretch.first_unit + request.first if stretch.session else 0,
                cut=True,
                labels=request.labels,
            )
            write_sidecar(kept)
        except OSError as error:
            logger.warning("Cannot save the cut of %s: %s", opened.id, error)
            self.cut_failed.emit(
                tr("Cannot save the title: {error}").format(error=error.strerror or error)
            )
            if old is None or old in self._kept:
                path.unlink(missing_ok=True)  # broken off before it replaced anything
            else:
                self.changed.emit(self.ring, self.kept)  # the old one went all the same
            return
        self._add_kept(kept)
        self.cut_saved.emit(kept, "" if old is None else old.id)

    # --- Segments ---------------------------------------------------------------------

    def _start_segment(
        self,
        codec: str,
        audio_format: AudioFormat | Mp2Format,
        bitrate: int,
        start: float | None = None,
        continues: bool = False,
        back: int = 0,
    ) -> _Segment | None:
        """Begins a title. `continues`: it follows the one before in the same stream
        (a title boundary, `MAX_TITLE_SECONDS`), else a stream begins; `back`: it begins
        with that many units of the stream already written (`_split`)."""
        directory, station = self._directory, self._station
        if directory is None or station is None:
            return None
        if not directory.is_dir():
            self._scanned = False
            self._set_status(False, tr(DIRECTORY_MISSING))
            return None
        if not self._scanned:
            # The directory came (or came back): what lies there is read before this
            # title writes its own sidecar there.
            self._finish_closing()
            self._read_directory()
        free = free_bytes(directory)
        if free is not None and free < MIN_FREE_BYTES:
            self._set_status(False, tr(NO_SPACE))
            return None
        start = self._now() if start is None else start
        if not continues or not self._session:
            self._session = f"{start:.3f}"
            self._stream_units = 0
        info = self._tracker.current or TitleInfo()
        suffix = ".aac" if codec == CODEC_DAB_PLUS else ".mp2"
        stem = (
            time.strftime("%Y-%m-%d %H.%M.%S", time.localtime(start))
            + " "
            + safe_name(station.label)
        )
        try:
            buffer = directory / BUFFER_DIRECTORY
            buffer.mkdir(exist_ok=True)
            path = unique_path(buffer / (stem.strip() + suffix))
            segment = _Segment(
                path,
                codec,
                station,
                start,
                info,
                audio_format,
                bitrate,
                self._monotonic(),
                self._session,
                self._stream_units - back,
            )
            if not continues and info.text:
                # The label already there tells of what records from the first unit on.
                tags = (info.artist, info.title) if info.tagged else ("", "")
                segment.note_label(0.0, info.text, *tags)
            write_sidecar(segment.recording())
        except OSError as error:
            self._fail(error)
            return None
        self._segment = segment
        self._error = ""
        self._set_status(True, tr(RECORDING), force=True)
        return segment

    def _after_write(self, segment: _Segment) -> None:
        now = self._monotonic()
        if now - segment.last_flush >= FLUSH_INTERVAL_S:
            segment.last_flush = now
            try:
                segment.file.flush()
            except OSError as error:
                self._fail(error)
                return
        if now - segment.last_space_check >= SPACE_CHECK_INTERVAL_S:
            segment.last_space_check = now
            free = free_bytes(segment.path.parent)
            if free is not None and free < MIN_FREE_BYTES:
                self._close_segment()
                self._set_status(False, tr(NO_SPACE), force=True)
                return
        if segment.pending is not None and segment.units >= segment.pending[1]:
            self._split(segment, segment.pending[2])
            return
        if segment.duration >= MAX_TITLE_SECONDS:
            audio_format, codec, bitrate = segment.audio_format, segment.codec, segment.bitrate
            self._close_segment()
            self._start_segment(codec, audio_format, bitrate, continues=True)
            return
        self._set_status(True, tr(RECORDING))

    def _close_segment(self) -> None:
        """Ends the title recording and one running on: kept in the ring where long
        enough, else thrown away."""
        self._finish_closing()
        segment, self._segment = self._segment, None
        if segment is not None:
            self._finish(segment)

    def _finish_closing(self) -> None:
        closing, self._closing = self._closing, None
        if closing is not None:
            self._finish(closing)

    def _finish(self, segment: _Segment) -> None:
        try:
            segment.close()
        except OSError as error:
            logger.warning("Cannot close %s: %s", segment.path, error)
        recording = segment.recording()
        if recording.duration < MIN_TITLE_SECONDS:
            remove_files(recording)
            return
        complete = self._finalize(recording)
        if complete is None:
            return
        if segment.keep:
            kept = self._move_to_kept(complete)
            if kept is not None:
                self._add_kept(kept)
            return
        self._ring.append(complete)
        self._ring.sort(key=lambda r: r.start)
        self._trim_ring()
        self.changed.emit(self.ring, self.kept)

    def _finalize(self, recording: Recording) -> Recording | None:
        """The complete file of a title: the `.m4a` packed from the ADTS stream, or
        the `.mp2` under its name; None where that failed."""
        try:
            info = TitleInfo(recording.title, recording.artist, recording.text)
            stem = final_stem(recording.start, recording.station, info)
            if recording.codec == CODEC_DAB_PLUS:
                audio_format = AudioFormat(**recording.audio_format)
                units = split_adts(recording.path.read_bytes())
                if not units:
                    remove_files(recording)
                    return None
                target = unique_path(recording.path.with_name(stem + ".m4a"))
                tags = {
                    "title": recording.title or recording.text,
                    "artist": recording.artist,
                    "album": recording.station,
                    "date": time.strftime("%Y-%m-%d", time.localtime(recording.start)),
                }
                duration = write_m4a(
                    target, units, audio_format, tags, recording.bitrate * 1000, recording.start
                )
                recording.path.unlink(missing_ok=True)
            else:
                target = unique_path(recording.path.with_name(stem + ".mp2"))
                recording.path.rename(target)
                duration = recording.duration
            recording.sidecar.unlink(missing_ok=True)
            complete = replace(
                recording,
                id=target.stem,
                path=target,
                complete=True,
                duration=duration,
                size=target.stat().st_size,
            )
            write_sidecar(complete)
        except OSError as error:
            self._fail(error)
            return None
        return complete

    def _trim_ring(self) -> bool:
        """Drops the oldest titles beyond the ring's size; whether any went."""
        dropped = False
        while len(self._ring) > self._ring_size:
            remove_files(self._ring.pop(0))
            dropped = True
        return dropped

    def _scan(self) -> None:
        """Reads the titles in the buffer and the kept ones from their sidecars; a
        title left recording when the radio ended (the power went) is completed."""
        buffer = self.buffer_directory
        assert self._directory is not None and buffer is not None
        try:
            buffered = sorted(buffer.glob("*" + SIDECAR_SUFFIX)) if buffer.is_dir() else []
            kept = sorted(self._directory.glob("*" + SIDECAR_SUFFIX))
        except OSError as error:
            self._fail(error)
            return
        for sidecar in buffered:
            recording = load_recording(sidecar, kept=False)
            if recording is None:
                continue
            if not recording.complete:
                recording = self._complete_leftover(recording)
                if recording is None:
                    continue
            if recording.path.is_file():
                self._ring.append(recording)
        for sidecar in kept:
            recording = load_recording(sidecar, kept=True)
            if recording is not None and recording.path.is_file():
                self._kept.append(recording)
        self._ring.sort(key=lambda r: r.start)
        self._kept.sort(key=lambda r: r.start)
        self._trim_ring()

    def _complete_leftover(self, recording: Recording) -> Recording | None:
        """A title whose recording broke off: what is in the file, if enough."""
        if not recording.path.is_file():
            remove_files(recording)
            return None
        try:
            if recording.codec == CODEC_DAB_PLUS:
                audio_format = AudioFormat(**recording.audio_format)
                units = len(split_adts(recording.path.read_bytes()))
                duration = units * SAMPLES_PER_UNIT / audio_format.core_sample_rate
            else:
                audio_format = Mp2Format(**recording.audio_format)
                frames = recording.path.stat().st_size // audio_format.frame_length
                duration = frames * MP2_SAMPLES_PER_FRAME / audio_format.sample_rate
        except (OSError, TypeError, ValueError, KeyError) as error:
            logger.warning("Cannot complete %s: %s", recording.path, error)
            remove_files(recording)
            return None
        if duration < MIN_TITLE_SECONDS:
            remove_files(recording)
            return None
        return self._finalize(replace(recording, duration=duration))

    # --- Status -----------------------------------------------------------------------

    def _fail(self, error: OSError) -> None:
        """A file operation failed (the stick pulled, the folder read-only): the title
        recording is lost, the next one is tried as usual."""
        logger.warning("Recording failed: %s", error)
        segments = (self._segment, self._closing)
        self._segment = self._closing = None
        for segment in segments:
            if segment is not None:
                try:
                    segment.close()
                except OSError:
                    pass
                remove_files(segment.recording())
        self._error = tr("Error while writing: {error}").format(error=error.strerror or error)
        self._set_status(False, self._error, force=True)

    def _update_status(self, force: bool = False) -> None:
        if self._segment is not None:
            self._set_status(True, tr(RECORDING), force)
        elif not self._enabled:
            self._set_status(False, tr(DISABLED), force)
        elif self._error:
            self._set_status(False, self._error, force)
        elif self._directory is None:
            self._set_status(False, tr(NO_DIRECTORY), force)
        elif not self._directory.is_dir():
            self._set_status(False, tr(DIRECTORY_MISSING), force)
        elif self._station is None:
            self._set_status(False, tr(NO_STATION), force)
        else:
            self._set_status(False, tr(WAITING), force)

    def _set_status(self, recording: bool, message: str, force: bool = False) -> None:
        current = None if self._segment is None else self._segment.recording()
        status = RecorderStatus(recording, message, current)
        now = self._monotonic()
        changed = (status.recording, status.message) != (
            self._status.recording,
            self._status.message,
        )
        if not changed and not force and now - self._status_sent_at < STATUS_INTERVAL_S:
            self._status = status
            return
        self._status = status
        self._status_sent_at = now
        self.status_changed.emit(status)
