"""The ring of the last titles (`recording.recorder`), on a temporary directory as the
USB drive: titles begin and end with the labels, short ones go, the ring drops the
oldest, "Behalten" moves a title out, and what happens without space or drive."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from dab.dls import DlPlus, DlPlusContentType, DynamicLabel
from dsp.mp2_frame import CHANNEL_MODE_JOINT_STEREO, Mp2Format, Mp2FrameResult
from dsp.superframe import AudioFormat, SuperframeResult
from recording import recorder as recorder_module
from recording.adts import split_adts
from recording.mp4 import read_boxes
from recording.playback import read_m4a
from recording.recorder import (
    BUFFER_DIRECTORY,
    CODEC_DAB,
    CODEC_DAB_PLUS,
    DIRECTORY_MISSING,
    DISABLED,
    NO_DIRECTORY,
    NO_SPACE,
    NO_STATION,
    RECORDING,
    WAITING,
    Recording,
    Station,
    TitleRecorder,
    final_stem,
    safe_name,
)
from recording.titles import TitleInfo

# SBR at 48 kHz: the core runs at 24 kHz, three units of 40 ms per superframe.
FORMAT = AudioFormat(
    dac_rate_48k=True, sbr=True, stereo_core=True, parametric_stereo=False, mpeg_surround_config=0
)
FORMAT_MONO = AudioFormat(
    dac_rate_48k=False, sbr=True, stereo_core=False, parametric_stereo=False, mpeg_surround_config=0
)
MP2 = Mp2Format(sample_rate=48_000, bitrate=128, channel_mode=CHANNEL_MODE_JOINT_STEREO)
STATION = Station("5C", 0xD210, "Dlf", "DR Deutschland")
OTHER_STATION = Station("5C", 0xD220, "Dlf Kultur", "DR Deutschland")
SUPERFRAME_S = 0.12


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


class _Clock:
    """Wall time and a monotonic one that the test moves."""

    def __init__(self) -> None:
        self.now = 1_791_000_000.0
        self.mono = 5000.0

    def __call__(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.mono

    def advance(self, seconds: float) -> None:
        self.now += seconds
        self.mono += seconds


class _Harness:
    def __init__(self, directory: Path) -> None:
        self.clock = _Clock()
        self.recorder = TitleRecorder(now=self.clock, monotonic=self.clock.monotonic)
        self.changes: list[tuple] = []
        self.statuses: list = []
        self.recorder.changed.connect(lambda ring, kept: self.changes.append((ring, kept)))
        self.recorder.status_changed.connect(self.statuses.append)
        self.directory = directory
        self.buffer = directory / BUFFER_DIRECTORY

    def feed(self, seconds: float, *, sync_ok: bool = True, audio_format=FORMAT) -> None:
        """Superframes worth `seconds` of audio."""
        for _ in range(round(seconds / SUPERFRAME_S)):
            self.recorder.process_superframe(
                _superframe(sync_ok=sync_ok, audio_format=audio_format)
            )
            self.clock.advance(SUPERFRAME_S)

    def feed_mp2(self, seconds: float) -> None:
        for _ in range(round(seconds / MP2.duration_seconds)):
            self.recorder.process_mp2_frame(
                Mp2FrameResult(MP2, sync_ok=True, crc_ok=True, frame=b"\xff\xfd" + bytes(382))
            )
            self.clock.advance(MP2.duration_seconds)

    def label(self, text: str, title: str | None = None, artist: str | None = None, running=True):
        objects = {}
        if title is not None:
            objects[DlPlusContentType.ITEM_TITLE] = title
        if artist is not None:
            objects[DlPlusContentType.ITEM_ARTIST] = artist
        dl_plus = DlPlus(False, running, objects) if objects or not running else None
        self.recorder.process_dynamic_label(DynamicLabel(text, dl_plus))

    @property
    def ring(self) -> tuple[Recording, ...]:
        return self.recorder.ring

    @property
    def status(self):
        return self.recorder.status


def _superframe(sync_ok: bool = True, audio_format=FORMAT) -> SuperframeResult:
    units = [bytes([0x21, 0x00]) + bytes(98)] * audio_format.num_audio_units
    return SuperframeResult(
        sync_ok=sync_ok,
        audio_units=units if sync_ok else [],
        num_audio_units=audio_format.num_audio_units,
        audio_format=audio_format if sync_ok else None,
        data_length=1440 - 48,
        total_length=1440,  # 96 kbit/s
    )


@pytest.fixture
def harness(tmp_path):
    harness = _Harness(tmp_path)
    harness.recorder.set_directory(tmp_path)
    harness.recorder.set_station(STATION)
    return harness


# --- Names ----------------------------------------------------------------------------


def test_names_take_only_what_every_file_system_takes():
    assert safe_name('AC/DC: "Back" <in> Black?') == "AC_DC_ _Back_ _in_ Black_"
    assert safe_name("  Ende mit Punkt. ") == "Ende mit Punkt"
    assert len(safe_name("x" * 100)) == 60
    stem = final_stem(1_791_000_000.0, "Dlf", TitleInfo("TAKE ON ME", "A-HA", "x", True))
    assert (
        stem
        == time.strftime("%Y-%m-%d %H.%M", time.localtime(1_791_000_000.0))
        + " Dlf – A-HA – TAKE ON ME"
    )
    assert final_stem(0, "Dlf", TitleInfo(text="Nachrichten")).endswith(" Dlf – Nachrichten")


# --- Recording ------------------------------------------------------------------------


def test_without_directory_or_station_nothing_is_written(tmp_path):
    harness = _Harness(tmp_path)
    recorder = harness.recorder
    assert recorder.status.message == NO_DIRECTORY
    recorder.set_station(STATION)
    harness.feed(1)
    assert list(tmp_path.iterdir()) == []
    recorder.set_directory(tmp_path)
    recorder.set_station(None)
    assert recorder.status.message == NO_STATION
    harness.feed(1)
    assert list(tmp_path.iterdir()) == []
    recorder.set_directory(tmp_path / "gone")
    assert recorder.status.message == DIRECTORY_MISSING


def test_a_title_records_as_adts_and_becomes_an_m4a_when_the_next_begins(harness):
    harness.label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA")
    harness.feed(30)

    running = harness.status
    assert running.recording and running.message == RECORDING
    assert running.current.title == "TAKE ON ME" and running.current.codec == CODEC_DAB_PLUS
    assert running.current.duration == pytest.approx(30, abs=SUPERFRAME_S)
    assert not running.current.complete
    assert running.current.path.suffix == ".aac" and running.current.path.parent == harness.buffer
    harness.recorder._segment.file.flush()  # as every few seconds of recording
    assert len(split_adts(running.current.path.read_bytes())) == 250 * 3
    sidecar = json.loads(running.current.sidecar.read_text(encoding="utf-8"))
    assert sidecar["complete"] is False and sidecar["audio_format"]["sbr"] is True

    harness.label("QUEEN - RADIO GA GA", "RADIO GA GA", "QUEEN")

    assert len(harness.ring) == 1
    title = harness.ring[0]
    assert title.complete and title.kept is False
    assert (
        title.path.name == final_stem(title.start, "Dlf", TitleInfo("TAKE ON ME", "A-HA")) + ".m4a"
    )
    assert title.id == title.path.stem
    assert title.duration == pytest.approx(30, abs=SUPERFRAME_S)
    assert title.bitrate == 96 and title.lost == 0 and title.size == title.path.stat().st_size
    assert title.station == "Dlf" and title.channel == "5C" and title.sid == 0xD210
    assert title.name == "A-HA – TAKE ON ME"
    assert [kind for kind, _a, _b in read_boxes(title.path.read_bytes())] == [
        b"ftyp",
        b"mdat",
        b"moov",
    ]
    assert not running.current.path.exists() and not running.current.sidecar.exists()
    stored = json.loads(title.sidecar.read_text(encoding="utf-8"))
    assert stored["title"] == "TAKE ON ME" and stored["path"] == title.path.name
    assert harness.changes[-1] == (harness.ring, ())
    # The next title records already.
    assert harness.status.current.title == "RADIO GA GA"
    assert sorted(p.suffix for p in harness.buffer.iterdir()) == [".aac", ".json", ".json", ".m4a"]


def test_a_short_stretch_is_thrown_away(harness):
    harness.label("Jingle")
    harness.feed(10)
    harness.label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA")

    assert harness.ring == ()
    assert sorted(p.suffix for p in harness.buffer.iterdir()) == [".aac", ".json"]


def test_tags_that_come_late_name_the_title_recording(harness):
    harness.feed(5)
    assert harness.status.current.title == ""
    harness.label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA")
    assert harness.status.current.title == "TAKE ON ME"  # no new file
    harness.feed(25)
    harness.recorder.stop()  # the exit of the radio keeps the title

    assert len(harness.ring) == 1
    assert harness.ring[0].artist == "A-HA" and harness.ring[0].duration == pytest.approx(
        30, abs=0.2
    )
    assert not harness.status.recording


def test_the_ring_drops_the_oldest(harness):
    harness.recorder.set_ring_size(4)
    for number in range(5):
        harness.label(f"Titel {number}", f"Titel {number}", "Band")
        harness.feed(21)
    harness.recorder.reset()

    assert [r.title for r in harness.ring] == ["Titel 1", "Titel 2", "Titel 3", "Titel 4"]
    assert sorted(p.suffix for p in harness.buffer.iterdir()) == [".json"] * 4 + [".m4a"] * 4
    harness.recorder.set_ring_size(3)
    assert [r.title for r in harness.ring] == ["Titel 2", "Titel 3", "Titel 4"]
    assert len(list(harness.buffer.glob("*.m4a"))) == 3
    harness.recorder.set_ring_size(1)  # below the range: its lower end
    assert len(harness.ring) == 3


def test_keeping_moves_a_title_out_of_the_buffer_and_deleting_removes_it(harness):
    for number in range(2):
        harness.label(f"Titel {number}", f"Titel {number}", "Band")
        harness.feed(21)
    harness.recorder.reset()
    first = harness.ring[0]

    harness.recorder.keep(first.id)

    assert [r.title for r in harness.ring] == ["Titel 1"]
    kept = harness.recorder.kept
    assert len(kept) == 1 and kept[0].kept and kept[0].title == "Titel 0"
    assert kept[0].path == harness.directory / first.path.name and kept[0].path.is_file()
    assert json.loads(kept[0].sidecar.read_text(encoding="utf-8"))["kept"] is True
    assert not first.path.exists() and not first.sidecar.exists()
    assert harness.changes[-1] == ((harness.ring[0],), kept)
    harness.recorder.keep("nothing")  # unknown: nothing happens

    harness.recorder.delete(kept[0].id)
    assert harness.recorder.kept == () and not kept[0].path.exists()
    harness.recorder.delete(harness.ring[0].id)
    assert harness.ring == () and list(harness.buffer.iterdir()) == []


def test_the_titles_are_read_again_at_the_next_start_and_a_broken_off_one_completed(
    harness, tmp_path
):
    harness.label("Titel 0", "Titel 0", "Band")
    harness.feed(21)
    harness.recorder.reset()
    harness.recorder.keep(harness.ring[0].id)
    harness.label("Titel 1", "Titel 1", "Band")
    harness.feed(21)
    harness.recorder.reset()
    harness.label("Titel 2", "Titel 2", "Band")
    harness.feed(30)
    harness.label("Titel 3", "Titel 3", "Band")
    harness.feed(5)
    # The power goes: the file handles are left as they are.
    harness.recorder._segment.file.close()
    harness.recorder._segment = None

    again = _Harness(tmp_path)
    again.recorder.set_directory(tmp_path)

    assert [r.title for r in again.ring] == ["Titel 1", "Titel 2"]
    assert all(r.complete and r.path.suffix == ".m4a" for r in again.ring)
    assert again.ring[1].duration == pytest.approx(30, abs=0.2)
    assert [r.title for r in again.recorder.kept] == ["Titel 0"]
    assert sorted(p.suffix for p in again.buffer.iterdir()) == [".json", ".json", ".m4a", ".m4a"]
    assert again.recorder.status.message == NO_STATION


def test_a_long_title_is_cut_into_pieces(harness, monkeypatch):
    monkeypatch.setattr(recorder_module, "MAX_TITLE_SECONDS", 30)
    harness.label("Sendung", "Sendung", "Moderator")
    harness.feed(75)

    assert [r.title for r in harness.ring] == ["Sendung", "Sendung"]
    assert harness.ring[0].duration == pytest.approx(30, abs=SUPERFRAME_S)
    assert harness.ring[0].path.name != harness.ring[1].path.name
    assert harness.status.current.duration == pytest.approx(15, abs=SUPERFRAME_S)


def test_a_station_change_a_format_change_and_lost_superframes(harness):
    harness.label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA")
    harness.feed(21)
    harness.feed(1, sync_ok=False)  # 8 superframes of 3 units lost
    assert harness.status.current.lost == 24

    harness.recorder.set_station(OTHER_STATION)
    assert len(harness.ring) == 1 and harness.ring[0].lost == 24
    assert harness.status.message == WAITING

    harness.feed(21)
    harness.feed(21, audio_format=FORMAT_MONO)
    assert len(harness.ring) == 2  # the format change ended the title
    assert harness.ring[1].station == "Dlf Kultur" and harness.ring[1].title == ""
    assert harness.ring[1].name == "Untitled"
    assert harness.status.current.audio_format["stereo_core"] is False

    # A new name for the same service renames what records.
    harness.recorder.set_station(Station("5C", 0xD220, "Deutschlandfunk Kultur", "DR"))
    assert harness.status.current.station == "Deutschlandfunk Kultur"
    assert len(harness.ring) == 2


def test_dab_frames_record_as_mp2(harness):
    harness.label("Nachrichten")
    harness.feed_mp2(30)
    harness.recorder.stop()

    assert len(harness.ring) == 1
    title = harness.ring[0]
    assert title.codec == CODEC_DAB and title.path.suffix == ".mp2" and title.bitrate == 128
    assert title.duration == pytest.approx(30, abs=0.03)
    assert title.path.stat().st_size == 1250 * 384
    assert title.title == "Nachrichten" and title.name == "Nachrichten"


def test_without_space_the_ring_stops(harness, monkeypatch):
    monkeypatch.setattr(recorder_module, "free_bytes", lambda _directory: 10_000_000)
    harness.feed(1)

    assert harness.status.message == NO_SPACE and not harness.status.recording
    assert list(harness.buffer.iterdir()) == [] if harness.buffer.exists() else True


def test_a_write_that_fails_ends_the_title_and_the_next_is_tried(harness, monkeypatch):
    harness.label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA")
    harness.feed(5)
    broken = harness.status.current.path

    def fail(self, data, *args):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(recorder_module._Segment, "write", fail)
    harness.feed(1)
    assert harness.status.message.startswith("Error while writing")
    assert not harness.status.recording
    assert not broken.exists() and not broken.with_suffix(".json").exists()

    monkeypatch.undo()
    harness.feed(1)
    assert harness.status.recording and harness.status.current.path != broken


def test_a_drive_mounted_after_the_start_is_read_before_the_first_title(tmp_path):
    # On the Pi the radio sets the directory before it mounts the stick; the titles
    # of the last run were then never read, a broken off one never completed, and the
    # ring grew by up to its size with every boot (seen 10.10.2026).
    stick = tmp_path / "stick"
    stick.mkdir()
    before = _Harness(stick)
    before.recorder.set_ring_size(3)
    before.recorder.set_directory(stick)
    before.recorder.set_station(STATION)
    for number in range(3):
        before.label(f"Titel {number}", f"Titel {number}", "Band")
        before.feed(21)
    before.label("Titel 3", "Titel 3", "Band")
    before.feed(25)
    before.recorder._segment.file.close()  # the power goes
    before.recorder._segment = None
    stick.rename(tmp_path / "unmounted")

    again = _Harness(stick)
    again.recorder.set_ring_size(3)
    again.recorder.set_directory(stick)
    again.recorder.set_station(STATION)
    again.feed(1)
    assert again.ring == () and again.status.message == DIRECTORY_MISSING
    (tmp_path / "unmounted").rename(stick)  # mounted
    again.label("Titel 4", "Titel 4", "Band")
    again.feed(1)

    assert [r.title for r in again.ring] == ["Titel 1", "Titel 2", "Titel 3"]
    assert all(r.complete and r.path.suffix == ".m4a" for r in again.ring)
    assert again.changes[-1] == (again.ring, ())
    assert again.status.recording and again.status.current.title == "Titel 4"
    assert len(list(again.buffer.glob("*.m4a"))) == 3
    assert [p.name for p in again.buffer.glob("*.aac")] == [again.status.current.path.name]


def test_a_directory_that_goes_is_reported(harness, tmp_path):
    harness.recorder.set_directory(tmp_path / "stick")
    assert harness.status.message == DIRECTORY_MISSING
    harness.feed(1)
    assert not harness.status.recording
    (tmp_path / "stick").mkdir()
    harness.feed(1)
    assert harness.status.recording  # the drive came: the next title records


def test_the_status_with_the_running_title_comes_once_a_second(harness):
    harness.label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA")
    harness.statuses.clear()
    harness.feed(3)
    recording = [s for s in harness.statuses if s.recording]
    assert 3 <= len(recording) <= 5
    assert recording[-1].current.duration > recording[0].current.duration


def test_the_title_recording_can_be_marked_to_be_kept(harness):
    harness.label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA")
    harness.feed(5)
    running = harness.status.current
    assert not running.kept

    harness.recorder.keep(running.id)
    assert harness.status.current.kept
    harness.feed(20)
    harness.label("QUEEN - RADIO GA GA", "RADIO GA GA", "QUEEN")

    assert harness.ring == ()
    kept = harness.recorder.kept
    assert len(kept) == 1 and kept[0].kept and kept[0].title == "TAKE ON ME"
    assert kept[0].path.parent == harness.directory and kept[0].path.is_file()
    assert kept[0].path.suffix == ".m4a" and not list(harness.buffer.glob("*.m4a"))
    assert harness.changes[-1] == ((), kept)


def test_a_late_label_starts_the_next_title_earlier_and_leaves_the_one_before_whole(harness):
    """The station names the title 30 s after it began: the file that begins holds the
    last 30 s of the one that ends as well, which keeps them. The delay varies, so
    nothing is cut off; the files overlap instead."""
    harness.recorder.set_label_delay(30)
    harness.label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA")
    harness.feed(60)
    harness.label("QUEEN - RADIO GA GA", "RADIO GA GA", "QUEEN")

    assert len(harness.ring) == 1
    first = harness.ring[0]
    assert first.duration == pytest.approx(60, abs=SUPERFRAME_S)
    running = harness.status.current
    assert running.title == "RADIO GA GA"
    assert running.duration == pytest.approx(30, abs=SUPERFRAME_S)
    assert running.start == pytest.approx(harness.clock.now - 30, abs=SUPERFRAME_S)
    harness.feed(10)
    harness.recorder.stop()
    second = harness.ring[1]
    assert second.duration == pytest.approx(40, abs=SUPERFRAME_S)
    # Both m4a hold every unit: 60 and 40 s at 25 units of 100 bytes a second.
    assert first.size > second.size > 40 * 25 * 100


def test_a_title_shorter_than_the_delay_gives_what_it_has(harness):
    harness.recorder.set_label_delay(30)
    harness.label("Jingle", "Jingle", "")
    harness.feed(10)
    harness.label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA")

    assert harness.ring == ()  # the jingle is too short to keep; its 10 s open the title
    assert harness.status.current.duration == pytest.approx(10, abs=SUPERFRAME_S)
    assert harness.status.current.title == "TAKE ON ME"


def test_an_early_label_moves_the_boundary_forward(harness):
    """The station names the title 10 s before it begins: the next begins 10 s after
    the label, and labels in between refine nothing. The one that ends is cut at the
    label all the same (the stop delay says otherwise, see below)."""
    harness.recorder.set_label_delay(-10)
    harness.label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA")
    harness.feed(30)
    harness.label("QUEEN - RADIO GA GA", "RADIO GA GA", "QUEEN")
    assert harness.ring == () and harness.status.current.title == "TAKE ON ME"
    harness.feed(5)
    harness.label("Radio XY - Hits", "RADIO GA GA", "QUEEN")
    assert harness.status.current.title == "TAKE ON ME"
    harness.feed(5)

    assert len(harness.ring) == 1
    assert harness.ring[0].title == "TAKE ON ME"
    assert harness.ring[0].duration == pytest.approx(30, abs=SUPERFRAME_S)
    assert harness.status.current.title == "RADIO GA GA"
    assert harness.status.current.duration == pytest.approx(0, abs=SUPERFRAME_S)


# --- Reading a title back for playback --------------------------------------------------


def test_a_title_is_read_back_for_playback(harness):
    loaded: list = []
    failed: list[tuple[str, str]] = []
    harness.recorder.loaded.connect(loaded.append)
    harness.recorder.load_failed.connect(lambda i, m: failed.append((i, m)))
    harness.label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA")
    harness.feed(30)
    harness.label("QUEEN - RADIO GA GA", "RADIO GA GA", "QUEEN")  # the first title complete
    title = harness.ring[0]

    harness.recorder.load(title.id)

    audio = loaded[-1]
    assert audio.recording_id == title.id and audio.audio_format == FORMAT
    assert len(audio.units) == 250 * 3 and audio.duration == pytest.approx(30, abs=0.04)
    assert audio.units[0] == bytes([0x21, 0x00]) + bytes(98)

    harness.recorder.load("nothing")
    assert failed[-1] == ("nothing", "The recording no longer exists")
    title.path.unlink()
    harness.recorder.load(title.id)
    assert failed[-1] == (title.id, "The recording cannot be read")


def test_a_dab_title_is_read_back_frame_by_frame(harness):
    loaded: list = []
    harness.recorder.loaded.connect(loaded.append)
    harness.label("Nachrichten")
    harness.feed_mp2(30)
    harness.recorder.stop()
    title = harness.ring[0]

    harness.recorder.load(title.id)

    audio = loaded[-1]
    assert audio.audio_format == MP2 and len(audio.units) == 1250
    assert audio.units[0][:2] == b"\xff\xfd" and audio.duration == pytest.approx(30, abs=0.03)


# --- The stop delay and the switch ----------------------------------------------------


def test_a_stop_delay_cuts_the_title_before_short_of_the_label(harness):
    """The station names the next title 30 s after it began: with the stop delay as
    well, the title before ends 30 s before the label, as the boundary once moved."""
    harness.recorder.set_label_delay(30)
    harness.recorder.set_stop_delay(30)
    harness.label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA")
    harness.feed(60)
    harness.label("QUEEN - RADIO GA GA", "RADIO GA GA", "QUEEN")

    first = harness.ring[0]
    assert first.duration == pytest.approx(30, abs=SUPERFRAME_S)
    assert len(read_m4a(first.path)[1]) == 750
    running = harness.status.current
    assert running.title == "RADIO GA GA"
    assert running.duration == pytest.approx(30, abs=SUPERFRAME_S)


def test_a_negative_stop_delay_lets_the_title_before_run_on_beside_the_next(harness):
    """The title before ends 10 s after the label, in its own file, while the next
    already records: both hold those 10 s; a station change ends one running on too."""
    harness.recorder.set_stop_delay(-10)
    harness.label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA")
    harness.feed(60)
    harness.label("QUEEN - RADIO GA GA", "RADIO GA GA", "QUEEN")

    assert harness.ring == () and harness.status.current.title == "RADIO GA GA"
    harness.feed(5)
    assert harness.ring == ()
    harness.feed(5)
    assert len(harness.ring) == 1
    first = harness.ring[0]
    assert first.title == "TAKE ON ME" and first.duration == pytest.approx(70, abs=SUPERFRAME_S)
    assert 1750 <= len(read_m4a(first.path)[1]) <= 1752  # whole superframes of 3 units
    assert harness.status.current.duration == pytest.approx(10, abs=SUPERFRAME_S)
    assert harness.changes[-1] == ((first,), ())

    harness.recorder.set_stop_delay(-30)
    harness.feed(30)
    harness.label("Nachrichten", "Nachrichten", "")
    harness.recorder.set_station(OTHER_STATION)
    assert [r.title for r in harness.ring] == ["TAKE ON ME", "RADIO GA GA"]
    assert harness.ring[1].duration == pytest.approx(40, abs=SUPERFRAME_S)


def test_a_label_running_ahead_counts_the_wait_against_the_stop_delay(harness):
    """The label names the next title 10 s before it begins (start delay -10): with the
    stop delay -10 the one before ends when the next begins, as the boundary once
    moved; with 0 it ends at the label (the test above)."""
    harness.recorder.set_label_delay(-10)
    harness.recorder.set_stop_delay(-10)
    harness.label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA")
    harness.feed(30)
    harness.label("QUEEN - RADIO GA GA", "RADIO GA GA", "QUEEN")
    harness.feed(5)
    harness.feed(5)  # two steps: 84 superframes reach the 10 s, 83 fall one unit short

    assert harness.ring[0].duration == pytest.approx(40, abs=SUPERFRAME_S)
    assert harness.status.current.title == "RADIO GA GA"
    assert harness.status.current.duration == pytest.approx(0, abs=SUPERFRAME_S)


def test_switched_off_the_recorder_writes_nothing_and_says_so(harness):
    harness.label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA")
    harness.feed(30)

    harness.recorder.set_enabled(False)

    assert harness.status.message == DISABLED and not harness.status.recording
    assert len(harness.ring) == 1  # the title recording ended, kept
    harness.feed(30)
    assert len(harness.ring) == 1 and harness.status.message == DISABLED
    harness.recorder.set_enabled(True)
    assert harness.status.message == WAITING
    harness.feed(1)
    assert harness.status.recording


# --- For the cutter: where a file lies in its stream, its labels, cuts ----------------


class _Numbered:
    """Superframes whose units say their place in the stream, and what was fed."""

    def __init__(self, harness: _Harness) -> None:
        self.harness = harness
        self.fed: list[bytes] = []

    def feed(self, seconds: float) -> None:
        for _ in range(round(seconds / SUPERFRAME_S)):
            units = []
            for _ in range(FORMAT.num_audio_units):
                units.append(bytes([0x21, 0x00]) + len(self.fed).to_bytes(4, "big") + bytes(94))
                self.fed.append(units[-1])
            self.harness.recorder.process_superframe(
                SuperframeResult(
                    sync_ok=True,
                    audio_units=units,
                    num_audio_units=FORMAT.num_audio_units,
                    audio_format=FORMAT,
                    data_length=1440 - 48,
                    total_length=1440,
                )
            )
            self.harness.clock.advance(SUPERFRAME_S)


def _three_titles(harness: _Harness) -> _Numbered:
    """Three titles, the next beginning 10 s before its label, the one before running
    on 5 s after it: neighbours that overlap on both sides."""
    harness.recorder.set_label_delay(10)
    harness.recorder.set_stop_delay(-5)
    stream = _Numbered(harness)
    harness.label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA")
    stream.feed(40)
    harness.label("QUEEN - RADIO GA GA", "RADIO GA GA", "QUEEN")
    stream.feed(40)
    harness.label("TOTO - AFRICA", "AFRICA", "TOTO")
    stream.feed(30)
    harness.recorder.stop()
    return stream


def test_the_files_of_a_stream_say_where_they_lie_in_it(harness):
    """Side by side by their stream and first unit, three overlapping titles hold every
    unit fed exactly once."""
    from recording.cutter import build_stretch

    stream = _three_titles(harness)
    first, second, third = harness.ring
    assert first.session == second.session == third.session != ""
    assert first.first_unit == 0
    assert 0 < second.first_unit < round(first.duration / 0.04) < third.first_unit
    pieces = [(r, read_m4a(r.path)[1]) for r in harness.ring]
    stretch = build_stretch(second, pieces, FORMAT)
    assert stretch.units == tuple(stream.fed)
    assert [c.recording.title for c in stretch.clips] == ["TAKE ON ME", "RADIO GA GA", "AFRICA"]


def test_anything_that_breaks_the_audio_begins_another_stream(harness):
    harness.feed(30)
    harness.recorder.reset()  # another service comes
    harness.feed(30)
    harness.recorder.stop()
    first, second = harness.ring
    assert first.session != second.session and second.first_unit == 0


def test_a_long_title_cut_into_pieces_stays_one_stream(harness, monkeypatch):
    monkeypatch.setattr(recorder_module, "MAX_TITLE_SECONDS", 30)
    harness.feed(85)  # the last piece above `MIN_TITLE_SECONDS`
    harness.recorder.stop()
    first, second, third = harness.ring
    assert first.session == second.session == third.session
    assert second.first_unit == round(first.duration / 0.04)
    assert third.first_unit == second.first_unit + round(second.duration / 0.04)


def test_the_labels_seen_are_noted_where_the_audio_changed(harness):
    """With a start delay of 10 s a label tells of the audio 10 s before it came; a
    label repeated is noted once."""
    harness.recorder.set_label_delay(10)
    harness.label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA")
    harness.feed(30)
    harness.label("Jetzt: A-HA", "TAKE ON ME", "A-HA")
    harness.feed(5)
    harness.label("Jetzt: A-HA", "TAKE ON ME", "A-HA")
    harness.feed(25)
    harness.recorder.stop()
    (title,) = harness.ring
    assert title.labels == (
        (0.0, "A-HA - TAKE ON ME", "A-HA", "TAKE ON ME"),
        (pytest.approx(20.0, abs=SUPERFRAME_S), "Jetzt: A-HA", "A-HA", "TAKE ON ME"),
    )
    sidecar = json.loads(title.sidecar.read_text(encoding="utf-8"))
    assert sidecar["labels"][1][1] == "Jetzt: A-HA" and sidecar["session"] == title.session


def test_older_sidecars_read_with_what_they_lack_by_default(harness):
    harness.feed(30)
    harness.recorder.stop()
    (title,) = harness.ring
    data = json.loads(title.sidecar.read_text(encoding="utf-8"))
    for key in ("session", "first_unit", "cut", "labels"):
        del data[key]
    old = Recording.from_dict(data, title.path.parent)
    assert (old.session, old.first_unit, old.cut, old.labels) == ("", 0, False, ())


def test_a_title_is_read_with_its_neighbours_for_cutting(harness):
    stream = _three_titles(harness)
    loaded, failed = [], []
    harness.recorder.stretch_loaded.connect(loaded.append)
    harness.recorder.stretch_failed.connect(lambda *args: failed.append(args))

    harness.recorder.load_stretch(harness.ring[1].id)
    (stretch,) = loaded
    assert [c.recording.id for c in stretch.clips] == [r.id for r in harness.ring]
    assert stretch.units == tuple(stream.fed)
    assert stretch.opened.id == harness.ring[1].id

    harness.recorder.load_stretch("gone")
    assert failed and failed[0][0] == "gone"


def test_a_cut_is_kept_beside_the_ring(harness):
    from recording.cutter import CutRequest

    _three_titles(harness)
    ring = harness.ring
    stretches, saved = [], []
    harness.recorder.stretch_loaded.connect(stretches.append)
    harness.recorder.cut_saved.connect(lambda kept, replaced: saved.append((kept, replaced)))
    harness.recorder.load_stretch(ring[1].id)
    stretch = stretches[0]
    first = stretch.unit_at(stretch.opened_clip().start - 5.0)

    request = CutRequest(stretch, first, first + 1000, "QUEEN", "RADIO GA GA")
    harness.recorder.save_cut(request)

    ((kept, replaced),) = saved
    assert replaced == "" and harness.ring == ring
    assert harness.recorder.kept == (kept,)
    assert kept.cut and kept.kept and kept.complete and kept.path.parent == harness.directory
    assert kept.path.name.endswith("QUEEN – RADIO GA GA.m4a")
    assert kept.duration == pytest.approx(40.0) and kept.start == pytest.approx(request.start)
    assert (kept.session, kept.first_unit) == (stretch.session, stretch.first_unit + first)
    assert read_m4a(kept.path)[1] == request.units
    assert json.loads(kept.sidecar.read_text(encoding="utf-8"))["cut"] is True

    # Saved again under the same name, it takes the place of the one before.
    harness.recorder.save_cut(
        CutRequest(stretch, first, first + 500, "QUEEN", "RADIO GA GA", kept.id)
    )
    again, replaced = saved[-1]
    assert replaced == kept.id and again.path == kept.path
    assert harness.recorder.kept == (again,) and again.duration == pytest.approx(20.0)
    assert len(list(harness.directory.glob("*.m4a"))) == 1


def test_a_cut_without_its_directory_is_not_kept(harness, tmp_path):
    from recording.cutter import CutRequest

    _three_titles(harness)
    stretches, failed = [], []
    harness.recorder.stretch_loaded.connect(stretches.append)
    harness.recorder.cut_failed.connect(failed.append)
    harness.recorder.load_stretch(harness.ring[0].id)
    harness.recorder.set_directory(tmp_path / "gone")
    harness.recorder.save_cut(CutRequest(stretches[0], 0, 100, "A-HA", "TAKE ON ME"))
    assert failed == [DIRECTORY_MISSING]


def test_a_dab_title_is_cut_frame_by_frame(harness):
    from recording.cutter import CutRequest

    harness.feed_mp2(30)
    harness.recorder.stop()
    stretches, saved = [], []
    harness.recorder.stretch_loaded.connect(stretches.append)
    harness.recorder.cut_saved.connect(lambda kept, _replaced: saved.append(kept))
    harness.recorder.load_stretch(harness.ring[0].id)
    harness.recorder.save_cut(CutRequest(stretches[0], 10, 510, "", "Nachrichten"))
    (kept,) = saved
    assert kept.path.suffix == ".mp2" and kept.path.stat().st_size == 500 * 384
    assert kept.duration == pytest.approx(500 * MP2.duration_seconds)
