"""The sub-page "Recordings" of the radio page (`ui.touch.recordings_view`), the badge on
the slide and the ring in the settings, on a controller whose recorder is told what to
report; and the bar of the recording playing."""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication, QWidget

from recording.playback import LOADING, PAUSED, PLAYING, PlaybackState
from recording.recorder import NO_DIRECTORY, RECORDING, WAITING, RecorderStatus, Recording
from ui.touch.recordings_view import (
    RecordingEntry,
    RecordingsPage,
    entry_detail,
    kept_entries,
    length_text,
    ring_entries,
    status_text,
    times_text,
)
from ui.touch.window import SCREEN_HEIGHT, SCREEN_WIDTH

START = 1_791_000_000.0


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window():
    window = QWidget()
    window.resize(SCREEN_WIDTH, SCREEN_HEIGHT)
    window.show()
    yield window
    window.deleteLater()


def _recording(
    number: int, title: str = "", artist: str = "", *, kept=False, complete=True, duration=200.0
) -> Recording:
    stem = f"2026-10-09 18.{number:02d} Dlf"
    return Recording(
        id=stem,
        path=Path("/stick/Puffer") / (stem + ".m4a"),
        kept=kept,
        complete=complete,
        codec="DAB+",
        station="Dlf",
        ensemble="DR Deutschland",
        channel="5C",
        sid=0xD210,
        title=title,
        artist=artist,
        text=title,
        start=START + number * 60,
        duration=duration,
        size=1000,
        lost=0,
        bitrate=96,
        audio_format={},
    )


def _clock(number: int) -> str:
    return time.strftime("%H:%M", time.localtime(START + number * 60))


# --- Texts and entries ----------------------------------------------------------------


def test_lengths_and_the_status_read_well():
    assert length_text(225) == "3:45"
    assert length_text(3723) == "1:02:03"
    assert length_text(5.4) == "0:05"
    assert status_text(RecorderStatus(False, NO_DIRECTORY)) == NO_DIRECTORY
    running = _recording(3, "TAKE ON ME", "A-HA", complete=False, duration=83)
    assert status_text(RecorderStatus(True, RECORDING, running)) == (
        "Now recording: A-HA – TAKE ON ME · 1:23"
    )
    assert times_text(PlaybackState("x", PLAYING, 65.0, 200.0)) == "1:05 / 3:20"
    assert times_text(PlaybackState("x", PAUSED, 65.0, 200.0), 90.0) == "1:30 / 3:20"
    assert times_text(PlaybackState("x", LOADING, 0.0, 200.0)) == "Loading …"


def test_the_entries_put_the_title_recording_first_and_the_newest_next():
    ring = (_recording(1, "Eins"), _recording(2, "Zwei"))
    running = _recording(3, "Drei", complete=False, duration=10)

    entries = ring_entries(ring, RecorderStatus(True, RECORDING, running))
    assert [e.name for e in entries] == ["Drei", "Zwei", "Eins"]
    assert entries[0] == RecordingEntry(
        running.id, "Drei", f"Dlf · {_clock(3)} · 0:10 · recording", True, False
    )
    assert entries[1].detail == f"Dlf · {_clock(2)} · 3:20" and not entries[1].running
    assert entry_detail(_recording(4)) == f"Dlf · {_clock(4)} · 3:20"

    # Not recording: the ring alone, whatever `current` says.
    assert [e.name for e in ring_entries(ring, RecorderStatus(False, WAITING))] == ["Zwei", "Eins"]
    # Marked to be kept: shown so.
    marked = RecorderStatus(True, RECORDING, _recording(3, "Drei", kept=True, complete=False))
    assert ring_entries(ring, marked)[0].marked
    # The one playing: marked so.
    playing = ring_entries(ring, RecorderStatus(False, WAITING), ring[0].id)
    assert [e.playing for e in playing] == [False, True]

    kept = kept_entries((_recording(5, "Fünf", kept=True), _recording(6, "Sechs", kept=True)))
    assert [e.name for e in kept] == ["Sechs", "Fünf"] and all(e.marked for e in kept)
    assert [e.playing for e in kept_entries((_recording(5),), _recording(5).id)] == [True]


# --- The page -------------------------------------------------------------------------


@pytest.fixture
def page(window, make_controller, data_directory, monkeypatch):
    controller = make_controller(data_directory)
    page = RecordingsPage(controller, window)
    page.resize(SCREEN_WIDTH - 112, SCREEN_HEIGHT)
    page.show()
    kept: list[str] = []
    deleted: list[str] = []
    played: list[str] = []
    monkeypatch.setattr(controller, "keep_recording", kept.append)
    monkeypatch.setattr(controller, "delete_recording", deleted.append)
    monkeypatch.setattr(controller, "play_recording", played.append)
    return page, controller, kept, deleted, played


def _report(controller, ring=(), kept=(), status=None) -> None:
    """What the recorder would report through its signals."""
    controller._on_recordings_changed(ring, kept)
    if status is not None:
        controller._on_recorder_status(status)


def _playing(controller, recording: Recording, state: str, position: float = 0.0) -> None:
    """What the controller holds while `recording` plays, as the player reports it."""
    controller._playback_source_id = recording.id
    controller._playback_recording = recording
    controller._on_playback_changed(
        PlaybackState(recording.id, state, position, recording.duration)
    )


def test_the_page_shows_why_nothing_records_and_the_empty_lists(page):
    page, _controller, _kept, _deleted, _played = page
    assert page.summary.text() == NO_DIRECTORY
    assert page.ring_page.entries() == [] and page.kept_page.entries() == []
    assert page.ring_page._empty.isVisibleTo(page) and page.ring_page.status.text() == ""
    assert not page.bar.isVisibleTo(page)


def test_the_lists_follow_the_recorder_and_the_running_title_grows_in_place(page):
    page, controller, _kept, _deleted, _played = page
    ring = (_recording(1, "Eins"), _recording(2, "Zwei"))
    running = _recording(3, "Drei", complete=False, duration=10)
    _report(controller, ring, (), RecorderStatus(True, RECORDING, running))

    assert page.summary.text() == "Now recording: Drei · 0:10"
    assert [e.name for e in page.ring_page.entries()] == ["Drei", "Zwei", "Eins"]
    assert page.ring_page.status.text() == "3 titles"
    rows = page.ring_page.rows()
    assert rows[0]._mark.text().endswith("●</span>")

    controller._on_recorder_status(
        RecorderStatus(True, RECORDING, _recording(3, "Drei", complete=False, duration=70))
    )
    assert page.ring_page.rows() == rows  # the same rows, refreshed
    assert rows[0].entry.detail.endswith("1:10 · recording")
    assert page.summary.text() == "Now recording: Drei · 1:10"

    _report(controller, ring, (_recording(0, "Null", kept=True),), RecorderStatus(False, WAITING))
    assert [e.name for e in page.ring_page.entries()] == ["Zwei", "Eins"]
    assert [e.name for e in page.kept_page.entries()] == ["Null"]
    assert page.kept_page.status.text() == "1 title"
    assert page.summary.text() == WAITING


def test_a_tap_on_a_title_offers_to_play_cut_keep_or_delete_it(page):
    page, controller, kept, deleted, played = page
    cut: list[str] = []
    page.cut_requested.connect(cut.append)
    _report(controller, (_recording(1, "Eins"),), (), RecorderStatus(False, WAITING))

    page.ring_page.rows()[0].tapped.emit()
    sheet = page.sheet()
    assert sheet.isVisible() and sheet.title.text() == "Eins"
    assert [b.text() for b in sheet.actions] == ["Play", "Cut", "Delete", "Keep"]
    assert [b.property("role") for b in sheet.actions] == [None, "notice", "danger", "accent"]
    sheet.actions[3].click()
    assert kept == [_recording(1).id] and not sheet.isVisible()

    page.ring_page.rows()[0].tapped.emit()
    sheet.actions[0].click()
    assert played == [_recording(1).id]

    page.ring_page.rows()[0].tapped.emit()
    sheet.actions[1].click()
    assert cut == [_recording(1).id]

    page.ring_page.rows()[0].tapped.emit()
    sheet.actions[2].click()
    assert deleted == [_recording(1).id]


def test_the_title_recording_can_be_marked_once(page):
    page, controller, kept, _deleted, _played = page
    running = _recording(3, "Drei", complete=False)
    _report(controller, (), (), RecorderStatus(True, RECORDING, running))

    page.ring_page.rows()[0].tapped.emit()
    sheet = page.sheet()
    assert [b.text() for b in sheet.actions] == ["Keep"]
    sheet.actions[0].click()
    assert kept == [running.id]

    controller._on_recorder_status(
        RecorderStatus(True, RECORDING, _recording(3, "Drei", kept=True, complete=False))
    )
    row = page.ring_page.rows()[0]
    assert row._mark.text().endswith("●</span>") and row.entry.detail.endswith("will be kept")
    row.tapped.emit()
    assert sheet.actions == [] and sheet.detail.text().startswith("Will be kept")


def test_a_kept_title_can_be_played_cut_or_deleted(page):
    page, controller, kept, deleted, played = page
    cut: list[str] = []
    page.cut_requested.connect(cut.append)
    _report(controller, (), (_recording(5, "Fünf", kept=True),), RecorderStatus(False, WAITING))
    page.pager.set_current_index(1)

    page.kept_page.rows()[0].tapped.emit()
    sheet = page.sheet()
    assert [b.text() for b in sheet.actions] == ["Play", "Cut", "Delete"]
    sheet.actions[0].click()
    assert played == [_recording(5).id]
    page.kept_page.rows()[0].tapped.emit()
    sheet.actions[1].click()
    assert cut == [_recording(5).id]
    page.kept_page.rows()[0].tapped.emit()
    sheet.actions[2].click()
    assert deleted == [_recording(5).id] and kept == []


# --- The recording playing ------------------------------------------------------------


def test_the_bar_shows_the_recording_playing_and_its_row_carries_the_mark(page):
    page, controller, _kept, _deleted, _played = page
    ring = (_recording(1, "Eins"), _recording(2, "Zwei"))
    _report(controller, ring, (), RecorderStatus(False, WAITING))
    assert not page.bar.isVisibleTo(page)

    _playing(controller, ring[0], LOADING)
    assert page.bar.isVisibleTo(page) and page.bar.times.text() == "Loading …"
    assert not page.bar.toggle_button.isEnabled()

    _playing(controller, ring[0], PLAYING, 65.0)
    assert page.bar.name.text() == "Eins" and page.bar.detail.text() == f"Dlf · {_clock(1)}"
    assert page.bar.times.text() == "1:05 / 3:20" and page.bar.toggle_button.isEnabled()
    assert page.bar.slider.maximum() == 200 and page.bar.slider.value() == 65
    rows = page.ring_page.rows()
    assert rows[1].entry.playing and rows[1]._mark.text().endswith("▶</span>")
    assert not rows[0].entry.playing and rows[0]._mark.text() == ""

    # The playing title's sheet offers to stop it.
    rows[1].tapped.emit()
    assert [b.text() for b in page.sheet().actions] == ["Stop", "Cut", "Delete", "Keep"]
    page.sheet().dismiss()

    controller._on_playback_changed(None)
    assert not page.bar.isVisibleTo(page)
    assert page.ring_page.rows() == rows and not rows[1].entry.playing


def test_the_bar_drives_the_playback(page):
    page, controller, _kept, _deleted, _played = page
    ring = (_recording(1, "Eins"),)
    _report(controller, ring, (), RecorderStatus(False, WAITING))
    asked: list[object] = []
    controller._playback_paused.connect(lambda: asked.append("pause"))
    controller._playback_resumed.connect(lambda: asked.append("resume"))
    controller._playback_seek.connect(lambda s: asked.append(("seek", s)))
    controller._playback_stop_requested.connect(lambda: asked.append("stop"))

    _playing(controller, ring[0], PLAYING, 10.0)
    page.bar.toggle_button.click()
    assert asked == ["pause"]
    _playing(controller, ring[0], PAUSED, 10.0)
    page.bar.toggle_button.click()
    assert asked[-1] == "resume"

    # Dragging shows the target and keeps the handle where the finger is.
    slider = page.bar.slider
    slider.setSliderDown(True)
    slider.setValue(90)
    assert page.bar.times.text() == "1:30 / 3:20"
    _playing(controller, ring[0], PLAYING, 12.0)
    assert slider.value() == 90
    slider.setSliderDown(False)
    slider.sliderReleased.emit()
    assert asked[-1] == ("seek", 90.0)

    page.bar.stop_button.click()
    assert asked[-1] == "stop" and controller.playback is None
    assert not page.bar.isVisibleTo(page)

    # Leaving the page by any way ends the playback too: the page is hidden, by
    # "Back" (the radio page shows its player again) or by the menu.
    _playing(controller, ring[0], PLAYING, 12.0)
    page.hide()
    assert asked[-1] == "stop" and controller.playback is None
    asked.clear()
    page.show()
    page.hide()  # nothing playing: nothing to stop
    assert asked == []
    # Also when a page above it goes (the radio page, as the menu shows another).
    page.show()
    _playing(controller, ring[0], PLAYING, 12.0)
    page.window().hide()
    assert asked[-1] == "stop" and controller.playback is None
    page.window().show()


def test_a_recording_that_cannot_be_played_is_said_so(page):
    page, controller, _kept, _deleted, _played = page
    controller.playback_failed.emit("The recording cannot be read")
    sheet = page.sheet()
    assert sheet.isVisible() and sheet.title.text() == "Cannot play"
    assert sheet.detail.text() == "The recording cannot be read" and sheet.actions == []
