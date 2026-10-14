"""The cutter without widgets (`ui.cut_model`) on a controller whose recorder, level
reader and player are stood in for: opening, the handles, playing (while it plays,
every move plays from there; paused, it only puts the position there), the name and
keeping."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from dsp.superframe import AudioFormat
from recording.cutter import PAUSE_KEEP_S, build_stretch
from recording.levels import STEPS_PER_UNIT
from recording.playback import PAUSED, PLAYING, PlaybackState
from recording.recorder import Recording
from ui.cut_model import END, PREVIEW_S, START, CutModel

FORMAT = AudioFormat(
    dac_rate_48k=True, sbr=True, stereo_core=True, parametric_stereo=False, mpeg_surround_config=0
)
UNIT_S = 0.04
T0 = 1_791_919_609.0


def _recording(name, start, units, first_unit, *, kept=False, artist="", title="", labels=()):
    return Recording(
        id=name,
        path=Path("/stick/Puffer") / (name + ".m4a"),
        kept=kept,
        complete=True,
        codec="DAB+",
        station="Absolut relax",
        ensemble="DR Deutschland",
        channel="5C",
        sid=0x17FA,
        title=title,
        artist=artist,
        text=f"{artist} - {title}",
        start=start,
        duration=units * UNIT_S,
        size=1000,
        lost=0,
        bitrate=72,
        audio_format={},
        session="s",
        first_unit=first_unit,
        labels=labels,
    )


ADELE = _recording("adele", T0, 1600, 0, artist="ADELE", title="SET FIRE TO THE RAIN")
BOYZONE = _recording(
    "boyzone",
    T0 + 64.0,
    3000,
    1600,
    artist="BOYZONE",
    title="NO MATTER WHAT",
    labels=((1.0, "Absolut relax - Entspannt", "", ""),),
)


def _stretch(opened=BOYZONE):
    """Adele and the title after her; `opened` is one of them, or stands for the one
    after."""
    units = tuple(n.to_bytes(3, "big") for n in range(4600))
    after = BOYZONE if opened is ADELE else opened
    return build_stretch(opened, [(ADELE, units[:1600]), (after, units[1600:])], FORMAT)


class _Calls:
    """What the model asked of the controller."""

    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.levels: list[object] = []
        self.played: list[tuple] = []
        self.sought: list[tuple] = []
        self.paused = 0
        self.resumed = 0
        self.stopped = 0
        self.saved: list[object] = []


@pytest.fixture
def model(make_controller, data_directory, monkeypatch):
    controller = make_controller(data_directory)
    calls = _Calls()
    monkeypatch.setattr(controller, "load_stretch", calls.loaded.append)
    monkeypatch.setattr(controller, "read_levels", calls.levels.append)
    monkeypatch.setattr(controller, "cancel_levels", lambda: None)
    monkeypatch.setattr(controller, "save_cut", calls.saved.append)

    def play(stretch, position, stop, back_to):
        calls.played.append((position, stop, back_to))
        controller._playback_source_id = stretch.id
        controller._on_playback_changed(
            PlaybackState(stretch.id, PLAYING, position, stretch.duration)
        )

    def seek(position, stop, back_to):
        calls.sought.append((position, stop, back_to))

    def pause():
        calls.paused += 1
        state = controller.playback
        controller._on_playback_changed(
            PlaybackState(state.recording_id, PAUSED, state.position, state.duration)
        )

    def resume():
        calls.resumed += 1

    def stop():
        calls.stopped += 1
        controller._forget_playback()

    # The controller's own playback state, as the player would report it.
    controller._playback = None
    monkeypatch.setattr(controller, "play_stretch", play)
    monkeypatch.setattr(controller, "seek_stretch", seek)
    monkeypatch.setattr(controller, "pause_playback", pause)
    monkeypatch.setattr(controller, "resume_playback", resume)
    monkeypatch.setattr(controller, "stop_playback", stop)
    cut = CutModel(controller)
    return cut, controller, calls


def _open(model, controller, calls, stretch=None):
    stretch = stretch or _stretch()
    model.open(stretch.opened.id)
    assert calls.loaded[-1] == stretch.opened.id and model.loading
    controller.stretch_loaded.emit(stretch)
    return stretch


def test_a_recording_opens_with_its_file_kept_whole(model):
    model, controller, calls = model
    opened: list[bool] = []
    model.opened.connect(lambda: opened.append(True))
    stretch = _open(model, controller, calls)

    assert opened and model.stretch is stretch and not model.loading
    assert (model.start, model.end) == (pytest.approx(64.0), pytest.approx(184.0))
    assert (model.artist, model.title) == ("BOYZONE", "NO MATTER WHAT")
    assert model.selected == START and model.position == pytest.approx(64.0)
    assert calls.levels == [stretch] and model.levels_read == 0.0
    assert not model.replacing and not model.saved
    assert model.marks == [(pytest.approx(65.0), "Absolut relax - Entspannt")]
    assert ("BOYZONE", "NO MATTER WHAT") in model.suggestions

    # Another stretch's levels, or another recording loaded, are not this one's.
    controller.stretch_loaded.emit(_stretch(ADELE))
    assert model.stretch is stretch


def test_the_level_comes_in_pieces_and_shows_the_pauses(model):
    model, controller, calls = model
    stretch = _open(model, controller, calls)
    assert model.level_step_seconds == pytest.approx(UNIT_S / STEPS_PER_UNIT)
    assert len(model.rms_db) == 4600 * STEPS_PER_UNIT
    # Units 1000 to 1999, their levels in eighths: 0.52 s quiet from 52.005 s on.
    levels = np.full(1000 * STEPS_PER_UNIT, -20.0, dtype=np.float32)
    levels[1300 * STEPS_PER_UNIT + 1 - 8000 : 1313 * STEPS_PER_UNIT + 1 - 8000] = -91.0
    controller.levels_read.emit(stretch.id, 1000 * STEPS_PER_UNIT, levels, levels)
    assert model.levels_read == pytest.approx(2000 / 4600)
    assert model.pauses == [(pytest.approx(52.005), pytest.approx(52.525))]
    controller.levels_read.emit("cut:other", 0, levels, levels)
    controller.levels_finished.emit(stretch.id)
    assert model.levels_read == 1.0


def test_the_handles_snap_to_units_keep_apart_and_jump_to_pauses(model):
    model, controller, calls = model
    stretch = _open(model, controller, calls)
    model.move(START, 52.013)
    assert model.start == pytest.approx(52.0) and model.selected == START
    model.move(START, 500.0)  # not past the end
    assert model.start == pytest.approx(183.0)
    model.move(END, 0.0)  # nor the end before the beginning
    assert model.end == pytest.approx(184.0) and model.selected == END

    model.move(START, 64.0)
    model.select(START)
    levels = np.full(4600 * STEPS_PER_UNIT, -20.0, dtype=np.float32)
    # A pause that ends between two unit boundaries (the level comes in eighths).
    levels[1300 * STEPS_PER_UNIT + 1 : 1313 * STEPS_PER_UNIT + 3] = -91.0
    controller.levels_read.emit(stretch.id, 0, levels, levels)
    assert model.can_jump(False) and not model.can_jump(True)
    assert model.jump_to_pause(False)
    assert model.start == pytest.approx(52.535 - PAUSE_KEEP_S, abs=UNIT_S)
    # The handle snapped to the unit beside the pause: it stands there, no button
    # jumps on the spot.
    assert not model.can_jump(False) and not model.can_jump(True)
    model.step(-1.0)
    assert model.start == pytest.approx(51.42, abs=UNIT_S)

    # The name follows the file holding most of what is kept.
    model.move(START, 0.0)
    model.move(END, 70.0)
    assert (model.artist, model.title) == ("ADELE", "SET FIRE TO THE RAIN")


def test_while_it_plays_every_move_plays_from_there(model):
    model, controller, calls = model
    _open(model, controller, calls)
    model.toggle()
    assert calls.played == [(pytest.approx(64.0), pytest.approx(184.0), pytest.approx(64.0))]
    assert model.playing

    model.move(START, 60.0)  # the beginning on, up to the end mark
    assert calls.sought[-1] == (pytest.approx(60.0), pytest.approx(184.0), pytest.approx(60.0))
    model.move(END, 150.0)  # the last seconds up to the end mark
    assert calls.sought[-1] == (
        pytest.approx(150.0 - PREVIEW_S),
        pytest.approx(150.0),
        pytest.approx(150.0 - PREVIEW_S),
    )
    # A finger dragging: only once it lets go.
    sought = len(calls.sought)
    model.move(END, 151.0, preview=False)
    model.move(END, 152.0, preview=False)
    assert len(calls.sought) == sought
    model.let_go(END)
    assert calls.sought[-1][0] == pytest.approx(152.0 - PREVIEW_S)

    # A tap beyond the end mark plays to the end of the stretch.
    model.tap(170.0)
    assert calls.sought[-1] == (pytest.approx(170.0), pytest.approx(184.0), pytest.approx(170.0))


def test_paused_a_move_only_puts_the_position_there(model):
    model, controller, calls = model
    _open(model, controller, calls)
    model.toggle()
    model.toggle()
    assert calls.paused == 1 and not model.playing

    model.move(START, 52.0)
    # The player is told where it goes on from, and stays paused.
    assert calls.sought[-1][0] == pytest.approx(52.0) and calls.resumed == 0
    assert not model.playing
    model.toggle()
    assert calls.resumed == 1


def test_before_it_ever_played_a_move_puts_the_position_for_play(model):
    model, controller, calls = model
    _open(model, controller, calls)
    model.move(END, 150.0)
    assert calls.sought == [] and model.position == pytest.approx(150.0 - PREVIEW_S)
    model.toggle()
    assert calls.played[-1] == (
        pytest.approx(150.0 - PREVIEW_S),
        pytest.approx(150.0),
        pytest.approx(150.0 - PREVIEW_S),
    )


def test_playing_follows_the_player_and_ends_with_the_session(model):
    model, controller, calls = model
    stretch = _open(model, controller, calls)
    model.toggle()
    controller._on_playback_changed(PlaybackState(stretch.id, PLAYING, 80.0, stretch.duration))
    assert model.position == pytest.approx(80.0)
    controller._on_playback_changed(PlaybackState(stretch.id, PAUSED, 64.0, stretch.duration))
    assert not model.playing and model.position == pytest.approx(64.0)

    model.stop_playback()  # the page left: what plays ends, the session stays
    assert calls.stopped == 1 and model.stretch is stretch
    model.toggle()
    model.close()
    assert calls.stopped == 2 and model.stretch is None


def test_a_title_of_the_ring_is_kept_then_saved_in_its_place(model):
    model, controller, calls = model
    _open(model, controller, calls)
    model.move(START, 52.0)
    model.set_name("Boyzone", "  No  Matter What ")
    assert (model.artist, model.title) == ("Boyzone", "No Matter What")
    model.move(START, 53.0)
    assert model.artist == "Boyzone", "a name given by hand stays"

    model.save()
    (request,) = calls.saved
    assert (request.first, request.last) == (round(53.0 / UNIT_S), 4600)
    assert request.replaces is None and model.saving
    model.save()  # not twice while it saves
    assert len(calls.saved) == 1

    kept = _recording("kept", T0 + 53, 3275, 1325, kept=True)
    saved: list[object] = []
    model.title_saved.connect(saved.append)
    controller.cut_saved.emit(kept, "")
    assert saved == [kept] and model.saved and model.replacing and not model.saving

    model.move(END, 180.0)
    assert not model.saved
    model.save()
    assert calls.saved[-1].replaces == "kept"


def test_a_kept_title_opens_saved_and_a_failure_is_told(model):
    model, controller, calls = model
    kept = _recording("boyzone kept", T0 + 64.0, 3000, 1600, kept=True, artist="B", title="N")
    _open(model, controller, calls, _stretch(kept))
    assert model.replacing and model.saved

    model.move(START, 60.0)
    failed: list[str] = []
    model.save_failed.connect(failed.append)
    model.save()
    controller.cut_failed.emit("Recording folder unavailable")
    assert failed == ["Recording folder unavailable"] and not model.saving and not model.saved


def test_a_recording_that_cannot_be_read_is_told(model):
    model, controller, _calls = model
    failed: list[str] = []
    model.failed.connect(failed.append)
    model.open("gone")
    controller.stretch_failed.emit("other", "nope")
    controller.stretch_failed.emit("gone", "The recording no longer exists")
    assert failed == ["The recording no longer exists"] and not model.loading
