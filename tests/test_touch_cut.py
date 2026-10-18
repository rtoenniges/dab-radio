"""The cutter on the touch surface (`ui.touch.cut_view`): what it shows of a stretch, a
finger on a handle, a tap, a scroll, what is shown, the steps, the names and keeping;
and the way there from the recordings and back."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QWidget

from dsp.superframe import AudioFormat
from net import wifi
from recording.cutter import build_stretch
from recording.levels import STEPS_PER_UNIT
from recording.playback import PLAYING, PlaybackState
from recording.recorder import WAITING, RecorderStatus, Recording
from ui.cut_model import END, START, CutModel
from ui.touch import backlight, cut_view
from ui.touch.cut_view import (
    CutPage,
    clock_text,
    length_text,
    level_columns,
    level_images,
    name_text,
    view_preset,
)
from ui.touch.theme import THEMES
from ui.touch.window import SCREEN_HEIGHT, SCREEN_WIDTH, TouchWindow

FORMAT = AudioFormat(
    dac_rate_48k=True, sbr=True, stereo_core=True, parametric_stereo=False, mpeg_surround_config=0
)
UNIT_S = 0.04
T0 = 1_791_919_609.0


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def _recording(name, start, units, first_unit, *, artist="", title="", labels=(), kept=False):
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


def _stretch():
    units = tuple(n.to_bytes(3, "big") for n in range(4600))
    return build_stretch(BOYZONE, [(ADELE, units[:1600]), (BOYZONE, units[1600:])], FORMAT)


@pytest.fixture
def page(make_controller, data_directory, monkeypatch):
    controller = make_controller(data_directory)
    saved: list[object] = []
    played: list[tuple] = []
    monkeypatch.setattr(controller, "load_stretch", lambda _id: None)
    monkeypatch.setattr(controller, "read_levels", lambda _stretch: None)
    monkeypatch.setattr(controller, "cancel_levels", lambda: None)
    monkeypatch.setattr(controller, "save_cut", saved.append)

    def play(stretch, position, stop, back_to):
        played.append((position, stop, back_to))
        controller._playback_source_id = stretch.id
        controller._on_playback_changed(
            PlaybackState(stretch.id, PLAYING, position, stretch.duration)
        )

    monkeypatch.setattr(controller, "play_stretch", play)
    monkeypatch.setattr(controller, "seek_stretch", lambda *args: played.append(args))
    window = QWidget()
    window.resize(SCREEN_WIDTH, SCREEN_HEIGHT)
    model = CutModel(controller)
    cut = CutPage(model, window)
    cut.resize(SCREEN_WIDTH - 112, SCREEN_HEIGHT)
    window.show()
    yield cut, model, controller, saved, played
    window.deleteLater()


def _open(cut: CutPage, controller) -> object:
    stretch = _stretch()
    cut.open(BOYZONE.id)
    controller.stretch_loaded.emit(stretch)
    return stretch


def test_texts_read_well():
    assert clock_text(T0 + 0.46, tenths=True) == clock_text(T0) + ".4"
    assert length_text(131.64) == "2:11.6" and length_text(5.0) == "0:05.0"
    assert (
        name_text("ADELE", "RAIN") == "ADELE – RAIN"
        and name_text("", "Nachrichten") == "Nachrichten"
    )


def test_the_level_of_a_column_is_the_highest_of_its_units():
    levels = np.array([-30.0, -10.0, np.nan, -50.0], dtype=np.float32)
    # Two columns of two units each, one beyond the end.
    columns = level_columns(levels, np.array([0.0, 0.08, 0.16, 0.24]), UNIT_S)
    np.testing.assert_allclose(columns[:2], [-10.0, -50.0])
    assert np.isnan(columns[2])
    # Zoomed in: several columns over one unit.
    columns = level_columns(levels, np.array([0.0, 0.01, 0.02, 0.05]), UNIT_S)
    np.testing.assert_allclose(columns, [-30.0, -30.0, -30.0])


def test_the_page_shows_the_stretch_opened(page):
    cut, model, controller, _saved, _played = page
    assert not cut.keep_button.isEnabled() and not cut.play_button.isEnabled()
    _open(cut, controller)

    assert cut.summary.text().startswith("Absolut relax · ")
    assert cut.summary.text().endswith(" · 2 recordings")
    assert cut.view_buttons["title"].isChecked()
    # The title with room either side, as far as the stretch goes.
    v0, v1 = cut.wave.view
    assert v0 < model.start - 20 and model.end < v1 <= _stretch().duration + 10
    assert cut.artist_button.text() == "BOYZONE" and cut.title_button.text() == "NO MATTER WHAT"
    assert cut.keep_button.text() == "Keep" and cut.keep_button.isEnabled()
    assert cut.names_button.isEnabled()
    assert not cut.earlier_pause.isEnabled(), "no pause read yet"
    cut.apply_theme(THEMES["light"])
    cut.grab()  # paints without the level, then with it
    controller.levels_read.emit(
        _stretch().id,
        0,
        np.full(4600 * STEPS_PER_UNIT, -20.0, dtype=np.float32),
        np.full(4600 * STEPS_PER_UNIT, -6.0, dtype=np.float32),
    )
    cut.grab()


def test_a_finger_drags_a_handle_and_a_tap_moves_where_it_plays(page):
    cut, model, controller, _saved, _played = page
    _open(cut, controller)
    wave = cut.wave
    y = wave.plot_rect().center().y()

    wave.press(QPointF(wave.x_of(model.end) - 10, y))
    wave.drag_to(QPointF(wave.x_of(150.0) - 10, y))
    assert model.end == pytest.approx(150.0, abs=wave.t_of(1) - wave.t_of(0) + UNIT_S)
    assert model.selected == END
    wave.release(QPointF(wave.x_of(150.0) - 10, y))

    wave.press(QPointF(wave.x_of(100.0), y))
    wave.release(QPointF(wave.x_of(100.0) + 3, y))
    assert model.position == pytest.approx(100.0, abs=0.5)

    # Moved further than a tap: it scrolls the time, and the view follows no handle.
    before = wave.view
    wave.press(QPointF(wave.x_of(100.0), y))
    wave.drag_to(QPointF(wave.x_of(100.0) + 200, y))
    wave.release(QPointF(wave.x_of(100.0) + 200, y))
    assert wave.view[0] < before[0]
    assert not any(button.isChecked() for button in cut.view_buttons.values())


def test_the_level_is_a_picture_mirrored_about_the_middle():
    """Full scale fills the column, −40 dB a tenth of it (the square root of the
    amplitude), silence a breath at the middle; the colour is premultiplied by its
    alpha."""
    levels = np.array([0.0, -40.0, -120.0], dtype=np.float32)
    accent, faint = level_images(
        [levels],
        np.array([0.0, UNIT_S, 2 * UNIT_S, 3 * UNIT_S]),
        UNIT_S,
        100,
        50.0,
        [(QColor("#4fc3f7"), (255,)), (QColor("#4fc3f7"), (128,))],
    )
    assert (accent.width(), accent.height()) == (3, 100)
    assert QColor(accent.pixel(0, 1)).name() == "#4fc3f7" and accent.pixelColor(0, 1).alpha() == 255
    assert accent.pixelColor(1, 50).alpha() == 255 and accent.pixelColor(1, 40).alpha() == 0
    assert accent.pixelColor(1, 45).alpha() == 255 and accent.pixelColor(1, 44).alpha() == 0
    assert accent.pixelColor(2, 50).alpha() < 20 and accent.pixelColor(2, 47).alpha() == 0
    assert faint.pixelColor(0, 50).alpha() == 128


def test_a_handle_dragged_leaves_the_view_and_the_level_drawn_as_they_are(page, monkeypatch):
    """Painting the whole level anew with every move of a handle takes 40 to 80 ms in
    the GUI thread, enough to push the signal processing beyond real time; and in the
    view of the beginning, a view that follows the handle runs the level away beneath
    the finger."""
    cut, model, controller, _saved, _played = page
    _open(cut, controller)
    controller.levels_read.emit(
        _stretch().id,
        0,
        np.full(4600 * STEPS_PER_UNIT, -20.0, dtype=np.float32),
        np.full(4600 * STEPS_PER_UNIT, -6.0, dtype=np.float32),
    )
    cut.view_buttons[START].click()
    wave = cut.wave
    view = wave.view
    wave.grab()
    drawn: list[int] = []
    original = cut_view.level_images
    monkeypatch.setattr(cut_view, "level_images", lambda *args: drawn.append(1) or original(*args))

    y = wave.plot_rect().center().y()
    x = wave.x_of(model.start)
    wave.press(QPointF(x, y))
    for step in range(1, 30):
        wave.drag_to(QPointF(x + 7 * step, y))
        wave.grab()
    assert wave.view == view and model.start > 64.0 + 3.0
    assert drawn == [], "the level is cut anew, not drawn anew"
    wave.release(QPointF(x + 7 * 29, y))
    assert wave.view == view

    # Scrolling draws it anew, as it must.
    wave.press(QPointF(wave.x_of(model.end) - 200, y))
    wave.drag_to(QPointF(wave.x_of(model.end) - 100, y))
    wave.grab()
    assert drawn


def test_the_beginning_and_the_end_are_shown_and_stepped(page):
    cut, model, controller, _saved, _played = page
    _open(cut, controller)

    cut.view_buttons[START].click()
    assert model.selected == START
    assert cut.wave.view == pytest.approx(view_preset(model, START))
    cut.step_buttons[3].click()  # +1 s
    assert model.start == pytest.approx(65.0)
    assert cut.wave.view == pytest.approx(view_preset(model, START)), "the view goes along"

    cut.view_buttons[END].click()
    assert model.selected == END
    cut.step_buttons[0].click()  # -1 s
    assert model.end == pytest.approx(183.0)

    levels = np.full(4600 * STEPS_PER_UNIT, -20.0, dtype=np.float32)
    levels[4000 * STEPS_PER_UNIT : 4010 * STEPS_PER_UNIT] = -91.0  # a pause at 160 s
    controller.levels_read.emit(_stretch().id, 0, levels, levels)
    assert cut.earlier_pause.isEnabled() and not cut.later_pause.isEnabled()
    cut.earlier_pause.click()
    assert model.end == pytest.approx(160.1, abs=UNIT_S)


def test_play_toggles_and_the_icon_follows(page):
    cut, model, controller, _saved, played = page
    _open(cut, controller)
    cut.play_button.click()
    assert played and model.playing


def test_names_the_station_sent_are_offered(page):
    cut, model, controller, _saved, _played = page
    _open(cut, controller)
    cut.names_button.click()
    sheet = cut.name_sheet()
    texts = [button.text() for button in sheet.buttons]
    assert "BOYZONE – NO MATTER WHAT" in texts and "Absolut relax – Entspannt" in texts
    sheet.buttons[texts.index("Absolut relax – Entspannt")].click()
    assert (model.artist, model.title) == ("Absolut relax", "Entspannt")
    assert not sheet.isVisible()

    cut.title_button.click()
    keyboard = cut.keyboard()
    assert keyboard.isVisible()


def test_keep_writes_the_title_and_says_so(page):
    cut, _model, controller, saved, _played = page
    _open(cut, controller)
    cut.keep_button.click()
    assert len(saved) == 1 and not cut.keep_button.isEnabled()

    kept = _recording(
        "kept", T0 + 64.0, 3000, 1600, kept=True, artist="BOYZONE", title="NO MATTER WHAT"
    )
    controller.cut_saved.emit(kept, "")
    assert cut.keep_button.text() == "Keep ✓" and not cut.keep_button.isEnabled()
    assert cut.summary.text() == "Kept: BOYZONE – NO MATTER WHAT · 2:00.0"

    cut.step_buttons[2].click()  # +0.1 s: changed, it may be saved in its place
    assert cut.keep_button.text() == "Save" and cut.keep_button.isEnabled()
    assert cut.summary.text().startswith("Absolut relax")

    controller.cut_failed.emit("Recording folder unavailable")  # not asked: nothing
    cut.keep_button.click()
    controller.cut_failed.emit("Recording folder unavailable")
    assert cut.sheet().isVisible() and cut.sheet().detail.text() == "Recording folder unavailable"


def test_the_cutter_is_reached_from_a_recording_and_left_back_to_them(
    make_controller, data_directory, monkeypatch
):
    monkeypatch.setattr(wifi, "find_nmcli", lambda: None)
    monkeypatch.setattr(backlight, "BACKLIGHT_CLASS", Path("nowhere") / "backlight")
    controller = make_controller(data_directory, start=False)
    requested: list[str] = []
    monkeypatch.setattr(controller, "load_stretch", requested.append)
    window = TouchWindow(controller)
    controller.start()
    window.show()
    radio = window._radio_page
    radio.show_recordings()
    controller._on_recordings_changed((BOYZONE,), ())
    controller._on_recorder_status(RecorderStatus(False, WAITING))

    radio.recordings.ring_page.rows()[0].tapped.emit()
    sheet = radio.recordings.sheet()
    next(button for button in sheet.actions if button.text() == "Cut").click()
    assert requested == [BOYZONE.id]
    assert radio._stack.currentWidget() is radio.cut

    radio.cut.back_button.click()
    assert radio._stack.currentWidget() is radio.recordings
    assert radio.cut._model.stretch is None
    window.deleteLater()
