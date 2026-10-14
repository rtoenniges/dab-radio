"""The cutter without widgets (`recording.cutter`), so far for `ui.touch`: a recording
opened with its neighbours, the level as it is read, the two handles that mark what is
kept, its name, where it plays, and keeping it.

One session makes one title. A title of the ring becomes a kept title of its own with
"Keep", the ring stays as it is; from then on, and for a kept title opened, "Save"
replaces it. Another piece of the same recording is cut by opening it again.

What plays is the stretch, through the player in place of the live audio, as a
recording plays (`ReceiverController.play_stretch`); it halts at the end mark and goes
back to where it began (`AudioPlayer.set_playback_stop`), so the same passage plays
again. While it plays, every move of a handle plays it anew from there: the beginning
from the beginning on, the end its last `PREVIEW_S` up to the end. While paused, a move
only puts the position there.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from recording import cutter
from recording.cutter import CutRequest, Stretch
from recording.levels import STEPS_PER_UNIT
from recording.playback import PAUSED, PlaybackState
from recording.recorder import Recording
from ui.receiver_controller import ReceiverController

START = "start"
END = "end"
# How much of the end plays before the end mark.
PREVIEW_S = 4.0


class CutModel(QObject):
    # A recording opened (`open`) is there with its neighbours, or could not be read
    # (translated).
    opened = pyqtSignal()
    failed = pyqtSignal(str)
    # The handles, the name, where it plays and whether, the titles kept: anything shown
    # but the level.
    changed = pyqtSignal()
    levels_changed = pyqtSignal()
    # The title kept or saved, or why not (translated).
    title_saved = pyqtSignal(object)  # Recording
    save_failed = pyqtSignal(str)

    def __init__(self, controller: ReceiverController, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._opening: str | None = None
        self._stretch: Stretch | None = None
        self._rms = np.empty(0, dtype=np.float32)
        self._peak = np.empty(0, dtype=np.float32)
        self._read = 0  # units with a level so far
        self._levels_done = False
        self._levels_version = 0
        self._pauses: list[tuple[float, float]] = []
        # Of the labels of the stretch, which do not change: computed once.
        self._marks: list[tuple[float, str]] = []
        self._suggestions: list[tuple[str, str]] = []
        self._start = 0.0
        self._end = 0.0
        self._selected = START
        self._artist = ""
        self._title = ""
        self._named = False  # the name given by hand: it no longer follows the handles
        # Where it plays or would, up to where and back to where (`play_stretch`).
        self._position = 0.0
        self._stop = 0.0
        self._back_to = 0.0
        self._playing = False
        self._loaded_playback = False  # the player has the stretch, playing or paused
        self._replaces: str | None = None  # the kept title "Save" replaces
        self._saved: tuple | None = None  # what was kept last, as `_state_key` has it
        self._saving = False
        self._kept: list[tuple[float, float]] = []

        controller.stretch_loaded.connect(self._on_stretch_loaded)
        controller.stretch_failed.connect(self._on_stretch_failed)
        controller.levels_read.connect(self._on_levels_read)
        controller.levels_finished.connect(self._on_levels_finished)
        controller.playback_changed.connect(self._on_playback_changed)
        controller.recordings_changed.connect(self._on_recordings_changed)
        controller.cut_saved.connect(self._on_cut_saved)
        controller.cut_failed.connect(self._on_cut_failed)

    # --- What is shown ----------------------------------------------------------------

    @property
    def stretch(self) -> Stretch | None:
        return self._stretch

    @property
    def loading(self) -> bool:
        """A recording opened is being read."""
        return self._opening is not None and self._stretch is None

    @property
    def rms_db(self) -> np.ndarray:
        """The level in dBFS every `level_step_seconds` (`STEPS_PER_UNIT` a unit), NaN
        where not read yet."""
        return self._rms

    @property
    def peak_db(self) -> np.ndarray:
        return self._peak

    @property
    def levels_read(self) -> float:
        """How much of the level is read, 0 to 1."""
        if self._levels_done or not len(self._rms):
            return 1.0 if self._stretch is not None else 0.0
        return self._read / len(self._rms)

    @property
    def pauses(self) -> list[tuple[float, float]]:
        return self._pauses

    @property
    def level_step_seconds(self) -> float:
        """How much of the stretch a level stands for: an eighth of a unit."""
        stretch = self._stretch
        return 1.0 if stretch is None else stretch.unit_seconds / STEPS_PER_UNIT

    @property
    def levels_version(self) -> int:
        """Counts up with every level read: what is drawn of it is drawn anew."""
        return self._levels_version

    @property
    def marks(self) -> list[tuple[float, str]]:
        """Where a dynamic label appears first, on the stretch."""
        return self._marks

    @property
    def suggestions(self) -> list[tuple[str, str]]:
        return self._suggestions

    @property
    def kept(self) -> list[tuple[float, float]]:
        """Where titles kept lie on the stretch."""
        return self._kept

    @property
    def start(self) -> float:
        return self._start

    @property
    def end(self) -> float:
        return self._end

    @property
    def selected(self) -> str:
        """The handle the steps and the pauses move: START or END."""
        return self._selected

    @property
    def artist(self) -> str:
        return self._artist

    @property
    def title(self) -> str:
        return self._title

    @property
    def position(self) -> float:
        return self._position

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def replacing(self) -> bool:
        """ "Save" replaces a kept title; "Keep" makes one."""
        return self._replaces is not None

    @property
    def saved(self) -> bool:
        """What the handles and the name say is kept as it stands."""
        return self._saved is not None and self._saved == self._state_key()

    @property
    def saving(self) -> bool:
        return self._saving

    def clock(self, seconds: float) -> float:
        """The Unix time at `seconds` on the stretch."""
        return 0.0 if self._stretch is None else self._stretch.origin + seconds

    # --- Opening and closing ----------------------------------------------------------

    def open(self, recording_id: str) -> None:
        """Reads a title with its neighbours (`opened`, `failed`)."""
        self.close()
        self._opening = recording_id
        self._controller.load_stretch(recording_id)
        self.changed.emit()

    def close(self) -> None:
        """Gives the session up: its playback ends, the station is heard again."""
        if self._stretch is not None:
            self._controller.cancel_levels()
            if self._controller.playback_source_id == self._stretch.id:
                self._controller.stop_playback()
        self._opening = None
        self._stretch = None
        self._rms = self._peak = np.empty(0, dtype=np.float32)
        self._read = 0
        self._levels_done = False
        self._levels_version += 1
        self._pauses = []
        self._marks = []
        self._suggestions = []
        self._playing = False
        self._loaded_playback = False
        self._replaces = None
        self._saved = None
        self._saving = False
        self._kept = []

    def stop_playback(self) -> None:
        """Ends what plays of the stretch (the page left), keeping the session."""
        if self._stretch is not None and self._controller.playback_source_id == self._stretch.id:
            self._controller.stop_playback()

    def _on_stretch_loaded(self, stretch: Stretch) -> None:
        if stretch.opened.id != self._opening:
            return
        self._stretch = stretch
        count = len(stretch.units) * STEPS_PER_UNIT
        self._rms = np.full(count, np.nan, dtype=np.float32)
        self._peak = np.full(count, np.nan, dtype=np.float32)
        self._read = 0
        self._levels_done = False
        self._levels_version += 1
        self._pauses = []
        self._marks = cutter.label_marks(stretch)
        self._suggestions = cutter.name_suggestions(stretch)
        clip = stretch.opened_clip()
        self._start = stretch.snap(clip.start)
        self._end = stretch.snap(clip.end)
        self._selected = START
        self._named = False
        self._artist, self._title = cutter.default_name(stretch, self._start, self._end)
        self._position = self._start
        self._stop = self._end
        self._back_to = self._start
        opened = stretch.opened
        self._replaces = opened.id if opened.kept else None
        # A kept title as it is needs no saving.
        self._saved = self._state_key() if opened.kept else None
        self._update_kept()
        self._controller.read_levels(stretch)
        self.opened.emit()
        self.changed.emit()

    def _on_stretch_failed(self, recording_id: str, message: str) -> None:
        if recording_id != self._opening:
            return
        self._opening = None
        self.failed.emit(message)
        self.changed.emit()

    def _on_levels_read(self, stretch_id: str, first: int, rms: object, peak: object) -> None:
        stretch = self._stretch
        if stretch is None or stretch_id != stretch.id:
            return
        end = min(first + len(rms), len(self._rms))
        self._rms[first:end] = rms[: end - first]
        self._peak[first:end] = peak[: end - first]
        self._read = max(self._read, end)
        self._levels_version += 1
        self._pauses = cutter.find_pauses(self._rms, self.level_step_seconds)
        self.levels_changed.emit()

    def _on_levels_finished(self, stretch_id: str) -> None:
        if self._stretch is not None and stretch_id == self._stretch.id:
            self._levels_done = True
            self.levels_changed.emit()

    def _on_recordings_changed(self) -> None:
        if self._stretch is not None:
            self._update_kept()
            self.changed.emit()

    def _update_kept(self) -> None:
        assert self._stretch is not None
        self._kept = cutter.kept_spans(self._stretch, self._controller.kept_recordings)

    # --- The handles ------------------------------------------------------------------

    def select(self, handle: str) -> None:
        if handle != self._selected:
            self._selected = handle
            self.changed.emit()

    def move(self, handle: str, seconds: float, preview: bool = True) -> None:
        """Puts a handle at `seconds` (on a unit boundary, the other at least
        `MIN_CUT_S` away) and chooses it; `preview`: the move is over, it plays from
        there if playing (a finger dragging passes False until it lets go)."""
        stretch = self._stretch
        if stretch is None:
            return
        self._selected = handle
        if handle == START:
            self._start = stretch.snap(min(max(0.0, seconds), self._end - cutter.MIN_CUT_S))
        else:
            self._end = stretch.snap(
                max(min(stretch.duration, seconds), self._start + cutter.MIN_CUT_S)
            )
        if not self._named:
            self._artist, self._title = cutter.default_name(stretch, self._start, self._end)
        if preview:
            self._preview(handle)
        self.changed.emit()

    def let_go(self, handle: str) -> None:
        """The finger dragging a handle let go: it plays from there if playing."""
        self._preview(handle)
        self.changed.emit()

    def step(self, seconds: float) -> None:
        """Moves the handle chosen by `seconds`."""
        value = self._start if self._selected == START else self._end
        self.move(self._selected, value + seconds)

    def jump_to_pause(self, forward: bool) -> bool:
        """Moves the handle chosen to the next pause after it, or before it; whether
        there was one."""
        target = self._pause_target(forward)
        if target is None:
            return False
        self.move(self._selected, target)
        return True

    def can_jump(self, forward: bool) -> bool:
        """Whether there is a pause after the handle chosen, or before it."""
        return self._stretch is not None and self._pause_target(forward) is not None

    def _pause_target(self, forward: bool) -> float | None:
        stretch = self._stretch
        if stretch is None:
            return None
        beginning = self._selected == START
        value = self._start if beginning else self._end
        # A pause the handle already stands at, but for the unit it snaps to, is no
        # pause to jump to: the button stays dark rather than jump on the spot.
        here = stretch.unit_seconds / 2
        return cutter.pause_target(self._pauses, value, beginning, forward, here)

    def _preview(self, handle: str) -> None:
        """Where the handle's passage plays from: the beginning on, the last seconds up to
        the end. Playing, it plays from there at once; else the position goes there."""
        if handle == START:
            position = self._start
        else:
            position = max(self._start, self._end - PREVIEW_S)
        self._go_to(position)

    # --- Playing ----------------------------------------------------------------------

    def tap(self, seconds: float) -> None:
        """Plays from `seconds` if playing, else puts the position there."""
        if self._stretch is None:
            return
        self._go_to(min(max(0.0, seconds), self._stretch.duration))
        self.changed.emit()

    def toggle(self) -> None:
        """Plays from the position, or pauses."""
        stretch = self._stretch
        if stretch is None:
            return
        controller = self._controller
        playback = controller.playback
        ours = playback is not None and controller.playback_source_id == stretch.id
        if ours and playback.state != PAUSED:
            controller.pause_playback()
        elif ours:
            controller.resume_playback()
        else:
            self._playing = True
            controller.play_stretch(stretch, self._position, self._stop, self._back_to)
        self.changed.emit()

    def _go_to(self, position: float) -> None:
        """Plays up to the end mark from `position` before it, beyond it to the end of
        the stretch, and back to `position` then."""
        stretch = self._stretch
        assert stretch is not None
        self._position = position
        self._stop = self._end if position < self._end - 1e-6 else stretch.duration
        self._back_to = position
        if self._loaded_playback:
            self._controller.seek_stretch(position, self._stop, self._back_to)

    def _on_playback_changed(self, state: PlaybackState | None) -> None:
        stretch = self._stretch
        if stretch is None:
            return
        ours = state is not None and self._controller.playback_source_id == stretch.id
        playing = ours and state.state != PAUSED
        if ours:
            self._position = state.position
        if (playing, ours) != (self._playing, self._loaded_playback) or ours:
            self._playing = playing
            self._loaded_playback = ours
            self.changed.emit()

    # --- The name and keeping ---------------------------------------------------------

    def set_name(self, artist: str, title: str) -> None:
        """The name of the cut; given by hand, it no longer follows the handles."""
        self._artist = " ".join(artist.split())
        self._title = " ".join(title.split())
        self._named = True
        self.changed.emit()

    def save(self) -> None:
        """Keeps what lies between the handles (`saved`, `save_failed`)."""
        stretch = self._stretch
        if stretch is None or self._saving:
            return
        self._saving = True
        self._controller.save_cut(
            CutRequest(
                stretch,
                stretch.unit_at(self._start),
                stretch.unit_at(self._end),
                self._artist,
                self._title,
                self._replaces,
            )
        )
        self.changed.emit()

    def _state_key(self) -> tuple:
        stretch = self._stretch
        if stretch is None:
            return ()
        return (stretch.unit_at(self._start), stretch.unit_at(self._end), self._artist, self._title)

    def _on_cut_saved(self, recording: Recording, _replaced: str) -> None:
        if not self._saving:
            return
        self._saving = False
        self._replaces = recording.id
        self._saved = self._state_key()
        self.title_saved.emit(recording)
        self.changed.emit()

    def _on_cut_failed(self, message: str) -> None:
        if not self._saving:
            return
        self._saving = False
        self.save_failed.emit(message)
        self.changed.emit()
