"""The recordings: a sub-page of the radio page, reached by the badge on
the corner of the slide (`ui.touch.radio_page`) or from the settings
(`ui.touch.storage_view`). Two pages to swipe (`ui.touch.pager`):

- the last titles: the ring (`recording.recorder`), newest on top, the title recording
  right now above them with a red dot. A tap offers to play a title, to cut it (the
  cutter, `ui.touch.cut_view`), to keep it (out of the ring, into the directory for
  good) or to delete it; the title recording can be marked to be kept as soon as it is
  complete.
- the kept titles, newest on top; a tap offers to play, to cut or to delete one.

A recording playing (`ReceiverController.play_recording`, in place of the live audio)
has the bar beneath the pages: pause and go on, its name, where it is as a slider to
drag or to tap, and stop; its row carries ▶. Leaving the page (by "Back" or through
the menu), stop and the end of the title give the station back.

The line beside the title says what the recorder does: the title it records with its
length so far, or why it does not record (no folder, no drive, no station, ...).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QHideEvent, QMouseEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from i18n import tr, tr_noop, tr_plural
from recording.playback import LOADING, PAUSED, PlaybackState
from recording.recorder import RecorderStatus, Recording
from ui.receiver_controller import ReceiverController
from ui.touch.fine_slider import FineSlider
from ui.touch.icons import symbol_icon
from ui.touch.overlay import ActionSheet
from ui.touch.pager import Pager
from ui.touch.pages import Page
from ui.touch.theme import DARK, THEMES, Theme
from ui.touch.widgets import ElidedLabel, TapWidget, repolish, set_role

ROW_HEIGHT = 64
MARK_WIDTH = 28
RUNNING_MARK = "●"
KEPT_MARK = "✓"
PLAYING_MARK = "▶"
# The bar of the recording playing: its buttons, the room for its texts and times.
BAR_BUTTON = 72
BAR_ICON = QSize(32, 32)
BAR_TEXT_WIDTH = 360
TIMES_WIDTH = 190
# The spacings of the ticks on the scale of the loupe while seeking, in seconds.
SEEK_TICKS = (1, 5, 10, 30, 60, 120, 600, 1200, 3600)
_NO_TITLES = tr_noop("No title recorded yet.")
_NOTHING_KEPT = tr_noop("Nothing kept yet.")
# Translators: a title being recorded right now.
_RUNNING = tr_noop("recording")
# Translators: the title being recorded is kept once it is complete.
_MARKED = tr_noop("will be kept")
_LOADING = tr_noop("Loading …")
_CANNOT_PLAY = tr_noop("Cannot play")


def clock_text(timestamp: float) -> str:
    return time.strftime("%H:%M", time.localtime(timestamp))


def length_text(seconds: float) -> str:
    """ "3:45", "1:02:03"."""
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def count_text(count: int) -> str:
    return tr_plural("{count} title", "{count} titles", count).format(count=count)


def status_text(status: RecorderStatus) -> str:
    """What the recorder does, for the header: the title recording with its length,
    or why it does not record."""
    if status.recording and status.current is not None:
        return f"{status.message}: {status.current.name} · {length_text(status.current.duration)}"
    return status.message


def times_text(state: PlaybackState, position: float | None = None) -> str:
    """ "1:05 / 3:20" for the bar, the position dragged to while the finger is on the
    slider; "Loading …" while the file is read."""
    if state.state == LOADING:
        return tr(_LOADING)
    position = state.position if position is None else position
    return f"{length_text(position)} / {length_text(state.duration)}"


@dataclass(frozen=True)
class RecordingEntry:
    """A line of either list."""

    id: str
    name: str
    detail: str
    running: bool
    marked: bool  # to be kept once complete, or kept
    playing: bool = False


def entry_detail(recording: Recording, running: bool = False, marked: bool = False) -> str:
    """ "Dlf · 18:32 · 3:45", "· recording" while recording, "· will be kept" once
    marked."""
    parts = [recording.station, clock_text(recording.start), length_text(recording.duration)]
    if running:
        parts.append(tr(_RUNNING))
    if running and marked:
        parts.append(tr(_MARKED))
    return " · ".join(part for part in parts if part)


def ring_entries(
    ring: tuple[Recording, ...], status: RecorderStatus, playing: str | None = None
) -> list[RecordingEntry]:
    """The title recording first, then the ring newest first; `playing` is the id of
    the recording playing, if any."""
    entries = []
    current = status.current if status.recording else None
    if current is not None:
        entries.append(
            RecordingEntry(
                current.id,
                current.name,
                entry_detail(current, True, current.kept),
                True,
                current.kept,
            )
        )
    entries.extend(
        RecordingEntry(r.id, r.name, entry_detail(r), False, False, r.id == playing)
        for r in reversed(ring)
    )
    return entries


def kept_entries(kept: tuple[Recording, ...], playing: str | None = None) -> list[RecordingEntry]:
    return [
        RecordingEntry(r.id, r.name, entry_detail(r), False, True, r.id == playing)
        for r in reversed(kept)
    ]


class _RecordingRow(TapWidget):
    """A red dot for the title recording, ▶ for the one playing or ✓ for one to be
    kept, the name with a line beneath. It lights up while pressed."""

    def __init__(self, entry: RecordingEntry, theme: Theme) -> None:
        super().__init__()
        set_role(self, "row")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setMinimumHeight(ROW_HEIGHT)
        self.entry = entry
        self._mark = set_role(QLabel(), "large")
        self._mark.setFixedWidth(MARK_WIDTH)
        self._name = ElidedLabel()
        self._detail = set_role(ElidedLabel(), "small")
        texts = QVBoxLayout()
        texts.setContentsMargins(0, 0, 0, 0)
        texts.setSpacing(0)
        texts.addWidget(self._name)
        texts.addWidget(self._detail)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 12, 4)
        layout.setSpacing(12)
        layout.addWidget(self._mark)
        layout.addLayout(texts, stretch=1)
        self.show_entry(entry, theme)

    def show_entry(self, entry: RecordingEntry, theme: Theme) -> None:
        self.entry = entry
        if entry.running:
            self._mark.setText(f'<span style="color: {theme.danger}">{RUNNING_MARK}</span>')
        elif entry.playing:
            self._mark.setText(f'<span style="color: {theme.accent}">{PLAYING_MARK}</span>')
        elif entry.marked:
            self._mark.setText(f'<span style="color: {theme.accent}">{KEPT_MARK}</span>')
        else:
            self._mark.setText("")
        self._name.setText(entry.name)
        self._detail.setText(entry.detail)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._set_pressed(True)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._set_pressed(False)
        super().mouseReleaseEvent(event)

    def _set_pressed(self, pressed: bool) -> None:
        self.setProperty("pressed", pressed)
        repolish(self)


class _ListPage(Page):
    """A page of the pager: its name and the count, the rows in a panel beneath. The
    pager scrolls the list (no QScroller, see `ui.touch.pager`)."""

    def __init__(self, title: str, empty: str) -> None:
        super().__init__(title)
        self._theme = THEMES[DARK]
        self._empty_text = empty
        self._rows: list[_RecordingRow] = []
        self._shown: list[RecordingEntry] = []
        self.tapped: list = []  # the slots called with the entry of a tapped row

        self._list_layout = QVBoxLayout()
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(0)
        self._list_layout.addStretch()
        rows = set_role(QWidget(), "transparent")
        rows.setLayout(self._list_layout)
        self._area = set_role(QScrollArea(), "transparent")
        self._area.setWidget(rows)
        self._area.setWidgetResizable(True)
        self._area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        set_role(self._area.viewport(), "transparent")
        self._empty = set_role(QLabel(empty), "muted")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        panel = set_role(QWidget(), "panel")
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.addWidget(self._area, stretch=1)
        panel_layout.addWidget(self._empty, stretch=1)
        self.body.addWidget(panel, stretch=1)

    def render(self) -> None:
        pass

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        for row in self._rows:
            row.show_entry(row.entry, theme)

    def rows(self) -> list[_RecordingRow]:
        return list(self._rows)

    def entries(self) -> list[RecordingEntry]:
        return list(self._shown)

    def show_entries(self, entries: list[RecordingEntry]) -> None:
        """The rows anew where titles came or went; otherwise each row refreshed in
        place (the length of the title recording grows)."""
        self.status.setText(count_text(len(entries)) if entries else "")
        if [e.id for e in entries] != [e.id for e in self._shown]:
            for row in self._rows:
                self._list_layout.removeWidget(row)
                row.hide()
                row.deleteLater()
            self._rows = []
            for entry in entries:
                row = _RecordingRow(entry, self._theme)
                row.tapped.connect(lambda r=row: self._on_tapped(r.entry))
                self._list_layout.insertWidget(self._list_layout.count() - 1, row)
                self._rows.append(row)
        else:
            for row, entry in zip(self._rows, entries, strict=True):
                if entry != row.entry:
                    row.show_entry(entry, self._theme)
        self._shown = list(entries)
        self._area.setVisible(bool(entries))
        self._empty.setVisible(not entries)

    def _on_tapped(self, entry: RecordingEntry) -> None:
        for slot in self.tapped:
            slot(entry)


class _PlaybackBar(QWidget):
    """The recording playing: pause and go on, its name and station, where it is as a
    slider to drag or to tap, the times, and stop. Shown only while one is loaded."""

    toggled = pyqtSignal()
    stopped = pyqtSignal()
    sought = pyqtSignal(float)

    def __init__(self) -> None:
        super().__init__()
        set_role(self, "panel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self._theme = THEMES[DARK]
        self._state: PlaybackState | None = None

        self.toggle_button = QToolButton()
        self.toggle_button.setFixedSize(BAR_BUTTON, BAR_BUTTON)
        self.toggle_button.setIconSize(BAR_ICON)
        self.toggle_button.clicked.connect(self.toggled)
        self.name = ElidedLabel()
        self.detail = set_role(ElidedLabel(), "small")
        texts = QVBoxLayout()
        texts.setContentsMargins(0, 0, 0, 0)
        texts.setSpacing(0)
        texts.addWidget(self.name)
        texts.addWidget(self.detail)
        texts_widget = set_role(QWidget(), "transparent")
        texts_widget.setLayout(texts)
        texts_widget.setFixedWidth(BAR_TEXT_WIDTH)
        # A finger coming down beside the handle moves it there and drags on from there;
        # pulled up, it seeks finer (`ui.touch.fine_slider`).
        self.slider = FineSlider(length_text, SEEK_TICKS, length_text, jump=True)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._show_times)
        self.slider.sliderReleased.connect(self._on_released)
        self.times = set_role(QLabel(), "muted")
        self.times.setFixedWidth(TIMES_WIDTH)
        self.times.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.stop_button = QToolButton()
        self.stop_button.setFixedSize(BAR_BUTTON, BAR_BUTTON)
        self.stop_button.setIconSize(BAR_ICON)
        self.stop_button.clicked.connect(self.stopped)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(16)
        layout.addWidget(self.toggle_button)
        layout.addWidget(texts_widget)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.times)
        layout.addWidget(self.stop_button)
        self.hide()

    def set_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.slider.set_theme(theme)
        self._show_icons()

    def show_state(self, state: PlaybackState | None, recording: Recording | None) -> None:
        """What plays and where it is; hidden with None."""
        self._state = state
        self.setVisible(state is not None)
        if state is None:
            return
        if recording is not None:
            self.name.setText(recording.name)
            self.detail.setText(f"{recording.station} · {clock_text(recording.start)}")
        duration = max(0, round(state.duration))
        if self.slider.maximum() != duration:
            self.slider.setRange(0, duration)
        if not self.slider.isSliderDown():
            self.slider.setValue(round(state.position))
        self.toggle_button.setEnabled(state.state != LOADING)
        self._show_icons()
        self._show_times()

    def _show_icons(self) -> None:
        color = self._theme.color("text")
        paused = self._state is not None and self._state.state == PAUSED
        self.toggle_button.setIcon(symbol_icon("play" if paused else "pause", color, BAR_ICON))
        self.stop_button.setIcon(symbol_icon("stop", color, BAR_ICON))

    def _show_times(self) -> None:
        state = self._state
        if state is None:
            return
        dragged = float(self.slider.value()) if self.slider.isSliderDown() else None
        self.times.setText(times_text(state, dragged))

    def _on_released(self) -> None:
        self.sought.emit(float(self.slider.value()))


class RecordingsPage(QWidget):
    """Title, what the recorder does and the way back above two pages: the last titles
    and the kept ones; beneath them the bar of the recording playing."""

    # "Cut" in the sheet of a title: the radio page opens the cutter (`ui.touch.cut_view`).
    cut_requested = pyqtSignal(str)

    def __init__(self, controller: ReceiverController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._theme = THEMES[DARK]
        self._sheet: ActionSheet | None = None
        self._playing_id: str | None = None

        title = set_role(QLabel(tr("Recordings")), "large")
        self.summary = set_role(ElidedLabel(), "muted")
        self.back_button = QPushButton(tr("Back"))
        header = QHBoxLayout()
        header.setSpacing(16)
        header.addWidget(title)
        header.addWidget(self.summary, stretch=1)
        header.addWidget(self.back_button)

        self.ring_page = _ListPage(tr("Recent titles"), tr(_NO_TITLES))
        self.ring_page.tapped.append(self._on_ring_tapped)
        # Translators: the page of the titles kept for good, out of the ring.
        self.kept_page = _ListPage(tr("Kept"), tr(_NOTHING_KEPT))
        self.kept_page.tapped.append(self._on_kept_tapped)
        self.pager = Pager()
        self.pager.add_page(self.ring_page)
        self.pager.add_page(self.kept_page)
        self.bar = _PlaybackBar()
        self.bar.toggled.connect(controller.toggle_playback)
        self.bar.stopped.connect(controller.stop_playback)
        self.bar.sought.connect(controller.seek_playback)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(16)
        layout.addLayout(header)
        layout.addWidget(self.pager, stretch=1)
        layout.addWidget(self.bar)

        controller.recordings_changed.connect(self._show)
        controller.recorder_status_changed.connect(self._on_status)
        controller.playback_changed.connect(self._on_playback_changed)
        controller.playback_failed.connect(self._on_playback_failed)
        self._show()
        self._on_playback_changed(controller.playback)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        self.pager.set_theme(theme)
        self.ring_page.apply_theme(theme)
        self.kept_page.apply_theme(theme)
        self.bar.set_theme(theme)

    def sheet(self) -> ActionSheet:
        if self._sheet is None:
            self._sheet = ActionSheet(self.window())
        return self._sheet

    def show_first_page(self) -> None:
        self.pager.set_current_index(0)

    def hideEvent(self, event: QHideEvent) -> None:
        """Leaving the page, by "Back" or any other way (another page of the menu),
        ends the recording playing: the station is heard again. Qt sends the event to
        the page also when a page above it is hidden."""
        self._controller.stop_playback()
        super().hideEvent(event)

    def _on_status(self, _status: object) -> None:
        self._show()

    def _show(self) -> None:
        controller = self._controller
        status = controller.recorder_status
        playing = self._playing_id
        self.summary.setText(status_text(status))
        self.ring_page.show_entries(ring_entries(controller.recordings, status, playing))
        self.kept_page.show_entries(kept_entries(controller.kept_recordings, playing))

    def _on_playback_changed(self, state: object) -> None:
        self.bar.show_state(state, self._controller.playback_recording)
        playing = None if state is None else state.recording_id
        if playing != self._playing_id:
            self._playing_id = playing
            self._show()

    def _on_playback_failed(self, message: str) -> None:
        self.sheet().open(tr(_CANNOT_PLAY), message, [])

    def _play_action(self, entry: RecordingEntry) -> tuple[str, str, object]:
        controller = self._controller
        if entry.playing:
            return (tr("Stop"), "", controller.stop_playback)
        return (tr("Play"), "", lambda: controller.play_recording(entry.id))

    def _on_ring_tapped(self, entry: RecordingEntry) -> None:
        controller = self._controller
        if entry.running:
            if entry.marked:
                self.sheet().open(entry.name, tr("Will be kept once the title is over."), [])
            else:
                self.sheet().open(
                    entry.name,
                    tr("{detail}. Keep it once the title is over?").format(detail=entry.detail),
                    # Translators: an action: keep the title for good, out of the ring.
                    [(tr("Keep"), "accent", lambda: controller.keep_recording(entry.id))],
                )
            return
        self.sheet().open(
            entry.name,
            entry.detail,
            [
                self._play_action(entry),
                self._cut_action(entry),
                (tr("Delete"), "danger", lambda: controller.delete_recording(entry.id)),
                (tr("Keep"), "accent", lambda: controller.keep_recording(entry.id)),
            ],
        )

    def _on_kept_tapped(self, entry: RecordingEntry) -> None:
        controller = self._controller
        self.sheet().open(
            entry.name,
            entry.detail,
            [
                self._play_action(entry),
                self._cut_action(entry),
                (tr("Delete"), "danger", lambda: controller.delete_recording(entry.id)),
            ],
        )

    def _cut_action(self, entry: RecordingEntry) -> tuple[str, str, object]:
        # In the yellow of the notices, so that it stands out beside "Play" (the user's
        # wish, 2026-10-14).
        # Translators: an action: open a recording in the cutter.
        return (tr("Cut"), "notice", lambda: self.cut_requested.emit(entry.id))
