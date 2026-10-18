"""The sub-page "Storage" of the settings: where the recordings go, and the USB drives
to put them on.

The card "Recordings" holds the directory chosen for the recordings
(`ReceiverController.recording_directory`) with the free space of its drive beneath, or
that the drive is not there right now; a tap on it opens the folder browser
(`ui.touch.folder_browser`) on the drive the directory lies on, else on the first drive
mounted. The list "Drives" beneath holds the USB drives UDisks2 knows (`storage.volumes`):
a tap on a mounted one offers to choose a folder on it or to eject it, one on a drive
not mounted mounts it. On a PC the list shows what the system has mounted, and only
the folder can be chosen there. The ring and the start delay are on the sub-page
"Recording" (`ui.touch.recording_view`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QHideEvent, QMouseEvent, QShowEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QScroller,
    QVBoxLayout,
    QWidget,
)

from i18n import tr, tr_noop
from storage.volumes import Volume, VolumeManager, free_space, size_text
from ui.receiver_controller import ReceiverController
from ui.touch.cards import Card, Row, heading_label
from ui.touch.folder_browser import SEPARATOR, FolderBrowser
from ui.touch.icons import symbol_icon, symbol_pixmap
from ui.touch.overlay import ActionSheet
from ui.touch.theme import DARK, THEMES, Theme
from ui.touch.widgets import ElidedLabel, TapWidget, repolish, set_role

ROW_HEIGHT = 64
KIND_ICON = 28
REFRESH_ICON = QSize(32, 32)
CHECK_MARK = "✓"

_HINT = tr_noop("A USB drive plugged in is mounted automatically. Eject it before unplugging.")
_NO_UDISKS = tr_noop(
    "UDisks2 is missing, so the radio cannot mount anything: on the Pi "
    "“sudo apt install udisks2”, plus the polkit rule from the folder pi/."
)
_SYSTEM_MOUNTS = tr_noop("Here the system mounts the drives; only the folder can be chosen.")
_NO_DIRECTORY = tr_noop("No folder chosen")
_NO_VOLUMES = tr_noop("No USB drive found.")
_NO_MOUNTED_VOLUME = tr_noop("No drive mounted to choose a folder on.")


@dataclass(frozen=True)
class VolumeEntry:
    """A line of the list of drives."""

    path: str
    name: str
    detail: str
    mounted: bool
    holds_recordings: bool  # the directory for the recordings lies on it


def volume_detail(volume: Volume) -> str:
    """ "31 GB, vfat, mounted at /media/pi/USB-STICK" or "…, not mounted"."""
    parts = [size_text(volume.size)]
    if volume.filesystem:
        parts.append(volume.filesystem)
    if volume.mounted:
        parts.append(tr("mounted at {mount_point}").format(mount_point=volume.mount_point))
    else:
        parts.append(tr("not mounted"))
    if volume.read_only:
        parts.append(tr("read-only"))
    return ", ".join(parts)


def volume_entries(volumes: list[Volume], recordings: Volume | None) -> list[VolumeEntry]:
    """The lines of the list, in the order `VolumeManager.volume_list` gives."""
    return [
        VolumeEntry(
            volume.path,
            volume.name,
            volume_detail(volume),
            volume.mounted,
            recordings is not None and volume.path == recordings.path,
        )
        for volume in volumes
    ]


def directory_text(directory: Path | None, volume: Volume | None) -> str:
    """ "USB-Stick › Recordings" where the directory lies on a volume known, else the
    path as stored; "No folder chosen" without one."""
    if directory is None:
        return tr(_NO_DIRECTORY)
    if volume is None:
        return str(directory)
    parts = [volume.name, *directory.relative_to(Path(volume.mount_point)).parts]
    return SEPARATOR.join(part for part in parts if part)


def free_text(space: tuple[int, int] | None) -> str:
    """ "12 GB free of 31 GB"; "" where the drive cannot be asked."""
    if space is None:
        return ""
    free, total = space
    return tr("{free} free of {total}").format(free=size_text(free), total=size_text(total))


class _VolumeRow(TapWidget):
    """✓ for the drive holding the recordings, the name with a line beneath, and the
    symbol of a USB stick. It lights up while pressed."""

    def __init__(self, entry: VolumeEntry, theme: Theme) -> None:
        super().__init__()
        set_role(self, "row")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setMinimumHeight(ROW_HEIGHT)
        self.entry = entry
        mark = set_role(QLabel(), "large")
        mark.setFixedWidth(28)
        if entry.holds_recordings:
            mark.setText(f'<span style="color: {theme.accent}">{CHECK_MARK}</span>')
        name = ElidedLabel()
        name.setText(entry.name)
        if not entry.mounted:
            set_role(name, "muted")
        texts = QVBoxLayout()
        texts.setContentsMargins(0, 0, 0, 0)
        texts.setSpacing(0)
        texts.addWidget(name)
        # Elided, not wrapped: a mount point on a PC can be long, and a plain label
        # would push the symbol out of the row.
        detail = set_role(ElidedLabel(), "small")
        detail.setText(entry.detail)
        texts.addWidget(detail)
        kind = QLabel()
        kind.setFixedWidth(KIND_ICON)
        kind.setPixmap(
            symbol_pixmap("usb", theme.color("muted"), KIND_ICON, self.devicePixelRatioF())
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 12, 4)
        layout.setSpacing(12)
        layout.addWidget(mark)
        layout.addLayout(texts, stretch=1)
        layout.addWidget(kind)

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


class StorageView(QWidget):
    """The card "Recordings" (the folder, a tap chooses another) and the list "Drives"
    beneath, which fills what the page leaves, with the button to look again under it.
    `message` (errors, what is missing) is for the settings page to place in its bottom
    row. Reads the drives anew while shown (`VolumeManager.start`); the state it opens
    with was read at the start of the radio (`VolumeManager.watch`)."""

    def __init__(
        self,
        controller: ReceiverController,
        volumes: VolumeManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._volumes = volumes
        self._theme = THEMES[DARK]
        self._browser: FolderBrowser | None = None
        self._sheet: ActionSheet | None = None
        self._rows: list[_VolumeRow] = []
        self._shown_entries: list[VolumeEntry] | None = None
        self._error = ""

        self.message = set_role(QLabel(), "small")

        # The directory as the title, the space left beneath; the row opens the browser.
        self._choose = Row(chevron=True)
        self._choose.tapped.connect(self._choose_directory)
        self._directory = self._choose.title
        self._space = self._choose.subtitle

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
        QScroller.grabGesture(
            self._area.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture
        )
        self._empty = set_role(QLabel(), "muted")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setWordWrap(True)
        self._list_panel = set_role(QWidget(), "panel")
        self._list_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        panel_layout = QVBoxLayout(self._list_panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.addWidget(self._area, stretch=1)
        panel_layout.addWidget(self._empty, stretch=1)
        drives = set_role(QWidget(), "transparent")
        drives_layout = QVBoxLayout(drives)
        drives_layout.setContentsMargins(0, 0, 0, 0)
        drives_layout.setSpacing(6)
        drives_layout.addWidget(heading_label(tr("Drives")))
        drives_layout.addWidget(self._list_panel, stretch=1)

        self._refresh_button = QPushButton()
        self._refresh_button.setIconSize(REFRESH_ICON)
        self._refresh_button.clicked.connect(self._on_refresh_clicked)
        buttons = QHBoxLayout()
        buttons.setSpacing(12)
        buttons.addWidget(self._refresh_button)
        buttons.addStretch()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(Card(tr("Recordings"), [self._choose]))
        layout.addWidget(drives, stretch=1)
        layout.addLayout(buttons)

        volumes.changed.connect(self._refresh)
        volumes.failed.connect(self._on_failed)
        # A bound method, not a lambda: Qt then drops the connection with the view.
        controller.recording_directory_changed.connect(self._on_directory_changed)
        self.apply_theme(self._theme)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        self._refresh_button.setIcon(symbol_icon("refresh", theme.color("text"), REFRESH_ICON))
        if self._browser is not None:
            self._browser.apply_theme(theme)
        self._shown_entries = None  # the rows draw in the colours of the scheme
        self._refresh()

    def browser(self) -> FolderBrowser:
        """The folder browser over the whole window, made on first use."""
        if self._browser is None:
            self._browser = FolderBrowser(self.window())
            self._browser.apply_theme(self._theme)
        return self._browser

    def sheet(self) -> ActionSheet:
        if self._sheet is None:
            self._sheet = ActionSheet(self.window())
        return self._sheet

    def entries(self) -> list[VolumeEntry]:
        return list(self._shown_entries or [])

    def rows(self) -> list[_VolumeRow]:
        return list(self._rows)

    # --- Showing ----------------------------------------------------------------------

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._volumes.start()

    def hideEvent(self, event: QHideEvent) -> None:
        super().hideEvent(event)
        self._volumes.stop()

    def _refresh(self) -> None:
        volumes = self._volumes
        busy = bool(volumes.activity)
        self._list_panel.setEnabled(not busy)
        self._refresh_button.setEnabled(not busy)
        self._show_directory()

        recordings = volumes.volume_of(self._controller.recording_directory)
        entries = volume_entries(volumes.volume_list, recordings)
        if entries != self._shown_entries:
            self._fill_list(entries)
        self._empty.setText(volumes.activity or tr(_NO_VOLUMES))
        self._area.setVisible(bool(entries) and not busy)
        self._empty.setVisible(not entries or busy)
        self._show_message()

    def _on_directory_changed(self, _directory: object) -> None:
        self._refresh()

    def _show_directory(self) -> None:
        directory = self._controller.recording_directory
        volume = self._volumes.volume_of(directory)
        self._directory.setText(directory_text(directory, volume))
        available = self._controller.recording_directory_available
        set_role(self._directory, "" if available else "muted")
        repolish(self._directory)
        if directory is None:
            space = ""
        elif available:
            space = free_text(free_space(directory))
        else:
            space = tr("Drive not mounted")
        self._space.setText(space)
        self._space.setVisible(bool(space))

    def _fill_list(self, entries: list[VolumeEntry]) -> None:
        for row in self._rows:
            self._list_layout.removeWidget(row)
            row.hide()  # out of sight at once: deleted only later, it would still be drawn
            row.deleteLater()
        self._rows = []
        for entry in entries:
            row = _VolumeRow(entry, self._theme)
            row.tapped.connect(lambda r=row: self._on_row_tapped(r.entry))
            self._list_layout.insertWidget(self._list_layout.count() - 1, row)
            self._rows.append(row)
        self._shown_entries = entries

    def _show_message(self) -> None:
        volumes = self._volumes
        if self._error:
            text, role = self._error, "error"
        elif not volumes.available:
            text, role = tr(_SYSTEM_MOUNTS), "small"
        elif not volumes.service_present:
            text, role = tr(_NO_UDISKS), "small"
        else:
            text, role = tr(_HINT), "small"
        self.message.setText(text)
        if self.message.property("role") != role:
            set_role(self.message, role)
            repolish(self.message)

    def _on_failed(self, message: str) -> None:
        self._error = message
        self._show_message()

    def _clear_error(self) -> None:
        self._error = ""
        self._show_message()

    # --- Actions ----------------------------------------------------------------------

    def _on_refresh_clicked(self) -> None:
        self._clear_error()
        self._volumes.watch()

    def _choose_directory(self) -> None:
        """The browser on the drive the directory lies on, else on the first mounted."""
        self._clear_error()
        directory = self._controller.recording_directory
        volume = self._volumes.volume_of(directory)
        if volume is None:
            mounted = self._volumes.mounted_volumes
            if not mounted:
                self._on_failed(tr(_NO_MOUNTED_VOLUME))
                return
            volume, directory = mounted[0], None
        self._open_browser(volume, directory)

    def _open_browser(self, volume: Volume, start: Path | None) -> None:
        self.browser().open(
            volume.name, volume.mount_point, start, self._controller.set_recording_directory
        )

    def _on_row_tapped(self, entry: VolumeEntry) -> None:
        volumes = self._volumes
        volume = volumes.volume(entry.path)
        if volumes.activity or volume is None:
            return
        self._clear_error()
        if not volume.mounted:
            volumes.mount(volume)
            return
        directory = self._controller.recording_directory
        start = directory if volumes.volume_of(directory) is volume else None
        choose = (tr("Choose folder"), "accent", lambda: self._open_browser(volume, start))
        if not volumes.can_mount:
            self._open_browser(volume, start)  # the system mounted it; nothing to eject
            return
        self.sheet().open(
            volume.name,
            volume_detail(volume),
            [(tr("Eject"), "", lambda: volumes.eject(volume)), choose],
        )
