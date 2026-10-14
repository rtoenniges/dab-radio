"""A folder on a USB drive, chosen with a finger: an overlay over the whole window shows
the folders of one directory, a tap on one goes down into it, "Up" goes back up,
"New folder" asks for a name on the keyboard (`ui.touch.keyboard`), and "Record here"
takes the directory shown. It is only for choosing where the recordings go, so it never
leaves the drive it was opened on."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QMouseEvent
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
from ui.touch.icons import symbol_icon, symbol_pixmap
from ui.touch.keyboard import Keyboard
from ui.touch.overlay import Overlay
from ui.touch.theme import DARK, THEMES, Theme
from ui.touch.widgets import ElidedLabel, TapWidget, repolish, set_role

PANEL_WIDTH = 1040
PANEL_HEIGHT = 620
ROW_HEIGHT = 64
FOLDER_ICON = 28
CHEVRON_ICON = 24
BUTTON_ICON = QSize(32, 32)
# Between the drive and the folders of the path shown.
SEPARATOR = " › "

# Folders FAT, Windows and ext4 keep for themselves; nothing to record into.
_SYSTEM_FOLDERS = {"system volume information", "$recycle.bin", "lost+found", "found.000"}
# What no file system among FAT, exFAT, NTFS and ext4 takes in a name.
_FORBIDDEN = set('<>:"/\\|?*')
# As many bytes as the keyboard types (`ui.touch.keyboard.MAX_BYTES`), well under the
# 255 of every file system.
MAX_NAME_BYTES = 64
_NAME_HINT = tr_noop('Without < > : " / \\ | ? * and not ending in a dot or space')


def list_folders(directory: Path) -> list[Path]:
    """The folders in `directory`, by name; hidden ones and those the system keeps for
    itself left out. Raises OSError where the directory cannot be read."""
    folders = [
        entry
        for entry in directory.iterdir()
        if entry.is_dir()
        and not entry.name.startswith(".")
        and entry.name.casefold() not in _SYSTEM_FOLDERS
    ]
    return sorted(folders, key=lambda folder: folder.name.casefold())


def valid_folder_name(name: str) -> bool:
    """A name FAT, exFAT, NTFS and ext4 all take: none of `<>:"/\\|?*`, no control
    characters, neither empty nor "." or "..", no space or dot at the end."""
    if not name or name in (".", "..") or len(name.encode()) > MAX_NAME_BYTES:
        return False
    if name != name.strip() or name.endswith("."):
        return False
    return not any(character in _FORBIDDEN or ord(character) < 32 for character in name)


def path_text(volume_name: str, root: Path, directory: Path) -> str:
    """ "USB-Stick › Aufnahmen › 2026" for the directory within the drive."""
    parts = [volume_name, *directory.relative_to(root).parts]
    return SEPARATOR.join(part for part in parts if part)


class _FolderRow(TapWidget):
    """A folder with its name and a chevron: a tap goes into it."""

    def __init__(self, folder: Path, theme: Theme) -> None:
        super().__init__()
        set_role(self, "row")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setMinimumHeight(ROW_HEIGHT)
        self.folder = folder
        icon = QLabel()
        icon.setFixedWidth(FOLDER_ICON)
        icon.setPixmap(
            symbol_pixmap("folder", theme.color("muted"), FOLDER_ICON, self.devicePixelRatioF())
        )
        name = ElidedLabel()
        name.setText(folder.name)
        chevron = QLabel()
        chevron.setFixedWidth(CHEVRON_ICON)
        chevron.setPixmap(
            symbol_pixmap("chevron", theme.color("muted"), CHEVRON_ICON, self.devicePixelRatioF())
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(12)
        layout.addWidget(icon)
        layout.addWidget(name, stretch=1)
        layout.addWidget(chevron)

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


class FolderBrowser(Overlay):
    """`open` shows the folders of a drive from `start` (or its root); `on_accept` gets
    the directory chosen with "Record here"."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.panel.setFixedSize(PANEL_WIDTH, PANEL_HEIGHT)
        self._theme = THEMES[DARK]
        self._volume_name = ""
        self._root = Path()
        self._directory = Path()
        self._on_accept: Callable[[Path], None] | None = None
        self._keyboard: Keyboard | None = None
        self._rows: list[_FolderRow] = []

        self._title = set_role(QLabel(tr("Folder for the recordings")), "muted")
        self._up = QPushButton()
        self._up.setIconSize(BUTTON_ICON)
        self._up.setToolTip(tr("Up one level"))
        self._up.clicked.connect(self.go_up)
        self._path = set_role(ElidedLabel(), "large")
        cancel = QPushButton(tr("Cancel"))
        cancel.clicked.connect(self.dismiss)
        header = QHBoxLayout()
        header.setSpacing(12)
        header.addWidget(self._up)
        header.addWidget(self._path, stretch=1)
        header.addWidget(cancel)

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
        self._empty = set_role(QLabel(tr("No subfolders.")), "muted")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._message = set_role(QLabel(), "error")
        self._message.setVisible(False)
        self._new_folder = QPushButton(tr("New folder"))
        self._new_folder.clicked.connect(self._ask_folder_name)
        self._accept = set_role(QPushButton(tr("Record here")), "accent")
        self._accept.clicked.connect(self.accept)
        bottom = QHBoxLayout()
        bottom.setSpacing(12)
        bottom.addWidget(self._new_folder)
        bottom.addWidget(self._message)
        bottom.addStretch()
        bottom.addWidget(self._accept)

        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(24, 16, 24, 20)
        layout.setSpacing(12)
        layout.addWidget(self._title)
        layout.addLayout(header)
        layout.addWidget(self._area, stretch=1)
        layout.addWidget(self._empty, stretch=1)
        layout.addLayout(bottom)
        self.apply_theme(self._theme)

    def apply_theme(self, theme: Theme) -> None:
        self._theme = theme
        self._up.setIcon(symbol_icon("arrow_up", theme.color("text"), BUTTON_ICON))
        if self._keyboard is not None:
            self._keyboard.apply_theme(theme)
        if self.isVisible():
            self._show_directory(self._directory)

    def keyboard(self) -> Keyboard:
        """The keyboard over the whole window (above this overlay), made on first use."""
        if self._keyboard is None:
            self._keyboard = Keyboard(self.parentWidget())
            self._keyboard.apply_theme(self._theme)
        return self._keyboard

    @property
    def directory(self) -> Path:
        """The directory shown."""
        return self._directory

    @property
    def root(self) -> Path:
        return self._root

    def rows(self) -> list[_FolderRow]:
        return list(self._rows)

    def open(
        self,
        volume_name: str,
        root: Path | str,
        start: Path | str | None,
        on_accept: Callable[[Path], None],
    ) -> None:
        """Shows the folders of the drive `root` (named `volume_name`), at `start` where
        that lies on it and exists, else at the root."""
        self._volume_name = volume_name
        self._root = Path(root)
        self._on_accept = on_accept
        directory = self._root
        if start is not None:
            candidate = Path(start)
            if candidate.is_relative_to(self._root) and candidate.is_dir():
                directory = candidate
        self._show_directory(directory)
        self.show_over()

    def dismiss(self) -> None:
        self._on_accept = None
        super().dismiss()

    def enter(self, folder: Path) -> None:
        self._show_directory(folder)

    def go_up(self) -> None:
        if self._directory != self._root:
            self._show_directory(self._directory.parent)

    def accept(self) -> None:
        directory, on_accept = self._directory, self._on_accept
        self.dismiss()
        if on_accept is not None:
            on_accept(directory)

    def create_folder(self, name: str) -> None:
        """Makes the folder `name` in the directory shown and goes into it."""
        if not valid_folder_name(name):
            return
        folder = self._directory / name
        try:
            folder.mkdir(exist_ok=True)
        except OSError as error:
            self._show_message(
                tr("Folder not created: {error}").format(error=error.strerror or error)
            )
            return
        self._show_directory(folder)

    def _ask_folder_name(self) -> None:
        self.keyboard().open(
            tr("Name of the new folder"),
            self.create_folder,
            accept_text=tr("Create"),
            check=valid_folder_name,
            hint=tr(_NAME_HINT),
        )

    def _show_directory(self, directory: Path) -> None:
        self._directory = directory
        self._path.setText(path_text(self._volume_name, self._root, directory))
        self._up.setEnabled(directory != self._root)
        try:
            folders = list_folders(directory)
            self._show_message("")
        except OSError as error:
            folders = []
            self._show_message(
                tr("Folder not readable: {error}").format(error=error.strerror or error)
            )
        for row in self._rows:
            self._list_layout.removeWidget(row)
            row.hide()  # out of sight at once: deleted only later, it would still be drawn
            row.deleteLater()
        self._rows = []
        for folder in folders:
            row = _FolderRow(folder, self._theme)
            row.tapped.connect(lambda r=row: self.enter(r.folder))
            self._list_layout.insertWidget(self._list_layout.count() - 1, row)
            self._rows.append(row)
        self._area.setVisible(bool(folders))
        self._empty.setVisible(not folders)
        self._area.verticalScrollBar().setValue(0)

    def _show_message(self, text: str) -> None:
        self._message.setText(text)
        self._message.setVisible(bool(text))
