"""The station page: every known station as a tile, filtered by ensemble or to the
favourites, and the scan. A finger resting on a tile marks the station as favourite or
unmarks it."""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QPointF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QMouseEvent, QPixmap, QResizeEvent, QShowEvent
from PyQt6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QScroller,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from i18n import tr, tr_noop
from ui.receiver_controller import ReceiverController, ReceptionState
from ui.touch.icons import symbol_icon, symbol_pixmap
from ui.touch.pager import SWIPE_SLOP
from ui.touch.stations import FAVOURITES, Station, StationDirectory, placeholder_pixmap
from ui.touch.theme import DARK, EDGE_FADE_WIDTH, THEMES, TILE_HEIGHT, Theme
from ui.touch.widgets import LONG_PRESS_MS, ChipRow, Spinner, repolish, set_role

COLUMNS = 4
TILE_ICON = 72
BOTTOM_ROW_HEIGHT = 96
# The spinner shown during the scan: one and a half tiles high.
SPINNER_SIZE = round(1.5 * TILE_HEIGHT)
# The star of a favourite in the lower right corner of its tile, inside the border.
STAR_SIZE = 24
STAR_MARGIN = 14
CHIP_ICON = QSize(24, 24)

_HINT = tr_noop(
    "A tap plays the station, a long press adds it to the favourites or takes it out. "
    "The scan reads the station lists of all channels."
)
_EMPTY = tr_noop(
    "No stations known yet.\n\n"
    "The scan searches all channels in Band III and reads their station lists; "
    "that takes about a minute."
)
_NO_FAVOURITES = tr_noop(
    "No favourites yet.\n\n"
    "Press and hold a station to add it here, "
    "or tap the star on the “Radio” page."
)


def _kinetic(area: QScrollArea) -> None:
    """Scrolls by swiping, as a finger expects; taps still reach the buttons."""
    QScroller.grabGesture(area.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture)


class _StationTile(QToolButton):
    """The tile of a station. A tap plays it; a finger resting on it for `LONG_PRESS_MS`
    asks to mark it as favourite or unmark it (`long_pressed`), and lifting the finger
    then is no tap. A star in the lower right corner shows a favourite."""

    long_pressed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        set_role(self, "tile")
        self._pressed_at: QPointF | None = None
        self._held = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(LONG_PRESS_MS)
        self._timer.timeout.connect(self._on_held)
        self._star = QLabel(self)
        self._star.setFixedSize(STAR_SIZE, STAR_SIZE)
        self._star.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._star.hide()

    def show_star(self, pixmap: QPixmap | None) -> None:
        """The star of a favourite; None for none."""
        if pixmap is not None:
            self._star.setPixmap(pixmap)
        self._star.setVisible(pixmap is not None)

    def has_star(self) -> bool:
        return not self._star.isHidden()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed_at = event.position()
            self._held = False
            self._timer.start()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._held:
            return  # the button stays up, so that lifting the finger does not click
        if self._pressed_at is not None:
            moved = event.position() - self._pressed_at
            if abs(moved.x()) >= SWIPE_SLOP or abs(moved.y()) >= SWIPE_SLOP:
                self._timer.stop()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._timer.stop()
        self._pressed_at = None
        self._held = False
        # Up since `_on_held`, the button does not click.
        super().mouseReleaseEvent(event)

    def _on_held(self) -> None:
        self._held = True
        self.setDown(False)
        self.long_pressed.emit()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        corner = STAR_MARGIN + STAR_SIZE
        self._star.move(self.width() - corner, self.height() - corner)


class StationsPage(QWidget):
    def __init__(
        self,
        controller: ReceiverController,
        directory: StationDirectory,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._directory = directory
        self._theme = THEMES[DARK]
        self._star = self._star_pixmap()
        self._shown: list[Station] = []
        self._grid_filter: str | None = None  # the filter last shown, see `_refresh`
        self._tiles: dict[tuple[str, int], _StationTile] = {}
        self._shown_filters: list[tuple[str, str]] | None = None
        # The scan needs the device: reception is stopped for it and resumed after.
        self._scan_when_stopped = False
        self._resume_after_scan = False
        self._show_scan_summary = False
        self._previous_state = controller.state

        self._chip_row = QWidget()
        self._chip_layout = QHBoxLayout(self._chip_row)
        self._chip_layout.setContentsMargins(0, 0, 0, 0)
        self._chip_layout.setSpacing(12)
        self._chip_group = QButtonGroup(self)
        self._chip_group.setExclusive(True)
        # Swiped sideways where it is wider than the page, however many ensembles
        # there are (`ui.touch.widgets.ChipRow`, as above the settings).
        self._chips_area = ChipRow(self._chip_row)
        self._chip_fades = self._chips_area.fades

        self._grid_widget = QWidget()
        self._grid = QGridLayout(self._grid_widget)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(16)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._grid_area = QScrollArea()
        self._grid_area.setWidget(self._grid_widget)
        self._grid_area.setWidgetResizable(True)
        self._grid_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        _kinetic(self._grid_area)

        self._empty_label = set_role(QLabel(tr(_EMPTY)), "large")
        self._empty_label.setWordWrap(True)
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._status_label = set_role(QLabel(tr(_HINT)), "muted")
        self._status_label.setWordWrap(True)
        self._scan_button = QPushButton(tr("Scan"))
        self._scan_button.setMinimumWidth(260)
        self._scan_button.clicked.connect(self._on_scan_clicked)
        bottom = QHBoxLayout()
        bottom.setSpacing(16)
        bottom.addWidget(self._status_label, stretch=1)
        bottom.addWidget(self._scan_button)
        self._bottom_row = QWidget()
        self._bottom_row.setFixedHeight(BOTTOM_ROW_HEIGHT)
        self._bottom_row.setLayout(bottom)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 8)
        layout.setSpacing(16)
        layout.addWidget(self._chips_area)
        layout.addWidget(self._grid_area, stretch=1)
        layout.addWidget(self._empty_label, stretch=1)
        layout.addWidget(self._bottom_row)

        # Over the tiles while the scan runs, with the channel being searched.
        self._spinner = Spinner(SPINNER_SIZE, self)

        directory.changed.connect(self._refresh)
        directory.logos_changed.connect(self._refresh_icons)
        controller.state_changed.connect(self._on_state_changed)
        controller.status_changed.connect(self._on_status_changed)
        controller.scan_progress.connect(self._on_scan_progress)
        self._refresh()
        self._on_state_changed(controller.state)

    def apply_theme(self, theme: Theme) -> None:
        """The placeholders are drawn in the scheme's colours; the chips are measured
        with the style sheet now in force."""
        self._theme = theme
        self._star = self._star_pixmap()
        self._refresh_icons()
        self._show_stars()
        self._show_chip_icons()
        self._fit_chip_row()
        self._spinner.set_theme(theme)

    def _star_pixmap(self) -> QPixmap:
        return symbol_pixmap("star", self._theme.color("accent"), STAR_SIZE)

    def _refresh_icons(self) -> None:
        for station in self._shown:
            self._tiles[station.key].setIcon(self._icon(station))

    # --- Filters and tiles ------------------------------------------------------------

    def _refresh(self) -> None:
        self._refresh_chips()
        stations = self._directory.stations()
        if stations != self._shown:
            self._fill_grid(stations)
        if self._directory.filter != self._grid_filter:
            # Another filter starts at the top. Under the same one the tiles stay where
            # they were swiped to, e.g. when a received list brings a station more.
            self._grid_filter = self._directory.filter
            self._grid_area.verticalScrollBar().setValue(0)
        current = self._directory.current()
        for key, tile in self._tiles.items():
            is_current = current is not None and key == current.key
            if tile.property("current") != is_current:
                tile.setProperty("current", is_current)
                repolish(tile)
        self._show_stars()
        none_known = not self._directory.all_stations()
        no_favourites = self._directory.filter == FAVOURITES and not self._shown
        self._empty_label.setText(tr(_EMPTY) if none_known else tr(_NO_FAVOURITES))
        self._grid_area.setVisible(not (none_known or no_favourites))
        # During the first scan the spinner stands in for the note that none are known.
        scanning = self._controller.state is ReceptionState.SCANNING
        self._empty_label.setVisible((none_known or no_favourites) and not scanning)

    def _show_stars(self) -> None:
        favourites = self._controller.favourites
        for key, tile in self._tiles.items():
            tile.show_star(self._star if key in favourites else None)

    def _refresh_chips(self) -> None:
        filters = self._directory.filters()
        if filters != self._shown_filters:
            self._shown_filters = filters
            for button in self._chip_group.buttons():
                self._chip_group.removeButton(button)
                button.hide()  # stays a child of the row until deleted
                button.deleteLater()
            while self._chip_layout.count():
                self._chip_layout.takeAt(0)
            fixed = [(f, self._directory.filter_name(f)) for f in (FAVOURITES, None)]
            for channel, label in [*fixed, *filters]:
                chip = set_role(QPushButton(), "chip")
                chip.setProperty("label", label)  # the count follows, see below
                chip.setCheckable(True)
                chip.setIconSize(CHIP_ICON)
                # Full width each: the row scrolls sideways rather than squeezing them.
                chip.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                chip.setProperty("channel", channel)
                chip.clicked.connect(lambda _checked, c=channel: self._directory.set_filter(c))
                self._chip_group.addButton(chip)
                self._chip_layout.addWidget(chip)
                # Shown at once, not only with the next event: the row is measured
                # right after this, and hidden buttons count for nothing.
                chip.show()
            self._chip_layout.addStretch()
            self._show_chip_icons()
        # The number of stations behind every name, "Favourites (3)": it changes also
        # where the filters stay, e.g. with a favourite marked.
        counts = self._directory.counts()
        for chip in self._chip_group.buttons():
            text = f"{chip.property('label')} ({counts.get(chip.property('channel'), 0)})"
            if chip.text() != text:
                chip.setText(text)
        self._fit_chip_row()
        for chip in self._chip_group.buttons():
            if chip.property("channel") == self._directory.filter:
                chip.setChecked(True)
                # Clear of the fade at the edge.
                self._chips_area.ensureWidgetVisible(chip, EDGE_FADE_WIDTH + 8, 0)

    def _show_chip_icons(self) -> None:
        """The star on the chip of the favourites, in the colour of the text on it."""
        theme = self._theme
        icon = symbol_icon("star", theme.color("accent"), CHIP_ICON, theme.color("on_accent"))
        for chip in self._chip_group.buttons():
            if chip.property("channel") == FAVOURITES:
                chip.setIcon(icon)

    def _fit_chip_row(self) -> None:
        """Every chip at its full width, the row as wide as they are together."""
        self._chips_area.fit()

    def _fill_grid(self, stations: list[Station]) -> None:
        for tile in self._tiles.values():
            self._grid.removeWidget(tile)
            tile.deleteLater()
        self._tiles = {}
        self._shown = stations
        for index, station in enumerate(stations):
            tile = _StationTile()
            tile.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            tile.setIconSize(QSize(TILE_ICON, TILE_ICON))
            tile.setIcon(self._icon(station))
            tile.setText(f"{station.label}\n{station.channel}")
            tile.setProperty("current", False)
            tile.clicked.connect(lambda _checked, s=station: self._play(s))
            tile.long_pressed.connect(lambda s=station: self._directory.toggle_favourite(s))
            self._grid.addWidget(tile, index // COLUMNS, index % COLUMNS)
            self._tiles[station.key] = tile

    def _scroll_to_current(self) -> None:
        """Brings the tile of the station playing to the middle of the view, as far as
        the tiles reach. Nothing moves if the filter hides it."""
        current = self._directory.current()
        tile = None if current is None else self._tiles.get(current.key)
        if tile is None:
            return
        middle = tile.geometry().center().y() - self._grid_area.viewport().height() // 2
        self._grid_area.verticalScrollBar().setValue(middle)

    def _icon(self, station: Station) -> QIcon:
        logo = self._directory.logo(station.sid)
        if logo is None:
            logo = placeholder_pixmap(
                station.label, TILE_ICON, self._theme.color("pressed"), self._theme.color("text")
            )
        return QIcon(logo)

    def _play(self, station: Station) -> None:
        self._controller.play_service(station.channel, station.sid)

    # --- Scan -------------------------------------------------------------------------

    def _on_scan_clicked(self) -> None:
        controller = self._controller
        if controller.state is ReceptionState.SCANNING:
            controller.cancel_scan()
            self._scan_button.setEnabled(False)
        elif controller.is_receiving:
            self._scan_when_stopped = True
            self._resume_after_scan = True
            self._scan_button.setEnabled(False)
            controller.stop_reception()
        else:
            controller.start_scan()

    def _on_state_changed(self, state: ReceptionState) -> None:
        previous, self._previous_state = self._previous_state, state
        scanning = state is ReceptionState.SCANNING
        self._scan_button.setText(tr("Cancel scan") if scanning else tr("Scan"))
        self._scan_button.setEnabled(state is not ReceptionState.CONNECTING)
        for tile in self._tiles.values():
            tile.setEnabled(not scanning)
        if scanning and not self._spinner.running:
            self._spinner.set_text("", tr("Scan"))
            self._place_spinner()
            self._spinner.start()
        elif not scanning:
            self._spinner.stop()
        self._refresh()
        if state is not ReceptionState.IDLE:
            # A failed scan sends no result; what follows is no longer about the scan.
            self._show_scan_summary = False
            if not scanning:
                self._status_label.setText(tr(_HINT))
            return
        if self._scan_when_stopped:
            self._scan_when_stopped = False
            self._controller.start_scan()
        elif previous is ReceptionState.SCANNING:
            # The controller reports the result right after this.
            self._show_scan_summary = True
            if self._resume_after_scan:
                self._resume_after_scan = False
                QTimer.singleShot(0, self._resume)

    def _resume(self) -> None:
        """Plays the station of before the scan again, if the scan kept it."""
        controller = self._controller
        if controller.state is ReceptionState.IDLE and controller.selected_sid is not None:
            controller.start_reception()

    def _on_scan_progress(self, channel: str, index: int, total: int) -> None:
        self._spinner.set_text(
            f"{index + 1}/{total}", tr("Channel {channel}").format(channel=channel)
        )

    def _place_spinner(self) -> None:
        """In the middle of the space between the filters and the bottom row."""
        top = self._chips_area.geometry().bottom()
        bottom = self._bottom_row.geometry().top()
        centre = QPoint(self._chips_area.geometry().center().x(), (top + bottom) // 2)
        self._spinner.move(centre - self._spinner.rect().center())

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._place_spinner()

    def showEvent(self, event: QShowEvent) -> None:
        """Opened, the page shows the station playing, wherever it was swiped to before."""
        super().showEvent(event)
        # Only once the scroll area has fitted the grid: tiles filled in while the page
        # was hidden have their places only after the layout request posted by the show.
        QTimer.singleShot(0, self._scroll_to_current)

    def _on_status_changed(self, text: str) -> None:
        if self._controller.state is ReceptionState.SCANNING or self._show_scan_summary:
            self._show_scan_summary = False
            self._status_label.setText(text)
