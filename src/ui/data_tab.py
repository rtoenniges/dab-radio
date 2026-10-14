"""The "Data services" tab: the data services of the ensemble, any number of them decoded
in packet mode at the same time, and for one of them the decoding from the outer
Reed-Solomon code to the MOT carousel."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFontDatabase
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dab.data_group import parse_data_group
from dab.data_services import DataComponent, dscty_name, user_application_name
from dab.fig import FEC_SCHEME_RS
from dab.mot_directory import MotCarouselStatus
from dsp.packet import (
    ADT_COLUMNS,
    FEC_COLUMNS,
    FEC_ROWS,
    PADDING_ADDRESS,
    FecFrameResult,
    PacketStatistics,
)
from dsp.pipeline import DataComponentStatus
from i18n import tr, tr_noop, tr_plural

FecCounts = tuple[int, int, int]  # corrected bytes, CRC errors before and after the FEC

# FEC frames kept in the history plot. At 24 kbit/s one arrives every 0.8 s, so this is
# a good two minutes there.
FEC_HISTORY_FRAMES = 150

# Bytes of the most recent data group shown in hex.
HEX_PREVIEW_BYTES = 96

Colour = tuple[int, int, int]

# Colours of the FEC frame image: packets of decoded addresses, of other addresses,
# padding packets, the RS data table, corrected bytes, rows beyond repair.
_OWN_PACKET: Colour = (70, 175, 185)
_OTHER_PACKET: Colour = (45, 110, 140)
_PADDING_PACKET: Colour = (60, 60, 72)
_UNREADABLE: Colour = (25, 25, 30)
_RS_TABLE: Colour = (105, 105, 120)
_CORRECTED: Colour = (250, 215, 60)
_UNCORRECTABLE: Colour = (235, 60, 60)

_TABLE_HEADERS = [
    tr_noop("Service"),
    "SId",
    tr_noop("Application"),
    tr_noop("Transport"),
    tr_noop("Subchannel"),
    tr_noop("Protection"),
]
_ADDRESS_HEADERS = [
    tr_noop("Address"),
    tr_noop("Component"),
    tr_noop("Packets"),
    tr_noop("CRC errors"),
    tr_noop("Data groups"),
    # Translators: data group errors, a table header kept short.
    tr_noop("DG errors"),
    tr_noop("Payload"),
]


def _hex_preview(data: bytes) -> str:
    shown = data[:HEX_PREVIEW_BYTES]
    lines = [shown[i : i + 16].hex(" ") for i in range(0, len(shown), 16)]
    if len(data) > HEX_PREVIEW_BYTES:
        more = len(data) - HEX_PREVIEW_BYTES
        lines.append(
            tr_plural("… ({count} byte more)", "… ({count} bytes more)", more).format(count=more)
        )
    return "\n".join(lines)


def _quoted(name: str) -> str:
    # Translators: quotation marks around the name of a service.
    return tr("“{name}”").format(name=name)


def describe_applications(component: DataComponent) -> str:
    if not component.applications:
        return tr("unknown (FIG 0/13 missing)")
    return ", ".join(user_application_name(app) for app in component.app_types)


def why_not_decodable(component: DataComponent) -> str:
    """Why a component cannot be decoded (for one that `decodable` says cannot)."""
    if not component.is_packet_mode:
        return tr("Data services in stream mode are not decoded.")
    if component.component.ca:
        return tr("The component is encrypted (conditional access).")
    if component.packet is None:
        return tr("Subchannel and packet address (FIG 0/3) have not been received yet.")
    return tr("The subchannel configuration (FIG 0/1) has not been received yet.")


def describe_transport(component: DataComponent) -> str:
    if not component.is_packet_mode:
        dscty = component.dscty
        if dscty is None:
            return tr("Stream mode")
        return tr("Stream mode, {type}").format(type=dscty_name(dscty))
    packet = component.packet
    if packet is None:
        return tr("Packet mode, SCId {scid} (FIG 0/3 missing)").format(scid=component.scid)
    if packet.data_groups:
        text = tr("Packet mode, address {address}, {type}, data groups")
    else:
        text = tr("Packet mode, address {address}, {type}, without data groups")
    return text.format(address=packet.packet_address, type=dscty_name(packet.dscty))


def describe_subchannel(component: DataComponent) -> str:
    subchannel = component.subchannel
    if subchannel is None:
        if component.subchannel_id is None:
            return "-"
        return tr("{subchannel} (FIG 0/1 missing)").format(subchannel=component.subchannel_id)
    return f"{subchannel.subchannel_id}: {subchannel.bitrate} kbit/s"


def describe_protection(component: DataComponent) -> str:
    subchannel = component.subchannel
    inner = "-" if subchannel is None else subchannel.description()
    if not component.is_packet_mode:
        return inner
    if component.fec_scheme is None:
        outer = tr("packet FEC unknown (FIG 0/14 missing)")
    elif component.fec_scheme == FEC_SCHEME_RS:
        outer = "RS(204,188)"
    else:
        outer = tr("without packet FEC")
    return f"{inner} + {outer}"


class FecFramePlot:
    """The most recent FEC frame as the table of the standard (EN 300 401, figure 15):
    12 rows, 188 columns of packets and 16 columns of Reed-Solomon parity.

    Every packet fills whole columns (24 bytes = 2 columns), so the packets appear as
    vertical stripes, coloured by whether their address is decoded. Bytes the decoder corrected light up; a row
    with more than 8 errors turns red. Since the packets run down the columns but the
    code protects the rows, an error burst in the byte stream shows up as a short
    vertical run spread over all rows, each of which only has to correct a few bytes.
    """

    def __init__(self) -> None:
        self.plot = pg.PlotWidget(title=tr("FEC frame (12 rows × 204 columns)"))
        self.plot.setLabel("bottom", tr("Column"))
        self.plot.setLabel("left", tr("Row (RS codeword)"))
        self.plot.invertY(True)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.plot.setXRange(0, FEC_COLUMNS, padding=0.01)
        self.plot.setYRange(0, FEC_ROWS, padding=0.02)
        self._image = pg.ImageItem()
        self.plot.addItem(self._image)
        # The legend sits below the plot: inside, it would cover rows of the flat image.
        self.legend = QLabel(
            "  ".join(
                f'<span style="color: rgb{colour}">■</span>&nbsp;{label}'
                for colour, label in (
                    (_OWN_PACKET, tr("decoded address")),
                    (_OTHER_PACKET, tr("other address")),
                    (_PADDING_PACKET, tr("padding packet")),
                    (_RS_TABLE, tr("RS parity bytes")),
                    (_CORRECTED, tr("corrected")),
                    (_UNCORRECTABLE, tr("row uncorrectable")),
                )
            )
        )
        self.legend.setWordWrap(True)
        self.plot.setToolTip(
            tr(
                "The outer error correction of packet mode (ETSI EN 300 401, clause 5.3.5). "
                "The packets are written column by column into a table of 12 rows and 188 "
                "columns; each row is a Reed-Solomon codeword RS(204,188) with 16 parity "
                "bytes on the right, which corrects up to 8 wrong bytes. A packet takes up "
                "2, 4, 6 or 8 whole columns and appears as a vertical stripe.\n\n"
                "Because the bytes are transmitted column by column but protected row by "
                "row, an error burst in the data stream is spread over all twelve rows: each "
                "only has to correct a part of it. Corrected bytes light up yellow."
            )
        )
        self._pending: tuple[FecFrameResult, frozenset[int]] | None = None

    def clear(self) -> None:
        self._pending = None
        self._image.clear()

    def update(self, frame: FecFrameResult, own_addresses: frozenset[int]) -> None:
        self._pending = (frame, own_addresses)

    def render(self) -> None:
        if self._pending is None:
            return
        frame, own_addresses = self._pending
        self._pending = None
        rgb = np.empty((FEC_ROWS, FEC_COLUMNS, 3), dtype=np.uint8)
        for column, address in enumerate(frame.column_addresses):
            if address < 0:
                colour = _UNREADABLE
            elif address == PADDING_ADDRESS:
                colour = _PADDING_PACKET
            elif address in own_addresses:
                colour = _OWN_PACKET
            else:
                colour = _OTHER_PACKET
            rgb[:, column] = colour
        rgb[:, ADT_COLUMNS:] = _RS_TABLE
        for row, corrections in enumerate(frame.row_corrections):
            if corrections < 0:
                rgb[row] = (rgb[row] * 0.4 + np.array(_UNCORRECTABLE) * 0.6).astype(np.uint8)
        rgb[frame.corrected] = _CORRECTED
        # ImageItem takes (x, y); column c, row r covers [c, c+1) x [r, r+1).
        self._image.setImage(rgb.transpose(1, 0, 2), autoLevels=False)


class FecHistoryPlot:
    """Corrections and packet errors per FEC frame over the recent frames: how much work
    the outer code does, and how many packets it saves.

    Draws a history kept by the tab for each decoded subchannel (`show`), so that it is
    there at once when switching between them.
    """

    def __init__(self) -> None:
        self.plot = pg.PlotWidget(title=tr("Packet FEC per frame"))
        self.plot.setLabel("bottom", tr("FEC frame (0 = latest)"))
        self.plot.setLabel("left", tr("Count"))
        self.plot.setXRange(-FEC_HISTORY_FRAMES, 0, padding=0.02)
        # Counts start at 0; with error-free reception an automatic range would centre
        # the zero line and show negative counts.
        self.plot.enableAutoRange(y=False)
        self.plot.setYRange(0, 1)
        self.plot.addLegend(offset=(10, 10), labelTextColor="w")
        self._corrected = self.plot.plot(
            pen=pg.mkPen(_CORRECTED, width=2), name=tr("corrected bytes")
        )
        self._before = self.plot.plot(
            pen=pg.mkPen((235, 150, 50), width=1), name=tr("packets with CRC error before FEC")
        )
        self._after = self.plot.plot(
            pen=None,
            symbol="o",
            symbolSize=7,
            symbolBrush=_UNCORRECTABLE,
            symbolPen=None,
            name=tr("… after FEC"),
        )
        self.plot.setToolTip(
            tr(
                "For each FEC frame: how many bytes the Reed-Solomon decoder corrected and "
                "how many packets would have had a wrong CRC without it (orange). Red dots: "
                "packets whose CRC is wrong even after the correction; they are lost and only "
                "come again with the next cycle of the carousel."
            )
        )
        self._history: deque[FecCounts] = deque()
        self._dirty = False

    def clear(self) -> None:
        self.show(deque())

    def show(self, history: deque[FecCounts]) -> None:
        self._history = history
        self._dirty = True

    def refresh(self) -> None:
        """The history shown has grown."""
        self._dirty = True

    def render(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        if not self._history:
            for curve in (self._corrected, self._before, self._after):
                curve.clear()
            return
        values = np.array(self._history, dtype=np.float64)
        x = np.arange(len(values)) - (len(values) - 1)
        self._corrected.setData(x, values[:, 0])
        self._before.setData(x, values[:, 1])
        lost = values[:, 2] > 0
        self._after.setData(x[lost], values[lost, 2])
        self.plot.setYRange(0, max(1.0, float(values.max())) * 1.1, padding=0)


@dataclass
class _SubchannelRecord:
    """What the tab keeps of a decoded subchannel, so that switching between the decoded
    components shows their state at once."""

    history: deque[FecCounts] = field(default_factory=lambda: deque(maxlen=FEC_HISTORY_FRAMES))
    last_frame: FecFrameResult | None = None
    statistics: PacketStatistics | None = None


class DataTab(QWidget):
    """Top: the data components of the ensemble; the check box in front of one decodes
    it in addition to the audio, any number at the same time. Below: for the component
    clicked last, what the decoding of its packet mode subchannel yields.

    Plots only record in `update_*`/`add_*` and draw in `render()`, which the main window
    calls at the shared refresh rate while this tab is visible.
    """

    component_toggled = pyqtSignal(int, bool)  # SCId, True = decode it, False = stop

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._components: list[DataComponent] = []
        # The components being decoded, by SCId, as far as the FIC describes them.
        self._decoding: dict[int, DataComponent | None] = {}
        # The component whose decoding the lower half shows (SCId).
        self._shown_scid: int | None = None
        self._subchannels: dict[int, _SubchannelRecord] = {}  # by SubChId
        self._carousels: dict[int, MotCarouselStatus] = {}  # by SCId

        self._table = QTableWidget(0, len(_TABLE_HEADERS))
        self._table.setHorizontalHeaderLabels([tr(header) for header in _TABLE_HEADERS])
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        # Room for three services; ensembles rarely carry more data services.
        self._table.setMinimumHeight(110)
        self._table.setMaximumHeight(160)
        self._table.cellClicked.connect(self._on_cell_clicked)
        self._table.itemChanged.connect(self._on_item_changed)
        self._table.setToolTip(
            tr(
                "The data services of the ensemble from FIG 0/2 (services and components), "
                "FIG 0/3 (subchannel and packet address), FIG 0/13 (application) and FIG 0/14 "
                "(packet FEC). Every service in packet mode is decoded in addition to the "
                "audio as soon as the FIC describes it. Without a tick it stays off on this "
                "channel, even after a restart, until the tick is set again. For this, each "
                "subchannel is deinterleaved and Viterbi-decoded like that of the station; "
                "that costs computing time per subchannel, services in the same subchannel "
                "share it.\n\n"
                "A click on a service shows below what its decoding yields."
            )
        )

        self._stop_button = QPushButton(tr("Stop all"))
        self._stop_button.setToolTip(
            tr(
                "Stops the decoding of all data services. They stay off on this channel until "
                "their tick is set again."
            )
        )
        self._stop_button.clicked.connect(self._stop_all)
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)

        self._fec_plot = FecFramePlot()
        self._history_plot = FecHistoryPlot()

        self._address_table = QTableWidget(0, len(_ADDRESS_HEADERS))
        self._address_table.setHorizontalHeaderLabels([tr(header) for header in _ADDRESS_HEADERS])
        self._address_table.verticalHeader().setVisible(False)
        self._address_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._address_table.horizontalHeader().setStretchLastSection(True)
        self._address_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._address_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._address_table.setToolTip(
            tr(
                "All packet addresses in the subchannel (a subchannel can carry several data "
                "services). CRC errors count packets that are wrong even after the packet "
                "FEC. Data groups are the MSC data groups put together from the packets with "
                "a valid CRC; DG errors those with a wrong CRC. Payload: data rate of the "
                "error-free packets without header, CRC and padding bytes."
            )
        )

        self._carousel_label = QLabel()
        self._carousel_label.setWordWrap(True)
        self._carousel_bar = QProgressBar()
        self._carousel_bar.setRange(0, 1000)
        self._carousel_bar.setTextVisible(False)
        self._carousel_bar.setMaximumHeight(12)
        carousel_tooltip = tr(
            "MOT in directory mode (ETSI EN 301 234, clause 7.2): the directory names all "
            "files of the carousel, which are repeated all the time. Each file is split into "
            "segments that come in any order; a file is complete as soon as all its segments "
            "are there. Missing segments come in the next cycle. The bar shows the share of "
            "the bytes already received."
        )
        self._carousel_label.setToolTip(carousel_tooltip)
        self._carousel_bar.setToolTip(carousel_tooltip)

        self._preview_label = QLabel()
        self._preview_label.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self._preview_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._preview_label.setWordWrap(True)
        self._preview_label.setToolTip(
            tr(
                "The start of the most recently received data group of the service shown, as "
                "it comes from the packets: header of the data group, payload, CRC. What TPEG "
                "sends is shown by the “Traffic” tab, what GEPOS sends by the “GNSS” tab."
            )
        )

        controls = QHBoxLayout()
        controls.addWidget(self._stop_button, alignment=Qt.AlignmentFlag.AlignTop)
        controls.addWidget(self._status_label, stretch=1)

        fec_side = QVBoxLayout()
        fec_side.addWidget(self._fec_plot.plot, stretch=1)
        fec_side.addWidget(self._fec_plot.legend)
        plots = QHBoxLayout()
        plots.addLayout(fec_side, stretch=3)
        plots.addWidget(self._history_plot.plot, stretch=2)

        details = QVBoxLayout()
        details.addWidget(self._carousel_label)
        details.addWidget(self._carousel_bar)
        details.addWidget(self._preview_label, stretch=1)
        bottom = QHBoxLayout()
        bottom.addWidget(self._address_table, stretch=3)
        bottom.addLayout(details, stretch=2)

        layout = QVBoxLayout()
        layout.addWidget(self._table)
        layout.addLayout(controls)
        layout.addLayout(plots, stretch=3)
        layout.addLayout(bottom, stretch=2)
        self.setLayout(layout)
        self.reset()

    # --- Input from the main window ----------------------------------------------

    def reset(self) -> None:
        """Forgets everything decoded so far (new ensemble)."""
        self._shown_scid = None
        self._subchannels.clear()
        self._carousels.clear()
        self._show_details()

    def update_components(self, components: list[DataComponent]) -> None:
        self._components = list(components)
        blocked = self._table.blockSignals(True)
        self._table.setRowCount(len(self._components))
        for row, component in enumerate(self._components):
            sid = f"{component.sid:08X}" if component.sid > 0xFFFF else f"{component.sid:04X}"
            cells = (
                component.label or "-",
                sid,
                describe_applications(component),
                describe_transport(component),
                describe_subchannel(component),
                describe_protection(component),
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if not component.decodable:
                    item.setForeground(QBrush(QColor(140, 140, 140)))
                    item.setToolTip(self._why_not_decodable(component))
                elif column == 0:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(self._check_state(component))
                    item.setToolTip(tr("Tick: decode in addition to the audio"))
                self._table.setItem(row, column, item)
        self._table.blockSignals(blocked)
        self._mark_shown_row()
        self._update_status()

    def set_decoding(self, statuses: tuple[DataComponentStatus, ...]) -> None:
        """Which data components the signal processing decodes, as far as the FIC
        describes them."""
        previous = self._decoding
        self._decoding = {status.scid: status.component for status in statuses}
        started = [scid for scid in self._decoding if scid not in previous]
        if started:
            self._shown_scid = started[-1]
        elif self._shown_scid in previous and self._shown_scid not in self._decoding:
            # The component shown stopped: show another one that is still decoded.
            self._shown_scid = next(iter(self._decoding), self._shown_scid)
        # Forget what belongs to components no longer decoded.
        used = {c.subchannel_id for c in self._decoding.values() if c is not None}
        self._subchannels = {k: v for k, v in self._subchannels.items() if k in used}
        self._carousels = {k: v for k, v in self._carousels.items() if k in self._decoding}
        self._update_check_states()
        self._mark_shown_row()
        self._show_details()

    def add_fec_frame(self, subchannel_id: int, frame: FecFrameResult) -> None:
        record = self._subchannels.setdefault(subchannel_id, _SubchannelRecord())
        record.last_frame = frame
        record.history.append(
            (frame.corrected_bytes, frame.crc_errors_before, frame.crc_errors_after)
        )
        if subchannel_id == self._shown_subchannel_id():
            self._fec_plot.update(frame, self._decoded_addresses(subchannel_id))
            self._history_plot.refresh()

    def update_statistics(self, subchannel_id: int, statistics: PacketStatistics) -> None:
        self._subchannels.setdefault(subchannel_id, _SubchannelRecord()).statistics = statistics
        if subchannel_id == self._shown_subchannel_id():
            self._show_statistics(statistics)
            self._update_status()

    def update_carousel(self, scid: int, status: MotCarouselStatus) -> None:
        self._carousels[scid] = status
        if scid == self._shown_scid:
            self._show_carousel(status)

    def render(self) -> None:
        self._fec_plot.render()
        self._history_plot.render()

    # --- Internals ----------------------------------------------------------------

    def _on_cell_clicked(self, row: int, _column: int) -> None:
        if 0 <= row < len(self._components):
            component = self._components[row]
            if component.is_packet_mode and component.scid != self._shown_scid:
                self._shown_scid = component.scid
                self._show_details()
        self._mark_shown_row()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """The check box of a row: start or stop its decoding."""
        row = item.row()
        if item.column() != 0 or not 0 <= row < len(self._components):
            return
        component = self._components[row]
        if component.decodable:
            checked = item.checkState() == Qt.CheckState.Checked
            self.component_toggled.emit(component.scid, checked)

    def _stop_all(self) -> None:
        for scid in list(self._decoding):
            self.component_toggled.emit(scid, False)

    def _check_state(self, component: DataComponent) -> Qt.CheckState:
        if component.scid in self._decoding:
            return Qt.CheckState.Checked
        return Qt.CheckState.Unchecked

    def _update_check_states(self) -> None:
        """Shows which components are decoded, as the signal processing confirms."""
        blocked = self._table.blockSignals(True)
        for row, component in enumerate(self._components):
            item = self._table.item(row, 0)
            if item is not None and component.decodable:
                item.setCheckState(self._check_state(component))
        self._table.blockSignals(blocked)

    def _mark_shown_row(self) -> None:
        self._table.clearSelection()
        for row, component in enumerate(self._components):
            if component.is_packet_mode and component.scid == self._shown_scid:
                self._table.selectRow(row)

    def _component(self, scid: int | None) -> DataComponent | None:
        """A packet mode component, as the decoding describes it or else as listed."""
        if scid is None:
            return None
        component = self._decoding.get(scid)
        if component is not None:
            return component
        return next((c for c in self._components if c.is_packet_mode and c.scid == scid), None)

    def _shown_subchannel_id(self) -> int | None:
        """The subchannel of the component shown, while it is being decoded."""
        component = self._decoding.get(self._shown_scid)
        return None if component is None else component.subchannel_id

    def _decoded_addresses(self, subchannel_id: int) -> frozenset[int]:
        return frozenset(
            component.packet.packet_address
            for component in self._decoding.values()
            if component is not None
            and component.packet is not None
            and component.subchannel_id == subchannel_id
        )

    def _show_details(self) -> None:
        """Fills the lower half with what is known of the component shown."""
        subchannel_id = self._shown_subchannel_id()
        record = None if subchannel_id is None else self._subchannels.get(subchannel_id)
        if record is None or record.last_frame is None:
            self._fec_plot.clear()
        else:
            self._fec_plot.update(record.last_frame, self._decoded_addresses(subchannel_id))
        self._history_plot.show(deque() if record is None else record.history)
        self._show_statistics(None if record is None else record.statistics)
        self._show_carousel(
            None if subchannel_id is None else self._carousels.get(self._shown_scid)
        )
        self._update_status()

    def _show_statistics(self, statistics: PacketStatistics | None) -> None:
        self._preview_label.setText("")
        if statistics is None:
            self._address_table.setRowCount(0)
            return
        self._address_table.setRowCount(len(statistics.addresses))
        for row, (address, stats) in enumerate(statistics.addresses.items()):
            rate = stats.useful_bytes * 8 / statistics.seconds if statistics.seconds else 0.0
            cells = (
                str(address),
                self._component_at(address),
                str(stats.packets),
                str(stats.crc_errors),
                str(stats.data_groups),
                str(stats.data_group_errors),
                f"{rate / 1000:.1f} kbit/s",
            )
            for column, text in enumerate(cells):
                self._address_table.setItem(row, column, QTableWidgetItem(text))
        shown = self._component(self._shown_scid)
        own = None if shown is None else shown.packet
        if own is not None and own.packet_address in statistics.addresses:
            group = statistics.addresses[own.packet_address].last_data_group
            if group:
                parsed = parse_data_group(group)
                if parsed is None:
                    text = tr("Last data group ({size} bytes):\n{data}")
                else:
                    text = tr("Last data group (type {type}, {size} bytes):\n{data}")
                self._preview_label.setText(
                    text.format(
                        type=None if parsed is None else parsed.group_type,
                        size=len(group),
                        data=_hex_preview(group),
                    )
                )

    def _show_carousel(self, status: MotCarouselStatus | None) -> None:
        if status is None:
            self._carousel_label.setText("")
            self._carousel_bar.setVisible(False)
            return
        self._carousel_bar.setVisible(True)
        if status.directory_objects is None:
            self._carousel_label.setText(
                tr(
                    "MOT carousel: directory not received yet, {size} kB of segments collected."
                ).format(size=f"{status.received_bytes / 1000:.0f}")
            )
            self._carousel_bar.setValue(0)
            return
        share = status.received_bytes / status.directory_bytes if status.directory_bytes else 1.0
        text = tr(
            "MOT carousel: {complete} of {files} files complete, {received} of {total} kB "
            "({percent} %)."
        ).format(
            complete=status.complete_objects,
            files=status.directory_objects,
            received=f"{status.received_bytes / 1000:.0f}",
            total=f"{status.directory_bytes / 1000:.0f}",
            percent=f"{share * 100:.0f}",
        )
        if status.carousel_period is not None:
            text += " " + tr("Cycle time according to the directory: {minutes} min.").format(
                minutes=f"{status.carousel_period / 60:.0f}"
            )
        self._carousel_label.setText(text)
        self._carousel_bar.setValue(round(share * 1000))

    @staticmethod
    def _why_not_decodable(component: DataComponent) -> str:
        return why_not_decodable(component)

    def _component_at(self, address: int) -> str:
        """Name of the component carried at `address` in the subchannel shown."""
        subchannel_id = self._shown_subchannel_id()
        for component in self._components:
            packet = component.packet
            if (
                packet is not None
                and packet.subchannel_id == subchannel_id
                and packet.packet_address == address
            ):
                return f"{component.label} ({describe_applications(component)})"
        return tr("unknown")

    def _update_status(self) -> None:
        self._stop_button.setEnabled(bool(self._decoding))
        text = self._shown_status()
        others = [self._label(scid) for scid in self._decoding if scid != self._shown_scid]
        if others:
            text += "\n" + tr("Also decoding: {services}.").format(services=", ".join(others))
        self._status_label.setText(text)

    def _label(self, scid: int) -> str:
        component = self._component(scid)
        return _quoted(component.label) if component is not None else f"SCId {scid}"

    def _shown_status(self) -> str:
        """What the status line says about the component shown."""
        if self._shown_scid is None:
            return tr(
                "No data service is being decoded. The tick in front of a service in packet "
                "mode decodes it in addition to the audio, several at the same time too; "
                "without a tick it stays off on this channel."
            )
        component = self._component(self._shown_scid)
        if self._shown_scid not in self._decoding:
            if component is None:
                return tr("SCId {scid} is not being decoded.").format(scid=self._shown_scid)
            if not component.decodable:
                return f"{_quoted(component.label)}: {self._why_not_decodable(component)}"
            return tr(
                "“{name}” is not being decoded; the tick in front of it starts the decoding."
            ).format(name=component.label)
        if component is None or not component.decodable:
            return tr(
                "Waiting for the description of the component (SCId {scid}) in the FIC..."
            ).format(scid=self._shown_scid)
        subchannel = component.subchannel
        text = tr(
            "Decoding “{name}”: subchannel {subchannel}, {bitrate} kbit/s, {protection}, "
            "address {address}."
        ).format(
            name=component.label,
            subchannel=subchannel.subchannel_id,
            bitrate=subchannel.bitrate,
            protection=describe_protection(component),
            address=component.packet.packet_address,
        )
        record = self._subchannels.get(subchannel.subchannel_id)
        statistics = None if record is None else record.statistics
        if statistics is not None:
            if statistics.fec:
                if statistics.fec_locked:
                    fec = tr(
                        "Packet FEC in sync: {frames} frames, {corrected} bytes corrected, "
                        "{rows} rows uncorrectable, {packets} packets with CRC errors before "
                        "the FEC."
                    )
                else:
                    fec = tr(
                        "Packet FEC searching for frames: {frames} frames, {corrected} bytes "
                        "corrected, {rows} rows uncorrectable, {packets} packets with CRC "
                        "errors before the FEC."
                    )
                text += "\n" + fec.format(
                    frames=statistics.fec_frames,
                    corrected=statistics.fec_corrected_bytes,
                    rows=statistics.fec_uncorrectable_rows,
                    packets=statistics.crc_errors_before_fec,
                )
            else:
                text += "\n" + tr("Without packet FEC: packets straight from the logical frames.")
            text += " " + tr("Padding packets: {count}.").format(count=statistics.padding_packets)
        return text
