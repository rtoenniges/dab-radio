"""The "Station" tab: the selected service's announcements, bits, frame layout and audio
level."""

from __future__ import annotations

from collections import deque
from typing import ClassVar

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from audio.player import LEVEL_INTERVAL_SECONDS
from dab.announcement import (
    ALARM_CLUSTER,
    ANNOUNCEMENT_TIMEOUT_SECONDS,
    AnnouncementStatus,
    announcement_types,
)
from dsp.mp2_frame import Mp2FrameResult
from dsp.superframe import SUPERFRAME_SECONDS, SuperframeResult
from i18n import tr, tr_noop
from ui.plots import RollingBitmapPlot, SmoothedRange, colour_legend
from ui.receiver_controller import AnnouncementVisit

# Number of most recent CIFs shown in the raw MSC bit plot of the selected service
# (24 ms per CIF, 200 rows ~ 4.8 s of history).
MSC_HISTORY_ROWS = 200

# Time span shown by the frame layout and the audio level. Both sit one above the
# other in this tab and get the same time axis scale.
HISTORY_SECONDS = 12.0

# Number of readings in the audio level display (300 points of 40 ms).
AUDIO_LEVEL_HISTORY = round(HISTORY_SECONDS / LEVEL_INTERVAL_SECONDS)

Colour = tuple[int, int, int]

_ANNOUNCEMENT_STYLE = "background-color: #f0a020; color: black; font-weight: bold; padding: 6px;"
_ALARM_STYLE = "background-color: #b00020; color: white; font-weight: bold; padding: 6px;"


def _names(flags: int) -> str:
    """The short names of the announcement types in `flags`, separated by commas."""
    return ", ".join(name.short for name in announcement_types(flags))


def _station(sid: int) -> str:
    """A station without a known name."""
    return tr("Station {sid:04X}").format(sid=sid)


def describe_announcement_support(status: AnnouncementStatus | None) -> str:
    """One line on which announcements may interrupt the selected service."""
    if status is None:
        return tr("Announcements: -")
    if status.support is None or not status.support.flags:
        text = tr("Announcements: none reported")
    else:
        clusters = ", ".join(str(cluster) for cluster in status.support.clusters)
        text = tr("Announcements: {names} (cluster {clusters})").format(
            names=_names(status.support.flags), clusters=clusters or "-"
        )
    oe_support = status.oe_support
    if oe_support is not None and oe_support.flags:
        targets = ", ".join(
            tr("cluster {cluster} from {eid:04X}").format(cluster=cluster, eid=eid)
            for cluster, eid in oe_support.targets
        )
        # Translators: appended to "Announcements: …".
        text += tr("; from other ensembles: {names} ({targets})").format(
            names=_names(oe_support.flags), targets=targets
        )
    if status.alarm_enabled:
        # Translators: appended to "Announcements: …".
        text += tr("; alarm announcements enabled in the ensemble")
    return text


def describe_active_announcement(status: AnnouncementStatus | None, source: str | None) -> str:
    """The banner text for a running announcement, in this ensemble or another, empty if
    none applies. `source` is the name of the service whose audio carries it, if known."""
    if status is None:
        return ""
    if status.oe_active is not None:
        announcement = status.oe_active
        alarm = announcement.cluster_id == ALARM_CLUSTER
        text = (
            tr("Alarm announcement running in ensemble {eid:04X}: {names}")
            if alarm
            else tr("Announcement running in ensemble {eid:04X}: {names}")
        ).format(eid=status.oe_eid, names=_names(announcement.flags))
        if announcement.new:
            text = tr("{announcement} (just started)").format(announcement=text)
        if alarm:
            return text
        return f"{text} - {source or _station(announcement.target_sid)}"
    if status.active is None:
        return ""
    announcement = status.active
    text = (
        tr("Alarm announcement running: {names}")
        if announcement.cluster_id == ALARM_CLUSTER
        else tr("Announcement running: {names}")
    ).format(names=_names(announcement.flags))
    if announcement.new:
        text = tr("{announcement} (just started)").format(announcement=text)
    where = tr("Subchannel {id}").format(id=announcement.subchannel_id)
    if source:
        where += f", {source}"
    return f"{text} - {where}"


def describe_visit(visit: AnnouncementVisit) -> str:
    """The banner text while reception has gone to another ensemble for an announcement."""
    announcement = visit.announcement
    text = (
        tr(
            "Switched to the alarm announcement on channel {channel}: {names} - {there}. "
            "Back to {origin} on {origin_channel} afterwards."
        )
        if announcement.cluster_id == ALARM_CLUSTER
        else tr(
            "Switched to the announcement on channel {channel}: {names} - {there}. "
            "Back to {origin} on {origin_channel} afterwards."
        )
    )
    return text.format(
        channel=visit.channel,
        names=_names(announcement.flags),
        there=visit.label or _station(visit.sid),
        origin=visit.origin_label or _station(visit.origin_sid),
        origin_channel=visit.origin_channel,
    )


class FrameLayoutPlot:
    """Shows how the frames of the audio stream split into bytes, over time.

    Drawn as bands over time, one row per frame: the height of a band is the size of
    that part of the frame in bytes. Parts that fail their check are marked with a red
    dot. The two subclasses below provide the split for DAB+ superframes and for the
    MP2 frames of classic DAB.
    """

    _BOUNDARY_PEN = (215, 240, 245)

    def __init__(
        self,
        title: str | None,
        tooltip: str | None,
        legend: list[tuple[Colour, str]],
        broken_label: str,
        seconds_per_row: float,
        legend_offset: tuple[int, int] = (-10, 10),
        legend_columns: int = 2,
    ) -> None:
        """Without `title` and `tooltip` the plot is left to be labelled by what
        surrounds it (a panel of the touch surface)."""
        self.plot = pg.PlotWidget(title=title)
        self.plot.setLabel("bottom", tr("Time (0 = now)"), units="s")
        self.plot.setLabel("left", tr("Position in the frame"), units="B")
        # Fixed time axis over the whole history: after a service change the diagram
        # fills from the right instead of rescaling with every frame.
        self.plot.setXRange(-HISTORY_SECONDS, 0, padding=0.02)
        if tooltip:
            self.plot.setToolTip(tooltip)
        # By default top right in two columns, so that the legend stays flat and does
        # not cover the newest frames at the right edge.
        self._legend = self.plot.addLegend(
            offset=legend_offset, labelTextColor="w", brush=(20, 20, 25, 220)
        )
        self._legend.setColumnCount(legend_columns)
        for colour, label in legend:
            self._legend.addItem(self.plot.plot([], [], pen=pg.mkPen(colour, width=8)), label)

        self._bands: list[tuple[pg.PlotCurveItem, pg.PlotCurveItem, pg.FillBetweenItem]] = []
        self._broken_marker = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush(235, 60, 60), symbol="o", size=8
        )
        self.plot.addItem(self._broken_marker)
        self._legend.addItem(self._broken_marker, broken_label)

        # Per frame: the byte boundaries of the bands and the indices of broken bands.
        self._history: deque[tuple[list[int], list[int]]] = deque()
        self._seconds_per_row = 0.0
        self._set_seconds_per_row(seconds_per_row)
        self._colours: list[Colour] = []  # colours of the bands currently in the plot
        self._row_colours: list[Colour] = []  # colours belonging to `_history`
        self._dirty = False

    def clear(self) -> None:
        self._history.clear()
        self._row_colours = []
        self._dirty = False
        self._rebuild_bands([])
        self._broken_marker.setData([], [])

    def set_legend_colours(self, text: QColor | str, background: QColor | str) -> None:
        """Colours the legend for a ground other than the desktop's black one."""
        colour_legend(self._legend, text, background)

    def _set_seconds_per_row(self, seconds_per_row: float) -> None:
        self._seconds_per_row = seconds_per_row
        self._history = deque(maxlen=max(1, round(HISTORY_SECONDS / seconds_per_row)))

    def _add_row(
        self,
        boundaries: list[int],
        colours: list[Colour],
        broken_bands: list[int],
        seconds_per_row: float,
    ) -> None:
        """Records one frame: `boundaries` has one entry more than `colours` (0 first,
        the frame length last); `broken_bands` are indices into `colours`.

        A different split (another number of audio units, another frame duration)
        starts the history afresh: rows of different layouts cannot be drawn as
        continuous bands.
        """
        if seconds_per_row != self._seconds_per_row:
            self._set_seconds_per_row(seconds_per_row)
            self._row_colours = []
        if colours != self._row_colours:
            self._history.clear()
            self._row_colours = colours
        self._history.append((boundaries, broken_bands))
        self._dirty = True

    def _rebuild_bands(self, colours: list[Colour]) -> None:
        """Creates one filled band per colour.

        The boundary curves must be `PlotCurveItem`s: with the `PlotDataItem`s that
        `PlotWidget.plot()` returns, `FillBetweenItem` in pyqtgraph 0.14 draws no area.
        """
        for lower, upper, fill in self._bands:
            for item in (fill, lower, upper):
                self.plot.removeItem(item)
        self._bands = []
        self._colours = colours

        for colour in colours:
            lower = pg.PlotCurveItem(pen=None)
            # Only the upper boundary gets a line; it is also the lower boundary of
            # the next band.
            upper = pg.PlotCurveItem(pen=pg.mkPen(self._BOUNDARY_PEN, width=1))
            fill = pg.FillBetweenItem(lower, upper, brush=pg.mkBrush(*colour))
            self.plot.addItem(fill)
            self.plot.addItem(lower)
            self.plot.addItem(upper)
            self._bands.append((lower, upper, fill))

    def render(self) -> None:
        if not self._dirty or not self._history:
            return
        self._dirty = False

        if self._row_colours != self._colours:
            self._rebuild_bands(self._row_colours)

        count = len(self._history)
        times = (np.arange(count) - (count - 1)) * self._seconds_per_row
        boundaries = np.array([entry[0] for entry in self._history], dtype=np.float64)

        for index, (lower, upper, _fill) in enumerate(self._bands):
            lower.setData(times, boundaries[:, index])
            upper.setData(times, boundaries[:, index + 1])

        # Red dots in the middle of every band that failed its check.
        broken_x: list[float] = []
        broken_y: list[float] = []
        for row, (limits, broken) in enumerate(self._history):
            for band in broken:
                broken_x.append(times[row])
                broken_y.append((limits[band] + limits[band + 1]) / 2)
        self._broken_marker.setData(broken_x, broken_y)


class SuperframeLayoutPlot(FrameLayoutPlot):
    """How a DAB+ superframe splits into bytes.

    Answers what is behind a figure like "3/3 audio units". A superframe always lasts
    120 ms and consists of:

    * a short header (firecode and the table of AU start addresses),
    * several audio units, each one AAC access unit,
    * the Reed-Solomon parity bytes at the end.

    Since AAC encodes with a variable bit rate, the boundaries between the audio units
    visibly breathe with the music, while header and parity stay constant.
    """

    _HEADER_COLOUR = (235, 170, 50)
    _PARITY_COLOUR = (105, 105, 120)
    # Two alternating shades so that neighbouring audio units can be told apart
    # without counting.
    _AUDIO_UNIT_COLOURS: ClassVar[list[Colour]] = [(70, 175, 185), (45, 120, 150)]
    _TITLE = tr_noop("Layout of the superframes (DAB+, 120 ms of audio per superframe)")
    _TOOLTIP = tr_noop(
        "The bands show how a superframe splits into bytes. Between two boundary lines "
        "lies one audio unit each: an AAC access unit. Its size varies with the music, "
        "because AAC encodes with a variable bit rate - which is just what the wandering "
        "boundaries show. Header and Reed-Solomon parity bytes, by contrast, keep their "
        "length."
    )

    def __init__(self, labelled: bool = True) -> None:
        """`labelled=False` leaves out title and tooltip (see `FrameLayoutPlot`)."""
        super().__init__(
            title=tr(self._TITLE) if labelled else None,
            tooltip=tr(self._TOOLTIP) if labelled else None,
            legend=[
                # Translators: a part of the frame in the legend.
                (self._HEADER_COLOUR, tr("Header")),
                (self._AUDIO_UNIT_COLOURS[0], tr("Audio unit")),
                (self._PARITY_COLOUR, tr("Reed-Solomon parity bytes")),
            ],
            broken_label=tr("Audio unit with CRC error"),
            seconds_per_row=SUPERFRAME_SECONDS,
        )

    def update(self, result: SuperframeResult) -> None:
        if not result.sync_ok or not result.total_length or not result.au_segments:
            return
        boundaries = [0, result.au_segments[0][0]]
        boundaries += [end for _, end, _ in result.au_segments]
        boundaries.append(result.total_length)

        num_units = len(result.au_segments)
        colours = [self._HEADER_COLOUR]
        colours += [
            self._AUDIO_UNIT_COLOURS[i % len(self._AUDIO_UNIT_COLOURS)] for i in range(num_units)
        ]
        colours.append(self._PARITY_COLOUR)
        # Band 0 is the header, so audio unit i is band i + 1.
        broken = [unit + 1 for unit, (_, _, ok) in enumerate(result.au_segments) if not ok]
        self._add_row(boundaries, colours, broken, SUPERFRAME_SECONDS)


class Mp2FrameLayoutPlot(FrameLayoutPlot):
    """How an MP2 frame of classic DAB splits into bytes.

    Unlike a superframe, an MP2 frame has a fixed length and is not protected by a
    Reed-Solomon code. What varies is the X-PAD at the end: while the slideshow
    transfers an image, it takes up more of the frame, at the expense of the audio
    data.
    """

    _HEADER_COLOUR = (235, 170, 50)
    _AUDIO_COLOUR = (70, 175, 185)
    _X_PAD_COLOUR = (170, 110, 205)
    _SCF_CRC_COLOUR = (105, 105, 120)
    _F_PAD_COLOUR = (220, 175, 240)
    _COLOURS: ClassVar[list[Colour]] = [
        _HEADER_COLOUR,
        _AUDIO_COLOUR,
        _X_PAD_COLOUR,
        _SCF_CRC_COLOUR,
        _F_PAD_COLOUR,
    ]
    _AUDIO_BAND = 1
    _TOOLTIP = tr_noop(
        "The bands show how an MP2 frame splits into bytes: at the front header and "
        "header CRC, then the audio data, at the end the programme-associated data "
        "(X-PAD with radio text and slideshow images, stored backwards), the checksum of "
        "the scale factors (ScF-CRC) and the F-PAD. The frame size is fixed; what the "
        "X-PAD occupies is missing from the audio data. While an image is being "
        "transferred, the X-PAD band can therefore be seen to grow.\n\n"
        "Unlike DAB+, DAB has no Reed-Solomon protection: errors the Viterbi decoder "
        "does not correct are only detected by the header CRC. Such frames are muted."
    )

    def __init__(self, labelled: bool = True) -> None:
        """`labelled=False` leaves out title and tooltip (see `FrameLayoutPlot`)."""
        super().__init__(
            title=tr("Layout of the MP2 frames (DAB)") if labelled else None,
            tooltip=tr(self._TOOLTIP) if labelled else None,
            legend=[
                (self._HEADER_COLOUR, "Header + CRC"),
                (self._AUDIO_COLOUR, tr("Audio data")),
                (self._X_PAD_COLOUR, "X-PAD"),
                (self._SCF_CRC_COLOUR, "ScF-CRC"),
                (self._F_PAD_COLOUR, "F-PAD"),
            ],
            broken_label=tr("Frame with CRC error"),
            seconds_per_row=0.024,
            # Bottom left: at the top lie the X-PAD and the other small bands of the
            # newest frames, the most telling part of this diagram; bottom left only
            # the audio data of the oldest frames.
            legend_offset=(10, -10),
            legend_columns=3,
        )
        self._labelled = labelled
        self._title_duration: float | None = None  # frame duration the title states

    def update(self, result: Mp2FrameResult) -> None:
        if not result.sync_ok:
            return
        duration = result.audio_format.duration_seconds
        if self._labelled and duration != self._title_duration:
            self._title_duration = duration
            self.plot.setTitle(
                tr(
                    "Layout of the MP2 frames (DAB, {milliseconds:.0f} ms of audio per frame)"
                ).format(milliseconds=duration * 1000)
            )
        broken = [] if result.crc_ok else [self._AUDIO_BAND]
        self._add_row([0, *result.segments()], self._COLOURS, broken, duration)


class ServiceTab(QWidget):
    """The selected service from raw subchannel bits to the decoded audio's level, and
    above that its announcements.

    All plots only record in `update_*` and draw in `render()`, which the main window
    calls at the shared refresh rate and only while this tab is visible.
    """

    # Whether the audio follows a running announcement into its subchannel.
    announcement_switching_toggled = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._announcement_label = QLabel(describe_announcement_support(None))
        self._announcement_label.setToolTip(
            tr(
                "Announcement types the station lets itself be interrupted for, such as "
                "traffic news (FIG 0/18, ETSI EN 300 401, clause 8.1.6.1). The names are the "
                "short forms from ETSI TS 101 756. Stations with the same announcements form "
                "a cluster; a running announcement is reported for a cluster. Alarm "
                "announcements, by contrast, interrupt all stations of the ensemble, provided "
                "the ensemble enables them in the alarm flag of FIG 0/0."
            )
        )
        self._announcement_switch = QCheckBox(tr("Switch to announcements"))
        self._announcement_switch.setToolTip(
            tr(
                "If an announcement for the selected station runs in another subchannel, the "
                "audio switches there for its duration and back afterwards. Dynamic Label and "
                "images stay as they are meanwhile. If it runs in another ensemble (FIG 0/25, "
                "0/26), the receiver retunes to that ensemble's channel for its duration and "
                "returns to the selected station afterwards. The note on the announcement "
                "appears without switching too."
            )
        )
        self._announcement_switch.toggled.connect(self.announcement_switching_toggled)
        self._announcement_banner = QLabel()
        self._announcement_banner.setWordWrap(True)
        self._announcement_banner.setToolTip(
            tr(
                "A running announcement for the selected station (FIG 0/19, clause 8.1.6.2): "
                "its type and the subchannel carrying its audio. In the first seconds the "
                "station marks it as new (“just started”). It ends when the station reports "
                "the end, but at the latest when its report has been missing for {timeout:.0f} "
                "s - otherwise it would stay forever after a reception fault."
            ).format(timeout=ANNOUNCEMENT_TIMEOUT_SECONDS)
        )
        self._announcement_banner.hide()
        # What the banner is made from: the announcement status with the name of the
        # service carrying it, and reception gone to another ensemble for one.
        self._announcement: tuple[AnnouncementStatus | None, str | None] = (None, None)
        self._visit: AnnouncementVisit | None = None

        self._msc_raw_plot = RollingBitmapPlot(
            tr("1. Raw MSC bits (interleaved, coded)"),
            tr("Bit position in the subchannel"),
            tr("CIF (time)"),
            MSC_HISTORY_ROWS,
        )
        self._msc_decoded_plot = RollingBitmapPlot(
            tr("2. After deinterleaving, Viterbi, descrambling"),
            tr("Bit position (info bit)"),
            tr("Logical frame (time)"),
            MSC_HISTORY_ROWS,
        )
        self._msc_raw_plot.plot.setToolTip(
            tr(
                "The bits allotted to the selected station, as they come out of the CIF: "
                "time-interleaved over 16 CIFs (384 ms) and convolutionally coded. No "
                "recognisable pattern is to be expected here."
            )
        )
        self._msc_decoded_plot.plot.setToolTip(
            tr(
                "The same bits after time deinterleaving, Viterbi decoding and descrambling "
                "(energy dispersal). With DAB+ the picture stays noise-like - AAC is entropy "
                "coded, its bits are practically uniformly distributed even descrambled. With "
                "DAB (MP2), by contrast, the frame header repeats at a fixed distance: at 48 "
                "kHz it sits at the same place in every logical frame and shows as a vertical "
                "stripe. The diagram below shows the whole layout."
            )
        )

        # Only one of the two is visible, depending on whether the service is DAB+ or DAB.
        self._superframe_layout_plot = SuperframeLayoutPlot()
        self._mp2_layout_plot = Mp2FrameLayoutPlot()
        self._mp2_layout_plot.plot.hide()

        self._audio_level_plot = pg.PlotWidget(title=tr("Level of the decoded audio"))
        self._audio_level_plot.setLabel("bottom", tr("Time (0 = now)"), units="s")
        self._audio_level_plot.setLabel("left", tr("Peak level"), units="dBFS")
        self._audio_level_plot.enableAutoRange(x=False, y=False)
        # The same fixed time axis as the superframe layout above it.
        self._audio_level_plot.setXRange(-HISTORY_SECONDS, 0, padding=0.02)
        self._audio_level_range = SmoothedRange()
        # Legend bottom left. It has no background, so it never covers a curve even
        # when the level reaches down there.
        self._audio_level_plot.addLegend(offset=(10, -10), labelTextColor="w")
        self._audio_level_plot.setToolTip(
            tr(
                "Peak level over {milliseconds:.0f} ms each for every channel of the decoded "
                "audio. Shows the dynamics and dropouts of the station."
            ).format(milliseconds=LEVEL_INTERVAL_SECONDS * 1000)
        )
        self._audio_level_curves = [
            # Translators: the audio channels in the legend.
            self._audio_level_plot.plot(pen="c", name=tr("left")),
            self._audio_level_plot.plot(pen="m", name=tr("right")),
        ]
        self._audio_levels: deque[np.ndarray] = deque(maxlen=AUDIO_LEVEL_HISTORY)
        self._audio_levels_dirty = False

        # The two bitmaps side by side: they show the same data stream before and
        # after channel decoding and belong together. That leaves enough height below
        # for the two more telling diagrams.
        msc_row = QHBoxLayout()
        msc_row.addWidget(self._msc_raw_plot.plot)
        msc_row.addWidget(self._msc_decoded_plot.plot)

        announcement_row = QHBoxLayout()
        announcement_row.addWidget(self._announcement_label, stretch=1)
        announcement_row.addWidget(self._announcement_switch)

        layout = QVBoxLayout()
        layout.addLayout(announcement_row)
        layout.addWidget(self._announcement_banner)
        layout.addLayout(msc_row, stretch=2)
        layout.addWidget(self._superframe_layout_plot.plot, stretch=3)
        layout.addWidget(self._mp2_layout_plot.plot, stretch=3)
        layout.addWidget(self._audio_level_plot, stretch=2)
        self.setLayout(layout)

    def clear(self) -> None:
        """Empties all histories (on start and on a service change)."""
        self._msc_raw_plot.clear()
        self._msc_decoded_plot.clear()
        self._superframe_layout_plot.clear()
        self._mp2_layout_plot.clear()
        self._audio_levels.clear()
        self._audio_levels_dirty = False
        for curve in self._audio_level_curves:
            curve.clear()

    def set_announcement_switching(self, enabled: bool) -> None:
        self._announcement_switch.setChecked(enabled)

    def announcement_switching(self) -> bool:
        return self._announcement_switch.isChecked()

    def show_announcements(self, status: AnnouncementStatus | None, source: str | None) -> None:
        """Which announcements the selected service accepts, and the one running for
        it; `source` names the service whose audio carries that one, if known. Not
        touched by `clear`: the signal processing reports afresh on every service
        selection and channel change."""
        self._announcement_label.setText(describe_announcement_support(status))
        self._announcement = (status, source)
        self._show_banner()

    @property
    def announcement(self) -> tuple[AnnouncementStatus | None, str | None]:
        """The announcement status last shown, with the name of the service carrying it."""
        return self._announcement

    def show_visit(self, visit: AnnouncementVisit | None) -> None:
        """Reception has gone to another ensemble for an announcement (None: no more);
        the banner says so instead of what runs for the service decoded there."""
        self._visit = visit
        self._show_banner()

    def _show_banner(self) -> None:
        status, source = self._announcement
        if self._visit is not None:
            text = describe_visit(self._visit)
            announcement = self._visit.announcement
        else:
            text = describe_active_announcement(status, source)
            announcement = None if status is None else (status.oe_active or status.active)
        self._announcement_banner.setText(text)
        self._announcement_banner.setVisible(bool(text))
        if text:
            alarm = announcement.cluster_id == ALARM_CLUSTER
            self._announcement_banner.setStyleSheet(_ALARM_STYLE if alarm else _ANNOUNCEMENT_STYLE)

    def update_msc_bits(self, cif_bits: np.ndarray) -> None:
        self._msc_raw_plot.update(cif_bits)

    def update_audio_bits(self, info_bits: np.ndarray) -> None:
        self._msc_decoded_plot.update(info_bits)

    def update_superframe(self, result: SuperframeResult) -> None:
        self._show_layout(self._superframe_layout_plot)
        self._superframe_layout_plot.update(result)

    def update_mp2_frame(self, result: Mp2FrameResult) -> None:
        self._show_layout(self._mp2_layout_plot)
        self._mp2_layout_plot.update(result)

    def _show_layout(self, layout_plot: FrameLayoutPlot) -> None:
        """Shows the layout diagram matching the audio coding of the service."""
        for candidate in (self._superframe_layout_plot, self._mp2_layout_plot):
            candidate.plot.setVisible(candidate is layout_plot)

    def update_audio_level(self, levels: np.ndarray) -> None:
        self._audio_levels.append(levels)
        self._audio_levels_dirty = True

    def render(self) -> None:
        self._msc_raw_plot.render()
        self._msc_decoded_plot.render()
        self._superframe_layout_plot.render()
        self._mp2_layout_plot.render()
        self._render_audio_levels()

    def _render_audio_levels(self) -> None:
        if not self._audio_levels_dirty or not self._audio_levels:
            return
        self._audio_levels_dirty = False
        history = np.array(self._audio_levels)
        # As in the layout diagram: 0 is now, the past extends to the left.
        times = (np.arange(len(history)) - (len(history) - 1)) * LEVEL_INTERVAL_SECONDS
        for channel, curve in enumerate(self._audio_level_curves):
            if channel < history.shape[1]:
                curve.setData(times, history[:, channel])
            else:
                curve.clear()
        self._audio_level_range.apply(
            self._audio_level_plot, float(history.min()), float(history.max())
        )
