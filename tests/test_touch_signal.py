"""The signal page of the touch surface: five pages to swipe through, each drawing one
step of the signal chain from the signals of the signal processing, and only while it
is shown."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QWidget

from dab.constants import MODE_I, SAMPLE_RATE
from dab.fig import AudioComponent, Subchannel
from dsp.iq_correction import IqCorrectionReport
from dsp.mp2_frame import CHANNEL_MODE_JOINT_STEREO, Mp2Format, Mp2FrameResult
from dsp.pipeline import SYMBOL_WINDOW_EXCERPT_SYMBOLS, SubchannelStatus, SymbolWindows
from dsp.superframe import AudioFormat, SuperframeResult
from dsp.tii import TiiPath, TiiResult, TiiTransmitter
from ui.ofdm_tab import OfdmTab
from ui.touch.signal_page import (
    CORRECTION,
    LOCKED,
    OFDM,
    OVERVIEW,
    PAGES,
    SEARCHING,
    SFN,
    SYNC,
)
from ui.touch.signal_panels import describe_format
from ui.touch.theme import touch_font
from ui.touch.window import SIGNAL, TouchWindow

# One application for the module: plots built outside a window need one alive.
APP = QApplication.instance() or QApplication([])

HE_AAC_V2 = AudioFormat(
    dac_rate_48k=True, sbr=True, stereo_core=False, parametric_stereo=True, mpeg_surround_config=0
)
MP2_FORMAT = Mp2Format(sample_rate=48_000, bitrate=128, channel_mode=CHANNEL_MODE_JOINT_STEREO)


@pytest.fixture
def window(data_directory, make_controller, fake_workers):
    controller = make_controller(data_directory, start=False)
    window = TouchWindow(controller)
    controller.start()
    window.show()
    window.show_page(SIGNAL)
    yield window
    window.close()
    window.deleteLater()


def _superframe(corrections: int = 0, broken: bool = False) -> SuperframeResult:
    segments = [(11, 300, True), (300, 620, not broken), (620, 990, True)]
    return SuperframeResult(
        sync_ok=True,
        rs_corrections=corrections,
        rs_uncorrectable_blocks=int(broken),
        num_audio_units=3,
        audio_format=HE_AAC_V2,
        au_segments=segments,
        data_length=990,
        total_length=1080,
    )


def _symbol_windows() -> SymbolWindows:
    length = MODE_I.null_symbol_length + SYMBOL_WINDOW_EXCERPT_SYMBOLS * MODE_I.symbol_length
    rng = np.random.default_rng(0)
    spectra = rng.normal(size=(76, MODE_I.fft_length)) + 1j * rng.normal(
        size=(76, MODE_I.fft_length)
    )
    return SymbolWindows(
        excerpt_start=-MODE_I.null_symbol_length,
        magnitude=np.abs(rng.normal(size=length)),
        null_spectrum=rng.normal(size=MODE_I.fft_length) + 0j,
        spectra=spectra,
    )


def test_the_pages_follow_the_signal_chain_without_tooltips(window) -> None:
    page = window._signal_page
    titles = [page.page(key).title.text() for key in PAGES]

    assert titles == [
        "Overview",
        "Synchronisation",
        "OFDM",
        "Detail view",
        "Single frequency network",
        "Error correction",
    ]
    assert page.current_page() == OVERVIEW
    assert page.pager.count() == len(PAGES)
    # No tooltips: a finger has nothing to hover with.
    assert all(not widget.toolTip() for widget in page.findChildren(QWidget))


def test_only_the_page_shown_is_drawn(window) -> None:
    page = window._signal_page
    ofdm = page.page(OFDM)
    window._controller.dsp_worker.symbol_windows_ready.emit(_symbol_windows())

    window._render_plots()
    assert ofdm.plots.time_plot.getPlotItem().listDataItems()[0].getData()[0] is None

    # Turned to, the page is drawn at once.
    page.show_page(OFDM)
    x, _y = ofdm.plots.time_plot.getPlotItem().listDataItems()[0].getData()
    assert (
        len(x) == MODE_I.null_symbol_length + SYMBOL_WINDOW_EXCERPT_SYMBOLS * MODE_I.symbol_length
    )
    assert ofdm.status.text() == f"{MODE_I.num_carriers} carriers · symbols 0 to 76"


def test_a_long_figure_leaves_the_tiles_of_the_overview_where_they_are(window) -> None:
    overview = window._signal_page.page(OVERVIEW)
    APP.processEvents()
    widths = [panel.width() for panel in overview._panels]

    # In German, the overload the RSPduo reports made the figures wider than half the
    # page: the tiles on the left narrowed, until the overload went.
    window._controller._adc_overload = True
    window._controller.dsp_worker.clipping_ready.emit(0.1234)
    overview.figures.set("load", "DSP 100 %, waiting 100 %, CPU 400 % " * 4)
    APP.processEvents()

    assert [panel.width() for panel in overview._panels] == widths
    assert widths[0] == widths[1]
    assert overview.figures.text("clipping") == "12.34 %, ADC overloaded"


def test_the_overload_stands_out_while_the_receiver_is_overloaded(window) -> None:
    controller = window._controller
    page = window._signal_page
    values = [page.page(key).figures._values["clipping"] for key in (OVERVIEW, SYNC)]
    controller.start_reception()
    worker = window._controller._worker
    controller.dsp_worker.stream_restarted.emit(worker.tuning)
    normal = values[0].font().bold()

    controller.dsp_worker.clipping_ready.emit(0.05)
    assert [value.property("alert") for value in values] == [True, True]
    assert [value.font().bold() for value in values] == [True, True]

    controller.stop_reception()
    worker.stopped.emit()
    assert [value.property("alert") for value in values] == [False, False]
    assert values[0].font().bold() == normal


def test_the_synchronisation_page_marks_the_null_symbols_and_follows_the_offset(window) -> None:
    page = window._signal_page
    sync = page.page(SYNC)
    page.show_page(SYNC)
    dsp = window._controller.dsp_worker
    time_axis = np.arange(500) * 0.0005

    dsp.power_envelope_ready.emit(time_axis, np.full(500, -20.0), [0.03, 0.126])
    for offset in (300.0, 310.0, 312.0):
        dsp.frequency_offset_ready.emit(offset)
    dsp.iq_correction_ready.emit(IqCorrectionReport(0.0042, None, None))
    window._render_plots()

    assert sync.status.text() == LOCKED
    markers = [marker for marker in sync._null_markers if marker.isVisible()]
    assert [marker.value() for marker in markers] == [0.03, 0.126]
    assert list(sync.offset.curve.getData()[1]) == [300.0, 310.0, 312.0]
    assert sync.figures.text("offset") == "+312 Hz"
    assert sync.figures.text("dc") == "0.0042"
    assert sync.figures.text("phase") == "Correction off"

    dsp.power_envelope_ready.emit(time_axis, np.full(500, -20.0), [])
    window._render_plots()
    assert sync.status.text() == SEARCHING
    assert not any(marker.isVisible() for marker in sync._null_markers)


def test_the_transmitters_of_the_network_are_listed(window) -> None:
    page = window._signal_page
    sfn = page.page(SFN)
    page.show_page(SFN)
    dsp = window._controller.dsp_worker
    transmitters = [TiiTransmitter(7, 3, 0.0, 0.61), TiiTransmitter(12, 1, -9.8, 0.07)]

    dsp.tii_ready.emit(TiiResult(transmitters, -22.0, 128))
    dsp.sfn_share_ready.emit(0.18, 0.003)
    window._render_plots()

    rows = [
        tuple(sfn.table.item(row, column).text() for column in range(4))
        for row in range(sfn.table.rowCount())
    ]
    # No path carries their codes yet: no delay.
    assert rows == [("07-03", "0.0 dB", "61.0 %", "-"), ("12-01", "-9.8 dB", "7.0 %", "-")]
    assert sfn.tii_note.text() == "128 null symbols averaged, detection limit -22 dB."
    assert sfn.status.text() == "SFN share 18.0 %, outside the guard 0.3 %"

    # Another frequency: nothing of the old network stays.
    window._controller.live_reset.emit()
    assert sfn.table.rowCount() == 0
    assert sfn.status.text() == ""


def test_a_single_transmitter_fits_its_column(window) -> None:
    """The code turns bold after the table fitted its columns to the plain text; with
    one row nothing fitted them again, and "31-03" showed as "31...". Measured in the
    typeface of the touch surface, as `main` sets it: in the fallback of the tests the
    header is the widest."""
    fallback = QApplication.font()
    QApplication.setFont(touch_font())
    try:
        page = window._signal_page
        sfn = page.page(SFN)
        page.show_page(SFN)
        dsp = window._controller.dsp_worker

        result = TiiResult([TiiTransmitter(31, 3, 0.0, 1.0, -0.04e-6)], -24.0, 32)
        dsp.tii_ready.emit(result)
        window._render_plots()

        table = sfn.table
        assert table.columnWidth(0) >= table.sizeHintForColumn(0)
        assert table.item(0, 3).text() == "0.0 µs"  # not "-0.0 µs"
    finally:
        QApplication.setFont(fallback)


def test_the_paths_are_marked_in_the_colours_of_their_transmitters(window) -> None:
    page = window._signal_page
    sfn = page.page(SFN)
    page.show_page(SFN)
    dsp = window._controller.dsp_worker
    transmitters = [
        TiiTransmitter(19, 4, 0.0, 0.997, 0.2e-6),
        TiiTransmitter(24, 3, -24.6, 0.003, 94.7e-6),
    ]
    paths = [
        TiiPath(0.2e-6, 0.0, (19, 4)),
        TiiPath(19.7e-6, -15.0, (19, 4), echo=True),
        TiiPath(94.7e-6, -24.6, (24, 3)),
        TiiPath(125.9e-6, -30.7, None),
    ]

    dsp.tii_ready.emit(TiiResult(transmitters, -14.0, 32, paths))
    window._render_plots()

    rows = [
        tuple(sfn.table.item(row, column).text() for column in range(4))
        for row in range(sfn.table.rowCount())
    ]
    assert rows == [
        ("19-04", "0.0 dB", "99.7 %", "0.2 µs"),
        ("24-03", "-24.6 dB", "0.3 %", "94.7 µs"),
    ]
    shown = [(label, filled, colour.name()) for label, filled, colour, _tier in sfn.marks.shown()]
    assert [(label, filled) for label, filled, _colour in shown] == [
        ("19-04", True),
        (None, False),
        ("24-03", True),
        ("?", False),
    ]
    # The code in the list has the colour of its marks, each transmitter its own.
    first, second = (sfn.table.item(row, 0).foreground().color().name() for row in range(2))
    assert first == shown[0][2] == shown[1][2]
    assert second == shown[2][2] != first
    assert sfn.tii_note.text() == "32 null symbols averaged, detection limit -14 dB."

    window._controller.live_reset.emit()
    assert sfn.marks.shown() == []


def test_the_error_correction_page_counts_the_frames_and_names_the_codec(window) -> None:
    page = window._signal_page
    correction = page.page(CORRECTION)
    overview = page.page(OVERVIEW)
    page.show_page(CORRECTION)
    controller = window._controller
    dsp = controller.dsp_worker
    subchannel = Subchannel(5, 0, 72, is_uep=False, protection_index=2, eep_option=0)

    dsp.subchannel_info_ready.emit(SubchannelStatus(AudioComponent(5, 63), subchannel))
    dsp.superframe_ready.emit(_superframe(corrections=4))
    dsp.superframe_ready.emit(_superframe(corrections=2, broken=True))
    controller.audio_player.buffer_changed.emit(0.82, 1)
    window._render_plots()

    assert correction.status.text() == "DAB+ · Subchannel 5 · 96 kbit/s · EEP-A Level 3"
    assert correction.figures.text("frames") == "2/2 Superframes"
    assert correction.figures.text("corrected") == "6 bytes (Reed-Solomon)"
    assert correction.figures.text("lost") == "1 superframe"
    assert correction.figures.text("codec") == "HE-AAC v2"
    assert correction.figures.text("audio") == "48 kHz, Stereo"
    assert correction.figures.text("buffer") == "820 ms, 1 dropout"
    # The overview counts the same frames.
    assert overview.figures.text("frames") == "Superframes 2/2 in sync"
    assert correction.superframe_plot.plot.isVisible()

    controller.service_reset.emit()
    assert correction.figures.text("frames") == "-"
    assert correction.figures.text("codec") == "-"
    assert overview.figures.text("frames") == "-"


def test_a_station_in_classic_dab_shows_its_mp2_frames(window) -> None:
    page = window._signal_page
    correction = page.page(CORRECTION)
    page.show_page(CORRECTION)
    dsp = window._controller.dsp_worker

    dsp.mp2_frame_ready.emit(Mp2FrameResult(MP2_FORMAT, sync_ok=True, crc_ok=True))
    dsp.mp2_frame_ready.emit(Mp2FrameResult(MP2_FORMAT, sync_ok=True, crc_ok=False))
    window._render_plots()

    assert correction.mp2_plot.plot.isVisible()
    assert not correction.superframe_plot.plot.isVisible()
    assert (
        correction.layout_panel.heading.text()
        == "Layout of the MP2 frames, 24 ms of audio per frame"
    )
    assert correction.figures.text("frames") == "2/2 MP2 frames"
    assert correction.figures.text("corrected") == "does not apply to DAB"
    assert correction.figures.text("lost") == "1 frame (header CRC)"
    assert correction.figures.text("codec") == "MPEG-1 Layer II"


def test_the_codec_and_the_sound_are_named_apart():
    assert describe_format(HE_AAC_V2) == ("HE-AAC v2", "48 kHz, Stereo")
    he_aac = AudioFormat(True, True, True, False, 0)
    assert describe_format(he_aac) == ("HE-AAC", "48 kHz, Stereo")
    aac_lc_mono = AudioFormat(False, False, False, False, 0)
    assert describe_format(aac_lc_mono) == ("AAC-LC", "32 kHz, Mono")
    assert describe_format(MP2_FORMAT) == ("MPEG-1 Layer II", "48 kHz, Joint Stereo")


def test_the_ofdm_plots_are_labelled_by_the_desktop_tab_only(window) -> None:
    # In the window, which takes the tab with it when it goes.
    tab = OfdmTab(MODE_I, SAMPLE_RATE, SYMBOL_WINDOW_EXCERPT_SYMBOLS, parent=window)
    plots = window._signal_page.page(OFDM).plots

    assert tab._plots.time_plot.toolTip()
    assert tab._plots.spectrum_plot.getPlotItem().titleLabel.text
    assert not plots.time_plot.toolTip()
    assert not plots.spectrum_plot.getPlotItem().titleLabel.text
