"""The receiver's signal processing chain, running in its own thread.

`DspWorker` takes the IQ blocks from `sdr.worker.RtlSdrWorker` and drives everything
from the spectrum display to the decoded audio: frame synchronisation and tracking,
OFDM demodulation, FIC decoding (ensemble and service list), MSC decoding of the
selected service (time deinterleaving, Viterbi, then DAB+ superframes or the MP2 frames
of classic DAB) and the programme associated data (dynamic label, slideshow,
Journaline). In addition to the audio, any number of data services in packet mode can
be decoded at the same time (packet FEC, packets, data groups, MOT carousel of the
programme guide, RTCM stream of GEPOS, TPEG stream of ARD TPEG). Every intermediate
result that the GUI visualises is emitted as a Qt signal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from dab.announcement import AnnouncementMonitor, AnnouncementStatus
from dab.constants import TransmissionMode
from dab.data_group import DataGroup
from dab.data_services import (
    DSCTY_MOT,
    DSCTY_TDC,
    USER_APPLICATION_JOURNALINE,
    USER_APPLICATION_SPI,
    USER_APPLICATION_TPEG,
    DataComponent,
    XPadApplication,
    data_components,
    xpad_applications,
)
from dab.dls import DynamicLabelDecoder
from dab.fig import (
    FEC_SCHEME_RS,
    Announcement,
    AudioComponent,
    Ensemble,
    PacketComponent,
    Subchannel,
    parse_fib,
)
from dab.journaline import JournalineDecoder, JournalineParameters, parse_user_application_data
from dab.mot import MotDecoder, MotObject
from dab.mot_directory import MotCarousel
from dab.pad import PadReader, XPadSubfield
from dab.rtcm import RtcmStreamDecoder
from dab.spi import Schedule, ServiceInformation, decode_spi_object
from dab.tpeg import TpegDecoder
from dsp.channel import impulse_response, sfn_shares
from dsp.fic import check_fib_crc, extract_fibs_from_symbols
from dsp.iq_correction import IqCorrectionReport, IqCorrector
from dsp.mp2_frame import Mp2FrameSynchronizer
from dsp.msc import NUM_MSC_SYMBOLS_PER_FRAME, assemble_msc_cifs
from dsp.ofdm import FrameDemodulator
from dsp.packet import PacketModeDecoder
from dsp.qpsk import hard_decisions
from dsp.scan import MIN_VALID_FIBS, service_list_settled
from dsp.signal_capture import NullSnapshotter, SignalCaptureRecorder
from dsp.spectrum import compute_power_spectrum, detect_clipping
from dsp.subchannel import SubchannelDecoder
from dsp.superframe import NUM_LOGICAL_FRAMES_PER_SUPERFRAME, SuperframeResult, decode_superframe
from dsp.sync import find_null_symbol
from dsp.tii import TiiDetector, null_symbol_carriers, null_symbol_spectrum

# Half width of the search window in which the null symbol is still searched once
# locked. Transmitter and receiver clocks differ by only a few ppm (typically 1 sample
# per frame, i.e. ~5 ppm), so the position of the next null symbol one frame length
# later is predictable to within a few samples. What matters is that the window is
# guaranteed to contain ONLY ONE null symbol (they lie one frame length = 196 608
# samples apart).
NULL_TRACKING_WINDOW = 1024

# Reserve beyond the end of the frame before a frame is processed. The PRS fine
# alignment in `dsp.ofdm.demodulate_frame_symbols` can shift the frame start BACKWARDS
# by up to `fft_length/2 - guard_length` samples; without this reserve the buffer may
# then hold only 74 instead of 75 data symbols, and the frame silently drops out for
# the MSC.
FRAME_SLICE_MARGIN = 1024

# This many consecutive unreliable frames (see `dsp.ofdm.demodulate_frame_symbols`)
# count as a loss of frame synchronisation; after that the frame position is searched
# afresh. Needed because tracking a frame grid once locked does not, unlike a search
# over the whole buffer, correct itself: if acquisition locks onto a wrong position,
# e.g. while the gain control is still settling right after start-up, it would stay
# there forever. With intact reception a single unreliable frame is normal (about 1 in
# 460), a whole second of them is not.
MAX_UNRELIABLE_FRAMES = 10

# Interval at which the compute load is reported (in seconds of processed signal).
# Short enough to see load peaks, long enough not to update the display on every read
# block.
LOAD_REPORT_INTERVAL_SECONDS = 1.0

# The clocks of the load report: the wall clock, and the CPU time of the calling thread
# (CLOCK_THREAD_CPUTIME_ID on Linux; GetThreadTimes on Windows, which counts in ticks
# of 15.6 ms, coarse but exact enough over a second of signal). Module attributes so
# that tests can put their own in.
_wall_clock = time.perf_counter
_cpu_clock = time.thread_time

# This many consecutive superframes may miss the firecode before the superframe grid
# is searched afresh (ETSI TS 102 563, 5.2).
MAX_SUPERFRAME_MISSES = 5

# Interval at which the packet statistics and the progress of the MOT carousel are
# reported, in logical frames of the data subchannel (24 ms each): about once a second.
DATA_REPORT_INTERVAL_FRAMES = 42

# Duration of a logical frame (one CIF) in seconds.
LOGICAL_FRAME_SECONDS = 0.024

# This many symbols from the PRS onwards are shown in the time excerpt of
# `SymbolWindows`, in addition to the null symbol before it. Three suffice to see the
# grid of guard intervals and FFT windows; more would make the individual samples
# unrecognisable.
SYMBOL_WINDOW_EXCERPT_SYMBOLS = 3


@dataclass(frozen=True)
class SymbolWindows:
    """Input and output of the FFT per symbol for one frame, for the display.

    Positions in samples relative to the start of the PRS, exactly as the demodulator
    actually cut up the frame (after the fine alignment).
    """

    excerpt_start: int  # position of the first sample of `magnitude`
    magnitude: np.ndarray  # |x| of the IQ samples from the null symbol to the end of the excerpt
    # FFT of the null symbol (same window as for TII); None if not entirely in the buffer.
    null_spectrum: np.ndarray | None
    # FFT output per symbol from l=1 (PRS) onwards: (symbols, fft_length), DC in the middle.
    spectra: np.ndarray


@dataclass(frozen=True)
class SubchannelStatus:
    """What the FIC has revealed so far about the subchannel of the selected service.

    Both fields fill in as FIG 0/2 (component) and FIG 0/1 (subchannel) arrive; the
    GUI shows a waiting message for whatever is still None. During an announcement the
    receiver switched to (see `DspWorker.set_announcement_switching_enabled`), they
    describe the announcement's subchannel, and `announcement` is set.
    """

    component: AudioComponent | None
    subchannel: Subchannel | None
    announcement: Announcement | None = None


@dataclass(frozen=True)
class DataComponentStatus:
    """A data component being decoded, as far as the FIC has described it: `component`
    stays None until FIG 0/2 names the SCId in a service."""

    scid: int
    component: DataComponent | None


@dataclass(frozen=True)
class CarouselObject:
    """A completed object of the MOT carousel of a decoded data component; for the
    programme guide (SPI) with its SI or PI content already decoded."""

    mot_object: MotObject
    spi: ServiceInformation | Schedule | None = None


class _DataSubchannelChain:
    """Channel and packet decoding of one packet mode subchannel.

    Shared by all decoded components the subchannel carries: it is Viterbi decoded
    once, and the packet decoder separates the components by their packet address
    anyway.
    """

    def __init__(self) -> None:
        self.subchannel = SubchannelDecoder()
        self.packets: PacketModeDecoder | None = None
        # (FEC, addresses without data groups) the packet decoder was made for.
        self.packets_config: tuple[bool, frozenset[int]] | None = None
        self.frames_since_report = 0

    def reset(self) -> None:
        """After a gap in the CIF sequence: packets under way are lost."""
        self.subchannel.reset()
        if self.packets is not None:
            self.packets.reset()


class _DataComponentChain:
    """What is built from the data groups of one decoded component: the MOT carousel of
    a MOT component; for a TDC component the TPEG stream if FIG 0/13 announces TPEG, the
    RTCM stream otherwise (GEPOS)."""

    def __init__(self, scid: int) -> None:
        self.scid = scid
        self.component: DataComponent | None = None  # as far as the FIC describes it
        self.carousel = MotCarousel()
        self.rtcm = RtcmStreamDecoder()
        self.tpeg = TpegDecoder()

    def is_tpeg(self) -> bool:
        return self.component is not None and USER_APPLICATION_TPEG in self.component.app_types

    def interrupt(self) -> None:
        """The data groups have a gap: the streams lose the frame under way."""
        self.rtcm.interrupt()
        self.tpeg.interrupt()
        # Signal duration decoded since the component was chosen, for its data rates.
        self.seconds = 0.0


class DspWorker(QObject):
    """Processes the incoming IQ sample blocks: from spectrum and frame sync through
    OFDM demodulation, FIC and MSC decoding to DAB+ superframes or DAB audio frames and
    programme data.

    Lives in its own thread via `QObject.moveToThread` so that the signal processing
    never blocks the PyQt6 GUI. Input and output go exclusively through Qt
    signals/slots, so calls across thread boundaries are automatically queued in the
    respective thread's event loop.
    """

    spectrum_ready = pyqtSignal(object, object)  # freqs (Hz), power (dB)
    power_envelope_ready = pyqtSignal(
        object, object, object
    )  # time (s), power (dB), frame starts (s)
    constellation_ready = pyqtSignal(object)  # complex D-QPSK constellation points
    impulse_response_ready = pyqtSignal(object, object)  # delay (s), power (dB rel. strongest path)
    sfn_share_ready = pyqtSignal(
        float, float
    )  # share of other paths, share outside the guard interval (0..1)
    tii_ready = pyqtSignal(object)  # dsp.tii.TiiResult: detected transmitters of the SFN
    symbol_windows_ready = pyqtSignal(
        object
    )  # SymbolWindows: input and output of the FFT per symbol
    services_updated = pyqtSignal(
        object, object, object
    )  # ensemble label (str|None), service list (list[(SId, str)]), {SId: PTy code}
    # The service list counts as complete, by the rule of the scan
    # (`dsp.scan.service_list_settled`); once per stream, after its `services_updated`.
    service_list_completed = pyqtSignal()
    frequency_information_updated = pyqtSignal(
        object, object
    )  # EId of the ensemble (int|None), {EId: DAB frequencies in kHz} from FIG 0/21
    # {SId: EIds of the ensembles carrying the service} from FIG 0/24.
    service_ensembles_updated = pyqtSignal(object)
    frequency_offset_ready = pyqtSignal(float)  # estimated carrier frequency offset (Hz)
    clipping_ready = pyqtSignal(float)  # fraction of overdriven samples (0..1)
    subchannel_info_ready = pyqtSignal(
        object
    )  # SubchannelStatus, or None while no service is selected
    msc_bits_ready = pyqtSignal(
        object
    )  # (4, size*64) raw (still coded/interleaved) subchannel bits, hard decisions
    audio_bits_ready = pyqtSignal(
        object
    )  # (4, 24*bitRate) EEP/UEP-decoded info bits per logical frame (CIF)
    superframe_ready = pyqtSignal(object)  # dsp.superframe.SuperframeResult (DAB+ only)
    mp2_frame_ready = pyqtSignal(object)  # dsp.mp2_frame.Mp2FrameResult (DAB only)
    iq_correction_ready = pyqtSignal(
        object
    )  # IqCorrectionReport, or None while both corrections are off
    # Demand of the signal processing as a fraction of real time (wall-clock, 1.0 = just
    # keeping up), and the share of real time its thread spent computing in that (its
    # CPU time); the difference is time spent waiting, see `_report_load`.
    load_ready = pyqtSignal(float, float)
    dynamic_label_ready = pyqtSignal(object)  # dab.dls.DynamicLabel: text and DL Plus objects
    slide_ready = pyqtSignal(object)  # dab.mot.MotObject (image or header update of the slideshow)
    journaline_object_ready = pyqtSignal(object)  # dab.journaline.JmlObject
    # dab.journaline.JournalineParameters from FIG 0/13, None if the selected service
    # announces no Journaline (or FIG 0/13 has not arrived yet).
    journaline_parameters_ready = pyqtSignal(object)
    data_services_updated = pyqtSignal(object)  # list[dab.data_services.DataComponent]
    # tuple[DataComponentStatus, ...]: the data components being decoded, empty if none.
    data_decoding_ready = pyqtSignal(object)
    # Several data components can be decoded at once, and one subchannel can carry
    # several of them: the packet mode results therefore name their subchannel
    # (SubChId), whatever is built from the data groups names its component (SCId).
    packet_fec_frame_ready = pyqtSignal(int, object)  # SubChId, dsp.packet.FecFrameResult
    packet_statistics_ready = pyqtSignal(int, object)  # SubChId, dsp.packet.PacketStatistics, ~1/s
    mot_carousel_ready = pyqtSignal(int, object)  # SCId, dab.mot_directory.MotCarouselStatus, ~1/s
    mot_directory_ready = pyqtSignal(int, object)  # SCId, dab.mot_directory.MotDirectory
    data_object_ready = pyqtSignal(int, object)  # SCId, CarouselObject completed in the carousel
    rtcm_statistics_ready = pyqtSignal(int, object)  # SCId, dab.rtcm.RtcmStatistics, ~1/s
    # SCId, list[bytes]: the RTCM frames just completed, byte for byte as received, for
    # passing the stream on (net.ntrip). Connected directly, it is served from this
    # thread without waiting for the GUI.
    rtcm_frames_ready = pyqtSignal(int, object)
    # dab.rtcm.KeplerEphemeris or GlonassEphemeris, each only when it is new for its
    # satellite (the same ephemeris is repeated every few seconds).
    ephemeris_ready = pyqtSignal(object)
    # What an SSRZ message of the RTCM stream said (dab.ssrz.SsrzResult): metadata,
    # grids, ionosphere and troposphere corrections, time.
    ssrz_ready = pyqtSignal(object)
    # SCId, dab.tpeg.TpegFrame: a transport frame of a TPEG component with the messages
    # it carried, and the service information and statistics after it.
    tpeg_frame_ready = pyqtSignal(int, object)
    date_time_ready = pyqtSignal(object)  # dab.timecode.BroadcastTime (FIG 0/10, ~once per second)
    # dab.announcement.AnnouncementStatus of the selected service (FIG 0/18, 0/19), None
    # while none is selected; on every change and on every service selection.
    announcement_ready = pyqtSignal(object)
    # Number of the tuning (see `sdr.worker.RtlSdrWorker.tuned`) that all results emitted
    # from here on belong to; see `restart_stream`.
    stream_restarted = pyqtSignal(int)
    # dsp.signal_capture.SignalCapture: the frames asked for with `capture_signal`
    signal_captured = pyqtSignal(object)
    # dsp.signal_capture.SignalCapture of one frame: the stretch around its null symbol,
    # for every frame demodulated while `set_null_snapshots` is on
    null_snapshot_ready = pyqtSignal(object)

    def __init__(
        self,
        sample_rate: float,
        mode: TransmissionMode,
        fft_size: int = 2048,
        envelope_window: int = 128,
        max_data_symbols: int = 16,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._sample_rate = sample_rate
        self._mode = mode
        self._fft_size = fft_size
        self._envelope_window = envelope_window
        self._max_data_symbols = max_data_symbols
        self._ensemble = Ensemble()
        self._selected_sid: int | None = None
        self._demodulator = FrameDemodulator(mode, sample_rate)
        self._pad_reader = PadReader()
        self._dynamic_label = DynamicLabelDecoder()
        self._mot_decoder = MotDecoder()
        # Only while FIG 0/13 announces Journaline for the selected service; the X-PAD
        # application type it uses is only known from there.
        self._journaline: JournalineDecoder | None = None
        self._xpad_applications: tuple[XPadApplication, ...] | None = None
        self._announcement_monitor = AnnouncementMonitor()
        self._announcement_status: AnnouncementStatus | None = None  # as last reported
        # Off by default, like the GUI setting: whether the audio follows a running
        # announcement into its subchannel, and the component decoded while it does.
        self._switch_to_announcements = False
        self._announcement_audio: AudioComponent | None = None
        self._iq_corrector = IqCorrector()
        self._tii_detector = TiiDetector(sample_rate=sample_rate)
        # Same defaults as the GUI, which only reports changes.
        self._remove_dc = False
        self._correct_iq = False
        self._audio_subchannel = SubchannelDecoder()
        self._superframe_buffer: list[np.ndarray] = []
        self._superframe_locked = False
        self._superframe_misses = 0
        self._mp2_synchronizer = Mp2FrameSynchronizer()
        # The data components decoded in addition to the audio, by SCId (packet mode
        # components), in the order they were chosen; and the chains of the subchannels
        # they are in, by SubChId.
        self._data_components: dict[int, _DataComponentChain] = {}
        self._data_subchannels: dict[int, _DataSubchannelChain] = {}
        self._frame_buffer: np.ndarray | None = None  # samples accumulated across read blocks
        self._busy_seconds = 0.0  # wall-clock time spent on blocks since the last load report
        self._cpu_seconds = 0.0  # CPU time of this thread in that
        self._signal_seconds = 0.0  # signal duration processed in that time
        self._samples_received = 0  # absolute position of the next incoming sample
        self._buffer_start = 0  # absolute position of `_frame_buffer[0]`
        self._expected_null: int | None = None  # expected null symbol position in the buffer
        self._unreliable_frames = 0  # consecutive unreliable frames
        self._valid_fibs = 0  # CRC-valid FIBs of this stream
        self._ensemble_found_at: float | None = None  # signal time of the confirmation
        self._service_list_complete = False
        # Collects frames for the detail view while asked to.
        self._signal_capture: SignalCaptureRecorder | None = None
        self._null_snapshots: NullSnapshotter | None = None

    @pyqtSlot(int)
    def restart_stream(self, tuning: int) -> None:
        """Starts over on the samples of a new frequency: discards all ensemble and
        service information and the state of every stage collected so far.

        Slot for `sdr.worker.RtlSdrWorker.tuned`, which the worker emits in the sample
        stream right before the first block of the new frequency. Through the queued
        connection the reset therefore falls exactly between the last block of the old
        and the first of the new frequency, even if blocks of the old one are still
        waiting to be processed. Confirmed with `stream_restarted`: since all signals
        of this worker arrive in order, everything the GUI receives after it belongs
        to the new frequency, and anything before it can be dropped.
        """
        self._ensemble = Ensemble()
        self._audio_subchannel.forget()
        self._reset_audio_frames()
        self._set_xpad_applications(None)
        self._announcement_monitor.reset()
        self._announcement_audio = None
        # A data component is identified by its SCId, which only means something within
        # one ensemble.
        self._data_components.clear()
        self._data_subchannels.clear()
        self._iq_corrector.reset()
        self._tii_detector.reset()
        self._demodulator.reset()
        self._frame_buffer = None
        self._busy_seconds = 0.0
        self._cpu_seconds = 0.0
        self._signal_seconds = 0.0
        self._samples_received = 0
        self._buffer_start = 0
        self._expected_null = None
        self._unreliable_frames = 0
        self._valid_fibs = 0
        self._ensemble_found_at = None
        self._service_list_complete = False
        if self._signal_capture is not None:
            # The frames asked for come from the new frequency.
            self._signal_capture = SignalCaptureRecorder(
                self._signal_capture.frames, 0, self._mode, self._sample_rate
            )
        if self._null_snapshots is not None:
            self._null_snapshots = NullSnapshotter(self._mode, self._sample_rate)
        self.services_updated.emit(None, [], {})
        self.frequency_information_updated.emit(None, {})
        self.service_ensembles_updated.emit({})
        self._emit_subchannel_info()
        self._update_announcements(force=True)
        self.data_services_updated.emit([])
        self._emit_data_decoding()
        self.stream_restarted.emit(tuning)

    @pyqtSlot(int)
    def capture_signal(self, frames: int) -> None:
        """Keeps the next `frames` complete frames with their samples and reports them
        with `signal_captured` (see `dsp.signal_capture`); a capture under way starts
        over."""
        self._signal_capture = SignalCaptureRecorder(
            frames, self._samples_received, self._mode, self._sample_rate
        )

    @pyqtSlot(bool)
    def set_null_snapshots(self, enabled: bool) -> None:
        """Reports the stretch around the null symbol of every frame from now on, or no
        more (see `dsp.signal_capture.NullSnapshotter`)."""
        if not enabled:
            self._null_snapshots = None
        elif self._null_snapshots is None:
            self._null_snapshots = NullSnapshotter(self._mode, self._sample_rate)

    @pyqtSlot(bool)
    def set_dc_removal_enabled(self, enabled: bool) -> None:
        """Switches the DC offset correction (see `dsp.iq_correction`)."""
        self._remove_dc = enabled
        self._iq_corrector.reset()

    @pyqtSlot(bool)
    def set_iq_correction_enabled(self, enabled: bool) -> None:
        """Switches the IQ imbalance correction (see `dsp.iq_correction`)."""
        self._correct_iq = enabled
        self._iq_corrector.reset()

    @pyqtSlot(object)
    def select_service(self, sid: int | None) -> None:
        """Remembers the service the user selected (SId, None = none) and reports its
        subchannel configuration as soon as the FIC has delivered it.

        Survives `restart_stream`: a service selected from the stored service list
        before starting, or kept across a channel change, is decoded as soon as the
        FIC delivers its subchannel."""
        self._selected_sid = sid
        self._update_announcements(force=True)
        self._invalidate_audio_chain()
        self._emit_subchannel_info()
        self._xpad_applications = None  # unknown for the new service: report afresh
        self._update_xpad_applications()

    @pyqtSlot(bool)
    def set_announcement_switching_enabled(self, enabled: bool) -> None:
        """Whether the audio switches to the subchannel of a running announcement
        (FIG 0/19) that applies to the selected service, and back after its end.

        Only needed when the announcement is carried in another subchannel than the
        service; the service itself stays selected throughout.
        """
        self._switch_to_announcements = enabled
        self._update_announcements()

    @pyqtSlot(int, bool)
    def set_data_component_enabled(self, scid: int, enabled: bool) -> None:
        """Starts or stops decoding the packet mode data component with this SCId in
        addition to the audio.

        Any number of components can be decoded at the same time. Each subchannel
        among them costs a channel decoding of its own; components in the same
        subchannel share it. Like the service, a component is decoded as soon as the
        FIC has described its subchannel (FIG 0/3, FIG 0/1).
        """
        if enabled == (scid in self._data_components):
            return
        if enabled:
            self._data_components[scid] = _DataComponentChain(scid)
        else:
            del self._data_components[scid]
            if not self._data_components:
                self._data_subchannels.clear()
        self._emit_data_decoding()

    @pyqtSlot(object)
    def process_samples(self, samples: np.ndarray) -> None:
        started, cpu_started = _wall_clock(), _cpu_clock()
        self._process_samples(samples)
        self._report_load(
            _wall_clock() - started, _cpu_clock() - cpu_started, len(samples) / self._sample_rate
        )

    def _report_load(self, busy_seconds: float, cpu_seconds: float, signal_seconds: float) -> None:
        """Reports which fraction of real time the signal processing needs, and which
        fraction of real time its thread spent computing in that.

        The first is the decisive figure for this application: if it stays above 1.0,
        decoding falls behind reception. Hardly visible in the displays, but fatal for
        the audio, since playback runs exactly in real time and its reserve is then
        inevitably used up (see `audio.player`).

        It is measured by the wall clock, so it includes the time this thread waited:
        above all for the GIL, which the GUI thread keeps through every call into Qt
        (a QPainter painting the map, OpenGL waiting for the GPU to finish the globe)
        and only hands over between two of its own bytecodes, and beyond that for a
        core. The second figure is the thread's own CPU time, which counts its NumPy
        and numba work but no waiting. Far apart, the signal processing is not short
        of computing power but of turns at the GIL.
        """
        self._busy_seconds += busy_seconds
        self._cpu_seconds += cpu_seconds
        self._signal_seconds += signal_seconds
        if self._signal_seconds >= LOAD_REPORT_INTERVAL_SECONDS:
            self.load_ready.emit(
                self._busy_seconds / self._signal_seconds,
                self._cpu_seconds / self._signal_seconds,
            )
            self._busy_seconds = 0.0
            self._cpu_seconds = 0.0
            self._signal_seconds = 0.0

    def _process_samples(self, samples: np.ndarray) -> None:
        # Clipping is deliberately measured on the RAW signal: it is a property of the
        # ADC full scale, and a subtracted DC offset would move the samples away from
        # the limits and flatter the display.
        self.clipping_ready.emit(detect_clipping(samples))

        # Correct receiver errors (DC offset, IQ imbalance) before everything else, so
        # that spectrum and constellation show the cleaned-up signal as well (see
        # `dsp.iq_correction`).
        samples = self._iq_corrector.process(
            samples, remove_dc=self._remove_dc, correct_iq=self._correct_iq
        )
        self._emit_iq_correction_info()

        self._update_spectrum(samples)

        # Accumulate samples in a buffer of their own instead of decoding just one
        # frame per read block and discarding the rest: read blocks
        # (sdr.worker.DEFAULT_READ_SIZE, two frame lengths) and frame length are not
        # congruent, so otherwise an irregular number of frames would be skipped
        # between two consecutive decoded frames. For the FIC (every frame decodable
        # on its own) that is harmless, for the MSC time deinterleaver (see
        # dsp.time_interleave.TimeDeinterleaver) it is fatal: it assumes a gapless,
        # continuous sequence of CIFs. Skipped frames effectively shift what "16 CIFs
        # ago" means and thereby destroy every subsequent decode.
        block_start = self._samples_received
        self._samples_received += len(samples)
        if self._signal_capture is not None:
            self._signal_capture.add_block(block_start, samples)
        if self._null_snapshots is not None:
            self._null_snapshots.add_block(block_start, samples)
        if self._frame_buffer is None:
            self._frame_buffer = samples
            self._buffer_start = block_start
        else:
            self._frame_buffer = np.concatenate((self._frame_buffer, samples))

        # Decode first, then report the power envelope: its markers are meant to show
        # the frame starts the decoder actually synchronised to. That is why the frame
        # buffer is displayed and not the bare read block (see `_emit_power_envelope`).
        pending = self._frame_buffer
        pending_start = self._buffer_start
        frame_starts = self._drain_complete_frames()
        self._emit_power_envelope(pending, pending_start, frame_starts)
        if self._signal_capture is not None:
            capture = self._signal_capture.capture()
            if capture is not None:
                self._signal_capture = None
                self.signal_captured.emit(capture)

    def _drain_complete_frames(self) -> list[int]:
        """Cuts the sample buffer into consecutive transmission frames.

        After the first acquisition the frame boundaries are TRACKED instead of
        searched afresh every time. A plain minimum search over the whole buffer is
        unusable for this: with every incoming read block the search span grows beyond
        two frame lengths, then contains TWO null symbols, and `find_null_symbol`
        simply returns the quieter of the two. Since the power in the null symbol
        varies from frame to frame (by up to some 16 dB, partly because of the TII
        signal), the later of the two is the quieter one in a good half of the cases,
        and the frame in between is lost without replacement. For the FIC that is
        harmless (every frame is decoded independently), for the MSC it is fatal: the
        time deinterleaver (see `dsp.time_interleave.TimeDeinterleaver`) spreads every
        logical frame over 16 consecutive CIFs, so a single skipped frame permanently
        breaks the mapping and makes audio decoding impossible.

        Returns:
            The absolute sample positions of the null symbols of all frames processed
            in this call (for the markers in the power envelope).
        """
        mode = self._mode
        frame_starts: list[int] = []
        while True:
            null_index = self._locate_next_null()
            if null_index is None:
                return frame_starts
            # `FRAME_SLICE_MARGIN` of reserve so that the PRS fine alignment can shift
            # the frame backwards without data symbols missing at the end (see the
            # comment on the constant).
            if len(self._frame_buffer) < null_index + mode.frame_length + FRAME_SLICE_MARGIN:
                return frame_starts

            frame_starts.append(self._buffer_start + null_index)
            frame_usable = self._process_frame(self._frame_buffer, null_index)

            # Release the next frame up to `NULL_TRACKING_WINDOW` samples before the
            # expected null symbol; the expectation then lies exactly
            # `NULL_TRACKING_WINDOW` behind the new buffer start.
            consumed = null_index + mode.frame_length - NULL_TRACKING_WINDOW
            self._frame_buffer = self._frame_buffer[consumed:]
            self._buffer_start += consumed
            self._expected_null = NULL_TRACKING_WINDOW

            if not frame_usable or self._unreliable_frames >= MAX_UNRELIABLE_FRAMES:
                # Either the frame could not be demodulated completely or the frame
                # position is obviously wrong by now (see MAX_UNRELIABLE_FRAMES). In
                # both cases the CIF sequence has a gap, and the frame position is
                # searched afresh.
                self._expected_null = None
                self._unreliable_frames = 0
                self._invalidate_msc_chain()

    def _locate_next_null(self) -> int | None:
        """Position of the next null symbol in the buffer, or None (wait for more).

        Without a lock the search span is limited to at most ONE frame length so that
        it is guaranteed to contain only one null symbol; afterwards the search is
        confined to a narrow window around the predicted position (see
        `_drain_complete_frames`).
        """
        mode = self._mode
        buffer = self._frame_buffer

        if self._expected_null is None:
            span = min(len(buffer) - mode.frame_length, mode.frame_length)
            if span < mode.null_symbol_length:
                return None
            return find_null_symbol(buffer, mode.null_symbol_length, max_start=span)

        low = max(0, self._expected_null - NULL_TRACKING_WINDOW)
        high = self._expected_null + NULL_TRACKING_WINDOW
        if len(buffer) < high + mode.null_symbol_length:
            return None
        window = buffer[low : high + mode.null_symbol_length]
        return low + find_null_symbol(window, mode.null_symbol_length, max_start=high - low)

    def _invalidate_msc_chain(self) -> None:
        """Discards the state of all stages that require a gapless CIF sequence: time
        deinterleavers, the grid of superframes or MP2 frames, and packets and data
        groups under way.

        To be called as soon as the CIF sequence is interrupted, be it by a frame that
        could not be demodulated or a pause in which no subchannel bits at all came in.
        Otherwise the deinterleaver would carry on with data from before the gap and
        mix them with the new ones.
        """
        self._invalidate_audio_chain()
        self._invalidate_data_chains()

    def _invalidate_audio_chain(self) -> None:
        """The part of `_invalidate_msc_chain` for the audio subchannel, also used on a
        change of service."""
        self._audio_subchannel.reset()
        self._reset_audio_frames()

    def _reset_audio_frames(self) -> None:
        """Everything built from the audio subchannel's logical frames."""
        self._superframe_buffer.clear()
        self._superframe_locked = False
        self._superframe_misses = 0
        self._mp2_synchronizer.reset()
        # The dynamic label, the images and the Journaline pages are assembled over
        # many superframes and are as worthless after a break as the deinterleaver
        # contents.
        self._pad_reader.reset()
        self._dynamic_label.reset()
        self._mot_decoder.reset()
        if self._journaline is not None:
            self._journaline.reset()

    def _invalidate_data_chains(self) -> None:
        """The part of `_invalidate_msc_chain` for the data subchannels. Packets and data
        groups under way are lost; the MOT carousels keep their segments: they are
        identified by transport identifier and segment number, and the missing ones
        come with the next turn of the carousel."""
        for subchannel in self._data_subchannels.values():
            subchannel.reset()
        for component in self._data_components.values():
            component.interrupt()

    def _decode_programme_data(self, result: SuperframeResult) -> None:
        """Evaluates the programme associated data of a superframe.

        The dynamic label, the slideshow images and Journaline sit in the PAD field of
        the audio units (see `dab.pad`), i.e. in the same data stream as the audio, not
        in a subchannel of their own.
        """
        if not result.sync_ok:
            return
        for audio_unit in result.audio_units:
            self._process_pad(self._pad_reader.read(audio_unit))

    def _process_pad(self, subfields: list[XPadSubfield]) -> None:
        """Hands the X-PAD subfields of one access unit or audio frame to the dynamic
        label, the slideshow and Journaline.

        Not while the audio of an announcement is decoded: label, images and the X-PAD
        applications FIG 0/13 announces belong to the selected service, and the
        announcement's images would end up in its slideshow.
        """
        if not subfields or self._announcement_audio is not None:
            return
        label = self._dynamic_label.process(subfields)
        if label is not None:
            self.dynamic_label_ready.emit(label)
        for mot_object in self._mot_decoder.process(subfields):
            if mot_object.is_image or mot_object.is_header_update:
                self.slide_ready.emit(mot_object)
        if self._journaline is not None:
            for jml_object in self._journaline.process(subfields):
                self.journaline_object_ready.emit(jml_object)

    def _update_xpad_applications(self) -> None:
        """Takes the X-PAD applications FIG 0/13 announces for the selected service,
        once they are known or whenever they change."""
        applications = (
            None
            if self._selected_sid is None
            else xpad_applications(self._ensemble, self._selected_sid)
        )
        if applications is None or applications != self._xpad_applications:
            self._set_xpad_applications(applications)

    def _set_xpad_applications(self, applications: tuple[XPadApplication, ...] | None) -> None:
        """Configures PAD reader and Journaline for the applications in the X-PAD of the
        selected service (None: none selected or not known yet).

        FIG 0/13 names the X-PAD application type each application starts its data
        groups in; the next one continues them. Journaline is only decoded where it is
        announced, with its extended header length and point of entry (ETSI TS 102
        979, clause 8.1.2).
        """
        self._xpad_applications = applications
        applications = applications or ()
        self._pad_reader.set_signalled_applications(
            {app.xpad_app_type: app.continuation_type for app in applications if app.data_groups}
        )
        journaline = next(
            (
                app
                for app in applications
                if app.app_type == USER_APPLICATION_JOURNALINE and app.data_groups and not app.ca
            ),
            None,
        )
        parameters = None
        self._journaline = None
        if journaline is not None:
            parameters = parse_user_application_data(journaline.data) or JournalineParameters(0, 0)
            self._journaline = JournalineDecoder(
                journaline.xpad_app_type, parameters.extended_header_length
            )
        self.journaline_parameters_ready.emit(parameters)

    def _emit_iq_correction_info(self) -> None:
        if not self._remove_dc and not self._correct_iq:
            self.iq_correction_ready.emit(None)
            return
        corrector = self._iq_corrector
        self.iq_correction_ready.emit(
            IqCorrectionReport(
                dc_offset=abs(corrector.dc_offset) if self._remove_dc else None,
                quadrature_error_deg=corrector.quadrature_error_deg if self._correct_iq else None,
                gain_imbalance_db=corrector.gain_imbalance_db if self._correct_iq else None,
            )
        )

    def _update_spectrum(self, samples: np.ndarray) -> None:
        try:
            freqs, power_db = compute_power_spectrum(samples, self._sample_rate, self._fft_size)
        except ValueError:
            return
        self.spectrum_ready.emit(freqs, power_db)

    def _emit_power_envelope(
        self, samples: np.ndarray, buffer_start: int, frame_starts: list[int]
    ) -> None:
        """Reports the power envelope together with the null symbol positions the
        frame synchronisation locked onto.

        The markers deliberately come from the decoder and not from a separate minimum
        search run only for the display: such a second search would also find a real
        null symbol, but not necessarily the one being decoded right now, and the
        display would then show something the receiver does not actually do.

        Displayed is therefore the frame buffer as it stood before decoding, not the
        incoming read block: the buffer always holds back a remainder of the previous
        block, so decoding trails the newest block by about one block length. Over the
        bare read block, not a single one of the frame starts actually used would be
        visible. The buffer always ends with the newest sample, so the display stays
        current; by construction its length varies between two and three frames.
        """
        window = self._envelope_window
        trimmed_len = (len(samples) // window) * window
        if trimmed_len == 0:
            return

        power = np.abs(samples[:trimmed_len]) ** 2
        envelope_db = 10.0 * np.log10(np.maximum(power.reshape(-1, window).mean(axis=1), 1e-20))
        time_axis = np.arange(len(envelope_db)) * window / self._sample_rate

        marker_times = [
            (start - buffer_start) / self._sample_rate
            for start in frame_starts
            if 0 <= start - buffer_start < trimmed_len
        ]
        self.power_envelope_ready.emit(time_axis, envelope_db, marker_times)

    def _process_frame(self, samples: np.ndarray, null_index: int) -> bool:
        """Demodulates one frame and hands it to the FIC and MSC decoders.

        Returns:
            True if the frame could be demodulated completely. False means the CIF
            sequence has a gap at this point and the caller must reset the
            time-dependent stages.
        """
        # Without a selected service the smaller default number of data symbols
        # suffices for the constellation display and FIC decoding; only with a
        # selected service (or data components) are all 72 MSC symbols (l=5..76)
        # demodulated as well, to extract their subchannel data.
        msc_needed = self._selected_sid is not None or bool(self._data_components)
        max_symbols = self._mode.symbols_per_frame - 1 if msc_needed else self._max_data_symbols
        frame = self._demodulator.demodulate(samples, null_index, max_symbols)
        if frame is None:
            return False

        self._unreliable_frames = 0 if frame.reliable else self._unreliable_frames + 1

        # The constellation is ALWAYS shown, for frames rated unreliable as well: the
        # diagram is meant to show the signal actually received, especially when it is
        # bad. For an unreliable frame `frequency_offset_hz` is the last trustworthy
        # value (see `demodulate_frame_symbols`), i.e. exactly the frequency correction
        # applied to this frame, so the display stays calm by itself.
        self.constellation_ready.emit(np.concatenate(frame.symbols))
        self.frequency_offset_ready.emit(frame.frequency_offset_hz)
        response_power = self._emit_channel(frame.channel_estimate)
        frame_start = null_index + self._mode.null_symbol_length + frame.sample_correction
        if self._signal_capture is not None:
            self._signal_capture.add_frame(
                self._buffer_start + frame_start, frame.frequency_offset_hz
            )
        if self._null_snapshots is not None:
            snapshot = self._null_snapshots.add_frame(
                self._buffer_start + frame_start, frame.frequency_offset_hz
            )
            if snapshot is not None:
                self.null_snapshot_ready.emit(snapshot)
        # The null symbol preceding the frame, in the window of the TII.
        null_spectrum = null_symbol_spectrum(
            samples, frame_start, frame.frequency_offset_hz, self._sample_rate, self._mode
        )
        self._update_tii(null_spectrum, frame.channel_estimate, response_power)
        self._emit_symbol_windows(samples, frame_start, null_spectrum, frame.spectra)

        # The first three data symbols of a frame (l=2,3,4) carry the FIC (ETSI EN
        # 300 401, clause 14.4.1) independently of the MSC configuration, so they can
        # be decoded here unconditionally. An unreliable frame mostly yields FIBs with
        # a wrong CRC anyway, which are discarded automatically right there.
        if len(frame.symbols) >= 3:
            self._decode_fic(frame.symbols[:3])

        # The MSC needs ALL 72 symbols l=5..76. A truncated frame must not simply be
        # skipped silently: for the time deinterleaver that would be exactly the gap
        # that permanently breaks its 16-CIF mapping. Instead the return value reports
        # the gap upwards.
        if not msc_needed:
            return True
        if len(frame.symbols) < 3 + NUM_MSC_SYMBOLS_PER_FRAME:
            return False
        self._decode_msc(frame.symbols[3 : 3 + NUM_MSC_SYMBOLS_PER_FRAME])
        return True

    def _emit_channel(self, channel_estimate: np.ndarray) -> np.ndarray:
        """Reports impulse response and SFN share (see `dsp.channel`); returns the
        impulse response (linear power) for the transmitter identification.

        Like the constellation, for every frame, unreliable ones included.
        """
        delays, power = impulse_response(channel_estimate, self._mode, self._sample_rate)
        self.sfn_share_ready.emit(*sfn_shares(delays, power, self._mode, self._sample_rate))
        # Relative to the strongest path. The floor only prevents the logarithm of 0;
        # -120 dB is far below any noise.
        strongest = max(float(power.max()), np.finfo(float).tiny)
        self.impulse_response_ready.emit(
            delays, 10 * np.log10(np.maximum(power / strongest, 1e-12))
        )
        return power

    def _update_tii(
        self,
        null_spectrum: np.ndarray | None,
        channel_estimate: np.ndarray,
        response_power: np.ndarray,
    ) -> None:
        """Adds the null symbol preceding the frame, the frame's channel estimate and its
        impulse response to the transmitter identification (see `dsp.tii`); skipped
        while the null symbol does not lie entirely in the buffer."""
        if null_spectrum is not None:
            carriers = null_symbol_carriers(null_spectrum, self._mode)
            result = self._tii_detector.add(carriers, channel_estimate, response_power)
            self.tii_ready.emit(result)

    def _emit_symbol_windows(
        self,
        samples: np.ndarray,
        frame_start: int,
        null_spectrum: np.ndarray | None,
        spectra: np.ndarray,
    ) -> None:
        """Reports what the FFT per symbol received and delivered in this frame.

        Like the constellation, for every frame, unreliable ones included. The time
        excerpt is the signal as it entered the frame buffer; the frequency correction
        before the FFT only rotates the phase and leaves the magnitude unchanged.
        """
        mode = self._mode
        first = max(0, frame_start - mode.null_symbol_length)
        end = frame_start + SYMBOL_WINDOW_EXCERPT_SYMBOLS * mode.symbol_length
        self.symbol_windows_ready.emit(
            SymbolWindows(
                excerpt_start=first - frame_start,
                magnitude=np.abs(samples[first:end]),
                null_spectrum=null_spectrum,
                spectra=spectra,
            )
        )

    def _decode_fic(self, fic_symbols: list[np.ndarray]) -> None:
        fibs = extract_fibs_from_symbols(
            fic_symbols, self._mode.num_carriers, self._mode.fft_length
        )
        ensemble = self._ensemble

        services_changed = False
        subchannels_changed = False
        date_time_before = ensemble.date_time
        data_before = self._data_configuration_state()
        location_before = (ensemble.eid, tuple(ensemble.frequencies.items()))
        service_ensembles_before = dict(ensemble.service_ensembles)
        for fib in fibs:
            if not check_fib_crc(fib):
                continue
            self._valid_fibs += 1
            services_before = self._service_list_state()
            subchannels_before = (
                tuple(ensemble.audio_components.items()),
                tuple(ensemble.subchannels.items()),
            )
            parse_fib(fib, ensemble)
            services_after = self._service_list_state()
            subchannels_after = (
                tuple(ensemble.audio_components.items()),
                tuple(ensemble.subchannels.items()),
            )
            services_changed = services_changed or services_before != services_after
            subchannels_changed = subchannels_changed or subchannels_before != subchannels_after

        if services_changed:
            services = sorted(ensemble.services.items(), key=lambda item: item[1])
            self.services_updated.emit(ensemble.label, services, dict(ensemble.programme_types))
        self._check_service_list_complete()
        if (ensemble.eid, tuple(ensemble.frequencies.items())) != location_before:
            self.frequency_information_updated.emit(ensemble.eid, dict(ensemble.frequencies))
        if ensemble.service_ensembles != service_ensembles_before:
            self.service_ensembles_updated.emit(dict(ensemble.service_ensembles))
        if subchannels_changed:
            self._emit_subchannel_info()
        if ensemble.date_time is not date_time_before:
            self.date_time_ready.emit(ensemble.date_time)
        if self._data_configuration_state() != data_before:
            self.data_services_updated.emit(data_components(ensemble))
            self._emit_data_decoding()
            self._update_xpad_applications()
        # Also when no FIB was valid: an announcement whose FIG 0/19 stays away ends by
        # its timeout.
        self._announcement_monitor.update(ensemble, self._signal_time())
        self._update_announcements()

    def _check_service_list_complete(self) -> None:
        """Reports once per stream that the service list is complete, timed like the
        scan from the confirmation of the ensemble."""
        if self._service_list_complete:
            return
        now = self._signal_time()
        if self._ensemble_found_at is None:
            if self._valid_fibs < MIN_VALID_FIBS:
                return
            self._ensemble_found_at = now
        if service_list_settled(self._ensemble, now - self._ensemble_found_at):
            self._service_list_complete = True
            self.service_list_completed.emit()

    def _signal_time(self) -> float:
        """Seconds of signal received since the start of the stream."""
        return self._samples_received / self._sample_rate

    def _update_announcements(self, *, force: bool = False) -> None:
        """Reports the announcement situation of the selected service when it changed
        (with `force` in any case), and moves the audio to the subchannel of a running
        announcement and back, if that is enabled."""
        status = (
            None
            if self._selected_sid is None
            else self._announcement_monitor.status(
                self._ensemble, self._selected_sid, self._signal_time()
            )
        )
        changed = status != self._announcement_status
        if changed or force:
            self._announcement_status = status
            self.announcement_ready.emit(status)

        audio = self._announcement_audio_target()
        if audio != self._announcement_audio or (audio is not None and changed):
            # The subchannel decoder notices the other subchannel by itself and starts
            # afresh, the frame searches and the PAD along with it.
            self._announcement_audio = audio
            self._emit_subchannel_info()

    def _announcement_audio_target(self) -> AudioComponent | None:
        """The audio component to decode instead of the selected service's own: that of
        the running announcement, if switching is enabled and the announcement is in
        another subchannel whose configuration is known."""
        status = self._announcement_status
        if not self._switch_to_announcements or status is None:
            return None
        target = status.source_component
        if target is None or target.subchannel_id not in self._ensemble.subchannels:
            return None
        own = self._ensemble.audio_components.get(self._selected_sid)
        if own is not None and own.subchannel_id == target.subchannel_id:
            return None
        return target

    def _data_configuration_state(self) -> tuple:
        """What the data components and the X-PAD applications are resolved from, for
        detecting a change (see `dab.data_services`)."""
        ensemble = self._ensemble
        return tuple(
            tuple(table.items())
            for table in (
                ensemble.service_components,
                ensemble.audio_components,
                ensemble.packet_components,
                ensemble.component_references,
                ensemble.user_applications,
                ensemble.fec_schemes,
                ensemble.subchannels,
                ensemble.services,
                ensemble.data_services,
            )
        )

    def _service_list_state(self) -> tuple:
        """What `services_updated` reports, for detecting a change."""
        ensemble = self._ensemble
        return (
            ensemble.label,
            tuple(ensemble.services.items()),
            tuple(ensemble.programme_types.items()),
        )

    def _selected_subchannel(self) -> tuple[AudioComponent | None, Subchannel | None]:
        """Component and subchannel of the selected service, as far as the FIC has
        delivered them; during an announcement switched to, those of the announcement."""
        component = self._announcement_audio or self._ensemble.audio_components.get(
            self._selected_sid
        )
        if component is None:
            return None, None
        return component, self._ensemble.subchannels.get(component.subchannel_id)

    def _decode_msc(self, msc_symbols: list[np.ndarray]) -> None:
        cifs = assemble_msc_cifs(msc_symbols, self._mode.num_carriers, self._mode.fft_length)
        if self._selected_sid is not None:
            self._decode_audio_subchannel(cifs)
        if self._data_components:
            self._decode_data_subchannels(cifs)

    def _decode_audio_subchannel(self, cifs: list[np.ndarray]) -> None:
        component, subchannel = self._selected_subchannel()
        if subchannel is None:
            # No subchannel bits for this frame; any deinterleaver state would no
            # longer line up in time afterwards.
            self._invalidate_audio_chain()
            return

        frames = self._audio_subchannel.push(cifs, subchannel)
        self.msc_bits_ready.emit(hard_decisions(np.stack(frames.raw_bits)))
        if frames.restarted:
            self._reset_audio_frames()
        if not frames.logical_frames:
            return
        self.audio_bits_ready.emit(np.stack(frames.logical_frames))

        if component.is_dab_plus:
            for frame_bits in frames.logical_frames:
                self._feed_superframe(frame_bits)
        elif component.is_dab:
            for frame_bits in frames.logical_frames:
                self._feed_mp2_frames(frame_bits)

    def _decode_data_subchannels(self, cifs: list[np.ndarray]) -> None:
        """Decodes the subchannels of the data components being decoded, each only once
        however many of them it carries."""
        ensemble = self._ensemble
        members: dict[int, list[tuple[_DataComponentChain, PacketComponent]]] = {}
        for scid, component in self._data_components.items():
            packet = ensemble.packet_components.get(scid)
            if packet is not None and packet.subchannel_id in ensemble.subchannels:
                members.setdefault(packet.subchannel_id, []).append((component, packet))
        # A subchannel no longer needed (its components stopped, or a reconfiguration
        # moved them) starts from scratch should it be needed again.
        for subchannel_id in self._data_subchannels.keys() - members.keys():
            del self._data_subchannels[subchannel_id]
        for subchannel_id, components in members.items():
            self._decode_data_subchannel(cifs, ensemble.subchannels[subchannel_id], components)

    def _decode_data_subchannel(
        self,
        cifs: list[np.ndarray],
        subchannel: Subchannel,
        components: list[tuple[_DataComponentChain, PacketComponent]],
    ) -> None:
        """Decodes one data subchannel down to its packets and feeds the data groups of
        the components being decoded in it on: to the carousel for a MOT component, to
        the RTCM stream for a transparent data channel (TDC)."""
        subchannel_id = subchannel.subchannel_id
        chain = self._data_subchannels.get(subchannel_id)
        if chain is None:
            chain = self._data_subchannels[subchannel_id] = _DataSubchannelChain()
        decoder = self._packet_decoder_for(chain, subchannel_id)
        frames = chain.subchannel.push(cifs, subchannel)
        if frames.restarted:
            decoder.reset()
            for component, _ in components:
                component.interrupt()

        # Only components whose packets carry data groups (FIG 0/3) get any.
        routes = {
            packet.packet_address: (component, packet)
            for component, packet in components
            if packet.data_groups
        }
        for frame_bits in frames.logical_frames:
            output = decoder.push(np.packbits(frame_bits).tobytes())
            for fec_frame in output.fec_frames:
                self.packet_fec_frame_ready.emit(subchannel_id, fec_frame)
            for address, group in output.data_groups:
                route = routes.get(address)
                if route is not None:
                    self._feed_data_group(*route, group)
            chain.frames_since_report += 1
        for component, _ in components:
            component.seconds += len(frames.logical_frames) * LOGICAL_FRAME_SECONDS

        if chain.frames_since_report >= DATA_REPORT_INTERVAL_FRAMES:
            chain.frames_since_report = 0
            self.packet_statistics_ready.emit(subchannel_id, decoder.statistics())
            for component, packet in routes.values():
                if packet.dscty == DSCTY_MOT:
                    self.mot_carousel_ready.emit(component.scid, component.carousel.status())
                elif packet.dscty == DSCTY_TDC and not component.is_tpeg():
                    self.rtcm_statistics_ready.emit(
                        component.scid, component.rtcm.statistics(component.seconds)
                    )

    def _packet_decoder_for(
        self, chain: _DataSubchannelChain, subchannel_id: int
    ) -> PacketModeDecoder:
        """The packet decoder of a data subchannel, made afresh when FIG 0/14 or FIG 0/3
        change what it has to do (FEC on or off, addresses without data groups)."""
        ensemble = self._ensemble
        fec = ensemble.fec_schemes.get(subchannel_id) == FEC_SCHEME_RS
        raw_addresses = frozenset(
            other.packet_address
            for other in ensemble.packet_components.values()
            if other.subchannel_id == subchannel_id and not other.data_groups
        )
        config = (fec, raw_addresses)
        if chain.packets is None or config != chain.packets_config:
            chain.packets = PacketModeDecoder(fec=fec, raw_addresses=raw_addresses)
            chain.packets_config = config
        return chain.packets

    def _feed_data_group(
        self, component: _DataComponentChain, packet: PacketComponent, group: DataGroup
    ) -> None:
        if packet.dscty == DSCTY_MOT:
            self._feed_carousel(component, group)
        elif packet.dscty == DSCTY_TDC and component.is_tpeg():
            for frame in component.tpeg.push(group):
                self.tpeg_frame_ready.emit(component.scid, frame)
        elif packet.dscty == DSCTY_TDC:
            # Any other TDC component is searched for RTCM frames: the CRC-24Q tells
            # whether it carries them. GEPOS does.
            for ephemeris in component.rtcm.push(group):
                self.ephemeris_ready.emit(ephemeris)
            for result in component.rtcm.take_ssrz():
                self.ssrz_ready.emit(result)
            frames = component.rtcm.take_frames()
            if frames:
                self.rtcm_frames_ready.emit(component.scid, frames)

    def _feed_carousel(self, component: _DataComponentChain, group: DataGroup) -> None:
        carousel = component.carousel
        directory = carousel.directory
        completed = carousel.process(group)
        if carousel.directory is not directory:
            self.mot_directory_ready.emit(component.scid, carousel.directory)
        described = component.component
        is_spi = described is not None and USER_APPLICATION_SPI in described.app_types
        for mot_object in completed:
            spi = decode_spi_object(mot_object) if is_spi else None
            self.data_object_ready.emit(component.scid, CarouselObject(mot_object, spi))

    def _feed_superframe(self, frame_bits: np.ndarray) -> None:
        """Collects logical frames into DAB+ superframes (ETSI TS 102 563, 5.1).

        A superframe spans 5 logical frames, but its boundary is not tied to the frame
        clock and has to be found through the firecode. A plain sliding window over
        the last 5 logical frames therefore systematically fails in 4 out of 5 cases.
        Here the grid is locked once and then counted on in steps of 5, so that every
        reported `SuperframeResult` is a genuine statement about reception quality.
        """
        self._superframe_buffer.append(frame_bits)
        while len(self._superframe_buffer) >= NUM_LOGICAL_FRAMES_PER_SUPERFRAME:
            result = decode_superframe(self._superframe_buffer[:NUM_LOGICAL_FRAMES_PER_SUPERFRAME])
            if result.sync_ok:
                self._superframe_locked = True
                self._superframe_misses = 0
            elif not self._superframe_locked:
                # Not locked yet: advance frame by frame until the firecode confirms
                # the superframe boundary.
                del self._superframe_buffer[0]
                continue
            else:
                # Locked: a single corrupted superframe does not shift the grid but is
                # reported (and counts towards the re-search).
                self._superframe_misses += 1
            self.superframe_ready.emit(result)
            self._decode_programme_data(result)
            del self._superframe_buffer[:NUM_LOGICAL_FRAMES_PER_SUPERFRAME]
            if self._superframe_misses >= MAX_SUPERFRAME_MISSES:
                self._superframe_locked = False
                self._superframe_misses = 0

    def _feed_mp2_frames(self, frame_bits: np.ndarray) -> None:
        """Hands a logical frame to the MP2 frame search of classic DAB and reports the
        frames completed with it (see `dsp.mp2_frame.Mp2FrameSynchronizer`).

        The PAD is read before reporting, so that the display can show how much of
        the frame the X-PAD occupies. It is read even when the header CRC failed: that
        CRC does not cover the PAD, which is protected by the CRCs of its own data
        groups (the same approach as DABlin).
        """
        for result in self._mp2_synchronizer.push(frame_bits):
            if result.sync_ok:
                subfields = self._pad_reader.read_mp2_frame(
                    result.frame, result.audio_format.scf_crc_length
                )
                result.x_pad_length = self._pad_reader.x_pad_length
                self._process_pad(subfields)
            self.mp2_frame_ready.emit(result)

    def _emit_subchannel_info(self) -> None:
        if self._selected_sid is None:
            self.subchannel_info_ready.emit(None)
            return
        announcement = None
        if self._announcement_audio is not None and self._announcement_status is not None:
            announcement = self._announcement_status.active
        self.subchannel_info_ready.emit(
            SubchannelStatus(*self._selected_subchannel(), announcement=announcement)
        )

    def _emit_data_decoding(self) -> None:
        described: dict[int, DataComponent] = {}
        if self._data_components:
            for component in data_components(self._ensemble):
                if component.is_packet_mode:
                    described.setdefault(component.scid, component)
        for scid, chain in self._data_components.items():
            chain.component = described.get(scid)
        self.data_decoding_ready.emit(
            tuple(
                DataComponentStatus(scid, chain.component)
                for scid, chain in self._data_components.items()
            )
        )
