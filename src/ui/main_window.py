"""The main window: controls, status line, service panel and the tabs, on top of a
`ui.receiver_controller.ReceiverController` that runs the receiver."""

from __future__ import annotations

from PyQt6.QtCore import QSignalBlocker, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QActionGroup, QBrush, QCloseEvent, QColor, QIcon
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from dab.announcement import AnnouncementStatus, announcement_types
from dab.channels import BAND_III_CHANNELS
from dab.constants import MODE_I, SAMPLE_RATE
from dab.journaline import JournalineParameters
from dsp.channel import MAIN_PATH_HALF_WIDTH_SECONDS
from dsp.iq_correction import IqCorrectionReport
from dsp.mp2_frame import Mp2FrameResult
from dsp.pipeline import SYMBOL_WINDOW_EXCERPT_SYMBOLS, SubchannelStatus
from dsp.superframe import SUPERFRAME_SECONDS, SuperframeResult
from dsp.tii import TII_AVERAGED_FRAMES, TiiResult
from i18n import available_languages, tr, tr_context, tr_noop, tr_plural
from sdr.rspduo import FREQ_CORRECTION_RANGE_PPM, IF_GAIN_REDUCTION_RANGE_DB, LNA_GAIN_REDUCTION_DB
from ui.atmosphere_tab import AtmosphereTab
from ui.corrections_tab import CorrectionsTab
from ui.data_tab import DataTab
from ui.epg_tab import EpgTab
from ui.gnss_model import GnssModel
from ui.gnss_tab import GnssTab
from ui.journaline_tab import JournalineTab
from ui.language import restart_question, store_language, stored_language
from ui.ofdm_tab import OfdmTab
from ui.programme_tab import ProgrammeTab
from ui.receiver_controller import (
    CLIPPING_WARNING_THRESHOLD,
    DEVICE_RTLSDR,
    DEVICES,
    FREQUENCY_RANGE_MHZ,
    LOAD_WARNING_THRESHOLD,
    RTLSDR_PPM_RANGE,
    AnnouncementVisit,
    DeviceSettings,
    ReceiverController,
    ReceptionState,
    describe_other_channels,
    describe_stored,
    volume_text,
)
from ui.service_list import LOGO_SIZE, SID_ROLE, ServiceListDelegate, service_item, show_logos
from ui.service_tab import ServiceTab
from ui.signal_tab import SignalTab
from ui.traffic_model import TrafficModel
from ui.traffic_tab import TrafficTab

CHANNEL_MANUAL = tr_noop("Manual (frequency)")

# Highlighting of channels in the selection for which an ensemble is stored. White
# text on dark green stays readable in light and dark colour schemes alike.
RECEIVABLE_CHANNEL_BACKGROUND = (46, 125, 50)
RECEIVABLE_CHANNEL_FOREGROUND = (255, 255, 255)

# Rate at which the expensive plots are redrawn. The data are still recorded without
# gaps; only the drawing is batched. Needed because constellation and MSC bitmap cost
# about 11 ms per redraw and at about 10 frames per second together consumed almost a
# quarter of a core. That load is missing from the signal processing, which has to
# deliver the audio in real time; nothing of it is visible, since none of the plots
# shows single events.
PLOT_REFRESH_INTERVAL_MS = 200


class MainWindow(QMainWindow):
    """The surface for a desktop screen: every step of the signal chain in a tab of its
    own. Call `controller.start()` once it is constructed."""

    # The user asked to start anew in the language chosen; `main` does it once the window
    # has closed.
    restart_requested = pyqtSignal()

    def __init__(self, controller: ReceiverController) -> None:
        super().__init__()
        self.setWindowTitle("DAB+ Radio")
        self.resize(1000, 900)

        self._controller = controller
        self._frames_synced = 0
        self._frames_total = 0
        self._frames_crc_errors = 0
        self._pending_tii: TiiResult | None = None
        self._audio_format_text: str | None = None
        self._audio_state_text = "-"
        self._audio_buffer_seconds = 0.0
        # SId -> logo from the programme guides, for the service list.
        self._service_logos: dict[int, QIcon] = {}
        self._restart_box: QMessageBox | None = None

        self._build_menu()
        controls_layout = self._build_controls()
        status_layout = self._build_status_labels()
        self._tabs = self._build_tabs()
        services_layout = self._build_side_panel()

        plots_layout = QVBoxLayout()
        plots_layout.addLayout(controls_layout)
        plots_layout.addLayout(status_layout)
        plots_layout.addWidget(self._tabs)

        root_layout = QHBoxLayout()
        root_layout.addLayout(plots_layout, stretch=3)
        root_layout.addLayout(services_layout, stretch=1)

        central_widget = QWidget()
        central_widget.setLayout(root_layout)
        self.setCentralWidget(central_widget)

        self._connect_controller()
        self._start_timers()

    # --- Construction ---------------------------------------------------------------

    def _build_menu(self) -> None:
        """The menu bar: "View › Language", each language by its own name."""
        view = self.menuBar().addMenu(tr("&View"))
        languages = view.addMenu(tr("&Language"))
        self._language_group = QActionGroup(self)
        chosen = stored_language(self._controller.settings)
        for code, name in available_languages().items():
            action = languages.addAction(name)
            action.setCheckable(True)
            action.setChecked(code == chosen)
            action.setData(code)
            self._language_group.addAction(action)
            action.triggered.connect(lambda _checked=False, c=code: self._on_language_chosen(c))

    def _build_controls(self) -> QVBoxLayout:
        """The two rows of controls at the top: channel, scan, frequency, audio and the
        start/stop button; beneath, the device with its gain and frequency correction and
        the corrections."""
        controller = self._controller
        self._channel_combo = QComboBox()
        # The entries get coloured (see `_refresh_channel_item`). Some styles otherwise
        # draw the drop-down like a menu and ignore the background colour of
        # individual entries.
        self._channel_combo.setItemDelegate(QStyledItemDelegate(self._channel_combo))
        # With ensemble names the entries get wider; the selection grows with them.
        self._channel_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        for channel in BAND_III_CHANNELS:
            self._channel_combo.addItem(channel, channel)
            self._refresh_channel_item(channel)
        self._channel_combo.addItem(tr(CHANNEL_MANUAL), None)
        self._channel_combo.setCurrentIndex(self._channel_index(controller.channel))
        self._channel_combo.currentIndexChanged.connect(self._on_channel_changed)

        self._scan_button = QPushButton(tr("Scan"))
        self._scan_button.setToolTip(
            tr(
                "Searches all channels in Band III for DAB ensembles and stores their station "
                "lists. Channels with an ensemble appear green in the channel selection, with "
                "the name of the ensemble.\n\n"
                "A channel counts as receivable if FIBs can be decoded there without errors "
                "(with a valid CRC) - the level alone does not count. Uses device, gain and "
                "PPM correction as set; only possible while no reception is running."
            )
        )
        self._scan_button.clicked.connect(self._on_scan_clicked)

        self._frequency_spin = QDoubleSpinBox()
        self._frequency_spin.setSuffix(" MHz")
        self._frequency_spin.setRange(*FREQUENCY_RANGE_MHZ)
        self._frequency_spin.setDecimals(3)
        self._frequency_spin.setValue(controller.frequency_mhz)
        self._frequency_spin.setEnabled(controller.channel is None)
        # Report typed frequencies only once confirmed (Enter, leaving the field), not
        # with every keystroke: during reception every change retunes the stick.
        self._frequency_spin.setKeyboardTracking(False)
        self._frequency_spin.valueChanged.connect(self._on_frequency_changed)

        settings = controller.device_settings
        self._gain_spin = QDoubleSpinBox()
        self._gain_spin.setSuffix(" dB")
        self._gain_spin.setRange(*controller.gain_range_db)
        self._gain_spin.setDecimals(1)
        self._gain_spin.setSingleStep(0.1)
        self._gain_spin.setValue(settings.gain_db)
        self._gain_spin.setToolTip(
            tr(
                "Gain of LNA and mixer in the tuner. The IF control of the RTL2832U keeps the "
                "level at the A/D converter at about -9 dBFS anyway; the gain mainly decides "
                "the noise of the input itself. Weak channels therefore need more gain, even "
                "if level and spectrum hardly change with it."
            )
        )

        self._agc_checkbox = QCheckBox("AGC")
        self._agc_checkbox.setToolTip(
            tr(
                "Automatic gain control instead of a manual gain. Not recommended for DAB: "
                "the control runs into the null symbol, which is sent every 96 ms, and "
                "changes the level within the frame. That disturbs the differential "
                "demodulation, and the tuner overloads on top of it."
            )
        )
        self._agc_checkbox.toggled.connect(self._on_agc_toggled)
        # Set after connecting so that the gain field is disabled along with it. Off by
        # default, see `ui.receiver_controller.DEFAULT_GAIN_DB`.
        self._agc_checkbox.setChecked(settings.agc)

        self._dc_checkbox = QCheckBox(tr("DC correction"))
        self._dc_checkbox.setChecked(controller.dc_correction)
        self._dc_checkbox.setToolTip(
            tr(
                "Removes the DC offset that the direct-conversion mixer of the tuner produces "
                "(in the spectrum the narrow peak exactly in the centre of the band). In DAB "
                "it hits the unused centre carrier k=0, so it hardly disturbs the decoding - "
                "but it makes spectrum and constellation cleaner."
            )
        )
        self._dc_checkbox.toggled.connect(controller.set_dc_correction)

        self._iq_checkbox = QCheckBox(tr("IQ correction"))
        self._iq_checkbox.setChecked(controller.iq_correction)
        self._iq_checkbox.setToolTip(
            tr(
                "Evens out gain and phase errors between the I and Q branches. Without it a "
                "weak mirror image of the signal appears about the centre of the band, which "
                "in DAB falls onto the signal itself and acts as additional noise."
            )
        )
        self._iq_checkbox.toggled.connect(controller.set_iq_correction)

        self._ppm_spin = QSpinBox()
        self._ppm_spin.setSuffix(" ppm")
        self._ppm_spin.setRange(*RTLSDR_PPM_RANGE)
        self._ppm_spin.setValue(settings.ppm)
        self._ppm_spin.setToolTip(
            tr(
                "Frequency correction of the RTL-SDR crystal in ppm. Takes effect only when "
                "reception is (re)started; adjust it step by step until the frequency offset "
                "shown is zero."
            )
        )

        self._device_combo = QComboBox()
        for device, name in DEVICES.items():
            self._device_combo.addItem(name, device)
        self._device_combo.setCurrentIndex(self._device_combo.findData(settings.device))
        self._device_combo.setToolTip(
            tr(
                "Receiver: an RTL-SDR stick or one of the two tuners of the SDRplay RSPduo. "
                "The RSPduo runs via the SDRplay API, whose service “SDRplay API Service” "
                "must be running. It receives in single tuner mode and so occupies the whole "
                "device: as long as the radio receives, no other program can use the second "
                "tuner. Can only be changed while no reception is running."
            )
        )
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)

        self._lna_combo = QComboBox()
        for state, reduction in enumerate(LNA_GAIN_REDUCTION_DB):
            self._lna_combo.addItem(
                f"{state} (−{reduction} dB)" if reduction else "0 (0 dB)", state
            )
        self._lna_combo.setCurrentIndex(self._lna_combo.findData(settings.lna_state))
        self._lna_combo.setToolTip(
            tr(
                "State of the low-noise amplifier (LNA) in the RSPduo, in brackets the gain "
                "reduction in Band III. State 0 amplifies most and adds the least noise; "
                "higher states only if the RSPduo reports overload."
            )
        )

        self._if_gain_reduction_spin = QSpinBox()
        self._if_gain_reduction_spin.setRange(*IF_GAIN_REDUCTION_RANGE_DB)
        self._if_gain_reduction_spin.setSuffix(" dB")
        self._if_gain_reduction_spin.setValue(settings.if_gain_reduction_db)
        self._if_gain_reduction_spin.setToolTip(
            tr(
                "Reduction of the gain at the intermediate frequency, ahead of the A/D "
                "converter of the RSPduo. It mainly decides the level at the converter: less "
                "reduction lifts weak signals further above its resolution, too little "
                "overloads it."
            )
        )

        self._rspduo_ppm_spin = QDoubleSpinBox()
        self._rspduo_ppm_spin.setRange(*FREQ_CORRECTION_RANGE_PPM)
        self._rspduo_ppm_spin.setDecimals(2)
        self._rspduo_ppm_spin.setSingleStep(0.01)
        self._rspduo_ppm_spin.setSuffix(" ppm")
        self._rspduo_ppm_spin.setValue(settings.rspduo_ppm)
        self._rspduo_ppm_spin.setToolTip(
            tr(
                "Frequency correction of the RSPduo oscillator in ppm; the SDRplay API "
                "corrects mixer and sampling clock together. Rarely needed, the oscillator is "
                "accurate to 0.5 ppm. Takes effect only when reception is (re)started; adjust "
                "it step by step until the frequency offset shown is zero."
            )
        )

        # Only the controls of the selected device are shown (see `_on_device_changed`).
        rtlsdr_layout = QHBoxLayout()
        rtlsdr_layout.setContentsMargins(0, 0, 0, 0)
        rtlsdr_layout.addWidget(QLabel(tr("Gain:")))
        rtlsdr_layout.addWidget(self._gain_spin)
        rtlsdr_layout.addWidget(self._agc_checkbox)
        rtlsdr_layout.addWidget(QLabel(tr("PPM correction:")))
        rtlsdr_layout.addWidget(self._ppm_spin)
        self._rtlsdr_controls = QWidget()
        self._rtlsdr_controls.setLayout(rtlsdr_layout)

        rspduo_layout = QHBoxLayout()
        rspduo_layout.setContentsMargins(0, 0, 0, 0)
        rspduo_layout.addWidget(QLabel(tr("LNA state:")))
        rspduo_layout.addWidget(self._lna_combo)
        rspduo_layout.addWidget(QLabel(tr("IF gain reduction:")))
        rspduo_layout.addWidget(self._if_gain_reduction_spin)
        rspduo_layout.addWidget(QLabel(tr("PPM correction:")))
        rspduo_layout.addWidget(self._rspduo_ppm_spin)
        self._rspduo_controls = QWidget()
        self._rspduo_controls.setLayout(rspduo_layout)
        self._on_device_changed()

        # Translators: the switch for the sound of the selected station.
        self._audio_checkbox = QCheckBox(tr_context("audio switch", "Audio"))
        self._audio_checkbox.setChecked(controller.audio_enabled)
        self._audio_checkbox.setToolTip(
            tr("Plays the decoded audio of the selected station through the sound card.")
        )
        # The box tells the controller and follows it: the audio is also muted from
        # elsewhere (MQTT).
        self._audio_checkbox.clicked.connect(controller.set_audio_enabled)
        controller.audio_enabled_changed.connect(self._audio_checkbox.setChecked)

        self._volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(controller.volume_percent)
        self._volume_slider.setFixedWidth(90)
        self._volume_slider.valueChanged.connect(controller.set_volume_percent)
        self._show_volume()
        controller.volume_changed.connect(self._show_volume)

        self._start_stop_button = QPushButton(tr("Start"))
        self._start_stop_button.clicked.connect(self._on_start_stop_clicked)

        # Handed to the controller, which stores them, with every change.
        for changed in (
            self._device_combo.currentIndexChanged,
            self._gain_spin.valueChanged,
            self._agc_checkbox.toggled,
            self._ppm_spin.valueChanged,
            self._lna_combo.currentIndexChanged,
            self._if_gain_reduction_spin.valueChanged,
            self._rspduo_ppm_spin.valueChanged,
        ):
            changed.connect(self._on_device_settings_changed)

        # Two rows, so that the controls do not set the width of the window: what is
        # received and heard, then the receiver and its corrections.
        reception_row = QHBoxLayout()
        reception_row.addWidget(QLabel(tr("Channel:")))
        reception_row.addWidget(self._channel_combo)
        reception_row.addWidget(self._scan_button)
        reception_row.addWidget(QLabel(tr("Frequency:")))
        reception_row.addWidget(self._frequency_spin)
        reception_row.addWidget(self._audio_checkbox)
        reception_row.addWidget(self._volume_slider)
        reception_row.addWidget(self._start_stop_button)
        reception_row.addStretch()

        receiver_row = QHBoxLayout()
        receiver_row.addWidget(QLabel(tr("Device:")))
        receiver_row.addWidget(self._device_combo)
        receiver_row.addWidget(self._rtlsdr_controls)
        receiver_row.addWidget(self._rspduo_controls)
        receiver_row.addWidget(self._dc_checkbox)
        receiver_row.addWidget(self._iq_checkbox)
        receiver_row.addStretch()

        layout = QVBoxLayout()
        layout.addLayout(reception_row)
        layout.addLayout(receiver_row)
        return layout

    def _build_status_labels(self) -> QVBoxLayout:
        """The status lines beneath the controls."""
        self._status_label = QLabel(tr("Ready."))
        # Device errors run to two lines (the RSPduo's ask the user to start a service or
        # close another program); unwrapped they would widen the whole window.
        self._status_label.setWordWrap(True)
        self._freq_offset_label = QLabel(tr("Frequency offset: -"))

        self._sfn_label = QLabel(tr("SFN share: -"))
        self._sfn_label.setToolTip(
            tr(
                "Share of the received power that does not come via the strongest path but "
                "from further transmitters of the single frequency network (SFN) or via "
                "reflections. Measured on the impulse response in the Signal tab, per frame "
                "and unsmoothed; the noise is taken out. Paths at most {width:.1f} µs from the "
                "strongest count as part of it, the bandwidth resolves no finer.\n\n"
                "Outside the guard interval: the part arriving before the path the receiver "
                "synchronises to, or more than one guard interval after it. It mixes "
                "neighbouring symbols into the FFT window and acts like noise."
            ).format(width=MAIN_PATH_HALF_WIDTH_SECONDS * 1e6)
        )

        self._clipping_label = QLabel(tr("Overload: -"))
        self._clipping_label.setToolTip(
            tr(
                "Share of the IQ samples at full scale of the ADC. If the value is too high, "
                "the tuner input is overloaded - raising the gain further then has no effect "
                "any more, only lowering it does. The RSPduo also reports an overload of its "
                "A/D converter itself."
            )
        )

        self._load_label = QLabel(tr("Load: -"))
        self._load_label.setToolTip(
            tr(
                "On the left the share of real time the signal processing needs per second "
                "of signal - the decisive value: above 100% the decoding falls behind the "
                "reception, and the audio drops out. “Waiting” is the part of it in which its "
                "thread did not compute but waited for the GIL (which the surface holds while "
                "drawing) or for a core. On the right the processor load of the whole program "
                "across all threads (above 100% = more than one core)."
            )
        )

        self._iq_correction_label = QLabel(tr("Receiver errors: -"))
        self._iq_correction_label.setToolTip(
            tr(
                "Errors of the receiver, estimated continuously, that DC and IQ correction "
                "even out: magnitude of the DC offset, deviation of the I/Q phase from 90° "
                "and gain difference between the I and Q branches."
            )
        )

        layout = QVBoxLayout()
        for label in (
            self._status_label,
            self._freq_offset_label,
            self._sfn_label,
            self._clipping_label,
            self._load_label,
            self._iq_correction_label,
        ):
            label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            layout.addWidget(label)
        return layout

    def _build_tabs(self) -> QTabWidget:
        """The plot area: one widget per tab."""
        controller = self._controller
        self._signal_tab = SignalTab()
        self._ofdm_tab = OfdmTab(MODE_I, SAMPLE_RATE, SYMBOL_WINDOW_EXCERPT_SYMBOLS)
        self._service_tab = ServiceTab()
        self._service_tab.set_announcement_switching(controller.announcement_switching)
        self._service_tab.announcement_switching_toggled.connect(
            controller.set_announcement_switching
        )
        self._programme_tab = ProgrammeTab()
        self._programme_tab.alert_changed.connect(self._on_alert_changed)
        self._journaline_tab = JournalineTab()
        self._journaline_tab.object_count_changed.connect(self._on_journaline_count_changed)
        self._data_tab = DataTab()
        self._data_tab.component_toggled.connect(controller.set_data_component_enabled)
        # The buttons of the other tabs add their data service to those being decoded.
        self._epg_tab = EpgTab(controller.guide_cache, controller.ensemble_cache)
        self._epg_tab.component_requested.connect(self._on_data_component_requested)
        # The play button in the programme guide: tunes to the channel of the service,
        # selects it and starts reception, as if both had been chosen by hand.
        self._epg_tab.service_requested.connect(controller.play_service)
        self._epg_tab.logos_changed.connect(self._on_guide_logos_changed)
        # What GEPOS carries, for the three tabs under "GNSS".
        self._gnss_model = GnssModel.for_controller(controller, self)
        self._gnss_tab = GnssTab(self._gnss_model, controller.settings, controller.ntrip_caster)
        self._gnss_tab.component_requested.connect(self._on_data_component_requested)
        # The position is entered in the "Satellites" tab (or set there by a receiver on
        # the NTRIP caster) and kept by the controller.
        self._gnss_tab.observer_changed.connect(controller.set_observer)
        controller.observer_changed.connect(self._gnss_tab.set_observer)
        self._atmosphere_tab = AtmosphereTab(self._gnss_model)
        self._atmosphere_tab.component_requested.connect(self._on_data_component_requested)
        self._corrections_tab = CorrectionsTab(self._gnss_model)
        self._corrections_tab.component_requested.connect(self._on_data_component_requested)
        # The traffic messages, sorted by their distance from the same position.
        self._traffic_model = TrafficModel.for_controller(controller, self)
        self._traffic_tab = TrafficTab(self._traffic_model)
        self._traffic_tab.component_requested.connect(self._on_data_component_requested)

        # Everything GEPOS carries, under one tab: first the stream and the sky, then
        # what the SSRZ corrections say.
        self._gnss_tabs = QTabWidget()
        self._satellites_tab_index = self._gnss_tabs.addTab(self._gnss_tab, tr("Satellites"))
        self._atmosphere_tab_index = self._gnss_tabs.addTab(self._atmosphere_tab, tr("Atmosphere"))
        self._corrections_tab_index = self._gnss_tabs.addTab(
            self._corrections_tab, tr("Corrections")
        )
        self._gnss_tabs.currentChanged.connect(self._render_plots)

        # In the order of decoding: the data services belong to the ensemble and are
        # decoded before (and independently of) the selected service.
        tabs = QTabWidget()
        self._signal_tab_index = tabs.addTab(self._signal_tab, tr("Signal"))
        self._ofdm_tab_index = tabs.addTab(self._ofdm_tab, "OFDM")
        self._data_tab_index = tabs.addTab(self._data_tab, tr("Data services"))
        self._epg_tab_index = tabs.addTab(self._epg_tab, tr("Programme guide"))
        self._gnss_tab_index = tabs.addTab(self._gnss_tabs, "GNSS")
        self._traffic_tab_index = tabs.addTab(self._traffic_tab, tr("Traffic"))
        self._service_tab_index = tabs.addTab(self._service_tab, tr("Station"))
        # Translators: the tab with Dynamic Label and slideshow of the station heard.
        self._programme_tab_index = tabs.addTab(self._programme_tab, tr_context("tab", "Programme"))
        self._journaline_tab_index = tabs.addTab(self._journaline_tab, "Journaline")
        tabs.currentChanged.connect(self._render_plots)
        return tabs

    def _build_side_panel(self) -> QVBoxLayout:
        """The side panel: ensemble, service list, transmitters and decoder status."""
        self._ensemble_label = QLabel(tr("Ensemble: {label}").format(label="-"))
        self._service_source_label = QLabel()
        self._service_source_label.setWordWrap(True)
        self._service_source_label.setToolTip(
            tr(
                "Where the station list comes from. During reception it always shows what the "
                "FIC delivers at the moment. Without reception it shows the stored list of the "
                "selected channel; a click on a station notes it down, so that it is decoded "
                "right after the start. That also holds across a restart of the program."
            )
        )
        self._service_list = QListWidget()
        self._service_list.setMinimumWidth(220)
        self._service_list.setIconSize(LOGO_SIZE)
        self._service_list.setItemDelegate(ServiceListDelegate(self._service_list))
        self._service_list.itemClicked.connect(self._on_service_clicked)

        self._update_services_button = QPushButton(tr("Update station list"))
        self._update_services_button.setToolTip(
            tr(
                "Stores the received station list with the ensemble name for this channel - "
                "like a scan, only for this one channel and with the data that reception "
                "delivers anyway. The channel then appears green in the channel selection."
            )
        )
        self._update_services_button.clicked.connect(self._controller.store_live_services)

        self._tii_label = QLabel(tr("Transmitters (TII): -"))
        self._tii_label.setWordWrap(True)
        self._tii_table = QTableWidget(0, 3)
        self._tii_table.setHorizontalHeaderLabels(
            # Translators: the columns of the transmitter list: main and sub identifier
            # of a transmitter (TII), its level, its share of the power.
            [tr_context("transmitter identification", "ID"), tr("Level"), tr("Share")]
        )
        self._tii_table.verticalHeader().setVisible(False)
        self._tii_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._tii_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tii_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._tii_table.setMaximumHeight(140)
        tii_tooltip = tr(
            "The transmitters of the single frequency network, detected by their "
            "identification in the null symbol (TII, ETSI EN 300 401, clause 14.8). ID: main "
            "ID-sub ID. Level relative to the strongest transmitter, share of the power of all "
            "transmitters detected.\n\n"
            "Unlike the other displays averaged, over the last {frames} null symbols: each "
            "transmitter occupies only 32 of 1536 carriers there, and only every second null "
            "symbol carries the identification. Transmitters below the detection limit stay "
            "undetected, even if they can be seen as a peak in the impulse response."
        ).format(frames=TII_AVERAGED_FRAMES)
        self._tii_label.setToolTip(tii_tooltip)
        self._tii_table.setToolTip(tii_tooltip)

        # The channel of the selected service, and in brackets the others carrying it.
        self._service_channel_label = QLabel()
        self._service_channel_label.setToolTip(
            tr(
                "The channel of the selected station. In brackets further channels on which, "
                "according to its ensemble, it runs as well (FIG 0/24, ETSI EN 300 401, clause "
                "8.1.10). Without a question mark the ensemble there has been received here "
                "(channel list); with a question mark it is only announced there (FIG 0/21) "
                "and may apply to another region. Ensembles without a known channel are left "
                "out."
            )
        )
        self._service_channel_label.hide()

        self._subchannel_info_label = QLabel(tr("No station selected."))
        self._subchannel_info_label.setWordWrap(True)

        self._frame_status_label = QLabel(tr("Audio frames: -"))
        self._frame_status_label.setWordWrap(True)
        self._frame_status_label.setToolTip(
            tr(
                "State of the frame decoding for the selected station.\n\n"
                "DAB+: superframes with Reed-Solomon error correction, Fire code "
                "synchronisation and checksums of the audio units.\n\n"
                "DAB: MP2 frames, found by sync word and header CRC. There is no error "
                "correction beyond the Viterbi decoder; frames with a wrong header CRC are "
                "muted."
            )
        )

        self._audio_status_label = QLabel(tr("Audio: -"))
        self._audio_status_label.setWordWrap(True)
        self._audio_status_label.setToolTip(
            tr(
                "State of the audio decoding: format according to the superframe or MP2 "
                "header, state of the playback and fill level of the output buffer. The buffer "
                "bridges the audio frames coming from the signal processing in bursts."
            )
        )

        layout = QVBoxLayout()
        layout.addWidget(self._ensemble_label)
        layout.addWidget(self._service_source_label)
        layout.addWidget(self._service_list)
        layout.addWidget(self._update_services_button)
        layout.addWidget(self._tii_label)
        layout.addWidget(self._tii_table)
        layout.addWidget(self._service_channel_label)
        layout.addWidget(self._subchannel_info_label)
        layout.addWidget(self._frame_status_label)
        layout.addWidget(self._audio_status_label)
        return layout

    def _connect_controller(self) -> None:
        """Wires the results of the signal processing and of the audio output to the
        widgets, and the controller's signals to the controls and the side panel."""
        controller = self._controller
        dsp = controller.dsp_worker
        dsp.spectrum_ready.connect(self._signal_tab.update_spectrum)
        dsp.power_envelope_ready.connect(self._signal_tab.update_power_envelope)
        dsp.constellation_ready.connect(self._signal_tab.update_constellation)
        dsp.impulse_response_ready.connect(self._signal_tab.update_impulse_response)
        dsp.sfn_share_ready.connect(self._on_sfn_share_ready)
        dsp.tii_ready.connect(self._on_tii_ready)
        dsp.symbol_windows_ready.connect(self._ofdm_tab.update)
        dsp.frequency_offset_ready.connect(self._on_frequency_offset_ready)
        dsp.clipping_ready.connect(self._on_clipping_ready)
        dsp.iq_correction_ready.connect(self._on_iq_correction_ready)
        dsp.dynamic_label_ready.connect(self._programme_tab.show_dynamic_label)
        dsp.slide_ready.connect(self._programme_tab.add_mot_object)
        dsp.journaline_object_ready.connect(self._journaline_tab.add_object)
        dsp.journaline_parameters_ready.connect(self._on_journaline_parameters_ready)
        dsp.data_services_updated.connect(self._data_tab.update_components)
        dsp.data_services_updated.connect(self._epg_tab.update_components)
        dsp.data_decoding_ready.connect(self._data_tab.set_decoding)
        dsp.data_decoding_ready.connect(self._epg_tab.set_data_decoding)
        dsp.packet_fec_frame_ready.connect(self._data_tab.add_fec_frame)
        dsp.packet_statistics_ready.connect(self._data_tab.update_statistics)
        dsp.mot_carousel_ready.connect(self._data_tab.update_carousel)
        dsp.mot_carousel_ready.connect(self._epg_tab.update_carousel)
        dsp.mot_directory_ready.connect(self._epg_tab.set_directory)
        dsp.data_object_ready.connect(self._epg_tab.add_object)
        # Through a method of the programme tab rather than directly to the clock: the
        # clock is then set in the GUI thread, where it is also read.
        dsp.date_time_ready.connect(self._programme_tab.synchronise_clock)
        dsp.subchannel_info_ready.connect(self._on_subchannel_info_ready)
        dsp.msc_bits_ready.connect(self._service_tab.update_msc_bits)
        dsp.audio_bits_ready.connect(self._service_tab.update_audio_bits)
        dsp.superframe_ready.connect(self._on_superframe_ready)
        dsp.mp2_frame_ready.connect(self._on_mp2_frame_ready)

        audio = controller.audio_player
        audio.format_changed.connect(self._on_audio_format_changed)
        audio.status_changed.connect(self._on_audio_status_changed)
        audio.buffer_changed.connect(self._on_audio_buffer_changed)
        audio.level_ready.connect(self._service_tab.update_audio_level)

        controller.state_changed.connect(self._on_state_changed)
        controller.status_changed.connect(self._status_label.setText)
        controller.tuning_changed.connect(self._on_tuning_changed)
        controller.channel_stored.connect(self._refresh_channel_item)
        controller.channel_list_save_failed.connect(self._on_channel_list_save_failed)
        controller.service_list_changed.connect(self._on_service_list_changed)
        controller.service_source_changed.connect(self._on_service_source_changed)
        controller.selected_service_changed.connect(self._on_selected_service_changed)
        controller.other_ensembles_changed.connect(self._show_service_channel)
        controller.stored_list_shown.connect(self._on_stored_list_shown)
        controller.live_reset.connect(self._on_live_reset)
        controller.service_reset.connect(self._on_service_reset)
        controller.reception_stopped.connect(self._on_reception_stopped)
        controller.ensemble_received.connect(self._epg_tab.set_ensemble)
        controller.frequency_information_received.connect(self._epg_tab.set_frequency_information)
        controller.guide_stored.connect(self._epg_tab.note_stored)
        controller.guide_store_failed.connect(self._epg_tab.show_store_error)
        controller.announcement_changed.connect(self._on_announcement_changed)
        controller.announcement_visit_changed.connect(self._on_announcement_visit_changed)
        controller.load_changed.connect(self._on_load_changed)
        controller.scan_spectrum_ready.connect(self._signal_tab.update_spectrum)

    def _start_timers(self) -> None:
        self._plot_timer = QTimer(self)
        self._plot_timer.setInterval(PLOT_REFRESH_INTERVAL_MS)
        self._plot_timer.timeout.connect(self._render_plots)
        self._plot_timer.start()

    # --- Controls ---------------------------------------------------------------------

    def _channel_index(self, channel: str | None) -> int:
        """Index of a channel in the selection; None is the manual frequency."""
        if channel is None:
            return self._channel_combo.findText(tr(CHANNEL_MANUAL))
        return self._channel_combo.findData(channel)

    def _on_channel_changed(self, _index: int) -> None:
        self._controller.select_channel(self._channel_combo.currentData())

    def _on_frequency_changed(self, value: float) -> None:
        # Only a manually entered frequency: the field is disabled while a channel is
        # selected, and the controller's own changes arrive with signals blocked.
        self._controller.set_manual_frequency(value)

    def _on_tuning_changed(self, channel: str | None, frequency_mhz: float) -> None:
        """Follows the tuning, also where it does not come from these controls (the play
        button of the programme guide)."""
        with QSignalBlocker(self._channel_combo), QSignalBlocker(self._frequency_spin):
            self._channel_combo.setCurrentIndex(self._channel_index(channel))
            self._frequency_spin.setValue(frequency_mhz)
        self._frequency_spin.setEnabled(self._channel_combo.isEnabled() and channel is None)

    def _on_device_changed(self, _index: int = 0) -> None:
        rtlsdr = self._device_combo.currentData() == DEVICE_RTLSDR
        self._rtlsdr_controls.setVisible(rtlsdr)
        self._rspduo_controls.setVisible(not rtlsdr)

    def _on_agc_toggled(self, checked: bool) -> None:
        self._gain_spin.setEnabled(not checked)

    def _on_device_settings_changed(self) -> None:
        self._controller.set_device_settings(
            DeviceSettings(
                device=self._device_combo.currentData(),
                gain_db=self._gain_spin.value(),
                agc=self._agc_checkbox.isChecked(),
                ppm=self._ppm_spin.value(),
                lna_state=self._lna_combo.currentData(),
                if_gain_reduction_db=self._if_gain_reduction_spin.value(),
                rspduo_ppm=self._rspduo_ppm_spin.value(),
            )
        )

    def _show_volume(self) -> None:
        # The slider follows when another output brings its own volume.
        with QSignalBlocker(self._volume_slider):
            self._volume_slider.setValue(self._controller.volume_percent)
        controller = self._controller
        self._volume_slider.setToolTip(
            tr("Volume: {volume}").format(
                volume=volume_text(controller.volume_db, controller.volume_device_percent)
            )
        )

    def _on_start_stop_clicked(self) -> None:
        if self._controller.is_receiving:
            self._controller.stop_reception()
        else:
            self._controller.start_reception()

    def _on_scan_clicked(self) -> None:
        if self._controller.state is ReceptionState.SCANNING:
            self._controller.cancel_scan()
            self._scan_button.setEnabled(False)
        else:
            self._controller.start_scan()

    def _on_state_changed(self, state: ReceptionState) -> None:
        """Offers what the state allows: everything while idle, during reception only
        channel, frequency and Stop, during a scan only cancelling it."""
        self._set_controls_enabled(state is ReceptionState.IDLE)
        receiving = state is ReceptionState.RECEIVING
        if receiving:
            self._set_tuning_enabled(True)
            self._start_stop_button.setEnabled(True)
        self._start_stop_button.setText(tr("Stop") if receiving else tr("Start"))
        scanning = state is ReceptionState.SCANNING
        self._scan_button.setText(tr("Cancel scan") if scanning else tr("Scan"))
        if scanning:
            self._scan_button.setEnabled(True)

    def _set_controls_enabled(self, enabled: bool) -> None:
        self._set_tuning_enabled(enabled)
        self._scan_button.setEnabled(enabled)
        self._device_combo.setEnabled(enabled)
        self._agc_checkbox.setEnabled(enabled)
        self._gain_spin.setEnabled(enabled and not self._agc_checkbox.isChecked())
        self._ppm_spin.setEnabled(enabled)
        self._lna_combo.setEnabled(enabled)
        self._if_gain_reduction_spin.setEnabled(enabled)
        self._rspduo_ppm_spin.setEnabled(enabled)
        self._start_stop_button.setEnabled(enabled)

    def _set_tuning_enabled(self, enabled: bool) -> None:
        """Channel and frequency: changeable before and during reception, only not
        while connecting or scanning."""
        self._channel_combo.setEnabled(enabled)
        self._frequency_spin.setEnabled(enabled and self._controller.channel is None)

    def _refresh_channel_item(self, channel: str) -> None:
        """Labels a channel in the selection: green and with its ensemble's name if one
        is stored, otherwise with the channel name only."""
        index = self._channel_combo.findData(channel)
        info = self._controller.ensemble_cache.get(channel)
        frequency = f"{BAND_III_CHANNELS[channel]:.3f} MHz"
        if info is None:
            self._channel_combo.setItemText(index, channel)
            self._channel_combo.setItemData(index, None, Qt.ItemDataRole.BackgroundRole)
            self._channel_combo.setItemData(index, None, Qt.ItemDataRole.ForegroundRole)
            self._channel_combo.setItemData(
                index,
                tr("{frequency}, no ensemble stored").format(frequency=frequency),
                Qt.ItemDataRole.ToolTipRole,
            )
            return

        label = info.label or tr("Ensemble without a name")
        self._channel_combo.setItemText(index, f"{channel} - {label}")
        self._channel_combo.setItemData(
            index, QBrush(QColor(*RECEIVABLE_CHANNEL_BACKGROUND)), Qt.ItemDataRole.BackgroundRole
        )
        self._channel_combo.setItemData(
            index, QBrush(QColor(*RECEIVABLE_CHANNEL_FOREGROUND)), Qt.ItemDataRole.ForegroundRole
        )
        stored = describe_stored(info)
        if info.fib_success is not None:
            stored = tr("{stored}, {share:.0f}% of the FIBs without errors").format(
                stored=stored, share=info.fib_success * 100
            )
        count = len(info.services)
        tooltip = tr_plural(
            "{frequency}: {label}, {count} station\n{stored}",
            "{frequency}: {label}, {count} stations\n{stored}",
            count,
        ).format(frequency=frequency, label=label, count=count, stored=stored)
        self._channel_combo.setItemData(index, tooltip, Qt.ItemDataRole.ToolTipRole)

    def _on_channel_list_save_failed(self, reason: str) -> None:
        QMessageBox.warning(
            self,
            tr("Channel list"),
            tr("The channel list could not be saved:\n{reason}").format(reason=reason),
        )

    # --- Reception --------------------------------------------------------------------

    def _on_live_reset(self) -> None:
        """Clears everything that shows results of the previously received frequency;
        what belongs to the selected service goes with `_on_service_reset`."""
        self._freq_offset_label.setText(tr("Frequency offset: -"))
        self._sfn_label.setText(tr("SFN share: -"))
        self._tii_label.setText(tr("Transmitters (TII): -"))
        self._tii_table.setRowCount(0)
        self._pending_tii = None
        self._clipping_label.setText(tr("Overload: -"))
        self._clipping_label.setStyleSheet("")
        self._programme_tab.reset_clock()
        # The data services belong to the ensemble, not to the selected service. The
        # programme guide starts from the stored ones.
        self._data_tab.reset()
        self._epg_tab.show_channel(self._controller.list_channel, live=True)
        self._epg_tab.set_selected_service(self._controller.selected_sid)

    def _on_service_reset(self) -> None:
        """Clears everything that belongs to the previously selected service: decoder
        status, programme data and the plots of the service tab."""
        self._frame_status_label.setText(tr("Audio frames: -"))
        self._frames_synced = 0
        self._frames_total = 0
        self._frames_crc_errors = 0
        self._programme_tab.reset()
        self._journaline_tab.reset()
        self._service_tab.clear()
        self._signal_tab.reset()

    def _on_reception_stopped(self, list_kept: bool) -> None:
        if list_kept:
            # The guide on display stays; it is stored already.
            self._epg_tab.reception_stopped()

    # --- Status line ------------------------------------------------------------------

    def _on_sfn_share_ready(self, others: float, outside_guard: float) -> None:
        self._sfn_label.setText(
            tr(
                "SFN share: {others:.1f}% of the received power, "
                "outside the guard interval {outside:.1f}%"
            ).format(others=others * 100, outside=outside_guard * 100)
        )

    def _on_frequency_offset_ready(self, offset_hz: float) -> None:
        center_freq_hz = self._controller.center_frequency_hz
        ppm = offset_hz / center_freq_hz * 1e6 if center_freq_hz else 0.0
        self._freq_offset_label.setText(
            tr("Frequency offset: {offset:+.0f} Hz ({ppm:+.2f} ppm)").format(
                offset=offset_hz, ppm=ppm
            )
        )

    def _on_clipping_ready(self, fraction: float) -> None:
        """The RSPduo's own report of an overload of its ADC is shown with the next
        clipping figure, which follows within one block."""
        if self._controller.adc_overload:
            self._clipping_label.setText(
                tr(
                    "Overload: {share:.1f}% - the RSPduo reports overload, "
                    "raise LNA state or IF gain reduction!"
                ).format(share=fraction * 100)
            )
            self._clipping_label.setStyleSheet("color: red; font-weight: bold;")
        elif fraction > CLIPPING_WARNING_THRESHOLD:
            advice = (
                tr("Reduce the gain!")
                if self._controller.device_settings.is_rtlsdr
                else tr("Raise LNA state or IF gain reduction!")
            )
            self._clipping_label.setText(
                tr("Overload: {share:.1f}% - {advice}").format(share=fraction * 100, advice=advice)
            )
            self._clipping_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self._clipping_label.setText(tr("Overload: {share:.2f}%").format(share=fraction * 100))
            self._clipping_label.setStyleSheet("")

    def _on_iq_correction_ready(self, report: IqCorrectionReport | None) -> None:
        if report is None:
            self._iq_correction_label.setText(tr("Receiver errors: correction switched off"))
            return
        parts = []
        if report.dc_offset is not None:
            parts.append(f"DC {report.dc_offset:.4f}")
        if report.quadrature_error_deg is not None:
            parts.append(
                tr("quadrature error {error:+.2f}°, gain I/Q {imbalance:+.2f} dB").format(
                    error=report.quadrature_error_deg, imbalance=report.gain_imbalance_db
                )
            )
        self._iq_correction_label.setText(
            tr("Receiver errors: {errors}").format(errors="; ".join(parts))
        )

    def _on_load_changed(self, dsp_load: float, dsp_waiting: float, process_load: float) -> None:
        self._load_label.setText(
            tr(
                "Load: signal processing {dsp:.0f}% of real time, "
                "{waiting:.0f}% of it waiting, "
                "whole program {process:.0f}% of a core"
            ).format(dsp=dsp_load * 100, waiting=dsp_waiting * 100, process=process_load * 100)
        )
        overloaded = dsp_load > LOAD_WARNING_THRESHOLD
        self._load_label.setStyleSheet("color: red; font-weight: bold;" if overloaded else "")

    # --- Plots --------------------------------------------------------------------------

    def _render_plots(self) -> None:
        """Redraws the expensive plots at the shared rate.

        See `PLOT_REFRESH_INTERVAL_MS`: the data keep arriving without gaps, only the
        drawing is batched.

        Only what is currently visible is drawn. The constellation alone costs almost
        8 ms per drawing; spending that on a hidden tab would take compute time from
        the signal processing, which needs it for the audio in real time. On a tab
        switch the plots are redrawn immediately, so the display is never stale.
        """
        # The transmitter list sits next to the tabs and is always visible.
        self._render_tii()
        current_tab = self._tabs.currentIndex()
        if current_tab == self._signal_tab_index:
            self._signal_tab.render()
        elif current_tab == self._ofdm_tab_index:
            self._ofdm_tab.render()
        elif current_tab == self._service_tab_index:
            self._service_tab.render()
        elif current_tab == self._data_tab_index:
            self._data_tab.render()
        elif current_tab == self._gnss_tab_index:
            gnss_tab = self._gnss_tabs.currentIndex()
            if gnss_tab == self._satellites_tab_index:
                self._gnss_tab.render()
            elif gnss_tab == self._atmosphere_tab_index:
                self._atmosphere_tab.render()
            elif gnss_tab == self._corrections_tab_index:
                self._corrections_tab.render()
        elif current_tab == self._traffic_tab_index:
            self._traffic_tab.render()

    def _on_tii_ready(self, result: TiiResult) -> None:
        # Only note it; the table is filled at the shared refresh rate.
        self._pending_tii = result

    def _render_tii(self) -> None:
        result = self._pending_tii
        if result is None:
            return
        self._pending_tii = None

        if result.detection_limit_db is None:
            self._tii_label.setText(
                tr_plural(
                    "Transmitters (TII): none detected, {frames} null symbol averaged",
                    "Transmitters (TII): none detected, {frames} null symbols averaged",
                    result.frames,
                ).format(frames=result.frames)
            )
        else:
            self._tii_label.setText(
                tr_plural(
                    "Transmitters (TII), {frames} null symbol averaged, "
                    "detection limit {limit:.0f} dB:",
                    "Transmitters (TII), {frames} null symbols averaged, "
                    "detection limit {limit:.0f} dB:",
                    result.frames,
                ).format(frames=result.frames, limit=result.detection_limit_db)
            )
        self._tii_table.setRowCount(len(result.transmitters))
        for row, transmitter in enumerate(result.transmitters):
            cells = (
                f"{transmitter.main_id:02d}-{transmitter.sub_id:02d}",
                f"{transmitter.level_db:.1f} dB",
                f"{transmitter.share * 100:.1f}%",
            )
            for column, text in enumerate(cells):
                self._tii_table.setItem(row, column, QTableWidgetItem(text))

    def _on_alert_changed(self, active: bool) -> None:
        """Flags the programme tab while an emergency warning is on display, so that
        the warning is not missed from the other tabs."""
        self._tabs.setTabText(
            self._programme_tab_index,
            tr("Programme (emergency warning)") if active else tr_context("tab", "Programme"),
        )

    def _on_announcement_changed(
        self, status: AnnouncementStatus | None, source: str | None
    ) -> None:
        """Announcements of the selected service in the "Station" tab. While one is
        running, the tab title says so, so that it is not missed from the other tabs."""
        self._service_tab.show_announcements(status, source)
        self._show_announcement_in_tab_title()

    def _on_announcement_visit_changed(self, visit: AnnouncementVisit | None) -> None:
        """Reception has gone to another ensemble for an announcement, or is back."""
        self._service_tab.show_visit(visit)
        self._show_announcement_in_tab_title()

    def _show_announcement_in_tab_title(self) -> None:
        status, _source = self._service_tab.announcement
        active = self._controller.announcement_visit is not None or (
            status is not None and (status.active or status.oe_active) is not None
        )
        self._tabs.setTabText(
            self._service_tab_index, tr("Station (announcement)") if active else tr("Station")
        )

    def _on_journaline_parameters_ready(self, parameters: JournalineParameters | None) -> None:
        """FIG 0/13 names the point of entry of the selected service's Journaline."""
        if parameters is not None:
            self._journaline_tab.set_entry_object(parameters.entry_object)

    def _on_journaline_count_changed(self, count: int) -> None:
        """Shows in the tab title whether, and how far, the selected service's
        Journaline pages have arrived; most services send none."""
        self._tabs.setTabText(
            self._journaline_tab_index, f"Journaline ({count})" if count else "Journaline"
        )

    def _on_data_component_requested(self, scid: int) -> None:
        self._controller.set_data_component_enabled(scid, True)

    # --- Side panel -------------------------------------------------------------------

    def _on_service_list_changed(
        self,
        ensemble_label: str | None,
        services: list[tuple[int, str]],
        programme_types: dict[int, int],
    ) -> None:
        self._ensemble_label.setText(tr("Ensemble: {label}").format(label=ensemble_label or "-"))
        self._fill_service_list(services, programme_types)

    def _fill_service_list(
        self, services: list[tuple[int, str]], programme_types: dict[int, int]
    ) -> None:
        """Fills the service list, each service with its programme type and logo if
        known; the selected service stays highlighted."""
        self._service_list.clear()
        for sid, label in services:
            item = service_item(sid, label, programme_types.get(sid))
            self._service_list.addItem(item)
            if sid == self._controller.selected_sid:
                self._service_list.setCurrentItem(item)
        show_logos(self._service_list, self._service_logos)

    def _on_service_source_changed(self, text: str, can_store: bool) -> None:
        self._service_source_label.setText(text)
        self._update_services_button.setEnabled(can_store)
        self._show_service_channel()  # comes with every change of the list's channel

    def _show_service_channel(self) -> None:
        """The channel of the selected service, with the others carrying it in brackets;
        hidden while none is selected."""
        controller = self._controller
        channel = controller.list_channel
        if channel is None:
            text = tr("Manual frequency")
        else:
            text = tr("Channel {channel}").format(channel=channel)
        others = describe_other_channels(controller.other_ensembles)
        self._service_channel_label.setText(f"{text} {others}" if others else text)
        self._service_channel_label.setVisible(controller.selected_sid is not None)

    def _on_stored_list_shown(self, channel: str | None) -> None:
        # The programme guide likewise shows what is stored.
        self._epg_tab.show_channel(channel, live=False)
        self._epg_tab.set_selected_service(self._controller.selected_sid)

    def _on_guide_logos_changed(self, logos: dict[int, QIcon]) -> None:
        self._service_logos = logos
        show_logos(self._service_list, logos)

    def _on_service_clicked(self, item: QListWidgetItem) -> None:
        self._controller.select_service(item.data(SID_ROLE))

    def _on_selected_service_changed(self, sid: int | None) -> None:
        for row in range(self._service_list.count()):
            item = self._service_list.item(row)
            if item.data(SID_ROLE) == sid:
                self._service_list.setCurrentItem(item)
        if sid is not None:
            self._subchannel_info_label.setText(tr("Waiting for service information..."))
        self._show_service_channel()
        self._epg_tab.set_selected_service(sid)

    def _on_subchannel_info_ready(self, status: SubchannelStatus | None) -> None:
        if status is None:
            text = tr("No station selected.")
        elif status.component is None:
            text = tr("Waiting for service information...")
        elif status.subchannel is None:
            text = tr("Waiting for the subchannel configuration...")
        else:
            subchannel = status.subchannel
            kind = "DAB+" if status.component.is_dab_plus else "DAB (MP2)"
            text = tr(
                "Subchannel {id} ({kind}, {bitrate} kbit/s)\n"
                "Start: {start} CU, size: {size} CU\n"
                "Protection: {protection}"
            ).format(
                id=subchannel.subchannel_id,
                kind=kind,
                bitrate=subchannel.bitrate,
                start=subchannel.start_address,
                size=subchannel.size,
                protection=subchannel.description(),
            )
        if status is not None and status.announcement is not None:
            names = ", ".join(name.short for name in announcement_types(status.announcement.flags))
            text = tr("Switched to an announcement ({names}):\n{text}").format(
                names=names, text=text
            )
        self._subchannel_info_label.setText(text)

    def _on_superframe_ready(self, result: SuperframeResult) -> None:
        self._service_tab.update_superframe(result)
        self._frames_total += 1
        if not result.sync_ok:
            self._frame_status_label.setText(
                tr("Superframe: no sync ({synced}/{total})").format(
                    synced=self._frames_synced, total=self._frames_total
                )
            )
            return

        self._frames_synced += 1
        valid_aus = len(result.audio_units)
        # Spell out the sizes of the individual audio units: they make tangible what
        # an audio unit is (40 ms of audio) and that its length varies with the music
        # because AAC encodes with a variable bit rate.
        sizes = ", ".join(str(end - start) for start, end, _ in result.au_segments)
        milliseconds = SUPERFRAME_SECONDS * 1000 / max(result.num_audio_units, 1)
        corrections = (
            tr("RS corrections: {count}, uncorrectable!")
            if result.rs_uncorrectable_blocks
            else tr("RS corrections: {count}")
        ).format(count=result.rs_corrections)
        self._frame_status_label.setText(
            tr(
                "Superframe: sync OK ({synced}/{total})\n"
                "{corrections}\n"
                "Audio units: {valid}/{units} valid, {milliseconds:.0f} ms of audio each\n"
                "Sizes: {sizes} bytes"
            ).format(
                synced=self._frames_synced,
                total=self._frames_total,
                corrections=corrections,
                valid=valid_aus,
                units=result.num_audio_units,
                milliseconds=milliseconds,
                sizes=sizes,
            )
        )

    def _on_mp2_frame_ready(self, result: Mp2FrameResult) -> None:
        self._service_tab.update_mp2_frame(result)
        self._frames_total += 1
        if result.sync_ok:
            self._frames_synced += 1
            self._frames_crc_errors += not result.crc_ok
        # Updated at most every tenth frame (a quarter of a second at 48 kHz): at 42
        # frames per second the text would otherwise flicker illegibly.
        if self._frames_total % 10 and result.sync_ok:
            return

        audio_format = result.audio_format
        state = tr("sync OK") if result.sync_ok else tr("no sync")
        self._frame_status_label.setText(
            # Translators: {state} is "sync OK" or "no sync".
            tr(
                "MP2 frames: {state} ({synced}/{total})\n"
                "Header CRC errors: {errors}\n"
                "{length} bytes per frame, {milliseconds:.0f} ms of audio\n"
                "X-PAD: {x_pad} bytes"
            ).format(
                state=state,
                synced=self._frames_synced,
                total=self._frames_total,
                errors=self._frames_crc_errors,
                length=audio_format.frame_length,
                milliseconds=audio_format.duration_seconds * 1000,
                x_pad=result.x_pad_length,
            )
        )

    def _on_audio_format_changed(self, description: str | None) -> None:
        self._audio_format_text = description
        self._update_audio_status()

    def _on_audio_status_changed(self, state: str) -> None:
        self._audio_state_text = state
        self._update_audio_status()

    def _on_audio_buffer_changed(self, seconds: float, _underruns: int) -> None:
        self._audio_buffer_seconds = seconds
        self._update_audio_status()

    def _update_audio_status(self) -> None:
        if self._audio_format_text is None:
            self._audio_status_label.setText(tr("Audio: -"))
            return
        self._audio_status_label.setText(
            tr("Audio: {format}\n{state}, buffer {buffer:.0f} ms").format(
                format=self._audio_format_text,
                state=self._audio_state_text,
                buffer=self._audio_buffer_seconds * 1000,
            )
        )

    # --- Language -------------------------------------------------------------------

    def _on_language_chosen(self, chosen: str) -> None:
        """Keeps the language chosen and offers the new start it takes, asked in that
        language; back to the language running, there is nothing to restart."""
        store_language(self._controller.settings, chosen)
        question = restart_question(chosen)
        if question is None:
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(self.windowTitle())
        box.setText(question.title)
        box.setInformativeText(question.detail)
        restart = box.addButton(question.restart, QMessageBox.ButtonRole.AcceptRole)
        box.addButton(question.later, QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(restart)
        box.buttonClicked.connect(
            lambda button: self.restart_requested.emit() if button is restart else None
        )
        box.finished.connect(box.deleteLater)
        # Not modal for the whole application, and not blocking: the radio plays on.
        box.open()
        self._restart_box = box

    # --- Shutdown ---------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        self._controller.shutdown()
        event.accept()
