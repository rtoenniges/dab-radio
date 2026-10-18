"""The receiver behind the window: device, tuning, scan, service selection, audio and the
stored state, without a single widget.

A surface (`ui.main_window.MainWindow`, `ui.touch.window.TouchWindow`) builds its widgets on
one `ReceiverController`: it calls the controller's methods for what the user does and
follows its signals for what changes. The plots connect to the signals of the signal
processing directly (`dsp_worker`), the audio display to those of `audio_player`; the
controller only takes those of their signals that change its own state.

Status texts are translated (`i18n`), as everything the user sees; the names of the
devices (`DEVICES`) are product names and stay as they are.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Collection
from dataclasses import dataclass, replace
from enum import Enum, auto
from pathlib import Path

import numpy as np
from PyQt6.QtCore import (
    QMetaObject,
    QObject,
    QSettings,
    QStandardPaths,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtMultimedia import QMediaDevices

from audio.bluetooth_route import BluetoothRoute, sink_address, sink_name
from audio.equalizer import EqualizerSettings, EqualizerState
from audio.outputs import AudioOutput, audio_outputs, is_bluetooth
from audio.player import AudioPlayer
from audio.system_volume import (
    VOLUME_NORM,
    SystemVolume,
    db_of_raw,
    raw_of_share,
    raw_volume,
    share_of_raw,
)
from audio.volume import (
    CEILING_RANGE_DB,
    DEFAULT_CEILING_DB,
    NEW_OUTPUT_PERCENT,
    ceiling_share,
    db_of_share,
    device_share,
    percent_of_db,
    percent_of_share,
    volume_db,
)
from dab.announcement import ALARM_CLUSTER, AnnouncementStatus
from dab.channels import BAND_III_CHANNELS, DEFAULT_CHANNEL
from dab.constants import MODE_I, SAMPLE_RATE
from dab.data_services import USER_APPLICATION_SPI, DataComponent
from dab.ensemble_cache import (
    SOURCE_LIVE,
    SOURCE_SCAN,
    EnsembleCache,
    EnsembleInfo,
    keep_service_ensembles,
)
from dab.fig import OeAnnouncement
from dab.gnss import Observer
from dab.guide_cache import GuideCache
from dab.mot_directory import MotDirectory
from dab.service_following import EnsembleLocation, locate_ensemble, other_ensembles
from dab.spi import Schedule, ServiceInformation
from dsp.pipeline import CarouselObject, DataComponentStatus, DspWorker
from dsp.signal_capture import CAPTURE_FRAMES
from i18n import exception_text, format_date, join, tr, tr_plural
from net.bluetooth import BluetoothManager, Device
from net.mqtt import MqttBridge
from net.ntrip import NtripCaster, RtcmRelay
from net.radio_state import RadioStateModel
from recording.cutter import CutRequest, Stretch
from recording.levels import LevelReader
from recording.playback import LOADING, PAUSED, PLAYING, PlaybackState, RecordingAudio
from recording.recorder import (
    DEFAULT_LABEL_DELAY_S,
    DEFAULT_RING_SIZE,
    DEFAULT_STOP_DELAY_S,
    LABEL_DELAY_RANGE_S,
    NO_DIRECTORY,
    RING_SIZE_RANGE,
    RecorderStatus,
    Recording,
    Station,
    TitleRecorder,
)
from sdr.exceptions import DeviceError
from sdr.gain import FALLBACK_GAINS_DB, query_gains
from sdr.rspduo import (
    DEFAULT_IF_GAIN_REDUCTION_DB,
    DEFAULT_LNA_STATE,
    FREQ_CORRECTION_RANGE_PPM,
    IF_GAIN_REDUCTION_RANGE_DB,
    LNA_GAIN_REDUCTION_DB,
    TUNERS,
    RspDuoSettings,
)
from sdr.rtl_tcp import DEFAULT_PORT as RTL_TCP_DEFAULT_PORT
from sdr.rtl_tcp import PORT_RANGE as RTL_TCP_PORT_RANGE
from sdr.rtl_tcp import RtlTcpSettings
from sdr.scanner import ChannelScanner
from sdr.worker import RspDuoWorker, RtlSdrWorker, RtlTcpWorker
from ui.notifications import NotificationCenter

logger = logging.getLogger(__name__)

# Name of the application; determines the data directory in which the channel list is
# stored (under Linux ~/.local/share/dab-radio/, under Windows %APPDATA%\dab-radio).
APPLICATION_NAME = "dab-radio"
# Version of the radio (semantic versioning); the same as in pyproject.toml, which
# `tests/test_version.py` checks.
APPLICATION_VERSION = "1.5.0"
ENSEMBLE_CACHE_FILENAME = "ensembles.json"
# Directory of the stored programme guides, one subdirectory per channel.
GUIDE_CACHE_DIRECTORY = "programme_guides"
# Settings that outlast a restart: the controls of the receiver, the selected service,
# the data services switched off and the listener's position.
SETTINGS_FILENAME = "settings.ini"
_SETTINGS_CHANNEL = "controls/channel"
_SETTINGS_FREQUENCY = "controls/frequency_mhz"
_SETTINGS_GAIN = "controls/gain_db"
_SETTINGS_AGC = "controls/agc"
_SETTINGS_DC_CORRECTION = "controls/dc_correction"
_SETTINGS_IQ_CORRECTION = "controls/iq_correction"
_SETTINGS_PPM = "controls/ppm"
_SETTINGS_DEVICE = "controls/device"
_SETTINGS_RSPDUO_TUNER = "controls/rspduo_tuner"
_SETTINGS_RSPDUO_LNA = "controls/rspduo_lna_state"
_SETTINGS_RSPDUO_IF_GAIN_REDUCTION = "controls/rspduo_if_gain_reduction_db"
_SETTINGS_RSPDUO_PPM = "controls/rspduo_ppm"
_SETTINGS_RTL_TCP_HOST = "controls/rtl_tcp_host"
_SETTINGS_RTL_TCP_PORT = "controls/rtl_tcp_port"
_SETTINGS_AUDIO = "controls/audio"
# The slider position on the decibel scale (`audio.volume`) and the ceiling under it.
_SETTINGS_VOLUME = "controls/volume"
_SETTINGS_VOLUME_CEILING = "controls/volume_ceiling_db"
# The output chosen ("" = the default of the system, else the id of the device) and
# the volume (slider and ceiling) remembered for every output the radio has played to,
# as JSON {id: [percent, ceiling_db]}: headphones and the loudspeaker need different ones.
_SETTINGS_AUDIO_OUTPUT = "audio/output"
_SETTINGS_OUTPUT_VOLUMES = "audio/volumes"
# How often the radio tries to give A2DP to a device before it leaves it be, until the
# device connects anew or is chosen, and how long it waits after a try that failed
# (`ReceiverController._route_bluetooth`): a device that has just connected may refuse
# the stream for a moment (some 0.2 s after connecting).
BLUETOOTH_MOVES = 3
BLUETOOTH_RETRY_MS = 1000
# The equalizer (see `ReceiverController.equalizer_state`): per station or not, the one
# of all stations and those of single stations, each as JSON of `EqualizerState.to_dict`,
# the latter by SId in hex ({"D220": {...}}).
_SETTINGS_EQUALIZER_PER_STATION = "equalizer/per_station"
_SETTINGS_EQUALIZER = "equalizer/all_stations"
_SETTINGS_STATION_EQUALIZERS = "equalizer/stations"
# Stored in place of a channel name while a manual frequency is selected.
_SETTINGS_MANUAL_CHANNEL = "manual"
# SId of the selected service; missing while none is selected.
_SETTINGS_SERVICE = "service/sid"
# Whether the audio follows a running announcement into its subchannel ("Station" tab).
_SETTINGS_ANNOUNCEMENT_SWITCHING = "service/announcement_switching"
# The stations marked as favourites, as "channel/SId" with the SId in hex ("5C/D220").
_SETTINGS_FAVOURITES = "service/favourites"
# The data components switched off by hand, as keys of `ReceiverController._data_key`.
# Every other decodable component is decoded.
_SETTINGS_DATA_SWITCHED_OFF = "data/switched_off"
# The listener's position (see `ReceiverController.observer`).
_SETTINGS_LATITUDE = "gnss/latitude"
_SETTINGS_LONGITUDE = "gnss/longitude"
_SETTINGS_HEIGHT = "gnss/height"
# Where the recordings go (see `ReceiverController.recording_directory`): a directory
# on a USB drive (`storage.volumes`), stored as its path; missing while none is chosen.
_SETTINGS_RECORDING_DIRECTORY = "recording/directory"
# How many of the last titles the ring keeps (`ReceiverController.ring_size`).
_SETTINGS_RING_SIZE = "recording/ring_titles"
# How late the dynamic label runs against the audio, seconds (`ReceiverController.label_delay_s`):
# the default for every station, and the own values of stations as JSON by SId in hex
# ({"D210": 30}).
_SETTINGS_LABEL_DELAY = "recording/label_delay_s"
_SETTINGS_STATION_LABEL_DELAYS = "recording/station_label_delays"
_SETTINGS_STOP_DELAY = "recording/stop_delay_s"
_SETTINGS_STATION_STOP_DELAYS = "recording/station_stop_delays"
_SETTINGS_RECORDING_ENABLED = "recording/enabled"

# The receivers to choose from: key in the settings -> name in the selection. The
# RSPduo receives Band III on either tuner (`DeviceSettings.rspduo_tuner`); rtl_tcp is
# an RTL-SDR stick elsewhere in the network, served by rtl_tcp (`sdr.rtl_tcp`).
DEVICE_RTLSDR = "rtlsdr"
DEVICE_RSPDUO = "rspduo"
DEVICE_RTL_TCP = "rtltcp"
DEVICES = {
    DEVICE_RTLSDR: "RTL-SDR",
    DEVICE_RSPDUO: "RSPduo",
    DEVICE_RTL_TCP: "RTL_TCP",
}
# The tuners of the RSPduo, as labelled on it.
RSPDUO_TUNERS = tuple(TUNERS)
# Up to 1.3.1 each tuner was a device of its own; one of them stored stands for the
# RSPduo with that tuner.
_RSPDUO_TUNER_DEVICES = {"rspduo1": 1, "rspduo2": 2}

# Fixed gain by default, no AGC. Both gain controls of the RTL-SDR hurt DAB: the null
# symbol every 96 ms is a gap the control runs into, so the level moves within the
# frame. Level spread over one frame and how tightly the D-QPSK points cluster
# (|mean(u^4)| of the unit points, 1 = perfect) on a strong ensemble:
#
#   tuner AGC             spread 0.151, quality 0.15, 43 % of samples clipped
#   tuner + digital AGC   spread 0.247, quality 0.24, 0.4 % clipped
#   fixed 15 dB           spread 0.013, quality 0.99, no clipping
#
# The clipping indicator alone does not reveal this: with the digital AGC on top,
# almost nothing clips while the signal is distorted all the same.
#
# The fixed gain only fixes the tuner's LNA and mixer. librtlsdr leaves the IF amplifier
# (VGA) behind them to the RTL2832U's IF AGC loop, which keeps the ADC at about -9 dBFS
# at any gain from 8.7 to 49.6 dB. That is how the library of pyrtlsdrlib (the
# librtlsdr/librtlsdr fork) behaves, and the radio's own build likewise through
# `native/rtlsdr-r82xx-if-agc.patch` (see `sdr.library`). Plain rtl-sdr 2.0.3 fixes the
# VGA at 16.3 dB instead, and then no gain suits all ensembles: at 29.7 dB a strong one
# had 25 % of its samples clipped, at 40.2 dB half of them and FIBs failed, while a weak
# one delivered no valid FIB below 25.4 dB.
#
# The gain therefore hardly moves the level but sets the noise figure of the front end,
# which decides for weak ensembles. On a weak one (MER of the D-QPSK points, share of
# CRC-valid FIBs):
#
#   gain (dB)       8.7    14.4    19.7    25.4    29.7    49.6
#   MER (dB)        3.1     3.6     4.7     5.3     5.5     5.4
#   valid FIBs      0 %    ~30 %   100 %   100 %   100 %   100 %
#
# Strong ensembles kept their MER within ±0.5 dB from 14.4 up to 49.6 dB, without
# clipping. 29.7 dB is the lowest step at which the weak one reaches its best MER; at
# 33.8 dB its IF AGC repeatably settled 4 dB too high (-5 dBFS, 1.6 % clipped).
DEFAULT_GAIN_DB = 29.7

# Frequency correction of the RTL-SDR's crystal that can be set, in ppm.
RTLSDR_PPM_RANGE = (-200, 200)
# Frequencies that can be entered by hand (MHz): what the R820T/2 tunes to.
FREQUENCY_RANGE_MHZ = (24.0, 1766.0)
# The volume slider at the first start (-18 dB, see `audio.volume`); afterwards the
# stored one.
DEFAULT_VOLUME_PERCENT = 70
# The listener's position until one is set: the geographic centre of Germany, the
# service area of GEPOS. A degree more or less shifts the satellites by about a degree.
# The height (above the ellipsoid: about 200 m above sea level plus 48 m of geoid)
# plays no part in the sky, but in the troposphere of the corrections.
DEFAULT_OBSERVER = Observer(51.1634, 10.4477, 250.0)
# What can be entered: degrees north and east, metres above the ellipsoid.
LATITUDE_RANGE = (-90.0, 90.0)
LONGITUDE_RANGE = (-180.0, 180.0)
HEIGHT_RANGE_M = (-500.0, 9000.0)

# Interval at which the compute load of the whole program is measured.
PROCESS_LOAD_INTERVAL_MS = 1000

# Compute load (fraction of real time) from which a warning is shown: above it,
# decoding falls behind reception and the audio gets dropouts.
LOAD_WARNING_THRESHOLD = 0.9

# Fraction of overdriven samples from which the receiver counts as overloaded
# (`ReceiverController.overloaded`). Calibrated empirically on real hardware
# (dsp.spectrum.detect_clipping): a clean signal is typically well below 0.2 %, a tuner
# input overdriven by too much gain typically above 10 %; the threshold sits deliberately
# with wide margins in between.
CLIPPING_WARNING_THRESHOLD = 0.01
# How long the receiver still counts as overloaded after the last block that was. The
# share is measured per block (96 ms) and, near the threshold, crosses it from block to
# block: the warning would flicker.
OVERLOAD_HOLD_MS = 2000

# Gone to another ensemble for an announcement (FIG 0/26): if FIG 0/19 there does not
# show it running within this time, it is over or not to be found, and reception goes
# back. Retuning and the FIC take some 2 s; FIG 0/19 comes ten times per second at the
# start of an announcement and once per second after.
ANNOUNCEMENT_SEARCH_MS = 10_000


class ReceptionState(Enum):
    """What the receiver is doing; decides which controls a surface offers."""

    IDLE = auto()  # neither reception nor scan: everything can be changed
    CONNECTING = auto()  # reception starts, the device is not open yet
    RECEIVING = auto()  # only channel, frequency and service can be changed
    SCANNING = auto()  # only the scan can be cancelled


@dataclass(frozen=True)
class DeviceSettings:
    """The receiver with its gain and frequency correction. They take effect at the next
    start of reception or of a scan."""

    device: str = DEVICE_RTLSDR  # a key of `DEVICES`
    # RTL-SDR, here or behind rtl_tcp
    gain_db: float = DEFAULT_GAIN_DB
    agc: bool = False
    ppm: int = 0
    # RSPduo
    rspduo_tuner: int = 1  # as labelled on it, see `RSPDUO_TUNERS`
    lna_state: int = DEFAULT_LNA_STATE
    if_gain_reduction_db: int = DEFAULT_IF_GAIN_REDUCTION_DB
    rspduo_ppm: float = 0.0
    # rtl_tcp: the server; an empty host is none set
    rtl_tcp_host: str = ""
    rtl_tcp_port: int = RTL_TCP_DEFAULT_PORT

    @property
    def is_rtlsdr(self) -> bool:
        """An RTL-SDR stick, here or behind rtl_tcp: its gain and correction apply."""
        return self.device in (DEVICE_RTLSDR, DEVICE_RTL_TCP)

    @property
    def receiver(self) -> tuple[object, ...]:
        """What receives: the device, the RSPduo with its tuner, rtl_tcp with its server.
        Another one after an error starts reception again
        (`ReceiverController.set_device_settings`)."""
        if self.device == DEVICE_RSPDUO:
            return (self.device, self.rspduo_tuner)
        if self.device == DEVICE_RTL_TCP:
            return (self.device, self.rtl_tcp_host, self.rtl_tcp_port)
        return (self.device,)

    def gain(self) -> str | float:
        """The gain for `RtlSdrWorker`, `RtlTcpWorker` and `ChannelScanner`: "auto" with
        AGC."""
        return "auto" if self.agc else self.gain_db

    def rtl_tcp(self) -> RtlTcpSettings | None:
        """Server, gain and frequency correction for rtl_tcp; None for another device."""
        if self.device != DEVICE_RTL_TCP:
            return None
        return RtlTcpSettings(
            host=self.rtl_tcp_host,
            port=self.rtl_tcp_port,
            gain=self.gain(),
            freq_correction_ppm=self.ppm,
        )

    def rspduo(self) -> RspDuoSettings | None:
        """Tuner, gain and frequency correction of the RSPduo; None for another device."""
        if self.device != DEVICE_RSPDUO:
            return None
        return RspDuoSettings(
            tuner=self.rspduo_tuner,
            lna_state=self.lna_state,
            if_gain_reduction_db=self.if_gain_reduction_db,
            freq_correction_ppm=self.rspduo_ppm,
        )


def rspduo_tuner_name(tuner: int) -> str:
    """A tuner of the RSPduo as labelled on it: "Tuner 2"."""
    return f"Tuner {tuner}"


def describe_stored(info: EnsembleInfo) -> str:
    """Where and when a stored service list was taken, e.g. "scan of 2026-10-06 14:30"."""
    moment = time.localtime(info.updated_at)
    date, clock = format_date(moment), time.strftime("%H:%M", moment)
    if info.source == SOURCE_SCAN:
        return tr("scan of {date} {time}").format(date=date, time=clock)
    return tr("taken from reception on {date} {time}").format(date=date, time=clock)


def describe_other_channels(others: tuple[EnsembleLocation, ...]) -> str:
    """The channels of `ReceiverController.other_ensembles` in brackets, to follow the
    channel of the service, e.g. "[11D, 12D?]"; empty without any. A channel only
    announced (FIG 0/21, not received here) gets a question mark; an ensemble with no
    channel known is left out."""
    channels: dict[str, bool] = {}
    for location in others:
        if location.channel is not None:
            channels.setdefault(location.channel, location.received)
    if not channels:
        return ""
    names = (channel if received else f"{channel}?" for channel, received in channels.items())
    return f"[{', '.join(names)}]"


@dataclass(frozen=True)
class AnnouncementVisit:
    """Reception gone to another ensemble for an announcement there (FIG 0/26), to come
    back once it is over."""

    announcement: OeAnnouncement  # as the ensemble left behind reported it
    channel: str  # where it is heard
    sid: int  # the service decoded there
    label: str | None  # its name in the stored list of that channel
    origin_channel: str
    origin_sid: int
    origin_label: str | None


class _StationDelay:
    """A delay in seconds against the dynamic label (`recording.recorder`): a default
    for every station and values of stations' own, kept in the settings, `key_default`
    as an int and `key_stations` as a JSON object by SId (as the equalizer keeps its
    per-station settings)."""

    def __init__(
        self, settings: QSettings, key_default: str, key_stations: str, fallback: int
    ) -> None:
        self._settings = settings
        self._key_default = key_default
        self._key_stations = key_stations
        try:
            stored = int(settings.value(key_default, fallback, type=int))
        except TypeError, ValueError:
            stored = fallback
        self.default = _clamp(stored, *LABEL_DELAY_RANGE_S)
        self.stations: dict[int, int] = self._load()

    def value(self, sid: int | None) -> int:
        """The station's own value, else the default."""
        if sid is not None and sid in self.stations:
            return self.stations[sid]
        return self.default

    def is_own(self, sid: int | None) -> bool:
        return sid is not None and sid in self.stations

    def set(self, sid: int | None, seconds: int) -> bool:
        """The value of the station (its own from then on), or the default without a
        station; whether anything changed."""
        seconds = _clamp(int(seconds), *LABEL_DELAY_RANGE_S)
        if sid is None:
            return self.set_default(seconds)
        if self.stations.get(sid) == seconds:
            return False
        self.stations[sid] = seconds
        self._store()
        return True

    def set_default(self, seconds: int) -> bool:
        seconds = _clamp(int(seconds), *LABEL_DELAY_RANGE_S)
        if seconds == self.default:
            return False
        self.default = seconds
        self._settings.setValue(self._key_default, seconds)
        return True

    def clear(self, sid: int | None) -> bool:
        """The station follows the default again; whether it had a value of its own."""
        if sid is None or sid not in self.stations:
            return False
        del self.stations[sid]
        self._store()
        return True

    def _store(self) -> None:
        stored = {f"{sid:04X}": delay for sid, delay in sorted(self.stations.items())}
        self._settings.setValue(self._key_stations, json.dumps(stored))

    def _load(self) -> dict[int, int]:
        try:
            stored = json.loads(self._settings.value(self._key_stations, "{}", type=str))
            return {
                int(key, 16): _clamp(int(value), *LABEL_DELAY_RANGE_S)
                for key, value in stored.items()
            }
        except TypeError, ValueError, AttributeError:
            return {}


class ReceiverController(QObject):
    """Runs the receiver for a surface; see the module docstring.

    Construct it, connect the surface, then call `start`: only from there on it emits,
    beginning with the stored service list of the stored channel. `shutdown` ends all
    threads and must be called before the application exits.
    """

    # Must be directed at the DspWorker and the AudioPlayer through signals rather than
    # direct method calls: they live in QThreads of their own, and a direct call would
    # write into their state while they are processing (data race).
    _service_selected = pyqtSignal(object)  # SId of the selected service, None = none
    _data_component_toggled = pyqtSignal(int, bool)  # SCId, decode it (True) or stop (False)
    _dc_removal_toggled = pyqtSignal(bool)
    _iq_correction_toggled = pyqtSignal(bool)
    _signal_capture_requested = pyqtSignal(int)  # frames to capture
    _null_snapshots_toggled = pyqtSignal(bool)
    _announcement_switching_toggled = pyqtSignal(bool)
    _audio_enabled_toggled = pyqtSignal(bool)
    _volume_changed = pyqtSignal(float)
    _equalizer_set = pyqtSignal(object)  # EqualizerSettings, None = off
    _audio_spectrum_toggled = pyqtSignal(bool)
    _audio_stop_requested = pyqtSignal()
    _audio_output_set = pyqtSignal(object)  # the id of the device (bytes), None = default
    _audio_held = pyqtSignal(bool)  # silent while the Bluetooth audio moves
    # To the recorder (`recording.recorder.TitleRecorder`), in its own thread likewise.
    _recorder_directory_set = pyqtSignal(object)  # Path, None = none
    _recorder_ring_size_set = pyqtSignal(int)
    _recorder_label_delay_set = pyqtSignal(int)
    _recorder_stop_delay_set = pyqtSignal(int)
    _recorder_enabled_set = pyqtSignal(bool)
    _recorder_station_set = pyqtSignal(object)  # recording.recorder.Station, None = none
    _recorder_reset = pyqtSignal()
    _recording_kept = pyqtSignal(str)
    _recording_deleted = pyqtSignal(str)
    # Playback of a recording: the recorder reads the file, the player plays it.
    _playback_load_requested = pyqtSignal(str)
    _playback_requested = pyqtSignal(object, float)  # RecordingAudio, position (s)
    _playback_paused = pyqtSignal()
    _playback_resumed = pyqtSignal()
    _playback_seek = pyqtSignal(float)
    _playback_stop_requested = pyqtSignal()
    _playback_stop_set = pyqtSignal(float, float)  # where it halts, where it goes back to
    # The cutter: the recorder reads a title with its neighbours and writes a cut, the
    # level reader decodes the level in its own thread.
    _stretch_load_requested = pyqtSignal(str)
    _cut_save_requested = pyqtSignal(object)  # recording.cutter.CutRequest
    _levels_requested = pyqtSignal(object)  # recording.playback.RecordingAudio
    _levels_cancelled = pyqtSignal()

    state_changed = pyqtSignal(object)  # ReceptionState
    # The status line: connecting, receiving, errors, scan. Its texts are objects, not
    # `str`, so that a translated one keeps its English for the remote interfaces
    # (`i18n.Translated`); likewise `error_occurred`.
    status_changed = pyqtSignal(object)
    # A device error ended reception or the scan (translated, for display); also reported
    # through `status_changed`, but on its own for surfaces that show only errors.
    error_occurred = pyqtSignal(object)
    # Channel (None = manual frequency) and frequency in MHz, on every change, also those
    # the surface did not cause (the play button of the programme guide).
    tuning_changed = pyqtSignal(object, float)
    # The stored ensemble of this channel changed or was removed (scan, `store_live_services`).
    channel_stored = pyqtSignal(str)
    channel_list_save_failed = pyqtSignal(str)  # the reason, from the OSError
    # The service list on display: ensemble label (None = unknown), [(SId, label)] and
    # {SId: programme type}. Either the stored one of a channel or the received one,
    # never a mix of both.
    service_list_changed = pyqtSignal(object, object, object)
    # The received service list is complete (`live_list_complete`), with the arguments of
    # `service_list_changed`; what changes after this comes through that one as usual.
    service_list_completed = pyqtSignal(object, object, object)
    # Where the service list comes from (translated, for display), and whether
    # `store_live_services` would store it.
    service_source_changed = pyqtSignal(str, bool)
    selected_service_changed = pyqtSignal(object)  # SId, None = none
    # The stored list of this channel (None = manual frequency) is on display; nothing
    # received belongs to it.
    stored_list_shown = pyqtSignal(object)
    # Reception starts afresh on `list_channel`: everything shown from the previous
    # frequency is void. At the start of reception, and on a channel change once the
    # signal processing has reached the new frequency.
    live_reset = pyqtSignal()
    # Everything shown of the previously selected service is void; with `live_reset`
    # and on every selection of a service.
    service_reset = pyqtSignal()
    # Reception has ended; True if the received service list stays on display.
    reception_stopped = pyqtSignal(bool)
    # What the FIC says about the received ensemble: label and {SId: label}, then the
    # EId and the frequencies of FIG 0/21. Only from the frequency on display.
    ensemble_received = pyqtSignal(object, object)
    frequency_information_received = pyqtSignal(object, object)
    # The other ensembles carrying the selected service (FIG 0/24), located (a tuple of
    # `dab.service_following.EnsembleLocation`, see `other_ensembles`); on every change.
    other_ensembles_changed = pyqtSignal(object)
    # dab.announcement.AnnouncementStatus of the selected service (None without
    # reception) and the label of the service the running announcement comes from.
    announcement_changed = pyqtSignal(object, object)
    # Reception has gone to another ensemble for an announcement there, as an
    # `AnnouncementVisit`; None once it is back or the user chose something else.
    announcement_visit_changed = pyqtSignal(object)
    # Switching to announcements was turned on or off (`announcement_switching`).
    announcement_switching_changed = pyqtSignal(bool)
    # Compute demand of the signal processing as a fraction of real time, the share of
    # real time its thread spent waiting in that (for the GIL above all, see
    # `DspWorker._report_load`), and the CPU load of the whole program in cores;
    # whenever any of them is measured anew.
    load_changed = pyqtSignal(float, float, float)
    # The receiver counts as overloaded, or no longer (`overloaded`).
    overloaded_changed = pyqtSignal(bool)
    scan_spectrum_ready = pyqtSignal(object, object)  # freqs (Hz), power (dB) during a scan
    # The scan moved on to this channel: its running number (from 0) and the count.
    scan_progress = pyqtSignal(str, int, int)
    # The data components of the received ensemble, which of them are decoded or which
    # are switched off changed (see `data_components`).
    data_components_changed = pyqtSignal()
    # An object of the programme guide was stored for this channel, at this time; True
    # if it may change service names or logos (an image or the SI), not only schedules.
    guide_stored = pyqtSignal(str, float, bool)
    guide_store_failed = pyqtSignal(str)  # the reason, from the OSError
    observer_changed = pyqtSignal(object)  # dab.gnss.Observer: the listener's position
    # The device or one of its settings changed (`device_settings`).
    device_settings_changed = pyqtSignal()
    favourites_changed = pyqtSignal()  # a station was marked as favourite or unmarked
    # The equalizer in effect changed: switched, set, per station or not, or (while per
    # station) another station selected.
    equalizer_changed = pyqtSignal()
    # The volume or its ceiling changed (`volume_percent`, `volume_db`).
    volume_changed = pyqtSignal()
    # The audio was muted or unmuted (`audio_enabled`).
    audio_enabled_changed = pyqtSignal(bool)
    # The outputs of the system, the one chosen or the one in use changed
    # (`audio_outputs`, `audio_output_choice`, `audio_output_in_use`).
    audio_outputs_changed = pyqtSignal()
    # The directory for the recordings was chosen or cleared (`recording_directory`).
    recording_directory_changed = pyqtSignal(object)  # Path, None = none
    # The ring of the last titles or the kept ones changed (`recordings`,
    # `kept_recordings`).
    recordings_changed = pyqtSignal()
    # What the recorder does, or why not (`recorder_status`).
    recorder_status_changed = pyqtSignal(object)  # recording.recorder.RecorderStatus
    # The label delay in effect changed: set, or another station selected (`label_delay_s`).
    label_delay_changed = pyqtSignal()
    # The stop delay in effect changed likewise (`stop_delay_s`).
    stop_delay_changed = pyqtSignal()
    # Recording switched on or off (`recording_enabled`).
    recording_enabled_changed = pyqtSignal(bool)
    # A recording plays in place of the live audio, paused, or no longer (`playback`);
    # about four times a second while one plays.
    playback_changed = pyqtSignal(object)  # recording.playback.PlaybackState, None = none
    # A recording could not be read for playback (translated, for the surface).
    playback_failed = pyqtSignal(str)
    # The cutter (`ui.cut_model`): a title read with its neighbours (`load_stretch`), or
    # its id and why not (translated); the level of a stretch as it is read
    # (`read_levels`: its id, the first unit, RMS and peak in dBFS) and its end; a cut
    # written (`save_cut`: the kept title, the id of the one it replaced or ""), or why
    # not (translated).
    stretch_loaded = pyqtSignal(object)  # recording.cutter.Stretch
    stretch_failed = pyqtSignal(str, str)
    levels_read = pyqtSignal(str, int, object, object)
    levels_finished = pyqtSignal(str)
    cut_saved = pyqtSignal(object, str)  # Recording, id replaced
    cut_failed = pyqtSignal(str)

    def __init__(
        self,
        data_directory: Path | None = None,
        parent: QObject | None = None,
        system_volume: bool = False,
        remote_control: bool = True,
    ) -> None:
        """`system_volume`: whether the radio is the one program playing and sets the
        volume of the sound system (`--system-volume`, `audio.system_volume`).
        `remote_control`: whether programs outside may read and control it, over MQTT
        (`remote_state`, `mqtt_bridge`); the touch surface only, the desktop has none."""
        super().__init__(parent)
        data_directory = data_directory or app_data_directory()
        self._settings = QSettings(
            str(data_directory / SETTINGS_FILENAME), QSettings.Format.IniFormat
        )
        self._data_directory = data_directory
        # Whether the radio sets the volume of the sound system (`_create_workers`).
        self._system_volume_enabled = system_volume
        self._remote_control = remote_control
        # The notifications from outside (`ui.notifications`), over MQTT; the surface
        # shows them.
        self._notifications = NotificationCenter(self)
        # Stored ensembles per channel (from a scan or taken over from reception).
        self._ensemble_cache = EnsembleCache(data_directory / ENSEMBLE_CACHE_FILENAME)
        self._ensemble_cache.load()
        # Stored programme guides per channel, filled automatically while the guide is
        # being received (see `ui.epg_tab.EpgTab`).
        self._guide_cache = GuideCache(data_directory / GUIDE_CACHE_DIRECTORY)
        # Passes the RTCM stream of GEPOS on to GNSS receivers; switched in the "Satellites" tab.
        self._ntrip_caster = NtripCaster()
        self._rtcm_relay = RtcmRelay(self._ntrip_caster)

        self._gain_steps_db = query_gains() or FALLBACK_GAINS_DB
        self._gain_range_db = (self._gain_steps_db[0], self._gain_steps_db[-1])
        self._device_settings = self._load_device_settings()
        self._channel, self._frequency_mhz = self._load_tuning()
        # DC and IQ correction are off by default (as in `DspWorker`): they cost compute
        # time and are mostly unnecessary.
        self._dc_correction = self._stored(_SETTINGS_DC_CORRECTION, False)
        self._iq_correction = self._stored(_SETTINGS_IQ_CORRECTION, False)
        self._announcement_switching = self._stored(_SETTINGS_ANNOUNCEMENT_SWITCHING, False)
        self._audio_enabled = self._stored(_SETTINGS_AUDIO, True)
        self._volume_ceiling_db = _clamp(
            self._stored(_SETTINGS_VOLUME_CEILING, DEFAULT_CEILING_DB), *CEILING_RANGE_DB
        )
        self._volume_percent = _clamp(
            self._stored(_SETTINGS_VOLUME, DEFAULT_VOLUME_PERCENT), 0, 100
        )
        self._audio_output_choice = str(self._stored(_SETTINGS_AUDIO_OUTPUT, ""))
        self._output_volumes = self._load_output_volumes()
        self._audio_output_in_use: AudioOutput | None = None
        # The Bluetooth devices, where a surface manages them (`attach_bluetooth`), and
        # the moving of the audio between them, where the radio decides it; the sink
        # "Automatic" plays to (that of the device given A2DP, else None for the
        # default); how often A2DP was to go to each device (`BLUETOOTH_MOVES`).
        self._bluetooth: BluetoothManager | None = None
        self._bluetooth_route: BluetoothRoute | None = None
        self._bluetooth_output: str | None = None
        self._bluetooth_moves: dict[str, int] = {}
        # The devices that held A2DP at the last look, and those a move of the radio's
        # releases (`_note_lost_a2dp`).
        self._bluetooth_a2dp: set[str] = set()
        self._bluetooth_released: set[str] = set()
        # Whether the player is held silent (`_hold_audio`); the pause before the next
        # try after one failed.
        self._audio_held_now = False
        self._bluetooth_retry = QTimer(self)
        self._bluetooth_retry.setSingleShot(True)
        self._bluetooth_retry.setInterval(BLUETOOTH_RETRY_MS)
        self._bluetooth_retry.timeout.connect(self._route_bluetooth)
        # Reports outputs appearing and going (headphones connecting) to the surface.
        self._media_devices = QMediaDevices(self)
        self._media_devices.audioOutputsChanged.connect(self.audio_outputs_changed)
        self._observer = self._load_observer()
        self._favourites = self._load_favourites()
        # Off at first (`EqualizerState`): a new feature should not change how the radio
        # sounds.
        self._equalizer_per_station = self._stored(_SETTINGS_EQUALIZER_PER_STATION, False)
        self._equalizer_all = self._load_equalizer()
        self._station_equalizers = self._load_station_equalizers()
        self._recording_directory = self._load_recording_directory()
        self._ring_size = _clamp(
            self._stored(_SETTINGS_RING_SIZE, DEFAULT_RING_SIZE), *RING_SIZE_RANGE
        )
        self._label_delay = _StationDelay(
            self._settings,
            _SETTINGS_LABEL_DELAY,
            _SETTINGS_STATION_LABEL_DELAYS,
            DEFAULT_LABEL_DELAY_S,
        )
        self._stop_delay = _StationDelay(
            self._settings,
            _SETTINGS_STOP_DELAY,
            _SETTINGS_STATION_STOP_DELAYS,
            DEFAULT_STOP_DELAY_S,
        )
        self._recording_enabled = self._stored(_SETTINGS_RECORDING_ENABLED, True)
        self._recordings: tuple[Recording, ...] = ()
        self._kept_recordings: tuple[Recording, ...] = ()
        self._recorder_status = RecorderStatus(False, tr(NO_DIRECTORY))
        # The recording playing (`play_recording`): the id the recorder reads and the
        # player plays, the title as shown (it follows a move into "Kept"), the
        # position asked for, and the state reported.
        self._playback_source_id: str | None = None
        self._playback_recording: Recording | None = None
        self._playback_position = 0.0
        self._playback: PlaybackState | None = None

        self._state = ReceptionState.IDLE
        self._worker: RtlSdrWorker | RspDuoWorker | RtlTcpWorker | None = None
        self._scanner: ChannelScanner | None = None
        self._sample_count = 0
        self._center_freq_hz = 0.0
        # Start reception again once it has ended (see `restart_reception`).
        self._restart_pending = False
        # Reception ended with an error of the device: choosing another one starts it
        # again (see `set_device_settings`).
        self._resume_on_device_change = False
        # The RSPduo reports an overload of its ADC itself (see `_on_overload_changed`).
        self._adc_overload = False
        # Overloaded in a block of the last `OVERLOAD_HOLD_MS` (`overloaded`).
        self._overloaded = False
        self._overload_hold = QTimer(self)
        self._overload_hold.setSingleShot(True)
        self._overload_hold.setInterval(OVERLOAD_HOLD_MS)
        self._overload_hold.timeout.connect(lambda: self._set_overloaded(False))
        # Number of the tuning (see `RtlSdrWorker.tuned`) whose results are awaited,
        # None once the signal processing has confirmed it. Results arriving in
        # between still come from the previous frequency.
        self._awaited_tuning: int | None = None
        self._dsp_load = 0.0
        self._dsp_waiting = 0.0
        self._process_load = 0.0
        self._scan_position = (0, 0)  # (running number, count) of the channel being scanned
        self._scan_found = 0
        self._scan_done = 0
        self._scan_failed = False

        # The service list shows either what the FIC currently delivers ("live") or,
        # while no reception is running, the stored list of the selected channel. The
        # two are never mixed.
        self._list_channel: str | None = self._channel  # None = manual frequency
        self._list_is_live = False
        self._live_list_complete = False
        self._list_services: list[tuple[int, str]] = []
        self._live_ensemble_label: str | None = None
        self._live_services: list[tuple[int, str]] = []
        self._live_programme_types: dict[int, int] = {}
        self._live_eid: int | None = None
        self._live_frequencies: dict[int, tuple[int, ...]] = {}
        self._live_service_ensembles: dict[int, tuple[int, ...]] = {}  # FIG 0/24
        self._selected_sid: int | None = None
        self._other_ensembles: tuple[EnsembleLocation, ...] = ()

        # Gone to another ensemble for an announcement (see `_follow_announcements`).
        self._visit: AnnouncementVisit | None = None
        self._visit_heard = False  # its announcement has shown up there
        # (OE Cluster Id, target SId) of the announcement gone to last, and of the one the
        # ensemble reports now. The first does not lead there again while the ensemble
        # still reports it; after coming back the ensemble first gets the time to report
        # it again (`_revisit_timer`), since its reports start afresh with the tuning.
        self._visited: tuple[int, int] | None = None
        self._reported: tuple[int, int] | None = None
        self._visit_timer = QTimer(self)
        self._visit_timer.setSingleShot(True)
        self._visit_timer.setInterval(ANNOUNCEMENT_SEARCH_MS)
        self._visit_timer.timeout.connect(self._on_visit_search_timeout)
        self._revisit_timer = QTimer(self)
        self._revisit_timer.setSingleShot(True)
        self._revisit_timer.setInterval(ANNOUNCEMENT_SEARCH_MS)
        self._revisit_timer.timeout.connect(self._on_revisit_timeout)

        # The data components of the received ensemble (FIC), the SCIds being decoded
        # and those asked for since the tuning; SCIds only mean something within one
        # ensemble. Switched off by hand (`_data_key`) is kept across tunings and
        # restarts.
        self._data_components: list[DataComponent] = []
        self._data_decoding: set[int] = set()
        self._data_requested: set[int] = set()
        self._data_switched_off: set[str] = set(self._stored(_SETTINGS_DATA_SWITCHED_OFF, []))
        # SCId of the programme guide (SPI) being decoded, whose objects are stored.
        self._decoded_spi: int | None = None

        self._create_workers()

    # --- Construction and shutdown ----------------------------------------------------

    def _create_workers(self) -> None:
        """Creates the DSP and audio workers and wires what the controller itself needs;
        their threads start with `start`."""
        self._dsp_thread = QThread()
        self._dsp_worker = DspWorker(sample_rate=SAMPLE_RATE, mode=MODE_I)
        self._dsp_worker.moveToThread(self._dsp_thread)
        dsp = self._dsp_worker
        dsp.services_updated.connect(self._on_services_updated)
        dsp.service_list_completed.connect(self._on_service_list_completed)
        dsp.frequency_information_updated.connect(self._on_frequency_information_updated)
        dsp.service_ensembles_updated.connect(self._on_service_ensembles_updated)
        dsp.load_ready.connect(self._on_load_ready)
        dsp.clipping_ready.connect(self._on_clipping_ready)
        dsp.announcement_ready.connect(self._on_announcement_ready)
        dsp.stream_restarted.connect(self._on_stream_restarted)
        dsp.data_services_updated.connect(self._on_data_services_updated)
        dsp.data_decoding_ready.connect(self._on_data_decoding_ready)
        dsp.data_object_ready.connect(self._on_data_object_ready)
        dsp.mot_directory_ready.connect(self._on_mot_directory_ready)
        dsp.data_services_updated.connect(self._rtcm_relay.set_components)
        # Directly: the frames go from the thread of the signal processing into the
        # queues of the caster, without waiting for the GUI.
        dsp.rtcm_frames_ready.connect(self._rtcm_relay.publish, Qt.ConnectionType.DirectConnection)
        self._service_selected.connect(dsp.select_service)
        self._data_component_toggled.connect(dsp.set_data_component_enabled)
        self._dc_removal_toggled.connect(dsp.set_dc_removal_enabled)
        self._iq_correction_toggled.connect(dsp.set_iq_correction_enabled)
        self._signal_capture_requested.connect(dsp.capture_signal)
        self._null_snapshots_toggled.connect(dsp.set_null_snapshots)
        self._announcement_switching_toggled.connect(dsp.set_announcement_switching_enabled)

        self._audio_thread = QThread()
        self._audio_player = AudioPlayer()
        self._audio_player.moveToThread(self._audio_thread)
        self._audio_thread.started.connect(self._audio_player.start)
        dsp.superframe_ready.connect(self._audio_player.process_superframe)
        dsp.mp2_frame_ready.connect(self._audio_player.process_mp2_frame)
        self._audio_enabled_toggled.connect(self._audio_player.set_enabled)
        self._volume_changed.connect(self._audio_player.set_volume)
        self._equalizer_set.connect(self._audio_player.set_equalizer)
        self._audio_spectrum_toggled.connect(self._audio_player.set_spectrum_enabled)
        self._audio_stop_requested.connect(self._audio_player.stop)
        self._audio_output_set.connect(self._audio_player.set_output)
        self._audio_held.connect(self._audio_player.set_held)
        self._audio_player.output_changed.connect(self._on_audio_output_in_use)
        self._playback_requested.connect(self._audio_player.play)
        self._playback_paused.connect(self._audio_player.pause)
        self._playback_resumed.connect(self._audio_player.resume)
        self._playback_seek.connect(self._audio_player.seek)
        self._playback_stop_requested.connect(self._audio_player.stop_playback)
        self._playback_stop_set.connect(self._audio_player.set_playback_stop)
        self._audio_player.playback_changed.connect(self._on_playback_changed)
        # The player (with its timer) must be torn down in its own thread, otherwise
        # Qt rightly complains on exit ("Timers cannot be stopped from another thread").
        self._audio_thread.finished.connect(self._audio_player.deleteLater)

        # The volume of the sink in the sound system (PipeWire on the Pi through pactl,
        # `audio.system_volume`), where the radio is the one program playing
        # (`system_volume`): a Bluetooth device that applies it itself (AVRCP) takes the
        # level, the player plays at 1.0, and its buttons move the slider. Every other
        # sink stands at 100 % and the player attenuates, as without pactl: PipeWire
        # applies its volume in software, and after a reboot the value it shows need not
        # be the one its node plays at. Next to other programs that 100 % would play
        # their sounds at full volume, so there the system's volume is left alone and
        # the player attenuates on every output.
        self._system_volume = SystemVolume(self, enabled=self._system_volume_enabled)
        self._system_volume.applied.connect(self._on_system_volume_applied)
        self._system_volume.changed.connect(self._on_system_volume_changed)
        self._system_volume.failed.connect(self._on_system_volume_failed)
        self._system_volume.software_volume.connect(self._on_system_volume_in_software)
        # The sink the system takes the volume for: that of the output in use; None
        # without pactl, then the player attenuates.
        self._system_volume_sink: str | None = None

        # The recorder of the last titles in a thread of its own: a write to the USB
        # stick may stall for a moment, and neither the signal processing nor the
        # surface is to wait for it. The frames reach it queued, as the player's do.
        self._recorder_thread = QThread()
        self._recorder = TitleRecorder()
        self._recorder.moveToThread(self._recorder_thread)
        dsp.superframe_ready.connect(self._recorder.process_superframe)
        dsp.mp2_frame_ready.connect(self._recorder.process_mp2_frame)
        dsp.dynamic_label_ready.connect(self._recorder.process_dynamic_label)
        self._recorder_directory_set.connect(self._recorder.set_directory)
        self._recorder_ring_size_set.connect(self._recorder.set_ring_size)
        self._recorder_label_delay_set.connect(self._recorder.set_label_delay)
        self._recorder_stop_delay_set.connect(self._recorder.set_stop_delay)
        self._recorder_enabled_set.connect(self._recorder.set_enabled)
        self._recorder_station_set.connect(self._recorder.set_station)
        self._recorder_reset.connect(self._recorder.reset)
        self._recording_kept.connect(self._recorder.keep)
        self._recording_deleted.connect(self._recorder.delete)
        self._recorder.changed.connect(self._on_recordings_changed)
        self._recorder.status_changed.connect(self._on_recorder_status)
        self._playback_load_requested.connect(self._recorder.load)
        self._recorder.loaded.connect(self._on_playback_loaded)
        self._recorder.load_failed.connect(self._on_playback_load_failed)
        self._stretch_load_requested.connect(self._recorder.load_stretch)
        self._recorder.stretch_loaded.connect(self.stretch_loaded)
        self._recorder.stretch_failed.connect(self.stretch_failed)
        self._cut_save_requested.connect(self._recorder.save_cut)
        self._recorder.cut_saved.connect(self.cut_saved)
        self._recorder.cut_failed.connect(self.cut_failed)
        self._recorder_thread.finished.connect(self._recorder.deleteLater)

        # The level of a stretch for the cutter, decoded in a thread of its own: some
        # seconds of work on the Raspberry Pi, which neither the recorder nor the
        # surface is to wait for.
        self._levels_thread = QThread()
        self._level_reader = LevelReader()
        self._level_reader.moveToThread(self._levels_thread)
        self._levels_requested.connect(self._level_reader.start)
        self._levels_cancelled.connect(self._level_reader.cancel)
        self._level_reader.read.connect(self.levels_read)
        self._level_reader.finished.connect(self.levels_finished)
        self._levels_thread.finished.connect(self._level_reader.deleteLater)

        self._load_timer = QTimer(self)
        self._load_timer.setInterval(PROCESS_LOAD_INTERVAL_MS)
        self._load_timer.timeout.connect(self._update_process_load)

        # The state for programs outside the radio and the MQTT bridge on it
        # (`net.radio_state`, `net.mqtt`): built before `start`, as a surface.
        self._remote_state: RadioStateModel | None = None
        self._mqtt: MqttBridge | None = None
        if self._remote_control:
            self._remote_state = RadioStateModel(self)
            self._mqtt = MqttBridge(self._remote_state, self._settings)

    def start(self) -> None:
        """Starts the worker threads with the stored settings, shows the stored service
        list of the stored channel and selects the service of the last session again."""
        self._audio_thread.start()
        # The workers start with their own defaults; the stored ones may differ.
        self._audio_enabled_toggled.emit(self._audio_enabled)
        self._volume_changed.emit(self._volume_gain())
        self._audio_output_set.emit(self._player_output())
        self._equalizer_set.emit(self._equalizer_in_use())
        self._dsp_thread.start()
        self._dc_removal_toggled.emit(self._dc_correction)
        self._iq_correction_toggled.emit(self._iq_correction)
        self._announcement_switching_toggled.emit(self._announcement_switching)
        self._recorder_thread.start()
        self._recorder_ring_size_set.emit(self._ring_size)
        self._recorder_label_delay_set.emit(self.label_delay_s)
        self._recorder_stop_delay_set.emit(self.stop_delay_s)
        self._recorder_enabled_set.emit(self._recording_enabled)
        self._recorder_directory_set.emit(self._recording_directory)
        self._levels_thread.start()

        self._last_cpu_time = time.process_time()
        self._last_wall_time = time.monotonic()
        self._load_timer.start()

        self._show_stored_services(self._channel)
        self._restore_selected_service()
        if self._mqtt is not None:
            self._mqtt.start()

    def shutdown(self) -> None:
        """Stops reception or scan and ends all threads; for the application's exit."""
        visit = self._visit
        if visit is not None:
            # Not to start in the middle of another ensemble's announcement next time.
            self._settings.setValue(_SETTINGS_CHANNEL, visit.origin_channel)
            self._settings.setValue(_SETTINGS_FREQUENCY, BAND_III_CHANNELS[visit.origin_channel])
            self._settings.setValue(_SETTINGS_SERVICE, visit.origin_sid)
        if self._worker is not None:
            self._worker.stop()
        if self._scanner is not None:
            self._scanner.stop()
        # End the audio output first: it holds the sound card open, and its slot would
        # otherwise still be triggered by the last superframes. Blocking, so that
        # `stop()` really runs through in the audio thread before its event loop
        # ends; a plain `emit()` would only be queued and dropped on exit. Only while
        # that thread runs: before `start` nobody would answer, and this would hang.
        if self._audio_thread.isRunning():
            QMetaObject.invokeMethod(
                self._audio_player, "stop", Qt.ConnectionType.BlockingQueuedConnection
            )
        self._audio_thread.quit()
        self._audio_thread.wait(2000)
        self._system_volume.shutdown()
        if self._bluetooth_route is not None:
            self._bluetooth_route.shutdown()
        # The title recording is kept: closing it packs the file, which takes a moment
        # on a slow stick.
        if self._recorder_thread.isRunning():
            QMetaObject.invokeMethod(
                self._recorder, "stop", Qt.ConnectionType.BlockingQueuedConnection
            )
        self._recorder_thread.quit()
        self._recorder_thread.wait(15000)
        self._levels_thread.quit()
        self._levels_thread.wait(2000)
        self._dsp_thread.quit()
        self._dsp_thread.wait()
        # Every notification gets its answer before the bridge says goodbye.
        self._notifications.close_all()
        if self._mqtt is not None:
            self._mqtt.stop()
        self._ntrip_caster.stop()
        # QSettings writes to disk from the event loop, which ends with the window.
        self._settings.sync()

    # --- What the surface reads -------------------------------------------------------

    @property
    def dsp_worker(self) -> DspWorker:
        """The signal processing; the plots connect to its signals."""
        return self._dsp_worker

    @property
    def audio_player(self) -> AudioPlayer:
        """The audio output; its signals report format, state, buffer and level."""
        return self._audio_player

    @property
    def settings(self) -> QSettings:
        return self._settings

    @property
    def ensemble_cache(self) -> EnsembleCache:
        return self._ensemble_cache

    @property
    def guide_cache(self) -> GuideCache:
        return self._guide_cache

    @property
    def ntrip_caster(self) -> NtripCaster:
        return self._ntrip_caster

    @property
    def remote_state(self) -> RadioStateModel | None:
        """The state for programs outside the radio and their commands; None without
        remote control."""
        return self._remote_state

    @property
    def mqtt_bridge(self) -> MqttBridge | None:
        """The radio on MQTT; its settings and status are read and set there. None
        without remote control."""
        return self._mqtt

    @property
    def notifications(self) -> NotificationCenter:
        """The notifications from outside the radio, for the surface to show."""
        return self._notifications

    @property
    def version(self) -> str:
        """The version of the radio (`APPLICATION_VERSION`)."""
        return APPLICATION_VERSION

    @property
    def device_name(self) -> str:
        """The name of the receiver chosen (`DEVICES`), the RSPduo with its tuner in
        brackets: "RSPduo(2)"."""
        settings = self._device_settings
        name = DEVICES.get(settings.device, settings.device)
        rspduo = settings.rspduo()
        return name if rspduo is None else f"{name}({rspduo.tuner})"

    def _device_text(self) -> str:
        """The receiver for the status line: its name, the RSPduo with its tuner, rtl_tcp
        with the server."""
        settings = self._device_settings
        rtl_tcp = settings.rtl_tcp()
        if rtl_tcp is not None and rtl_tcp.host:
            return f"{DEVICES[DEVICE_RTL_TCP]} {rtl_tcp.server}"
        rspduo = settings.rspduo()
        if rspduo is not None:
            return f"{DEVICES[DEVICE_RSPDUO]} {rspduo_tuner_name(rspduo.tuner)}"
        return self.device_name

    @property
    def state(self) -> ReceptionState:
        return self._state

    @property
    def is_receiving(self) -> bool:
        """Reception runs, is being connected or is ending."""
        return self._worker is not None

    @property
    def gain_range_db(self) -> tuple[float, float]:
        """Gain range of the connected RTL-SDR's tuner, of a typical one without a stick."""
        return self._gain_range_db

    @property
    def gain_steps_db(self) -> tuple[float, ...]:
        """The gains the connected RTL-SDR's tuner supports, ascending; those of a
        typical one without a stick. Others are rounded to one of them by the tuner."""
        return self._gain_steps_db

    @property
    def device_settings(self) -> DeviceSettings:
        return self._device_settings

    @property
    def channel(self) -> str | None:
        """The selected channel, None for a manual frequency."""
        return self._channel

    @property
    def frequency_mhz(self) -> float:
        """The frequency tuned to: that of the channel, or the manual one."""
        return self._frequency_mhz

    @property
    def center_frequency_hz(self) -> float:
        """The frequency the device is tuned to, 0 before the first reception."""
        return self._center_freq_hz

    @property
    def dc_correction(self) -> bool:
        return self._dc_correction

    @property
    def iq_correction(self) -> bool:
        return self._iq_correction

    @property
    def announcement_switching(self) -> bool:
        return self._announcement_switching

    @property
    def audio_enabled(self) -> bool:
        return self._audio_enabled

    @property
    def volume_percent(self) -> int:
        """The slider position, even in decibels (`audio.volume`)."""
        return self._volume_percent

    @property
    def volume_ceiling_db(self) -> int:
        """What 100 % of the slider gives, 0 dB or less ("Maximum volume")."""
        return self._volume_ceiling_db

    @property
    def volume_db(self) -> float | None:
        """The gain of the audio output in dB; None while the slider is at 0 %. For an
        output with a volume of its own the nominal value of its share on the cubic
        scale (`audio.volume.db_of_share`); what the device makes of it is its own."""
        if self._output_has_own_volume():
            return db_of_share(device_share(self._volume_percent, self._volume_ceiling_db))
        return volume_db(self._volume_percent, self._volume_ceiling_db)

    @property
    def volume_device_percent(self) -> int | None:
        """The volume of an output with a volume of its own (a Bluetooth device the
        system sets) as the share of its range in percent, what the surfaces show in
        place of decibels (`volume_text`); None for every other output."""
        if not self._output_has_own_volume():
            return None
        return round(100.0 * device_share(self._volume_percent, self._volume_ceiling_db))

    @property
    def audio_outputs(self) -> list[AudioOutput]:
        """The outputs of the system the radio can play to."""
        return audio_outputs()

    @property
    def audio_output_choice(self) -> str:
        """The id of the output chosen, "" for the default of the system (which takes
        headphones as soon as they connect)."""
        return self._audio_output_choice

    @property
    def audio_output_in_use(self) -> AudioOutput | None:
        """The output the audio goes (or would go) to; None without one."""
        return self._audio_output_in_use

    @property
    def selected_sid(self) -> int | None:
        return self._selected_sid

    @property
    def other_ensembles(self) -> tuple[EnsembleLocation, ...]:
        """The ensembles other than the one of the service list on display that carry the
        selected service (FIG 0/24), with the channel to hear each on: received there
        first, then announced, then unknown. From the stored list of the channel,
        during reception joined by what the ensemble sends; the stored list is left out
        once another ensemble turns out to be received there. Empty while the
        ensemble's own EId is unknown, since it may be among them."""
        return self._other_ensembles

    @property
    def announcement_visit(self) -> AnnouncementVisit | None:
        return self._visit

    @property
    def list_channel(self) -> str | None:
        """The channel the service list on display belongs to, None = manual frequency."""
        return self._list_channel

    @property
    def list_is_live(self) -> bool:
        """The service list on display is the received one rather than a stored one."""
        return self._list_is_live

    @property
    def live_list_complete(self) -> bool:
        """The received service list on display is complete: the FIC has delivered every
        service with its name, or the scan's maximum listening time has passed. Until then
        a surface may keep showing the stored list, which a partial one would reorder."""
        return self._list_is_live and self._live_list_complete

    @property
    def live_ensemble_label(self) -> str | None:
        """The label of the ensemble received (or last received), as the FIC says it."""
        return self._live_ensemble_label

    @property
    def data_directory(self) -> Path:
        """Where settings, stored ensembles and guides lie, and the road network of the
        traffic map (`dab.road_network`)."""
        return self._data_directory

    @property
    def adc_overload(self) -> bool:
        """The RSPduo reports an overload of its ADC right now."""
        return self._adc_overload

    @property
    def overloaded(self) -> bool:
        """The receiver is overdriven: in a block of the last `OVERLOAD_HOLD_MS`, more of
        the samples than `CLIPPING_WARNING_THRESHOLD` stood at full scale, or the RSPduo
        reported an overload of its ADC (`adc_overload`)."""
        return self._overloaded

    @property
    def observer(self) -> Observer:
        """The listener's position: the sky and the corrections of GEPOS are computed
        for it, and the distances to traffic messages are measured from it."""
        return self._observer

    @property
    def recording_directory(self) -> Path | None:
        """Where the recordings go: a directory on a USB drive, chosen on the storage
        page of the settings; None while none is chosen. The drive may be missing right
        now (`recording_directory_available`)."""
        return self._recording_directory

    @property
    def recording_directory_available(self) -> bool:
        """Whether the directory for the recordings is there and can be written to (its
        drive mounted)."""
        directory = self._recording_directory
        return directory is not None and directory.is_dir() and os.access(directory, os.W_OK)

    # --- Stored settings --------------------------------------------------------------

    def _stored[T](self, key: str, default: T) -> T:
        """A stored setting of the type of `default`; `default` if it is missing or
        unreadable (e.g. edited by hand)."""
        try:
            return self._settings.value(key, default, type=type(default))
        except TypeError, ValueError:
            return default

    def _load_device_settings(self) -> DeviceSettings:
        """The stored device settings, each clamped to what can be set: the stored gain
        may come from a stick with a different tuner, and the file may be edited by hand."""
        device = self._stored(_SETTINGS_DEVICE, DEVICE_RTLSDR)
        tuner = self._stored(_SETTINGS_RSPDUO_TUNER, RSPDUO_TUNERS[0])
        if device in _RSPDUO_TUNER_DEVICES:
            device, tuner = DEVICE_RSPDUO, _RSPDUO_TUNER_DEVICES[device]
        return DeviceSettings(
            device=device if device in DEVICES else DEVICE_RTLSDR,
            rspduo_tuner=tuner if tuner in RSPDUO_TUNERS else RSPDUO_TUNERS[0],
            gain_db=_clamp(self._stored(_SETTINGS_GAIN, DEFAULT_GAIN_DB), *self._gain_range_db),
            agc=self._stored(_SETTINGS_AGC, False),
            ppm=_clamp(self._stored(_SETTINGS_PPM, 0), *RTLSDR_PPM_RANGE),
            lna_state=_clamp(
                self._stored(_SETTINGS_RSPDUO_LNA, DEFAULT_LNA_STATE),
                0,
                len(LNA_GAIN_REDUCTION_DB) - 1,
            ),
            if_gain_reduction_db=_clamp(
                self._stored(_SETTINGS_RSPDUO_IF_GAIN_REDUCTION, DEFAULT_IF_GAIN_REDUCTION_DB),
                *IF_GAIN_REDUCTION_RANGE_DB,
            ),
            rspduo_ppm=_clamp(self._stored(_SETTINGS_RSPDUO_PPM, 0.0), *FREQ_CORRECTION_RANGE_PPM),
            rtl_tcp_host=self._stored(_SETTINGS_RTL_TCP_HOST, "").strip(),
            rtl_tcp_port=_clamp(
                self._stored(_SETTINGS_RTL_TCP_PORT, RTL_TCP_DEFAULT_PORT), *RTL_TCP_PORT_RANGE
            ),
        )

    def _load_observer(self) -> Observer:
        default = DEFAULT_OBSERVER
        return Observer(
            _clamp(self._stored(_SETTINGS_LATITUDE, default.latitude), *LATITUDE_RANGE),
            _clamp(self._stored(_SETTINGS_LONGITUDE, default.longitude), *LONGITUDE_RANGE),
            _clamp(self._stored(_SETTINGS_HEIGHT, default.height), *HEIGHT_RANGE_M),
        )

    def _load_tuning(self) -> tuple[str | None, float]:
        """The stored channel (the default channel if it is unknown) and frequency."""
        channel = self._stored(_SETTINGS_CHANNEL, DEFAULT_CHANNEL)
        if channel != _SETTINGS_MANUAL_CHANNEL:
            channel = channel if channel in BAND_III_CHANNELS else DEFAULT_CHANNEL
            return channel, BAND_III_CHANNELS[channel]
        frequency = self._stored(_SETTINGS_FREQUENCY, BAND_III_CHANNELS[DEFAULT_CHANNEL])
        return None, _clamp(frequency, *FREQUENCY_RANGE_MHZ)

    # Stored with every change rather than on exit, so that they survive a crash.

    def set_device_settings(self, settings: DeviceSettings) -> None:
        """Stores the device settings; they take effect at the next start. After
        reception ended with an error of the device, another device starts it again
        right away (switching back from a tuner of an RSPduo not plugged in plays on);
        the error shown belongs to the device left. Another tuner of the RSPduo or
        another rtl_tcp server counts as another device (`DeviceSettings.receiver`)."""
        resume = (
            self._resume_on_device_change
            and self._state is ReceptionState.IDLE
            and settings.receiver != self._device_settings.receiver
        )
        self._device_settings = settings
        values = {
            _SETTINGS_DEVICE: settings.device,
            _SETTINGS_GAIN: settings.gain_db,
            _SETTINGS_AGC: settings.agc,
            _SETTINGS_PPM: settings.ppm,
            _SETTINGS_RSPDUO_TUNER: settings.rspduo_tuner,
            _SETTINGS_RSPDUO_LNA: settings.lna_state,
            _SETTINGS_RSPDUO_IF_GAIN_REDUCTION: settings.if_gain_reduction_db,
            _SETTINGS_RSPDUO_PPM: settings.rspduo_ppm,
            _SETTINGS_RTL_TCP_HOST: settings.rtl_tcp_host,
            _SETTINGS_RTL_TCP_PORT: settings.rtl_tcp_port,
        }
        for key, value in values.items():
            self._settings.setValue(key, value)
        self.device_settings_changed.emit()
        if resume:
            self.start_reception()

    def set_dc_correction(self, enabled: bool) -> None:
        self._dc_correction = enabled
        self._settings.setValue(_SETTINGS_DC_CORRECTION, enabled)
        self._dc_removal_toggled.emit(enabled)

    def set_iq_correction(self, enabled: bool) -> None:
        self._iq_correction = enabled
        self._settings.setValue(_SETTINGS_IQ_CORRECTION, enabled)
        self._iq_correction_toggled.emit(enabled)

    def capture_signal(self, frames: int = CAPTURE_FRAMES) -> None:
        """Asks the signal processing to keep the next `frames` frames for the detail
        view; they come with `dsp_worker.signal_captured` once received (about 0.1 s
        a frame, after the frame synchronisation has locked)."""
        self._signal_capture_requested.emit(frames)

    def set_null_snapshots(self, enabled: bool) -> None:
        """Whether the signal processing reports the stretch around the null symbol of
        every frame (`dsp_worker.null_snapshot_ready`, about ten a second), for the
        null symbol on the radio page."""
        self._null_snapshots_toggled.emit(enabled)

    def set_announcement_switching(self, enabled: bool) -> None:
        """Whether the audio follows a running announcement into its subchannel."""
        self._announcement_switching = enabled
        self._settings.setValue(_SETTINGS_ANNOUNCEMENT_SWITCHING, enabled)
        self._announcement_switching_toggled.emit(enabled)
        self.announcement_switching_changed.emit(enabled)

    def set_audio_enabled(self, enabled: bool) -> None:
        self._audio_enabled = enabled
        self._settings.setValue(_SETTINGS_AUDIO, enabled)
        self._audio_enabled_toggled.emit(enabled)
        self.audio_enabled_changed.emit(enabled)

    def set_volume_percent(self, percent: int) -> None:
        self._volume_percent = _clamp(percent, 0, 100)
        self._settings.setValue(_SETTINGS_VOLUME, self._volume_percent)
        self._store_output_volume()
        self._apply_volume()

    def set_volume_ceiling_db(self, ceiling_db: int) -> None:
        """Lowers what the whole slider gives: 100 % becomes `ceiling_db` (0 or less),
        every position below moves by the same amount."""
        self._volume_ceiling_db = _clamp(ceiling_db, *CEILING_RANGE_DB)
        self._settings.setValue(_SETTINGS_VOLUME_CEILING, self._volume_ceiling_db)
        self._store_output_volume()
        self._apply_volume()

    def set_audio_output(self, choice: str) -> None:
        """Plays to the output with this id from now on, to the system's default with "".
        A Bluetooth device chosen that is connected over the hands-free profile only gets
        A2DP (`_route_bluetooth`)."""
        self._audio_output_choice = choice
        self._settings.setValue(_SETTINGS_AUDIO_OUTPUT, choice)
        chosen = self._bluetooth_device_of(choice)
        if chosen is not None:
            self._bluetooth_moves.pop(chosen.path, None)
            self._bluetooth_retry.stop()
        self._route_bluetooth()
        self._audio_output_set.emit(self._player_output())
        self.audio_outputs_changed.emit()

    def attach_bluetooth(
        self, bluetooth: BluetoothManager, route: BluetoothRoute | None = None
    ) -> None:
        """The Bluetooth devices of the system as a surface manages them (`net.bluetooth`;
        the touch surface on the Pi). Where the radio is the one program playing
        (`system_volume`), it decides from then on which of them plays over A2DP, which
        the adapter carries to one device at a time (`_route_bluetooth`); elsewhere
        BlueZ and the system decide."""
        self._bluetooth = bluetooth
        if not self._system_volume.available or not bluetooth.available:
            return
        self._bluetooth_route = route if route is not None else BluetoothRoute(bluetooth, self)
        self._bluetooth_route.finished.connect(self._on_bluetooth_moved)
        bluetooth.changed.connect(self._route_bluetooth)
        bluetooth.device_connected.connect(self._on_bluetooth_connected)
        self._route_bluetooth()

    def connect_bluetooth(self, device: Device) -> None:
        """Connects a paired device for the surface. One that takes audio gets A2DP, from
        the device holding it if need be: without it a loudspeaker lacking the hands-free
        profile drops again. Connected by hand it is to play: with a Bluetooth device
        chosen as output, the choice moves to it."""
        bluetooth, route = self._bluetooth, self._bluetooth_route
        if bluetooth is None:
            return
        if route is None or not device.audio:
            bluetooth.connect_device(device)
            return
        self._bluetooth_moves.pop(device.path, None)
        self._bluetooth_retry.stop()
        if sink_address(self._audio_output_choice) is not None:
            self.set_audio_output(sink_name(device.address))
        if route.busy:
            bluetooth.connect_device(device)
            return
        self._move_bluetooth(device)

    def _player_output(self) -> bytes | None:
        """What the player is to play to: the output chosen, else under "Automatic" the
        Bluetooth device the radio gave A2DP, else the default of the system (None)."""
        if self._audio_output_choice:
            return self._audio_output_choice.encode()
        return self._bluetooth_output.encode() if self._bluetooth_output else None

    def _route_bluetooth(self) -> None:
        """Which Bluetooth device plays, over A2DP: under "Automatic" the one connected
        last, with one chosen that one while it is connected; and one the player falls
        back to (the default) where none holds A2DP. That device gets A2DP where it has
        none (`audio.bluetooth_route`), at most `BLUETOOTH_MOVES` tries in a row until
        it connects anew or is chosen; one that lost A2DP by itself gets none
        (`_note_lost_a2dp`). "Automatic" plays to it explicitly, not to the default of
        the system: WirePlumber may keep a device on the hands-free profile as default
        next to one with A2DP. A try that failed is followed by the next after
        `BLUETOOTH_RETRY_MS`. The player is silent from the moment a move is asked for
        until none is in progress or due (`_hold_audio`), and is told where to play
        before it plays again."""
        bluetooth, route = self._bluetooth, self._bluetooth_route
        if bluetooth is None or route is None:
            return
        self._note_lost_a2dp(bluetooth)
        target = self._bluetooth_target()
        waiting = self._bluetooth_retry.isActive()
        if target is not None and not target.a2dp and not route.busy and not waiting:
            moves = self._bluetooth_moves.get(target.path, 0)
            if moves < BLUETOOTH_MOVES:
                self._bluetooth_moves[target.path] = moves + 1
                self._move_bluetooth(target)
        if route.busy or waiting:
            return
        output = None
        if not self._audio_output_choice and target is not None and target.a2dp:
            output = sink_name(target.address)
        if output != self._bluetooth_output:
            self._bluetooth_output = output
            self._audio_output_set.emit(self._player_output())
        self._hold_audio(False)

    def _move_bluetooth(self, device: Device) -> None:
        """Moves A2DP to `device`, the player silent; the device holding it is released
        by the radio, not losing it by itself (`_note_lost_a2dp`)."""
        assert self._bluetooth is not None and self._bluetooth_route is not None
        holder = self._bluetooth.a2dp_device
        released = holder is not None and holder.path != device.path
        self._bluetooth_released = {holder.path} if released else set()
        self._hold_audio(True)
        self._bluetooth_route.move_to(device)

    def _note_lost_a2dp(self, bluetooth: BluetoothManager) -> None:
        """A device still connected that lost A2DP without a move of the radio's is not
        given it again until it connects anew or is chosen: it is being switched off
        (BlueZ ends the stream before it reports the device gone, and setting the stream
        up again fails and shows the failure), or a phone took it (a multipoint box),
        from which the radio must not take it back. Which device a move releases is
        noted when the move is asked for: the route follows BlueZ's signals before the
        controller and may be through before the controller sees the stream gone."""
        holding = {device.path for device in bluetooth.devices.values() if device.a2dp}
        for path in self._bluetooth_a2dp - holding:
            if path in self._bluetooth_released:
                self._bluetooth_released.discard(path)
                continue
            device = bluetooth.device(path)
            if device is not None and device.connected:
                logger.info("%s lost its audio stream by itself; left to it", device.name)
                self._bluetooth_moves[path] = BLUETOOTH_MOVES
        self._bluetooth_released &= holding
        self._bluetooth_a2dp = holding

    def _hold_audio(self, held: bool) -> None:
        """Keeps the player from the sound card while the Bluetooth audio moves
        (`AudioPlayer.set_held`)."""
        if held != self._audio_held_now:
            self._audio_held_now = held
            self._audio_held.emit(held)

    def _bluetooth_target(self) -> Device | None:
        """The Bluetooth device to play over A2DP (`_route_bluetooth`), None for none."""
        bluetooth = self._bluetooth
        assert bluetooth is not None
        if self._audio_output_choice:
            chosen = self._bluetooth_device_of(self._audio_output_choice)
            if chosen is not None and chosen.connected and chosen.audio:
                return chosen
        else:
            # The one connected last, unless it would not take the stream: then the one
            # before, whose stream went to it, gets it back.
            for device in bluetooth.audio_devices:
                if device.a2dp or self._bluetooth_moves.get(device.path, 0) < BLUETOOTH_MOVES:
                    return device
        output = self._audio_output_in_use
        fallback = None if output is None else self._bluetooth_device_of(output.id)
        if (
            fallback is not None
            and fallback.connected
            and fallback.audio
            and bluetooth.a2dp_device is None
        ):
            return fallback
        return None

    def _bluetooth_device_of(self, output_id: str) -> Device | None:
        """The Bluetooth device an output plays to; None for any other or unknown."""
        address = sink_address(output_id)
        if address is None or self._bluetooth is None:
            return None
        return self._bluetooth.device_by_address(address)

    def _on_bluetooth_connected(self, path: str) -> None:
        self._bluetooth_moves.pop(path, None)
        self._route_bluetooth()

    def _on_bluetooth_moved(self, path: str, ok: bool, why: object) -> None:
        """A move is over: through, the tries start over; failed, the next try after a
        pause where tries are left, the reason to the surface once the radio gives up."""
        if ok:
            self._bluetooth_moves.pop(path, None)
        else:
            moves = self._bluetooth_moves.get(path, 0)
            logger.warning(
                "The Bluetooth audio could not be moved to %s (try %d of %d)",
                path,
                moves,
                BLUETOOTH_MOVES,
            )
            if 0 < moves < BLUETOOTH_MOVES:
                self._bluetooth_retry.start()
            elif why and self._bluetooth is not None:
                self._bluetooth.report_failure(str(why))
        self._route_bluetooth()
        if self._bluetooth_route is not None and not self._bluetooth_route.busy:
            # A device the move did not get to release keeps no mark.
            self._bluetooth_released.clear()

    def _on_audio_output_in_use(self, device_id: object, description: str) -> None:
        """The player tells where the audio goes; the volume remembered for that output
        takes over. An output not played to before starts at `NEW_OUTPUT_PERCENT`
        without a ceiling, however loud the one before played: a loudspeaker turned up
        is far too loud in headphones, and its ceiling was lowered for it alone. Where
        the system has a volume to set, the level goes to the sink of the output
        (`audio.system_volume`), else to the player."""
        if device_id is None:
            self._audio_output_in_use = None
            self._take_system_volume(None)
            self.audio_outputs_changed.emit()
            self._route_bluetooth()
            return
        output_id = bytes(device_id).decode("utf-8", "replace")
        self._audio_output_in_use = AudioOutput(
            output_id, description, is_bluetooth(output_id, description)
        )
        remembered = self._output_volumes.get(output_id)
        if remembered is None:
            remembered = (NEW_OUTPUT_PERCENT, DEFAULT_CEILING_DB)
        self._volume_percent, self._volume_ceiling_db = remembered
        self._settings.setValue(_SETTINGS_VOLUME, self._volume_percent)
        self._settings.setValue(_SETTINGS_VOLUME_CEILING, self._volume_ceiling_db)
        self._store_output_volume()
        if self._system_volume.available and self._audio_output_in_use.bluetooth:
            self._take_system_volume(output_id)
        else:
            self._take_system_volume(None)
            self._pin_system_volume(output_id)
        self._apply_volume(output_changed=True)
        self.audio_outputs_changed.emit()
        self._route_bluetooth()

    def _store_output_volume(self) -> None:
        if self._audio_output_in_use is None:
            return
        self._output_volumes[self._audio_output_in_use.id] = (
            self._volume_percent,
            self._volume_ceiling_db,
        )
        self._settings.setValue(_SETTINGS_OUTPUT_VOLUMES, json.dumps(self._output_volumes))

    def _load_output_volumes(self) -> dict[str, tuple[int, int]]:
        try:
            stored = json.loads(self._stored(_SETTINGS_OUTPUT_VOLUMES, "{}"))
            return {
                str(key): (
                    _clamp(int(value[0]), 0, 100),
                    _clamp(int(value[1]), *CEILING_RANGE_DB),
                )
                for key, value in stored.items()
            }
        except ValueError, TypeError, AttributeError, IndexError:  # edited by hand
            return {}

    def _apply_volume(self, output_changed: bool = False) -> None:
        """Hands the level on: to the sink of the output where the sound system takes
        it, with the player at 1.0 once it is there (`_on_system_volume_applied`), else
        to the player. After a change of output the player attenuates until the sink
        has the level: the sink may stand louder than wanted (new headphones at their
        full volume), and a moment too quiet does no harm."""
        sink = self._system_volume_sink
        if sink is None or output_changed:
            self._volume_changed.emit(self._volume_gain())
        if sink is not None:
            self._system_volume.set_volume(sink, self._system_raw())
        self.volume_changed.emit()

    def _volume_gain(self) -> float:
        """The factor for the player where it attenuates: the slider's gain, for an
        output with a volume of its own the nominal one of its share."""
        db = self.volume_db
        return 0.0 if db is None else 10.0 ** (db / 20.0)

    def _output_has_own_volume(self) -> bool:
        """Whether the output in use has a volume of its own that the system sets for
        it (a Bluetooth device through pactl): the slider is linear over its range
        then (`audio.volume.device_share`)."""
        output = self._audio_output_in_use
        return self._system_volume_sink is not None and output is not None and output.bluetooth

    def _system_raw(self) -> int:
        """The level for the sink in pactl's raw scale: the share of its own range for
        an output that has one, else the gain in dB."""
        if self._output_has_own_volume():
            return raw_of_share(device_share(self._volume_percent, self._volume_ceiling_db))
        return raw_volume(volume_db(self._volume_percent, self._volume_ceiling_db))

    def _pin_system_volume(self, sink: str) -> None:
        """Sets a sink the radio does not leave the level to at 100 %: the player
        attenuates for it, so a value lost between the route and the node of PipeWire
        can make it only quieter, never louder."""
        if self._system_volume.available:
            self._system_volume.set_volume(sink, VOLUME_NORM)

    def _on_system_volume_in_software(self, sink: str) -> None:
        """The Bluetooth device applies no volume of its own (no AVRCP absolute volume):
        its sink is pinned at 100 % and the player attenuates, as for a loudspeaker."""
        if sink != self._system_volume_sink:
            return
        self._take_system_volume(None)
        self._pin_system_volume(sink)
        self._apply_volume(output_changed=True)

    def _take_system_volume(self, sink: str | None) -> None:
        """The sink the sound system takes the volume for from now on; None for none."""
        self._system_volume_sink = sink
        self._system_volume.follow(sink)

    def _on_system_volume_applied(self, sink: str) -> None:
        if sink == self._system_volume_sink:
            self._volume_changed.emit(1.0)

    def _on_system_volume_failed(self, sink: str, why: str) -> None:
        """The sink would not take the volume (pactl without a server, a sink that has
        just gone): the player attenuates until a later set gets through, and the next
        change tries again. Giving the sink up for good would leave it standing low
        under a player that attenuates as well: silence at every slider position."""
        if sink != self._system_volume_sink:
            return
        logger.warning(
            "%s took no volume from the system (%s); the player attenuates until it does",
            sink,
            why,
        )
        self._volume_changed.emit(self._volume_gain())

    def _on_system_volume_changed(self, sink: str, raw: int) -> None:
        """The volume of the sink was changed elsewhere (the buttons on a Bluetooth
        device): the slider follows, on the scale of the output (the share of its own
        range, or decibels), and the output remembers it. Above the "Maximum volume"
        it goes back to the ceiling, which is what the ceiling is for."""
        if sink != self._system_volume_sink:
            return
        ceiling_db = self._volume_ceiling_db
        if self._output_has_own_volume():
            share = share_of_raw(raw)
            above = share > ceiling_share(ceiling_db) + 0.005
            percent = 100 if above else percent_of_share(share, ceiling_db)
            level = f"{round(100 * share)} % of its range"
        else:
            db = db_of_raw(raw)
            above = db is not None and db > ceiling_db + 0.05
            if above:
                percent = 100
            elif db is None:
                percent = 0
            else:
                percent = percent_of_db(db, ceiling_db)
            level = "silence" if db is None else f"{db:.1f} dB"
        logger.info("Volume of %s changed elsewhere: %s, slider %d %%", sink, level, percent)
        if percent != self._volume_percent:
            self._volume_percent = percent
            self._settings.setValue(_SETTINGS_VOLUME, percent)
            self._store_output_volume()
            self.volume_changed.emit()
        if above:
            self._system_volume.set_volume(sink, self._system_raw())

    # --- Equalizer --------------------------------------------------------------------

    @property
    def equalizer_per_station(self) -> bool:
        """Whether each station keeps its own equalizer (switched on or off and its
        settings); one without sounds like all the others."""
        return self._equalizer_per_station

    @property
    def equalizer_state(self) -> EqualizerState:
        """The equalizer of the selected station: its own while per station and it has
        one, else the one of all stations."""
        own = self._own_equalizer()
        return self._equalizer_all if own is None else own

    @property
    def equalizer_enabled(self) -> bool:
        return self.equalizer_state.enabled

    @property
    def equalizer(self) -> EqualizerSettings:
        """The settings of `equalizer_state`, also while switched off (to be shown)."""
        return self.equalizer_state.settings

    @property
    def equalizer_is_station_own(self) -> bool:
        """Whether `equalizer_state` is the selected station's own."""
        return self._own_equalizer() is not None

    def set_equalizer_enabled(self, enabled: bool) -> None:
        """Switches the equalizer on or off: while per station for the selected station,
        which keeps its own equalizer from then on; else (or with none selected) for
        all."""
        state = self.equalizer_state
        if enabled != state.enabled:
            self._store_equalizer(replace(state, enabled=enabled))

    def set_equalizer(self, settings: EqualizerSettings) -> None:
        """New settings, for the selected station or for all as `set_equalizer_enabled`.
        Settings made by hand (of no preset) are also kept as the custom ones."""
        state = self.equalizer_state
        custom = settings if settings.preset is None else state.custom
        if (settings, custom) != (state.settings, state.custom):
            self._store_equalizer(replace(state, settings=settings, custom=custom))

    def set_equalizer_per_station(self, per_station: bool) -> None:
        """Switches between one equalizer for all stations and one per station. Those of
        single stations stay stored when switched back, for the next time."""
        if per_station == self._equalizer_per_station:
            return
        self._equalizer_per_station = per_station
        self._settings.setValue(_SETTINGS_EQUALIZER_PER_STATION, per_station)
        self._push_equalizer()

    def reset_equalizers(self) -> None:
        """Deletes the equalizers of all single stations and the one of all: switched
        off and flat, as at the first start. Whether it is per station stays."""
        self._equalizer_all = EqualizerState()
        self._station_equalizers.clear()
        self._settings.remove(_SETTINGS_EQUALIZER)
        self._settings.remove(_SETTINGS_STATION_EQUALIZERS)
        self._push_equalizer()

    def set_audio_spectrum_enabled(self, enabled: bool) -> None:
        """Whether `audio_player` reports the spectrum of the audio
        (`AudioPlayer.spectrum_ready`); only while a display shows it, it costs an FFT
        ten times a second."""
        self._audio_spectrum_toggled.emit(enabled)

    def _own_equalizer(self) -> EqualizerState | None:
        if not self._equalizer_per_station or self._selected_sid is None:
            return None
        return self._station_equalizers.get(self._selected_sid)

    def _store_equalizer(self, state: EqualizerState) -> None:
        if self._equalizer_per_station and self._selected_sid is not None:
            self._station_equalizers[self._selected_sid] = state
            stations = {
                f"{sid:X}": stored.to_dict()
                for sid, stored in sorted(self._station_equalizers.items())
            }
            self._settings.setValue(_SETTINGS_STATION_EQUALIZERS, json.dumps(stations))
        else:
            self._equalizer_all = state
            self._settings.setValue(_SETTINGS_EQUALIZER, json.dumps(state.to_dict()))
        self._push_equalizer()

    def _equalizer_in_use(self) -> EqualizerSettings | None:
        """What the audio is filtered with: None while switched off."""
        state = self.equalizer_state
        return state.settings if state.enabled else None

    def _push_equalizer(self) -> None:
        self._equalizer_set.emit(self._equalizer_in_use())
        self.equalizer_changed.emit()

    def _load_equalizer(self) -> EqualizerState:
        """The stored equalizer of all stations; off and flat if missing or not
        understood (edited by hand)."""
        text = self._stored(_SETTINGS_EQUALIZER, "")
        if not text:
            return EqualizerState()
        try:
            return EqualizerState.from_dict(json.loads(text))
        except ValueError:
            logger.warning("Equalizer %r in the settings not understood", text)
            return EqualizerState()

    def _load_station_equalizers(self) -> dict[int, EqualizerState]:
        """The stored equalizers of single stations by SId; an entry that cannot be read
        is left out."""
        text = self._stored(_SETTINGS_STATION_EQUALIZERS, "")
        if not text:
            return {}
        try:
            entries = dict(json.loads(text))
        except TypeError, ValueError:
            logger.warning("Station equalizers %r in the settings not understood", text)
            return {}
        stations = {}
        for sid, data in entries.items():
            try:
                stations[int(sid, 16)] = EqualizerState.from_dict(data)
            except TypeError, ValueError:
                logger.warning("Equalizer of station %r in the settings not understood", sid)
        return stations

    def set_observer(self, observer: Observer) -> None:
        """Stores the listener's position, each coordinate clamped to its range."""
        observer = Observer(
            _clamp(float(observer.latitude), *LATITUDE_RANGE),
            _clamp(float(observer.longitude), *LONGITUDE_RANGE),
            _clamp(float(observer.height), *HEIGHT_RANGE_M),
        )
        if observer == self._observer:
            return
        self._observer = observer
        self._settings.setValue(_SETTINGS_LATITUDE, observer.latitude)
        self._settings.setValue(_SETTINGS_LONGITUDE, observer.longitude)
        self._settings.setValue(_SETTINGS_HEIGHT, observer.height)
        self.observer_changed.emit(observer)

    def set_recording_directory(self, directory: Path | str | None) -> None:
        """Stores where the recordings go; None clears the choice. The directory need
        not exist right now: the drive it lies on may be plugged in later."""
        directory = Path(directory) if directory is not None else None
        if directory == self._recording_directory:
            return
        self._recording_directory = directory
        if directory is None:
            self._settings.remove(_SETTINGS_RECORDING_DIRECTORY)
        else:
            self._settings.setValue(_SETTINGS_RECORDING_DIRECTORY, str(directory))
        self._recorder_directory_set.emit(directory)
        self.recording_directory_changed.emit(directory)

    def _load_recording_directory(self) -> Path | None:
        stored = self._stored(_SETTINGS_RECORDING_DIRECTORY, "")
        return Path(stored) if stored else None

    # --- Recording --------------------------------------------------------------------

    @property
    def ring_size(self) -> int:
        """How many of the last titles the ring keeps (`recording.recorder`)."""
        return self._ring_size

    def set_ring_size(self, size: int) -> None:
        size = _clamp(int(size), *RING_SIZE_RANGE)
        if size == self._ring_size:
            return
        self._ring_size = size
        self._settings.setValue(_SETTINGS_RING_SIZE, size)
        self._recorder_ring_size_set.emit(size)

    @property
    def label_delay_s(self) -> int:
        """How late the dynamic label runs against the audio at a title boundary
        (seconds; negative: ahead of it) for the station selected: its own value, else
        the default of every station. The title begins that much before its label."""
        return self._label_delay.value(self._selected_sid)

    @property
    def label_delay_is_station_own(self) -> bool:
        """Whether the station selected has a value of its own."""
        return self._label_delay.is_own(self._selected_sid)

    @property
    def default_label_delay_s(self) -> int:
        return self._label_delay.default

    @property
    def stop_delay_s(self) -> int:
        """How the title before ends against the label of the next, for the station
        selected (seconds): cut short that much before the label, or negative, running
        on that much after it while the next already records."""
        return self._stop_delay.value(self._selected_sid)

    @property
    def stop_delay_is_station_own(self) -> bool:
        return self._stop_delay.is_own(self._selected_sid)

    @property
    def default_stop_delay_s(self) -> int:
        return self._stop_delay.default

    @property
    def selected_service_label(self) -> str | None:
        """The name of the service selected, from the list on display; None without one."""
        sid = self._selected_sid
        if sid is None:
            return None
        return dict(self._list_services).get(sid) or f"{sid:04X}"

    def set_label_delay_s(self, seconds: int) -> None:
        """Sets the delay for the station selected (its own from then on); without a
        station, the default."""
        if self._label_delay.set(self._selected_sid, seconds):
            self._push_delays()

    def set_default_label_delay_s(self, seconds: int) -> None:
        """Sets the default, what a station without a value of its own gets; the
        station selected keeps its own, if it has one."""
        if self._label_delay.set_default(seconds):
            self._push_delays()

    def clear_station_label_delay(self) -> None:
        """The station selected follows the default again."""
        if self._label_delay.clear(self._selected_sid):
            self._push_delays()

    def set_stop_delay_s(self, seconds: int) -> None:
        """As `set_label_delay_s`, for the end of the title before."""
        if self._stop_delay.set(self._selected_sid, seconds):
            self._push_delays()

    def set_default_stop_delay_s(self, seconds: int) -> None:
        if self._stop_delay.set_default(seconds):
            self._push_delays()

    def clear_station_stop_delay(self) -> None:
        if self._stop_delay.clear(self._selected_sid):
            self._push_delays()

    def _push_delays(self) -> None:
        """Tells the recorder the delays of the station selected, and the surface that
        they may have changed: set, or another station selected."""
        self._recorder_label_delay_set.emit(self.label_delay_s)
        self._recorder_stop_delay_set.emit(self.stop_delay_s)
        self.label_delay_changed.emit()
        self.stop_delay_changed.emit()

    @property
    def recording_enabled(self) -> bool:
        """Whether the recorder writes at all (the switch "Recording" of the settings)."""
        return self._recording_enabled

    def set_recording_enabled(self, enabled: bool) -> None:
        """Switched off, the title recording ends and nothing is written until on."""
        enabled = bool(enabled)
        if enabled == self._recording_enabled:
            return
        self._recording_enabled = enabled
        self._settings.setValue(_SETTINGS_RECORDING_ENABLED, enabled)
        self._recorder_enabled_set.emit(enabled)
        self.recording_enabled_changed.emit(enabled)

    @property
    def recordings(self) -> tuple[Recording, ...]:
        """The ring of the last titles, oldest first; complete titles only, the one
        recording is in `recorder_status`."""
        return self._recordings

    @property
    def kept_recordings(self) -> tuple[Recording, ...]:
        """The titles kept out of the ring, oldest first."""
        return self._kept_recordings

    @property
    def recorder_status(self) -> RecorderStatus:
        """What the recorder does, with the title recording, or why it does not."""
        return self._recorder_status

    def keep_recording(self, recording_id: str) -> None:
        """Moves a title of the ring out of it, into the directory for good."""
        self._recording_kept.emit(recording_id)

    def delete_recording(self, recording_id: str) -> None:
        """Removes a title of the ring or a kept one, file and all; one playing stops."""
        if self._playback_recording is not None and recording_id == self._playback_recording.id:
            self.stop_playback()
        self._recording_deleted.emit(recording_id)

    def _push_recorder_station(self) -> None:
        """Tells the recorder which service plays and what it is called; the name
        from the list on display (stored or received), the SId where it has none."""
        sid = self._selected_sid
        if sid is None:
            self._recorder_station_set.emit(None)
            return
        channel = self._list_channel or f"{self._frequency_mhz:.3f} MHz"
        label = dict(self._list_services).get(sid) or f"{sid:04X}"
        ensemble = self._live_ensemble_label
        if ensemble is None and self._list_channel is not None:
            info = self._ensemble_cache.get(self._list_channel)
            ensemble = info.label if info is not None else None
        self._recorder_station_set.emit(Station(channel, sid, label, ensemble or ""))

    def _on_recordings_changed(self, ring: object, kept: object) -> None:
        self._recordings = tuple(ring)
        self._kept_recordings = tuple(kept)
        self._follow_playing_title()
        self.recordings_changed.emit()

    def _on_recorder_status(self, status: object) -> None:
        self._recorder_status = status
        self.recorder_status_changed.emit(status)

    # --- Playback of a recording ------------------------------------------------------

    @property
    def playback(self) -> PlaybackState | None:
        """The recording playing in place of the live audio: loading, playing or paused,
        with its position; None while the station is heard."""
        return self._playback

    @property
    def playback_recording(self) -> Recording | None:
        """The title playing, as last listed: its id follows a move into "Kept"; one
        that went meanwhile (deleted, dropped by the ring) keeps its last entry."""
        return self._playback_recording

    def play_recording(self, recording_id: str, position: float = 0.0) -> None:
        """Plays a complete title of the ring or a kept one from `position` seconds on;
        the live audio is dropped meanwhile, reception and recording go on. Lifts the
        mute: whoever asks to hear a recording wants to hear it."""
        recording = self._find_recording(recording_id)
        if recording is None:
            return
        if not self._audio_enabled:
            self.set_audio_enabled(True)
        self._playback_source_id = recording_id
        self._playback_recording = recording
        self._playback_position = max(0.0, float(position))
        self._set_playback(
            PlaybackState(recording_id, LOADING, self._playback_position, recording.duration)
        )
        self._playback_load_requested.emit(recording_id)

    def pause_playback(self) -> None:
        if self._playback is not None and self._playback.state == PLAYING:
            self._playback_paused.emit()

    def resume_playback(self) -> None:
        if self._playback is not None and self._playback.state == PAUSED:
            self._playback_resumed.emit()

    def toggle_playback(self) -> None:
        """Pauses the recording playing, or goes on with it."""
        if self._playback is None:
            return
        if self._playback.state == PAUSED:
            self._playback_resumed.emit()
        elif self._playback.state == PLAYING:
            self._playback_paused.emit()

    def seek_playback(self, position: float) -> None:
        """Goes on from `position` seconds into the recording playing."""
        playback = self._playback
        if playback is None:
            return
        position = max(0.0, min(float(position), playback.duration))
        if playback.state == LOADING:
            self._playback_position = position  # played from there once loaded
            self._set_playback(replace(playback, position=position))
            return
        self._playback_seek.emit(position)

    def stop_playback(self) -> None:
        """Ends the recording playing; the station is heard again."""
        if self._playback_source_id is None:
            return
        self._forget_playback()
        self._playback_stop_requested.emit()

    def _find_recording(self, recording_id: str) -> Recording | None:
        recordings = (*self._recordings, *self._kept_recordings)
        return next((r for r in recordings if r.id == recording_id), None)

    def _forget_playback(self) -> None:
        self._playback_source_id = None
        self._playback_recording = None
        self._set_playback(None)

    def _set_playback(self, state: PlaybackState | None) -> None:
        if state != self._playback:
            self._playback = state
            self.playback_changed.emit(state)

    def _follow_playing_title(self) -> None:
        """The title playing moved into "Kept" (another id, the same title): the
        display follows it."""
        recording = self._playback_recording
        if recording is None or self._find_recording(recording.id) is not None:
            return
        recordings = (*self._recordings, *self._kept_recordings)
        moved = next(
            (r for r in recordings if r.start == recording.start and r.sid == recording.sid),
            None,
        )
        if moved is None:
            return
        self._playback_recording = moved
        if self._playback is not None:
            self._set_playback(replace(self._playback, recording_id=moved.id))

    def _on_playback_loaded(self, audio: object) -> None:
        if self._playback_source_id is not None and audio.recording_id == self._playback_source_id:
            self._playback_stop_set.emit(-1.0, 0.0)  # to its end, not where the cutter halts
            self._playback_requested.emit(audio, self._playback_position)

    # --- The cutter ---------------------------------------------------------------------

    @property
    def playback_source_id(self) -> str | None:
        """The id of what plays: a recording, or a stretch of the cutter (`Stretch.id`)."""
        return self._playback_source_id

    def load_stretch(self, recording_id: str) -> None:
        """Has a complete title read with its neighbours for the cutter
        (`stretch_loaded`, `stretch_failed`)."""
        self._stretch_load_requested.emit(recording_id)

    def read_levels(self, stretch: Stretch) -> None:
        """Has the level of a stretch read (`levels_read`, `levels_finished`), giving up
        the one read before."""
        self._levels_requested.emit(RecordingAudio(stretch.id, stretch.audio_format, stretch.units))

    def cancel_levels(self) -> None:
        self._levels_cancelled.emit()

    def play_stretch(self, stretch: Stretch, position: float, stop: float, back_to: float) -> None:
        """Plays a stretch of the cutter from `position` seconds on, in place of the live
        audio as a recording: up to `stop`, then it pauses back at `back_to`
        (`AudioPlayer.set_playback_stop`). Lifts the mute, as `play_recording`."""
        if not self._audio_enabled:
            self.set_audio_enabled(True)
        position = max(0.0, min(float(position), stretch.duration))
        self._playback_source_id = stretch.id
        self._playback_recording = stretch.opened
        self._playback_position = position
        self._set_playback(PlaybackState(stretch.id, LOADING, position, stretch.duration))
        self._playback_stop_set.emit(float(stop), float(back_to))
        self._playback_requested.emit(
            RecordingAudio(stretch.id, stretch.audio_format, stretch.units), position
        )

    def seek_stretch(self, position: float, stop: float, back_to: float) -> None:
        """Goes on from `position` in the stretch playing or paused, now up to `stop`.
        The player has the stretch from `play_stretch` on: a seek goes to it even before
        it reports playing."""
        playback = self._playback
        if playback is None:
            return
        position = max(0.0, min(float(position), playback.duration))
        self._playback_stop_set.emit(float(stop), float(back_to))
        self._playback_seek.emit(position)
        if playback.state == LOADING:
            self._set_playback(replace(playback, position=position))

    def save_cut(self, request: CutRequest) -> None:
        """Has a cut written as a kept title (`cut_saved`, `cut_failed`)."""
        self._cut_save_requested.emit(request)

    def _on_playback_load_failed(self, recording_id: str, message: str) -> None:
        if recording_id != self._playback_source_id:
            return
        self._forget_playback()
        self.playback_failed.emit(message)

    def _on_playback_changed(self, state: object) -> None:
        """What the player reports, followed only for the recording asked for last: an
        earlier one may still report until the new one plays."""
        if self._playback_source_id is None:
            return
        if state is None:
            # The recording ended, or `stop` ended it with the station. Not while the
            # next one loads: that is the end of the one before.
            if self._playback is None or self._playback.state != LOADING:
                self._forget_playback()
            return
        if state.recording_id != self._playback_source_id:
            return
        shown = self._playback_recording
        self._set_playback(
            replace(state, recording_id=state.recording_id if shown is None else shown.id)
        )

    def _load_favourites(self) -> set[tuple[str, int]]:
        """The stored favourites; an entry that cannot be read (edited by hand) is left
        out."""
        favourites = set()
        for entry in self._stored(_SETTINGS_FAVOURITES, []):
            channel, _, sid = str(entry).partition("/")
            try:
                favourites.add((channel, int(sid, 16)))
            except ValueError:
                logger.warning("Favourite %r in the settings not understood", entry)
        return favourites

    def _store_tuning(self) -> None:
        channel = _SETTINGS_MANUAL_CHANNEL if self._channel is None else self._channel
        self._settings.setValue(_SETTINGS_CHANNEL, channel)
        self._settings.setValue(_SETTINGS_FREQUENCY, self._frequency_mhz)

    def _restore_selected_service(self) -> None:
        """Selects the service of the last session again if the stored list of the
        channel carries it, so that it is decoded right after the start."""
        sid = self._stored(_SETTINGS_SERVICE, -1)  # SIds are never negative
        if any(listed == sid for listed, _ in self._list_services):
            self.select_service(sid)

    # --- Tuning -----------------------------------------------------------------------

    def select_channel(self, channel: str | None) -> None:
        """Selects a channel of Band III, or with None the manual frequency, which starts
        from the frequency tuned to so far. During reception the device is retuned,
        otherwise the stored service list of the channel is shown."""
        if channel is not None and channel not in BAND_III_CHANNELS:
            raise ValueError(f"unknown channel {channel!r}")
        self._stay_from_visit()
        self._select_channel(channel)

    def _select_channel(self, channel: str | None) -> None:
        self._channel = channel
        if channel is not None:
            self._frequency_mhz = BAND_III_CHANNELS[channel]
        self._store_tuning()
        self.tuning_changed.emit(channel, self._frequency_mhz)
        if self._worker is not None:
            self._retune(channel)
        else:
            self._show_stored_services(channel)

    def set_manual_frequency(self, frequency_mhz: float) -> None:
        """Tunes to a frequency entered by hand; a selected channel gives way to it."""
        self._frequency_mhz = _clamp(frequency_mhz, *FREQUENCY_RANGE_MHZ)
        if self._channel is not None:
            self.select_channel(None)
            return
        self._store_tuning()
        self.tuning_changed.emit(None, self._frequency_mhz)
        if self._worker is not None:
            self._retune(None)

    # --- Stored service lists ---------------------------------------------------------

    def _save_ensemble_cache(self) -> None:
        try:
            self._ensemble_cache.save()
        except OSError as exc:
            self.channel_list_save_failed.emit(str(exc))

    def _show_stored_services(self, channel: str | None) -> None:
        """Shows the stored service list of a channel (without running reception)."""
        self._list_channel = channel
        self._list_is_live = False
        self._live_list_complete = False
        self._drop_selection_missing_from(channel)
        info = None if channel is None else self._ensemble_cache.get(channel)
        self._list_services = [] if info is None else list(info.services)
        programme_types = {} if info is None else dict(info.programme_types)
        label = None if info is None else info.label
        self.service_list_changed.emit(label, list(self._list_services), programme_types)
        self._update_service_source()
        self._update_other_ensembles()
        self.stored_list_shown.emit(channel)

    def _drop_selection_missing_from(self, channel: str | None) -> None:
        """Drops the selected service if the stored list of the channel does not
        contain it, otherwise a service the ensemble does not even contain would be
        searched for after the start or the channel change."""
        info = None if channel is None else self._ensemble_cache.get(channel)
        services = () if info is None else info.services
        if self._selected_sid is not None and all(sid != self._selected_sid for sid, _ in services):
            self._selected_sid = None
            self._service_selected.emit(None)
            self._settings.remove(_SETTINGS_SERVICE)
            self.selected_service_changed.emit(None)
            self._update_other_ensembles()
            if self._equalizer_per_station:
                self._push_equalizer()

    def _update_service_source(self) -> None:
        """Describes where the displayed service list comes from and whether it can be
        stored: as soon as a received list for a channel is available."""
        channel = self._list_channel
        stored = None if channel is None else self._ensemble_cache.get(channel)
        if not self._list_is_live:
            if channel is None:
                text = tr("Manual frequency: no stored station list.")
            elif stored is None:
                text = tr("No station list stored for channel {channel}.").format(channel=channel)
            else:
                text = tr("Stored station list ({source}).").format(source=describe_stored(stored))
        else:
            if self._worker is not None:
                origin = tr("Live from the FIC.")
            else:
                origin = tr("Last received.")
            if channel is None:
                storage = tr("Nothing is stored at a manual frequency.")
            elif stored is None:
                storage = tr("Not yet stored for channel {channel}.").format(channel=channel)
            else:
                storage = tr("Stored: {source}.").format(source=describe_stored(stored))
            text = f"{origin} {storage}"
        can_store = (
            self._list_is_live
            and channel is not None
            and self._scanner is None
            and bool(self._live_ensemble_label or self._live_services)
        )
        self.service_source_changed.emit(text, can_store)

    def store_live_services(self) -> None:
        """Stores the received service list with the ensemble's name for its channel, like
        a scan of this one channel."""
        channel = self._list_channel
        if channel is None or not self._list_is_live:
            return
        info = EnsembleInfo(
            label=self._live_ensemble_label,
            services=tuple(self._live_services),
            updated_at=time.time(),
            source=SOURCE_LIVE,
            programme_types=tuple(sorted(self._live_programme_types.items())),
            eid=self._live_eid,
            frequencies=tuple(sorted(self._live_frequencies.items())),
            service_ensembles=tuple(sorted(self._live_service_ensembles.items())),
        )
        self._ensemble_cache.put(
            channel, keep_service_ensembles(info, self._ensemble_cache.get(channel))
        )
        self._save_ensemble_cache()
        self.channel_stored.emit(channel)
        self._update_service_source()
        self._update_other_ensembles()

    # --- Channel scan -----------------------------------------------------------------

    def start_scan(self) -> None:
        """Scans all channels of Band III with the device settings and stores what it
        finds; only while neither reception nor a scan runs."""
        if self._state is not ReceptionState.IDLE:
            return
        self._resume_on_device_change = False
        self._scan_found = 0
        self._scan_done = 0
        self._scan_failed = False
        self._set_state(ReceptionState.SCANNING)
        self._set_status(tr("Scan: connecting to {device}...").format(device=self._device_text()))

        settings = self._device_settings
        self._scanner = ChannelScanner(
            BAND_III_CHANNELS,
            sample_rate=SAMPLE_RATE,
            gain=settings.gain(),
            freq_correction_ppm=settings.ppm,
            rspduo=settings.rspduo(),
            rtl_tcp=settings.rtl_tcp(),
        )
        self._scanner.channel_started.connect(self._on_scan_channel_started)
        self._scanner.channel_progress.connect(self._on_scan_progress)
        self._scanner.channel_scanned.connect(self._on_channel_scanned)
        self._scanner.spectrum_ready.connect(self.scan_spectrum_ready)
        self._scanner.error.connect(self._on_scan_error)
        self._scanner.scan_finished.connect(self._on_scan_finished)
        self._scanner.start()
        self._update_service_source()

    def cancel_scan(self) -> None:
        """Ends the scan after the block being read; it reports back through
        `state_changed` like a completed one."""
        if self._scanner is None:
            return
        # Do not block: the block being read is finished first, then the scan reports
        # back through `scan_finished`.
        self._scanner.cancel()
        self._set_status(tr("Cancelling scan..."))

    def _on_scan_channel_started(self, channel: str, index: int, total: int) -> None:
        self._scan_position = (index, total)
        self._show_scan_status(channel, "")
        self.scan_progress.emit(channel, index, total)

    def _on_scan_progress(
        self,
        channel: str,
        valid_fibs: int,
        total_fibs: int,
        ensemble_label: str | None,
        services: int,
    ) -> None:
        details = [tr("{valid}/{total} FIBs error-free").format(valid=valid_fibs, total=total_fibs)]
        if ensemble_label:
            details.append(tr("ensemble {label}").format(label=ensemble_label))
        if services:
            stations = tr_plural("{count} station", "{count} stations", services)
            details.append(stations.format(count=services))
        self._show_scan_status(channel, join(", ", details))

    def _show_scan_status(self, channel: str, detail: str) -> None:
        index, total = self._scan_position
        text = tr("Scan {number}/{total}: channel {channel} ({frequency} MHz)").format(
            number=index + 1,
            total=total,
            channel=channel,
            frequency=f"{BAND_III_CHANNELS[channel]:.3f}",
        )
        if detail:
            text = text + " - " + detail
        found = tr("{count} found so far").format(count=self._scan_found)
        self._set_status(text + " - " + found)

    def _on_channel_scanned(self, channel: str, info: EnsembleInfo | None) -> None:
        # An empty channel loses a previously stored entry as well: the channel list
        # shows what was actually receivable most recently.
        if info is None:
            self._ensemble_cache.remove(channel)
            self._remove_stored_guide(channel)
        else:
            self._ensemble_cache.put(
                channel, keep_service_ensembles(info, self._ensemble_cache.get(channel))
            )
            self._scan_found += 1
        self._scan_done += 1
        self.channel_stored.emit(channel)
        self._update_other_ensembles()

    def _remove_stored_guide(self, channel: str) -> None:
        """Without an ensemble on the channel its stored programme guide is void."""
        try:
            self._guide_cache.remove(channel)
        except OSError as exc:
            logger.warning("Stored guide of %s not removed: %s", channel, exc)

    def _on_scan_error(self, message: str) -> None:
        self._scan_failed = True
        self._set_status(tr("Error: {message}").format(message=message))
        self.error_occurred.emit(message)

    def _on_scan_finished(self, completed: bool) -> None:
        scanner = self._scanner
        self._scanner = None
        if scanner is not None:
            scanner.wait()  # `run()` ends right after this signal
        if self._scan_done:
            self._save_ensemble_cache()
        self._set_state(ReceptionState.IDLE)
        if not self._scan_failed:
            done, found = self._scan_done, self._scan_found
            channels = tr_plural("{count} channel searched", "{count} channels searched", done)
            ensembles = tr_plural("{count} ensemble found", "{count} ensembles found", found)
            if completed:
                text = tr(
                    "Scan finished: {channels}, {ensembles} (green in the channel selection)."
                )
            else:
                text = tr(
                    "Scan cancelled: {channels}, {ensembles} (green in the channel selection)."
                )
            self._set_status(
                text.format(
                    channels=channels.format(count=done), ensembles=ensembles.format(count=found)
                )
            )
        self._show_stored_services(self._channel)

    # --- Reception --------------------------------------------------------------------

    def _set_state(self, state: ReceptionState) -> None:
        self._state = state
        self.state_changed.emit(state)

    def _set_status(self, text: str) -> None:
        self.status_changed.emit(text)

    def start_reception(self) -> None:
        """Opens the device with the device settings and receives the frequency tuned
        to; only while neither reception nor a scan runs."""
        if self._state is not ReceptionState.IDLE:
            return
        self._resume_on_device_change = False
        self._sample_count = 0
        self._set_state(ReceptionState.CONNECTING)
        settings = self._device_settings
        self._set_status(tr("Connecting to {device}...").format(device=self._device_text()))
        self._reset_live()

        self._center_freq_hz = self._frequency_mhz * 1e6
        rspduo = settings.rspduo()
        rtl_tcp = settings.rtl_tcp()
        if rspduo is not None:
            self._worker = RspDuoWorker(center_freq=self._center_freq_hz, settings=rspduo)
            self._worker.overload_changed.connect(self._on_overload_changed)
        elif rtl_tcp is not None:
            self._worker = RtlTcpWorker(center_freq=self._center_freq_hz, settings=rtl_tcp)
        else:
            self._worker = RtlSdrWorker(
                center_freq=self._center_freq_hz,
                sample_rate=SAMPLE_RATE,
                gain=settings.gain(),
                freq_correction_ppm=settings.ppm,
            )
        self._worker.connected.connect(self._on_connected)
        # Resets the signal processing within the sample stream, at the start and at
        # every channel change (see `DspWorker.restart_stream`).
        self._worker.tuned.connect(self._dsp_worker.restart_stream)
        self._worker.samples_ready.connect(self._on_samples_ready)
        self._worker.samples_ready.connect(self._dsp_worker.process_samples)
        self._worker.error.connect(self._on_error)
        self._worker.stopped.connect(self._on_stopped)
        self._worker.start()
        self._show_live_services(self._worker.tuning, self._channel)

    def stop_reception(self) -> None:
        """Ends reception without blocking; `state_changed` reports when it has ended."""
        if self._worker is None:
            return
        self._set_status(tr("Stopping reception..."))
        self._worker.stop()

    def restart_reception(self) -> None:
        """Ends reception and starts it again as soon as the device is closed, so that
        changed device settings take effect: neither the stick nor the RSPduo takes them
        while running. Does nothing without reception."""
        if self._worker is None:
            return
        self._restart_pending = True
        self.stop_reception()

    def _retune(self, channel: str | None) -> None:
        """Switches the running reception to the selected channel or the manually
        entered frequency.

        The device stays open and is only retuned (see `RtlSdrWorker.retune` and
        `RspDuoWorker.retune`), which takes about 42 ms in this thread on the stick. The
        signal processing resets itself at the boundary in the sample stream and
        confirms it (`_on_stream_restarted`); until then its results still come from
        the old frequency.
        """
        center_freq_hz = self._frequency_mhz * 1e6
        if center_freq_hz == self._center_freq_hz:
            # Only the name changed, e.g. to a manual frequency on the same channel:
            # the received service list stays valid.
            self._list_channel = channel
            self._update_service_source()
            return

        try:
            tuning = self._worker.retune(center_freq_hz)
        except DeviceError as exc:
            # Treated like a failed read: reception ends and the message stays
            # visible instead of being replaced by "Stopped.".
            self._worker.stopped.disconnect(self._on_stopped)
            self._worker.stop()
            self._on_error(exception_text(exc))
            return
        if tuning is None:
            # Reception is just ending.
            self._show_stored_services(channel)
            return

        self._center_freq_hz = center_freq_hz
        frequency = f"{center_freq_hz / 1e6:.3f}"
        if channel is None:
            text = tr("Tuning to {frequency} MHz...").format(frequency=frequency)
        else:
            text = tr("Tuning to channel {channel} ({frequency} MHz)...").format(
                channel=channel, frequency=frequency
            )
        self._set_status(text)
        self._drop_selection_missing_from(channel)
        self._show_live_services(tuning, channel)

    def _show_live_services(self, tuning: int, channel: str | None) -> None:
        """From now on the service list shows what the FIC delivers, for the given
        tuning once the signal processing has confirmed it.

        A selected service stays selected: pre-selected from the stored list at the
        start, or on a channel change if the new ensemble carries it as well.
        """
        self._awaited_tuning = tuning
        self._list_channel = channel
        self._list_is_live = True
        self._live_list_complete = False
        self._list_services = []
        self._live_ensemble_label = None
        self._live_services = []
        self._live_programme_types = {}
        self._live_eid = None
        self._live_frequencies = {}
        self._live_service_ensembles = {}
        self.service_list_changed.emit(None, [], {})
        self._update_service_source()
        self._update_other_ensembles()

    def _on_stream_restarted(self, tuning: int) -> None:
        # Several quick channel changes: only the most recent one counts.
        if tuning != self._awaited_tuning:
            return
        self._awaited_tuning = None
        # Up to here the displays could still receive results of the old frequency.
        self._reset_live()

    def _reset_live(self) -> None:
        """Voids everything that belongs to the previously received frequency."""
        self._clear_overload()
        self._data_components = []
        self._data_decoding = set()
        self._data_requested = set()
        self._decoded_spi = None
        self._reset_service()
        self.live_reset.emit()
        self.data_components_changed.emit()

    def _reset_service(self) -> None:
        """Voids everything that belongs to the previously selected service."""
        self._audio_stop_requested.emit()  # ends a recording playing as well
        self._forget_playback()
        self._recorder_reset.emit()
        self.service_reset.emit()

    def _on_connected(self, message: str) -> None:
        self._set_status(message)
        self._set_state(ReceptionState.RECEIVING)

    def _on_samples_ready(self, samples: np.ndarray) -> None:
        self._sample_count += len(samples)
        # During a channel change "Tuning to..." stays until it is complete.
        if self._awaited_tuning is None:
            count = f"{self._sample_count:,}"
            self._set_status(tr("Receiving... {count} IQ samples").format(count=count))

    def _on_error(self, message: str) -> None:
        # Not again after an error: the device would most likely fail the same way.
        # Another device may well work.
        self._restart_pending = False
        self._resume_on_device_change = True
        self._set_status(tr("Error: {message}").format(message=message))
        self.error_occurred.emit(message)
        self._reset_after_stop()

    def _on_stopped(self) -> None:
        self._set_status(tr("Stopped."))
        self._reset_after_stop()

    def _reset_after_stop(self) -> None:
        self._audio_stop_requested.emit()
        self._forget_playback()
        self._recorder_reset.emit()  # the title ends with reception
        self._worker = None
        self._clear_overload()
        self._awaited_tuning = None
        # The signal processing only reports its load per second of signal, so without
        # reception the last report would stay on display.
        self._dsp_load = self._dsp_waiting = 0.0
        self._emit_load()
        # Without reception nothing confirms a running announcement any more.
        self._on_announcement_ready(None)
        self._set_state(ReceptionState.IDLE)
        kept = bool(self._live_ensemble_label or self._live_services)
        if kept:
            # The received list stays and can still be stored.
            self._update_service_source()
        else:
            # Nothing received (e.g. because the stick is missing): show the stored
            # list again instead of an empty one.
            self._show_stored_services(self._channel)
        self.reception_stopped.emit(kept)
        visit = self._end_visit()
        if visit is not None:
            # Reception ended in another ensemble's announcement: back to where it came
            # from, ready for the next start.
            self._select_channel(visit.origin_channel)
            self._select_service(visit.origin_sid)
        if self._restart_pending:
            self._restart_pending = False
            self.start_reception()

    def _on_overload_changed(self, overloaded: bool) -> None:
        """The RSPduo reports an overload of its ADC, or its end."""
        self._adc_overload = overloaded
        if overloaded:
            self._note_overload()

    def _on_clipping_ready(self, fraction: float) -> None:
        if self._worker is None or self._awaited_tuning is not None:
            # A straggler from blocks still queued when reception ended, or a block of
            # the frequency before.
            return
        if fraction > CLIPPING_WARNING_THRESHOLD or self._adc_overload:
            self._note_overload()

    def _note_overload(self) -> None:
        self._overload_hold.start()
        self._set_overloaded(True)

    def _clear_overload(self) -> None:
        self._adc_overload = False
        self._overload_hold.stop()
        self._set_overloaded(False)

    def _set_overloaded(self, overloaded: bool) -> None:
        if overloaded != self._overloaded:
            self._overloaded = overloaded
            self.overloaded_changed.emit(overloaded)

    # --- Services ---------------------------------------------------------------------

    def select_service(self, sid: int) -> None:
        """Decodes the service with this SId; before reception, as soon as it starts."""
        self._stay_from_visit()
        self._select_service(sid)

    def _select_service(self, sid: int) -> None:
        self._selected_sid = sid
        self._settings.setValue(_SETTINGS_SERVICE, sid)
        self._reset_service()
        self._service_selected.emit(sid)
        self._push_recorder_station()
        self._push_delays()
        self.selected_service_changed.emit(sid)
        self._update_other_ensembles()
        if self._equalizer_per_station:
            self._push_equalizer()

    def play_service(self, channel: str | None, sid: int) -> None:
        """Tunes to the channel of a service (None: stays on the manual frequency),
        selects the service and starts reception if it is not running, as if channel
        and service had been chosen by hand. Ignored while connecting or scanning."""
        if self._state in (ReceptionState.CONNECTING, ReceptionState.SCANNING):
            return
        if channel != self._channel and (channel is None or channel not in BAND_III_CHANNELS):
            return
        self._stay_from_visit()
        self._play(channel, sid)

    def _play(self, channel: str | None, sid: int) -> None:
        if channel != self._channel:
            self._select_channel(channel)
        self._select_service(sid)
        if self._worker is None:
            self.start_reception()

    def _on_services_updated(
        self,
        ensemble_label: str | None,
        services: list[tuple[int, str]],
        programme_types: dict[int, int],
    ) -> None:
        # The signal processing may lag behind reception. Stragglers of the previous
        # frequency no longer belong to the displayed list, be it after a channel
        # change during reception or after stopping and selecting another channel.
        if not self._list_is_live or self._awaited_tuning is not None:
            return
        self._live_ensemble_label = ensemble_label
        self._live_services = list(services)
        self._live_programme_types = dict(programme_types)
        self._list_services = list(services)
        self._push_recorder_station()  # the name of the service recorded
        self.service_list_changed.emit(ensemble_label, list(services), dict(programme_types))
        self.ensemble_received.emit(ensemble_label, dict(services))
        self._update_service_source()

    def _on_service_list_completed(self) -> None:
        if not self._list_is_live or self._awaited_tuning is not None:
            return  # a straggler of the previous frequency, see `_on_services_updated`
        self._live_list_complete = True
        self.service_list_completed.emit(
            self._live_ensemble_label, list(self._live_services), dict(self._live_programme_types)
        )

    def _on_frequency_information_updated(
        self, eid: int | None, frequencies: dict[int, tuple[int, ...]]
    ) -> None:
        """Which ensemble the channel carries, and where it and others are broadcast
        according to FIG 0/21: stored with the service list, used by the programme
        guide to find the channel of a service."""
        if not self._list_is_live or self._awaited_tuning is not None:
            return  # a straggler of the previous frequency, see `_on_services_updated`
        self._live_eid = eid
        self._live_frequencies = dict(frequencies)
        self.frequency_information_received.emit(eid, dict(frequencies))
        self._store_service_ensembles()
        self._update_other_ensembles()

    def _on_service_ensembles_updated(self, service_ensembles: dict[int, tuple[int, ...]]) -> None:
        if not self._list_is_live or self._awaited_tuning is not None:
            return  # a straggler of the previous frequency, see `_on_services_updated`
        withdrawn = self._live_service_ensembles.keys() - service_ensembles.keys()
        self._live_service_ensembles = dict(service_ensembles)
        self._store_service_ensembles(withdrawn)
        self._update_other_ensembles()

    def _store_service_ensembles(self, withdrawn: Collection[int] = ()) -> None:
        """Adds what FIG 0/24 of the received ensemble says to the stored list of its
        channel, if that came from the same ensemble (see
        `dab.ensemble_cache.keep_service_ensembles`), and drops the SIds whose entry it
        has withdrawn since the tuning. A scan hears little of FIG 0/24, and only the
        desktop stores a received list by hand. Without `channel_stored`: nothing a
        service list shows changes."""
        channel = self._list_channel
        stored = None if channel is None else self._ensemble_cache.get(channel)
        if stored is None or stored.eid is None or stored.eid != self._live_eid:
            return
        kept = {sid: eids for sid, eids in stored.service_ensembles if sid not in withdrawn}
        service_ensembles = tuple(sorted((kept | self._live_service_ensembles).items()))
        if service_ensembles != stored.service_ensembles:
            self._ensemble_cache.put(channel, replace(stored, service_ensembles=service_ensembles))
            self._save_ensemble_cache()

    def _update_other_ensembles(self) -> None:
        """Locates the other ensembles of the selected service anew (see
        `other_ensembles`) and reports them if that changed anything."""
        channel = self._list_channel
        stored = None if channel is None else self._ensemble_cache.get(channel)
        if stored is not None and self._list_is_live and self._live_eid not in (None, stored.eid):
            stored = None  # another ensemble is received there now
        own_eid = None if stored is None else stored.eid
        service_ensembles = {} if stored is None else dict(stored.service_ensembles)
        announced = {} if stored is None else dict(stored.frequencies)
        if self._list_is_live:
            own_eid = own_eid if self._live_eid is None else self._live_eid
            service_ensembles |= self._live_service_ensembles
            announced |= self._live_frequencies
        located = ()
        if own_eid is not None and self._selected_sid is not None:
            located = other_ensembles(
                service_ensembles.get(self._selected_sid, ()),
                own_eid,
                self._ensemble_cache.entries(),
                announced,
            )
        if located != self._other_ensembles:
            self._other_ensembles = located
            self.other_ensembles_changed.emit(located)

    def _on_announcement_ready(self, status: AnnouncementStatus | None) -> None:
        """Only during reception: afterwards the signal processing still answers a
        service selection from the last ensemble, and its signal time stands still, so
        an announcement running at the stop would never end."""
        if self._worker is None:
            status = None
        source = None
        if status is not None and status.source_sid is not None:
            source = dict(self._live_services).get(status.source_sid)
        elif status is not None and status.oe_active is not None:
            source = self._oe_source_label(status)
        self.announcement_changed.emit(status, source)
        self._follow_announcements(status)

    # --- Favourites -------------------------------------------------------------------

    @property
    def favourites(self) -> frozenset[tuple[str, int]]:
        """The stations marked as favourites, as (channel, SId): a station is a service
        on a channel, as on the station page. One that a later scan no longer finds
        stays marked, should it come back."""
        return frozenset(self._favourites)

    def is_favourite(self, channel: str | None, sid: int | None) -> bool:
        return (channel, sid) in self._favourites

    def set_favourite(self, channel: str, sid: int, favourite: bool) -> None:
        """Marks the station as favourite or unmarks it, also for later sessions."""
        if self.is_favourite(channel, sid) == favourite:
            return
        if favourite:
            self._favourites.add((channel, sid))
        else:
            self._favourites.discard((channel, sid))
        entries = sorted(f"{channel}/{sid:X}" for channel, sid in self._favourites)
        self._settings.setValue(_SETTINGS_FAVOURITES, entries)
        self.favourites_changed.emit()

    # --- Announcements of other ensembles ---------------------------------------------

    def return_from_announcement(self) -> None:
        """Goes back to the service left for another ensemble's announcement before it
        is over; does nothing unless reception went there."""
        if self._visit is not None:
            self._return_from_visit()

    def _follow_announcements(self, status: AnnouncementStatus | None) -> None:
        """Goes to another ensemble for an announcement running there (FIG 0/26) that
        applies to the selected service, if switching to announcements is on, and back
        once it is over.

        There the target service is decoded; the announcement counts as over once FIG
        0/19 there no longer shows one in its subchannel (or, for an alarm, for all
        services), or if none shows up within `ANNOUNCEMENT_SEARCH_MS`. Statuses of a
        tuning not yet confirmed are stragglers of the frequency before.
        """
        if self._awaited_tuning is not None:
            return
        if self._visit is not None:
            if status is None:
                return
            if status.active is not None or status.carried is not None:
                self._visit_heard = True
                self._visit_timer.stop()
            elif self._visit_heard:
                self._return_from_visit()
            return
        oe = None if status is None else status.oe_active
        key = None if oe is None else (oe.cluster_id, oe.target_sid)
        self._reported = key
        if key == self._visited or (key is None and self._revisit_timer.isActive()):
            return  # still reported after coming back, or not yet again
        self._visited = None
        if (
            key is not None
            and self._announcement_switching
            and self._state is ReceptionState.RECEIVING
            and self._channel is not None
            and self._selected_sid is not None
        ):
            self._start_visit(status)

    def _start_visit(self, status: AnnouncementStatus) -> None:
        announcement = status.oe_active
        channel = self._channel_of_ensemble(status.oe_eid)
        if channel is None or channel == self._channel:
            return
        info = self._ensemble_cache.get(channel)
        services = {} if info is None else dict(info.services)
        if announcement.cluster_id == ALARM_CLUSTER:
            # The alarm interrupts every service there; the signal processing moves the
            # audio to its subchannel.
            sid = next(iter(services), None)
        else:
            sid = announcement.target_sid  # its primary component, whatever the SCIdS
        if sid is None:
            return
        self._visited = (announcement.cluster_id, announcement.target_sid)
        self._visit = AnnouncementVisit(
            announcement=announcement,
            channel=channel,
            sid=sid,
            label=services.get(sid),
            origin_channel=self._channel,
            origin_sid=self._selected_sid,
            origin_label=dict(self._live_services).get(self._selected_sid),
        )
        self._visit_heard = False
        logger.info(
            "Announcement in ensemble %04X: going to %04X on channel %s",
            status.oe_eid,
            sid,
            channel,
        )
        self._play(channel, sid)
        self._visit_timer.start()
        self.announcement_visit_changed.emit(self._visit)

    def _return_from_visit(self) -> None:
        visit = self._end_visit()
        logger.info(
            "Announcement over: back to %04X on channel %s", visit.origin_sid, visit.origin_channel
        )
        self._reported = None
        self._revisit_timer.start()
        self._play(visit.origin_channel, visit.origin_sid)

    def _stay_from_visit(self) -> None:
        """The user chose a channel or service: the announcement visited no longer
        leads back."""
        self._end_visit()

    def _end_visit(self) -> AnnouncementVisit | None:
        """Forgets the visit, if there is one, and returns it."""
        visit = self._visit
        if visit is not None:
            self._visit = None
            self._visit_timer.stop()
            self.announcement_visit_changed.emit(None)
        return visit

    def _on_visit_search_timeout(self) -> None:
        if self._visit is not None and not self._visit_heard:
            logger.info("Announcement not found on channel %s", self._visit.channel)
            self._return_from_visit()

    def _on_revisit_timeout(self) -> None:
        """The ensemble came back to has had the time to report the announcement gone to
        again; if it no longer does, it is over, and the next one leads there again."""
        if self._reported != self._visited:
            self._visited = None

    def _channel_of_ensemble(self, eid: int) -> str | None:
        """The channel to hear an ensemble on: one whose stored list came from it
        (receivable here, then), else one FIG 0/21 of the tuned ensemble names for it
        (see `dab.service_following.locate_ensemble`)."""
        return locate_ensemble(eid, self._ensemble_cache.entries(), self._live_frequencies).channel

    def _oe_source_label(self, status: AnnouncementStatus) -> str | None:
        """The name of the service carrying another ensemble's announcement, from the
        stored list of that ensemble's channel; None for an alarm or if unknown."""
        if status.oe_active.cluster_id == ALARM_CLUSTER:
            return None
        channel = self._channel_of_ensemble(status.oe_eid)
        info = None if channel is None else self._ensemble_cache.get(channel)
        return None if info is None else dict(info.services).get(status.oe_active.target_sid)

    # --- Data services ----------------------------------------------------------------

    @property
    def data_components(self) -> list[DataComponent]:
        """The data components of the received ensemble, as far as the FIC describes
        them; empty without reception."""
        return list(self._data_components)

    def is_data_component_decoding(self, scid: int) -> bool:
        """The signal processing decodes it (confirmed, not only asked for)."""
        return scid in self._data_decoding

    def is_data_component_switched_off(self, component: DataComponent) -> bool:
        return self._data_key(component) in self._data_switched_off

    def set_data_component_enabled(self, scid: int, enabled: bool) -> None:
        """Starts or stops decoding a packet mode data component besides the audio.

        Every decodable component is started as soon as the FIC describes it; one
        stopped here stays off on this channel until it is started again, also after a
        restart.
        """
        component = next((c for c in self._data_components if c.scid == scid), None)
        if component is not None:
            key = self._data_key(component)
            if enabled:
                self._data_switched_off.discard(key)
            else:
                self._data_switched_off.add(key)
            self._settings.setValue(_SETTINGS_DATA_SWITCHED_OFF, sorted(self._data_switched_off))
        if enabled:
            self._data_requested.add(scid)
        else:
            self._data_requested.discard(scid)
        self._data_component_toggled.emit(scid, enabled)
        self.data_components_changed.emit()

    def set_all_data_components_enabled(self, enabled: bool) -> None:
        for component in self._data_components:
            if component.decodable:
                self.set_data_component_enabled(component.scid, enabled)

    def _data_key(self, component: DataComponent) -> str:
        """What a component switched off is remembered by: its channel, service and
        SCId. The SCId alone is only unique within the ensemble."""
        channel = self._list_channel or "manuell"
        return f"{channel}/{component.sid:X}/{component.scid}"

    def _on_data_services_updated(self, components: list[DataComponent]) -> None:
        """Starts every decodable component that is not switched off: decoding them all
        costs little compute time next to the audio (a programme guide at 24 kbit/s and
        GEPOS at 8 kbit/s together some 2 % of real time)."""
        if not self._list_is_live or self._awaited_tuning is not None:
            return  # a straggler of the previous frequency, see `_on_services_updated`
        self._data_components = list(components)
        for component in self._data_components:
            if (
                component.decodable
                and component.scid not in self._data_requested
                and self._data_key(component) not in self._data_switched_off
            ):
                self._data_requested.add(component.scid)
                self._data_component_toggled.emit(component.scid, True)
        self.data_components_changed.emit()

    def _on_data_decoding_ready(self, statuses: tuple[DataComponentStatus, ...]) -> None:
        if self._awaited_tuning is not None:
            return
        self._data_decoding = {status.scid for status in statuses}
        self._decoded_spi = next(
            (
                status.scid
                for status in statuses
                if status.component is not None
                and USER_APPLICATION_SPI in status.component.app_types
            ),
            None,
        )
        self.data_components_changed.emit()

    # --- Programme guide --------------------------------------------------------------

    def is_received_guide(self, scid: int) -> bool:
        """The component is the programme guide (SPI) of the ensemble received, being
        decoded now; what its carousel completes belongs to `list_channel`."""
        return scid == self._decoded_spi and self._list_is_live and self._awaited_tuning is None

    @property
    def guide_scid(self) -> int | None:
        """SCId of the programme guide being decoded, None if none is."""
        return self._decoded_spi

    def _guide_channel(self, scid: int) -> str | None:
        """The channel whose guide the object of `scid` belongs to; None if it is not
        the guide being decoded, or with a manual frequency (nothing is stored)."""
        return self._list_channel if self.is_received_guide(scid) else None

    def _on_data_object_ready(self, scid: int, carousel_object: CarouselObject) -> None:
        """Stores what the guide's carousel completes, so that the guide is there right
        away the next time the channel comes up, on every surface (`dab.guide_cache`)."""
        channel = self._guide_channel(scid)
        if channel is None:
            return
        mot_object = carousel_object.mot_object
        services = mot_object.is_image or isinstance(carousel_object.spi, ServiceInformation)
        if not services and not isinstance(carousel_object.spi, Schedule):
            return  # nothing a guide takes (see `dab.spi.ProgrammeGuide.add`)
        now = time.time()
        try:
            written = self._guide_cache.store(channel, mot_object, now)
            self._guide_cache.store_labels(channel, dict(self._live_services))
        except OSError as exc:
            logger.warning("Guide of %s not stored: %s", channel, exc)
            self.guide_store_failed.emit(str(exc))
            return
        if written:
            self.guide_stored.emit(channel, now, services)

    def _on_mot_directory_ready(self, scid: int, directory: MotDirectory) -> None:
        """A new version of the carousel: what it no longer lists is outdated, in the
        stored guide as well."""
        channel = self._guide_channel(scid)
        if channel is None:
            return
        names = {entry.header.content_name for entry in directory.entries}
        try:
            self._guide_cache.retain(channel, {name for name in names if name}, time.time())
        except OSError as exc:
            logger.warning("Guide of %s not stored: %s", channel, exc)
            self.guide_store_failed.emit(str(exc))

    # --- Compute load -----------------------------------------------------------------

    def _update_process_load(self) -> None:
        """CPU load of the whole program across all threads.

        `time.process_time()` sums the compute time of all threads; divided by the
        elapsed wall-clock time that gives the utilisation in cores. Needs no extra
        dependency.
        """
        cpu_time = time.process_time()
        wall_time = time.monotonic()
        elapsed = wall_time - self._last_wall_time
        if elapsed > 0:
            self._process_load = (cpu_time - self._last_cpu_time) / elapsed
        self._last_cpu_time = cpu_time
        self._last_wall_time = wall_time
        self._emit_load()

    def _on_load_ready(self, load: float, compute: float) -> None:
        if self._worker is None:
            # A straggler from blocks still queued when reception ended.
            return
        self._dsp_load = load
        # Never below zero: on Windows the CPU time of a thread is counted in ticks and
        # can exceed the wall clock by one.
        self._dsp_waiting = max(0.0, load - compute)
        self._emit_load()

    def _emit_load(self) -> None:
        self.load_changed.emit(self._dsp_load, self._dsp_waiting, self._process_load)


def volume_text(db: float | None, device_percent: int | None = None) -> str:
    """The volume for display: "−18 dB", "0 dB", "muted" at 0 %; for an output with a
    volume of its own its share of that range, "40 %"."""
    if db is None:
        # Translators: the volume at 0 %, where a level in dB stands otherwise.
        return tr("muted")
    if device_percent is not None:
        # Translators: the volume of a Bluetooth device as a share of its own range.
        return tr("{percent} %").format(percent=device_percent)
    return f"{round(db)} dB".replace("-", "−")


def _clamp[T: float](value: T, low: T, high: T) -> T:
    return min(max(value, low), high)


def app_data_directory() -> Path:
    """The application's data directory (under Linux ~/.local/share/dab-radio/, under
    Windows %APPDATA%\\dab-radio)."""
    location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    return Path(location)
