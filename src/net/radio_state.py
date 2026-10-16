"""The state of the radio for programs outside it, and the commands they may give it:
one model for every transport, MQTT first (`net.mqtt`), HTTP perhaps later.

`RadioStateModel` is a surface without widgets on
`ui.receiver_controller.ReceiverController`: it follows the controller's signals, and
those of the signal processing and the audio output the controller provides, and keeps
what a smart home would ask in sections (`SECTIONS`) of plain values: JSON-ready dicts
of str, int, float, bool, None, lists and dicts, SIds as four hex digits as the
surfaces show them. `changed` names a section whenever its content changes; `section`
and `snapshot` read them. The slide of the slideshow is binary and stands apart
(`slide`, reported as `changed("slide")`).

Sections that change many times a second (the figures of the signal, the position of a
recording playing) are reported at most once per `THROTTLE_MS`, the first change at
once; everything else immediately.

Commands come as a name and JSON-ready arguments (`COMMANDS`). `execute` runs one in
the GUI thread and answers None, or a text saying why not (in English), for
whoever asked. A transport thread calls `submit` instead: that queues the command into
the GUI thread, and `command_finished` reports the answer.

The remote interfaces speak English whatever language the surfaces show: the texts in
the sections (the status line, the recorder's message, programme and announcement types,
the reception quality) are the English of what the surfaces show (`i18n.english`), the
answers to commands are English only. Broadcast content (labels, Dynamic Label, titles)
comes as sent.

Build the model before `controller.start()`, as a surface: the first service list
comes from there, and the controller offers no accessor for the list on display.

Notifications (`notify`) go to `ui.notifications.NotificationCenter` of the controller,
which the touch surface shows; the open ones are the section `notifications`, and
`notification_answered` reports how each one ended.
"""

from __future__ import annotations

import base64
import logging
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage

from audio.outputs import AudioOutput
from dab.announcement import ALARM_CLUSTER, AnnouncementStatus, announcement_types
from dab.channels import BAND_III_CHANNELS
from dab.dls import DynamicLabel
from dab.fig import Announcement, OeAnnouncement
from dab.mot import MotObject
from dab.programme_type import programme_type_name
from dab.slideshow import SlideStore
from dab.timecode import BroadcastTime, ReferenceClock
from dsp.mp2_frame import Mp2FrameResult
from dsp.pipeline import SubchannelStatus
from dsp.superframe import SuperframeResult
from i18n import english
from recording.playback import PlaybackState
from recording.recorder import RecorderStatus, Recording
from ui.notifications import (
    BUTTON_STYLES,
    MAX_BUTTONS,
    MAX_ID_LENGTH,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_SIDE,
    MAX_LABEL_LENGTH,
    MAX_MESSAGE_LENGTH,
    MAX_TIMEOUT_S,
    MAX_TITLE_LENGTH,
    Level,
    Notification,
    NotificationAnswer,
    NotificationButton,
)
from ui.touch.quality import ReceptionQuality

if TYPE_CHECKING:
    from ui.receiver_controller import AnnouncementVisit, ReceiverController

logger = logging.getLogger(__name__)

SECTIONS = (
    "receiver",
    "services",
    "service",
    "now_playing",
    "audio",
    "signal",
    "announcement",
    "favourites",
    "recorder",
    "recordings",
    "playback",
    "notifications",
)
# Sections reported at most this often; see the module docstring.
THROTTLE_MS = 1000
# The slideshow's trigger and expiry times are checked this often, as on the radio page.
SLIDE_TICK_MS = 1000
# MIME types of the slideshow's image formats (`dab.mot.IMAGE_SUBTYPES`).
_MIME_TYPES = {0: "image/gif", 1: "image/jpeg", 2: "image/bmp", 3: "image/png"}
_UNKNOWN_MIME_TYPE = "application/octet-stream"
# The channels in the order of Band III, for sorting favourites.
_CHANNEL_ORDER = {channel: index for index, channel in enumerate(BAND_III_CHANNELS)}
# What the audio frames of the station are counted in (`_FrameTally`).
_SUPERFRAMES = "Superframes"
_MP2_FRAMES = "MP2 frames"

Arguments = dict[str, Any]


@dataclass(frozen=True)
class SlideImage:
    """The image the station shows right now."""

    data: bytes
    content_type: str  # MIME type
    name: str | None  # the ContentName of the MOT object


class CommandRejected(Exception):
    """A command cannot be carried out; the text says why, in English."""


COMMANDS = (
    "select_channel",
    "select_service",
    "play",
    "start_reception",
    "stop_reception",
    "start_scan",
    "cancel_scan",
    "set_volume",
    "set_muted",
    "set_output",
    "set_equalizer_enabled",
    "set_favourite",
    "set_announcement_switching",
    "return_from_announcement",
    "set_recording_enabled",
    "keep_recording",
    "delete_recording",
    "play_recording",
    "pause_playback",
    "resume_playback",
    "stop_playback",
    "seek_playback",
    "notify",
    "dismiss_notification",
)

# Texts longer than this (an image in base64 above all) appear in logs and in the echo
# of a command by their length; a message (`MAX_MESSAGE_LENGTH`) stays whole.
BRIEF_LENGTH = 2 * MAX_MESSAGE_LENGTH
# How an image may come besides plain base64: as a data URL.
_DATA_URL = re.compile(r"^data:[\w/+.-]*;base64,", re.IGNORECASE)

# By the name of the `ui.receiver_controller.ReceptionState`, which this module does
# not import: the controller imports this one.
_BUSY = {
    "CONNECTING": "Reception is just starting.",
    "SCANNING": "The scan is running.",
}


# --- Arguments --------------------------------------------------------------------------


def _hex(sid: int | None) -> str | None:
    return None if sid is None else f"{sid:04X}"


def _missing(key: str) -> CommandRejected:
    return CommandRejected(f"The argument “{key}” is missing.")


def _not_a_number(key: str, value: Any) -> CommandRejected:
    return CommandRejected(f"“{key}” must be a number, not {value!r}.")


def _require(arguments: Arguments, key: str) -> Any:
    if key not in arguments or arguments[key] is None:
        raise _missing(key)
    return arguments[key]


def _bool_argument(arguments: Arguments, key: str) -> bool:
    value = _require(arguments, key)
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "on", "1", "ja", "an"):
            return True
        if lowered in ("false", "off", "0", "nein", "aus"):
            return False
    raise CommandRejected(f"“{key}” must be true or false, not {value!r}.")


def _int_argument(arguments: Arguments, key: str, low: int, high: int) -> int:
    value = _require(arguments, key)
    try:
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            raise TypeError
        number = int(value)
    except ValueError, TypeError:
        raise _not_a_number(key, value) from None
    if not low <= number <= high:
        raise CommandRejected(f"“{key}” must be between {low} and {high}.")
    return number


def _float_argument(arguments: Arguments, key: str, default: float | None = None) -> float:
    value = arguments.get(key)
    if value is None:
        if default is None:
            raise _missing(key)
        return default
    try:
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            raise TypeError
        number = float(value)
    except ValueError, TypeError:
        raise _not_a_number(key, value) from None
    if number < 0:
        raise CommandRejected(f"“{key}” must not be negative.")
    return number


def _text_argument(arguments: Arguments, key: str) -> str:
    value = _require(arguments, key)
    if not isinstance(value, str):
        raise CommandRejected(f"“{key}” must be a string, not {value!r}.")
    return value


def _channel_argument(arguments: Arguments) -> str:
    channel = _text_argument(arguments, "channel").strip().upper()
    if channel not in BAND_III_CHANNELS:
        raise CommandRejected(f"Unknown channel “{channel}”.")
    return channel


def _not_a_sid(value: Any) -> CommandRejected:
    return CommandRejected(f"“sid” must be an SId, not {value!r}.")


def _sid_argument(value: Any) -> int:
    """A SId as the surfaces show it ("D210"), also with "0x", or as a number."""
    if isinstance(value, bool):
        raise _not_a_sid(value)
    if isinstance(value, int):
        sid = value
    elif isinstance(value, str):
        try:
            sid = int(value.strip().removeprefix("0x").removeprefix("0X"), 16)
        except ValueError:
            raise _not_a_sid(value) from None
    else:
        raise _not_a_sid(value)
    if not 0 <= sid <= 0xFFFFFFFF:
        raise _not_a_sid(value)
    return sid


def _normalised(name: str) -> str:
    return " ".join(name.split()).casefold()


def brief_arguments(arguments: Any) -> Any:
    """`arguments` for a log or the echo of a command: a text longer than
    `BRIEF_LENGTH` by its start and length."""
    if isinstance(arguments, dict):
        return {key: brief_arguments(value) for key, value in arguments.items()}
    if isinstance(arguments, list):
        return [brief_arguments(value) for value in arguments]
    if isinstance(arguments, str) and len(arguments) > BRIEF_LENGTH:
        return f"{arguments[:40]}… ({len(arguments)} characters)"
    return arguments


def _limited_text(arguments: Arguments, key: str, limit: int, default: str = "") -> str:
    """An optional text without the blanks at its ends, at most `limit` characters."""
    value = arguments.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise CommandRejected(f"“{key}” must be a string, not {value!r}.")
    text = value.strip()
    if len(text) > limit:
        raise CommandRejected(f"“{key}” must not be longer than {limit} characters.")
    return text


def _buttons_argument(arguments: Arguments) -> tuple[NotificationButton, ...]:
    """The buttons of a notification: a list of objects with "label" and optional
    "id" and "style", or of labels alone; a button without an id has its position
    ("1", "2" …)."""
    value = arguments.get("buttons")
    if value is None:
        return ()
    if not isinstance(value, list):
        raise CommandRejected("“buttons” must be a list.")
    if len(value) > MAX_BUTTONS:
        raise CommandRejected(f"A notification has at most {MAX_BUTTONS} buttons.")
    buttons: list[NotificationButton] = []
    for position, entry in enumerate(value, start=1):
        if isinstance(entry, str):
            entry = {"label": entry}
        if not isinstance(entry, dict):
            raise CommandRejected(f"Button {position} must be an object or a label.")
        label = _limited_text(entry, "label", MAX_LABEL_LENGTH)
        if not label:
            raise CommandRejected(f"Button {position} has no “label”.")
        button_id = _limited_text(entry, "id", MAX_ID_LENGTH, str(position)) or str(position)
        style = _limited_text(entry, "style", MAX_ID_LENGTH).lower()
        if style not in BUTTON_STYLES:
            choices = ", ".join(f"“{s}”" for s in BUTTON_STYLES if s)
            raise CommandRejected(f"“style” must be {choices} or empty, not “{style}”.")
        if any(button.id == button_id for button in buttons):
            raise CommandRejected(f"Two buttons have the id “{button_id}”.")
        buttons.append(NotificationButton(button_id, label, style))
    return tuple(buttons)


def _level_argument(arguments: Arguments) -> Level:
    text = _limited_text(arguments, "level", MAX_ID_LENGTH).lower()
    if not text:
        return Level.INFO
    try:
        return Level(text)
    except ValueError:
        choices = ", ".join(f"“{level.value}”" for level in Level)
        raise CommandRejected(f"“level” must be {choices}, not “{text}”.") from None


def _timeout_argument(arguments: Arguments) -> float | None:
    """Seconds until the notification closes by itself; None (also for 0) for never."""
    if arguments.get("timeout") is None:
        return None
    seconds = _float_argument(arguments, "timeout")
    if seconds > MAX_TIMEOUT_S:
        raise CommandRejected(f"“timeout” must be at most {MAX_TIMEOUT_S} seconds.")
    return seconds or None


def _image_argument(arguments: Arguments) -> QImage | None:
    """The image of a notification, sent as base64 (or as a data URL), read and scaled
    down to `MAX_IMAGE_SIDE`."""
    value = arguments.get("image")
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise CommandRejected("“image” must be a string of base64.")
    encoded = "".join(_DATA_URL.sub("", value.strip(), count=1).split())
    if len(encoded) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
        raise CommandRejected(f"“image” must not be larger than {MAX_IMAGE_BYTES} bytes.")
    try:
        data = base64.b64decode(encoded, validate=True)
    except ValueError:  # binascii.Error among them
        raise CommandRejected("“image” is not valid base64.") from None
    image = QImage.fromData(data)
    if image.isNull():
        raise CommandRejected("“image” is not an image (JPEG, PNG, GIF or BMP).")
    if max(image.width(), image.height()) > MAX_IMAGE_SIDE:
        image = image.scaled(
            MAX_IMAGE_SIDE,
            MAX_IMAGE_SIDE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    return image


def _service_argument(
    arguments: Arguments, services: list[tuple[int, str]], *, strict: bool
) -> int:
    """The service meant by "sid" or "name" among `services`; a SId not in there is
    rejected if `strict`, a name always."""
    if "sid" in arguments and arguments["sid"] is not None:
        sid = _sid_argument(arguments["sid"])
        if strict and all(listed != sid for listed, _ in services):
            raise CommandRejected(f"Station {_hex(sid)} is not in the station list.")
        return sid
    if "name" in arguments and arguments["name"] is not None:
        name = _text_argument(arguments, "name")
        wanted = _normalised(name)
        for sid, label in services:
            if _normalised(label) == wanted:
                return sid
        raise CommandRejected(f"Station “{name}” is not in the station list.")
    raise CommandRejected("The argument “sid” or “name” is missing.")


# --- Values -----------------------------------------------------------------------------


def _output_json(output: AudioOutput) -> dict:
    return {
        "id": output.id,
        "name": output.name,
        "bluetooth": output.bluetooth,
        "default": output.default,
    }


def _recording_json(recording: Recording) -> dict:
    return {
        "id": recording.id,
        "name": english(recording.name),
        "station": recording.station,
        "ensemble": recording.ensemble,
        "channel": recording.channel,
        "sid": _hex(recording.sid),
        "title": recording.title,
        "artist": recording.artist,
        "text": recording.text,
        "start": recording.start,
        "duration": recording.duration,
        "size": recording.size,
        "lost": recording.lost,
        "codec": recording.codec,
        "bitrate_kbps": recording.bitrate,
        "kept": recording.kept,
        "complete": recording.complete,
    }


def _announcement_json(announcement: Announcement | OeAnnouncement, source: str | None) -> dict:
    return {
        "types": [english(name.short) for name in announcement_types(announcement.flags)],
        "alarm": announcement.cluster_id == ALARM_CLUSTER,
        "new": announcement.new,
        "source": source,
    }


def _notification_json(notification: Notification) -> dict:
    return {
        "id": notification.id,
        "title": notification.title,
        "message": notification.message,
        "level": notification.level.value,
        "buttons": [
            {"id": button.id, "label": button.label, "style": button.style}
            for button in notification.buttons
        ],
        "image": notification.image is not None,
        "received": notification.received,
        "expires": notification.expires,
    }


def _programme_type(code: int | None) -> str | None:
    if code is None:
        return None
    name = programme_type_name(code)
    return None if name is None else english(name.full)


class _FrameTally:
    """The audio frames of the station playing since it was chosen, as the figures of
    the signal page count them (`ui.touch.signal_panels.FrameTally`)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.unit: str | None = None
        self.synced = 0
        self.total = 0
        self.lost = 0
        self.rs_corrections = 0

    def add_superframe(self, result: SuperframeResult) -> None:
        self.unit = _SUPERFRAMES
        self.total += 1
        self.synced += result.sync_ok
        self.rs_corrections += result.rs_corrections
        self.lost += bool(result.rs_uncorrectable_blocks)

    def add_mp2_frame(self, result: Mp2FrameResult) -> None:
        self.unit = _MP2_FRAMES
        self.total += 1
        self.synced += result.sync_ok
        self.lost += result.sync_ok and not result.crc_ok

    def json(self) -> dict:
        return {
            "unit": self.unit,
            "synced": self.synced,
            "total": self.total,
            "lost": self.lost,
            "rs_corrections": self.rs_corrections,
        }


# --- The model --------------------------------------------------------------------------


class RadioStateModel(QObject):
    """See the module docstring."""

    changed = pyqtSignal(str)  # the name of the section (or "slide") whose content changed
    # A submitted command has run: its name, its arguments and the answer, None if it
    # was carried out, else a text saying why not (in English).
    command_finished = pyqtSignal(str, object, object)
    # A notification ended (`ui.notifications.NotificationAnswer.json`): the button
    # pressed, or why not.
    notification_answered = pyqtSignal(object)  # dict
    _submitted = pyqtSignal(str, object)

    def __init__(
        self,
        controller: ReceiverController,
        parent: QObject | None = None,
        throttle_ms: int = THROTTLE_MS,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._sections: dict[str, dict] = {}
        self._pending: set[str] = set()
        self._throttle = QTimer(self)
        self._throttle.setSingleShot(True)
        self._throttle.setInterval(throttle_ms)
        self._throttle.timeout.connect(self._flush)

        # What the sections are built from beyond the controller's own accessors.
        self._status = ""
        self._error: str | None = None
        self._scan: dict | None = None
        self._ensemble_label: str | None = None
        self._eid: int | None = None
        self._services: list[tuple[int, str]] = []
        self._programme_types: dict[int, int] = {}
        self._list_complete = False
        self._subchannel: SubchannelStatus | None = None
        self._audio_format: str | None = None
        self._label: DynamicLabel | None = None
        self._outputs: list[AudioOutput] = controller.audio_outputs
        self._clock = ReferenceClock()
        self._slides = SlideStore()
        self._slide: SlideImage | None = None
        self._quality = ReceptionQuality()
        self._frames = _FrameTally()
        self._offset_hz: float | None = None
        self._clipping: float | None = None
        self._sfn: tuple[float, float] | None = None
        self._load: tuple[float, float, float] | None = None
        self._announcement: tuple[AnnouncementStatus | None, str | None] = (None, None)

        self._slide_timer = QTimer(self)
        self._slide_timer.setInterval(SLIDE_TICK_MS)
        self._slide_timer.timeout.connect(self._on_slide_tick)
        self._slide_timer.start()

        self._commands: dict[str, Callable[[Arguments], None]] = {
            name: getattr(self, f"_command_{name}") for name in COMMANDS
        }
        self._submitted.connect(self._on_submitted)
        self._connect(controller)
        for name in SECTIONS:
            self._sections[name] = self._build(name)

    def _connect(self, controller: ReceiverController) -> None:
        controller.state_changed.connect(self._on_state)
        controller.status_changed.connect(self._on_status)
        controller.error_occurred.connect(self._on_error)
        controller.tuning_changed.connect(lambda *_: self._refresh("receiver"))
        controller.device_settings_changed.connect(lambda: self._refresh("receiver"))
        controller.scan_progress.connect(self._on_scan_progress)
        controller.service_list_changed.connect(self._on_service_list)
        controller.service_list_completed.connect(self._on_service_list)
        controller.frequency_information_received.connect(self._on_frequency_information)
        controller.live_reset.connect(self._on_live_reset)
        controller.service_reset.connect(self._on_service_reset)
        controller.selected_service_changed.connect(lambda _sid: self._refresh("service"))
        controller.favourites_changed.connect(self._on_favourites)
        controller.channel_stored.connect(lambda _channel: self._refresh("favourites"))
        controller.volume_changed.connect(lambda: self._refresh("audio"))
        controller.audio_enabled_changed.connect(lambda _enabled: self._refresh("audio"))
        controller.audio_outputs_changed.connect(self._on_audio_outputs)
        controller.equalizer_changed.connect(lambda: self._refresh("audio"))
        controller.load_changed.connect(self._on_load)
        controller.announcement_changed.connect(self._on_announcement)
        controller.announcement_visit_changed.connect(lambda _v: self._refresh("announcement"))
        controller.announcement_switching_changed.connect(
            lambda _enabled: self._refresh("announcement")
        )
        controller.recordings_changed.connect(self._on_recorder)
        controller.recorder_status_changed.connect(lambda _status: self._on_recorder())
        controller.recording_directory_changed.connect(lambda _d: self._on_recorder())
        controller.recording_enabled_changed.connect(lambda _enabled: self._on_recorder())
        controller.playback_changed.connect(self._on_playback)
        notifications = controller.notifications
        notifications.changed.connect(lambda: self._refresh("notifications"))
        notifications.answered.connect(self._on_notification_answered)

        dsp = controller.dsp_worker
        dsp.subchannel_info_ready.connect(self._on_subchannel_info)
        dsp.dynamic_label_ready.connect(self._on_dynamic_label)
        dsp.slide_ready.connect(self._on_slide)
        dsp.date_time_ready.connect(self._on_date_time)
        dsp.superframe_ready.connect(self._on_superframe)
        dsp.mp2_frame_ready.connect(self._on_mp2_frame)
        dsp.frequency_offset_ready.connect(self._on_frequency_offset)
        dsp.clipping_ready.connect(self._on_clipping)
        dsp.sfn_share_ready.connect(self._on_sfn_share)
        controller.audio_player.format_changed.connect(self._on_audio_format)

    # --- Reading --------------------------------------------------------------------

    def section(self, name: str) -> dict:
        """The content of a section (`SECTIONS`), JSON-ready."""
        return self._sections[name]

    def snapshot(self) -> dict[str, dict]:
        """Every section, JSON-ready."""
        return dict(self._sections)

    @property
    def slide(self) -> SlideImage | None:
        """The image of the slideshow on display, None without one."""
        return self._slide

    # --- Commands -------------------------------------------------------------------

    def execute(self, name: str, arguments: Arguments | None = None) -> str | None:
        """Carries out a command in the calling thread, which must be the GUI thread;
        None once done, else a text saying why not (in English)."""
        handler = self._commands.get(name)
        if handler is None:
            return f"Unknown command “{name}”."
        try:
            handler(dict(arguments or {}))
        except CommandRejected as exc:
            return str(exc)
        return None

    def submit(self, name: str, arguments: Arguments | None = None) -> None:
        """Hands a command to the GUI thread from any thread; `command_finished`
        reports the answer."""
        self._submitted.emit(name, dict(arguments or {}))

    def _on_submitted(self, name: str, arguments: Arguments) -> None:
        answer = self.execute(name, arguments)
        if answer is not None:
            logger.info("command %s %r rejected: %s", name, brief_arguments(arguments), answer)
        self.command_finished.emit(name, arguments, answer)

    def _not_busy(self) -> None:
        reason = _BUSY.get(self._controller.state.name)
        if reason is not None:
            raise CommandRejected(reason)

    def _command_select_channel(self, arguments: Arguments) -> None:
        self._not_busy()
        self._controller.select_channel(_channel_argument(arguments))

    def _command_select_service(self, arguments: Arguments) -> None:
        """Selects a service of the list on display, as a tap on it would."""
        self._not_busy()
        self._controller.select_service(
            _service_argument(arguments, self._listed()[1], strict=True)
        )

    def _command_play(self, arguments: Arguments) -> None:
        """Plays a service: tunes to its channel (the one given, else the one on
        display), selects it and starts reception if it is not running."""
        self._not_busy()
        controller = self._controller
        if arguments.get("channel") is None:
            sid = _service_argument(arguments, self._listed()[1], strict=True)
            channel = controller.list_channel
        else:
            channel = _channel_argument(arguments)
            info = controller.ensemble_cache.get(channel)
            stored = [] if info is None else list(info.services)
            # A SId the stored list does not know may still be on air there.
            sid = _service_argument(arguments, stored, strict=False)
        controller.play_service(channel, sid)

    def _command_start_reception(self, _arguments: Arguments) -> None:
        if self._controller.state.name == "RECEIVING":
            raise CommandRejected("Reception is already running.")
        self._not_busy()
        self._controller.start_reception()

    def _command_stop_reception(self, _arguments: Arguments) -> None:
        if not self._controller.is_receiving:
            raise CommandRejected("Reception is not running.")
        self._controller.stop_reception()

    def _command_start_scan(self, _arguments: Arguments) -> None:
        if self._controller.state.name == "RECEIVING":
            raise CommandRejected("Reception is running; stop it before scanning.")
        self._not_busy()
        self._controller.start_scan()

    def _command_cancel_scan(self, _arguments: Arguments) -> None:
        if self._controller.state.name != "SCANNING":
            raise CommandRejected("No scan is running.")
        self._controller.cancel_scan()

    def _command_set_volume(self, arguments: Arguments) -> None:
        self._controller.set_volume_percent(_int_argument(arguments, "percent", 0, 100))

    def _command_set_muted(self, arguments: Arguments) -> None:
        self._controller.set_audio_enabled(not _bool_argument(arguments, "muted"))

    def _command_set_output(self, arguments: Arguments) -> None:
        """Plays to the output with this id, to the system's default with ""."""
        choice = _text_argument(arguments, "id")
        if choice and all(output.id != choice for output in self._controller.audio_outputs):
            raise CommandRejected(f"Unknown output “{choice}”.")
        self._controller.set_audio_output(choice)

    def _command_set_equalizer_enabled(self, arguments: Arguments) -> None:
        self._controller.set_equalizer_enabled(_bool_argument(arguments, "enabled"))

    def _command_set_favourite(self, arguments: Arguments) -> None:
        """Marks a station (a service on a channel, the one on display unless given)."""
        controller = self._controller
        if arguments.get("channel") is None:
            channel = controller.list_channel
            if channel is None:
                raise CommandRejected("No channel selected; please give “channel”.")
        else:
            channel = _channel_argument(arguments)
        info = controller.ensemble_cache.get(channel)
        stored = [] if info is None else list(info.services)
        services = self._listed()[1] if channel == controller.list_channel else stored
        sid = _service_argument(arguments, services, strict=False)
        controller.set_favourite(channel, sid, _bool_argument(arguments, "favourite"))

    def _command_set_announcement_switching(self, arguments: Arguments) -> None:
        self._controller.set_announcement_switching(_bool_argument(arguments, "enabled"))

    def _command_return_from_announcement(self, _arguments: Arguments) -> None:
        if self._controller.announcement_visit is None:
            raise CommandRejected("Reception is not at an announcement of another ensemble.")
        self._controller.return_from_announcement()

    def _command_set_recording_enabled(self, arguments: Arguments) -> None:
        self._controller.set_recording_enabled(_bool_argument(arguments, "enabled"))

    def _recording_argument(self, arguments: Arguments, *, kept: bool, current: bool) -> str:
        """The id of a title of the ring, of the kept ones if `kept`, of the one
        recording if `current`."""
        recording_id = _text_argument(arguments, "id")
        controller = self._controller
        known = {recording.id for recording in controller.recordings}
        if kept:
            known |= {recording.id for recording in controller.kept_recordings}
        running = controller.recorder_status.current
        if current and running is not None:
            known.add(running.id)
        if recording_id not in known:
            raise CommandRejected(f"Unknown recording “{recording_id}”.")
        return recording_id

    def _command_keep_recording(self, arguments: Arguments) -> None:
        self._controller.keep_recording(
            self._recording_argument(arguments, kept=False, current=True)
        )

    def _command_delete_recording(self, arguments: Arguments) -> None:
        self._controller.delete_recording(
            self._recording_argument(arguments, kept=True, current=False)
        )

    def _command_play_recording(self, arguments: Arguments) -> None:
        recording_id = self._recording_argument(arguments, kept=True, current=False)
        self._controller.play_recording(recording_id, _float_argument(arguments, "position", 0.0))

    def _playing(self) -> None:
        if self._controller.playback is None:
            raise CommandRejected("No recording is playing.")

    def _command_pause_playback(self, _arguments: Arguments) -> None:
        self._playing()
        self._controller.pause_playback()

    def _command_resume_playback(self, _arguments: Arguments) -> None:
        self._playing()
        self._controller.resume_playback()

    def _command_stop_playback(self, _arguments: Arguments) -> None:
        self._playing()
        self._controller.stop_playback()

    def _command_seek_playback(self, arguments: Arguments) -> None:
        self._playing()
        self._controller.seek_playback(_float_argument(arguments, "position"))

    def _command_notify(self, arguments: Arguments) -> None:
        """Shows a notification until a button is pressed, its time runs out or it is
        taken back (`ui.notifications`)."""
        title = _limited_text(arguments, "title", MAX_TITLE_LENGTH)
        message = _limited_text(arguments, "message", MAX_MESSAGE_LENGTH)
        if not title and not message:
            raise CommandRejected("A notification needs a “title” or a “message”.")
        notification = Notification(
            id=_limited_text(arguments, "id", MAX_ID_LENGTH) or secrets.token_hex(4),
            title=title,
            message=message,
            buttons=_buttons_argument(arguments),
            level=_level_argument(arguments),
            timeout_s=_timeout_argument(arguments),
            image=_image_argument(arguments),
        )
        try:
            self._controller.notifications.post(notification)
        except ValueError as exc:
            raise CommandRejected(str(exc)) from None

    def _command_dismiss_notification(self, arguments: Arguments) -> None:
        notification_id = _text_argument(arguments, "id").strip()
        if not self._controller.notifications.dismiss(notification_id):
            raise CommandRejected(f"No notification “{notification_id}” is open.")

    # --- Sections -------------------------------------------------------------------

    def _build(self, name: str) -> dict:
        return getattr(self, f"_build_{name}")()

    def _refresh(self, *names: str, throttle: bool = False) -> None:
        for name in names:
            self._set(name, self._build(name), throttle=throttle)

    def _set(self, name: str, value: dict, *, throttle: bool = False) -> None:
        if self._sections.get(name) == value:
            return
        self._sections[name] = value
        if throttle and self._throttle.isActive():
            self._pending.add(name)
            return
        self._pending.discard(name)
        self.changed.emit(name)
        if throttle:
            self._throttle.start()

    def _flush(self) -> None:
        pending, self._pending = self._pending, set()
        for name in sorted(pending):
            self.changed.emit(name)
        if pending:
            self._throttle.start()

    def _listed(self) -> tuple[str | None, list[tuple[int, str]], dict[int, int]]:
        """The ensemble label, services and programme types on display: the received
        list, or, while it is still filling, the stored one of the channel, which the
        surfaces keep showing until then (`ReceiverController.live_list_complete`)."""
        controller = self._controller
        if controller.list_is_live and not self._list_complete:
            channel = controller.list_channel
            info = None if channel is None else controller.ensemble_cache.get(channel)
            if info is not None:
                return info.label, list(info.services), dict(info.programme_types)
        return self._ensemble_label, self._services, self._programme_types

    def _build_receiver(self) -> dict:
        controller = self._controller
        return {
            "state": controller.state.name.lower(),
            "status": english(self._status),
            "error": english(self._error),
            "device": controller.device_name,
            "channel": controller.channel,
            "frequency_mhz": controller.frequency_mhz,
            "ensemble": self._listed()[0],
            "ensemble_id": _hex(self._eid),
            "scan": self._scan,
            "version": controller.version,
        }

    def _build_services(self) -> dict:
        controller = self._controller
        channel = controller.list_channel
        _label, services, programme_types = self._listed()
        return {
            "channel": channel,
            "live": controller.list_is_live,
            "complete": self._list_complete,
            "list": [
                {
                    "sid": _hex(sid),
                    "name": name,
                    "programme_type": _programme_type(programme_types.get(sid)),
                    "favourite": controller.is_favourite(channel, sid),
                }
                for sid, name in services
            ],
        }

    def _build_service(self) -> dict:
        controller = self._controller
        sid = controller.selected_sid
        status = self._subchannel
        _label, _services, programme_types = self._listed()
        codec = bitrate = None
        if status is not None and status.component is not None and status.subchannel is not None:
            codec = "DAB+" if status.component.is_dab_plus else "DAB"
            bitrate = status.subchannel.bitrate
        return {
            "sid": _hex(sid),
            "name": None if sid is None else controller.selected_service_label,
            "favourite": sid is not None and controller.is_favourite(controller.list_channel, sid),
            "programme_type": _programme_type(programme_types.get(sid)),
            "codec": codec,
            "bitrate_kbps": bitrate,
            "audio_format": self._audio_format,
        }

    def _build_now_playing(self) -> dict:
        label = self._label
        if label is None:
            return {"text": None, "title": None, "artist": None, "item_running": None}
        dl_plus = label.dl_plus
        running = None if dl_plus is None else dl_plus.item_running
        return {
            "text": label.text,
            "title": dl_plus.title if running else None,
            "artist": dl_plus.artist if running else None,
            "item_running": running,
        }

    def _build_audio(self) -> dict:
        controller = self._controller
        in_use = controller.audio_output_in_use
        return {
            "muted": not controller.audio_enabled,
            "volume_percent": controller.volume_percent,
            "volume_db": controller.volume_db,
            "volume_device_percent": controller.volume_device_percent,
            "volume_ceiling_db": controller.volume_ceiling_db,
            "output": None if in_use is None else _output_json(in_use),
            "output_choice": controller.audio_output_choice,
            "outputs": [_output_json(output) for output in self._outputs],
            "equalizer": {
                "enabled": controller.equalizer_enabled,
                "per_station": controller.equalizer_per_station,
                "station_own": controller.equalizer_is_station_own,
            },
        }

    def _build_signal(self) -> dict:
        sfn = self._sfn
        load = self._load
        return {
            "quality": {"bars": self._quality.bars, "text": english(self._quality.describe())},
            "frames": self._frames.json(),
            "frequency_offset_hz": self._offset_hz,
            "clipping": self._clipping,
            "sfn_others": None if sfn is None else sfn[0],
            "sfn_outside_guard": None if sfn is None else sfn[1],
            "adc_overload": self._controller.adc_overload,
            "load": None
            if load is None
            else {"dsp": load[0], "dsp_waiting": load[1], "cpu": load[2]},
        }

    def _build_announcement(self) -> dict:
        controller = self._controller
        status, source = self._announcement
        active = elsewhere = None
        if status is not None and status.active is not None:
            active = _announcement_json(status.active, source)
        if status is not None and status.oe_active is not None:
            elsewhere = _announcement_json(status.oe_active, source)
            elsewhere["ensemble_id"] = _hex(status.oe_eid)
        return {
            "active": active,
            "elsewhere": elsewhere,
            "visit": self._visit_json(controller.announcement_visit),
            "switching": controller.announcement_switching,
        }

    @staticmethod
    def _visit_json(visit: AnnouncementVisit | None) -> dict | None:
        if visit is None:
            return None
        return {
            **_announcement_json(visit.announcement, visit.label),
            "channel": visit.channel,
            "sid": _hex(visit.sid),
            "name": visit.label,
            "origin_channel": visit.origin_channel,
            "origin_sid": _hex(visit.origin_sid),
            "origin_name": visit.origin_label,
        }

    def _build_favourites(self) -> dict:
        controller = self._controller
        entries = []
        for channel, sid in controller.favourites:
            info = controller.ensemble_cache.get(channel)
            name = None if info is None else dict(info.services).get(sid)
            entries.append(
                {
                    "channel": channel,
                    "sid": _hex(sid),
                    "name": name or _hex(sid),
                    "ensemble": None if info is None else info.label,
                }
            )
        entries.sort(
            key=lambda entry: (
                _CHANNEL_ORDER.get(entry["channel"], len(_CHANNEL_ORDER)),
                _normalised(entry["name"]),
            )
        )
        return {"list": entries}

    def _build_recorder(self) -> dict:
        controller = self._controller
        status: RecorderStatus = controller.recorder_status
        directory = controller.recording_directory
        return {
            "enabled": controller.recording_enabled,
            "directory": None if directory is None else str(directory),
            "available": controller.recording_directory_available,
            "recording": status.recording,
            "message": english(status.message),
            "current": None if status.current is None else _recording_json(status.current),
            "ring_size": controller.ring_size,
        }

    def _build_recordings(self) -> dict:
        controller = self._controller
        return {
            "ring": [_recording_json(recording) for recording in controller.recordings],
            "kept": [_recording_json(recording) for recording in controller.kept_recordings],
        }

    def _build_playback(self) -> dict:
        controller = self._controller
        playback: PlaybackState | None = controller.playback
        if playback is None:
            return {"id": None, "name": None, "state": None, "position": None, "duration": None}
        recording = controller.playback_recording
        return {
            "id": playback.recording_id,
            "name": None if recording is None else english(recording.name),
            "state": playback.state,
            "position": playback.position,
            "duration": playback.duration,
        }

    def _build_notifications(self) -> dict:
        notifications = self._controller.notifications
        shown = notifications.shown
        return {
            "shown": None if shown is None else shown.id,
            "open": [_notification_json(notification) for notification in notifications.open],
        }

    # --- What the controller reports --------------------------------------------------

    def _on_state(self, state: object) -> None:
        name = getattr(state, "name", "")
        if name in ("CONNECTING", "SCANNING"):
            self._error = None
        if name != "SCANNING":
            self._scan = None
        self._refresh("receiver")

    def _on_status(self, text: str) -> None:
        self._status = text
        self._refresh("receiver")

    def _on_error(self, text: str) -> None:
        self._error = text
        self._refresh("receiver")

    def _on_scan_progress(self, channel: str, index: int, total: int) -> None:
        self._scan = {"channel": channel, "index": index, "total": total}
        self._refresh("receiver")

    def _on_service_list(self, label: object, services: object, programme_types: object) -> None:
        self._ensemble_label = label
        self._services = list(services)
        self._programme_types = dict(programme_types)
        self._list_complete = self._controller.live_list_complete
        self._refresh("receiver", "services", "service")

    def _on_frequency_information(self, eid: object, _frequencies: object) -> None:
        self._eid = eid
        self._refresh("receiver")

    def _on_live_reset(self) -> None:
        self._eid = None
        self._offset_hz = self._clipping = self._sfn = None
        self._clock.reset()
        self._refresh("receiver")
        self._refresh("signal", throttle=True)

    def _on_service_reset(self) -> None:
        self._subchannel = None
        self._audio_format = None
        self._label = None
        self._slides.clear()
        self._quality.reset()
        self._frames.reset()
        self._refresh("service", "now_playing")
        self._refresh("signal", throttle=True)
        self._update_slide()

    def _on_favourites(self) -> None:
        self._refresh("services", "service", "favourites")

    def _on_audio_outputs(self) -> None:
        self._outputs = self._controller.audio_outputs
        self._refresh("audio")

    def _on_load(self, dsp_load: float, dsp_waiting: float, process_load: float) -> None:
        self._load = (dsp_load, dsp_waiting, process_load)
        self._refresh("signal", throttle=True)

    def _on_announcement(self, status: object, source: object) -> None:
        self._announcement = (status, source)
        self._refresh("announcement")

    def _on_recorder(self) -> None:
        self._refresh("recorder", "recordings")

    def _on_notification_answered(self, answer: NotificationAnswer) -> None:
        self.notification_answered.emit(answer.json())

    def _on_playback(self, _state: object) -> None:
        before = self._sections["playback"]
        now = self._build_playback()
        # Only the position moved on: four times a second while a title plays.
        throttle = (
            before["id"] is not None
            and before["id"] == now["id"]
            and before["state"] == now["state"]
        )
        self._set("playback", now, throttle=throttle)

    # --- What the signal processing and the audio output report -----------------------

    def _on_subchannel_info(self, status: object) -> None:
        self._subchannel = status
        self._refresh("service")

    def _on_audio_format(self, description: object) -> None:
        self._audio_format = description
        self._refresh("service")

    def _on_dynamic_label(self, label: DynamicLabel) -> None:
        self._label = label
        self._refresh("now_playing")

    def _on_date_time(self, broadcast_time: BroadcastTime) -> None:
        self._clock.synchronise(broadcast_time)

    def _on_slide(self, mot_object: MotObject) -> None:
        now = self._clock.now()
        if mot_object.is_header_update:
            self._slides.apply_update(mot_object, now=now)
        else:
            self._slides.add(mot_object, received_at=time.time(), now=now)
        self._update_slide()

    def _on_slide_tick(self) -> None:
        if not len(self._slides):
            return
        now = self._clock.now()
        self._slides.expire(now)
        self._slides.tick(now)
        self._update_slide()

    def _update_slide(self) -> None:
        self._slides.take_presented()
        slide = self._slides.live_slide()
        image = None
        if slide is not None:
            content_type = _MIME_TYPES.get(slide.content_subtype, _UNKNOWN_MIME_TYPE)
            image = SlideImage(slide.data, content_type, slide.name)
        if image == self._slide:
            return
        self._slide = image
        self.changed.emit("slide")

    def _on_superframe(self, result: SuperframeResult) -> None:
        self._quality.add_superframe(result)
        self._frames.add_superframe(result)
        self._refresh("signal", throttle=True)

    def _on_mp2_frame(self, result: Mp2FrameResult) -> None:
        self._quality.add_mp2_frame(result)
        self._frames.add_mp2_frame(result)
        self._refresh("signal", throttle=True)

    def _on_frequency_offset(self, offset_hz: float) -> None:
        self._offset_hz = offset_hz
        self._refresh("signal", throttle=True)

    def _on_clipping(self, fraction: float) -> None:
        self._clipping = fraction
        self._refresh("signal", throttle=True)

    def _on_sfn_share(self, others: float, outside_guard: float) -> None:
        self._sfn = (others, outside_guard)
        self._refresh("signal", throttle=True)
