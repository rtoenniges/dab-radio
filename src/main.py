"""Entry point of the DAB+ receiver: `uv run python src/main.py`, for the touch display
`uv run python src/main.py --touch` (on the Raspberry Pi with `--fullscreen`)."""

import argparse
import logging
import os
import subprocess
import sys

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication

# Nothing of the radio but the picture shown while it starts: the rest is imported in
# `main`, behind that picture.
from ui.touch.splash import StartupSplash

# Switch interval of the GIL. The thread that reads the receiver sleeps between its
# visits to the device and must get the GIL back next to the signal processing and the
# GUI; with the default 5 ms a thread computing in Python besides can keep it from doing
# so for seconds. Measured on 04.10.2026 with the RSPduo (`sdr.worker.RspDuoWorker` and
# the signal processing on noise, plus a thread looping in Python): with 5 ms the reader
# delivered 0.2 s of signal in 15 s and the packet ring overflowed, with 0.5 ms 14.4 s
# without a loss, and without that thread the processing load was the same with both.
# The GNSS receiver (`../gnss-receiver`) needed the same setting for the same reason.
# pyrtlsdr's reading thread for the RTL-SDR waits for the GIL just the same.
GIL_SWITCH_INTERVAL_SECONDS = 0.0005

# The audio backend of Qt Multimedia on Linux. Qt 6.11 brings two and takes its native
# PipeWire one where PipeWire runs (the Pi), which leaks: when the sink of a QAudioSink
# goes away (Bluetooth headphones disconnected), Qt reports an IOError and never
# destroys its PipeWire stream, not on stop(), reset() or deleting the QAudioSink. The
# stream stays as an unlinked node with a connection of its own (it carries
# node.dont-reconnect, so nothing moves it), one more with every disconnect, and
# removing it from outside (pw-cli destroy) crashed the process. Its PulseAudio backend,
# through pipewire-pulse, lets PipeWire move the stream to the default output instead,
# reports no error and removes the stream when the player tears the sink down.
# Reproduced with a virtual sink on the Pi, 2026-10-11 (Qt 6.11.2, PipeWire 1.4.2).
# Any other value than "pipewire" selects the PulseAudio backend; one set in the
# environment is kept.
QT_AUDIO_BACKEND = "pulseaudio"

# Properties of the radio's stream for pipewire-pulse, which libpulse takes from
# PULSE_PROP when Qt connects: the stream plays to its own sink or nowhere. Without
# them PipeWire moves a stream whose sink goes (headphones switched off) to the default,
# the loudspeaker, which the radio pins at 100 % (`--system-volume`) while the player
# plays at 1.0 to a device applying its volume itself; and Qt 6.11's PulseAudio backend
# does not report a sink going unless it was the default. On the Pi a tone played so
# reached a stand-in loudspeaker (a null sink) unattenuated for as long as it ran; with
# these properties the stream ended with its sink, and one made for a sink gone failed
# instead of landing on the default (2026-10-11). The player notices such a stream by
# the sound card taking nothing more (`audio.player.STALL_SECONDS`).
PINNED_STREAM_PROPERTIES = "node.dont-reconnect=true node.dont-fallback=true"


def parse_arguments(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """The radio's own options, and the rest of the command line for Qt (e.g.
    `-platform`), led by the program name as Qt expects it."""
    parser = argparse.ArgumentParser(prog="main.py", description="DAB+ receiver")
    parser.add_argument(
        "--touch",
        action="store_true",
        help="surface for the 7-inch touch display (1280 × 720, landscape)",
    )
    parser.add_argument("--fullscreen", action="store_true", help="full screen, not a window")
    parser.add_argument(
        "--system-volume",
        action="store_true",
        help="set the volume of the sound system, for the radio as the one program "
        "playing (pi/start-touch.sh); without it the radio attenuates its own sound",
    )
    arguments, qt_arguments = parser.parse_known_args(argv[1:])
    return arguments, [argv[0], *qt_arguments]


# The parts of the radio whose info messages are kept (to stderr, on the Pi into the
# journal): how an output is treated, Bluetooth, storage, recording. Those of the
# signal processing and the receivers stay at warnings: an RSPduo reports every
# change of its overload. Without a configuration Python kept only warnings.
INFO_LOGGERS = ("audio", "net", "recording", "storage", "ui")


def configure_logging() -> None:
    """Warnings of everything, info messages of `INFO_LOGGERS`."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    for name in INFO_LOGGERS:
        logging.getLogger(name).setLevel(logging.INFO)


def choose_audio_backend(environ: dict[str, str], platform: str) -> None:
    """Qt's PulseAudio backend on Linux (`QT_AUDIO_BACKEND`), unless the environment
    names one; set before Qt Multimedia starts."""
    if platform.startswith("linux"):
        environ.setdefault("QT_AUDIO_BACKEND", QT_AUDIO_BACKEND)


def pin_audio_stream(environ: dict[str, str], platform: str, system_volume: bool) -> None:
    """Keeps the radio's stream on its own sink (`PINNED_STREAM_PROPERTIES`) where the
    radio sets the system's volume; properties already in PULSE_PROP stay, and a
    restart (`restart`, same environment) adds nothing twice."""
    present = environ.get("PULSE_PROP", "")
    if not platform.startswith("linux") or not system_volume:
        return
    if PINNED_STREAM_PROPERTIES not in present:
        environ["PULSE_PROP"] = f"{present} {PINNED_STREAM_PROPERTIES}".strip()


def restart(command: list[str]) -> None:
    """Starts the radio anew with `command`, for a new language of the surface. On
    Linux in place of this process: nothing ends, so cage keeps its client and systemd its
    unit. Elsewhere a new process, and this one ends."""
    sys.stdout.flush()
    sys.stderr.flush()
    if sys.platform == "win32":
        subprocess.Popen(command)
        return
    os.execv(command[0], command)


def main() -> None:
    arguments, qt_arguments = parse_arguments(sys.argv)
    sys.setswitchinterval(GIL_SWITCH_INTERVAL_SECONDS)
    configure_logging()
    choose_audio_backend(os.environ, sys.platform)  # type: ignore[arg-type]
    pin_audio_stream(os.environ, sys.platform, arguments.system_volume)  # type: ignore[arg-type]
    app = QApplication(qt_arguments)
    splash = None
    if arguments.touch and arguments.fullscreen:
        # The touch display needs no pointer, but cage shows one as soon as any device
        # reports as a mouse (such as the receiver of a wireless keyboard). cage takes
        # over the cursor of the window it focuses, right at the start without the mouse
        # moving. In a window on the PC the pointer stays to try the touch surface.
        app.setOverrideCursor(QCursor(Qt.CursorShape.BlankCursor))
        # On the Pi the radio takes 2 to 4 s from here to its window, after Plymouth
        # has let go of the display (`ui.touch.splash`): the last picture of the boot
        # splash stands meanwhile, rather than the empty screen of cage.
        splash = StartupSplash()
        splash.show_now()
    # Imported only now, behind the picture: signal processing, plots and pages.
    from ui.receiver_controller import (
        APPLICATION_NAME,
        APPLICATION_VERSION,
        SETTINGS_FILENAME,
        ReceiverController,
        app_data_directory,
    )

    app.setApplicationName(APPLICATION_NAME)
    app.setApplicationVersion(APPLICATION_VERSION)
    from ui.language import configure_language, restart_command

    # Before anything is built that shows a text, the controller included: it hands
    # texts to the surface too.
    settings_path = app_data_directory() / SETTINGS_FILENAME
    configure_language(app, QSettings(str(settings_path), QSettings.Format.IniFormat))
    # The application name decides the data directory the controller reads from. The
    # volume of the sound system only where the radio is the one program playing: next
    # to others a loudspeaker pinned at 100 % would play their sounds at full volume.
    # MQTT for the radio on the touch display alone: the desktop is not controlled from
    # a smart home and shows no notifications.
    controller = ReceiverController(
        system_volume=arguments.system_volume, remote_control=arguments.touch
    )
    if arguments.touch:
        from ui.touch.theme import touch_font
        from ui.touch.window import TouchWindow

        # The typeface ships with the surface: the display shows what the PC does.
        app.setFont(touch_font())
        window = TouchWindow(controller)
    else:
        from ui.main_window import MainWindow

        window = MainWindow(controller)
    # A new language takes effect in a new start, which both surfaces offer.
    restart_wanted = []
    window.restart_requested.connect(lambda: (restart_wanted.append(True), window.close()))
    # Only now: the window has to be connected before the controller first reports.
    controller.start()
    if arguments.touch:
        # A radio plays when switched on.
        window.start_playing()
    if arguments.fullscreen:
        window.showFullScreen()
    else:
        window.show()
    if splash is not None:
        splash.close_over(window)
    status = app.exec()
    if restart_wanted:
        restart(restart_command())
    sys.exit(status)


if __name__ == "__main__":
    main()
