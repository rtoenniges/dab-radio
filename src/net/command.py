"""Programs of the system, run without waiting for them.

`CommandRunner` runs one command and reports its result (nmcli for `net.wifi`, pactl for
`audio.system_volume`); `LineMonitor` runs a program that goes on and on and reports
every line it prints (`pactl subscribe`). Both are `QProcess`es in the thread of their
owner. The programs answer in English in UTF-8 (`LC_ALL=C.UTF-8`), so that their
messages can be recognised and names with umlauts stay intact.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

logger = logging.getLogger(__name__)

# Exit codes of our own for a command that did not run to its end.
NOT_STARTED = -1
KILLED = -2
# A monitor that ended by itself (its server went, or it could not start) is started
# again after this long.
RESTART_MS = 10_000


@dataclass(frozen=True)
class CommandResult:
    exit_code: int  # of the program, or NOT_STARTED, KILLED
    output: str
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


Done = Callable[[CommandResult], None]


def english_environment() -> QProcessEnvironment:
    """The environment of the programs: English messages in UTF-8 (in the plain C
    locale nmcli would mangle names with umlauts)."""
    environment = QProcessEnvironment.systemEnvironment()
    environment.insert("LC_ALL", "C.UTF-8")
    environment.remove("LANGUAGE")
    return environment


class Command(QObject):
    """One run of a program; reports its result once and goes (with `deleteLater`: it is
    still inside the signal of its process)."""

    def __init__(
        self, program: str, arguments: list[str], timeout_ms: int, done: Done, parent: QObject
    ) -> None:
        super().__init__(parent)
        self._done: Done | None = done
        self._killed = False
        self._reported = False
        self.process = QProcess(self)
        self.process.setProcessEnvironment(english_environment())
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_error)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.kill)
        self._timer.start(timeout_ms)
        self.process.start(program, arguments)

    def kill(self) -> None:
        self._killed = True
        self.process.kill()

    def cancel(self) -> None:
        """Kills the process without reporting."""
        self._done = None
        self.kill()
        self.process.waitForFinished(1000)

    def _on_finished(self, exit_code: int, status: QProcess.ExitStatus) -> None:
        crashed = status != QProcess.ExitStatus.NormalExit
        self._report(KILLED if self._killed or crashed else exit_code)

    def _on_error(self, error: QProcess.ProcessError) -> None:
        # Otherwise `finished` follows.
        if error == QProcess.ProcessError.FailedToStart:
            self._report(NOT_STARTED)

    def _report(self, exit_code: int) -> None:
        if self._reported:
            return
        self._reported = True
        self._timer.stop()
        output = bytes(self.process.readAllStandardOutput()).decode("utf-8", "replace")
        error = bytes(self.process.readAllStandardError()).decode("utf-8", "replace")
        self.deleteLater()
        if self._done is not None:
            self._done(CommandResult(exit_code, output, error))


class CommandRunner:
    """Runs a program, one command at a time (the owner queues them)."""

    def __init__(self, program: str, parent: QObject) -> None:
        self._program = program
        self._parent = parent
        self._command: Command | None = None

    def run(self, arguments: list[str], timeout_ms: int, done: Done) -> None:
        self._command = Command(self._program, arguments, timeout_ms, done, self._parent)

    def stop(self) -> None:
        """Kills the command running, without waiting for its result."""
        command, self._command = self._command, None
        if command is not None and not command._reported:
            command.cancel()


class LineMonitor(QObject):
    """A program that runs on and reports every line it prints (`line`); one that ends
    by itself, or cannot start, is started again after `RESTART_MS`."""

    line = pyqtSignal(str)

    def __init__(self, program: str, arguments: list[str], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._program = program
        self._arguments = list(arguments)
        self._process: QProcess | None = None
        self._partial = ""
        self._restart = QTimer(self)
        self._restart.setSingleShot(True)
        self._restart.setInterval(RESTART_MS)
        self._restart.timeout.connect(self.start)

    @property
    def running(self) -> bool:
        return self._process is not None

    @property
    def restarting(self) -> bool:
        """Whether the program ended and is to be started again."""
        return self._restart.isActive()

    def start(self) -> None:
        """Starts the program; nothing happens while it runs."""
        if self._process is not None:
            return
        self._restart.stop()
        self._partial = ""
        process = QProcess(self)
        process.setProcessEnvironment(english_environment())
        process.readyReadStandardOutput.connect(self._on_output)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        self._process = process
        process.start(self._program, self._arguments)

    def stop(self) -> None:
        """Ends the program and does not start it again."""
        self._restart.stop()
        process, self._process = self._process, None
        if process is None:
            return
        process.kill()
        process.waitForFinished(1000)
        process.deleteLater()

    def _on_output(self) -> None:
        process = self._process
        if process is None:
            return
        text = self._partial + bytes(process.readAllStandardOutput()).decode("utf-8", "replace")
        lines = text.split("\n")
        self._partial = lines.pop()
        for line in lines:
            self.line.emit(line.rstrip("\r"))

    def _on_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._ended(f"exit code {exit_code}")

    def _on_error(self, error: QProcess.ProcessError) -> None:
        # Otherwise `finished` follows.
        if error == QProcess.ProcessError.FailedToStart:
            self._ended("cannot be started")

    def _ended(self, why: str) -> None:
        process, self._process = self._process, None
        if process is None:  # stopped on purpose
            return
        error = bytes(process.readAllStandardError()).decode("utf-8", "replace").strip()
        process.deleteLater()
        logger.warning(
            "%s %s ended (%s%s); starting it again in %d s",
            self._program,
            " ".join(self._arguments),
            why,
            f": {error}" if error else "",
            RESTART_MS // 1000,
        )
        self._restart.start()
