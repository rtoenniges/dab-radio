"""Programs of the system run without waiting for them (`net.command`), with Python in
place of nmcli and pactl: a command's output, error and exit code in English UTF-8, one
killed when it takes too long, one that cannot start; a monitor that reports every line
and starts its program again when it ends."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from collections.abc import Callable

import pytest
from PyQt6.QtCore import QEventLoop, QObject, QTimer
from PyQt6.QtWidgets import QApplication

from net.command import KILLED, NOT_STARTED, CommandResult, CommandRunner, LineMonitor

_app = QApplication.instance() or QApplication([])


def _run(runner: CommandRunner, arguments: list[str], timeout_ms: int = 10_000) -> CommandResult:
    results = []
    loop = QEventLoop()

    def done(result: CommandResult) -> None:
        results.append(result)
        loop.quit()

    runner.run(arguments, timeout_ms, done)
    guard = QTimer(loop)
    guard.setSingleShot(True)
    guard.timeout.connect(loop.quit)
    guard.start(15_000)
    if not results:
        loop.exec()
    return results[0]


def _wait_until(condition: Callable[[], bool], timeout_ms: int = 15_000) -> bool:
    """Runs the event loop until the condition holds; False when it did not in time."""
    loop = QEventLoop()
    poll = QTimer(loop)
    poll.setInterval(20)
    poll.timeout.connect(lambda: loop.quit() if condition() else None)
    guard = QTimer(loop)
    guard.setSingleShot(True)
    guard.timeout.connect(loop.quit)
    guard.start(timeout_ms)
    poll.start()
    if not condition():
        loop.exec()
    return condition()


@pytest.fixture
def owner():
    owner = QObject()
    yield owner
    owner.deleteLater()


# --- One command ------------------------------------------------------------------------


def test_the_runner_reports_output_errors_and_exit_code_in_english_utf8(owner):
    runner = CommandRunner(sys.executable, owner)
    script = (
        "import os, sys; sys.stdout.buffer.write('Müller:'.encode() + os.environ['LC_ALL'].encode());"
        "sys.stderr.write('Error: x'); sys.exit(4)"
    )
    result = _run(runner, ["-c", script])
    assert result == CommandResult(4, "Müller:C.UTF-8", "Error: x")
    assert not result.ok


def test_the_runner_kills_a_command_that_takes_too_long(owner):
    runner = CommandRunner(sys.executable, owner)
    result = _run(runner, ["-c", "import time; time.sleep(10)"], timeout_ms=300)
    assert result.exit_code == KILLED


def test_the_runner_reports_a_program_that_cannot_start(owner):
    runner = CommandRunner("nmcli-does-not-exist", owner)
    assert _run(runner, ["radio", "wifi"]).exit_code == NOT_STARTED


# --- A program that runs on -------------------------------------------------------------


def test_the_monitor_reports_every_line_and_starts_again_when_the_program_ends(owner):
    script = "print('Event a'); print('Event b', flush=True)"
    monitor = LineMonitor(sys.executable, ["-c", script], owner)
    lines: list[str] = []
    monitor.line.connect(lines.append)

    monitor.start()
    assert monitor.running
    monitor.start()  # nothing happens while it runs
    assert _wait_until(lambda: not monitor.running)
    assert lines == ["Event a", "Event b"]
    assert monitor.restarting

    monitor.stop()
    assert not monitor.restarting and not monitor.running


def test_a_monitor_whose_program_cannot_start_tries_again_later(owner):
    monitor = LineMonitor("pactl-does-not-exist", ["subscribe"], owner)
    monitor.start()
    assert _wait_until(lambda: not monitor.running)
    assert monitor.restarting
    monitor.stop()
    assert not monitor.restarting


def test_a_monitor_stopped_ends_its_program(owner):
    script = "import time; print('x', flush=True); time.sleep(10)"
    monitor = LineMonitor(sys.executable, ["-c", script], owner)
    lines: list[str] = []
    monitor.line.connect(lines.append)
    monitor.start()
    assert _wait_until(lambda: lines == ["x"])
    monitor.stop()
    assert not monitor.running and not monitor.restarting
