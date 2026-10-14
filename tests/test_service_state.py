"""State of the SDRplay API service as systemd reports it (Linux)."""

import subprocess

import pytest

from sdr import service_state
from sdr.service_state import ServiceState


def _systemctl(stdout: str):
    def run(args, **kwargs):
        assert args[:3] == ["systemctl", "show", "sdrplay.service"]
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    return run


@pytest.mark.parametrize(
    ("stdout", "state"),
    [
        ("LoadState=loaded\nActiveState=active\n", ServiceState.RUNNING),
        ("LoadState=loaded\nActiveState=inactive\n", ServiceState.STOPPED),
        ("LoadState=loaded\nActiveState=failed\n", ServiceState.STOPPED),
        ("LoadState=loaded\nActiveState=activating\n", ServiceState.PENDING),
        # systemd answers for any name; an unknown unit is "inactive" but not found.
        ("LoadState=not-found\nActiveState=inactive\n", ServiceState.NOT_INSTALLED),
        ("", ServiceState.UNKNOWN),
    ],
)
def test_systemd_states(monkeypatch, stdout, state):
    monkeypatch.setattr(service_state.sys, "platform", "linux")
    monkeypatch.setattr(service_state.subprocess, "run", _systemctl(stdout))
    assert service_state.query_service_state() is state


def test_without_systemctl_the_state_is_unknown(monkeypatch):
    def missing(args, **kwargs):
        raise FileNotFoundError(args[0])

    monkeypatch.setattr(service_state.sys, "platform", "linux")
    monkeypatch.setattr(service_state.subprocess, "run", missing)
    assert service_state.query_service_state() is ServiceState.UNKNOWN
