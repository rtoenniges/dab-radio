"""Tests for finding the decoder libraries (`audio.native`)."""

from __future__ import annotations

import sys

import pytest

from audio.native import find_library


@pytest.fixture
def nothing_on_path(monkeypatch):
    """As if the environment's Scripts folder were not on the PATH."""
    monkeypatch.setattr("ctypes.util.find_library", lambda _name: None)


def test_dll_in_the_scripts_folder_is_found_without_the_path(
    monkeypatch, tmp_path, nothing_on_path
):
    (tmp_path / "Scripts").mkdir()
    (tmp_path / "Scripts" / "libmpg123-0.dll").write_bytes(b"")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))

    assert find_library("mpg123", "libmpg123-0") == str(tmp_path / "Scripts" / "libmpg123-0.dll")


def test_library_on_the_path_comes_first(monkeypatch, tmp_path):
    (tmp_path / "Scripts").mkdir()
    (tmp_path / "Scripts" / "faad.dll").write_bytes(b"")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(
        "ctypes.util.find_library", lambda name: "libfaad.so.2" if name == "faad" else None
    )

    assert find_library("faad", "libfaad-2") == "libfaad.so.2"


def test_missing_library_gives_none(monkeypatch, tmp_path, nothing_on_path):
    monkeypatch.setattr(sys, "prefix", str(tmp_path))

    assert find_library("faad", "libfaad-2") is None
