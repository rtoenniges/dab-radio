"""Tests for finding the decoder libraries (`audio.native`)."""

from __future__ import annotations

import sys

import pytest

from audio import native
from audio.native import find_library


@pytest.fixture
def folders(monkeypatch, tmp_path):
    """Empty `native` and Scripts folders in place of the real ones."""
    (tmp_path / "native").mkdir()
    (tmp_path / "Scripts").mkdir()
    monkeypatch.setattr(native, "NATIVE_DIRECTORY", tmp_path / "native")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    return tmp_path


@pytest.fixture
def nothing_on_path(monkeypatch):
    """As if neither folder were on the PATH."""
    monkeypatch.setattr("ctypes.util.find_library", lambda _name: None)


def test_dll_in_the_native_folder_is_found_without_the_path(folders, nothing_on_path):
    (folders / "native" / "faad.dll").write_bytes(b"")

    assert find_library("faad", "libfaad-2") == str(folders / "native" / "faad.dll")


def test_dll_in_the_scripts_folder_is_found_without_the_path(folders, nothing_on_path):
    (folders / "Scripts" / "libmpg123-0.dll").write_bytes(b"")

    assert find_library("mpg123", "libmpg123-0") == str(folders / "Scripts" / "libmpg123-0.dll")


def test_native_folder_comes_before_the_scripts_folder(folders, nothing_on_path):
    (folders / "native" / "mpg123.dll").write_bytes(b"")
    (folders / "Scripts" / "libmpg123-0.dll").write_bytes(b"")

    assert find_library("mpg123", "libmpg123-0") == str(folders / "native" / "mpg123.dll")


def test_library_on_the_path_comes_first(monkeypatch, folders):
    (folders / "native" / "faad.dll").write_bytes(b"")
    (folders / "Scripts" / "faad.dll").write_bytes(b"")
    monkeypatch.setattr(
        "ctypes.util.find_library", lambda name: "libfaad.so.2" if name == "faad" else None
    )

    assert find_library("faad", "libfaad-2") == "libfaad.so.2"


def test_missing_library_gives_none(folders, nothing_on_path):
    assert find_library("faad", "libfaad-2") is None
