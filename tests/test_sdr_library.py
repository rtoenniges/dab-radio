"""`sdr.library`: which librtlsdr pyrtlsdr gets, without touching a stick."""

import importlib.util
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pyrtlsdrlib
import pytest

import sdr  # noqa: F401 - importing the package installs the loader
from sdr import library


@pytest.fixture
def bundled(monkeypatch):
    """Stands in for the library of `pyrtlsdrlib`."""
    dll = SimpleNamespace(name="bundled")
    monkeypatch.setattr(library, "_load_bundled", lambda: dll)
    return dll


@pytest.fixture
def cdll(monkeypatch):
    """Replaces `ctypes.CDLL` in `sdr.library`; returns the paths it was asked for."""
    paths = []

    def load(path):
        paths.append(path)
        # What osmocom's rtl-sdr exports, of the functions that matter here.
        return SimpleNamespace(rtlsdr_open=object())

    monkeypatch.setattr(library, "ctypes", SimpleNamespace(CDLL=load))
    return paths


def test_importing_sdr_makes_pyrtlsdr_load_through_this_module():
    assert pyrtlsdrlib.load_librtlsdr is library.load_librtlsdr


def test_installing_after_rtlsdr_was_imported_warns(monkeypatch, caplog):
    """pyrtlsdr has bound the bundled library by then and keeps it."""
    monkeypatch.setattr(pyrtlsdrlib, "load_librtlsdr", lambda: None)
    modules = {"rtlsdr.librtlsdr": object()}
    monkeypatch.setattr(library, "sys", SimpleNamespace(platform="win32", modules=modules))

    with caplog.at_level(logging.WARNING, logger="sdr.library"):
        library.install()

    assert pyrtlsdrlib.load_librtlsdr is library.load_librtlsdr
    assert "imported before sdr" in caplog.text


def test_the_radios_own_build_comes_first(monkeypatch, tmp_path, bundled, cdll):
    built = tmp_path / "rtlsdr.dll"
    built.write_bytes(b"")
    monkeypatch.setattr(library, "NATIVE_LIBRARY", built)
    monkeypatch.setattr(library, "sys", SimpleNamespace(platform="win32"))

    dll = library.load_librtlsdr()

    assert cdll == [str(built)]
    assert dll is not bundled


def test_own_build_gets_the_functions_only_the_fork_has(monkeypatch, tmp_path, bundled, cdll):
    """pyrtlsdr binds them on import and switches the dithering on with every open."""
    built = tmp_path / "rtlsdr.dll"
    built.write_bytes(b"")
    monkeypatch.setattr(library, "NATIVE_LIBRARY", built)
    monkeypatch.setattr(library, "sys", SimpleNamespace(platform="win32"))

    dll = library.load_librtlsdr()

    for name in library._FORK_ONLY_FUNCTIONS:
        function = getattr(dll, name)
        function.restype, function.argtypes = int, []  # as pyrtlsdr does on import
    assert dll.rtlsdr_set_dithering(None, 1) == 0
    assert dll.rtlsdr_set_dithering(None, 0) < 0
    assert dll.rtlsdr_set_gpio_bit(None, 0, 1) < 0


def test_without_the_own_build_the_bundled_library_serves_and_the_log_warns(
    monkeypatch, tmp_path, bundled, cdll, caplog
):
    monkeypatch.setattr(library, "NATIVE_LIBRARY", tmp_path / "rtlsdr.dll")
    monkeypatch.setattr(library, "sys", SimpleNamespace(platform="win32"))

    with caplog.at_level(logging.WARNING, logger="sdr.library"):
        dll = library.load_librtlsdr()

    assert dll is bundled
    assert cdll == []
    assert "RAW_IO" in caplog.text
    assert "build_rtlsdr.ps1" in caplog.text


def test_other_platforms_keep_the_bundled_library(monkeypatch, tmp_path, bundled, cdll, caplog):
    built = tmp_path / "rtlsdr.dll"
    built.write_bytes(b"")
    monkeypatch.setattr(library, "NATIVE_LIBRARY", built)
    monkeypatch.setattr(library, "sys", SimpleNamespace(platform="linux"))

    with caplog.at_level(logging.WARNING, logger="sdr.library"):
        dll = library.load_librtlsdr()

    assert dll is bundled
    assert cdll == []
    assert caplog.text == ""


@pytest.mark.skipif(
    sys.platform != "win32" or not library.NATIVE_LIBRARY.is_file(),
    reason="native/rtlsdr.dll has not been built",
)
def test_the_built_library_has_every_function_pyrtlsdr_binds():
    """Guards against a pyrtlsdr that binds further functions only the fork has: the
    import of `rtlsdr` would fail with them."""
    binding = Path(importlib.util.find_spec("rtlsdr").origin).with_name("librtlsdr.py")
    bound = set(re.findall(r"librtlsdr\.(rtlsdr_\w+)", binding.read_text(encoding="utf-8")))
    # pyrtlsdr itself checks for these two and does without them.
    bound -= {"rtlsdr_set_and_get_tuner_bandwidth", "rtlsdr_set_tuner_bandwidth"}

    dll = library.load_librtlsdr()

    assert bound
    assert [name for name in sorted(bound) if not hasattr(dll, name)] == []
