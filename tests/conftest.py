"""Fixtures for the tests of `ui.receiver_controller` and the surfaces on top of it."""

from __future__ import annotations

import gc
import re
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QApplication

from audio import system_volume
from dab.ensemble_cache import SOURCE_SCAN, EnsembleCache, EnsembleInfo
from tests.fake_receiver import (
    ENSEMBLE_5C,
    SERVICES_5C,
    SERVICES_11D,
    FakeChannelScanner,
    FakeRspDuoWorker,
    FakeRtlSdrWorker,
)
from ui import receiver_controller
from ui.receiver_controller import ReceiverController

# The vectors from real broadcasts (`tests/*_vectors.py`) are kept locally only, see
# .gitignore. A test module that needs one that is missing is left out of the run
# rather than failing on its import.
_VECTOR_IMPORT = re.compile(r"^from tests\.(\w+_vectors) import", re.MULTILINE)
_TEST_IMPORT = re.compile(r"^from tests\.(test_\w+) import", re.MULTILINE)


def _without_vectors() -> list[str]:
    """The test modules that import a missing vector module, directly or through another
    test module they import from."""
    here = Path(__file__).parent
    sources = {path.stem: path.read_text(encoding="utf-8") for path in here.glob("test_*.py")}
    left_out = {
        name
        for name, source in sources.items()
        if any(not (here / f"{vector}.py").exists() for vector in _VECTOR_IMPORT.findall(source))
    }
    while True:
        more = {
            name
            for name, source in sources.items()
            if name not in left_out and left_out.intersection(_TEST_IMPORT.findall(source))
        }
        if not more:
            return sorted(f"{name}.py" for name in left_out)
        left_out |= more


collect_ignore = _without_vectors()


def pytest_report_header(config) -> str | None:
    if collect_ignore:
        return "vectors from real broadcasts missing, left out: " + ", ".join(collect_ignore)
    return None


@pytest.fixture(autouse=True, scope="module")
def _collect_garbage_after_each_module():
    """Destroys at the end of a module what its tests left in reference cycles. Left to
    the garbage collector, such Qt objects went whenever it happened to run, also in the
    middle of a later test: there a pyqtgraph LabelItem was resized after its text item
    was gone, and PyQt6 aborts on the exception in that virtual (in
    `test_touch_traffic.py`, after tests had been added to another module, 07.10.2026).
    Collected after every test instead, the run takes twice as long."""
    yield
    gc.collect()


@pytest.fixture
def fake_workers(monkeypatch) -> list[FakeRtlSdrWorker]:
    """The workers the controller creates, of the stick and of the RSPduo, newest last;
    no device is opened, not even the stick for the gain range, and no pactl is found:
    the player attenuates (`test_audio_outputs` fakes the system volume)."""
    monkeypatch.setattr(receiver_controller, "query_gains", lambda: None)
    monkeypatch.setattr(system_volume, "find_pactl", lambda: None)
    monkeypatch.setattr(receiver_controller, "RtlSdrWorker", FakeRtlSdrWorker)
    monkeypatch.setattr(receiver_controller, "RspDuoWorker", FakeRspDuoWorker)
    FakeRtlSdrWorker.instances = []
    return FakeRtlSdrWorker.instances


@pytest.fixture
def fake_scanners(monkeypatch) -> list[FakeChannelScanner]:
    """The scans the controller starts, newest last; none opens a device."""
    monkeypatch.setattr(receiver_controller, "ChannelScanner", FakeChannelScanner)
    FakeChannelScanner.instances = []
    return FakeChannelScanner.instances


@pytest.fixture
def data_directory(tmp_path: Path) -> Path:
    """A data directory in which channel 5C is stored with its services."""
    cache = EnsembleCache(tmp_path / receiver_controller.ENSEMBLE_CACHE_FILENAME)
    cache.put("5C", EnsembleInfo(ENSEMBLE_5C, SERVICES_5C, 0.0, SOURCE_SCAN))
    cache.save()
    return tmp_path


@pytest.fixture
def dlf_kultur_elsewhere(data_directory: Path) -> Path:
    """`data_directory` with 5C stored as ensemble 10BC, which says that Dlf Kultur is
    carried in 1188 as well, stored on 11D, in 1189, which it announces on 12D, and in
    1205, of which nothing is known (FIG 0/24, FIG 0/21)."""
    cache = EnsembleCache(data_directory / receiver_controller.ENSEMBLE_CACHE_FILENAME)
    cache.load()
    cache.put(
        "5C",
        EnsembleInfo(
            ENSEMBLE_5C,
            SERVICES_5C,
            0.0,
            SOURCE_SCAN,
            eid=0x10BC,
            frequencies=((0x1189, (229_072,)),),
            service_ensembles=((SERVICES_5C[1][0], (0x10BC, 0x1188, 0x1189, 0x1205)),),
        ),
    )
    cache.put("11D", EnsembleInfo("NDR NDS", SERVICES_11D, 0.0, SOURCE_SCAN, eid=0x1188))
    cache.save()
    return data_directory


@pytest.fixture
def make_controller(fake_workers):
    """Creates controllers on a data directory, started unless asked otherwise, with
    the options given (`system_volume`), and shuts them down after the test."""
    _app = QApplication.instance() or QApplication([])
    controllers = []

    def make(data_directory: Path, start: bool = True, **options) -> ReceiverController:
        controller = ReceiverController(data_directory, **options)
        controllers.append(controller)
        if start:
            controller.start()
        return controller

    yield make
    # Also those the test started itself: a QThread destroyed while running aborts.
    for controller in controllers:
        controller.shutdown()
