"""The stations of the touch surface (`ui.touch.stations`) and how it judges the
reception (`ui.touch.quality`)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QBuffer, QEventLoop, QIODevice, QTimer
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication

from dab.ensemble_cache import SOURCE_SCAN, EnsembleCache, EnsembleInfo
from dsp.mp2_frame import CHANNEL_MODE_JOINT_STEREO, Mp2Format, Mp2FrameResult
from dsp.superframe import SuperframeResult
from tests.fake_receiver import ENSEMBLE_5C, SERVICES_5C, SERVICES_11D
from ui import receiver_controller
from ui.touch import stations
from ui.touch.quality import ReceptionQuality
from ui.touch.stations import (
    FAVOURITES,
    EnsembleListing,
    Station,
    StationDirectory,
    build_catalogue,
    initials,
    step_station,
)

LISTINGS = {
    "11D": EnsembleListing("NDR NDS", ((0xD311, "NDR 2"), (0xD312, "N-JOY"))),
    "5C": EnsembleListing(ENSEMBLE_5C, ((0xD220, "Dlf Kultur"), (0xD210, "dlf"))),
}


def test_the_catalogue_is_ordered_by_channel_frequency_then_name():
    stations = build_catalogue(LISTINGS)

    assert [(s.channel, s.label) for s in stations] == [
        ("5C", "dlf"),
        ("5C", "Dlf Kultur"),
        ("11D", "N-JOY"),
        ("11D", "NDR 2"),
    ]


def test_stepping_wraps_round_at_either_end():
    stations = build_catalogue(LISTINGS)

    assert step_station(stations, ("11D", 0xD311), 1).key == ("5C", 0xD210)
    assert step_station(stations, ("5C", 0xD210), -1).key == ("11D", 0xD311)
    assert step_station(stations, ("5C", 0xD210), 1).key == ("5C", 0xD220)


def test_stepping_from_a_station_outside_the_list_starts_at_its_end():
    stations = [s for s in build_catalogue(LISTINGS) if s.channel == "11D"]

    assert step_station(stations, ("5C", 0xD210), 1).label == "N-JOY"
    assert step_station(stations, ("5C", 0xD210), -1).label == "NDR 2"
    assert step_station([], None, 1) is None


def test_initials_stand_in_for_a_logo():
    assert initials("Dlf Kultur") == "DK"
    assert initials("80s80s") == "80"
    assert initials("NDR 2") == "N2"
    assert initials("  ") == "?"


def _add_11d(data_directory) -> None:
    cache = EnsembleCache(data_directory / receiver_controller.ENSEMBLE_CACHE_FILENAME)
    cache.load()
    cache.put("11D", EnsembleInfo("NDR NDS", SERVICES_11D, 0.0, SOURCE_SCAN))
    cache.save()


def test_the_directory_lists_every_stored_channel(data_directory, make_controller):
    _add_11d(data_directory)
    directory = StationDirectory(make_controller(data_directory))

    assert [s.channel for s in directory.stations()] == ["5C", "5C", "11D"]
    assert directory.filters() == [("5C", f"5C · {ENSEMBLE_5C}"), ("11D", "11D · NDR NDS")]
    assert directory.filter_name(FAVOURITES) == "Favourites"
    assert directory.filter_name(None) == "All"
    assert directory.filter_name("11D") == "11D · NDR NDS"

    directory.set_filter("11D")
    assert directory.stations() == [Station("11D", 0xD311, "NDR 2")]


def test_steps_stay_within_the_filter(data_directory, make_controller):
    _add_11d(data_directory)
    controller = make_controller(data_directory)
    directory = StationDirectory(controller)
    controller.select_service(SERVICES_5C[0][0])
    assert directory.current() == Station("5C", 0xD210, "Dlf")

    assert directory.neighbour(1).label == "Dlf Kultur"
    directory.set_filter("11D")
    assert directory.neighbour(1).label == "NDR 2"


def test_the_favourites_filter_keeps_the_order_and_its_steps(data_directory, make_controller):
    _add_11d(data_directory)
    controller = make_controller(data_directory)
    directory = StationDirectory(controller)
    directory.set_filter(FAVOURITES)
    assert directory.stations() == []
    # Not the directory itself in the lambda: the cycle would keep it for the garbage
    # collector, which then runs at other times in later tests.
    changes = []
    directory.changed.connect(lambda: changes.append(True))

    controller.set_favourite("11D", 0xD311, True)
    directory.toggle_favourite(Station("5C", 0xD220, "Dlf Kultur"))
    # Marked, but not among the stations known: not shown.
    controller.set_favourite("12D", 0xD311, True)

    assert [s.label for s in directory.stations()] == ["Dlf Kultur", "NDR 2"]
    assert len(changes) == 3
    controller.select_service(0xD220)
    assert directory.neighbour(1).label == "NDR 2"
    assert directory.neighbour(2).label == "Dlf Kultur"

    directory.toggle_favourite(Station("5C", 0xD220, "Dlf Kultur"))
    assert [s.label for s in directory.stations()] == ["NDR 2"]


def test_the_stations_of_every_filter_are_counted(data_directory, make_controller):
    _add_11d(data_directory)
    controller = make_controller(data_directory)
    # Marked, but not among the stations known: not counted.
    controller.set_favourite("12D", 0xD311, True)
    controller.set_favourite("11D", 0xD311, True)

    assert StationDirectory(controller).counts() == {None: 3, FAVOURITES: 1, "5C": 2, "11D": 1}


def test_the_filter_is_kept_for_the_next_start(data_directory, make_controller):
    _add_11d(data_directory)
    controller = make_controller(data_directory)
    StationDirectory(controller).set_filter(FAVOURITES)
    assert StationDirectory(controller).filter == FAVOURITES

    StationDirectory(controller).set_filter("11D")
    assert StationDirectory(controller).filter == "11D"

    StationDirectory(controller).set_filter(None)
    assert StationDirectory(controller).filter is None


def test_a_kept_filter_of_a_channel_gone_shows_all(data_directory, make_controller):
    controller = make_controller(data_directory)
    controller.settings.setValue(stations.SETTINGS_STATION_FILTER, "11D")

    assert StationDirectory(controller).filter is None


def test_the_received_list_counts_until_the_channel_is_stored(
    data_directory, make_controller, fake_workers
):
    controller = make_controller(data_directory)
    directory = StationDirectory(controller)
    controller.start_reception()
    controller.dsp_worker.stream_restarted.emit(fake_workers[-1].tuning)
    controller.dsp_worker.services_updated.emit(ENSEMBLE_5C, [(0xD299, "Neu")], {})
    controller.dsp_worker.service_list_completed.emit()

    assert [s.label for s in directory.stations()] == ["Neu"]

    controller.channel_stored.emit("5C")
    assert [s.label for s in directory.stations()] == ["Dlf", "Dlf Kultur"]


def test_the_received_list_counts_only_once_complete(data_directory, make_controller, fake_workers):
    """While the FIC delivers the list station by station, the stored one stays; the
    complete list changes the catalogue only where it differs, so the page holds still."""
    controller = make_controller(data_directory)
    directory = StationDirectory(controller)
    changes = []
    directory.changed.connect(lambda: changes.append(directory.all_stations()))
    dsp = controller.dsp_worker
    controller.start_reception()
    dsp.stream_restarted.emit(fake_workers[-1].tuning)

    dsp.services_updated.emit(ENSEMBLE_5C, [SERVICES_5C[1]], {})
    dsp.services_updated.emit(ENSEMBLE_5C, list(SERVICES_5C), {})
    assert [s.label for s in directory.stations()] == ["Dlf", "Dlf Kultur"]
    assert changes == []

    dsp.service_list_completed.emit()
    assert changes == []

    # Once complete, a change is taken over at once.
    dsp.services_updated.emit(ENSEMBLE_5C, [*SERVICES_5C, (0xD299, "Neu")], {})
    assert [s.label for s in directory.stations()] == ["Dlf", "Dlf Kultur", "Neu"]
    assert len(changes) == 1


def _superframe(*, sync=True, corrections=0, lost=0, units=3) -> SuperframeResult:
    return SuperframeResult(
        sync_ok=sync,
        rs_corrections=corrections,
        rs_uncorrectable_blocks=lost,
        audio_units=[b""] * units,
        num_audio_units=3,
    )


def test_the_bars_follow_the_audio_frames():
    quality = ReceptionQuality()
    assert quality.bars is None

    for _ in range(10):
        quality.add_superframe(_superframe())
    assert (quality.bars, quality.describe()) == (4, "very good")

    quality.add_superframe(_superframe(corrections=2))
    assert quality.bars == 3
    quality.add_superframe(_superframe(units=2))
    assert quality.bars == 2
    for _ in range(4):
        quality.add_superframe(_superframe(sync=False))
    assert quality.bars == 1
    for _ in range(10):
        quality.add_superframe(_superframe(lost=1))
    assert (quality.bars, quality.describe()) == (0, "no audio")


def test_mp2_frames_count_by_their_header_crc():
    quality = ReceptionQuality()
    audio_format = Mp2Format(48_000, 128, CHANNEL_MODE_JOINT_STEREO)
    for _ in range(49):
        quality.add_mp2_frame(Mp2FrameResult(audio_format, sync_ok=True, crc_ok=True))
    assert quality.bars == 4
    quality.add_mp2_frame(Mp2FrameResult(audio_format, sync_ok=True, crc_ok=False))
    assert quality.bars == 2


def test_logos_are_read_again_once_a_guide_brought_new_ones(
    data_directory, make_controller, monkeypatch
):
    monkeypatch.setattr(stations, "LOGO_RELOAD_DELAY_MS", 0)
    controller = make_controller(data_directory)
    directory = StationDirectory(controller)
    changed = []
    directory.logos_changed.connect(lambda: changed.append(True))
    assert directory.logo(SERVICES_5C[0][0]) is None

    image = QImage(32, 32, QImage.Format.Format_RGB32)
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    monkeypatch.setattr(
        stations, "stored_logos", lambda *_: {SERVICES_5C[0][0]: bytes(buffer.data())}
    )
    controller.guide_stored.emit("5C", 0.0, False)  # a schedule: nothing to read
    QApplication.processEvents()
    assert changed == []

    controller.guide_stored.emit("5C", 0.0, True)
    loop = QEventLoop()
    QTimer.singleShot(50, loop.quit)
    loop.exec()
    assert changed == [True]
    assert directory.logo(SERVICES_5C[0][0]) is not None
