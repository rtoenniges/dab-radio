"""The sub-page "Storage" of the touch settings (`ui.touch.storage_view`) and the folder
browser (`ui.touch.folder_browser`), on a UDisks2 that answers from a table
(`tests.fake_udisks`) and on a drive made of a temporary directory."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication, QWidget

from storage.volumes import DRIVE, FILESYSTEM, Volume, VolumeManager
from tests.fake_udisks import INTENSO_STICK, MOUNT_OPTIONS, SANDISK, STICK, STICK_MOUNT, FakeBus
from ui.touch.folder_browser import FolderBrowser, list_folders, path_text, valid_folder_name
from ui.touch.storage_view import (
    StorageView,
    VolumeEntry,
    directory_text,
    free_text,
    volume_entries,
)
from ui.touch.window import SCREEN_HEIGHT, SCREEN_WIDTH


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window():
    window = QWidget()
    window.resize(SCREEN_WIDTH, SCREEN_HEIGHT)
    window.show()
    yield window
    window.deleteLater()


# --- Folder browser -------------------------------------------------------------------


@pytest.fixture
def drive(tmp_path: Path) -> Path:
    """A stick with a few folders, hidden ones and a file."""
    (tmp_path / "Aufnahmen" / "2026").mkdir(parents=True)
    (tmp_path / "Musik").mkdir()
    (tmp_path / ".Trash-1000").mkdir()
    (tmp_path / "System Volume Information").mkdir()
    (tmp_path / "liste.txt").write_text("x")
    return tmp_path


def test_folders_are_listed_by_name_without_hidden_and_system_ones(drive):
    assert [f.name for f in list_folders(drive)] == ["Aufnahmen", "Musik"]
    assert list_folders(drive / "Musik") == []


@pytest.mark.parametrize(
    ("name", "valid"),
    [
        ("Aufnahmen", True),
        ("Radio 2026", True),
        ("Jörg's Mix", True),
        ("", False),
        (".", False),
        ("..", False),
        ("a/b", False),
        ("a:b", False),
        ("a?", False),
        ("a ", False),
        (" a", False),
        ("a.", False),
        ("a\tb", False),
        ("x" * 65, False),
    ],
)
def test_folder_names_every_file_system_takes(name, valid):
    assert valid_folder_name(name) is valid


def test_the_path_shown_names_the_drive_and_the_folders(drive):
    assert path_text("USB-STICK", drive, drive) == "USB-STICK"
    assert (
        path_text("USB-STICK", drive, drive / "Aufnahmen" / "2026")
        == "USB-STICK › Aufnahmen › 2026"
    )


@pytest.fixture
def browser(window, drive):
    browser = FolderBrowser(window)
    chosen = []
    browser.open("USB-STICK", drive, None, chosen.append)
    return browser, chosen


def test_the_browser_goes_down_and_up_and_takes_the_folder_shown(browser, drive):
    browser, chosen = browser
    assert browser.isVisible() and browser.directory == drive
    assert [row.folder.name for row in browser.rows()] == ["Aufnahmen", "Musik"]
    assert not browser._up.isEnabled()

    browser.rows()[0].tapped.emit()
    assert browser.directory == drive / "Aufnahmen"
    assert browser._path.text() == "USB-STICK › Aufnahmen"
    assert browser._up.isEnabled()
    browser.rows()[0].tapped.emit()
    assert browser.directory == drive / "Aufnahmen" / "2026"
    assert browser.rows() == [] and browser._empty.isVisibleTo(browser)

    browser.go_up()
    assert browser.directory == drive / "Aufnahmen"
    browser._accept.click()
    assert chosen == [drive / "Aufnahmen"]
    assert not browser.isVisible()


def test_the_browser_opens_at_the_directory_chosen_before_if_it_lies_on_the_drive(window, drive):
    browser = FolderBrowser(window)
    browser.open("USB-STICK", drive, drive / "Musik", lambda _d: None)
    assert browser.directory == drive / "Musik"
    browser.open("USB-STICK", drive, Path("/elsewhere"), lambda _d: None)
    assert browser.directory == drive
    browser.open("USB-STICK", drive, drive / "gone", lambda _d: None)
    assert browser.directory == drive


def test_a_new_folder_is_made_on_the_keyboard_and_entered(browser, drive):
    browser, _chosen = browser
    browser._new_folder.click()
    keyboard = browser.keyboard()
    assert keyboard.isVisible()
    for character in "Neu":
        keyboard.press(character)
    keyboard.accept()

    assert (drive / "Neu").is_dir()
    assert browser.directory == drive / "Neu"
    assert not keyboard.isVisible()

    browser.go_up()
    browser.create_folder("a/b")  # refused by the check, nothing made
    assert not (drive / "a").exists()
    assert browser.directory == drive


def test_cancelling_forgets_the_choice(browser):
    browser, chosen = browser
    browser.dismiss()
    assert not browser.isVisible()
    browser.accept()
    assert chosen == []


def test_the_browser_fits_the_screen(browser):
    browser, _chosen = browser
    assert browser.panel.width() <= SCREEN_WIDTH
    assert browser.panel.height() <= SCREEN_HEIGHT


# --- Storage view ---------------------------------------------------------------------


def test_the_entries_tell_what_is_mounted_and_where_the_recordings_go():
    manager = VolumeManager(bus=FakeBus(), auto_mount=False)
    manager.watch()
    recordings = manager.volume(STICK)
    entries = volume_entries(manager.volume_list, recordings)

    assert entries[0] == VolumeEntry(
        STICK,
        "USB-STICK",
        f"31 GB, vfat, mounted at {STICK_MOUNT}",
        True,
        True,
    )
    assert entries[1].name == "Intenso Speed Line"
    assert entries[1].detail == "64 GB, exfat, not mounted"
    assert not entries[1].mounted and not entries[1].holds_recordings


def test_the_directory_is_shown_within_its_drive():
    stick = Volume(STICK, "/dev/sda1", "USB-STICK", "", "vfat", 1, STICK_MOUNT)
    assert directory_text(None, None) == "No folder chosen"
    assert directory_text(Path(STICK_MOUNT) / "Aufnahmen", stick) == "USB-STICK › Aufnahmen"
    assert directory_text(Path(STICK_MOUNT), stick) == "USB-STICK"
    assert directory_text(Path("/media/pi/OTHER/x"), None) == str(Path("/media/pi/OTHER/x"))
    assert free_text((12_000_000_000, 31_000_000_000)) == "12 GB free of 31 GB"
    assert free_text(None) == ""


@pytest.fixture
def bus() -> FakeBus:
    return FakeBus()


@pytest.fixture
def view(window, bus, make_controller, data_directory):
    controller = make_controller(data_directory)
    manager = VolumeManager(bus=bus)
    view = StorageView(controller, manager, window)
    view.resize(SCREEN_WIDTH - 200, 520)
    view.show()  # reads everything, mounts the Intenso stick
    yield view
    manager.shutdown()


def _row(view: StorageView, name: str):
    return next(row for row in view.rows() if row.entry.name == name)


def test_the_sub_page_shows_no_directory_and_the_drives(view):
    assert view._directory.text() == "No folder chosen"
    assert not view._space.isVisibleTo(view)
    assert [e.name for e in view.entries()] == ["Intenso Speed Line", "USB-STICK"]
    assert all(e.mounted for e in view.entries())
    assert view.message.text().startswith("A USB drive plugged in")


def test_a_directory_on_a_drive_missing_is_shown_as_not_available(view, tmp_path):
    view._controller.set_recording_directory(Path(STICK_MOUNT) / "Aufnahmen")
    assert view._directory.text() == "USB-STICK › Aufnahmen"
    assert view._space.text() == "Drive not mounted"
    assert _row(view, "USB-STICK").entry.holds_recordings

    # A directory that is there (here a temporary one) shows its free space.
    view._controller.set_recording_directory(tmp_path)
    assert view._directory.text() == str(tmp_path)
    assert view._space.text().endswith("free of " + view._space.text().split(" of ")[1])
    assert " free of " in view._space.text()

    view._controller.set_recording_directory(None)
    assert view._directory.text() == "No folder chosen"
    assert not view._space.isVisibleTo(view)


def test_a_tap_on_a_mounted_drive_offers_to_eject_or_to_choose_a_folder(view, bus, tmp_path):
    _row(view, "USB-STICK").tapped.emit()
    sheet = view.sheet()
    assert sheet.isVisible() and sheet.title.text() == "USB-STICK"
    assert [b.text() for b in sheet.actions] == ["Eject", "Choose folder"]

    sheet.actions[0].click()
    assert bus.calls_of("Unmount") == [(STICK, FILESYSTEM, "Unmount", [MOUNT_OPTIONS])]
    assert bus.calls_of("PowerOff") == [(SANDISK, DRIVE, "PowerOff", [MOUNT_OPTIONS])]
    assert [e.name for e in view.entries()] == ["Intenso Speed Line"]


def test_choosing_a_folder_opens_the_browser_on_the_drive_and_stores_the_choice(
    view, bus, tmp_path
):
    # The Intenso stick's mount point is a directory that exists here.
    bus.objects[INTENSO_STICK][FILESYSTEM]["MountPoints"] = [str(tmp_path).encode() + b"\0"]
    bus.properties_changed.emit(
        INTENSO_STICK, FILESYSTEM, {"MountPoints": [str(tmp_path).encode() + b"\0"]}
    )
    (tmp_path / "Radio").mkdir()
    _row(view, "Intenso Speed Line").tapped.emit()
    view.sheet().actions[1].click()
    browser = view.browser()
    assert browser.isVisible() and browser.root == tmp_path
    assert [row.folder.name for row in browser.rows()] == ["Radio"]

    browser.rows()[0].tapped.emit()
    browser.accept()

    assert view._controller.recording_directory == tmp_path / "Radio"
    assert view._directory.text() == "Intenso Speed Line › Radio"
    assert " free of " in view._space.text()
    assert _row(view, "Intenso Speed Line").entry.holds_recordings

    # "Choose folder" beside the directory opens where it is.
    view._choose.click()
    assert browser.directory == tmp_path / "Radio"


def test_a_tap_on_a_drive_not_mounted_mounts_it(window, bus, make_controller, data_directory):
    controller = make_controller(data_directory)
    manager = VolumeManager(bus=bus, auto_mount=False)
    view = StorageView(controller, manager, window)
    view.show()
    assert not _row(view, "Intenso Speed Line").entry.mounted

    _row(view, "Intenso Speed Line").tapped.emit()
    assert bus.calls_of("Mount") == [(INTENSO_STICK, FILESYSTEM, "Mount", [MOUNT_OPTIONS])]
    assert _row(view, "Intenso Speed Line").entry.mounted
    manager.shutdown()


def test_without_a_mounted_drive_choosing_says_so(window, make_controller, data_directory):
    controller = make_controller(data_directory)
    manager = VolumeManager(bus=FakeBus(objects={}))
    view = StorageView(controller, manager, window)
    view.show()
    assert view._empty.text() == "No USB drive found."

    view._choose.click()
    assert view.message.text().startswith("No drive mounted")
    assert view.message.property("role") == "error"
    view._refresh_button.click()
    assert view.message.property("role") == "small"
    manager.shutdown()


def test_without_a_system_bus_the_page_lists_the_system_and_says_so(
    window, make_controller, data_directory, tmp_path
):
    controller = make_controller(data_directory)
    system = [Volume(str(tmp_path), "D:", "Stick", "", "exFAT", 64_000_000_000, str(tmp_path))]
    manager = VolumeManager(bus=FakeBus(connected=False), system=lambda: list(system))
    view = StorageView(controller, manager, window)
    view.show()

    assert [e.name for e in view.entries()] == ["Stick"]
    assert view.message.text().startswith("Here the system mounts")

    # Nothing to eject: a tap goes straight to the folders.
    _row(view, "Stick").tapped.emit()
    assert view.browser().isVisible() and view.browser().root == tmp_path
    manager.shutdown()


def test_an_action_in_progress_holds_the_list(view, bus):
    bus.holding = True
    _row(view, "USB-STICK").tapped.emit()
    view.sheet().actions[0].click()

    assert not view._list_panel.isEnabled()
    assert view._empty.text() == "Ejecting “USB-STICK” …"

    bus.release()
    assert view._list_panel.isEnabled()
