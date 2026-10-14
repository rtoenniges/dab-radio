"""The notification dialog of the touch surface (`ui.touch.notification_view`): what it
shows of a notification, that a button answers it only after its pause, and that the
window shows notifications from the controller and wakes the screen for them."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QWidget

from net import wifi
from ui.notifications import (
    PRESSED,
    Level,
    Notification,
    NotificationAnswer,
    NotificationButton,
    NotificationCenter,
)
from ui.touch import backlight
from ui.touch.notification_view import (
    GUARD_MS,
    PANEL_WIDTH,
    PANEL_WIDTH_WITH_IMAGE,
    NotificationDialog,
    button_text,
    countdown_interval_ms,
)
from ui.touch.window import TouchWindow

DOOR = Notification(
    "door",
    "Front door",
    "It rang.",
    (NotificationButton("open", "Open", "accent"), NotificationButton("rock", "Rock & Roll")),
)


@pytest.fixture
def dialog():
    """A dialog over a widget of the size of the screen, its pause switched off."""
    _app = QApplication.instance() or QApplication([])
    parent = QWidget()
    parent.resize(1280, 720)
    parent.show()
    center = NotificationCenter()
    answers: list[NotificationAnswer] = []
    center.answered.connect(answers.append)
    dialog = NotificationDialog(parent, center, guard_ms=0)
    yield dialog, center, answers
    parent.deleteLater()


def test_it_shows_the_notification_shown_and_answers_with_its_button(dialog):
    dialog, center, answers = dialog
    assert not dialog.isVisible()

    center.post(DOOR)

    assert dialog.isVisible()
    assert dialog.title.text() == "Front door"
    assert dialog.message.text() == "It rang."
    assert [button.text() for button in dialog.buttons] == ["Open", "Rock && Roll"]
    assert dialog.buttons[0].property("role") == "accent"
    assert dialog.level_bar.property("level") == "info"
    assert not dialog.image.isVisible()
    assert dialog.panel.width() == PANEL_WIDTH
    assert not dialog.waiting.isVisible()
    QTest.mouseClick(dialog.buttons[1], Qt.MouseButton.LeftButton)
    assert answers == [NotificationAnswer("door", PRESSED, "rock", "Rock & Roll")]
    assert not dialog.isVisible()


def test_texts_are_plain(dialog):
    dialog, center, _answers = dialog

    center.post(Notification("x", "<b>bold</b>", "<img src='file:///etc/passwd'>"))

    assert dialog.title.textFormat() == Qt.TextFormat.PlainText
    assert dialog.message.textFormat() == Qt.TextFormat.PlainText
    assert dialog.title.text() == "<b>bold</b>"


def test_without_buttons_it_offers_ok_and_without_title_shows_the_message(dialog):
    dialog, center, answers = dialog

    center.post(Notification("wash", "", "The washing is done."))

    assert not dialog.title.isVisible()
    assert [button.text() for button in dialog.buttons] == ["OK"]
    QTest.mouseClick(dialog.buttons[0], Qt.MouseButton.LeftButton)
    assert answers == [NotificationAnswer("wash", PRESSED, "ok", "OK")]


def test_a_tap_beside_it_leaves_it_open(dialog):
    dialog, center, answers = dialog
    center.post(DOOR)

    QTest.mouseClick(dialog, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))

    assert dialog.isVisible()
    assert answers == []


def test_the_others_wait_and_an_alarm_comes_first(dialog):
    dialog, center, _answers = dialog
    center.post(DOOR)
    center.post(Notification("wash", "Washing machine", ""))
    assert dialog.notification.id == "door"
    assert dialog.waiting.isVisible()
    assert dialog.waiting.text() == "1 more waiting"

    center.post(Notification("smoke", "Smoke", "", level=Level.ALARM))

    assert dialog.notification.id == "smoke"
    assert dialog.level_bar.property("level") == "alarm"
    assert dialog.waiting.text() == "2 more waiting"
    center.dismiss("smoke")
    assert dialog.notification.id == "door"
    assert [button.text() for button in dialog.buttons] == ["Open", "Rock && Roll"]


def test_an_image_stands_beside_the_texts(dialog):
    dialog, center, _answers = dialog
    image = QImage(640, 480, QImage.Format.Format_RGB32)
    image.fill(QColor("orange"))

    center.post(Notification("door", "Front door", "It rang.", image=image))

    assert dialog.image.isVisible()
    assert dialog.panel.width() == PANEL_WIDTH_WITH_IMAGE
    pixmap = dialog.image.pixmap()
    assert pixmap.deviceIndependentSize().width() == 420
    assert pixmap.deviceIndependentSize().height() == 315
    # The panel stays within the screen.
    assert dialog.panel.geometry().top() >= 0
    assert dialog.panel.geometry().bottom() < 720


def test_its_buttons_wait_for_the_pause(dialog):
    _dialog, center, answers = dialog
    parent = QWidget()
    parent.resize(1280, 720)
    paused = NotificationDialog(parent, center)
    center.post(DOOR)

    QTest.mouseClick(paused.buttons[0], Qt.MouseButton.LeftButton)
    assert answers == []
    QTest.qWait(GUARD_MS + 50)
    QTest.mouseClick(paused.buttons[0], Qt.MouseButton.LeftButton)
    assert answers == [NotificationAnswer("door", PRESSED, "open", "Open")]
    parent.deleteLater()


def test_button_text_keeps_an_ampersand():
    assert button_text("A & B && C") == "A && B &&&& C"


def test_the_window_shows_what_comes_over_mqtt_and_wakes_the_screen(
    data_directory, make_controller, monkeypatch
):
    monkeypatch.setattr(wifi, "find_nmcli", lambda: None)
    monkeypatch.setattr(backlight, "BACKLIGHT_CLASS", Path("nowhere") / "backlight")
    controller = make_controller(data_directory, start=False)
    window = TouchWindow(controller)
    controller.start()
    window.show()
    woken = []
    monkeypatch.setattr(window._screen, "touched", lambda: woken.append(True))

    answer = controller.remote_state.execute(
        "notify", {"id": "door", "title": "Front door", "buttons": ["Open", "Ignore"]}
    )

    assert answer is None
    dialog = window._notification_dialog
    assert dialog.isVisible()
    assert dialog.geometry() == window.rect()
    assert dialog.title.text() == "Front door"
    assert woken == [True]
    window.close()


def test_the_bar_shrinks_from_the_right_as_the_time_runs_out():
    _app = QApplication.instance() or QApplication([])
    parent = QWidget()
    parent.resize(1280, 720)
    parent.show()
    now = [1000.0]
    center = NotificationCenter(clock=lambda: now[0])
    dialog = NotificationDialog(parent, center, guard_ms=0)

    center.post(Notification("door", "Front door", "", timeout_s=60))
    track = dialog.level_track
    assert dialog.counting_down
    assert dialog.level_bar.geometry().left() == 0
    assert dialog.level_bar.width() == track.width() == PANEL_WIDTH - 48
    now[0] += 15
    QTest.qWait(countdown_interval_ms(60, track.width()) + 50)
    assert dialog.level_bar.geometry().left() == 0
    assert dialog.level_bar.width() == round(track.width() * 0.75)

    # Without a timeout the bar stays whole, and an alarm in front stops nothing that
    # comes back: the time of the first goes on where it stands.
    center.post(Notification("smoke", "Smoke", "", level=Level.ALARM))
    assert not dialog.counting_down
    assert dialog.level_bar.width() == track.width()
    now[0] += 15
    center.dismiss("smoke")
    assert dialog.counting_down
    assert dialog.level_bar.width() == round(track.width() * 0.5)
    center.dismiss("door")
    assert not dialog.counting_down
    parent.deleteLater()


def test_the_bar_is_redrawn_once_a_pixel_within_bounds():
    assert countdown_interval_ms(60, 792) == 76
    assert countdown_interval_ms(5, 792) == 33
    assert countdown_interval_ms(3600, 792) == 1000


def test_a_notification_in_place_of_the_shown_one_keeps_its_height(dialog):
    dialog, center, _answers = dialog
    center.post(DOOR)
    QApplication.processEvents()
    height = dialog.panel.height()

    # The same again: its buttons come anew into a panel already shown.
    center.post(DOOR)

    assert all(button.isVisible() for button in dialog.buttons)
    assert dialog.panel.height() == height
    center.post(Notification("smoke", "Smoke", "", (NotificationButton("ack", "OK"),)))
    center.post(Notification("fire", "Front door", "It rang.", DOOR.buttons, level=Level.ALARM))
    assert dialog.notification.id == "fire"
    assert all(button.isVisible() for button in dialog.buttons)
    assert dialog.panel.height() == height
