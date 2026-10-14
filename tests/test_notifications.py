"""Notifications from outside the radio: the centre that holds them
(`ui.notifications`) and the commands `notify` and `dismiss_notification` of the state
model (`net.radio_state`) with the section `notifications`, on a controller without a
device (`tests.fake_receiver`)."""

from __future__ import annotations

import base64
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QBuffer, QByteArray, QEventLoop, QIODevice, QTimer
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QApplication

from net.radio_state import BRIEF_LENGTH, RadioStateModel, brief_arguments
from ui.notifications import (
    CLOSED,
    DEFAULT_BUTTON_ID,
    DISMISSED,
    MAX_BUTTONS,
    MAX_IMAGE_SIDE,
    MAX_OPEN,
    PRESSED,
    REPLACED,
    TIMEOUT,
    Level,
    Notification,
    NotificationAnswer,
    NotificationButton,
    NotificationCenter,
)

START = 1_791_000_000.0
OPEN = NotificationButton("open", "Open", "accent")
IGNORE = NotificationButton("ignore", "Ignore")


def _spin(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def _png(width: int = 40, height: int = 30) -> bytes:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor("orange"))
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return bytes(data)


@pytest.fixture
def center():
    # Kept while the test runs: without an application the centre goes with its timers.
    _app = QApplication.instance() or QApplication([])
    center = NotificationCenter(clock=lambda: START)
    answers: list[NotificationAnswer] = []
    center.answered.connect(answers.append)
    yield center, answers


# --- The centre -----------------------------------------------------------------------


def test_a_notification_waits_for_its_button(center):
    center, answers = center
    posted = []
    center.posted.connect(posted.append)

    shown = center.post(Notification("door", "Front door", "It rang.", (OPEN, IGNORE)))

    assert shown.received == START
    assert center.shown is shown
    assert posted == [shown]
    center.press("door", "nonsense")
    assert center.shown is shown
    center.press("door", "open")
    assert center.shown is None
    assert answers == [NotificationAnswer("door", PRESSED, "open", "Open")]
    assert answers[0].json() == {
        "id": "door",
        "reason": "pressed",
        "button": "open",
        "label": "Open",
    }
    center.press("door", "open")  # answered already
    assert len(answers) == 1


def test_without_buttons_ok_answers(center):
    center, answers = center
    center.post(Notification("wash", "", "The washing is done."))

    center.press("wash", DEFAULT_BUTTON_ID)

    assert answers == [NotificationAnswer("wash", PRESSED, "ok", "OK")]


def test_alarms_go_first_and_the_rest_in_order(center):
    center, answers = center
    center.post(Notification("a", "A", ""))
    center.post(Notification("b", "B", "", level=Level.WARNING))
    center.post(Notification("c", "C", ""))
    assert [n.id for n in center.open] == ["b", "a", "c"]

    center.post(Notification("d", "D", "", level=Level.ALARM))

    assert center.shown.id == "d"
    assert [n.id for n in center.open] == ["d", "b", "a", "c"]
    center.press("d", DEFAULT_BUTTON_ID)
    assert center.shown.id == "b"
    assert answers == [NotificationAnswer("d", PRESSED, "ok", "OK")]


def test_the_same_id_takes_the_place_of_the_open_one(center):
    center, answers = center
    center.post(Notification("door", "Front door", "It rang."))
    center.post(Notification("later", "Later", ""))

    center.post(Notification("door", "Front door", "It rang twice."))

    assert answers == [NotificationAnswer("door", REPLACED)]
    assert [n.id for n in center.open] == ["door", "later"]
    assert center.shown.message == "It rang twice."


def test_dismiss_and_close_all(center):
    center, answers = center
    center.post(Notification("a", "A", ""))
    center.post(Notification("b", "B", ""))
    changes = []
    center.changed.connect(lambda: changes.append(len(center.open)))

    assert center.dismiss("a") is True
    assert center.dismiss("a") is False
    center.close_all()

    assert answers == [NotificationAnswer("a", DISMISSED), NotificationAnswer("b", CLOSED)]
    assert changes == [1, 0]
    assert center.shown is None


def test_the_time_runs_out(center):
    center, answers = center
    center.post(Notification("door", "Front door", "", timeout_s=0.05))
    center.post(Notification("stays", "Stays", ""))
    notification = center.get("door")
    assert notification.expires == START + 0.05

    _spin(150)

    assert answers == [NotificationAnswer("door", TIMEOUT)]
    assert [n.id for n in center.open] == ["stays"]


def test_a_replacement_starts_its_own_time(center):
    center, answers = center
    center.post(Notification("door", "Front door", "", timeout_s=0.05))
    center.post(Notification("door", "Front door", "Again"))

    _spin(150)

    assert answers == [NotificationAnswer("door", REPLACED)]
    assert center.shown.message == "Again"


def test_at_most_so_many_are_open(center):
    center, _answers = center
    for number in range(MAX_OPEN):
        center.post(Notification(str(number), "N", ""))

    with pytest.raises(ValueError, match="open already"):
        center.post(Notification("one too many", "N", ""))
    center.post(Notification("0", "Replaced", ""))  # a replacement still goes
    assert len(center.open) == MAX_OPEN


# --- The commands of the state model --------------------------------------------------


@pytest.fixture
def radio(data_directory, make_controller):
    controller = make_controller(data_directory, start=False)
    model = RadioStateModel(controller)
    controller.start()
    answers: list[dict] = []
    model.notification_answered.connect(answers.append)
    return model, controller, answers


def test_notify_shows_and_the_answer_is_reported(radio):
    model, controller, answers = radio
    changes = []
    model.changed.connect(changes.append)

    assert (
        model.execute(
            "notify",
            {
                "id": "door",
                "title": " Front door ",
                "message": "It rang.\nOpen?",
                "buttons": [
                    {"id": "open", "label": "Open", "style": "Accent"},
                    "Ignore",
                    {"label": "Later"},
                ],
                "level": "WARNING",
                "timeout": 60,
            },
        )
        is None
    )

    shown = controller.notifications.shown
    assert shown.title == "Front door"
    assert shown.message == "It rang.\nOpen?"
    assert shown.buttons == (
        NotificationButton("open", "Open", "accent"),
        NotificationButton("2", "Ignore"),
        NotificationButton("3", "Later"),
    )
    assert shown.level is Level.WARNING
    assert shown.timeout_s == 60
    assert "notifications" in changes
    section = model.section("notifications")
    assert section["shown"] == "door"
    [entry] = section["open"]
    assert entry["buttons"][1] == {"id": "2", "label": "Ignore", "style": ""}
    assert entry["level"] == "warning"
    assert entry["image"] is False
    assert entry["expires"] == entry["received"] + 60

    controller.notifications.press("door", "2")
    assert answers == [{"id": "door", "reason": "pressed", "button": "2", "label": "Ignore"}]
    assert model.section("notifications") == {"shown": None, "open": []}


def test_notify_without_id_gets_one_and_dismiss_takes_it_back(radio):
    model, controller, answers = radio

    assert model.execute("notify", {"message": "The washing is done."}) is None
    notification_id = controller.notifications.shown.id
    assert notification_id
    assert controller.notifications.shown.buttons == ()
    assert controller.notifications.shown.timeout_s is None

    assert model.execute("dismiss_notification", {"id": notification_id}) is None
    assert answers == [
        {"id": notification_id, "reason": "dismissed", "button": None, "label": None}
    ]
    assert model.execute("dismiss_notification", {"id": notification_id}) == (
        f"No notification “{notification_id}” is open."
    )


@pytest.mark.parametrize(
    ("arguments", "answer"),
    [
        ({}, "A notification needs a “title” or a “message”."),
        ({"title": "  "}, "A notification needs a “title” or a “message”."),
        ({"title": 5}, "“title” must be a string, not 5."),
        ({"title": "x" * 121}, "“title” must not be longer than 120 characters."),
        ({"message": "x" * 1001}, "“message” must not be longer than 1000 characters."),
        ({"title": "T", "buttons": "Open"}, "“buttons” must be a list."),
        (
            {"title": "T", "buttons": ["A"] * (MAX_BUTTONS + 1)},
            f"A notification has at most {MAX_BUTTONS} buttons.",
        ),
        ({"title": "T", "buttons": [{"id": "a"}]}, "Button 1 has no “label”."),
        ({"title": "T", "buttons": [3]}, "Button 1 must be an object or a label."),
        (
            {"title": "T", "buttons": [{"id": "a", "label": "A"}, {"id": "a", "label": "B"}]},
            "Two buttons have the id “a”.",
        ),
        (
            {"title": "T", "buttons": [{"label": "A", "style": "red"}]},
            "“style” must be “accent”, “danger” or empty, not “red”.",
        ),
        (
            {"title": "T", "level": "panic"},
            "“level” must be “info”, “warning”, “alarm”, not “panic”.",
        ),
        ({"title": "T", "timeout": -1}, "“timeout” must not be negative."),
        ({"title": "T", "timeout": "soon"}, "“timeout” must be a number, not 'soon'."),
        ({"title": "T", "timeout": 100_000}, "“timeout” must be at most 86400 seconds."),
        ({"title": "T", "image": "not base64!"}, "“image” is not valid base64."),
        (
            {"title": "T", "image": base64.b64encode(b"no picture").decode()},
            "“image” is not an image (JPEG, PNG, GIF or BMP).",
        ),
        ({"title": "T", "image": 7}, "“image” must be a string of base64."),
    ],
)
def test_notify_checks_its_arguments(radio, arguments, answer):
    model, controller, _answers = radio

    assert model.execute("notify", arguments) == answer
    assert controller.notifications.shown is None


def test_notify_takes_an_image_and_scales_it_down(radio):
    model, controller, _answers = radio
    encoded = base64.b64encode(_png(2 * MAX_IMAGE_SIDE, 600)).decode()

    # As a data URL, wrapped over lines as base64 tools write it.
    wrapped = "\n".join(encoded[i : i + 76] for i in range(0, len(encoded), 76))
    assert (
        model.execute("notify", {"title": "Door", "image": "data:image/png;base64," + wrapped})
        is None
    )

    image = controller.notifications.shown.image
    assert (image.width(), image.height()) == (MAX_IMAGE_SIDE, 300)
    assert model.section("notifications")["open"][0]["image"] is True


def test_timeout_zero_means_until_answered(radio):
    model, controller, _answers = radio

    assert model.execute("notify", {"title": "T", "timeout": 0}) is None
    assert controller.notifications.shown.timeout_s is None


def test_brief_arguments_shorten_an_image():
    image = "A" * (BRIEF_LENGTH + 1)
    message = "M" * BRIEF_LENGTH

    brief = brief_arguments({"image": image, "message": message, "buttons": ["Open", image]})

    assert brief["image"] == "A" * 40 + f"… ({BRIEF_LENGTH + 1} characters)"
    assert brief["message"] == message
    assert brief["buttons"][0] == "Open"
    assert brief["buttons"][1].endswith("characters)")


def test_the_controller_answers_every_notification_as_it_shuts_down(
    data_directory, make_controller
):
    controller = make_controller(data_directory)
    answers = []
    controller.notifications.answered.connect(answers.append)
    controller.notifications.post(Notification("door", "Front door", ""))

    controller.shutdown()

    assert answers == [NotificationAnswer("door", CLOSED)]


def test_without_remote_control_there_is_no_mqtt(data_directory, make_controller):
    controller = make_controller(data_directory, remote_control=False)

    assert controller.remote_state is None
    assert controller.mqtt_bridge is None
    assert controller.notifications.open == ()
