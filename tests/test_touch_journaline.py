"""Journaline on the touch surface: the model of the station's pages
(`ui.journaline_model`), the page in two columns and the card on the radio page
(`ui.touch.journaline_view`), on a controller whose DSP worker is told what it decoded."""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QEventLoop, QTimer
from PyQt6.QtGui import QFont, QFontMetrics
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget

from dab.announcement import AnnouncementStatus
from dab.dls import DlPlus, DlPlusContentType, DynamicLabel
from dab.fig import Announcement, AnnouncementSupport, AudioComponent
from dab.journaline import (
    JmlObject,
    JournalineParameters,
    LinkTarget,
    LinkType,
    MenuItem,
    ObjectType,
)
from dab.spi import Names, Programme
from net import wifi
from tests.fake_receiver import SERVICES_5C
from ui.journaline_model import JournalineModel
from ui.touch import backlight
from ui.touch.journaline_view import (
    JournalinePage,
    elided_lines,
    has_more,
    link_payload,
    page_count_text,
    split_topic,
)
from ui.touch.theme import touch_font
from ui.touch.window import SCREEN_HEIGHT, SCREEN_WIDTH, TouchWindow

MAIN = 0x0000
NEWS = 0x1770
PREVIEW = 0x1F40
CONTACT = 0x0001
IMPRINT = 0x0002
FIRST_NEWS = 0x0FA2
SECOND_NEWS = 0x0FA3


def _menu(object_id: int, title: str, *items: tuple[int, str]) -> JmlObject:
    entries = tuple(MenuItem(target, text) for target, text in items)
    return JmlObject(object_id, ObjectType.MENU, True, 0, False, title, menu_items=entries)


def _text(
    object_id: int,
    title: str,
    body: str = "Text.",
    *,
    revision: int = 0,
    links: tuple[LinkTarget, ...] = (),
) -> JmlObject:
    return JmlObject(
        object_id, ObjectType.PLAIN_TEXT, False, revision, True, title, body=body, links=links
    )


MAIN_MENU = _menu(
    MAIN,
    "Deutschlandfunk",
    (NEWS, "Nachrichten"),
    (PREVIEW, "Programmvorschau"),
    (CONTACT, "Kontakt"),
    (IMPRINT, "Impressum"),
)
NEWS_MENU = _menu(
    NEWS,
    "Deutschlandfunk - Die Nachrichten",
    (FIRST_NEWS, "Formel 1 - Verstappen gewinnt Grand Prix in Singapur"),
    (SECOND_NEWS, "Vatikan - Papst kündigt Lungenoperation an"),
)
FIRST = _text(FIRST_NEWS, "Formel 1 - Verstappen gewinnt Grand Prix in Singapur")
CONTACT_PAGE = _text(
    CONTACT,
    "Deutschlandfunk",
    "Raderberggürtel 40",
    links=(LinkTarget(LinkType.TELEPHONE, "0221 345-1831", "Hörerservice"),),
)
DLF = JournalineParameters(0, 0)
SPORT = 0x1774
EIGHT_O_CLOCK = 0x1775
# The programme preview of the Dlf: a programme with a text, and the news as a short
# message with only the link every entry has.
SPORT_PAGE = _text(SPORT, "19:10 Sport am Sonntag", "Live aus den Stadien.")
EIGHT_O_CLOCK_PAGE = JmlObject(
    EIGHT_O_CLOCK,
    ObjectType.TITLE_ONLY,
    False,
    0,
    True,
    "20:00 Nachrichten",
    links=(
        LinkTarget(
            LinkType.URL,
            "https://www.deutschlandfunk.de/programmvorschau.281.de.html",
            "Programmvorschau im Internet",
        ),
    ),
)
PREVIEW_MENU = _menu(
    PREVIEW,
    "Programmvorschau",
    (SPORT, "19:10 Sport am Sonntag"),
    (EIGHT_O_CLOCK, "20:00 Nachrichten"),
)


@pytest.fixture(scope="module", autouse=True)
def shipped_font():
    """The typeface of the touch surface as the application's font, as `main` sets it:
    the room in the column beside the slide depends on it."""
    app = QApplication.instance() or QApplication([])
    previous = app.font()
    app.setFont(touch_font())
    yield
    app.setFont(previous)


def _spin(milliseconds: int = 50) -> None:
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


@pytest.fixture
def controller(make_controller, data_directory):
    controller = make_controller(data_directory)
    controller.select_service(SERVICES_5C[0][0])
    return controller


@pytest.fixture
def model(controller):
    return JournalineModel(controller)


def _receive(controller, *objects: JmlObject, parameters=DLF) -> None:
    """What the DSP worker reports for the station: Journaline announced, pages decoded."""
    dsp = controller.dsp_worker
    if parameters is not None:
        dsp.journaline_parameters_ready.emit(parameters)
    for jml_object in objects:
        dsp.journaline_object_ready.emit(jml_object)


# --- Texts -------------------------------------------------------------------------------


def test_the_topic_of_a_headline_stands_apart():
    assert split_topic("Formel 1 - Verstappen gewinnt") == ("Formel 1", "Verstappen gewinnt")
    # Only the first dash parts: a quote after a second one stays in the headline.
    assert split_topic("Klausurtagung - CDU berät - Merz: Partei") == (
        "Klausurtagung",
        "CDU berät - Merz: Partei",
    )
    assert split_topic("Nachrichten") == ("", "Nachrichten")
    assert split_topic("Europride 2029-Dresden") == ("", "Europride 2029-Dresden")
    assert split_topic(" - Ohne Thema") == ("", " - Ohne Thema")


def test_a_long_headline_takes_two_lines_and_is_cut_short():
    metrics = QFontMetrics(QFont(QApplication.instance().font()))
    text = "Verdächtiger nach mutmaßlichen Anschlagsplänen auf polnischen Regierungschef"
    width = metrics.horizontalAdvance("Verdächtiger nach mutmaßlichen")

    lines = elided_lines(text, metrics, width, 2).split("\n")

    assert len(lines) == 2
    assert lines[0] == "Verdächtiger nach mutmaßlichen"
    assert lines[1].endswith("…")
    assert all(metrics.horizontalAdvance(line) <= width for line in lines)
    assert elided_lines("Nepal", metrics, width, 2) == "Nepal"


def test_more_than_a_headline_is_a_menu_a_text_a_list_or_a_link_to_a_page():
    assert has_more(MAIN_MENU)
    assert has_more(SPORT_PAGE)
    assert has_more(
        JmlObject(0x10, ObjectType.LIST, False, 0, True, "Liste", list_items=(("19:00", "N"),))
    )
    linked = JmlObject(
        0x11,
        ObjectType.TITLE_ONLY,
        False,
        0,
        True,
        "Siehe Kontakt",
        links=(LinkTarget(LinkType.JML_OBJECT, "", "Kontakt", CONTACT),),
    )
    assert has_more(linked)
    # A link to the web alone is no more: the Dlf gives every preview entry the same.
    assert not has_more(EIGHT_O_CLOCK_PAGE)
    assert not has_more(_text(0x12, "Leer", "  "))
    assert not has_more(_menu(0x13, "Leeres Menü"))


def test_a_qr_code_holds_what_a_phone_can_open():
    def payload(link_type: LinkType, address: str) -> str | None:
        return link_payload(LinkTarget(link_type, address, ""))

    url = "https://www.deutschlandfunk.de/nachrichten-100.html"
    assert payload(LinkType.URL, url) == url
    assert payload(LinkType.URL, "www.deutschlandfunk.de") == "http://www.deutschlandfunk.de"
    assert payload(LinkType.URL, "mailto:hoererservice@deutschlandradio.de") == (
        "mailto:hoererservice@deutschlandradio.de"
    )
    assert payload(LinkType.TELEPHONE, "+492213451831") == "tel:+492213451831"
    assert payload(LinkType.TELEPHONE, "0221 - 345 18 31") == "tel:02213451831"
    assert payload(LinkType.SMS, "+49 171 1234567") == "sms:+491711234567"
    # Nothing a phone could open.
    assert payload(LinkType.DAB_URI, "dab://5C/D210") is None
    assert payload(LinkType.TELEPHONE, "Hörertelefon") is None
    assert payload(LinkType.URL, " ") is None
    assert link_payload(LinkTarget(LinkType.JML_OBJECT, "", "Kontakt", CONTACT)) is None


def test_the_pages_are_counted():
    assert page_count_text(1) == "1 page"
    assert page_count_text(30) == "30 pages"


# --- Model -------------------------------------------------------------------------------


def test_the_model_keeps_the_pages_of_the_station_playing(controller, model):
    changes = []
    model.changed.connect(changes.append)
    assert not model.announced

    _receive(controller, MAIN_MENU, NEWS_MENU)

    assert model.announced
    assert model.entry == MAIN
    assert model.get(NEWS) == NEWS_MENU
    assert [o.object_id for o in model.objects()] == [MAIN, NEWS]
    assert model.received_at(NEWS) == pytest.approx(time.time(), abs=5)
    assert changes == [None, frozenset({MAIN}), frozenset({NEWS})]

    # The station repeats its pages all the time: an unchanged one changes nothing.
    controller.dsp_worker.journaline_object_ready.emit(MAIN_MENU)
    assert len(changes) == 3


def test_another_station_voids_the_pages(controller, model):
    _receive(controller, MAIN_MENU)
    changes = []
    model.changed.connect(changes.append)

    controller.select_service(SERVICES_5C[1][0])

    assert not model.announced
    assert model.get(MAIN) is None
    assert changes == [None]


def test_the_point_of_entry_comes_from_fig_0_13(controller, model):
    _receive(controller, parameters=JournalineParameters(0, 0, entry_object=NEWS))

    assert model.entry == NEWS


def test_a_page_past_its_timeout_goes(controller, model):
    gone = JmlObject(
        CONTACT, ObjectType.PLAIN_TEXT, True, 0, False, "Kontakt", absolute_timeout=time.time() - 1
    )
    _receive(controller, MAIN_MENU, gone)
    changes = []
    model.changed.connect(changes.append)

    model._expire()

    assert model.get(CONTACT) is None
    assert changes == [frozenset({CONTACT})]


# --- Page --------------------------------------------------------------------------------


@pytest.fixture
def page(model):
    window = QWidget()
    window.resize(SCREEN_WIDTH, SCREEN_HEIGHT)
    page = JournalinePage(model)
    QVBoxLayout(window).addWidget(page)
    window.show()
    yield page
    window.deleteLater()


def _texts(rows) -> list[str]:
    return [row.headline.text().replace("\n", " ") for row in rows]


def test_the_page_starts_at_the_main_menu_with_its_first_entry(controller, page):
    _receive(controller, MAIN_MENU, NEWS_MENU)

    assert page.path == (MAIN,)
    assert page.path_label.text() == "Deutschlandfunk"
    assert not page.up_button.isEnabled()
    assert _texts(page.rows) == ["Nachrichten", "Programmvorschau", "Kontakt", "Impressum"]
    assert page.selected == NEWS
    assert page.rows[0].property("selected")
    # The news on the right, their topics small above the headlines.
    assert _texts(page.right_rows) == [
        "Verstappen gewinnt Grand Prix in Singapur",
        "Papst kündigt Lungenoperation an",
    ]
    assert page.right_rows[0].topic.text() == "Formel 1"


def test_pages_not_received_yet_are_muted_and_come_when_they_arrive(controller, page):
    _receive(controller, MAIN_MENU, NEWS_MENU)
    imprint = page.rows[3]
    assert imprint.headline.property("role") == "muted"

    imprint.tapped.emit()
    assert page.selected == IMPRINT
    assert page.shown is None

    _receive(controller, _text(IMPRINT, "Deutschlandradio"), parameters=None)
    assert page.shown.object_id == IMPRINT
    assert page.rows[3].headline.property("role") != "muted"


def test_an_arrow_marks_the_entries_with_more_behind_them(controller, page):
    _receive(controller, MAIN_MENU, NEWS_MENU, PREVIEW_MENU, SPORT_PAGE, EIGHT_O_CLOCK_PAGE)
    # Menus have more; the imprint has not arrived, so nothing is known of it.
    assert [row.more for row in page.rows] == [True, True, False, False]

    page.rows[1].tapped.emit()
    assert [row.more for row in page.right_rows] == [True, False]

    # The short message can be tapped all the same: it shows its link.
    page.right_rows[1].tapped.emit()
    assert page.path == (MAIN, PREVIEW)
    assert page.shown == EIGHT_O_CLOCK_PAGE


def test_a_menu_on_the_right_moves_to_the_left_and_back(controller, page):
    _receive(controller, MAIN_MENU, NEWS_MENU, FIRST)

    page.right_rows[0].tapped.emit()

    assert page.path == (MAIN, NEWS)
    # The menu named as the main menu names it, not by its own title.
    assert page.path_label.text() == "Deutschlandfunk › Nachrichten"
    assert page.selected == FIRST_NEWS
    assert page.shown == FIRST
    assert page.up_button.isEnabled()

    page.up_button.click()
    assert page.path == (MAIN,)
    assert page.selected == NEWS


def test_a_changed_message_waits_until_asked_for(controller, page):
    _receive(controller, MAIN_MENU, NEWS_MENU, FIRST)
    page.right_rows[0].tapped.emit()
    assert page.update_button is None

    changed = _text(FIRST.object_id, FIRST.title, "Neuer Text.", revision=1)
    _receive(controller, changed, parameters=None)

    assert page.shown == FIRST
    assert page.update_button is not None
    page.update_button.click()
    assert page.shown == changed
    assert page.update_button is None


def test_a_change_out_of_sight_leaves_the_columns_alone(controller, page):
    _receive(controller, MAIN_MENU, NEWS_MENU)
    rows = page.rows

    _receive(controller, _text(0x5000, "Woanders"), parameters=None)

    assert page.rows is rows


def test_a_link_to_another_page_is_followed(controller, page):
    linked = _text(
        FIRST_NEWS, FIRST.title, links=(LinkTarget(LinkType.JML_OBJECT, "", "Kontakt", CONTACT),)
    )
    _receive(controller, MAIN_MENU, NEWS_MENU, linked, CONTACT_PAGE)
    page.right_rows[0].tapped.emit()

    page.link_rows[0].tapped.emit()

    assert page.shown == CONTACT_PAGE
    assert page.path == (MAIN, NEWS)


def test_a_link_to_the_web_or_a_number_shows_its_qr_code(controller, page):
    links = (
        LinkTarget(LinkType.TELEPHONE, "+492213451831", "Hörerservice"),
        LinkTarget(LinkType.DAB_URI, "dab://5C/D220", "Dlf Kultur"),
        LinkTarget(LinkType.URL, "http://www.deutschlandfunk.de", "www.deutschlandfunk.de"),
    )
    contact = _text(CONTACT, "Deutschlandfunk", "Raderberggürtel 40", links=links)
    _receive(controller, MAIN_MENU, contact)
    page.rows[2].tapped.emit()
    # The DAB link is text only: a phone could not open it.
    assert len(page.qr_rows) == 2

    page.qr_rows[0].tapped.emit()
    sheet = page.qr_sheet()
    assert sheet.isVisible()
    assert sheet.payload == "tel:+492213451831"
    assert sheet.title.text() == "Hörerservice"

    page.qr_rows[1].tapped.emit()
    assert sheet.payload == "http://www.deutschlandfunk.de"

    # It goes with the page.
    page.hide()
    assert not sheet.isVisible()


def test_another_station_starts_the_page_afresh(controller, page):
    _receive(controller, MAIN_MENU, NEWS_MENU, FIRST)
    page.right_rows[0].tapped.emit()

    controller.select_service(SERVICES_5C[1][0])

    assert page.path == (MAIN,)
    assert page.selected is None
    assert page.rows == []


# --- On the radio page -------------------------------------------------------------------


@pytest.fixture
def touch_window(make_controller, data_directory, monkeypatch):
    monkeypatch.setattr(wifi, "find_nmcli", lambda: None)
    monkeypatch.setattr(backlight, "BACKLIGHT_CLASS", Path("nowhere") / "backlight")
    controller = make_controller(data_directory, start=False)
    window = TouchWindow(controller)
    controller.start()
    controller.select_service(SERVICES_5C[0][0])
    window.show()
    _spin()
    yield window
    window.deleteLater()


def test_the_card_shows_while_the_station_sends_journaline(touch_window):
    radio = touch_window._radio_page
    card = radio.journaline_card
    assert card.isHidden()

    _receive(touch_window._controller, MAIN_MENU)
    _spin()

    assert not card.isHidden()
    assert card.detail.text() == "Nachrichten · Programmvorschau · Kontakt · Impressum"
    assert card.count.text() == "1 page"


def test_the_card_opens_the_pages_and_another_station_closes_them(touch_window):
    controller = touch_window._controller
    radio = touch_window._radio_page
    _receive(controller, MAIN_MENU, NEWS_MENU)
    _spin()

    radio.journaline_card.tapped.emit()
    assert radio.in_journaline()
    radio.journaline.back_button.click()
    assert not radio.in_journaline()

    radio.journaline_card.tapped.emit()
    controller.select_service(SERVICES_5C[1][0])
    _spin()
    assert not radio.in_journaline()
    assert radio.journaline_card.isHidden()


# News for cluster 2 in subchannel 5, as the Dlf flags its news on the hour.
NEWS_STATUS = AnnouncementStatus(
    AnnouncementSupport(0x0010, (2,)),
    alarm_enabled=False,
    active=Announcement(2, 0x0010, new=True, subchannel_id=5),
    source_sid=0xD210,
    source_component=AudioComponent(5, 63),
)


def test_the_card_gives_way_where_the_column_has_no_room(touch_window, monkeypatch):
    controller = touch_window._controller
    radio = touch_window._radio_page
    card = radio.journaline_card
    # The guide has a programme running and the one after it: the guide card shows.
    now = time.time()
    schedule = [
        Programme(Names(long="Sport am Sonntag"), now - 600, 3000),
        Programme(Names(long="Nachrichten"), now + 2400, 300),
    ]
    monkeypatch.setattr(touch_window._guide_model, "programmes", lambda _sid: schedule)
    radio._load_schedule()
    _receive(controller, MAIN_MENU)
    _spin()
    assert not radio._guide_card.isHidden()
    assert not card.isHidden()

    # An announcement still leaves room for it.
    controller.announcement_changed.emit(NEWS_STATUS, "Dlf")
    _spin()
    assert not card.isHidden()

    # Two lines of DL Plus beside it do not.
    tags = {
        DlPlusContentType.ITEM_ARTIST: "Max Mustermann und das Orchester der Stadt",
        DlPlusContentType.ITEM_TITLE: "Ein sehr langer Titel",
    }
    label = DynamicLabel("Jetzt läuft", DlPlus(False, True, tags))
    controller.dsp_worker.dynamic_label_ready.emit(label)
    _spin()
    assert card.isHidden()

    # Back as soon as the announcement is over.
    controller.announcement_changed.emit(None, None)
    _spin()
    assert not card.isHidden()
