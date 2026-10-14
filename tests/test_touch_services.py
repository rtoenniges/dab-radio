"""The services page of the touch surface: the data services of the ensemble
(`ui.touch.data_services_view`) and the programme guide (`ui.touch.guide_view` on
`ui.guide_model`), with the schedule of Dlf Nova as broadcast on 5C."""

from __future__ import annotations

import os
import time
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QEvent, QEventLoop, QPoint, QPointF, Qt, QTimer
from PyQt6.QtGui import QMouseEvent

from dab.data_group import parse_data_group
from dab.data_services import USER_APPLICATION_TPEG, data_components
from dab.ensemble_cache import SOURCE_SCAN, EnsembleCache, EnsembleInfo
from dab.guide_cache import GuideCache
from dab.mot_directory import MotCarousel
from dab.spi import decode_spi_object
from dab.timecode import BroadcastTime, ReferenceClock
from dsp.packet import AddressStatistics, PacketStatistics
from dsp.pipeline import CarouselObject, DataComponentStatus
from tests.fake_receiver import ENSEMBLE_5C, SERVICES_5C
from tests.spi_vectors import (
    NDR_DIRECTORY_GROUP_0,
    NDR_DIRECTORY_GROUP_1,
    NDR_DIRECTORY_GROUP_2,
    NDR_SI_GROUP_0,
    NDR_SI_GROUP_1,
    NDR_SI_GROUP_2,
)
from tests.test_combined_guide import DLF_NOVA_PI, EID_5C, _service, _si
from tests.test_data_services import FIGS_5C, _ensemble
from ui import guide_model, receiver_controller
from ui.guide_model import GuideModel, now_and_next
from ui.receiver_controller import ReceptionState
from ui.touch.data_services_view import DataServicesView
from ui.touch.guide_view import (
    _ENTRY_ROLE,
    _SCHEDULE_ROLE,
    CAROUSEL,
    EXPAND_MARK,
    SCHEDULE,
    GuideView,
    _ProgrammeRow,
    day_title,
    duration_text,
)
from ui.touch.icons import symbol_pixmap
from ui.touch.radio_page import clock_text, starts_text
from ui.touch.services_page import (
    DATA_SERVICES,
    GNSS,
    GUIDE,
    MENUS,
    Menu,
    menu_decoding,
    menu_offered,
)
from ui.touch.theme import DARK, THEMES, stylesheet
from ui.touch.window import SERVICES, TouchWindow

DLF_NOVA = 0xD230
SI_5C = _si(EID_5C, _service("Dlf Nova", DLF_NOVA, EID_5C))


def _spin(milliseconds: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def _finger(row, start: QPointF, end: QPointF | None = None) -> None:
    """Down at `start`, lifted at `end` (where it came down: a tap), straight to the row:
    `QTest.mouseClick` goes through the QScroller of the list, which holds the press back
    to tell a tap from a swipe. Freed with the view at the end of the test while that was
    pending, it crashed later tests now and then (2 of 25 runs); the view of the
    application lives as long as the program."""
    end = start if end is None else end
    left = Qt.MouseButton.LeftButton
    no_modifier = Qt.KeyboardModifier.NoModifier
    row.mousePressEvent(
        QMouseEvent(QEvent.Type.MouseButtonPress, start, start, left, left, no_modifier)
    )
    row.mouseReleaseEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonRelease, end, end, left, Qt.MouseButton.NoButton, no_modifier
        )
    )


@pytest.fixture
def guide_directory(data_directory, monkeypatch):
    """5C with Dlf Nova in its service list and its guide stored, the schedule of
    3 October 2026 kept however long ago that is."""
    monkeypatch.setattr(guide_model, "STORED_SCHEDULE_MAX_AGE_SECONDS", 10**9)
    monkeypatch.setattr(guide_model, "CHANGED_DELAY_MS", 0)
    cache = EnsembleCache(data_directory / receiver_controller.ENSEMBLE_CACHE_FILENAME)
    cache.load()
    services = (*SERVICES_5C, (DLF_NOVA, "Dlf Nova"))
    cache.put("5C", EnsembleInfo(ENSEMBLE_5C, services, 0.0, SOURCE_SCAN, eid=EID_5C))
    cache.save()
    guides = GuideCache(data_directory / receiver_controller.GUIDE_CACHE_DIRECTORY)
    guides.store("5C", SI_5C, 1.0)
    guides.store("5C", DLF_NOVA_PI, 1.0)
    return data_directory


def _receive_5c(controller, worker) -> list:
    controller.start_reception()
    worker = worker()
    worker.connected.emit("Verbunden.")
    controller.dsp_worker.stream_restarted.emit(worker.tuning)
    components = data_components(_ensemble(FIGS_5C))
    controller.dsp_worker.data_services_updated.emit(components)
    return components


def _ndr_carousel() -> MotCarousel:
    """The carousel of NDR EPG on 7A with its directory of 33 files, of which the
    service information is complete (see `tests.test_mot_directory`)."""
    carousel = MotCarousel()
    for raw in (
        NDR_SI_GROUP_0,
        NDR_DIRECTORY_GROUP_0,
        NDR_SI_GROUP_1,
        NDR_DIRECTORY_GROUP_1,
        NDR_SI_GROUP_2,
        NDR_DIRECTORY_GROUP_2,
    ):
        carousel.process(parse_data_group(raw))
    return carousel


def _packets(address: int) -> PacketStatistics:
    return PacketStatistics(
        seconds=10.0,
        fec=True,
        fec_locked=True,
        fec_frames=12,
        fec_corrected_bytes=3,
        fec_uncorrectable_rows=0,
        crc_errors_before_fec=1,
        padding_packets=5,
        addresses={
            address: AddressStatistics(
                packets=250,
                crc_errors=1,
                useful_bytes=30_000,
                data_groups=4,
                data_group_errors=0,
                continuity_errors=1,
            )
        },
    )


def test_texts():
    assert duration_text(300) == "5 min"
    assert duration_text(7200) == "2 h"
    assert duration_text(6420) == "1 h 47 min"
    evening = time.mktime((2026, 10, 5, 22, 0, 0, 0, 0, -1))  # local time
    assert day_title("2026-10-05", evening) == "Today"
    assert day_title("2026-10-06", evening) == "Tomorrow"
    assert day_title("2026-10-07", evening) == "Wed 7 Oct"


def test_the_model_reads_the_stored_guides_on_first_use(guide_directory, make_controller):
    model = GuideModel(make_controller(guide_directory))

    groups = model.groups()

    assert [(g.channel, [e.sid for e in g.entries]) for g in groups] == [("5C", [DLF_NOVA])]
    assert len(model.programmes(DLF_NOVA)) == 5


def test_the_model_takes_what_the_carousel_completes(data_directory, make_controller, fake_workers):
    controller = make_controller(data_directory)
    model = GuideModel(controller)
    model.load()
    epg, _gepos = _receive_5c(controller, lambda: fake_workers[-1])
    controller.dsp_worker.data_decoding_ready.emit((DataComponentStatus(epg.scid, epg),))

    for mot_object in (SI_5C, DLF_NOVA_PI):
        carousel_object = CarouselObject(mot_object, decode_spi_object(mot_object))
        controller.dsp_worker.data_object_ready.emit(epg.scid, carousel_object)

    assert len(model.programmes(DLF_NOVA)) == 5


def test_the_data_services_have_a_card_each_with_a_switch(
    data_directory, make_controller, fake_workers
):
    controller = make_controller(data_directory)
    view = DataServicesView(controller)
    epg, gepos = _receive_5c(controller, lambda: fake_workers[-1])
    controller.dsp_worker.data_decoding_ready.emit((DataComponentStatus(epg.scid, epg),))

    cards = view._cards
    assert list(cards) == [epg.scid, gepos.scid]
    assert cards[epg.scid].state.text() == "decoding"
    assert cards[gepos.scid].state.text() == "starting …"
    assert cards[epg.scid].switch.isChecked()

    cards[gepos.scid].switch.click()
    assert controller.is_data_component_switched_off(gepos)
    assert cards[gepos.scid].switch.text() == "Off"
    assert cards[gepos.scid].state.text().startswith("off · stays off on 5C")


def test_without_reception_there_are_no_data_services(data_directory, make_controller):
    view = DataServicesView(make_controller(data_directory))

    assert view._cards == {}
    assert not view._area.isVisibleTo(view)


def test_the_guide_shows_the_schedule_of_the_service_chosen(guide_directory, make_controller):
    controller = make_controller(guide_directory)
    view = GuideView(controller, GuideModel(controller), ReferenceClock())
    view.resize(1100, 640)
    view.show()
    schedule = view.page(SCHEDULE)

    assert schedule._name.text() == "Dlf Nova"
    assert len(schedule._rows) == 5
    assert [b.text() for b in schedule._day_group.buttons()] == ["Sat 3 Oct"]
    assert schedule._play.isEnabled()

    row = next(row for row in schedule._rows if row.programme.description)
    # A scroll is no tap: the finger moved, or QScroller ended the press far outside.
    _finger(row, QPointF(5, 5), QPointF(5, 60))
    _finger(row, QPointF(5, 5), QPointF(16777215, 16777215))
    assert not row.is_open

    _finger(row, QPointF(5, 5))
    assert row._description.isVisibleTo(row)


def test_the_days_fade_out_where_more_follow(guide_directory, make_controller):
    controller = make_controller(guide_directory)
    view = GuideView(controller, GuideModel(controller), ReferenceClock())
    view.resize(1100, 640)
    view.show()
    schedule = view.page(SCHEDULE)
    fades = schedule._day_fades
    days_area = schedule._days_area
    on_days = QPointF(days_area.mapToGlobal(QPoint(10, 10)))
    assert not fades.start.isVisible()
    assert not fades.end.isVisible()  # one day fits
    assert not days_area.claims_drag(on_days)  # a finger there may turn the page

    now = time.time()
    days = [time.strftime("%Y-%m-%d", time.localtime(now + n * 86400)) for n in range(14)]
    schedule._fill_days(days, now)
    assert days_area.horizontalScrollBar().maximum() > 0
    assert (fades.start.isVisible(), fades.end.isVisible()) == (False, True)
    assert days_area.claims_drag(on_days)  # now it scrolls the days itself

    bar = days_area.horizontalScrollBar()
    bar.setValue(bar.maximum())
    assert (fades.start.isVisible(), fades.end.isVisible()) == (True, False)


def test_hearing_a_service_of_the_guide_plays_it(guide_directory, make_controller, fake_workers):
    controller = make_controller(guide_directory, start=False)
    window = TouchWindow(controller)
    controller.start()
    window.show()
    window.show_page(SERVICES)
    window._services_page.show_view("guide")
    _spin(20)

    window._services_page._guide_view.page(SCHEDULE)._play.click()

    assert controller.selected_sid == DLF_NOVA
    assert controller.state is ReceptionState.CONNECTING
    assert window.current_page() == SERVICES
    window.deleteLater()


def test_only_one_description_is_open_at_a_time(guide_directory, make_controller):
    controller = make_controller(guide_directory)
    view = GuideView(controller, GuideModel(controller), ReferenceClock())
    view.resize(1100, 640)
    view.show()
    rows = view.page(SCHEDULE)._rows
    first, second = [row for row in rows if row.programme.description][:2]

    _finger(first, QPointF(5, 5))
    _finger(second, QPointF(5, 5))

    assert second.is_open
    assert not first.is_open


def test_an_arrow_marks_the_programmes_with_a_description(guide_directory, make_controller):
    controller = make_controller(guide_directory)
    view = GuideView(controller, GuideModel(controller), ReferenceClock())
    view.resize(1100, 640)
    view.show()
    theme = THEMES[DARK]

    def arrow(symbol: str, color: str = "muted"):
        return symbol_pixmap(symbol, theme.color(color), EXPAND_MARK).toImage()

    schedule = view.page(SCHEDULE)
    row = schedule._rows[0]
    assert row._mark.pixmap().toImage() == arrow("expand")
    _finger(row, QPointF(5, 5))
    assert row._mark.pixmap().toImage() == arrow("collapse")
    row.set_now(row.programme.start)
    assert row._mark.pixmap().toImage() == arrow("collapse", "accent")
    schedule._close_others(schedule._rows[1])
    assert row._mark.pixmap().toImage() == arrow("expand", "accent")

    plain = _ProgrammeRow(replace(row.programme, short_description="", long_description=""), theme)
    assert plain._mark.pixmap().isNull()
    _finger(plain, QPointF(5, 5))
    assert not plain.is_open


def test_the_time_left_of_a_long_programme_fits_its_column(guide_directory, make_controller):
    """The column was 170 px wide: "läuft, noch 3 h 43 min" (German, "running, 3 h 43 min
    left") lost its first letters."""
    controller = make_controller(guide_directory)
    view = GuideView(controller, GuideModel(controller), ReferenceClock())
    view.setStyleSheet(stylesheet(THEMES[DARK]))
    view.resize(1100, 640)
    view.show()
    schedule = view.page(SCHEDULE)
    programme = schedule._rows[0].programme
    names = replace(programme.names, long="Eine lange Sendung " * 8)
    row = _ProgrammeRow(replace(programme, names=names, duration=86_399), THEMES[DARK])
    schedule._row_layout.insertWidget(0, row)
    _spin(20)

    assert row.set_now(row.programme.start + 60)
    duration = row._duration
    assert duration.text() == "running, 23 h 59 min left"
    assert duration.fontMetrics().horizontalAdvance(duration.text()) <= duration.width()


def test_a_menu_is_offered_while_the_channel_carries_its_service():
    components = data_components(_ensemble(FIGS_5C))
    epg, gepos = components
    gnss = next(menu for menu in MENUS if menu.key == GNSS)
    traffic = Menu("traffic", "Traffic", USER_APPLICATION_TPEG, always=False)

    assert menu_offered(gnss, components)
    assert not menu_offered(traffic, components)
    assert not menu_offered(gnss, [])
    # Without reception only those always offered.
    assert [menu.key for menu in MENUS if menu_offered(menu, [])] == [DATA_SERVICES, GUIDE]
    assert menu_decoding(gnss, components, {gepos.scid})
    assert not menu_decoding(gnss, components, {epg.scid})


def test_the_guide_menu_is_marked_while_the_guide_is_decoded(
    data_directory, make_controller, fake_workers
):
    controller = make_controller(data_directory, start=False)
    window = TouchWindow(controller)
    controller.start()
    page = window._services_page
    button = page.menu_button(GUIDE)
    epg, _gepos = _receive_5c(controller, lambda: fake_workers[-1])
    assert button.property("decoding") is False

    controller.dsp_worker.data_decoding_ready.emit((DataComponentStatus(epg.scid, epg),))
    assert button.property("decoding") is True
    assert not button.icon().isNull()

    controller.dsp_worker.data_decoding_ready.emit(())
    assert button.property("decoding") is False
    window.deleteLater()


def test_services_with_a_schedule_stand_out(guide_directory, make_controller):
    # The SI of 5C names another service, of which no schedule has come.
    guides = GuideCache(guide_directory / receiver_controller.GUIDE_CACHE_DIRECTORY)
    guides.store(
        "5C",
        _si(
            EID_5C,
            _service("Dlf Nova", DLF_NOVA, EID_5C),
            _service("Schwarzwaldradio", 0x100D, EID_5C),
        ),
        2.0,
    )
    controller = make_controller(guide_directory)
    view = GuideView(controller, GuideModel(controller), ReferenceClock())
    view.show()
    services = view.page(SCHEDULE)._list

    items = {
        services.item(row).text(): services.item(row)
        for row in range(services.count())
        if services.item(row).data(_ENTRY_ROLE) is not None
    }
    assert items["Dlf Nova"].data(_SCHEDULE_ROLE) is True
    assert items["Dlf Nova"].font().bold()
    assert items["Schwarzwaldradio"].data(_SCHEDULE_ROLE) is False
    assert not items["Schwarzwaldradio"].font().bold()


def test_the_programme_running_and_the_one_after_it(guide_directory, make_controller):
    programmes = GuideModel(make_controller(guide_directory)).programmes(DLF_NOVA)
    evening, night = programmes[3], programmes[4]  # 18:00 and 19:00 on 3 October

    assert now_and_next(programmes, evening.start + 60) == (evening, night)
    assert now_and_next(programmes, programmes[0].start - 60) == (None, programmes[0])
    assert now_and_next(programmes, night.end) == (None, None)
    later = replace(night, start=night.start + 600)
    assert now_and_next([evening, later], evening.end + 60) == (None, later)
    assert now_and_next([], 0.0) == (None, None)


def test_when_a_programme_starts():
    now = time.mktime((2026, 10, 5, 18, 40, 0, 0, 0, -1))  # local time

    assert starts_text(now + 20 * 60, now) == "starts in 20 min"
    assert starts_text(now + 6 * 3600, now) == "starts in 6 h"  # after midnight
    assert starts_text(now + 14 * 3600, now) == "Tomorrow"
    assert starts_text(now + 48 * 3600, now) == "Wed 7 Oct"


def _at(window: TouchWindow, timestamp: float) -> None:
    """Sets the broadcast time and lets the radio page catch up with it."""
    window._clock.synchronise(BroadcastTime(timestamp, True))
    window._radio_page._on_tick()


def test_the_radio_page_shows_what_runs_and_leads_to_the_guide(guide_directory, make_controller):
    controller = make_controller(guide_directory, start=False)
    window = TouchWindow(controller)
    controller.start()
    window.show()
    controller.select_service(DLF_NOVA)
    radio = window._radio_page
    card = radio._guide_card
    programmes = window._guide_model.programmes(DLF_NOVA)
    evening, night = programmes[3], programmes[4]

    _at(window, evening.start + 1800)
    assert card.isVisible()
    assert (card.time.text(), card.title.text()) == (
        clock_text(evening.start),
        "Deutschlandfunk Nova",
    )
    assert card.detail.text() == "running, 30 min left"
    assert (card.next_time.text(), card.next_title.text()) == (
        clock_text(night.start),
        "Club der Republik",
    )

    # Before the schedule begins its first programme, after it ends nothing.
    _at(window, programmes[0].start - 1200)
    assert card.detail.text() == "starts in 20 min"
    assert not card.next_title.isVisibleTo(card)
    _at(window, night.end + 60)
    assert not card.isVisible()

    _at(window, evening.start + 1800)
    _finger(card, QPointF(5, 5))

    assert window.current_page() == SERVICES
    assert window._services_page.current_view() == GUIDE
    view = window._services_page._guide_view
    assert view.current_page() == SCHEDULE
    schedule = view.page(SCHEDULE)
    assert schedule._name.text() == "Dlf Nova"
    assert [row.programme for row in schedule._rows if row.property("running")] == [evening]
    window.deleteLater()


def test_a_station_without_a_schedule_has_no_guide_card(guide_directory, make_controller):
    controller = make_controller(guide_directory, start=False)
    window = TouchWindow(controller)
    controller.start()
    window.show()
    card = window._radio_page._guide_card
    _at(window, window._guide_model.programmes(DLF_NOVA)[3].start)

    controller.select_service(DLF_NOVA)
    assert card.isVisible()
    controller.select_service(SERVICES_5C[0][0])
    assert not card.isVisible()
    window.deleteLater()


def test_the_carousel_page_lists_the_files_of_the_guide(
    data_directory, make_controller, fake_workers
):
    controller = make_controller(data_directory, start=False)
    window = TouchWindow(controller)
    controller.start()
    window.show()
    epg, _gepos = _receive_5c(controller, lambda: fake_workers[-1])
    dsp = controller.dsp_worker
    dsp.data_decoding_ready.emit((DataComponentStatus(epg.scid, epg),))
    window.show_page(SERVICES)
    window._services_page.show_view(GUIDE)
    view = window._services_page._guide_view
    view.show_page(CAROUSEL)
    page = view.page(CAROUSEL)
    window._render_plots()
    assert page.status.text() == "Decoding “EPG Deutschland”"
    assert page.table.rowCount() == 0

    dsp.mot_carousel_ready.emit(epg.scid, _ndr_carousel().status())
    dsp.packet_statistics_ready.emit(epg.subchannel_id, _packets(epg.packet.packet_address))
    window._render_plots()

    table = page.table
    rows = {
        table.item(row, 0).text(): [table.item(row, column).text() for column in range(1, 5)]
        for row in range(table.rowCount())
    }
    assert len(rows) == 33
    assert rows["spilogos95661E67.EIB"] == [
        "Service Information (SI)",
        "1.5 kB",
        "3 of 3",
        "complete",
    ]
    logo = rows["e0_d354_112x32_1.png"]
    assert (logo[0], logo[3]) == ("Image PNG", "-")
    check = page.check.text()
    assert check.startswith("Directory 7A06 (no. 1 since the start")
    assert "· 33 files, 1 of " in check
    assert (
        "Data groups 4, 0 with CRC error, 1 interrupted · Packets 250, 1 with CRC error, "
        "24.0 kbit/s payload · Packet FEC in sync: 12 frames, 3 bytes corrected, "
        "0 rows uncorrectable"
    ) in check
    # The line beneath the guide tells of the same carousel.
    schedule = view.page(SCHEDULE)
    assert schedule._status.text() == "Receiving the programme guide from 5C: 1 of 33 files."

    dsp.data_decoding_ready.emit(())
    window._render_plots()
    assert page.status.text() == "The programme guide is not being decoded."
    assert (page.table.rowCount(), page.check.text()) == (0, "")
    assert schedule._status.text() == ""
    window.deleteLater()


def test_a_note_switches_the_guide_on_again(data_directory, make_controller, fake_workers):
    controller = make_controller(data_directory)
    view = GuideView(controller, GuideModel(controller), ReferenceClock())
    epg, _gepos = _receive_5c(controller, lambda: fake_workers[-1])
    assert not view._note.isVisibleTo(view)

    controller.set_data_component_enabled(epg.scid, False)
    assert view._note.isVisibleTo(view)
    view._switch_on.click()
    assert not view._note.isVisibleTo(view)
