"""Announcements in the "Station" tab (`ui.service_tab`): the line on the supported
types and the banner for a running announcement."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from dab.announcement import ALARM_CLUSTER, AnnouncementStatus
from dab.fig import Announcement, AnnouncementSupport, OeAnnouncement, OeAnnouncementSupport
from ui.receiver_controller import AnnouncementVisit
from ui.service_tab import (
    ServiceTab,
    describe_active_announcement,
    describe_announcement_support,
    describe_visit,
)

TRAFFIC_IN_CLUSTER_3 = AnnouncementSupport(0x0002, (3,))
RUNNING = Announcement(3, 0x0002, new=True, subchannel_id=7)


def test_describe_announcement_support():
    assert describe_announcement_support(None) == "Announcements: -"
    assert (
        describe_announcement_support(AnnouncementStatus(None, alarm_enabled=False))
        == "Announcements: none reported"
    )
    assert describe_announcement_support(
        AnnouncementStatus(TRAFFIC_IN_CLUSTER_3, alarm_enabled=True)
    ) == (
        "Announcements: Road Traffic flash (cluster 3); alarm announcements enabled in the ensemble"
    )


def test_describe_active_announcement():
    status = AnnouncementStatus(TRAFFIC_IN_CLUSTER_3, alarm_enabled=False, active=RUNNING)

    assert (
        describe_active_announcement(status, "NDR 2 NDS")
        == "Announcement running: Road Traffic flash (just started) - Subchannel 7, NDR 2 NDS"
    )
    assert describe_active_announcement(AnnouncementStatus(None, False), None) == ""


def test_describe_an_alarm_announcement():
    alarm = Announcement(ALARM_CLUSTER, 0x0001, new=False, subchannel_id=9)
    status = AnnouncementStatus(None, alarm_enabled=True, active=alarm)

    assert describe_active_announcement(status, None) == (
        "Alarm announcement running: Alarm - Subchannel 9"
    )


@pytest.fixture
def service_tab() -> ServiceTab:
    _app = QApplication.instance() or QApplication([])
    tab = ServiceTab()
    yield tab
    tab.deleteLater()


def test_the_banner_shows_only_while_an_announcement_runs(service_tab: ServiceTab):
    service_tab.show_announcements(
        AnnouncementStatus(TRAFFIC_IN_CLUSTER_3, alarm_enabled=False, active=RUNNING), None
    )
    assert not service_tab._announcement_banner.isHidden()

    service_tab.clear()  # a service change leaves it to the signal processing
    assert not service_tab._announcement_banner.isHidden()

    service_tab.show_announcements(AnnouncementStatus(TRAFFIC_IN_CLUSTER_3, False), None)
    assert service_tab._announcement_banner.isHidden()


def test_the_switching_setting_is_reported(service_tab: ServiceTab):
    reported = []
    service_tab.announcement_switching_toggled.connect(reported.append)

    service_tab.set_announcement_switching(True)

    assert service_tab.announcement_switching()
    assert reported == [True]


# --- Announcements of other ensembles (FIG 0/25, FIG 0/26) ------------------------------

OE_TRAFFIC = OeAnnouncement(5, 0x0002, new=False, scids=0, target_sid=0xD382)


def test_describe_oe_announcement_support():
    status = AnnouncementStatus(
        TRAFFIC_IN_CLUSTER_3,
        alarm_enabled=False,
        oe_support=OeAnnouncementSupport(0x0002, ((5, 0x1188), (6, 0x1215))),
    )

    assert describe_announcement_support(status) == (
        "Announcements: Road Traffic flash (cluster 3); "
        "from other ensembles: Road Traffic flash (cluster 5 from 1188, cluster 6 from 1215)"
    )


def test_describe_an_announcement_of_another_ensemble():
    status = AnnouncementStatus(None, False, oe_active=OE_TRAFFIC, oe_eid=0x1188)
    alarm = OeAnnouncement(ALARM_CLUSTER, 0x0001, new=True, scids=0, target_sid=0x1188)

    assert describe_active_announcement(status, "NDR 2") == (
        "Announcement running in ensemble 1188: Road Traffic flash - NDR 2"
    )
    assert describe_active_announcement(status, None).endswith("- Station D382")
    assert describe_active_announcement(
        AnnouncementStatus(None, True, oe_active=alarm, oe_eid=0x1188), None
    ) == ("Alarm announcement running in ensemble 1188: Alarm (just started)")


def test_the_banner_tells_of_reception_gone_to_another_ensemble(service_tab: ServiceTab):
    visit = AnnouncementVisit(OE_TRAFFIC, "11D", 0xD382, "NDR 2", "5C", 0xD220, "Dlf Kultur")
    assert describe_visit(visit) == (
        "Switched to the announcement on channel 11D: Road Traffic flash - NDR 2. "
        "Back to Dlf Kultur on 5C afterwards."
    )

    service_tab.show_visit(visit)
    assert service_tab._announcement_banner.text() == describe_visit(visit)
    service_tab.show_announcements(AnnouncementStatus(None, False), None)  # NDR 2 there
    assert not service_tab._announcement_banner.isHidden()

    service_tab.show_visit(None)
    assert service_tab._announcement_banner.isHidden()
