"""Announcements (`dab.announcement`): which running announcement applies to a
service, the timeout for a lost end, and the names of the types."""

from __future__ import annotations

from dab.announcement import (
    ALARM_CLUSTER,
    ANNOUNCEMENT_TIMEOUT_SECONDS,
    AnnouncementMonitor,
    announcement_types,
    audio_in_subchannel,
)
from dab.fig import (
    ASCTY_DAB_PLUS,
    TMID_STREAM_AUDIO,
    Announcement,
    AnnouncementSupport,
    AudioComponent,
    Ensemble,
    OeAnnouncement,
    OeAnnouncementSupport,
    ServiceComponent,
    parse_fib,
)
from tests.dab_signal import fib

ROAD_TRAFFIC = 0x0002
NEWS = 0x0010
ALARM = 0x0001

NDR_2 = 0xD382
NDR_INFO = 0xD384


def _traffic(cluster_id: int = 3, subchannel_id: int = 7) -> Announcement:
    return Announcement(cluster_id, ROAD_TRAFFIC, new=False, subchannel_id=subchannel_id)


def _alarm(subchannel_id: int = 9) -> Announcement:
    return Announcement(ALARM_CLUSTER, ALARM, new=True, subchannel_id=subchannel_id)


def _ensemble(*announcements: Announcement, alarm_enabled: bool = False) -> Ensemble:
    """NDR Info takes road traffic announcements in cluster 3."""
    ensemble = Ensemble(alarm_announcements=alarm_enabled)
    ensemble.announcement_support[NDR_INFO] = AnnouncementSupport(ROAD_TRAFFIC, (3,))
    for announcement in announcements:
        ensemble.announcements[announcement.cluster_id] = announcement
    return ensemble


def _status(ensemble: Ensemble, sid: int = NDR_INFO, now: float = 0.0):
    monitor = AnnouncementMonitor()
    monitor.update(ensemble, now)
    return monitor.status(ensemble, sid, now)


def test_names_of_the_announcement_types():
    assert [name.short for name in announcement_types(ROAD_TRAFFIC | NEWS)] == [
        "Road Traffic flash",
        "News flash",
    ]
    assert announcement_types(1 << 12)[0].short == "Type 12"  # reserved, still listed
    assert announcement_types(0) == []


def test_an_announcement_for_the_cluster_of_the_service_applies():
    status = _status(_ensemble(_traffic()))

    assert status.support == AnnouncementSupport(ROAD_TRAFFIC, (3,))
    assert status.active == _traffic()


def test_an_announcement_for_another_cluster_does_not_apply():
    assert _status(_ensemble(_traffic(cluster_id=4))).active is None


def test_an_announcement_of_a_type_the_service_does_not_accept_does_not_apply():
    news = Announcement(3, NEWS, new=False, subchannel_id=7)

    assert _status(_ensemble(news)).active is None


def test_a_service_without_announcement_support_is_not_interrupted():
    status = _status(_ensemble(_traffic()), sid=NDR_2)

    assert status.support is None
    assert status.active is None


def test_alarm_announcements_interrupt_every_service_while_the_al_flag_is_set():
    ensemble = _ensemble(_traffic(), _alarm(), alarm_enabled=True)

    assert _status(ensemble, sid=NDR_2).active == _alarm()
    assert _status(ensemble).active == _alarm()  # before the traffic announcement


def test_alarm_announcements_are_ignored_without_the_al_flag():
    """Clause 8.1.6.0: receivers shall ignore them unless FIG 0/0 enables them."""
    ensemble = _ensemble(_traffic(), _alarm(), alarm_enabled=False)

    assert _status(ensemble, sid=NDR_2).active is None
    assert _status(ensemble).active == _traffic()


def test_an_announcement_ends_when_fig_0_19_stays_away():
    ensemble = _ensemble(_traffic())
    monitor = AnnouncementMonitor()
    monitor.update(ensemble, 10.0)

    monitor.update(ensemble, 10.0 + ANNOUNCEMENT_TIMEOUT_SECONDS)  # nothing new came
    assert monitor.status(ensemble, NDR_INFO, 10.0 + ANNOUNCEMENT_TIMEOUT_SECONDS).active

    later = 10.1 + ANNOUNCEMENT_TIMEOUT_SECONDS
    monitor.update(ensemble, later)
    assert monitor.status(ensemble, NDR_INFO, later).active is None


def test_every_repetition_of_fig_0_19_confirms_the_announcement():
    ensemble = _ensemble()
    monitor = AnnouncementMonitor()
    fig0_19 = fib(bytes.fromhex("051303000207"))  # cluster 3, road traffic, subchannel 7
    for second in range(20):
        parse_fib(fig0_19, ensemble)
        monitor.update(ensemble, float(second))

    assert monitor.status(ensemble, NDR_INFO, 19.0).active == _traffic()


def test_the_service_whose_audio_carries_the_announcement():
    ensemble = _ensemble(_traffic(subchannel_id=7))
    ensemble.service_components[NDR_2] = (
        ServiceComponent(TMID_STREAM_AUDIO, True, False, ASCTY_DAB_PLUS, subchannel_id=7),
    )

    status = _status(ensemble)

    assert status.source_sid == NDR_2
    assert status.source_component == AudioComponent(7, ASCTY_DAB_PLUS)


def test_audio_in_subchannel_prefers_a_primary_component():
    ensemble = Ensemble()
    secondary = ServiceComponent(TMID_STREAM_AUDIO, False, False, ASCTY_DAB_PLUS, subchannel_id=7)
    primary = ServiceComponent(TMID_STREAM_AUDIO, True, False, ASCTY_DAB_PLUS, subchannel_id=7)
    ensemble.service_components[NDR_INFO] = (
        ServiceComponent(TMID_STREAM_AUDIO, True, False, ASCTY_DAB_PLUS, subchannel_id=3),
        secondary,
    )
    ensemble.service_components[NDR_2] = (primary,)

    assert audio_in_subchannel(ensemble, 7) == (NDR_2, AudioComponent(7, ASCTY_DAB_PLUS))
    assert audio_in_subchannel(ensemble, 8) is None


# --- Announcements of other ensembles (FIG 0/25, FIG 0/26) ------------------------------

ENSEMBLE_7A = 0x1188
NDR_1 = 0xD381  # carries the announcements in ensemble 7A


def _oe_traffic(cluster_id: int = 5) -> OeAnnouncement:
    return OeAnnouncement(cluster_id, ROAD_TRAFFIC, new=True, scids=0, target_sid=NDR_1)


def _oe_alarm() -> OeAnnouncement:
    return OeAnnouncement(ALARM_CLUSTER, ALARM, new=True, scids=0, target_sid=ENSEMBLE_7A)


def _oe_ensemble(*announcements, alarm_enabled: bool = False) -> Ensemble:
    """NDR Info takes road traffic announcements in cluster 3 of this ensemble and from
    OE cluster 5, served by ensemble 7A."""
    ensemble = _ensemble(
        *(a for a in announcements if isinstance(a, Announcement)), alarm_enabled=alarm_enabled
    )
    ensemble.oe_announcement_support[NDR_INFO] = OeAnnouncementSupport(
        ROAD_TRAFFIC, ((5, ENSEMBLE_7A),)
    )
    for announcement in announcements:
        if isinstance(announcement, OeAnnouncement):
            key = (announcement.cluster_id, announcement.target_sid)
            ensemble.oe_announcements[key] = announcement
    return ensemble


def test_an_announcement_of_another_ensemble_for_an_oe_cluster_of_the_service_applies():
    status = _status(_oe_ensemble(_oe_traffic()))

    assert status.oe_support == OeAnnouncementSupport(ROAD_TRAFFIC, ((5, ENSEMBLE_7A),))
    assert status.oe_active == _oe_traffic()
    assert status.oe_eid == ENSEMBLE_7A
    assert status.active is None


def test_an_announcement_of_another_ensemble_needs_cluster_type_and_support():
    news = OeAnnouncement(5, NEWS, new=True, scids=0, target_sid=NDR_1)

    assert _status(_oe_ensemble(_oe_traffic(cluster_id=6))).oe_active is None
    assert _status(_oe_ensemble(news)).oe_active is None
    assert _status(_oe_ensemble(_oe_traffic()), sid=NDR_2).oe_active is None  # no FIG 0/25


def test_an_announcement_of_the_tuned_ensemble_comes_first():
    status = _status(_oe_ensemble(_traffic(), _oe_traffic()))

    assert status.active == _traffic()
    assert status.oe_active is None


def test_an_alarm_of_another_ensemble_interrupts_every_service_while_the_al_flag_is_set():
    ensemble = _oe_ensemble(_traffic(), _oe_alarm(), alarm_enabled=True)

    for sid in (NDR_INFO, NDR_2):  # before the traffic announcement, and without FIG 0/25
        status = _status(ensemble, sid=sid)
        assert (status.active, status.oe_active, status.oe_eid) == (None, _oe_alarm(), ENSEMBLE_7A)

    assert _status(_oe_ensemble(_oe_alarm()), sid=NDR_2).oe_active is None  # Al flag off


def test_an_alarm_of_the_tuned_ensemble_comes_before_one_of_another():
    status = _status(_oe_ensemble(_alarm(), _oe_alarm(), alarm_enabled=True))

    assert status.active == _alarm()
    assert status.oe_active is None


def test_an_announcement_of_another_ensemble_ends_when_fig_0_26_stays_away():
    ensemble = _oe_ensemble(_oe_traffic())
    monitor = AnnouncementMonitor()
    monitor.update(ensemble, 10.0)

    later = 10.1 + ANNOUNCEMENT_TIMEOUT_SECONDS
    monitor.update(ensemble, later)
    assert monitor.status(ensemble, NDR_INFO, later).oe_active is None


def test_an_announcement_in_the_own_subchannel_is_carried_whatever_its_cluster():
    """How a receiver gone to another ensemble for NDR 2 tells that the announcement
    there runs: NDR 2 is in no cluster, but cluster 4 is served from its subchannel 7."""
    ensemble = _ensemble(_traffic(cluster_id=4, subchannel_id=7))
    ensemble.audio_components[NDR_2] = AudioComponent(7, ASCTY_DAB_PLUS)

    status = _status(ensemble, sid=NDR_2)

    assert status.active is None
    assert status.carried == _traffic(cluster_id=4, subchannel_id=7)
    assert _status(ensemble).carried is None  # NDR Info has no audio component here
