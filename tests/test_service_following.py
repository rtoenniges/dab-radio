"""Where else a service can be heard: ensembles located by the channel list and FIG 0/21
(`dab.service_following`)."""

from __future__ import annotations

from dab.ensemble_cache import SOURCE_SCAN, EnsembleInfo
from dab.service_following import EnsembleLocation, locate_ensemble, other_ensembles

EID_7A, EID_8C = 0x1188, 0x1215


def _stored(label: str | None, eid: int | None) -> EnsembleInfo:
    return EnsembleInfo(label, (), 0.0, SOURCE_SCAN, eid=eid)


STORED = {"7A": _stored("NDR NDS HAN", EID_7A), "8C": _stored("NI HANNOVER K8C", EID_8C)}


def test_an_ensemble_received_on_a_channel_is_found_there():
    assert locate_ensemble(EID_7A, STORED, {}) == EnsembleLocation(
        EID_7A, "7A", received=True, label="NDR NDS HAN"
    )


def test_what_was_received_counts_before_what_is_announced():
    announced = {EID_7A: (229_072,)}  # 12D

    assert locate_ensemble(EID_7A, STORED, announced).channel == "7A"


def test_an_ensemble_not_received_is_found_where_fig_0_21_announces_it():
    announced = {0x1226: (223_936,)}  # 12A

    assert locate_ensemble(0x1226, STORED, announced) == EnsembleLocation(
        0x1226, "12A", received=False
    )


def test_an_announced_channel_carrying_another_ensemble_here_is_passed_over():
    """5D announces its ensemble 11F7 on 8C as well, where NI HANNOVER is received here;
    a frequency off the Band III raster (L band) has no channel."""
    announced = {0x11F7: (199_360, 1_452_960)}
    assert locate_ensemble(0x11F7, STORED, announced) == EnsembleLocation(
        0x11F7, None, received=False
    )

    announced = {0x11F7: (199_360, 180_064)}  # 8C, then 5D
    assert locate_ensemble(0x11F7, STORED, announced).channel == "5D"


def test_a_stored_list_without_eid_rules_out_no_channel():
    stored = {"8C": _stored("NI HANNOVER K8C", None)}  # stored before the EId was kept

    assert locate_ensemble(0x11F7, stored, {0x11F7: (199_360,)}).channel == "8C"


def test_other_ensembles_leave_out_the_tuned_one_and_come_received_first():
    """NDR 2 NDS (D882) as received on 8C in seven ensembles; 12A for 1226 is made up.
    Received ones first, then announced ones, then unknown ones, each in the order sent."""
    eids = (0x1150, 0x1157, 0x1177, EID_7A, 0x11A7, 0x11FD, 0x1226, EID_7A, EID_8C)

    located = other_ensembles(eids, EID_8C, STORED, {0x1226: (223_936,)})

    assert [(location.eid, location.channel) for location in located] == [
        (EID_7A, "7A"),
        (0x1226, "12A"),
        (0x1150, None),
        (0x1157, None),
        (0x1177, None),
        (0x11A7, None),
        (0x11FD, None),
    ]


def test_a_service_carried_only_here_has_no_other_ensembles():
    """5D names only itself for RTL RADIO."""
    assert other_ensembles((0x11F7,), 0x11F7, STORED, {}) == ()
