"""The Emergency Warning System (ETSI TS 104 089): FIG 0/15 as received and as built
after annexes C and E, followed over time, and the choice of the alert to play."""

from __future__ import annotations

import pytest

from dab.ews import (
    ALERT_TIMEOUT_SECONDS,
    EWS_TIMEOUT_SECONDS,
    FORGET_SECONDS,
    TRIGGER_TIMEOUT_SECONDS,
    AlertSelector,
    EwsAlert,
    EwsMonitor,
    EwsState,
    Phase,
    Stage,
    location_matches,
    stage_matches,
)
from dab.fig import Ensemble, EwsSignal, parse_ews_signal, parse_fib
from dab.location_code import LocationCode, location_code
from tests.dab_signal import fib, fig0_2_audio_services

EID = 0x10BC  # 5C, which takes part
OTHER_EID = 0x11F7
ALERT_SID = 0x10C4  # "ASA DE" on 5C
ALERT_SUBCHANNEL = 5

# Annex C: the Cardiff alert area, four location codes (22 bytes, 17 rectangles at five
# digits), here in the trigger phase in subchannel 5, last of its group, Level 1 Start,
# incident 3.
CARDIFF_CODES = "0abb6240cc00" + "0abb6250f730" + "0a4b6283" + "0abb62900007"
CARDIFF_TRIGGER = "0f" + "45" + "83" + CARDIFF_CODES
CARDIFF = location_code(51.4816, -3.1791)  # the city centre, Z10:B625DA
BRISTOL = location_code(51.4545, -2.5879)  # outside, Z10:B63925


def _fig(field_hex: str) -> bytes:
    """A FIG 0 of this type 0 field (the byte with the flags and the extension first)."""
    field = bytes.fromhex(field_hex)
    return bytes([len(field)]) + field


def _parse(*fields_hex: str) -> Ensemble:
    ensemble = Ensemble()
    for field in fields_hex:
        parse_fib(fib(_fig(field)), ensemble)
    return ensemble


# --- FIG 0/15 ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fig", "discard"), [("018f", False), ("01af", True)], ids=["seconds 0-29", "seconds 30-59"]
)
def test_the_heartbeat_as_received_on_5c(fig, discard):
    ensemble = Ensemble()
    parse_fib(fib(bytes.fromhex(fig)), ensemble)

    assert list(ensemble.ews_signals) == [
        EwsSignal(current=True, other_ensemble=False, discard=discard)
    ]
    assert ensemble.ews_signals[0].heartbeat


def test_the_trigger_with_the_cardiff_alert_area():
    (signal,) = _parse(CARDIFF_TRIGGER).ews_signals

    assert (signal.phase, signal.subchannel_id) == (Phase.TRIGGER, ALERT_SUBCHANNEL)
    assert (signal.last, signal.stage, signal.incident) == (True, Stage.LEVEL_1_START, 3)
    assert signal.following == 0
    assert sorted(code.digits for code in signal.locations) == [
        *("B624A", "B624B", "B624E", "B624F"),
        *("B6254", "B6255", "B6258", "B6259", "B625A", "B625C", "B625D", "B625E", "B625F"),
        "B6283",
        *("B6290", "B6291", "B6292"),
    ]
    assert all(code.zone == 10 for code in signal.locations)


def test_an_alert_in_another_ensemble():
    """OE flag set: the EId instead of phase and subchannel, always the trigger phase;
    no location codes, so the whole coverage area. Second of two in its alert set."""
    signal = parse_ews_signal(bytes.fromhex("4f" + "11f7" + "1b"))

    assert signal.other_ensemble
    assert (signal.phase, signal.eid, signal.subchannel_id) == (Phase.TRIGGER, OTHER_EID, None)
    assert (signal.last, signal.stage, signal.incident) == (False, Stage.LEVEL_1_UPDATE, 11)
    assert signal.locations == ()


def test_the_pre_trigger_names_the_start_second():
    signal = parse_ews_signal(bytes.fromhex("0f" + "05" + "3f" + "80" + "4a4b6283"))

    assert (signal.phase, signal.subchannel_id, signal.start_second) == (Phase.PRE_TRIGGER, 5, 63)
    assert signal.locations == (LocationCode(10, "B6283"),)
    assert signal.following == 1  # one more FIG 0/15 of the set to come


@pytest.mark.parametrize(("byte", "phase"), [("85", Phase.SUSTAIN), ("c5", Phase.END)])
def test_sustain_and_end_carry_the_id_field_only(byte, phase):
    signal = parse_ews_signal(bytes.fromhex("8f" + byte))

    assert (signal.phase, signal.subchannel_id, signal.stage) == (phase, 5, None)


def test_a_location_code_cut_short_ends_the_reading():
    signal = parse_ews_signal(bytes.fromhex("0f4583" + "0a4b6283" + "0abb6240cc"))

    assert signal.locations == (LocationCode(10, "B6283"),)


def test_a_trigger_without_its_status_is_dropped():
    assert parse_ews_signal(bytes.fromhex("0f45")) is None


# --- Following the signals over time ------------------------------------------------------


def _ensemble(*, subchannel: bool = True) -> Ensemble:
    """5C as far as an alert needs it: EId (FIG 0/0), the alert service in FIG 0/2 and,
    unless left out, its subchannel in FIG 0/1."""
    figs = [bytes.fromhex("050010bc2342"), fig0_2_audio_services([(ALERT_SID, ALERT_SUBCHANNEL)])]
    if subchannel:
        figs.append(bytes.fromhex("05011400" + "8824"))  # SubChId 5, EEP-3A, 36 CU
    ensemble = Ensemble()
    for fig in figs:
        parse_fib(fib(fig), ensemble)
    return ensemble


def _signals(*fields_hex: str) -> list[EwsSignal]:
    return [parse_ews_signal(bytes.fromhex(field)) for field in fields_hex]


def test_an_ensemble_takes_part_while_fig_0_15_comes():
    monitor, ensemble = EwsMonitor(), _ensemble()

    assert not monitor.state(ensemble, 0.0).participating
    monitor.update(_signals("8f"), 1.0)
    assert monitor.state(ensemble, 1.0).participating
    assert monitor.state(ensemble, 1.0 + EWS_TIMEOUT_SECONDS).participating
    assert not monitor.state(ensemble, 1.1 + EWS_TIMEOUT_SECONDS).participating


def test_an_alert_carried_in_the_ensemble_through_its_phases():
    monitor, ensemble = EwsMonitor(), _ensemble()

    monitor.update(_signals(CARDIFF_TRIGGER), 0.0)
    triggered = monitor.state(ensemble, 0.0)
    monitor.update(_signals("8f85"), 6.0)
    sustained = monitor.state(ensemble, 6.0)
    monitor.update(_signals("8fc5"), 30.0)
    ended = monitor.state(ensemble, 30.0)

    (alert,) = triggered.alerts
    assert (alert.eid, alert.stage, alert.incident) == (EID, Stage.LEVEL_1_START, 3)
    assert (alert.subchannel_id, alert.sid, alert.receivable) == (ALERT_SUBCHANNEL, ALERT_SID, True)
    assert len(alert.locations) == 17 and not alert.whole_area
    assert triggered.running == {ALERT_SUBCHANNEL}
    assert sustained.alerts == () and sustained.running == {ALERT_SUBCHANNEL}
    assert ended.running == frozenset()


def test_an_alert_message_without_sustain_runs_out():
    monitor, ensemble = EwsMonitor(), _ensemble()
    monitor.update(_signals(CARDIFF_TRIGGER), 0.0)

    assert monitor.state(ensemble, TRIGGER_TIMEOUT_SECONDS).alerts
    assert not monitor.state(ensemble, TRIGGER_TIMEOUT_SECONDS + 0.1).alerts
    assert monitor.state(ensemble, ALERT_TIMEOUT_SECONDS).running
    assert not monitor.state(ensemble, ALERT_TIMEOUT_SECONDS + 0.1).running


def test_the_area_collects_over_the_alert_set():
    monitor, ensemble = EwsMonitor(), _ensemble()

    monitor.update(_signals("0f4503" + "4a4b6283", "0f4583" + "0a4b6290"), 0.0)

    (alert,) = monitor.state(ensemble, 0.0).alerts
    assert alert.locations == {LocationCode(10, "B6283"), LocationCode(10, "B6290")}


def test_alerts_of_the_tuned_ensemble_come_first():
    monitor, ensemble = EwsMonitor(), _ensemble()

    monitor.update(_signals("4f11f700"), 0.0)
    monitor.update(_signals("0f4500"), 1.0)

    first, second = monitor.state(ensemble, 1.0).alerts
    assert (first.other_ensemble, first.eid, first.whole_area) == (False, EID, True)
    assert (second.other_ensemble, second.eid, second.subchannel_id) == (True, OTHER_EID, None)
    assert second.receivable is None


def test_pre_triggers_are_not_for_receivers():
    monitor, ensemble = EwsMonitor(), _ensemble()

    monitor.update(_signals("0f053f80"), 0.0)

    state = monitor.state(ensemble, 0.0)
    assert state.participating and state.alerts == () and state.running == frozenset()


def test_an_alert_in_a_subchannel_fig_0_1_does_not_know_cannot_be_heard():
    monitor, ensemble = EwsMonitor(), _ensemble(subchannel=False)

    monitor.update(_signals("0f4500"), 0.0)

    assert monitor.state(ensemble, 0.0).alerts[0].receivable is False


# --- Matching -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stage", "repeats", "incident", "expected"),
    [
        (Stage.LEVEL_1_START, True, True, True),
        (Stage.LEVEL_2_START, True, True, True),
        (Stage.LEVEL_1_CRITICAL, True, True, True),
        (Stage.LEVEL_1_UPDATE, True, False, True),
        (Stage.LEVEL_2_UPDATE, False, True, False),
        (Stage.LEVEL_1_REPEAT, False, False, True),
        (Stage.LEVEL_2_REPEAT, True, False, False),
        (Stage.LEVEL_1_REPEAT, False, True, False),
        (Stage.TEST, False, False, False),
    ],
)
def test_stage_matching_in_audio_mode(stage, repeats, incident, expected):
    """Table 1 of clause 7.5.3, the column for receivers playing audio."""
    assert stage_matches(stage, repeats_dismissed=repeats, incident_dismissed=incident) is expected


def _alert(
    stage: Stage = Stage.LEVEL_1_START,
    *,
    incident: int = 3,
    locations: frozenset[LocationCode] = frozenset(),
    other: bool = False,
) -> EwsAlert:
    return EwsAlert(
        other_ensemble=other,
        eid=OTHER_EID if other else EID,
        stage=stage,
        incident=incident,
        locations=locations,
        whole_area=not locations,
        subchannel_id=None if other else ALERT_SUBCHANNEL,
        sid=None if other else ALERT_SID,
        receivable=None if other else True,
    )


CARDIFF_AREA = frozenset(parse_ews_signal(bytes.fromhex(CARDIFF_TRIGGER)).locations)


def test_location_matching():
    assert location_matches(_alert(locations=CARDIFF_AREA), CARDIFF)
    assert not location_matches(_alert(locations=CARDIFF_AREA), BRISTOL)
    assert location_matches(_alert(), BRISTOL)  # the whole coverage area
    # Without a location of its own a receiver plays alerts for the whole area only.
    assert location_matches(_alert(), None)
    assert not location_matches(_alert(locations=CARDIFF_AREA), None)


def _choose(selector: AlertSelector, *alerts: EwsAlert, receiver=CARDIFF, receivable=True, now=0.0):
    state = EwsState(participating=True, eid=EID, alerts=alerts)
    selector.observe(state, now)
    return selector.choose(state, receiver, lambda alert: receivable)


def test_the_selector_plays_what_concerns_the_receiver():
    selector = AlertSelector()
    outside = _alert(locations=frozenset({LocationCode(10, "B63")}))
    inside = _alert(incident=4, locations=CARDIFF_AREA)

    assert _choose(selector, outside) is None
    assert _choose(selector, outside, inside) == inside
    assert _choose(selector, _alert(Stage.TEST)) is None


def test_the_alert_carried_in_the_tuned_ensemble_comes_first():
    selector = AlertSelector()
    own, other = _alert(), _alert(other=True)

    assert _choose(selector, own, other) == own
    assert _choose(selector, other, receivable=False) is None
    assert _choose(selector, other) == other


def test_an_alert_in_a_subchannel_not_configured_is_skipped():
    unconfigured = EwsAlert(
        other_ensemble=False, eid=EID, stage=Stage.LEVEL_1_START, incident=1,
        whole_area=True, subchannel_id=9, receivable=False,
    )  # fmt: skip

    assert _choose(AlertSelector(), unconfigured) is None


def test_an_alert_played_is_not_played_again_while_signalled():
    selector = AlertSelector()
    alert = _alert()
    _choose(selector, alert, now=0.0)

    selector.finished(alert, 1.0)

    assert _choose(selector, alert, now=20.0) is None
    assert _choose(selector, alert, now=20.0 + FORGET_SECONDS) is None  # seen at 20 s
    selector.observe(EwsState(), 21.0 + 2 * FORGET_SECONDS)
    assert _choose(selector, alert, now=22.0 + 2 * FORGET_SECONDS) == alert


def test_dismissing_repeats_and_incidents():
    """Clause 7.4.2; the next start of the incident clears what was dismissed."""
    selector = AlertSelector()
    start = _alert(Stage.LEVEL_1_START)
    _choose(selector, start, now=0.0)

    selector.dismiss(start, 1.0, repeats=True, incident=False)
    assert _choose(selector, _alert(Stage.LEVEL_1_REPEAT), now=10.0) is None
    assert _choose(selector, _alert(Stage.LEVEL_1_UPDATE), now=10.0) is not None

    selector.dismiss(start, 11.0, repeats=False, incident=True)
    assert _choose(selector, _alert(Stage.LEVEL_2_UPDATE), now=20.0) is None
    assert _choose(selector, _alert(Stage.LEVEL_1_CRITICAL), now=20.0) is not None
    assert _choose(selector, _alert(Stage.LEVEL_1_UPDATE, incident=5), now=20.0) is not None

    selector.observe(EwsState(), 20.0 + 2 * FORGET_SECONDS)  # the start forgotten
    assert _choose(selector, start, now=300.0) == start
    assert _choose(selector, _alert(Stage.LEVEL_1_REPEAT), now=300.0) is not None


def test_a_start_still_signalled_keeps_what_was_dismissed_during_it():
    selector = AlertSelector()
    start = _alert(Stage.LEVEL_1_START)
    _choose(selector, start, now=0.0)

    selector.dismiss(start, 2.0, repeats=False, incident=True)
    _choose(selector, start, now=4.0)  # its trigger phase goes on

    assert _choose(selector, _alert(Stage.LEVEL_1_UPDATE), now=200.0) is None
