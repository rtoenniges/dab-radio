"""Emergency warnings in the receiver (`ui.receiver_controller`): which are played, at
which volume, in the tuned ensemble and in another one, and how they end. The states of
the Emergency Warning System are emitted by the test, as the signal processing would
(`dab.ews.EwsState`); no device is needed (`tests.fake_receiver`)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from dab.channels import BAND_III_CHANNELS
from dab.dls import DynamicLabel
from dab.ensemble_cache import SOURCE_SCAN, EnsembleCache, EnsembleInfo
from dab.ews import EwsAlert, EwsState, Stage
from dab.gnss import Observer
from dab.location_code import LocationCode, location_code, parse_presentation
from tests.fake_receiver import ENSEMBLE_5C, SERVICES_5C, SERVICES_11D
from tests.test_receiver_controller import _Recorder, _settings, _wait_for
from ui import receiver_controller
from ui.receiver_controller import DEFAULT_ALARM_VOLUME_PERCENT, ReceptionState

EID_5C = 0x10BC
EID_11D = 0x1188
DLF_KULTUR = SERVICES_5C[1][0]
ASA_DE = 0x10C4
ASA_SUBCHANNEL = 9
NDR_2 = SERVICES_11D[0][0]
REICHSTAG = Observer(52.5186538, 13.3761015, 60.0)  # Z1:91BB81
BERLIN_MITTE = frozenset({LocationCode(1, "91B")})
PARIS = frozenset({LocationCode(1, "894")})


def _alert(
    stage: Stage = Stage.LEVEL_1_START,
    *,
    incident: int = 1,
    locations: frozenset[LocationCode] = frozenset(),
    other: bool = False,
) -> EwsAlert:
    return EwsAlert(
        other_ensemble=other,
        eid=EID_11D if other else EID_5C,
        stage=stage,
        incident=incident,
        locations=locations,
        whole_area=not locations,
        subchannel_id=None if other else ASA_SUBCHANNEL,
        sid=None if other else ASA_DE,
        receivable=None if other else True,
    )


def _state(*alerts: EwsAlert, running=(), eid: int = EID_5C, sid: int = ASA_DE) -> EwsState:
    return EwsState(
        participating=True,
        eid=eid,
        alerts=alerts,
        running=frozenset(running),
        services=tuple((subchannel, sid) for subchannel in running),
    )


QUIET = _state()


@pytest.fixture
def receiving(data_directory, make_controller, fake_workers, monkeypatch):
    """A controller playing Dlf Kultur on 5C (ensemble 10BC, which carries ASA DE) at
    40 %, the listener at the Reichstag; 11D is stored as ensemble 1188. The search in
    another ensemble lasts 50 ms."""
    monkeypatch.setattr(receiver_controller, "ALERT_SEARCH_MS", 50)
    cache = EnsembleCache(data_directory / receiver_controller.ENSEMBLE_CACHE_FILENAME)
    cache.load()
    cache.put("5C", EnsembleInfo(ENSEMBLE_5C, SERVICES_5C, 0.0, SOURCE_SCAN, eid=EID_5C))
    cache.put("11D", EnsembleInfo("NDR NDS", SERVICES_11D, 0.0, SOURCE_SCAN, eid=EID_11D))
    cache.save()
    controller = make_controller(data_directory)
    controller.set_observer(REICHSTAG)
    controller.set_ews_enabled(True)
    controller.set_volume_percent(40)
    controller.play_service("5C", DLF_KULTUR)
    worker = fake_workers[-1]
    worker.connected.emit("Connected.")
    _arrive(controller, worker)
    return controller, worker


def _arrive(controller, worker, state: EwsState = QUIET) -> None:
    """The signal processing reaches the frequency tuned to and reports from there."""
    dsp = controller.dsp_worker
    dsp.stream_restarted.emit(worker.tuning)
    services = [*SERVICES_5C, (ASA_DE, "ASA DE")] if controller.channel == "5C" else SERVICES_11D
    dsp.services_updated.emit("ensemble", list(services), {})
    dsp.ews_ready.emit(state)


def test_a_warning_of_the_tuned_ensemble_plays_at_the_alarm_volume(receiving):
    controller, _worker = receiving
    played = _Recorder(controller._alert_played)
    gains = _Recorder(controller._volume_changed)
    alerts = _Recorder(controller.alert_changed)
    controller.set_alarm_volume_percent(90)

    controller.dsp_worker.ews_ready.emit(_state(_alert(), running=[ASA_SUBCHANNEL]))
    playing = controller.alert
    controller.dsp_worker.alert_label_ready.emit(DynamicLabel("Achtung! Probealarm"))
    assert (playing.subchannel_id, playing.label, playing.origin) == (
        ASA_SUBCHANNEL,
        "ASA DE",
        None,
    )
    assert controller.alert.text == "Achtung! Probealarm"
    assert played.calls == [(ASA_SUBCHANNEL,)]
    assert controller.volume_percent == 90
    assert gains.last == (pytest.approx(10 ** (-6 / 20)),)
    assert controller.selected_sid == DLF_KULTUR  # stays selected

    controller.dsp_worker.ews_ready.emit(_state(running=[ASA_SUBCHANNEL]))  # sustain
    assert controller.alert is not None
    controller.dsp_worker.ews_ready.emit(QUIET)  # end

    assert controller.alert is None
    assert played.last == (None,)
    assert controller.volume_percent == 40
    assert alerts.last == (None,)


def test_a_warning_plays_while_muted_and_leaves_the_mute(receiving):
    controller, _worker = receiving
    controller.set_audio_enabled(False)
    toggled = _Recorder(controller._audio_enabled_toggled)

    controller.dsp_worker.ews_ready.emit(_state(_alert(), running=[ASA_SUBCHANNEL]))
    assert controller.audio_enabled
    assert toggled.last == (True,)
    controller.dsp_worker.ews_ready.emit(QUIET)

    assert not controller.audio_enabled
    assert toggled.last == (False,)


def test_muted_the_radio_takes_level_1_only(receiving):
    """Putting out no audio, it is a receiver in monitor mode (ETSI TS 104 089, table 1)."""
    controller, _worker = receiving
    controller.set_audio_enabled(False)
    level_2 = _state(_alert(Stage.LEVEL_2_START), running=[ASA_SUBCHANNEL])

    controller.dsp_worker.ews_ready.emit(level_2)
    assert controller.alert is None
    controller.set_audio_enabled(True)
    controller.dsp_worker.ews_ready.emit(level_2)

    assert controller.alert is not None


def test_only_warnings_for_the_listeners_place_play(receiving):
    controller, _worker = receiving
    dsp = controller.dsp_worker

    dsp.ews_ready.emit(_state(_alert(locations=PARIS), running=[ASA_SUBCHANNEL]))
    assert controller.alert is None

    controller.set_ews_location(parse_presentation("1253-3513-3668"))  # the ASA test code
    assert controller.ews_location_entered
    assert controller.alert is not None
    controller.dsp_worker.ews_ready.emit(QUIET)

    controller.set_ews_location(None)  # the position again
    assert controller.ews_location == location_code(REICHSTAG.latitude, REICHSTAG.longitude)
    dsp.ews_ready.emit(_state(_alert(incident=2, locations=BERLIN_MITTE), running=[9]))
    assert controller.alert is not None


def test_without_a_position_only_warnings_for_the_whole_area_play(
    data_directory, make_controller, fake_workers
):
    controller = make_controller(data_directory)
    controller.set_ews_enabled(True)
    controller.play_service("5C", DLF_KULTUR)
    worker = fake_workers[-1]
    worker.connected.emit("Connected.")
    _arrive(controller, worker)
    assert controller.ews_location is None

    controller.dsp_worker.ews_ready.emit(_state(_alert(locations=BERLIN_MITTE), running=[9]))
    assert controller.alert is None
    controller.dsp_worker.ews_ready.emit(_state(_alert(), running=[9]))
    assert controller.alert is not None


def test_switched_off_no_warning_plays(receiving):
    controller, _worker = receiving
    controller.set_ews_enabled(False)

    controller.dsp_worker.ews_ready.emit(_state(_alert(), running=[ASA_SUBCHANNEL]))

    assert controller.alert is None


def test_switching_off_ends_the_warning(receiving):
    controller, _worker = receiving
    controller.dsp_worker.ews_ready.emit(_state(_alert(), running=[ASA_SUBCHANNEL]))

    controller.set_ews_enabled(False)

    assert controller.alert is None
    assert controller.volume_percent == 40


def test_the_slider_sets_the_level_of_the_warning_for_as_long_as_it_plays(receiving):
    controller, _worker = receiving
    controller.dsp_worker.ews_ready.emit(_state(_alert(), running=[ASA_SUBCHANNEL]))

    controller.set_volume_percent(55)
    assert controller.volume_percent == 55
    assert controller.alarm_volume_percent == DEFAULT_ALARM_VOLUME_PERCENT
    controller.dsp_worker.ews_ready.emit(QUIET)

    assert controller.volume_percent == 40
    assert int(controller.settings.value("controls/volume")) == 40


def test_breaking_off_and_dismissing_an_incident(receiving):
    controller, _worker = receiving
    dsp = controller.dsp_worker
    dsp.ews_ready.emit(_state(_alert(), running=[ASA_SUBCHANNEL]))

    controller.end_alert(dismiss_incident=True)
    assert controller.alert is None
    dsp.ews_ready.emit(_state(_alert(), running=[ASA_SUBCHANNEL]))  # still signalled
    assert controller.alert is None
    dsp.ews_ready.emit(_state(_alert(Stage.LEVEL_1_UPDATE), running=[ASA_SUBCHANNEL]))
    assert controller.alert is None
    dsp.ews_ready.emit(_state(_alert(Stage.LEVEL_1_CRITICAL), running=[ASA_SUBCHANNEL]))
    assert controller.alert is not None


def test_muting_breaks_the_warning_off(receiving):
    controller, _worker = receiving
    controller.dsp_worker.ews_ready.emit(_state(_alert(), running=[ASA_SUBCHANNEL]))

    controller.set_audio_enabled(False)

    assert controller.alert is None
    assert not controller.audio_enabled


def test_a_new_warning_cuts_the_one_playing_short(receiving):
    controller, _worker = receiving
    played = _Recorder(controller._alert_played)
    dsp = controller.dsp_worker
    dsp.ews_ready.emit(_state(_alert(), running=[ASA_SUBCHANNEL]))

    dsp.ews_ready.emit(_state(_alert(incident=4), running=[ASA_SUBCHANNEL]))

    assert controller.alert.alert.incident == 4
    assert controller.volume_percent == DEFAULT_ALARM_VOLUME_PERCENT
    assert played.calls == [(ASA_SUBCHANNEL,), (ASA_SUBCHANNEL,)]


def test_the_users_choice_ends_the_warning_where_reception_is(receiving):
    controller, _worker = receiving
    controller.dsp_worker.ews_ready.emit(_state(_alert(), running=[ASA_SUBCHANNEL]))

    controller.select_service(SERVICES_5C[0][0])
    controller.dsp_worker.ews_ready.emit(_state(_alert(), running=[ASA_SUBCHANNEL]))

    assert controller.alert is None
    assert controller.selected_sid == SERVICES_5C[0][0]


def test_a_warning_of_another_ensemble_is_heard_there(receiving):
    controller, worker = receiving
    held = _Recorder(controller._audio_held)
    played = _Recorder(controller._alert_played)

    controller.dsp_worker.ews_ready.emit(_state(_alert(other=True)))
    assert controller.channel == "11D"
    assert worker.retuned_to[-1] == BAND_III_CHANNELS["11D"] * 1e6
    assert controller.alert.origin == ("5C", DLF_KULTUR)
    assert controller.alert.subchannel_id is None
    assert held.last == (True,)  # 11D's stations are not to be heard meanwhile

    _arrive(controller, worker, _state(running=[5], eid=EID_11D, sid=NDR_2))
    assert controller.alert.subchannel_id == 5
    assert controller.alert.label == "NDR 2"
    assert played.last == (5,)
    assert held.last == (False,)

    controller.dsp_worker.ews_ready.emit(_state(eid=EID_11D))  # over
    assert (controller.channel, controller.selected_sid) == ("5C", DLF_KULTUR)
    assert controller.alert is None

    # 5C may still signal it for a moment: it is not gone to again.
    _arrive(controller, worker, _state(_alert(other=True)))
    assert controller.channel == "5C"


def test_a_warning_not_found_there_leads_back(receiving):
    controller, worker = receiving
    controller.dsp_worker.ews_ready.emit(_state(_alert(other=True)))
    _arrive(controller, worker, _state(eid=EID_11D))

    _wait_for(lambda: controller.alert is None)

    assert (controller.channel, controller.selected_sid) == ("5C", DLF_KULTUR)


def test_an_ensemble_not_in_the_tuning_memory_is_not_gone_to(receiving):
    controller, _worker = receiving
    unknown = EwsAlert(True, 0x1205, Stage.LEVEL_1_START, 1, whole_area=True)

    controller.dsp_worker.ews_ready.emit(_state(unknown))

    assert controller.alert is None
    assert controller.channel == "5C"


def test_stopping_there_leaves_the_service_left_behind_selected(receiving):
    controller, worker = receiving
    controller.dsp_worker.ews_ready.emit(_state(_alert(other=True)))

    worker.stopped.emit()

    assert controller.state is ReceptionState.IDLE
    assert (controller.channel, controller.selected_sid) == ("5C", DLF_KULTUR)
    assert controller.alert is None
    assert controller.ews_state == EwsState()


def test_closing_there_keeps_the_service_left_behind_for_the_next_start(data_directory, receiving):
    controller, _worker = receiving
    controller.dsp_worker.ews_ready.emit(_state(_alert(other=True)))

    controller.shutdown()

    settings = _settings(data_directory)
    assert settings.value("controls/channel") == "5C"
    assert int(settings.value("service/sid")) == DLF_KULTUR


def test_the_settings_are_kept(data_directory, make_controller):
    controller = make_controller(data_directory, start=False)
    assert not controller.ews_enabled  # until the listener switches them on
    assert controller.alarm_volume_percent == DEFAULT_ALARM_VOLUME_PERCENT
    controller.set_ews_enabled(True)
    controller.set_ews_location(parse_presentation("1255-4467-1251"))
    controller.set_alarm_volume_percent(0)  # never silent
    controller.settings.sync()

    again = make_controller(data_directory, start=False)

    assert again.ews_enabled
    assert again.ews_location == LocationCode(1, "91BB81")
    assert again.alarm_volume_percent == 5
    with pytest.raises(ValueError):
        again.set_ews_location(LocationCode(1, "91B"))


def test_the_state_of_the_ensemble_is_reported(receiving):
    controller, _worker = receiving
    changes = _Recorder(controller.ews_changed)

    controller.dsp_worker.ews_ready.emit(EwsState(participating=False, eid=EID_5C))
    controller.dsp_worker.ews_ready.emit(EwsState(participating=False, eid=EID_5C))

    assert not controller.ews_state.participating
    assert len(changes.calls) == 1
