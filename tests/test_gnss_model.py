"""The `GnssModel` with real messages of GEPOS: the sky from broadcast ephemerides, the
statistics of the decoded stream, the corrections at the listener's position, and how it
follows the controller."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from dab.data_services import data_components
from dab.gnss import MAX_KEPLER_AGE_SECONDS, Observer, satellite_positions
from dab.rtcm import RtcmStatistics, decode_ephemeris
from dab.ssrz import SsrzDecoder
from dsp.pipeline import DataComponentStatus
from tests.rtcm_vectors import (
    BEIDOU_C08_FRAME,
    GALILEO_E02_FRAME,
    GLONASS_R11_FRAME,
    GPS_G12_FRAME,
    REFERENCE_EPOCH_UTC,
)
from tests.ssrz_vectors import EPHEMERIS_FRAMES
from tests.test_corrections import FRAMES
from tests.test_data_services import FIGS_5C, _ensemble
from ui.gnss_model import ORBIT_POINTS, TRACK_MINUTES, TRACK_STEP_MINUTES, GnssModel

HANNOVER = Observer(52.5, 9.5, 100.0)
MUENCHEN = Observer(48.14, 11.58, 560.0)


def _recorded(signal) -> list[tuple]:
    calls: list[tuple] = []
    signal.connect(lambda *args: calls.append(args))
    return calls


@pytest.fixture
def model():
    # Kept until the test ends: without an application the model would go with it.
    app = QApplication.instance() or QApplication([])
    yield GnssModel(HANNOVER)
    del app


def test_the_sky_from_the_ephemerides(model) -> None:
    changed = _recorded(model.sky_changed)
    for frame in (GPS_G12_FRAME, GALILEO_E02_FRAME, BEIDOU_C08_FRAME, GLONASS_R11_FRAME):
        model.add_ephemeris(decode_ephemeris(frame[3:-3]))

    assert len(changed) == 4
    satellites = model.satellites(REFERENCE_EPOCH_UTC)
    assert sorted(s.ephemeris.name for s in satellites) == ["C08", "E02", "G12", "R11"]
    elevations = [s.elevation for s in satellites]
    assert elevations == sorted(elevations, reverse=True)
    tracks = model.tracks(REFERENCE_EPOCH_UTC)
    assert tracks
    for system, azimuth, elevation in tracks.values():
        assert len(azimuth) == len(elevation) == TRACK_MINUTES // TRACK_STEP_MINUTES + 1


def test_the_satellites_in_space_with_their_orbits(model) -> None:
    for frame in (GPS_G12_FRAME, GALILEO_E02_FRAME, BEIDOU_C08_FRAME, GLONASS_R11_FRAME):
        model.add_ephemeris(decode_ephemeris(frame[3:-3]))

    orbits = model.orbits(REFERENCE_EPOCH_UTC)
    sky = {s.ephemeris.name: s for s in model.satellites(REFERENCE_EPOCH_UTC)}
    assert sorted(o.ephemeris.name for o in orbits) == sorted(sky)
    for orbit in orbits:
        assert orbit.ring.shape == (ORBIT_POINTS + 1, 3)
        position = satellite_positions(orbit.ephemeris, REFERENCE_EPOCH_UTC)[0]
        assert np.allclose(orbit.position, position, atol=1e-3)
        assert orbit.elevation == pytest.approx(sky[orbit.ephemeris.name].elevation)
    # Long after the reference times of the ephemerides there is no orbit.
    assert model.orbits(REFERENCE_EPOCH_UTC + MAX_KEPLER_AGE_SECONDS + 3 * 3600) == []


def test_an_ephemeris_not_renewed_is_dropped(model) -> None:
    model.add_ephemeris(decode_ephemeris(GPS_G12_FRAME[3:-3]))
    changed = _recorded(model.sky_changed)

    model.drop_stale(time.time() + 60)
    assert model.ephemeris_count == 1
    assert not changed

    model.drop_stale(time.time() + MAX_KEPLER_AGE_SECONDS + 60)
    assert model.ephemeris_count == 0
    assert len(changed) == 1


def test_gepos_and_its_decoding(model) -> None:
    epg, gepos = data_components(_ensemble(FIGS_5C))
    changed = _recorded(model.decoding_changed)

    model.update_components([epg, gepos])
    model.update_components([epg, gepos])
    assert model.gepos == gepos
    assert not model.gepos_decoded
    assert len(changed) == 1  # the same components again change nothing

    model.set_data_decoding((DataComponentStatus(epg.scid, epg),))
    assert model.decoding is None  # the programme guide is no TDC component
    model.set_data_decoding(
        (DataComponentStatus(epg.scid, epg), DataComponentStatus(gepos.scid, gepos))
    )
    assert model.decoding == gepos
    assert model.gepos_decoded
    assert len(changed) == 2


def test_the_statistics_of_the_decoded_stream_only(model) -> None:
    epg, gepos = data_components(_ensemble(FIGS_5C))
    model.update_components([epg, gepos])
    model.set_data_decoding((DataComponentStatus(gepos.scid, gepos),))
    statistics = RtcmStatistics(10.0, 40, 0, 120, 1, 0, 4000)
    changed = _recorded(model.stream_changed)

    model.update_statistics(epg.scid, statistics)
    assert model.statistics is None
    model.update_statistics(gepos.scid, statistics)
    assert model.statistics is statistics

    # Another ensemble, or none decoded any more: the statistics are void.
    model.reset_stream()
    assert model.statistics is None
    model.update_statistics(gepos.scid, statistics)
    model.set_data_decoding(())
    assert model.statistics is None
    assert len(changed) == 4


def test_corrections_at_the_position_and_at_a_new_one(model) -> None:
    for frame in EPHEMERIS_FRAMES.values():
        model.add_ephemeris(decode_ephemeris(frame[3:-3]))
    corrections = _recorded(model.corrections_changed)
    atmosphere = _recorded(model.atmosphere_changed)
    decoder = SsrzDecoder()
    for frame in FRAMES:
        model.add_ssrz(decoder.push(frame[3:-3]))

    # One epoch, the high rate message after the low rate one (see `tests.ssrz_vectors`).
    assert len(corrections) == 1
    assert [c.name for c in model.corrections] == ["E03", "G06", "C11", "R02"]
    assert "G11" in model.missing_ephemerides()
    assert len(model.history("E03").samples) == 1
    assert atmosphere
    assert model.atmosphere.received is not None

    model.set_observer(MUENCHEN)
    assert model.observer == MUENCHEN
    assert len(corrections) == 2
    # The course starts afresh with the epoch evaluated at the new position.
    assert len(model.history("E03").samples) == 1


def test_it_follows_the_controller(data_directory, make_controller) -> None:
    controller = make_controller(data_directory, start=False)
    model = GnssModel.for_controller(controller)
    assert model.observer == controller.observer

    controller.set_observer(MUENCHEN)
    assert model.observer == MUENCHEN

    controller.dsp_worker.ephemeris_ready.emit(decode_ephemeris(GPS_G12_FRAME[3:-3]))
    assert model.ephemeris_count == 1
