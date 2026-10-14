"""The cutter without widgets (`recording.cutter`): a title and its neighbours side by
side as one stretch, to the unit by their stream, older files by their time; pauses,
where a handle jumps, the labels and the names they offer, and what a cut keeps."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dsp.mp2_frame import CHANNEL_MODE_JOINT_STEREO, Mp2Format
from dsp.superframe import AudioFormat
from recording import cutter
from recording.cutter import (
    PAUSE_KEEP_S,
    CutRequest,
    build_stretch,
    default_name,
    find_pauses,
    kept_spans,
    label_marks,
    name_suggestions,
    neighbours,
    pause_target,
    split_text,
    unit_seconds,
)
from recording.recorder import Recording

# SBR at 48 kHz: units of 40 ms.
FORMAT = AudioFormat(
    dac_rate_48k=True, sbr=True, stereo_core=True, parametric_stereo=False, mpeg_surround_config=0
)
UNIT_S = 0.04
T0 = 1_791_000_000.0


def _units(first: int, count: int) -> tuple[bytes, ...]:
    """Units numbered by their place in the stream, each different."""
    return tuple(n.to_bytes(3, "big") for n in range(first, first + count))


def _recording(
    name: str,
    start: float,
    units: int,
    *,
    session: str = "",
    first_unit: int = 0,
    labels: tuple = (),
    kept: bool = False,
    cut: bool = False,
    sid: int = 0x1A45,
    artist: str = "",
    title: str = "",
    text: str = "",
) -> Recording:
    return Recording(
        id=name,
        path=Path("/stick") / (name + ".m4a"),
        kept=kept,
        complete=True,
        codec="DAB+",
        station="Absolut relax",
        ensemble="DR Deutschland",
        channel="5C",
        sid=sid,
        title=title,
        artist=artist,
        text=text,
        start=start,
        duration=units * UNIT_S,
        size=1000,
        lost=0,
        bitrate=72,
        audio_format={"dac_rate_48k": True},
        session=session,
        first_unit=first_unit,
        cut=cut,
        labels=labels,
    )


def test_a_unit_lasts_as_its_format_says():
    assert unit_seconds(FORMAT) == pytest.approx(UNIT_S)
    no_sbr = replace(FORMAT, sbr=False)
    assert unit_seconds(no_sbr) == pytest.approx(0.02)
    mp2 = Mp2Format(sample_rate=48_000, bitrate=128, channel_mode=CHANNEL_MODE_JOINT_STEREO)
    assert unit_seconds(mp2) == pytest.approx(0.024)


# --- Side by side ---------------------------------------------------------------------


def test_files_of_one_stream_lie_side_by_side_to_the_unit():
    """The title after began 10 units before its label (start delay): those units are
    in both files and on the stretch once."""
    before = _recording("A", T0, 100, session="s", first_unit=0, title="A")
    opened = _recording("B", T0 + 3.6, 50, session="s", first_unit=90, title="B")
    stretch = build_stretch(opened, [(opened, _units(90, 50)), (before, _units(0, 100))], FORMAT)
    assert stretch.units == _units(0, 140)
    assert [(c.recording.id, c.start) for c in stretch.clips] == [
        ("A", 0.0),
        ("B", pytest.approx(90 * UNIT_S)),
    ]
    assert stretch.opened_clip().duration == pytest.approx(2.0)
    assert stretch.origin == pytest.approx(T0)
    assert stretch.session == "s" and stretch.first_unit == 0
    assert stretch.id == "cut:B" and stretch.duration == pytest.approx(5.6)


def test_a_stretch_of_the_stream_in_neither_file_closes_up():
    """A stop delay that cut more than the start delay kept leaves units in neither
    file: the files meet there."""
    before = _recording("A", T0, 100, session="s", first_unit=0)
    opened = _recording("B", T0 + 4.2, 50, session="s", first_unit=105)
    stretch = build_stretch(opened, [(before, _units(0, 100)), (opened, _units(105, 50))], FORMAT)
    assert len(stretch.units) == 150
    assert stretch.opened_clip().start == pytest.approx(100 * UNIT_S)


def test_older_files_back_to_back_meet_where_one_ends():
    """Recorded before the files knew their stream: their times say they meet (within
    the jitter of the clock), so they go back to back."""
    before = _recording("A", T0, 100)
    opened = _recording("B", T0 + 4.0 - 0.16, 50)
    stretch = build_stretch(opened, [(opened, _units(100, 50)), (before, _units(0, 100))], FORMAT)
    assert stretch.units == _units(0, 150)
    assert stretch.opened_clip().start == pytest.approx(4.0)
    assert stretch.origin == pytest.approx(opened.start - 4.0)


def test_older_files_that_overlap_are_placed_by_the_units_they_share():
    before = _recording("A", T0, 100)
    # Began 20 units before A ended; its clock says 0.3 s later than that.
    opened = _recording("B", T0 + 80 * UNIT_S + 0.3, 50)
    stretch = build_stretch(opened, [(before, _units(0, 100)), (opened, _units(80, 50))], FORMAT)
    assert stretch.units == _units(0, 130)
    assert stretch.opened_clip().start == pytest.approx(80 * UNIT_S)

    # The title after likewise, by a whole 40 units.
    after = _recording("C", T0 + 60 * UNIT_S + 0.2, 50)
    stretch = build_stretch(before, [(before, _units(0, 100)), (after, _units(60, 50))], FORMAT)
    assert stretch.units == _units(0, 110)
    assert stretch.clip_of("C").start == pytest.approx(60 * UNIT_S)


def test_digital_silence_alike_in_both_files_is_placed_by_the_clock():
    """Silent units encode alike: several overlaps would fit, the clock chooses."""
    silence = (b"\x00",) * 30
    before = _recording("A", T0, 100)
    opened = _recording("B", T0 + 4.0 - 0.1, 50)  # meets A, by the clock
    stretch = build_stretch(
        opened,
        [(before, _units(0, 70) + silence), (opened, silence[:20] + _units(500, 30))],
        FORMAT,
    )
    # 8 to 20 units shared would fit as well; the clock says they meet: back to back.
    assert stretch.opened_clip().start == pytest.approx(100 * UNIT_S)
    assert len(stretch.units) == 150


def test_the_labels_of_the_files_go_onto_the_stretch_once():
    before = _recording(
        "A",
        T0,
        100,
        session="s",
        labels=((1.0, "A - EINS", "A", "EINS"), (3.7, "B - ZWEI", "", "")),
    )
    # B noted the same label at its beginning: it lies in both files.
    opened = _recording(
        "B", T0 + 3.6, 50, session="s", first_unit=90, labels=((0.1, "B - ZWEI", "", ""),)
    )
    stretch = build_stretch(opened, [(before, _units(0, 100)), (opened, _units(90, 50))], FORMAT)
    assert stretch.labels == ((1.0, "A - EINS", "A", "EINS"), (3.7, "B - ZWEI", "", ""))


# --- Neighbours -----------------------------------------------------------------------


def test_the_neighbours_of_a_title_are_those_of_its_stream_that_meet_it():
    opened = _recording("B", T0 + 4, 100, session="s", first_unit=100)
    before = _recording("A", T0, 100, session="s", first_unit=0)
    earlier = _recording("Z", T0 - 4, 100, session="s", first_unit=-100)
    after = _recording("C", T0 + 8, 100, session="s", first_unit=195)
    cut = _recording("B cut", T0 + 4, 80, session="s", first_unit=105, kept=True, cut=True)
    other_stream = _recording("X", T0 + 8, 100, session="t", first_unit=200)
    other_station = _recording("Y", T0, 100, session="s", sid=0x15DD)
    recordings = (earlier, before, opened, after, cut, other_stream, other_station)
    assert neighbours(opened, recordings, UNIT_S) == (before, after)

    # A gap in the stream (a title deleted between): no neighbour there.
    assert neighbours(opened, (earlier, opened), UNIT_S) == (None, None)

    # A cut opened: the recording it was cut from holds it, with what lies around it.
    assert neighbours(cut, recordings, UNIT_S) == (opened, None)


def test_older_files_are_neighbours_by_their_time():
    opened = _recording("B", T0 + 4, 100)
    before = _recording("A", T0, 100)  # ends as B begins
    long_before = _recording("Z", T0 - 100, 100)  # ended 96 s before
    after = _recording("C", T0 + 8 + 1.5, 100)  # 1.5 s after B ended
    too_late = _recording("D", T0 + 8 + 5, 100)
    assert neighbours(opened, (long_before, before, opened, after), UNIT_S) == (before, after)
    assert neighbours(opened, (long_before, opened, too_late), UNIT_S) == (None, None)


# --- Pauses ---------------------------------------------------------------------------


def test_pauses_are_quiet_long_enough():
    levels = np.full(100, -20.0)
    levels[40:50] = -91.0  # 0.4 s
    levels[70:73] = -91.0  # 0.12 s: too short
    levels[80:90] = np.nan  # not read yet: not quiet
    assert find_pauses(levels, UNIT_S) == [(pytest.approx(1.6), pytest.approx(2.0))]
    assert find_pauses(np.full(10, -20.0), UNIT_S) == []


def test_a_handle_jumps_to_where_the_sound_begins_or_ends():
    pauses = [(10.0, 11.0), (50.0, 50.5)]
    # The beginning goes to the end of a pause, keeping a little of the silence.
    assert pause_target(pauses, 30.0, beginning=True, forward=False) == 11.0 - PAUSE_KEEP_S
    assert pause_target(pauses, 30.0, beginning=True, forward=True) == 50.5 - PAUSE_KEEP_S
    assert pause_target(pauses, 10.9, beginning=True, forward=True) == 50.5 - PAUSE_KEEP_S
    # The end to its beginning.
    assert pause_target(pauses, 30.0, beginning=False, forward=False) == 10.0 + PAUSE_KEEP_S
    assert pause_target(pauses, 60.0, beginning=False, forward=True) is None


# --- Labels and names -----------------------------------------------------------------


def _stretch(*labels, clips=None) -> cutter.Stretch:
    opened = _recording("B", T0, 100, session="s", labels=labels, artist="ADELE", title="RAIN")
    return build_stretch(opened, [(opened, _units(0, 100))], FORMAT)


def test_a_label_is_marked_where_it_appears_first():
    stretch = _stretch(
        (0.5, "ADELE - RAIN", "ADELE", "RAIN"),
        (1.0, "Absolut relax", "", ""),
        (1.5, "ADELE - RAIN", "ADELE", "RAIN"),
        (2.0, "BOYZONE - NO MATTER WHAT", "", ""),
    )
    assert label_marks(stretch) == [
        (0.5, "ADELE - RAIN"),
        (1.0, "Absolut relax"),
        (2.0, "BOYZONE - NO MATTER WHAT"),
    ]


def test_names_come_from_the_files_and_the_labels_each_once():
    stretch = _stretch(
        (0.5, "Jetzt: ADELE - RAIN", "Adele", "Rain"),
        (1.0, "Absolut relax", "", ""),
        (2.0, "BOYZONE - NO MATTER WHAT", "", ""),
    )
    assert name_suggestions(stretch) == [
        ("ADELE", "RAIN"),
        ("", "Absolut relax"),
        ("BOYZONE", "NO MATTER WHAT"),
    ]
    assert split_text("A-HA – TAKE ON ME") == ("A-HA", "TAKE ON ME")
    assert split_text("Nachrichten") == ("", "Nachrichten")
    assert split_text("A - B - C") == ("", "A - B - C")


def test_the_name_of_a_cut_is_that_of_the_file_holding_most_of_it():
    before = _recording("A", T0, 100, session="s", artist="ADELE", title="RAIN")
    opened = _recording("B", T0 + 4, 100, session="s", first_unit=100, text="BOYZONE - NO MATTER")
    stretch = build_stretch(opened, [(before, _units(0, 100)), (opened, _units(100, 100))], FORMAT)
    assert default_name(stretch, 3.5, 8.0) == ("BOYZONE", "NO MATTER")
    assert default_name(stretch, 0.0, 4.5) == ("ADELE", "RAIN")


# --- What a cut keeps -----------------------------------------------------------------


def test_a_cut_keeps_its_units_and_the_labels_within():
    stretch = _stretch((0.5, "EINS", "", ""), (2.0, "ZWEI", "", ""), (3.5, "DREI", "", ""))
    request = CutRequest(stretch, 25, 75, "A", "T")
    assert request.units == _units(25, 50)
    assert request.start == pytest.approx(T0 + 1.0)
    assert request.duration == pytest.approx(2.0)
    assert request.labels == ((1.0, "ZWEI", "", ""),)
    assert request.text == "ZWEI"


def test_titles_kept_are_found_on_the_stretch():
    opened = _recording("B", T0, 100, session="s", first_unit=1000)
    stretch = build_stretch(opened, [(opened, _units(1000, 100))], FORMAT)
    cut = _recording("cut", T0 + 1, 50, session="s", first_unit=1025, kept=True, cut=True)
    old = _recording("old", T0 + 2.0, 25, kept=True)  # by its time
    elsewhere = _recording("far", T0 + 100, 25, kept=True)
    other = _recording("other", T0, 100, kept=True, sid=0x15DD)
    assert kept_spans(stretch, (cut, old, elsewhere, other)) == [
        (pytest.approx(1.0), pytest.approx(3.0)),
        (pytest.approx(2.0), pytest.approx(3.0)),
    ]
