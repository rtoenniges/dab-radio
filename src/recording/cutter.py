"""Cutting a title out of the recordings, without widgets: a station whose dynamic label
comes late, now and then, or not at all leaves its titles in the wrong files, the
beginning of one at the end of the title before.

The cutter therefore shows a recording together with its neighbours of the same station,
the title before and the one after as far as they are still there, as one stretch of
audio (`Stretch`): the access units (DAB+) or frames (DAB) side by side, the overlap of
two files (`recording.recorder`, start and stop delay) once only. What is kept between
two handles becomes a title of its own, cut at the units, losslessly, without decoding
(`CutRequest`, written by `recording.recorder.TitleRecorder.save_cut`).

Where the files lie on the stretch: each file knows the unbroken stream it belongs to
(`Recording.session`) and the number of its first unit in it (`Recording.first_unit`),
which places them to the unit. Files of earlier versions of the radio, without these,
are placed by the time they began: back to back where the one ends as the other
begins, and where they overlap by the first units of the later one found in the
earlier.

Also here: the pauses in the level (`find_pauses`, where a handle can jump to), the
dynamic labels seen during the stretch (`label_marks`) and the names they offer
(`name_suggestions`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from dsp.mp2_frame import SAMPLES_PER_FRAME, Mp2Format
from dsp.superframe import AudioFormat
from recording.mp4 import SAMPLES_PER_UNIT

if TYPE_CHECKING:  # the recorder imports this module
    from recording.recorder import Recording

# A stretch of audio this quiet for this long is a pause between two titles: radio
# automation leaves digital silence there (−91 dBFS between two titles of Absolut relax,
# 2026-10-13), a quiet passage of music stays well above it.
PAUSE_BELOW_DB = -45.0
PAUSE_MIN_S = 0.2
# How much of a pause a handle that jumped to it keeps: the beginning keeps the last of
# the silence before the sound, the end the first of it after.
PAUSE_KEEP_S = 0.1
# Two files without a stream are neighbours when the one began at most this much
# after the other ended, or overlap by at most this much (the start and stop delays
# reach 60 s).
NEIGHBOUR_GAP_S = 2.0
NEIGHBOUR_OVERLAP_S = 61.0
# How many units of the later file have to be found in the earlier to place an overlap.
MATCH_UNITS = 8
# The shortest title the cutter keeps.
MIN_CUT_S = 1.0
# Two labels of the same text this close on the stretch are the same (two files that
# overlap both noted it).
SAME_LABEL_S = 2.0
MAX_SUGGESTIONS = 8
# What a label is: seconds on the file or stretch, its text, and DL Plus artist and title.
Label = tuple[float, str, str, str]


@dataclass(frozen=True)
class Clip:
    """A file on the stretch: where it begins and how long it is, in seconds."""

    recording: Recording
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class Stretch:
    """A recording opened for cutting, with its neighbours side by side."""

    opened: Recording
    clips: tuple[Clip, ...]  # by their beginning
    units: tuple[bytes, ...]
    audio_format: AudioFormat | Mp2Format
    origin: float  # Unix seconds at the beginning of the stretch
    session: str  # the stream of the files, "" for older ones
    first_unit: int  # the number of its first unit in that stream
    labels: tuple[Label, ...]  # by their seconds on the stretch

    @property
    def id(self) -> str:
        """The id of the stretch, for playback and the levels."""
        return "cut:" + self.opened.id

    @property
    def unit_seconds(self) -> float:
        return unit_seconds(self.audio_format)

    @property
    def duration(self) -> float:
        return len(self.units) * self.unit_seconds

    def clip_of(self, recording_id: str) -> Clip | None:
        return next((c for c in self.clips if c.recording.id == recording_id), None)

    def opened_clip(self) -> Clip:
        clip = self.clip_of(self.opened.id)
        assert clip is not None
        return clip

    def unit_at(self, seconds: float) -> int:
        """The number of the unit boundary nearest to `seconds`."""
        return max(0, min(round(seconds / self.unit_seconds), len(self.units)))

    def snap(self, seconds: float) -> float:
        """`seconds` on the nearest unit boundary: where a cut can be."""
        return self.unit_at(seconds) * self.unit_seconds


@dataclass(frozen=True)
class CutRequest:
    """What to keep of a stretch: units `first` up to `last` (not included), under a
    name; `replaces`, the id of a kept title the cut takes the place of."""

    stretch: Stretch
    first: int
    last: int
    artist: str
    title: str
    replaces: str | None = None

    @property
    def units(self) -> tuple[bytes, ...]:
        return self.stretch.units[self.first : self.last]

    @property
    def start(self) -> float:
        """Unix seconds at its beginning."""
        return self.stretch.origin + self.first * self.stretch.unit_seconds

    @property
    def duration(self) -> float:
        return (self.last - self.first) * self.stretch.unit_seconds

    @property
    def labels(self) -> tuple[Label, ...]:
        """The labels of the stretch within the cut, in seconds of the cut."""
        offset = self.first * self.stretch.unit_seconds
        return tuple(
            (round(at - offset, 3), text, artist, title)
            for at, text, artist, title in self.stretch.labels
            if offset <= at < offset + self.duration
        )

    @property
    def text(self) -> str:
        """The dynamic label of the cut: the last within it, else the one before."""
        offset = self.first * self.stretch.unit_seconds
        before = [text for at, text, _a, _t in self.stretch.labels if at < offset + self.duration]
        return before[-1] if before else self.stretch.opened.text


def unit_seconds(audio_format: AudioFormat | Mp2Format) -> float:
    """How long a unit plays: 960 core samples of AAC or 1152 samples of MP2."""
    if isinstance(audio_format, Mp2Format):
        return SAMPLES_PER_FRAME / audio_format.sample_rate
    return SAMPLES_PER_UNIT / audio_format.core_sample_rate


def _unit_count(recording: Recording, seconds_per_unit: float) -> int:
    return round(recording.duration / seconds_per_unit)


def same_kind(a: Recording, b: Recording) -> bool:
    """Of the same service in the same format: a stretch can hold both."""
    return (a.channel, a.sid, a.codec, a.audio_format) == (
        b.channel,
        b.sid,
        b.codec,
        b.audio_format,
    )


def neighbours(
    opened: Recording, recordings: tuple[Recording, ...], seconds_per_unit: float
) -> tuple[Recording | None, Recording | None]:
    """The title before `opened` and the one after it among `recordings` (ring and
    kept), each where its audio meets that of `opened` without a gap; titles cut before
    are no neighbours, their audio comes from the recordings."""
    candidates = [
        r
        for r in recordings
        if r.id != opened.id and not r.cut and r.complete and same_kind(r, opened)
    ]
    if opened.session:
        stream = [r for r in candidates if r.session == opened.session]
        first = opened.first_unit
        last = first + _unit_count(opened, seconds_per_unit)
        before = [
            r
            for r in stream
            if r.first_unit < first and r.first_unit + _unit_count(r, seconds_per_unit) >= first
        ]
        after = [r for r in stream if first < r.first_unit <= last]
        previous = max(before, key=lambda r: r.first_unit, default=None)
        following = min(after, key=lambda r: r.first_unit, default=None)
        return previous, following
    end = opened.start + opened.duration
    before = [
        r
        for r in candidates
        if not r.session
        and r.start < opened.start
        and opened.start - NEIGHBOUR_GAP_S
        <= r.start + r.duration
        <= opened.start + NEIGHBOUR_OVERLAP_S
    ]
    after = [
        r
        for r in candidates
        if not r.session
        and r.start > opened.start
        and end - NEIGHBOUR_OVERLAP_S <= r.start <= end + NEIGHBOUR_GAP_S
    ]
    previous = max(before, key=lambda r: r.start, default=None)
    following = min(after, key=lambda r: r.start, default=None)
    return previous, following


def _overlap(
    earlier: tuple[bytes, ...], later: tuple[bytes, ...], expected: float, seconds_per_unit: float
) -> int | None:
    """How many units the end of `earlier` shares with the beginning of `later`: the
    recorder copies the last seconds of a title to the beginning of the next (start
    delay), and runs a title on beside the next (stop delay), so a shared stretch is
    always the end of the one and the beginning of the other. At least `MATCH_UNITS`,
    or none at all where the clock says they meet; of several (digital silence encodes
    alike), the one nearest to what the clock says (`expected` units); None where
    nothing fits."""
    longest = min(len(earlier), len(later), round(NEIGHBOUR_OVERLAP_S / seconds_per_unit))
    found = [
        count
        for count in range(MATCH_UNITS, longest + 1)
        if earlier[len(earlier) - count :] == later[:count]
    ]
    if abs(expected) * seconds_per_unit <= NEIGHBOUR_GAP_S:
        found.append(0)  # back to back
    return min(found, key=lambda count: abs(count - expected), default=None)


def _place_without_session(
    opened: Recording,
    opened_units: tuple[bytes, ...],
    other: Recording,
    other_units: tuple[bytes, ...],
    seconds_per_unit: float,
) -> int:
    """Where the first unit of `other` lies, counted from the first of `opened`, for
    files recorded before they knew their stream: by the units they share, else back to
    back where the clock says they meet, else by the clock."""
    if other.start < opened.start:
        overlap = other.start + other.duration - opened.start
        shared = _overlap(other_units, opened_units, overlap / seconds_per_unit, seconds_per_unit)
        if shared is not None:
            return shared - len(other_units)
    else:
        overlap = opened.start + opened.duration - other.start
        shared = _overlap(opened_units, other_units, overlap / seconds_per_unit, seconds_per_unit)
        if shared is not None:
            return len(opened_units) - shared
    return round((other.start - opened.start) / seconds_per_unit)


def build_stretch(
    opened: Recording,
    pieces: list[tuple[Recording, tuple[bytes, ...]]],
    audio_format: AudioFormat | Mp2Format,
) -> Stretch:
    """The stretch of `opened` and its neighbours: `pieces`, each file with its units
    (`opened` among them). Units in two files are taken once; should a stretch of the
    stream lie in neither (a stop delay that cut more than the start delay kept), the
    files meet there."""
    seconds_per_unit = unit_seconds(audio_format)
    opened_units = next(units for recording, units in pieces if recording.id == opened.id)
    placed: list[tuple[Recording, tuple[bytes, ...], int]] = []
    for recording, units in pieces:
        if opened.session and recording.session == opened.session:
            number = recording.first_unit
        elif recording.id == opened.id:
            number = 0
        else:
            number = _place_without_session(
                opened, opened_units, recording, units, seconds_per_unit
            )
        placed.append((recording, units, number))
    by_number: dict[int, bytes] = {}
    for _recording, units, number in placed:
        for offset, unit in enumerate(units):
            by_number[number + offset] = unit
    numbers = sorted(by_number)
    index_of = {number: index for index, number in enumerate(numbers)}
    clips = []
    labels: list[Label] = []
    for recording, units, number in placed:
        start = index_of[number] * seconds_per_unit if units else 0.0
        clips.append(Clip(recording, start, len(units) * seconds_per_unit))
        labels.extend(
            (start + at, text, artist, title) for at, text, artist, title in recording.labels
        )
    clips.sort(key=lambda clip: clip.start)
    opened_clip = next(clip for clip in clips if clip.recording.id == opened.id)
    return Stretch(
        opened=opened,
        clips=tuple(clips),
        units=tuple(by_number[number] for number in numbers),
        audio_format=audio_format,
        origin=opened.start - opened_clip.start,
        session=opened.session,
        first_unit=numbers[0] if numbers and opened.session else 0,
        labels=_merge_labels(labels),
    )


def _merge_labels(labels: list[Label]) -> tuple[Label, ...]:
    """By their seconds; the same text noted by two overlapping files once."""
    merged: list[Label] = []
    for label in sorted(labels, key=lambda label: label[0]):
        if any(
            other[1:] == label[1:] and abs(other[0] - label[0]) <= SAME_LABEL_S
            for other in merged[-4:]
        ):
            continue
        merged.append(label)
    return tuple(merged)


def kept_spans(stretch: Stretch, kept: tuple[Recording, ...]) -> list[tuple[float, float]]:
    """Where titles kept lie on the stretch, in seconds: the cuts made from it and
    titles kept whole."""
    spans = []
    seconds_per_unit = stretch.unit_seconds
    for recording in kept:
        if (recording.channel, recording.sid) != (stretch.opened.channel, stretch.opened.sid):
            continue
        if stretch.session and recording.session == stretch.session:
            start = (recording.first_unit - stretch.first_unit) * seconds_per_unit
        else:
            start = recording.start - stretch.origin
        end = start + recording.duration
        if end - start <= 0 or end <= 0 or start >= stretch.duration:
            continue
        spans.append((max(0.0, start), min(stretch.duration, end)))
    return sorted(spans)


def find_pauses(
    levels_db: np.ndarray, seconds_per_level: float, below_db: float = PAUSE_BELOW_DB
) -> list[tuple[float, float]]:
    """The pauses in the level of a stretch (RMS in dBFS every `seconds_per_level`, NaN
    where not read yet): as (beginning, end) in seconds, each at least `PAUSE_MIN_S`
    long."""
    quiet = np.nan_to_num(np.asarray(levels_db, dtype=float), nan=0.0) < below_db
    if not quiet.any():
        return []
    edges = np.diff(np.concatenate(([0], quiet.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    shortest = math.ceil(PAUSE_MIN_S / seconds_per_level - 1e-9)
    return [
        (float(start * seconds_per_level), float(end * seconds_per_level))
        for start, end in zip(starts, ends, strict=True)
        if end - start >= shortest
    ]


def pause_target(
    pauses: list[tuple[float, float]],
    position: float,
    beginning: bool,
    forward: bool,
    here: float = 1e-3,
) -> float | None:
    """Where a handle jumps to at the next pause from `position` on (`forward`) or
    before it: the beginning to the end of a pause (the sound begins), the end to its
    beginning, each keeping `PAUSE_KEEP_S` of the silence. A target within `here` of
    `position` is where the handle is already: half a unit, as the handle stands on a
    unit boundary and the pauses are found to the eighth of one. None without one."""
    targets = [
        max(start, end - PAUSE_KEEP_S) if beginning else min(end, start + PAUSE_KEEP_S)
        for start, end in pauses
    ]
    if forward:
        later = [t for t in targets if t > position + here]
        return min(later) if later else None
    earlier = [t for t in targets if t < position - here]
    return max(earlier) if earlier else None


def label_marks(stretch: Stretch) -> list[tuple[float, str]]:
    """Where a text appears on the stretch for the first time: stations alternate the
    title with a slogan and their web address, only the first appearance tells."""
    seen: set[str] = set()
    marks = []
    for at, text, _artist, _title in stretch.labels:
        if text and text not in seen:
            seen.add(text)
            marks.append((at, text))
    return marks


def split_text(text: str) -> tuple[str, str]:
    """ "ARTIST - TITLE" as artist and title, as many stations send it without DL Plus;
    any other text as the title."""
    for separator in (" - ", " – ", " — "):
        artist, found, title = text.partition(separator)
        if found and artist.strip() and title.strip() and separator not in title:
            return artist.strip(), title.strip()
    return "", text.strip()


def name_suggestions(stretch: Stretch) -> list[tuple[str, str]]:
    """Names for a cut, as (artist, title): those of the files, then those of the
    labels (DL Plus, or the text split at its dash), each once."""
    names: list[tuple[str, str]] = []
    for clip in stretch.clips:
        recording = clip.recording
        if recording.artist or recording.title:
            names.append((recording.artist, recording.title))
    for _at, text, artist, title in stretch.labels:
        if artist or title:
            names.append((artist, title))
        elif text:
            names.append(split_text(text))
    unique: list[tuple[str, str]] = []
    for name in names:
        key = (name[0].casefold(), name[1].casefold())
        if any((u[0].casefold(), u[1].casefold()) == key for u in unique):
            continue
        unique.append(name)
    return unique[:MAX_SUGGESTIONS]


def default_name(stretch: Stretch, start: float, end: float) -> tuple[str, str]:
    """The name of what lies between `start` and `end`: that of the file holding most
    of it, its text where it has none."""
    best, best_share = stretch.opened_clip(), -1.0
    for clip in stretch.clips:
        share = min(end, clip.end) - max(start, clip.start)
        if share > best_share:
            best, best_share = clip, share
    recording = best.recording
    if recording.artist or recording.title:
        return recording.artist, recording.title
    return split_text(recording.text)
