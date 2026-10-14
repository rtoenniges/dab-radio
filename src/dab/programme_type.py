"""Programme types (PTy): what a service broadcasts, e.g. pop music or news.

FIG 0/17 carries one code per service (ETSI EN 300 401, clause 8.1.5). The codes come
from the international table for DAB outside North America (ETSI TS 101 756, table 12),
which largely matches the programme types of RDS. The names below are the English ones
of that table: the full name and the 16-character abbreviation that is recommended for
receiver displays. The German catalog has those of annex A of the same document (table
A.3).
"""

from __future__ import annotations

from typing import NamedTuple

from i18n import tr_context

PTY_NONE = 0  # "no programme type"


class ProgrammeTypeName(NamedTuple):
    short: str  # 16-character abbreviation, for lists
    full: str


def _short_names() -> tuple[str, ...]:
    """The abbreviations, indexed by the PTy code, in the language set."""
    # Translators: the 16-character abbreviations of the programme types (ETSI TS 101
    # 756, table 12; German: table A.3, written there with "_" for the spaces). Each must
    # fit in 16 characters.
    return (
        tr_context("programme type, 16 characters", "None"),
        tr_context("programme type, 16 characters", "News"),
        tr_context("programme type, 16 characters", "Current Affairs"),
        tr_context("programme type, 16 characters", "Information"),
        tr_context("programme type, 16 characters", "Sport"),
        tr_context("programme type, 16 characters", "Education"),
        tr_context("programme type, 16 characters", "Drama"),
        tr_context("programme type, 16 characters", "Culture"),
        tr_context("programme type, 16 characters", "Science"),
        tr_context("programme type, 16 characters", "Varied"),
        tr_context("programme type, 16 characters", "Pop Music"),
        tr_context("programme type, 16 characters", "Rock Music"),
        tr_context("programme type, 16 characters", "Easy Listening"),
        tr_context("programme type, 16 characters", "Light Classical"),
        tr_context("programme type, 16 characters", "Serious Classics"),
        tr_context("programme type, 16 characters", "Other Music"),
        tr_context("programme type, 16 characters", "Weather"),
        tr_context("programme type, 16 characters", "Finance"),
        tr_context("programme type, 16 characters", "Children's Progs"),
        tr_context("programme type, 16 characters", "Social Affairs"),
        tr_context("programme type, 16 characters", "Religion"),
        tr_context("programme type, 16 characters", "Phone In"),
        tr_context("programme type, 16 characters", "Travel"),
        tr_context("programme type, 16 characters", "Leisure"),
        tr_context("programme type, 16 characters", "Jazz Music"),
        tr_context("programme type, 16 characters", "Country Music"),
        tr_context("programme type, 16 characters", "National Music"),
        tr_context("programme type, 16 characters", "Oldies Music"),
        tr_context("programme type, 16 characters", "Folk Music"),
        tr_context("programme type, 16 characters", "Documentary"),
    )


def _full_names() -> tuple[str, ...]:
    """The full names, indexed by the PTy code, in the language set."""
    # Translators: the programme types (ETSI TS 101 756, table 12; German: table A.3).
    return (
        tr_context("programme type", "No programme type"),
        tr_context("programme type", "News"),
        tr_context("programme type", "Current Affairs"),
        tr_context("programme type", "Information"),
        tr_context("programme type", "Sport"),
        tr_context("programme type", "Education"),
        tr_context("programme type", "Drama"),
        tr_context("programme type", "Culture"),
        tr_context("programme type", "Science"),
        tr_context("programme type", "Varied"),
        tr_context("programme type", "Pop Music"),
        tr_context("programme type", "Rock Music"),
        tr_context("programme type", "Easy Listening Music"),
        tr_context("programme type", "Light Classical"),
        tr_context("programme type", "Serious Classical"),
        tr_context("programme type", "Other Music"),
        tr_context("programme type", "Weather/meteorology"),
        tr_context("programme type", "Finance/Business"),
        tr_context("programme type", "Children's programmes"),
        tr_context("programme type", "Social Affairs"),
        tr_context("programme type", "Religion"),
        tr_context("programme type", "Phone In"),
        tr_context("programme type", "Travel"),
        tr_context("programme type", "Leisure"),
        tr_context("programme type", "Jazz Music"),
        tr_context("programme type", "Country Music"),
        tr_context("programme type", "National Music"),
        tr_context("programme type", "Oldies Music"),
        tr_context("programme type", "Folk Music"),
        tr_context("programme type", "Documentary"),
    )


# Codes 30 and 31 are not used in DAB (in RDS they are the alarm codes).
_CODES = 30


def programme_type_name(code: int) -> ProgrammeTypeName | None:
    """The name of a PTy code in the language set, None for a code that is not used in
    DAB."""
    if 0 <= code < _CODES:
        return ProgrammeTypeName(_short_names()[code], _full_names()[code])
    return None
