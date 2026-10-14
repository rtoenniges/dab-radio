"""Tests for the programme type names."""

from __future__ import annotations

import pytest

from dab.programme_type import PTY_NONE, programme_type_name


@pytest.mark.parametrize(
    ("code", "short", "full"),
    [
        (PTY_NONE, "None", "No programme type"),
        (3, "Information", "Information"),
        (10, "Pop Music", "Pop Music"),
        (12, "Easy Listening", "Easy Listening Music"),
        (18, "Children's Progs", "Children's programmes"),
        (29, "Documentary", "Documentary"),
    ],
)
def test_names_from_ts_101_756(code, short, full):
    assert programme_type_name(code) == (short, full)


@pytest.mark.parametrize("code", range(30))
def test_every_abbreviation_fits_16_characters(code):
    assert len(programme_type_name(code).short) <= 16, "meant for 16-character displays"


@pytest.mark.parametrize("code", [30, 31, 32, -1])
def test_codes_not_used_in_dab_have_no_name(code):
    assert programme_type_name(code) is None
