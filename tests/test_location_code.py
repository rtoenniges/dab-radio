"""DAB location codes (ETSI TS 104 089, annexes A and F) against the examples of the
standard and of the German set-up guide for ASA."""

import pytest

from dab.location_code import (
    LocationCode,
    cell,
    centre,
    location_code,
    parse_presentation,
)

# Annex F.4 and F.5 with annex A.3, and the guide of asa.radio (its example address).
EXAMPLES = [
    (
        "BBC Broadcasting House",
        51.5187412,
        -0.1434571,
        LocationCode(10, "B736BB"),
        "2366-7443-8484",
    ),
    ("Svalbard Museum", 78.222609, 15.651605, LocationCode(0, "152FF1"), "1116-3388-7268"),
    ("Reichstag", 52.5186538, 13.3761015, LocationCode(1, "91BB81"), "1255-4467-1251"),
]


@pytest.mark.parametrize(
    ("latitude", "longitude", "code"),
    [example[1:4] for example in EXAMPLES],
    ids=[example[0] for example in EXAMPLES],
)
def test_the_code_of_a_position(latitude, longitude, code):
    assert location_code(latitude, longitude) == code


@pytest.mark.parametrize(
    ("code", "presentation"),
    [example[3:] for example in EXAMPLES],
    ids=[example[0] for example in EXAMPLES],
)
def test_the_presentation_format_both_ways(code, presentation):
    assert code.presentation == presentation
    assert parse_presentation(presentation) == code
    assert parse_presentation(presentation.replace("-", " ")) == code
    assert parse_presentation(presentation.replace("-", "")) == code


def test_the_test_code_of_asa_is_the_eiffel_tower():
    """The code of the function test the German guide gives: a cell of about 1 km in
    Paris, so that only receivers set to it play the test alert."""
    code = parse_presentation("1253-3513-3668")

    south, north, west, east = cell(code)
    assert code == LocationCode(1, "894095")
    assert south < 48.8582 <= north and west <= 2.2945 < east


@pytest.mark.parametrize(
    "text",
    ["1255-4467-1252", "1255-4467-125", "1255-4467-12510", "1255-4467-1259", "0255-4467-1251", ""],
    ids=["check sum", "too short", "too long", "symbol 9", "symbol 0", "empty"],
)
def test_entries_that_are_no_code(text):
    assert parse_presentation(text) is None


def test_the_digits_shorten_from_the_end():
    """Each digit fewer is the area that contains the finer one (annex F.1)."""
    full = location_code(52.5186538, 13.3761015)

    for digits in range(1, 6):
        coarse = location_code(52.5186538, 13.3761015, digits)
        assert coarse.digits == full.digits[:digits]
        assert coarse.contains(full)


def test_the_example_of_coding_an_alert_area():
    """Annex C: Z10:B6 reaches from (49.5, -4.5) to (51.75, -2.25)."""
    assert cell(LocationCode(10, "B6")) == (49.5, 51.75, -4.5, -2.25)


def test_matching_takes_the_digits_both_codes_have():
    """The example of clause 7.5.4."""
    receiver = LocationCode(1, "92CB81")
    alert_area = [LocationCode(1, digits) for digits in ("91F", "92C", "953", "960")]

    assert [code.contains(receiver) for code in alert_area] == [False, True, False, False]
    assert not LocationCode(2, "92C").contains(receiver)  # another zone


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [
        (51.4816, -3.1791),  # banded zone west of Greenwich
        (-33.8688, 151.2093),  # southern hemisphere
        (-17.7, 178.9),  # next to 180°
        (64.1, -21.9),  # zone row next to the north polar zone
        (85.0, -120.0),  # inner cap of the north polar zone
        (-77.846, 166.676),  # outer ring of the south polar zone
        (-88.0, 10.0),  # inner cap of the south polar zone
        (0.0, 0.0),
    ],
)
def test_the_middle_of_a_cell_has_its_code(latitude, longitude):
    code = location_code(latitude, longitude)
    south, north, west, east = cell(code)

    assert location_code(*centre(code)) == code
    # A cell takes its northern and western edges: the codes count from the north pole
    # and eastwards from Greenwich.
    assert south < latitude <= north
    assert west <= longitude < east or (east < west and (longitude >= west or longitude < east))


def test_a_whole_polar_zone_has_no_digits():
    assert cell(LocationCode(0)) == (72.0, 90.0, 0.0, 0.0)
    assert LocationCode(0).contains(location_code(85.0, -120.0))


def test_impossible_codes_are_refused():
    with pytest.raises(ValueError):
        LocationCode(42, "1")
    with pytest.raises(ValueError):
        LocationCode(1, "1234567")
    with pytest.raises(ValueError):
        location_code(52.0, 13.0, digits=7)
