"""Tests for time stamps (FIG 0/10, MOT time parameters) and the reference clock."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from dab.timecode import (
    BroadcastTime,
    ReferenceClock,
    mjd_to_timestamp,
    parse_date_time,
    parse_time_parameter,
)


def _utc(*args: int) -> float:
    return datetime(*args, tzinfo=UTC).timestamp()


def _long_form(mjd: int, h: int, m: int, s: int = 0, ms: int = 0, *, first_bit: int = 1) -> bytes:
    """48 bits: first bit, MJD, 2 free bits, UTC flag 1, h/m/s/ms."""
    value = (first_bit << 47) | (mjd << 30) | (1 << 27) | (h << 22) | (m << 16) | (s << 10) | ms
    return value.to_bytes(6, "big")


def _short_form(mjd: int, h: int, m: int, *, first_bit: int = 1) -> bytes:
    """32 bits: first bit, MJD, 2 free bits, UTC flag 0, h/m."""
    value = (first_bit << 31) | (mjd << 14) | (h << 6) | m
    return value.to_bytes(4, "big")


# MJD of 19 September 2026, computed and confirmed on real reception (see below).
MJD_2026_09_19 = 61302


def test_mjd_example_from_the_standard():
    """EN 300 401, clause 8.1.3.1: "MJD 58 000 corresponds to 2017-09-04"."""
    assert mjd_to_timestamp(58000, 0, 0) == _utc(2017, 9, 4)


def test_fig0_10_as_received():
    """The field exactly as the Bundesmux (5C) sent it on 19 September 2026; the
    computer's clock showed 12:51:21.42 UTC at the time."""
    received = parse_date_time(bytes.fromhex("3bdd8b335544"))

    assert received == BroadcastTime(_utc(2026, 9, 19, 12, 51, 21, 324000), has_seconds=True)


def test_fig0_10_short_form_has_no_seconds():
    """The short form is only intended for old transmitters any more, but must still
    be understood (EN 300 401, note in 8.1.3.1)."""
    received = parse_date_time(_short_form(MJD_2026_09_19, 14, 30, first_bit=0))

    assert received == BroadcastTime(_utc(2026, 9, 19, 14, 30), has_seconds=False)


def test_fig0_10_ignores_the_leading_rfu_bit():
    """In FIG 0/10 the first bit is reserved, not a validity flag as in MOT."""
    assert parse_date_time(_long_form(MJD_2026_09_19, 1, 2, 3, first_bit=1)) is not None
    assert parse_date_time(_long_form(MJD_2026_09_19, 1, 2, 3, first_bit=0)) is not None


@pytest.mark.parametrize("field", [b"", bytes(3), _long_form(MJD_2026_09_19, 1, 2)[:5]])
def test_fig0_10_too_short(field):
    assert parse_date_time(field) is None


def test_impossible_time_is_rejected():
    assert parse_date_time(_long_form(MJD_2026_09_19, 24, 0)) is None
    assert parse_date_time(_long_form(MJD_2026_09_19, 12, 60)) is None


# --- MOT time parameters (EN 301 234, clause 6.2.4.1) --------------------------


def test_time_parameter_long_form():
    value = parse_time_parameter(_long_form(MJD_2026_09_19, 18, 5, 30))
    assert value is not None and not value.is_now
    assert value.timestamp == _utc(2026, 9, 19, 18, 5, 30)


def test_time_parameter_short_form():
    value = parse_time_parameter(_short_form(MJD_2026_09_19, 18, 5))
    assert value is not None
    assert value.timestamp == _utc(2026, 9, 19, 18, 5)


@pytest.mark.parametrize("length", [4, 6])
def test_validity_flag_zero_means_now(length):
    """Validity bit 0: "Now"; MJD and UTC are meaningless then."""
    value = parse_time_parameter(bytes(length))
    assert value is not None and value.is_now


def test_time_parameter_length_must_match_the_utc_flag():
    # Short-form flag but 6 bytes long, and vice versa.
    assert parse_time_parameter(_short_form(MJD_2026_09_19, 1, 2) + b"\x00\x00") is None
    assert parse_time_parameter(_long_form(MJD_2026_09_19, 1, 2)[:4]) is None


@pytest.mark.parametrize("value", [None, b"", bytes([0x80]), bytes(5), bytes(7)])
def test_malformed_time_parameters(value):
    assert parse_time_parameter(value) is None


# --- Reference clock (TS 101 499, Annex D) -----------------------------------


class _FakeMonotonic:
    def __init__(self) -> None:
        self.value = 1000.0

    def __call__(self) -> float:
        return self.value


def test_clock_is_unsynchronised_until_the_first_time_arrives():
    clock = ReferenceClock(_FakeMonotonic())
    assert not clock.is_synchronised
    assert clock.now() is None


def test_clock_runs_on_from_the_received_time():
    monotonic = _FakeMonotonic()
    clock = ReferenceClock(monotonic)
    clock.synchronise(BroadcastTime(_utc(2026, 9, 19, 12, 0, 0), has_seconds=True))

    monotonic.value += 2.5

    assert clock.now() == _utc(2026, 9, 19, 12, 0, 2, 500000)


def test_clock_ignores_the_computer_clock():
    """The reference time comes from the signal alone: even a broadcast time far from
    the computer's clock is adopted."""
    clock = ReferenceClock(_FakeMonotonic())
    clock.synchronise(BroadcastTime(_utc(2001, 1, 1), has_seconds=True))
    assert clock.now() == _utc(2001, 1, 1)


def test_short_form_sets_the_clock_only_at_the_minute_change():
    """Clause D.2: with the short form, set at the minute change; before that it is
    unknown how far the minute has progressed."""
    monotonic = _FakeMonotonic()
    clock = ReferenceClock(monotonic)

    clock.synchronise(BroadcastTime(_utc(2026, 9, 19, 12, 0), has_seconds=False))
    monotonic.value += 1
    clock.synchronise(BroadcastTime(_utc(2026, 9, 19, 12, 0), has_seconds=False))
    assert clock.now() is None

    monotonic.value += 1
    clock.synchronise(BroadcastTime(_utc(2026, 9, 19, 12, 1), has_seconds=False))
    monotonic.value += 0.5
    assert clock.now() == _utc(2026, 9, 19, 12, 1, 0, 500000)


def test_reset_forgets_the_time():
    clock = ReferenceClock(_FakeMonotonic())
    clock.synchronise(BroadcastTime(_utc(2026, 9, 19), has_seconds=True))
    clock.reset()
    assert clock.now() is None
