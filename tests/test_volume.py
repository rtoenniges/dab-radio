"""Tests for the volume scale (`audio.volume`): even in decibels over the slider and
silent at 0 %, the ceiling shifting the whole scale, a gain of the system as a
position, and the linear scale of an output with a volume of its own."""

from __future__ import annotations

import pytest

from audio.volume import (
    RANGE_DB,
    ceiling_share,
    db_of_share,
    device_share,
    percent_of_db,
    percent_of_share,
    volume_db,
    volume_gain,
)


def test_the_slider_is_even_in_decibels_and_silent_at_zero():
    assert volume_gain(0) == 0.0
    assert volume_db(0) is None
    assert volume_gain(100) == pytest.approx(1.0)
    assert volume_db(100) == pytest.approx(0.0)
    assert volume_db(50) == pytest.approx(-RANGE_DB / 2)
    assert volume_db(1) == pytest.approx(-RANGE_DB * 0.99)
    steps = [volume_db(percent + 1) - volume_db(percent) for percent in range(1, 100)]
    assert steps == pytest.approx([RANGE_DB / 100] * 99)
    assert volume_gain(80) == pytest.approx(10 ** (-12 / 20))


def test_the_ceiling_shifts_the_whole_scale():
    assert volume_db(100, -6) == pytest.approx(-6.0)
    assert volume_db(80, -6) == pytest.approx(volume_db(80) - 6)
    assert volume_gain(100, -6) == pytest.approx(10 ** (-6 / 20))
    assert volume_gain(0, -6) == 0.0
    assert volume_gain(120, -6) == volume_gain(100, -6)


def test_a_gain_the_system_reports_comes_back_as_a_position():
    assert percent_of_db(0.0, 0) == 100
    assert percent_of_db(-30.0, 0) == 50
    assert percent_of_db(-36.0, -6) == 50
    assert percent_of_db(-100.0, 0) == 1  # below the scale, still audible
    assert percent_of_db(3.0, 0) == 100  # above it
    for percent in range(1, 101):
        assert percent_of_db(volume_db(percent, -6), -6) == percent


def test_an_output_with_a_volume_of_its_own_takes_the_slider_linearly():
    assert device_share(0) == 0.0
    assert device_share(1) == pytest.approx(0.01)
    assert device_share(100) == pytest.approx(1.0)
    assert device_share(100, -6) == pytest.approx(0.794, abs=0.001)
    assert ceiling_share(0) == 1.0
    assert ceiling_share(-30) == pytest.approx(0.316, abs=0.001)
    assert percent_of_share(0.0) == 0
    assert percent_of_share(1 / 127) == 1  # the first step of a Bluetooth device
    assert percent_of_share(76 / 127) == 60
    assert percent_of_share(1.0, -6) == 100  # above the ceiling
    assert percent_of_share(0.397, -6) == 50
    for percent in range(101):
        assert percent_of_share(device_share(percent, -6), -6) == percent


def test_a_share_of_a_device_has_a_nominal_level_on_the_cubic_scale():
    assert db_of_share(0.0) is None
    assert db_of_share(1.0) == pytest.approx(0.0)
    assert db_of_share(0.5) == pytest.approx(-18.06, abs=0.01)
    assert db_of_share(0.1) == pytest.approx(-60.0)
