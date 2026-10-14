"""Tests for the equalizer (`audio.equalizer`): each filter of the cookbook has the gain
asked for where it should, the audio is lowered by the highest boost and never clips,
the state carries from one block to the next, and the settings come back from the
settings file as they went in."""

from __future__ import annotations

import numpy as np
import pytest

from audio.equalizer import (
    BASS,
    DEFAULT_BANDS,
    GAIN_RANGE_DB,
    HIGH_SHELF,
    LINEAR,
    LOW_SHELF,
    PEAK,
    PRESETS,
    SPEECH,
    Band,
    Equalizer,
    EqualizerSettings,
    EqualizerState,
    coefficients,
    headroom_db,
    response_db,
)

RATE = 48_000


def _only(band: Band) -> EqualizerSettings:
    return EqualizerSettings((band,), None)


def _sine(frequency: float, amplitude: float = 0.5, seconds: float = 0.5) -> np.ndarray:
    """A tone on both channels, interleaved int16."""
    t = np.arange(int(RATE * seconds)) / RATE
    tone = np.round(amplitude * 32767 * np.sin(2 * np.pi * frequency * t)).astype(np.int16)
    return np.repeat(tone, 2)


def _level_db(pcm: np.ndarray) -> float:
    """RMS of the left channel after the filters have settled (the first 0.1 s left out)."""
    left = pcm.reshape(-1, 2)[RATE // 10 :, 0].astype(np.float64)
    return 20 * np.log10(np.sqrt(np.mean(left**2)))


def test_a_peaking_band_has_its_gain_at_its_frequency_and_half_of_it_at_its_edges():
    band = Band(PEAK, 1000.0, 6.0, width_octaves=1.0)
    edges = [1000.0 / np.sqrt(2), 1000.0 * np.sqrt(2)]

    at_centre, *at_edges = response_db(_only(band), [1000.0, *edges], RATE)

    assert at_centre == pytest.approx(6.0, abs=0.01)
    assert at_edges == pytest.approx([3.0, 3.0], abs=0.15)
    assert response_db(_only(band), [50.0, 15_000.0], RATE) == pytest.approx([0, 0], abs=0.1)


@pytest.mark.parametrize(
    ("kind", "frequency", "boosted", "untouched"),
    [(LOW_SHELF, 100.0, 20.0, 2000.0), (HIGH_SHELF, 8000.0, 20_000.0, 400.0)],
)
def test_a_shelf_raises_everything_beyond_its_frequency(kind, frequency, boosted, untouched):
    shelf = _only(Band(kind, frequency, 6.0))

    far, near, halfway = response_db(shelf, [boosted, untouched, frequency], RATE)

    assert far == pytest.approx(6.0, abs=0.3)
    assert near == pytest.approx(0.0, abs=0.2)
    assert halfway == pytest.approx(3.0, abs=0.1)


def test_a_cut_lowers_a_tone_at_its_frequency_and_nothing_is_raised_to_make_up():
    equalizer = Equalizer()
    equalizer.configure(_only(Band(PEAK, 1000.0, -6.0)))
    tone = _sine(1000.0)

    assert _level_db(equalizer.process(tone, 2, RATE)) - _level_db(tone) == pytest.approx(
        -6.0, abs=0.1
    )


def test_a_boost_lowers_the_rest_so_that_full_scale_does_not_clip():
    settings = _only(Band(PEAK, 1000.0, 6.0))
    assert headroom_db(settings, RATE) == pytest.approx(6.0, abs=0.01)
    loud = _sine(1000.0, amplitude=0.99)
    low = _sine(100.0, amplitude=0.5)

    equalizer = Equalizer()
    equalizer.configure(settings)
    boosted = equalizer.process(loud, 2, RATE)
    equalizer.reset()
    lowered = equalizer.process(low, 2, RATE)

    assert np.abs(boosted.astype(np.int32)).max() < 32767
    assert _level_db(boosted) == pytest.approx(_level_db(loud), abs=0.1)
    assert _level_db(lowered) - _level_db(low) == pytest.approx(-6.0, abs=0.1)


def test_flat_settings_leave_the_audio_alone():
    equalizer = Equalizer()
    tone = _sine(440.0)

    equalizer.configure(EqualizerSettings())
    assert not equalizer.active
    assert equalizer.process(tone, 2, RATE) is tone
    equalizer.configure(None)
    assert equalizer.process(tone, 2, RATE) is tone


def test_the_state_carries_over_so_blocks_join_without_a_seam():
    settings = EqualizerSettings.from_preset(BASS)
    tone = _sine(80.0, seconds=0.24)
    whole, parts = Equalizer(), Equalizer()
    whole.configure(settings)
    parts.configure(settings)

    in_one = whole.process(tone, 2, RATE)
    in_blocks = np.concatenate([parts.process(block, 2, RATE) for block in np.split(tone, 6)])

    np.testing.assert_array_equal(in_one, in_blocks)


def test_each_channel_is_filtered_on_its_own():
    equalizer = Equalizer()
    equalizer.configure(EqualizerSettings.from_preset(BASS))
    stereo = _sine(80.0).reshape(-1, 2).copy()
    stereo[:, 1] = 0

    out = equalizer.process(stereo.reshape(-1), 2, RATE).reshape(-1, 2)

    assert np.abs(out[:, 0]).max() > 0
    assert not out[:, 1].any()


def test_a_filter_near_half_the_sample_rate_stays_in_shape():
    band = Band(HIGH_SHELF, 20_000.0, 6.0)

    for rate in (16_000, 24_000, 32_000, 48_000):
        assert np.isfinite(coefficients(band, rate)).all()
        assert headroom_db(_only(band), rate) <= 6.0 + 1e-9


def test_presets_set_the_gains_of_the_default_bands():
    speech = EqualizerSettings.from_preset(SPEECH)

    assert speech.preset == SPEECH
    assert [band.gain_db for band in speech.bands] == list(PRESETS[SPEECH])
    assert [band.frequency_hz for band in speech.bands] == [b.frequency_hz for b in DEFAULT_BANDS]
    assert EqualizerSettings.from_preset(LINEAR).is_flat
    assert EqualizerSettings() == EqualizerSettings.from_preset(LINEAR)


def test_a_band_changed_by_hand_is_clamped_and_ends_the_preset():
    bass = EqualizerSettings.from_preset(BASS)

    changed = bass.with_band(3, Band(LOW_SHELF, 3000.0, 30.0, 9.0))

    assert changed.preset is None
    assert changed.bands[3] == Band(PEAK, 3000.0, GAIN_RANGE_DB[1], 4.0)
    assert changed.bands[:3] == bass.bands[:3]


def test_settings_come_back_from_their_stored_form():
    settings = EqualizerSettings.from_preset(SPEECH).with_band(1, Band(PEAK, 320.0, -3.5, 0.8))

    assert EqualizerSettings.from_dict(settings.to_dict()) == settings
    assert EqualizerSettings.from_dict(EqualizerSettings().to_dict()) == EqualizerSettings()


def test_a_state_comes_back_from_its_stored_form_with_its_switch_and_custom_settings():
    own = EqualizerSettings().with_band(2, Band(PEAK, 800.0, -2.0, 2.0))
    state = EqualizerState(True, EqualizerSettings.from_preset(BASS), own)

    assert EqualizerState.from_dict(state.to_dict()) == state
    assert EqualizerState.from_dict(EqualizerState().to_dict()) == EqualizerState()

    for switch in (1, "yes", None):
        stored = state.to_dict()
        stored["enabled"] = switch
        with pytest.raises(ValueError):
            EqualizerState.from_dict(stored)
    with pytest.raises(ValueError):
        EqualizerState.from_dict(state.settings.to_dict())  # no switch at all


def test_stored_settings_edited_by_hand_are_clamped_or_refused():
    stored = EqualizerSettings.from_preset(BASS).to_dict()
    stored["bands"][0][2] = 40.0  # a gain beyond the range
    clamped = EqualizerSettings.from_dict(stored)
    assert clamped.bands[0].gain_db == GAIN_RANGE_DB[1]
    assert clamped.preset is None  # no longer the preset's bands

    for broken in ({}, {"bands": [["peak", "x", 0, 1]]}, {"bands": [["peak", 1000, 0, 1]]}, []):
        with pytest.raises(ValueError):
            EqualizerSettings.from_dict(broken)
