"""A parametric equalizer for the decoded audio, run in the audio thread before the PCM
goes to the sound card (`audio.player.AudioPlayer`).

Five filters in a row: a low shelf, three peaking filters and a high shelf, each a
biquad after the Audio EQ Cookbook of Robert Bristow-Johnson. The settings are plain
values (`EqualizerSettings`): the controller stores them, and a surface draws their
response (`response_db`) without touching the filter state of the audio thread.

A boost would clip a programme that already reaches full scale, as most stations are
mastered. The whole signal is therefore lowered by the highest point of the response
(`headroom_db`): with the bass 6 dB up, the audio gets 6 dB quieter where nothing was
boosted, but never distorts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numba
import numpy as np

LOW_SHELF = "low_shelf"
PEAK = "peak"
HIGH_SHELF = "high_shelf"
KINDS = (LOW_SHELF, PEAK, HIGH_SHELF)

GAIN_RANGE_DB = (-12.0, 12.0)
FREQUENCY_RANGE_HZ = (20.0, 20_000.0)
# The width of a peaking filter in octaves between the points of half its gain in dB
# (the bandwidth of the cookbook); shelves have none.
WIDTH_RANGE_OCTAVES = (0.2, 4.0)
# A Q of about 1, the usual width of a band of a five-band equalizer.
DEFAULT_WIDTH_OCTAVES = 1.4
# Shelves have the steepest slope without overshoot (S = 1 in the cookbook).
SHELF_Q = 1.0 / math.sqrt(2.0)
# No filter is set closer to half the sample rate than this, where the bilinear
# transform squeezes it out of shape.
_HIGHEST_FRACTION = 0.45
# The headroom is the maximum of the response on this many points, log-spaced.
_HEADROOM_POINTS = 400


@dataclass(frozen=True)
class Band:
    kind: str  # LOW_SHELF, PEAK or HIGH_SHELF
    frequency_hz: float
    gain_db: float = 0.0
    width_octaves: float = DEFAULT_WIDTH_OCTAVES  # only for PEAK

    @property
    def q(self) -> float:
        if self.kind != PEAK:
            return SHELF_Q
        ratio = 2.0**self.width_octaves
        return math.sqrt(ratio) / (ratio - 1.0)

    def clamped(self) -> Band:
        """The band with each value in its range."""
        return replace(
            self,
            frequency_hz=_clamp(float(self.frequency_hz), *FREQUENCY_RANGE_HZ),
            gain_db=_clamp(float(self.gain_db), *GAIN_RANGE_DB),
            width_octaves=_clamp(float(self.width_octaves), *WIDTH_RANGE_OCTAVES),
        )


DEFAULT_BANDS = (
    Band(LOW_SHELF, 60.0),
    Band(PEAK, 250.0),
    Band(PEAK, 1000.0),
    Band(PEAK, 4000.0),
    Band(HIGH_SHELF, 12_500.0),
)

LINEAR = "linear"
SPEECH = "speech"
BASS = "bass"
TREBLE = "treble"
# The gain of each of the default bands, at their default frequency and width.
PRESETS: dict[str, tuple[float, ...]] = {
    LINEAR: (0.0, 0.0, 0.0, 0.0, 0.0),
    # Less boom below the voice, more of the consonants that carry intelligibility.
    SPEECH: (-6.0, -2.0, 1.0, 3.0, 0.0),
    BASS: (6.0, 2.0, 0.0, 0.0, 0.0),
    TREBLE: (0.0, 0.0, 0.0, 2.0, 5.0),
}


@dataclass(frozen=True)
class EqualizerSettings:
    bands: tuple[Band, ...] = DEFAULT_BANDS
    # The preset the bands come from unchanged; None once changed by hand.
    preset: str | None = LINEAR

    @classmethod
    def from_preset(cls, key: str) -> EqualizerSettings:
        bands = tuple(
            replace(band, gain_db=gain) for band, gain in zip(DEFAULT_BANDS, PRESETS[key])
        )
        return cls(bands, key)

    def with_band(self, index: int, band: Band) -> EqualizerSettings:
        """The settings with band `index` replaced (clamped to the ranges); no longer
        those of a preset."""
        bands = list(self.bands)
        bands[index] = replace(band.clamped(), kind=self.bands[index].kind)
        return EqualizerSettings(tuple(bands), None)

    @property
    def is_flat(self) -> bool:
        """Whether no band changes anything: then the filters are left out."""
        return all(band.gain_db == 0.0 for band in self.bands)

    def to_dict(self) -> dict:
        """A form for the settings file (JSON), read back by `from_dict`."""
        return {
            "preset": self.preset,
            "bands": [
                [band.kind, band.frequency_hz, band.gain_db, band.width_octaves]
                for band in self.bands
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> EqualizerSettings:
        """Settings from `to_dict`, each value clamped to its range. Raises ValueError
        for anything else (a file edited by hand), as for bands of other kinds than
        the default ones: the surfaces show those five."""
        try:
            preset = data.get("preset")
            entries = data["bands"]
            bands = tuple(
                Band(str(kind), float(frequency), float(gain), float(width)).clamped()
                for kind, frequency, gain, width in entries
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"equalizer settings not understood: {data!r}") from exc
        if tuple(band.kind for band in bands) != tuple(band.kind for band in DEFAULT_BANDS):
            raise ValueError(f"equalizer bands not understood: {data!r}")
        if preset is not None and preset not in PRESETS:
            preset = None
        if preset is not None and bands != EqualizerSettings.from_preset(preset).bands:
            preset = None  # edited by hand, or a preset that has changed since
        return cls(bands, preset)


@dataclass(frozen=True)
class EqualizerState:
    """The equalizer of all stations or of a single one: switched on or not, its
    settings, and the last settings made by hand, which the choice "Eigene" brings back
    after a preset."""

    enabled: bool = False
    settings: EqualizerSettings = EqualizerSettings()
    custom: EqualizerSettings | None = None

    def to_dict(self) -> dict:
        """A form for the settings file (JSON), read back by `from_dict`."""
        data = {"enabled": self.enabled, **self.settings.to_dict()}
        if self.custom is not None:
            data["custom"] = self.custom.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Mapping) -> EqualizerState:
        """A state from `to_dict`; raises ValueError for anything else."""
        try:
            enabled = data["enabled"]
            custom = data.get("custom")
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError(f"equalizer state not understood: {data!r}") from exc
        if enabled is not True and enabled is not False:  # 1 or "yes" by hand
            raise ValueError(f"equalizer state not understood: {data!r}")
        return cls(
            enabled,
            EqualizerSettings.from_dict(data),
            None if custom is None else EqualizerSettings.from_dict(custom),
        )


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def coefficients(band: Band, sample_rate: float) -> np.ndarray:
    """b0, b1, b2, a1, a2 of the band's biquad, normalised to a0 = 1."""
    a = 10.0 ** (band.gain_db / 40.0)
    frequency = min(band.frequency_hz, _HIGHEST_FRACTION * sample_rate)
    w0 = 2.0 * math.pi * frequency / sample_rate
    cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
    alpha = sin_w0 / (2.0 * band.q)
    if band.kind == PEAK:
        b = (1.0 + alpha * a, -2.0 * cos_w0, 1.0 - alpha * a)
        den = (1.0 + alpha / a, -2.0 * cos_w0, 1.0 - alpha / a)
    else:
        shelf = 2.0 * math.sqrt(a) * alpha
        if band.kind == LOW_SHELF:
            b = (
                a * ((a + 1.0) - (a - 1.0) * cos_w0 + shelf),
                2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0),
                a * ((a + 1.0) - (a - 1.0) * cos_w0 - shelf),
            )
            den = (
                (a + 1.0) + (a - 1.0) * cos_w0 + shelf,
                -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0),
                (a + 1.0) + (a - 1.0) * cos_w0 - shelf,
            )
        elif band.kind == HIGH_SHELF:
            b = (
                a * ((a + 1.0) + (a - 1.0) * cos_w0 + shelf),
                -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0),
                a * ((a + 1.0) + (a - 1.0) * cos_w0 - shelf),
            )
            den = (
                (a + 1.0) - (a - 1.0) * cos_w0 + shelf,
                2.0 * ((a - 1.0) - (a + 1.0) * cos_w0),
                (a + 1.0) - (a - 1.0) * cos_w0 - shelf,
            )
        else:
            raise ValueError(f"unknown kind of band: {band.kind!r}")
    return np.array([b[0], b[1], b[2], den[1], den[2]]) / den[0]


def response_db(
    settings: EqualizerSettings, frequencies: Sequence[float] | np.ndarray, sample_rate: float
) -> np.ndarray:
    """The gain of all bands together at each frequency, in dB."""
    z = np.exp(-2j * np.pi * np.asarray(frequencies, dtype=np.float64) / sample_rate)
    response = np.ones_like(z)
    for band in settings.bands:
        b0, b1, b2, a1, a2 = coefficients(band, sample_rate)
        response *= (b0 + b1 * z + b2 * z * z) / (1.0 + a1 * z + a2 * z * z)
    return 20.0 * np.log10(np.abs(response))


def headroom_db(settings: EqualizerSettings, sample_rate: float) -> float:
    """By how much the audio is lowered so that the highest boost does not clip: the
    maximum of the response up to 20 kHz (or near half the sample rate), at least 0."""
    highest = min(FREQUENCY_RANGE_HZ[1], 0.49 * sample_rate)
    grid = np.geomspace(FREQUENCY_RANGE_HZ[0], highest, _HEADROOM_POINTS)
    return max(0.0, float(response_db(settings, grid, sample_rate).max()))


@numba.njit(cache=True)
def _run_biquads(frames: np.ndarray, sections: np.ndarray, state: np.ndarray) -> None:
    """Runs the biquads one after the other over `frames` (frames x channels, in
    place), transposed direct form II, each section and channel with its two state
    values in `state` (sections x channels x 2), which carry over to the next block.

    A loop over the samples, which NumPy cannot vectorise (each output depends on the
    one before): JIT-compiled with numba. Five sections on 48 kHz stereo are some
    half a million steps per second, well under a millisecond per 120 ms superframe."""
    for section in range(sections.shape[0]):
        b0, b1, b2, a1, a2 = sections[section]
        for channel in range(frames.shape[1]):
            z1 = state[section, channel, 0]
            z2 = state[section, channel, 1]
            for n in range(frames.shape[0]):
                x = frames[n, channel]
                y = b0 * x + z1
                z1 = b1 * x - a1 * y + z2
                z2 = b2 * x - a2 * y
                frames[n, channel] = y
            state[section, channel, 0] = z1
            state[section, channel, 1] = z2


def warm_up() -> None:
    """Compiles (or loads from the cache) the filter loop, so that the first block to
    be filtered does not wait for numba: in the audio thread a second of compiling would
    run the sound card dry."""
    _run_biquads(np.zeros((1, 1)), np.zeros((1, 5)), np.zeros((1, 1, 2)))


class Equalizer:
    """Runs the filters of an `EqualizerSettings` over interleaved int16 PCM, keeping
    their state from one block to the next. Lives in the audio thread; `configure` may
    come between any two blocks."""

    def __init__(self) -> None:
        self._settings: EqualizerSettings | None = None
        self._sample_rate = 0
        self._channels = 0
        self._sections = np.zeros((0, 5))
        self._state = np.zeros((0, 0, 2))
        self._gain = 1.0
        self._stale = True  # coefficients and gain have to be worked out anew

    @property
    def active(self) -> bool:
        return self._settings is not None

    def configure(self, settings: EqualizerSettings | None) -> None:
        """New settings, None to switch the filters off. A change while playing keeps
        the state of the filters, so the audio goes on without a gap."""
        if settings is not None and settings.is_flat:
            settings = None  # nothing to filter
        if settings is None:
            self._state[:] = 0.0
        self._settings = settings
        self._stale = True

    def reset(self) -> None:
        """Forgets the state: for a new stream, whose start has nothing to do with the
        end of the last one."""
        self._state[:] = 0.0

    def process(self, pcm: np.ndarray, channels: int, sample_rate: int) -> np.ndarray:
        """The filtered samples, lowered by the headroom; `pcm` itself while off."""
        settings = self._settings
        if settings is None or channels <= 0 or len(pcm) == 0 or len(pcm) % channels:
            return pcm
        if (channels, sample_rate) != (self._channels, self._sample_rate):
            self._channels, self._sample_rate = channels, sample_rate
            self._state = np.zeros((len(settings.bands), channels, 2))
            self._stale = True
        if self._stale:
            self._sections = np.array([coefficients(b, sample_rate) for b in settings.bands])
            if self._state.shape[0] != len(settings.bands):
                self._state = np.zeros((len(settings.bands), channels, 2))
            self._gain = 10.0 ** (-headroom_db(settings, sample_rate) / 20.0)
            self._stale = False
        frames = pcm.reshape(-1, channels).astype(np.float64)
        _run_biquads(frames, self._sections, self._state)
        frames *= self._gain
        np.rint(frames, out=frames)
        np.clip(frames, -32768.0, 32767.0, out=frames)
        return frames.astype(np.int16).reshape(-1)
