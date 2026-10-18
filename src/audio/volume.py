"""The volume control: the slider position (percent) to the gain of the audio output.

The slider is even in decibels, not in amplitude. The ear hears loudness
logarithmically; with a linear gain the upper three quarters of the slider would cover
12 dB and everything from quiet to normal would crowd into the first quarter. Over
`RANGE_DB` every percent is 0.6 dB, about the smallest change heard; 0 % is silence.

A ceiling ("Maximum volume" in the settings) shifts the whole scale: 100 % is the
ceiling, every position below moves by the same amount and keeps its resolution. This
is for loudspeakers that take less than the amplifier delivers (5 W speakers on a 15 W
amplifier); the slider stays usable over its whole length instead of stopping
somewhere in the middle. Every output keeps its own ceiling and slider position; one
played to for the first time starts at `NEW_OUTPUT_PERCENT` without a ceiling,
whatever the output before played at.

An output with a volume of its own (a Bluetooth device, which PipeWire sets through
AVRCP absolute volume in 127 steps) takes the slider as a phone's: linear over its
steps (`device_share`), 1 % the first of them, 0 % silence; what its steps do to the
loudness is the device's own curve. The even decibels would compress its steps into
the upper part of the slider and reach none below the 13th, 10 % of the cubic scale
PulseAudio counts in (`CUBIC_DECADE_DB`), where a device may still be well audible.
The ceiling holds for such an output as the share of its range at 100 %, a factor on
that cubic scale (`ceiling_share`: -6 dB is 79 % of the range).

The gain goes to the sound system where it has a volume to set (`audio.system_volume`:
PipeWire on the Pi, which hands it on to Bluetooth devices as their own), else to the
audio sink (`AudioPlayer.set_volume`, the sample multiply Qt does anyway); this module
only turns the position into decibels, the factor or the share of a device's range.
"""

from __future__ import annotations

import math

# The slider from 1 % to 100 % spans this many decibels below the ceiling.
RANGE_DB = 60.0
# How far the ceiling can be lowered (dB, whole steps) and where it starts.
CEILING_RANGE_DB = (-30, 0)
DEFAULT_CEILING_DB = 0
# Where the slider of an output played to for the first time stands, without a
# ceiling: 10 % of a Bluetooth device's own range is quiet but plain, -54 dB on the
# scale of a loudspeaker barely audible; either is turned up from there rather than
# down from too loud.
NEW_OUTPUT_PERCENT = 10
# PulseAudio's volume scale is cubic in amplitude: this many decibels per decade of
# its value (`pa_sw_volume_to_dB`), which is also how a Bluetooth device's steps lie
# on it. The share of a device's range and a gain in dB convert over it.
CUBIC_DECADE_DB = 60.0


def volume_db(percent: int, ceiling_db: float = 0.0) -> float | None:
    """The gain of the slider position in dB; None for silence (0 %)."""
    if percent <= 0:
        return None
    return ceiling_db - RANGE_DB * (1.0 - min(percent, 100) / 100.0)


def volume_gain(percent: int, ceiling_db: float = 0.0) -> float:
    """The linear factor for the audio sink: 0.0 at 0 %, the ceiling at 100 %."""
    db = volume_db(percent, ceiling_db)
    return 0.0 if db is None else 10.0 ** (db / 20.0)


def percent_of_db(db: float, ceiling_db: float) -> int:
    """The slider position under `ceiling_db` that gives a gain in dB (one the sound
    system reports, `audio.system_volume`): 1 % below the scale, 100 % above it."""
    return max(1, min(100, round(100.0 * (1.0 + (db - ceiling_db) / RANGE_DB))))


# --- Outputs with a volume of their own ------------------------------------------------


def ceiling_share(ceiling_db: float) -> float:
    """The share of a device's own range the ceiling allows: 1.0 at 0 dB, 0.79 at -6."""
    return 10.0 ** (ceiling_db / CUBIC_DECADE_DB)


def device_share(percent: int, ceiling_db: float = 0.0) -> float:
    """The share of a device's own volume range for the slider position: 0.0 at 0 %,
    linear up to `ceiling_share` at 100 %."""
    return min(max(percent, 0), 100) / 100.0 * ceiling_share(ceiling_db)


def percent_of_share(share: float, ceiling_db: float = 0.0) -> int:
    """The slider position under `ceiling_db` for a share of the device's range (one
    the sound system reports): 0 for none, 1 % for the least, 100 % at the ceiling and
    above it."""
    if share <= 0.0:
        return 0
    return max(1, min(100, round(100.0 * share / ceiling_share(ceiling_db))))


def db_of_share(share: float) -> float | None:
    """The nominal gain in dB of a share of a device's range, on the cubic scale: what
    PulseAudio counts for it, not what the device makes of it; None for 0."""
    return None if share <= 0.0 else CUBIC_DECADE_DB * math.log10(share)
