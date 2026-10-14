"""The Band III channel raster used for terrestrial DAB/DAB+."""

from __future__ import annotations

# DAB/DAB+ channel raster for Band III (VHF) as used for terrestrial DAB+ in Europe,
# Germany included. Centre frequencies in MHz. Source: the ETSI/CEPT frequency plan for
# Band III ("5A" .. "13F").
BAND_III_CHANNELS: dict[str, float] = {
    "5A": 174.928,
    "5B": 176.640,
    "5C": 178.352,
    "5D": 180.064,
    "6A": 181.936,
    "6B": 183.648,
    "6C": 185.360,
    "6D": 187.072,
    "7A": 188.928,
    "7B": 190.640,
    "7C": 192.352,
    "7D": 194.064,
    "8A": 195.936,
    "8B": 197.648,
    "8C": 199.360,
    "8D": 201.072,
    "9A": 202.928,
    "9B": 204.640,
    "9C": 206.352,
    "9D": 208.064,
    "10A": 209.936,
    "10B": 211.648,
    "10C": 213.360,
    "10D": 215.072,
    "11A": 216.928,
    "11B": 218.640,
    "11C": 220.352,
    "11D": 222.064,
    "12A": 223.936,
    "12B": 225.648,
    "12C": 227.360,
    "12D": 229.072,
    "13A": 230.784,
    "13B": 232.496,
    "13C": 234.208,
    "13D": 235.776,
    "13E": 237.488,
    "13F": 239.200,
}

_CHANNELS_BY_KHZ = {round(mhz * 1000): channel for channel, mhz in BAND_III_CHANNELS.items()}


def channel_for_frequency(frequency_khz: int) -> str | None:
    """The Band III channel with this centre frequency (in kHz), if there is one."""
    return _CHANNELS_BY_KHZ.get(frequency_khz)


# The nationwide German DAB+ multiplex ("Bundesmux").
DEFAULT_CHANNEL = "5C"
