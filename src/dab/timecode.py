"""Time stamps in the DAB signal and the slideshow's reference clock.

DAB encodes points in time the same way everywhere: a date as a Modified Julian Date
(MJD, days since 17 November 1858) and a UTC time of day in short form (hours, minutes)
or long form (additionally seconds and milliseconds). That is how FIG 0/10, the
ensemble's time of day (ETSI EN 300 401, clause 8.1.3.1), and the time parameters of
MOT objects (ETSI EN 301 234, clause 6.2.4.1) look, for instance the ExpireTime of a
slideshow image.

Which clock a receiver compares such time stamps against is laid down by ETSI TS 101
499, Annex D: the "SlideShow Reference Time", set to UTC from FIG 0/10, not the
device's own clock, and not adjustable by the user (`ReferenceClock`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

# EN 300 401, clause 8.1.3.1: "MJD 0 is 1858-11-17" (00:00 UTC).
MJD_EPOCH = datetime(1858, 11, 17, tzinfo=UTC)

_SHORT_FORM_BYTES = 4  # validity/Rfu, MJD, UTC flag, hours, minutes
_LONG_FORM_BYTES = 6  # additionally seconds and milliseconds


def mjd_to_timestamp(
    mjd: int, hours: int, minutes: int, seconds: int = 0, milliseconds: int = 0
) -> float:
    """Converts an MJD and a UTC time of day to POSIX seconds (UTC)."""
    moment = MJD_EPOCH + timedelta(
        days=mjd, hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds
    )
    return moment.timestamp()


def _decode_mjd_utc(data: bytes) -> tuple[bool, float, bool] | None:
    """Reads the format shared by FIG 0/10 and the MOT time parameters.

    1 leading bit (validity for MOT, reserved for FIG 0/10), 17 bits MJD, 2 bits that
    do not concern the time (LSI and Rfa for FIG 0/10, reserved for MOT), 1 bit UTC
    flag, then 11 bits (hours, minutes) or 27 bits (additionally seconds, milliseconds).

    Returns:
        (leading bit, time stamp, long form), or None for a wrong length or an
        impossible time of day.
    """
    if len(data) < _SHORT_FORM_BYTES:
        return None
    first = int.from_bytes(data[:_SHORT_FORM_BYTES], "big")
    long_form = bool(first & 0x800)
    if long_form:
        if len(data) < _LONG_FORM_BYTES:
            return None
        field = int.from_bytes(data[:_LONG_FORM_BYTES], "big")
        leading = bool(field >> 47)
        mjd = (field >> 30) & 0x1FFFF
        hours = (field >> 22) & 0x1F
        minutes = (field >> 16) & 0x3F
        seconds = (field >> 10) & 0x3F
        milliseconds = field & 0x3FF
    else:
        leading = bool(first >> 31)
        mjd = (first >> 14) & 0x1FFFF
        hours = (first >> 6) & 0x1F
        minutes = first & 0x3F
        seconds = milliseconds = 0

    if hours > 23 or minutes > 59 or seconds > 59 or milliseconds > 999:
        return None
    return leading, mjd_to_timestamp(mjd, hours, minutes, seconds, milliseconds), long_form


@dataclass(frozen=True)
class BroadcastTime:
    """The ensemble's time of day from FIG 0/10."""

    timestamp: float  # POSIX seconds, UTC
    has_seconds: bool  # False for the obsolete short form: hours and minutes only


def parse_date_time(field: bytes) -> BroadcastTime | None:
    """Reads the date and time field of FIG 0/10 (EN 300 401, clause 8.1.3.1).

    `field` starts after the FIG 0 header byte (C/N, OE, P/D, extension). The time
    refers to the start of the null symbol of the transmission frame that carries it
    (clause 14.3.3). The short form is only still permitted for old transmitters, but
    receivers must keep understanding it.
    """
    decoded = _decode_mjd_utc(field)
    if decoded is None:
        return None
    _rfu, timestamp, long_form = decoded
    return BroadcastTime(timestamp, long_form)


@dataclass(frozen=True)
class TimeParameter:
    """A MOT time parameter such as TriggerTime or ExpireTime."""

    timestamp: float | None  # POSIX seconds, UTC; None = "Now"

    @property
    def is_now(self) -> bool:
        return self.timestamp is None


def parse_time_parameter(value: bytes | None) -> TimeParameter | None:
    """Reads a MOT time parameter (EN 301 234, clause 6.2.4.1).

    If the validity bit is 0 the parameter means "Now"; MJD and time of day are
    meaningless then. The length must match the UTC flag: 4 bytes for the short form,
    6 bytes for the long form.

    Returns:
        The time stamp, or None if the parameter is absent or does not follow this
        format.
    """
    if value is None or len(value) not in (_SHORT_FORM_BYTES, _LONG_FORM_BYTES):
        return None
    if not value[0] & 0x80:
        return TimeParameter(None)
    decoded = _decode_mjd_utc(value)
    if decoded is None:
        return None
    _valid, timestamp, long_form = decoded
    if len(value) != (_LONG_FORM_BYTES if long_form else _SHORT_FORM_BYTES):
        return None
    return TimeParameter(timestamp)


class ReferenceClock:
    """The "SlideShow Reference Time" (ETSI TS 101 499, Annex D).

    It is set exclusively from FIG 0/10 and keeps running on a monotonic clock in
    between; the computer's own clock plays no part. Until a time has been received it
    is not synchronised (`now()` returns None), and trigger and expiry times must not
    be evaluated then (clauses 8.2 and D.1).

    The short form of FIG 0/10 lacks the seconds. The clock is then only set at a
    minute change, i.e. when the reported minute differs from the previous report: at
    that moment the second is 0 (clause D.2). Since FIG 0/10 arrives once per second,
    this keeps the clock accurate to one second, the accuracy TS 101 499 specifies for
    the time parameters.
    """

    def __init__(self, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._monotonic = monotonic
        self._base: tuple[float, float] | None = None  # (UTC, monotonic time)
        self._last_short_form: float | None = None

    def reset(self) -> None:
        self._base = None
        self._last_short_form = None

    @property
    def is_synchronised(self) -> bool:
        return self._base is not None

    def synchronise(self, broadcast_time: BroadcastTime) -> None:
        """Sets the clock from a received FIG 0/10."""
        if broadcast_time.has_seconds:
            self._base = (broadcast_time.timestamp, self._monotonic())
            return
        previous = self._last_short_form
        self._last_short_form = broadcast_time.timestamp
        if previous is not None and broadcast_time.timestamp != previous:
            self._base = (broadcast_time.timestamp, self._monotonic())

    def now(self) -> float | None:
        """The current reference time in POSIX seconds (UTC), or None."""
        if self._base is None:
            return None
        utc, at = self._base
        return utc + (self._monotonic() - at)
