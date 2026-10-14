"""How well the station playing is received, for the bars of the radio page.

Judged from what the listener hears: the last audio frames of the selected service.
A frame counts as clean if it came through without any correction, as corrected if the
Reed-Solomon code had to repair it (DAB+ only), and as lost if audio of it is missing:
no superframe sync, an uncorrectable block or an invalid audio unit for DAB+, no sync
or a wrong header CRC for DAB.
"""

from __future__ import annotations

from collections import deque
from enum import Enum, auto

from dsp.mp2_frame import Mp2FrameResult
from dsp.superframe import SuperframeResult
from i18n import tr, tr_noop

# Frames judged: about 1.2 s of audio either way (a superframe lasts 120 ms, an MP2
# frame 24 ms at 48 kHz).
SUPERFRAME_WINDOW = 10
MP2_WINDOW = 50

# Translators: how well the station is received, after "Reception" on the radio page.
DESCRIPTIONS = {
    4: tr_noop("very good"),
    3: tr_noop("good"),
    2: tr_noop("fair"),
    1: tr_noop("poor"),
    0: tr_noop("no audio"),
}


class _Frame(Enum):
    CLEAN = auto()
    CORRECTED = auto()
    LOST = auto()


class ReceptionQuality:
    def __init__(self) -> None:
        self._frames: deque[_Frame] = deque(maxlen=SUPERFRAME_WINDOW)

    def reset(self) -> None:
        self._frames.clear()

    def add_superframe(self, result: SuperframeResult) -> None:
        self._use_window(SUPERFRAME_WINDOW)
        if (
            not result.sync_ok
            or result.rs_uncorrectable_blocks
            or len(result.audio_units) < result.num_audio_units
        ):
            self._frames.append(_Frame.LOST)
        elif result.rs_corrections:
            self._frames.append(_Frame.CORRECTED)
        else:
            self._frames.append(_Frame.CLEAN)

    def add_mp2_frame(self, result: Mp2FrameResult) -> None:
        self._use_window(MP2_WINDOW)
        clean = result.sync_ok and result.crc_ok
        self._frames.append(_Frame.CLEAN if clean else _Frame.LOST)

    def _use_window(self, length: int) -> None:
        if self._frames.maxlen != length:
            self._frames = deque(maxlen=length)

    @property
    def bars(self) -> int | None:
        """4 every frame clean, 3 some corrected but none lost, 2 up to a third lost,
        1 more lost, 0 all lost; None before the first frame."""
        if not self._frames:
            return None
        lost = self._frames.count(_Frame.LOST)
        if lost == len(self._frames):
            return 0
        if lost > len(self._frames) / 3:
            return 1
        if lost:
            return 2
        return 3 if _Frame.CORRECTED in self._frames else 4

    def describe(self) -> str | None:
        bars = self.bars
        return None if bars is None else tr(DESCRIPTIONS[bars])
