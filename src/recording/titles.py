"""Where one title ends and the next begins, read from the dynamic label.

With DL Plus (ETSI TS 102 980) the station says it: `ITEM.TITLE` and `ITEM.ARTIST` name
the item on air, the Item Running flag says whether one is on air at all. A new item
begins when a running item's title or artist change to others, and when the flag goes
from off to on; the flag going off ends the item and begins a stretch without one (the
presenter, the news). Tags that only arrive a while after the item began, or come back
after a station dropped them (some toggle the Item Toggle bit with every text, which
drops the objects by the book, see `dab.dls`), do not begin anything: they refine the
stretch on air. Without DL Plus only the text is there, and stations alternate a few
texts (the title, a slogan, the web address): a text seen within the last
`TEXT_MEMORY_S` is one of those and begins nothing, one not seen for that long is a new
title. The same holds for DL Plus without item tags (some stations send the Item
Running flag alone, on from one title to the next): as long as no tags name the item
on air, the text decides.

`TitleTracker.update` takes every label as it arrives and returns the info of a new
item where one begins; `current` always carries the latest info of the stretch on air,
with the text last seen.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from dab.dls import DlPlus, DynamicLabel

# A text not seen for this long is a new title, where there is no DL Plus to tell.
TEXT_MEMORY_S = 600.0


@dataclass(frozen=True)
class TitleInfo:
    """What is known of the stretch on air."""

    title: str = ""
    artist: str = ""
    text: str = ""  # the dynamic label last seen
    tagged: bool = False  # title and artist come from DL Plus

    @property
    def named(self) -> bool:
        return bool(self.title or self.artist)

    @property
    def name(self) -> str:
        """ "Artist – Title", one of them, the text, or ""."""
        if self.title and self.artist:
            return f"{self.artist} – {self.title}"
        return self.title or self.artist or self.text


class TitleTracker:
    def __init__(self, now: Callable[[], float] = time.monotonic) -> None:
        self._now = now
        self._current: TitleInfo | None = None  # None until the first label
        self._running: bool | None = None  # the Item Running flag last seen
        self._seen: dict[str, float] = {}  # texts and when they were last seen (no DL Plus)

    @property
    def current(self) -> TitleInfo | None:
        return self._current

    def reset(self) -> None:
        """Another service: nothing of the labels so far applies."""
        self._current = None
        self._running = None
        self._seen.clear()

    def update(self, label: DynamicLabel) -> TitleInfo | None:
        """Takes a label; the info of a new item where one begins, else None."""
        text = " ".join(label.text.split())
        if label.dl_plus is None:
            return self._update_from_text(text)
        return self._update_from_dl_plus(text, label.dl_plus)

    def _update_from_dl_plus(self, text: str, dl_plus: DlPlus) -> TitleInfo | None:
        title = (dl_plus.title or "").strip()
        artist = (dl_plus.artist or "").strip()
        was_running, self._running = self._running, dl_plus.item_running
        current = self._current
        text_is_new = self._remember(text)
        if not dl_plus.item_running:
            # Nothing on air: the end of an item, or of a stretch that was running
            # without tags, begins a stretch without one.
            if current is None or current.named or was_running:
                return self._begin(TitleInfo(text=text))
            self._current = replace(current, text=text)
            return None
        if title or artist:
            if (
                current is not None
                and current.tagged
                and (title, artist)
                != (
                    current.title,
                    current.artist,
                )
            ):
                return self._begin(TitleInfo(title, artist, text, tagged=True))
            if current is None or was_running is False:
                return self._begin(TitleInfo(title, artist, text, tagged=True))
            # Tags for the stretch on air: they came late, came back, or replace a
            # title read from the text alone.
            self._current = TitleInfo(title, artist, text, tagged=True)
            return None
        if current is None or was_running is False:
            # An item began (the flag went on), its tags may follow.
            return self._begin(TitleInfo(text=text))
        if not current.tagged and text_is_new:
            # No tags name the item on air, and the flag stays on from one title to
            # the next: the text has to tell, as without DL Plus.
            return self._begin(TitleInfo(text=text))
        self._current = replace(current, text=text)
        return None

    def _update_from_text(self, text: str) -> TitleInfo | None:
        text_is_new = self._remember(text)
        current = self._current
        if current is not None and not text_is_new:
            self._current = replace(current, text=text)
            return None
        return self._begin(TitleInfo(title=text, text=text))

    def _remember(self, text: str) -> bool:
        """Notes the text as seen now; whether it had not been seen within
        `TEXT_MEMORY_S` (texts older than that are forgotten)."""
        now = self._now()
        last_seen = self._seen.get(text)
        self._seen = {t: at for t, at in self._seen.items() if now - at <= TEXT_MEMORY_S}
        self._seen[text] = now
        return last_seen is None or now - last_seen > TEXT_MEMORY_S

    def _begin(self, info: TitleInfo) -> TitleInfo:
        self._current = info
        return info
