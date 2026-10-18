"""Where a title begins and ends, read from the dynamic label (`recording.titles`)."""

from __future__ import annotations

from dab.dls import DlPlus, DlPlusContentType, DynamicLabel
from recording.titles import TEXT_MEMORY_S, TitleInfo, TitleTracker


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _label(
    text: str, title: str | None = None, artist: str | None = None, *, running=True, toggle=False
):
    objects = {}
    if title is not None:
        objects[DlPlusContentType.ITEM_TITLE] = title
    if artist is not None:
        objects[DlPlusContentType.ITEM_ARTIST] = artist
    return DynamicLabel(text, DlPlus(toggle, running, objects))


def _plain(text: str) -> DynamicLabel:
    return DynamicLabel(text)


def test_a_title_begins_with_its_tags_and_ends_with_the_next():
    tracker = TitleTracker(_Clock())

    first = tracker.update(_label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA"))
    assert first == TitleInfo("TAKE ON ME", "A-HA", "A-HA - TAKE ON ME", tagged=True)
    assert first.name == "A-HA – TAKE ON ME"
    # The same item again, with another text in between: nothing new.
    assert tracker.update(_label("Absolut relax", "TAKE ON ME", "A-HA")) is None
    assert tracker.current.text == "Absolut relax"
    assert tracker.update(_label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA")) is None

    second = tracker.update(_label("QUEEN - RADIO GA GA", "RADIO GA GA", "QUEEN"))
    assert second == TitleInfo("RADIO GA GA", "QUEEN", "QUEEN - RADIO GA GA", tagged=True)


def test_tags_dropped_by_an_item_toggle_and_back_begin_nothing():
    """A station toggles the Item Toggle bit with every text, which drops the objects
    by the book; the title that comes back is still the same title."""
    tracker = TitleTracker(_Clock())
    tracker.update(_label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA", toggle=False))

    assert tracker.update(_label("Absolut relax - Entspannt", toggle=True)) is None
    assert tracker.current.title == "TAKE ON ME"
    assert tracker.update(_label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA", toggle=False)) is None


def test_the_item_running_flag_ends_a_title_and_begins_the_next():
    tracker = TitleTracker(_Clock())
    tracker.update(_label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA"))

    talk = tracker.update(_label("Nachrichten", running=False))
    assert talk == TitleInfo(text="Nachrichten")
    assert not talk.named
    assert tracker.update(_label("Wetter", running=False)) is None
    assert tracker.current.text == "Wetter"

    # The flag goes on before the tags: the stretch begins now, the tags refine it.
    started = tracker.update(_label("Jetzt: Musik", running=True))
    assert started == TitleInfo(text="Jetzt: Musik")
    assert tracker.update(_label("QUEEN - RADIO GA GA", "RADIO GA GA", "QUEEN")) is None
    assert tracker.current == TitleInfo("RADIO GA GA", "QUEEN", "QUEEN - RADIO GA GA", tagged=True)


def test_tags_that_come_late_refine_a_title_read_from_the_text():
    tracker = TitleTracker(_Clock())
    assert tracker.update(_plain("A-HA - TAKE ON ME")) == TitleInfo(
        "A-HA - TAKE ON ME", "", "A-HA - TAKE ON ME"
    )
    # The first DL Plus command arrives: the same stretch, now with proper tags.
    assert tracker.update(_label("A-HA - TAKE ON ME", "TAKE ON ME", "A-HA")) is None
    assert tracker.current.tagged and tracker.current.artist == "A-HA"


def test_without_dl_plus_a_text_not_seen_for_a_while_is_a_new_title():
    clock = _Clock()
    tracker = TitleTracker(clock)

    assert tracker.update(_plain("A-HA - TAKE ON ME")) is not None
    clock.now += 20
    assert tracker.update(_plain("Sie hören Radio XY")) is not None  # the first time
    clock.now += 20
    assert tracker.update(_plain("A-HA - TAKE ON ME")) is None  # seen 40 s ago
    clock.now += 20
    assert tracker.update(_plain("Sie hören Radio XY")) is None  # the slogan, seen before
    clock.now += 20
    new = tracker.update(_plain("QUEEN - RADIO GA GA"))
    assert new == TitleInfo("QUEEN - RADIO GA GA", "", "QUEEN - RADIO GA GA")
    assert tracker.update(_plain("Sie hören Radio XY")) is None

    clock.now += TEXT_MEMORY_S + 1
    assert tracker.update(_plain("Sie hören Radio XY")) is not None  # forgotten by now


def test_texts_are_tidied_and_reset_forgets_everything():
    tracker = TitleTracker(_Clock())
    assert tracker.update(_plain("  A-HA  -\nTAKE ON ME ")).text == "A-HA - TAKE ON ME"
    tracker.reset()
    assert tracker.current is None
    assert tracker.update(_plain("A-HA - TAKE ON ME")) is not None


def test_dl_plus_without_item_tags_leaves_the_boundaries_to_the_text():
    """89.0 RTL sends DL Plus with the running flag on but without item tags, and the
    flag stays on from one title to the next: the text has to tell, as without DL Plus;
    tags, once they come, settle it."""
    clock = _Clock()
    tracker = TitleTracker(clock)
    first = tracker.update(_label("Purple Disco Machine - Test My Luck"))
    assert first == TitleInfo(text="Purple Disco Machine - Test My Luck")
    clock.now += 30
    assert tracker.update(_label("89.0 RTL - Der Hitsender")) is not None  # the first time
    clock.now += 30
    assert tracker.update(_label("Purple Disco Machine - Test My Luck")) is None
    clock.now += 30
    assert tracker.update(_label("89.0 RTL - Der Hitsender")) is None
    clock.now += 30

    new = tracker.update(_label("Aitch - RMB"))
    assert new == TitleInfo(text="Aitch - RMB") and new.name == "Aitch - RMB"

    assert tracker.update(_label("Aitch - RMB", "RMB", "Aitch")) is None  # tags came late
    clock.now += 30
    assert tracker.update(_label("Ein neuer Text")) is None  # tagged: the text begins nothing
    assert tracker.current.title == "RMB" and tracker.current.text == "Ein neuer Text"
