"""MOT SlideShow according to ETSI TS 101 499 V3.2.1 (2023-07).

A station may sort its images into categories and number them. The receiver keeps them
in a "holding buffer" and offers two modes of operation (clause 5.2):

* In **normal mode** it shows every image as soon as it is due; the sequence is
  dictated by the station through the TriggerTime (clause 5.3.2).
* In **interactive mode** the user picks a category and browses its images in the order
  the station prescribes through the slide number.

Categories have no fixed meaning: the CategoryID is a freely assigned 8-bit number and
becomes readable through the CategoryTitle sent along with it (clause 5.3.5).

In practice stations use neither categories nor links, so that part follows the
standard without having been verified on air. Several stations encode a classification
in the file name instead; for uncategorised images it is derived from there as a
fallback (`category_from_name`). That is not part of the standard and is marked as such
in the GUI.

The images on air carry the TriggerTime "NOW" (four zero bytes) and no parameter other
than the ContentName. ExpireTime, Alert, categories, trigger times and header updates
are only tested on constructed data.

Retroactive changes arrive as a "header update" (clause 6.3): a MOT object without a
body that refers to an image sent earlier through its ContentName
(`SlideStore.apply_update`).

Images with an ExpireTime disappear from display and store at that moment (clause
5.3.4). The comparison uses the "SlideShow Reference Time", which is set from FIG 0/10
(Annex D, see `dab.timecode.ReferenceClock`); as long as it is not synchronised, no
image expires.

An image with the parameter Alert = 0x01 is an emergency warning (clause 6.2.10): when
it is shown the receiver returns from a category to normal mode. The standard defines
no further values.

When an image appears in normal mode is decided by its TriggerTime (clause 5.3.2,
table 2): "NOW" shows it on reception, a future time at that time. An image without a
TriggerTime, or with a TriggerTime that has passed, is only stored, not shown; a header
update may trigger it later. This too follows the broadcast time from FIG 0/10;
without it only "NOW" triggers (clause 8.2).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from urllib.parse import urlsplit

from dab.mot import IMAGE_SUBTYPES, MotObject
from dab.timecode import TimeParameter, parse_time_parameter
from i18n import tr, tr_plural

# ETSI TS 101 499 V3.2.1, table 3: the slideshow's MOT parameters. The two time
# parameters come from ETSI EN 301 234 (there called "Expiration"), the rest are
# specific to the slideshow.
PARAM_EXPIRE_TIME = 0x04
PARAM_TRIGGER_TIME = 0x05
PARAM_CATEGORY_SLIDE_ID = 0x25
PARAM_CATEGORY_TITLE = 0x26
PARAM_CLICK_THROUGH_URL = 0x27
PARAM_ALTERNATIVE_LOCATION_URL = 0x28
PARAM_ALERT = 0x29

# Clause 6.2.10, table 4: the only defined value of the Alert parameter. 0x00 is
# forbidden, 0x02 to 0xFF are reserved.
ALERT_EMERGENCY_WARNING = 0x01

# Clause 5.3.2: "The TriggerTime has an accuracy of 1 s." A TriggerTime counts as
# "equal" to the reference time as long as the latter is within the same second; only
# after that has it passed.
TRIGGER_RESOLUTION_SECONDS = 1.0

# Clause 9.2.2: the holding buffer of a receiver with the enhanced profile must hold 1
# to 64 images.
MAX_SLIDES = 64

# Clause 9.4.1: "Devices shall ignore URLs with any scheme other than http/https."
_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})

# Clause 9.2.3: permitted scaling. Enlarging is only allowed by at least 150 %,
# shrinking exclusively to 50 %, and the 50 % are mandatory. In between the image keeps
# its original size.
MIN_UPSCALE_FACTOR = 1.5
DOWNSCALE_FACTOR = 0.5

# Fallback classification from the file name. NOT part of the standard, merely an
# observation: NDR, for example, names its images "20260919_024300Z_WETTER_61065.jpg".
# Recognised is an all-upper-case word set off by underscores; names like "12.JPG"
# yield nothing.
_CATEGORY_FROM_NAME = re.compile(r"_([A-ZÄÖÜ]{3,})_")


def decode_utf8_parameter(value: bytes | None) -> str | None:
    """Decodes a text parameter of the slideshow.

    CategoryTitle, ClickThroughURL and AlternativeLocationURL are plain UTF-8 without a
    leading character set byte (clauses 5.3.3, 5.3.5.4, 6.2.9), unlike the ContentName,
    see `dab.mot.decode_content_name`.

    Returns:
        None if the parameter is absent; an empty string if it is present but empty.
        The difference matters for the category title (see `SlideStore.add`).
    """
    if value is None:
        return None
    return value.decode("utf-8", errors="replace")


def http_url(text: str | None) -> str | None:
    """Lets only http and https addresses through (clause 9.4.1).

    Not only for conformance: without this check a transmitted image could make an
    arbitrary address clickable, a local file for instance.
    """
    if not text:
        return None
    parts = urlsplit(text.strip())
    if parts.scheme.lower() not in _ALLOWED_URL_SCHEMES or not parts.netloc:
        return None
    return text.strip()


def permitted_scale_factor(image_width: int, image_height: int, width: int, height: int) -> float:
    """The scale factor permitted by clause 9.2.3 for a display area.

    The image is fitted to the area if that needs at least 150 % enlargement. If it
    roughly fits (between 100 and 150 %) it keeps its original size. If it is too big it
    is shrunk to 50 %, the only permitted reduction; if it still does not fit, clause
    9.1.3 has it cropped at the right and bottom.
    """
    if image_width <= 0 or image_height <= 0:
        return 1.0
    fit = min(width / image_width, height / image_height)
    if fit >= MIN_UPSCALE_FACTOR:
        return fit
    if fit >= 1.0:
        return 1.0
    return DOWNSCALE_FACTOR


def category_from_name(name: str | None) -> str | None:
    """Guesses a classification from the file name (see `_CATEGORY_FROM_NAME`)."""
    if not name:
        return None
    match = _CATEGORY_FROM_NAME.search(name)
    return match.group(1).capitalize() if match else None


@dataclass(frozen=True)
class Slide:
    """An image in the holding buffer together with its classification."""

    data: bytes
    content_subtype: int
    name: str | None
    category_id: int | None  # 1..255, None = not categorised
    slide_id: int | None  # 1..255, None = not categorised
    category_title: str | None  # None = parameter was absent
    click_through_url: str | None  # http/https only
    alternative_url: str | None  # http/https only; an image source on the web, not an info page
    received_at: float
    expire_time: float | None = None  # POSIX seconds UTC; None = never expires
    alert: int | None = None  # value of the Alert parameter; None = absent
    trigger: TimeParameter | None = None  # most recently received TriggerTime; None = absent

    @property
    def is_emergency_warning(self) -> bool:
        return self.alert == ALERT_EMERGENCY_WARNING

    @property
    def is_categorised(self) -> bool:
        return self.category_id is not None and self.slide_id is not None

    @property
    def category_key(self) -> int | str | None:
        """What the image is grouped by: the transmitted CategoryID, otherwise the
        name derived from the file name."""
        if self.is_categorised:
            return self.category_id
        return category_from_name(self.name)

    @property
    def key(self) -> tuple:
        """The image's slot in the holding buffer.

        The ContentName is decisive: it "is used for uniquely identifying the object
        for the purposes of cache management", and an image whose ContentName is
        already present replaces the older version (clause 6.2.2). The ContentName is
        mandatory; the fall-back to the content only applies to non-conforming objects
        without a name.
        """
        if self.name:
            return ("name", self.name)
        return ("data", hash(self.data))

    def describe(self) -> str:
        """Short description for the GUI."""
        kind = IMAGE_SUBTYPES.get(self.content_subtype)
        if kind is None:
            kind = tr("Type {subtype}").format(subtype=self.content_subtype)
        size = len(self.data)
        parts = [
            tr_plural("{kind}, {size} byte", "{kind}, {size} bytes", size).format(
                kind=kind, size=size
            )
        ]
        if self.name:
            parts.append(self.name)
        return " - ".join(parts)


def _category_and_slide(value: bytes | None) -> tuple[int | None, int | None]:
    """Reads the CategoryID/SlideID parameter (clause 6.2.6).

    16 bits: the upper byte is the CategoryID, the lower one the SlideID. The value 0
    is not permitted in either half except to take an image out of its category
    (clauses 5.3.5.2 and 5.3.5.3), so a 0 means "not categorised".
    """
    if value is None or len(value) != 2:
        return None, None
    category_id, slide_id = value[0], value[1]
    if category_id == 0 or slide_id == 0:
        return None, None
    return category_id, slide_id


def _is_expired(slide: Slide, now: float | None) -> bool:
    """Clause 5.3.4: "Once this is reached or passed", so already on equality. Both
    sides are accurate to one second."""
    return slide.expire_time is not None and now is not None and slide.expire_time <= now


def slide_from_object(mot_object: MotObject, received_at: float | None = None) -> Slide:
    """Interprets the slideshow parameters of a MOT object."""
    parameters = mot_object.parameters
    category_id, slide_id = _category_and_slide(parameters.get(PARAM_CATEGORY_SLIDE_ID))
    expire = parse_time_parameter(parameters.get(PARAM_EXPIRE_TIME))
    alert = parameters.get(PARAM_ALERT)
    trigger = parse_time_parameter(parameters.get(PARAM_TRIGGER_TIME))
    return Slide(
        data=mot_object.body,
        content_subtype=mot_object.content_subtype,
        name=mot_object.content_name,
        category_id=category_id,
        slide_id=slide_id,
        category_title=decode_utf8_parameter(parameters.get(PARAM_CATEGORY_TITLE)),
        click_through_url=http_url(decode_utf8_parameter(parameters.get(PARAM_CLICK_THROUGH_URL))),
        alternative_url=http_url(
            decode_utf8_parameter(parameters.get(PARAM_ALTERNATIVE_LOCATION_URL))
        ),
        received_at=time.time() if received_at is None else received_at,
        expire_time=None if expire is None else expire.timestamp,
        # 8 bits (clause 6.2.10); any other length is not a valid Alert.
        alert=alert[0] if alert is not None and len(alert) == 1 else None,
        trigger=trigger,
    )


class Presentation(Enum):
    """Where an image stands in normal mode (clause 5.3.2, table 2)."""

    SHOWN = "shown"  # triggered, now or earlier
    SCHEDULED = "scheduled"  # TriggerTime in the future
    AWAITING_CLOCK = "awaiting clock"  # TriggerTime known, reference clock not yet set
    NO_TRIGGER = "no trigger"  # stored, but not to be shown
    MISSED = "missed"  # TriggerTime had passed on reception: stored, not shown


@dataclass
class SlideStore:
    """The holding buffer of the enhanced profile (clause 9.2.2).

    For every image it also tracks whether and when it is to be shown in normal mode.
    `live_slide` tells which image currently stands there; `take_presented` reports a
    newly triggered image exactly once.
    """

    max_slides: int = MAX_SLIDES
    _slides: dict[tuple, Slide] = field(default_factory=dict)
    _titles: dict[int, str] = field(default_factory=dict)
    # TriggerTime (UTC) of the images waiting for their moment.
    _scheduled: dict[tuple, float] = field(default_factory=dict)
    # Likewise, but the reference clock was not yet set on reception: whether the
    # TriggerTime lies in the future only becomes clear once the clock is set.
    _awaiting_clock: dict[tuple, float] = field(default_factory=dict)
    _shown: set[tuple] = field(default_factory=set)
    _live: tuple | None = None
    _newly_presented: tuple | None = None

    def clear(self) -> None:
        self._slides.clear()
        self._titles.clear()
        self._scheduled.clear()
        self._awaiting_clock.clear()
        self._shown.clear()
        self._live = None
        self._newly_presented = None

    def add(
        self,
        mot_object: MotObject,
        received_at: float | None = None,
        now: float | None = None,
    ) -> Slide | None:
        """Takes in a received image.

        Args:
            received_at: time of reception by the computer's clock, for display and
                eviction.
            now: the SlideShow Reference Time, None = not synchronised.

        Returns:
            The stored image, or None if the object is not an image or had already
            expired on reception.
        """
        if not mot_object.is_image or not mot_object.body:
            return None
        # "Now" as ExpireTime means: invalid already on reception. TS 101 499 does not
        # foresee the value for the ExpireTime, but EN 301 234 (6.2.4.1) gives it this
        # meaning, and a station must not send expired objects at all (6.2.3.1.1).
        expire = parse_time_parameter(mot_object.parameters.get(PARAM_EXPIRE_TIME))
        if expire is not None and expire.is_now:
            return None
        slide = slide_from_object(mot_object, received_at)

        # Clauses 5.3.4 and 6.2.5: the ExpireTime is set only once; later values for
        # the same image (same ContentName) do not count.
        previous = self._slides.get(slide.key)
        if previous is not None and previous.expire_time is not None:
            slide = replace(slide, expire_time=previous.expire_time)
        if _is_expired(slide, now):
            if self._slides.pop(slide.key, None) is not None:
                self._forget(slide.key)
            return None

        # Clause 5.3.5.4: an object with an existing CategoryID and a new CategoryTitle
        # updates the title. An empty title is a "null CategoryTitle": the category is
        # then not offered.
        if slide.is_categorised and slide.category_title is not None:
            self._titles[slide.category_id] = slide.category_title

        self._decategorise_others(slide)

        # Clause 6.2.2: the same ContentName replaces the older version.
        self._slides.pop(slide.key, None)
        self._slides[slide.key] = slide
        self._evict(keep=slide.key)
        # The TriggerTime applies "at the instant of reception" (table 2): every
        # received version is evaluated afresh. If the image is currently on display,
        # the display is left alone (clause 6.2.2).
        self._schedule(slide.key, slide.trigger, now)
        return slide

    def apply_update(self, update: MotObject, now: float | None = None) -> Slide | None:
        """Applies a header update to an image in the holding buffer.

        The update finds its image through the ContentName (clause 6.3). Only
        TriggerTime and CategoryID/SlideID may change; all other parameters, including a
        CategoryTitle sent along, are to be ignored. If CategoryID/SlideID is missing
        from the update (or is not 16 bits long), the classification stays unchanged; a
        0 decategorises the image. A TriggerTime in the update is evaluated as on
        reception of an image: this is how a station triggers an image sent in advance
        without transmitting it again (EN 301 234, clause 7.1.1).

        Args:
            now: the SlideShow Reference Time, None = not synchronised.

        Returns:
            The changed image, or None if no image of that name is in the store, for
            instance because the update arrived before the image, which was missed when
            switching on in the middle of its transmission.
        """
        if not update.is_header_update or not update.content_name:
            return None
        key = ("name", update.content_name)
        slide = self._slides.get(key)
        if slide is None:
            return None

        value = update.parameters.get(PARAM_CATEGORY_SLIDE_ID)
        if value is not None and len(value) == 2:
            category_id, slide_id = _category_and_slide(value)
            slide = replace(slide, category_id=category_id, slide_id=slide_id)
            self._decategorise_others(slide)
        trigger = parse_time_parameter(update.parameters.get(PARAM_TRIGGER_TIME))
        if trigger is not None:
            slide = replace(slide, trigger=trigger)
        self._slides[key] = slide
        if trigger is not None:
            self._schedule(key, trigger, now)
        return slide

    def tick(self, now: float | None) -> bool:
        """Triggers the images whose TriggerTime has been reached.

        Several images due in the same tick are triggered in the order of their
        TriggerTime; the last one stays on display.

        Args:
            now: the SlideShow Reference Time. Without it (None) nothing is triggered
                (clause 8.2).

        Returns:
            Whether the trigger state of any image has changed.
        """
        if now is None:
            return False
        changed = bool(self._awaiting_clock)
        for key, when in list(self._awaiting_clock.items()):
            del self._awaiting_clock[key]
            self._evaluate(key, when, now)
        due = sorted(
            ((when, key) for key, when in self._scheduled.items() if when <= now),
            key=lambda item: item[0],
        )
        for _when, key in due:
            del self._scheduled[key]
            self._present(key)
        return changed or bool(due)

    def live_slide(self) -> Slide | None:
        """The image currently to be shown in normal mode."""
        return None if self._live is None else self._slides.get(self._live)

    def take_presented(self) -> Slide | None:
        """The image triggered since the last call, exactly once.

        A re-triggered, already shown image is reported too: for normal mode it is a
        new event."""
        key, self._newly_presented = self._newly_presented, None
        return None if key is None else self._slides.get(key)

    def presentation(self, slide: Slide) -> Presentation:
        """How the image is treated in normal mode. The associated moment is
        `slide.trigger`."""
        key = slide.key
        if key in self._scheduled:
            return Presentation.SCHEDULED
        if key in self._awaiting_clock:
            return Presentation.AWAITING_CLOCK
        # The image on display stays "shown" even if a new version without a
        # TriggerTime arrived: the display is left alone (6.2.2).
        if key in self._shown or key == self._live:
            return Presentation.SHOWN
        if slide.trigger is None:
            return Presentation.NO_TRIGGER
        return Presentation.MISSED

    def _schedule(self, key: tuple, trigger: TimeParameter | None, now: float | None) -> None:
        """Evaluates a received TriggerTime according to table 2."""
        self._scheduled.pop(key, None)
        self._awaiting_clock.pop(key, None)
        self._shown.discard(key)
        if trigger is None:
            return  # stored, not shown; a header update may trigger it
        if trigger.timestamp is None:  # "NOW"
            self._present(key)
            return
        self._evaluate(key, trigger.timestamp, now)

    def _evaluate(self, key: tuple, when: float, now: float | None) -> None:
        if now is None:
            # Clause 8.2: values other than "NOW" only with a synchronised reference
            # time; until then it stays open whether the moment is still to come.
            self._awaiting_clock[key] = when
        elif when > now:
            self._scheduled[key] = when
        elif now - when < TRIGGER_RESOLUTION_SECONDS:
            self._present(key)
        # otherwise passed: stored, but not shown

    def _present(self, key: tuple) -> None:
        self._live = key
        self._shown.add(key)
        self._newly_presented = key

    def _forget(self, key: tuple) -> None:
        """Clears the trigger state of a removed image. Clause 5.3.2: what is no longer
        in the store is not shown either."""
        self._scheduled.pop(key, None)
        self._awaiting_clock.pop(key, None)
        self._shown.discard(key)
        if self._live == key:
            self._live = None
        if self._newly_presented == key:
            self._newly_presented = None

    def expire(self, now: float | None) -> list[Slide]:
        """Removes all images whose ExpireTime has been reached or passed (clause
        5.3.4), from the store and thereby from the display.

        Args:
            now: the SlideShow Reference Time. Without it (None) nothing expires:
                whether an ExpireTime has been reached can only be told once the clock
                is set from FIG 0/10 (Annex D).

        Returns:
            The removed images.
        """
        expired = [slide for slide in self._slides.values() if _is_expired(slide, now)]
        for slide in expired:
            del self._slides[slide.key]
            self._forget(slide.key)
        return expired

    def _decategorise_others(self, slide: Slide) -> None:
        """Clauses 5.3.5.1 and 6.3: other images with the same CategoryID/SlideID are
        decategorised; they stay in the store and only lose their place in the
        category."""
        if not slide.is_categorised:
            return
        for key, other in list(self._slides.items()):
            if (
                key != slide.key
                and other.is_categorised
                and (other.category_id, other.slide_id) == (slide.category_id, slide.slide_id)
            ):
                self._slides[key] = replace(other, category_id=None, slide_id=None)

    def _evict(self, keep: tuple) -> None:
        """Makes room in the order of precedence from clause 9.2.2: uncategorised
        images first, categorised ones only after that, the oldest first in each case.
        The image just received is exempt; room is being made for it, after all."""
        while len(self._slides) > self.max_slides:
            candidates = [slide for key, slide in self._slides.items() if key != keep]
            if not candidates:
                return
            uncategorised = [slide for slide in candidates if not slide.is_categorised]
            victim = min(uncategorised or candidates, key=lambda slide: slide.received_at)
            del self._slides[victim.key]
            self._forget(victim.key)

    def categories(self) -> list[tuple[int | str, str]]:
        """The categories available for interactive mode.

        First the ones transmitted by the station, only those with a title and at least
        one image (clauses 5.2 and 5.3.5.4), ordered by CategoryID; then the groups
        derived from file names.
        """
        signalled = {slide.category_id for slide in self._slides.values() if slide.is_categorised}
        derived = {
            slide.category_key
            for slide in self._slides.values()
            if not slide.is_categorised and slide.category_key is not None
        }
        result: list[tuple[int | str, str]] = [
            (category, self._titles[category])
            for category in sorted(signalled)
            if self._titles.get(category)
        ]
        result += [(name, name) for name in sorted(derived)]
        return result

    def category_title(self, category_key: int | str | None) -> str:
        """Display title of a category; falls back to a generic label."""
        if isinstance(category_key, str):
            return category_key
        if category_key is None:
            return ""
        return self._titles.get(category_key) or tr("Category {number}").format(number=category_key)

    def slides(self, category_key: int | str) -> list[Slide]:
        """The images of a category.

        Transmitted categories in SlideID order (clause 5.3.5.3); derived groups have
        no slide number and are therefore ordered by time of reception, newest first.
        """
        if isinstance(category_key, str):
            slides = [
                slide
                for slide in self._slides.values()
                if not slide.is_categorised and slide.category_key == category_key
            ]
            return sorted(slides, key=lambda slide: slide.received_at, reverse=True)
        slides = [
            slide
            for slide in self._slides.values()
            if slide.is_categorised and slide.category_id == category_key
        ]
        return sorted(slides, key=lambda slide: slide.slide_id)

    def all_slides(self) -> list[Slide]:
        """All images, the most recently received first (for normal mode)."""
        return sorted(self._slides.values(), key=lambda slide: slide.received_at, reverse=True)

    def __len__(self) -> int:
        return len(self._slides)
