"""Tests for the MOT slideshow according to ETSI TS 101 499 V3.2.1.

The test data follow the encoding from the standard. That is the point: an earlier
version of these tests had built category titles and URLs with a character set byte in
front, the way the implementation wrongly expected. Both agreed with each other and
were wrong together.
"""

from __future__ import annotations

import pytest

from dab.mot import (
    CONTENT_TYPE_IMAGE,
    CONTENT_TYPE_MOT_TRANSPORT,
    MOT_TRANSPORT_HEADER_UPDATE,
    MotObject,
)
from dab.slideshow import (
    MAX_SLIDES,
    PARAM_ALERT,
    PARAM_ALTERNATIVE_LOCATION_URL,
    PARAM_CATEGORY_SLIDE_ID,
    PARAM_CATEGORY_TITLE,
    PARAM_CLICK_THROUGH_URL,
    PARAM_EXPIRE_TIME,
    PARAM_TRIGGER_TIME,
    Presentation,
    SlideStore,
    category_from_name,
    http_url,
    permitted_scale_factor,
    slide_from_object,
)


def _image(
    body: bytes = b"\xff\xd8\xff\xe0BILD",
    *,
    name: str | None = "bild.jpg",
    category_slide: bytes | None = None,
    category_title: str | None = None,
    url: str | None = None,
    alternative_url: str | None = None,
    expire: bytes | None = None,
    trigger: bytes | None = None,
) -> MotObject:
    """An image object with parameters encoded as laid down in TS 101 499:
    CategoryID/SlideID as a 16-bit field, texts as plain UTF-8."""
    parameters: dict[int, bytes] = {}
    if expire is not None:
        parameters[PARAM_EXPIRE_TIME] = expire
    if trigger is not None:
        parameters[PARAM_TRIGGER_TIME] = trigger
    if category_slide is not None:
        parameters[PARAM_CATEGORY_SLIDE_ID] = category_slide
    if category_title is not None:
        parameters[PARAM_CATEGORY_TITLE] = category_title.encode()
    if url is not None:
        parameters[PARAM_CLICK_THROUGH_URL] = url.encode()
    if alternative_url is not None:
        parameters[PARAM_ALTERNATIVE_LOCATION_URL] = alternative_url.encode()
    return MotObject(
        transport_id=1,
        content_type=CONTENT_TYPE_IMAGE,
        content_subtype=1,
        content_name=name,
        body=body,
        parameters=parameters,
    )


def _update(
    name: str | None,
    *,
    category_slide: bytes | None = None,
    category_title: str | None = None,
    url: str | None = None,
    trigger: bytes | None = None,
) -> MotObject:
    """A header update: no body, content type "MOT transport"/0 (ETSI EN 301 234,
    clause 7.1.3). It refers to an image through the ContentName."""
    parameters: dict[int, bytes] = {}
    if trigger is not None:
        parameters[PARAM_TRIGGER_TIME] = trigger
    if category_slide is not None:
        parameters[PARAM_CATEGORY_SLIDE_ID] = category_slide
    if category_title is not None:
        parameters[PARAM_CATEGORY_TITLE] = category_title.encode()
    if url is not None:
        parameters[PARAM_CLICK_THROUGH_URL] = url.encode()
    return MotObject(
        transport_id=99,
        content_type=CONTENT_TYPE_MOT_TRANSPORT,
        content_subtype=MOT_TRANSPORT_HEADER_UPDATE,
        content_name=name,
        body=b"",
        parameters=parameters,
    )


def _at(timestamp: float) -> bytes:
    """Time parameter in long form (EN 301 234, 6.2.4.1), milliseconds 0."""
    days, seconds = divmod(int(timestamp), 86400)
    mjd = days + 40587  # 1970-01-01 is MJD 40587
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    value = (1 << 47) | (mjd << 30) | (1 << 27) | (hours << 22) | (minutes << 16) | (secs << 10)
    return value.to_bytes(6, "big")


# A fixed reference time for the expiry tests: 19 September 2026, 12:00:00 UTC.
NOON = 1789819200.0


def _in(category: int, slide: int) -> bytes:
    """CategoryID/SlideID: upper byte category, lower byte slide number (6.2.6)."""
    return bytes([category, slide])


# --- Encoding -----------------------------------------------------------------


def test_text_parameters_are_plain_utf8():
    """Clauses 5.3.3 and 5.3.5.4: plain UTF-8, no character set byte. With exactly
    these values the earlier version turned into garbage or lost the first letter."""
    slide = slide_from_object(
        _image(
            category_slide=_in(1, 1),
            category_title="Nachrichten",
            url="https://www.swr3.de/wetter",
        )
    )

    assert slide.category_title == "Nachrichten"
    assert slide.click_through_url == "https://www.swr3.de/wetter"


def test_non_ascii_titles_are_decoded_as_utf8():
    slide = slide_from_object(_image(category_slide=_in(2, 1), category_title="Straße & Grüße"))
    assert slide.category_title == "Straße & Grüße"


def test_category_and_slide_id_are_read_from_the_16_bit_field():
    slide = slide_from_object(_image(category_slide=_in(3, 7)))

    assert slide.is_categorised
    assert (slide.category_id, slide.slide_id) == (3, 7)


@pytest.mark.parametrize("value", [_in(0, 0), _in(0, 5), _in(3, 0)])
def test_zero_in_either_half_means_not_categorised(value):
    """Clauses 5.3.5.2 and 5.3.5.3: 0 is permitted neither as CategoryID nor as
    SlideID, except to take an image out of its category."""
    slide = slide_from_object(_image(category_slide=value))
    assert not slide.is_categorised


@pytest.mark.parametrize("value", [b"\x03", b"\x03\x07\x00"])
def test_category_slide_id_must_be_exactly_16_bit(value):
    assert not slide_from_object(_image(category_slide=value)).is_categorised


# --- Links --------------------------------------------------------------------


@pytest.mark.parametrize("url", ["http://example.org/a", "https://example.org/b?c=d"])
def test_http_and_https_urls_are_kept(url):
    assert http_url(url) == url


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "javascript:alert(1)", "ftp://example.org/x", "/relativ", ""]
)
def test_other_url_schemes_are_ignored(url):
    """Clause 9.4.1: "Devices shall ignore URLs with any scheme other than
    http/https." Without the check every transmitted address would be clickable."""
    assert http_url(url) is None
    assert slide_from_object(_image(url=url)).click_through_url is None


def test_alternative_location_url_is_kept_separately():
    """The AlternativeLocationURL is an image source on the web (6.2.9), not a page
    with further information; it must not take the place of the ClickThroughURL."""
    slide = slide_from_object(_image(alternative_url="https://example.org/bild.jpg"))

    assert slide.alternative_url == "https://example.org/bild.jpg"
    assert slide.click_through_url is None


# --- Holding buffer -----------------------------------------------------------


def test_same_category_and_slide_id_decategorises_the_older_slide():
    """Clause 5.3.5.1: the older image is decategorised, not deleted; it stays in the
    store, just no longer in the category."""
    store = SlideStore()
    store.add(
        _image(b"alt", name="alt.jpg", category_slide=_in(1, 2), category_title="Wetter"), 1.0
    )
    store.add(_image(b"neu", name="neu.jpg", category_slide=_in(1, 2)), 2.0)

    assert [slide.data for slide in store.slides(1)] == [b"neu"]
    assert {slide.data for slide in store.all_slides()} == {b"alt", b"neu"}


def test_same_content_name_overwrites_the_prior_version():
    """Clause 6.2.2."""
    store = SlideStore()
    store.add(_image(b"alt", name="logo.jpg"), 1.0)
    store.add(_image(b"neu", name="logo.jpg"), 2.0)

    assert [slide.data for slide in store.all_slides()] == [b"neu"]


def test_slides_of_a_category_follow_the_slide_id():
    """Clause 5.3.5.3: in SlideID order, not in order of reception."""
    store = SlideStore()
    for received, slide_id in enumerate([3, 1, 2]):
        store.add(
            _image(
                name=f"{slide_id}.jpg", category_slide=_in(1, slide_id), category_title="Wetter"
            ),
            float(received),
        )

    assert [slide.slide_id for slide in store.slides(1)] == [1, 2, 3]


def test_category_without_title_is_not_offered():
    """Clause 5.3.5.4: "Categories with CategoryID but with a null CategoryTitle
    shall not be shown to the user." """
    store = SlideStore()
    store.add(_image(name="a.jpg", category_slide=_in(9, 1)))

    assert store.categories() == []
    assert len(store.all_slides()) == 1, "in normal mode the image stays visible"


def test_category_title_arriving_later_makes_the_category_visible():
    """Not every image has to carry the title; it only has to arrive once."""
    store = SlideStore()
    store.add(_image(name="a.jpg", category_slide=_in(4, 1)), 1.0)
    store.add(_image(name="b.jpg", category_slide=_in(4, 2), category_title="Events"), 2.0)

    assert store.categories() == [(4, "Events")]
    assert len(store.slides(4)) == 2


def test_category_title_is_updated_by_a_later_object():
    store = SlideStore()
    store.add(_image(name="a.jpg", category_slide=_in(4, 1), category_title="Alt"), 1.0)
    store.add(_image(name="b.jpg", category_slide=_in(4, 2), category_title="Neu"), 2.0)

    assert store.categories() == [(4, "Neu")]


def test_empty_category_title_hides_the_category():
    store = SlideStore()
    store.add(_image(name="a.jpg", category_slide=_in(4, 1), category_title="Events"), 1.0)
    store.add(_image(name="b.jpg", category_slide=_in(4, 2), category_title=""), 2.0)

    assert store.categories() == []


def test_categories_without_slides_are_not_offered():
    """Clause 5.2: only categories with at least one image."""
    store = SlideStore()
    store.add(_image(b"a", name="a.jpg", category_slide=_in(1, 1), category_title="Wetter"), 1.0)
    # Same ContentName without a category: the image leaves category 1.
    store.add(_image(b"b", name="a.jpg"), 2.0)

    assert store.categories() == []


def test_categories_are_ordered_by_id():
    store = SlideStore()
    store.add(_image(name="b.jpg", category_slide=_in(7, 1), category_title="Zweite"))
    store.add(_image(name="a.jpg", category_slide=_in(2, 1), category_title="Erste"))

    assert store.categories() == [(2, "Erste"), (7, "Zweite")]


def test_eviction_removes_uncategorised_slides_first():
    """Clause 9.2.2: uncategorised images first, categorised ones only after that."""
    store = SlideStore(max_slides=2)
    store.add(_image(name="kat.jpg", category_slide=_in(1, 1), category_title="K"), 1.0)
    store.add(_image(name="ohne1.jpg"), 2.0)
    store.add(_image(name="ohne2.jpg"), 3.0)

    assert {slide.name for slide in store.all_slides()} == {"kat.jpg", "ohne2.jpg"}


def test_eviction_takes_the_oldest_categorised_slide_when_nothing_else_is_left():
    store = SlideStore(max_slides=2)
    store.add(_image(name="a.jpg", category_slide=_in(1, 1), category_title="K"), 1.0)
    store.add(_image(name="b.jpg", category_slide=_in(1, 2)), 2.0)
    store.add(_image(name="c.jpg", category_slide=_in(1, 3)), 3.0)

    assert [slide.name for slide in store.slides(1)] == ["b.jpg", "c.jpg"]


def test_the_newly_received_slide_is_never_evicted_to_make_room_for_itself():
    store = SlideStore(max_slides=1)
    store.add(_image(name="kat.jpg", category_slide=_in(1, 1), category_title="K"), 1.0)
    store.add(_image(name="neu.jpg"), 2.0)

    assert [slide.name for slide in store.all_slides()] == ["neu.jpg"]


def test_capacity_follows_the_enhanced_profile():
    """Clause 9.2.2: 1 to 64 images."""
    assert MAX_SLIDES == 64
    store = SlideStore()
    for index in range(MAX_SLIDES + 5):
        store.add(_image(bytes([index]), name=f"{index}.jpg"), float(index))

    assert len(store) == MAX_SLIDES


def test_all_slides_newest_first():
    store = SlideStore()
    store.add(_image(b"alt", name="a.jpg"), 1.0)
    store.add(_image(b"neu", name="b.jpg"), 2.0)

    assert [slide.data for slide in store.all_slides()] == [b"neu", b"alt"]


def test_non_image_objects_are_ignored():
    store = SlideStore()
    other = MotObject(1, content_type=5, content_subtype=0, content_name="x", body=b"abc")

    assert store.add(other) is None
    assert len(store) == 0


def test_clear_empties_the_store():
    store = SlideStore()
    store.add(_image(category_slide=_in(1, 1), category_title="K"))
    store.clear()

    assert len(store) == 0
    assert store.categories() == []


# --- Derivation from the file name (not part of the standard) ------------------


def test_category_is_derived_from_the_file_name():
    assert category_from_name("20260919_024300Z_WETTER_61065.jpg") == "Wetter"
    assert category_from_name("20260919_024240Z_COVER_61007.jpg") == "Cover"
    assert category_from_name("12.JPG") is None
    assert category_from_name(None) is None


def test_derived_groups_follow_the_signalled_categories():
    store = SlideStore()
    store.add(_image(name="20260919_0243Z_WETTER_1.jpg"), 1.0)
    store.add(_image(name="20260919_0244Z_WETTER_2.jpg"), 2.0)
    store.add(_image(name="x.jpg", category_slide=_in(5, 1), category_title="Nachrichten"), 3.0)

    assert store.categories() == [(5, "Nachrichten"), ("Wetter", "Wetter")]
    assert [slide.name for slide in store.slides("Wetter")] == [
        "20260919_0244Z_WETTER_2.jpg",
        "20260919_0243Z_WETTER_1.jpg",
    ]


def test_signalled_category_takes_precedence_over_the_file_name():
    store = SlideStore()
    store.add(_image(name="x_WETTER_1.jpg", category_slide=_in(3, 1), category_title="Nachrichten"))

    assert store.categories() == [(3, "Nachrichten")]


# --- Display ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("area", "expected"),
    [
        ((800, 600), 2.5),  # fits with more than 150 %: fit to the area
        ((480, 360), 1.5),  # exactly 150 %
        ((400, 300), 1.0),  # only 125 %: not permitted, original size
        ((320, 240), 1.0),
        ((200, 150), 0.5),  # too small: the only permitted reduction
        ((100, 75), 0.5),  # even when 50 % is not enough (cropping)
    ],
)
def test_permitted_scale_factor(area, expected):
    """Clause 9.2.3: enlarge from 150 % on, shrink only to 50 %."""
    assert permitted_scale_factor(320, 240, *area) == pytest.approx(expected)


def test_scale_factor_for_empty_image():
    assert permitted_scale_factor(0, 0, 800, 600) == 1.0


# --- Header updates (clause 6.3) --------------------------------------------------


def test_update_moves_a_slide_into_a_category():
    store = SlideStore()
    store.add(_image(b"bild", name="bild.jpg"), 1.0)
    store.add(_image(b"x", name="x.jpg", category_slide=_in(4, 1), category_title="Wetter"), 2.0)

    updated = store.apply_update(_update("bild.jpg", category_slide=_in(4, 2)))

    assert updated is not None
    assert (updated.category_id, updated.slide_id) == (4, 2)
    assert [slide.data for slide in store.slides(4)] == [b"x", b"bild"]


def test_update_decategorises_the_slide_holding_that_position():
    """Clause 6.3: "Any other slide with the same CategoryID/SlideID value shall be
    decategorized." """
    store = SlideStore()
    store.add(_image(b"alt", name="alt.jpg", category_slide=_in(1, 1), category_title="A"), 1.0)
    store.add(_image(b"neu", name="neu.jpg"), 2.0)

    store.apply_update(_update("neu.jpg", category_slide=_in(1, 1)))

    assert [slide.data for slide in store.slides(1)] == [b"neu"]
    assert len(store) == 2, "the displaced image stays in the store"


@pytest.mark.parametrize("value", [bytes([0, 0]), bytes([0, 5]), bytes([5, 0])])
def test_update_with_zero_decategorises(value):
    store = SlideStore()
    store.add(_image(b"b", name="b.jpg", category_slide=_in(2, 3), category_title="B"), 1.0)

    updated = store.apply_update(_update("b.jpg", category_slide=value))

    assert updated is not None and not updated.is_categorised
    assert store.slides(2) == []
    assert store.categories() == []


def test_update_without_category_parameter_keeps_the_category():
    store = SlideStore()
    store.add(_image(b"b", name="b.jpg", category_slide=_in(2, 3), category_title="B"), 1.0)

    store.apply_update(_update("b.jpg"))

    assert [slide.data for slide in store.slides(2)] == [b"b"]


def test_malformed_category_parameter_in_an_update_changes_nothing():
    store = SlideStore()
    store.add(_image(b"b", name="b.jpg", category_slide=_in(2, 3), category_title="B"), 1.0)

    store.apply_update(_update("b.jpg", category_slide=bytes([7])))

    assert [slide.data for slide in store.slides(2)] == [b"b"]


def test_update_ignores_all_other_parameters():
    """Clause 6.3: only TriggerTime and CategoryID/SlideID may change; a title or
    link sent along has no effect."""
    store = SlideStore()
    store.add(_image(b"b", name="b.jpg", category_slide=_in(2, 3), category_title="Alt"), 1.0)

    updated = store.apply_update(_update("b.jpg", category_title="Neu", url="https://example.org/"))

    assert store.category_title(2) == "Alt"
    assert updated is not None and updated.click_through_url is None


def test_update_changes_neither_image_nor_reception_time():
    store = SlideStore()
    store.add(_image(b"bild", name="b.jpg"), 5.0)

    updated = store.apply_update(_update("b.jpg", category_slide=_in(1, 1)))

    assert updated is not None
    assert (updated.data, updated.received_at) == (b"bild", 5.0)


def test_update_for_an_unknown_name_is_ignored():
    """The image may have been missed when switching on; the update then has nothing
    to refer to."""
    store = SlideStore()
    store.add(_image(b"b", name="b.jpg"), 1.0)

    assert store.apply_update(_update("fehlt.jpg", category_slide=_in(1, 1))) is None
    assert len(store) == 1
    assert store.slides(1) == []


def test_update_without_content_name_is_ignored():
    store = SlideStore()
    store.add(_image(b"b", name="b.jpg"), 1.0)

    assert store.apply_update(_update(None, category_slide=_in(1, 1))) is None


def test_an_image_is_not_applied_as_an_update():
    store = SlideStore()
    store.add(_image(b"b", name="b.jpg"), 1.0)

    assert store.apply_update(_image(b"c", name="b.jpg", category_slide=_in(1, 1))) is None
    assert [slide.data for slide in store.all_slides()] == [b"b"]


def test_header_update_is_not_added_as_an_image():
    store = SlideStore()
    assert store.add(_update("b.jpg")) is None
    assert len(store) == 0


# --- ExpireTime (clauses 5.3.4 and 6.2.5) ---------------------------------------


def test_expire_time_is_read():
    slide = slide_from_object(_image(expire=_at(NOON + 60)))
    assert slide.expire_time == NOON + 60


def test_slide_without_expire_time_never_expires():
    store = SlideStore()
    store.add(_image(name="a.jpg"), 1.0, now=NOON)

    assert store.expire(NOON + 10 * 365 * 86400) == []
    assert len(store) == 1


def test_slide_is_removed_when_its_expire_time_is_reached():
    """ "Once this is reached or passed", so already on equality."""
    store = SlideStore()
    store.add(_image(b"a", name="a.jpg", expire=_at(NOON + 60)), 1.0, now=NOON)
    store.add(_image(b"b", name="b.jpg"), 2.0, now=NOON)

    assert store.expire(NOON + 59.9) == []
    removed = store.expire(NOON + 60)

    assert [slide.data for slide in removed] == [b"a"]
    assert [slide.data for slide in store.all_slides()] == [b"b"]


def test_nothing_expires_without_a_synchronised_reference_time():
    """Annex D: without a set reference clock it cannot be told whether an
    ExpireTime has been reached, so the image stays."""
    store = SlideStore()
    store.add(_image(name="a.jpg", expire=_at(NOON - 3600)), 1.0, now=None)

    assert len(store) == 1
    assert store.expire(None) == []
    assert len(store) == 1

    assert len(store.expire(NOON)) == 1, "once the clock is set, it expires"


def test_already_expired_slide_is_not_taken_in():
    store = SlideStore()
    assert store.add(_image(name="a.jpg", expire=_at(NOON - 1)), 1.0, now=NOON) is None
    assert len(store) == 0


def test_expire_time_now_means_expired_on_arrival():
    """Validity bit 0 stands for "Now" (EN 301 234, 6.2.4.1); as an expiry time that
    means expired already on reception, even without a reference clock."""
    store = SlideStore()
    assert store.add(_image(name="a.jpg", expire=bytes(6)), 1.0, now=None) is None


def test_expire_time_is_set_only_once():
    """Clause 5.3.4: "any subsequent updates shall be ignored"; a newly received
    object of the same name does not move the expiry."""
    store = SlideStore()
    store.add(_image(b"v1", name="a.jpg", expire=_at(NOON + 60)), 1.0, now=NOON)
    store.add(_image(b"v2", name="a.jpg", expire=_at(NOON + 3600)), 2.0, now=NOON)

    [slide] = store.all_slides()
    assert slide.data == b"v2"
    assert slide.expire_time == NOON + 60


def test_first_expire_time_can_come_with_a_later_version():
    store = SlideStore()
    store.add(_image(b"v1", name="a.jpg"), 1.0, now=NOON)
    store.add(_image(b"v2", name="a.jpg", expire=_at(NOON + 60)), 2.0, now=NOON)

    [slide] = store.all_slides()
    assert slide.expire_time == NOON + 60


def test_repeated_object_after_its_expire_time_does_not_come_back():
    store = SlideStore()
    store.add(_image(name="a.jpg", expire=_at(NOON + 60)), 1.0, now=NOON)
    store.expire(NOON + 60)

    assert store.add(_image(name="a.jpg", expire=_at(NOON + 60)), 2.0, now=NOON + 61) is None
    assert len(store) == 0


def test_header_update_cannot_change_the_expire_time():
    """Clause 6.3: only TriggerTime and CategoryID/SlideID can change."""
    store = SlideStore()
    store.add(_image(name="a.jpg", expire=_at(NOON + 60)), 1.0, now=NOON)
    update = _update("a.jpg")
    update.parameters[PARAM_EXPIRE_TIME] = _at(NOON + 3600)

    updated = store.apply_update(update)

    assert updated is not None and updated.expire_time == NOON + 60


# --- Alert (clause 6.2.10) -------------------------------------------------------


def _alert(value: bytes) -> MotObject:
    image = _image(name="warnung.jpg")
    image.parameters[PARAM_ALERT] = value
    return image


def test_alert_0x01_is_an_emergency_warning():
    slide = slide_from_object(_alert(bytes([0x01])))
    assert slide.alert == 0x01
    assert slide.is_emergency_warning


def test_slide_without_alert_is_no_warning():
    slide = slide_from_object(_image())
    assert slide.alert is None
    assert not slide.is_emergency_warning


@pytest.mark.parametrize("value", [0x00, 0x02, 0xFF])
def test_forbidden_and_reserved_alert_values_are_no_warning(value):
    """0x00 "shall not be used", 0x02..0xFF are reserved (table 4); none of them may
    tear the display out of a category."""
    slide = slide_from_object(_alert(bytes([value])))
    assert slide.alert == value
    assert not slide.is_emergency_warning


@pytest.mark.parametrize("value", [b"", bytes([0x01, 0x00])])
def test_alert_must_be_exactly_8_bit(value):
    slide = slide_from_object(_alert(value))
    assert slide.alert is None
    assert not slide.is_emergency_warning


def test_header_update_cannot_raise_an_alert():
    """Clause 6.3: an update only changes TriggerTime and CategoryID/SlideID."""
    store = SlideStore()
    store.add(_image(name="a.jpg"), 1.0)
    update = _update("a.jpg")
    update.parameters[PARAM_ALERT] = bytes([0x01])

    updated = store.apply_update(update)

    assert updated is not None and not updated.is_emergency_warning


def test_alert_is_part_of_the_object_version():
    """A new object of the same name replaces the old one entirely (clause 6.2.2),
    including its Alert parameter or the absence of one."""
    store = SlideStore()
    store.add(_alert(bytes([0x01])), 1.0)
    store.add(_image(name="warnung.jpg"), 2.0)

    [slide] = store.all_slides()
    assert not slide.is_emergency_warning


# --- TriggerTime (clause 5.3.2, table 2) ----------------------------------------

# "NOW" stands for a validity bit of 0 (clause 6.2.4); that is exactly how Dlf, Dlf
# Kultur, Dlf Nova and KLASSIK RADIO send it: four zero bytes.
NOW = bytes(4)


def _presented(store: SlideStore) -> bytes | None:
    slide = store.take_presented()
    return None if slide is None else slide.data


def test_trigger_now_shows_the_slide_on_reception():
    store = SlideStore()
    store.add(_image(b"a", name="a.jpg", trigger=NOW), 1.0, now=None)

    assert _presented(store) == b"a"
    assert store.live_slide().data == b"a"


def test_presentation_is_reported_only_once():
    store = SlideStore()
    store.add(_image(b"a", name="a.jpg", trigger=NOW), 1.0)

    assert _presented(store) == b"a"
    assert _presented(store) is None
    assert store.live_slide().data == b"a", "it stays on display nonetheless"


def test_slide_without_trigger_time_is_held_but_not_shown():
    """Table 2, last row."""
    store = SlideStore()
    slide = store.add(_image(b"a", name="a.jpg"), 1.0, now=NOON)

    assert _presented(store) is None
    assert store.live_slide() is None
    assert len(store) == 1
    assert store.presentation(slide) is Presentation.NO_TRIGGER


def test_future_trigger_time_waits_until_it_is_reached():
    store = SlideStore()
    slide = store.add(_image(b"a", name="a.jpg", trigger=_at(NOON + 30)), 1.0, now=NOON)
    assert store.presentation(slide) is Presentation.SCHEDULED

    store.tick(NOON + 29.9)
    assert _presented(store) is None

    store.tick(NOON + 30)
    assert _presented(store) == b"a"
    assert store.presentation(slide) is Presentation.SHOWN


def test_scheduled_slide_is_shown_even_if_the_tick_comes_late():
    """The image was scheduled; a late tick is the receiver's business, not the
    station's."""
    store = SlideStore()
    store.add(_image(b"a", name="a.jpg", trigger=_at(NOON + 30)), 1.0, now=NOON)

    store.tick(NOON + 45)

    assert _presented(store) == b"a"


def test_trigger_time_equal_to_the_reference_time_shows_the_slide():
    """Accuracy 1 s: 12:00:00 is "equal" as long as the clock stands within that
    second."""
    store = SlideStore()
    store.add(_image(b"a", name="a.jpg", trigger=_at(NOON)), 1.0, now=NOON + 0.7)

    assert _presented(store) == b"a"


def test_past_trigger_time_is_held_but_not_shown():
    store = SlideStore()
    slide = store.add(_image(b"a", name="a.jpg", trigger=_at(NOON)), 1.0, now=NOON + 1)

    store.tick(NOON + 2)

    assert _presented(store) is None
    assert store.presentation(slide) is Presentation.MISSED


def test_datetime_trigger_waits_for_the_reference_time():
    """Clause 8.2: values other than "NOW" only with a synchronised clock."""
    store = SlideStore()
    slide = store.add(_image(b"a", name="a.jpg", trigger=_at(NOON + 30)), 1.0, now=None)
    assert store.presentation(slide) is Presentation.AWAITING_CLOCK

    store.tick(None)
    assert _presented(store) is None

    store.tick(NOON)
    assert store.presentation(slide) is Presentation.SCHEDULED
    store.tick(NOON + 30)
    assert _presented(store) == b"a"


def test_trigger_that_passed_before_the_clock_was_set_is_not_shown():
    store = SlideStore()
    slide = store.add(_image(b"a", name="a.jpg", trigger=_at(NOON)), 1.0, now=None)

    store.tick(NOON + 60)

    assert _presented(store) is None
    assert store.presentation(slide) is Presentation.MISSED


def test_several_due_slides_are_shown_in_trigger_order():
    store = SlideStore()
    store.add(_image(b"spaet", name="b.jpg", trigger=_at(NOON + 20)), 1.0, now=NOON)
    store.add(_image(b"frueh", name="a.jpg", trigger=_at(NOON + 10)), 2.0, now=NOON)

    store.tick(NOON + 30)

    assert store.live_slide().data == b"spaet"


def test_newer_trigger_replaces_the_live_slide():
    store = SlideStore()
    store.add(_image(b"a", name="a.jpg", trigger=NOW), 1.0)
    store.add(_image(b"b", name="b.jpg"), 2.0)
    store.add(_image(b"c", name="c.jpg", trigger=NOW), 3.0)

    assert store.live_slide().data == b"c"


def test_repeated_reception_with_now_triggers_again():
    """ "NOW" applies at the instant of reception; a station that repeats its images
    in turn shows them again every time."""
    store = SlideStore()
    store.add(_image(b"a", name="a.jpg", trigger=NOW), 1.0)
    store.add(_image(b"b", name="b.jpg", trigger=NOW), 2.0)
    store.take_presented()

    store.add(_image(b"a", name="a.jpg", trigger=NOW), 3.0)

    assert _presented(store) == b"a"


def test_new_version_without_trigger_leaves_the_display_alone():
    """Clause 6.2.2: if the displayed image is overwritten, the display is left
    alone."""
    store = SlideStore()
    store.add(_image(b"v1", name="a.jpg", trigger=NOW), 1.0)
    store.take_presented()

    store.add(_image(b"v2", name="a.jpg"), 2.0)

    assert _presented(store) is None
    assert store.live_slide().data == b"v2"


def test_header_update_with_now_shows_a_held_slide():
    """The main purpose of header updates (clause 6.1): trigger an image sent in
    advance without transmitting it again."""
    store = SlideStore()
    store.add(_image(b"a", name="a.jpg"), 1.0, now=NOON)

    store.apply_update(_update("a.jpg", trigger=NOW), now=NOON + 5)

    assert _presented(store) == b"a"


def test_header_update_with_now_shows_the_slide_only_once():
    """Table 2: "should be displayed only once unless a subsequent TriggerTime update
    of NOW is received"; not a permanent state."""
    store = SlideStore()
    store.add(_image(b"a", name="a.jpg"), 1.0, now=NOON)
    store.apply_update(_update("a.jpg", trigger=NOW), now=NOON)
    store.take_presented()

    store.tick(NOON + 10)
    assert _presented(store) is None

    store.apply_update(_update("a.jpg", trigger=NOW), now=NOON + 20)
    assert _presented(store) == b"a"


def test_header_update_schedules_a_slide():
    store = SlideStore()
    store.add(_image(b"a", name="a.jpg"), 1.0, now=NOON)

    store.apply_update(_update("a.jpg", trigger=_at(NOON + 60)), now=NOON)
    store.tick(NOON + 60)

    assert _presented(store) == b"a"


def test_header_update_moves_a_scheduled_trigger():
    store = SlideStore()
    store.add(_image(b"a", name="a.jpg", trigger=_at(NOON + 60)), 1.0, now=NOON)

    store.apply_update(_update("a.jpg", trigger=_at(NOON + 120)), now=NOON)
    store.tick(NOON + 60)
    assert _presented(store) is None

    store.tick(NOON + 120)
    assert _presented(store) == b"a"


def test_header_update_without_trigger_time_keeps_the_schedule():
    store = SlideStore()
    store.add(_image(b"a", name="a.jpg", trigger=_at(NOON + 60)), 1.0, now=NOON)

    store.apply_update(_update("a.jpg", category_slide=_in(1, 1)), now=NOON)
    store.tick(NOON + 60)

    assert _presented(store) == b"a"


def test_expired_slide_leaves_the_display_and_its_schedule():
    store = SlideStore()
    store.add(_image(b"a", name="a.jpg", trigger=NOW, expire=_at(NOON + 10)), 1.0, now=NOON)
    store.add(
        _image(b"b", name="b.jpg", trigger=_at(NOON + 30), expire=_at(NOON + 10)), 2.0, now=NOON
    )
    store.take_presented()

    store.expire(NOON + 10)
    store.tick(NOON + 30)

    assert store.live_slide() is None
    assert _presented(store) is None


def test_evicted_slide_is_not_shown():
    """Clause 5.3.2: if the image cannot be held, it is not shown."""
    store = SlideStore(max_slides=1)
    store.add(_image(b"a", name="a.jpg", trigger=_at(NOON + 30)), 1.0, now=NOON)
    store.add(_image(b"b", name="b.jpg"), 2.0, now=NOON)

    store.tick(NOON + 30)

    assert _presented(store) is None


def test_clear_forgets_schedules_and_the_live_slide():
    store = SlideStore()
    store.add(_image(b"a", name="a.jpg", trigger=NOW), 1.0)
    store.add(_image(b"b", name="b.jpg", trigger=_at(NOON + 30)), 2.0, now=NOON)
    store.clear()

    store.tick(NOON + 30)

    assert store.live_slide() is None
    assert _presented(store) is None


def test_malformed_trigger_time_counts_as_missing():
    store = SlideStore()
    slide = store.add(_image(b"a", name="a.jpg", trigger=bytes(5)), 1.0, now=NOON)

    assert _presented(store) is None
    assert store.presentation(slide) is Presentation.NO_TRIGGER


def test_tick_reports_whether_anything_changed():
    store = SlideStore()
    store.add(_image(b"a", name="a.jpg", trigger=_at(NOON + 30)), 1.0, now=None)

    assert store.tick(None) is False
    assert store.tick(NOON) is True, "the clock is there: 'awaiting' becomes 'scheduled'"
    assert store.tick(NOON + 10) is False
    assert store.tick(NOON + 30) is True
