"""Tests for the stored programme guides (`dab.guide_cache`)."""

from __future__ import annotations

import json

import pytest

from dab.guide_cache import GuideCache
from dab.mot import MotObject
from dab.spi import PARAM_SCOPE_ID, ProgrammeGuide, decode_spi_object
from tests.spi_vectors import DLF_NOVA_PI_BODY

PI = MotObject(
    0x7C93,
    7,
    1,
    "w20261003dd230c0.EHB",
    DLF_NOVA_PI_BODY,
    {PARAM_SCOPE_ID: bytes.fromhex("40e010bcd230"), 0x0D: bytes.fromhex("6aba5749")},
)
LOGO = MotObject(0x7030, 2, 3, "d210_Dlf_320x240.png", b"\x89PNG logo", {})


def _names(guide) -> list[str]:
    return sorted(o.content_name for o in guide.objects)


def test_stored_objects_come_back_in_a_new_session(tmp_path):
    cache = GuideCache(tmp_path)
    cache.store("5C", PI, now=100.0)
    cache.store("5C", LOGO, now=200.0)

    stored = GuideCache(tmp_path).load("5C")

    assert stored.updated_at == 200.0
    assert _names(stored) == ["d210_Dlf_320x240.png", "w20261003dd230c0.EHB"]
    pi = next(o for o in stored.objects if o.content_type == 7)
    assert (pi.content_subtype, pi.body, pi.parameters) == (1, PI.body, PI.parameters)
    assert decode_spi_object(pi).service.sid == 0xD230


def test_service_labels_are_stored_with_the_guide(tmp_path):
    cache = GuideCache(tmp_path)
    cache.store_labels("5C", {0xD230: "Dlf Nova"})  # no guide yet: nothing to name
    cache.store("5C", PI, now=1.0)
    cache.store_labels("5C", {0xD230: "Dlf Nova"})
    cache.store_labels("5C", {0xD210: "Dlf"})  # later labels add to the earlier ones

    assert GuideCache(tmp_path).load("5C").labels == {0xD230: "Dlf Nova", 0xD210: "Dlf"}


def test_channels_are_kept_apart(tmp_path):
    cache = GuideCache(tmp_path)
    cache.store("5C", LOGO, now=1.0)

    assert cache.load("7A") is None
    assert GuideCache(tmp_path).load("5C") is not None


def test_an_unchanged_object_is_not_written_again(tmp_path):
    cache = GuideCache(tmp_path)
    cache.store("5C", LOGO, now=1.0)
    cache.store("5C", LOGO, now=2.0)  # the next turn of the carousel

    assert cache.load("5C").updated_at == 1.0

    cache.store("5C", MotObject(1, 2, 3, LOGO.content_name, b"new version", {}), now=3.0)
    stored = GuideCache(tmp_path).load("5C")
    assert stored.updated_at == 3.0
    assert [o.body for o in stored.objects] == [b"new version"]


def test_retain_deletes_what_the_directory_no_longer_lists(tmp_path):
    cache = GuideCache(tmp_path)
    cache.store("5C", PI, now=1.0)
    cache.store("5C", LOGO, now=1.0)

    cache.retain("5C", {LOGO.content_name}, now=5.0)

    stored = GuideCache(tmp_path).load("5C")
    assert _names(stored) == [LOGO.content_name]
    assert stored.updated_at == 5.0
    assert len(list((tmp_path / "5C").glob("*.bin"))) == 1


def test_remove_deletes_the_channel(tmp_path):
    cache = GuideCache(tmp_path)
    cache.store("5C", LOGO, now=1.0)

    cache.remove("5C")

    assert not (tmp_path / "5C").exists()
    assert cache.load("5C") is None
    cache.remove("7A")  # nothing stored: no error


@pytest.mark.parametrize("content", ["not json", json.dumps({"version": 99, "objects": {}})])
def test_an_unreadable_index_counts_as_no_guide(tmp_path, content):
    (tmp_path / "5C").mkdir()
    (tmp_path / "5C" / "index.json").write_text(content, encoding="utf-8")

    assert GuideCache(tmp_path).load("5C") is None


def test_an_object_whose_file_is_missing_is_skipped(tmp_path):
    cache = GuideCache(tmp_path)
    cache.store("5C", PI, now=1.0)
    cache.store("5C", LOGO, now=1.0)
    for path in (tmp_path / "5C").glob("*.bin"):
        if path.read_bytes() == LOGO.body:
            path.unlink()

    assert _names(GuideCache(tmp_path).load("5C")) == [PI.content_name]


def test_channel_names_cannot_leave_the_directory(tmp_path):
    with pytest.raises(ValueError, match="Invalid channel"):
        GuideCache(tmp_path).store("../x", LOGO, now=1.0)


def test_guide_drops_schedules_that_ended_long_ago():
    guide = ProgrammeGuide()
    guide.add(PI, decode_spi_object(PI))  # 3 October 2026, ends 22:00 UTC
    guide.add(LOGO, None)
    end = decode_spi_object(PI).stop

    assert guide.drop_schedules_ended_before(end) is False
    assert guide.drop_schedules_ended_before(end + 1) is True
    assert guide.content_names() == {LOGO.content_name}


def test_channels_lists_the_channels_with_a_stored_guide(tmp_path):
    assert GuideCache(tmp_path / "missing").channels() == []

    cache = GuideCache(tmp_path)
    cache.store("7A", LOGO, now=1.0)
    cache.store("5C", PI, now=1.0)
    (tmp_path / "leftover").mkdir()  # no index: no guide

    assert GuideCache(tmp_path).channels() == ["5C", "7A"]
