"""CHANGELOG.md describes the version of the radio, and `tools/release_notes.py` takes
the notes of a release from it as the release workflow does."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ui.receiver_controller import APPLICATION_VERSION

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import release_notes

CHANGELOG = """\
# Changelog

Intro.

## [Unreleased]

## [1.1.0] - 2026-10-15

### Added

- Something new.

## [1.0.0] - 2026-10-14

The first release.

[Unreleased]: https://example.org/compare/v1.1.0...HEAD
[1.1.0]: https://example.org/compare/v1.0.0...v1.1.0
[1.0.0]: https://example.org/releases/tag/v1.0.0
"""


def test_each_section_ends_at_the_next_or_at_the_links():
    entries = release_notes.entries(CHANGELOG)
    assert list(entries) == ["Unreleased", "1.1.0", "1.0.0"]
    assert entries["Unreleased"].text == ""
    assert entries["1.1.0"] == release_notes.Entry(
        "1.1.0", "2026-10-15", "### Added\n\n- Something new."
    )
    assert entries["1.0.0"].text == "The first release."


def test_the_notes_end_with_the_comparison_with_the_version_before():
    assert release_notes.release_notes(CHANGELOG, "v1.1.0") == (
        "### Added\n\n- Something new.\n\n"
        "**Full changelog**: https://example.org/compare/v1.0.0...v1.1.0\n"
    )
    assert release_notes.release_notes(CHANGELOG, "1.0.0") == "The first release.\n"


def test_wrapped_paragraphs_and_list_items_become_one_line_each():
    text = (
        "A paragraph\nwrapped once.\n\n### Fixed\n\n- An item\n  wrapped once.\n"
        "  - A nested item\n    wrapped.\n- Another.\n\n```sh\nsudo sh\npi/install.sh\n```\n"
        "After the code."
    )
    assert release_notes.unwrap(text) == (
        "A paragraph wrapped once.\n\n### Fixed\n\n- An item wrapped once.\n"
        "  - A nested item wrapped.\n- Another.\n\n```sh\nsudo sh\npi/install.sh\n```\n"
        "After the code."
    )


@pytest.mark.parametrize("version", ["v1.2.0", "Unreleased"])
def test_a_version_without_notes_is_an_error(version):
    with pytest.raises(ValueError):
        release_notes.release_notes(CHANGELOG, version)


def test_the_changelog_describes_the_version_of_the_radio():
    changelog = release_notes.CHANGELOG.read_text(encoding="utf-8")
    entry = release_notes.entries(changelog)[APPLICATION_VERSION]
    assert entry.date is not None
    assert release_notes.release_notes(changelog, APPLICATION_VERSION)
    assert APPLICATION_VERSION in release_notes.links(changelog)
    assert f"/compare/v{APPLICATION_VERSION}...HEAD" in release_notes.links(changelog)["Unreleased"]
