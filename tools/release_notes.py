"""Prints the notes of one version from CHANGELOG.md, for its release on GitHub.

The release workflow (`.github/workflows/release.yml`) runs it with the tag:

    python3 tools/release_notes.py v1.3.1

The notes are the section of the version without its heading, followed by the link that
compares the version with the one before, as the link references at the end of the
changelog give it. A version without a section, or with an empty one, is an error, so
that a tag pushed without its entry fails instead of publishing an empty release.

GitHub shows a release like a comment, where every line break of the text is one on the
page; the lines of the changelog are wrapped for an editor, so each paragraph and list
item becomes one line again.

Runs on the Python of the GitHub runner, so it uses nothing newer than Python 3.12.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"

HEADING = re.compile(r"## \[(?P<name>[^\]]+)\](?: - (?P<date>\d{4}-\d{2}-\d{2}))?\s*$")
LINK = re.compile(r"\[(?P<name>[^\]]+)\]: (?P<url>\S+)\s*$")
# A line that begins a block of its own rather than continuing the one before.
BLOCK_START = re.compile(r"#|[-*+] |\d+\. |>|\||```")


@dataclass(frozen=True)
class Entry:
    """One section of the changelog: a version, or "Unreleased"."""

    name: str
    date: str | None
    text: str


def entries(changelog: str) -> dict[str, Entry]:
    """The sections of the changelog by name, each up to the next one or the links."""
    found: dict[str, Entry] = {}
    heading = None
    lines: list[str] = []

    def close() -> None:
        if heading is not None:
            name = heading["name"]
            found[name] = Entry(name, heading["date"], "\n".join(lines).strip())

    for line in changelog.splitlines():
        if match := HEADING.match(line):
            close()
            heading, lines = match, []
        elif LINK.match(line):
            close()
            heading = None
        elif heading is not None:
            lines.append(line)
    close()
    return found


def links(changelog: str) -> dict[str, str]:
    """The link references at the end of the changelog, by the name of the section."""
    return {
        match["name"]: match["url"]
        for line in changelog.splitlines()
        if (match := LINK.match(line))
    }


def unwrap(text: str) -> str:
    """Joins the wrapped lines of each paragraph and list item; code blocks stay."""
    lines: list[str] = []
    fenced = False
    for line in text.splitlines():
        stripped = line.strip()
        previous = lines[-1].strip() if lines else ""
        continues = (
            not fenced
            and stripped
            and previous
            and not BLOCK_START.match(stripped)
            and not previous.startswith(("#", "|", "```"))
        )
        if continues:
            lines[-1] = f"{lines[-1].rstrip()} {stripped}"
        else:
            lines.append(line)
        if stripped.startswith("```"):
            fenced = not fenced
    return "\n".join(lines)


def release_notes(changelog: str, version: str) -> str:
    """The notes of `version` (with or without the "v" of the tag) for its release."""
    version = version.removeprefix("v")
    entry = entries(changelog).get(version)
    if entry is None:
        raise ValueError(f"CHANGELOG.md has no section for {version}")
    if not entry.text:
        raise ValueError(f"the section for {version} in CHANGELOG.md is empty")
    text = unwrap(entry.text)
    url = links(changelog).get(version, "")
    if "/compare/" in url:
        return f"{text}\n\n**Full changelog**: {url}\n"
    return f"{text}\n"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: release_notes.py <version or tag>", file=sys.stderr)
        return 2
    try:
        notes = release_notes(CHANGELOG.read_text(encoding="utf-8"), argv[1])
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
