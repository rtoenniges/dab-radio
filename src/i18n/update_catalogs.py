"""Brings the catalogs up to date with the texts in the code.

    uv run python src/i18n/update_catalogs.py          # every catalog
    uv run python src/i18n/update_catalogs.py --new fr # start a catalog for French

Collects every marked text in `src` (see `i18n`), adds new texts to each `<code>.po`
untranslated, drops texts no longer in the code, marks a translation whose text changed
slightly as fuzzy (to be checked, then the flag removed), and compiles each catalog into
`<code>.mo`, which the radio loads. Untranslated and fuzzy entries stay English in the
radio; `tests/test_i18n.py` fails on them. Needs Babel (development group).
"""

from __future__ import annotations

import argparse
import io
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from babel.messages.catalog import Catalog
from babel.messages.extract import extract_from_dir
from babel.messages.mofile import write_mo
from babel.messages.pofile import read_po, write_po

CATALOG_DIRECTORY = Path(__file__).parent
SOURCE_DIRECTORY = CATALOG_DIRECTORY.parent
PROJECT = "dab-radio"
VERSION = tomllib.loads((SOURCE_DIRECTORY.parent / "pyproject.toml").read_text("utf-8"))["project"][
    "version"
]
# The marking functions of `i18n` and where their texts are: the argument positions,
# (n, "c") for a context.
KEYWORDS = {
    "tr": None,
    "tr_plural": (1, 2),
    "tr_context": ((1, "c"), 2),
    "tr_noop": None,
    "tr_into": (2,),
    "pgettext": ((1, "c"), 2),
}
COMMENT_TAGS = ["Translators:"]
# Line width of the catalogs, as Babel's default.
PO_WIDTH = 76


def extract_template() -> Catalog:
    """Every marked text in the code, with the files it is in."""
    template = Catalog(project=PROJECT, fuzzy=False)
    found = extract_from_dir(
        SOURCE_DIRECTORY,
        method_map=[("**.py", "python")],
        keywords=KEYWORDS,
        comment_tags=COMMENT_TAGS,
        strip_comment_tags=True,
    )
    for filename, _lineno, message, comments, context in found:
        template.add(
            message,
            locations=[(f"src/{Path(filename).as_posix()}", None)],
            auto_comments=comments,
            context=context,
        )
    return template


def catalog_paths() -> list[Path]:
    return sorted(CATALOG_DIRECTORY.glob("*.po"))


def read_catalog(path: Path) -> Catalog:
    with open(path, "rb") as file:
        return read_po(file, locale=path.stem, domain=PROJECT)


def updated(catalog: Catalog, template: Catalog) -> Catalog:
    """`catalog` with the texts of `template`; its dates stay, so that a run without
    changes leaves the files as they are."""
    catalog.update(template, update_creation_date=False)
    catalog.version = VERSION
    return catalog


def po_bytes(catalog: Catalog) -> bytes:
    buffer = io.BytesIO()
    write_po(buffer, catalog, width=PO_WIDTH, ignore_obsolete=True, include_lineno=False)
    return buffer.getvalue()


def mo_bytes(catalog: Catalog) -> bytes:
    buffer = io.BytesIO()
    write_mo(buffer, catalog)  # without fuzzy entries
    return buffer.getvalue()


def new_catalog(language: str, template: Catalog) -> Catalog:
    now = datetime.now(UTC)
    catalog = Catalog(
        locale=language,
        domain=PROJECT,
        header_comment=(
            f"# Translation of the DAB+ radio's interface ({language}).\n"
            "# Brought up to date by src/i18n/update_catalogs.py."
        ),
        project=PROJECT,
        creation_date=now,
        revision_date=now,
        fuzzy=False,
    )
    return updated(catalog, template)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--new", metavar="CODE", help="start a catalog for this language")
    args = parser.parse_args()
    template = extract_template()
    if args.new:
        path = CATALOG_DIRECTORY / f"{args.new}.po"
        if path.exists():
            parser.error(f"{path.name} exists already")
        path.write_bytes(po_bytes(new_catalog(args.new, template)))
    for path in catalog_paths():
        catalog = updated(read_catalog(path), template)
        path.write_bytes(po_bytes(catalog))
        path.with_suffix(".mo").write_bytes(mo_bytes(catalog))
        missing = sum(1 for m in catalog if m.id and (not _translated(m) or m.fuzzy))
        print(f"{path.name}: {len(catalog)} texts, {missing} to translate or check")


def _translated(message) -> bool:
    strings = message.string if isinstance(message.string, tuple) else (message.string,)
    return all(strings)


if __name__ == "__main__":
    main()
