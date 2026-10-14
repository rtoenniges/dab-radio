"""The translation of the interface: the catalogs against the texts in the code, and
switching the language."""

from __future__ import annotations

import ast
from string import Formatter

import pytest

import i18n
from i18n import (
    Translated,
    exception_text,
    join,
    set_language,
    tr,
    tr_context,
    tr_into,
    tr_plural,
)
from i18n.update_catalogs import (
    SOURCE_DIRECTORY,
    catalog_paths,
    extract_template,
    mo_bytes,
    po_bytes,
    read_catalog,
    updated,
)

# The marking functions and the position of the text among their arguments.
_TEXT_ARGUMENTS = {
    "tr": (0,),
    "tr_plural": (0, 1),
    "tr_context": (0, 1),
    "tr_noop": (0,),
    "tr_into": (1,),
}
# The functions that translate when called, and so must not run at import.
_TRANSLATING = {"tr", "tr_plural", "tr_context"}


@pytest.fixture
def english():
    """Back to English after the test, as the other tests expect."""
    yield
    set_language(i18n.SOURCE_LANGUAGE)


def _key(message) -> tuple:
    return (message.id, message.context)


def test_every_catalog_has_each_text_of_the_code_translated():
    template = extract_template()
    wanted = {_key(m) for m in template if m.id}
    paths = catalog_paths()
    assert paths
    for path in paths:
        catalog = read_catalog(path)
        present = {_key(m) for m in catalog if m.id}
        assert wanted - present == set(), f"{path.name}: run src/i18n/update_catalogs.py"
        assert present - wanted == set(), f"{path.name}: run src/i18n/update_catalogs.py"
        for message in catalog:
            if not message.id:
                continue
            strings = message.string if message.pluralizable else (message.string,)
            assert all(strings), f"{path.name}: untranslated {message.id!r}"
            assert not message.fuzzy, f"{path.name}: fuzzy {message.id!r}, check and unflag"


def test_every_catalog_and_its_compiled_form_are_current():
    template = extract_template()
    for path in catalog_paths():
        catalog = updated(read_catalog(path), template)
        assert path.read_bytes() == po_bytes(catalog), "run src/i18n/update_catalogs.py"
        mo_path = path.with_suffix(".mo")
        assert mo_path.read_bytes() == mo_bytes(catalog), "run src/i18n/update_catalogs.py"


def _fields(text: str) -> set[str]:
    return {name for _, name, _, _ in Formatter().parse(text) if name is not None}


# Texts whose code fills more placeholders than the English uses, so that a language can
# write them its own way: the fields filled.
_MORE_FIELDS = {"{day} {month_name}": {"day", "month", "month_name"}}


def test_translations_use_only_the_placeholders_of_the_code():
    # A placeholder the code does not fill would raise KeyError in `format`.
    for path in catalog_paths():
        for message in read_catalog(path):
            if not message.id:
                continue
            ids = message.id if message.pluralizable else (message.id,)
            strings = message.string if message.pluralizable else (message.string,)
            allowed = set().union(*(_fields(text) for text in ids))
            allowed |= _MORE_FIELDS.get(message.id, set())
            for string in strings:
                assert _fields(string) <= allowed, (path.name, message.id, string)


def test_translations_keep_the_placeholders_of_the_code():
    # A value left out by mistake would go unnoticed; plural forms may leave the count out
    # ("one title"), and the texts above choose among their fields.
    for path in catalog_paths():
        for message in read_catalog(path):
            if not message.id or message.pluralizable or message.id in _MORE_FIELDS:
                continue
            assert _fields(message.string) == _fields(message.id), (path.name, message.id)


def _written_out(argument: ast.expr | None, function: str) -> bool:
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return True
    # `tr` also takes an entry of a table marked with `tr_noop`, by name, index or attribute.
    return function == "tr" and isinstance(argument, (ast.Name, ast.Subscript, ast.Attribute))


def _source_trees():
    for path in sorted(SOURCE_DIRECTORY.rglob("*.py")):
        yield path, ast.parse(path.read_text(encoding="utf-8"))


def test_marked_texts_are_written_out_in_the_code():
    # The catalog looks a text up as written: an f-string, a `format` or a `+` inside `tr`
    # would never be found, and the extraction passes over it without a word.
    wrong = []
    for path, tree in _source_trees():
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            for index in _TEXT_ARGUMENTS.get(node.func.id, ()):
                argument = node.args[index] if index < len(node.args) else None
                if not _written_out(argument, node.func.id):
                    wrong.append(f"{path.relative_to(SOURCE_DIRECTORY)}:{node.lineno}")
    assert wrong == []


def _calls_at_import(node: ast.AST, found: list[int]) -> None:
    """Line numbers of translating calls under `node` that run when the module is
    imported: in the module and class bodies, decorators and default values, but not in
    the bodies of functions and lambdas."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _TRANSLATING
    ):
        found.append(node.lineno)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for child in [*node.decorator_list, *node.args.defaults, *node.args.kw_defaults]:
            if child is not None:
                _calls_at_import(child, found)
        return
    if isinstance(node, ast.Lambda):
        for child in [*node.args.defaults, *node.args.kw_defaults]:
            if child is not None:
                _calls_at_import(child, found)
        return
    for child in ast.iter_child_nodes(node):
        _calls_at_import(child, found)


def test_no_text_is_translated_at_import():
    # The language is set after the modules are imported; a table translated at import
    # would stay English. Tables mark their texts with `tr_noop` and translate on showing.
    wrong = []
    for path, tree in _source_trees():
        found: list[int] = []
        _calls_at_import(tree, found)
        wrong += [f"{path.relative_to(SOURCE_DIRECTORY)}:{line}" for line in found]
    assert wrong == []


def test_each_language_is_listed_by_its_own_name():
    languages = i18n.available_languages()
    assert next(iter(languages)) == "en"  # the language of the code first
    assert {"en": "English", "de": "Deutsch"}.items() <= languages.items()


def test_the_language_set_translates_from_then_on(english):
    assert i18n.language() == "en" and tr("Settings") == "Settings"
    assert set_language("de") == "de"
    assert i18n.language() == "de"
    assert tr("Settings") == "Einstellungen"
    assert tr("a text no catalog has") == "a text no catalog has"
    assert tr_context("language name", "English") == "Deutsch"
    assert tr_into("en", "Language") == "Language"
    assert set_language("xx") == "en"  # no catalog
    assert tr("Settings") == "Settings"
    assert tr_into("de", "Language") == "Sprache"
    assert set_language("../de") == "en"  # not a file name


def test_plural_forms_follow_the_count(english):
    set_language("de")
    assert tr_plural("{count} title", "{count} titles", 1) == "{count} Titel"
    assert tr_plural("{count} title", "{count} titles", 2) == "{count} Titel"


def test_in_english_the_texts_are_plain_strings():
    text = tr("Error: {message}").format(message=tr("Stopped."))
    assert type(text) is str and i18n.english(text) == "Error: Stopped."


def test_a_translation_keeps_its_english_through_format_plus_and_join(english):
    # For the remote interfaces, which stay English whatever the surfaces show.
    set_language("de")
    text = tr("Error: {message}").format(message=tr("Stopped."))
    assert text == "Fehler: Gestoppt." and i18n.english(text) == "Error: Stopped."
    titles = tr_plural("{count} title", "{count} titles", 3).format(count=3)
    assert i18n.english(titles) == "3 titles"
    joined = join(", ", [tr("Stopped."), "Dlf"]) + " - " + tr("Settings")
    assert joined == "Gestoppt., Dlf - Einstellungen"
    assert i18n.english(joined) == "Stopped., Dlf - Settings"
    assert i18n.english(tr_context("language name", "English")) == "English"
    # A text translated twice stays what it was.
    assert i18n.english(tr(tr("Stopped."))) == "Stopped."
    # Anything that is no translation is its own English.
    assert i18n.english("Dlf Kultur") == "Dlf Kultur" and i18n.english(5) == 5


def test_an_exception_keeps_the_english_of_its_message(english):
    set_language("de")
    error = RuntimeError(tr("Stopped."))
    assert isinstance(exception_text(error), Translated)
    assert i18n.english(error) == "Stopped."
    assert i18n.english(tr("Error: {message}").format(message=error)) == "Error: Stopped."
    assert exception_text(RuntimeError("a", 1)) == "('a', 1)"
