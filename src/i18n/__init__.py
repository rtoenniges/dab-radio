"""Translation of the user interface.

The code carries the English texts; a gettext catalog per further language translates
them (`<code>.po` beside this module, compiled to `<code>.mo` by `update_catalogs.py`).
English needs no catalog, and a text missing from a catalog stays English. The language is
set once at start (`set_language`), before the window is built (`ui.language`, one
setting for both surfaces: the desktop's menu "View › Language", the touch settings page
"Display"). A change takes effect at the next start, which both surfaces offer.

Every text the user sees is marked, so that `update_catalogs.py` finds it:
- `tr("…")` where it is used; with values `tr("{count} stations").format(count=n)`. Never
  an f-string, `format` or `+` inside `tr`: the catalog looks a text up as it is written
  in the code.
- `tr_plural("{count} minute", "{count} minutes", n).format(count=n)` where a number
  decides the form; the catalog has as many forms as its language needs.
- `tr_context("context", "…")` where one English text needs two translations, the
  context telling them apart ("Off" of a switch, "Off" of a screen).
- `tr_noop("…")` in tables and constants built at import, before the language is set;
  `tr(table[key])` where the entry is shown. A `tr` that runs at import would stay English
  (`tests/test_i18n.py` finds one).
A comment beginning with `Translators:` right above the call goes into the catalog. Logs
and exceptions meant for the log stay English and unmarked.

The remote interfaces (MQTT, `net.radio_state`) speak English whatever the surfaces show.
In another language than English, `tr` and its siblings therefore return a `Translated`:
the translation, which keeps the English of the code beside it, also through `format`
and `+` (and `join` here); `english(text)` gives it back. Texts that pass between threads
on their way there travel in signals of type `object`, as a `str` signal would leave a
plain copy.
"""

from __future__ import annotations

import gettext
import time
from pathlib import Path
from typing import Self

CATALOG_DIRECTORY = Path(__file__).parent
DOMAIN = "dab-radio"
# The language of the texts in the code, which needs no catalog.
SOURCE_LANGUAGE = "en"
DEFAULT_LANGUAGE = SOURCE_LANGUAGE

_translations: gettext.NullTranslations = gettext.NullTranslations()
_language = SOURCE_LANGUAGE


class Translated(str):
    """A text in the language set that keeps its English (`english`). Formatting and `+`
    keep both; anything else (an f-string, `str.join`, `%`, `replace`) gives a plain `str`,
    whose English is then the text itself."""

    english: str

    def __new__(cls, text: str, english_text: str) -> Self:
        translated = super().__new__(cls, text)
        translated.english = english_text
        return translated

    def format(self, *args: object, **kwargs: object) -> Translated:
        return Translated(
            str.format(self, *args, **kwargs),
            self.english.format(
                *(english(value) for value in args),
                **{key: english(value) for key, value in kwargs.items()},
            ),
        )

    def __add__(self, other: object) -> Translated:
        if not isinstance(other, str):
            return NotImplemented
        return Translated(str.__add__(self, other), self.english + english(other))

    def __radd__(self, other: object) -> Translated:
        if not isinstance(other, str):
            return NotImplemented
        return Translated(str.__add__(other, self), english(other) + self.english)


def english(value: object) -> object:
    """The English of a text from `tr` and its siblings, for the interfaces that stay
    English; an exception by the English of its message; anything else as it is."""
    if isinstance(value, Translated):
        return value.english
    if isinstance(value, BaseException):
        message = exception_text(value)
        return message.english if isinstance(message, Translated) else message
    return value


def exception_text(exc: BaseException) -> str:
    """The message of `exc` as it was raised: a `Translated` keeps its English, where
    `str(exc)` would leave a plain copy."""
    if len(exc.args) == 1 and isinstance(exc.args[0], str):
        return exc.args[0]
    return str(exc)


def join(separator: str, texts: list[str]) -> str:
    """`separator.join(texts)`, keeping the English of translated texts."""
    joined = separator.join(texts)
    if not any(isinstance(text, Translated) for text in texts):
        return joined
    return Translated(joined, english(separator).join(english(text) for text in texts))


def available_languages() -> dict[str, str]:
    """Code → name of the language in itself ("Deutsch"): English, then every catalog."""
    languages = {SOURCE_LANGUAGE: _language_name(gettext.NullTranslations())}
    for path in sorted(CATALOG_DIRECTORY.glob("*.mo")):
        languages[path.stem] = _language_name(_load(path.stem))
    return languages


def set_language(language: str) -> str:
    """Translates into `language` from now on; into English if it has no catalog. Returns
    the language set."""
    global _translations, _language
    if language != SOURCE_LANGUAGE and language in available_languages():
        _translations, _language = _load(language), language
    else:
        _translations, _language = gettext.NullTranslations(), SOURCE_LANGUAGE
    return _language


def language() -> str:
    return _language


def tr(text: str) -> str:
    """`text` in the language set."""
    if _language == SOURCE_LANGUAGE or isinstance(text, Translated):
        return text
    return Translated(_translations.gettext(text), text)


def tr_plural(singular: str, plural: str, count: int) -> str:
    """The form of `singular`/`plural` for `count` in the language set."""
    if _language == SOURCE_LANGUAGE:
        return singular if count == 1 else plural
    return Translated(
        _translations.ngettext(singular, plural, count), singular if count == 1 else plural
    )


def tr_context(context: str, text: str) -> str:
    """`text` in the language set, as meant in `context`."""
    if _language == SOURCE_LANGUAGE:
        return text
    return Translated(_translations.pgettext(context, text), text)


def tr_noop(text: str) -> str:
    """Marks `text` for the catalog without translating it, for tables built at import."""
    return text


def tr_into(language: str, text: str) -> str:
    """`text` in another language than the one set: to say something in the language
    just chosen."""
    if language != SOURCE_LANGUAGE and language in available_languages():
        return _load(language).gettext(text)
    return text


WEEKDAYS = (
    tr_noop("Monday"),
    tr_noop("Tuesday"),
    tr_noop("Wednesday"),
    tr_noop("Thursday"),
    tr_noop("Friday"),
    tr_noop("Saturday"),
    tr_noop("Sunday"),
)
# Translators: the weekdays shortened, as the programme guide's row of days shows them.
WEEKDAYS_SHORT = (
    tr_noop("Mon"),
    tr_noop("Tue"),
    tr_noop("Wed"),
    tr_noop("Thu"),
    tr_noop("Fri"),
    tr_noop("Sat"),
    tr_noop("Sun"),
)
# Translators: the months shortened, for a day and month like "6 Oct"; unused where the
# language writes the month as a number.
MONTHS_SHORT = (
    tr_noop("Jan"),
    tr_noop("Feb"),
    tr_noop("Mar"),
    tr_noop("Apr"),
    tr_noop("May"),
    tr_noop("Jun"),
    tr_noop("Jul"),
    tr_noop("Aug"),
    tr_noop("Sep"),
    tr_noop("Oct"),
    tr_noop("Nov"),
    tr_noop("Dec"),
)


def weekday(moment: time.struct_time, short: bool = False) -> str:
    """The day of the week of `moment`: "Tuesday", shortened "Tue"."""
    return tr((WEEKDAYS_SHORT if short else WEEKDAYS)[moment.tm_wday])


def format_date(moment: time.struct_time) -> str:
    """The date of `moment` as the language writes it: 2026-10-06, 06.10.2026."""
    # Translators: a date; German writes {day:02d}.{month:02d}.{year}.
    return tr("{year}-{month:02d}-{day:02d}").format(
        year=moment.tm_year, month=moment.tm_mon, day=moment.tm_mday
    )


def format_day_month(moment: time.struct_time) -> str:
    """Day and month of `moment` without the year: "6 Oct", "6.10."."""
    # Translators: day and month without the year; {month} is the number, {month_name}
    # the shortened name. A translation may leave either out: German writes {day}.{month}.
    return tr("{day} {month_name}").format(
        day=moment.tm_mday, month=moment.tm_mon, month_name=tr(MONTHS_SHORT[moment.tm_mon - 1])
    )


def _load(language: str) -> gettext.GNUTranslations:
    with open(CATALOG_DIRECTORY / f"{language}.mo", "rb") as file:
        return gettext.GNUTranslations(file)


def _language_name(translations: gettext.NullTranslations) -> str:
    # Translators: the name of your language in itself, as the language setting lists it:
    # Deutsch, Français, Español.
    return translations.pgettext("language name", "English")
