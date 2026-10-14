"""The language of the surfaces: English unless German (or another catalog of `i18n`) was
chosen, on the desktop in the menu "View › Language", on the touch surface on the settings
page "Display". One setting for both (`interface/language`).

The widgets set their texts once, when they are built; a new language therefore takes
effect when the radio starts anew, which both surfaces offer right away, asked in the
language chosen (`restart_question`, `restart_command`)."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

from PyQt6.QtCore import QLibraryInfo, QLocale, QSettings, QTranslator
from PyQt6.QtWidgets import QApplication

from i18n import (
    DEFAULT_LANGUAGE,
    SOURCE_LANGUAGE,
    available_languages,
    language,
    set_language,
    tr_into,
)

logger = logging.getLogger(__name__)

SETTINGS_LANGUAGE = "interface/language"


def stored_language(settings: QSettings) -> str:
    """The language chosen, English where none was or its catalog is gone."""
    chosen = str(settings.value(SETTINGS_LANGUAGE, DEFAULT_LANGUAGE))
    return chosen if chosen in available_languages() else DEFAULT_LANGUAGE


def store_language(settings: QSettings, chosen: str) -> None:
    settings.setValue(SETTINGS_LANGUAGE, chosen)


def configure_language(app: QApplication, settings: QSettings) -> str:
    """Sets the language chosen, before any widget is built, for the radio's texts and Qt's
    own (the buttons of its message boxes and the like), and the numbers of Qt's widgets:
    the spin boxes wrote a decimal comma in the English desktop on a German Windows, as the
    system's locale has it. Returns the language."""
    chosen = set_language(stored_language(settings))
    QLocale.setDefault(QLocale(chosen))
    if chosen != SOURCE_LANGUAGE:
        translator = QTranslator(app)
        directory = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
        if translator.load(f"qtbase_{chosen}", directory):
            app.installTranslator(translator)
        else:
            logger.warning("No Qt translation for %s in %s", chosen, directory)
    logger.info("Language of the surface: %s", chosen)
    return chosen


@dataclass(frozen=True)
class RestartQuestion:
    """What a surface asks once a language is chosen, in that language."""

    title: str
    detail: str
    restart: str  # the button that starts the radio anew
    later: str  # the button that keeps it running in the language it has


def restart_question(chosen: str) -> RestartQuestion | None:
    """The question for `chosen`; None when it is the language running, which needs no
    new start."""
    if chosen == language():
        return None
    return RestartQuestion(
        # Translators: asked in the language just chosen; "English" is that language.
        tr_into(chosen, "Restart the radio in English?"),
        tr_into(
            chosen,
            "The new language takes effect when the radio starts again. Reception "
            "pauses for a few seconds.",
        ),
        tr_into(chosen, "Restart now"),
        tr_into(chosen, "Later"),
    )


def restart_command() -> list[str]:
    """The command line this radio was started with, to start it anew: the same
    interpreter, its options, the script and the radio's arguments."""
    return [sys.executable, *sys.orig_argv[1:]]
