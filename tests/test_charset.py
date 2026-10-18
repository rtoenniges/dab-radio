"""Tests for the character sets (ETSI TS 101 756, table 1 and Annex C)."""

from __future__ import annotations

import unicodedata

import pytest

from dab.charset import EBU_LATIN, decode_charset, decode_ebu_latin
from dab.fig import decode_label


def test_umlauts_as_received():
    """Received exactly like this: Dlf and KLASSIK RADIO with 0x99 for ü,
    SCHLAGERPARADIES with 0xD9 for Ü, character set 0."""
    assert (
        decode_charset(b"Klassische Musik zum Wohlf\x99hlen und Entspannen", 0)
        == "Klassische Musik zum Wohlfühlen und Entspannen"
    )
    assert decode_charset(b"B\x99chermarkt mit Nora Karches", 0) == "Büchermarkt mit Nora Karches"
    assert (
        decode_charset(b"SCHLAGERPARADE MIT JOACHIM M\xd9LLER BIS 19 UHR", 0)
        == "SCHLAGERPARADE MIT JOACHIM MÜLLER BIS 19 UHR"
    )


def test_german_special_characters():
    assert decode_ebu_latin(bytes([0x91, 0x97, 0x99, 0xD1, 0xD7, 0xD9, 0x8D])) == "äöüÄÖÜß"


def test_currency_signs():
    assert decode_ebu_latin(bytes([0xA9, 0xAA, 0xAB])) == "€£$"


def test_differences_to_ascii():
    """EBU Latin is not a superset of ASCII: these positions hold other characters
    (table C.1)."""
    assert decode_ebu_latin(bytes([0x24, 0x5C, 0x5E, 0x60])) == "łŮŁĄ"
    assert decode_ebu_latin(bytes([0x7B, 0x7C, 0x7D, 0x7E, 0x7F])) == "«ů»ĽĦ"


def test_ascii_letters_digits_and_punctuation_are_unchanged():
    text = "Radio 21: Rock & Pop (100%) - jetzt!"
    assert decode_ebu_latin(text.encode("ascii")) == text


def test_reserved_codes_are_not_displayed():
    """Annex C: 0x00, 0x0A, 0x0B and 0x1F "are not displayed"."""
    assert decode_ebu_latin(b"A\x00B\x0aC\x0bD\x1fE") == "ABCDE"


def test_every_other_code_is_a_single_printable_character():
    for code, character in enumerate(EBU_LATIN):
        if code in (0x00, 0x0A, 0x0B, 0x1F):
            continue
        assert len(character) == 1, hex(code)
        assert unicodedata.category(character)[0] in "LNPSZ", hex(code)


def test_table_has_no_duplicates():
    characters = [character for character in EBU_LATIN if character]
    assert len(characters) == len(set(characters)) == 252


def test_utf8_and_utf16():
    assert decode_charset("Grüße".encode(), 15) == "Grüße"
    assert decode_charset("Grüße".encode("utf-16-be"), 6) == "Grüße"


@pytest.mark.parametrize("charset", [1, 4, 14])
def test_unregistered_charsets_are_not_guessed(charset):
    assert decode_charset(b"Radio \x99", charset) == "Radio ?"


def test_service_label_uses_the_table():
    assert decode_label(b"Radio K\x99ste    ", 0) == "Radio Küste"


def test_label_padding_is_removed():
    assert decode_label("Café".encode().ljust(16, b"\x00"), 15) == "Café"
    assert decode_label(b"Test Radio      ", 0) == "Test Radio"
