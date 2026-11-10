"""Tests for app.tts_text_normalization."""

from __future__ import annotations

import pytest

from app.tts_text_normalization import normalize_tts_text


class TestGermanAbbreviations:
    def test_zum_beispiel(self):
        assert normalize_tts_text("Das ist z.B. ein Test.", lang="de") == "Das ist zum Beispiel ein Test."

    def test_zb_no_dot(self):
        assert normalize_tts_text("Das ist zb ein Test.", lang="de") == "Das ist zum Beispiel ein Test."

    def test_unter_anderm(self):
        assert normalize_tts_text("Dabei sind u.a. die folgenden.", lang="de") == "Dabei sind unter anderem die folgenden."

    def test_das_heisst(self):
        assert normalize_tts_text("Das ist d.h. ein Beispiel.", lang="de") == "Das ist das heißt ein Beispiel."

    def test_etc(self):
        assert normalize_tts_text("Äpfel, Birnen, etc.", lang="de") == "Äpfel, Birnen, et cetera"

    def test_usw(self):
        assert normalize_tts_text("Und so weiter usw.", lang="de") == "Und so weiter und so weiter"

    def test_ca(self):
        assert normalize_tts_text("Es sind ca. 50 Leute da.", lang="de") == "Es sind circa fünfzig Leute da."

    def test_bzw(self):
        assert normalize_tts_text("Das ist bzw. war so.", lang="de") == "Das ist beziehungsweise war so."

    def test_inkl_exkl(self):
        assert normalize_tts_text("Inkl. Frühstück, exkl. Steuern.", lang="de") == "inklusive Frühstück, exklusive Steuern."

    def test_vgl(self):
        assert normalize_tts_text("Vgl. Seite 10.", lang="de") == "vergleiche Seite zehn."


class TestEnglishAbbreviations:
    def test_ie(self):
        assert normalize_tts_text("Use i.e. for examples.", lang="en") == "Use that is for examples."

    def test_eg(self):
        assert normalize_tts_text("Fruits e.g. apples.", lang="en") == "Fruits for example apples."

    def test_dr(self):
        assert normalize_tts_text("Dr. Smith is here.", lang="en") == "Doctor Smith is here."

    def test_vs(self):
        assert normalize_tts_text("A vs. B", lang="en") == "A versus B"

    def test_am_pm(self):
        assert normalize_tts_text("The meeting is at 3 p.m.", lang="en") == "The meeting is at three post meridiem"


class TestUnits:
    def test_celsius(self):
        assert normalize_tts_text("Es sind 20 °C.", lang="de") == "Es sind zwanzig Grad Celsius."

    def test_celsius_no_space(self):
        assert normalize_tts_text("Es sind 20°C.", lang="de") == "Es sind zwanzig Grad Celsius."

    def test_fahrenheit(self):
        assert normalize_tts_text("It is 68 °F.", lang="en") == "It is sixty-eight Grad Fahrenheit."

    def test_percent_german(self):
        assert normalize_tts_text("50% Rabatt", lang="de") == "fünfzig Prozent Rabatt"

    def test_percent_english(self):
        assert normalize_tts_text("50% off", lang="en") == "fifty percent off"

    def test_euro(self):
        assert normalize_tts_text("10 €", lang="de") == "zehn Euro"

    def test_degree_standalone(self):
        assert normalize_tts_text("20°", lang="de") == "zwanzig Grad"


class TestNumbers:
    def test_german_decimal_comma(self):
        result = normalize_tts_text("Es sind 3,5 Meter.", lang="de")
        # num2words produces "drei Komma fünf" for German decimals
        assert "drei" in result
        assert "Komma" in result
        assert "fünf" in result
        assert "3,5" not in result

    def test_english_decimal_point(self):
        result = normalize_tts_text("It is 3.5 meters.", lang="en")
        assert "three" in result
        assert "point" in result
        assert "five" in result
        assert "3.5" not in result

    def test_integer(self):
        result = normalize_tts_text("Ich habe 42 Äpfel.", lang="de")
        assert "zweiundvierzig" in result
        assert "42" not in result

    def test_english_integer(self):
        result = normalize_tts_text("I have 42 apples.", lang="en")
        assert "forty-two" in result
        assert "42" not in result

    def test_thousands_german(self):
        result = normalize_tts_text("1.234,56", lang="de")
        assert "1.234,56" not in result

    def test_thousands_english(self):
        result = normalize_tts_text("1,234.56", lang="en")
        assert "1,234.56" not in result


class TestEdgeCases:
    def test_empty(self):
        assert normalize_tts_text("", lang="de") == ""

    def test_no_changes(self):
        text = "Hallo Welt."
        assert normalize_tts_text(text, lang="de") == "Hallo Welt."

    def test_mixed(self):
        result = normalize_tts_text("Es sind ca. 25 °C und 50% Luftfeuchtigkeit.", lang="de")
        assert "circa" in result
        assert "Grad Celsius" in result
        assert "Prozent" in result
        assert "25" not in result
        assert "50%" not in result

    def test_case_insensitive_abbrev(self):
        result = normalize_tts_text("Das ist Z.B. ein Test.", lang="de")
        assert "zum Beispiel" in result


class TestParagraphBreaks:
    def test_bullet_points_get_periods(self):
        text = "Freitag: Regen\n\nSamstag: Sonne"
        result = normalize_tts_text(text, lang="de")
        assert "Freitag: Regen." in result
        assert "Samstag: Sonne." in result
        assert result.count(".") == 2

    def test_no_double_period(self):
        text = "Erster Satz.\n\nZweiter Satz."
        result = normalize_tts_text(text, lang="de")
        assert ".." not in result
        assert result == "Erster Satz. Zweiter Satz."

    def test_inline_newline_collapsed(self):
        text = "Dies ist ein\nweicher Umbruch."
        result = normalize_tts_text(text, lang="de")
        assert "\n" not in result
        assert "weicher Umbruch" in result

    def test_multiple_blank_lines_normalized(self):
        text = "A\n\n\nB"
        result = normalize_tts_text(text, lang="de")
        assert "A. B." in result or result == "A. B."

    def test_mixed_paragraphs_and_sentences(self):
        text = "Intro.\n\nBullet one\n\nBullet two!\n\nOutro"
        result = normalize_tts_text(text, lang="de")
        assert "Intro." in result
        assert "Bullet one." in result
        assert "Bullet two!" in result
        assert "Outro." in result
        assert ".." not in result
        assert "!!" not in result
