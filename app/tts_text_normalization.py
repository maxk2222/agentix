"""Text normalization for TTS pipelines.

Expands abbreviations, units, and numeric forms into spoken text
before synthesis so that TTS engines (Kokoro, Piper) receive
clean, pronounceable strings.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abbreviation tables (case-insensitive, punctuation-tolerant)
# ---------------------------------------------------------------------------

_GERMAN_ABBREVIATIONS: Dict[str, str] = {
    "z.b": "zum Beispiel",
    "zb": "zum Beispiel",
    "u.a": "unter anderem",
    "ua": "unter anderem",
    "d.h": "das heißt",
    "dh": "das heißt",
    "etc": "et cetera",
    "usw": "und so weiter",
    "ca": "circa",
    "vgl": "vergleiche",
    "bzw": "beziehungsweise",
    "inkl": "inklusive",
    "exkl": "exklusive",
    "mio": "Millionen",
    "mrd": "Milliarden",
    "nr": "Nummer",
    "std": "Stunden",
    "min": "Minuten",
    "sek": "Sekunden",
    "tel": "Telefon",
    "str": "Straße",
    "dr": "Doktor",
    "prof": "Professor",
    "jr": "Junior",
    "sr": "Senior",
    "co": "Company",
    "inc": "Incorporated",
    "ltd": "Limited",
    "kg": "Kommanditgesellschaft",
    "gmbh": "Gesellschaft mit beschränkter Haftung",
    "ag": "Aktiengesellschaft",
    "ev": "eingetragener Verein",
}

_ENGLISH_ABBREVIATIONS: Dict[str, str] = {
    "i.e": "that is",
    "ie": "that is",
    "e.g": "for example",
    "eg": "for example",
    "etc": "et cetera",
    "et al": "and others",
    "vs": "versus",
    "ca": "circa",
    "approx": "approximately",
    "dr": "Doctor",
    "mr": "Mister",
    "mrs": "Misses",
    "ms": "Miss",
    "prof": "Professor",
    "jr": "Junior",
    "sr": "Senior",
    "st": "Saint",
    "ave": "Avenue",
    "blvd": "Boulevard",
    "rd": "Road",
    "ln": "Lane",
    "ct": "Court",
    "inc": "Incorporated",
    "ltd": "Limited",
    "co": "Company",
    "corp": "Corporation",
    "llc": "Limited Liability Company",
    "plc": "Public Limited Company",
    "vol": "Volume",
    "vols": "Volumes",
    "rev": "Revised",
    "fig": "Figure",
    "figs": "Figures",
    "gov": "Government",
    "dept": "Department",
    "nos": "Numbers",
    "art": "Article",
    "arts": "Articles",
    "pp": "pages",
    "par": "paragraph",
    "pars": "paragraphs",
    "eq": "equation",
    "eqs": "equations",
    "ch": "chapter",
    "chs": "chapters",
    "ed": "edition",
    "eds": "editors",
    "trans": "translated by",
    "bk": "book",
    "bks": "books",
    "pt": "part",
    "pts": "parts",
    "sec": "section",
    "secs": "sections",
    "a.m": "ante meridiem",
    "am": "ante meridiem",
    "p.m": "post meridiem",
    "pm": "post meridiem",
    "b.c": "before Christ",
    "bc": "before Christ",
    "a.d": "anno Domini",
    "ad": "anno Domini",
    "ph.d": "Doctor of Philosophy",
    "m.d": "Doctor of Medicine",
    "b.a": "Bachelor of Arts",
    "m.a": "Master of Arts",
    "m.b.a": "Master of Business Administration",
    "r.n": "Registered Nurse",
    "rep": "Representative",
    "reps": "Representatives",
    "sen": "Senator",
    "sens": "Senators",
    "col": "Colonel",
    "cols": "Colonels",
    "gen": "General",
    "gens": "Generals",
    "lt": "Lieutenant",
    "lts": "Lieutenants",
    "maj": "Major",
    "sgt": "Sergeant",
    "cpl": "Corporal",
    "capt": "Captain",
    "cmdr": "Commander",
    "adm": "Admiral",
    "assoc": "Associate",
    "asst": "Assistant",
    "assts": "Assistants",
    "ext": "Extension",
    "jan": "January",
    "feb": "February",
    "mar": "March",
    "apr": "April",
    "jun": "June",
    "jul": "July",
    "aug": "August",
    "sep": "September",
    "sept": "September",
    "oct": "October",
    "nov": "November",
    "dec": "December",
    "mon": "Monday",
    "tue": "Tuesday",
    "wed": "Wednesday",
    "thu": "Thursday",
    "fri": "Friday",
    "sat": "Saturday",
    "sun": "Sunday",
}

# ---------------------------------------------------------------------------
# Helper: digit / number expansion
# ---------------------------------------------------------------------------

try:
    from num2words import num2words

    def _expand_number(num_str: str, lang: str) -> str:
        """Convert a numeric string to words using num2words."""
        # num2words language codes differ slightly; normalise them.
        code = "de" if lang.startswith("de") else "en"
        try:
            if lang.startswith("de"):
                # German: dots are thousands separators, comma is decimal
                clean = num_str.replace(" ", "").replace(".", "").replace(",", ".")
            else:
                # English: commas are thousands separators, dot is decimal
                clean = num_str.replace(" ", "").replace(",", "")
            value = float(clean)
            # Integers without decimal part → integer words
            if value == int(value):
                return num2words(int(value), lang=code)
            return num2words(value, lang=code)
        except Exception:
            return num_str

except Exception:
    # Fallback if num2words is not installed
    def _expand_number(num_str: str, lang: str) -> str:  # noqa: F811
        _DIGITS_DE = {
            "0": "null",
            "1": "eins",
            "2": "zwei",
            "3": "drei",
            "4": "vier",
            "5": "fünf",
            "6": "sechs",
            "7": "sieben",
            "8": "acht",
            "9": "neun",
            ",": "Komma",
            ".": "Punkt",
            "-": "minus",
        }
        _DIGITS_EN = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine",
            ",": "comma",
            ".": "point",
            "-": "minus",
        }
        mapping = _DIGITS_DE if lang.startswith("de") else _DIGITS_EN
        return " ".join(mapping.get(ch, ch) for ch in num_str)


def _replace_abbreviations(text: str, table: Dict[str, str]) -> str:
    """Replace abbreviations in a case-insensitive way, preserving surrounding text.

    The regex looks for the abbreviation possibly followed by '.' and possibly
    preceded/followed by word boundaries so we don't replace inside words.
    """
    for abbr, expansion in table.items():
        # Escape dots in abbreviation for regex
        escaped = re.escape(abbr)
        # Pattern: word boundary, abbreviation (case-insensitive), optional dot, word boundary
        pattern = rf"\b{escaped}\b\.?"
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    return text


def _expand_numbers(text: str, lang: str) -> str:
    """Expand standalone numeric tokens (integers and decimals) to words.

    Handles both comma and dot as decimal separator depending on language:
      - German:  3,5  → drei komma fünf
      - English: 3.5  → three point five
    Thousands separators (space or dot for German, comma/space for English)
    are stripped before conversion.
    """
    is_de = lang.startswith("de")
    # Regex for numbers: optional sign, digits, optional thousands separators,
    # optional decimal part (comma or dot), more digits.
    # We capture the whole number token.
    def _repl(match: re.Match) -> str:
        num_str = match.group(0)
        # Don't expand years (e.g. 2024) unless explicitly requested –
        # num2words handles them fine, so we expand everything.
        return _expand_number(num_str, lang)

    # Pattern: optional +/-, digits with optional group separators,
    # optional decimal part.
    if is_de:
        # German: 1.234,56 or 1234,56 or 1 234,56
        pattern = r"[+-]?\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+(?:,\d+)?"
    else:
        # English: 1,234.56 or 1234.56 or 1 234.56
        pattern = r"[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"

    return re.sub(pattern, _repl, text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_tts_text(text: str, lang: str = "de") -> str:
    """Return a pronunciation-friendly version of *text* for TTS synthesis.

    Steps applied (order matters):
      1. Expand abbreviations (z.B., u.a., d.h., etc.)
      2. Expand units (°C, °F, %)
      3. Expand numbers (integers and decimals)
    """
    if not text:
        return text

    original = text

    # 1. Abbreviations
    is_de = lang.startswith("de")
    if is_de:
        text = _replace_abbreviations(text, _GERMAN_ABBREVIATIONS)
    else:
        text = _replace_abbreviations(text, _ENGLISH_ABBREVIATIONS)

    # 2. Units & symbols
    # Temperature
    text = re.sub(r"°\s*C\b", " Grad Celsius", text, flags=re.IGNORECASE)
    text = re.sub(r"°\s*F\b", " Grad Fahrenheit", text, flags=re.IGNORECASE)
    # Standalone degree symbol (e.g. "20°")
    text = text.replace("°", " Grad ")
    # Percent
    text = text.replace("%", " Prozent " if is_de else " percent ")
    # Currency symbols (simple)
    text = text.replace("€", " Euro " if is_de else " Euros ")
    text = text.replace("$", " Dollar ")
    text = text.replace("£", " Pfund " if is_de else " Pounds ")

    # 3. Paragraph / bullet-point handling
    # Ensure paragraph breaks become sentence breaks so *all* TTS providers
    # insert a natural pause between bullet points.
    text = _ensure_sentence_breaks(text)

    # 4. Numbers
    text = _expand_numbers(text, lang)

    # 5. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    logger.debug("TTS normalize: %r -> %r", original, text)
    return text


def _ensure_sentence_breaks(text: str) -> str:
    """Split text at paragraph boundaries and ensure each paragraph ends with
    sentence punctuation so TTS engines insert a natural pause between them.

    Single newlines inside a paragraph are preserved as spaces; only blank-line
    separators (``\\n\\n`` and variants) trigger a sentence boundary.
    """
    # Fast-path: no paragraph breaks → just collapse soft line breaks
    if "\n\n" not in text and not re.search(r"\n\s*\n", text):
        return text.replace("\n", " ")

    # Normalise blank-line separators to a single marker
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    paragraphs = text.split("\n\n")
    processed: list[str] = []
    for para in paragraphs:
        # Soft line breaks inside a paragraph become spaces
        para = para.replace("\n", " ").strip()
        if not para:
            continue
        # Add a period if the paragraph does not end with sentence punctuation
        if not para.endswith((".", "!", "?", ":", ";")):
            para += "."
        processed.append(para)
    return " ".join(processed)
