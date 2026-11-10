"""Lightweight utilities for history-aware retrieval without LLM calls."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


# Curated conversational noise / politeness patterns in German and English
CONVERSATIONAL_PATTERNS = [
    # ---- German confirmations ----
    r"\balles klar\b", r"\bin ordnung\b", r"\bok cool\b",
    r"\bverstanden\b", r"\bperfekt\b", r"\bstimmt\b", r"\bgenau\b", r"\bprima\b", r"\bfein\b",
    r"\brichtig\b", r"\bpasst\b", r"\bsuper\b", r"\bklar\b", r"\bgut\b", r"\btop\b",
    r"\bja\b", r"\bnein\b", r"\bjep\b", r"\bnope\b",
    # ---- English confirmations ----
    r"\bsounds good\b", r"\bmakes sense\b", r"\bwill do\b", r"\bgot it\b",
    r"\bperfect\b", r"\bawesome\b", r"\bexactly\b", r"\bindeed\b", r"\bcorrect\b",
    r"\bgreat\b", r"\bfine\b", r"\bnice\b", r"\bcool\b", r"\bright\b", r"\bsure\b",
    r"\byes\b", r"\bno\b", r"\byep\b", r"\byeah\b", r"\bnah\b", r"\broger\b",
    # ---- German wellbeing ----
    r"\bwie geht es dir\b", r"\bwie geht('?s)?\s+dir\b", r"\bwie geht('?s)?\b", r"\bwas geht\b",
    # ---- English wellbeing ----
    r"\bhow is it going\b", r"\bhow's it going\b", r"\bhow do you do\b",
    r"\bhow are you\b", r"\bhow r u\b", r"\bwhat's up\b", r"\bsup\b",
    # ---- German pleasantries ----
    r"\bschönen tag noch\b", r"\bviel erfolg\b", r"\balles gute\b", r"\bgute besserung\b",
    r"\bfrohes fest\b", r"\bzum wohl\b", r"\bmach's gut\b", r"\bgute nacht\b", r"\bprost\b",
    # ---- English pleasantries ----
    r"\bhave a nice day\b", r"\bget well soon\b", r"\bhappy holidays\b",
    r"\bgood luck\b", r"\bcongratulations\b", r"\bwell done\b", r"\bbless you\b",
    r"\btake care\b", r"\bgood night\b", r"\bcheers\b",
    # ---- German farewells ----
    r"\bauf wiedersehen\b", r"\bauf bald\b", r"\bbis später\b", r"\bbis morgen\b",
    r"\bbis bald\b", r"\bbis dann\b", r"\btschüss\b", r"\btschö\b", r"\bciao\b",
    # ---- English farewells ----
    r"\btalk to you later\b", r"\bhave a good one\b", r"\bgoodbye\b", r"\bsee you\b",
    r"\bsee ya\b", r"\blater\b", r"\bpeace\b", r"\bcya\b", r"\bcu\b", r"\bbye\b", r"\bnight\b",
    # ---- German regional greetings ----
    r"\bgrüß gott\b", r"\bservus\b", r"\bguten abend\b", r"\bguten morgen\b",
    r"\bguten tag\b", r"\bmoien\b", r"\bnabend\b", r"\bmoin\b", r"\btach\b",
    # ---- English regional greetings ----
    r"\bhey there\b", r"\bhowdy\b",
    # ---- German thanks ----
    r"\bvielen herzlichen dank\b", r"\bvielen dank dafür\b", r"\btausend dank\b",
    r"\bbesten dank\b", r"\bvielen dank\b", r"\bdanke schön\b", r"\bdanke sehr\b",
    r"\bdanke euch\b", r"\bdanke dir\b", r"\bok danke\b", r"\bdanke\b",
    r"\bbitte schön\b", r"\bbitte sehr\b", r"\bgerne\b", r"\bgern geschehen\b",
    r"\bkein problem\b", r"\bbitte\b",
    # ---- English thanks ----
    r"\bthanks so much\b", r"\bthanks a lot\b", r"\bmany thanks\b", r"\bbig thanks\b",
    r"\bmuch appreciated\b", r"\bappreciate it\b", r"\bthank you\b", r"\bthank u\b",
    r"\bok thanks\b", r"\bthanks\b", r"\bplease\b", r"\bthx\b", r"\bty\b",
    # ---- German polite requests / apologies ----
    r"\bkönntest du\b", r"\bwürdest du\b", r"\bhast du\b", r"\bmach mal\b",
    r"\bes tut mir leid\b", r"\bentschuldigung\b", r"\bverzeihung\b",
    # ---- English polite requests / apologies ----
    r"\bcould you\b", r"\bwould you\b", r"\bcan you\b", r"\bwill you\b",
    r"\bi apologize\b", r"\bapologies\b", r"\bexcuse me\b", r"\bpardon\b",
    r"\bmy bad\b", r"\bsorry\b",
    # ---- German chit-chat ----
    r"\bwas gibts neues\b", r"\bwas machst du\b", r"\bwie war dein tag\b", r"\bna und\b",
    # ---- English chit-chat ----
    r"\bnice to meet you\b", r"\blong time no see\b", r"\bwhat's the weather like\b",
    r"\bwhat's going on\b", r"\bwhat's new\b", r"\bhow was your day\b",
    # ---- Common greetings ----
    r"\bhello\b", r"\bhey\b", r"\bhi\b", r"\bhallo\b",
    # ---- Emoji (with optional skin-tone modifiers) ----
    "[👋🙏👍🤝🎉😄😉🤗🤙][🏻-🏿]?",
    "[✌✌️][🏻-🏿]?",
    "[🙂😊🙌💪❤️🔥💯✨]",
]

_CONVERSATIONAL_REGEX = re.compile(
    "|".join(CONVERSATIONAL_PATTERNS),
    re.IGNORECASE
)

# Standalone query patterns: definition/explanation questions that should not be mixed with history
_STANDALONE_PATTERNS = re.compile(
    r"^(was\s+(ist|sind|bedeutet|heißt|versteht\s+man\s+unter)|"
    r"erkläre|definition\s*(von)?|"
    r"was\s+weißt\s+du\s+über|was\s+kannst\s+du\s+mir\s+sagen\s+über|"
    r"mehr\s+(infos|informationen)\s+zu|"
    r"what\s+(is|are|does)|explain|definition\s+of|"
    r"more\s+(info|information)\s+about|tell\s+me\s+about)",
    re.IGNORECASE,
)

# Pronouns that suggest the query refers to prior conversation context
_CONTEXT_REFERRING_PRONOUNS = {
    "das", "dies", "es", "dieses", "jenes", "dem", "denen",
    "it", "this", "that", "these", "those", "them", "him", "her",
    "er", "sie", "ihn", "ihm", "ihr", "ihnen",
}

# German + English stopwords for keyword search precision
_STOPWORDS = {
    "was", "ist", "sind", "ein", "eine", "einer", "eines", "einem", "einen",
    "der", "die", "das", "den", "dem", "des", "und", "oder", "aber", "denn",
    "weil", "wenn", "als", "wie", "zu", "zum", "zur", "für", "von", "vom",
    "mit", "nach", "bei", "in", "im", "an", "am", "auf", "aus", "über",
    "unter", "durch", "wer", "wo", "wann", "warum", "welche", "welcher",
    "welches", "was", "wie",
    "what", "is", "are", "the", "a", "an", "and", "or", "but", "for", "of",
    "to", "in", "on", "at", "by", "with", "from", "about", "into", "through",
    "mehr", "infos", "informationen", "info", "bitte", "please", "more",
    "ok", "danke", "thanks", "thank", "you", "hallo", "hello", "hi",
    # Common typos
    "ztu",  # typo for "zu"
}


def _clean_noise(text: str) -> str:
    """Strip conversational filler and normalize whitespace/punctuation."""
    if not text:
        return ""
    # Replace common punctuation (including repeated and dashes) with spaces
    cleaned = re.sub(r'[?!.,:;\-–—…]+', ' ', text)
    # Remove matched conversational phrases and emoji
    cleaned = _CONVERSATIONAL_REGEX.sub(' ', cleaned)
    # Normalize multiple spaces into single ones
    return " ".join(cleaned.strip().split())


def _is_standalone_query(text: str) -> bool:
    """Return True when the query is self-contained and should not be mixed with history.

    Standalone queries include definition/explanation questions ("was ist X?",
    "explain Y") and short topic requests ("mehr infos zu Z") that do not
    reference prior context via pronouns.
    """
    cleaned = _clean_noise(text)
    if not cleaned:
        return False
    tokens = cleaned.lower().split()
    # If it matches a standalone pattern and doesn't reference prior context
    if _STANDALONE_PATTERNS.search(cleaned):
        if not any(t in _CONTEXT_REFERRING_PRONOUNS for t in tokens):
            return True
    return False


def _remove_stopwords(text: str) -> str:
    """Remove German/English stopwords to keep only meaningful search terms."""
    if not text:
        return ""
    tokens = text.strip().split()
    filtered = [t for t in tokens if t.lower() not in _STOPWORDS]
    if not filtered:
        # If all words are stopwords, keep the original to avoid empty queries
        return text.strip()
    return " ".join(filtered)


def _build_retrieval_query(
    current_input: str,
    chat_history: list,
    enabled: bool = True,
    max_user_messages: int = 2,
    max_chars: int = 800,
    standalone_detection: bool = True,
) -> str:
    """Build a retrieval query from the current input and recent human messages.

    The current input is always placed first.  Recent human messages are
    appended in reverse chronological order (most recent first) and the total
    length is hard-capped to *max_chars*.

    Purely conversational / polite turns are stripped of noise. If a turn
    contains only noise/greetings, RAG retrieval is bypassed by returning "".

    When *standalone_detection* is True, short definition/explanation queries
    ("was ist X?", "mehr infos zu Y") are treated as self-contained and prior
    history is not mixed into the retrieval query.
    """
    if not enabled:
        return current_input.strip()

    cleaned_input = _clean_noise(current_input)
    if not cleaned_input:
        # Purely conversational filler or greeting. No search query should be triggered.
        return ""

    # Standalone queries (definitions, explanations, short topic requests) should not
    # be mixed with prior history because that drifts the semantic search away from the
    # actual topic the user is asking about.
    if standalone_detection and _is_standalone_query(current_input):
        return cleaned_input

    # Collect recent human messages (skip system/assistant/tool)
    human_parts: list[str] = []
    for msg in reversed(chat_history):
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        if role not in ("human", "user"):
            continue
        content = getattr(msg, "content", "") or ""
        if not isinstance(content, str):
            continue
        
        cleaned_content = _clean_noise(content)
        # Avoid empty content or duplicates of the current cleaned query
        if cleaned_content and cleaned_content != cleaned_input:
            human_parts.append(cleaned_content)
        if len(human_parts) >= max_user_messages:
            break

    if not human_parts:
        return cleaned_input

    # Build query: current input first, then recent human messages
    parts = [cleaned_input] + human_parts
    query = " ".join(parts)
    if len(query) > max_chars:
        query = query[:max_chars].rsplit(" ", 1)[0]
    return query.strip()


def _is_short_query(query: str, short_threshold: int = 5) -> bool:
    """Return True when the query is very short (few tokens)."""
    return len(query.strip().split()) < short_threshold


def _is_long_query(query: str, long_threshold: int = 15) -> bool:
    """Return True when the query is long (many tokens)."""
    return len(query.strip().split()) > long_threshold


def _compute_dynamic_candidates(
    query: str,
    base_vector: int = 40,
    base_keyword: int = 40,
    enabled: bool = True,
) -> tuple[int, int]:
    """Adjust candidate counts based on query length heuristics.

    - Short query: fewer vector, more keyword candidates.
    - Normal query: base values.
    - Long query: more vector, fewer keyword candidates.
    """
    if not enabled:
        return base_vector, base_keyword

    if _is_short_query(query):
        return max(10, base_vector // 2), max(20, int(base_keyword * 1.25))
    if _is_long_query(query):
        return max(20, int(base_vector * 1.25)), max(10, base_keyword // 2)
    return base_vector, base_keyword
