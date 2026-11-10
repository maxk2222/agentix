"""Long document summarizer builtin tool."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

from .web_scraper import WebScraperTool

logger = logging.getLogger(__name__)


class LongDocumentSummarizerTool:
    """Extract a concise summary from a long text, URL, or collection file."""

    name = "long_document_summarizer"
    description = (
        "Summarize a long document provided as text, URL, or collection file path. "
        "Returns an extractive summary. Use 'text', 'url', or 'collection_file' (one required). "
        "Optional 'max_sentences' (default 5), 'max_words', 'min_sentences' (default 1), "
        "'focus' (topic to emphasize), 'format' ('paragraph' or 'bullet_points'), "
        "'deduplicate' (default true), and 'read_limit' (chars loaded from a "
        "collection file, default 250000 — increase when the summary reports truncation)."
    )
    _READ_LIMIT_DEFAULT = 250_000
    _READ_LIMIT_MAX = 1_000_000

    def __init__(self, rag_app: Optional[Any] = None) -> None:
        self._rag_app = rag_app
        self._reader: Optional[Any] = None

    def _rag_search(self) -> Any:
        """Lazily build a RagSearchTool for the collection read chain."""
        if self._reader is None:
            from .rag_search import RagSearchTool
            self._reader = RagSearchTool(self._rag_app)
        return self._reader

    @property
    def tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Raw text to summarize. Mutually exclusive with url and collection_file."},
                        "url": {"type": "string", "description": "URL to fetch and summarize. Mutually exclusive with text and collection_file."},
                        "collection_file": {"type": "string", "description": "Collection file path like 'collection/filename.md', or an absolute stored source path (e.g. '/r1/.webdav_cache/Col/doc.pdf'). Mutually exclusive with text and url."},
                        "max_sentences": {"type": "integer", "description": "Maximum number of sentences in the summary (default 5, max 50).", "default": 5},
                        "max_words": {"type": "integer", "description": "Optional maximum total words in the summary. Acts as a hard limit alongside max_sentences."},
                        "min_sentences": {"type": "integer", "description": "Minimum number of sentences (default 1).", "default": 1},
                        "focus": {"type": "string", "description": "Optional topic or keyword to emphasize. Multi-word phrases supported."},
                        "format": {"type": "string", "enum": ["paragraph", "bullet_points"], "description": "Output format: paragraph or bullet_points.", "default": "paragraph"},
                        "deduplicate": {"type": "boolean", "description": "Remove near-duplicate sentences before scoring (default true).", "default": True},
                        "read_limit": {"type": "integer", "description": "Maximum characters loaded from collection_file (default 250000, max 1000000). Increase when the summary reports input truncation.", "default": 250000},
                    },
                    "required": [],
                },
            },
        }

    async def execute(
        self,
        text: Optional[str] = None,
        url: Optional[str] = None,
        collection_file: Optional[str] = None,
        max_sentences: int = 5,
        max_words: Optional[int] = None,
        min_sentences: int = 1,
        focus: Optional[str] = None,
        format: str = "paragraph",
        deduplicate: bool = True,
        read_limit: int = _READ_LIMIT_DEFAULT,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        source = text
        truncated_at: Optional[int] = None
        if url:
            scraper = WebScraperTool()
            source = await scraper.execute(url=url)
        elif collection_file:
            source, truncated_at = await self._read_collection_file(
                collection_file, context, read_limit
            )

        if not source or not source.strip():
            raise ValueError("No content to summarize. Provide text, url, or collection_file.")

        effective_max = min(max(max_sentences, min_sentences), 50)
        summary = self._extractive_summary(
            source, effective_max, focus, format, deduplicate, max_words, min_sentences
        )
        if truncated_at is not None:
            summary += (
                f"\n\n[Input truncated at {truncated_at} characters — "
                f"rerun with a higher 'read_limit' (max {self._READ_LIMIT_MAX:,}) "
                "for the full document.]"
            )
        return summary

    async def _read_collection_file(
        self, path: str, context: Optional[Dict[str, Any]], read_limit: int
    ) -> tuple[str, Optional[int]]:
        """Load a collection document as ``collection/filename``.

        Uses the RagSearchTool read chain: byte-exact raw file first, then
        vector-store resolution + chunk reassembly for sources without a raw
        file (e.g. WebDAV-ingested or binary documents). Returns
        ``(text, truncated_at)`` — ``truncated_at`` is the applied read_limit
        when the document exceeded it, else ``None``.
        """
        if "/" not in path:
            raise ValueError("collection_file must be in 'collection/filename.md' format")
        if self._rag_app is None:
            raise RuntimeError("collection_file reads require application context")
        if path.startswith("/") or ".webdav_cache/" in path:
            # Absolute stored source path — the collection is a path segment,
            # longest name match wins. split("/",1) would otherwise drop the
            # leading '/' and mangle the collection name. The explicit
            # 'collection//absolute/path' form names the collection upfront.
            managers = getattr(self._rag_app, "vector_managers", None) or {}
            head, sep, tail = path.partition("//")
            if sep and head in managers:
                collection, filename = head, "/" + tail
            else:
                matches = [n for n in managers if f"/{n}/" in path]
                if not matches:
                    raise FileNotFoundError(
                        "Collection not determinable from absolute path — "
                        "use 'collection/filename'"
                    )
                collection, filename = max(matches, key=len), path
        else:
            collection, filename = path.split("/", 1)
        text, error = await self._rag_search().read_document_text(
            collection, filename, context=context
        )
        if error is not None or text is None:
            raise FileNotFoundError(error or f"Collection file not found: {path}")
        limit = min(max(int(read_limit or 0), 1000), self._READ_LIMIT_MAX)
        if len(text) > limit:
            return text[:limit], limit
        return text, None

    _ABBREVIATIONS = {
        "dr.", "prof.", "mr.", "mrs.", "ms.", "etc.", "e.g.", "i.e.", "vs.", "vol.", "inc.",
        "ltd.", "jr.", "sr.", "st.", "ave.", "blvd.", "rd.", "no.", "pp.", "et al.",
        "ca.", "z.b.", "u.a.", "d.h.", "bzw.", "usw.", "vgl.", "s.", "abs.",
        "fig.", "tab.", "eg.", "ie.", "pg.",
    }

    def _split_sentences(self, text: str) -> List[str]:
        import re

        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        placeholder = "##DOT##"
        masked = text
        for abbr in self._ABBREVIATIONS:
            pattern = re.compile(r"(?<!\w)" + re.escape(abbr) + r"(?!\w)", re.IGNORECASE)
            masked = pattern.sub(lambda m: m.group(0).replace(".", placeholder), masked)
        masked = re.sub(r"(\d)\.(\d)", r"\1" + placeholder + r"\2", masked)

        raw = re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ"\(0-9])', masked)
        sentences = []
        for s in raw:
            s = s.replace(placeholder, ".").strip()
            if len(s) > 10:
                sentences.append(s)
        return sentences

    def _sentence_similarity(self, a: str, b: str) -> float:
        import re

        words_a = set(re.findall(r'\b[a-zA-ZäöüßÄÖÜ]{3,}\b', a.lower()))
        words_b = set(re.findall(r'\b[a-zA-ZäöüßÄÖÜ]{3,}\b', b.lower()))
        if not words_a or not words_b:
            return 0.0
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return intersection / union if union else 0.0

    def _deduplicate_sentences(self, sentences: List[str]) -> List[str]:
        kept: List[str] = []
        for s in sentences:
            is_dup = False
            for existing in kept:
                if self._sentence_similarity(s, existing) > 0.85:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(s)
        return kept

    def _score_sentences(self, sentences: List[str], focus: Optional[str]) -> List[str]:
        import re

        word_freq: Counter[str] = Counter()
        for sentence in sentences:
            words = re.findall(r'\b[a-zA-ZäöüßÄÖÜ]{3,}\b', sentence.lower())
            word_freq.update(words)

        focus_phrases: List[str] = []
        focus_words: set[str] = set()
        if focus:
            focus_lower = focus.lower()
            focus_phrases = [p.strip() for p in re.split(r'[,;|]', focus_lower) if len(p.strip()) > 2]
            focus_words = set(re.findall(r'\b[a-zA-ZäöüßÄÖÜ]{3,}\b', focus_lower))

        def score(sentence: str) -> float:
            words = re.findall(r'\b[a-zA-ZäöüßÄÖÜ]{3,}\b', sentence.lower())
            if not words:
                return 0.0
            base_score = sum(word_freq[w] for w in words) / len(words)
            sent_lower = sentence.lower()
            for phrase in focus_phrases:
                if phrase in sent_lower:
                    base_score += 15
            if focus_words:
                focus_hits = sum(1 for w in words if w in focus_words)
                base_score += focus_hits * 10
            return base_score

        scored = [(s, score(s)) for s in sentences]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored]

    def _format_output(self, sentences: List[str], format: str) -> str:
        if format == "bullet_points":
            return "\n".join(f"- {s}" for s in sentences)
        return " ".join(sentences)

    def _extractive_summary(
        self,
        text: str,
        max_sentences: int,
        focus: Optional[str],
        format: str,
        deduplicate: bool,
        max_words: Optional[int],
        min_sentences: int,
    ) -> str:
        text = text.strip()
        sentences = self._split_sentences(text)
        if not sentences:
            return text[:2_000] + ("\n[...truncated]" if len(text) > 2_000 else "")

        # Tabular guard: sentence scoring is meaningless on structured row
        # fragments (xlsx/csv-style extractions produce few, over-long
        # "sentences"). Only fires on substantial inputs so short genuine
        # prose is never misclassified.
        chars = len(text)
        avg_sentence_len = chars / max(len(sentences), 1)
        looks_tabular = (
            len(sentences) < max(3, chars / 3000) or avg_sentence_len > 400
        ) and chars > 6000
        if looks_tabular:
            return (
                "[Input looks tabular/structured, not prose — extractive "
                "summary not meaningful. Read it via rag_search action='read' "
                "in chunks.]\n\n" + text[:2_000]
            )

        if deduplicate:
            sentences = self._deduplicate_sentences(sentences)

        if len(sentences) <= min_sentences:
            return self._format_output(sentences, format)

        effective_max = min(max_sentences, len(sentences))
        top = self._score_sentences(sentences, focus)[:effective_max]

        ordered = [s for s in sentences if s in top]

        if max_words:
            word_count = 0
            trimmed = []
            for s in ordered:
                w_count = len(s.split())
                if word_count + w_count <= max_words:
                    trimmed.append(s)
                    word_count += w_count
                else:
                    break
            if len(trimmed) < min_sentences:
                trimmed = ordered[:min_sentences]
            ordered = trimmed

        return self._format_output(ordered, format)
