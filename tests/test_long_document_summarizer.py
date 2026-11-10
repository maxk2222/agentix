"""Tests for the long_document_summarizer builtin tool.

Covers absolute-path collection_file resolution (regression: split("/",1)
mangled leading-slash stored sources), the collection//absolute form, and
the tabular-input guard that replaces meaningless pseudo-summaries for
spreadsheet-style extractions with an honest hint.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.tools.builtin_tools.long_document_summarizer import LongDocumentSummarizerTool


def _make_tool(collections: list[str] | None = None) -> LongDocumentSummarizerTool:
    rag_app = MagicMock()
    rag_app.vector_managers = {name: MagicMock() for name in (collections or [])}
    tool = LongDocumentSummarizerTool(rag_app)
    tool._reader = MagicMock()
    tool._reader.read_document_text = AsyncMock(return_value=("doc body", None))
    return tool


# ---------------------------------------------------------------------------
# _read_collection_file — absolute source paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_collection_filename_format():
    tool = _make_tool(["CCL-Calls"])
    text, truncated = await tool._read_collection_file(
        "CCL-Calls/notes.md", None, 250_000
    )
    assert text == "doc body" and truncated is None
    tool._reader.read_document_text.assert_awaited_with(
        "CCL-Calls", "notes.md", context=None
    )


@pytest.mark.asyncio
async def test_absolute_source_path_locates_collection():
    tool = _make_tool(["CCL-Calls", "BSI"])
    await tool._read_collection_file(
        "/r1/.webdav_cache/CCL-Calls/doc.pdf", None, 250_000
    )
    # Verbatim stored source passed through — leading '/' NOT mangled away.
    tool._reader.read_document_text.assert_awaited_with(
        "CCL-Calls", "/r1/.webdav_cache/CCL-Calls/doc.pdf", context=None
    )


@pytest.mark.asyncio
async def test_absolute_path_longest_collection_match():
    tool = _make_tool(["CCL", "CCL-Calls"])
    await tool._read_collection_file(
        "/x/.webdav_cache/CCL-Calls/f.pdf", None, 250_000
    )
    tool._reader.read_document_text.assert_awaited_with(
        "CCL-Calls", "/x/.webdav_cache/CCL-Calls/f.pdf", context=None
    )


@pytest.mark.asyncio
async def test_collection_double_slash_absolute_form():
    tool = _make_tool(["CCL-Calls"])
    await tool._read_collection_file(
        "CCL-Calls//r1/.webdav_cache/CCL-Calls/f.pdf", None, 250_000
    )
    tool._reader.read_document_text.assert_awaited_with(
        "CCL-Calls", "/r1/.webdav_cache/CCL-Calls/f.pdf", context=None
    )


@pytest.mark.asyncio
async def test_absolute_path_unknown_collection():
    tool = _make_tool(["Other"])
    with pytest.raises(FileNotFoundError, match="Collection not determinable"):
        await tool._read_collection_file(
            "/r1/.webdav_cache/CCL-Calls/f.pdf", None, 250_000
        )


@pytest.mark.asyncio
async def test_no_slash_rejected():
    tool = _make_tool(["CCL-Calls"])
    with pytest.raises(ValueError, match="collection/filename"):
        await tool._read_collection_file("bare.md", None, 250_000)


# ---------------------------------------------------------------------------
# _extractive_summary — tabular guard
# ---------------------------------------------------------------------------

def _summarize(tool: LongDocumentSummarizerTool, text: str, **kw: Any) -> str:
    return tool._extractive_summary(
        text,
        kw.get("max_sentences", 5),
        kw.get("focus"),
        kw.get("format", "paragraph"),
        kw.get("deduplicate", True),
        kw.get("max_words"),
        kw.get("min_sentences", 1),
    )


def test_tabular_input_returns_hint_not_pseudo_summary():
    tool = _make_tool()
    # xlsx-style extraction: very long text, few "sentences" (row fragments).
    row = "2023-01 | Projekt | 12500.00 | abgeschlossen | Bemerkung ohne Satzende "
    text = row * 300  # ~18k chars, only a handful of sentence splits
    out = _summarize(tool, text)
    assert out.startswith("[Input looks tabular/structured")
    assert "rag_search action='read'" in out


def test_short_tabular_not_flagged():
    tool = _make_tool()
    # Under the 6000-char threshold the guard must stay off — short text
    # passes through the normal (degenerate) path.
    row = "a | b | c | d | e "
    text = row * 200  # ~3.6k chars
    out = _summarize(tool, text)
    assert "tabular/structured" not in out


def test_prose_unaffected():
    tool = _make_tool()
    sentences = [
        f"Sentence number {i} contains several normal words and ends properly."
        for i in range(40)
    ]
    text = " ".join(sentences) * 3  # >6000 chars, plenty of real sentences
    out = _summarize(tool, text, max_sentences=3)
    assert "tabular/structured" not in out
    assert "Sentence number" in out


def test_empty_text_still_returns_excerpt():
    tool = _make_tool()
    out = _summarize(tool, "   ")
    assert isinstance(out, str)
