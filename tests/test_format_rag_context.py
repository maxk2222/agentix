"""Tests for the _format_rag_context helper."""

from __future__ import annotations

from langchain.schema import Document

from agent.application import _format_rag_context


def test_format_rag_context_with_metadata():
    docs = [
        Document(
            page_content="First document content.",
            metadata={"source": "test.pdf", "page": 5, "chunk_index": 2},
        ),
        Document(
            page_content="Second document content.",
            metadata={"source": "other.md", "page_number": 10},
        ),
    ]
    result = _format_rag_context(docs)
    assert "Source: test.pdf, Page: 5, Chunk: 2" in result
    assert "First document content." in result
    assert "Source: other.md, Page: 10" in result
    assert "Second document content." in result
    assert result.count("\n\n") == 1


def test_format_rag_context_without_metadata():
    docs = [Document(page_content="Plain content.", metadata={})]
    result = _format_rag_context(docs)
    assert result == "Plain content."


def test_format_rag_context_empty_list():
    result = _format_rag_context([])
    assert result == ""
