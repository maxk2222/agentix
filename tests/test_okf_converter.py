"""Tests for app.okf_converter."""

from __future__ import annotations

from langchain.schema import Document

from app.okf_converter import OKFConverter, _okf_type_from_extension


def test_okf_type_from_extension():
    assert _okf_type_from_extension("/docs/report.pdf") == "document"
    assert _okf_type_from_extension("/data/sheet.xlsx") == "spreadsheet"
    assert _okf_type_from_extension("/slides.pptx") == "presentation"
    assert _okf_type_from_extension("/notes.txt") == "text"
    assert _okf_type_from_extension("/unknown.bin") == "text"


def test_convert_document_basic():
    doc = Document(
        page_content="This is a sample document content for testing.",
        metadata={"title": "Test Doc", "source": "/docs/test.txt"},
    )
    meta = {"etag": '"abc"', "content_length": 1234}
    result = OKFConverter.convert_document(doc, "/docs/test.txt", meta)

    assert result.page_content == "This is a sample document content for testing."
    assert result.metadata["okf_concept_id"] == "/docs/test.txt"
    assert result.metadata["okf_type"] == "text"
    assert result.metadata["okf_title"] == "Test Doc"
    assert result.metadata["okf_frontmatter"]["title"] == "Test Doc"
    assert result.metadata["okf_frontmatter"]["type"] == "text"
    assert result.metadata["okf_frontmatter"]["etag"] == '"abc"'
    assert result.metadata["okf_frontmatter"]["file_size"] == 1234
    assert "date_downloaded" in result.metadata["okf_frontmatter"]


def test_convert_document_uses_filename_as_title():
    doc = Document(
        page_content="Content without explicit title.",
        metadata={},
    )
    meta = {}
    result = OKFConverter.convert_document(doc, "/docs/report.pdf", meta)

    assert result.metadata["okf_title"] == "report.pdf"
    assert result.metadata["okf_frontmatter"]["title"] == "report.pdf"
    assert result.metadata["okf_type"] == "document"


def test_convert_document_description_truncation():
    long_text = "word " * 200
    doc = Document(page_content=long_text, metadata={})
    meta = {}
    result = OKFConverter.convert_document(doc, "/big.txt", meta)

    desc = result.metadata["okf_frontmatter"]["description"]
    assert len(desc) <= 503  # max 500 + "..."
    assert desc.endswith("...")


def test_convert_document_preserves_existing_metadata():
    doc = Document(
        page_content="Hello",
        metadata={"author": "Alice", "source": "/old/path.txt"},
    )
    meta = {}
    result = OKFConverter.convert_document(doc, "/new/path.txt", meta)

    assert result.metadata["author"] == "Alice"
    assert result.metadata["source"] == "/new/path.txt"
