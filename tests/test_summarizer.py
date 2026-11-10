"""Tests for the built-in long_document_summarizer tool."""

from __future__ import annotations

import pytest

from agent.tools.builtin import LongDocumentSummarizerTool


@pytest.fixture
def summarizer() -> LongDocumentSummarizerTool:
    return LongDocumentSummarizerTool()


class TestSentenceSplitting:
    def test_basic_split(self, summarizer):
        text = "First sentence. Second sentence! Third sentence?"
        sentences = summarizer._split_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence!"
        assert sentences[2] == "Third sentence?"

    def test_abbreviations_not_split(self, summarizer):
        text = "Dr. Smith visited Prof. Johnson, e.g. on Monday. Then he left."
        sentences = summarizer._split_sentences(text)
        assert len(sentences) == 2
        assert "Dr. Smith" in sentences[0]
        assert "Prof. Johnson" in sentences[0]
        assert "e.g." in sentences[0]

    def test_decimal_numbers_not_split(self, summarizer):
        text = "The value is 3.14 meters. It is correct."
        sentences = summarizer._split_sentences(text)
        assert len(sentences) == 2
        assert "3.14" in sentences[0]

    def test_german_abbreviations(self, summarizer):
        text = "Z.B. sollte man ca. 10 min. warten. Dann geht es weiter."
        sentences = summarizer._split_sentences(text)
        assert len(sentences) == 2
        assert "Z.B." in sentences[0]
        assert "ca." in sentences[0]


class TestDeduplication:
    def test_removes_duplicates(self, summarizer):
        text = "The cat sat on the mat. The dog ran in the park. The cat sat on the mat. The bird flew high."
        sentences = summarizer._split_sentences(text)
        deduped = summarizer._deduplicate_sentences(sentences)
        assert len(deduped) == 3

    def test_keeps_similar_but_not_identical(self, summarizer):
        text = "The cat sat on the mat. The cat sat on the chair. The dog ran in the park."
        sentences = summarizer._split_sentences(text)
        deduped = summarizer._deduplicate_sentences(sentences)
        assert len(deduped) == 3


class TestFocusPhrases:
    def test_single_word_focus(self, summarizer):
        text = "Apples are red. Bananas are yellow. Apples are tasty. Grapes are purple."
        summary = summarizer._extractive_summary(
            text, max_sentences=2, focus="apples", format="paragraph",
            deduplicate=True, max_words=None, min_sentences=1,
        )
        assert "Apples" in summary

    def test_multi_word_focus(self, summarizer):
        text = (
            "Machine learning is powerful. Deep learning is a subset. "
            "Machine learning models need data. The weather is nice."
        )
        summary = summarizer._extractive_summary(
            text, max_sentences=2, focus="machine learning", format="paragraph",
            deduplicate=True, max_words=None, min_sentences=1,
        )
        assert "Machine learning" in summary

    def test_focus_via_comma_separated(self, summarizer):
        text = "Solar panels are green. Wind turbines spin. Hydro power flows."
        summary = summarizer._extractive_summary(
            text, max_sentences=2, focus="solar, hydro", format="paragraph",
            deduplicate=True, max_words=None, min_sentences=1,
        )
        assert "Solar" in summary or "Hydro" in summary


class TestFormatting:
    def test_bullet_points(self, summarizer):
        text = "First important point. Second important point. Third important detail."
        summary = summarizer._extractive_summary(
            text, max_sentences=2, focus=None, format="bullet_points",
            deduplicate=True, max_words=None, min_sentences=1,
        )
        lines = summary.split("\n")
        assert all(line.startswith("- ") for line in lines)
        assert len(lines) == 2

    def test_paragraph(self, summarizer):
        text = "First sentence. Second sentence. Third sentence."
        summary = summarizer._extractive_summary(
            text, max_sentences=2, focus=None, format="paragraph",
            deduplicate=True, max_words=None, min_sentences=1,
        )
        assert "\n" not in summary
        # expect two sentences selected
        assert summary.count(".") == 2


class TestMaxWords:
    def test_max_words_limits(self, summarizer):
        text = "Short sentence one. Short two. Short three here."
        summary = summarizer._extractive_summary(
            text, max_sentences=3, focus=None, format="paragraph",
            deduplicate=True, max_words=5, min_sentences=1,
        )
        words = summary.split()
        assert len(words) <= 5

    def test_min_sentences_overrides_max_words(self, summarizer):
        text = "Short one. Another short one. Third sentence here."
        summary = summarizer._extractive_summary(
            text, max_sentences=3, focus=None, format="paragraph",
            deduplicate=True, max_words=2, min_sentences=2,
        )
        sentences = summarizer._split_sentences(summary)
        assert len(sentences) >= 2


class TestEdgeCases:
    def test_short_text_returns_all(self, summarizer):
        text = "Only one sentence here."
        summary = summarizer._extractive_summary(
            text, max_sentences=5, focus=None, format="paragraph",
            deduplicate=True, max_words=None, min_sentences=1,
        )
        assert summary == "Only one sentence here."

    def test_empty_text_fallback(self, summarizer):
        text = "   "
        sentences = summarizer._split_sentences(text)
        assert sentences == []
        summary = summarizer._extractive_summary(
            text, max_sentences=5, focus=None, format="paragraph",
            deduplicate=True, max_words=None, min_sentences=1,
        )
        assert "truncated" not in summary  # empty string, not truncated
        assert summary == ""

    def test_no_deduplicate(self, summarizer):
        text = "Repeat me. Repeat me. Repeat me."
        summary = summarizer._extractive_summary(
            text, max_sentences=2, focus=None, format="paragraph",
            deduplicate=False, max_words=None, min_sentences=1,
        )
        # Without deduplication, identical sentences can appear if scored high
        assert "Repeat me" in summary


class TestCollectionFileRead:
    """collection_file reads go through the RagSearchTool read chain."""

    def _tool(self, read_result):
        from unittest.mock import AsyncMock, MagicMock
        tool = LongDocumentSummarizerTool(rag_app=MagicMock())
        reader = MagicMock()
        reader.read_document_text = AsyncMock(return_value=read_result)
        tool._reader = reader
        return tool, reader

    @pytest.mark.asyncio
    async def test_collection_file_summarizes(self):
        tool, reader = self._tool((
            "Alpha content here. Beta content there. Gamma content elsewhere.", None
        ))
        result = await tool.execute(
            collection_file="col/doc.md",
            context={"conversation_id": "c1"},
        )
        assert "content" in result
        reader.read_document_text.assert_awaited_once_with(
            "col", "doc.md", context={"conversation_id": "c1"}
        )

    @pytest.mark.asyncio
    async def test_collection_file_error_raises(self):
        tool, _ = self._tool((None, "Document 'x' not found in collection 'col'."))
        with pytest.raises(FileNotFoundError):
            await tool.execute(collection_file="col/missing.md")

    @pytest.mark.asyncio
    async def test_no_rag_app_raises(self):
        tool = LongDocumentSummarizerTool()
        with pytest.raises(RuntimeError, match="application context"):
            await tool.execute(collection_file="col/doc.md")

    @pytest.mark.asyncio
    async def test_truncation_marks_summary(self):
        big = "Word sentence. " * 40_000  # ~600k chars
        tool, _ = self._tool((big, None))
        result = await tool.execute(collection_file="col/big.md", read_limit=250_000)
        assert "Input truncated at 250000" in result
        assert "read_limit" in result

    @pytest.mark.asyncio
    async def test_read_limit_override_avoids_truncation(self):
        text = "Short doc. Another sentence here."
        tool, _ = self._tool((text, None))
        result = await tool.execute(
            collection_file="col/doc.md", read_limit=500_000
        )
        assert "truncated" not in result

    @pytest.mark.asyncio
    async def test_read_limit_clamped_to_bounds(self):
        # read_limit below 1000 clamps up to 1000
        text = "x" * 5_000
        tool, _ = self._tool((text, None))
        result = await tool.execute(collection_file="col/doc.md", read_limit=10)
        assert "Input truncated at 1000" in result

    @pytest.mark.asyncio
    async def test_path_without_slash_rejected(self):
        tool, _ = self._tool(("text", None))
        with pytest.raises(ValueError, match="collection/filename"):
            await tool.execute(collection_file="no_separator.md")

    @pytest.mark.asyncio
    async def test_text_and_url_paths_unaffected(self):
        tool, _ = self._tool(("unused", None))
        result = await tool.execute(
            text="First sentence is long enough. Second one is here too."
        )
        assert "First sentence" in result
