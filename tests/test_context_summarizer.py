from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage

from app.context_summarizer import summarize_context_if_needed, _split_text_for_summary, _hard_truncate
from app.history_budget import count_text_tokens


class FakeSummaryLLM:
    async def ainvoke(self, payload):
        return AIMessage(content="Kompakte Zusammenfassung der wichtigsten Fakten.")


class FakeSummaryLLMWithUsage:
    async def ainvoke(self, payload):
        return AIMessage(
            content="Kompakte Zusammenfassung der wichtigsten Fakten.",
            usage_metadata={
                "input_tokens": 120,
                "output_tokens": 20,
                "total_tokens": 140,
            },
        )


class FailingLLM:
    async def ainvoke(self, payload):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_summarize_context_if_needed_returns_unchanged_when_within_budget():
    text = "Kurzer Text."
    result, summarized = await summarize_context_if_needed(
        text,
        token_budget=1000,
        model="gpt-4o-mini",
        summary_llm_factory=lambda: FakeSummaryLLM(),
    )
    assert result == text
    assert summarized is False


@pytest.mark.asyncio
async def test_summarize_context_if_needed_summarizes_large_text():
    text = "Wichtige Fakten. " * 500
    result, summarized = await summarize_context_if_needed(
        text,
        token_budget=50,
        model="gpt-4o-mini",
        summary_llm_factory=lambda: FakeSummaryLLM(),
        summary_context_window=8192,
    )
    assert summarized is True
    assert result == "Kompakte Zusammenfassung der wichtigsten Fakten."


@pytest.mark.asyncio
async def test_summarize_context_if_needed_falls_back_to_truncation_on_failure():
    text = "Wichtige Fakten. " * 500
    result, summarized = await summarize_context_if_needed(
        text,
        token_budget=50,
        model="gpt-4o-mini",
        summary_llm_factory=lambda: FailingLLM(),
        summary_context_window=8192,
    )
    assert summarized is False
    assert len(result) < len(text)


@pytest.mark.asyncio
async def test_summarize_context_if_needed_falls_back_when_no_factory():
    text = "Wichtige Fakten. " * 500
    result, summarized = await summarize_context_if_needed(
        text,
        token_budget=50,
        model="gpt-4o-mini",
        summary_llm_factory=None,
    )
    assert summarized is False
    assert len(result) < len(text)


@pytest.mark.asyncio
async def test_summarize_context_if_needed_handles_empty_text():
    result, summarized = await summarize_context_if_needed(
        "",
        token_budget=100,
        model="gpt-4o-mini",
        summary_llm_factory=lambda: FakeSummaryLLM(),
    )
    assert result == ""
    assert summarized is False


@pytest.mark.asyncio
async def test_summarize_context_if_needed_handles_task_factory():
    text = "Wichtige Fakten. " * 500

    async def _get_llm():
        return FakeSummaryLLM()

    result, summarized = await summarize_context_if_needed(
        text,
        token_budget=50,
        model="gpt-4o-mini",
        summary_llm_factory=lambda: asyncio.create_task(_get_llm()),
        summary_context_window=8192,
    )
    assert summarized is True
    assert result == "Kompakte Zusammenfassung der wichtigsten Fakten."


@pytest.mark.asyncio
async def test_summarize_context_if_needed_uses_map_reduce_for_very_large_input():
    text = "Wichtige Fakten. " * 2000
    result, summarized = await summarize_context_if_needed(
        text,
        token_budget=100,
        model="gpt-4o-mini",
        summary_llm_factory=lambda: FakeSummaryLLM(),
        summary_context_window=512,
    )
    assert summarized is True
    assert count_text_tokens(result, "gpt-4o-mini") <= 2000


@pytest.mark.asyncio
async def test_summarize_context_if_needed_map_reduce_falls_back_on_failure():
    text = "Wichtige Fakten. " * 2000
    result, summarized = await summarize_context_if_needed(
        text,
        token_budget=100,
        model="gpt-4o-mini",
        summary_llm_factory=lambda: FailingLLM(),
        summary_context_window=512,
    )
    assert len(result) < len(text)


def test_split_text_for_summary_returns_single_chunk_when_small():
    text = "Kurzer Text."
    chunks = _split_text_for_summary(text, 1000, "gpt-4o-mini")
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_text_for_summary_splits_on_paragraph_boundaries():
    para1 = "Erste Aussage. " * 200
    para2 = "Zweite Aussage. " * 200
    text = para1 + "\n\n" + para2
    chunks = _split_text_for_summary(text, 300, "gpt-4o-mini")
    assert len(chunks) >= 2
    for chunk in chunks:
        assert count_text_tokens(chunk, "gpt-4o-mini") <= 350


def test_hard_truncate_returns_unchanged_when_within_budget():
    text = "Kurzer Text."
    result = _hard_truncate(text, 1000, "gpt-4o-mini")
    assert result == text


def test_hard_truncate_truncates_to_budget():
    text = "Wichtige Fakten. " * 500
    result = _hard_truncate(text, 50, "gpt-4o-mini")
    assert count_text_tokens(result, "gpt-4o-mini") <= 50


@pytest.mark.asyncio
async def test_summarize_context_if_needed_records_usage_in_sink():
    """Every summary LLM call's usage lands in usage_sink so callers can
    charge compression tokens to the run total."""
    sink = []
    text = "Wichtige Fakten. " * 500
    result, summarized = await summarize_context_if_needed(
        text,
        token_budget=50,
        model="gpt-4o-mini",
        summary_llm_factory=lambda: FakeSummaryLLMWithUsage(),
        summary_context_window=8192,
        usage_sink=sink,
    )
    assert summarized is True
    assert len(sink) == 1
    assert sink[0]["prompt_tokens"] == 120
    assert sink[0]["completion_tokens"] == 20


@pytest.mark.asyncio
async def test_summarize_context_if_needed_sink_empty_without_llm_call():
    sink = []
    result, summarized = await summarize_context_if_needed(
        "short",
        token_budget=1000,
        model="gpt-4o-mini",
        summary_llm_factory=lambda: FakeSummaryLLMWithUsage(),
        usage_sink=sink,
    )
    assert summarized is False
    assert sink == []
