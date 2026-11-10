"""Unit tests for agent/custom_llm.py."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from agent.custom_llm import (
    _extract_reasoning_from_chunk,
    _extract_reasoning_from_response,
    ChatOpenAIWithReasoning,
    _REASONING_KEY,
)


class TestExtractReasoningFromChunk:
    """Tests for the low-level chunk extractor."""

    def test_empty_dict(self):
        assert _extract_reasoning_from_chunk({}) == ""

    def test_no_choices(self):
        assert _extract_reasoning_from_chunk({"choices": []}) == ""

    def test_missing_delta(self):
        assert _extract_reasoning_from_chunk({"choices": [{}]}) == ""

    def test_standard_key(self):
        chunk = {"choices": [{"delta": {_REASONING_KEY: "think"}}]}
        assert _extract_reasoning_from_chunk(chunk) == "think"

    def test_fallback_reasoning_key(self):
        chunk = {"choices": [{"delta": {"reasoning": "fallback"}}]}
        assert _extract_reasoning_from_chunk(chunk) == "fallback"

    def test_non_dict_choices(self):
        assert _extract_reasoning_from_chunk({"choices": "bad"}) == ""

    def test_non_dict_delta(self):
        assert _extract_reasoning_from_chunk({"choices": [{"delta": "bad"}]}) == ""


class TestExtractReasoningFromResponse:
    """Tests for the non-streaming response extractor."""

    def test_empty(self):
        assert _extract_reasoning_from_response({}) == ""

    def test_standard_key(self):
        resp = {"choices": [{"message": {_REASONING_KEY: "reason"}}]}
        assert _extract_reasoning_from_response(resp) == "reason"

    def test_fallback_reasoning_key(self):
        resp = {"choices": [{"message": {"reasoning": "reason2"}}]}
        assert _extract_reasoning_from_response(resp) == "reason2"

    def test_openai_basemodel(self):
        """Simulate an openai pydantic-style response object."""
        msg = MagicMock()
        msg.reasoning_content = "basemodel_reason"
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        assert _extract_reasoning_from_response(resp) == "basemodel_reason"

    def test_openai_basemodel_fallback(self):
        msg = MagicMock()
        del msg.reasoning_content
        msg.reasoning = "fallback_reason"
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        assert _extract_reasoning_from_response(resp) == "fallback_reason"


class TestReasoningAccumulator:
    """Tests for the context-local reasoning buffer."""

    def test_append_and_clear(self):
        llm = ChatOpenAIWithReasoning(openai_api_key="fake")
        assert llm.get_and_clear_reasoning_text() == ""
        llm._append_reasoning("hello")
        llm._append_reasoning(" world")
        assert llm.get_and_clear_reasoning_text() == "hello world"
        assert llm.get_and_clear_reasoning_text() == ""


class TestThreadSafety:
    """The lock must prevent corrupted data when multiple threads touch the buffer."""

    def test_concurrent_appends_are_atomic(self):
        llm = ChatOpenAIWithReasoning(openai_api_key="fake")

        def worker() -> None:
            for i in range(100):
                llm._append_reasoning("x")

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 1000 appends should be present (lock prevents lost writes)
        assert len(llm._accumulated_reasoning_text) == 1000
        text = llm.get_and_clear_reasoning_text()
        assert text == "x" * 1000


class TestStreamReasoningInjection:
    """Mock-streaming tests that verify reasoning lands in chunks and accumulator."""

    @staticmethod
    def _make_mock_chunk(content: str = "", reasoning: str = "") -> MagicMock:
        m = MagicMock()
        m.model_dump.return_value = {
            "choices": [{"delta": {"content": content, _REASONING_KEY: reasoning}}]
        }
        return m

    def test_sync_stream_injects_reasoning(self):
        llm = ChatOpenAIWithReasoning(openai_api_key="fake", model="gpt-4")
        chunks = [
            self._make_mock_chunk(content="Hello ", reasoning="think1"),
            self._make_mock_chunk(content="world", reasoning="think2"),
        ]

        class _MockResponse:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def __iter__(self):
                return iter(chunks)

        with patch.object(llm.client, "create", return_value=_MockResponse()):
            result = list(llm._stream([HumanMessage(content="test")]))

        assert len(result) == 2
        assert result[0].message.additional_kwargs[_REASONING_KEY] == "think1"
        assert result[1].message.additional_kwargs[_REASONING_KEY] == "think2"
        assert llm.get_and_clear_reasoning_text() == "think1think2"

    @pytest.mark.asyncio
    async def test_async_stream_injects_reasoning(self):
        llm = ChatOpenAIWithReasoning(openai_api_key="fake", model="gpt-4")
        chunks = [
            self._make_mock_chunk(content="Hi ", reasoning="async1"),
            self._make_mock_chunk(content="there", reasoning="async2"),
        ]

        class _MockAsyncResponse:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def __aiter__(self):
                for item in chunks:
                    yield item

        with patch.object(
            llm.async_client, "create", new_callable=AsyncMock, return_value=_MockAsyncResponse()
        ):
            result = []
            async for chunk in llm._astream([HumanMessage(content="test")]):
                result.append(chunk)

        assert len(result) == 2
        assert result[0].message.additional_kwargs[_REASONING_KEY] == "async1"
        assert result[1].message.additional_kwargs[_REASONING_KEY] == "async2"
        assert llm.get_and_clear_reasoning_text() == "async1async2"
