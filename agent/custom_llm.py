"""Custom ChatOpenAI subclass that preserves reasoning_content from vLLM responses.

LangChain's standard ChatOpenAI targets only the official OpenAI API spec and
silently drops non-standard fields like ``reasoning_content`` returned by vLLM
and other OpenAI-compatible providers.

This subclass intercepts the raw streaming chunks, extracts ``reasoning_content``
from ``choices[0].delta.reasoning_content``, and injects it into
``AIMessageChunk.additional_kwargs`` so downstream extractors can find it.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, AsyncIterator, Iterator, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

from langchain_openai.chat_models.base import _convert_chunk_to_generation_chunk

_REASONING_KEY = "reasoning_content"
"""Key used by vLLM and other providers for reasoning text."""


def _extract_reasoning_from_chunk(chunk: dict) -> str:
    """Pull reasoning_content from a raw OpenAI-compatible chunk."""
    if not isinstance(chunk, dict):
        return ""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    delta = first_choice.get("delta")
    if not isinstance(delta, dict):
        return ""
    return delta.get(_REASONING_KEY, "") or delta.get("reasoning", "") or ""


def _extract_reasoning_from_response(response: Any) -> str:
    """Pull reasoning_content from a raw OpenAI-compatible non-streaming response."""
    if isinstance(response, dict):
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return ""
        message = first_choice.get("message")
        if not isinstance(message, dict):
            return ""
        return message.get(_REASONING_KEY, "") or message.get("reasoning", "") or ""
    # openai BaseModel response
    if hasattr(response, "choices") and response.choices:
        message = response.choices[0].message
        if hasattr(message, _REASONING_KEY):
            return getattr(message, _REASONING_KEY) or ""
        if hasattr(message, "reasoning"):
            return getattr(message, "reasoning") or ""
    return ""


class ChatOpenAIWithReasoning(ChatOpenAI):
    """ChatOpenAI that preserves reasoning_content from OpenAI-compatible providers.

    The reasoning accumulator is protected by a :py:class:`threading.Lock` so that
    parallel sync threads do not corrupt each other's data when the same cached LLM
    instance is reused.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._accumulated_reasoning_text: list[str] = []
        self._reasoning_lock = threading.Lock()

    def get_and_clear_reasoning_text(self) -> str:
        """Return accumulated reasoning text and clear the buffer."""
        with self._reasoning_lock:
            text = "".join(self._accumulated_reasoning_text)
            self._accumulated_reasoning_text = []
            return text

    def peek_reasoning_text(self) -> str:
        """Return accumulated reasoning text without clearing the buffer."""
        with self._reasoning_lock:
            return "".join(self._accumulated_reasoning_text)

    def _append_reasoning(self, reasoning: str) -> None:
        """Append reasoning text to the accumulator."""
        with self._reasoning_lock:
            self._accumulated_reasoning_text.append(reasoning)

    def _build_stream_payload(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]],
        **kwargs: Any,
    ) -> Tuple[dict, Type[BaseMessageChunk], dict]:
        """Prepare payload and metadata for a streaming request."""
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        default_chunk_class: Type[BaseMessageChunk] = AIMessageChunk
        base_generation_info: dict = {}
        return payload, default_chunk_class, base_generation_info

    def _process_chunk(
        self,
        raw_chunk: dict,
        *,
        is_first_chunk: bool,
        default_chunk_class: Type[BaseMessageChunk],
        base_generation_info: dict,
    ) -> Tuple[Optional[ChatGenerationChunk], Type[BaseMessageChunk]]:
        """Transform a raw API chunk into a ChatGenerationChunk with reasoning injected."""
        generation_chunk = _convert_chunk_to_generation_chunk(
            raw_chunk,
            default_chunk_class,
            base_generation_info if is_first_chunk else {},
        )
        if generation_chunk is None:
            return None, default_chunk_class

        reasoning = _extract_reasoning_from_chunk(raw_chunk)
        if reasoning:
            self._append_reasoning(reasoning)
            if hasattr(generation_chunk.message, "additional_kwargs"):
                ak = dict(generation_chunk.message.additional_kwargs or {})
                ak[_REASONING_KEY] = ak.get(_REASONING_KEY, "") + reasoning
                generation_chunk.message.additional_kwargs = ak

        return generation_chunk, generation_chunk.message.__class__

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        with self._reasoning_lock:
            self._accumulated_reasoning_text = []
        payload, default_chunk_class, base_generation_info = self._build_stream_payload(
            messages, stop, **kwargs
        )

        if self.include_response_headers:
            raw_response = self.client.with_raw_response.create(**payload)
            response = raw_response.parse()
            base_generation_info = {"headers": dict(raw_response.headers)}
        else:
            response = self.client.create(**payload)

        with response:
            is_first_chunk = True
            for chunk in response:
                if not isinstance(chunk, dict):
                    raw_chunk = chunk.model_dump()
                else:
                    raw_chunk = chunk

                generation_chunk, default_chunk_class = self._process_chunk(
                    raw_chunk,
                    is_first_chunk=is_first_chunk,
                    default_chunk_class=default_chunk_class,
                    base_generation_info=base_generation_info,
                )
                if generation_chunk is None:
                    continue

                logprobs = (generation_chunk.generation_info or {}).get("logprobs")
                if run_manager:
                    run_manager.on_llm_new_token(
                        generation_chunk.text,
                        chunk=generation_chunk,
                        logprobs=logprobs,
                    )
                is_first_chunk = False
                yield generation_chunk

    def _create_chat_result(
        self,
        response: Any,
        generation_info: Optional[dict] = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        reasoning = _extract_reasoning_from_response(response)
        if reasoning and result.generations:
            message = result.generations[0].message
            if hasattr(message, "additional_kwargs"):
                ak = dict(message.additional_kwargs or {})
                ak[_REASONING_KEY] = reasoning
                message.additional_kwargs = ak
        return result

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        with self._reasoning_lock:
            self._accumulated_reasoning_text = []
        payload, default_chunk_class, base_generation_info = self._build_stream_payload(
            messages, stop, **kwargs
        )

        if self.include_response_headers:
            raw_response = await self.async_client.with_raw_response.create(**payload)
            response = raw_response.parse()
            base_generation_info = {"headers": dict(raw_response.headers)}
        else:
            response = await self.async_client.create(**payload)

        async with response:
            is_first_chunk = True
            async for chunk in response:
                if not isinstance(chunk, dict):
                    raw_chunk = chunk.model_dump()
                else:
                    raw_chunk = chunk

                generation_chunk, default_chunk_class = self._process_chunk(
                    raw_chunk,
                    is_first_chunk=is_first_chunk,
                    default_chunk_class=default_chunk_class,
                    base_generation_info=base_generation_info,
                )
                if generation_chunk is None:
                    continue

                logprobs = (generation_chunk.generation_info or {}).get("logprobs")
                if run_manager:
                    await run_manager.on_llm_new_token(
                        generation_chunk.text,
                        chunk=generation_chunk,
                        logprobs=logprobs,
                    )
                is_first_chunk = False
                yield generation_chunk
