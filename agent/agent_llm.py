"""Agent-specific LLM wrapper that preserves reasoning content for streaming.

Reasoning models like GPT-OSS 120B always emit ``reasoning_content``.  This subclass
extracts ``reasoning_content`` from both streaming chunks and non-streaming responses
and injects it into ``additional_kwargs`` so the agent runner can stream it in real time.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Iterator, List, Optional, Tuple, Type

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import AIMessageChunk, BaseMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

from langchain_openai.chat_models.base import _convert_chunk_to_generation_chunk

from agent.custom_llm import _extract_reasoning_from_chunk, _extract_reasoning_from_response

logger = logging.getLogger(__name__)


class ChatOpenAIAgentLLM(ChatOpenAI):
    """ChatOpenAI that preserves reasoning_content for agents.

    Use this LLM exclusively inside :class:`AgentRunner`.  Reasoning text is
    kept in ``additional_kwargs`` so the runner can stream it during iterations.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _build_stream_payload(
        self,
        messages: List[BaseMessageChunk],
        stop: Optional[List[str]] = None,
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
        """Convert raw chunk and inject reasoning_content into additional_kwargs."""
        generation_chunk = _convert_chunk_to_generation_chunk(
            raw_chunk,
            default_chunk_class,
            base_generation_info if is_first_chunk else {},
        )
        if generation_chunk is None:
            return None, default_chunk_class

        reasoning = _extract_reasoning_from_chunk(raw_chunk)
        if reasoning and hasattr(generation_chunk.message, "additional_kwargs"):
            ak = dict(generation_chunk.message.additional_kwargs or {})
            ak["reasoning_content"] = ak.get("reasoning_content", "") + reasoning
            generation_chunk.message.additional_kwargs = ak

        return generation_chunk, generation_chunk.message.__class__

    def _stream(
        self,
        messages: List[BaseMessageChunk],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
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

    async def _astream(
        self,
        messages: List[BaseMessageChunk],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
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
                ak["reasoning_content"] = reasoning
                message.additional_kwargs = ak
        return result
