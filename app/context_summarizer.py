from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.chat_utils import extract_answer_text, extract_usage_metadata
from app.history_budget import count_text_tokens, _encoding_for_model

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Fasse den folgenden Inhalt kompakt auf Deutsch zusammen. "
        "Bewahre: Fakten, Entscheidungen, Nutzerpräferenzen, offene Aufgaben, wichtige Einschränkungen, Quellenangaben und relevante Sicherheitskontexte. "
        "Lass weg: Smalltalk, Höflichkeitsfloskeln, Wiederholungen, irrelevante Details. "
        "Formuliere neutral als Kontextnotiz — keine direkte Rede, keine Anrede. "
        "Erfinde nichts. Sei so knapp wie möglich, aber verliere keine wesentlichen Informationen.",
    ),
    ("human", "{content}"),
])


async def summarize_context_if_needed(
    text: str,
    *,
    token_budget: int,
    model: str,
    timeout_seconds: float = 30.0,
    summary_llm_factory: Callable[[], BaseChatModel | None]
    | Callable[[], Awaitable[BaseChatModel | None]]
    | None = None,
    summary_context_window: int = 0,
    usage_sink: Optional[List[Dict[str, Any]]] = None,
) -> tuple[str, bool]:
    """Summarize *text* via LLM if it exceeds *token_budget*.

    Returns ``(text, summarized)`` where *summarized* indicates whether the
    text was actually condensed by the LLM.  If the text fits within the
    budget it is returned unchanged.  If the LLM is unavailable or fails, the
    text is hard-truncated to *token_budget* as a fallback.

    If *summary_context_window* is set and the input is too large for a single
    LLM call, a map-reduce strategy is used: the text is split into chunks,
    each chunk is summarized independently, and the results are merged.

    When *usage_sink* is a list, every LLM call's extracted usage metadata is
    appended to it so callers can charge compression tokens to the run total.
    """
    if not text:
        return text, False

    token_count = count_text_tokens(text, model)
    if token_count <= token_budget:
        return text, False

    if summary_llm_factory is None:
        return _hard_truncate(text, token_budget, model), False

    # Determine the effective context window for the summary LLM.
    # If not provided, assume 4x the token budget as a heuristic.
    eff_ctx = summary_context_window if summary_context_window > 0 else max(2048, token_budget * 4)

    # If the input is too large for a single LLM call, use map-reduce.
    if token_count > eff_ctx * 0.6:
        return await _map_reduce_summarize(
            text,
            token_budget=token_budget,
            model=model,
            timeout_seconds=timeout_seconds,
            summary_llm_factory=summary_llm_factory,
            summary_context_window=eff_ctx,
            usage_sink=usage_sink,
        )

    llm = await _resolve_llm(summary_llm_factory)
    if llm is None:
        return _hard_truncate(text, token_budget, model), False

    result = await _invoke_summary_llm(llm, text, timeout_seconds)
    _record_usage(result, usage_sink)
    if result is None:
        return _hard_truncate(text, token_budget, model), False

    summary = extract_answer_text(result).strip()
    if not summary:
        logger.warning("Context summarization returned empty output; falling back to truncation.")
        return _hard_truncate(text, token_budget, model), False

    return summary, True


async def _resolve_llm(summary_llm_factory):
    """Call factory and await if necessary."""
    llm = summary_llm_factory()
    if inspect.isawaitable(llm):
        llm = await llm
    return llm


async def _invoke_summary_llm(llm, text: str, timeout_seconds: float):
    """Invoke the summary LLM with the given text, handling pipe and fallback."""
    chain = None
    try:
        chain = _SUMMARY_PROMPT | llm
    except (AttributeError, TypeError):
        chain = None

    try:
        if chain is not None:
            result = await asyncio.wait_for(
                chain.ainvoke({"content": text}),
                timeout=timeout_seconds,
            )
        else:
            formatted = _SUMMARY_PROMPT.format_messages(content=text)
            result = await asyncio.wait_for(
                llm.ainvoke(formatted),
                timeout=timeout_seconds,
            )
    except asyncio.TimeoutError:
        logger.warning("Context summarization timed out after %.1fs", timeout_seconds)
        return None
    except Exception as exc:
        logger.warning("Context summarization failed: %r", exc)
        return None
    return result


def _record_usage(result: Any, usage_sink: Optional[List[Dict[str, Any]]]) -> None:
    """Append extracted usage metadata of one summary LLM call to *usage_sink*."""
    if usage_sink is None or result is None:
        return
    usage = extract_usage_metadata(result)
    if usage:
        usage_sink.append(usage)


async def _map_reduce_summarize(
    text: str,
    *,
    token_budget: int,
    model: str,
    timeout_seconds: float,
    summary_llm_factory,
    summary_context_window: int,
    usage_sink: Optional[List[Dict[str, Any]]] = None,
) -> tuple[str, bool]:
    """Split text into chunks, summarize each, then merge and optionally re-summarize."""
    chunk_token_size = max(256, int(summary_context_window * 0.3))
    chunks = _split_text_for_summary(text, chunk_token_size, model)
    logger.info("Map-reduce summarization: %d chunks of ~%d tokens each", len(chunks), chunk_token_size)

    llm = await _resolve_llm(summary_llm_factory)
    if llm is None:
        return _hard_truncate(text, token_budget, model), False

    # Summarize each chunk in parallel
    async def _summarize_chunk(chunk: str) -> str:
        result = await _invoke_summary_llm(llm, chunk, timeout_seconds)
        _record_usage(result, usage_sink)
        if result is not None:
            summary = extract_answer_text(result).strip()
            if summary:
                return summary
        return _hard_truncate(chunk, chunk_token_size // 2, model)

    chunk_summaries = list(await asyncio.gather(*[_summarize_chunk(c) for c in chunks]))
    chunk_summaries = [s for s in chunk_summaries if s]

    if not chunk_summaries:
        return _hard_truncate(text, token_budget, model), False

    merged = "\n\n".join(chunk_summaries)
    merged_tokens = count_text_tokens(merged, model)

    if merged_tokens <= token_budget:
        return merged, True

    # Merged result is still too large — try one more summarization pass
    if merged_tokens <= summary_context_window * 0.6:
        result = await _invoke_summary_llm(llm, merged, timeout_seconds)
        _record_usage(result, usage_sink)
        if result is not None:
            final_summary = extract_answer_text(result).strip()
            if final_summary and count_text_tokens(final_summary, model) <= token_budget:
                return final_summary, True

    # Last resort: hard-truncate the merged result
    return _hard_truncate(merged, token_budget, model), True


def _hard_truncate(text: str, token_budget: int, model: str) -> str:
    encoding = _encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) <= token_budget:
        return text
    return encoding.decode(tokens[:token_budget]).strip()


def _split_text_for_summary(text: str, chunk_token_size: int, model: str) -> list[str]:
    """Split *text* into chunks of approximately *chunk_token_size* tokens.

    Splits on paragraph boundaries first, then on sentence boundaries, to keep
    chunks semantically coherent.
    """
    encoding = _encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) <= chunk_token_size:
        return [text]

    # Decode in chunks, trying to break on paragraph/sentence boundaries
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_token_size, len(tokens))
        chunk_text = encoding.decode(tokens[start:end])

        # Try to break at a paragraph boundary
        if end < len(tokens):
            para_break = chunk_text.rfind("\n\n")
            if para_break > chunk_token_size // 2:
                chunk_text = chunk_text[:para_break]
                end = start + len(encoding.encode(chunk_text))
            else:
                # Try sentence boundary
                sent_break = chunk_text.rfind(". ")
                if sent_break > chunk_token_size // 2:
                    chunk_text = chunk_text[:sent_break + 1]
                    end = start + len(encoding.encode(chunk_text))

        chunks.append(chunk_text.strip())
        start = end

    return [c for c in chunks if c]
