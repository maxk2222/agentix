"""Utilities shared by the OpenAI-compatible chat router and metadata helpers.

The functions defined here perform lossless conversions between OpenAI wire payloads and
LangChain primitives, implement lightweight heuristics for routing special metadata-only
prompts, and provide resilient fallbacks for title/tag generation so the API remains
useful even when the upstream LLM produces empty results.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
import asyncio
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import orjson
import tiktoken
import yaml

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)
from langchain_core.messages.base import BaseMessageChunk
from langchain_core.prompts import ChatPromptTemplate

from app.reasoning_params import normalize_reasoning_dict

logger = logging.getLogger(__name__)

# Upper bound for a plausible token generation rate — anything above is a
# timing artifact (sub-millisecond durations) and must be dropped, not shown.
MAX_PLAUSIBLE_TPS = 2000.0

__all__ = [
    "classify_message",
    "convert_openai_messages",
    "build_multimodal_content",
    "split_history_and_latest_user",
    "limit_chat_history",
    "stringify_message_content",
    "extract_answer_text",
    "generate_title",
    "generate_tags",
    "parse_chat_history_block",
    "load_instructions",
    "count_tokens",
    "extract_usage_metadata",
    "extract_reasoning_text",
    "render_system_instructions",
]


_GERMAN_WEEKDAYS = [
    "Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"
]
_GERMAN_MONTHS = [
    "Januar", "Februar", "März", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Dezember",
]


def _format_current_datetime_german() -> str:
    """Return the current date/time as a German long-form string."""
    now = datetime.now(ZoneInfo("Europe/Berlin"))
    weekday = _GERMAN_WEEKDAYS[now.weekday()]
    month = _GERMAN_MONTHS[now.month - 1]
    return f"{weekday}, {now.day}. {month} {now.year}, {now.hour:02d}:{now.minute:02d} Uhr"


def render_system_instructions(text: str, *, rag_context: str = "", agent_memory: Optional[str] = None, keep_agent_memory_placeholder: bool = False) -> str:
    """Replace dynamic placeholders in system instructions with live values.

    Supported placeholders:
    - ``{{current_datetime}}`` → e.g. *Mittwoch, 11. Juni 2026, 20:17 Uhr*
    - ``{{context}}`` → replaced with the provided *rag_context* string
    - ``{{agent_memory}}`` → replaced with the provided *agent_memory* string.
      With ``keep_agent_memory_placeholder=True`` the placeholder survives so
      the agent runner can substitute fresh memory on every iteration.
    """
    if not text:
        text = ""
    has_datetime = "{{current_datetime}}" in text or "{{current\\_datetime}}" in text
    has_context = "{{context}}" in text or "{{context}}" in text
    has_agent_memory = "{{agent_memory}}" in text
    text = text.replace("{{current_datetime}}", _format_current_datetime_german())
    text = text.replace("{{current\\_datetime}}", _format_current_datetime_german())
    if rag_context:
        text = text.replace("{{context}}", rag_context)
    if has_agent_memory and not keep_agent_memory_placeholder:
        text = text.replace("{{agent_memory}}", agent_memory or "")
    logger.debug("[render_system_instructions] has_datetime=%s has_context=%s has_agent_memory=%s input_len=%d output_len=%d", has_datetime, bool(rag_context and has_context), has_agent_memory, len(text), len(text))
    return text


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Approximate token count using tiktoken. Falls back to length/4 heuristic."""
    try:
        enc = tiktoken.get_encoding(model)
        return len(enc.encode(text or ""))
    except Exception:
        logger.debug("tiktoken failed for %s, falling back to heuristic", model)
        return max(1, len(text or "") // 4)


def estimate_rag_context_tokens(docs: Sequence[Any]) -> tuple[int, list[int]]:
    """Calculate estimated tokens for retrieved RAG documents.

    Returns (total_tokens, per_doc_tokens).
    """
    total_tokens = 0
    per_doc_tokens: list[int] = []
    for doc in docs:
        content = getattr(doc, "page_content", "") or ""
        tokens = count_tokens(content)
        per_doc_tokens.append(tokens)
        total_tokens += tokens
    return total_tokens, per_doc_tokens


def estimate_full_prompt_tokens(
    system_instructions: str,
    chat_history: Sequence[Any],
    rag_context_tokens: int,
    user_message: str,
) -> int:
    """Estimate total prompt tokens including system, history, RAG context and user message."""
    tokens = count_tokens(system_instructions)
    for msg in chat_history:
        tokens += count_tokens(str(msg.content) if hasattr(msg, "content") else str(msg))
    tokens += rag_context_tokens
    tokens += count_tokens(user_message)
    # rough format overhead per message pair
    tokens += len(chat_history) * 4
    return tokens


def extract_usage_metadata(result: Any) -> Optional[Dict[str, Any]]:
    """Extract token usage from LangChain result objects (robust across providers).

    Tries multiple common paths where usage data may live:
      - ``response_metadata["token_usage"]`` (OpenAI-style)
      - ``usage_metadata`` (LangChain 0.3.x native)

    Returns a dict with ``prompt_tokens``, ``completion_tokens``,
    ``total_tokens``, and ``reasoning_tokens`` when found, otherwise ``None``.
    """
    if result is None:
        return None

    # Try to locate usage metadata from various shapes
    usage_candidates: List[Any] = []

    if isinstance(result, dict):
        # OpenAI-style in response_metadata
        resp_meta = result.get("response_metadata")
        if isinstance(resp_meta, dict):
            token_usage = resp_meta.get("token_usage")
            if isinstance(token_usage, dict):
                usage_candidates.append(token_usage)
        # LangChain 0.3.x usage_metadata directly on dict
        um = result.get("usage_metadata")
        if isinstance(um, dict):
            usage_candidates.append(um)
    elif hasattr(result, "response_metadata"):
        resp_meta = getattr(result, "response_metadata", None)
        if isinstance(resp_meta, dict):
            token_usage = resp_meta.get("token_usage")
            if isinstance(token_usage, dict):
                usage_candidates.append(token_usage)
        um = getattr(result, "usage_metadata", None)
        if isinstance(um, dict):
            usage_candidates.append(um)

    if not usage_candidates:
        return None

    for candidate in usage_candidates:
        prompt_tokens = candidate.get("prompt_tokens") if "prompt_tokens" in candidate else candidate.get("input_tokens")
        completion_tokens = candidate.get("completion_tokens") if "completion_tokens" in candidate else candidate.get("output_tokens")
        total_tokens = candidate.get("total_tokens")
        reasoning_tokens = candidate.get("reasoning_tokens")
        # OpenAI completion_tokens_details.reasoning_tokens path
        if reasoning_tokens is None and "completion_tokens_details" in candidate:
            details = candidate["completion_tokens_details"]
            if isinstance(details, dict):
                reasoning_tokens = details.get("reasoning_tokens")
        # LangChain output_token_details.reasoning path
        if reasoning_tokens is None and "output_token_details" in candidate:
            details = candidate["output_token_details"]
            if isinstance(details, dict):
                reasoning_tokens = details.get("reasoning")
        # Anthropic-style cache tokens live in input_token_details and are NOT
        # part of input_tokens — collect them separately so totals can include
        # them. OpenAI's prompt_tokens_details.cached_tokens is already inside
        # prompt_tokens, so it is intentionally ignored to avoid double counting.
        cache_tokens = 0
        input_details = candidate.get("input_token_details")
        if isinstance(input_details, dict):
            cache_tokens = (_coerce_int(input_details.get("cache_read")) or 0) + (
                _coerce_int(input_details.get("cache_creation")) or 0
            )

        if prompt_tokens is not None or completion_tokens is not None or total_tokens is not None:
            return {
                "prompt_tokens": _coerce_int(prompt_tokens),
                "completion_tokens": _coerce_int(completion_tokens),
                "total_tokens": _coerce_int(total_tokens),
                "reasoning_tokens": _coerce_int(reasoning_tokens),
                "cache_tokens": cache_tokens or None,
            }

    return None


def _coerce_int(value: Any) -> Optional[int]:
    """Coerce a value to int, returning None on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def finalize_token_fields(
    *,
    prompt: Optional[int] = None,
    completion: Optional[int] = None,
    total: Optional[int] = None,
    reasoning: Optional[int] = None,
    cache: Optional[int] = None,
    est_answer: Optional[int] = None,
    est_reasoning: Optional[int] = None,
) -> Dict[str, Optional[int]]:
    """Normalise provider/estimated token counts into one consistent shape.

    Provider semantics (OpenAI / LangChain usage_metadata): ``completion_tokens``
    already contains the reasoning tokens. Estimated fallbacks (tiktoken counts
    of the produced text) only measure the visible answer — this helper lifts
    them onto the same convention so invariants hold:

        reasoning <= completion <= total
        answer  =  completion - reasoning
        total  >=  prompt + completion + cache

    ``cache`` covers provider cache-read/creation tokens that sit outside
    ``prompt_tokens`` (Anthropic shape). ``est_answer``/``est_reasoning`` are
    only used when the provider did not report the respective field.
    """
    reasoning = _coerce_int(reasoning) or _coerce_int(est_reasoning)
    completion = _coerce_int(completion)
    if completion is None:
        est_answer_i = _coerce_int(est_answer)
        if est_answer_i is not None or reasoning is not None:
            completion = (est_answer_i or 0) + (reasoning or 0)
    if completion is not None and reasoning is not None:
        reasoning = min(reasoning, completion)
    answer: Optional[int] = None
    if completion is not None:
        answer = max(completion - (reasoning or 0), 0)
    else:
        answer = _coerce_int(est_answer)
    prompt = _coerce_int(prompt)
    total = _coerce_int(total)
    cache = _coerce_int(cache)
    if prompt is not None and completion is not None:
        total = max(total or 0, prompt + completion + (cache or 0))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "reasoning_tokens": reasoning,
        "answer_tokens": answer,
        "cache_tokens": cache,
    }


def aux_usage_totals(retrieval_info: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Sum auxiliary usage dicts recorded outside the LLM iteration loop —
    e.g. RAG context compression via ``retrieval_info["_compression_usage"]``.
    Returns zeroed prompt/completion/total/reasoning/cache counters."""
    out = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0, "cache_tokens": 0}
    entries = retrieval_info.get("_compression_usage") if isinstance(retrieval_info, dict) else None
    for usage in entries or []:
        if not isinstance(usage, dict):
            continue
        for key in out:
            value = usage.get(key)
            if isinstance(value, (int, float)) and value:
                out[key] += int(value)
    return out


def build_message_metrics(
    *,
    retrieval_info: Optional[Dict[str, Any]],
    assistant_text: str,
    reasoning_text: Optional[str],
    response_ms: Optional[float],
    prompt_estimate: Optional[int] = None,
    usage_meta: Optional[Dict[str, Any]] = None,
    stream_start: Optional[float] = None,
    first_token_at: Optional[float] = None,
    first_answer_token_at: Optional[float] = None,
    first_reasoning_token_at: Optional[float] = None,
    subagent_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Per-message metrics payload — emitted in the SSE trailer and persisted
    as ``retrieval_info["_msg_metrics"]`` so reloads render the same badge.

    ``total_tokens`` covers the whole run (all LLM calls incl. subagents,
    observer, compression and cache tokens); ``final_*`` fields isolate the
    last answer-generating call so the UI can show "final vs. total".
    """
    if usage_meta is None:
        usage_meta = retrieval_info.get("_llm_usage_metadata") if isinstance(retrieval_info, dict) else None
    final_usage = None
    if isinstance(retrieval_info, dict):
        final_usage = retrieval_info.get("_final_answer_usage")
    if not isinstance(final_usage, dict) and isinstance(usage_meta, dict):
        final_usage = usage_meta.get("final_answer_usage")

    m_prompt = usage_meta.get("prompt_tokens") if isinstance(usage_meta, dict) else None
    m_completion = usage_meta.get("completion_tokens") if isinstance(usage_meta, dict) else None
    m_total = usage_meta.get("total_tokens") if isinstance(usage_meta, dict) else None
    m_reasoning = usage_meta.get("reasoning_tokens") if isinstance(usage_meta, dict) else None
    m_cache = usage_meta.get("cache_tokens") if isinstance(usage_meta, dict) else None
    # App-level compression (RAG context summarization) lives outside
    # usage_meta — fold it into the totals so "total" really means everything.
    aux = aux_usage_totals(retrieval_info)
    aux_compression = aux["total_tokens"] or (aux["prompt_tokens"] + aux["completion_tokens"])
    if aux_compression:
        m_prompt = (m_prompt or 0) + aux["prompt_tokens"]
        m_completion = (m_completion or 0) + aux["completion_tokens"]
        m_total = (m_total or 0) + aux["total_tokens"]
        m_reasoning = (m_reasoning or 0) + aux["reasoning_tokens"]
        m_cache = (m_cache or 0) + aux["cache_tokens"]
    if m_prompt is None:
        m_prompt = prompt_estimate
    m_fields = finalize_token_fields(
        prompt=m_prompt,
        completion=m_completion,
        total=m_total,
        reasoning=m_reasoning,
        cache=m_cache,
        est_answer=count_tokens(assistant_text or "") if m_completion is None else None,
        est_reasoning=count_tokens(reasoning_text) if (reasoning_text and not m_reasoning) else None,
    )
    m_answer = m_fields["answer_tokens"]

    # tokens/s is the run's output generation rate (completion over LLM-active
    # time, computed by the runner). Fall back to the final answer's call rate
    # and lastly to answer-over-runtime when no timing data exists.
    tps: Optional[float] = None
    if isinstance(usage_meta, dict):
        agg_tps = usage_meta.get("tokens_per_second")
        if isinstance(agg_tps, (int, float)) and 0 < agg_tps <= MAX_PLAUSIBLE_TPS:
            tps = float(agg_tps)
    if tps is None and isinstance(final_usage, dict):
        fa_completion = final_usage.get("completion_tokens")
        fa_ms = final_usage.get("_response_time_ms")
        # >=1ms gate: a real LLM call never completes sub-millisecond —
        # corrupted timing would explode the displayed rate.
        if fa_completion and fa_ms and fa_ms >= 1:
            tps = fa_completion / (fa_ms / 1000.0)
            if tps > MAX_PLAUSIBLE_TPS:
                tps = None
    if tps is None and (m_answer or m_completion) and response_ms and response_ms >= 1:
        tps = (m_answer or m_completion) / (response_ms / 1000.0)
        if tps > MAX_PLAUSIBLE_TPS:
            tps = None

    def _tt_ms(marker: Optional[float]) -> Optional[float]:
        if marker is None or stream_start is None:
            return None
        return round((marker - stream_start) * 1000, 1)

    metrics: Dict[str, Any] = {
        "prompt_tokens": m_fields["prompt_tokens"],
        "completion_tokens": m_fields["completion_tokens"],
        "total_tokens": m_fields["total_tokens"],
        "reasoning_tokens": m_fields["reasoning_tokens"],
        "answer_tokens": m_answer,
        "cache_tokens": m_fields["cache_tokens"],
        "tool_result_tokens": usage_meta.get("tool_result_tokens") if isinstance(usage_meta, dict) else None,
        "rag_context_tokens": int(retrieval_info.get("rag_context_tokens", 0) or 0) if isinstance(retrieval_info, dict) else None,
        "observer_tokens": usage_meta.get("observer_tokens") if isinstance(usage_meta, dict) else None,
        "compression_tokens": ((usage_meta.get("compression_tokens") or 0) if isinstance(usage_meta, dict) else 0) + aux_compression or None,
        "subagent_tokens": usage_meta.get("subagent_tokens") if isinstance(usage_meta, dict) else None,
        "final_prompt_tokens": final_usage.get("prompt_tokens") if isinstance(final_usage, dict) else None,
        "final_completion_tokens": final_usage.get("completion_tokens") if isinstance(final_usage, dict) else None,
        "final_reasoning_tokens": final_usage.get("reasoning_tokens") if isinstance(final_usage, dict) else None,
        "final_answer_ms": round(final_usage["_response_time_ms"], 1) if isinstance(final_usage, dict) and final_usage.get("_response_time_ms") else None,
        "response_time_ms": round(response_ms, 1) if response_ms is not None else None,
        "time_to_first_token_ms": _tt_ms(first_token_at),
        "time_to_first_answer_token_ms": _tt_ms(first_answer_token_at),
        "time_to_first_reasoning_token_ms": _tt_ms(first_reasoning_token_at),
        "tokens_per_second": round(tps, 2) if tps is not None else None,
        "agent_iterations": usage_meta.get("iterations") if isinstance(usage_meta, dict) else None,
        "llm_calls": (
            (usage_meta.get("iterations") or 0 if isinstance(usage_meta, dict) else 0)
            + (usage_meta.get("compression_calls") or 0 if isinstance(usage_meta, dict) else 0)
            + (usage_meta.get("observer_calls") or 0 if isinstance(usage_meta, dict) else 0)
            + len(retrieval_info.get("_compression_usage") or [] if isinstance(retrieval_info, dict) else [])
        ) or None,
        "subagent_count": subagent_count,
        # Per-LLM-call token series for the badge's mini column chart.
        "per_call": usage_meta.get("per_call") if isinstance(usage_meta, dict) else None,
    }
    if isinstance(usage_meta, dict) and usage_meta.get("estimated_iterations"):
        metrics["estimated"] = True
    return metrics


def collect_tool_names(calls: Any) -> Optional[List[str]]:
    """Return the sorted, deduplicated tool names from a tool-call list.

    Handles both normalised entries (``{"name": ...}``) and agent-trace
    records (``{"tool_name": ...}``). Returns ``None`` when empty.
    """
    if not calls:
        return None
    names = {
        str(call.get("name") or call.get("tool_name"))
        for call in calls
        if isinstance(call, dict) and (call.get("name") or call.get("tool_name"))
    }
    return sorted(names) or None


_CHAT_HISTORY_BLOCK = re.compile(r"<chat_history>(.*?)</chat_history>", re.IGNORECASE | re.DOTALL)
_CHAT_ENTRY = re.compile(
    r"(?is)(user|assistant)\s*:\s*(.*?)(?=(?:\n(?:user|assistant)\s*:)|\Z)"
)



def classify_message(content: str) -> str:
    """Heuristically label a request as ``"chat"``, ``"title"``, or ``"tags"``.

    OpenWebUI and similar clients send natural-language instructions when they only need
    metadata.  Recognizing these phrases lets the router short-circuit the full RAG
    pipeline and answer with cheaper helper routines.
    """
    content_lower = content.lower()
    if "generate a concise, 3-5 word title" in content_lower:
        return "title"
    if "generate 1-3 broad tags" in content_lower:
        return "tags"
    return "chat"


def stringify_message_content(content: Any) -> str:
    """Normalize OpenAI message content into a deterministic string.

    The payload may contain nested lists and dicts (function call blocks, tool outputs,
    etc.).  This helper extracts all human text in-order so prompts, logs, and metadata
    routines can treat the result as plain UTF-8.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if part is None:
                continue
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                # Skip image_url parts — only extract text for stringification
                if part.get("type") == "image_url":
                    continue
                text = part.get("text")
                if text:
                    parts.append(str(text))
        return "".join(parts)
    if isinstance(content, dict):
        text = content.get("text")
        if text:
            return stringify_message_content(text)
    return str(content)


def _coerce_message_to_text(message: BaseMessage | BaseMessageChunk | None) -> str:
    """Convert LangChain message objects into plain text without raising on ``None``."""
    if message is None:
        return ""
    return stringify_message_content(message.content)


def extract_answer_text(payload: Any) -> str:
    """Pull the human-readable answer out of arbitrarily nested LangChain outputs.

    Handles dicts, ``BaseMessage`` instances, and raw lists so streaming and
    non-streaming call sites can share a single normalization helper.
    """
    if payload is None:
        return ""
    if isinstance(payload, (BaseMessage, BaseMessageChunk)):
        return _coerce_message_to_text(payload)
    if isinstance(payload, dict):
        for key in ("answer", "output", "output_text", "content", "text", "message"):
            if key in payload and payload[key]:
                result = extract_answer_text(payload[key])
                if result:
                    return result
        return ""
    if isinstance(payload, list):
        return "".join(extract_answer_text(item) for item in payload if item)
    return str(payload)


def extract_reasoning_text(payload: Any) -> str:
    """Extract reasoning/thinking text from LangChain outputs.

    Looks in ``additional_kwargs`` (OpenAI-compatible / vLLM) and
    ``response_metadata.message`` for ``reasoning_content`` or ``reasoning``.
    Returns the raw text without touching normal message content.
    """
    if payload is None:
        return ""

    # Attribute access (BaseMessage / BaseMessageChunk)
    if hasattr(payload, "additional_kwargs"):
        ak = getattr(payload, "additional_kwargs", None)
        if isinstance(ak, dict):
            for key in ("reasoning_content", "reasoning"):
                if key in ak and ak[key]:
                    text = stringify_message_content(ak[key])
                    if text:
                        return text

    # ChatGenerationChunk has reasoning in message.additional_kwargs
    if hasattr(payload, "message"):
        msg = getattr(payload, "message", None)
        if msg is not None and hasattr(msg, "additional_kwargs"):
            ak = getattr(msg, "additional_kwargs", None)
            if isinstance(ak, dict):
                for key in ("reasoning_content", "reasoning"):
                    if key in ak and ak[key]:
                        text = stringify_message_content(ak[key])
                        if text:
                            return text

    # Dict access
    if isinstance(payload, dict):
        for key in ("reasoning_content", "reasoning"):
            if key in payload and payload[key]:
                text = stringify_message_content(payload[key])
                if text:
                    return text
        ak = payload.get("additional_kwargs")
        if isinstance(ak, dict):
            for key in ("reasoning_content", "reasoning"):
                if key in ak and ak[key]:
                    text = stringify_message_content(ak[key])
                    if text:
                        return text
        resp_meta = payload.get("response_metadata")
        if isinstance(resp_meta, dict):
            # response_metadata.message.reasoning_content
            msg = resp_meta.get("message")
            if isinstance(msg, dict):
                for key in ("reasoning_content", "reasoning"):
                    if key in msg and msg[key]:
                        text = stringify_message_content(msg[key])
                        if text:
                            return text
            # response_metadata.reasoning_content directly
            for key in ("reasoning_content", "reasoning"):
                if key in resp_meta and resp_meta[key]:
                    text = stringify_message_content(resp_meta[key])
                    if text:
                        return text
            # response_metadata.delta.reasoning_content (streaming chunk shape)
            delta = resp_meta.get("delta")
            if isinstance(delta, dict):
                for key in ("reasoning_content", "reasoning"):
                    if key in delta and delta[key]:
                        text = stringify_message_content(delta[key])
                        if text:
                            return text

    return ""


def parse_chat_history_block(text: str) -> List[BaseMessage]:
    """Convert ``<chat_history>`` markup back into LangChain ``BaseMessage`` objects.

    The metadata prompts embed chat logs inside a single string; this reverses the
    process so the normal title/tag generators can operate on structured history.
    """
    if not text:
        return []
    match = _CHAT_HISTORY_BLOCK.search(text)
    if not match:
        return []
    block = match.group(1).strip()
    if not block:
        return []

    messages: List[BaseMessage] = []
    for speaker, content in _CHAT_ENTRY.findall(block):
        content_clean = stringify_message_content(content).strip()
        if not content_clean:
            continue
        if speaker.lower() == "assistant":
            messages.append(AIMessage(content=content_clean))
        else:
            messages.append(HumanMessage(content=content_clean))
    return messages


def _has_multimodal_content(content: Any) -> bool:
    """Check if content is a list containing image_url parts."""
    if not isinstance(content, list):
        return False
    return any(
        isinstance(part, dict) and part.get("type") == "image_url"
        for part in content
    )


def convert_openai_messages(messages: Sequence[Any]) -> List[BaseMessage]:
    """Translate OpenAI-style payloads into LangChain ``BaseMessage`` instances.

    Supports both dict-based inputs (raw JSON) and attribute access (Pydantic models)
    while preserving tool/system roles. Multimodal content (lists with image_url
    parts) is passed through as-is for user messages.
    """
    def _coerce(role: str, content: Any) -> BaseMessage:
        role_normalized = (role or "user").lower()
        if role_normalized == "assistant":
            return AIMessage(content=stringify_message_content(content))
        if role_normalized == "system":
            return SystemMessage(content=stringify_message_content(content))
        # For user messages: pass through multimodal content as-is
        if _has_multimodal_content(content):
            return HumanMessage(content=content)
        return HumanMessage(content=stringify_message_content(content))

    converted: List[BaseMessage] = []
    for message in messages:
        if message is None:
            continue
        if hasattr(message, "role") and hasattr(message, "content"):
            converted.append(_coerce(message.role, message.content))
        else:
            role = str(message.get("role", "user"))
            converted.append(_coerce(role, message.get("content")))
    return converted


def build_multimodal_content(text: str, images: list[str]) -> list[dict]:
    """Build OpenAI-compatible multimodal content array from text and image data URIs.

    Each image should be a Base64 data-URI (``data:image/...;base64,...``) or a raw
    Base64 string (will be prefixed with a generic image/png data-URI).
    """
    content: list[dict] = []
    if text:
        content.append({"type": "text", "text": text})
    for image_data in images:
        if image_data.startswith("data:"):
            url = image_data
        else:
            url = f"data:image/png;base64,{image_data}"
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content


def split_history_and_latest_user(
    messages: Sequence[BaseMessage],
) -> Tuple[List[BaseMessage], HumanMessage]:
    """Return historical context and the final user utterance, validating the speaker."""
    if not messages:
        raise ValueError("No messages provided")
    latest = messages[-1]
    if not isinstance(latest, HumanMessage):
        raise ValueError("The final message must come from the user")
    history = list(messages[:-1])
    return history, latest


def limit_chat_history(
    history: Sequence[BaseMessage],
    *,
    max_messages: int,
    keep_system_message: bool = True,
) -> List[BaseMessage]:
    """Truncate chat history to bound prompt size.

    The latest user message is handled separately by the router; this function only
    limits the *prior* messages.

    Strategy:
    - keep at most one system message (the last one) when enabled
    - keep the last ``max_messages`` non-system messages
    - preserve the relative order (system first)
    """
    max_messages = int(max_messages)
    if max_messages <= 0:
        return list(history)

    system_messages: List[SystemMessage] = []
    non_system: List[BaseMessage] = []
    for message in history:
        if isinstance(message, SystemMessage):
            system_messages.append(message)
        else:
            non_system.append(message)

    trimmed_non_system = non_system[-max_messages:]

    limited: List[BaseMessage] = []
    if keep_system_message and system_messages:
        limited.append(system_messages[-1])
    limited.extend(trimmed_non_system)
    return limited


def _build_metadata_prompt(system_prompt: str) -> ChatPromptTemplate:
    """Create a standard two-turn prompt that pipes the chat history into a metadata LLM.

    Having a single builder keeps the title/tag generators consistent and makes it easy
    to adjust wording for all metadata requests in one place.
    """
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{chat}"),
    ])


def _fallback_title_from_history(history: Sequence[BaseMessage]) -> str:
    """Synthesize a short title from the most recent user message as a safe fallback."""
    for message in reversed(history):
        if isinstance(message, HumanMessage):
            text = stringify_message_content(message.content).strip()
            if text:
                words = text.split()
                snippet = " ".join(words[:5])
                break
    else:
        snippet = "Allgemeiner Chat"
    snippet = snippet or "Allgemeiner Chat"
    return f"💬 {snippet}"


_TAG_KEYWORDS: List[Tuple[Tuple[str, ...], str]] = [
    (("cyber", "security", "sicherheit"), "Cybersecurity"),
    (("entwickler", "developer", "coding"), "Software Development"),
    (("phishing",), "Phishing"),
    (("ransom", "erpress"), "Ransomware"),
    (("hello", "hallo", "hi"), "Greeting"),
]


def _fallback_tags_from_history(history: Sequence[BaseMessage]) -> List[str]:
    """Derive up to three tags via keyword heuristics when the metadata LLM fails."""
    text = " ".join(
        stringify_message_content(message.content) or "" for message in history
    ).lower()
    tags: List[str] = []
    for keywords, tag in _TAG_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            tags.append(tag)
    if not tags:
        tags.append("General")
    return tags[:3]


_TITLE_PLACEHOLDER_PHRASES: Tuple[str, ...] = (
    "kein inhalt",
    "keinen inhalt",
    "leere nachricht",
    "leer",
    "empty",
    "no content",
    "no message",
)

_TAG_PLACEHOLDER_PHRASES: Tuple[str, ...] = (
    "no content",
    "kein inhalt",
    "leere nachricht",
    "empty",
    "leer",
)


def _looks_like_placeholder(text: str, phrases: Sequence[str]) -> bool:
    """Detect low-information placeholder phrases often emitted by small models."""
    normalized = (text or "").strip().lower()
    if not normalized:
        return True
    return any(phrase in normalized for phrase in phrases)


async def _invoke_metadata_prompt(
    llm: BaseChatModel | None, prompt: ChatPromptTemplate, history: Sequence[BaseMessage], timeout: float | None = 6.0
) -> str:
    """Execute the metadata prompt and return its raw string response.

    Raises when the metadata model is not configured so callers can fall back to
    deterministic heuristics.
    """
    if llm is None:
        raise ValueError("LLM instance is not configured")
    chat_string = get_buffer_string(list(history))
    # Optional Redis cache — identical prompt+context returns the previous
    # result without a model call (title/tags/summary are repeatable).
    cache_key = None
    try:
        import hashlib

        from app.redis_client import get_redis

        r = get_redis()
        if r is not None:
            cache_key = "meta_cache:" + hashlib.sha256(
                (chat_string + "|" + str(prompt.messages[0].prompt.template)).encode()
            ).hexdigest()
            cached = await r.get(cache_key)
            if cached:
                return str(cached)
    except Exception:
        cache_key = None
        r = None
    chain = prompt | llm
    # Prefer native async invoke when available to avoid blocking the event loop.
    try:
        if hasattr(chain, "ainvoke"):
            result = await asyncio.wait_for(chain.ainvoke({"chat": chat_string}), timeout=timeout)
        else:
            result = await asyncio.wait_for(asyncio.to_thread(chain.invoke, {"chat": chat_string}), timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError("Metadata LLM invocation timed out")
    text = stringify_message_content(result.content).strip()
    if cache_key is not None and r is not None:
        try:
            await r.set(cache_key, text, ex=86400)
        except Exception:
            pass
    return text


def _extract_json_field(payload: str, key: str) -> str | None:
    """Extract a string field from a JSON blob, returning ``None`` when absent."""
    text = (payload or "").strip()
    if not text:
        return None
    try:
        data = orjson.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        value = data.get(key)
        if isinstance(value, str):
            return value.strip()
    return None


async def generate_title(llm: BaseChatModel | None, chat_history: Sequence[BaseMessage]) -> str:
    """Return a short German conversation title (with emoji) with resilient fallbacks.

    Titles are sourced from the metadata LLM when possible and otherwise derived from the
    recent chat history, with special handling for very short conversations.
    """
    user_turns = sum(1 for msg in chat_history if isinstance(msg, HumanMessage))
    message_count = len(chat_history)
    if user_turns < 2 and message_count < 4:
        return "Cybersecurity Enquiry"

    prompt = _build_metadata_prompt(
        "Fasse den folgenden Chatverlauf in einem prägnanten deutschen Titel mit Emoji zusammen."
        " Nutze maximal 5 Wörter. Nur ein Emoji, keine Sonderzeichen oder andere Sprachen."
        " Antwort nur auf Deutsch! Wenn der Verlauf sehr kurz ist, gib trotzdem einen passenden Titel"
        " an (z.B. Begrüßung, Frage, Incident). Gib niemals Platzhalter wie 'Kein Inhalt' zurück."
    )
    try:
        response = await _invoke_metadata_prompt(llm, prompt, chat_history, timeout=6.0)
    except Exception:
        return _fallback_title_from_history(chat_history)

    extracted = _extract_json_field(response, "title")
    candidate = (extracted or response).strip()
    if _looks_like_placeholder(candidate, _TITLE_PLACEHOLDER_PHRASES):
        return _fallback_title_from_history(chat_history)
    return candidate


async def generate_tags(llm: BaseChatModel | None, chat_history: Sequence[BaseMessage]) -> List[str]:
    """Produce 1-5 topical tags, preferring the metadata LLM but falling back gracefully.

    Empty or placeholder responses trigger deterministic keyword-based heuristics so the
    caller still receives a meaningful tag list.
    """
    prompt = _build_metadata_prompt(
        "Analysiere den folgenden Chatverlauf und gib 1–5 passende Tags zurück, durch Komma getrennt."
        " Keine Erklärungen, keine Formatierung, keine JSON-Struktur."
        " Nur die Tags als kommagetrennte Liste in einer Zeile. Wenn der Chat zu kurz ist, wähle"
        " allgemeine, aber sinnvolle Tags (z.B. 'Greeting', 'Cybersecurity')."
    )
    try:
        response = await _invoke_metadata_prompt(llm, prompt, chat_history, timeout=6.0)
    except Exception:
        return _fallback_tags_from_history(chat_history)

    tag_line = response.splitlines()[0].strip() if response else ""
    if not tag_line or tag_line.lower().startswith("empty"):
        return _fallback_tags_from_history(chat_history)
    tags = [tag.strip() for tag in tag_line.split(",") if tag.strip()]
    filtered = [
        tag
        for tag in tags
        if not _looks_like_placeholder(tag, _TAG_PLACEHOLDER_PHRASES)
    ]
    return filtered or _fallback_tags_from_history(chat_history)



def _parse_assistant_entry(entry: Dict[str, Any], *, source: str) -> Tuple[str, Dict[str, Any]]:
    """Validate one raw assistant entry from instructions YAML.

    Returns ``(name, parsed_dict)``. ``source`` is only used to make error
    messages actionable (file path or a description like ``"YAML text"``).
    """
    name = entry.get("name")
    if not name:
        raise ValueError(f"Assistant entry missing 'name' field in {source}")
    model_name = (entry.get("model") or "").strip()
    if not model_name:
        raise ValueError(f"Assistant '{name}' missing 'model' field in {source}")
    temperature_raw = entry.get("temperature", 0.0)
    try:
        temperature_value = float(temperature_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Assistant '{name}' has invalid temperature '{temperature_raw}'"
        ) from exc
    reasoning_entry = entry.get("reasoning") or {}
    if reasoning_entry and not isinstance(reasoning_entry, dict):
        raise ValueError(
            f"Assistant '{name}' has invalid reasoning block; expected mapping."
        )
    reasoning_entry = normalize_reasoning_dict(reasoning_entry) or {}

    extra_body = entry.get("extra_body")
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ValueError(
            f"Assistant '{name}' has invalid 'extra_body' field; expected a mapping."
        )
    extra_parameters = entry.get("extra_parameters")
    if extra_parameters is not None and not isinstance(extra_parameters, dict):
        raise ValueError(
            f"Assistant '{name}' has invalid 'extra_parameters' field; expected a mapping."
        )
    enable_thinking = entry.get("enable_thinking")
    if enable_thinking is not None and not isinstance(enable_thinking, bool):
        raise ValueError(
            f"Assistant '{name}' has invalid 'enable_thinking' field; expected a boolean."
        )

    collections_entry = entry.get("collections")
    if collections_entry is None:
        collections: List[str] = []
    elif isinstance(collections_entry, list):
        collections = [str(value).strip() for value in collections_entry if str(value).strip()]
    else:
        raise ValueError(
            f"Assistant '{name}' has invalid 'collections' field; expected a list of names."
        )

    tools_entry = entry.get("tools")
    if tools_entry is None:
        tools: Dict[str, Dict[str, Any]] = {}
    elif isinstance(tools_entry, dict):
        tools = {str(k): dict(v) if isinstance(v, dict) else {} for k, v in tools_entry.items()}
    elif isinstance(tools_entry, list):
        # Backward compatibility: convert old list of tool names to dict with empty configs
        logger.warning(
            "Assistant '%s' uses deprecated 'tools' list format; converting to dict. "
            "Please update your instructions YAML to use the new dict format (tool_name: {}).",
            name,
        )
        tools = {str(value).strip(): {} for value in tools_entry if str(value).strip()}
    else:
        raise ValueError(
            f"Assistant '{name}' has invalid 'tools' field; expected a dict of tool_name -> config."
        )

    subagents_entry = entry.get("subagents")
    subagents: Optional[Dict[str, Any]] = None
    if subagents_entry is not None:
        if not isinstance(subagents_entry, dict):
            raise ValueError(
                f"Assistant '{name}' has invalid 'subagents' field; expected a mapping."
            )
        enabled = subagents_entry.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError(f"Assistant '{name}' subagents.enabled must be a boolean.")
        defaults = {
            "max_concurrent": (2, 1, 10),
            "max_depth": (1, 1, 5),
            "max_total": (8, 1, 500),
        }
        subagents = {"enabled": enabled}
        for key, (default, minimum, maximum) in defaults.items():
            raw = subagents_entry.get(key, default)
            try:
                value = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Assistant '{name}' subagents.{key} must be numeric.") from exc
            if value < minimum or value > maximum:
                raise ValueError(
                    f"Assistant '{name}' subagents.{key} must be between {minimum} and {maximum}."
                )
            subagents[key] = value
        report_chars = subagents_entry.get("report_answer_chars")
        if report_chars is not None:
            try:
                report_chars = int(report_chars)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Assistant '{name}' subagents.report_answer_chars must be numeric.") from exc
            if report_chars < 200 or report_chars > 200000:
                raise ValueError(
                    f"Assistant '{name}' subagents.report_answer_chars must be between 200 and 200000."
                )
            subagents["report_answer_chars"] = report_chars
        exclude_tools = subagents_entry.get("exclude_tools")
        if exclude_tools is not None:
            if not isinstance(exclude_tools, list) or not all(
                isinstance(t, str) for t in exclude_tools
            ):
                raise ValueError(
                    f"Assistant '{name}' subagents.exclude_tools must be a list of tool names."
                )
            subagents["exclude_tools"] = exclude_tools

    observer_entry = entry.get("observer")
    observer: Optional[Dict[str, Any]] = None
    if observer_entry is not None:
        if not isinstance(observer_entry, dict):
            raise ValueError(
                f"Assistant '{name}' has invalid 'observer' field; expected a mapping."
            )
        enabled = observer_entry.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError(f"Assistant '{name}' observer.enabled must be a boolean.")
        observer = {"enabled": enabled}
        int_bounds = {
            "every_iterations": (4, 1, 20),
            "max_checks": (8, 1, 50),
            "max_nudges": (4, 1, 20),
        }
        float_bounds = {
            "every_seconds": (180.0, 30.0, 3600.0),
            "check_timeout_seconds": (60.0, 10.0, 300.0),
        }
        for key, (default, minimum, maximum) in int_bounds.items():
            raw = observer_entry.get(key, default)
            try:
                value = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Assistant '{name}' observer.{key} must be numeric.") from exc
            if value < minimum or value > maximum:
                raise ValueError(
                    f"Assistant '{name}' observer.{key} must be between {minimum} and {maximum}."
                )
            observer[key] = value
        for key, (default, minimum, maximum) in float_bounds.items():
            raw = observer_entry.get(key, default)
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Assistant '{name}' observer.{key} must be numeric.") from exc
            if value < minimum or value > maximum:
                raise ValueError(
                    f"Assistant '{name}' observer.{key} must be between {minimum:g} and {maximum:g}."
                )
            observer[key] = value
        assistant = observer_entry.get("assistant")
        if assistant is not None and not isinstance(assistant, str):
            raise ValueError(f"Assistant '{name}' observer.assistant must be a string.")
        observer["assistant"] = (assistant or "").strip() or None
        include_subagents = observer_entry.get("include_subagents", True)
        if not isinstance(include_subagents, bool):
            raise ValueError(
                f"Assistant '{name}' observer.include_subagents must be a boolean."
            )
        observer["include_subagents"] = include_subagents
        instructions = observer_entry.get("instructions")
        if instructions is not None:
            if not isinstance(instructions, str):
                raise ValueError(
                    f"Assistant '{name}' observer.instructions must be a string."
                )
            observer["instructions"] = instructions[:4000]

    workspace_entry = entry.get("workspace")
    workspace: Optional[Dict[str, Any]] = None
    if workspace_entry is not None:
        if not isinstance(workspace_entry, dict):
            raise ValueError(
                f"Assistant '{name}' has invalid 'workspace' field; expected a mapping."
            )
        enabled = workspace_entry.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError(f"Assistant '{name}' workspace.enabled must be a boolean.")
        scope = workspace_entry.get("scope", "conversation")
        if scope not in ("assistant", "user", "conversation"):
            raise ValueError(
                f"Assistant '{name}' workspace.scope must be assistant|user|conversation."
            )
        api_enabled = workspace_entry.get("api_enabled", False)
        if not isinstance(api_enabled, bool):
            raise ValueError(f"Assistant '{name}' workspace.api_enabled must be a boolean.")
        if api_enabled and not enabled:
            raise ValueError(
                f"Assistant '{name}' workspace.api_enabled requires workspace.enabled."
            )
        vision_read = workspace_entry.get("vision_read", False)
        if not isinstance(vision_read, bool):
            raise ValueError(f"Assistant '{name}' workspace.vision_read must be a boolean.")
        workspace = {"enabled": enabled, "scope": scope, "api_enabled": api_enabled, "vision_read": vision_read}
        for key, (default, minimum, maximum) in {
            "max_file_mb": (50, 1, 500),
            "max_total_mb": (500, 10, 10000),
            "max_files": (1000, 1, 100000),
        }.items():
            raw = workspace_entry.get(key, default)
            try:
                value = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Assistant '{name}' workspace.{key} must be numeric.") from exc
            if value < minimum or value > maximum:
                raise ValueError(
                    f"Assistant '{name}' workspace.{key} must be between {minimum} and {maximum}."
                )
            workspace[key] = value

    return name, {
        "description": entry.get("description", "").strip(),
        "system_instructions": entry.get("system_instructions", "").strip(),
        "model": model_name,
        "temperature": temperature_value,
        "reasoning": reasoning_entry,
        "collections": collections,
        "context_window": entry.get("context_window"),
        "max_output_tokens": entry.get("max_output_tokens"),
        "rag_reserved_tokens": entry.get("rag_reserved_tokens"),
        "min_recent_turns": entry.get("min_recent_turns"),
        "provider_id": entry.get("provider_id"),
        "extra_parameters": extra_parameters or None,
        "extra_body": extra_body or None,
        "enable_thinking": enable_thinking,
        "agent_enabled": bool(entry.get("agent_enabled", False)),
        "tools": tools,
        "max_iterations": entry.get("max_iterations"),
        "max_tool_calls": entry.get("max_tool_calls"),
        "max_parallel_tool_calls": entry.get("max_parallel_tool_calls"),
        "agent_memory_enabled": bool(entry.get("agent_memory_enabled", False)),
        "subagents": subagents,
        "observer": observer,
        "workspace": workspace,
        "api_accessible": bool(entry.get("api_accessible", True)),
        "audio_stt_enabled": bool(entry.get("audio_stt_enabled", False)),
        "audio_tts_enabled": bool(entry.get("audio_tts_enabled", False)),
        "audio_phone_enabled": bool(entry.get("audio_phone_enabled", False)),
        "audio_tts_auto_read": bool(entry.get("audio_tts_auto_read", False)),
        "audio_tts_streaming_enabled": bool(entry.get("audio_tts_streaming_enabled", True)),
        "audio_voice_id": entry.get("audio_voice_id") or None,
        "emotion_detection_enabled": bool(entry.get("emotion_detection_enabled", False)),
        "emotion_text_enabled": bool(entry.get("emotion_text_enabled", True)),
        "emotion_webcam_enabled": bool(entry.get("emotion_webcam_enabled", False)),
        "emotion_prompt_steering_enabled": bool(entry.get("emotion_prompt_steering_enabled", True)),
        "emotion_analytics_enabled": bool(entry.get("emotion_analytics_enabled", True)),
    }


def load_instructions(yaml_path: str | Path) -> Dict[str, Dict[str, str]]:
    """Load assistant instruction sets from YAML and expose them as a name-indexed dict.

    Validates that each entry contains the expected ``name`` and instruction fields to
    avoid silent misconfiguration at startup.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Instructions file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not data or "assistants" not in data:
        raise ValueError(f"No 'assistants' section found in {path}")

    assistants: Dict[str, Dict[str, Any]] = {}
    for entry in data.get("assistants", []):
        name, parsed = _parse_assistant_entry(entry, source=str(path))
        assistants[name] = parsed

    if not assistants:
        raise ValueError(f"No valid assistant definitions found in {path}")
    return assistants


def load_instructions_from_yaml_text(yaml_text: str) -> Dict[str, Dict[str, Any]]:
    """Parse assistant instructions from an in-memory YAML string."""
    data = yaml.safe_load(yaml_text)
    if not data or "assistants" not in data:
        raise ValueError("No 'assistants' section found in YAML text")

    assistants: Dict[str, Dict[str, Any]] = {}
    for entry in data.get("assistants", []):
        name, parsed = _parse_assistant_entry(entry, source="YAML text")
        assistants[name] = parsed

    if not assistants:
        raise ValueError("No valid assistant definitions found in YAML text")
    return assistants