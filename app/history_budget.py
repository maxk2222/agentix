from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence

import tiktoken
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_core.prompts import ChatPromptTemplate

from app.chat_utils import extract_answer_text, render_system_instructions, stringify_message_content

logger = logging.getLogger(__name__)

_SUMMARY_PREFIX = "BISHERIGE ZUSAMMENFASSUNG DES ÄLTEREN CHATVERLAUFS:"
_SUMMARY_NOTE = "Diese Nachricht wurde automatisch aus älteren Chatnachrichten komprimiert und dient nur als Kontext."
_MESSAGE_OVERHEAD_TOKENS = 4
_PROMPT_END_OVERHEAD_TOKENS = 2
_MIN_HISTORY_BUDGET_TOKENS = 128


@dataclass(frozen=True)
class RollingHistorySettings:
    enabled: bool
    context_window: int
    max_output_tokens: int
    rag_reserved_tokens: int
    min_recent_turns: int
    safety_margin_ratio: float
    summary_timeout_seconds: float
    model: str
    system_instructions: str
    agent_memory_reserved_tokens: int = 0


def _int_override(entry: dict[str, Any], key: str, default: int) -> int:
    value = entry.get(key)
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def resolve_rolling_history_settings(
    config: Any,
    assistant_entry: dict[str, Any],
    *,
    provider_windows: dict[str, int] | None = None,
    agent_memory: str | None = None,
) -> RollingHistorySettings:
    from app.model_context import resolve_context_window

    vector_search_k = max(0, int(getattr(config, "vector_search_k", 0) or 0))
    chunk_size = max(0, int(getattr(config, "chunk_size", 0) or 0))
    configured_rag_reserve = int(getattr(config, "rolling_history_rag_reserved_tokens", 0) or 0)
    derived_rag_reserve = min(8_192, int(vector_search_k * chunk_size / 3.5)) if vector_search_k and chunk_size else 0
    default_rag_reserve = configured_rag_reserve if configured_rag_reserve > 0 else derived_rag_reserve

    context_window = resolve_context_window(
        str(assistant_entry.get("model") or ""), assistant_entry, config, provider_windows
    )
    max_output_tokens = max(256, _int_override(assistant_entry, "max_output_tokens", getattr(config, "rolling_history_max_output_tokens", 2_048)))
    rag_reserved_tokens = max(0, _int_override(assistant_entry, "rag_reserved_tokens", default_rag_reserve))
    min_recent_turns = max(1, _int_override(assistant_entry, "min_recent_turns", getattr(config, "rolling_history_min_recent_turns", 4)))

    return RollingHistorySettings(
        enabled=bool(getattr(config, "rolling_history_enabled", True)),
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        rag_reserved_tokens=rag_reserved_tokens,
        min_recent_turns=min_recent_turns,
        safety_margin_ratio=float(getattr(config, "rolling_history_safety_margin_ratio", 0.08) or 0.08),
        summary_timeout_seconds=float(getattr(config, "rolling_history_summary_timeout_seconds", 8.0) or 8.0),
        model=str(assistant_entry.get("model") or ""),
        system_instructions=render_system_instructions(str(assistant_entry.get("system_instructions") or "")),
        agent_memory_reserved_tokens=(
            count_text_tokens(agent_memory, str(assistant_entry.get("model") or ""))
            if agent_memory and assistant_entry.get("agent_memory_enabled")
            else 0
        ),
    )


def _encoding_for_model(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_text_tokens(text: str, model: str) -> int:
    return len(_encoding_for_model(model).encode(text or ""))


def count_message_tokens(message: BaseMessage, model: str) -> int:
    role_overhead = _MESSAGE_OVERHEAD_TOKENS
    return role_overhead + count_text_tokens(stringify_message_content(message.content), model)


def count_messages_tokens(messages: Sequence[BaseMessage], model: str) -> int:
    return _PROMPT_END_OVERHEAD_TOKENS + sum(count_message_tokens(message, model) for message in messages)


def _safety_margin(settings: RollingHistorySettings) -> int:
    return max(256, int(settings.context_window * max(0.0, settings.safety_margin_ratio)))


def calculate_history_budget(settings: RollingHistorySettings, latest_user_message: HumanMessage) -> int:
    fixed_tokens = (
        count_text_tokens(settings.system_instructions, settings.model)
        + count_message_tokens(latest_user_message, settings.model)
        + settings.max_output_tokens
        + settings.rag_reserved_tokens
        + settings.agent_memory_reserved_tokens
        + _safety_margin(settings)
        + _PROMPT_END_OVERHEAD_TOKENS
    )
    return settings.context_window - fixed_tokens


def _split_system_messages(history: Sequence[BaseMessage]) -> tuple[list[SystemMessage], list[BaseMessage]]:
    system_messages: list[SystemMessage] = []
    non_system: list[BaseMessage] = []
    for message in history:
        if isinstance(message, SystemMessage):
            system_messages.append(message)
        else:
            non_system.append(message)
    return system_messages, non_system


def _split_recent_turns(messages: Sequence[BaseMessage], min_recent_turns: int) -> tuple[list[BaseMessage], list[BaseMessage]]:
    if not messages:
        return [], []
    user_seen = 0
    split_index = 0
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            user_seen += 1
            if user_seen >= min_recent_turns:
                split_index = index
                break
    else:
        return [], list(messages)
    return list(messages[:split_index]), list(messages[split_index:])


def _trim_to_budget(messages: Sequence[BaseMessage], budget: int, model: str) -> list[BaseMessage]:
    trimmed = list(messages)
    while trimmed and count_messages_tokens(trimmed, model) > budget:
        trimmed.pop(0)
    return trimmed


def _format_summary_message(summary: str) -> HumanMessage:
    summary_text = summary.strip()
    content = f"{_SUMMARY_PREFIX}\n{_SUMMARY_NOTE}\n\n<summary>\n{summary_text}\n</summary>"
    return HumanMessage(content=content)


def _truncate_summary(summary: str, target_tokens: int, model: str) -> str:
    encoding = _encoding_for_model(model)
    tokens = encoding.encode(summary or "")
    if len(tokens) <= target_tokens:
        return summary.strip()
    return encoding.decode(tokens[:target_tokens]).strip()


async def _generate_history_summary(
    llm: BaseChatModel | None,
    messages: Sequence[BaseMessage],
    *,
    settings: RollingHistorySettings,
) -> str:
    if llm is None:
        raise ValueError("Summary LLM instance is not configured")
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Fasse den älteren Chatverlauf kompakt auf Deutsch zusammen. "
            "Bewahre: Fakten, Entscheidungen, Nutzerpräferenzen, offene Aufgaben, wichtige Einschränkungen und relevante Sicherheitskontexte. "
            "Lass weg: Smalltalk, Höflichkeitsfloskeln, Wiederholungen, irrelevante Details. "
            "Formuliere neutral als Kontextnotiz — keine direkte Rede, keine Anrede. "
            "Erfinde nichts. Sei so knapp wie möglich, aber verliere keine wesentlichen Informationen.",
        ),
        ("human", "Älterer Chatverlauf:\n{chat}"),
    ])
    chat_string = get_buffer_string(list(messages))
    chain = None
    try:
        chain = prompt | llm
    except (AttributeError, TypeError):
        chain = None

    try:
        if chain is not None:
            result = await asyncio.wait_for(chain.ainvoke({"chat": chat_string}), timeout=settings.summary_timeout_seconds)
        elif hasattr(llm, "ainvoke"):
            result = await asyncio.wait_for(llm.ainvoke(prompt.format_messages(chat=chat_string)), timeout=settings.summary_timeout_seconds)
        else:
            result = await asyncio.wait_for(asyncio.to_thread(llm.invoke, prompt.format_messages(chat=chat_string)), timeout=settings.summary_timeout_seconds)
    except asyncio.TimeoutError:
        raise RuntimeError(f"History summary timed out after {settings.summary_timeout_seconds:.1fs}")
    summary = extract_answer_text(result).strip()
    if not summary:
        raise ValueError("Summary LLM returned empty output")
    return summary


async def build_rolling_history(
    history: Sequence[BaseMessage],
    latest_user_message: HumanMessage,
    *,
    settings: RollingHistorySettings,
    summary_llm_factory: Callable[[], BaseChatModel | None]
    | Callable[[], Awaitable[BaseChatModel | None]]
    | None = None,
) -> list[BaseMessage]:
    if not settings.enabled:
        return list(history)

    history_budget = calculate_history_budget(settings, latest_user_message)
    if history_budget <= _MIN_HISTORY_BUDGET_TOKENS:
        logger.warning("Rolling history budget too small (%d tokens); dropping prior history.", history_budget)
        return []

    system_messages, non_system = _split_system_messages(history)
    preserved_system = system_messages[-1:] if system_messages else []
    base_history = preserved_system + non_system
    if count_messages_tokens(base_history, settings.model) <= history_budget:
        return base_history

    older_messages, recent_messages = _split_recent_turns(non_system, settings.min_recent_turns)
    recent_with_system = preserved_system + recent_messages
    if count_messages_tokens(recent_with_system, settings.model) > history_budget:
        return _trim_to_budget(recent_with_system, history_budget, settings.model)

    if not older_messages:
        return _trim_to_budget(recent_with_system, history_budget, settings.model)

    summary_llm = summary_llm_factory() if summary_llm_factory is not None else None
    if inspect.isawaitable(summary_llm):
        summary_llm = await summary_llm
    try:
        summary = await _generate_history_summary(summary_llm, older_messages, settings=settings)
        candidate = preserved_system + [_format_summary_message(summary)] + recent_messages
    except Exception as exc:
        logger.warning("Rolling history summary failed; falling back to token trimming: %r", exc)
        candidate = recent_with_system

    while candidate and count_messages_tokens(candidate, settings.model) > history_budget:
        summary_index = next(
            (
                index for index, message in enumerate(candidate)
                if isinstance(message, HumanMessage)
                and stringify_message_content(message.content).startswith(_SUMMARY_PREFIX)
            ),
            None,
        )
        if summary_index is not None:
            current = stringify_message_content(candidate[summary_index].content)
            inner = current.split("<summary>", 1)[-1].split("</summary>", 1)[0].strip()
            current_tokens = count_text_tokens(inner, settings.model)
            shorter = _truncate_summary(inner, max(128, int(current_tokens * 0.75)), settings.model)
            shortened = _format_summary_message(shorter)
            if count_message_tokens(shortened, settings.model) < count_message_tokens(candidate[summary_index], settings.model):
                candidate[summary_index] = shortened
                continue
            candidate.pop(summary_index)
            continue
        removable_index = 1 if preserved_system and len(candidate) > 1 else 0
        candidate.pop(removable_index)

    return candidate


SUMMARY_PREFIX = _SUMMARY_PREFIX
