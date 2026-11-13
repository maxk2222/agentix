"""Detect live reasoning-parameter support of an LLM endpoint.

Mirrors ``tool_calling_detector``: sends minimal probe chat completions and
analyses whether the backend accepts reasoning controls. Results are cached
per (provider, model) in ``provider_model_reasoning`` by the caller.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, Optional

import httpx

from app.reasoning_params import KNOWN_EFFORT_LADDER

logger = logging.getLogger(__name__)

_PROBE_MESSAGES = [{"role": "user", "content": "hi"}]

# Keywords that indicate the parameter itself is rejected by the backend.
_REJECT_KEYWORDS = (
    "unrecognized",
    "unknown parameter",
    "extra fields not permitted",
    "not supported",
    "not allowed",
    "invalid parameter",
)

# Keywords that indicate the parameter exists but the value was rejected —
# vLLM replies e.g. "Unexpected reasoning effort high. Supported types are
# xhigh (default), medium, and low."
_VALUE_REJECT_KEYWORDS = (
    "unexpected reasoning effort",
    "supported types",
    "supported values",
    "supported efforts",
    "invalid reasoning effort",
)


def _error_text(exc: httpx.HTTPStatusError) -> str:
    try:
        return (exc.response.text or "")[:800]
    except Exception:
        return ""


def _parse_supported_levels(error_text: str) -> list[str]:
    """Extract effort levels a backend advertises inside an error message."""
    text = error_text.lower()
    if not any(keyword in text for keyword in _VALUE_REJECT_KEYWORDS):
        return []
    # Only scan the part of the message that advertises the supported set —
    # the rejected value itself (e.g. "effort high") must not be picked up.
    idx = -1
    for keyword in ("supported types", "supported values", "supported efforts"):
        pos = text.rfind(keyword)
        if pos > idx:
            idx = pos
    scan = text[idx:] if idx >= 0 else text
    found: list[str] = []
    for level in KNOWN_EFFORT_LADDER:
        if re.search(rf"\b{level}\b", scan):
            found.append(level)
    return found


async def _probe(
    client: httpx.AsyncClient, url: str, headers: Dict[str, str], payload: Dict[str, Any]
) -> tuple[Optional[bool], str]:
    """POST one probe. Returns (accepted, error_text)."""
    try:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return True, ""
    except httpx.HTTPStatusError as exc:
        text = _error_text(exc)
        lowered = text.lower()
        if any(keyword in lowered for keyword in _REJECT_KEYWORDS):
            return False, f"HTTP {exc.response.status_code}: {text}"
        if any(keyword in lowered for keyword in _VALUE_REJECT_KEYWORDS):
            # Parameter understood, value rejected — still counts as support.
            return True, f"HTTP {exc.response.status_code}: {text}"
        return None, f"HTTP {exc.response.status_code}: {text}"
    except Exception as exc:
        return None, str(exc)


async def detect_reasoning_support(
    *,
    base_url: str,
    api_key: Optional[str] = None,
    model: str,
    timeout: float = 15.0,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Probe a chat-completions endpoint for reasoning controls.

    Returns a dict with keys:
    - ``supports_effort`` (Optional[bool]) — ``reasoning_effort`` accepted.
    - ``effort_levels`` (list[str]) — levels advertised by the backend, parsed
      from its error message when available.
    - ``supports_thinking_toggle`` (Optional[bool]) — ``chat_template_kwargs``
      ``enable_thinking`` accepted.
    - ``error`` (Optional[str]) — set on transport/inconclusive failures.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    base_payload: Dict[str, Any] = {
        "model": model,
        "messages": list(_PROBE_MESSAGES),
        "max_tokens": 1,
        "temperature": 0,
    }

    result: Dict[str, Any] = {
        "supports_effort": None,
        "effort_levels": [],
        "supports_thinking_toggle": None,
        "error": None,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        # Probe 1: reasoning_effort (top-level OpenAI/vLLM field). A "value
        # rejected but parameter known" error also reveals supported levels.
        accepted, error = await _probe(
            client,
            url,
            headers,
            {**base_payload, "reasoning_effort": "medium"},
        )
        result["supports_effort"] = accepted
        if accepted is True and error:
            result["effort_levels"] = _parse_supported_levels(error)
        if accepted is True and not result["effort_levels"]:
            # Accepted silently — ask the backend which values it supports by
            # sending a deliberately invalid level; vLLM-style errors list the
            # supported set ("Supported types are xhigh, medium, low").
            _, level_error = await _probe(
                client,
                url,
                headers,
                {**base_payload, "reasoning_effort": "__probe_invalid__"},
            )
            if level_error:
                result["effort_levels"] = _parse_supported_levels(level_error)
        if accepted is None:
            result["error"] = error
            return result

        # Probe 2: chat_template_kwargs.enable_thinking (Qwen/Gemma-style).
        accepted2, error2 = await _probe(
            client,
            url,
            headers,
            {**base_payload, "chat_template_kwargs": {"enable_thinking": False}},
        )
        result["supports_thinking_toggle"] = accepted2
        if accepted2 is None and not result["error"]:
            result["error"] = error2

    return result


async def ensure_detected_for_provider(
    provider_store,
    provider_id: str,
    model: str,
) -> Optional[Dict[str, Any]]:
    """Return cached reasoning capabilities or schedule a background probe.

    On the first call this returns ``None`` immediately; subsequent calls
    return the persisted capability dict once the background probe finished.
    """
    store = provider_store
    provider = await store.get_provider(provider_id)
    if provider is None or not getattr(provider, "enabled", True):
        return None

    cached = await store.get_model_reasoning_status(provider_id, model)
    if cached is not None:
        return cached

    asyncio.create_task(_run_detection(store, provider, model))
    return None


async def _run_detection(store: Any, provider: Any, model: str) -> None:
    try:
        api_key = await store.get_api_key(provider.id)
        result = await detect_reasoning_support(
            base_url=provider.base_url,
            api_key=api_key,
            model=model,
            timeout=float(getattr(provider, "timeout_seconds", 30) or 30),
            extra_headers=getattr(provider, "extra_headers", None),
        )
    except Exception as exc:  # never let the background task die unrecorded
        logger.exception(
            "Reasoning detection crashed for provider %s (model %s)",
            provider.id,
            model,
        )
        try:
            await store.update_model_reasoning_status(
                provider_id=provider.id,
                model_id=model,
                capabilities=None,
                error=f"detection error: {exc}",
            )
        except Exception:
            logger.exception("Failed to persist reasoning detection error")
        return
    persistable = (
        result.get("supports_effort") is not None
        or result.get("supports_thinking_toggle") is not None
    )
    try:
        await store.update_model_reasoning_status(
            provider_id=provider.id,
            model_id=model,
            capabilities=result if persistable else None,
            error=None if persistable else result.get("error") or "inconclusive",
        )
    except Exception:
        logger.exception(
            "Failed to persist reasoning detection for provider %s (model %s)",
            provider.id,
            model,
        )
    if result.get("error"):
        logger.warning(
            "Reasoning detection error for provider %s (model %s): %s",
            provider.id,
            model,
            result["error"],
        )
    else:
        logger.info(
            "Reasoning detection for provider %s (model %s): effort=%s toggle=%s levels=%s",
            provider.id,
            model,
            result.get("supports_effort"),
            result.get("supports_thinking_toggle"),
            result.get("effort_levels"),
        )
