"""Detect whether an LLM provider supports native OpenAI-style tool calling.

Performs a lightweight probe request and caches the result in the provider store.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"],
        },
    },
}

_MESSAGES = [
    {"role": "system", "content": "You have access to tools."},
    {"role": "user", "content": "What's the weather in Berlin?"},
]


def _has_json_content(text: str) -> bool:
    """Return True if the text contains a JSON object or array anywhere.

    Covers plain JSON, fenced markdown blocks, and JSON embedded in prose.
    """
    if not text:
        return False
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return True
    # Fenced JSON blocks
    if "```" in stripped:
        return True
    # JSON object or array somewhere in the text
    if re.search(r'[\{\[]', stripped):
        return True
    return False


async def detect_tool_calling_support(
    *,
    base_url: str,
    api_key: Optional[str] = None,
    model: str,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Send a probe chat completion with tools and analyse the response.

    Returns a dict with keys:
    - ``supports_native`` (Optional[bool]) – True if tool_calls array is non-empty.
    - ``used_content_json`` (bool) – True if content contains JSON but tool_calls is empty.
    - ``raw_tool_calls`` (Any) – the raw tool_calls field from the response.
    - ``raw_content`` (str) – the raw content field from the response.
    - ``error`` (Optional[str]) – error message if the request failed.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": _MESSAGES,
        "tools": [_TOOL_SCHEMA],
        "tool_choice": "auto",
        "temperature": 0.1,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning("Tool-calling detection HTTP error for %s: %s", model, exc)
        error_text = exc.response.text[:500]
        # LM Studio and similar backends may return a 400/500 with a Jinja
        # template rendering error when the model's prompt template is broken
        # for tool-calling.  Treat this as "does not support native tool calls".
        is_template_error = any(
            keyword in error_text.lower()
            for keyword in ("jinja", "template", "rendering", "undefinedvalue")
        )
        return {
            "supports_native": False if is_template_error else None,
            "used_content_json": False,
            "raw_tool_calls": None,
            "raw_content": "",
            "error": f"HTTP {exc.response.status_code}: {error_text}",
        }
    except Exception as exc:
        logger.warning("Tool-calling detection failed for %s: %s", model, exc)
        error_text = str(exc)
        is_template_error = any(
            keyword in error_text.lower()
            for keyword in ("jinja", "template", "rendering", "undefinedvalue")
        )
        return {
            "supports_native": False if is_template_error else None,
            "used_content_json": False,
            "raw_tool_calls": None,
            "raw_content": "",
            "error": error_text,
        }

    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    tool_calls = message.get("tool_calls")
    content = message.get("content", "")

    # Determine capability
    has_native_tool_calls = bool(tool_calls and len(tool_calls) > 0)
    has_content_json = bool(
        not has_native_tool_calls
        and content
        and _has_json_content(content)
    )

    return {
        "supports_native": has_native_tool_calls,
        "used_content_json": has_content_json,
        "raw_tool_calls": tool_calls,
        "raw_content": content,
        "error": None,
    }


async def ensure_detected_for_provider(
    provider_store,
    provider_id: str,
    model: str,
) -> Optional[bool]:
    """Return cached detection result, or run detection if unknown.

    The detection runs in a background task so the caller is never blocked.
    On the first call this returns ``None`` immediately; subsequent calls
    will return the cached boolean once the background task finishes.
    """
    from app.llm_providers import LLMProviderStore

    store: LLMProviderStore = provider_store
    provider = await store.get_provider(provider_id)
    if provider is None:
        return None

    # Check per-model cache first
    cached = await store.get_model_tool_calling_status(provider_id, model)
    if cached is not None:
        return cached

    # Fallback to provider-wide legacy column
    if provider.supports_native_tool_calls is not None:
        return provider.supports_native_tool_calls

    # Fire-and-forget detection
    asyncio.create_task(_run_detection(store, provider, model))
    return None


async def _run_detection(
    store: Any,
    provider: Any,
    model: str,
) -> None:
    """Run detection and persist result per model."""
    api_key = await store.get_api_key(provider.id)
    result = await detect_tool_calling_support(
        base_url=provider.base_url,
        api_key=api_key,
        model=model,
        timeout=float(provider.timeout_seconds),
    )
    await store.update_model_tool_calling_status(
        provider_id=provider.id,
        model_id=model,
        supports_native=result.get("supports_native"),
        error=result.get("error"),
    )
    if result.get("error"):
        logger.warning(
            "Tool-calling detection error for provider %s (model %s): %s",
            provider.id,
            model,
            result["error"],
        )
    else:
        logger.info(
            "Tool-calling detection for provider %s (model %s): supports_native=%s",
            provider.id,
            model,
            result.get("supports_native"),
        )
