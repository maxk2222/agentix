"""Resolve the effective context window for a model.

Resolution order: per-assistant ``context_window`` override > provider
``/v1/models`` metadata > static model-family map > global
``rolling_history_context_window`` default.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Mapping, Optional

import httpx

from app.llm_providers import normalize_base_url

logger = logging.getLogger(__name__)

# Ordered (substring, context_window) rules — first match wins, so more
# specific prefixes must come before their shorter family prefixes. Where a
# family ships multiple context sizes the *smaller* common value is used:
# under-estimating compresses earlier, over-estimating risks 400s.
_MODEL_CONTEXT_RULES = [
    ("gpt-5", 400_000),
    ("gpt-4.1", 1_000_000),
    ("gpt-4o", 128_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-4", 8_192),
    ("o4-mini", 200_000),
    ("o3", 200_000),
    ("o1", 200_000),
    ("claude-opus-4", 200_000),
    ("claude-sonnet-4", 200_000),
    ("claude-3", 200_000),
    ("claude-2", 100_000),
    ("gemini-2", 1_000_000),
    ("gemini-1.5", 1_000_000),
    ("llama-3.1", 128_000),
    ("llama3.1", 128_000),
    ("llama-3.2", 128_000),
    ("llama3.2", 128_000),
    ("llama-3.3", 128_000),
    ("llama3.3", 128_000),
    ("llama-3", 8_192),
    ("llama3", 8_192),
    ("qwen3", 128_000),
    ("qwen2.5", 128_000),
    ("qwen2", 32_768),
    ("mistral-large", 128_000),
    ("mistral-small", 32_768),
    ("mistral", 32_768),
    ("mixtral", 32_768),
    ("deepseek-v3", 64_000),
    ("deepseek-r1", 64_000),
    ("deepseek", 64_000),
    ("phi-4", 16_000),
    ("gemma-3", 128_000),
    ("gemma-2", 8_192),
    ("gemma", 8_192),
]

_PROVIDER_TTL_SECONDS = 300.0
_PROVIDER_KEYS = ("context_length", "context_window", "num_ctx", "max_context_length", "n_ctx_train")
_provider_windows_cache: Dict[str, tuple[float, Dict[str, int]]] = {}


async def get_provider_context_windows(config: Any) -> Dict[str, int]:
    """Fetch per-model context windows from the provider's ``/v1/models``.

    Providers like LM Studio, Ollama and OpenRouter expose a context field on
    each model object; OpenAI does not (returns an empty map). Results are
    cached per base URL for ``_PROVIDER_TTL_SECONDS``.
    """
    base = normalize_base_url(getattr(config, "llm_api_base", "") or "")
    if not base:
        return {}
    now = time.monotonic()
    cached = _provider_windows_cache.get(base)
    if cached is not None and now - cached[0] < _PROVIDER_TTL_SECONDS:
        return cached[1]
    windows: Dict[str, int] = {}
    try:
        headers: Dict[str, str] = {}
        api_key = getattr(config, "llm_api_key", None)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{base}/v1/models", headers=headers)
            resp.raise_for_status()
            for model in resp.json().get("data", []):
                model_id = model.get("id")
                if not model_id:
                    continue
                for key in _PROVIDER_KEYS:
                    value = model.get(key)
                    if isinstance(value, (int, float)) and value > 0:
                        windows[model_id] = int(value)
                        break
    except Exception:
        logger.debug("Could not fetch provider model metadata from %s", base, exc_info=True)
    _provider_windows_cache[base] = (now, windows)
    return windows


def resolve_context_window(
    model_name: str,
    assistant_entry: Optional[Mapping[str, Any]],
    config: Any,
    provider_windows: Optional[Mapping[str, int]] = None,
) -> int:
    """Effective context window for *model_name*.

    Order: per-assistant ``context_window`` override > provider metadata >
    static model-family map > ``rolling_history_context_window`` default.
    """
    default = int(getattr(config, "rolling_history_context_window", 32_768) or 32_768)
    raw = (assistant_entry or {}).get("context_window")
    if raw not in (None, ""):
        try:
            return max(1_024, int(raw))
        except (TypeError, ValueError):
            pass
    if provider_windows:
        window = provider_windows.get(model_name or "")
        if window and window > 0:
            return max(1_024, int(window))
    lowered = (model_name or "").lower()
    if lowered:
        for needle, window in _MODEL_CONTEXT_RULES:
            if needle in lowered:
                return max(1_024, window)
    return max(1_024, default)
