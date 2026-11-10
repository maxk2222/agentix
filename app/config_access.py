"""Per-request access to *effective* runtime configuration values.

Runtime config semantics (mirrors ``_build_effective_config`` in admin.py):

- An environment-backed value always wins — it was locked in at startup and
  runtime overrides are rejected for those keys.
- Otherwise a stored runtime-config override wins over the default.
- The ``app_config`` attribute already reflects env + startup-applied
  overrides; the runtime store is only consulted when ``app_config`` still
  holds the field default.

Reading effective values per request means admin-UI toggles take effect
immediately — no "Apply" hot-reload or restart is required for consumers
that go through this helper.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import Request


async def effective_config_value(request: Request, key: str, default: Any = None) -> Any:
    """Return the effective value for ``key``: env > stored override > default."""
    return await effective_config_value_for(
        getattr(request.app, "state", None), key, default
    )


async def effective_config_value_for(app_state: Any, key: str, default: Any = None) -> Any:
    """Request-free variant of :func:`effective_config_value` — resolves the
    same precedence (env > stored override > default) against ``app.state`` for
    background tasks and schedulers."""
    app_config = getattr(app_state, "app_config", None)
    if app_config is None:
        return default
    env_backed = getattr(app_config, "_env_backed_values", {})
    base_val = env_backed.get(key, getattr(app_config, key, None))

    from agent.config import AppConfig

    field = AppConfig.model_fields.get(key)
    field_default = field.default if field is not None else default
    if base_val != field_default:
        # Value differs from the default — either env-backed or set at startup;
        # env-backed keys cannot carry stored overrides, so this wins.
        return base_val

    store = getattr(app_state, "_runtime_config_store", None)
    if store is not None:
        try:
            stored = await store.get(key)
        except Exception:
            stored = None
        if stored is not None:
            return stored
    return base_val


async def effective_config_bool(request: Request, key: str) -> bool:
    """Boolean convenience wrapper for feature flags."""
    return bool(await effective_config_value(request, key, False))
