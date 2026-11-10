"""Shared lazy Redis client.

``REDIS_URL`` unset or unreachable → ``get_redis()`` returns ``None`` and every
consumer degrades to the existing in-process path. Redis is an optional
add-on for multi-replica fanout and restart resilience, never a hard dep.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_client = None
_client_failed = False


def get_redis():
    """Return a process-wide redis.asyncio client, or None when disabled."""
    global _client, _client_failed
    if _client is not None:
        return _client
    if _client_failed:
        return None
    try:
        import os

        url = os.environ.get("REDIS_URL") or ""
        if not url:
            return None
        import redis.asyncio as aioredis

        _client = aioredis.from_url(url, decode_responses=True)
        logger.info("Redis enabled: %s", url.rsplit("/", 1)[0])
        return _client
    except Exception:
        _client_failed = True
        logger.warning("Redis unavailable — continuing in-process", exc_info=True)
        return None


def reset_redis() -> None:
    """Drop the cached client (tests)."""
    global _client, _client_failed
    _client = None
    _client_failed = False
