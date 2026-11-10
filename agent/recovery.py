"""Retrieval health monitoring and graceful fallback utilities."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Dict

from openai import APIConnectionError, APITimeoutError, APIStatusError
from sqlalchemy.exc import DataError

from agent.exceptions import UpstreamLLMError, VectorStoreError

if TYPE_CHECKING:
    from agent.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


class RecoveryManager:
    """Thread-safe retrieval degradation tracking and automatic recovery scheduling."""

    def __init__(self, vector_managers: Dict[str, "VectorStoreManager"]) -> None:
        self._vector_managers = vector_managers
        self._degraded = False
        self._lock = asyncio.Lock()

    @property
    def degraded(self) -> bool:
        """Return ``True`` when retrieval is currently unavailable."""
        return self._degraded

    async def mark_unhealthy(self) -> None:
        """Set the degraded flag with a single log line."""
        async with self._lock:
            if not self._degraded:
                logger.warning("Marking vector store retriever as degraded; falling back to LLM-only mode.")
            self._degraded = True

    async def attempt_recovery(self) -> bool:
        """Probe each vector manager and clear ``degraded`` on first healthy result."""
        if not self._degraded or not self._vector_managers:
            return False
        for manager in self._vector_managers.values():
            try:
                healthy = await manager.health_check()
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Vector store health check failed: %s", exc)
                continue
            if healthy:
                async with self._lock:
                    self._degraded = False
                logger.info("Vector store health restored; re-enabling retrieval.")
                return True
        return False

    def schedule_recovery(self) -> None:
        """Fire-and-forget a background recovery attempt when an event loop is running."""
        if not self._vector_managers:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Best-effort: if no loop is running we can't schedule a background task
            return
        loop.create_task(self.attempt_recovery())

    @staticmethod
    def should_use_llm_only(exc: Exception) -> bool:
        """Return ``True`` when the exception indicates retrieval/embedding issues."""
        if isinstance(exc, (DataError, VectorStoreError)):
            return True
        if isinstance(exc, (ValueError, RuntimeError)):
            message = str(exc).lower()
            return "embedding" in message or "vector" in message
        message = str(exc).lower()
        return "pgvector" in message or "embedding" in message

    @staticmethod
    def is_upstream_llm_error(exc: Exception) -> bool:
        """Return ``True`` for transient upstream model service failures."""
        if isinstance(exc, UpstreamLLMError):
            return True
        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            return True
        if isinstance(exc, APIStatusError):
            status_code = getattr(exc, "status_code", None)
            if status_code is None:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
            try:
                return int(status_code) >= 500
            except Exception:
                return False
        return False
