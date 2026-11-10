"""Shared exceptions used across the agent package."""

from __future__ import annotations


class AgentError(RuntimeError):
    """Base exception for all agent-level errors."""


class ConfigurationError(AgentError):
    """Raised when mandatory configuration is missing or invalid."""


class InitializationError(AgentError):
    """Raised when the RAG pipeline fails to initialize."""


class VectorStoreError(AgentError):
    """Raised when PGVector operations fail."""


class RetrievalError(AgentError):
    """Raised when document retrieval fails."""


class EmbeddingError(AgentError):
    """Raised when embedding generation fails."""


class UpstreamLLMError(AgentError):
    """Raised when the upstream model endpoint returns a transient server failure."""
