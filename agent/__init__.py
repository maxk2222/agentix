"""Agent package for the cybersecurity RAG assistant.

Public façade and configuration helpers used by the FastAPI layer.
"""

from __future__ import annotations

from agent.application import RAGApplication
from agent.config import AppConfig
from agent.exceptions import (
    AgentError,
    ConfigurationError,
    EmbeddingError,
    InitializationError,
    RetrievalError,
    UpstreamLLMError,
    VectorStoreError,
)

__all__ = [
    "RAGApplication",
    "AppConfig",
    "AgentError",
    "ConfigurationError",
    "EmbeddingError",
    "InitializationError",
    "RetrievalError",
    "UpstreamLLMError",
    "VectorStoreError",
]
