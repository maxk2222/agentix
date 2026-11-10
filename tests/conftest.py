"""Shared pytest fixtures for the agent test suite."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List

import pytest
from langchain.schema import Document

from agent.config import AppConfig


class FakeEmbeddings:
    """Deterministic fake embeddings for unit tests."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(ord(c) % 10) for c in text[: self.dim].ljust(self.dim, "x")] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return [float(ord(c) % 10) for c in text[: self.dim].ljust(self.dim, "x")]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings(dim=4)


@pytest.fixture
def sample_documents() -> List[Document]:
    return [
        Document(page_content="First document content", metadata={"source": "a.txt"}),
        Document(page_content="Second document content", metadata={"source": "b.txt"}),
    ]


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    """A minimal AppConfig that points data directories into tmp_path."""
    return AppConfig(
        data_dir=tmp_path / "data",
        instructions_file=tmp_path / "instructions.yaml",
        pg_host="localhost",
        pg_port=5432,
        pg_database="test_db",
        pg_user="test",
        pg_password="test",
    )
