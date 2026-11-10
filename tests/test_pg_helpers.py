"""Tests for agent.utils.pg_helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.utils.pg_helpers import (
    _quote_identifier,
    _vector_literal,
    _to_async_connection_string,
    _collection_table,
    _embedding_table,
    _source_state_table,
    CollectionUuidCache,
)
from agent.config import AppConfig


class TestQuoteIdentifier:
    def test_simple(self):
        assert _quote_identifier("foo") == '"foo"'

    def test_with_quotes(self):
        assert _quote_identifier('foo"bar') == '"foo""bar"'


class TestVectorLiteral:
    def test_basic(self):
        assert _vector_literal([1.0, 2.0, 3.0]) == "[1.0000000000,2.0000000000,3.0000000000]"

    def test_empty(self):
        assert _vector_literal([]) == "[]"


class TestToAsyncConnectionString:
    def test_already_async(self):
        url = _to_async_connection_string("postgresql+asyncpg://u:p@localhost/db")
        assert "+asyncpg" in str(url)

    def test_sync_driver(self):
        url = _to_async_connection_string("postgresql+psycopg://u:p@localhost/db")
        assert "+asyncpg" in str(url)

    def test_plain_postgresql(self):
        url = _to_async_connection_string("postgresql://u:p@localhost/db")
        assert "+asyncpg" in str(url)


class TestAppConfigConnectionString:
    def test_quotes_special_characters(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PGHOST", "localhost")
        monkeypatch.setenv("PGPORT", "5432")
        monkeypatch.setenv("PGDATABASE", "agentix db")
        monkeypatch.setenv("PGUSER", "user@example.com")
        monkeypatch.setenv("PGPASSWORD", "p@ss/word:with spaces")
        config = AppConfig(
            data_dir=tmp_path / "data",
            instructions_file=tmp_path / "instructions.yaml",
        )

        assert config.connection_string == (
            "postgresql+psycopg://user%40example.com:p%40ss%2Fword%3Awith%20spaces"
            "@localhost:5432/agentix%20db"
        )


class TestTableFactories:
    def test_collection_table_no_schema(self):
        tbl = _collection_table()
        assert tbl.name == "langchain_pg_collection"
        assert tbl.schema is None

    def test_collection_table_with_schema(self):
        tbl = _collection_table("public")
        assert tbl.name == "langchain_pg_collection"
        assert tbl.schema == "public"

    def test_source_state_table_columns(self):
        tbl = _source_state_table()
        assert "collection_name" in tbl.c
        assert "source" in tbl.c
        assert "source_hash" in tbl.c
        assert "raw_file_hash" in tbl.c
        assert "updated_at" in tbl.c

    def test_embedding_table_columns(self):
        tbl = _embedding_table()
        assert "collection_id" in tbl.c
        assert "cmetadata" in tbl.c
        assert "document" in tbl.c


class TestCollectionUuidCache:
    @pytest.mark.asyncio
    async def test_returns_cached_uuid(self):
        cache = CollectionUuidCache("my_collection")
        cache._uuid = "cached-uuid"
        import time
        cache._ts = time.time()
        result = await cache.get(None, "public")
        assert result == "cached-uuid"

    @pytest.mark.asyncio
    async def test_queries_db_when_cache_expired(self):
        cache = CollectionUuidCache("my_collection")
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = "db-uuid"
        mock_conn.execute = AsyncMock(return_value=mock_result)
        result = await cache.get(mock_conn, "public")
        assert result == "db-uuid"
        assert cache._uuid == "db-uuid"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        cache = CollectionUuidCache("my_collection")
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        mock_conn.execute = AsyncMock(return_value=mock_result)
        result = await cache.get(mock_conn, "public")
        assert result is None
