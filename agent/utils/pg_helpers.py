"""Shared Postgres / PGVector utilities used across the agent package."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Sequence

from sqlalchemy import (
    Column,
    DateTime,
    MetaData,
    Table,
    Text,
    bindparam,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

LANGCHAIN_COLLECTION_TABLE = "langchain_pg_collection"
LANGCHAIN_EMBEDDING_TABLE = "langchain_pg_embedding"
SOURCE_STATE_TABLE = "agentix_source_state"
INSTRUCTIONS_VERSIONS_TABLE = "agentix_instructions_versions"
RAW_FILES_TABLE = "agentix_raw_files"
AGENT_MEMORIES_TABLE = "agentix_agent_memories"
WIDGET_DEFAULTS_TABLE = "agentix_widget_defaults"
PUBLIC_CHAT_CONFIG_TABLE = "agentix_public_chat_config"
WEBDAV_SOURCES_TABLE = "agentix_webdav_sources"
WEBDAV_SOURCE_FILES_TABLE = "agentix_webdav_source_files"
BACKUP_CONFIG_TABLE = "agentix_backup_config"
BACKUP_JOBS_TABLE = "agentix_backup_jobs"
BRANDING_TABLE = "agentix_branding"

SCHEMA_CACHE_TTL = 300
COLLECTION_UUID_TTL = 300


def _collection_table(schema: Optional[str] = None) -> Table:
    return Table(
        LANGCHAIN_COLLECTION_TABLE,
        MetaData(),
        Column("uuid", UUID),
        Column("name", Text),
        schema=schema,
    )


def _embedding_table(schema: Optional[str] = None) -> Table:
    return Table(
        LANGCHAIN_EMBEDDING_TABLE,
        MetaData(),
        Column("collection_id", UUID),
        Column("cmetadata", JSONB),
        Column("document", Text),
        schema=schema,
    )


def _source_state_table(schema: Optional[str] = None) -> Table:
    return Table(
        SOURCE_STATE_TABLE,
        MetaData(),
        Column("collection_name", Text, primary_key=True),
        Column("source", Text, primary_key=True),
        Column("source_hash", Text, nullable=False),
        Column("raw_file_hash", Text, nullable=True),
        Column("updated_at", DateTime(timezone=True), nullable=False),
        schema=schema,
    )


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _vector_literal(values: Sequence[float]) -> str:
    # pgvector literal format: '[1,2,3]'
    return "[" + ",".join(f"{float(v):.10f}" for v in values) + "]"


def _to_async_connection_string(conn_str: str) -> URL:
    """Return an asyncpg-based SQLAlchemy URL for ``AsyncEngine``.

    Keeping the connection details structured avoids driver-specific re-rendering
    differences when SQLAlchemy hands credentials to asyncpg.
    """
    url = make_url(conn_str)
    if "+asyncpg" in url.drivername:
        return url
    base_driver = url.drivername.split("+")[0]
    async_driver = f"{base_driver}+asyncpg"
    return url.set(drivername=async_driver)


async def _resolve_langchain_schema(conn) -> Optional[str]:
    """Return the schema name that hosts the LangChain collection table."""
    try:
        res = await conn.execute(
            text(
                """
                SELECT table_schema
                FROM information_schema.tables
                WHERE table_name = :table_name
                ORDER BY
                    CASE
                        WHEN table_schema = current_schema() THEN 0
                        WHEN table_schema = 'public' THEN 1
                        ELSE 2
                    END
                LIMIT 1
                """
            ),
            {"table_name": LANGCHAIN_COLLECTION_TABLE},
        )
    except SQLAlchemyError:
        logger.exception("Failed to resolve langchain schema")
        return None
    schema = res.scalar()
    return str(schema) if schema else None


class SchemaCache:
    """Simple TTL cache for Postgres schema lookups."""

    def __init__(self, ttl: float = SCHEMA_CACHE_TTL) -> None:
        self._schema: Optional[str] = None
        self._ts: float = 0.0
        self._ttl = ttl

    async def get(self, conn) -> Optional[str]:
        import time

        now = time.time()
        if self._schema and now - self._ts < self._ttl:
            return self._schema
        schema = await _resolve_langchain_schema(conn)
        if schema:
            self._schema = schema
            self._ts = now
        return schema

    def invalidate(self) -> None:
        self._schema = None
        self._ts = 0.0


class CollectionUuidCache:
    """TTL cache for collection UUID lookups inside a schema."""

    def __init__(self, collection_name: str, ttl: float = COLLECTION_UUID_TTL) -> None:
        self._collection_name = collection_name
        self._uuid: Optional[str] = None
        self._ts: float = 0.0
        self._ttl = ttl

    async def get(self, conn, schema: str) -> Optional[str]:
        import time

        now = time.time()
        if self._uuid and now - self._ts < self._ttl:
            return self._uuid

        tbl = _collection_table(schema)
        try:
            res = await conn.execute(
                select(tbl.c.uuid).where(tbl.c.name == bindparam("name")).limit(1),
                {"name": self._collection_name},
            )
        except SQLAlchemyError:
            logger.exception(
                "Failed to resolve collection uuid for '%s' in schema '%s'",
                self._collection_name,
                schema,
            )
            return None

        uuid = res.scalar()
        if uuid is None:
            logger.debug("No collection uuid found for '%s' in schema '%s'", self._collection_name, schema)
            return None

        self._uuid = str(uuid)
        self._ts = now
        return self._uuid

    def invalidate(self) -> None:
        self._uuid = None
        self._ts = 0.0
