"""PGVector orchestration helpers used by :class:`agent.application.RAGApplication`.

Vector store bootstrap, ingestion, and retriever construction live here so the rest of
the application can treat PGVector as a ready-to-use dependency.  The module handles
database/extension creation, batched inserts, and reuse of existing collections.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Dict, Iterable, List, Optional, Sequence

from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import (
    Base as PGVectorBase,
    _get_embedding_collection_store,
)
from sqlalchemy import bindparam, delete, func, literal_column, select, text, create_engine
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

from agent.exceptions import VectorStoreError
from agent.utils.pg_helpers import (
    _quote_identifier,
    _to_async_connection_string,
    _resolve_langchain_schema,
    SchemaCache,
    CollectionUuidCache,
    LANGCHAIN_EMBEDDING_TABLE,
    SOURCE_STATE_TABLE,
    _collection_table,
    _embedding_table,
    _source_state_table,
)

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Encapsulate PGVector lifecycle management, ingestion, and retriever creation."""

    def __init__(
        self,
        connection_string: str,
        collection_name: str,
        search_k: int = 5,
        batch_size: int = 200,
        engine: Optional[AsyncEngine] = None,
    ) -> None:
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.search_k = search_k
        self.batch_size = batch_size
        self.vectorstore: Optional[PGVector] = None
        self.retriever = None
        self._last_health_check: Optional[bool] = None
        self._schema_retry_attempts = 3
        self._schema_retry_delay = 2.0
        # Async engine used for metadata and lightweight queries
        if engine is not None:
            self._engine = engine
        else:
            async_conn = _to_async_connection_string(self.connection_string)
            self._engine: AsyncEngine = create_async_engine(async_conn, pool_size=10, max_overflow=20, pool_pre_ping=True)
        self._schema_cache = SchemaCache()
        self._uuid_cache = CollectionUuidCache(collection_name)

    @property
    def ready(self) -> bool:
        return self.vectorstore is not None and self.retriever is not None

    async def initialize(
        self,
        documents: List[Document],
        embeddings: Embeddings,
        *,
        force: bool = False,
    ) -> None:
        """Create or reuse the PGVector collection and expose a LangChain retriever.

        Parameters
        ----------
        documents:
            Cleaned and chunked LangChain documents ready for embedding.
        embeddings:
            Embedding function implementing LangChain's ``Embeddings`` protocol.
        force:
            When ``True``, delete and rebuild the collection even if it already exists.
        """
        try:
            await self._ensure_schema()
            await self._ensure_support_tables_and_indexes()
        except Exception as exc:
            raise VectorStoreError("Failed to prepare PGVector schema") from exc

        if not force and await self._collection_exists():
            logger.info(
                "Reusing existing PGVector collection '%s'; skipping embedding rebuild.",
                self.collection_name,
            )
            self.vectorstore = PGVector(
                embeddings=embeddings,
                collection_name=self.collection_name,
                connection=self.connection_string,
                use_jsonb=True,
                async_mode=True,
            )
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.search_k}
            )
            return

        if not documents:
            logger.warning("No documents available for vector store initialization.")
            return

        batches = list(_chunk(documents, self.batch_size))
        first_batch = batches[0]

        try:
            self.vectorstore = await PGVector.afrom_documents(
                embedding=embeddings,
                documents=first_batch,
                collection_name=self.collection_name,
                connection=self.connection_string,
                use_jsonb=True,
                pre_delete_collection=force,
            )
        except Exception as exc:  # pragma: no cover - depends on PGVector internals
            logger.error("PGVector initialization failed: %s", exc)
            raise VectorStoreError("Failed to initialize PGVector store") from exc

        for batch in batches[1:]:
            await self._add_batch(batch)

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.search_k})

    async def _ensure_schema(self) -> None:
        """Ensure the Postgres database, ``vector`` extension, and tables are present."""
        url = make_url(self.connection_string)
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._schema_retry_attempts + 1):
            try:
                # Test connectivity and create extensions + metadata using async engine
                async with self._engine.connect() as conn:
                    try:
                        await conn.execute(text("SELECT 1"))
                    except OperationalError as exc:
                        pgcode = getattr(getattr(exc, "orig", None), "pgcode", None)
                        if pgcode == "3D000" or "does not exist" in str(exc).lower():
                            # Create database using sync admin connection
                            admin_url = url.set(drivername=url.drivername.split("+")[0], database="postgres")
                            admin_engine = create_engine(str(admin_url))
                            try:
                                with admin_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as admin_conn:
                                    admin_conn.execute(text(f"CREATE DATABASE {_quote_identifier(url.database)}"))
                                    logger.info("Created missing Postgres database '%s'", url.database)
                            finally:
                                admin_engine.dispose()
                        else:
                            raise

                # Extensions in their own transaction — concurrent collection
                # bootstraps race the pg_extension catalog insert and one side
                # loses with a duplicate-key error. That just means the other
                # task already created it, so duplicate-key is tolerated.
                async with self._engine.begin() as conn:
                    try:
                        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    except SQLAlchemyError as exc:
                        if "duplicate key" not in str(exc):
                            raise
                    try:
                        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
                    except SQLAlchemyError as exc:  # pragma: no cover
                        if "duplicate key" not in str(exc):
                            logger.warning("Unable to enable pg_trgm extension: %s", exc)
                async with self._engine.begin() as conn:
                    # ensure embedding collection store and metadata
                    _get_embedding_collection_store()
                    await conn.run_sync(PGVectorBase.metadata.create_all)
                return
            except SQLAlchemyError as exc:
                last_exc = exc
                logger.warning(
                    "Unable to bootstrap PGVector schema (attempt %s/%s): %s",
                    attempt,
                    self._schema_retry_attempts,
                    exc,
                )
            except Exception as exc:  # pragma: no cover - unexpected bootstrap failures
                last_exc = exc
                logger.warning(
                    "Unexpected error during PG bootstrap (attempt %s/%s): %s",
                    attempt,
                    self._schema_retry_attempts,
                    exc,
                )
            if attempt < self._schema_retry_attempts:
                await asyncio.sleep(self._schema_retry_delay * attempt)

        if last_exc:
            raise last_exc

    async def _ensure_support_tables_and_indexes(self) -> None:
        """Create auxiliary tables + indexes used for incremental indexing and hybrid retrieval."""
        async with self._engine.begin() as conn:
            schema = await self._schema_cache.get(conn)
            if schema is None:
                return

            embedding_table = f"{_quote_identifier(schema)}.{_quote_identifier(LANGCHAIN_EMBEDDING_TABLE)}"
            state_table = f"{_quote_identifier(schema)}.{_quote_identifier(SOURCE_STATE_TABLE)}"

            # Track per-source hashes so we can skip re-embedding unchanged sources.
            try:
                await conn.execute(
                    text(
                        f"""
                        CREATE TABLE IF NOT EXISTS {state_table} (
                            collection_name TEXT NOT NULL,
                            source TEXT NOT NULL,
                            source_hash TEXT NOT NULL,
                            raw_file_hash TEXT,
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            PRIMARY KEY (collection_name, source)
                        )
                        """
                    )
                )
                await conn.execute(
                    text(
                        f"""
                        ALTER TABLE {state_table}
                        ADD COLUMN IF NOT EXISTS raw_file_hash TEXT
                        """
                    )
                )
            except SQLAlchemyError as exc:  # pragma: no cover
                logger.warning("Unable to create source state table: %s", exc)
                return

            # Fast lookup for per-source deletes/updates.
            for ddl in (
                f"""
                CREATE INDEX IF NOT EXISTS idx_agentix_embedding_collection_source
                ON {embedding_table} (collection_id, (cmetadata->>'source'))
                """,
                f"""
                CREATE INDEX IF NOT EXISTS idx_agentix_embedding_doc_id
                ON {embedding_table} ((cmetadata->>'doc_id'))
                """,
                f"""
                CREATE INDEX IF NOT EXISTS idx_agentix_embedding_fts
                ON {embedding_table}
                USING GIN (to_tsvector('simple', document))
                """,
                f"""
                CREATE INDEX IF NOT EXISTS idx_agentix_embedding_trgm
                ON {embedding_table}
                USING GIN (document gin_trgm_ops)
                """,
            ):
                try:
                    await conn.execute(text(ddl))
                except SQLAlchemyError as exc:  # pragma: no cover
                    logger.warning("Unable to apply support DDL: %s", exc)

            # Generated column for pre-computed tsvector (Postgres 12+)
            try:
                await conn.execute(
                    text(
                        f"""
                        ALTER TABLE {embedding_table}
                        ADD COLUMN IF NOT EXISTS document_tsvector tsvector
                        GENERATED ALWAYS AS (to_tsvector('simple', document)) STORED
                        """
                    )
                )
                await conn.execute(
                    text(
                        f"""
                        CREATE INDEX IF NOT EXISTS idx_agentix_embedding_fts_generated
                        ON {embedding_table} USING GIN (document_tsvector)
                        """
                    )
                )
            except SQLAlchemyError as exc:  # pragma: no cover
                logger.warning("Unable to create generated tsvector column/index: %s", exc)

    async def prepare_metadata_store(self) -> None:
        """Ensure schema, extensions, and support tables exist before querying metadata."""
        try:
            await self._ensure_schema()
            await self._ensure_support_tables_and_indexes()
        except Exception as exc:
            raise VectorStoreError("Failed to prepare PGVector metadata store") from exc

    async def _get_source_hash(self, source: str) -> Optional[str]:
        """Return stored hash for a (collection_name, source) tuple."""

        async with self._engine.connect() as conn:
            schema = await self._schema_cache.get(conn)
            if schema is None:
                return None
            tbl = _source_state_table(schema)
            try:
                res = await conn.execute(
                    select(tbl.c.source_hash)
                    .where(tbl.c.collection_name == bindparam("collection_name"))
                    .where(tbl.c.source == bindparam("source")),
                    {"collection_name": self.collection_name, "source": source},
                )
                row = res.scalar()
                return str(row) if row else None
            except SQLAlchemyError as exc:  # pragma: no cover
                logger.warning("Unable to read source state for '%s': %s", self.collection_name, exc)
                return None

    async def get_raw_file_hash(self, source: str) -> Optional[str]:
        """Return stored raw_file_hash for a (collection_name, source) tuple."""

        hashes = await self.get_raw_file_hashes([source])
        return hashes.get(source)

    async def get_raw_file_hashes(self, sources: Sequence[str]) -> Dict[str, Optional[str]]:
        """Return a mapping of source -> raw_file_hash for a given list of sources."""
        result: Dict[str, Optional[str]] = {s: None for s in sources}
        if not sources:
            return result

        async with self._engine.connect() as conn:
            schema = await self._schema_cache.get(conn)
            if schema is None:
                return result
            tbl = _source_state_table(schema)
            try:
                res = await conn.execute(
                    select(tbl.c.source, tbl.c.raw_file_hash)
                    .where(tbl.c.collection_name == bindparam("collection_name"))
                    .where(tbl.c.source.in_(sources)),
                    {"collection_name": self.collection_name},
                )
                for row in res:
                    result[str(row.source)] = str(row.raw_file_hash) if row.raw_file_hash is not None else None
                return result
            except SQLAlchemyError as exc:  # pragma: no cover
                logger.warning("Unable to read raw file hashes for '%s': %s", self.collection_name, exc)
                return result

    async def upsert_source_hash(
        self, source: str, source_hash: str, raw_file_hash: Optional[str] = None
    ) -> None:
        """Insert or update the stored source hash and optional raw file hash."""

        async with self._engine.begin() as conn:
            schema = await self._schema_cache.get(conn)
            if schema is None:
                return
            tbl = _source_state_table(schema)
            try:
                values = {
                    "collection_name": self.collection_name,
                    "source": source,
                    "source_hash": source_hash,
                }
                if raw_file_hash is not None:
                    values["raw_file_hash"] = raw_file_hash
                set_ = {
                    "source_hash": pg_insert(tbl).excluded.source_hash,
                    "updated_at": func.now(),
                }
                if raw_file_hash is not None:
                    set_["raw_file_hash"] = pg_insert(tbl).excluded.raw_file_hash
                stmt = (
                    pg_insert(tbl)
                    .values(values)
                    .on_conflict_do_update(
                        index_elements=["collection_name", "source"],
                        set_=set_,
                    )
                )
                await conn.execute(stmt)
            except SQLAlchemyError as exc:  # pragma: no cover
                logger.warning("Unable to write source state for '%s': %s", self.collection_name, exc)

    async def _delete_embeddings_for_source(self, source: str) -> int:
        """Delete all embedding rows for a specific source within this collection."""

        async with self._engine.begin() as conn:
            collection_uuid = await self._resolve_collection_uuid(conn)
            if collection_uuid is None:
                return 0
            schema = await self._schema_cache.get(conn)
            if schema is None:
                return 0
            tbl = _embedding_table(schema)
            try:
                result = await conn.execute(
                    delete(tbl)
                    .where(tbl.c.collection_id == bindparam("collection_id"))
                    .where(tbl.c.cmetadata["source"].astext == bindparam("source")),
                    {"collection_id": collection_uuid, "source": source},
                )
                return int(getattr(result, "rowcount", 0) or 0)
            except SQLAlchemyError as exc:  # pragma: no cover
                logger.warning("Unable to delete embeddings for source '%s': %s", source, exc)
                return 0

    async def count_documents_for_source(self, source: str) -> int:
        """Return the number of embedding rows for a specific source within this collection."""
        try:
            async with self._engine.connect() as conn:
                collection_uuid = await self._resolve_collection_uuid(conn)
                if collection_uuid is None:
                    return 0
                schema = await self._schema_cache.get(conn)
                if schema is None:
                    return 0
                tbl = _embedding_table(schema)
                res = await conn.execute(
                    select(func.count())
                    .where(tbl.c.collection_id == bindparam("collection_id"))
                    .where(tbl.c.cmetadata["source"].astext == bindparam("source")),
                    {"collection_id": collection_uuid, "source": source},
                )
                count = res.scalar()
                return int(count or 0)
        except SQLAlchemyError as exc:  # pragma: no cover
            logger.warning("Unable to count embeddings for source '%s': %s", source, exc)
            return 0

    async def count_documents_for_sources(self, sources: List[str]) -> Dict[str, int]:
        """Return the number of embedding rows for multiple sources in a single query."""
        if not sources:
            return {}
        try:
            async with self._engine.connect() as conn:
                collection_uuid = await self._resolve_collection_uuid(conn)
                if collection_uuid is None:
                    return {}
                schema = await self._schema_cache.get(conn)
                if schema is None:
                    return {}
                tbl = _embedding_table(schema)
                source_col = literal_column("cmetadata->>'source'")
                res = await conn.execute(
                    select(source_col.label("source"), func.count())
                    .where(tbl.c.collection_id == bindparam("collection_id"))
                    .where(source_col.in_(sources))
                    .group_by(source_col),
                    {"collection_id": collection_uuid},
                )
                return {str(row.source): int(row[1]) for row in res}
        except SQLAlchemyError as exc:  # pragma: no cover
            logger.warning("Unable to count embeddings for sources in '%s': %s", self.collection_name, exc)
            return {}

    @staticmethod
    def hash_source_documents(docs: Sequence[Document]) -> str:
        joined = "\n".join((doc.page_content or "").strip() for doc in docs if doc.page_content)
        return hashlib.sha256(joined.encode("utf-8", errors="ignore")).hexdigest()

    async def source_unchanged(self, source: str, docs: Sequence[Document]) -> Optional[str]:
        """Compare computed hash with stored hash for a source.

        Returns ``None`` when the source is unchanged, otherwise the new hash.
        """
        computed = self.hash_source_documents(docs)
        stored = await self._get_source_hash(source)
        return None if stored == computed else computed

    async def upsert_documents(
        self,
        source: str,
        doc_splits: List[Document],
        embeddings: Embeddings,
        *,
        source_hash: Optional[str] = None,
        raw_file_hash: Optional[str] = None,
    ) -> dict:
        """Insert, update, or skip documents for a source based on content hash.

        Passing ``source_hash`` avoids recomputing the hash inside this method.

        Returns a dict with ``action`` (``"skipped"`` or ``"updated"``) and
        metadata about what changed.
        """
        source_hash = source_hash or self.hash_source_documents(doc_splits)
        stored_hash = await self._get_source_hash(source)
        if stored_hash == source_hash:
            logger.info("Source '%s' unchanged; skipping embedding update.", source)
            if raw_file_hash is not None:
                await self.upsert_source_hash(source, source_hash, raw_file_hash=raw_file_hash)
            return {"action": "skipped"}

        logger.info("Source '%s' changed; refreshing embeddings.", source)
        deleted = await self._delete_embeddings_for_source(source)
        if doc_splits and self.vectorstore is not None:
            await self.vectorstore.aadd_documents(doc_splits)
        await self.upsert_source_hash(source, source_hash, raw_file_hash=raw_file_hash)
        return {"action": "updated", "deleted": deleted, "added": len(doc_splits)}

    async def _add_batch(self, batch: List[Document]) -> None:
        """Append a batch of documents to the existing PGVector collection."""
        if not batch or self.vectorstore is None:
            return
        try:
            await self.vectorstore.aadd_documents(batch)
        except Exception as exc:  # pragma: no cover - depends on PGVector internals
            raise VectorStoreError("Failed to add documents to PGVector") from exc

    async def _resolve_collection_uuid(self, conn) -> Optional[str]:
        """Return the UUID for the configured collection, using the UUID cache."""
        schema = await self._schema_cache.get(conn)
        if schema is None:
            return None
        return await self._uuid_cache.get(conn, schema)

    async def _collection_exists(self) -> bool:
        """Return ``True`` when the LangChain collection table contains the target name."""
        try:
            async with self._engine.connect() as conn:
                collection_uuid = await self._resolve_collection_uuid(conn)
                if collection_uuid is None:
                    return False
                schema = await self._schema_cache.get(conn)
                if schema is None:
                    return False
                tbl = _embedding_table(schema)
                res = await conn.execute(
                    select(1)
                    .where(tbl.c.collection_id == bindparam("collection_id"))
                    .limit(1),
                    {"collection_id": collection_uuid},
                )
                has_embeddings = res.scalar()
                return has_embeddings is not None
        except SQLAlchemyError as exc:  # pragma: no cover - best effort guard
            pgcode = getattr(getattr(exc, "orig", None), "pgcode", None)
            if pgcode == "42P01":
                return False
            logger.warning("Unable to check PGVector collection existence: %s", exc)
            return False

    async def collection_exists(self) -> bool:
        """Public wrapper around the internal existence check."""
        return await self._collection_exists()

    async def get_document_count(self) -> int:
        """Public wrapper around the internal document count."""
        return await self._get_document_count()

    async def _get_document_count(self) -> int:
        """Return the number of embedding rows for the configured collection."""
        try:
            async with self._engine.connect() as conn:
                collection_uuid = await self._resolve_collection_uuid(conn)
                if collection_uuid is None:
                    return 0
                schema = await self._schema_cache.get(conn)
                if schema is None:
                    return 0
                tbl = _embedding_table(schema)
                res = await conn.execute(
                    select(func.count())
                    .where(tbl.c.collection_id == bindparam("collection_id")),
                    {"collection_id": collection_uuid},
                )
                count = res.scalar()
                return int(count or 0)
        except SQLAlchemyError:
            return 0

    async def list_sources(self) -> List[str]:
        """Return all stored source names for this collection."""
        try:
            async with self._engine.connect() as conn:
                schema = await self._schema_cache.get(conn)
                if schema is None:
                    return []
                tbl = _source_state_table(schema)
                res = await conn.execute(
                    select(tbl.c.source)
                    .where(tbl.c.collection_name == bindparam("collection_name")),
                    {"collection_name": self.collection_name},
                )
                return [str(row.source) for row in res]
        except SQLAlchemyError as exc:  # pragma: no cover
            logger.warning("Unable to list sources for '%s': %s", self.collection_name, exc)
            return []

    async def get_source_chunks(self, source: str) -> List[Dict[str, Any]]:
        """Return all chunks of a single source, ordered by ``chunk_index``.

        Rows without ``chunk_index`` metadata (legacy data) sort last via
        ``NULLS LAST``. Used by the ``rag_search`` read action to reassemble
        full document text from stored chunks.
        """
        try:
            async with self._engine.connect() as conn:
                collection_uuid = await self._resolve_collection_uuid(conn)
                if collection_uuid is None:
                    return []
                schema = await self._schema_cache.get(conn)
                if schema is None:
                    return []
                tbl = _embedding_table(schema)
                res = await conn.execute(
                    select(
                        tbl.c.document,
                        tbl.c.cmetadata,
                        literal_column("cmetadata->>'chunk_index'").label("chunk_index"),
                    )
                    .where(tbl.c.collection_id == bindparam("collection_id"))
                    .where(tbl.c.cmetadata["source"].astext == bindparam("source"))
                    .order_by(
                        text(
                            "CASE WHEN cmetadata->>'chunk_index' ~ '^[0-9]+$' "
                            "THEN CAST(cmetadata->>'chunk_index' AS INTEGER) END NULLS LAST"
                        )
                    ),
                    {"collection_id": collection_uuid, "source": source},
                )
                return [
                    {
                        "document": str(row.document or ""),
                        "metadata": dict(row.cmetadata or {}),
                        "chunk_index": row.chunk_index,
                    }
                    for row in res.mappings()
                ]
        except SQLAlchemyError as exc:  # pragma: no cover
            logger.warning(
                "Unable to fetch chunks for source '%s' in '%s': %s",
                source, self.collection_name, exc,
            )
            return []

    async def list_embedding_sources(self) -> List[Dict[str, Any]]:
        """Return all distinct sources from the embedding table with chunk counts."""
        try:
            async with self._engine.connect() as conn:
                collection_uuid = await self._resolve_collection_uuid(conn)
                if collection_uuid is None:
                    return []
                schema = await self._schema_cache.get(conn)
                if schema is None:
                    return []
                tbl = _embedding_table(schema)
                source_col = literal_column("cmetadata->>'source'")
                res = await conn.execute(
                    select(source_col.label("source"), func.count().label("chunks"))
                    .where(tbl.c.collection_id == bindparam("collection_id"))
                    .group_by(source_col)
                    .order_by(source_col),
                    {"collection_id": collection_uuid},
                )
                return [
                    {"source": str(row.source), "chunks": int(row.chunks)}
                    for row in res
                ]
        except SQLAlchemyError as exc:  # pragma: no cover
            logger.warning(
                "Unable to list embedding sources for '%s': %s", self.collection_name, exc
            )
            return []

    async def get_okf_concepts(self) -> List[Dict[str, Any]]:
        """Return all distinct OKF concepts stored in this collection.

        Queries the embedding table's JSONB ``cmetadata`` for rows where
        ``okf_concept_id`` is present, deduplicates by concept ID, and
        returns a list of concept metadata dicts.
        """
        try:
            async with self._engine.connect() as conn:
                collection_uuid = await self._resolve_collection_uuid(conn)
                if collection_uuid is None:
                    return []
                schema = await self._schema_cache.get(conn)
                if schema is None:
                    return []
                tbl = _embedding_table(schema)
                okf_id_col = literal_column("cmetadata->>'okf_concept_id'")
                okf_type_col = literal_column("cmetadata->>'okf_type'")
                okf_title_col = literal_column("cmetadata->>'okf_title'")
                source_col = literal_column("cmetadata->>'source'")
                collection_col = literal_column("cmetadata->>'collection'")
                res = await conn.execute(
                    select(
                        okf_id_col.label("okf_concept_id"),
                        okf_type_col.label("okf_type"),
                        okf_title_col.label("okf_title"),
                        source_col.label("source"),
                        collection_col.label("collection"),
                        literal_column("cmetadata->'okf_frontmatter'->>'description'").label("description"),
                        literal_column("cmetadata->'okf_frontmatter'->>'date_downloaded'").label("date_downloaded"),
                    )
                    .where(tbl.c.collection_id == bindparam("collection_id"))
                    .where(okf_id_col.isnot(None))
                    .group_by(okf_id_col, okf_type_col, okf_title_col, source_col, collection_col,
                              literal_column("cmetadata->'okf_frontmatter'->>'description'"),
                              literal_column("cmetadata->'okf_frontmatter'->>'date_downloaded'"))
                    .order_by(okf_title_col),
                    {"collection_id": collection_uuid},
                )
                concepts: List[Dict[str, Any]] = []
                for row in res:
                    concepts.append({
                        "okf_concept_id": str(row.okf_concept_id) if row.okf_concept_id else "",
                        "okf_type": str(row.okf_type) if row.okf_type else "text",
                        "okf_title": str(row.okf_title) if row.okf_title else "",
                        "source": str(row.source) if row.source else "",
                        "collection": str(row.collection) if row.collection else self.collection_name,
                        "description": str(row.description) if row.description else "",
                        "date_downloaded": str(row.date_downloaded) if row.date_downloaded else "",
                    })
                return concepts
        except SQLAlchemyError as exc:  # pragma: no cover
            logger.warning(
                "Unable to list OKF concepts for '%s': %s", self.collection_name, exc
            )
            return []

    async def delete_source(self, source: str) -> dict:
        """Remove embeddings and source state for a specific source."""
        deleted_embeddings = await self._delete_embeddings_for_source(source)
        deleted_state = 0
        try:
            async with self._engine.begin() as conn:
                schema = await self._schema_cache.get(conn)
                if schema is not None:
                    tbl = _source_state_table(schema)
                    result = await conn.execute(
                        delete(tbl)
                        .where(tbl.c.collection_name == bindparam("collection_name"))
                        .where(tbl.c.source == bindparam("source")),
                        {"collection_name": self.collection_name, "source": source},
                    )
                    deleted_state = int(getattr(result, "rowcount", 0) or 0)
        except SQLAlchemyError as exc:  # pragma: no cover
            logger.warning("Unable to delete source state for '%s': %s", source, exc)
        return {"deleted_embeddings": deleted_embeddings, "deleted_state": deleted_state}

    async def drop_collection(self) -> dict:
        """Remove all embeddings, source states, and the collection entry."""
        deleted_embeddings = 0
        deleted_state = 0
        try:
            async with self._engine.begin() as conn:
                collection_uuid = await self._resolve_collection_uuid(conn)
                schema = await self._schema_cache.get(conn)
                if schema is not None:
                    tbl = _embedding_table(schema)
                    if collection_uuid is not None:
                        result = await conn.execute(
                            delete(tbl)
                            .where(tbl.c.collection_id == bindparam("collection_id")),
                            {"collection_id": collection_uuid},
                        )
                        deleted_embeddings = int(getattr(result, "rowcount", 0) or 0)
                    state_tbl = _source_state_table(schema)
                    result = await conn.execute(
                        delete(state_tbl)
                        .where(state_tbl.c.collection_name == bindparam("collection_name")),
                        {"collection_name": self.collection_name},
                    )
                    deleted_state = int(getattr(result, "rowcount", 0) or 0)
                    coll_tbl = _collection_table(schema)
                    await conn.execute(
                        delete(coll_tbl)
                        .where(coll_tbl.c.name == bindparam("collection_name")),
                        {"collection_name": self.collection_name},
                    )
        except SQLAlchemyError as exc:  # pragma: no cover
            logger.warning("Unable to drop collection '%s': %s", self.collection_name, exc)
        self.vectorstore = None
        self.retriever = None
        self._uuid_cache.invalidate()
        return {"deleted_embeddings": deleted_embeddings, "deleted_state": deleted_state}

    async def health_check(self) -> bool:
        """Verify the configured PGVector collection is readable and populated."""
        if self.vectorstore is None:
            self._last_health_check = False
            return False
        try:
            exists = await self._collection_exists()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Vector store health check failed: %s", exc)
            self._last_health_check = False
            return False

        self._last_health_check = exists
        return exists


def _chunk(items: Iterable[Document], size: int) -> Iterable[List[Document]]:
    """Yield fixed-size chunks from ``items`` while preserving original order."""
    batch: List[Document] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
