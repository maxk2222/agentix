"""Hybrid Postgres retriever (PGVector + FTS + pg_trgm).

This retriever queries the LangChain PGVector tables directly:
- vector similarity over the `embedding` column (pgvector)
- keyword relevance via Postgres FTS and trigram similarity

Results are fused using Reciprocal Rank Fusion (RRF) and then optionally diversified
by limiting the number of chunks per source.

It is designed for small-to-medium corpora and keeps the FastAPI layer thin.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from agent.retrieval_utils import _remove_stopwords
from agent.utils.pg_helpers import (
    _quote_identifier,
    _vector_literal,
    _to_async_connection_string,
    SchemaCache,
    CollectionUuidCache,
    LANGCHAIN_COLLECTION_TABLE,
    LANGCHAIN_EMBEDDING_TABLE,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HybridRetrieverConfig:
    """Configuration for hybrid vector + keyword retrieval inside Postgres."""

    search_k: int = 5
    vector_candidates: int = 40
    keyword_candidates: int = 40
    rrf_k: int = 60
    max_chunks_per_source: int = 3
    query_expansion_enabled: bool = True
    max_expanded_queries: int = 2
    dynamic_candidates_enabled: bool = True
    post_rank_boost_enabled: bool = True
    cache_enabled: bool = True
    cache_ttl_seconds: float = 60.0
    cache_max_entries: int = 256
    rerank_enabled: bool = True
    min_semantic_similarity: float = 0.0


class HybridPostgresRetriever(BaseRetriever):
    """Hybrid retriever that fuses vector + keyword retrieval inside Postgres."""

    def __init__(
        self,
        *,
        connection_string: str,
        collection_name: str,
        embeddings: Any,
        config: HybridRetrieverConfig,
        engine: Optional[AsyncEngine] = None,
        shared_embed_cache: Optional[Dict[str, tuple[Any, float]]] = None,
        shared_cache_lock: Optional[asyncio.Lock] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._connection_string = connection_string
        self._collection_name = collection_name
        self._embeddings = embeddings
        self._config = config
        if engine is not None:
            self._engine = engine
        else:
            async_conn = _to_async_connection_string(self._connection_string)
            self._engine: AsyncEngine = create_async_engine(async_conn, pool_pre_ping=True)

        self._schema_cache = SchemaCache()
        self._uuid_cache = CollectionUuidCache(collection_name)

        self._embed_cache: Dict[str, tuple[Any, float]] = {}
        self._result_cache: Dict[str, tuple[List[Document], float]] = {}
        self._cache_lock = asyncio.Lock()
        self._shared_embed_cache = shared_embed_cache
        self._shared_cache_lock = shared_cache_lock

    def _get_relevant_documents(self, query: str) -> List[Document]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._aget_relevant_documents(query))
        raise RuntimeError(
            "HybridPostgresRetriever cannot run synchronously inside an active event loop. "
            "Use the async interface instead."
        )

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        if not query:
            return []

        start_total = time.perf_counter()
        cache_key = self._cache_key(query)

        # Check result cache
        if self._config.cache_enabled:
            async with self._cache_lock:
                cached = self._result_cache.get(cache_key)
                if cached:
                    docs, ts = cached
                    if (time.perf_counter() - ts) < self._config.cache_ttl_seconds:
                        logger.debug("Result cache hit for collection '%s'", self._collection_name)
                        return list(docs)
                    self._result_cache.pop(cache_key, None)

        # Compute dynamic candidates based on query length
        if self._config.dynamic_candidates_enabled:
            from agent.retrieval_utils import _compute_dynamic_candidates

            vector_candidates, keyword_candidates = _compute_dynamic_candidates(
                query,
                base_vector=self._config.vector_candidates,
                base_keyword=self._config.keyword_candidates,
            )
        else:
            vector_candidates = max(1, int(self._config.vector_candidates))
            keyword_candidates = max(1, int(self._config.keyword_candidates))

        start_embed = time.perf_counter()
        query_embedding = await self._get_embedding(query, cache_key)
        embed_ms = (time.perf_counter() - start_embed) * 1000
        if not query_embedding:
            return []

        vector_literal = _vector_literal(query_embedding)

        # Run DB candidate queries concurrently and gather results
        keyword_queries = self._expand_queries(query)
        vector_rows, keyword_rows = await asyncio.gather(
            self._query_vector_candidates(vector_literal, vector_candidates),
            self._query_keyword_candidates_multi(keyword_queries, vector_literal, keyword_candidates),
        )
        db_ms = (time.perf_counter() - start_embed) * 1000 - embed_ms

        start_fuse = time.perf_counter()
        try:
            fused = self._fuse_rrf(vector_rows, keyword_rows)
            if self._config.rerank_enabled:
                fused = self._rerank(fused)
            if self._config.min_semantic_similarity > 0:
                fused = self._apply_similarity_threshold(fused)
            if self._config.post_rank_boost_enabled:
                fused = self._post_rank_boost(fused, vector_rows, keyword_rows, query)
            diversified = self._diversify(fused)
        except Exception:
            logger.exception("Error while fusing or diversifying results for '%s'", self._collection_name)
            fused = []
            diversified = []
        results = diversified[: max(1, int(self._config.search_k))]
        fuse_ms = (time.perf_counter() - start_fuse) * 1000
        total_ms = (time.perf_counter() - start_total) * 1000

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Hybrid retrieval for '%s': vector=%d keyword=%d fused=%d returned=%d",
                self._collection_name,
                len(vector_rows),
                len(keyword_rows),
                len(fused),
                len(results),
            )
            logger.debug(
                "Hybrid retrieval timings for '%s': embed=%.2fms db=%.2fms fuse=%.2fms total=%.2fms",
                self._collection_name,
                embed_ms,
                max(0.0, db_ms),
                fuse_ms,
                total_ms,
            )

        docs = [Document(page_content=row["document"], metadata=row["metadata"]) for row in results]

        # Store in result cache
        if self._config.cache_enabled:
            async with self._cache_lock:
                self._result_cache[cache_key] = (docs, time.perf_counter())
                self._cleanup_cache(self._result_cache, self._config.cache_max_entries)

        return docs

    def _cache_key(self, query: str) -> str:
        h = hashlib.md5(query.encode("utf-8", errors="ignore")).hexdigest()
        return f"{self._collection_name}|{h}"

    @staticmethod
    def _query_cache_key(query: str) -> str:
        """Collection-agnostic cache key so identical queries share embeddings across instances."""
        return hashlib.md5(query.encode("utf-8", errors="ignore")).hexdigest()

    async def _get_embedding(self, query: str, cache_key: str) -> Any:
        query_only_key = self._query_cache_key(query)

        # 1) Shared cross-collection cache
        if self._shared_embed_cache is not None and self._shared_cache_lock is not None:
            async with self._shared_cache_lock:
                cached = self._shared_embed_cache.get(query_only_key)
                if cached:
                    emb, ts = cached
                    if (time.perf_counter() - ts) < self._config.cache_ttl_seconds:
                        logger.debug("Shared embed cache hit for collection '%s'", self._collection_name)
                        return emb
                    self._shared_embed_cache.pop(query_only_key, None)

        # 2) Local instance cache
        if self._config.cache_enabled:
            async with self._cache_lock:
                cached = self._embed_cache.get(cache_key)
                if cached:
                    emb, ts = cached
                    if (time.perf_counter() - ts) < self._config.cache_ttl_seconds:
                        logger.debug("Embedding cache hit for collection '%s'", self._collection_name)
                        return emb
                    self._embed_cache.pop(cache_key, None)

        # 3) Compute
        try:
            emb = await self._embeddings.aembed_query(query)
        except Exception:
            logger.exception("Embedding generation failed for collection '%s'", self._collection_name)
            return None

        # 4) Store in shared and local caches
        if emb:
            now = time.perf_counter()
            if self._shared_embed_cache is not None and self._shared_cache_lock is not None:
                async with self._shared_cache_lock:
                    self._shared_embed_cache[query_only_key] = (emb, now)
            if self._config.cache_enabled:
                async with self._cache_lock:
                    self._embed_cache[cache_key] = (emb, now)
                    self._cleanup_cache(self._embed_cache, self._config.cache_max_entries)
        return emb

    @staticmethod
    def _cleanup_cache(cache: Dict[str, Any], max_entries: int) -> None:
        if len(cache) > max_entries:
            oldest = sorted(cache.items(), key=lambda item: item[1][1])[: len(cache) - max_entries]
            for key, _ in oldest:
                cache.pop(key, None)

    def _expand_queries(self, query: str) -> List[str]:
        if not self._config.query_expansion_enabled:
            # Even when expansion is disabled, strip stopwords for cleaner keyword search
            cleaned = _remove_stopwords(query)
            return [cleaned if cleaned else query]
        normalized = " ".join(query.strip().split())
        lower = normalized.lower()
        umlaut = (
            lower.replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace("ß", "ss")
        )
        candidates = [normalized]
        for item in (lower, umlaut):
            if item and item not in candidates:
                candidates.append(item)
        # Strip stopwords from all candidates so keyword search stays focused on
        # meaningful terms rather than generic filler words.
        cleaned_candidates = []
        for c in candidates:
            stripped = _remove_stopwords(c)
            if stripped:
                cleaned_candidates.append(stripped)
            else:
                cleaned_candidates.append(c)
        return cleaned_candidates[: max(1, int(self._config.max_expanded_queries))]

    async def _query_vector_candidates(self, query_vector_literal: str, limit: int) -> List[Dict[str, Any]]:
        start = time.perf_counter()
        try:
            async with self._engine.connect() as conn:
                schema = await self._schema_cache.get(conn)
                if schema is None:
                    return []

                collection_uuid = await self._uuid_cache.get(conn, schema)
                if collection_uuid is None:
                    return []

                embedding_table = f"{_quote_identifier(schema)}.{_quote_identifier(LANGCHAIN_EMBEDDING_TABLE)}"

                res = await conn.execute(
                    text(
                        f"""
                        SELECT
                            (cmetadata->>'doc_id') AS doc_id,
                            document,
                            cmetadata,
                            1 - (embedding <=> (:query_vec)::vector) AS semantic_similarity
                        FROM {embedding_table}
                        WHERE collection_id = :collection_id
                        ORDER BY embedding <=> (:query_vec)::vector
                        LIMIT :limit
                        """
                    ),
                    {
                        "collection_id": collection_uuid,
                        "query_vec": query_vector_literal,
                        "limit": limit,
                    },
                )
                rows = res.mappings().all()

                result = [
                    {
                        "doc_id": row.get("doc_id") or "",
                        "document": row.get("document") or "",
                        "metadata": dict(row.get("cmetadata") or {}),
                        "semantic_similarity": row.get("semantic_similarity"),
                    }
                    for row in rows
                    if row.get("document")
                ]
                elapsed = (time.perf_counter() - start) * 1000
                logger.debug("Vector query: collection=%s schema=%s returned=%d time=%.2fms", self._collection_name, schema, len(result), elapsed)
                return result
        except SQLAlchemyError as exc:
            logger.exception("Vector candidate query failed for '%s'", self._collection_name)
            return []

    async def _query_keyword_candidates_multi(self, queries: Sequence[str], query_vector_literal: str, limit: int) -> List[Dict[str, Any]]:
        start = time.perf_counter()

        async def _run_one(q: str) -> List[Dict[str, Any]]:
            if not q:
                return []
            
            # Extract alphanumeric words and build an OR-based query to avoid AND collapse
            words = re.findall(r'\w+', q.lower())
            fts_query = " | ".join(words) if words else ""
            if not fts_query:
                fts_query = q

            try:
                async with self._engine.connect() as conn:
                    schema = await self._schema_cache.get(conn)
                    if schema is None:
                        return []

                    collection_uuid = await self._uuid_cache.get(conn, schema)
                    if collection_uuid is None:
                        return []

                    embedding_table = f"{_quote_identifier(schema)}.{_quote_identifier(LANGCHAIN_EMBEDDING_TABLE)}"

                    res = await conn.execute(
                        text(
                            f"""
                            SELECT
                                (cmetadata->>'doc_id') AS doc_id,
                                document,
                                cmetadata,
                                1 - (embedding <=> (:query_vec)::vector) AS semantic_similarity,
                                ts_rank_cd(to_tsvector('simple', document), to_tsquery('simple', :fts_q)) AS fts_score,
                                similarity(document, :q) AS trgm_score
                            FROM {embedding_table}
                            WHERE collection_id = :collection_id
                              AND (
                                  to_tsvector('simple', document) @@ to_tsquery('simple', :fts_q)
                                  OR document % :q
                              )
                            ORDER BY (ts_rank_cd(to_tsvector('simple', document), to_tsquery('simple', :fts_q)) * 1.0)
                                   + (similarity(document, :q) * 0.5)
                                   DESC
                            LIMIT :limit
                            """
                        ),
                        {
                            "collection_id": collection_uuid,
                            "query_vec": query_vector_literal,
                            "q": q,
                            "fts_q": fts_query,
                            "limit": limit,
                        },
                    )
                    rows = res.mappings().all()
                    return [
                        {
                            "doc_id": (row.get("doc_id") or "").strip(),
                            "document": row.get("document") or "",
                            "metadata": dict(row.get("cmetadata") or {}),
                            "semantic_similarity": row.get("semantic_similarity"),
                        }
                        for row in rows
                        if row.get("document")
                    ]
            except SQLAlchemyError:
                logger.exception("Keyword candidate query failed for '%s'", self._collection_name)
                return []

        # Execute expanded queries in parallel using separate connections from the pool
        results = await asyncio.gather(*[_run_one(q) for q in queries])
        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for rows in results:
            for row in rows:
                doc_id = row["doc_id"]
                if doc_id and doc_id in seen:
                    continue
                if doc_id:
                    seen.add(doc_id)
                merged.append(row)

        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(
            "Keyword query: collection=%s queries=%d returned=%d time=%.2fms",
            self._collection_name,
            len(queries),
            len(merged),
            elapsed,
        )
        return merged

    def _fuse_rrf(
        self,
        vector_rows: List[Dict[str, Any]],
        keyword_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rrf_k = max(1, int(self._config.rrf_k))
        def key(row: Dict[str, Any]) -> str:
            doc_id = (row.get("doc_id") or "").strip()
            if doc_id:
                return doc_id
            # Fallback: stable-ish key
            meta = row.get("metadata") or {}
            # include source and a deterministic hash of a document prefix for memory-safety
            doc_text = (row.get("document") or "")[:2048]
            h = hashlib.sha256(doc_text.encode("utf-8", errors="ignore")).hexdigest()
            return f"{meta.get('source','')}|{h}"

        scores: Dict[str, float] = {}
        chosen: Dict[str, Dict[str, Any]] = {}

        # Combine both lists in the same manner but preserve separate ranking within each
        def add_rows(rows: List[Dict[str, Any]]):
            for rank, row in enumerate(rows, start=1):
                k = key(row)
                chosen.setdefault(k, row)
                scores[k] = scores.get(k, 0.0) + 1.0 / (rrf_k + rank)

        add_rows(vector_rows)
        add_rows(keyword_rows)

        # deterministic ordering: primary by score desc, secondary by key
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [chosen[k] for k, _ in ordered]

    def _post_rank_boost(
        self,
        fused_rows: List[Dict[str, Any]],
        vector_rows: List[Dict[str, Any]],
        keyword_rows: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """Apply lightweight heuristic boosts after RRF fusion.

        - Documents present in both vector and keyword results get a bonus.
        - Documents with query terms in the source metadata get a bonus.
        - Documents whose collection matches query terms get a bonus.
        - Very short documents (< 30 chars) get a slight penalty.
        """
        if not fused_rows:
            return fused_rows

        # Identify doc_ids present in both vector and keyword
        vector_ids = {self._row_key(r) for r in vector_rows}
        keyword_ids = {self._row_key(r) for r in keyword_rows}
        both_ids = vector_ids & keyword_ids

        # Query terms for exact-match boosting (stopwords stripped so "was ist agentix"
        # gives us {"agentix"} rather than {"was", "ist", "agentix"})
        query_terms = set(_remove_stopwords(query).lower().strip().split())

        rank_map = {self._row_key(row): idx for idx, row in enumerate(fused_rows)}

        def score_with_boost(row: Dict[str, Any]) -> float:
            # Start with the RRF rank as a proxy score (higher rank = lower score)
            rank = rank_map.get(self._row_key(row), len(fused_rows))
            base_score = 1.0 / (1 + rank)

            k = self._row_key(row)
            if k in both_ids:
                base_score += 0.15

            meta = row.get("metadata") or {}
            source = str(meta.get("source") or "").lower()
            if source and any(term in source for term in query_terms):
                base_score += 0.10

            collection = str(meta.get("collection") or "").lower()
            if collection and any(term in collection for term in query_terms):
                base_score += 0.10

            doc_text = (row.get("document") or "").strip()
            if len(doc_text) < 30:
                base_score -= 0.03

            return base_score

        scored = [(row, score_with_boost(row)) for row in fused_rows]
        scored.sort(key=lambda item: (-item[1], self._row_key(item[0])))
        return [row for row, _ in scored]

    @staticmethod
    def _row_key(row: Dict[str, Any]) -> str:
        doc_id = (row.get("doc_id") or "").strip()
        if doc_id:
            return doc_id
        meta = row.get("metadata") or {}
        doc_text = (row.get("document") or "")[:2048]
        h = hashlib.sha256(doc_text.encode("utf-8", errors="ignore")).hexdigest()
        return f"{meta.get('source', '')}|{h}"

    def _rerank(
        self,
        fused_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rerank fused candidates by the pgvector cosine similarity.

        ``semantic_similarity`` is computed in SQL (``1 - (embedding <=> :vec)``)
        so this only consumes the precomputed score instead of parsing
        serialized vectors in Python.
        """
        if not fused_rows:
            return fused_rows

        scored_rows = []
        for row in fused_rows:
            sim_value = row.get("semantic_similarity")
            sim = float(sim_value) if sim_value is not None else 0.0
            if math.isnan(sim):
                # pgvector returns NaN for cosine distance on zero-norm vectors
                sim = 0.0

            # Enrich metadata with the calculated semantic similarity for diagnostic transparency
            meta = row.setdefault("metadata", {})
            meta["semantic_similarity"] = round(sim, 4)

            scored_rows.append((row, sim))

        # Sort by similarity score descending, with a stable secondary sort key
        scored_rows.sort(key=lambda item: (-item[1], self._row_key(item[0])))
        return [item[0] for item in scored_rows]

    def _apply_similarity_threshold(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out rows whose semantic similarity is below the configured minimum.

        If all rows fall below the threshold, an empty list is returned so the
        retriever avoids injecting irrelevant context into the RAG prompt.
        """
        threshold = float(self._config.min_semantic_similarity)
        if threshold <= 0.0:
            return rows
        filtered = [
            row
            for row in rows
            if (row.get("metadata") or {}).get("semantic_similarity", 0.0) >= threshold
        ]
        if not filtered:
            logger.debug(
                "All %d results for collection '%s' fall below similarity threshold %.3f; returning empty",
                len(rows),
                self._collection_name,
                threshold,
            )
            return []
        return filtered

    def _diversify(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        max_per_source = max(1, int(self._config.max_chunks_per_source))
        counts: Dict[str, int] = {}
        output: List[Dict[str, Any]] = []
        for row in rows:
            meta = row.get("metadata") or {}
            source = str(meta.get("source") or "unknown")
            counts[source] = counts.get(source, 0) + 1
            if counts[source] > max_per_source:
                continue
            output.append(row)
        return output

    async def aclose(self) -> None:
        """Gracefully dispose the internal AsyncEngine.

        This runs the sync `dispose()` method in a thread to avoid blocking the
        event loop. Call this during application shutdown when you want to
        explicitly release DB resources.
        """
        try:
            await asyncio.to_thread(self._engine.dispose)
        except Exception:
            logger.exception("Error disposing AsyncEngine for collection '%s'", self._collection_name)
