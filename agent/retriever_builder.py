"""Retriever construction and multi-collection routing utilities."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.retrievers import BaseRetriever
from sqlalchemy.ext.asyncio import AsyncEngine

from agent.config import AppConfig
from agent.hybrid_retriever import HybridPostgresRetriever, HybridRetrieverConfig
from agent.retrievers import CollectionRouterRetriever, TimeoutRetriever
from agent.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


class RetrieverBuilder:
    """Builds retrievers for single or multiple collections with optional hybrid search."""

    def __init__(
        self,
        config: AppConfig,
        vector_managers: Dict[str, VectorStoreManager],
        embeddings: Any,
        engine: Optional[AsyncEngine] = None,
    ) -> None:
        self.config = config
        self.vector_managers = vector_managers
        self.embeddings = embeddings
        self.engine = engine
        self._router_cache: Dict[Tuple[str, ...], CollectionRouterRetriever] = {}
        self._single_cache: Dict[str, BaseRetriever] = {}
        self._cache_lock = asyncio.Lock()
        # Shared embedding cache across all hybrid retriever instances to avoid
        # redundant HTTP calls when multiple collections query the same input.
        self._shared_embed_cache: Dict[str, tuple[Any, float]] = {}
        self._shared_cache_lock = asyncio.Lock()

    async def get_retriever_for_collections(
        self,
        requested: Sequence[str],
        retrieval_degraded: bool,
        *,
        search_k: Optional[int] = None,
    ) -> Optional[BaseRetriever]:
        """Return a retriever for the requested collections.

        Falls back to ``None`` when retrieval is degraded or no collections are available.
        For multiple collections a cached ``CollectionRouterRetriever`` is returned.

        ``search_k`` overrides the configured ``vector_search_k`` result count.
        Retrievers built with an explicit ``search_k`` bypass the caches so the
        shared default retrievers are never mutated.
        """
        if not requested or retrieval_degraded:
            return None

        available: List[str] = []
        missing: List[str] = []
        for collection in requested:
            if collection in self.vector_managers and self.vector_managers[collection].retriever is not None:
                available.append(collection)
            else:
                missing.append(collection)

        if missing:
            logger.warning("Requested collections not available: %s", ", ".join(missing))

        if not available:
            return None

        if search_k is not None:
            search_k = max(1, int(search_k))
            if len(available) == 1:
                return self._build_single(available[0], search_k=search_k)
            retrievers = {
                name: self._build_single(name, search_k=search_k) for name in available
            }
            return CollectionRouterRetriever(
                retrievers,
                search_k=search_k,
                per_retriever_timeout=getattr(self.config, "retriever_timeout_seconds", None),
            )

        if len(available) == 1:
            name = available[0]
            async with self._cache_lock:
                if name not in self._single_cache:
                    self._single_cache[name] = self._build_single(name)
                return self._single_cache[name]

        cache_key = tuple(sorted(available))
        async with self._cache_lock:
            if cache_key not in self._router_cache:
                retrievers: Dict[str, BaseRetriever] = {}
                for name in available:
                    if self.vector_managers[name].retriever is None:
                        continue
                    retrievers[name] = self._build_single(name)
                if not retrievers:
                    return None
                self._router_cache[cache_key] = CollectionRouterRetriever(
                    retrievers,
                    search_k=self.config.vector_search_k,
                    per_retriever_timeout=getattr(self.config, "retriever_timeout_seconds", None),
                )
            return self._router_cache[cache_key]

    def _build_single(self, name: str, *, search_k: Optional[int] = None) -> BaseRetriever:
        """Build a retriever for a single collection (hybrid or plain vector)."""
        retriever: BaseRetriever
        if (
            self.config.retrieval_hybrid_enabled
            and self.embeddings is not None
        ):
            cfg = HybridRetrieverConfig(
                search_k=self.config.vector_search_k,
                vector_candidates=self.config.hybrid_vector_candidates,
                keyword_candidates=self.config.hybrid_keyword_candidates,
                rrf_k=self.config.hybrid_rrf_k,
                max_chunks_per_source=self.config.hybrid_max_chunks_per_source,
                query_expansion_enabled=self.config.hybrid_query_expansion_enabled,
                max_expanded_queries=self.config.hybrid_max_expanded_queries,
                dynamic_candidates_enabled=True,
                post_rank_boost_enabled=True,
                cache_enabled=True,
                cache_ttl_seconds=60.0,
                cache_max_entries=256,
                rerank_enabled=self.config.hybrid_rerank_enabled,
                min_semantic_similarity=self.config.hybrid_min_semantic_similarity,
            )
            if search_k is not None:
                cfg = replace(
                    cfg,
                    search_k=search_k,
                    vector_candidates=max(cfg.vector_candidates, search_k),
                    keyword_candidates=max(cfg.keyword_candidates, search_k),
                )
            retriever = HybridPostgresRetriever(
                connection_string=self.config.connection_string,
                collection_name=name,
                embeddings=self.embeddings,
                config=cfg,
                engine=self.engine,
                shared_embed_cache=self._shared_embed_cache,
                shared_cache_lock=self._shared_cache_lock,
            )
        else:
            if search_k is not None:
                vectorstore = self.vector_managers[name].vectorstore
                if vectorstore is None:
                    raise RuntimeError(f"Retriever for collection '{name}' is not available")
                retriever = vectorstore.as_retriever(search_kwargs={"k": search_k})
            else:
                retriever = self.vector_managers[name].retriever
            if retriever is None:
                raise RuntimeError(f"Retriever for collection '{name}' is not available")

        timeout = getattr(self.config, "retriever_timeout_seconds", None)
        if timeout and float(timeout) > 0:
            retriever = TimeoutRetriever(retriever, timeout=float(timeout))
        return retriever

    async def clear_cache(self) -> None:
        """Drop all cached single and router retrievers."""
        async with self._cache_lock:
            self._single_cache.clear()
            self._router_cache.clear()
