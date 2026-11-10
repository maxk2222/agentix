"""Custom LangChain retrievers for multi-collection routing."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import re
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple, Optional

from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

logger = logging.getLogger(__name__)


class MetricsRecordingRetriever(BaseRetriever):
    """Wrap any retriever and record timing / chunk count via a callback."""

    tags: List[str] = Field(default_factory=list)

    def __init__(self, retriever: BaseRetriever, callback: Optional[Callable[[dict[str, Any]], None]] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._retriever = retriever
        self._callback = callback

    def _get_relevant_documents(self, query: str) -> List[Document]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._aget_relevant_documents(query))
        raise RuntimeError(
            "MetricsRecordingRetriever cannot run synchronously inside an active event loop. "
            "Use the async interface instead."
        )

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        start = time.perf_counter()
        try:
            docs: List[Document] = []
            if hasattr(self._retriever, "ainvoke"):
                maybe = self._retriever.ainvoke(query)
                if inspect.isawaitable(maybe):
                    docs = list(await maybe)
                else:
                    docs = list(maybe)
            elif hasattr(self._retriever, "aget_relevant_documents"):
                maybe = self._retriever.aget_relevant_documents(query)
                if inspect.isawaitable(maybe):
                    docs = list(await maybe)
                else:
                    docs = list(maybe)
            elif hasattr(self._retriever, "get_relevant_documents"):
                docs = list(await asyncio.to_thread(self._retriever.get_relevant_documents, query))
            else:
                raise RuntimeError("Wrapped retriever does not expose a supported interface")

            elapsed = (time.perf_counter() - start) * 1000
            from app.chat_utils import estimate_rag_context_tokens
            total_tokens, per_doc_tokens = estimate_rag_context_tokens(docs)
            info = {
                "retrieval_time_ms": elapsed,
                "chunks_retrieved": len(docs),
                "status": "ok",
                "rag_context": [_serialize_document(doc) for doc in docs[:10]],
                "rag_context_tokens": total_tokens,
                "rag_context_doc_tokens": per_doc_tokens,
            }
            if self._callback:
                self._callback(info)
            return docs
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            info = {"retrieval_time_ms": elapsed, "chunks_retrieved": 0, "status": "error", "error_type": type(exc).__name__}
            if self._callback:
                self._callback(info)
            raise


class CollectionRouterRetriever(BaseRetriever):
    """Fan out retrieval requests across multiple vector stores and merge the results."""

    tags: List[str] = Field(default_factory=list)

    def __init__(self, retrievers: Dict[str, BaseRetriever], search_k: int = 5, **kwargs) -> None:
        super().__init__(**kwargs)
        if not retrievers:
            raise ValueError("CollectionRouterRetriever requires at least one retriever")
        self._retrievers = dict(retrievers)
        self._search_k = max(1, search_k)
        # Optional per-retriever timeout (seconds). If provided, each collection call
        # will be timeboxed to avoid a single slow backend blocking the fan-out.
        self._per_retriever_timeout: Optional[float] = None
        if "per_retriever_timeout" in kwargs:
            try:
                self._per_retriever_timeout = float(kwargs.pop("per_retriever_timeout"))
            except Exception:
                self._per_retriever_timeout = None

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Synchronous wrapper used by LangChain when async execution isn't requested."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._aget_relevant_documents(query))
        raise RuntimeError(
            "CollectionRouterRetriever cannot run synchronously inside an active event loop."
            " Use the async interface instead."
        )

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Gather results from every configured collection and return the top ``k`` items.

        Instead of concatenating results in collection order and blindly truncating,
        this merges all candidates, deduplicates them, and then sorts by a composite
        relevance score that respects cross-collection semantic similarity and boosts
        documents whose collection or source matches the query terms.
        """
        tasks = [self._invoke_retriever(name, retriever, query) for name, retriever in self._retrievers.items()]
        all_docs: List[Document] = []
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for name, result in zip(self._retrievers.keys(), results):
                if isinstance(result, Exception):
                    # Individual collection failures should not break the entire retrieval step.
                    logger.warning(
                        "Collection '%s' retriever failed, skipping for this query: %s",
                        name,
                        result,
                    )
                    continue
                all_docs.extend(result)

        deduped = _deduplicate(all_docs)
        if not deduped:
            return []

        scored = _score_and_sort(deduped, query)
        return scored[: self._search_k]

    async def _invoke_retriever(self, name: str, retriever: BaseRetriever, query: str) -> Sequence[Document]:
        """Call the target retriever via the modern Runnable interface."""
        async def _call() -> Sequence[Document]:
            # Preferred: LangChain Runnable async API.
            if hasattr(retriever, "ainvoke"):
                maybe = retriever.ainvoke(query)
                if inspect.isawaitable(maybe):
                    return await maybe
                return maybe

            # Backward-compatible retriever helper for older implementations.
            if hasattr(retriever, "aget_relevant_documents"):
                maybe = retriever.aget_relevant_documents(query)
                if inspect.isawaitable(maybe):
                    return await maybe
                return maybe

            # Fallback to synchronous call executed in a thread
            if hasattr(retriever, "get_relevant_documents"):
                return await asyncio.to_thread(retriever.get_relevant_documents, query)

            raise RuntimeError(f"Retriever for collection '{name}' does not expose a supported interface")

        try:
            if self._per_retriever_timeout and self._per_retriever_timeout > 0:
                return await asyncio.wait_for(_call(), timeout=self._per_retriever_timeout)
            return await _call()
        except asyncio.TimeoutError:
            raise TimeoutError(f"Retriever '{name}' timed out after {self._per_retriever_timeout}s")


class TimeoutRetriever(BaseRetriever):
    """Wrap any retriever and enforce a maximum duration per ``aget_relevant_documents`` call.

    Timeouts surface as ``asyncio.TimeoutError`` so upstream callers can degrade to
    LLM-only mode rather than hanging indefinitely.
    """

    def __init__(self, retriever: BaseRetriever, timeout: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self._retriever = retriever
        self._timeout = max(0.0, float(timeout))

    def _get_relevant_documents(self, query: str) -> List[Document]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._aget_relevant_documents(query))
        raise RuntimeError(
            "TimeoutRetriever cannot run synchronously inside an active event loop. "
            "Use the async interface instead."
        )

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        if self._timeout <= 0:
            return await self._retriever.ainvoke(query)
        return await asyncio.wait_for(
            self._retriever.ainvoke(query),
            timeout=self._timeout,
        )


class RecencyWeightedRetriever(BaseRetriever):
    """Two-phase retrieval that guarantees the latest user message dominates.

    Accepts either a plain query string (single pass-through to the wrapped
    retriever) or a dict with ``primary`` (the latest user message alone) and
    ``contextual`` (the history-mixed query). In the dict case both searches
    run in parallel and the merged result reserves the majority of slots for
    primary hits — history only backfills what the fresh intent did not find.
    """

    tags: List[str] = Field(default_factory=list)

    def __init__(self, retriever: BaseRetriever, primary_share: float = 0.6, **kwargs) -> None:
        super().__init__(**kwargs)
        self._retriever = retriever
        self._primary_share = max(0.0, min(1.0, float(primary_share)))

    def _get_relevant_documents(self, query: str) -> List[Document]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._aget_relevant_documents(query))
        raise RuntimeError(
            "RecencyWeightedRetriever cannot run synchronously inside an active event loop. "
            "Use the async interface instead."
        )

    async def _aget_relevant_documents(self, query: Any) -> List[Document]:
        primary: Optional[str] = None
        contextual: Optional[str] = None
        if isinstance(query, dict):
            primary = str(query.get("primary") or "").strip() or None
            contextual = str(query.get("contextual") or "").strip() or None
        else:
            contextual = str(query or "").strip() or None

        if primary is None:
            return await self._invoke(contextual or "")
        if contextual is None or contextual == primary:
            return await self._invoke(primary)

        primary_docs, contextual_docs = await asyncio.gather(
            self._invoke(primary), self._invoke(contextual)
        )

        # Reserve the primary share of the final slots, then backfill with the
        # best of both lists (contextual first so genuinely relevant history
        # hits survive). The cap matches the retriever's natural output size —
        # using only the primary lane's count would shrink results below what
        # a single query yields when the fresh intent finds few hits.
        k = max(len(primary_docs), len(contextual_docs), 1)
        quota = math.ceil(k * self._primary_share)
        merged = _deduplicate(
            list(primary_docs[:quota]) + list(contextual_docs) + list(primary_docs[quota:])
        )[:k]
        # Lane tag for observability — a hit found by the primary (latest
        # message) lane counts as primary even when the contextual lane also
        # surfaced it.
        primary_keys = {_dedup_key(doc) for doc in primary_docs}
        for doc in merged:
            doc.metadata["retrieval_lane"] = (
                "primary" if _dedup_key(doc) in primary_keys else "contextual"
            )
        return merged

    async def _invoke(self, query: str) -> List[Document]:
        maybe = self._retriever.ainvoke(query)
        if inspect.isawaitable(maybe):
            return list(await maybe)
        return list(maybe)


def _dedup_key(doc: Document) -> Tuple[str, str, str, str]:
    """Dedup key used across merge paths: filename/source/page metadata."""
    metadata = doc.metadata or {}
    return (
        str(metadata.get("collection", "")),
        str(metadata.get("source", "")),
        str(metadata.get("page", metadata.get("page_number", ""))),
        doc.page_content[:200],
    )


def _deduplicate(documents: Sequence[Document]) -> List[Document]:
    """Merge duplicate entries using filename/source/page metadata as the key."""
    seen: set[Tuple[str, str, str, str]] = set()
    merged: List[Document] = []
    for doc in documents:
        key = _dedup_key(doc)
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
    return merged


def _extract_query_terms(query: str) -> set[str]:
    """Return meaningful lower-cased query terms without stopwords / typos."""
    try:
        from agent.retrieval_utils import _remove_stopwords
    except Exception:
        _remove_stopwords = None  # type: ignore[assignment]

    if _remove_stopwords is not None:
        cleaned = _remove_stopwords(query)
    else:
        cleaned = query

    # Strip punctuation and split
    text = re.sub(r"[^\w\s]", " ", cleaned.lower())
    terms = {t for t in text.split() if t}
    # Drop very short tokens that are likely noise/typos (single char, or 2-char
    # tokens that are not meaningful acronyms like "it", "ai", "kmu")
    return {t for t in terms if len(t) > 2 or t in {"it", "ai", "kmu", "ddos", "bsi"}}


def _score_document(doc: Document, query_terms: set[str]) -> float:
    """Compute a composite relevance score for a document across collections.

    - Base: semantic_similarity from the HybridPostgresRetriever rerank step.
    - Boost: when query terms appear in the collection name or source path.
    """
    metadata = doc.metadata or {}
    base = float(metadata.get("semantic_similarity", 0.0))

    collection = str(metadata.get("collection", "")).lower()
    source = str(metadata.get("source", "")).lower()

    boost = 0.0
    for term in query_terms:
        if term in collection:
            boost += 0.30
        if term in source:
            boost += 0.15

    # Small tie-breaker: prefer documents with longer content (more informative)
    content_len = len((doc.page_content or "").strip())
    length_bonus = min(content_len / 5000.0, 0.05)

    return base + boost + length_bonus


def _score_and_sort(documents: List[Document], query: str) -> List[Document]:
    """Sort documents by composite relevance score descending."""
    query_terms = _extract_query_terms(query)
    scored = [(doc, _score_document(doc, query_terms)) for doc in documents]
    # Primary: score descending; Secondary: content length descending (stable)
    scored.sort(key=lambda item: (-item[1], -len(item[0].page_content or "")))
    if logger.isEnabledFor(logging.DEBUG):
        for doc, score in scored[:10]:
            meta = doc.metadata or {}
            logger.debug(
                "Router score: %.3f | coll=%s | source=%s | sim=%s",
                score,
                meta.get("collection", "?"),
                meta.get("source", "?"),
                meta.get("semantic_similarity", "?"),
            )
    return [doc for doc, _ in scored]


def _serialize_document(doc: Document) -> dict[str, Any]:
    metadata = dict(doc.metadata or {})
    return {
        "content": (doc.page_content or "")[:2000],
        "metadata": metadata,
    }
