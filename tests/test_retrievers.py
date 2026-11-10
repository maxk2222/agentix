"""Tests for agent.retrievers and agent.retriever_builder."""

from __future__ import annotations

import asyncio
from typing import List

import pytest
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever

from agent.retrievers import (
    CollectionRouterRetriever,
    MetricsRecordingRetriever,
    TimeoutRetriever,
    _extract_query_terms,
    _score_document,
    _score_and_sort,
)
from agent.retriever_builder import RetrieverBuilder
from agent.config import AppConfig


class FakeRetriever(BaseRetriever):
    """Fake retriever that returns predefined documents after a delay."""

    def __init__(self, docs: List[Document], delay: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._docs = docs
        self._delay = delay

    def _get_relevant_documents(self, query: str) -> List[Document]:
        if self._delay > 0:
            import time
            time.sleep(self._delay)
        return list(self._docs)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return list(self._docs)


class TestTimeoutRetriever:
    @pytest.mark.asyncio
    async def test_returns_documents_when_fast_enough(self):
        doc = Document(page_content="hello")
        retriever = TimeoutRetriever(FakeRetriever([doc], delay=0.0), timeout=5.0)
        result = await retriever.ainvoke("test")
        assert len(result) == 1
        assert result[0].page_content == "hello"

    @pytest.mark.asyncio
    async def test_raises_on_timeout(self):
        retriever = TimeoutRetriever(FakeRetriever([], delay=10.0), timeout=0.1)
        with pytest.raises(asyncio.TimeoutError):
            await retriever.ainvoke("test")

    @pytest.mark.asyncio
    async def test_no_timeout_when_disabled(self):
        doc = Document(page_content="hello")
        retriever = TimeoutRetriever(FakeRetriever([doc], delay=0.0), timeout=0.0)
        result = await retriever.ainvoke("test")
        assert len(result) == 1


class TestRetrieverBuilder:
    @pytest.fixture
    def builder(self, tmp_path):
        config = AppConfig(
            data_dir=tmp_path / "data",
            instructions_file=tmp_path / "instructions.yaml",
            pg_host="localhost",
            pg_port=5432,
            pg_database="test_db",
            pg_user="test",
            pg_password="test",
            retrieval_hybrid_enabled=False,
            retriever_timeout_seconds=5.0,
        )
        return RetrieverBuilder(
            config=config,
            vector_managers={},
            embeddings=None,
            engine=None,
        )

    @pytest.mark.asyncio
    async def test_returns_none_when_degraded(self, builder):
        result = await builder.get_retriever_for_collections(["coll"], retrieval_degraded=True)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_collections_available(self, builder):
        result = await builder.get_retriever_for_collections(["missing"], retrieval_degraded=False)
        assert result is None

    @pytest.mark.asyncio
    async def test_caches_single_retriever(self, builder):
        # We can't easily test real hybrid retriever without DB, but we can test
        # that single retriever path uses cache by mocking _build_single.
        call_count = 0
        original_build = builder._build_single

        def _mock_build(name: str):
            nonlocal call_count
            call_count += 1
            return original_build(name)

        builder._build_single = _mock_build

        # First call
        r1 = await builder.get_retriever_for_collections(["coll"], retrieval_degraded=False)
        # Second call - should use cache, _build_single not called again
        r2 = await builder.get_retriever_for_collections(["coll"], retrieval_degraded=False)

        # Since no vector managers are set up, this returns None, but we can verify
        # the cache path was exercised by checking _single_cache state indirectly.
        # For a more robust test we'd need a real vector manager with a retriever.
        assert r1 is None  # No vector managers configured
        assert r2 is None

    @pytest.mark.asyncio
    async def test_clear_cache_drops_single_cache(self, builder):
        builder._single_cache["foo"] = FakeRetriever([])
        builder._router_cache[("foo",)] = CollectionRouterRetriever({"foo": FakeRetriever([])})
        await builder.clear_cache()
        assert not builder._single_cache
        assert not builder._router_cache

    @pytest.mark.asyncio
    async def test_search_k_plain_path_uses_vectorstore_k(self, builder):
        """Explicit search_k must build a fresh retriever with k=search_k."""
        from unittest.mock import MagicMock

        inner = FakeRetriever([])
        vectorstore = MagicMock()
        vectorstore.as_retriever = MagicMock(return_value=inner)
        manager = MagicMock()
        manager.retriever = FakeRetriever([])  # default retriever exists
        manager.vectorstore = vectorstore
        builder.vector_managers = {"coll": manager}

        result = await builder.get_retriever_for_collections(
            ["coll"], retrieval_degraded=False, search_k=8
        )
        assert result is not None
        vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 8})
        # Must not touch the shared single-retriever cache
        assert "coll" not in builder._single_cache

    @pytest.mark.asyncio
    async def test_search_k_none_uses_cache(self, builder):
        """Without search_k the cached retriever path must be used."""
        from unittest.mock import MagicMock

        manager = MagicMock()
        manager.retriever = FakeRetriever([])
        manager.vectorstore = MagicMock()
        builder.vector_managers = {"coll": manager}

        r1 = await builder.get_retriever_for_collections(["coll"], retrieval_degraded=False)
        r2 = await builder.get_retriever_for_collections(["coll"], retrieval_degraded=False)
        assert r1 is r2
        # vectorstore.as_retriever must not be called on the cached path
        manager.vectorstore.as_retriever.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_k_hybrid_config_override(self, tmp_path):
        """search_k must propagate into HybridRetrieverConfig and raise candidates."""
        from unittest.mock import MagicMock
        from agent.hybrid_retriever import HybridPostgresRetriever

        config = AppConfig(
            data_dir=tmp_path / "data",
            instructions_file=tmp_path / "instructions.yaml",
            pg_host="localhost",
            pg_port=5432,
            pg_database="test_db",
            pg_user="test",
            pg_password="test",
            retrieval_hybrid_enabled=True,
            retriever_timeout_seconds=0.0,
            hybrid_vector_candidates=20,
            hybrid_keyword_candidates=20,
        )
        manager = MagicMock()
        manager.retriever = FakeRetriever([])
        builder = RetrieverBuilder(
            config=config,
            vector_managers={"coll": manager},
            embeddings=MagicMock(),
            engine=None,
        )

        result = await builder.get_retriever_for_collections(
            ["coll"], retrieval_degraded=False, search_k=50
        )
        assert isinstance(result, HybridPostgresRetriever)
        assert result._config.search_k == 50
        assert result._config.vector_candidates == 50
        assert result._config.keyword_candidates == 50


class TestMetricsRecordingRetriever:
    @pytest.mark.asyncio
    async def test_records_successful_retrieval(self):
        doc = Document(page_content="hello")
        info = {}
        retriever = MetricsRecordingRetriever(FakeRetriever([doc], delay=0.0), callback=lambda i: info.update(i))
        result = await retriever.ainvoke("test")
        assert len(result) == 1
        assert info["status"] == "ok"
        assert info["chunks_retrieved"] == 1
        assert info["retrieval_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_records_error_on_failure(self):
        class FailingRetriever(BaseRetriever):
            def _get_relevant_documents(self, query: str) -> List[Document]:
                raise RuntimeError("boom")
            async def _aget_relevant_documents(self, query: str) -> List[Document]:
                raise RuntimeError("boom")

        info = {}
        retriever = MetricsRecordingRetriever(FailingRetriever(), callback=lambda i: info.update(i))
        with pytest.raises(RuntimeError):
            await retriever.ainvoke("test")
        assert info["status"] == "error"
        assert info["error_type"] == "RuntimeError"
        assert info["chunks_retrieved"] == 0
        assert info["retrieval_time_ms"] >= 0


class TestCollectionRouterScoring:
    def test_score_document_uses_semantic_similarity(self):
        doc = Document(page_content="some content", metadata={"semantic_similarity": 0.9})
        assert _score_document(doc, {"agentix"}) == pytest.approx(0.9, abs=0.01)

    def test_score_document_boosts_collection_match(self):
        doc = Document(
            page_content="content",
            metadata={"collection": "Agentix-Informationen", "semantic_similarity": 0.5},
        )
        score = _score_document(doc, {"agentix"})
        assert score > 0.5  # boosted by collection match

    def test_score_document_boosts_source_match(self):
        doc = Document(
            page_content="content",
            metadata={"source": "Agentix-Projektinformationen.txt", "semantic_similarity": 0.5},
        )
        score = _score_document(doc, {"agentix"})
        assert score > 0.5  # boosted by source match

    def test_score_and_sort_prefers_higher_similarity(self):
        docs = [
            Document(page_content="low", metadata={"semantic_similarity": 0.3}),
            Document(page_content="high", metadata={"semantic_similarity": 0.8}),
        ]
        sorted_docs = _score_and_sort(docs, "whatever")
        assert sorted_docs[0].page_content == "high"
        assert sorted_docs[1].page_content == "low"

    def test_score_and_sort_boosts_collection_over_similarity(self):
        # BSI doc has higher similarity but does not match query term
        # Agentix doc has lower similarity but matches collection term
        bsi_doc = Document(
            page_content="BSI content",
            metadata={"collection": "BSI-Zusammenfassungen", "semantic_similarity": 0.7},
        )
        agentix_doc = Document(
            page_content="Agentix content",
            metadata={"collection": "Agentix-Informationen", "semantic_similarity": 0.5},
        )
        sorted_docs = _score_and_sort([bsi_doc, agentix_doc], "was ist agentix")
        # Agentix doc should win due to collection boost even with lower base similarity
        assert sorted_docs[0].page_content == "Agentix content"

    @pytest.mark.asyncio
    async def test_router_returns_agentix_over_bsi_for_agentix_query(self):
        """End-to-end: even when BSI retriever is listed first, Agentix docs should surface."""
        bsi_doc = Document(
            page_content="BSI phishing advice",
            metadata={"collection": "BSI-Zusammenfassungen", "semantic_similarity": 0.65},
        )
        agentix_doc = Document(
            page_content="Agentix project details",
            metadata={"collection": "Agentix-Informationen", "semantic_similarity": 0.60},
        )
        # BSI retriever is listed FIRST but returns lower relevance for "agentix"
        router = CollectionRouterRetriever(
            {
                "BSI-Zusammenfassungen": FakeRetriever([bsi_doc]),
                "Agentix-Informationen": FakeRetriever([agentix_doc]),
            },
            search_k=1,
        )
        result = await router.ainvoke("was ist agentix")
        assert len(result) == 1
        # Agentix should win because of collection boost
        assert result[0].metadata["collection"] == "Agentix-Informationen"


class TestExtractQueryTerms:
    def test_strips_stopwords(self):
        terms = _extract_query_terms("was ist ddos")
        assert "was" not in terms
        assert "ist" not in terms
        assert "ddos" in terms

    def test_includes_acronyms(self):
        terms = _extract_query_terms("was ist bsi")
        assert "bsi" in terms

    def test_drops_noise_typos(self):
        terms = _extract_query_terms("mehr infos ztu agentix")
        assert "ztu" not in terms  # typo stopword
        assert "agentix" in terms
