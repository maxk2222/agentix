from __future__ import annotations

import pytest

from agent.hybrid_retriever import HybridPostgresRetriever, HybridRetrieverConfig


class TestPostRankBoost:
    def _make_retriever(self):
        # Minimal config for unit testing ranking logic
        config = HybridRetrieverConfig(
            search_k=5,
            post_rank_boost_enabled=True,
        )
        # We only need the retriever instance for its methods
        retriever = HybridPostgresRetriever.__new__(HybridPostgresRetriever)
        retriever._config = config
        return retriever

    def _make_row(self, doc_id: str, document: str, source: str = ""):
        return {
            "doc_id": doc_id,
            "document": document,
            "metadata": {"source": source} if source else {},
        }

    def test_boost_for_both_lists(self):
        retriever = self._make_retriever()
        vector_rows = [
            self._make_row("doc1", "important document about security", "BSI"),
            self._make_row("doc2", "another document", "NIST"),
        ]
        keyword_rows = [
            self._make_row("doc1", "important document about security", "BSI"),
            self._make_row("doc3", "third document", "ISO"),
        ]
        fused = retriever._fuse_rrf(vector_rows, keyword_rows)
        boosted = retriever._post_rank_boost(fused, vector_rows, keyword_rows, "security")

        # doc1 should be first because it appears in both lists
        assert boosted[0]["doc_id"] == "doc1"

    def test_penalty_for_short_documents(self):
        retriever = self._make_retriever()
        # doc2 appears in both lists (bonus), doc1 is very short (penalty)
        vector_rows = [
            self._make_row("doc1", "x"),
            self._make_row("doc2", "this is a normal length document with enough content"),
        ]
        keyword_rows = [
            self._make_row("doc2", "this is a normal length document with enough content"),
        ]
        fused = retriever._fuse_rrf(vector_rows, keyword_rows)
        boosted = retriever._post_rank_boost(fused, vector_rows, keyword_rows, "test")

        # doc2 should be preferred: it appears in both lists AND avoids short penalty
        assert boosted[0]["doc_id"] == "doc2"

    def test_source_term_boost(self):
        retriever = self._make_retriever()
        vector_rows = [
            self._make_row("doc1", "some content", "BSI-Grundschutz"),
            self._make_row("doc2", "other content", "NIST"),
        ]
        keyword_rows = []
        fused = retriever._fuse_rrf(vector_rows, keyword_rows)
        boosted = retriever._post_rank_boost(fused, vector_rows, keyword_rows, "bsi")

        # doc1 should be boosted because "bsi" is in its source
        assert boosted[0]["doc_id"] == "doc1"

    def test_empty_fused_returns_empty(self):
        retriever = self._make_retriever()
        result = retriever._post_rank_boost([], [], [], "query")
        assert result == []


class TestRowKey:
    def _make_retriever(self):
        config = HybridRetrieverConfig()
        retriever = HybridPostgresRetriever.__new__(HybridPostgresRetriever)
        retriever._config = config
        return retriever

    def test_doc_id_key(self):
        retriever = self._make_retriever()
        row = {"doc_id": "abc123", "document": "content", "metadata": {}}
        assert retriever._row_key(row) == "abc123"

    def test_fallback_key(self):
        retriever = self._make_retriever()
        row = {"doc_id": "", "document": "hello world", "metadata": {"source": "test.pdf"}}
        key = retriever._row_key(row)
        assert key.startswith("test.pdf|")
        assert len(key) > len("test.pdf|")


class TestCacheCleanup:
    def test_cleanup_removes_oldest(self):
        cache = {
            "a": (None, 1.0),
            "b": (None, 2.0),
            "c": (None, 3.0),
        }
        HybridPostgresRetriever._cleanup_cache(cache, max_entries=2)
        assert "a" not in cache
        assert "b" in cache
        assert "c" in cache


class TestSemanticRerank:
    def _make_retriever(self):
        config = HybridRetrieverConfig(rerank_enabled=True)
        retriever = HybridPostgresRetriever.__new__(HybridPostgresRetriever)
        retriever._config = config
        return retriever

    def test_semantic_rerank_orders_by_cosine_similarity(self):
        retriever = self._make_retriever()
        # semantic_similarity comes precomputed from SQL (1 - (embedding <=> :vec))
        fused_rows = [
            {"doc_id": "doc2", "document": "some content", "semantic_similarity": 0.0, "metadata": {}},
            {"doc_id": "doc1", "document": "matching content", "semantic_similarity": 1.0, "metadata": {}},
        ]

        reranked = retriever._rerank(fused_rows)

        # doc1 should come first now!
        assert reranked[0]["doc_id"] == "doc1"
        assert reranked[0]["metadata"]["semantic_similarity"] == 1.0
        assert reranked[1]["doc_id"] == "doc2"
        assert reranked[1]["metadata"]["semantic_similarity"] == 0.0

    def test_semantic_rerank_treats_missing_similarity_as_zero(self):
        retriever = self._make_retriever()
        fused_rows = [
            {"doc_id": "no-emb", "document": "keyword only", "semantic_similarity": None, "metadata": {}},
            {"doc_id": "doc1", "document": "vector hit", "semantic_similarity": 0.5, "metadata": {}},
        ]

        reranked = retriever._rerank(fused_rows)

        assert reranked[0]["doc_id"] == "doc1"
        assert reranked[1]["doc_id"] == "no-emb"
        assert reranked[1]["metadata"]["semantic_similarity"] == 0.0


class TestApplySimilarityThreshold:
    def _make_retriever(self, threshold: float = 0.0):
        config = HybridRetrieverConfig(min_semantic_similarity=threshold)
        retriever = HybridPostgresRetriever.__new__(HybridPostgresRetriever)
        retriever._config = config
        retriever._collection_name = "test-collection"
        return retriever

    def test_threshold_disabled_returns_all(self):
        retriever = self._make_retriever(0.0)
        rows = [
            {"doc_id": "doc1", "document": "content", "metadata": {"semantic_similarity": 0.3}},
            {"doc_id": "doc2", "document": "content", "metadata": {"semantic_similarity": 0.7}},
        ]
        result = retriever._apply_similarity_threshold(rows)
        assert len(result) == 2

    def test_threshold_filters_low_similarity(self):
        retriever = self._make_retriever(0.5)
        rows = [
            {"doc_id": "doc1", "document": "content", "metadata": {"semantic_similarity": 0.3}},
            {"doc_id": "doc2", "document": "content", "metadata": {"semantic_similarity": 0.7}},
        ]
        result = retriever._apply_similarity_threshold(rows)
        assert len(result) == 1
        assert result[0]["doc_id"] == "doc2"

    def test_threshold_returns_empty_when_all_below(self):
        retriever = self._make_retriever(0.9)
        rows = [
            {"doc_id": "doc1", "document": "content", "metadata": {"semantic_similarity": 0.3}},
            {"doc_id": "doc2", "document": "content", "metadata": {"semantic_similarity": 0.7}},
        ]
        result = retriever._apply_similarity_threshold(rows)
        assert result == []

    def test_missing_similarity_defaults_to_zero(self):
        retriever = self._make_retriever(0.1)
        rows = [
            {"doc_id": "doc1", "document": "content", "metadata": {}},
            {"doc_id": "doc2", "document": "content", "metadata": {"semantic_similarity": 0.5}},
        ]
        result = retriever._apply_similarity_threshold(rows)
        assert len(result) == 1
        assert result[0]["doc_id"] == "doc2"
