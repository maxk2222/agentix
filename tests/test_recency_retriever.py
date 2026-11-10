"""Tests for agent.retrievers.RecencyWeightedRetriever."""

from __future__ import annotations

from typing import Dict, List

import pytest
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from agent.retrievers import RecencyWeightedRetriever


class QueryMapRetriever(BaseRetriever):
    """Fake retriever that returns different documents per query string."""

    calls: List[str] = Field(default_factory=list)

    def __init__(self, docs_by_query: Dict[str, List[Document]], **kwargs) -> None:
        super().__init__(**kwargs)
        self._docs = docs_by_query

    def _get_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        self.calls.append(query)
        return list(self._docs.get(query, []))


def _doc(content: str, source: str = "src") -> Document:
    return Document(page_content=content, metadata={"source": source})


class TestRecencyWeightedRetriever:
    @pytest.mark.asyncio
    async def test_string_query_passes_through_once(self):
        inner = QueryMapRetriever({"q": [_doc("a"), _doc("b")]})
        retriever = RecencyWeightedRetriever(inner)
        docs = await retriever.ainvoke("q")
        assert [d.page_content for d in docs] == ["a", "b"]
        assert inner.calls == ["q"]

    @pytest.mark.asyncio
    async def test_dict_query_runs_both_lanes(self):
        inner = QueryMapRetriever({"latest": [_doc("p1")], "ctx": [_doc("c1")]})
        retriever = RecencyWeightedRetriever(inner)
        await retriever.ainvoke({"primary": "latest", "contextual": "ctx"})
        assert set(inner.calls) == {"latest", "ctx"}

    @pytest.mark.asyncio
    async def test_primary_quota_then_backfill_deduped_and_capped(self):
        inner = QueryMapRetriever(
            {
                "latest": [_doc("p1"), _doc("p2"), _doc("dup")],
                "ctx": [_doc("dup"), _doc("c1"), _doc("c2")],
            }
        )
        retriever = RecencyWeightedRetriever(inner, primary_share=0.6)
        docs = await retriever.ainvoke({"primary": "latest", "contextual": "ctx"})
        contents = [d.page_content for d in docs]
        # k = len(primary) = 3; quota = ceil(3*0.6) = 2 → p1,p2 first,
        # then contextual+leftover primary deduped: dup,c1,c2 → capped to 3
        assert contents == ["p1", "p2", "dup"]
        lanes = {d.page_content: d.metadata.get("retrieval_lane") for d in docs}
        assert lanes["p1"] == "primary" and lanes["dup"] == "primary"

    @pytest.mark.asyncio
    async def test_contextual_fills_remaining_slots(self):
        inner = QueryMapRetriever(
            {
                "latest": [_doc("p1"), _doc("p2"), _doc("p3"), _doc("p4")],
                "ctx": [_doc("c1"), _doc("c2")],
            }
        )
        retriever = RecencyWeightedRetriever(inner, primary_share=0.5)
        docs = await retriever.ainvoke({"primary": "latest", "contextual": "ctx"})
        # k=4, quota=2 → p1,p2 + c1,c2 + p3(p4 cut)
        assert [d.page_content for d in docs] == ["p1", "p2", "c1", "c2"]

    @pytest.mark.asyncio
    async def test_sparse_primary_does_not_shrink_results(self):
        # When the fresh intent finds few hits, the contextual lane must still
        # backfill up to the retriever's natural size — capping at the primary
        # count would silently drop relevant history hits.
        inner = QueryMapRetriever(
            {
                "latest": [_doc("p1")],
                "ctx": [_doc("c1"), _doc("c2"), _doc("c3"), _doc("c4")],
            }
        )
        retriever = RecencyWeightedRetriever(inner, primary_share=0.6)
        docs = await retriever.ainvoke({"primary": "latest", "contextual": "ctx"})
        assert [d.page_content for d in docs] == ["p1", "c1", "c2", "c3"]

    @pytest.mark.asyncio
    async def test_empty_primary_falls_back_to_contextual(self):
        inner = QueryMapRetriever({"ctx": [_doc("c1")]})
        retriever = RecencyWeightedRetriever(inner)
        docs = await retriever.ainvoke({"primary": "", "contextual": "ctx"})
        assert [d.page_content for d in docs] == ["c1"]
        assert inner.calls == ["ctx"]

    @pytest.mark.asyncio
    async def test_identical_queries_run_once(self):
        inner = QueryMapRetriever({"same": [_doc("x")]})
        retriever = RecencyWeightedRetriever(inner)
        docs = await retriever.ainvoke({"primary": "same", "contextual": "same"})
        assert [d.page_content for d in docs] == ["x"]
        assert inner.calls == ["same"]

    @pytest.mark.asyncio
    async def test_share_one_keeps_only_primary(self):
        inner = QueryMapRetriever(
            {"latest": [_doc("p1"), _doc("p2")], "ctx": [_doc("c1")]}
        )
        retriever = RecencyWeightedRetriever(inner, primary_share=1.0)
        docs = await retriever.ainvoke({"primary": "latest", "contextual": "ctx"})
        assert [d.page_content for d in docs] == ["p1", "p2"]
