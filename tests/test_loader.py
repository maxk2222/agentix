from pathlib import Path

import pytest
from langchain.schema import Document

from agent.loader import KnowledgeLoader


def test_merge_documents_by_source_combines_fragments_in_order() -> None:
    loader = KnowledgeLoader(folder_path=Path("."))

    docs = [
        Document(page_content="Einleitung", metadata={"source": "report.pdf", "page": 1}),
        Document(page_content="Details A", metadata={"source": "report.pdf", "page": 2}),
        Document(page_content="Details B", metadata={"source": "report.pdf", "page": 2}),
    ]

    merged = loader._merge_documents_by_source(docs)

    assert len(merged) == 1
    assert merged[0].page_content == "Einleitung\n\nDetails A\n\nDetails B"
    assert merged[0].metadata["source"] == "report.pdf"
    assert merged[0].metadata["page_numbers"] == [1, 2]
    assert merged[0].metadata["page_count"] == 2
    assert merged[0].metadata["element_count"] == 3


def test_merge_documents_by_source_keeps_separate_sources() -> None:
    loader = KnowledgeLoader(folder_path=Path("."))

    docs = [
        Document(page_content="Alpha", metadata={"source": "a.pdf"}),
        Document(page_content="Beta", metadata={"source": "b.pdf"}),
    ]

    merged = loader._merge_documents_by_source(docs)

    assert len(merged) == 2
    assert merged[0].page_content == "Alpha"
    assert merged[1].page_content == "Beta"


@pytest.mark.asyncio
async def test_load_local_documents_reports_progress_and_counts(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")

    loader = KnowledgeLoader(folder_path=tmp_path)
    events: list[tuple[int, int, str]] = []

    docs = await loader.load_local_documents(
        progress_cb=lambda done, total, path: events.append((done, total, path))
    )

    assert len(docs) == 2
    assert len(events) == 2
    assert events[-1][0] == 2 and events[-1][1] == 2


@pytest.mark.asyncio
async def test_load_local_documents_times_out_stalled_file(tmp_path: Path) -> None:
    import asyncio
    from unittest.mock import patch

    (tmp_path / "ok.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "stuck.pdf").write_text("%PDF-1.4 fake", encoding="utf-8")

    loader = KnowledgeLoader(folder_path=tmp_path)

    async def _never_finishes(*args, **kwargs):
        await asyncio.sleep(3600)
        return []

    with patch("agent.loader.UnstructuredLoader") as mock_loader_cls:
        mock_loader_cls.return_value.aload = _never_finishes
        docs = await loader.load_local_documents(file_timeout=0.05)

    sources = [d.metadata.get("source") for d in docs]
    assert str(tmp_path / "ok.md") in sources
    assert not any(str(tmp_path / "stuck.pdf") in (s or "") for s in sources)