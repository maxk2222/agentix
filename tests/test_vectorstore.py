"""Tests for agent.vectorstore."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.schema import Document

from agent.vectorstore import VectorStoreManager


class TestHashSourceDocuments:
    def test_empty_sequence(self):
        assert VectorStoreManager.hash_source_documents([]) == VectorStoreManager.hash_source_documents([])

    def test_stable_hash(self):
        docs = [Document(page_content="hello"), Document(page_content="world")]
        h1 = VectorStoreManager.hash_source_documents(docs)
        h2 = VectorStoreManager.hash_source_documents(docs)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_content(self):
        d1 = [Document(page_content="a")]
        d2 = [Document(page_content="b")]
        assert VectorStoreManager.hash_source_documents(d1) != VectorStoreManager.hash_source_documents(d2)


class TestSourceUnchanged:
    @pytest.fixture
    def manager(self):
        return VectorStoreManager(
            connection_string="postgresql+asyncpg://u:p@localhost/db",
            collection_name="test",
        )

    @pytest.mark.asyncio
    async def test_returns_none_when_unchanged(self, manager):
        docs = [Document(page_content="same")]
        with patch.object(
            manager,
            "_get_source_hash",
            new=AsyncMock(return_value=manager.hash_source_documents(docs)),
        ):
            result = await manager.source_unchanged("src.txt", docs)
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_new_hash_when_changed(self, manager):
        docs = [Document(page_content="new")]
        with patch.object(
            manager,
            "_get_source_hash",
            new=AsyncMock(return_value="oldhash"),
        ):
            result = await manager.source_unchanged("src.txt", docs)
            assert result == manager.hash_source_documents(docs)
            assert result != "oldhash"


class TestUpsertDocuments:
    @pytest.fixture
    def manager(self):
        return VectorStoreManager(
            connection_string="postgresql+asyncpg://u:p@localhost/db",
            collection_name="test",
        )

    @pytest.mark.asyncio
    async def test_skips_when_hash_unchanged(self, manager):
        docs = [Document(page_content="unchanged")]
        with patch.object(
            manager,
            "_get_source_hash",
            new=AsyncMock(return_value=manager.hash_source_documents(docs)),
        ):
            with patch.object(
                manager,
                "upsert_source_hash",
                new=AsyncMock(),
            ):
                result = await manager.upsert_documents("src.txt", docs, MagicMock())
                assert result == {"action": "skipped"}
                manager.upsert_source_hash.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_and_backfills_raw_file_hash(self, manager):
        docs = [Document(page_content="unchanged")]
        with patch.object(
            manager,
            "_get_source_hash",
            new=AsyncMock(return_value=manager.hash_source_documents(docs)),
        ):
            with patch.object(
                manager,
                "upsert_source_hash",
                new=AsyncMock(),
            ):
                result = await manager.upsert_documents(
                    "src.txt", docs, MagicMock(), raw_file_hash="rawhash123"
                )
                assert result == {"action": "skipped"}
                manager.upsert_source_hash.assert_awaited_once_with(
                    "src.txt", manager.hash_source_documents(docs), raw_file_hash="rawhash123"
                )

    @pytest.mark.asyncio
    async def test_skips_when_external_hash_matches(self, manager):
        docs = [Document(page_content="data")]
        external_hash = "matchinghash"
        with patch.object(
            manager,
            "_get_source_hash",
            new=AsyncMock(return_value=external_hash),
        ):
            result = await manager.upsert_documents("src.txt", docs, MagicMock(), source_hash=external_hash)
            assert result == {"action": "skipped"}

    @pytest.mark.asyncio
    async def test_updates_when_hash_changed(self, manager):
        docs = [Document(page_content="changed")]
        with patch.object(
            manager,
            "_get_source_hash",
            new=AsyncMock(return_value="oldhash"),
        ):
            manager.vectorstore = MagicMock()
            manager.vectorstore.aadd_documents = AsyncMock()
            with patch.object(
                manager,
                "_delete_embeddings_for_source",
                new=AsyncMock(return_value=3),
            ):
                with patch.object(
                    manager,
                    "upsert_source_hash",
                    new=AsyncMock(),
                ):
                    result = await manager.upsert_documents(
                        "src.txt", docs, MagicMock(), raw_file_hash="rawhash456"
                    )
                    assert result["action"] == "updated"
                    assert result["deleted"] == 3
                    assert result["added"] == 1
                    manager.vectorstore.aadd_documents.assert_awaited_once_with(docs)
                    manager.upsert_source_hash.assert_awaited_once_with(
                        "src.txt", manager.hash_source_documents(docs), raw_file_hash="rawhash456"
                    )

    @pytest.mark.asyncio
    async def test_uses_external_hash_when_provided(self, manager):
        docs = [Document(page_content="data")]
        external_hash = "externalhash123"
        with patch.object(
            manager,
            "_get_source_hash",
            new=AsyncMock(return_value="oldhash"),
        ):
            manager.vectorstore = MagicMock()
            manager.vectorstore.aadd_documents = AsyncMock()
            with patch.object(
                manager,
                "_delete_embeddings_for_source",
                new=AsyncMock(return_value=2),
            ):
                with patch.object(
                    manager,
                    "upsert_source_hash",
                    new=AsyncMock(),
                ):
                    result = await manager.upsert_documents(
                        "src.txt", docs, MagicMock(), source_hash=external_hash, raw_file_hash="rawhash789"
                    )
                    assert result["action"] == "updated"
                    assert result["deleted"] == 2
                    assert result["added"] == 1
                    manager.vectorstore.aadd_documents.assert_awaited_once_with(docs)
                    manager.upsert_source_hash.assert_awaited_once_with(
                        "src.txt", external_hash, raw_file_hash="rawhash789"
                    )


class TestGetRawFileHashes:
    @pytest.fixture
    def manager(self):
        return VectorStoreManager(
            connection_string="postgresql+asyncpg://u:p@localhost/db",
            collection_name="test",
        )

    @pytest.mark.asyncio
    async def test_delegates_to_get_raw_file_hash_for_single_source(self, manager):
        with patch.object(
            manager,
            "get_raw_file_hashes",
            new=AsyncMock(return_value={"src.txt": "hash123"}),
        ):
            result = await manager.get_raw_file_hash("src.txt")
            assert result == "hash123"
            manager.get_raw_file_hashes.assert_awaited_once_with(["src.txt"])


class TestPrepareMetadataStore:
    @pytest.fixture
    def manager(self):
        return VectorStoreManager(
            connection_string="postgresql+asyncpg://u:p@localhost/db",
            collection_name="test",
        )

    @pytest.mark.asyncio
    async def test_calls_ensure_schema_and_support_tables(self, manager):
        with patch.object(manager, "_ensure_schema", new=AsyncMock()):
            with patch.object(manager, "_ensure_support_tables_and_indexes", new=AsyncMock()):
                await manager.prepare_metadata_store()
                manager._ensure_schema.assert_awaited_once()
                manager._ensure_support_tables_and_indexes.assert_awaited_once()
