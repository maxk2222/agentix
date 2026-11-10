"""Tests for the rag_search builtin tool."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.schema import Document

from agent.tools.builtin_tools.rag_search import RagSearchTool


class TestRagSearchTool:
    def _make_tool(self):
        rag_app = MagicMock()
        rag_app._tools_store = None
        rag_app._retriever_builder = MagicMock()
        rag_app.engine = MagicMock()
        # Let RawFileStore queries resolve cleanly (no real DB round-trip):
        # conn.execute -> res whose .mappings().first()/.all() return empty.
        _conn = rag_app.engine.begin.return_value.__aenter__.return_value
        _res = MagicMock()
        _res.mappings.return_value.first.return_value = None
        _res.mappings.return_value.all.return_value = []
        _conn.execute = AsyncMock(return_value=_res)
        rag_app.engine.connect.return_value = rag_app.engine.begin.return_value
        return RagSearchTool(rag_app)

    def _patch_collections(self, tool, collection_names=None, max_top_k=10):
        """Patch _get_collection_names and set max_top_k."""
        if collection_names is None:
            collection_names = ["test_collection"]
        tool._collection_names = collection_names
        tool._max_top_k = max_top_k
        tool._get_collection_names = AsyncMock(return_value=collection_names)

    @pytest.mark.asyncio
    async def test_tool_schema(self):
        tool = self._make_tool()
        self._patch_collections(tool)
        ts = tool.tool_schema
        assert ts["type"] == "function"
        assert ts["function"]["name"] == "rag_search"
        props = ts["function"]["parameters"]["properties"]
        assert "query" in props
        assert "collection" in props
        assert "top_k" in props
        assert "action" in props
        assert props["action"]["enum"] == ["search", "list", "read", "missing"]
        assert "source" in props
        assert "document_collection" in props
        assert "patterns" in props
        assert "limit" in props
        assert "offset" in props
        assert "lines_start" in props
        assert "lines_end" in props
        assert props["top_k"]["maximum"] == 10
        assert ts["function"]["parameters"]["required"] == ["collection"]

    @pytest.mark.asyncio
    async def test_refresh_schema_populates_enum(self):
        tool = self._make_tool()
        with patch.object(
            tool, "_get_collection_names", new=AsyncMock(return_value=["col_a", "col_b"])
        ):
            with patch.object(
                tool, "_get_tool_config", new=AsyncMock(return_value={"max_top_k": 20})
            ):
                await tool.refresh_schema()
        ts = tool.tool_schema
        props = ts["function"]["parameters"]["properties"]
        assert props["collection"].get("enum") == ["col_a", "col_b"]
        assert props["top_k"]["maximum"] == 20
        assert "col_a, col_b" in ts["function"]["description"]

    @pytest.mark.asyncio
    async def test_refresh_schema_empty_collections(self):
        tool = self._make_tool()
        with patch.object(
            tool, "_get_collection_names", new=AsyncMock(return_value=[])
        ):
            with patch.object(
                tool, "_get_tool_config", new=AsyncMock(return_value={})
            ):
                await tool.refresh_schema()
        ts = tool.tool_schema
        props = ts["function"]["parameters"]["properties"]
        assert "enum" not in props["collection"]
        assert "Configured collections" not in ts["function"]["description"]

    @pytest.mark.asyncio
    async def test_refresh_schema_no_store(self):
        tool = self._make_tool()
        tool._rag_app._tools_store = None
        await tool.refresh_schema()
        assert tool._collection_names == []
        assert tool._max_top_k == 10
        ts = tool.tool_schema
        props = ts["function"]["parameters"]["properties"]
        assert "enum" not in props["collection"]

    @pytest.mark.asyncio
    async def test_no_collections_configured(self):
        tool = self._make_tool()
        self._patch_collections(tool, [])
        result = await tool.execute("test query", collection="test_collection")
        assert "No collections configured" in result

    @pytest.mark.asyncio
    async def test_invalid_collection(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["allowed_collection"])
        result = await tool.execute("test query", collection="forbidden_collection")
        assert "is not allowed" in result
        assert "allowed_collection" in result

    @pytest.mark.asyncio
    async def test_retriever_not_available(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["allowed_collection"])
        tool._rag_app._retriever_builder.get_retriever_for_collections = AsyncMock(
            return_value=None
        )
        result = await tool.execute("test query", collection="allowed_collection")
        assert "Retriever for collection" in result
        assert "is not available" in result

    @pytest.mark.asyncio
    async def test_retriever_builder_not_available(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["allowed_collection"])
        tool._rag_app._retriever_builder = None
        result = await tool.execute("test query", collection="allowed_collection")
        assert "RAG retriever is not available" in result

    @pytest.mark.asyncio
    async def test_execute_success(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["allowed_collection"])

        fake_doc = Document(
            page_content="Test content",
            metadata={"source": "doc.md", "page": 1, "chunk_index": 0},
        )
        fake_retriever = MagicMock()
        fake_retriever.ainvoke = AsyncMock(return_value=[fake_doc])

        tool._rag_app._retriever_builder.get_retriever_for_collections = AsyncMock(
            return_value=fake_retriever
        )

        result = await tool.execute("test query", collection="allowed_collection")
        assert "Test content" in result
        assert "Source: doc.md" in result
        assert "Page: 1" in result
        assert "Chunk: 0" in result

    @pytest.mark.asyncio
    async def test_execute_top_k_clamping(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["allowed_collection"], max_top_k=3)

        fake_docs = [
            Document(page_content=f"doc {i}", metadata={"source": f"{i}.md"})
            for i in range(10)
        ]
        fake_retriever = MagicMock()
        fake_retriever.ainvoke = AsyncMock(return_value=fake_docs)

        tool._rag_app._retriever_builder.get_retriever_for_collections = AsyncMock(
            return_value=fake_retriever
        )

        result = await tool.execute(
            "test query", collection="allowed_collection", top_k=100
        )
        # Should only include 3 results due to clamping
        assert result.count("Source:") == 3

    @pytest.mark.asyncio
    async def test_execute_top_k_zero_clamped_to_one(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["allowed_collection"], max_top_k=5)

        fake_docs = [
            Document(page_content=f"doc {i}", metadata={"source": f"{i}.md"})
            for i in range(2)
        ]
        fake_retriever = MagicMock()
        fake_retriever.ainvoke = AsyncMock(return_value=fake_docs)

        tool._rag_app._retriever_builder.get_retriever_for_collections = AsyncMock(
            return_value=fake_retriever
        )

        result = await tool.execute(
            "test query", collection="allowed_collection", top_k=0
        )
        assert result.count("Source:") == 1

    @pytest.mark.asyncio
    async def test_execute_no_results(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["allowed_collection"])

        fake_retriever = MagicMock()
        fake_retriever.ainvoke = AsyncMock(return_value=[])

        tool._rag_app._retriever_builder.get_retriever_for_collections = AsyncMock(
            return_value=fake_retriever
        )

        result = await tool.execute("test query", collection="allowed_collection")
        assert "No results found" in result

    @pytest.mark.asyncio
    async def test_execute_truncation(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["allowed_collection"])

        long_content = "A" * 20_000
        fake_doc = Document(page_content=long_content, metadata={"source": "doc.md"})
        fake_retriever = MagicMock()
        fake_retriever.ainvoke = AsyncMock(return_value=[fake_doc])

        tool._rag_app._retriever_builder.get_retriever_for_collections = AsyncMock(
            return_value=fake_retriever
        )

        result = await tool.execute("test query", collection="allowed_collection")
        assert "truncated" in result
        assert len(result) <= 15_400

    def test_format_results(self):
        docs = [
            Document(
                page_content="Hello",
                metadata={"source": "a.md", "page": 1, "chunk_index": 0},
            ),
            Document(
                page_content="World",
                metadata={"source": "b.md"},
            ),
            Document(page_content="Bare"),
        ]
        result = RagSearchTool._format_results(docs)
        assert "Source: a.md, Page: 1, Chunk: 0\nHello" in result
        assert "Source: b.md\nWorld" in result
        assert "Bare" in result
        assert "Source:" not in result.split("Bare")[0].split("World")[1]

    @pytest.mark.asyncio
    async def test_execute_with_context_override(self):
        tool = self._make_tool()
        # DB config has no collections
        tool._rag_app._tools_store = None
        # But assistant override provides them
        fake_doc = Document(page_content="override content", metadata={"source": "o.md"})
        fake_retriever = MagicMock()
        fake_retriever.ainvoke = AsyncMock(return_value=[fake_doc])
        tool._rag_app._retriever_builder.get_retriever_for_collections = AsyncMock(
            return_value=fake_retriever
        )

        result = await tool.execute(
            "test query",
            collection="override_col",
            context={"tool_config": {"collection_names": ["override_col"]}},
        )
        assert "override content" in result

    # --- schema_with_config tests ---

    def test_schema_with_config_collections(self):
        tool = self._make_tool()
        tool._collection_names = None  # Simulate no DB config
        schema = tool.schema_with_config({"collection_names": ["CCL-Publikationen", "BSI"]})
        props = schema["function"]["parameters"]["properties"]
        assert props["collection"].get("enum") == ["CCL-Publikationen", "BSI"]
        assert "CCL-Publikationen, BSI" in schema["function"]["description"]

    def test_schema_with_config_no_collections(self):
        tool = self._make_tool()
        tool._collection_names = None
        schema = tool.schema_with_config({})
        props = schema["function"]["parameters"]["properties"]
        assert "enum" not in props["collection"]
        assert "Configured collections" not in schema["function"]["description"]

    def test_schema_with_config_max_top_k(self):
        tool = self._make_tool()
        schema = tool.schema_with_config({"max_top_k": 3})
        props = schema["function"]["parameters"]["properties"]
        assert props["top_k"]["maximum"] == 3

    def test_schema_with_config_falls_back_to_cached(self):
        tool = self._make_tool()
        tool._collection_names = ["cached_col"]
        tool._max_top_k = 7
        schema = tool.schema_with_config({})
        props = schema["function"]["parameters"]["properties"]
        assert props["collection"].get("enum") == ["cached_col"]
        assert props["top_k"]["maximum"] == 7

    def test_tool_schema_delegates_to_schema_with_config(self):
        tool = self._make_tool()
        tool._collection_names = ["db_col"]
        tool._max_top_k = 5
        s1 = tool.tool_schema
        s2 = tool.schema_with_config({})
        assert s1 == s2

    # --- ToolRegistry.tool_schemas with overrides ---

    def test_registry_tool_schemas_with_overrides(self):
        """ToolRegistry.tool_schemas should use schema_with_config when overrides are present."""
        from agent.tools import ToolRegistry
        rag_app = MagicMock()
        rag_app._tools_store = None
        rag_app._retriever_builder = None
        tool = RagSearchTool(rag_app)
        tool._collection_names = None
        registry = ToolRegistry()
        registry.register_builtin("rag_search", tool)
        overrides = {"rag_search": {"collection_names": ["override_col"]}}
        schemas = registry.tool_schemas(["rag_search"], tool_overrides=overrides)
        assert len(schemas) == 1
        props = schemas[0]["function"]["parameters"]["properties"]
        assert props["collection"].get("enum") == ["override_col"]

    def test_registry_tool_schemas_without_overrides(self):
        """ToolRegistry.tool_schemas should fall back to tool_schema property when no overrides."""
        from agent.tools import ToolRegistry
        rag_app = MagicMock()
        rag_app._tools_store = None
        rag_app._retriever_builder = None
        tool = RagSearchTool(rag_app)
        tool._collection_names = ["db_col"]
        registry = ToolRegistry()
        registry.register_builtin("rag_search", tool)
        schemas = registry.tool_schemas(["rag_search"])
        assert len(schemas) == 1
        props = schemas[0]["function"]["parameters"]["properties"]
        assert props["collection"].get("enum") == ["db_col"]

    # --- String collection_names parsing (dashboard KeyValueEditor) ---

    def test_schema_with_config_string_collection(self):
        """Dashboard stores collection_names as string; schema should still show enum."""
        tool = self._make_tool()
        tool._collection_names = None
        schema = tool.schema_with_config({"collection_names": "CCL-Publikationen"})
        props = schema["function"]["parameters"]["properties"]
        assert props["collection"].get("enum") == ["CCL-Publikationen"]
        assert "CCL-Publikationen" in schema["function"]["description"]

    def test_schema_with_config_json_string_collection(self):
        """JSON array string like '["col_a", "col_b"]' should be parsed correctly."""
        tool = self._make_tool()
        tool._collection_names = None
        schema = tool.schema_with_config({"collection_names": '["col_a", "col_b"]'})
        props = schema["function"]["parameters"]["properties"]
        assert props["collection"].get("enum") == ["col_a", "col_b"]

    def test_schema_with_config_comma_separated_string(self):
        """Comma-separated string 'col_a, col_b' should be parsed correctly."""
        tool = self._make_tool()
        tool._collection_names = None
        schema = tool.schema_with_config({"collection_names": "col_a, col_b"})
        props = schema["function"]["parameters"]["properties"]
        assert props["collection"].get("enum") == ["col_a", "col_b"]

    @pytest.mark.asyncio
    async def test_execute_with_string_collection_names(self):
        """execute() should work when collection_names comes as string from dashboard."""
        tool = self._make_tool()
        tool._rag_app._tools_store = None
        fake_doc = Document(page_content="test content", metadata={"source": "t.md"})
        fake_retriever = MagicMock()
        fake_retriever.ainvoke = AsyncMock(return_value=[fake_doc])
        tool._rag_app._retriever_builder.get_retriever_for_collections = AsyncMock(
            return_value=fake_retriever
        )
        result = await tool.execute(
            "query",
            collection="CCL-Publikationen",
            context={"tool_config": {"collection_names": "CCL-Publikationen"}},
        )
        assert "test content" in result

    def test_normalize_collection_names_list(self):
        from agent.tools.builtin_tools.common import normalize_collection_names
        assert normalize_collection_names(["a", "b"]) == ["a", "b"]

    def test_normalize_collection_names_string(self):
        from agent.tools.builtin_tools.common import normalize_collection_names
        assert normalize_collection_names("CCL-Publikationen") == ["CCL-Publikationen"]

    def test_normalize_collection_names_json_string(self):
        from agent.tools.builtin_tools.common import normalize_collection_names
        assert normalize_collection_names('["a", "b"]') == ["a", "b"]

    def test_normalize_collection_names_comma_string(self):
        from agent.tools.builtin_tools.common import normalize_collection_names
        assert normalize_collection_names("a, b, c") == ["a", "b", "c"]

    def test_normalize_collection_names_empty(self):
        from agent.tools.builtin_tools.common import normalize_collection_names
        assert normalize_collection_names("") == []
        assert normalize_collection_names([]) == []
        assert normalize_collection_names(None) == []

    # --- action=list tests ---

    @pytest.mark.asyncio
    async def test_list_documents(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        with patch("app.stores.RawFileStore") as MockStore:
            instance = MockStore.return_value
            instance.list_by_collection = AsyncMock(return_value=[
                {
                    "source": "test_collection/doc1.md",
                    "filename": "doc1.md",
                    "content_type": "text/markdown",
                    "file_hash": "abc123def",
                    "updated_at": "2024-01-01T00:00:00",
                },
                {
                    "source": "test_collection/doc2.md",
                    "filename": "doc2.md",
                    "content_type": "text/markdown",
                    "file_hash": "xyz789ghi",
                    "updated_at": "2024-02-01T00:00:00",
                },
            ])
            result = await tool.execute(collection="test_collection", action="list")
        assert "Documents in collection 'test_collection' (2 files):" in result
        assert "doc1.md" in result
        assert "doc2.md" in result
        assert "abc123de" in result
        assert "text/markdown" in result

    @pytest.mark.asyncio
    async def test_list_documents_empty(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        tool._rag_app._index_manager = None
        with patch("app.stores.RawFileStore") as MockStore:
            instance = MockStore.return_value
            instance.list_by_collection = AsyncMock(return_value=[])
            result = await tool.execute(collection="test_collection", action="list")
        assert "No documents found" in result

    @pytest.mark.asyncio
    async def test_list_documents_vector_fallback(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        mock_manager = MagicMock()
        mock_manager.prepare_metadata_store = AsyncMock()
        mock_manager.list_embedding_sources = AsyncMock(return_value=[
            {"source": "doc1.md", "chunks": 12},
            {"source": "doc2.md", "chunks": 5},
        ])
        tool._rag_app.vector_managers = {"test_collection": mock_manager}
        with patch("app.stores.RawFileStore") as MockStore:
            instance = MockStore.return_value
            instance.list_by_collection = AsyncMock(return_value=[])
            result = await tool.execute(collection="test_collection", action="list")
        assert "Documents in collection 'test_collection'" in result
        assert "doc1.md" in result
        assert "doc2.md" in result
        assert "(12 chunks)" in result
        assert "(5 chunks)" in result

    # --- action=missing tests ---

    @pytest.mark.asyncio
    async def test_missing_reports_uncovered_files(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["pubs", "overview_col"])
        with patch("app.stores.RawFileStore") as MockStore:
            instance = MockStore.return_value
            instance.list_by_collection = AsyncMock(return_value=[
                {"source": "pubs/a.md", "filename": "a.md"},
                {"source": "pubs/b.md", "filename": "b.md"},
                {"source": "pubs/c.md", "filename": "c.md"},
            ])
            instance.get = AsyncMock(return_value={
                "file_data": "Overview\n- a.md\n- b.md is also covered",
            })
            result = await tool.execute(
                collection="pubs",
                action="missing",
                source="overview.md",
                document_collection="overview_col",
            )
        assert "Coverage check" in result
        assert "Missing (1)" in result
        assert "- c.md" in result
        assert "Covered: 2/3" in result

    @pytest.mark.asyncio
    async def test_missing_stem_match(self):
        """A file counts as covered when its stem appears in the document."""
        tool = self._make_tool()
        self._patch_collections(tool, ["pubs"])
        with patch("app.stores.RawFileStore") as MockStore:
            instance = MockStore.return_value
            instance.list_by_collection = AsyncMock(return_value=[
                {"source": "pubs/annual-report.pdf", "filename": "annual-report.pdf"},
                {"source": "pubs/other.pdf", "filename": "other.pdf"},
            ])
            instance.get = AsyncMock(return_value={
                "file_data": "See the annual-report publication for details.",
            })
            result = await tool.execute(
                collection="pubs", action="missing", source="overview.md",
            )
        assert "Missing (1)" in result
        assert "- other.pdf" in result
        assert "annual-report.pdf" in result  # listed under stem-only matches
        assert "Matched by stem only" in result

    @pytest.mark.asyncio
    async def test_missing_all_covered(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["pubs"])
        with patch("app.stores.RawFileStore") as MockStore:
            instance = MockStore.return_value
            instance.list_by_collection = AsyncMock(return_value=[
                {"source": "pubs/a.md", "filename": "a.md"},
            ])
            instance.get = AsyncMock(return_value={"file_data": "- a.md"})
            result = await tool.execute(
                collection="pubs", action="missing", source="overview.md",
            )
        assert "Missing: none" in result
        assert "Covered: 1/1" in result

    @pytest.mark.asyncio
    async def test_missing_document_not_found(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["pubs"])
        tool._rag_app._index_manager = None
        tool._rag_app.vector_managers = {}
        with patch("app.stores.RawFileStore") as MockStore:
            instance = MockStore.return_value
            instance.list_by_collection = AsyncMock(return_value=[
                {"source": "pubs/a.md", "filename": "a.md"},
            ])
            instance.get = AsyncMock(return_value=None)
            result = await tool.execute(
                collection="pubs", action="missing", source="nope.md",
            )
        assert "not found or unreadable" in result

    @pytest.mark.asyncio
    async def test_missing_requires_source(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["pubs"])
        result = await tool.execute(collection="pubs", action="missing", source="")
        assert "required" in result

    # --- patterns (multi-match) tests ---

    @pytest.mark.asyncio
    async def test_read_with_patterns_groups_matches(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["col"])
        tool._get_vector_manager = AsyncMock(return_value=MagicMock(
            prepare_metadata_store=AsyncMock(),
            get_source_chunks=AsyncMock(return_value=[{"content": "x"}]),
        ))
        tool._read_raw_text = AsyncMock(
            return_value="alpha here\nmiddle\nbeta there\n"
        )
        result = await tool.execute(
            collection="col", action="read", source="doc.md",
            patterns=["alpha", "beta"],
        )
        assert "'alpha' (1 matches)" in result
        assert "'beta' (1 matches)" in result
        assert "alpha here" in result
        assert "beta there" in result

    @pytest.mark.asyncio
    async def test_list_with_patterns_union_filter(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        with patch("app.stores.RawFileStore") as MockStore:
            instance = MockStore.return_value
            instance.list_by_collection = AsyncMock(return_value=[
                {"source": "test_collection/alpha-1.md", "filename": "alpha-1.md"},
                {"source": "test_collection/beta-2.md", "filename": "beta-2.md"},
                {"source": "test_collection/gamma-3.md", "filename": "gamma-3.md"},
            ])
            result = await tool.execute(
                collection="test_collection", action="list",
                patterns=["alpha", "gamma"],
            )
        assert "alpha-1.md" in result
        assert "gamma-3.md" in result
        assert "beta-2.md" not in result

    @pytest.mark.asyncio
    async def test_list_documents_no_engine(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        tool._rag_app.engine = None
        result = await tool.execute(collection="test_collection", action="list")
        assert "Database engine not available" in result

    def _many_files(self, n: int = 300) -> list:
        return [
            {
                "source": f"test_collection/doc{i:03d}.md",
                "filename": f"doc{i:03d}.md",
                "content_type": "text/markdown",
                "file_hash": f"hash{i:08d}",
                "updated_at": "2024-01-01T00:00:00",
            }
            for i in range(n)
        ]

    @pytest.mark.asyncio
    async def test_list_documents_paged_by_lines(self):
        """action=list must honor lines_start/lines_end for long listings."""
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        with patch("app.stores.RawFileStore") as MockStore:
            MockStore.return_value.list_by_collection = AsyncMock(
                return_value=self._many_files(300)
            )
            page1 = await tool.execute(
                collection="test_collection", action="list",
                lines_start=1, lines_end=30,
            )
            page2 = await tool.execute(
                collection="test_collection", action="list",
                lines_start=31, lines_end=60,
            )
        assert "doc000.md" in page1
        assert "doc029.md" not in page1
        assert "doc029.md" in page2 or "doc030.md" in page2
        assert "doc000.md" not in page2
        assert "showing lines" in page1 and "showing lines" in page2

    @pytest.mark.asyncio
    async def test_list_documents_paged_by_offset(self):
        """action=list must honor character offset — identical output was the bug."""
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        with patch("app.stores.RawFileStore") as MockStore:
            MockStore.return_value.list_by_collection = AsyncMock(
                return_value=self._many_files(300)
            )
            page1 = await tool.execute(collection="test_collection", action="list")
            page2 = await tool.execute(
                collection="test_collection", action="list", offset=15000
            )
        assert page1 != page2
        assert "showing chars" in page2 or "truncated" in page2

    @pytest.mark.asyncio
    async def test_list_documents_vector_fallback_paged(self):
        """The vector-store listing path must be pageable too."""
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        mock_manager = MagicMock()
        mock_manager.prepare_metadata_store = AsyncMock()
        mock_manager.list_embedding_sources = AsyncMock(return_value=[
            {"source": f"dir/doc{i:03d}.md", "chunks": 3} for i in range(300)
        ])
        tool._rag_app.vector_managers = {"test_collection": mock_manager}
        with patch("app.stores.RawFileStore") as MockStore:
            MockStore.return_value.list_by_collection = AsyncMock(return_value=[])
            result = await tool.execute(
                collection="test_collection", action="list",
                lines_start=200, lines_end=230,
            )
        assert "showing lines 200-230" in result

    @pytest.mark.asyncio
    async def test_search_missing_query(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        result = await tool.execute(collection="test_collection", action="search")
        assert "A query is required" in result

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        result = await tool.execute("test", collection="test_collection", action="foo")
        assert "Unknown action" in result

    def test_format_file_list(self):
        files = [
            {
                "filename": "a.md",
                "source": "col/a.md",
                "content_type": "text/markdown",
                "file_hash": "abc123",
                "updated_at": "2024-01-01",
            },
            {
                "filename": "b.pdf",
                "source": "col/b.pdf",
                "content_type": "application/pdf",
                "file_hash": "",
            },
        ]
        result = RagSearchTool._format_file_list(files)
        assert "a.md" in result
        assert "b.pdf" in result
        assert "text/markdown" in result
        assert "application/pdf" in result
        assert "abc123" in result
        assert "unknown" in result

    def test_format_file_list_empty(self):
        result = RagSearchTool._format_file_list([])
        assert "No documents found" in result

    # --- action=read tests ---

    def _mock_manager(
        self,
        chunks=None,
        sources=None,
    ):
        """Build a mock VectorStoreManager for read tests."""
        manager = MagicMock()
        manager.prepare_metadata_store = AsyncMock()
        manager.get_source_chunks = AsyncMock(return_value=chunks or [])
        manager.list_embedding_sources = AsyncMock(return_value=sources or [])
        return manager

    def _chunks(self, *texts, start_index=0):
        """Build chunk dicts as returned by VectorStoreManager.get_source_chunks."""
        return [
            {
                "document": t,
                "metadata": {"chunk_index": start_index + i, "source": "col/doc.md"},
                "chunk_index": str(start_index + i),
            }
            for i, t in enumerate(texts)
        ]

    @pytest.mark.asyncio
    async def test_read_requires_source(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        result = await tool.execute(collection="test_collection", action="read")
        assert "source is required" in result
        assert "action='list'" in result

    @pytest.mark.asyncio
    async def test_read_collapses_duplicate_rel_candidates(self):
        """Two physical cache copies of one logical file resolve, not 'Ambiguous'."""
        tool = self._make_tool()
        self._patch_collections(tool, ["CCL-Calls"])
        manager = self._mock_manager(
            chunks=[],
            sources=[
                {"source": "/r1/.webdav_cache/CCL-Calls/f.pdf", "chunks": 1},
                {"source": "/r2/.webdav_cache/CCL-Calls/f.pdf", "chunks": 1},
            ],
        )
        manager.get_source_chunks = AsyncMock(
            side_effect=[[], self._chunks("resolved body")]
        )
        tool._rag_app.vector_managers = {"CCL-Calls": manager}

        result = await tool.execute(
            collection="CCL-Calls",
            action="read",
            source="f.pdf",
        )
        assert "Ambiguous source" not in result
        assert "resolved body" in result
        # Deterministic: the sorted-first raw path wins.
        manager.get_source_chunks.assert_awaited_with("/r1/.webdav_cache/CCL-Calls/f.pdf")

    @pytest.mark.asyncio
    async def test_read_ambiguous_shows_rel_raw_pairs(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["webdav_col"])
        manager = self._mock_manager(
            chunks=[],
            sources=[
                {"source": "/dav/u/a/Report.pdf", "chunks": 1},
                {"source": "/dav/u/b/Report.pdf", "chunks": 1},
            ],
        )
        tool._rag_app.vector_managers = {"webdav_col": manager}

        result = await tool.execute(
            collection="webdav_col",
            action="read",
            source="Report.pdf",
        )
        assert "Ambiguous source" in result
        assert "→" in result
        assert "/dav/u/a/Report.pdf" in result

    @pytest.mark.asyncio
    async def test_list_shows_path_hint_for_rel_duplicates(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["CCL-Calls"])
        manager = self._mock_manager(
            chunks=[],
            sources=[
                {"source": "/r1/.webdav_cache/CCL-Calls/f.pdf", "chunks": 1},
                {"source": "/r2/.webdav_cache/CCL-Calls/f.pdf", "chunks": 1},
                {"source": "/r1/.webdav_cache/CCL-Calls/unique.pdf", "chunks": 1},
            ],
        )
        tool._rag_app.vector_managers = {"CCL-Calls": manager}

        result = await tool.execute(
            collection="CCL-Calls",
            action="list",
        )
        assert "path: /r1/.webdav_cache/CCL-Calls/f.pdf" in result
        assert "path: /r2/.webdav_cache/CCL-Calls/f.pdf" in result
        assert "path: /r1/.webdav_cache/CCL-Calls/unique.pdf" not in result

    @pytest.mark.asyncio
    async def test_read_collection_not_allowed(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["allowed_collection"])
        result = await tool.execute(
            collection="other", action="read", source="doc.md"
        )
        assert "is not allowed" in result

    @pytest.mark.asyncio
    async def test_read_no_index_manager(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        tool._rag_app._index_manager = None
        result = await tool.execute(
            collection="test_collection", action="read", source="doc.md"
        )
        assert "No documents found" in result

    @pytest.mark.asyncio
    async def test_read_exact_source(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        chunks = self._chunks("Hello world, this is chunk one.")
        manager = self._mock_manager(chunks=chunks)
        tool._rag_app.vector_managers = {"test_collection": manager}

        result = await tool.execute(
            collection="test_collection",
            action="read",
            source="col/doc.md",
        )
        assert "Source: col/doc.md" in result
        assert "chunks: 1" in result
        assert "Hello world, this is chunk one." in result
        manager.get_source_chunks.assert_called_with("col/doc.md")

    @pytest.mark.asyncio
    async def test_read_overlap_dedup(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        # chunks share a 20-char boundary overlap (>= _MIN_OVERLAP)
        c1 = "A" * 20 + "T" * 20
        c2 = "T" * 20 + "G" * 20
        c3 = "G" * 20 + "C" * 10
        chunks = self._chunks(c1, c2, c3)
        manager = self._mock_manager(chunks=chunks)
        tool._rag_app.vector_managers = {"test_collection": manager}

        result = await tool.execute(
            collection="test_collection",
            action="read",
            source="col/doc.md",
        )
        # overlap of T*20 and G*20 must be removed
        assert "A" * 20 + "T" * 20 + "G" * 20 + "C" * 10 in result
        assert "70 chars" in result
        assert "type: reconstructed" in result

    @pytest.mark.asyncio
    async def test_read_small_overlap_kept_verbatim(self):
        """Sub-threshold boundary matches must not drop chars (no silent loss)."""
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        # chunks share only a 10-char boundary string (< _MIN_OVERLAP=16)
        c1 = "AAAAAAAAAATTTTTTTTTT"
        c2 = "TTTTTTTTTTGGGGGGGGGG"
        c3 = "GGGGGGGGGGCCCC"
        chunks = self._chunks(c1, c2, c3)
        manager = self._mock_manager(chunks=chunks)
        tool._rag_app.vector_managers = {"test_collection": manager}

        result = await tool.execute(
            collection="test_collection",
            action="read",
            source="col/doc.md",
        )
        # nothing may be dropped: verbatim concatenation of all chunks
        assert c1 + c2 + c3 in result
        assert "54 chars" in result

    @pytest.mark.asyncio
    async def test_read_no_overlap_joins_verbatim(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        chunks = self._chunks("First part.", "Completely different.")
        manager = self._mock_manager(chunks=chunks)
        tool._rag_app.vector_managers = {"test_collection": manager}

        result = await tool.execute(
            collection="test_collection",
            action="read",
            source="col/doc.md",
        )
        # no overlap -> verbatim join (never invent whitespace)
        assert "First part.Completely different." in result

    @pytest.mark.asyncio
    async def test_read_chunks_sorted_by_index(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        # supply chunks out of order; reassembly must sort by chunk_index
        chunks = [
            {"document": "second", "metadata": {"chunk_index": 1}, "chunk_index": "1"},
            {"document": "first", "metadata": {"chunk_index": 0}, "chunk_index": "0"},
        ]
        manager = self._mock_manager(chunks=chunks)
        tool._rag_app.vector_managers = {"test_collection": manager}

        result = await tool.execute(
            collection="test_collection",
            action="read",
            source="col/doc.md",
        )
        assert "firstsecond" in result

    @pytest.mark.asyncio
    async def test_read_resolves_by_filename(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["webdav_col"])
        manager = self._mock_manager(
            chunks=[],
            sources=[
                {"source": "/remote.php/dav/files/u/dir/Report.pdf", "chunks": 9},
                {"source": "/remote.php/dav/files/u/dir/Other.pdf", "chunks": 3},
            ],
        )
        # After basename resolution the tool must re-fetch chunks for the resolved href
        resolved_chunks = self._chunks("PDF body text")
        manager.get_source_chunks = AsyncMock(
            side_effect=[[], resolved_chunks]
        )
        tool._rag_app.vector_managers = {"webdav_col": manager}

        result = await tool.execute(
            collection="webdav_col",
            action="read",
            source="Report.pdf",
        )
        assert "Source: /remote.php/dav/files/u/dir/Report.pdf" in result
        assert "PDF body text" in result

    @pytest.mark.asyncio
    async def test_read_resolves_url_encoded_basename(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["webdav_col"])
        manager = self._mock_manager(
            chunks=[],
            sources=[
                {"source": "/dav/files/u/My%20Report.pdf", "chunks": 2},
            ],
        )
        manager.get_source_chunks = AsyncMock(
            side_effect=[[], self._chunks("decoded content")]
        )
        tool._rag_app.vector_managers = {"webdav_col": manager}

        result = await tool.execute(
            collection="webdav_col",
            action="read",
            source="My Report.pdf",
        )
        assert "Source: /dav/files/u/My%20Report.pdf" in result
        assert "decoded content" in result

    @pytest.mark.asyncio
    async def test_read_ambiguous_source(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["webdav_col"])
        manager = self._mock_manager(
            chunks=[],
            sources=[
                {"source": "/dav/u/a/Report.pdf", "chunks": 1},
                {"source": "/dav/u/b/Report.pdf", "chunks": 1},
            ],
        )
        tool._rag_app.vector_managers = {"webdav_col": manager}

        result = await tool.execute(
            collection="webdav_col",
            action="read",
            source="Report.pdf",
        )
        assert "Ambiguous source" in result
        assert "/dav/u/a/Report.pdf" in result
        assert "/dav/u/b/Report.pdf" in result

    @pytest.mark.asyncio
    async def test_read_source_not_found(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["webdav_col"])
        manager = self._mock_manager(
            chunks=[],
            sources=[{"source": "/dav/u/a.pdf", "chunks": 1}],
        )
        tool._rag_app.vector_managers = {"webdav_col": manager}

        result = await tool.execute(
            collection="webdav_col",
            action="read",
            source="missing.pdf",
        )
        assert "not found" in result
        assert "action='list'" in result

    @pytest.mark.asyncio
    async def test_read_pagination_limit(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        chunks = self._chunks("A" * 100)
        manager = self._mock_manager(chunks=chunks)
        tool._rag_app.vector_managers = {"test_collection": manager}

        result = await tool.execute(
            collection="test_collection",
            action="read",
            source="col/doc.md",
            limit=10,
        )
        # 10 chars of body (numbered) + truncation footer with continue-hint
        assert "1│ AAAAAAAAAA" in result
        assert "truncated" in result
        assert "continue with" in result

    @pytest.mark.asyncio
    async def test_read_pagination_offset(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        chunks = self._chunks("0123456789abcdef")
        manager = self._mock_manager(chunks=chunks)
        tool._rag_app.vector_managers = {"test_collection": manager}

        result = await tool.execute(
            collection="test_collection",
            action="read",
            source="col/doc.md",
            offset=10,
        )
        assert "abcdef" in result
        assert "0123456789" not in result.split("Source:")[-1]

    @pytest.mark.asyncio
    async def test_read_pagination_lines(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        chunks = self._chunks("line1\nline2\nline3")
        manager = self._mock_manager(chunks=chunks)
        tool._rag_app.vector_managers = {"test_collection": manager}

        result = await tool.execute(
            collection="test_collection",
            action="read",
            source="col/doc.md",
            lines_start=2,
            lines_end=2,
        )
        assert "line2" in result
        body = result.split("\n\n", 1)[-1]
        assert "line1" not in body
        assert "line3" not in body

    @pytest.mark.asyncio
    async def test_read_empty_document(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        manager = self._mock_manager(
            chunks=[{"document": "", "metadata": {"chunk_index": 0}, "chunk_index": "0"}],
        )
        tool._rag_app.vector_managers = {"test_collection": manager}

        result = await tool.execute(
            collection="test_collection",
            action="read",
            source="col/doc.md",
        )
        assert "no readable text" in result

    @pytest.mark.asyncio
    async def test_read_lines_out_of_range_returns_message(self):
        """Out-of-range line reads must return an explanation, not empty output."""
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        chunks = self._chunks("line1\nline2\nline3")
        manager = self._mock_manager(chunks=chunks)
        tool._rag_app.vector_managers = {"test_collection": manager}

        result = await tool.execute(
            collection="test_collection",
            action="read",
            source="col/doc.md",
            lines_start=100,
            lines_end=200,
        )
        assert "no content" in result
        assert "100-200" in result
        assert "3 lines" in result

    @pytest.mark.asyncio
    async def test_read_offset_beyond_end_returns_message(self):
        """An offset past the document end must return an explanation, not empty output."""
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        chunks = self._chunks("0123456789abcdef")
        manager = self._mock_manager(chunks=chunks)
        tool._rag_app.vector_managers = {"test_collection": manager}

        result = await tool.execute(
            collection="test_collection",
            action="read",
            source="col/doc.md",
            offset=9999,
        )
        assert "no content" in result
        assert "16 chars" in result

    @pytest.mark.asyncio
    async def test_read_excerpt_shows_position_footer(self):
        """Partial reads must state which range was returned so callers can navigate."""
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        chunks = self._chunks("line1\nline2\nline3\nline4\nline5")
        manager = self._mock_manager(chunks=chunks)
        tool._rag_app.vector_managers = {"test_collection": manager}

        result = await tool.execute(
            collection="test_collection",
            action="read",
            source="col/doc.md",
            lines_start=2,
            lines_end=3,
        )
        assert "line2" in result
        assert "line3" in result
        assert "showing lines 2-3 of 5" in result

        result = await tool.execute(
            collection="test_collection",
            action="read",
            source="col/doc.md",
            offset=6,
            limit=6,
        )
        assert "chars 6-11 of 29" in result
        assert "continue with offset=12" in result

    # --- top_k search_k passthrough ---

    @pytest.mark.asyncio
    async def test_search_passes_search_k_to_builder(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"], max_top_k=10)
        fake_retriever = MagicMock()
        fake_retriever.ainvoke = AsyncMock(return_value=[])
        builder = tool._rag_app._retriever_builder
        builder.get_retriever_for_collections = AsyncMock(return_value=fake_retriever)

        await tool.execute(
            "query", collection="test_collection", top_k=8
        )
        builder.get_retriever_for_collections.assert_called_once()
        _, kwargs = builder.get_retriever_for_collections.call_args
        assert kwargs.get("search_k") == 8

    @pytest.mark.asyncio
    async def test_search_search_k_clamped_by_max_top_k(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"], max_top_k=3)
        fake_retriever = MagicMock()
        fake_retriever.ainvoke = AsyncMock(return_value=[])
        builder = tool._rag_app._retriever_builder
        builder.get_retriever_for_collections = AsyncMock(return_value=fake_retriever)

        await tool.execute(
            "query", collection="test_collection", top_k=10
        )
        _, kwargs = builder.get_retriever_for_collections.call_args
        assert kwargs.get("search_k") == 3

    def test_format_vector_sources_shows_basename(self):
        sources = [
            {"source": "/remote.php/dav/files/u/dir/My%20Report.pdf", "chunks": 4},
        ]
        result = RagSearchTool._format_vector_sources(sources)
        assert "My Report.pdf" in result
        assert "source: /remote.php/dav/files/u/dir/My Report.pdf" in result
        assert "(4 chunks)" in result
        assert "/remote.php/dav/files/u/dir/" in result

    def test_format_vector_sources_groups_by_directory(self):
        sources = [
            {"source": "Ordner/Sub/Doc.pdf", "chunks": 65},
            {"source": "Ordner/Sub/Other.pdf", "chunks": 12},
            {"source": "readme.md", "chunks": 3},
        ]
        result = RagSearchTool._format_vector_sources(sources, "sciebo")
        assert "Documents in collection 'sciebo' (3 files, 80 chunks):" in result
        assert "Ordner/Sub/" in result
        assert "Root:" in result
        assert "- Doc.pdf  (65 chunks)  |  source: Ordner/Sub/Doc.pdf" in result
        assert "- readme.md  (3 chunks)  |  source: readme.md" in result
        # root group sorts first (empty parent key), then dirs alphabetically
        assert result.index("Root:") < result.index("Ordner/Sub/")

    def test_format_vector_sources_duplicate_basenames_stay_distinct(self):
        sources = [
            {"source": "a/report.pdf", "chunks": 1},
            {"source": "b/report.pdf", "chunks": 2},
        ]
        result = RagSearchTool._format_vector_sources(sources)
        assert "a/" in result and "b/" in result
        assert "source: a/report.pdf" in result
        assert "source: b/report.pdf" in result

    def test_format_vector_sources_strips_webdav_cache_prefix(self):
        """Absolute .webdav_cache paths must display relative to the collection."""
        sources = [
            {"source": "/app/data/.webdav_cache/CCL-Publikationen/sub/deep/paper.pdf", "chunks": 5},
            {"source": "/Users/me/agentix/data/.webdav_cache/CCL-Publikationen/root.pdf", "chunks": 1},
            {"source": "/app/data/.webdav_cache/Other-Col/hidden.pdf", "chunks": 2},
        ]
        result = RagSearchTool._format_vector_sources(sources, "CCL-Publikationen")
        assert "source: sub/deep/paper.pdf" in result
        assert "source: root.pdf" in result
        assert ".webdav_cache" not in result.replace("Other-Col/hidden.pdf", "")
        # foreign collection dirs keep their name but still lose the cache prefix
        assert "Other-Col/hidden.pdf" in result
        assert "sub/deep/" in result
        assert "Root:" in result

    def test_format_vector_sources_folder_filter(self):
        sources = [
            {"source": "papers/2024/a.pdf", "chunks": 1},
            {"source": "papers/2025/b.pdf", "chunks": 1},
            {"source": "top.pdf", "chunks": 1},
        ]
        result = RagSearchTool._format_vector_sources(
            sources, "col", folder="papers/2024"
        )
        assert "under 'papers/2024/'" in result
        assert "source: papers/2024/a.pdf" in result
        assert "b.pdf" not in result
        assert "top.pdf" not in result

    def test_format_vector_sources_folder_no_match(self):
        sources = [{"source": "papers/a.pdf", "chunks": 1}]
        result = RagSearchTool._format_vector_sources(sources, "col", folder="missing")
        assert "No documents found under 'missing/'" in result

    def test_format_vector_sources_depth_collapses_folders(self):
        sources = [
            {"source": "a/deep/x.pdf", "chunks": 1},
            {"source": "a/deep/y.pdf", "chunks": 2},
            {"source": "a/top.pdf", "chunks": 4},
            {"source": "root.pdf", "chunks": 3},
        ]
        result = RagSearchTool._format_vector_sources(sources, "col", depth=1)
        # level-1 folder collapsed into a single summary line
        assert "a/  (3 files, 7 chunks)" in result
        assert "top.pdf  (4 chunks)" not in result
        assert "source: root.pdf" in result

        result2 = RagSearchTool._format_vector_sources(sources, "col", depth=2)
        assert "a/" in result2
        assert "source: a/top.pdf" in result2
        assert "a/deep/  (2 files, 3 chunks)" in result2

    def test_format_file_list_strips_webdav_cache_prefix(self):
        files = [
            {
                "filename": "paper.pdf",
                "source": "/srv/data/.webdav_cache/Col/sub/paper.pdf",
                "content_type": "application/pdf",
                "file_hash": "h12345",
                "updated_at": "2024-01-01",
            }
        ]
        result = RagSearchTool._format_file_list(files, "Col")
        assert ".webdav_cache" not in result
        assert "source: sub/paper.pdf" in result
        assert "sub/" in result

    @pytest.mark.asyncio
    async def test_read_resolves_relative_display_path(self):
        """A relative path shown by list must resolve to the absolute stored source."""
        tool = self._make_tool()
        self._patch_collections(tool, ["CCL"])
        stored = "/app/data/.webdav_cache/CCL/sub/doc.pdf"
        manager = self._mock_manager(
            chunks=self._chunks("content here"),
            sources=[{"source": stored, "chunks": 1}],
        )
        tool._rag_app.vector_managers = {"CCL": manager}
        # first exact-lookup misses, then fuzzy match on the relative path
        manager.get_source_chunks = AsyncMock(
            side_effect=lambda s: self._chunks("content here") if s == stored else []
        )
        result = await tool.execute(
            collection="CCL", action="read", source="sub/doc.pdf"
        )
        assert "content here" in result

    # --- raw fast path / pattern / hidden ---

    @pytest.mark.asyncio
    async def test_read_prefers_raw_file(self):
        """Text files are served byte-exact from RawFileStore, not reconstructed."""
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        manager = self._mock_manager(chunks=self._chunks("chunk version"))
        tool._rag_app.vector_managers = {"test_collection": manager}
        with patch("app.stores.RawFileStore") as MockStore:
            instance = MockStore.return_value
            instance.get = AsyncMock(
                return_value={"file_data": b"# Raw exact content\nline2"}
            )
            result = await tool.execute(
                collection="test_collection", action="read", source="col/doc.md"
            )
        assert "type: raw" in result
        assert "1│ # Raw exact content" in result
        assert "chunk version" not in result

    @pytest.mark.asyncio
    async def test_read_raw_only_unindexed(self):
        """Files stored but not yet indexed still resolve via RawFileStore."""
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        manager = self._mock_manager(chunks=[], sources=[])
        tool._rag_app.vector_managers = {"test_collection": manager}
        with patch("app.stores.RawFileStore") as MockStore:
            MockStore.return_value.get = AsyncMock(
                return_value={"file_data": b"fresh unindexed content"}
            )
            result = await tool.execute(
                collection="test_collection", action="read", source="col/doc.md"
            )
        assert "type: raw" in result
        assert "fresh unindexed content" in result

    @pytest.mark.asyncio
    async def test_read_binary_falls_back_to_reconstructed(self):
        """Binary raw rows cannot be decoded → chunk reassembly is used."""
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        manager = self._mock_manager(chunks=self._chunks("pdf text chunk"))
        tool._rag_app.vector_managers = {"test_collection": manager}
        with patch("app.stores.RawFileStore") as MockStore:
            MockStore.return_value.get = AsyncMock(
                return_value={"file_data": b"\x89PNG\r\n\x1a\n\xff\xfe binary"}
            )
            result = await tool.execute(
                collection="test_collection", action="read", source="col/doc.pdf"
            )
        assert "type: reconstructed" in result
        assert "pdf text chunk" in result

    @pytest.mark.asyncio
    async def test_read_pattern_locates_content(self):
        """pattern= on read returns numbered match windows with context."""
        lines = "\n".join(f"line {i}" for i in range(1, 31))
        chunks = self._chunks(lines + "\nCluster B overview here\nline 31")
        manager = self._mock_manager(chunks=chunks)
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        tool._rag_app.vector_managers = {"test_collection": manager}
        with patch("app.stores.RawFileStore") as MockStore:
            MockStore.return_value.get = AsyncMock(return_value=None)
            result = await tool.execute(
                collection="test_collection",
                action="read",
                source="col/doc.md",
                pattern="cluster b",
            )
        assert "31│ Cluster B overview here" in result
        assert "30│ line 30" in result  # context line
        assert "matches" in result

    @pytest.mark.asyncio
    async def test_list_pattern_filters_paths(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        with patch("app.stores.RawFileStore") as MockStore:
            MockStore.return_value.list_by_collection = AsyncMock(
                return_value=[
                    {"source": "test_collection/doc-a.md", "filename": "doc-a.md",
                     "content_type": "text/markdown", "file_hash": "h1", "updated_at": "x"},
                    {"source": "test_collection/sheet.pdf", "filename": "sheet.pdf",
                     "content_type": "application/pdf", "file_hash": "h2", "updated_at": "x"},
                    {"source": "test_collection/doc-b.md", "filename": "doc-b.md",
                     "content_type": "text/markdown", "file_hash": "h3", "updated_at": "x"},
                ]
            )
            result = await tool.execute(
                collection="test_collection", action="list", pattern="doc-"
            )
        assert "doc-a.md" in result
        assert "doc-b.md" in result
        assert "sheet.pdf" not in result
        assert "2 of 3 files" in result

    @pytest.mark.asyncio
    async def test_list_hides_dotfiles_by_default(self):
        tool = self._make_tool()
        self._patch_collections(tool, ["test_collection"])
        files = [
            {"source": "test_collection/.trash-999/old.md", "filename": "old.md",
             "content_type": "text/markdown", "file_hash": "h1", "updated_at": "x"},
            {"source": "test_collection/visible.md", "filename": "visible.md",
             "content_type": "text/markdown", "file_hash": "h2", "updated_at": "x"},
        ]
        with patch("app.stores.RawFileStore") as MockStore:
            MockStore.return_value.list_by_collection = AsyncMock(return_value=files)
            result = await tool.execute(collection="test_collection", action="list")
        assert "visible.md" in result
        assert ".trash" not in result
        with patch("app.stores.RawFileStore") as MockStore:
            MockStore.return_value.list_by_collection = AsyncMock(return_value=files)
            result = await tool.execute(
                collection="test_collection", action="list", include_hidden=True
            )
        assert ".trash-999" in result


class TestReadWindow:
    """Configurable read/list window limits (read_limit / limit_max)."""

    def _make_tool(self):
        rag_app = MagicMock()
        rag_app._tools_store = None
        rag_app._retriever_builder = MagicMock()
        rag_app.engine = MagicMock()
        return RagSearchTool(rag_app)

    def _patch_read(self, tool, text: str):
        tool._get_collection_names = AsyncMock(return_value=["test_collection"])
        tool._get_vector_manager = AsyncMock(return_value=MagicMock())
        tool._resolve_source = AsyncMock(return_value=("doc.md", None))
        tool._read_source_text = AsyncMock(return_value=(text, "reconstructed", 5))

    @pytest.mark.asyncio
    async def test_read_default_window_is_50k(self):
        tool = self._make_tool()
        big = "x" * 120_000
        self._patch_read(tool, big)
        result = await tool.execute(collection="test_collection", action="read", source="doc.md")
        assert "truncated" in result
        assert result.count("x") > 49_000
        assert result.count("x") < 60_000

    @pytest.mark.asyncio
    async def test_read_explicit_limit_honored_below_max(self):
        tool = self._make_tool()
        big = "x" * 120_000
        self._patch_read(tool, big)
        result = await tool.execute(
            collection="test_collection", action="read", source="doc.md", limit=20_000
        )
        assert "truncated" in result
        assert result.count("x") < 21_000

    @pytest.mark.asyncio
    async def test_read_limit_above_max_is_capped(self):
        tool = self._make_tool()
        big = "x" * 120_000
        self._patch_read(tool, big)
        result = await tool.execute(
            collection="test_collection", action="read", source="doc.md", limit=999_999
        )
        assert result.count("x") < 60_000

    @pytest.mark.asyncio
    async def test_read_limit_tool_config_override(self):
        tool = self._make_tool()
        big = "x" * 120_000
        self._patch_read(tool, big)
        result = await tool.execute(
            collection="test_collection", action="read", source="doc.md",
            context={"tool_config": {"read_limit": 8_000}},
        )
        assert "truncated" in result
        assert result.count("x") < 9_000

    @pytest.mark.asyncio
    async def test_schema_limit_maximum_dynamic(self):
        tool = self._make_tool()
        tool._collection_names = ["test_collection"]
        schema = tool.schema_with_config({"read_limit": 60_000})
        limit_prop = schema["function"]["parameters"]["properties"]["limit"]
        assert limit_prop["maximum"] == 60_000
        assert "60000" in limit_prop["description"]

    @pytest.mark.asyncio
    async def test_list_keeps_15k_default_allows_larger(self):
        tool = self._make_tool()
        tool._get_collection_names = AsyncMock(return_value=["test_collection"])
        files = [
            {"source": f"test_collection/dir/doc-{i:04d}.md", "filename": f"doc-{i}.md",
             "content_type": "text/markdown", "file_hash": f"h{i}", "updated_at": "x"}
            for i in range(400)
        ]
        with patch("app.stores.RawFileStore") as MockStore:
            MockStore.return_value.list_by_collection = AsyncMock(return_value=files)
            small = await tool.execute(collection="test_collection", action="list")
        assert "truncated" in small
        assert len(small) < 16_500
        with patch("app.stores.RawFileStore") as MockStore:
            MockStore.return_value.list_by_collection = AsyncMock(return_value=files)
            big = await tool.execute(
                collection="test_collection", action="list", limit=40_000
            )
        assert len(big) > len(small)


class TestRagSearchVectorstoreMetrics:
    """rag_search search/read must record vectorstore_metrics rows."""

    def _make_tool(self):
        rag_app = MagicMock()
        rag_app._tools_store = None
        rag_app._retriever_builder = MagicMock()
        rag_app.engine = MagicMock()
        _conn = rag_app.engine.begin.return_value.__aenter__.return_value
        _res = MagicMock()
        _res.mappings.return_value.first.return_value = None
        _res.mappings.return_value.all.return_value = []
        _conn.execute = AsyncMock(return_value=_res)
        rag_app.engine.connect.return_value = rag_app.engine.begin.return_value
        tool = RagSearchTool(rag_app)
        store = MagicMock()
        store.save = AsyncMock()
        tool._vs_store = store
        return tool, store

    def _patch_collections(self, tool, names=("test_collection",)):
        tool._collection_names = list(names)
        tool._get_collection_names = AsyncMock(return_value=list(names))

    def _ctx(self):
        return {"conversation_id": "conv-1", "assistant_name": "asst-1"}

    @pytest.mark.asyncio
    async def test_search_records_ok_row(self):
        tool, store = self._make_tool()
        self._patch_collections(tool)
        docs = [
            Document(page_content=f"chunk {i} text", metadata={"source": f"{i}.md"})
            for i in range(3)
        ]
        retriever = MagicMock()
        retriever.ainvoke = AsyncMock(return_value=docs)
        tool._rag_app._retriever_builder.get_retriever_for_collections = AsyncMock(
            return_value=retriever
        )
        result = await tool.execute(
            "q", collection="test_collection", context=self._ctx()
        )
        assert "chunk 0 text" in result
        store.save.assert_awaited_once()
        rec = store.save.await_args.args[0]
        assert rec.conversation_id == "conv-1"
        assert rec.model_name == "asst-1"
        assert rec.collection_names == "test_collection"
        assert rec.status == "ok"
        assert rec.chunks_retrieved == 3
        assert rec.retrieval_time_ms >= 0
        assert rec.error_type is None
        assert rec.rag_context_tokens > 0
        assert rec.started_at

    @pytest.mark.asyncio
    async def test_search_records_error_row(self):
        tool, store = self._make_tool()
        self._patch_collections(tool)
        retriever = MagicMock()
        retriever.ainvoke = AsyncMock(side_effect=ValueError("boom"))
        tool._rag_app._retriever_builder.get_retriever_for_collections = AsyncMock(
            return_value=retriever
        )
        result = await tool.execute(
            "q", collection="test_collection", context=self._ctx()
        )
        assert "Search failed" in result
        rec = store.save.await_args.args[0]
        assert rec.status == "error"
        assert rec.error_type == "ValueError"
        assert rec.chunks_retrieved == 0
        assert rec.rag_context_tokens is None

    @pytest.mark.asyncio
    async def test_search_records_empty_ok_row(self):
        tool, store = self._make_tool()
        self._patch_collections(tool)
        retriever = MagicMock()
        retriever.ainvoke = AsyncMock(return_value=[])
        tool._rag_app._retriever_builder.get_retriever_for_collections = AsyncMock(
            return_value=retriever
        )
        result = await tool.execute(
            "q", collection="test_collection", context=self._ctx()
        )
        assert "No results found" in result
        rec = store.save.await_args.args[0]
        assert rec.status == "ok"
        assert rec.chunks_retrieved == 0

    @pytest.mark.asyncio
    async def test_search_counts_truncated_docs(self):
        tool, store = self._make_tool()
        self._patch_collections(tool)
        tool._max_top_k = 2
        docs = [
            Document(page_content=f"c{i}", metadata={"source": f"{i}.md"})
            for i in range(5)
        ]
        retriever = MagicMock()
        retriever.ainvoke = AsyncMock(return_value=docs)
        tool._rag_app._retriever_builder.get_retriever_for_collections = AsyncMock(
            return_value=retriever
        )
        await tool.execute(
            "q", collection="test_collection", context=self._ctx()
        )
        rec = store.save.await_args.args[0]
        assert rec.chunks_retrieved == 2

    @pytest.mark.asyncio
    async def test_read_records_ok_row(self):
        tool, store = self._make_tool()
        self._patch_collections(tool)
        tool._get_vector_manager = AsyncMock(return_value=MagicMock())
        tool._resolve_source = AsyncMock(return_value=("doc.md", None))
        tool._read_source_text = AsyncMock(
            return_value=("full document text", "reconstructed", 5)
        )
        result = await tool.execute(
            collection="test_collection", action="read",
            source="doc.md", context=self._ctx(),
        )
        assert "full document text" in result
        store.save.assert_awaited_once()
        rec = store.save.await_args.args[0]
        assert rec.status == "ok"
        assert rec.chunks_retrieved == 5
        assert rec.rag_context_tokens > 0
        assert rec.retrieval_time_ms >= 0

    @pytest.mark.asyncio
    async def test_read_records_raw_chunk_as_one(self):
        tool, store = self._make_tool()
        self._patch_collections(tool)
        tool._get_vector_manager = AsyncMock(return_value=MagicMock())
        tool._resolve_source = AsyncMock(return_value=(None, None))
        tool._read_raw_text = AsyncMock(return_value="raw body")
        result = await tool.execute(
            collection="test_collection", action="read",
            source="file.md", context=self._ctx(),
        )
        assert "raw body" in result
        rec = store.save.await_args.args[0]
        assert rec.status == "ok"
        assert rec.chunks_retrieved == 1

    @pytest.mark.asyncio
    async def test_read_records_not_found_error(self):
        tool, store = self._make_tool()
        self._patch_collections(tool)
        tool._get_vector_manager = AsyncMock(return_value=MagicMock())
        tool._resolve_source = AsyncMock(return_value=(None, None))
        tool._read_raw_text = AsyncMock(return_value=None)
        result = await tool.execute(
            collection="test_collection", action="read",
            source="missing.md", context=self._ctx(),
        )
        assert "not found" in result
        rec = store.save.await_args.args[0]
        assert rec.status == "error"
        assert rec.error_type == "not_found"
        assert rec.chunks_retrieved == 0
        assert rec.rag_context_tokens is None

    @pytest.mark.asyncio
    async def test_read_records_empty_content_error(self):
        tool, store = self._make_tool()
        self._patch_collections(tool)
        tool._get_vector_manager = AsyncMock(return_value=MagicMock())
        tool._resolve_source = AsyncMock(return_value=("doc.md", None))
        tool._read_source_text = AsyncMock(
            return_value=("", "reconstructed", 0)
        )
        result = await tool.execute(
            collection="test_collection", action="read",
            source="doc.md", context=self._ctx(),
        )
        assert "no indexed content" in result
        rec = store.save.await_args.args[0]
        assert rec.status == "error"
        assert rec.error_type == "empty_content"

    @pytest.mark.asyncio
    async def test_list_records_nothing(self):
        tool, store = self._make_tool()
        self._patch_collections(tool)
        with patch("app.stores.RawFileStore") as MockStore:
            MockStore.return_value.list_by_collection = AsyncMock(return_value=[])
            await tool.execute(
                collection="test_collection", action="list",
                context=self._ctx(),
            )
        store.save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_records_nothing(self):
        tool, store = self._make_tool()
        self._patch_collections(tool)
        tool._execute_missing = AsyncMock(return_value="coverage report")
        await tool.execute(
            collection="test_collection", action="missing",
            source="doc.md", document_collection="docs",
            context=self._ctx(),
        )
        store.save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_record_without_conversation_id(self):
        tool, store = self._make_tool()
        self._patch_collections(tool)
        retriever = MagicMock()
        retriever.ainvoke = AsyncMock(return_value=[Document(page_content="x")])
        tool._rag_app._retriever_builder.get_retriever_for_collections = AsyncMock(
            return_value=retriever
        )
        await tool.execute("q", collection="test_collection", context={})
        await tool.execute("q", collection="test_collection", context=None)
        store.save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_save_failure_does_not_break_tool(self):
        tool, store = self._make_tool()
        self._patch_collections(tool)
        store.save = AsyncMock(side_effect=RuntimeError("db down"))
        retriever = MagicMock()
        retriever.ainvoke = AsyncMock(return_value=[Document(page_content="x")])
        tool._rag_app._retriever_builder.get_retriever_for_collections = AsyncMock(
            return_value=retriever
        )
        result = await tool.execute(
            "q", collection="test_collection", context=self._ctx()
        )
        assert "x" in result

    @pytest.mark.asyncio
    async def test_no_store_no_crash(self):
        tool, _ = self._make_tool()
        tool._vs_store = None
        tool._rag_app.engine = None  # lazy store can't be built
        self._patch_collections(tool)
        retriever = MagicMock()
        retriever.ainvoke = AsyncMock(return_value=[Document(page_content="x")])
        tool._rag_app._retriever_builder.get_retriever_for_collections = AsyncMock(
            return_value=retriever
        )
        result = await tool.execute(
            "q", collection="test_collection", context=self._ctx()
        )
        assert "x" in result


class TestReadDocumentText:
    """read_document_text — shared collection-read chain for other tools."""

    def _make_tool(self):
        rag_app = MagicMock()
        rag_app._tools_store = None
        rag_app._retriever_builder = MagicMock()
        rag_app.engine = MagicMock()
        _conn = rag_app.engine.begin.return_value.__aenter__.return_value
        _res = MagicMock()
        _res.mappings.return_value.first.return_value = None
        _res.mappings.return_value.all.return_value = []
        _conn.execute = AsyncMock(return_value=_res)
        rag_app.engine.connect.return_value = rag_app.engine.begin.return_value
        tool = RagSearchTool(rag_app)
        store = MagicMock()
        store.save = AsyncMock()
        tool._vs_store = store
        return tool, store

    def _ctx(self):
        return {"conversation_id": "conv-1", "assistant_name": "asst-1"}

    @pytest.mark.asyncio
    async def test_raw_hit_without_manager(self):
        tool, store = self._make_tool()
        tool._read_raw_text = AsyncMock(return_value="raw text body")
        tool._get_vector_manager = AsyncMock(side_effect=AssertionError("must not be called"))
        text, err = await tool.read_document_text("col", "doc.md", context=self._ctx())
        assert err is None
        assert text == "raw text body"
        rec = store.save.await_args.args[0]
        assert rec.status == "ok"
        assert rec.chunks_retrieved == 1
        assert rec.rag_context_tokens > 0

    @pytest.mark.asyncio
    async def test_resolve_and_reconstruct(self):
        tool, store = self._make_tool()
        tool._read_raw_text = AsyncMock(return_value=None)
        tool._get_vector_manager = AsyncMock(return_value=MagicMock())
        tool._resolve_source = AsyncMock(return_value=("col/doc.md", None))
        tool._read_source_text = AsyncMock(
            return_value=("reassembled text", "reconstructed", 4)
        )
        text, err = await tool.read_document_text("col", "doc.md", context=self._ctx())
        assert err is None
        assert text == "reassembled text"
        rec = store.save.await_args.args[0]
        assert rec.status == "ok"
        assert rec.chunks_retrieved == 4
        assert rec.collection_names == "col"

    @pytest.mark.asyncio
    async def test_not_found(self):
        tool, store = self._make_tool()
        tool._read_raw_text = AsyncMock(return_value=None)
        tool._get_vector_manager = AsyncMock(return_value=MagicMock())
        tool._resolve_source = AsyncMock(return_value=(None, None))
        text, err = await tool.read_document_text("col", "missing.md", context=self._ctx())
        assert text is None
        assert "not found" in err
        rec = store.save.await_args.args[0]
        assert rec.status == "error"
        assert rec.error_type == "not_found"

    @pytest.mark.asyncio
    async def test_resolve_error_recorded(self):
        tool, store = self._make_tool()
        tool._read_raw_text = AsyncMock(return_value=None)
        tool._get_vector_manager = AsyncMock(return_value=MagicMock())
        tool._resolve_source = AsyncMock(return_value=(None, "Ambiguous source 'x'. Matching: a, b"))
        text, err = await tool.read_document_text("col", "x", context=self._ctx())
        assert text is None
        assert "Ambiguous" in err
        rec = store.save.await_args.args[0]
        assert rec.error_type == "resolve_error"

    @pytest.mark.asyncio
    async def test_empty_content_recorded(self):
        tool, store = self._make_tool()
        tool._read_raw_text = AsyncMock(return_value=None)
        tool._get_vector_manager = AsyncMock(return_value=MagicMock())
        tool._resolve_source = AsyncMock(return_value=("col/doc.md", None))
        tool._read_source_text = AsyncMock(return_value=("", "reconstructed", 0))
        text, err = await tool.read_document_text("col", "doc.md", context=self._ctx())
        assert text is None
        assert "no readable text" in err
        rec = store.save.await_args.args[0]
        assert rec.error_type == "empty_content"

    @pytest.mark.asyncio
    async def test_no_manager_records_error(self):
        tool, store = self._make_tool()
        tool._read_raw_text = AsyncMock(return_value=None)
        tool._get_vector_manager = AsyncMock(return_value=None)
        text, err = await tool.read_document_text("col", "doc.md", context=self._ctx())
        assert text is None
        assert "No documents" in err
        rec = store.save.await_args.args[0]
        assert rec.error_type == "not_found"

    @pytest.mark.asyncio
    async def test_empty_source_no_record(self):
        tool, store = self._make_tool()
        text, err = await tool.read_document_text("col", "  ", context=self._ctx())
        assert text is None
        assert "required" in err
        store.save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_context_no_record(self):
        tool, store = self._make_tool()
        tool._read_raw_text = AsyncMock(return_value="text")
        text, err = await tool.read_document_text("col", "doc.md", context=None)
        assert err is None and text == "text"
        store.save.assert_not_awaited()
