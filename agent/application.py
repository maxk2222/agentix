"""High-level orchestration for the cybersecurity RAG assistant.

The module coordinates document ingestion, embedding generation, PGVector lifecycle
management, and the LangChain chains that back both streaming and non-streaming
responses.  FastAPI routes interact with this layer exclusively, keeping API code thin
while allowing the RAG pipeline to evolve independently.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Sequence, Tuple

from langchain.schema import Document
from langchain_core.messages import BaseMessage, HumanMessage
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

from app.chat_utils import extract_answer_text, extract_reasoning_text, extract_usage_metadata, load_instructions, load_instructions_from_yaml_text, render_system_instructions
from app.context_summarizer import summarize_context_if_needed
from app.history_budget import resolve_rolling_history_settings
from app.model_context import get_provider_context_windows
from app.logging_utils import colorize_process_time
from .chains import ChainBuilder
from .config import AppConfig
from .embeddings import CustomOpenAICompatibleEmbeddings, SlowSystemEmbeddingWrapper
from .indexing import IndexManager
from .loader import KnowledgeLoader
from .processing import clean_documents
from .exceptions import ConfigurationError, UpstreamLLMError
from .recovery import RecoveryManager
from .retriever_builder import RetrieverBuilder
from .retrieval_utils import _build_retrieval_query, _clean_noise
from .retrievers import MetricsRecordingRetriever, RecencyWeightedRetriever
from .stream_watchdog import DEFAULT_STREAM_IDLE_SECONDS
from app.stores import AgentMemoryStore
from .tools import ToolRegistry
from .tools.builtin import CollectionManagerTool, RagSearchTool
from .tools.builtin_tools.assistant_caller import AssistantCallerTool
from .tools.builtin_tools.spawn_subagent import SpawnSubagentTool
from .tools.builtin_tools.wait_subagent import WaitSubagentTool
from .tools.builtin_tools.subagent_status import SubagentStatusTool
from .tools.builtin_tools.nudge_subagent import NudgeSubagentTool
from .tools.builtin_tools.todo_list import TODO_TOOL_NAME
from .subagents import SubagentManager, SubagentState, make_subagent_state
from .tools.runner import AgentRunner
from .utils.pg_helpers import _to_async_connection_string
from .vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)
REMOTE_COLLECTION_NAME = "external_urls"
# Subagents inherit the parent's full tool set (incl. web_search, web_scraper
# and all mcp__* tools) minus this default exclusion list. ``agent_memory``
# would write under the child conversation id, where nothing ever reads it
# back — excluding it avoids pointless orphaned memories. Per-assistant
# override via subagents.exclude_tools (replaces the default entirely).
DEFAULT_SUBAGENT_EXCLUDE_TOOLS = {"agent_memory"}


def _resolve_allowed_tools(
    tools_dict: Dict[str, Any],
    subagents_cfg: Dict[str, Any],
    subagent_state: Optional["SubagentState"],
) -> List[str]:
    """Resolve the tool names for this run; subagents inherit the parent's
    configured set minus ``subagents.exclude_tools`` (or the defaults)."""
    allowed = list(tools_dict.keys())
    if subagent_state is not None and not subagent_state.is_root:
        # ``exclude_tools`` present (even empty) replaces the defaults —
        # absent means "use the defaults".
        excluded = (
            set(subagents_cfg["exclude_tools"] or [])
            if "exclude_tools" in subagents_cfg
            else set(DEFAULT_SUBAGENT_EXCLUDE_TOOLS)
        )
        allowed = [name for name in allowed if name not in excluded]
    return allowed


def _subagent_excluded_tools(
    subagents_cfg: Dict[str, Any],
    subagent_state: Optional["SubagentState"],
) -> set:
    """The effective ``exclude_tools`` set for non-root subagent runs (empty
    for the root run). Auto-appended tools must respect it too — otherwise an
    admin could not keep e.g. ``agent_workspace`` or ``spawn_subagent`` out of
    subagent runs."""
    if subagent_state is None or subagent_state.is_root:
        return set()
    return (
        set(subagents_cfg["exclude_tools"] or [])
        if "exclude_tools" in subagents_cfg
        else set(DEFAULT_SUBAGENT_EXCLUDE_TOOLS)
    )


def _format_rag_context(docs: Sequence[Document]) -> str:
    """Format retrieved documents into a string with source citations for the system prompt."""
    parts: List[str] = []
    for doc in docs:
        meta = doc.metadata or {}
        source = meta.get("source", "")
        page = meta.get("page") or meta.get("page_number", "")
        chunk_idx = meta.get("chunk_index", "")
        content = getattr(doc, "page_content", "") or ""

        header_parts: List[str] = []
        if source:
            header_parts.append(f"Source: {source}")
        if page:
            header_parts.append(f"Page: {page}")
        if chunk_idx != "":
            header_parts.append(f"Chunk: {chunk_idx}")

        if header_parts:
            parts.append(f"{', '.join(header_parts)}\n{content}")
        else:
            parts.append(content)

    return "\n\n".join(parts)


class RAGApplication:
    """Façade combining loaders, embeddings, PGVector, and conversational chains.

    One instance is created during FastAPI startup and shared across all requests.  It
    encapsulates slow-startup concerns (document loading, vector rebuilding) and exposes
    lightweight async helpers for conversational and streaming interactions.
    """

    def __init__(
        self,
        config: AppConfig,
        loader: Optional[KnowledgeLoader] = None,
        embeddings_factory: Optional[
            Callable[[AppConfig], CustomOpenAICompatibleEmbeddings]
        ] = None,
        provider_store: Any = None,
        engine: Optional[AsyncEngine] = None,
    ) -> None:
        self.config = config
        self._provider_store = provider_store
        self._engine = engine
        if config.user_agent:
            os.environ.setdefault("USER_AGENT", config.user_agent)
        self.vector_managers: Dict[str, VectorStoreManager] = {}
        self.collection_paths: Dict[str, Optional[Path]] = {}
        self.available_collections: set[str] = set()

        self._instructions_lock = asyncio.Lock()
        self._instructions_mtime = 0.0
        self._instructions_store: Optional[Any] = None
        self._instructions_db_version_id = 0
        self._provider_model_cache: Dict[str, str] = {}
        self._provider_cache_lock = asyncio.Lock()
        self._provider_refresh_task: Optional[asyncio.Task] = None
        self._provider_refresh_requested = False
        self.config.bootstrap_default_runtime_data()
        try:
            initial_instructions = self._load_instructions(self.config.instructions_file)
        except Exception as exc:
            logger.warning("Instructions file not readable at startup: %s. Will rely on DB.", exc)
            initial_instructions = {}
        self.embeddings: Optional[CustomOpenAICompatibleEmbeddings] = None
        self._ready = False
        self._chain_builder: Optional[ChainBuilder] = None
        self._recovery_manager: Optional[RecoveryManager] = None
        self._tool_registry = ToolRegistry()
        self._tool_registry.register_builtin(
            "collection_manager",
            CollectionManagerTool(self),
        )
        self._tool_registry.register_builtin(
            "rag_search",
            RagSearchTool(self),
        )
        from agent.tools.builtin_tools.long_document_summarizer import LongDocumentSummarizerTool
        self._tool_registry.register_builtin(
            "long_document_summarizer",
            LongDocumentSummarizerTool(self),
        )
        self._tool_registry.register_builtin("spawn_subagent", SpawnSubagentTool(self))
        self._tool_registry.register_builtin("wait_subagent", WaitSubagentTool())
        self._tool_registry.register_builtin("subagent_status", SubagentStatusTool())
        self._tool_registry.register_builtin("nudge_subagent", NudgeSubagentTool())
        from agent.tools.builtin_tools.document_generator import DocumentGeneratorTool
        self._tool_registry.register_builtin(
            "document_generator",
            DocumentGeneratorTool(Path(config.data_dir) / "artifacts"),
        )
        from agent.tools.builtin_tools.code_execution import CodeExecutionTool
        self._tool_registry.register_builtin(
            "code_execution",
            CodeExecutionTool(Path(config.data_dir) / "artifacts"),
        )
        from agent.tools.builtin_tools.email_sender import EmailSenderTool
        self._tool_registry.register_builtin(
            "email_sender",
            EmailSenderTool(Path(config.data_dir) / "artifacts"),
        )
        from app.agent_workspace import AgentWorkspaceStore
        self._agent_workspace_store = AgentWorkspaceStore()
        from app.chat_attachments import ChatAttachmentStore
        from agent.tools.builtin_tools.agent_workspace import AgentWorkspaceTool
        self._tool_registry.register_builtin(
            "agent_workspace",
            AgentWorkspaceTool(
                Path(config.data_dir) / "artifacts",
                store=self._agent_workspace_store,
                attachment_store=ChatAttachmentStore(Path(config.data_dir) / "chat_uploads"),
            ),
        )
        from app.mcp_client import MCPClientManager
        self._mcp_manager = MCPClientManager(
            lambda: getattr(self, "_tools_store", None)
        )
        self._tool_registry.set_mcp_manager(self._mcp_manager)
        self._mcp_manager.set_registry(self._tool_registry)
        self._agent_memory_store = AgentMemoryStore(self._engine)
        self._index_manager = IndexManager(
            config=config,
            loader=loader,
            embeddings_factory=embeddings_factory,
            llm_factory=self._create_ingestion_llm_factory(),
        )
        self._retriever_builder: Optional[RetrieverBuilder] = None
        self._apply_instructions(initial_instructions, clear_cache=False)
        self._instructions_mtime = self._get_instructions_mtime()

    def set_provider_store(self, store: Any) -> None:
        self._provider_store = store
        if self._chain_builder is not None:
            self._chain_builder.set_provider_store(store)
            self._chain_builder.set_provider_model_cache(self._provider_model_cache)

    def _create_ingestion_llm_factory(self) -> Optional[Callable[[], Any]]:
        """Create a sync LLM factory for ingestion-time image analysis.

        Returns a callable that creates a ``ChatOpenAI`` using the default LLM
        config. Returns ``None`` when no LLM API base is configured.
        """
        cfg = self.config
        api_base = getattr(cfg, "llm_api_base", None)
        api_key = getattr(cfg, "llm_api_key", None)
        model = getattr(cfg, "llm_model", None)
        if not api_base or not model:
            return None

        def _factory() -> Any:
            from langchain_openai import ChatOpenAI
            from app.llm_providers import normalize_base_url
            return ChatOpenAI(
                base_url=f"{normalize_base_url(api_base)}/v1",
                api_key=(api_key or "none"),
                model=model,
                temperature=0.0,
                max_tokens=2000,
                timeout=60,
            )

        return _factory

    async def __aenter__(self) -> "RAGApplication":
        """Optionally auto-initialize when used as an async context manager."""
        if self.config.auto_initialize:
            await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Release resources when leaving the async context manager."""
        await self.shutdown()

    @property
    def ready(self) -> bool:
        """Return ``True`` once initialization has completed."""
        return self._ready

    @property
    def engine(self) -> Optional[AsyncEngine]:
        """Return the shared async engine when available."""
        return getattr(self, "_engine", None)

    async def _init_single_collection(
        self,
        collection_name: str,
        folder: Optional[Path],
        embeddings: Any,
        force: bool,
        sem: asyncio.Semaphore,
    ) -> None:
        """Initialize a single knowledge collection (parallel-safe)."""
        async with sem:
            manager = self._index_manager.build_vector_manager(collection_name)

            # WebDAV sources have no local folder; skip document loading
            if folder is None and collection_name != REMOTE_COLLECTION_NAME:
                collection_exists = await manager.collection_exists()
                if collection_exists:
                    await manager.initialize([], embeddings, force=False)
                    self.vector_managers[collection_name] = manager
                    self.available_collections.add(collection_name)
                    vector_doc_count = await manager.get_document_count()
                    logger.info(
                        "Collection '%s' ready (WebDAV, vector_documents=%d)",
                        collection_name,
                        vector_doc_count,
                    )
                else:
                    logger.info("WebDAV collection '%s' has no vectors yet; skipping.", collection_name)
                return

            incremental_allowed = (
                self.config.indexing_incremental_enabled
                and not force
                and collection_name != REMOTE_COLLECTION_NAME
            )

            if incremental_allowed:
                try:
                    await manager.prepare_metadata_store()
                except Exception as exc:
                    logger.warning(
                        "Failed to prepare metadata store for '%s': %s", collection_name, exc
                    )

            collection_exists = await manager.collection_exists() if incremental_allowed else False

            filepaths_to_load: Optional[List[str]] = None
            fingerprints: Dict[str, str] = {}
            skipped_by_fingerprint: List[str] = []

            if collection_exists and incremental_allowed and folder is not None:
                fingerprints = await self._index_manager.compute_local_file_fingerprints_async(folder)
                stored_hashes = await manager.get_raw_file_hashes(list(fingerprints.keys()))
                changed_filepaths: List[str] = []
                for source, file_hash in fingerprints.items():
                    stored_hash = stored_hashes.get(source)
                    if stored_hash == file_hash:
                        skipped_by_fingerprint.append(source)
                    else:
                        abs_path = str(self.config.knowledge_base_root / source)
                        changed_filepaths.append(abs_path)

                if skipped_by_fingerprint:
                    logger.info(
                        "Preload fingerprint check for '%s' skipped %d unchanged files.",
                        collection_name,
                        len(skipped_by_fingerprint),
                    )

                if not changed_filepaths:
                    await manager.initialize([], embeddings, force=False)
                    self.vector_managers[collection_name] = manager
                    self.available_collections.add(collection_name)
                    vector_doc_count = await manager.get_document_count()
                    logger.info(
                        "Collection '%s' ready (sources=%d, vector_documents=%d)",
                        collection_name,
                        len(fingerprints),
                        vector_doc_count,
                    )
                    return

                filepaths_to_load = changed_filepaths

            documents = await self._index_manager.load_documents_for_collection(
                collection_name, folder, filepaths=filepaths_to_load
            )
            cleaned = clean_documents(documents)
            cleaned = [self._index_manager.normalize_document_metadata(collection_name, doc) for doc in cleaned]

            sources: Dict[str, List[Document]] = {}
            for doc in cleaned:
                meta = dict(doc.metadata or {})
                source = str(meta.get("source") or "unknown")
                sources.setdefault(source, []).append(doc)

            try:
                if collection_exists and incremental_allowed:
                    await manager.initialize([], embeddings, force=False)
                    updated_sources: List[str] = []
                    skipped_sources: List[str] = []
                    for source, docs_for_source in sources.items():
                        changed_hash = await manager.source_unchanged(source, docs_for_source)
                        if changed_hash is None:
                            logger.info("Source '%s' unchanged; skipping chunking and embedding update.", source)
                            raw_file_hash = fingerprints.get(source)
                            if raw_file_hash is not None:
                                stored_source_hash = await manager._get_source_hash(source)
                                if stored_source_hash is not None:
                                    await manager.upsert_source_hash(
                                        source, stored_source_hash, raw_file_hash=raw_file_hash
                                    )
                            skipped_sources.append(source)
                            continue
                        logger.info("Source '%s' changed; chunking and refreshing embeddings.", source)
                        doc_splits = self._index_manager.prepare_splits(collection_name, source, docs_for_source)
                        result = await manager.upsert_documents(
                            source, doc_splits, embeddings,
                            source_hash=changed_hash,
                            raw_file_hash=fingerprints.get(source),
                        )
                        if result["action"] == "updated":
                            updated_sources.append(f"{source} (deleted={result['deleted']}, added={result['added']})")

                    if skipped_sources:
                        logger.info(
                            "Incremental reindex for '%s' skipped %d unchanged sources.",
                            collection_name,
                            len(skipped_sources),
                        )
                    if updated_sources:
                        logger.info(
                            "Incremental reindex for '%s' updated %d sources: %s",
                            collection_name,
                            len(updated_sources),
                            "; ".join(updated_sources[:10]) + ("…" if len(updated_sources) > 10 else ""),
                        )
                else:
                    all_splits: List[Document] = []
                    for source, docs_for_source in sources.items():
                        all_splits.extend(self._index_manager.prepare_splits(collection_name, source, docs_for_source))
                    await manager.initialize(all_splits, embeddings, force=force)
                    if incremental_allowed:
                        for source, docs_for_source in sources.items():
                            raw_file_hash = fingerprints.get(source)
                            await manager.upsert_source_hash(
                                source, manager.hash_source_documents(docs_for_source), raw_file_hash=raw_file_hash
                            )

                if manager.ready:
                    self.vector_managers[collection_name] = manager
                    self.available_collections.add(collection_name)
                    vector_doc_count = await manager.get_document_count()
                    logger.info(
                        "Collection '%s' ready (sources=%d, vector_documents=%d)",
                        collection_name,
                        len(sources),
                        vector_doc_count,
                    )
            except Exception as exc:
                logger.error(
                    "Vector store initialization skipped for collection '%s' due to error: %s",
                    collection_name,
                    exc,
                )
                if self._recovery_manager is not None:
                    await self._recovery_manager.mark_unhealthy()

    async def initialize(self, force: bool = False, skip_engine_creation: bool = False) -> None:
        """Load documents, refresh the vector store, and build the LangChain pipelines.

        Passing ``force=True`` (or setting ``VECTOR_FORCE_REBUILD``) rebuilds the PGVector
        collection even when a prior run already populated it.
        """
        force = force or self.config.vector_force_rebuild
        init_start = time.perf_counter()
        if self.ready and not force:
            logger.info("RAGApplication already initialized; skipping.")
            return

        logger.info("Initializing RAG application pipeline...")
        self.collection_paths = self._index_manager.discover_collections()
        if self.config.knowledge_urls:
            if REMOTE_COLLECTION_NAME in self.collection_paths:
                logger.warning(
                    "Skipping remote URL collection because '%s' already exists as a folder.",
                    REMOTE_COLLECTION_NAME,
                )
            else:
                self.collection_paths[REMOTE_COLLECTION_NAME] = None
        self.vector_managers = {}
        self.available_collections = set()

        if not self.collection_paths:
            logger.warning(
                "No knowledge collections found under %s. Assistants will run without retrieval.",
                self.config.knowledge_base_root,
            )

        self.embeddings = self._index_manager.embeddings_factory(self.config)

        # Resolve the API key for the configured embedding provider
        if self._provider_store and self.config.embeddings_api_url:
            try:
                from app.llm_providers import normalize_base_url
                url = normalize_base_url(self.config.embeddings_api_url)
                provider = None
                for p in await self._provider_store.list_providers():
                    if normalize_base_url(p.base_url) == url:
                        provider = p
                        break
                if provider is not None:
                    key = await self._provider_store.get_api_key(provider.id)
                    if key:
                        self.embeddings.api_key = key
            except Exception:
                pass

        embeddings = (
            SlowSystemEmbeddingWrapper(self.embeddings)
            if self.config.system_slow
            else self.embeddings
        )

        # Shared engine for all vector managers and hybrid retrievers
        if not skip_engine_creation or getattr(self, "_engine", None) is None:
            async_conn = _to_async_connection_string(self.config.connection_string)
            self._engine: AsyncEngine = create_async_engine(async_conn, pool_size=10, max_overflow=20, pool_pre_ping=True)
        self._index_manager.engine = self._engine

        # Discover DB-only collections and hydrate files
        if self._engine is not None:
            from app.stores import RawFileStore
            raw_store = RawFileStore(self._engine)
            db_collections = await raw_store.list_collections()
            for col_name in db_collections:
                if col_name not in self.collection_paths:
                    folder = self.config.knowledge_base_root / col_name
                    folder.mkdir(parents=True, exist_ok=True)
                    self.collection_paths[col_name] = folder
                    raw_files = await raw_store.list_by_collection(col_name)
                    for rf in raw_files:
                        file_path = folder / rf["filename"]
                        if not file_path.exists():
                            db_file = await raw_store.get(col_name, rf["source"])
                            if db_file is not None:
                                data = db_file["file_data"]
                                if isinstance(data, bytes):
                                    file_path.write_bytes(data)
                                    logger.info("Hydrated '%s' for DB-only collection '%s'.", rf["filename"], col_name)

            # Discover WebDAV sources and register as collection_paths (no local folder)
            from app.webdav_store import WebDAVSourceStore
            webdav_store = WebDAVSourceStore(self._engine)
            try:
                webdav_sources = await webdav_store.list_sources()
                for src in webdav_sources:
                    src_name = src.get("name")
                    if src_name and src_name not in self.collection_paths:
                        self.collection_paths[src_name] = None
                        logger.info("Registered WebDAV source '%s' as collection.", src_name)
            except Exception as exc:
                logger.warning("Failed to load WebDAV sources during init: %s", exc)

        # DB-first instructions load / bootstrap
        from app.stores import InstructionsStore
        self._instructions_store = InstructionsStore(self._engine)
        bootstrapped = await self._instructions_store.bootstrap_from_file(
            str(self.config.instructions_file)
        )
        db_yaml = await self._instructions_store.get_current()
        if db_yaml is not None:
            db_instructions = load_instructions_from_yaml_text(db_yaml)
            if db_instructions:
                self._apply_instructions(db_instructions, clear_cache=False)
                self._instructions_db_version_id = await self._get_current_instructions_version_id()
                logger.info("Assistant instructions loaded from database.")
        elif bootstrapped:
            db_yaml = await self._instructions_store.get_current()
            if db_yaml is not None:
                db_instructions = load_instructions_from_yaml_text(db_yaml)
                if db_instructions:
                    self._apply_instructions(db_instructions, clear_cache=False)
                    self._instructions_db_version_id = await self._get_current_instructions_version_id()

        # Load custom tools from DB — sync MCP servers FIRST so freshly
        # materialized kind="mcp" entries get registered in the same pass.
        tools_store = getattr(self, "_tools_store", None)
        if tools_store is not None:
            try:
                await self._mcp_manager.sync()
            except Exception as exc:
                logger.warning("MCP server sync failed: %s", exc)
            try:
                await self._tool_registry.load_custom_tools(tools_store)
                logger.info("Loaded %d custom tools from DB", len(self._tool_registry._custom))
            except Exception as exc:
                logger.warning("Failed to load custom tools from DB: %s", exc)

        await self._tool_registry.refresh_schemas()

        _collection_sem = asyncio.Semaphore(3)
        tasks = [
            self._init_single_collection(collection_name, folder, embeddings, force, _collection_sem)
            for collection_name, folder in self.collection_paths.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for collection_name, result in zip(self.collection_paths.keys(), results):
            if isinstance(result, Exception):
                logger.error(
                    "Collection '%s' initialization failed: %s",
                    collection_name,
                    result,
                )

        if not self.vector_managers:
            logger.warning("No vector collections are ready; system will operate in LLM-only mode.")
            if self._recovery_manager is not None:
                await self._recovery_manager.mark_unhealthy()

        self._retriever_builder = RetrieverBuilder(
            config=self.config,
            vector_managers=self.vector_managers,
            embeddings=self.embeddings,
            engine=self._engine,
        )
        self._recovery_manager = RecoveryManager(self.vector_managers)
        self._chain_builder = ChainBuilder(
            config=self.config,
            instructions=self.instructions,
            provider_store=self._provider_store,
        )
        self._chain_builder.set_provider_model_cache(self._provider_model_cache)
        await self.refresh_provider_model_cache()

        self._ready = True
        duration_ms = (time.perf_counter() - init_start) * 1000
        logger.info(
            "RAG application initialized. | Process Time: %s",
            colorize_process_time(duration_ms),
            extra={"process_time": duration_ms},
        )

    async def index_knowledge_files(self, collection: str, filenames: List[str]) -> dict:
        """Load and index only the specified filenames into the vector store for a collection."""
        if not self.ready:
            raise RuntimeError("RAGApplication is not initialized")

        folder = self.config.knowledge_base_root / collection
        folder.mkdir(parents=True, exist_ok=True)

        # Hydrate files from DB if they are missing on filesystem
        if self._engine is not None:
            from app.stores import RawFileStore
            raw_store = RawFileStore(self._engine)
            for fn in filenames:
                file_path = folder / fn
                if not file_path.exists():
                    source = f"{collection}/{fn}"
                    db_file = await raw_store.get(collection, source)
                    if db_file is not None:
                        data = db_file["file_data"]
                        if isinstance(data, bytes):
                            file_path.write_bytes(data)
                            logger.info("Hydrated '%s' from database to filesystem.", fn)

        manager = self.vector_managers.get(collection)
        if manager is None:
            manager = self._index_manager.build_vector_manager(collection)
            self.vector_managers[collection] = manager
            self.available_collections.add(collection)

        embeddings = (
            SlowSystemEmbeddingWrapper(self.embeddings)
            if self.config.system_slow
            else self.embeddings
        )

        filepaths = [str(folder / fn) for fn in filenames]
        documents = await self._index_manager.load_documents_for_collection(
            collection, folder, filepaths=filepaths
        )
        cleaned = clean_documents(documents)
        cleaned = [
            self._index_manager.normalize_document_metadata(collection, doc)
            for doc in cleaned
        ]

        sources: Dict[str, List[Document]] = {}
        for doc in cleaned:
            meta = dict(doc.metadata or {})
            sources.setdefault(str(meta.get("source") or "unknown"), []).append(doc)

        # Ensure schema exists before checking collection existence
        await manager.prepare_metadata_store()
        collection_exists = await manager.collection_exists()
        fingerprints = await self._index_manager.compute_local_file_fingerprints_async(folder)

        if not collection_exists:
            all_splits: List[Document] = []
            for source, docs_for_source in sources.items():
                all_splits.extend(
                    self._index_manager.prepare_splits(collection, source, docs_for_source)
                )
            await manager.initialize(all_splits, embeddings, force=False)
            for source, docs_for_source in sources.items():
                raw_file_hash = fingerprints.get(source)
                await manager.upsert_source_hash(
                    source,
                    manager.hash_source_documents(docs_for_source),
                    raw_file_hash=raw_file_hash,
                )
            return {
                "initialized": True,
                "sources_count": len(sources),
                "chunks": len(all_splits),
            }

        await manager.initialize([], embeddings, force=False)
        results: List[dict] = []
        for source, docs_for_source in sources.items():
            changed_hash = await manager.source_unchanged(source, docs_for_source)
            if changed_hash is None:
                raw_file_hash = fingerprints.get(source)
                if raw_file_hash is not None:
                    stored = await manager._get_source_hash(source)
                    if stored is not None:
                        await manager.upsert_source_hash(
                            source, stored, raw_file_hash=raw_file_hash
                        )
                results.append({"source": source, "action": "skipped"})
                continue

            doc_splits = self._index_manager.prepare_splits(
                collection, source, docs_for_source
            )
            res = await manager.upsert_documents(
                source,
                doc_splits,
                embeddings,
                source_hash=changed_hash,
                raw_file_hash=fingerprints.get(source),
            )
            results.append(
                {
                    "source": source,
                    "action": res["action"],
                    "deleted": res.get("deleted"),
                    "added": res.get("added"),
                }
            )

        return {"initialized": False, "sources": results}

    async def list_indexed_knowledge_sources(self, collection: str) -> List[str]:
        """Return all sources currently indexed in the vector store for a collection."""
        if not self.ready:
            raise RuntimeError("RAGApplication is not initialized")
        manager = self.vector_managers.get(collection)
        if manager is None:
            manager = self._index_manager.build_vector_manager(collection)
        await manager.prepare_metadata_store()
        return await manager.list_sources()

    async def get_okf_concepts(self, collection: str) -> List[Dict[str, Any]]:
        """Return all OKF concepts stored in the vector store for a collection."""
        if not self.ready:
            raise RuntimeError("RAGApplication is not initialized")
        manager = self.vector_managers.get(collection)
        if manager is None:
            manager = self._index_manager.build_vector_manager(collection)
        await manager.prepare_metadata_store()
        return await manager.get_okf_concepts()

    async def delete_knowledge_source(
        self, collection: str, source: str, *, delete_file: bool = True
    ) -> dict:
        """Remove a source from the vector store and optionally the filesystem."""
        if not self.ready:
            raise RuntimeError("RAGApplication is not initialized")

        result: dict = {"file_deleted": False, "db_deleted": {}}
        if delete_file:
            target = self.config.knowledge_base_root / collection / source
            if target.exists():
                target.unlink()
                result["file_deleted"] = True

        manager = self.vector_managers.get(collection)
        if manager is None:
            manager = self._index_manager.build_vector_manager(collection)
        await manager.prepare_metadata_store()
        if await manager.collection_exists():
            # Try plain source name first; fall back to collection-prefixed path
            db_result = await manager.delete_source(source)
            if db_result.get("deleted_state", 0) == 0:
                prefixed = f"{collection}/{source}"
                db_result = await manager.delete_source(prefixed)
            result["db_deleted"] = db_result
            self._refresh_retriever_builder()

        return result

    async def delete_knowledge_collection(
        self, collection: str, *, delete_folder: bool = True
    ) -> dict:
        """Remove a collection from the vector store and optionally the filesystem."""
        if not self.ready:
            raise RuntimeError("RAGApplication is not initialized")

        result: dict = {"folder_deleted": False, "db_deleted": {}}
        if delete_folder:
            target = self.config.knowledge_base_root / collection
            if target.exists():
                import shutil
                shutil.rmtree(target)
                result["folder_deleted"] = True

        manager = self.vector_managers.get(collection)
        if manager is None:
            manager = self._index_manager.build_vector_manager(collection)
        await manager.prepare_metadata_store()
        if await manager.collection_exists():
            result["db_deleted"] = await manager.drop_collection()
            self.vector_managers.pop(collection, None)
            self.available_collections.discard(collection)
            self._refresh_retriever_builder()

        return result

    def _refresh_retriever_builder(self) -> None:
        """Rebuild retriever builder with current vector managers."""
        if self.embeddings is None:
            return
        self._retriever_builder = RetrieverBuilder(
            config=self.config,
            vector_managers=self.vector_managers,
            embeddings=self.embeddings,
            engine=self._engine,
        )

    async def shutdown(self) -> None:
        """Dispose embeddings, chain state and sandbox containers so the instance can be re-used later."""
        refresh_task = getattr(self, "_provider_refresh_task", None)
        if refresh_task is not None and not refresh_task.done():
            refresh_task.cancel()
            try:
                await refresh_task
            except (asyncio.CancelledError, Exception):
                pass
        self._provider_refresh_task = None
        try:
            from agent.tools.docker_sandbox import ConversationSandboxManager
            await ConversationSandboxManager().destroy_all()
        except Exception as exc:
            logger.warning("Sandbox cleanup during shutdown failed: %s", exc)
        if self.embeddings:
            await self.embeddings.close()
            self.embeddings = None
        if self._chain_builder is not None:
            await self._chain_builder.clear_cache()
        if getattr(self, "_engine", None) is not None:
            await self._engine.dispose()
            self._engine = None  # type: ignore[assignment]
        self._ready = False
        logger.info("RAG application resources released.")

    async def run(
        self,
        input_text: str,
        chat_history: Sequence[BaseMessage] | None = None,
        model_name: str = "CyberSecurityAssistant",
        retrieval_info: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        conversation_id: Optional[str] = None,
        multimodal_input: Optional[HumanMessage] = None,
        subagent_state: Optional[SubagentState] = None,
        chat_user_id: Optional[str] = None,
    ) -> dict:
        """Execute the non-streaming RAG chain and return the LangChain payload.

        When ``multimodal_input`` is provided, its content (a list of text/image
        parts) is used as the ``input`` payload value instead of the plain
        ``input_text`` string, enabling vision-capable LLMs to see images.
        """
        if not self.ready:
            raise RuntimeError("RAGApplication is not initialized")
        history_list = list(chat_history or [])
        assistant_entry = self.instructions.get(model_name, {})
        agent_enabled = bool(assistant_entry.get("agent_enabled", False))

        retrieval_query = _build_retrieval_query(
            input_text,
            history_list,
            enabled=self.config.retrieval_history_enabled,
            max_user_messages=self.config.retrieval_history_user_messages,
            max_chars=self.config.retrieval_history_max_chars,
            standalone_detection=self.config.retrieval_history_standalone_detection,
        )
        primary_query = _clean_noise(input_text)
        if retrieval_info is not None:
            retrieval_info["retrieval_query"] = retrieval_query
            retrieval_info["retrieval_query_primary"] = primary_query
        # Use multimodal content as input if provided, otherwise plain text
        effective_input = multimodal_input.content if multimodal_input is not None else input_text
        # `input` stays text-only for retrievers/contextualize; `input_messages`
        # carries the real HumanMessage (multimodal parts intact) into the
        # MessagesPlaceholder prompts — ("human","{input}") would stringify a
        # content list into a Python repr and destroy image parts.
        payload = {"input": input_text, "input_messages": [HumanMessage(content=effective_input)], "chat_history": history_list, "retrieval_query": retrieval_query, "retrieval_query_primary": primary_query}
        if self._recovery_manager is not None and self._recovery_manager.degraded:
            await self._recovery_manager.attempt_recovery()
        retrieval_degraded = self._recovery_manager.degraded if self._recovery_manager else True
        use_contextualize = bool(history_list) and self.config.contextualize_question_enabled
        retriever = None
        if self._retriever_builder is not None:
            retriever = await self._retriever_builder.get_retriever_for_collections(
                self._get_assistant_collections(model_name),
                retrieval_degraded,
            )
        if retriever is not None:
            retriever = RecencyWeightedRetriever(
                retriever,
                primary_share=getattr(self.config, "retrieval_recency_primary_share", 0.6),
            )
            if retrieval_info is not None:
                retriever = MetricsRecordingRetriever(retriever, callback=lambda info: retrieval_info.update(info))

        # Retrieve RAG context for agent mode (if not degraded)
        rag_context = ""
        if retriever is not None and not retrieval_degraded:
            try:
                recency_enabled = bool(getattr(self.config, "retrieval_recency_enabled", True))
                if retrieval_query and primary_query and primary_query != retrieval_query and recency_enabled:
                    docs = await retriever.ainvoke({"primary": primary_query, "contextual": retrieval_query})
                else:
                    docs = await retriever.ainvoke(retrieval_query)
                rag_context = _format_rag_context(docs)
            except Exception as exc:
                logger.warning("Failed to retrieve RAG context for agent mode: %s", exc)

        # Summarize RAG context if it exceeds the reserved token budget
        provider_windows = await get_provider_context_windows(self.config)
        agent_memory_text: Optional[str] = None
        if (
            assistant_entry.get("agent_memory_enabled")
            and self._agent_memory_store
            and conversation_id
            and not (subagent_state is not None and not subagent_state.is_root)
        ):
            try:
                agent_memory_text = await self._agent_memory_store.get_memory(conversation_id, model_name)
            except Exception:
                logger.debug("Failed to load agent memory for conversation %s", conversation_id, exc_info=True)
        rh_settings = resolve_rolling_history_settings(
            self.config, assistant_entry, provider_windows=provider_windows, agent_memory=agent_memory_text
        )
        if rag_context and rh_settings.enabled and rh_settings.rag_reserved_tokens > 0:
            rag_context, _summarized = await summarize_context_if_needed(
                rag_context,
                token_budget=rh_settings.rag_reserved_tokens,
                model=rh_settings.model or "gpt-4o-mini",
                timeout_seconds=rh_settings.summary_timeout_seconds,
                summary_llm_factory=lambda: asyncio.create_task(self.get_metadata_llm(model_name, "summary")),
                summary_context_window=rh_settings.context_window,
                usage_sink=retrieval_info.setdefault("_compression_usage", []) if isinstance(retrieval_info, dict) else None,
            )

        emotion_hint = retrieval_info.pop("_emotion_hint", None) if retrieval_info else None
        streaming_chain, conversational_chain, llm_only_chain = await self._chain_builder.build_chains(
            model_name, use_contextualize=use_contextualize, retriever=retriever,
            retrieval_degraded=retrieval_degraded, rag_context=rag_context,
            emotion_hint=emotion_hint,
        )

        # Agent mode: use multi-step tool loop instead of single chain invocation
        tools_dict = assistant_entry.get("tools") or {}
        subagents_cfg = assistant_entry.get("subagents") or {}
        allowed_tools = _resolve_allowed_tools(tools_dict, subagents_cfg, subagent_state)
        # Agent workspace: resolve the scope dir once in the root run; subagents
        # inherit the resolved dir via subagent_state.workspace_ctx (their own
        # conversation ids are derived and would resolve to wrong dirs).
        workspace_ctx: Optional[Dict[str, Any]] = None
        workspace_cfg = assistant_entry.get("workspace") or {}
        if agent_enabled and workspace_cfg.get("enabled"):
            if subagent_state is not None and subagent_state.workspace_ctx is not None:
                workspace_ctx = subagent_state.workspace_ctx
            else:
                workspace_ctx = {
                    "dir": self._agent_workspace_store.scope_dir(
                        model_name, workspace_cfg.get("scope"), conversation_id, chat_user_id
                    ),
                    "cfg": workspace_cfg,
                    "conv": conversation_id,
                }
        auto_excluded = _subagent_excluded_tools(subagents_cfg, subagent_state)
        if (
            agent_enabled and workspace_ctx is not None
            and "agent_workspace" not in allowed_tools and "agent_workspace" not in auto_excluded
        ):
            allowed_tools.append("agent_workspace")
        if agent_enabled and "wait" not in allowed_tools:
            allowed_tools.append("wait")
        if agent_enabled and TODO_TOOL_NAME not in allowed_tools:
            allowed_tools.append(TODO_TOOL_NAME)
        if agent_enabled and (subagents_cfg.get("enabled") or "spawn_subagent" in allowed_tools):
            if subagent_state is None:
                manager = SubagentManager(
                    subagents_cfg, conversation_id, self,
                    idle_seconds=getattr(self.config, "stream_idle_timeout_seconds", DEFAULT_STREAM_IDLE_SECONDS),
                )
                subagent_state = make_subagent_state(manager, model_name, workspace_ctx=workspace_ctx)
            if subagent_state.depth < subagent_state.manager.max_depth:
                for name in ("spawn_subagent", "wait_subagent", "subagent_status", "nudge_subagent"):
                    if name not in allowed_tools and name not in auto_excluded:
                        allowed_tools.append(name)
        runner: Optional[AgentRunner] = None
        if agent_enabled and allowed_tools and self._chain_builder is not None:
            try:
                llm = await self._chain_builder._get_agent_llm(model_name, max_tokens=None)
                max_iterations = assistant_entry.get("max_iterations") or 10
                max_tool_calls = assistant_entry.get("max_tool_calls") or 10
                max_parallel_tool_calls = assistant_entry.get("max_parallel_tool_calls") or 5
                agent_memory_enabled = bool(assistant_entry.get("agent_memory_enabled", False)) and not (
                    subagent_state is not None and not subagent_state.is_root
                )
                if agent_memory_enabled and "agent_memory" not in allowed_tools:
                    allowed_tools = list(allowed_tools) + ["agent_memory"]
                logger.debug("[Agent run] max_iter=%s max_tool_calls=%s max_parallel=%s for assistant %s", max_iterations, max_tool_calls, max_parallel_tool_calls, model_name)
                raw_instructions = assistant_entry.get("system_instructions", "")
                has_agent_memory_placeholder = "{{agent_memory}}" in raw_instructions
                tool_calling_hint = await self._get_provider_tool_calling_hint(model_name)
                observer = self._build_observer(assistant_entry, model_name, subagent_state)
                runner = AgentRunner(
                    self._tool_registry, allowed_tools,
                    max_iterations=max_iterations,
                    max_tool_calls=max_tool_calls,
                    max_parallel_tool_calls=max_parallel_tool_calls,
                    supports_native_tool_calls=tool_calling_hint,
                    conversation_id=conversation_id,
                    agent_memory_enabled=agent_memory_enabled,
                    assistant_name=model_name,
                    agent_memory_store=self._agent_memory_store,
                    agent_memory_already_injected=has_agent_memory_placeholder,
                    tool_overrides=tools_dict,
                    rolling_history_settings=rh_settings if rh_settings.enabled else None,
                    summary_llm_factory=lambda: asyncio.create_task(self.get_metadata_llm(model_name, "summary")) if rh_settings.enabled else None,
                    subagent_state=subagent_state,
                    stream_idle_seconds=getattr(self.config, "stream_idle_timeout_seconds", DEFAULT_STREAM_IDLE_SECONDS),
                    observer=observer,
                    workspace_ctx=workspace_ctx,
                )
                system_instructions = render_system_instructions(
                    raw_instructions, rag_context=rag_context,
                    keep_agent_memory_placeholder=agent_memory_enabled,
                )
                logger.debug("[Agent run] raw_instructions contains {{current_datetime}}: %s", "{{current_datetime}}" in raw_instructions or "{{current\\_datetime}}" in raw_instructions)
                logger.debug("[Agent run] system_instructions contains {{current_datetime}}: %s", "{{current_datetime}}" in system_instructions or "{{current\\_datetime}}" in system_instructions)
                if rag_context and "{{context}}" not in raw_instructions:
                    system_instructions = system_instructions.rstrip() + "\n\nRelevant context:\n" + rag_context
                if emotion_hint:
                    system_instructions = system_instructions.rstrip() + "\n\n" + emotion_hint
                result = await runner.run(llm, input_text, history_list, system_instructions, rag_context=rag_context, conversation_id=conversation_id, multimodal_input=multimodal_input)
                if retrieval_info is not None:
                    retrieval_info["agent_mode"] = True
                    if observer is not None and observer.usage:
                        retrieval_info["_observer_usage"] = dict(observer.usage)
                    todo_list = result.get("todo_list")
                    if todo_list is not None:
                        retrieval_info["todo_list"] = todo_list
                    trace = result.get("agent_trace")
                    if trace:
                        retrieval_info["_agent_trace"] = trace
                    usage = result.get("agent_usage_metadata")
                    nested_usage = (
                        subagent_state.manager.usage_totals()
                        if subagent_state and subagent_state.is_root
                        else {}
                    )
                    if nested_usage:
                        usage = dict(usage or {})
                        for key, value in nested_usage.items():
                            usage[key] = int(usage.get(key) or 0) + value
                        subagent_tokens = int(nested_usage.get("prompt_tokens") or 0) + int(
                            nested_usage.get("completion_tokens") or 0
                        )
                        if subagent_tokens:
                            usage["subagent_tokens"] = subagent_tokens
                    if usage:
                        retrieval_info["_llm_usage_metadata"] = usage
                return result
            except Exception as exc:
                if retrieval_info is not None and runner is not None:
                    todo_list = runner.todo_list_snapshot()
                    if todo_list is not None:
                        retrieval_info["todo_list"] = todo_list
                logger.warning("Agent run failed for %s, falling back to normal chain: %s", model_name, exc, exc_info=True)

        if (self._recovery_manager is not None and self._recovery_manager.degraded) and streaming_chain is None:
            if retrieval_info is not None:
                retrieval_info.update({"retrieval_time_ms": 0.0, "chunks_retrieved": 0, "status": "skipped"})
            result = await self._invoke_llm_only(llm_only_chain, payload)
        else:
            result = await self._invoke_with_fallback(conversational_chain, llm_only_chain, payload)
        usage_meta = extract_usage_metadata(result)
        if usage_meta is not None and retrieval_info is not None:
            retrieval_info["_llm_usage_metadata"] = usage_meta
        return result

    async def run_streaming(
        self,
        user_input: str,
        chat_history: Sequence[BaseMessage] | None = None,
        model_name: str = "CyberSecurityAssistant",
        retrieval_info: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        conversation_id: Optional[str] = None,
        multimodal_input: Optional[HumanMessage] = None,
        subagent_state: Optional[SubagentState] = None,
        injected_input_queue: Optional[asyncio.Queue] = None,
        input_event: Optional[asyncio.Event] = None,
        subagent_channel_registry: Optional[Dict[str, Any]] = None,
        chat_user_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Yield incremental answer chunks with automatic fallback on retrieval failures."""
        if not self.ready:
            raise RuntimeError("RAGApplication is not initialized")

        history_list = list(chat_history or [])
        assistant_entry = self.instructions.get(model_name, {})
        agent_enabled = bool(assistant_entry.get("agent_enabled", False))

        retrieval_query = _build_retrieval_query(
            user_input,
            history_list,
            enabled=self.config.retrieval_history_enabled,
            max_user_messages=self.config.retrieval_history_user_messages,
            max_chars=self.config.retrieval_history_max_chars,
            standalone_detection=self.config.retrieval_history_standalone_detection,
        )
        primary_query = _clean_noise(user_input)
        if retrieval_info is not None:
            retrieval_info["retrieval_query"] = retrieval_query
            retrieval_info["retrieval_query_primary"] = primary_query
        # Use multimodal content as input if provided, otherwise plain text
        effective_input = multimodal_input.content if multimodal_input is not None else user_input
        payload = {"input": user_input, "input_messages": [HumanMessage(content=effective_input)], "chat_history": history_list, "retrieval_query": retrieval_query, "retrieval_query_primary": primary_query}
        streamed = False

        if self._recovery_manager is not None and self._recovery_manager.degraded:
            await self._recovery_manager.attempt_recovery()

        retrieval_degraded = self._recovery_manager.degraded if self._recovery_manager else True
        use_contextualize = bool(history_list) and self.config.contextualize_question_enabled
        retriever = None
        if self._retriever_builder is not None:
            retriever = await self._retriever_builder.get_retriever_for_collections(
                self._get_assistant_collections(model_name),
                retrieval_degraded,
            )
        if retriever is not None:
            retriever = RecencyWeightedRetriever(
                retriever,
                primary_share=getattr(self.config, "retrieval_recency_primary_share", 0.6),
            )
            if retrieval_info is not None:
                retriever = MetricsRecordingRetriever(retriever, callback=lambda info: retrieval_info.update(info))

        # Retrieve RAG context for agent mode (if not degraded)
        rag_context = ""
        if retriever is not None and not retrieval_degraded:
            try:
                recency_enabled = bool(getattr(self.config, "retrieval_recency_enabled", True))
                if retrieval_query and primary_query and primary_query != retrieval_query and recency_enabled:
                    docs = await retriever.ainvoke({"primary": primary_query, "contextual": retrieval_query})
                else:
                    docs = await retriever.ainvoke(retrieval_query)
                rag_context = _format_rag_context(docs)
            except Exception as exc:
                logger.warning("Failed to retrieve RAG context for agent mode: %s", exc)

        # Summarize RAG context if it exceeds the reserved token budget
        provider_windows = await get_provider_context_windows(self.config)
        agent_memory_text: Optional[str] = None
        if (
            assistant_entry.get("agent_memory_enabled")
            and self._agent_memory_store
            and conversation_id
            and not (subagent_state is not None and not subagent_state.is_root)
        ):
            try:
                agent_memory_text = await self._agent_memory_store.get_memory(conversation_id, model_name)
            except Exception:
                logger.debug("Failed to load agent memory for conversation %s", conversation_id, exc_info=True)
        rh_settings = resolve_rolling_history_settings(
            self.config, assistant_entry, provider_windows=provider_windows, agent_memory=agent_memory_text
        )
        if rag_context and rh_settings.enabled and rh_settings.rag_reserved_tokens > 0:
            rag_context, _summarized = await summarize_context_if_needed(
                rag_context,
                token_budget=rh_settings.rag_reserved_tokens,
                model=rh_settings.model or "gpt-4o-mini",
                timeout_seconds=rh_settings.summary_timeout_seconds,
                summary_llm_factory=lambda: asyncio.create_task(self.get_metadata_llm(model_name, "summary")),
                summary_context_window=rh_settings.context_window,
                usage_sink=retrieval_info.setdefault("_compression_usage", []) if isinstance(retrieval_info, dict) else None,
            )

        emotion_hint = retrieval_info.pop("_emotion_hint", None) if retrieval_info else None
        streaming_chain, conversational_chain, llm_only_chain = await self._chain_builder.build_chains(
            model_name, use_contextualize=use_contextualize, retriever=retriever,
            retrieval_degraded=retrieval_degraded, rag_context=rag_context,
            emotion_hint=emotion_hint,
        )

        # Agent mode: delegate to streaming agent runner
        tools_dict = assistant_entry.get("tools") or {}
        subagents_cfg = assistant_entry.get("subagents") or {}
        allowed_tools = _resolve_allowed_tools(tools_dict, subagents_cfg, subagent_state)
        # Agent workspace: root run resolves the scope dir; subagents inherit it
        # via subagent_state.workspace_ctx (their conv ids are derived).
        workspace_ctx: Optional[Dict[str, Any]] = None
        workspace_cfg = assistant_entry.get("workspace") or {}
        if agent_enabled and workspace_cfg.get("enabled"):
            if subagent_state is not None and subagent_state.workspace_ctx is not None:
                workspace_ctx = subagent_state.workspace_ctx
            else:
                workspace_ctx = {
                    "dir": self._agent_workspace_store.scope_dir(
                        model_name, workspace_cfg.get("scope"), conversation_id, chat_user_id
                    ),
                    "cfg": workspace_cfg,
                    "conv": conversation_id,
                }
        auto_excluded = _subagent_excluded_tools(subagents_cfg, subagent_state)
        if (
            agent_enabled and workspace_ctx is not None
            and "agent_workspace" not in allowed_tools and "agent_workspace" not in auto_excluded
        ):
            allowed_tools.append("agent_workspace")
        if agent_enabled and "wait" not in allowed_tools:
            allowed_tools.append("wait")
        if agent_enabled and TODO_TOOL_NAME not in allowed_tools:
            allowed_tools.append(TODO_TOOL_NAME)
        if agent_enabled and (subagents_cfg.get("enabled") or "spawn_subagent" in allowed_tools):
            if subagent_state is None:
                manager = SubagentManager(
                    subagents_cfg, conversation_id, self,
                    idle_seconds=getattr(self.config, "stream_idle_timeout_seconds", DEFAULT_STREAM_IDLE_SECONDS),
                    channel_registry=subagent_channel_registry,
                )
                subagent_state = make_subagent_state(manager, model_name, workspace_ctx=workspace_ctx)
            if subagent_state.depth < subagent_state.manager.max_depth:
                for name in ("spawn_subagent", "wait_subagent", "subagent_status", "nudge_subagent"):
                    if name not in allowed_tools and name not in auto_excluded:
                        allowed_tools.append(name)
        runner: Optional[AgentRunner] = None
        if agent_enabled and allowed_tools and self._chain_builder is not None:
            try:
                llm = await self._chain_builder._get_agent_llm(model_name, max_tokens=None)
                max_iterations = assistant_entry.get("max_iterations") or 10
                max_tool_calls = assistant_entry.get("max_tool_calls") or 10
                max_parallel_tool_calls = assistant_entry.get("max_parallel_tool_calls") or 5
                agent_memory_enabled = bool(assistant_entry.get("agent_memory_enabled", False)) and not (
                    subagent_state is not None and not subagent_state.is_root
                )
                if agent_memory_enabled and "agent_memory" not in allowed_tools:
                    allowed_tools = list(allowed_tools) + ["agent_memory"]
                logger.debug("[Agent streaming] max_iter=%s max_tool_calls=%s max_parallel=%s for assistant %s", max_iterations, max_tool_calls, max_parallel_tool_calls, model_name)
                raw_instructions = assistant_entry.get("system_instructions", "")
                has_agent_memory_placeholder = "{{agent_memory}}" in raw_instructions
                tool_calling_hint = await self._get_provider_tool_calling_hint(model_name)
                observer = self._build_observer(assistant_entry, model_name, subagent_state)
                runner = AgentRunner(
                    self._tool_registry, allowed_tools,
                    max_iterations=max_iterations,
                    max_tool_calls=max_tool_calls,
                    max_parallel_tool_calls=max_parallel_tool_calls,
                    supports_native_tool_calls=tool_calling_hint,
                    conversation_id=conversation_id,
                    agent_memory_enabled=agent_memory_enabled,
                    assistant_name=model_name,
                    agent_memory_store=self._agent_memory_store,
                    agent_memory_already_injected=has_agent_memory_placeholder,
                    tool_overrides=tools_dict,
                    rolling_history_settings=rh_settings if rh_settings.enabled else None,
                    summary_llm_factory=lambda: asyncio.create_task(self.get_metadata_llm(model_name, "summary")) if rh_settings.enabled else None,
                    subagent_state=subagent_state,
                    stream_idle_seconds=getattr(self.config, "stream_idle_timeout_seconds", DEFAULT_STREAM_IDLE_SECONDS),
                    injected_input_queue=injected_input_queue,
                    input_event=input_event,
                    observer=observer,
                    workspace_ctx=workspace_ctx,
                )
                system_instructions = render_system_instructions(
                    raw_instructions, rag_context=rag_context,
                    keep_agent_memory_placeholder=agent_memory_enabled,
                )
                logger.debug("[Agent streaming] raw_instructions contains {{current_datetime}}: %s", "{{current_datetime}}" in raw_instructions or "{{current\\_datetime}}" in raw_instructions)
                logger.debug("[Agent streaming] system_instructions contains {{current_datetime}}: %s", "{{current_datetime}}" in system_instructions or "{{current\\_datetime}}" in system_instructions)
                if rag_context and "{{context}}" not in raw_instructions:
                    system_instructions = system_instructions.rstrip() + "\n\nRelevant context:\n" + rag_context
                if emotion_hint:
                    system_instructions = system_instructions.rstrip() + "\n\n" + emotion_hint
                if retrieval_info is not None:
                    retrieval_info["agent_mode"] = True
                agent_agen = runner.run_streaming(
                    llm, user_input, history_list, system_instructions,
                    rag_context=rag_context, conversation_id=conversation_id,
                    multimodal_input=multimodal_input,
                )
                try:
                    async for chunk in agent_agen:
                        todo_list = chunk.get("todo_list")
                        if todo_list is not None and retrieval_info is not None:
                            retrieval_info["todo_list"] = todo_list
                        yield chunk
                finally:
                    # Close the runner generator deterministically: its finally
                    # (subagent shutdown, observer aclose) then runs in-band
                    # rather than via asyncgen GC finalization.
                    await agent_agen.aclose()
                if retrieval_info is not None:
                    todo_list = runner.todo_list_snapshot()
                    if todo_list is not None:
                        retrieval_info["todo_list"] = todo_list
                    trace = getattr(runner, "_last_trace", None)
                    if trace:
                        retrieval_info["_agent_trace"] = trace.to_dict()
                    if observer is not None and observer.usage:
                        retrieval_info["_observer_usage"] = dict(observer.usage)
                    usage = dict(runner._aggregate_iteration_usages(runner._iteration_usages) or {})
                    usage["tool_result_tokens"] = getattr(runner, "_tool_result_tokens", 0)
                    usage["rag_tool_tokens"] = getattr(runner, "_rag_tool_tokens", 0)
                    nested_usage = (
                        subagent_state.manager.usage_totals()
                        if subagent_state and subagent_state.is_root
                        else {}
                    )
                    for key, value in nested_usage.items():
                        usage[key] = int(usage.get(key) or 0) + value
                    subagent_tokens = int(nested_usage.get("prompt_tokens") or 0) + int(
                        nested_usage.get("completion_tokens") or 0
                    )
                    if subagent_tokens:
                        usage["subagent_tokens"] = subagent_tokens
                    if usage:
                        retrieval_info["_llm_usage_metadata"] = usage
                    final_answer_usage = getattr(runner, "_final_answer_usage", None)
                    if final_answer_usage is not None:
                        retrieval_info["_final_answer_usage"] = final_answer_usage
                return
            except Exception as exc:
                if retrieval_info is not None and runner is not None:
                    todo_list = runner.todo_list_snapshot()
                    if todo_list is not None:
                        retrieval_info["todo_list"] = todo_list
                logger.warning("Agent streaming failed for %s, falling back to normal chain: %s", model_name, exc, exc_info=True)

        if streaming_chain is None:
            if retrieval_info is not None:
                retrieval_info.update({"retrieval_time_ms": 0.0, "chunks_retrieved": 0, "status": "skipped"})
            fallback_result = await self._invoke_llm_only(llm_only_chain, payload)
            fallback = extract_answer_text(fallback_result)
            if fallback:
                yield {"content": fallback, "reasoning": ""}
            return

        accumulated_usage: Optional[Dict[str, int]] = None
        accumulated_reasoning_text: list[str] = []
        last_peeked_reasoning = ""
        full_reasoning_text = ""
        try:
            async for chunk in streaming_chain.astream(payload):
                chunk_usage = extract_usage_metadata(chunk)
                if chunk_usage is not None:
                    if accumulated_usage is None:
                        accumulated_usage = {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "reasoning_tokens": 0,
                        }
                    # Usage events are reported per LLM call (e.g. an extra
                    # contextualisation call before the main answer). Sum every
                    # field so total_tokens really covers all calls.
                    accumulated_usage["prompt_tokens"] += chunk_usage.get("prompt_tokens") or 0
                    accumulated_usage["completion_tokens"] += chunk_usage.get("completion_tokens") or 0
                    accumulated_usage["total_tokens"] += chunk_usage.get("total_tokens") or 0
                    accumulated_usage["reasoning_tokens"] += chunk_usage.get("reasoning_tokens") or 0
                reasoning_delta = extract_reasoning_text(chunk)
                if reasoning_delta:
                    accumulated_reasoning_text.append(reasoning_delta)
                    logger.debug("[DEBUG app] reasoning_delta=%r len=%d", reasoning_delta, len(reasoning_delta))
                delta = extract_answer_text(chunk)
                # Peek accumulated reasoning from chain builder to stream it live
                reasoning_to_send = reasoning_delta or ""
                if self._chain_builder is not None:
                    current_reasoning = await self._chain_builder.peek_reasoning_text(model_name)
                    if current_reasoning and len(current_reasoning) > len(last_peeked_reasoning):
                        reasoning_to_send = current_reasoning[len(last_peeked_reasoning):]
                        last_peeked_reasoning = current_reasoning
                        full_reasoning_text = current_reasoning
                if not delta and not reasoning_to_send:
                    continue
                if delta:
                    streamed = True
                yield {"content": delta or "", "reasoning": reasoning_to_send}
        except Exception as exc:
            if RecoveryManager.should_use_llm_only(exc):
                if self._recovery_manager is not None:
                    await self._recovery_manager.mark_unhealthy()
                logger.warning("Streaming retrieval failed; falling back to LLM-only mode: %s", exc)
                fallback = await self._invoke_llm_only(llm_only_chain, payload)
                fallback_text = extract_answer_text(fallback)
                if fallback_text:
                    yield {"content": fallback_text, "reasoning": ""}
                if self._recovery_manager is not None:
                    self._recovery_manager.schedule_recovery()
                return
            if RecoveryManager.is_upstream_llm_error(exc):
                raise UpstreamLLMError("Upstream LLM request failed") from exc
            raise

        if not streamed:
            fallback_result = await self._invoke_with_fallback(
                conversational_chain, llm_only_chain, payload
            )
            fallback = extract_answer_text(fallback_result)
            if fallback:
                yield {"content": fallback, "reasoning": ""}
        if accumulated_usage is not None and retrieval_info is not None:
            retrieval_info["_llm_usage_metadata"] = accumulated_usage
        # Try to get reasoning text from LLM accumulator (chains transform chunks to strings,
        # so extract_reasoning_text on chain output won't find it)
        reasoning_text = full_reasoning_text
        if not reasoning_text and accumulated_reasoning_text:
            reasoning_text = "".join(accumulated_reasoning_text)
        if not reasoning_text and self._chain_builder is not None:
            llm_reasoning = await self._chain_builder.get_reasoning_text(model_name)
            if llm_reasoning:
                reasoning_text = llm_reasoning
        if reasoning_text and retrieval_info is not None:
            retrieval_info["_llm_reasoning_text"] = reasoning_text
            logger.debug("[DEBUG app] accumulated_reasoning_text len=%d", len(reasoning_text))

    def _build_observer(
        self,
        assistant_entry: Dict[str, Any],
        model_name: str,
        subagent_state: Optional[SubagentState],
    ) -> Optional[Any]:
        """Create a per-run observer when the assistant enabled one.

        The observer LLM resolves via ``observer.assistant`` when configured,
        otherwise via the ``observer`` purpose (``observer`` assistant, falling
        back to the default assistant). Failures degrade to ``None`` — an
        observer must never break the run it watches.
        """
        observer_cfg = assistant_entry.get("observer") or {}
        if not (assistant_entry.get("agent_enabled") and observer_cfg.get("enabled")):
            return None
        if (
            subagent_state is not None
            and not subagent_state.is_root
            and not observer_cfg.get("include_subagents", True)
        ):
            return None
        try:
            from agent.observer import ObserverConfig, RunObserver

            obs_assistant = (observer_cfg.get("assistant") or "").strip() or None
            if obs_assistant and obs_assistant not in self.instructions:
                logger.warning(
                    "observer.assistant '%s' not found in instructions; using purpose fallback",
                    obs_assistant,
                )
                obs_assistant = None
            if obs_assistant:
                llm_factory = lambda a=obs_assistant: self.get_metadata_llm(model_name=a)
            else:
                llm_factory = lambda: self.get_metadata_llm(purpose="observer")
            return RunObserver(
                ObserverConfig.from_dict(observer_cfg),
                llm_factory,
                assistant_name=model_name,
            )
        except Exception:
            logger.warning("Failed to build run observer for %s", model_name, exc_info=True)
            return None

    async def get_metadata_llm(self, model_name: str | None = None, purpose: str | None = None) -> Any:
        """Return a non-streaming LLM for metadata helpers (title/tags/summary)."""
        if self._chain_builder is None:
            raise RuntimeError("RAGApplication is not initialized")
        return await self._chain_builder.get_metadata_llm(model_name=model_name, purpose=purpose)

    async def _get_current_instructions_version_id(self) -> int:
        if self._instructions_store is None:
            return 0
        return await self._instructions_store.get_current_version_id()

    async def refresh_provider_model_cache(self) -> None:
        """Rebuild the ``model -> provider_id`` membership map.

        The map deliberately stores *membership only* — endpoint details
        (URL, key, headers, timeouts) are resolved fresh per request, so
        admin edits take effect without waiting for this refresh. A refresh
        is only needed when model lists change (provider added/deleted,
        enabled toggled, upstream catalog changed).
        """
        if self._provider_store is None:
            return
        try:
            providers = await self._provider_store.get_enabled_providers()
        except Exception:
            logger.warning("Failed to list providers for model cache refresh", exc_info=True)
            return
        from app.llm_providers import build_provider_model_map

        previous = self._provider_model_cache
        models_by_provider: Dict[str, List[str]] = {}
        for provider in providers:
            try:
                models = await self._provider_store.fetch_models(provider.id)
            except Exception:
                logger.warning("Failed to fetch models from provider %s", provider.id, exc_info=True)
                models = None
            if models is None:
                # A transient fetch failure must not silently unroute every
                # model of that provider — preserve the previous membership.
                models = [m for m, pid in previous.items() if pid == provider.id]
            models_by_provider[provider.id] = models
        cache = build_provider_model_map(providers, models_by_provider)
        async with self._provider_cache_lock:
            self._provider_model_cache = cache
        if self._chain_builder is not None:
            self._chain_builder.set_provider_model_cache(cache)
        logger.info("Provider model cache refreshed with %d model mappings.", len(cache))

    def schedule_provider_model_refresh(self) -> None:
        """Schedule a coalesced background membership refresh.

        Called fire-and-forget by admin mutations — provider edits already
        propagate instantly via fresh endpoint resolution; this rebuild only
        keeps the ``model -> provider_id`` membership map current without
        blocking the admin request on upstream ``/models`` fetches.
        """
        if self._provider_store is None:
            return
        self._provider_refresh_requested = True
        task = getattr(self, "_provider_refresh_task", None)
        if task is None or task.done():
            try:
                self._provider_refresh_task = asyncio.create_task(
                    self._provider_refresh_worker()
                )
            except RuntimeError:
                self._provider_refresh_task = None
                logger.debug("No running event loop; provider model refresh deferred")

    async def _provider_refresh_worker(self) -> None:
        """Drain pending refresh requests; mutations during a run re-arm once."""
        while self._provider_refresh_requested:
            self._provider_refresh_requested = False
            try:
                await self.refresh_provider_model_cache()
            except Exception:
                logger.warning("Scheduled provider model refresh failed", exc_info=True)

    async def refresh_instructions_if_needed(self) -> bool:
        """Reload assistant instructions when the DB version or backing file changes."""
        # DB-first check
        if self._instructions_store is not None:
            current_db_id = await self._get_current_instructions_version_id()
            if current_db_id and current_db_id != self._instructions_db_version_id:
                async with self._instructions_lock:
                    current_db_id = await self._get_current_instructions_version_id()
                    if current_db_id == self._instructions_db_version_id:
                        return False
                    db_yaml = await self._instructions_store.get_current()
                    if db_yaml is not None:
                        try:
                            updated = load_instructions_from_yaml_text(db_yaml)
                        except Exception as exc:
                            logger.error("Failed to reload instructions from DB: %s", exc)
                            return False
                        self._apply_instructions(updated)
                        self._instructions_db_version_id = current_db_id
                logger.info("Assistant instructions reloaded from database (version %s).", current_db_id)
                return True

        # File fallback (legacy)
        current_mtime = self._get_instructions_mtime()
        if current_mtime and self._instructions_mtime and current_mtime <= self._instructions_mtime:
            return False

        async with self._instructions_lock:
            current_mtime = self._get_instructions_mtime()
            if current_mtime and self._instructions_mtime and current_mtime <= self._instructions_mtime:
                return False
            try:
                updated = self._load_instructions(self.config.instructions_file)
            except ConfigurationError as exc:
                logger.error("Failed to reload instructions: %s", exc)
                return False
            self._apply_instructions(updated)
            self._instructions_mtime = current_mtime
        logger.info("Assistant instructions reloaded from %s", self.config.instructions_file)
        return True

    async def force_refresh_instructions(self) -> bool:
        """Always reload assistant instructions from DB (or file fallback)."""
        if self._instructions_store is not None:
            async with self._instructions_lock:
                db_yaml = await self._instructions_store.get_current()
                if db_yaml is not None:
                    try:
                        updated = load_instructions_from_yaml_text(db_yaml)
                    except Exception as exc:
                        logger.error("Failed to reload instructions from DB: %s", exc)
                        return False
                    self._apply_instructions(updated)
                    self._instructions_db_version_id = await self._get_current_instructions_version_id()
            logger.info("Assistant instructions force-reloaded from database.")
            # Trigger background tool-calling detection for any new/updated providers
            try:
                await self._trigger_tool_calling_detection_for_assistants()
            except Exception:
                logger.debug("Tool-calling detection trigger failed after force refresh", exc_info=True)
            return True
        async with self._instructions_lock:
            try:
                updated = self._load_instructions(self.config.instructions_file)
            except ConfigurationError as exc:
                logger.error("Failed to reload instructions: %s", exc)
                return False
            self._apply_instructions(updated)
            self._instructions_mtime = self._get_instructions_mtime()
        logger.info("Assistant instructions force-reloaded from %s", self.config.instructions_file)
        # Trigger background tool-calling detection for any new/updated providers
        try:
            await self._trigger_tool_calling_detection_for_assistants()
        except Exception:
            logger.debug("Tool-calling detection trigger failed after force refresh", exc_info=True)
        return True

    async def _trigger_tool_calling_detection_for_assistants(self) -> None:
        """Fire-and-forget tool-calling detection for all providers referenced by assistants."""
        if self._provider_store is None:
            return
        seen_providers: Dict[str, str] = {}  # provider_id -> model_name
        for assistant_name in self.instructions:
            provider, model_name = await self._resolve_serving_provider(assistant_name)
            if provider is not None and model_name and provider.id not in seen_providers:
                seen_providers[provider.id] = model_name
        for provider_id, model_name in seen_providers.items():
            try:
                from app.tool_calling_detector import ensure_detected_for_provider
                asyncio.create_task(
                    ensure_detected_for_provider(self._provider_store, provider_id, model_name)
                )
            except Exception:
                logger.debug("Failed to start detection for provider %s", provider_id, exc_info=True)

    async def _enabled_provider(self, provider_id: Optional[str]) -> Optional[Any]:
        """Fetch a provider that may receive traffic — None when missing/disabled."""
        if not provider_id or self._provider_store is None:
            return None
        try:
            provider = await self._provider_store.get_provider(provider_id)
        except Exception:
            logger.exception("Failed to fetch provider %s", provider_id)
            return None
        return provider if provider is not None and provider.enabled else None

    async def _resolve_serving_provider(
        self, model_name: str
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Resolve the provider that will actually serve *model_name*.

        Mirrors ``ChainBuilder._resolve_provider_endpoint``: explicit pin →
        refreshed membership map → enabled default provider, skipping
        disabled/deleted providers at every stage. ``model_name`` is the
        assistant name; the returned model is the upstream (wire) model id —
        capability hints and detection must describe the same target the
        request will actually hit.
        """
        if self._provider_store is None:
            return None, None
        entry = self.instructions.get(model_name, {})
        explicit_id, resolved_model = self._resolve_provider_for_assistant_entry(entry)
        provider = await self._enabled_provider(explicit_id)
        if provider is not None:
            return provider, resolved_model
        wire_model = resolved_model or model_name
        cached_id = self._provider_model_cache.get(wire_model)
        if cached_id and cached_id != explicit_id:
            provider = await self._enabled_provider(cached_id)
            if provider is not None:
                return provider, wire_model
        try:
            default_provider = await self._provider_store.get_default_provider()
        except Exception:
            logger.exception("Failed to fetch default provider")
            default_provider = None
        if default_provider is not None and default_provider.enabled:
            return default_provider, wire_model
        return None, None

    def _resolve_provider_for_assistant_entry(
        self, entry: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Return (provider_id, model_name) for an assistant entry dict."""
        model_name = entry.get("model")
        provider_id = entry.get("provider_id")
        if provider_id:
            return provider_id, model_name
        if model_name and self._provider_model_cache:
            cached_provider_id = self._provider_model_cache.get(model_name)
            if cached_provider_id:
                return cached_provider_id, model_name
        return None, model_name

    async def _get_provider_tool_calling_hint(self, model_name: str) -> Optional[bool]:
        """Return cached tool-calling support for the provider+model backing *model_name*."""
        provider, resolved = await self._resolve_serving_provider(model_name)
        if provider is None:
            logger.debug("No provider resolved for %s, tool calling hint = None", model_name)
            return None
        provider_id = provider.id
        # Per-model lookup first, then legacy provider-wide fallback
        hint = await self._provider_store.get_model_tool_calling_status(provider_id, resolved)
        if hint is None:
            hint = provider.supports_native_tool_calls
        logger.debug(
            "Tool calling hint for %s (provider=%s, model=%s): supports_native=%s",
            model_name, provider_id, resolved, hint,
        )
        return hint

    def _get_instructions_mtime(self) -> float:
        try:
            return self.config.instructions_file.stat().st_mtime
        except FileNotFoundError:
            return 0.0

    def _apply_instructions(self, instructions: Dict[str, Any], *, clear_cache: bool = True) -> None:
        self.instructions = instructions
        if self._chain_builder is not None:
            self._chain_builder.instructions = self.instructions
            if clear_cache:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._chain_builder.clear_cache())
                except RuntimeError:
                    pass
        if self._retriever_builder is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._retriever_builder.clear_cache())
            except RuntimeError:
                pass
        self.default_assistant = self._determine_default_assistant()
        self._title_metadata_assistant = (
            "title_generator"
            if "title_generator" in self.instructions
            else self.default_assistant
        )
        self._tag_metadata_assistant = (
            "tag_extractor"
            if "tag_extractor" in self.instructions
            else self.default_assistant
        )
        self._refresh_assistant_tools()

    def _refresh_assistant_tools(self) -> None:
        """Register (or re-register) assistant-derived tools from current instructions."""
        self._tool_registry.clear_assistants()
        for assistant_name, entry in (self.instructions or {}).items():
            description = entry.get("description", "").strip()
            tool = AssistantCallerTool(
                rag_app=self,
                assistant_name=assistant_name,
                description=description,
            )
            self._tool_registry.register_assistant(tool.name, tool)

    def _get_assistant_collections(self, model_name: str) -> List[str]:
        entry = self.instructions.get(model_name, {})
        return list(entry.get("collections") or [])

    def _determine_default_assistant(self) -> str:
        if "CyberSecurityAssistant" in self.instructions:
            return "CyberSecurityAssistant"
        keys = list(self.instructions.keys())
        if keys:
            return keys[0]
        return "default"

    @staticmethod
    def _load_instructions(path: Path):
        """Load assistant instructions from disk while validating existence."""
        if not path.exists():
            raise ConfigurationError(f"Instructions file not found: {path}")
        return load_instructions(str(path))

    async def _invoke_with_fallback(
        self,
        conversational_chain,
        llm_only_chain,
        payload: dict,
    ) -> dict:
        """Run the conversational chain and fall back to the LLM-only variant on errors."""
        try:
            return await conversational_chain.ainvoke(payload)
        except Exception as exc:
            if RecoveryManager.should_use_llm_only(exc):
                if self._recovery_manager is not None:
                    await self._recovery_manager.mark_unhealthy()
                logger.warning("Retrieval failed; using LLM-only fallback: %s", exc)
                result = await self._invoke_llm_only(llm_only_chain, payload)
                if self._recovery_manager is not None:
                    self._recovery_manager.schedule_recovery()
                return result
            if RecoveryManager.is_upstream_llm_error(exc):
                raise UpstreamLLMError("Upstream LLM request failed") from exc
            raise

    async def _invoke_llm_only(self, llm_only_chain, payload: dict) -> dict:
        """Invoke the non-retrieval chain and normalize upstream failures.

        Ensures the result contains both ``answer`` and ``reasoning`` keys.
        """
        try:
            result = await llm_only_chain.ainvoke(payload)
            if isinstance(result, dict):
                payload_out = dict(result)
                text = extract_answer_text(payload_out.get("answer") or payload_out)
                if text:
                    payload_out["answer"] = text
                reasoning = extract_reasoning_text(payload_out)
                if reasoning:
                    payload_out["reasoning"] = reasoning
                return payload_out
            text = extract_answer_text(result)
            reasoning = extract_reasoning_text(result)
            out: dict[str, Any] = {"answer": text}
            if reasoning:
                out["reasoning"] = reasoning
            return out
        except Exception as exc:
            if RecoveryManager.is_upstream_llm_error(exc):
                raise UpstreamLLMError("Upstream LLM request failed") from exc
            raise

    async def run_chain(
        self,
        chain_definition: Dict[str, Any],
        input_text: str,
        chat_history: Sequence[BaseMessage] | None = None,
        conversation_id: Optional[str] = None,
        chat_user_id: Optional[str] = None,
        execution_options: Optional[Dict[str, Any]] = None,
        run_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute an agent chain flow definition and return the result."""
        from app.chain_runner import ChainRunner
        runner = ChainRunner(
            chain_definition,
            self,
            execution_options=execution_options,
            run_context=run_context,
        )
        return await runner.run(
            input_text=input_text,
            chat_history=list(chat_history) if chat_history else None,
            conversation_id=conversation_id,
            chat_user_id=chat_user_id,
        )

    async def run_chain_streaming(
        self,
        chain_definition: Dict[str, Any],
        input_text: str,
        chat_history: Sequence[BaseMessage] | None = None,
        conversation_id: Optional[str] = None,
        chat_user_id: Optional[str] = None,
        execution_options: Optional[Dict[str, Any]] = None,
        run_context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, str], None]:
        """Execute an agent chain flow definition with streaming output."""
        from app.chain_runner import ChainRunner
        runner = ChainRunner(
            chain_definition,
            self,
            execution_options=execution_options,
            run_context=run_context,
        )
        chain_agen = runner.run_streaming(
            input_text=input_text,
            chat_history=list(chat_history) if chat_history else None,
            conversation_id=conversation_id,
            chat_user_id=chat_user_id,
        )
        try:
            async for chunk in chain_agen:
                yield chunk
        finally:
            await chain_agen.aclose()
