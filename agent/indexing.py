"""Document discovery, loading, chunking, and metadata normalization."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain.schema import Document

from agent.config import AppConfig
from agent.embeddings import CustomOpenAICompatibleEmbeddings
from agent.loader import KnowledgeLoader
from agent.processing import chunk_documents
from agent.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)
REMOTE_COLLECTION_NAME = "external_urls"


class IndexManager:
    """Coordinates document ingestion, embedding, and incremental vector store updates."""

    def __init__(
        self,
        config: AppConfig,
        loader: Optional[KnowledgeLoader] = None,
        embeddings_factory: Optional[
            Callable[[AppConfig], CustomOpenAICompatibleEmbeddings]
        ] = None,
        engine: Any = None,
        llm_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.config = config
        self.loader = loader or self._build_default_loader(config, llm_factory=llm_factory)
        self.embeddings_factory = embeddings_factory or self._default_embeddings_factory
        self.engine = engine

    def discover_collections(self) -> Dict[str, Optional[Path]]:
        """Return a mapping of top-level knowledge folders to their collection names."""
        collections: Dict[str, Optional[Path]] = {}
        root = self.config.knowledge_base_root
        if not root.exists():
            logger.warning("Knowledge base root '%s' does not exist.", root)
            return collections
        for child in sorted(root.iterdir()):
            if child.is_dir():
                collections[child.name] = child
        return collections

    async def load_documents_for_collection(
        self,
        name: str,
        folder: Optional[Path],
        filepaths: Optional[List[str]] = None,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        file_timeout: Optional[float] = 900.0,
    ) -> List[Document]:
        """Load documents for a single collection (local folder or remote URLs).

        When ``filepaths`` is provided for a local collection, only those files are
        loaded instead of the entire folder. ``progress_cb``/``file_timeout`` are
        forwarded to the loader.
        """
        if name == REMOTE_COLLECTION_NAME:
            docs = await self.loader.load_web_documents()
            for doc in docs:
                doc.metadata = dict(doc.metadata or {})
                doc.metadata["source"] = doc.metadata.get("source") or doc.metadata.get("url") or "remote"
            return docs

        if folder is None:
            logger.warning("Collection '%s' has no backing folder; skipping.", name)
            return []

        return await self.loader.load_local_documents(
            folder=folder, filepaths=filepaths,
            progress_cb=progress_cb, file_timeout=file_timeout,
        )

    def prepare_splits(self, collection_name: str, source: str, docs: List[Document]) -> List[Document]:
        """Chunk documents and enrich metadata with collection, source, and stable IDs."""
        if not docs:
            return []
        doc_splits = chunk_documents(
            docs,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        for idx, chunk in enumerate(doc_splits):
            chunk.metadata = dict(chunk.metadata or {})
            chunk.metadata["collection"] = collection_name
            chunk.metadata["source"] = source
            chunk.metadata["chunk_index"] = idx
            content_hash = hashlib.sha256(
                (chunk.page_content or "").encode("utf-8", errors="ignore")
            ).hexdigest()
            chunk.metadata["content_hash"] = content_hash
            chunk.metadata["doc_id"] = hashlib.sha256(
                f"{collection_name}|{source}|{idx}|{content_hash}".encode("utf-8", errors="ignore")
            ).hexdigest()
        return doc_splits

    def normalize_document_metadata(self, collection_name: str, doc: Document) -> Document:
        """Normalize source paths relative to the knowledge base root."""
        metadata = dict(doc.metadata or {})
        raw_source = metadata.get("source") or metadata.get("file") or metadata.get("path")
        normalized_source = self._normalize_source(raw_source)
        if raw_source and str(raw_source) != normalized_source:
            metadata.setdefault("source_path", str(raw_source))
        metadata["source"] = normalized_source
        metadata["collection"] = collection_name
        doc.metadata = metadata
        return doc

    def _normalize_source(self, raw_source: object | None) -> str:
        if raw_source is None:
            return "unknown"
        source = str(raw_source).strip()
        if not source:
            return "unknown"
        try:
            path = Path(source)
            if path.is_absolute():
                kb_root = self.config.knowledge_base_root
                with contextlib.suppress(Exception):
                    if kb_root in path.parents:
                        return path.relative_to(kb_root).as_posix()
        except Exception:
            pass
        return source

    @staticmethod
    def _hash_single_file(
        filepath: Path, normalize_source: Callable[[str], str]
    ) -> Optional[tuple[str, str]]:
        """Compute (normalized_source, sha256_hex) for a single file. Returns None on error."""
        ext = filepath.suffix.lower()
        supported_exts = {
            ".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg",
            ".doc", ".odt",
            ".xls", ".xlsx",
            ".ppt", ".pptx",
            ".csv",
            ".gif", ".bmp", ".tiff", ".webp",
        }
        if ext not in supported_exts:
            return None
        source = normalize_source(str(filepath))
        try:
            with open(filepath, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            return source, file_hash
        except Exception as exc:
            logger.warning("Unable to compute fingerprint for '%s': %s", filepath, exc)
            return None

    def compute_local_file_fingerprints(self, folder: Path) -> Dict[str, str]:
        """Return a mapping of normalized source -> SHA256(raw file content)."""
        result: Dict[str, str] = {}
        if not folder.exists():
            return result
        for filepath in folder.rglob("*"):
            if not filepath.is_file():
                continue
            pair = self._hash_single_file(filepath, self._normalize_source)
            if pair is not None:
                result[pair[0]] = pair[1]
        return result

    async def compute_local_file_fingerprints_async(self, folder: Path) -> Dict[str, str]:
        """Parallel version of :meth:`compute_local_file_fingerprints` using threads."""
        result: Dict[str, str] = {}
        if not folder.exists():
            return result
        filepaths = [p for p in folder.rglob("*") if p.is_file()]
        if not filepaths:
            return result
        tasks = [
            asyncio.to_thread(self._hash_single_file, fp, self._normalize_source)
            for fp in filepaths
        ]
        for pair in await asyncio.gather(*tasks):
            if pair is not None:
                result[pair[0]] = pair[1]
        return result

    def build_vector_manager(self, collection_name: str) -> VectorStoreManager:
        """Create a :class:`VectorStoreManager` wired to the shared engine."""
        return VectorStoreManager(
            connection_string=self.config.connection_string,
            collection_name=collection_name,
            search_k=self.config.vector_search_k,
            batch_size=self.config.vector_batch_size,
            engine=self.engine,
        )

    def _build_default_loader(self, config: AppConfig, llm_factory: Optional[Callable[[], Any]] = None) -> KnowledgeLoader:
        """Create the default :class:`KnowledgeLoader` using filesystem and URL settings."""
        return KnowledgeLoader(
            folder_path=Path(config.data_dir),
            urls=[str(url) for url in config.knowledge_urls],
            ocr_lang=config.ocr_language,
            tesseract_cmd=config.tesseract_cmd,
            user_agent=config.user_agent,
            max_concurrency=config.loader_max_concurrency,
            ocr_mode=getattr(config, "ocr_mode", "tesseract"),
            ocr_llm_prompt=getattr(config, "ocr_llm_prompt", "Extract all text and relevant information from this image."),
            ocr_llm_concurrency=getattr(config, "ocr_llm_concurrency", 4),
            llm_factory=llm_factory,
            pdf_image_extraction_enabled=getattr(config, "pdf_image_extraction_enabled", False),
            pdf_image_extraction_mode=getattr(config, "pdf_image_extraction_mode", "llm"),
            pptx_slide_analysis_enabled=getattr(config, "pptx_slide_analysis_enabled", False),
            pptx_slide_analysis_mode=getattr(config, "pptx_slide_analysis_mode", "llm"),
        )

    def _default_embeddings_factory(self, config: AppConfig) -> CustomOpenAICompatibleEmbeddings:
        """Instantiate the embeddings client configured for the OpenAI-compatible endpoint."""
        endpoint = config.embeddings_endpoint or f"{config.embeddings_api_base}/v1/embeddings"
        return CustomOpenAICompatibleEmbeddings(
            endpoint=endpoint,
            model=config.embedding_model,
            api_key=(config.embeddings_api_key or "none"),
            timeout=int(config.embeddings_request_timeout),
            max_retries=int(config.embeddings_max_retries),
            backoff_factor=float(config.embeddings_backoff_factor),
            max_chars=int(getattr(config, "embedding_max_chars", 0)),
        )
